#!/usr/bin/env python3
"""Summarize predicted saliency gaps (higher − lower patch) for valid rules in pair_results CSVs."""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

PAIR_RESULTS_PATTERN = re.compile(
    r'^pair_results_subset_(?P<subset>\d+)_seed_(?P<seed>\d+)\.csv$'
)

DEFAULT_OUTPUT_DIR = Path(
    '/data/quantization/zaima/videosal_datasets/dhf1k/contextual_prot_consistency'
)


@dataclass
class Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def merge(self, n: int, mean: float, m2: float) -> None:
        if n <= 0 or not math.isfinite(mean):
            return
        if self.n == 0:
            self.n = n
            self.mean = mean
            self.m2 = m2
            return
        total = self.n + n
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.n * n / total
        self.mean += delta * n / total
        self.n = total

    def std(self) -> float:
        if self.n < 2:
            return 0.0 if self.n == 1 else math.nan
        return math.sqrt(max(0.0, self.m2 / (self.n - 1)))


def parse_bool(value: str) -> bool:
    text = (value or '').strip().lower()
    return text in ('1', 'true', 'yes', 't')


def parse_float(value: str) -> float:
    text = (value or '').strip()
    if not text:
        return math.nan
    return float(text)


def parse_int(value: str) -> int:
    text = (value or '').strip()
    if not text:
        return 0
    return int(float(text))


def subset_moments(row: dict, subset: str, direction: str) -> tuple[int, float, float] | None:
    """Return (n, mean, m2) of higher−lower predicted gap for one discovery/confirmation block."""
    if direction not in ('A', 'B'):
        return None
    prefix = f'{subset}_pred_'
    n = parse_int(row.get(f'{prefix}rows', ''))
    mean = parse_float(row.get(f'{prefix}difference_mean', ''))
    std = parse_float(row.get(f'{prefix}difference_std', ''))
    if n <= 0 or not math.isfinite(mean):
        return None
    # Stored mean is patch-A minus patch-B; flip when B is the higher-saliency side.
    if direction == 'B':
        mean = -mean
    m2 = (std * std * (n - 1)) if n > 1 and math.isfinite(std) else 0.0
    return n, mean, m2


def pair_higher_lower_moments(row: dict) -> tuple[int, float, float] | None:
    direction = (row.get('direction') or '').strip()
    blocks = [subset_moments(row, name, direction) for name in ('discovery', 'confirmation')]
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return None
    total = Moments()
    for n, mean, m2 in blocks:
        total.merge(n, mean, m2)
    if total.n <= 0:
        return None
    return total.n, total.mean, total.m2


def analyze_pair_results(path: Path) -> dict:
    valid_rows = 0
    pooled = Moments()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not parse_bool(row.get('valid_rule', '')):
                continue
            valid_rows += 1
            stats = pair_higher_lower_moments(row)
            if stats is None:
                continue
            n, mean, m2 = stats
            pooled.merge(n, mean, m2)
    return {
        'valid_rules': valid_rows,
        'comparisons': pooled.n,
        'mean_higher_minus_lower': pooled.mean if pooled.n else math.nan,
        'std_higher_minus_lower': pooled.std(),
    }


def discover_pair_results(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        raise ValueError(f'Output directory not found: {output_dir}')
    paths = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and PAIR_RESULTS_PATTERN.match(path.name):
            paths.append(path)
    return paths


def format_stat(value: float) -> str:
    if not math.isfinite(value):
        return 'n/a'
    return f'{value:.8g}'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory containing pair_results_subset_*_seed_*.csv (default: {DEFAULT_OUTPUT_DIR})',
    )
    args = parser.parse_args(argv)
    paths = discover_pair_results(args.output_dir)
    if not paths:
        print(f'No pair_results_subset_*_seed_*.csv files in {args.output_dir}')
        return 1
    for path in paths:
        match = PAIR_RESULTS_PATTERN.match(path.name)
        label = f'subset={match.group("subset")}, seed={match.group("seed")}'
        stats = analyze_pair_results(path)
        print(f'\n{path.name} ({label})')
        print(f'  Valid rules: {stats["valid_rules"]:,}')
        print(f'  Predicted patch-pair comparisons (pooled): {stats["comparisons"]:,}')
        print(f'  Mean (higher − lower predicted saliency): {format_stat(stats["mean_higher_minus_lower"])}')
        print(f'  Std  (higher − lower predicted saliency): {format_stat(stats["std_higher_minus_lower"])}')
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
