#!/usr/bin/env python3
"""Plot how often each prototype appears across concept groups.

Reads ``concept_summary.csv`` (one row per group), parses the semicolon-separated
``prototypes`` column, and counts how many groups list each prototype ID.
Writes a frequency CSV and a bar chart (x = prototype id, y = group count).

Example:

    python concept_analyzer.py
    python concept_analyzer.py \\
        --summary /data/.../concepts_v5/concept_summary.csv \\
        --output-dir ./concept_analyzer_outputs
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_SUMMARY = (
    "/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/"
    "concepts_v5/concept_summary.csv"
)
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "concept_analyzer_outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--summary", type=Path, default=Path(DEFAULT_SUMMARY),
                        help="Path to concept_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                        help="Directory for the frequency CSV and chart PNG")
    parser.add_argument("--delimiter", default=";",
                        help="Delimiter inside the prototypes cell (default: ;)")
    return parser.parse_args()


def load_group_prototypes(summary_path: Path, delimiter: str = ";") -> dict[str, set[int]]:
    """Return {concept_id: set of prototype ids} from concept_summary.csv."""
    path = summary_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")

        # Accept either "prototypes" (current export) or "prototype".
        field_map = {name.lower(): name for name in reader.fieldnames}
        proto_key = field_map.get("prototypes") or field_map.get("prototype")
        group_key = (
            field_map.get("concept_id")
            or field_map.get("group")
            or field_map.get("group_id")
        )
        if proto_key is None:
            raise ValueError(
                f"CSV must contain a 'prototypes' or 'prototype' column. "
                f"Found: {reader.fieldnames}"
            )
        if group_key is None:
            raise ValueError(
                f"CSV must contain a 'concept_id' or 'group' column. "
                f"Found: {reader.fieldnames}"
            )

        groups: dict[str, set[int]] = {}
        for row_number, row in enumerate(reader, start=2):
            concept_id = (row.get(group_key) or "").strip()
            if not concept_id:
                raise ValueError(f"Missing {group_key} on line {row_number} of {path}")
            raw = (row.get(proto_key) or "").strip()
            ids: set[int] = set()
            if raw:
                for token in raw.split(delimiter):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        ids.add(int(token))
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid prototype id {token!r} on line {row_number} of {path}"
                        ) from exc
            if concept_id in groups:
                raise ValueError(f"Duplicate {group_key} {concept_id!r} in {path}")
            groups[concept_id] = ids
    return groups


def count_prototype_groups(groups: dict[str, set[int]]) -> tuple[Counter, dict[int, list[str]]]:
    """Count groups per prototype and keep the group ids that mention it."""
    counts: Counter = Counter()
    membership: dict[int, list[str]] = defaultdict(list)
    for concept_id, prototypes in groups.items():
        for pid in prototypes:
            counts[pid] += 1
            membership[pid].append(concept_id)
    for pid in membership:
        membership[pid].sort()
    return counts, dict(membership)


def write_frequency_csv(
    path: Path,
    counts: Counter,
    membership: dict[int, list[str]],
    group_count: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "prototype_index",
                "group_count",
                "group_frequency",
                "groups",
            ],
        )
        writer.writeheader()
        for pid in sorted(counts):
            n = counts[pid]
            writer.writerow(
                {
                    "prototype_index": pid,
                    "group_count": n,
                    "group_frequency": f"{n / group_count:.8f}" if group_count else "0",
                    "groups": ";".join(membership.get(pid, [])),
                }
            )


def write_chart(path: Path, counts: Counter, group_count: int, summary_path: Path) -> None:
    if not counts:
        raise ValueError("No prototypes found to plot.")

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [str(pid) for pid, _ in ordered]
    values = [n for _, n in ordered]

    fig_width = max(6.0, min(12.0, 0.18 * len(labels) + 1.2))
    fig, axis = plt.subplots(figsize=(fig_width, 4.5))
    axis.bar(range(len(labels)), values, width=0.4, color="#146d68", edgecolor="none")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=8)
    axis.set_xlabel("Prototype id")
    axis.set_ylabel("Number of groups")
    axis.set_title(
        f"Prototype frequency across concept groups\n"
        f"{group_count} groups · {len(counts)} distinct prototypes · {summary_path.name}"
    )
    axis.set_ylim(0, max(values) * 1.12)
    for index, value in enumerate(values):
        axis.text(index, value + 0.05, str(value), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_group_prototypes(summary_path, delimiter=args.delimiter)
    counts, membership = count_prototype_groups(groups)
    group_count = len(groups)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"frequency.csv"
    png_path = output_dir / f"frequency.png"

    write_frequency_csv(csv_path, counts, membership, group_count)
    write_chart(png_path, counts, group_count, summary_path)

    top = counts.most_common(10)
    print(f"Summary: {summary_path}")
    print(f"Groups: {group_count}")
    print(f"Distinct prototypes: {len(counts)}")
    print(f"Top prototypes by #groups: {top}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
