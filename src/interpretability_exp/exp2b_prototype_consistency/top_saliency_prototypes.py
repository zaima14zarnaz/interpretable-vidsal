#!/usr/bin/env python3
"""Prototype IDs in the top-20 lists of the most salient patches, sorted by frequency.

Reads the same patch_data.json stream as overlap_analysis.py. Selects the
highest ceil(percent) predicted-saliency patches. Equal scores at the cutoff
keep earlier records until that count is filled. Each selected patch contributes
its exported top-20 prototype IDs, in activation-rank order, with no cosine
reranking and no padding of short lists.

The reported list is those prototype IDs sorted by how many selected patches
contain them, then by prototype ID. Mean pairwise Dice uses the same exact
frequency-count definition as overlap_analysis.py:
2*|A intersection B|/(|A|+|B|), averaged over selected pairs whose lists are
both nonempty. For each of those IDs, the table also reports the percentage of
the lowest BOTTOM_PERCENT% predicted-saliency patches whose top-20 list contains it.
Cutoff ties in that lower half keep earlier records. The input file is not modified.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from heapq import nlargest, nsmallest
import json
import math
from pathlib import Path
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap_analysis as overlap


SALIENCY_KEY = 'patch_pred_saliency'
TOP_K = 20
BOTTOM_PERCENT = 60.0
OUTPUT_NAMES = ('top_saliency_prototype_frequency.json',
                'top_saliency_prototype_frequency.csv',
                'top_saliency_prototype_frequency.txt')


def selection_cutoff(scores, percent, highest):
    """Return selected count, cutoff score, and how many cutoff ties to keep."""
    count = len(scores)
    if not count:
        raise ValueError('patch_data.json contains no patch records.')
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError('percent must be in (0,100].')
    selected = min(count, max(1, math.ceil(count * percent / 100)))
    if highest:
        threshold = nlargest(selected, scores)[-1]
        outside = sum(score > threshold for score in scores)
    else:
        threshold = nsmallest(selected, scores)[-1]
        outside = sum(score < threshold for score in scores)
    return selected, threshold, selected - outside


def analyze(path, top_percent=3.0):
    print('Measuring predicted saliency...', flush=True)
    scores = []
    for index, record in enumerate(overlap.iter_patch_records(path)):
        try:
            scores.append(overlap.finite_score(record, SALIENCY_KEY))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Patch record {index}: {exc}') from exc
    selected_count, threshold, ties_needed = selection_cutoff(scores, top_percent, True)
    bottom_count, bottom_threshold, bottom_ties_needed = selection_cutoff(scores, BOTTOM_PERCENT, False)
    dataset_count = len(scores)
    del scores
    print(f'Selecting the top {selected_count:,} and bottom {bottom_count:,} of {dataset_count:,} patches...',
          flush=True)
    accumulator = overlap.BinAccumulator(TOP_K)
    bottom_frequency = Counter()
    ties_left = ties_needed
    bottom_ties_left = bottom_ties_needed
    bottom_seen = 0
    minimum = maximum = None
    for index, record in enumerate(overlap.iter_patch_records(path)):
        try:
            score = overlap.finite_score(record, SALIENCY_KEY)
            take_top = score > threshold or (score == threshold and ties_left > 0)
            take_bottom = score < bottom_threshold or (score == bottom_threshold and bottom_ties_left > 0)
            ids = overlap.ranked_prototype_ids(record) if take_top or take_bottom else None
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Patch record {index}: {exc}') from exc
        if ids is None:
            continue
        if take_top:
            if score == threshold:
                ties_left -= 1
            minimum = score if minimum is None else min(minimum, score)
            maximum = score if maximum is None else max(maximum, score)
            accumulator.add(ids)
        if take_bottom:
            if score == bottom_threshold:
                bottom_ties_left -= 1
            bottom_frequency.update(ids)
            bottom_seen += 1
        if (accumulator.patch_count + bottom_seen) % 100000 == 0:
            print(f'Top patches {accumulator.patch_count:,}; bottom patches {bottom_seen:,}...', flush=True)
    if accumulator.patch_count != selected_count or ties_left != 0:
        raise RuntimeError('Selected patch count changed between the saliency scan and ID collection.')
    if bottom_seen != bottom_count or bottom_ties_left != 0:
        raise RuntimeError('Bottom-half patch count changed between the saliency scan and ID collection.')
    summary = accumulator.summarize(0)
    prototypes = []
    for rank, item in enumerate(summary['prototype_frequencies'], start=1):
        occurred = bottom_frequency[item['prototype_index']]
        prototypes.append({
            'rank': rank, 'prototype_index': item['prototype_index'],
            'patch_count': item['patch_count'], 'patch_percent': 100 * item['patch_fraction'],
            f'bottom_{BOTTOM_PERCENT}_patch_count': occurred,
            f'bottom_{BOTTOM_PERCENT}_patch_percent': 100 * occurred / bottom_count,
        })
    return {
        'patch_data_path': str(Path(path).resolve()),
        'saliency_field': SALIENCY_KEY,
        'top_percent': top_percent,
        'dataset_patch_count': dataset_count,
        'selected_patch_count': selected_count,
        'selection_rule': 'highest ceil(top_percent) predicted-saliency patches; cutoff ties keep earlier records',
        'saliency_cutoff': threshold,
        'selected_saliency_min': minimum,
        'selected_saliency_max': maximum,
        'bottom_percent': BOTTOM_PERCENT,
        'bottom_patch_count': bottom_count,
        'bottom_saliency_cutoff': bottom_threshold,
        'bottom_selection_rule': f'lowest ceil({BOTTOM_PERCENT}) percent) predicted-saliency patches; cutoff ties keep earlier records',
        'top_k': TOP_K,
        'ranking': 'exported forward-activation order; no cosine reranking',
        'mean_pairwise_dice_percent': summary['mean_overlap_percent'],
        'mean_shared_prototype_count': summary['mean_shared_prototype_count'],
        'eligible_pair_count': summary['eligible_pair_count'],
        'definitions': {
            'sorted_list': 'top-20 prototype IDs of the selected patches, sorted by descending patch count then prototype ID',
            'patch_count': 'number of selected patches whose top-20 list contains the prototype ID',
            f'bottom_{BOTTOM_PERCENT}_patch_percent': f'percentage of bottom-{BOTTOM_PERCENT}% patches whose top-20 list contains this top-list prototype ID',
            'mean_pairwise_dice_percent': '100 * mean(2*|A intersection B|/(|A|+|B|)) over selected pairs with both lists nonempty',
        },
        'prototypes': prototypes,
    }


def text_report(summary):
    dice = summary['mean_pairwise_dice_percent']
    dice_text = 'N/A' if dice is None else f'{dice:.4f}'
    lines = [
        f'Top {summary["top_percent"]:g}% salient patches by {summary["saliency_field"]}',
        f'Selected {summary["selected_patch_count"]:,} of {summary["dataset_patch_count"]:,} patches.',
        f'Cutoff {summary["saliency_cutoff"]:.6g}; selected range '
        f'{summary["selected_saliency_min"]:.6g} to {summary["selected_saliency_max"]:.6g}.',
        'Prototype IDs are the exported top-20 activations. '
        'Sorted by how many selected patches contain each ID.',
        f'Mean pairwise Dice overlap: {dice_text}% '
        f'over {summary["eligible_pair_count"]:,} pairs with nonempty lists.',
        f'Bottom {summary["bottom_percent"]:g}%: {summary["bottom_patch_count"]:,} patches '
        f'at or below predicted saliency {summary["bottom_saliency_cutoff"]:.6g}.',
        f'bottom {BOTTOM_PERCENT}% percent is how many of those patches include the prototype in their top-20 list.',
        '',
        f'rank  prototype ID   top patches   percent of top   bottom patches   percent of bottom {BOTTOM_PERCENT}%',
    ]
    for row in summary['prototypes']:
        lines.append(f'{row["rank"]:>4}  {row["prototype_index"]:>12}  '
                     f'{row["patch_count"]:>12,}  {row["patch_percent"]:>15.4f}  '
                     f'{row[f"bottom_{BOTTOM_PERCENT}_patch_count"]:>15,}  {row[f"bottom_{BOTTOM_PERCENT}_patch_percent"]:>22.4f}')
    if not summary['prototypes']:
        lines.append('(none)')
    return '\n'.join(lines) + '\n'


def save_reports(summary, output_dir, overwrite=False):
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((root / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError('Top-saliency reports exist. Use another --output-dir or --overwrite-results.')
    with tempfile.TemporaryDirectory(prefix='.top_saliency_', dir=root) as temporary:
        tmp = Path(temporary)
        (tmp / OUTPUT_NAMES[0]).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        with (tmp / OUTPUT_NAMES[1]).open('w', newline='', encoding='utf-8') as handle:
            fields = ('rank', 'prototype_index', 'patch_count', 'patch_percent',
                      f'bottom_{BOTTOM_PERCENT}_patch_count', f'bottom_{BOTTOM_PERCENT}_patch_percent')
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary['prototypes'])
        (tmp / OUTPUT_NAMES[2]).write_text(text_report(summary), encoding='utf-8')
        for name in OUTPUT_NAMES:
            (tmp / name).replace(root / name)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--patch-data', required=True, help='Path to patch_data.json.')
    parser.add_argument('--output-dir', default=None, help='Default: directory containing patch_data.json.')
    parser.add_argument('--top-percent', type=float, default=3.0,
                        help='Highest predicted-saliency percentage to keep (default 3).')
    parser.add_argument('--overwrite-results', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.top_percent) or not 0 < args.top_percent <= 100:
        parser.error('top-percent must be in (0,100].')
    source = Path(args.patch_data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else source.parent
    try:
        summary = analyze(source, args.top_percent)
        # root = save_reports(summary, output_dir, args.overwrite_results)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(text_report(summary), end='')
    print(f'Saved reports to {root}')


if __name__ == '__main__':
    main()
