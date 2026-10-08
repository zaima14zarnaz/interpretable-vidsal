#!/usr/bin/env python3
"""Compare within-bin and cross-bin prototype overlap at top-5/10/20.

Reads patch_data.json produced by prototype_consistency_sal.py. Prototype lists
are used in their exported activation-rank order, never re-ranked by cosine.
Each bin compares every unordered pair of distinct patch records, including
pairs from the same video. Cross-bin pairs compare every patch in one bin with
every patch in a bin at least two bins away. For saliency bin i, numbered 1..n,
the cross set is bins 1..i-2 and i+2..n. Adjacent bins are excluded. ALL shared
prototype IDs contribute, even if they are
absent from other patches. Universal membership is descriptive only.
Rank affects top-k membership; overlap compares IDs.

Binary ID-overlap metrics are exact, using prototype occurrence counts rather than explicitly
enumerating O(N^2) pairs. Mean shared count includes all pairs. Mean overlap is
the pairwise Dice score 2*|A intersection B|/(|A|+|B|), averaged over pairs where
BOTH patches have nonempty lists. When both lists have k entries, this equals
shared/k. Short lists are not padded. Bins with fewer than two eligible patches
have null overlap. A separate shared/requested-k percentage includes all pairs.

Predicted scores are rebinned into global equal-width bins (--saliency-bins,
default 20), matching the supplied modified analysis script. The input is never
modified. --prototype-presence-percent controls the displayed frequency list
only; it does not filter pairs, prototypes contributing to overlap, or bins.

For each bin, compare within-bin Dice with the equally weighted mean cross-bin
Dice across eligible bins at least two away. Also report a pair-weighted cross
mean. Positive delta = within-bin overlap exceeds the cross-bin mean. Overall
comparison averages the per-bin within/cross values over the same eligible
bins. These are descriptive differences, not significance tests: pairs share
patches and videos and are not independent observations.

Outputs: prototype_overlap_summary.json/.csv/.txt,
prototype_overlap_cross_bin.csv, prototype_overlap_within_vs_cross.csv.
JSON includes a symmetric Dice matrix for every requested k, with within-bin
overlap on the diagonal. Adjacent off-diagonal cells are null because those
bins are excluded from cross. Other bin and bin-pair rows are retained, including zeros and nulls.
Python standard library only; no GPU needed.

Strength analysis uses the saved --strength-field (default forward_activation).
Pair strength Dice = 2*sum(min(abs(a_p),abs(b_p)) for same-sign shared IDs)
                    / (sum(abs(a_p)) + sum(abs(b_p))).
Negative scores occupy separate sign channels, never counting as positive matches.
No per-patch normalization is applied: magnitude differences remain visible.
Shared-strength agreement uses the same numerator but only shared-ID magnitudes
in the denominator, isolating strength agreement from identity overlap.

Binary ID metrics and per-prototype profiles use the full dataset exactly.
Strength pair means use bounded uniform per-bin reservoirs and uniformly sampled
pairs unless the entire eligible population fits and all pairs are evaluated.
Sampling provenance/counts are saved. No independence-based error bars are used.
Profiles report both presence frequency and strength conditional on selection;
correlations with predicted AND GT saliency are computed among selected patches.
Unselected/truncated prototypes are zero only in the explicit masked-mean field.
Observational associations do not establish causal effects on saliency.
Additional CSVs: prototype_activation_strength_overlap,
prototype_activation_strength_within_vs_cross, prototype_activation_by_bin,
prototype_activation_saliency_association, and prototype_overlap_dataset_presence.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import tempfile


TOP_K_VALUES = (5, 10, 20)
DEFAULT_TOP_K_VALUES = (5,)
OUTPUT_NAMES = ('prototype_overlap_summary.json', 'prototype_overlap_summary.csv',
                'prototype_overlap_summary.txt', 'prototype_overlap_cross_bin.csv',
                'prototype_overlap_within_vs_cross.csv',
                'prototype_overlap_dataset_presence.csv',
                'prototype_activation_strength_overlap.csv',
                'prototype_activation_strength_within_vs_cross.csv',
                'prototype_activation_by_bin.csv',
                'prototype_activation_saliency_association.csv')
STRENGTH_FIELDS = ('forward_activation', 'cosine_similarity', 'assignment_probability', 'scaled_logit')


def iter_patch_records(path, chunk_size=1 << 20):
    """Stream a JSON array of objects, including pretty-printed JSON arrays."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8-sig') as handle:
        buffer, pos, eof = '', 0, False

        def refill():
            nonlocal buffer, pos, eof
            buffer = buffer[pos:]
            pos = 0
            chunk = handle.read(chunk_size)
            buffer += chunk
            eof = not chunk

        def next_char():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer):
                    return buffer[pos]
                if eof:
                    return None
                refill()

        if next_char() != '[':
            raise ValueError('patch_data.json must contain a JSON array of patch objects.')
        pos += 1
        first = True
        while True:
            char = next_char()
            if char == ']':
                pos += 1
                break
            if not first:
                if char != ',':
                    raise ValueError('Expected a comma between patch objects.')
                pos += 1
                char = next_char()
            if char != '{':
                raise ValueError('Expected a patch object (truncated input or trailing comma).')
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, pos)
                    pos = end
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError('Invalid or truncated patch JSON.') from exc
                    refill()
            yield record
            first = False
        if next_char() is not None:
            raise ValueError('Unexpected content after the JSON array.')


def require_int(value, label, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be an integer >= {minimum}.')
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f'{label} must be an integer >= {minimum}.')
        value = int(value)
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}.')
    return value


def detail_prototype_index(item):
    for key in ('prototype_index', 'prot_idx'):
        if key in item:
            return require_int(item[key], 'prototype index')
    raise ValueError('Prototype detail must contain prototype_index or prot_idx.')


def ranked_prototype_ids(record):
    """Use details when available; also support the exported ordered ID list."""
    details = record.get('top_20_activated_prototypes')
    ids = record.get('top_20_activated_prototype_indices')
    if details is not None:
        if not isinstance(details, list) or any(not isinstance(item, dict) for item in details):
            raise ValueError('top_20_activated_prototypes must be a list of objects.')
        if any('rank' in item for item in details):
            ranks = [require_int(item.get('rank'), 'prototype rank', 1) for item in details]
            if len(set(ranks)) != len(ranks):
                raise ValueError('Duplicate prototype ranks.')
            details = sorted(details, key=lambda item: item['rank'])
        extracted = [detail_prototype_index(item) for item in details]
        if ids is not None and ids != extracted:
            raise ValueError('Ranked prototype details and ordered index list disagree.')
        ids = extracted
    if not isinstance(ids, list):
        raise ValueError('Patch must contain an ordered top_20_activated_prototypes or index list.')
    ids = [require_int(value, 'prototype index') for value in ids]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prototype IDs in one patch.')
    if len(ids) > 20:
        raise ValueError('Expected at most 20 exported prototype IDs per patch.')
    return ids


def choose_two(count):
    return count * (count - 1) // 2


def ranked_strengths(record, ids, field):
    details = record.get('top_20_activated_prototypes')
    if not isinstance(details, list):
        raise ValueError('Strength analysis requires top_20_activated_prototypes details, not IDs alone.')
    values = {}
    for item in details:
        if field not in item:
            raise ValueError(f'Missing {field} for prototype {item.get("prototype_index")}; re-export patch data with scores.')
        values[item['prototype_index']] = finite_score(item, field)
    return [values[prototype] for prototype in ids]


class ActivationStats:
    """Stable online moments and correlations, conditional on selection."""
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = self.abs_sum = 0.0
        self.pred_mean = self.pred_m2 = self.pred_cov = 0.0
        self.gt_mean = self.gt_m2 = self.gt_cov = 0.0
        self.minimum = self.maximum = None
        self.negative = 0

    def add(self, x, predicted, ground_truth):
        self.n += 1
        dx = x - self.mean
        self.mean += dx / self.n
        self.m2 += dx * (x - self.mean)
        self.abs_sum += abs(x)
        self.minimum = x if self.minimum is None else min(self.minimum, x)
        self.maximum = x if self.maximum is None else max(self.maximum, x)
        self.negative += int(x < 0)
        for label, y in (('pred', predicted), ('gt', ground_truth)):
            delta = y - getattr(self, label + '_mean')
            mean = getattr(self, label + '_mean') + delta / self.n
            setattr(self, label + '_mean', mean)
            setattr(self, label + '_m2', getattr(self, label + '_m2') + delta * (y - mean))
            setattr(self, label + '_cov', getattr(self, label + '_cov') + dx * (y - mean))

    def summary(self, patch_count):
        def correlation(label):
            a, b = max(0.0, self.m2), max(0.0, getattr(self, label + '_m2'))
            if self.n < 3 or a <= 1e-20 or b <= 1e-20:
                return None
            return max(-1.0, min(1.0, getattr(self, label + '_cov') / math.sqrt(a * b)))
        return {
            'selected_patch_count': self.n,
            'selected_patch_percent': 100 * self.n / patch_count if patch_count else None,
            'mean_strength_when_selected': self.mean if self.n else None,
            'strength_std_when_selected': math.sqrt(max(0.0, self.m2) / self.n) if self.n else None,
            'mean_abs_strength_when_selected': self.abs_sum / self.n if self.n else None,
            'masked_mean_strength_all_patches': self.mean * self.n / patch_count if patch_count else None,
            'masked_mean_abs_strength_all_patches': self.abs_sum / patch_count if patch_count else None,
            'strength_min': self.minimum, 'strength_max': self.maximum,
            'negative_strength_count': self.negative,
            'mean_pred_saliency_when_selected': self.pred_mean if self.n else None,
            'mean_gt_saliency_when_selected': self.gt_mean if self.n else None,
            'strength_pred_saliency_pearson_when_selected': correlation('pred'),
            'strength_gt_saliency_pearson_when_selected': correlation('gt'),
        }


class StrengthBin:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.patch_count = self.eligible_count = 0
        self.samples = []
        self.prototype_stats = defaultdict(ActivationStats)
        self.total_magnitude = 0.0

    def add(self, ids, scores, predicted, ground_truth):
        self.patch_count += 1
        for p, value in zip(ids, scores):
            self.prototype_stats[p].add(value, predicted, ground_truth)
        vector = dict(zip(ids, scores))
        magnitude = sum(abs(value) for value in vector.values())
        self.total_magnitude += magnitude
        if magnitude == 0:
            return
        self.eligible_count += 1
        sample = (vector, magnitude)
        if len(self.samples) < self.capacity:
            self.samples.append(sample)
        else:
            index = self.rng.randrange(self.eligible_count)
            if index < self.capacity:
                self.samples[index] = sample


def strength_pair_metrics(left, right):
    a, mass_a = left
    b, mass_b = right
    common = a.keys() & b.keys()
    matched = sum(min(abs(a[p]), abs(b[p])) for p in common if (a[p] > 0) == (b[p] > 0))
    shared_mass = sum(abs(a[p]) + abs(b[p]) for p in common)
    weighted_dice = 2 * matched / (mass_a + mass_b)
    shared_agreement = 2 * matched / shared_mass if shared_mass else None
    return weighted_dice, shared_agreement, matched


def strength_overlap_summary(left, right, bin_a, bin_b, k, max_pairs, seed):
    within = right is None
    right = left if within else right
    n, m = len(left.samples), len(right.samples)
    candidates = choose_two(n) if within else n * m
    population = choose_two(left.eligible_count) if within else left.eligible_count * right.eligible_count
    count = min(candidates, max_pairs)
    rng = random.Random(seed)
    total = shared_total = magnitude_total = 0.0
    shared_pairs = 0
    if count == candidates:
        pairs = ((i, j) for i in range(n) for j in range(i + 1, n)) if within else ((i, j) for i in range(n) for j in range(m))
    elif within:
        # Independent uniform ordered distinct endpoint pairs; replacement between pairs.
        def draw_pairs():
            for _ in range(count):
                i = rng.randrange(n)
                j = rng.randrange(n - 1)
                yield i, j + int(j >= i)
        pairs = draw_pairs()
    else:
        pairs = ((rng.randrange(n), rng.randrange(m)) for _ in range(count))
    for i, j in pairs:
        dice, agreement, matched = strength_pair_metrics(left.samples[i], right.samples[j])
        total += dice
        magnitude_total += matched
        if agreement is not None:
            shared_total += agreement
            shared_pairs += 1
    exact = n == left.eligible_count and m == right.eligible_count and count == candidates
    return {
        'scope': 'within' if within else 'cross', 'saliency_bin_a': bin_a, 'saliency_bin_b': bin_b,
        'top_k': k, 'eligible_pair_population': population, 'evaluated_pair_count': count,
        'reservoir_patch_count_a': n, 'reservoir_patch_count_b': m,
        'nonzero_strength_patch_count_a': left.eligible_count, 'nonzero_strength_patch_count_b': right.eligible_count,
        'zero_strength_patches_a': left.patch_count - left.eligible_count,
        'zero_strength_patches_b': right.patch_count - right.eligible_count,
        'exact_population_mean': exact,
        'sampling': 'all reservoir pairs' if count == candidates else 'uniform endpoint pairs with replacement within uniform reservoirs',
        'mean_overlap_percent': 100 * total / count if count else None,
        'shared_strength_agreement_percent': 100 * shared_total / shared_pairs if shared_pairs else None,
        'evaluated_pairs_with_shared_nonzero_strength': shared_pairs,
        'mean_shared_activation_magnitude': magnitude_total / count if count else None,
    }


def strength_comparisons(rows, bins, top_k_values):
    comparisons = []
    for b in bins:
        for k in top_k_values:
            within = next(r for r in rows if r['scope'] == 'within' and r['saliency_bin_a'] == b and r['top_k'] == k)
            cross = [r for r in rows if r['scope'] == 'cross' and r['top_k'] == k and b in (r['saliency_bin_a'], r['saliency_bin_b']) and r['evaluated_pair_count']]
            result = {'saliency_bin': b, 'top_k': k, 'other_bins_compared': len(cross),
                      'within_evaluated_pair_count': within['evaluated_pair_count']}
            for metric, label in (('mean_overlap_percent', 'weighted_dice'), ('shared_strength_agreement_percent', 'shared_strength_agreement')):
                eligible = [r for r in cross if r[metric] is not None]
                avg = sum(r[metric] for r in eligible) / len(eligible) if eligible else None
                w = within[metric]
                delta = w - avg if w is not None and avg is not None else None
                result.update({label + '_within_percent': w, label + '_cross_equal_bin_percent': avg,
                               label + '_within_minus_cross_pp': delta,
                               label + '_within_exceeds_cross': delta > 0 if delta is not None else None})
            comparisons.append(result)
    return comparisons


def dataset_presence_summary(groups, top_k_values, presence_percent, total):
    minimum = max(1, math.ceil(total * presence_percent / 100))
    rows = []
    for k in top_k_values:
        frequencies = Counter()
        for group in groups.values():
            frequencies.update(group[k].frequency)
        for p, count in sorted(frequencies.items(), key=lambda pair: (-pair[1], pair[0])):
            rows.append({'top_k': k, 'prototype_index': p, 'patch_count': count,
                         'patch_percent': 100 * count / total,
                         'presence_min_patch_count': minimum, 'meets_presence_threshold': count >= minimum})
    return rows


def selection_saliency_comparison(selected, population):
    result = {}
    absent = population.n - selected.n
    for label in ('pred', 'gt'):
        mean_selected = getattr(selected, label + '_mean')
        mean_absent = (population.n * getattr(population, label + '_mean') - selected.n * mean_selected) / absent if absent else None
        result['mean_' + label + '_saliency_when_not_selected'] = mean_absent
        result['selected_minus_not_selected_' + label + '_saliency'] = mean_selected - mean_absent if absent and selected.n else None
    return result


class BinAccumulator:
    def __init__(self, top_k):
        self.top_k = top_k
        self.patch_count = 0
        self.size_counts = Counter()
        self.frequency_by_size = defaultdict(Counter)
        self.frequency = Counter()

    def add(self, ranked_ids):
        ids = ranked_ids[:self.top_k]
        size = len(ids)
        self.patch_count += 1
        self.size_counts[size] += 1
        self.frequency_by_size[size].update(ids)
        self.frequency.update(ids)

    def summarize(self, saliency_bin, presence_percent=5.0, extent=None):
        n = self.patch_count
        pairs = choose_two(n)
        empty = self.size_counts[0]
        eligible_pairs = choose_two(n - empty)
        shared_total = sum(choose_two(freq) for freq in self.frequency.values())
        dice_total = Fraction(0)
        sizes = sorted(size for size in self.size_counts if size > 0)
        for i, a in enumerate(sizes):
            freq_a = self.frequency_by_size[a]
            for b in sizes[i:]:
                if a == b:
                    common = sum(choose_two(freq) for freq in freq_a.values())
                else:
                    freq_b = self.frequency_by_size[b]
                    common = sum(freq * freq_b.get(prototype, 0)
                                 for prototype, freq in freq_a.items())
                # Every pair in this size group has the same Dice denominator.
                dice_total += Fraction(2 * common, a + b)
        universal = sorted(p for p, freq in self.frequency.items() if freq == n) if n else []
        minimum = max(1, math.ceil(n * presence_percent / 100)) if n else 0
        presence = sorted(p for p, freq in self.frequency.items() if freq >= minimum) if n else []
        return {
            'saliency_bin': saliency_bin, 'top_k': self.top_k,
            'patch_count': n, 'pair_count': pairs, 'eligible_pair_count': eligible_pairs,
            'empty_prototype_lists': empty,
            'patches_with_at_least_k_prototypes': self.size_counts[self.top_k],
            'patches_with_fewer_than_k_prototypes': n - self.size_counts[self.top_k],
            'mean_actual_top_k_length': sum(size * count for size, count in self.size_counts.items()) / n if n else None,
            'actual_top_k_length_histogram': dict(sorted(self.size_counts.items())),
            'mean_shared_prototype_count': shared_total / pairs if pairs else None,
            'mean_overlap_percent': float(100 * dice_total / eligible_pairs) if eligible_pairs else None,
            'mean_shared_fraction_of_requested_k_percent': 100 * shared_total / (pairs * self.top_k) if pairs else None,
            'sum_shared_prototypes': shared_total, 'sum_pairwise_dice': float(dice_total),
            'presence_min_patch_count': minimum, 'presence_prototype_count': len(presence),
            'presence_prototype_indices': presence,
            'presence_prototype_list': ', '.join(map(str, presence)),
            'universal_prototype_count': len(universal), 'prototypes_present_in_all_patches': universal,
            **(extent or {}),
            'prototype_frequencies': [
                {'prototype_index': prototype, 'patch_count': freq,
                 'patch_fraction': freq / n}
                for prototype, freq in sorted(self.frequency.items(), key=lambda pair: (-pair[1], pair[0]))
            ],
        }


def cross_bin_summary(left, right, bin_a, bin_b):
    if left.top_k != right.top_k:
        raise ValueError('Cross-bin comparison requires matching top-k.')
    pairs = left.patch_count * right.patch_count
    eligible = (left.patch_count - left.size_counts[0]) * (right.patch_count - right.size_counts[0])
    shared_ids = sorted(left.frequency.keys() & right.frequency.keys())
    shared_total = sum(left.frequency[p] * right.frequency[p] for p in shared_ids)
    dice_total = Fraction(0)
    for size_a, freq_a in left.frequency_by_size.items():
        if size_a == 0:
            continue
        for size_b, freq_b in right.frequency_by_size.items():
            if size_b == 0:
                continue
            common = sum(freq * freq_b.get(p, 0) for p, freq in freq_a.items())
            dice_total += Fraction(2 * common, size_a + size_b)
    universal_left = {p for p, freq in left.frequency.items() if freq == left.patch_count}
    universal_right = {p for p, freq in right.frequency.items() if freq == right.patch_count}
    return {
        'saliency_bin_a': bin_a, 'saliency_bin_b': bin_b, 'top_k': left.top_k,
        'patch_count_a': left.patch_count, 'patch_count_b': right.patch_count,
        'pair_count': pairs, 'eligible_pair_count': eligible,
        'overlapping_prototype_count': len(shared_ids),
        'overlapping_prototype_indices': shared_ids,
        'overlapping_prototype_list': ', '.join(map(str, shared_ids)),
        'cross_bin_universal_prototype_indices': sorted(universal_left & universal_right),
        'mean_shared_prototype_count': shared_total / pairs if pairs else None,
        'mean_overlap_percent': float(100 * dice_total / eligible) if eligible else None,
        'mean_shared_fraction_of_requested_k_percent': 100 * shared_total / (pairs * left.top_k) if pairs else None,
        'sum_shared_prototypes': shared_total, 'sum_pairwise_dice': float(dice_total),
    }


def finite_score(record, key):
    score = record.get(key)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError(f'{key} must be a finite number.')
    return float(score)


def make_bin_edges(low, high, saliency_bins):
    edges = [low + (high - low) * i / saliency_bins for i in range(saliency_bins + 1)]
    edges[0], edges[-1] = low, high
    return edges


def saliency_bin_number(score, edges):
    if not math.isfinite(score) or not edges[0] <= score <= edges[-1]:
        raise ValueError('Patch score outside the observed finite saliency range.')
    if edges[0] == edges[-1]:
        return 0
    return min(len(edges) - 2, max(0, bisect_right(edges, score) - 1))


def is_distant_cross_pair(bin_a, bin_b):
    """Cross partners of bin i are bins 1..i-2 and i+2..n, with bins numbered from 1."""
    return abs(bin_a - bin_b) >= 2


def within_vs_cross_comparisons(within_rows, cross_rows, top_k_values):
    comparisons = []
    adjacency = defaultdict(list)
    for row in cross_rows:
        if row['eligible_pair_count']:
            adjacency[(row['saliency_bin_a'], row['top_k'])].append(row)
            adjacency[(row['saliency_bin_b'], row['top_k'])].append(row)
    for row in within_rows:
        others = adjacency[(row['saliency_bin'], row['top_k'])]
        eligible = sum(other['eligible_pair_count'] for other in others)
        equal_mean = sum(other['mean_overlap_percent'] for other in others) / len(others) if others else None
        weighted_mean = 100 * sum(other['sum_pairwise_dice'] for other in others) / eligible if eligible else None
        within = row['mean_overlap_percent']
        delta = within - equal_mean if within is not None and equal_mean is not None else None
        comparisons.append({
            'saliency_bin': row['saliency_bin'], 'top_k': row['top_k'], 'patch_count': row['patch_count'],
            'within_overlap_percent': within, 'within_eligible_pair_count': row['eligible_pair_count'],
            'cross_equal_bin_overlap_percent': equal_mean,
            'cross_pair_weighted_overlap_percent': weighted_mean,
            'other_bins_compared': len(others), 'cross_eligible_pair_count': eligible,
            'within_minus_cross_percentage_points': delta,
            'within_minus_pair_weighted_cross_percentage_points': within - weighted_mean if within is not None and weighted_mean is not None else None,
            'within_to_cross_ratio': within / equal_mean if within is not None and equal_mean is not None and equal_mean > 0 else None,
            'within_exceeds_cross': delta > 0 if delta is not None else None,
        })
    overall = []
    for k in top_k_values:
        valid = [r for r in comparisons if r['top_k'] == k and r['within_minus_cross_percentage_points'] is not None]
        within = sum(r['within_overlap_percent'] for r in valid) / len(valid) if valid else None
        cross = sum(r['cross_equal_bin_overlap_percent'] for r in valid) / len(valid) if valid else None
        delta = within - cross if valid else None
        overall.append({
            'top_k': k, 'comparable_bin_count': len(valid),
            'mean_within_overlap_percent': within, 'mean_cross_overlap_percent': cross,
            'within_minus_cross_percentage_points': delta,
            'within_exceeds_cross': delta > 0 if delta is not None else None,
            'bins_with_higher_within_overlap': sum(r['within_exceeds_cross'] for r in valid),
            'fraction_bins_with_higher_within_overlap': sum(r['within_exceeds_cross'] for r in valid) / len(valid) if valid else None,
        })
    return comparisons, overall


def analyze(path, saliency_bins=20, cross_bin=True, presence_percent=5.0, top_k_values=TOP_K_VALUES,
            strength_field='forward_activation', reservoir_size=2000, strength_pairs=20000, seed=42):
    require_int(saliency_bins, 'saliency_bins', 1)
    if not math.isfinite(presence_percent) or not 0 < presence_percent <= 100:
        raise ValueError('presence_percent must be in (0,100].')
    if not top_k_values or any(k not in TOP_K_VALUES for k in top_k_values) or len(set(top_k_values)) != len(top_k_values):
        raise ValueError('top_k_values must contain distinct values from 5,10,20.')
    if strength_field not in STRENGTH_FIELDS:
        raise ValueError(f'strength_field must be one of {STRENGTH_FIELDS}.')
    require_int(reservoir_size, 'reservoir_size', 2)
    require_int(strength_pairs, 'strength_pairs', 1)
    print('Measuring the predicted-saliency range...', flush=True)
    low, high, expected = None, None, 0
    for index, record in enumerate(iter_patch_records(path)):
        try:
            score = finite_score(record, 'patch_pred_saliency')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Patch record {index}: {exc}') from exc
        low = score if low is None else min(low, score)
        high = score if high is None else max(high, score)
        expected += 1
    if not expected:
        raise ValueError('patch_data.json contains no patch records.')
    edges = make_bin_edges(low, high, saliency_bins)
    groups = {b: {k: BinAccumulator(k) for k in top_k_values} for b in range(saliency_bins)}
    strength_groups = {b: {k: StrengthBin(reservoir_size, seed + b * 1009 + k * 101) for k in top_k_values} for b in groups}
    global_strength_stats = defaultdict(ActivationStats)
    field_diagnostic = ActivationStats()
    global_patch_saliency = ActivationStats()
    bin_patch_saliency = {b: ActivationStats() for b in groups}
    extents = {b: {'pred_saliency_min': None, 'pred_saliency_max': None,
                   'gt_saliency_min': None, 'gt_saliency_max': None} for b in groups}
    total = 0
    for index, record in enumerate(iter_patch_records(path)):
        try:
            score = finite_score(record, 'patch_pred_saliency')
            gt = finite_score(record, 'patch_gt_saliency')
            number = saliency_bin_number(score, edges)
            ids = ranked_prototype_ids(record)
            scores = ranked_strengths(record, ids, strength_field)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Patch record {index}: {exc}') from exc
        for key, value in (('pred_saliency', score), ('gt_saliency', gt)):
            extent = extents[number]
            extent[key + '_min'] = value if extent[key + '_min'] is None else min(extent[key + '_min'], value)
            extent[key + '_max'] = value if extent[key + '_max'] is None else max(extent[key + '_max'], value)
        for accumulator in groups[number].values():
            accumulator.add(ids)
        for k in top_k_values:
            strength_groups[number][k].add(ids[:k], scores[:k], score, gt)
            for p, value in zip(ids[:k], scores[:k]):
                global_strength_stats[(k, p)].add(value, score, gt)
        for value in scores[:max(top_k_values)]:
            field_diagnostic.add(value, score, gt)
        global_patch_saliency.add(0.0, score, gt)
        bin_patch_saliency[number].add(0.0, score, gt)
        total += 1
        if total % 100000 == 0:
            print(f'Read {total:,} patch records...', flush=True)
    if total != expected:
        raise RuntimeError('Patch count changed between the range scan and bin assignment.')
    rows = [groups[b][k].summarize(b, presence_percent, extents[b]) for b in groups for k in top_k_values]
    cross_rows = [cross_bin_summary(groups[a][k], groups[b][k], a, b)
                  for a in groups for b in groups if is_distant_cross_pair(a, b) and b > a
                  for k in top_k_values] if cross_bin else []
    comparisons, overall = within_vs_cross_comparisons(rows, cross_rows, top_k_values)
    print('Computing strength-weighted patch-pair overlap...', flush=True)
    strength_rows = []
    for b in groups:
        for k in top_k_values:
            strength_rows.append(strength_overlap_summary(strength_groups[b][k], None, b, b, k,
                                                          strength_pairs, seed + b * 1009 + k))
    if cross_bin:
        for a in groups:
            print(f'Strength comparisons for bin {a + 1}/{saliency_bins}...', flush=True)
            for b in groups:
                if is_distant_cross_pair(a, b) and b > a:
                    for k in top_k_values:
                        strength_rows.append(strength_overlap_summary(strength_groups[a][k], strength_groups[b][k],
                                                                      a, b, k, strength_pairs,
                                                                      seed + a * 100003 + b * 1009 + k))
    for row in strength_rows:
        row['strength_field'] = strength_field
    strength_vs_cross = strength_comparisons(strength_rows, groups, top_k_values)
    strength_overall = []
    for k in top_k_values:
        block = {'top_k': k}
        for label in ('weighted_dice', 'shared_strength_agreement'):
            valid = [r for r in strength_vs_cross if r['top_k'] == k and r[label + '_within_minus_cross_pp'] is not None]
            w = sum(r[label + '_within_percent'] for r in valid) / len(valid) if valid else None
            c = sum(r[label + '_cross_equal_bin_percent'] for r in valid) / len(valid) if valid else None
            block.update({label + '_comparable_bin_count': len(valid), label + '_within_percent': w,
                          label + '_cross_percent': c, label + '_within_minus_cross_pp': w - c if valid else None})
        strength_overall.append(block)
    bin_profiles = []
    for b in groups:
        for k in top_k_values:
            for p, stat in sorted(strength_groups[b][k].prototype_stats.items()):
                bin_profiles.append({'saliency_bin': b, 'top_k': k, 'prototype_index': p,
                                     'strength_field': strength_field,
                                     'bin_patch_count': strength_groups[b][k].patch_count,
                                     **stat.summary(strength_groups[b][k].patch_count),
                                     **selection_saliency_comparison(stat, bin_patch_saliency[b])})
    associations = [{'top_k': k, 'prototype_index': p, 'strength_field': strength_field,
                     'dataset_patch_count': total, **stat.summary(total),
                     **selection_saliency_comparison(stat, global_patch_saliency)}
                    for (k, p), stat in sorted(global_strength_stats.items())]
    diagnostic = field_diagnostic.summary(field_diagnostic.n)
    warnings = []
    if field_diagnostic.n == 0:
        warnings.append('No selected prototype scores were available; strength comparisons are undefined.')
    elif field_diagnostic.maximum - field_diagnostic.minimum <= 1e-8 * max(1.0, abs(field_diagnostic.mean)):
        warnings.append(f'{strength_field} is nearly constant across selected assignments. Strength overlap adds little beyond ID overlap; consider --strength-field cosine_similarity if saved.')
    constant_prototypes = sum(1 for stat in global_strength_stats.values()
                              if stat.n >= 3 and stat.maximum - stat.minimum <= 1e-8 * max(1.0, abs(stat.mean)))
    if constant_prototypes:
        warnings.append(f'{constant_prototypes} top-k/prototype profiles have nearly constant selected strengths; their strength correlations may be uninformative.')
    matrices = {}
    for k in top_k_values:
        matrix = [[None] * saliency_bins for _ in groups]
        for row in rows:
            if row['top_k'] == k:
                matrix[row['saliency_bin']][row['saliency_bin']] = row['mean_overlap_percent']
        for row in cross_rows:
            if row['top_k'] == k:
                a, b = row['saliency_bin_a'], row['saliency_bin_b']
                matrix[a][b] = matrix[b][a] = row['mean_overlap_percent']
        matrices[str(k)] = matrix
    return {
        'patch_data_path': str(Path(path).resolve()), 'patch_count': total,
        'saliency_bins': saliency_bins, 'bin_edges': edges,
        'bin_counts': [groups[b][top_k_values[0]].patch_count for b in groups],
        'bin_source': 'patch_pred_saliency', 'bin_method': 'global_equal_width_observed_range',
        'bin_interval_rule': '[left,right); last includes max; constant range uses bin 0',
        'top_k_values': list(top_k_values), 'prototype_presence_percent': presence_percent,
        'cross_bin_enabled': cross_bin,
        'comparison_scope': 'all within-bin unordered patch pairs; cross pairs only between bins at least two apart (1-based bins 1..i-2 and i+2..n); same-video pairs included',
        'cross_bin_partner_rule': 'for saliency bin i numbered 1..n, cross uses bins 1..i-2 and i+2..n; adjacent bins are excluded',
        'ranking': 'exported forward-activation order; no cosine reranking',
        'method': 'exact prototype frequency counts grouped by actual list length; no pair sampling',
        'report_filter': 'adjacent bin pairs are omitted from cross; retained cross rows include zero overlaps',
        'definitions': {
            'mean_shared_prototype_count': 'mean |A intersection B| over all compared patch pairs; every shared ID counts',
            'mean_overlap_percent': '100 * mean(2*|A intersection B|/(|A|+|B|)) over pairs with both lists nonempty',
            'mean_shared_fraction_of_requested_k_percent': '100 * mean |A intersection B| / requested k, over all pairs',
            'short_lists': 'Use actual length up to k; no padding; report length counts.',
            'universal_prototypes': 'Separate descriptive field only; never used to restrict overlap.',
            'presence_threshold': 'Descriptive ID list only; never used to restrict metrics or omit bins.',
            'cross_equal_bin_overlap_percent': 'arithmetic mean of this bin\'s cross-overlap percentages over bins at least two away with eligible pairs',
            'cross_pair_weighted_overlap_percent': 'pooled cross-bin Dice over eligible pairs with bins at least two away',
            'within_minus_cross_percentage_points': 'within overlap minus cross_equal_bin overlap; positive means higher within-bin overlap',
            'overall_comparison': 'equal weight per comparable bin, averaging its within and cross_equal_bin values over the same bins',
            'dice_matrices_percent': 'rows/columns in ascending 0-based bin order; diagonal = within; |i-j|>=2 = cross; adjacent and unavailable cells are null',
            'insufficient_pairs': 'null when relevant pair count is zero',
            'uncertainty': 'descriptive comparison, no p-values or confidence intervals; pairs share patches/videos',
        },
        'results': rows, 'cross_bin_results': cross_rows,
        'within_vs_cross_results': comparisons, 'overall_comparison': overall,
        'dice_matrices_percent': matrices,
        'dataset_presence': dataset_presence_summary(groups, top_k_values, presence_percent, total),
        'strength_analysis': {
            'strength_field': strength_field, 'reservoir_size_per_bin_per_k': reservoir_size,
            'max_evaluated_pairs_per_comparison': strength_pairs, 'seed': seed,
            'pair_means_method': 'uniform eligible-patch reservoirs plus uniform endpoint pairs; exact only where flagged',
            'zero_strength_rule': 'exclude patches with zero total absolute selected strength from strength pair comparisons',
            'signed_strength_rule': 'same prototype and same sign required; use absolute magnitudes with separate positive/negative channels',
            'normalization': 'none: original magnitudes are retained',
            'weighted_dice_definition': '2 * sum same-sign shared minimum magnitudes / sum absolute magnitudes of both full selected vectors',
            'shared_strength_agreement_definition': 'same numerator, but denominator includes only shared prototype IDs; mean conditional on shared positive denominator',
            'profile_method': 'exact full-data statistics conditional on being in the exported top-k list',
            'masked_mean_rule': 'scores absent from exported top-k are zero only for masked_mean_strength_all_patches; no unexported score is inferred',
            'correlation_rule': 'Pearson among selected patches; null for fewer than 3 observations or zero variance; association, not causation',
            'selection_comparison': 'mean saliency when prototype is selected minus mean when not selected; separates selection association from conditional-strength association',
            'field_diagnostic': diagnostic, 'warnings': warnings,
            'overlap_results': strength_rows, 'within_vs_cross_results': strength_vs_cross,
            'overall_comparison': strength_overall,
            'prototype_bin_profiles': bin_profiles, 'prototype_saliency_associations': associations,
        },
    }


def text_report(summary):
    lines = [f'Within-bin prototype overlap ({summary["patch_count"]:,} patches)',
             'Exact all-pair means. Overlap = Dice percentage; empty-list pairs excluded from this percentage.',
             'Every shared ID counts. Universal membership and presence thresholds are descriptive only.',
             'Short lists use their actual lengths. Same-video pairs are included. No bins are omitted.',
             '', 'bin  top-k    patches          pairs   short lists   mean shared   overlap %   IDs at presence threshold']
    for row in summary['results']:
        shared = 'N/A' if row['mean_shared_prototype_count'] is None else f'{row["mean_shared_prototype_count"]:.4f}'
        overlap = 'N/A' if row['mean_overlap_percent'] is None else f'{row["mean_overlap_percent"]:.2f}'
        lines.append(f'{row["saliency_bin"]:>3}  {row["top_k"]:>5}  {row["patch_count"]:>9,} '
                     f'{row["pair_count"]:>14,}  {row["patches_with_fewer_than_k_prototypes"]:>12,} '
                     f'{shared:>13}  {overlap:>10}   {row["presence_prototype_list"]}')
    if summary['cross_bin_enabled']:
        def number(value):
            return 'N/A' if value is None else f'{value:.4f}'
        lines += ['', 'Within versus cross-bin overlap',
                  'Cross for bin i uses bins 1..i-2 and i+2..n (bins numbered from 1). Adjacent bins are excluded.',
                  'cross = equal mean over those eligible bins; delta = within - cross, in percentage points.',
                  'bin  top-k   within %    cross %    delta pp   within higher?']
        for row in summary['within_vs_cross_results']:
            higher = 'N/A' if row['within_exceeds_cross'] is None else ('yes' if row['within_exceeds_cross'] else 'no')
            lines.append(f'{row["saliency_bin"]:>3} {row["top_k"]:>6} '
                         f'{number(row["within_overlap_percent"]):>10} '
                         f'{number(row["cross_equal_bin_overlap_percent"]):>10} '
                         f'{number(row["within_minus_cross_percentage_points"]):>11} {higher:>16}')
        lines += ['', 'Overall comparison (equal weight per comparable bin)',
                  'top-k   bins   within %    cross %    delta pp   bins with higher within']
        for row in summary['overall_comparison']:
            lines.append(f'{row["top_k"]:>5} {row["comparable_bin_count"]:>6} '
                         f'{number(row["mean_within_overlap_percent"]):>10} '
                         f'{number(row["mean_cross_overlap_percent"]):>10} '
                         f'{number(row["within_minus_cross_percentage_points"]):>11} '
                         f'{row["bins_with_higher_within_overlap"]:>25}')
        lines += ['', 'Full bin-pair overlap table: prototype_overlap_cross_bin.csv.',
                  'These differences are descriptive; they do not establish statistical significance.']
    strength = summary['strength_analysis']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4f}'
    lines += ['', f'Strength-weighted overlap: {strength["strength_field"]}',
              'Cross for bin i uses bins 1..i-2 and i+2..n (bins numbered from 1). Adjacent bins are excluded.',
              'Original magnitudes retained. Signed scores match only the same prototype and sign.',
              'Pair strength metrics may be sampled; see counts and exact_population_mean in the strength CSV.',
              'Agreement is conditional on shared IDs with nonzero strength; it isolates strength from identity.',
              'bin top-k  weighted within %  weighted cross %   delta pp  shared agreement within %  shared agreement cross %']
    for row in strength['within_vs_cross_results']:
        lines.append(f'{row["saliency_bin"]:>3} {row["top_k"]:>5} '
                     f'{fmt(row["weighted_dice_within_percent"]):>18} '
                     f'{fmt(row["weighted_dice_cross_equal_bin_percent"]):>17} '
                     f'{fmt(row["weighted_dice_within_minus_cross_pp"]):>10} '
                     f'{fmt(row["shared_strength_agreement_within_percent"]):>26} '
                     f'{fmt(row["shared_strength_agreement_cross_equal_bin_percent"]):>25}')
    lines += ['', 'Overall strength comparison (equal weight per comparable bin)',
              'top-k  weighted within %  weighted cross %  weighted delta pp  agreement within %  agreement cross %']
    for row in strength['overall_comparison']:
        lines.append(f'{row["top_k"]:>5} {fmt(row["weighted_dice_within_percent"]):>18} '
                     f'{fmt(row["weighted_dice_cross_percent"]):>17} '
                     f'{fmt(row["weighted_dice_within_minus_cross_pp"]):>18} '
                     f'{fmt(row["shared_strength_agreement_within_percent"]):>19} '
                     f'{fmt(row["shared_strength_agreement_cross_percent"]):>18}')
    for warning in strength['warnings']:
        lines.append('Score diagnostic: ' + warning)
    lines += ['', 'Exact strength profiles by prototype/bin: prototype_activation_by_bin.csv.',
              'Exact strength/saliency associations: prototype_activation_saliency_association.csv.',
              'These associations do not measure a prototype\'s causal contribution to saliency.']
    lines += ['', f'Dataset-wide IDs meeting the {summary["prototype_presence_percent"]:g}% presence threshold:']
    for k in summary['top_k_values']:
        ids = [row['prototype_index'] for row in summary['dataset_presence']
               if row['top_k'] == k and row['meets_presence_threshold']]
        lines.append(f'Top-{k}: ' + (', '.join(map(str, ids)) if ids else '(none)'))
    return '\n'.join(lines) + '\n'


def save_reports(summary, output_dir, overwrite=False):
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((root / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError('Overlap reports exist. Use another --output-dir or --overwrite-results.')
    # Complete all outputs before replacing any prior report.
    with tempfile.TemporaryDirectory(prefix='.overlap_', dir=root) as temporary:
        tmp = Path(temporary)
        (tmp / OUTPUT_NAMES[0]).write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        for name, rows, empty_header in (
                (OUTPUT_NAMES[1], summary['results'], ('saliency_bin', 'top_k')),
                (OUTPUT_NAMES[3], summary['cross_bin_results'], ('saliency_bin_a', 'saliency_bin_b', 'top_k')),
                (OUTPUT_NAMES[4], summary['within_vs_cross_results'], ('saliency_bin', 'top_k')),
                (OUTPUT_NAMES[5], summary['dataset_presence'], ('top_k', 'prototype_index')),
                (OUTPUT_NAMES[6], summary['strength_analysis']['overlap_results'], ('scope', 'top_k')),
                (OUTPUT_NAMES[7], summary['strength_analysis']['within_vs_cross_results'], ('saliency_bin', 'top_k')),
                (OUTPUT_NAMES[8], summary['strength_analysis']['prototype_bin_profiles'], ('saliency_bin', 'top_k', 'prototype_index')),
                (OUTPUT_NAMES[9], summary['strength_analysis']['prototype_saliency_associations'], ('top_k', 'prototype_index'))):
            fields = [key for key, value in rows[0].items() if not isinstance(value, (list, dict))] if rows else list(empty_header)
            with (tmp / name).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: row[key] for key in fields} for row in rows)
        (tmp / OUTPUT_NAMES[2]).write_text(text_report(summary), encoding='utf-8')
        for name in OUTPUT_NAMES:
            (tmp / name).replace(root / name)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--patch-data', required=True, help='Path to patch_data.json.')
    parser.add_argument('--output-dir', default=None, help='Default: directory containing patch_data.json.')
    parser.add_argument('--saliency-bins', '--saliency_bins', type=int, default=20)
    parser.add_argument('--top-k', type=int, nargs='+', choices=TOP_K_VALUES, default=list(DEFAULT_TOP_K_VALUES),
                        help='Default: 5; use --top-k 5 10 20 for all three cutoffs.')
    parser.add_argument('--prototype-presence-percent', '--prototype_presence_percent', type=float, default=5.0,
                        help='Descriptive prototype frequency threshold only (default 5%%).')
    parser.add_argument('--cross-bin', dest='cross_bin', action='store_true', default=True)
    parser.add_argument('--no-cross-bin', dest='cross_bin', action='store_false')
    parser.add_argument('--strength-field', choices=STRENGTH_FIELDS, default='forward_activation',
                        help='Saved score used as strength; never silently switched or reranked.')
    parser.add_argument('--strength-reservoir-size', type=int, default=2000,
                        help='Uniform nonzero-strength patch sample per bin/k for pair analysis.')
    parser.add_argument('--strength-pairs', type=int, default=20000,
                        help='Maximum evaluated strength pairs per within/cross bin comparison.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite-results', action='store_true')
    args = parser.parse_args()
    if args.saliency_bins < 1 or len(set(args.top_k)) != len(args.top_k):
        parser.error('saliency-bins must be positive and top-k values must be distinct.')
    if not math.isfinite(args.prototype_presence_percent) or not 0 < args.prototype_presence_percent <= 100:
        parser.error('prototype-presence-percent must be in (0,100].')
    if args.strength_reservoir_size < 2 or args.strength_pairs < 1:
        parser.error('strength-reservoir-size must be >=2 and strength-pairs >=1.')
    source = Path(args.patch_data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else source.parent
    try:
        summary = analyze(source, args.saliency_bins, args.cross_bin, args.prototype_presence_percent, tuple(args.top_k),
                          args.strength_field, args.strength_reservoir_size, args.strength_pairs, args.seed)
        root = save_reports(summary, output_dir, args.overwrite_results)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(text_report(summary), end='')
    print(f'Saved reports to {root}')


if __name__ == '__main__':
    main()
