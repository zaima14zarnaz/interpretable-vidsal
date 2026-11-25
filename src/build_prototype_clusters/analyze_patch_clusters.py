#!/usr/bin/env python3
"""Report within/between-cluster patch-pair visual similarity, mean +/- SD.

Dependency: python -m pip install numpy
    python analyze_patch_clusters.py
    python analyze_patch_clusters.py --cluster-dir /path/to/patch_cluster
    python analyze_patch_clusters.py --intra-pairs 20000 --inter-pairs 2000
    python analyze_patch_clusters.py --exact    # potentially VERY expensive

Reads the output of form_patch_clusters.py, independently of the other scripts.
Default input: /data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster
Default reports: <cluster-dir>/similarity_analysis/
    cluster_similarity_summary.json
    intra_cluster_similarity.csv   (one row per prototype/cluster)
    inter_cluster_similarity.csv   (one row per unordered cluster pair)

VISUAL SIMILARITY = stage4 feature cosine + (1 - RGB MAE)
                  + per-channel centered RGB structure cosine.
It is NOT distance to the medoid and NOT cosine between cluster centroids.
Feature/structure cosines range from -1 to 1; the sum ranges from -2 to 3.
Both flat RGB structures have cosine 1; one flat structure gives cosine 0.
All component means/SDs and a manifest-weighted similarity are also reported.

Intra: unordered distinct patch pairs i<j inside each cluster.
Inter: Cartesian patch pairs between each unordered cluster pair a<b.
Self pairs are excluded. Distinct occurrences from the same video are included.

Small populations are evaluated exactly; larger populations use a seeded,
uniform sample WITHOUT replacement of unique pair ranks in each stratum.
--intra-pairs (default 5000) caps pairs PER CLUSTER.
--inter-pairs (default 500) caps pairs PER CLUSTER PAIR.
--exact enumerates all pairs in bounded batches, without an NxN matrix.
Large sampled reports are ESTIMATES, not exact all-pair statistics.

SD is population-style pair-score dispersion (ddof=0), NOT uncertainty or a
standard error of the mean. For sampled rows it is the observed sample's
population-style SD, used as an estimate of the full stratum's dispersion.
Singleton clusters have no intra pairs: their means/SDs are null/blank.

Overall micro results weight each stratum by its TOTAL POSSIBLE pair count,
not by the sampled count. Macro results give each eligible cluster/cluster pair
equal weight; their SD describes variation AMONG STRATUM MEANS, not pair SD.
The summary records possible/evaluated counts, seeds and exact/estimated modes.
These metrics quantify consistency in the descriptors used to form clusters;
they are not independent evidence of human visual interpretability.

No inference, source images, graph edges, model assignments or torch required.
Source arrays/databases are read-only. Only the analysis reports are written.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

DEFAULT_CLUSTER_DIR = Path('/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster')
FORMAT_VERSION = 'fixed_stage4_patch_clusters_v1'
METRICS = ('visual_similarity', 'feature_cosine', 'rgb_mae', 'rgb_structure_cosine', 'weighted_visual_similarity')
INT64_MAX = np.iinfo(np.int64).max


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def json_write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Moments:
    """Stable batch-merged moments, one independent accumulator per metric."""
    def __init__(self):
        self.count = 0
        self.mean = np.zeros(len(METRICS), dtype=np.float64)
        self.m2 = np.zeros(len(METRICS), dtype=np.float64)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(METRICS) or not np.isfinite(values).all():
            raise ValueError('Invalid/nonfinite pair metric values.')
        n = len(values)
        if not n:
            return
        mean = values.mean(axis=0)
        m2 = ((values - mean) ** 2).sum(axis=0)
        self.merge(n, mean, m2)

    def merge(self, count, mean, m2):
        if count == 0:
            return
        total = self.count + count
        delta = np.asarray(mean) - self.mean
        self.m2 += m2 + delta ** 2 * (self.count / total) * count
        self.mean += delta * (count / total)
        self.count = total

    def merge_population_estimate(self, population, observed):
        if not population or not observed.count:
            return
        # Reweight each stratum's estimated first/second moments by the true
        # population size, preserving within- and between-stratum variation.
        estimated_m2 = (observed.m2 / observed.count) * population
        self.merge(population, observed.mean, estimated_m2)

    def report(self):
        return {name: {'mean': float(self.mean[i]) if self.count else None,
                       'std': float(math.sqrt(max(0., self.m2[i] / self.count))) if self.count else None}
                for i, name in enumerate(METRICS)}


class ClusterData:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest_path = self.directory / 'manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('format_version') != FORMAT_VERSION or self.manifest.get('status') != 'complete':
            raise ValueError('Expected a complete fixed patch-cluster run from form_patch_clusters.py.')
        self.count = int(self.manifest['num_patches'])
        self.k = int(self.manifest['num_prototypes'])
        if self.count < 1 or self.k < 1:
            raise ValueError('Invalid patch/prototype counts in manifest.')
        root = self.directory / 'descriptors'
        for name in ('feature', 'rgb', 'structure', 'flat', 'node_ids', 'prototype_ids'):
            value = np.load(root / f'{name}.npy', mmap_mode='r', allow_pickle=False)
            setattr(self, name, value)
        if self.feature.shape != (self.count, int(self.manifest['feature_dim'])):
            raise ValueError('Feature-array shape differs from manifest.')
        rgb_dim = 3 * int(self.manifest['rgb_descriptor_size']) ** 2
        if self.rgb.shape != (self.count, rgb_dim) or self.structure.shape != self.rgb.shape:
            raise ValueError('RGB/structure array shapes differ from manifest.')
        for name in ('flat', 'node_ids', 'prototype_ids'):
            if getattr(self, name).shape != (self.count,):
                raise ValueError(f'{name} array shape differs from manifest.')
        if self.flat.dtype != np.bool_ or self.node_ids.dtype.kind not in 'iu' or self.prototype_ids.dtype.kind not in 'iu':
            raise ValueError('Invalid flat-mask/node-ID/prototype-ID dtypes.')
        if any(getattr(self, name).dtype.kind != 'f' for name in ('feature', 'rgb', 'structure')):
            raise ValueError('Descriptor arrays must have floating-point dtypes.')
        if np.any(self.prototype_ids < 0) or np.any(self.prototype_ids >= self.k):
            raise ValueError('Prototype IDs outside the manifest range.')
        weights = self.manifest['weights']
        self.weights = np.array([weights['feature'], weights['rgb_mae'], weights['structure']], dtype=np.float64)
        if not np.isfinite(self.weights).all() or (self.weights < 0).any() or not self.weights.any():
            raise ValueError('Invalid metric weights in manifest.')
        self._verify_memberships()
        sizes = np.bincount(self.prototype_ids, minlength=self.k)
        sorted_rows = np.argsort(self.prototype_ids, kind='stable')
        boundaries = np.concatenate(([0], np.cumsum(sizes)))
        self.groups = [sorted_rows[boundaries[j]:boundaries[j + 1]] for j in range(self.k)]

    def _verify_memberships(self):
        # The viewer uses SQLite. Ensure the NPY membership arrays describe
        # exactly those same patch identities and cluster labels.
        uri = (self.directory / 'patch_clusters.sqlite').as_uri() + '?mode=ro'
        with closing(sqlite3.connect(uri, uri=True)) as con:
            cursor = con.execute('SELECT descriptor_index,node_id,prototype_id FROM patches ORDER BY descriptor_index')
            position = 0
            while True:
                rows = cursor.fetchmany(16384)
                if not rows:
                    break
                values = np.asarray(rows, dtype=np.int64)
                stop = position + len(values)
                if (stop > self.count or not np.array_equal(values[:, 0], np.arange(position, stop)) or
                        not np.array_equal(values[:, 1], self.node_ids[position:stop]) or
                        not np.array_equal(values[:, 2], self.prototype_ids[position:stop])):
                    raise ValueError('Descriptor identity/membership arrays disagree with the cluster database.')
                position = stop
            if position != self.count:
                raise ValueError('Database patch count differs from manifest.')
            prototypes = con.execute('SELECT id,patch_count FROM prototypes ORDER BY id').fetchall()
            expected = np.bincount(self.prototype_ids, minlength=self.k)
            if prototypes != [(j, int(expected[j])) for j in range(self.k)]:
                raise ValueError('Prototype summary counts disagree with patch memberships.')

    def score_pairs(self, left, right):
        """One row per actual patch pair; no Cartesian descriptor broadcast."""
        fa = np.asarray(self.feature[left], dtype=np.float64)
        fb = np.asarray(self.feature[right], dtype=np.float64)
        ra = np.asarray(self.rgb[left], dtype=np.float64)
        rb = np.asarray(self.rgb[right], dtype=np.float64)
        sa = np.asarray(self.structure[left], dtype=np.float64)
        sb = np.asarray(self.structure[right], dtype=np.float64)
        if any(not np.isfinite(v).all() for v in (fa, fb, ra, rb, sa, sb)):
            raise ValueError('Nonfinite descriptor encountered in an evaluated pair.')
        if ra.min() < 0 or rb.min() < 0 or ra.max() > 1 or rb.max() > 1:
            raise ValueError('RGB descriptors must be in [0,1].')
        feature_denominator = np.linalg.norm(fa, axis=1) * np.linalg.norm(fb, axis=1)
        if (np.linalg.norm(fa, axis=1) < 1e-12).any() or (np.linalg.norm(fb, axis=1) < 1e-12).any():
            raise ValueError('Zero stage4 feature encountered; similarity is undefined.')
        fc = np.clip(np.einsum('ij,ij->i', fa, fb) / feature_denominator, -1., 1.)
        mae = np.abs(ra - rb).mean(axis=1)
        flat_a, flat_b = np.asarray(self.flat[left]), np.asarray(self.flat[right])
        na, nb = np.linalg.norm(sa, axis=1), np.linalg.norm(sb, axis=1)
        if (((na < 1e-12) != flat_a).any() or ((nb < 1e-12) != flat_b).any()):
            raise ValueError('RGB structure vectors disagree with their flat flags.')
        sc = np.divide(np.einsum('ij,ij->i', sa, sb), na * nb,
                       out=np.zeros(len(left), dtype=np.float64), where=~(flat_a | flat_b))
        sc = np.clip(sc, -1., 1.)
        sc[flat_a & flat_b] = 1.
        visual = fc + (1. - mae) + sc
        weighted = self.weights[0] * fc + self.weights[1] * (1. - mae) + self.weights[2] * sc
        return np.column_stack((visual, fc, mae, sc, weighted))


def sample_unique_ranks(population, budget, rng):
    """Floyd sampling: uniform subset, O(budget) memory, no O(population) array."""
    if not 0 <= budget <= population or population > INT64_MAX:
        raise ValueError('Invalid/unsupported pair population or sampling budget.')
    selected = set()
    for j in range(population - budget, population):
        draw = int(rng.integers(j + 1))
        selected.add(j if draw in selected else draw)
    return np.array(sorted(selected), dtype=np.int64)


def intra_batches(members, budget, exact, rng, batch_size):
    n = len(members)
    possible = n * (n - 1) // 2
    if possible > INT64_MAX:
        raise ValueError('Intra-cluster pair population exceeds supported int64 indexing.')
    if exact or possible <= budget:
        for i in range(n - 1):
            for start in range(i + 1, n, batch_size):
                right = members[start:start + batch_size]
                yield np.full(len(right), members[i], dtype=np.int64), right
    else:
        ranks = sample_unique_ranks(possible, budget, rng)
        # Row i of the upper triangle starts at i*(2*n-i-1)/2.
        # Searchsorted avoids a numerically fragile inverse quadratic formula.
        rows = np.arange(n, dtype=np.int64)
        starts = rows * (2 * n - rows - 1) // 2
        for start in range(0, len(ranks), batch_size):
            r = ranks[start:start + batch_size]
            i = np.searchsorted(starts, r, side='right') - 1
            j = i + 1 + r - starts[i]
            yield members[i], members[j]


def inter_batches(left, right, budget, exact, rng, batch_size):
    possible = len(left) * len(right)
    if possible > INT64_MAX:
        raise ValueError('Inter-cluster pair population exceeds supported int64 indexing.')
    if possible == 0:
        return
    ranks = None if exact or possible <= budget else sample_unique_ranks(possible, budget, rng)
    evaluated = possible if ranks is None else len(ranks)
    for start in range(0, evaluated, batch_size):
        r = np.arange(start, min(evaluated, start + batch_size), dtype=np.int64) if ranks is None else ranks[start:start + batch_size]
        yield left[r // len(right)], right[r % len(right)]


def rng_for(seed, kind, a, b=0):
    return np.random.default_rng(np.random.SeedSequence([seed, kind, a, b]))


def evaluate(data, pairs):
    observed = Moments()
    for left, right in pairs:
        if len(left):
            observed.add(data.score_pairs(left, right))
    return observed


def row_with_stats(identity, population, observed):
    row = dict(identity)
    row.update(possible_pair_count=population, evaluated_pair_count=observed.count,
               mode=('no_pairs' if not population else 'exact' if observed.count == population else 'sampled_estimate'))
    for metric, statistics in observed.report().items():
        row[metric + '_mean'] = statistics['mean']
        row[metric + '_std'] = statistics['std']
    return row


class Summary:
    def __init__(self):
        self.pair_weighted = Moments()
        self.stratum_means = Moments()
        self.strata = self.eligible_strata = self.exact_strata = self.sampled_strata = 0
        self.evaluated_pairs = 0

    def add(self, population, observed):
        self.strata += 1
        if not population:
            return
        self.eligible_strata += 1
        self.exact_strata += int(observed.count == population)
        self.sampled_strata += int(observed.count != population)
        self.evaluated_pairs += observed.count
        self.pair_weighted.merge_population_estimate(population, observed)
        self.stratum_means.add(observed.mean[None, :])

    def report(self, stratum_label):
        return {'mode': 'sampled_estimate' if self.sampled_strata else 'exact',
                'possible_pair_count': self.pair_weighted.count, 'evaluated_pair_count': self.evaluated_pairs,
                'strata': self.strata, 'eligible_strata': self.eligible_strata,
                'exact_strata': self.exact_strata, 'sampled_strata': self.sampled_strata,
                'micro_pair_weighted': self.pair_weighted.report(),
                'macro_equal_stratum_weight': self.stratum_means.report(),
                'macro_std_definition': f'SD across {stratum_label} mean scores; not SD across individual patch pairs'}


def write_csv(path, rows, identity_columns):
    columns = list(identity_columns) + ['possible_pair_count', 'evaluated_pair_count', 'mode']
    columns += [metric + suffix for metric in METRICS for suffix in ('_mean', '_std')]
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def analyze(args):
    started = time.monotonic()
    data = ClusterData(args.cluster_dir)
    output = (args.output_dir.expanduser().resolve() if args.output_dir else data.directory / 'similarity_analysis')
    output.mkdir(parents=True, exist_ok=True)
    intra_populations = [len(group) * (len(group) - 1) // 2 for group in data.groups]
    total_intra = sum(intra_populations)
    total_inter = data.count * (data.count - 1) // 2 - total_intra
    planned_intra = total_intra if args.exact else sum(min(n, args.intra_pairs) for n in intra_populations)
    planned_inter = (total_inter if args.exact else
                     sum(min(len(data.groups[a]) * len(data.groups[b]), args.inter_pairs)
                         for a, b in itertools.combinations(range(data.k), 2)))
    print(f'{data.count:,} patches in {data.k:,} clusters.', flush=True)
    print(f'Intra: evaluating {planned_intra:,} of {total_intra:,} possible unique pairs.', flush=True)
    print(f'Inter: evaluating {planned_inter:,} of {total_inter:,} possible unique pairs.', flush=True)
    intra, inter = Summary(), Summary()
    intra_rows, inter_rows = [], []
    for prototype_id, group in enumerate(data.groups):
        observed = evaluate(data, intra_batches(group, args.intra_pairs, args.exact,
                            rng_for(args.seed, 1, prototype_id), args.batch_size))
        population = intra_populations[prototype_id]
        intra.add(population, observed)
        intra_rows.append(row_with_stats({'prototype_id': prototype_id, 'patch_count': len(group)}, population, observed))
        if prototype_id == 0 or (prototype_id + 1) % 16 == 0 or prototype_id + 1 == data.k:
            print(f'Intra clusters: {prototype_id + 1}/{data.k}', flush=True)
    pair_count = data.k * (data.k - 1) // 2
    for position, (a, b) in enumerate(itertools.combinations(range(data.k), 2), 1):
        left, right = data.groups[a], data.groups[b]
        population = len(left) * len(right)
        observed = evaluate(data, inter_batches(left, right, args.inter_pairs, args.exact,
                            rng_for(args.seed, 2, a, b), args.batch_size))
        inter.add(population, observed)
        inter_rows.append(row_with_stats({'prototype_a': a, 'prototype_b': b,
                                          'patch_count_a': len(left), 'patch_count_b': len(right)}, population, observed))
        if position == 1 or position % 100 == 0 or position == pair_count:
            print(f'Inter cluster pairs: {position}/{pair_count}', flush=True)
    if (intra.evaluated_pairs != planned_intra or inter.evaluated_pairs != planned_inter or
            intra.pair_weighted.count != total_intra or inter.pair_weighted.count != total_inter):
        raise RuntimeError('Pair accounting failed; reports were not written.')
    summary = {'format_version': 'patch_cluster_similarity_analysis_v1', 'status': 'complete', 'created_at': now(),
               'cluster_directory': str(data.directory), 'input_manifest_sha256': sha256_file(data.manifest_path),
               'input_reference_sha256': data.manifest.get('references_sha256'),
               'num_patches': data.count, 'num_prototypes': data.k,
               'visual_similarity_formula': 'feature_cosine + (1 - rgb_mae) + centered_rgb_structure_cosine',
               'visual_similarity_range': [-2, 3],
               'weighted_similarity_formula': 'wf*feature_cosine + wr*(1-rgb_mae) + ws*centered_rgb_structure_cosine',
               'weights': dict(zip(('feature', 'rgb_mae', 'structure'), data.weights.tolist())),
               'rgb_descriptor_size': data.manifest['rgb_descriptor_size'],
               'pair_definition': {'intra': 'unordered distinct patch pairs within each cluster',
                                   'inter': 'Cartesian patch pairs between unordered different cluster pairs',
                                   'self_pairs': False, 'same_video_pairs_included': True},
               'sampling': {'seed': args.seed, 'exact_requested': args.exact,
                            'intra_pairs_per_cluster': args.intra_pairs,
                            'inter_pairs_per_cluster_pair': args.inter_pairs,
                            'method': 'uniform unique pair ranks without replacement per stratum (Floyd sampling)',
                            'batch_size': args.batch_size},
               'std_definition': 'population-style SD (ddof=0) of pair scores; sampled results estimate full-population dispersion, not uncertainty',
               'intra': intra.report('cluster'), 'inter': inter.report('cluster-pair'),
               'interpretation': 'Descriptor-space cluster consistency; not independent human interpretability evidence.',
               'elapsed_seconds': round(time.monotonic() - started, 3)}
    im = summary['intra']['micro_pair_weighted']['visual_similarity']['mean']
    em = summary['inter']['micro_pair_weighted']['visual_similarity']['mean']
    summary['intra_minus_inter_visual_similarity'] = im - em if im is not None and em is not None else None
    write_csv(output / 'intra_cluster_similarity.csv', intra_rows, ('prototype_id', 'patch_count'))
    write_csv(output / 'inter_cluster_similarity.csv', inter_rows, ('prototype_a', 'prototype_b', 'patch_count_a', 'patch_count_b'))
    json_write(output / 'cluster_similarity_summary.json', summary)
    print('\nVisual similarity (patch-pair-weighted mean +/- population-style SD):')
    for name in ('intra', 'inter'):
        part = summary[name]
        score = part['micro_pair_weighted']['visual_similarity']
        formatted = 'undefined (no eligible pairs)' if score['mean'] is None else f'{score["mean"]:.6f} +/- {score["std"]:.6f}'
        print(f'  {name.capitalize():5s}: {formatted} [{part["mode"]}; {part["evaluated_pair_count"]:,}/{part["possible_pair_count"]:,} pairs]')
        macro = part['macro_equal_stratum_weight']['visual_similarity']
        if macro['mean'] is not None:
            print(f'         Equal-stratum mean {macro["mean"]:.6f}; SD across stratum means {macro["std"]:.6f}')
    if summary['intra_minus_inter_visual_similarity'] is not None:
        print(f'  Intra - inter mean: {summary["intra_minus_inter_visual_similarity"]:.6f}')
    print(f'Reports: {output}', flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cluster-dir', type=Path, default=DEFAULT_CLUSTER_DIR)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--intra-pairs', type=int, default=5000, help='Sample cap per individual cluster (minimum 2).')
    p.add_argument('--inter-pairs', type=int, default=500, help='Sample cap per unordered cluster pair (minimum 2).')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exact', action='store_true', help='Evaluate every eligible pair; time grows quadratically with patch count.')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.intra_pairs < 2 or args.inter_pairs < 2 or args.batch_size < 1 or args.seed < 0:
        p.error('Pair budgets must be >= 2, batch-size must be positive, and seed must be nonnegative.')
    try:
        analyze(args)
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
