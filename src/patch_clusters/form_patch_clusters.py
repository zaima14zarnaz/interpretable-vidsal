#!/usr/bin/env python3
"""Build fixed stage4 patch clusters for appearance supervision.

Two input modes:

  cache (default; recommended for training)
    Pair training RGB windows with FrozenBackboneFeatureCache stage4 last-slice
    features. No prototype activations, assignments, graph edges, or graph.sqlite.
    All eligible patches from the configured training-window sampling are used
    (no saliency / activation filter). Records verified frozen-backbone
    fingerprint + preprocessing provenance.

  graph (optional)
    Read complete visual_descriptors from a prototype-graph graph.sqlite
    (generate_prot_graph.py ACTION='cache_similarity'). Graph provenance is
    promoted when present; otherwise marked unverified.

Examples:
    python form_patch_clusters.py --output-dir /data/.../patch_cluster_frozen_ref
    python form_patch_clusters.py --source-mode cache --max-windows 64
    python form_patch_clusters.py --source-mode graph --source-db /path/to/graph.sqlite

Distance = wf*(1-feature_cosine) + wr*RGB_MAE + ws*(1-RGB_structure_cosine).
Structure is per-channel spatially centered RGB; both flat=1, one flat=0.
RGB bank vectors are HWC (converted to CHW inside concept_creation.py).

Clustering (--clustering):
  finch     (default) FINCH on a video-capped sample using the composite distance
            above. The number of clusters is NOT given; FINCH yields a hierarchy
            of partitions and one is chosen automatically (or via --finch-level).
  kmedoids  seeded approximate k-medoids with a fixed --num-prototypes.
Either way every patch is then assigned to the closest final medoid. Write only to a new
--output-dir; never overwrite the active bank or feature caches in place.

Output schema (concept_creation.py / view_patch_clusters.py):
    manifest.json, patch_clusters.sqlite, cluster_references.npz,
    fit_sample_indices.npy, descriptors/{feature,rgb,structure,flat,node_ids,prototype_ids}.npy
"""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

DEFAULT_SOURCE_DB = Path('/data/quantization/zaima/videosal_datasets/dhf1k/explanation_graphs/prototype_graph_outputs/graph.sqlite')
# Legacy graph-built bank. Cache mode defaults to a different path so the old bank is never overwritten.
DEFAULT_OUTPUT_DIR = Path('/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster')
DEFAULT_CACHE_OUTPUT_DIR = Path('/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster_frozen_ref')
DEFAULT_DATASET_ROOT = Path('/data/quantization/zaima/videosal_datasets/dhf1k')
DEFAULT_CACHE_ROOT = Path('/data/quantization/zaima/videosal_datasets/dhf1k/cached_feats')
# Same Kinetics VideoSwin pack as train.py / FrozenBackboneFeatureCache.
DEFAULT_REFERENCE_CHECKPOINT = Path(
    '/data/quantization/zaima/videoswin/swin_small_patch244_window877_kinetics400_1k.pth'
)
FORMAT_VERSION = 'fixed_stage4_patch_clusters_v1'
SRC_DIR = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def connect_readonly(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    return con


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unpack(blob, label):
    if blob is None:
        raise ValueError(f'{label}: visual descriptor missing; run ACTION="cache_similarity" in the source graph generator.')
    try:
        raw = zlib.decompress(blob)
        if len(raw) % 4:
            raise ValueError('Invalid float32 byte count')
        value = np.frombuffer(raw, dtype='<f4').astype(np.float64)
    except (zlib.error, ValueError) as exc:
        raise ValueError(f'{label}: corrupt compressed float32 vector.') from exc
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f'{label}: empty/nonfinite descriptor.')
    return value


def normalize_feature(value, label):
    scale = float(np.max(np.abs(value)))
    if not scale:
        raise ValueError(f'{label}: zero stage4 feature; all three metrics cannot be evaluated.')
    scaled = value / scale
    norm = float(np.linalg.norm(scaled))
    if norm * scale < 1e-12:
        raise ValueError(f'{label}: near-zero stage4 feature.')
    return (scaled / norm).astype(np.float32)


def structure_descriptor(rgb):
    # Source graph descriptors are HWC flattened. Float64 centering matches
    # its cosine calculation, including exactly constant channel colors.
    pixels = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    centered = (pixels - pixels.mean(axis=0, keepdims=True)).reshape(-1)
    norm = float(np.linalg.norm(centered))
    flat = norm < 1e-12
    return (np.zeros_like(centered) if flat else centered / norm).astype(np.float32), flat


class Descriptors:
    def __init__(self, directory, count=None, feature_dim=None, rgb_dim=None, mode='r'):
        directory = Path(directory)
        if mode == 'w+':
            directory.mkdir(parents=True, exist_ok=True)
        specs = {'feature': ((count, feature_dim), np.float32),
                 'rgb': ((count, rgb_dim), np.float32),
                 'structure': ((count, rgb_dim), np.float32),
                 'flat': ((count,), np.bool_), 'node_ids': ((count,), np.int64)}
        for name, (shape, dtype) in specs.items():
            value = (np.lib.format.open_memmap(directory / f'{name}.npy', mode=mode, shape=shape, dtype=dtype)
                     if mode == 'w+' else np.load(directory / f'{name}.npy', mmap_mode=mode, allow_pickle=False))
            setattr(self, name, value)
        self.count = len(self.node_ids)

    def take(self, indices):
        return tuple(np.asarray(getattr(self, name)[indices]) for name in ('feature', 'rgb', 'structure', 'flat'))

    def flush(self):
        for name in ('feature', 'rgb', 'structure', 'flat', 'node_ids'):
            getattr(self, name).flush()


def pair_metrics(a, b):
    fa, ra, sa, za = a
    fb, rb, sb, zb = b
    feature_cosine = np.clip(fa @ fb.T, -1., 1.).astype(np.float64)
    rgb_mae = cdist(ra, rb, metric='cityblock') / ra.shape[1]
    structure_cosine = np.clip(sa @ sb.T, -1., 1.).astype(np.float64)
    structure_cosine += np.asarray(za, dtype=np.float64)[:, None] * np.asarray(zb, dtype=np.float64)[None, :]
    return feature_cosine, rgb_mae, np.clip(structure_cosine, -1., 1.)


def pair_distance(a, b, weights):
    fc, mae, sc = pair_metrics(a, b)
    return np.maximum(0., weights[0] * (1. - fc) + weights[1] * mae + weights[2] * (1. - sc))


def assign(data, rows, medoids, weights, batch_size):
    """Closest-medoid assignments with deterministic ties and self ownership."""
    rows = np.asarray(rows, dtype=np.int64)
    refs = data.take(medoids)
    labels = np.empty(len(rows), dtype=np.int32)
    distances = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        ids = rows[start:start + batch_size]
        costs = pair_distance(data.take(ids), refs, weights)
        chosen = costs.argmin(axis=1)
        # Each distinct medoid owns itself. This only resolves zero-distance
        # ties; it prevents identical input patches from emptying K-1 clusters.
        for j, medoid in enumerate(medoids):
            own = ids == medoid
            chosen[own] = j
            costs[own, j] = 0.
        labels[start:start + len(ids)] = chosen
        distances[start:start + len(ids)] = costs[np.arange(len(ids)), chosen]
    return labels, distances


def choose_fit_sample(con, size, per_video, rng):
    """Reservoir sampling per video, then cap the combined discovery sample."""
    reservoirs, seen = {}, {}
    sql = '''SELECT p.descriptor_index,f.dataset,f.video FROM patches p
             JOIN frames f ON f.id=p.frame_id ORDER BY p.descriptor_index'''
    for row in con.execute(sql):
        key = (row['dataset'], row['video'])
        count = seen.get(key, 0) + 1
        seen[key] = count
        bucket = reservoirs.setdefault(key, [])
        if per_video == 0 or len(bucket) < per_video:
            bucket.append(int(row['descriptor_index']))
        else:
            position = int(rng.integers(count))
            if position < per_video:
                bucket[position] = int(row['descriptor_index'])
    pool = np.array([i for key in sorted(reservoirs) for i in reservoirs[key]], dtype=np.int64)
    if len(pool) > size:
        pool = rng.choice(pool, size=size, replace=False)
    return np.sort(pool), len(seen)


def initialize_medoids(data, sample, k, weights, rng, batch_size):
    # Distance-weighted seeding for this composite distance, not an assumption
    # that Euclidean k-means minimizes a cosine+MAE objective.
    first = int(rng.choice(sample))
    medoids = [first]
    closest = np.full(len(sample), np.inf)
    for j in range(k):
        for start in range(0, len(sample), batch_size):
            ids = sample[start:start + batch_size]
            d = pair_distance(data.take(ids), data.take([medoids[-1]]), weights)[:, 0]
            closest[start:start + len(ids)] = np.minimum(closest[start:start + len(ids)], d)
        if j + 1 == k:
            break
        available = ~np.isin(sample, medoids)
        probabilities = np.where(available, closest, 0.)
        total = float(probabilities.sum())
        next_id = (int(rng.choice(sample, p=probabilities / total)) if total > 1e-12
                   else int(rng.choice(sample[available])))
        medoids.append(next_id)
    return np.array(medoids, dtype=np.int64)


def fit_medoids(data, sample, args, weights):
    best = None
    history = []
    for restart in range(args.restarts):
        rng = np.random.default_rng(args.seed + 1009 * (restart + 1))
        medoids = initialize_medoids(data, sample, args.num_prototypes, weights, rng, args.batch_size)
        for iteration in range(args.iterations + 1):
            labels, distances = assign(data, sample, medoids, weights, args.batch_size)
            objective = float(distances.mean())
            history.append({'restart': restart, 'iteration': iteration, 'mean_distance': objective})
            print(f'Fit {restart + 1}/{args.restarts}, iteration {iteration}: mean distance {objective:.6f}', flush=True)
            if iteration == args.iterations:
                break
            updated = medoids.copy()
            for j in range(args.num_prototypes):
                members = sample[labels == j]
                if not len(members):
                    raise RuntimeError('Internal error: medoid self-ownership failed.')
                other = members[members != medoids[j]]
                if len(other) > args.medoid_candidates - 1:
                    other = rng.choice(other, size=args.medoid_candidates - 1, replace=False)
                candidates = np.sort(np.append(other, medoids[j]))
                costs = np.zeros(len(candidates), dtype=np.float64)
                for start in range(0, len(members), args.batch_size):
                    costs += pair_distance(data.take(candidates), data.take(members[start:start + args.batch_size]), weights).sum(axis=1)
                # Keeping the current medoid among candidates makes each
                # update non-worsening up to floating-point roundoff.
                updated[j] = candidates[int(costs.argmin())]
            if np.array_equal(updated, medoids):
                break
            medoids = updated
        # Objective above always corresponds to the current medoids.
        key = (objective, tuple(sorted(int(x) for x in data.node_ids[medoids])))
        if best is None or key < best[0]:
            best = (key, medoids.copy())
    medoids = best[1]
    # Stable vocabulary ordering: ascending source node ID of each medoid.
    medoids = medoids[np.argsort(data.node_ids[medoids], kind='stable')]
    return medoids, history, best[0][0]


def first_neighbors(data, reps, weights, batch_size):
    """Exact first neighbour (self excluded) of every representative under the composite distance."""
    refs = data.take(reps)
    nn = np.empty(len(reps), dtype=np.int64)
    for start in range(0, len(reps), batch_size):
        stop = min(start + batch_size, len(reps))
        costs = pair_distance(tuple(a[start:stop] for a in refs), refs, weights)
        costs[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nn[start:stop] = costs.argmin(axis=1)
    return nn


def cluster_representative(data, members, weights, rng, candidates, cap):
    """Real patch (descriptor index) of a cluster with the lowest summed distance to its members.

    Exact for small clusters; large clusters use sampled candidates / sampled members.
    """
    if len(members) == 1:
        return int(members[0])
    cand = members if len(members) <= candidates else rng.choice(members, size=candidates, replace=False)
    pool = members if len(members) <= cap else rng.choice(members, size=cap, replace=False)
    costs = pair_distance(data.take(cand), data.take(pool), weights).sum(axis=1)
    return int(cand[int(costs.argmin())])


def finch_partitions(data, sample, weights, args, rng):
    """FINCH hierarchy (Sarfraz et al., CVPR 2019) with the composite distance.

    Level 0 links every sample patch to its first (nearest) neighbour; connected
    components of that graph are the first partition. Each later level repeats this
    on one representative patch per cluster. Representatives are real medoid patches
    (the composite distance has no meaningful mean), not feature means as in the
    original FINCH. Returns partitions finest to coarsest.
    """
    labels = np.arange(len(sample), dtype=np.int32)
    reps = np.asarray(sample, dtype=np.int64).copy()
    partitions = []
    while len(reps) > 1:
        m = len(reps)
        nn = first_neighbors(data, reps, weights, args.batch_size)
        # Edge i - nn[i]; components of this graph equal FINCH's adjacency
        # (i->j, j->i, or shared first neighbour) components.
        graph = coo_matrix((np.ones(m, dtype=np.int8), (np.arange(m), nn)), shape=(m, m))
        k, comp = connected_components(graph, directed=False)
        labels = comp[labels].astype(np.int32)
        order = np.argsort(labels, kind='stable')
        bounds = np.flatnonzero(np.diff(labels[order])) + 1
        groups = np.split(np.asarray(sample)[order], bounds)  # groups[j] <-> label j
        reps = np.array([cluster_representative(data, g, weights, rng, args.medoid_candidates, 512)
                         for g in groups], dtype=np.int64)
        partitions.append({'k': int(k), 'labels': labels.copy(), 'reps': reps.copy()})
        print(f'FINCH level {len(partitions) - 1}: {k:,} clusters', flush=True)
        if k >= m:  # defensive; cannot happen for m >= 2
            break
    return partitions


def score_partition(data, sample, labels, reps, weights, batch_size):
    """Mean distance to own representative and a medoid-based simplified silhouette."""
    refs = data.take(reps)
    own_sum, sil_sum = 0., 0.
    for start in range(0, len(sample), batch_size):
        stop = min(start + batch_size, len(sample))
        costs = pair_distance(data.take(sample[start:stop]), refs, weights)
        pos = np.arange(stop - start)
        own = costs[pos, labels[start:stop]].copy()
        costs[pos, labels[start:stop]] = np.inf
        other = costs.min(axis=1)
        denom = np.maximum(own, other)
        sil_sum += float(np.where(denom > 1e-12, (other - own) / np.where(denom > 1e-12, denom, 1.), 0.).sum())
        own_sum += float(own.sum())
    n = len(sample)
    return own_sum / n, sil_sum / n


def fit_finch(data, sample, args, weights):
    """Run FINCH and pick one partition. Returns (medoids, history, mean_distance, info)."""
    rng = np.random.default_rng(args.seed + 1009)
    parts = finch_partitions(data, sample, weights, args, rng)
    if not parts:
        raise ValueError('FINCH produced no partitions (fit sample too small).')
    history = []
    for level, p in enumerate(parts):
        row = {'level': level, 'num_clusters': p['k'], 'mean_distance': None, 'silhouette': None}
        if p['k'] >= 2:
            row['mean_distance'], row['silhouette'] = score_partition(
                data, sample, p['labels'], p['reps'], weights, args.batch_size)
        history.append(row)
    max_k = args.max_clusters if args.max_clusters else math.inf
    if args.finch_level is not None:
        if not 0 <= args.finch_level < len(parts):
            raise ValueError(f'--finch-level must be in [0, {len(parts) - 1}] (0 = finest); '
                             f'cluster counts per level: {[p["k"] for p in parts]}')
        chosen, rule = args.finch_level, 'explicit_level'
    else:
        eligible = [r for r in history if r['silhouette'] is not None and args.min_clusters <= r['num_clusters'] <= max_k]
        if not eligible:
            raise ValueError(f'No FINCH level has {args.min_clusters}..{args.max_clusters or "inf"} clusters; '
                             f'counts per level: {[p["k"] for p in parts]}. Adjust --min/--max-clusters or --finch-level.')
        # Highest silhouette; ties go to fewer clusters.
        chosen = max(eligible, key=lambda r: (round(r['silhouette'], 9), -r['num_clusters']))['level']
        rule = 'max_simplified_silhouette_within_bounds'
    for row in history:
        row['selected'] = row['level'] == chosen
        print('FINCH level {level}: k={num_clusters:,} mean_dist={mean_distance} silhouette={silhouette}{mark}'.format(
            mark='  <== selected' if row['selected'] else '', **row), flush=True)
    medoids = parts[chosen]['reps']
    medoids = medoids[np.argsort(data.node_ids[medoids], kind='stable')]  # stable vocabulary order
    mean_distance = history[chosen]['mean_distance']
    info = {'selection_rule': rule, 'selected_level': chosen, 'num_clusters': int(len(medoids)),
            'min_clusters': args.min_clusters, 'max_clusters': args.max_clusters,
            'cluster_counts_per_level': [p['k'] for p in parts]}
    return medoids, history, (0. if mean_distance is None else float(mean_distance)), info


def create_schema(con):
    con.executescript('''
    PRAGMA foreign_keys=ON;
    CREATE TABLE frames (
      id INTEGER PRIMARY KEY,dataset TEXT NOT NULL,video TEXT NOT NULL,
      dataset_index INTEGER NOT NULL,window_start INTEGER NOT NULL,
      target_index INTEGER NOT NULL,target_name TEXT NOT NULL,
      source_path TEXT NOT NULL,thumb_path TEXT,input_h INTEGER NOT NULL,input_w INTEGER NOT NULL);
    CREATE TABLE patches (
      descriptor_index INTEGER PRIMARY KEY,node_id INTEGER NOT NULL UNIQUE,
      frame_id INTEGER NOT NULL REFERENCES frames(id),stage TEXT NOT NULL,
      row INTEGER NOT NULL,col INTEGER NOT NULL,grid_h INTEGER NOT NULL,grid_w INTEGER NOT NULL,
      pred_score REAL,pred_percentile REAL,gt_score REAL,gt_percentile REAL,
      prototype_id INTEGER,distance REAL,feature_cosine REAL,rgb_mae REAL,structure_cosine REAL);
    CREATE TABLE prototypes (
      id INTEGER PRIMARY KEY,medoid_index INTEGER NOT NULL,medoid_node_id INTEGER NOT NULL,
      patch_count INTEGER NOT NULL,video_count INTEGER NOT NULL,mean_distance REAL,
      mean_saliency_percentile REAL,std_saliency_percentile REAL);
    CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE INDEX patches_frame ON patches(frame_id);
    ''')


def read_source_info(source, args):
    for table in ('nodes', 'frames', 'visual_descriptors'):
        if not source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            raise ValueError(f'Source DB lacks {table}; use graph.sqlite with a complete visual descriptor cache.')
    labels = [row[0] for row in source.execute('SELECT DISTINCT dataset FROM frames ORDER BY dataset')]
    datasets = labels if args.datasets == ['ALL'] else args.datasets
    missing = set(datasets) - set(labels)
    if missing:
        raise ValueError(f'Datasets {sorted(missing)} not found. Available: {labels}. Select --datasets explicitly; default is train.')
    marks = ','.join('?' for _ in datasets)
    where = f"n.stage='stage4' AND f.dataset IN ({marks})"
    count = source.execute(f'SELECT COUNT(*) FROM nodes n JOIN frames f ON f.id=n.frame_id WHERE {where}', datasets).fetchone()[0]
    if count < args.num_prototypes:
        raise ValueError(f'{count} selected stage4 patches cannot form {args.num_prototypes} nonempty clusters.')
    columns = '''n.*,f.dataset,f.video,f.dataset_index,f.window_start,f.target_index,
                 f.target_name,f.source_path,f.thumb_path,f.input_h,f.input_w,
                 d.feature_dim,d.stage4_vector,d.rgb_vector'''
    query = f'''SELECT {columns} FROM nodes n JOIN frames f ON f.id=n.frame_id
                LEFT JOIN visual_descriptors d ON d.node_id=n.id WHERE {where} ORDER BY n.id'''
    settings = {}
    if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='similarity_metadata'").fetchone():
        row = source.execute("SELECT value FROM similarity_metadata WHERE key='settings'").fetchone()
        if row:
            settings = json.loads(row[0])
    return count, query, datasets, settings


def extract(source, destination, directory, count, query, datasets, source_directory):
    cursor = source.execute(query, datasets)
    data = None
    copied_frames = set()
    digest = hashlib.sha256()
    feature_dim = rgb_dim = None
    started = time.monotonic()
    for index, row in enumerate(cursor):
        node_id = int(row['id'])
        feature = unpack(row['stage4_vector'], f'Node {node_id} stage4')
        rgb = unpack(row['rgb_vector'], f'Node {node_id} RGB')
        if len(feature) != row['feature_dim']:
            raise ValueError(f'Node {node_id}: cached feature_dim disagrees with vector.')
        side = math.isqrt(len(rgb) // 3)
        if len(rgb) != side * side * 3 or side < 1 or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError(f'Node {node_id}: expected square HWC RGB descriptor in [0,1].')
        if data is None:
            feature_dim, rgb_dim = len(feature), len(rgb)
            data = Descriptors(directory / 'descriptors', count, feature_dim, rgb_dim, mode='w+')
        if (len(feature), len(rgb)) != (feature_dim, rgb_dim):
            raise ValueError(f'Node {node_id}: mixed descriptor dimensions in source cache.')
        if not (0 <= row['row'] < row['grid_h'] and 0 <= row['col'] < row['grid_w']):
            raise ValueError(f'Node {node_id}: invalid patch grid coordinates.')
        data.feature[index] = normalize_feature(feature, f'Node {node_id}')
        data.rgb[index] = rgb
        data.structure[index], data.flat[index] = structure_descriptor(rgb)
        data.node_ids[index] = node_id
        if row['frame_id'] not in copied_frames:
            thumbnail = row['thumb_path']
            if thumbnail:
                thumbnail = str((source_directory / thumbnail).resolve())
            destination.execute('INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (row['frame_id'], row['dataset'], row['video'], row['dataset_index'], row['window_start'],
                 row['target_index'], row['target_name'], row['source_path'], thumbnail, row['input_h'], row['input_w']))
            copied_frames.add(row['frame_id'])
        destination.execute('''INSERT INTO patches(descriptor_index,node_id,frame_id,stage,row,col,grid_h,grid_w,
                               pred_score,pred_percentile,gt_score,gt_percentile) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (index, node_id, row['frame_id'], row['stage'], row['row'], row['col'], row['grid_h'], row['grid_w'],
             row['pred_score'], row['pred_percentile'], row['gt_score'], row['gt_percentile']))
        identity = [node_id, row['dataset'], row['video'], row['dataset_index'], row['window_start'], row['target_index'],
                    row['stage'], row['row'], row['col'], row['grid_h'], row['grid_w']]
        digest.update(json.dumps(identity, separators=(',', ':')).encode())
        digest.update(row['stage4_vector'])
        digest.update(row['rgb_vector'])
        if (index + 1) % 5000 == 0:
            destination.commit()
            print(f'Descriptors {index + 1:,}/{count:,} ({time.monotonic() - started:.1f}s)', flush=True)
    if data is None or index + 1 != count:
        raise RuntimeError('Source row count changed or no usable descriptors were found.')
    destination.commit()
    data.flush()
    return data, digest.hexdigest(), len(copied_frames)


def save_assignments(con, data, directory, medoids, args, weights):
    labels = np.lib.format.open_memmap(directory / 'descriptors/prototype_ids.npy', mode='w+', dtype=np.int32, shape=(data.count,))
    distance_array = np.lib.format.open_memmap(directory / 'descriptors/distances.npy', mode='w+', dtype=np.float32, shape=(data.count,))
    refs = data.take(medoids)
    totals = np.zeros(args.num_prototypes, dtype=np.int64)
    for start in range(0, data.count, args.batch_size):
        rows = np.arange(start, min(data.count, start + args.batch_size), dtype=np.int64)
        fc, mae, sc = pair_metrics(data.take(rows), refs)
        costs = np.maximum(0., weights[0] * (1 - fc) + weights[1] * mae + weights[2] * (1 - sc))
        group = costs.argmin(axis=1)
        for j, medoid in enumerate(medoids):
            own = rows == medoid
            group[own] = j
            fc[own, j], mae[own, j], sc[own, j], costs[own, j] = 1., 0., 1., 0.
        pos = np.arange(len(rows))
        distance = costs[pos, group]
        labels[rows] = group
        distance_array[rows] = distance
        totals += np.bincount(group, minlength=args.num_prototypes)
        records = [(int(g), float(d), float(f), float(m), float(s), int(i)) for g, d, f, m, s, i in
                   zip(group, distance, fc[pos, group], mae[pos, group], sc[pos, group], rows)]
        con.executemany('''UPDATE patches SET prototype_id=?,distance=?,feature_cosine=?,rgb_mae=?,structure_cosine=?
                           WHERE descriptor_index=?''', records)
        if start == 0 or start + len(rows) == data.count or start // args.batch_size % 50 == 0:
            con.commit()
            print(f'Assigned {start + len(rows):,}/{data.count:,} patches', flush=True)
    con.executescript('''CREATE INDEX patches_cluster_distance ON patches(prototype_id,distance,node_id);
                         CREATE INDEX patches_cluster_node ON patches(prototype_id,node_id);''')
    if (totals == 0).any():
        raise RuntimeError('Internal error: empty final reference cluster.')
    stats = []
    for j, medoid in enumerate(medoids):
        row = con.execute('''SELECT COUNT(*) AS n,AVG(distance) AS distance,
                 AVG(pred_percentile) AS mean,AVG(pred_percentile*pred_percentile) AS square
                 FROM patches WHERE prototype_id=?''', (j,)).fetchone()
        videos = con.execute('''SELECT COUNT(*) FROM (SELECT DISTINCT f.dataset,f.video FROM patches p
                 JOIN frames f ON f.id=p.frame_id WHERE p.prototype_id=?)''', (j,)).fetchone()[0]
        mean = row['mean']
        std = math.sqrt(max(0., row['square'] - mean * mean)) if mean is not None else None
        item = (j, int(medoid), int(data.node_ids[medoid]), int(row['n']), videos, row['distance'], mean, std)
        con.execute('INSERT INTO prototypes VALUES (?,?,?,?,?,?,?,?)', item)
        stats.append({'prototype_id': j, 'medoid_node_id': item[2], 'patch_count': item[3],
                      'video_count': videos, 'mean_distance': row['distance'],
                      'mean_saliency_percentile': mean, 'std_saliency_percentile': std})
    con.commit()
    labels.flush()
    distance_array.flush()
    np.savez_compressed(directory / 'cluster_references.npz', prototype_ids=np.arange(len(medoids), dtype=np.int32),
                        descriptor_indices=medoids, medoid_node_ids=np.asarray(data.node_ids[medoids]),
                        feature=refs[0], rgb=refs[1], structure=refs[2], flat=refs[3], weights=np.asarray(weights),
                        format_version=np.array(FORMAT_VERSION), rgb_order=np.array('HWC'))
    return stats


def prepare_output_directory(directory: Path, overwrite: bool) -> Path:
    directory = Path(directory).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f'{directory} is not empty. Use a new --output-dir (recommended) '
                'or --overwrite to rename the existing folder to a timestamped backup.'
            )
        suffix = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        backup = directory.with_name(directory.name + '.backup-' + suffix)
        directory.rename(backup)
        print(f'Previous output preserved at {backup}', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def base_manifest(args, weights, *, source_mode: str) -> dict:
    return {
        'format_version': FORMAT_VERSION,
        'status': 'building',
        'created_at': now(),
        'source_mode': source_mode,
        'stage': 'stage4',
        'prototype_ids': 'new fixed reference IDs, zero-based; unrelated to old activation IDs',
        'num_prototypes': args.num_prototypes,
        'seed': args.seed,
        'distance': 'wf*(1-feature_cosine) + wr*rgb_mae + ws*(1-centered_rgb_structure_cosine)',
        'weights': {'feature': weights[0], 'rgb_mae': weights[1], 'structure': weights[2]},
        'structure_flat_policy': 'both flat=1, one flat=0',
        'rgb_order': 'HWC',
        'reference_type': 'real medoid patches',
        'reference_groups_fixed': True,
        'algorithm': 'sampled_candidate_k_medoids_then_full_closest_medoid_assignment',
        'fit_sample_size_requested': args.fit_sample_size,
        'fit_max_per_video': args.fit_max_per_video,
        'iterations': args.iterations,
        'restarts': args.restarts,
        'medoid_candidates': args.medoid_candidates,
        'batch_size': args.batch_size,
        'saliency_used_for_clustering': False,
        'model_assignments_used_for_clustering': False,
        'all_selected_stage4_patches_included': True,
    }


def finalize_fit(destination, directory, data, manifest, args, weights, started):
    sample, videos = choose_fit_sample(
        destination, args.fit_sample_size, args.fit_max_per_video, np.random.default_rng(args.seed)
    )
    if len(sample) < args.num_prototypes:
        raise ValueError(
            'Video-capped fitting sample is smaller than --num-prototypes; '
            'increase --fit-max-per-video (0 disables cap).'
        )
    np.save(directory / 'fit_sample_indices.npy', sample, allow_pickle=False)
    print(
        f'{data.count:,} patches, {videos:,} videos; fitting on {len(sample):,} patches.',
        flush=True,
    )
    if args.clustering == 'finch':
        medoids, fit_history, objective, finch_info = fit_finch(data, sample, args, weights)
        args.num_prototypes = int(len(medoids))  # cluster count is an output of FINCH
        manifest.update(
            algorithm='sampled_finch_composite_distance_medoid_representatives_then_full_closest_medoid_assignment',
            num_prototypes=args.num_prototypes,
            finch=finch_info,
        )
        for stale in ('iterations', 'restarts'):
            manifest.pop(stale, None)
    else:
        medoids, fit_history, objective = fit_medoids(data, sample, args, weights)
    stats = save_assignments(destination, data, directory, medoids, args, weights)
    remaining = destination.execute('SELECT COUNT(*) FROM patches WHERE prototype_id IS NULL').fetchone()[0]
    if remaining or destination.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
        raise RuntimeError('Final assignment/database integrity check failed.')
    manifest.update(
        status='complete',
        completed_at=now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
        fit_sample_size=len(sample),
        num_videos=videos,
        fit_mean_distance=objective,
        fit_history=fit_history,
        clusters=stats,
        database='patch_clusters.sqlite',
        references='cluster_references.npz',
        descriptor_arrays='descriptors/*.npy, indexed by patches.descriptor_index',
    )
    manifest['references_sha256'] = sha256_file(directory / 'cluster_references.npz')
    destination.execute(
        'INSERT INTO metadata VALUES (?,?)',
        ('manifest', json.dumps(manifest, allow_nan=False)),
    )
    destination.execute('ANALYZE')
    destination.commit()
    return manifest


def rgb_patch_hwc(rgb_hwc: np.ndarray, row: int, col: int, grid_h: int, grid_w: int, side: int) -> np.ndarray:
    """Uniform-grid crop + PIL bilinear resize → flat HWC float32 in [0,1]."""
    from PIL import Image

    a = np.asarray(rgb_hwc, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all() or a.min() < -0.01 or a.max() > 1.01:
        raise ValueError('Expected finite HWC RGB in [0,1].')
    h, w = a.shape[:2]
    crop = a[
        round(row * h / grid_h): round((row + 1) * h / grid_h),
        round(col * w / grid_w): round((col + 1) * w / grid_w),
    ].clip(0, 1)
    if not crop.size:
        raise ValueError('Empty RGB patch under the reference grid mapping.')
    size = (int(side), int(side))
    return np.stack(
        [
            np.asarray(Image.fromarray(crop[..., c]).resize(size, Image.Resampling.BILINEAR))
            for c in range(3)
        ],
        axis=-1,
    ).clip(0, 1).astype(np.float32).reshape(-1)


def resize_last_rgb_canvas(rgb_tchw, resize_to):
    """Unnormalized last-frame RGB on the backbone spatial canvas (no ImageNet norm)."""
    import torch
    import torch.nn.functional as F

    clip = rgb_tchw.detach().float()
    if clip.dim() != 4 or clip.shape[1] != 3:
        raise ValueError(f'Expected [T,3,H,W] RGB, got {tuple(clip.shape)}')
    if clip.numel() > 0 and float(clip.max()) > 2.0:
        clip = clip / 255.0
    last = clip[-1:].clamp(0.0, 1.0)  # [1,3,H,W]
    if isinstance(resize_to, int):
        target = (int(resize_to), int(resize_to))
    else:
        target = (int(resize_to[0]), int(resize_to[1]))
    if tuple(last.shape[-2:]) != target:
        last = F.interpolate(last, size=target, mode='bilinear', align_corners=False)
    return last[0].cpu()  # [3,H,W]


def extract_from_training_cache(args, directory, destination):
    """Enumerate training windows; pair cache stage4 features with RGB patches."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import torch
    from model.backbone_feature_cache import (
        DEFAULT_RESIZE_TO,
        FrozenBackboneFeatureCache,
        compare_reference_feature_provenance,
        make_identity_from_loader_sample,
        provenance_from_cache,
    )
    from pre_process.dataloader import DatasetLoader

    split = str(args.split)
    if split != 'train':
        raise ValueError('Cache mode uses training data only; set --split train.')
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f'Training split not found: {split_dir}')

    resize_to = tuple(int(x) for x in args.resize_to)
    if len(resize_to) != 2 or min(resize_to) < 1:
        raise ValueError('--resize-to must be two positive integers: HEIGHT WIDTH')
    rgb_side = int(args.rgb_size)
    if rgb_side < 1:
        raise ValueError('--rgb-size must be positive.')

    ref_ckpt = str(Path(args.reference_checkpoint).expanduser().resolve())
    if not Path(ref_ckpt).is_file():
        raise FileNotFoundError(f'Reference checkpoint not found: {ref_ckpt}')
    allow_compute = bool(args.allow_compute) and not bool(getattr(args, 'cache_read_only', False))
    cache_root = Path(args.cache_root).expanduser().resolve()
    cache = FrozenBackboneFeatureCache(
        root=cache_root,
        checkpoint=ref_ckpt,
        device=args.device,
        storage_dtype=args.storage_dtype,
        resize_to=resize_to,
        normalize=True,
    )
    feature_provenance = provenance_from_cache(cache)
    if not feature_provenance.get('backbone_fingerprint'):
        raise RuntimeError('FrozenBackboneFeatureCache did not produce a backbone_fingerprint.')
    # Self-check: helpers used by train.py accept this provenance object.
    compare_reference_feature_provenance(feature_provenance, feature_provenance)

    dataset = DatasetLoader(
        str(split_dir),
        window_len=int(args.window_len),
        stride=int(args.stride),
        random_train_sampling=False,
    )
    total_windows = len(dataset)
    limit = total_windows if args.max_windows is None else min(total_windows, int(args.max_windows))
    if limit < 1:
        raise ValueError('No training windows available under the configured sampling.')

    print(
        f'Cache mode | split={split_dir} | windows={limit:,}/{total_windows:,} | '
        f'window_len={args.window_len} | stride={args.stride} | '
        f'num_prototypes={args.num_prototypes} | '
        f'cache={cache.root} | fingerprint={cache.reference_fingerprint[:16]}… | '
        f'allow_compute={allow_compute} (reuse compatible entries; never overwrite others)',
        flush=True,
    )

    # Probe first window for stage4 grid / channel count, then allocate.
    video0, start0 = dataset.windows[0]
    frames0 = dataset.video_frame_names[video0][start0: start0 + args.window_len]
    sample0 = dataset[0]
    _, rgb_np0, _, _, n_frames0 = sample0[:5]
    identity0, rgb0 = make_identity_from_loader_sample(
        dataset_root=str(split_dir),
        video_id=video0,
        window_start=start0,
        frame_names=frames0,
        rgb_thwc_or_tchw=torch.as_tensor(rgb_np0),
        split=split,
        stride=int(args.stride),
        view_id='identity',
        preprocess=cache.preprocess,
    )
    feats0 = cache.get_or_compute(
        rgb0, identity=identity0, force=False, allow_compute=allow_compute
    )
    stage4_0 = feats0['stage4']
    if stage4_0.dim() != 4 or stage4_0.shape[1] != 1:
        raise ValueError(f'Expected stage4 [C,1,H,W], got {tuple(stage4_0.shape)}')
    feature_dim = int(stage4_0.shape[0])
    grid_h, grid_w = int(stage4_0.shape[2]), int(stage4_0.shape[3])
    patches_per_frame = grid_h * grid_w
    count = limit * patches_per_frame
    if count < args.num_prototypes:
        raise ValueError(
            f'{count} stage4 patches from {limit} windows cannot form {args.num_prototypes} clusters.'
        )
    rgb_dim = 3 * rgb_side * rgb_side
    data = Descriptors(directory / 'descriptors', count, feature_dim, rgb_dim, mode='w+')
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                'resize_to': list(resize_to),
                'rgb_size': rgb_side,
                'fingerprint': cache.reference_fingerprint,
                'window_len': int(args.window_len),
                'stride': int(args.stride),
            },
            separators=(',', ':'),
            sort_keys=True,
        ).encode()
    )

    index = 0
    frame_id = 0
    started = time.monotonic()
    for window_i in range(limit):
        video_id, start = dataset.windows[window_i]
        frame_names = dataset.video_frame_names[video_id][start: start + args.window_len]
        sample = dataset[window_i]
        sample_video, rgb_np, _, _, n_frames = sample[:5]
        if sample_video != video_id or int(n_frames) != int(args.window_len):
            raise RuntimeError(f'Loader identity mismatch at window {window_i}.')
        identity, rgb = make_identity_from_loader_sample(
            dataset_root=str(split_dir),
            video_id=video_id,
            window_start=start,
            frame_names=frame_names,
            rgb_thwc_or_tchw=torch.as_tensor(rgb_np),
            split=split,
            stride=int(args.stride),
            view_id='identity',
            preprocess=cache.preprocess,
        )
        features = cache.get_or_compute(
            rgb, identity=identity, force=False, allow_compute=allow_compute
        )
        stage4 = features['stage4']
        if tuple(stage4.shape) != (feature_dim, 1, grid_h, grid_w):
            raise ValueError(
                f'stage4 shape changed at {video_id}@{start}: '
                f'{tuple(stage4.shape)} vs {(feature_dim, 1, grid_h, grid_w)}'
            )
        # [H,W,C] feature layout matching concept / graph row-major order.
        feat_bhwc = stage4[:, 0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
        last_rgb = resize_last_rgb_canvas(rgb, resize_to)  # [3,H,W] on backbone canvas
        rgb_hwc = last_rgb.permute(1, 2, 0).contiguous().numpy().astype(np.float32)
        input_h, input_w = int(rgb_hwc.shape[0]), int(rgb_hwc.shape[1])
        target_index = int(start) + int(args.window_len) - 1
        target_name = str(frame_names[-1])
        source_path = str((split_dir / video_id / 'images' / target_name).resolve())
        destination.execute(
            'INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (
                frame_id, split, str(video_id), int(window_i), int(start),
                target_index, target_name, source_path, None, input_h, input_w,
            ),
        )
        for row in range(grid_h):
            for col in range(grid_w):
                node_id = index  # cache mode: stable sequential IDs
                feature = feat_bhwc[row, col].reshape(-1)
                rgb_vec = rgb_patch_hwc(rgb_hwc, row, col, grid_h, grid_w, rgb_side)
                data.feature[index] = normalize_feature(feature, f'node {node_id}')
                data.rgb[index] = rgb_vec
                data.structure[index], data.flat[index] = structure_descriptor(rgb_vec)
                data.node_ids[index] = node_id
                destination.execute(
                    '''INSERT INTO patches(descriptor_index,node_id,frame_id,stage,row,col,grid_h,grid_w,
                       pred_score,pred_percentile,gt_score,gt_percentile)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (index, node_id, frame_id, 'stage4', row, col, grid_h, grid_w,
                     None, None, None, None),
                )
                identity_payload = [
                    node_id, split, video_id, window_i, start, target_index,
                    'stage4', row, col, grid_h, grid_w, identity.content_sha256,
                ]
                digest.update(json.dumps(identity_payload, separators=(',', ':')).encode())
                digest.update(feature.astype('<f4').tobytes())
                digest.update(rgb_vec.astype('<f4').tobytes())
                index += 1
        frame_id += 1
        if (window_i + 1) % 50 == 0 or (window_i + 1) == limit:
            destination.commit()
            print(
                f'Descriptors {index:,}/{count:,} from {window_i + 1:,}/{limit:,} windows '
                f'({time.monotonic() - started:.1f}s)',
                flush=True,
            )

    if index != count:
        raise RuntimeError(f'Expected {count} patches, wrote {index}.')
    destination.commit()
    data.flush()
    rgb_settings = {
        'version': 'stage4_rgb_structure_v1',
        'formula': 'stage4_cosine + (1 - rgb_mae) + rgb_structure_cosine',
        'rgb_size': rgb_side,
        'rgb_range': [0, 1],
        'rgb_resize': 'bilinear, per-channel float32, uniform input-grid crop',
        'rgb_canvas': f'backbone_resize_to_{list(resize_to)}_unnormalized',
        'structure': 'cosine of per-channel spatially centered RGB',
        'feature_module': 'frozen_reference_backbone',
        'feature_output_path': ['stage4'],
        'feature_layout': 'BCTHW_last_slice_to_BHWC',
        'descriptor_feature_source': 'frozen_reference',
    }
    meta = {
        'data': data,
        'fingerprint': digest.hexdigest(),
        'frames': frame_id,
        'count': count,
        'feature_dim': feature_dim,
        'rgb_descriptor_size': rgb_side,
        'rgb_settings': rgb_settings,
        'feature_provenance': feature_provenance,
        'datasets': [split],
        'grid_h': grid_h,
        'grid_w': grid_w,
        'cache_root': str(cache.root),
        'reference_checkpoint': ref_ckpt,
        'window_len': int(args.window_len),
        'stride': int(args.stride),
        'dataset_root': str(dataset_root),
        'default_resize_to': list(DEFAULT_RESIZE_TO) if not isinstance(DEFAULT_RESIZE_TO, int) else DEFAULT_RESIZE_TO,
    }
    return meta


def build_from_graph(args, directory, weights):
    source_path = Path(args.source_db).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f'Source graph DB not found: {source_path}. Supply --source-db /path/to/graph.sqlite '
            'or use --source-mode cache.'
        )
    if source_path.is_relative_to(directory):
        raise ValueError('Output directory must not contain the source database.')
    manifest_path = source_path.parent / 'manifest.json'
    source_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if source_manifest.get('status', 'complete') != 'complete':
        raise ValueError('Source graph extraction is not complete; finish or repair it before defining fixed references.')
    feature_provenance = None
    if isinstance(source_manifest, dict):
        feature_provenance = source_manifest.get('reference_feature_provenance')
        if not isinstance(feature_provenance, dict) or not feature_provenance.get('backbone_fingerprint'):
            feature_provenance = None
    provenance_status = 'verified' if feature_provenance else 'unverified_legacy_source'
    started = time.monotonic()
    manifest = base_manifest(args, weights, source_mode='graph')
    manifest.update(
        source_database=str(source_path),
        source_manifest=source_manifest,
        source_manifest_sha256=sha256_file(manifest_path) if manifest_path.is_file() else None,
        source_feature_semantics=(
            'fixed snapshot of stored raw stage4 vectors; '
            'backbone identity is reference_feature_provenance.backbone_fingerprint '
            '(not source_checkpoint_sha256)'
        ),
        reference_feature_provenance=feature_provenance,
        reference_feature_provenance_status=provenance_status,
    )
    if feature_provenance is None:
        print(
            'WARNING: source graph lacks reference_feature_provenance.backbone_fingerprint. '
            'Prefer --source-mode cache for verified frozen-reference training banks.',
            flush=True,
        )
    write_json(directory / 'manifest.json', manifest)
    try:
        with closing(connect_readonly(source_path)) as source, closing(
            sqlite3.connect(directory / 'patch_clusters.sqlite')
        ) as destination:
            destination.row_factory = sqlite3.Row
            source.execute('BEGIN')
            count, query, datasets, settings = read_source_info(source, args)
            create_schema(destination)
            manifest.update(datasets=datasets, rgb_settings=settings)
            data, fingerprint, frames = extract(
                source, destination, directory, count, query, datasets, source_path.parent
            )
            side = math.isqrt(data.rgb.shape[1] // 3)
            if settings.get('rgb_size', side) != side:
                raise ValueError('Source similarity metadata disagrees with RGB descriptor dimensions.')
            manifest.update(
                num_patches=count,
                num_frames=frames,
                feature_dim=data.feature.shape[1],
                rgb_descriptor_size=side,
                source_patch_descriptor_sha256=fingerprint,
            )
            write_json(directory / 'manifest.json', manifest)
            finalize_fit(destination, directory, data, manifest, args, weights, started)
        write_json(directory / 'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(status='failed', failed_at=now(), error=str(exc))
        write_json(directory / 'manifest.json', manifest)
        raise
    print(f'Complete: {directory}/patch_clusters.sqlite', flush=True)
    return manifest


def build_from_cache(args, directory, weights):
    started = time.monotonic()
    manifest = base_manifest(args, weights, source_mode='cache')
    write_json(directory / 'manifest.json', manifest)
    try:
        with closing(sqlite3.connect(directory / 'patch_clusters.sqlite')) as destination:
            destination.row_factory = sqlite3.Row
            create_schema(destination)
            meta = extract_from_training_cache(args, directory, destination)
            feature_provenance = meta['feature_provenance']
            if not feature_provenance.get('backbone_fingerprint'):
                raise RuntimeError('Cache mode requires a verified backbone_fingerprint.')
            # Cross-check preprocess resize_to against CLI.
            preprocess = feature_provenance.get('preprocess') or {}
            expected_resize = list(args.resize_to)
            got_resize = preprocess.get('resize_to')
            if got_resize is not None and list(got_resize) != expected_resize:
                raise ValueError(
                    f'Cache preprocess resize_to={got_resize} disagrees with --resize-to {expected_resize}.'
                )
            manifest.update(
                datasets=meta['datasets'],
                rgb_settings=meta['rgb_settings'],
                num_patches=meta['count'],
                num_frames=meta['frames'],
                feature_dim=meta['feature_dim'],
                rgb_descriptor_size=meta['rgb_descriptor_size'],
                source_patch_descriptor_sha256=meta['fingerprint'],
                reference_feature_provenance=feature_provenance,
                reference_feature_provenance_status='verified',
                source_feature_semantics=(
                    'fixed snapshot of FrozenBackboneFeatureCache stage4 last-slice vectors; '
                    'paired with training RGB last frames resized to the backbone canvas'
                ),
                source_cache_root=meta['cache_root'],
                source_dataset_root=meta['dataset_root'],
                source_reference_checkpoint=meta['reference_checkpoint'],
                window_len=meta['window_len'],
                window_stride=meta['stride'],
                stage4_grid={'H': meta['grid_h'], 'W': meta['grid_w']},
                prototype_activations_used=False,
                graph_sqlite_used=False,
            )
            write_json(directory / 'manifest.json', manifest)
            finalize_fit(destination, directory, meta['data'], manifest, args, weights, started)
        write_json(directory / 'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(status='failed', failed_at=now(), error=str(exc))
        write_json(directory / 'manifest.json', manifest)
        raise
    print(f'Complete: {directory}/patch_clusters.sqlite', flush=True)
    print(
        f"Verified provenance fingerprint={manifest['reference_feature_provenance']['backbone_fingerprint'][:16]}…",
        flush=True,
    )
    return manifest


def build(args):
    weights = (args.feature_weight, args.rgb_weight, args.structure_weight)
    source_mode = str(args.source_mode).lower()
    if source_mode not in ('cache', 'graph'):
        raise ValueError("--source-mode must be 'cache' or 'graph'.")
    directory = prepare_output_directory(args.output_dir, args.overwrite)
    # Refuse silently clobbering the legacy graph bank when using cache defaults.
    legacy = DEFAULT_OUTPUT_DIR.expanduser().resolve()
    if source_mode == 'cache' and directory == legacy and not args.overwrite:
        raise FileExistsError(
            f'Cache mode refuses to write into the legacy bank path {legacy} without '
            '--overwrite. Pass a new --output-dir (default cache path is '
            f'{DEFAULT_CACHE_OUTPUT_DIR}).'
        )
    if source_mode == 'cache':
        return build_from_cache(args, directory, weights)
    return build_from_graph(args, directory, weights)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--source-mode',
        choices=('cache', 'graph'),
        default='cache',
        help="cache: training RGB + FrozenBackboneFeatureCache (default). graph: optional graph.sqlite.",
    )
    p.add_argument('--source-db', type=Path, default=DEFAULT_SOURCE_DB, help='graph.sqlite for --source-mode graph')
    p.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=f'New bank directory (default: cache→{DEFAULT_CACHE_OUTPUT_DIR}, graph→{DEFAULT_OUTPUT_DIR})',
    )
    p.add_argument('--datasets', nargs='+', default=['train'], help='graph mode dataset labels; ALL includes every label.')
    p.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET_ROOT, help='DHF1K root containing train/')
    p.add_argument('--split', default='train', help='Cache mode split (training only).')
    p.add_argument('--window-len', type=int, default=32)
    p.add_argument('--stride', type=int, default=1, help='Training-window stride (1 matches training starts).')
    p.add_argument('--cache-root', type=Path, default=DEFAULT_CACHE_ROOT,
                   help='Existing FrozenBackboneFeatureCache root (compatible entries are reused).')
    p.add_argument(
        '--reference-checkpoint',
        type=Path,
        default=DEFAULT_REFERENCE_CHECKPOINT,
        help='Frozen Kinetics VideoSwin weights (must match train.py).',
    )
    p.add_argument('--device', default='cuda:0' if _cuda_default() else 'cpu')
    p.add_argument('--storage-dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--resize-to', nargs=2, type=int, default=[224, 384], metavar=('H', 'W'),
                   help='Backbone spatial canvas; must match train.py (default 224 384).')
    p.add_argument('--rgb-size', type=int, default=16, help='RGB descriptor side length (HWC bank vectors).')
    p.add_argument('--max-windows', type=int, default=None, help='Optional cap for debugging.')
    p.add_argument(
        '--allow-compute',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='On cache miss, compute via frozen reference (default: true). Never deletes/overwrites incompatible entries.',
    )
    p.add_argument(
        '--cache-read-only',
        action='store_true',
        help='Fail on missing cache entries (disables --allow-compute).',
    )
    p.add_argument('--clustering', choices=('finch', 'kmedoids'), default='finch',
                   help='finch: cluster count determined automatically (default). kmedoids: fixed --num-prototypes.')
    p.add_argument('--num-prototypes', '--num-clusters', dest='num_prototypes', type=int, default=256,
                   help='kmedoids only: number of fixed reference clusters. Ignored with --clustering finch.')
    p.add_argument('--finch-level', type=int, default=None,
                   help='finch only: use this hierarchy level (0 = finest) instead of automatic selection.')
    p.add_argument('--min-clusters', type=int, default=16,
                   help='finch auto-selection: smallest acceptable cluster count.')
    p.add_argument('--max-clusters', type=int, default=1024,
                   help='finch auto-selection: largest acceptable cluster count (0 = no limit).')
    p.add_argument('--fit-sample-size', type=int, default=4096)
    p.add_argument('--fit-max-per-video', type=int, default=128, help='Discovery-sample cap per video; 0 disables it. Every patch is still assigned.')
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--restarts', type=int, default=2)
    p.add_argument('--medoid-candidates', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--feature-weight', type=float, default=1.)
    p.add_argument('--rgb-weight', type=float, default=1.)
    p.add_argument('--structure-weight', type=float, default=1.)
    p.add_argument('--threads', type=int, default=4, help='BLAS thread cap if threadpoolctl is installed.')
    p.add_argument('--overwrite', action='store_true', help='Rename an existing nonempty output directory to a timestamped backup first.')
    return p


def _cuda_default() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_CACHE_OUTPUT_DIR if args.source_mode == 'cache' else DEFAULT_OUTPUT_DIR
    if args.cache_read_only:
        args.allow_compute = False
    if args.clustering == 'finch':
        if args.num_prototypes != p.get_default('num_prototypes'):
            print('NOTE: --num-prototypes is ignored with --clustering finch.', flush=True)
        if args.min_clusters < 2 or args.max_clusters < 0 or (args.max_clusters and args.max_clusters < args.min_clusters):
            p.error('--min-clusters must be >= 2 and <= --max-clusters (0 = no limit).')
        # Placeholder so the pre-checks (enough patches / sample) use the lower bound;
        # fit_finch overwrites it with the discovered count.
        args.num_prototypes = args.min_clusters
    for name in ('num_prototypes', 'fit_sample_size', 'iterations', 'restarts', 'medoid_candidates', 'batch_size', 'threads', 'window_len', 'stride', 'rgb_size'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive.')
    if args.fit_sample_size < args.num_prototypes or args.fit_max_per_video < 0:
        p.error('Fit sample must be >= num-prototypes (--min-clusters with finch); fit-max-per-video must be >= 0.')
    if args.max_windows is not None and args.max_windows < 1:
        p.error('--max-windows must be positive when set.')
    weights = (args.feature_weight, args.rgb_weight, args.structure_weight)
    if not all(math.isfinite(w) and w >= 0 for w in weights) or not any(weights):
        p.error('Metric weights must be finite, nonnegative, and at least one must be positive.')
    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=args.threads)
    except ImportError:
        context = nullcontext()
    try:
        with context:
            build(args)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, FileNotFoundError, FileExistsError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
