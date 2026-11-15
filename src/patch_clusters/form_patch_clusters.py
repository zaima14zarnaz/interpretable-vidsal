#!/usr/bin/env python3
"""Build fixed stage4 patch clusters, independently of model assignments.

Dependencies: python -m pip install numpy scipy pillow
No torch, checkpoint loading, model inference, activation filter or graph edges.

Examples (defaults match this project's prototype graph):
    python form_patch_clusters.py
    python form_patch_clusters.py --num-prototypes 128 --fit-sample-size 4096
    python form_patch_clusters.py --source-db /path/to/graph.sqlite --datasets train

Input: graph.sqlite from generate_prot_graph.py or generate_prot_overlap_graph.py.
ALL stage4 nodes in the selected datasets are read, including inactive patches.
Complete visual_descriptors are required; run the source generator's
ACTION='cache_similarity' if the cache is incomplete. Zero/nonfinite descriptors
fail explicitly instead of assigning patches using invented similarities.

Distance = wf*(1-feature_cosine) + wr*RGB_MAE + ws*(1-RGB_structure_cosine).
Structure is per-channel spatially centered RGB; both flat=1, one flat=0.
With unit weights, distance = 3 - the graph's visual similarity score.
The source RGB descriptors/preprocessing are preserved, not regenerated from
JPEG previews. Source features become a FIXED SNAPSHOT: this script does not
claim they came from the original frozen teacher. Use a graph extracted with
your chosen reference backbone/checkpoint, and record that provenance.

Fitting uses seeded approximate k-medoids on a video-capped sample. Medoid
updates evaluate a bounded number of candidates per cluster. Every source
patch is then assigned to the closest final medoid under the full distance.
This is approximate fitting, not an exact global k-medoids optimum, and the
reference is a real MEDOID patch, not an average RGB signature. No global NxN
matrix is allocated. Fitting uses at most O(sample_size*K) distance memory;
full assignment uses O(batch_size*K). Descriptor arrays are disk-backed.

Output, by default /data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster:
    manifest.json              settings, provenance, hashes, coverage, status
    patch_clusters.sqlite      fixed cluster/prototype IDs + every patch/frame
    cluster_references.npz     ordered medoid descriptors and source identities
    fit_sample_indices.npy     exact sample used for prototype discovery
    descriptors/*.npy          aligned per-patch vectors, node IDs and labels

Prototype IDs 0..K-1 are NEW reference-cluster IDs, not the old model's IDs.
Do not rebuild/renumber this reference bank midway through a training run.
The training loss must use matching RGB settings and reference-feature source.
Patch identity is dataset/video/window/target/stage/grid cell; node IDs refer to
this source DB only. Train.py/concept_creation.py are not changed by this tool.
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
from scipy.spatial.distance import cdist

DEFAULT_SOURCE_DB = Path('/data/quantization/zaima/videosal_datasets/dhf1k/explanation_graphs/prototype_graph_outputs/graph.sqlite')
DEFAULT_OUTPUT_DIR = Path('/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster')
FORMAT_VERSION = 'fixed_stage4_patch_clusters_v1'


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


def build(args):
    source_path = args.source_db.expanduser().resolve()
    directory = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f'Source graph DB not found: {source_path}. Supply --source-db /path/to/graph.sqlite.')
    if source_path.is_relative_to(directory):
        raise ValueError('Output directory must not contain the source database.')
    manifest_path = source_path.parent / 'manifest.json'
    source_manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if source_manifest.get('status', 'complete') != 'complete':
        raise ValueError('Source graph extraction is not complete; finish or repair it before defining fixed references.')
    if directory.exists() and any(directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f'{directory} is not empty. Use a new --output-dir or --overwrite to preserve the old folder as a backup.')
        suffix = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        backup = directory.with_name(directory.name + '.backup-' + suffix)
        directory.rename(backup)
        print(f'Previous output preserved at {backup}', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    weights = (args.feature_weight, args.rgb_weight, args.structure_weight)
    manifest = {'format_version': FORMAT_VERSION, 'status': 'building', 'created_at': now(),
                'source_database': str(source_path), 'source_manifest': source_manifest,
                'source_manifest_sha256': sha256_file(manifest_path) if manifest_path.is_file() else None,
                'source_feature_semantics': 'fixed snapshot of stored raw stage4 vectors; source checkpoint recorded above',
                'stage': 'stage4', 'prototype_ids': 'new fixed reference IDs, zero-based; unrelated to old activation IDs',
                'num_prototypes': args.num_prototypes, 'seed': args.seed,
                'distance': 'wf*(1-feature_cosine) + wr*rgb_mae + ws*(1-centered_rgb_structure_cosine)',
                'weights': {'feature': weights[0], 'rgb_mae': weights[1], 'structure': weights[2]},
                'structure_flat_policy': 'both flat=1, one flat=0', 'rgb_order': 'HWC',
                'reference_type': 'real medoid patches', 'reference_groups_fixed': True,
                'algorithm': 'sampled_candidate_k_medoids_then_full_closest_medoid_assignment',
                'fit_sample_size_requested': args.fit_sample_size, 'fit_max_per_video': args.fit_max_per_video,
                'iterations': args.iterations, 'restarts': args.restarts, 'medoid_candidates': args.medoid_candidates,
                'batch_size': args.batch_size, 'saliency_used_for_clustering': False,
                'model_assignments_used_for_clustering': False, 'all_selected_stage4_patches_included': True}
    write_json(directory / 'manifest.json', manifest)
    try:
        with closing(connect_readonly(source_path)) as source, closing(sqlite3.connect(directory / 'patch_clusters.sqlite')) as destination:
            destination.row_factory = sqlite3.Row
            source.execute('BEGIN')  # One consistent source/WAL snapshot.
            count, query, datasets, settings = read_source_info(source, args)
            create_schema(destination)
            manifest.update(datasets=datasets, rgb_settings=settings)
            data, fingerprint, frames = extract(source, destination, directory, count, query, datasets, source_path.parent)
            side = math.isqrt(data.rgb.shape[1] // 3)
            if settings.get('rgb_size', side) != side:
                raise ValueError('Source similarity metadata disagrees with RGB descriptor dimensions.')
            manifest.update(num_patches=count, num_frames=frames, feature_dim=data.feature.shape[1],
                            rgb_descriptor_size=side, source_patch_descriptor_sha256=fingerprint)
            write_json(directory / 'manifest.json', manifest)
            sample, videos = choose_fit_sample(destination, args.fit_sample_size, args.fit_max_per_video, np.random.default_rng(args.seed))
            if len(sample) < args.num_prototypes:
                raise ValueError('Video-capped fitting sample is smaller than --num-prototypes; increase --fit-max-per-video (0 disables cap).')
            np.save(directory / 'fit_sample_indices.npy', sample, allow_pickle=False)
            print(f'{count:,} patches, {videos:,} videos; fitting on {len(sample):,} patches.', flush=True)
            medoids, fit_history, objective = fit_medoids(data, sample, args, weights)
            stats = save_assignments(destination, data, directory, medoids, args, weights)
            remaining = destination.execute('SELECT COUNT(*) FROM patches WHERE prototype_id IS NULL').fetchone()[0]
            if remaining or destination.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Final assignment/database integrity check failed.')
            manifest.update(status='complete', completed_at=now(), elapsed_seconds=round(time.monotonic() - started, 3),
                            fit_sample_size=len(sample), num_videos=videos, fit_mean_distance=objective,
                            fit_history=fit_history, clusters=stats,
                            database='patch_clusters.sqlite', references='cluster_references.npz',
                            descriptor_arrays='descriptors/*.npy, indexed by patches.descriptor_index')
            manifest['references_sha256'] = sha256_file(directory / 'cluster_references.npz')
            destination.execute('INSERT INTO metadata VALUES (?,?)', ('manifest', json.dumps(manifest, allow_nan=False)))
            destination.execute('ANALYZE')
            destination.commit()
        write_json(directory / 'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(status='failed', failed_at=now(), error=str(exc))
        write_json(directory / 'manifest.json', manifest)
        raise
    print(f'Complete: {directory}/patch_clusters.sqlite', flush=True)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-db', type=Path, default=DEFAULT_SOURCE_DB)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--datasets', nargs='+', default=['train'], help='Source dataset labels; ALL explicitly includes every label.')
    p.add_argument('--num-prototypes', '--num-clusters', dest='num_prototypes', type=int, default=128)
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


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ('num_prototypes', 'fit_sample_size', 'iterations', 'restarts', 'medoid_candidates', 'batch_size', 'threads'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive.')
    if args.fit_sample_size < args.num_prototypes or args.fit_max_per_video < 0:
        p.error('Fit sample must be >= num-prototypes; fit-max-per-video must be >= 0.')
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
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
