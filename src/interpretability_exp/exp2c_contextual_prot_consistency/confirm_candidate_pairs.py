#!/usr/bin/env python3
"""Batch signature-pair confirmation with one JSON scan and shared SQLite storage.

Requires Python >=3.9 and numpy (tqdm optional). Major CSV/JSON/SQLite passes use tqdm
unless --no-tqdm. No project modules/checkpoint needed.
Scan all of candidate_pairs.csv and uniformly sample --max-pairs qualifying rows
without replacement (reservoir sampling; memory O(max-pairs)). Either orientation
is eligible: exactly 'low' versus 'high', 'high+low', or 'low+high'. Keep CSV A/B.
Use --pair-selection-seed (default 42) for sampling; --split-seed is only for videos.
A signature matches only if ALL of its prototypes are in a patch's ranked top 8.
Only distinct patches within the same physical frame form witnesses.

Outputs under --output-dir (names include _subset_{N}_seed_{S} from --subset-size and
--pair-selection-seed):
  selected_pairs_*.csv     source rows and selection ranks
  rules_report_*.csv       per-pair validity (appended as each pair is evaluated)
  all_pair_validity_*.csv  full validity table after the run completes
  shared_patch_data_*.sqlite frames, patches, signature memberships (no Cartesian rows)
  video_split_*.json       frozen video split (global default, per-pair optional)
  pair_results_*.csv       direction and all confidence measures for every pair
  per_video_results_*.csv  pooled counts, frame means, sample difference mean/std
  summary_*.json           number of valid rules and settings

Confidence defaults to mean of within-frame predicted non-tie agreement, matching
find_patterns.py's discovery statistic. Ordering is frozen from discovery only.
A valid rule needs discovery confidence AND confirmation confidence strictly above
--consistency-threshold, with nonempty subsets. Both pass flags and a confirmation-
only count are reported. --confidence-metric can select pooled or video_mean;
all three measures are always reported. GT never selects the predicted direction.

Global splitting shuffles sorted ALL source video IDs once, seed 42, fraction .5.
--split-mode per-pair reproduces the old pair-specific split on witness videos.
If signatures were mined using confirmation videos, this is a robustness check,
not independent held-out discovery. This script does not remine the signatures.

Memory scales with one frame, bounded ingestion/matching caches, and pair/video
aggregates. Disk stores each matched patch once and its signature memberships;
per-pair frame membership is reconstructed by the pair_frame_lists SQLite view.
For example: SELECT video,frame FROM pair_frame_lists WHERE selection_rank=1;
This view contains one entry per valid pair/frame and does not store extra rows.
Comparisons use sorted arrays and prefix/suffix searches, avoiding O(A*B) witness
materialization. A/B intersection patches are removed from self-comparisons.
The existing patch_archive.bin is not required: activation and saliency are read
in the SAME single streaming pass over patch_data.json. Ranked top_8/top_20
exports are supported; --prototype-ids-field overrides automatic field choice.
Use --reuse-store to rerun statistics without rescanning JSON; inputs/settings
are checked against stored provenance. Without --reuse-store, an existing tagged
shared_patch_data_*.sqlite in --output-dir is deleted and rebuilt automatically.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np


def progress_iter(iterable, desc, args, unit='it', total=None):
    if args.no_tqdm:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, dynamic_ncols=True, total=total)


def output_tag(args):
    return f'_subset_{args.subset_size}_seed_{args.pair_selection_seed}'


def tagged_output_path(args, filename):
    """Insert subset/seed tag before the extension, e.g. rules_report_subset_6_seed_42.csv."""
    path = Path(filename)
    return args.output_dir / f'{path.stem}{output_tag(args)}{path.suffix}'


# Streaming JSON reader adapted from generate_candidate_pairs.py.
def iter_json_array(handle, chunk_size=1 << 20):
    decoder = json.JSONDecoder()
    buffer, position, eof = '', 0, False
    def refill():
        nonlocal buffer, position, eof
        buffer = buffer[position:]
        position = 0
        chunk = handle.read(chunk_size)
        buffer += chunk
        eof = not chunk
    def peek():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                return buffer[position]
            if eof:
                return None
            refill()
    if peek() != '[':
        raise ValueError('Input must be a JSON array of patch objects.')
    position += 1
    first = True
    while True:
        char = peek()
        if char == ']':
            position += 1
            break
        if not first:
            if char != ',':
                raise ValueError('Expected comma between patch objects.')
            position += 1
            char = peek()
        if char != '{':
            raise ValueError('Expected patch object; input may be truncated.')
        while True:
            try:
                record, end = decoder.raw_decode(buffer, position)
                position = end
                break
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError('Invalid or truncated patch JSON.') from exc
                refill()
        yield record
        first = False
    if peek() is not None:
        raise ValueError('Unexpected content after JSON array.')


def integer(value, label):
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{label} must be a nonnegative integer: {value!r}')
    return value


def prototype_ids(record, args):
    if args.prototype_ids_field:
        raw = record.get(args.prototype_ids_field)
        details = None
    else:
        raw = details = None
        for k in (8, 20):
            ids_name = f'top_{k}_activated_prototype_indices'
            detail_name = f'top_{k}_activated_prototypes'
            if ids_name in record or detail_name in record:
                raw, details = record.get(ids_name), record.get(detail_name)
                break
    if details is not None:
        if not isinstance(details, list) or any(not isinstance(x, dict) for x in details):
            raise ValueError('Prototype details must be a list of objects.')
        if any('rank' in x for x in details):
            ranks = [integer(x.get('rank'), 'prototype rank') for x in details]
            if len(set(ranks)) != len(ranks):
                raise ValueError('Duplicate prototype ranks.')
            details = sorted(details, key=lambda x: x['rank'])
        extracted = []
        for item in details:
            values = [integer(item[k], k) for k in ('prototype_index', 'prot_idx') if k in item]
            if not values or len(set(values)) != 1:
                raise ValueError('Missing or inconsistent prototype index in details.')
            extracted.append(values[0])
        if raw is not None and raw != extracted:
            raise ValueError('Ranked prototype details and index list disagree.')
        raw = extracted
    if not isinstance(raw, list) or len(raw) < 8:
        raise ValueError('Need a ranked top-8 (or longer) prototype list; use --prototype-ids-field if needed.')
    ids = [integer(v, 'prototype index') for v in raw]
    if len(set(ids)) != len(ids) or any(i >= 512 for i in ids):
        raise ValueError('Expected distinct prototype IDs in [0,511].')
    return tuple(sorted(ids[:8]))


class Resolver:
    def __init__(self, record, args):
        self.video = args.video_field or next((k for k in ('video_fname', 'video_id', 'video_name') if k in record), None)
        self.frame = args.frame_field or next((k for k in ('absolute_frame_index', 'frame_filename', 'frame_no', 'frame_idx', 'frame_id') if k in record), None)
        if not self.video or not self.frame:
            raise ValueError('Patch records need video/frame identifiers; use --video-field/--frame-field.')
        self.stage = record.get('stage')
        self.has_stage = 'stage' in record
        self.context_fields = tuple(k for k in ('dataset_index', 'window_start_index', 'window_length') if k in record)
    def key(self, record):
        if ('stage' in record) != self.has_stage or record.get('stage') != self.stage:
            raise ValueError('Input mixes prototype stages; export/analyze one stage at a time.')
        values = []
        for name in (self.video, self.frame):
            val = record.get(name)
            if isinstance(val, bool) or not isinstance(val, (str, int)) or not str(val).strip():
                raise ValueError(f'Invalid identifier {name}: {val!r}')
            values.append(str(val))
        context = []
        for name in self.context_fields:
            val = record.get(name)
            if name not in record or isinstance(val, (dict, list)):
                raise ValueError(f'Missing/non-scalar frame context {name}')
            context.append(val)
        return (*values, json.dumps(context, separators=(',', ':')))


def row_qualifies_candidate_pair(row, signatures, source_row):
    a, b = row['discovery_origin_a'].strip(), row['discovery_origin_b'].strip()
    mixed = {'high', 'high+low', 'low+high'}
    if not ((a == 'low' and b in mixed) or (b == 'low' and a in mixed)):
        return False
    sa, sb = row['signature_id_a'].strip(), row['signature_id_b'].strip()
    if sa not in signatures or sb not in signatures:
        raise ValueError(f'Missing signature definition at candidate row {source_row}')
    if sa == sb:
        raise ValueError(f'Same signature on both sides at row {source_row}')
    return True


def load_selection(args):
    signatures = {}
    with args.activation_signatures.open(encoding='utf-8-sig', newline='') as handle:
        for row in progress_iter(csv.DictReader(handle), 'Load activation signatures', args, unit='sig'):
            sid = row['signature_id'].strip()
            ids = tuple(sorted(int(x.strip()) for x in row['prototype_ids'].split(',') if x.strip()))
            if not ids or len(set(ids)) != len(ids) or any(x < 0 or x >= 512 for x in ids):
                raise ValueError(f'Invalid signature {sid}')
            if sid in signatures:
                raise ValueError(f'Duplicate signature ID {sid}')
            signatures[sid] = ids
    rng = random.Random(args.pair_selection_seed)
    sample_size = args.max_pairs
    reservoir = []
    qualifying_total = 0
    with args.candidate_pairs.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for source_row, row in progress_iter(enumerate(reader, 1), 'Scan candidate pairs', args, unit='row'):
            if not row_qualifies_candidate_pair(row, signatures, source_row):
                continue
            qualifying_total += 1
            entry = dict(row, source_row=source_row)
            if len(reservoir) < sample_size:
                reservoir.append(entry)
            else:
                replace_at = rng.randrange(qualifying_total)
                if replace_at < sample_size:
                    reservoir[replace_at] = entry
    if not reservoir:
        raise ValueError('No qualifying candidate pairs.')
    reservoir.sort(key=lambda item: item['source_row'])
    for rank, item in enumerate(reservoir, 1):
        item['selection_rank'] = rank
    used = {row[k].strip() for row in reservoir for k in ('signature_id_a', 'signature_id_b')}
    return reservoir, {s: signatures[s] for s in sorted(used)}, qualifying_total


class Matcher:
    def __init__(self, signatures, args=None):
        self.root = {}
        items = signatures.items()
        if args is not None:
            items = progress_iter(items, 'Build signature matcher', args, unit='sig', total=len(signatures))
        for sid, ids in items:
            node = self.root
            for p in ids:
                node = node.setdefault(p, {})
            node.setdefault(-1, []).append(sid)
        self.matches = lru_cache(maxsize=16384)(self._matches)
    def _matches(self, ids):
        found = []
        def visit(node, start):
            found.extend(node.get(-1, ()))
            # Continue below terminals: selected signatures may have different sizes.
            for pos in range(start, len(ids)):
                child = node.get(ids[pos])
                if child is not None:
                    visit(child, pos+1)
        visit(self.root, 0)
        return tuple(found)


def fingerprint(path):
    st = path.stat()
    return {'path': str(path.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def provenance(args, selected, signatures):
    return {'version': 3, 'patch_data': fingerprint(args.patch_data),
            'selection_sha256': hashlib.sha256(json.dumps([selected, signatures], sort_keys=True).encode()).hexdigest(),
            'pair_selection_method': 'uniform_without_replacement_reservoir',
            'pair_selection_seed': args.pair_selection_seed,
            'subset_size': args.subset_size,
            'max_pairs': args.max_pairs,
            'top_k': 8, 'prototype_ids_field': args.prototype_ids_field,
            'pred_field': args.pred_saliency_field, 'gt_field': args.gt_saliency_field,
            'video_field': args.video_field, 'frame_field': args.frame_field}


def read_store_metadata(db):
    try:
        return {k: json.loads(v) for k, v in db.execute('SELECT key,value FROM metadata')}
    except sqlite3.OperationalError:
        return {}


def ingest_tables_present(db):
    """True once build_store has created schema (not an empty file from sqlite3.connect)."""
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patches' LIMIT 1"
    ).fetchone()
    return row is not None


def provenance_diff(stored, expected):
    hints = []
    if not stored:
        return hints
    for key in ('version', 'pair_selection_method', 'pair_selection_seed', 'subset_size', 'max_pairs',
                'prototype_ids_field', 'pred_field', 'gt_field', 'video_field', 'frame_field'):
        if stored.get(key) != expected.get(key):
            hints.append(f'{key}: stored={stored.get(key)!r} current={expected.get(key)!r}')
    if stored.get('patch_data') != expected.get('patch_data'):
        hints.append('patch_data fingerprint changed')
    if stored.get('selection_sha256') != expected.get('selection_sha256'):
        hints.append('candidate pair sample changed (selection_sha256 differs)')
    return hints


def assert_reuse_store(db, dbpath, expected):
    metadata = read_store_metadata(db)
    if not metadata.get('complete'):
        raise ValueError(
            f'{dbpath} is incomplete (ingest was interrupted or metadata missing). '
            'Delete shared_patch_data.sqlite and rerun without --reuse-store, or use a new --output-dir.')
    stored = metadata.get('provenance')
    if stored == expected:
        return metadata['ingest_stats']
    hints = provenance_diff(stored, expected)
    detail = '; '.join(hints) if hints else 'provenance JSON differs'
    raise ValueError(
        f'Stored settings do not match this run ({detail}). '
        'Rebuild with a new --output-dir, or delete shared_patch_data.sqlite and rerun without --reuse-store.')


def remove_store_files(dbpath):
    dbpath = Path(dbpath)
    for path in (dbpath, Path(f'{dbpath}-wal'), Path(f'{dbpath}-shm'), Path(f'{dbpath}-journal')):
        path.unlink(missing_ok=True)


def open_store_db(dbpath, reuse_store):
    """Open SQLite; without --reuse-store, replace any existing store file first."""
    dbpath = Path(dbpath)
    if reuse_store:
        return sqlite3.connect(dbpath)
    candidates = [dbpath]
    legacy = dbpath.parent / 'shared_patch_data.sqlite'
    if legacy not in candidates:
        candidates.append(legacy)
    removed = [path for path in candidates if path.exists()]
    for path in removed:
        remove_store_files(path)
    if removed:
        names = ', '.join(str(p) for p in removed)
        print(f'Removed existing store file(s) ({names}); starting fresh ingest.', flush=True)
    return sqlite3.connect(dbpath)


def build_store(db, args, selected, signatures, expected):
    db.executescript('''
      CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE frames(frame_id INTEGER PRIMARY KEY, video TEXT NOT NULL,
        frame TEXT NOT NULL, context TEXT NOT NULL, UNIQUE(video,frame));
      CREATE TABLE patches(frame_id INTEGER NOT NULL, patch INTEGER NOT NULL,
        pred REAL NOT NULL, gt REAL NOT NULL, PRIMARY KEY(frame_id,patch)) WITHOUT ROWID;
      CREATE TABLE memberships(frame_id INTEGER NOT NULL, signature TEXT NOT NULL,
        patch INTEGER NOT NULL, PRIMARY KEY(frame_id,signature,patch)) WITHOUT ROWID;
      CREATE TABLE candidate_pairs(selection_rank INTEGER PRIMARY KEY,
        signature_id_a TEXT NOT NULL, signature_id_b TEXT NOT NULL);
      CREATE VIEW pair_frame_lists AS
        WITH sf AS (SELECT frame_id,signature,COUNT(*) AS n,MIN(patch) AS first_patch
                    FROM memberships GROUP BY frame_id,signature)
        SELECT c.selection_rank,f.frame_id,f.video,f.frame
        FROM candidate_pairs c JOIN sf a ON a.signature=c.signature_id_a
        JOIN sf b ON b.frame_id=a.frame_id AND b.signature=c.signature_id_b
        JOIN frames f ON f.frame_id=a.frame_id
        WHERE a.n*b.n>1 OR a.first_patch<>b.first_patch;
    ''')
    db.executemany('INSERT INTO candidate_pairs VALUES(?,?,?)',
                   ((r['selection_rank'],r['signature_id_a'].strip(),r['signature_id_b'].strip()) for r in selected))
    matcher = Matcher(signatures, args)
    resolver = None
    videos = set()
    n_records = n_patches = n_members = 0
    @lru_cache(maxsize=8192)
    def frame_id(video, frame, context):
        db.execute('INSERT OR IGNORE INTO frames(video,frame,context) VALUES(?,?,?)', (video, frame, context))
        fid, stored_context = db.execute('SELECT frame_id,context FROM frames WHERE video=? AND frame=?', (video, frame)).fetchone()
        if context != stored_context:
            raise ValueError(f'Frame {video}/{frame} mixes temporal contexts.')
        return fid
    with args.patch_data.open(encoding='utf-8-sig') as handle:
        records = progress_iter(iter_json_array(handle), 'Single patch scan', args, unit='patch')
        for record in records:
            n_records += 1
            if resolver is None:
                resolver = Resolver(record, args)
            video, frame, context = resolver.key(record)
            videos.add(video)
            fid = frame_id(video, frame, context)
            patch = integer(record.get('patch_index'), 'patch_index')
            matched = matcher.matches(prototype_ids(record, args))
            if matched:
                values = []
                for name in (args.pred_saliency_field, args.gt_saliency_field):
                    val = record.get(name)
                    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                        raise ValueError(f'{name} must be finite at {video}/{frame}/{patch}')
                    values.append(float(val))
                try:
                    db.execute('INSERT INTO patches VALUES(?,?,?,?)', (fid, patch, *values))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f'Duplicate matched patch {video}/{frame}/{patch}; input must contain one record per patch.') from exc
                db.executemany('INSERT INTO memberships VALUES(?,?,?)', ((fid, sid, patch) for sid in matched))
                n_patches += 1
                n_members += len(matched)
            if n_records % 50000 == 0:
                db.commit()
                if args.no_tqdm:
                    print(f'Scanned {n_records:,}; retained {n_patches:,} patches', flush=True)
    if provenance(args, selected, signatures) != expected:
        raise ValueError('Input changed while scanning; rebuild on stable files.')
    db.execute('CREATE INDEX membership_signature_frame ON memberships(signature,frame_id)')
    stats = {'scanned_patches': n_records, 'stored_patches': n_patches, 'stored_memberships': n_members,
             'frames': db.execute('SELECT COUNT(*) FROM frames').fetchone()[0], 'videos': sorted(videos)}
    for key, value in [('provenance', expected), ('ingest_stats', stats), ('complete', True)]:
        db.execute('INSERT INTO metadata VALUES(?,?)', (key, json.dumps(value)))
    db.commit()
    return stats


# Exact counts without Cartesian witness materialization. Each array has unique
# patch IDs. searchsorted uses the SAME subtraction comparisons as the old script,
# including epsilon boundary behavior (avoids rounded a +/- epsilon thresholds).
def prepare_signal(values):
    mean = float(values.mean())
    return np.sort(values), mean, float(np.square(values-mean).sum())


def ordering_counts(a_ids, av, b_ids, bv, epsilon, common_count=None, prepared_a=None, prepared_b=None):
    n = len(av)*len(bv)
    if common_count is None:
        _, ai, bi = np.intersect1d(a_ids, b_ids, assume_unique=True, return_indices=True)
        if len(ai) and not np.array_equal(av[ai], bv[bi]):
            raise ValueError('Shared patch saliency disagrees.')
        common_count = len(ai)
    n -= common_count
    if not n:
        return (0, 0, 0, 0, 0.0, 0.0)
    prepared_a = prepare_signal(av) if prepared_a is None else prepared_a
    prepared_b = prepare_signal(bv) if prepared_b is None else prepared_b
    sorted_b = prepared_b[0]
    # Vectorized binary searches, O(A log B), bounded to length A.
    def first_not(predicate):
        lo = np.zeros(len(av), dtype=np.int64)
        hi = np.full(len(av), len(sorted_b), dtype=np.int64)
        while np.any(lo < hi):
            active = lo < hi
            mid = (lo + hi)//2
            values = sorted_b[np.minimum(mid, len(sorted_b)-1)]
            move = active & predicate(av - values)
            lo = np.where(move, mid+1, lo)
            hi = np.where(active & ~move, mid, hi)
        return lo
    if epsilon == 0:
        # Default case uses compiled binary searches, sorted once per signature.
        a_higher = int(np.searchsorted(sorted_b, av, side='left').sum())
        b_higher = int((len(sorted_b)-np.searchsorted(sorted_b, av, side='right')).sum())
    else:
        a_higher = int(first_not(lambda diff: diff > epsilon).sum())
        b_higher = int((len(sorted_b) - first_not(lambda diff: diff >= -epsilon)).sum())
    ties = n-a_higher-b_higher
    # Closed-form moments of all differences, centered to reduce cancellation.
    ma, mb = prepared_a[1], prepared_b[1]
    sumdiff = len(av)*len(bv)*(ma-mb)
    sumsq = (len(bv)*prepared_a[2] + len(av)*prepared_b[2]
             + len(av)*len(bv)*(ma-mb)**2)
    mean = sumdiff/n
    m2 = max(0.0, sumsq - n*mean*mean)
    return n, a_higher, b_higher, ties, mean, m2


class Aggregate:
    def __init__(self):
        self.n = self.a = self.b = self.ties = 0
        self.mean = self.m2 = 0.0
        self.frames = self.usable = 0
        self.frame_b_sum = self.frame_b_all_sum = 0.0
    def add(self, stats):
        n, a, b, ties, mean, m2 = stats
        if not n:
            return
        delta = mean-self.mean
        total = self.n+n
        self.m2 += m2+delta*delta*self.n*n/total
        self.mean += delta*n/total
        self.n = total
        self.a += a
        self.b += b
        self.ties += ties
        self.frames += 1
        self.frame_b_all_sum += b/n
        if a+b:
            self.usable += 1
            self.frame_b_sum += b/(a+b)
    def merged(self, other):
        if not other.n:
            return
        delta = other.mean-self.mean
        total = self.n+other.n
        self.m2 += other.m2+delta*delta*self.n*other.n/total
        self.mean += delta*other.n/total
        self.n = total
        for name in ('a','b','ties','frames','usable','frame_b_sum','frame_b_all_sum'):
            setattr(self, name, getattr(self, name)+getattr(other, name))
    def b_rate(self, metric='frame_mean'):
        if metric == 'frame_mean':
            return self.frame_b_sum/self.usable if self.usable else math.nan
        return self.b/(self.a+self.b) if self.a+self.b else math.nan
    def fields(self, prefix):
        return {prefix+k: v for k,v in {
            'rows': self.n, 'a_higher': self.a, 'b_higher': self.b, 'ties': self.ties,
            'difference_mean': self.mean if self.n else math.nan,
            'difference_std': math.sqrt(max(0,self.m2/(self.n-1))) if self.n>1 else (0.0 if self.n else math.nan),
            'pooled_b_confidence': self.b_rate('pooled'),
            'frame_b_confidence': self.b_rate(),
            'pooled_b_including_ties': self.b/self.n if self.n else math.nan,
            'frame_b_including_ties': self.frame_b_all_sum/self.frames if self.frames else math.nan,
            'usable_frames': self.usable}.items()}


def analyze_store(db, args, selected):
    adjacency = defaultdict(list)
    for i,row in enumerate(selected):
        adjacency[row['signature_id_a'].strip()].append((i,row['signature_id_b'].strip()))
    by_pair = [defaultdict(lambda: (Aggregate(), Aggregate())) for _ in selected]
    query = '''SELECT m.frame_id,f.video,m.signature,m.patch,p.pred,p.gt
        FROM memberships m JOIN frames f ON f.frame_id=m.frame_id
        JOIN patches p ON p.frame_id=m.frame_id AND p.patch=m.patch
        ORDER BY m.frame_id,m.signature,m.patch'''
    current = None
    buckets = defaultdict(list)
    frames_done = 0
    def process(video):
        nonlocal frames_done
        arrays = {sid: (np.array([v[0] for v in rows], dtype=np.int64),
                        np.array([v[1] for v in rows]), np.array([v[2] for v in rows]))
                  for sid,rows in buckets.items()}
        prepared = {sid:(prepare_signal(a[1]),prepare_signal(a[2])) for sid,a in arrays.items()}
        for sa, aa in arrays.items():
            for pi,sb in adjacency.get(sa, ()):
                bb = arrays.get(sb)
                if bb is None:
                    continue
                if len(aa[0])*len(bb[0]) == 1 and aa[0][0] == bb[0][0]:
                    continue
                common_count = len(np.intersect1d(aa[0],bb[0],assume_unique=True))
                target = by_pair[pi][video]
                target[0].add(ordering_counts(aa[0],aa[1],bb[0],bb[1],args.tie_epsilon,
                                             common_count,prepared[sa][0],prepared[sb][0]))
                target[1].add(ordering_counts(aa[0],aa[2],bb[0],bb[2],args.tie_epsilon,
                                             common_count,prepared[sa][1],prepared[sb][1]))
        frames_done += 1
        if args.no_tqdm and frames_done % 2000 == 0:
            print(f'Analyzed {frames_done:,} shared frames', flush=True)
    membership_rows = db.execute(
        'SELECT COUNT(*) FROM memberships m JOIN patches p ON p.frame_id=m.frame_id AND p.patch=m.patch'
    ).fetchone()[0]
    row_iter = progress_iter(
        db.execute(query), 'Analyze membership rows', args, unit='row', total=membership_rows)
    for fid, video, sid, patch, pred, gt in row_iter:
        if current is not None and fid != current[0]:
            process(current[1])
            buckets.clear()
        current = fid, video
        buckets[sid].append((patch,pred,gt))
    if current is not None:
        process(current[1])
    return by_pair


def split(videos, args):
    ordered = sorted(videos)
    if args.discovery_videos:
        return set(args.discovery_videos)&set(ordered)
    random.Random(args.split_seed).shuffle(ordered)
    if len(ordered)<2:
        return set(ordered)
    n = min(len(ordered)-1, max(1,int(len(ordered)*args.discovery_fraction+.5)))
    return set(ordered[:n])


def finite_mean(values):
    vals = [v for v in values if math.isfinite(v)]
    return math.fsum(vals)/len(vals) if vals else math.nan


def subset_summary(groups, signal, direction):
    total = Aggregate()
    entries = [pair[signal] for pair in groups.values()]
    for entry in entries:
        total.merged(entry)
    fields = total.fields('')
    fields['frames'] = total.frames
    fields['videos'] = len(groups)
    for metric in ('pooled','frame_mean','video_mean'):
        rate = finite_mean(x.b_rate() for x in entries) if metric=='video_mean' else total.b_rate(metric)
        fields[metric+'_confidence'] = rate if direction=='B' else 1-rate if direction=='A' else math.nan
    return fields


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v,float) and not math.isfinite(v) else v for k,v in row.items()})


RULES_REPORT_FIELDS = (
    'candidate_pair_no', 'act_sign1', 'act_sign2', 'act_sign_source1', 'act_sign_source2',
    'higher_pred_sal', 'higher_gt_sal', 'is rule valid',
)


def format_csv_cell(value):
    return '' if isinstance(value, float) and not math.isfinite(value) else value


class RulesReportWriter:
    """Append rules_report.csv when a newly evaluated pair is valid (flush after each)."""

    def __init__(self, path):
        self.path = path
        self.valid_count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as handle:
            csv.DictWriter(handle, fieldnames=RULES_REPORT_FIELDS).writeheader()

    def note_pair(self, row):
        if row['is rule valid'] != 'valid':
            return
        self.valid_count += 1
        with self.path.open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=RULES_REPORT_FIELDS)
            writer.writerow({k: format_csv_cell(row[k]) for k in RULES_REPORT_FIELDS})
            handle.flush()
        print(f"Valid rule #{self.valid_count} (pair {row['candidate_pair_no']}): "
              f"{row['act_sign1']} vs {row['act_sign2']} — "
              f"pooled pred B>A {row['higher_pred_sal']:.4g}%", flush=True)


def pooled_b_over_a_percent(aggregate):
    """Share of non-tie witness comparisons where patch B saliency exceeds patch A."""
    rate = aggregate.b_rate('pooled')
    return 100.0 * rate if math.isfinite(rate) else math.nan


def witness_subset_totals(db, by_pair, selected, args, global_discovery):
    """Distinct witness videos/frames and predicted patch-pair comparisons per split."""
    pair_discovery = {}
    for i, _row in enumerate(selected):
        groups = {v: c for v, c in by_pair[i].items() if c[0].n}
        rank = i + 1
        if args.split_mode == 'global':
            pair_discovery[rank] = global_discovery
        else:
            pair_discovery[rank] = split(groups, args)
    buckets = {
        'discovery': {'videos': set(), 'frames': set(), 'patch_pair_comparisons': 0},
        'confirmation': {'videos': set(), 'frames': set(), 'patch_pair_comparisons': 0},
    }
    query = '''SELECT c.selection_rank, c.frame_id, c.video
               FROM pair_frame_lists c'''
    for rank, frame_id, video in db.execute(query):
        discovery = pair_discovery[int(rank)]
        subset = 'discovery' if video in discovery else 'confirmation'
        bucket = buckets[subset]
        bucket['videos'].add(video)
        bucket['frames'].add(int(frame_id))
    for i, _row in enumerate(selected):
        rank = i + 1
        discovery = pair_discovery[rank]
        for video, (pred, _gt) in by_pair[i].items():
            if not pred.n:
                continue
            subset = 'discovery' if video in discovery else 'confirmation'
            buckets[subset]['patch_pair_comparisons'] += pred.n
    return {
        name: {
            'videos': len(data['videos']),
            'distinct_frames': len(data['frames']),
            'patch_pair_comparisons': data['patch_pair_comparisons'],
        }
        for name, data in buckets.items()
    }


def candidate_pair_report_row(selection_rank, row, groups, args):
    pred_total = Aggregate()
    gt_total = Aggregate()
    for pred, gt in groups.values():
        pred_total.merged(pred)
        gt_total.merged(gt)
    higher_pred = pooled_b_over_a_percent(pred_total)
    higher_gt = pooled_b_over_a_percent(gt_total)
    pred_rate = pred_total.b_rate('pooled')
    valid = ('valid' if math.isfinite(pred_rate) and pred_rate > args.consistency_threshold else 'invalid')
    return {
        'candidate_pair_no': selection_rank,
        'act_sign1': row['signature_id_a'].strip(),
        'act_sign2': row['signature_id_b'].strip(),
        'act_sign_source1': row['discovery_origin_a'].strip(),
        'act_sign_source2': row['discovery_origin_b'].strip(),
        'higher_pred_sal': higher_pred,
        'higher_gt_sal': higher_gt,
        'is rule valid': valid,
    }


def save_reports(args, selected, by_pair, ingest_stats, qualifying_total, db):
    global_discovery = split(ingest_stats['videos'],args)
    splits = {'mode':args.split_mode, 'seed':args.split_seed, 'fraction':args.discovery_fraction,
              'discovery_videos': sorted(global_discovery),
              'confirmation_videos': sorted(set(ingest_stats['videos'])-global_discovery)}
    if args.split_mode=='per-pair':
        splits['pairs'] = {}
    results, video_rows, candidate_report = [], [], []
    rules_report = RulesReportWriter(tagged_output_path(args, 'rules_report.csv'))
    for i, row in progress_iter(enumerate(selected), 'Report candidate pairs', args,
                                unit='pair', total=len(selected)):
        groups = {v:c for v,c in by_pair[i].items() if c[0].n}
        report_row = candidate_pair_report_row(i + 1, row, groups, args)
        candidate_report.append(report_row)
        rules_report.note_pair(report_row)
        discovery = global_discovery if args.split_mode=='global' else split(groups,args)
        if args.split_mode=='per-pair':
            splits['pairs'][str(i+1)] = {'discovery':sorted(discovery), 'confirmation': sorted(set(groups)-discovery)}
        dg = {v:c for v,c in groups.items() if v in discovery}
        cg = {v:c for v,c in groups.items() if v not in discovery}
        # Preserve original direction selection, regardless of reporting metric.
        disc_b = subset_summary(dg,0,'B')['frame_mean_confidence']
        direction = 'B' if disc_b>.5 else 'A' if disc_b<.5 else ''
        result = {'selection_rank': i+1, 'source_row':row['source_row'],
                  'signature_id_a':row['signature_id_a'], 'signature_id_b':row['signature_id_b'],
                  'discovery_origin_a':row['discovery_origin_a'], 'discovery_origin_b':row['discovery_origin_b'],
                  'direction':direction, 'confidence_metric':args.confidence_metric}
        for name,subset in [('discovery',dg),('confirmation',cg)]:
            for label,signal in [('pred',0),('gt',1)]:
                result.update({f'{name}_{label}_{k}':v for k,v in subset_summary(subset,signal,direction).items()})
        dc = result[f'discovery_pred_{args.confidence_metric}_confidence']
        cc = result[f'confirmation_pred_{args.confidence_metric}_confidence']
        dpass = bool(direction and math.isfinite(dc) and dc>args.consistency_threshold)
        cpass = bool(direction and math.isfinite(cc) and cc>args.consistency_threshold)
        result.update(discovery_pass=dpass, confirmation_pass=cpass, valid_rule=dpass and cpass,
                      status='no_discovery_order' if not direction else 'no_confirmation_non_ties' if not math.isfinite(cc) else 'valid' if dpass and cpass else 'below_threshold')
        results.append(result)
        for video,(pred,gt) in sorted(groups.items()):
            item = {'selection_rank':i+1,'signature_id_a':row['signature_id_a'],
                    'signature_id_b':row['signature_id_b'], 'video':video,
                    'subset':'discovery' if video in discovery else 'confirmation',
                    'direction':direction, 'frames':pred.frames}
            item.update(pred.fields('pred_'))
            item.update(gt.fields('gt_'))
            for label,entry in [('pred',pred),('gt',gt)]:
                br = entry.b_rate()
                item[f'{label}_frozen_order_frame_confidence'] = br if direction=='B' else 1-br if direction=='A' else math.nan
            video_rows.append(item)
    selected_pairs_path = tagged_output_path(args, 'selected_pairs.csv')
    all_pair_validity_path = tagged_output_path(args, 'all_pair_validity.csv')
    pair_results_path = tagged_output_path(args, 'pair_results.csv')
    per_video_path = tagged_output_path(args, 'per_video_results.csv')
    video_split_path = tagged_output_path(args, 'video_split.json')
    summary_path = tagged_output_path(args, 'summary.json')
    write_csv(selected_pairs_path, selected)
    write_csv(all_pair_validity_path, candidate_report)
    write_csv(pair_results_path, results)
    write_csv(per_video_path, video_rows)
    video_split_path.write_text(json.dumps(splits, indent=2) + '\n')
    summary = {'requested_pairs':args.max_pairs, 'selected_pairs':len(selected),
               'qualifying_rows_total':qualifying_total,
               'pair_selection_method':'uniform_without_replacement_reservoir',
               'pair_selection_seed':args.pair_selection_seed,
               'discovery_pass_count':sum(r['discovery_pass'] for r in results),
               'confirmation_pass_count':sum(r['confirmation_pass'] for r in results),
               'valid_rule_count':sum(r['valid_rule'] for r in results),
               'valid_rule_fraction':sum(r['valid_rule'] for r in results)/len(results),
               'confidence_metric':args.confidence_metric, 'threshold':args.consistency_threshold,
               'comparison':'>', 'ties_excluded_from_confidence':True,
               'direction_selection':'discovery predicted frame mean, non-ties',
               'split_mode':args.split_mode, 'split_seed':args.split_seed,
               'discovery_fraction':args.discovery_fraction, 'tie_epsilon':args.tie_epsilon,
               'ingest':ingest_stats,
               'database_bytes':tagged_output_path(args, 'shared_patch_data.sqlite').stat().st_size,
               'output_tag': output_tag(args),
               'note':'If candidates were mined from confirmation videos, this measures split robustness, not independent held-out confirmation.'}
    summary['counts_by_metric'] = {}
    for metric in ('pooled','frame_mean','video_mean'):
        dp = [bool(r['direction'] and r[f'discovery_pred_{metric}_confidence']>args.consistency_threshold) for r in results]
        cp = [bool(r['direction'] and r[f'confirmation_pred_{metric}_confidence']>args.consistency_threshold) for r in results]
        summary['counts_by_metric'][metric] = {'discovery_pass_count':sum(dp),
                                               'confirmation_pass_count':sum(cp),
                                               'valid_rule_count':sum(d and c for d,c in zip(dp,cp))}
    witness_totals = witness_subset_totals(db, by_pair, selected, args, global_discovery)
    summary['witness_totals'] = witness_totals
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f"\nRandom sample: {len(selected):,} / requested {args.max_pairs:,} "
          f"from {qualifying_total:,} qualifying rows (seed {args.pair_selection_seed})\n"
          f"Discovery passes: {summary['discovery_pass_count']:,}\n"
          f"Confirmation passes for frozen order: {summary['confirmation_pass_count']:,}\n"
          f"Valid in both subsets: {summary['valid_rule_count']:,} / {len(selected):,}\n"
          f"Metric: {args.confidence_metric}; strict confidence > {args.consistency_threshold:.2%}\n"
          f"Results: {pair_results_path}\n"
          f"Valid rules (incremental): {rules_report.path} "
          f"({rules_report.valid_count:,} rows)\n"
          f"All pairs: {all_pair_validity_path}\n"
          f"Witness pool ({args.split_mode} split): "
          f"discovery — {witness_totals['discovery']['videos']:,} videos, "
          f"{witness_totals['discovery']['distinct_frames']:,} distinct frames, "
          f"{witness_totals['discovery']['patch_pair_comparisons']:,} patch-pair comparisons; "
          f"confirmation — {witness_totals['confirmation']['videos']:,} videos, "
          f"{witness_totals['confirmation']['distinct_frames']:,} distinct frames, "
          f"{witness_totals['confirmation']['patch_pair_comparisons']:,} patch-pair comparisons",
          flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--candidate-pairs',type=Path,required=True)
    p.add_argument('--activation-signatures',type=Path,required=True)
    p.add_argument('--patch-data',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True,help='Use a dedicated new batch-results directory')
    p.add_argument('--max-pairs',type=int,default=1000,
                   help='Uniform random sample size over qualifying candidate rows')
    p.add_argument('--pair-selection-seed',type=int,default=42,
                   help='RNG seed for candidate-pair sampling (independent of --split-seed)')
    p.add_argument('--subset-size', type=int, default=None,
                   help='Filename tag only: outputs named *_subset_N_seed_<pair-selection-seed>.* '
                        '(default: same as --max-pairs)')
    p.add_argument('--discovery-fraction',type=float,default=.5)
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--split-mode',choices=['global','per-pair'],default='global')
    p.add_argument('--discovery-videos',nargs='+')
    p.add_argument('--consistency-threshold',type=float,default=.95)
    p.add_argument('--confidence-metric',choices=['frame_mean','pooled','video_mean'],default='frame_mean')
    p.add_argument('--tie-epsilon',type=float,default=0.0)
    p.add_argument('--pred-saliency-field',default='patch_pred_saliency')
    p.add_argument('--gt-saliency-field',default='patch_gt_saliency')
    p.add_argument('--video-field')
    p.add_argument('--frame-field')
    p.add_argument('--prototype-ids-field',help='Ranked prototype-ID list; first 8 are used')
    p.add_argument('--reuse-store',action='store_true')
    p.add_argument('--no-tqdm',action='store_true')
    a = p.parse_args(argv)
    if a.max_pairs<1 or not 0<a.discovery_fraction<1 or not .5<a.consistency_threshold<=1:
        p.error('Need positive max-pairs, fraction in (0,1), threshold in (.5,1].')
    if not math.isfinite(a.tie_epsilon) or a.tie_epsilon<0:
        p.error('tie-epsilon must be finite and nonnegative.')
    if a.subset_size is None:
        a.subset_size = a.max_pairs
    if a.subset_size < 1:
        p.error('--subset-size must be >= 1.')
    return a


def main(argv=None):
    args = parse_args(argv)
    selected,signatures,qualifying_total = load_selection(args)
    print(f'Randomly sampled {len(selected):,} pair(s) from {qualifying_total:,} qualifying CSV rows '
          f'(pair-selection-seed={args.pair_selection_seed}); {len(signatures):,} distinct signatures',
          flush=True)
    if qualifying_total < args.max_pairs:
        print('Warning: candidate file has fewer qualifying rows than --max-pairs; using all of them.',
              flush=True)
    impossible = [sid for sid,ids in signatures.items() if len(ids)>8]
    if impossible:
        print(f'Note: {len(impossible)} signatures have >8 prototypes and cannot match top-8 patches.',flush=True)
    expected = provenance(args,selected,signatures)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    dbpath = tagged_output_path(args, 'shared_patch_data.sqlite')
    if args.reuse_store and not dbpath.exists():
        raise ValueError('--reuse-store requires an existing completed database.')
    db = open_store_db(dbpath, args.reuse_store)
    try:
        db.execute('PRAGMA cache_size=-65536')
        db.execute('PRAGMA temp_store=FILE')
        if args.reuse_store:
            ingest_stats = assert_reuse_store(db, dbpath, expected)
        else:
            ingest_stats = build_store(db,args,selected,signatures,expected)
        print(f"Shared store: {ingest_stats['stored_patches']:,} patches; {ingest_stats['stored_memberships']:,} memberships",flush=True)
        by_pair = analyze_store(db,args,selected)
        save_reports(args,selected,by_pair,ingest_stats,qualifying_total,db)
    finally:
        db.close()
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,sqlite3.Error) as exc:
        raise SystemExit(f'Error: {exc}')
