#!/usr/bin/env python3
"""Patch-to-patch prototype overlap graph, with an offline interactive viewer.

Lives under interpretability_exp/exp4_expl_report/; finds src/evaluation.py
via SRC_DIR. Edit SETTINGS, then run:
    python generate_prot_overlap_graph.py

Actions (edit ACTION; no command-line arguments):
    extract       Run the checkpoint, save every patch in the selected stages,
                  and write a self-contained graph_viewer.html neighborhood.
    cache_similarity  Add missing visual descriptors to an existing database.
    view          Regenerate the offline viewer from an existing graph database.
    serve         Explore any stored node at http://127.0.0.1:8765 (no inference).
    export_edges  Materialize ALL qualifying undirected edges as CSV; can be huge.
    selftest      Exercise overlap, filtering, ranking and HTML export without torch.

This is a PATCH graph, not a graph comparing prototype vectors. Each node is a
patch occurrence (dataset/window/target frame/stage/grid cell). Edge weight is
the number of shared prototype IDs. Different stages never share an edge.

Two separate, queryable graph definitions are stored:
    topk:   TOP_K highest visual_concept_logits (default 16); SIMILARITY
            CANDIDATES, not a claim that all participated in prediction.
            Extraction stores the full raw top-K lists. At serve/view/export
            time, TOP_K_FREQUENCY_EXCLUDE drops prototypes whose patch-level
            frequency (# patches with p in top-K / # patches) exceeds the
            threshold. Higher threshold keeps more prototypes/edges; 1.0 keeps
            all. Change the setting and re-run serve/view without re-extracting.
    active: active_visual_prototype_indices filtered by visual_validity_mask;
            the model's actual forward-valid selection. It may contain one ID.
            Active sets are not frequency-filtered.
Edge COLOR = stage4 cosine + (1 - RGB MAE) + RGB structure cosine.
Edge WIDTH and membership/ranking still use shared prototype count.
RGB descriptors use bilinear-resized 16x16 patches in [0,1]. RGB structure
cosine compares flattened RGB after subtracting each channel's spatial mean.
Feature cosine uses the raw final-frame stage4 backbone patch vector, never
prototype logits or prototype-ID overlap. Missing/zero features produce grey
edges with a reason. Other-stage edges remain grey (no stage4 mapping assumed).
Existing databases need ACTION='cache_similarity' once; this runs checkpoint
inference for uncached stage4 frames and preserves the graph and node IDs.
The script never changes assignments, weights, or model architecture.

The full graph is stored EXACTLY as indexed patch/prototype incidence in SQLite.
For any pair i,j, w(i,j)=|P_i intersect P_j|. This avoids allocating an N x N
matrix or enumerating every edge of a potentially enormous graph. No neighbors
are discarded from that representation. The HTML contains a clearly bounded
neighborhood; serve mode queries the complete database on demand.

Outputs:
    graph.sqlite         all nodes, top-k/active membership and frame metadata
    manifest.json        configuration, provenance, coverage, completion status
    frames/*.jpg         one input-frame thumbnail per evaluated window
    graph_viewer.html    offline, embedded images, pan/zoom/drag/click/filter
    neighborhood.json    same displayed nodes and weighted edges, including IDs

Select a node and click "Top 50 patches" to open a 10-column x 5-row gallery.
The selected patch occupies the first cell; the next 49 cells contain neighbors
ordered by overlap count descending, with node-ID ascending as the tie breaker.
The overlap definition, minimum overlap and other-video filter follow the viewer.
Serve mode ranks the complete database; offline mode ranks only the embedded
snapshot and explicitly labels that scope. Fewer matches leave empty cells.

Dependencies for extraction: the existing evaluation.py environment, NumPy, PIL.
Viewing/query/export/selftest do not import torch or evaluation.py.
Extraction and cache_similarity require the original checkpoint environment.

By default this follows evaluation.py's validation root with dense stride=1.
Set WINDOW_STRIDE=32 to match the supplied evaluation's sampling. All grid cells of every processed
full window are kept, including low-scoring cells. Early frames without a full
history and truncated tail windows are not evaluated. 'All patches' therefore
means all target-frame cells of the configured roots/windows/stages, not every
RGB frame of every dataset split. Add roots/stages below as needed.

The prediction score is the regional mean of saliency_map as returned by the
model (possibly raw logits with output_activation='none'). Its within-frame
percentile uses midranks, so tied flat maps stay at 0.5. It is not a fixation
probability or a calibrated cross-frame saliency score. GT is stored separately.

Only the expected model detail keys are accepted: unsupported tensor layouts
fail with an actionable error instead of silently reconstructing assignments.
"""

from __future__ import annotations

import base64
import zlib
from functools import lru_cache
from collections import Counter
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# SETTINGS: edit here, as in evaluation.py.
# ---------------------------------------------------------------------------
ACTION = "serve"  # extract | cache_similarity | view | serve | export_edges | selftest
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[1]  # .../src containing evaluation.py
EVALUATION_FILE = SRC_DIR / "evaluation.py"
CHECKPOINT_PATH = "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20261112_154638/epoch_011.pth"  # None -> evaluation.CHECKPOINT_PATH
DATASET_ROOTS = "/data/quantization/zaima/videosal_datasets/dhf1k/train"
# None -> {"val": evaluation.VAL_DATASET_DIR}
# str/Path -> {folder_name: path}; or a dict, e.g. {"train": "/data/.../dhf1k/train"}
STAGES = ("stage4",)  # add stage3 etc.; IDs are namespaced by stage
WINDOW_STRIDE = 64  # dense full windows; set 32 to match supplied evaluation.py
BATCH_SIZE = 1  # avoids mixing variable-length windows; safe default
NUM_WORKERS = None  # None -> evaluation.NUM_WORKERS
GPU_ID = 0
USE_EVAL_AMP = True  # reuse train.py's _amp_enabled/_amp_dtype when available
TOP_K = 8
# Applied at serve/view/export time only (raw top-K stays in the database).
# Frequency(p) = (# patches where p is in the raw top-K) / (# patches).
# Exclude prototypes with frequency > this value. Higher => keep more edges.
# On this DHF1K stage4 graph, patch frequencies top out near ~0.5, so 0.75 keeps
# everything; try ~0.25–0.40 to drop the most common prototypes. 1.0 keeps all.
# Frame-level "appears anywhere in the sample" frequencies are ~1.0 for almost
# every prototype and are intentionally NOT used (they remove all edges).
TOP_K_FREQUENCY_EXCLUDE = 0.9
TOP_K_FREQUENCY_STATS_VERSION = "patch_fraction_v1"
MAX_WINDOWS = None  # None = no limit; optional small pilot, explicitly recorded
OUTPUT_DIR = Path("/data/quantization/zaima/videosal_datasets/dhf1k/explanation_graphs/prototype_graph_outputs")
SAVE_FRAME_THUMBNAILS = True
FRAME_THUMB_WIDTH = 384

# Viewer/export settings. Edge definition is always intersection COUNT.
GRAPH_KIND = "topk"  # topk | active
VIEW_SEED_NODE = None  # None = first node with high within-frame predicted rank
VIEW_MAX_NODES = 160
VIEW_MIN_OVERLAP = 1
VIEW_CROSS_VIDEO_ONLY = True  # viewer neighbor filter only; full DB keeps all
SERVER_PORT = 8765
EXPORT_MIN_OVERLAP = 1
EXPORT_MAX_EDGES = None  # None = all; otherwise CSV/metadata explicitly truncated
SEED = 42

# Visual similarity settings. Existing caches record and enforce these semantics.
RGB_DESCRIPTOR_SIZE = 16
LAYOUT_DISTANCE_MODE = "overlap"  # overlap | activation (also switchable in the viewer)
LAYOUT_ACTIVATION_WEIGHT = 100.0  # extra layout pixels at activation difference 1
EDGE_COLOR_RANGE = None  # None: min/max of valid displayed edge scores; or (low, high).
# Capture the native backbone output on the model's EXISTING forward pass.
# Supports module calls and direct forward()/forward_features()/extract_features()
# calls, which bypass a normal nn.Module forward hook.
# Default: model.backbone returns a dict with key "stage4".
# If your wrapper differs, set its named-module path and output path explicitly.
STAGE4_FEATURE_MODULE = "backbone"
STAGE4_FEATURE_OUTPUT_PATH = ("stage4",)  # () if the selected module returns the tensor itself.
STAGE4_FEATURE_LAYOUT = "auto"  # auto | BCTHW | BTCHW | BTHWC | BCHW | BHWC
SIMILARITY_VERSION = "stage4_rgb_structure_v1"
SIMILARITY_FORMULA = "stage4_cosine + (1 - rgb_mae) + rgb_structure_cosine"

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_dataset_roots(roots, default_val_dir):
    if roots is None:
        return {"val": str(default_val_dir)}
    if isinstance(roots, (str, Path)):
        path = Path(roots)
        return {path.name or "dataset": str(path)}
    if isinstance(roots, dict) and roots:
        return {str(k): str(v) for k, v in roots.items()}
    raise TypeError("DATASET_ROOTS must be None, a path, or a non-empty {label: path} dict.")


def dump_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def midrank_percentiles(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite patch saliency scores.")
    if a.size <= 1:
        return np.full(a.size, 0.5)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts[inv] + (counts[inv] - 1) / 2) / (a.size - 1)


def connect_db(path, readonly=False):
    if readonly:
        con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    else:
        con = sqlite3.connect(path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    con.row_factory = sqlite3.Row
    return con


def create_schema(con):
    con.executescript("""
    CREATE TABLE frames (
        id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, video TEXT NOT NULL,
        dataset_index INTEGER NOT NULL, window_start INTEGER NOT NULL,
        target_index INTEGER NOT NULL, target_name TEXT NOT NULL,
        source_path TEXT NOT NULL, thumb_path TEXT, window_names TEXT NOT NULL,
        input_h INTEGER NOT NULL, input_w INTEGER NOT NULL,
        UNIQUE(dataset, dataset_index)
    );
    CREATE TABLE nodes (
        id INTEGER PRIMARY KEY, frame_id INTEGER NOT NULL, stage TEXT NOT NULL,
        row INTEGER NOT NULL, col INTEGER NOT NULL, grid_h INTEGER NOT NULL,
        grid_w INTEGER NOT NULL, pred_score REAL NOT NULL, pred_percentile REAL NOT NULL,
        gt_score REAL, gt_percentile REAL, topk_ids TEXT NOT NULL,
        topk_logits TEXT NOT NULL, active_ids TEXT NOT NULL, active_weights TEXT NOT NULL,
        UNIQUE(frame_id, stage, row, col), FOREIGN KEY(frame_id) REFERENCES frames(id)
    );
    CREATE TABLE memberships (
        node_id INTEGER NOT NULL, stage TEXT NOT NULL, kind TEXT NOT NULL,
        prototype_id INTEGER NOT NULL, PRIMARY KEY(node_id, kind, prototype_id),
        FOREIGN KEY(node_id) REFERENCES nodes(id)
    );
    CREATE INDEX postings ON memberships(stage, kind, prototype_id, node_id);
    CREATE INDEX nodes_by_frame ON nodes(frame_id,stage);
    CREATE INDEX frames_by_video ON frames(dataset,video);
    """)
    ensure_similarity_schema(con)


def get_node(con, node_id):
    row = con.execute("""SELECT n.*, f.dataset, f.video, f.target_name,
        f.target_index, f.window_start, f.dataset_index, f.source_path,
        f.thumb_path, f.input_h, f.input_w FROM nodes n
        JOIN frames f ON f.id=n.frame_id WHERE n.id=?""", (int(node_id),)).fetchone()
    if row is None:
        raise ValueError(f"Node {node_id} does not exist.")
    d = dict(row)
    for key in ("topk_ids", "topk_logits", "active_ids", "active_weights"):
        d[key] = json.loads(d[key])
    return d


def filter_parallel_lists(ids, values, excluded):
    """Drop excluded prototype IDs while keeping optional parallel values aligned."""
    kept_ids, kept_values = [], []
    excluded = {int(p) for p in excluded}
    values = list(values) if values is not None else None
    for index, prototype in enumerate(ids):
        prototype = int(prototype)
        if prototype in excluded:
            continue
        kept_ids.append(prototype)
        if values is not None:
            kept_values.append(values[index] if index < len(values) else None)
    return kept_ids, kept_values


def compute_topk_sample_frequencies(con):
    """Patch-level frequencies of raw top-K prototypes.

    Does not modify the database. Frequency for prototype p at a stage is
    (# patches / nodes where p appears in the raw top-K) / (# patches at that stage).
    Frame-level "present in any patch of the sample" rates are near 1.0 here and
    are not used, because they exclude nearly every prototype.
    """
    stages = [str(row[0]) for row in con.execute("SELECT DISTINCT stage FROM nodes ORDER BY stage")]
    report = {
        "version": TOP_K_FREQUENCY_STATS_VERSION,
        "top_k": int(TOP_K),
        "sample_definition": "patch occurrences (stage4 grid cells), not frames",
        "frequency_definition": (
            "fraction of patches whose raw top-K logit list contains the prototype"
        ),
        "stages": {},
    }
    for stage in stages:
        n_patches = int(con.execute(
            "SELECT COUNT(*) FROM nodes WHERE stage=?", (stage,)).fetchone()[0])
        n_samples = int(con.execute(
            "SELECT COUNT(DISTINCT frame_id) FROM nodes WHERE stage=?", (stage,)).fetchone()[0])
        rows = con.execute("""
            SELECT m.prototype_id, COUNT(*) AS patch_count
            FROM memberships m
            WHERE m.stage=? AND m.kind='topk'
            GROUP BY m.prototype_id
            ORDER BY patch_count DESC, m.prototype_id ASC
        """, (stage,)).fetchall()
        frequencies = {}
        patch_counts = {}
        for prototype_id, patch_count in rows:
            pid = int(prototype_id)
            patch_counts[str(pid)] = int(patch_count)
            frequencies[str(pid)] = (float(patch_count) / n_patches) if n_patches else 0.0
        report["stages"][stage] = {
            "num_patches": n_patches,
            "num_samples": n_samples,
            "num_prototypes_observed": len(frequencies),
            "frequencies": frequencies,
            "patch_counts": patch_counts,
        }
        if frequencies:
            vals = sorted(frequencies.values())
            print(f"Top-{TOP_K} patch frequencies [{stage}]: {n_patches:,} patches, "
                  f"{len(frequencies):,} prototypes; freq min/median/max="
                  f"{vals[0]:.4f}/{vals[len(vals)//2]:.4f}/{vals[-1]:.4f}.", flush=True)
        else:
            print(f"Top-{TOP_K} patch frequencies [{stage}]: no top-K memberships.", flush=True)
    return report


def excluded_prototypes_by_stage(frequency_report, max_frequency=TOP_K_FREQUENCY_EXCLUDE):
    """Map stage -> set of prototype IDs whose patch frequency exceeds the threshold."""
    max_frequency = float(max_frequency)
    if not math.isfinite(max_frequency) or not 0.0 < max_frequency <= 1.0:
        raise ValueError("TOP_K_FREQUENCY_EXCLUDE must be in (0, 1].")
    excluded = {}
    summary = {}
    for stage, info in (frequency_report or {}).get("stages", {}).items():
        frequencies = {int(pid): float(freq) for pid, freq in (info.get("frequencies") or {}).items()}
        dropped = {pid for pid, freq in frequencies.items() if freq > max_frequency}
        excluded[str(stage)] = dropped
        summary[str(stage)] = {
            "num_excluded": len(dropped),
            "excluded_prototype_ids": sorted(dropped),
            "excluded_frequencies": {str(pid): frequencies[pid] for pid in sorted(dropped) if pid in frequencies},
            "kept_prototype_count": max(0, len(frequencies) - len(dropped)),
            "num_patches": info.get("num_patches"),
            "num_samples": info.get("num_samples"),
            "num_prototypes_observed": info.get("num_prototypes_observed", len(frequencies)),
        }
    return excluded, {
        "max_frequency": max_frequency,
        "version": (frequency_report or {}).get("version", TOP_K_FREQUENCY_STATS_VERSION),
        "top_k": int((frequency_report or {}).get("top_k", TOP_K)),
        "sample_definition": (frequency_report or {}).get("sample_definition"),
        "frequency_definition": (frequency_report or {}).get("frequency_definition"),
        "stages": summary,
    }


def _frequency_report_is_complete(report):
    if (report or {}).get("version") != TOP_K_FREQUENCY_STATS_VERSION:
        return False
    stages = (report or {}).get("stages") or {}
    return bool(stages) and all(isinstance(info.get("frequencies"), dict) for info in stages.values())


def load_topk_frequency_report(directory, con=None):
    """Load saved top-K patch frequencies, recomputing from the DB if needed."""
    directory = Path(directory)
    candidates = []
    stats_path = directory / "topk_frequency_stats.json"
    if stats_path.is_file():
        candidates.append(json.loads(stats_path.read_text(encoding="utf-8")))
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = manifest.get("topk_frequency_stats")
        if isinstance(report, dict):
            candidates.append(report)
    for report in candidates:
        if _frequency_report_is_complete(report):
            return report
    own = con is None
    if own:
        con = connect_db(directory / "graph.sqlite", readonly=True)
    try:
        report = compute_topk_sample_frequencies(con)
        # Refresh on-disk stats when the definition changes so later serves reuse it.
        try:
            dump_json(directory / "topk_frequency_stats.json", report)
        except OSError:
            pass
        return report
    finally:
        if own:
            con.close()


def resolve_topk_frequency_exclusion(directory, con=None, max_frequency=None):
    """Return (excluded_by_stage, active_filter_summary) for the current threshold."""
    max_frequency = TOP_K_FREQUENCY_EXCLUDE if max_frequency is None else float(max_frequency)
    report = load_topk_frequency_report(directory, con=con)
    return excluded_prototypes_by_stage(report, max_frequency)


def filter_node_for_topk_frequency(node, excluded_by_stage):
    """Return a shallow-copied node with high-frequency top-K prototypes removed."""
    if not excluded_by_stage:
        return node
    excluded = excluded_by_stage.get(str(node["stage"])) or excluded_by_stage.get(node["stage"]) or set()
    if not excluded:
        return node
    out = dict(node)
    raw_ids = list(node.get("topk_ids") or [])
    out["topk_ids_raw"] = raw_ids
    out["topk_ids"], out["topk_logits"] = filter_parallel_lists(
        raw_ids, node.get("topk_logits"), excluded)
    if node.get("topk_weights") is not None:
        out["topk_weights"] = filter_parallel_lists(raw_ids, node["topk_weights"], excluded)[1]
    return out


def neighbors(con, node_id, kind="topk", min_overlap=1, limit=None, cross_video_only=False,
              excluded_by_stage=None):
    """Exact neighbors ordered by overlap DESC, node ID ASC; optional result cap."""
    if kind not in ("topk", "active") or int(min_overlap) < 1:
        raise ValueError("Use kind=topk|active and min_overlap >= 1.")
    node = get_node(con, node_id)
    if kind == "topk" and excluded_by_stage:
        node = filter_node_for_topk_frequency(node, excluded_by_stage)
    ids = node["topk_ids" if kind == "topk" else "active_ids"]
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    params = [node["stage"], kind, *ids, int(node_id)]
    restriction = ""
    if cross_video_only:
        restriction = " AND NOT (f.dataset=? AND f.video=?)"
        params += [node["dataset"], node["video"]]
    sql = f"""SELECT m.node_id, COUNT(*) AS overlap FROM memberships m
        JOIN nodes n ON n.id=m.node_id JOIN frames f ON f.id=n.frame_id
        WHERE m.stage=? AND m.kind=? AND m.prototype_id IN ({marks})
        AND m.node_id<>? {restriction}
        GROUP BY m.node_id HAVING COUNT(*)>=?
        ORDER BY overlap DESC, m.node_id ASC"""
    params.append(int(min_overlap))
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(0, int(limit)))
    return [(int(r["node_id"]), int(r["overlap"])) for r in con.execute(sql, params)]


def induced_edges(nodes, kind, min_overlap=1):
    key = "topk_ids" if kind == "topk" else "active_ids"
    buckets, pairs = {}, Counter()
    for node in nodes:
        for pid in set(node[key]):
            for other in buckets.get((node["stage"], pid), []):
                pairs[(other, node["id"])] += 1
            buckets.setdefault((node["stage"], pid), []).append(node["id"])
    return [{"source": a, "target": b, "weight": w}
            for (a, b), w in sorted(pairs.items()) if w >= min_overlap]


def add_preview_images(nodes, directory):
    cache = {}
    for node in nodes:
        rel = node.get("thumb_path")
        if not rel or not (Path(directory) / rel).is_file():
            continue
        if rel not in cache:
            data = (Path(directory) / rel).read_bytes()
            cache[rel] = ("data:image/jpeg;base64," + base64.b64encode(data).decode(),
                          Image.open(io.BytesIO(data)).convert("RGB"))
        uri, image = cache[rel]
        node["frame_image"] = uri
        w, h = image.size
        box = (round(node["col"] * w / node["grid_w"]),
               round(node["row"] * h / node["grid_h"]),
               round((node["col"] + 1) * w / node["grid_w"]),
               round((node["row"] + 1) * h / node["grid_h"]))
        crop = image.crop(box)
        crop.thumbnail((128, 128))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        node["crop_image"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def choose_seed(con):
    if VIEW_SEED_NODE is not None:
        get_node(con, VIEW_SEED_NODE)
        return int(VIEW_SEED_NODE)
    row = con.execute("SELECT id FROM nodes WHERE pred_percentile>=0.95 ORDER BY id LIMIT 1").fetchone()
    if row is None:
        row = con.execute("SELECT id FROM nodes ORDER BY id LIMIT 1").fetchone()
    if row is None:
        raise ValueError("The graph has no nodes.")
    return int(row[0])


def make_payload(con, directory, seed=None, kind=GRAPH_KIND, limit=VIEW_MAX_NODES,
                 min_overlap=VIEW_MIN_OVERLAP, cross_video=VIEW_CROSS_VIDEO_ONLY,
                 excluded_by_stage=None, frequency_filter=None):
    seed = choose_seed(con) if seed is None else int(seed)
    limit = max(1, min(int(limit), 600))
    if excluded_by_stage is None:
        excluded_by_stage, frequency_filter = resolve_topk_frequency_exclusion(directory, con=con)
    nearby = neighbors(con, seed, kind, int(min_overlap), limit, bool(cross_video),
                       excluded_by_stage=excluded_by_stage)
    selected = [seed] + [n for n, _ in nearby[:limit - 1]]
    nodes = [get_node(con, nid) for nid in selected]
    load_strengths(con, nodes)
    nodes = [filter_node_for_topk_frequency(node, excluded_by_stage) for node in nodes]
    edges = induced_edges(nodes, kind, int(min_overlap))
    pairs, similarity = annotate_edges(con, nodes, edges, all_pairs=True, kind=kind)
    add_preview_images(nodes, directory)
    return {"nodes": nodes, "edges": edges, "seed": seed, "kind": kind,
            "visual_similarity_pairs": pairs, "visual_similarity": similarity,
            "total_nodes": int(con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]),
            "display_limit": limit, "neighbor_results_capped": len(nearby) >= limit,
            "min_overlap": int(min_overlap), "cross_video_neighbors": bool(cross_video),
            "topk_frequency_filter": frequency_filter,
            "scope": "Bounded 1-hop neighborhood; edges include all displayed-node overlaps. "
                     "Full exact raw top-K graph is retained in graph.sqlite; serve/view apply "
                     "TOP_K_FREQUENCY_EXCLUDE. Cross-video filtering applies to seed-neighbor "
                     "retrieval, not every displayed edge."}


def make_patch_gallery(con, directory, node_id, kind=GRAPH_KIND, min_overlap=1, cross_video=False,
                       excluded_by_stage=None, frequency_filter=None):
    if excluded_by_stage is None:
        excluded_by_stage, frequency_filter = resolve_topk_frequency_exclusion(directory, con=con)
    ranked = neighbors(con, int(node_id), kind=kind, min_overlap=int(min_overlap),
                       limit=49, cross_video_only=bool(cross_video),
                       excluded_by_stage=excluded_by_stage)
    key = "topk_ids" if kind == "topk" else "active_ids"
    raw_nodes = [get_node(con, int(node_id))] + [get_node(con, other_id) for other_id, _ in ranked]
    load_strengths(con, raw_nodes)
    patches = [filter_node_for_topk_frequency(node, excluded_by_stage) for node in raw_nodes]
    anchor = patches[0]
    anchor_ids = set(anchor[key])
    anchor.update(gallery_selected=True, edge_strength=None, shared_prototypes=sorted(anchor_ids))
    for node, (_, strength) in zip(patches[1:], ranked):
        node.update(gallery_selected=False, edge_strength=int(strength),
                    shared_prototypes=sorted(anchor_ids.intersection(node[key])))
    edges = [{"source": anchor["id"], "target": n["id"], "weight": n["edge_strength"]}
             for n in patches[1:]]
    annotate_edges(con, patches, edges, kind=kind)
    for node, edge in zip(patches[1:], edges):
        node["visual_similarity"] = {k: v for k, v in edge.items() if k not in ("source", "target", "weight")}
    add_preview_images(patches, directory)
    return {"selected_node": int(node_id), "patches": patches, "kind": kind,
            "min_overlap": int(min_overlap), "cross_video": bool(cross_video),
            "grid_columns": 10, "grid_rows": 5, "total_slots": 50,
            "num_neighbors": len(ranked), "scope": "complete_database",
            "topk_frequency_filter": frequency_filter,
            "ranking": "overlap count descending; node ID ascending for ties"}

def similarity_settings():
    return {"version": SIMILARITY_VERSION, "formula": SIMILARITY_FORMULA,
            "rgb_size": RGB_DESCRIPTOR_SIZE, "rgb_range": [0, 1],
            "rgb_resize": "bilinear, per-channel float32, uniform input-grid crop",
            "structure": "cosine of per-channel spatially centered RGB",
            "feature_module": STAGE4_FEATURE_MODULE,
            "feature_output_path": list(STAGE4_FEATURE_OUTPUT_PATH),
            "feature_layout": STAGE4_FEATURE_LAYOUT}


def has_table(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def ensure_similarity_schema(con):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS visual_descriptors (
            node_id INTEGER PRIMARY KEY, feature_dim INTEGER NOT NULL,
            stage4_vector BLOB NOT NULL, rgb_vector BLOB NOT NULL,
            FOREIGN KEY(node_id) REFERENCES nodes(id));
        CREATE TABLE IF NOT EXISTS node_strengths (
            node_id INTEGER PRIMARY KEY, topk_weights TEXT NOT NULL,
            active_weights TEXT NOT NULL, FOREIGN KEY(node_id) REFERENCES nodes(id));
        CREATE TABLE IF NOT EXISTS similarity_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    expected = json.dumps(similarity_settings(), sort_keys=True)
    row = con.execute("SELECT value FROM similarity_metadata WHERE key='settings'").fetchone()
    if row is not None and row[0] != expected:
        if con.execute("SELECT 1 FROM visual_descriptors LIMIT 1").fetchone() is not None:
            raise ValueError("Visual descriptor settings differ from this database's populated cache. "
                             "Restore the original settings or use a new OUTPUT_DIR.")
        # A failed first capture can leave metadata with no descriptors. Permit
        # correcting its source configuration without discarding the patch graph.
        con.execute("UPDATE similarity_metadata SET value=? WHERE key='settings'", (expected,))
    con.execute("INSERT OR IGNORE INTO similarity_metadata VALUES ('settings',?)", (expected,))
    con.commit()


def pack_vector(a):
    return zlib.compress(np.asarray(a, dtype='<f4').tobytes(), level=3)


@lru_cache(maxsize=2048)
def unpack_vector(blob):
    a = np.frombuffer(zlib.decompress(blob), dtype='<f4').astype(np.float64)
    a.flags.writeable = False
    return a


def rgb_descriptor(rgb, row, col, grid_h, grid_w):
    """Actual loader RGB, not lossy preview JPEGs. Same normalized grid as nodes."""
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError("Expected finite HWC RGB for visual similarity.")
    if a.min() < -0.01 or a.max() > 1.01:
        raise ValueError("RGB similarity requires RGB in [0,1].")
    h, w = a.shape[:2]
    y0, y1 = round(row * h / grid_h), round((row + 1) * h / grid_h)
    x0, x1 = round(col * w / grid_w), round((col + 1) * w / grid_w)
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Empty RGB crop for this patch grid.")
    crop = a[y0:y1, x0:x1].clip(0, 1)
    size = (int(RGB_DESCRIPTOR_SIZE), int(RGB_DESCRIPTOR_SIZE))
    return np.stack([np.asarray(Image.fromarray(crop[..., c]).resize(size, Image.Resampling.BILINEAR))
                     for c in range(3)], axis=-1).clip(0, 1).astype(np.float32).reshape(-1)


def store_descriptor(con, node_id, feature, rgb):
    feature, rgb = np.asarray(feature).reshape(-1), np.asarray(rgb).reshape(-1)
    if not np.isfinite(feature).all() or not np.isfinite(rgb).all():
        raise ValueError("Non-finite visual descriptor.")
    con.execute("INSERT OR REPLACE INTO visual_descriptors VALUES (?,?,?,?)",
                (int(node_id), int(feature.size), pack_vector(feature), pack_vector(rgb)))


def store_strengths(con, node_id, info, bi, row, col):
    weights = info["weights"]
    if weights is None:
        top, active = [], []
    else:
        top = [float(weights[bi, row, col, p]) for p in info["top"][bi, row, col]]
        native = sorted(set(int(p) for p in info["active"][bi, row, col][info["mask"][bi, row, col]]))
        active = [float(weights[bi, row, col, p]) for p in native]
    con.execute("INSERT OR REPLACE INTO node_strengths VALUES (?,?,?)",
                (int(node_id), json.dumps(top), json.dumps(active)))


def load_strengths(con, nodes):
    exists = has_table(con, "node_strengths")
    for node in nodes:
        row = con.execute("SELECT topk_weights,active_weights FROM node_strengths WHERE node_id=?",
                          (node["id"],)).fetchone() if exists else None
        # Old active_weights are native activations; old topk_logits are NOT activations.
        node["topk_weights"] = json.loads(row[0]) if row else []
        if row:
            node["active_weights"] = json.loads(row[1])


def activation_difference(left, right, kind):
    """Mean symmetric relative activation difference. Independent of overlap count.

    d = mean_p |a_p-b_p| / (|a_p|+|b_p|), p in shared prototypes.
    An exactly zero/zero pair contributes zero. The result is in [0,1],
    invariant to a common nonzero scaling of a and b. Never substitute logits.
    """
    key, values_key = ("topk_ids", "topk_weights") if kind == "topk" else ("active_ids", "active_weights")
    li, ri = left[key], right[key]
    shared = sorted(set(li) & set(ri))
    if not shared:
        return None, "No shared prototypes"
    lv, rv = left.get(values_key, []), right.get(values_key, [])
    if len(li) != len(lv) or len(ri) != len(rv):
        return None, "Native activation strengths missing; run cache_similarity"
    lm, rm = dict(zip(li, lv)), dict(zip(ri, rv))
    terms = []
    for p in shared:
        a, b = lm[p], rm[p]
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            return None, "Native activation strength unavailable or non-finite"
        # Scale first to avoid overflow and retain scale invariance near zero.
        scale = max(abs(a), abs(b))
        a, b = (a / scale, b / scale) if scale else (0., 0.)
        denom = abs(a) + abs(b)
        terms.append(abs(a - b) / denom if denom else 0.)
    return float(np.mean(terms)), None


def cosine(a, b, flat_equal=False):
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        if not flat_equal:
            return None
        return 1. if na < 1e-12 and nb < 1e-12 else 0.
    return float(np.clip(np.dot(a / na, b / nb), -1, 1))


def visual_metrics(left, right):
    if left is None or right is None:
        return {"visual_score": None, "similarity_missing_reason": "Stage4/RGB descriptors missing; run ACTION='cache_similarity'"}
    fa, ra = left
    fb, rb = right
    fc = cosine(fa, fb)
    if fc is None or ra.shape != rb.shape or ra.size % 3:
        return {"visual_score": None, "similarity_missing_reason": "Zero or incompatible stage4 feature / RGB descriptors"}
    mae = float(np.mean(np.abs(ra - rb)))
    aa, bb = ra.reshape(-1, 3), rb.reshape(-1, 3)
    sa = (aa - aa.mean(axis=0, keepdims=True)).reshape(-1)
    sb = (bb - bb.mean(axis=0, keepdims=True)).reshape(-1)
    sc = cosine(sa, sb, flat_equal=True)
    if sc is None:
        return {"visual_score": None, "similarity_missing_reason": "Invalid RGB structure descriptor"}
    return {"stage4_cosine": fc, "rgb_mae": mae, "rgb_structure_cosine": sc,
            "visual_score": fc + (1. - mae) + sc, "similarity_missing_reason": None}


def annotate_edges(con, nodes, edges, all_pairs=False, kind=GRAPH_KIND):
    load_strengths(con, nodes)
    by_id = {n["id"]: n for n in nodes}
    descriptors = {}
    if has_table(con, "visual_descriptors"):
        for n in nodes:
            row = con.execute("SELECT feature_dim,stage4_vector,rgb_vector FROM visual_descriptors WHERE node_id=?",
                              (n["id"],)).fetchone()
            if row:
                feature = unpack_vector(row[1])
                if len(feature) != row[0]:
                    raise ValueError(f"Corrupt stage4 descriptor for node {n['id']}")
                descriptors[n["id"]] = (feature, unpack_vector(row[2]))
    if all_pairs:
        pairs_to_score = [(a["id"], b["id"]) for i, a in enumerate(nodes)
                          for b in nodes[i+1:] if a["stage"] == b["stage"]]
    else:
        pairs_to_score = [(e["source"], e["target"]) for e in edges]
    scored, lookup = [], {}
    for a, b in pairs_to_score:
        values = visual_metrics(descriptors.get(a), descriptors.get(b))
        if by_id[a]["stage"] != "stage4" or by_id[b]["stage"] != "stage4":
            values = {"visual_score": None, "similarity_missing_reason": "Visual metric is defined for stage4 patches only"}
        for graph_kind in ("topk", "active"):
            diff, reason = activation_difference(by_id[a], by_id[b], graph_kind)
            values[graph_kind + "_activation_difference"] = diff
            values[graph_kind + "_activation_missing_reason"] = reason
        item = {"source": a, "target": b, **values}
        scored.append(item)
        lookup[tuple(sorted((a, b)))] = item
    for edge in edges:
        item = lookup[tuple(sorted((edge["source"], edge["target"])))]
        edge.update({k: v for k, v in item.items() if k not in ("source", "target")})
        edge["activation_difference"] = item[kind + "_activation_difference"]
        edge["activation_missing_reason"] = item[kind + "_activation_missing_reason"]
    scores = [e["visual_score"] for e in edges if e["visual_score"] is not None]
    if EDGE_COLOR_RANGE is not None:
        low, high = map(float, EDGE_COLOR_RANGE)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("EDGE_COLOR_RANGE must be a finite increasing (low, high) pair.")
    else:
        low, high = (min(scores), max(scores)) if scores else (0., 3.)
    return scored, {"formula": SIMILARITY_FORMULA, "color_min": low, "color_max": high,
                    "normalization": "fixed" if EDGE_COLOR_RANGE is not None else "displayed-edge min/max",
                    "valid_edges": len(scores), "missing_edges": len(edges) - len(scores),
                    "structure_definition": "per-channel spatially centered RGB cosine; both flat=1, one flat=0",
                    "activation_difference": "mean shared-prototype |a-b|/(|a|+|b|); zero/zero=0"}


class Stage4Capture:
    """Capture existing backbone computations, including direct method calls.

    nn.Module forward hooks do not observe module.forward(...) or calls to
    forward_features(...). Temporarily instrument these methods as well as the
    module-call path. Each wrapper calls the original method exactly once and
    returns its exact output; close() restores the previous instance state.
    Output-path errors are deferred so an inner helper with a different return
    format cannot interrupt a valid outer forward returning {"stage4": tensor}.
    """
    def __init__(self, model):
        modules = dict(model.named_modules())
        if STAGE4_FEATURE_MODULE not in modules:
            names = [n for n in modules if n][:80]
            raise RuntimeError(f"Stage4 feature module {STAGE4_FEATURE_MODULE!r} not found. "
                               f"Set STAGE4_FEATURE_MODULE to the native backbone/stage4 module; available: {names}")
        self.module = modules[STAGE4_FEATURE_MODULE]
        self.value = None
        self.source = None
        self.calls = []
        self.output_errors = []
        self._source_reported = False
        self._original_methods = {}
        self._child_names = [name for name in modules if name.startswith(STAGE4_FEATURE_MODULE + ".")][:40]
        self.handle = self.module.register_forward_hook(self._capture)
        try:
            for name in ("forward", "forward_features", "extract_features"):
                original = getattr(self.module, name, None)
                if not callable(original):
                    continue
                self._original_methods[name] = (name in self.module.__dict__, self.module.__dict__.get(name))
                def wrapped(*args, _original=original, _name=name, **kwargs):
                    output = _original(*args, **kwargs)
                    self._observe(output, STAGE4_FEATURE_MODULE + "." + _name)
                    return output
                setattr(self.module, name, wrapped)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _describe(output):
        if isinstance(output, dict):
            return "dict keys=" + repr(list(output.keys())[:20])
        if isinstance(output, (tuple, list)):
            entries = [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for v in output[:8]]
            return type(output).__name__ + " entries=" + repr(entries)
        if hasattr(output, "shape"):
            return type(output).__name__ + " shape=" + repr(tuple(output.shape))
        return type(output).__name__

    def _observe(self, output, source):
        self.calls.append(source + " -> " + self._describe(output))
        self.calls = self.calls[-12:]
        value = output
        try:
            for part in STAGE4_FEATURE_OUTPUT_PATH:
                value = value[part]
            if not callable(getattr(value, "detach", None)):
                raise TypeError("selected value is not a tensor")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            self.output_errors.append(source + ": " + type(exc).__name__ + " " + str(exc))
            self.output_errors = self.output_errors[-8:]
            return
        self.value = value.detach()
        self.source = source

    def _capture(self, module, inputs, output):
        self._observe(output, STAGE4_FEATURE_MODULE + ".__call__")
        # Return None: never substitute or alter the model's output.

    def reset(self):
        self.value = None
        self.source = None
        self.calls.clear()
        self.output_errors.clear()

    def last_frame(self, info, batch_size, layout=None):
        if self.value is None:
            calls = "\n  ".join(getattr(self, "calls", [])) or "No configured backbone method was called."
            errors = "\n  ".join(getattr(self, "output_errors", [])) or "None"
            raise RuntimeError(
                "Native stage4 features were not captured. Module hooks and direct "
                "forward/forward_features/extract_features calls are now supported.\n"
                f"Configured module: {STAGE4_FEATURE_MODULE!r}; output path: {STAGE4_FEATURE_OUTPUT_PATH!r}\n"
                f"Observed calls and return formats:\n  {calls}\n"
                f"Output selection errors:\n  {errors}\n"
                "If the selected native stage4 module returns a tensor, set "
                "STAGE4_FEATURE_OUTPUT_PATH=(). If it returns a tuple/list, set the correct "
                "index path explicitly. Do not substitute prototype logits for backbone features.\n"
                f"Backbone child module names: {getattr(self, '_child_names', [])}\n"
                "If no method was called, inspect the actual backbone call in model.py and "
                "set STAGE4_FEATURE_MODULE to that native feature-producing module.")
        a = self.value.detach().float().cpu().numpy()
        b, t, h, w = batch_size, info["t"], info["h"], info["w"]

        def temporal_ok(feature_t):
            # Concept outputs may report T=1 after last-frame selection while the
            # backbone still carries the full clip length; always take index -1.
            return feature_t >= 1 and (feature_t == t or t == 1)

        layouts = {}
        if a.ndim == 5:
            # BCTHW: (B, C, T, H, W) — VideoSwin / this project's backbone convention.
            if a.shape[0] == b and a.shape[3:] == (h, w) and temporal_ok(int(a.shape[2])):
                layouts["BCTHW"] = lambda: a[:, :, -1].transpose(0, 2, 3, 1)
            # BTHWC: (B, T, H, W, C)
            if a.shape[0] == b and a.shape[2:4] == (h, w) and temporal_ok(int(a.shape[1])):
                layouts["BTHWC"] = lambda: a[:, -1]
            # BTCHW: (B, T, C, H, W)
            if a.shape[0] == b and a.shape[3:] == (h, w) and temporal_ok(int(a.shape[1])):
                layouts["BTCHW"] = lambda: a[:, -1].transpose(0, 2, 3, 1)
        # A last-frame-only feature is accepted only when metadata also has T=1.
        if a.ndim == 4 and t == 1:
            if a.shape[0] == b and a.shape[2:] == (h, w):
                layouts["BCHW"] = lambda: a.transpose(0, 2, 3, 1)
            if a.shape[:3] == (b, h, w):
                layouts["BHWC"] = lambda: a
        choice = layout or STAGE4_FEATURE_LAYOUT
        if choice == "auto":
            if not layouts:
                raise RuntimeError(f"Ambiguous/incompatible stage4 shape {a.shape}; expected B={b},T={t},H={h},W={w}. "
                                   f"Possible layouts: []. Set STAGE4_FEATURE_LAYOUT explicitly.")
            if len(layouts) > 1:
                exact = {}
                if a.ndim == 5:
                    if "BCTHW" in layouts and int(a.shape[2]) == t:
                        exact["BCTHW"] = layouts["BCTHW"]
                    if "BTHWC" in layouts and int(a.shape[1]) == t:
                        exact["BTHWC"] = layouts["BTHWC"]
                    if "BTCHW" in layouts and int(a.shape[1]) == t:
                        exact["BTCHW"] = layouts["BTCHW"]
                if len(exact) == 1:
                    layouts = exact
                elif "BCTHW" in layouts:
                    # Prefer the project's backbone layout when T was collapsed in concept metadata.
                    layouts = {"BCTHW": layouts["BCTHW"]}
                else:
                    raise RuntimeError(f"Ambiguous/incompatible stage4 shape {a.shape}; expected B={b},T={t},H={h},W={w}. "
                                       f"Possible layouts: {list(layouts)}. Set STAGE4_FEATURE_LAYOUT explicitly.")
            choice = next(iter(layouts))
        if choice not in layouts:
            raise RuntimeError(f"Stage4 shape {a.shape} does not match configured layout {choice} and native prototype grid.")
        result = layouts[choice]()
        if not np.isfinite(result).all():
            raise ValueError("Non-finite native stage4 features.")
        if getattr(self, "source", None) and not getattr(self, "_source_reported", False):
            print(f"Stage4 descriptor source: {self.source}; path={STAGE4_FEATURE_OUTPUT_PATH}; "
                  f"layout={choice}; raw shape={tuple(a.shape)}; concept T={t}", flush=True)
            self._source_reported = True
        return result

    def close(self):
        if getattr(self, "handle", None) is not None:
            self.handle.remove()
            self.handle = None
        for name, (was_instance_attribute, previous) in getattr(self, "_original_methods", {}).items():
            if was_instance_attribute:
                setattr(self.module, name, previous)
            else:
                delattr(self.module, name)
        self._original_methods = {}
        self.value = None

VIEWER_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patch prototype overlap explorer</title><style>
:root{font-family:system-ui,sans-serif;color:#172236;background:#f4f6fa}*{box-sizing:border-box}
body{margin:0}header{padding:18px 24px;background:white;border-bottom:1px solid #dce1e8}
h1{font-size:22px;margin:0 0 6px}p{margin:6px 0;line-height:1.5}.muted{color:#5c697b;font-size:13px}
.controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin-top:14px}label{font-size:12px;display:grid;gap:5px}
input,select,button{border:1px solid #b7c3d4;border-radius:6px;padding:7px;background:white;color:#172236}
button{cursor:pointer}button.primary{background:#244c7c;color:white}.shell{display:grid;grid-template-columns:minmax(0,1fr) 350px}
#scene{height:68vh;min-height:430px;position:relative;background:#fff}canvas{width:100%;height:100%;touch-action:none}
aside{padding:18px;max-height:78vh;overflow:auto;border-left:1px solid #dce1e8;background:#fafbfd}
.pill{display:inline-block;background:#e3eaf3;border-radius:5px;padding:4px 6px;margin:3px;font-size:12px}
.frame{position:relative;margin-top:10px}.frame img{width:100%;display:block}.box{position:absolute;border:3px solid #ef6638;pointer-events:none}
.crop{max-width:160px;max-height:120px;image-rendering:auto;border:1px solid #dce1e8;margin-top:8px}
#status{padding:10px 24px;min-height:40px;border-bottom:1px solid #dce1e8}.legend{padding:10px 24px;font-size:13px}
.gradient{display:inline-block;width:150px;height:10px;background:linear-gradient(90deg,#427ca8,#e29e58,#d15a45);margin:0 7px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px;border-bottom:1px solid #dce1e8;text-align:left}
#rows{max-height:240px;overflow:auto;margin:0 24px 24px}tr[data-id]{cursor:pointer}tr[data-id]:hover{background:#e7eef7}
code{font-size:12px;word-break:break-all}a{color:#244c7c}@media(max-width:850px){.shell{grid-template-columns:1fr}aside{max-height:500px;border-left:0}}
.patch-gallery{width:min(1500px,97vw);max-width:97vw;max-height:92vh;padding:0;border:1px solid #b7c3d4;border-radius:12px;color:#172236;background:#f7f9fc;box-shadow:0 20px 80px #0005}
.patch-gallery::backdrop{background:rgba(15,25,42,.65)}
.gallery-header{padding:16px 20px;background:white;border-bottom:1px solid #dce1e8;position:sticky;top:0;z-index:2}
.gallery-heading{display:flex;justify-content:space-between;align-items:center;gap:16px}.gallery-heading h2{font-size:19px;margin:0}
.gallery-scroll{overflow:auto;padding:14px}.patch-grid{display:grid;grid-template-columns:repeat(10,minmax(104px,1fr));grid-template-rows:repeat(5,auto);gap:10px;list-style:none;padding:0;margin:0;min-width:1130px}
.gallery-card{min-width:0;border:1px solid #d8e0eb;border-radius:8px;background:white;overflow:hidden}.gallery-card.is-selected{border:3px solid #ef6638;background:#fff5ec}
.gallery-card button{display:block;width:100%;height:100%;padding:7px;border:0;border-radius:0;background:transparent;text-align:left;color:inherit}
.gallery-card button:hover{background:#eef4fc}.gallery-card button:focus-visible{outline:3px solid #244c7c;outline-offset:-3px}
.gallery-rank{display:block;min-height:20px;font-size:11px;font-weight:700;color:#244c7c}.is-selected .gallery-rank{color:#a94416}
.gallery-crop{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#e9edf3;border-radius:4px}
.gallery-no-image{display:flex;align-items:center;justify-content:center;font-size:12px;color:#657287;text-align:center}
.gallery-node{display:block;font-weight:650;font-size:12px;margin-top:6px}.gallery-info{display:block;font-size:10px;color:#5c697b;line-height:1.4;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.gallery-empty{min-height:150px;border:1px dashed #d0d8e5;border-radius:8px;background:#eef2f7}.gallery-note{font-size:12px;color:#5c697b;margin-bottom:0}
.edge-gradient{display:inline-block;width:160px;height:12px;background:linear-gradient(90deg,rgb(33,102,172),rgb(178,24,43));margin:0 7px} .edge-score{display:block;font-size:11px;color:#5c697b;line-height:1.5}
</style></head><body>
<header><h1>Patch prototype overlap explorer</h1>
<p class="muted">Nodes are patch occurrences. Edge weight = shared prototype count. Edge color shows measured visual similarity. Choose how prototype overlap and activation differences set layout distances.</p>
<div class="controls">
<label>Overlap definition<select id="kind"><option value="topk">Top-eight similarity candidates</option><option value="active">Forward-valid prototypes</option></select></label>
<label>Minimum overlap<input id="threshold" type="number" min="1" max="512" value="1" style="width:85px"></label>
<label>Seed node ID<input id="seed" type="number" min="0" style="width:105px"></label>
<label>Max displayed nodes<input id="limit" type="number" min="1" max="600" style="width:100px"></label>
<label>Neighbor filter<select id="cross"><option value="1">Other videos</option><option value="0">Any video</option></select></label>
<button id="load" class="primary">Load neighborhood</button><button id="fit">Reset layout</button>
<label>Node distance<select id="distanceMode"><option value="overlap">Prototype overlap count only</option><option value="activation">Overlap count + normalized activation difference</option></select></label>
<label>Activation distance weight<input id="activationWeight" type="number" min="0" max="500" step="10" value="100" style="width:90px"></label>
<label>Edges<select id="edgeMode"><option value="selected">Touching selected node</option><option value="all">All visible edges</option><option value="none">Hide edges</option></select></label>
</div><div class="controls"><label>Find video / frame / stage<input id="query" placeholder="e.g. 0301 or stage4"></label><button id="search">Search nodes</button><span id="mode" class="muted"></span></div></header>
<div id="status"></div><div class="shell"><div id="scene"><canvas id="canvas"></canvas></div><aside id="details"></aside></div>
<div class="legend">Within-frame predicted score percentile: low <span class="gradient"></span> high. This is a rank, not a fixation probability. Orange outline: seed; dark outline: selected. Scroll to zoom; drag a node or the background.</div>
<div class="legend"><b>Edge visual similarity:</b> low <span class="edge-gradient"></span> high <span id="edgeLegend"></span><br><span id="distanceLegend" class="muted"></span></div>
<div id="edgeTooltip" role="tooltip" style="display:none;position:fixed;pointer-events:none;background:#172236;color:white;padding:9px 12px;border-radius:7px;font-size:12px;z-index:10;max-width:360px;white-space:pre-line"></div>
<div id="rows"></div>
<dialog id="patchGallery" class="patch-gallery" aria-labelledby="galleryTitle" aria-describedby="gallerySummary">
<div class="gallery-header"><div class="gallery-heading"><h2 id="galleryTitle">Top 50 patches</h2><button id="closeGallery" type="button" aria-label="Close patch gallery">Close</button></div><p id="gallerySummary" class="gallery-note" role="status" aria-live="polite"></p></div>
<div class="gallery-scroll"><ol id="galleryGrid" class="patch-grid" aria-label="Selected patch followed by strongest overlapping patches"></ol></div>
</dialog><script>
const INITIAL=__PAYLOAD__;
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data=INITIAL,nodes=[],edges=[],byId=new Map(),selected=null,zoom=1,panX=0,panY=0,drag=null,hover=null,settle=0;
const online=false;
const canvas=$('canvas'),ctx=canvas.getContext('2d');let width=800,height=550;
$('mode').textContent=online?'Live local explorer: searches the complete saved graph.':'Offline snapshot: graph and images embedded; use ACTION="serve" for complete-graph search.';
function color(p){p=Math.max(0,Math.min(1,p));return p<.5?`rgb(${66+Math.round(p*320)},${124+Math.round(p*68)},${168-Math.round(p*160)})`:`rgb(${226-Math.round((p-.5)*34)},${158-Math.round((p-.5)*136)},${88-Math.round((p-.5)*38)})`;}
const pairKey=(a,b)=>Math.min(a,b)+':'+Math.max(a,b);
const snapshotPairs=new Map((INITIAL.visual_similarity_pairs||[]).map(e=>[pairKey(e.source,e.target),e]));
let edgeMin=0,edgeMax=3;
function pairInfo(a,b){return snapshotPairs.get(pairKey(a,b))||{};}
function formatScore(v){return v==null||!Number.isFinite(v)?'unavailable':Number(v).toFixed(4);}
function edgeColor(score){if(score==null||!Number.isFinite(score))return 'rgba(128,136,148,.65)';const t=edgeMax>edgeMin?Math.max(0,Math.min(1,(score-edgeMin)/(edgeMax-edgeMin))):.5;return `rgba(${Math.round(33+145*t)},${Math.round(102-78*t)},${Math.round(172-129*t)},.8)`;}
function edgeInfo(e){return `visual ${formatScore(e.visual_score)} = stage4 ${formatScore(e.stage4_cosine)} + (1 - RGB MAE ${formatScore(e.rgb_mae)}) + structure ${formatScore(e.rgb_structure_cosine)}; activation difference ${formatScore(e.activation_difference)}`;}
function targetDistance(e){const overlap=35+180/(1+e.weight);return overlap+($('distanceMode').value==='activation'&&e.activation_difference!=null?Number($('activationWeight').value)*e.activation_difference:0);}
function updateLegends(){const valid=edges.map(e=>e.visual_score).filter(v=>v!=null&&Number.isFinite(v)),m=data.visual_similarity||{};if(m.normalization==='fixed'){edgeMin=m.color_min;edgeMax=m.color_max;}else{edgeMin=valid.length?valid.reduce((a,b)=>Math.min(a,b),Infinity):0;edgeMax=valid.length?valid.reduce((a,b)=>Math.max(a,b),-Infinity):3;}
$('edgeLegend').textContent=`${edgeMin.toFixed(3)} to ${edgeMax.toFixed(3)} (${m.normalization||'displayed-edge min/max'}); ${edges.length-valid.length} unavailable edges shown grey. Equal scores share one midpoint color.`;
const missing=edges.filter(e=>e.activation_difference==null).length;
$('distanceLegend').textContent=$('distanceMode').value==='activation'?`Desired edge length = 35 + 180/(1 + shared prototype count) + ${Number($('activationWeight').value)} × mean normalized activation difference. Difference = mean |a-b|/(|a|+|b|) over shared prototypes; zero/zero = 0. ${missing?missing+' edges lack native activations and use overlap-only length (dashed); run cache_similarity.':''}`:'Desired edge length = 35 + 180/(1 + shared prototype count). More overlap pulls nodes closer. Force layout approximates these lengths; positions also reflect repulsion and other edges.';}
function changeDistances(){const value=Number($('activationWeight').value);if(!Number.isFinite(value)||value<0||value>500){$('activationWeight').value=100;}updateLegends();settle=240;showDetails();}
function edgeHit(p){let best=null,bestDist=6/zoom;const mode=$('edgeMode').value;if(mode==='none')return null;for(const e of edges){if(mode==='selected'&&e.a!==selected&&e.b!==selected)continue;const dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,len=dx*dx+dy*dy;if(!len)continue;const t=Math.max(0,Math.min(1,((p.x-e.a.x)*dx+(p.y-e.a.y)*dy)/len));const d=Math.hypot(p.x-e.a.x-t*dx,p.y-e.a.y-t*dy);if(d<bestDist){best=e;bestDist=d;}}return best;}
function showEdgeTooltip(e,event){const box=$('edgeTooltip');if(!e){box.style.display='none';return;}box.style.display='block';box.style.left=Math.max(8,Math.min(innerWidth-365,event.clientX+14))+'px';box.style.top=Math.max(8,Math.min(innerHeight-150,event.clientY+14))+'px';box.textContent=`Nodes ${e.source} ↔ ${e.target}\nShared prototypes: ${e.weight}\n${edgeInfo(e)}\nDesired layout length: ${targetDistance(e).toFixed(2)}${e.similarity_missing_reason?'\n'+e.similarity_missing_reason:''}${e.activation_missing_reason?'\n'+e.activation_missing_reason:''}`;}

function loadData(d){data=d;$('kind').value=d.kind;$('seed').value=d.seed;$('limit').value=d.display_limit;$('threshold').value=d.min_overlap;$('cross').value=d.cross_video_neighbors?'1':'0';
 nodes=d.nodes.map((n,i)=>({...n,x:Math.cos(i*2.399)*Math.sqrt(i)*14,y:Math.sin(i*2.399)*Math.sqrt(i)*14,vx:0,vy:0}));byId=new Map(nodes.map(n=>[n.id,n]));
 edges=d.edges.map(e=>{const m={...pairInfo(e.source,e.target),...e};return {...m,activation_difference:m[d.kind+'_activation_difference']??m.activation_difference??null,activation_missing_reason:m[d.kind+'_activation_missing_reason']??m.activation_missing_reason??null,a:byId.get(e.source),b:byId.get(e.target)};});updateLegends();zoom=1;panX=panY=0;settle=180;selected=byId.get(d.seed)||nodes[0];
 $('status').textContent=`${nodes.length} of ${d.total_nodes.toLocaleString()} stored nodes shown · ${edges.length.toLocaleString()} displayed edges · ${d.kind==='topk'?'similarity-candidate overlap':'forward-valid overlap'}${d.neighbor_results_capped?' · neighborhood capped':''}. ${d.scope}`;
 showDetails();showTable(nodes);}
function showDetails(){const n=selected;if(!n){$('details').textContent='Select a node.';return;}const key=$('kind').value==='topk'?'topk_ids':'active_ids';
 const ns=edges.filter(e=>e.source===n.id||e.target===n.id).sort((a,b)=>b.weight-a.weight).slice(0,15);
 $('details').innerHTML=`<h2 style="font-size:18px;margin-top:0">Patch ${n.id}</h2><p>${esc(n.dataset)} / ${esc(n.video)} / ${esc(n.target_name)}</p><p class="muted">${esc(n.stage)} · row ${n.row}, column ${n.col} · grid ${n.grid_h}×${n.grid_w} · window ${n.window_start}…${n.target_index} (indices zero-based)</p>
 ${n.crop_image?`<img class="crop" src="${n.crop_image}" alt="Patch crop">`:''}
 ${n.frame_image?`<div class="frame"><img src="${n.frame_image}" alt="Full input frame"><div class="box" style="left:${100*n.col/n.grid_w}%;top:${100*n.row/n.grid_h}%;width:${100/n.grid_w}%;height:${100/n.grid_h}%"></div></div>`:'<p class="muted">Frame thumbnail unavailable.</p>'}
 <p>Predicted patch score: <b>${Number(n.pred_score).toPrecision(5)}</b><br>Within-frame percentile: <b>${(100*n.pred_percentile).toFixed(1)}%</b>${n.gt_score==null?'':`<br>GT patch mean: ${Number(n.gt_score).toPrecision(4)} · rank ${(100*n.gt_percentile).toFixed(1)}%`}</p>
 <p><b>Top similarity candidates</b></p>${n.topk_ids.map(p=>`<span class="pill">${p}</span>`).join('')}
 <p><b>Forward-valid prototypes</b></p>${n.active_ids.map(p=>`<span class="pill">${p}</span>`).join('')||'<span class="muted">None</span>'}
 <p class="muted">Current overlap uses ${esc(key)}. Top candidates can be inactive.</p><button id="recenter">Use this node as seed</button> <button id="topPatches" type="button">${online?'Top 50 patches':'Top 50 in snapshot'}</button>
 <p><b>Strongest displayed neighbors</b></p>${ns.map(e=>{const o=e.a.id===n.id?e.b:e.a;const shared=n[key].filter(p=>o[key].includes(p));return `<p><button data-select="${o.id}">Node ${o.id}</button> overlap ${e.weight}: ${shared.join(', ')}<span class="edge-score">${esc(edgeInfo(e))}<br>Desired layout length: ${targetDistance(e).toFixed(2)}</span></p>`;}).join('')}
 <p class="muted">Source: <code>${esc(n.source_path)}</code></p>`;
 $('recenter').onclick=()=>{$('seed').value=n.id;fetchNeighborhood();};
 $('topPatches').onclick=()=>openPatchGallery(n);
 $('details').querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>{selected=byId.get(Number(b.dataset.select));showDetails();});}

let galleryRequest=0,galleryReturnFocus=null;
const galleryDialog=$('patchGallery');
function closePatchGallery(){galleryRequest++;if(galleryDialog.open)galleryDialog.close();if(galleryReturnFocus&&galleryReturnFocus.isConnected)galleryReturnFocus.focus();}
$('closeGallery').onclick=closePatchGallery;
galleryDialog.addEventListener('cancel',()=>{galleryRequest++;});
galleryDialog.addEventListener('click',event=>{if(event.target===galleryDialog){const r=galleryDialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)closePatchGallery();}});
function rankSnapshotGallery(anchor,kind,minOverlap,cross){const key=kind==='topk'?'topk_ids':'active_ids',ids=new Set(anchor[key]);
 const ranked=INITIAL.nodes.filter(n=>n.id!==anchor.id&&n.stage===anchor.stage&&(!cross||n.dataset!==anchor.dataset||n.video!==anchor.video))
 .map(n=>{const shared=[...new Set(n[key])].filter(p=>ids.has(p)).sort((a,b)=>a-b);return{...n,gallery_selected:false,edge_strength:shared.length,shared_prototypes:shared};})
 .filter(n=>n.edge_strength>=minOverlap).sort((a,b)=>b.edge_strength-a.edge_strength||a.id-b.id).slice(0,49);
 return{selected_node:anchor.id,patches:[{...anchor,gallery_selected:true,edge_strength:null,shared_prototypes:[...ids].sort((a,b)=>a-b)},...ranked],kind,min_overlap:minOverlap,cross_video:cross,scope:'offline_snapshot',num_neighbors:ranked.length,total_slots:50};}
function renderPatchGallery(result){const mode=result.kind==='topk'?'top-eight candidate overlap':'forward-valid prototype overlap',scope=result.scope==='complete_database'?'Ranked across the complete saved graph.':'Ranked only within the original offline snapshot; use ACTION="serve" for complete-graph ranking.';
 $('galleryTitle').textContent=`Top 50 patches · selected node ${result.selected_node}`;
 $('gallerySummary').textContent=`Selected patch first, followed by ${result.num_neighbors} neighbors in decreasing edge strength (ties: node ID). ${mode}; minimum overlap ${result.min_overlap}; ${result.cross_video?'other videos only':'any video'}. ${scope} ${result.patches.length<50?'Empty cells indicate fewer matching patches.':''}`;
 const cards=result.patches.map((n,i)=>`<li class="gallery-card${i===0?' is-selected':''}"><button type="button" data-gallery-node="${n.id}" title="${esc(n.dataset+' / '+n.video+' / '+n.target_name+'; shared prototypes: '+n.shared_prototypes.join(', '))}"><span class="gallery-rank">${i===0?'Selected patch':`#${i} · overlap ${n.edge_strength}`}</span>${n.crop_image?`<img class="gallery-crop" loading="lazy" src="${n.crop_image}" alt="Patch ${n.id}, ${esc(n.video)}, cell ${n.row},${n.col}">`:'<span class="gallery-crop gallery-no-image">No thumbnail</span>'}<span class="gallery-node">Node ${n.id}</span><span class="gallery-info">${esc(n.video)} / ${esc(n.target_name)}</span><span class="gallery-info">${esc(n.stage)} · cell ${n.row},${n.col}</span>${i===0?'':`<span class="gallery-info">Visual score ${formatScore((n.visual_similarity||pairInfo(result.selected_node,n.id)).visual_score)}</span>`}<span class="gallery-info">Predicted percentile ${(100*n.pred_percentile).toFixed(1)}%</span></button></li>`);
 while(cards.length<50)cards.push('<li class="gallery-empty" aria-hidden="true"></li>');
 $('galleryGrid').innerHTML=cards.join('');
 $('galleryGrid').querySelectorAll('[data-gallery-node]').forEach(button=>button.onclick=()=>{const id=Number(button.dataset.galleryNode);closePatchGallery();if(byId.has(id)){selected=byId.get(id);showDetails();}else{$('seed').value=id;fetchNeighborhood();}});}
async function openPatchGallery(anchor){const request=++galleryRequest;galleryReturnFocus=document.activeElement;$('galleryTitle').textContent=`Top 50 patches · selected node ${anchor.id}`;$('gallerySummary').textContent='Loading strongest prototype overlaps…';$('galleryGrid').innerHTML='';if(!galleryDialog.open)galleryDialog.showModal();
 try{const kind=data.kind,minOverlap=Math.max(1,Number(data.min_overlap)),cross=Boolean(data.cross_video_neighbors);let result;
 if(online){const q=new URLSearchParams({node:anchor.id,kind,min_overlap:minOverlap,cross:Number(cross)});const response=await fetch('/api/top-patches?'+q);result=await response.json();if(!response.ok)throw Error(result.error||'Could not load patch gallery.');}
 else result=rankSnapshotGallery(anchor,kind,minOverlap,cross);
 if(request===galleryRequest&&galleryDialog.open)renderPatchGallery(result);
 }catch(error){if(request===galleryRequest&&galleryDialog.open)$('gallerySummary').textContent=error.message;}}
function showTable(list){$('rows').innerHTML=`<table><thead><tr><th>Node</th><th>Video / frame</th><th>Stage · cell</th><th>Predicted percentile</th><th>Top candidates</th><th>Forward-valid</th></tr></thead><tbody>${list.slice(0,200).map(n=>`<tr data-id="${n.id}"><td>${n.id}</td><td>${esc(n.video)} / ${esc(n.target_name)}</td><td>${esc(n.stage)} · ${n.row},${n.col}</td><td>${(100*n.pred_percentile).toFixed(1)}%</td><td>${n.topk_ids.join(', ')}</td><td>${n.active_ids.join(', ')}</td></tr>`).join('')}</tbody></table>`;
 $('rows').querySelectorAll('[data-id]').forEach(tr=>tr.onclick=()=>{const id=Number(tr.dataset.id);if(byId.has(id)){selected=byId.get(id);showDetails();}else{$('seed').value=id;fetchNeighborhood();}});}
async function fetchNeighborhood(){try{const kind=$('kind').value,t=Math.max(1,Number($('threshold').value)),seed=Number($('seed').value),limit=Number($('limit').value),cross=$('cross').value==='1';
 if(online){$('status').textContent='Querying exact overlaps…';const q=new URLSearchParams({seed,kind,min_overlap:t,limit,cross:Number(cross)});const r=await fetch('/api/graph?'+q);const d=await r.json();if(!r.ok)throw Error(d.error);loadData(d);}
 else{const all=INITIAL.nodes;const root=all.find(n=>n.id===seed);if(!root)throw Error('This node is outside the offline snapshot. Use ACTION="serve" to query it.');const key=kind==='topk'?'topk_ids':'active_ids';const candidates=all.filter(n=>n.id!==seed&&n.stage===root.stage&&(!cross||n.dataset!==root.dataset||n.video!==root.video)).map(n=>({n,w:n[key].filter(p=>root[key].includes(p)).length})).filter(x=>x.w>=t).sort((a,b)=>b.w-a.w||a.n.id-b.n.id).slice(0,Math.max(0,limit-1));const keep=[root,...candidates.map(x=>x.n)];const ee=[];for(let i=0;i<keep.length;i++)for(let j=i+1;j<keep.length;j++){if(keep[i].stage!==keep[j].stage)continue;const w=keep[i][key].filter(p=>keep[j][key].includes(p)).length;if(w>=t)ee.push({...pairInfo(keep[i].id,keep[j].id),source:keep[i].id,target:keep[j].id,weight:w});}loadData({...INITIAL,nodes:keep,edges:ee,seed,kind,min_overlap:t,display_limit:limit,cross_video_neighbors:cross,scope:'Filtered within the original offline snapshot only; other graph neighbors may exist.'});}
 }catch(e){$('status').textContent=e.message;}}
$('distanceMode').onchange=changeDistances;$('activationWeight').onchange=changeDistances;
$('load').onclick=fetchNeighborhood;$('kind').onchange=fetchNeighborhood;
$('fit').onclick=()=>loadData(data);$('edgeMode').onchange=()=>{settle=Math.max(settle,1);};
$('search').onclick=async()=>{try{const q=$('query').value.trim();if(online){const r=await fetch('/api/search?q='+encodeURIComponent(q));const d=await r.json();if(!r.ok)throw Error(d.error);showTable(d.nodes);$('status').textContent=`Search results: ${d.nodes.length} (at most 100); select a row to load its neighborhood.`;}else showTable(INITIAL.nodes.filter(n=>(n.video+' '+n.target_name+' '+n.stage+' '+n.id).includes(q)));}catch(e){$('status').textContent=e.message;}};
function point(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left-width/2-panX)/zoom,y:(e.clientY-r.top-height/2-panY)/zoom};}
function hit(p){return nodes.find(n=>Math.hypot(n.x-p.x,n.y-p.y)<10/zoom+4);}
canvas.onpointerdown=e=>{const p=point(e),n=hit(p);drag={n,lastX:e.clientX,lastY:e.clientY};if(n){selected=n;showDetails();}canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{const p=point(e);hover=hit(p);showEdgeTooltip(hover||drag?null:edgeHit(p),e);canvas.style.cursor=hover?'pointer':'grab';if(drag){if(drag.n){drag.n.x=p.x;drag.n.y=p.y;drag.n.vx=drag.n.vy=0;settle=Math.max(settle,90);}else{panX+=e.clientX-drag.lastX;panY+=e.clientY-drag.lastY;}drag.lastX=e.clientX;drag.lastY=e.clientY;}};
canvas.onpointerleave=()=>showEdgeTooltip(null);canvas.onpointerup=()=>{drag=null;};canvas.onpointercancel=()=>{drag=null;};
canvas.onwheel=e=>{e.preventDefault();const before=point(e);zoom=Math.max(.12,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));const r=canvas.getBoundingClientRect();panX=e.clientX-r.left-width/2-before.x*zoom;panY=e.clientY-r.top-height/2-before.y*zoom;};
function tick(){const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();width=rect.width;height=rect.height;if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
 if(settle>0){settle--;for(let i=0;i<nodes.length;i++){const a=nodes[i];a.vx-=a.x*.0006;a.vy-=a.y*.0006;for(let j=i+1;j<nodes.length;j++){const b=nodes[j],dx=a.x-b.x,dy=a.y-b.y,dd=dx*dx+dy*dy+35,f=150/dd;a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f;}}
 for(const e of edges){const dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,dist=Math.hypot(dx,dy)||1,f=(dist-targetDistance(e))*.0008;e.a.vx+=dx/dist*f;e.a.vy+=dy/dist*f;e.b.vx-=dx/dist*f;e.b.vy-=dy/dist*f;}
 for(const n of nodes){if(drag?.n===n)continue;n.vx*=.75;n.vy*=.75;n.x+=Math.max(-8,Math.min(8,n.vx));n.y+=Math.max(-8,Math.min(8,n.vy));}}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);ctx.translate(width/2+panX,height/2+panY);ctx.scale(zoom,zoom);
 const mode=$('edgeMode').value;for(const e of edges){if(mode==='none'||(mode==='selected'&&e.a!==selected&&e.b!==selected))continue;ctx.strokeStyle=edgeColor(e.visual_score);ctx.setLineDash(e.visual_score==null||($('distanceMode').value==='activation'&&e.activation_difference==null)?[4/zoom,3/zoom]:[]);ctx.lineWidth=(.4+e.weight*.27)/Math.sqrt(zoom);ctx.beginPath();ctx.moveTo(e.a.x,e.a.y);ctx.lineTo(e.b.x,e.b.y);ctx.stroke();}
 ctx.setLineDash([]);for(const n of nodes){ctx.beginPath();ctx.arc(n.x,n.y,n===selected?8:6,0,2*Math.PI);ctx.fillStyle=color(n.pred_percentile);ctx.fill();ctx.strokeStyle=n.id===data.seed?'#ef6638':n===selected?'#172236':'#fff';ctx.lineWidth=n===selected||n.id===data.seed?2.5:1;ctx.stroke();if(n===selected||n===hover||nodes.length<35){ctx.fillStyle='#172236';ctx.font='11px system-ui';ctx.fillText(n.id,n.x+10,n.y-6);}}
 requestAnimationFrame(tick);}
if(INITIAL.layout_defaults){$('distanceMode').value=INITIAL.layout_defaults.mode==='activation'?'activation':'overlap';$('activationWeight').value=INITIAL.layout_defaults.activation_weight;}
loadData(INITIAL);tick();
</script></body></html>'''


def write_viewer(directory, con=None, excluded_by_stage=None, frequency_filter=None):
    directory = Path(directory)
    own = con is None
    con = connect_db(directory / "graph.sqlite", readonly=True) if own else con
    try:
        if excluded_by_stage is None:
            excluded_by_stage, frequency_filter = resolve_topk_frequency_exclusion(directory, con=con)
        payload = make_payload(con, directory, excluded_by_stage=excluded_by_stage,
                               frequency_filter=frequency_filter)
        payload["layout_defaults"] = {"mode": LAYOUT_DISTANCE_MODE, "activation_weight": LAYOUT_ACTIVATION_WEIGHT}
        dump_json(directory / "neighborhood.json", payload)
        encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
        html = (VIEWER_HTML
                .replace("__PAYLOAD__", encoded)
                .replace("Top-eight similarity candidates", f"Top-{TOP_K} similarity candidates")
                .replace("top-eight candidate overlap", f"top-{TOP_K} candidate overlap"))
        (directory / "graph_viewer.html").write_text(html, encoding="utf-8")
        missing = payload["visual_similarity"]["missing_edges"]
        if missing:
            print(f"{missing} displayed edges have no complete visual descriptors (grey). "
                  "For an existing graph, set ACTION='cache_similarity' once.", flush=True)
        if frequency_filter:
            excluded_total = sum(stage["num_excluded"] for stage in frequency_filter["stages"].values())
            print(f"Viewer top-K frequency filter: exclude frequency > "
                  f"{frequency_filter['max_frequency']:g} ({excluded_total:,} prototypes).", flush=True)
    finally:
        if own:
            con.close()
    print(f"Viewer: {directory / 'graph_viewer.html'}", flush=True)


def load_evaluation():
    if not Path(EVALUATION_FILE).is_file():
        raise FileNotFoundError(f"Place this script in src/interpretability_exp/exp4_expl_report/ "
                                f"or edit SRC_DIR. Not found: {EVALUATION_FILE}")
    sys.path.insert(0, str(SRC_DIR.resolve()))
    spec = importlib.util.spec_from_file_location("_prototype_graph_evaluation", EVALUATION_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frame_names_for_video(dataset, root, video):
    mapping = getattr(dataset, "video_frame_names", None)
    if isinstance(mapping, dict) and video in mapping:
        return [Path(str(n)).name for n in mapping[video]]
    listing = getattr(dataset, "_list_frame_names", None)
    if callable(listing):
        import inspect
        if len(inspect.signature(listing).parameters) == 2:
            return [Path(str(n)).name for n in listing(str(Path(root) / video / "images"), str(Path(root) / video / "maps"))]
    raise RuntimeError("DatasetLoader must expose video_frame_names[video] or "
                       "_list_frame_names(images_dir, maps_dir). Cannot safely infer frame IDs.")


def unpack_batch(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) not in (5, 6):
        raise RuntimeError("Expected collate output (video_names,rgb,sal,[fix,]n_frames,valid_mask).")
    names, rgb, sal = batch[:3]
    n_frames, valid = batch[-2:]
    return list(names), rgb, sal, n_frames, valid


def extract_stage(stage_out, batch_size, stage):
    import torch
    required = ("active_visual_prototype_indices", "visual_validity_mask", "visual_concept_logits", "visual_metadata")
    missing = [key for key in required if key not in stage_out]
    if missing:
        raise RuntimeError(f"{stage}: missing {missing}; available={list(stage_out)}. "
                           "Expose native model outputs; do not infer validity from logits.")
    ids, mask = stage_out[required[0]], stage_out[required[1]]
    if ids.ndim != 5 or mask.shape != ids.shape or ids.shape[0] != batch_size:
        raise RuntimeError(f"{stage}: expected indices/mask [B,T,H,W,K], got {tuple(ids.shape)} / {tuple(mask.shape)}.")
    b, t, h, w, k = ids.shape
    fs = stage_out["visual_metadata"].get("feature_shape", {})
    if any(int(fs.get(key, -1)) != value for key, value in (("B", b), ("T", t), ("H", h), ("W", w))):
        raise RuntimeError(f"{stage}: tensor shape disagrees with visual_metadata: {fs}.")
    logits = stage_out["visual_concept_logits"]
    if logits.ndim != 2 or logits.shape[0] != b * t * h * w:
        raise RuntimeError(f"{stage}: expected logits [B*T*H*W,M], got {tuple(logits.shape)}.")
    m = int(logits.shape[-1])
    log = logits.detach().float().reshape(b, t, h, w, m)[:, -1]
    if not torch.isfinite(log).all():
        raise ValueError(f"{stage}: non-finite prototype logits.")
    vals, top = torch.topk(log, min(TOP_K, m), dim=-1)
    native_ids = ids[:, -1].detach().long().cpu().numpy()
    native_mask = mask[:, -1].detach().bool().cpu().numpy()
    if np.any(native_ids[native_mask] < 0) or np.any(native_ids[native_mask] >= m):
        raise ValueError(f"{stage}: forward-valid prototype ID outside bank [0,{m}).")
    weights, acts = None, stage_out.get("visual_activations")
    if acts is not None:
        if acts.shape != logits.shape:
            raise RuntimeError(f"{stage}: visual_activations must match flat logits shape.")
        weights = acts.detach().float().reshape(b, t, h, w, m)[:, -1].cpu().numpy()
        if not np.isfinite(weights).all():
            raise ValueError(f"{stage}: non-finite activations.")
    return {"h": h, "w": w, "t": t, "m": m, "top": top.cpu().numpy(),
            "logits": vals.cpu().numpy(), "active": native_ids, "mask": native_mask, "weights": weights}


def last_gt_map(sal, sample, length):
    if sal is None:
        return None
    a = sal[sample].detach().float().cpu()
    if a.ndim == 4:
        if a.shape[1] == 1:
            a = a[:, 0]
        elif a.shape[0] == 1:
            a = a[0]
        else:
            raise RuntimeError(f"Unsupported GT layout {tuple(a.shape)}.")
    if a.ndim == 3:
        a = a[min(length - 1, a.shape[0] - 1)]
    if a.ndim != 2:
        raise RuntimeError(f"Expected 2D last GT map, got {tuple(a.shape)}.")
    return a


def last_rgb_frame(rgb):
    if rgb.ndim != 5:
        raise RuntimeError("RGB batch must have five dimensions.")
    if rgb.shape[2] == 3:
        a = rgb[:, -1]
    elif rgb.shape[-1] == 3:
        a = rgb[:, -1].permute(0, 3, 1, 2)
    else:
        raise RuntimeError(f"Expected BTCHW or BTHWC RGB, got {tuple(rgb.shape)}.")
    a = a.detach().float().cpu()
    if a.max() > 2:
        a = a / 255
    if not np.isfinite(a.numpy()).all() or a.min() < -0.01 or a.max() > 1.01:
        raise ValueError("Loader RGB must be in [0,1] or [0,255].")
    return a


def build_runtime(expected_checkpoint_sha256=None):
    import torch
    ev = load_evaluation()
    ev.train_cfg.set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        if GPU_ID >= torch.cuda.device_count() or GPU_ID < 0:
            raise ValueError(f"GPU_ID={GPU_ID} unavailable.")
        device = torch.device(f"cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")
    ckpt = str(CHECKPOINT_PATH or ev.CHECKPOINT_PATH)
    digest = sha256_file(ckpt)
    if expected_checkpoint_sha256 and digest != expected_checkpoint_sha256:
        raise ValueError("Checkpoint differs from the graph's original checkpoint; refusing to mix model features.")
    print(f"Building evaluation architecture on {device}; checkpoint={ckpt}", flush=True)
    model = ev.build_model(device, device)
    checkpoint = ev.load_checkpoint(model, ckpt)
    model.eval()
    enabled_fn, dtype_fn = getattr(ev.train_cfg, "_amp_enabled", None), getattr(ev.train_cfg, "_amp_dtype", None)
    amp = bool(USE_EVAL_AMP and callable(enabled_fn) and enabled_fn(device))
    dtype = dtype_fn(device) if amp and callable(dtype_fn) else torch.float16
    return ev, model, device, ckpt, checkpoint, digest, amp, dtype


def validate_full_batch(names, rgb, lengths, valid, window_len):
    if any(int(n) != window_len for n in lengths) or tuple(valid.shape) != (len(names), window_len) or not bool(valid.all()):
        raise RuntimeError("Unexpected short/padded window; refusing invalid target frames.")
    return last_rgb_frame(rgb)


def extract_graph():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    if WINDOW_STRIDE < 1 or BATCH_SIZE < 1 or TOP_K < 1 or not STAGES:
        raise ValueError("WINDOW_STRIDE, BATCH_SIZE, TOP_K and STAGES must be positive/nonempty.")
    if not math.isfinite(TOP_K_FREQUENCY_EXCLUDE) or not 0.0 < float(TOP_K_FREQUENCY_EXCLUDE) <= 1.0:
        raise ValueError("TOP_K_FREQUENCY_EXCLUDE must be in (0, 1].")
    outdir = Path(OUTPUT_DIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    dbpath = outdir / "graph.sqlite"
    if dbpath.exists():
        raise FileExistsError(f"{dbpath} exists. Use cache_similarity/view/serve, or a new OUTPUT_DIR.")
    ev, model, device, ckpt, checkpoint, digest, amp, dtype = build_runtime()
    for stage in STAGES:
        if stage not in model.backbone_stages:
            raise ValueError(f"{stage} not in model.backbone_stages={model.backbone_stages}.")
    capture = Stage4Capture(model) if "stage4" in STAGES else None
    roots = normalize_dataset_roots(DATASET_ROOTS, ev.VAL_DATASET_DIR)
    workers = int(ev.NUM_WORKERS if NUM_WORKERS is None else NUM_WORKERS)
    window_len = int(ev.WINDOW_LEN)
    manifest = {"created_at": now(), "status": "building", "checkpoint": str(Path(ckpt).resolve()),
                "checkpoint_sha256": digest, "checkpoint_epoch": checkpoint.get("epoch"),
                "evaluation_file": str(Path(EVALUATION_FILE).resolve()), "evaluation_sha256": sha256_file(EVALUATION_FILE),
                "dataset_roots": {str(k): str(Path(v).resolve()) for k, v in roots.items()},
                "stages": list(STAGES), "window_len": window_len, "stride": WINDOW_STRIDE,
                "batch_size": BATCH_SIZE, "top_k": TOP_K, "max_windows": MAX_WINDOWS,
                "topk_frequency_exclude": float(TOP_K_FREQUENCY_EXCLUDE),
                "device": str(device), "seed": SEED, "coverage": [], "amp_enabled": amp,
                "amp_dtype": str(dtype) if amp else None,
                "graph_representation": "exact indexed memberships; no NxN materialization",
                "edge_weight": "intersection count, stage-specific", "edge_color": SIMILARITY_FORMULA,
                "similarity_settings": similarity_settings(),
                "topk_semantics": (
                    f"raw highest {TOP_K} matching logits stored in the database; "
                    "serve/view/export drop prototypes with document frequency > "
                    f"TOP_K_FREQUENCY_EXCLUDE over distinct samples; candidates may be inactive"
                ),
                "active_semantics": "native active indices AND native validity mask",
                "pred_score_semantics": "mean raw model saliency_map in patch cell; midrank within frame",
                "cell_mapping": "uniform concept grid over model input; not a receptive-field boundary"}
    dump_json(outdir / "manifest.json", manifest)
    con = connect_db(dbpath)
    create_schema(con)
    (outdir / "frames").mkdir(exist_ok=True)
    node_id = frame_id = processed = 0
    active_counts, started = Counter(), time.monotonic()
    try:
        for label, root in roots.items():
            if MAX_WINDOWS is not None and processed >= MAX_WINDOWS:
                break
            dataset = ev.DatasetLoader(str(root), window_len=window_len, stride=WINDOW_STRIDE)
            windows = getattr(dataset, "windows", None)
            if windows is None:
                raise RuntimeError("DatasetLoader must expose .windows=[(video_name,start_index),...].")
            frame_names, selected_indices = {}, []
            for di, entry in enumerate(windows):
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise RuntimeError("Unsupported dataset.windows entry.")
                video, start = str(entry[0]), int(entry[1])
                if video not in frame_names:
                    frame_names[video] = frame_names_for_video(dataset, root, video)
                if start >= 0 and start + window_len <= len(frame_names[video]):
                    selected_indices.append(di)
            eligible = len(selected_indices)
            if MAX_WINDOWS is not None:
                selected_indices = selected_indices[:max(0, MAX_WINDOWS - processed)]
            manifest["coverage"].append({"dataset": label, "loader_windows": len(windows), "full_windows": eligible,
                                         "scheduled_windows": len(selected_indices), "excluded_partial_windows": len(windows) - eligible})
            kwargs = dict(num_workers=workers, collate_fn=ev.video_saliency_collate_fn, pin_memory=device.type == "cuda")
            if workers > 0:
                kwargs.update(persistent_workers=True, prefetch_factor=2)
            loader = DataLoader(Subset(dataset, selected_indices), batch_size=BATCH_SIZE, shuffle=False, **kwargs)
            offset = 0
            progress = tqdm(loader, total=math.ceil(len(selected_indices)/BATCH_SIZE), desc=f"{label} batches")
            for batch in progress:
                names, rgb, sal, lengths, valid = unpack_batch(batch)
                b = len(names)
                indices = selected_indices[offset:offset+b]
                offset += b
                last_rgb = validate_full_batch(names, rgb, lengths, valid, window_len)
                if capture:
                    capture.reset()
                with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=amp):
                    result = model(rgb.to(device, non_blocking=True), return_details=True,
                                   return_concept_losses=False, return_decoder_diagnostics=False)
                if not isinstance(result, dict) or "concept_out" not in result:
                    raise RuntimeError("Model must return details['concept_out'][stage].")
                pred = result["saliency_map"].detach().float().cpu()
                if pred.ndim != 4 or tuple(pred.shape[:2]) != (b, 1) or not torch.isfinite(pred).all():
                    raise RuntimeError(f"Expected finite saliency_map [B,1,H,W], got {tuple(pred.shape)}.")
                this_frames = []
                for bi, di in enumerate(indices):
                    video, start = str(windows[di][0]), int(windows[di][1])
                    if str(names[bi]) != video:
                        raise RuntimeError("Loader order/name mismatch.")
                    target, fnames = start + window_len - 1, frame_names[video]
                    input_h, input_w = map(int, last_rgb.shape[-2:])
                    thumb = None
                    if SAVE_FRAME_THUMBNAILS:
                        thumb = f"frames/frame_{frame_id:09d}.jpg"
                        image = Image.fromarray((last_rgb[bi].permute(1, 2, 0).numpy().clip(0,1)*255).astype(np.uint8))
                        image.thumbnail((FRAME_THUMB_WIDTH, FRAME_THUMB_WIDTH))
                        image.save(outdir / thumb, quality=83)
                    source = str((Path(root) / video / "images" / fnames[target]).resolve())
                    con.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                (frame_id, str(label), video, int(di), start, target, fnames[target], source,
                                 thumb, json.dumps(fnames[start:target+1]), input_h, input_w))
                    this_frames.append(frame_id)
                    frame_id += 1
                for stage in STAGES:
                    detail = result["concept_out"].get(stage)
                    if detail is None:
                        raise RuntimeError(f"No concept_out for {stage}.")
                    info = extract_stage(detail, b, stage)
                    h, w = info["h"], info["w"]
                    features = capture.last_frame(info, b) if stage == "stage4" else None
                    patch_pred = F.adaptive_avg_pool2d(pred, (h,w))[:,0].numpy()
                    records, memberships, descriptors, strengths = [], [], [], []
                    for bi in range(b):
                        pranks = midrank_percentiles(patch_pred[bi])
                        gt = last_gt_map(sal, bi, window_len)
                        gtgrid = F.adaptive_avg_pool2d(gt[None,None], (h,w))[0,0].numpy() if gt is not None else None
                        granks = midrank_percentiles(gtgrid) if gtgrid is not None else None
                        rgb_hwc = last_rgb[bi].permute(1,2,0).numpy()
                        for row in range(h):
                            for col in range(w):
                                flat = row*w+col
                                topids = [int(x) for x in info["top"][bi,row,col]]
                                topvals = [float(x) for x in info["logits"][bi,row,col]]
                                active = sorted(set(int(x) for x in info["active"][bi,row,col][info["mask"][bi,row,col]]))
                                aw = [float(info["weights"][bi,row,col,p]) for p in active] if info["weights"] is not None else [None]*len(active)
                                active_counts[(stage,len(active))] += 1
                                records.append((node_id,this_frames[bi],stage,row,col,h,w,float(patch_pred[bi,row,col]),
                                                float(pranks[flat]),float(gtgrid[row,col]) if gtgrid is not None else None,
                                                float(granks[flat]) if granks is not None else None,
                                                json.dumps(topids),json.dumps(topvals),json.dumps(active),json.dumps(aw)))
                                memberships.extend((node_id,stage,"topk",p) for p in topids)
                                memberships.extend((node_id,stage,"active",p) for p in active)
                                strengths.append((node_id, bi, row, col))
                                if features is not None:
                                    descriptors.append((node_id, features[bi,row,col], rgb_descriptor(rgb_hwc,row,col,h,w)))
                                node_id += 1
                    con.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)
                    con.executemany("INSERT INTO memberships VALUES (?,?,?,?)", memberships)
                    for nid, bi, row, col in strengths:
                        store_strengths(con, nid, info, bi, row, col)
                    for descriptor in descriptors:
                        store_descriptor(con, *descriptor)
                processed += b
                con.commit()
                progress.set_postfix(patches=f"{node_id:,}", refresh=False)
                del result, pred, batch, rgb, sal, last_rgb
            if MAX_WINDOWS is not None and processed >= MAX_WINDOWS:
                break
        if node_id == 0:
            raise RuntimeError("No full windows/patches were extracted.")
        print(f"Collected raw top-{TOP_K} memberships for {node_id:,} patches across "
              f"{processed:,} samples. Computing prototype document frequencies…", flush=True)
        frequency_stats = compute_topk_sample_frequencies(con)
        dump_json(outdir / "topk_frequency_stats.json", frequency_stats)
        excluded_by_stage, frequency_filter = excluded_prototypes_by_stage(
            frequency_stats, TOP_K_FREQUENCY_EXCLUDE)
        dump_json(outdir / "topk_frequency_filter.json", frequency_filter)
        con.execute("ANALYZE")
        con.commit()
        manifest.update(status="complete", completed_at=now(), num_windows=processed, num_nodes=node_id,
                        elapsed_seconds=round(time.monotonic()-started,2),
                        topk_frequency_stats=frequency_stats,
                        topk_frequency_filter=frequency_filter,
                        native_active_count_histogram={f"{s}:{count}": n for (s,count),n in sorted(active_counts.items())})
        dump_json(outdir / "manifest.json", manifest)
        write_viewer(outdir, con, excluded_by_stage=excluded_by_stage, frequency_filter=frequency_filter)
        excluded_total = sum(stage_info["num_excluded"] for stage_info in frequency_filter["stages"].values())
        print(f"Complete: {node_id:,} patch nodes with raw top-{TOP_K} stored. "
              f"Current TOP_K_FREQUENCY_EXCLUDE={TOP_K_FREQUENCY_EXCLUDE:g} would drop "
              f"{excluded_total:,} prototypes at serve/view time. Full graph: {dbpath}", flush=True)
    except BaseException as exc:
        con.rollback()
        manifest.update(status="failed", failed_at=now(), error=str(exc),
                        committed_node_count=int(con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]))
        dump_json(outdir / "manifest.json", manifest)
        raise
    finally:
        con.close()
        if capture:
            capture.close()

def require_complete(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Graph extraction is {manifest.get('status')!r}, not complete.")
    return manifest


def cache_similarity(directory):
    """Resume a descriptor/activation backfill without modifying original graph data."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    directory = Path(directory).resolve()
    manifest = require_complete(directory)
    if int(manifest.get("top_k", TOP_K)) != TOP_K or int(manifest.get("seed", SEED)) != SEED:
        raise ValueError("Use the original graph's TOP_K and SEED settings for cache backfill.")
    con = connect_db(directory / "graph.sqlite")
    capture = None
    try:
        ensure_similarity_schema(con)
        pending = con.execute("""SELECT DISTINCT f.* FROM frames f JOIN nodes n ON n.frame_id=f.id
            LEFT JOIN visual_descriptors v ON v.node_id=n.id
            LEFT JOIN node_strengths s ON s.node_id=n.id
            WHERE (n.stage='stage4' AND v.node_id IS NULL) OR s.node_id IS NULL
            ORDER BY f.dataset,f.dataset_index""").fetchall()
        if not pending:
            print("All stored nodes already have cached descriptor/activation records.", flush=True)
            write_viewer(directory, con)
            return
        ev, model, device, _, _, _, amp, dtype = build_runtime(manifest.get("checkpoint_sha256"))
        if "amp_enabled" in manifest and bool(manifest["amp_enabled"]) != amp:
            raise ValueError("Use the original graph's AMP setting/device for feature backfill.")
        if amp and manifest.get("amp_dtype") and str(dtype) != manifest["amp_dtype"]:
            raise ValueError("Use the original graph's AMP dtype for feature backfill.")
        need_stage4 = con.execute("SELECT 1 FROM nodes WHERE stage='stage4' LIMIT 1").fetchone() is not None
        capture = Stage4Capture(model) if need_stage4 else None
        roots = manifest["dataset_roots"]
        window_len, stride = int(manifest["window_len"]), int(manifest["stride"])
        workers = int(ev.NUM_WORKERS if NUM_WORKERS is None else NUM_WORKERS)
        manifest.update(similarity_cache_status="building", similarity_settings=similarity_settings())
        dump_json(directory / "manifest.json", manifest)
        for label in dict.fromkeys(row["dataset"] for row in pending):
            if label not in roots:
                raise ValueError(f"Missing original dataset root for {label}.")
            frames = [dict(r) for r in pending if r["dataset"] == label]
            root = roots[label]
            dataset = ev.DatasetLoader(str(root), window_len=window_len, stride=stride)
            for frame in frames:
                di = frame["dataset_index"]
                if di >= len(dataset.windows) or tuple(map(str,dataset.windows[di])) != (str(frame["video"]),str(frame["window_start"])):
                    raise RuntimeError("Dataset windows changed since graph extraction; refusing misaligned descriptors.")
                names = frame_names_for_video(dataset, root, frame["video"])
                if frame["target_index"] >= len(names) or names[frame["target_index"]] != frame["target_name"]:
                    raise RuntimeError("Target frame names changed since graph extraction.")
            loader = DataLoader(Subset(dataset,[f["dataset_index"] for f in frames]), batch_size=1, shuffle=False,
                                num_workers=workers, collate_fn=ev.video_saliency_collate_fn, pin_memory=device.type=="cuda")
            for frame, batch in tqdm(zip(frames,loader), total=len(frames), desc=f"Cache {label}"):
                names, rgb, _, lengths, valid = unpack_batch(batch)
                last_rgb = validate_full_batch(names,rgb,lengths,valid,window_len)
                if str(names[0]) != frame["video"] or tuple(last_rgb.shape[-2:]) != (frame["input_h"],frame["input_w"]):
                    raise RuntimeError("Original frame coordinates/loader identity changed.")
                if capture:
                    capture.reset()
                with torch.inference_mode(), torch.autocast(device.type,dtype=dtype,enabled=amp):
                    result = model(rgb.to(device,non_blocking=True),return_details=True,
                                   return_concept_losses=False,return_decoder_diagnostics=False)
                node_rows = con.execute("SELECT id FROM nodes WHERE frame_id=? ORDER BY id",(frame["id"],)).fetchall()
                stored = [get_node(con,r[0]) for r in node_rows]
                rgb_hwc = last_rgb[0].permute(1,2,0).numpy()
                stage_info = {}
                for node in stored:
                    stage = node["stage"]
                    if stage not in stage_info:
                        if stage not in result.get("concept_out",{}):
                            raise RuntimeError(f"Model no longer exposes {stage}.")
                        info = extract_stage(result["concept_out"][stage],1,stage)
                        features = capture.last_frame(info,1) if stage=="stage4" else None
                        stage_info[stage] = (info,features)
                    info,features = stage_info[stage]
                    row,col = node["row"],node["col"]
                    if (info["h"],info["w"]) != (node["grid_h"],node["grid_w"]):
                        raise RuntimeError("Native prototype grid changed since extraction.")
                    top = list(map(int,info["top"][0,row,col]))
                    active = sorted(set(map(int,info["active"][0,row,col][info["mask"][0,row,col]])))
                    if top != node["topk_ids"] or active != node["active_ids"]:
                        raise RuntimeError(f"Prototype assignments changed for node {node['id']}; refusing mixed descriptors.")
                    store_strengths(con,node["id"],info,0,row,col)
                    if features is not None:
                        store_descriptor(con,node["id"],features[0,row,col],rgb_descriptor(rgb_hwc,row,col,info["h"],info["w"]))
                con.commit()
                del result, batch, rgb, last_rgb
        manifest.update(similarity_cache_status="complete",similarity_cached_at=now(),
                        num_visual_descriptors=int(con.execute("SELECT COUNT(*) FROM visual_descriptors").fetchone()[0]))
        dump_json(directory / "manifest.json",manifest)
        write_viewer(directory,con)
    except BaseException as exc:
        con.rollback()
        manifest.update(similarity_cache_status="failed",similarity_cache_error=str(exc))
        dump_json(directory / "manifest.json",manifest)
        raise
    finally:
        if capture:
            capture.close()
        con.close()


def serve_graph(directory):
    directory = Path(directory).resolve()
    require_complete(directory)
    with closing(connect_db(directory / "graph.sqlite", readonly=True)) as con:
        excluded_by_stage, frequency_filter = resolve_topk_frequency_exclusion(directory, con=con)
    excluded_total = sum(stage["num_excluded"] for stage in frequency_filter["stages"].values())
    print(f"Serving with TOP_K_FREQUENCY_EXCLUDE={TOP_K_FREQUENCY_EXCLUDE:g} "
          f"({excluded_total:,} high-frequency top-K prototypes hidden). "
          "Change the setting and restart serve to retune without re-extracting.", flush=True)
    write_viewer(directory, excluded_by_stage=excluded_by_stage, frequency_filter=frequency_filter)
    html_path = directory / "graph_viewer.html"
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/graph_viewer.html"):
                    content = html_path.read_text(encoding="utf-8").replace("const online=false;", "const online=true;").encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                elif parsed.path in ("/api/graph", "/api/search", "/api/top-patches"):
                    q = parse_qs(parsed.query)
                    with closing(connect_db(directory / "graph.sqlite",readonly=True)) as con:
                        if parsed.path == "/api/graph":
                            payload = make_payload(con,directory,seed=int(q.get("seed",[choose_seed(con)])[0]),
                                                   kind=q.get("kind",[GRAPH_KIND])[0],limit=int(q.get("limit",[VIEW_MAX_NODES])[0]),
                                                   min_overlap=int(q.get("min_overlap",[1])[0]),cross_video=q.get("cross",["1"])[0]=="1",
                                                   excluded_by_stage=excluded_by_stage, frequency_filter=frequency_filter)
                        elif parsed.path == "/api/top-patches":
                            payload = make_patch_gallery(con,directory,node_id=int(q.get("node",[choose_seed(con)])[0]),
                                                         kind=q.get("kind",[GRAPH_KIND])[0],min_overlap=int(q.get("min_overlap",[1])[0]),
                                                         cross_video=q.get("cross",["1"])[0]=="1",
                                                         excluded_by_stage=excluded_by_stage, frequency_filter=frequency_filter)
                        else:
                            term = q.get("q",[""])[0][:200]
                            ids = con.execute("""SELECT n.id FROM nodes n JOIN frames f ON f.id=n.frame_id
                                WHERE f.video LIKE ? OR f.target_name LIKE ? OR n.stage LIKE ?
                                OR CAST(n.id AS TEXT)=? ORDER BY n.id LIMIT 100""",
                                              (f"%{term}%",f"%{term}%",f"%{term}%",term)).fetchall()
                            payload = {"nodes":[filter_node_for_topk_frequency(get_node(con,r[0]), excluded_by_stage) for r in ids],
                                       "topk_frequency_filter": frequency_filter}
                    content,content_type = json.dumps(payload,allow_nan=False).encode(),"application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type",content_type)
                self.send_header("Content-Length",str(len(content)))
                self.send_header("Cache-Control","no-store")
                self.end_headers()
                self.wfile.write(content)
            except (ValueError,KeyError,sqlite3.Error,RuntimeError) as exc:
                content = json.dumps({"error":str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length",str(len(content)))
                self.end_headers()
                self.wfile.write(content)
    server = ThreadingHTTPServer(("127.0.0.1",SERVER_PORT),Handler)
    print(f"Open http://127.0.0.1:{SERVER_PORT} — stop with Ctrl+C.",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def export_edges(directory):
    directory = Path(directory)
    require_complete(directory)
    if GRAPH_KIND not in ("topk","active") or EXPORT_MIN_OVERLAP<1:
        raise ValueError("Invalid export kind/minimum overlap.")
    path = directory / f"edges_{GRAPH_KIND}.csv"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    count,truncated = 0,False
    columns = ["source","target","overlap_count","shared_prototype_ids","stage4_cosine","rgb_mae",
               "rgb_structure_cosine","visual_score","activation_difference","similarity_missing_reason","activation_missing_reason"]
    with closing(connect_db(directory / "graph.sqlite",readonly=True)) as con,path.open("x",newline="") as stream:
        excluded_by_stage, frequency_filter = resolve_topk_frequency_exclusion(directory, con=con)
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in con.execute("SELECT id FROM nodes ORDER BY id"):
            source = int(row[0])
            left = filter_node_for_topk_frequency(get_node(con,source), excluded_by_stage)
            key = "topk_ids" if GRAPH_KIND=="topk" else "active_ids"
            for target,weight in neighbors(con,source,GRAPH_KIND,EXPORT_MIN_OVERLAP,
                                           excluded_by_stage=excluded_by_stage):
                if target<=source:
                    continue
                if EXPORT_MAX_EDGES is not None and count>=EXPORT_MAX_EDGES:
                    truncated = True
                    break
                right = filter_node_for_topk_frequency(get_node(con,target), excluded_by_stage)
                edge = {"source":source,"target":target,"weight":weight}
                annotate_edges(con,[left,right],[edge],kind=GRAPH_KIND)
                writer.writerow([source,target,weight,json.dumps(sorted(set(left[key])&set(right[key]))),
                                 *[edge.get(k) for k in columns[4:]]])
                count += 1
            if truncated:
                break
    dump_json(path.with_suffix(".json"),{"kind":GRAPH_KIND,"min_overlap":EXPORT_MIN_OVERLAP,"edges":count,
                                        "truncated":truncated,"complete":not truncated,"created_at":now(),
                                        "visual_similarity_formula":SIMILARITY_FORMULA,
                                        "topk_frequency_filter": frequency_filter})
    print(f"Exported {count:,} edges to {path}; truncated={truncated}.")


def selftest():
    """Graph oracle, visual-metric correctness and overlap-count-independent distances."""
    assert np.allclose(midrank_percentiles([0,0,1,2]),[1/6,1/6,2/3,1])
    assert np.allclose(midrank_percentiles([5,5,5]),[.5,.5,.5])
    feature = np.array([1.,2.,3.])
    rgb = np.random.default_rng(42).random((RGB_DESCRIPTOR_SIZE**2*3,))
    identical = visual_metrics((feature,rgb),(feature,rgb))
    assert np.isclose(identical["visual_score"],3.)
    assert np.isclose(identical["rgb_mae"],0.)
    opposite = visual_metrics((feature,rgb),(-feature,1-rgb))
    assert np.isclose(opposite["stage4_cosine"],-1.)
    assert np.isclose(opposite["rgb_structure_cosine"],-1.)
    assert opposite["visual_score"] < identical["visual_score"]
    assert visual_metrics(None,(feature,rgb))["visual_score"] is None
    assert visual_metrics((np.zeros(3),rgb),(feature,rgb))["visual_score"] is None
    flat = visual_metrics((feature,np.zeros_like(rgb)),(feature,np.ones_like(rgb)))
    assert np.isclose(flat["rgb_mae"],1.) and np.isclose(flat["rgb_structure_cosine"],1.)
    assert np.array_equal(unpack_vector(pack_vector(feature)),feature)
    def strengths(ids,values):
        return {"topk_ids":ids,"topk_weights":values,"active_ids":ids,"active_weights":values}
    for kind in ("topk","active"):
        one = activation_difference(strengths([1],[1.]),strengths([1],[3.]),kind)[0]
        many = activation_difference(strengths([1,2,3],[1.,1.,1.]),strengths([1,2,3],[3.,3.,3.]),kind)[0]
        assert np.isclose(one,.5) and np.isclose(many,one)
        assert activation_difference(strengths([1],[0.]),strengths([1],[0.]),kind)[0] == 0.
        assert activation_difference(strengths([1],[-1.]),strengths([1],[1.]),kind)[0] == 1.
        assert np.isclose(activation_difference(strengths([1],[1e-30]),strengths([1],[3e-30]),kind)[0],one)
        assert activation_difference(strengths([1],[None]),strengths([1],[1.]),kind)[0] is None
    color_frame = np.zeros((20, 20, 3),dtype=np.float32)
    color_frame[:,10:,0] = 1.
    assert np.allclose(rgb_descriptor(color_frame,0,0,1,2),0.)
    assert np.allclose(rgb_descriptor(color_frame,0,1,1,2).reshape(-1,3)[:,0],1.)
    class ArrayTensor:
        def __init__(self,a): self.a = a
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return self.a
    native = np.arange(2*7*3*4*5,dtype=np.float32).reshape(2,7,3,4,5)
    capture = object.__new__(Stage4Capture)
    capture.value = ArrayTensor(native)
    capture.source = None
    capture._source_reported = False
    actual = capture.last_frame({"t":3,"h":4,"w":5},2,layout="BCTHW")
    assert np.array_equal(actual,native[:,:,-1].transpose(0,2,3,1))
    try:
        capture.last_frame({"t":9,"h":4,"w":5},2,layout="BCTHW")
        raise AssertionError("Misaligned stage4 features were accepted")
    except RuntimeError:
        pass
    # Concept metadata may report T=1 while backbone features keep the full clip.
    full_clip = np.arange(1*8*16*7*12,dtype=np.float32).reshape(1,8,16,7,12)
    capture.value = ArrayTensor(full_clip)
    collapsed = capture.last_frame({"t":1,"h":7,"w":12},1)  # auto -> BCTHW last frame
    assert collapsed.shape == (1,7,12,8)
    assert np.array_equal(collapsed, full_clip[:,:,-1].transpose(0,2,3,1))
    assert filter_parallel_lists([1,2,3],[.1,.2,.3],{2}) == ([1,3],[.1,.3])
    with tempfile.TemporaryDirectory() as td_freq:
        d_freq = Path(td_freq)
        con_freq = connect_db(d_freq / "graph.sqlite")
        create_schema(con_freq)
        # 4 samples; prototype 0 appears in every sample (>75%); others are rare.
        for frame_id in range(4):
            con_freq.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (frame_id,"test",f"video{frame_id}",frame_id,0,15,f"{frame_id}.jpg",
                              "missing.jpg",None,"[]",224,384))
            top = [0, 1] if frame_id == 0 else [0, 2 + frame_id]
            logits = [1.0, 0.5]
            con_freq.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (frame_id,frame_id,"stage4",0,0,1,1,0.5,0.5,None,None,
                              json.dumps(top),json.dumps(logits),json.dumps([0]),json.dumps([1.0])))
            con_freq.executemany("INSERT INTO memberships VALUES (?,?,?,?)",
                                 [(frame_id,"stage4","topk",p) for p in top] +
                                 [(frame_id,"stage4","active",0)])
            con_freq.execute("INSERT INTO node_strengths VALUES (?,?,?)",
                             (frame_id, json.dumps([0.9, 0.4]), json.dumps([1.0])))
        con_freq.commit()
        stats = compute_topk_sample_frequencies(con_freq)
        dump_json(d_freq / "topk_frequency_stats.json", stats)
        dump_json(d_freq / "manifest.json", {"status": "complete", "topk_frequency_stats": stats})
        excluded, summary = excluded_prototypes_by_stage(stats, 0.75)
        assert summary["stages"]["stage4"]["excluded_prototype_ids"] == [0]
        # Raw DB keeps all prototypes; serve-time filtering drops frequency > 0.75.
        raw0 = get_node(con_freq, 0)
        assert raw0["topk_ids"] == [0, 1]
        filtered0 = filter_node_for_topk_frequency(raw0, excluded)
        assert filtered0["topk_ids"] == [1]
        assert filtered0["active_ids"] == [0]
        assert [r[0] for r in con_freq.execute(
            "SELECT prototype_id FROM memberships WHERE kind='topk' ORDER BY node_id, prototype_id")] == [0, 1, 0, 3, 0, 4, 0, 5]
        assert neighbors(con_freq, 0, "topk", 1, excluded_by_stage=excluded) == []
        assert neighbors(con_freq, 0, "topk", 1, excluded_by_stage=None)
        resolved_excluded, resolved_summary = resolve_topk_frequency_exclusion(d_freq, con=con_freq, max_frequency=0.75)
        assert resolved_excluded["stage4"] == {0}
        assert resolved_summary["max_frequency"] == 0.75
        con_freq.close()
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        con = connect_db(d / "graph.sqlite")
        create_schema(con)
        rng = np.random.default_rng(4)
        top_size = min(TOP_K, 8)
        for i in range(6):
            con.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (i,"test",f"video{i%3}",i,0,15,f"{i}.jpg","missing.jpg",None,"[]",224,384))
        for i in range(60):
            stage = "stage3" if i%2 else "stage4"
            top = sorted(map(int,rng.choice(20,size=top_size,replace=False)))
            active = top[:i%4]
            aw = [float(x) for x in rng.random(len(active))]
            con.execute("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (i,i%6,stage,i//6,0,20,1,i/60,i/60,None,None,json.dumps(top),"[]",json.dumps(active),json.dumps(aw)))
            con.executemany("INSERT INTO memberships VALUES (?,?,?,?)",
                            [(i,stage,k,p) for k,ps in (("topk",top),("active",active)) for p in ps])
            con.execute("INSERT INTO node_strengths VALUES (?,?,?)",
                        (i,json.dumps(rng.random(len(top)).tolist()),json.dumps(aw)))
            if stage=="stage4":
                store_descriptor(con,i,rng.random(12),rng.random(RGB_DESCRIPTOR_SIZE**2*3))
        con.commit()
        all_nodes = [get_node(con,i) for i in range(60)]
        for kind in ("topk","active"):
            key = "topk_ids" if kind=="topk" else "active_ids"
            for seed in (0,1,7,15):
                for cross in (False,True):
                    for threshold in (1,3,min(8, top_size)):
                        a,brute = all_nodes[seed],[]
                        for b in all_nodes:
                            if a["id"]==b["id"] or a["stage"]!=b["stage"]:
                                continue
                            if cross and (a["dataset"],a["video"])==(b["dataset"],b["video"]):
                                continue
                            weight = len(set(a[key])&set(b[key]))
                            if weight>=threshold:
                                brute.append((b["id"],weight))
                        brute.sort(key=lambda x:(-x[1],x[0]))
                        assert neighbors(con,seed,kind,threshold,None,cross)==brute
                        assert neighbors(con,seed,kind,threshold,4,cross)==brute[:4]
            actual = {(e["source"],e["target"]):e["weight"] for e in induced_edges(all_nodes,kind)}
            expected = {(a["id"],b["id"]):len(set(a[key])&set(b[key])) for j,a in enumerate(all_nodes)
                        for b in all_nodes[j+1:] if a["stage"]==b["stage"] and set(a[key])&set(b[key])}
            assert actual==expected
        no_exclude, keep_all = {}, {"max_frequency": 1.0, "top_k": TOP_K, "stages": {}}
        payload = make_payload(con,d,seed=0,cross_video=False,excluded_by_stage=no_exclude,frequency_filter=keep_all)
        assert payload["edges"] and all(e["visual_score"] is not None for e in payload["edges"])
        assert all(e["activation_difference"] is not None for e in payload["edges"])
        assert len(payload["visual_similarity_pairs"]) >= len(payload["edges"])
        json.dumps(payload,allow_nan=False)
        gallery = make_patch_gallery(con,d,0,cross_video=False,excluded_by_stage=no_exclude,frequency_filter=keep_all)
        assert gallery["patches"][0]["gallery_selected"] and gallery["total_slots"]==50
        write_viewer(d,con,excluded_by_stage=no_exclude,frequency_filter=keep_all)
        html = (d/"graph_viewer.html").read_text()
        assert "__PAYLOAD__" not in html and 'http://cdn' not in html and '<canvas' in html
        assert 'id="distanceMode"' in html and 'id="edgeLegend"' in html
        # Legacy graph remains viewable; no fake feature cosine or logit-as-activation fallback.
        con.execute("DROP TABLE visual_descriptors")
        con.execute("DROP TABLE node_strengths")
        legacy = make_payload(con,d,seed=0,cross_video=False)
        assert all(e["visual_score"] is None and e["activation_difference"] is None for e in legacy["edges"])
        try:
            neighbors(con,0,"invalid")
            raise AssertionError("Invalid graph kind accepted")
        except ValueError:
            pass
        con.close()
    print("PASS: exact graph oracle; stage/video filters; galleries; visual formula and degenerate inputs; "
          "count-independent normalized activation differences; legacy databases; offline HTML.")


def main():
    actions = {"extract":extract_graph,"cache_similarity":lambda:cache_similarity(OUTPUT_DIR),
               "view":lambda:(require_complete(OUTPUT_DIR),write_viewer(OUTPUT_DIR)),
               "serve":lambda:serve_graph(OUTPUT_DIR),"export_edges":lambda:export_edges(OUTPUT_DIR),"selftest":selftest}
    if ACTION not in actions:
        raise ValueError(f"Unknown ACTION={ACTION!r}; choose from {list(actions)}")
    actions[ACTION]()


if __name__ == "__main__":
    main()
