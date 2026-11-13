#!/usr/bin/env python3
"""Prototype/patch visual-similarity explorer (offline HTML or local server).

Edit SETTINGS and run this script in src/interpretability_exp/exp4_expl_report/.
No command-line arguments. Actions:
  extract: checkpoint inference; save patches, descriptors and ALL stage4
           prototype activations; build ranked index and offline viewer.
  cache_similarity: upgrade an existing complete patch database in place.
           Replays only missing frames, verifies checkpoint/window/assignments,
           preserves original nodes, memberships and IDs. Run once for old DBs.
  view / serve: no inference; show ALL stage4 prototypes by default, each linked
           to its N strongest patches across all cached target-frame patches.
  export_edges: export the new graph's two edge types to CSV.
  selftest: test exact rankings, ties, similarity, filtering, migration and HTML.

PATCHES_PER_PROTOTYPE is N. Ranking uses native visual_activations across the
entire prototype bank, never just a patch's top-K list and never logits as a
fallback. These links describe strongest matches; forward validity is separately
reported. Every stage4 prototype appears, including banks with no valid matches.
No frequency or cross-video filter is applied to this graph.

A prototype's visual signature is the arithmetic mean of the cached stage4
backbone and RGB descriptors of its N strongest patches. Edge visual similarity
uses stage4 cosine + (1 - RGB MAE) + centered RGB structure cosine. Both edge
types use that same formula: signature vs patch, or signature vs signature.
This compares retrieved visual patterns, NOT learned prototype parameter vectors.
Missing/zero descriptors produce a grey edge with a reason; no scores invented.
The right panel's coherence is the mean of the SAME formula over all unordered
pairs among the prototype's N linked patches, with valid/total pair counts.
It describes the retrieved exemplars, not all patches that ever activate it.

Efficiency: dense activations are stored as compressed per-frame float32 blocks.
A streaming bounded top-N merge uses O(M*(N+patches/frame)) ranking memory;
SQLite indexes persist (prototype,rank)->patch matches for subsequent requests.
No patch-by-patch all-pairs graph or global patch similarity matrix is built.
Only selected patch descriptors are loaded; prototype signatures/coherence are
computed once on startup and reused for filtering. All prototype-pair edges are
precomputed (128 prototypes -> 8,128 edges). Frame images are deduplicated in
HTML, and layout/rendering has no quadratic patch force simulation.

N can change without inference. Increasing N beyond the indexed capacity rebuilds
only rankings from cached activations. Changing the N changes signatures and
coherence, so prototype filters always reuse signatures computed using the same N.
Existing descriptor caches retain the original source/layout/RGB semantics.
Backfill uses original dataset/window/AMP/checkpoint provenance. Uniform grid
crops are not exact receptive-field boundaries. Saliency remains a within-frame
rank, not a calibrated fixation probability. Model weights/forward outputs are
never changed.
"""
from __future__ import annotations
import base64
import csv
import hashlib
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
import zlib
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
from PIL import Image

# SETTINGS: edit here, then run the script.
ACTION = "serve"  # extract | cache_similarity | view | serve | export_edges | selftest
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[1]
EVALUATION_FILE = SRC_DIR / "evaluation.py"
CHECKPOINT_PATH = "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20261112_191642/epoch_189.pth"
DATASET_ROOTS = "/data/quantization/zaima/videosal_datasets/dhf1k/train"
STAGES = ("stage4",)
WINDOW_STRIDE = 128
BATCH_SIZE = 1
NUM_WORKERS = None
GPU_ID = 0
USE_EVAL_AMP = True
TOP_K = 3  # original candidate memberships, retained for provenance only
MAX_WINDOWS = None
OUTPUT_DIR = Path("/data/quantization/zaima/videosal_datasets/dhf1k/explanation_graphs/prototype_graph_outputs")
SAVE_FRAME_THUMBNAILS = True
FRAME_THUMB_WIDTH = 384
SEED = 42
SERVER_PORT = 8765
PATCHES_PER_PROTOTYPE = 10  # N: exact strongest matches per stage4 prototype
RANK_CACHE_CAPACITY = 50  # speed future changes of N; not a gallery limit
GALLERY_COLUMNS = 8
# None = displayed-edge min/max; each tooltip also shows the raw score.
EDGE_COLOR_RANGE = None  # or an increasing finite pair, e.g. (-2.0, 3.0)
RGB_DESCRIPTOR_SIZE = 16
STAGE4_FEATURE_MODULE = "backbone"
STAGE4_FEATURE_OUTPUT_PATH = ("stage4",)
STAGE4_FEATURE_LAYOUT = "auto"  # auto | BCTHW | BTCHW | BTHWC | BCHW | BHWC
SIMILARITY_VERSION = "stage4_rgb_structure_v1"
SIMILARITY_FORMULA = "stage4_cosine + (1 - rgb_mae) + rgb_structure_cosine"
GRAPH_VERSION = "prototype_patch_v2"

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

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
    """Ascending midrank in [0,1], ties preserved; flat maps -> 0.5."""
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
        topk_logits TEXT NOT NULL, active_ids TEXT NOT NULL,
        active_weights TEXT NOT NULL,
        UNIQUE(frame_id, stage, row, col),
        FOREIGN KEY(frame_id) REFERENCES frames(id)
    );
    CREATE TABLE memberships (
        node_id INTEGER NOT NULL, stage TEXT NOT NULL, kind TEXT NOT NULL,
        prototype_id INTEGER NOT NULL,
        PRIMARY KEY(node_id, kind, prototype_id),
        FOREIGN KEY(node_id) REFERENCES nodes(id)
    );
    CREATE INDEX postings ON memberships(stage, kind, prototype_id, node_id);
    CREATE INDEX nodes_by_frame ON nodes(frame_id,stage);
    CREATE INDEX frames_by_video ON frames(dataset,video);
    """)
    ensure_graph_schema(con)

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

def add_preview_images(nodes, directory):
    cache = {}
    for node in nodes:
        rel = node.get("thumb_path")
        if not rel:
            continue
        path = Path(directory) / rel
        if not path.is_file():
            continue
        if rel not in cache:
            data = path.read_bytes()
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

def load_evaluation():
    if not Path(EVALUATION_FILE).is_file():
        raise FileNotFoundError(f"Place this script beside evaluation.py, or edit SRC_DIR. "
                                f"Not found: {EVALUATION_FILE}")
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
    images = Path(root) / video / "images"
    maps = Path(root) / video / "maps"
    # The older supplied loader uses this exact two-directory method.
    listing = getattr(dataset, "_list_frame_names", None)
    if callable(listing):
        import inspect
        if len(inspect.signature(listing).parameters) == 2:
            return [Path(str(n)).name for n in listing(str(images), str(maps))]
    raise RuntimeError("DatasetLoader must expose video_frame_names[video] or "
                       "_list_frame_names(images_dir, maps_dir). Cannot safely infer frame IDs.")

def unpack_batch(batch):
    # Current train.py has six entries; the earlier loader has five.
    if not isinstance(batch, (list, tuple)) or len(batch) not in (5, 6):
        raise RuntimeError("Expected collate output (video_names,rgb,sal,[fix,]n_frames,valid_mask). "
                           f"Received {type(batch).__name__} with length "
                           f"{len(batch) if hasattr(batch, '__len__') else '?'}.")
    names, rgb, sal = batch[:3]
    n_frames, valid = batch[-2:]
    return list(names), rgb, sal, n_frames, valid

def extract_stage(stage_out, batch_size, stage):
    import torch
    required = ("active_visual_prototype_indices", "visual_validity_mask",
                "visual_concept_logits", "visual_metadata")
    missing = [key for key in required if key not in stage_out]
    if missing:
        raise RuntimeError(f"{stage}: missing {missing}; available={list(stage_out)}. "
                           "Expose the native model outputs; do not infer validity from logits.")
    ids, mask = stage_out[required[0]], stage_out[required[1]]
    if ids.ndim != 5 or mask.shape != ids.shape or ids.shape[0] != batch_size:
        raise RuntimeError(f"{stage}: expected indices/mask [B,T,H,W,K], got "
                           f"{tuple(ids.shape)} / {tuple(mask.shape)}.")
    b, t, h, w, k = ids.shape
    meta = stage_out["visual_metadata"]
    fs = meta.get("feature_shape", {})
    if any(int(fs.get(key, -1)) != value for key, value in
           (("B", b), ("T", t), ("H", h), ("W", w))):
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
    weights = None
    acts = stage_out.get("visual_activations")
    if acts is not None:
        if acts.shape != logits.shape:
            raise RuntimeError(f"{stage}: visual_activations must match flat logits shape.")
        weights = acts.detach().float().reshape(b, t, h, w, m)[:, -1].cpu().numpy()
        if not np.isfinite(weights).all():
            raise ValueError(f"{stage}: non-finite activations.")
    return {"h": h, "w": w, "t": t, "m": m, "top": top.cpu().numpy(),
            "logits": vals.cpu().numpy(), "active": native_ids,
            "mask": native_mask, "weights": weights}

def last_gt_map(sal, sample, length):
    if sal is None:
        return None
    a = sal[sample].detach().float().cpu()
    # [T,H,W], [T,1,H,W], [1,T,H,W] or last-frame [H,W].
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

def extract_graph():
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    if "stage4" not in STAGES:
        raise ValueError("The prototype graph requires stage4 in STAGES.")
    if WINDOW_STRIDE < 1 or BATCH_SIZE < 1 or TOP_K < 1 or not STAGES:
        raise ValueError("WINDOW_STRIDE, BATCH_SIZE, TOP_K and STAGES must be positive/nonempty.")
    outdir = Path(OUTPUT_DIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    dbpath = outdir / "graph.sqlite"
    if dbpath.exists():
        raise FileExistsError(f"{dbpath} already exists. Use ACTION='view'/'serve', "
                              "or choose a new OUTPUT_DIR for a fresh extraction.")
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
    print(f"Building evaluation architecture on {device}; checkpoint={ckpt}", flush=True)
    model = ev.build_model(device, device)
    checkpoint = ev.load_checkpoint(model, ckpt)
    model.eval()
    amp_enabled_fn = getattr(ev.train_cfg, "_amp_enabled", None)
    amp_dtype_fn = getattr(ev.train_cfg, "_amp_dtype", None)
    amp_enabled = bool(USE_EVAL_AMP and callable(amp_enabled_fn) and amp_enabled_fn(device))
    amp_dtype = amp_dtype_fn(device) if amp_enabled and callable(amp_dtype_fn) else torch.float16
    for stage in STAGES:
        if stage not in model.backbone_stages:
            raise ValueError(f"{stage} not in model.backbone_stages={model.backbone_stages}.")
    capture = Stage4Capture(model)
    roots = normalize_dataset_roots(DATASET_ROOTS, ev.VAL_DATASET_DIR)
    workers = int(ev.NUM_WORKERS if NUM_WORKERS is None else NUM_WORKERS)
    window_len = int(ev.WINDOW_LEN)
    manifest = {"created_at": now(), "status": "building", "checkpoint": str(Path(ckpt).resolve()),
                "checkpoint_sha256": sha256_file(ckpt), "checkpoint_epoch": checkpoint.get("epoch"),
                "evaluation_file": str(Path(EVALUATION_FILE).resolve()),
                "evaluation_sha256": sha256_file(EVALUATION_FILE),
                "dataset_roots": {str(k): str(Path(v).resolve()) for k, v in roots.items()},
                "stages": list(STAGES), "window_len": window_len, "stride": WINDOW_STRIDE,
                "batch_size": BATCH_SIZE, "top_k": TOP_K, "max_windows": MAX_WINDOWS,
                "device": str(device), "seed": SEED, "coverage": [],
                "amp_enabled": amp_enabled, "amp_dtype": str(amp_dtype) if amp_enabled else None,
                "graph_representation": "prototype-patch top-N incidence plus complete stage4 prototype graph",
                "graph_version": GRAPH_VERSION, "similarity_settings": similarity_settings(),
                "edge_weight": "native prototype activation",
                "edge_color": SIMILARITY_FORMULA,
                "prototype_signature": "mean descriptors of top-N strongest matches",
                "patches_per_prototype": PATCHES_PER_PROTOTYPE,
                "saliency_std": "population SD over connected patch within-frame percentiles",
                "topk_semantics": "highest matching logits; candidates may be inactive",
                "active_semantics": "native active indices AND native validity mask",
                "pred_score_semantics": "mean raw model saliency_map in patch cell; midrank within frame",
                "cell_mapping": "uniform concept grid over model input; not a receptive-field boundary"}
    dump_json(outdir / "manifest.json", manifest)
    con = connect_db(dbpath)
    create_schema(con)
    (outdir / "frames").mkdir(exist_ok=True)
    node_id = frame_id = processed = 0
    active_counts = Counter()
    started = time.monotonic()
    try:
        for label, root in roots.items():
            if MAX_WINDOWS is not None and processed >= MAX_WINDOWS:
                break
            dataset = ev.DatasetLoader(str(root), window_len=window_len, stride=WINDOW_STRIDE)
            windows = getattr(dataset, "windows", None)
            if windows is None:
                raise RuntimeError("DatasetLoader must expose .windows=[(video_name,start_index),...].")
            frame_names = {}
            selected_indices = []
            for di, entry in enumerate(windows):
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise RuntimeError("Unsupported dataset.windows entry; expected (video_name,start_index).")
                video, start = str(entry[0]), int(entry[1])
                if video not in frame_names:
                    frame_names[video] = frame_names_for_video(dataset, root, video)
                if start >= 0 and start + window_len <= len(frame_names[video]):
                    selected_indices.append(di)
            eligible = len(selected_indices)
            if MAX_WINDOWS is not None:
                selected_indices = selected_indices[:max(0, MAX_WINDOWS - processed)]
            manifest["coverage"].append({"dataset": label, "loader_windows": len(windows),
                "full_windows": eligible, "scheduled_windows": len(selected_indices),
                "excluded_partial_windows": len(windows) - eligible})
            print(f"{label}: {len(selected_indices)} full windows, stride={WINDOW_STRIDE}; "
                  f"{len(windows)-eligible} partial windows excluded.", flush=True)
            kwargs = dict(num_workers=workers, collate_fn=ev.video_saliency_collate_fn,
                          pin_memory=device.type == "cuda")
            if workers > 0:
                kwargs.update(persistent_workers=True, prefetch_factor=2)
            loader = DataLoader(Subset(dataset, selected_indices), batch_size=BATCH_SIZE,
                                shuffle=False, **kwargs)
            offset = 0
            progress = tqdm(loader, total=math.ceil(len(selected_indices)/BATCH_SIZE), desc=f"{label} batches",
                            unit="window", leave=True)
            for batch in progress:
                names, rgb, sal, n_frames, valid = unpack_batch(batch)
                b = len(names)
                indices = selected_indices[offset:offset+b]
                offset += b
                lengths = [int(x) for x in n_frames]
                if any(n != window_len for n in lengths):
                    raise RuntimeError("Loader returned a short/padded sample despite full-window filtering.")
                if tuple(valid.shape) != (b, window_len) or not bool(valid.all()):
                    raise RuntimeError("Unexpected temporal validity mask; refusing padded target frames.")
                capture.reset()
                with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype,
                                                           enabled=amp_enabled):
                    result = model(rgb.to(device, non_blocking=True), return_details=True,
                                   return_concept_losses=False, return_decoder_diagnostics=False)
                if not isinstance(result, dict) or "concept_out" not in result:
                    raise RuntimeError("Model must return details['concept_out'][stage].")
                pred = result["saliency_map"].detach().float().cpu()
                if pred.ndim != 4 or pred.shape[:2] != (b, 1):
                    raise RuntimeError(f"Expected saliency_map [B,1,H,W], got {tuple(pred.shape)}.")
                if not torch.isfinite(pred).all():
                    raise ValueError("Non-finite prediction.")
                # This is the loader RGB coordinate system, before backbone resize.
                if rgb.ndim != 5:
                    raise RuntimeError("RGB batch must have five dimensions.")
                if rgb.shape[2] == 3:
                    last_rgb = rgb[:, -1]
                elif rgb.shape[-1] == 3:
                    last_rgb = rgb[:, -1].permute(0, 3, 1, 2)
                else:
                    raise RuntimeError(f"Expected BTCHW or BTHWC RGB, got {tuple(rgb.shape)}.")
                last_rgb = last_rgb.detach().float().cpu()
                if last_rgb.max() > 2:
                    last_rgb /= 255
                if not np.isfinite(last_rgb.numpy()).all() or last_rgb.min() < -0.01 or last_rgb.max() > 1.01:
                    raise ValueError("RGB thumbnails require loader RGB in [0,1] or [0,255].")
                this_frames = []
                for bi, di in enumerate(indices):
                    video, start = str(windows[di][0]), int(windows[di][1])
                    if str(names[bi]) != video:
                        raise RuntimeError(f"Loader order/name mismatch: {names[bi]} vs {video}.")
                    target = start + window_len - 1
                    fnames = frame_names[video]
                    input_h, input_w = map(int, last_rgb.shape[-2:])
                    thumb = None
                    if SAVE_FRAME_THUMBNAILS:
                        thumb = f"frames/frame_{frame_id:09d}.jpg"
                        image = Image.fromarray((last_rgb[bi].permute(1, 2, 0).numpy().clip(0,1)*255).astype(np.uint8))
                        image.thumbnail((FRAME_THUMB_WIDTH, FRAME_THUMB_WIDTH))
                        image.save(outdir / thumb, quality=83)
                    source = str((Path(root) / video / "images" / fnames[target]).resolve())
                    con.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (frame_id, str(label), video, int(di), start, target, fnames[target],
                         source, thumb, json.dumps(fnames[start:target+1]), input_h, input_w))
                    this_frames.append(frame_id)
                    frame_id += 1
                for stage in STAGES:
                    detail = result["concept_out"].get(stage)
                    if detail is None:
                        raise RuntimeError(f"No concept_out for {stage}.")
                    info = extract_stage(detail, b, stage)
                    h, w = info["h"], info["w"]
                    patch_pred = F.adaptive_avg_pool2d(pred, (h,w))[:,0].numpy()
                    features = capture.last_frame(info, b) if stage == "stage4" else None
                    records, memberships = [], []
                    stage_first_node = node_id
                    for bi in range(b):
                        pranks = midrank_percentiles(patch_pred[bi])
                        gt = last_gt_map(sal, bi, window_len)
                        gtgrid = (F.adaptive_avg_pool2d(gt[None,None], (h,w))[0,0].numpy()
                                  if gt is not None else None)
                        granks = midrank_percentiles(gtgrid) if gtgrid is not None else None
                        for row in range(h):
                            for col in range(w):
                                flat = row*w+col
                                topids = [int(x) for x in info["top"][bi,row,col]]
                                topvals = [float(x) for x in info["logits"][bi,row,col]]
                                active = sorted(set(int(x) for x in info["active"][bi,row,col][info["mask"][bi,row,col]]))
                                aw = ([float(info["weights"][bi,row,col,p]) for p in active]
                                      if info["weights"] is not None else [None]*len(active))
                                active_counts[(stage,len(active))] += 1
                                records.append((node_id,this_frames[bi],stage,row,col,h,w,
                                    float(patch_pred[bi,row,col]),float(pranks[flat]),
                                    float(gtgrid[row,col]) if gtgrid is not None else None,
                                    float(granks[flat]) if granks is not None else None,
                                    json.dumps(topids),json.dumps(topvals),json.dumps(active),json.dumps(aw)))
                                memberships.extend((node_id,stage,"topk",p) for p in topids)
                                memberships.extend((node_id,stage,"active",p) for p in active)
                                node_id += 1
                    con.executemany("INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)
                    con.executemany("INSERT INTO memberships VALUES (?,?,?,?)", memberships)
                    if stage == "stage4":
                        for bi in range(b):
                            ids = np.arange(stage_first_node + bi*h*w, stage_first_node + (bi+1)*h*w)
                            store_frame_cache(con, this_frames[bi], stage, ids, info, bi,
                                              features[bi], last_rgb[bi].permute(1,2,0).numpy())
                processed += b
                con.commit()
                progress.set_postfix(patches=f"{node_id:,}", refresh=False)
                del result, pred, batch, rgb, sal, last_rgb
        if node_id == 0:
            raise RuntimeError("No full windows/patches were extracted.")
        con.execute("ANALYZE")
        con.commit()
        manifest.update(status="complete", completed_at=now(), num_windows=processed,
                        num_nodes=node_id, elapsed_seconds=round(time.monotonic()-started,2),
                        native_active_count_histogram={f"{s}:{count}": n for (s,count),n in sorted(active_counts.items())})
        dump_json(outdir / "manifest.json", manifest)
        rebuild_rank_cache(con, max(PATCHES_PER_PROTOTYPE, RANK_CACHE_CAPACITY))
        write_viewer(outdir, con)
        print(f"Complete: {node_id:,} patch nodes. Full graph: {dbpath}", flush=True)
        print("Open graph_viewer.html. For all-node exploration, set ACTION='serve' and rerun.")
    except BaseException as exc:
        con.rollback()
        manifest.update(status="failed", failed_at=now(), error=str(exc),
                        committed_node_count=int(con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]))
        dump_json(outdir / "manifest.json", manifest)
        raise
    finally:
        con.close()
        capture.close()

def require_complete(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Graph extraction is {manifest.get('status')!r}, not complete. "
                           "Check manifest.json; use a new OUTPUT_DIR for a fresh run.")
    return manifest


def normalize_dataset_roots(roots, default_val_dir):
    if roots is None:
        return {"val": str(default_val_dir)}
    if isinstance(roots, (str, Path)):
        p = Path(roots)
        return {p.name or "dataset": str(p)}
    if isinstance(roots, dict) and roots:
        return {str(k): str(v) for k, v in roots.items()}
    raise TypeError("DATASET_ROOTS must be None, a path, or a nonempty mapping.")


def has_table(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def similarity_settings():
    return {"version": SIMILARITY_VERSION, "formula": SIMILARITY_FORMULA,
            "rgb_size": RGB_DESCRIPTOR_SIZE, "rgb_range": [0, 1],
            "rgb_resize": "bilinear, per-channel float32, uniform input-grid crop",
            "structure": "cosine of per-channel spatially centered RGB",
            "feature_module": STAGE4_FEATURE_MODULE,
            "feature_output_path": list(STAGE4_FEATURE_OUTPUT_PATH),
            "feature_layout": STAGE4_FEATURE_LAYOUT}


def ensure_graph_schema(con):
    # Additional tables only: original frames/nodes/memberships stay unchanged.
    con.executescript("""
        CREATE TABLE IF NOT EXISTS visual_descriptors (
            node_id INTEGER PRIMARY KEY, feature_dim INTEGER NOT NULL,
            stage4_vector BLOB NOT NULL, rgb_vector BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS similarity_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS prototype_banks (
            stage TEXT PRIMARY KEY, prototype_count INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS prototype_activation_frames (
            frame_id INTEGER NOT NULL, stage TEXT NOT NULL,
            patch_count INTEGER NOT NULL, prototype_count INTEGER NOT NULL,
            node_ids BLOB NOT NULL, activations BLOB NOT NULL,
            PRIMARY KEY(frame_id,stage));
        CREATE TABLE IF NOT EXISTS prototype_patches (
            stage TEXT NOT NULL, prototype_id INTEGER NOT NULL, rank INTEGER NOT NULL,
            node_id INTEGER NOT NULL, activation REAL NOT NULL,
            PRIMARY KEY(stage,prototype_id,rank), UNIQUE(stage,prototype_id,node_id));
        CREATE INDEX IF NOT EXISTS prototype_patch_lookup
            ON prototype_patches(stage,prototype_id,rank,node_id,activation);
        CREATE TABLE IF NOT EXISTS prototype_graph_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    expected = json.dumps(similarity_settings(), sort_keys=True)
    row = con.execute("SELECT value FROM similarity_metadata WHERE key='settings'").fetchone()
    if row and row[0] != expected:
        if con.execute("SELECT 1 FROM visual_descriptors LIMIT 1").fetchone():
            raise ValueError("Visual descriptor settings differ from the populated cache. Restore its original settings or use a new OUTPUT_DIR.")
    con.execute("INSERT OR REPLACE INTO similarity_metadata VALUES ('settings',?)", (expected,))
    con.commit()


def pack_vector(a):
    return zlib.compress(np.asarray(a, dtype='<f4').tobytes(), level=3)


@lru_cache(maxsize=4096)
def unpack_vector(blob):
    a = np.frombuffer(zlib.decompress(blob), dtype='<f4').astype(np.float64)
    a.flags.writeable = False
    return a


def rgb_descriptor(rgb, row, col, grid_h, grid_w):
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all() or a.min() < -.01 or a.max() > 1.01:
        raise ValueError("Expected finite HWC loader RGB in [0,1].")
    h, w = a.shape[:2]
    crop = a[round(row*h/grid_h):round((row+1)*h/grid_h),
             round(col*w/grid_w):round((col+1)*w/grid_w)].clip(0, 1)
    if not crop.size:
        raise ValueError("Empty RGB patch.")
    size = (int(RGB_DESCRIPTOR_SIZE), int(RGB_DESCRIPTOR_SIZE))
    return np.stack([np.asarray(Image.fromarray(crop[..., c]).resize(size, Image.Resampling.BILINEAR))
                     for c in range(3)], axis=-1).clip(0, 1).astype(np.float32).reshape(-1)


def store_descriptor(con, node_id, feature, rgb):
    feature, rgb = np.asarray(feature).reshape(-1), np.asarray(rgb).reshape(-1)
    if not np.isfinite(feature).all() or not np.isfinite(rgb).all():
        raise ValueError("Non-finite visual descriptor.")
    con.execute("INSERT OR REPLACE INTO visual_descriptors VALUES (?,?,?,?)",
                (int(node_id), int(feature.size), pack_vector(feature), pack_vector(rgb)))


def store_frame_cache(con, frame_id, stage, node_ids, info, bi, features, rgb, preserve_descriptors=False):
    weights = info["weights"]
    if weights is None:
        raise RuntimeError(f"{stage}: visual_activations is required to rank all prototypes; no logit fallback is used.")
    h, w, m = info["h"], info["w"], info["m"]
    ids = np.asarray(node_ids, dtype='<i8').reshape(-1)
    a = np.asarray(weights[bi], dtype='<f4').reshape(h*w, m)
    if len(ids) != h*w or len(set(ids.tolist())) != len(ids) or not np.isfinite(a).all():
        raise ValueError("Invalid dense prototype activation block.")
    bank = con.execute("SELECT prototype_count FROM prototype_banks WHERE stage=?", (stage,)).fetchone()
    if bank and int(bank[0]) != m:
        raise RuntimeError("Prototype bank size changed between frames.")
    con.execute("INSERT OR IGNORE INTO prototype_banks VALUES (?,?)", (stage, m))
    con.execute("INSERT OR REPLACE INTO prototype_activation_frames VALUES (?,?,?,?,?,?)",
                (int(frame_id), stage, h*w, m, zlib.compress(ids.tobytes(), 3), pack_vector(a)))
    con.execute("DELETE FROM prototype_graph_metadata WHERE key='ranking'")
    existing = set()
    if preserve_descriptors:
        existing = {int(r[0]) for r in con.execute("SELECT d.node_id FROM visual_descriptors d JOIN nodes n ON n.id=d.node_id WHERE n.frame_id=? AND n.stage=?", (int(frame_id), stage))}
    for idx, nid in enumerate(ids):
        if int(nid) in existing:
            continue
        row, col = divmod(idx, w)
        store_descriptor(con, nid, features[row, col], rgb_descriptor(rgb, row, col, h, w))


def cosine(a, b, flat_equal=False):
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return (1. if na < 1e-12 and nb < 1e-12 else 0.) if flat_equal else None
    return float(np.clip(np.dot(a/na, b/nb), -1, 1))


def visual_metrics(left, right):
    if left is None or right is None:
        return {"visual_score": None, "similarity_missing_reason": "Missing visual descriptor/signature; run ACTION='cache_similarity'"}
    fa, ra = left
    fb, rb = right
    fc = cosine(fa, fb)
    if fc is None or ra.shape != rb.shape or not ra.size or ra.size % 3:
        return {"visual_score": None, "similarity_missing_reason": "Zero or incompatible stage4/RGB descriptor"}
    mae = float(np.mean(np.abs(ra-rb)))
    aa, bb = ra.reshape(-1, 3), rb.reshape(-1, 3)
    sc = cosine((aa-aa.mean(0)).reshape(-1), (bb-bb.mean(0)).reshape(-1), flat_equal=True)
    if sc is None:
        return {"visual_score": None, "similarity_missing_reason": "Invalid RGB structure descriptor"}
    return {"stage4_cosine": fc, "rgb_mae": mae, "rgb_structure_cosine": sc,
            "visual_score": fc + (1-mae) + sc, "similarity_missing_reason": None}


def merge_top_matches(old_ids, old_scores, ids, scores, capacity):
    """Exact bounded top-N per prototype, deterministic ascending node-ID ties.

    Vectorized lexsort on bounded frame+candidate blocks also resolves ties at
    the cut boundary correctly (argpartition alone does not).
    """
    scores = np.asarray(scores, dtype=np.float32)
    m = scores.shape[1]
    ids2 = np.broadcast_to(np.asarray(ids, dtype=np.int64)[:, None], scores.shape)
    if old_scores is not None:
        ids2 = np.concatenate((old_ids, ids2), axis=0)
        scores = np.concatenate((old_scores, scores), axis=0)
    order = np.lexsort((ids2, -scores), axis=0)[:capacity]
    return np.take_along_axis(ids2, order, axis=0), np.take_along_axis(scores, order, axis=0)


def require_dense_cache(con):
    if not has_table(con, "prototype_activation_frames") or not has_table(con, "prototype_banks"):
        raise RuntimeError("Existing patch database needs ACTION='cache_similarity' once to cache ALL stage4 activations.")
    m = con.execute("SELECT prototype_count FROM prototype_banks WHERE stage='stage4'").fetchone()
    if not m:
        raise RuntimeError("No stage4 prototype bank cached. Run ACTION='cache_similarity'.")
    # Verify frame coverage and all patch-ID mappings without decoding large blobs.
    missing = con.execute("""SELECT COUNT(*) FROM (
        SELECT n.frame_id FROM nodes n LEFT JOIN prototype_activation_frames a
          ON a.frame_id=n.frame_id AND a.stage=n.stage
        WHERE n.stage='stage4' GROUP BY n.frame_id
        HAVING COUNT(*) != MAX(a.patch_count) OR MAX(a.patch_count) IS NULL
          OR MAX(a.prototype_count) != ?)""", (int(m[0]),)).fetchone()[0]
    if missing:
        raise RuntimeError(f"{missing} stage4 frames lack a complete activation cache. Run ACTION='cache_similarity'.")
    return int(m[0])


def rebuild_rank_cache(con, capacity):
    m = require_dense_cache(con)
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError("N / ranking capacity must be positive.")
    old_ids = old_scores = None
    count = 0
    for frame in con.execute("SELECT * FROM prototype_activation_frames WHERE stage='stage4' ORDER BY frame_id"):
        ids = np.frombuffer(zlib.decompress(frame["node_ids"]), dtype='<i8')
        a = np.frombuffer(zlib.decompress(frame["activations"]), dtype='<f4').reshape(frame["patch_count"], m)
        if len(ids) != len(a) or not np.isfinite(a).all():
            raise ValueError("Corrupt activation cache.")
        old_ids, old_scores = merge_top_matches(old_ids, old_scores, ids, a, capacity)
        count += len(ids)
    if old_ids is None:
        raise RuntimeError("No cached patches to rank.")
    con.execute("DELETE FROM prototype_patches WHERE stage='stage4'")
    con.executemany("INSERT INTO prototype_patches VALUES (?,?,?,?,?)",
        (("stage4", p, r+1, int(old_ids[r, p]), float(old_scores[r, p]))
         for p in range(m) for r in range(len(old_ids))))
    metadata = {"capacity": capacity, "patch_count": count, "prototype_count": m,
                "activation_source": "visual_activations", "ties": "ascending patch node ID",
                "created_at": now()}
    con.execute("INSERT OR REPLACE INTO prototype_graph_metadata VALUES ('ranking',?)", (json.dumps(metadata),))
    con.commit()
    print(f"Indexed exact top-{min(capacity,count)} matches for {m} prototypes across {count:,} patches.", flush=True)


def ensure_rank_cache(directory, con=None):
    if int(PATCHES_PER_PROTOTYPE) < 1 or int(RANK_CACHE_CAPACITY) < 1 or int(GALLERY_COLUMNS) < 1:
        raise ValueError("PATCHES_PER_PROTOTYPE, RANK_CACHE_CAPACITY and GALLERY_COLUMNS must be positive.")
    own = con is None
    if own:
        con = connect_db(Path(directory)/"graph.sqlite")
    try:
        ensure_graph_schema(con)
        m = require_dense_cache(con)
        row = con.execute("SELECT value FROM prototype_graph_metadata WHERE key='ranking'").fetchone()
        meta = json.loads(row[0]) if row else {}
        patches = int(con.execute("SELECT COUNT(*) FROM nodes WHERE stage='stage4'").fetchone()[0])
        cached = int(con.execute("SELECT COUNT(*) FROM prototype_patches WHERE stage='stage4'").fetchone()[0])
        if (meta.get("capacity", 0) < PATCHES_PER_PROTOTYPE or meta.get("patch_count") != patches
                or meta.get("prototype_count") != m or cached != m*min(patches, meta.get("capacity", 0))):
            rebuild_rank_cache(con, max(PATCHES_PER_PROTOTYPE, RANK_CACHE_CAPACITY))
    finally:
        if own:
            con.close()


class Stage4Capture:
    """Observe native backbone outputs, including direct forward/helper calls."""
    def __init__(self, model):
        modules = dict(model.named_modules())
        if STAGE4_FEATURE_MODULE not in modules:
            raise RuntimeError(f"No feature module {STAGE4_FEATURE_MODULE!r}; set STAGE4_FEATURE_MODULE. Available: {list(modules)[:80]}")
        self.module = modules[STAGE4_FEATURE_MODULE]
        self.value = self.source = None
        self.calls, self.output_errors = [], []
        self.original = {}
        self.reported = False
        self.handle = self.module.register_forward_hook(self._hook)
        try:
            for name in ("forward", "forward_features", "extract_features"):
                original = getattr(self.module, name, None)
                if callable(original):
                    self.original[name] = (name in self.module.__dict__, self.module.__dict__.get(name))
                    def wrapped(*args, _original=original, _name=name, **kwargs):
                        output = _original(*args, **kwargs)
                        self._observe(output, STAGE4_FEATURE_MODULE+"."+_name)
                        return output
                    setattr(self.module, name, wrapped)
        except BaseException:
            self.close()
            raise

    def _observe(self, output, source):
        description = ("dict keys="+repr(list(output)[:20])) if isinstance(output, dict) else str(getattr(output, "shape", type(output).__name__))
        self.calls.append(source+" -> "+description)
        self.calls = self.calls[-12:]
        value = output
        try:
            for part in STAGE4_FEATURE_OUTPUT_PATH:
                value = value[part]
            if not callable(getattr(value, "detach", None)):
                raise TypeError("selected output is not a tensor")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            self.output_errors.append(source+": "+str(exc))
            self.output_errors = self.output_errors[-8:]
            return
        self.value, self.source = value.detach(), source

    def _hook(self, module, inputs, output):
        self._observe(output, STAGE4_FEATURE_MODULE+".__call__")

    def reset(self):
        self.value = self.source = None
        self.calls.clear()
        self.output_errors.clear()

    def last_frame(self, info, batch_size, layout=None):
        if self.value is None:
            raise RuntimeError(f"Native stage4 features not captured. Module={STAGE4_FEATURE_MODULE!r}; output path={STAGE4_FEATURE_OUTPUT_PATH!r}. "
                               f"Observed calls: {self.calls}. Selection errors: {self.output_errors}. "
                               "For tensor output set STAGE4_FEATURE_OUTPUT_PATH=(); for tuple/list output select its index.")
        a = self.value.detach().float().cpu().numpy()
        b, t, h, w = batch_size, info["t"], info["h"], info["w"]
        ok = lambda ft: ft >= 1 and (ft == t or t == 1)
        layouts = {}
        if a.ndim == 5:
            if a.shape[0] == b and a.shape[3:] == (h,w) and ok(a.shape[2]):
                layouts["BCTHW"] = lambda: a[:,:,-1].transpose(0,2,3,1)
            if a.shape[0] == b and a.shape[2:4] == (h,w) and ok(a.shape[1]):
                layouts["BTHWC"] = lambda: a[:,-1]
            if a.shape[0] == b and a.shape[3:] == (h,w) and ok(a.shape[1]):
                layouts["BTCHW"] = lambda: a[:,-1].transpose(0,2,3,1)
        if a.ndim == 4 and t == 1:
            if a.shape[0] == b and a.shape[2:] == (h,w):
                layouts["BCHW"] = lambda: a.transpose(0,2,3,1)
            if a.shape[:3] == (b,h,w):
                layouts["BHWC"] = lambda: a
        choice = layout or STAGE4_FEATURE_LAYOUT
        if choice == "auto":
            if len(layouts) > 1 and "BCTHW" in layouts:
                if t != 1 and a.shape[1] == t and a.shape[2] != t:
                    layouts.pop("BCTHW")
                else:
                    layouts = {"BCTHW": layouts["BCTHW"]}
            if len(layouts) != 1:
                raise RuntimeError(f"Ambiguous/incompatible stage4 shape {a.shape}, B={b},T={t},H={h},W={w}; candidates={list(layouts)}. Set STAGE4_FEATURE_LAYOUT.")
            choice = next(iter(layouts))
        if choice not in layouts:
            raise RuntimeError(f"Stage4 shape {a.shape} incompatible with {choice} and prototype grid {h}x{w}.")
        result = layouts[choice]()
        if not np.isfinite(result).all():
            raise ValueError("Non-finite native stage4 features.")
        if not self.reported:
            print(f"Feature source: {self.source}, layout {choice}, shape {a.shape}, concept T={t}.", flush=True)
            self.reported = True
        return result

    def close(self):
        if getattr(self, "handle", None):
            self.handle.remove()
            self.handle = None
        for name, (existed, previous) in getattr(self, "original", {}).items():
            if existed:
                setattr(self.module, name, previous)
            else:
                delattr(self.module, name)
        self.original = {}
        self.value = None
def last_rgb_frame(rgb):
    if rgb.ndim != 5:
        raise RuntimeError("Expected BTCHW or BTHWC RGB.")
    if rgb.shape[2] == 3:
        a = rgb[:, -1]
    elif rgb.shape[-1] == 3:
        a = rgb[:, -1].permute(0, 3, 1, 2)
    else:
        raise RuntimeError(f"Unsupported RGB layout {tuple(rgb.shape)}.")
    a = a.detach().float().cpu()
    if a.max() > 2:
        a = a/255
    if not np.isfinite(a.numpy()).all() or a.min() < -.01 or a.max() > 1.01:
        raise ValueError("Loader RGB must be finite in [0,1] or [0,255].")
    return a


def cache_similarity(directory):
    """Resume the all-prototype cache upgrade; one inference per missing frame."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    directory = Path(directory).resolve()
    manifest = require_complete(directory)
    if int(manifest.get("top_k", TOP_K)) != TOP_K or int(manifest.get("seed", SEED)) != SEED:
        raise ValueError("Use the original graph's TOP_K and SEED for backfill.")
    con = connect_db(directory/"graph.sqlite")
    capture = None
    try:
        ensure_graph_schema(con)
        pending = con.execute("""SELECT DISTINCT f.* FROM frames f JOIN nodes n ON n.frame_id=f.id
            LEFT JOIN prototype_activation_frames a ON a.frame_id=f.id AND a.stage=n.stage
            LEFT JOIN visual_descriptors d ON d.node_id=n.id
            WHERE n.stage='stage4' AND (a.frame_id IS NULL OR d.node_id IS NULL OR a.patch_count != n.grid_h*n.grid_w)
            ORDER BY f.dataset,f.dataset_index""").fetchall()
        if pending:
            ev = load_evaluation()
            ev.train_cfg.set_seed(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            if torch.cuda.is_available() and (GPU_ID < 0 or GPU_ID >= torch.cuda.device_count()):
                raise ValueError(f"GPU_ID={GPU_ID} unavailable.")
            device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
            ckpt = str(CHECKPOINT_PATH or manifest.get("checkpoint") or ev.CHECKPOINT_PATH)
            if sha256_file(ckpt) != manifest.get("checkpoint_sha256"):
                raise ValueError("Checkpoint differs from original graph; refusing mixed caches.")
            model = ev.build_model(device, device)
            ev.load_checkpoint(model, ckpt)
            model.eval()
            enabled, dtype_fn = getattr(ev.train_cfg,"_amp_enabled",None), getattr(ev.train_cfg,"_amp_dtype",None)
            amp = bool(USE_EVAL_AMP and callable(enabled) and enabled(device))
            dtype = dtype_fn(device) if amp and callable(dtype_fn) else torch.float16
            if "amp_enabled" in manifest and bool(manifest["amp_enabled"]) != amp:
                raise ValueError("Backfill requires original AMP setting/device.")
            if amp and manifest.get("amp_dtype") and manifest["amp_dtype"] != str(dtype):
                raise ValueError("Backfill requires original AMP dtype.")
            capture = Stage4Capture(model)
            window_len, stride = int(manifest["window_len"]), int(manifest["stride"])
            workers = int(ev.NUM_WORKERS if NUM_WORKERS is None else NUM_WORKERS)
            manifest.update(prototype_cache_status="building")
            dump_json(directory/"manifest.json", manifest)
            for label in dict.fromkeys(r["dataset"] for r in pending):
                frames = [dict(r) for r in pending if r["dataset"] == label]
                root = manifest["dataset_roots"][label]
                dataset = ev.DatasetLoader(str(root),window_len=window_len,stride=stride)
                for f in frames:
                    di = f["dataset_index"]
                    if di >= len(dataset.windows) or tuple(map(str,dataset.windows[di])) != (str(f["video"]),str(f["window_start"])):
                        raise RuntimeError("Dataset windows changed since graph extraction.")
                    names = frame_names_for_video(dataset,root,f["video"])
                    if f["target_index"] >= len(names) or names[f["target_index"]] != f["target_name"]:
                        raise RuntimeError("Target frame names changed since graph extraction.")
                loader = DataLoader(Subset(dataset,[f["dataset_index"] for f in frames]),batch_size=1,shuffle=False,
                    num_workers=workers,collate_fn=ev.video_saliency_collate_fn,pin_memory=device.type=="cuda")
                for f, batch in tqdm(zip(frames,loader),total=len(frames),desc=f"All-prototype cache {label}"):
                    names,rgb,_,lengths,valid = unpack_batch(batch)
                    if any(int(n)!=window_len for n in lengths) or tuple(valid.shape)!=(1,window_len) or not bool(valid.all()):
                        raise RuntimeError("Unexpected short/padded backfill window.")
                    last = last_rgb_frame(rgb)
                    if str(names[0]) != f["video"] or tuple(last.shape[-2:]) != (f["input_h"],f["input_w"]):
                        raise RuntimeError("Backfill frame identity/coordinates changed.")
                    capture.reset()
                    with torch.inference_mode(), torch.autocast(device.type,dtype=dtype,enabled=amp):
                        result = model(rgb.to(device,non_blocking=True),return_details=True,
                                       return_concept_losses=False,return_decoder_diagnostics=False)
                    info = extract_stage(result["concept_out"]["stage4"],1,"stage4")
                    rows = con.execute("SELECT id FROM nodes WHERE frame_id=? AND stage='stage4' ORDER BY row,col",(f["id"],)).fetchall()
                    stored = [get_node(con,r[0]) for r in rows]
                    if len(stored) != info["h"]*info["w"]:
                        raise RuntimeError("Stored grid no longer agrees with native model outputs.")
                    for flat,n in enumerate(stored):
                        row,col = divmod(flat,info["w"])
                        top = list(map(int,info["top"][0,row,col]))
                        active = sorted(set(map(int,info["active"][0,row,col][info["mask"][0,row,col]])))
                        if ((n["grid_h"],n["grid_w"],n["row"],n["col"]) != (info["h"],info["w"],row,col)
                                or top != n["topk_ids"] or active != n["active_ids"]):
                            raise RuntimeError(f"Native assignments/grid changed for node {n['id']}; refusing mixed cache.")
                    features = capture.last_frame(info,1)
                    store_frame_cache(con,f["id"],"stage4",[n["id"] for n in stored],info,0,
                                      features[0],last[0].permute(1,2,0).numpy(),preserve_descriptors=True)
                    con.commit()
                    del result,batch,rgb,last,features,info
        ensure_rank_cache(directory,con)
        manifest.update(prototype_cache_status="complete",prototype_cached_at=now(),
                        graph_version=GRAPH_VERSION,similarity_settings=similarity_settings())
        dump_json(directory/"manifest.json",manifest)
        write_viewer(directory,con)
    except BaseException as exc:
        con.rollback()
        manifest.update(prototype_cache_status="failed",prototype_cache_error=str(exc))
        dump_json(directory/"manifest.json",manifest)
        raise
    finally:
        if capture:
            capture.close()
        con.close()


def load_descriptors(con, ids):
    descriptors = {}
    ids = list(ids)
    for offset in range(0,len(ids),500):
        group = ids[offset:offset+500]
        marks = ','.join('?' for _ in group)
        for r in con.execute(f"SELECT * FROM visual_descriptors WHERE node_id IN ({marks})",group):
            feature,rgb = unpack_vector(r["stage4_vector"]),unpack_vector(r["rgb_vector"])
            if len(feature) != r["feature_dim"]:
                raise ValueError(f"Corrupt feature for patch {r['node_id']}.")
            descriptors[int(r["node_id"])] = feature,rgb
    return descriptors


def summarize_prototype(patches, descriptors):
    """All summary values describe exactly the N linked patch occurrences."""
    percentiles = np.asarray([p["pred_percentile"] for p in patches],dtype=np.float64)*100
    valid = [descriptors[p["id"]] for p in patches if p["id"] in descriptors]
    # Require full descriptor coverage for the prototype's mean signature.
    signature = None
    if valid and len(valid) == len(patches):
        try:
            signature = (np.stack([v[0] for v in valid]).mean(0), np.stack([v[1] for v in valid]).mean(0))
        except ValueError:
            pass
    total_pairs = len(patches)*(len(patches)-1)//2
    score_sum = 0.
    valid_pairs = 0
    for i,a in enumerate(patches):
        for b in patches[i+1:]:
            metric = visual_metrics(descriptors.get(a["id"]),descriptors.get(b["id"]))
            if metric["visual_score"] is not None:
                score_sum += metric["visual_score"]
                valid_pairs += 1
    stats = {"patch_count":len(patches), "descriptor_count":len(valid),
             "mean_pairwise_visual_similarity":score_sum/valid_pairs if valid_pairs else None,
             "valid_pair_count":valid_pairs,"total_pair_count":total_pairs,
             "mean_saliency_percentile":float(percentiles.mean()) if percentiles.size else None,
             "std_saliency_percentile":float(percentiles.std(ddof=0)) if percentiles.size else None,
             "saliency_percentile_scale":[0,100],"std_definition":"population standard deviation (ddof=0)",
             "scope":"the N linked strongest-activation patches, not all activating patches"}
    return signature,stats


def build_graph(con, directory, n=None):
    n = PATCHES_PER_PROTOTYPE if n is None else int(n)
    m = require_dense_cache(con)
    if n < 1:
        raise ValueError("N must be at least 1.")
    row = con.execute("SELECT value FROM prototype_graph_metadata WHERE key='ranking'").fetchone()
    if not row or json.loads(row[0]).get("capacity",0) < n:
        raise RuntimeError("Rank cache capacity too small; regenerate view/serve to rebuild it without inference.")
    matches = list(con.execute("SELECT * FROM prototype_patches WHERE stage='stage4' AND rank<=? ORDER BY prototype_id,rank",(n,)))
    node_ids = sorted({int(r["node_id"]) for r in matches})
    # Fetch nodes in chunks, not one connection/query per patch.
    patches = {}
    for offset in range(0,len(node_ids),500):
        group = node_ids[offset:offset+500]
        marks = ','.join('?' for _ in group)
        rows = con.execute(f"""SELECT n.*,f.dataset,f.video,f.target_name,f.target_index,f.window_start,
            f.dataset_index,f.source_path,f.thumb_path,f.input_h,f.input_w FROM nodes n JOIN frames f
            ON f.id=n.frame_id WHERE n.id IN ({marks})""",group)
        for r in rows:
            d = dict(r)
            for key in ("topk_ids","topk_logits","active_ids","active_weights"):
                d[key] = json.loads(d[key])
            patches[d["id"]] = d
    descriptors = load_descriptors(con,node_ids)
    ranked = {p:[] for p in range(m)}
    for r in matches:
        ranked[int(r["prototype_id"])].append(r)
    prototypes,signatures,edges = [],{},[]
    for p in range(m):
        pid = f"prototype:stage4:{p}"
        linked = [patches[int(r["node_id"])] for r in ranked[p]]
        signature,stats = summarize_prototype(linked,descriptors)
        signatures[p] = signature
        prototypes.append({"id":pid,"node_type":"prototype","stage":"stage4","prototype_id":p,
                           "label":f"P{p}","stats":stats,
                           "exemplar_patch_id":f"patch:{linked[0]['id']}" if linked else None})
        for r in ranked[p]:
            patch = patches[int(r["node_id"])]
            edges.append({"source":pid,"target":f"patch:{patch['id']}","edge_type":"prototype_patch",
                          "prototype_id":p,"rank":int(r["rank"]),"activation":float(r["activation"]),
                          "forward_valid":p in patch["active_ids"],
                          **visual_metrics(signature,descriptors.get(patch["id"]))})
    for p in range(m):
        for q in range(p+1,m):
            edges.append({"source":f"prototype:stage4:{p}","target":f"prototype:stage4:{q}",
                          "edge_type":"prototype_prototype",**visual_metrics(signatures[p],signatures[q])})
    patch_nodes = [patches[nid] for nid in node_ids]
    add_preview_images(patch_nodes,directory)
    image_pool = {}
    for patch in patch_nodes:
        frame_image = patch.pop("frame_image",None)
        if frame_image:
            image_pool[str(patch["frame_id"])] = frame_image
        patch.update(patch_id=patch["id"],id=f"patch:{patch['id']}",node_type="patch",label=f"patch {patch['id']}")
    return {"graph_version":GRAPH_VERSION,"nodes":prototypes+patch_nodes,"edges":edges,
            "image_pool":image_pool,"prototype_options":[p["prototype_id"] for p in prototypes],
            "selected_prototypes":None,"patches_per_prototype":n,"gallery_columns":GALLERY_COLUMNS,
            "total_prototypes":m,"total_stored_patches":int(con.execute("SELECT COUNT(*) FROM nodes WHERE stage='stage4'").fetchone()[0]),
            "visual_similarity":{"formula":SIMILARITY_FORMULA,"signature":"mean feature and RGB descriptors of each prototype's N linked patches",
                                 "color_range":list(EDGE_COLOR_RANGE) if EDGE_COLOR_RANGE is not None else None},
            "scope":"All stage4 prototypes; exact top-N matches over all cached target-frame patch occurrences. No prototype frequency filtering."}


def select_prototypes(graph, selection=None):
    if selection is None:
        return graph
    requested = set(map(int,selection))
    unknown = requested-set(graph["prototype_options"])
    if unknown:
        raise ValueError(f"Unknown prototype IDs: {sorted(unknown)}")
    protos = {f"prototype:stage4:{p}" for p in requested}
    edges = [e for e in graph["edges"] if (e["edge_type"]=="prototype_patch" and e["source"] in protos)
             or (e["edge_type"]=="prototype_prototype" and e["source"] in protos and e["target"] in protos)]
    ids = protos|{e["target"] for e in edges if e["edge_type"]=="prototype_patch"}
    nodes = [n for n in graph["nodes"] if n["id"] in ids]
    frames = {str(n["frame_id"]) for n in nodes if n["node_type"]=="patch"}
    return {**graph,"nodes":nodes,"edges":edges,"selected_prototypes":sorted(requested),
            "image_pool":{k:v for k,v in graph["image_pool"].items() if k in frames}}


def write_viewer(directory, con=None, graph=None):
    directory = Path(directory)
    if EDGE_COLOR_RANGE is not None:
        low,high = map(float,EDGE_COLOR_RANGE)
        if not math.isfinite(low) or not math.isfinite(high) or low>=high:
            raise ValueError("EDGE_COLOR_RANGE must be a finite increasing pair.")
    if graph is None:
        ensure_rank_cache(directory,con)
        own = con is None
        if own:
            con = connect_db(directory/"graph.sqlite",readonly=True)
        try:
            graph = build_graph(con,directory)
        finally:
            if own:
                con.close()
    dump_json(directory/"neighborhood.json",graph)
    encoded = json.dumps(graph,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    (directory/"graph_viewer.html").write_text(VIEWER_HTML.replace('__PAYLOAD__',encoded),encoding='utf-8')
    print(f"Viewer: {directory/'graph_viewer.html'} ({graph['total_prototypes']} prototypes; N={graph['patches_per_prototype']}).",flush=True)
    return graph


def serve_graph(directory):
    directory = Path(directory).resolve()
    require_complete(directory)
    graph = write_viewer(directory)
    html = (directory/"graph_viewer.html").read_text().replace('const online=false;','const online=true;').encode('utf-8')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in ('/','/graph_viewer.html'):
                    content,ctype = html,'text/html; charset=utf-8'
                elif parsed.path == '/api/graph':
                    q = parse_qs(parsed.query,keep_blank_values=True)
                    value = q.get('prototypes',['ALL'])[0]
                    selected = None if value.upper()=='ALL' else [int(x) for x in value.split(',') if x.strip()]
                    content = json.dumps(select_prototypes(graph,selected),allow_nan=False,separators=(',',':')).encode()
                    ctype = 'application/json; charset=utf-8'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type',ctype)
                self.send_header('Content-Length',str(len(content)))
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                self.wfile.write(content)
            except (ValueError,KeyError,RuntimeError) as exc:
                content = json.dumps({'error':str(exc)}).encode()
                self.send_response(400)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(content)))
                self.end_headers()
                self.wfile.write(content)
    server = ThreadingHTTPServer(('127.0.0.1',SERVER_PORT),Handler)
    print(f"Open http://127.0.0.1:{SERVER_PORT} — stop with Ctrl+C. Filtering reuses precomputed matches/signatures.",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def export_edges(directory):
    directory = Path(directory)
    require_complete(directory)
    ensure_rank_cache(directory)
    with closing(connect_db(directory/'graph.sqlite',readonly=True)) as con:
        graph = build_graph(con,directory)
    path = directory/'prototype_patch_edges.csv'
    columns = ['source','target','edge_type','prototype_id','rank','activation','forward_valid',
               'stage4_cosine','rgb_mae','rgb_structure_cosine','visual_score','similarity_missing_reason']
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(graph['edges'])
    dump_json(directory/'prototype_patch_edges_metadata.json',{
        'created_at':now(),'edges':len(graph['edges']),'patches_per_prototype':PATCHES_PER_PROTOTYPE,
        'similarity':graph['visual_similarity'],'complete':True})
    print(f"Exported {len(graph['edges']):,} edges: {path}",flush=True)


VIEWER_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prototype and patch explorer</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#172236;background:#f4f6fa}*{box-sizing:border-box}body{margin:0}header{background:#fff;padding:18px 24px;border-bottom:1px solid #dce3ec}h1{font-size:23px;margin:0 0 5px}p{line-height:1.5;margin:8px 0}.muted{color:#667489;font-size:12px}.controls{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin-top:14px}label{display:grid;gap:5px;font-size:12px}button,input,select{font:inherit;border:1px solid #bac7d7;border-radius:6px;background:white;color:inherit;padding:8px}button{cursor:pointer}button:hover{background:#edf3fa}button.primary{background:#244f85;color:white}button:disabled{opacity:.5;cursor:default}button:focus-visible,summary:focus-visible{outline:3px solid #81b8ed}summary{cursor:pointer;border:1px solid #bac7d7;border-radius:6px;padding:8px;min-width:230px;list-style:none}summary::after{content:' ▾';float:right}.picker{position:relative}.picker-pop{position:absolute;left:0;top:44px;width:310px;padding:12px;background:white;border:1px solid #b7c5d8;border-radius:8px;box-shadow:0 12px 32px #17223625;z-index:5}.picker-pop input[type=search]{width:100%;margin:7px 0}.picker-actions{display:flex;gap:8px;margin:4px 0 8px}.picker-actions button{flex:1;padding:6px 8px;font-size:12px}.check{display:flex;align-items:center;gap:8px;padding:5px 2px;font-size:13px;cursor:pointer}.check input{accent-color:#244f85}.prototype-list{max-height:260px;overflow:auto;display:grid;grid-template-columns:1fr 1fr}#status{font-size:13px;padding:11px 24px;border-bottom:1px solid #dce3ec;background:#f9fbfe}.shell{display:grid;grid-template-columns:minmax(0,1fr) 340px}#scene{height:72vh;min-height:480px;position:relative;background:white}canvas{width:100%;height:100%;touch-action:none;display:block}aside{padding:20px;background:#fafbfd;border-left:1px solid #dce3ec;max-height:72vh;overflow:auto}aside h2{font-size:19px;margin:0 0 6px}.metric{padding:12px;margin:12px 0;border:1px solid #dce4ef;background:#fff;border-radius:8px}.metric strong{display:block;font-size:24px;font-weight:650;letter-spacing:-.5px;margin:5px 0}.metric span{font-size:12px;color:#60728a}.metric .title{font-size:13px;color:#244f85;font-weight:600}.metric .empty{font-size:16px;color:#60728a}.crop{max-width:135px;max-height:120px;border-radius:5px;border:1px solid #dce3ec}.frame{position:relative;margin:10px 0}.frame img{width:100%;display:block}.box{position:absolute;border:3px solid #ef6638;pointer-events:none}.pill{display:inline-block;background:#e3ebf5;border:0;margin:3px;padding:5px 7px;font-size:12px}.legend{padding:10px 24px;background:#fff;font-size:12px;border-top:1px solid #dce3ec;line-height:1.8}.gradient{display:inline-block;vertical-align:middle;width:120px;height:10px;margin:0 7px;background:linear-gradient(90deg,#2166ac,#b2182b)}.sal-gradient{background:linear-gradient(90deg,#427ca8,#e29e58,#d15a45)}#tooltip{position:fixed;pointer-events:none;max-width:380px;z-index:20;background:#172236;color:#fff;border-radius:7px;padding:10px 13px;font-size:12px;white-space:pre-line;display:none}code{font-size:11px;word-break:break-all}.patch-gallery{width:min(1480px,96vw);max-width:96vw;max-height:92vh;padding:0;border:1px solid #b7c5d8;border-radius:12px;background:#f5f8fc;color:#172236;box-shadow:0 25px 80px #0004}.patch-gallery::backdrop{background:#101d32aa}.gallery-head{padding:16px 20px;position:sticky;top:0;background:white;z-index:2;border-bottom:1px solid #dce3ec}.gallery-title{display:flex;justify-content:space-between;gap:16px;align-items:center}.gallery-title h2{font-size:19px;margin:0}.gallery-scroll{padding:16px;overflow:auto}.patch-grid{display:grid;grid-template-columns:repeat(var(--gallery-cols,8),minmax(110px,1fr));gap:10px;min-width:600px}.group-title{grid-column:1/-1;padding:12px 0 4px;margin:0;font-size:15px}.gallery-card{border:1px solid #d9e3ef;background:#fff;border-radius:8px;overflow:hidden;padding:8px;text-align:left;min-width:0}.gallery-card:hover{border-color:#356fa7}.gallery-card img,.noimage{width:100%;aspect-ratio:1;object-fit:contain;background:#e9eef5;border-radius:4px;display:block}.noimage{display:flex;align-items:center;justify-content:center;color:#667489;font-size:12px}.gallery-card b{display:block;font-size:12px;margin:6px 0}.gallery-card small{display:block;font-size:10px;line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#60728a}.badge{font-size:11px;font-weight:650;display:block;margin-bottom:7px;color:#244f85}.links{max-height:200px;overflow:auto}#error{color:#a32c35}.hint{position:absolute;bottom:13px;left:16px;font-size:11px;color:#60728a;background:#ffffffdf;padding:7px 10px;border-radius:5px;pointer-events:none}@media(max-width:850px){.shell{grid-template-columns:1fr}aside{max-height:600px;border-left:0;border-top:1px solid #dce3ec}#scene{height:62vh;min-height:390px}.picker-pop{width:280px}header{padding:16px}.patch-grid{grid-template-columns:repeat(4,minmax(110px,1fr))}}
</style></head><body>
<header><h1>Prototype and patch explorer</h1><p class="muted">Stage 4 prototypes and their strongest patch matches. Clusters group patches around a prototype; every prototype pair is connected.</p>
<div class="controls"><label>Prototypes<details id="picker" class="picker"><summary id="pickerSummary">ALL prototypes</summary><div class="picker-pop"><label class="check"><input id="allPrototypes" type="checkbox" checked>ALL prototypes</label><div class="picker-actions"><button type="button" id="selectAllPrototypes">Select all</button><button type="button" id="deselectAllPrototypes">Deselect all</button></div><input id="prototypeSearch" type="search" placeholder="Find a prototype ID…" aria-label="Find prototype ID"><div id="prototypeList" class="prototype-list"></div><p class="muted">Apply your selection with Load neighbourhood.</p></div></details></label>
<button id="load" class="primary">Load neighbourhood</button><button id="viewPatches">View patches</button><button id="fit">Reset layout</button>
<label>Edges<select id="edgeMode"><option value="all">All edges</option><option value="selected">Touching selected node</option><option value="matches">Prototype–patch edges</option><option value="prototypes">Prototype–prototype edges</option><option value="none">Hide edges</option></select></label>
<label>Find a displayed patch<input id="query" placeholder="video, frame or patch ID" style="width:190px"></label><button id="find">Find</button></div><p id="mode" class="muted"></p></header>
<div id="status" role="status" aria-live="polite"></div><div class="shell"><div id="scene"><canvas id="canvas" aria-label="Interactive prototype and patch graph"></canvas><div class="hint">Scroll to zoom · drag background to pan · drag nodes · click a prototype for statistics</div></div><aside id="details"></aside></div>
<div class="legend"><b>Nodes:</b> ◆ prototype · ● patch. Shared patches appear once and retain every prototype link; their position follows their strongest linked activation. Cluster position does not encode similarity.<br><b>Edge visual similarity:</b> low <span class="gradient"></span> high <span id="edgeLegend"></span><br><b>Patch saliency percentile:</b> low <span class="gradient sal-gradient"></span> high. Within-frame rank, not a calibrated probability. <span class="muted">Grey/dashed edges have unavailable similarity. Hover an edge for components.</span></div>
<div id="tooltip" role="tooltip"></div>
<dialog id="patchGallery" class="patch-gallery" aria-labelledby="galleryTitle"><div class="gallery-head"><div class="gallery-title"><h2 id="galleryTitle">View patches</h2><button id="closeGallery">Close</button></div><p id="gallerySummary" class="muted"></p></div><div class="gallery-scroll"><div id="galleryGrid" class="patch-grid"></div></div></dialog>
<script>
const INITIAL=__PAYLOAD__;
const online=false;
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(n,k=4)=>n==null||!Number.isFinite(n)?'Unavailable':Number(n).toFixed(k);
const canvas=$('canvas'),ctx=canvas.getContext('2d');
let data=INITIAL,nodes=[],edges=[],byId=new Map(),selected=null,hover=null,drag=null,zoom=1,panX=0,panY=0,width=800,height=600,edgeMin=0,edgeMax=3,clusters=[],dirty=true,pendingAll=true,pendingSet=new Set(INITIAL.prototype_options),requestId=0;
$('mode').textContent=(online?'Local server · ':'Offline · ')+`N = ${INITIAL.patches_per_prototype} strongest patches per prototype. Rankings cover ${INITIAL.total_stored_patches.toLocaleString()} stored stage 4 patches. Both modes use the same complete prototype graph.`;
function pickerText(){const k=pendingSet.size,m=INITIAL.prototype_options.length;$('pickerSummary').textContent=pendingAll?'ALL prototypes':`${k} prototype${k===1?'':'s'} selected`;$('allPrototypes').checked=pendingAll;$('allPrototypes').indeterminate=!pendingAll&&k>0&&k<m;}
function renderPicker(){const q=$('prototypeSearch').value.trim();$('prototypeList').innerHTML=INITIAL.prototype_options.filter(p=>String(p).includes(q)).map(p=>`<label class="check"><input type="checkbox" data-prototype="${p}" ${pendingSet.has(p)?'checked':''}>P${p}</label>`).join('');$('prototypeList').querySelectorAll('[data-prototype]').forEach(input=>input.onchange=()=>{const p=Number(input.dataset.prototype);input.checked?pendingSet.add(p):pendingSet.delete(p);pendingAll=pendingSet.size===INITIAL.prototype_options.length;pickerText();});pickerText();}
function setPrototypeSelection(all){pendingAll=!!all;pendingSet=new Set(pendingAll?INITIAL.prototype_options:[]);renderPicker();}
$('allPrototypes').onchange=()=>setPrototypeSelection($('allPrototypes').checked);
$('selectAllPrototypes').onclick=()=>setPrototypeSelection(true);
$('deselectAllPrototypes').onclick=()=>setPrototypeSelection(false);
$('prototypeSearch').oninput=renderPicker;
document.addEventListener('pointerdown',e=>{if(!$('picker').contains(e.target))$('picker').open=false;});
function salColor(v){const p=Math.max(0,Math.min(1,v));return p<.5?`rgb(${66+Math.round(p*320)},${124+Math.round(p*68)},${168-Math.round(p*160)})`:`rgb(${226-Math.round((p-.5)*34)},${158-Math.round((p-.5)*136)},${88-Math.round((p-.5)*38)})`;}
function edgeColor(v,alpha=1){if(v==null||!Number.isFinite(v))return `rgba(128,136,148,${alpha})`;const t=edgeMax>edgeMin?Math.max(0,Math.min(1,(v-edgeMin)/(edgeMax-edgeMin))):.5;return `rgba(${Math.round(33+145*t)},${Math.round(102-78*t)},${Math.round(172-129*t)},${alpha})`;}
function edgeInfo(e){return `Visual similarity: ${fmt(e.visual_score)}\nStage 4 cosine: ${fmt(e.stage4_cosine)}\nRGB MAE: ${fmt(e.rgb_mae)}\nRGB structure cosine: ${fmt(e.rgb_structure_cosine)}${e.similarity_missing_reason?'\n'+e.similarity_missing_reason:''}`;}
function visibleEdge(e){const mode=$('edgeMode').value;return mode==='all'||(mode==='selected'&&(e.a===selected||e.b===selected))||(mode==='matches'&&e.edge_type==='prototype_patch')||(mode==='prototypes'&&e.edge_type==='prototype_prototype');}
function layout(){const prototypes=nodes.filter(n=>n.node_type==='prototype');const cols=Math.max(1,Math.ceil(Math.sqrt(prototypes.length*1.35)));const radius=Math.max(58,Math.sqrt(data.patches_per_prototype)*23);const spacing=2*radius+60;clusters=[];
 prototypes.forEach((p,i)=>{p.x=(i%cols)*spacing;p.y=Math.floor(i/cols)*spacing;p.r=12;clusters.push({p,r:radius+18});});
 const owner=new Map();for(const e of edges){if(e.edge_type!=='prototype_patch')continue;const old=owner.get(e.target);if(!old||e.activation>old.activation||(e.activation===old.activation&&e.a.prototype_id<old.a.prototype_id))owner.set(e.target,e);}
 const groups=new Map();for(const n of nodes){if(n.node_type==='prototype')continue;const e=owner.get(n.id);if(!e)continue;if(!groups.has(e.source))groups.set(e.source,[]);groups.get(e.source).push(n);n.owner=e.source;n.r=5;}
 for(const [id,group] of groups){const p=byId.get(id);group.sort((a,b)=>a.patch_id-b.patch_id);group.forEach((n,i)=>{const a=i*2.399963,r=32+Math.sqrt((i+.5)/group.length)*(radius-32);n.x=p.x+Math.cos(a)*r;n.y=p.y+Math.sin(a)*r;});}fit();}
function fit(){const rect=canvas.getBoundingClientRect();width=rect.width;height=rect.height;if(!nodes.length){zoom=1;panX=panY=0;dirty=true;return;}let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const c of clusters){minX=Math.min(minX,c.p.x-c.r);maxX=Math.max(maxX,c.p.x+c.r);minY=Math.min(minY,c.p.y-c.r);maxY=Math.max(maxY,c.p.y+c.r);}for(const n of nodes){minX=Math.min(minX,n.x-20);maxX=Math.max(maxX,n.x+20);minY=Math.min(minY,n.y-20);maxY=Math.max(maxY,n.y+20);}zoom=Math.max(.02,Math.min(2,(width-50)/(maxX-minX||1),(height-50)/(maxY-minY||1)));panX=-(minX+maxX)/2*zoom;panY=-(minY+maxY)/2*zoom;dirty=true;}
function loadData(d){data=d;nodes=d.nodes.map(n=>({...n}));byId=new Map(nodes.map(n=>[n.id,n]));edges=d.edges.map(e=>({...e,a:byId.get(e.source),b:byId.get(e.target)}));selected=byId.get(selected?.id)||nodes.find(n=>n.node_type==='prototype')||null;hover=null;
 const values=edges.map(e=>e.visual_score).filter(v=>v!=null&&Number.isFinite(v));let lo=Infinity,hi=-Infinity;for(const v of values){lo=Math.min(lo,v);hi=Math.max(hi,v);}const fixed=d.visual_similarity.color_range;edgeMin=fixed?fixed[0]:values.length?lo:0;edgeMax=fixed?fixed[1]:values.length?hi:3;
 $('edgeLegend').textContent=`${edgeMin.toFixed(3)} to ${edgeMax.toFixed(3)} · ${fixed?'fixed range':'displayed-edge range'} · ${edges.length-values.length} unavailable`;
 const pc=nodes.filter(n=>n.node_type==='prototype').length,ec=edges.filter(e=>e.edge_type==='prototype_patch').length;
 $('status').textContent=`${pc} / ${d.total_prototypes} prototypes · ${nodes.length-pc} unique patches · ${ec.toLocaleString()} prototype–patch links · ${(edges.length-ec).toLocaleString()} prototype–prototype links. ${d.selected_prototypes==null?'ALL selected.':'Selection applied.'}`;
 $('viewPatches').disabled=ec===0;layout();showDetails();}
function showDetails(){const n=selected;if(!n){$('details').innerHTML='<h2>No prototypes selected</h2><p class="muted">Choose prototypes above, then load the neighbourhood.</p>';return;}
 if(n.node_type==='prototype'){const s=n.stats,linked=edges.filter(e=>e.edge_type==='prototype_patch'&&e.source===n.id).sort((a,b)=>a.rank-b.rank),exemplar=byId.get(n.exemplar_patch_id);const sal=s.mean_saliency_percentile==null?'Unavailable':`${fmt(s.mean_saliency_percentile,2)} ± ${fmt(s.std_saliency_percentile,2)}`;
 $('details').innerHTML=`<h2>Prototype ${n.prototype_id}</h2><p class="muted">Stage 4 · ${s.patch_count} strongest-activation patches</p><div class="metric"><div class="title">Pairwise visual similarity</div><strong class="${s.mean_pairwise_visual_similarity==null?'empty':''}">${fmt(s.mean_pairwise_visual_similarity)}</strong><span>Mean across ${s.valid_pair_count} / ${s.total_pair_count} unordered patch pairs${s.total_pair_count===0?' · at least 2 patches needed':''}</span></div><div class="metric"><div class="title">Saliency percentile · mean ± std</div><strong class="${s.mean_saliency_percentile==null?'empty':''}">${sal}</strong><span>Within-frame percentiles (0–100)<br>Population standard deviation · ${s.patch_count} patches</span></div>${exemplar?.crop_image?`<p class="muted">Strongest-activation exemplar</p><img class="crop" src="${exemplar.crop_image}" alt="Strongest patch for prototype ${n.prototype_id}">`:''}<p><button id="prototypeGallery" class="primary">View patches for P${n.prototype_id}</button></p><p class="muted">Statistics use exactly the N connected patches. The pairwise mean compares patches with one another; edge colours compare them with the prototype's mean visual signature.</p><p class="muted">A strongest match may be forward-inactive. All patch occurrences are eligible, including those from the same frame or video.</p><p><b>Connected patches</b></p><div class="links">${linked.map(e=>`<p><button class="pill" data-select="${esc(e.target)}">#${e.rank} · patch ${e.b.patch_id}</button><span class="muted"> activation ${fmt(e.activation)} · ${e.forward_valid?'valid':'inactive'}<br>Visual similarity ${fmt(e.visual_score)}</span></p>`).join('')}</div>`;
 $('prototypeGallery').onclick=()=>openGallery([n.prototype_id]);
 }else{const links=edges.filter(e=>e.edge_type==='prototype_patch'&&e.target===n.id).sort((a,b)=>b.activation-a.activation);const frame=data.image_pool[String(n.frame_id)];$('details').innerHTML=`<h2>Patch ${n.patch_id}</h2><p>${esc(n.dataset)} / ${esc(n.video)} / ${esc(n.target_name)}</p><p class="muted">${esc(n.stage)} · row ${n.row}, column ${n.col} · ${n.grid_h} × ${n.grid_w} grid</p>${n.crop_image?`<img class="crop" src="${n.crop_image}" alt="Patch crop">`:''}${frame?`<div class="frame"><img src="${frame}" alt="Full frame"><div class="box" style="left:${100*n.col/n.grid_w}%;top:${100*n.row/n.grid_h}%;width:${100/n.grid_w}%;height:${100/n.grid_h}%"></div></div>`:''}<div class="metric"><div class="title">Within-frame saliency percentile</div><strong>${fmt(n.pred_percentile*100,2)}</strong><span>Predicted score ${fmt(n.pred_score,5)}${n.gt_percentile==null?'':` · GT percentile ${fmt(n.gt_percentile*100,2)}`}</span></div><p><b>Connected prototypes</b></p>${links.map(e=>`<p><button class="pill" data-select="${esc(e.source)}">P${e.prototype_id}</button><span class="muted">rank #${e.rank} · activation ${fmt(e.activation)}<br>Visual similarity ${fmt(e.visual_score)} · ${e.forward_valid?'forward-valid':'forward-inactive'}</span></p>`).join('')}<p class="muted">Candidate IDs: ${n.topk_ids.join(', ')}<br>Forward-valid IDs: ${n.active_ids.join(', ')||'none'}</p><p class="muted">Source: <code>${esc(n.source_path)}</code></p>`;}
 $('details').querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>selectNode(b.dataset.select));dirty=true;}
function selectNode(id){selected=byId.get(id)||null;showDetails();}
function filterSnapshot(selection){if(selection==null)return INITIAL;const allowed=new Set(selection.map(p=>'prototype:stage4:'+p));const ee=INITIAL.edges.filter(e=>e.edge_type==='prototype_patch'?allowed.has(e.source):allowed.has(e.source)&&allowed.has(e.target));const ids=new Set(allowed);for(const e of ee)if(e.edge_type==='prototype_patch')ids.add(e.target);return {...INITIAL,nodes:INITIAL.nodes.filter(n=>ids.has(n.id)),edges:ee,selected_prototypes:selection};}
async function fetchNeighborhood(){const selection=pendingAll?null:[...pendingSet].sort((a,b)=>a-b);const request=++requestId;$('picker').open=false;$('load').disabled=true;try{let result;if(online){$('status').textContent='Loading selected prototypes and their connected patches…';const r=await fetch('/api/graph?'+new URLSearchParams({prototypes:selection==null?'ALL':selection.join(',')}));result=await r.json();if(!r.ok)throw Error(result.error||'Could not load graph');}else result=filterSnapshot(selection);if(request===requestId)loadData(result);}catch(e){if(request===requestId)$('status').textContent=e.message;}finally{if(request===requestId)$('load').disabled=false;}}
let galleryReturnFocus=null;
function openGallery(prototypeIds=null){galleryReturnFocus=document.activeElement;const requested=prototypeIds==null?null:new Set(prototypeIds);const prototypes=nodes.filter(n=>n.node_type==='prototype'&&(!requested||requested.has(n.prototype_id)));const groups=new Map(prototypes.map(p=>[p.id,[]]));for(const e of edges)if(e.edge_type==='prototype_patch'&&groups.has(e.source))groups.get(e.source).push(e);let count=0;const seen=new Set(),cards=[];
 for(const p of prototypes){const group=groups.get(p.id).sort((a,b)=>a.rank-b.rank);count+=group.length;cards.push(`<h3 class="group-title">Prototype ${p.prototype_id} · ${group.length} connected patches</h3>`);for(const e of group){const n=e.b;seen.add(n.id);cards.push(`<button type="button" class="gallery-card" data-gallery-node="${esc(n.id)}" title="${esc(n.video+' / '+n.target_name)}"><span class="badge">P${p.prototype_id} · rank #${e.rank}</span>${n.crop_image?`<img loading="lazy" src="${n.crop_image}" alt="Patch ${n.patch_id}">`:'<span class="noimage">No thumbnail</span>'}<b>Patch ${n.patch_id}</b><small>${esc(n.video)} / ${esc(n.target_name)}</small><small>Cell ${n.row},${n.col} · saliency ${fmt(n.pred_percentile*100,1)}</small><small>Activation ${fmt(e.activation)}</small><small>Visual similarity ${fmt(e.visual_score)}</small><small>${e.forward_valid?'Forward-valid':'Forward-inactive'}</small></button>`);}}
 $('galleryTitle').textContent=prototypes.length===1?`View patches · prototype ${prototypes[0].prototype_id}`:'View patches · selected prototypes';$('gallerySummary').textContent=`${count} links across ${prototypes.length} prototypes · ${seen.size} unique patches. Grouped by prototype, then activation descending; ties use patch ID. A shared patch appears in each connected prototype group. No 50-patch cap.`;$('galleryGrid').style.setProperty('--gallery-cols',data.gallery_columns);$('galleryGrid').innerHTML=cards.join('');$('galleryGrid').querySelectorAll('[data-gallery-node]').forEach(b=>b.onclick=()=>{closeGallery();selectNode(b.dataset.galleryNode);});if(!$('patchGallery').open)$('patchGallery').showModal();}
function closeGallery(){$('patchGallery').close();if(galleryReturnFocus?.isConnected)galleryReturnFocus.focus();}
$('closeGallery').onclick=closeGallery;$('patchGallery').addEventListener('click',e=>{if(e.target!==$('patchGallery'))return;const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)closeGallery();});
$('load').onclick=fetchNeighborhood;$('viewPatches').onclick=()=>openGallery();$('fit').onclick=layout;$('edgeMode').onchange=()=>{dirty=true;};$('find').onclick=()=>{const q=$('query').value.trim().toLowerCase();if(!q)return;const n=nodes.find(n=>n.node_type==='patch'&&(String(n.patch_id)===q||(n.video+' '+n.target_name).toLowerCase().includes(q)));if(n){selectNode(n.id);zoom=Math.max(zoom,1.4);panX=-n.x*zoom;panY=-n.y*zoom;dirty=true;}else $('status').textContent='No displayed patch matches that search.';};$('query').onkeydown=e=>{if(e.key==='Enter')$('find').click();};
function point(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left-width/2-panX)/zoom,y:(e.clientY-r.top-height/2-panY)/zoom};}
function hit(p){let best=null,distance=Infinity;for(const n of nodes){const d=Math.hypot(n.x-p.x,n.y-p.y);if(d<Math.max(n.r+3,5/zoom)&&d<distance){best=n;distance=d;}}return best;}
function edgeHit(p){let best=null,closest=4/zoom;for(const e of edges){if(!visibleEdge(e))continue;const dx=e.b.x-e.a.x,dy=e.b.y-e.a.y,len=dx*dx+dy*dy;if(!len)continue;const t=Math.max(0,Math.min(1,((p.x-e.a.x)*dx+(p.y-e.a.y)*dy)/len));const d=Math.hypot(p.x-e.a.x-t*dx,p.y-e.a.y-t*dy);if(d<closest){closest=d;best=e;}}return best;}
function tooltip(e,event){const box=$('tooltip');if(!e){box.style.display='none';return;}box.textContent=`${e.a.label} ↔ ${e.b.label}\n${e.edge_type==='prototype_patch'?`Rank #${e.rank} · activation ${fmt(e.activation)}\n`:''}${edgeInfo(e)}`;box.style.display='block';box.style.left=Math.max(8,Math.min(innerWidth-390,event.clientX+12))+'px';box.style.top=Math.max(8,Math.min(innerHeight-200,event.clientY+12))+'px';}
canvas.onpointerdown=e=>{const p=point(e),n=hit(p);drag={n,x:e.clientX,y:e.clientY};if(n)selectNode(n.id);tooltip(null);canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{const p=point(e);hover=hit(p);canvas.style.cursor=hover?'pointer':'grab';if(drag){if(drag.n){const dx=p.x-drag.n.x,dy=p.y-drag.n.y;if(drag.n.node_type==='prototype'){for(const n of nodes)if(n!==drag.n&&n.node_type==='patch'){if(n.owner===drag.n.id){n.x+=dx;n.y+=dy;}}}drag.n.x=p.x;drag.n.y=p.y;}else{panX+=e.clientX-drag.x;panY+=e.clientY-drag.y;}drag.x=e.clientX;drag.y=e.clientY;tooltip(null);}else tooltip(hover?null:edgeHit(p),e);dirty=true;};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};canvas.onpointerleave=()=>{hover=null;tooltip(null);dirty=true;};canvas.onwheel=e=>{e.preventDefault();const p=point(e),r=canvas.getBoundingClientRect();zoom=Math.max(.02,Math.min(12,zoom*Math.exp(-e.deltaY*.001)));panX=e.clientX-r.left-width/2-p.x*zoom;panY=e.clientY-r.top-height/2-p.y*zoom;tooltip(null);dirty=true;};
function drawEdge(e,focused){ctx.strokeStyle=edgeColor(e.visual_score,focused?.95:e.edge_type==='prototype_prototype'?.25:.68);ctx.lineWidth=(focused?1.8:e.edge_type==='prototype_prototype'?.5:.85)/Math.sqrt(zoom);ctx.setLineDash(e.visual_score==null?[4/zoom,3/zoom]:[]);ctx.beginPath();ctx.moveTo(e.a.x,e.a.y);ctx.lineTo(e.b.x,e.b.y);ctx.stroke();}
function draw(){const dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect();if(width!==r.width||height!==r.height||canvas.width!==Math.round(r.width*dpr)||canvas.height!==Math.round(r.height*dpr)){width=r.width;height=r.height;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);dirty=true;}if(dirty){dirty=false;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);ctx.translate(width/2+panX,height/2+panY);ctx.scale(zoom,zoom);for(const c of clusters){ctx.fillStyle='#f1f5fa';ctx.strokeStyle='#dde7f2';ctx.lineWidth=1/zoom;ctx.beginPath();ctx.arc(c.p.x,c.p.y,c.r,0,Math.PI*2);ctx.fill();ctx.stroke();}const focused=[];for(const e of edges){if(!visibleEdge(e))continue;if(e.a===selected||e.b===selected)focused.push(e);else drawEdge(e,false);}for(const e of focused)drawEdge(e,true);ctx.setLineDash([]);
 for(const n of nodes){const proto=n.node_type==='prototype',r=n.r;ctx.beginPath();if(proto){ctx.moveTo(n.x,n.y-r);ctx.lineTo(n.x+r,n.y);ctx.lineTo(n.x,n.y+r);ctx.lineTo(n.x-r,n.y);ctx.closePath();}else ctx.arc(n.x,n.y,r,0,Math.PI*2);ctx.fillStyle=proto?'#244f85':salColor(n.pred_percentile);ctx.fill();ctx.strokeStyle=n===selected?'#ef6638':n===hover?'#172236':'#fff';ctx.lineWidth=(n===selected?3:1)/Math.sqrt(zoom);ctx.stroke();if(proto||n===selected||n===hover){ctx.font=`${Math.max(11,Math.min(45,9/zoom))}px system-ui`;ctx.fillStyle='#172236';ctx.fillText(proto?'P'+n.prototype_id:String(n.patch_id),n.x+r+5,n.y-5);}}}requestAnimationFrame(draw);}
renderPicker();loadData(INITIAL);draw();
</script></body></html>'''
def _selftest_fixture(directory):
    """Small complete graph with ties and a prototype absent from all top-K lists."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    con = connect_db(directory/'graph.sqlite')
    create_schema(con)
    rng = np.random.default_rng(3)
    count,m = 18,4
    activations = rng.integers(0,5,(count,m)).astype(np.float32)
    activations[:,3] = np.arange(count)%4
    activations[17,3] = 12.
    descriptors = {}
    for frame in range(3):
        image = Image.fromarray(rng.integers(0,256,(80,120,3),dtype=np.uint8))
        thumb = f'frame_{frame}.jpg'
        image.save(directory/thumb)
        con.execute('INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            (frame,'test',f'video{frame}',frame,0,15,f'{frame}.jpg','source.jpg',thumb,'[]',80,120))
        for cell in range(6):
            nid = frame*6+cell
            top,active = [0,1],[0]
            con.execute('INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (nid,frame,'stage4',cell//3,cell%3,2,3,cell/5,cell/5,None,None,
                 json.dumps(top),json.dumps([1.,.5]),json.dumps(active),json.dumps([.7])))
            con.executemany('INSERT INTO memberships VALUES (?,?,?,?)',
                [(nid,'stage4','topk',p) for p in top]+[(nid,'stage4','active',0)])
        con.execute('INSERT INTO prototype_activation_frames VALUES (?,?,?,?,?,?)',
            (frame,'stage4',6,m,zlib.compress(np.arange(frame*6,(frame+1)*6,dtype='<i8').tobytes()),
             pack_vector(activations[frame*6:(frame+1)*6])))
    con.execute("INSERT INTO prototype_banks VALUES ('stage4',?)",(m,))
    for nid in range(count):
        feature,rgb = rng.random(8),rng.random(RGB_DESCRIPTOR_SIZE**2*3)
        store_descriptor(con,nid,feature,rgb)
        descriptors[nid] = feature.astype(np.float32).astype(float),rgb.astype(np.float32).astype(float)
    con.commit()
    dump_json(directory/'manifest.json',{'status':'complete','top_k':TOP_K,'seed':SEED})
    return con,activations,descriptors


def selftest():
    assert np.allclose(midrank_percentiles([0,0,1,2]),[1/6,1/6,2/3,1])
    feature = np.array([1.,2.,3.])
    rgb = np.random.default_rng(42).random(RGB_DESCRIPTOR_SIZE**2*3)
    assert np.isclose(visual_metrics((feature,rgb),(feature,rgb))['visual_score'],3.)
    assert visual_metrics(None,(feature,rgb))['visual_score'] is None
    assert visual_metrics((np.zeros(3),rgb),(feature,rgb))['visual_score'] is None
    opposite = visual_metrics((feature,rgb),(-feature,1-rgb))
    assert np.isclose(opposite['stage4_cosine'],-1.) and np.isclose(opposite['rgb_structure_cosine'],-1.)
    # Population SD in percentile points; unordered pairs exclude self-pairs.
    ps = [{'id':i,'pred_percentile':p} for i,p in enumerate([.2,.5,.8])]
    desc = {i:(feature,rgb) for i in range(3)}
    _,s = summarize_prototype(ps,desc)
    assert np.isclose(s['mean_saliency_percentile'],50.)
    assert np.isclose(s['std_saliency_percentile'],np.std([20.,50.,80.],ddof=0))
    assert s['total_pair_count']==s['valid_pair_count']==3 and np.isclose(s['mean_pairwise_visual_similarity'],3.)
    _,one = summarize_prototype(ps[:1],desc)
    assert one['std_saliency_percentile']==0 and one['mean_pairwise_visual_similarity'] is None
    _,partial = summarize_prototype(ps,{0:desc[0],1:desc[1]})
    assert partial['valid_pair_count']==1 and partial['total_pair_count']==3
    assert summarize_prototype(ps,{0:desc[0]})[0] is None
    _,empty = summarize_prototype([],{})
    assert empty['mean_saliency_percentile'] is None and empty['std_saliency_percentile'] is None
    # Streaming ranking oracle: ties must survive arbitrary block boundaries.
    rng = np.random.default_rng(4)
    a = rng.integers(-2,4,(71,9)).astype(np.float32)
    ids = rng.permutation(71).astype(np.int64)
    for cap in (1,3,50,100):
        old_ids = old_scores = None
        for offset in range(0,len(a),7):
            old_ids,old_scores = merge_top_matches(old_ids,old_scores,ids[offset:offset+7],a[offset:offset+7],cap)
        for p in range(a.shape[1]):
            expected = sorted(zip(ids.tolist(),a[:,p].tolist()),key=lambda x:(-x[1],x[0]))[:cap]
            assert old_ids[:,p].tolist()==[v[0] for v in expected]
            assert old_scores[:,p].tolist()==[v[1] for v in expected]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        con,a,desc = _selftest_fixture(d)
        original = [tuple(r) for r in con.execute('SELECT * FROM nodes ORDER BY id')]
        rebuild_rank_cache(con,5)
        for p in range(4):
            rows = list(con.execute("SELECT node_id,activation FROM prototype_patches WHERE stage='stage4' AND prototype_id=? ORDER BY rank",(p,)))
            expected = sorted(enumerate(a[:,p]),key=lambda x:(-float(x[1]),x[0]))[:5]
            assert [(int(r[0]),float(r[1])) for r in rows]==[(i,float(v)) for i,v in expected]
        assert con.execute("SELECT node_id FROM prototype_patches WHERE prototype_id=3 AND rank=1").fetchone()[0]==17
        assert con.execute("SELECT COUNT(*) FROM memberships WHERE prototype_id=3").fetchone()[0]==0
        graph = build_graph(con,d,n=3)
        protos = [n for n in graph['nodes'] if n['node_type']=='prototype']
        links = [e for e in graph['edges'] if e['edge_type']=='prototype_patch']
        pairs = [e for e in graph['edges'] if e['edge_type']=='prototype_prototype']
        assert len(protos)==4 and len(links)==12 and len(pairs)==6
        assert len(graph['nodes'])==len({n['id'] for n in graph['nodes']})
        assert all(e['visual_score'] is not None for e in graph['edges'])
        for p in protos:
            matched = [e for e in links if e['source']==p['id']]
            nodes_by_id = {n['id']:n for n in graph['nodes']}
            percentiles = [nodes_by_id[e['target']]['pred_percentile']*100 for e in matched]
            assert np.isclose(p['stats']['mean_saliency_percentile'],np.mean(percentiles))
            assert np.isclose(p['stats']['std_saliency_percentile'],np.std(percentiles,ddof=0))
            expected_scores = [visual_metrics(desc[nodes_by_id[e['target']]['patch_id']],desc[nodes_by_id[f['target']]['patch_id']])['visual_score']
                for i,e in enumerate(matched) for f in matched[i+1:]]
            assert np.isclose(p['stats']['mean_pairwise_visual_similarity'],np.mean(expected_scores))
        filtered = select_prototypes(graph,[1,3])
        assert len([n for n in filtered['nodes'] if n['node_type']=='prototype'])==2
        assert len([e for e in filtered['edges'] if e['edge_type']=='prototype_patch'])==6
        assert len([e for e in filtered['edges'] if e['edge_type']=='prototype_prototype'])==1
        ids_filtered = {n['id'] for n in filtered['nodes']}
        assert all(e['source'] in ids_filtered and e['target'] in ids_filtered for e in filtered['edges'])
        assert not select_prototypes(graph,[])['nodes']
        assert select_prototypes(graph,None) is graph
        try:
            select_prototypes(graph,[999])
            raise AssertionError('Unknown prototype accepted')
        except ValueError:
            pass
        assert len(graph['image_pool'])<=3
        assert all('frame_image' not in n for n in graph['nodes'])
        write_viewer(d,con,graph=graph)
        html = (d/'graph_viewer.html').read_text()
        assert '__PAYLOAD__' not in html and '<canvas' in html and 'Top 50' not in html
        assert 'mean_saliency_percentile' in html and 'std_saliency_percentile' in html
        assert 'mean_pairwise_visual_similarity' in html and 'Load neighbourhood' in html
        json.dumps(filtered,allow_nan=False)
        # Increasing N consumes cached activations, never checkpoint inference.
        rebuild_rank_cache(con,10)
        assert len([e for e in build_graph(con,d,n=7)['edges'] if e['edge_type']=='prototype_patch'])==28
        assert [tuple(r) for r in con.execute('SELECT * FROM nodes ORDER BY id')]==original
        # Incomplete old database must request backfill, never rank top-K only.
        con.execute('DELETE FROM prototype_activation_frames WHERE frame_id=1')
        try:
            require_dense_cache(con)
            raise AssertionError('Incomplete activation cache accepted')
        except RuntimeError as exc:
            assert 'cache_similarity' in str(exc)
        con.close()
    print('PASS: exact streaming top-N vs brute force; deterministic ties; all prototypes; shared patches; '
          'two edge types; pairwise similarity; mean ± population SD; filtering; offline HTML; incomplete-cache detection; preserved original nodes.')


def main():
    actions = {'extract':extract_graph,'cache_similarity':lambda:cache_similarity(OUTPUT_DIR),
               'view':lambda:(require_complete(OUTPUT_DIR),write_viewer(OUTPUT_DIR)),
               'serve':lambda:serve_graph(OUTPUT_DIR),'export_edges':lambda:export_edges(OUTPUT_DIR),
               'selftest':selftest}
    if ACTION not in actions:
        raise ValueError(f'Unknown ACTION={ACTION!r}; choose from {list(actions)}')
    actions[ACTION]()


if __name__=='__main__':
    main()
