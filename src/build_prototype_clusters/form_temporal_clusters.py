#!/usr/bin/env python3
"""Build target-aligned, camera-relative, FLOW-ONLY temporal patch references.

Run from src/ (or standalone):
  python patch_clusters/form_temporal_clusters.py --output-dir /path/to/new_bank
  python patch_clusters/form_temporal_clusters.py --flow-gaps 4 6 8
  python patch_clusters/form_temporal_clusters.py --flow-estimator dis-fast
  python patch_clusters/form_temporal_clusters.py --clustering finch --finch-level 0
  python form_temporal_clusters.py --from-window-cache /path/to/index.sqlite --output-dir /path/to/new_bank --clustering finch --finch-level 0 --gpu 0
  python patch_clusters/form_temporal_clusters.py --flow-estimator raft-small --raft-weights /local/raft.pth

Defaults: CPU Farneback, six-frame gap, 32x32 spatial cells, soft circular flow
histograms, clipped log magnitudes, separate static/unknown references, and
average-linkage discovery on a video-capped sample of MOVING cells only.
Optional multiple gaps all END AT THE SAME TARGET FRAME; they are converted to
pixels/frame before averaging. Their variation is cross-gap rate variation,
not an acceleration estimate. Backward/forward consistency uses displacement.

No Video Swin, image embeddings, feature cache, checkpoint inspection, dataset
loader, saliency maps, or training model is imported or executed. Only selected
RGB frame pairs are decoded. Torch/torchvision are imported ONLY for RAFT.
RAFT internally computes its own optical-flow features, as required by RAFT.

Input: dataset_root/train/<video>/images/*.{jpg,jpeg,png,bmp,tif,tiff,webp}.
Frames are naturally sorted. Windows contain contiguous frames, start every
--stride frames, and use their last frame as target. Incomplete windows are
excluded. Changing this convention changes which windows enter the bank.

Output v2 deliberately records changed semantics. Legacy SQL change_* fields
are zero/invalid compatibility placeholders; change_direction.npy is [N,0].
Existing v1 banks/caches are never silently reused. Consumers must inspect the
v2 manifest, apply its scales/distance/role routing, and use flow descriptors;
v1 feature-change consumers need updating. Spatial cells match stage4 geometry
only when --patch-size matches the backbone's spatial stride (default 32 32).

Dependencies: numpy scipy opencv-python[-headless] (tqdm optional).
FINCH optionally uses the project's existing clustering_common.py; other
clustering methods are self-contained. No automatic model download occurs.

--from-window-cache clusters unnormalized patch descriptors already saved by
cache_optical_flow.py. Only indexed train/original windows are used; old labels
and validation/test data are ignored. No RGB decoding or flow estimation occurs
during clustering. Descriptor extraction settings are inherited from the cache;
only clustering/distance/normalization settings may change. All matching cached
starts are used by default; --stride can subsample them. Partial caches require
--allow-partial-cache. Source caches are read-only and never modified.

After clustering, --viewer-examples 12 (default) prepares the medoid plus the
11 nearest distinct patches of each cluster. Dense bidirectional flow is reused
or calculated ONLY for their frame pairs, with recorded full-frame camera fits;
histograms are recalculated only for selected patches and checked against cached
descriptors. Shared pairs are saved once under the bank's viewer_flow/ directory.
viewer_examples/ contains small per-cluster summaries, and the viewer_examples
SQL table records the exact gallery. Use the updated view_temporal_clusters.py.
--viewer-examples 0 skips this phase. --prepare-viewer-only retries it on an
existing clustered bank without reclustering. GPU/RAFT weights are needed only
if the selected examples require fresh estimates. CUDA Farneback follows cache
provenance, requires CUDA OpenCV, and never falls back to CPU.
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
import re
import sqlite3
import sys
import tempfile

import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist, squareform

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


FORMAT_VERSION = "fixed_stage4_temporal_clusters_v2_flow_only"
FLOW_CACHE_SCHEMA = "bidirectional_selected_rgb_flow_npz_v2"
DEFAULT_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROLES = ("invalid", "static", "moving")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest_json(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def target_flow_validity(forward, backward, absolute=1.0, relative=0.05):
    """Forward/backward consistency at CURRENT-frame pixels."""
    import cv2

    h, w = backward.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    px, py = x + backward[..., 0], y + backward[..., 1]
    sampled = cv2.remap(
        forward, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    error = np.linalg.norm(backward + sampled, axis=-1)
    tolerance = absolute + relative * (
        np.linalg.norm(backward, axis=-1) + np.linalg.norm(sampled, axis=-1)
    )
    return (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1) & (error <= tolerance)


def compensate_camera(motion, valid, mode="median", seed=42, threshold=1.5):
    """Fit global apparent motion on the target pixel lattice; retain it separately."""
    import cv2

    h, w = motion.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    model = {"requested": mode, "used": mode, "valid_fraction": float(valid.mean())}
    if mode == "none":
        global_flow = np.zeros_like(motion)
    else:
        if not valid.any():
            global_flow = np.zeros_like(motion)
            model.update(used="no_valid_flow", vector=[0.0, 0.0])
        else:
            median = np.median(motion[valid], axis=0)
            global_flow = np.broadcast_to(median, motion.shape).copy()
            model["vector"] = median.astype(float).tolist()
            if mode == "affine":
                positions = np.stack((x, y), axis=-1)[valid]
                destinations = positions + motion[valid]
                select = np.linspace(
                    0, len(positions) - 1, min(4096, len(positions)), dtype=np.int64
                )
                cv2.setRNGSeed(int(seed) % (2**31 - 1))
                affine, inliers = (
                    cv2.estimateAffine2D(
                        positions[select],
                        destinations[select],
                        method=cv2.RANSAC,
                        ransacReprojThreshold=threshold,
                        maxIters=2000,
                        confidence=0.99,
                        refineIters=10,
                    )
                    if len(select) >= 6
                    else (None, None)
                )
                if (
                    affine is not None
                    and np.isfinite(affine).all()
                    and int(inliers.sum()) >= 6
                ):
                    grid = np.stack((x, y, np.ones_like(x)), axis=-1)
                    global_flow = (grid @ affine.T - np.stack((x, y), axis=-1)).astype(
                        np.float32
                    )
                    model.update(
                        affine=affine.tolist(), inlier_fraction=float(inliers.mean())
                    )
                else:
                    model["used"] = "median_fallback"
    return motion - global_flow, global_flow, model


def create_schema(con):
    con.executescript("""
    PRAGMA foreign_keys=ON;
    CREATE TABLE frames (id INTEGER PRIMARY KEY,dataset TEXT NOT NULL,video TEXT NOT NULL,
       dataset_index INTEGER NOT NULL,window_start INTEGER NOT NULL,target_index INTEGER NOT NULL,
       target_name TEXT NOT NULL,previous_index INTEGER NOT NULL,previous_name TEXT NOT NULL,
       previous_slice INTEGER NOT NULL,current_slice INTEGER NOT NULL,previous_offset INTEGER NOT NULL,
       current_offset INTEGER NOT NULL,input_h INTEGER NOT NULL,input_w INTEGER NOT NULL,
       source_path TEXT NOT NULL,previous_source_path TEXT NOT NULL,pair_sha256 TEXT NOT NULL,global_model_json TEXT NOT NULL);
    CREATE TABLE patches (descriptor_index INTEGER PRIMARY KEY,node_id INTEGER NOT NULL UNIQUE,
       frame_id INTEGER NOT NULL REFERENCES frames(id),stage TEXT NOT NULL,row INTEGER NOT NULL,col INTEGER NOT NULL,
       grid_h INTEGER NOT NULL,grid_w INTEGER NOT NULL,flow_magnitude REAL NOT NULL,change_magnitude REAL NOT NULL,
       flow_valid_fraction REAL NOT NULL,flow_valid INTEGER NOT NULL,change_valid INTEGER NOT NULL,
       change_static INTEGER NOT NULL,prototype_id INTEGER,distance REAL);
    CREATE TABLE prototypes (id INTEGER PRIMARY KEY,medoid_index INTEGER NOT NULL,medoid_node_id INTEGER NOT NULL,
       patch_count INTEGER NOT NULL,video_count INTEGER NOT NULL,mean_distance REAL,
       mean_flow_magnitude REAL,mean_change_magnitude REAL);
    CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE INDEX patches_frame ON patches(frame_id);
    """)


def prepare_directory(path, overwrite):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} is not empty; use a new output directory, or --overwrite to preserve it in a timestamped backup."
            )
        backup = path.with_name(
            path.name
            + ".backup-"
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        )
        path.rename(backup)
        print(f"Previous bank preserved at {backup}", flush=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_output_location(args):
    output = Path(args.output_dir).expanduser().resolve()
    dataset = Path(args.dataset_root).expanduser().resolve()
    training = dataset / "train"
    if (
        output == dataset
        or output in dataset.parents
        or output == training
        or training in output.parents
    ):
                raise ValueError(
            "The output bank must be outside the training-image tree and must not replace the dataset root."
        )
    for root in (args.feature_cache_root, args.flow_cache_root):
        root = Path(root).expanduser().resolve()
        if output == root or output in root.parents or root in output.parents:
            raise ValueError("The output bank and cache directories must not overlap.")


class FlowEstimator:
    def __init__(self, args):
        import cv2

        self.cv2 = cv2
        self.kind = args.flow_estimator
        self.model = None
        self.dis = None
        self.updates = args.raft_updates
        self.config = {
            "estimator": self.kind,
            "opencv_version": cv2.__version__,
            "canvas": list(args.resize_to),
            "bidirectional": True,
            "input": "RGB uint8; cv2 INTER_LINEAR canvas resize before grayscale",
            "flow_units": "canvas pixels over recorded frame interval",
        }
        if self.kind == "farneback":
            self.config["parameters"] = dict(
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        elif self.kind.startswith("dis-"):
            preset = (
                cv2.DISOPTICAL_FLOW_PRESET_FAST
                if self.kind == "dis-fast"
                else cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
            )
            self.dis = cv2.DISOpticalFlow_create(preset)
            self.config["preset"] = int(preset)
        else:
            import torch
            import torchvision
            from torchvision.models.optical_flow import raft_large, raft_small

            if not args.raft_weights or not args.raft_weights.is_file():
                raise FileNotFoundError(
                    "RAFT requires --raft-weights with a local torchvision state_dict."
                )
            if min(args.resize_to) < 128:
                raise ValueError("RAFT requires a canvas of at least 128x128.")
            self.torch = torch
            self.device = (
                args.flow_device
                or args.device
                or (f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
            )
            factory = raft_small if self.kind == "raft-small" else raft_large
            self.model = factory(weights=None, progress=False).to(self.device).eval()
            state = torch.load(args.raft_weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=True)
            self.config.update(
                torch_version=str(torch.__version__),
                torchvision_version=torchvision.__version__,
                iterations=self.updates,
                checkpoint_sha256=sha256_file(args.raft_weights),
                normalization="uint8 RGB -> [-1,1]; replicate pad to multiple of 8",
            )

    def estimate(self, previous, current):
        cv2 = self.cv2
        if self.model is None:
            from concurrent.futures import ThreadPoolExecutor

            a = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
            # OpenCV releases the GIL inside Farneback/DIS, so both directions
            # can run concurrently on CPU. CUDA OpenCV Farneback is unavailable
            # in typical wheels; RAFT (below) is the GPU estimator path.
            if self.dis is not None:
                # No initial flow: each pair and direction is estimated independently.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_f = pool.submit(self.dis.calc, a, b, None)
                    fut_b = pool.submit(self.dis.calc, b, a, None)
                    f, bwd = fut_f.result(), fut_b.result()
            else:
                params = self.config["parameters"]

                def _farneback(src, dst):
                    return cv2.calcOpticalFlowFarneback(src, dst, None, **params)

                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_f = pool.submit(_farneback, a, b)
                    fut_b = pool.submit(_farneback, b, a)
                    f, bwd = fut_f.result(), fut_b.result()
            return f.astype(np.float32), bwd.astype(np.float32)
        torch = self.torch
        import torch.nn.functional as F

        with torch.inference_mode():

            def tensor(rgb):
                return (
                    torch.from_numpy(rgb.copy())
                    .permute(2, 0, 1)[None]
                    .float()
                    .to(self.device)
                    / 127.5
                    - 1
                )

            a, b = tensor(previous), tensor(current)
            h, w = a.shape[-2:]
            pad = (0, (-w) % 8, 0, (-h) % 8)
            a, b = F.pad(a, pad, mode="replicate"), F.pad(b, pad, mode="replicate")

            def run(x, y):
                return (
                    self.model(x, y, num_flow_updates=self.updates)[-1][0, :, :h, :w]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            return run(a, b), run(b, a)


class FlowCache:
    def __init__(self, args, estimator=None):
        self.estimator = estimator or FlowEstimator(args)
        self.config = {
            "schema": FLOW_CACHE_SCHEMA,
            "estimator": self.estimator.config,
            "storage_dtype": args.flow_storage_dtype,
        }
        self.namespace = digest_json(self.config)
        self.directory = (
            args.flow_cache_root.expanduser().resolve()
            / FLOW_CACHE_SCHEMA
            / self.namespace
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.config:
                raise ValueError(f"Flow cache namespace metadata mismatch: {manifest}")
        else:
            write_json(manifest, self.config)
        self.dtype = np.dtype(args.flow_storage_dtype)
        self.allow_compute = not args.cache_read_only
        self.stats = dict(hits=0, misses=0, writes=0, estimator_calls=0)

    def _load(self, path, key, shape):
        with np.load(path, allow_pickle=False) as payload:
            if (
                str(payload["namespace"]) != self.namespace
                or str(payload["pair_sha256"]) != key
            ):
                raise ValueError(f"Invalid flow cache identity (preserved): {path}")
            values = []
            for name in ("forward", "backward"):
                value = payload[name]
                if (
                    value.shape != (*shape[:2], 2)
                    or value.dtype != self.dtype
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid flow cache tensor (preserved): {path}")
                values.append(value.astype(np.float32))
            return tuple(values)

    def get(self, previous, current):
        if (
            previous.shape != current.shape
            or previous.dtype != np.uint8
            or current.dtype != np.uint8
        ):
            raise ValueError("Flow inputs must be equal-shape uint8 RGB canvases.")
        h = hashlib.sha256(str(previous.shape).encode())
        h.update(np.ascontiguousarray(previous).tobytes())
        h.update(np.ascontiguousarray(current).tobytes())
        key = h.hexdigest()
        path = self.directory / "entries" / key[:2] / f"{key}.npz"
        if path.exists():
            self.stats["hits"] += 1
            return *self._load(path, key, previous.shape), key
        self.stats["misses"] += 1
        if not self.allow_compute:
            raise FileNotFoundError(f"Flow cache miss in read-only mode: {path}")
        self.stats["estimator_calls"] += 1
        f, b = self.estimator.estimate(previous, current)
        f, b = f.astype(self.dtype), b.astype(self.dtype)
        if any(
            x.shape != (*previous.shape[:2], 2) or not np.isfinite(x).all()
            for x in (f, b)
        ):
            raise ValueError(
                "Nonfinite or wrong-shape estimated flow; cache entry not written."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as stream:
            temp = Path(stream.name)
            np.savez_compressed(
                stream,
                forward=f,
                backward=b,
                namespace=np.array(self.namespace),
                pair_sha256=np.array(key),
            )
        try:
            try:
                os.link(temp, path)  # Create-only, including competing writers.
                self.stats["writes"] += 1
            except FileExistsError:
                pass
        finally:
            temp.unlink(missing_ok=True)
        return *self._load(path, key, previous.shape), key


def natural_key(value):
    return tuple(
        (1, int(p)) if p.isdigit() else (0, p.casefold())
        for p in re.split(r"(\d+)", str(value))
    )


def discover_windows(args):
    split = args.dataset_root.expanduser().resolve() / "train"
    if not split.is_dir():
        raise FileNotFoundError(f"Training split missing: {split}")
    windows = []
    frames_by_video = {}
    for video in sorted(
        (x for x in split.iterdir() if x.is_dir()), key=lambda x: natural_key(x.name)
    ):
        images = video / "images"
        if not images.is_dir():
            continue
        frames = sorted(
            (
                p
                for p in images.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda p: natural_key(p.name),
        )
        if len(frames) < args.window_len:
            continue
        frames_by_video[video.name] = frames
        for start in range(0, len(frames) - args.window_len + 1, args.stride):
            windows.append((video.name, start))
            if args.max_windows and len(windows) >= args.max_windows:
                return windows, frames_by_video
    if not windows:
        raise ValueError("No complete training RGB windows found.")
    return windows, frames_by_video


def read_canvas(path, hw):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot decode RGB frame: {path}")
    image = cv2.resize(image, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def soft_histogram(vectors, threshold):
    """Eight circular direction bins + static mass; valid vectors only."""
    out = np.zeros(9, np.float32)
    magnitude = np.linalg.norm(vectors, axis=-1)
    moving = magnitude > threshold
    out[-1] = 1.0 - moving.mean()
    if moving.any():
        angle = np.mod(np.arctan2(vectors[moving, 1], vectors[moving, 0]), 2 * np.pi)
        bins = angle * 8 / (2 * np.pi)
        lower = np.floor(bins).astype(np.int64)
        fraction = bins - lower
        # Log weights prevent a few fast/outlier pixels dominating direction.
        weight = np.log1p(magnitude[moving])
        counts = np.bincount(lower % 8, weights=weight * (1 - fraction), minlength=8)
        counts += np.bincount((lower + 1) % 8, weights=weight * fraction, minlength=8)
        out[:8] = moving.mean() * counts / counts.sum()
    return out


def summarize_flow(backward, valid, gh, gw, args, frame_gap):
    # Camera fit/FB thresholds are in displacement pixels. Convert ONLY after fit.
    residual, global_flow, camera = compensate_camera(
        -backward, valid, args.camera_motion, args.seed, args.camera_ransac_threshold
    )
    residual = residual / frame_gap
    global_flow = global_flow / frame_gap
    ph, pw = args.patch_size
    n = gh * gw
    hist = np.zeros((n, 9), np.float32)
    mag = np.zeros(n, np.float32)
    mean_xy = np.zeros((n, 2), np.float32)
    spread = np.zeros(n, np.float32)
    global_xy = np.zeros((n, 2), np.float32)
    fraction = np.zeros(n, np.float32)
    for row in range(gh):
        for col in range(gw):
            i = row * gw + col
            sl = np.s_[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw]
            mask = valid[sl]
            fraction[i] = mask.mean()
            if not mask.any():
                continue
            vectors = residual[sl][mask]
            hist[i] = soft_histogram(vectors, args.flow_static_threshold)
            mag[i] = np.linalg.norm(vectors, axis=-1).mean()
            mean_xy[i] = vectors.mean(axis=0)
            spread[i] = np.sqrt(np.square(vectors - mean_xy[i]).sum(axis=-1).mean())
            global_xy[i] = global_flow[sl][mask].mean(axis=0)
    return (
        dict(
            flow_hist=hist,
            flow_magnitude=mag,
            flow_mean=mean_xy,
            flow_spread=spread,
            global_motion_raw=global_xy,
            flow_valid_fraction=fraction,
        ),
        camera,
    )


class TemporalDescriptors:
    SPECS = {
        "flow_hist": (9, np.float32),
        "flow_magnitude": (0, np.float32),
        "flow_mean": (2, np.float32),
        "flow_spread": (0, np.float32),
        "flow_variance": (0, np.float32),
        "global_motion_raw": (2, np.float32),
        "flow_valid_fraction": (0, np.float32),
        "flow_valid": (0, np.bool_),
        "role": (0, np.int8),
        "node_ids": (0, np.int64),
        "magnitude": (2, np.float32),
        "compact": (3, np.float32),
        "global_motion": (2, np.float32),
        # Explicitly neutral legacy fields; not used in flow-only distances.
        "change_direction": (None, np.float32),
        "change_magnitude": (0, np.float32),
        "change_valid": (0, np.bool_),
        "change_static": (0, np.bool_),
    }

    def __init__(
        self, directory, count=None, feature_dim=0, mode="r", metric="circular-emd"
    ):
        self.directory = Path(directory)
        self.metric = metric
        self.directory.mkdir(parents=True, exist_ok=True)
        for name, (dim, dtype) in self.SPECS.items():
            # npy headers allow [N,0] arrays without storing feature vectors.
            size = feature_dim if dim is None else dim
            shape = (count,) if dim == 0 else (count, size)
            value = (
                np.lib.format.open_memmap(
                    self.directory / f"{name}.npy", mode=mode, shape=shape, dtype=dtype
                )
                if mode == "w+"
                else np.load(
                    self.directory / f"{name}.npy", mmap_mode=mode, allow_pickle=False
                )
            )
            setattr(self, name, value)
        self.count = len(self.node_ids)

    def take(self, indices):
        return tuple(
            np.asarray(getattr(self, name)[indices])
            for name in (
                "flow_hist",
                "magnitude",
                "compact",
                "global_motion",
                "flow_valid",
            )
        )

    def distance(self, a, b, weights):
        return temporal_distance(a, b, weights, self.metric)

    def flush(self):
        for name in self.SPECS:
            getattr(self, name).flush()


def circular_hist_distance(a, b):
    """Exact circular EMD for conditional equal-mass direction histograms.

    Eight-bin circle, unit neighbor cost. Optimal circulation subtracts median
    cumulative imbalance; divide by four (opposite directions). Static fraction
    is a separate mass, with unit static/moving transition cost.
    """
    ma, mb = a[:, :8].sum(axis=1), b[:, :8].sum(axis=1)
    pa = np.divide(
        a[:, :8], ma[:, None], out=np.zeros_like(a[:, :8]), where=ma[:, None] > 1e-12
    )
    pb = np.divide(
        b[:, :8], mb[:, None], out=np.zeros_like(b[:, :8]), where=mb[:, None] > 1e-12
    )
    result = np.empty((len(a), len(b)), dtype=np.float64)
    # Chunk to avoid [4096,4096,8] temporaries during sample fitting.
    for start in range(0, len(a), 64):
        c = np.cumsum(
            pa[start : start + 64, None, :].astype(np.float64) - pb[None, :, :], axis=-1
        )
        emd = np.abs(c - np.median(c, axis=-1, keepdims=True)).sum(axis=-1) / 4
        result[start : start + 64] = np.minimum(
            ma[start : start + 64, None], mb[None, :]
        ) * emd + np.abs(ma[start : start + 64, None] - mb[None, :])
    return result


def temporal_distance(a, b, weights, metric="circular-emd"):
    ha, ma, ca, ga, va = a
    hb, mb, cb, gb, vb = b
    # weights retain the legacy four positions: flow, removed direction, mag, global.
    if metric == "circular-emd":
        flow = circular_hist_distance(ha, hb)
    elif metric == "l1":
        flow = cdist(ha, hb, metric="cityblock") / 2
    else:
        flow = cdist(ca, cb, metric="euclidean") / np.sqrt(3)
    both = va[:, None] & vb[None, :]
    flow[~va[:, None] & ~vb[None, :]] = 0
    flow[va[:, None] != vb[None, :]] = 1
    # Column 0 is ONLY residual-flow magnitude. Column 1 is rate dispersion across
    # requested gaps. No feature-change magnitude enters either column.
    magnitude = np.abs(ma[:, None, 0] - mb[None, :, 0])
    variance = np.abs(ma[:, None, 1] - mb[None, :, 1])
    global_motion = cdist(ga, gb, metric="cityblock") / 2
    return np.maximum(
        0,
        weights[0] * flow
        + both
        * (weights[2] * magnitude + weights[1] * variance + weights[3] * global_motion),
    )


def extract_descriptors(args, directory, con, feature_cache=None, flow_cache=None):
    if feature_cache is not None:
        raise ValueError(
            "This script is flow-only; a feature cache must never be supplied."
        )
    windows, by_video = discover_windows(args)
    gh, gw = (h // p for h, p in zip(args.resize_to, args.patch_size))
    per_frame = gh * gw
    data = TemporalDescriptors(
        directory / "descriptors",
        len(windows) * per_frame,
        mode="w+",
        metric=args.flow_metric,
    )
    # Explicit placeholders, never backed by a model or image features.
    data.change_magnitude[:] = 0
    data.change_valid[:] = False
    data.change_static[:] = True
    fingerprint = hashlib.sha256(digest_json(descriptor_config(args)).encode())
    for window, (video, start) in enumerate(
        tqdm(windows, desc="Flow-only windows", unit="window")
    ):
        target_index = start + args.window_len - 1
        paths = by_video[video]
        current = read_canvas(paths[target_index], args.resize_to)
        summaries, pair_keys, cameras = [], [], []
        for gap in args.flow_gaps:
            previous = read_canvas(paths[target_index - gap], args.resize_to)
            forward, backward, pair_key = flow_cache.get(previous, current)
            valid = target_flow_validity(
                forward, backward, args.flow_fb_absolute, args.flow_fb_relative
            )
            summary, camera = summarize_flow(backward, valid, gh, gw, args, gap)
            summaries.append(summary)
            pair_keys.append(pair_key)
            cameras.append(dict(frame_gap=gap, **camera))
        # Invalid gap estimates contribute no evidence. Average only usable gaps;
        # require a configured fraction of requested gaps for a usable descriptor.
        usable = np.stack(
            [
                s["flow_valid_fraction"] >= args.min_flow_valid_fraction
                for s in summaries
            ]
        )
        count = usable.sum(axis=0)
        good = count >= math.ceil(len(summaries) * args.min_valid_gap_fraction)
        merged = {}
        for name in summaries[0]:
            values = np.stack([s[name] for s in summaries])
            if name == "flow_valid_fraction":
                merged[name] = values.mean(axis=0)
            else:
                mask = usable[..., None] if values.ndim == 3 else usable
                denominator = np.maximum(count, 1)
                if values.ndim == 3:
                    denominator = denominator[:, None]
                merged[name] = (values * mask).sum(axis=0) / denominator
        means = np.stack([s["flow_mean"] for s in summaries])
        variance = (np.square(means - merged["flow_mean"]).sum(axis=-1) * usable).sum(
            axis=0
        ) / np.maximum(count, 1)
        merged.update(flow_variance=variance, flow_valid=good)
        roles = np.where(
            ~good,
            0,
            np.where(merged["flow_hist"][:, -1] >= args.static_patch_fraction, 1, 2),
        ).astype(np.int8)
        sl = slice(window * per_frame, (window + 1) * per_frame)
        for name, value in merged.items():
            getattr(data, name)[sl] = value
        data.role[sl] = roles
        ids = np.arange(sl.start, sl.stop, dtype=np.int64)
        data.node_ids[sl] = ids
        # Legacy frame columns mark absence of backbone slices with -1.
        previous_index = target_index - max(args.flow_gaps)
        pair_id = pair_keys[0] if len(pair_keys) == 1 else digest_json(pair_keys)
        con.execute(
            "INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                window,
                "train",
                video,
                window,
                start,
                target_index,
                paths[target_index].name,
                previous_index,
                paths[previous_index].name,
                -1,
                -1,
                args.window_len - 1 - max(args.flow_gaps),
                args.window_len - 1,
                *args.resize_to,
                str(paths[target_index].resolve()),
                str(paths[previous_index].resolve()),
                pair_id,
                json.dumps({"pairs": cameras, "pair_keys": pair_keys}, sort_keys=True),
            ),
        )
        con.executemany(
            "INSERT INTO patches(descriptor_index,node_id,frame_id,stage,row,col,grid_h,grid_w,flow_magnitude,change_magnitude,flow_valid_fraction,flow_valid,change_valid,change_static) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
            (
                int(i),
                int(i),
                window,
                "stage4",
                j // gw,
                j % gw,
                gh,
                gw,
                    float(merged["flow_magnitude"][j]),
                    0.0,
                    float(merged["flow_valid_fraction"][j]),
                    int(good[j]),
                    0,
                    1,
            )
            for j, i in enumerate(ids)
            ],
        )
        fingerprint.update(
            digest_json([video, start, paths[target_index].name, pair_keys]).encode()
        )
        for name in sorted(merged):
            fingerprint.update(np.ascontiguousarray(merged[name]).tobytes())
        if (window + 1) % 25 == 0:
            con.commit()
    con.commit()
    data.flush()
    return data, dict(
        num_frames=len(windows),
        num_patches=data.count,
        feature_dim=0,
        stage4_grid={"H": gh, "W": gw},
        temporal_stride=None,
        feature_lag=None,
        frame_span=max(args.flow_gaps),
        flow_frame_gaps=args.flow_gaps,
        previous_slice=None,
        current_slice=None,
        previous_frame_offset=args.window_len - 1 - max(args.flow_gaps),
        current_frame_offset=args.window_len - 1,
        source_patch_descriptor_sha256=fingerprint.hexdigest(),
    )


def percentile_scale(values, valid, percentile):
    selected = np.asarray(values[valid], dtype=np.float64)
    if not len(selected):
        return 1.0, "no_valid_values"
    value = float(np.percentile(selected, percentile))
    if value <= 1e-12:
        positive = selected[selected > 1e-12]
        return (
            (float(np.percentile(positive, percentile)), "positive_values_fallback")
            if len(positive)
            else (1.0, "all_zero_fallback")
        )
    return value, "valid_moving_training_values"


def normalize_magnitudes(data, args):
    moving = data.role == 2
    population = moving if moving.any() else data.flow_valid
    flow_scale, rule = percentile_scale(
        data.flow_magnitude, population, args.magnitude_percentile
    )
    variation_scale, variation_rule = percentile_scale(
        np.sqrt(data.flow_variance), population, args.magnitude_percentile
    )
    global_scale, global_rule = percentile_scale(
        np.linalg.norm(data.global_motion_raw, axis=1),
        data.flow_valid,
        args.magnitude_percentile,
    )

    def transform(value, scale):
        value = np.maximum(value, 0)
        normalized = (
            value / scale
            if args.magnitude_transform == "clip"
            else np.log1p(value) / np.log1p(scale)
        )
        return np.clip(normalized, 0, 1)

    for start in range(0, data.count, 8192):
        sl = slice(start, start + 8192)
        data.magnitude[sl, 0] = transform(data.flow_magnitude[sl], flow_scale)
        data.magnitude[sl, 1] = transform(
            np.sqrt(data.flow_variance[sl]), variation_scale
        )
        raw = np.asarray(data.global_motion_raw[sl]) / global_scale
        raw /= np.maximum(1, np.linalg.norm(raw, axis=1, keepdims=True))
        data.global_motion[sl] = raw
        # Compact alternative is mean dx,dy and spatial spread in units of the
        # training p95 speed, each bounded; it never uses image embeddings.
        data.compact[sl, :2] = np.clip(data.flow_mean[sl] / flow_scale, -1, 1)
        data.compact[sl, 2] = transform(data.flow_spread[sl], flow_scale)
    data.flush()
    return dict(
        percentile=args.magnitude_percentile,
        population="valid moving training cells; valid fallback if none moving",
        flow_magnitude_scale=flow_scale,
        variation_std_scale=variation_scale,
        global_motion_scale=global_scale,
        flow_scale_rule=rule,
        variation_scale_rule=variation_rule,
        global_scale_rule=global_rule,
        magnitude_transform=args.magnitude_transform,
        clipped=True,
        clip_range=[0, 1],
        flow_transform=(
            "clip(log1p(speed)/log1p(scale),0,1)"
            if args.magnitude_transform == "log"
            else "clip(speed/scale,0,1)"
        ),
        future_inference="Reuse frozen training scales, thresholds, metric and role routing.",
    )


def choose_fit_sample(con, data, args):
    rng = np.random.default_rng(args.seed)
    pool, videos = [], 0
    for frame in con.execute("SELECT DISTINCT video FROM frames ORDER BY video"):
        rows = np.array(
            [
                r[0]
                for r in con.execute(
                    "SELECT p.descriptor_index FROM patches p JOIN frames f ON p.frame_id=f.id WHERE f.video=? ORDER BY p.descriptor_index",
                    (frame[0],),
                )
            ],
            dtype=np.int64,
        )
        rows = rows[data.role[rows] == 2]
        if not len(rows):
            continue
        videos += 1
        if args.fit_max_per_video and len(rows) > args.fit_max_per_video:
            rows = rng.choice(rows, args.fit_max_per_video, replace=False)
        pool.append(rows)
    if not pool:
        return np.empty(0, np.int64), videos
    pool = np.concatenate(pool)
    if len(pool) > args.fit_sample_size:
        pool = rng.choice(pool, args.fit_sample_size, replace=False)
    return np.sort(pool), videos


def distance_matrix(data, rows, columns, weights, batch_size=128):
    matrix = np.empty((len(rows), len(columns)), np.float64)
    b = data.take(columns)
    for start in range(0, len(rows), batch_size):
        matrix[start : start + batch_size] = data.distance(
            data.take(rows[start : start + batch_size]), b, weights
        )
    return matrix


def assign(data, rows, medoids, weights, batch_size=128):
    """Route by reliability/motion role, then nearest medoid within that role.

    Role thresholds are part of the frozen assignment rule, not only discovery.
    Medoid rows own themselves on ties, so duplicate descriptors never empty an
    otherwise valid reference. Unknown flow cannot be mistaken for static flow.
    """
    matrix = distance_matrix(data, rows, medoids, weights, batch_size)
    matrix[data.role[rows, None] != data.role[medoids][None, :]] = np.inf
    group = matrix.argmin(axis=1)
    for j, medoid in enumerate(medoids):
        own = rows == medoid
        group[own] = j
    distance = matrix[np.arange(len(rows)), group]
    if not np.isfinite(distance).all():
        raise RuntimeError("No reference for a descriptor's motion/reliability role.")
    return group, distance


def labels_from_matrix(matrix, medoids):
    labels = matrix[:, medoids].argmin(axis=1)
    labels[medoids] = np.arange(len(medoids))
    return labels


def kmedoids_matrix(matrix, k, args, initial=None):
    """Candidate medoid updates; all fit distances use the configured metric."""
    rng = np.random.default_rng(args.seed)
    best = None
    n = len(matrix)
    for restart in range(1 if initial is not None else args.restarts):
        if initial is not None:
            medoids = np.asarray(initial, dtype=np.int64).copy()
        else:
            chosen = [int(rng.integers(n))]
            closest = matrix[:, chosen[0]].copy()
            for _ in range(1, k):
                probabilities = closest.copy()
                probabilities[chosen] = 0
                remaining = np.setdiff1d(np.arange(n), chosen)
                pick = (
                    int(rng.choice(n, p=probabilities / probabilities.sum()))
                    if probabilities.sum() > 1e-12
                    else int(rng.choice(remaining))
                )
                chosen.append(pick)
                closest = np.minimum(closest, matrix[:, pick])
            medoids = np.asarray(chosen, np.int64)
        history = []
        for iteration in range(args.iterations):
            labels = labels_from_matrix(matrix, medoids)
            updated = medoids.copy()
            for cluster in range(k):
                members = np.flatnonzero(labels == cluster)
                candidates = (
                    members
                    if len(members) <= args.medoid_candidates
                    else np.unique(
                        np.r_[
                            medoids[cluster],
                            rng.choice(members, args.medoid_candidates, replace=False),
                        ]
                    )
                )
                costs = matrix[np.ix_(members, candidates)].mean(axis=0)
                # Keeping the existing medoid is allowed; accepted updates never
                # increase within-cluster cost for the current memberships.
                updated[cluster] = candidates[costs.argmin()]
            labels = labels_from_matrix(matrix, updated)
            objective = float(matrix[np.arange(n), updated[labels]].mean())
            history.append(
                dict(restart=restart, iteration=iteration, mean_distance=objective)
            )
            unchanged = np.array_equal(updated, medoids)
            medoids = updated
            if unchanged:
                break
        if best is None or objective < best[0]:
            best = objective, medoids.copy(), history
    return best[1], best[2], best[0]


def group_medoids(matrix, labels):
    result = []
    for group in np.unique(labels):
        members = np.flatnonzero(labels == group)
        costs = matrix[np.ix_(members, members)].sum(axis=0)
        result.append(members[costs.argmin()])
    return np.asarray(result, np.int64)


def fit_clusters(data, sample, args, weights):
    info, history = {}, []
    if not len(sample):
        moving_medoids, objective = np.empty(0, np.int64), None
    elif len(sample) == 1:
        moving_medoids, objective = sample.copy(), 0.0
        info["singleton_moving_sample"] = True
    else:
        if args.clustering == "finch":
            try:
                if __package__:
                    from .clustering_common import fit_finch
                else:
                    from clustering_common import fit_finch
            except ImportError as exc:
                raise ImportError(
                    "--clustering finch requires your existing clustering_common.py beside this file; average-linkage and kmedoids work standalone."
                ) from exc
            moving_medoids, history, objective, finch = fit_finch(
                data, sample, args, weights
            )
            moving_medoids = np.asarray(moving_medoids, np.int64)
            info["finch"] = finch
        matrix = distance_matrix(data, sample, sample, weights, args.batch_size)
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        if args.clustering != "finch":
            k = min(args.num_prototypes, len(sample))
            if k != args.num_prototypes:
                info["requested_count_capped_to_available_moving_sample"] = True
            if args.clustering == "average-linkage":
                tree = linkage(squareform(matrix, checks=False), method="average")
                # cut_tree supplies the requested count even when merge distances
                # tie; fcluster(maxclust) may silently return fewer clusters.
                from scipy.cluster.hierarchy import cut_tree

                groups = cut_tree(tree, n_clusters=[k]).ravel()
                local_medoids = group_medoids(matrix, groups)
                info["initial_linkage_clusters"] = len(local_medoids)
                if args.linkage_refine:
                    local_medoids, history, objective = kmedoids_matrix(
                        matrix, len(local_medoids), args, initial=local_medoids
                    )
            else:
                local_medoids, history, objective = kmedoids_matrix(matrix, k, args)
            moving_medoids = sample[local_medoids]
        elif args.finch_refine:
            lookup = {int(v): i for i, v in enumerate(sample)}
            if any(int(v) not in lookup for v in moving_medoids):
                raise ValueError(
                    "Shared FINCH returned representatives outside the fit sample."
                )
            initial = np.array([lookup[int(v)] for v in moving_medoids])
            local_medoids, refined_history, objective = kmedoids_matrix(
                matrix, len(initial), args, initial
            )
            moving_medoids = sample[local_medoids]
            history = list(history) + [
                {"phase": "post_finch_kmedoids", **h} for h in refined_history
            ]
        labels, distances = assign(
            data, sample, moving_medoids, weights, args.batch_size
        )
        objective = float(distances.mean())
        info["moving_fit_silhouette"] = silhouette_summary(matrix, labels)
        rng = np.random.default_rng(args.seed + 391)
        info["shuffled_label_baseline"] = silhouette_summary(
            matrix, rng.permutation(labels)
        )
        info["baseline_definition"] = (
            "Shuffle cluster labels across the same fit descriptors, preserving cluster sizes; no reclustering or model feature extraction."
        )
    # One empirical, valid medoid for the entire static population. Unsupported
    # flow gets a separate reference. Use capped representative samples only.
    special_medoids = []
    rng = np.random.default_rng(args.seed + 1)
    for role in (0, 1):
        candidates = np.flatnonzero(data.role == role)
        if not len(candidates):
            continue
        if len(candidates) > 512:
            candidates = np.sort(rng.choice(candidates, 512, replace=False))
        if role == 0:
            medoid = candidates[0]
        else:
            d = distance_matrix(data, candidates, candidates, weights, args.batch_size)
            medoid = candidates[d.mean(axis=0).argmin()]
        special_medoids.append(int(medoid))
    medoids = np.r_[np.asarray(special_medoids, np.int64), moving_medoids]
    if not len(medoids):
        raise ValueError("No descriptors to cluster.")
    info.update(
        moving_reference_count=len(moving_medoids),
        static_reference_count=int(np.any(data.role == 1)),
        invalid_reference_count=int(np.any(data.role == 0)),
        reference_roles=[ROLES[int(data.role[m])] for m in medoids],
    )
    return medoids, history, objective, info


def silhouette_values(matrix, labels):
    """Precomputed dissimilarities; singleton silhouette=0 (standard definition)."""
    groups = np.unique(labels)
    if len(groups) < 2 or len(groups) >= len(labels):
        return None
    means = np.column_stack([matrix[:, labels == g].mean(axis=1) for g in groups])
    indices = np.searchsorted(groups, labels)
    sizes = np.bincount(indices, minlength=len(groups))
    own_n = sizes[indices]
    a = means[np.arange(len(labels)), indices] * own_n / np.maximum(own_n - 1, 1)
    means[np.arange(len(labels)), indices] = np.inf
    b = means.min(axis=1)
    values = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    values[own_n == 1] = 0
    return values


def silhouette_summary(matrix, labels):
    values = silhouette_values(matrix, labels)
    return dict(
        mean=None if values is None else float(values.mean()),
        per_cluster=(
            {}
            if values is None
            else {
                str(int(g)): float(values[labels == g].mean())
                for g in np.unique(labels)
            }
        ),
        sample_size=len(labels),
        reason="requires 2..N-1 populated clusters" if values is None else None,
    )


def save_assignments(con, data, directory, medoids, args, weights):
    k = len(medoids)
    labels = np.lib.format.open_memmap(
        directory / "descriptors/prototype_ids.npy",
        mode="w+",
        shape=(data.count,),
        dtype=np.int32,
    )
    distances = np.lib.format.open_memmap(
        directory / "descriptors/distances.npy",
        mode="w+",
        shape=(data.count,),
        dtype=np.float32,
    )
    for start in tqdm(
        range(0, data.count, args.batch_size), desc="Assign flow patches", unit="batch"
    ):
        rows = np.arange(start, min(start + args.batch_size, data.count))
        group, distance = assign(data, rows, medoids, weights, args.batch_size)
        labels[rows], distances[rows] = group, distance
        con.executemany(
            "UPDATE patches SET prototype_id=?,distance=? WHERE descriptor_index=?",
            [(int(g), float(d), int(i)) for g, d, i in zip(group, distance, rows)],
        )
        if start // args.batch_size % 50 == 0:
            con.commit()
    con.executescript(
        "CREATE INDEX patches_cluster_distance ON patches(prototype_id,distance,node_id); CREATE INDEX patches_cluster_node ON patches(prototype_id,node_id);"
    )
    stats = []
    for j, medoid in enumerate(medoids):
        row = con.execute(
            "SELECT COUNT(*) AS n,AVG(distance) AS d,AVG(flow_magnitude) AS fm FROM patches WHERE prototype_id=?",
            (j,),
        ).fetchone()
        if not row["n"]:
            raise RuntimeError("Empty reference despite medoid self-ownership.")
        videos = con.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT f.video FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.prototype_id=?)",
            (j,),
        ).fetchone()[0]
        con.execute(
            "INSERT INTO prototypes VALUES (?,?,?,?,?,?,?,?)",
            (
                j,
                int(medoid),
                int(data.node_ids[medoid]),
                row["n"],
                videos,
                row["d"],
                row["fm"],
                0.0,
            ),
        )
        stats.append(
            dict(
                prototype_id=j,
                medoid_node_id=int(data.node_ids[medoid]),
                role=ROLES[int(data.role[medoid])],
                patch_count=row["n"],
                video_count=videos,
                mean_distance=row["d"],
                mean_flow_magnitude=row["fm"],
                mean_change_magnitude=0.0,
            )
        )
    con.commit()
    labels.flush()
    distances.flush()
    np.savez_compressed(
        directory / "cluster_references.npz",
        format_version=np.array(FORMAT_VERSION),
        prototype_ids=np.arange(k, dtype=np.int32),
        descriptor_indices=medoids,
        medoid_node_ids=np.asarray(data.node_ids[medoids]),
        weights=np.asarray(weights),
        flow_metric=np.array(args.flow_metric),
        weight_names=np.array(
            ["flow", "cross_gap_variation", "flow_magnitude", "global_motion"]
        ),
        **{name: np.asarray(getattr(data, name)[medoids]) for name in data.SPECS},
    )
    return stats


def cohesion_diagnostics(data, directory, medoids, clusters, args, weights):
    rng = np.random.default_rng(args.seed + 204)
    # A role-balanced audit, in addition to the moving-only discovery audit.
    parts = []
    for role in (0, 1, 2):
        rows = np.flatnonzero(data.role == role)
        if len(rows) > args.diagnostic_sample_size // 3:
            rows = rng.choice(rows, args.diagnostic_sample_size // 3, replace=False)
        parts.append(rows)
    rows = np.sort(np.concatenate(parts))
    labels, distances = assign(data, rows, medoids, weights, args.batch_size)
    matrix = distance_matrix(data, rows, rows, weights, args.batch_size)
    # Assignment deliberately separates roles, so diagnostic distance includes
    # the role boundary: add the metric's maximum for cross-role comparisons.
    role_penalty = float(
        weights[0] * (np.sqrt(3) if args.flow_metric == "compact" else 1)
        + sum(weights[1:])
    )
    mismatch = data.role[rows, None] != data.role[rows][None, :]
    matrix[mismatch] += role_penalty
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    actual = silhouette_summary(matrix, labels)
    shuffled_labels = rng.permutation(labels)
    baseline = silhouette_summary(matrix, shuffled_labels)
    for cluster in clusters:
        cluster["silhouette_audit"] = actual["per_cluster"].get(
            str(cluster["prototype_id"])
        )
        cluster["shuffled_label_silhouette_audit"] = baseline["per_cluster"].get(
            str(cluster["prototype_id"])
        )
        cluster["silhouette_audit_patch_count"] = int(
            np.count_nonzero(labels == cluster["prototype_id"])
        )
    # Direct shuffled-assignment medoid cohesion, compared on the identical rows.
    shuffled_distance = []
    for start in range(0, len(rows), args.batch_size):
        r = rows[start : start + args.batch_size]
        d = distance_matrix(data, r, medoids, weights, args.batch_size)
        g = shuffled_labels[start : start + len(r)]
        selected = d[np.arange(len(r)), g]
        selected += role_penalty * (data.role[r] != data.role[medoids[g]])
        shuffled_distance.extend(selected.tolist())
    result = dict(
        sample_size=len(rows),
        sample_method="uniform per role up to one third of audit limit",
        silhouette=actual,
        shuffled_label_silhouette=baseline,
        mean_distance_to_own_medoid=float(distances.mean()),
        shuffled_assignment_mean_distance_to_medoid=float(np.mean(shuffled_distance)),
        role_boundary_penalty=role_penalty,
        note="Discovery diagnostics evaluate moving cells alone. This audit includes static/invalid roles; null cluster silhouettes indicate no audit support, or undefined global silhouette. Baseline shuffles labels, preserving sizes.",
    )
    np.save(directory / "diagnostic_sample_indices.npy", rows, allow_pickle=False)
    write_json(directory / "cohesion_diagnostics.json", result)
    return result


def descriptor_config(args):
    return dict(
        mode="flow-only",
        image_features_extracted=False,
        backbone_loaded=False,
        grid_source="explicit spatial patch size; no backbone geometry inference",
        patch_size=list(args.patch_size),
        flow_frame_gaps=args.flow_gaps,
        units="canvas pixels per frame index (not pixels per second)",
        flow_anchor="All pairs end at current-frame pixels; apparent forward motion = -backward flow",
        histogram="8 circular, linearly interpolated, log-speed-weighted direction bins plus static fraction",
        histogram_metric=args.flow_metric,
        direction_axes="x right, y down; +x bin0, clockwise",
        camera_motion=args.camera_motion,
        camera_ransac_threshold=args.camera_ransac_threshold,
        flow_static_threshold=args.flow_static_threshold,
        static_threshold_units="canvas pixels/frame",
        static_patch_fraction=args.static_patch_fraction,
        min_flow_valid_fraction=args.min_flow_valid_fraction,
        min_valid_gap_fraction=args.min_valid_gap_fraction,
        forward_backward_absolute=args.flow_fb_absolute,
        forward_backward_relative=args.flow_fb_relative,
        consistency_units="displacement canvas pixels before division by gap",
        static_policy="valid cells with >= static_patch_fraction static pixels route to one static reference; moving fit sample excludes them",
        invalid_policy="insufficient valid gaps routes to one unknown-flow reference; never treated as static",
        pooling="equal mean of usable per-gap descriptors at same target spatial cell; not Lagrangian tracking",
        variation="sqrt(mean squared spread of per-gap patch mean residual velocity); not acceleration",
        legacy_change_fields="zero placeholders; change_direction [N,0]; change_valid false",
        assignment="nearest medoid within role; reference rows own themselves on distance ties",
    )


class CudaFarnebackEstimator:
    """Bidirectional Farneback via OpenCV CUDA, without Torch or CPU fallback."""

    def __init__(self, recorded_config, device=0):
        self.device = int(device)
        self.model = None
        self.gpu_previous = None
        self.gpu_current = None
        cuda = getattr(cv2, "cuda", None)
        error = (
            "CUDA Farneback requires an NVIDIA GPU, compatible driver and an OpenCV "
            "build with CUDA + opencv_contrib/cudaoptflow. Standard opencv-python/"
            "opencv-contrib-python PyPI wheels are CPU-only. No CPU fallback was used."
        )
        if cuda is None or not hasattr(cuda, "getCudaEnabledDeviceCount"):
            raise RuntimeError(error)
        try:
            count = cuda.getCudaEnabledDeviceCount()
        except cv2.error as exc:
            raise RuntimeError(error) from exc
        if count <= 0:
            raise RuntimeError(f"{error} OpenCV reports {count} usable CUDA devices.")
        if not 0 <= self.device < count:
            raise ValueError(f"--cuda-device {self.device} is out of range; {count} device(s) available.")
        factory_class = getattr(cv2, "cuda_FarnebackOpticalFlow", None)
        if factory_class is None:
            factory_class = getattr(cuda, "FarnebackOpticalFlow", None)
        self.factory = getattr(factory_class, "create", None)
        if self.factory is None:
            self.factory = getattr(cuda, "FarnebackOpticalFlow_create", None)
        self.gpu_mat_factory = getattr(cv2, "cuda_GpuMat", None)
        if self.gpu_mat_factory is None:
            self.gpu_mat_factory = getattr(cuda, "GpuMat", None)
        if self.factory is None or self.gpu_mat_factory is None:
            raise RuntimeError(f"{error} CUDA Farneback/GpuMat Python bindings are missing.")
        try:
            cuda.setDevice(self.device)
        except cv2.error as exc:
            raise RuntimeError(f"Cannot select OpenCV CUDA device {self.device}: {exc}") from exc
        self.config = json.loads(json.dumps(recorded_config))
        if self.config.get("estimator") not in ("farneback", "farneback-cuda"):
            raise ValueError("--farneback-backend cuda requires a Farneback bank (not DIS/RAFT).")
        parameters = self.config.get("parameters", {})
        required = {"pyr_scale", "levels", "winsize", "iterations", "poly_n", "poly_sigma", "flags"}
        if not required.issubset(parameters):
            raise ValueError("The bank must record all Farneback parameters for CUDA mapping.")
        # The module estimates every direction independently with no initial flow.
        if parameters["flags"] != 0 or parameters.get("fast_pyramids", False):
            raise ValueError("CUDA mode currently requires flags=0 and fast_pyramids=False.")
        self.config.update(
            estimator="farneback-cuda",
            backend="opencv_cuda",
            opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            cuda_implementation="bidirectional_grayscale_farneback_v1",
        )
        self.config["parameters"]["fast_pyramids"] = False
        if recorded_config["estimator"] == "farneback-cuda" and self.config != recorded_config:
            raise ValueError("CUDA Farneback build/settings differ from the GPU bank's recorded provenance.")

    def estimate(self, previous, current):
        try:
            cv2.cuda.setDevice(self.device)
            if self.model is None:
                p = self.config["parameters"]
                # Positional order follows cv::cuda::FarnebackOpticalFlow::create.
                self.model = self.factory(
                    int(p["levels"]), float(p["pyr_scale"]), False,
                    int(p["winsize"]), int(p["iterations"]), int(p["poly_n"]),
                    float(p["poly_sigma"]), int(p["flags"]),
                )
                self.gpu_previous = self.gpu_mat_factory()
                self.gpu_current = self.gpu_mat_factory()
            # Match the CPU bank's RGB -> grayscale conversion exactly; only
            # flow estimation runs on CUDA. Reuse uploaded inputs both ways.
            self.gpu_previous.upload(cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY))
            self.gpu_current.upload(cv2.cvtColor(current, cv2.COLOR_RGB2GRAY))
            forward = self.model.calc(self.gpu_previous, self.gpu_current, None).download()
            backward = self.model.calc(self.gpu_current, self.gpu_previous, None).download()
            values = tuple(np.asarray(v, dtype=np.float32) for v in (forward, backward))
            if any(v.shape != (*previous.shape[:2], 2) or not np.isfinite(v).all() for v in values):
                raise ValueError("CUDA Farneback returned invalid flow; no entry written.")
            return values
        except cv2.error as exc:
            raise RuntimeError(f"CUDA Farneback failed on device {self.device}; no CPU fallback: {exc}") from exc

def resolve_window_index(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.name != "index.sqlite":
            raise ValueError("--from-window-cache must name index.sqlite or its cache directory.")
        return path
    candidates = sorted(set(path.glob("index.sqlite")) |
                        set(path.glob("*/index.sqlite")) |
                        set(path.glob("optical_flow_training_inputs_v1/*/index.sqlite")))
    if len(candidates) != 1:
        choices = "\n".join(str(p) for p in candidates)
        raise ValueError(f"Expected one window-cache index under {path}; found {len(candidates)}. "
                         f"Pass an exact index.sqlite.\n{choices}")
    return candidates[0]


class WindowDescriptorSource:
    """Compact training windows only; no RGB decoding, raw flow or old labels."""

    FIELDS = {"flow_hist": (9,), "flow_magnitude": (), "flow_mean": (2,),
              "flow_spread": (), "flow_variance": (), "global_motion_raw": (2,),
              "flow_valid_fraction": (), "flow_valid": (), "role": ()}

    def __init__(self, path, allow_partial=False):
        self.index = resolve_window_index(path)
        self.directory = self.index.parent
        with closing(sqlite3.connect(self.index.as_uri() + "?mode=ro", uri=True)) as con:
            row = con.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if row is None:
            raise ValueError("Window cache has no fixed descriptor contract.")
        self.contract = json.loads(row[0])
        if self.contract.get("schema") != "optical_flow_training_inputs_v1":
            raise ValueError("Expected an optical_flow_training_inputs_v1 cache.")
        manifest = self.directory / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if self.manifest.get("contract", self.contract) != self.contract:
            raise ValueError("Source cache manifest/index contracts disagree.")
        if self.manifest.get("status") != "complete" and not allow_partial:
            raise ValueError("Source cache is not complete. Finish its build, or explicitly use --allow-partial-cache.")
        snapshot = self.directory / "bank_contract.json"
        self.bank_manifest = json.loads(snapshot.read_text()).get("manifest", {}) if snapshot.is_file() else {}
        self.camera_seed = self.bank_manifest.get("seed")
        if self.camera_seed is None and self.contract["descriptor"]["camera_motion"] == "affine":
            raise ValueError("Affine viewer reconstruction requires the source bank_contract.json with its camera-fit seed.")
        self.camera_seed = int(self.camera_seed if self.camera_seed is not None else 42)
        if self.contract["flow_config"]["schema"] != FLOW_CACHE_SCHEMA:
            raise ValueError("Unsupported raw-flow provenance schema in window cache.")

    def rows(self, stride, maximum):
        with closing(sqlite3.connect(self.index.as_uri() + "?mode=ro", uri=True)) as con:
            con.row_factory = sqlite3.Row
            result = con.execute(
                "SELECT video,window_start,target_index,path,cache_key FROM windows "
                "WHERE split='train' AND augmentation='original' AND window_start % ?=0 "
                "ORDER BY video,window_start,cache_key", (stride,)
            ).fetchall()
        if maximum is not None:
            result = result[:maximum]
        if not result:
            raise ValueError("No indexed original-view training windows match the requested stride/limit.")
        return result

    def load(self, row):
        path = (self.directory / row["path"]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Window cache index contains a path outside its namespace.")
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["metadata"]))
            if (meta.get("cache_key") != row["cache_key"]
                    or meta.get("namespace") != self.contract["namespace"]
                    or meta.get("bank_identity") != self.contract["bank_identity"]
                    or meta.get("split") != "train" or meta.get("augmentation") != "original"
                    or meta.get("video") != row["video"]
                    or meta.get("window_start") != row["window_start"]
                    or meta.get("target_index") != row["target_index"]
                    or meta.get("grid_hw") != self.contract["grid_hw"]
                    or meta.get("patch_size") != self.contract["patch_size"]
                    or meta.get("canvas_hw") != self.contract["canvas_hw"]
                    or meta.get("flow_frame_gaps") != self.contract["flow_frame_gaps"]):
                raise ValueError(f"Source window identity/geometry mismatch: {path}")
            n = int(np.prod(meta["grid_hw"]))
            values = {}
            for name, tail in self.FIELDS.items():
                if name not in payload:
                    raise ValueError(f"Window lacks unnormalized descriptor {name}: {path}")
                value = payload[name]
                if value.shape != (n, *tail) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid cached descriptor {name}: {path}")
                values[name] = value
            if np.any((values["role"] < 0) | (values["role"] > 2)):
                raise ValueError(f"Invalid cached static/moving/unknown roles: {path}")
            pairs = meta["pairs"]
            if [p["gap"] for p in pairs] != self.contract["flow_frame_gaps"]:
                raise ValueError(f"Inconsistent cached flow pairs: {path}")
        return values, meta


def inherit_window_settings(args, explicit, parser):
    source = WindowDescriptorSource(args.from_window_cache, args.allow_partial_cache)
    args.window_source = source
    contract, desc = source.contract, source.contract["descriptor"]
    fixed = {
        "resize_to": contract["canvas_hw"], "patch_size": contract["patch_size"],
        "window_len": contract["window_len"], "flow_estimator": contract["flow_config"]["estimator"]["estimator"],
        "flow_storage_dtype": contract["flow_config"]["storage_dtype"],
        "camera_motion": desc["camera_motion"], "camera_ransac_threshold": desc["camera_ransac_threshold"],
        "flow_static_threshold": desc["flow_static_threshold"], "static_patch_fraction": desc["static_patch_fraction"],
        "min_flow_valid_fraction": desc["min_flow_valid_fraction"], "min_valid_gap_fraction": desc["min_valid_gap_fraction"],
        "flow_fb_absolute": desc["forward_backward_absolute"], "flow_fb_relative": desc["forward_backward_relative"],
        "raft_updates": contract["flow_config"]["estimator"].get("iterations", 12),
    }
    for name, value in fixed.items():
        option = "--" + name.replace("_", "-")
        if option in explicit and getattr(args, name) != value:
            parser.error(f"{option} changes descriptor extraction; cache records {value}. Rebuild descriptors to change it.")
        setattr(args, name, value)
    requested = args.flow_gaps if args.flow_gaps is not None else ([args.flow_gap] if args.flow_gap is not None else None)
    if requested is not None and sorted(set(requested)) != contract["flow_frame_gaps"]:
        parser.error("Flow gaps differ from cached descriptors; they cannot change during cache-only clustering.")
    args.flow_gaps = contract["flow_frame_gaps"]
    defaults = {
        "stride": 1, "flow_metric": desc["histogram_metric"],
        "magnitude_transform": contract["normalization"]["magnitude_transform"],
        "magnitude_percentile": contract["normalization"]["percentile"],
        "flow_weight": contract["weights"]["flow"],
        "variation_weight": contract["weights"]["cross_gap_variation"],
        "magnitude_weight": contract["weights"]["flow_magnitude"],
        "global_weight": contract["weights"]["global_motion"],
    }
    for name, value in defaults.items():
        if "--" + name.replace("_", "-") not in explicit:
            setattr(args, name, value)
    if "--dataset-root" not in explicit:
        train_root = source.manifest.get("settings", {}).get("roots", {}).get("train")
        if train_root:
            args.dataset_root = Path(train_root).parent
    output = args.output_dir.expanduser().resolve()
    cache_root = source.directory.parent.parent
    if output == cache_root or output in cache_root.parents or cache_root in output.parents:
        parser.error("Output cluster bank must be separate from the source window-cache tree.")


def extract_window_descriptors(args, directory, con):
    source = args.window_source
    rows = source.rows(args.stride, args.max_windows)
    gh, gw = source.contract["grid_hw"]
    per_frame = gh * gw
    data = TemporalDescriptors(directory / "descriptors", len(rows) * per_frame,
                               mode="w+", metric=args.flow_metric)
    data.change_magnitude[:] = 0
    data.change_valid[:] = False
    data.change_static[:] = True
    fingerprint = hashlib.sha256(digest_json(descriptor_config(args)).encode())
    for frame_id, row in enumerate(tqdm(rows, desc="Read cached training descriptors", unit="window")):
        values, meta = source.load(row)
        sl = slice(frame_id * per_frame, (frame_id + 1) * per_frame)
        for name, value in values.items():
            getattr(data, name)[sl] = value
        data.node_ids[sl] = np.arange(sl.start, sl.stop, dtype=np.int64)
        target = row["target_index"]
        previous = target - max(args.flow_gaps)
        pair = next(p for p in meta["pairs"] if p["gap"] == max(args.flow_gaps))
        images = args.dataset_root.expanduser().resolve() / "train" / row["video"] / "images"
        keys = [p["raw_pair_key"] for p in meta["pairs"]]
        models = {"pairs": [{"frame_gap": g, "used": "viewer_not_prepared"} for g in args.flow_gaps],
                  "pair_keys": keys, "viewer_flow_prepared": False}
        con.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            frame_id, "train", row["video"], frame_id, row["window_start"], target,
            meta["target_name"], previous, pair["previous_name"], -1, -1,
            args.window_len - 1 - max(args.flow_gaps), args.window_len - 1, *args.resize_to,
            str(images / meta["target_name"]), str(images / pair["previous_name"]),
            keys[0] if len(keys) == 1 else digest_json(keys), json.dumps(models, sort_keys=True)))
        con.executemany(
            "INSERT INTO patches(descriptor_index,node_id,frame_id,stage,row,col,grid_h,grid_w,flow_magnitude,change_magnitude,flow_valid_fraction,flow_valid,change_valid,change_static) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(i, i, frame_id, "stage4", j // gw, j % gw, gh, gw,
              float(values["flow_magnitude"][j]), 0., float(values["flow_valid_fraction"][j]),
              int(values["flow_valid"][j]), 0, 1) for j, i in enumerate(range(sl.start, sl.stop))])
        fingerprint.update(row["cache_key"].encode())
        for name in sorted(values):
            fingerprint.update(np.ascontiguousarray(values[name]).tobytes())
        if (frame_id + 1) % 25 == 0:
            con.commit()
    con.commit()
    data.flush()
    return data, dict(num_frames=len(rows), num_patches=data.count, feature_dim=0,
        stage4_grid={"H": gh, "W": gw}, temporal_stride=None, feature_lag=None,
        frame_span=max(args.flow_gaps), flow_frame_gaps=args.flow_gaps,
        previous_slice=None, current_slice=None,
        previous_frame_offset=args.window_len - 1 - max(args.flow_gaps),
        current_frame_offset=args.window_len - 1,
        source_patch_descriptor_sha256=fingerprint.hexdigest(),
        clustering_flow_estimator_calls=0, clustering_rgb_decodes=0,
        source_cache_index=str(source.index), source_cache_status=source.manifest.get("status"),
        source_cached_views="train/original only; old cluster labels ignored")


class ViewerFlowArchive:
    """Save raw flow only for selected examples, reusing shared frame pairs."""

    def __init__(self, args, directory, config):
        self.args, self.config = args, config
        self.namespace = digest_json(config)
        self.directory = directory / "viewer_flow" / FLOW_CACHE_SCHEMA / self.namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / "manifest.json", config)
        self.estimator = None
        self.dtype = np.dtype(config["storage_dtype"])
        source = args.window_source
        candidates = [self.directory, source.directory.parent.parent / "raw_pairs" / FLOW_CACHE_SCHEMA / self.namespace]
        candidates += [Path(p) for p in source.manifest.get("raw_cache_sources", [])]
        if source.manifest.get("raw_cache_directory"):
            candidates.append(Path(source.manifest["raw_cache_directory"]))
        self.sources = []
        for path in candidates:
            path = path.expanduser().resolve()
            manifest = path / "manifest.json"
            if path not in self.sources and manifest.is_file() and json.loads(manifest.read_text()) == config:
                self.sources.append(path)
        self.stats = dict(estimator_calls=0, reused_pairs=0, saved_pairs=0)

    def get(self, previous, current, expected_key):
        h = hashlib.sha256(str(previous.shape).encode())
        h.update(np.ascontiguousarray(previous).tobytes())
        h.update(np.ascontiguousarray(current).tobytes())
        key = h.hexdigest()
        if key != expected_key:
            raise ValueError("Selected RGB pair differs from cached content. Check dataset root, frame order and preprocessing.")
        relative = Path("entries") / key[:2] / f"{key}.npz"
        values = None
        for source in self.sources:
            path = source / relative
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as payload:
                if str(payload["namespace"]) != self.namespace or str(payload["pair_sha256"]) != key:
                    raise ValueError(f"Selected raw-flow identity mismatch: {path}")
                values = [payload[name] for name in ("forward", "backward")]
            self.stats["reused_pairs"] += 1
            break
        if values is None:
            recorded = self.config["estimator"]
            if self.estimator is None:
                if recorded["estimator"] == "farneback-cuda":
                    self.estimator = CudaFarnebackEstimator(recorded, self.args.gpu or 0)
                else:
                    self.estimator = FlowEstimator(self.args)
                    if self.estimator.config != recorded:
                        raise ValueError("Viewer estimator/version/settings differ from cache provenance. Use matching OpenCV/weights.")
            values = [v.astype(self.dtype) for v in self.estimator.estimate(previous, current)]
            self.stats["estimator_calls"] += 1
        if any(v.shape != (*current.shape[:2], 2) or v.dtype != self.dtype or not np.isfinite(v).all() for v in values):
            raise ValueError("Invalid selected-example raw flow.")
        path = self.directory / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, forward=values[0], backward=values[1],
                                    namespace=np.array(self.namespace), pair_sha256=np.array(key))
            try:
                try:
                    os.link(temporary, path)
                    self.stats["saved_pairs"] += 1
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)
        return values[0].astype(np.float32), values[1].astype(np.float32)


def selected_patch_summary(residual, global_flow, valid, patches, args):
    n = len(patches)
    result = dict(flow_hist=np.zeros((n, 9), np.float32), flow_magnitude=np.zeros(n, np.float32),
                  flow_mean=np.zeros((n, 2), np.float32), flow_spread=np.zeros(n, np.float32),
                  global_motion_raw=np.zeros((n, 2), np.float32), flow_valid_fraction=np.zeros(n, np.float32))
    ph, pw = args.patch_size
    for i, patch in enumerate(patches):
        sl = np.s_[patch["row"] * ph:(patch["row"] + 1) * ph,
                   patch["col"] * pw:(patch["col"] + 1) * pw]
        mask = valid[sl]
        result["flow_valid_fraction"][i] = mask.mean()
        if not mask.any():
            continue
        vectors = residual[sl][mask]
        result["flow_hist"][i] = soft_histogram(vectors, args.flow_static_threshold)
        result["flow_magnitude"][i] = np.linalg.norm(vectors, axis=-1).mean()
        result["flow_mean"][i] = vectors.mean(axis=0)
        result["flow_spread"][i] = np.sqrt(np.square(vectors - result["flow_mean"][i]).sum(axis=-1).mean())
        result["global_motion_raw"][i] = global_flow[sl][mask].mean(axis=0)
    return result


def merge_selected_summaries(summaries, args):
    usable = np.stack([s["flow_valid_fraction"] >= args.min_flow_valid_fraction for s in summaries])
    count = usable.sum(axis=0)
    merged = {}
    for name in summaries[0]:
        values = np.stack([s[name] for s in summaries])
        if name == "flow_valid_fraction":
            merged[name] = values.mean(axis=0)
        else:
            mask = usable[..., None] if values.ndim == 3 else usable
            denominator = np.maximum(count, 1)
            if values.ndim == 3:
                denominator = denominator[:, None]
            merged[name] = (values * mask).sum(axis=0) / denominator
    means = np.stack([s["flow_mean"] for s in summaries])
    variance = (np.square(means - merged["flow_mean"]).sum(axis=-1) * usable).sum(axis=0) / np.maximum(count, 1)
    merged = {k: np.asarray(v, np.float32) for k, v in merged.items()}
    merged["flow_variance"] = variance.astype(np.float32)
    merged["flow_valid"] = count >= math.ceil(len(summaries) * args.min_valid_gap_fraction)
    return merged


def prepare_viewer_examples(args, directory, con, data, manifest):
    if args.viewer_examples == 0:
        manifest.update(viewer_examples_only=True, viewer_examples_per_cluster=0,
                        viewer_cache={"status": "disabled", "estimator_calls": 0})
        return
    recorded_cv = manifest["flow_provenance"]["estimator"].get("opencv_version")
    if recorded_cv and recorded_cv != cv2.__version__:
        raise ValueError(f"Viewer reconstruction needs recorded OpenCV {recorded_cv}; installed {cv2.__version__}.")
    con.execute("CREATE TABLE IF NOT EXISTS viewer_examples (prototype_id INTEGER NOT NULL, node_id INTEGER NOT NULL UNIQUE, rank INTEGER NOT NULL, is_medoid INTEGER NOT NULL, PRIMARY KEY(prototype_id,rank))")
    con.execute("DELETE FROM viewer_examples")
    selections = []
    for prototype in con.execute("SELECT id,medoid_node_id FROM prototypes ORDER BY id").fetchall():
        patches = con.execute(
            "SELECT * FROM patches WHERE prototype_id=? ORDER BY CASE WHEN node_id=? THEN 0 ELSE 1 END,distance,node_id LIMIT ?",
            (prototype["id"], prototype["medoid_node_id"], args.viewer_examples)).fetchall()
        if not patches or patches[0]["node_id"] != prototype["medoid_node_id"]:
            raise ValueError("Viewer selection must start with its cluster medoid.")
        for rank, patch in enumerate(patches):
            patch = dict(patch)
            patch.update(rank=rank, is_medoid=int(rank == 0))
            selections.append(patch)
            con.execute("INSERT INTO viewer_examples VALUES (?,?,?,?)",
                        (prototype["id"], patch["node_id"], rank, int(rank == 0)))
    con.commit()
    archive = ViewerFlowArchive(args, directory, manifest["flow_provenance"])
    grouped = {}
    for patch in selections:
        grouped.setdefault(patch["frame_id"], []).append(patch)
    recomputed = {}
    paths_by_video = {}
    for frame_id, patches in tqdm(sorted(grouped.items()), desc="Prepare selected viewer frames", unit="frame"):
        frame = con.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
        if frame["video"] not in paths_by_video:
            images = Path(frame["source_path"]).parent
            paths_by_video[frame["video"]] = sorted(
                [p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
                key=lambda p: natural_key(p.name))
        paths = paths_by_video[frame["video"]]
        target = frame["target_index"]
        if target >= len(paths) or paths[target].name != frame["target_name"]:
            raise ValueError("Selected viewer frame ordering differs from the source cache.")
        current = read_canvas(paths[target], args.resize_to)
        keys = json.loads(frame["global_model_json"])["pair_keys"]
        summaries, cameras = [], []
        for gap, key in zip(args.flow_gaps, keys):
            previous = read_canvas(paths[target - gap], args.resize_to)
            forward, backward = archive.get(previous, current, key)
            valid = target_flow_validity(forward, backward, args.flow_fb_absolute, args.flow_fb_relative)
            residual, global_flow, camera = compensate_camera(
                -backward, valid, args.camera_motion, args.window_source.camera_seed, args.camera_ransac_threshold)
            summaries.append(selected_patch_summary(residual / gap, global_flow / gap, valid, patches, args))
            cameras.append(dict(frame_gap=gap, **camera))
        merged = merge_selected_summaries(summaries, args)
        for i, patch in enumerate(patches):
            row = patch["descriptor_index"]
            for name, values in merged.items():
                expected = np.asarray(getattr(data, name)[row])
                if not np.allclose(values[i], expected, atol=args.viewer_descriptor_tolerance,
                                   rtol=args.viewer_descriptor_tolerance):
                    raise ValueError(f"Viewer descriptor differs from cached {name}, node {patch['node_id']}. "
                                     "Check recorded estimator/preprocessing; clusters remain saved. "
                                     "Use --prepare-viewer-only after correcting the environment.")
            recomputed[patch["node_id"]] = {name: value[i] for name, value in merged.items()}
        con.execute("UPDATE frames SET global_model_json=? WHERE id=?", (
            json.dumps(dict(pairs=cameras, pair_keys=keys, viewer_flow_prepared=True), sort_keys=True), frame_id))
        con.commit()
    example_dir = directory / "viewer_examples"
    example_dir.mkdir(exist_ok=True)
    for cluster in manifest["clusters"]:
        patches = sorted([p for p in selections if p["prototype_id"] == cluster["prototype_id"]], key=lambda p: p["rank"])
        np.savez_compressed(example_dir / f"cluster_{cluster['prototype_id']:04}.npz",
            node_ids=np.array([p["node_id"] for p in patches], np.int64),
            distances=np.array([p["distance"] for p in patches], np.float32),
            **{name: np.stack([recomputed[p["node_id"]][name] for p in patches]) for name in next(iter(recomputed.values()))})
        cluster["viewer_example_count"] = len(patches)
    manifest.update(viewer_examples_only=True, viewer_examples_per_cluster=args.viewer_examples,
        viewer_example_selection="medoid first, then nearest distinct patches by distance/node_id; undersized clusters keep all",
        flow_cache_directory=str(archive.directory), flow_cache_namespace=archive.namespace,
        viewer_cache=dict(status="complete", selected_patches=len(selections), selected_frames=len(grouped),
                          **archive.stats), viewer_camera_seed=args.window_source.camera_seed)


def prepare_viewer_only(args):
    directory = args.output_dir.expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("format_version") != FORMAT_VERSION or not manifest.get("num_prototypes")
            or sha256_file(directory / manifest["references"]) != manifest["references_sha256"]):
        raise ValueError("--prepare-viewer-only needs an already clustered bank with intact references.")
    if manifest.get("source_cache_index") != str(args.window_source.index):
        raise ValueError("Viewer-only source cache differs from the clustered bank.")
    if descriptor_config(args) != manifest["descriptor"] or args.window_len != manifest["window_len"]:
        raise ValueError("Viewer-only descriptor settings differ from the clustered bank.")
    data = TemporalDescriptors(directory / "descriptors", mode="r", metric=args.flow_metric)
    try:
        with closing(sqlite3.connect(directory / manifest["database"])) as con:
            con.row_factory = sqlite3.Row
            prepare_viewer_examples(args, directory, con, data, manifest)
            manifest.update(status="complete", completed_at=now())
            manifest.pop("error", None)
            manifest.pop("failed_at", None)
            con.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", ("manifest", json.dumps(manifest, allow_nan=False)))
            con.commit()
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), failed_at=now())
        write_json(directory / "manifest.json", manifest)
        raise
    write_json(directory / "manifest.json", manifest)
    return manifest


def build(args, feature_cache=None, flow_cache=None):
    if feature_cache is not None:
        raise ValueError("Feature caches are disabled in this flow-only script.")
    validate_output_location(args)
    cached = getattr(args, "window_source", None)
    if cached is not None and args.prepare_viewer_only:
        return prepare_viewer_only(args)
    if cached is None:
        flow_cache = flow_cache or FlowCache(args)
        flow_config, flow_namespace = flow_cache.config, flow_cache.namespace
        flow_directory = str(flow_cache.directory)
    else:
        if flow_cache is not None:
            raise ValueError("Window-cache clustering must not be supplied a raw-flow extractor.")
        flow_config = cached.contract["flow_config"]
        flow_namespace = digest_json(flow_config)
        flow_directory = None
    directory = prepare_directory(args.output_dir, args.overwrite)
    weights = (
        args.flow_weight,
        args.variation_weight,
        args.magnitude_weight,
        args.global_weight,
    )
    manifest = dict(
        format_version=FORMAT_VERSION,
        status="building",
        created_at=now(),
        stage="stage4",
        source_mode="cached_training_patch_descriptors" if cached else "training_rgb_flow_only",
        datasets=["train"],
        source_dataset_root=str(args.dataset_root.expanduser().resolve()),
        reference_groups_fixed=True,
        saliency_used_for_clustering=False,
        model_assignments_used_for_clustering=False,
        window_len=args.window_len,
        window_stride=args.stride,
        descriptor=descriptor_config(args),
        distance="w_flow*flow_metric + w_mag*abs(normalized_flow_speed) + w_variation*abs(normalized_cross_gap_rate_std) + w_global*L1(normalized_camera_velocity)/2; validity gating; nearest within role",
        weights=dict(
            zip(
                ("flow", "cross_gap_variation", "flow_magnitude", "global_motion"),
                weights,
            )
        ),
        seed=args.seed,
        clustering=args.clustering,
        fit_sample_size_requested=args.fit_sample_size,
        fit_max_per_video=args.fit_max_per_video,
        fit_population="moving valid cells only",
        image_features_extracted=False,
        backbone_loaded=False,
        reference_feature_provenance=None,
        flow_provenance=flow_config,
        flow_cache_namespace=flow_namespace,
        flow_cache_directory=flow_directory,
        compatibility_note="v2 flow-only semantics require updated consumers; old change fields are neutral placeholders",
    )
    write_json(directory / "manifest.json", manifest)
    try:
        with closing(sqlite3.connect(directory / "temporal_clusters.sqlite")) as con:
            con.row_factory = sqlite3.Row
            create_schema(con)
            data, extraction = (
                extract_window_descriptors(args, directory, con) if cached else
                extract_descriptors(args, directory, con, None, flow_cache)
            )
            manifest.update(extraction)
            manifest["coverage"] = {
                f"{ROLES[r]}_patches": int(np.count_nonzero(data.role == r))
                for r in (0, 1, 2)
            }
            manifest["normalization"] = normalize_magnitudes(data, args)
            sample, num_moving_videos = choose_fit_sample(con, data, args)
            np.save(directory / "fit_sample_indices.npy", sample, allow_pickle=False)
            print(
                f"{data.count:,} cells; {len(sample):,} moving fit cells from {num_moving_videos} videos; RGB gaps {args.flow_gaps}; NO backbone/features",
                flush=True,
            )
            medoids, history, objective, info = fit_clusters(
                    data, sample, args, weights
                )
            clusters = save_assignments(con, data, directory, medoids, args, weights)
            diagnostics = cohesion_diagnostics(
                data, directory, medoids, clusters, args, weights
            )
            if (
                con.execute(
                    "SELECT COUNT(*) FROM patches WHERE prototype_id IS NULL"
                ).fetchone()[0]
                or con.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            ):
                raise RuntimeError("Final database integrity check failed.")
            manifest.update(
                status="complete",
                completed_at=now(),
                num_prototypes=len(medoids),
                num_videos=con.execute(
                    "SELECT COUNT(DISTINCT video) FROM frames"
                ).fetchone()[0],
                moving_fit_video_count=num_moving_videos,
                fit_sample_size=len(sample),
                fit_history=history,
                fit_mean_distance=objective,
                clustering_details=info,
                clusters=clusters,
                cohesion_diagnostics=diagnostics,
                references_sha256=sha256_file(directory / "cluster_references.npz"),
                feature_cache_stats=dict(hits=0, misses=0, writes=0, forwards=0),
                flow_cache_stats=flow_cache.stats if flow_cache is not None else dict(hits=0, misses=0, writes=0, estimator_calls=0),
                database="temporal_clusters.sqlite",
                references="cluster_references.npz",
                algorithm=f"sampled_{args.clustering}_moving_references_then_role_routed_nearest_medoid_assignment",
            )
            if cached is not None:
                manifest["status"] = "clusters_complete"
                write_json(directory / "manifest.json", manifest)
                prepare_viewer_examples(args, directory, con, data, manifest)
                manifest["status"] = "complete"
            con.execute(
                "INSERT INTO metadata VALUES (?,?)",
                ("manifest", json.dumps(manifest, allow_nan=False)),
            )
            con.execute("ANALYZE")
            con.commit()
        write_json(directory / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", failed_at=now(), error=str(exc))
        write_json(directory / "manifest.json", manifest)
        raise
    print(
        f"Complete: {directory} | {len(medoids)} flow-only references; backbone forwards=0",
        flush=True,
    )
    return manifest


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--from-window-cache", type=Path,
                   help="Cluster saved descriptors from an exact index.sqlite, namespace or cache root; no flow extraction during clustering.")
    p.add_argument("--allow-partial-cache", action="store_true",
                   help="Explicitly allow clustering only currently indexed training windows from an unfinished source cache.")
    p.add_argument("--viewer-examples", type=int, default=12,
                   help="Selected examples per cluster, including medoid (default 12 = medoid + top 11); 0 skips viewer preparation.")
    p.add_argument("--prepare-viewer-only", action="store_true",
                   help="Prepare/retry selected examples for an existing --output-dir bank without reclustering; requires --from-window-cache.")
    p.add_argument("--viewer-descriptor-tolerance", type=float, default=1e-4,
                   help="Absolute/relative tolerance for checking reconstructed selected examples against source cached descriptors.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "concept_clusters/temporal_clusters_flow_only_v2",
    )
    p.add_argument("--window-len", type=int, default=32)
    p.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride between contiguous-window starts, in input frame indices.",
    )
    p.add_argument("--max-windows", type=int)
    p.add_argument(
        "--resize-to", type=int, nargs=2, default=[224, 384], metavar=("H", "W")
    )
    p.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("H", "W"),
        help="Explicit spatial cell size on resized canvas; does not load a backbone.",
    )
    gap = p.add_mutually_exclusive_group()
    gap.add_argument(
        "--flow-gap",
        type=int,
        default=None,
        help="Frame index gap, independent of backbone stride. Default: 6.",
    )
    gap.add_argument(
        "--flow-gaps",
        type=int,
        nargs="+",
        default=None,
        help="Optional multi-gap rate averaging, e.g. 4 6 8; every pair shares the target frame.",
    )
    p.add_argument(
        "--flow-estimator",
        choices=("farneback", "farneback-cuda", "dis-fast", "dis-ultrafast", "raft-small", "raft-large"),
        default="farneback",
    )
    p.add_argument(
        "--flow-cache-root", type=Path, default=DEFAULT_ROOT / "cached_temporal_flow"
    )
    p.add_argument(
        "--flow-storage-dtype", choices=("float16", "float32"), default="float32"
    )
    p.add_argument(
        "--cache-read-only",
        action="store_true",
        help="Require compatible flow cache entries; feature caches are never accessed.",
    )
    device = p.add_mutually_exclusive_group()
    device.add_argument(
        "--gpu", type=int, help="RAFT device index; CPU Farneback/DIS do not use it."
    )
    device.add_argument(
        "--device", help="RAFT device; default cpu. No effect on CPU flow."
    )
    p.add_argument("--flow-device", help="Optional RAFT device override.")
    p.add_argument("--raft-weights", type=Path)
    p.add_argument("--raft-updates", type=int, default=12)
    p.add_argument(
        "--camera-motion", choices=("median", "affine", "none"), default="median"
    )
    p.add_argument(
        "--camera-ransac-threshold",
        type=float,
        default=1.5,
        help="RANSAC displacement threshold, canvas pixels.",
    )
    p.add_argument(
        "--flow-static-threshold",
        type=float,
        default=0.05,
        help="Residual velocity threshold in canvas pixels/frame. Old v1 used pixels per interval.",
    )
    p.add_argument(
        "--static-patch-fraction",
        type=float,
        default=0.9,
        help="Static-pixel fraction required to route a valid cell to the static reference.",
    )
    p.add_argument("--min-flow-valid-fraction", type=float, default=0.2)
    p.add_argument(
        "--min-valid-gap-fraction",
        type=float,
        default=0.5,
        help="Minimum fraction of requested gaps with usable flow per cell.",
    )
    p.add_argument("--flow-fb-absolute", type=float, default=1.0)
    p.add_argument("--flow-fb-relative", type=float, default=0.05)
    p.add_argument(
        "--flow-metric",
        choices=("circular-emd", "l1", "compact"),
        default="circular-emd",
    )
    p.add_argument("--magnitude-transform", choices=("log", "clip"), default="log")
    p.add_argument("--magnitude-percentile", type=float, default=95.0)
    p.add_argument("--flow-weight", type=float, default=1.0)
    p.add_argument("--magnitude-weight", type=float, default=1.0)
    p.add_argument(
        "--variation-weight",
        type=float,
        default=0.25,
        help="Cross-gap rate std term; identically zero with one gap.",
    )
    p.add_argument(
        "--global-weight",
        type=float,
        default=0.0,
        help="Optional camera-motion velocity term; camera compensation remains active at weight 0.",
    )
    p.add_argument(
        "--clustering",
        choices=("average-linkage", "finch", "kmedoids"),
        default="average-linkage",
    )
    p.add_argument(
        "--num-prototypes",
        "--num-clusters",
        dest="num_prototypes",
        type=int,
        default=128,
        help="Moving reference count for average-linkage/kmedoids; up to two observed static/invalid references are added. FINCH ignores this.",
    )
    p.add_argument(
        "--finch-level",
        type=int,
        default=0,
        help="FINCH hierarchy level; default 0 is finest. Requires existing clustering_common.py.",
    )
    p.add_argument("--min-clusters", type=int, default=16)
    p.add_argument("--max-clusters", type=int, default=1024)
    p.add_argument(
        "--finch-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine FINCH medoids with kmedoids at the discovered count.",
    )
    p.add_argument(
        "--linkage-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally refine average-linkage representatives with kmedoids.",
    )
    p.add_argument("--fit-sample-size", type=int, default=4096)
    p.add_argument(
        "--fit-max-per-video",
        type=int,
        default=128,
        help="Moving fit cells per video; 0 disables cap.",
    )
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--medoid-candidates", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--diagnostic-sample-size",
        type=int,
        default=1536,
        help="Role-balanced silhouette audit limit; discovery silhouette also uses fit sample.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Move previous output to timestamped backup; caches are create-only.",
    )
    # Accept common old command-line options without resurrecting feature work.
    p.add_argument(
        "--change-direction-weight",
        type=float,
        default=0.0,
        help="Removed feature term: only 0 is accepted.",
    )
    p.add_argument(
        "--reference-checkpoint", type=Path, default=None, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--feature-cache-root",
        type=Path,
        default=DEFAULT_ROOT / "cached_temporal_feats",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--feature-storage-dtype",
        choices=("float16", "float32"),
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--feature-lag", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--cache-slices", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--change-static-threshold", type=float, default=None, help=argparse.SUPPRESS
    )
    p.add_argument("--warp-features", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--flow-only", action="store_true", help=argparse.SUPPRESS)
    return p


def parse_args(argv=None):
    p = parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = p.parse_args(argv)
    args.window_source = None
    explicit = {v.split("=", 1)[0] for v in argv if v.startswith("--")}
    if args.viewer_examples < 0 or not math.isfinite(args.viewer_descriptor_tolerance) or args.viewer_descriptor_tolerance < 0:
        p.error("--viewer-examples and --viewer-descriptor-tolerance must be nonnegative and finite.")
    if args.prepare_viewer_only and args.from_window_cache is None:
        p.error("--prepare-viewer-only requires --from-window-cache.")
    if args.from_window_cache is not None:
        try:
            inherit_window_settings(args, explicit, p)
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            p.error(f"Invalid window-cache source: {exc}")
    elif args.flow_estimator == "farneback-cuda":
        p.error("farneback-cuda is inherited from --from-window-cache; direct RGB mode supports the original estimators.")
    args.flow_gaps = sorted(
        set(args.flow_gaps or [args.flow_gap if args.flow_gap is not None else 6])
    )
    for name in (
        "window_len",
        "stride",
        "raft_updates",
        "num_prototypes",
        "fit_sample_size",
        "iterations",
        "restarts",
        "medoid_candidates",
        "batch_size",
        "threads",
        "min_clusters",
        "diagnostic_sample_size",
    ):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_','-')} must be positive.")
    if args.diagnostic_sample_size < 3:
        p.error("--diagnostic-sample-size must be >=3.")
    if min(args.resize_to + args.patch_size) < 1 or any(
        h % p_ for h, p_ in zip(args.resize_to, args.patch_size)
    ):
        p.error(
            "Resize dimensions must be positive multiples of positive --patch-size values."
        )
    if min(args.flow_gaps) < 1 or max(args.flow_gaps) >= args.window_len:
        p.error("Every flow gap must satisfy 1 <= gap < window-len.")
    if args.max_windows is not None and args.max_windows < 1:
        p.error("--max-windows must be positive.")
    if (
        args.fit_max_per_video < 0
        or args.min_clusters < 2
        or args.max_clusters < 0
        or args.max_clusters
        and args.max_clusters < args.min_clusters
    ):
        p.error("Invalid cluster bounds / video sampling cap.")
    if args.finch_level is not None and args.finch_level < 0:
        p.error("--finch-level must be nonnegative.")
    for name in (
        "min_flow_valid_fraction",
        "min_valid_gap_fraction",
        "static_patch_fraction",
    ):
        if not math.isfinite(getattr(args, name)) or not 0 < getattr(args, name) <= 1:
            p.error(f"--{name.replace('_','-')} must be in (0,1].")
    if (
        not math.isfinite(args.magnitude_percentile)
        or not 0 < args.magnitude_percentile <= 100
    ):
        p.error("Magnitude percentile must be in (0,100].")
    for name in (
        "flow_static_threshold",
        "flow_fb_absolute",
        "flow_fb_relative",
        "camera_ransac_threshold",
        "flow_weight",
        "magnitude_weight",
        "variation_weight",
        "global_weight",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_','-')} must be finite and nonnegative.")
    if not any((args.flow_weight, args.magnitude_weight, args.global_weight)) and not (
        args.variation_weight and len(args.flow_gaps) > 1
    ):
        p.error("At least one effective flow distance term must have positive weight.")
    if args.change_direction_weight != 0:
        p.error(
            "Feature changes are removed. Use --change-direction-weight 0 (default)."
        )
    if args.warp_features:
        p.error("--warp-features is unavailable in this flow-only script.")
    if args.gpu is not None and args.gpu < 0:
        p.error("--gpu must be nonnegative.")
    if args.seed < 0:
        p.error("--seed must be nonnegative.")
    legacy = [
        name
        for name in (
            "reference_checkpoint",
            "feature_storage_dtype",
            "feature_lag",
            "cache_slices",
            "change_static_threshold",
        )
        if getattr(args, name) is not None
    ]
    if legacy:
            print(
            "NOTE: flow-only mode ignores legacy options: " + ", ".join(legacy),
            file=sys.stderr,
            )
    if args.clustering == "finch":
        if args.num_prototypes != p.get_default("num_prototypes"):
            print("NOTE: --num-prototypes is ignored with FINCH.", file=sys.stderr)
        # Preserve the shared engine's interpretation of the lower-bound count.
        args.num_prototypes = args.min_clusters
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        import cv2

        cv2.setNumThreads(args.threads)
        cv2.setRNGSeed(args.seed % (2**31 - 1))
        try:
            from threadpoolctl import threadpool_limits

            context = threadpool_limits(limits=args.threads)
        except ImportError:
            context = nullcontext()
        with context:
            build(args)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        ImportError,
        sqlite3.Error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
