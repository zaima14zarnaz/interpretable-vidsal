#!/usr/bin/env python3
"""Cache compact optical-flow training inputs against an existing frozen bank.

Example:
  python cache_optical_flow.py --gpu 0 --bank-dir /path/to/bank \
      --dataset-root /path/to/dhf1k --cache-root /path/to/cache \
      --splits train val --augmentations original hflip --stride 1

--gpu N selects CUDA Farneback for a Farneback bank, or CUDA for a RAFT bank.
CUDA Farneback requires CUDA-enabled OpenCV; there is no CPU fallback.
Without --gpu, --farneback-backend cuda --cuda-device N is also supported.

Only compact windows, SQLite indexing, manifests and frozen references are
written. Raw forward/backward flow and dense processed pairs stay in memory
for the current window and are then released. Existing compatible raw flow
may be read, but raw_pairs/ and pairs/ are never created, changed or deleted.
Flow is quantized to the bank dtype before descriptors are computed, preserving
numeric behavior. Original/hflip views share one flow estimate per pair.

This is a training-cache builder: --bank-dir is required and no clustering is
fitted. Flow/descriptor helpers from the newly supplied source are included;
--clustering-script is optional. All bank normalization/thresholds stay frozen.
The bank must be a complete fixed_stage4_temporal_clusters_v2_flow_only bank.
CUDA overrides use separate provenance and recompute labels against that bank.

For arbitrary trainer starts use --stride 1 and the trainer's --window-len.
Point the trainer to the printed index.sqlite with load_dense=False. Dense
loading is unavailable for newly generated compact windows.

Resume this compact build by rerunning the command. To continue an older dense
cache without rebuilding its completed windows, provide --resume-index with
its exact index.sqlite. Bank, geometry, flow and descriptor contracts must match;
source hashes for the current run are recorded separately. Completed windows
are verified against decoded RGB content, without loading or estimating flow.
Keep the old job stopped while resuming its index. No existing files are deleted.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import uuid

import cv2
import numpy as np

SCHEMA = "optical_flow_training_inputs_v1"
PAIR_SCHEMA = "target_aligned_flow_training_pair_v1"
BANK_FORMAT = "fixed_stage4_temporal_clusters_v2_flow_only"
ROLES = ("invalid", "static", "moving")


# Flow/descriptor implementation retained from the newly attached source.
import re
from scipy.spatial.distance import cdist
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

FLOW_CACHE_SCHEMA = "bidirectional_selected_rgb_flow_npz_v2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

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

def natural_key(value):
    return tuple(
        (1, int(p)) if p.isdigit() else (0, p.casefold())
        for p in re.split(r"(\d+)", str(value))
    )

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

_FLOW_CORE = SimpleNamespace(
    FORMAT_VERSION=BANK_FORMAT, FLOW_CACHE_SCHEMA=FLOW_CACHE_SCHEMA,
    IMAGE_SUFFIXES=IMAGE_SUFFIXES, tqdm=tqdm,
    FlowEstimator=FlowEstimator,
    circular_hist_distance=circular_hist_distance,
    compensate_camera=compensate_camera,
    descriptor_config=descriptor_config,
    natural_key=natural_key,
    read_canvas=read_canvas,
    sha256_file=sha256_file,
    soft_histogram=soft_histogram,
    target_flow_validity=target_flow_validity,
    temporal_distance=temporal_distance,
)



def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", delete=False
    ) as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npz_once(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
        np.savez_compressed(f, **arrays)
    created = False
    try:
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    return created


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def readonly(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load_core(path=None):
    if path is None:
        return _FLOW_CORE, Path(__file__).resolve()
    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Clustering helper not found: {selected}")
    spec = importlib.util.spec_from_file_location("_fixed_flow_bank_core", selected)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    if getattr(core, "FORMAT_VERSION", None) != BANK_FORMAT:
        raise ValueError("The clustering helper must be the v2 flow-only bank builder.")
    return core, selected


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


def configure_flow_backend(bank, options):
    """Keep frozen bank identity, but fingerprint an explicitly changed estimator."""
    bank.bank_flow_config = bank.flow_config
    bank.cuda_estimator = None
    bank.flow_backend_changed = False
    requested = options.farneback_backend
    recorded_kind = bank.estimator_config["estimator"]
    if options.gpu is not None:
        if recorded_kind in ("farneback", "farneback-cuda"):
            requested = "cuda"
        elif recorded_kind.startswith("raft-"):
            bank.args.device = f"cuda:{options.gpu}"
            bank.args.flow_device = f"cuda:{options.gpu}"
        else:
            raise ValueError("--gpu requires a Farneback or RAFT bank; DIS is CPU-only.")
    if requested != "cuda" and recorded_kind != "farneback-cuda":
        return
    if recorded_kind not in ("farneback", "farneback-cuda"):
        raise ValueError("--farneback-backend cuda requires a Farneback bank (not DIS/RAFT).")
    bank.cuda_estimator = CudaFarnebackEstimator(bank.estimator_config, options.cuda_device)
    bank.estimator_config = bank.cuda_estimator.config
    bank.flow_config = dict(bank.flow_config, estimator=bank.estimator_config)
    bank.raw_namespace = digest(bank.flow_config)
    bank.flow_backend_changed = bank.flow_config != bank.bank_flow_config
    print(
        f"CUDA Farneback selected on device {options.cuda_device}; separate raw namespace "
        f"{bank.raw_namespace}; frozen bank assignments will be recomputed"
        if bank.flow_backend_changed else f"Using bank's CUDA Farneback on device {options.cuda_device}",
        flush=True,
    )


class Bank:
    def __init__(self, path, core, options):
        self.directory = Path(path).expanduser().resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        m = self.manifest
        if m.get("status") != "complete" or m.get("format_version") != BANK_FORMAT:
            raise ValueError("A complete v2 flow-only temporal bank is required.")
        self.reference_path = self.directory / m.get(
            "references", "cluster_references.npz"
        )
        self.reference_sha = file_sha(self.reference_path)
        if m.get("references_sha256") != self.reference_sha:
            raise ValueError(
                "Reference NPZ fingerprint does not match the bank manifest."
            )
        self.references = load_npz(self.reference_path)
        self.count = int(m["num_prototypes"])
        if (
            options.expected_clusters is not None
            and self.count != options.expected_clusters
        ):
            raise ValueError(
                f"Expected {options.expected_clusters} clusters, found {self.count}."
            )
        self.ids = self.references["prototype_ids"].astype(np.int64)
        if self.ids.shape != (self.count,) or not np.array_equal(
            self.ids, np.arange(self.count)
        ):
            raise ValueError("Expected contiguous frozen prototype IDs 0..K-1.")
        self.desc = m["descriptor"]
        self.norm = m["normalization"]
        self.flow_config = m["flow_provenance"]
        self.estimator_config = self.flow_config["estimator"]
        if self.flow_config["schema"] != core.FLOW_CACHE_SCHEMA:
            raise ValueError("The helper raw-flow cache schema differs from this bank.")
        self.raw_namespace = digest(self.flow_config)
        if self.raw_namespace != m["flow_cache_namespace"]:
            raise ValueError("Recorded flow namespace/config mismatch.")
        recorded_cv = self.estimator_config.get("opencv_version")
        if recorded_cv and recorded_cv != cv2.__version__:
            raise ValueError(
                f"Bank uses OpenCV {recorded_cv}; this environment has {cv2.__version__}. Install the recorded version to avoid preprocessing drift."
            )
        d = self.desc
        self.gaps = [int(g) for g in d["flow_frame_gaps"]]
        self.canvas = tuple(int(x) for x in self.estimator_config["canvas"])
        self.patch = tuple(int(x) for x in d["patch_size"])
        self.bank_window_len = int(m["window_len"])
        self.window_len = options.window_len or self.bank_window_len
        if (
            not self.gaps
            or min(self.gaps) < 1
            or max(self.gaps) >= self.bank_window_len
        ):
            raise ValueError("Invalid bank flow gaps/window length.")
        if max(self.gaps) >= self.window_len:
            raise ValueError(
                "The model window must contain every bank flow pair: "
                f"--window-len must be at least {max(self.gaps) + 1}. "
                "The fixed bank gaps cannot be shortened."
            )
        if any(h <= 0 or p <= 0 or h % p for h, p in zip(self.canvas, self.patch)):
            raise ValueError("Bank canvas does not divide into its patch grid.")
        self.grid = tuple(h // p for h, p in zip(self.canvas, self.patch))
        if self.grid != (int(m["stage4_grid"]["H"]), int(m["stage4_grid"]["W"])):
            raise ValueError("Bank grid/patch geometry mismatch.")
        self.metric = d["histogram_metric"]
        self.weights = self.references["weights"].astype(np.float64)
        names = ("flow", "cross_gap_variation", "flow_magnitude", "global_motion")
        if self.weights.shape != (4,) or not np.allclose(
            self.weights, [m["weights"][n] for n in names]
        ):
            raise ValueError("Reference/manifest distance weights mismatch.")
        for name in (
            "flow_magnitude_scale",
            "variation_std_scale",
            "global_motion_scale",
        ):
            if not math.isfinite(float(self.norm[name])) or float(self.norm[name]) <= 0:
                raise ValueError("Invalid frozen training normalization scale: " + name)
        if not self.norm.get("clipped") or self.norm["magnitude_transform"] not in (
            "log",
            "clip",
        ):
            raise ValueError(
                "Unsupported normalization recipe; use the matching flow-only builder."
            )
        self.args = SimpleNamespace(
            resize_to=list(self.canvas),
            patch_size=list(self.patch),
            flow_gaps=self.gaps,
            flow_estimator=self.estimator_config["estimator"],
            camera_motion=d["camera_motion"],
            seed=int(m["seed"]),
            camera_ransac_threshold=d["camera_ransac_threshold"],
            flow_static_threshold=d["flow_static_threshold"],
            static_patch_fraction=d["static_patch_fraction"],
            min_flow_valid_fraction=d["min_flow_valid_fraction"],
            min_valid_gap_fraction=d["min_valid_gap_fraction"],
            flow_fb_absolute=d["forward_backward_absolute"],
            flow_fb_relative=d["forward_backward_relative"],
            flow_metric=self.metric,
            raft_weights=options.raft_weights,
            raft_updates=int(self.estimator_config.get("iterations", 12)),
            flow_device=options.flow_device,
            device=options.device,
            gpu=None,
        )
        if core.descriptor_config(self.args) != d:
            raise ValueError(
                "Helper descriptor contract differs from the bank. Use the exact flow-only builder that generated it."
            )
        self.identity = digest(
            dict(
                format=m["format_version"],
                reference_sha256=self.reference_sha,
                descriptor=d,
                normalization=self.norm,
                weights=m["weights"],
                flow=self.flow_config,
                window_len=self.bank_window_len,
                seed=m["seed"],
                num_prototypes=self.count,
            )
        )
        self.db_path = self.directory / m.get("database", "temporal_clusters.sqlite")
        self.db = readonly(self.db_path) if self.db_path.is_file() else None
        self.components = []
        if self.weights[0] > 0:
            self.components.append(
                ("compact", 3) if self.metric == "compact" else ("flow_hist", 9)
            )
        if self.weights[2] > 0:
            self.components.append(("normalized_flow_magnitude", 1))
        if self.weights[1] > 0:
            self.components.append(("normalized_cross_gap_std", 1))
        if self.weights[3] > 0:
            self.components.append(("global_motion", 2))
        self.components.extend([("flow_valid_fraction", 1), ("flow_valid", 1)])

    def close(self):
        if self.db is not None:
            self.db.close()

    def original_training_labels(self, video, start, target, pair_keys):
        if self.db is None or getattr(self, "flow_backend_changed", False):
            return None
        # The model may use another context length. The same target and
        # verified RGB flow pairs still define exactly the same descriptor.
        bank_start = target - self.bank_window_len + 1
        if bank_start < 0:
            return None
        row = self.db.execute(
            "SELECT id,global_model_json FROM frames WHERE dataset=? AND video=? AND window_start=? AND target_index=?",
            ("train", video, bank_start, target),
        ).fetchone()
        if row is None:
            return None
        model = json.loads(row["global_model_json"])
        if model.get("pair_keys") != pair_keys:
            return None
        rows = self.db.execute(
            "SELECT row,col,prototype_id,distance FROM patches WHERE frame_id=? ORDER BY row,col",
            (row["id"],),
        ).fetchall()
        if len(rows) != self.grid[0] * self.grid[1] or any(
            r["prototype_id"] is None for r in rows
        ):
            raise ValueError("Incomplete saved training assignments in reference bank.")
        return np.array([r["prototype_id"] for r in rows], np.int32), np.array(
            [r["distance"] for r in rows], np.float32
        )


class RawFlowCache:
    """Read existing raw entries or estimate in memory; never write dense flow."""

    def __init__(
        self, bank, core, root, reuse_dirs=(), read_only=False, copy_reused=False
    ):
        self.bank = bank
        self.core = core
        self.estimator = getattr(bank, "cuda_estimator", None)
        self.directory = (
            Path(root) / "raw_pairs" / core.FLOW_CACHE_SCHEMA / bank.raw_namespace
        )
        if copy_reused:
            raise ValueError("This compact-only builder does not support --copy-reused-flow.")
        manifest = self.directory / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != bank.flow_config:
            raise ValueError("Raw output namespace has incompatible metadata.")
        self.sources = [self.directory]
        candidates = []
        if bank.manifest.get("flow_cache_directory"):
            candidates.append((Path(bank.manifest["flow_cache_directory"]), False))
        candidates.extend((Path(p).expanduser(), True) for p in reuse_dirs)
        for directory, explicit in candidates:
            directory = directory.expanduser().resolve()
            if not directory.is_dir():
                if explicit:
                    raise FileNotFoundError(f"Requested raw cache missing: {directory}")
                continue
            if directory in self.sources:
                continue
            if (
                not (directory / "manifest.json").is_file()
                or json.loads((directory / "manifest.json").read_text())
                != bank.flow_config
            ):
                if explicit:
                    raise ValueError(f"Raw cache does not match the bank: {directory}")
                continue
            self.sources.append(directory)
        self.dtype = np.dtype(bank.flow_config["storage_dtype"])
        self.read_only = read_only
        self.copy_reused = copy_reused
        self.stats = dict(hits=0, reused=0, misses=0, estimator_calls=0, writes=0)
        self.active = {}

    def begin_window(self):
        self.active.clear()

    def _read(self, path, key):
        p = load_npz(path)
        if (
            str(p["namespace"]) != self.bank.raw_namespace
            or str(p["pair_sha256"]) != key
        ):
            raise ValueError(f"Raw flow identity mismatch; file preserved: {path}")
        fields = []
        for name in ("forward", "backward"):
            value = p[name]
            if (
                value.shape != (*self.bank.canvas, 2)
                or value.dtype != self.dtype
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid raw cached flow; file preserved: {path}")
            fields.append(value.astype(np.float32))
        return fields

    def get(self, previous, current):
        if (
            previous.shape != (*self.bank.canvas, 3)
            or previous.dtype != np.uint8
            or current.shape != previous.shape
            or current.dtype != np.uint8
        ):
            raise ValueError("Raw flow requires the bank canvas as uint8 RGB.")
        h = hashlib.sha256(str(previous.shape).encode())
        h.update(np.ascontiguousarray(previous).tobytes())
        h.update(np.ascontiguousarray(current).tobytes())
        key = h.hexdigest()
        if key in self.active:
            return self.active[key]
        relative = Path("entries") / key[:2] / (key + ".npz")
        destination = self.directory / relative
        for i, directory in enumerate(self.sources):
            path = directory / relative
            if not path.is_file():
                continue
            fields = self._read(path, key)
            self.stats["hits"] += 1
            if i:
                self.stats["reused"] += 1
            result = (*fields, key, path)
            self.active[key] = result
            return result
        self.stats["misses"] += 1
        if self.read_only:
            raise FileNotFoundError(
                f"No matching raw flow entry for pair {key}; read-only mode forbids estimation."
            )
        if self.estimator is None:
            self.estimator = self.core.FlowEstimator(self.bank.args)
            if self.estimator.config != self.bank.estimator_config:
                raise ValueError(
                    "Flow estimator settings/weights/versions differ from the bank; no flow entry written."
                )
        self.stats["estimator_calls"] += 1
        f, b = self.estimator.estimate(previous, current)
        f = f.astype(self.dtype)
        b = b.astype(self.dtype)
        if any(
            v.shape != (*self.bank.canvas, 2) or not np.isfinite(v).all()
            for v in (f, b)
        ):
            raise ValueError("Invalid estimated flow; no cache entry written.")
        result = (f.astype(np.float32), b.astype(np.float32), key, None)
        self.active[key] = result
        return result


def flip_vectors(field):
    result = field[:, ::-1].copy()
    result[..., 0] *= -1
    return result


class ProcessedPairs:
    def __init__(self, bank, core, directory):
        self.bank = bank
        self.core = core
        self.directory = Path(directory) / "pairs"
        self.stats = dict(hits=0, writes=0)
        self.active = {}

    def begin_window(self):
        self.active.clear()

    def get(self, forward, backward, raw_key, raw_path, gap, augmentation):
        key = digest(
            dict(
                schema=PAIR_SCHEMA,
                bank=self.bank.identity,
                raw_pair=raw_key,
                gap=gap,
                augmentation=augmentation,
            )
        )
        path = self.directory / key[:2] / (key + ".npz")
        if key in self.active:
            payload = self.active[key]
            self.stats["hits"] += 1
        else:
            if augmentation == "original":
                valid = self.core.target_flow_validity(
                    forward,
                    backward,
                    self.bank.args.flow_fb_absolute,
                    self.bank.args.flow_fb_relative,
                )
                residual, global_flow, camera = self.core.compensate_camera(
                    -backward,
                    valid,
                    self.bank.args.camera_motion,
                    self.bank.args.seed,
                    self.bank.args.camera_ransac_threshold,
                )
                residual = residual / gap
                global_flow = global_flow / gap
            elif augmentation == "hflip":
                original, _ = self.get(
                    forward, backward, raw_key, raw_path, gap, "original"
                )
                residual = flip_vectors(original["residual_velocity"])
                global_flow = flip_vectors(original["global_velocity"])
                valid = np.ascontiguousarray(original["valid"][:, ::-1])
                source_meta = json.loads(str(original["metadata"]))
                camera = dict(
                    used="hflip_of_recorded_full_canvas_fit",
                    source_model=source_meta["camera_model"],
                )
            else:
                raise ValueError("Unsupported augmentation: " + augmentation)
            meta = dict(
                schema=PAIR_SCHEMA,
                key=key,
                bank_identity=self.bank.identity,
                raw_pair_key=raw_key,
                raw_path=None,
                gap=int(gap),
                augmentation=augmentation,
                camera_model=camera,
                velocity_units="canvas pixels/frame",
                anchor="current frame",
                flip_policy="reverse columns and negate dx; transform camera field and validity; no refit",
            )
            payload = dict(
                residual_velocity=residual.astype(np.float32),
                global_velocity=global_flow.astype(np.float32),
                valid=valid.astype(bool),
                metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
            )
            self.active[key] = payload
        for name in ("residual_velocity", "global_velocity"):
            if (
                payload[name].shape != (*self.bank.canvas, 2)
                or not np.isfinite(payload[name]).all()
            ):
                raise ValueError(f"Invalid derived flow tensor: {path}")
        if (
            payload["valid"].shape != self.bank.canvas
            or payload["valid"].dtype != np.bool_
        ):
            raise ValueError(f"Invalid derived validity mask: {path}")
        return payload, None


def pool_pair(payload, bank, core):
    gh, gw = bank.grid
    ph, pw = bank.patch
    n = gh * gw
    residual = payload["residual_velocity"]
    global_flow = payload["global_velocity"]
    valid = payload["valid"]
    out = dict(
        flow_hist=np.zeros((n, 9), np.float32),
        flow_magnitude=np.zeros(n, np.float32),
        flow_mean=np.zeros((n, 2), np.float32),
        flow_spread=np.zeros(n, np.float32),
        global_motion_raw=np.zeros((n, 2), np.float32),
        flow_valid_fraction=np.zeros(n, np.float32),
    )
    for row in range(gh):
        for col in range(gw):
            i = row * gw + col
            sl = np.s_[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw]
            mask = valid[sl]
            out["flow_valid_fraction"][i] = mask.mean()
            if not mask.any():
                continue
            vectors = residual[sl][mask]
            out["flow_hist"][i] = core.soft_histogram(
                vectors, bank.args.flow_static_threshold
            )
            out["flow_magnitude"][i] = np.linalg.norm(vectors, axis=-1).mean()
            out["flow_mean"][i] = vectors.mean(axis=0)
            out["flow_spread"][i] = np.sqrt(
                np.square(vectors - out["flow_mean"][i]).sum(axis=-1).mean()
            )
            out["global_motion_raw"][i] = global_flow[sl][mask].mean(axis=0)
    return out


def merge_pairs(summaries, bank):
    usable = np.stack(
        [
            s["flow_valid_fraction"] >= bank.args.min_flow_valid_fraction
            for s in summaries
        ]
    )
    count = usable.sum(axis=0)
    good = count >= math.ceil(len(summaries) * bank.args.min_valid_gap_fraction)
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
    merged.update(
        flow_variance=variance.astype(np.float32),
        flow_valid=good,
        role=np.where(
            ~good,
            0,
            np.where(
                merged["flow_hist"][:, -1] >= bank.args.static_patch_fraction, 1, 2
            ),
        ).astype(np.int8),
    )
    # Bank extraction writes float32 memmaps before normalization. Reproduce
    # that rounding boundary rather than normalizing temporary float64 means.
    for name in merged:
        if name not in ("flow_valid", "role"):
            merged[name] = np.asarray(merged[name], np.float32)
    variance = merged["flow_variance"]
    # All statistics are normalized using FROZEN TRAINING values.
    norm = bank.norm

    def transform(value, scale):
        value = np.maximum(value, 0)
        result = (
            value / scale
            if norm["magnitude_transform"] == "clip"
            else np.log1p(value) / np.log1p(scale)
        )
        return np.clip(result, 0, 1)

    magnitude = np.stack(
        [
            transform(merged["flow_magnitude"], norm["flow_magnitude_scale"]),
            transform(np.sqrt(variance), norm["variation_std_scale"]),
        ],
        axis=-1,
    ).astype(np.float32)
    global_motion = merged["global_motion_raw"] / norm["global_motion_scale"]
    global_motion /= np.maximum(1, np.linalg.norm(global_motion, axis=1, keepdims=True))
    compact = np.column_stack(
        [
            np.clip(merged["flow_mean"] / norm["flow_magnitude_scale"], -1, 1),
            transform(merged["flow_spread"], norm["flow_magnitude_scale"]),
        ]
    ).astype(np.float32)
    merged.update(
        magnitude=magnitude,
        compact=compact,
        global_motion=global_motion.astype(np.float32),
    )
    parts = []
    for name, _ in bank.components:
        if name == "normalized_flow_magnitude":
            value = magnitude[:, :1]
        elif name == "normalized_cross_gap_std":
            value = magnitude[:, 1:2]
        elif name in ("flow_valid", "flow_valid_fraction"):
            value = np.asarray(merged[name], np.float32)[:, None]
        else:
            value = merged[name]
        parts.append(value)
    encoder_inputs = np.concatenate(parts, axis=-1).astype(np.float32)
    encoder_inputs[~good, :-2] = 0
    if not np.isfinite(encoder_inputs).all():
        raise ValueError("Nonfinite encoder inputs.")
    merged["encoder_inputs"] = encoder_inputs
    return merged


def distance_inputs(arrays):
    return tuple(
        arrays[n]
        for n in ("flow_hist", "magnitude", "compact", "global_motion", "flow_valid")
    )


def assign_frozen(arrays, bank, core):
    distances = core.temporal_distance(
        distance_inputs(arrays),
        distance_inputs(bank.references),
        bank.weights,
        bank.metric,
    )
    distances[arrays["role"][:, None] != bank.references["role"][None, :]] = np.inf
    column = distances.argmin(axis=1)
    best = distances[np.arange(len(column)), column]
    valid = np.isfinite(best)
    labels = np.where(valid, bank.ids[column], -1).astype(np.int32)
    return labels, best.astype(np.float32), valid


def parse_mappings(items, convert=str):
    result = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ValueError("Expected NAME=VALUE: " + item)
        if key in result:
            raise ValueError("Duplicate mapping for " + key)
        result[key] = convert(value)
    return result


def source_windows(options, bank, core):
    roots = {
        s: (options.dataset_root / s).expanduser().resolve() for s in options.splits
    }
    custom = parse_mappings(
        options.split_root, lambda p: Path(p).expanduser().resolve()
    )
    if set(custom) - set(options.splits):
        raise ValueError("A --split-root name is not selected in --splits.")
    roots.update(custom)
    strides = {
        s: int(
            options.stride
            if options.stride is not None
            else bank.manifest["window_stride"]
        )
        for s in options.splits
    }
    custom_strides = parse_mappings(options.split_stride, int)
    if set(custom_strides) - set(options.splits):
        raise ValueError("A --split-stride name is not selected in --splits.")
    strides.update(custom_strides)
    if any(n < 1 for n in strides.values()):
        raise ValueError("Window strides must be positive.")
    for split, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(
                f"Split {split} missing: {root}. Use --split-root {split}=/actual/path."
            )
    explicit = None
    if options.window_index:
        explicit = []
        with options.window_index.open() as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if item["split"] not in roots:
                    continue
                start = int(item["window_start"])
                target = start + bank.window_len - 1
                if (
                    start < 0
                    or "target_index" in item
                    and int(item["target_index"]) != target
                ):
                    raise ValueError(
                        f"Window index line {line_no}: incompatible start/target."
                    )
                explicit.append((item["split"], str(item["video"]), start))
        if not explicit:
            raise ValueError("No selected windows in --window-index.")
    windows = []
    files = {}
    counts = {s: 0 for s in roots}
    seen = set()

    def add(split, video, start, paths):
        identity = (split, video, start)
        if identity in seen:
            return
        if options.max_windows is not None and counts[split] >= options.max_windows:
            return
        if start < 0 or start + bank.window_len > len(paths):
            raise ValueError(f"Incomplete window: {identity}")
        seen.add(identity)
        counts[split] += 1
        windows.append(identity)

    if explicit is not None:
        for split, video, start in explicit:
            # Only a video directory name is accepted, never traversal outside split.
            if video in (".", "..") or "/" in video or "\\" in video:
                raise ValueError("Invalid video directory name.")
            key = (split, video)
            if key not in files:
                folder = roots[split] / video / "images"
                if not folder.is_dir():
                    raise FileNotFoundError(f"Video images missing: {folder}")
                files[key] = sorted(
                    (
                        p
                        for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in core.IMAGE_SUFFIXES
                    ),
                    key=lambda p: core.natural_key(p.name),
                )
            add(split, video, start, files[key])
    else:
        for split, root in roots.items():
            for video in sorted(
                (p for p in root.iterdir() if p.is_dir()),
                key=lambda p: core.natural_key(p.name),
            ):
                folder = video / "images"
                if not folder.is_dir():
                    continue
                paths = sorted(
                    (
                        p
                        for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in core.IMAGE_SUFFIXES
                    ),
                    key=lambda p: core.natural_key(p.name),
                )
                if len(paths) < bank.window_len:
                    continue
                files[(split, video.name)] = paths
                for start in range(0, len(paths) - bank.window_len + 1, strides[split]):
                    add(split, video.name, start, paths)
                if (
                    options.max_windows is not None
                    and counts[split] >= options.max_windows
                ):
                    break
    if any(n == 0 for n in counts.values()):
        raise ValueError(
            f"No eligible windows for at least one requested split: {counts}"
        )
    return windows, files, roots, strides


def init_index(path, contract):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(
        """PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,started_at TEXT NOT NULL,completed_at TEXT,status TEXT NOT NULL,settings_json TEXT NOT NULL,stats_json TEXT);
    CREATE TABLE IF NOT EXISTS windows(cache_key TEXT PRIMARY KEY,split TEXT NOT NULL,video TEXT NOT NULL,window_start INTEGER NOT NULL,target_index INTEGER NOT NULL,augmentation TEXT NOT NULL,path TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS window_lookup ON windows(split,video,window_start,augmentation,updated_at);
    CREATE TABLE IF NOT EXISTS run_windows(run_id TEXT NOT NULL REFERENCES runs(id),cache_key TEXT NOT NULL REFERENCES windows(cache_key),PRIMARY KEY(run_id,cache_key));"""
    )
    row = con.execute(
        "SELECT value FROM metadata WHERE key=?", ("contract",)
    ).fetchone()
    if row and json.loads(row[0]) != contract:
        con.close()
        raise ValueError("Cache index contract mismatch.")
    con.execute(
        "INSERT OR IGNORE INTO metadata VALUES (?,?)",
        ("contract", json.dumps(contract, sort_keys=True, allow_nan=False)),
    )
    con.commit()
    return con


def validate_location(root, bank, split_roots):
    root = Path(root).expanduser().resolve()
    if (
        root == bank.directory
        or bank.directory in root.parents
        or root in bank.directory.parents
    ):
        raise ValueError("Cache output must not overlap the fixed bank directory.")
    for split in split_roots.values():
        if root == split or split in root.parents or root in split.parents:
            raise ValueError(
                "Cache output must be outside split image trees and must not replace their ancestors."
            )
    return root


def prepare_canvases(paths, start, bank, core):
    target = start + bank.window_len - 1
    current = core.read_canvas(paths[target], bank.canvas)
    return [
        (gap, core.read_canvas(paths[target - gap], bank.canvas), current)
        for gap in bank.gaps
    ]


def canvas_pair_key(previous, current):
    h = hashlib.sha256(str(previous.shape).encode())
    h.update(np.ascontiguousarray(previous).tobytes())
    h.update(np.ascontiguousarray(current).tobytes())
    return h.hexdigest()


def prepare_pairs(paths, start, bank, core, raw, canvases=None):
    if canvases is None:
        canvases = prepare_canvases(paths, start, bank, core)
    return [(gap, *raw.get(previous, current)) for gap, previous, current in canvases]


def process_window(
    split,
    video,
    start,
    paths,
    augmentation,
    bank,
    core,
    raw,
    processed,
    directory,
    contract,
    prepared=None,
    canvases=None,
):
    target = start + bank.window_len - 1
    if prepared is not None:
        keys = [p[3] for p in prepared]
    else:
        if canvases is None:
            canvases = prepare_canvases(paths, start, bank, core)
        keys = [canvas_pair_key(previous, current) for _, previous, current in canvases]
    key = digest(
        dict(
            namespace=contract["namespace"],
            split=split,
            video=video,
            start=start,
            target=target,
            augmentation=augmentation,
            pair_keys=keys,
        )
    )
    path = directory / "windows" / key[:2] / (key + ".npz")
    if path.is_file():
        old = load_npz(path)
        meta = json.loads(str(old["metadata"]))
        if meta["cache_key"] != key or meta["bank_identity"] != bank.identity:
            raise ValueError(f"Window cache identity mismatch; preserved: {path}")
        if (
            meta.get("namespace") != contract["namespace"]
            or meta.get("split") != split
            or meta.get("video") != video
            or meta.get("window_start") != start
            or meta.get("target_index") != target
            or meta.get("augmentation") != augmentation
            or [p["raw_pair_key"] for p in meta["pairs"]] != keys
        ):
            raise ValueError(f"Window metadata/content identity mismatch; preserved: {path}")
        n = int(np.prod(bank.grid))
        if (
            old["encoder_inputs"].shape != (n, sum(int(d) for _, d in bank.components))
            or not np.isfinite(old["encoder_inputs"]).all()
            or old["cluster_ids"].shape != (n,)
            or np.any((old["cluster_ids"] < -1) | (old["cluster_ids"] >= bank.count))
            or old["temporal_train_mask"].shape != (n,)
        ):
            raise ValueError(f"Invalid compact window arrays; preserved: {path}")
        return path, key, old, True
    if prepared is None:
        prepared = prepare_pairs(paths, start, bank, core, raw, canvases)
    summaries = []
    pairs = []
    for gap, forward, backward, raw_key, raw_path in prepared:
        payload, pair_path = processed.get(
            forward, backward, raw_key, raw_path, gap, augmentation
        )
        summaries.append(pool_pair(payload, bank, core))
        pairs.append(
            dict(
                gap=gap,
                previous_index=target - gap,
                target_index=target,
                previous_name=paths[target - gap].name,
                target_name=paths[target].name,
                raw_pair_key=raw_key,
                raw_path=None,
                processed_path=None,
            )
        )
    arrays = merge_pairs(summaries, bank)
    labels, distances, assignment_valid = assign_frozen(arrays, bank, core)
    label_source = "frozen_role_routed_nearest_medoid"
    if split == "train" and augmentation == "original":
        original = bank.original_training_labels(video, start, target, keys)
        if original is not None:
            labels, distances = original
            assignment_valid = np.isfinite(distances)
            label_source = "exact_original_bank_training_assignments"
    arrays.update(
        cluster_ids=labels,
        cluster_distances=distances,
        assignment_valid=assignment_valid,
        temporal_train_mask=assignment_valid & arrays["flow_valid"],
    )
    meta = dict(
        schema=SCHEMA,
        cache_key=key,
        bank_identity=bank.identity,
        namespace=contract["namespace"],
        split=split,
        video=video,
        window_start=start,
        window_len=bank.window_len,
        target_index=target,
        target_name=paths[target].name,
        flow_frame_gaps=bank.gaps,
        augmentation=augmentation,
        grid_hw=list(bank.grid),
        patch_size=list(bank.patch),
        canvas_hw=list(bank.canvas),
        row_major_patches=True,
        pairs=pairs,
        assignment_source=label_source,
        encoder_input_components=bank.components,
        flow_units="canvas pixels/frame",
        assignment_invalid_label=-1,
        learned_embeddings_cached=False,
        dense_saved=False,
    )
    if not save_npz_once(
        path,
        metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
        **arrays,
    ):
        old = load_npz(path)
        if json.loads(str(old["metadata"]))["cache_key"] != key:
            raise ValueError("Concurrent cache identity mismatch.")
        winner = load_npz(path)
        return path, key, winner, True
    return path, key, dict(metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)), **arrays), False


def build(options):
    core, core_path = load_core(options.clustering_script)
    bank = Bank(options.bank_dir, core, options)
    con = None
    try:
        configure_flow_backend(bank, options)
        windows, files, roots, strides = source_windows(options, bank, core)
        cache_root = validate_location(options.cache_root, bank, roots)
        recipe = dict(
            schema=SCHEMA,
            bank_identity=bank.identity,
            bank_reference_sha256=bank.reference_sha,
            bank_format=BANK_FORMAT,
            core_sha256=file_sha(core_path),
            implementation_sha256=file_sha(__file__),
            window_len=bank.window_len,
            bank_window_len=bank.bank_window_len,
            flow_frame_gaps=bank.gaps,
            canvas_hw=list(bank.canvas),
            patch_size=list(bank.patch),
            grid_hw=list(bank.grid),
            num_prototypes=bank.count,
            normalization=bank.norm,
            descriptor=bank.desc,
            weights=bank.manifest["weights"],
            flow_config=bank.flow_config,
            encoder_input_components=bank.components,
            augmentation_recipe="hflip_full_lattice_negate_dx_v1",
            learned_embeddings_cached=False,
        )
        if bank.flow_backend_changed:
            recipe.update(
                bank_flow_config=bank.bank_flow_config,
                flow_backend_override="cuda_farneback_reassign_to_frozen_bank",
            )
        if options.resume_index is not None:
            index_path = options.resume_index.expanduser().resolve()
            if not index_path.is_file() or index_path.name != "index.sqlite":
                raise FileNotFoundError("--resume-index must name an existing index.sqlite.")
            if index_path.parent.parent != cache_root / SCHEMA:
                raise ValueError("--resume-index must be inside this --cache-root's training-cache directory.")
            with closing(readonly(index_path)) as previous_index:
                row = previous_index.execute(
                    "SELECT value FROM metadata WHERE key='contract'"
                ).fetchone()
            if row is None:
                raise ValueError("The resume index has no bank contract.")
            contract = json.loads(row[0])
            # Program source hashes are run provenance, not flow/descriptor
            # settings. Preserve the old contract and record current hashes below.
            excluded = {"namespace", "implementation_sha256", "core_sha256"}
            expected = {k: v for k, v in recipe.items() if k not in excluded}
            recorded = {k: v for k, v in contract.items() if k not in excluded}
            expected = json.loads(json.dumps(expected))
            if recorded != expected:
                changed = sorted(k for k in recorded.keys() | expected.keys()
                                 if recorded.get(k) != expected.get(k))
                raise ValueError("Resume bank/flow/descriptor contract differs: " + ", ".join(changed))
            directory = index_path.parent
            if directory.name != contract["namespace"]:
                raise ValueError("Resume directory and recorded namespace disagree.")
        else:
            namespace = digest(recipe)
            contract = json.loads(json.dumps(dict(namespace=namespace, **recipe)))
            directory = cache_root / SCHEMA / namespace
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            prior = json.loads(manifest_path.read_text())
            if prior.get("contract") != contract:
                raise ValueError("Existing output manifest contract mismatch.")
        else:
            atomic_json(
                manifest_path,
                dict(schema=SCHEMA, status="initialized", contract=contract),
            )
        frozen = directory / "bank_references.npz"
        if frozen.is_file():
            if file_sha(frozen) != bank.reference_sha:
                raise ValueError("Cached frozen bank snapshot is incompatible.")
        else:
            # Bytes are the immutable reference NPZ, copied without changing the bank.
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
                temp = Path(f.name)
            try:
                shutil.copyfile(bank.reference_path, temp)
                try:
                    os.link(temp, frozen)
                except FileExistsError:
                    pass
            finally:
                temp.unlink(missing_ok=True)
            if file_sha(frozen) != bank.reference_sha:
                raise ValueError("Frozen bank snapshot fingerprint mismatch.")
        raw = RawFlowCache(
            bank,
            core,
            cache_root,
            options.reuse_flow_cache_dir,
            options.read_only_flow_cache,
            options.copy_reused_flow,
        )
        processed = ProcessedPairs(bank, core, directory)
        con = init_index(directory / "index.sqlite", contract)
        run_id = uuid.uuid4().hex
        settings = dict(
            splits=options.splits,
            roots={k: str(v) for k, v in roots.items()},
            strides=strides,
            augmentations=options.augmentations,
            window_index=str(options.window_index) if options.window_index else None,
            max_windows_per_split=options.max_windows,
            farneback_backend=options.farneback_backend,
            cuda_device=options.cuda_device if bank.cuda_estimator is not None else None,
            storage_mode="compact_only",
            current_implementation_sha256=file_sha(__file__),
            current_core_sha256=file_sha(core_path),
        )
        stats = dict(
            source_windows=0,
            cache_windows=0,
            window_hits=0,
            window_writes=0,
            original_training_label_windows=0,
            unassigned_patches=0,
            supervised_patches=0,
            prototype_counts_by_split_view={},
        )
        started = now()
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?)",
            (run_id, started, None, "running", json.dumps(settings), None),
        )
        con.commit()
        output_manifest = dict(
            schema=SCHEMA,
            status="running",
            contract=contract,
            bank_directory=str(bank.directory),
            bank_manifest_sha256=file_sha(bank.manifest_path),
            source_builder=str(core_path),
            raw_cache_directory=None,
            raw_cache_sources=[str(p) for p in raw.sources if p.is_dir()],
            raw_flow_saved=False,
            dense_processed_pairs_saved=False,
            flow_backend_changed=bank.flow_backend_changed,
            flow_estimator=bank.estimator_config["estimator"],
            index="index.sqlite",
            last_run_id=run_id,
            last_run_started_at=started,
            settings=settings,
            dataset_sampling_note="Cache coverage must match trainer windows/augmentations. Stride 1 covers arbitrary complete-window starts.",
            unsupported_role_policy="cluster_ids=-1, assignment_valid=False; never invent a different-role target",
            training_encoder_note="Run your trainable temporal encoder live on encoder_inputs; cluster_ids are targets, not inputs.",
        )
        atomic_json(
            directory / "bank_contract.json",
            dict(bank_identity=bank.identity, manifest=bank.manifest),
        )
        atomic_json(manifest_path, output_manifest)
        print(
            f"Bank: {bank.count} fixed clusters | gaps {bank.gaps} ending at target | grid {bank.grid} | no backbone/encoder",
            flush=True,
        )
        print(
            f"Source windows: {len(windows):,} | strides {strides} | augmentations {options.augmentations}",
            flush=True,
        )
        try:
            for split, video, start in core.tqdm(
                windows, desc="Cache optical flow", unit="window"
            ):
                stats["source_windows"] += 1
                raw.begin_window()
                processed.begin_window()
                canvases = prepare_canvases(files[(split, video)], start, bank, core)
                for augmentation in options.augmentations:
                    path, key, payload, hit = process_window(
                        split,
                        video,
                        start,
                        files[(split, video)],
                        augmentation,
                        bank,
                        core,
                        raw,
                        processed,
                        directory,
                        contract,
                        canvases=canvases,
                    )
                    meta = json.loads(str(payload["metadata"]))
                    con.execute(
                        """INSERT INTO windows VALUES (?,?,?,?,?,?,?,?)
                        ON CONFLICT(cache_key) DO UPDATE SET updated_at=excluded.updated_at""",
                        (
                            key,
                            split,
                            video,
                            start,
                            meta["target_index"],
                            augmentation,
                            str(path.relative_to(directory)),
                            now(),
                        ),
                    )
                    con.execute(
                        "INSERT OR IGNORE INTO run_windows VALUES (?,?)", (run_id, key)
                    )
                    stats["cache_windows"] += 1
                    stats["window_hits" if hit else "window_writes"] += 1
                    stats["original_training_label_windows"] += int(
                        meta["assignment_source"]
                        == "exact_original_bank_training_assignments"
                    )
                    stats["unassigned_patches"] += int(
                        np.count_nonzero(~payload["assignment_valid"])
                    )
                    stats["supervised_patches"] += int(
                        np.count_nonzero(payload["temporal_train_mask"])
                    )
                    category = split + "/" + augmentation
                    if category not in stats["prototype_counts_by_split_view"]:
                        stats["prototype_counts_by_split_view"][category] = np.zeros(
                            bank.count, np.int64
                        )
                    valid_labels = payload["cluster_ids"][payload["assignment_valid"]]
                    stats["prototype_counts_by_split_view"][category] += np.bincount(
                        valid_labels, minlength=bank.count
                    )
                if stats["source_windows"] % 25 == 0:
                    con.commit()
            con.commit()
            stats["prototype_counts_by_split_view"] = {
                k: v.tolist()
                for k, v in stats["prototype_counts_by_split_view"].items()
            }
            stats.update(raw_flow=raw.stats, processed_pairs=processed.stats)
            completed = now()
            con.execute(
                "UPDATE runs SET completed_at=?,status=?,stats_json=? WHERE id=?",
                (completed, "complete", json.dumps(stats, allow_nan=False), run_id),
            )
            con.commit()
            output_manifest.update(
                status="complete", last_run_completed_at=completed, stats=stats
            )
            atomic_json(manifest_path, output_manifest)
        except BaseException as exc:
            con.rollback()
            con.execute(
                "UPDATE runs SET completed_at=?,status=?,stats_json=? WHERE id=?",
                (now(), "failed", json.dumps({"error": str(exc)}), run_id),
            )
            con.commit()
            output_manifest.update(status="failed", error=str(exc), failed_at=now())
            atomic_json(manifest_path, output_manifest)
            raise
        print(
            f'Complete: {directory}/index.sqlite | new raw estimator calls: {raw.stats["estimator_calls"]:,} | cached windows: {stats["cache_windows"]:,}',
            flush=True,
        )
        return directory, output_manifest
    finally:
        if con is not None:
            con.close()
        bank.close()


def load_cached_window(
    index_path,
    split,
    video,
    window_start,
    augmentation="original",
    load_dense=False,
    expected_bank_identity=None,
):
    """Read one fixed input; no estimator or trainable network runs here.

    Arrays follow row-major patch order [N,D]. Torch callers convert
    encoder_inputs to float tensors and cluster_ids to long tensors, and use
    temporal_train_mask for concept supervision. The metadata is plain Python.
    Dense arrays, when requested, follow manifest gap order [G,H,W,...].
    """
    index_path = Path(index_path).expanduser().resolve()
    root = index_path.parent
    with closing(readonly(index_path)) as con:
        row = con.execute(
            "SELECT value FROM metadata WHERE key=?", ("contract",)
        ).fetchone()
        if row is None:
            raise ValueError("Cache index has no fixed bank contract.")
        contract = json.loads(row[0])
        if (
            expected_bank_identity is not None
            and contract["bank_identity"] != expected_bank_identity
        ):
            raise ValueError("Trainer bank identity differs from the cached targets.")
        row = con.execute(
            """SELECT path,cache_key FROM windows WHERE split=? AND video=? AND window_start=? AND augmentation=?
            ORDER BY updated_at DESC LIMIT 1""",
            (split, str(video), int(window_start), augmentation),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"No cached window: {split}/{video}, start={window_start}, augmentation={augmentation}. Match trainer stride/index and augmentation coverage."
            )
        path = root / row["path"]
        key = row["cache_key"]
    result = load_npz(path)
    meta = json.loads(str(result.pop("metadata")))
    if meta["cache_key"] != key or meta["bank_identity"] != contract["bank_identity"]:
        raise ValueError("Window identity differs from its index/contract.")
    result["metadata"] = meta
    n = int(np.prod(meta["grid_hw"]))
    if (
        result["encoder_inputs"].shape
        != (n, sum(int(d) for _, d in contract["encoder_input_components"]))
        or not np.isfinite(result["encoder_inputs"]).all()
    ):
        raise ValueError("Invalid cached encoder-input tensor.")
    if result["cluster_ids"].shape != (n,) or np.any(
        (result["cluster_ids"] < -1)
        | (result["cluster_ids"] >= contract["num_prototypes"])
    ):
        raise ValueError("Invalid cached fixed cluster IDs.")
    if load_dense:
        if meta.get("dense_saved") is False:
            raise ValueError(
                "This window is a compact-only cache entry. Use load_dense=False; "
                "raw_pairs/ and pairs/ are intentionally not saved."
            )
        residual = []
        global_flow = []
        valid = []
        forward = []
        backward = []
        models = []
        raw_config = contract["flow_config"]
        raw_namespace = digest(raw_config)
        for pair in meta["pairs"]:
            derived = load_npz(root / pair["processed_path"])
            pair_meta = json.loads(str(derived["metadata"]))
            if (
                pair_meta["bank_identity"] != contract["bank_identity"]
                or pair_meta["gap"] != pair["gap"]
                or pair_meta["augmentation"] != augmentation
            ):
                raise ValueError("Dense pair identity mismatch.")
            key = pair["raw_pair_key"]
            # Prefer the self-contained output raw cache, including after moving it.
            candidates = [
                root.parent.parent
                / "raw_pairs"
                / raw_config["schema"]
                / raw_namespace
                / "entries"
                / key[:2]
                / (key + ".npz"),
                Path(pair["raw_path"]),
            ]
            raw_path = next((p for p in candidates if p.is_file()), None)
            if raw_path is None:
                raise FileNotFoundError(
                    f"Raw pair {key} missing. Use --copy-reused-flow before removing/moving source caches."
                )
            raw = load_npz(raw_path)
            if str(raw["namespace"]) != raw_namespace or str(raw["pair_sha256"]) != key:
                raise ValueError("Raw pair identity mismatch.")
            f, b = raw["forward"].astype(np.float32), raw["backward"].astype(np.float32)
            if augmentation == "hflip":
                f, b = flip_vectors(f), flip_vectors(b)
            forward.append(f)
            backward.append(b)
            residual.append(derived["residual_velocity"])
            global_flow.append(derived["global_velocity"])
            valid.append(derived["valid"])
            models.append(pair_meta["camera_model"])
        result.update(
            dense_forward=np.stack(forward),
            dense_backward=np.stack(backward),
            residual_velocity=np.stack(residual),
            global_velocity=np.stack(global_flow),
            flow_valid_mask=np.stack(valid),
            camera_models=models,
        )
    return result


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bank-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument(
        "--resume-index", type=Path,
        help="Continue an existing index.sqlite with matching bank/flow settings; completed compact windows need no dense files.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/quantization/zaima/videosal_datasets/dhf1k"),
    )
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--split-root", action="append", default=[], metavar="SPLIT=PATH")
    p.add_argument(
        "--window-len",
        type=int,
        help="Model input context length; defaults to bank window_len. Every fixed bank flow gap must fit inside it.",
    )
    p.add_argument(
        "--stride",
        type=int,
        help="Window start stride; default bank window_stride. Use 1 for arbitrary complete windows.",
    )
    p.add_argument("--split-stride", action="append", default=[], metavar="SPLIT=N")
    p.add_argument(
        "--window-index",
        type=Path,
        help="JSONL with split,video,window_start and optional target_index from your trainer.",
    )
    p.add_argument(
        "--augmentations",
        nargs="+",
        choices=("original", "hflip"),
        default=["original"],
    )
    p.add_argument(
        "--max-windows",
        type=int,
        help="Optional maximum source windows PER split for a smoke run.",
    )
    p.add_argument(
        "--expected-clusters",
        type=int,
        help="Optional bank count check, e.g. 131; the count always comes from the manifest.",
    )
    p.add_argument("--clustering-script", type=Path)
    p.add_argument(
        "--reuse-flow-cache-dir",
        type=Path,
        action="append",
        default=[],
        help="Exact compatible namespace directory; repeat for multiple sources.",
    )
    p.add_argument(
        "--read-only-flow-cache",
        action="store_true",
        help="Reuse raw flows only; fail on a miss. Derived inputs may still be written.",
    )
    p.add_argument(
        "--copy-reused-flow",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--raft-weights",
        type=Path,
        help="Local weights required only if a RAFT raw-flow miss must be computed.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="RAFT device if needed. CUDA Farneback uses --farneback-backend cuda --cuda-device N.",
    )
    p.add_argument("--flow-device", default=None)
    p.add_argument(
        "--gpu", type=int,
        help="CUDA device index. Enables CUDA Farneback for a Farneback bank, or CUDA for a RAFT bank.",
    )
    p.add_argument(
        "--farneback-backend", choices=("bank", "cuda"), default="bank",
        help="Default bank preserves the bank's estimator. cuda explicitly switches Farneback "
             "to OpenCV CUDA, using separate cache namespaces and new assignments; no CPU fallback.",
    )
    p.add_argument(
        "--cuda-device", type=int, default=0,
        help="OpenCV CUDA device index for CUDA Farneback (default 0); requires CUDA-enabled OpenCV.",
    )
    p.add_argument("--threads", type=int, default=4)
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.copy_reused_flow:
        p.error("Remove --copy-reused-flow: this builder never saves raw or processed dense pairs.")
    if args.gpu is not None:
        if args.gpu < 0:
            p.error("--gpu must be nonnegative.")
        args.cuda_device = args.gpu
    args.splits = list(dict.fromkeys(args.splits))
    args.augmentations = list(dict.fromkeys(args.augmentations))
    for split in args.splits:
        if split in (".", "..") or "/" in split or "\\" in split:
            p.error(
                "Split names must be simple identifiers; use --split-root to map paths."
            )
    for name in ("window_len", "stride", "max_windows", "expected_clusters"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f'--{name.replace("_","-")} must be positive.')
    if args.threads < 1:
        p.error("--threads must be positive.")
    if args.cuda_device < 0:
        p.error("--cuda-device must be nonnegative.")
    return args


def main(argv=None):
    options = parse_args(argv)
    try:
        cv2.setNumThreads(options.threads)
        cv2.setRNGSeed(42)
        build(options)
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        KeyError,
        sqlite3.Error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
