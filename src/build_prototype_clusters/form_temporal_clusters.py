#!/usr/bin/env python3
"""Build fixed stage4 temporal-change reference clusters.

Examples (run from src/):
  python patch_clusters/form_temporal_clusters.py --output-dir /path/to/temporal_bank
  python patch_clusters/form_temporal_clusters.py --clustering kmedoids --num-prototypes 128
  python patch_clusters/form_temporal_clusters.py --flow-estimator raft-small --raft-weights /local/raft.pth

One descriptor per last-frame stage4 cell: target-aligned, camera-relative flow
histogram (8 directions + static); direction/magnitude of frozen stage4 feature
change; and separate global camera motion. FINCH/medoid clustering is shared
with form_patch_clusters.py. New namespaces never reuse the last-slice visual
cache. No model weights, visual bank, or old cache entries are modified.

Backbone temporal stride is checked from the actual patch embedding and output
shape. A stride of two is a sampling step, NOT a two-frame receptive field:
Video Swin attention gives these features broader temporal context.
"""

from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
from tqdm import tqdm

if __package__:
    from .clustering_common import assign, choose_fit_sample, fit_finch, fit_medoids
else:
    from clustering_common import assign, choose_fit_sample, fit_finch, fit_medoids

FORMAT_VERSION = "fixed_stage4_temporal_clusters_v1"
FEATURE_CACHE_SCHEMA = "frozen_stage4_temporal_tail_v1"
FLOW_CACHE_SCHEMA = "bidirectional_canvas_flow_v1"
DEFAULT_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k")
DEFAULT_CHECKPOINT = Path(
    "/data/quantization/zaima/videoswin/swin_small_patch244_window877_kinetics400_1k.pth"
)
SRC_DIR = Path(__file__).resolve().parents[1]


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


def save_cache_once(path, payload):
    """Atomic, create-only cache write; another writer's entry is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        try:
            os.link(temporary, path)
            return True
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)


def load_payload(path):
    # These caches contain only tensors and plain Python metadata.
    return torch.load(path, map_location="cpu", weights_only=True)


def project_cache_module():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from model import backbone_feature_cache
    except ImportError as exc:
        raise ImportError(
            "Install this file and clustering_common.py in src/patch_clusters/ beside the existing src/model/backbone_feature_cache.py."
        ) from exc
    return backbone_feature_cache


def infer_backbone_geometry(reference):
    """Inspect patch embedding; current custom Swin merges only H/W thereafter."""
    inner = reference.backbone
    stride = tuple(int(v) for v in inner.patch_embed.proj.stride)
    merges = sum(
        getattr(layer, "downsample", None) is not None for layer in inner.layers
    )
    for layer in inner.layers:
        down = getattr(layer, "downsample", None)
        if down is not None and type(down).__name__ != "PatchMerging":
            raise ValueError(
                "Unknown backbone downsampler: verify temporal/spatial stride before using this script."
            )
    return {
        "temporal_stride": stride[0],
        "spatial_stride": [stride[1] * 2**merges, stride[2] * 2**merges],
        "patch_embedding_stride": list(stride),
        "spatial_merge_count": merges,
    }


def canonical_clip(rgb):
    rgb = torch.as_tensor(rgb).detach().cpu()
    integer_rgb = not rgb.is_floating_point()
    if rgb.ndim != 4:
        raise ValueError("RGB must be [T,H,W,3] or [T,3,H,W].")
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    if rgb.shape[1] != 3:
        raise ValueError("RGB must have three channels.")
    rgb = rgb.float().contiguous()
    if integer_rgb or rgb.numel() and rgb.max() > 2:
        rgb = rgb / 255.0
    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB must contain finite values in [0,1] (or uint8 [0,255]).")
    return rgb


def clip_hash(rgb):
    a = rgb.contiguous().numpy().astype("<f4", copy=False)
    h = hashlib.sha256(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class TemporalFeatureCache:
    """Separate content-addressed frozen stage4 cache, storing up to 3 tail slices."""

    def __init__(self, args, reference=None):
        helpers = project_cache_module()
        self.device = torch.device(args.device)
        checkpoint = Path(args.reference_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Local reference backbone checkpoint missing: {checkpoint}"
            )
        self.reference = (
            reference
            if reference is not None
            else helpers.build_reference_backbone(
                checkpoint=str(checkpoint),
                device=self.device,
                resize_to=tuple(args.resize_to),
                normalize=True,
            )
        )
        self.reference.to(self.device).float().eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.reference.return_stages = ("stage4",)
        self.geometry = infer_backbone_geometry(self.reference)
        stride = self.geometry["temporal_stride"]
        if args.window_len % stride or args.window_len // stride <= args.feature_lag:
            raise ValueError(
                f"--window-len must be divisible by temporal stride {stride} and contain at least {args.feature_lag + 1} output slices."
            )
        sh, sw = self.geometry["spatial_stride"]
        if args.resize_to[0] % sh or args.resize_to[1] % sw:
            raise ValueError(
                f"--resize-to must be divisible by stage4 spatial stride {(sh, sw)}; padded grid cells cannot be pooled as full patches."
            )
        self.channels = int(self.reference.get_feature_channels()["stage4"])
        self.fingerprint = helpers.fingerprint_state_dict(self.reference.state_dict())
        self.config = {
            "schema": FEATURE_CACHE_SCHEMA,
            "backbone_class": "model.backbones.video_swin_custom.VideoSwinTransformer",
            "backbone_fingerprint": self.fingerprint,
            "geometry": self.geometry,
            "resize_to": list(args.resize_to),
            "normalize": True,
            "input_range": "canonical RGB [0,1]; integer clips are divided by 255 even when very dark",
            "mean": list(helpers.DEFAULT_MEAN),
            "std": list(helpers.DEFAULT_STD),
            "tail_slices": args.cache_slices,
            "storage_dtype": args.feature_storage_dtype,
            "feature_dim": self.channels,
            "torch_version": str(torch.__version__),
            "implementation_sha256": {
                "wrapper": sha256_file(inspect.getfile(type(self.reference))),
                "backbone": sha256_file(inspect.getfile(type(self.reference.backbone))),
            },
            "feature_precision": "float32",
            "temporal_selection": "last_min(cache_slices,total_slices)_raw_stage4_slices",
        }
        self.namespace = digest_json(self.config)
        self.directory = (
            Path(args.feature_cache_root).expanduser().resolve()
            / FEATURE_CACHE_SCHEMA
            / self.namespace
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._ensure_manifest()
        self.dtype = (
            torch.float16 if args.feature_storage_dtype == "float16" else torch.float32
        )
        self.allow_compute = not args.cache_read_only
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "forwards": 0}
        self.provenance = {
            **self.config,
            "namespace": self.namespace,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "cache_directory": str(self.directory),
            "receptive_field_note": "Slices are temporally contextualized; patch stride is not receptive-field extent.",
        }

    def _ensure_manifest(self):
        path = self.directory / "manifest.json"
        if path.is_file():
            if json.loads(path.read_text()) != self.config:
                raise ValueError(f"Feature cache namespace metadata mismatch: {path}")
        else:
            # Namespace is content-derived; identical writers emit identical metadata.
            write_json(path, self.config)

    def _validate(self, payload, key, shape):
        if (
            payload.get("schema") != FEATURE_CACHE_SCHEMA
            or payload.get("namespace") != self.namespace
            or payload.get("content_sha256") != key
            or payload.get("input_shape") != list(shape)
        ):
            raise ValueError("Feature cache identity/schema mismatch.")
        total = shape[0] // self.geometry["temporal_stride"]
        indices = list(range(max(0, total - self.config["tail_slices"]), total))
        features = payload.get("features")
        h = self.config["resize_to"][0] // self.geometry["spatial_stride"][0]
        w = self.config["resize_to"][1] // self.geometry["spatial_stride"][1]
        if (
            payload.get("slice_indices") != indices
            or payload.get("total_slices") != total
        ):
            raise ValueError("Feature cache temporal indices mismatch.")
        if (
            not torch.is_tensor(features)
            or features.shape != (self.channels, len(indices), h, w)
            or features.dtype != self.dtype
            or not torch.isfinite(features).all()
        ):
            raise ValueError("Feature cache tensor shape/dtype/finiteness mismatch.")
        return features.float(), indices

    @torch.no_grad()
    def get(self, rgb):
        key = clip_hash(rgb)
        path = self.directory / "entries" / key[:2] / f"{key}.pt"
        if path.is_file():
            self.stats["hits"] += 1
            try:
                return self._validate(load_payload(path), key, rgb.shape)
            except Exception as exc:
                raise ValueError(
                    f"Invalid feature cache entry preserved at {path}: {exc}"
                ) from exc
        self.stats["misses"] += 1
        if not self.allow_compute:
            raise FileNotFoundError(f"Feature cache miss in read-only mode: {path}")
        self.stats["forwards"] += 1
        f = self.reference.forward_features(rgb.unsqueeze(0).to(self.device))["stage4"]
        total = rgb.shape[0] // self.geometry["temporal_stride"]
        if f.ndim != 5 or f.shape[:3] != (1, self.channels, total):
            raise ValueError(
                f"Actual stage4 shape {tuple(f.shape)} disagrees with inferred temporal stride. No cache entry written."
            )
        indices = list(range(max(0, total - self.config["tail_slices"]), total))
        payload = {
            "schema": FEATURE_CACHE_SCHEMA,
            "namespace": self.namespace,
            "content_sha256": key,
            "input_shape": list(rgb.shape),
            "slice_indices": indices,
            "total_slices": total,
            "features": f[0, :, indices].detach().cpu().to(self.dtype).contiguous(),
        }
        self._validate(payload, key, rgb.shape)
        if save_cache_once(path, payload):
            self.stats["writes"] += 1
        else:
            payload = load_payload(path)  # Validate a concurrent writer's result.
        return self._validate(payload, key, rgb.shape)


class FlowEstimator:
    """Farneback by default; torchvision RAFT only from an explicit local file."""

    def __init__(self, args):
        import cv2

        self.cv2 = cv2
        self.device = torch.device(args.flow_device or args.device)
        self.kind = args.flow_estimator
        self.updates = args.raft_updates
        self.model = None
        self.config = {
            "estimator": self.kind,
            "canvas": list(args.resize_to),
            "input": "RGB uint8 rounded after backbone-canvas bilinear resize",
            "flow_units": "canvas pixels over recorded frame interval",
            "bidirectional": True,
        }
        if self.kind == "farneback":
            self.config.update(
                opencv_version=cv2.__version__,
                parameters={
                    "pyr_scale": 0.5,
                    "levels": 3,
                    "winsize": 15,
                    "iterations": 3,
                    "poly_n": 5,
                    "poly_sigma": 1.2,
                    "flags": 0,
                },
            )
        else:
            import torchvision
            from torchvision.models.optical_flow import raft_large, raft_small

            if not args.raft_weights or not Path(args.raft_weights).is_file():
                raise FileNotFoundError(
                    "--flow-estimator raft-* requires --raft-weights pointing to a local torchvision RAFT state_dict; no automatic download is attempted."
                )
            if min(args.resize_to) < 128:
                raise ValueError(
                    "Torchvision RAFT requires a canvas of at least 128x128."
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
                checkpoint_sha256=sha256_file(args.raft_weights),
                iterations=self.updates,
                normalization="uint8 RGB -> float [-1,1]; right/bottom replicate pad to multiple of 8",
            )

    @torch.no_grad()
    def estimate(self, previous, current):
        if self.model is None:
            cv2 = self.cv2
            a = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
            p = self.config["parameters"]
            forward = cv2.calcOpticalFlowFarneback(a, b, None, **p)
            backward = cv2.calcOpticalFlowFarneback(b, a, None, **p)
            return forward.astype(np.float32), backward.astype(np.float32)
        a = (
            torch.from_numpy(previous.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 127.5
            - 1
        )
        b = (
            torch.from_numpy(current.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 127.5
            - 1
        )
        h, w = a.shape[-2:]
        pad = (0, (-w) % 8, 0, (-h) % 8)
        a, b = F.pad(a, pad, mode="replicate"), F.pad(b, pad, mode="replicate")

        def run(x, y):
            flow = self.model(x, y, num_flow_updates=self.updates)[-1][0, :, :h, :w]
            return flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)

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
            Path(args.flow_cache_root).expanduser().resolve()
            / FLOW_CACHE_SCHEMA
            / self.namespace
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "manifest.json"
        if manifest.is_file() and json.loads(manifest.read_text()) != self.config:
            raise ValueError("Flow cache namespace metadata mismatch.")
        if not manifest.exists():
            write_json(manifest, self.config)
        self.dtype = (
            torch.float16 if args.flow_storage_dtype == "float16" else torch.float32
        )
        self.allow_compute = not args.cache_read_only
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "estimator_calls": 0}

    def _validate(self, payload, key, shape):
        if (
            payload.get("schema") != FLOW_CACHE_SCHEMA
            or payload.get("namespace") != self.namespace
            or payload.get("pair_sha256") != key
            or payload.get("shape") != list(shape)
        ):
            raise ValueError("Flow cache identity/schema mismatch.")
        arrays = []
        for name in ("forward", "backward"):
            f = payload.get(name)
            if (
                not torch.is_tensor(f)
                or f.shape != (*shape[:2], 2)
                or f.dtype != self.dtype
                or not torch.isfinite(f).all()
            ):
                raise ValueError("Invalid cached flow field shape/dtype/finiteness.")
            arrays.append(f.float().numpy())
        return tuple(arrays)

    def get(self, previous, current):
        if (
            previous.shape != current.shape
            or previous.dtype != np.uint8
            or current.dtype != np.uint8
        ):
            raise ValueError(
                "Flow input images must be equal-shape uint8 RGB canvases."
            )
        h = hashlib.sha256(str(previous.shape).encode())
        h.update(np.ascontiguousarray(previous).tobytes())
        h.update(np.ascontiguousarray(current).tobytes())
        key = h.hexdigest()
        path = self.directory / "entries" / key[:2] / f"{key}.pt"
        if path.is_file():
            self.stats["hits"] += 1
            try:
                f, b = self._validate(load_payload(path), key, previous.shape)
            except Exception as exc:
                raise ValueError(
                    f"Invalid flow cache entry preserved at {path}: {exc}"
                ) from exc
            return f, b, key
        self.stats["misses"] += 1
        if not self.allow_compute:
            raise FileNotFoundError(f"Flow cache miss in read-only mode: {path}")
        self.stats["estimator_calls"] += 1
        f, b = self.estimator.estimate(previous, current)
        payload = {
            "schema": FLOW_CACHE_SCHEMA,
            "namespace": self.namespace,
            "pair_sha256": key,
            "shape": list(previous.shape),
            "forward": torch.from_numpy(f).to(self.dtype),
            "backward": torch.from_numpy(b).to(self.dtype),
        }
        self._validate(payload, key, previous.shape)
        if save_cache_once(path, payload):
            self.stats["writes"] += 1
        else:
            payload = load_payload(path)
        f, b = self._validate(payload, key, previous.shape)
        return f, b, key


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


def summarize_flow(backward, valid, grid_h, grid_w, args):
    """Physical forward-direction estimate at target pixels is -backward flow."""
    motion = -backward
    residual, global_flow, model = compensate_camera(
        motion, valid, args.camera_motion, args.seed, args.camera_ransac_threshold
    )
    h, w = backward.shape[:2]
    if h % grid_h or w % grid_w:
        raise ValueError("Flow canvas must divide evenly into the stage4 grid.")
    ph, pw = h // grid_h, w // grid_w
    n = grid_h * grid_w
    hist = np.zeros((n, 9), dtype=np.float32)
    mean_mag = np.zeros(n, dtype=np.float32)
    global_xy = np.zeros((n, 2), dtype=np.float32)
    mean_backward = np.zeros((n, 2), dtype=np.float32)
    fractions = np.zeros(n, dtype=np.float32)
    for row in range(grid_h):
        for col in range(grid_w):
            i = row * grid_w + col
            sl = np.s_[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw]
            mask = valid[sl]
            fractions[i] = mask.mean()
            hist[i, -1] = 1.0
            global_xy[i] = global_flow[sl].mean((0, 1))
            if not mask.any():
                continue
            vectors = residual[sl][mask]
            magnitude = np.linalg.norm(vectors, axis=-1)
            moving = magnitude > args.flow_static_threshold
            fraction = float(moving.mean())
            hist[i, -1] = 1.0 - fraction
            if moving.any():
                angles = np.mod(
                    np.arctan2(vectors[moving, 1], vectors[moving, 0]), 2 * np.pi
                )
                bins = np.minimum((angles * 8 / (2 * np.pi)).astype(int), 7)
                counts = np.bincount(bins, weights=magnitude[moving], minlength=8)
                hist[i, :8] = fraction * counts / counts.sum()
            mean_mag[i] = magnitude.mean()
            mean_backward[i] = backward[sl][mask].mean(axis=0)
    flow_valid = fractions >= args.min_flow_valid_fraction
    return hist, mean_mag, global_xy, mean_backward, fractions, flow_valid, model


def feature_change(previous, current, mean_backward, flow_valid, canvas_hw, args):
    """Signed difference of unit feature vectors; optional target-aligned warp."""
    h, w = current.shape[-2:]
    valid = torch.ones(h * w, dtype=torch.bool)
    previous = previous.float()
    current = current.float()
    if args.warp_features:
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        shift = torch.from_numpy(mean_backward).reshape(h, w, 2)
        px = x + shift[..., 0] * w / canvas_hw[1]
        py = y + shift[..., 1] * h / canvas_hw[0]
        valid = ((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)).reshape(
            -1
        ) & torch.from_numpy(flow_valid)
        grid = torch.stack(
            (2 * (px + 0.5) / w - 1, 2 * (py + 0.5) / h - 1), dim=-1
        ).unsqueeze(0)
        previous = F.grid_sample(
            previous.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0]
    a = previous.permute(1, 2, 0).reshape(-1, previous.shape[0])
    b = current.permute(1, 2, 0).reshape(-1, current.shape[0])
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    valid &= (na > 1e-12) & (nb > 1e-12)
    delta = F.normalize(b, dim=-1, eps=1e-12) - F.normalize(a, dim=-1, eps=1e-12)
    magnitude = delta.norm(dim=-1)
    static = magnitude <= args.change_static_threshold
    direction = delta / magnitude.clamp_min(1e-12).unsqueeze(-1)
    direction[static | ~valid] = 0
    magnitude[~valid] = 0
    return direction.numpy(), magnitude.numpy(), static.numpy(), valid.numpy()


class TemporalDescriptors:
    """Memory-mapped fixed descriptors; distance is pluggable into the shared engine."""

    SPECS = {
        "flow_hist": (9, np.float32),
        "change_direction": (None, np.float32),
        "flow_magnitude": (0, np.float32),
        "change_magnitude": (0, np.float32),
        "global_motion_raw": (2, np.float32),
        "flow_valid_fraction": (0, np.float32),
        "flow_valid": (0, np.bool_),
        "change_valid": (0, np.bool_),
        "change_static": (0, np.bool_),
        "node_ids": (0, np.int64),
        "magnitude": (2, np.float32),
        "global_motion": (2, np.float32),
    }

    def __init__(self, directory, count=None, feature_dim=None, mode="r"):
        self.directory = Path(directory)
        if mode == "w+":
            self.directory.mkdir(parents=True, exist_ok=True)
        for name, (dim, dtype) in self.SPECS.items():
            size = feature_dim if dim is None else dim
            shape = (count,) if size == 0 else (count, size)
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
                "change_direction",
                "change_static",
                "magnitude",
                "global_motion",
                "flow_valid",
                "change_valid",
            )
        )

    def distance(self, a, b, weights):
        return temporal_distance(a, b, weights)

    def flush(self):
        for name in self.SPECS:
            getattr(self, name).flush()


def temporal_distance(a, b, weights):
    ha, da, sa, ma, ga, fa, va = a
    hb, db, sb, mb, gb, fb, vb = b
    flow = cdist(ha, hb, metric="cityblock") / 2.0
    flow[~fa[:, None] & ~fb[None, :]] = 0
    flow[fa[:, None] != fb[None, :]] = 1
    direction = 1.0 - np.clip(da @ db.T, -1.0, 1.0).astype(np.float64)
    both_static = sa[:, None] & sb[None, :]
    one_static = sa[:, None] != sb[None, :]
    direction[both_static] = 0.0
    direction[one_static] = 2.0
    direction[~va[:, None] & ~vb[None, :]] = 0.0
    direction[va[:, None] != vb[None, :]] = 2.0
    # Static directions are exactly self-identical; roundoff is clamped for others.
    magnitude = (
        np.abs(ma[:, None, 0] - mb[None, :, 0]) * (fa[:, None] & fb[None, :])
        + np.abs(ma[:, None, 1] - mb[None, :, 1]) * (va[:, None] & vb[None, :])
    ) / 2.0
    global_motion = cdist(ga, gb, metric="cityblock") / 2.0
    global_motion *= fa[:, None] & fb[None, :]
    return np.maximum(
        0.0,
        weights[0] * flow
        + weights[1] * direction
        + weights[2] * magnitude
        + weights[3] * global_motion,
    )


def percentile_scale(values, valid, percentile):
    selected = np.asarray(values[valid], dtype=np.float64)
    if not len(selected):
        return 1.0, "no_valid_values"
    scale = float(np.percentile(selected, percentile))
    if scale <= 1e-12:
        positive = selected[selected > 1e-12]
        return (
            (float(np.percentile(positive, percentile)), "positive_values_fallback")
            if len(positive)
            else (1.0, "all_zero_fallback")
        )
    return scale, "all_valid_values"


def normalize_magnitudes(data, percentile):
    flow_scale, flow_rule = percentile_scale(
        data.flow_magnitude, data.flow_valid, percentile
    )
    change_scale, change_rule = percentile_scale(
        data.change_magnitude, data.change_valid, percentile
    )
    global_mag = np.linalg.norm(data.global_motion_raw, axis=1)
    global_scale, global_rule = percentile_scale(
        global_mag, data.flow_valid, percentile
    )
    for start in tqdm(range(0, data.count, 8192), desc="Normalize magnitudes", unit="batch"):
        sl = slice(start, start + 8192)
        data.magnitude[sl, 0] = data.flow_magnitude[sl] / flow_scale
        data.magnitude[sl, 1] = data.change_magnitude[sl] / change_scale
        data.global_motion[sl] = data.global_motion_raw[sl] / global_scale
    data.flush()
    return {
        "percentile": percentile,
        "population": "all valid training descriptors, not only discovery sample",
        "flow_magnitude_scale": flow_scale,
        "change_magnitude_scale": change_scale,
        "global_motion_scale": global_scale,
        "flow_scale_rule": flow_rule,
        "change_scale_rule": change_rule,
        "global_scale_rule": global_rule,
        "clipped": False,
        "future_inference": "reuse these frozen scales; never recompute them per batch/validation/test",
    }


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


def extract_descriptors(args, directory, con, feature_cache, flow_cache):
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from pre_process.dataloader import DatasetLoader

    split_dir = Path(args.dataset_root).expanduser().resolve() / "train"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Training split missing: {split_dir}")
    dataset = DatasetLoader(
        str(split_dir),
        window_len=args.window_len,
        stride=args.stride,
        random_train_sampling=False,
    )
    limit = min(len(dataset), args.max_windows) if args.max_windows else len(dataset)
    if not limit:
        raise ValueError("No eligible training windows.")
    gh = args.resize_to[0] // feature_cache.geometry["spatial_stride"][0]
    gw = args.resize_to[1] // feature_cache.geometry["spatial_stride"][1]
    per_frame = gh * gw
    data = TemporalDescriptors(
        directory / "descriptors", limit * per_frame, feature_cache.channels, mode="w+"
    )
    stride = feature_cache.geometry["temporal_stride"]
    total_slices = args.window_len // stride
    current_slice, previous_slice = (
        total_slices - 1,
        total_slices - 1 - args.feature_lag,
    )
    current_offset, previous_offset = (
        args.window_len - 1,
        args.window_len - 1 - args.feature_lag * stride,
    )
    fingerprint = hashlib.sha256()
    fingerprint.update(
        digest_json(
            {
                "feature": feature_cache.config,
                "flow": flow_cache.config,
                "descriptor": descriptor_config(args),
            }
        ).encode()
    )
    for window in tqdm(range(limit), desc="Temporal windows", unit="window"):
        video, start = dataset.windows[window]
        frame_names = dataset.video_frame_names[video][start : start + args.window_len]
        sample = dataset[window]
        if sample[0] != video or int(sample[4]) != args.window_len:
            raise RuntimeError(f"Dataset window identity mismatch at {window}.")
        rgb = canonical_clip(sample[1])
        features, indices = feature_cache.get(rgb)
        a, b = (
            features[:, indices.index(previous_slice)],
            features[:, indices.index(current_slice)],
        )
        pair = F.interpolate(
            rgb[[previous_offset, current_offset]],
            size=tuple(args.resize_to),
            mode="bilinear",
            align_corners=False,
        )
        pair = (
            (pair.permute(0, 2, 3, 1).numpy() * 255.0)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )
        forward, backward, pair_key = flow_cache.get(pair[0], pair[1])
        valid = target_flow_validity(
            forward, backward, args.flow_fb_absolute, args.flow_fb_relative
        )
        hist, mag, global_xy, mean_b, fractions, flow_valid, camera = summarize_flow(
            backward, valid, gh, gw, args
        )
        direction, change_mag, static, change_valid = feature_change(
            a, b, mean_b, flow_valid, tuple(args.resize_to), args
        )
        sl = slice(window * per_frame, (window + 1) * per_frame)
        for name, value in [
            ("flow_hist", hist),
            ("flow_magnitude", mag),
            ("global_motion_raw", global_xy),
            ("flow_valid_fraction", fractions),
            ("flow_valid", flow_valid),
            ("change_direction", direction),
            ("change_magnitude", change_mag),
            ("change_static", static),
            ("change_valid", change_valid),
        ]:
            getattr(data, name)[sl] = value
        ids = np.arange(sl.start, sl.stop, dtype=np.int64)
        data.node_ids[sl] = ids
        source = split_dir / video / "images"
        con.execute(
            "INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                window,
                "train",
                video,
                window,
                start,
                start + current_offset,
                frame_names[current_offset],
                start + previous_offset,
                frame_names[previous_offset],
                previous_slice,
                current_slice,
                previous_offset,
                current_offset,
                args.resize_to[0],
                args.resize_to[1],
                str((source / frame_names[current_offset]).resolve()),
                str((source / frame_names[previous_offset]).resolve()),
                pair_key,
                json.dumps(camera, sort_keys=True),
            ),
        )
        records = [
            (
                int(i),
                int(i),
                window,
                "stage4",
                j // gw,
                j % gw,
                gh,
                gw,
                float(mag[j]),
                float(change_mag[j]),
                float(fractions[j]),
                int(flow_valid[j]),
                int(change_valid[j]),
                int(static[j]),
            )
            for j, i in enumerate(ids)
        ]
        con.executemany(
            "INSERT INTO patches(descriptor_index,node_id,frame_id,stage,row,col,grid_h,grid_w,flow_magnitude,change_magnitude,flow_valid_fraction,flow_valid,change_valid,change_static) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            records,
        )
        fingerprint.update(
            digest_json([video, start, frame_names, clip_hash(rgb), pair_key]).encode()
        )
        for value in (
            hist,
            mag,
            global_xy,
            direction,
            change_mag,
            static,
            flow_valid,
            change_valid,
        ):
            fingerprint.update(np.ascontiguousarray(value).tobytes())
        if (window + 1) % 25 == 0 or window + 1 == limit:
            con.commit()
    con.commit()
    data.flush()
    return data, {
        "num_frames": limit,
        "num_patches": data.count,
        "feature_dim": feature_cache.channels,
        "stage4_grid": {"H": gh, "W": gw},
        "temporal_stride": stride,
        "feature_lag": args.feature_lag,
        "frame_span": current_offset - previous_offset,
        "previous_slice": previous_slice,
        "current_slice": current_slice,
        "previous_frame_offset": previous_offset,
        "current_frame_offset": current_offset,
        "source_patch_descriptor_sha256": fingerprint.hexdigest(),
    }


def descriptor_config(args):
    return {
        "flow_anchor": "current-frame pixels; apparent forward motion = -backward flow",
        "histogram": "8 magnitude-weighted moving direction bins, scaled by moving-pixel fraction, plus static fraction; sum=1",
        "direction_axes": "x right, y down; bins start at +x and increase clockwise",
        "camera_motion": args.camera_motion,
        "camera_ransac_threshold": args.camera_ransac_threshold,
        "flow_static_threshold": args.flow_static_threshold,
        "flow_threshold_units": "canvas pixels per recorded frame span",
        "feature_change": "normalize(current) - normalize(previous); normalize previous after optional grid warp",
        "feature_lag": args.feature_lag,
        "warp_features": args.warp_features,
        "warp": "pooled backward displacement / cell size; bilinear, align_corners=False; invalid samples marked unknown",
        "change_static_threshold": args.change_static_threshold,
        "min_flow_valid_fraction": args.min_flow_valid_fraction,
        "forward_backward_absolute": args.flow_fb_absolute,
        "forward_backward_relative": args.flow_fb_relative,
        "static_policy": "flow all-static => bin8=1; feature both-static distance=0, one-static distance=2",
        "invalid_policy": "retain patches, mark validity; both unknown distance=0, one unknown is max for histogram/direction; magnitude/global comparisons require both valid",
        "magnitudes": "mean residual-flow magnitude and normalized-feature-difference norm, each divided by its training percentile",
        "global_term": "separate mean global displacement vector / training global-magnitude percentile",
        "feature_receptive_field": "Video Swin features contain contextual information beyond their two-frame sampling stride",
    }


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
    totals = np.zeros(k, dtype=np.int64)
    for start in tqdm(
        range(0, data.count, args.batch_size), desc="Assign temporal patches", unit="batch"
    ):
        rows = np.arange(start, min(start + args.batch_size, data.count))
        group, distance = assign(data, rows, medoids, weights, args.batch_size)
        labels[rows], distances[rows] = group, distance
        totals += np.bincount(group, minlength=k)
        con.executemany(
            "UPDATE patches SET prototype_id=?,distance=? WHERE descriptor_index=?",
            [(int(g), float(d), int(i)) for g, d, i in zip(group, distance, rows)],
        )
        if (
            start == 0
            or start + len(rows) == data.count
            or start // args.batch_size % 50 == 0
        ):
            con.commit()
    if (totals == 0).any():
        raise RuntimeError("Empty final cluster despite medoid self-ownership.")
    con.executescript(
        "CREATE INDEX patches_cluster_distance ON patches(prototype_id,distance,node_id); CREATE INDEX patches_cluster_node ON patches(prototype_id,node_id);"
    )
    stats = []
    for j, medoid in enumerate(medoids):
        row = con.execute(
            "SELECT COUNT(*) AS n,AVG(distance) AS d,AVG(flow_magnitude) AS fm,AVG(change_magnitude) AS cm FROM patches WHERE prototype_id=?",
            (j,),
        ).fetchone()
        videos = con.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT f.dataset,f.video FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.prototype_id=?)",
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
                row["cm"],
            ),
        )
        stats.append(
            {
                "prototype_id": j,
                "medoid_node_id": int(data.node_ids[medoid]),
                "patch_count": row["n"],
                "video_count": videos,
                "mean_distance": row["d"],
                "mean_flow_magnitude": row["fm"],
                "mean_change_magnitude": row["cm"],
            }
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
        **{name: np.asarray(getattr(data, name)[medoids]) for name in data.SPECS},
    )
    return stats


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


def build(args, feature_cache=None, flow_cache=None):
    validate_output_location(args)
    weights = (
        args.flow_weight,
        args.change_direction_weight,
        args.magnitude_weight,
        args.global_weight,
    )
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "building",
        "created_at": now(),
        "stage": "stage4",
        "source_mode": "training_temporal_cache",
        "source_dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "datasets": ["train"],
        "reference_groups_fixed": True,
        "saliency_used_for_clustering": False,
        "model_assignments_used_for_clustering": False,
        "window_len": args.window_len,
        "window_stride": args.stride,
        "descriptor": descriptor_config(args),
        "distance": "w_flow*L1(hist)/2 + w_dir*(1-cos(direction)) + w_mag*mean_L1([flow_mag,change_mag]) + w_global*mean_L1(global_xy)",
        "weights": dict(
            zip(("flow", "change_direction", "magnitude", "global_motion"), weights)
        ),
        "seed": args.seed,
        "clustering": args.clustering,
        "fit_sample_size_requested": args.fit_sample_size,
        "fit_max_per_video": args.fit_max_per_video,
        "cache_namespaces_are_separate_from_visual_cache": True,
    }
    # Check model/checkpoint geometry before creating an output directory.
    feature_cache = feature_cache or TemporalFeatureCache(args)
    flow_cache = flow_cache or FlowCache(args)
    directory = prepare_directory(args.output_dir, args.overwrite)
    manifest.update(
        reference_feature_provenance=feature_cache.provenance,
        flow_provenance=flow_cache.config,
        flow_cache_namespace=flow_cache.namespace,
        flow_cache_directory=str(flow_cache.directory),
    )
    write_json(directory / "manifest.json", manifest)
    try:
        with closing(sqlite3.connect(directory / "temporal_clusters.sqlite")) as con:
            con.row_factory = sqlite3.Row
            create_schema(con)
            data, extraction = extract_descriptors(
                args, directory, con, feature_cache, flow_cache
            )
            manifest.update(extraction)
            manifest["coverage"] = {
                "flow_valid_patches": int(np.count_nonzero(data.flow_valid)),
                "change_valid_patches": int(np.count_nonzero(data.change_valid)),
                "static_feature_changes": int(
                    np.count_nonzero(data.change_static & data.change_valid)
                ),
                "mean_flow_valid_fraction": float(data.flow_valid_fraction.mean()),
            }
            print(
                f"Stage4 grid {extraction['stage4_grid']} | slices {extraction['previous_slice']} -> {extraction['current_slice']} | "
                f"RGB offsets {extraction['previous_frame_offset']} -> {extraction['current_frame_offset']} "
                f"({extraction['frame_span']} frames)",
                flush=True,
            )
            if data.count < args.num_prototypes:
                raise ValueError(
                    "Not enough target-frame patches for the requested cluster lower bound/count."
                )
            manifest["normalization"] = normalize_magnitudes(
                data, args.magnitude_percentile
            )
            sample, num_videos = choose_fit_sample(
                con,
                args.fit_sample_size,
                args.fit_max_per_video,
                np.random.default_rng(args.seed),
            )
            if len(sample) < args.num_prototypes:
                raise ValueError(
                    "Video-capped discovery sample is too small. Increase --fit-sample-size/--fit-max-per-video."
                )
            np.save(directory / "fit_sample_indices.npy", sample, allow_pickle=False)
            print(
                f"{data.count:,} temporal patches, {num_videos:,} videos; fitting on {len(sample):,} patches.",
                flush=True,
            )
            if args.clustering == "finch":
                medoids, history, objective, finch = fit_finch(
                    data, sample, args, weights
                )
                manifest["finch"] = finch
            else:
                medoids, history, objective = fit_medoids(data, sample, args, weights)
            clusters = save_assignments(con, data, directory, medoids, args, weights)
            if (
                con.execute(
                    "SELECT COUNT(*) FROM patches WHERE prototype_id IS NULL"
                ).fetchone()[0]
                or con.execute("PRAGMA quick_check").fetchone()[0] != "ok"
            ):
                raise RuntimeError(
                    "Final temporal bank/database integrity check failed."
                )
            manifest.update(
                status="complete",
                completed_at=now(),
                num_prototypes=len(medoids),
                num_videos=num_videos,
                fit_sample_size=len(sample),
                fit_history=history,
                fit_mean_distance=objective,
                clusters=clusters,
                references_sha256=sha256_file(directory / "cluster_references.npz"),
                feature_cache_stats=feature_cache.stats,
                flow_cache_stats=flow_cache.stats,
                database="temporal_clusters.sqlite",
                references="cluster_references.npz",
                algorithm="sampled_finch_medoid_representatives_then_full_closest_medoid_assignment"
                if args.clustering == "finch"
                else "sampled_candidate_k_medoids_then_full_closest_medoid_assignment",
            )
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
        f"Complete: {directory} | {len(medoids)} temporal reference clusters",
        flush=True,
    )
    print(
        "These are temporal references. A temporal encoder/prototype branch and matching loss are still needed to use them in saliency prediction.",
        flush=True,
    )
    return manifest


def parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "concept_clusters/temporal_clusters_frozen_ref",
    )
    p.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--window-len", type=int, default=32)
    p.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Stride between window starts; frames inside each window remain contiguous.",
    )
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument(
        "--resize-to", type=int, nargs=2, default=[224, 384], metavar=("H", "W")
    )
    p.add_argument(
        "--feature-lag",
        type=int,
        default=1,
        help="Difference between last slice and this many backbone slices earlier.",
    )
    p.add_argument("--cache-slices", type=int, choices=(2, 3), default=3)
    p.add_argument(
        "--feature-cache-root",
        type=Path,
        default=DEFAULT_ROOT / "cached_temporal_feats",
    )
    p.add_argument(
        "--flow-cache-root", type=Path, default=DEFAULT_ROOT / "cached_temporal_flow"
    )
    p.add_argument(
        "--feature-storage-dtype", choices=("float16", "float32"), default="float16"
    )
    p.add_argument(
        "--flow-storage-dtype", choices=("float16", "float32"), default="float32"
    )
    p.add_argument(
        "--cache-read-only",
        action="store_true",
        help="Require compatible entries in BOTH new caches. Never falls back to the visual last-slice cache.",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="CUDA device index (sets --device to cuda:{id}). Mutually exclusive with --device.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device string for the backbone, e.g. cuda:0 or cpu. Default: cuda:0 if available else cpu.",
    )
    p.add_argument(
        "--flow-device",
        default=None,
        help="Optional separate device for optical flow (defaults to --device / --gpu).",
    )
    p.add_argument(
        "--flow-estimator",
        choices=("farneback", "raft-small", "raft-large"),
        default="farneback",
    )
    p.add_argument(
        "--raft-weights",
        type=Path,
        default=None,
        help="Local torchvision-format RAFT state_dict; nothing is downloaded automatically.",
    )
    p.add_argument("--raft-updates", type=int, default=12)
    p.add_argument(
        "--camera-motion", choices=("median", "affine", "none"), default="median"
    )
    p.add_argument("--camera-ransac-threshold", type=float, default=1.5)
    p.add_argument(
        "--warp-features",
        action="store_true",
        help="Optional coarse stage4 feature warp ablation; off by default.",
    )
    p.add_argument("--flow-static-threshold", type=float, default=0.25)
    p.add_argument("--change-static-threshold", type=float, default=1e-6)
    p.add_argument("--min-flow-valid-fraction", type=float, default=0.2)
    p.add_argument("--flow-fb-absolute", type=float, default=1.0)
    p.add_argument("--flow-fb-relative", type=float, default=0.05)
    p.add_argument("--magnitude-percentile", type=float, default=95.0)
    p.add_argument("--flow-weight", type=float, default=1.0)
    p.add_argument("--change-direction-weight", type=float, default=1.0)
    p.add_argument("--magnitude-weight", type=float, default=1.0)
    p.add_argument(
        "--global-weight",
        type=float,
        default=0.25,
        help="Separate camera-motion term; set 0 to exclude it from distance while retaining it in outputs.",
    )
    p.add_argument("--clustering", choices=("finch", "kmedoids"), default="finch")
    p.add_argument(
        "--num-prototypes",
        "--num-clusters",
        dest="num_prototypes",
        type=int,
        default=128,
        help="kmedoids count; ignored by FINCH.",
    )
    p.add_argument("--finch-level", type=int, default=None)
    p.add_argument("--min-clusters", type=int, default=16)
    p.add_argument("--max-clusters", type=int, default=1024)
    p.add_argument("--fit-sample-size", type=int, default=4096)
    p.add_argument("--fit-max-per-video", type=int, default=128)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--medoid-candidates", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p


def resolve_device(args, parser_: argparse.ArgumentParser) -> None:
    """Resolve --gpu / --device into args.device."""
    if args.gpu is not None and args.device is not None:
        parser_.error("Pass only one of --gpu or --device")
    if args.gpu is not None:
        if args.gpu < 0:
            parser_.error("--gpu must be a nonnegative integer")
        if not torch.cuda.is_available():
            parser_.error("--gpu requires CUDA, but no GPU is available")
        count = torch.cuda.device_count()
        if args.gpu >= count:
            parser_.error(f"--gpu {args.gpu} is out of range; found {count} CUDA device(s)")
        args.device = f"cuda:{args.gpu}"
    elif args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    resolve_device(args, p)
    positive = (
        "window_len",
        "stride",
        "feature_lag",
        "raft_updates",
        "num_prototypes",
        "fit_sample_size",
        "iterations",
        "restarts",
        "medoid_candidates",
        "batch_size",
        "threads",
        "min_clusters",
    )
    for name in positive:
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive.")
    if args.feature_lag >= args.cache_slices:
        p.error("--cache-slices must be greater than --feature-lag.")
    if min(args.resize_to) < 1 or args.max_windows is not None and args.max_windows < 1:
        p.error("Resize dimensions and --max-windows must be positive.")
    if (
        args.fit_max_per_video < 0
        or args.max_clusters < 0
        or args.max_clusters
        and args.max_clusters < args.min_clusters
    ):
        p.error("Invalid cluster bounds / video sample cap.")
    if (
        args.min_clusters < 2
        or not 0 < args.magnitude_percentile <= 100
        or not 0 < args.min_flow_valid_fraction <= 1
    ):
        p.error(
            "Require min-clusters>=2, 0<percentile<=100, and 0<min-flow-valid-fraction<=1."
        )
    for name in (
        "flow_static_threshold",
        "change_static_threshold",
        "flow_fb_absolute",
        "flow_fb_relative",
        "camera_ransac_threshold",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    weights = (
        args.flow_weight,
        args.change_direction_weight,
        args.magnitude_weight,
        args.global_weight,
    )
    if not all(math.isfinite(w) and w >= 0 for w in weights) or not any(weights):
        p.error(
            "Weights must be finite, nonnegative, and at least one must be positive."
        )
    if args.clustering == "finch":
        if args.num_prototypes != p.get_default("num_prototypes"):
            print(
                "NOTE: --num-prototypes is ignored with FINCH; choose --finch-level or cluster bounds.",
                flush=True,
            )
        args.num_prototypes = args.min_clusters
    if args.fit_sample_size < args.num_prototypes:
        p.error("Fit sample is smaller than the requested cluster count/lower bound.")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        import cv2

        cv2.setNumThreads(args.threads)
        torch.set_num_threads(args.threads)
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
