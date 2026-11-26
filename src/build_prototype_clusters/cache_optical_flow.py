#!/usr/bin/env python3
"""Precompute fixed optical-flow inputs against an existing temporal concept bank.

Place beside the FLOW-ONLY form_temporal_clusters.py used to build your bank,
inside src/patch_clusters/, or supply --clustering-script. No training model,
backbone, image embeddings or trainable temporal encoder is loaded.

Examples (run from src/):
  python patch_clusters/cache_optical_flow.py \
      --bank-dir /path/to/temporal_bank --cache-root /path/to/flow_training_cache \
      --dataset-root /path/to/dhf1k --splits train val test \
      --augmentations original hflip --expected-clusters 131

  # Match your TRAINER's window starts. Default stride is the bank window stride.
  # For arbitrary/random window starts, --stride 1 covers every complete window.
  # If the model uses 16-frame windows, add --window-len 16. Bank gaps stay fixed.
  python patch_clusters/cache_optical_flow.py --bank-dir /path/to/bank \
      --cache-root /path/to/cache --stride 1 --splits train

  # Different split directory names/strides are explicit, never silently skipped.
  python patch_clusters/cache_optical_flow.py --bank-dir /path/to/bank \
      --cache-root /path/to/cache --split-root val=/data/dhf1k/validation \
      --split-stride train=8 --split-stride val=1 --split-stride test=1

Dependencies: numpy scipy opencv-python[-headless]; tqdm optional. RAFT imports
Torch/torchvision only on a raw-flow cache MISS and requires local weights.
CPU Farneback/DIS need no Torch. The estimator, frame gaps, canvas, patch grid,
static/validity thresholds, camera compensation, metric, weights and TRAINING
normalization scales come from manifest.json. They cannot be overridden here.
The bank must be fixed_stage4_temporal_clusters_v2_flow_only and complete.

For a target t, only (t-gap,t) is decoded for EACH recorded gap; the full window
and backbone stride do not define the flow interval. Flow/consistency/camera fit
operate on the FULL resized canvas before pooling patches. Units: canvas px/frame.
Multiple gaps are pooled exactly as the bank builder did, with its validity rule.
No validation/test statistics, prototypes, normalization or clusters are fitted.

Existing raw NPZ flow entries are reused from the bank's recorded namespace or
--reuse-flow-cache-dir overrides. New pairs are written under cache-root/raw_pairs;
existing bank/cache files are never overwritten. Every raw field is quantized to
the bank's recorded storage dtype BEFORE processing, matching bank generation.
Forward/backward fields stay as displacement; processed residual/global velocities
and FB validity masks are cached separately, with pair keys and provenance.

Horizontal flip is an explicit derived view: reverse the pixel lattice and
negate dx in forward/backward and residual/global vector fields; flip validity.
Camera compensation is transformed, not refit on a crop. Histograms, patch means
and cluster targets are recomputed. Original cluster IDs are NEVER simply flipped.
Transformed estimates are stored in a derived namespace, never masquerading as
fresh estimator results for flipped RGB inputs. Other spatial/color augmentations
are not implemented and must not reuse these views as though they were identical.

Training targets use frozen role-routed nearest reference medoids. Matching
original training windows reuse their exact saved labels (including medoid tie
ownership) after content pair keys are verified. New train/val/test/flip windows
are assigned to the SAME fixed references. A role absent from the reference bank
gets label -1 and assignment_valid=False. Static flow is distinct from invalid.

Output: cache-root/SCHEMA/<namespace>/index.sqlite, manifest.json, frozen bank
contract/references, and compressed per-window descriptors. Dense derived pairs
are deduplicated. Reruns are content-addressed and resumable; RGB pair decoding
still checks content. Each invocation records its own run/window membership.
Use the JSONL --window-index option for an exact exported trainer window list:
  {"split":"train","video":"001","window_start":0}
Optional target_index must equal window_start+model_window_len-1. Window length
defaults to the bank window length; --window-len may match a different model
context, provided it contains all the bank's end-of-window flow pairs.
Directory layout is split_root/<video>/images/<naturally sorted RGB files>.

Training usage (no Torch dependency in this module):
  from cache_optical_flow import load_cached_window
  item = load_cached_window('/cache/.../index.sqlite', 'train', '001', 0,
                            augmentation='original', load_dense=False)
  # encoder_inputs: [Hgrid*Wgrid,D], finite float32, row-major patch order.
  # cluster_ids: [N]; assignment_valid/temporal_train_mask select supervised rows.
  # Run E_temporal(item['encoder_inputs']) LIVE during each model forward.
  # No learned embeddings are cached. load_dense=True also returns [G,H,W,2]
  # residual/global velocity, forward/backward displacement and [G,H,W] validity.
Do not pass cluster_ids as encoder inputs. Use exactly the cached augmentation.
The cache does not integrate itself into your dataloader/model automatically.
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
    candidates = (
        [Path(path)]
        if path
        else [
            Path(__file__).with_name("form_temporal_clusters.py"),
            Path(__file__).parent / "patch_clusters/form_temporal_clusters.py",
        ]
    )
    selected = next(
        (p.expanduser().resolve() for p in candidates if p.expanduser().is_file()), None
    )
    if selected is None:
        raise FileNotFoundError(
            "Put this file beside your flow-only form_temporal_clusters.py, or pass --clustering-script."
        )
    spec = importlib.util.spec_from_file_location("_fixed_flow_bank_core", selected)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    if getattr(core, "FORMAT_VERSION", None) != BANK_FORMAT:
        raise ValueError("The clustering helper must be the v2 FLOW-ONLY bank builder.")
    return core, selected


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
        if self.db is None:
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
    """Read old caches, lazily estimate misses into a new create-only cache."""

    def __init__(
        self, bank, core, root, reuse_dirs=(), read_only=False, copy_reused=False
    ):
        self.bank = bank
        self.core = core
        self.estimator = None
        self.directory = (
            Path(root) / "raw_pairs" / core.FLOW_CACHE_SCHEMA / bank.raw_namespace
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self.directory / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != bank.flow_config:
            raise ValueError("Raw output namespace has incompatible metadata.")
        if not manifest.exists():
            atomic_json(manifest, bank.flow_config)
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
                if self.copy_reused:
                    if save_npz_once(
                        destination,
                        forward=fields[0].astype(self.dtype),
                        backward=fields[1].astype(self.dtype),
                        namespace=np.array(self.bank.raw_namespace),
                        pair_sha256=np.array(key),
                    ):
                        self.stats["writes"] += 1
                    self._read(destination, key)
                    path = destination
            return *fields, key, path
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
        if save_npz_once(
            destination,
            forward=f,
            backward=b,
            namespace=np.array(self.bank.raw_namespace),
            pair_sha256=np.array(key),
        ):
            self.stats["writes"] += 1
        return *self._read(destination, key), key, destination


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
        if path.is_file():
            payload = load_npz(path)
            meta = json.loads(str(payload["metadata"]))
            if meta["key"] != key or meta["bank_identity"] != self.bank.identity:
                raise ValueError(f"Derived-pair identity mismatch; preserved: {path}")
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
                raw_path=str(raw_path),
                gap=int(gap),
                augmentation=augmentation,
                camera_model=camera,
                velocity_units="canvas pixels/frame",
                anchor="current frame",
                flip_policy="reverse columns and negate dx; transform camera field and validity; no refit",
            )
            if save_npz_once(
                path,
                residual_velocity=residual.astype(np.float32),
                global_velocity=global_flow.astype(np.float32),
                valid=valid.astype(bool),
                metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
            ):
                self.stats["writes"] += 1
            payload = load_npz(path)
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
        return payload, path


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


def prepare_pairs(paths, start, bank, core, raw):
    target = start + bank.window_len - 1
    current = core.read_canvas(paths[target], bank.canvas)
    return [
        (gap, *raw.get(core.read_canvas(paths[target - gap], bank.canvas), current))
        for gap in bank.gaps
    ]


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
):
    target = start + bank.window_len - 1
    prepared = (
        prepare_pairs(paths, start, bank, core, raw) if prepared is None else prepared
    )
    keys = [p[3] for p in prepared]
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
        if all((directory / p["processed_path"]).is_file() for p in meta["pairs"]):
            return path, key, old, True
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
                raw_path=str(raw_path),
                processed_path=str(pair_path.relative_to(directory)),
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
    )
    if not save_npz_once(
        path,
        metadata=np.array(json.dumps(meta, sort_keys=True, allow_nan=False)),
        **arrays,
    ):
        old = load_npz(path)
        if json.loads(str(old["metadata"]))["cache_key"] != key:
            raise ValueError("Concurrent cache identity mismatch.")
    return path, key, load_npz(path), False


def build(options):
    core, core_path = load_core(options.clustering_script)
    bank = Bank(options.bank_dir, core, options)
    con = None
    try:
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
            raw_cache_directory=str(raw.directory),
            raw_cache_sources=[str(p) for p in raw.sources],
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
                prepared = prepare_pairs(files[(split, video)], start, bank, core, raw)
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
                        prepared=prepared,
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
        help="Copy reused raw entries into output cache for portability; references/bank are never changed.",
    )
    p.add_argument(
        "--raft-weights",
        type=Path,
        help="Local weights required only if a RAFT raw-flow miss must be computed.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="RAFT device if needed; Farneback/DIS remain CPU.",
    )
    p.add_argument("--flow-device", default=None)
    p.add_argument("--threads", type=int, default=4)
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
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
