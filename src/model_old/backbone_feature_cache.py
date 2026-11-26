"""Frozen VideoSwin reference-backbone feature cache.

Caches last-temporal-slice stage features from an independent frozen
``VideoSwinTransformer`` (not the trainable prediction backbone).

Cache root:
  /data/quantization/zaima/videosal_datasets/dhf1k/cached_vis_feats

Layout:
  reference_backbone.pt          fingerprint + frozen weight snapshot
  features/<hh>/<input_sha256>.pt
      content-addressed last-slice feature packs (legacy-compatible)
  v2/<reference_fingerprint>/manifest.json
      versioned namespace describing the active reference + preprocess

Entries are accepted as hits only after validating reference fingerprint,
shapes/channels, finiteness and provenance. Directory presence alone is not
enough. Flipped or differently padded views hash differently and do not collide.

Saved tensors use layout [C_stage, 1, H_stage, W_stage] per sample (no batch
dim). Loaders always return detached float32 tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from model.backbones.video_swin_custom import VideoSwinTransformer

CACHE_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k/cached_vis_feats")
DEFAULT_REFERENCE_CHECKPOINT = VideoSwinTransformer.DEFAULT_PRETRAINED_PATH
SCHEMA = "frozen_backbone_last_slice_v2"
SCHEMA_VERSION = 2
LEGACY_ENTRY_VERSION = 1
STAGES = ("stage1", "stage2", "stage3", "stage4")
STAGE_CHANNELS = {
    "stage1": 96,
    "stage2": 192,
    "stage3": 384,
    "stage4": 768,
}
DEFAULT_RESIZE_TO = (224, 384)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
# Fingerprint string used by the existing features/*.pt packs for the default
# Kinetics-pretrained VideoSwin weights. Kept so compatible entries remain hits
# even when our hashing recipe differs from the original writer.
LEGACY_FEATURES_REFERENCE_SHA256 = (
    "07311026a139e72085344fdc6686b9a6d9ee9af700de268b5483a734dc74acfd"
)


def ensure_cache_root(root: Union[str, Path] = CACHE_ROOT) -> Path:
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "features").mkdir(parents=True, exist_ok=True)
    (path / "v2").mkdir(parents=True, exist_ok=True)
    return path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(
            json.dumps(_json_ready(obj), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    """Canonical SHA-256 of a backbone (or module) state_dict.

    Hashes sorted keys with float32 CPU bytes. This is the identity used for
    reference-feature provenance — not a whole-model checkpoint file hash.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            continue
        array = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


REFERENCE_FEATURE_PROVENANCE_SCHEMA = "reference_backbone_features_v1"


def extract_backbone_state_dict_from_checkpoint(
    checkpoint_path: Union[str, Path],
) -> Dict[str, torch.Tensor]:
    """Load backbone tensors from a Kinetics .pth or full model checkpoint.

    Full ExplainableVidSalModel checkpoints store keys under ``backbone.``.
    Native VideoSwin Kinetics packs store SwinTransformer3D keys (possibly under
    ``state_dict`` / ``model``). Returns a VideoSwinTransformer.state_dict-shaped
    mapping (keys prefixed with ``backbone.``) suitable for fingerprinting.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and any(
        torch.is_tensor(v) for v in payload.values()
    ) and not any(k in payload for k in ("model_state_dict", "state_dict", "model")):
        raw = payload
        source = "raw_state_dict"
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        raw = payload["model_state_dict"]
        source = "model_state_dict"
    elif isinstance(payload, dict) and "state_dict" in payload:
        raw = payload["state_dict"]
        source = "state_dict"
    elif isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        raw = payload["model"]
        source = "model"
    elif isinstance(payload, dict):
        raw = payload
        source = "dict"
    else:
        raise ValueError(f"Unrecognized checkpoint format at {path}")

    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint {path} does not contain a state_dict mapping.")

    # Prefer full-model backbone.* keys when present.
    backbone_prefixed = {
        k: v for k, v in raw.items()
        if isinstance(k, str) and k.startswith("backbone.") and torch.is_tensor(v)
    }
    if backbone_prefixed:
        return dict(backbone_prefixed)

    # Kinetics / Swin3D packs: wrap as VideoSwinTransformer.state_dict keys.
    inner = {
        k: v for k, v in raw.items()
        if torch.is_tensor(v) and isinstance(k, str)
    }
    if not inner:
        raise ValueError(
            f"No tensor weights found in {path} (parsed as {source})."
        )
    return {f"backbone.{k}": v for k, v in inner.items()}


def backbone_fingerprint_from_checkpoint(checkpoint_path: Union[str, Path]) -> str:
    """Fingerprint backbone weights as they appear after the same load path as the cache.

    Native Kinetics packs remapping keys inside ``SwinTransformer3D.init_weights``.
    Fingerprinting raw file tensors therefore disagrees with
    ``FrozenBackboneFeatureCache.reference_fingerprint``. This helper loads
    weights the same way the cache does (or injects ``backbone.*`` from a full
    model checkpoint into that module) and hashes the resulting state_dict.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Preferred path: identical to FrozenBackboneFeatureCache construction.
    try:
        reference = build_reference_backbone(checkpoint=str(path), device="cpu")
        return fingerprint_state_dict(reference.state_dict())
    except Exception:
        pass
    # Full ExplainableVidSalModel checkpoint: copy backbone.* into a scaffold.
    backbone_state = extract_backbone_state_dict_from_checkpoint(path)
    reference = build_reference_backbone(
        checkpoint=DEFAULT_REFERENCE_CHECKPOINT, device="cpu"
    )
    incompatible = reference.load_state_dict(backbone_state, strict=False)
    if incompatible.missing_keys:
        raise ValueError(
            f"Could not inject backbone weights from {path} into VideoSwinTransformer; "
            f"missing={incompatible.missing_keys[:8]}"
        )
    return fingerprint_state_dict(reference.state_dict())


def build_reference_feature_provenance(
    *,
    backbone_fingerprint: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    weight_scope: str,
    preprocess: Optional[Mapping[str, Any]] = None,
    temporal_selection: str = "last_backbone_step",
    feature_module: str = "backbone",
    feature_output_path: Sequence[Any] = ("stage4",),
    feature_layout: str = "BCTHW_last_slice",
    spatial_grid: str = "native_stage4_HxW_matching_concept_grid",
    verified: bool = True,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Structured provenance for fixed-cluster / cache feature agreement."""
    payload: Dict[str, Any] = {
        "schema": REFERENCE_FEATURE_PROVENANCE_SCHEMA,
        "backbone_class": "model.backbones.video_swin_custom.VideoSwinTransformer",
        "backbone_fingerprint": str(backbone_fingerprint),
        "weight_scope": str(weight_scope),
        "temporal_selection": str(temporal_selection),
        "feature_module": str(feature_module),
        "feature_output_path": list(feature_output_path),
        "feature_layout": str(feature_layout),
        "spatial_grid": str(spatial_grid),
        "preprocess": dict(preprocess or preprocess_config()),
        "verified": bool(verified),
        # Whole-file hashes are informational only; never treat them as the
        # backbone identity used for training/cache agreement.
        "source_checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "source_checkpoint_sha256": (
            None if checkpoint_sha256 is None else str(checkpoint_sha256)
        ),
        "checkpoint_file_hash_is_not_backbone_identity": True,
    }
    if notes:
        payload["notes"] = str(notes)
    return payload


def provenance_from_cache(cache: "FrozenBackboneFeatureCache") -> Dict[str, Any]:
    """Build provenance describing the active FrozenBackboneFeatureCache reference."""
    return build_reference_feature_provenance(
        backbone_fingerprint=cache.reference_fingerprint,
        checkpoint_path=cache.checkpoint_path,
        checkpoint_sha256=cache.checkpoint_sha256,
        weight_scope="FrozenBackboneFeatureCache.reference.state_dict",
        preprocess=cache.preprocess,
        temporal_selection="last_backbone_step",
        feature_module="backbone",
        feature_output_path=list(STAGES),
        feature_layout="BCTHW_last_slice_per_stage",
        spatial_grid="native_stage_resolutions_after_resize_to",
        verified=True,
        notes=(
            "Training queries must use this fingerprint. Flipped/padded views are "
            "content-addressed separately; never reuse identity-view features for hflip."
        ),
    )


def compare_reference_feature_provenance(
    cluster_provenance: Optional[Mapping[str, Any]],
    cache_provenance: Mapping[str, Any],
    *,
    context: str = "stage4 fixed-cluster feature supervision",
) -> None:
    """Raise if cluster and cache reference features cannot be verified as identical.

    Missing provenance ⇒ unverified (do not invent agreement from feature dim).
    Present but disagreeing fingerprints/preprocess ⇒ hard mismatch.
    """
    if not cluster_provenance or not isinstance(cluster_provenance, Mapping):
        raise RuntimeError(
            f"{context}: reference-feature provenance is UNVERIFIED.\n"
            "The fixed patch-cluster bank does not record a backbone_fingerprint "
            "(state_dict hash). A whole-model checkpoint file SHA alone is not "
            "enough — it may include non-backbone weights and cannot be compared "
            "to FrozenBackboneFeatureCache.reference_fingerprint.\n"
            "Feature dimension matching is not sufficient.\n\n"
            "Regenerate the fixed cluster bank with recorded provenance, using the "
            "SAME frozen reference backbone as training (Kinetics / "
            f"{DEFAULT_REFERENCE_CHECKPOINT}):\n"
            "  Recommended (no prototype graph):\n"
            "    python patch_clusters/form_patch_clusters.py --source-mode cache "
            "--output-dir /data/quantization/zaima/videosal_datasets/dhf1k/"
            "patch_cluster_frozen_ref/patch_cluster_level0 --stride 1 --num-prototypes 256\n"
            "  Then train with:\n"
            "    python train.py --phase initial --visual-cluster-dir "
            "/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster_frozen_ref/patch_cluster_level0\n"
            "  Optional legacy path: rebuild graph descriptors with "
            "DESCRIPTOR_FEATURE_SOURCE='frozen_reference', then "
            "form_patch_clusters.py --source-mode graph --source-db <graph.sqlite> "
            "into a NEW output dir. Do not overwrite existing banks/caches."
        )
    cluster_fp = cluster_provenance.get("backbone_fingerprint")
    cache_fp = cache_provenance.get("backbone_fingerprint")
    if not cluster_fp or not cache_fp:
        raise RuntimeError(
            f"{context}: provenance present but backbone_fingerprint missing on "
            f"cluster={bool(cluster_fp)} cache={bool(cache_fp)}. Regenerate the "
            "cluster bank / rebuild the cache with reference_feature_provenance."
        )
    if str(cluster_fp) != str(cache_fp):
        raise RuntimeError(
            f"{context}: CONFIRMED MISMATCH between fixed-cluster features and "
            "FrozenBackboneFeatureCache.\n"
            f"  cluster backbone_fingerprint={cluster_fp}\n"
            f"  cache   backbone_fingerprint={cache_fp}\n"
            f"  cluster checkpoint={cluster_provenance.get('source_checkpoint_path')}\n"
            f"  cache   checkpoint={cache_provenance.get('source_checkpoint_path')}\n"
            "Do not compare whole-model checkpoint file hashes to these fingerprints.\n"
            "Fix: rebuild EITHER the graph descriptors+clusters OR the feature cache "
            "so both use one fixed reference backbone (recommended: Kinetics "
            f"{DEFAULT_REFERENCE_CHECKPOINT}). Do not overwrite existing banks/"
            "caches — write to a new directory / new fingerprint namespace.\n"
            "Example:\n"
            "  python cache_backbone_features.py --split train "
            f"--reference-checkpoint {DEFAULT_REFERENCE_CHECKPOINT}\n"
            "  # and regenerate clusters from a graph whose "
            "reference_feature_provenance.backbone_fingerprint matches the cache."
        )
    # Preprocess must also agree when both sides declare it.
    c_pre = cluster_provenance.get("preprocess") or {}
    q_pre = cache_provenance.get("preprocess") or {}
    for key in ("resize_to", "normalize", "input_format", "mean", "std"):
        if key in c_pre and key in q_pre and c_pre[key] != q_pre[key]:
            raise RuntimeError(
                f"{context}: backbone fingerprints match but preprocess[{key!r}] "
                f"differs (cluster={c_pre[key]!r}, cache={q_pre[key]!r}). "
                "Align resize/normalize/mean/std before training."
            )
    c_temp = cluster_provenance.get("temporal_selection")
    q_temp = cache_provenance.get("temporal_selection")
    if c_temp and q_temp and c_temp != q_temp:
        raise RuntimeError(
            f"{context}: temporal_selection differs "
            f"(cluster={c_temp!r}, cache={q_temp!r})."
        )


def fingerprint_rgb_clip(rgb_btchw: torch.Tensor) -> str:
    """Content fingerprint of a single BTCHW float clip (pre-backbone resize)."""
    if rgb_btchw.dim() != 4 or rgb_btchw.shape[1] != 3:
        raise ValueError(
            "rgb clip must have shape [T,3,H,W], "
            f"got {tuple(rgb_btchw.shape)}"
        )
    array = (
        rgb_btchw.detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .numpy()
    )
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fingerprint_identity_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def preprocess_config(
    *,
    resize_to: Optional[Union[int, Tuple[int, int]]] = DEFAULT_RESIZE_TO,
    normalize: bool = True,
    input_format: str = "BTCHW",
    output_format: str = "BCTHW",
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> Dict[str, Any]:
    if isinstance(resize_to, int):
        resize_value: Any = int(resize_to)
    elif resize_to is None:
        resize_value = None
    else:
        resize_value = [int(resize_to[0]), int(resize_to[1])]
    return {
        "input_format": str(input_format),
        "output_format": str(output_format),
        "resize_to": resize_value,
        "normalize": bool(normalize),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "gradient_checkpointing": False,
        "freeze_backbone": True,
        "return_stages": list(STAGES),
    }


@dataclass(frozen=True)
class SampleIdentity:
    """Exact sample/view identity for cache keys.

    Training uses random clip starts, so video ID alone is insufficient.
    ``content_sha256`` fingerprints the exact RGB clip (after view transforms,
    before backbone resize/normalization). Different flips/crops/pads must not
    share an entry unless their content fingerprints match.
    """

    dataset_root: str
    video_id: str
    window_start: int
    frame_names: Tuple[str, ...]
    window_len: int
    content_sha256: str
    split: Optional[str] = None
    temporal_sampling: Mapping[str, Any] = field(
        default_factory=lambda: {"mode": "contiguous", "stride": 1}
    )
    view_id: str = "identity"
    input_shape: Optional[Tuple[int, int, int, int]] = None
    padded_input_geometry: Optional[Mapping[str, Any]] = None
    preprocess: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_root": str(Path(self.dataset_root).expanduser().resolve()),
            "split": self.split,
            "video_id": str(self.video_id),
            "window_start": int(self.window_start),
            "frame_names": list(self.frame_names),
            "window_len": int(self.window_len),
            "temporal_sampling": dict(self.temporal_sampling),
            "view_id": str(self.view_id),
            "content_sha256": str(self.content_sha256),
            "input_shape": list(self.input_shape) if self.input_shape is not None else None,
            "padded_input_geometry": (
                dict(self.padded_input_geometry)
                if self.padded_input_geometry is not None
                else None
            ),
            "preprocess": dict(self.preprocess) if self.preprocess is not None else None,
        }

    def identity_sha256(self) -> str:
        return fingerprint_identity_payload(self.to_dict())


def build_reference_backbone(
    *,
    checkpoint: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    resize_to: Optional[Union[int, Tuple[int, int]]] = DEFAULT_RESIZE_TO,
    normalize: bool = True,
) -> VideoSwinTransformer:
    """Independent frozen VideoSwin used only for cache generation."""
    ckpt = DEFAULT_REFERENCE_CHECKPOINT if checkpoint is None else str(checkpoint)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Reference checkpoint not found: {ckpt}")
    backbone = VideoSwinTransformer(
        pretrained=ckpt,
        pretrained_path=ckpt,
        freeze_backbone=True,
        return_stages=STAGES,
        input_format="BTCHW",
        output_format="BCTHW",
        resize_to=resize_to,
        normalize=normalize,
        gradient_checkpointing=False,
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if device is not None:
        backbone = backbone.to(device)
    return backbone


def _states_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a = left[key]
        b = right[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.equal(a.cpu(), b.cpu()):
            return False
    return True


def _feature_path(root: Path, content_sha256: str) -> Path:
    return root / "features" / content_sha256[:2] / f"{content_sha256}.pt"


def _validate_feature_pack(
    features: Mapping[str, torch.Tensor],
    *,
    storage_dtype: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    if set(features) != set(STAGES):
        raise ValueError(
            f"Feature pack stages {sorted(features)} != expected {list(STAGES)}"
        )
    out: Dict[str, torch.Tensor] = {}
    for stage in STAGES:
        tensor = features[stage]
        if not torch.is_tensor(tensor):
            raise TypeError(f"{stage} feature must be a tensor.")
        if tensor.dim() != 4:
            raise ValueError(
                f"{stage} must be [C,1,H,W], got {tuple(tensor.shape)}"
            )
        channels, temporal, _, _ = tensor.shape
        if channels != STAGE_CHANNELS[stage]:
            raise ValueError(
                f"{stage} expected {STAGE_CHANNELS[stage]} channels, got {channels}"
            )
        if temporal != 1:
            raise ValueError(
                f"{stage} must keep only the last temporal slice "
                f"(T=1), got T={temporal}"
            )
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"{stage} contains non-finite values.")
        if storage_dtype is not None:
            expected = {"float16": torch.float16, "float32": torch.float32}[storage_dtype]
            if tensor.dtype != expected:
                # Allow float32 legacy packs even when requesting float16 storage
                # only at write time; loaders cast to float32 for consumers.
                if tensor.dtype not in (torch.float16, torch.float32):
                    raise ValueError(
                        f"{stage} dtype {tensor.dtype} is unsupported."
                    )
        out[stage] = tensor.detach()
    return out


class FrozenBackboneFeatureCache:
    """Load/generate last-slice reference features outside Dataset workers."""

    def __init__(
        self,
        *,
        root: Union[str, Path] = CACHE_ROOT,
        reference_backbone: Optional[VideoSwinTransformer] = None,
        checkpoint: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        storage_dtype: str = "float16",
        resize_to: Optional[Union[int, Tuple[int, int]]] = DEFAULT_RESIZE_TO,
        normalize: bool = True,
        create_reference_if_missing: bool = True,
    ):
        if storage_dtype not in ("float16", "float32"):
            raise ValueError("storage_dtype must be 'float16' or 'float32'")
        self.root = ensure_cache_root(root)
        self.storage_dtype = storage_dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.preprocess = preprocess_config(
            resize_to=resize_to,
            normalize=normalize,
        )
        self.checkpoint_path = str(
            Path(
                DEFAULT_REFERENCE_CHECKPOINT if checkpoint is None else checkpoint
            ).expanduser().resolve()
        )
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

        if reference_backbone is None:
            reference_backbone = build_reference_backbone(
                checkpoint=self.checkpoint_path,
                device=self.device,
                resize_to=resize_to,
                normalize=normalize,
            )
        else:
            reference_backbone = reference_backbone.to(self.device)
            reference_backbone.eval()
            for parameter in reference_backbone.parameters():
                parameter.requires_grad = False

        self.reference = reference_backbone
        self.reference_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.reference.state_dict().items()
            if torch.is_tensor(value)
        }
        self.reference_fingerprint = fingerprint_state_dict(self.reference_state)
        self.legacy_reference_sha256 = self._adopt_legacy_reference_fingerprint()
        self.namespace_dir = self.root / "v2" / self.reference_fingerprint
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.namespace_dir / "manifest.json"
        self.identity_index_path = self.namespace_dir / "identity_index.jsonl"
        if create_reference_if_missing:
            self._write_reference_snapshot()
            self._write_manifest()

        self.stats = {
            "hits": 0,
            "misses": 0,
            "regenerated": 0,
            "legacy_hits": 0,
            "writes": 0,
            "rejected": 0,
            "reference_forwards": 0,
        }

    def _adopt_legacy_reference_fingerprint(self) -> Optional[str]:
        """Find the reference hash string used by reusable features/*.pt packs."""
        path = self.root / "reference_backbone.pt"
        candidate_hashes: List[str] = []
        if path.is_file():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                payload = None
            if isinstance(payload, dict) and "state" in payload:
                legacy_state = {
                    key: value.detach().cpu().contiguous()
                    for key, value in payload["state"].items()
                    if torch.is_tensor(value)
                }
                shared = {
                    key: legacy_state[key]
                    for key in legacy_state
                    if key in self.reference_state
                }
                current = {key: self.reference_state[key] for key in shared}
                if shared and _states_equal(shared, current):
                    missing = set(self.reference_state) - set(shared)
                    if not any(key.startswith("backbone.") for key in missing):
                        sha = payload.get("state_sha256")
                        if sha:
                            candidate_hashes.append(str(sha))
                        cfg = payload.get("config") or {}
                        legacy_cfg = cfg.get("legacy_reference_sha256")
                        if legacy_cfg:
                            candidate_hashes.append(str(legacy_cfg))
        # Default Kinetics checkpoint packs in this repo use a known legacy hash.
        if Path(self.checkpoint_path).resolve() == Path(
            DEFAULT_REFERENCE_CHECKPOINT
        ).resolve():
            candidate_hashes.append(LEGACY_FEATURES_REFERENCE_SHA256)
        # Prefer the historical features/*.pt hash when present.
        if LEGACY_FEATURES_REFERENCE_SHA256 in candidate_hashes:
            return LEGACY_FEATURES_REFERENCE_SHA256
        return candidate_hashes[0] if candidate_hashes else None

    def _reference_config(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "tag": "v2",
            "backbone_class": "model.backbones.video_swin_custom.VideoSwinTransformer",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "reference_fingerprint": self.reference_fingerprint,
            "legacy_reference_sha256": self.legacy_reference_sha256,
            "stages": list(STAGES),
            "stage_channels": dict(STAGE_CHANNELS),
            "temporal_selection": "last_backbone_step",
            "storage_dtype": self.storage_dtype,
            "preprocess": self.preprocess,
            "concept_grid_caps": {stage: None for stage in STAGES},
        }

    def _write_reference_snapshot(self) -> None:
        """Write namespace snapshot; never clobber a compatible root snapshot."""
        state = dict(self.reference_state)
        if hasattr(self.reference, "mean"):
            state["mean"] = self.reference.mean.detach().cpu().contiguous()
        if hasattr(self.reference, "std"):
            state["std"] = self.reference.std.detach().cpu().contiguous()
        payload = {
            "config": {
                **self._reference_config(),
                "input_format": self.preprocess["input_format"],
                "normalize": self.preprocess["normalize"],
                "resize_to": self.preprocess["resize_to"],
                "reference_precision": "float32",
            },
            "state": state,
            # Prefer the legacy features/*.pt hash so new writes remain compatible.
            "state_sha256": self.legacy_reference_sha256 or self.reference_fingerprint,
        }
        _atomic_torch_save(payload, self.namespace_dir / "reference_snapshot.pt")

        root_path = self.root / "reference_backbone.pt"
        if root_path.is_file():
            # Do not overwrite existing root snapshot bytes. Compatibility is
            # handled via accepted reference hashes, not by rewriting this file.
            return
        _atomic_torch_save(payload, root_path)

    def _write_manifest(self) -> None:
        manifest = {
            **self._reference_config(),
            "cache_root": str(self.root),
            "features_dir": str(self.root / "features"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": (
                "Features are raw frozen-backbone outputs for the last temporal "
                "slice only: [C,1,H,W] per stage. No L2-normalization or projection. "
                "Exhaustive stride-1 caching of all four stages is large."
            ),
        }
        _atomic_write_json(manifest, self.manifest_path)

    def _accepted_reference_hashes(self) -> set[str]:
        hashes = {self.reference_fingerprint}
        if self.legacy_reference_sha256:
            hashes.add(self.legacy_reference_sha256)
        return hashes

    def _cast_for_storage(self, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dtype = torch.float16 if self.storage_dtype == "float16" else torch.float32
        return {
            stage: features[stage].detach().to(dtype=dtype, device="cpu").contiguous()
            for stage in STAGES
        }

    def _features_to_float32(
        self, features: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            stage: features[stage].detach().to(dtype=torch.float32).contiguous()
            for stage in STAGES
        }

    def entry_path(self, content_sha256: str) -> Path:
        return _feature_path(self.root, content_sha256)

    def load_entry(
        self,
        content_sha256: str,
        *,
        require_identity: Optional[SampleIdentity] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        path = self.entry_path(content_sha256)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                self.stats["rejected"] += 1
                return None
        except Exception:
            self.stats["rejected"] += 1
            return None
        if not isinstance(payload, dict):
            self.stats["rejected"] += 1
            return None
        try:
            if str(payload.get("input_sha256")) != str(content_sha256):
                raise ValueError("content hash mismatch inside entry")
            if str(payload.get("reference_sha256")) not in self._accepted_reference_hashes():
                raise ValueError("reference fingerprint mismatch")
            features = _validate_feature_pack(payload.get("features") or {})
            if require_identity is not None:
                stored = payload.get("identity")
                if not isinstance(stored, dict):
                    # Legacy packs lack rich identity; content+reference match is enough.
                    pass
                else:
                    expected = require_identity.to_dict()
                    for key in (
                        "dataset_root",
                        "video_id",
                        "window_start",
                        "frame_names",
                        "window_len",
                        "view_id",
                        "content_sha256",
                    ):
                        if key in stored and stored[key] != expected[key]:
                            raise ValueError(f"identity field mismatch: {key}")
        except Exception:
            self.stats["rejected"] += 1
            return None

        version = int(payload.get("version", LEGACY_ENTRY_VERSION))
        if version == LEGACY_ENTRY_VERSION:
            self.stats["legacy_hits"] += 1
        self.stats["hits"] += 1
        return self._features_to_float32(features)

    def save_entry(
        self,
        content_sha256: str,
        features: Mapping[str, torch.Tensor],
        *,
        input_shape: Sequence[int],
        identity: Optional[SampleIdentity] = None,
        overwrite: bool = False,
    ) -> Path:
        path = self.entry_path(content_sha256)
        if path.is_file() and not overwrite:
            # Never clobber an on-disk pack unless explicitly forced. Incompatible
            # legacy/other-reference files remain untouched for later reuse.
            existing = self.load_entry(content_sha256, require_identity=identity)
            if existing is not None and identity is not None:
                self._append_identity_index(identity, content_sha256)
            return path
        validated = _validate_feature_pack(features)
        stored = self._cast_for_storage(validated)
        payload = {
            "version": SCHEMA_VERSION,
            "schema": SCHEMA,
            "input_sha256": content_sha256,
            "reference_sha256": self.legacy_reference_sha256 or self.reference_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "input_shape": [int(x) for x in input_shape],
            "input_dtype": "torch.float32",
            "storage_dtype": self.storage_dtype,
            "features": stored,
            "identity": identity.to_dict() if identity is not None else None,
            "provenance": {
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "preprocess": self.preprocess,
                "temporal_selection": "last_backbone_step",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        _atomic_torch_save(payload, path)
        if identity is not None:
            self._append_identity_index(identity, content_sha256)
        self.stats["writes"] += 1
        return path

    def _append_identity_index(self, identity: SampleIdentity, content_sha256: str) -> None:
        record = {
            "identity_sha256": identity.identity_sha256(),
            "content_sha256": content_sha256,
            "identity": identity.to_dict(),
        }
        line = json.dumps(_json_ready(record), sort_keys=True) + "\n"
        with open(self.identity_index_path, "a", encoding="utf-8") as handle:
            handle.write(line)

    @torch.no_grad()
    def compute_last_slice_features(
        self,
        rgb_btchw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run frozen reference backbone; keep only the last temporal slice."""
        if rgb_btchw.dim() == 4:
            video = rgb_btchw.unsqueeze(0)
        elif rgb_btchw.dim() == 5:
            video = rgb_btchw
        else:
            raise ValueError(
                "Expected [T,3,H,W] or [B,T,3,H,W] RGB, "
                f"got {tuple(rgb_btchw.shape)}"
            )
        if video.shape[2] != 3:
            raise ValueError(
                f"Expected BTCHW with 3 channels, got {tuple(video.shape)}"
            )
        video = video.detach().to(device=self.device, dtype=torch.float32)
        if video.numel() > 0 and float(video.max()) > 2.0:
            video = video / 255.0
        self.stats["reference_forwards"] += 1
        outputs = self.reference.forward_features(video)
        packed: Dict[str, torch.Tensor] = {}
        for stage in STAGES:
            feat = outputs[stage]
            if feat.dim() != 5:
                raise ValueError(
                    f"{stage} expected [B,C,T,H,W], got {tuple(feat.shape)}"
                )
            last = feat[:, :, -1:, :, :].detach()
            if last.shape[0] != 1:
                raise ValueError(
                    "compute_last_slice_features expects a single sample; "
                    "use resolve_batch for batched inputs."
                )
            packed[stage] = last[0].to(dtype=torch.float32, device="cpu").contiguous()
        return _validate_feature_pack(packed)

    def get_or_compute(
        self,
        rgb_btchw: torch.Tensor,
        *,
        identity: Optional[SampleIdentity] = None,
        force: bool = False,
        allow_compute: bool = True,
        _verified_content_sha256: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        clip = rgb_btchw.detach()
        if clip.dim() == 5:
            if clip.shape[0] != 1:
                raise ValueError("Pass one sample to get_or_compute, or use resolve_batch.")
            clip = clip[0]
        if clip.dim() != 4 or clip.shape[1] != 3:
            raise ValueError(f"Expected [T,3,H,W], got {tuple(clip.shape)}")
        clip = clip.to(dtype=torch.float32)
        if clip.numel() > 0 and float(clip.max()) > 2.0:
            clip = clip / 255.0
        content_sha = (_verified_content_sha256 if _verified_content_sha256 is not None
                       else fingerprint_rgb_clip(clip))
        if identity is not None and identity.content_sha256 != content_sha:
            raise ValueError(
                "SampleIdentity.content_sha256 does not match the provided RGB clip."
            )
        if identity is None:
            identity = SampleIdentity(
                dataset_root="unknown",
                video_id="unknown",
                window_start=-1,
                frame_names=tuple(),
                window_len=int(clip.shape[0]),
                content_sha256=content_sha,
                view_id="content_only",
                input_shape=tuple(int(x) for x in clip.shape),
                preprocess=self.preprocess,
            )
        elif identity.input_shape is None or identity.preprocess is None:
            identity = SampleIdentity(
                dataset_root=identity.dataset_root,
                video_id=identity.video_id,
                window_start=identity.window_start,
                frame_names=tuple(identity.frame_names),
                window_len=identity.window_len,
                content_sha256=content_sha,
                split=identity.split,
                temporal_sampling=dict(identity.temporal_sampling),
                view_id=identity.view_id,
                input_shape=tuple(int(x) for x in clip.shape),
                padded_input_geometry=identity.padded_input_geometry,
                preprocess=dict(identity.preprocess or self.preprocess),
            )

        if not force:
            hit = self.load_entry(content_sha, require_identity=identity)
            if hit is not None:
                return hit

        self.stats["misses"] += 1
        if not allow_compute:
            raise FileNotFoundError(
                f"Missing compatible cached reference features for "
                f"{content_sha[:16]}… (read-only cache mode)."
            )
        features = self.compute_last_slice_features(clip)
        path = self.entry_path(content_sha)
        if force and path.is_file():
            self.stats["regenerated"] += 1
        if path.is_file() and not force:
            # Compatible validation failed (e.g. unexpected corruption). Keep the
            # file and return freshly computed features without destroying cache.
            return self._features_to_float32(features)
        self.save_entry(
            content_sha,
            features,
            input_shape=tuple(int(x) for x in clip.shape),
            identity=identity,
            overwrite=force,
        )
        return self._features_to_float32(features)

    def resolve_batch(
        self,
        rgb_batch: torch.Tensor,
        *,
        identities: Optional[Sequence[SampleIdentity]] = None,
        n_frames: Optional[torch.Tensor] = None,
        content_hw: Optional[Sequence[Tuple[int, int]]] = None,
        view_id: Union[str, Sequence[str]] = "identity",
        force: bool = False,
        allow_compute: bool = True,
        verify_identity_content: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Resolve a BTCHW batch into stage tensors [B,C,1,H,W].

        Uses each sample's exact view content. Spatial padding is stripped via
        ``content_hw`` when provided; otherwise the full H,W of the batch tensor
        is used (callers should avoid accidental pad collisions). Temporal pad is
        stripped with ``n_frames``.

        Hash once by default. Internal training callers can set
        verify_identity_content=False only when identities were just hashed
        from the exact supplied CPU clips. Direct get_or_compute calls still
        verify input content normally.
        """
        if rgb_batch.dim() != 5 or rgb_batch.shape[2] != 3:
            raise ValueError(
                f"rgb_batch must be [B,T,3,H,W], got {tuple(rgb_batch.shape)}"
            )
        batch_size = int(rgb_batch.shape[0])
        if identities is not None and len(identities) != batch_size:
            raise ValueError("identities length must match batch size.")
        if content_hw is not None and len(content_hw) != batch_size:
            raise ValueError("content_hw length must match batch size.")
        if isinstance(view_id, str):
            view_ids = [view_id] * batch_size
        else:
            view_ids = list(view_id)
            if len(view_ids) != batch_size:
                raise ValueError("view_id sequence length must match batch size.")

        per_sample: List[Dict[str, torch.Tensor]] = []
        for index in range(batch_size):
            frames = int(n_frames[index]) if n_frames is not None else int(rgb_batch.shape[1])
            if content_hw is not None:
                height, width = int(content_hw[index][0]), int(content_hw[index][1])
            else:
                height, width = int(rgb_batch.shape[-2]), int(rgb_batch.shape[-1])
            clip = rgb_batch[index, :frames, :, :height, :width].detach().to(dtype=torch.float32)
            if clip.numel() > 0 and float(clip.max()) > 2.0:
                clip = clip / 255.0
            if identities is not None:
                identity = identities[index]
                content_sha = (fingerprint_rgb_clip(clip) if verify_identity_content
                               else identity.content_sha256)
                if content_sha != identity.content_sha256:
                    raise ValueError("SampleIdentity.content_sha256 does not match the provided RGB clip.")
            else:
                content_sha = fingerprint_rgb_clip(clip)
                identity = SampleIdentity(
                    dataset_root="batch",
                    video_id=f"batch_{index}",
                    window_start=-1,
                    frame_names=tuple(),
                    window_len=frames,
                    content_sha256=content_sha,
                    view_id=str(view_ids[index]),
                    input_shape=tuple(int(x) for x in clip.shape),
                    padded_input_geometry={
                        "batch_shape": list(rgb_batch.shape),
                        "n_frames": frames,
                        "content_hw": [height, width],
                    },
                    preprocess=self.preprocess,
                )
            per_sample.append(
                self.get_or_compute(
                    clip,
                    identity=identity,
                    force=force,
                    allow_compute=allow_compute,
                    _verified_content_sha256=content_sha,
                )
            )

        batched: Dict[str, torch.Tensor] = {}
        for stage in STAGES:
            batched[stage] = torch.stack(
                [sample[stage] for sample in per_sample], dim=0
            ).contiguous()
        return batched

    def estimate_entry_bytes(self, sample_hw: Tuple[int, int] = (360, 640)) -> int:
        """Rough per-window storage estimate for all four last-slice stages."""
        # Feature maps after resize_to=(224,384): stage grids 56x96 ... 7x12.
        grids = {
            "stage1": (96, 56, 96),
            "stage2": (192, 28, 48),
            "stage3": (384, 14, 24),
            "stage4": (768, 7, 12),
        }
        itemsize = 2 if self.storage_dtype == "float16" else 4
        total = 0
        for channels, height, width in grids.values():
            total += channels * 1 * height * width * itemsize
        # Ignore identity metadata overhead beyond a small constant.
        return total + 2048


def make_identity_from_loader_sample(
    *,
    dataset_root: str,
    video_id: str,
    window_start: int,
    frame_names: Sequence[str],
    rgb_thwc_or_tchw: torch.Tensor,
    split: Optional[str] = None,
    stride: int = 1,
    view_id: str = "identity",
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Tuple[SampleIdentity, torch.Tensor]:
    """Build identity + BTCHW clip from a DatasetLoader RGB sample."""
    rgb = rgb_thwc_or_tchw.detach()
    if not torch.is_tensor(rgb):
        rgb = torch.as_tensor(rgb)
    rgb = rgb.to(dtype=torch.float32)
    if rgb.dim() != 4:
        raise ValueError(f"Expected 4D RGB clip, got {tuple(rgb.shape)}")
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
    elif rgb.shape[1] != 3:
        raise ValueError(f"Unsupported RGB layout {tuple(rgb.shape)}")
    if rgb.numel() > 0 and float(rgb.max()) > 2.0:
        rgb = rgb / 255.0
    content_sha = fingerprint_rgb_clip(rgb)
    identity = SampleIdentity(
        dataset_root=str(Path(dataset_root).expanduser().resolve()),
        split=split,
        video_id=str(video_id),
        window_start=int(window_start),
        frame_names=tuple(str(name) for name in frame_names),
        window_len=int(rgb.shape[0]),
        temporal_sampling={"mode": "contiguous", "stride": int(stride)},
        view_id=str(view_id),
        content_sha256=content_sha,
        input_shape=tuple(int(x) for x in rgb.shape),
        preprocess=dict(preprocess or preprocess_config()),
    )
    return identity, rgb

