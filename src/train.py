import argparse
import os

# Reduce CUDA fragmentation (must be set before the first CUDA allocation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import sys
import math
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from pre_process.collate import video_saliency_collate_fn
from model.losses import compute_total_loss
from model.metrics import (
    MetricAverager,
    compute_saliency_metrics,
    compute_saliency_metrics_excluding_empty_fixations,
)
from metrics import prepare_target_last_map
from pre_process.dataloader import DatasetLoader
from model.model import (
    ExplainableVidSalModel,
    DEFAULT_PATCH_CLUSTER_DIR,
    VisualConceptCreation,
    build_patch_cluster_reference_checkpoint_metadata,
    CACHE_ROOT as DEFAULT_BACKBONE_FEATURE_CACHE_ROOT,
    DEFAULT_REFERENCE_CHECKPOINT,
    FrozenBackboneFeatureCache,
    SampleIdentity,
    compare_reference_feature_provenance,
    fingerprint_rgb_clip,
    provenance_from_cache,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Hide i8 when stdout is redirected (e.g. nohup ... > out.log).
SHOW_PROGRESS_BAR = sys.stdout.isatty()

TRAIN_DATASET_DIR = (
    # "/data/quantization/zaima/videosal_datasets/hollywood2/videos/training"
    # "/data/quantization/zaima/videosal_datasets/ucfsports/training"
    "/data/quantization/zaima/videosal_datasets/dhf1k/train"
)
VAL_DATASET_DIR = (
    # "/data/quantization/zaima/videosal_datasets/hollywood2/videos/testing"
    # "/data/quantization/zaima/videosal_datasets/ucfsports/testing"
    "/data/quantization/zaima/videosal_datasets/dhf1k/val"
)
WINDOW_LEN = 32

INITIAL_EPOCHS = 200
FINETUNE_EPOCHS = 200
RESUME_EPOCHS = 100
BATCH_SIZE = 4  # effective optimizer batch size
FREEZE_BACKBONE = False
# When fine-tuning the backbone, use a smaller per-forward micro-batch and accumulate
# gradients so the optimizer still sees BATCH_SIZE samples per step.
MICRO_BATCH_SIZE = 2
# Gradient checkpointing trades recompute for lower activation memory during backbone fine-tuning.
BACKBONE_GRADIENT_CHECKPOINTING = True
SKIP_VISUAL_EQUIV_WHEN_BACKBONE_TRAINABLE = True

# Frozen reference-feature cache for the fixed stage4 appearance objective.
USE_CACHED_BACKBONE_FEATURES = True
BACKBONE_FEATURE_CACHE_ROOT = str(DEFAULT_BACKBONE_FEATURE_CACHE_ROOT)
REFERENCE_BACKBONE_CHECKPOINT = DEFAULT_REFERENCE_CHECKPOINT
# True: load only; miss raises. False: compute missing with the frozen reference.
BACKBONE_FEATURE_CACHE_READ_ONLY = False
BACKBONE_FEATURE_CACHE_STORAGE_DTYPE = "float16"

# Fixed stage4 patch-cluster appearance references (immutable memberships).
VISUAL_CLUSTER_DIR = DEFAULT_PATCH_CLUSTER_DIR
VISUAL_CLUSTER_MAX_PATCHES_PER_PROTOTYPE = 0  # 0 = all fixed reference members
VISUAL_CLUSTER_SEED = 42
LR = 5e-5
FINETUNE_LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
OUTPUT_DIR = "training_outputs"
CKPTS_DIR = os.path.join(OUTPUT_DIR, "ckpts")
MAP_SAVE_INTERVAL = 1000
VAL_EVERY_N_EPOCHS = 1
VAL_METRICS_CSV_COLUMNS = (
    "Epoch_no",
    "Val Loss",
    "CC",
    "SIM",
    "NSS",
    "App_loss_patch_count",
    "app_loss",
)
OVERFIT_ONE_BATCH = False
OVERFIT_STEPS = 300
MAX_SAMPLES = 500
USE_AMP = True
# Default GPU index when --gpu is not provided (backbone + head on same device).
DEFAULT_GPU_ID = 1

# Auxiliary-loss switches; these do not disable the model's visual modules.
VISUAL_CONCEPT_ON = True
TEMPORAL_CONCEPTS_ON = False
MODEL_TOP_K = 1  # Shared by train/eval; recorded in new checkpoints.

# Last-slice residual priority masking at stages3/4:
# guided_last = base_last * ((1 - strength) + strength * priority_mask)
PROTOTYPE_SPATIAL_MASK_STRENGTH = 1.0
# Residual unary mask strength at stage1/stage2 (after refine):
# guided_last = decoded_base_last * ((1 - s) + s * priority_mask_last)
FINE_UNARY_MASK_STRENGTH = 1.0
# When False: stage2 fused = prev_scale * prev_up (no stage2 backbone/FiLM
# addition into the decode stream). Mask scoring still uses stage2 FiLM features.
# Non-parameter flag; same checkpoint loads for controlled comparison.
STAGE2_BACKBONE_FUSION_ENABLED = True

# Last-frame patch-priority map KL at Stage 3 and Stage 4.
PRIORITY_MAP_LOSS_WEIGHT = 0.1
PRIORITY_LOSS_WEIGHT = PRIORITY_MAP_LOSS_WEIGHT

# Epochs 1..N: zero dense/CC/SIM/NSS/KL, side-stage, and priority-map loss weights.
# From epoch N+1 onward, restore the values configured in LOSS_LAMBDA below.
INTERPRETABILITY_LOSS_WARMUP_EPOCHS = 0

FIXATION_THRESHOLD = 0.5
TOP_PERCENT = 0.05

# LOSS_LAMBDA = {
#     "lambda_delta": 0.0,
#     "lambda_dense": 0.25,
#     "lambda_bce": 0.0,
#     "lambda_kl": 2.0,
#     "lambda_fid": 1.0,
#     "lambda_cc": 0.5,
#     "lambda_nss": 0.5,
#     "topk_percent": 0.005,
#     "topk_bg_weight": 0.0,
#     "lambda_concept_dense": 0.25,
#     "lambda_concept_kl": 0.25,
#     "lambda_align": 0.25,
#     "lambda_sparse": 0.5,
#     "lambda_div": 0.5,
#     "lambda_gate": 0.0,
#     "lambda_visual_entropy": 0.02,
#     "lambda_visual_usage": 0.05,
#     "lambda_visual_equiv": 0.02,
# #     # Used only if compute_total_loss supports these ConceptCreation losses.
# #     "lambda_visual": 0.5,
# #     "lambda_visual_div": 0.5,
# #     "patch_from_logits": True,
# }
LOSS_LAMBDA = {
    # Old / auxiliary saliency-head losses: off for new decoder
    "lambda_delta": 0.0,
    "lambda_fid": 0.0,
    # Main dense saliency losses
    "lambda_dense": 1.0,
    "lambda_bce": 0.0,
    "lambda_kl": 1.0,
    "lambda_cc": 1.0,
    "lambda_nss": 0.1,
    "lambda_similarity": 0.05,
    # Disable explicit background suppression for now
    "topk_percent": 0.000,
    "topk_bg_weight": 0.0,
    # Disable old concept-map saliency supervision for now
    "lambda_concept_dense": 0.0,
    "lambda_concept_kl": 0.0,
    # Temporarily weaken concept regularizers until maps stop collapsing
    "lambda_align": 0.00,
    "lambda_sparse": 0.00,
    "lambda_div": 0.00,
    "lambda_gate": 0.0,
    # Temporarily reduce visual regularizers
    "lambda_visual_entropy": 0.005,
    "lambda_visual_usage": 3.0,
    "lambda_visual_equiv": 0.00,
    "lambda_temporal_attention_entropy": 0.0,
    "priority_map_loss_weight": PRIORITY_MAP_LOSS_WEIGHT,
    "priority_loss_weight": PRIORITY_MAP_LOSS_WEIGHT,
    "patch_from_logits": False,
    # Deep supervision on per-stage decoder side outputs (training only).
    "side_stage_weights": {
        "stage1": 0.20,
        "stage2": 0.15,
        "stage3": 0.10,
        "stage4": 0.05,
    },
}


def get_model_kwargs(
    *,
    visual_cluster_dir: Optional[str] = None,
    visual_cluster_max_patches_per_prototype: Optional[int] = None,
    visual_cluster_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """One architecture configuration shared by training and evaluation."""
    return {
        "backbone_stages": ("stage1", "stage2", "stage3", "stage4"),
        "pretrained_backbone": True,
        "freeze_backbone": FREEZE_BACKBONE,
        "backbone_gradient_checkpointing": BACKBONE_GRADIENT_CHECKPOINTING,
        "input_format": "BTCHW",
        "resize_to": (224, 384),
        "concept_dim": 128,
        "num_concepts": 561,
        "concept_hidden_dim": 256,
        "saliency_hidden_dim": 256,
        "top_k": MODEL_TOP_K,
        "output_activation": "none",
        "return_details": True,
        "visual_concept_residual_weight": 1.0,
        "use_temporal_feature_infusion": False,
        "use_shared_concept_activations": True,
        "prototype_spatial_mask_strength": PROTOTYPE_SPATIAL_MASK_STRENGTH,
        "fine_unary_mask_strength": FINE_UNARY_MASK_STRENGTH,
        "stage2_backbone_fusion_enabled": STAGE2_BACKBONE_FUSION_ENABLED,
        "visual_cluster_dir": VISUAL_CLUSTER_DIR
        if visual_cluster_dir is None
        else visual_cluster_dir,
        "visual_cluster_max_patches_per_prototype": VISUAL_CLUSTER_MAX_PATCHES_PER_PROTOTYPE
        if visual_cluster_max_patches_per_prototype is None
        else visual_cluster_max_patches_per_prototype,
        "visual_cluster_seed": VISUAL_CLUSTER_SEED
        if visual_cluster_seed is None
        else visual_cluster_seed,
    }


def build_model(
    backbone_device: torch.device,
    head_device: torch.device,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> ExplainableVidSalModel:
    """Construct the consolidated model without obsolete constructor options."""
    kwargs = get_model_kwargs() if model_kwargs is None else dict(model_kwargs)
    return ExplainableVidSalModel(**kwargs).to_split_devices(
        backbone_device, head_device
    )


def _resolve_devices(gpu_id: int = DEFAULT_GPU_ID) -> Tuple[torch.device, torch.device]:
    """Place backbone and head on the same GPU."""
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU_ID={gpu_id} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)."
            )
        device = torch.device(f"cuda:{gpu_id}")
        return device, device
    device = torch.device("cpu")
    return device, device


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _amp_enabled(device: torch.device) -> bool:
    return USE_AMP and device.type == "cuda"


def _dataloader_batch_size() -> int:
    if FREEZE_BACKBONE:
        return BATCH_SIZE
    return MICRO_BATCH_SIZE


def _grad_accum_steps() -> int:
    if FREEZE_BACKBONE:
        return 1
    if BATCH_SIZE % MICRO_BATCH_SIZE != 0:
        raise ValueError(
            f"BATCH_SIZE ({BATCH_SIZE}) must be divisible by "
            f"MICRO_BATCH_SIZE ({MICRO_BATCH_SIZE})"
        )
    return BATCH_SIZE // MICRO_BATCH_SIZE


def _backbone_is_trainable(model: ExplainableVidSalModel) -> bool:
    return not model._backbone_frozen


def _should_run_visual_equiv(
    model: ExplainableVidSalModel, epoch: Optional[int] = None
) -> bool:
    loss_lambda = _effective_loss_lambda(epoch)
    if loss_lambda.get("lambda_visual_equiv", 0.0) <= 0.0 or not VISUAL_CONCEPT_ON:
        return False
    if SKIP_VISUAL_EQUIV_WHEN_BACKBONE_TRAINABLE and _backbone_is_trainable(model):
        return False
    return True


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    trainable_params: list,
    scaler: Optional[GradScaler],
    device: torch.device,
) -> None:
    if scaler is not None and _amp_enabled(device):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()


def _effective_loss_lambda(epoch: Optional[int] = None) -> dict:
    """Apply ablation flags and the dense-loss warmup schedule for this epoch.

    For epochs 1..INTERPRETABILITY_LOSS_WARMUP_EPOCHS, zero the main dense/ranking saliency
    losses (dense, KL, CC, NSS, SIM), side-stage deep supervision, and priority
    map losses. From epoch INTERPRETABILITY_LOSS_WARMUP_EPOCHS+1 onward, restore LOSS_LAMBDA.
    When epoch is None, keep the configured LOSS_LAMBDA values (subject to
    branch ablation flags below).
    """
    loss_lambda = dict(LOSS_LAMBDA)
    loss_lambda["side_stage_weights"] = dict(LOSS_LAMBDA["side_stage_weights"])
    if epoch is not None and epoch <= INTERPRETABILITY_LOSS_WARMUP_EPOCHS:
        for key in (
            "lambda_dense",
            "lambda_kl",
            "lambda_cc",
            "lambda_nss",
            "lambda_similarity",
            "priority_map_loss_weight",
            "priority_loss_weight",
        ):
            loss_lambda[key] = 0.0
        loss_lambda["side_stage_weights"] = {
            stage: 0.0 for stage in loss_lambda["side_stage_weights"]
        }
    if not TEMPORAL_CONCEPTS_ON:
        for key in ("lambda_align", "lambda_sparse", "lambda_div", "lambda_gate"):
            loss_lambda[key] = 0.0
    if not VISUAL_CONCEPT_ON:
        for key in (
            "lambda_visual",
            "lambda_visual_div",
            "lambda_visual_entropy",
            "lambda_visual_usage",
            "lambda_visual_equiv",
        ):
            loss_lambda[key] = 0.0
    return loss_lambda


def _concept_stage(model: ExplainableVidSalModel, stage: str):
    """Look up a concept module from ``nn.ModuleDict`` (no dict ``.get``)."""
    modules = model.concept_creations
    if stage not in modules:
        return None
    return modules[stage]


def _stage4_appearance_enabled(
    model: Optional[ExplainableVidSalModel] = None,
) -> bool:
    """True when stage4's combined appearance objective has any active component."""
    stage4 = _concept_stage(model, "stage4") if model is not None else None
    if stage4 is not None:
        weights = stage4.loss_weights
    else:
        weights = VisualConceptCreation.DEFAULT_LOSS_WEIGHTS
    if float(weights.get("visual_appearance", 0.0)) <= 0.0:
        return False
    return any(
        float(weights.get(name, 0.0)) > 0.0
        for name in ("visual_rgb", "visual_feature", "visual_structure")
    )


def _return_concept_losses(
    model: Optional[ExplainableVidSalModel] = None,
) -> bool:
    """True when concept auxiliaries or stage4 appearance need return_losses=True."""
    loss_lambda = _effective_loss_lambda()
    if any(
        loss_lambda.get(k, 0) > 0
        for k in (
            "lambda_align",
            "lambda_sparse",
            "lambda_div",
            "lambda_gate",
            "lambda_visual",
            "lambda_visual_div",
            "lambda_visual_entropy",
            "lambda_visual_usage",
            "lambda_visual_equiv",
        )
    ):
        return True
    # Appearance lives inside loss_total_concept; allow appearance-only training
    # even when entropy/usage/equiv lambdas are all zero.
    return _stage4_appearance_enabled(model)


def _unpack_loader_batch(batch):
    """Support legacy 6-field batches and metadata-enabled 7-field batches."""
    if len(batch) == 6:
        video_filenames, rgb_batch, sal_batch, fix_batch, n_frames, valid_mask = batch
        return (
            video_filenames,
            rgb_batch,
            sal_batch,
            fix_batch,
            n_frames,
            valid_mask,
            None,
        )
    if len(batch) == 7:
        (
            video_filenames,
            rgb_batch,
            sal_batch,
            fix_batch,
            n_frames,
            valid_mask,
            sample_metadata,
        ) = batch
        return (
            video_filenames,
            rgb_batch,
            sal_batch,
            fix_batch,
            n_frames,
            valid_mask,
            sample_metadata,
        )
    raise ValueError(
        f"Unexpected loader batch length {len(batch)}; expected 6 or 7 fields."
    )


def _needs_reference_features(model: ExplainableVidSalModel) -> bool:
    """Only training with stage4's enabled feature appearance term needs cache I/O."""
    if not USE_CACHED_BACKBONE_FEATURES or not model.training:
        return False
    if not _return_concept_losses(model):
        return False
    stage4 = _concept_stage(model, "stage4")
    if stage4 is None:
        return False
    weights = stage4.loss_weights
    return (
        float(weights.get("visual_appearance", 0.0)) > 0.0
        and float(weights.get("visual_feature", 0.0)) > 0.0
    )


def _verify_stage4_patch_cluster_bank(model: ExplainableVidSalModel) -> None:
    """Fail before the first batch if required stage4 fixed references are missing."""
    if "stage4" not in model.concept_creations:
        raise RuntimeError(
            "ExplainableVidSalModel has no stage4 concept module; "
            "fixed patch-cluster appearance requires stage4."
        )
    stage4 = model.concept_creations["stage4"]
    if stage4.stage_name != "stage4":
        raise RuntimeError(
            f"stage4 concept module reports stage_name={stage4.stage_name!r}; "
            "pass stage_name='stage4' explicitly."
        )
    if not _stage4_appearance_enabled(model):
        print(
            "Stage4 appearance objective disabled "
            "(visual_appearance / component weights); skipping reference-bank check.",
            flush=True,
        )
        return
    if not stage4._cluster_loaded:
        # Constructor loads when the bank exists; missing bank must fail here.
        stage4.load_visual_patch_clusters()
    if not stage4._cluster_loaded:
        raise RuntimeError(
            f"Stage4 fixed patch-cluster bank failed to load from {stage4.visual_cluster_dir}."
        )
    meta = dict(stage4._cluster_metadata)
    full_counts = meta.get("full_counts") or []
    selected_counts = (
        meta.get("selected_counts") or meta.get("reference_patch_counts") or []
    )
    bytes_used = int(meta.get("reference_tensor_bytes") or 0)
    print(
        "Stage4 fixed patch-cluster bank | "
        f"dir={stage4.visual_cluster_dir} | "
        f"fingerprint={meta.get('fingerprint')} | "
        f"clusters={len(selected_counts)} | "
        f"full_members={int(sum(full_counts)) if full_counts else 0} | "
        f"selected_members={int(meta.get('num_reference_patches', sum(selected_counts) if selected_counts else 0))} | "
        f"rgb_descriptor_size={stage4.visual_appearance_patch_size} | "
        f"reference_memory_bytes={bytes_used} "
        f"({bytes_used / (1024**2):.2f} MiB) | "
        f"max_patches_per_prototype={stage4.visual_cluster_max_patches_per_prototype} | "
        f"seed={stage4.visual_cluster_seed}",
        flush=True,
    )


def _build_reference_cache(
    device: torch.device,
    model: Optional[ExplainableVidSalModel] = None,
) -> Optional[FrozenBackboneFeatureCache]:
    if not USE_CACHED_BACKBONE_FEATURES or (
        model is not None and not _needs_reference_features(model)
    ):
        return None
    cache = FrozenBackboneFeatureCache(
        root=BACKBONE_FEATURE_CACHE_ROOT,
        checkpoint=REFERENCE_BACKBONE_CHECKPOINT,
        device=device,
        storage_dtype=BACKBONE_FEATURE_CACHE_STORAGE_DTYPE,
        resize_to=(224, 384),
        normalize=True,
    )
    print(
        "Frozen reference-feature cache | "
        f"root={BACKBONE_FEATURE_CACHE_ROOT} | "
        f"read_only={BACKBONE_FEATURE_CACHE_READ_ONLY} | "
        f"checkpoint={REFERENCE_BACKBONE_CHECKPOINT} | "
        f"fingerprint={cache.reference_fingerprint[:16]}… | "
        f"legacy_compatible={cache.legacy_reference_sha256 is not None}",
        flush=True,
    )
    return cache


def _verify_reference_feature_provenance(
    model: ExplainableVidSalModel,
    feature_cache: Optional[FrozenBackboneFeatureCache],
) -> None:
    """Once at startup: cluster descriptors vs training cache must share a backbone."""
    if not _needs_reference_features(model):
        print(
            "Reference-feature provenance check skipped "
            "(stage4 feature appearance term inactive or cache unused).",
            flush=True,
        )
        return
    if feature_cache is None:
        raise RuntimeError(
            "Stage4 visual_feature appearance is enabled but FrozenBackboneFeatureCache "
            "was not constructed. Enable USE_CACHED_BACKBONE_FEATURES and keep "
            "REFERENCE_BACKBONE_CHECKPOINT aligned with the cluster bank."
        )
    stage4 = _concept_stage(model, "stage4")
    if stage4 is None or not stage4._cluster_loaded:
        raise RuntimeError(
            "Stage4 feature appearance requires a loaded fixed patch-cluster bank "
            "before provenance verification."
        )
    cluster_prov = (stage4._cluster_metadata or {}).get("reference_feature_provenance")
    cache_prov = provenance_from_cache(feature_cache)
    print(
        "Reference-feature provenance | "
        f"cache_fp={cache_prov.get('backbone_fingerprint', '')[:16]}… | "
        f"cluster_status={(stage4._cluster_metadata or {}).get('reference_feature_provenance_status')} | "
        f"cluster_fp="
        f"{(cluster_prov or {}).get('backbone_fingerprint', 'MISSING')[:16] if cluster_prov else 'MISSING'}…",
        flush=True,
    )
    compare_reference_feature_provenance(cluster_prov, cache_prov)


def _content_hw_from_meta(meta: Dict[str, Any]) -> Tuple[int, int]:
    content_hw = meta.get("content_hw")
    if content_hw is None and meta.get("padded_input_geometry"):
        content_hw = meta["padded_input_geometry"].get("content_hw")
    if content_hw is None:
        content_hw = meta.get("original_spatial_hw")
    if content_hw is None or len(content_hw) != 2:
        raise ValueError(f"sample metadata missing content_hw: {meta!r}")
    return int(content_hw[0]), int(content_hw[1])


def _identities_for_rgb_view(
    sample_metadata: Sequence[Dict[str, Any]],
    rgb_batch: torch.Tensor,
    n_frames: torch.Tensor,
    *,
    view_id: str,
    preprocess: Dict[str, Any],
) -> Tuple[List[SampleIdentity], List[Tuple[int, int]]]:
    identities: List[SampleIdentity] = []
    content_hw_list: List[Tuple[int, int]] = []
    for index, meta in enumerate(sample_metadata):
        height, width = _content_hw_from_meta(meta)
        frames = int(n_frames[index])
        clip = (
            rgb_batch[index, :frames, :, :height, :width]
            .detach()
            .to(dtype=torch.float32)
        )
        if clip.numel() > 0 and float(clip.max()) > 2.0:
            clip = clip / 255.0
        content_sha = fingerprint_rgb_clip(clip)
        padded = dict(meta.get("padded_input_geometry") or {})
        padded["view_id"] = view_id
        identities.append(
            SampleIdentity(
                dataset_root=str(meta.get("dataset_root") or meta.get("dataset_dir")),
                video_id=str(meta["video_id"]),
                window_start=int(meta["window_start"]),
                frame_names=tuple(meta["frame_names"]),
                window_len=int(meta.get("window_len", frames)),
                content_sha256=content_sha,
                split=meta.get("split"),
                temporal_sampling={
                    "mode": "contiguous",
                    "stride": int(meta.get("stride", 1)),
                },
                view_id=str(view_id),
                input_shape=tuple(int(x) for x in clip.shape),
                padded_input_geometry=padded,
                preprocess=dict(preprocess),
            )
        )
        content_hw_list.append((height, width))
    return identities, content_hw_list


def _resolve_cached_backbone_features(
    cache: Optional[FrozenBackboneFeatureCache],
    rgb_batch: torch.Tensor,
    sample_metadata: Optional[Sequence[Dict[str, Any]]],
    n_frames: torch.Tensor,
    *,
    view_id: str = "identity",
) -> Optional[Dict[str, torch.Tensor]]:
    """Resolve frozen reference last-slice features for the exact RGB view."""
    if cache is None or not USE_CACHED_BACKBONE_FEATURES:
        return None
    if sample_metadata is None:
        raise ValueError(
            "USE_CACHED_BACKBONE_FEATURES requires DatasetLoader "
            "return_sample_metadata=True and collated sample metadata."
        )
    identities, content_hw = _identities_for_rgb_view(
        sample_metadata,
        rgb_batch,
        n_frames,
        view_id=view_id,
        preprocess=cache.preprocess,
    )
    with torch.no_grad():
        return cache.resolve_batch(
            rgb_batch.detach(),
            identities=identities,
            n_frames=n_frames,
            content_hw=content_hw,
            view_id=view_id,
            force=False,
            allow_compute=not BACKBONE_FEATURE_CACHE_READ_ONLY,
            # Identity was hashed from this exact CPU view immediately above.
            verify_identity_content=False,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ExplainableSaliency model.")
    parser.add_argument(
        "--phase",
        choices=("initial", "finetune", "resume"),
        required=True,
        help=(
            "Training phase: 'initial' trains for 200 epochs from scratch; "
            "'finetune' loads model weights and trains for 100 epochs with "
            "finetune hyperparameters; 'resume' loads model weights and trains "
            "for 100 epochs using the LR and weight decay stored in the checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file (required when --phase finetune or resume).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=DEFAULT_GPU_ID,
        help=(
            "CUDA device index for the full model (backbone + head). "
            f"Default: {DEFAULT_GPU_ID}."
        ),
    )
    parser.add_argument(
        "--visual-cluster-dir",
        type=str,
        default=VISUAL_CLUSTER_DIR,
        help=(
            "Directory with fixed stage4 patch-cluster references "
            f"(manifest.json + descriptors). Default: {VISUAL_CLUSTER_DIR}."
        ),
    )
    parser.add_argument(
        "--visual-cluster-max-patches-per-prototype",
        type=int,
        default=VISUAL_CLUSTER_MAX_PATCHES_PER_PROTOTYPE,
        help=(
            "Cap on fixed reference members per prototype (0 = use all). "
            f"Default: {VISUAL_CLUSTER_MAX_PATCHES_PER_PROTOTYPE}."
        ),
    )
    parser.add_argument(
        "--visual-cluster-seed",
        type=int,
        default=VISUAL_CLUSTER_SEED,
        help=(
            "Seed for optional per-prototype reference subsetting. "
            f"Default: {VISUAL_CLUSTER_SEED}."
        ),
    )
    args = parser.parse_args()
    if args.phase in ("finetune", "resume") and not args.checkpoint:
        parser.error(
            f"{args.phase.capitalize()} (--phase {args.phase}) requires --checkpoint <path>."
        )
    if args.visual_cluster_max_patches_per_prototype < 0:
        parser.error("--visual-cluster-max-patches-per-prototype must be >= 0.")
    if args.visual_cluster_seed < 0:
        parser.error("--visual-cluster-seed must be >= 0.")
    return args


def _read_checkpoint(checkpoint_path: str) -> dict:
    if not checkpoint_path:
        raise ValueError("A checkpoint path is required.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint '{checkpoint_path}': expected a dict containing "
            "'model_state_dict'."
        )
    return checkpoint


def _is_allowed_missing_fine_unary_key(key: str) -> bool:
    """Newly added stage1/stage2 unary-mask modules may be absent in old ckpts."""
    for stage in ("stage1", "stage2"):
        prefix = f"saliency_prediction.fusion_blocks.{stage}."
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        allowed_roots = (
            "feature_token_proj.",
            "concept_proto_proj.",
            "unary_priority_mlp.",
            "shared_patch_proj.",
            "null_prototype",
        )
        if any(
            suffix == root.rstrip(".") or suffix.startswith(root)
            for root in allowed_roots
        ):
            return True
    return False


# Obsolete FIFO appearance buffers that may exist only in pre-fixed-cluster ckpts.
# Ignored on load; never restored into the immutable bank.
_OBSOLETE_FIFO_STATE_MARKERS = (
    "_prototype_queue",
    "_prototype_feature_queue",
    "_prototype_feature_count",
    "_prototype_feature_valid",
    "_prototype_feature_pointer",
)


def _is_obsolete_fifo_state_key(key: str) -> bool:
    return any(marker in key for marker in _OBSOLETE_FIFO_STATE_MARKERS)


def _load_model_checkpoint(
    model: ExplainableVidSalModel, checkpoint: dict, checkpoint_path: str
) -> None:
    state_dict = checkpoint["model_state_dict"]
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = [k for k in missing if _is_allowed_missing_fine_unary_key(k)]
    disallowed_missing = [
        k for k in missing if not _is_allowed_missing_fine_unary_key(k)
    ]
    obsolete_fifo = [k for k in unexpected if _is_obsolete_fifo_state_key(k)]
    disallowed_unexpected = [
        k for k in unexpected if not _is_obsolete_fifo_state_key(k)
    ]
    if disallowed_unexpected or disallowed_missing:
        details = []
        if disallowed_missing:
            details.append(
                "unexpected missing keys:\n  " + "\n  ".join(disallowed_missing)
            )
        if disallowed_unexpected:
            details.append("unexpected keys:\n  " + "\n  ".join(disallowed_unexpected))
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}' with selective "
            "compatibility for new stage1/stage2 unary-mask parameters.\n"
            + "\n".join(details)
        )
    print(f"Loaded model weights from checkpoint: {checkpoint_path}")
    if obsolete_fifo:
        print(
            "Ignored obsolete FIFO appearance buffers in checkpoint "
            f"({len(obsolete_fifo)} keys); fixed bank memberships are unchanged."
        )
    if allowed_missing:
        print(
            "Initialized newly added stage1/stage2 unary-mask parameters "
            f"({len(allowed_missing)} missing keys):"
        )
        for key in allowed_missing:
            print(f"  {key}")


def _verify_patch_cluster_checkpoint_metadata(
    model: ExplainableVidSalModel,
    checkpoint: dict,
    *,
    is_resume: bool,
    checkpoint_path: str,
) -> None:
    """Compare checkpoint patch_cluster_reference metadata to the loaded bank.

    Large reference arrays are never restored from the checkpoint; they stay in
    the external bank. Resume rejects fingerprint / subset / feature-space drift.
    Older FIFO checkpoints without this metadata may start the fixed-cluster
    pipeline as a migration (not an exact continuation).
    """
    recorded = checkpoint.get("patch_cluster_reference")
    stage4 = _concept_stage(model, "stage4")
    if stage4 is None:
        return
    if recorded is None:
        if is_resume:
            print(
                "MIGRATION: checkpoint has no patch_cluster_reference metadata "
                f"({checkpoint_path}). Treating this as a move from the old FIFO "
                "appearance objective to fixed patch-cluster supervision — not an "
                "exact continuation. Any historical queue contents are ignored; "
                "immutable bank memberships come only from visual_cluster_dir.",
                flush=True,
            )
        return
    if (
        not isinstance(recorded, dict)
        or recorded.get("format") != "patch_cluster_reference_v1"
    ):
        raise ValueError(
            f"Unrecognized patch_cluster_reference format in {checkpoint_path}: "
            f"{recorded.get('format') if isinstance(recorded, dict) else type(recorded)!r}"
        )
    if not stage4._cluster_loaded or not _stage4_appearance_enabled(model):
        return
    live = build_patch_cluster_reference_checkpoint_metadata(model)
    if live is None:
        return
    checks = (
        ("bank_fingerprint", "Fixed cluster bank fingerprint"),
        ("manifest_sha256", "Cluster manifest fingerprint"),
        ("num_clusters", "Cluster count"),
        ("rgb_descriptor_size", "RGB descriptor size"),
        ("max_patches_per_prototype", "Reference member cap"),
        ("reference_seed", "Reference subset seed"),
    )
    mismatches = []
    for key, label in checks:
        if key not in recorded:
            continue
        if recorded.get(key) != live.get(key):
            mismatches.append(
                f"{label}: checkpoint={recorded.get(key)!r} live={live.get(key)!r}"
            )
    for key, label in (
        ("full_member_counts", "Full member counts"),
        ("selected_member_counts", "Selected member counts"),
    ):
        if key not in recorded:
            continue
        if list(recorded.get(key) or []) != list(live.get(key) or []):
            mismatches.append(f"{label} disagree between checkpoint and live bank")
    rec_prov = recorded.get("reference_feature_provenance")
    live_prov = live.get("reference_feature_provenance")
    if isinstance(rec_prov, dict) and isinstance(live_prov, dict):
        for key in ("backbone_fingerprint", "temporal_selection"):
            if (
                rec_prov.get(key)
                and live_prov.get(key)
                and rec_prov.get(key) != live_prov.get(key)
            ):
                mismatches.append(
                    f"Reference feature {key}: checkpoint={rec_prov.get(key)!r} "
                    f"live={live_prov.get(key)!r}"
                )
        rec_pre = rec_prov.get("preprocess") or {}
        live_pre = live_prov.get("preprocess") or {}
        for key in ("resize_to", "normalize", "mean", "std", "input_format"):
            if key in rec_pre and key in live_pre and rec_pre[key] != live_pre[key]:
                mismatches.append(
                    f"Reference preprocess[{key}]: checkpoint={rec_pre[key]!r} "
                    f"live={live_pre[key]!r}"
                )
    if mismatches:
        if is_resume:
            raise RuntimeError(
                f"Resume rejected: patch_cluster_reference in {checkpoint_path} does not "
                "match the currently loaded stage4 bank.\n  " + "\n  ".join(mismatches)
            )
        print(
            "WARNING: checkpoint patch_cluster_reference differs from the loaded "
            f"stage4 bank ({checkpoint_path}); continuing because this is not "
            "--phase resume.\n  " + "\n  ".join(mismatches),
            flush=True,
        )
        return
    print(
        "Verified patch_cluster_reference against loaded stage4 bank | "
        f"fingerprint={(live.get('bank_fingerprint') or '')[:16]}…",
        flush=True,
    )


def _optimizer_hparams_from_checkpoint(
    checkpoint: dict, checkpoint_path: str
) -> Tuple[float, float]:
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError(
            f"Resume checkpoint '{checkpoint_path}' must contain "
            "'optimizer_state_dict' with saved LR and weight decay."
        )
    param_groups = optimizer_state.get("param_groups")
    if not param_groups:
        raise ValueError(
            f"Resume checkpoint '{checkpoint_path}' has an invalid optimizer_state_dict: "
            "missing param_groups."
        )
    lr = float(param_groups[0]["lr"])
    weight_decay = float(param_groups[0].get("weight_decay", WEIGHT_DECAY))
    return lr, weight_decay


def _init_saliency_head_prior_bias(model: ExplainableVidSalModel) -> None:
    """Initialize final saliency logits to the observed dataset prior."""
    target_prior = 0.0177
    bias_value = math.log(target_prior / (1.0 - target_prior))
    initialized = []

    final_head = getattr(model.saliency_prediction, "final_upsample_head", None)
    if final_head is not None and hasattr(final_head, "head"):
        final_conv = final_head.head
        if hasattr(final_conv, "bias") and final_conv.bias is not None:
            final_conv.bias.fill_(bias_value)
            initialized.append("final_upsample_head.head")

    patch_head = getattr(model.saliency_prediction, "patch_logit_head", None)
    if (
        patch_head is not None
        and hasattr(patch_head, "bias")
        and patch_head.bias is not None
    ):
        patch_head.bias.fill_(bias_value)
        initialized.append("patch_logit_head")

    pred_head = getattr(model.saliency_prediction, "pred_head", None)
    if pred_head is not None and len(pred_head) > 0:
        final_conv = pred_head[-1]
        if hasattr(final_conv, "bias") and final_conv.bias is not None:
            final_conv.bias.fill_(bias_value)
            initialized.append("pred_head[-1]")

    if initialized:
        print(
            f"Initialized saliency bias to logit({target_prior}) = {bias_value:.4f} "
            f"for: {', '.join(initialized)}"
        )
    else:
        print(
            "Skipped saliency bias initialization: no known decoder output bias parameter found."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def prepare_last_saliency_for_visualization(sal_batch: torch.Tensor) -> torch.Tensor:
    """
    First sample, last temporal frame -> [H, W] normalized to [0, 1] for display.
    """
    if not isinstance(sal_batch, torch.Tensor):
        raise ValueError("sal_batch must be a torch.Tensor")

    x = sal_batch.detach().float()
    if x.dim() == 3:
        # [B, H, W]
        x = x[0]
    elif x.dim() == 4:
        if x.shape[1] == 1:
            # [B, 1, H, W]
            x = x[0, 0]
        else:
            # [B, T, H, W]
            x = x[0, -1]
    elif x.dim() == 5:
        if x.shape[1] == 1:
            # [B, 1, T, H, W]
            x = x[0, 0, -1]
        elif x.shape[2] == 1:
            # [B, T, 1, H, W]
            x = x[0, -1, 0]
        else:
            raise ValueError(f"Unsupported 5D saliency shape {tuple(x.shape)}")
    else:
        raise ValueError(f"Unsupported saliency shape {tuple(sal_batch.shape)}")

    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def prepare_pred_for_visualization(pred_saliency: torch.Tensor) -> torch.Tensor:
    """First sample -> [H, W] normalized to [0, 1] for display."""
    x = pred_saliency.detach().float()
    if x.dim() == 4:
        if x.shape[1] == 1:
            x = x[0, 0]
        else:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(pred_saliency.shape)}")
    elif x.dim() == 3:
        x = x[0]
    else:
        raise ValueError(f"Unsupported pred shape {tuple(pred_saliency.shape)}")

    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def prepare_patch_map_for_visualization(x: torch.Tensor) -> torch.Tensor:
    """
    First sample, first channel -> [H, W] normalized to [0, 1].
    Accepts [B, 1, H, W] or [B, C, H, W].
    """
    x = x.detach().float()
    if x.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    x = x[0, 0]
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def prepare_last_rgb_for_visualization(rgb_batch: torch.Tensor) -> torch.Tensor:
    """First sample, last temporal frame -> [H, W, 3] in [0, 1]."""
    if not isinstance(rgb_batch, torch.Tensor):
        raise ValueError("rgb_batch must be a torch.Tensor")

    x = rgb_batch.detach().float()
    if x.dim() == 5:
        if x.shape[-1] == 3:
            x = x[0, -1]
        elif x.shape[2] == 3:
            x = x[0, -1].permute(1, 2, 0)
        else:
            raise ValueError(f"Unsupported 5D rgb shape {tuple(rgb_batch.shape)}")
    elif x.dim() == 4:
        if x.shape[-1] == 3:
            x = x[0]
        elif x.shape[1] == 3:
            x = x[0].permute(1, 2, 0)
        else:
            raise ValueError(f"Unsupported 4D rgb shape {tuple(rgb_batch.shape)}")
    else:
        raise ValueError(f"Unsupported rgb shape {tuple(rgb_batch.shape)}")

    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def save_map_png(map_tensor: torch.Tensor, save_path: str) -> None:
    arr = map_tensor.detach().cpu().numpy()
    plt.imsave(save_path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def save_rgb_png(rgb_tensor: torch.Tensor, save_path: str) -> None:
    arr = rgb_tensor.detach().cpu().numpy()
    plt.imsave(save_path, arr)


def save_batch_maps(
    model_out: dict,
    sal_batch: torch.Tensor,
    fix_batch: torch.Tensor,
    rgb_batch: torch.Tensor,
    output_dir: str,
    epoch: int,
    batch_idx: int,
    maps_subdir: str = "maps",
) -> None:
    vis_dir = os.path.join(
        output_dir,
        maps_subdir,
        f"epoch_{epoch:03d}",
        f"batch_{batch_idx:05d}",
    )
    os.makedirs(vis_dir, exist_ok=True)

    gt_map = prepare_last_saliency_for_visualization(sal_batch)
    fix_map = prepare_last_saliency_for_visualization(fix_batch)
    pred_map = prepare_pred_for_visualization(model_out["saliency_map"])
    rgb_frame = prepare_last_rgb_for_visualization(rgb_batch)
    save_map_png(gt_map, os.path.join(vis_dir, "gt_sal_map.png"))
    save_map_png(fix_map, os.path.join(vis_dir, "gt_fixation_map.png"))
    save_map_png(pred_map, os.path.join(vis_dir, "pred_sal_map.png"))
    save_rgb_png(rgb_frame, os.path.join(vis_dir, "rgb_frame.png"))

    pred_out = model_out.get("prediction_out")
    if pred_out is not None:
        optional_patch_maps = {
            "patch_transition_region.png": pred_out.get("patch_transition_region"),
            "patch_persistence_region.png": pred_out.get("patch_persistence_region"),
            "concept_context_patch_logits.png": pred_out.get(
                "concept_context_patch_logits"
            ),
            "temporal_saliency_map.png": pred_out.get("temporal_saliency_map"),
            "visual_saliency_map.png": pred_out.get("visual_saliency_map"),
        }

        for filename, tensor in optional_patch_maps.items():
            if tensor is not None and torch.is_tensor(tensor):
                vis = prepare_patch_map_for_visualization(tensor)
                save_map_png(vis, os.path.join(vis_dir, filename))


def save_first_batch_maps(
    model_out: dict,
    sal_batch: torch.Tensor,
    fix_batch: torch.Tensor,
    rgb_batch: torch.Tensor,
    output_dir: str,
) -> None:
    save_batch_maps(
        model_out, sal_batch, fix_batch, rgb_batch, output_dir, epoch=1, batch_idx=0
    )


def update_loss_curve(
    train_losses: list,
    val_losses: list,
    output_dir: str,
) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train", marker="o")
    # ax.plot(epochs, val_losses, label="Validation", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_progress.png"), dpi=150)
    plt.close(fig)


def _compute_batch_loss(
    model_out: dict,
    sal_batch: torch.Tensor,
    fix_batch: Optional[torch.Tensor] = None,
    equiv_model_out: Optional[dict] = None,
    epoch: Optional[int] = None,
    enable_side_aux: bool = True,
) -> dict:
    return compute_total_loss(
        model_out,
        sal_batch,
        equiv_model_out=equiv_model_out,
        fixation_maps=fix_batch,
        enable_side_aux=enable_side_aux,
        **_effective_loss_lambda(epoch),
    )


def tensor_stats(name: str, x: torch.Tensor) -> None:
    x = x.detach().float()
    print(
        f"{name}: shape={tuple(x.shape)} "
        f"min={x.min().item():.4f} max={x.max().item():.4f} "
        f"mean={x.mean().item():.4f} std={x.std().item():.4f}"
    )


def _print_first_batch_debug(model_out: dict, sal_batch: torch.Tensor) -> None:
    tensor_stats("pred_saliency", model_out["saliency_map"])
    tensor_stats("sal_batch", sal_batch.float())
    if "features_shape" in model_out:
        print("DEBUG features_shape:", model_out["features_shape"])
    pred_out = model_out.get("prediction_out")
    if pred_out is not None and "patch_coverage_count" in pred_out:
        cov = pred_out["patch_coverage_count"].detach().float()
        print(
            f"patch_coverage_count: shape={tuple(cov.shape)} "
            f"min={cov.min().item():.4f} max={cov.max().item():.4f} "
            f"mean={cov.mean().item():.4f}"
        )


def _print_decoder_priority_debug(
    model: ExplainableVidSalModel,
    model_out: dict,
) -> None:
    """Print the actual priority masks and contextual comparison diagnostics."""
    decoder = getattr(model, "saliency_prediction", None)
    if decoder is None:
        print("DEBUG decoder gates: no saliency_prediction module found")
        return

    pred_out = model_out.get("prediction_out")
    stage_gates = pred_out.get("stage_gates") if isinstance(pred_out, dict) else None

    fusion_blocks = getattr(decoder, "fusion_blocks", None)
    if fusion_blocks is not None:
        for stage, block in fusion_blocks.items():
            msg = f"DEBUG decoder priority mask [{stage}]"
            if isinstance(stage_gates, dict) and stage in stage_gates:
                gamma = stage_gates[stage].detach().float()
                gamma_mean = float(gamma.mean().cpu())
                gamma_std = float(gamma.std().cpu())
                gamma_min = float(gamma.min().cpu())
                gamma_max = float(gamma.max().cpu())
                msg += (
                    f" | priority_mask mean/std/min/max="
                    f"{gamma_mean:.4f}/{gamma_std:.4f}/{gamma_min:.4f}/{gamma_max:.4f}"
                )
            else:
                msg += (
                    " | priority_mask=unavailable "
                    "(pass return_decoder_diagnostics=True)"
                )
            if getattr(block, "priority_mode", "none") in ("pairwise", "unary_mask"):
                msg += (
                    " | prototype_spatial_mask_strength="
                    f"{float(block.prototype_spatial_mask_strength if block.priority_mode == 'pairwise' else block.fine_unary_mask_strength):.4f}"
                    " | prototype_application_position="
                    f"{block.prototype_application_position}"
                )
            print(msg)

    if isinstance(pred_out, dict):
        factorized_stats = pred_out.get("stage_factorized_comparison_stats")
        if isinstance(factorized_stats, dict) and factorized_stats:
            for stage in sorted(factorized_stats.keys()):
                stats = factorized_stats[stage]
                if not isinstance(stats, dict):
                    continue
                print(
                    f"DEBUG factorized comparison [{stage}] | "
                    f"query_norm_mean={float(stats.get('query_norm_mean', float('nan'))):.4f} | "
                    f"key_norm_mean={float(stats.get('key_norm_mean', float('nan'))):.4f} | "
                    f"context_mean={float(stats.get('context_mean', float('nan'))):.4f} | "
                    f"context_std={float(stats.get('context_std', float('nan'))):.4f} | "
                    f"context_min={float(stats.get('context_min', float('nan'))):.4f} | "
                    f"context_max={float(stats.get('context_max', float('nan'))):.4f} | "
                    f"total_candidate_weight={float(stats.get('total_candidate_weight', float('nan'))):.4f} | "
                    f"spatial_strength={float(stats.get('spatial_strength', float('nan'))):.4f} | "
                    f"spatial_freq_entropy={float(stats.get('spatial_frequency_weight_entropy', float('nan'))):.4f} | "
                    f"spatial_freq_weight_mean={float(stats.get('spatial_frequency_weights', float('nan'))):.4f} | "
                    f"rank={float(stats.get('factorized_rank', float('nan'))):.0f}"
                )


def _print_patch_priority_diagnostics(model_out: dict) -> None:
    """Print per-stage patch-priority map diagnostics."""
    pred_out = model_out.get("prediction_out")
    if not isinstance(pred_out, dict):
        print("DEBUG patch-priority map: unavailable (missing prediction_out)")
        return

    stage_diag = pred_out.get("stage_mask_diagnostics")
    if not isinstance(stage_diag, dict) or not stage_diag:
        print(
            "DEBUG patch-priority map: unavailable "
            "(pass return_decoder_diagnostics=True on the logging batch)"
        )
        return

    for stage in sorted(stage_diag.keys()):
        diag = stage_diag.get(stage)
        if not isinstance(diag, dict):
            continue
        ratio = diag.get(
            "prototype_to_base_norm_ratio",
            diag.get("prototype_to_film_norm_ratio", float("nan")),
        )
        print(
            f"patch-priority map [{stage}] | "
            f"mean={float(diag.get('patch_priority_mean', float('nan'))):.6f} | "
            f"std={float(diag.get('patch_priority_std', float('nan'))):.6f} | "
            f"min={float(diag.get('patch_priority_min', float('nan'))):.6f} | "
            f"max={float(diag.get('patch_priority_max', float('nan'))):.6f} | "
            f"prototype_application_position="
            f"{diag.get('prototype_application_position', 'n/a')} | "
            f"prototype_spatial_mask_strength="
            f"{float(diag.get('prototype_spatial_mask_strength', float('nan'))):.6f} | "
            f"prototype_to_base_norm_ratio={float(ratio):.6f}"
        )


def _print_visual_concept_usage_debug(model_out: dict, top_n: int = 10) -> None:
    concept_out = model_out.get("concept_out")
    if not isinstance(concept_out, dict):
        return

    for stage, stage_out in concept_out.items():
        if not isinstance(stage_out, dict):
            continue

        indices = stage_out.get("visual_concept_indices")
        logits = stage_out.get("visual_concept_logits")

        if not torch.is_tensor(indices):
            continue

        idx = indices.detach().reshape(-1).cpu()
        unique, counts = idx.unique(return_counts=True)

        total = max(int(idx.numel()), 1)
        pairs = sorted(
            zip(unique.tolist(), counts.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        print(f"DEBUG visual concept usage [{stage}] top {top_n}:")
        for concept_id, count in pairs:
            print(
                f"  concept {int(concept_id)}: "
                f"{int(count)} patches ({100.0 * count / total:.2f}%)"
            )

        if torch.is_tensor(logits):
            probs = F.softmax(logits.detach().float(), dim=-1)
            mean_probs = probs.mean(dim=0)
            usage_entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum() / max(
                math.log(mean_probs.numel()), 1e-8
            )
            print(f"  normalized usage entropy: {float(usage_entropy.cpu()):.4f}")


def _compute_batch_metrics(
    model_out: dict,
    sal_batch: torch.Tensor,
    fix_batch: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return compute_saliency_metrics(
        model_out["saliency_map"],
        target_density=sal_batch,
        fixation_target=fix_batch,
        fixation_threshold=0.5,
        top_percent=None,
        allow_pseudo_fixations=False,
        dh1k_exact=True,
    )


def _compute_batch_metrics_excluding_empty_fixations(
    model_out: dict,
    sal_batch: torch.Tensor,
    fix_batch: torch.Tensor,
) -> Tuple[Optional[Dict[str, torch.Tensor]], int]:
    """Compute CC/SIM/NSS on samples with non-empty fixation maps only."""
    return compute_saliency_metrics_excluding_empty_fixations(
        model_out["saliency_map"],
        sal_batch,
        fix_batch,
        fixation_threshold=0.5,
        top_percent=None,
        allow_pseudo_fixations=False,
        dh1k_exact=True,
    )


def _detached_diag_scalar(value: Any) -> float:
    """Convert a diagnostics scalar to a Python float without retaining graphs."""
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.detach().float().reshape(()).cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_train_concept_stages(
    concept_out: Any,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Stage outputs from the main training forward's concept_out dict."""
    if not isinstance(concept_out, dict) or not concept_out:
        return []
    first_val = next(iter(concept_out.values()))
    if isinstance(first_val, dict) and (
        "losses" in first_val
        or "visual_appearance_diagnostics" in first_val
        or "visual_concept_logits" in first_val
    ):
        return [
            (str(stage), stage_out)
            for stage, stage_out in concept_out.items()
            if isinstance(stage_out, dict)
        ]
    return [("default", concept_out)] if isinstance(concept_out, dict) else []


def _collect_main_forward_appearance_stats(model_out: dict) -> Dict[str, Any]:
    """Detached stage4 appearance coverage + raw loss from the main forward only.

    Reads diagnostics/losses embedded in ``model_out`` immediately so flipped or
    diagnostic forwards cannot overwrite mutable module state after the fact.

    ``app_loss`` is stage4 ``loss_visual_appearance`` only (internal RGB / feature
    / structure weights included; **before** ``loss_weights['visual_appearance']``
    and **before** gradient-accumulation scaling). Stage1/2/3 zeros are ignored.

    ``App_loss_patch_count`` counts each evaluated stage4 query patch once
    (not once per prototype, reference member, or metric).

    Relationship to the optimized total (optimizer aggregation unchanged):

        stage4 loss_total_concept += w_visual_appearance * loss_visual_appearance
        loss_total_concept = mean_s(stage_s.loss_total_concept)   # N stages
        loss_total += loss_total_concept   # appearance is NOT added again

    So with four contributing stages, the raw logged value contributes

        (w_visual_appearance * app_loss) / 4

    to ``loss_total`` (before grad-accum division).
    """
    stage_rows: Dict[str, Dict[str, float]] = {}
    batch_app_loss: Optional[float] = None
    patch_count = 0.0

    stages = dict(_iter_train_concept_stages(model_out.get("concept_out")))
    stage_out = stages.get("stage4")
    if isinstance(stage_out, dict):
        diag = stage_out.get("visual_appearance_diagnostics")
        if not isinstance(diag, dict):
            diag = {}
        evaluated = _detached_diag_scalar(diag.get("app_loss_patch_count"))
        total = _detached_diag_scalar(diag.get("app_loss_patch_count_total"))
        rgb = _detached_diag_scalar(diag.get("app_loss_rgb_patch_count"))
        feature = _detached_diag_scalar(diag.get("app_loss_feature_patch_count"))
        structure = _detached_diag_scalar(diag.get("app_loss_structure_patch_count"))
        stage_rows["stage4"] = {
            "evaluated": evaluated,
            "total": total,
            "rgb": rgb,
            "feature": feature,
            "structure": structure,
        }
        patch_count = evaluated
        losses = stage_out.get("losses")
        if isinstance(losses, dict):
            loss_app = losses.get("loss_visual_appearance")
            if torch.is_tensor(loss_app):
                batch_app_loss = float(loss_app.detach().float().reshape(()).cpu())

    return {
        "App_loss_patch_count": patch_count,
        "app_loss": batch_app_loss,
        "stages": stage_rows,
    }


def _new_appearance_epoch_accumulator() -> Dict[str, Any]:
    return {
        "patch_count_sum": 0.0,
        # sum(raw_stage4_app_loss * evaluated_count) for patch-weighted epoch mean
        "weighted_app_loss_sum": 0.0,
        "num_batches": 0,
        "stages": {},  # stage4 -> evaluated/total/rgb/feature/structure sums
    }


def _accumulate_appearance_epoch_stats(
    acc: Dict[str, Any],
    batch_stats: Dict[str, Any],
) -> None:
    """Accumulate one microbatch of main-forward stage4 appearance stats (no graph)."""
    acc["num_batches"] += 1
    evaluated = float(batch_stats.get("App_loss_patch_count", 0.0))
    acc["patch_count_sum"] += evaluated
    batch_app = batch_stats.get("app_loss")
    # Zero-evaluated batches contribute nothing (do not dilute the weighted mean).
    if evaluated > 0.0 and batch_app is not None:
        acc["weighted_app_loss_sum"] += float(batch_app) * evaluated
    for stage, row in batch_stats.get("stages", {}).items():
        slot = acc["stages"].setdefault(
            stage,
            {
                "evaluated": 0.0,
                "total": 0.0,
                "rgb": 0.0,
                "feature": 0.0,
                "structure": 0.0,
            },
        )
        for key in ("evaluated", "total", "rgb", "feature", "structure"):
            slot[key] += float(row.get(key, 0.0))


def _finalize_appearance_epoch_stats(acc: Dict[str, Any]) -> Dict[str, Any]:
    """Epoch stage4 patch sum and patch-weighted mean of raw appearance loss.

    app_loss = sum(batch_raw_stage4_app_loss * batch_evaluated_patch_count)
               / sum(batch_evaluated_patch_count)

    Batches with no evaluated queries are excluded from both sums. If the epoch
    has no evaluated queries, ``app_loss`` is NaN (CSV writes an empty cell).
    """
    patch_count = float(acc["patch_count_sum"])
    if patch_count > 0.0:
        app_loss = float(acc["weighted_app_loss_sum"]) / patch_count
    else:
        app_loss = float("nan")
    return {
        "App_loss_patch_count": patch_count,
        "app_loss": app_loss,
        "stages": {
            stage: {
                "evaluated": float(row["evaluated"]),
                "total": float(row["total"]),
                "rgb": float(row["rgb"]),
                "feature": float(row["feature"]),
                "structure": float(row.get("structure", 0.0)),
            }
            for stage, row in acc["stages"].items()
        },
    }


def _print_appearance_coverage_summary(
    epoch: int, appearance_stats: Dict[str, Any]
) -> None:
    """Print stage4 evaluated/total coverage and RGB / feature / structure counts."""
    stages = appearance_stats.get("stages") or {}
    row = stages.get("stage4")
    patch_count = appearance_stats.get("App_loss_patch_count", 0.0)
    app_loss = appearance_stats.get("app_loss")
    if app_loss is None or app_loss != app_loss:
        app_loss_txt = "n/a"
    else:
        app_loss_txt = f"{float(app_loss):.6f}"
    if not isinstance(row, dict):
        print(
            f"Appearance coverage epoch {epoch}: stage4 unavailable | "
            f"App_loss_patch_count={patch_count} | app_loss={app_loss_txt}",
            flush=True,
        )
        return
    total = max(float(row.get("total", 0.0)), 0.0)
    print(
        f"Appearance coverage epoch {epoch}: stage4 "
        f"evaluated {row.get('evaluated', 0.0):.0f}/{total:.0f} | "
        f"rgb {row.get('rgb', 0.0):.0f}/{total:.0f} | "
        f"feature {row.get('feature', 0.0):.0f}/{total:.0f} | "
        f"structure {row.get('structure', 0.0):.0f}/{total:.0f} | "
        f"App_loss_patch_count={float(patch_count):.0f} | "
        f"app_loss={app_loss_txt} "
        f"(raw loss_visual_appearance; contrib to loss_total ≈ "
        f"w_visual_appearance*app_loss/N_stages)",
        flush=True,
    )


def train_one_epoch(
    model: ExplainableVidSalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    trainable_params: list,
    device: torch.device,
    epoch: int,
    output_dir: str,
    calculate_metrics: bool = True,
    scaler: Optional[GradScaler] = None,
    feature_cache: Optional[FrozenBackboneFeatureCache] = None,
) -> Tuple[float, Optional[Dict[str, float]], Dict[str, Any]]:
    model.train()
    reference_stats_before = (
        dict(feature_cache.stats) if feature_cache is not None else {}
    )
    running_loss = 0.0
    num_batches = 0
    metric_averager = MetricAverager()
    appearance_acc = _new_appearance_epoch_accumulator()
    accum_steps = _grad_accum_steps()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        loader,
        desc=f"Train epoch {epoch}",
        leave=False,
        disable=not SHOW_PROGRESS_BAR,
    )
    for batch_idx, batch in enumerate(pbar):
        (
            video_filenames,
            rgb_batch,
            sal_batch,
            fix_batch,
            n_frames,
            valid_mask,
            sample_metadata,
        ) = _unpack_loader_batch(batch)
        if not torch.is_tensor(sal_batch):
            raise ValueError("sal_batch must be a torch.Tensor for loss calculation.")
        # Hash/read before uploading RGB; retain the CPU view for optional flips.
        rgb_batch_cpu, n_frames_cpu = rgb_batch, n_frames
        cached_backbone_features = None
        if _needs_reference_features(model):
            cached_backbone_features = _resolve_cached_backbone_features(
                feature_cache,
                rgb_batch_cpu,
                sample_metadata,
                n_frames_cpu,
                view_id="identity",
            )
        rgb_batch, sal_batch, fix_batch = model.prepare_training_batch(
            rgb_batch,
            sal_batch,
            fix_batch,
        )
        fix_batch = (fix_batch > 0).float()

        equiv_model_out = None
        with autocast(
            device.type,
            dtype=_amp_dtype(device),
            enabled=_amp_enabled(device),
        ):
            model_out = model(
                rgb_batch,
                saliency_maps=sal_batch,
                return_details=True,
                return_concept_losses=_return_concept_losses(model),
                return_decoder_diagnostics=(batch_idx == 0),
                cached_backbone_features=cached_backbone_features,
            )
            # Capture main-forward appearance stats before any flipped forward.
            appearance_batch_stats = _collect_main_forward_appearance_stats(model_out)
            _accumulate_appearance_epoch_stats(appearance_acc, appearance_batch_stats)

            if _should_run_visual_equiv(model, epoch):
                rgb_batch_flip = torch.flip(rgb_batch, dims=[-1])
                cached_backbone_features_flip = None
                if _needs_reference_features(model):
                    cached_backbone_features_flip = _resolve_cached_backbone_features(
                        feature_cache,
                        torch.flip(rgb_batch_cpu, dims=[-1]),
                        sample_metadata,
                        n_frames_cpu,
                        view_id="hflip",
                    )
                equiv_model_out = model(
                    rgb_batch_flip,
                    saliency_maps=sal_batch,
                    return_details=True,
                    return_concept_losses=_return_concept_losses(model),
                    return_decoder_diagnostics=False,
                    cached_backbone_features=cached_backbone_features_flip,
                )

            if batch_idx == 0:
                _print_first_batch_debug(model_out, sal_batch)
                _print_decoder_priority_debug(model, model_out)
                _print_patch_priority_diagnostics(model_out)
                _print_visual_concept_usage_debug(model_out)

            if batch_idx % MAP_SAVE_INTERVAL == 0:
                save_batch_maps(
                    model_out,
                    sal_batch,
                    fix_batch,
                    rgb_batch,
                    output_dir,
                    epoch,
                    batch_idx,
                )

            loss_dict = _compute_batch_loss(
                model_out,
                sal_batch,
                fix_batch=fix_batch,
                equiv_model_out=equiv_model_out,
                epoch=epoch,
            )
            loss = loss_dict["loss_total"] / accum_steps

        if batch_idx == 0:
            with torch.no_grad():
                pred = model_out["saliency_map"].detach()
                target_last = prepare_target_last_map(sal_batch).detach().float()
                if target_last.numel() > 0 and target_last.max() > 2.0:
                    target_last = target_last / 255.0

                print(
                    "DEBUG pred min/max/mean/std:",
                    float(pred.min().cpu()),
                    float(pred.max().cpu()),
                    float(pred.mean().cpu()),
                    float(pred.std().cpu()),
                )
                print(
                    "DEBUG target min/max/mean/std:",
                    float(target_last.min().cpu()),
                    float(target_last.max().cpu()),
                    float(target_last.mean().cpu()),
                    float(target_last.std().cpu()),
                )
                target_fix_last = prepare_target_last_map(fix_batch).detach().float()
                print(
                    "DEBUG fixation pixels per sample:",
                    target_fix_last.flatten(1).sum(dim=1).detach().cpu().tolist(),
                )
                print(
                    "DEBUG fix_batch min/max/mean/std:",
                    float(fix_batch.min().detach().cpu()),
                    float(fix_batch.max().detach().cpu()),
                    float(fix_batch.mean().detach().cpu()),
                    float(fix_batch.std().detach().cpu()),
                )
                print(
                    "DEBUG losses:",
                    {
                        k: float(v.detach().cpu())
                        for k, v in loss_dict.items()
                        if torch.is_tensor(v) and v.ndim == 0
                    },
                )

        if scaler is not None and _amp_enabled(device):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (batch_idx + 1) % accum_steps == 0
        is_last_batch = batch_idx + 1 == len(loader)
        if should_step or is_last_batch:
            _optimizer_step(optimizer, trainable_params, scaler, device)
            optimizer.zero_grad(set_to_none=True)

        if calculate_metrics:
            with torch.no_grad():
                metric_dict = _compute_batch_metrics(model_out, sal_batch, fix_batch)
                metric_averager.update(metric_dict, batch_size=rgb_batch.shape[0])

        running_loss += loss.item() * accum_steps
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Drop graph references so activations can be freed before the next batch.
        del (
            model_out,
            loss_dict,
            loss,
            rgb_batch,
            sal_batch,
            fix_batch,
            rgb_batch_cpu,
            cached_backbone_features,
        )
        if equiv_model_out is not None:
            del equiv_model_out

    mean_loss = running_loss / max(num_batches, 1)
    appearance_stats = _finalize_appearance_epoch_stats(appearance_acc)
    _print_appearance_coverage_summary(epoch, appearance_stats)
    if feature_cache is not None:
        delta = {
            k: int(v) - int(reference_stats_before.get(k, 0))
            for k, v in feature_cache.stats.items()
        }
        print(
            f"Reference cache epoch {epoch}: hits={delta.get('hits', 0)} "
            f"misses={delta.get('misses', 0)} rejected={delta.get('rejected', 0)} "
            f"teacher_forwards={delta.get('reference_forwards', 0)} writes={delta.get('writes', 0)}",
            flush=True,
        )
    if calculate_metrics:
        return mean_loss, metric_averager.mean(), appearance_stats
    else:
        return mean_loss, None, appearance_stats


def _format_val_metrics_csv_row(
    epoch: int,
    val_loss: Optional[float],
    val_metrics: Optional[Dict[str, float]],
    appearance_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {"Epoch_no": epoch}
    if val_loss is None or val_loss != val_loss:
        row["Val Loss"] = "-"
    else:
        row["Val Loss"] = val_loss

    if val_metrics is not None:
        row["CC"] = val_metrics["CC"]
        row["SIM"] = val_metrics["SIM"]
        row["NSS"] = val_metrics["NSS"]
    else:
        row["CC"] = "-"
        row["SIM"] = "-"
        row["NSS"] = "-"

    # Training-epoch appearance columns (empty when unavailable / no queries).
    if appearance_stats is None:
        row["App_loss_patch_count"] = ""
        row["app_loss"] = ""
    else:
        patch_count = appearance_stats.get("App_loss_patch_count")
        app_loss = appearance_stats.get("app_loss")
        if patch_count is None:
            row["App_loss_patch_count"] = ""
        else:
            row["App_loss_patch_count"] = float(patch_count)
        if app_loss is None or app_loss != app_loss:
            row["app_loss"] = ""
        else:
            row["app_loss"] = float(app_loss)
    return row


@torch.no_grad()
def validate_one_epoch(
    model: ExplainableVidSalModel,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: str,
    feature_cache: Optional[FrozenBackboneFeatureCache] = None,
) -> Tuple[float, Dict[str, float], None]:
    model.eval()
    running_loss = 0.0
    num_batches = 0
    metric_averager = MetricAverager()

    pbar = tqdm(
        loader,
        desc=f"Val epoch {epoch}",
        leave=False,
        disable=not SHOW_PROGRESS_BAR,
    )
    for batch_idx, batch in enumerate(pbar):
        (
            video_filenames,
            rgb_batch,
            sal_batch,
            fix_batch,
            n_frames,
            valid_mask,
            sample_metadata,
        ) = _unpack_loader_batch(batch)
        if not torch.is_tensor(sal_batch):
            raise ValueError("sal_batch must be a torch.Tensor for loss calculation.")
        rgb_batch, sal_batch, fix_batch = model.prepare_training_batch(
            rgb_batch,
            sal_batch,
            fix_batch,
        )
        fix_batch = (fix_batch > 0).float()
        # Appearance losses skip eval/no_grad. Keep ordinary validation losses,
        # but never load or compute unused frozen-reference targets here.
        cached_backbone_features = None

        with autocast(
            device.type,
            dtype=_amp_dtype(device),
            enabled=_amp_enabled(device),
        ):
            model_out = model(
                rgb_batch,
                saliency_maps=sal_batch,
                return_details=True,
                return_concept_losses=_return_concept_losses(model),
                return_decoder_diagnostics=(batch_idx == 0),
                cached_backbone_features=cached_backbone_features,
            )
            if batch_idx == 0:
                _print_decoder_priority_debug(model, model_out)
                _print_patch_priority_diagnostics(model_out)
            if batch_idx % MAP_SAVE_INTERVAL == 0:
                save_batch_maps(
                    model_out,
                    sal_batch,
                    fix_batch,
                    rgb_batch,
                    output_dir,
                    epoch,
                    batch_idx,
                    maps_subdir="val_maps",
                )
            loss_dict = _compute_batch_loss(
                model_out,
                sal_batch,
                fix_batch=fix_batch,
                epoch=epoch,
                enable_side_aux=False,
            )
            batch_loss = loss_dict["loss_total"].item()
        running_loss += batch_loss
        num_batches += 1

        metric_dict, valid_count = _compute_batch_metrics_excluding_empty_fixations(
            model_out,
            sal_batch,
            fix_batch,
        )
        if metric_dict is not None:
            metric_averager.update(metric_dict, batch_size=valid_count)
        running_means = metric_averager.mean()
        postfix = {"loss": f"{batch_loss:.4f}"}
        if metric_dict is not None:
            postfix.update(
                CC=f"{running_means['CC']:.4f}",
                SIM=f"{running_means['SIM']:.4f}",
                NSS=f"{running_means['NSS']:.4f}",
            )
        pbar.set_postfix(postfix)

        del model_out, loss_dict, rgb_batch, sal_batch, fix_batch

    mean_loss = running_loss / max(num_batches, 1)
    return mean_loss, metric_averager.mean(), None


def main() -> None:
    args = _parse_args()
    is_finetune = args.phase == "finetune"
    is_resume = args.phase == "resume"
    loads_checkpoint = is_finetune or is_resume
    train_weight_decay = WEIGHT_DECAY

    if is_finetune:
        num_epochs = FINETUNE_EPOCHS
        train_lr = FINETUNE_LR
        scheduler_t_max = FINETUNE_EPOCHS
        scheduler_eta_min = 1e-7
    elif is_resume:
        num_epochs = RESUME_EPOCHS
        train_lr = LR
        scheduler_t_max = RESUME_EPOCHS
        scheduler_eta_min = 1e-6
    else:
        num_epochs = INITIAL_EPOCHS
        train_lr = LR
        scheduler_t_max = INITIAL_EPOCHS
        scheduler_eta_min = 1e-6

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = os.path.join(CKPTS_DIR, run_timestamp)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(run_ckpt_dir, "best_checkpoint.pth")
    last_ckpt_path = os.path.join(run_ckpt_dir, "last_checkpoint.pth")
    set_seed(SEED)

    backbone_device, head_device = _resolve_devices(args.gpu)
    device = head_device
    print(f"Using single device: {device} (gpu={args.gpu})")
    if is_resume:
        print(
            f"Training phase: {args.phase} | epochs={num_epochs} | lr/weight_decay=<from checkpoint>"
        )
    else:
        print(
            f"Training phase: {args.phase} | epochs={num_epochs} | "
            f"lr={train_lr:.1e} | weight_decay={train_weight_decay:.1e}"
        )
    if loads_checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Run checkpoints will be saved to: {run_ckpt_dir}")
    print(
        "Auxiliary-loss switches | "
        f"visual_concept_on={VISUAL_CONCEPT_ON} | "
        f"temporal_concepts_on={TEMPORAL_CONCEPTS_ON} | "
        f"priority_map_loss_weight={PRIORITY_MAP_LOSS_WEIGHT} | "
        f"top_k={MODEL_TOP_K}"
    )
    print(
        "Dense-loss warmup | "
        f"epochs 1-{INTERPRETABILITY_LOSS_WARMUP_EPOCHS}: "
        "lambda_dense/kl/cc/nss/similarity, side_stage_weights, "
        "priority_map_loss_weight, and priority_loss_weight = 0; "
        f"from epoch {INTERPRETABILITY_LOSS_WARMUP_EPOCHS + 1}: restore LOSS_LAMBDA values"
    )
    print(
        "Stage4 fixed patch clusters | "
        f"dir={args.visual_cluster_dir} | "
        f"max_patches_per_prototype={args.visual_cluster_max_patches_per_prototype} | "
        f"seed={args.visual_cluster_seed}"
    )
    print(
        "Appearance CSV logging (train-epoch measurements in val_metrics_hw2.csv) | "
        "App_loss_patch_count = sum of evaluated stage4 query patches from main "
        "training forwards only (one count per query; exclude flips/validation); "
        "app_loss = patch-weighted mean of raw stage4 loss_visual_appearance "
        "(before visual_appearance weight and before grad-accum scaling). "
        "Route into the optimizer: loss_visual_appearance -> "
        "w_visual_appearance * loss inside stage4 loss_total_concept -> "
        "mean over stages in _aggregate_concept_total_loss -> loss_total "
        "(not added again). With four stages, stage4 appearance is divided by 4; "
        "weights are not re-scaled to compensate. Prior CSV rows keep their old "
        "definitions and are not rewritten.",
        flush=True,
    )

    train_dataset = DatasetLoader(
        TRAIN_DATASET_DIR,
        window_len=WINDOW_LEN,
        stride=1,
        random_train_sampling=True,
        return_sample_metadata=USE_CACHED_BACKBONE_FEATURES,
    )
    val_dataset = DatasetLoader(
        VAL_DATASET_DIR,
        window_len=WINDOW_LEN,
        stride=32,
        random_train_sampling=False,
        return_sample_metadata=USE_CACHED_BACKBONE_FEATURES,
    )

    # g = torch.Generator().manual_seed(SEED)
    # idx = torch.randperm(len(train_dataset), generator=g)[:MAX_SAMPLES].tolist()
    # train_dataset = Subset(train_dataset, idx)

    # g = torch.Generator().manual_seed(SEED + 1)
    # idx = torch.randperm(len(val_dataset), generator=g)[:MAX_SAMPLES].tolist()
    # val_dataset = Subset(val_dataset, idx)

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "collate_fn": video_saliency_collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=_dataloader_batch_size(),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=_dataloader_batch_size(),
        shuffle=False,
        **loader_kwargs,
    )

    print(
        f"Train: {len(train_dataset)} windows ({TRAIN_DATASET_DIR}) | "
        f"Val: {len(val_dataset)} windows ({VAL_DATASET_DIR}) | "
        f"window_len={WINDOW_LEN}"
    )

    model_kwargs = get_model_kwargs(
        visual_cluster_dir=args.visual_cluster_dir,
        visual_cluster_max_patches_per_prototype=args.visual_cluster_max_patches_per_prototype,
        visual_cluster_seed=args.visual_cluster_seed,
    )
    model = build_model(backbone_device, head_device, model_kwargs=model_kwargs)

    # Immutable stage4 appearance bank must be loaded/compatible before any batch.
    _verify_stage4_patch_cluster_bank(model)

    checkpoint = None
    if loads_checkpoint:
        checkpoint = _read_checkpoint(args.checkpoint)
        _load_model_checkpoint(model, checkpoint, args.checkpoint)
        _verify_patch_cluster_checkpoint_metadata(
            model,
            checkpoint,
            is_resume=is_resume,
            checkpoint_path=args.checkpoint,
        )
        if is_resume:
            train_lr, train_weight_decay = _optimizer_hparams_from_checkpoint(
                checkpoint, args.checkpoint
            )
            print(
                f"Resume optimizer hyperparameters from checkpoint | "
                f"lr={train_lr:.1e} | weight_decay={train_weight_decay:.1e}"
            )
            # Bank already loaded from disk; re-confirm after weight restore.
            _verify_stage4_patch_cluster_bank(model)
    else:
        with torch.no_grad():
            _init_saliency_head_prior_bias(model)

    if not FREEZE_BACKBONE:
        print(
            "Backbone fine-tuning enabled | "
            f"effective_batch_size={BATCH_SIZE} | "
            f"micro_batch_size={_dataloader_batch_size()} | "
            f"grad_accum_steps={_grad_accum_steps()} | "
            f"gradient_checkpointing={BACKBONE_GRADIENT_CHECKPOINTING} | "
            f"visual_equiv={'off' if SKIP_VISUAL_EQUIV_WHEN_BACKBONE_TRAINABLE else 'on'}"
        )

    trainable_params = list(model.get_trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_lr,
        weight_decay=train_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=scheduler_eta_min,
    )
    print(
        f"Optimizer | lr={optimizer.param_groups[0]['lr']:.1e} | "
        f"weight_decay={optimizer.param_groups[0]['weight_decay']:.1e} | "
        f"scheduler T_max={scheduler_t_max} eta_min={scheduler_eta_min:.1e}"
    )
    scaler = GradScaler(device.type, enabled=_amp_enabled(device))
    feature_cache = _build_reference_cache(device, model=model)
    # Fail before any batch when cluster descriptors and the query cache disagree.
    _verify_reference_feature_provenance(model, feature_cache)

    if OVERFIT_ONE_BATCH:
        _run_overfit_one_batch(
            model,
            train_loader,
            optimizer,
            trainable_params,
            device,
            OUTPUT_DIR,
            feature_cache=feature_cache,
        )
        return

    train_losses: list = []
    val_losses: list = []
    train_metrics_history: list = []
    val_metrics_history: list = []
    best_val_loss = float("inf")
    val_metrics_csv_path = os.path.join(OUTPUT_DIR, "val_metrics_hw2.csv")
    csv_epoch_offset = 0
    if is_resume and os.path.isfile(val_metrics_csv_path):
        # Keep prior CSV rows. Newly added appearance columns stay empty for
        # older measurements (never back-fill fabricated zeros).
        prior_df = pd.read_csv(val_metrics_csv_path)
        for col in VAL_METRICS_CSV_COLUMNS:
            if col not in prior_df.columns:
                prior_df[col] = pd.NA
        prior_df = prior_df.reindex(columns=list(VAL_METRICS_CSV_COLUMNS))
        val_metrics_history = prior_df.where(prior_df.notna(), other=None).to_dict(
            orient="records"
        )
        if "Epoch_no" in prior_df.columns and len(prior_df) > 0:
            try:
                csv_epoch_offset = int(
                    pd.to_numeric(prior_df["Epoch_no"], errors="coerce").max()
                )
            except (TypeError, ValueError):
                csv_epoch_offset = len(prior_df)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        if epoch == 1 and INTERPRETABILITY_LOSS_WARMUP_EPOCHS > 0:
            print(
                f"  Dense/CC/SIM/NSS/KL + side-stage + priority losses "
                f"disabled for epochs 1-{INTERPRETABILITY_LOSS_WARMUP_EPOCHS}."
            )
        if epoch == INTERPRETABILITY_LOSS_WARMUP_EPOCHS + 1:
            print(
                "  Restoring dense/CC/SIM/NSS/KL + side-stage + priority "
                "losses to LOSS_LAMBDA values."
            )

        val_every_n_epochs = VAL_EVERY_N_EPOCHS
        if epoch >= 48:
            val_every_n_epochs = 1

        train_loss, train_metrics, appearance_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            trainable_params,
            device,
            epoch,
            OUTPUT_DIR,
            calculate_metrics=False,
            scaler=scaler,
            feature_cache=feature_cache,
        )
        run_validation = epoch % val_every_n_epochs == 0
        if run_validation:
            val_loss, val_metrics, _ = validate_one_epoch(
                model,
                val_loader,
                device,
                epoch,
                OUTPUT_DIR,
                feature_cache=feature_cache,
            )
        else:
            val_loss = None
            val_metrics = None
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss if val_loss is not None else float("nan"))
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(
            _format_val_metrics_csv_row(
                csv_epoch_offset + epoch,
                val_loss,
                val_metrics,
                appearance_stats=appearance_stats,
            )
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Mean train loss: {train_loss:.6f}")
        if run_validation:
            print(f"  Mean val loss:   {val_loss:.6f}")
        else:
            print(f"  Validation:    skipped (runs every {val_every_n_epochs} epochs)")
        print(f"  Current LR:      {current_lr:.2e}")
        app_loss = appearance_stats.get("app_loss") if appearance_stats else None
        app_count = (
            appearance_stats.get("App_loss_patch_count") if appearance_stats else None
        )
        if app_loss is None or app_loss != app_loss:
            app_txt = "n/a"
        else:
            app_txt = f"{float(app_loss):.6f}"
        print(
            f"  Train appearance | App_loss_patch_count="
            f"{0.0 if app_count is None else float(app_count):.0f} | "
            f"app_loss={app_txt}"
        )
        if train_metrics is not None:
            print(
                f"Train metrics | CC: {train_metrics['CC']:.4f} | "
                f"SIM: {train_metrics['SIM']:.4f} | "
                # f"AUC: {train_metrics['AUC']:.4f} | "
                # f"sAUC: {train_metrics['sAUC']:.4f} | "
                f"NSS: {train_metrics['NSS']:.4f}"
            )
        if val_metrics is not None:
            print(
                f"Val metrics   | CC: {val_metrics['CC']:.4f} | "
                f"SIM: {val_metrics['SIM']:.4f} | "
                f"NSS: {val_metrics['NSS']:.4f}"
            )

        update_loss_curve(train_losses, val_losses, OUTPUT_DIR)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_kwargs),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": val_metrics_history,
            # Identity only — large fixed reference arrays stay in visual_cluster_dir.
            "patch_cluster_reference": build_patch_cluster_reference_checkpoint_metadata(
                model
            ),
        }
        epoch_ckpt_path = os.path.join(run_ckpt_dir, f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, epoch_ckpt_path)
        torch.save(checkpoint, last_ckpt_path)
        print(f"  Saved checkpoint: {epoch_ckpt_path}")

        if run_validation and val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_ckpt_path)
            print(f"  New best val loss: {best_val_loss:.6f} (saved {best_ckpt_path})")

        # columns= forces the full header. Older resumed/history rows that omit
        # App_loss_patch_count / app_loss become empty cells (NaN), not zeros.
        val_metrics_df = pd.DataFrame(
            val_metrics_history,
            columns=list(VAL_METRICS_CSV_COLUMNS),
        )
        val_metrics_df.to_csv(
            val_metrics_csv_path,
            index=False,
            na_rep="",
        )

    print(f"\nTraining complete. Outputs saved to {OUTPUT_DIR}/")
    print(f"Run checkpoints: {run_ckpt_dir}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")


def _run_overfit_one_batch(
    model: ExplainableVidSalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    trainable_params: list,
    device: torch.device,
    output_dir: str,
    feature_cache: Optional[FrozenBackboneFeatureCache] = None,
) -> None:
    """Debug mode: repeatedly train on the first batch."""
    print(f"OVERFIT_ONE_BATCH: {OVERFIT_STEPS} steps on first batch")

    batch = next(iter(loader))
    (
        video_filenames,
        rgb_batch,
        sal_batch,
        fix_batch,
        n_frames,
        valid_mask,
        sample_metadata,
    ) = _unpack_loader_batch(batch)
    model.train()
    cached_backbone_features = None
    if _needs_reference_features(model):
        cached_backbone_features = _resolve_cached_backbone_features(
            feature_cache,
            rgb_batch,
            sample_metadata,
            n_frames,
            view_id="identity",
        )
    rgb_batch, sal_batch, fix_batch = model.prepare_training_batch(
        rgb_batch,
        sal_batch,
        fix_batch,
    )
    fix_batch = (fix_batch > 0).float()

    model.train()
    for step in range(1, OVERFIT_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        model_out = model(
            rgb_batch,
            saliency_maps=sal_batch,
            return_details=True,
            return_concept_losses=_return_concept_losses(model),
            return_decoder_diagnostics=True,
            cached_backbone_features=cached_backbone_features,
        )
        if step == 1:
            save_first_batch_maps(
                model_out, sal_batch, fix_batch, rgb_batch, output_dir
            )
            _print_first_batch_debug(model_out, sal_batch)
            _print_decoder_priority_debug(model, model_out)
            _print_patch_priority_diagnostics(model_out)
        loss_dict = _compute_batch_loss(
            model_out,
            sal_batch,
            fix_batch=fix_batch,
        )
        loss = loss_dict["loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"  step {step}/{OVERFIT_STEPS} loss_total={loss.item():.6f}")

    print("Overfit run complete.")


if __name__ == "__main__":
    main()
