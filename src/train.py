import os

# Reduce CUDA fragmentation (must be set before the first CUDA allocation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import sys
import math
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pre_process.collate import video_saliency_collate_fn
from model.losses import compute_total_loss
from model.metrics import MetricAverager, compute_saliency_metrics
from model.model import ExplainableVidSalModel
from pre_process.dataloader import DatasetLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Hide i8 when stdout is redirected (e.g. nohup ... > out.log).
SHOW_PROGRESS_BAR = sys.stdout.isatty()

TRAIN_DATASET_DIR = (
    "/data/quantization/zaima/dh1k/training"
)
VAL_DATASET_DIR = ( 
    "/data/quantization/zaima/dh1k/testing"
)
WINDOW_LEN = 16

EPOCHS = 100
BATCH_SIZE = 4  # effective optimizer batch size
FREEZE_BACKBONE = False
# When fine-tuning the backbone, use a smaller per-forward micro-batch and accumulate
# gradients so the optimizer still sees BATCH_SIZE samples per step.
MICRO_BATCH_SIZE = 4
# Gradient checkpointing disabled (saves memory when True, but adds recompute overhead).
BACKBONE_GRADIENT_CHECKPOINTING = False
SKIP_VISUAL_EQUIV_WHEN_BACKBONE_TRAINABLE = True
LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42
OUTPUT_DIR = "training_outputs"
CKPTS_DIR = os.path.join(OUTPUT_DIR, "ckpts")
MAP_SAVE_INTERVAL = 500
OVERFIT_ONE_BATCH = False
OVERFIT_STEPS = 300
MAX_SAMPLES = 500
USE_AMP = True

# Concept-branch switches. Set either branch to False for ablations.
VISUAL_CONCEPT_ON = True
TEMPORAL_CONCEPTS_ON = False
VISUAL_CONCEPT_LOGIT_SCALE = 1.0

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
    "lambda_dense": 0.05,
    "lambda_bce": 0.0,
    "lambda_kl": 2.0,
    "lambda_cc": 0.5,
    "lambda_nss": 0.5,

    # Disable explicit background suppression for now
    "topk_percent": 0.005,
    "topk_bg_weight": 0.0,

    # Disable old concept-map saliency supervision for now
    "lambda_concept_dense": 0.0,
    "lambda_concept_kl": 0.0,

    # Temporarily weaken concept regularizers until maps stop collapsing
    "lambda_align": 0.05,
    "lambda_sparse": 0.05,
    "lambda_div": 0.05,
    "lambda_gate": 0.0,

    # Temporarily reduce visual regularizers
    "lambda_visual_entropy": 0.02,
    "lambda_visual_usage": 0.05,
    "lambda_visual_equiv": 0.02,
    "lambda_temporal_attention_entropy": 0.0,

    "patch_from_logits": False,

    # Deep supervision on per-stage decoder side outputs (training only).
    "side_stage_weights": {
        "stage1": 0.20,
        "stage2": 0.15,
        "stage3": 0.10,
        "stage4": 0.05,
    },
}

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


def _should_run_visual_equiv(model: ExplainableVidSalModel, epoch: Optional[int] = None) -> bool:
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
    """Disable branch-specific auxiliary losses when a branch is ablated."""
    loss_lambda = dict(LOSS_LAMBDA)
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


def _return_concept_losses() -> bool:
    """True only when enabled concept regularizers need auxiliary losses."""
    loss_lambda = _effective_loss_lambda()
    return any(
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
            "concept_context_patch_logits.png": pred_out.get("concept_context_patch_logits"),
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
    save_batch_maps(model_out, sal_batch, fix_batch, rgb_batch, output_dir, epoch=1, batch_idx=0)


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


def _print_temporal_aggregation_debug(model_out: dict) -> None:
    pred_out = model_out.get("prediction_out")
    if pred_out is None:
        return

    tw = pred_out.get("temporal_transition_weights")
    if tw is not None:
        print("temporal_transition_weights shape:", tuple(tw.shape))
        print(
            "temporal weight min/max/mean:",
            tw.min().item(),
            tw.max().item(),
            tw.mean().item(),
        )
        print("temporal weight sum over time mean:", tw.sum(dim=1).mean().item())

    tpl = pred_out.get("temporal_patch_saliency_logits")
    if tpl is not None:
        print("temporal_patch_saliency_logits shape:", tuple(tpl.shape))


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
    _print_temporal_aggregation_debug(model_out)


def _print_gate_debug(
    model_out: dict,
    model: Optional[ExplainableVidSalModel] = None,
    sal_batch: Optional[torch.Tensor] = None,
) -> None:
    concept_out = model_out.get("concept_out")
    if not isinstance(concept_out, dict):
        print("DEBUG gate: no concept_out found")
        return

    found = False

    for stage, out_s in concept_out.items():
        if not isinstance(out_s, dict):
            continue

        losses = out_s.get("losses", {})
        dbg: Optional[dict] = None
        if isinstance(losses, dict):
            maybe_dbg = losses.get("gate_debug")
            if isinstance(maybe_dbg, dict) and maybe_dbg:
                dbg = maybe_dbg

        if dbg is None and model is not None and sal_batch is not None:
            features_shape = model_out.get("features_shape", {}).get(stage)
            gate_probs = out_s.get("gate_probs")
            metadata = out_s.get("metadata")
            if (
                features_shape is not None
                and torch.is_tensor(gate_probs)
                and isinstance(metadata, dict)
                and stage in model.concept_creations
            ):
                dbg = model.concept_creations[stage].summarize_gate_debug(
                    sal_batch,
                    metadata,
                    gate_probs,
                    tuple(features_shape),
                )

        if not isinstance(dbg, dict) or not dbg:
            continue

        found = True

        valid_total = 100.0 * float(dbg.get("gate_valid_frac_total", 0.0))
        tr_total = 100.0 * float(dbg.get("gate_transition_frac_total", 0.0))
        per_total = 100.0 * float(dbg.get("gate_persistence_frac_total", 0.0))
        amb_total = 100.0 * float(dbg.get("gate_ambiguous_frac_total", 0.0))

        tr_valid = 100.0 * float(dbg.get("gate_transition_frac_valid", 0.0))
        per_valid = 100.0 * float(dbg.get("gate_persistence_frac_valid", 0.0))

        conf_mean = float(dbg.get("gate_confidence_mean", 0.0))
        conf_valid_mean = float(dbg.get("gate_confidence_valid_mean", 0.0))
        visual_mean = float(dbg.get("gate_visual_sim_mean", 0.0))
        visual_std = float(dbg.get("gate_visual_sim_std", 0.0))
        visual_tr_mean = float(dbg.get("gate_visual_sim_transition_mean", 0.0))
        visual_tr_std = float(dbg.get("gate_visual_sim_transition_std", 0.0))
        visual_per_mean = float(dbg.get("gate_visual_sim_persistence_mean", 0.0))
        visual_per_std = float(dbg.get("gate_visual_sim_persistence_std", 0.0))
        visual_valid_mean = float(dbg.get("gate_visual_sim_valid_mean", 0.0))
        visual_valid_std = float(dbg.get("gate_visual_sim_valid_std", 0.0))

        n_tr = int(dbg.get("gate_transition_count", 0))
        n_per = int(dbg.get("gate_persistence_count", 0))
        n_valid = int(dbg.get("gate_valid_count", 0))
        n_total = int(dbg.get("gate_total_count", 0))
        delta_mean = float(dbg.get("gate_delta_abs_mean", 0.0))
        dist_mean = float(dbg.get("gate_dist_mean", 0.0))

        print(
            f"DEBUG gate [{stage}] | "
            f"valid: {valid_total:.2f}% | "
            f"transition: {tr_total:.2f}% total / {tr_valid:.2f}% valid | "
            f"persistence: {per_total:.2f}% total / {per_valid:.2f}% valid | "
            f"ambiguous: {amb_total:.2f}% | "
            f"conf mean: {conf_mean:.4f} | "
            f"conf valid mean: {conf_valid_mean:.4f} | "
            f"visual_sim all: {visual_mean:.4f}±{visual_std:.4f} | "
            f"visual_sim transition: {visual_tr_mean:.4f}±{visual_tr_std:.4f} (n={n_tr}) | "
            f"visual_sim persistence: {visual_per_mean:.4f}±{visual_per_std:.4f} (n={n_per}) | "
            f"visual_sim valid: {visual_valid_mean:.4f}±{visual_valid_std:.4f} (n={n_valid}/{n_total}) | "
            f"|delta_s| mean: {delta_mean:.4f} | "
            f"dist mean: {dist_mean:.4f}"
        )

    if not found:
        print(
            "DEBUG gate: no gate_debug available. "
            "This usually means return_concept_losses=False or lambda_gate=0."
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
            usage_entropy = -(
                mean_probs * (mean_probs + 1e-8).log()
            ).sum() / max(math.log(mean_probs.numel()), 1e-8)
            print(
                f"  normalized usage entropy: {float(usage_entropy.cpu()):.4f}"
            )


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
) -> Tuple[float, Optional[Dict[str, float]]]:
    model.train()
    running_loss = 0.0
    num_batches = 0
    metric_averager = MetricAverager()
    accum_steps = _grad_accum_steps()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        loader,
        desc=f"Train epoch {epoch}",
        leave=False,
        # disable=not SHOW_PROGRESS_BAR,
    )
    for batch_idx, (
        video_filenames,
        rgb_batch,
        sal_batch,
        fix_batch,
        n_frames,
        valid_mask,
    ) in enumerate(pbar):
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        if not torch.is_tensor(sal_batch):
            raise ValueError("sal_batch must be a torch.Tensor for loss calculation.")
        sal_batch = sal_batch.to(device, non_blocking=True)
        fix_batch = fix_batch.to(device, non_blocking=True)
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
                return_concept_losses=_return_concept_losses(),
            )

            if _should_run_visual_equiv(model, epoch):
                rgb_batch_flip = torch.flip(rgb_batch, dims=[-1])
                equiv_model_out = model(
                    rgb_batch_flip,
                    saliency_maps=sal_batch,
                    return_details=True,
                    return_concept_losses=_return_concept_losses(),
                )

            if batch_idx == 0:
                _print_first_batch_debug(model_out, sal_batch)
                _print_gate_debug(model_out, model=model, sal_batch=sal_batch)
                _print_visual_concept_usage_debug(model_out)

            if batch_idx % MAP_SAVE_INTERVAL == 0:
                save_batch_maps(
                    model_out, sal_batch, fix_batch, rgb_batch, output_dir, epoch, batch_idx
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
                target_last = sal_batch[:, -1:].detach().float()
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
                target_fix_last = fix_batch[:, -1:].detach().float()
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
        del model_out, loss_dict, loss, rgb_batch, sal_batch, fix_batch
        if equiv_model_out is not None:
            del equiv_model_out

    mean_loss = running_loss / max(num_batches, 1)
    if calculate_metrics:
        return mean_loss, metric_averager.mean()
    else:
        return mean_loss, None


@torch.no_grad()
def validate_one_epoch(
    model: ExplainableVidSalModel,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: str,
) -> Tuple[float, Optional[Dict[str, float]]]:
    model.eval()
    running_loss = 0.0
    num_batches = 0
    metric_averager = MetricAverager()

    pbar = tqdm(
        loader,
        desc=f"Val epoch {epoch}",
        leave=False,
        # disable=not SHOW_PROGRESS_BAR,
    )
    for batch_idx, (
        video_filenames,
        rgb_batch,
        sal_batch,
        fix_batch,
        n_frames,
        valid_mask,
    ) in enumerate(pbar):
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        if not torch.is_tensor(sal_batch):
            raise ValueError("sal_batch must be a torch.Tensor for loss calculation.")
        sal_batch = sal_batch.to(device, non_blocking=True)
        fix_batch = fix_batch.to(device, non_blocking=True)
        fix_batch = (fix_batch > 0).float()

        with autocast(
            device.type,
            dtype=_amp_dtype(device),
            enabled=_amp_enabled(device),
        ):
            model_out = model(
                rgb_batch,
                saliency_maps=sal_batch,
                return_details=True,
                return_concept_losses=_return_concept_losses(),
            )
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
                model_out, sal_batch, fix_batch=fix_batch, epoch=epoch,
                enable_side_aux=False,
            )
            batch_loss = loss_dict["loss_total"].item()
        running_loss += batch_loss
        num_batches += 1

        metric_dict = _compute_batch_metrics(model_out, sal_batch, fix_batch)
        metric_averager.update(metric_dict, batch_size=rgb_batch.shape[0])
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            NSS=f"{metric_averager.mean()['NSS']:.4f}",
        )

        del model_out, loss_dict, rgb_batch, sal_batch, fix_batch

    mean_loss = running_loss / max(num_batches, 1)
    return mean_loss, metric_averager.mean()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = os.path.join(CKPTS_DIR, run_timestamp)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(run_ckpt_dir, "best_checkpoint.pth")
    last_ckpt_path = os.path.join(run_ckpt_dir, "last_checkpoint.pth")
    set_seed(SEED)


    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run checkpoints will be saved to: {run_ckpt_dir}")
    print(
        "Concept branches | "
        f"visual_concept_on={VISUAL_CONCEPT_ON} | "
        f"temporal_concepts_on={TEMPORAL_CONCEPTS_ON} | "
        f"visual_concept_logit_scale={VISUAL_CONCEPT_LOGIT_SCALE}"
    )

    train_dataset = DatasetLoader(TRAIN_DATASET_DIR, window_len=WINDOW_LEN, stride=16)
    val_dataset = DatasetLoader(VAL_DATASET_DIR, window_len=WINDOW_LEN, stride=16)

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

    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=True,
        freeze_backbone=FREEZE_BACKBONE,
        backbone_gradient_checkpointing=False,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=128,
        num_concepts=512,
        concept_hidden_dim=256,
        saliency_hidden_dim=96,
        top_k=3,
        max_source_patches=64,
        tau_pi=0.5,
        tau_alpha=0.07,
        tau_concept=0.07,
        concept_residual_weight=0.0,
        last_transition_only=True,
        use_rgb_refinement=False,
        use_feature_refinement=False,
        output_activation="none",
        return_details=True,
        use_subpatch_head=True,
        subpatch_factor=4,
        subpatch_residual_scale=0.5,
        use_temporal_transition_aggregation=True,
        temporal_aggregation_hidden_channels=128,
        temporal_aggregation_temperature=1.0,
        visual_concept_on=VISUAL_CONCEPT_ON,
        temporal_concepts_on=TEMPORAL_CONCEPTS_ON,
        visual_concept_logit_scale=VISUAL_CONCEPT_LOGIT_SCALE,
        visual_concept_residual_weight=1.0,
    ).to(device)

    with torch.no_grad():
        # Initialize final saliency logits to the observed dataset prior.
        target_prior = 0.0177
        bias_value = math.log(target_prior / (1.0 - target_prior))
        initialized = []

        # New spatiotemporal decoder: learned upsample head + optional patch head.
        final_head = getattr(model.saliency_prediction, "final_upsample_head", None)
        if final_head is not None and hasattr(final_head, "head"):
            final_conv = final_head.head
            if hasattr(final_conv, "bias") and final_conv.bias is not None:
                final_conv.bias.fill_(bias_value)
                initialized.append("final_upsample_head.head")

        patch_head = getattr(model.saliency_prediction, "patch_logit_head", None)
        if patch_head is not None and hasattr(patch_head, "bias") and patch_head.bias is not None:
            patch_head.bias.fill_(bias_value)
            initialized.append("patch_logit_head")

        # Backward compatibility: legacy decoder had pred_head[-1].
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
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )
    scaler = GradScaler(device.type, enabled=_amp_enabled(device))

    if OVERFIT_ONE_BATCH:
        _run_overfit_one_batch(
            model, train_loader, optimizer, trainable_params, device, OUTPUT_DIR
        )
        return

    train_losses: list = []
    val_losses: list = []
    train_metrics_history: list = []
    val_metrics_history: list = []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            trainable_params,
            device,
            epoch,
            OUTPUT_DIR,
            calculate_metrics=False,
            scaler=scaler,
        )
        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, device, epoch, OUTPUT_DIR
        )
        # if epoch == 3:
            # val_loss, val_metrics = validate_one_epoch(model, val_loader, device, epoch)
        # else:
            # val_loss, val_metrics = 0, None
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Mean train loss: {train_loss:.6f}")
        print(f"  Mean val loss:   {val_loss:.6f}")
        print(f"  Current LR:      {current_lr:.2e}")
        if train_metrics is not None:
            print(
                f"Train metrics | CC: {train_metrics['CC']:.4f} | "
                f"SIM: {train_metrics['SIM']:.4f} | AUC: {train_metrics['AUC']:.4f} | "
                f"sAUC: {train_metrics['sAUC']:.4f} | NSS: {train_metrics['NSS']:.4f}"
            )
        if val_metrics is not None:
            print(
                f"Val metrics   | CC: {val_metrics['CC']:.4f} | "
                f"SIM: {val_metrics['SIM']:.4f} | AUC: {val_metrics['AUC']:.4f} | "
                f"sAUC: {val_metrics['sAUC']:.4f} | NSS: {val_metrics['NSS']:.4f}"
            )

        update_loss_curve(train_losses, val_losses, OUTPUT_DIR)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_metrics_history": train_metrics_history,
            "val_metrics_history": val_metrics_history,
        }
        epoch_ckpt_path = os.path.join(run_ckpt_dir, f"epoch_{epoch:03d}.pth")
        torch.save(checkpoint, epoch_ckpt_path)
        torch.save(checkpoint, last_ckpt_path)
        print(f"  Saved checkpoint: {epoch_ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_ckpt_path)
            print(f"  New best val loss: {best_val_loss:.6f} (saved {best_ckpt_path})")

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
) -> None:
    """Debug mode: repeatedly train on the first batch."""
    print(f"OVERFIT_ONE_BATCH: {OVERFIT_STEPS} steps on first batch")

    batch = next(iter(loader))
    video_filenames, rgb_batch, sal_batch, fix_batch, n_frames, valid_mask = batch
    rgb_batch = rgb_batch.to(device)
    sal_batch = sal_batch.to(device)
    fix_batch = fix_batch.to(device)
    fix_batch = (fix_batch > 0).float()

    model.train()
    for step in range(1, OVERFIT_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        model_out = model(
            rgb_batch,
            saliency_maps=sal_batch,
            return_details=True,
            return_concept_losses=_return_concept_losses(),
        )
        if step == 1:
            save_first_batch_maps(model_out, sal_batch, fix_batch, rgb_batch, output_dir)
            _print_first_batch_debug(model_out, sal_batch)
            _print_gate_debug(model_out, model=model, sal_batch=sal_batch)

        loss_dict = _compute_batch_loss(model_out, sal_batch, fix_batch=fix_batch)
        loss = loss_dict["loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"  step {step}/{OVERFIT_STEPS} loss_total={loss.item():.6f}")

    print("Overfit run complete.")


if __name__ == "__main__":
    main()
