"""
Loss functions for the explainable video saliency model.

All supervision is computed outside the model from:
  - dense last-frame saliency targets
  - patch-grid fidelity and delta targets on the feature grid
  - optional concept regularizers from ConceptCreation
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}")
    x = x.float()
    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    return x


def prepare_saliency_sequence(saliency_maps: torch.Tensor) -> torch.Tensor:
    """
    Normalize saliency layout to [B, 1, T, H, W].

    Accepts [B,H,W], [B,1,H,W], [B,T,H,W], [B,T,1,H,W], or [B,1,T,H,W].
    """
    if not isinstance(saliency_maps, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(saliency_maps)}")

    x = _to_float_tensor(saliency_maps)

    if x.dim() == 3:
        # [B, H, W] -> T=1
        return x.unsqueeze(1).unsqueeze(1)
    if x.dim() == 4:
        if x.shape[1] == 1:
            # [B, 1, H, W]
            return x.unsqueeze(2)
        # [B, T, H, W]
        return x.unsqueeze(1)
    if x.dim() == 5:
        if x.shape[1] == 1:
            # [B, 1, T, H, W]
            return x
        if x.shape[2] == 1:
            # [B, T, 1, H, W] -> [B, 1, T, H, W]
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(
            f"Unsupported 5D saliency shape {tuple(x.shape)}; "
            "expected [B,1,T,H,W] or [B,T,1,H,W]"
        )

    raise ValueError(
        f"Unsupported saliency_maps shape {tuple(x.shape)}; "
        "expected 3D, 4D, or 5D tensor"
    )


def resize_saliency_sequence(
    saliency_maps: torch.Tensor,
    target_t: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize saliency to [B, 1, target_t, target_h, target_w]."""
    x = prepare_saliency_sequence(saliency_maps)
    if x.shape[2:] == (target_t, target_h, target_w):
        return x
    return F.interpolate(
        x,
        size=(target_t, target_h, target_w),
        mode="trilinear",
        align_corners=False,
    )


def prepare_last_saliency_map(
    saliency_maps: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Last temporal frame, resized to [B, 1, target_h, target_w]."""
    x = prepare_saliency_sequence(saliency_maps)
    last = x[:, :, -1, :, :]  # [B, 1, H, W]
    if last.shape[-2:] == (target_h, target_w):
        return last
    return F.interpolate(
        last,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )


def has_temporal_saliency_sequence(saliency_maps: torch.Tensor) -> bool:
    """True only when saliency_maps has at least two real temporal frames."""
    if not isinstance(saliency_maps, torch.Tensor):
        return False
    if saliency_maps.dim() == 4 and saliency_maps.shape[1] >= 2:
        return True  # [B, T, H, W]
    if saliency_maps.dim() == 5:
        if saliency_maps.shape[1] == 1 and saliency_maps.shape[2] >= 2:
            return True  # [B, 1, T, H, W]
        if saliency_maps.shape[2] == 1 and saliency_maps.shape[1] >= 2:
            return True  # [B, T, 1, H, W]
    return False


def prepare_patch_target_from_last_frame(
    saliency_maps: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Last-frame GT saliency resized to feature patch grid [B, 1, H, W]."""
    return prepare_last_saliency_map(saliency_maps, target_h, target_w)


def _read_feature_shape_from_metadata(metadata: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    if "feature_shape" not in metadata:
        raise ValueError("metadata must contain 'feature_shape'")

    feature_shape = metadata["feature_shape"]
    if isinstance(feature_shape, dict):
        B = int(feature_shape["B"])
        C = int(feature_shape["C"])
        T = int(feature_shape["T"])
        H = int(feature_shape["H"])
        W = int(feature_shape["W"])
    else:
        shape = tuple(int(v) for v in feature_shape)
        if len(shape) != 5:
            raise ValueError(
                f"feature_shape must have 5 entries (B,C,T,H,W), got {shape}"
            )
        B, C, T, H, W = shape
    return B, C, T, H, W


def _to_long_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long).reshape(-1)
    return torch.tensor(x, device=device, dtype=torch.long).reshape(-1)


def _aggregate_incoming_source_saliency(
    source_values: torch.Tensor,
    pi: torch.Tensor,
    batch_idx: torch.Tensor,
    target_idx: torch.Tensor,
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Aggregate incoming source saliency onto the target patch grid.

    incoming_source[b, target] += pi * source_value
    """
    device = source_values.device
    dtype = source_values.dtype
    N = H * W
    flat_idx = batch_idx.long() * N + target_idx.long()
    weighted = source_values * pi
    patch_flat = torch.zeros(B * N, device=device, dtype=dtype)
    patch_flat.index_add_(0, flat_idx, weighted)
    return patch_flat.view(B, 1, H, W)


def _aggregate_target_patch_grid(
    target_values: torch.Tensor,
    batch_idx: torch.Tensor,
    target_idx: torch.Tensor,
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Mean target saliency per patch when multiple trajectories share a target."""
    device = target_values.device
    dtype = target_values.dtype
    N = H * W
    flat_idx = batch_idx.long() * N + target_idx.long()

    target_sum = torch.zeros(B * N, device=device, dtype=dtype)
    target_count = torch.zeros(B * N, device=device, dtype=dtype)
    target_sum.index_add_(0, flat_idx, target_values)
    target_count.index_add_(0, flat_idx, torch.ones_like(target_values))
    target_patch = target_sum / target_count.clamp(min=1.0)
    return target_patch.view(B, 1, H, W)


def compute_delta_target(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """
    Patch-grid delta target for the final transition into the last feature frame.

    Delta s_b = s_b^{t+1} - sum_a pi_ab * s_a^t

    Returns:
        delta_target, target_patch_grid, source_mixture_grid — each [B, 1, H, W],
        or (None, None, None) when saliency_maps has no real temporal sequence.
    """
    if not has_temporal_saliency_sequence(saliency_maps):
        return None, None, None

    required_pred = ("incoming_weights", "selected_metadata")
    for key in required_pred:
        if key not in prediction_out:
            raise ValueError(f"prediction_out must contain '{key}'")

    metadata = prediction_out["selected_metadata"]
    required_meta = (
        "batch_idx",
        "time_idx",
        "source_idx",
        "target_idx",
        "feature_shape",
    )
    for key in required_meta:
        if key not in metadata:
            raise ValueError(f"selected_metadata must contain '{key}'")

    B, _, T, H, W = _read_feature_shape_from_metadata(metadata)
    N = H * W

    pi = _to_float_tensor(prediction_out["incoming_weights"])
    device = pi.device

    batch_idx = _to_long_tensor(metadata["batch_idx"], device)
    time_idx = _to_long_tensor(metadata["time_idx"], device)
    source_idx = _to_long_tensor(metadata["source_idx"], device)
    target_idx = _to_long_tensor(metadata["target_idx"], device)

    sal_seq = resize_saliency_sequence(saliency_maps, T, H, W)  # [B, 1, T, H, W]
    sal_flat = sal_seq.squeeze(1).reshape(B, T, N)  # [B, T, N]

    s_a_t = sal_flat[batch_idx, time_idx, source_idx]
    s_b_t1 = sal_flat[batch_idx, time_idx + 1, target_idx]

    source_mixture_grid = _aggregate_incoming_source_saliency(
        s_a_t, pi, batch_idx, target_idx, B, H, W
    )
    target_patch_grid = _aggregate_target_patch_grid(
        s_b_t1, batch_idx, target_idx, B, H, W
    )
    delta_target = target_patch_grid - source_mixture_grid
    return delta_target, target_patch_grid, source_mixture_grid


def _get_zero_like_loss(reference_tensor: torch.Tensor) -> torch.Tensor:
    return reference_tensor.sum() * 0.0


def minmax_per_sample(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dim() < 3:
        return x
    B = x.shape[0]
    flat = x.reshape(B, -1)
    xmin = flat.min(dim=1, keepdim=True)[0]
    xmax = flat.max(dim=1, keepdim=True)[0]
    flat = (flat - xmin) / (xmax - xmin + eps)
    return flat.view_as(x)


def topk_weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    top_percent: float = 0.005,
    bg_weight: float = 0.15,
) -> torch.Tensor:
    """
    Weighted L1 loss that emphasizes the top target saliency pixels.

    Args:
        pred: [B,1,H,W], predicted saliency map in [0,1] or logits after activation.
        target: [B,1,H,W], target saliency map in [0,1].
        top_percent: fraction of pixels with highest target saliency to emphasize.
        bg_weight: weight for non-top-k pixels.

    Returns:
        scalar loss.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have same shape, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )

    B = pred.shape[0]
    pred_f = pred.reshape(B, -1)
    target_f = target.reshape(B, -1)

    n_pixels = pred_f.shape[1]
    k = max(1, int(top_percent * n_pixels))

    _, top_idx = torch.topk(target_f, k=k, dim=1)

    weights = torch.full_like(target_f, bg_weight)
    weights.scatter_(1, top_idx, 1.0)

    return (weights * (pred_f - target_f).abs()).mean()


def spatial_kl_loss(
    pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if pred_logits.shape != target.shape:
        raise ValueError(
            f"pred_logits and target must have the same shape, got "
            f"{tuple(pred_logits.shape)} and {tuple(target.shape)}"
        )

    B = pred_logits.shape[0]
    pred_flat = pred_logits.reshape(B, -1)
    target_flat = target.clamp(min=0.0).reshape(B, -1)
    target_flat = target_flat / target_flat.sum(dim=1, keepdim=True).clamp(min=eps)

    log_pred = F.log_softmax(pred_flat, dim=1)
    return F.kl_div(log_pred, target_flat, reduction="batchmean")


def compute_fidelity_loss(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    lambda_delta: float = 1.0,
    lambda_dense: float = 1.0,
    lambda_bce: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_topk: float = 0.0,
    topk_percent: float = 0.05,
    topk_bg_weight: float = 0.15,
    patch_from_logits: bool = True,
) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Fidelity loss on patch saliency, optional patch delta, and dense last-frame saliency.

    Patch target always uses the last-frame GT (no fake temporal interpolation).
    Delta loss only when saliency_maps has T >= 2.
    """
    required = ("saliency_map", "patch_saliency_logits")
    for key in required:
        if key not in prediction_out:
            raise ValueError(f"prediction_out must contain '{key}'")

    patch_logits = prediction_out["patch_saliency_logits"]
    if patch_logits.dim() != 4 or patch_logits.shape[1] != 1:
        raise ValueError(
            f"patch_saliency_logits must be [B,1,H,W], got {tuple(patch_logits.shape)}"
        )

    if patch_from_logits:
        patch_saliency_pred = torch.sigmoid(patch_logits)
    else:
        patch_saliency_pred = patch_logits

    target_patch_grid = prepare_patch_target_from_last_frame(
        saliency_maps,
        patch_logits.shape[-2],
        patch_logits.shape[-1],
    )
    target_patch_grid = minmax_per_sample(target_patch_grid)
    loss_patch_fid = F.l1_loss(patch_saliency_pred, target_patch_grid)

    target_dense_last = prepare_last_saliency_map(
        saliency_maps,
        prediction_out["saliency_map"].shape[-2],
        prediction_out["saliency_map"].shape[-1],
    )
    target_dense_last = minmax_per_sample(target_dense_last)
    pred_dense = prediction_out["saliency_map"]
    if target_dense_last.shape[-2:] != pred_dense.shape[-2:]:
        target_dense_last = F.interpolate(
            target_dense_last,
            size=pred_dense.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    loss_dense_fid = F.l1_loss(pred_dense, target_dense_last)

    loss_topk = _get_zero_like_loss(patch_saliency_pred)
    if lambda_topk > 0:
        loss_topk = topk_weighted_l1_loss(
            pred_dense,
            target_dense_last,
            top_percent=topk_percent,
            bg_weight=topk_bg_weight,
        )

    loss_dense_bce = _get_zero_like_loss(patch_saliency_pred)
    if lambda_bce > 0:
        loss_dense_bce = F.binary_cross_entropy_with_logits(
            prediction_out["saliency_logits"],
            target_dense_last.clamp(0.0, 1.0),
        )

    loss_dense_kl = _get_zero_like_loss(patch_saliency_pred)
    if lambda_kl > 0:
        loss_dense_kl = spatial_kl_loss(
            prediction_out["saliency_logits"],
            target_dense_last,
        )

    patch_delta_logits = prediction_out.get("patch_delta_logits")
    use_delta = (
        has_temporal_saliency_sequence(saliency_maps)
        and patch_delta_logits is not None
    )

    if use_delta:
        delta_target, _, source_mixture_grid = compute_delta_target(
            prediction_out, saliency_maps
        )
        loss_delta = F.l1_loss(patch_delta_logits, delta_target)
    else:
        delta_target = None
        source_mixture_grid = None
        loss_delta = _get_zero_like_loss(patch_saliency_pred)

    loss_fid = (
        loss_patch_fid
        + lambda_delta * loss_delta
        + lambda_dense * loss_dense_fid
        + lambda_bce * loss_dense_bce
        + lambda_kl * loss_dense_kl
        + lambda_topk * loss_topk
    )

    return {
        "loss_fid": loss_fid,
        "loss_patch_fid": loss_patch_fid,
        "loss_dense_fid": loss_dense_fid,
        "loss_dense_bce": loss_dense_bce,
        "loss_dense_kl": loss_dense_kl,
        "loss_topk": loss_topk,
        "loss_delta": loss_delta,
        "delta_target": delta_target,
        "target_patch_grid": target_patch_grid,
        "source_mixture_grid": source_mixture_grid,
    }


def _resolve_prediction_out(model_out: Dict[str, Any]) -> Dict[str, Any]:
    if "prediction_out" in model_out:
        return model_out["prediction_out"]
    return model_out


def _resolve_primary_concept_out(
    concept_out: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Single-scale concept_out or primary stage from multiscale dict."""
    if concept_out is None:
        return None
    if not isinstance(concept_out, dict) or not concept_out:
        return None

    first_val = next(iter(concept_out.values()))
    if isinstance(first_val, dict) and (
        "concept_representation" in first_val or "metadata" in first_val
    ):
        if "stage1" in concept_out:
            return concept_out["stage1"]
        return first_val
    return concept_out


def _inject_feature_shape(
    prediction_out: Dict[str, Any],
    concept_out: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Ensure selected_metadata carries feature_shape for delta targets."""
    metadata = prediction_out.get("selected_metadata")
    if metadata is None:
        raise ValueError("prediction_out must contain 'selected_metadata'")

    if "feature_shape" in metadata:
        return prediction_out

    primary_concept_out = _resolve_primary_concept_out(concept_out)
    if primary_concept_out is not None and "metadata" in primary_concept_out:
        fs = primary_concept_out["metadata"].get("feature_shape")
        if fs is not None:
            metadata = dict(metadata)
            metadata["feature_shape"] = fs
            prediction_out = dict(prediction_out)
            prediction_out["selected_metadata"] = metadata
            return prediction_out

    raise ValueError(
        "feature_shape missing from selected_metadata and concept_out.metadata"
    )


def _aggregate_concept_losses(
    concept_out: Any, reference: torch.Tensor
) -> Dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0

    keys = ["loss_align", "loss_sparse", "loss_div", "loss_gate"]
    totals = {k: zero for k in keys}
    count = 0

    if isinstance(concept_out, dict) and concept_out:
        first_val = next(iter(concept_out.values()))
        if isinstance(first_val, dict) and (
            "losses" in first_val or "concept_representation" in first_val
        ):
            iterable = concept_out.values()
        else:
            iterable = [concept_out]
    elif isinstance(concept_out, dict):
        iterable = [concept_out]
    else:
        iterable = []

    for out in iterable:
        losses = out.get("losses", {}) if isinstance(out, dict) else {}
        if not losses:
            continue
        count += 1
        for k in keys:
            if k in losses and losses[k] is not None:
                totals[k] = totals[k] + losses[k]

    if count > 0:
        for k in keys:
            totals[k] = totals[k] / count

    return totals


def compute_concept_branch_loss(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    lambda_dense: float = 0.0,
    lambda_kl: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Fidelity on fused concept-only saliency (explainability branch)."""
    ref = prediction_out.get("patch_saliency_logits")
    if ref is None:
        ref = prediction_out.get("saliency_logits")
    if ref is None:
        raise ValueError("prediction_out missing reference tensor for concept branch loss")

    zero = _get_zero_like_loss(ref)
    concept_map = prediction_out.get("concept_saliency_map")
    concept_logits = prediction_out.get("concept_saliency_logits")

    if concept_map is None or concept_logits is None:
        return {
            "loss_concept_fid": zero,
            "loss_concept_dense": zero,
            "loss_concept_kl": zero,
        }

    target_dense_last = prepare_last_saliency_map(
        saliency_maps,
        concept_map.shape[-2],
        concept_map.shape[-1],
    )
    target_dense_last = minmax_per_sample(target_dense_last)

    loss_concept_dense = zero
    if lambda_dense > 0:
        loss_concept_dense = F.l1_loss(concept_map, target_dense_last)

    loss_concept_kl = zero
    if lambda_kl > 0:
        loss_concept_kl = spatial_kl_loss(concept_logits, target_dense_last)

    loss_concept_fid = loss_concept_dense + loss_concept_kl

    return {
        "loss_concept_fid": loss_concept_fid,
        "loss_concept_dense": loss_concept_dense,
        "loss_concept_kl": loss_concept_kl,
    }


def compute_total_loss(
    model_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    lambda_delta: float = 1.0,
    lambda_dense: float = 1.0,
    lambda_bce: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_fid: float = 1.0,
    lambda_topk: float = 0.0,
    topk_percent: float = 0.05,
    topk_bg_weight: float = 0.15,
    lambda_concept_dense: float = 0.0,
    lambda_concept_kl: float = 0.0,
    lambda_align: float = 0.1,
    lambda_sparse: float = 0.01,
    lambda_div: float = 0.1,
    lambda_gate: float = 0.1,
    patch_from_logits: bool = True,
) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Total training loss for ExplainableVidSalModel (return_details=True).

    L = L_fid
      + lambda_align * L_align
      + lambda_sparse * L_sparse
      + lambda_div * L_div
      + lambda_gate * L_gate
    """
    prediction_out = _resolve_prediction_out(model_out)
    concept_out = model_out.get("concept_out")

    prediction_out = _inject_feature_shape(prediction_out, concept_out)

    fid_out = compute_fidelity_loss(
        prediction_out,
        saliency_maps,
        lambda_delta=lambda_delta,
        lambda_dense=lambda_dense,
        lambda_bce=lambda_bce,
        lambda_kl=lambda_kl,
        lambda_topk=lambda_topk,
        topk_percent=topk_percent,
        topk_bg_weight=topk_bg_weight,
        patch_from_logits=patch_from_logits,
    )

    loss_fid = fid_out["loss_fid"]
    ref = loss_fid

    use_concept_reg = (
        lambda_align > 0.0
        or lambda_sparse > 0.0
        or lambda_div > 0.0
        or lambda_gate > 0.0
    )
    if use_concept_reg:
        concept_losses = _aggregate_concept_losses(concept_out, ref)
        loss_align = concept_losses["loss_align"]
        loss_sparse = concept_losses["loss_sparse"]
        loss_div = concept_losses["loss_div"]
        loss_gate = concept_losses["loss_gate"]
    else:
        zero = ref.sum() * 0.0
        loss_align = loss_sparse = loss_div = loss_gate = zero

    use_concept_branch = lambda_concept_dense > 0.0 or lambda_concept_kl > 0.0
    if use_concept_branch:
        concept_branch_out = compute_concept_branch_loss(
            prediction_out,
            saliency_maps,
            lambda_dense=lambda_concept_dense,
            lambda_kl=lambda_concept_kl,
        )
        loss_concept_dense = concept_branch_out["loss_concept_dense"]
        loss_concept_kl = concept_branch_out["loss_concept_kl"]
    else:
        zero = ref.sum() * 0.0
        loss_concept_dense = loss_concept_kl = zero

    loss_total = (
        lambda_fid * loss_fid
        + lambda_align * loss_align
        + lambda_sparse * loss_sparse
        + lambda_div * loss_div
        + lambda_gate * loss_gate
        + lambda_concept_dense * loss_concept_dense
        + lambda_concept_kl * loss_concept_kl
    )

    return {
        "loss_total": loss_total,
        "loss_fid": fid_out["loss_fid"],
        "loss_patch_fid": fid_out["loss_patch_fid"],
        "loss_dense_fid": fid_out["loss_dense_fid"],
        "loss_dense_bce": fid_out["loss_dense_bce"],
        "loss_dense_kl": fid_out["loss_dense_kl"],
        "loss_topk": fid_out["loss_topk"],
        "loss_delta": fid_out["loss_delta"],
        "loss_concept_dense": loss_concept_dense,
        "loss_concept_kl": loss_concept_kl,
        "loss_align": loss_align,
        "loss_sparse": loss_sparse,
        "loss_div": loss_div,
        "loss_gate": loss_gate,
        "delta_target": fid_out["delta_target"],
        "target_patch_grid": fid_out["target_patch_grid"],
        "source_mixture_grid": fid_out["source_mixture_grid"],
    }
