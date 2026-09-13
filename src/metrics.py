"""
Evaluation metrics for video saliency prediction (last-frame GT).

CC and SIM use continuous saliency density maps.
NSS, AUC, and sAUC use binary fixation maps.
Pseudo-fixation fallback from density maps is debug-only and must be
explicitly enabled via ``allow_pseudo_fixations=True``.

OpenCV is used for TMFI-Net-compatible prediction resizing and post-blur.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_TMFI_BLUR_KERNEL_SIZE = 11


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}")
    x = x.float()
    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    return x


def prepare_prediction_map(pred: torch.Tensor) -> torch.Tensor:
    """Return prediction as [B, 1, H, W]."""
    if not isinstance(pred, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(pred)}")

    pred = _to_float_tensor(pred)
    if pred.dim() == 3:
        return pred.unsqueeze(1)
    if pred.dim() == 4 and pred.shape[1] == 1:
        return pred
    raise ValueError(
        f"pred must be [B,H,W] or [B,1,H,W], got shape {tuple(pred.shape)}"
    )


def prepare_target_last_map(target: torch.Tensor) -> torch.Tensor:
    """Return last-frame ground truth as [B, 1, H, W]."""
    if not isinstance(target, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(target)}")

    x = _to_float_tensor(target)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        if x.shape[1] == 1:
            return x
        # [B, T, H, W]
        return x[:, -1].unsqueeze(1)
    if x.dim() == 5:
        if x.shape[1] == 1:
            # [B, 1, T, H, W]
            return x[:, :, -1, :, :]
        if x.shape[2] == 1:
            # [B, T, 1, H, W]
            return x[:, -1, 0, :, :].unsqueeze(1)
        raise ValueError(
            f"Unsupported 5D target shape {tuple(x.shape)}; "
            "expected [B,1,T,H,W] or [B,T,1,H,W]"
        )
    raise ValueError(
        f"target must be 3D–5D, got shape {tuple(target.shape)}"
    )


def resize_to_match(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Both [B,1,H,W]; resize target spatially to pred if needed."""
    if pred.dim() != 4 or pred.shape[1] != 1:
        raise ValueError(f"pred must be [B,1,H,W], got {tuple(pred.shape)}")
    if target.dim() != 4 or target.shape[1] != 1:
        raise ValueError(f"target must be [B,1,H,W], got {tuple(target.shape)}")
    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch mismatch: pred B={pred.shape[0]}, target B={target.shape[0]}"
        )
    if pred.shape[-2:] != target.shape[-2:]:
        target = F.interpolate(
            target,
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return pred, target


def _tmfi_gaussian_blur(smap: np.ndarray) -> np.ndarray:
    """Match TMFI-Net ``utils1.blur``: 11x11 Gaussian, sigma derived from kernel."""
    k_size = _TMFI_BLUR_KERNEL_SIZE
    return cv2.GaussianBlur(smap, (k_size, k_size), 0)


def _tmfi_resize_and_blur_pred(
    pred_hw: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """
    Match TMFI-Net validation/test preprocessing on predictions.

    ``cv2.resize`` uses default ``INTER_LINEAR`` bilinear interpolation, then
    applies the same Gaussian blur as ``TMFI-Net/utils1.blur``.
    """
    if pred_hw.shape != (target_h, target_w):
        pred_hw = cv2.resize(pred_hw, (target_w, target_h))
    return _tmfi_gaussian_blur(pred_hw)


def resize_pred_to_target(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TMFI-Net-style resize direction:
    resize prediction map to the target map size with ``cv2.resize``, then blur.

    Both outputs are [B,1,H,W]. The target map is left unchanged.
    """
    pred = prepare_prediction_map(pred)
    target = prepare_target_last_map(target)

    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch mismatch: pred B={pred.shape[0]}, target B={target.shape[0]}"
        )

    target_h, target_w = int(target.shape[-2]), int(target.shape[-1])
    device = pred.device
    dtype = pred.dtype

    resized_preds = []
    for batch_idx in range(pred.shape[0]):
        pred_np = pred[batch_idx, 0].detach().cpu().numpy()
        pred_np = _tmfi_resize_and_blur_pred(pred_np, target_h, target_w)
        resized_preds.append(torch.from_numpy(pred_np))

    pred = torch.stack(resized_preds, dim=0).unsqueeze(1).to(
        device=device,
        dtype=dtype,
    )
    return pred, target


def normalize_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max to [0, 1]; x is [B, 1, H, W]."""
    B = x.shape[0]
    flat = x.reshape(B, -1)
    xmin = flat.min(dim=1, keepdim=True)[0]
    xmax = flat.max(dim=1, keepdim=True)[0]
    flat = (flat - xmin) / (xmax - xmin + eps)
    return flat.view_as(x)


def normalize_sum(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample nonnegative map normalized to sum to 1."""
    x = torch.clamp(x, min=0.0)
    B = x.shape[0]
    flat = x.reshape(B, -1)
    s = flat.sum(dim=1, keepdim=True).clamp(min=eps)
    return (flat / s).view_as(x)


def _to_float_tensor_no_autoscale(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}")
    return x.float()


def _prepare_tmfi_gt_density_bhw(x: torch.Tensor) -> torch.Tensor:
    """
    TMFI-Net dataloader-equivalent density map layout ``[B, H, W]``.

    Matches ``DHF1KDataset._load_sample``: grayscale float, divided by 255 only
    when ``max > 1.0``.
    """
    x = _to_float_tensor_no_autoscale(x)
    if x.dim() == 3:
        pass
    elif x.dim() == 4:
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            x = x[:, -1]
    elif x.dim() == 5:
        if x.shape[1] == 1:
            x = x[:, 0, -1]
        elif x.shape[2] == 1:
            x = x[:, -1, 0]
        else:
            raise ValueError(
                f"Unsupported 5D target shape {tuple(x.shape)}; "
                "expected [B,1,T,H,W] or [B,T,1,H,W]"
            )
    else:
        raise ValueError(f"target must be 3D–5D, got shape {tuple(x.shape)}")

    if x.numel() > 0 and float(x.max()) > 1.0:
        x = x / 255.0
    return x


def _prepare_tmfi_pred_bhw(x: torch.Tensor) -> torch.Tensor:
    """Prediction layout ``[B, H, W]`` for TMFI-Net metrics (float, no /255)."""
    x = _to_float_tensor_no_autoscale(x)
    if x.dim() == 3:
        return x
    if x.dim() == 4 and x.shape[1] == 1:
        return x[:, 0]
    raise ValueError(
        f"pred must be [B,H,W] or [B,1,H,W], got shape {tuple(x.shape)}"
    )


def _tmfi_normalize_map(s_map: torch.Tensor) -> torch.Tensor:
    """Exact port of ``TMFI-Net/loss.py::normalize_map`` for ``[B, H, W]``."""
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(
        batch_size, 1, 1
    ).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(
        batch_size, 1, 1
    ).expand(batch_size, w, h)

    return (s_map - min_s_map) / (max_s_map - min_s_map * 1.0)


def _tmfi_similarity(s_map: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Exact port of ``TMFI-Net/loss.py::similarity`` for ``[B, H, W]`` maps."""
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    s_map = _tmfi_normalize_map(s_map)
    gt = _tmfi_normalize_map(gt)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = s_map / (expand_s_map * 1.0)
    gt = gt / (expand_gt * 1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    return torch.mean(torch.sum(torch.min(s_map, gt), 1))


def _tmfi_resize_blur_pred_bhw(
    pred_bhw: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize/blur prediction to ``[B, H, W]`` using TMFI validate/test preprocessing."""
    device = pred_bhw.device
    dtype = pred_bhw.dtype
    resized_preds = []
    for batch_idx in range(pred_bhw.shape[0]):
        pred_np = pred_bhw[batch_idx].detach().cpu().numpy()
        pred_np = _tmfi_resize_and_blur_pred(pred_np, target_h, target_w)
        resized_preds.append(torch.from_numpy(pred_np))
    return torch.stack(resized_preds, dim=0).to(device=device, dtype=dtype)


def _looks_binary(x: torch.Tensor, eps: float = 1e-6) -> bool:
    flat = x.reshape(x.shape[0], -1)
    mn = flat.min(dim=1)[0]
    mx = flat.max(dim=1)[0]
    # All samples roughly in [0,1] with values near 0 or 1 only
    in_range = (mn >= -eps) & (mx <= 1.0 + eps)
    rounded = torch.round(flat)
    close = (flat - rounded).abs().max(dim=1)[0] < 0.1
    return bool(in_range.all() and close.all())


def make_fixation_binary(
    target: torch.Tensor,
    threshold: float = 0.5,
    top_percent: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Binary fixation map [B, 1, H, W] (bool).

    Ensures at least one fixation per sample.
    """
    if target.dim() != 4 or target.shape[1] != 1:
        raise ValueError(f"target must be [B,1,H,W], got {tuple(target.shape)}")

    B, _, H, W = target.shape
    out = torch.zeros(B, 1, H, W, dtype=torch.bool, device=target.device)

    for i in range(B):
        t = target[i, 0]
        if top_percent is not None:
            n_pixels = H * W
            k = max(1, int(torch.ceil(torch.tensor(top_percent * n_pixels)).item()))
            flat = t.reshape(-1)
            _, idx = torch.topk(flat, k=k, largest=True)
            fix = torch.zeros_like(flat, dtype=torch.bool)
            fix[idx] = True
            fix = fix.view(H, W)
        elif _looks_binary(target[i : i + 1]):
            fix = t > 0
        else:
            t_norm = (t - t.min()) / (t.max() - t.min() + eps)
            fix = t_norm >= threshold

        if not fix.any():
            flat = t.reshape(-1)
            fix = torch.zeros_like(flat, dtype=torch.bool)
            fix[flat.argmax()] = True
            fix = fix.view(H, W)

        out[i, 0] = fix

    return out


def _zscore_per_sample(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    B = x.shape[0]
    flat = x.reshape(B, -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, unbiased=False, keepdim=True).clamp(min=eps)
    return ((flat - mean) / std).view_as(x)


def cc_score(
    pred: torch.Tensor,
    target_density: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    TMFI-Net-style CC.

    Uses continuous saliency density maps.
    Prediction is resized/blurred to the target size via ``resize_pred_to_target``.
    Both maps are z-scored per sample with population std, then correlation is computed.
    """
    pred, target = resize_pred_to_target(pred, target_density)

    B = pred.shape[0]
    pred_f = pred.reshape(B, -1)
    target_f = target.reshape(B, -1)

    pred_mean = pred_f.mean(dim=1, keepdim=True)
    target_mean = target_f.mean(dim=1, keepdim=True)

    # Match TMFI-Net ``loss.cc``: population std (unbiased=False), no epsilon.
    del eps
    pred_std = pred_f.std(dim=1, unbiased=False, keepdim=True)
    target_std = target_f.std(dim=1, unbiased=False, keepdim=True)

    pred_z = (pred_f - pred_mean) / pred_std
    target_z = (target_f - target_mean) / target_std

    ab = (pred_z * target_z).sum(dim=1)
    aa = (pred_z * pred_z).sum(dim=1)
    bb = (target_z * target_z).sum(dim=1)
    cc = ab / torch.sqrt(aa * bb)

    return cc.mean()


def sim_score(
    pred: torch.Tensor,
    target_density: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    TMFI-Net ``loss.similarity`` (MIT/ViNet histogram intersection).

    Prediction is resized/blurred to the GT size with OpenCV, then both maps are
    min-max normalized, sum-normalized, and compared with histogram intersection.
    Ground truth uses TMFI dataloader scaling: ``/255`` only when ``max > 1``.
    """
    del eps

    gt = _prepare_tmfi_gt_density_bhw(target_density)
    pred_bhw = _prepare_tmfi_pred_bhw(pred)

    if pred_bhw.shape[0] != gt.shape[0]:
        raise ValueError(
            f"Batch mismatch: pred B={pred_bhw.shape[0]}, target B={gt.shape[0]}"
        )

    target_h, target_w = int(gt.shape[-2]), int(gt.shape[-1])
    pred_bhw = _tmfi_resize_blur_pred_bhw(pred_bhw, target_h, target_w)
    return _tmfi_similarity(pred_bhw, gt)


def _is_binary_fixation_map(x: torch.Tensor, eps: float = 1e-6) -> bool:
    """
    Check whether a map is binary-like: values are close to 0 or 1.
    """
    x = _to_float_tensor(x)
    flat = x.reshape(x.shape[0], -1)
    rounded = torch.round(flat)
    close_to_binary = (flat - rounded).abs().max(dim=1)[0] <= eps
    in_range = (flat.min(dim=1)[0] >= -eps) & (flat.max(dim=1)[0] <= 1.0 + eps)
    return bool((close_to_binary & in_range).all())


def validate_fixation_map(fixation: torch.Tensor, name: str = "fixation") -> None:
    if fixation.dim() != 4 or fixation.shape[1] != 1:
        raise ValueError(f"{name} must be [B,1,H,W], got {tuple(fixation.shape)}")
    if fixation.numel() > 0:
        mn = float(fixation.min().detach().cpu())
        mx = float(fixation.max().detach().cpu())
        if mn < 0 or mx > 1:
            raise ValueError(f"{name} values must be in [0,1], got min={mn}, max={mx}")


def nss_score(
    pred: torch.Tensor,
    fixation_map: torch.Tensor,
    eps: float = 2.2204e-16,
) -> torch.Tensor:
    """
    True NSS.

    Prediction is z-scored per sample.
    NSS is the mean z-scored prediction value at binary fixation locations.

    Args:
        pred: predicted saliency map [B,H,W] or [B,1,H,W]
        fixation_map: binary fixation map [B,H,W], [B,1,H,W], [B,T,H,W],
                      [B,1,T,H,W], or [B,T,1,H,W].
                      Only the last frame is used for temporal inputs.

    Returns:
        scalar mean NSS over batch.
    """
    pred, fixation = resize_pred_to_target(pred, fixation_map)

    # Force binary fixation map. Any positive value is treated as a fixation.
    fixation = (fixation > 0).float()

    B = pred.shape[0]
    pred_f = pred.reshape(B, -1)
    fix_f = fixation.reshape(B, -1) > 0

    pred_mean = pred_f.mean(dim=1, keepdim=True)
    # Match the baseline: sample standard deviation and epsilon added to the
    # denominator after standard-deviation calculation.
    pred_std = pred_f.std(dim=1, unbiased=True, keepdim=True)
    pred_z = (pred_f - pred_mean) / (pred_std + eps)

    scores = []
    for i in range(B):
        if fix_f[i].any():
            scores.append(pred_z[i][fix_f[i]].mean())
        else:
            # No fixation pixels: return 0 for this sample instead of crashing.
            scores.append(pred_z.new_zeros(()))

    return torch.stack(scores).mean()


def _evenly_subsample(x: torch.Tensor, max_count: int) -> torch.Tensor:
    """Deterministic subsample (evenly spaced indices)."""
    n = x.numel()
    if n <= max_count:
        return x.reshape(-1)
    idx = torch.linspace(0, n - 1, steps=max_count, device=x.device).long()
    return x.reshape(-1)[idx]


def _rank_auc(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    max_count: int = 4096,
) -> torch.Tensor:
    """
    Deterministic rank-based AUC with tie handling.

    AUC = probability that a positive score is greater than a negative score.
    Ranks are computed in ascending order so higher scores receive larger ranks.

    AUC = (sum_pos_ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg), ranks 1-indexed.
    """
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return pos_scores.new_zeros(())

    pos = _evenly_subsample(pos_scores, max_count)
    neg = _evenly_subsample(neg_scores, max_count)
    n_pos = pos.numel()
    n_neg = neg.numel()

    all_scores = torch.cat([pos, neg])
    is_pos = torch.zeros(all_scores.numel(), dtype=torch.bool, device=all_scores.device)
    is_pos[:n_pos] = True

    order = torch.argsort(all_scores, descending=False)
    sorted_scores = all_scores[order]
    ranks = torch.empty_like(all_scores, dtype=torch.float32)

    rank = 1
    i = 0
    n = all_scores.numel()
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        rank += j - i + 1
        i = j + 1

    sum_pos_ranks = ranks[is_pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc.clamp(0.0, 1.0)


def auc_judd_score(
    pred: torch.Tensor,
    fixation_map: torch.Tensor,
    fixation_threshold: float = 0.5,
    top_percent: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = prepare_prediction_map(pred)
    fixation_map = prepare_target_last_map(fixation_map)
    pred, fixation_map = resize_to_match(pred, fixation_map)
    pred = normalize_minmax(pred, eps)

    fix = make_fixation_binary(
        fixation_map, fixation_threshold, top_percent, eps
    )

    B = pred.shape[0]
    aucs = []
    for i in range(B):
        p = pred[i, 0].reshape(-1)
        f = fix[i, 0].reshape(-1)
        pos = p[f]
        neg = p[~f]
        aucs.append(_rank_auc(pos, neg))
    return torch.stack(aucs).mean()


def sauc_score(
    pred: torch.Tensor,
    fixation_map: torch.Tensor,
    fixation_threshold: float = 0.5,
    top_percent: Optional[float] = None,
    eps: float = 1e-8,
    other_map: Optional[torch.Tensor] = None,
    splits: int = 100,
    stepsize: float = 0.1,
) -> torch.Tensor:
    """
    Baseline-compatible shuffled AUC.

    This intentionally mirrors the supplied baseline ``auc_shuff`` protocol,
    including its nine fixed thresholds, use of every location in
    ``other_map`` on every split, TPR-first point sorting, and its original
    flattened-coordinate encode/decode convention.  It is provided for direct
    comparison with that baseline rather than as a canonical sAUC reference.

    ``other_map`` should contain the shuffled-fixation map(s).  It may have one
    map per prediction or a single map shared by the batch.  For backward
    compatibility, if it is omitted and B > 1, each sample uses the union of
    the other samples' fixation maps in the current batch.  With B == 1 and no
    ``other_map``, the score is NaN because the baseline calculation requires
    a shuffled-fixation map.

    ``stepsize`` is retained to match the baseline interface; the supplied
    baseline hard-codes thresholds 0.1 through 0.9 and does not use it.
    """
    del stepsize

    if splits <= 0:
        raise ValueError(f"splits must be positive, got {splits}")

    # The baseline resizes the saliency prediction to the fixation-map size.
    pred, fixation_map = resize_pred_to_target(pred, fixation_map)
    pred = normalize_minmax(pred, eps)
    fix = make_fixation_binary(
        fixation_map, fixation_threshold, top_percent, eps
    )

    prepared_other: Optional[torch.Tensor] = None
    if other_map is not None:
        prepared_other = prepare_target_last_map(other_map)
        if prepared_other.shape[0] not in (1, pred.shape[0]):
            raise ValueError(
                "other_map batch size must be 1 or match pred; "
                f"got {prepared_other.shape[0]} and {pred.shape[0]}"
            )
        if prepared_other.shape[-2:] != fix.shape[-2:]:
            # Match the baseline expectation that gt and other_map share the
            # same pixel grid. Nearest-neighbor preserves binary locations.
            prepared_other = F.interpolate(
                prepared_other,
                size=fix.shape[-2:],
                mode="nearest",
            )
        prepared_other = prepared_other == 1

    B = pred.shape[0]
    thresholds = pred.new_tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    sample_aucs = []

    for sample_idx in range(B):
        saliency = pred[sample_idx, 0]
        gt = fix[sample_idx, 0]
        num_fixations = int(gt.sum().item())

        if prepared_other is not None:
            other = prepared_other[
                0 if prepared_other.shape[0] == 1 else sample_idx, 0
            ]
        elif B > 1:
            keep = torch.arange(B, device=fix.device) != sample_idx
            other = fix[keep, 0].any(dim=0)
        else:
            sample_aucs.append(saliency.new_tensor(float("nan")))
            continue

        coords = torch.nonzero(other == 1, as_tuple=False)
        ind = coords.shape[0]
        if num_fixations == 0 or ind == 0:
            sample_aucs.append(saliency.new_tensor(float("nan")))
            continue

        # Reproduce the baseline's original flattened-coordinate convention:
        #   encoded = row * height + column
        #   sampled = s_map[encoded % height - 1, encoded // height]
        # This is intentionally not conventional row-major indexing.
        height, width = saliency.shape
        encoded = coords[:, 0] * other.shape[0] + coords[:, 1]
        sampled_rows = encoded.remainder(height) - 1
        sampled_cols = torch.div(encoded, height, rounding_mode="floor")
        if (sampled_cols < 0).any() or (sampled_cols >= width).any():
            raise IndexError(
                "The baseline auc_shuff coordinate convention produced an "
                "out-of-range column. Exact compatibility requires square, "
                "same-resolution saliency and shuffled-fixation maps."
            )

        split_aucs = []
        for _ in range(splits):
            # The baseline permutes all candidates but then uses all of them.
            # Retaining the permutation reproduces that behavior exactly.
            permutation = torch.randperm(ind, device=saliency.device)
            random_saliency = saliency[
                sampled_rows[permutation], sampled_cols[permutation]
            ]

            area = [(saliency.new_tensor(0.0), saliency.new_tensor(0.0))]
            for threshold in thresholds:
                thresholded = (saliency >= threshold).to(saliency.dtype)
                gt_numeric = gt.to(saliency.dtype)
                num_overlap = ((thresholded + gt_numeric) == 2).sum()
                tp = num_overlap.to(saliency.dtype) / float(num_fixations)
                fp = (random_saliency > threshold).sum().to(saliency.dtype)
                fp = fp / float(num_fixations)

                # Match round(tp, 4) and round(fp, 4) in the baseline.
                tp = torch.round(tp * 10000.0) / 10000.0
                fp = torch.round(fp * 10000.0) / 10000.0
                area.append((tp, fp))

            area.append((saliency.new_tensor(1.0), saliency.new_tensor(1.0)))
            area.sort(key=lambda point: float(point[0].detach().cpu()))
            tp_values = torch.stack([point[0] for point in area])
            fp_values = torch.stack([point[1] for point in area])
            split_aucs.append(torch.trapz(tp_values, fp_values))

        sample_aucs.append(torch.stack(split_aucs).mean())

    valid_aucs = [score for score in sample_aucs if not torch.isnan(score)]
    if not valid_aucs:
        return pred.new_tensor(float("nan"))
    return torch.stack(valid_aucs).mean()


def compute_saliency_metrics(
    pred: torch.Tensor,
    target_density: torch.Tensor,
    fixation_target: Optional[torch.Tensor] = None,
    fixation_threshold: float = 0.5,
    top_percent: Optional[float] = None,
    allow_pseudo_fixations: bool = False,
    sauc_other_map: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute saliency metrics.

    CC and SIM use continuous saliency density maps.

    NSS, AUC, and sAUC should use true binary fixation maps.
    If fixation_target is missing, these metrics are returned as NaN unless
    allow_pseudo_fixations=True.
    """
    out = {
        "CC": cc_score(pred, target_density),
        "SIM": sim_score(pred, target_density),
    }

    if fixation_target is not None:
        fix_map = prepare_target_last_map(fixation_target)
        fix_map = (fix_map > 0).float()

        # out["AUC"] = auc_judd_score(
        #     pred,
        #     fix_map,
        #     fixation_threshold=0.5,
        #     top_percent=None,
        # )
        # out["sAUC"] = sauc_score(
        #     pred,
        #     fix_map,
        #     fixation_threshold=0.5,
        #     top_percent=None,
        #     other_map=sauc_other_map,
        # )
        out["NSS"] = nss_score(pred, fix_map)
        return out

    if not allow_pseudo_fixations:
        nan = prepare_prediction_map(pred).new_tensor(float("nan"))
        out["AUC"] = nan
        out["sAUC"] = nan
        out["NSS"] = nan
        return out

    # Debug-only pseudo-fixation fallback.
    # WARNING: these are not true fixation-based metrics.
    fix_map = prepare_target_last_map(target_density)
    pseudo_fix = make_fixation_binary(
        fix_map,
        threshold=fixation_threshold,
        top_percent=top_percent,
    ).float()

    # out["AUC"] = auc_judd_score(
    #     pred,
    #     pseudo_fix,
    #     fixation_threshold=0.5,
    #     top_percent=None,
    # )
    # out["sAUC"] = sauc_score(
    #     pred,
    #     pseudo_fix,
    #     fixation_threshold=0.5,
    #     top_percent=None,
    #     other_map=sauc_other_map,
    # )
    out["NSS"] = nss_score(pred, pseudo_fix)
    return out


class MetricAverager:
    """Running average of saliency metrics over evaluation batches."""

    # METRIC_KEYS = ("CC", "SIM", "AUC", "sAUC", "NSS")
    METRIC_KEYS = ("CC", "SIM", "NSS")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.totals: Dict[str, float] = {k: 0.0 for k in self.METRIC_KEYS}
        self.counts: Dict[str, int] = {k: 0 for k in self.METRIC_KEYS}

    def update(self, metric_dict: Dict[str, torch.Tensor], batch_size: int = 1) -> None:
        for key in self.METRIC_KEYS:
            if key not in metric_dict:
                raise ValueError(f"metric_dict missing key '{key}'")
            value = float(metric_dict[key].detach().cpu())
            if value == value:  # not NaN
                self.totals[key] += value * batch_size
                self.counts[key] += batch_size

    def mean(self) -> Dict[str, float]:
        return {
            k: self.totals[k] / self.counts[k] if self.counts[k] > 0 else float("nan")
            for k in self.METRIC_KEYS
        }
