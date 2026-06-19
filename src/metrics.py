"""
Evaluation metrics for video saliency prediction (last-frame GT).

CC and SIM use continuous saliency density maps.
NSS, AUC, and sAUC use binary fixation maps.
Pseudo-fixation fallback from density maps is debug-only and must be
explicitly enabled via ``allow_pseudo_fixations=True``.

Pure PyTorch — no sklearn/scipy required.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


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


def resize_pred_to_target(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ViNet/MIT-style resize direction:
    resize prediction map to the target map size.

    Both outputs are [B,1,H,W].
    """
    pred = prepare_prediction_map(pred)
    target = prepare_target_last_map(target)

    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch mismatch: pred B={pred.shape[0]}, target B={target.shape[0]}"
        )

    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(
            pred,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
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


def _vinet_minmax_sum_normalize(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    ViNet-style SIM normalization.

    If the map contains nonzero values:
      1. min-max normalize to [0,1]
      2. normalize so the map sums to 1

    If the map is all zeros, keep it all zeros.
    """
    B = x.shape[0]
    flat = x.reshape(B, -1)
    out = torch.zeros_like(flat)

    for i in range(B):
        m = flat[i]
        if torch.any(m != 0):
            mn = m.min()
            mx = m.max()
            m = (m - mn) / (mx - mn + eps)
            s = m.sum()
            if s > eps:
                m = m / s
            out[i] = m
        else:
            out[i] = m

    return out.view_as(x)


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
    ViNet-style CC.

    Uses continuous saliency density maps.
    Prediction is resized to the target size.
    Both maps are z-scored per sample, then correlation is computed.
    """
    pred, target = resize_pred_to_target(pred, target_density)

    B = pred.shape[0]
    pred_f = pred.reshape(B, -1)
    target_f = target.reshape(B, -1)

    pred_mean = pred_f.mean(dim=1, keepdim=True)
    target_mean = target_f.mean(dim=1, keepdim=True)

    pred_std = pred_f.std(dim=1, unbiased=True, keepdim=True).clamp(min=eps)
    target_std = target_f.std(dim=1, unbiased=True, keepdim=True).clamp(min=eps)

    pred_z = (pred_f - pred_mean) / pred_std
    target_z = (target_f - target_mean) / target_std

    # Equivalent to MATLAB corr2 after z-scoring with sample std.
    n = pred_f.shape[1]
    cc = (pred_z * target_z).sum(dim=1) / max(n - 1, 1)

    return cc.mean()


def sim_score(
    pred: torch.Tensor,
    target_density: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    ViNet-style SIM / similarity.

    Uses continuous saliency density maps.
    Prediction is resized to the target size.
    Both maps are min-max normalized, sum-normalized, then histogram intersection is computed.
    """
    pred, target = resize_pred_to_target(pred, target_density)

    pred_n = _vinet_minmax_sum_normalize(pred, eps)
    target_n = _vinet_minmax_sum_normalize(target, eps)

    sim = torch.minimum(pred_n, target_n).reshape(pred.shape[0], -1).sum(dim=1)
    return sim.mean()


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
    eps: float = 1e-8,
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
    pred_std = pred_f.std(dim=1, unbiased=False, keepdim=True).clamp(min=eps)
    pred_z = (pred_f - pred_mean) / pred_std

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
        f_i = fix[i, 0].reshape(-1)
        pos = p[f_i]

        if B > 1:
            neg_parts = []
            for j in range(B):
                if j == i:
                    continue
                fj = fix[j, 0].reshape(-1)
                if fj.any():
                    neg_parts.append(p[fj])
            neg = torch.cat(neg_parts) if neg_parts else p[~f_i]
        else:
            neg = p[~f_i]

        aucs.append(_rank_auc(pos, neg))
    return torch.stack(aucs).mean()


def compute_saliency_metrics(
    pred: torch.Tensor,
    target_density: torch.Tensor,
    fixation_target: Optional[torch.Tensor] = None,
    fixation_threshold: float = 0.5,
    top_percent: Optional[float] = None,
    allow_pseudo_fixations: bool = False,
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

        out["AUC"] = auc_judd_score(
            pred,
            fix_map,
            fixation_threshold=0.5,
            top_percent=None,
        )
        out["sAUC"] = sauc_score(
            pred,
            fix_map,
            fixation_threshold=0.5,
            top_percent=None,
        )
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

    out["AUC"] = auc_judd_score(
        pred,
        pseudo_fix,
        fixation_threshold=0.5,
        top_percent=None,
    )
    out["sAUC"] = sauc_score(
        pred,
        pseudo_fix,
        fixation_threshold=0.5,
        top_percent=None,
    )
    out["NSS"] = nss_score(pred, pseudo_fix)
    return out


class MetricAverager:
    """Running average of saliency metrics over evaluation batches."""

    METRIC_KEYS = ("CC", "SIM", "AUC", "sAUC", "NSS")

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
