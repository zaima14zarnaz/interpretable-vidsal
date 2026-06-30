"""
Simplified concept-conditioned saliency decoder.

Each backbone stage contributes patch-grid logits decoded from visual and temporal
concept representations. Multi-scale fusion happens on the patch grid before a
single upsample to RGB resolution.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long).reshape(-1)
    return torch.tensor(value, device=device, dtype=torch.long).reshape(-1)


def _to_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.tensor(value, device=device, dtype=torch.float32).reshape(-1)


def _prepare_last_frame(last_rgb_frame: torch.Tensor) -> torch.Tensor:
    """Normalize last-frame RGB to [B, 3, H, W] in [0, 1]."""
    x = last_rgb_frame
    if x.dim() == 5:
        if x.shape[1] == 3:
            x = x[:, :, -1, :, :]
        elif x.shape[2] == 3:
            x = x[:, -1, :, :, :]
        elif x.shape[-1] == 3:
            x = x[:, -1, :, :, :].permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "5D last_rgb_frame must have channel dim 1, 2, or last (-1)"
            )
    elif x.dim() == 4:
        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        elif x.shape[1] != 3:
            raise ValueError("4D last_rgb_frame must be [B,3,H,W] or [B,H,W,3]")
    else:
        raise ValueError(
            f"last_rgb_frame must be 4D or 5D, got shape {tuple(x.shape)}"
        )

    x = x.float()
    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    return x


def _feature_shape_from_metadata(
    metadata: Dict[str, Any],
) -> Tuple[int, int, int, int, int]:
    feature_shape = metadata.get("feature_shape")
    if feature_shape is None:
        raise ValueError("metadata must contain 'feature_shape'")

    if isinstance(feature_shape, dict):
        return (
            int(feature_shape["B"]),
            int(feature_shape["C"]),
            int(feature_shape["T"]),
            int(feature_shape["H"]),
            int(feature_shape["W"]),
        )

    shape = tuple(int(v) for v in feature_shape)
    if len(shape) != 5:
        raise ValueError(
            f"feature_shape must have 5 entries (B,C,T,H,W), got {shape}"
        )
    return shape[0], shape[1], shape[2], shape[3], shape[4]


def _get_feature_shape(
    concept_out: Dict[str, Any], metadata: Dict[str, Any]
) -> Tuple[int, int, int, int, int]:
    device = concept_out["concept_representation"].device
    feature_shape = metadata.get("feature_shape")

    if feature_shape is not None:
        if isinstance(feature_shape, dict):
            B = int(feature_shape["B"])
            T = int(feature_shape["T"])
            H = int(feature_shape["H"])
            W = int(feature_shape["W"])
        else:
            shape = tuple(int(v) for v in feature_shape)
            if len(shape) != 5:
                raise ValueError(
                    f"feature_shape must have 5 entries (B,C,T,H,W), got {shape}"
                )
            B, _, T, H, W = shape
    else:
        batch_idx = _to_long_tensor(metadata["batch_idx"], device)
        time_idx = _to_long_tensor(metadata["time_idx"], device)
        target_idx = _to_long_tensor(metadata["target_idx"], device)

        B = int(batch_idx.max().item()) + 1
        T = int(time_idx.max().item()) + 2
        N = int(target_idx.max().item()) + 1
        root = int(math.isqrt(N))
        if root * root != N:
            raise ValueError(
                f"Cannot infer square patch grid: N={N} is not a perfect square"
            )
        H = W = root

    if T < 2:
        raise ValueError(f"Feature time dimension T must be >= 2, got T={T}")

    N = H * W
    return B, T, H, W, N


def _select_last_transition(
    concept_out: Dict[str, Any],
    metadata: Dict[str, Any],
    B: int,
    T: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    device = concept_out["concept_representation"].device
    time_idx = _to_long_tensor(metadata["time_idx"], device)
    last_t = T - 2
    mask = time_idx == last_t

    if not mask.any():
        raise ValueError(
            f"No trajectories found for last transition time_idx={last_t} (T={T})"
        )

    concept_repr = concept_out["concept_representation"]
    if concept_repr.shape[0] != time_idx.shape[0]:
        raise ValueError(
            "concept_representation length does not match metadata trajectory count"
        )

    selected_meta: Dict[str, torch.Tensor] = {}
    required_keys = (
        "batch_idx",
        "time_idx",
        "source_idx",
        "target_idx",
        "source_coords",
        "target_coords",
        "alpha",
    )
    scalar_keys = {
        "batch_idx",
        "time_idx",
        "source_idx",
        "target_idx",
        "alpha",
        "affinity_logit",
    }
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"metadata missing required key '{key}'")
        tensor = metadata[key]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, device=device)
        else:
            tensor = tensor.to(device)
        if tensor.shape[0] != time_idx.shape[0]:
            raise ValueError(
                f"metadata['{key}'] length {tensor.shape[0]} != "
                f"trajectory count {time_idx.shape[0]}"
            )
        if tensor.dim() == 1 or key in scalar_keys:
            selected_meta[key] = tensor.reshape(-1)[mask]
        else:
            selected_meta[key] = tensor[mask]

    if "affinity_logit" in metadata:
        tensor = metadata["affinity_logit"]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, device=device)
        else:
            tensor = tensor.to(device)
        selected_meta["affinity_logit"] = tensor.reshape(-1)[mask]

    if "feature_shape" in metadata:
        selected_meta["feature_shape"] = metadata["feature_shape"]

    return concept_repr[mask], selected_meta, mask


def _incoming_softmax(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
    tau_pi: float,
) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.zeros_like(scores)

    scaled = scores / tau_pi
    group_ids_long = group_ids.long()

    group_max = torch.full(
        (num_groups,),
        float("-inf"),
        device=scores.device,
        dtype=scores.dtype,
    )
    group_max.scatter_reduce_(
        0, group_ids_long, scaled, reduce="amax", include_self=True
    )
    shifted = scaled - group_max[group_ids_long]
    exp_scores = torch.exp(shifted)

    group_sum = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_sum.scatter_add_(0, group_ids_long, exp_scores)
    return exp_scores / group_sum[group_ids_long]


def _aggregate_vector_to_patch_grid(
    values: torch.Tensor,
    pi: torch.Tensor,
    batch_idx: torch.Tensor,
    target_idx: torch.Tensor,
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"values must be [M,D], got {tuple(values.shape)}")

    device = values.device
    dtype = values.dtype
    M, _ = values.shape
    N = H * W

    if pi.shape[0] != M:
        raise ValueError(f"pi length {pi.shape[0]} must match values length {M}")

    flat_idx = batch_idx.long() * N + target_idx.long()
    weighted = (values * pi.unsqueeze(-1)).to(dtype=dtype)

    patch_flat = torch.zeros(B * N, values.shape[1], device=device, dtype=dtype)
    patch_flat.index_add_(0, flat_idx, weighted)
    return patch_flat.view(B, H, W, values.shape[1]).permute(0, 3, 1, 2).contiguous()


def _build_visual_patch_context(
    concept_out: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    B: int,
    T: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    visual_repr = concept_out.get("visual_concept_representation")
    if not torch.is_tensor(visual_repr) or visual_repr.dim() != 2:
        raise ValueError(
            "concept_out must contain visual_concept_representation with shape "
            "[B*T*H*W, concept_dim]."
        )

    visual_metadata = concept_out.get("visual_metadata")
    shape_metadata = (
        visual_metadata
        if isinstance(visual_metadata, dict) and "feature_shape" in visual_metadata
        else metadata
    )
    B_vis, _, T_vis, H_vis, W_vis = _feature_shape_from_metadata(shape_metadata)
    expected = B_vis * T_vis * H_vis * W_vis
    if visual_repr.shape[0] != expected:
        raise ValueError(
            "visual_concept_representation length must equal B*T*H*W, "
            f"expected {expected}, got {visual_repr.shape[0]}"
        )

    visual_context = (
        visual_repr.reshape(B_vis, T_vis, H_vis, W_vis, -1)[:, -1]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    if visual_context.shape[-2:] != (H, W):
        visual_context = F.interpolate(
            visual_context,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
    return visual_context.to(device=device)


class StageConceptSaliencyDecoder(nn.Module):
    """Decode one stage's concept outputs into patch-grid saliency logits."""

    def __init__(
        self,
        concept_dim: int = 256,
        hidden_dim: int = 256,
        tau_pi: float = 0.1,
        use_visual_context: bool = True,
        use_temporal_context: bool = True,
    ) -> None:
        super().__init__()
        if not use_visual_context and not use_temporal_context:
            raise ValueError(
                "At least one of use_visual_context or use_temporal_context must be True."
            )

        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.tau_pi = tau_pi
        self.use_visual_context = use_visual_context
        self.use_temporal_context = use_temporal_context

        in_channels = concept_dim * (
            int(use_visual_context) + int(use_temporal_context)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        concept_out: Dict[str, Any],
        output_size: Tuple[int, int],
        *,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if "metadata" not in concept_out:
            raise ValueError("concept_out must contain 'metadata'")

        metadata = concept_out["metadata"]
        B, T, H, W, N = _get_feature_shape(concept_out, metadata)
        if torch.is_tensor(concept_out.get("concept_representation")):
            device = concept_out["concept_representation"].device
        elif torch.is_tensor(concept_out.get("visual_concept_representation")):
            device = concept_out["visual_concept_representation"].device
        else:
            raise ValueError(
                "concept_out must contain concept_representation and/or "
                "visual_concept_representation tensors."
            )

        context_parts: list[torch.Tensor] = []
        visual_context: Optional[torch.Tensor] = None
        selected_meta: Optional[Dict[str, torch.Tensor]] = None
        incoming_weights: Optional[torch.Tensor] = None
        temporal_patch_context: Optional[torch.Tensor] = None

        if self.use_visual_context:
            visual_context = _build_visual_patch_context(
                concept_out, metadata, B=B, T=T, H=H, W=W, device=device
            )
            context_parts.append(visual_context)

        if self.use_temporal_context:
            if "concept_representation" not in concept_out:
                raise ValueError(
                    "concept_out must contain 'concept_representation' when "
                    "use_temporal_context=True"
                )
            concept_repr, selected_meta, _ = _select_last_transition(
                concept_out, metadata, B, T
            )
            batch_idx = _to_long_tensor(selected_meta["batch_idx"], device)
            target_idx = _to_long_tensor(selected_meta["target_idx"], device)
            if "affinity_logit" in selected_meta:
                incoming_score = _to_float_tensor(
                    selected_meta["affinity_logit"], device
                )
            else:
                incoming_score = _to_float_tensor(selected_meta["alpha"], device)

            group_ids = batch_idx * N + target_idx
            incoming_weights = _incoming_softmax(
                incoming_score, group_ids, B * N, self.tau_pi
            )
            temporal_patch_context = _aggregate_vector_to_patch_grid(
                concept_repr,
                incoming_weights,
                batch_idx,
                target_idx,
                B,
                H,
                W,
            )
            context_parts.append(temporal_patch_context)

        decoder_input = (
            context_parts[0]
            if len(context_parts) == 1
            else torch.cat(context_parts, dim=1)
        )

        patch_logits = self.decoder(decoder_input)
        saliency_logits = patch_logits
        if patch_logits.shape[-2:] != output_size:
            saliency_logits = F.interpolate(
                patch_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        out: Dict[str, torch.Tensor] = {
            "patch_saliency_logits": patch_logits,
            "saliency_logits": saliency_logits,
        }
        if visual_context is not None:
            out["patch_concept_context"] = visual_context
        if return_details:
            if selected_meta is not None:
                out["selected_metadata"] = selected_meta
            if incoming_weights is not None:
                out["incoming_weights"] = incoming_weights
            if temporal_patch_context is not None:
                out["temporal_patch_context"] = temporal_patch_context
        return out


class MultiScaleSaliencyPrediction(nn.Module):
    """
    Fuse per-stage concept decoders on the patch grid, then upsample once to RGB.
    """

    def __init__(
        self,
        stage_channels: Dict[str, int],
        concept_dim: int = 256,
        hidden_dim: int = 256,
        tau_pi: float = 0.5,
        output_activation: str = "sigmoid",
        fusion_hidden_channels: int = 64,
        use_visual_context: bool = True,
        use_temporal_context: bool = True,
    ) -> None:
        super().__init__()

        if output_activation not in ("sigmoid", "none"):
            raise ValueError("output_activation must be 'sigmoid' or 'none'")

        self.stage_names = tuple(stage_channels.keys())
        self.stage_channels = dict(stage_channels)
        self.output_activation = output_activation
        self.concept_dim = concept_dim

        self.stage_decoders = nn.ModuleDict(
            {
                stage: StageConceptSaliencyDecoder(
                    concept_dim=concept_dim,
                    hidden_dim=hidden_dim,
                    tau_pi=tau_pi,
                    use_visual_context=use_visual_context,
                    use_temporal_context=use_temporal_context,
                )
                for stage in self.stage_names
            }
        )

        self.fusion_head = nn.Sequential(
            nn.Conv2d(len(self.stage_names), fusion_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_hidden_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        concept_outs: Dict[str, Dict[str, Any]],
        last_rgb_frame: torch.Tensor,
        video_features_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_details: bool = False,
        last_rgb_prepared: bool = False,
    ) -> Dict[str, Any]:
        _ = video_features_dict  # kept for API compatibility; decoder is concept-only.

        last_rgb = (
            last_rgb_frame
            if last_rgb_prepared
            else _prepare_last_frame(last_rgb_frame)
        )
        output_size = last_rgb.shape[-2:]

        stage_patch_logits = []
        stage_outs: Dict[str, Dict[str, Any]] = {}
        detail_stage = (
            "stage1" if "stage1" in self.stage_names else self.stage_names[0]
        )

        for stage in self.stage_names:
            if stage not in concept_outs:
                raise ValueError(f"Missing concept_out for stage {stage}")

            stage_return_details = return_details and stage == detail_stage
            out_s = self.stage_decoders[stage](
                concept_outs[stage],
                output_size=output_size,
                return_details=stage_return_details,
            )
            stage_outs[stage] = out_s
            patch_logits = out_s["patch_saliency_logits"]
            stage_patch_logits.append(patch_logits)

        target_hw = stage_patch_logits[0].shape[-2:]
        aligned_patch_logits = []
        for logits in stage_patch_logits:
            if logits.shape[-2:] != target_hw:
                logits = F.interpolate(
                    logits,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_patch_logits.append(logits)

        fused_patch_logits = self.fusion_head(
            torch.cat(aligned_patch_logits, dim=1)
        )
        saliency_logits = fused_patch_logits
        if fused_patch_logits.shape[-2:] != output_size:
            saliency_logits = F.interpolate(
                fused_patch_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        if self.output_activation == "sigmoid":
            saliency_map = torch.sigmoid(saliency_logits)
        else:
            saliency_map = saliency_logits

        main_out = stage_outs[detail_stage]
        out: Dict[str, Any] = {
            "saliency_map": saliency_map,
            "saliency_logits": saliency_logits,
            "coarse_saliency_logits": saliency_logits,
            "patch_saliency_logits": fused_patch_logits,
            "temporal_saliency_logits": saliency_logits,
            "temporal_saliency_map": saliency_map,
            "main_stage": detail_stage,
            "patch_concept_context": main_out.get("patch_concept_context"),
        }

        if return_details:
            if "selected_metadata" in main_out:
                out["selected_metadata"] = main_out["selected_metadata"]
            if "incoming_weights" in main_out:
                out["incoming_weights"] = main_out["incoming_weights"]
            if "temporal_patch_context" in main_out:
                out["temporal_patch_context"] = main_out["temporal_patch_context"]
            out["stage_outputs"] = stage_outs

        return out


# Backward-compatible aliases for older imports/tests.
SaliencyPrediction = StageConceptSaliencyDecoder
