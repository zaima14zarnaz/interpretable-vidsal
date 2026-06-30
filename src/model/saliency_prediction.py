"""
Concept-gated multi-scale saliency decoder for the last frame in a video window.

Stages are decoded coarse-to-fine (stage4 -> stage1). Each stage fuses visual/temporal
concept maps with last-frame backbone features through a concept-controlled gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_STAGE_ORDER = ("stage4", "stage3", "stage2", "stage1")


def _to_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device == device and value.dtype == torch.long:
            return value if value.dim() == 1 else value.reshape(-1)
        return value.to(device=device, dtype=torch.long).reshape(-1)
    return torch.tensor(value, device=device, dtype=torch.long).reshape(-1)


def _to_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device == device and value.dtype == torch.float32:
            return value if value.dim() == 1 else value.reshape(-1)
        return value.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.tensor(value, device=device, dtype=torch.float32).reshape(-1)


def _feature_shape_from_metadata(metadata: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
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
        raise ValueError(f"feature_shape must have 5 entries (B,C,T,H,W), got {shape}")
    return shape[0], shape[1], shape[2], shape[3], shape[4]


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
    group_sum.index_add_(0, group_ids_long, exp_scores)
    return exp_scores / group_sum[group_ids_long].clamp(min=1e-8)


def _build_visual_concept_map(
    concept_out: Dict[str, Any],
    target_hw: Tuple[int, int],
) -> Optional[torch.Tensor]:
    visual_repr = concept_out.get("visual_concept_representation")
    visual_metadata = concept_out.get("visual_metadata")
    if not torch.is_tensor(visual_repr) or not isinstance(visual_metadata, dict):
        return None
    if "feature_shape" not in visual_metadata:
        return None

    try:
        B, _, T, H, W = _feature_shape_from_metadata(visual_metadata)
    except (ValueError, KeyError, TypeError):
        return None

    expected = B * T * H * W
    if visual_repr.shape[0] != expected:
        return None

    concept_dim = visual_repr.shape[-1]
    visual_map = (
        visual_repr.reshape(B, T, H, W, concept_dim)[:, -1]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    if visual_map.shape[-2:] != target_hw:
        visual_map = F.interpolate(
            visual_map,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return visual_map


def _build_temporal_concept_map(
    concept_out: Dict[str, Any],
    target_hw: Tuple[int, int],
    tau_pi: float = 0.1,
) -> Optional[torch.Tensor]:
    metadata = concept_out.get("metadata")
    concept_repr = concept_out.get("concept_representation")
    if not isinstance(metadata, dict) or not torch.is_tensor(concept_repr):
        return None
    if concept_repr.dim() != 2:
        return None

    try:
        B, _, T, H, W = _feature_shape_from_metadata(metadata)
    except (ValueError, KeyError, TypeError):
        return None

    device = concept_repr.device
    dtype = concept_repr.dtype
    required = ("time_idx", "batch_idx", "target_idx")
    for key in required:
        if key not in metadata:
            return None

    time_idx = _to_long_tensor(metadata["time_idx"], device)
    last_t = T - 2
    mask = time_idx == last_t
    if not mask.any():
        return None

    if concept_repr.shape[0] != time_idx.shape[0]:
        return None

    selected_repr = concept_repr[mask]
    batch_idx = _to_long_tensor(metadata["batch_idx"], device)[mask]
    target_idx = _to_long_tensor(metadata["target_idx"], device)[mask]

    if "affinity_logit" in metadata:
        scores = _to_float_tensor(metadata["affinity_logit"], device)[mask]
    elif "alpha" in metadata:
        scores = _to_float_tensor(metadata["alpha"], device)[mask]
    else:
        return None

    N = H * W
    group_ids = batch_idx * N + target_idx
    pi = _incoming_softmax(scores, group_ids, B * N, tau_pi)

    concept_dim = selected_repr.shape[-1]
    aggregated = torch.zeros(B * N, concept_dim, device=device, dtype=dtype)
    aggregated.index_add_(0, group_ids, selected_repr * pi.unsqueeze(-1))

    temporal_map = aggregated.view(B, H, W, concept_dim).permute(0, 3, 1, 2).contiguous()
    if temporal_map.shape[-2:] != target_hw:
        temporal_map = F.interpolate(
            temporal_map,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return temporal_map


def _build_stage_concept_map(
    concept_out: Dict[str, Any],
    target_hw: Tuple[int, int],
    temporal_fusion: Optional[nn.Module],
    tau_pi: float = 0.1,
) -> torch.Tensor:
    """
    Build [B, concept_dim, H, W] concept map for one stage.

    Uses visual_concept_representation as the primary dense map and optionally
    fuses a temporally aggregated concept map from trajectory metadata.
    """
    visual_map = _build_visual_concept_map(concept_out, target_hw)
    if visual_map is None:
        raise ValueError(
            "concept_out must provide visual_concept_representation and "
            "visual_metadata['feature_shape'] to build a stage concept map."
        )

    temporal_map = _build_temporal_concept_map(concept_out, target_hw, tau_pi=tau_pi)
    if temporal_map is not None and temporal_fusion is not None:
        return temporal_fusion(torch.cat([visual_map, temporal_map], dim=1))

    return visual_map


class ConceptGatedFusionBlock(nn.Module):
    """
    Fuse last-frame backbone features with a concept map, optionally refining
    a coarser decoder feature map from a previous stage.
    """

    def __init__(
        self,
        feature_channels: int,
        concept_dim: int,
        decoder_channels: int,
        feature_residual_scale: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_residual_scale = feature_residual_scale

        self.feature_proj = nn.Conv2d(feature_channels, decoder_channels, kernel_size=1)
        self.concept_proj = nn.Conv2d(concept_dim, decoder_channels, kernel_size=1)
        self.prev_proj = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=1)
        self.gate_conv = nn.Conv2d(
            decoder_channels * 2,
            decoder_channels,
            kernel_size=3,
            padding=1,
        )
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(
        self,
        features: torch.Tensor,
        concept_map: torch.Tensor,
        prev_decoder: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_proj = self.feature_proj(features)
        concept_proj = self.concept_proj(concept_map)

        if prev_decoder is not None:
            prev_up = F.interpolate(
                prev_decoder,
                size=concept_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            prev_up = self.prev_proj(prev_up)
        else:
            prev_up = torch.zeros_like(concept_proj)

        gate_input = torch.cat([concept_proj, prev_up], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))

        fused = concept_proj + self.feature_residual_scale * gate * feature_proj
        if prev_decoder is not None:
            fused = fused + prev_up

        decoded = self.refine(fused)
        return decoded, gate


class ConceptGatedMultiScaleSaliencyDecoder(nn.Module):
    """
    Coarse-to-fine concept-gated decoder over Video Swin stages.

    Final saliency is produced only from concept maps and last-frame stage features;
    no RGB refinement, subpatch heads, or trajectory scalar aggregation.
    """

    def __init__(
        self,
        stage_channels: Dict[str, int],
        concept_dim: int = 256,
        decoder_channels: int = 128,
        feature_residual_scale: float = 0.25,
        dropout: float = 0.1,
        tau_pi: float = 0.1,
        output_activation: str = "sigmoid",
    ):
        super().__init__()

        if output_activation not in ("sigmoid", "none"):
            raise ValueError("output_activation must be 'sigmoid' or 'none'")

        self.stage_channels = dict(stage_channels)
        self.concept_dim = concept_dim
        self.decoder_channels = decoder_channels
        self.feature_residual_scale = feature_residual_scale
        self.dropout = dropout
        self.tau_pi = tau_pi
        self.output_activation = output_activation

        self.temporal_fusion = nn.Conv2d(concept_dim * 2, concept_dim, kernel_size=1)

        self.fusion_blocks = nn.ModuleDict(
            {
                stage: ConceptGatedFusionBlock(
                    feature_channels=channels,
                    concept_dim=concept_dim,
                    decoder_channels=decoder_channels,
                    feature_residual_scale=feature_residual_scale,
                    dropout=dropout,
                )
                for stage, channels in self.stage_channels.items()
            }
        )

        self.pred_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1),
        )

    def _ordered_stages(
        self,
        concept_outs: Dict[str, Dict[str, Any]],
        features_dict: Dict[str, torch.Tensor],
    ) -> List[str]:
        available = set(concept_outs.keys()) & set(features_dict.keys()) & set(
            self.fusion_blocks.keys()
        )
        return [stage for stage in _STAGE_ORDER if stage in available]

    def forward(
        self,
        concept_outs: Dict[str, Dict[str, Any]],
        features_dict: Dict[str, torch.Tensor],
        output_size: Tuple[int, int],
        return_details: bool = False,
    ) -> Dict[str, Any]:
        stages = self._ordered_stages(concept_outs, features_dict)
        if not stages:
            raise ValueError("No overlapping stages found in concept_outs and features_dict")

        stage_concept_maps: Dict[str, torch.Tensor] = {}
        stage_feature_maps: Dict[str, torch.Tensor] = {}
        stage_gates: Dict[str, torch.Tensor] = {}
        decoded_stage_features: Dict[str, torch.Tensor] = {}

        prev_decoder: Optional[torch.Tensor] = None
        device: Optional[torch.device] = None

        for stage in stages:
            features_5d = features_dict[stage]
            if features_5d.dim() != 5:
                raise ValueError(
                    f"features_dict[{stage}] must be [B,C,T,H,W], "
                    f"got {tuple(features_5d.shape)}"
                )

            feature_map = features_5d[:, :, -1, :, :]
            target_hw = feature_map.shape[-2:]
            if device is None:
                device = feature_map.device

            concept_map = _build_stage_concept_map(
                concept_outs[stage],
                target_hw,
                temporal_fusion=self.temporal_fusion,
                tau_pi=self.tau_pi,
            )

            decoded, gate = self.fusion_blocks[stage](
                feature_map,
                concept_map,
                prev_decoder=prev_decoder,
            )
            prev_decoder = decoded

            if return_details:
                stage_concept_maps[stage] = concept_map.detach()
                stage_feature_maps[stage] = feature_map.detach()
                stage_gates[stage] = gate.detach()
                decoded_stage_features[stage] = decoded.detach()

        patch_logits = self.pred_head(prev_decoder)
        saliency_logits = patch_logits
        if patch_logits.shape[-2:] != output_size:
            saliency_logits = F.interpolate(
                patch_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        if self.output_activation == "sigmoid":
            saliency_map = torch.sigmoid(saliency_logits)
        else:
            saliency_map = saliency_logits

        out: Dict[str, Any] = {
            "saliency_logits": saliency_logits,
            "saliency_map": saliency_map,
            "patch_saliency_logits": patch_logits,
            "coarse_saliency_logits": saliency_logits,
        }

        if return_details:
            out["stage_concept_maps"] = stage_concept_maps
            out["stage_feature_maps"] = stage_feature_maps
            out["stage_gates"] = stage_gates
            out["decoded_stage_features"] = decoded_stage_features

        return out
