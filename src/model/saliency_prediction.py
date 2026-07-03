"""
Concept-gated multi-scale spatiotemporal saliency decoder for the last frame in a window.

Stages are decoded coarse-to-fine (stage4 -> stage1). Each stage fuses full
[B, C, T, H, W] feature and concept volumes, aggregates over time with learned
attention, and upsamples with learned conv-transpose blocks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_STAGE_ORDER = ("stage4", "stage3", "stage2", "stage1")
_SIDE_UPSAMPLE_SCALES: Dict[str, int] = {
    "stage1": 4,
    "stage2": 8,
    "stage3": 16,
    "stage4": 32,
}


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


def _metadata_feature_shape_repr(metadata: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(metadata, dict):
        return None
    return metadata.get("feature_shape")


def _resize_concept_map_to_target(
    concept_map: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Bilinearly resize [B, C, H, W] concept map to decoder feature-map size."""
    if concept_map.shape[-2:] != target_hw:
        return F.interpolate(
            concept_map,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return concept_map


def _assert_concept_map_matches_features(
    *,
    stage: str,
    concept_map: torch.Tensor,
    feature_map: torch.Tensor,
    concept_out: Dict[str, Any],
) -> None:
    if (
        concept_map.shape[0] == feature_map.shape[0]
        and concept_map.shape[-2:] == feature_map.shape[-2:]
    ):
        return

    visual_metadata = concept_out.get("visual_metadata")
    metadata = concept_out.get("metadata")
    raise ValueError(
        f"Concept map / feature map shape mismatch at {stage}: "
        f"feature_map shape={tuple(feature_map.shape)}, "
        f"concept_map shape={tuple(concept_map.shape)}, "
        f"visual_metadata['feature_shape']="
        f"{_metadata_feature_shape_repr(visual_metadata if isinstance(visual_metadata, dict) else None)}, "
        f"metadata['feature_shape']="
        f"{_metadata_feature_shape_repr(metadata if isinstance(metadata, dict) else None)}"
    )


def _build_visual_concept_map(
    concept_out: Dict[str, Any],
    target_hw: Tuple[int, int],
    *,
    stage: str = "unknown",
) -> Optional[torch.Tensor]:
    visual_repr = concept_out.get("visual_concept_representation")
    visual_metadata = concept_out.get("visual_metadata")
    if not torch.is_tensor(visual_repr) or not isinstance(visual_metadata, dict):
        return None
    if "feature_shape" not in visual_metadata:
        return None

    try:
        B, _, T, H, W = _feature_shape_from_metadata(visual_metadata)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid visual_metadata['feature_shape'] at {stage}: "
            f"{_metadata_feature_shape_repr(visual_metadata)}"
        ) from exc

    expected = B * T * H * W
    if visual_repr.shape[0] != expected:
        raise ValueError(
            f"visual_concept_representation length mismatch at {stage}: "
            f"expected {expected} (=B*T*H*W from feature_shape "
            f"{_metadata_feature_shape_repr(visual_metadata)}), "
            f"got {visual_repr.shape[0]}"
        )

    concept_dim = visual_repr.shape[-1]
    visual_map = (
        visual_repr.reshape(B, T, H, W, concept_dim)[:, -1]
        .permute(0, 3, 1, 2)
    )
    return _resize_concept_map_to_target(visual_map, target_hw)


def _build_visual_concept_volume(
    concept_out: Dict[str, Any],
    *,
    stage: str = "unknown",
) -> Optional[torch.Tensor]:
    visual_repr = concept_out.get("visual_concept_representation")
    visual_metadata = concept_out.get("visual_metadata")
    if not torch.is_tensor(visual_repr) or not isinstance(visual_metadata, dict):
        return None
    if "feature_shape" not in visual_metadata:
        return None

    try:
        B, _, T, H, W = _feature_shape_from_metadata(visual_metadata)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid visual_metadata['feature_shape'] at {stage}: "
            f"{_metadata_feature_shape_repr(visual_metadata)}"
        ) from exc

    expected = B * T * H * W
    if visual_repr.shape[0] != expected:
        raise ValueError(
            f"visual_concept_representation length mismatch at {stage}: "
            f"expected {expected} (=B*T*H*W from feature_shape "
            f"{_metadata_feature_shape_repr(visual_metadata)}), "
            f"got {visual_repr.shape[0]}"
        )

    concept_dim = visual_repr.shape[-1]
    return (
        visual_repr.reshape(B, T, H, W, concept_dim)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


def _build_stage_concept_volume(
    concept_out: Dict[str, Any],
    *,
    stage: str = "unknown",
) -> torch.Tensor:
    """
    Build [B, concept_dim, T, H, W] visual concept volume for one stage.
    """
    concept_volume = _build_visual_concept_volume(concept_out, stage=stage)
    if concept_volume is None:
        visual_metadata = concept_out.get("visual_metadata")
        raise ValueError(
            f"concept_out at {stage} must provide visual_concept_representation and "
            f"visual_metadata['feature_shape']; got visual_metadata="
            f"{_metadata_feature_shape_repr(visual_metadata if isinstance(visual_metadata, dict) else None)}"
        )
    return concept_volume


def _assert_concept_volume_matches_features(
    *,
    stage: str,
    concept_volume: torch.Tensor,
    features_5d: torch.Tensor,
    concept_out: Dict[str, Any],
) -> None:
    if concept_volume.dim() != 5:
        raise ValueError(
            f"Concept volume at {stage} must be 5D [B,concept_dim,T,H,W], "
            f"got {tuple(concept_volume.shape)}"
        )
    if features_5d.dim() != 5:
        raise ValueError(
            f"features_dict[{stage}] must be 5D [B,C,T,H,W], "
            f"got {tuple(features_5d.shape)}"
        )

    B_c, concept_dim, T_c, H_c, W_c = concept_volume.shape
    B_f, _, T_f, H_f, W_f = features_5d.shape
    if (
        B_c == B_f
        and T_c == T_f
        and H_c == H_f
        and W_c == W_f
        and concept_dim == concept_volume.shape[1]
    ):
        return

    visual_metadata = concept_out.get("visual_metadata")
    metadata = concept_out.get("metadata")
    raise ValueError(
        f"Concept volume / feature volume shape mismatch at {stage}: "
        f"features_5d shape={tuple(features_5d.shape)}, "
        f"concept_volume shape={tuple(concept_volume.shape)}, "
        f"visual_metadata['feature_shape']="
        f"{_metadata_feature_shape_repr(visual_metadata if isinstance(visual_metadata, dict) else None)}, "
        f"metadata['feature_shape']="
        f"{_metadata_feature_shape_repr(metadata if isinstance(metadata, dict) else None)}"
    )


def _concept_volume_to_fusion_map(concept_volume: torch.Tensor) -> torch.Tensor:
    """Aggregate a temporal concept volume to a 2D map for conv fusion blocks."""
    if concept_volume.dim() != 5:
        raise ValueError(
            f"concept_volume must be [B,concept_dim,T,H,W], got {tuple(concept_volume.shape)}"
        )
    return concept_volume.mean(dim=2)


def _resize_concept_volume_to_features(
    concept_volume: torch.Tensor,
    features_5d: torch.Tensor,
) -> torch.Tensor:
    """Trilinearly resize concept volume to match decoder feature [T,H,W]."""
    _, _, T_f, H_f, W_f = features_5d.shape
    if concept_volume.shape[2:] == (T_f, H_f, W_f):
        return concept_volume
    return F.interpolate(
        concept_volume,
        size=(T_f, H_f, W_f),
        mode="trilinear",
        align_corners=False,
    )


def _build_stage_concept_map(
    concept_out: Dict[str, Any],
    target_hw: Tuple[int, int],
    *,
    stage: str = "unknown",
) -> torch.Tensor:
    """
    Build [B, concept_dim, target_h, target_w] visual concept map for one stage.

    Concept tensors are reshaped using visual_metadata feature_shape (B,C,T,H,W),
    then bilinearly interpolated to the decoder backbone feature-map size.
    """
    visual_map = _build_visual_concept_map(concept_out, target_hw, stage=stage)
    if visual_map is None:
        visual_metadata = concept_out.get("visual_metadata")
        raise ValueError(
            f"concept_out at {stage} must provide visual_concept_representation and "
            f"visual_metadata['feature_shape']; got visual_metadata="
            f"{_metadata_feature_shape_repr(visual_metadata if isinstance(visual_metadata, dict) else None)}"
        )

    return visual_map


def _pick_2d_groups(channels: int, preferred: int = 8) -> int:
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _pick_3d_groups(channels: int, preferred: int = 8) -> int:
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _crop_or_pad_3d(
    x: torch.Tensor,
    target_thw: Tuple[int, int, int],
) -> torch.Tensor:
    """Match [T, H, W] via center crop or zero-pad on the temporal/spatial ends."""
    T_t, H_t, W_t = (int(target_thw[0]), int(target_thw[1]), int(target_thw[2]))
    _, _, T, H, W = x.shape

    if T > T_t:
        start = (T - T_t) // 2
        x = x[:, :, start : start + T_t, :, :]
    elif T < T_t:
        x = F.pad(x, (0, 0, 0, 0, 0, T_t - T))

    _, _, T, H, W = x.shape
    if H > H_t:
        start = (H - H_t) // 2
        x = x[:, :, :, start : start + H_t, :]
    elif H < H_t:
        x = F.pad(x, (0, 0, 0, H_t - H))

    _, _, _, H, W = x.shape
    if W > W_t:
        start = (W - W_t) // 2
        x = x[:, :, :, :, start : start + W_t]
    elif W < W_t:
        x = F.pad(x, (0, W_t - W))

    return x


def _align_concept_volume_to_features(
    concept_volume: torch.Tensor,
    features_5d: torch.Tensor,
    *,
    stage: str,
) -> torch.Tensor:
    """
    Align concept volume to decoder features.

    Exact match is expected after concept-feature resizing is fixed. Only
    off-by-one H/W differences are corrected via crop/pad; larger mismatches
    raise an error instead of silently padding large zero regions.
    """
    _, _, T_f, H_f, W_f = features_5d.shape
    _, _, T_c, H_c, W_c = concept_volume.shape

    if T_c != T_f:
        raise ValueError(
            f"Concept volume / feature volume temporal mismatch at {stage}: "
            f"concept_volume shape={tuple(concept_volume.shape)}, "
            f"features_5d shape={tuple(features_5d.shape)}"
        )

    if abs(H_c - H_f) > 1 or abs(W_c - W_f) > 1:
        raise ValueError(
            f"Concept volume / feature volume shape mismatch at {stage}: "
            f"concept_volume shape={tuple(concept_volume.shape)}, "
            f"features_5d shape={tuple(features_5d.shape)}"
        )

    if (H_c, W_c) != (H_f, W_f):
        concept_volume = _crop_or_pad_3d(concept_volume, (T_f, H_f, W_f))

    return concept_volume


def _crop_or_pad_2d(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """Match [H, W] via center crop or zero-pad on the bottom/right."""
    H_t, W_t = int(target_hw[0]), int(target_hw[1])
    _, _, H, W = x.shape

    if H > H_t:
        start = (H - H_t) // 2
        x = x[:, :, start : start + H_t, :]
    elif H < H_t:
        x = F.pad(x, (0, 0, 0, H_t - H))

    _, _, H, W = x.shape
    if W > W_t:
        start = (W - W_t) // 2
        x = x[:, :, :, start : start + W_t]
    elif W < W_t:
        x = F.pad(x, (0, W_t - W))

    return x


class Conv3DGNAct(nn.Module):
    """Conv3d + GroupNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Tuple[int, int, int] = 1,
        padding: int | Tuple[int, int, int] = 0,
        num_groups: int = 8,
    ):
        super().__init__()
        groups = _pick_3d_groups(out_channels, num_groups)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Conv2DGNAct(nn.Module):
    """Conv2d + GroupNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 1,
        padding: int = 0,
        num_groups: int = 8,
    ):
        super().__init__()
        groups = _pick_2d_groups(out_channels, num_groups)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Residual3DRefineBlock(nn.Module):
    """Factorized 3D residual refinement with learnable layer scale."""

    def __init__(
        self,
        channels: int,
        *,
        dropout: float = 0.0,
        num_groups: int = 8,
        layer_scale_init: float = 0.1,
    ):
        super().__init__()
        groups = _pick_3d_groups(channels, num_groups)
        self.temporal_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.spatial_dw = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        self.layer_scale = nn.Parameter(
            torch.full((1, channels, 1, 1, 1), float(layer_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.temporal_dw(x)
        y = self.spatial_dw(y)
        y = self.pointwise(y)
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return residual + self.layer_scale * y


class Residual2DRefineBlock(nn.Module):
    """Lightweight 2D residual refinement block."""

    def __init__(
        self,
        channels: int,
        *,
        dropout: float = 0.05,
        num_groups: int = 8,
    ):
        super().__init__()
        groups = _pick_2d_groups(channels, num_groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.act(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        return self.act(residual + y)


class LearnedSpatialUpsample3D(nn.Module):
    """Learned spatial upsampling for decoder volumes; temporal length is unchanged."""

    def __init__(
        self,
        channels: int,
        *,
        scale_factor: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.scale_factor = int(scale_factor)
        k = self.scale_factor * 2
        p = self.scale_factor // 2
        self.upsample = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=(1, k, k),
            stride=(1, self.scale_factor, self.scale_factor),
            padding=(0, p, p),
            output_padding=(0, 0, 0),
            bias=False,
        )
        groups = _pick_3d_groups(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.refine = Residual3DRefineBlock(channels, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        target_thw: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        y = self.upsample(x)
        y = self.act(self.norm(y))
        y = self.refine(y)
        if target_thw is not None:
            y = _crop_or_pad_3d(y, target_thw)
        return y


class LearnedFinalUpsample2D(nn.Module):
    """Learned final upsampling from decoder patch resolution to output resolution."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        out_channels: int = 1,
        scale_factor: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        if scale_factor < 1 or (scale_factor & (scale_factor - 1)) != 0:
            raise ValueError("scale_factor must be a power of 2")

        self.scale_factor = int(scale_factor)
        num_stages = int(math.log2(self.scale_factor)) if self.scale_factor > 1 else 0
        hidden_channels = max(int(hidden_channels), 1)

        self.upsample_stages = nn.ModuleList()
        ch_in = in_channels
        for _ in range(num_stages):
            self.upsample_stages.append(
                nn.ModuleDict(
                    {
                        "up": nn.ConvTranspose2d(
                            ch_in,
                            hidden_channels,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        "refine": Conv2DGNAct(
                            hidden_channels,
                            hidden_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                    }
                )
            )
            ch_in = hidden_channels

        head_in = hidden_channels if num_stages > 0 else in_channels
        self.head = nn.Conv2d(head_in, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        y = x
        for stage in self.upsample_stages:
            y = stage["up"](y)
            y = stage["refine"](y)
            y = self.dropout(y)
        y = self.head(y)
        if target_hw is not None:
            y = _crop_or_pad_2d(y, target_hw)
        return y


class ConvGNAct(nn.Module):
    """Conv2d + GroupNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 1,
        padding: int = 0,
        num_groups: int = 8,
    ):
        super().__init__()
        groups = _pick_2d_groups(out_channels, num_groups)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualRefineBlock(nn.Module):
    """Lightweight 3x3 residual refinement block."""

    def __init__(
        self,
        channels: int,
        *,
        dropout: float = 0.05,
        num_groups: int = 8,
    ):
        super().__init__()
        groups = _pick_2d_groups(channels, num_groups)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.act(self.norm1(self.conv1(x)))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        return self.act(residual + y)


class ConceptGatedFusionBlock(nn.Module):
    """
    Feature-first fusion block with concept-guided modulation.

    Dense backbone features form the base signal; visual concept maps modulate
    them through FiLM and a gated concept-guidance term. An optional coarser
    decoder map from a previous stage is added with a learnable scale.
    """

    def __init__(
        self,
        feature_channels: int,
        concept_dim: int,
        decoder_channels: int,
        feature_residual_scale: float = 0.25,
        dropout: float = 0.05,
    ):
        super().__init__()
        del feature_residual_scale

        self.feature_proj = ConvGNAct(
            feature_channels,
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.concept_proj = ConvGNAct(
            concept_dim,
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.prev_proj = ConvGNAct(
            decoder_channels,
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.gate_conv = nn.Conv2d(
            decoder_channels * 3,
            decoder_channels,
            kernel_size=3,
            padding=1,
        )
        self.film = nn.Conv2d(decoder_channels, decoder_channels * 2, kernel_size=1)
        self.refine1 = ResidualRefineBlock(decoder_channels, dropout=dropout)
        self.refine2 = ResidualRefineBlock(decoder_channels, dropout=dropout)

        self.concept_scale = nn.Parameter(torch.tensor(0.25))
        self.prev_scale = nn.Parameter(torch.tensor(1.0))

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

        film_params = self.film(concept_proj)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * beta
        feature_proj = feature_proj * gamma + beta

        gate = torch.sigmoid(
            self.gate_conv(torch.cat([feature_proj, concept_proj, prev_up], dim=1))
        )

        fused = feature_proj + self.concept_scale * gate * concept_proj
        if prev_decoder is not None:
            fused = fused + self.prev_scale * prev_up

        decoded = self.refine2(self.refine1(fused))
        return decoded, gate


class SpatioTemporalConceptGatedFusionBlock(nn.Module):
    """
    Spatiotemporal feature-first fusion over [B, C, T, H, W] volumes.

    Fuses full temporal backbone features and concept volumes with optional
    learned upsampling of a coarser previous decoder volume.
    """

    def __init__(
        self,
        feature_channels: int,
        concept_dim: int,
        decoder_channels: int,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.feature_proj = Conv3DGNAct(
            feature_channels,
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.concept_proj = Conv3DGNAct(
            concept_dim,
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.prev_proj = Conv3DGNAct(
            decoder_channels,
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.prev_upsample = LearnedSpatialUpsample3D(
            decoder_channels,
            scale_factor=2,
            dropout=dropout,
        )
        self.gate_conv = nn.Conv3d(
            decoder_channels * 3,
            decoder_channels,
            kernel_size=3,
            padding=1,
        )
        self.film = nn.Conv3d(decoder_channels, decoder_channels * 2, kernel_size=1)
        self.refine1 = Residual3DRefineBlock(decoder_channels, dropout=dropout)
        self.refine2 = Residual3DRefineBlock(decoder_channels, dropout=dropout)

        self.concept_scale = nn.Parameter(torch.tensor(0.25))
        self.prev_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        features: torch.Tensor,
        concept_volume: torch.Tensor,
        prev_decoder: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_proj = self.feature_proj(features)
        concept_proj = self.concept_proj(concept_volume)

        if prev_decoder is not None:
            prev_up = self.prev_upsample(
                prev_decoder,
                target_thw=feature_proj.shape[-3:],
            )
            prev_up = self.prev_proj(prev_up)
        else:
            prev_up = torch.zeros_like(feature_proj)

        film_params = self.film(concept_proj)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * beta
        feature_proj = feature_proj * gamma + beta

        gate = torch.sigmoid(
            self.gate_conv(torch.cat([feature_proj, concept_proj, prev_up], dim=1))
        )

        fused = feature_proj + self.concept_scale * gate * concept_proj
        if prev_decoder is not None:
            fused = fused + self.prev_scale * prev_up

        decoded = self.refine2(self.refine1(fused))
        return decoded, gate


class ConceptGatedMultiScaleSaliencyDecoder(nn.Module):
    """
    Coarse-to-fine spatiotemporal concept-gated decoder.

    Fuses full [B, C, T, H, W] feature and concept volumes per stage, aggregates
    temporally with learned attention, and upsamples with learned conv transpose
    blocks to the output image resolution.
    """

    def __init__(
        self,
        stage_channels: Dict[str, int],
        concept_dim: int = 256,
        decoder_channels: int = 96,
        feature_residual_scale: float = 0.25,
        dropout: float = 0.05,
        tau_pi: float = 0.1,
        output_activation: str = "sigmoid",
        temporal_aggregation: str = "learned_all_frames",
        use_side_logit_fusion: bool = True,
    ):
        super().__init__()
        del feature_residual_scale, tau_pi

        if output_activation not in ("sigmoid", "none"):
            raise ValueError("output_activation must be 'sigmoid' or 'none'")
        if temporal_aggregation not in ("learned_all_frames", "mean", "last"):
            raise ValueError(
                "temporal_aggregation must be one of "
                "'learned_all_frames', 'mean', or 'last'"
            )

        self.stage_channels = dict(stage_channels)
        self.concept_dim = concept_dim
        self.decoder_channels = decoder_channels
        self.dropout = dropout
        self.output_activation = output_activation
        self.temporal_aggregation = temporal_aggregation
        self.use_side_logit_fusion = bool(use_side_logit_fusion)

        hidden_channels = max(decoder_channels // 2, 32)
        attn_hidden = max(decoder_channels // 2, 1)

        self.fusion_blocks = nn.ModuleDict(
            {
                stage: SpatioTemporalConceptGatedFusionBlock(
                    feature_channels=channels,
                    concept_dim=concept_dim,
                    decoder_channels=decoder_channels,
                    dropout=dropout,
                )
                for stage, channels in self.stage_channels.items()
            }
        )

        self.temporal_weight_head = nn.Sequential(
            nn.Conv3d(decoder_channels, attn_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(attn_hidden, 1, kernel_size=1),
        )
        self.temporal_context_gate = nn.Parameter(torch.tensor(-2.0))

        self.final_upsample_head = LearnedFinalUpsample2D(
            in_channels=decoder_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            scale_factor=4,
            dropout=dropout,
        )
        self.patch_logit_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

        self.side_feature_heads = nn.ModuleDict(
            {
                stage: nn.Conv3d(decoder_channels, decoder_channels, kernel_size=1)
                for stage in self.stage_channels
            }
        )
        self.side_patch_heads = nn.ModuleDict(
            {
                stage: nn.Conv2d(decoder_channels, 1, kernel_size=1)
                for stage in self.stage_channels
            }
        )
        self.side_upsample_heads = nn.ModuleDict(
            {
                stage: LearnedFinalUpsample2D(
                    in_channels=decoder_channels,
                    hidden_channels=hidden_channels,
                    out_channels=1,
                    scale_factor=_SIDE_UPSAMPLE_SCALES.get(stage, 4),
                    dropout=dropout,
                )
                for stage in self.stage_channels
            }
        )

        self._decode_stages = tuple(
            stage for stage in _STAGE_ORDER if stage in self.stage_channels
        )
        num_logits = len(self._decode_stages) + 1
        side_fusion_init = torch.full((num_logits,), -2.0)
        side_fusion_init[0] = 4.0
        self.side_fusion_logits = nn.Parameter(side_fusion_init)

    def _fuse_main_and_side_logits(
        self,
        main_logits: torch.Tensor,
        side_saliency_logits: Dict[str, torch.Tensor],
        stages: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        logits_to_fuse = [main_logits]
        for stage in stages:
            side_logits = side_saliency_logits.get(stage)
            if side_logits is not None:
                logits_to_fuse.append(side_logits)

        if not self.use_side_logit_fusion or len(logits_to_fuse) == 1:
            return main_logits, main_logits, None

        weights = torch.softmax(
            self.side_fusion_logits[: len(logits_to_fuse)],
            dim=0,
        )
        fused_logits = sum(
            weight * logit for weight, logit in zip(weights, logits_to_fuse)
        )
        return fused_logits, main_logits, weights.detach()

    def _ordered_stages(
        self,
        concept_outs: Dict[str, Dict[str, Any]],
        features_dict: Dict[str, torch.Tensor],
    ) -> List[str]:
        return [
            stage
            for stage in self._decode_stages
            if stage in concept_outs and stage in features_dict
        ]

    def _aggregate_temporal_features(
        self,
        decoded_volume: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if decoded_volume.dim() != 5:
            raise ValueError(
                f"decoded_volume must be [B,D,T,H,W], got {tuple(decoded_volume.shape)}"
            )

        if self.temporal_aggregation == "learned_all_frames":
            last = decoded_volume[:, :, -1]
            scores = self.temporal_weight_head(decoded_volume)
            weights = torch.softmax(scores, dim=2)
            context = (decoded_volume * weights).sum(dim=2)
            alpha = torch.sigmoid(self.temporal_context_gate)
            feature_2d = last + alpha * context
            return feature_2d, weights

        if self.temporal_aggregation == "mean":
            return decoded_volume.mean(dim=2), None

        if self.temporal_aggregation == "last":
            return decoded_volume[:, :, -1], None

        raise ValueError(f"Unsupported temporal_aggregation: {self.temporal_aggregation}")

    def _build_decoder_temporal_diagnostics(
        self,
        temporal_weights: Optional[torch.Tensor],
        side_temporal_weights: Dict[str, torch.Tensor],
        *,
        include_weight_stats: bool = False,
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "temporal_aggregation": self.temporal_aggregation,
            "side_stages": sorted(side_temporal_weights.keys()),
        }
        if include_weight_stats and temporal_weights is not None:
            diagnostics["final_temporal_weight_std"] = float(
                temporal_weights.detach().std().cpu()
            )
            diagnostics["final_temporal_weight_max"] = float(
                temporal_weights.detach().max().cpu()
            )
        return diagnostics

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

        fusion_blocks = self.fusion_blocks
        side_feature_heads = self.side_feature_heads
        side_patch_heads = self.side_patch_heads
        side_upsample_heads = self.side_upsample_heads

        stage_concept_volumes: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_feature_volumes: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_gates: Optional[Dict[str, torch.Tensor]] = {} if return_details else None
        decoded_stage_volumes: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        side_saliency_logits: Dict[str, torch.Tensor] = {}
        side_patch_logits: Dict[str, torch.Tensor] = {}
        side_temporal_weights: Dict[str, torch.Tensor] = {}

        prev_decoder: Optional[torch.Tensor] = None

        for stage in stages:
            features_5d = features_dict[stage]
            if features_5d.dim() != 5:
                raise ValueError(
                    f"features_dict[{stage}] must be [B,C,T,H,W], "
                    f"got {tuple(features_5d.shape)}"
                )

            concept_volume = _build_stage_concept_volume(
                concept_outs[stage],
                stage=stage,
            )
            concept_volume = _align_concept_volume_to_features(
                concept_volume,
                features_5d,
                stage=stage,
            )
            _assert_concept_volume_matches_features(
                stage=stage,
                concept_volume=concept_volume,
                features_5d=features_5d,
                concept_out=concept_outs[stage],
            )

            decoded, gate = fusion_blocks[stage](
                features_5d,
                concept_volume,
                prev_decoder=prev_decoder,
            )
            prev_decoder = decoded

            side_volume = side_feature_heads[stage](decoded)
            side_feature_2d, side_weights = self._aggregate_temporal_features(side_volume)
            side_patch_logits[stage] = side_patch_heads[stage](side_feature_2d)
            side_saliency_logits[stage] = side_upsample_heads[stage](
                side_feature_2d,
                target_hw=output_size,
            )
            if side_weights is not None:
                side_temporal_weights[stage] = side_weights.detach()

            if return_details:
                stage_concept_volumes[stage] = concept_volume.detach()
                stage_feature_volumes[stage] = features_5d.detach()
                stage_gates[stage] = gate.detach()
                decoded_stage_volumes[stage] = decoded.detach()

        if prev_decoder is None:
            raise RuntimeError("Decoder produced no stage outputs")

        final_feature_2d, temporal_weights = self._aggregate_temporal_features(
            prev_decoder
        )
        patch_logits = self.patch_logit_head(final_feature_2d)
        main_saliency_logits_unfused = self.final_upsample_head(
            final_feature_2d,
            target_hw=output_size,
        )
        saliency_logits, _, side_fusion_weights = self._fuse_main_and_side_logits(
            main_saliency_logits_unfused,
            side_saliency_logits,
            stages,
        )

        if self.output_activation == "sigmoid":
            saliency_map = torch.sigmoid(saliency_logits)
        else:
            saliency_map = saliency_logits

        decoder_temporal_diagnostics = self._build_decoder_temporal_diagnostics(
            temporal_weights,
            side_temporal_weights,
            include_weight_stats=return_details,
        )

        out: Dict[str, Any] = {
            "saliency_logits": saliency_logits,
            "saliency_map": saliency_map,
            "patch_saliency_logits": patch_logits,
            "coarse_saliency_logits": saliency_logits,
            "main_saliency_logits_unfused": main_saliency_logits_unfused,
            "side_fusion_weights": side_fusion_weights,
            "side_saliency_logits": side_saliency_logits,
            "side_patch_saliency_logits": side_patch_logits,
            "temporal_weights": temporal_weights.detach()
            if temporal_weights is not None
            else None,
            "side_temporal_weights": side_temporal_weights,
            "output_activation": self.output_activation,
            "decoder_temporal_diagnostics": decoder_temporal_diagnostics,
        }

        if return_details:
            out["stage_concept_volumes"] = stage_concept_volumes
            out["stage_feature_volumes"] = stage_feature_volumes
            out["stage_gates"] = stage_gates
            out["decoded_stage_volumes"] = decoded_stage_volumes

        return out
