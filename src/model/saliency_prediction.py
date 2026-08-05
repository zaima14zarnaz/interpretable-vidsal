"""
Concept-guided FiLM-and-mask multi-scale spatiotemporal saliency decoder for the last frame in a window.

Stages are decoded coarse-to-fine (stage4 -> stage1). Each stage fuses full
[B, C, T, H, W] feature and concept volumes through FiLM modulation and
concept-derived feature masking, aggregates over time with learned attention,
and upsamples with learned conv-transpose blocks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import compute_stage_mask_diversity_losses

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


def _build_visual_agreement_volume(
    concept_out: Dict[str, Any],
    *,
    stage: str = "unknown",
) -> Optional[torch.Tensor]:
    """
    Build [B, 1, T, H, W] visual feature-concept agreement volume for one stage.
    """
    agreement = concept_out.get("visual_feature_concept_agreement")
    visual_metadata = concept_out.get("visual_metadata")
    if not torch.is_tensor(agreement) or not isinstance(visual_metadata, dict):
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
    if agreement.dim() != 1 or agreement.shape[0] != expected:
        raise ValueError(
            f"visual_feature_concept_agreement length mismatch at {stage}: "
            f"expected {expected} (=B*T*H*W from feature_shape "
            f"{_metadata_feature_shape_repr(visual_metadata)}), "
            f"got {tuple(agreement.shape)}"
        )

    return (
        agreement.reshape(B, T, H, W)
        .unsqueeze(1)
        .contiguous()
    )


def _build_motion_concept_volume(
    concept_out: Dict[str, Any],
    *,
    stage: str = "unknown",
) -> Optional[torch.Tensor]:
    """
    Build [B, concept_dim, T, H, W] motion concept volume for one stage.

    Motion representations are spatial maps repeated across time because motion
    concepts summarize temporal change at each cell rather than per-frame appearance.
    """
    motion_repr = concept_out.get("motion_concept_representation")
    motion_metadata = concept_out.get("motion_metadata")
    if not torch.is_tensor(motion_repr) or not isinstance(motion_metadata, dict):
        return None
    if "feature_shape" not in motion_metadata:
        return None

    try:
        B, _, T, H, W = _feature_shape_from_metadata(motion_metadata)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid motion_metadata['feature_shape'] at {stage}: "
            f"{_metadata_feature_shape_repr(motion_metadata)}"
        ) from exc

    if motion_repr.dim() == 2:
        expected = B * H * W
        if motion_repr.shape[0] != expected:
            raise ValueError(
                f"motion_concept_representation length mismatch at {stage}: "
                f"expected {expected} (=B*H*W from feature_shape "
                f"{_metadata_feature_shape_repr(motion_metadata)}), "
                f"got {motion_repr.shape[0]}"
            )
        concept_dim = motion_repr.shape[-1]
        motion_map = (
            motion_repr.reshape(B, H, W, concept_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
    elif motion_repr.dim() == 4:
        if motion_repr.shape[0] != B:
            raise ValueError(
                f"motion_concept_representation batch mismatch at {stage}: "
                f"expected B={B} from motion_metadata, got {motion_repr.shape[0]}"
            )
        concept_dim = motion_repr.shape[1]
        if motion_repr.shape[-2:] != (H, W):
            raise ValueError(
                f"motion_concept_representation spatial mismatch at {stage}: "
                f"expected H,W=({H},{W}) from motion_metadata, "
                f"got {tuple(motion_repr.shape[-2:])}"
            )
        motion_map = motion_repr
    else:
        raise ValueError(
            f"motion_concept_representation at {stage} must be "
            f"[B*H*W, concept_dim] or [B, concept_dim, H, W], "
            f"got shape {tuple(motion_repr.shape)}"
        )

    motion_volume = (
        motion_map.unsqueeze(2)
        .expand(B, concept_dim, T, H, W)
        .contiguous()
    )
    return motion_volume


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
    """Conv3d + GroupNorm + ReLU."""

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
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Conv2DGNAct(nn.Module):
    """Conv2d + GroupNorm + ReLU."""

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
        self.act = nn.ReLU()

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
        self.act = nn.ReLU()
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
        self.act = nn.ReLU()
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


def _compute_multi_mask_diagnostics(
    multi_masks: torch.Tensor,
    *,
    mask_strength: float,
) -> Dict[str, float]:
    """
    Scalar diagnostics for final cosine-derived multi-masks [B,R,T,H,W].

    All statistics are computed under ``torch.no_grad()``.
    """
    with torch.no_grad():
        if multi_masks.dim() != 5:
            raise ValueError(
                f"multi_masks must be [B,R,T,H,W], got {tuple(multi_masks.shape)}"
            )

        between_mask_variance = float(
            multi_masks.var(dim=1, unbiased=False).mean().cpu()
        )
        per_mask_spatial_variance = float(
            multi_masks.flatten(2).var(dim=-1, unbiased=False).mean().cpu()
        )

        masks_flat = multi_masks.flatten(start_dim=2)
        centered = masks_flat - masks_flat.mean(dim=-1, keepdim=True)
        normalized = F.normalize(centered, p=2, dim=-1, eps=1e-6)
        gram = normalized @ normalized.transpose(1, 2)
        num_masks = int(gram.shape[1])
        if num_masks <= 1:
            mean_pairwise_mask_correlation = 0.0
        else:
            off_diag_mask = ~torch.eye(
                num_masks,
                dtype=torch.bool,
                device=gram.device,
            )
            mean_pairwise_mask_correlation = float(
                gram[:, off_diag_mask].abs().mean().cpu()
            )

        diagnostics: Dict[str, float] = {
            "between_mask_variance": between_mask_variance,
            "per_mask_spatial_variance": per_mask_spatial_variance,
            "mean_pairwise_mask_correlation": mean_pairwise_mask_correlation,
            "mask_strength": float(mask_strength),
        }

        for mask_idx in range(num_masks):
            mask_r = multi_masks[:, mask_idx]
            label = mask_idx + 1
            diagnostics[f"mask_{label}_mean"] = float(mask_r.mean().cpu())
            diagnostics[f"mask_{label}_std"] = float(
                mask_r.std(unbiased=False).cpu()
            )
            diagnostics[f"mask_{label}_min"] = float(mask_r.min().cpu())
            diagnostics[f"mask_{label}_max"] = float(mask_r.max().cpu())

        return diagnostics


def _masked_softmax(
    logits: torch.Tensor,
    validity_mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Softmax over dim with invalid concept slots masked to -inf."""
    if validity_mask.dim() == 2 and logits.dim() == 3:
        mask = validity_mask.unsqueeze(1).expand(-1, logits.shape[1], -1)
    else:
        mask = validity_mask
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(masked_logits, dim=dim)


def _motion_prototypes_available(concept_out: Dict[str, Any]) -> bool:
    prototypes = concept_out.get("active_motion_prototypes")
    validity = concept_out.get("motion_validity_mask")
    if not torch.is_tensor(prototypes) or not torch.is_tensor(validity):
        return False
    return bool(validity.any().item())


class SpatioTemporalConceptGatedFusionBlock(nn.Module):
    """
    Feature-first 3D fusion over [B, C, T, H, W] volumes.

    Dense features form the base signal. Concept volumes first modulate dense
    features through FiLM, then R independent cosine-similarity concept masks
    gate separate lightweight feature branches whose fused update is added
    residually. A coarser previous decoder volume may be upsampled and added
    afterward.
    """

    def __init__(
        self,
        feature_channels: int,
        concept_dim: int,
        decoder_channels: int,
        dropout: float = 0.05,
        num_masks: int = 4,
        mask_embed_dim: Optional[int] = None,
        max_strength: float = 1.0,
    ):
        super().__init__()

        if num_masks < 1:
            raise ValueError(f"num_masks must be >= 1, got {num_masks}")

        self.num_masks = int(num_masks)
        self.decoder_channels = decoder_channels
        self.mask_embed_dim = (
            int(mask_embed_dim)
            if mask_embed_dim is not None
            else max(decoder_channels // 2, 32)
        )
        self.max_strength = float(max_strength)

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
        self.film = nn.Conv3d(decoder_channels, decoder_channels * 2, kernel_size=1)
        self.refine1 = Residual3DRefineBlock(decoder_channels, dropout=dropout)
        self.refine2 = Residual3DRefineBlock(decoder_channels, dropout=dropout)

        self.feature_token_proj = nn.Linear(decoder_channels, self.mask_embed_dim)
        self.concept_proto_proj = nn.Linear(concept_dim, self.mask_embed_dim)
        self.visual_mask_queries = nn.Parameter(
            torch.randn(self.num_masks, self.mask_embed_dim) * 0.02
        )
        self.motion_mask_queries = nn.Parameter(
            torch.randn(self.num_masks, self.mask_embed_dim) * 0.02
        )
        self.mask_feature_branches = nn.ModuleList(
            [
                Conv3DGNAct(
                    decoder_channels,
                    decoder_channels,
                    kernel_size=1,
                    padding=0,
                )
                for _ in range(self.num_masks)
            ]
        )
        self.mask_fusion = nn.Conv3d(
            decoder_channels * self.num_masks,
            decoder_channels,
            kernel_size=1,
            bias=True,
        )

        self.gate_temperature = nn.Parameter(torch.ones(self.num_masks))
        self.gate_threshold = nn.Parameter(torch.zeros(self.num_masks))
        # sigmoid(-2.197) ≈ 0.1 -> strength ≈ 0.1 when max_strength=1.0
        self.raw_strength = nn.Parameter(torch.tensor(1.0))

        self.prev_scale = nn.Parameter(torch.tensor(1.0))

    def _compute_modality_masks(
        self,
        feature_proj: torch.Tensor,
        active_prototypes: torch.Tensor,
        validity_mask: torch.Tensor,
        mask_queries: torch.Tensor,
        num_masks: int,
        gate_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, T, H, W = feature_proj.shape
        N = T * H * W
        K = active_prototypes.shape[1]

        feature_tokens = feature_proj.permute(0, 2, 3, 4, 1).reshape(B, N, -1)
        feature_tokens = F.normalize(self.feature_token_proj(feature_tokens), dim=-1)
        proto_tokens = F.normalize(
            self.concept_proto_proj(active_prototypes),
            dim=-1,
        )

        concept_maps_flat = torch.einsum("bkd,bnd->bkn", proto_tokens, feature_tokens)
        concept_maps = concept_maps_flat.reshape(B, K, T, H, W)

        weight_logits = torch.stack(
            [
                torch.einsum("bkd,d->bk", proto_tokens, mask_queries[r])
                for r in range(num_masks)
            ],
            dim=1,
        )
        concept_weights = _masked_softmax(weight_logits, validity_mask, dim=-1)

        multi_masks_flat = torch.einsum(
            "brk,bkn->brn",
            concept_weights,
            concept_maps_flat,
        )
        multi_masks = multi_masks_flat.reshape(B, num_masks, T, H, W)

        gate_indices = torch.arange(
            gate_offset,
            gate_offset + num_masks,
            device=feature_proj.device,
        )
        temps = self.gate_temperature.index_select(0, gate_indices).view(
            1, num_masks, 1, 1, 1
        )
        thresholds = self.gate_threshold.index_select(0, gate_indices).view(
            1, num_masks, 1, 1, 1
        )
        multi_mask_gates = torch.sigmoid(temps * (multi_masks - thresholds))

        return concept_maps, concept_weights, multi_masks, multi_mask_gates

    def _apply_multi_mask_branches(
        self,
        feature_proj: torch.Tensor,
        multi_mask_gates: torch.Tensor,
    ) -> torch.Tensor:
        gated_branches = []
        for r in range(self.num_masks):
            branch_r = self.mask_feature_branches[r](feature_proj)
            gate_r = multi_mask_gates[:, r : r + 1]
            gated_branches.append(gate_r * branch_r)
        return self.mask_fusion(torch.cat(gated_branches, dim=1))

    def forward(
        self,
        features: torch.Tensor,
        concept_volume: torch.Tensor,
        prev_decoder: Optional[torch.Tensor] = None,
        *,
        active_visual_prototypes: torch.Tensor,
        visual_validity_mask: torch.Tensor,
        active_motion_prototypes: Optional[torch.Tensor] = None,
        motion_validity_mask: Optional[torch.Tensor] = None,
        use_motion_masks: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        motion_available = (
            use_motion_masks
            and active_motion_prototypes is not None
            and motion_validity_mask is not None
            and bool(motion_validity_mask.any().item())
        )

        if motion_available:
            num_visual_masks = self.num_masks // 2
            num_motion_masks = self.num_masks - num_visual_masks
            visual_maps, visual_weights, visual_masks, visual_gates = (
                self._compute_modality_masks(
                    feature_proj,
                    active_visual_prototypes,
                    visual_validity_mask,
                    self.visual_mask_queries,
                    num_visual_masks,
                    gate_offset=0,
                )
            )
            motion_maps, motion_weights, motion_masks, motion_gates = (
                self._compute_modality_masks(
                    feature_proj,
                    active_motion_prototypes,
                    motion_validity_mask,
                    self.motion_mask_queries,
                    num_motion_masks,
                    gate_offset=num_visual_masks,
                )
            )
            multi_masks = torch.cat([visual_masks, motion_masks], dim=1)
            multi_mask_gates = torch.cat([visual_gates, motion_gates], dim=1)
            mask_modality_labels = torch.tensor(
                [0] * num_visual_masks + [1] * num_motion_masks,
                device=feature_proj.device,
                dtype=torch.long,
            )
            motion_concept_maps = motion_maps
            motion_concept_weights = motion_weights
        else:
            visual_maps, visual_weights, multi_masks, multi_mask_gates = (
                self._compute_modality_masks(
                    feature_proj,
                    active_visual_prototypes,
                    visual_validity_mask,
                    self.visual_mask_queries,
                    self.num_masks,
                    gate_offset=0,
                )
            )
            mask_modality_labels = torch.zeros(
                self.num_masks,
                device=feature_proj.device,
                dtype=torch.long,
            )
            motion_concept_maps = None
            motion_concept_weights = None

        fused_update = self._apply_multi_mask_branches(feature_proj, multi_mask_gates)
        strength = self.max_strength * torch.sigmoid(self.raw_strength)
        feature_proj = feature_proj + strength * fused_update

        fused = feature_proj
        if prev_decoder is not None:
            fused = fused + self.prev_scale * prev_up

        decoded = self.refine2(self.refine1(fused))

        concept_mask = multi_mask_gates.mean(dim=1, keepdim=True)

        mask_outputs: Dict[str, torch.Tensor] = {
            "concept_mask": concept_mask,
            "multi_masks": multi_masks,
            "multi_mask_gates": multi_mask_gates,
            "visual_concept_maps": visual_maps,
            "visual_concept_weights": visual_weights,
            "mask_modality_labels": mask_modality_labels,
        }
        if motion_concept_maps is not None:
            mask_outputs["motion_concept_maps"] = motion_concept_maps
        if motion_concept_weights is not None:
            mask_outputs["motion_concept_weights"] = motion_concept_weights

        return decoded, mask_outputs


class ConceptGatedMultiScaleSaliencyDecoder(nn.Module):
    """
    Coarse-to-fine spatiotemporal concept-guided FiLM-and-mask decoder.

    Fuses full [B, C, T, H, W] feature and concept volumes per stage through
    concept-guided FiLM modulation and concept-derived feature masking, aggregates
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
        motion_concept_stages: Tuple[str, ...] = ("stage3",),
        num_masks: int = 4,
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
        self.num_masks = int(num_masks)

        hidden_channels = max(decoder_channels // 2, 32)
        attn_hidden = max(decoder_channels // 2, 1)

        self.fusion_blocks = nn.ModuleDict(
            {
                stage: SpatioTemporalConceptGatedFusionBlock(
                    feature_channels=channels,
                    concept_dim=concept_dim,
                    decoder_channels=decoder_channels,
                    dropout=dropout,
                    num_masks=self.num_masks,
                )
                for stage, channels in self.stage_channels.items()
            }
        )

        self.temporal_weight_head = nn.Sequential(
            nn.Conv3d(decoder_channels, attn_hidden, kernel_size=1),
            nn.ReLU(),
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

        motion_concept_stages = tuple(
            stage for stage in motion_concept_stages if stage != "stage4"
        )
        allowed_motion_stages = {"stage3"}
        invalid_motion_stages = set(motion_concept_stages) - allowed_motion_stages
        if invalid_motion_stages:
            raise ValueError(
                "Only stage3 motion concept fusion is currently supported. "
                f"Got invalid stages: {sorted(invalid_motion_stages)}"
            )
        self.motion_concept_stages = motion_concept_stages
        self.motion_concept_logit_scales = nn.ParameterDict(
            {
                stage: nn.Parameter(torch.tensor(-1.0))
                for stage in self.motion_concept_stages
            }
        )
        self.motion_concept_stage = (
            self.motion_concept_stages[0]
            if len(self.motion_concept_stages) > 0
            else "stage3"
        )

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
        stage_concept_masks: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_mask_diagnostics: Optional[Dict[str, Dict[str, float]]] = (
            {} if return_details else None
        )
        stage_multi_masks: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_multi_mask_gates: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_visual_concept_maps: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_visual_concept_weights: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_motion_concept_maps: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_motion_concept_weights: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_mask_modality_labels: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        decoded_stage_volumes: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_motion_concept_volumes: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        side_saliency_logits: Dict[str, torch.Tensor] = {}
        side_patch_logits: Dict[str, torch.Tensor] = {}
        side_temporal_weights: Dict[str, torch.Tensor] = {}
        stage_multi_masks_for_loss: Dict[str, torch.Tensor] = {}

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

            motion_volume: Optional[torch.Tensor] = None
            # Motion concepts are intentionally fused only at stage3.
            # Stage4 motion fusion was tested and removed because it hurt performance.
            # Stage4 visual/backbone features are still decoded normally.
            if stage == "stage3" and stage in self.motion_concept_stages:
                motion_volume = _build_motion_concept_volume(
                    concept_outs[stage],
                    stage=stage,
                )
                if motion_volume is not None:
                    motion_volume = _align_concept_volume_to_features(
                        motion_volume,
                        features_5d,
                        stage=stage,
                    )
                    motion_scale = torch.sigmoid(
                        self.motion_concept_logit_scales[stage]
                    )
                    concept_volume = concept_volume + motion_scale * motion_volume

            _assert_concept_volume_matches_features(
                stage=stage,
                concept_volume=concept_volume,
                features_5d=features_5d,
                concept_out=concept_outs[stage],
            )

            stage_concept_out = concept_outs[stage]
            active_visual_prototypes = stage_concept_out.get(
                "active_visual_prototypes"
            )
            visual_validity_mask = stage_concept_out.get("visual_validity_mask")
            if (
                not torch.is_tensor(active_visual_prototypes)
                or not torch.is_tensor(visual_validity_mask)
            ):
                raise ValueError(
                    f"concept_out at {stage} must provide active_visual_prototypes "
                    f"and visual_validity_mask"
                )

            use_motion_masks = (
                stage == "stage3"
                and stage in self.motion_concept_stages
                and _motion_prototypes_available(stage_concept_out)
            )
            active_motion_prototypes = stage_concept_out.get(
                "active_motion_prototypes"
            )
            motion_validity_mask = stage_concept_out.get("motion_validity_mask")

            decoded, mask_outputs = fusion_blocks[stage](
                features_5d,
                concept_volume,
                prev_decoder=prev_decoder,
                active_visual_prototypes=active_visual_prototypes,
                visual_validity_mask=visual_validity_mask,
                active_motion_prototypes=active_motion_prototypes,
                motion_validity_mask=motion_validity_mask,
                use_motion_masks=use_motion_masks,
            )
            concept_mask = mask_outputs["concept_mask"]
            stage_multi_masks_for_loss[stage] = mask_outputs["multi_masks"]
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
                stage_concept_masks[stage] = concept_mask.detach()
                stage_gates[stage] = concept_mask.detach()
                stage_multi_masks[stage] = mask_outputs["multi_masks"]
                stage_multi_mask_gates[stage] = mask_outputs["multi_mask_gates"]
                stage_visual_concept_maps[stage] = mask_outputs["visual_concept_maps"]
                stage_visual_concept_weights[stage] = mask_outputs[
                    "visual_concept_weights"
                ]
                stage_mask_modality_labels[stage] = mask_outputs[
                    "mask_modality_labels"
                ]
                if mask_outputs.get("motion_concept_maps") is not None:
                    stage_motion_concept_maps[stage] = mask_outputs[
                        "motion_concept_maps"
                    ]
                if mask_outputs.get("motion_concept_weights") is not None:
                    stage_motion_concept_weights[stage] = mask_outputs[
                        "motion_concept_weights"
                    ]
                decoded_stage_volumes[stage] = decoded.detach()
                with torch.no_grad():
                    multi_masks = mask_outputs["multi_masks"]
                    multi_mask_gates = mask_outputs["multi_mask_gates"]
                    mask_strength = float(
                        (
                            fusion_blocks[stage].max_strength
                            * torch.sigmoid(fusion_blocks[stage].raw_strength)
                        )
                        .detach()
                        .cpu()
                    )
                    multi_mask_diag = _compute_multi_mask_diagnostics(
                        multi_masks,
                        mask_strength=mask_strength,
                    )
                    stage_mask_diagnostics[stage] = {
                        "concept_mask_mean": float(
                            concept_mask.detach().mean().cpu()
                        ),
                        "concept_mask_std": float(
                            concept_mask.detach().std().cpu()
                        ),
                        "concept_mask_min": float(
                            concept_mask.detach().min().cpu()
                        ),
                        "concept_mask_max": float(
                            concept_mask.detach().max().cpu()
                        ),
                        "multi_mask_mean": float(multi_masks.mean().cpu()),
                        "multi_mask_std": float(multi_masks.std().cpu()),
                        "multi_mask_gate_mean": float(
                            multi_mask_gates.mean().cpu()
                        ),
                        "multi_mask_gate_std": float(
                            multi_mask_gates.std().cpu()
                        ),
                        "residual_strength": mask_strength,
                        "num_masks": float(self.num_masks),
                        **multi_mask_diag,
                    }
                if motion_volume is not None:
                    stage_motion_concept_volumes[stage] = motion_volume.detach()

        if prev_decoder is None:
            raise RuntimeError("Decoder produced no stage outputs")

        mask_diversity_loss, stage_mask_diversity_losses = (
            compute_stage_mask_diversity_losses(
                stage_multi_masks_for_loss,
                reference=prev_decoder,
            )
        )

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
            "mask_diversity_loss": mask_diversity_loss,
            "stage_mask_diversity_losses": stage_mask_diversity_losses,
        }

        if return_details:
            out["stage_concept_volumes"] = stage_concept_volumes
            out["stage_feature_volumes"] = stage_feature_volumes
            out["stage_gates"] = stage_gates
            out["stage_concept_masks"] = stage_concept_masks
            out["stage_mask_diagnostics"] = stage_mask_diagnostics
            out["stage_multi_masks"] = stage_multi_masks
            out["stage_multi_mask_gates"] = stage_multi_mask_gates
            out["stage_visual_concept_maps"] = stage_visual_concept_maps
            out["stage_visual_concept_weights"] = stage_visual_concept_weights
            out["stage_motion_concept_maps"] = stage_motion_concept_maps
            out["stage_motion_concept_weights"] = stage_motion_concept_weights
            out["stage_mask_modality_labels"] = stage_mask_modality_labels
            out["decoded_stage_volumes"] = decoded_stage_volumes
            out["stage_motion_concept_volumes"] = stage_motion_concept_volumes

        return out