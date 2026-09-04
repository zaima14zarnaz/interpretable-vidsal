"""
Concept-guided FiLM-and-mask multi-scale spatiotemporal saliency decoder for the last frame in a window.

Stages are decoded coarse-to-fine (stage4 -> stage1). Each stage fuses full
[B, C, T, H, W] feature and concept volumes through FiLM modulation. Stages 3
and 4 score every valid concept-patch activation with unary scores and a
low-rank factorized antisymmetric concept-patch comparison modulated by an
explicit learned spatial-relevance kernel. Every directed pair interaction is
defined implicitly through query/key projections and a symmetric Fourier spatial
kernel, while its activation-weighted aggregate is computed exactly via augmented
global query/key summaries in O(N) memory and computation. FiLM features are then
updated with a 1x1x1 conv-GN-ReLU branch and the patch-priority map is applied
as residual spatial gating of that update. Temporal aggregation and learned
upsampling produce the last-frame saliency map.
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


_GROUPING_EPS = 1e-6
_PRIORITY_EPS = 1e-6
_MASKED_LOGIT_FILL = -1e4
_PATCH_PRIORITY_STAGES = ("stage3", "stage4")
_POS_NUM_FREQUENCIES = 8


def _masked_softmax(
    logits: torch.Tensor,
    validity_mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Numerically safe softmax over valid concept slots.

    Invalid slots receive exactly zero weight. Rows with at least one valid
    slot sum to one. Rows with no valid slot return zeros rather than NaNs.
    """
    if validity_mask.dim() == 2 and logits.dim() == 3:
        mask = validity_mask.unsqueeze(1).expand(-1, logits.shape[1], -1)
    else:
        mask = validity_mask
    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, _MASKED_LOGIT_FILL)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    normalizer = weights.sum(dim=dim, keepdim=True)
    weights = weights / normalizer.clamp_min(_GROUPING_EPS)
    return torch.where(normalizer > 0, weights, torch.zeros_like(weights))


def _sinusoidal_2d_encoding(
    x: torch.Tensor,
    y: torch.Tensor,
    num_frequencies: int,
) -> torch.Tensor:
    """Fixed sinusoidal encoding of normalized coordinates in ``[0, 1]``."""
    freqs = (2.0 ** torch.arange(
        num_frequencies, device=x.device, dtype=x.dtype
    )) * math.pi
    x_ang = x.unsqueeze(-1) * freqs
    y_ang = y.unsqueeze(-1) * freqs
    return torch.cat(
        [x_ang.sin(), x_ang.cos(), y_ang.sin(), y_ang.cos()],
        dim=-1,
    )


class SpatioTemporalConceptGatedFusionBlock(nn.Module):
    """
    Feature-first 3D fusion over [B, C, T, H, W] volumes.

    Dense features form the base signal. Concept volumes first modulate dense
    features through FiLM.     At stages 3 and 4, every valid concept-patch activation is scored with
    unary terms and a low-rank factorized antisymmetric concept-patch
    comparison modulated by an explicit learned spatial-relevance kernel.
    Each directed pair score combines content antisymmetry with a symmetric
    Fourier spatial kernel; the activation-weighted aggregate is computed
    exactly from augmented global query/key summaries without materializing
    ``N x N`` tensors. A 1x1x1 conv-GroupNorm-ReLU branch then updates the
    FiLM features; the patch-priority map gates that update residually:

    ``guided = film_features + alpha_stage * (priority_mask * branch(film_features))``.

    A coarser previous decoder volume may be upsampled and added afterward.
    """

    def __init__(
        self,
        feature_channels: int,
        concept_dim: int,
        decoder_channels: int,
        dropout: float = 0.05,
        mask_embed_dim: Optional[int] = None,
        max_strength: float = 1.0,
        assignment_temperature: float = 0.07,
        priority_temperature: float = 0.1,
        use_null_prototype: bool = True,
        enable_patch_prioritization: bool = False,
        factorized_rank: int = 32,
        pos_num_frequencies: int = _POS_NUM_FREQUENCIES,
    ):
        super().__init__()

        if assignment_temperature <= 0.0:
            raise ValueError(
                f"assignment_temperature must be > 0, got {assignment_temperature}"
            )
        if priority_temperature <= 0.0:
            raise ValueError(
                f"priority_temperature must be > 0, got {priority_temperature}"
            )
        if factorized_rank < 1:
            raise ValueError(f"factorized_rank must be >= 1, got {factorized_rank}")
        if pos_num_frequencies < 1:
            raise ValueError(
                f"pos_num_frequencies must be >= 1, got {pos_num_frequencies}"
            )

        self.concept_dim = int(concept_dim)
        self.decoder_channels = decoder_channels
        self.mask_embed_dim = (
            int(mask_embed_dim)
            if mask_embed_dim is not None
            else max(decoder_channels // 2, 32)
        )
        self.max_strength = float(max_strength)
        self.assignment_temperature = float(assignment_temperature)
        self.priority_temperature = float(priority_temperature)
        self.use_null_prototype = bool(use_null_prototype)
        self.enable_patch_prioritization = bool(enable_patch_prioritization)
        self.factorized_rank = int(factorized_rank)
        self.pos_num_frequencies = int(pos_num_frequencies)
        self.pos_dim = 4 * self.pos_num_frequencies
        self.token_dim = 2 * self.mask_embed_dim + self.pos_dim + 1
        priority_hidden = max(int(decoder_channels), 32)

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
        self.prev_scale = nn.Parameter(torch.tensor(1.0))

        if self.enable_patch_prioritization:
            self.feature_token_proj = nn.Linear(decoder_channels, self.mask_embed_dim)
            self.concept_proto_proj = nn.Linear(concept_dim, self.mask_embed_dim)
            self.unary_priority_mlp = nn.Sequential(
                nn.Linear(self.token_dim, priority_hidden),
                nn.ReLU(),
                # Softmax over candidates is shift-invariant, so a last-layer
                # bias cannot receive a useful gradient.
                nn.Linear(priority_hidden, 1, bias=False),
            )
            self.factor_query = nn.Sequential(
                nn.Linear(self.token_dim, priority_hidden),
                nn.ReLU(),
                nn.Linear(priority_hidden, self.factorized_rank, bias=False),
            )
            self.factor_key = nn.Sequential(
                nn.Linear(self.token_dim, priority_hidden),
                nn.ReLU(),
                nn.Linear(priority_hidden, self.factorized_rank, bias=False),
            )
            nn.init.normal_(self.factor_query[-1].weight, std=1e-3)
            nn.init.normal_(self.factor_key[-1].weight, std=1.2e-3)
            self.raw_spatial_frequency_weights = nn.Parameter(
                torch.zeros(2 * self.pos_num_frequencies)
            )
            # sigmoid(-2.1972246) ≈ 0.1; begins near content-only comparison.
            self.raw_spatial_strength = nn.Parameter(torch.tensor(-2.1972246))
            self.mask_feature_branch = Conv3DGNAct(
                decoder_channels,
                decoder_channels,
                kernel_size=1,
                padding=0,
            )
            self.null_prototype = nn.Parameter(torch.randn(self.concept_dim) * 0.02)
            self.raw_pairwise_strength = nn.Parameter(torch.tensor(0.0))
            # sigmoid(-2.197) ≈ 0.1; residual strength for
            # film + alpha * (priority_mask * branch(film)).
            self.raw_strength = nn.Parameter(torch.tensor(-2.197))
        else:
            self.feature_token_proj = None
            self.concept_proto_proj = None
            self.unary_priority_mlp = None
            self.factor_query = None
            self.factor_key = None
            self.mask_feature_branch = None
            self.null_prototype = None
            self.raw_pairwise_strength = None
            self.raw_strength = None
            self.raw_spatial_frequency_weights = None
            self.raw_spatial_strength = None

    @property
    def spatial_strength(self) -> torch.Tensor:
        if self.raw_spatial_strength is None:
            raise RuntimeError(
                "spatial_strength is only defined when patch prioritization is enabled"
            )
        return torch.sigmoid(self.raw_spatial_strength)

    @property
    def pairwise_strength(self) -> torch.Tensor:
        if self.raw_pairwise_strength is None:
            raise RuntimeError("pairwise_strength is only defined when patch prioritization is enabled")
        return F.softplus(self.raw_pairwise_strength)

    def _normalized_spatial_grid(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return grid_y, grid_x

    def _compute_soft_activations(
        self,
        feature_proj: torch.Tensor,
        active_prototypes: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Soft concept-patch activations ``a_ij`` with a null prototype in assignment."""
        B, _, T, H, W = feature_proj.shape
        K = int(active_prototypes.shape[1])
        dtype = feature_proj.dtype
        validity = validity_mask.bool()
        if validity.dim() != 2 or validity.shape[0] != B or validity.shape[1] != K:
            raise ValueError(
                f"validity_mask must be [B, K]=[{B}, {K}], got {tuple(validity.shape)}"
            )

        projected_features = self.feature_token_proj(
            feature_proj.permute(0, 2, 3, 4, 1)
        )
        projected_prototypes = self.concept_proto_proj(active_prototypes)
        feature_tokens = F.normalize(projected_features, dim=-1)
        proto_tokens = F.normalize(projected_prototypes, dim=-1)
        similarity_maps = torch.einsum(
            "bke,bthwe->btkhw",
            proto_tokens,
            feature_tokens,
        )

        valid_spatial = validity[:, None, :, None, None]
        assignment_logits = similarity_maps / self.assignment_temperature
        assignment_logits = assignment_logits.masked_fill(
            ~valid_spatial, _MASKED_LOGIT_FILL
        )
        if self.use_null_prototype:
            null_tokens = F.normalize(
                self.concept_proto_proj(
                    self.null_prototype.to(dtype=dtype).view(1, 1, -1)
                ),
                dim=-1,
            ).expand(B, 1, -1)
            null_similarity = torch.einsum(
                "bke,bthwe->btkhw",
                null_tokens,
                feature_tokens,
            )
            assignment_logits = torch.cat(
                [assignment_logits, null_similarity / self.assignment_temperature],
                dim=2,
            )
            assignment = torch.softmax(assignment_logits, dim=2)
            activations = assignment[:, :, :K]
        else:
            activations = torch.nan_to_num(
                torch.softmax(assignment_logits, dim=2),
                nan=0.0,
            )
        activations = activations * valid_spatial.to(dtype=activations.dtype)
        return {
            "similarity_maps": similarity_maps,
            "activations": activations,
            "validity_mask": validity,
            "prototypes": active_prototypes,
            "projected_features": projected_features,
            "projected_prototypes": projected_prototypes,
        }

    def _build_activation_tokens(
        self,
        projected_features: torch.Tensor,
        projected_prototypes: torch.Tensor,
        activations: torch.Tensor,
        concept_id_offset: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """One token per concept-patch pair; coordinates are in ``[0, 1]``."""
        B, T, H, W, embed_dim = projected_features.shape
        K = int(projected_prototypes.shape[1])
        dtype = projected_features.dtype
        device = projected_features.device
        N = K * H * W

        grid_y, grid_x = self._normalized_spatial_grid(
            H, W, device=device, dtype=dtype
        )
        pos_hw = _sinusoidal_2d_encoding(
            grid_x, grid_y, self.pos_num_frequencies
        )

        feat_flat = (
            projected_features.reshape(B, T, 1, H * W, embed_dim)
            .expand(-1, -1, K, -1, -1)
            .reshape(B, T, N, embed_dim)
        )
        proto_flat = (
            projected_prototypes[:, None, :, None, :]
            .expand(-1, T, -1, H * W, -1)
            .reshape(B, T, N, embed_dim)
        )
        pos_flat = (
            pos_hw.reshape(1, 1, 1, H * W, self.pos_dim)
            .expand(B, T, K, -1, -1)
            .reshape(B, T, N, self.pos_dim)
        )
        act_flat = activations.reshape(B, T, N, 1)
        tokens = torch.cat([feat_flat, proto_flat, pos_flat, act_flat], dim=-1)

        patch_index = torch.arange(H * W, device=device).repeat(K)
        concept_index = (
            torch.arange(K, device=device).repeat_interleave(H * W) + concept_id_offset
        )
        x_coords = grid_x.reshape(-1).repeat(K)
        y_coords = grid_y.reshape(-1).repeat(K)
        return {
            "tokens": tokens,
            "activations": act_flat.squeeze(-1),
            "patch_index": patch_index,
            "concept_index": concept_index,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "num_concepts": activations.new_tensor(K, dtype=torch.long),
            "spatial_hw": (H, W),
        }

    @property
    def factor_scale(self) -> float:
        return float(self.factorized_rank) ** -0.5

    def _build_spatial_factors(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build augmented spatial factors ``eta`` and normalized ``rho`` for ``[N]`` coords."""
        x = x_coords.to(device=device, dtype=dtype)
        y = y_coords.to(device=device, dtype=dtype)
        pos_encoding = _sinusoidal_2d_encoding(
            x, y, self.pos_num_frequencies
        )
        beta = F.softplus(self.raw_spatial_frequency_weights.to(dtype=dtype)) + _PRIORITY_EPS
        num_freq = self.pos_num_frequencies
        beta_x = beta[:num_freq]
        beta_y = beta[num_freq:]
        expanded_beta = torch.cat(
            [beta_x, beta_x, beta_y, beta_y],
            dim=-1,
        )
        weighted_pos = pos_encoding * expanded_beta.sqrt()
        weighted_pos = F.normalize(weighted_pos, dim=-1, eps=_PRIORITY_EPS)
        rho = torch.cat(
            [torch.ones_like(weighted_pos[..., :1]), weighted_pos],
            dim=-1,
        ) / math.sqrt(2.0)

        spatial_strength = self.spatial_strength.to(dtype=dtype, device=device)
        eta = torch.cat(
            [
                (1.0 - spatial_strength).clamp_min(_PRIORITY_EPS).sqrt().expand(
                    rho.shape[0], 1
                ),
                spatial_strength.clamp_min(0.0).sqrt() * rho,
            ],
            dim=-1,
        )
        return eta, rho

    def _spatial_relevance_for_pairs(
        self,
        eta: torch.Tensor,
        candidate_n: torch.Tensor,
        candidate_m: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the symmetric spatial kernel for selected candidate pairs only."""
        return torch.einsum("ns,ms->nm", eta[candidate_n], eta[candidate_m])

    def _spatial_diagnostic_stats(self, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        beta = F.softplus(self.raw_spatial_frequency_weights.to(dtype=dtype)) + _PRIORITY_EPS
        beta_norm = beta / beta.sum().clamp_min(_PRIORITY_EPS)
        entropy = -(beta_norm * beta_norm.log()).sum()
        return {
            "spatial_strength": self.spatial_strength.to(dtype=dtype).detach(),
            "spatial_frequency_weights": beta.detach(),
            "spatial_frequency_weight_entropy": entropy.detach(),
        }

    def _factorized_comparison_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        context: torch.Tensor,
        weights: torch.Tensor,
        total_weight: torch.Tensor,
        candidate_valid: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Detached diagnostics for low-rank factorized antisymmetric comparison."""
        valid = candidate_valid.unsqueeze(1).expand_as(context)
        ctx = context.masked_fill(~valid, float("nan")).reshape(-1)
        finite_ctx = ctx[torch.isfinite(ctx)]
        zero = context.new_zeros(())
        q_norm = q.norm(dim=-1).masked_fill(~valid, float("nan"))
        k_norm = k.norm(dim=-1).masked_fill(~valid, float("nan"))
        if finite_ctx.numel() == 0:
            ctx_mean = ctx_std = ctx_min = ctx_max = zero
        else:
            ctx_mean = finite_ctx.mean()
            ctx_std = finite_ctx.std(unbiased=False)
            ctx_min = finite_ctx.min()
            ctx_max = finite_ctx.max()
        stats = {
            "query_norm_mean": q_norm.nanmean().detach(),
            "key_norm_mean": k_norm.nanmean().detach(),
            "context_mean": ctx_mean.detach(),
            "context_std": ctx_std.detach(),
            "context_min": ctx_min.detach(),
            "context_max": ctx_max.detach(),
            "total_candidate_weight": total_weight.mean().detach(),
            "factorized_rank": context.new_tensor(float(self.factorized_rank)),
        }
        stats.update(self._spatial_diagnostic_stats(dtype))
        return stats

    def _factorized_pairwise_context(
        self,
        tokens: torch.Tensor,
        activations: torch.Tensor,
        candidate_valid: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Low-rank factorized antisymmetric concept-patch comparison with spatial kernel.

        For each candidate ``n``, aggregates competitor scores
        ``g_{nm} * (dot(q_n, k_m) - dot(q_m, k_n)) / sqrt(R)`` weighted by
        nonnegative activations over ``m != n``, where ``g_{nm} = eta_n^T eta_m``
        is the explicit symmetric spatial relevance kernel. Self terms cancel in
        the numerator, so the aggregate is computed exactly from augmented
        global query/key summaries in ``O(B T N R S)`` time and memory.
        """
        valid = candidate_valid[:, None, :].to(dtype=tokens.dtype)
        weights = activations.clamp_min(0.0) * valid

        q = self.factor_query(tokens)
        k = self.factor_key(tokens)
        eta, _ = self._build_spatial_factors(
            x_coords,
            y_coords,
            dtype=tokens.dtype,
            device=tokens.device,
        )

        q_global = torch.einsum("btn,btnr,ns->btrs", weights, q, eta)
        k_global = torch.einsum("btn,btnr,ns->btrs", weights, k, eta)
        total_weight = weights.sum(dim=2)

        q_against_k = torch.einsum("btnr,ns,btrs->btn", q, eta, k_global)
        q_against_q = torch.einsum("btrs,btnr,ns->btn", q_global, k, eta)
        numerator = (q_against_k - q_against_q) / math.sqrt(self.factorized_rank)
        denominator = total_weight.unsqueeze(-1) - weights
        context = numerator / denominator.clamp_min(_PRIORITY_EPS)
        context = torch.where(
            denominator > _PRIORITY_EPS,
            context,
            torch.zeros_like(context),
        )
        context = context * valid
        stats = self._factorized_comparison_stats(
            q,
            k,
            context,
            weights,
            total_weight,
            candidate_valid,
            tokens.dtype,
        )
        return context, stats

    def _prioritize_concept_patch_activations(
        self,
        modality_outs: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Score every valid concept-patch activation independently per frame."""
        if not modality_outs:
            raise ValueError("at least one concept modality is required")

        token_parts: List[Dict[str, torch.Tensor]] = []
        concept_offset = 0
        activation_maps: List[torch.Tensor] = []
        validity_masks: List[torch.Tensor] = []
        for modality in modality_outs:
            H, W = modality["activations"].shape[-2:]
            part = self._build_activation_tokens(
                modality["projected_features"],
                modality["projected_prototypes"],
                modality["activations"],
                concept_id_offset=concept_offset,
            )
            token_parts.append(part)
            activation_maps.append(modality["activations"])
            validity_masks.append(modality["validity_mask"])
            concept_offset += int(modality["activations"].shape[2])

        tokens = torch.cat([part["tokens"] for part in token_parts], dim=2)
        activations = torch.cat([part["activations"] for part in token_parts], dim=2)
        x_coords = torch.cat([part["x_coords"] for part in token_parts], dim=0)
        y_coords = torch.cat([part["y_coords"] for part in token_parts], dim=0)
        validity = torch.cat(validity_masks, dim=1)
        B, T, N, _ = tokens.shape
        H, W = token_parts[0]["spatial_hw"]
        K = N // (H * W)
        candidate_valid = (
            validity[:, :, None]
            .expand(-1, -1, H * W)
            .reshape(B, N)
        )

        unary = self.unary_priority_mlp(tokens).squeeze(-1)
        context, factorized_stats = self._factorized_pairwise_context(
            tokens,
            activations,
            candidate_valid,
            x_coords,
            y_coords,
        )
        scores = unary + self.pairwise_strength * context
        priority_logits = (
            scores / self.priority_temperature
            + torch.log(activations.clamp_min(0.0) + _PRIORITY_EPS)
        )
        concept_priorities_flat = _masked_softmax(
            priority_logits,
            candidate_valid,
            dim=-1,
        )
        if not bool(validity.any(dim=-1).all().item()):
            raise ValueError(
                "Priority softmax requires at least one valid concept prototype "
                "for every batch item."
            )

        concept_priorities = concept_priorities_flat.view(B, T, K, H, W)
        patch_priority_map = concept_priorities.sum(dim=2)
        # [B, 1, T, H, W] broadcasts over channels of [B, C, T, H, W] FiLM features.
        priority_mask = patch_priority_map.unsqueeze(1)
        return {
            "priority_mask": priority_mask,
            "concept_priorities": concept_priorities,
            "concept_validity": validity,
            "patch_priority_map": patch_priority_map,
            "unary_scores": unary.view(B, T, K, H, W),
            "pairwise_context": context.view(B, T, K, H, W),
            "factorized_comparison_stats": factorized_stats,
        }

    def _assert_required_visual_prototypes(
        self,
        visual_validity_mask: torch.Tensor,
    ) -> None:
        validity = visual_validity_mask.bool()
        if validity.numel() == 0 or not bool(validity.any(dim=-1).all().item()):
            raise ValueError(
                "Visual concept activations are required for mask construction, but "
                "at least one batch item has no valid visual prototypes."
            )

    def forward(
        self,
        features: torch.Tensor,
        concept_volume: torch.Tensor,
        prev_decoder: Optional[torch.Tensor] = None,
        *,
        active_visual_prototypes: torch.Tensor,
        visual_validity_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feature_proj = self.feature_proj(features)
        concept_proj = self.concept_proj(concept_volume)

        prev_up = None
        if prev_decoder is not None:
            prev_up = self.prev_upsample(
                prev_decoder,
                target_thw=feature_proj.shape[-3:],
            )
            prev_up = self.prev_proj(prev_up)

        film_params = self.film(concept_proj)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * beta
        film_features = feature_proj * gamma + beta

        mask_outputs: Dict[str, torch.Tensor] = {}
        if self.enable_patch_prioritization:
            self._assert_required_visual_prototypes(visual_validity_mask)
            visual_out = self._compute_soft_activations(
                film_features,
                active_visual_prototypes,
                visual_validity_mask,
            )
            priority_out = self._prioritize_concept_patch_activations([visual_out])
            priority_mask = priority_out["priority_mask"]
            if priority_mask.shape[1] != 1:
                raise ValueError(
                    "patch-priority mask must have a singleton channel dimension "
                    f"[B, 1, T, H, W], got {tuple(priority_mask.shape)}"
                )
            alpha = self.max_strength * torch.sigmoid(self.raw_strength)
            branched = self.mask_feature_branch(film_features)
            feature_proj = film_features + alpha * (priority_mask * branched)
            mask_outputs.update(
                {
                    "priority_mask": priority_mask,
                    "concept_priorities": priority_out["concept_priorities"],
                    "concept_validity": priority_out["concept_validity"],
                    "patch_priority_map": priority_out["patch_priority_map"],
                    "unary_scores": priority_out["unary_scores"],
                    "pairwise_context": priority_out["pairwise_context"],
                    "factorized_comparison_stats": priority_out[
                        "factorized_comparison_stats"
                    ],
                    "visual_similarity_maps": visual_out["similarity_maps"],
                    "concept_patch_activations": visual_out["activations"],
                    "residual_strength": alpha.detach(),
                }
            )
        else:
            feature_proj = film_features

        fused = feature_proj
        if prev_up is not None:
            fused = fused + self.prev_scale * prev_up

        decoded = self.refine2(self.refine1(fused))
        return decoded, mask_outputs


class ConceptGatedMultiScaleSaliencyDecoder(nn.Module):
    """
    Coarse-to-fine spatiotemporal decoder with concept-patch prioritization at
    stages 3 and 4.

    Stages decode stage4 -> stage1. Each stage fuses full [B, C, T, H, W]
    feature and concept volumes through FiLM. Stages 3 and 4 score every valid
    concept-patch activation, update FiLM features with a 1x1x1 conv-GN-ReLU
    branch, and apply the residual patch-priority mask to that update. Temporal
    aggregation and learned upsampling produce the last-frame saliency map.
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
        assignment_temperature: float = 0.07,
        priority_temperature: float = 0.1,
        factorized_rank: int = 32,
        distance_gate_range: float = 0.5,
    ):
        super().__init__()
        del feature_residual_scale, tau_pi, distance_gate_range

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
        self.patch_priority_stages = tuple(
            stage
            for stage in _PATCH_PRIORITY_STAGES
            if stage in self.stage_channels
        )

        hidden_channels = max(decoder_channels // 2, 32)
        attn_hidden = max(decoder_channels // 2, 1)

        self.fusion_blocks = nn.ModuleDict(
            {
                stage: SpatioTemporalConceptGatedFusionBlock(
                    feature_channels=channels,
                    concept_dim=concept_dim,
                    decoder_channels=decoder_channels,
                    dropout=dropout,
                    assignment_temperature=assignment_temperature,
                    priority_temperature=priority_temperature,
                    enable_patch_prioritization=stage in self.patch_priority_stages,
                    factorized_rank=factorized_rank,
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
        stage_concept_priorities: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_concept_validity: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_visual_similarity_maps: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_concept_patch_activations: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_unary_scores: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_pairwise_context: Optional[Dict[str, torch.Tensor]] = (
            {} if return_details else None
        )
        stage_factorized_comparison_stats: Optional[Dict[str, Dict[str, float]]] = (
            {} if return_details else None
        )
        stage_patch_priority_maps: Dict[str, torch.Tensor] = {}
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

            decoded, mask_outputs = fusion_blocks[stage](
                features_5d,
                concept_volume,
                prev_decoder=prev_decoder,
                active_visual_prototypes=active_visual_prototypes,
                visual_validity_mask=visual_validity_mask,
            )
            if "patch_priority_map" in mask_outputs:
                stage_patch_priority_maps[stage] = mask_outputs["patch_priority_map"]
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
                decoded_stage_volumes[stage] = decoded.detach()
                priority_mask = mask_outputs.get("priority_mask")
                if torch.is_tensor(priority_mask):
                    stage_concept_masks[stage] = priority_mask.detach()
                    stage_gates[stage] = priority_mask.detach()
                if "concept_priorities" in mask_outputs:
                    stage_concept_priorities[stage] = mask_outputs[
                        "concept_priorities"
                    ].detach()
                if "concept_validity" in mask_outputs:
                    stage_concept_validity[stage] = mask_outputs[
                        "concept_validity"
                    ].detach()
                if "visual_similarity_maps" in mask_outputs:
                    stage_visual_similarity_maps[stage] = mask_outputs[
                        "visual_similarity_maps"
                    ].detach()
                if "concept_patch_activations" in mask_outputs:
                    stage_concept_patch_activations[stage] = mask_outputs[
                        "concept_patch_activations"
                    ].detach()
                if "unary_scores" in mask_outputs:
                    stage_unary_scores[stage] = mask_outputs["unary_scores"].detach()
                if "pairwise_context" in mask_outputs:
                    stage_pairwise_context[stage] = mask_outputs[
                        "pairwise_context"
                    ].detach()
                factorized_stats = mask_outputs.get("factorized_comparison_stats")
                if isinstance(factorized_stats, dict):
                    stage_factorized_comparison_stats[stage] = {}
                    for key, value in factorized_stats.items():
                        if torch.is_tensor(value):
                            if value.numel() == 1:
                                stage_factorized_comparison_stats[stage][key] = float(
                                    value.detach().cpu()
                                )
                            elif key == "spatial_frequency_weights":
                                stage_factorized_comparison_stats[stage][key] = float(
                                    value.detach().mean().cpu()
                                )
                            else:
                                stage_factorized_comparison_stats[stage][key] = float(
                                    value.detach().mean().cpu()
                                )
                        else:
                            stage_factorized_comparison_stats[stage][key] = float(value)
                with torch.no_grad():
                    raw_strength = getattr(fusion_blocks[stage], "raw_strength", None)
                    if torch.is_tensor(priority_mask) and raw_strength is not None:
                        mask_strength = float(
                            (
                                fusion_blocks[stage].max_strength
                                * torch.sigmoid(raw_strength)
                            )
                            .detach()
                            .cpu()
                        )
                        stage_mask_diagnostics[stage] = {
                            "patch_priority_mean": float(
                                priority_mask.detach().mean().cpu()
                            ),
                            "patch_priority_std": float(
                                priority_mask.detach().std().cpu()
                            ),
                            "patch_priority_min": float(
                                priority_mask.detach().min().cpu()
                            ),
                            "patch_priority_max": float(
                                priority_mask.detach().max().cpu()
                            ),
                            "residual_strength": mask_strength,
                        }

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
            "stage_patch_priority_maps": stage_patch_priority_maps,
        }

        if return_details:
            out["stage_concept_volumes"] = stage_concept_volumes
            out["stage_feature_volumes"] = stage_feature_volumes
            out["stage_gates"] = stage_gates
            out["stage_priority_masks"] = stage_concept_masks
            out["stage_mask_diagnostics"] = stage_mask_diagnostics
            out["stage_concept_priorities"] = stage_concept_priorities
            out["stage_concept_validity"] = stage_concept_validity
            out["stage_visual_similarity_maps"] = stage_visual_similarity_maps
            out["stage_concept_patch_activations"] = stage_concept_patch_activations
            out["stage_unary_scores"] = stage_unary_scores
            out["stage_pairwise_context"] = stage_pairwise_context
            out["stage_factorized_comparison_stats"] = stage_factorized_comparison_stats
            out["decoded_stage_volumes"] = decoded_stage_volumes

        return out