"""
Factorized 3D spatiotemporal feature infusion.

Input/output shape:
    [B, C, T, H, W] -> [B, C, T, H, W]

Each stage tensor is projected into a 3D working space, processed by stacked
factorized temporal/spatial depthwise Conv3d residual blocks, then projected
back with a near-zero initialized residual gate.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_num_groups(channels: int, preferred: int = 8) -> int:
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class Factorized3DResBlock(nn.Module):
    """Factorized 3D residual block with temporal then spatial depthwise conv."""

    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: float = 4.0,
        num_groups: int = 8,
        layer_scale_init: float = 1e-3,
        dropout: float = 0.0,
    ):
        super().__init__()

        groups = _pick_num_groups(dim, num_groups)
        self.temporal_dw = nn.Conv3d(
            dim,
            dim,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=dim,
            bias=False,
        )
        self.spatial_dw = nn.Conv3d(
            dim,
            dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=dim,
            bias=False,
        )
        self.pointwise = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.GELU()

        hidden = max(int(dim * mlp_ratio), dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv3d(hidden, dim, kernel_size=1, bias=True),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, dim, 1, 1, 1), float(layer_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.temporal_dw(x)
        y = self.spatial_dw(y)
        y = self.pointwise(y)
        y = self.norm(y)
        y = self.act(y)
        y = self.ffn(y)
        return residual + self.layer_scale * y


class SpatioTemporal3DFeatureInfusion(nn.Module):
    """
    Factorized 3D spatiotemporal residual infusion for stage features.

    Args:
        in_channels: stage feature channels.
        temporal_dim: hidden dimension inside the 3D residual stack.
        num_layers: number of Factorized3DResBlock layers.
        mlp_ratio: FFN expansion ratio inside each block.
        dropout: dropout inside block FFN.
        use_temporal_difference: concatenate temporal deltas before blocks.
        residual_init: initial residual gate (near zero for identity start).
        enhance_last_only: if True, only the last temporal slice is updated.
        num_groups: preferred GroupNorm group count.
        layer_scale_init: initial per-block layer scale.
    """

    def __init__(
        self,
        in_channels: int,
        temporal_dim: int = 256,
        num_layers: int = 2,
        num_blocks: Optional[int] = None,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_temporal_len: int = 32,
        use_temporal_difference: bool = True,
        residual_init: Optional[float] = None,
        residual_gate_init: float = -2.0,
        enhance_last_only: bool = False,
        num_groups: int = 8,
        layer_scale_init: float = 1e-3,
        **_deprecated_kwargs: object,
    ):
        super().__init__()
        del num_heads, max_temporal_len, _deprecated_kwargs

        if num_blocks is not None:
            num_layers = int(num_blocks)
        if residual_init is not None:
            gate_init = float(residual_init)
        else:
            gate_init = float(residual_gate_init)

        self.in_channels = int(in_channels)
        self.temporal_dim = int(temporal_dim)
        self.use_temporal_difference = bool(use_temporal_difference)
        self.enhance_last_only = bool(enhance_last_only)

        self.input_proj = nn.Conv3d(
            in_channels,
            temporal_dim,
            kernel_size=1,
            bias=True,
        )

        if self.use_temporal_difference:
            self.delta_proj = nn.Conv3d(
                temporal_dim * 2,
                temporal_dim,
                kernel_size=1,
                bias=True,
            )
        else:
            self.delta_proj = None

        self.blocks = nn.ModuleList(
            [
                Factorized3DResBlock(
                    temporal_dim,
                    mlp_ratio=mlp_ratio,
                    num_groups=num_groups,
                    layer_scale_init=layer_scale_init,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

        self.output_proj = nn.Conv3d(
            temporal_dim,
            in_channels,
            kernel_size=1,
            bias=True,
        )
        self.residual_gate = nn.Parameter(torch.tensor(gate_init))
        self._last_diagnostics: Dict[str, float] = {}

    def get_last_diagnostics(self) -> Dict[str, float]:
        """Return detached diagnostics from the most recent forward pass."""
        return dict(self._last_diagnostics)

    def _apply_temporal_difference(self, x: torch.Tensor) -> torch.Tensor:
        if self.delta_proj is None:
            return x

        delta = torch.zeros_like(x)
        if x.shape[2] > 1:
            delta[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        return self.delta_proj(torch.cat([x, delta], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, T, H, W]

        Returns:
            enhanced_features: [B, C, T, H, W]
        """
        if features.dim() != 5:
            raise ValueError(
                f"Expected features [B,C,T,H,W], got {tuple(features.shape)}"
            )

        B, C, T, H, W = features.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {C}"
            )

        x = self.input_proj(features)
        x = self._apply_temporal_difference(x)

        for block in self.blocks:
            x = block(x)

        temporal_residual = self.output_proj(x)

        gate = torch.sigmoid(self.residual_gate)

        if self.enhance_last_only:
            out = features.clone()
            out[:, :, -1:] = (
                features[:, :, -1:]
                + gate * temporal_residual[:, :, -1:]
            )
        else:
            out = features + gate * temporal_residual

        with torch.no_grad():
            input_abs_mean = features.detach().abs().mean().item()
            temporal_residual_abs_mean = temporal_residual.detach().abs().mean().item()
            enhanced_delta_abs_mean = (out.detach() - features.detach()).abs().mean().item()
            self._last_diagnostics = {
                "temporal_gate": gate.detach().item(),
                "temporal_residual_abs_mean": temporal_residual_abs_mean,
                "input_abs_mean": input_abs_mean,
                "residual_to_input_ratio": temporal_residual_abs_mean
                / (input_abs_mean + 1e-6),
                "enhanced_delta_abs_mean": enhanced_delta_abs_mean,
                "enhanced_delta_to_input_ratio": enhanced_delta_abs_mean
                / (input_abs_mean + 1e-6),
            }

        return out


SameLocationPatchTemporalTransformer = SpatioTemporal3DFeatureInfusion
