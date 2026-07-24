"""
Frozen Video Swin-T backbone for intermediate spatiotemporal features.

Input windows from the dataloader are typically [B, T, C, H, W] (BTCHW).
TorchVision ``swin3d_t`` expects [B, C, T, H, W] (BCTHW) internally.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import Swin3D_T_Weights, swin3d_t


class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer feature extractor (no classification head).

    Returns multi-scale features in layout ``output_format`` (default BCTHW):
      - stage1:  96 channels
      - stage2: 192 channels
      - stage3: 384 channels
      - stage4: 768 channels
    """

    STAGE_INDICES = {
        "stage1": 0,
        "stage2": 2,
        "stage3": 4,
        "stage4": 6,
    }

    FEATURE_CHANNELS = {
        "stage1": 96,
        "stage2": 192,
        "stage3": 384,
        "stage4": 768,
    }

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        return_stages: Sequence[str] = ("stage2", "stage3", "stage4"),
        input_format: str = "BTCHW",
        output_format: str = "BCTHW",
        resize_to: Optional[Union[int, Tuple[int, int]]] = None,
        normalize: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if input_format not in ("BTCHW", "BCTHW"):
            raise ValueError("input_format must be 'BTCHW' or 'BCTHW'")
        if output_format not in ("BCTHW", "BTHWC"):
            raise ValueError("output_format must be 'BCTHW' or 'BTHWC'")

        for stage in return_stages:
            if stage not in self.STAGE_INDICES:
                raise ValueError(
                    f"Unknown stage '{stage}'. Choose from {list(self.STAGE_INDICES)}"
                )

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.return_stages = tuple(return_stages)
        self.input_format = input_format
        self.output_format = output_format
        self.resize_to = resize_to
        self.normalize = normalize
        self.gradient_checkpointing = bool(gradient_checkpointing)

        weights = Swin3D_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin3d_t(weights=weights)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self._requested_stage_indices = frozenset(
            self.STAGE_INDICES[s] for s in self.return_stages
        )
        self._index_to_stage = {
            idx: name for name, idx in self.STAGE_INDICES.items()
        }

    def get_feature_channels(self) -> Dict[str, int]:
        """Per-stage channel counts for ``swin3d_t``."""
        return dict(self.FEATURE_CHANNELS)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize layout and value range to [B, C, T, H, W] for the backbone.

        Args:
            x: [B, T, C, H, W] if input_format='BTCHW', else [B, C, T, H, W].

        Returns:
            x: [B, C, T, H, W], float32, optionally resized and ImageNet-normalized.
        """
        x = x.float()
        if x.numel() > 0 and x.max() > 2.0:
            x = x / 255.0

        if self.input_format == "BTCHW":
            # [B, T, C, H, W] -> [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.resize_to is not None:
            if isinstance(self.resize_to, int):
                size = (x.shape[2], self.resize_to, self.resize_to)
            else:
                size = (x.shape[2], int(self.resize_to[0]), int(self.resize_to[1]))
            # Trilinear resize: spatial only (T unchanged).
            x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        if self.normalize:
            x = (x - self.mean) / self.std

        return x

    def _to_output_format(self, x: torch.Tensor) -> torch.Tensor:
        """Convert backbone tensor [B, T, H, W, C] to requested output layout."""
        if self.output_format == "BCTHW":
            # [B, T, H, W, C] -> [B, C, T, H, W]
            return x.permute(0, 4, 1, 2, 3).contiguous()
        # BTHWC
        return x.contiguous()

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate Video Swin features.

        Args:
            x: Video clip; layout controlled by ``input_format``.

        Returns:
            Dict with keys among stage1..stage4 (only ``return_stages``).
            Each value is [B, C, T, H, W] or [B, T, H, W, C] per ``output_format``.
        """
        if self.freeze_backbone:
            self.backbone.eval()

        x = self._prepare_input(x)

        # TorchVision Video Swin uses [B, T, H, W, C] inside the feature stack.
        x = self.backbone.patch_embed(x)
        x = self.backbone.pos_drop(x)

        outputs: Dict[str, torch.Tensor] = {}
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and not self.freeze_backbone
        )
        for i, layer in enumerate(self.backbone.features):
            if use_checkpoint:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if i in self._requested_stage_indices:
                stage_name = self._index_to_stage[i]
                outputs[stage_name] = self._to_output_format(x)

        return outputs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for ``forward_features`` (no classification logits)."""
        return self.forward_features(x)


class VideoSwin(nn.Module):
    """
    Backward-compatible wrapper: returns stage4 features by default.

    For multi-scale features, use ``VideoSwinTransformer`` directly.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
        super().__init__()
        self.extractor = VideoSwinTransformer(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            return_stages=("stage4",),
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return stage4 features [B, C, T, H, W] (or BTHWC if configured)."""
        return self.extractor(x)["stage4"]

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.extractor.forward_features(x)

    def get_feature_channels(self) -> Dict[str, int]:
        return self.extractor.get_feature_channels()
