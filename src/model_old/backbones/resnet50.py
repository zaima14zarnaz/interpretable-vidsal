"""
Frozen ResNet-50 backbone for intermediate spatiotemporal features.

Video windows are processed frame-wise with a shared 2D ResNet-50. Each stage
tensor is stacked back to [B, C, T, H, W] (BCTHW) for ConceptCreation and the
multi-scale saliency decoder.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 feature extractor (no classification head).

    Returns multi-scale features in layout ``output_format`` (default BCTHW):
      - stage1:  256 channels  (after layer1)
      - stage2:  512 channels  (after layer2)
      - stage3: 1024 channels  (after layer3)
      - stage4: 2048 channels  (after layer4)
    """

    STAGE_MODULES = {
        "stage1": "layer1",
        "stage2": "layer2",
        "stage3": "layer3",
        "stage4": "layer4",
    }

    FEATURE_CHANNELS = {
        "stage1": 256,
        "stage2": 512,
        "stage3": 1024,
        "stage4": 2048,
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
            if stage not in self.STAGE_MODULES:
                raise ValueError(
                    f"Unknown stage '{stage}'. Choose from {list(self.STAGE_MODULES)}"
                )

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.return_stages = tuple(return_stages)
        self.input_format = input_format
        self.output_format = output_format
        self.resize_to = resize_to
        self.normalize = normalize
        self.gradient_checkpointing = bool(gradient_checkpointing)

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def get_feature_channels(self) -> Dict[str, int]:
        """Per-stage channel counts for ResNet-50."""
        return dict(self.FEATURE_CHANNELS)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize layout and value range to [B, C, T, H, W].

        Args:
            x: [B, T, C, H, W] if input_format='BTCHW', else [B, C, T, H, W].

        Returns:
            x: [B, C, T, H, W], float32, optionally resized and ImageNet-normalized.
        """
        x = x.float()
        if x.numel() > 0 and x.max() > 2.0:
            x = x / 255.0

        if self.input_format == "BTCHW":
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.resize_to is not None:
            b, c, t, h, w = x.shape
            if isinstance(self.resize_to, int):
                size = (self.resize_to, self.resize_to)
            else:
                size = (int(self.resize_to[0]), int(self.resize_to[1]))
            frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            frames = F.interpolate(
                frames,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            x = frames.view(b, t, c, size[0], size[1]).permute(0, 2, 1, 3, 4)

        if self.normalize:
            x = (x - self.mean.unsqueeze(2)) / self.std.unsqueeze(2)

        return x

    def _frames_to_video(
        self,
        frames: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Reshape [B*T, C, H, W] to requested video layout."""
        _, c, h, w = frames.shape
        video = frames.view(batch_size, num_frames, c, h, w)
        if self.output_format == "BCTHW":
            return video.permute(0, 2, 1, 3, 4).contiguous()
        return video.contiguous()

    def _run_stage(
        self,
        module: nn.Module,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        use_checkpoint = (
            self.gradient_checkpointing
            and self.training
            and not self.freeze_backbone
        )
        if use_checkpoint:
            return checkpoint(module, frames, use_reentrant=False)
        return module(frames)

    def _stem(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(frames)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        return x

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate ResNet-50 features for each video frame.

        Args:
            x: Video clip; layout controlled by ``input_format``.

        Returns:
            Dict with keys among stage1..stage4 (only ``return_stages``).
            Each value is [B, C, T, H, W] or [B, T, H, W, C] per ``output_format``.
        """
        if self.freeze_backbone:
            self.backbone.eval()

        x = self._prepare_input(x)
        batch_size, _, num_frames, _, _ = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames,
            x.shape[1],
            x.shape[3],
            x.shape[4],
        )

        outputs: Dict[str, torch.Tensor] = {}
        feat = self._stem(frames)
        for stage_name in ("stage1", "stage2", "stage3", "stage4"):
            module = getattr(self.backbone, self.STAGE_MODULES[stage_name])
            feat = self._run_stage(module, feat)
            if stage_name in self.return_stages:
                outputs[stage_name] = self._frames_to_video(
                    feat,
                    batch_size,
                    num_frames,
                )

        return outputs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for ``forward_features`` (no classification logits)."""
        return self.forward_features(x)


class ResNet50(nn.Module):
    """
    Backward-compatible wrapper: returns stage4 features by default.

    For multi-scale features, use ``ResNet50Backbone`` directly.
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
        super().__init__()
        self.extractor = ResNet50Backbone(
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
