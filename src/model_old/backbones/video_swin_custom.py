"""
HSFI-Net Video Swin wrapper.

This replaces the TorchVision ``swin3d_t`` backbone with the original/custom
Video Swin implementation used by HSFI-Net.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


from .video_swin_hsfi import SwinTransformer3D


class VideoSwinTransformer(nn.Module):
    """
    Original/custom Video Swin feature extractor used by HSFI-Net.

    Returns multi-scale features in layout ``output_format`` (default BCTHW):
      - stage1:  96 channels
      - stage2: 192 channels
      - stage3: 384 channels
      - stage4: 768 channels
    """

    DEFAULT_PRETRAINED_PATH = (
        "/data/quantization/zaima/videoswin/"
        "swin_small_patch244_window877_kinetics400_1k.pth"
    )

    FEATURE_CHANNELS = {
        "stage1": 96,
        "stage2": 192,
        "stage3": 384,
        "stage4": 768,
    }

    STAGE_NAMES = ("stage1", "stage2", "stage3", "stage4")

    def __init__(
        self,
        pretrained: Union[bool, str, None] = True,
        pretrained_path: Optional[str] = None,
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
            if stage not in self.FEATURE_CHANNELS:
                raise ValueError(
                    f"Unknown stage '{stage}'. Choose from {list(self.FEATURE_CHANNELS)}"
                )

        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.freeze_backbone = bool(freeze_backbone)
        self.return_stages = tuple(return_stages)
        self.input_format = input_format
        self.output_format = output_format
        self.resize_to = resize_to
        self.normalize = bool(normalize)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        checkpoint_path = self._resolve_pretrained_path(pretrained, pretrained_path)

        self.backbone = SwinTransformer3D(
            pretrained=checkpoint_path,
            patch_size=(2, 4, 4),
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=False,
            frozen_stages=-1,
            use_checkpoint=self.gradient_checkpointing,
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    @classmethod
    def _resolve_pretrained_path(
        cls,
        pretrained: Union[bool, str, None],
        pretrained_path: Optional[str],
    ) -> Optional[str]:
        if isinstance(pretrained, str):
            return pretrained
        if pretrained_path is not None:
            return pretrained_path
        if pretrained is True:
            return cls.DEFAULT_PRETRAINED_PATH
        if pretrained in (False, None):
            return None
        raise TypeError("pretrained must be a bool, str path, or None")

    def get_feature_channels(self) -> Dict[str, int]:
        return dict(self.FEATURE_CHANNELS)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input to [B, C, T, H, W] for the custom HSFI Video Swin.
        """
        x = x.float()
        if x.numel() > 0 and x.max() > 2.0:
            x = x / 255.0

        if self.input_format == "BTCHW":
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.resize_to is not None:
            if isinstance(self.resize_to, int):
                size = (x.shape[2], self.resize_to, self.resize_to)
            else:
                size = (x.shape[2], int(self.resize_to[0]), int(self.resize_to[1]))
            x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        if self.normalize:
            x = (x - self.mean) / self.std

        return x

    def _to_output_format(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_format == "BCTHW":
            return x.contiguous()
        return x.permute(0, 2, 3, 4, 1).contiguous()

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.freeze_backbone:
            self.backbone.eval()

        x = self._prepare_input(x)
        stage_outputs = self.backbone(x)

        outputs: Dict[str, torch.Tensor] = {}
        for stage_name, stage_feature in zip(self.STAGE_NAMES, stage_outputs):
            if stage_name in self.return_stages:
                outputs[stage_name] = self._to_output_format(stage_feature)
        return outputs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward_features(x)


class VideoSwin(nn.Module):
    """
    Backward-compatible wrapper: returns stage4 features by default.
    """

    DEFAULT_PRETRAINED_PATH = VideoSwinTransformer.DEFAULT_PRETRAINED_PATH

    def __init__(
        self,
        pretrained: Union[bool, str, None] = True,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.extractor = VideoSwinTransformer(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            return_stages=("stage4",),
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)["stage4"]

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.extractor.forward_features(x)

    def get_feature_channels(self) -> Dict[str, int]:
        return self.extractor.get_feature_channels()

