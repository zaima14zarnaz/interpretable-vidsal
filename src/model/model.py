"""Explainable video saliency: model, fixed references, cache, and diagnostics.

All objective calculations live in losses.py. The custom Video Swin backbone
remains an external import. See README.md for installation and migration.
"""

from __future__ import annotations
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones.video_swin_custom import VideoSwinTransformer

if __package__:
    from .losses import VisualConceptLossMixin
else:
    from losses import VisualConceptLossMixin


# End-to-end model


class ExplainableVidSalModel(nn.Module):
    """Explainable video saliency for the last frame in a window."""

    def __init__(
        self,
        backbone_stage: str = "stage2",
        backbone_stages: Optional[Tuple[str, ...]] = (
            "stage1",
            "stage2",
            "stage3",
            "stage4",
        ),
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        backbone_gradient_checkpointing: bool = False,
        backbone_normalize: bool = True,
        input_format: str = "BTCHW",
        resize_to: Union[int, Tuple[int, int]] = (224, 384),
        concept_dim: int = 256,
        num_concepts: int = 32,
        concept_hidden_dim: int = 512,
        saliency_hidden_dim: int = 96,
        top_k: int = 9,
        output_activation: str = "sigmoid",
        return_details: bool = False,
        visual_concept_residual_weight: float = 0.1,
        visual_assignment_mode: str = "straight_through",
        visual_assignment_temperature: float = 0.07,
        visual_entropy_weight: float = 0.01,
        visual_usage_weight: float = 0.02,
        use_visual_saliency_alignment: bool = True,
        visual_saliency_align_weight: float = 0.05,
        allow_eval_concept_losses: bool = False,
        temporal_dim: Optional[int] = None,
        temporal_dropout: float = 0.0,
        temporal_use_difference: bool = True,
        temporal_num_blocks: int = 2,
        temporal_mlp_ratio: float = 2.0,
        temporal_residual_gate_init: float = -2.0,
        temporal_enhance_last_only: bool = False,
        decoder_use_side_logit_fusion: bool = True,
        use_temporal_feature_infusion: bool = True,
        use_shared_concept_activations: bool = False,
        prototype_bottleneck_strength: float = 0.4,
        fine_unary_mask_strength: float = 0.6,
        stage2_backbone_fusion_enabled: bool = True,
        visual_cluster_dir: Optional[str] = None,
        visual_cluster_max_patches_per_prototype: int = 0,
        visual_cluster_seed: int = 42,
        visual_cluster_expected_checkpoint_sha256: Optional[str] = None,
        visual_appearance_patch_size: int = 16,
    ):
        super().__init__()
        if input_format not in ("BTCHW", "BCTHW"):
            raise ValueError("input_format must be 'BTCHW' or 'BCTHW'")
        if backbone_stages is None:
            backbone_stages = (backbone_stage,)
        else:
            backbone_stages = tuple(backbone_stages)
        self.backbone_stages = backbone_stages
        self.input_format = input_format
        self.return_details = return_details
        self._backbone_frozen = freeze_backbone
        self.allow_eval_concept_losses = bool(allow_eval_concept_losses)
        if temporal_dim is None:
            temporal_dim = concept_dim
        self.temporal_dim = int(temporal_dim)
        self.use_temporal_feature_infusion = bool(use_temporal_feature_infusion)
        self.visual_cluster_dir = visual_cluster_dir
        self.visual_cluster_max_patches_per_prototype = int(
            visual_cluster_max_patches_per_prototype
        )
        self.visual_cluster_seed = int(visual_cluster_seed)
        self.visual_cluster_expected_checkpoint_sha256 = (
            visual_cluster_expected_checkpoint_sha256
        )
        self.visual_appearance_patch_size = int(visual_appearance_patch_size)
        self.backbone = VideoSwinTransformer(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            return_stages=self.backbone_stages,
            input_format=input_format,
            output_format="BCTHW",
            resize_to=resize_to,
            normalize=backbone_normalize,
            gradient_checkpointing=backbone_gradient_checkpointing,
        )
        feature_channels = self.backbone.get_feature_channels()
        self.stage_channels = {
            stage: feature_channels[stage] for stage in self.backbone_stages
        }
        self.concept_creations = nn.ModuleDict()
        for stage in self.backbone_stages:
            self.concept_creations[stage] = VisualConceptCreation(
                in_channels=self.stage_channels[stage],
                concept_dim=concept_dim,
                num_concepts=num_concepts,
                hidden_dim=concept_hidden_dim,
                top_k=top_k,
                visual_concept_residual_weight=visual_concept_residual_weight,
                visual_assignment_mode=visual_assignment_mode,
                visual_assignment_temperature=visual_assignment_temperature,
                visual_entropy_weight=visual_entropy_weight,
                visual_usage_weight=visual_usage_weight,
                use_visual_saliency_alignment=use_visual_saliency_alignment,
                visual_saliency_align_weight=visual_saliency_align_weight,
                stage_name=stage,
                visual_appearance_patch_size=self.visual_appearance_patch_size,
                visual_cluster_dir=self.visual_cluster_dir,
                visual_cluster_max_patches_per_prototype=self.visual_cluster_max_patches_per_prototype,
                visual_cluster_seed=self.visual_cluster_seed,
                visual_cluster_expected_checkpoint_sha256=self.visual_cluster_expected_checkpoint_sha256,
            )
        self.temporal_feature_infusers = nn.ModuleDict()
        for stage in self.backbone_stages:
            self.temporal_feature_infusers[stage] = SpatioTemporal3DFeatureInfusion(
                in_channels=self.stage_channels[stage],
                temporal_dim=self.temporal_dim,
                num_blocks=temporal_num_blocks,
                mlp_ratio=temporal_mlp_ratio,
                dropout=temporal_dropout,
                use_temporal_difference=temporal_use_difference,
                residual_gate_init=temporal_residual_gate_init,
                enhance_last_only=temporal_enhance_last_only,
                layer_scale_init=0.001,
            )
        self.saliency_prediction = ConceptGatedMultiScaleSaliencyDecoder(
            stage_channels=self.stage_channels,
            concept_dim=concept_dim,
            decoder_channels=saliency_hidden_dim,
            dropout=0.05,
            output_activation=output_activation,
            use_side_logit_fusion=decoder_use_side_logit_fusion,
            use_shared_concept_activations=use_shared_concept_activations,
            prototype_bottleneck_strength=prototype_bottleneck_strength,
            fine_unary_mask_strength=fine_unary_mask_strength,
            stage2_backbone_fusion_enabled=stage2_backbone_fusion_enabled,
        )
        if freeze_backbone:
            self.freeze_backbone()
        self._backbone_device: Optional[torch.device] = None
        self._head_device: Optional[torch.device] = None

    @property
    def input_device(self) -> torch.device:
        """Device for model inputs (video tensors)."""
        if self._backbone_device is not None:
            return self._backbone_device
        return next(self.backbone.parameters()).device

    @property
    def output_device(self) -> torch.device:
        """Device for model outputs and supervision tensors."""
        if self._head_device is not None:
            return self._head_device
        return self.input_device

    def to_split_devices(
        self, backbone_device: torch.device, head_device: torch.device
    ) -> "ExplainableVidSalModel":
        """Place backbone on one device and all downstream modules on another."""
        self.backbone.to(backbone_device)
        self.concept_creations.to(head_device)
        self.temporal_feature_infusers.to(head_device)
        self.saliency_prediction.to(head_device)
        self._backbone_device = torch.device(backbone_device)
        self._head_device = torch.device(head_device)
        return self

    def _move_features_dict_to_head(
        self, features_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        head_device = self.output_device
        if head_device == self.input_device:
            return features_dict
        return {
            stage: features.to(head_device, non_blocking=True)
            for stage, features in features_dict.items()
        }

    def prepare_training_batch(
        self, rgb_batch: torch.Tensor, sal_batch: torch.Tensor, fix_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move a training batch to the backbone/head devices."""
        rgb_batch = rgb_batch.to(self.input_device, non_blocking=True)
        sal_batch = sal_batch.to(self.output_device, non_blocking=True)
        fix_batch = fix_batch.to(self.output_device, non_blocking=True)
        return (rgb_batch, sal_batch, fix_batch)

    def _normalize_video_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Accept dataloader layouts and convert to configured input_format."""
        if x.dim() != 5:
            return x
        if x.shape[2] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 1, 4, 2, 3)
        return x

    def _extract_last_rgb_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Last frame from the original (pre-backbone) video tensor."""
        if x.dim() != 5:
            raise ValueError(f"Expected 5D video tensor, got shape {tuple(x.shape)}")
        if self.input_format == "BTCHW":
            if x.shape[2] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)
            if x.shape[2] != 3:
                raise ValueError(
                    f"BTCHW input expected 3 channels at dim 2, got shape {tuple(x.shape)}"
                )
            last_rgb = x[:, -1]
        elif self.input_format == "BCTHW":
            if x.shape[1] != 3:
                raise ValueError(
                    f"BCTHW input expected 3 channels at dim 1, got shape {tuple(x.shape)}"
                )
            last_rgb = x[:, :, -1]
        else:
            raise ValueError(f"Unsupported input_format: {self.input_format}")
        last_rgb = last_rgb.float()
        if last_rgb.numel() > 0 and last_rgb.max() > 2.0:
            last_rgb = last_rgb / 255.0
        return last_rgb

    def _prepare_rgb_last_frame_for_concepts(
        self, last_rgb_frame: torch.Tensor
    ) -> torch.Tensor:
        """Match backbone spatial canvas for concept RGB plumbing; no ImageNet norm."""
        rgb = last_rgb_frame.detach().float()
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"last_rgb_frame must have shape [B,3,H,W], got {tuple(rgb.shape)}"
            )
        resize_to = getattr(self.backbone, "resize_to", None)
        if resize_to is not None:
            if isinstance(resize_to, int):
                target_hw = (int(resize_to), int(resize_to))
            elif isinstance(resize_to, (tuple, list)) and len(resize_to) == 2:
                target_hw = (int(resize_to[0]), int(resize_to[1]))
            else:
                raise ValueError(
                    f"Unsupported backbone.resize_to={resize_to!r}; expected None, int, or (height, width)."
                )
            if rgb.shape[-2:] != target_hw:
                rgb = F.interpolate(
                    rgb, size=target_hw, mode="bilinear", align_corners=False
                )
        return rgb.to(self.output_device, non_blocking=True)

    def _decoder_output_size(self, last_rgb_frame: torch.Tensor) -> Tuple[int, int]:
        resize_to = getattr(self.backbone, "resize_to", None)
        if resize_to is None:
            return last_rgb_frame.shape[-2:]
        if isinstance(resize_to, int):
            return (int(resize_to), int(resize_to))
        return (int(resize_to[0]), int(resize_to[1]))

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
        self._backbone_frozen = False

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Parameters for the optimizer (concept + saliency, optionally backbone)."""
        params: List[nn.Parameter] = []
        params.extend(self.concept_creations.parameters())
        if self.use_temporal_feature_infusion:
            params.extend(self.temporal_feature_infusers.parameters())
        params.extend(self.saliency_prediction.parameters())
        if not self._backbone_frozen:
            params.extend(self.backbone.parameters())
        return params

    def optimize_for_inference(self) -> "ExplainableVidSalModel":
        """Enable cudnn benchmark and TF32 matmul for fixed-size inference."""
        self.eval()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        return self

    def get_temporal_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Return last temporal-infusion diagnostics for each backbone stage."""
        if not self.use_temporal_feature_infusion:
            return {}
        return {
            stage: self.temporal_feature_infusers[stage].get_last_diagnostics()
            for stage in self.backbone_stages
        }

    def _prepare_cached_backbone_features_for_stage(
        self,
        cached_backbone_features: Optional[Dict[str, torch.Tensor]],
        stage: str,
        concept_features: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Align a detached reference last-slice tensor to this stage's concept grid."""
        if cached_backbone_features is None:
            return None
        if stage not in cached_backbone_features:
            raise KeyError(
                f"cached_backbone_features missing stage {stage!r}; got keys {sorted(cached_backbone_features)}"
            )
        reference = cached_backbone_features[stage]
        if not torch.is_tensor(reference):
            raise TypeError(
                f"cached_backbone_features[{stage!r}] must be a tensor, got {type(reference).__name__}"
            )
        if reference.requires_grad:
            raise ValueError(
                f"cached_backbone_features[{stage!r}] must have requires_grad=False."
            )
        reference = reference.detach().to(
            device=concept_features.device, dtype=torch.float32
        )
        if reference.dim() != 5:
            raise ValueError(
                f"cached_backbone_features[{stage!r}] must be [B,C,1,H,W], got {tuple(reference.shape)}"
            )
        if reference.shape[0] != concept_features.shape[0]:
            raise ValueError(
                f"cached_backbone_features[{stage!r}] batch {reference.shape[0]} != live features batch {concept_features.shape[0]}"
            )
        expected_c = int(self.stage_channels[stage])
        if reference.shape[1] != expected_c:
            raise ValueError(
                f"cached_backbone_features[{stage!r}] channels {reference.shape[1]} != expected {expected_c}"
            )
        if reference.shape[2] != 1:
            raise ValueError(
                f"cached_backbone_features[{stage!r}] must have temporal length 1, got T={reference.shape[2]}"
            )
        if not torch.isfinite(reference).all():
            raise ValueError(
                f"cached_backbone_features[{stage!r}] contains non-finite values."
            )
        live_last = concept_features[:, :, -1:, :, :]
        if reference.shape[-2:] != live_last.shape[-2:]:
            raise ValueError(
                f"After concept-grid resize, cached_backbone_features[{stage!r}] spatial {tuple(reference.shape[-2:])} != live last-slice {tuple(live_last.shape[-2:])}"
            )
        if reference.shape[1] != live_last.shape[1]:
            raise ValueError(
                f"cached_backbone_features[{stage!r}] channels {reference.shape[1]} != live {live_last.shape[1]}"
            )
        return reference.detach()

    def forward(
        self,
        x: torch.Tensor,
        saliency_maps: Optional[torch.Tensor] = None,
        return_details: Optional[bool] = None,
        return_concept_losses: Optional[bool] = None,
        return_decoder_diagnostics: bool = False,
        cached_backbone_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Args:
        x: RGB video window from the dataloader.
        saliency_maps: optional GT saliency for auxiliary ConceptCreation losses
            during training (never passed to the final saliency prediction path).
        return_details: if True, return intermediate outputs; default from __init__.
        return_concept_losses: if None, concept losses are computed only while
            training (and when saliency_maps is provided). In eval mode, concept
            losses are disabled unless explicitly requested or
            allow_eval_concept_losses=True.
        return_decoder_diagnostics: if True, include decoder stage maps/gates in
            prediction_out. Off by default so diagnostics never change training.
        cached_backbone_features: optional dict of detached frozen-reference
            last-slice tensors ``{stage: [B,C_stage,1,H,W]}``. Live prediction
            features are still computed from the trainable backbone; reference
            tensors are only forwarded into concept modules for later losses."""
        if return_details is None:
            return_details = self.return_details
        inference_ctx = torch.inference_mode if not self.training else nullcontext
        with inference_ctx():
            x = self._normalize_video_layout(x)
            x = x.to(self.input_device, non_blocking=True)
            last_rgb_frame = self._extract_last_rgb_frame(x)
            if self._backbone_frozen:
                with torch.no_grad():
                    features_dict = self.backbone.forward_features(x)
            else:
                features_dict = self.backbone.forward_features(x)
            features_dict = self._move_features_dict_to_head(features_dict)
            for stage in self.backbone_stages:
                if stage in features_dict:
                    _diag_record(f"backbone.{stage}", features_dict[stage])
            if saliency_maps is not None:
                saliency_maps = saliency_maps.to(self.output_device, non_blocking=True)
            concept_outs: Dict[str, Dict[str, Any]] = {}
            decoder_features_dict: Dict[str, torch.Tensor] = {}
            concept_features_shape: Optional[Dict[str, Tuple[int, ...]]] = (
                {} if return_details else None
            )
            if return_concept_losses is None:
                if self.training:
                    return_concept_losses = saliency_maps is not None
                else:
                    return_concept_losses = (
                        self.allow_eval_concept_losses and saliency_maps is not None
                    )
            saliency_maps_for_concepts = (
                saliency_maps if return_concept_losses else None
            )
            rgb_last_frame_for_concepts = None
            if self.training and return_concept_losses:
                rgb_last_frame_for_concepts = self._prepare_rgb_last_frame_for_concepts(
                    last_rgb_frame
                )
            concept_creations = self.concept_creations
            temporal_feature_infusers = self.temporal_feature_infusers
            for stage in self.backbone_stages:
                stage_features = features_dict[stage]
                concept_features = stage_features
                if return_details:
                    concept_features_shape[stage] = tuple(concept_features.shape)
                reference_stage_features = (
                    self._prepare_cached_backbone_features_for_stage(
                        cached_backbone_features, stage, concept_features
                    )
                )
                concept_creations[stage]._diagnostic_label = f"concept.{stage}"
                visual_out = concept_creations[stage](
                    concept_features,
                    saliency_maps=saliency_maps_for_concepts,
                    return_losses=return_concept_losses,
                    rgb_last_frame=rgb_last_frame_for_concepts,
                    cached_backbone_features=reference_stage_features,
                )
                concept_outs[stage] = visual_out
                if self.use_temporal_feature_infusion:
                    decoder_features_dict[stage] = temporal_feature_infusers[stage](
                        stage_features
                    )
                else:
                    decoder_features_dict[stage] = concept_features
                _diag_record(f"decoder_features.{stage}", decoder_features_dict[stage])
            pred_out = self.saliency_prediction(
                concept_outs=concept_outs,
                features_dict=decoder_features_dict,
                output_size=self._decoder_output_size(last_rgb_frame),
                return_details=return_decoder_diagnostics,
            )
            if not return_details:
                return pred_out["saliency_map"]
            decoder_features_shape = {
                stage: tuple(decoder_features_dict[stage].shape)
                for stage in self.backbone_stages
            }
            return {
                "saliency_map": pred_out["saliency_map"],
                "saliency_logits": pred_out["saliency_logits"],
                "concept_out": concept_outs,
                "prediction_out": pred_out,
                "concept_features_shape": concept_features_shape,
                "decoder_features_shape": decoder_features_shape,
                "features_shape": concept_features_shape,
                "temporal_diagnostics": self.get_temporal_diagnostics(),
            }


# Optional diagnostic tracing

_TRACE_BUF: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "explainable_diag_trace_buf", default=None
)

_TRACE_COUNTS: ContextVar[Optional[Dict[str, int]]] = ContextVar(
    "explainable_diag_trace_counts", default=None
)


def diagnostic_trace_enabled() -> bool:
    return _TRACE_BUF.get() is not None


def _diag_record(op: str, value: Any, *, kind: str = "tensor") -> None:
    """Append a detached snapshot of ``value`` under ``op`` if tracing is on."""
    buf = _TRACE_BUF.get()
    if buf is None:
        return
    counts = _TRACE_COUNTS.get()
    if counts is None:
        counts = {}
        _TRACE_COUNTS.set(counts)
    n = counts.get(op, 0) + 1
    counts[op] = n
    name = op if n == 1 else f"{op}#{n}"
    entry: Dict[str, Any] = {"name": name, "op": op, "call_index": n, "kind": kind}
    if value is None:
        entry["available"] = False
        entry["reason"] = "value_is_none"
    elif torch.is_tensor(value):
        detached = value.detach()
        entry["available"] = True
        entry["tensor"] = detached
        entry["shape"] = list(detached.shape)
        entry["dtype"] = str(detached.dtype)
        entry["device"] = str(detached.device)
    elif isinstance(value, (int, float, bool)):
        entry["available"] = True
        entry["scalar"] = value
        entry["kind"] = "scalar"
    else:
        entry["available"] = False
        entry["reason"] = f"unsupported_type:{type(value).__name__}"
    buf.append(entry)


@contextmanager
def capture_diagnostic_trace() -> Iterator[List[Dict[str, Any]]]:
    """Enable tracing for the duration of the context; yield the buffer."""
    buf: List[Dict[str, Any]] = []
    token_buf = _TRACE_BUF.set(buf)
    token_counts = _TRACE_COUNTS.set({})
    try:
        yield buf
    finally:
        _TRACE_BUF.reset(token_buf)
        _TRACE_COUNTS.reset(token_counts)


# Temporal feature infusion


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
        layer_scale_init: float = 0.001,
        dropout: float = 0.0,
    ):
        super().__init__()
        groups = _pick_num_groups(dim, num_groups)
        self.temporal_dw = nn.Conv3d(
            dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim, bias=False
        )
        self.spatial_dw = nn.Conv3d(
            dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=dim, bias=False
        )
        self.pointwise = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.ReLU()
        hidden = max(int(dim * mlp_ratio), dim)
        self.ffn = nn.Sequential(
            nn.Conv3d(dim, hidden, kernel_size=1, bias=True),
            nn.ReLU(),
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
    """Factorized 3D spatiotemporal residual infusion for stage features."""

    def __init__(
        self,
        in_channels: int,
        temporal_dim: int = 256,
        num_layers: int = 2,
        num_blocks: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_temporal_difference: bool = True,
        residual_init: Optional[float] = None,
        residual_gate_init: float = -2.0,
        enhance_last_only: bool = False,
        num_groups: int = 8,
        layer_scale_init: float = 0.001,
    ):
        super().__init__()
        if num_blocks is not None:
            num_layers = int(num_blocks)
        if residual_init is not None:
            gate_init = float(residual_init)
        else:
            gate_init = float(residual_gate_init)
        self.in_channels = int(in_channels)
        self.use_temporal_difference = bool(use_temporal_difference)
        self.enhance_last_only = bool(enhance_last_only)
        self.input_proj = nn.Conv3d(in_channels, temporal_dim, kernel_size=1, bias=True)
        if self.use_temporal_difference:
            self.delta_proj = nn.Conv3d(
                temporal_dim * 2, temporal_dim, kernel_size=1, bias=True
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
            temporal_dim, in_channels, kernel_size=1, bias=True
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
        """Args:
        features: [B, C, T, H, W]"""
        if features.dim() != 5:
            raise ValueError(
                f"Expected features [B,C,T,H,W], got {tuple(features.shape)}"
            )
        B, C, T, H, W = features.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")
        x = self.input_proj(features)
        x = self._apply_temporal_difference(x)
        for block in self.blocks:
            x = block(x)
        temporal_residual = self.output_proj(x)
        gate = torch.sigmoid(self.residual_gate)
        if self.enhance_last_only:
            out = features.clone()
            out[:, :, -1:] = features[:, :, -1:] + gate * temporal_residual[:, :, -1:]
        else:
            out = features + gate * temporal_residual
        with torch.no_grad():
            input_abs_mean = features.detach().abs().mean().item()
            temporal_residual_abs_mean = temporal_residual.detach().abs().mean().item()
            enhanced_delta_abs_mean = (
                (out.detach() - features.detach()).abs().mean().item()
            )
            self._last_diagnostics = {
                "temporal_gate": gate.detach().item(),
                "temporal_residual_abs_mean": temporal_residual_abs_mean,
                "input_abs_mean": input_abs_mean,
                "residual_to_input_ratio": temporal_residual_abs_mean
                / (input_abs_mean + 1e-06),
                "enhanced_delta_abs_mean": enhanced_delta_abs_mean,
                "enhanced_delta_to_input_ratio": enhanced_delta_abs_mean
                / (input_abs_mean + 1e-06),
            }
        return out


# Patch prototypes and immutable reference bank

DEFAULT_PATCH_CLUSTER_DIR = (
    "/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster"
)

PATCH_CLUSTER_REFERENCE_FORMAT = "patch_cluster_reference_v1"

DEFAULT_REFERENCE_SNAPSHOT_DIR = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/prot_queue"
)


class VisualConceptCreation(VisualConceptLossMixin, nn.Module):
    """Patch-level visual concept assignment from backbone features.
    Expected feature input: [B, C, T, H, W].
    Patch-concept assignments are computed on the last frame only; earlier
    frames in the window are retained in metadata via ``window_T`` for the
    decoder. Each normalized last-frame patch feature is encoded and matched
    to a visual concept bank via ``visual_encoder`` and ``visual_concepts``."""

    DROPOUT_P = 0.2

    def __init__(
        self,
        in_channels: int,
        concept_dim: int = 256,
        num_concepts: int = 32,
        hidden_dim: int = 512,
        top_k: int = 8,
        diversity_margin: float = 0.2,
        loss_weights: Optional[Dict[str, float]] = None,
        num_visual_concepts: Optional[int] = None,
        visual_concept_residual_weight: float = 0.1,
        visual_assignment_mode: str = "topk_soft",
        visual_assignment_temperature: float = 0.07,
        visual_entropy_weight: float = 0.01,
        visual_usage_weight: float = 0.1,
        use_visual_saliency_alignment: bool = True,
        visual_saliency_align_weight: float = 0.05,
        visual_appearance_patch_size: int = 16,
        stage_name: Optional[str] = None,
        visual_appearance_query_chunk_size: int = 64,
        visual_cluster_dir: Optional[str] = None,
        visual_cluster_max_patches_per_prototype: int = 0,
        visual_cluster_seed: int = 42,
        visual_cluster_expected_checkpoint_sha256: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        if stage_name is not None and stage_name not in (
            "stage1",
            "stage2",
            "stage3",
            "stage4",
        ):
            raise ValueError(
                "stage_name must be stage1, stage2, stage3, stage4, or None."
            )
        self.stage_name = (
            stage_name
            if stage_name is not None
            else {96: "stage1", 192: "stage2", 384: "stage3", 768: "stage4"}.get(
                in_channels
            )
        )
        self._appearance_stage_source = (
            "explicit" if stage_name is not None else "video_swin_channels"
        )
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self.top_k = top_k
        self.diversity_margin = diversity_margin
        self.num_visual_concepts = (
            num_visual_concepts if num_visual_concepts is not None else num_concepts
        )
        self.visual_concept_residual_weight = visual_concept_residual_weight
        self.visual_assignment_mode = (
            "topk_soft"
            if visual_assignment_mode in ("straight_through", "soft")
            else visual_assignment_mode
        )
        if (
            not math.isfinite(float(visual_assignment_temperature))
            or visual_assignment_temperature <= 0
        ):
            raise ValueError(
                "visual_assignment_temperature must be positive and finite."
            )
        self.visual_assignment_temperature = visual_assignment_temperature
        self.visual_entropy_weight = visual_entropy_weight
        self.visual_usage_weight = visual_usage_weight
        self.use_visual_saliency_alignment = use_visual_saliency_alignment
        self.visual_saliency_align_weight = visual_saliency_align_weight
        if (
            isinstance(visual_appearance_patch_size, bool)
            or not isinstance(visual_appearance_patch_size, int)
            or visual_appearance_patch_size < 1
        ):
            raise ValueError("visual_appearance_patch_size must be a positive integer.")
        self.visual_appearance_patch_size = visual_appearance_patch_size
        if (
            isinstance(visual_appearance_query_chunk_size, bool)
            or not isinstance(visual_appearance_query_chunk_size, int)
            or visual_appearance_query_chunk_size < 1
        ):
            raise ValueError(
                "visual_appearance_query_chunk_size must be a positive integer."
            )
        self.visual_appearance_query_chunk_size = visual_appearance_query_chunk_size
        self.visual_appearance_dim = 3 * visual_appearance_patch_size**2
        weights = dict(self.DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            removed = {"visual_queue", "visual_queue_prototype"}.intersection(
                loss_weights
            )
            if removed:
                raise ValueError(
                    "Embedding queue losses were removed; use loss_weights['visual_appearance'] instead."
                )
            weights.update(loss_weights)
        for name in (
            "visual_appearance",
            "visual_rgb",
            "visual_feature",
            "visual_structure",
        ):
            value = weights[name]
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(
                    f"loss_weights[{name!r}] must be finite and nonnegative."
                )
        self.loss_weights = weights
        if (
            isinstance(visual_cluster_max_patches_per_prototype, bool)
            or not isinstance(visual_cluster_max_patches_per_prototype, int)
            or visual_cluster_max_patches_per_prototype < 0
        ):
            raise ValueError(
                "visual_cluster_max_patches_per_prototype must be an integer >= 0 (0 means all members)."
            )
        if (
            isinstance(visual_cluster_seed, bool)
            or not isinstance(visual_cluster_seed, int)
            or visual_cluster_seed < 0
        ):
            raise ValueError("visual_cluster_seed must be a nonnegative integer.")
        self.visual_cluster_dir = str(
            Path(visual_cluster_dir or DEFAULT_PATCH_CLUSTER_DIR).expanduser().resolve()
        )
        self.visual_cluster_max_patches_per_prototype = (
            visual_cluster_max_patches_per_prototype
        )
        self.visual_cluster_seed = visual_cluster_seed
        self.visual_cluster_expected_checkpoint_sha256 = (
            visual_cluster_expected_checkpoint_sha256
        )
        self.visual_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.DROPOUT_P),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, concept_dim),
            nn.LayerNorm(concept_dim),
        )
        self.visual_concepts = nn.Parameter(
            torch.randn(self.num_visual_concepts, concept_dim)
        )
        # The loss uses this head, but it stays here to preserve checkpoint keys.
        self.visual_saliency_head = nn.Sequential(
            nn.Linear(concept_dim * 2 + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.DROPOUT_P),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._init_concept_parameters()
        self._grid_cache: Dict[
            Tuple[int, int, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._visual_meta_cache: Dict[
            Tuple[int, int, int, torch.device], Dict[str, torch.Tensor]
        ] = {}
        self._cluster_loaded = False
        self._cluster_metadata = {}
        self._cluster_feature_mean = torch.empty(0, dtype=torch.float32)
        self._cluster_structure_mean = torch.empty(0, dtype=torch.float32)
        self._cluster_flat_fraction = torch.empty(0, dtype=torch.float32)
        self._cluster_counts = torch.empty(0, dtype=torch.long)
        self._cluster_rgb_sorted = ()
        self._cluster_rgb_prefix = ()
        self._appearance_forward_count = 0
        self.reset_appearance_forward_counter()
        if (
            self.stage_name == "stage4"
            and self.loss_weights["visual_appearance"] > 0
            and any(
                self.loss_weights[k] > 0
                for k in ("visual_rgb", "visual_feature", "visual_structure")
            )
            and (Path(self.visual_cluster_dir) / "manifest.json").is_file()
        ):
            self.load_visual_patch_clusters()

    _CLUSTER_FLOAT_TENSORS = (
        "_cluster_feature_mean",
        "_cluster_structure_mean",
        "_cluster_flat_fraction",
    )
    _CLUSTER_INT_TENSORS = ("_cluster_counts",)

    def _apply(self, fn, recurse=True):
        # References move with the module but retain FP32, including under half().
        result = super()._apply(fn, recurse=recurse)
        for name in self._CLUSTER_FLOAT_TENSORS + self._CLUSTER_INT_TENSORS:
            value = getattr(self, name, None)
            if value is not None:
                device = fn(
                    torch.empty(0, device=value.device, dtype=value.dtype)
                ).device
                setattr(self, name, value.to(device=device))
        for name in ("_cluster_rgb_sorted", "_cluster_rgb_prefix"):
            values = getattr(self, name, ())
            moved = []
            for value in values:
                device = fn(
                    torch.empty(0, device=value.device, dtype=value.dtype)
                ).device
                moved.append(value.to(device=device))
            setattr(self, name, tuple(moved))
        return result

    def _init_concept_parameters(self) -> None:
        with torch.no_grad():
            self.visual_concepts.copy_(F.normalize(self.visual_concepts, dim=-1))

    @torch.no_grad()
    def reset_appearance_forward_counter(self) -> None:
        """Reset the appearance-forward counter only; fixed memberships never clear."""
        self._appearance_forward_count = 0
        self.visual_appearance_diagnostics = self._appearance_status(
            False, "fixed_references_unchanged"
        )

    @torch.no_grad()
    def fixed_reference_identity_state(self) -> Dict[str, Any]:
        """Lightweight identity snapshot for external callers (never reference arrays)."""
        return {
            "format": "fixed_patch_cluster_reference_v1",
            "stage_name": self.stage_name,
            "num_prototypes": self.num_visual_concepts,
            "in_channels": self.in_channels,
            "stage_enabled": self.stage_name == "stage4",
            "loaded": self._cluster_loaded,
            "reference_directory": self.visual_cluster_dir,
            "reference_fingerprint": self._cluster_metadata.get("fingerprint"),
            "max_patches_per_prototype": self.visual_cluster_max_patches_per_prototype,
            "reference_seed": self.visual_cluster_seed,
            "step": self._appearance_forward_count,
        }

    @torch.no_grad()
    def _validated_reference_identity_state(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate reference identity; explicitly ignore obsolete FIFO history."""
        if not isinstance(state, dict):
            raise ValueError("Reference snapshot must be a dictionary.")
        if state.get("format") in (
            "prototype_rgb_fifo_v2",
            "prototype_rgb_feature_fifo_v3",
        ):
            if (
                state.get("num_prototypes") != self.num_visual_concepts
                or state.get("in_channels") != self.in_channels
            ):
                raise ValueError("Legacy snapshot dimensions differ from the model.")
            import warnings

            warnings.warn(
                "Ignoring obsolete FIFO history; fixed patch clusters remain the appearance targets.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"step": 0, "legacy_ignored": True}
        if (
            state.get("format") != "fixed_patch_cluster_reference_v1"
            or state.get("stage_name") != self.stage_name
            or state.get("num_prototypes") != self.num_visual_concepts
            or (state.get("in_channels") != self.in_channels)
        ):
            raise ValueError(
                "Reference snapshot stage/dimensions differ from the model."
            )
        if self.stage_name == "stage4" and state.get("loaded"):
            if not self._cluster_loaded or state.get(
                "reference_fingerprint"
            ) != self._cluster_metadata.get("fingerprint"):
                raise ValueError(
                    "Fixed cluster bank changed since the snapshot; restore the original bank/settings before resuming."
                )
        step = state.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Invalid appearance-forward count in reference snapshot.")
        return {"step": step, "legacy_ignored": False}

    @torch.no_grad()
    def _restore_validated_reference_identity_state(
        self, state: Dict[str, Any]
    ) -> None:
        """Restore forward counters only; never rewrite fixed memberships."""
        self._appearance_forward_count = state["step"]
        self.visual_appearance_diagnostics = self._appearance_status(
            False,
            "legacy_fifo_ignored"
            if state["legacy_ignored"]
            else "fixed_reference_identity_restored",
        )

    @torch.no_grad()
    def load_fixed_reference_identity_state(self, state: Dict[str, Any]) -> None:
        """Validate external identity snapshots; ignore obsolete FIFO contents."""
        self._restore_validated_reference_identity_state(
            self._validated_reference_identity_state(state)
        )

    def save_fixed_reference_identity(
        self,
        directory: str = DEFAULT_REFERENCE_SNAPSHOT_DIR,
        filename: str = "prototype_queue.pt",
    ) -> str:
        """Persist identity/counters only (no reference arrays, no FIFO history)."""
        return _atomic_reference_snapshot_save(
            self.fixed_reference_identity_state(), directory, filename
        )

    @staticmethod
    def _build_grid(
        H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Normalized patch coordinates in [-1, 1].
        Returns:
            grid: [N, 2] with N=H*W, order (x, y) matching row-major flatten of H x W."""
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.reshape(H * W, 2)

    def _make_grid(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (H, W, device, dtype)
        cached = self._grid_cache.get(key)
        if cached is None:
            cached = self._build_grid(H, W, device, dtype)
            self._grid_cache[key] = cached
        return cached

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """Args:
            features: [B, C, T, H, W]
        Returns:
            z: [B, T, N, C] with L2-normalized channels (N = H*W)."""
        B, C, T, H, W = features.shape
        z = features.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, C)
        return F.normalize(z, dim=-1)

    def _visual_metadata_indices(
        self, B: int, T: int, N: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        key = (B, T, N, device)
        cached = self._visual_meta_cache.get(key)
        if cached is None:
            patch_idx = torch.arange(N, device=device, dtype=torch.long)
            cached = {
                "batch_idx": torch.arange(
                    B, device=device, dtype=torch.long
                ).repeat_interleave(T * N),
                "time_idx": torch.arange(T, device=device, dtype=torch.long)
                .repeat_interleave(N)
                .repeat(B),
                "patch_idx": patch_idx.repeat(B * T),
            }
            self._visual_meta_cache[key] = cached
        return cached

    @staticmethod
    def _extract_active_prototypes(
        activations: torch.Tensor,
        prototypes: torch.Tensor,
        top_k: int,
        *,
        B: int,
        T: int,
        H: int,
        W: int,
        validity_eps: float = 1e-06,
        selected_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather selected prototypes independently at each patch.
        For topk_soft, selected_indices comes from the original similarity
        ranking: every selected slot is valid regardless of its small weight.
        Legacy hard_eval uses activation thresholding. Bank-size padding stays
        invalid in both modes.
        Returns:
            active_prototypes: [B, T, H, W, K, concept_dim]
            validity_mask: [B, T, H, W, K] bool
            active_indices: [B, T, H, W, K] int64 bank indices per patch"""
        num_patches = B * T * H * W
        if activations.shape[0] != num_patches:
            raise ValueError(
                f"activations must be [B*T*H*W, num_concepts], expected {num_patches} rows, got {activations.shape[0]}"
            )
        num_concepts = int(prototypes.shape[0])
        K = min(int(top_k), num_concepts)
        if selected_indices is None:
            topk_vals, topk_idx = activations.topk(K, dim=-1)
            validity = topk_vals > validity_eps
        else:
            if selected_indices.shape != (num_patches, K):
                raise ValueError(
                    "selected_indices must have shape [B*T*H*W, min(top_k, bank_size)]."
                )
            topk_idx = selected_indices
            validity = torch.ones_like(topk_idx, dtype=torch.bool)
        proto_n = F.normalize(prototypes, dim=-1)
        active = proto_n[topk_idx]
        if K < top_k:
            pad_k = top_k - K
            active = F.pad(active, (0, 0, 0, pad_k))
            validity = F.pad(validity, (0, pad_k), value=False)
            topk_idx = F.pad(topk_idx, (0, pad_k), value=0)
        active = active.view(B, T, H, W, top_k, -1)
        validity = validity.view(B, T, H, W, top_k)
        active_indices = topk_idx.view(B, T, H, W, top_k)
        return (active, validity, active_indices)

    def _compute_visual_assignments(
        self, raw_similarity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Sparse soft assignments over the actual top-k cosine matches.
        Training and evaluation use the same weighted forward for topk_soft.
        Gradients flow through the selected weights, embeddings, and vectors;
        top-k index selection is discrete. "straight_through" and "soft" are
        compatibility aliases for topk_soft. Explicit hard_eval retains top-1.
        Full-bank probabilities are retained for the existing regularizers,
        while visual_activations is the sparse evidence used by prediction."""
        if self.visual_assignment_mode not in (
            "topk_soft",
            "straight_through",
            "soft",
            "hard_eval",
        ):
            raise ValueError(
                f"visual_assignment_mode must be one of 'topk_soft', 'straight_through', 'soft', or 'hard_eval', got {self.visual_assignment_mode!r}"
            )
        temperature = max(float(self.visual_assignment_temperature), 1e-08)
        similarity = (
            raw_similarity.float()
            if raw_similarity.dtype in (torch.float16, torch.bfloat16)
            else raw_similarity
        )
        visual_logits = similarity / temperature
        visual_probs = F.softmax(visual_logits, dim=-1)
        visual_indices = visual_logits.argmax(dim=-1)
        topk_indices = None
        # Preserve full-bank probabilities for losses and sparse top-k for prediction.
        if self.visual_assignment_mode == "hard_eval":
            visual_activations = F.one_hot(
                visual_indices, num_classes=self.num_visual_concepts
            ).to(dtype=visual_probs.dtype)
        else:
            k = min(self.top_k, int(visual_logits.shape[-1]))
            topk_logits, topk_indices = visual_logits.topk(k, dim=-1)
            weights = F.softmax(topk_logits, dim=-1)
            weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            visual_activations = torch.zeros_like(visual_logits).scatter(
                dim=-1, index=topk_indices, src=weights
            )
        return {
            "visual_logits": visual_logits,
            "visual_probs": visual_probs,
            "visual_activations": visual_activations,
            "visual_indices": visual_indices,
            "topk_indices": topk_indices,
        }

    def _build_visual_concepts_from_patches(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Build visual-only concept assignments from the last-frame patch features.
        Args:
            features: [B, C, T, H, W]
        Returns:
            visual_patch_embeddings: [B*N, concept_dim]
            visual_concept_logits: [B*N, num_visual_concepts]
            visual_concept_indices: [B*N]
            visual_activations: [B*N, num_visual_concepts]
            visual_concept_representation: [B*N, concept_dim]
            visual_feature_concept_agreement: [B*N]
            visual_metadata: dict with batch_idx, time_idx, patch_idx, patch_coords,
                feature_shape (T=1), and window_T (full input temporal length)"""
        B, C, window_T, H, W = features.shape
        device = features.device
        dtype = features.dtype
        N = H * W
        features_last = features[:, :, -1:, :, :]
        T = 1
        z = self._flatten_features(features_last)
        patch_vectors = z.reshape(B * T * N, C)
        q_vis = self.visual_encoder(patch_vectors)
        q_vis = F.normalize(q_vis, dim=-1)
        c_vis = F.normalize(self.visual_concepts, dim=-1)
        raw_similarity = q_vis @ c_vis.T
        assignment_out = self._compute_visual_assignments(raw_similarity)
        visual_logits = assignment_out["visual_logits"]
        visual_probs = assignment_out["visual_probs"]
        visual_indices = assignment_out["visual_indices"]
        visual_activations = assignment_out["visual_activations"]
        visual_repr = visual_activations @ c_vis
        if self.visual_concept_residual_weight > 0:
            visual_repr = visual_repr + self.visual_concept_residual_weight * q_vis
        visual_repr = F.normalize(visual_repr, dim=-1)
        visual_feature_concept_agreement = F.cosine_similarity(
            q_vis, visual_repr, dim=-1
        )
        reg_out = self._visual_assignment_regularizers(visual_probs)
        grid = self._make_grid(H, W, device, dtype)
        meta_idx = self._visual_metadata_indices(B, T, N, device)
        patch_idx = meta_idx["patch_idx"]
        visual_patch_coords = grid[patch_idx]
        visual_metadata: Dict[str, Any] = {
            "batch_idx": meta_idx["batch_idx"],
            "time_idx": meta_idx["time_idx"],
            "patch_idx": patch_idx,
            "patch_coords": visual_patch_coords,
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
            "window_T": window_T,
            "assignment_time_index": window_T - 1,
        }
        (
            active_visual_prototypes,
            visual_validity_mask,
            active_visual_prototype_indices,
        ) = self._extract_active_prototypes(
            visual_activations,
            self.visual_concepts,
            self.top_k,
            B=B,
            T=T,
            H=H,
            W=W,
            selected_indices=assignment_out["topk_indices"],
        )
        _diag_label = getattr(self, "_diagnostic_label", "concept")
        _diag_record(f"{_diag_label}.patch_embeddings", q_vis)
        _diag_record(f"{_diag_label}.prototype_cosine_similarity", raw_similarity)
        _diag_record(f"{_diag_label}.visual_activations", visual_activations)
        _diag_record(f"{_diag_label}.topk_indices", active_visual_prototype_indices)
        _diag_record(f"{_diag_label}.active_prototypes", active_visual_prototypes)
        _diag_record(f"{_diag_label}.concept_representation", visual_repr)
        return {
            "active_visual_prototypes": active_visual_prototypes,
            "active_visual_prototype_indices": active_visual_prototype_indices,
            "visual_validity_mask": visual_validity_mask,
            "visual_patch_embeddings": q_vis,
            "visual_concept_logits": visual_logits,
            "visual_concept_indices": visual_indices,
            "visual_activations": visual_activations,
            "visual_concept_representation": visual_repr,
            "visual_feature_concept_agreement": visual_feature_concept_agreement,
            "visual_patch_coords": visual_patch_coords,
            "visual_assignment_probs": visual_probs,
            "visual_assignment_entropy": reg_out["visual_assignment_entropy"],
            "visual_assignment_usage": reg_out["visual_assignment_usage"],
            "loss_visual_assignment_usage": reg_out["loss_visual_assignment_usage"],
            "visual_metadata": visual_metadata,
        }

    def forward(
        self,
        features: torch.Tensor,
        saliency_maps: Optional[torch.Tensor] = None,
        return_losses: bool = True,
        rgb_last_frame: Optional[torch.Tensor] = None,
        cached_backbone_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:
            features: [B, C, T, H, W] backbone features (frozen or trainable).
            saliency_maps: optional GT saliency for training-only auxiliary losses.
            return_losses: whether to compute visual concept auxiliary losses.
            rgb_last_frame: optional unnormalized RGB last frame, float32 in [0, 1],
                shaped [B, 3, H_rgb, W_rgb], on the same device as ``features``.
                Prepared by the parent model to match the backbone spatial canvas
                (``resize_to``) without ImageNet mean/std. Existing callers may omit
                it. Appearance descriptors use the actual ``features`` H,W grid,
                in batch-major, row-major patch order. RGB appearance cost is fixed;
                gradients reach model parameters through live soft assignments.
            cached_backbone_features: optional detached frozen-reference last-slice
                tensor shaped [B, C, 1, H, W], aligned with ``features[:, :, -1:]``.
                Flattening order matches live visual patches:
                [B,C,1,H,W] -> [B,H,W,C] -> [B*H*W,C]. Each frozen vector is
                L2 normalized for fixed-cluster cosine-distance supervision. It remains
                separate from learned ``visual_patch_embeddings``. When the
                feature component is enabled, missing cached features raise an error.
                Zero vectors are excluded from every component.
                Set visual_feature=0 for an RGB/structure-only ablation.
        Returns:
            Visual concept outputs and optional auxiliary losses."""
        if features.dim() != 5:
            raise ValueError(
                f"features must be [B,C,T,H,W], got shape {tuple(features.shape)}"
            )
        if features.size(2) < 1:
            raise ValueError(f"Need T>=1 for visual concepts, got T={features.size(2)}")
        if self.stage_name != "stage4":
            rgb_last_frame = None
            cached_backbone_features = None
        if cached_backbone_features is not None:
            if not torch.is_tensor(cached_backbone_features):
                raise TypeError(
                    "cached_backbone_features must be a torch.Tensor or None."
                )
            if cached_backbone_features.requires_grad:
                raise ValueError(
                    "cached_backbone_features must be detached (requires_grad=False)."
                )
            if not cached_backbone_features.is_floating_point():
                raise TypeError(
                    "cached_backbone_features must be a floating-point tensor."
                )
            ref = cached_backbone_features.detach()
            if ref.dim() != 5 or ref.shape[2] != 1:
                raise ValueError(
                    f"cached_backbone_features must have shape [B,C,1,H,W], got {tuple(ref.shape)}"
                )
            live_last = features[:, :, -1:, :, :]
            if ref.shape[0] != live_last.shape[0]:
                raise ValueError(
                    f"cached_backbone_features batch must match features: {ref.shape[0]} vs {live_last.shape[0]}"
                )
            if ref.shape[1] != live_last.shape[1]:
                raise ValueError(
                    f"cached_backbone_features channels must match features: {ref.shape[1]} vs {live_last.shape[1]}"
                )
            if ref.shape[-2:] != live_last.shape[-2:]:
                raise ValueError(
                    f"cached_backbone_features spatial grid must match features[:, :, -1:]: {tuple(ref.shape[-2:])} vs {tuple(live_last.shape[-2:])}"
                )
            if ref.device != features.device:
                raise ValueError(
                    f"cached_backbone_features must be on the same device as features: {ref.device} vs {features.device}"
                )
            if not torch.isfinite(ref).all():
                raise ValueError("cached_backbone_features contains non-finite values.")
            cached_backbone_features = ref
        else:
            cached_backbone_features = None
        if rgb_last_frame is not None:
            if not torch.is_tensor(rgb_last_frame):
                raise TypeError("rgb_last_frame must be a torch.Tensor or None.")
            if rgb_last_frame.dim() != 4 or rgb_last_frame.shape[1] != 3:
                raise ValueError(
                    f"rgb_last_frame must have shape [B,3,H_rgb,W_rgb], got {tuple(rgb_last_frame.shape)}"
                )
            if rgb_last_frame.dtype != torch.float32:
                raise ValueError(
                    "rgb_last_frame must be unnormalized float32 RGB in [0,1]."
                )
            if min(rgb_last_frame.shape[-2:]) < 1:
                raise ValueError("rgb_last_frame spatial dimensions must be positive.")
            if rgb_last_frame.shape[0] != features.shape[0]:
                raise ValueError(
                    f"rgb_last_frame batch size must match features: {rgb_last_frame.shape[0]} vs {features.shape[0]}"
                )
            if rgb_last_frame.device != features.device:
                raise ValueError(
                    f"rgb_last_frame must be on the same device as features: {rgb_last_frame.device} vs {features.device}"
                )
        visual_out = self._build_visual_concepts_from_patches(features)
        losses, align_out = self._compute_concept_losses(
            visual_out,
            saliency_maps,
            rgb_last_frame,
            cached_backbone_features,
            return_losses,
        )
        return {
            "losses": losses,
            "active_visual_prototypes": visual_out["active_visual_prototypes"],
            "active_visual_prototype_indices": visual_out[
                "active_visual_prototype_indices"
            ],
            "visual_validity_mask": visual_out["visual_validity_mask"],
            "visual_patch_embeddings": visual_out["visual_patch_embeddings"],
            "visual_concept_representation": visual_out[
                "visual_concept_representation"
            ],
            "visual_feature_concept_agreement": visual_out[
                "visual_feature_concept_agreement"
            ],
            "visual_activations": visual_out["visual_activations"],
            "visual_concept_logits": visual_out["visual_concept_logits"],
            "visual_concept_indices": visual_out["visual_concept_indices"],
            "visual_assignment_probs": visual_out["visual_assignment_probs"],
            "visual_assignment_entropy": visual_out["visual_assignment_entropy"],
            "visual_assignment_usage": visual_out["visual_assignment_usage"],
            "visual_patch_coords": visual_out["visual_patch_coords"],
            "visual_saliency_patch_logits": align_out["visual_saliency_patch_logits"],
            "visual_saliency_align_valid_frac": align_out[
                "visual_saliency_align_valid_frac"
            ],
            "visual_metadata": visual_out["visual_metadata"],
            "visual_appearance_diagnostics": dict(self.visual_appearance_diagnostics),
        }

    @torch.no_grad()
    def summarize_concepts(self, top_n: int = 5) -> Dict[str, Union[float, int]]:
        """Lightweight visual concept-bank statistics for debugging."""
        c_vis = F.normalize(self.visual_concepts, dim=-1)
        cos = c_vis @ c_vis.T
        mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
        off_diag = cos[mask]
        return {
            "num_concepts": self.num_concepts,
            "num_visual_concepts": self.num_visual_concepts,
            "concept_dim": self.concept_dim,
            "top_n": top_n,
            "visual_norm_mean": c_vis.norm(dim=-1).mean().item(),
            "visual_pairwise_cos_max": off_diag.max().item()
            if off_diag.numel() > 0
            else 0.0,
            "visual_pairwise_cos_mean": off_diag.mean().item()
            if off_diag.numel() > 0
            else 0.0,
        }

    @torch.no_grad()
    def load_visual_patch_clusters(
        self, directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load/precompute immutable reference groups OUTSIDE forward."""
        if self.stage_name != "stage4":
            return {"loaded": False, "reason": "stage_excluded"}
        import sqlite3

        root = Path(directory or self.visual_cluster_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Fixed patch clusters not found: {manifest_path}. Run form_patch_clusters.py first."
            )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if (
            manifest.get("format_version") != "fixed_stage4_patch_clusters_v1"
            or manifest.get("status") != "complete"
        ):
            raise ValueError(
                "Expected a complete fixed_stage4_patch_clusters_v1 reference bank."
            )
        if manifest.get("stage") != "stage4" or manifest.get("rgb_order") != "HWC":
            raise ValueError("Reference bank must use stage4 and HWC RGB descriptors.")
        if int(manifest["num_prototypes"]) != self.num_visual_concepts:
            raise ValueError(
                f"Cluster count {manifest['num_prototypes']} != stage4 prototype count {self.num_visual_concepts}."
            )
        if int(manifest["feature_dim"]) != self.in_channels:
            raise ValueError(
                f"Cluster feature dimension {manifest['feature_dim']} != stage4 channels {self.in_channels}."
            )
        if int(manifest["rgb_descriptor_size"]) != self.visual_appearance_patch_size:
            raise ValueError(
                f"Cluster RGB size {manifest['rgb_descriptor_size']} != visual_appearance_patch_size {self.visual_appearance_patch_size}; use matching settings."
            )
        resize = str((manifest.get("rgb_settings") or {}).get("rgb_resize", "")).lower()
        if "bilinear" not in resize:
            raise ValueError(
                "Cluster bank must document bilinear RGB crop resizing to match the training descriptor."
            )
        feature_provenance = manifest.get("reference_feature_provenance")
        if not isinstance(feature_provenance, dict):
            feature_provenance = None
        checkpoint = (manifest.get("source_manifest") or {}).get("checkpoint_sha256")
        expected = self.visual_cluster_expected_checkpoint_sha256
        if expected is not None:
            expected_fp = (
                feature_provenance.get("backbone_fingerprint")
                if feature_provenance
                else None
            )
            if expected_fp is not None and expected_fp == expected:
                pass
            elif checkpoint != expected:
                raise ValueError(
                    "Reference bank checkpoint SHA / backbone fingerprint does not match visual_cluster_expected_checkpoint_sha256. Prefer comparing reference_feature_provenance.backbone_fingerprint, not a whole-model checkpoint file hash."
                )
        references_path = root / "cluster_references.npz"
        if not references_path.is_file():
            raise FileNotFoundError(f"Cluster references missing: {references_path}")
        h = hashlib.sha256()
        with references_path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != manifest.get("references_sha256"):
            raise ValueError(
                "cluster_references.npz fingerprint differs from the completed manifest."
            )
        count, m, d = (
            int(manifest["num_patches"]),
            self.num_visual_concepts,
            self.visual_appearance_dim,
        )
        if count < m:
            raise ValueError("Reference bank has fewer patches than prototypes.")
        feature = np.load(
            root / "descriptors/feature.npy", mmap_mode="r", allow_pickle=False
        )
        rgb = np.load(root / "descriptors/rgb.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(
            root / "descriptors/prototype_ids.npy", mmap_mode="r", allow_pickle=False
        )
        node_ids = np.load(
            root / "descriptors/node_ids.npy", mmap_mode="r", allow_pickle=False
        )
        if (
            feature.shape != (count, self.in_channels)
            or rgb.shape != (count, d)
            or labels.shape != (count,)
            or (node_ids.shape != (count,))
        ):
            raise ValueError(
                "Cluster descriptor arrays disagree with manifest dimensions."
            )
        if (
            feature.dtype.kind != "f"
            or rgb.dtype.kind != "f"
            or labels.dtype.kind not in "iu"
            or (node_ids.dtype.kind not in "iu")
        ):
            raise ValueError("Invalid reference descriptor/identity dtypes.")
        if np.any(labels < 0) or np.any(labels >= m):
            raise ValueError("Reference prototype IDs are outside the model bank.")
        full_counts = np.bincount(labels.astype(np.int64), minlength=m)
        if (full_counts == 0).any():
            raise ValueError(
                "Every stage4 prototype requires a nonempty fixed reference cluster."
            )
        con = sqlite3.connect(
            (root / "patch_clusters.sqlite").as_uri() + "?mode=ro", uri=True
        )
        try:
            cursor = con.execute(
                "SELECT descriptor_index,node_id,prototype_id FROM patches ORDER BY descriptor_index"
            )
            start = 0
            while True:
                rows = cursor.fetchmany(16384)
                if not rows:
                    break
                actual = np.asarray(rows, dtype=np.int64)
                stop = start + len(rows)
                if (
                    stop > count
                    or not np.array_equal(actual[:, 0], np.arange(start, stop))
                    or (not np.array_equal(actual[:, 1], node_ids[start:stop]))
                    or (not np.array_equal(actual[:, 2], labels[start:stop]))
                ):
                    raise ValueError(
                        "Reference memberships disagree with patch_clusters.sqlite."
                    )
                start = stop
            if start != count:
                raise ValueError(
                    "Cluster database row count differs from descriptor arrays."
                )
            if con.execute(
                "SELECT id,patch_count FROM prototypes ORDER BY id"
            ).fetchall() != [(j, int(full_counts[j])) for j in range(m)]:
                raise ValueError(
                    "Prototype summary counts disagree with reference memberships."
                )
        finally:
            con.close()
        feature_means, structure_means, flat_fractions = ([], [], [])
        sorted_values, prefix_sums, selected_counts = ([], [], [])
        digest = hashlib.sha256(manifest_bytes)
        digest.update(
            str(
                (
                    self.visual_cluster_max_patches_per_prototype,
                    self.visual_cluster_seed,
                )
            ).encode()
        )
        order = np.argsort(labels, kind="stable")
        boundaries = np.concatenate(([0], np.cumsum(full_counts)))
        for j in range(m):
            indices = order[boundaries[j] : boundaries[j + 1]]
            limit = self.visual_cluster_max_patches_per_prototype
            if limit > 0 and len(indices) > limit:
                rng = np.random.default_rng(
                    np.random.SeedSequence([self.visual_cluster_seed, j])
                )
                indices = np.sort(rng.choice(indices, size=limit, replace=False))
            f = np.asarray(feature[indices], dtype=np.float64)
            r_hwc = np.asarray(rgb[indices], dtype=np.float64)
            if (
                not np.isfinite(f).all()
                or not np.isfinite(r_hwc).all()
                or r_hwc.min() < 0
                or (r_hwc.max() > 1)
            ):
                raise ValueError(
                    f"Invalid feature/RGB values in reference cluster {j}."
                )
            scales = np.max(np.abs(f), axis=1, keepdims=True)
            scaled = f / np.maximum(scales, 1e-12)
            norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            if (norms[:, 0] * scales[:, 0] < 1e-12).any():
                raise ValueError(f"Zero stage4 reference feature in cluster {j}.")
            unit_feature = scaled / norms
            r = self.visual_appearance_patch_size
            pixels = r_hwc.reshape(-1, r * r, 3)
            centered = pixels - pixels.mean(axis=1, keepdims=True)
            structure_norms = np.linalg.norm(centered.reshape(len(indices), -1), axis=1)
            flat = structure_norms < 1e-12
            unit_structure = centered / np.maximum(
                structure_norms[:, None, None], 1e-12
            )
            unit_structure[flat] = 0.0
            structure_chw = (
                unit_structure.reshape(-1, r, r, 3).transpose(0, 3, 1, 2).reshape(-1, d)
            )
            rgb_chw = r_hwc.reshape(-1, r, r, 3).transpose(0, 3, 1, 2).reshape(-1, d)
            # Do not renormalize these means: they encode mean pairwise cosine costs.
            feature_means.append(unit_feature.mean(axis=0).astype(np.float32))
            structure_means.append(structure_chw.mean(axis=0).astype(np.float32))
            flat_fractions.append(float(flat.mean()))
            # Sorted values + prefix sums recover exact mean L1 without pair tensors.
            ordered_rgb = np.sort(rgb_chw.astype(np.float32).T, axis=1)
            prefix = np.concatenate(
                (
                    np.zeros((d, 1), dtype=np.float64),
                    np.cumsum(ordered_rgb, axis=1, dtype=np.float64),
                ),
                axis=1,
            ).astype(np.float32)
            sorted_values.append(torch.from_numpy(np.ascontiguousarray(ordered_rgb)))
            prefix_sums.append(torch.from_numpy(np.ascontiguousarray(prefix)))
            selected_counts.append(len(indices))
            digest.update(np.asarray(indices, dtype="<i8").tobytes())
            digest.update(np.asarray(f, dtype="<f4").tobytes())
            digest.update(np.asarray(rgb_chw, dtype="<f4").tobytes())
        fingerprint = digest.hexdigest()
        if (
            self._cluster_loaded
            and self._cluster_metadata.get("fingerprint") != fingerprint
        ):
            raise ValueError(
                "Cannot replace an already loaded fixed cluster bank; construct a new model for a different bank/reference subset."
            )
        device = self.visual_concepts.device
        fixed_feature = torch.from_numpy(np.stack(feature_means)).to(device=device)
        fixed_structure = torch.from_numpy(np.stack(structure_means)).to(device=device)
        fixed_flat = torch.tensor(flat_fractions, dtype=torch.float32, device=device)
        fixed_counts = torch.tensor(selected_counts, dtype=torch.long, device=device)
        fixed_sorted = tuple(value.to(device=device) for value in sorted_values)
        fixed_prefix = tuple(value.to(device=device) for value in prefix_sums)
        metadata = {
            "fingerprint": fingerprint,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_checkpoint_sha256": checkpoint,
            "source_feature_semantics": manifest.get("source_feature_semantics"),
            "reference_feature_provenance": feature_provenance,
            "reference_feature_provenance_status": manifest.get(
                "reference_feature_provenance_status",
                "verified" if feature_provenance else "unverified_legacy_source",
            ),
            "selected_counts": selected_counts,
            "full_counts": full_counts.tolist(),
            "num_reference_patches": int(sum(selected_counts)),
            "rgb_resize": "bilinear_crop_antialias",
            "descriptor_order": "CHW (converted from bank HWC)",
            "cluster_seed": self.visual_cluster_seed,
            "max_patches_per_prototype": self.visual_cluster_max_patches_per_prototype,
            "source_clustering_weights": manifest.get("weights"),
            "reference_tensor_bytes": sum(
                v.numel() * v.element_size()
                for v in (
                    fixed_feature,
                    fixed_structure,
                    fixed_flat,
                    fixed_counts,
                    *fixed_sorted,
                    *fixed_prefix,
                )
            ),
        }
        self._cluster_feature_mean, self._cluster_structure_mean = (
            fixed_feature,
            fixed_structure,
        )
        self._cluster_flat_fraction, self._cluster_counts = (fixed_flat, fixed_counts)
        self._cluster_rgb_sorted, self._cluster_rgb_prefix = (
            fixed_sorted,
            fixed_prefix,
        )
        self._cluster_metadata, self._cluster_loaded = (metadata, True)
        self.visual_cluster_dir = str(root)
        self.visual_appearance_diagnostics = self._appearance_status(
            False, "fixed_clusters_loaded"
        )
        return dict(metadata)


def _unwrap_model(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _concept_modules(model):
    """Named VisualConceptCreation modules with stable parallel-wrapper unwrap."""
    model = _unwrap_model(model)
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, VisualConceptCreation)
    }


def _atomic_reference_snapshot_save(
    payload: Dict[str, Any], directory: str, filename: str
) -> str:
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Reference snapshot filename must be a simple .pt filename.")
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".ref-", suffix=".pt", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(path)


def build_patch_cluster_reference_checkpoint_metadata(
    model,
) -> Optional[Dict[str, Any]]:
    """Lightweight identity for ordinary training checkpoints (no reference arrays)."""
    model = _unwrap_model(model)
    concept_creations = getattr(model, "concept_creations", None)
    if concept_creations is None or not hasattr(concept_creations, "__contains__"):
        return None
    if "stage4" not in concept_creations:
        return None
    stage4 = concept_creations["stage4"]
    if stage4 is None or not getattr(stage4, "_cluster_loaded", False):
        return None
    meta = stage4._cluster_metadata
    payload: Dict[str, Any] = {
        "format": PATCH_CLUSTER_REFERENCE_FORMAT,
        "metadata_format_version": 1,
        "bank_fingerprint": meta.get("fingerprint"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "num_clusters": int(stage4.num_visual_concepts),
        "rgb_descriptor_size": int(stage4.visual_appearance_patch_size),
        "rgb_descriptor_dim": int(stage4.visual_appearance_dim),
        "max_patches_per_prototype": int(
            stage4.visual_cluster_max_patches_per_prototype
        ),
        "reference_seed": int(stage4.visual_cluster_seed),
        "full_member_counts": list(meta.get("full_counts") or []),
        "selected_member_counts": list(meta.get("selected_counts") or []),
        "reference_directory": stage4.visual_cluster_dir,
        "reference_feature_provenance_status": meta.get(
            "reference_feature_provenance_status"
        ),
    }
    provenance = meta.get("reference_feature_provenance")
    if isinstance(provenance, dict):
        payload["reference_feature_provenance"] = {
            "backbone_fingerprint": provenance.get("backbone_fingerprint"),
            "temporal_selection": provenance.get("temporal_selection"),
            "preprocess": provenance.get("preprocess"),
            "feature_source": provenance.get("feature_source"),
        }
    return payload


def save_fixed_reference_identities(
    model,
    directory: str = DEFAULT_REFERENCE_SNAPSHOT_DIR,
    filename: str = "prototype_queues.pt",
) -> Optional[str]:
    """Save fixed-reference identities/counters for all stages (no arrays/FIFO)."""
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_rank() != 0)
    ):
        return None
    modules = _concept_modules(model)
    if not modules:
        raise ValueError("No VisualConceptCreation modules found in model.")
    states = {
        name: module.fixed_reference_identity_state()
        for name, module in modules.items()
    }
    return _atomic_reference_snapshot_save(
        {"format": "fixed_patch_cluster_model_v1", "stages": states},
        directory,
        filename,
    )


def load_fixed_reference_identities(
    model,
    directory: str = DEFAULT_REFERENCE_SNAPSHOT_DIR,
    filename: str = "prototype_queues.pt",
) -> None:
    """Validate external identity snapshots after weights load. Never from forward."""
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Reference snapshot filename must be a simple .pt filename.")
    payload = torch.load(
        Path(directory).expanduser() / filename, map_location="cpu", weights_only=True
    )
    modules = _concept_modules(model)
    if (
        not isinstance(payload, dict)
        or payload.get("format")
        not in (
            "prototype_rgb_fifo_model_v2",
            "prototype_rgb_feature_fifo_model_v3",
            "fixed_patch_cluster_model_v1",
        )
        or (not isinstance(payload.get("stages"), dict))
        or (set(payload["stages"]) != set(modules))
    ):
        raise ValueError("Reference snapshot format/stages do not match the model.")
    expected_format = {
        "prototype_rgb_fifo_model_v2": "prototype_rgb_fifo_v2",
        "prototype_rgb_feature_fifo_model_v3": "prototype_rgb_feature_fifo_v3",
        "fixed_patch_cluster_model_v1": "fixed_patch_cluster_reference_v1",
    }[payload["format"]]
    validated = {}
    for name, module in modules.items():
        state = payload["stages"][name]
        if not isinstance(state, dict) or state.get("format") != expected_format:
            raise ValueError(
                f"Reference stage {name!r} format disagrees with its model snapshot."
            )
        validated[name] = module._validated_reference_identity_state(state)
    for name, module in modules.items():
        module._restore_validated_reference_identity_state(validated[name])


# Prototype-guided multiscale decoder

_STAGE_ORDER = ("stage4", "stage3", "stage2", "stage1")

_PAIRWISE_PRIORITY_STAGES = ("stage3", "stage4")

_UNARY_MASK_STAGES = ("stage1", "stage2")

_PRIORITY_MODES = ("none", "pairwise", "unary_mask")


_SIDE_UPSAMPLE_SCALES: Dict[str, int] = {
    "stage1": 4,
    "stage2": 8,
    "stage3": 16,
    "stage4": 32,
}


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
        raise ValueError(f"feature_shape must have 5 entries (B,C,T,H,W), got {shape}")
    return (shape[0], shape[1], shape[2], shape[3], shape[4])


def _metadata_feature_shape_repr(metadata: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(metadata, dict):
        return None
    return metadata.get("feature_shape")


def _window_temporal_length(metadata: Dict[str, Any]) -> int:
    if "window_T" in metadata:
        return int(metadata["window_T"])
    return int(_feature_shape_from_metadata(metadata)[2])


def _expand_last_frame_volume_to_window(
    volume: torch.Tensor, window_t: int
) -> torch.Tensor:
    """Broadcast a last-frame [B,D,1,H,W] concept volume across the full window."""
    if volume.shape[2] == window_t:
        return volume
    if volume.shape[2] != 1:
        raise ValueError(
            f"Expected a single-frame concept volume to expand across the window, got temporal length {volume.shape[2]} for window_t={window_t}"
        )
    return volume.expand(-1, -1, window_t, -1, -1).contiguous()


def _scatter_last_frame_update(
    base_volume: torch.Tensor, last_frame_update: torch.Tensor
) -> torch.Tensor:
    """Insert a last-frame [B,C,1,H,W] update into a [B,C,T,H,W] tensor."""
    if last_frame_update.shape[2] != 1:
        raise ValueError(
            f"last_frame_update must have temporal length 1, got {tuple(last_frame_update.shape)}"
        )
    out = base_volume.clone()
    out[:, :, -1:] = last_frame_update
    return out


def _scatter_last_frame_mask(
    batch_size: int,
    temporal_len: int,
    last_frame_mask: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B,1,T,H,W] with zeros on earlier frames and mask values on the last frame."""
    if last_frame_mask.shape[2] != 1:
        raise ValueError(
            f"last_frame_mask must have temporal length 1, got {tuple(last_frame_mask.shape)}"
        )
    _, _, _, height, width = last_frame_mask.shape
    mask = torch.zeros(
        batch_size, 1, temporal_len, height, width, device=device, dtype=dtype
    )
    mask[:, :, -1:] = last_frame_mask
    return mask


def _gather_active_visual_activations(
    visual_activations: torch.Tensor,
    active_indices: torch.Tensor,
    validity_mask: torch.Tensor,
    visual_metadata: Dict[str, Any],
) -> torch.Tensor:
    """Gather the upstream sparse top-k weights at their actual selected indices."""
    B, _, T, H, W = _feature_shape_from_metadata(visual_metadata)
    num_concepts = int(visual_activations.shape[-1])
    expected = B * T * H * W
    if visual_activations.shape[0] != expected:
        raise ValueError(
            f"visual_activations length mismatch: expected {expected} (=B*T*H*W), got {visual_activations.shape[0]}"
        )
    activations = visual_activations.float().view(B, T, H, W, num_concepts)
    if active_indices.dim() == 5:
        gathered = torch.gather(activations, dim=-1, index=active_indices)
        gathered = gathered.permute(0, 1, 4, 2, 3).contiguous()
        valid = validity_mask.permute(0, 1, 4, 2, 3).to(dtype=gathered.dtype)
        return gathered * valid
    raise ValueError(
        f"active_indices must be [B, T, H, W, K] for per-patch concept selection, got shape {tuple(active_indices.shape)}"
    )


def _build_shared_visual_modality_from_concept_out(
    concept_out: Dict[str, Any], *, concept_proto_proj: nn.Linear
) -> Dict[str, torch.Tensor]:
    """Use every valid upstream top-k slot, with its original assignment weight."""
    required = (
        "visual_activations",
        "active_visual_prototype_indices",
        "active_visual_prototypes",
        "visual_validity_mask",
        "visual_metadata",
    )
    missing = [key for key in required if key not in concept_out]
    if missing:
        raise ValueError(
            f"concept_out is missing keys required for shared concept activations: {missing}"
        )
    visual_metadata = concept_out["visual_metadata"]
    activations = _gather_active_visual_activations(
        concept_out["visual_activations"],
        concept_out["active_visual_prototype_indices"],
        concept_out["visual_validity_mask"],
        visual_metadata,
    )
    projected_prototypes = concept_proto_proj(concept_out["active_visual_prototypes"])
    return {
        "activations": activations,
        "validity_mask": concept_out["visual_validity_mask"],
        "projected_prototypes": projected_prototypes,
        "similarity_maps": None,
    }


def _build_visual_concept_volume(
    concept_out: Dict[str, Any], *, stage: str = "unknown"
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
            f"Invalid visual_metadata['feature_shape'] at {stage}: {_metadata_feature_shape_repr(visual_metadata)}"
        ) from exc
    expected = B * T * H * W
    if visual_repr.shape[0] != expected:
        raise ValueError(
            f"visual_concept_representation length mismatch at {stage}: expected {expected} (=B*T*H*W from feature_shape {_metadata_feature_shape_repr(visual_metadata)}), got {visual_repr.shape[0]}"
        )
    concept_dim = visual_repr.shape[-1]
    last_frame_volume = (
        visual_repr.reshape(B, T, H, W, concept_dim).permute(0, 4, 1, 2, 3).contiguous()
    )
    window_t = _window_temporal_length(visual_metadata)
    return _expand_last_frame_volume_to_window(last_frame_volume, window_t)


def _build_stage_concept_volume(
    concept_out: Dict[str, Any], *, stage: str = "unknown"
) -> torch.Tensor:
    """Build [B, concept_dim, T, H, W] visual concept volume for one stage."""
    concept_volume = _build_visual_concept_volume(concept_out, stage=stage)
    if concept_volume is None:
        visual_metadata = concept_out.get("visual_metadata")
        raise ValueError(
            f"concept_out at {stage} must provide visual_concept_representation and visual_metadata['feature_shape']; got visual_metadata={_metadata_feature_shape_repr(visual_metadata if isinstance(visual_metadata, dict) else None)}"
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
            f"Concept volume at {stage} must be 5D [B,concept_dim,T,H,W], got {tuple(concept_volume.shape)}"
        )
    if features_5d.dim() != 5:
        raise ValueError(
            f"features_dict[{stage}] must be 5D [B,C,T,H,W], got {tuple(features_5d.shape)}"
        )
    B_c, concept_dim, T_c, H_c, W_c = concept_volume.shape
    B_f, _, T_f, H_f, W_f = features_5d.shape
    if (
        B_c == B_f
        and T_c == T_f
        and (H_c == H_f)
        and (W_c == W_f)
        and (concept_dim == concept_volume.shape[1])
    ):
        return
    visual_metadata = concept_out.get("visual_metadata")
    metadata = concept_out.get("metadata")
    raise ValueError(
        f"Concept volume / feature volume shape mismatch at {stage}: features_5d shape={tuple(features_5d.shape)}, concept_volume shape={tuple(concept_volume.shape)}, visual_metadata['feature_shape']={_metadata_feature_shape_repr(visual_metadata if isinstance(visual_metadata, dict) else None)}, metadata['feature_shape']={_metadata_feature_shape_repr(metadata if isinstance(metadata, dict) else None)}"
    )


def _crop_or_pad_3d(x: torch.Tensor, target_thw: Tuple[int, int, int]) -> torch.Tensor:
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
    concept_volume: torch.Tensor, features_5d: torch.Tensor, *, stage: str
) -> torch.Tensor:
    """Align concept volume to decoder features."""
    _, _, T_f, H_f, W_f = features_5d.shape
    _, _, T_c, H_c, W_c = concept_volume.shape
    if T_c != T_f:
        raise ValueError(
            f"Concept volume / feature volume temporal mismatch at {stage}: concept_volume shape={tuple(concept_volume.shape)}, features_5d shape={tuple(features_5d.shape)}"
        )
    if abs(H_c - H_f) > 1 or abs(W_c - W_f) > 1:
        raise ValueError(
            f"Concept volume / feature volume shape mismatch at {stage}: concept_volume shape={tuple(concept_volume.shape)}, features_5d shape={tuple(features_5d.shape)}"
        )
    if (H_c, W_c) != (H_f, W_f):
        concept_volume = _crop_or_pad_3d(concept_volume, (T_f, H_f, W_f))
    return concept_volume


def _crop_or_pad_2d(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Match [H, W] via center crop or zero-pad on the bottom/right."""
    H_t, W_t = (int(target_hw[0]), int(target_hw[1]))
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
        groups = _pick_num_groups(out_channels, num_groups)
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
        groups = _pick_num_groups(out_channels, num_groups)
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
        groups = _pick_num_groups(channels, num_groups)
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

    def __init__(self, channels: int, *, scale_factor: int = 2, dropout: float = 0.05):
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
        groups = _pick_num_groups(channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.act = nn.ReLU()
        self.refine = Residual3DRefineBlock(channels, dropout=dropout)

    def forward(
        self, x: torch.Tensor, target_thw: Optional[Tuple[int, int, int]] = None
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
        if scale_factor < 1 or scale_factor & scale_factor - 1 != 0:
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
                            hidden_channels, hidden_channels, kernel_size=3, padding=1
                        ),
                    }
                )
            )
            ch_in = hidden_channels
        head_in = hidden_channels if num_stages > 0 else in_channels
        self.head = nn.Conv2d(head_in, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None
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


_GROUPING_EPS = 1e-06

_PRIORITY_EPS = 1e-06

_MASKED_LOGIT_FILL = -10000.0

_POS_NUM_FREQUENCIES = 8


def _masked_softmax(
    logits: torch.Tensor, validity_mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Numerically safe softmax over valid concept slots."""
    if validity_mask.dim() == 2 and logits.dim() == 3:
        mask = validity_mask.unsqueeze(1).expand(-1, logits.shape[1], -1)
    else:
        mask = validity_mask
    mask = mask.bool()
    masked_logits = logits.float().masked_fill(~mask, -torch.inf)
    has_valid = mask.any(dim=dim, keepdim=True)
    masked_logits = torch.where(
        has_valid, masked_logits, torch.zeros_like(masked_logits)
    )
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    weights = weights * mask.to(dtype=weights.dtype)
    normalizer = weights.sum(dim=dim, keepdim=True)
    weights = weights / normalizer.clamp_min(_GROUPING_EPS)
    return torch.where(normalizer > 0, weights, torch.zeros_like(weights))


def _sinusoidal_2d_encoding(
    x: torch.Tensor, y: torch.Tensor, num_frequencies: int
) -> torch.Tensor:
    """Fixed sinusoidal encoding of normalized coordinates in ``[0, 1]``."""
    freqs = (
        2.0 ** torch.arange(num_frequencies, device=x.device, dtype=x.dtype) * math.pi
    )
    x_ang = x.unsqueeze(-1) * freqs
    y_ang = y.unsqueeze(-1) * freqs
    return torch.cat([x_ang.sin(), x_ang.cos(), y_ang.sin(), y_ang.cos()], dim=-1)


class SpatioTemporalConceptGatedFusionBlock(nn.Module):
    """FiLM, decoder fusion, refinement, then last-slice residual spatial mask."""

    def __init__(
        self,
        feature_channels: int,
        concept_dim: int,
        decoder_channels: int,
        dropout: float = 0.05,
        mask_embed_dim: Optional[int] = None,
        assignment_temperature: float = 0.07,
        priority_temperature: float = 0.1,
        use_null_prototype: bool = True,
        enable_patch_prioritization: bool = False,
        priority_mode: Optional[str] = None,
        use_shared_concept_activations: bool = False,
        factorized_rank: int = 32,
        pos_num_frequencies: int = _POS_NUM_FREQUENCIES,
        prototype_bottleneck_strength: float = 0.4,
        fine_unary_mask_strength: float = 0.6,
        include_backbone_in_fusion: bool = True,
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
        if not 0.0 <= prototype_bottleneck_strength <= 1.0:
            raise ValueError(
                f"prototype_bottleneck_strength must be in [0, 1], got {prototype_bottleneck_strength}"
            )
        if not 0.0 <= fine_unary_mask_strength <= 1.0:
            raise ValueError(
                f"fine_unary_mask_strength must be in [0, 1], got {fine_unary_mask_strength}"
            )
        if priority_mode is None:
            priority_mode = "pairwise" if enable_patch_prioritization else "none"
        if priority_mode not in _PRIORITY_MODES:
            raise ValueError(
                f"priority_mode must be one of {_PRIORITY_MODES}, got {priority_mode!r}"
            )
        self.concept_dim = int(concept_dim)
        self.mask_embed_dim = (
            int(mask_embed_dim)
            if mask_embed_dim is not None
            else max(decoder_channels // 2, 32)
        )
        self.assignment_temperature = float(assignment_temperature)
        self.priority_temperature = float(priority_temperature)
        self.use_null_prototype = bool(use_null_prototype)
        self.priority_mode = str(priority_mode)
        self.use_shared_concept_activations = bool(use_shared_concept_activations)
        self.factorized_rank = int(factorized_rank)
        self.pos_num_frequencies = int(pos_num_frequencies)
        self.pos_dim = 4 * self.pos_num_frequencies
        self.token_dim = self.mask_embed_dim + self.pos_dim + 1
        self.prototype_bottleneck_strength = float(prototype_bottleneck_strength)
        self.prototype_application_position = "post_refine"
        self.fine_unary_mask_strength = float(fine_unary_mask_strength)
        self.include_backbone_in_fusion = bool(include_backbone_in_fusion)
        priority_hidden = max(int(decoder_channels), 32)
        self.feature_proj = Conv3DGNAct(
            feature_channels, decoder_channels, kernel_size=1, padding=0
        )
        self.concept_proj = Conv3DGNAct(
            concept_dim, decoder_channels, kernel_size=1, padding=0
        )
        self.prev_proj = Conv3DGNAct(
            decoder_channels, decoder_channels, kernel_size=1, padding=0
        )
        self.prev_upsample = LearnedSpatialUpsample3D(
            decoder_channels, scale_factor=2, dropout=dropout
        )
        self.film = nn.Conv3d(decoder_channels, decoder_channels * 2, kernel_size=1)
        self.refine1 = Residual3DRefineBlock(decoder_channels, dropout=dropout)
        self.refine2 = Residual3DRefineBlock(decoder_channels, dropout=dropout)
        self.prev_scale = nn.Parameter(torch.tensor(1.0))
        if self.priority_mode in ("pairwise", "unary_mask"):
            self.feature_token_proj = (
                None
                if self.use_shared_concept_activations
                else nn.Linear(decoder_channels, self.mask_embed_dim)
            )
            self.concept_proto_proj = nn.Linear(concept_dim, self.mask_embed_dim)
            self.unary_priority_mlp = nn.Sequential(
                nn.Linear(self.token_dim, priority_hidden),
                nn.ReLU(),
                nn.Linear(priority_hidden, 1, bias=False),
            )
            self.null_prototype = nn.Parameter(torch.randn(self.concept_dim) * 0.02)
        else:
            self.feature_token_proj = None
            self.concept_proto_proj = None
            self.unary_priority_mlp = None
            self.null_prototype = None
        if self.priority_mode == "pairwise":
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
            nn.init.normal_(self.factor_query[-1].weight, std=0.001)
            nn.init.normal_(self.factor_key[-1].weight, std=0.0012)
            self.raw_spatial_frequency_weights = nn.Parameter(
                torch.zeros(2 * self.pos_num_frequencies)
            )
            self.raw_spatial_strength = nn.Parameter(torch.tensor(-2.1972246))
            self.raw_pairwise_strength = nn.Parameter(torch.tensor(0.0))
        else:
            self.factor_query = None
            self.factor_key = None
            self.raw_pairwise_strength = None
            self.raw_spatial_frequency_weights = None
            self.raw_spatial_strength = None

    @property
    def spatial_strength(self) -> torch.Tensor:
        if self.raw_spatial_strength is None:
            raise RuntimeError(
                "spatial_strength is only defined when priority_mode='pairwise'"
            )
        return torch.sigmoid(self.raw_spatial_strength)

    @property
    def pairwise_strength(self) -> torch.Tensor:
        if self.raw_pairwise_strength is None:
            raise RuntimeError(
                "pairwise_strength is only defined when priority_mode='pairwise'"
            )
        return F.softplus(self.raw_pairwise_strength)

    def _normalized_spatial_grid(
        self, height: int, width: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return (grid_y, grid_x)

    def _compute_soft_activations(
        self,
        feature_proj: torch.Tensor,
        active_prototypes: torch.Tensor,
        validity_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Soft concept-patch activations ``a_ij`` with a null prototype in assignment."""
        B, _, T, H, W = feature_proj.shape
        dtype = feature_proj.dtype
        validity = validity_mask.bool()
        if active_prototypes.dim() != 6 or validity.dim() != 5:
            raise ValueError(
                f"Per-patch concept selection requires active_prototypes [B,T,H,W,K,D] and validity_mask [B,T,H,W,K], got prototype shape {tuple(active_prototypes.shape)} and validity shape {tuple(validity.shape)}"
            )
        K = int(active_prototypes.shape[-2])
        projected_features = self.feature_token_proj(
            feature_proj.permute(0, 2, 3, 4, 1)
        )
        projected_prototypes = self.concept_proto_proj(active_prototypes)
        feature_tokens = F.normalize(projected_features, dim=-1)
        proto_tokens = F.normalize(projected_prototypes, dim=-1)
        similarity_hwk = torch.einsum(
            "bthwe,bthwke->bthwk", feature_tokens, proto_tokens
        )
        similarity_maps = similarity_hwk.permute(0, 1, 4, 2, 3).contiguous()
        valid_spatial = validity.permute(0, 1, 4, 2, 3)
        assignment_logits = similarity_maps.float() / self.assignment_temperature
        assignment_logits = assignment_logits.masked_fill(
            ~valid_spatial, _MASKED_LOGIT_FILL
        )
        if self.use_null_prototype:
            null_tokens = F.normalize(
                self.concept_proto_proj(
                    self.null_prototype.to(dtype=dtype).view(1, 1, -1)
                ),
                dim=-1,
            ).expand(B, T, H, W, -1)
            null_similarity = (
                torch.einsum("bthwe,bthwe->bthw", feature_tokens, null_tokens)
                / self.assignment_temperature
            )
            assignment_logits = torch.cat(
                [assignment_logits, null_similarity.unsqueeze(2)], dim=2
            )
            assignment = torch.softmax(assignment_logits.float(), dim=2)
            assignment = assignment.clamp_min(torch.finfo(assignment.dtype).tiny)
            assignment = assignment / assignment.sum(dim=2, keepdim=True)
            activations = assignment[:, :, :K]
        else:
            activations = torch.nan_to_num(
                torch.softmax(assignment_logits.float(), dim=2), nan=0.0
            )
            activations = activations.clamp_min(torch.finfo(activations.dtype).tiny)
        activations = activations * valid_spatial.to(dtype=activations.dtype)
        return {
            "similarity_maps": similarity_maps,
            "activations": activations,
            "validity_mask": validity,
            "prototypes": active_prototypes,
            "projected_prototypes": projected_prototypes,
        }

    def _build_activation_tokens(
        self,
        projected_prototypes: torch.Tensor,
        activations: torch.Tensor,
        concept_id_offset: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """One token per activation: [normalized prototype, position, strength]."""
        if projected_prototypes.dim() != 6:
            raise ValueError(
                f"projected_prototypes must be [B,T,H,W,K,embed_dim], got {tuple(projected_prototypes.shape)}"
            )
        B, T, H, W, K, embed_dim = projected_prototypes.shape
        if tuple(activations.shape) != (B, T, K, H, W):
            raise ValueError(
                f"activations must be {(B, T, K, H, W)}, got {tuple(activations.shape)}"
            )
        dtype = projected_prototypes.dtype
        device = projected_prototypes.device
        N = K * H * W
        grid_y, grid_x = self._normalized_spatial_grid(H, W, device=device, dtype=dtype)
        pos_hw = _sinusoidal_2d_encoding(grid_x, grid_y, self.pos_num_frequencies)
        normalized_proto_flat = F.normalize(
            projected_prototypes, dim=-1, eps=_PRIORITY_EPS
        ).reshape(B, T, N, embed_dim)
        pos_flat = (
            pos_hw.reshape(1, 1, H * W, self.pos_dim)
            .unsqueeze(-2)
            .expand(B, T, -1, K, -1)
            .reshape(B, T, N, self.pos_dim)
        )
        act_flat = activations.permute(0, 1, 3, 4, 2).reshape(B, T, N, 1)
        tokens = torch.cat([normalized_proto_flat, pos_flat, act_flat], dim=-1)
        patch_index = torch.arange(H * W, device=device).repeat_interleave(K)
        concept_index = torch.arange(K, device=device).repeat(H * W) + concept_id_offset
        return {
            "tokens": tokens,
            "activations": act_flat.squeeze(-1),
            "patch_index": patch_index,
            "concept_index": concept_index,
            "x_coords": grid_x.reshape(-1).repeat_interleave(K),
            "y_coords": grid_y.reshape(-1).repeat_interleave(K),
            "num_concepts": activations.new_tensor(K, dtype=torch.long),
            "spatial_hw": (H, W),
        }

    @property
    def factor_scale(self) -> float:
        return float(self.factorized_rank) ** (-0.5)

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
        pos_encoding = _sinusoidal_2d_encoding(x, y, self.pos_num_frequencies)
        beta = (
            F.softplus(self.raw_spatial_frequency_weights.to(dtype=dtype))
            + _PRIORITY_EPS
        )
        num_freq = self.pos_num_frequencies
        beta_x = beta[:num_freq]
        beta_y = beta[num_freq:]
        expanded_beta = torch.cat([beta_x, beta_x, beta_y, beta_y], dim=-1)
        weighted_pos = pos_encoding * expanded_beta.sqrt()
        weighted_pos = F.normalize(weighted_pos, dim=-1, eps=_PRIORITY_EPS)
        rho = torch.cat(
            [torch.ones_like(weighted_pos[..., :1]), weighted_pos], dim=-1
        ) / math.sqrt(2.0)
        spatial_strength = self.spatial_strength.to(dtype=dtype, device=device)
        eta = torch.cat(
            [
                (1.0 - spatial_strength)
                .clamp_min(_PRIORITY_EPS)
                .sqrt()
                .expand(rho.shape[0], 1),
                spatial_strength.clamp_min(0.0).sqrt() * rho,
            ],
            dim=-1,
        )
        return (eta, rho)

    def _spatial_relevance_for_pairs(
        self, eta: torch.Tensor, candidate_n: torch.Tensor, candidate_m: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the symmetric spatial kernel for selected candidate pairs only."""
        return torch.einsum("ns,ms->nm", eta[candidate_n], eta[candidate_m])

    def _spatial_diagnostic_stats(self, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        beta = (
            F.softplus(self.raw_spatial_frequency_weights.to(dtype=dtype))
            + _PRIORITY_EPS
        )
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
        total_weight: torch.Tensor,
        candidate_valid: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Detached diagnostics for low-rank factorized antisymmetric comparison."""
        if candidate_valid.dim() == 2:
            valid = candidate_valid.unsqueeze(1).expand_as(context)
        else:
            valid = candidate_valid
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
        """Low-rank factorized antisymmetric concept-patch comparison with spatial kernel."""
        if candidate_valid.dim() == 2:
            valid = candidate_valid[:, None, :].to(dtype=tokens.dtype)
        else:
            valid = candidate_valid.to(dtype=tokens.dtype)
        weights = activations.clamp_min(0.0) * valid
        q = self.factor_query(tokens)
        k = self.factor_key(tokens)
        eta, _ = self._build_spatial_factors(
            x_coords, y_coords, dtype=tokens.dtype, device=tokens.device
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
            denominator > _PRIORITY_EPS, context, torch.zeros_like(context)
        )
        context = context * valid
        stats = self._factorized_comparison_stats(
            q, k, context, total_weight, candidate_valid, tokens.dtype
        )
        return (context, stats)

    def _prioritize_concept_patch_activations(
        self, modality_outs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Score valid activations from prototype, position, and strength tokens."""
        if not modality_outs:
            raise ValueError("at least one concept modality is required")
        token_parts: List[Dict[str, torch.Tensor]] = []
        concept_offset = 0
        activation_maps: List[torch.Tensor] = []
        validity_masks: List[torch.Tensor] = []
        for modality in modality_outs:
            H, W = modality["activations"].shape[-2:]
            part = self._build_activation_tokens(
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
        B, T, N, _ = tokens.shape
        H, W = token_parts[0]["spatial_hw"]
        K = N // (H * W)
        validity_parts = []
        for validity_mask in validity_masks:
            if validity_mask.dim() == 5:
                k_local = int(validity_mask.shape[-1])
                validity_parts.append(validity_mask.reshape(B, T, H * W * k_local))
            else:
                raise ValueError(
                    f"validity_mask must be [B,T,H,W,K] for per-patch concept selection, got {tuple(validity_mask.shape)}"
                )
        validity = torch.cat(validity_parts, dim=2)
        candidate_valid = validity
        unary = self.unary_priority_mlp(tokens).squeeze(-1)
        if self.priority_mode == "unary_mask":
            scores = unary
            context = None
            factorized_stats = None
        else:
            context, factorized_stats = self._factorized_pairwise_context(
                tokens, activations, candidate_valid, x_coords, y_coords
            )
            scores = unary + self.pairwise_strength * context
        priority_logits = scores / self.priority_temperature + torch.log(
            activations.clamp_min(0.0) + _PRIORITY_EPS
        )
        concept_priorities_flat = _masked_softmax(
            priority_logits, candidate_valid, dim=-1
        )
        if not bool(validity.any(dim=-1).all().item()):
            raise ValueError(
                "Priority softmax requires at least one valid concept prototype for every patch."
            )
        concept_priorities = (
            concept_priorities_flat.view(B, T, H, W, K)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        patch_priority_distribution = concept_priorities.sum(dim=2)
        spatial_mean = patch_priority_distribution.mean(dim=(-2, -1), keepdim=True)
        patch_priority_gate = patch_priority_distribution / spatial_mean.clamp_min(
            _PRIORITY_EPS
        )
        if len(validity_masks) == 1:
            concept_validity = validity_masks[0]
        else:
            concept_validity = torch.cat(validity_masks, dim=-1)
        result: Dict[str, Any] = {
            "priority_mask": patch_priority_gate.unsqueeze(1),
            "patch_priority_gate": patch_priority_gate,
            "patch_priority_distribution": patch_priority_distribution,
            "patch_priority_map": patch_priority_distribution,
            "concept_priorities": concept_priorities,
            "concept_validity": concept_validity,
            "unary_scores": unary.view(B, T, H, W, K)
            .permute(0, 1, 4, 2, 3)
            .contiguous(),
            "priority_mode": self.priority_mode,
            "pairwise_context_used": self.priority_mode == "pairwise",
        }
        if context is not None:
            result["pairwise_context"] = (
                context.view(B, T, H, W, K).permute(0, 1, 4, 2, 3).contiguous()
            )
        if factorized_stats is not None:
            result["factorized_comparison_stats"] = factorized_stats
        return result

    def _assert_required_visual_prototypes(
        self, visual_validity_mask: torch.Tensor
    ) -> None:
        validity = visual_validity_mask.bool()
        if validity.dim() != 5:
            raise ValueError(
                f"visual_validity_mask must be [B,T,H,W,K] for per-patch concept selection, got {tuple(validity.shape)}"
            )
        if validity.numel() == 0 or not bool(validity.any(dim=-1).all().item()):
            raise ValueError(
                "Visual concept activations are required for mask construction, but at least one patch has no valid visual prototypes."
            )

    def forward(
        self,
        features: torch.Tensor,
        concept_volume: torch.Tensor,
        prev_decoder: Optional[torch.Tensor] = None,
        *,
        active_visual_prototypes: torch.Tensor,
        visual_validity_mask: torch.Tensor,
        concept_out: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feature_proj = self.feature_proj(features)
        concept_proj = self.concept_proj(concept_volume)
        film_params = self.film(concept_proj)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        beta = 0.1 * beta
        film_features = feature_proj * gamma + beta
        prev_up = None
        if prev_decoder is not None:
            prev_up = self.prev_upsample(
                prev_decoder, target_thw=feature_proj.shape[-3:]
            )
            prev_up = self.prev_proj(prev_up)
        if self.include_backbone_in_fusion:
            fused = film_features
            if prev_up is not None:
                fused = fused + self.prev_scale * prev_up
        else:
            if prev_up is None:
                raise ValueError(
                    "A decoder stage without backbone fusion requires a previous stage"
                )
            fused = self.prev_scale * prev_up
        decoded = self.refine2(self.refine1(fused))
        mask_outputs: Dict[str, Any] = {}
        if self.priority_mode in ("pairwise", "unary_mask"):
            self._assert_required_visual_prototypes(visual_validity_mask)
            visual_out = self._resolve_visual_activations(
                film_features[:, :, -1:],
                active_visual_prototypes,
                visual_validity_mask,
                concept_out,
            )
            priority_out = self._prioritize_concept_patch_activations([visual_out])
            priority_mask_last = priority_out["priority_mask"]
            expected_mask_shape = (
                decoded.shape[0],
                1,
                1,
                decoded.shape[-2],
                decoded.shape[-1],
            )
            if tuple(priority_mask_last.shape) != expected_mask_shape:
                raise ValueError(
                    f"Last-frame priority mask must be {expected_mask_shape}, got {tuple(priority_mask_last.shape)}"
                )
            mask_strength = (
                self.prototype_bottleneck_strength
                if self.priority_mode == "pairwise"
                else self.fine_unary_mask_strength
            )
            decoded_base_last = decoded[:, :, -1:]
            mask_multiplier = 1.0 - mask_strength + mask_strength * priority_mask_last
            guided_last = decoded_base_last * mask_multiplier
            decoded = _scatter_last_frame_update(decoded, guided_last)
            priority_mask = _scatter_last_frame_mask(
                decoded.shape[0],
                int(decoded.shape[2]),
                priority_mask_last,
                device=decoded.device,
                dtype=decoded.dtype,
            )
            base_norm = decoded_base_last.detach().norm()
            guided_norm = guided_last.detach().norm()
            norm_ratio = guided_norm / base_norm.clamp_min(1e-06)
            mask_outputs.update(priority_out)
            mask_outputs.update(
                {
                    "priority_mask": priority_mask,
                    "concept_patch_activations": visual_out["activations"],
                    "prototype_application_position": "post_refine_spatial_mask",
                    "spatial_mask_strength": film_features.new_tensor(
                        mask_strength
                    ).detach(),
                    "decoded_base_last_norm": base_norm,
                    "guided_last_norm": guided_norm,
                    "guided_to_base_norm_ratio": norm_ratio,
                    "prototype_bottleneck_strength": film_features.new_tensor(
                        mask_strength
                    ).detach(),
                    "prototype_branch_norm": guided_norm,
                    "film_last_norm": base_norm,
                    "prototype_to_base_norm_ratio": norm_ratio,
                    "prototype_to_film_norm_ratio": norm_ratio,
                }
            )
            if self.priority_mode == "unary_mask":
                mask_outputs["fine_unary_mask_strength"] = film_features.new_tensor(
                    self.fine_unary_mask_strength
                ).detach()
            if torch.is_tensor(visual_out.get("similarity_maps")):
                mask_outputs["visual_similarity_maps"] = visual_out["similarity_maps"]
        return (decoded, mask_outputs)

    def _resolve_visual_activations(
        self,
        film_last: torch.Tensor,
        active_visual_prototypes: torch.Tensor,
        visual_validity_mask: torch.Tensor,
        concept_out: Optional[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        if self.use_shared_concept_activations:
            if concept_out is None:
                raise ValueError(
                    "concept_out is required when use_shared_concept_activations=True"
                )
            return _build_shared_visual_modality_from_concept_out(
                concept_out, concept_proto_proj=self.concept_proto_proj
            )
        return self._compute_soft_activations(
            film_last, active_visual_prototypes, visual_validity_mask
        )


class ConceptGatedMultiScaleSaliencyDecoder(nn.Module):
    """Coarse-to-fine decoder with the same residual mask path at every stage."""

    def __init__(
        self,
        stage_channels: Dict[str, int],
        concept_dim: int = 256,
        decoder_channels: int = 96,
        dropout: float = 0.05,
        output_activation: str = "sigmoid",
        use_side_logit_fusion: bool = True,
        use_shared_concept_activations: bool = False,
        assignment_temperature: float = 0.07,
        priority_temperature: float = 0.1,
        factorized_rank: int = 32,
        prototype_bottleneck_strength: float = 1.0,
        fine_unary_mask_strength: float = 0.6,
        stage2_backbone_fusion_enabled: bool = True,
    ):
        super().__init__()
        if output_activation not in ("sigmoid", "none"):
            raise ValueError("output_activation must be 'sigmoid' or 'none'")
        if not 0.0 <= prototype_bottleneck_strength <= 1.0:
            raise ValueError(
                f"prototype_bottleneck_strength must be in [0, 1], got {prototype_bottleneck_strength}"
            )
        if not 0.0 <= fine_unary_mask_strength <= 1.0:
            raise ValueError(
                f"fine_unary_mask_strength must be in [0, 1], got {fine_unary_mask_strength}"
            )
        self.stage_channels = dict(stage_channels)
        self.concept_dim = concept_dim
        self.output_activation = output_activation
        self.temporal_aggregation = "last"
        self.use_side_logit_fusion = bool(use_side_logit_fusion)
        self.use_shared_concept_activations = bool(use_shared_concept_activations)
        self.prototype_bottleneck_strength = float(prototype_bottleneck_strength)
        self.prototype_application_position = "post_refine"
        self.fine_unary_mask_strength = float(fine_unary_mask_strength)
        self.stage2_backbone_fusion_enabled = bool(stage2_backbone_fusion_enabled)
        self.pairwise_priority_stages = tuple(
            stage for stage in _PAIRWISE_PRIORITY_STAGES if stage in self.stage_channels
        )
        self.unary_mask_stages = tuple(
            stage for stage in _UNARY_MASK_STAGES if stage in self.stage_channels
        )
        hidden_channels = max(decoder_channels // 2, 32)

        def _priority_mode_for_stage(stage: str) -> str:
            if stage in _PAIRWISE_PRIORITY_STAGES:
                return "pairwise"
            if stage in _UNARY_MASK_STAGES:
                return "unary_mask"
            return "none"

        self.fusion_blocks = nn.ModuleDict(
            {
                stage: SpatioTemporalConceptGatedFusionBlock(
                    feature_channels=channels,
                    concept_dim=concept_dim,
                    decoder_channels=decoder_channels,
                    dropout=dropout,
                    assignment_temperature=assignment_temperature,
                    priority_temperature=priority_temperature,
                    priority_mode=_priority_mode_for_stage(stage),
                    use_shared_concept_activations=use_shared_concept_activations,
                    factorized_rank=factorized_rank,
                    prototype_bottleneck_strength=self.prototype_bottleneck_strength,
                    fine_unary_mask_strength=self.fine_unary_mask_strength,
                    include_backbone_in_fusion=True
                    if stage != "stage2"
                    else self.stage2_backbone_fusion_enabled,
                )
                for stage, channels in self.stage_channels.items()
            }
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
            return (main_logits, main_logits, None)
        weights = torch.softmax(self.side_fusion_logits[: len(logits_to_fuse)], dim=0)
        fused_logits = sum(
            (weight * logit for weight, logit in zip(weights, logits_to_fuse))
        )
        return (fused_logits, main_logits, weights.detach())

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

    @staticmethod
    def _last_decoder_slice(decoded_volume: torch.Tensor) -> torch.Tensor:
        """Keep the original last-slice readout; no temporal averaging."""
        if decoded_volume.dim() != 5:
            raise ValueError(
                f"decoded_volume must be [B,D,T,H,W], got {tuple(decoded_volume.shape)}"
            )
        return decoded_volume[:, :, -1]

    def forward(
        self,
        concept_outs: Dict[str, Dict[str, Any]],
        features_dict: Dict[str, torch.Tensor],
        output_size: Tuple[int, int],
        return_details: bool = False,
    ) -> Dict[str, Any]:
        stages = self._ordered_stages(concept_outs, features_dict)
        if not stages:
            raise ValueError(
                "No overlapping stages found in concept_outs and features_dict"
            )
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
        stage_patch_priority_distributions: Dict[str, torch.Tensor] = {}
        stage_patch_priority_gates: Dict[str, torch.Tensor] = {}
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
                    f"features_dict[{stage}] must be [B,C,T,H,W], got {tuple(features_5d.shape)}"
                )
            concept_volume = _build_stage_concept_volume(
                concept_outs[stage], stage=stage
            )
            concept_volume = _align_concept_volume_to_features(
                concept_volume, features_5d, stage=stage
            )
            _assert_concept_volume_matches_features(
                stage=stage,
                concept_volume=concept_volume,
                features_5d=features_5d,
                concept_out=concept_outs[stage],
            )
            stage_concept_out = concept_outs[stage]
            active_visual_prototypes = stage_concept_out.get("active_visual_prototypes")
            visual_validity_mask = stage_concept_out.get("visual_validity_mask")
            if not torch.is_tensor(active_visual_prototypes) or not torch.is_tensor(
                visual_validity_mask
            ):
                raise ValueError(
                    f"concept_out at {stage} must provide active_visual_prototypes and visual_validity_mask"
                )
            decoded, mask_outputs = fusion_blocks[stage](
                features_5d,
                concept_volume,
                prev_decoder=prev_decoder,
                active_visual_prototypes=active_visual_prototypes,
                visual_validity_mask=visual_validity_mask,
                concept_out=stage_concept_out
                if self.use_shared_concept_activations
                else None,
            )
            if "patch_priority_distribution" in mask_outputs:
                stage_patch_priority_distributions[stage] = mask_outputs[
                    "patch_priority_distribution"
                ]
                stage_patch_priority_maps[stage] = mask_outputs[
                    "patch_priority_distribution"
                ]
            elif "patch_priority_map" in mask_outputs:
                stage_patch_priority_maps[stage] = mask_outputs["patch_priority_map"]
            if "patch_priority_gate" in mask_outputs:
                stage_patch_priority_gates[stage] = mask_outputs["patch_priority_gate"]
            prev_decoder = decoded
            side_volume = side_feature_heads[stage](decoded)
            side_feature_2d = self._last_decoder_slice(side_volume)
            side_patch_logits[stage] = side_patch_heads[stage](side_feature_2d)
            side_saliency_logits[stage] = side_upsample_heads[stage](
                side_feature_2d, target_hw=output_size
            )
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
                visual_similarity_maps = mask_outputs.get("visual_similarity_maps")
                if torch.is_tensor(visual_similarity_maps):
                    stage_visual_similarity_maps[stage] = (
                        visual_similarity_maps.detach()
                    )
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
                            else:
                                stage_factorized_comparison_stats[stage][key] = float(
                                    value.detach().mean().cpu()
                                )
                        else:
                            stage_factorized_comparison_stats[stage][key] = float(value)
                with torch.no_grad():
                    if torch.is_tensor(priority_mask):
                        bottleneck = mask_outputs.get("prototype_bottleneck_strength")
                        if torch.is_tensor(bottleneck):
                            bottleneck_value = float(bottleneck.detach().cpu())
                        else:
                            bottleneck_value = float(
                                getattr(
                                    fusion_blocks[stage],
                                    "prototype_bottleneck_strength",
                                    float("nan"),
                                )
                            )
                        ratio = mask_outputs.get("prototype_to_base_norm_ratio")
                        if not torch.is_tensor(ratio):
                            ratio = mask_outputs.get("prototype_to_film_norm_ratio")
                        ratio_value = (
                            float(ratio.detach().cpu())
                            if torch.is_tensor(ratio)
                            else float("nan")
                        )
                        position = mask_outputs.get(
                            "prototype_application_position",
                            getattr(
                                fusion_blocks[stage],
                                "prototype_application_position",
                                "pre_refine",
                            ),
                        )
                        base_norm = mask_outputs.get("decoded_base_last_norm")
                        base_norm_value = (
                            float(base_norm.detach().cpu())
                            if torch.is_tensor(base_norm)
                            else float("nan")
                        )
                        branch_norm = mask_outputs.get("prototype_branch_norm")
                        branch_norm_value = (
                            float(branch_norm.detach().cpu())
                            if torch.is_tensor(branch_norm)
                            else float("nan")
                        )
                        stage_mask_diagnostics[stage] = {
                            "prototype_application_position": str(position),
                            "prototype_bottleneck_strength": bottleneck_value,
                            "decoded_base_last_norm": base_norm_value,
                            "prototype_branch_norm": branch_norm_value,
                            "prototype_to_base_norm_ratio": ratio_value,
                            "prototype_to_film_norm_ratio": ratio_value,
                            "film_last_norm": base_norm_value,
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
                            "priority_mode": str(
                                mask_outputs.get(
                                    "priority_mode",
                                    getattr(
                                        fusion_blocks[stage], "priority_mode", "none"
                                    ),
                                )
                            ),
                            "pairwise_context_used": bool(
                                mask_outputs.get(
                                    "pairwise_context_used",
                                    getattr(
                                        fusion_blocks[stage], "priority_mode", "none"
                                    )
                                    == "pairwise",
                                )
                            ),
                        }
        if prev_decoder is None:
            raise RuntimeError("Decoder produced no stage outputs")
        final_feature_2d = self._last_decoder_slice(prev_decoder)
        patch_logits = self.patch_logit_head(final_feature_2d)
        main_saliency_logits_unfused = F.interpolate(
            patch_logits, size=output_size, mode="bilinear", align_corners=False
        )
        saliency_logits, _, side_fusion_weights = self._fuse_main_and_side_logits(
            main_saliency_logits_unfused, side_saliency_logits, stages
        )
        if self.output_activation == "sigmoid":
            saliency_map = torch.sigmoid(saliency_logits)
        else:
            saliency_map = saliency_logits
        decoder_temporal_diagnostics = {
            "temporal_aggregation": "last",
            "side_stages": [],
        }
        out: Dict[str, Any] = {
            "saliency_logits": saliency_logits,
            "saliency_map": saliency_map,
            "patch_saliency_logits": patch_logits,
            "coarse_saliency_logits": saliency_logits,
            "main_saliency_logits_unfused": main_saliency_logits_unfused,
            "side_fusion_weights": side_fusion_weights,
            "side_saliency_logits": side_saliency_logits,
            "side_patch_saliency_logits": side_patch_logits,
            "temporal_weights": None,
            "side_temporal_weights": side_temporal_weights,
            "output_activation": self.output_activation,
            "decoder_temporal_diagnostics": decoder_temporal_diagnostics,
            "stage_patch_priority_maps": stage_patch_priority_maps,
            "stage_patch_priority_distributions": stage_patch_priority_distributions,
            "stage_patch_priority_gates": stage_patch_priority_gates,
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


# Frozen reference backbone feature cache and provenance

CACHE_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k/cached_feats")

DEFAULT_REFERENCE_CHECKPOINT = VideoSwinTransformer.DEFAULT_PRETRAINED_PATH

SCHEMA = "frozen_backbone_last_slice_v2"

SCHEMA_VERSION = 2

LEGACY_ENTRY_VERSION = 1

STAGES = ("stage1", "stage2", "stage3", "stage4")

STAGE_CHANNELS = {"stage1": 96, "stage2": 192, "stage3": 384, "stage4": 768}

DEFAULT_RESIZE_TO = (224, 384)

DEFAULT_MEAN = (0.485, 0.456, 0.406)

DEFAULT_STD = (0.229, 0.224, 0.225)

LEGACY_FEATURES_REFERENCE_SHA256 = (
    "07311026a139e72085344fdc6686b9a6d9ee9af700de268b5483a734dc74acfd"
)


def ensure_cache_root(root: Union[str, Path] = CACHE_ROOT) -> Path:
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "features").mkdir(parents=True, exist_ok=True)
    (path / "v2").mkdir(parents=True, exist_ok=True)
    return path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(
            json.dumps(_json_ready(obj), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_state_dict(state: Mapping[str, torch.Tensor]) -> str:
    """Canonical SHA-256 of a backbone (or module) state_dict."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if not torch.is_tensor(tensor):
            continue
        array = (
            tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
        )
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


REFERENCE_FEATURE_PROVENANCE_SCHEMA = "reference_backbone_features_v1"


def extract_backbone_state_dict_from_checkpoint(
    checkpoint_path: Union[str, Path],
) -> Dict[str, torch.Tensor]:
    """Load backbone tensors from a Kinetics .pth or full model checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        isinstance(payload, dict)
        and any(torch.is_tensor(v) for v in payload.values())
        and (not any(k in payload for k in ("model_state_dict", "state_dict", "model")))
    ):
        raw = payload
        source = "raw_state_dict"
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        raw = payload["model_state_dict"]
        source = "model_state_dict"
    elif isinstance(payload, dict) and "state_dict" in payload:
        raw = payload["state_dict"]
        source = "state_dict"
    elif (
        isinstance(payload, dict)
        and "model" in payload
        and isinstance(payload["model"], dict)
    ):
        raw = payload["model"]
        source = "model"
    elif isinstance(payload, dict):
        raw = payload
        source = "dict"
    else:
        raise ValueError(f"Unrecognized checkpoint format at {path}")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Checkpoint {path} does not contain a state_dict mapping.")
    backbone_prefixed = {
        k: v
        for k, v in raw.items()
        if isinstance(k, str) and k.startswith("backbone.") and torch.is_tensor(v)
    }
    if backbone_prefixed:
        return dict(backbone_prefixed)
    inner = {k: v for k, v in raw.items() if torch.is_tensor(v) and isinstance(k, str)}
    if not inner:
        raise ValueError(f"No tensor weights found in {path} (parsed as {source}).")
    return {f"backbone.{k}": v for k, v in inner.items()}


def backbone_fingerprint_from_checkpoint(checkpoint_path: Union[str, Path]) -> str:
    """Fingerprint backbone weights as they appear after the same load path as the cache."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        reference = build_reference_backbone(checkpoint=str(path), device="cpu")
        return fingerprint_state_dict(reference.state_dict())
    except Exception:
        pass
    backbone_state = extract_backbone_state_dict_from_checkpoint(path)
    reference = build_reference_backbone(
        checkpoint=DEFAULT_REFERENCE_CHECKPOINT, device="cpu"
    )
    incompatible = reference.load_state_dict(backbone_state, strict=False)
    if incompatible.missing_keys:
        raise ValueError(
            f"Could not inject backbone weights from {path} into VideoSwinTransformer; missing={incompatible.missing_keys[:8]}"
        )
    return fingerprint_state_dict(reference.state_dict())


def build_reference_feature_provenance(
    *,
    backbone_fingerprint: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    weight_scope: str,
    preprocess: Optional[Mapping[str, Any]] = None,
    temporal_selection: str = "last_backbone_step",
    feature_module: str = "backbone",
    feature_output_path: Sequence[Any] = ("stage4",),
    feature_layout: str = "BCTHW_last_slice",
    spatial_grid: str = "native_stage4_HxW_matching_concept_grid",
    verified: bool = True,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Structured provenance for fixed-cluster / cache feature agreement."""
    payload: Dict[str, Any] = {
        "schema": REFERENCE_FEATURE_PROVENANCE_SCHEMA,
        "backbone_class": "model.backbones.video_swin_custom.VideoSwinTransformer",
        "backbone_fingerprint": str(backbone_fingerprint),
        "weight_scope": str(weight_scope),
        "temporal_selection": str(temporal_selection),
        "feature_module": str(feature_module),
        "feature_output_path": list(feature_output_path),
        "feature_layout": str(feature_layout),
        "spatial_grid": str(spatial_grid),
        "preprocess": dict(preprocess or preprocess_config()),
        "verified": bool(verified),
        "source_checkpoint_path": None
        if checkpoint_path is None
        else str(checkpoint_path),
        "source_checkpoint_sha256": None
        if checkpoint_sha256 is None
        else str(checkpoint_sha256),
        "checkpoint_file_hash_is_not_backbone_identity": True,
    }
    if notes:
        payload["notes"] = str(notes)
    return payload


def provenance_from_cache(cache: "FrozenBackboneFeatureCache") -> Dict[str, Any]:
    """Build provenance describing the active FrozenBackboneFeatureCache reference."""
    return build_reference_feature_provenance(
        backbone_fingerprint=cache.reference_fingerprint,
        checkpoint_path=cache.checkpoint_path,
        checkpoint_sha256=cache.checkpoint_sha256,
        weight_scope="FrozenBackboneFeatureCache.reference.state_dict",
        preprocess=cache.preprocess,
        temporal_selection="last_backbone_step",
        feature_module="backbone",
        feature_output_path=list(STAGES),
        feature_layout="BCTHW_last_slice_per_stage",
        spatial_grid="native_stage_resolutions_after_resize_to",
        verified=True,
        notes="Training queries must use this fingerprint. Flipped/padded views are content-addressed separately; never reuse identity-view features for hflip.",
    )


def compare_reference_feature_provenance(
    cluster_provenance: Optional[Mapping[str, Any]],
    cache_provenance: Mapping[str, Any],
    *,
    context: str = "stage4 fixed-cluster feature supervision",
) -> None:
    """Raise if cluster and cache reference features cannot be verified as identical."""
    if not cluster_provenance or not isinstance(cluster_provenance, Mapping):
        raise RuntimeError(
            f"{context}: reference-feature provenance is UNVERIFIED.\nThe fixed patch-cluster bank does not record a backbone_fingerprint (state_dict hash). A whole-model checkpoint file SHA alone is not enough — it may include non-backbone weights and cannot be compared to FrozenBackboneFeatureCache.reference_fingerprint.\nFeature dimension matching is not sufficient.\n\nRegenerate the fixed cluster bank with recorded provenance, using the SAME frozen reference backbone as training (Kinetics / {DEFAULT_REFERENCE_CHECKPOINT}):\n  Recommended (no prototype graph):\n    python patch_clusters/form_patch_clusters.py --source-mode cache --output-dir /data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster_frozen_ref --stride 1 --num-prototypes 256\n  Then train with:\n    python train.py --phase initial --visual-cluster-dir /data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster_frozen_ref\n  Optional legacy path: rebuild graph descriptors with DESCRIPTOR_FEATURE_SOURCE='frozen_reference', then form_patch_clusters.py --source-mode graph --source-db <graph.sqlite> into a NEW output dir. Do not overwrite existing banks/caches."
        )
    cluster_fp = cluster_provenance.get("backbone_fingerprint")
    cache_fp = cache_provenance.get("backbone_fingerprint")
    if not cluster_fp or not cache_fp:
        raise RuntimeError(
            f"{context}: provenance present but backbone_fingerprint missing on cluster={bool(cluster_fp)} cache={bool(cache_fp)}. Regenerate the cluster bank / rebuild the cache with reference_feature_provenance."
        )
    if str(cluster_fp) != str(cache_fp):
        raise RuntimeError(
            f"{context}: CONFIRMED MISMATCH between fixed-cluster features and FrozenBackboneFeatureCache.\n  cluster backbone_fingerprint={cluster_fp}\n  cache   backbone_fingerprint={cache_fp}\n  cluster checkpoint={cluster_provenance.get('source_checkpoint_path')}\n  cache   checkpoint={cache_provenance.get('source_checkpoint_path')}\nDo not compare whole-model checkpoint file hashes to these fingerprints.\nFix: rebuild EITHER the graph descriptors+clusters OR the feature cache so both use one fixed reference backbone (recommended: Kinetics {DEFAULT_REFERENCE_CHECKPOINT}). Do not overwrite existing banks/caches — write to a new directory / new fingerprint namespace.\nExample:\n  python cache_backbone_features.py --split train --reference-checkpoint {DEFAULT_REFERENCE_CHECKPOINT}\n  # and regenerate clusters from a graph whose reference_feature_provenance.backbone_fingerprint matches the cache."
        )
    c_pre = cluster_provenance.get("preprocess") or {}
    q_pre = cache_provenance.get("preprocess") or {}
    for key in ("resize_to", "normalize", "input_format", "mean", "std"):
        if key in c_pre and key in q_pre and (c_pre[key] != q_pre[key]):
            raise RuntimeError(
                f"{context}: backbone fingerprints match but preprocess[{key!r}] differs (cluster={c_pre[key]!r}, cache={q_pre[key]!r}). Align resize/normalize/mean/std before training."
            )
    c_temp = cluster_provenance.get("temporal_selection")
    q_temp = cache_provenance.get("temporal_selection")
    if c_temp and q_temp and (c_temp != q_temp):
        raise RuntimeError(
            f"{context}: temporal_selection differs (cluster={c_temp!r}, cache={q_temp!r})."
        )


def fingerprint_rgb_clip(rgb_btchw: torch.Tensor) -> str:
    """Content fingerprint of a single BTCHW float clip (pre-backbone resize)."""
    if rgb_btchw.dim() != 4 or rgb_btchw.shape[1] != 3:
        raise ValueError(
            f"rgb clip must have shape [T,3,H,W], got {tuple(rgb_btchw.shape)}"
        )
    array = (
        rgb_btchw.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    )
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fingerprint_identity_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def preprocess_config(
    *,
    resize_to: Optional[Union[int, Tuple[int, int]]] = DEFAULT_RESIZE_TO,
    normalize: bool = True,
    input_format: str = "BTCHW",
    output_format: str = "BCTHW",
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> Dict[str, Any]:
    if isinstance(resize_to, int):
        resize_value: Any = int(resize_to)
    elif resize_to is None:
        resize_value = None
    else:
        resize_value = [int(resize_to[0]), int(resize_to[1])]
    return {
        "input_format": str(input_format),
        "output_format": str(output_format),
        "resize_to": resize_value,
        "normalize": bool(normalize),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "gradient_checkpointing": False,
        "freeze_backbone": True,
        "return_stages": list(STAGES),
    }


@dataclass(frozen=True)
class SampleIdentity:
    """Exact sample/view identity for cache keys."""

    dataset_root: str
    video_id: str
    window_start: int
    frame_names: Tuple[str, ...]
    window_len: int
    content_sha256: str
    split: Optional[str] = None
    temporal_sampling: Mapping[str, Any] = field(
        default_factory=lambda: {"mode": "contiguous", "stride": 1}
    )
    view_id: str = "identity"
    input_shape: Optional[Tuple[int, int, int, int]] = None
    padded_input_geometry: Optional[Mapping[str, Any]] = None
    preprocess: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_root": str(Path(self.dataset_root).expanduser().resolve()),
            "split": self.split,
            "video_id": str(self.video_id),
            "window_start": int(self.window_start),
            "frame_names": list(self.frame_names),
            "window_len": int(self.window_len),
            "temporal_sampling": dict(self.temporal_sampling),
            "view_id": str(self.view_id),
            "content_sha256": str(self.content_sha256),
            "input_shape": list(self.input_shape)
            if self.input_shape is not None
            else None,
            "padded_input_geometry": dict(self.padded_input_geometry)
            if self.padded_input_geometry is not None
            else None,
            "preprocess": dict(self.preprocess)
            if self.preprocess is not None
            else None,
        }

    def identity_sha256(self) -> str:
        return fingerprint_identity_payload(self.to_dict())


def build_reference_backbone(
    *,
    checkpoint: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    resize_to: Optional[Union[int, Tuple[int, int]]] = DEFAULT_RESIZE_TO,
    normalize: bool = True,
) -> VideoSwinTransformer:
    """Independent frozen VideoSwin used only for cache generation."""
    ckpt = DEFAULT_REFERENCE_CHECKPOINT if checkpoint is None else str(checkpoint)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Reference checkpoint not found: {ckpt}")
    backbone = VideoSwinTransformer(
        pretrained=ckpt,
        pretrained_path=ckpt,
        freeze_backbone=True,
        return_stages=STAGES,
        input_format="BTCHW",
        output_format="BCTHW",
        resize_to=resize_to,
        normalize=normalize,
        gradient_checkpointing=False,
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if device is not None:
        backbone = backbone.to(device)
    return backbone


def _states_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a = left[key]
        b = right[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.equal(a.cpu(), b.cpu()):
            return False
    return True


def _feature_path(root: Path, content_sha256: str) -> Path:
    return root / "features" / content_sha256[:2] / f"{content_sha256}.pt"


def _validate_feature_pack(
    features: Mapping[str, torch.Tensor], *, storage_dtype: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    if set(features) != set(STAGES):
        raise ValueError(
            f"Feature pack stages {sorted(features)} != expected {list(STAGES)}"
        )
    out: Dict[str, torch.Tensor] = {}
    for stage in STAGES:
        tensor = features[stage]
        if not torch.is_tensor(tensor):
            raise TypeError(f"{stage} feature must be a tensor.")
        if tensor.dim() != 4:
            raise ValueError(f"{stage} must be [C,1,H,W], got {tuple(tensor.shape)}")
        channels, temporal, _, _ = tensor.shape
        if channels != STAGE_CHANNELS[stage]:
            raise ValueError(
                f"{stage} expected {STAGE_CHANNELS[stage]} channels, got {channels}"
            )
        if temporal != 1:
            raise ValueError(
                f"{stage} must keep only the last temporal slice (T=1), got T={temporal}"
            )
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"{stage} contains non-finite values.")
        if storage_dtype is not None:
            expected = {"float16": torch.float16, "float32": torch.float32}[
                storage_dtype
            ]
            if tensor.dtype != expected:
                if tensor.dtype not in (torch.float16, torch.float32):
                    raise ValueError(f"{stage} dtype {tensor.dtype} is unsupported.")
        out[stage] = tensor.detach()
    return out


class FrozenBackboneFeatureCache:
    """Load/generate last-slice reference features outside Dataset workers."""

    def __init__(
        self,
        *,
        root: Union[str, Path] = CACHE_ROOT,
        reference_backbone: Optional[VideoSwinTransformer] = None,
        checkpoint: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        storage_dtype: str = "float16",
        resize_to: Optional[Union[int, Tuple[int, int]]] = DEFAULT_RESIZE_TO,
        normalize: bool = True,
        create_reference_if_missing: bool = True,
    ):
        if storage_dtype not in ("float16", "float32"):
            raise ValueError("storage_dtype must be 'float16' or 'float32'")
        self.root = ensure_cache_root(root)
        self.storage_dtype = storage_dtype
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.preprocess = preprocess_config(resize_to=resize_to, normalize=normalize)
        self.checkpoint_path = str(
            Path(DEFAULT_REFERENCE_CHECKPOINT if checkpoint is None else checkpoint)
            .expanduser()
            .resolve()
        )
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        if reference_backbone is None:
            reference_backbone = build_reference_backbone(
                checkpoint=self.checkpoint_path,
                device=self.device,
                resize_to=resize_to,
                normalize=normalize,
            )
        else:
            reference_backbone = reference_backbone.to(self.device)
            reference_backbone.eval()
            for parameter in reference_backbone.parameters():
                parameter.requires_grad = False
        self.reference = reference_backbone
        self.reference_state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.reference.state_dict().items()
            if torch.is_tensor(value)
        }
        self.reference_fingerprint = fingerprint_state_dict(self.reference_state)
        self.legacy_reference_sha256 = self._adopt_legacy_reference_fingerprint()
        self.namespace_dir = self.root / "v2" / self.reference_fingerprint
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.namespace_dir / "manifest.json"
        self.identity_index_path = self.namespace_dir / "identity_index.jsonl"
        if create_reference_if_missing:
            self._write_reference_snapshot()
            self._write_manifest()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "regenerated": 0,
            "legacy_hits": 0,
            "writes": 0,
            "rejected": 0,
            "reference_forwards": 0,
        }

    def _adopt_legacy_reference_fingerprint(self) -> Optional[str]:
        """Find the reference hash string used by reusable features/*.pt packs."""
        path = self.root / "reference_backbone.pt"
        candidate_hashes: List[str] = []
        if path.is_file():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                payload = None
            if isinstance(payload, dict) and "state" in payload:
                legacy_state = {
                    key: value.detach().cpu().contiguous()
                    for key, value in payload["state"].items()
                    if torch.is_tensor(value)
                }
                shared = {
                    key: legacy_state[key]
                    for key in legacy_state
                    if key in self.reference_state
                }
                current = {key: self.reference_state[key] for key in shared}
                if shared and _states_equal(shared, current):
                    missing = set(self.reference_state) - set(shared)
                    if not any(key.startswith("backbone.") for key in missing):
                        sha = payload.get("state_sha256")
                        if sha:
                            candidate_hashes.append(str(sha))
                        cfg = payload.get("config") or {}
                        legacy_cfg = cfg.get("legacy_reference_sha256")
                        if legacy_cfg:
                            candidate_hashes.append(str(legacy_cfg))
        if (
            Path(self.checkpoint_path).resolve()
            == Path(DEFAULT_REFERENCE_CHECKPOINT).resolve()
        ):
            candidate_hashes.append(LEGACY_FEATURES_REFERENCE_SHA256)
        if LEGACY_FEATURES_REFERENCE_SHA256 in candidate_hashes:
            return LEGACY_FEATURES_REFERENCE_SHA256
        return candidate_hashes[0] if candidate_hashes else None

    def _reference_config(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "tag": "v2",
            "backbone_class": "model.backbones.video_swin_custom.VideoSwinTransformer",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "reference_fingerprint": self.reference_fingerprint,
            "legacy_reference_sha256": self.legacy_reference_sha256,
            "stages": list(STAGES),
            "stage_channels": dict(STAGE_CHANNELS),
            "temporal_selection": "last_backbone_step",
            "storage_dtype": self.storage_dtype,
            "preprocess": self.preprocess,
            "concept_grid_caps": {stage: None for stage in STAGES},
        }

    def _write_reference_snapshot(self) -> None:
        """Write namespace snapshot; never clobber a compatible root snapshot."""
        state = dict(self.reference_state)
        if hasattr(self.reference, "mean"):
            state["mean"] = self.reference.mean.detach().cpu().contiguous()
        if hasattr(self.reference, "std"):
            state["std"] = self.reference.std.detach().cpu().contiguous()
        payload = {
            "config": {
                **self._reference_config(),
                "input_format": self.preprocess["input_format"],
                "normalize": self.preprocess["normalize"],
                "resize_to": self.preprocess["resize_to"],
                "reference_precision": "float32",
            },
            "state": state,
            "state_sha256": self.legacy_reference_sha256 or self.reference_fingerprint,
        }
        _atomic_torch_save(payload, self.namespace_dir / "reference_snapshot.pt")
        root_path = self.root / "reference_backbone.pt"
        if root_path.is_file():
            return
        _atomic_torch_save(payload, root_path)

    def _write_manifest(self) -> None:
        manifest = {
            **self._reference_config(),
            "cache_root": str(self.root),
            "features_dir": str(self.root / "features"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": "Features are raw frozen-backbone outputs for the last temporal slice only: [C,1,H,W] per stage. No L2-normalization or projection. Exhaustive stride-1 caching of all four stages is large.",
        }
        _atomic_write_json(manifest, self.manifest_path)

    def _accepted_reference_hashes(self) -> set[str]:
        hashes = {self.reference_fingerprint}
        if self.legacy_reference_sha256:
            hashes.add(self.legacy_reference_sha256)
        return hashes

    def _cast_for_storage(
        self, features: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        dtype = torch.float16 if self.storage_dtype == "float16" else torch.float32
        return {
            stage: features[stage].detach().to(dtype=dtype, device="cpu").contiguous()
            for stage in STAGES
        }

    def _features_to_float32(
        self, features: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            stage: features[stage].detach().to(dtype=torch.float32).contiguous()
            for stage in STAGES
        }

    def entry_path(self, content_sha256: str) -> Path:
        return _feature_path(self.root, content_sha256)

    def load_entry(
        self, content_sha256: str, *, require_identity: Optional[SampleIdentity] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        path = self.entry_path(content_sha256)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                self.stats["rejected"] += 1
                return None
        except Exception:
            self.stats["rejected"] += 1
            return None
        if not isinstance(payload, dict):
            self.stats["rejected"] += 1
            return None
        try:
            if str(payload.get("input_sha256")) != str(content_sha256):
                raise ValueError("content hash mismatch inside entry")
            if (
                str(payload.get("reference_sha256"))
                not in self._accepted_reference_hashes()
            ):
                raise ValueError("reference fingerprint mismatch")
            features = _validate_feature_pack(payload.get("features") or {})
            if require_identity is not None:
                stored = payload.get("identity")
                if not isinstance(stored, dict):
                    pass
                else:
                    expected = require_identity.to_dict()
                    for key in (
                        "dataset_root",
                        "video_id",
                        "window_start",
                        "frame_names",
                        "window_len",
                        "view_id",
                        "content_sha256",
                    ):
                        if key in stored and stored[key] != expected[key]:
                            raise ValueError(f"identity field mismatch: {key}")
        except Exception:
            self.stats["rejected"] += 1
            return None
        version = int(payload.get("version", LEGACY_ENTRY_VERSION))
        if version == LEGACY_ENTRY_VERSION:
            self.stats["legacy_hits"] += 1
        self.stats["hits"] += 1
        return self._features_to_float32(features)

    def save_entry(
        self,
        content_sha256: str,
        features: Mapping[str, torch.Tensor],
        *,
        input_shape: Sequence[int],
        identity: Optional[SampleIdentity] = None,
        overwrite: bool = False,
    ) -> Path:
        path = self.entry_path(content_sha256)
        if path.is_file() and (not overwrite):
            existing = self.load_entry(content_sha256, require_identity=identity)
            if existing is not None and identity is not None:
                self._append_identity_index(identity, content_sha256)
            return path
        validated = _validate_feature_pack(features)
        stored = self._cast_for_storage(validated)
        payload = {
            "version": SCHEMA_VERSION,
            "schema": SCHEMA,
            "input_sha256": content_sha256,
            "reference_sha256": self.legacy_reference_sha256
            or self.reference_fingerprint,
            "reference_fingerprint": self.reference_fingerprint,
            "input_shape": [int(x) for x in input_shape],
            "input_dtype": "torch.float32",
            "storage_dtype": self.storage_dtype,
            "features": stored,
            "identity": identity.to_dict() if identity is not None else None,
            "provenance": {
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "preprocess": self.preprocess,
                "temporal_selection": "last_backbone_step",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        _atomic_torch_save(payload, path)
        if identity is not None:
            self._append_identity_index(identity, content_sha256)
        self.stats["writes"] += 1
        return path

    def _append_identity_index(
        self, identity: SampleIdentity, content_sha256: str
    ) -> None:
        record = {
            "identity_sha256": identity.identity_sha256(),
            "content_sha256": content_sha256,
            "identity": identity.to_dict(),
        }
        line = json.dumps(_json_ready(record), sort_keys=True) + "\n"
        with open(self.identity_index_path, "a", encoding="utf-8") as handle:
            handle.write(line)

    @torch.no_grad()
    def compute_last_slice_features(
        self, rgb_btchw: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run frozen reference backbone; keep only the last temporal slice."""
        if rgb_btchw.dim() == 4:
            video = rgb_btchw.unsqueeze(0)
        elif rgb_btchw.dim() == 5:
            video = rgb_btchw
        else:
            raise ValueError(
                f"Expected [T,3,H,W] or [B,T,3,H,W] RGB, got {tuple(rgb_btchw.shape)}"
            )
        if video.shape[2] != 3:
            raise ValueError(
                f"Expected BTCHW with 3 channels, got {tuple(video.shape)}"
            )
        video = video.detach().to(device=self.device, dtype=torch.float32)
        if video.numel() > 0 and float(video.max()) > 2.0:
            video = video / 255.0
        self.stats["reference_forwards"] += 1
        outputs = self.reference.forward_features(video)
        packed: Dict[str, torch.Tensor] = {}
        for stage in STAGES:
            feat = outputs[stage]
            if feat.dim() != 5:
                raise ValueError(
                    f"{stage} expected [B,C,T,H,W], got {tuple(feat.shape)}"
                )
            last = feat[:, :, -1:, :, :].detach()
            if last.shape[0] != 1:
                raise ValueError(
                    "compute_last_slice_features expects a single sample; use resolve_batch for batched inputs."
                )
            packed[stage] = last[0].to(dtype=torch.float32, device="cpu").contiguous()
        return _validate_feature_pack(packed)

    def get_or_compute(
        self,
        rgb_btchw: torch.Tensor,
        *,
        identity: Optional[SampleIdentity] = None,
        force: bool = False,
        allow_compute: bool = True,
        _verified_content_sha256: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        clip = rgb_btchw.detach()
        if clip.dim() == 5:
            if clip.shape[0] != 1:
                raise ValueError(
                    "Pass one sample to get_or_compute, or use resolve_batch."
                )
            clip = clip[0]
        if clip.dim() != 4 or clip.shape[1] != 3:
            raise ValueError(f"Expected [T,3,H,W], got {tuple(clip.shape)}")
        clip = clip.to(dtype=torch.float32)
        if clip.numel() > 0 and float(clip.max()) > 2.0:
            clip = clip / 255.0
        content_sha = (
            _verified_content_sha256
            if _verified_content_sha256 is not None
            else fingerprint_rgb_clip(clip)
        )
        if identity is not None and identity.content_sha256 != content_sha:
            raise ValueError(
                "SampleIdentity.content_sha256 does not match the provided RGB clip."
            )
        if identity is None:
            identity = SampleIdentity(
                dataset_root="unknown",
                video_id="unknown",
                window_start=-1,
                frame_names=tuple(),
                window_len=int(clip.shape[0]),
                content_sha256=content_sha,
                view_id="content_only",
                input_shape=tuple(int(x) for x in clip.shape),
                preprocess=self.preprocess,
            )
        elif identity.input_shape is None or identity.preprocess is None:
            identity = SampleIdentity(
                dataset_root=identity.dataset_root,
                video_id=identity.video_id,
                window_start=identity.window_start,
                frame_names=tuple(identity.frame_names),
                window_len=identity.window_len,
                content_sha256=content_sha,
                split=identity.split,
                temporal_sampling=dict(identity.temporal_sampling),
                view_id=identity.view_id,
                input_shape=tuple(int(x) for x in clip.shape),
                padded_input_geometry=identity.padded_input_geometry,
                preprocess=dict(identity.preprocess or self.preprocess),
            )
        if not force:
            hit = self.load_entry(content_sha, require_identity=identity)
            if hit is not None:
                return hit
        self.stats["misses"] += 1
        if not allow_compute:
            raise FileNotFoundError(
                f"Missing compatible cached reference features for {content_sha[:16]}… (read-only cache mode)."
            )
        features = self.compute_last_slice_features(clip)
        path = self.entry_path(content_sha)
        if force and path.is_file():
            self.stats["regenerated"] += 1
        if path.is_file() and (not force):
            return self._features_to_float32(features)
        self.save_entry(
            content_sha,
            features,
            input_shape=tuple(int(x) for x in clip.shape),
            identity=identity,
            overwrite=force,
        )
        return self._features_to_float32(features)

    def resolve_batch(
        self,
        rgb_batch: torch.Tensor,
        *,
        identities: Optional[Sequence[SampleIdentity]] = None,
        n_frames: Optional[torch.Tensor] = None,
        content_hw: Optional[Sequence[Tuple[int, int]]] = None,
        view_id: Union[str, Sequence[str]] = "identity",
        force: bool = False,
        allow_compute: bool = True,
        verify_identity_content: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Resolve a BTCHW batch into stage tensors [B,C,1,H,W]."""
        if rgb_batch.dim() != 5 or rgb_batch.shape[2] != 3:
            raise ValueError(
                f"rgb_batch must be [B,T,3,H,W], got {tuple(rgb_batch.shape)}"
            )
        batch_size = int(rgb_batch.shape[0])
        if identities is not None and len(identities) != batch_size:
            raise ValueError("identities length must match batch size.")
        if content_hw is not None and len(content_hw) != batch_size:
            raise ValueError("content_hw length must match batch size.")
        if isinstance(view_id, str):
            view_ids = [view_id] * batch_size
        else:
            view_ids = list(view_id)
            if len(view_ids) != batch_size:
                raise ValueError("view_id sequence length must match batch size.")
        per_sample: List[Dict[str, torch.Tensor]] = []
        for index in range(batch_size):
            frames = (
                int(n_frames[index])
                if n_frames is not None
                else int(rgb_batch.shape[1])
            )
            if content_hw is not None:
                height, width = (int(content_hw[index][0]), int(content_hw[index][1]))
            else:
                height, width = (int(rgb_batch.shape[-2]), int(rgb_batch.shape[-1]))
            clip = (
                rgb_batch[index, :frames, :, :height, :width]
                .detach()
                .to(dtype=torch.float32)
            )
            if clip.numel() > 0 and float(clip.max()) > 2.0:
                clip = clip / 255.0
            if identities is not None:
                identity = identities[index]
                content_sha = (
                    fingerprint_rgb_clip(clip)
                    if verify_identity_content
                    else identity.content_sha256
                )
                if content_sha != identity.content_sha256:
                    raise ValueError(
                        "SampleIdentity.content_sha256 does not match the provided RGB clip."
                    )
            else:
                content_sha = fingerprint_rgb_clip(clip)
                identity = SampleIdentity(
                    dataset_root="batch",
                    video_id=f"batch_{index}",
                    window_start=-1,
                    frame_names=tuple(),
                    window_len=frames,
                    content_sha256=content_sha,
                    view_id=str(view_ids[index]),
                    input_shape=tuple(int(x) for x in clip.shape),
                    padded_input_geometry={
                        "batch_shape": list(rgb_batch.shape),
                        "n_frames": frames,
                        "content_hw": [height, width],
                    },
                    preprocess=self.preprocess,
                )
            per_sample.append(
                self.get_or_compute(
                    clip,
                    identity=identity,
                    force=force,
                    allow_compute=allow_compute,
                    _verified_content_sha256=content_sha,
                )
            )
        batched: Dict[str, torch.Tensor] = {}
        for stage in STAGES:
            batched[stage] = torch.stack(
                [sample[stage] for sample in per_sample], dim=0
            ).contiguous()
        return batched

    def estimate_entry_bytes(self) -> int:
        """Rough per-window storage estimate for all four last-slice stages."""
        grids = {
            "stage1": (96, 56, 96),
            "stage2": (192, 28, 48),
            "stage3": (384, 14, 24),
            "stage4": (768, 7, 12),
        }
        itemsize = 2 if self.storage_dtype == "float16" else 4
        total = 0
        for channels, height, width in grids.values():
            total += channels * 1 * height * width * itemsize
        return total + 2048


def make_identity_from_loader_sample(
    *,
    dataset_root: str,
    video_id: str,
    window_start: int,
    frame_names: Sequence[str],
    rgb_thwc_or_tchw: torch.Tensor,
    split: Optional[str] = None,
    stride: int = 1,
    view_id: str = "identity",
    preprocess: Optional[Mapping[str, Any]] = None,
) -> Tuple[SampleIdentity, torch.Tensor]:
    """Build identity + BTCHW clip from a DatasetLoader RGB sample."""
    rgb = rgb_thwc_or_tchw.detach()
    if not torch.is_tensor(rgb):
        rgb = torch.as_tensor(rgb)
    rgb = rgb.to(dtype=torch.float32)
    if rgb.dim() != 4:
        raise ValueError(f"Expected 4D RGB clip, got {tuple(rgb.shape)}")
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2).contiguous()
    elif rgb.shape[1] != 3:
        raise ValueError(f"Unsupported RGB layout {tuple(rgb.shape)}")
    if rgb.numel() > 0 and float(rgb.max()) > 2.0:
        rgb = rgb / 255.0
    content_sha = fingerprint_rgb_clip(rgb)
    identity = SampleIdentity(
        dataset_root=str(Path(dataset_root).expanduser().resolve()),
        split=split,
        video_id=str(video_id),
        window_start=int(window_start),
        frame_names=tuple(str(name) for name in frame_names),
        window_len=int(rgb.shape[0]),
        temporal_sampling={"mode": "contiguous", "stride": int(stride)},
        view_id=str(view_id),
        content_sha256=content_sha,
        input_shape=tuple(int(x) for x in rgb.shape),
        preprocess=dict(preprocess or preprocess_config()),
    )
    return (identity, rgb)
