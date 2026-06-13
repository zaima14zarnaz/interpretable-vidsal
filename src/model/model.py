"""
End-to-end explainable video saliency model.

Pipeline: RGB window -> VideoSwinTransformer -> ConceptCreation (per stage) -> SaliencyPrediction.
"""

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.video_swin import VideoSwinTransformer
from model.concept_creation import ConceptCreation
from model.saliency_prediction import MultiScaleSaliencyPrediction, SaliencyPrediction


class ExplainableVidSalModel(nn.Module):
    """
    Explainable video saliency for the last frame in a window.

    Shapes:
        Input video x:     [B, T, C, H, W] (BTCHW) or [B, C, T, H, W] (BCTHW);
                           collate may supply [B, T, H, W, 3], converted internally.
        Backbone features: per-stage [B, Cf, Tf, Hf, Wf]
        Concept repr:      per-stage [M, concept_dim] (trajectory-level)
        Output saliency:   [B, 1, H, W] (last RGB frame resolution)
    """

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
        input_format: str = "BTCHW",
        resize_to: Union[int, Tuple[int, int]] = (224, 224),
        concept_dim: int = 256,
        num_concepts: int = 32,
        concept_hidden_dim: int = 512,
        saliency_hidden_dim: int = 256,
        top_k: int = 9,
        max_source_patches: int = 128,
        tau_pi: float = 0.1,
        tau_alpha: float = 0.07,
        tau_concept: float = 0.1,
        concept_residual_weight: float = 0.1,
        last_transition_only: bool = True,
        use_feature_refinement: bool = True,
        feature_refine_channels: int = 128,
        use_rgb_refinement: bool = False,
        use_gated_trajectory_head: bool = True,
        gated_trajectory_residual_scale: float = 0.2,
        output_activation: str = "sigmoid",
        return_details: bool = False,
    ):
        super().__init__()

        if input_format not in ("BTCHW", "BCTHW"):
            raise ValueError("input_format must be 'BTCHW' or 'BCTHW'")

        if backbone_stages is None:
            backbone_stages = (backbone_stage,)
        else:
            backbone_stages = tuple(backbone_stages)

        self.backbone_stage = backbone_stage
        self.backbone_stages = backbone_stages
        self.input_format = input_format
        self.return_details = return_details
        self.last_transition_only = last_transition_only
        self._backbone_frozen = freeze_backbone

        self.backbone = VideoSwinTransformer(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            return_stages=self.backbone_stages,
            input_format=input_format,
            output_format="BCTHW",
            resize_to=resize_to,
            normalize=True,
        )

        feature_channels = self.backbone.get_feature_channels()
        self.stage_channels = {
            stage: feature_channels[stage] for stage in self.backbone_stages
        }

        self.concept_creations = nn.ModuleDict()
        for stage in self.backbone_stages:
            self.concept_creations[stage] = ConceptCreation(
                in_channels=self.stage_channels[stage],
                concept_dim=concept_dim,
                num_concepts=num_concepts,
                hidden_dim=concept_hidden_dim,
                top_k=top_k,
                tau_alpha=tau_alpha,
                tau_concept=tau_concept,
                max_source_patches=max_source_patches,
                concept_residual_weight=concept_residual_weight,
                use_target_centric=True,
                last_transition_only=last_transition_only,
            )

        self.saliency_prediction = MultiScaleSaliencyPrediction(
            stage_channels=self.stage_channels,
            concept_dim=concept_dim,
            hidden_dim=saliency_hidden_dim,
            tau_pi=tau_pi,
            output_activation=output_activation,
            use_feature_refinement=use_feature_refinement,
            feature_refine_channels=feature_refine_channels,
            use_peak_refinement=True,
            peak_refine_channels=128,
            peak_residual_scale=0.3,
            fusion_hidden_channels=64,
            predict_delta=True,
            use_gated_trajectory_head=use_gated_trajectory_head,
            gated_trajectory_residual_scale=gated_trajectory_residual_scale,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _resize_feature_for_concepts(
        self,
        features: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        max_hw_by_stage = {
            "stage1": 28,
            "stage2": 28,
            "stage3": 14,
            "stage4": 7,
        }
        max_hw = max_hw_by_stage.get(stage, 28)
        B, C, T, H, W = features.shape
        if H <= max_hw and W <= max_hw:
            return features
        return F.interpolate(
            features,
            size=(T, max_hw, max_hw),
            mode="trilinear",
            align_corners=False,
        )

    def _normalize_video_layout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept dataloader layout [B, T, H, W, 3] and convert to configured input_format.
        """
        if x.dim() == 5 and x.shape[-1] == 3:
            # [B, T, H, W, C] -> [B, T, C, H, W]
            return x.permute(0, 1, 4, 2, 3).contiguous()
        return x

    def _extract_last_rgb_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Last frame from the original (pre-backbone) video tensor.

        Args:
            x: [B, T, C, H, W] if input_format='BTCHW', else [B, C, T, H, W].

        Returns:
            last_rgb: [B, 3, H, W] float in [0, 1] when input was normalized.
        """
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D video tensor, got shape {tuple(x.shape)}"
            )

        if self.input_format == "BTCHW":
            if x.shape[2] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)
            if x.shape[2] != 3:
                raise ValueError(
                    f"BTCHW input expected 3 channels at dim 2, got shape {tuple(x.shape)}"
                )
            last_rgb = x[:, -1]  # [B, 3, H, W]
        elif self.input_format == "BCTHW":
            if x.shape[1] != 3:
                raise ValueError(
                    f"BCTHW input expected 3 channels at dim 1, got shape {tuple(x.shape)}"
                )
            last_rgb = x[:, :, -1]  # [B, 3, H, W]
        else:
            raise ValueError(f"Unsupported input_format: {self.input_format}")

        last_rgb = last_rgb.float()
        if last_rgb.numel() > 0 and last_rgb.max() > 2.0:
            last_rgb = last_rgb / 255.0
        return last_rgb

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
        params.extend(self.saliency_prediction.parameters())
        if not self._backbone_frozen:
            params.extend(self.backbone.parameters())
        return params

    def optimize_for_inference(self) -> "ExplainableVidSalModel":
        """Enable cudnn benchmark for fixed-size inference (no math changes)."""
        self.eval()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        return self

    def forward(
        self,
        x: torch.Tensor,
        saliency_maps: Optional[torch.Tensor] = None,
        return_details: Optional[bool] = None,
        return_concept_losses: Optional[bool] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: RGB video window from the dataloader.
            saliency_maps: optional GT saliency for ConceptCreation gate labels only
                (not passed to SaliencyPrediction).
            return_details: if True, return intermediate outputs; default from __init__.

        Returns:
            saliency_map [B, 1, H, W] when return_details is False, else a result dict.
        """
        if return_details is None:
            return_details = self.return_details

        inference_ctx = torch.inference_mode if not self.training else nullcontext

        with inference_ctx():
            x = self._normalize_video_layout(x)
            last_rgb_frame = self._extract_last_rgb_frame(x)

            if self._backbone_frozen:
                with torch.no_grad():
                    features_dict = self.backbone.forward_features(x)
            else:
                features_dict = self.backbone.forward_features(x)

            concept_outs: Dict[str, Dict[str, Any]] = {}
            concept_features_dict: Dict[str, torch.Tensor] = {}

            if return_concept_losses is None:
                return_concept_losses = saliency_maps is not None
            for stage in self.backbone_stages:
                stage_features = features_dict[stage]
                concept_features = self._resize_feature_for_concepts(stage_features, stage)
                concept_features_dict[stage] = concept_features
                concept_outs[stage] = self.concept_creations[stage](
                    concept_features,
                    saliency_maps=saliency_maps,
                    return_losses=return_concept_losses,
                )

            pred_out = self.saliency_prediction(
                concept_outs,
                last_rgb_frame,
                video_features_dict=concept_features_dict,
                return_details=return_details,
            )

            if not return_details:
                return pred_out["saliency_map"]

            return {
                "saliency_map": pred_out["saliency_map"],
                "saliency_logits": pred_out["saliency_logits"],
                "patch_saliency_logits": pred_out.get("patch_saliency_logits"),
                "concept_saliency_map": pred_out.get("concept_saliency_map"),
                "concept_saliency_logits": pred_out.get("concept_saliency_logits"),
                "concept_only_saliency_map": pred_out.get("concept_only_saliency_map"),
                "concept_only_saliency_logits": pred_out.get("concept_only_saliency_logits"),
                "concept_out": concept_outs,
                "prediction_out": pred_out,
                "features_shape": {
                    stage: tuple(concept_features_dict[stage].shape)
                    for stage in self.backbone_stages
                },
            }
