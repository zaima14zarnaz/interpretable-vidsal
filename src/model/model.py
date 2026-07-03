"""
End-to-end explainable video saliency model.

Pipeline: RGB window -> VideoSwinTransformer -> ConceptCreation (raw features)
         -> SpatioTemporal3DFeatureInfusion (decoder features) -> SaliencyDecoder.
"""

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.video_swin import VideoSwinTransformer
from model.concept_creation import ConceptCreation
from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder
from model.temporal_feature_infusion import SpatioTemporal3DFeatureInfusion


class ExplainableVidSalModel(nn.Module):
    """
    Explainable video saliency for the last frame in a window.

    Shapes:
        Input video x:     [B, T, C, H, W] (BTCHW) or [B, C, T, H, W] (BCTHW);
                           collate may supply [B, T, H, W, 3], converted internally.
        Backbone features: per-stage [B, Cf, Tf, Hf, Wf]
        Concept repr:      per-stage temporal concepts [M, concept_dim]
                           and optional visual concepts [B*T*N, concept_dim]
        Output saliency:   [B, 1, H, W] (last RGB frame resolution)
    """

    # stage1 is preserved at full backbone resolution because it is the finest
    # decoder stage; downsampling stage1 concepts can hurt spatial localization
    # and NSS.
    _CONCEPT_MAX_HW_BY_STAGE = {
        "stage1": None,
        "stage2": None,
        "stage3": None,
        "stage4": None,
    }

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
        input_format: str = "BTCHW",
        resize_to: Union[int, Tuple[int, int]] = (224, 384),
        concept_dim: int = 256,
        num_concepts: int = 32,
        concept_hidden_dim: int = 512,
        saliency_hidden_dim: int = 96,
        top_k: int = 9,
        max_source_patches: int = 128,
        tau_pi: float = 0.1,
        tau_alpha: float = 0.07,
        tau_concept: float = 0.1,
        concept_residual_weight: float = 0.1,
        last_transition_only: bool = True,
        # Deprecated: ignored by ConceptGatedMultiScaleSaliencyDecoder (kept for scripts).
        use_feature_refinement: bool = True,
        feature_refine_channels: int = 128,
        use_rgb_refinement: bool = False,
        use_gated_trajectory_head: bool = True,
        gated_trajectory_residual_scale: float = 0.2,
        use_subpatch_head: bool = True,
        subpatch_factor: int = 4,
        subpatch_hidden_dim: Optional[int] = None,
        subpatch_residual_scale: float = 1.0,
        output_activation: str = "sigmoid",
        return_details: bool = False,
        use_temporal_transition_aggregation: bool = False,
        visual_concept_on: bool = True,
        temporal_concepts_on: bool = True,
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
        decoder_temporal_aggregation: str = "learned_all_frames",
        decoder_use_side_logit_fusion: bool = True,
        **_deprecated_saliency_kwargs: Any,
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
        self.use_temporal_transition_aggregation = use_temporal_transition_aggregation
        self.visual_concept_on = bool(visual_concept_on)
        self.temporal_concepts_on = bool(temporal_concepts_on)
        self.output_activation = output_activation
        self._backbone_frozen = freeze_backbone
        self.backbone_gradient_checkpointing = bool(backbone_gradient_checkpointing)
        self.visual_concept_residual_weight = float(visual_concept_residual_weight)
        self.visual_assignment_mode = visual_assignment_mode
        self.visual_assignment_temperature = float(visual_assignment_temperature)
        self.visual_entropy_weight = float(visual_entropy_weight)
        self.visual_usage_weight = float(visual_usage_weight)
        self.use_visual_saliency_alignment = bool(use_visual_saliency_alignment)
        self.visual_saliency_align_weight = float(visual_saliency_align_weight)
        self.allow_eval_concept_losses = bool(allow_eval_concept_losses)
        if temporal_dim is None:
            temporal_dim = concept_dim
        self.temporal_dim = int(temporal_dim)
        self.decoder_temporal_aggregation = decoder_temporal_aggregation
        self.decoder_use_side_logit_fusion = bool(decoder_use_side_logit_fusion)
        if not self.visual_concept_on and not self.temporal_concepts_on:
            raise ValueError(
                "At least one concept branch must be enabled: "
                "visual_concept_on=True and/or temporal_concepts_on=True."
            )

        self.backbone = VideoSwinTransformer(
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            return_stages=self.backbone_stages,
            input_format=input_format,
            output_format="BCTHW",
            resize_to=resize_to,
            normalize=True,
            gradient_checkpointing=backbone_gradient_checkpointing,
        )

        # Video Swin-T stage channels: stage1=96, stage2=192, stage3=384, stage4=768.
        feature_channels = self.backbone.get_feature_channels()
        self.stage_channels = {
            stage: feature_channels[stage] for stage in self.backbone_stages
        }

        self.concept_creations = nn.ModuleDict()
        concept_last_transition_only = (
            False if use_temporal_transition_aggregation else last_transition_only
        )
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
                last_transition_only=concept_last_transition_only,
                visual_concept_residual_weight=visual_concept_residual_weight,
                visual_assignment_mode=visual_assignment_mode,
                visual_assignment_temperature=visual_assignment_temperature,
                visual_entropy_weight=visual_entropy_weight,
                visual_usage_weight=visual_usage_weight,
                use_visual_saliency_alignment=use_visual_saliency_alignment,
                visual_saliency_align_weight=visual_saliency_align_weight,
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
                layer_scale_init=1e-3,
            )

        self.saliency_prediction = ConceptGatedMultiScaleSaliencyDecoder(
            stage_channels=self.stage_channels,
            concept_dim=concept_dim,
            decoder_channels=saliency_hidden_dim,
            feature_residual_scale=0.25,
            dropout=0.05,
            tau_pi=tau_pi,
            output_activation=output_activation,
            temporal_aggregation=decoder_temporal_aggregation,
            use_side_logit_fusion=decoder_use_side_logit_fusion,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _resize_feature_for_concepts(
        self,
        features: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        cap = self._CONCEPT_MAX_HW_BY_STAGE.get(stage, None)
        if cap is None:
            return features

        _, _, T, H, W = features.shape

        if isinstance(cap, int):
            if H <= cap and W <= cap:
                return features
            return F.interpolate(
                features,
                size=(T, cap, cap),
                mode="trilinear",
                align_corners=False,
            )

        if isinstance(cap, (tuple, list)) and len(cap) == 2:
            target_h, target_w = int(cap[0]), int(cap[1])
            if H == target_h and W == target_w:
                return features
            return F.interpolate(
                features,
                size=(T, target_h, target_w),
                mode="trilinear",
                align_corners=False,
            )

        raise ValueError(
            f"Invalid concept resize cap for {stage}: {cap!r}. "
            "Expected None, int, or (target_h, target_w)."
        )

    def _normalize_video_layout(self, x: torch.Tensor) -> torch.Tensor:
        """Accept dataloader layout [B, T, H, W, 3] and convert to configured input_format."""
        if x.dim() == 5 and x.shape[-1] == 3:
            return x.permute(0, 1, 4, 2, 3)
        return x

    def _extract_last_rgb_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Last frame from the original (pre-backbone) video tensor."""
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
        return {
            stage: self.temporal_feature_infusers[stage].get_last_diagnostics()
            for stage in self.backbone_stages
        }

    def forward(
        self,
        x: torch.Tensor,
        saliency_maps: Optional[torch.Tensor] = None,
        return_details: Optional[bool] = None,
        return_concept_losses: Optional[bool] = None,
        collect_gate_debug: bool = False,
        return_decoder_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: RGB video window from the dataloader.
            saliency_maps: optional GT saliency for auxiliary ConceptCreation losses
                and gate regularization during training (never passed to the final
                saliency prediction path).
            return_details: if True, return intermediate outputs; default from __init__.
            return_concept_losses: if None, concept losses are computed only while
                training (and when saliency_maps is provided). In eval mode, concept
                losses are disabled unless explicitly requested or
                allow_eval_concept_losses=True.
            collect_gate_debug: deprecated; kept for API compatibility. Gate debug
                stats are computed post-hoc for logging and do not affect outputs.
            return_decoder_diagnostics: if True, include decoder stage maps/gates in
                prediction_out. Off by default so diagnostics never change training.

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

            concept_creations = self.concept_creations
            temporal_feature_infusers = self.temporal_feature_infusers
            for stage in self.backbone_stages:
                stage_features = features_dict[stage]
                concept_features = self._resize_feature_for_concepts(
                    stage_features, stage
                )
                if return_details:
                    concept_features_shape[stage] = tuple(concept_features.shape)
                concept_outs[stage] = concept_creations[stage](
                    concept_features,
                    saliency_maps=saliency_maps_for_concepts,
                    return_losses=return_concept_losses,
                    collect_gate_debug=False,
                )
                decoder_features_dict[stage] = temporal_feature_infusers[stage](
                    stage_features
                )

            # Learned ConvTranspose upsampling uses fixed scale factors tied to the
            # backbone feature resolution, so decoder output_size should match
            # backbone.resize_to when resize_to is set. Original-size output requires
            # either processing original-size frames or a separate external resize.
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
