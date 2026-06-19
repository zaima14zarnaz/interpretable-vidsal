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
        Concept repr:      per-stage trajectory concepts [M, concept_dim]
                           and optional visual concepts [B*T*N, concept_dim]
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
        use_subpatch_head: bool = True,
        subpatch_factor: int = 16,
        subpatch_hidden_dim: Optional[int] = None,
        subpatch_residual_scale: float = 1.0,
        use_temporal_transition_aggregation: bool = False,
        temporal_aggregation_hidden_channels: int = 64,
        temporal_aggregation_temperature: float = 1.0,
        visual_concept_on: bool = True,
        trajectory_concepts_on: bool = True,
        visual_concept_logit_scale: float = 0.2,
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
        self.trajectory_concepts_on = bool(trajectory_concepts_on)
        self.visual_concept_logit_scale = float(visual_concept_logit_scale)
        self.output_activation = output_activation
        self._backbone_frozen = freeze_backbone

        if not self.visual_concept_on and not self.trajectory_concepts_on:
            raise ValueError(
                "At least one concept branch must be enabled: "
                "visual_concept_on=True and/or trajectory_concepts_on=True."
            )

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
        # Option 2: when enabled, ConceptCreation emits all adjacent transitions and
        # SaliencyPrediction explicitly aggregates them over time before predicting the
        # last-frame saliency map.
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
            use_subpatch_head=use_subpatch_head,
            subpatch_factor=subpatch_factor,
            subpatch_hidden_dim=subpatch_hidden_dim,
            subpatch_residual_scale=subpatch_residual_scale,
            use_temporal_transition_aggregation=use_temporal_transition_aggregation,
            temporal_aggregation_hidden_channels=temporal_aggregation_hidden_channels,
            temporal_aggregation_temperature=temporal_aggregation_temperature,
        )

        # Patch-level visual concepts are converted directly to dense saliency logits
        # in this wrapper, so saliency_prediction.py does not need to know about the
        # visual branch. Trajectory concepts continue to use MultiScaleSaliencyPrediction.
        visual_head_hidden_dim = max(32, min(saliency_hidden_dim, concept_dim))
        self.visual_saliency_heads = nn.ModuleDict(
            {
                stage: nn.Sequential(
                    nn.Conv2d(concept_dim, visual_head_hidden_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(visual_head_hidden_dim, 1, kernel_size=1),
                )
                for stage in self.backbone_stages
            }
        )
        self.visual_stage_logits = nn.Parameter(
            torch.zeros(len(self.backbone_stages), dtype=torch.float32)
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


    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the same output activation convention used by the saliency head."""
        activation = self.output_activation.lower()
        if activation == "sigmoid":
            return torch.sigmoid(logits)
        if activation == "softplus":
            return F.softplus(logits)
        if activation == "relu":
            return F.relu(logits)
        if activation in {"identity", "none", "linear"}:
            return logits
        raise ValueError(f"Unsupported output_activation: {self.output_activation}")

    @staticmethod
    def _feature_shape_from_metadata(metadata: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        """Read a ConceptCreation feature shape dict as a tuple."""
        feature_shape = metadata.get("feature_shape")
        if not isinstance(feature_shape, dict):
            raise ValueError("Concept metadata must include feature_shape as a dict.")
        return (
            int(feature_shape["B"]),
            int(feature_shape["C"]),
            int(feature_shape["T"]),
            int(feature_shape["H"]),
            int(feature_shape["W"]),
        )

    def _visual_concept_logits_from_stage(
        self,
        stage: str,
        concept_out: Dict[str, Any],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Convert patch-level visual concept representations into a dense saliency-logit map.

        Requires ConceptCreation to return visual_concept_representation with shape
        [B*T*H*W, concept_dim]. The last temporal slice is reshaped to [B, D, H, W]
        and passed through a lightweight 1x1 saliency head.
        """
        if "visual_concept_representation" not in concept_out:
            raise RuntimeError(
                "visual_concept_on=True, but ConceptCreation did not return "
                "visual_concept_representation. Apply the visual-concept changes to "
                "concept_creation.py first."
            )

        visual_repr = concept_out["visual_concept_representation"]
        if not torch.is_tensor(visual_repr) or visual_repr.dim() != 2:
            raise ValueError(
                "visual_concept_representation must be a tensor with shape "
                "[B*T*H*W, concept_dim]."
            )

        visual_metadata = concept_out.get("visual_metadata")
        if isinstance(visual_metadata, dict) and "feature_shape" in visual_metadata:
            B, _, T, H, W = self._feature_shape_from_metadata(visual_metadata)
        else:
            # Fallback for older visual branch implementations that only attach the
            # feature shape to trajectory metadata.
            B, _, T, H, W = self._feature_shape_from_metadata(concept_out["metadata"])

        expected = B * T * H * W
        if visual_repr.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} visual patch representations for stage {stage}, "
                f"got {visual_repr.shape[0]}."
            )

        visual_grid = visual_repr.reshape(B, T, H, W, -1)
        last_visual_grid = visual_grid[:, -1].permute(0, 3, 1, 2).contiguous()
        stage_logits = self.visual_saliency_heads[stage](last_visual_grid)
        if stage_logits.shape[-2:] != output_size:
            stage_logits = F.interpolate(
                stage_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return stage_logits

    def _predict_visual_concept_logits(
        self,
        concept_outs: Dict[str, Dict[str, Any]],
        output_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fuse per-stage visual concept logits into one dense saliency-logit map."""
        stage_logits: List[torch.Tensor] = []
        stage_logits_by_name: Dict[str, torch.Tensor] = {}

        for stage in self.backbone_stages:
            logits_s = self._visual_concept_logits_from_stage(
                stage,
                concept_outs[stage],
                output_size,
            )
            stage_logits.append(logits_s)
            stage_logits_by_name[stage] = logits_s

        weights = torch.softmax(
            self.visual_stage_logits[: len(stage_logits)].to(
                device=stage_logits[0].device,
                dtype=stage_logits[0].dtype,
            ),
            dim=0,
        )
        fused = torch.zeros_like(stage_logits[0])
        for weight, logits_s in zip(weights, stage_logits):
            fused = fused + weight.view(1, 1, 1, 1) * logits_s
        return fused, stage_logits_by_name

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
        params.extend(self.visual_saliency_heads.parameters())
        params.append(self.visual_stage_logits)
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

    def forward(
        self,
        x: torch.Tensor,
        saliency_maps: Optional[torch.Tensor] = None,
        return_details: Optional[bool] = None,
        return_concept_losses: Optional[bool] = None,
        collect_gate_debug: bool = False,
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
                    collect_gate_debug=collect_gate_debug,
                )

            pred_out: Dict[str, Any] = {}
            trajectory_logits: Optional[torch.Tensor] = None
            trajectory_map: Optional[torch.Tensor] = None

            if self.trajectory_concepts_on:
                pred_out = self.saliency_prediction(
                    concept_outs,
                    last_rgb_frame,
                    video_features_dict=concept_features_dict,
                    return_details=return_details,
                    last_rgb_prepared=True,
                )
                trajectory_logits = pred_out["saliency_logits"]
                trajectory_map = pred_out["saliency_map"]

            visual_logits: Optional[torch.Tensor] = None
            visual_map: Optional[torch.Tensor] = None
            visual_stage_logits: Optional[Dict[str, torch.Tensor]] = None
            if self.visual_concept_on:
                visual_logits, visual_stage_logits = self._predict_visual_concept_logits(
                    concept_outs,
                    output_size=last_rgb_frame.shape[-2:],
                )
                visual_map = self._apply_output_activation(visual_logits)

            if trajectory_logits is not None and visual_logits is not None:
                saliency_logits = (
                    trajectory_logits + self.visual_concept_logit_scale * visual_logits
                )
            elif trajectory_logits is not None:
                saliency_logits = trajectory_logits
            elif visual_logits is not None:
                saliency_logits = visual_logits
            else:
                raise RuntimeError("No enabled concept branch produced saliency logits.")

            saliency_map = self._apply_output_activation(saliency_logits)

            # Keep the same prediction_out contract, but expose branch-specific maps.
            pred_out = dict(pred_out)
            pred_out["saliency_logits"] = saliency_logits
            pred_out["saliency_map"] = saliency_map
            pred_out["trajectory_saliency_logits"] = trajectory_logits
            pred_out["trajectory_saliency_map"] = trajectory_map
            pred_out["visual_saliency_logits"] = visual_logits
            pred_out["visual_saliency_map"] = visual_map
            pred_out["visual_stage_saliency_logits"] = visual_stage_logits
            pred_out["visual_concept_on"] = self.visual_concept_on
            pred_out["trajectory_concepts_on"] = self.trajectory_concepts_on
            pred_out["visual_concept_logit_scale"] = self.visual_concept_logit_scale

            if not return_details:
                return saliency_map

            return {
                "saliency_map": saliency_map,
                "saliency_logits": saliency_logits,
                "patch_saliency_logits": pred_out.get("patch_saliency_logits"),
                "concept_saliency_map": pred_out.get("concept_saliency_map"),
                "concept_saliency_logits": pred_out.get("concept_saliency_logits"),
                "concept_only_saliency_map": pred_out.get("concept_only_saliency_map"),
                "concept_only_saliency_logits": pred_out.get("concept_only_saliency_logits"),
                "trajectory_saliency_map": trajectory_map,
                "trajectory_saliency_logits": trajectory_logits,
                "visual_saliency_map": visual_map,
                "visual_saliency_logits": visual_logits,
                "visual_stage_saliency_logits": visual_stage_logits,
                "visual_concept_on": self.visual_concept_on,
                "trajectory_concepts_on": self.trajectory_concepts_on,
                "concept_out": concept_outs,
                "prediction_out": pred_out,
                "features_shape": {
                    stage: tuple(concept_features_dict[stage].shape)
                    for stage in self.backbone_stages
                },
            }
