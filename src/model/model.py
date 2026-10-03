"""Explainable video saliency with optional, fixed-reference feature caching.

Pass reference_cache_dir to forward(..., return_details=True) during training.
The first cache call saves a fixed snapshot of the *currently loaded* backbone.
Every later cache miss uses that snapshot, never the updated training weights.
Live backbone features still drive predictions and receive training gradients.

Cached references contain the final backbone time step at each concept scale:
reference_features[stage]: [B, C, 1, H, W]
concept_out[stage]['reference_patch_features']: [B*H*W, C], row-major.
This file passes fixed references into the concept module's preservation loss.
Use prototype_projection.py for a separate training-data projection sweep.
Requires PyTorch >= 2.0 for torch.func.functional_call.
"""

from contextlib import contextmanager, nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from model.backbones.video_swin_custom import VideoSwinTransformer
from model.concept_creation import VisualConceptCreation
from model.diagnostic_trace import record as _diag_record
from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder
from model.temporal_feature_infusion import SpatioTemporal3DFeatureInfusion


class _ReferenceBackboneForward(nn.Module):
    """Expose forward_features to functional_call without a second backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.backbone.forward_features(x)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # Byte view also handles bfloat16, for which numpy() alone is unsupported.
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _state_digest(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(_tensor_digest(value).encode())
    return digest.hexdigest()


def _publish_cache_file(payload: Dict[str, Any], path: Path) -> None:
    """Publish a complete immutable file; concurrent writers cannot replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link publication is atomic and does not overwrite another writer.
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _reference_eval(backbone: nn.Module):
    """Restore every module's mode, including deliberately frozen submodules."""
    modes = [(module, module.training) for module in backbone.modules()]
    try:
        backbone.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


class ExplainableVidSalModel(nn.Module):
    """Predict saliency for the last RGB frame of a video window."""

    _CONCEPT_MAX_HW_BY_STAGE = {
        "stage1": None, "stage2": None, "stage3": None, "stage4": None,
    }
    _REFERENCE_CACHE_VERSION = 1

    def __init__(
        self,
        backbone_stage: str = "stage2",
        backbone_stages: Optional[Tuple[str, ...]] = ("stage1", "stage2", "stage3", "stage4"),
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
        max_source_patches: int = 128,
        tau_pi: float = 0.1,
        tau_alpha: float = 0.07,
        tau_concept: float = 0.1,
        concept_residual_weight: float = 0.1,
        last_transition_only: bool = True,
        # Deprecated: retained for compatibility with existing scripts.
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
        visual_usage_weight: float = 0.001,
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
        use_temporal_feature_infusion: bool = True,
        use_shared_concept_activations: bool = False,
        prototype_bottleneck_strength: float = 0.4,
        prototype_application_position: str = "pre_refine",
        fine_unary_mask_strength: float = 0.6,
        stage2_backbone_fusion_enabled: bool = True,
        reference_cache_dir: Optional[Union[str, Path]] = None,
        reference_cache_tag: str = "v1",
        use_saliency_prototype_roles: bool = True,
        num_salient_prototypes: Optional[int] = None,
        salient_patch_threshold: float = 0.6,
        non_salient_patch_threshold: float = 0.2,
        prototype_role_weight: float = 0.1,
        prototype_role_cluster_weight: float = 0.5,
        prototype_role_separation_weight: float = 0.5,
        prototype_role_margin: float = 0.2,
        visual_preservation_weight: float = 1.0,
        preservation_max_patches: int = 256,
        require_reference_features: bool = True,
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
        self.temporal_dim = int(concept_dim if temporal_dim is None else temporal_dim)
        self.decoder_temporal_aggregation = decoder_temporal_aggregation
        self.decoder_use_side_logit_fusion = bool(decoder_use_side_logit_fusion)
        self.use_temporal_feature_infusion = bool(use_temporal_feature_infusion)
        self.use_shared_concept_activations = bool(use_shared_concept_activations)
        self.stage2_backbone_fusion_enabled = bool(stage2_backbone_fusion_enabled)
        self.reference_cache_dir = None if reference_cache_dir is None else str(reference_cache_dir)
        self.reference_cache_tag = str(reference_cache_tag)
        self._reference_backbone_normalize = bool(backbone_normalize)
        # CPU-only memoized reference state; not registered as model parameters/buffers.
        self._reference_snapshot_path: Optional[str] = None
        self._reference_snapshot: Optional[Dict[str, Any]] = None
        if not self.visual_concept_on and not self.temporal_concepts_on:
            raise ValueError("At least one concept branch must be enabled.")

        self.backbone = VideoSwinTransformer(
            pretrained=pretrained_backbone, freeze_backbone=freeze_backbone,
            return_stages=self.backbone_stages, input_format=input_format,
            output_format="BCTHW", resize_to=resize_to, normalize=backbone_normalize,
            gradient_checkpointing=backbone_gradient_checkpointing,
        )
        feature_channels = self.backbone.get_feature_channels()
        self.stage_channels = {stage: feature_channels[stage] for stage in self.backbone_stages}
        self.concept_creations = nn.ModuleDict()
        concept_last_transition_only = False if use_temporal_transition_aggregation else last_transition_only
        for stage in self.backbone_stages:
            self.concept_creations[stage] = VisualConceptCreation(
                in_channels=self.stage_channels[stage], concept_dim=concept_dim,
                num_concepts=num_concepts, hidden_dim=concept_hidden_dim, top_k=top_k,
                tau_alpha=tau_alpha, tau_concept=tau_concept,
                max_source_patches=max_source_patches,
                concept_residual_weight=concept_residual_weight, use_target_centric=True,
                last_transition_only=concept_last_transition_only,
                visual_concept_residual_weight=visual_concept_residual_weight,
                visual_assignment_mode=visual_assignment_mode,
                visual_assignment_temperature=visual_assignment_temperature,
                visual_entropy_weight=visual_entropy_weight, visual_usage_weight=visual_usage_weight,
                use_visual_saliency_alignment=use_visual_saliency_alignment,
                visual_saliency_align_weight=visual_saliency_align_weight,
                use_saliency_prototype_roles=use_saliency_prototype_roles,
                num_salient_prototypes=num_salient_prototypes,
                salient_patch_threshold=salient_patch_threshold,
                non_salient_patch_threshold=non_salient_patch_threshold,
                prototype_role_weight=prototype_role_weight,
                prototype_role_cluster_weight=prototype_role_cluster_weight,
                prototype_role_separation_weight=prototype_role_separation_weight,
                prototype_role_margin=prototype_role_margin,
                visual_preservation_weight=visual_preservation_weight,
                preservation_max_patches=preservation_max_patches,
                require_reference_features=require_reference_features,
            )
        self.temporal_feature_infusers = nn.ModuleDict()
        for stage in self.backbone_stages:
            self.temporal_feature_infusers[stage] = SpatioTemporal3DFeatureInfusion(
                in_channels=self.stage_channels[stage], temporal_dim=self.temporal_dim,
                num_blocks=temporal_num_blocks, mlp_ratio=temporal_mlp_ratio,
                dropout=temporal_dropout, use_temporal_difference=temporal_use_difference,
                residual_gate_init=temporal_residual_gate_init,
                enhance_last_only=temporal_enhance_last_only, layer_scale_init=1e-3,
            )
        self.saliency_prediction = ConceptGatedMultiScaleSaliencyDecoder(
            stage_channels=self.stage_channels, concept_dim=concept_dim,
            decoder_channels=saliency_hidden_dim, feature_residual_scale=0.25,
            dropout=0.05, tau_pi=tau_pi, output_activation=output_activation,
            temporal_aggregation=decoder_temporal_aggregation,
            use_side_logit_fusion=decoder_use_side_logit_fusion,
            use_shared_concept_activations=use_shared_concept_activations,
            prototype_bottleneck_strength=prototype_bottleneck_strength,
            prototype_application_position=prototype_application_position,
            fine_unary_mask_strength=fine_unary_mask_strength,
            stage2_backbone_fusion_enabled=stage2_backbone_fusion_enabled,
        )
        if freeze_backbone:
            self.freeze_backbone()
        self._backbone_device: Optional[torch.device] = None
        self._head_device: Optional[torch.device] = None

    @property
    def input_device(self) -> torch.device:
        if self._backbone_device is not None:
            return self._backbone_device
        return next(self.backbone.parameters()).device

    @property
    def output_device(self) -> torch.device:
        return self.input_device if self._head_device is None else self._head_device

    def to_split_devices(self, backbone_device: torch.device, head_device: torch.device):
        self.backbone.to(backbone_device)
        self.concept_creations.to(head_device)
        self.temporal_feature_infusers.to(head_device)
        self.saliency_prediction.to(head_device)
        self._backbone_device = torch.device(backbone_device)
        self._head_device = torch.device(head_device)
        return self

    def _move_features_dict_to_head(self, features_dict: Dict[str, torch.Tensor]):
        if self.output_device == self.input_device:
            return features_dict
        return {stage: value.to(self.output_device, non_blocking=True)
                for stage, value in features_dict.items()}

    def prepare_training_batch(self, rgb_batch, sal_batch, fix_batch):
        return (rgb_batch.to(self.input_device, non_blocking=True),
                sal_batch.to(self.output_device, non_blocking=True),
                fix_batch.to(self.output_device, non_blocking=True))

    def _resize_feature_for_concepts(self, features: torch.Tensor, stage: str):
        cap = self._CONCEPT_MAX_HW_BY_STAGE.get(stage, None)
        if cap is None:
            return features
        _, _, t, h, w = features.shape
        if isinstance(cap, int):
            if h <= cap and w <= cap:
                return features
            return F.interpolate(features, size=(t, cap, cap), mode="trilinear", align_corners=False)
        if isinstance(cap, (tuple, list)) and len(cap) == 2:
            target_h, target_w = int(cap[0]), int(cap[1])
            if h == target_h and w == target_w:
                return features
            return F.interpolate(features, size=(t, target_h, target_w),
                                 mode="trilinear", align_corners=False)
        raise ValueError(f"Invalid concept resize cap for {stage}: {cap!r}.")

    def _normalize_video_layout(self, x: torch.Tensor):
        # Preserve the supplied model's accepted dataloader layout behavior.
        if x.dim() != 5:
            return x
        if x.shape[2] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 1, 4, 2, 3)
        return x

    def _extract_last_rgb_frame(self, x: torch.Tensor):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D video tensor, got shape {tuple(x.shape)}")
        if self.input_format == "BTCHW":
            if x.shape[2] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)
            if x.shape[2] != 3:
                raise ValueError(f"BTCHW input expected 3 channels at dim 2, got {tuple(x.shape)}")
            last_rgb = x[:, -1]
        elif self.input_format == "BCTHW":
            if x.shape[1] != 3:
                raise ValueError(f"BCTHW input expected 3 channels at dim 1, got {tuple(x.shape)}")
            last_rgb = x[:, :, -1]
        else:
            raise ValueError(f"Unsupported input_format: {self.input_format}")
        last_rgb = last_rgb.float()
        if last_rgb.numel() > 0 and last_rgb.max() > 2.0:
            last_rgb = last_rgb / 255.0
        return last_rgb

    def _decoder_output_size(self, last_rgb_frame: torch.Tensor):
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
        params = list(self.concept_creations.parameters())
        if self.use_temporal_feature_infusion:
            params.extend(self.temporal_feature_infusers.parameters())
        params.extend(self.saliency_prediction.parameters())
        if not self._backbone_frozen:
            params.extend(self.backbone.parameters())
        return params

    def optimize_for_inference(self):
        self.eval()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        return self

    def get_temporal_diagnostics(self):
        if not self.use_temporal_feature_infusion:
            return {}
        return {stage: self.temporal_feature_infusers[stage].get_last_diagnostics()
                for stage in self.backbone_stages}

    def _reference_cache_config(self) -> Dict[str, Any]:
        try:
            source = inspect.getsource(type(self.backbone))
        except (OSError, TypeError):
            source = type(self.backbone).__module__ + "." + type(self.backbone).__qualname__
        # Round-trip normalizes tuples to lists for stable metadata comparisons.
        config = {
            "version": self._REFERENCE_CACHE_VERSION,
            "tag": self.reference_cache_tag,
            "backbone_class": type(self.backbone).__module__ + "." + type(self.backbone).__qualname__,
            "backbone_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "input_format": self.input_format,
            "resize_to": getattr(self.backbone, "resize_to", None),
            "normalize": self._reference_backbone_normalize,
            "stages": self.backbone_stages,
            "stage_channels": self.stage_channels,
            "concept_grid_caps": {stage: self._CONCEPT_MAX_HW_BY_STAGE.get(stage)
                                  for stage in self.backbone_stages},
            "temporal_selection": "last_backbone_step",
            "reference_precision": "float32",
        }
        return json.loads(json.dumps(config, sort_keys=True))

    def _get_reference_snapshot(self, cache_dir: Path) -> Dict[str, Any]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "reference_backbone.pt"
        config = self._reference_cache_config()
        if self._reference_snapshot_path == str(path) and path.is_file():
            snapshot = self._reference_snapshot
        else:
            if not path.exists():
                # Include nonpersistent buffers as well as parameters. Reference
                # tensors are independent CPU copies; optimizer parameters stay live.
                state = {}
                for name, value in list(self.backbone.named_parameters()) + list(self.backbone.named_buffers()):
                    copied = value.detach().cpu().clone()
                    if copied.is_floating_point():
                        copied = copied.float()
                    state[name] = copied
                _publish_cache_file({"config": config, "state": state,
                                     "state_sha256": _state_digest(state)}, path)
            snapshot = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(snapshot, dict) or not isinstance(snapshot.get("state"), dict):
                raise ValueError(f"Invalid reference snapshot: {path}")
            state = snapshot["state"]
            if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
                raise ValueError(f"Invalid reference tensors: {path}")
            if _state_digest(state) != snapshot.get("state_sha256"):
                raise ValueError(f"Reference snapshot checksum mismatch: {path}")
        if snapshot is None or snapshot.get("config") != config:
            raise ValueError("Reference-cache preprocessing/architecture differs from this model. "
                             "Use a new reference_cache_dir for this configuration.")
        live_state = dict(list(self.backbone.named_parameters()) + list(self.backbone.named_buffers()))
        if live_state.keys() != snapshot["state"].keys() or any(
            live_state[key].shape != snapshot["state"][key].shape for key in live_state
        ):
            raise ValueError("Reference snapshot does not match this backbone's tensor shapes.")
        self._reference_snapshot_path = str(path)
        self._reference_snapshot = snapshot
        return snapshot

    def _extract_fixed_reference(self, x: torch.Tensor, snapshot: Dict[str, Any]):
        # functional_call substitutes frozen tensors without load_state_dict() or
        # in-place weight updates. Pending training graphs/optimizer state survive.
        state = {"backbone." + name: value.to(x.device).clone()
                 for name, value in snapshot["state"].items()}
        adapter = _ReferenceBackboneForward(self.backbone)
        reference_x = x.float() if x.is_floating_point() else x
        with _reference_eval(self.backbone), torch.no_grad(), torch.autocast(
            device_type=x.device.type, enabled=False
        ):
            raw = functional_call(adapter, state, (reference_x,), strict=True)
            return {stage: self._resize_feature_for_concepts(raw[stage], stage)[:, :, -1:, :, :]
                    .detach().float().cpu().clone() for stage in self.backbone_stages}

    def get_cached_reference_features(
        self, x: torch.Tensor, cache_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load/build references for the exact RGB windows passed in x.

        Hashes include the complete window, shape and dtype, so augmentations and
        different window start positions cannot silently reuse the wrong features.
        Entries are per example, independent of batch order/size. References are
        returned on output_device, detached, as [B,C,1,H,W] per stage.

        Call after loading your intended initial checkpoint and before optimizer
        updates if creating a new cache. An existing directory intentionally keeps
        its saved reference checkpoint across training and process restarts.
        """
        directory = self.reference_cache_dir if cache_dir is None else cache_dir
        if directory is None:
            raise ValueError("Provide reference_cache_dir or cache_dir.")
        x = self._normalize_video_layout(x)
        if x.dim() != 5 or x.shape[0] == 0:
            raise ValueError("Reference caching requires a nonempty 5D RGB batch.")
        if not torch.isfinite(x).all():
            raise ValueError("Reference RGB windows contain NaN or infinity.")
        directory = Path(directory).expanduser().resolve()
        snapshot = self._get_reference_snapshot(directory)
        content_cpu = x.detach().cpu().contiguous()
        digests = [_tensor_digest(sample) for sample in content_cpu]
        paths = {key: directory / "features" / key[:2] / (key + ".pt") for key in digests}
        unique_index = {key: index for index, key in enumerate(digests)}
        missing = [key for key in unique_index if not paths[key].is_file()]
        if missing:
            indices = torch.tensor([unique_index[key] for key in missing], device=x.device)
            missing_x = x.index_select(0, indices).to(self.input_device)
            references = self._extract_fixed_reference(missing_x, snapshot)
            for index, key in enumerate(missing):
                sample_features = {stage: value[index].contiguous().clone()
                                   for stage, value in references.items()}
                self._validate_reference_sample(sample_features)
                _publish_cache_file({
                    "version": self._REFERENCE_CACHE_VERSION,
                    "input_sha256": key, "reference_sha256": snapshot["state_sha256"],
                    "input_shape": list(content_cpu[unique_index[key]].shape),
                    "input_dtype": str(content_cpu.dtype), "features": sample_features,
                }, paths[key])
        entries = {}
        for key in unique_index:
            payload = torch.load(paths[key], map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or (
                payload.get("version") != self._REFERENCE_CACHE_VERSION
                or payload.get("input_sha256") != key
                or payload.get("reference_sha256") != snapshot["state_sha256"]
                or payload.get("input_shape") != list(content_cpu[unique_index[key]].shape)
                or payload.get("input_dtype") != str(content_cpu.dtype)
            ):
                raise ValueError(f"Invalid or incompatible cached reference entry: {paths[key]}")
            self._validate_reference_sample(payload.get("features"))
            entries[key] = payload["features"]
        return {stage: torch.stack([entries[key][stage] for key in digests]).to(self.output_device)
                for stage in self.backbone_stages}

    def _validate_reference_sample(self, features):
        if not isinstance(features, dict) or set(features) != set(self.backbone_stages):
            raise ValueError("Cached reference must contain every requested backbone stage.")
        for stage, value in features.items():
            if (not torch.is_tensor(value) or value.dim() != 4
                    or value.shape[0] != self.stage_channels[stage] or value.shape[1] != 1
                    or min(value.shape[-2:]) <= 0 or value.dtype != torch.float32
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid cached reference feature tensor for {stage}.")

    def forward(
        self, x: torch.Tensor, saliency_maps: Optional[torch.Tensor] = None,
        return_details: Optional[bool] = None, return_concept_losses: Optional[bool] = None,
        collect_gate_debug: bool = False, return_decoder_diagnostics: bool = False,
        reference_cache_dir: Optional[Union[str, Path]] = None,
        use_reference_cache: Optional[bool] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Run the live model and optionally return fixed cached patch references.

        A forward cache directory explicitly enables caching in train/eval. A
        constructor directory enables it by default only in training. Set
        use_reference_cache=False to bypass it. Caching requires return_details=True.
        References are passed into each concept module before computing losses.
        Cached features never replace live decoder inputs.
        """
        if return_details is None:
            return_details = self.return_details
        directory = self.reference_cache_dir if reference_cache_dir is None else reference_cache_dir
        if use_reference_cache is None:
            use_reference_cache = directory is not None and (self.training or reference_cache_dir is not None)
        if use_reference_cache and (directory is None or not return_details):
            raise ValueError("Reference caching requires a directory and return_details=True.")
        inference_ctx = torch.inference_mode if not self.training else nullcontext
        with inference_ctx():
            x = self._normalize_video_layout(x)
            # Compute references before live features; cache misses use no gradients.
            reference_features = self.get_cached_reference_features(x, directory) if use_reference_cache else None
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
            concept_features_shape = {} if return_details else None
            if return_concept_losses is None:
                return_concept_losses = saliency_maps is not None and (
                    self.training or self.allow_eval_concept_losses)
            saliency_maps_for_concepts = saliency_maps if return_concept_losses else None
            for stage in self.backbone_stages:
                stage_features = features_dict[stage]
                concept_features = self._resize_feature_for_concepts(stage_features, stage)
                if return_details:
                    concept_features_shape[stage] = tuple(concept_features.shape)
                self.concept_creations[stage]._diagnostic_label = f"concept.{stage}"
                reference = None
                if reference_features is not None:
                    reference = reference_features[stage]
                    expected = (concept_features.shape[0], concept_features.shape[1], 1,
                                concept_features.shape[3], concept_features.shape[4])
                    if tuple(reference.shape) != expected:
                        raise ValueError(f"Reference/live concept-grid mismatch for {stage}: "
                                         f"{tuple(reference.shape)} != {expected}")
                visual_out = self.concept_creations[stage](
                    concept_features, saliency_maps=saliency_maps_for_concepts,
                    return_losses=return_concept_losses, collect_gate_debug=False,
                    reference_patch_features=reference,
                )
                # Preserve the explicit flattened reference output contract.
                if reference is not None:
                    visual_out["reference_patch_features"] = reference.permute(0, 2, 3, 4, 1).reshape(
                        -1, reference.shape[1])
                concept_outs[stage] = visual_out
                decoder_features_dict[stage] = (
                    self.temporal_feature_infusers[stage](stage_features)
                    if self.use_temporal_feature_infusion else concept_features)
                _diag_record(f"decoder_features.{stage}", decoder_features_dict[stage])
            pred_out = self.saliency_prediction(
                concept_outs=concept_outs, features_dict=decoder_features_dict,
                output_size=self._decoder_output_size(last_rgb_frame),
                return_details=return_decoder_diagnostics,
            )
            if not return_details:
                return pred_out["saliency_map"]
            result = {
                "saliency_map": pred_out["saliency_map"],
                "saliency_logits": pred_out["saliency_logits"],
                "concept_out": concept_outs, "prediction_out": pred_out,
                "concept_features_shape": concept_features_shape,
                "decoder_features_shape": {stage: tuple(value.shape)
                                           for stage, value in decoder_features_dict.items()},
                "features_shape": concept_features_shape,
                "temporal_diagnostics": self.get_temporal_diagnostics(),
            }
            if reference_features is not None:
                result["reference_features"] = reference_features
            return result
