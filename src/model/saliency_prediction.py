"""
Concept-conditioned saliency prediction for the last frame in a video window.

Aggregates trajectory-level saliency scores onto the feature patch grid using
incoming transition affinities, then optionally refines with target-frame Video Swin
features before upsampling to RGB resolution.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SaliencyPrediction(nn.Module):
    """
    Predict last-frame saliency from ConceptCreation trajectory representations.

    Uses trajectory-level concept representations, incoming trajectory aggregation
    (pi), target-patch concept context maps (G_b = sum_a pi_ab * concept_repr),
    gate-weighted transition/persistence activation regions, concept-context
    refinement of patch saliency logits, and optional Video Swin feature refinement.

    Per trajectory a -> b (into the last feature time step):
        s_tilde_b(a) = R_s(concept_repr)
        delta_tilde_ab = R_delta(concept_repr)

    Incoming aggregation per target patch b:
        pi_ab = softmax(incoming_score / tau_pi) over trajectories landing on b
        (incoming_score prefers affinity_logit, else alpha)
s_hat_b = sum_a pi_ab * s_tilde_b(a)
    """

    DROPOUT_P = 0.2

    def __init__(
        self,
        concept_dim: int = 256,
        hidden_dim: int = 256,
        tau_pi: float = 0.1,
        feature_channels: Optional[int] = None,
        use_feature_refinement: bool = True,
        feature_refine_channels: int = 128,
        use_rgb_refinement: bool = False,
        rgb_refine_channels: int = 32,
        use_concept_context_refinement: bool = True,
        concept_context_channels: int = 128,
        use_peak_refinement: bool = True,
        peak_refine_channels: int = 128,
        peak_residual_scale: float = 0.3,
        concept_gate_temperature: float = 0.15,
        concept_gate_threshold: float = 0.5,
        feature_residual_scale: float = 0.2,
        use_gated_trajectory_head: bool = True,
        gated_trajectory_residual_scale: float = 0.2,
        output_activation: str = "sigmoid",
        predict_delta: bool = True,
    ):
        super().__init__()

        if output_activation not in ("sigmoid", "none"):
            raise ValueError("output_activation must be 'sigmoid' or 'none'")

        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.tau_pi = tau_pi
        self.feature_channels = feature_channels
        self.use_feature_refinement = use_feature_refinement
        self.feature_refine_channels = feature_refine_channels
        self.use_rgb_refinement = use_rgb_refinement
        self.rgb_refine_channels = rgb_refine_channels
        self.use_concept_context_refinement = use_concept_context_refinement
        self.concept_context_channels = concept_context_channels
        self.use_peak_refinement = use_peak_refinement
        self.peak_refine_channels = peak_refine_channels
        self.peak_residual_scale = peak_residual_scale
        self.concept_gate_temperature = concept_gate_temperature
        self.concept_gate_threshold = concept_gate_threshold
        self.feature_residual_scale = feature_residual_scale
        self.use_gated_trajectory_head = use_gated_trajectory_head
        self.gated_trajectory_residual_scale = gated_trajectory_residual_scale
        self.output_activation = output_activation
        self.predict_delta = predict_delta

        self.saliency_head = self._make_prediction_head()
        self.delta_head = self._make_prediction_head() if predict_delta else None

        if use_gated_trajectory_head:
            if feature_channels is None:
                raise ValueError(
                    "feature_channels must be provided when use_gated_trajectory_head=True"
                )
            gated_input_dim = concept_dim + 3 * feature_channels + 1
            self.gated_trajectory_saliency_head = self._make_prediction_head(
                input_dim=gated_input_dim
            )
        else:
            self.gated_trajectory_saliency_head = None

        if use_peak_refinement:
            self.peak_refiner = nn.Sequential(
                nn.Conv2d(concept_dim + 1, peak_refine_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(peak_refine_channels, peak_refine_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(peak_refine_channels, 1, kernel_size=1),
            )
        else:
            self.peak_refiner = None

        if use_concept_context_refinement:
            self.concept_context_refiner = nn.Sequential(
                nn.Conv2d(concept_dim + 1, concept_context_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(self.DROPOUT_P),
                nn.Conv2d(
                    concept_context_channels, concept_context_channels, kernel_size=3, padding=1
                ),
                nn.GELU(),
                nn.Dropout2d(self.DROPOUT_P),
                nn.Conv2d(concept_context_channels, 1, kernel_size=1),
            )
        else:
            self.concept_context_refiner = None

        self.rgb_refiner = None

        if use_feature_refinement:
            if feature_channels is None:
                raise ValueError(
                    "feature_channels must be provided when use_feature_refinement=True"
                )
            self.feature_refiner = nn.Sequential(
                nn.Conv2d(3 * feature_channels + 2, feature_refine_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(self.DROPOUT_P),
                nn.Conv2d(
                    feature_refine_channels, feature_refine_channels, kernel_size=3, padding=1
                ),
                nn.GELU(),
                nn.Dropout2d(self.DROPOUT_P),
                nn.Conv2d(feature_refine_channels, 1, kernel_size=1),
            )
        else:
            self.feature_refiner = None

    def _make_prediction_head(self, input_dim: Optional[int] = None) -> nn.Sequential:
        if input_dim is None:
            input_dim = self.concept_dim
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.DROPOUT_P),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _to_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device == device and value.dtype == torch.long:
                return value if value.dim() == 1 else value.reshape(-1)
            return value.to(device=device, dtype=torch.long).reshape(-1)
        return torch.tensor(value, device=device, dtype=torch.long).reshape(-1)

    @staticmethod
    def _to_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.device == device and value.dtype == torch.float32:
                return value if value.dim() == 1 else value.reshape(-1)
            return value.to(device=device, dtype=torch.float32).reshape(-1)
        return torch.tensor(value, device=device, dtype=torch.float32).reshape(-1)

    def _prepare_last_frame(self, last_rgb_frame: torch.Tensor) -> torch.Tensor:
        """
        Normalize last-frame RGB to [B, 3, H, W] in [0, 1].

        Accepts [B,3,H,W], [B,H,W,3], [B,T,3,H,W], [B,T,H,W,3], or [B,3,T,H,W].
        """
        x = last_rgb_frame
        if x.dim() == 5:
            if x.shape[1] == 3:
                x = x[:, :, -1, :, :]  # [B, 3, H, W]
            elif x.shape[2] == 3:
                x = x[:, -1, :, :, :]  # [B, 3, H, W]
            elif x.shape[-1] == 3:
                x = x[:, -1, :, :, :].permute(0, 3, 1, 2)  # [B, T, H, W, 3]
            else:
                raise ValueError(
                    "5D last_rgb_frame must have channel dim 1, 2, or last (-1)"
                )
        elif x.dim() == 4:
            if x.shape[1] != 3 and x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2).contiguous()
            elif x.shape[1] != 3:
                raise ValueError("4D last_rgb_frame must be [B,3,H,W] or [B,H,W,3]")
        else:
            raise ValueError(
                f"last_rgb_frame must be 4D or 5D, got shape {tuple(x.shape)}"
            )

        x = x.float()
        if x.numel() > 0 and x.max() > 2.0:
            x = x / 255.0
        return x

    def _get_feature_shape(
        self, concept_out: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[int, int, int, int, int]:
        """
        Resolve B, T, H, W, N from metadata.

        Returns:
            B, T, H, W, N with N = H * W.
        """
        device = self._resolve_device(concept_out, metadata)
        feature_shape = metadata.get("feature_shape")

        if feature_shape is not None:
            if isinstance(feature_shape, dict):
                B = int(feature_shape["B"])
                T = int(feature_shape["T"])
                H = int(feature_shape["H"])
                W = int(feature_shape["W"])
            else:
                shape = tuple(int(v) for v in feature_shape)
                if len(shape) != 5:
                    raise ValueError(
                        f"feature_shape must have 5 entries (B,C,T,H,W), got {shape}"
                    )
                B, _, T, H, W = shape
        else:
            batch_idx = self._to_long_tensor(metadata["batch_idx"], device)
            time_idx = self._to_long_tensor(metadata["time_idx"], device)
            target_idx = self._to_long_tensor(metadata["target_idx"], device)

            B = int(batch_idx.max().item()) + 1
            T = int(time_idx.max().item()) + 2
            N = int(target_idx.max().item()) + 1
            root = int(math.isqrt(N))
            if root * root != N:
                raise ValueError(
                    f"Cannot infer square patch grid: N={N} is not a perfect square"
                )
            H = W = root

        if T < 2:
            raise ValueError(
                f"Feature time dimension T must be >= 2, got T={T}"
            )

        N = H * W
        return B, T, H, W, N

    @staticmethod
    def _resolve_device(
        concept_out: Dict[str, Any], metadata: Dict[str, Any]
    ) -> torch.device:
        concept_repr = concept_out["concept_representation"]
        if isinstance(concept_repr, torch.Tensor):
            return concept_repr.device
        for key in ("batch_idx", "target_idx", "alpha"):
            val = metadata.get(key)
            if isinstance(val, torch.Tensor):
                return val.device
        return torch.device("cpu")

    def _select_last_transition(
        self,
        concept_out: Dict[str, Any],
        metadata: Dict[str, Any],
        B: int,
        T: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Keep trajectories with time_idx == T-2 (final step into last feature frame)."""
        device = concept_out["concept_representation"].device
        time_idx = self._to_long_tensor(metadata["time_idx"], device)
        last_t = T - 2
        mask = time_idx == last_t

        if not mask.any():
            raise ValueError(
                f"No trajectories found for last transition time_idx={last_t} (T={T})"
            )

        concept_repr = concept_out["concept_representation"]
        if concept_repr.shape[0] != time_idx.shape[0]:
            raise ValueError(
                "concept_representation length does not match metadata trajectory count"
            )

        selected_meta: Dict[str, torch.Tensor] = {}
        required_keys = (
            "batch_idx",
            "time_idx",
            "source_idx",
            "target_idx",
            "source_coords",
            "target_coords",
            "alpha",
        )
        scalar_keys = {
            "batch_idx",
            "time_idx",
            "source_idx",
            "target_idx",
            "alpha",
            "affinity_logit",
        }
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"metadata missing required key '{key}'")
            tensor = metadata[key]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(device)
            else:
                tensor = torch.as_tensor(tensor, device=device)
            if tensor.shape[0] != time_idx.shape[0]:
                raise ValueError(
                    f"metadata['{key}'] length {tensor.shape[0]} != "
                    f"trajectory count {time_idx.shape[0]}"
                )
            if tensor.dim() == 1 or key in scalar_keys:
                selected_meta[key] = tensor.reshape(-1)[mask]
            else:
                selected_meta[key] = tensor[mask]

        if "affinity_logit" in metadata:
            tensor = metadata["affinity_logit"]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.to(device)
            else:
                tensor = torch.as_tensor(tensor, device=device)
            if tensor.shape[0] != time_idx.shape[0]:
                raise ValueError(
                    f"metadata['affinity_logit'] length {tensor.shape[0]} != "
                    f"trajectory count {time_idx.shape[0]}"
                )
            selected_meta["affinity_logit"] = tensor.reshape(-1)[mask]

        if "feature_shape" in metadata:
            selected_meta["feature_shape"] = metadata["feature_shape"]

        return concept_repr[mask], selected_meta, mask

    def _select_last_transition_tensor(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if tensor.shape[0] != mask.shape[0]:
            raise ValueError(
                f"{name} length {tensor.shape[0]} does not match trajectory count "
                f"{mask.shape[0]}"
            )
        return tensor[mask]

    def _incoming_softmax(
        self, scores: torch.Tensor, group_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Softmax over incoming trajectories sharing the same target patch.

        Args:
            scores: [M] incoming scores (affinity_logit or alpha).
            group_ids: [M] with group_id = batch_idx * N + target_idx.

        Returns:
            pi: [M] normalized incoming weights.
        """
        if scores.numel() == 0:
            return torch.zeros_like(scores)

        scaled = scores / self.tau_pi
        group_ids_long = group_ids.long()
        num_groups = int(group_ids_long.max().item()) + 1

        group_max = torch.full(
            (num_groups,),
            float("-inf"),
            device=scores.device,
            dtype=scores.dtype,
        )
        group_max.scatter_reduce_(
            0, group_ids_long, scaled, reduce="amax", include_self=True
        )
        shifted = scaled - group_max[group_ids_long]
        exp_scores = torch.exp(shifted)

        group_sum = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
        group_sum.scatter_add_(0, group_ids_long, exp_scores)
        return exp_scores / group_sum[group_ids_long]

    def _patch_coverage_count(
        self,
        batch_idx: torch.Tensor,
        target_idx: torch.Tensor,
        B: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Number of trajectories landing on each target patch [B, 1, H, W]."""
        device = batch_idx.device
        N = H * W
        flat_idx = batch_idx.long() * N + target_idx.long()
        counts = torch.zeros(B * N, device=device, dtype=torch.float32)
        ones = torch.ones(flat_idx.shape[0], device=device, dtype=torch.float32)
        counts.index_add_(0, flat_idx, ones)
        return counts.view(B, 1, H, W)

    def _aggregate_to_patch_grid(
        self,
        values: torch.Tensor,
        pi: torch.Tensor,
        batch_idx: torch.Tensor,
        target_idx: torch.Tensor,
        B: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Aggregate trajectory scalars onto the patch grid.

        Args:
            values: [M]
            pi: [M]
            batch_idx, target_idx: [M]

        Returns:
            patch_map: [B, 1, H, W]
        """
        device = values.device
        N = H * W
        flat_idx = batch_idx.long() * N + target_idx.long()
        weighted = (values * pi).to(dtype=values.dtype)
        patch_flat = torch.zeros(B * N, device=device, dtype=values.dtype)
        patch_flat.index_add_(0, flat_idx, weighted)
        return patch_flat.view(B, 1, H, W)

    def _aggregate_vector_to_patch_grid(
        self,
        values: torch.Tensor,
        pi: torch.Tensor,
        batch_idx: torch.Tensor,
        target_idx: torch.Tensor,
        B: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Aggregate trajectory vector values onto target patch grid.

        Args:
            values: [M, D]
            pi: [M]
            batch_idx: [M]
            target_idx: [M]

        Returns:
            patch_map: [B, D, H, W]
        """
        if values.dim() != 2:
            raise ValueError(f"values must be [M,D], got {tuple(values.shape)}")

        device = values.device
        dtype = values.dtype
        M, D = values.shape
        N = H * W

        if pi.shape[0] != M:
            raise ValueError(
                f"pi length {pi.shape[0]} must match values length {M}"
            )

        flat_idx = batch_idx.long() * N + target_idx.long()
        weighted = (values * pi.unsqueeze(-1)).to(dtype=dtype)

        patch_flat = torch.zeros(B * N, D, device=device, dtype=dtype)
        patch_flat.index_add_(0, flat_idx, weighted)

        return patch_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

    def _prepare_target_features(
        self,
        video_features: Optional[torch.Tensor],
        target_h: int,
        target_w: int,
    ) -> Optional[torch.Tensor]:
        if video_features is None:
            return None

        if video_features.dim() != 5:
            raise ValueError(
                f"video_features must be [B,C,T,H,W], got {tuple(video_features.shape)}"
            )

        target_features = video_features[:, :, -1, :, :]  # [B, C, Hf, Wf]

        if target_features.shape[-2:] != (target_h, target_w):
            target_features = F.interpolate(
                target_features,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        return target_features

    def _minmax_per_sample_map(
        self,
        x: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Per-sample min-max normalization for [B,1,H,W].
        """
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError(f"x must be [B,1,H,W], got {tuple(x.shape)}")
        B = x.shape[0]
        flat = x.reshape(B, -1)
        xmin = flat.min(dim=1, keepdim=True)[0]
        xmax = flat.max(dim=1, keepdim=True)[0]
        flat = (flat - xmin) / (xmax - xmin + eps)
        return flat.view_as(x)

    def _gather_target_features_flat(
        self,
        video_features: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Return last-frame target features as [B, N, C] after resizing if needed.
        """
        target_features = self._prepare_target_features(video_features, target_h, target_w)
        if target_features is None:
            raise ValueError("video_features must be passed when use_feature_refinement=True")

        B, C, H, W = target_features.shape
        return target_features.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def _gather_source_features_for_trajectories(
        self,
        video_features: torch.Tensor,
        selected_meta: Dict[str, torch.Tensor],
        B: int,
        T: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Gather source-frame features z_a for selected trajectories.

        Returns:
            source_features: [M, C]
        """
        if video_features is None:
            raise ValueError("video_features must be passed to gather source features")
        if video_features.dim() != 5:
            raise ValueError(
                f"video_features must be [B,C,T,H,W], got {tuple(video_features.shape)}"
            )

        if video_features.shape[-2:] != (H, W):
            video_features = F.interpolate(
                video_features,
                size=(video_features.shape[2], H, W),
                mode="trilinear",
                align_corners=False,
            )

        batch_idx = selected_meta["batch_idx"].long()
        time_idx = selected_meta["time_idx"].long()
        source_idx = selected_meta["source_idx"].long()

        Bf, C, Tf, Hf, Wf = video_features.shape
        if Bf != B or Hf != H or Wf != W:
            raise ValueError(
                f"video feature shape mismatch: got {(Bf, C, Tf, Hf, Wf)}, "
                f"expected B={B}, H={H}, W={W}"
            )

        N = H * W
        feat_flat = video_features.permute(0, 2, 3, 4, 1).reshape(B, Tf, N, C)
        source_features = feat_flat[batch_idx, time_idx, source_idx]
        return source_features

    def _gather_source_target_features_for_trajectories(
        self,
        video_features: torch.Tensor,
        selected_meta: Dict[str, torch.Tensor],
        B: int,
        T: int,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather source and target Swin features for selected trajectories.

        Returns:
            source_features: [M, C] for z_a at frame t
            target_features: [M, C] for z_b at frame t+1
        """
        if video_features is None:
            raise ValueError("video_features must be passed to gather trajectory features")
        if video_features.dim() != 5:
            raise ValueError(
                f"video_features must be [B,C,T,H,W], got {tuple(video_features.shape)}"
            )

        if video_features.shape[-2:] != (H, W):
            video_features = F.interpolate(
                video_features,
                size=(video_features.shape[2], H, W),
                mode="trilinear",
                align_corners=False,
            )

        batch_idx = selected_meta["batch_idx"].long()
        time_idx = selected_meta["time_idx"].long()
        source_idx = selected_meta["source_idx"].long()
        target_idx = selected_meta["target_idx"].long()

        Bf, C, Tf, Hf, Wf = video_features.shape
        if Bf != B or Hf != H or Wf != W:
            raise ValueError(
                f"video feature shape mismatch: got {(Bf, C, Tf, Hf, Wf)}, "
                f"expected B={B}, H={H}, W={W}"
            )

        if time_idx.max().item() + 1 >= Tf:
            raise ValueError(
                f"Selected target time index exceeds feature time dimension Tf={Tf}"
            )

        N = H * W
        feat_flat = video_features.permute(0, 2, 3, 4, 1).reshape(B, Tf, N, C)

        source_features = feat_flat[batch_idx, time_idx, source_idx]
        target_features = feat_flat[batch_idx, time_idx + 1, target_idx]

        return source_features, target_features

    def _compute_trajectory_concept_strength(
        self,
        concept_out: Dict[str, Any],
        last_transition_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute a scalar concept activation strength per selected trajectory.

        Important:
        Do NOT sum all concept activations, because softmax activations sum to 1.
        Use max activation/peakiness so the score reflects how strongly a trajectory
        matches a specific transition/persistence concept.
        """
        required = ("transition_activations", "persistence_activations", "gate_probs")
        if not all(k in concept_out for k in required):
            return None

        transition_activations = self._select_last_transition_tensor(
            concept_out["transition_activations"],
            last_transition_mask,
            "transition_activations",
        )
        persistence_activations = self._select_last_transition_tensor(
            concept_out["persistence_activations"],
            last_transition_mask,
            "persistence_activations",
        )
        gate_probs = self._select_last_transition_tensor(
            concept_out["gate_probs"],
            last_transition_mask,
            "gate_probs",
        )

        tr_strength = transition_activations.max(dim=1).values
        per_strength = persistence_activations.max(dim=1).values

        concept_strength = (
            gate_probs[:, 0] * tr_strength + gate_probs[:, 1] * per_strength
        )
        return concept_strength

    def sharpen_saliency_map(
        self,
        pred: torch.Tensor,
        gamma: float = 5.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        pred = pred.clamp_min(0)
        pred = pred / (pred.max() + eps)
        pred = pred.pow(gamma)
        pred = pred / (pred.max() + eps)
        return pred

    def forward(
        self,
        concept_out: Dict[str, Any],
        last_rgb_frame: torch.Tensor,
        video_features: Optional[torch.Tensor] = None,
        return_details: bool = False,
        last_rgb_prepared: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict saliency map for the last frame in the window.

        Args:
            concept_out: output dict from ConceptCreation.
            last_rgb_frame: last RGB frame(s); see ``_prepare_last_frame``.
            video_features: optional backbone features [B, C, T, H, W] for refinement.
            return_details: if False, omit ``selected_metadata`` from the output.

        Returns:
            Dictionary with saliency_map, logits, patch maps, and trajectory-level preds.
        """
        if "concept_representation" not in concept_out:
            raise ValueError("concept_out must contain 'concept_representation'")
        if "metadata" not in concept_out:
            raise ValueError("concept_out must contain 'metadata'")

        metadata = concept_out["metadata"]
        required_meta = ("batch_idx", "time_idx", "target_idx", "alpha")
        for key in required_meta:
            if key not in metadata:
                raise ValueError(f"metadata must contain '{key}'")

        last_rgb = (
            last_rgb_frame
            if last_rgb_prepared
            else self._prepare_last_frame(last_rgb_frame)
        )
        concept_repr = concept_out["concept_representation"]
        if not isinstance(concept_repr, torch.Tensor):
            raise ValueError("concept_representation must be a torch.Tensor")

        B, T, H, W, N = self._get_feature_shape(concept_out, metadata)
        if last_rgb.shape[0] != B:
            raise ValueError(
                f"last_rgb_frame batch {last_rgb.shape[0]} != feature batch {B}"
            )

        concept_repr, selected_meta, last_transition_mask = self._select_last_transition(
            concept_out, metadata, B, T
        )
        M = concept_repr.shape[0]
        device = concept_repr.device

        batch_idx = self._to_long_tensor(selected_meta["batch_idx"], device)
        target_idx = self._to_long_tensor(selected_meta["target_idx"], device)
        alpha = self._to_float_tensor(selected_meta["alpha"], device)

        if "affinity_logit" in selected_meta:
            incoming_score = self._to_float_tensor(
                selected_meta["affinity_logit"], device
            )
        else:
            incoming_score = alpha

        if (
            batch_idx.shape[0] != M
            or target_idx.shape[0] != M
            or incoming_score.shape[0] != M
        ):
            raise ValueError("Selected metadata length mismatch after last-transition filter")

        trajectory_concept_strength = self._compute_trajectory_concept_strength(
            concept_out,
            last_transition_mask,
        )

        if trajectory_concept_strength is None:
            trajectory_concept_strength = torch.ones(
                concept_repr.shape[0],
                device=concept_repr.device,
                dtype=concept_repr.dtype,
            )

        concept_only_saliency_logits = self.saliency_head(concept_repr).squeeze(-1)

        gated_trajectory_residual_logits = None
        saliency_logits = concept_only_saliency_logits

        if self.use_gated_trajectory_head and self.gated_trajectory_saliency_head is not None:
            if video_features is None:
                raise ValueError(
                    "video_features must be passed when use_gated_trajectory_head=True"
                )

            source_traj_features, target_traj_features = (
                self._gather_source_target_features_for_trajectories(
                    video_features,
                    selected_meta,
                    B,
                    T,
                    H,
                    W,
                )
            )

            strength = trajectory_concept_strength.unsqueeze(-1)

            gated_target_traj_features = strength * target_traj_features
            gated_source_traj_features = strength * source_traj_features
            gated_delta_traj_features = strength * (
                target_traj_features - source_traj_features
            )

            gated_head_input = torch.cat(
                [
                    concept_repr,
                    gated_target_traj_features,
                    gated_source_traj_features,
                    gated_delta_traj_features,
                    strength,
                ],
                dim=-1,
            )

            gated_trajectory_residual_logits = (
                self.gated_trajectory_saliency_head(gated_head_input).squeeze(-1)
            )

            saliency_logits = (
                concept_only_saliency_logits
                + self.gated_trajectory_residual_scale
                * gated_trajectory_residual_logits
            )

        delta_logits = (
            self.delta_head(concept_repr).squeeze(-1)
            if self.predict_delta and self.delta_head is not None
            else None
        )

        group_ids = batch_idx * N + target_idx
        pi = self._incoming_softmax(incoming_score, group_ids)

        patch_concept_gate_logits = self._aggregate_to_patch_grid(
            trajectory_concept_strength,
            pi,
            batch_idx,
            target_idx,
            B,
            H,
            W,
        )

        patch_concept_gate_norm = self._minmax_per_sample_map(patch_concept_gate_logits)

        patch_concept_gate = torch.sigmoid(
            (patch_concept_gate_norm - self.concept_gate_threshold)
            / max(self.concept_gate_temperature, 1e-6)
        )

        patch_concept_context = self._aggregate_vector_to_patch_grid(
            concept_repr,
            pi,
            batch_idx,
            target_idx,
            B,
            H,
            W,
        )

        patch_transition_activation = None
        patch_persistence_activation = None
        patch_transition_region = None
        patch_persistence_region = None

        if (
            "transition_activations" in concept_out
            and "persistence_activations" in concept_out
            and "gate_probs" in concept_out
        ):
            transition_activations = concept_out["transition_activations"][
                last_transition_mask
            ]
            persistence_activations = concept_out["persistence_activations"][
                last_transition_mask
            ]
            gate_probs = concept_out["gate_probs"][last_transition_mask]

            transition_weighted = gate_probs[:, 0:1] * transition_activations
            persistence_weighted = gate_probs[:, 1:2] * persistence_activations

            patch_transition_activation = self._aggregate_vector_to_patch_grid(
                transition_weighted,
                pi,
                batch_idx,
                target_idx,
                B,
                H,
                W,
            )
            patch_persistence_activation = self._aggregate_vector_to_patch_grid(
                persistence_weighted,
                pi,
                batch_idx,
                target_idx,
                B,
                H,
                W,
            )

            patch_transition_region = patch_transition_activation.sum(dim=1, keepdim=True)
            patch_persistence_region = patch_persistence_activation.sum(dim=1, keepdim=True)

        patch_coverage_count = self._patch_coverage_count(
            batch_idx, target_idx, B, H, W
        )

        patch_saliency_logits = self._aggregate_to_patch_grid(
            saliency_logits, pi, batch_idx, target_idx, B, H, W
        )
        concept_only_patch_saliency_logits = self._aggregate_to_patch_grid(
            concept_only_saliency_logits,
            pi,
            batch_idx,
            target_idx,
            B,
            H,
            W,
        )
        patch_delta_logits = None
        if delta_logits is not None:
            patch_delta_logits = self._aggregate_to_patch_grid(
                delta_logits, pi, batch_idx, target_idx, B, H, W
            )

        coarse_patch_logits = patch_saliency_logits
        peak_residual_logits = None
        peak_refined_patch_logits = coarse_patch_logits

        if self.use_peak_refinement and self.peak_refiner is not None:
            peak_residual_logits = self.peak_refiner(
                torch.cat([coarse_patch_logits, patch_concept_context], dim=1)
            )
            peak_refined_patch_logits = (
                coarse_patch_logits + self.peak_residual_scale * peak_residual_logits
            )

        concept_context_residual_logits = None
        concept_context_patch_logits = peak_refined_patch_logits

        if self.use_concept_context_refinement and self.concept_context_refiner is not None:
            concept_context_residual_logits = self.concept_context_refiner(
                torch.cat([peak_refined_patch_logits, patch_concept_context], dim=1)
            )
            concept_context_patch_logits = (
                peak_refined_patch_logits + concept_context_residual_logits
            )

        concept_logits = F.interpolate(
            concept_context_patch_logits,
            size=last_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.output_activation == "sigmoid":
            concept_saliency_map = torch.sigmoid(concept_logits)
        else:
            concept_saliency_map = concept_logits

        if self.use_feature_refinement and self.feature_refiner is not None:
            target_features = self._prepare_target_features(
                video_features,
                concept_context_patch_logits.shape[-2],
                concept_context_patch_logits.shape[-1],
            )
            if target_features is None:
                raise ValueError(
                    "video_features must be passed when use_feature_refinement=True"
                )

            source_features = self._gather_source_features_for_trajectories(
                video_features,
                selected_meta,
                B,
                T,
                H,
                W,
            )

            source_evidence_features = self._aggregate_vector_to_patch_grid(
                source_features * trajectory_concept_strength.unsqueeze(-1),
                pi,
                batch_idx,
                target_idx,
                B,
                H,
                W,
            )

            gated_target_features = target_features * patch_concept_gate
            target_minus_source_features = (
                gated_target_features - source_evidence_features
            )

            feature_residual_patch_logits = self.feature_refiner(
                torch.cat(
                    [
                        concept_context_patch_logits,
                        patch_concept_gate,
                        gated_target_features,
                        source_evidence_features,
                        target_minus_source_features,
                    ],
                    dim=1,
                )
            )

            final_patch_logits = (
                concept_context_patch_logits
                + self.feature_residual_scale
                * patch_concept_gate
                * feature_residual_patch_logits
            )
        else:
            target_features = None
            source_evidence_features = None
            gated_target_features = None
            target_minus_source_features = None
            feature_residual_patch_logits = None
            final_patch_logits = concept_context_patch_logits

        final_logits = F.interpolate(
            final_patch_logits,
            size=last_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.output_activation == "sigmoid":
            saliency_map = torch.sigmoid(final_logits)
        else:
            saliency_map = final_logits

        concept_only_logits = F.interpolate(
            concept_only_patch_saliency_logits,
            size=last_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.output_activation == "sigmoid":
            concept_only_saliency_map = torch.sigmoid(concept_only_logits)
        else:
            concept_only_saliency_map = concept_only_logits
        sharpened_saliency_map = self.sharpen_saliency_map(saliency_map)
        out: Dict[str, torch.Tensor] = {
            "saliency_map": saliency_map,
            "sharpened_saliency_map": sharpened_saliency_map,
            "saliency_logits": final_logits,
            "patch_saliency_logits": final_patch_logits,
            "coarse_patch_logits": coarse_patch_logits,
            "peak_residual_logits": peak_residual_logits,
            "peak_refined_patch_logits": peak_refined_patch_logits,
            "final_patch_logits": final_patch_logits,
            "concept_saliency_map": concept_saliency_map,
            "concept_saliency_logits": concept_logits,
            "concept_patch_saliency_logits": coarse_patch_logits,
            "concept_context_patch_logits": concept_context_patch_logits,
            "concept_context_residual_logits": concept_context_residual_logits,
            "patch_concept_context": patch_concept_context,
            "trajectory_concept_strength": trajectory_concept_strength,
            "patch_concept_gate_logits": patch_concept_gate_logits,
            "patch_concept_gate_norm": patch_concept_gate_norm,
            "patch_concept_gate": patch_concept_gate,
            "gated_target_features": gated_target_features,
            "source_evidence_features": source_evidence_features,
            "target_minus_source_features": target_minus_source_features,
            "feature_residual_scale": torch.as_tensor(
                self.feature_residual_scale,
                device=final_patch_logits.device,
                dtype=final_patch_logits.dtype,
            ),
            "patch_transition_activation": patch_transition_activation,
            "patch_persistence_activation": patch_persistence_activation,
            "patch_transition_region": patch_transition_region,
            "patch_persistence_region": patch_persistence_region,
            "feature_residual_patch_logits": feature_residual_patch_logits,
            "incoming_weights": pi,
            "incoming_scores": incoming_score,
            "patch_coverage_count": patch_coverage_count,
            "trajectory_saliency_logits": saliency_logits,
            "trajectory_delta_logits": delta_logits,
            "patch_delta_logits": patch_delta_logits,
            "concept_only_saliency_map": concept_only_saliency_map,
            "concept_only_saliency_logits": concept_only_logits,
            "concept_only_patch_saliency_logits": concept_only_patch_saliency_logits,
            "concept_only_trajectory_saliency_logits": concept_only_saliency_logits,
            "gated_trajectory_residual_logits": gated_trajectory_residual_logits,
            "gated_trajectory_residual_scale": torch.as_tensor(
                self.gated_trajectory_residual_scale,
                device=saliency_logits.device,
                dtype=saliency_logits.dtype,
            ),
        }
        if return_details:
            out["selected_metadata"] = selected_meta
        return out


class MultiScaleSaliencyPrediction(nn.Module):
    """
    Fuse per-stage SaliencyPrediction outputs into a single last-frame saliency map.

    Each stage uses trajectory concepts plus that stage's target-frame Video Swin features.
    Fusion operates on RGB-resolution logits before the final output activation.
    """

    def __init__(
        self,
        stage_channels: Dict[str, int],
        concept_dim: int = 256,
        hidden_dim: int = 256,
        tau_pi: float = 0.5,
        output_activation: str = "sigmoid",
        use_feature_refinement: bool = True,
        feature_refine_channels: int = 128,
        use_peak_refinement: bool = True,
        peak_refine_channels: int = 128,
        peak_residual_scale: float = 0.3,
        fusion_hidden_channels: int = 64,
        predict_delta: bool = True,
        use_gated_trajectory_head: bool = True,
        gated_trajectory_residual_scale: float = 0.2,
    ):
        super().__init__()

        if output_activation not in ("sigmoid", "none"):
            raise ValueError("output_activation must be 'sigmoid' or 'none'")

        self.stage_names = tuple(stage_channels.keys())
        self.stage_channels = dict(stage_channels)
        self.output_activation = output_activation

        self.stage_predictors = nn.ModuleDict()
        for stage, channels in self.stage_channels.items():
            self.stage_predictors[stage] = SaliencyPrediction(
                concept_dim=concept_dim,
                hidden_dim=hidden_dim,
                tau_pi=tau_pi,
                feature_channels=channels,
                use_feature_refinement=use_feature_refinement,
                feature_refine_channels=feature_refine_channels,
                use_peak_refinement=use_peak_refinement,
                peak_refine_channels=peak_refine_channels,
                peak_residual_scale=peak_residual_scale,
                use_rgb_refinement=False,
                output_activation="none",
                predict_delta=predict_delta,
                use_gated_trajectory_head=use_gated_trajectory_head,
                gated_trajectory_residual_scale=gated_trajectory_residual_scale,
            )

        num_stages = len(self.stage_names)
        self.fusion_head = nn.Sequential(
            nn.Conv2d(num_stages, fusion_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_hidden_channels, fusion_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_hidden_channels, 1, kernel_size=1),
        )
        self.stage_fusion_logits = nn.Parameter(torch.zeros(len(self.stage_names)))
        self._eval_stage_weights: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._eval_stage_weights = None
        return self

    def _get_stage_weights(self) -> torch.Tensor:
        if not self.training:
            if self._eval_stage_weights is None:
                self._eval_stage_weights = torch.softmax(
                    self.stage_fusion_logits, dim=0
                ).view(1, -1, 1, 1)
            return self._eval_stage_weights
        return torch.softmax(self.stage_fusion_logits, dim=0).view(1, -1, 1, 1)

    def forward(
        self,
        concept_outs: Dict[str, Dict[str, Any]],
        last_rgb_frame: torch.Tensor,
        video_features_dict: Dict[str, torch.Tensor],
        return_details: bool = False,
    ) -> Dict[str, Any]:
        stage_outs: Dict[str, Dict[str, Any]] = {}
        stage_logits = []
        stage_concept_logits = []
        stage_concept_only_logits = []

        first_stage = self.stage_names[0]
        last_rgb = self.stage_predictors[first_stage]._prepare_last_frame(last_rgb_frame)

        for stage in self.stage_names:
            if stage not in concept_outs:
                raise ValueError(f"Missing concept_out for stage {stage}")
            if stage not in video_features_dict:
                raise ValueError(f"Missing video features for stage {stage}")

            out_s = self.stage_predictors[stage](
                concept_outs[stage],
                last_rgb,
                video_features=video_features_dict[stage],
                return_details=return_details,
                last_rgb_prepared=True,
            )

            stage_outs[stage] = out_s
            stage_logits.append(out_s["saliency_logits"])
            if "concept_saliency_logits" in out_s:
                stage_concept_logits.append(out_s["concept_saliency_logits"])
            if "concept_only_saliency_logits" in out_s:
                stage_concept_only_logits.append(out_s["concept_only_saliency_logits"])

        multi_stage_logits = torch.cat(stage_logits, dim=1)
        stage_weights = self._get_stage_weights()
        weighted_base_logits = (multi_stage_logits * stage_weights).sum(
            dim=1, keepdim=True
        )
        fusion_residual_logits = self.fusion_head(multi_stage_logits)
        final_logits = weighted_base_logits + fusion_residual_logits

        if self.output_activation == "sigmoid":
            saliency_map = torch.sigmoid(final_logits)
        else:
            saliency_map = final_logits

        concept_saliency_map = None
        concept_logits = None
        if stage_concept_logits:
            multi_stage_concept_logits = torch.cat(stage_concept_logits, dim=1)
            concept_logits = multi_stage_concept_logits.mean(dim=1, keepdim=True)
            if self.output_activation == "sigmoid":
                concept_saliency_map = torch.sigmoid(concept_logits)
            else:
                concept_saliency_map = concept_logits

        concept_only_saliency_map = None
        concept_only_logits = None

        if stage_concept_only_logits:
            multi_stage_concept_only_logits = torch.cat(stage_concept_only_logits, dim=1)
            concept_only_logits = multi_stage_concept_only_logits.mean(dim=1, keepdim=True)

            if self.output_activation == "sigmoid":
                concept_only_saliency_map = torch.sigmoid(concept_only_logits)
            else:
                concept_only_saliency_map = concept_only_logits

        main_stage = "stage1" if "stage1" in stage_outs else self.stage_names[0]
        main_out = stage_outs[main_stage]

        out: Dict[str, Any] = {
            "saliency_map": saliency_map,
            "saliency_logits": final_logits,
            "main_stage": main_stage,
            "concept_saliency_map": concept_saliency_map,
            "concept_saliency_logits": concept_logits,
            "concept_only_saliency_map": concept_only_saliency_map,
            "concept_only_saliency_logits": concept_only_logits,
            "stage_fusion_weights": stage_weights.detach().reshape(-1),
        }

        for key in [
            "concept_context_patch_logits",
            "concept_context_residual_logits",
            "patch_concept_context",
            "patch_concept_gate",
            "patch_concept_gate_logits",
            "gated_target_features",
            "source_evidence_features",
            "patch_transition_activation",
            "patch_persistence_activation",
            "patch_transition_region",
            "patch_persistence_region",
        ]:
            if key in main_out:
                out[key] = main_out[key]

        for key in [
            "coarse_patch_logits",
            "peak_residual_logits",
            "final_patch_logits",
            "patch_saliency_logits",
            "concept_patch_saliency_logits",
            "concept_only_patch_saliency_logits",
            "concept_only_trajectory_saliency_logits",
            "gated_trajectory_residual_logits",
            "gated_trajectory_residual_scale",
            "incoming_weights",
            "incoming_scores",
            "patch_coverage_count",
            "trajectory_saliency_logits",
            "trajectory_delta_logits",
            "patch_delta_logits",
            "selected_metadata",
        ]:
            if key in main_out:
                out[key] = main_out[key]

        rgb_size = final_logits.shape[-2:]
        transition_regions = [
            F.interpolate(tr, size=rgb_size, mode="bilinear", align_corners=False)
            for out_s in stage_outs.values()
            if (tr := out_s.get("patch_transition_region")) is not None
        ]
        persistence_regions = [
            F.interpolate(per, size=rgb_size, mode="bilinear", align_corners=False)
            for out_s in stage_outs.values()
            if (per := out_s.get("patch_persistence_region")) is not None
        ]

        if transition_regions:
            out["multiscale_transition_region"] = torch.stack(
                transition_regions, dim=0
            ).mean(dim=0)
        if persistence_regions:
            out["multiscale_persistence_region"] = torch.stack(
                persistence_regions, dim=0
            ).mean(dim=0)

        return out
