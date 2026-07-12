"""
Visual and motion concept creation for explainable video saliency.

``VisualConceptCreation`` assigns patch-level appearance concepts from
normalized backbone features. ``MotionConceptCreation`` learns motion/change-
sensitive concepts from same-location feature deltas across a temporal window.

Temporal transition/persistence concepts are disabled in the visual branch;
legacy output keys are returned as None for compatibility with older training
and decoding code.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualConceptCreation(nn.Module):
    """
    Patch-level visual concept assignment from backbone features.

    Expected feature input: [B, C, T, H, W].
    Each normalized patch feature z_i^t is encoded and matched to a visual
    concept bank via ``visual_encoder`` and ``visual_concepts``.
    """

    DEFAULT_LOSS_WEIGHTS = {
        "visual": 0.1,
        "visual_div": 0.05,
    }

    DROPOUT_P = 0.2

    def __init__(
        self,
        in_channels: int,
        concept_dim: int = 256,
        num_concepts: int = 32,
        hidden_dim: int = 512,
        top_k: int = 10,
        tau_alpha: float = 0.07,
        tau_concept: float = 0.1,
        diversity_margin: float = 0.2,
        eps_s: float = 0.05,
        eps_p: float = 0.15,
        eps_alpha: float = 0.2,
        eps_v: float = 0.5,
        eps_sal: float = 0.05,
        gate_temp_s: float = 0.03,
        gate_temp_v: float = 0.10,
        gate_temp_p: float = 0.05,
        gate_temp_sal: float = 0.03,
        gate_min_conf: float = 0.10,
        max_source_patches: Optional[int] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        concept_residual_weight: float = 0.1,
        num_visual_concepts: Optional[int] = None,
        visual_concept_residual_weight: float = 0.1,
        visual_assignment_mode: str = "straight_through",
        visual_assignment_temperature: float = 0.07,
        visual_entropy_weight: float = 0.01,
        visual_usage_weight: float = 0.02,
        use_visual_saliency_alignment: bool = True,
        visual_saliency_align_weight: float = 0.05,
        use_target_centric: bool = True,
        last_transition_only: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.tau_alpha = tau_alpha
        self.tau_concept = tau_concept
        self.diversity_margin = diversity_margin
        self.concept_residual_weight = concept_residual_weight
        self.num_visual_concepts = (
            num_visual_concepts if num_visual_concepts is not None else num_concepts
        )
        self.visual_concept_residual_weight = visual_concept_residual_weight
        self.visual_assignment_mode = visual_assignment_mode
        self.visual_assignment_temperature = visual_assignment_temperature
        self.visual_entropy_weight = visual_entropy_weight
        self.visual_usage_weight = visual_usage_weight
        self.use_visual_saliency_alignment = use_visual_saliency_alignment
        self.visual_saliency_align_weight = visual_saliency_align_weight
        self.use_target_centric = use_target_centric
        self.last_transition_only = last_transition_only

        weights = dict(self.DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            weights.update(loss_weights)
        self.loss_weights = weights

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
        self.visual_saliency_head = nn.Sequential(
            nn.Linear(concept_dim * 2 + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.DROPOUT_P),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_concept_parameters()
        self._grid_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}
        self._visual_meta_cache: Dict[Tuple[int, int, int, torch.device], Dict[str, torch.Tensor]] = {}
        self.register_buffer(
            "inv_tau_concept",
            torch.tensor(1.0 / tau_concept),
            persistent=False,
        )

    def _init_concept_parameters(self) -> None:
        with torch.no_grad():
            self.visual_concepts.copy_(
                F.normalize(self.visual_concepts, dim=-1)
            )

    @staticmethod
    def _build_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Normalized patch coordinates in [-1, 1].

        Returns:
            grid: [N, 2] with N=H*W, order (x, y) matching row-major flatten of H x W.
        """
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.reshape(H * W, 2)

    def _make_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (H, W, device, dtype)
        cached = self._grid_cache.get(key)
        if cached is None:
            cached = self._build_grid(H, W, device, dtype)
            self._grid_cache[key] = cached
        return cached

    def _flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, T, H, W]

        Returns:
            z: [B, T, N, C] with L2-normalized channels (N = H*W).
        """
        B, C, T, H, W = features.shape
        z = features.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, C)
        return F.normalize(z, dim=-1)

    def _visual_metadata_indices(
        self,
        B: int,
        T: int,
        N: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        key = (B, T, N, device)
        cached = self._visual_meta_cache.get(key)
        if cached is None:
            patch_idx = torch.arange(N, device=device, dtype=torch.long)
            cached = {
                "batch_idx": torch.arange(B, device=device, dtype=torch.long).repeat_interleave(
                    T * N
                ),
                "time_idx": torch.arange(T, device=device, dtype=torch.long)
                .repeat_interleave(N)
                .repeat(B),
                "patch_idx": patch_idx.repeat(B * T),
            }
            self._visual_meta_cache[key] = cached
        return cached

    def _visual_diversity_loss(self) -> torch.Tensor:
        """Encourage visual concept bank prototypes to stay diverse."""
        bank_n = F.normalize(self.visual_concepts, dim=-1)
        cos = bank_n @ bank_n.T
        mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
        off_diag = cos[mask]
        return F.relu(off_diag - self.diversity_margin).pow(2).mean()

    def _compute_visual_assignments(
        self, raw_similarity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Differentiable visual concept assignment with optional straight-through hardening.

        Training with ``straight_through`` uses hard one-hot activations in the forward
        pass while routing gradients through soft softmax probabilities.
        """
        if self.visual_assignment_mode not in (
            "straight_through",
            "soft",
            "hard_eval",
        ):
            raise ValueError(
                "visual_assignment_mode must be one of "
                "'straight_through', 'soft', or 'hard_eval', "
                f"got {self.visual_assignment_mode!r}"
            )

        temperature = max(float(self.visual_assignment_temperature), 1e-8)
        visual_logits = raw_similarity / temperature
        visual_probs = F.softmax(visual_logits, dim=-1)
        visual_indices = visual_logits.argmax(dim=-1)
        hard_one_hot = F.one_hot(
            visual_indices,
            num_classes=self.num_visual_concepts,
        ).to(dtype=visual_probs.dtype)

        if self.visual_assignment_mode == "soft":
            visual_activations = visual_probs
        elif self.visual_assignment_mode == "hard_eval" or not self.training:
            visual_activations = hard_one_hot
        else:
            visual_activations = hard_one_hot - visual_probs.detach() + visual_probs

        return {
            "visual_logits": visual_logits,
            "visual_probs": visual_probs,
            "visual_activations": visual_activations,
            "visual_indices": visual_indices,
        }

    def _visual_concept_loss(
        self,
        visual_patch_embeddings: torch.Tensor,
        visual_activations: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable reconstruction loss for visual concept assignment."""
        c_vis = F.normalize(self.visual_concepts, dim=-1)
        recon = visual_activations @ c_vis
        return 1.0 - F.cosine_similarity(
            recon, visual_patch_embeddings, dim=-1
        ).mean()

    def _visual_assignment_regularizers(
        self,
        visual_probs: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        entropy = -(visual_probs * (visual_probs + eps).log()).sum(dim=-1).mean()
        mean_usage = visual_probs.mean(dim=0)
        uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        loss_usage = F.kl_div(
            (mean_usage + eps).log(),
            uniform,
            reduction="batchmean",
        )
        return {
            "visual_assignment_entropy": entropy,
            "visual_assignment_usage": mean_usage,
            "loss_visual_assignment_usage": loss_usage,
        }

    @staticmethod
    def _to_float_saliency(saliency_maps: torch.Tensor) -> torch.Tensor:
        sal = saliency_maps.float()
        if sal.numel() > 0 and sal.max() > 2.0:
            sal = sal / 255.0
        return sal

    def _downsample_saliency_to_concept_grid(
        self,
        saliency_maps: torch.Tensor,
        B: int,
        T: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build flattened patch saliency targets and a validity mask on the concept grid.

        Returns:
            targets: [B*T*H*W] in [0, 1]
            valid_mask: [B*T*H*W] bool
        """
        sal = self._to_float_saliency(saliency_maps)
        last_frame_only = False

        if sal.dim() == 3:
            sal = sal.unsqueeze(1)
            last_frame_only = True
        elif sal.dim() == 4:
            if sal.shape[1] == 1:
                last_frame_only = True
        elif sal.dim() == 5:
            if sal.shape[1] == 1:
                sal = sal[:, 0]
            elif sal.shape[2] == 1:
                sal = sal.squeeze(2)
            else:
                raise ValueError(
                    f"Unsupported 5D saliency shape {tuple(saliency_maps.shape)}"
                )
        else:
            raise ValueError(
                f"Unsupported saliency_maps shape {tuple(saliency_maps.shape)}"
            )

        if last_frame_only:
            if sal.dim() == 4:
                sal = sal[:, 0]
            if sal.dim() != 3:
                raise ValueError(
                    f"Expected last-frame saliency [B,H,W], got {tuple(sal.shape)}"
                )
            sal_last = sal
            resized = F.interpolate(
                sal_last.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            resized = resized.clamp(0.0, 1.0).to(device=device, dtype=dtype)
            targets_bt = torch.zeros(B, T, H, W, device=device, dtype=dtype)
            targets_bt[:, T - 1] = resized
            valid = torch.zeros(B, T, H, W, dtype=torch.bool, device=device)
            valid[:, T - 1] = True
        else:
            if sal.dim() != 4:
                raise ValueError(
                    f"Expected temporal saliency [B,T,H,W], got {tuple(sal.shape)}"
                )
            _, T_sal, H_i, W_i = sal.shape
            sal_flat = sal.reshape(B * T_sal, 1, H_i, W_i)
            sal_resized = F.interpolate(
                sal_flat,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, T_sal, H, W).clamp(0.0, 1.0).to(device=device, dtype=dtype)
            if T_sal != T:
                sal_resized = F.interpolate(
                    sal_resized.unsqueeze(1),
                    size=(T, H, W),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(1)
            targets_bt = sal_resized
            valid = torch.ones(B, T, H, W, dtype=torch.bool, device=device)

        if sal.shape[0] != B:
            raise ValueError(
                f"Saliency batch size {sal.shape[0]} does not match feature batch {B}"
            )

        targets = targets_bt.reshape(B * T * H * W)
        valid_mask = valid.reshape(B * T * H * W)
        return targets, valid_mask

    def _visual_saliency_alignment_loss(
        self,
        q_vis: torch.Tensor,
        visual_repr: torch.Tensor,
        visual_patch_coords: torch.Tensor,
        saliency_maps: Optional[torch.Tensor],
        visual_metadata: Dict[str, Any],
        *,
        reference: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = reference.sum() * 0.0
        if (
            not self.use_visual_saliency_alignment
            or saliency_maps is None
            or not torch.is_tensor(saliency_maps)
        ):
            return {
                "loss_visual_saliency_align": zero,
                "visual_saliency_patch_logits": None,
                "visual_saliency_align_valid_frac": zero.detach(),
            }

        feature_shape = visual_metadata["feature_shape"]
        B = int(feature_shape["B"])
        T = int(feature_shape["T"])
        H = int(feature_shape["H"])
        W = int(feature_shape["W"])

        targets, valid_mask = self._downsample_saliency_to_concept_grid(
            saliency_maps,
            B,
            T,
            H,
            W,
            device=q_vis.device,
            dtype=q_vis.dtype,
        )
        head_input = torch.cat([q_vis, visual_repr, visual_patch_coords], dim=-1)
        patch_logits = self.visual_saliency_head(head_input).squeeze(-1)

        if valid_mask.any():
            loss = F.binary_cross_entropy_with_logits(
                patch_logits[valid_mask],
                targets[valid_mask],
            )
        else:
            loss = zero

        return {
            "loss_visual_saliency_align": loss,
            "visual_saliency_patch_logits": patch_logits.detach(),
            "visual_saliency_align_valid_frac": valid_mask.float().mean().detach(),
        }

    def _build_visual_concepts_from_patches(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Build visual-only concept assignments from individual patch features.

        Args:
            features: [B, C, T, H, W]

        Returns:
            visual_patch_embeddings: [B*T*N, concept_dim]
            visual_concept_logits: [B*T*N, num_visual_concepts]
            visual_concept_indices: [B*T*N]
            visual_activations: [B*T*N, num_visual_concepts]
            visual_concept_representation: [B*T*N, concept_dim]
            visual_metadata: dict with batch_idx, time_idx, patch_idx, patch_coords, feature_shape
        """
        B, C, T, H, W = features.shape
        device = features.device
        dtype = features.dtype
        N = H * W

        z = self._flatten_features(features)
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
        }

        return {
            "visual_patch_embeddings": q_vis,
            "visual_concept_logits": visual_logits,
            "visual_concept_indices": visual_indices,
            "visual_activations": visual_activations,
            "visual_concept_representation": visual_repr,
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
        collect_gate_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, T, H, W] frozen backbone features.
            saliency_maps: optional GT saliency for training-only auxiliary losses.
            return_losses: whether to compute visual concept auxiliary losses.
            collect_gate_debug: ignored (kept for API compatibility).

        Returns:
            Visual concept outputs plus legacy temporal keys set to None.
        """
        del collect_gate_debug

        if features.dim() != 5:
            raise ValueError(
                f"features must be [B,C,T,H,W], got shape {tuple(features.shape)}"
            )
        if features.size(2) < 1:
            raise ValueError(
                f"Need T>=1 for visual concepts, got T={features.size(2)}"
            )

        visual_out = self._build_visual_concepts_from_patches(features)

        align_out: Dict[str, Any] = {
            "loss_visual_saliency_align": None,
            "visual_saliency_patch_logits": None,
            "visual_saliency_align_valid_frac": None,
        }

        losses: Dict[str, torch.Tensor] = {}
        if return_losses:
            loss_visual = self._visual_concept_loss(
                visual_out["visual_patch_embeddings"],
                visual_out["visual_activations"],
            )
            loss_visual_div = self._visual_diversity_loss()
            entropy = visual_out["visual_assignment_entropy"]
            loss_usage = visual_out["loss_visual_assignment_usage"]
            align_out = self._visual_saliency_alignment_loss(
                visual_out["visual_patch_embeddings"],
                visual_out["visual_concept_representation"],
                visual_out["visual_patch_coords"],
                saliency_maps,
                visual_out["visual_metadata"],
                reference=loss_visual,
            )
            loss_visual_saliency_align = align_out["loss_visual_saliency_align"]
            w = self.loss_weights
            losses = {
                "loss_visual": loss_visual,
                "loss_visual_div": loss_visual_div,
                "loss_visual_assignment_entropy": entropy,
                "loss_visual_assignment_usage": loss_usage,
                "loss_visual_saliency_align": loss_visual_saliency_align,
                "loss_total_concept": (
                    w["visual"] * loss_visual
                    + w["visual_div"] * loss_visual_div
                    + self.visual_entropy_weight * entropy
                    + self.visual_usage_weight * loss_usage
                    + self.visual_saliency_align_weight * loss_visual_saliency_align
                ),
            }

        return {
            "trajectory_vectors": None,
            "trajectory_embeddings": None,
            "concept_representation": None,
            "transition_activations": None,
            "persistence_activations": None,
            "gate_probs": None,
            "metadata": None,
            "losses": losses,
            "visual_patch_embeddings": visual_out["visual_patch_embeddings"],
            "visual_concept_representation": visual_out["visual_concept_representation"],
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
        }

    @torch.no_grad()
    def summarize_gate_debug(
        self,
        saliency_maps: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        gate_probs: torch.Tensor,
        feature_shape: Tuple[int, ...],
    ) -> Dict[str, float]:
        """Legacy no-op: temporal gate debug is disabled in visual-only mode."""
        del saliency_maps, metadata, gate_probs, feature_shape
        return {
            "gate_valid_frac_total": 0.0,
            "gate_transition_frac_total": 0.0,
            "gate_persistence_frac_total": 0.0,
            "gate_ambiguous_frac_total": 1.0,
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


class MotionConceptCreation(nn.Module):
    """
    Motion/change-sensitive concept assignment from spatiotemporal feature deltas.

    Learns prototypes over same-location temporal differences in backbone
    features ``[B, C, T, H, W]``. Unlike appearance concepts, motion concepts
    summarize how features change across time at each spatial cell rather than
    static patch appearance. An LSTM consumes the ordered sequence of signed
    consecutive differences per patch, preserving direction and transition order
    while supporting variable temporal lengths (``T >= 2``).
    """

    def __init__(
        self,
        in_channels: int,
        concept_dim: int = 256,
        num_motion_concepts: int = 32,
        hidden_dim: int = 512,
        assignment_temperature: float = 0.07,
        assignment_mode: str = "straight_through",
        diversity_margin: float = 0.2,
        dropout: float = 0.2,
        motionness_temperature: float = 2.0,
        motionness_sparsity_weight: float = 0.01,
        motion_recon_weight: float = 0.1,
        motion_div_weight: float = 0.05,
        motion_entropy_weight: float = 0.01,
        motion_usage_weight: float = 0.02,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = False,
        lstm_dropout: float = 0.0,
    ):
        super().__init__()

        if int(lstm_hidden_dim) <= 0:
            raise ValueError(f"lstm_hidden_dim must be > 0, got {lstm_hidden_dim!r}")
        if int(lstm_num_layers) < 1:
            raise ValueError(f"lstm_num_layers must be >= 1, got {lstm_num_layers!r}")

        self.in_channels = in_channels
        self.concept_dim = concept_dim
        self.num_motion_concepts = num_motion_concepts
        self.hidden_dim = hidden_dim
        self.assignment_temperature = assignment_temperature
        self.assignment_mode = assignment_mode
        self.diversity_margin = diversity_margin
        self.dropout_p = dropout
        self.motionness_temperature = motionness_temperature
        self.motionness_sparsity_weight = motionness_sparsity_weight
        self.motion_recon_weight = motion_recon_weight
        self.motion_div_weight = motion_div_weight
        self.motion_entropy_weight = motion_entropy_weight
        self.motion_usage_weight = motion_usage_weight
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_num_layers = int(lstm_num_layers)
        self.lstm_bidirectional = bool(lstm_bidirectional)
        # PyTorch LSTM dropout is only active between stacked layers (num_layers > 1).
        effective_lstm_dropout = (
            float(lstm_dropout) if self.lstm_num_layers > 1 else 0.0
        )

        # Per-timestep projection of each signed consecutive difference (C -> hidden).
        self.motion_input_projection = nn.Sequential(
            nn.Linear(in_channels, self.lstm_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.lstm_hidden_dim),
        )
        # Ordered temporal-difference encoder (variable-length sequences, T >= 2).
        self.motion_lstm = nn.LSTM(
            input_size=self.lstm_hidden_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=effective_lstm_dropout,
            bidirectional=self.lstm_bidirectional,
        )
        lstm_output_dim = self.lstm_hidden_dim * (2 if self.lstm_bidirectional else 1)
        # Post-LSTM projection to the concept space (name kept as ``motion_encoder``
        # so optimizer parameter grouping / external references remain valid).
        self.motion_encoder = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, concept_dim),
            nn.LayerNorm(concept_dim),
        )
        self.motion_concepts = nn.Parameter(
            torch.randn(num_motion_concepts, concept_dim)
        )

        self._init_motion_concept_parameters()
        self._grid_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}
        self._motion_meta_cache: Dict[Tuple[int, int, torch.device], Dict[str, torch.Tensor]] = {}

    def _init_motion_concept_parameters(self) -> None:
        with torch.no_grad():
            self.motion_concepts.copy_(
                F.normalize(self.motion_concepts, dim=-1)
            )

    @staticmethod
    def _build_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Normalized patch coordinates in [-1, 1], shape [H*W, 2]."""
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.reshape(H * W, 2)

    def _make_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (H, W, device, dtype)
        cached = self._grid_cache.get(key)
        if cached is None:
            cached = self._build_grid(H, W, device, dtype)
            self._grid_cache[key] = cached
        return cached

    def _motion_metadata_indices(
        self,
        B: int,
        N: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        key = (B, N, device)
        cached = self._motion_meta_cache.get(key)
        if cached is None:
            patch_idx = torch.arange(N, device=device, dtype=torch.long)
            cached = {
                "batch_idx": torch.arange(B, device=device, dtype=torch.long).repeat_interleave(
                    N
                ),
                "patch_idx": patch_idx.repeat(B),
            }
            self._motion_meta_cache[key] = cached
        return cached

    def _motion_diversity_loss(self) -> torch.Tensor:
        """Encourage motion concept bank prototypes to stay diverse."""
        bank_n = F.normalize(self.motion_concepts, dim=-1)
        cos = bank_n @ bank_n.T
        mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
        off_diag = cos[mask]
        return F.relu(off_diag - self.diversity_margin).pow(2).mean()

    def _compute_motion_assignments(
        self, raw_similarity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Differentiable motion concept assignment with optional straight-through hardening."""
        if self.assignment_mode not in (
            "straight_through",
            "soft",
            "hard_eval",
        ):
            raise ValueError(
                "assignment_mode must be one of "
                "'straight_through', 'soft', or 'hard_eval', "
                f"got {self.assignment_mode!r}"
            )

        temperature = max(float(self.assignment_temperature), 1e-8)
        motion_logits = raw_similarity / temperature
        motion_probs = F.softmax(motion_logits, dim=-1)
        motion_indices = motion_logits.argmax(dim=-1)
        hard_one_hot = F.one_hot(
            motion_indices,
            num_classes=self.num_motion_concepts,
        ).to(dtype=motion_probs.dtype)

        if self.assignment_mode == "soft":
            motion_activations = motion_probs
        elif self.assignment_mode == "hard_eval" or not self.training:
            motion_activations = hard_one_hot
        else:
            motion_activations = hard_one_hot - motion_probs.detach() + motion_probs

        return {
            "motion_logits": motion_logits,
            "motion_probs": motion_probs,
            "motion_activations": motion_activations,
            "motion_indices": motion_indices,
        }

    def _motion_assignment_regularizers(
        self,
        motion_probs: torch.Tensor,
        motionness_weights: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        entropy = -(motion_probs * (motion_probs + eps).log()).sum(dim=-1).mean()
        weights = motionness_weights.detach()
        weight_sum = weights.sum() + eps
        mean_usage = (motion_probs * weights.unsqueeze(-1)).sum(dim=0) / weight_sum
        uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        loss_usage = F.kl_div(
            (mean_usage + eps).log(),
            uniform,
            reduction="batchmean",
        )
        return {
            "motion_assignment_entropy": entropy,
            "motion_assignment_usage": mean_usage,
            "loss_motion_assignment_usage": loss_usage,
        }

    def _compute_motionness_map(self, abs_delta: torch.Tensor) -> torch.Tensor:
        """
        Build a per-cell soft motionness map from absolute temporal deltas.

        Args:
            abs_delta: [B, C, T-1, H, W]

        Returns:
            motionness: [B, 1, H, W]
        """
        motion_energy = abs_delta.norm(dim=1).mean(dim=1, keepdim=True)
        mean = motion_energy.mean(dim=(-2, -1), keepdim=True)
        std = motion_energy.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        normalized_energy = (motion_energy - mean) / std
        return torch.sigmoid(self.motionness_temperature * normalized_energy)

    def _build_motion_concepts(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Build motion concepts from same-location temporal feature differences.

        Args:
            features: [B, C, T, H, W] with T >= 2
        """
        B, C, T, H, W = features.shape
        device = features.device
        dtype = features.dtype
        N = H * W

        if T < 2:
            raise ValueError(
                f"MotionConceptCreation requires T >= 2, got T={T}"
            )

        delta = features[:, :, 1:] - features[:, :, :-1]  # [B, C, T-1, H, W]
        # abs_delta is still required by the motionness computation below.
        abs_delta = delta.abs()

        # Ordered per-patch difference sequence [delta_0, ..., delta_(T-2)], each
        # delta_t = features[:,:,t+1] - features[:,:,t] holding all C channels.
        delta_sequence = (
            delta.permute(0, 3, 4, 2, 1)
            .contiguous()
            .reshape(B * N, T - 1, C)
        )  # [B*N, T-1, C]

        # Project each timestep, then encode the ordered sequence with an LSTM.
        projected_sequence = self.motion_input_projection(
            delta_sequence
        )  # [B*N, T-1, lstm_hidden_dim]
        lstm_output, (h_n, c_n) = self.motion_lstm(projected_sequence)

        # Summarize the sequence from the final hidden states (no pooling). Using
        # h_n keeps forward/backward directions correctly aligned: for a
        # bidirectional LSTM the backward stream's summary lives at t=0 in
        # lstm_output, so lstm_output[:, -1] alone would be direction-inconsistent.
        num_directions = 2 if self.lstm_bidirectional else 1
        h_n = h_n.view(
            self.lstm_num_layers,
            num_directions,
            B * N,
            self.lstm_hidden_dim,
        )
        final_layer_h = h_n[-1]  # [num_directions, B*N, lstm_hidden_dim]
        if self.lstm_bidirectional:
            trajectory_repr = torch.cat(
                [final_layer_h[0], final_layer_h[1]], dim=-1
            )  # [B*N, 2*lstm_hidden_dim]
        else:
            trajectory_repr = final_layer_h[0]  # [B*N, lstm_hidden_dim]

        q_motion = self.motion_encoder(trajectory_repr)  # [B*N, concept_dim]
        q_motion = F.normalize(q_motion, dim=-1, eps=1e-8)

        c_motion = F.normalize(self.motion_concepts, dim=-1)
        raw_similarity = q_motion @ c_motion.T
        assignment_out = self._compute_motion_assignments(raw_similarity)
        motion_logits = assignment_out["motion_logits"]
        motion_probs = assignment_out["motion_probs"]
        motion_indices = assignment_out["motion_indices"]
        motion_activations = assignment_out["motion_activations"]

        motion_repr = motion_activations @ c_motion
        motion_repr = F.normalize(motion_repr, dim=-1)

        motionness_map = self._compute_motionness_map(abs_delta)
        motionness_flat = motionness_map.reshape(B * N, 1)
        motion_repr = F.normalize(motion_repr * motionness_flat + 1e-6, dim=-1)

        reg_out = self._motion_assignment_regularizers(motion_probs, motionness_flat.squeeze(-1))

        grid = self._make_grid(H, W, device, dtype)
        meta_idx = self._motion_metadata_indices(B, N, device)
        patch_idx = meta_idx["patch_idx"]
        patch_coords = grid[patch_idx]

        motion_metadata: Dict[str, Any] = {
            "batch_idx": meta_idx["batch_idx"],
            "patch_idx": patch_idx,
            "patch_coords": patch_coords,
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
        }

        return {
            "motion_patch_embeddings": q_motion,
            "motion_concept_representation": motion_repr,
            "motion_activations": motion_activations,
            "motion_concept_logits": motion_logits,
            "motion_concept_indices": motion_indices,
            "motion_assignment_probs": motion_probs,
            "motion_assignment_entropy": reg_out["motion_assignment_entropy"],
            "motion_assignment_usage": reg_out["motion_assignment_usage"],
            "loss_motion_assignment_usage": reg_out["loss_motion_assignment_usage"],
            "motionness_map": motionness_map,
            "motion_metadata": motion_metadata,
        }

    def _motion_reconstruction_loss(
        self,
        motion_patch_embeddings: torch.Tensor,
        motion_activations: torch.Tensor,
        motionness_weights: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        c_motion = F.normalize(self.motion_concepts, dim=-1)
        recon = motion_activations @ c_motion
        per_patch_loss = 1.0 - F.cosine_similarity(
            recon, motion_patch_embeddings, dim=-1
        )
        weights = motionness_weights.detach()
        return (per_patch_loss * weights).sum() / (weights.sum() + eps)

    def _build_empty_motion_output(
        self, features: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Zero / uniform motion outputs for inference when T < 2.

        Temporal deltas are undefined, so representations are zeros and
        assignment tensors default to a uniform distribution.
        """
        B, C, T, H, W = features.shape
        device = features.device
        dtype = features.dtype
        N = H * W
        K = self.num_motion_concepts
        num_patches = B * N

        uniform = torch.full((num_patches, K), 1.0 / K, device=device, dtype=dtype)
        zeros_patch = torch.zeros(num_patches, self.concept_dim, device=device, dtype=dtype)
        zeros_logits = torch.zeros(num_patches, K, device=device, dtype=dtype)
        indices = torch.zeros(num_patches, device=device, dtype=torch.long)
        motionness_map = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        usage = torch.full((K,), 1.0 / K, device=device, dtype=dtype)
        entropy = torch.zeros((), device=device, dtype=dtype)
        loss_usage = torch.zeros((), device=device, dtype=dtype)

        grid = self._make_grid(H, W, device, dtype)
        meta_idx = self._motion_metadata_indices(B, N, device)
        patch_idx = meta_idx["patch_idx"]
        patch_coords = grid[patch_idx]

        motion_metadata: Dict[str, Any] = {
            "batch_idx": meta_idx["batch_idx"],
            "patch_idx": patch_idx,
            "patch_coords": patch_coords,
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
        }

        return {
            "motion_patch_embeddings": zeros_patch,
            "motion_concept_representation": zeros_patch,
            "motion_activations": uniform,
            "motion_concept_logits": zeros_logits,
            "motion_concept_indices": indices,
            "motion_assignment_probs": uniform,
            "motion_assignment_entropy": entropy,
            "motion_assignment_usage": usage,
            "loss_motion_assignment_usage": loss_usage,
            "motionness_map": motionness_map,
            "motion_metadata": motion_metadata,
        }

    def _zero_motion_losses(self, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Zero-valued motion losses for the T < 2 inference fallback."""
        zero = reference.sum() * 0.0
        return {
            "loss_motion": zero,
            "loss_motion_div": zero,
            "loss_motion_assignment_entropy": zero,
            "loss_motion_assignment_usage": zero,
            "loss_motionness_sparsity": zero,
            "loss_total_motion_concept": zero,
        }

    def _compute_motion_losses(
        self, motion_out: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        B = int(motion_out["motion_metadata"]["feature_shape"]["B"])
        H = int(motion_out["motion_metadata"]["feature_shape"]["H"])
        W = int(motion_out["motion_metadata"]["feature_shape"]["W"])
        motionness_weights = motion_out["motionness_map"].reshape(B * H * W)

        loss_motion = self._motion_reconstruction_loss(
            motion_out["motion_patch_embeddings"],
            motion_out["motion_activations"],
            motionness_weights,
        )
        loss_motion_div = self._motion_diversity_loss()
        loss_motion_assignment_entropy = motion_out["motion_assignment_entropy"]
        loss_motion_assignment_usage = motion_out["loss_motion_assignment_usage"]
        loss_motionness_sparsity = motion_out["motionness_map"].mean()

        return {
            "loss_motion": loss_motion,
            "loss_motion_div": loss_motion_div,
            "loss_motion_assignment_entropy": loss_motion_assignment_entropy,
            "loss_motion_assignment_usage": loss_motion_assignment_usage,
            "loss_motionness_sparsity": loss_motionness_sparsity,
            "loss_total_motion_concept": (
                self.motion_recon_weight * loss_motion
                + self.motion_div_weight * loss_motion_div
                + self.motion_entropy_weight * loss_motion_assignment_entropy
                + self.motion_usage_weight * loss_motion_assignment_usage
                + self.motionness_sparsity_weight * loss_motionness_sparsity
            ),
        }

    def forward(
        self,
        features: torch.Tensor,
        return_losses: bool = True,
        collect_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            features: [B, C, T, H, W] backbone features.
            return_losses: whether to compute motion concept auxiliary losses.
            collect_debug: reserved for future diagnostics (currently ignored).

        Returns:
            Motion concept outputs and optional ``losses`` dict.
        """
        del collect_debug

        if features.dim() != 5:
            raise ValueError(
                f"features must be [B,C,T,H,W], got shape {tuple(features.shape)}"
            )

        T = features.size(2)
        if T < 2:
            if self.training:
                raise ValueError(
                    f"Need T>=2 for motion concepts during training, got T={T}"
                )
            motion_out = self._build_empty_motion_output(features)
            losses: Dict[str, torch.Tensor] = (
                self._zero_motion_losses(features) if return_losses else {}
            )
            return {
                "motion_patch_embeddings": motion_out["motion_patch_embeddings"],
                "motion_concept_representation": motion_out["motion_concept_representation"],
                "motion_activations": motion_out["motion_activations"],
                "motion_concept_logits": motion_out["motion_concept_logits"],
                "motion_concept_indices": motion_out["motion_concept_indices"],
                "motion_assignment_probs": motion_out["motion_assignment_probs"],
                "motion_assignment_entropy": motion_out["motion_assignment_entropy"],
                "motion_assignment_usage": motion_out["motion_assignment_usage"],
                "motionness_map": motion_out["motionness_map"],
                "motion_metadata": motion_out["motion_metadata"],
                "losses": losses,
            }

        motion_out = self._build_motion_concepts(features)
        losses = self._compute_motion_losses(motion_out) if return_losses else {}

        return {
            "motion_patch_embeddings": motion_out["motion_patch_embeddings"],
            "motion_concept_representation": motion_out["motion_concept_representation"],
            "motion_activations": motion_out["motion_activations"],
            "motion_concept_logits": motion_out["motion_concept_logits"],
            "motion_concept_indices": motion_out["motion_concept_indices"],
            "motion_assignment_probs": motion_out["motion_assignment_probs"],
            "motion_assignment_entropy": motion_out["motion_assignment_entropy"],
            "motion_assignment_usage": motion_out["motion_assignment_usage"],
            "motionness_map": motion_out["motionness_map"],
            "motion_metadata": motion_out["motion_metadata"],
            "losses": losses,
        }


# Backward-compatible alias for existing imports.
ConceptCreation = VisualConceptCreation
