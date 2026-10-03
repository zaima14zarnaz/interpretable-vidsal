"""
Visual concept creation for explainable video saliency.

``VisualConceptCreation`` assigns patch-level appearance concepts from
normalized backbone features.

The fixed bank roles are trained using scene-relative GT fixation-density
targets. Forward matching never uses those labels to choose a prototype.
The original cosine reconstruction is retained; a sampled relational loss
preserves similarities from a fixed cached backbone snapshot during training.
Projection is an explicit, role-restricted training-data sweep: begin, update,
then finalize (or use the companion prototype_projection.py helper).

Temporal transition/persistence concepts are disabled in the visual branch;
legacy output keys are returned as None for compatibility with older training
and decoding code.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diagnostic_trace import record as _diag_record


class VisualConceptCreation(nn.Module):
    """
    Patch-level visual concept assignment from backbone features.

    Expected feature input: [B, C, T, H, W].
    Patch-concept assignments are computed on the last frame only; earlier
    frames in the window are retained in metadata via ``window_T`` for the
    decoder. Each normalized last-frame patch feature is encoded and matched
    to a visual concept bank via ``visual_encoder`` and ``visual_concepts``.
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
        visual_usage_weight: float = 0.001,
        use_visual_saliency_alignment: bool = True,
        visual_saliency_align_weight: float = 0.05,
        use_target_centric: bool = True,
        last_transition_only: bool = True,
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
        self.use_saliency_prototype_roles = bool(use_saliency_prototype_roles)
        self.num_salient_prototypes = (
            max(1, self.num_visual_concepts // 2)
            if num_salient_prototypes is None else int(num_salient_prototypes)
        )
        if self.use_saliency_prototype_roles and not (
            0 < self.num_salient_prototypes < self.num_visual_concepts
        ):
            raise ValueError("Both attention-role banks must contain at least one prototype.")
        if not 0 <= non_salient_patch_threshold < salient_patch_threshold <= 1:
            raise ValueError("Require 0 <= low threshold < high threshold <= 1.")
        if preservation_max_patches < 2:
            raise ValueError("preservation_max_patches must be >= 2.")
        if visual_preservation_weight < 0:
            raise ValueError("visual_preservation_weight must be nonnegative.")
        self.salient_patch_threshold = float(salient_patch_threshold)
        self.non_salient_patch_threshold = float(non_salient_patch_threshold)
        self.prototype_role_weight = float(prototype_role_weight)
        self.prototype_role_cluster_weight = float(prototype_role_cluster_weight)
        self.prototype_role_separation_weight = float(prototype_role_separation_weight)
        self.prototype_role_margin = float(prototype_role_margin)
        self.visual_preservation_weight = float(visual_preservation_weight)
        self.preservation_max_patches = int(preservation_max_patches)
        self.require_reference_features = bool(require_reference_features)
        if self.top_k < 1 or self.num_visual_concepts < 1:
            raise ValueError("top_k and the number of visual concepts must be positive.")
        # Bank indices are stable: first bank high attention (1), second low (0).
        self.register_buffer("prototype_roles", (
            torch.arange(self.num_visual_concepts) < self.num_salient_prototypes
        ).long())
        self._projection_state = None

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
        if off_diag.numel() == 0:
            return self.visual_concepts.sum() * 0.0
        return F.relu(off_diag - self.diversity_margin).pow(2).mean()

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
        validity_eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-K active concept prototypes independently at each patch.

        Returns:
            active_prototypes: [B, T, H, W, K, concept_dim]
            validity_mask: [B, T, H, W, K] bool
            active_indices: [B, T, H, W, K] int64 bank indices per patch
        """
        num_patches = B * T * H * W
        if activations.shape[0] != num_patches:
            raise ValueError(
                "activations must be [B*T*H*W, num_concepts], "
                f"expected {num_patches} rows, got {activations.shape[0]}"
            )

        num_concepts = int(prototypes.shape[0])
        K = min(int(top_k), num_concepts)

        topk_vals, topk_idx = activations.topk(K, dim=-1)
        proto_n = F.normalize(prototypes, dim=-1)
        active = proto_n[topk_idx]
        validity = topk_vals > validity_eps

        if K < top_k:
            pad_k = top_k - K
            active = F.pad(active, (0, 0, 0, pad_k))
            validity = F.pad(validity, (0, pad_k), value=False)
            topk_idx = F.pad(topk_idx, (0, pad_k), value=0)

        active = active.view(B, T, H, W, top_k, -1)
        validity = validity.view(B, T, H, W, top_k)
        active_indices = topk_idx.view(B, T, H, W, top_k)

        return active, validity, active_indices

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
            reduction="sum",
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

    def _patch_attention_roles(self, saliency_maps, metadata, device):
        """Area-mean density relative to the frame's maximum patch density.

        Labels: 1 high attention, 0 low attention, -1 ambiguous/uninformative.
        Use fixation-density maps rather than interpreting missing binary
        fixations as conclusive negative labels. Empty/constant maps are ignored.
        These are scene-relative attention roles, not object-class labels.
        """
        shape = metadata["feature_shape"]
        B, H, W = shape["B"], shape["H"], shape["W"]
        labels = torch.full((B * H * W,), -1, device=device, dtype=torch.long)
        targets = torch.zeros(B * H * W, device=device)
        if saliency_maps is None:
            return labels, targets
        sal = self._to_float_saliency(saliency_maps.detach()).to(device)
        if sal.dim() == 5:
            if sal.shape[1] == 1:  # [B,1,T,H,W]
                sal = sal[:, 0]
            elif sal.shape[2] == 1:  # [B,T,1,H,W]
                sal = sal[:, :, 0]
            else:
                raise ValueError("5D saliency must have a singleton channel dimension.")
        if sal.dim() == 4:  # [B,T,H,W] or [B,1,H,W]
            sal = sal[:, -1]
        if sal.dim() != 3 or sal.shape[0] != B:
            raise ValueError("saliency_maps must contain a matching batch of density maps.")
        if not torch.isfinite(sal).all() or (sal < 0).any():
            raise ValueError("saliency_maps must be finite and nonnegative.")
        patch_density = F.interpolate(sal[:, None], size=(H, W), mode="area")[:, 0]
        flat = patch_density.reshape(B, -1)
        peak = flat.amax(dim=1, keepdim=True)
        informative = (peak > 1e-8) & ((peak - flat.amin(dim=1, keepdim=True)) > 1e-8)
        relative = flat / peak.clamp_min(1e-8)
        role = labels.view(B, -1)
        role[(relative <= self.non_salient_patch_threshold) & informative] = 0
        role[(relative >= self.salient_patch_threshold) & informative] = 1
        return labels, relative.reshape(-1)

    def _prototype_role_losses(self, q_vis, labels):
        """Supervise banks without masking forward assignments with GT labels.

        Balanced bank probability loss plus nearest correct-role clustering and
        a margin between nearest correct/incorrect-role prototypes. The usual
        reconstruction still acts on the actual, unrestricted assignment.
        """
        zero = q_vis.sum() * 0.0
        if not self.use_saliency_prototype_roles or not (labels >= 0).any():
            return {"loss_prototype_role": zero, "loss_prototype_role_cluster": zero,
                    "loss_prototype_role_separation": zero}
        with torch.autocast(device_type=q_vis.device.type, enabled=False):
            similarity = F.normalize(q_vis.float(), dim=-1) @ F.normalize(
                self.visual_concepts.float(), dim=-1).T
            logits = similarity / max(float(self.visual_assignment_temperature), 1e-8)
            # Summed probability in each bank, rather than a separate predictor.
            bank_logits = torch.stack([
                torch.logsumexp(logits[:, self.prototype_roles == role], dim=-1)
                for role in (0, 1)
            ], dim=-1)
            role_terms, cluster_terms, separation_terms = [], [], []
            for role in (0, 1):
                selected = labels == role
                if not selected.any():
                    continue
                scores = similarity[selected]
                correct = scores[:, self.prototype_roles == role].amax(dim=-1)
                incorrect = scores[:, self.prototype_roles != role].amax(dim=-1)
                role_terms.append(F.cross_entropy(bank_logits[selected], labels[selected]))
                cluster_terms.append((1 - correct).mean())
                separation_terms.append(F.relu(
                    self.prototype_role_margin + incorrect - correct).mean())
            return {
                "loss_prototype_role": torch.stack(role_terms).mean(),
                "loss_prototype_role_cluster": torch.stack(cluster_terms).mean(),
                "loss_prototype_role_separation": torch.stack(separation_terms).mean(),
            }

    def _reference_patch_rows(self, reference_patch_features, features):
        """Accept row-major [B*H*W,C_ref] or aligned [B,C_ref,1,H,W]."""
        if reference_patch_features is None:
            return None
        ref = reference_patch_features.detach().to(features.device, dtype=torch.float32)
        B, _, _, H, W = features.shape
        if ref.dim() == 5:
            if (ref.shape[0], ref.shape[2], ref.shape[3], ref.shape[4]) != (B, 1, H, W):
                raise ValueError("Reference features must align with B, last time step, H, W.")
            ref = ref.permute(0, 2, 3, 4, 1).reshape(B * H * W, ref.shape[1])
        if ref.dim() != 2 or ref.shape[0] != B * H * W or ref.shape[1] < 1:
            raise ValueError("reference_patch_features must be [B*H*W,C_ref] in patch order.")
        if not torch.isfinite(ref).all():
            raise ValueError("Cached reference features contain nonfinite values.")
        return ref

    def _visual_preservation_loss(self, q_vis, reference_patch_features):
        """Sampled mean squared cosine discrepancy, excluding self-pairs.

        Samples globally across the batch (including different videos when
        present), independently of attention labels or prototype assignments.
        Reference channel dimension need not equal concept_dim. Zero reference
        vectors are excluded; no teacher/reference gradients are created.
        """
        zero = q_vis.sum() * 0.0
        if reference_patch_features is None:
            return zero, 0
        ref = reference_patch_features.detach()
        indices = (ref.norm(dim=-1) > 1e-8).nonzero(as_tuple=False).flatten()
        if indices.numel() > self.preservation_max_patches:
            indices = indices[torch.randperm(indices.numel(), device=indices.device)[
                :self.preservation_max_patches]]
        n = indices.numel()
        if n < 2:
            return zero, 0
        with torch.autocast(device_type=q_vis.device.type, enabled=False):
            q = F.normalize(q_vis[indices].float(), dim=-1)
            f = F.normalize(ref[indices].float(), dim=-1)
            difference = (q @ q.T - f @ f.T).square()
            off_diag = ~torch.eye(n, device=q.device, dtype=torch.bool)
            return difference[off_diag].mean(), n * (n - 1)

    def _projection_versions(self):
        return tuple(parameter._version for parameter in self.parameters())

    @torch.no_grad()
    def begin_prototype_projection(self):
        """Start a single-process training-data sweep with the model in eval mode.

        The encoder/backbone must remain unchanged for the entire sweep. No
        prototype is mutated until finalize_prototype_projection is called.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before prototype projection.")
        K, D = self.visual_concepts.shape
        self._projection_state = {
            "versions": self._projection_versions(),
            "scores": torch.full((K,), -float("inf"), device=self.visual_concepts.device),
            "vectors": torch.zeros(K, D, device=self.visual_concepts.device),
            "sources": [None] * K,
        }

    @torch.no_grad()
    def update_prototype_projection(self, concept_out, saliency_maps, sample_ids,
                                    frame_indices=None, chunk_size=2048):
        """Collect nearest live encoded training patches within each GT role.

        sample_ids must uniquely identify video windows; frame_indices, when
        supplied, identify the actual target frames. Stores grid/patch provenance.
        Projection uses live encoder outputs, not cached C_ref-dimensional rows.
        """
        state = self._projection_state
        if state is None:
            raise RuntimeError("Call begin_prototype_projection first.")
        if self.training or state["versions"] != self._projection_versions():
            raise RuntimeError("Projection requires eval mode and unchanged model parameters.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        q = F.normalize(concept_out["visual_patch_embeddings"].detach().float(), dim=-1)
        metadata = concept_out["visual_metadata"]
        shape = metadata["feature_shape"]
        B, H, W = shape["B"], shape["H"], shape["W"]
        if len(sample_ids) != B or (frame_indices is not None and len(frame_indices) != B):
            raise ValueError("Provide one sample ID and optional frame index per batch item.")
        if q.shape != (B * H * W, self.concept_dim) or not torch.isfinite(q).all():
            raise ValueError("Invalid projection patch embeddings.")
        labels, targets = self._patch_attention_roles(saliency_maps, metadata, q.device)
        if self.use_saliency_prototype_roles and saliency_maps is None:
            raise ValueError("Role-restricted projection requires GT density maps.")
        prototypes = F.normalize(self.visual_concepts.detach().float(), dim=-1)
        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            scores = q[start:end] @ prototypes.T
            if self.use_saliency_prototype_roles:
                eligible = labels[start:end, None] == self.prototype_roles[None]
                scores = scores.masked_fill(~eligible, -float("inf"))
            scores = scores.masked_fill(q[start:end].norm(dim=-1)[:, None] < 1e-8,
                                        -float("inf"))
            values, offsets = scores.max(dim=0)
            improved = (values > state["scores"]).nonzero(as_tuple=False).flatten()
            for k in improved.tolist():
                row = start + int(offsets[k])
                batch = row // (H * W)
                patch = row % (H * W)
                frame = None if frame_indices is None else frame_indices[batch]
                if torch.is_tensor(frame):
                    frame = frame.item()
                identifier = sample_ids[batch]
                if torch.is_tensor(identifier):
                    identifier = identifier.item()
                state["scores"][k] = values[k]
                state["vectors"][k] = q[row]
                state["sources"][k] = {
                    "sample_id": str(identifier), "target_frame": frame,
                    "window_target_offset": int(metadata["assignment_time_index"]),
                    "patch_index": patch, "grid_shape": [H, W],
                    "grid_row": patch // W, "grid_column": patch % W,
                    "prototype_role": int(self.prototype_roles[k]),
                    "relative_patch_density": float(targets[row]),
                    "cosine_before_projection": float(values[k]),
                }

    @torch.no_grad()
    def finalize_prototype_projection(self, optimizer=None):
        """Replace matched rows in place; keep prototypes with no eligible patch.

        Optionally clear row-wise optimizer momentum for replaced prototypes.
        Return a serializable report to save with the projected checkpoint.
        Further encoder/backbone/prototype updates can invalidate exact grounding;
        re-project at the end of training if such updates continue.
        """
        state = self._projection_state
        if state is None:
            raise RuntimeError("Call begin_prototype_projection first.")
        if self.training or state["versions"] != self._projection_versions():
            raise RuntimeError("Model parameters changed during the projection sweep.")
        matched = torch.isfinite(state["scores"])
        self.visual_concepts[matched] = state["vectors"][matched].to(self.visual_concepts.dtype)
        if optimizer is not None:
            opt_state = optimizer.state.get(self.visual_concepts, {})
            for value in opt_state.values():
                if torch.is_tensor(value) and value.shape == self.visual_concepts.shape:
                    value[matched.to(value.device)] = 0
        report = {
            "projected_count": int(matched.sum()),
            "unmatched_indices": (~matched).nonzero(as_tuple=False).flatten().tolist(),
            "sources": state["sources"],
        }
        self._projection_state = None
        return report

    def cancel_prototype_projection(self):
        """Discard candidate search state without changing prototypes."""
        self._projection_state = None

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

        if sal.shape[0] != B:
            raise ValueError("Saliency batch size does not match feature batch.")

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
            if T == 1:
                # Concept features represent the actual final frame, not an
                # interpolated average of the full temporal density sequence.
                sal = sal[:, -1:]
                T_sal = 1
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
        Build visual-only concept assignments from the last-frame patch features.

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
                feature_shape (T=1), and window_T (full input temporal length)
        """
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
            q_vis,
            visual_repr,
            dim=-1,
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
        )

        # Optional diagnostic snapshots (no-op unless capture_diagnostic_trace).
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
        collect_gate_debug: bool = False,
        reference_patch_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, T, H, W] live backbone features.
            saliency_maps: optional GT saliency for training-only auxiliary losses.
            return_losses: whether to compute visual concept auxiliary losses.
            collect_gate_debug: ignored (kept for API compatibility).
            reference_patch_features: fixed cached reference rows [B*H*W,C_ref]
                or an aligned [B,C_ref,1,H,W] tensor. Required for preservation
                during training when require_reference_features=True.

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
        references = self._reference_patch_rows(reference_patch_features, features)
        q_vis = visual_out["visual_patch_embeddings"]
        # Labels supervise auxiliary losses only. They never mask forward
        # assignments, prototype activations, or the decoder's live inputs.
        labels, attention_targets = self._patch_attention_roles(
            saliency_maps, visual_out["visual_metadata"], q_vis.device)

        align_out: Dict[str, Any] = {
            "loss_visual_saliency_align": None,
            "visual_saliency_patch_logits": None,
            "visual_saliency_align_valid_frac": None,
        }

        losses: Dict[str, torch.Tensor] = {}
        if return_losses:
            if (self.training and self.visual_preservation_weight > 0
                    and self.require_reference_features and references is None):
                raise ValueError("Preservation training requires cached reference_patch_features. "
                                 "Use the updated model.py with reference_cache_dir, or explicitly "
                                 "disable preservation via visual_preservation_weight=0.")
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
            role_losses = self._prototype_role_losses(q_vis, labels)
            if self.visual_preservation_weight > 0:
                loss_preserve, pair_count = self._visual_preservation_loss(q_vis, references)
            else:
                loss_preserve, pair_count = q_vis.sum() * 0.0, 0
            w = self.loss_weights
            losses = {
                "loss_visual": loss_visual,
                "loss_visual_div": loss_visual_div,
                "loss_visual_assignment_entropy": entropy,
                "loss_visual_assignment_usage": loss_usage,
                "loss_visual_saliency_align": loss_visual_saliency_align,
                **role_losses,
                "loss_visual_preservation": loss_preserve,
                "visual_preservation_pair_count": torch.tensor(
                    pair_count, device=q_vis.device, dtype=torch.long),
                "loss_total_concept": (
                    w["visual"] * loss_visual
                    + w["visual_div"] * loss_visual_div
                    + self.visual_entropy_weight * entropy
                    + self.visual_usage_weight * loss_usage
                    + self.visual_saliency_align_weight * loss_visual_saliency_align
                    + self.prototype_role_weight * role_losses["loss_prototype_role"]
                    + self.prototype_role_cluster_weight * role_losses["loss_prototype_role_cluster"]
                    + self.prototype_role_separation_weight * role_losses["loss_prototype_role_separation"]
                    + self.visual_preservation_weight * loss_preserve
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
            "reference_patch_features": references,
            "prototype_roles": self.prototype_roles,
            "visual_assigned_prototype_roles": self.prototype_roles[
                visual_out["visual_concept_indices"]],
            "visual_patch_role_targets": labels,
            "visual_patch_relative_attention": attention_targets,
            "active_visual_prototypes": visual_out["active_visual_prototypes"],
            "active_visual_prototype_indices": visual_out[
                "active_visual_prototype_indices"
            ],
            "visual_validity_mask": visual_out["visual_validity_mask"],
            "visual_patch_embeddings": visual_out["visual_patch_embeddings"],
            "visual_concept_representation": visual_out["visual_concept_representation"],
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


# Backward-compatible alias for existing imports.
ConceptCreation = VisualConceptCreation
