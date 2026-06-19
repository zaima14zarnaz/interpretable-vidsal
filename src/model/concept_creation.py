"""
Trajectory-based concept creation for explainable video saliency.

Builds transition trajectories between adjacent temporal patch tokens,
encodes them into concept space, and mixes transition / persistence concepts
via a learned gate. Optional saliency maps provide weak confidence-weighted
pseudo-label regularization for the gate (not ground-truth supervision).
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptCreation(nn.Module):
    """
    Encode patch-to-patch trajectories and map them to concept banks.

    Expected feature input: [B, C, T, H, W] (e.g. from VideoSwinTransformer).
    Target-centric trajectories (default): each target patch b@t+1 gets top_k
    incoming sources a@t. Trajectory vector encodes patch-pair change and interaction
    (not raw patch identity):
      [z_b-z_a, z_a*z_b, p_b-p_a, alpha] -> q -> concept mixture + residual.

    A separate visual concept branch assigns patch-level appearance concepts from
    individual normalized patch features z_i^t (independent of trajectories).

    By default only the last adjacent transition (T-2 -> T-1) is built
    (``last_transition_only=True``), matching SaliencyPrediction's filter.

    Metadata (per trajectory, length M) is consumed by SaliencyPrediction:
      batch_idx, time_idx (source t), source_idx, target_idx,
      source_coords, target_coords, alpha, affinity_logit, feature_shape,
      last_transition_only, num_time_transitions_built.
    """

    DEFAULT_LOSS_WEIGHTS = {
        "recon": 1.0,
        "align": 0.1,
        "sparse": 0.01,
        "div": 0.1,
        "gate": 0.1,
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
        self.eps_s = eps_s
        self.eps_p = eps_p
        self.eps_alpha = eps_alpha
        self.eps_v = eps_v
        self.eps_sal = eps_sal
        self.gate_temp_s = gate_temp_s
        self.gate_temp_v = gate_temp_v
        self.gate_temp_p = gate_temp_p
        self.gate_temp_sal = gate_temp_sal
        self.gate_min_conf = gate_min_conf
        self._last_gate_debug = {}
        self.max_source_patches = max_source_patches
        self.concept_residual_weight = concept_residual_weight
        self.num_visual_concepts = (
            num_visual_concepts if num_visual_concepts is not None else num_concepts
        )
        self.visual_concept_residual_weight = visual_concept_residual_weight
        self.use_target_centric = use_target_centric
        self.last_transition_only = last_transition_only

        weights = dict(self.DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            weights.update(loss_weights)
        self.loss_weights = weights

        # 2 * C (delta, product) + 2 (dx, dy) + 1 (alpha); raw z_a/z_b omitted
        self.trajectory_input_dim = 2 * in_channels + 2 + 1

        self.trajectory_encoder = nn.Sequential(
            nn.Linear(self.trajectory_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.DROPOUT_P),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, concept_dim),
            nn.LayerNorm(concept_dim),
        )
        self.gate = nn.Linear(concept_dim, 2)
        self.transition_concepts = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )
        self.persistence_concepts = nn.Parameter(
            torch.randn(num_concepts, concept_dim)
        )
        self.reconstruction_head = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.DROPOUT_P),
            nn.Linear(hidden_dim, concept_dim),
        )

        # Patch-level appearance concepts (visual-only branch).
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

        self._init_concept_parameters()
        self._grid_cache: Dict[Tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}
        self._source_idx_cache: Dict[Tuple[int, torch.device, Optional[int]], torch.Tensor] = {}
        self._time_indices_cache: Dict[int, list] = {}
        self._target_idx_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self.register_buffer(
            "inv_tau_concept",
            torch.tensor(1.0 / tau_concept),
            persistent=False,
        )

    def _init_concept_parameters(self) -> None:
        with torch.no_grad():
            self.transition_concepts.copy_(
                F.normalize(self.transition_concepts, dim=-1)
            )
            self.persistence_concepts.copy_(
                F.normalize(self.persistence_concepts, dim=-1)
            )
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
        grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
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

    def _sample_source_indices(self, N: int, device: torch.device) -> torch.Tensor:
        key = (N, device, self.max_source_patches)
        cached = self._source_idx_cache.get(key)
        if cached is not None:
            return cached

        if self.max_source_patches is None or self.max_source_patches >= N:
            cached = torch.arange(N, device=device, dtype=torch.long)
        else:
            n = int(self.max_source_patches)
            cached = torch.linspace(0, N - 1, steps=n, device=device).long()
        self._source_idx_cache[key] = cached
        return cached

    def _transition_time_indices(self, T: int) -> list[int]:
        """Adjacent transition source times t for t -> t+1."""
        cached = self._time_indices_cache.get(T)
        if cached is not None:
            return cached

        if T < 2:
            raise ValueError(
                f"Need T>=2 for adjacent trajectories, got T={T}"
            )
        if self.last_transition_only:
            t_last = T - 2
            if t_last < 0 or t_last >= T - 1:
                raise ValueError(
                    f"last_transition_only requires valid source time T-2={t_last} "
                    f"for T={T}"
                )
            cached = [t_last]
        else:
            cached = list(range(T - 1))
        self._time_indices_cache[T] = cached
        return cached

    def _target_idx_base(self, N: int, device: torch.device) -> torch.Tensor:
        key = (N, device)
        cached = self._target_idx_cache.get(key)
        if cached is None:
            cached = torch.arange(N, device=device, dtype=torch.long)
            self._target_idx_cache[key] = cached
        return cached

    @staticmethod
    def _package_metadata(
        meta_chunks: Dict[str, list],
        B: int,
        C: int,
        T: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """
        Concatenate per-time metadata chunks and attach feature_shape.

        All trajectory index tensors are int64 on ``device``; alpha and coords
        match ``dtype`` (same as input features). Tensors are not detached.
        """
        metadata: Dict[str, Any] = {
            "batch_idx": torch.cat(meta_chunks["batch_idx"], dim=0).long(),
            "time_idx": torch.cat(meta_chunks["time_idx"], dim=0).long(),
            "source_idx": torch.cat(meta_chunks["source_idx"], dim=0).long(),
            "target_idx": torch.cat(meta_chunks["target_idx"], dim=0).long(),
            "source_coords": torch.cat(meta_chunks["source_coords"], dim=0).to(
                device=device, dtype=dtype
            ),
            "target_coords": torch.cat(meta_chunks["target_coords"], dim=0).to(
                device=device, dtype=dtype
            ),
            "alpha": torch.cat(meta_chunks["alpha"], dim=0).to(device=device, dtype=dtype),
            "affinity_logit": torch.cat(meta_chunks["affinity_logit"], dim=0).to(
                device=device, dtype=dtype
            ),
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
        }
        return metadata

    def _build_trajectories(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.use_target_centric:
            return self._build_trajectories_target_centric(features)
        return self._build_trajectories_source_centric(features)

    def _build_trajectories_target_centric(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Target-centric trajectories: every target patch b@t+1 gets top_k sources a@t.

        Flatten order per transition: [B, N, k].
        """
        B, C, T, H, W = features.shape
        device = features.device
        dtype = features.dtype
        N = H * W
        k = min(self.top_k, N)

        z = self._flatten_features(features)  # [B, T, N, C]
        grid = self._make_grid(H, W, device, dtype)  # [N, 2]

        traj_chunks = []
        meta_chunks: Dict[str, list] = {
            "batch_idx": [],
            "time_idx": [],
            "source_idx": [],
            "target_idx": [],
            "source_coords": [],
            "target_coords": [],
            "alpha": [],
            "affinity_logit": [],
        }

        target_idx_base = self._target_idx_base(N, device)

        time_indices = self._transition_time_indices(T)
        for t in time_indices:
            z_src = z[:, t, :, :]  # [B, N, C]
            z_tgt = z[:, t + 1, :, :]  # [B, N, C]

            # Similarity: each target b to all sources a [B, N, N]
            sim = torch.matmul(z_tgt, z_src.transpose(-1, -2))
            sim_top, source_idx = torch.topk(sim, k=k, dim=-1)  # [B, N, k]

            alpha_full = F.softmax(sim / self.tau_alpha, dim=-1)
            alpha_top = torch.gather(alpha_full, dim=-1, index=source_idx)  # [B, N, k]

            z_b = z_tgt.unsqueeze(2).expand(B, N, k, C)
            gather_src = source_idx.unsqueeze(-1).expand(-1, -1, -1, C)
            z_a = torch.gather(
                z_src.unsqueeze(2).expand(B, N, N, C),
                dim=2,
                index=gather_src,
            )

            p_b = grid.view(1, N, 1, 2).expand(B, N, k, 2)
            p_a = grid[source_idx]
            disp = p_b - p_a

            traj = torch.cat(
                [
                    z_b - z_a,
                    z_a * z_b,
                    disp,
                    alpha_top.unsqueeze(-1),
                ],
                dim=-1,
            )
            traj_chunks.append(traj.reshape(-1, self.trajectory_input_dim))

            m = B * N * k
            meta_chunks["batch_idx"].append(
                torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N * k)
            )
            meta_chunks["time_idx"].append(
                torch.full((m,), t, device=device, dtype=torch.long)
            )
            meta_chunks["source_idx"].append(source_idx.reshape(-1))
            meta_chunks["target_idx"].append(
                target_idx_base.view(1, N, 1).expand(B, N, k).reshape(-1)
            )
            meta_chunks["source_coords"].append(p_a.reshape(-1, 2))
            meta_chunks["target_coords"].append(p_b.reshape(-1, 2))
            meta_chunks["alpha"].append(alpha_top.reshape(-1))
            meta_chunks["affinity_logit"].append(sim_top.reshape(-1))

        trajectory_vectors = torch.cat(traj_chunks, dim=0)
        metadata = self._package_metadata(meta_chunks, B, C, T, H, W, device, dtype)
        metadata["trajectory_mode"] = "target_centric"
        metadata["top_k_sources_per_target"] = k
        metadata["last_transition_only"] = self.last_transition_only
        metadata["num_time_transitions_built"] = len(time_indices)
        return trajectory_vectors, metadata

    def _build_trajectories_source_centric(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Legacy source-centric top-k targets per subsampled source patch."""
        B, C, T, H, W = features.shape
        device = features.device
        dtype = features.dtype
        N = H * W

        z = self._flatten_features(features)
        grid = self._make_grid(H, W, device, dtype)
        source_idx = self._sample_source_indices(N, device)
        S = source_idx.numel()
        k = min(self.top_k, N)

        traj_chunks = []
        meta_chunks: Dict[str, list] = {
            "batch_idx": [],
            "time_idx": [],
            "source_idx": [],
            "target_idx": [],
            "source_coords": [],
            "target_coords": [],
            "alpha": [],
            "affinity_logit": [],
        }

        z_src_all = z[:, :, source_idx, :]
        p_a_base = grid[source_idx]

        time_indices = self._transition_time_indices(T)
        for t in time_indices:
            z_src = z_src_all[:, t, :, :]
            z_tgt = z[:, t + 1, :, :]

            sim = torch.matmul(z_src, z_tgt.transpose(-1, -2))
            alpha_full = F.softmax(sim / self.tau_alpha, dim=-1)
            alpha_top, target_idx = torch.topk(alpha_full, k=k, dim=-1)

            gather_idx = target_idx.unsqueeze(-1).expand(-1, -1, -1, C)
            z_b = torch.gather(
                z_tgt.unsqueeze(1).expand(B, S, N, C),
                dim=2,
                index=gather_idx,
            )
            z_a = z_src.unsqueeze(2).expand(B, S, k, C)
            disp = grid[target_idx] - p_a_base.view(1, S, 1, 2)

            traj = torch.cat(
                [z_b - z_a, z_a * z_b, disp, alpha_top.unsqueeze(-1)],
                dim=-1,
            )
            traj_chunks.append(traj.reshape(-1, self.trajectory_input_dim))

            m = B * S * k
            meta_chunks["batch_idx"].append(
                torch.arange(B, device=device, dtype=torch.long).repeat_interleave(S * k)
            )
            meta_chunks["time_idx"].append(
                torch.full((m,), t, device=device, dtype=torch.long)
            )
            meta_chunks["source_idx"].append(
                source_idx.view(1, S, 1).expand(B, S, k).reshape(-1)
            )
            meta_chunks["target_idx"].append(target_idx.reshape(-1))
            meta_chunks["source_coords"].append(
                p_a_base.view(1, S, 1, 2).expand(B, S, k, 2).reshape(-1, 2)
            )
            meta_chunks["target_coords"].append(grid[target_idx].reshape(-1, 2))
            meta_chunks["alpha"].append(alpha_top.reshape(-1))
            meta_chunks["affinity_logit"].append(
                torch.gather(sim, dim=-1, index=target_idx).reshape(-1)
            )

        trajectory_vectors = torch.cat(traj_chunks, dim=0)
        metadata = self._package_metadata(meta_chunks, B, C, T, H, W, device, dtype)
        metadata["trajectory_mode"] = "source_centric"
        metadata["top_k_sources_per_target"] = k
        metadata["last_transition_only"] = self.last_transition_only
        metadata["num_time_transitions_built"] = len(time_indices)
        return trajectory_vectors, metadata

    def _normalized_concepts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        c_tr = F.normalize(self.transition_concepts, dim=-1)
        c_per = F.normalize(self.persistence_concepts, dim=-1)
        return c_tr, c_per

    def _alignment_loss(
        self,
        q: torch.Tensor,
        c_tr: torch.Tensor,
        c_per: torch.Tensor,
        a_tr: torch.Tensor,
        a_per: torch.Tensor,
        gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        # Weighted squared distance to concept mixtures [M]
        target_tr = a_tr @ c_tr
        target_per = a_per @ c_per
        dist_tr = (q - target_tr).pow(2).sum(dim=-1)
        dist_per = (q - target_per).pow(2).sum(dim=-1)
        loss = gate_probs[:, 0] * dist_tr + gate_probs[:, 1] * dist_per
        return loss.mean()

    def _sparsity_loss(
        self,
        a_tr: torch.Tensor,
        a_per: torch.Tensor,
        gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        def entropy(p: torch.Tensor) -> torch.Tensor:
            return -(p * (p + 1e-8).log()).sum(dim=-1)

        ent_tr = entropy(a_tr)
        ent_per = entropy(a_per)
        return (gate_probs[:, 0] * ent_tr + gate_probs[:, 1] * ent_per).mean()

    def _diversity_loss(self) -> torch.Tensor:
        def bank_penalty(bank: torch.Tensor) -> torch.Tensor:
            bank_n = F.normalize(bank, dim=-1)
            cos = bank_n @ bank_n.T
            mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
            off_diag = cos[mask]
            return F.relu(off_diag - self.diversity_margin).pow(2).mean()

        return bank_penalty(self.transition_concepts) + bank_penalty(
            self.persistence_concepts
        )

    def _visual_diversity_loss(self) -> torch.Tensor:
        """Encourage visual concept bank prototypes to stay diverse."""
        bank_n = F.normalize(self.visual_concepts, dim=-1)
        cos = bank_n @ bank_n.T
        mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
        off_diag = cos[mask]
        return F.relu(off_diag - self.diversity_margin).pow(2).mean()

    def _visual_concept_loss(
        self,
        visual_patch_embeddings: torch.Tensor,
        visual_concept_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pull selected visual concept prototypes toward assigned patch embeddings.

        Patch embeddings are detached so gradients update concept bank entries via
        autograd (no manual in-place concept updates).
        """
        c_vis = F.normalize(self.visual_concepts, dim=-1)
        selected_concepts = c_vis[visual_concept_indices]
        return F.mse_loss(selected_concepts, visual_patch_embeddings.detach())

    def _build_visual_concepts_from_patches(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Build visual-only concept assignments from individual patch features.

        Visual concepts capture patch-level appearance (z_i^t), separate from
        trajectory concepts that encode temporal change/interaction/displacement.

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

        z = self._flatten_features(features)  # [B, T, N, C]
        patch_vectors = z.reshape(B * T * N, C)

        q_vis = self.visual_encoder(patch_vectors)
        q_vis = F.normalize(q_vis, dim=-1)

        c_vis = F.normalize(self.visual_concepts, dim=-1)
        visual_logits = (q_vis @ c_vis.T) * self.inv_tau_concept
        visual_indices = visual_logits.argmax(dim=-1)
        visual_activations = F.one_hot(
            visual_indices,
            num_classes=self.num_visual_concepts,
        ).to(dtype=q_vis.dtype)

        visual_repr = visual_activations @ c_vis
        if self.visual_concept_residual_weight > 0:
            visual_repr = visual_repr + self.visual_concept_residual_weight * q_vis
        visual_repr = F.normalize(visual_repr, dim=-1)

        grid = self._make_grid(H, W, device, dtype)
        patch_idx_base = torch.arange(N, device=device, dtype=torch.long)

        visual_metadata: Dict[str, Any] = {
            "batch_idx": torch.arange(B, device=device, dtype=torch.long).repeat_interleave(
                T * N
            ),
            "time_idx": torch.arange(T, device=device, dtype=torch.long)
            .repeat_interleave(N)
            .repeat(B),
            "patch_idx": patch_idx_base.repeat(B * T),
            "patch_coords": grid[patch_idx_base.repeat(B * T)],
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
        }

        return {
            "visual_patch_embeddings": q_vis,
            "visual_concept_logits": visual_logits,
            "visual_concept_indices": visual_indices,
            "visual_activations": visual_activations,
            "visual_concept_representation": visual_repr,
            "visual_metadata": visual_metadata,
        }

    def _reconstruction_loss(self, q: torch.Tensor, recon_q: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon_q, q)

    @staticmethod
    def _has_temporal_saliency_sequence(saliency_maps: torch.Tensor) -> bool:
        """True only when saliency_maps has at least two temporal frames."""
        if not isinstance(saliency_maps, torch.Tensor):
            return False
        if saliency_maps.dim() == 4 and saliency_maps.shape[1] >= 2:
            return True  # [B, T, H, W]
        if saliency_maps.dim() == 5:
            if saliency_maps.shape[1] == 1 and saliency_maps.shape[2] >= 2:
                return True  # [B, 1, T, H, W]
            if saliency_maps.shape[2] == 1 and saliency_maps.shape[1] >= 2:
                return True  # [B, T, 1, H, W]
        return False

    def _prepare_saliency_maps(
        self,
        saliency_maps: torch.Tensor,
        B: int,
        T: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Normalize and resize saliency to [B, T, H, W]."""
        sal = saliency_maps
        if sal.dim() == 3:
            sal = sal.unsqueeze(1)  # [B, 1, H, W]
        elif sal.dim() == 4:
            if sal.shape[1] == 1:
                pass  # [B, 1, H, W]
            elif sal.shape[1] >= 1:
                pass  # [B, T, H, W]
            else:
                raise ValueError(f"Unsupported 4D saliency shape {tuple(sal.shape)}")
        elif sal.dim() == 5:
            if sal.shape[1] == 1:
                sal = sal.squeeze(1)  # [B, T, H, W]
            elif sal.shape[2] == 1:
                sal = sal.squeeze(2)  # [B, T, H, W]
            else:
                raise ValueError(
                    "5D saliency_maps must be [B,1,T,H,W] or [B,T,1,H,W]"
                )
        else:
            raise ValueError(
                f"saliency_maps must be 3D–5D, got shape {tuple(saliency_maps.shape)}"
            )

        sal = sal.float()
        if sal.numel() > 0 and sal.max() > 2.0:
            sal = sal / 255.0

        if sal.shape[0] != B:
            raise ValueError(f"saliency batch {sal.shape[0]} != feature batch {B}")

        if sal.shape[2:] != (H, W) or sal.shape[1] != T:
            sal = F.interpolate(
                sal.unsqueeze(1),
                size=(T, H, W),
                mode="trilinear",
                align_corners=False,
            ).squeeze(1)
        return sal

    def _gate_regularization_loss(
        self,
        saliency_maps: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        gate_probs: torch.Tensor,
        feature_shape: Tuple[int, ...],
        collect_gate_debug: bool = False,
    ) -> torch.Tensor:
        """
        Soft weak regularization for transition/persistence gate.

        This is NOT ground-truth supervision. It is a confidence-weighted pseudo-label
        regularizer.

        Persistence means:
            the trajectory connects visually similar evidence and saliency is stable.

        Transition means:
            saliency changes strongly, or attention shifts to visually different evidence.

        Spatial displacement alone does not force transition, because an attended object
        can move while remaining the same visual entity. Ambiguous trajectories are
        down-weighted by confidence.
        """
        B, _, T, H, W = feature_shape
        device = gate_probs.device
        dtype = gate_probs.dtype
        N = H * W
        eps = 1e-8

        sal = self._prepare_saliency_maps(saliency_maps, B, T, H, W)
        sal_flat = sal.reshape(B, T, N)

        b_idx = metadata["batch_idx"].to(device=device, dtype=torch.long)
        t_idx = metadata["time_idx"].to(device=device, dtype=torch.long)
        src_idx = metadata["source_idx"].to(device=device, dtype=torch.long)
        tgt_idx = metadata["target_idx"].to(device=device, dtype=torch.long)

        s_a = sal_flat[b_idx, t_idx, src_idx].to(device=device, dtype=dtype)
        s_b = sal_flat[b_idx, t_idx + 1, tgt_idx].to(device=device, dtype=dtype)

        delta_abs = (s_b - s_a).abs()
        sal_active_value = torch.maximum(s_a, s_b).clamp(0.0, 1.0)

        disp = (
            metadata["target_coords"].to(device=device, dtype=dtype)
            - metadata["source_coords"].to(device=device, dtype=dtype)
        )
        dist = disp.norm(dim=-1)

        # Prefer affinity_logit because features are L2-normalized before similarity,
        # so this is cosine-like source-target visual similarity.
        # Fall back to alpha if affinity_logit is unavailable.
        if "affinity_logit" in metadata:
            visual_sim = metadata["affinity_logit"].to(device=device, dtype=dtype)
            visual_threshold = self.eps_v
        else:
            visual_sim = metadata["alpha"].to(device=device, dtype=dtype)
            visual_threshold = self.eps_alpha

        # Soft evidence terms.
        sal_active_score = torch.sigmoid(
            (sal_active_value - self.eps_sal) / max(self.gate_temp_sal, eps)
        )

        sal_stable_score = torch.sigmoid(
            (self.eps_s - delta_abs) / max(self.gate_temp_s, eps)
        )
        sal_change_score = torch.sigmoid(
            (delta_abs - self.eps_s) / max(self.gate_temp_s, eps)
        )

        same_visual_score = torch.sigmoid(
            (visual_sim - visual_threshold) / max(self.gate_temp_v, eps)
        )
        different_visual_score = torch.sigmoid(
            (visual_threshold - visual_sim) / max(self.gate_temp_v, eps)
        )

        large_shift_score = torch.sigmoid(
            (dist - self.eps_p) / max(self.gate_temp_p, eps)
        )

        # Persistence evidence:
        # active saliency + stable saliency + visually similar source/target.
        r_per = sal_active_score * sal_stable_score * same_visual_score

        # Transition evidence:
        # active saliency + saliency change OR large spatial shift to visually different evidence.
        shift_to_different_visual = large_shift_score * different_visual_score
        r_tr = sal_active_score * (
            1.0 - (1.0 - sal_change_score) * (1.0 - shift_to_different_visual)
        )

        # Convert evidence to soft pseudo-label distribution.
        evidence_sum = (r_tr + r_per).clamp(min=eps)
        y_tr = (r_tr / evidence_sum).detach()
        y_per = (r_per / evidence_sum).detach()

        # Confidence is high when one side has strong evidence.
        # This avoids forcing ambiguous/background trajectories into either class.
        confidence = torch.maximum(r_tr, r_per).detach()
        valid = confidence >= self.gate_min_conf

        # Hard assignment only for debugging statistics.
        hard_tr = valid & (y_tr > y_per)
        hard_per = valid & (y_per > y_tr)
        hard_ambiguous = ~valid | (y_tr == y_per)

        if collect_gate_debug:
            with torch.no_grad():
                total = max(float(gate_probs.shape[0]), 1.0)
                n_valid = max(float(valid.sum().item()), 1.0)

                def _masked_mean_std(
                    x: torch.Tensor, mask: torch.Tensor
                ) -> tuple[float, float]:
                    if mask.any():
                        vals = x[mask]
                        mean = float(vals.mean().item())
                        std = float(vals.std().item()) if vals.numel() > 1 else 0.0
                        return mean, std
                    return 0.0, 0.0

                visual_sim_tr_mean, visual_sim_tr_std = _masked_mean_std(
                    visual_sim, hard_tr
                )
                visual_sim_per_mean, visual_sim_per_std = _masked_mean_std(
                    visual_sim, hard_per
                )
                visual_sim_valid_mean, visual_sim_valid_std = _masked_mean_std(
                    visual_sim, valid
                )

                self._last_gate_debug = {
                    "gate_valid_frac_total": float(valid.float().mean().item()),
                    "gate_transition_frac_total": float(hard_tr.float().mean().item()),
                    "gate_persistence_frac_total": float(hard_per.float().mean().item()),
                    "gate_ambiguous_frac_total": float(
                        hard_ambiguous.float().mean().item()
                    ),
                    "gate_transition_frac_valid": float(hard_tr.sum().item() / n_valid),
                    "gate_persistence_frac_valid": float(hard_per.sum().item() / n_valid),
                    "gate_confidence_mean": float(confidence.mean().item()),
                    "gate_confidence_valid_mean": float(
                        confidence[valid].mean().item() if valid.any() else 0.0
                    ),
                    "gate_y_tr_mean": float(y_tr.mean().item()),
                    "gate_y_per_mean": float(y_per.mean().item()),
                    "gate_visual_sim_mean": float(visual_sim.mean().item()),
                    "gate_visual_sim_std": float(
                        visual_sim.std().item() if visual_sim.numel() > 1 else 0.0
                    ),
                    "gate_visual_sim_transition_mean": visual_sim_tr_mean,
                    "gate_visual_sim_transition_std": visual_sim_tr_std,
                    "gate_visual_sim_persistence_mean": visual_sim_per_mean,
                    "gate_visual_sim_persistence_std": visual_sim_per_std,
                    "gate_visual_sim_valid_mean": visual_sim_valid_mean,
                    "gate_visual_sim_valid_std": visual_sim_valid_std,
                    "gate_transition_count": int(hard_tr.sum().item()),
                    "gate_persistence_count": int(hard_per.sum().item()),
                    "gate_valid_count": int(valid.sum().item()),
                    "gate_total_count": int(gate_probs.shape[0]),
                    "gate_delta_abs_mean": float(delta_abs.mean().item()),
                    "gate_dist_mean": float(dist.mean().item()),
                }

        if not valid.any():
            return torch.zeros((), device=device, dtype=dtype)

        log_g_tr = torch.log(gate_probs[:, 0].clamp(min=eps))
        log_g_per = torch.log(gate_probs[:, 1].clamp(min=eps))

        soft_ce = -(y_tr * log_g_tr + y_per * log_g_per)

        weighted_loss = (confidence[valid] * soft_ce[valid]).sum()
        denom = confidence[valid].sum().clamp(min=eps)

        return weighted_loss / denom

    def _compute_losses(
        self,
        q: torch.Tensor,
        recon_q: torch.Tensor,
        a_tr: torch.Tensor,
        a_per: torch.Tensor,
        gate_probs: torch.Tensor,
        c_tr: torch.Tensor,
        c_per: torch.Tensor,
        saliency_maps: Optional[torch.Tensor],
        metadata: Dict[str, torch.Tensor],
        feature_shape: Tuple[int, ...],
        collect_gate_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        w = self.loss_weights
        loss_recon = self._reconstruction_loss(q, recon_q)
        loss_align = self._alignment_loss(q, c_tr, c_per, a_tr, a_per, gate_probs)
        loss_sparse = self._sparsity_loss(a_tr, a_per, gate_probs)
        loss_div = self._diversity_loss()
        if saliency_maps is not None and self._has_temporal_saliency_sequence(
            saliency_maps
        ):
            loss_gate = self._gate_regularization_loss(
                saliency_maps,
                metadata,
                gate_probs,
                feature_shape,
                collect_gate_debug=collect_gate_debug,
            )
        else:
            if collect_gate_debug:
                self._last_gate_debug = {
                    "gate_valid_frac_total": 0.0,
                    "gate_transition_frac_total": 0.0,
                    "gate_persistence_frac_total": 0.0,
                    "gate_ambiguous_frac_total": 1.0,
                }
            loss_gate = torch.zeros((), device=q.device, dtype=q.dtype)

        gate_debug = getattr(self, "_last_gate_debug", {}) if collect_gate_debug else {}

        loss_total_concept = (
            w["recon"] * loss_recon
            + w["align"] * loss_align
            + w["sparse"] * loss_sparse
            + w["div"] * loss_div
            + w["gate"] * loss_gate
        )

        return {
            "loss_recon": loss_recon,
            "loss_align": loss_align,
            "loss_sparse": loss_sparse,
            "loss_div": loss_div,
            "loss_gate": loss_gate,
            "loss_total_concept": loss_total_concept,
            "gate_debug": gate_debug,
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
            saliency_maps: optional [B,T,H,W] (or 5D variants) for weak gate
                pseudo-label regularization only (not ground-truth supervision).
            return_losses: whether to compute auxiliary concept losses.

        Returns:
            Dictionary with trajectory embeddings, concept mixture, activations, and losses.
        """
        if features.dim() != 5:
            raise ValueError(
                f"features must be [B,C,T,H,W], got shape {tuple(features.shape)}"
            )
        if features.size(2) < 2:
            raise ValueError(
                f"Need T>=2 for adjacent trajectories, got T={features.size(2)}"
            )

        trajectory_vectors, metadata = self._build_trajectories(features)

        q = self.trajectory_encoder(trajectory_vectors)
        q = F.normalize(q, dim=-1)

        gate_probs = F.softmax(self.gate(q), dim=-1)  # [:,0]=transition, [:,1]=persistence

        c_tr, c_per = self._normalized_concepts()
        logits_both = (q @ torch.cat([c_tr, c_per], dim=0).T) * self.inv_tau_concept
        logits_tr, logits_per = logits_both.split(
            [self.num_concepts, self.num_concepts], dim=1
        )
        a_tr = F.softmax(logits_tr, dim=-1)
        a_per = F.softmax(logits_per, dim=-1)

        concept_tr = a_tr @ c_tr
        concept_per = a_per @ c_per
        concept_repr = (
            gate_probs[:, 0:1] * concept_tr + gate_probs[:, 1:2] * concept_per
        )
        if self.concept_residual_weight > 0:
            concept_repr = concept_repr + self.concept_residual_weight * q
        concept_repr = F.normalize(concept_repr, dim=-1)

        # Patch-level visual concepts (appearance); independent of trajectory concepts.
        visual_out = self._build_visual_concepts_from_patches(features)

        losses: Dict[str, torch.Tensor] = {}
        if return_losses:
            recon_q = self.reconstruction_head(concept_repr)
            losses = self._compute_losses(
                q,
                recon_q,
                a_tr,
                a_per,
                gate_probs,
                c_tr,
                c_per,
                saliency_maps,
                metadata,
                tuple(features.shape),
                collect_gate_debug=collect_gate_debug,
            )

            loss_visual = self._visual_concept_loss(
                visual_out["visual_patch_embeddings"],
                visual_out["visual_concept_indices"],
            )
            loss_visual_div = self._visual_diversity_loss()
            w = self.loss_weights
            losses["loss_visual"] = loss_visual
            losses["loss_visual_div"] = loss_visual_div
            losses["loss_total_concept"] = (
                losses["loss_total_concept"]
                + w["visual"] * loss_visual
                + w["visual_div"] * loss_visual_div
            )

        return {
            "trajectory_vectors": trajectory_vectors,
            "trajectory_embeddings": q,
            "concept_representation": concept_repr,
            "transition_activations": a_tr,
            "persistence_activations": a_per,
            "gate_probs": gate_probs,
            "metadata": metadata,
            "losses": losses,
            "visual_patch_embeddings": visual_out["visual_patch_embeddings"],
            "visual_concept_representation": visual_out["visual_concept_representation"],
            "visual_activations": visual_out["visual_activations"],
            "visual_concept_logits": visual_out["visual_concept_logits"],
            "visual_concept_indices": visual_out["visual_concept_indices"],
            "visual_metadata": visual_out["visual_metadata"],
        }

    @torch.no_grad()
    def summarize_concepts(self, top_n: int = 5) -> Dict[str, Union[float, int]]:
        """Lightweight concept-bank statistics for debugging."""
        c_tr = F.normalize(self.transition_concepts, dim=-1)
        c_per = F.normalize(self.persistence_concepts, dim=-1)

        def bank_stats(bank: torch.Tensor, name: str) -> Dict[str, float]:
            cos = bank @ bank.T
            mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
            off_diag = cos[mask]
            return {
                f"{name}_norm_mean": bank.norm(dim=-1).mean().item(),
                f"{name}_pairwise_cos_max": off_diag.max().item()
                if off_diag.numel() > 0
                else 0.0,
                f"{name}_pairwise_cos_mean": off_diag.mean().item()
                if off_diag.numel() > 0
                else 0.0,
            }

        out = {
            "num_concepts": self.num_concepts,
            "num_visual_concepts": self.num_visual_concepts,
            "concept_dim": self.concept_dim,
            "top_n": top_n,
        }
        out.update(bank_stats(c_tr, "transition"))
        out.update(bank_stats(c_per, "persistence"))
        c_vis = F.normalize(self.visual_concepts, dim=-1)
        out.update(bank_stats(c_vis, "visual"))
        return out
