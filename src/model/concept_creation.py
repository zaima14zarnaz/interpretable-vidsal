"""
Visual concept creation with per-prototype FIFO compactness (default N=8).

Every stage owns its own queues; no graph, RGB reload or extra backbone replay.
Queue entries are normalized learned patch embeddings, NOT RGB image crops.
Only the top-1 prototype owns an entry; prediction still uses original top-k.
Original reconstruction, diversity, entropy, usage and saliency losses remain.

loss_visual_queue: average cosine distance from each new patch to the entries
that precede it in its prototype's FIFO (including earlier patches in this
forward). loss_visual_queue_prototype applies the same targets to the matched
prototype vector, giving the prototype bank a direct compactness gradient.
Weights live in DEFAULT_LOSS_WEIGHTS as visual_queue / visual_queue_prototype.

Both losses use detached stored targets. Stable grouping, prefix sums and ring
updates reproduce sequential FIFO semantics without Python loops over patches
or prototypes and without constructing all patch-by-queue pairs. For normalized
vectors, mean cosine distance is EXACTLY 1 - dot(new, sum(queue)/count). Do not
normalize that queue mean: doing so changes the objective.

A training forward with return_losses=True and gradients enabled compares and
then enqueues. Eval, return_losses=False, no_grad and zero queue weights do not
mutate queues. Empty queues provide no target until the first insertion.
Queue vectors remain FP32, detached, on the concept module's device; capacity
8 and 512 prototypes of dimension 256 use about 4.7 MiB per stage with metadata.
Queues are rank-local for DDP and are not model parameters/checkpoint keys.

There is NO file I/O in forward. After each epoch, explicitly call:
    save_prototype_queues(model)
This atomically writes ONE snapshot containing every stage into:
    /home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/prot_queue/prototype_queues.pt
Snapshots contain current embeddings, FIFO pointers/counts, running sums,
confidence, and forward-step/batch/patch/grid/window locations (not video IDs
or RGB crops, since the current forward receives only feature maps).
Call load_prototype_queues(model) after loading model weights to resume queues;
loading old model checkpoints without this call safely starts empty queues.
Use clear_prototype_queue() after changing weights or assignment settings.

This is feature-space compactness, not a guarantee of human-visible similarity.
Frozen or trainable backbones are supported; only current embeddings receive
encoder/backbone gradients. Separate prototype attraction trains the bank.
Historical entries can become stale as the encoder or prototype meanings move;
capacity 8 limits count, not wall-clock age, especially for rare prototypes.

Learned parameter names/shapes and decoder outputs are unchanged. Legacy graph
constructor arguments are accepted only when no graph is configured. Old graph
loss keys remain explicit zeros for logging compatibility.
"""

from typing import Any, Dict, Optional, Tuple, Union
import math
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diagnostic_trace import record as _diag_record

DEFAULT_PROTOTYPE_QUEUE_DIR = "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/prot_queue"




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
        "visual_div": 1.0,
        "visual_queue": 1.0,
        "visual_queue_prototype": 0.1,
    }
    DROPOUT_P = 0.2

    def __init__(
        self,
        in_channels: int,
        concept_dim: int = 256,
        num_concepts: int = 32,
        hidden_dim: int = 512,
        top_k: int = 8,
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
        visual_assignment_mode: str = "topk_soft",
        visual_assignment_temperature: float = 0.07,
        visual_entropy_weight: float = 0.01,
        visual_usage_weight: float = 0.1,
        use_visual_saliency_alignment: bool = True,
        visual_saliency_align_weight: float = 0.05,
        use_target_centric: bool = True,
        last_transition_only: bool = True,
        visual_graph_db_path: Optional[str] = None,
        visual_graph_weight: float = 1.0,
        visual_graph_feature_weight: float = 0.1,
        visual_graph_settings: Optional[Dict[str, Any]] = None,
        visual_queue_size: int = 8,
        visual_queue_min_confidence: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self.top_k = top_k
        self.tau_alpha = tau_alpha
        self.tau_concept = tau_concept
        self.diversity_margin = diversity_margin
        self.concept_residual_weight = concept_residual_weight
        self.num_visual_concepts = (
            num_visual_concepts if num_visual_concepts is not None else num_concepts
        )
        self.visual_concept_residual_weight = visual_concept_residual_weight
        # Existing model.py passes "straight_through" by default. Treat legacy
        # names as aliases so the new forward works without editing model.py.
        # Record the effective mode truthfully; hard_eval remains explicit top-1.
        self.visual_assignment_mode_requested = visual_assignment_mode
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
        self.use_target_centric = use_target_centric
        self.last_transition_only = last_transition_only
        if isinstance(visual_queue_size, bool) or not isinstance(visual_queue_size, int) or visual_queue_size < 1:
            raise ValueError("visual_queue_size must be a positive integer.")
        if not math.isfinite(float(visual_queue_min_confidence)) or not 0 <= visual_queue_min_confidence <= 1:
            raise ValueError("visual_queue_min_confidence must lie in [0,1].")
        if visual_graph_db_path is not None or visual_graph_settings:
            raise ValueError("Graph supervision was replaced by per-prototype FIFO queues; remove graph configuration.")
        self.visual_queue_size = visual_queue_size
        self.visual_queue_min_confidence = float(visual_queue_min_confidence)
        self.visual_graph_diagnostics = {"enabled": False, "reason": "replaced_by_prototype_queue"}
        weights = dict(self.DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            weights.update(loss_weights)
        for name in ("visual_queue", "visual_queue_prototype"):
            value = weights[name]
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"loss_weights[{name!r}] must be finite and nonnegative.")
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
        self._grid_cache: Dict[
            Tuple[int, int, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._visual_meta_cache: Dict[
            Tuple[int, int, int, torch.device], Dict[str, torch.Tensor]
        ] = {}
        self.register_buffer(
            "inv_tau_concept",
            torch.tensor(1.0 / tau_concept),
            persistent=False,
        )

        self.clear_prototype_queue()

    _QUEUE_FLOAT_TENSORS = ("_prototype_queue", "_prototype_queue_sum", "_prototype_queue_confidence")
    _QUEUE_INT_TENSORS = ("_prototype_queue_count", "_prototype_queue_pointer", "_prototype_queue_locations")

    def _apply(self, fn, recurse=True):
        # Plain tensor attributes avoid DDP broadcasting the entire queue every
        # forward. Move them with the module, preserving FP32 compactness targets.
        result = super()._apply(fn, recurse=recurse)
        for name in self._QUEUE_FLOAT_TENSORS + self._QUEUE_INT_TENSORS:
            value = getattr(self, name, None)
            if value is not None:
                target_device = fn(torch.empty(0, device=value.device, dtype=value.dtype)).device
                setattr(self, name, value.to(device=target_device))
        return result

    def _init_concept_parameters(self) -> None:
        with torch.no_grad():
            self.visual_concepts.copy_(F.normalize(self.visual_concepts, dim=-1))

    def configure_visual_graph(self, db_path: Optional[str], **settings: Any) -> None:
        """Compatibility shim: graphs were replaced by per-prototype queues."""
        if db_path is not None or settings:
            raise ValueError("Remove graph configuration; per-prototype FIFO supervision is enabled automatically.")

    @torch.no_grad()
    def clear_prototype_queue(self) -> None:
        """Reset all memory for this stage. Does not read or write files."""
        m, n, d = self.num_visual_concepts, self.visual_queue_size, self.concept_dim
        device = self.visual_concepts.device
        self._prototype_queue = torch.zeros(m, n, d, device=device, dtype=torch.float32)
        self._prototype_queue_sum = torch.zeros(m, d, device=device, dtype=torch.float32)
        self._prototype_queue_confidence = torch.zeros(m, n, device=device, dtype=torch.float32)
        self._prototype_queue_count = torch.zeros(m, device=device, dtype=torch.long)
        self._prototype_queue_pointer = torch.zeros(m, device=device, dtype=torch.long)
        # Columns: training-forward index, batch index, row-major patch index,
        # grid height, grid width, temporal window length. -1 marks empty slots.
        self._prototype_queue_locations = torch.full((m, n, 6), -1, device=device, dtype=torch.long)
        self._prototype_queue_step = 0

    def _prototype_queue_losses(self, visual_out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Exact sequential FIFO loss, grouped/vectorized for the current batch.

        Old targets and preceding new patches are detached. For each incoming
        patch, use the last N PRECEDING entries of its winning prototype, then
        insert it. Stable grouping preserves original B,H,W order per prototype.
        Prototype attraction is a separate term so hard queue ownership does not
        leave the bank without a direct queue gradient. No loss gradient passes
        through the discrete top-1 choice; original assignment losses remain.
        """
        q = visual_out["visual_patch_embeddings"]
        zero = q.sum() * 0.0
        result = {"loss_visual_queue": zero, "loss_visual_queue_prototype": zero}
        if (not self.training or not torch.is_grad_enabled() or
                (self.loss_weights["visual_queue"] == 0
                 and self.loss_weights["visual_queue_prototype"] == 0)):
            return result
        with torch.autocast(device_type=q.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1)
            ids = visual_out["visual_concept_indices"].detach()
            confidence = visual_out["visual_assignment_probs"].detach().gather(1, ids[:, None]).squeeze(1).float()
            positions = torch.arange(q.shape[0], device=q.device)
            if self.visual_queue_min_confidence > 0:
                keep = confidence >= self.visual_queue_min_confidence
                q, ids, confidence, positions = q[keep], ids[keep], confidence[keep], positions[keep]
            order = torch.argsort(ids, stable=True)
            ids, q = ids[order], q[order]
            confidence, positions = confidence[order], positions[order]
            m, n = self.num_visual_concepts, self.visual_queue_size
            incoming_count = torch.bincount(ids, minlength=m)
            starts = incoming_count.cumsum(0) - incoming_count
            rank = torch.arange(q.shape[0], device=q.device) - starts[ids]
            prior_count = (self._prototype_queue_count[ids] + rank).clamp_max(n)

            # Sum the last min(rank,N) earlier patches in the same prototype's
            # current group. No pair tensor of shape [patches,N,dimension].
            history = q.detach()
            prefix = torch.cat((history.new_zeros(1, self.concept_dim), history.cumsum(0)), dim=0)
            offset = torch.arange(q.shape[0], device=q.device)
            recent_sum = prefix[offset] - prefix[offset - rank.clamp_max(n)]

            # Old queue contributions: drop as many oldest entries as prior
            # current-batch insertions would have evicted. Means are NOT unit
            # normalized; they preserve the exact mean cosine distance.
            slots = torch.arange(n, device=q.device)[None, :]
            chronological = (self._prototype_queue_pointer[:, None] -
                             self._prototype_queue_count[:, None] + slots) % n
            old = self._prototype_queue.gather(1, chronological[..., None].expand(-1, -1, self.concept_dim))
            old = old * (slots < self._prototype_queue_count[:, None])[..., None]
            old_prefix = torch.cat((old.new_zeros(m, 1, self.concept_dim), old.cumsum(1)), dim=1)
            dropped = (self._prototype_queue_count[ids] + rank - n).clamp_min(0)
            dropped = torch.minimum(dropped, self._prototype_queue_count[ids])
            old_sum = self._prototype_queue_sum[ids] - old_prefix[ids, dropped]
            targets = ((old_sum + recent_sum) / prior_count.clamp_min(1)[:, None]).detach()
            valid = (prior_count > 0).to(q.dtype)
            denominator = valid.sum().clamp_min(1.0)
            result["loss_visual_queue"] = ((1.0 - (q * targets).sum(-1)).clamp_min(0.0) * valid).sum() / denominator
            bank = F.normalize(self.visual_concepts.float(), dim=-1)
            result["loss_visual_queue_prototype"] = ((1.0 - (bank[ids] * targets).sum(-1)).clamp_min(0.0) * valid).sum() / denominator

            meta = visual_out["visual_metadata"]
            shape = meta["feature_shape"]
            locations = torch.stack((
                torch.full_like(positions, self._prototype_queue_step),
                meta["batch_idx"][positions], meta["patch_idx"][positions],
                torch.full_like(positions, int(shape["H"])),
                torch.full_like(positions, int(shape["W"])),
                torch.full_like(positions, int(meta["window_T"])),
            ), dim=1)
            self._enqueue_prototype_patches(ids, history, confidence, locations, rank, incoming_count)
        return result

    @torch.no_grad()
    def _enqueue_prototype_patches(self, ids, embeddings, confidence, locations, rank, incoming_count) -> None:
        """Bounded writes: retain only each group's last N incoming patches."""
        n, d = self.visual_queue_size, self.concept_dim
        keep = rank >= (incoming_count[ids] - n).clamp_min(0)
        owners, new = ids[keep], embeddings[keep]
        slot = (self._prototype_queue_pointer[owners] + rank[keep]) % n
        flat_slot = owners * n + slot
        flat_queue = self._prototype_queue.view(-1, d)
        # At most N unique slots per prototype are overwritten. Update sum by
        # subtracting the evicted vectors and adding replacements.
        evicted = flat_queue[flat_slot]
        self._prototype_queue_sum.index_add_(0, owners, new - evicted)
        flat_queue.index_copy_(0, flat_slot, new)
        self._prototype_queue_confidence.view(-1).index_copy_(0, flat_slot, confidence[keep])
        self._prototype_queue_locations.view(-1, 6).index_copy_(0, flat_slot, locations[keep])
        self._prototype_queue_pointer.add_(incoming_count).remainder_(n)
        self._prototype_queue_count.add_(incoming_count).clamp_max_(n)
        self._prototype_queue_step += 1
        # Bound floating-point drift after many subtract/add updates. This only
        # reads the tiny in-memory queues, never RGB images or disk files.
        if self._prototype_queue_step % 128 == 0:
            self._prototype_queue_sum.copy_(self._prototype_queue.sum(dim=1))

    @torch.no_grad()
    def prototype_queue_state(self) -> Dict[str, Any]:
        """Detached CPU snapshot; call at epoch/checkpoint boundaries, not per step."""
        return {
            "format": "prototype_fifo_v1",
            "num_prototypes": self.num_visual_concepts,
            "capacity": self.visual_queue_size,
            "embedding_dim": self.concept_dim,
            "in_channels": self.in_channels,
            "step": self._prototype_queue_step,
            "features": self._prototype_queue.detach().cpu().clone(),
            "sum": self._prototype_queue_sum.detach().cpu().clone(),
            "count": self._prototype_queue_count.detach().cpu().clone(),
            "pointer": self._prototype_queue_pointer.detach().cpu().clone(),
            "confidence": self._prototype_queue_confidence.detach().cpu().clone(),
            "locations": self._prototype_queue_locations.detach().cpu().clone(),
            "location_columns": ["forward_step", "batch_idx", "patch_idx", "grid_h", "grid_w", "window_T"],
            "assignment": "top1",
            "representation": "normalized_learned_embedding",
        }

    @torch.no_grad()
    def load_prototype_queue_state(self, state: Dict[str, Any]) -> None:
        """Restore a matching snapshot explicitly, normally after loading weights."""
        m, n, d = self.num_visual_concepts, self.visual_queue_size, self.concept_dim
        if (state.get("format") != "prototype_fifo_v1" or state.get("num_prototypes") != m or
                state.get("capacity") != n or state.get("embedding_dim") != d or
                state.get("in_channels") != self.in_channels):
            raise ValueError("Queue snapshot format/dimensions do not match this concept module.")
        specs = {"features": ((m, n, d), "_prototype_queue"),
                 "sum": ((m, d), "_prototype_queue_sum"),
                 "confidence": ((m, n), "_prototype_queue_confidence"),
                 "count": ((m,), "_prototype_queue_count"),
                 "pointer": ((m,), "_prototype_queue_pointer"),
                 "locations": ((m, n, 6), "_prototype_queue_locations")}
        for key, (shape, _) in specs.items():
            value = state.get(key)
            if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid queue snapshot tensor: {key}.")
        for key in ("count", "pointer", "locations"):
            if state[key].dtype != torch.long:
                raise ValueError(f"Queue snapshot {key} must be int64.")
        if ((state["count"] < 0) | (state["count"] > n)).any() or ((state["pointer"] < 0) | (state["pointer"] >= n)).any():
            raise ValueError("Invalid queue counts/pointers.")
        if not torch.allclose(state["sum"].float(), state["features"].float().sum(1), rtol=1e-4, atol=1e-4):
            raise ValueError("Queue sums do not match stored patches.")
        step = state.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Invalid queue step.")
        for key, (_, name) in specs.items():
            target = getattr(self, name)
            target.copy_(state[key].to(device=target.device, dtype=target.dtype))
        # Rebase after loading so snapshots don't perpetuate accumulation drift.
        self._prototype_queue_sum.copy_(self._prototype_queue.sum(1))
        self._prototype_queue_step = step

    def save_prototype_queue(self, directory: str = DEFAULT_PROTOTYPE_QUEUE_DIR,
                             filename: str = "prototype_queue.pt") -> str:
        """Explicit single-stage save; use save_prototype_queues(model) for all stages."""
        return _atomic_queue_save(self.prototype_queue_state(), directory, filename)

    @staticmethod
    def _build_grid(
        H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
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
        selected_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather selected prototypes independently at each patch.
        For topk_soft, selected_indices comes from the original similarity
        ranking: every selected slot is valid regardless of its small weight.
        Legacy hard_eval uses activation thresholding. Bank-size padding stays
        invalid in both modes.
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
        return active, validity, active_indices

    def _compute_visual_assignments(
        self, raw_similarity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Sparse soft assignments over the actual top-k cosine matches.
        Training and evaluation use the same weighted forward for topk_soft.
        Gradients flow through the selected weights, embeddings, and vectors;
        top-k index selection is discrete. "straight_through" and "soft" are
        compatibility aliases for topk_soft. Explicit hard_eval retains top-1.
        Full-bank probabilities are retained for the existing regularizers,
        while visual_activations is the sparse evidence used by prediction.
        """
        if self.visual_assignment_mode not in (
            "topk_soft",
            "straight_through",
            "soft",
            "hard_eval",
        ):
            raise ValueError(
                "visual_assignment_mode must be one of "
                "'topk_soft', 'straight_through', 'soft', or 'hard_eval', "
                f"got {self.visual_assignment_mode!r}"
            )
        temperature = max(float(self.visual_assignment_temperature), 1e-8)
        # Keep weights in FP32 under AMP: small selected weights must not become
        # zeros through FP16 underflow before the decoder gathers them.
        similarity = (
            raw_similarity.float()
            if raw_similarity.dtype in (torch.float16, torch.bfloat16)
            else raw_similarity
        )
        visual_logits = similarity / temperature
        visual_probs = F.softmax(visual_logits, dim=-1)
        visual_indices = visual_logits.argmax(dim=-1)
        topk_indices = None
        if self.visual_assignment_mode == "hard_eval":
            visual_activations = F.one_hot(
                visual_indices,
                num_classes=self.num_visual_concepts,
            ).to(dtype=visual_probs.dtype)
        else:
            k = min(self.top_k, int(visual_logits.shape[-1]))
            topk_logits, topk_indices = visual_logits.topk(k, dim=-1)
            weights = F.softmax(topk_logits, dim=-1)
            # Only repair numerical underflow at the dtype's smallest normal
            # positive value. This is not a support/validity threshold.
            weights = weights.clamp_min(torch.finfo(weights.dtype).tiny)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            visual_activations = torch.zeros_like(visual_logits).scatter(
                dim=-1,
                index=topk_indices,
                src=weights,
            )
        return {
            "visual_logits": visual_logits,
            "visual_probs": visual_probs,
            "visual_activations": visual_activations,
            "visual_indices": visual_indices,
            "topk_indices": topk_indices,
        }

    def _visual_concept_loss(
        self,
        visual_patch_embeddings: torch.Tensor,
        visual_activations: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable reconstruction loss for visual concept assignment."""
        c_vis = F.normalize(self.visual_concepts, dim=-1)
        recon = visual_activations @ c_vis
        return 1.0 - F.cosine_similarity(recon, visual_patch_embeddings, dim=-1).mean()

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
            sal_resized = (
                F.interpolate(
                    sal_flat,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                .reshape(B, T_sal, H, W)
                .clamp(0.0, 1.0)
                .to(device=device, dtype=dtype)
            )
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
            selected_indices=assignment_out["topk_indices"],
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
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, T, H, W] backbone features (frozen or trainable).
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
            raise ValueError(f"Need T>=1 for visual concepts, got T={features.size(2)}")
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
            queue_losses = self._prototype_queue_losses(visual_out)
            graph_losses = {
                "loss_visual_graph_assignment": loss_visual * 0.0,
                "loss_visual_graph_feature": loss_visual * 0.0,
            }
            w = self.loss_weights
            losses = {
                "loss_visual": loss_visual,
                "loss_visual_div": loss_visual_div,
                "loss_visual_assignment_entropy": entropy,
                "loss_visual_assignment_usage": loss_usage,
                "loss_visual_saliency_align": loss_visual_saliency_align,
                **graph_losses,
                **queue_losses,
                "loss_total_concept": (
                    w["visual"] * loss_visual
                    + w["visual_div"] * loss_visual_div
                    + self.visual_entropy_weight * entropy
                    + self.visual_usage_weight * loss_usage
                    + self.visual_saliency_align_weight * loss_visual_saliency_align
                    + w["visual_queue"] * queue_losses["loss_visual_queue"]
                    + w["visual_queue_prototype"] * queue_losses["loss_visual_queue_prototype"]
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
            "visual_graph_diagnostics": dict(self.visual_graph_diagnostics),
            "visual_queue_diagnostics": {
                "capacity": self.visual_queue_size,
                "training_forwards": self._prototype_queue_step,
                "assignment": "top1",
                "representation": "normalized_learned_embedding",
            },
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
            "visual_pairwise_cos_max": (
                off_diag.max().item() if off_diag.numel() > 0 else 0.0
            ),
            "visual_pairwise_cos_mean": (
                off_diag.mean().item() if off_diag.numel() > 0 else 0.0
            ),
        }


# Backward-compatible alias for existing imports.
ConceptCreation = VisualConceptCreation


def _atomic_queue_save(payload: Dict[str, Any], directory: str, filename: str) -> str:
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Queue snapshot filename must be a simple .pt filename.")
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".queue-", suffix=".pt", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(path)


def _queue_modules(model):
    # Unwrap standard parallel wrappers so save/resume names are stable.
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    return {name: module for name, module in model.named_modules()
            if isinstance(module, VisualConceptCreation)}


def save_prototype_queues(model, directory: str = DEFAULT_PROTOTYPE_QUEUE_DIR,
                          filename: str = "prototype_queues.pt") -> Optional[str]:
    """One explicit snapshot for ALL stage queues; no training-forward file I/O.

    Under DDP, rank 0 saves its local queue; there is no inter-rank gather or
    forward-time queue broadcast. Other ranks return None without writing.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    modules = _queue_modules(model)
    if not modules:
        raise ValueError("No VisualConceptCreation modules found in model.")
    states = {name: module.prototype_queue_state() for name, module in modules.items()}
    return _atomic_queue_save({"format": "prototype_fifo_model_v1", "stages": states}, directory, filename)


def load_prototype_queues(model, directory: str = DEFAULT_PROTOTYPE_QUEUE_DIR,
                          filename: str = "prototype_queues.pt") -> None:
    """Explicit resume after model weights load. Never called from forward.

    All DDP ranks may load the rank-0 snapshot and then maintain local queues.
    Missing files fail explicitly; omit this call when starting empty queues.
    """
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Queue snapshot filename must be a simple .pt filename.")
    payload = torch.load(Path(directory).expanduser() / filename, map_location="cpu", weights_only=True)
    modules = _queue_modules(model)
    if payload.get("format") != "prototype_fifo_model_v1" or set(payload.get("stages", {})) != set(modules):
        raise ValueError("Queue snapshot stage names do not match this model.")
    for name, module in modules.items():
        module.load_prototype_queue_state(payload["stages"][name])
