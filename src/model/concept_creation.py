"""
Visual concept creation with paired RGB and frozen-backbone feature queues.

For every stage patch, live full-bank logits produce soft assignment weights.
The appearance objective is:
    loss_visual_appearance = visual_rgb * RGB_MSE + visual_feature * FEATURE_COSINE
Its overall weight remains loss_weights['visual_appearance'] in
loss_total_concept, so the existing training loss aggregator needs no changes.
Defaults: visual_appearance=1.0, visual_rgb=1.0, visual_feature=1.0; these are
starting weights, not an empirically tuned balance. Set visual_feature=0 for
an exact RGB-only ablation (with unchanged RGB queue admissions/loss).

RGB descriptors preserve the original detached, unnormalized R x R area-pooled
cells (R=4; 48 values), aligned to the resized backbone canvas. Frozen teacher
features arrive as detached cached_backbone_features [B,C,1,H,W]. Flattening is
batch-major/row-major, matching live patch order. Each reference vector is L2
normalized; zero-norm vectors are excluded from feature queries and history.
The model never substitutes trainable embeddings for missing cached targets.

Both components compare to PRE-FORWARD history, then perform ONE paired ring
write under current top-1 ownership. Feature entries share RGB slots, confidence
filtering and FIFO eviction. Missing/zero cached vectors mark their slots invalid.
Legacy callers without cached features retain RGB-only behavior. Each bank has
capacity visual_queue_size (8 by default); old entries are overwritten in order.

RGB cost is exact mean MSE to queued RGB descriptors, including queue variance.
Feature cost is exact mean (1 - cosine) to valid frozen queue vectors:
    C_feature(i,j) = 1 - normalized_reference_i dot mean(normalized_history_j)
Do not L2-normalize that mean: doing so would change the pairwise-average cost.
RGB and feature costs are detached. Gradients reach the live visual encoder,
prototype bank and trainable backbone through their assignment logits only.
Feature history is masked separately when a slot has no valid cached vector.
All-empty history has zero loss and warms up on insertion; a single populated
prototype offers no assignment choice until another prototype has history.

No history updates occur in eval, no_grad, return_losses=False, absent RGB or
when the combined appearance objective is disabled. Existing reconstruction,
diversity, entropy, usage, saliency alignment and decoding remain unchanged.
There is no graph lookup, image read, extra backbone pass or file I/O in forward.
Costs use [patches,prototypes] matrices and running sums, never a full
[patches,prototypes,queue_size,feature_dim] pair tensor. Queue arithmetic stays
FP32 under AMP/half(), and histories remain rank-local outside model state_dict.

Explicit queue saves use prototype_rgb_feature_fifo_model_v3 / per-stage
prototype_rgb_feature_fifo_v3. RGB-only v2 snapshots load with feature history
empty; learned model checkpoint keys/shapes are unchanged. Clear queue history
when changing the frozen-reference weights or cached feature semantics.
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
        "visual_appearance": 1.0,  # Overall weight on the combined appearance objective.
        "visual_rgb": 1.0,
        "visual_feature": 1.0,
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
        visual_appearance_patch_size: int = 4,
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
        if (isinstance(visual_appearance_patch_size, bool) or
                not isinstance(visual_appearance_patch_size, int) or visual_appearance_patch_size < 1):
            raise ValueError("visual_appearance_patch_size must be a positive integer.")
        self.visual_appearance_patch_size = visual_appearance_patch_size
        self.visual_appearance_dim = 3 * visual_appearance_patch_size ** 2
        self.visual_queue_size = visual_queue_size
        self.visual_queue_min_confidence = float(visual_queue_min_confidence)
        self.visual_graph_diagnostics = {"enabled": False, "reason": "replaced_by_prototype_queue"}
        weights = dict(self.DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            removed = {"visual_queue", "visual_queue_prototype"}.intersection(loss_weights)
            if removed:
                raise ValueError("Embedding queue losses were removed; use loss_weights['visual_appearance'] instead.")
            weights.update(loss_weights)
        for name in ("visual_appearance", "visual_rgb", "visual_feature"):
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

    _QUEUE_FLOAT_TENSORS = ("_prototype_queue", "_prototype_queue_sum", "_prototype_queue_squared_norm_sum", "_prototype_queue_confidence",
                            "_prototype_feature_queue", "_prototype_feature_queue_sum")
    _QUEUE_INT_TENSORS = ("_prototype_queue_count", "_prototype_queue_pointer", "_prototype_queue_locations", "_prototype_feature_count")
    _QUEUE_BOOL_TENSORS = ("_prototype_feature_valid",)

    def _apply(self, fn, recurse=True):
        # Plain tensors keep queues rank-local under DDP. Device follows the
        # module, but descriptor/statistic precision stays FP32 under half()/AMP.
        result = super()._apply(fn, recurse=recurse)
        for name in self._QUEUE_FLOAT_TENSORS + self._QUEUE_INT_TENSORS + self._QUEUE_BOOL_TENSORS:
            value = getattr(self, name, None)
            if value is not None:
                device = fn(torch.empty(0, device=value.device, dtype=value.dtype)).device
                setattr(self, name, value.to(device=device))
        return result

    def _init_concept_parameters(self) -> None:
        with torch.no_grad():
            self.visual_concepts.copy_(F.normalize(self.visual_concepts, dim=-1))

    def configure_visual_graph(self, db_path: Optional[str], **settings: Any) -> None:
        """Compatibility shim; RGB FIFO supervision does not use graphs."""
        if db_path is not None or settings:
            raise ValueError("Remove graph configuration; RGB appearance FIFO supervision is enabled automatically.")

    def _appearance_coverage_zero(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Detached scalar coverage counters on the model device (no .item())."""
        device = self.visual_concepts.device if device is None else device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            "app_loss_patch_count_total": zero.clone(),
            "app_loss_patch_count": zero.clone(),
            "app_loss_rgb_patch_count": zero.clone(),
            "app_loss_feature_patch_count": zero.clone(),
        }

    def _appearance_status(self, active: bool, reason: str,
                           device: Optional[torch.device] = None) -> Dict[str, Any]:
        status = {"enabled": self.loss_weights["visual_appearance"] > 0,
                "active_this_forward": active, "reason": reason,
                "capacity": self.visual_queue_size,
                "training_forwards": self._prototype_queue_step,
                "assignment": "top1_insertion_soft_full_bank_loss",
                "representation": "area_pooled_rgb_and_frozen_backbone",
                "patch_size": self.visual_appearance_patch_size,
                "descriptor_dim": self.visual_appearance_dim,
                "rgb_weight": self.loss_weights["visual_rgb"],
                "feature_weight": self.loss_weights["visual_feature"],
                "feature_dim": self.in_channels,
                "feature_metric": "mean_cosine_distance_to_normalized_queue_entries",
                "feature_active_this_forward": False,
                "feature_reason": reason,
                "feature_zero_norm_policy": "exclude from feature queries/history"}
        status.update(self._appearance_coverage_zero(device))
        return status

    def _appearance_inactive_diagnostics(
        self,
        reason: str,
        *,
        num_patches: int = 0,
        device: Optional[torch.device] = None,
        feature_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reset coverage every forward; evaluated counts stay zero when inactive."""
        device = self.visual_concepts.device if device is None else device
        status = self._appearance_status(False, reason, device=device)
        zero = torch.zeros((), device=device, dtype=torch.float32)
        status.update(
            app_loss_patch_count_total=zero + float(num_patches),
            app_loss_patch_count=zero.clone(),
            app_loss_rgb_patch_count=zero.clone(),
            app_loss_feature_patch_count=zero.clone(),
        )
        if feature_reason is not None:
            status["feature_reason"] = feature_reason
        return status

    @torch.no_grad()
    def clear_prototype_queue(self) -> None:
        """Clear paired RGB/reference-feature history, without any file I/O."""
        m, n, d = self.num_visual_concepts, self.visual_queue_size, self.visual_appearance_dim
        device = self.visual_concepts.device
        self._prototype_queue = torch.zeros(m, n, d, device=device, dtype=torch.float32)
        self._prototype_queue_sum = torch.zeros(m, d, device=device, dtype=torch.float32)
        self._prototype_queue_squared_norm_sum = torch.zeros(m, device=device, dtype=torch.float32)
        self._prototype_queue_confidence = torch.zeros(m, n, device=device, dtype=torch.float32)
        self._prototype_queue_count = torch.zeros(m, device=device, dtype=torch.long)
        self._prototype_queue_pointer = torch.zeros(m, device=device, dtype=torch.long)
        self._prototype_queue_locations = torch.full((m, n, 6), -1, device=device, dtype=torch.long)
        # Same admissions/slots as RGB. Missing/zero reference vectors are marked
        # invalid, so legacy RGB-only batches never create fictitious targets.
        self._prototype_feature_queue = torch.zeros(m, n, self.in_channels, device=device, dtype=torch.float32)
        self._prototype_feature_queue_sum = torch.zeros(m, self.in_channels, device=device, dtype=torch.float32)
        self._prototype_feature_count = torch.zeros(m, device=device, dtype=torch.long)
        self._prototype_feature_valid = torch.zeros(m, n, device=device, dtype=torch.bool)
        self._prototype_queue_step = 0
        self.visual_appearance_diagnostics = self._appearance_status(False, "empty_queues")

    @torch.no_grad()
    def _rgb_appearance_descriptors(self, rgb: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Fixed RGB descriptor for every cell of the actual stage concept grid.

        Area-pool the spatial canvas to [grid_h*R,grid_w*R], then tile into R x R
        cells. For the normal divisible backbone canvas this is exactly a local
        area-pooled patch. For other sizes, adaptive-area bins define alignment.
        Flatten output in [batch,row,column,channel,subrow,subcolumn] order.
        No learned feature extraction, L2 normalization, image read or resize of
        neural features is involved. RGB values and resulting descriptors are
        fixed targets; the assignment probabilities supply the model gradient.
        """
        r = self.visual_appearance_patch_size
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            pooled = F.adaptive_avg_pool2d(rgb.detach().float(), (grid_h * r, grid_w * r))
            descriptors = pooled.reshape(rgb.shape[0], 3, grid_h, r, grid_w, r)
            return descriptors.permute(0, 2, 4, 1, 3, 5).reshape(-1, self.visual_appearance_dim).contiguous()

    @torch.no_grad()
    def _cached_feature_descriptors(self, cached: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed normalized [B*H*W,C] reference vectors in live patch order."""
        with torch.autocast(device_type=cached.device.type, enabled=False):
            vectors = cached.detach().float()[:, :, 0].permute(0, 2, 3, 1).reshape(-1, self.in_channels).contiguous()
            if not torch.isfinite(vectors).all():
                raise ValueError("Cached reference vectors must remain finite when converted to float32.")
            # Scale before the norm so finite large raw features cannot overflow
            # during squaring. This leaves the normalized vector unchanged.
            scale = vectors.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            scaled = vectors / scale
            norms = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
            valid = norms.squeeze(-1) > (1e-12 / scale.squeeze(-1))
            normalized = scaled / norms.clamp_min(1e-12)
            normalized = torch.where(valid[:, None], normalized, torch.zeros_like(normalized))
            return normalized, valid

    @staticmethod
    def _queue_assignment_weights(logits: torch.Tensor, populated: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft full-bank weights with a safe all-empty fallback; FP32 caller."""
        has_history = populated.any()
        fallback = (~has_history) & (torch.arange(logits.shape[1], device=logits.device) == 0)
        safe_mask = populated | fallback
        weights = F.softmax(logits.float().masked_fill(~safe_mask[None, :], float("-inf")), dim=-1)
        return weights, has_history.to(torch.float32)

    def _visual_appearance_losses(
        self,
        visual_out: Dict[str, Any],
        rgb_last_frame: Optional[torch.Tensor],
        cached_backbone_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Expected fixed RGB MSE + frozen-feature cosine distance to history.

        Both costs use PRE-FORWARD queue statistics and live assignment logits.
        Only assignments carry gradients. A single paired enqueue follows both
        losses, preserving the no-current-self-insertion comparison protocol.
        RGB and feature histories use separate validity masks when cached inputs
        are absent/zero or an old RGB-only snapshot has just been restored.
        """
        logits = visual_out["visual_concept_logits"]
        zero = logits.sum() * 0.0
        result = {"rgb": zero, "feature": zero, "combined": zero}
        w_rgb, w_feature = self.loss_weights["visual_rgb"], self.loss_weights["visual_feature"]
        num_patches = int(logits.shape[0])
        reason = None
        if not self.training:
            reason = "evaluation"
        elif not torch.is_grad_enabled():
            reason = "no_grad"
        elif self.loss_weights["visual_appearance"] == 0 or (w_rgb == 0 and w_feature == 0):
            reason = "zero_weight"
        elif rgb_last_frame is None:
            reason = "no_rgb"
        elif w_rgb == 0 and cached_backbone_features is None:
            reason = "no_cached_features"
        if reason is not None:
            self.visual_appearance_diagnostics = self._appearance_inactive_diagnostics(
                reason, num_patches=num_patches, device=logits.device,
            )
            return result
        shape = visual_out["visual_metadata"]["feature_shape"]
        descriptors = self._rgb_appearance_descriptors(rgb_last_frame, int(shape["H"]), int(shape["W"]))
        if descriptors.shape[0] != logits.shape[0]:
            raise ValueError("RGB descriptors and concept patches must have the same batch-major row-major ordering.")
        feature_descriptors = feature_valid = None
        feature_reason = "zero_weight" if w_feature == 0 else "no_cached_features"
        if w_feature > 0 and cached_backbone_features is not None:
            feature_descriptors, feature_valid = self._cached_feature_descriptors(cached_backbone_features)
            if feature_descriptors.shape[0] != logits.shape[0]:
                raise ValueError("Cached reference descriptors must match the live batch-major row-major patch order.")
            feature_reason = "pre_forward_queue_comparison"
        # Coverage uses PRE-FORWARD history only. Defaults cover RGB/feature-disabled paths.
        rgb_history = zero.detach()
        feature_history = zero.detach()
        with torch.autocast(device_type=logits.device.type, enabled=False):
            if w_rgb > 0:
                divisor = self._prototype_queue_count.clamp_min(1).float()
                means = self._prototype_queue_sum / divisor[:, None]
                mean_squared_norm = self._prototype_queue_squared_norm_sum / divisor
                rgb_costs = (descriptors.square().sum(-1, keepdim=True)
                             - 2.0 * descriptors @ means.T + mean_squared_norm[None, :])
                rgb_costs = (rgb_costs / self.visual_appearance_dim).clamp_min(0.0).detach()
                weights, rgb_history = self._queue_assignment_weights(logits, self._prototype_queue_count > 0)
                result["rgb"] = (weights * rgb_costs).sum(-1).mean() * rgb_history
            if feature_descriptors is not None:
                means = self._prototype_feature_queue_sum / self._prototype_feature_count.clamp_min(1).float()[:, None]
                # Do NOT normalize the mean: dot with the arithmetic mean of
                # unit vectors is the exact average pairwise cosine similarity.
                feature_costs = (1.0 - feature_descriptors @ means.T).clamp(0.0, 2.0).detach()
                feature_weights, feature_history = self._queue_assignment_weights(logits, self._prototype_feature_count > 0)
                query_valid = feature_valid.to(torch.float32)
                result["feature"] = ((feature_weights * feature_costs).sum(-1) * query_valid).sum() / query_valid.sum().clamp_min(1.0)
                result["feature"] = result["feature"] * feature_history
            result["combined"] = w_rgb * result["rgb"] + w_feature * result["feature"]
        # Query-patch coverage (not patch–prototype pairs). Detached; no .item().
        with torch.no_grad():
            cov_zero = torch.zeros((), device=logits.device, dtype=torch.float32)
            total = cov_zero + float(num_patches)
            rgb_count = total * rgb_history.detach() if w_rgb > 0 else cov_zero.clone()
            if feature_descriptors is not None:
                feature_count = feature_valid.to(torch.float32).sum() * feature_history.detach()
            else:
                feature_count = cov_zero.clone()
            # RGB is all-or-nothing across the query grid; union without double-count.
            evaluated = rgb_count + feature_count * (cov_zero + 1.0 - (rgb_history.detach() if w_rgb > 0 else cov_zero))
        # Costs/masks above are independent snapshots. Never enqueue before
        # either comparison, and never update the same ring twice per forward.
        self._enqueue_rgb_descriptors(visual_out, descriptors, feature_descriptors, feature_valid)
        self.visual_appearance_diagnostics = self._appearance_status(
            True, "pre_forward_queue_comparison", device=logits.device,
        )
        self.visual_appearance_diagnostics.update(
            feature_active_this_forward=feature_descriptors is not None,
            feature_reason=feature_reason,
            app_loss_patch_count_total=total,
            app_loss_patch_count=evaluated,
            app_loss_rgb_patch_count=rgb_count,
            app_loss_feature_patch_count=feature_count,
        )
        return result

    def _visual_appearance_loss(
        self,
        visual_out: Dict[str, Any],
        rgb_last_frame: Optional[torch.Tensor],
        cached_backbone_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compatibility wrapper returning the combined appearance objective."""
        return self._visual_appearance_losses(visual_out, rgb_last_frame, cached_backbone_features)["combined"]

    @torch.no_grad()
    def _enqueue_rgb_descriptors(
        self,
        visual_out: Dict[str, Any],
        descriptors: torch.Tensor,
        feature_descriptors: Optional[torch.Tensor] = None,
        feature_valid: Optional[torch.Tensor] = None,
    ) -> None:
        """Single bounded ring write: RGB and frozen features share every slot."""
        ids = visual_out["visual_concept_indices"].detach()
        confidence = visual_out["visual_assignment_probs"].detach().gather(1, ids[:, None]).squeeze(1).float()
        positions = torch.arange(ids.shape[0], device=ids.device)
        if self.visual_queue_min_confidence > 0:
            keep = confidence >= self.visual_queue_min_confidence
            ids, descriptors = ids[keep], descriptors[keep]
            confidence, positions = confidence[keep], positions[keep]
        order = torch.argsort(ids, stable=True)
        ids, descriptors = ids[order], descriptors[order]
        confidence, positions = confidence[order], positions[order]
        m, n, d = self.num_visual_concepts, self.visual_queue_size, self.visual_appearance_dim
        incoming = torch.bincount(ids, minlength=m)
        starts = incoming.cumsum(0) - incoming
        rank = torch.arange(ids.shape[0], device=ids.device) - starts[ids]
        keep = rank >= (incoming[ids] - n).clamp_min(0)
        owners, new = ids[keep], descriptors[keep]
        slots = (self._prototype_queue_pointer[owners] + rank[keep]) % n
        flat_slots = owners * n + slots
        flat_queue = self._prototype_queue.view(-1, d)
        old = flat_queue[flat_slots]
        self._prototype_queue_sum.index_add_(0, owners, new - old)
        self._prototype_queue_squared_norm_sum.index_add_(0, owners, new.square().sum(-1) - old.square().sum(-1))
        flat_queue.index_copy_(0, flat_slots, new)
        self._prototype_queue_confidence.view(-1).index_copy_(0, flat_slots, confidence[keep])
        selected_positions = positions[keep]
        if feature_descriptors is None:
            new_feature = self._prototype_feature_queue.new_zeros((owners.shape[0], self.in_channels))
            new_valid = torch.zeros(owners.shape[0], dtype=torch.bool, device=ids.device)
        else:
            if feature_valid is None:
                raise ValueError("Feature queue insertion requires an explicit validity mask.")
            new_valid = feature_valid[selected_positions]
            new_feature = torch.where(new_valid[:, None], feature_descriptors[selected_positions],
                                      torch.zeros_like(feature_descriptors[selected_positions]))
        flat_feature = self._prototype_feature_queue.view(-1, self.in_channels)
        flat_valid = self._prototype_feature_valid.view(-1)
        old_feature, old_valid = flat_feature[flat_slots], flat_valid[flat_slots]
        self._prototype_feature_queue_sum.index_add_(0, owners, new_feature - old_feature)
        self._prototype_feature_count.index_add_(0, owners, new_valid.long() - old_valid.long())
        flat_feature.index_copy_(0, flat_slots, new_feature)
        flat_valid.index_copy_(0, flat_slots, new_valid)
        meta, shape = visual_out["visual_metadata"], visual_out["visual_metadata"]["feature_shape"]
        locations = torch.stack((torch.full_like(positions, self._prototype_queue_step),
                                 meta["batch_idx"][positions], meta["patch_idx"][positions],
                                 torch.full_like(positions, int(shape["H"])),
                                 torch.full_like(positions, int(shape["W"])),
                                 torch.full_like(positions, int(meta["window_T"]))), dim=1)
        self._prototype_queue_locations.view(-1, 6).index_copy_(0, flat_slots, locations[keep])
        self._prototype_queue_pointer.add_(incoming).remainder_(n)
        self._prototype_queue_count.add_(incoming).clamp_max_(n)
        self._prototype_queue_step += 1
        if self._prototype_queue_step % 128 == 0:
            self._prototype_queue_sum.copy_(self._prototype_queue.sum(1))
            self._prototype_queue_squared_norm_sum.copy_(self._prototype_queue.square().sum((1, 2)))
            self._prototype_feature_queue_sum.copy_(self._prototype_feature_queue.sum(1))
            self._prototype_feature_count.copy_(self._prototype_feature_valid.sum(1))

    @torch.no_grad()
    def prototype_queue_state(self) -> Dict[str, Any]:
        """Explicit CPU snapshot of aligned RGB and frozen-feature queues."""
        m, n, r = self.num_visual_concepts, self.visual_queue_size, self.visual_appearance_patch_size
        return {"format": "prototype_rgb_feature_fifo_v3", "num_prototypes": m, "capacity": n,
                "patch_size": r, "descriptor_dim": self.visual_appearance_dim,
                "in_channels": self.in_channels, "step": self._prototype_queue_step,
                "rgb_patches": self._prototype_queue.detach().cpu().clone().reshape(m, n, 3, r, r),
                "sum": self._prototype_queue_sum.detach().cpu().clone(),
                "squared_norm_sum": self._prototype_queue_squared_norm_sum.detach().cpu().clone(),
                "count": self._prototype_queue_count.detach().cpu().clone(),
                "pointer": self._prototype_queue_pointer.detach().cpu().clone(),
                "confidence": self._prototype_queue_confidence.detach().cpu().clone(),
                "locations": self._prototype_queue_locations.detach().cpu().clone(),
                "location_columns": ["forward_step", "batch_idx", "patch_idx", "grid_h", "grid_w", "window_T"],
                "assignment": "top1", "representation": "area_pooled_rgb_and_frozen_backbone",
                "descriptor_order": ["channel", "subrow", "subcolumn"],
                "feature_dim": self.in_channels,
                "feature_metric": "mean_cosine_distance",
                "feature_vectors": self._prototype_feature_queue.detach().cpu().clone(),
                "feature_sum": self._prototype_feature_queue_sum.detach().cpu().clone(),
                "feature_count": self._prototype_feature_count.detach().cpu().clone(),
                "feature_valid": self._prototype_feature_valid.detach().cpu().clone()}

    @torch.no_grad()
    def _validated_queue_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate completely before mutating any queue; migrate RGB-only v2."""
        m, n, r, d = self.num_visual_concepts, self.visual_queue_size, self.visual_appearance_patch_size, self.visual_appearance_dim
        if not isinstance(state, dict):
            raise ValueError("Queue snapshot must be a dictionary.")
        legacy = state.get("format") == "prototype_rgb_fifo_v2"
        if (state.get("format") not in ("prototype_rgb_fifo_v2", "prototype_rgb_feature_fifo_v3")
                or state.get("num_prototypes") != m or state.get("capacity") != n
                or state.get("patch_size") != r or state.get("descriptor_dim") != d
                or state.get("in_channels") != self.in_channels):
            raise ValueError("Queue snapshot format/dimensions do not match. Learned-embedding queues cannot be reused.")
        specs = {"rgb_patches": (m, n, 3, r, r), "sum": (m, d),
                 "squared_norm_sum": (m,), "count": (m,), "pointer": (m,),
                 "confidence": (m, n), "locations": (m, n, 6)}
        for key, shape in specs.items():
            value = state.get(key)
            if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid RGB queue snapshot tensor: {key}.")
        for key in ("count", "pointer", "locations"):
            if state[key].dtype != torch.long:
                raise ValueError(f"Queue snapshot {key} must be int64.")
        count, pointer = state["count"].detach().cpu(), state["pointer"].detach().cpu()
        if ((count < 0) | (count > n)).any() or ((pointer < 0) | (pointer >= n)).any():
            raise ValueError("Invalid RGB queue counts/pointers.")
        if (pointer[count < n] != count[count < n]).any():
            raise ValueError("Partially filled queue pointers must equal their counts.")
        rgb = state["rgb_patches"].detach().cpu().float().reshape(m, n, d)
        if ((rgb < -1e-6) | (rgb > 1.0 + 1e-6)).any():
            raise ValueError("Saved RGB descriptors must be in [0,1].")
        if not torch.allclose(state["sum"].detach().cpu().float(), rgb.sum(1), rtol=1e-4, atol=1e-4):
            raise ValueError("RGB queue sum does not match patches.")
        if not torch.allclose(state["squared_norm_sum"].detach().cpu().float(), rgb.square().sum((1, 2)), rtol=1e-4, atol=1e-4):
            raise ValueError("RGB queue squared-norm sum does not match patches.")
        step = state.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Invalid queue step.")
        if legacy:
            feature = torch.zeros(m, n, self.in_channels, dtype=torch.float32)
            valid = torch.zeros(m, n, dtype=torch.bool)
        else:
            if state.get("feature_dim") != self.in_channels or state.get("feature_metric") != "mean_cosine_distance":
                raise ValueError("Frozen-feature queue descriptor semantics/dimensions do not match.")
            feature_specs = {"feature_vectors": (m, n, self.in_channels), "feature_sum": (m, self.in_channels),
                             "feature_count": (m,), "feature_valid": (m, n)}
            for key, shape in feature_specs.items():
                value = state.get(key)
                if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid frozen-feature queue snapshot tensor: {key}.")
            if state["feature_valid"].dtype != torch.bool or state["feature_count"].dtype != torch.long:
                raise ValueError("Feature validity/counts must have bool/int64 dtypes.")
            feature = state["feature_vectors"].detach().cpu().float()
            valid = state["feature_valid"].detach().cpu()
            occupied = (torch.arange(n)[None, :] < count[:, None]) | (count[:, None] == n)
            if (valid & ~occupied).any():
                raise ValueError("Feature validity cannot occupy an empty RGB queue slot.")
            if (feature[~valid] != 0).any():
                raise ValueError("Invalid feature queue slots must contain zero vectors.")
            norms = torch.linalg.vector_norm(feature[valid], dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4):
                raise ValueError("Valid feature queue vectors must be L2 normalized.")
            if not torch.equal(state["feature_count"].detach().cpu(), valid.sum(1)):
                raise ValueError("Feature queue counts do not match validity masks.")
            if not torch.allclose(state["feature_sum"].detach().cpu().float(), feature.sum(1), rtol=1e-4, atol=1e-4):
                raise ValueError("Feature queue sums do not match vectors.")
        return {"step": step, "legacy": legacy,
                "_prototype_queue": rgb, "_prototype_queue_sum": rgb.sum(1),
                "_prototype_queue_squared_norm_sum": rgb.square().sum((1, 2)),
                "_prototype_queue_count": count, "_prototype_queue_pointer": pointer,
                "_prototype_queue_confidence": state["confidence"].detach().cpu().float(),
                "_prototype_queue_locations": state["locations"].detach().cpu(),
                "_prototype_feature_queue": feature, "_prototype_feature_queue_sum": feature.sum(1),
                "_prototype_feature_count": valid.sum(1), "_prototype_feature_valid": valid}

    @torch.no_grad()
    def _restore_validated_queue_state(self, state: Dict[str, Any]) -> None:
        for name in self._QUEUE_FLOAT_TENSORS + self._QUEUE_INT_TENSORS + self._QUEUE_BOOL_TENSORS:
            target = getattr(self, name)
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))
        self._prototype_queue_step = state["step"]
        reason = "restored_rgb_only_feature_warmup" if state["legacy"] else "restored_rgb_feature_queue"
        self.visual_appearance_diagnostics = self._appearance_status(False, reason)

    @torch.no_grad()
    def load_prototype_queue_state(self, state: Dict[str, Any]) -> None:
        """Restore v3 paired queues or v2 RGB queues with empty feature history."""
        self._restore_validated_queue_state(self._validated_queue_state(state))

    def save_prototype_queue(self, directory: str = DEFAULT_PROTOTYPE_QUEUE_DIR,
                             filename: str = "prototype_queue.pt") -> str:
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
        rgb_last_frame: Optional[torch.Tensor] = None,
        cached_backbone_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, T, H, W] backbone features (frozen or trainable).
            saliency_maps: optional GT saliency for training-only auxiliary losses.
            return_losses: whether to compute visual concept auxiliary losses.
            collect_gate_debug: ignored (kept for API compatibility).
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
                L2 normalized for cosine-distance queue supervision. It remains
                separate from learned ``visual_patch_embeddings``. Missing cached
                features retain RGB-only behavior and feature history warms up
                when cached inputs become available.
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
                raise TypeError("cached_backbone_features must be a floating-point tensor.")
            ref = cached_backbone_features.detach()
            if ref.dim() != 5 or ref.shape[2] != 1:
                raise ValueError(
                    "cached_backbone_features must have shape [B,C,1,H,W], "
                    f"got {tuple(ref.shape)}"
                )
            live_last = features[:, :, -1:, :, :]
            if ref.shape[0] != live_last.shape[0]:
                raise ValueError(
                    "cached_backbone_features batch must match features: "
                    f"{ref.shape[0]} vs {live_last.shape[0]}"
                )
            if ref.shape[1] != live_last.shape[1]:
                raise ValueError(
                    "cached_backbone_features channels must match features: "
                    f"{ref.shape[1]} vs {live_last.shape[1]}"
                )
            if ref.shape[-2:] != live_last.shape[-2:]:
                raise ValueError(
                    "cached_backbone_features spatial grid must match "
                    f"features[:, :, -1:]: {tuple(ref.shape[-2:])} vs "
                    f"{tuple(live_last.shape[-2:])}"
                )
            if ref.device != features.device:
                raise ValueError(
                    "cached_backbone_features must be on the same device as features: "
                    f"{ref.device} vs {features.device}"
                )
            if not torch.isfinite(ref).all():
                raise ValueError("cached_backbone_features contains non-finite values.")
            # Fixed reference targets only; no graph or retained batch attribute.
            cached_backbone_features = ref
        else:
            cached_backbone_features = None
        if rgb_last_frame is not None:
            if not torch.is_tensor(rgb_last_frame):
                raise TypeError("rgb_last_frame must be a torch.Tensor or None.")
            if rgb_last_frame.dim() != 4 or rgb_last_frame.shape[1] != 3:
                raise ValueError(
                    "rgb_last_frame must have shape [B,3,H_rgb,W_rgb], "
                    f"got {tuple(rgb_last_frame.shape)}"
                )
            if rgb_last_frame.dtype != torch.float32:
                raise ValueError("rgb_last_frame must be unnormalized float32 RGB in [0,1].")
            if min(rgb_last_frame.shape[-2:]) < 1:
                raise ValueError("rgb_last_frame spatial dimensions must be positive.")
            if rgb_last_frame.shape[0] != features.shape[0]:
                raise ValueError(
                    "rgb_last_frame batch size must match features: "
                    f"{rgb_last_frame.shape[0]} vs {features.shape[0]}"
                )
            if rgb_last_frame.device != features.device:
                raise ValueError(
                    "rgb_last_frame must be on the same device as features: "
                    f"{rgb_last_frame.device} vs {features.device}"
                )
        visual_out = self._build_visual_concepts_from_patches(features)
        align_out: Dict[str, Any] = {
            "loss_visual_saliency_align": None,
            "visual_saliency_patch_logits": None,
            "visual_saliency_align_valid_frac": None,
        }
        losses: Dict[str, torch.Tensor] = {}
        _fs = visual_out["visual_metadata"]["feature_shape"]
        _n_query = int(_fs["B"]) * int(_fs["H"]) * int(_fs["W"])  # last-slice grid only (T=1)
        self.visual_appearance_diagnostics = self._appearance_inactive_diagnostics(
            "return_losses_false" if not return_losses else "not_computed",
            num_patches=_n_query,
            device=visual_out["visual_concept_logits"].device,
        )
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
            appearance = self._visual_appearance_losses(visual_out, rgb_last_frame, cached_backbone_features)
            loss_appearance = appearance["combined"]
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
                "loss_visual_appearance": loss_appearance,
                "loss_visual_appearance_rgb": appearance["rgb"],
                "loss_visual_feature": appearance["feature"],
                "loss_total_concept": (
                    w["visual"] * loss_visual
                    + w["visual_div"] * loss_visual_div
                    + self.visual_entropy_weight * entropy
                    + self.visual_usage_weight * loss_usage
                    + self.visual_saliency_align_weight * loss_visual_saliency_align
                    + w["visual_appearance"] * loss_appearance
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
            "visual_queue_diagnostics": dict(self.visual_appearance_diagnostics),
            "visual_appearance_diagnostics": dict(self.visual_appearance_diagnostics),
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
    return _atomic_queue_save({"format": "prototype_rgb_feature_fifo_model_v3", "stages": states}, directory, filename)


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
    if (not isinstance(payload, dict) or payload.get("format") not in
            ("prototype_rgb_fifo_model_v2", "prototype_rgb_feature_fifo_model_v3")
            or not isinstance(payload.get("stages"), dict) or set(payload["stages"]) != set(modules)):
        raise ValueError("Queue snapshot format/stages do not match. Learned-embedding queues cannot be reused.")
    # Validate every stage before restoring any, including legacy RGB migration.
    expected_format = ("prototype_rgb_fifo_v2" if payload["format"] == "prototype_rgb_fifo_model_v2"
                       else "prototype_rgb_feature_fifo_v3")
    validated = {}
    for name, module in modules.items():
        state = payload["stages"][name]
        if not isinstance(state, dict) or state.get("format") != expected_format:
            raise ValueError(f"Queue stage {name!r} format disagrees with its model snapshot.")
        validated[name] = module._validated_queue_state(state)
    for name, module in modules.items():
        module._restore_validated_queue_state(validated[name])

