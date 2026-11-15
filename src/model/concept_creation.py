"""
Stage4 visual consistency supervised by fixed appearance-based patch clusters.

References: /data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster
(output of form_patch_clusters.py). Learned prototype j corresponds to fixed
reference cluster j. RGB, structure and cached-feature distances are averaged
over every member by default, then weighted by the current full-bank assignment
probabilities. Model assignments never add, remove or relabel reference patches.

L_app = mean_i sum_j pi_ij * [w_rgb*mean_q RGB_MAE(i,q)
          + w_feature*mean_q (1-stage4_cosine(i,q))
          + w_structure*mean_q (1-centered_RGB_structure_cosine(i,q))], q in G_j.
Its existing overall visual_appearance weight is applied in loss_total_concept.
All costs are fixed targets; gradients flow through live assignment logits to
the encoder/prototypes/backbone. No additional raw-backbone alignment is added.

Only stage4 receives this objective. Other concept losses and prediction outputs
are preserved. Existing model.py callers work via the project channel mapping:
96/192/384/768 -> stage1/2/3/4; an explicit stage_name overrides that mapping.

The bank loads once in the constructor if present. If created later, call
module.load_visual_patch_clusters(path) outside forward. Training with supplied
appearance inputs fails explicitly when a required bank is not loaded; there
is no FIFO fallback. Evaluation/no_grad/return_losses=False do not use targets.

Cluster count, channels, RGB size, source resize convention, file hashes, and
SQLite/NPY membership identity are checked. RGB uses rounded uniform-grid crops
and antialiased bilinear resizing, matching the source graph's PIL float32 RGB
semantics (up to numerical roundoff); bank HWC vectors are converted to CHW.
Cached query features MUST come from the same reference backbone/checkpoint
and preprocessing as the cluster source. Their provenance cannot be inferred
from a bare tensor. Optional visual_cluster_expected_checkpoint_sha256 checks
the bank's declared source against a caller-supplied expected checkpoint hash.
No live/trainable feature fallback is used as an appearance target.

All fixed members are used by default (visual_cluster_max_patches_per_prototype=0).
A positive cap explicitly selects a seeded fixed subset, shared by all terms.
Cosine means and flat fractions are precomputed without renormalizing means.
Exact mean RGB MAE uses sorted coordinate values and prefix sums, avoiding an
O(queries*references*RGB_dim) pair tensor and direct query-to-every-reference
comparisons. FP32 reference memory is about 8*N_reference*RGB_dim bytes for MAE,
plus small cosine statistics. The optional reference cap bounds this memory.
There is no file I/O, inference, insertion or target update in forward, and
there is no empty-queue warmup. Zero cached query vectors are excluded from all
three components. Coverage and existing loss names remain available to train.py.

Reference tensors remain rank-local, immutable, FP32 and outside state_dict;
their device follows module.to()/cuda()/half(). Learned checkpoint keys/shapes
are unchanged. Keep the bank fixed during a run and across resume. Training
checkpoints store lightweight ``patch_cluster_reference`` identity metadata
only (fingerprints/counts/seed/cap); they never serialize reference arrays or
FIFO history. Deprecated queue-named aliases remain for external callers but
never restore historical patches or alter fixed memberships.
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

DEFAULT_PATCH_CLUSTER_DIR = "/data/quantization/zaima/videosal_datasets/dhf1k/patch_cluster"
PATCH_CLUSTER_REFERENCE_FORMAT = "patch_cluster_reference_v1"

# Deprecated path for legacy external snapshot helpers (no longer used by train.py).
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
        "visual_structure": 1.0,
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
        visual_queue_size: int = 12,  # Deprecated unused FIFO capacity; kept for callers.
        visual_queue_min_confidence: float = 0.0,  # Deprecated unused; kept for callers.
        visual_appearance_patch_size: int = 16,
        stage_name: Optional[str] = None,
        visual_appearance_query_chunk_size: int = 64,
        visual_cluster_dir: Optional[str] = None,
        visual_cluster_max_patches_per_prototype: int = 0,
        visual_cluster_seed: int = 42,
        visual_cluster_expected_checkpoint_sha256: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        if stage_name is not None and stage_name not in ("stage1", "stage2", "stage3", "stage4"):
            raise ValueError("stage_name must be stage1, stage2, stage3, stage4, or None.")
        # model.py currently supplies raw Video Swin stage channels but no name.
        # Do not infer a stage from H,W: feature grids vary with input resolution.
        self.stage_name = stage_name if stage_name is not None else {
            96: "stage1", 192: "stage2", 384: "stage3", 768: "stage4",
        }.get(in_channels)
        self._appearance_stage_source = "explicit" if stage_name is not None else "video_swin_channels"
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
        # Deprecated FIFO knobs: accepted for external constructors but unused.
        if isinstance(visual_queue_size, bool) or not isinstance(visual_queue_size, int) or visual_queue_size < 1:
            raise ValueError("visual_queue_size must be a positive integer.")
        if not math.isfinite(float(visual_queue_min_confidence)) or not 0 <= visual_queue_min_confidence <= 1:
            raise ValueError("visual_queue_min_confidence must lie in [0,1].")
        if visual_graph_db_path is not None or visual_graph_settings:
            raise ValueError("Use visual_cluster_dir for fixed patch-cluster supervision; remove graph configuration.")
        if (isinstance(visual_appearance_patch_size, bool) or
                not isinstance(visual_appearance_patch_size, int) or visual_appearance_patch_size < 1):
            raise ValueError("visual_appearance_patch_size must be a positive integer.")
        self.visual_appearance_patch_size = visual_appearance_patch_size
        if (isinstance(visual_appearance_query_chunk_size, bool) or
                not isinstance(visual_appearance_query_chunk_size, int) or
                visual_appearance_query_chunk_size < 1):
            raise ValueError("visual_appearance_query_chunk_size must be a positive integer.")
        self.visual_appearance_query_chunk_size = visual_appearance_query_chunk_size
        self.visual_appearance_dim = 3 * visual_appearance_patch_size ** 2
        self.visual_queue_size = int(visual_queue_size)  # unused; fixed bank has no capacity
        self.visual_queue_min_confidence = float(visual_queue_min_confidence)  # unused
        self.visual_graph_diagnostics = {"enabled": False, "reason": "replaced_by_fixed_patch_clusters"}
        weights = dict(self.DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            removed = {"visual_queue", "visual_queue_prototype"}.intersection(loss_weights)
            if removed:
                raise ValueError("Embedding queue losses were removed; use loss_weights['visual_appearance'] instead.")
            weights.update(loss_weights)
        for name in ("visual_appearance", "visual_rgb", "visual_feature", "visual_structure"):
            value = weights[name]
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"loss_weights[{name!r}] must be finite and nonnegative.")
        self.loss_weights = weights
        if (isinstance(visual_cluster_max_patches_per_prototype, bool) or
                not isinstance(visual_cluster_max_patches_per_prototype, int) or
                visual_cluster_max_patches_per_prototype < 0):
            raise ValueError("visual_cluster_max_patches_per_prototype must be an integer >= 0 (0 means all members).")
        if isinstance(visual_cluster_seed, bool) or not isinstance(visual_cluster_seed, int) or visual_cluster_seed < 0:
            raise ValueError("visual_cluster_seed must be a nonnegative integer.")
        self.visual_cluster_dir = str(Path(visual_cluster_dir or DEFAULT_PATCH_CLUSTER_DIR).expanduser().resolve())
        self.visual_cluster_max_patches_per_prototype = visual_cluster_max_patches_per_prototype
        self.visual_cluster_seed = visual_cluster_seed
        self.visual_cluster_expected_checkpoint_sha256 = visual_cluster_expected_checkpoint_sha256
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

        self._cluster_loaded = False
        self._cluster_metadata = {}
        self._cluster_feature_mean = torch.empty(0, dtype=torch.float32)
        self._cluster_structure_mean = torch.empty(0, dtype=torch.float32)
        self._cluster_flat_fraction = torch.empty(0, dtype=torch.float32)
        self._cluster_counts = torch.empty(0, dtype=torch.long)
        self._cluster_rgb_sorted = ()
        self._cluster_rgb_prefix = ()
        self._appearance_forward_count = 0
        self.reset_appearance_forward_counter()
        if (self.stage_name == "stage4" and self.loss_weights["visual_appearance"] > 0
                and any(self.loss_weights[k] > 0 for k in ("visual_rgb", "visual_feature", "visual_structure"))
                and (Path(self.visual_cluster_dir) / "manifest.json").is_file()):
            self.load_visual_patch_clusters()

    _CLUSTER_FLOAT_TENSORS = ("_cluster_feature_mean", "_cluster_structure_mean", "_cluster_flat_fraction")
    _CLUSTER_INT_TENSORS = ("_cluster_counts",)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Fixed reference statistics are plain, nonpersistent tensors. Device
        # follows the module, while targets retain FP32 precision under half().
        for name in self._CLUSTER_FLOAT_TENSORS + self._CLUSTER_INT_TENSORS:
            value = getattr(self, name, None)
            if value is not None:
                device = fn(torch.empty(0, device=value.device, dtype=value.dtype)).device
                setattr(self, name, value.to(device=device))
        for name in ("_cluster_rgb_sorted", "_cluster_rgb_prefix"):
            values = getattr(self, name, ())
            moved = []
            for value in values:
                device = fn(torch.empty(0, device=value.device, dtype=value.dtype)).device
                moved.append(value.to(device=device))
            setattr(self, name, tuple(moved))
        return result

    def _init_concept_parameters(self) -> None:
        with torch.no_grad():
            self.visual_concepts.copy_(F.normalize(self.visual_concepts, dim=-1))

    def configure_visual_graph(self, db_path: Optional[str], **settings: Any) -> None:
        """Legacy shim; use load_visual_patch_clusters for fixed references."""
        if db_path is not None or settings:
            raise ValueError("Use visual_cluster_dir/load_visual_patch_clusters, not graph configuration.")

    def _appearance_coverage_zero(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Detached scalar coverage counters on the model device (no .item())."""
        device = self.visual_concepts.device if device is None else device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            "app_loss_patch_count_total": zero.clone(),
            "app_loss_patch_count": zero.clone(),
            "app_loss_rgb_patch_count": zero.clone(),
            "app_loss_feature_patch_count": zero.clone(),
            "app_loss_structure_patch_count": zero.clone(),
        }

    def _appearance_status(self, active: bool, reason: str,
                           device: Optional[torch.device] = None) -> Dict[str, Any]:
        stage_enabled = self.stage_name == "stage4"
        metadata = self._cluster_metadata
        status = {
            "enabled": stage_enabled and self.loss_weights["visual_appearance"] > 0,
            "stage_name": self.stage_name, "appearance_scope": "stage4_only",
            "stage_source": self._appearance_stage_source,
            "active_this_forward": active and stage_enabled,
            "reason": reason if stage_enabled else "stage_excluded",
            "reference_mode": "fixed_patch_clusters", "reference_loaded": self._cluster_loaded,
            "reference_directory": self.visual_cluster_dir,
            "reference_fingerprint": metadata.get("fingerprint"),
            "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
            "source_feature_semantics": metadata.get("source_feature_semantics"),
            "reference_patch_counts": metadata.get("selected_counts", []),
            "full_cluster_patch_counts": metadata.get("full_counts", []),
            "reference_subset": self.visual_cluster_max_patches_per_prototype > 0,
            "capacity": 0,  # Fixed bank; no FIFO capacity.
            "training_forwards": self._appearance_forward_count,
            "assignment": "soft_full_bank_to_fixed_reference_cluster_ids",
            "representation": "bilinear_rgb_crops_and_frozen_reference_backbone",
            "patch_size": self.visual_appearance_patch_size,
            "descriptor_dim": self.visual_appearance_dim,
            "rgb_weight": self.loss_weights["visual_rgb"],
            "rgb_metric": "exact_mean_absolute_error_to_fixed_reference_members",
            "rgb_computation": "sorted_values_and_prefix_sums_no_patch_pair_tensor",
            "structure_weight": self.loss_weights["visual_structure"],
            "structure_metric": "per_channel_centered_rgb_cosine_both_flat_1_one_flat_0",
            "structure_active_this_forward": False,
            "query_chunk_size": self.visual_appearance_query_chunk_size,
            "metric_history": "immutable_same_reference_members_for_all_components",
            "feature_weight": self.loss_weights["visual_feature"],
            "feature_dim": self.in_channels,
            "feature_metric": "mean_cosine_distance_to_fixed_normalized_reference_members",
            "feature_active_this_forward": False,
            "feature_reason": reason if stage_enabled else "stage_excluded",
            "feature_zero_norm_policy": "exclude invalid queries from all components",
        }
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
        # Non-stage4 patches are outside this objective, including its coverage
        # denominator. Epoch coverage therefore measures stage4 patches only.
        if self.stage_name != "stage4":
            num_patches = 0
        zero = torch.zeros((), device=device, dtype=torch.float32)
        status.update(
            app_loss_patch_count_total=zero + float(num_patches),
            app_loss_patch_count=zero.clone(),
            app_loss_rgb_patch_count=zero.clone(),
            app_loss_feature_patch_count=zero.clone(),
            app_loss_structure_patch_count=zero.clone(),
        )
        if feature_reason is not None:
            status["feature_reason"] = feature_reason
        return status

    @torch.no_grad()
    def reset_appearance_forward_counter(self) -> None:
        """Reset the appearance-forward counter only; fixed memberships never clear."""
        self._appearance_forward_count = 0
        self.visual_appearance_diagnostics = self._appearance_status(
            False, "fixed_references_unchanged"
        )

    # Deprecated alias: never clears or rewrites fixed reference memberships.
    clear_prototype_queue = reset_appearance_forward_counter

    @torch.no_grad()
    def _rgb_appearance_descriptors(self, rgb: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """Uniform-grid crops resized like the bank's PIL float32 bilinear RGB.

        Output is batch-major/row-major, channel-major within each crop. Equal
        crop sizes use one batched resize. Odd sizes use matching round() grid
        boundaries. Constant channel offsets are preserved exactly.
        """
        r = self.visual_appearance_patch_size
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            image = rgb.detach().float()
            if not torch.isfinite(image).all() or image.min() < -0.01 or image.max() > 1.01:
                raise ValueError("Appearance RGB must be finite and unnormalized in [0,1].")
            image = image.clamp(0., 1.)
            b, _, h, w = image.shape
            def resize(patches):
                origin = patches[:, :, :1, :1]
                resized = F.interpolate(patches - origin, size=(r, r), mode="bilinear", align_corners=False, antialias=True)
                return (resized + origin).clamp(0., 1.)
            if h % grid_h == 0 and w % grid_w == 0:
                ph, pw = h // grid_h, w // grid_w
                if ph < 1 or pw < 1:
                    raise ValueError("Empty input RGB patch grid.")
                patches = image.unfold(2, ph, ph).unfold(3, pw, pw)
                patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, ph, pw)
                resized = resize(patches)
            else:
                patches = []
                for row in range(grid_h):
                    y0, y1 = round(row*h/grid_h), round((row+1)*h/grid_h)
                    for col in range(grid_w):
                        x0, x1 = round(col*w/grid_w), round((col+1)*w/grid_w)
                        if y1 <= y0 or x1 <= x0:
                            raise ValueError("Empty input RGB patch under the reference grid mapping.")
                        patches.append(resize(image[:, :, y0:y1, x0:x1]))
                resized = torch.stack(patches, dim=1).reshape(b*grid_h*grid_w, 3, r, r)
            return resized.reshape(-1, self.visual_appearance_dim).contiguous()

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
    def _reference_assignment_weights(logits: torch.Tensor, populated: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft full-bank weights over nonempty fixed clusters; FP32 caller."""
        has_references = populated.any()
        fallback = (~has_references) & (torch.arange(logits.shape[1], device=logits.device) == 0)
        safe_mask = populated | fallback
        weights = F.softmax(logits.float().masked_fill(~safe_mask[None, :], float("-inf")), dim=-1)
        return weights, has_references.to(torch.float32)

    # Deprecated alias name from the FIFO objective.
    _queue_assignment_weights = _reference_assignment_weights

    @torch.no_grad()
    def _rgb_structure_descriptors(self, descriptors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Channel-major RGB -> unit spatially centered RGB; flag flat patches."""
        rgb = descriptors.float().reshape(-1, 3, self.visual_appearance_patch_size ** 2)
        # Subtract a channel reference first. This is algebraically identical
        # to centering RGB, but keeps constant float32 colors exactly flat even
        # when a long reduction would round their channel mean slightly.
        offsets = rgb - rgb[:, :, :1]
        centered = (offsets - offsets.mean(dim=-1, keepdim=True)).reshape(-1, self.visual_appearance_dim)
        norms = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        flat = norms.squeeze(-1) < 1e-12
        unit = centered / norms.clamp_min(1e-12)
        unit = torch.where(flat[:, None], torch.zeros_like(unit), unit)
        return unit, flat

    @torch.no_grad()
    def _fixed_appearance_costs(self, descriptors: torch.Tensor,
                                feature_descriptors: Optional[torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Exact average distances to immutable cluster members, not medoids."""
        if not self._cluster_loaded:
            raise RuntimeError("Fixed patch clusters are not loaded; call load_visual_patch_clusters outside forward.")
        queries = descriptors.shape[0]
        costs = {name: descriptors.new_zeros((queries, self.num_visual_concepts)) for name in ("rgb", "feature", "structure")}
        if self.loss_weights["visual_rgb"] > 0:
            for start in range(0, queries, self.visual_appearance_query_chunk_size):
                stop = min(queries, start + self.visual_appearance_query_chunk_size)
                values = descriptors[start:stop].T.contiguous()
                for j, (ordered, prefix, count) in enumerate(zip(self._cluster_rgb_sorted,
                        self._cluster_rgb_prefix, self._cluster_metadata["selected_counts"])):
                    position = torch.searchsorted(ordered, values, right=False)
                    below = torch.gather(prefix, 1, position)
                    absolute_sum = (2.0 * position.float() - count) * values + prefix[:, -1:] - 2.0 * below
                    costs["rgb"][start:stop, j] = (absolute_sum.sum(0) / (count*self.visual_appearance_dim)).clamp(0., 1.)
        if self.loss_weights["visual_structure"] > 0:
            structure, flat = self._rgb_structure_descriptors(descriptors)
            similarity = structure @ self._cluster_structure_mean.T
            similarity += flat.float()[:, None] * self._cluster_flat_fraction[None, :]
            costs["structure"] = (1.0 - similarity).clamp(0., 2.)
        if feature_descriptors is not None:
            costs["feature"] = (1.0 - feature_descriptors @ self._cluster_feature_mean.T).clamp(0., 2.)
        return costs, self._cluster_counts > 0

    def _visual_appearance_losses(self, visual_out: Dict[str, Any],
                                   rgb_last_frame: Optional[torch.Tensor],
                                   cached_backbone_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """E_assignment[mean fixed-cluster appearance distance], stage4 only.

        Reference membership is independent of current assignments. No queue
        insertion, warmup, target drift or reference file I/O occurs in forward.
        """
        logits = visual_out["visual_concept_logits"]
        zero = logits.sum() * 0.0
        result = {"rgb": zero, "feature": zero, "structure": zero, "combined": zero}
        w_rgb, w_feature = self.loss_weights["visual_rgb"], self.loss_weights["visual_feature"]
        w_structure = self.loss_weights["visual_structure"]
        num_patches = int(logits.shape[0])
        reason = None
        if self.stage_name != "stage4":
            reason = "stage_excluded"
        elif not self.training:
            reason = "evaluation"
        elif not torch.is_grad_enabled():
            reason = "no_grad"
        elif self.loss_weights["visual_appearance"] == 0 or (w_rgb == 0 and w_feature == 0 and w_structure == 0):
            reason = "zero_weight"
        elif rgb_last_frame is None:
            reason = "no_rgb"
        elif w_feature > 0 and cached_backbone_features is None:
            # Enabled feature supervision must not silently become a zero loss.
            raise RuntimeError(
                "Stage4 appearance feature term is enabled but cached_backbone_features "
                "is missing. Resolve frozen reference features for this exact RGB view "
                "(including hflip as its own content identity) before the concept forward, "
                "or set loss_weights['visual_feature']=0. Evaluation/no_grad paths skip "
                "this requirement."
            )
        if reason is not None:
            self.visual_appearance_diagnostics = self._appearance_inactive_diagnostics(reason, num_patches=num_patches, device=logits.device)
            return result
        if not self._cluster_loaded:
            raise FileNotFoundError(f"Fixed cluster bank not loaded from {self.visual_cluster_dir}. Run form_patch_clusters.py before constructing the model, or call load_visual_patch_clusters(path) before training.")
        shape = visual_out["visual_metadata"]["feature_shape"]
        descriptors = self._rgb_appearance_descriptors(rgb_last_frame, int(shape["H"]), int(shape["W"]))
        if descriptors.shape[0] != num_patches:
            raise ValueError("RGB descriptors and assignment logits have different patch counts.")
        feature_descriptors = feature_valid = None
        if w_feature > 0:
            feature_descriptors, feature_valid = self._cached_feature_descriptors(cached_backbone_features)
            if feature_descriptors.shape[0] != num_patches:
                raise ValueError("Cached reference features and assignment logits have different patch counts.")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            costs, populated = self._fixed_appearance_costs(descriptors, feature_descriptors)
            weights, has_references = self._reference_assignment_weights(logits, populated)
            valid = feature_valid.float() if feature_valid is not None else torch.ones(num_patches, device=logits.device, dtype=torch.float32)
            query_count = valid.sum()
            for name, weight in (("rgb", w_rgb), ("feature", w_feature), ("structure", w_structure)):
                if weight > 0:
                    result[name] = ((weights * costs[name]).sum(-1) * valid).sum() / query_count.clamp_min(1.)
                    result[name] = result[name] * has_references
            result["combined"] = w_rgb * result["rgb"] + w_feature * result["feature"] + w_structure * result["structure"]
        evaluated = query_count.detach() * has_references.detach()
        cov_zero = evaluated.new_zeros(())
        self._appearance_forward_count += 1
        self.visual_appearance_diagnostics = self._appearance_status(True, "fixed_reference_cluster_comparison", device=logits.device)
        self.visual_appearance_diagnostics.update(
            feature_active_this_forward=feature_descriptors is not None,
            feature_reason="fixed_reference_cluster_comparison" if w_feature > 0 else "zero_weight",
            structure_active_this_forward=w_structure > 0,
            app_loss_patch_count_total=cov_zero + float(num_patches), app_loss_patch_count=evaluated,
            app_loss_rgb_patch_count=evaluated if w_rgb > 0 else cov_zero.clone(),
            app_loss_feature_patch_count=evaluated if w_feature > 0 else cov_zero.clone(),
            app_loss_structure_patch_count=evaluated if w_structure > 0 else cov_zero.clone())
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
    def _enqueue_rgb_descriptors(self, visual_out: Dict[str, Any], descriptors: torch.Tensor,
                                 feature_descriptors: Optional[torch.Tensor] = None,
                                 feature_valid: Optional[torch.Tensor] = None) -> None:
        """Deprecated no-op: predictions never append, evict, or relabel fixed members."""
        return None

    @torch.no_grad()
    def fixed_reference_identity_state(self) -> Dict[str, Any]:
        """Lightweight identity snapshot for external callers (never reference arrays)."""
        return {
            "format": "fixed_patch_cluster_reference_v1",
            "stage_name": self.stage_name,
            "num_prototypes": self.num_visual_concepts,
            "in_channels": self.in_channels,
            "stage_enabled": self.stage_name == "stage4",
            "loaded": self._cluster_loaded,
            "reference_directory": self.visual_cluster_dir,
            "reference_fingerprint": self._cluster_metadata.get("fingerprint"),
            "max_patches_per_prototype": self.visual_cluster_max_patches_per_prototype,
            "reference_seed": self.visual_cluster_seed,
            "step": self._appearance_forward_count,
        }

    # Deprecated alias: never serializes FIFO patch history.
    prototype_queue_state = fixed_reference_identity_state

    @torch.no_grad()
    def _validated_reference_identity_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reference identity; explicitly ignore obsolete FIFO history."""
        if not isinstance(state, dict):
            raise ValueError("Reference snapshot must be a dictionary.")
        if state.get("format") in ("prototype_rgb_fifo_v2", "prototype_rgb_feature_fifo_v3"):
            if state.get("num_prototypes") != self.num_visual_concepts or state.get("in_channels") != self.in_channels:
                raise ValueError("Legacy snapshot dimensions differ from the model.")
            import warnings
            warnings.warn(
                "Ignoring obsolete FIFO history; fixed patch clusters remain the appearance targets.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"step": 0, "legacy_ignored": True}
        if (state.get("format") != "fixed_patch_cluster_reference_v1"
                or state.get("stage_name") != self.stage_name
                or state.get("num_prototypes") != self.num_visual_concepts
                or state.get("in_channels") != self.in_channels):
            raise ValueError("Reference snapshot stage/dimensions differ from the model.")
        if self.stage_name == "stage4" and state.get("loaded"):
            if not self._cluster_loaded or state.get("reference_fingerprint") != self._cluster_metadata.get("fingerprint"):
                raise ValueError(
                    "Fixed cluster bank changed since the snapshot; restore the original "
                    "bank/settings before resuming."
                )
        step = state.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Invalid appearance-forward count in reference snapshot.")
        return {"step": step, "legacy_ignored": False}

    # Deprecated alias.
    _validated_queue_state = _validated_reference_identity_state

    @torch.no_grad()
    def _restore_validated_reference_identity_state(self, state: Dict[str, Any]) -> None:
        """Restore forward counters only; never rewrite fixed memberships."""
        self._appearance_forward_count = state["step"]
        self.visual_appearance_diagnostics = self._appearance_status(
            False,
            "legacy_fifo_ignored" if state["legacy_ignored"] else "fixed_reference_identity_restored",
        )

    # Deprecated alias.
    _restore_validated_queue_state = _restore_validated_reference_identity_state

    @torch.no_grad()
    def load_fixed_reference_identity_state(self, state: Dict[str, Any]) -> None:
        """Validate external identity snapshots; ignore obsolete FIFO contents."""
        self._restore_validated_reference_identity_state(
            self._validated_reference_identity_state(state)
        )

    # Deprecated alias: never restores historical patches into the bank.
    load_prototype_queue_state = load_fixed_reference_identity_state

    def save_fixed_reference_identity(self, directory: str = DEFAULT_PROTOTYPE_QUEUE_DIR,
                                      filename: str = "prototype_queue.pt") -> str:
        """Persist identity/counters only (no reference arrays, no FIFO history)."""
        return _atomic_reference_snapshot_save(
            self.fixed_reference_identity_state(), directory, filename
        )

    # Deprecated alias.
    save_prototype_queue = save_fixed_reference_identity

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
                L2 normalized for fixed-cluster cosine-distance supervision. It remains
                separate from learned ``visual_patch_embeddings``. When the
                feature component is enabled, missing cached features skip the
                appearance objective; zero vectors are excluded from every term.
                Set visual_feature=0 for an RGB/structure-only ablation.
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
        # Appearance inputs are unused outside stage4. Skip their validation,
        # finite-value reductions and descriptor work for the excluded stages.
        if self.stage_name != "stage4":
            rgb_last_frame = None
            cached_backbone_features = None
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
                "loss_visual_appearance_structure": appearance["structure"],
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
            "visual_patch_cluster_diagnostics": dict(self.visual_appearance_diagnostics),
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


    @torch.no_grad()
    def load_visual_patch_clusters(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """Load/precompute immutable reference groups OUTSIDE forward.

        All members are used by default. An explicit positive reference cap
        selects a seeded, fixed subset per cluster, shared by all three terms.
        Loading is transactional: failed validation preserves any existing bank.
        An already loaded bank may only be reloaded with the same fingerprint.
        This method loads targets only; learned prototype parameters are untouched.
        """
        if self.stage_name != "stage4":
            return {"loaded": False, "reason": "stage_excluded"}
        import hashlib
        import json
        import sqlite3
        import numpy as np

        root = Path(directory or self.visual_cluster_dir).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Fixed patch clusters not found: {manifest_path}. Run form_patch_clusters.py first.")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("format_version") != "fixed_stage4_patch_clusters_v1" or manifest.get("status") != "complete":
            raise ValueError("Expected a complete fixed_stage4_patch_clusters_v1 reference bank.")
        if manifest.get("stage") != "stage4" or manifest.get("rgb_order") != "HWC":
            raise ValueError("Reference bank must use stage4 and HWC RGB descriptors.")
        if int(manifest["num_prototypes"]) != self.num_visual_concepts:
            raise ValueError(f"Cluster count {manifest['num_prototypes']} != stage4 prototype count {self.num_visual_concepts}.")
        if int(manifest["feature_dim"]) != self.in_channels:
            raise ValueError(f"Cluster feature dimension {manifest['feature_dim']} != stage4 channels {self.in_channels}.")
        if int(manifest["rgb_descriptor_size"]) != self.visual_appearance_patch_size:
            raise ValueError(f"Cluster RGB size {manifest['rgb_descriptor_size']} != visual_appearance_patch_size {self.visual_appearance_patch_size}; use matching settings.")
        resize = str((manifest.get("rgb_settings") or {}).get("rgb_resize", "")).lower()
        if "bilinear" not in resize:
            raise ValueError("Cluster bank must document bilinear RGB crop resizing to match the training descriptor.")
        # Prefer backbone state_dict fingerprint provenance. Whole-model file
        # hashes are informational and are not treated as feature identity.
        feature_provenance = manifest.get("reference_feature_provenance")
        if not isinstance(feature_provenance, dict):
            feature_provenance = None
        checkpoint = (manifest.get("source_manifest") or {}).get("checkpoint_sha256")
        expected = self.visual_cluster_expected_checkpoint_sha256
        if expected is not None:
            expected_fp = feature_provenance.get("backbone_fingerprint") if feature_provenance else None
            if expected_fp is not None and expected_fp == expected:
                pass  # caller supplied a backbone fingerprint
            elif checkpoint != expected:
                raise ValueError(
                    "Reference bank checkpoint SHA / backbone fingerprint does not match "
                    "visual_cluster_expected_checkpoint_sha256. Prefer comparing "
                    "reference_feature_provenance.backbone_fingerprint, not a whole-model "
                    "checkpoint file hash."
                )
        references_path = root / "cluster_references.npz"
        if not references_path.is_file():
            raise FileNotFoundError(f"Cluster references missing: {references_path}")
        h = hashlib.sha256()
        with references_path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != manifest.get("references_sha256"):
            raise ValueError("cluster_references.npz fingerprint differs from the completed manifest.")

        count, m, d = int(manifest["num_patches"]), self.num_visual_concepts, self.visual_appearance_dim
        if count < m:
            raise ValueError("Reference bank has fewer patches than prototypes.")
        feature = np.load(root / "descriptors/feature.npy", mmap_mode="r", allow_pickle=False)
        rgb = np.load(root / "descriptors/rgb.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(root / "descriptors/prototype_ids.npy", mmap_mode="r", allow_pickle=False)
        node_ids = np.load(root / "descriptors/node_ids.npy", mmap_mode="r", allow_pickle=False)
        if (feature.shape != (count, self.in_channels) or rgb.shape != (count, d)
                or labels.shape != (count,) or node_ids.shape != (count,)):
            raise ValueError("Cluster descriptor arrays disagree with manifest dimensions.")
        if feature.dtype.kind != "f" or rgb.dtype.kind != "f" or labels.dtype.kind not in "iu" or node_ids.dtype.kind not in "iu":
            raise ValueError("Invalid reference descriptor/identity dtypes.")
        if np.any(labels < 0) or np.any(labels >= m):
            raise ValueError("Reference prototype IDs are outside the model bank.")
        full_counts = np.bincount(labels.astype(np.int64), minlength=m)
        if (full_counts == 0).any():
            raise ValueError("Every stage4 prototype requires a nonempty fixed reference cluster.")
        # Verify that the loss uses precisely the membership vocabulary shown
        # by view_patch_clusters.py, including stable source-node identities.
        con = sqlite3.connect((root / "patch_clusters.sqlite").as_uri() + "?mode=ro", uri=True)
        try:
            cursor = con.execute("SELECT descriptor_index,node_id,prototype_id FROM patches ORDER BY descriptor_index")
            start = 0
            while True:
                rows = cursor.fetchmany(16384)
                if not rows:
                    break
                actual = np.asarray(rows, dtype=np.int64)
                stop = start + len(rows)
                if (stop > count or not np.array_equal(actual[:, 0], np.arange(start, stop))
                        or not np.array_equal(actual[:, 1], node_ids[start:stop])
                        or not np.array_equal(actual[:, 2], labels[start:stop])):
                    raise ValueError("Reference memberships disagree with patch_clusters.sqlite.")
                start = stop
            if start != count:
                raise ValueError("Cluster database row count differs from descriptor arrays.")
            if con.execute("SELECT id,patch_count FROM prototypes ORDER BY id").fetchall() != [(j, int(full_counts[j])) for j in range(m)]:
                raise ValueError("Prototype summary counts disagree with reference memberships.")
        finally:
            con.close()

        feature_means, structure_means, flat_fractions = [], [], []
        sorted_values, prefix_sums, selected_counts = [], [], []
        digest = hashlib.sha256(manifest_bytes)
        digest.update(str((self.visual_cluster_max_patches_per_prototype, self.visual_cluster_seed)).encode())
        order = np.argsort(labels, kind="stable")
        boundaries = np.concatenate(([0], np.cumsum(full_counts)))
        for j in range(m):
            indices = order[boundaries[j]:boundaries[j + 1]]
            limit = self.visual_cluster_max_patches_per_prototype
            if limit > 0 and len(indices) > limit:
                rng = np.random.default_rng(np.random.SeedSequence([self.visual_cluster_seed, j]))
                indices = np.sort(rng.choice(indices, size=limit, replace=False))
            f = np.asarray(feature[indices], dtype=np.float64)
            r_hwc = np.asarray(rgb[indices], dtype=np.float64)
            if (not np.isfinite(f).all() or not np.isfinite(r_hwc).all()
                    or r_hwc.min() < 0 or r_hwc.max() > 1):
                raise ValueError(f"Invalid feature/RGB values in reference cluster {j}.")
            scales = np.max(np.abs(f), axis=1, keepdims=True)
            scaled = f / np.maximum(scales, 1e-12)
            norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            if (norms[:, 0] * scales[:, 0] < 1e-12).any():
                raise ValueError(f"Zero stage4 reference feature in cluster {j}.")
            unit_feature = scaled / norms
            r = self.visual_appearance_patch_size
            pixels = r_hwc.reshape(-1, r*r, 3)
            centered = pixels - pixels.mean(axis=1, keepdims=True)
            structure_norms = np.linalg.norm(centered.reshape(len(indices), -1), axis=1)
            flat = structure_norms < 1e-12
            unit_structure = centered / np.maximum(structure_norms[:, None, None], 1e-12)
            unit_structure[flat] = 0.
            # Convert external HWC vectors to the internal channel-major order.
            structure_chw = unit_structure.reshape(-1, r, r, 3).transpose(0, 3, 1, 2).reshape(-1, d)
            rgb_chw = r_hwc.reshape(-1, r, r, 3).transpose(0, 3, 1, 2).reshape(-1, d)
            feature_means.append(unit_feature.mean(axis=0).astype(np.float32))
            structure_means.append(structure_chw.mean(axis=0).astype(np.float32))
            flat_fractions.append(float(flat.mean()))
            # Exact mean L1 can be recovered from each coordinate's sorted
            # values and cumulative sums, with O(log n) lookup per query.
            ordered_rgb = np.sort(rgb_chw.astype(np.float32).T, axis=1)
            prefix = np.concatenate((np.zeros((d, 1), dtype=np.float64),
                                     np.cumsum(ordered_rgb, axis=1, dtype=np.float64)), axis=1).astype(np.float32)
            sorted_values.append(torch.from_numpy(np.ascontiguousarray(ordered_rgb)))
            prefix_sums.append(torch.from_numpy(np.ascontiguousarray(prefix)))
            selected_counts.append(len(indices))
            digest.update(np.asarray(indices, dtype="<i8").tobytes())
            digest.update(np.asarray(f, dtype="<f4").tobytes())
            digest.update(np.asarray(rgb_chw, dtype="<f4").tobytes())

        fingerprint = digest.hexdigest()
        if self._cluster_loaded and self._cluster_metadata.get("fingerprint") != fingerprint:
            raise ValueError("Cannot replace an already loaded fixed cluster bank; construct a new model for a different bank/reference subset.")
        device = self.visual_concepts.device
        # Prepare all tensors before replacing an existing loaded bank.
        fixed_feature = torch.from_numpy(np.stack(feature_means)).to(device=device)
        fixed_structure = torch.from_numpy(np.stack(structure_means)).to(device=device)
        fixed_flat = torch.tensor(flat_fractions, dtype=torch.float32, device=device)
        fixed_counts = torch.tensor(selected_counts, dtype=torch.long, device=device)
        fixed_sorted = tuple(value.to(device=device) for value in sorted_values)
        fixed_prefix = tuple(value.to(device=device) for value in prefix_sums)
        metadata = {"fingerprint": fingerprint, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "source_checkpoint_sha256": checkpoint,
                    "source_feature_semantics": manifest.get("source_feature_semantics"),
                    "reference_feature_provenance": feature_provenance,
                    "reference_feature_provenance_status": manifest.get(
                        "reference_feature_provenance_status",
                        "verified" if feature_provenance else "unverified_legacy_source",
                    ),
                    "selected_counts": selected_counts, "full_counts": full_counts.tolist(),
                    "num_reference_patches": int(sum(selected_counts)),
                    "rgb_resize": "bilinear_crop_antialias", "descriptor_order": "CHW (converted from bank HWC)",
                    "cluster_seed": self.visual_cluster_seed,
                    "max_patches_per_prototype": self.visual_cluster_max_patches_per_prototype,
                    "source_clustering_weights": manifest.get("weights"),
                    "reference_tensor_bytes": sum(v.numel()*v.element_size() for v in
                        (fixed_feature, fixed_structure, fixed_flat, fixed_counts, *fixed_sorted, *fixed_prefix))}
        self._cluster_feature_mean, self._cluster_structure_mean = fixed_feature, fixed_structure
        self._cluster_flat_fraction, self._cluster_counts = fixed_flat, fixed_counts
        self._cluster_rgb_sorted, self._cluster_rgb_prefix = fixed_sorted, fixed_prefix
        self._cluster_metadata, self._cluster_loaded = metadata, True
        self.visual_cluster_dir = str(root)
        self.visual_appearance_diagnostics = self._appearance_status(False, "fixed_clusters_loaded")
        return dict(metadata)



# Backward-compatible alias for existing imports.
ConceptCreation = VisualConceptCreation


def _unwrap_model(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def _concept_modules(model):
    """Named VisualConceptCreation modules with stable parallel-wrapper unwrap."""
    model = _unwrap_model(model)
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, VisualConceptCreation)
    }


# Deprecated alias.
_queue_modules = _concept_modules


def _atomic_reference_snapshot_save(payload: Dict[str, Any], directory: str, filename: str) -> str:
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Reference snapshot filename must be a simple .pt filename.")
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".ref-", suffix=".pt", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(path)


# Deprecated alias.
_atomic_queue_save = _atomic_reference_snapshot_save


def build_patch_cluster_reference_checkpoint_metadata(model) -> Optional[Dict[str, Any]]:
    """Lightweight identity for ordinary training checkpoints (no reference arrays).

    Reuses the bank fingerprint already computed by ``load_visual_patch_clusters``;
    does not rehash the bank.
    """
    model = _unwrap_model(model)
    concept_creations = getattr(model, "concept_creations", None)
    # nn.ModuleDict is Mapping-like but not a dict subclass; support both.
    if concept_creations is None or not hasattr(concept_creations, "__contains__"):
        return None
    if "stage4" not in concept_creations:
        return None
    stage4 = concept_creations["stage4"]
    if stage4 is None or not getattr(stage4, "_cluster_loaded", False):
        return None
    meta = stage4._cluster_metadata
    payload: Dict[str, Any] = {
        "format": PATCH_CLUSTER_REFERENCE_FORMAT,
        "metadata_format_version": 1,
        "bank_fingerprint": meta.get("fingerprint"),
        "manifest_sha256": meta.get("manifest_sha256"),
        "num_clusters": int(stage4.num_visual_concepts),
        "rgb_descriptor_size": int(stage4.visual_appearance_patch_size),
        "rgb_descriptor_dim": int(stage4.visual_appearance_dim),
        "max_patches_per_prototype": int(stage4.visual_cluster_max_patches_per_prototype),
        "reference_seed": int(stage4.visual_cluster_seed),
        "full_member_counts": list(meta.get("full_counts") or []),
        "selected_member_counts": list(meta.get("selected_counts") or []),
        "reference_directory": stage4.visual_cluster_dir,
        "reference_feature_provenance_status": meta.get(
            "reference_feature_provenance_status"
        ),
    }
    provenance = meta.get("reference_feature_provenance")
    if isinstance(provenance, dict):
        # Only verified identity fields when the bank recorded them.
        payload["reference_feature_provenance"] = {
            "backbone_fingerprint": provenance.get("backbone_fingerprint"),
            "temporal_selection": provenance.get("temporal_selection"),
            "preprocess": provenance.get("preprocess"),
            "feature_source": provenance.get("feature_source"),
        }
    return payload


def save_fixed_reference_identities(
    model,
    directory: str = DEFAULT_PROTOTYPE_QUEUE_DIR,
    filename: str = "prototype_queues.pt",
) -> Optional[str]:
    """Save fixed-reference identities/counters for all stages (no arrays/FIFO).

    Under DDP, rank 0 writes; other ranks return None. Never called from forward.
    """
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    ):
        return None
    modules = _concept_modules(model)
    if not modules:
        raise ValueError("No VisualConceptCreation modules found in model.")
    states = {
        name: module.fixed_reference_identity_state() for name, module in modules.items()
    }
    return _atomic_reference_snapshot_save(
        {"format": "fixed_patch_cluster_model_v1", "stages": states},
        directory,
        filename,
    )


# Deprecated alias: train.py no longer calls this per epoch.
save_prototype_queues = save_fixed_reference_identities


def load_fixed_reference_identities(
    model,
    directory: str = DEFAULT_PROTOTYPE_QUEUE_DIR,
    filename: str = "prototype_queues.pt",
) -> None:
    """Validate external identity snapshots after weights load. Never from forward.

    Does not restore historical FIFO patches or alter fixed memberships. Reference
    arrays remain in visual_cluster_dir and must already be loaded.
    """
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("Reference snapshot filename must be a simple .pt filename.")
    payload = torch.load(
        Path(directory).expanduser() / filename, map_location="cpu", weights_only=True
    )
    modules = _concept_modules(model)
    if (
        not isinstance(payload, dict)
        or payload.get("format")
        not in (
            "prototype_rgb_fifo_model_v2",
            "prototype_rgb_feature_fifo_model_v3",
            "fixed_patch_cluster_model_v1",
        )
        or not isinstance(payload.get("stages"), dict)
        or set(payload["stages"]) != set(modules)
    ):
        raise ValueError("Reference snapshot format/stages do not match the model.")
    expected_format = {
        "prototype_rgb_fifo_model_v2": "prototype_rgb_fifo_v2",
        "prototype_rgb_feature_fifo_model_v3": "prototype_rgb_feature_fifo_v3",
        "fixed_patch_cluster_model_v1": "fixed_patch_cluster_reference_v1",
    }[payload["format"]]
    validated = {}
    for name, module in modules.items():
        state = payload["stages"][name]
        if not isinstance(state, dict) or state.get("format") != expected_format:
            raise ValueError(
                f"Reference stage {name!r} format disagrees with its model snapshot."
            )
        validated[name] = module._validated_reference_identity_state(state)
    for name, module in modules.items():
        module._restore_validated_reference_identity_state(validated[name])


# Deprecated alias.
load_prototype_queues = load_fixed_reference_identities

