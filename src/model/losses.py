"""Concept and full training losses; no learned parameters are registered here.

The mixin leaves parameter ownership and checkpoint names on VisualConceptCreation.
Reconstruction, diversity, entropy, usage, saliency alignment, appearance costs,
and the exact weighted total preserve the supplied implementation.
"""

from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F


class VisualConceptLossMixin:
    """Stateless loss methods; learned heads stay registered on the model."""

    DEFAULT_LOSS_WEIGHTS = {
        "visual": 0.1,
        "visual_div": 1.0,
        "visual_appearance": 1.0,
        "visual_rgb": 1.0,
        "visual_feature": 1.0,
        "visual_structure": 1.0,
    }

    def _appearance_coverage_zero(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
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

    def _appearance_status(
        self, active: bool, reason: str, device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        stage_enabled = self.stage_name == "stage4"
        metadata = self._cluster_metadata
        status = {
            "enabled": stage_enabled and self.loss_weights["visual_appearance"] > 0,
            "stage_name": self.stage_name,
            "appearance_scope": "stage4_only",
            "stage_source": self._appearance_stage_source,
            "active_this_forward": active and stage_enabled,
            "reason": reason if stage_enabled else "stage_excluded",
            "reference_mode": "fixed_patch_clusters",
            "reference_loaded": self._cluster_loaded,
            "reference_directory": self.visual_cluster_dir,
            "reference_fingerprint": metadata.get("fingerprint"),
            "source_checkpoint_sha256": metadata.get("source_checkpoint_sha256"),
            "source_feature_semantics": metadata.get("source_feature_semantics"),
            "reference_patch_counts": metadata.get("selected_counts", []),
            "full_cluster_patch_counts": metadata.get("full_counts", []),
            "reference_subset": self.visual_cluster_max_patches_per_prototype > 0,
            "capacity": 0,
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
    def _rgb_appearance_descriptors(
        self, rgb: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Uniform-grid crops resized like the bank's PIL float32 bilinear RGB."""
        r = self.visual_appearance_patch_size
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            image = rgb.detach().float()
            if (
                not torch.isfinite(image).all()
                or image.min() < -0.01
                or image.max() > 1.01
            ):
                raise ValueError(
                    "Appearance RGB must be finite and unnormalized in [0,1]."
                )
            image = image.clamp(0.0, 1.0)
            b, _, h, w = image.shape

            def resize(patches):
                origin = patches[:, :, :1, :1]
                resized = F.interpolate(
                    patches - origin,
                    size=(r, r),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                return (resized + origin).clamp(0.0, 1.0)

            if h % grid_h == 0 and w % grid_w == 0:
                ph, pw = (h // grid_h, w // grid_w)
                if ph < 1 or pw < 1:
                    raise ValueError("Empty input RGB patch grid.")
                patches = image.unfold(2, ph, ph).unfold(3, pw, pw)
                patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, ph, pw)
                resized = resize(patches)
            else:
                patches = []
                for row in range(grid_h):
                    y0, y1 = (round(row * h / grid_h), round((row + 1) * h / grid_h))
                    for col in range(grid_w):
                        x0, x1 = (
                            round(col * w / grid_w),
                            round((col + 1) * w / grid_w),
                        )
                        if y1 <= y0 or x1 <= x0:
                            raise ValueError(
                                "Empty input RGB patch under the reference grid mapping."
                            )
                        patches.append(resize(image[:, :, y0:y1, x0:x1]))
                resized = torch.stack(patches, dim=1).reshape(
                    b * grid_h * grid_w, 3, r, r
                )
            return resized.reshape(-1, self.visual_appearance_dim).contiguous()

    @torch.no_grad()
    def _cached_feature_descriptors(
        self, cached: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed normalized [B*H*W,C] reference vectors in live patch order."""
        with torch.autocast(device_type=cached.device.type, enabled=False):
            vectors = (
                cached.detach()
                .float()[:, :, 0]
                .permute(0, 2, 3, 1)
                .reshape(-1, self.in_channels)
                .contiguous()
            )
            if not torch.isfinite(vectors).all():
                raise ValueError(
                    "Cached reference vectors must remain finite when converted to float32."
                )
            scale = vectors.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            scaled = vectors / scale
            norms = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
            valid = norms.squeeze(-1) > 1e-12 / scale.squeeze(-1)
            normalized = scaled / norms.clamp_min(1e-12)
            normalized = torch.where(
                valid[:, None], normalized, torch.zeros_like(normalized)
            )
            return (normalized, valid)

    @staticmethod
    def _reference_assignment_weights(
        logits: torch.Tensor, populated: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Soft full-bank weights over nonempty fixed clusters; FP32 caller."""
        has_references = populated.any()
        fallback = ~has_references & (
            torch.arange(logits.shape[1], device=logits.device) == 0
        )
        safe_mask = populated | fallback
        weights = F.softmax(
            logits.float().masked_fill(~safe_mask[None, :], float("-inf")), dim=-1
        )
        return (weights, has_references.to(torch.float32))

    @torch.no_grad()
    def _rgb_structure_descriptors(
        self, descriptors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Channel-major RGB -> unit spatially centered RGB; flag flat patches."""
        rgb = descriptors.float().reshape(-1, 3, self.visual_appearance_patch_size**2)
        offsets = rgb - rgb[:, :, :1]
        centered = (offsets - offsets.mean(dim=-1, keepdim=True)).reshape(
            -1, self.visual_appearance_dim
        )
        norms = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        flat = norms.squeeze(-1) < 1e-12
        unit = centered / norms.clamp_min(1e-12)
        unit = torch.where(flat[:, None], torch.zeros_like(unit), unit)
        return (unit, flat)

    @torch.no_grad()
    def _fixed_appearance_costs(
        self, descriptors: torch.Tensor, feature_descriptors: Optional[torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Exact average distances to immutable cluster members, not medoids."""
        if not self._cluster_loaded:
            raise RuntimeError(
                "Fixed patch clusters are not loaded; call load_visual_patch_clusters outside forward."
            )
        queries = descriptors.shape[0]
        costs = {
            name: descriptors.new_zeros((queries, self.num_visual_concepts))
            for name in ("rgb", "feature", "structure")
        }
        if self.loss_weights["visual_rgb"] > 0:
            for start in range(0, queries, self.visual_appearance_query_chunk_size):
                stop = min(queries, start + self.visual_appearance_query_chunk_size)
                values = descriptors[start:stop].T.contiguous()
                for j, (ordered, prefix, count) in enumerate(
                    zip(
                        self._cluster_rgb_sorted,
                        self._cluster_rgb_prefix,
                        self._cluster_metadata["selected_counts"],
                    )
                ):
                    position = torch.searchsorted(ordered, values, right=False)
                    below = torch.gather(prefix, 1, position)
                    absolute_sum = (
                        (2.0 * position.float() - count) * values
                        + prefix[:, -1:]
                        - 2.0 * below
                    )
                    costs["rgb"][start:stop, j] = (
                        absolute_sum.sum(0) / (count * self.visual_appearance_dim)
                    ).clamp(0.0, 1.0)
        if self.loss_weights["visual_structure"] > 0:
            structure, flat = self._rgb_structure_descriptors(descriptors)
            similarity = structure @ self._cluster_structure_mean.T
            similarity += flat.float()[:, None] * self._cluster_flat_fraction[None, :]
            costs["structure"] = (1.0 - similarity).clamp(0.0, 2.0)
        if feature_descriptors is not None:
            costs["feature"] = (
                1.0 - feature_descriptors @ self._cluster_feature_mean.T
            ).clamp(0.0, 2.0)
        return (costs, self._cluster_counts > 0)

    def _visual_appearance_losses(
        self,
        visual_out: Dict[str, Any],
        rgb_last_frame: Optional[torch.Tensor],
        cached_backbone_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """E_assignment[mean fixed-cluster appearance distance], stage4 only."""
        logits = visual_out["visual_concept_logits"]
        zero = logits.sum() * 0.0
        result = {"rgb": zero, "feature": zero, "structure": zero, "combined": zero}
        w_rgb, w_feature = (
            self.loss_weights["visual_rgb"],
            self.loss_weights["visual_feature"],
        )
        w_structure = self.loss_weights["visual_structure"]
        num_patches = int(logits.shape[0])
        reason = None
        if self.stage_name != "stage4":
            reason = "stage_excluded"
        elif not self.training:
            reason = "evaluation"
        elif not torch.is_grad_enabled():
            reason = "no_grad"
        elif self.loss_weights["visual_appearance"] == 0 or (
            w_rgb == 0 and w_feature == 0 and (w_structure == 0)
        ):
            reason = "zero_weight"
        elif rgb_last_frame is None:
            reason = "no_rgb"
        elif w_feature > 0 and cached_backbone_features is None:
            raise RuntimeError(
                "Stage4 appearance feature term is enabled but cached_backbone_features is missing. Resolve frozen reference features for this exact RGB view (including hflip as its own content identity) before the concept forward, or set loss_weights['visual_feature']=0. Evaluation/no_grad paths skip this requirement."
            )
        if reason is not None:
            self.visual_appearance_diagnostics = self._appearance_inactive_diagnostics(
                reason, num_patches=num_patches, device=logits.device
            )
            return result
        if not self._cluster_loaded:
            raise FileNotFoundError(
                f"Fixed cluster bank not loaded from {self.visual_cluster_dir}. Run form_patch_clusters.py before constructing the model, or call load_visual_patch_clusters(path) before training."
            )
        shape = visual_out["visual_metadata"]["feature_shape"]
        descriptors = self._rgb_appearance_descriptors(
            rgb_last_frame, int(shape["H"]), int(shape["W"])
        )
        if descriptors.shape[0] != num_patches:
            raise ValueError(
                "RGB descriptors and assignment logits have different patch counts."
            )
        feature_descriptors = feature_valid = None
        if w_feature > 0:
            feature_descriptors, feature_valid = self._cached_feature_descriptors(
                cached_backbone_features
            )
            if feature_descriptors.shape[0] != num_patches:
                raise ValueError(
                    "Cached reference features and assignment logits have different patch counts."
                )
        with torch.autocast(device_type=logits.device.type, enabled=False):
            costs, populated = self._fixed_appearance_costs(
                descriptors, feature_descriptors
            )
            weights, has_references = self._reference_assignment_weights(
                logits, populated
            )
            valid = (
                feature_valid.float()
                if feature_valid is not None
                else torch.ones(num_patches, device=logits.device, dtype=torch.float32)
            )
            query_count = valid.sum()
            for name, weight in (
                ("rgb", w_rgb),
                ("feature", w_feature),
                ("structure", w_structure),
            ):
                if weight > 0:
                    result[name] = (
                        (weights * costs[name]).sum(-1) * valid
                    ).sum() / query_count.clamp_min(1.0)
                    result[name] = result[name] * has_references
            result["combined"] = (
                w_rgb * result["rgb"]
                + w_feature * result["feature"]
                + w_structure * result["structure"]
            )
        evaluated = query_count.detach() * has_references.detach()
        cov_zero = evaluated.new_zeros(())
        self._appearance_forward_count += 1
        self.visual_appearance_diagnostics = self._appearance_status(
            True, "fixed_reference_cluster_comparison", device=logits.device
        )
        self.visual_appearance_diagnostics.update(
            feature_active_this_forward=feature_descriptors is not None,
            feature_reason="fixed_reference_cluster_comparison"
            if w_feature > 0
            else "zero_weight",
            structure_active_this_forward=w_structure > 0,
            app_loss_patch_count_total=cov_zero + float(num_patches),
            app_loss_patch_count=evaluated,
            app_loss_rgb_patch_count=evaluated if w_rgb > 0 else cov_zero.clone(),
            app_loss_feature_patch_count=evaluated
            if w_feature > 0
            else cov_zero.clone(),
            app_loss_structure_patch_count=evaluated
            if w_structure > 0
            else cov_zero.clone(),
        )
        return result

    def _visual_diversity_loss(self) -> torch.Tensor:
        """Encourage visual concept bank prototypes to stay diverse."""
        bank_n = F.normalize(self.visual_concepts, dim=-1)
        cos = bank_n @ bank_n.T
        mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
        off_diag = cos[mask]
        if off_diag.numel() == 0:
            return self.visual_concepts.sum() * 0.0
        return F.relu(off_diag - self.diversity_margin).pow(2).mean()

    def _visual_concept_loss(
        self, visual_patch_embeddings: torch.Tensor, visual_activations: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable reconstruction loss for visual concept assignment."""
        c_vis = F.normalize(self.visual_concepts, dim=-1)
        recon = visual_activations @ c_vis
        return 1.0 - F.cosine_similarity(recon, visual_patch_embeddings, dim=-1).mean()

    def _visual_assignment_regularizers(
        self, visual_probs: torch.Tensor, eps: float = 1e-08
    ) -> Dict[str, torch.Tensor]:
        entropy = -(visual_probs * (visual_probs + eps).log()).sum(dim=-1).mean()
        mean_usage = visual_probs.mean(dim=0)
        uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
        # Preserve KL direction and batchmean reduction from the supplied model.
        loss_usage = F.kl_div((mean_usage + eps).log(), uniform, reduction="batchmean")
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
        """Build flattened patch saliency targets and a validity mask on the concept grid.
        Returns:
            targets: [B*T*H*W] in [0, 1]
            valid_mask: [B*T*H*W] bool"""
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
                sal_last.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
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
                    sal_flat, size=(H, W), mode="bilinear", align_corners=False
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
        return (targets, valid_mask)

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
            or (not torch.is_tensor(saliency_maps))
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
            saliency_maps, B, T, H, W, device=q_vis.device, dtype=q_vis.dtype
        )
        head_input = torch.cat([q_vis, visual_repr, visual_patch_coords], dim=-1)
        patch_logits = self.visual_saliency_head(head_input).squeeze(-1)
        if valid_mask.any():
            loss = F.binary_cross_entropy_with_logits(
                patch_logits[valid_mask], targets[valid_mask]
            )
        else:
            loss = zero
        return {
            "loss_visual_saliency_align": loss,
            "visual_saliency_patch_logits": patch_logits.detach(),
            "visual_saliency_align_valid_frac": valid_mask.float().mean().detach(),
        }

    def _compute_concept_losses(
        self,
        visual_out: Dict[str, Any],
        saliency_maps: Optional[torch.Tensor],
        rgb_last_frame: Optional[torch.Tensor],
        cached_backbone_features: Optional[torch.Tensor],
        return_losses: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Compute auxiliary terms and their original weighted sum."""
        align_out: Dict[str, Any] = {
            "loss_visual_saliency_align": None,
            "visual_saliency_patch_logits": None,
            "visual_saliency_align_valid_frac": None,
        }
        losses: Dict[str, torch.Tensor] = {}
        _fs = visual_out["visual_metadata"]["feature_shape"]
        _n_query = int(_fs["B"]) * int(_fs["H"]) * int(_fs["W"])
        self.visual_appearance_diagnostics = self._appearance_inactive_diagnostics(
            "return_losses_false" if not return_losses else "not_computed",
            num_patches=_n_query,
            device=visual_out["visual_concept_logits"].device,
        )
        if return_losses:
            loss_visual = self._visual_concept_loss(
                visual_out["visual_patch_embeddings"], visual_out["visual_activations"]
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
            appearance = self._visual_appearance_losses(
                visual_out, rgb_last_frame, cached_backbone_features
            )
            loss_appearance = appearance["combined"]
            w = self.loss_weights
            losses = {
                "loss_visual": loss_visual,
                "loss_visual_div": loss_visual_div,
                "loss_visual_assignment_entropy": entropy,
                "loss_visual_assignment_usage": loss_usage,
                "loss_visual_saliency_align": loss_visual_saliency_align,
                "loss_visual_appearance": loss_appearance,
                "loss_visual_appearance_rgb": appearance["rgb"],
                "loss_visual_appearance_structure": appearance["structure"],
                "loss_visual_feature": appearance["feature"],
                "loss_total_concept": w["visual"] * loss_visual
                + w["visual_div"] * loss_visual_div
                + self.visual_entropy_weight * entropy
                + self.visual_usage_weight * loss_usage
                + self.visual_saliency_align_weight * loss_visual_saliency_align
                + w["visual_appearance"] * loss_appearance,
            }
        return (losses, align_out)


# Full training objective and helpers from src/losses.py (formulas preserved).
def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(x)}")
    x = x.float()
    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    return x


def prepare_saliency_sequence(saliency_maps: torch.Tensor) -> torch.Tensor:
    """
    Normalize saliency layout to [B, 1, T, H, W].

    Accepts [B,H,W], [B,1,H,W], [B,T,H,W], [B,T,1,H,W], or [B,1,T,H,W].
    """
    if not isinstance(saliency_maps, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(saliency_maps)}")

    x = _to_float_tensor(saliency_maps)

    if x.dim() == 3:
        # [B, H, W] -> T=1
        return x.unsqueeze(1).unsqueeze(1)
    if x.dim() == 4:
        if x.shape[1] == 1:
            # [B, 1, H, W]
            return x.unsqueeze(2)
        # [B, T, H, W]
        return x.unsqueeze(1)
    if x.dim() == 5:
        if x.shape[1] == 1:
            # [B, 1, T, H, W]
            return x
        if x.shape[2] == 1:
            # [B, T, 1, H, W] -> [B, 1, T, H, W]
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(
            f"Unsupported 5D saliency shape {tuple(x.shape)}; "
            "expected [B,1,T,H,W] or [B,T,1,H,W]"
        )

    raise ValueError(
        f"Unsupported saliency_maps shape {tuple(x.shape)}; "
        "expected 3D, 4D, or 5D tensor"
    )


def resize_saliency_sequence(
    saliency_maps: torch.Tensor,
    target_t: int,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize saliency to [B, 1, target_t, target_h, target_w]."""
    x = prepare_saliency_sequence(saliency_maps)
    if x.shape[2:] == (target_t, target_h, target_w):
        return x
    return F.interpolate(
        x,
        size=(target_t, target_h, target_w),
        mode="trilinear",
        align_corners=False,
    )


def prepare_last_saliency_map(
    saliency_maps: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Last temporal frame, resized to [B, 1, target_h, target_w]."""
    x = prepare_saliency_sequence(saliency_maps)
    last = x[:, :, -1, :, :]  # [B, 1, H, W]
    if last.shape[-2:] == (target_h, target_w):
        return last
    return F.interpolate(
        last,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )


def has_temporal_saliency_sequence(saliency_maps: torch.Tensor) -> bool:
    """True only when saliency_maps has at least two real temporal frames."""
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


def prepare_patch_target_from_last_frame(
    saliency_maps: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Last-frame GT saliency resized to feature patch grid [B, 1, H, W]."""
    return prepare_last_saliency_map(saliency_maps, target_h, target_w)


def _read_feature_shape_from_metadata(
    metadata: Dict[str, Any],
) -> Tuple[int, int, int, int, int]:
    if "feature_shape" not in metadata:
        raise ValueError("metadata must contain 'feature_shape'")

    feature_shape = metadata["feature_shape"]
    if isinstance(feature_shape, dict):
        B = int(feature_shape["B"])
        C = int(feature_shape["C"])
        T = int(feature_shape["T"])
        H = int(feature_shape["H"])
        W = int(feature_shape["W"])
    else:
        shape = tuple(int(v) for v in feature_shape)
        if len(shape) != 5:
            raise ValueError(
                f"feature_shape must have 5 entries (B,C,T,H,W), got {shape}"
            )
        B, C, T, H, W = shape
    return B, C, T, H, W


def _to_long_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long).reshape(-1)
    return torch.tensor(x, device=device, dtype=torch.long).reshape(-1)


def _aggregate_incoming_source_saliency(
    source_values: torch.Tensor,
    pi: torch.Tensor,
    batch_idx: torch.Tensor,
    target_idx: torch.Tensor,
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Aggregate incoming source saliency onto the target patch grid.

    incoming_source[b, target] += pi * source_value
    """
    device = source_values.device
    dtype = source_values.dtype
    N = H * W
    flat_idx = batch_idx.long() * N + target_idx.long()
    weighted = source_values * pi
    patch_flat = torch.zeros(B * N, device=device, dtype=dtype)
    patch_flat.index_add_(0, flat_idx, weighted)
    return patch_flat.view(B, 1, H, W)


def _aggregate_target_patch_grid(
    target_values: torch.Tensor,
    batch_idx: torch.Tensor,
    target_idx: torch.Tensor,
    B: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Mean target saliency per patch when multiple trajectories share a target."""
    device = target_values.device
    dtype = target_values.dtype
    N = H * W
    flat_idx = batch_idx.long() * N + target_idx.long()

    target_sum = torch.zeros(B * N, device=device, dtype=dtype)
    target_count = torch.zeros(B * N, device=device, dtype=dtype)
    target_sum.index_add_(0, flat_idx, target_values)
    target_count.index_add_(0, flat_idx, torch.ones_like(target_values))
    target_patch = target_sum / target_count.clamp(min=1.0)
    return target_patch.view(B, 1, H, W)


def compute_delta_target(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Patch-grid delta target for the final transition into the last feature frame.

    Delta s_b = s_b^{t+1} - sum_a pi_ab * s_a^t

    Returns:
        delta_target, target_patch_grid, source_mixture_grid — each [B, 1, H, W],
        or (None, None, None) when saliency_maps has no real temporal sequence
        or when legacy trajectory-aggregation outputs are unavailable.
    """
    if not has_temporal_saliency_sequence(saliency_maps):
        return None, None, None

    # Legacy trajectory-aggregation head only. New ConceptGatedMultiScaleSaliencyDecoder
    # does not emit incoming_weights/selected_metadata.
    required_pred = ("incoming_weights", "selected_metadata")
    for key in required_pred:
        if key not in prediction_out:
            return None, None, None

    metadata = prediction_out["selected_metadata"]
    if not isinstance(metadata, dict):
        return None, None, None

    required_meta = (
        "batch_idx",
        "time_idx",
        "source_idx",
        "target_idx",
        "feature_shape",
    )
    for key in required_meta:
        if key not in metadata:
            return None, None, None

    B, _, T, H, W = _read_feature_shape_from_metadata(metadata)
    N = H * W

    pi = _to_float_tensor(prediction_out["incoming_weights"])
    device = pi.device

    batch_idx = _to_long_tensor(metadata["batch_idx"], device)
    time_idx = _to_long_tensor(metadata["time_idx"], device)
    source_idx = _to_long_tensor(metadata["source_idx"], device)
    target_idx = _to_long_tensor(metadata["target_idx"], device)

    sal_seq = resize_saliency_sequence(saliency_maps, T, H, W)  # [B, 1, T, H, W]
    sal_flat = sal_seq.squeeze(1).reshape(B, T, N)  # [B, T, N]

    s_a_t = sal_flat[batch_idx, time_idx, source_idx]
    s_b_t1 = sal_flat[batch_idx, time_idx + 1, target_idx]

    source_mixture_grid = _aggregate_incoming_source_saliency(
        s_a_t, pi, batch_idx, target_idx, B, H, W
    )
    target_patch_grid = _aggregate_target_patch_grid(
        s_b_t1, batch_idx, target_idx, B, H, W
    )
    delta_target = target_patch_grid - source_mixture_grid
    return delta_target, target_patch_grid, source_mixture_grid


def _get_zero_like_loss(reference_tensor: torch.Tensor) -> torch.Tensor:
    return reference_tensor.sum() * 0.0


def minmax_per_sample(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dim() < 3:
        return x
    B = x.shape[0]
    flat = x.reshape(B, -1)
    xmin = flat.min(dim=1, keepdim=True)[0]
    xmax = flat.max(dim=1, keepdim=True)[0]
    flat = (flat - xmin) / (xmax - xmin + eps)
    return flat.view_as(x)


def topk_weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    top_percent: float = 0.005,
    bg_weight: float = 0.15,
) -> torch.Tensor:
    """
    Weighted L1 loss that emphasizes the top target saliency pixels.

    Args:
        pred: [B,1,H,W], predicted saliency map in [0,1] or logits after activation.
        target: [B,1,H,W], target saliency map in [0,1].
        top_percent: fraction of pixels with highest target saliency to emphasize.
        bg_weight: weight for non-top-k pixels.

    Returns:
        scalar loss.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have same shape, got {tuple(pred.shape)} and {tuple(target.shape)}"
        )

    B = pred.shape[0]
    pred_f = pred.reshape(B, -1)
    target_f = target.reshape(B, -1)

    n_pixels = pred_f.shape[1]
    k = max(1, int(top_percent * n_pixels))

    _, top_idx = torch.topk(target_f, k=k, dim=1)

    weights = torch.full_like(target_f, bg_weight)
    weights.scatter_(1, top_idx, 1.0)

    return (weights * (pred_f - target_f).abs()).mean()


def spatial_kl_loss(
    pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    if pred_logits.shape != target.shape:
        raise ValueError(
            f"pred_logits and target must have the same shape, got "
            f"{tuple(pred_logits.shape)} and {tuple(target.shape)}"
        )

    B = pred_logits.shape[0]
    pred_flat = pred_logits.reshape(B, -1)
    target_flat = target.clamp(min=0.0).reshape(B, -1)
    target_flat = target_flat / target_flat.sum(dim=1, keepdim=True).clamp(min=eps)

    log_pred = F.log_softmax(pred_flat, dim=1)
    return F.kl_div(log_pred, target_flat, reduction="batchmean")


def spatial_cc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    CC loss for dense saliency maps.

    Uses 1 - Pearson correlation so the loss is non-negative and minimized
    when prediction and target are highly correlated.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )

    B = pred.shape[0]
    pred_f = pred.float().reshape(B, -1)
    target_f = target.float().reshape(B, -1)

    pred_f = pred_f - pred_f.mean(dim=1, keepdim=True)
    target_f = target_f - target_f.mean(dim=1, keepdim=True)

    pred_std = pred_f.std(dim=1, keepdim=True).clamp(min=eps)
    target_std = target_f.std(dim=1, keepdim=True).clamp(min=eps)

    pred_z = pred_f / pred_std
    target_z = target_f / target_std

    cc = (pred_z * target_z).mean(dim=1)
    return 1.0 - cc.mean()


def spatial_nss_loss(
    pred: torch.Tensor,
    fixation_map: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    NSS loss for dense saliency maps.

    pred: [B,1,H,W] final predicted saliency map or logits.
    fixation_map: [B,1,H,W] binary fixation map.
    Returns -NSS, so minimizing this maximizes NSS.
    """
    if pred.shape != fixation_map.shape:
        raise ValueError(
            f"pred and fixation_map must have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(fixation_map.shape)}"
        )

    B = pred.shape[0]
    pred_f = pred.float().reshape(B, -1)
    fix_f = (fixation_map.float().reshape(B, -1) > 0.0).float()

    pred_f = pred_f - pred_f.mean(dim=1, keepdim=True)
    pred_f = pred_f / pred_f.std(dim=1, keepdim=True).clamp(min=eps)

    fix_count = fix_f.sum(dim=1)
    valid = fix_count > 0

    if not valid.any():
        return pred.sum() * 0.0

    nss_per_sample = (pred_f * fix_f).sum(dim=1) / fix_count.clamp(min=1.0)
    nss = nss_per_sample[valid].mean()

    return -nss


def spatial_similarity_loss(
    pred: torch.Tensor,
    saliency_maps: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    SIM loss for dense saliency maps.

    Uses TMFI-Net-style histogram intersection: min-max normalize prediction
    and target, sum-normalize both maps, then compute sum(min(p, t)).
    Returns ``1 - SIM`` so minimizing this maximizes similarity.
    """
    if pred.dim() != 4 or pred.shape[1] != 1:
        raise ValueError(f"pred must be [B,1,H,W], got {tuple(pred.shape)}")

    target_h, target_w = int(pred.shape[-2]), int(pred.shape[-1])
    target = prepare_last_saliency_map(saliency_maps, target_h, target_w)
    if target.shape != pred.shape:
        raise ValueError(
            f"pred and target must have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )

    pred_norm = minmax_per_sample(pred)
    target_norm = minmax_per_sample(target)

    pred_flat = pred_norm.reshape(pred.shape[0], -1)
    target_flat = target_norm.reshape(target.shape[0], -1)
    pred_prob = pred_flat / pred_flat.sum(dim=1, keepdim=True).clamp(min=eps)
    target_prob = target_flat / target_flat.sum(dim=1, keepdim=True).clamp(min=eps)

    sim = torch.minimum(pred_prob, target_prob).sum(dim=1).mean()
    return 1.0 - sim


def compute_dense_metric_losses(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    fixation_maps: Optional[torch.Tensor] = None,
    lambda_cc: float = 0.0,
    lambda_nss: float = 0.0,
    lambda_similarity: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Optional dense saliency metric losses.

    CC and SIM use smoothed saliency_maps.
    NSS uses binary fixation_maps.
    """
    pred_dense = _resolve_final_saliency_map(prediction_out)
    zero = _get_zero_like_loss(pred_dense)

    loss_cc = zero
    if lambda_cc > 0:
        target_dense_last = prepare_last_saliency_map(
            saliency_maps,
            pred_dense.shape[-2],
            pred_dense.shape[-1],
        )
        target_dense_last = minmax_per_sample(target_dense_last)
        if target_dense_last.shape[-2:] != pred_dense.shape[-2:]:
            target_dense_last = F.interpolate(
                target_dense_last,
                size=pred_dense.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss_cc = spatial_cc_loss(pred_dense, target_dense_last)

    loss_nss = zero
    if lambda_nss > 0:
        if fixation_maps is None:
            raise ValueError(
                "lambda_nss > 0 requires binary fixation_maps. "
                "Do not compute NSS loss from smoothed saliency_maps."
            )

        target_fix_last = prepare_last_saliency_map(
            fixation_maps,
            pred_dense.shape[-2],
            pred_dense.shape[-1],
        )

        if target_fix_last.shape[-2:] != pred_dense.shape[-2:]:
            target_fix_last = F.interpolate(
                target_fix_last,
                size=pred_dense.shape[-2:],
                mode="nearest",
            )

        target_fix_last = (target_fix_last > 0.0).float()
        loss_nss = spatial_nss_loss(pred_dense, target_fix_last)
    loss_similarity = zero
    if lambda_similarity > 0:
        loss_similarity = spatial_similarity_loss(pred_dense, saliency_maps)
    return {
        "loss_cc": loss_cc,
        "loss_nss": loss_nss,
        "loss_sim": loss_similarity,
    }


def _resolve_final_saliency_logits(prediction_out: Dict[str, Any]) -> torch.Tensor:
    for key in ("saliency_logits", "coarse_saliency_logits"):
        value = prediction_out.get(key)
        if torch.is_tensor(value):
            return value
    raise ValueError(
        "prediction_out must contain 'saliency_logits' or fallback 'coarse_saliency_logits'"
    )


def _resolve_final_saliency_map(prediction_out: Dict[str, Any]) -> torch.Tensor:
    """
    Resolve the final dense saliency map used for main supervision losses.

    Priority: saliency_map, then legacy 'prediction' alias.
    """
    pred_map = prediction_out.get("saliency_map")
    if pred_map is None and "prediction" in prediction_out:
        pred_map = prediction_out["prediction"]
    if pred_map is None or not torch.is_tensor(pred_map):
        raise ValueError(
            "No final saliency prediction found. Expected prediction_out['saliency_map']."
        )
    return pred_map


def _is_new_decoder(prediction_out: Dict[str, Any]) -> bool:
    """
    True for ConceptGatedMultiScaleSaliencyDecoder outputs.

    selected_metadata belonged to the old trajectory-aggregation saliency head.
    The new concept-gated multiscale decoder does not need it because final
    prediction is made by dense concept-feature fusion and upsampling.
    """
    return (
        isinstance(prediction_out, dict)
        and "saliency_map" in prediction_out
        and "selected_metadata" not in prediction_out
    )


def _loss_zero_reference(prediction_out: Dict[str, Any]) -> torch.Tensor:
    """Reference tensor for zero auxiliary losses on the correct device."""
    for key in (
        "patch_saliency_logits",
        "saliency_logits",
        "coarse_saliency_logits",
        "saliency_map",
    ):
        value = prediction_out.get(key)
        if torch.is_tensor(value):
            return _get_zero_like_loss(value)
    return _get_zero_like_loss(_resolve_final_saliency_map(prediction_out))


def compute_fidelity_loss(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    lambda_delta: float = 1.0,
    lambda_dense: float = 1.0,
    lambda_bce: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_topk: float = 0.0,
    topk_percent: float = 0.05,
    topk_bg_weight: float = 0.15,
    patch_from_logits: bool = True,
) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Fidelity loss on patch saliency, optional patch delta, and dense last-frame saliency.

    Patch target always uses the last-frame GT (no fake temporal interpolation).
    Delta loss only when saliency_maps has T >= 2 and legacy trajectory outputs exist.
    """
    pred_dense = _resolve_final_saliency_map(prediction_out)
    zero_ref = _loss_zero_reference(prediction_out)

    patch_logits = prediction_out.get("patch_saliency_logits")
    if patch_logits is not None and (
        patch_logits.dim() != 4 or patch_logits.shape[1] != 1
    ):
        raise ValueError(
            f"patch_saliency_logits must be [B,1,H,W], got {tuple(patch_logits.shape)}"
        )

    if patch_logits is not None:
        if patch_from_logits:
            patch_saliency_pred = torch.sigmoid(patch_logits)
        else:
            patch_saliency_pred = patch_logits

        target_patch_grid = prepare_patch_target_from_last_frame(
            saliency_maps,
            patch_logits.shape[-2],
            patch_logits.shape[-1],
        )
        target_patch_grid = minmax_per_sample(target_patch_grid)
        loss_patch_fid = F.l1_loss(patch_saliency_pred, target_patch_grid)
    else:
        patch_saliency_pred = None
        target_patch_grid = None
        loss_patch_fid = zero_ref

    target_dense_last = prepare_last_saliency_map(
        saliency_maps,
        pred_dense.shape[-2],
        pred_dense.shape[-1],
    )
    target_dense_last = minmax_per_sample(target_dense_last)
    if target_dense_last.shape[-2:] != pred_dense.shape[-2:]:
        target_dense_last = F.interpolate(
            target_dense_last,
            size=pred_dense.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    loss_dense_fid = F.l1_loss(pred_dense, target_dense_last)

    loss_topk = zero_ref
    if lambda_topk > 0:
        loss_topk = topk_weighted_l1_loss(
            pred_dense,
            target_dense_last,
            top_percent=topk_percent,
            bg_weight=topk_bg_weight,
        )

    loss_dense_bce = zero_ref
    if lambda_bce > 0:
        loss_dense_bce = F.binary_cross_entropy_with_logits(
            _resolve_final_saliency_logits(prediction_out),
            target_dense_last.clamp(0.0, 1.0),
        )

    loss_dense_kl = zero_ref
    if lambda_kl > 0:
        loss_dense_kl = spatial_kl_loss(
            _resolve_final_saliency_logits(prediction_out),
            target_dense_last,
        )

    # Legacy trajectory delta head (patch_delta_logits / temporal_delta_logits).
    patch_delta_logits = prediction_out.get("patch_delta_logits")
    if patch_delta_logits is None:
        patch_delta_logits = prediction_out.get("temporal_delta_logits")

    use_delta = (
        lambda_delta > 0.0
        and has_temporal_saliency_sequence(saliency_maps)
        and patch_delta_logits is not None
        and not _is_new_decoder(prediction_out)
    )

    if use_delta:
        delta_target, _, source_mixture_grid = compute_delta_target(
            prediction_out, saliency_maps
        )
        if delta_target is not None:
            loss_delta = F.l1_loss(patch_delta_logits, delta_target)
        else:
            delta_target = None
            source_mixture_grid = None
            loss_delta = zero_ref
    else:
        delta_target = None
        source_mixture_grid = None
        loss_delta = zero_ref

    loss_fid = (
        loss_patch_fid
        + lambda_delta * loss_delta
        + lambda_dense * loss_dense_fid
        + lambda_bce * loss_dense_bce
        + lambda_kl * loss_dense_kl
        + lambda_topk * loss_topk
    )

    return {
        "loss_fid": loss_fid,
        "loss_patch_fid": loss_patch_fid,
        "loss_dense_fid": loss_dense_fid,
        "loss_dense_bce": loss_dense_bce,
        "loss_dense_kl": loss_dense_kl,
        "loss_topk": loss_topk,
        "loss_delta": loss_delta,
        "delta_target": delta_target,
        "target_patch_grid": target_patch_grid,
        "source_mixture_grid": source_mixture_grid,
    }


def _resolve_prediction_out(model_out: Dict[str, Any]) -> Dict[str, Any]:
    if "prediction_out" in model_out:
        return model_out["prediction_out"]
    return model_out


def _resolve_primary_concept_out(
    concept_out: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Single-scale concept_out or primary stage from multiscale dict."""
    if concept_out is None:
        return None
    if not isinstance(concept_out, dict) or not concept_out:
        return None

    first_val = next(iter(concept_out.values()))
    if isinstance(first_val, dict) and (
        "concept_representation" in first_val or "metadata" in first_val
    ):
        if "stage1" in concept_out:
            return concept_out["stage1"]
        return first_val
    return concept_out


def _inject_feature_shape(
    prediction_out: Dict[str, Any],
    concept_out: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ensure selected_metadata carries feature_shape for legacy delta targets.

    selected_metadata belongs to the old trajectory-aggregation saliency head.
    ConceptGatedMultiScaleSaliencyDecoder does not produce it; missing
    selected_metadata is valid and this function returns prediction_out unchanged.
    """
    if prediction_out is None or not isinstance(prediction_out, dict):
        return prediction_out

    if "selected_metadata" not in prediction_out:
        return prediction_out

    metadata = prediction_out.get("selected_metadata")
    if not isinstance(metadata, dict):
        return prediction_out

    if "feature_shape" in metadata:
        return prediction_out

    primary_concept_out = _resolve_primary_concept_out(concept_out)
    if primary_concept_out is not None and "metadata" in primary_concept_out:
        fs = primary_concept_out["metadata"].get("feature_shape")
        if fs is not None:
            metadata = dict(metadata)
            metadata["feature_shape"] = fs
            prediction_out = dict(prediction_out)
            prediction_out["selected_metadata"] = metadata
            return prediction_out

    return prediction_out


def _aggregate_concept_losses(
    concept_out: Any, reference: torch.Tensor
) -> Dict[str, torch.Tensor]:
    zero = reference.sum() * 0.0

    keys = ["loss_align", "loss_sparse", "loss_div", "loss_gate"]
    totals = {k: zero for k in keys}
    count = 0

    if isinstance(concept_out, dict) and concept_out:
        first_val = next(iter(concept_out.values()))
        if isinstance(first_val, dict) and (
            "losses" in first_val or "concept_representation" in first_val
        ):
            iterable = concept_out.values()
        else:
            iterable = [concept_out]
    elif isinstance(concept_out, dict):
        iterable = [concept_out]
    else:
        iterable = []

    for out in iterable:
        losses = out.get("losses", {}) if isinstance(out, dict) else {}
        if not losses:
            continue
        count += 1
        for k in keys:
            if k in losses and losses[k] is not None:
                totals[k] = totals[k] + losses[k]

    if count > 0:
        for k in keys:
            totals[k] = totals[k] / count

    return totals


def _aggregate_concept_total_loss(
    concept_out: Any, reference: torch.Tensor
) -> torch.Tensor:
    """Average per-stage ``loss_total_concept`` into the optimizer concept term.

    Appearance reaches ``loss_total`` only through this route:

        stage4 ``loss_visual_appearance``
          -> weighted by ``loss_weights['visual_appearance']`` inside stage4
             ``loss_total_concept``
          -> averaged here with the other stages' ``loss_total_concept``
          -> added once as ``loss_total_concept`` in ``compute_total_loss``

    Do **not** also add ``loss_visual_appearance`` into ``loss_total``.

    Migration note (optimizer aggregation unchanged): this function uses an
    equal mean over contributing stages. With four stages present, stage4's
    appearance contribution to ``loss_total`` is therefore divided by four:

        (w_visual_appearance * raw_loss_visual_appearance) / N_stages

    Weights are not silently re-scaled to undo that dilution.
    """
    zero = reference.sum() * 0.0
    if not isinstance(concept_out, dict) or not concept_out:
        return zero

    stage_losses: list[torch.Tensor] = []
    for _, stage_out in _iter_stage_concept_outputs(concept_out):
        losses = stage_out.get("losses", {}) if isinstance(stage_out, dict) else {}
        loss_total = losses.get("loss_total_concept")
        if torch.is_tensor(loss_total):
            stage_losses.append(loss_total)

    if not stage_losses:
        return zero
    return torch.stack(stage_losses).mean()


def _visual_loss_reference(model_out: Dict[str, Any]) -> torch.Tensor:
    prediction_out = _resolve_prediction_out(model_out)
    for key in (
        "patch_saliency_logits",
        "coarse_saliency_logits",
        "saliency_logits",
        "saliency_map",
    ):
        ref = prediction_out.get(key)
        if torch.is_tensor(ref):
            return ref
    raise ValueError("model_out missing reference tensor for visual concept losses")


def _iter_stage_concept_outputs(
    concept_out: Any,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if not isinstance(concept_out, dict) or not concept_out:
        return

    first_val = next(iter(concept_out.values()))
    if isinstance(first_val, dict) and (
        "losses" in first_val
        or "concept_representation" in first_val
        or "visual_concept_logits" in first_val
    ):
        for stage, stage_out in concept_out.items():
            if isinstance(stage_out, dict):
                yield stage, stage_out
    else:
        yield "default", concept_out


def _visual_logits_to_maps(stage_out: Dict[str, Any]) -> torch.Tensor:
    visual_logits = stage_out["visual_concept_logits"]
    visual_metadata = stage_out["visual_metadata"]
    feature_shape = visual_metadata["feature_shape"]

    B = int(feature_shape["B"])
    T = int(feature_shape["T"])
    H = int(feature_shape["H"])
    W = int(feature_shape["W"])
    K = visual_logits.shape[-1]

    logits_5d = visual_logits.reshape(B, T, H, W, K)
    maps = logits_5d.permute(0, 1, 4, 2, 3).contiguous()  # [B,T,K,H,W]
    return maps


def compute_visual_entropy_loss(
    model_out: Dict[str, Any],
    eps: float = 1e-8,
) -> torch.Tensor:
    concept_out = model_out.get("concept_out")
    ref = _visual_loss_reference(model_out)
    zero = _get_zero_like_loss(ref)

    if concept_out is None:
        return zero

    stage_losses: list[torch.Tensor] = []
    for _, stage_out in _iter_stage_concept_outputs(concept_out):
        if (
            "visual_concept_logits" not in stage_out
            or "visual_metadata" not in stage_out
        ):
            continue

        maps = _visual_logits_to_maps(stage_out)
        probs = F.softmax(maps, dim=2)
        entropy = -(probs * (probs + eps).log()).sum(dim=2)
        loss_stage = entropy.mean() / max(math.log(probs.shape[2]), eps)
        stage_losses.append(loss_stage)

    if not stage_losses:
        return zero

    return torch.stack(stage_losses).mean()


def _zero_from_model_out(model_out: Dict[str, Any]) -> torch.Tensor:
    prediction_out = _resolve_prediction_out(model_out)
    for key in (
        "coarse_saliency_logits",
        "saliency_logits",
        "saliency_map",
        "patch_saliency_logits",
    ):
        value = prediction_out.get(key)
        if torch.is_tensor(value):
            return value.sum() * 0.0

    concept_out = model_out.get("concept_out")
    for _, stage_out in _iter_stage_concept_outputs(concept_out):
        value = stage_out.get("visual_concept_logits")
        if torch.is_tensor(value):
            return value.sum() * 0.0

    raise ValueError("Could not find a reference tensor for zero loss.")


def compute_visual_usage_balance_loss(
    model_out: Dict[str, Any],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage the visual concept branch to use multiple concepts across a batch.

    This prevents collapsed solutions such as all patches being assigned to one concept.
    The loss is 0 when the marginal concept distribution is uniform and approaches 1
    when usage collapses to a single concept.

    Uses softmax probabilities from visual_concept_logits, not hard argmax indices.
    Averages the loss across all stages that expose visual_concept_logits.
    """
    concept_out = model_out.get("concept_out")
    if concept_out is None:
        return _zero_from_model_out(model_out)

    stage_losses: list[torch.Tensor] = []

    for _, stage_out in _iter_stage_concept_outputs(concept_out):
        visual_logits = stage_out.get("visual_concept_logits")
        if not torch.is_tensor(visual_logits):
            continue

        if visual_logits.dim() != 2:
            raise ValueError(
                "visual_concept_logits must have shape [num_patches_total, K], "
                f"got {tuple(visual_logits.shape)}"
            )

        K = int(visual_logits.shape[-1])
        if K <= 1:
            continue

        probs = F.softmax(visual_logits.float(), dim=-1)  # [P, K]
        mean_probs = probs.mean(dim=0)  # [K]

        marginal_entropy = -(mean_probs * (mean_probs + eps).log()).sum()
        marginal_entropy = marginal_entropy / max(math.log(K), eps)

        # 0 = balanced usage, 1 = collapsed usage
        loss_stage = 1.0 - marginal_entropy
        stage_losses.append(loss_stage)

    if not stage_losses:
        return _zero_from_model_out(model_out)

    return torch.stack(stage_losses).mean()


def compute_visual_flip_equivariance_loss(
    model_out: Dict[str, Any],
    equiv_model_out: Optional[Dict[str, Any]],
    eps: float = 1e-8,
) -> torch.Tensor:
    _ = eps
    ref = _visual_loss_reference(model_out)
    zero = _get_zero_like_loss(ref)

    if equiv_model_out is None:
        return zero

    concept_out_orig = model_out.get("concept_out")
    concept_out_flip = equiv_model_out.get("concept_out")
    if concept_out_orig is None or concept_out_flip is None:
        return zero

    flip_stages = {
        stage: stage_out
        for stage, stage_out in _iter_stage_concept_outputs(concept_out_flip)
    }

    stage_losses: list[torch.Tensor] = []
    for stage, stage_out_orig in _iter_stage_concept_outputs(concept_out_orig):
        stage_out_flip = flip_stages.get(stage)
        if stage_out_flip is None:
            continue
        if (
            "visual_concept_logits" not in stage_out_orig
            or "visual_metadata" not in stage_out_orig
            or "visual_concept_logits" not in stage_out_flip
            or "visual_metadata" not in stage_out_flip
        ):
            continue

        maps_orig = _visual_logits_to_maps(stage_out_orig)
        maps_flip = _visual_logits_to_maps(stage_out_flip)

        probs_orig = F.softmax(maps_orig, dim=2)
        probs_flip = F.softmax(maps_flip, dim=2)
        probs_flip_back = torch.flip(probs_flip, dims=[-1])
        loss_stage = F.mse_loss(probs_orig, probs_flip_back)
        stage_losses.append(loss_stage)

    if not stage_losses:
        return zero

    return torch.stack(stage_losses).mean()


def compute_concept_branch_loss(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    lambda_dense: float = 0.0,
    lambda_kl: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Fidelity on fused concept-only saliency (explainability branch)."""
    ref = prediction_out.get("patch_saliency_logits")
    if ref is None:
        ref = prediction_out.get("coarse_saliency_logits")
    if ref is None:
        ref = prediction_out.get("saliency_logits")
    if ref is None:
        raise ValueError(
            "prediction_out missing reference tensor for concept branch loss"
        )

    zero = _get_zero_like_loss(ref)
    concept_map = prediction_out.get("concept_saliency_map")
    concept_logits = prediction_out.get("concept_saliency_logits")

    if concept_map is None or concept_logits is None:
        return {
            "loss_concept_fid": zero,
            "loss_concept_dense": zero,
            "loss_concept_kl": zero,
        }

    target_dense_last = prepare_last_saliency_map(
        saliency_maps,
        concept_map.shape[-2],
        concept_map.shape[-1],
    )
    target_dense_last = minmax_per_sample(target_dense_last)

    loss_concept_dense = zero
    if lambda_dense > 0:
        loss_concept_dense = F.l1_loss(concept_map, target_dense_last)

    loss_concept_kl = zero
    if lambda_kl > 0:
        loss_concept_kl = spatial_kl_loss(concept_logits, target_dense_last)

    loss_concept_fid = loss_concept_dense + loss_concept_kl

    return {
        "loss_concept_fid": loss_concept_fid,
        "loss_concept_dense": loss_concept_dense,
        "loss_concept_kl": loss_concept_kl,
    }


DEFAULT_DECODER_SIDE_STAGE_WEIGHTS = {
    "stage1": 0.20,
    "stage2": 0.15,
    "stage3": 0.10,
    "stage4": 0.05,
}


def _saliency_map_from_logits(
    logits: torch.Tensor,
    output_activation: str,
) -> torch.Tensor:
    if output_activation == "sigmoid":
        return torch.sigmoid(logits)
    return logits


def compute_decoder_side_aux_loss(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    *,
    fixation_maps: Optional[torch.Tensor] = None,
    side_stage_weights: Optional[Dict[str, float]] = None,
    lambda_fid: float = 1.0,
    lambda_dense: float = 1.0,
    lambda_bce: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_cc: float = 0.0,
    lambda_nss: float = 0.0,
    lambda_topk: float = 0.0,
    topk_percent: float = 0.05,
    topk_bg_weight: float = 0.15,
    patch_from_logits: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Deep supervision on per-stage decoder side outputs.

    Applies the same dense saliency losses as the main head to each
    side_saliency_logits entry, weighted by stage.
    """
    side_logits_dict = prediction_out.get("side_saliency_logits")
    zero = _loss_zero_reference(prediction_out)
    out: Dict[str, torch.Tensor] = {"loss_decoder_side_aux": zero}

    if not isinstance(side_logits_dict, dict) or not side_logits_dict:
        return out

    weights = side_stage_weights or DEFAULT_DECODER_SIDE_STAGE_WEIGHTS
    output_activation = str(prediction_out.get("output_activation", "sigmoid"))
    total = zero

    for stage, side_logits in side_logits_dict.items():
        if not torch.is_tensor(side_logits):
            continue
        stage_weight = float(weights.get(stage, 0.0))
        if stage_weight <= 0.0:
            continue

        side_pred = {
            "saliency_logits": side_logits,
            "saliency_map": _saliency_map_from_logits(side_logits, output_activation),
        }
        fid_out = compute_fidelity_loss(
            side_pred,
            saliency_maps,
            lambda_delta=0.0,
            lambda_dense=lambda_dense,
            lambda_bce=lambda_bce,
            lambda_kl=lambda_kl,
            lambda_topk=lambda_topk,
            topk_percent=topk_percent,
            topk_bg_weight=topk_bg_weight,
            patch_from_logits=patch_from_logits,
        )
        metric_out = compute_dense_metric_losses(
            side_pred,
            saliency_maps,
            fixation_maps=fixation_maps,
            lambda_cc=lambda_cc,
            lambda_nss=lambda_nss,
        )

        stage_loss = (
            lambda_fid * fid_out["loss_fid"]
            + lambda_cc * metric_out["loss_cc"]
            + lambda_nss * metric_out["loss_nss"]
        )
        out[f"loss_side_{stage}"] = stage_loss
        total = total + stage_weight * stage_loss

    out["loss_decoder_side_aux"] = total
    return out


def compute_temporal_attention_entropy_loss(
    prediction_out: Dict[str, Any],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage non-collapsed temporal attention over decoder frames.

    Returns a loss term to be minimized. We use negative normalized entropy so a
    positive lambda encourages higher entropy (less early collapse to one frame).
    """
    zero = _loss_zero_reference(prediction_out)
    temporal_weights = prediction_out.get("temporal_weights")
    if not torch.is_tensor(temporal_weights):
        return zero
    if temporal_weights.dim() != 5:
        return zero

    # Expected shape: [B, 1, T, H, W]
    T = int(temporal_weights.shape[2])
    if T <= 1:
        return zero

    weights = temporal_weights.float().clamp(min=0.0)
    weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=eps)
    entropy = -(weights * (weights + eps).log()).sum(dim=2)
    normalized_entropy = entropy / max(math.log(T), eps)
    # Minimize negative entropy -> maximize entropy.
    return -normalized_entropy.mean()


def _forward_spatial_kl(
    target_prob: torch.Tensor,
    pred_prob: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Forward KL ``sum_j Y_j (log(Y_j+eps) - log(M_j+eps))`` over spatial dims."""
    return (target_prob * ((target_prob + eps).log() - (pred_prob + eps).log())).sum(
        dim=-1
    )


def compute_patch_priority_map_loss(
    prediction_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    *,
    stages: Tuple[str, ...] = ("stage3", "stage4"),
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Forward KL between last-frame patch-priority maps and downsampled saliency GT.

    Both maps are normalized into spatial probability distributions per batch
    sample. Stage 3 and Stage 4 losses are averaged. Samples whose ground-truth
    map has zero spatial mass are excluded from the average.
    """
    zero = _loss_zero_reference(prediction_out)
    maps = prediction_out.get("stage_patch_priority_maps")
    out: Dict[str, torch.Tensor] = {
        "loss_priority_map": zero,
        "loss_priority_map_stage3": zero,
        "loss_priority_map_stage4": zero,
        "priority_map_loss": zero,
    }
    if not isinstance(maps, dict) or not maps:
        return out

    stage_losses: List[torch.Tensor] = []
    for stage in stages:
        priority_map = maps.get(stage)
        if not torch.is_tensor(priority_map) or priority_map.dim() < 3:
            continue

        last_priority = priority_map[:, -1]
        if last_priority.dim() == 2:
            last_priority = last_priority.unsqueeze(1)
        elif last_priority.dim() == 4 and last_priority.shape[1] == 1:
            last_priority = last_priority.squeeze(1)

        target = prepare_last_saliency_map(
            saliency_maps,
            int(last_priority.shape[-2]),
            int(last_priority.shape[-1]),
        )

        pred_flat = last_priority.reshape(last_priority.shape[0], -1).clamp_min(0.0)
        pred_prob = pred_flat / pred_flat.sum(dim=1, keepdim=True).clamp_min(eps)
        target_flat = target.reshape(target.shape[0], -1).clamp_min(0.0)
        target_mass = target_flat.sum(dim=1)
        valid = target_mass > eps
        if not bool(valid.any().item()):
            continue

        target_prob = target_flat / target_mass.unsqueeze(-1).clamp_min(eps)
        kl_per_sample = _forward_spatial_kl(target_prob, pred_prob, eps=eps)
        stage_loss = kl_per_sample[valid].mean()
        out[f"loss_priority_map_{stage}"] = stage_loss
        stage_losses.append(stage_loss)

    if stage_losses:
        out["loss_priority_map"] = torch.stack(stage_losses).mean()
        out["priority_map_loss"] = out["loss_priority_map"]
    return out


def compute_total_loss(
    model_out: Dict[str, Any],
    saliency_maps: torch.Tensor,
    lambda_delta: float = 1.0,
    lambda_dense: float = 1.0,
    lambda_bce: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_fid: float = 1.0,
    lambda_topk: float = 0.0,
    lambda_cc: float = 0.0,
    lambda_nss: float = 0.0,
    lambda_similarity: float = 0.0,
    topk_percent: float = 0.05,
    topk_bg_weight: float = 0.15,
    lambda_concept_dense: float = 0.0,
    lambda_concept_kl: float = 0.0,
    lambda_align: float = 0.1,
    lambda_sparse: float = 0.01,
    lambda_div: float = 0.1,
    lambda_gate: float = 0.1,
    lambda_visual_entropy: float = 0.0,
    lambda_visual_equiv: float = 0.0,
    lambda_visual_usage: float = 0.0,
    lambda_temporal_attention_entropy: float = 0.0,
    equiv_model_out: Optional[Dict[str, Any]] = None,
    fixation_maps: Optional[torch.Tensor] = None,
    patch_from_logits: bool = True,
    enable_side_aux: bool = False,
    side_stage_weights: Optional[Dict[str, float]] = None,
    priority_map_loss_weight: float = 0.05,
    priority_loss_weight: Optional[float] = None,
) -> Dict[str, Union[torch.Tensor, None]]:
    """
    Total training loss for ExplainableVidSalModel (return_details=True).

    L = existing saliency losses
      + priority_map_loss_weight * mean(Stage3, Stage4 patch-priority KL)
    """
    if priority_loss_weight is not None:
        priority_map_loss_weight = float(priority_loss_weight)
    prediction_out = _resolve_prediction_out(model_out)
    concept_out = model_out.get("concept_out")

    # Safe for new decoder: no-op when selected_metadata is absent.
    prediction_out = _inject_feature_shape(prediction_out, concept_out)

    # Legacy trajectory-head auxiliary losses (delta, incoming weights, subpatch, etc.)
    # are skipped automatically when prediction_out lacks those keys.
    _ = _is_new_decoder(prediction_out)

    fid_out = compute_fidelity_loss(
        prediction_out,
        saliency_maps,
        lambda_delta=lambda_delta,
        lambda_dense=lambda_dense,
        lambda_bce=lambda_bce,
        lambda_kl=lambda_kl,
        lambda_topk=lambda_topk,
        topk_percent=topk_percent,
        topk_bg_weight=topk_bg_weight,
        patch_from_logits=patch_from_logits,
    )

    metric_out = compute_dense_metric_losses(
        prediction_out,
        saliency_maps,
        fixation_maps=fixation_maps,
        lambda_cc=lambda_cc,
        lambda_nss=lambda_nss,
        lambda_similarity=lambda_similarity,
    )

    loss_cc = metric_out["loss_cc"]
    loss_nss = metric_out["loss_nss"]
    loss_similarity = metric_out["loss_sim"]

    loss_fid = fid_out["loss_fid"]
    ref = loss_fid

    use_concept_reg = (
        lambda_align > 0.0
        or lambda_sparse > 0.0
        or lambda_div > 0.0
        or lambda_gate > 0.0
    )
    if use_concept_reg:
        concept_losses = _aggregate_concept_losses(concept_out, ref)
        loss_align = concept_losses["loss_align"]
        loss_sparse = concept_losses["loss_sparse"]
        loss_div = concept_losses["loss_div"]
        loss_gate = concept_losses["loss_gate"]
    else:
        zero = ref.sum() * 0.0
        loss_align = loss_sparse = loss_div = loss_gate = zero

    use_concept_branch = lambda_concept_dense > 0.0 or lambda_concept_kl > 0.0
    if use_concept_branch:
        concept_branch_out = compute_concept_branch_loss(
            prediction_out,
            saliency_maps,
            lambda_dense=lambda_concept_dense,
            lambda_kl=lambda_concept_kl,
        )
        loss_concept_dense = concept_branch_out["loss_concept_dense"]
        loss_concept_kl = concept_branch_out["loss_concept_kl"]
    else:
        zero = ref.sum() * 0.0
        loss_concept_dense = loss_concept_kl = zero

    loss_visual_entropy = (
        compute_visual_entropy_loss(model_out)
        if lambda_visual_entropy > 0
        else ref.sum() * 0.0
    )
    loss_visual_equiv = (
        compute_visual_flip_equivariance_loss(model_out, equiv_model_out)
        if lambda_visual_equiv > 0 and equiv_model_out is not None
        else ref.sum() * 0.0
    )

    zero = ref.sum() * 0.0
    if lambda_visual_usage > 0.0:
        loss_visual_usage = compute_visual_usage_balance_loss(model_out)
    else:
        loss_visual_usage = zero

    if lambda_temporal_attention_entropy > 0.0:
        loss_temporal_attention_entropy = compute_temporal_attention_entropy_loss(
            prediction_out
        )
    else:
        loss_temporal_attention_entropy = zero

    loss_decoder_side_aux = zero
    side_loss_out: Dict[str, torch.Tensor] = {}
    if enable_side_aux:
        side_loss_out = compute_decoder_side_aux_loss(
            prediction_out,
            saliency_maps,
            fixation_maps=fixation_maps,
            side_stage_weights=side_stage_weights,
            lambda_fid=lambda_fid,
            lambda_dense=lambda_dense,
            lambda_bce=lambda_bce,
            lambda_kl=lambda_kl,
            lambda_cc=lambda_cc,
            lambda_nss=lambda_nss,
            lambda_topk=lambda_topk,
            topk_percent=topk_percent,
            topk_bg_weight=topk_bg_weight,
            patch_from_logits=patch_from_logits,
        )
        loss_decoder_side_aux = side_loss_out["loss_decoder_side_aux"]

    priority_out: Dict[str, torch.Tensor] = {
        "loss_priority_map": zero,
        "loss_priority_map_stage3": zero,
        "loss_priority_map_stage4": zero,
        "priority_map_loss": zero,
    }
    if priority_map_loss_weight > 0.0:
        priority_out = compute_patch_priority_map_loss(prediction_out, saliency_maps)
    loss_priority_map = priority_out["loss_priority_map"]
    loss_priority_map_weighted = priority_map_loss_weight * loss_priority_map

    loss_total_concept = _aggregate_concept_total_loss(concept_out, ref)

    loss_total = (
        lambda_fid * loss_fid
        + lambda_cc * loss_cc
        + lambda_nss * loss_nss
        + lambda_similarity * loss_similarity
        + lambda_align * loss_align
        + lambda_sparse * loss_sparse
        + lambda_div * loss_div
        + lambda_gate * loss_gate
        + lambda_concept_dense * loss_concept_dense
        + lambda_concept_kl * loss_concept_kl
        + lambda_visual_entropy * loss_visual_entropy
        + lambda_visual_equiv * loss_visual_equiv
        + lambda_visual_usage * loss_visual_usage
        + lambda_temporal_attention_entropy * loss_temporal_attention_entropy
        + loss_decoder_side_aux
        + loss_total_concept
        + loss_priority_map_weighted
    )

    result = {
        "loss_total": loss_total,
        "loss_fid": fid_out["loss_fid"],
        "loss_patch_fid": fid_out["loss_patch_fid"],
        "loss_dense_fid": fid_out["loss_dense_fid"],
        "loss_dense_bce": fid_out["loss_dense_bce"],
        "loss_dense_kl": fid_out["loss_dense_kl"],
        "loss_topk": fid_out["loss_topk"],
        "loss_delta": fid_out["loss_delta"],
        "loss_cc": loss_cc,
        "loss_nss": loss_nss,
        "loss_sim": loss_similarity,
        "loss_concept_dense": loss_concept_dense,
        "loss_concept_kl": loss_concept_kl,
        "loss_align": loss_align,
        "loss_sparse": loss_sparse,
        "loss_div": loss_div,
        "loss_gate": loss_gate,
        "loss_visual_entropy": loss_visual_entropy,
        "loss_visual_equiv": loss_visual_equiv,
        "loss_visual_usage": loss_visual_usage,
        "loss_temporal_attention_entropy": loss_temporal_attention_entropy,
        "loss_total_concept": loss_total_concept,
        "loss_decoder_side_aux": loss_decoder_side_aux,
        "loss_priority_map": loss_priority_map,
        "loss_priority_map_weighted": loss_priority_map_weighted,
        "priority_map_loss": loss_priority_map,
        "weighted_priority_map_loss": loss_priority_map_weighted,
        "loss_priority_map_stage3": priority_out["loss_priority_map_stage3"],
        "loss_priority_map_stage4": priority_out["loss_priority_map_stage4"],
        "delta_target": fid_out["delta_target"],
        "target_patch_grid": fid_out["target_patch_grid"],
        "source_mixture_grid": fid_out["source_mixture_grid"],
    }
    for key, value in side_loss_out.items():
        if key != "loss_decoder_side_aux":
            result[key] = value
    return result
