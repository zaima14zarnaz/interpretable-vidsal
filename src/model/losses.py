"""Auxiliary visual concept losses; no learned parameters are registered here.

The mixin leaves parameter ownership and checkpoint names on VisualConceptCreation.
Reconstruction, diversity, entropy, usage, saliency alignment, appearance costs,
and the exact weighted total preserve the supplied implementation.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
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
