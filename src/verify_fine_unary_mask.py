#!/usr/bin/env python3
"""Focused checks for stage1/stage2 unary-only prototype masking."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.saliency_prediction import (  # noqa: E402
    ConceptGatedMultiScaleSaliencyDecoder,
    SpatioTemporalConceptGatedFusionBlock,
    _scatter_last_frame_update,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _stage_channels() -> Dict[str, int]:
    return {
        "stage1": 24,
        "stage2": 32,
        "stage3": 40,
        "stage4": 48,
    }


def _spatial_for_stage(stage: str) -> Tuple[int, int]:
    return {
        "stage4": (2, 2),
        "stage3": (4, 4),
        "stage2": (8, 8),
        "stage1": (16, 16),
    }[stage]


def _build_decoder(
    *,
    fine_unary_mask_strength: float = 0.6,
    prototype_application_position: str = "post_refine",
) -> ConceptGatedMultiScaleSaliencyDecoder:
    torch.manual_seed(0)
    return ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels=_stage_channels(),
        concept_dim=16,
        decoder_channels=20,
        dropout=0.0,
        output_activation="none",
        temporal_aggregation="last",
        use_side_logit_fusion=False,
        use_shared_concept_activations=False,
        assignment_temperature=0.07,
        priority_temperature=0.1,
        factorized_rank=8,
        prototype_bottleneck_strength=0.4,
        prototype_application_position=prototype_application_position,
        fine_unary_mask_strength=fine_unary_mask_strength,
    )


def _synthetic_batch(
    decoder: ConceptGatedMultiScaleSaliencyDecoder,
    *,
    batch: int = 2,
    temporal: int = 3,
    num_concepts: int = 3,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]], Tuple[int, int]]:
    concept_dim = decoder.concept_dim
    features: Dict[str, torch.Tensor] = {}
    concept_outs: Dict[str, Dict[str, Any]] = {}
    for stage, channels in decoder.stage_channels.items():
        h, w = _spatial_for_stage(stage)
        features[stage] = torch.randn(batch, channels, temporal, h, w)
        active = torch.randn(batch, 1, h, w, num_concepts, concept_dim)
        validity = torch.ones(
            batch, 1, h, w, num_concepts, dtype=torch.bool
        )
        # Leave one invalid slot so validity handling is exercised.
        validity[..., -1] = False
        visual_repr = torch.randn(batch * temporal * h * w, concept_dim)
        concept_outs[stage] = {
            "active_visual_prototypes": active,
            "visual_validity_mask": validity,
            "visual_concept_representation": visual_repr,
            "visual_metadata": {
                "feature_shape": {
                    "B": batch,
                    "C": channels,
                    "T": temporal,
                    "H": h,
                    "W": w,
                },
                "window_T": temporal,
            },
        }
    return features, concept_outs, (64, 64)


def _run_forward(
    decoder: ConceptGatedMultiScaleSaliencyDecoder,
    features: Dict[str, torch.Tensor],
    concept_outs: Dict[str, Dict[str, Any]],
    output_size: Tuple[int, int],
) -> Dict[str, Any]:
    return decoder(
        concept_outs,
        features,
        output_size=output_size,
        return_details=True,
    )


def test_modes_and_modules() -> None:
    decoder = _build_decoder()
    for stage in ("stage3", "stage4"):
        block = decoder.fusion_blocks[stage]
        _assert(block.priority_mode == "pairwise", f"{stage} should be pairwise")
        _assert(block.factor_query is not None, f"{stage} needs factor_query")
        _assert(block.factor_key is not None, f"{stage} needs factor_key")
        _assert(
            block.mask_feature_branch is not None,
            f"{stage} needs mask_feature_branch",
        )
        _assert(
            block.unary_priority_mlp is not None,
            f"{stage} needs unary_priority_mlp",
        )
    for stage in ("stage1", "stage2"):
        block = decoder.fusion_blocks[stage]
        _assert(block.priority_mode == "unary_mask", f"{stage} should be unary_mask")
        _assert(block.factor_query is None, f"{stage} must not build factor_query")
        _assert(block.factor_key is None, f"{stage} must not build factor_key")
        _assert(
            block.raw_spatial_frequency_weights is None,
            f"{stage} must not build spatial kernels",
        )
        _assert(
            block.raw_pairwise_strength is None,
            f"{stage} must not build pairwise_strength",
        )
        _assert(
            block.mask_feature_branch is None,
            f"{stage} must not build concept-conditioned update branch",
        )
        _assert(
            block.priority_application_proto_proj is None,
            f"{stage} must not build priority_application_proto_proj",
        )
        _assert(
            block.unary_priority_mlp is not None,
            f"{stage} needs unary_priority_mlp",
        )
        _assert(
            block.feature_token_proj is not None,
            f"{stage} needs feature_token_proj",
        )
        _assert(
            block.concept_proto_proj is not None,
            f"{stage} needs concept_proto_proj",
        )


def test_mask_shapes_finite_and_flags() -> Dict[str, Any]:
    decoder = _build_decoder()
    features, concept_outs, output_size = _synthetic_batch(decoder)
    out = _run_forward(decoder, features, concept_outs, output_size)

    masks = out["stage_priority_masks"]
    diagnostics = out["stage_mask_diagnostics"]
    logits = out["saliency_logits"]
    _assert(torch.isfinite(logits).all(), "saliency logits must be finite")

    shapes = {}
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        mask = masks[stage]
        h, w = _spatial_for_stage(stage)
        _assert(torch.is_tensor(mask), f"{stage} mask missing")
        _assert(torch.isfinite(mask).all(), f"{stage} mask must be finite")
        shapes[stage] = tuple(mask.shape)
        # Native spatial resolution must match the stage.
        _assert(
            mask.shape[-2:] == (h, w),
            f"{stage} mask spatial {mask.shape[-2:]} != {(h, w)}",
        )
        # Scattered priority_mask is [B, 1, T, H, W].
        _assert(
            mask.dim() == 5 and mask.shape[1] == 1,
            f"{stage} expected [B,1,T,H,W], got {tuple(mask.shape)}",
        )
        _assert(
            diagnostics[stage]["pairwise_context_used"]
            is (stage in ("stage3", "stage4")),
            f"{stage} pairwise_context_used mismatch",
        )
        expected_mode = (
            "pairwise" if stage in ("stage3", "stage4") else "unary_mask"
        )
        _assert(
            diagnostics[stage]["priority_mode"] == expected_mode,
            f"{stage} priority_mode={diagnostics[stage]['priority_mode']}",
        )
        pairwise_ctx = out.get("stage_pairwise_context") or {}
        if stage in ("stage1", "stage2"):
            _assert(
                stage not in pairwise_ctx,
                f"{stage} must not publish pairwise_context",
            )
        else:
            _assert(
                stage in pairwise_ctx,
                f"{stage} must publish pairwise_context",
            )
            _assert(
                torch.isfinite(pairwise_ctx[stage]).all(),
                f"{stage} pairwise_context must be finite",
            )

    # Last-frame gate before scatter is [B,1,H,W] (T squeezed) or [B,T,H,W].
    for stage in ("stage1", "stage2", "stage3", "stage4"):
        gate = out["stage_patch_priority_gates"][stage]
        h, w = _spatial_for_stage(stage)
        _assert(
            gate.shape[-2:] == (h, w),
            f"{stage} gate spatial {gate.shape[-2:]} != {(h, w)}",
        )
        shapes[f"{stage}_gate"] = tuple(gate.shape)
        shapes[f"{stage}_last_frame_mask"] = (
            mask.shape[0],
            1,
            1,
            h,
            w,
        )

    return {"shapes": shapes, "logits_shape": tuple(logits.shape)}


def test_pairwise_raise_trap() -> None:
    decoder = _build_decoder()
    features, concept_outs, output_size = _synthetic_batch(decoder)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("pairwise context must not run at fine stages")

    for stage in ("stage1", "stage2"):
        block = decoder.fusion_blocks[stage]
        block._factorized_pairwise_context = _boom  # type: ignore[method-assign]
        _assert(
            block.priority_mode == "unary_mask",
            f"{stage} mode drifted",
        )

    out = _run_forward(decoder, features, concept_outs, output_size)
    _assert(torch.isfinite(out["saliency_logits"]).all(), "trap forward not finite")


def test_prototype_perturbation_effects() -> Dict[str, Any]:
    decoder = _build_decoder()
    features, concept_outs, output_size = _synthetic_batch(decoder)
    base = _run_forward(decoder, features, concept_outs, output_size)

    results: Dict[str, Any] = {}
    for stage in ("stage1", "stage2"):
        perturbed = copy.deepcopy(concept_outs)
        perturbed[stage]["active_visual_prototypes"] = (
            perturbed[stage]["active_visual_prototypes"] + 1.7
        )
        out = _run_forward(decoder, features, perturbed, output_size)
        mask_delta = (
            out["stage_priority_masks"][stage] - base["stage_priority_masks"][stage]
        ).abs().max().item()
        logit_delta = (
            out["saliency_logits"] - base["saliency_logits"]
        ).abs().max().item()
        _assert(mask_delta > 1e-6, f"{stage} mask did not change under proto perturb")
        _assert(
            logit_delta > 1e-6,
            f"final logits did not change under {stage} proto perturb",
        )
        results[stage] = {
            "mask_delta": mask_delta,
            "logit_delta": logit_delta,
        }
    return results


def test_earlier_temporal_slices_unchanged() -> None:
    decoder = _build_decoder(fine_unary_mask_strength=0.6)
    features, concept_outs, output_size = _synthetic_batch(decoder)

    captures: Dict[str, Dict[str, torch.Tensor]] = {}

    for stage in ("stage1", "stage2"):
        block = decoder.fusion_blocks[stage]
        original_scatter = _scatter_last_frame_update

        def _make_hook(stage_name: str, blk: SpatioTemporalConceptGatedFusionBlock):
            # Capture decoded_base vs guided by intercepting refine2 output and
            # comparing after the block forward via a local wrapper.
            orig_forward = blk.forward

            def wrapped_forward(*args, **kwargs):
                # Re-implement capture around refine by temporarily wrapping refine2.
                orig_refine2 = blk.refine2

                class _Capture(nn.Module):
                    def __init__(self, inner):
                        super().__init__()
                        self.inner = inner
                        self.pre_mask = None

                    def forward(self, x):
                        y = self.inner(x)
                        self.pre_mask = y
                        return y

                capturer = _Capture(orig_refine2)
                blk.refine2 = capturer
                try:
                    decoded, mask_outputs = orig_forward(*args, **kwargs)
                finally:
                    blk.refine2 = orig_refine2
                captures[stage_name] = {
                    "before": capturer.pre_mask.detach().clone(),
                    "after": decoded.detach().clone(),
                }
                return decoded, mask_outputs

            return wrapped_forward

        block.forward = _make_hook(stage, block)  # type: ignore[method-assign]

    _run_forward(decoder, features, concept_outs, output_size)

    for stage in ("stage1", "stage2"):
        before = captures[stage]["before"]
        after = captures[stage]["after"]
        _assert(before.shape == after.shape, f"{stage} temporal capture shape mismatch")
        _assert(
            torch.allclose(before[:, :, :-1], after[:, :, :-1], atol=0.0, rtol=0.0),
            f"{stage} earlier temporal slices changed after unary mask",
        )
        _assert(
            not torch.allclose(before[:, :, -1:], after[:, :, -1:], atol=1e-8),
            f"{stage} last temporal slice unexpectedly identical after mask",
        )


def test_gradients_nonzero() -> Dict[str, float]:
    decoder = _build_decoder()
    features, concept_outs, output_size = _synthetic_batch(decoder)
    for stage_feats in features.values():
        stage_feats.requires_grad_(False)
    for stage, cout in concept_outs.items():
        cout["active_visual_prototypes"] = cout["active_visual_prototypes"].detach().requires_grad_(False)

    decoder.zero_grad(set_to_none=True)
    out = _run_forward(decoder, features, concept_outs, output_size)
    out["saliency_logits"].sum().backward()

    grads: Dict[str, float] = {}
    for stage in ("stage1", "stage2"):
        block = decoder.fusion_blocks[stage]
        for name, param in block.named_parameters():
            if not any(
                key in name
                for key in (
                    "unary_priority_mlp",
                    "feature_token_proj",
                    "concept_proto_proj",
                )
            ):
                continue
            _assert(param.grad is not None, f"{stage}.{name} has no grad")
            gnorm = float(param.grad.detach().norm().cpu())
            grads[f"{stage}.{name}"] = gnorm
            _assert(gnorm > 0.0, f"{stage}.{name} grad norm is zero")
    return grads


def main() -> None:
    print("=== verify_fine_unary_mask ===")
    test_modes_and_modules()
    print("PASS modes_and_modules")

    shape_info = test_mask_shapes_finite_and_flags()
    print("PASS mask_shapes_finite_and_flags")
    print(f"  mask/gate shapes: {shape_info['shapes']}")
    print(f"  logits shape: {shape_info['logits_shape']}")

    test_pairwise_raise_trap()
    print("PASS pairwise_raise_trap")

    pert = test_prototype_perturbation_effects()
    print("PASS prototype_perturbation_effects")
    for stage, stats in pert.items():
        print(
            f"  {stage}: mask_delta={stats['mask_delta']:.6g} "
            f"logit_delta={stats['logit_delta']:.6g}"
        )

    test_earlier_temporal_slices_unchanged()
    print("PASS earlier_temporal_slices_unchanged")

    grads = test_gradients_nonzero()
    print("PASS gradients_nonzero")
    for name, gnorm in grads.items():
        print(f"  {name}: grad_norm={gnorm:.6g}")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
