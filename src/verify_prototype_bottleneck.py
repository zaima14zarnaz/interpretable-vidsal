#!/usr/bin/env python3
"""Lightweight checks for the partial concept bottleneck fusion."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model.saliency_prediction as saliency_prediction
from model.saliency_prediction import (
    ConceptGatedMultiScaleSaliencyDecoder,
    SpatioTemporalConceptGatedFusionBlock,
)


def _build_block(
    prototype_bottleneck_strength: float,
    *,
    feature_channels: int = 32,
    concept_dim: int = 16,
    decoder_channels: int = 24,
    prototype_application_position: str = "pre_refine",
) -> SpatioTemporalConceptGatedFusionBlock:
    return SpatioTemporalConceptGatedFusionBlock(
        feature_channels=feature_channels,
        concept_dim=concept_dim,
        decoder_channels=decoder_channels,
        enable_patch_prioritization=True,
        use_shared_concept_activations=False,
        prototype_bottleneck_strength=prototype_bottleneck_strength,
        prototype_application_position=prototype_application_position,
        factorized_rank=8,
        pos_num_frequencies=2,
    )


def _synthetic_inputs(
    block: SpatioTemporalConceptGatedFusionBlock,
    *,
    batch: int = 2,
    temporal: int = 4,
    height: int = 3,
    width: int = 4,
    num_concepts: int = 3,
):
    device = next(block.parameters()).device
    dtype = torch.float32
    C = block.feature_proj.conv.in_channels
    D = block.concept_dim
    features = torch.randn(batch, C, temporal, height, width, device=device, dtype=dtype)
    concept_volume = torch.randn(
        batch, D, temporal, height, width, device=device, dtype=dtype
    )
    active_visual_prototypes = torch.randn(
        batch, 1, height, width, num_concepts, D, device=device, dtype=dtype
    )
    visual_validity_mask = torch.ones(
        batch, 1, height, width, num_concepts, device=device, dtype=torch.bool
    )
    return features, concept_volume, active_visual_prototypes, visual_validity_mask


def _run_forward_with_capture(
    block: SpatioTemporalConceptGatedFusionBlock,
    features: torch.Tensor,
    concept_volume: torch.Tensor,
    active_visual_prototypes: torch.Tensor,
    visual_validity_mask: torch.Tensor,
):
    captured: dict = {}
    original_scatter = saliency_prediction._scatter_last_frame_update
    original_build = block._build_concept_conditioned_update

    def capturing_scatter(base, update):
        captured["film_features"] = base
        captured["guided_last"] = update
        result = original_scatter(base, update)
        captured["bottleneck_fused"] = result
        return result

    def capturing_build(*args, **kwargs):
        weights, context, update_last = original_build(*args, **kwargs)
        captured["concept_conditioned_update_last"] = update_last
        return weights, context, update_last

    saliency_prediction._scatter_last_frame_update = capturing_scatter
    block._build_concept_conditioned_update = capturing_build
    try:
        fused, mask_outputs = block(
            features,
            concept_volume,
            prev_decoder=None,
            active_visual_prototypes=active_visual_prototypes,
            visual_validity_mask=visual_validity_mask,
        )
    finally:
        saliency_prediction._scatter_last_frame_update = original_scatter
        block._build_concept_conditioned_update = original_build

    priority_mask = mask_outputs["priority_mask"]
    captured["priority_mask_last"] = priority_mask[:, :, -1:, :, :]
    captured["film_last"] = captured["film_features"][:, :, -1:, :, :]
    captured["prototype_branch_last"] = (
        captured["priority_mask_last"] * captured["concept_conditioned_update_last"]
    )
    return fused, mask_outputs, captured


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    max_diff = float((actual.detach() - expected.detach()).abs().max().cpu())
    if max_diff > atol:
        raise AssertionError(f"{name}: max abs diff {max_diff} exceeds atol={atol}")
    print(f"  PASS: {name}")


def test_strength_boundaries() -> None:
    print("=== boundary / temporal-slice checks ===")
    for strength in (0.0, 0.4, 1.0):
        print(f"-- prototype_bottleneck_strength={strength}")
        block = _build_block(strength)
        block.train()
        features, concept_volume, protos, validity = _synthetic_inputs(block)
        fused, mask_outputs, captured = _run_forward_with_capture(
            block, features, concept_volume, protos, validity
        )

        film_features = captured["film_features"]
        film_last = captured["film_last"]
        guided_last = captured["guided_last"]
        prototype_branch_last = captured["prototype_branch_last"]
        bottleneck_fused = captured["bottleneck_fused"]

        if fused.shape != film_features.shape:
            raise AssertionError(
                f"decoded shape {tuple(fused.shape)} != "
                f"film shape {tuple(film_features.shape)}"
            )
        print(f"  PASS: output shape {tuple(fused.shape)}")

        _assert_close(
            "earlier temporal slices unchanged before refine",
            bottleneck_fused[:, :, :-1],
            film_features[:, :, :-1],
        )
        _assert_close(
            "last temporal slice equals guided_last before refine",
            bottleneck_fused[:, :, -1:],
            guided_last,
        )

        expected_guided = (1.0 - strength) * film_last + strength * prototype_branch_last
        _assert_close("guided_last matches convex blend", guided_last, expected_guided)

        if strength == 0.0:
            _assert_close("strength=0 -> film_last", guided_last, film_last)
        if strength == 1.0:
            _assert_close(
                "strength=1 -> priority-gated prototype branch",
                guided_last,
                prototype_branch_last,
            )

        reported = float(mask_outputs["prototype_bottleneck_strength"].cpu())
        if abs(reported - strength) > 1e-8:
            raise AssertionError(
                f"mask_outputs strength {reported} != configured {strength}"
            )
        print("  PASS: mask_outputs reports prototype_bottleneck_strength")


def test_gradients() -> None:
    print("=== gradient checks (strength=0.4) ===")
    block = _build_block(0.4)
    block.train()
    features, concept_volume, protos, validity = _synthetic_inputs(block)
    protos = protos.clone().detach().requires_grad_(True)

    fused, _, _ = _run_forward_with_capture(
        block, features, concept_volume, protos, validity
    )
    loss = fused.square().mean()
    loss.backward()

    named = {
        "priority_application_proto_proj": block.priority_application_proto_proj,
        "mask_feature_branch": block.mask_feature_branch,
        "unary_priority_mlp": block.unary_priority_mlp,
        "factor_query": block.factor_query,
        "factor_key": block.factor_key,
    }
    for name, module in named.items():
        if module is None:
            continue
        grad_norm = 0.0
        found = False
        for param in module.parameters():
            if param.grad is not None:
                found = True
                grad_norm += float(param.grad.detach().norm().cpu())
        if not found or grad_norm <= 0.0:
            raise AssertionError(f"{name}: expected nonzero gradients, got {grad_norm}")
        print(f"  PASS: nonzero grad for {name} (norm={grad_norm:.6f})")

    if protos.grad is None or float(protos.grad.detach().norm().cpu()) <= 0.0:
        raise AssertionError("prototype inputs did not receive nonzero gradients")
    print(
        "  PASS: nonzero grad for active_visual_prototypes "
        f"(norm={float(protos.grad.detach().norm().cpu()):.6f})"
    )


def test_checkpoint_compatibility() -> None:
    print("=== checkpoint compatibility (raw_strength retained) ===")
    block = _build_block(0.4)
    state = copy.deepcopy(block.state_dict())
    if "raw_strength" not in state:
        raise AssertionError("raw_strength missing from state_dict")
    # Simulate an older checkpoint that still stores raw_strength.
    state["raw_strength"].fill_(-1.5)

    reloaded = _build_block(0.7)
    missing, unexpected = reloaded.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise AssertionError(
            f"strict load failed: missing={missing}, unexpected={unexpected}"
        )
    if not torch.allclose(reloaded.raw_strength, state["raw_strength"]):
        raise AssertionError("raw_strength did not load from checkpoint")
    if abs(reloaded.prototype_bottleneck_strength - 0.7) > 1e-8:
        raise AssertionError(
            "prototype_bottleneck_strength should stay at constructor value, "
            "not be inferred from raw_strength"
        )
    print("  PASS: strict load keeps raw_strength and ignores it for fusion strength")


def test_decoder_construction_print_and_validation() -> None:
    print("=== decoder construction / validation ===")
    decoder = ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels={"stage3": 32, "stage4": 48},
        concept_dim=16,
        decoder_channels=24,
        prototype_bottleneck_strength=0.4,
        factorized_rank=8,
    )
    for stage in ("stage3", "stage4"):
        block = decoder.fusion_blocks[stage]
        if abs(block.prototype_bottleneck_strength - 0.4) > 1e-8:
            raise AssertionError(f"{stage} did not receive bottleneck strength")
        if block.prototype_application_position != "pre_refine":
            raise AssertionError(
                f"{stage} default position should be pre_refine, "
                f"got {block.prototype_application_position}"
            )
        if not block.enable_patch_prioritization:
            raise AssertionError(f"{stage} should enable patch prioritization")
    print("  PASS: decoder default is pre_refine and propagates strength")

    post_decoder = ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels={"stage3": 32, "stage4": 48},
        concept_dim=16,
        decoder_channels=24,
        prototype_bottleneck_strength=0.4,
        prototype_application_position="post_refine",
        factorized_rank=8,
    )
    for stage in ("stage3", "stage4"):
        if (
            post_decoder.fusion_blocks[stage].prototype_application_position
            != "post_refine"
        ):
            raise AssertionError(f"{stage} did not receive post_refine")
    print("  PASS: decoder propagates post_refine placement")

    try:
        SpatioTemporalConceptGatedFusionBlock(
            feature_channels=8,
            concept_dim=8,
            decoder_channels=8,
            enable_patch_prioritization=True,
            prototype_bottleneck_strength=1.5,
        )
        raise AssertionError("expected ValueError for strength outside [0, 1]")
    except ValueError as exc:
        if "prototype_bottleneck_strength" not in str(exc):
            raise
        print("  PASS: invalid strength raises ValueError")


def test_post_refine_placement() -> None:
    print("=== post_refine placement checks ===")
    block = _build_block(1.0, prototype_application_position="post_refine")
    block.train()
    features, concept_volume, protos, validity = _synthetic_inputs(block)

    # Capture decoded_base by wrapping refine2 output path via scatter hook.
    captured: dict = {}
    original_scatter = saliency_prediction._scatter_last_frame_update
    original_build = block._build_concept_conditioned_update
    original_refine2 = block.refine2.forward

    def capturing_refine2(x):
        out = original_refine2(x)
        captured["decoded_base"] = out
        return out

    def capturing_scatter(base, update):
        captured["scatter_base"] = base
        captured["guided_last"] = update
        return original_scatter(base, update)

    def capturing_build(*args, **kwargs):
        weights, context, update_last = original_build(*args, **kwargs)
        captured["base_features"] = args[0]
        captured["concept_conditioned_update_last"] = update_last
        return weights, context, update_last

    saliency_prediction._scatter_last_frame_update = capturing_scatter
    block._build_concept_conditioned_update = capturing_build
    block.refine2.forward = capturing_refine2
    try:
        decoded, mask_outputs = block(
            features,
            concept_volume,
            prev_decoder=None,
            active_visual_prototypes=protos,
            visual_validity_mask=validity,
        )
    finally:
        saliency_prediction._scatter_last_frame_update = original_scatter
        block._build_concept_conditioned_update = original_build
        block.refine2.forward = original_refine2

    decoded_base = captured["decoded_base"]
    guided_last = captured["guided_last"]
    priority_mask_last = mask_outputs["priority_mask"][:, :, -1:, :, :]
    prototype_branch_last = (
        priority_mask_last * captured["concept_conditioned_update_last"]
    )

    _assert_close(
        "post_refine earlier slices unchanged",
        decoded[:, :, :-1],
        decoded_base[:, :, :-1],
    )
    _assert_close(
        "post_refine last slice equals guided_last",
        decoded[:, :, -1:],
        guided_last,
    )
    _assert_close(
        "post_refine strength=1 uses prototype branch vs decoded_base_last",
        guided_last,
        prototype_branch_last,
    )
    _assert_close(
        "post_refine builds update from decoded_base_last",
        captured["base_features"],
        decoded_base[:, :, -1:, :, :],
    )
    if mask_outputs.get("prototype_application_position") != "post_refine":
        raise AssertionError("mask_outputs missing post_refine position")
    if "decoded_base_last_norm" not in mask_outputs:
        raise AssertionError("decoded_base_last_norm missing")
    if "prototype_to_base_norm_ratio" not in mask_outputs:
        raise AssertionError("prototype_to_base_norm_ratio missing")
    print("  PASS: post_refine diagnostics present")


def main() -> None:
    torch.manual_seed(0)
    test_strength_boundaries()
    test_gradients()
    test_checkpoint_compatibility()
    test_decoder_construction_print_and_validation()
    test_post_refine_placement()
    print("\nAll prototype-bottleneck verification checks passed.")


if __name__ == "__main__":
    main()
