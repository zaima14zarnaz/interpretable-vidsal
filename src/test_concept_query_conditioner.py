"""Independent tests for ConceptQueryConditioner."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from model.saliency_prediction import ConceptQueryConditioner

STAGE_CONFIG: Dict[str, Tuple[int, int, int]] = {
    "stage1": (28, 28, 96),
    "stage2": (14, 14, 192),
    "stage3": (7, 7, 384),
    "stage4": (4, 4, 768),
}


def _make_inputs(
    *,
    B: int = 2,
    T: int = 4,
    concept_dim: int = 64,
    with_motion: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
    concept_outs: Dict[str, Dict[str, Any]] = {}
    features: Dict[str, torch.Tensor] = {}
    for stage, (H, W, C) in STAGE_CONFIG.items():
        features[stage] = torch.randn(B, C, T, H, W)
        N = B * T * H * W
        concept_outs[stage] = {
            "visual_concept_representation": torch.randn(N, concept_dim),
            "visual_metadata": {
                "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W}
            },
        }
        if with_motion and stage == "stage3":
            concept_outs[stage]["motion_concept_representation"] = torch.randn(
                B * H * W, concept_dim
            )
            concept_outs[stage]["motion_metadata"] = {
                "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W}
            }
    return concept_outs, features


def test_concept_query_conditioner_shapes_and_grads() -> None:
    B, Q, D, Cd = 2, 8, 64, 64
    stages = ("stage4", "stage3", "stage2", "stage1")
    conditioner = ConceptQueryConditioner(
        concept_dim=Cd,
        decoder_channels=D,
        stages=stages,
        motion_concept_stages=("stage3",),
        stage_pool="attn",
    )
    concept_outs, features = _make_inputs(B=B, concept_dim=Cd, with_motion=True)
    learned = torch.randn(B, Q, D, requires_grad=True)

    out = conditioner(
        concept_outs=concept_outs,
        features_dict=features,
        stages=list(stages),
        learned_queries=learned,
    )

    assert out["conditioned_queries"].shape == (B, Q, D)
    assert out["global_concept_token"].shape == (B, D)
    assert out["stage_concept_tokens"].shape == (B, 4, D)
    assert out["stage_pool_weights"] is not None
    assert out["stage_pool_weights"].shape == (B, 4)
    assert "stage3" in out["stage_motion_concept_volumes"]
    # Zero-init concept_to_query => conditioned == learned at init.
    torch.testing.assert_close(out["conditioned_queries"], learned)

    loss = out["conditioned_queries"].sum()
    loss.backward()
    assert conditioner.concept_to_query.weight.grad is not None
    # After backward through zero-init linear, grads to concept path still exist
    # via stage projs when we add a non-zero path: force by using global token.
    loss2 = out["global_concept_token"].sum()
    # need fresh forward for clean graph
    learned2 = torch.randn(B, Q, D, requires_grad=True)
    out2 = conditioner(concept_outs, features, list(stages), learned2)
    (out2["global_concept_token"].sum() + out2["conditioned_queries"].sum()).backward()
    assert conditioner.stage_projs["stage1"][0].weight.grad is not None
    assert conditioner.motion_concept_logit_scales["stage3"].grad is not None


def test_concept_query_conditioner_mean_pool_no_motion() -> None:
    conditioner = ConceptQueryConditioner(
        concept_dim=32,
        decoder_channels=48,
        stages=("stage4", "stage3", "stage2", "stage1"),
        stage_pool="mean",
    )
    concept_outs, features = _make_inputs(B=1, concept_dim=32, with_motion=False)
    out = conditioner(
        concept_outs,
        features,
        ["stage4", "stage3", "stage2", "stage1"],
        torch.zeros(1, 8, 48),
    )
    assert out["stage_pool_weights"] is None
    assert out["stage_motion_concept_volumes"] == {}
    assert out["conditioned_queries"].shape == (1, 8, 48)


def main() -> None:
    test_concept_query_conditioner_shapes_and_grads()
    print("ConceptQueryConditioner shapes/grads: OK")
    test_concept_query_conditioner_mean_pool_no_motion()
    print("ConceptQueryConditioner mean/no-motion: OK")


if __name__ == "__main__":
    main()
