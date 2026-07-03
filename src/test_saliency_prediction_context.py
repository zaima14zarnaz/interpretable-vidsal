"""Sanity check: visual-only patch_concept_context in SaliencyPrediction."""

from __future__ import annotations

import torch

from model.saliency_prediction import SaliencyPrediction


def _make_concept_out(
    *,
    B: int,
    T: int,
    H: int,
    W: int,
    concept_dim: int,
    num_concepts: int,
    M: int,
) -> dict:
    N = H * W
    last_t = T - 2
    device = torch.device("cpu")

    batch_idx = torch.randint(0, B, (M,), device=device)
    target_idx = torch.randint(0, N, (M,), device=device)
    source_idx = torch.randint(0, N, (M,), device=device)
    time_idx = torch.full((M,), last_t, device=device, dtype=torch.long)

    metadata = {
        "batch_idx": batch_idx,
        "time_idx": time_idx,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "source_coords": torch.randn(M, 2, device=device),
        "target_coords": torch.randn(M, 2, device=device),
        "alpha": torch.rand(M, device=device),
        "affinity_logit": torch.randn(M, device=device),
        "feature_shape": {"B": B, "C": 128, "T": T, "H": H, "W": W},
        "last_transition_only": True,
    }

    return {
        "concept_representation": torch.randn(M, concept_dim, device=device),
        "transition_activations": torch.softmax(
            torch.randn(M, num_concepts, device=device), dim=-1
        ),
        "persistence_activations": torch.softmax(
            torch.randn(M, num_concepts, device=device), dim=-1
        ),
        "gate_probs": torch.softmax(torch.randn(M, 2, device=device), dim=-1),
        "visual_concept_representation": torch.randn(
            B * T * H * W, concept_dim, device=device
        ),
        "visual_metadata": {
            "feature_shape": {"B": B, "C": 128, "T": T, "H": H, "W": W},
        },
        "metadata": metadata,
    }


def main() -> None:
    B, T, H, W = 2, 4, 7, 7
    concept_dim = 256
    num_concepts = 32
    M = 64
    last_rgb = torch.rand(B, 3, 224, 384)

    model = SaliencyPrediction(
        concept_dim=concept_dim,
        hidden_dim=64,
        feature_channels=128,
        use_gated_trajectory_head=False,
        use_feature_refinement=False,
        use_subpatch_head=True,
        subpatch_factor=4,
    )

    out = model(
        _make_concept_out(
            B=B,
            T=T,
            H=H,
            W=W,
            concept_dim=concept_dim,
            num_concepts=num_concepts,
            M=M,
        ),
        last_rgb,
        return_details=True,
    )

    assert tuple(out["patch_concept_context"].shape) == (B, concept_dim, H, W)
    for key in (
        "temporal_saliency_map",
        "patch_saliency_logits",
        "coarse_patch_logits",
        "final_patch_logits",
        "subpatch_logits",
        "patch_concept_context",
    ):
        assert key in out, f"missing output key: {key}"

    print("patch_concept_context:", tuple(out["patch_concept_context"].shape))
    print("coarse_patch_logits:", tuple(out["coarse_patch_logits"].shape))
    print("subpatch_logits:", tuple(out["subpatch_logits"].shape))
    print("OK")


if __name__ == "__main__":
    main()
