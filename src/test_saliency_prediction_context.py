"""Sanity check: ConceptGatedMultiScaleSaliencyDecoder concept-map construction."""

from __future__ import annotations

import torch

from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder


def _make_concept_out(
    *,
    B: int,
    T: int,
    H: int,
    W: int,
    concept_dim: int,
    num_concepts: int,
    M: int,
    feature_channels: int,
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
        "feature_shape": {"B": B, "C": feature_channels, "T": T, "H": H, "W": W},
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
            "feature_shape": {"B": B, "C": feature_channels, "T": T, "H": H, "W": W},
        },
        "metadata": metadata,
    }


def main() -> None:
    B, T = 2, 4
    concept_dim = 256
    num_concepts = 32
    M = 64
    output_size = (224, 384)

    stage_shapes = {
        "stage4": (7, 7, 768),
        "stage3": (14, 14, 384),
        "stage2": (28, 28, 192),
        "stage1": (28, 28, 96),
    }

    stage_channels = {stage: channels for stage, (_, _, channels) in stage_shapes.items()}
    concept_outs = {}
    features_dict = {}

    for stage, (H, W, C) in stage_shapes.items():
        concept_outs[stage] = _make_concept_out(
            B=B,
            T=T,
            H=H,
            W=W,
            concept_dim=concept_dim,
            num_concepts=num_concepts,
            M=M,
            feature_channels=C,
        )
        features_dict[stage] = torch.randn(B, C, T, H, W)

    model = ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels=stage_channels,
        concept_dim=concept_dim,
        decoder_channels=64,
        output_activation="sigmoid",
    )

    out = model(
        concept_outs=concept_outs,
        features_dict=features_dict,
        output_size=output_size,
        return_details=True,
    )

    assert tuple(out["saliency_logits"].shape) == (B, 1, *output_size)
    assert tuple(out["saliency_map"].shape) == (B, 1, *output_size)
    assert tuple(out["patch_saliency_logits"].shape)[0] == B
    for key in (
        "stage_concept_maps",
        "stage_feature_maps",
        "stage_gates",
        "decoded_stage_features",
    ):
        assert key in out, f"missing output key: {key}"
        assert len(out[key]) == len(stage_shapes)

    print("saliency_logits:", tuple(out["saliency_logits"].shape))
    print("patch_saliency_logits:", tuple(out["patch_saliency_logits"].shape))
    print("stages decoded:", list(out["decoded_stage_features"].keys()))
    print("OK")


if __name__ == "__main__":
    main()
