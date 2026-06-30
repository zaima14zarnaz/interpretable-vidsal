"""Shape and finiteness checks for ConceptGatedMultiScaleSaliencyDecoder and model forward."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from model.model import ExplainableVidSalModel
from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder

# Video Swin-T-like channel counts per stage.
STAGE_CONFIG: Dict[str, Tuple[int, int, int]] = {
    "stage1": (28, 28, 96),
    "stage2": (28, 28, 192),
    "stage3": (14, 14, 384),
    "stage4": (7, 7, 768),
}


def _make_fake_concept_out(
    *,
    B: int,
    T: int,
    H: int,
    W: int,
    C: int,
    concept_dim: int,
    num_trajectories: int,
    device: torch.device,
) -> Dict[str, Any]:
    N = H * W
    last_t = T - 2

    batch_idx = torch.randint(0, B, (num_trajectories,), device=device)
    target_idx = torch.randint(0, N, (num_trajectories,), device=device)
    source_idx = torch.randint(0, N, (num_trajectories,), device=device)
    time_idx = torch.full((num_trajectories,), last_t, device=device, dtype=torch.long)

    metadata = {
        "batch_idx": batch_idx,
        "time_idx": time_idx,
        "source_idx": source_idx,
        "target_idx": target_idx,
        "source_coords": torch.randn(num_trajectories, 2, device=device),
        "target_coords": torch.randn(num_trajectories, 2, device=device),
        "alpha": torch.rand(num_trajectories, device=device),
        "affinity_logit": torch.randn(num_trajectories, device=device),
        "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
    }

    return {
        "concept_representation": torch.randn(num_trajectories, concept_dim, device=device),
        "visual_concept_representation": torch.randn(B * T * H * W, concept_dim, device=device),
        "visual_metadata": {
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
        },
        "metadata": metadata,
    }


def _make_fake_decoder_inputs(
    *,
    B: int = 2,
    T: int = 4,
    concept_dim: int = 256,
    num_trajectories: int = 64,
    device: torch.device | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, int]]:
    if device is None:
        device = torch.device("cpu")

    concept_outs: Dict[str, Dict[str, Any]] = {}
    features_dict: Dict[str, torch.Tensor] = {}
    stage_channels: Dict[str, int] = {}

    for stage, (H, W, C) in STAGE_CONFIG.items():
        stage_channels[stage] = C
        concept_outs[stage] = _make_fake_concept_out(
            B=B,
            T=T,
            H=H,
            W=W,
            C=C,
            concept_dim=concept_dim,
            num_trajectories=num_trajectories,
            device=device,
        )
        features_dict[stage] = torch.randn(B, C, T, H, W, device=device)

    return concept_outs, features_dict, stage_channels


def test_decoder_output_shapes(
    *,
    B: int = 2,
    T: int = 4,
    output_size: Tuple[int, int] = (224, 224),
    concept_dim: int = 256,
    decoder_channels: int = 128,
) -> None:
    """Run decoder with fake multi-stage inputs and assert output shapes/finiteness."""
    device = torch.device("cpu")
    concept_outs, features_dict, stage_channels = _make_fake_decoder_inputs(
        B=B,
        T=T,
        concept_dim=concept_dim,
        device=device,
    )

    decoder = ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels=stage_channels,
        concept_dim=concept_dim,
        decoder_channels=decoder_channels,
        feature_residual_scale=0.25,
        dropout=0.1,
        output_activation="sigmoid",
    ).to(device)

    out = decoder(
        concept_outs=concept_outs,
        features_dict=features_dict,
        output_size=output_size,
        return_details=True,
    )

    assert out["saliency_map"].shape == (B, 1, *output_size), (
        f"expected saliency_map {(B, 1, *output_size)}, got {tuple(out['saliency_map'].shape)}"
    )
    assert out["saliency_logits"].shape == (B, 1, *output_size), (
        f"expected saliency_logits {(B, 1, *output_size)}, got {tuple(out['saliency_logits'].shape)}"
    )

    stage_gates = out["stage_gates"]
    assert len(stage_gates) == len(STAGE_CONFIG)
    for stage, gate in stage_gates.items():
        assert torch.isfinite(gate).all(), f"non-finite values in stage gate: {stage}"

    saliency_map = out["saliency_map"]
    assert not torch.isnan(saliency_map).any(), "saliency_map contains NaN"
    assert not torch.isinf(saliency_map).any(), "saliency_map contains Inf"


def test_full_model_forward(
    *,
    B: int = 1,
    T: int = 4,
    height: int = 224,
    width: int = 224,
) -> None:
    """Run ExplainableVidSalModel forward and verify output shapes and detail keys."""
    torch.manual_seed(0)
    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=False,
        freeze_backbone=True,
        resize_to=(height, width),
        concept_dim=64,
        num_concepts=8,
        concept_hidden_dim=64,
        saliency_hidden_dim=64,
        top_k=3,
        max_source_patches=16,
        return_details=True,
        visual_concept_on=True,
        temporal_concepts_on=True,
    )
    model.eval()

    video = torch.rand(B, T, 3, height, width)

    with torch.no_grad():
        saliency_map = model(video, return_details=False)
        details = model(video, return_details=True)

    assert saliency_map.shape == (B, 1, height, width), (
        f"expected {(B, 1, height, width)}, got {tuple(saliency_map.shape)}"
    )
    assert not torch.isnan(saliency_map).any(), "model saliency_map contains NaN"
    assert not torch.isinf(saliency_map).any(), "model saliency_map contains Inf"

    assert "concept_out" in details, "return_details=True must include concept_out"
    assert "prediction_out" in details, "return_details=True must include prediction_out"
    assert details["saliency_map"].shape == (B, 1, height, width)
    assert details["saliency_logits"].shape == (B, 1, height, width)
    assert set(details["concept_out"].keys()) == set(model.backbone_stages)


def test_decoder_diagnostics_do_not_change_output(
    *,
    B: int = 2,
    T: int = 4,
    output_size: Tuple[int, int] = (224, 224),
    concept_dim: int = 256,
    decoder_channels: int = 128,
) -> None:
    """return_details=True must not change saliency outputs used for training."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    concept_outs, features_dict, stage_channels = _make_fake_decoder_inputs(
        B=B,
        T=T,
        concept_dim=concept_dim,
        device=device,
    )

    decoder = ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels=stage_channels,
        concept_dim=concept_dim,
        decoder_channels=decoder_channels,
        dropout=0.0,
        output_activation="sigmoid",
    ).to(device)
    decoder.eval()

    with torch.no_grad():
        out_core = decoder(
            concept_outs=concept_outs,
            features_dict=features_dict,
            output_size=output_size,
            return_details=False,
        )
        out_diag = decoder(
            concept_outs=concept_outs,
            features_dict=features_dict,
            output_size=output_size,
            return_details=True,
        )

    for key in ("saliency_map", "saliency_logits", "patch_saliency_logits"):
        torch.testing.assert_close(
            out_core[key],
            out_diag[key],
            msg=f"diagnostic branch changed {key}",
        )
    assert "stage_gates" in out_diag


def main() -> None:
    test_decoder_output_shapes()
    print("decoder shape/finiteness checks: OK")

    test_decoder_diagnostics_do_not_change_output()
    print("decoder diagnostic invariance: OK")

    test_full_model_forward()
    print("full model forward checks: OK")


if __name__ == "__main__":
    main()
