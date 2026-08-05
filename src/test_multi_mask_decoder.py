"""Synthetic checks for multi-mask decoder outputs, diagnostics, and gradients."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from losses import compute_mask_diversity_loss
from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder

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
    top_k: int = 10,
    device: torch.device,
    include_motion: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "active_visual_prototypes": torch.randn(B, top_k, concept_dim, device=device),
        "visual_validity_mask": torch.ones(B, top_k, dtype=torch.bool, device=device),
        "visual_concept_representation": torch.randn(
            B * T * H * W,
            concept_dim,
            device=device,
        ),
        "visual_metadata": {
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
        },
    }
    if include_motion:
        out["active_motion_prototypes"] = torch.randn(
            B,
            top_k,
            concept_dim,
            device=device,
        )
        out["motion_validity_mask"] = torch.ones(
            B,
            top_k,
            dtype=torch.bool,
            device=device,
        )
        out["motion_concept_representation"] = torch.randn(
            B * H * W,
            concept_dim,
            device=device,
        )
        out["motion_metadata"] = {
            "feature_shape": {"B": B, "C": C, "T": T, "H": H, "W": W},
        }
    return out


def _make_fake_decoder_inputs(
    *,
    B: int = 2,
    T: int = 4,
    concept_dim: int = 64,
    device: torch.device | None = None,
    stage3_motion: bool = False,
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
            device=device,
            include_motion=(stage3_motion and stage == "stage3"),
        )
        features_dict[stage] = torch.randn(B, C, T, H, W, device=device)

    return concept_outs, features_dict, stage_channels


def _build_decoder(
    stage_channels: Dict[str, int],
    *,
    concept_dim: int = 64,
    decoder_channels: int = 32,
    device: torch.device,
) -> ConceptGatedMultiScaleSaliencyDecoder:
    return ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels=stage_channels,
        concept_dim=concept_dim,
        decoder_channels=decoder_channels,
        dropout=0.0,
        output_activation="sigmoid",
        num_masks=4,
        motion_concept_stages=("stage3",),
    ).to(device)


def test_multi_mask_shapes_and_gates() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, T = 2, 4
    output_size = (224, 224)
    concept_dim = 64

    concept_outs, features_dict, stage_channels = _make_fake_decoder_inputs(
        B=B,
        T=T,
        concept_dim=concept_dim,
        device=device,
    )
    decoder = _build_decoder(stage_channels, concept_dim=concept_dim, device=device)
    decoder.eval()

    with torch.no_grad():
        out = decoder(
            concept_outs=concept_outs,
            features_dict=features_dict,
            output_size=output_size,
            return_details=True,
        )

    assert out["saliency_map"].shape == (B, 1, *output_size)
    assert out["saliency_logits"].shape == (B, 1, *output_size)
    assert torch.is_tensor(out["mask_diversity_loss"])
    assert out["mask_diversity_loss"].ndim == 0
    assert torch.isfinite(out["mask_diversity_loss"])

    for stage in STAGE_CONFIG:
        multi_masks = out["stage_multi_masks"][stage]
        multi_gates = out["stage_multi_mask_gates"][stage]
        H, W, _ = STAGE_CONFIG[stage]

        assert multi_masks.shape == (B, 4, T, H, W), stage
        assert multi_gates.shape == (B, 4, T, H, W), stage
        assert float(multi_gates.min()) >= 0.0
        assert float(multi_gates.max()) <= 1.0

        diag = out["stage_mask_diagnostics"][stage]
        assert "between_mask_variance" in diag
        assert "per_mask_spatial_variance" in diag
        assert "mean_pairwise_mask_correlation" in diag
        assert "mask_strength" in diag
        for mask_idx in range(1, 5):
            assert f"mask_{mask_idx}_mean" in diag
            assert f"mask_{mask_idx}_std" in diag
            assert f"mask_{mask_idx}_min" in diag
            assert f"mask_{mask_idx}_max" in diag


def test_stage3_visual_and_motion_masks() -> None:
    torch.manual_seed(1)
    device = torch.device("cpu")
    B, T = 2, 4
    output_size = (112, 112)
    concept_dim = 64

    concept_outs, features_dict, stage_channels = _make_fake_decoder_inputs(
        B=B,
        T=T,
        concept_dim=concept_dim,
        device=device,
        stage3_motion=True,
    )
    decoder = _build_decoder(stage_channels, concept_dim=concept_dim, device=device)
    decoder.eval()

    with torch.no_grad():
        out = decoder(
            concept_outs=concept_outs,
            features_dict=features_dict,
            output_size=output_size,
            return_details=True,
        )

    labels = out["stage_mask_modality_labels"]["stage3"]
    assert labels.tolist() == [0, 0, 1, 1]
    assert "stage3" in out["stage_motion_concept_maps"]
    assert "stage3" in out["stage_motion_concept_weights"]
    assert out["stage_visual_concept_weights"]["stage3"].shape[1] == 2
    assert out["stage_motion_concept_weights"]["stage3"].shape[1] == 2


def test_mask_diversity_loss_ordering() -> None:
    identical = torch.ones(2, 4, 3, 8, 8)
    torch.manual_seed(0)
    different = torch.randn(2, 4, 3, 8, 8)

    loss_identical = compute_mask_diversity_loss(identical)
    loss_different = compute_mask_diversity_loss(different)
    assert torch.isfinite(loss_identical)
    assert torch.isfinite(loss_different)
    assert loss_identical.item() > loss_different.item()


def test_mask_diversity_gradients_reach_fusion_modules() -> None:
    torch.manual_seed(2)
    device = torch.device("cpu")
    B, T = 1, 3
    output_size = (56, 56)
    concept_dim = 32

    concept_outs, features_dict, stage_channels = _make_fake_decoder_inputs(
        B=B,
        T=T,
        concept_dim=concept_dim,
        device=device,
    )
    decoder = _build_decoder(
        stage_channels,
        concept_dim=concept_dim,
        decoder_channels=24,
        device=device,
    )
    decoder.train()
    decoder.zero_grad(set_to_none=True)

    out = decoder(
        concept_outs=concept_outs,
        features_dict=features_dict,
        output_size=output_size,
        return_details=False,
    )
    loss = out["mask_diversity_loss"] + out["saliency_map"].mean()
    assert out["mask_diversity_loss"].ndim == 0
    loss.backward()

    block = decoder.fusion_blocks["stage1"]
    assert block.feature_proj.conv.weight.grad is not None
    assert block.concept_proj.conv.weight.grad is not None
    assert block.visual_mask_queries.grad is not None
    for branch in block.mask_feature_branches:
        assert branch.conv.weight.grad is not None
    assert block.mask_fusion.weight.grad is not None


def main() -> None:
    test_multi_mask_shapes_and_gates()
    print("multi-mask shapes/gates/diagnostics: OK")

    test_stage3_visual_and_motion_masks()
    print("stage3 visual+motion masks: OK")

    test_mask_diversity_loss_ordering()
    print("mask diversity loss ordering: OK")

    test_mask_diversity_gradients_reach_fusion_modules()
    print("mask diversity gradients: OK")


if __name__ == "__main__":
    main()
