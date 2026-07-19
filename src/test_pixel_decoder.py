"""Independent shape tests for Mask2Former-style PixelDecoder."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from model.saliency_prediction import PixelDecoder

STAGE_CONFIG: Dict[str, Tuple[int, int, int]] = {
    "stage1": (28, 28, 96),
    "stage2": (14, 14, 192),
    "stage3": (7, 7, 384),
    "stage4": (4, 4, 768),
}


def _make_features(
    *,
    B: int = 2,
    T: int = 4,
) -> Dict[str, torch.Tensor]:
    features: Dict[str, torch.Tensor] = {}
    for stage, (H, W, C) in STAGE_CONFIG.items():
        features[stage] = torch.randn(B, C, T, H, W)
    return features


def _assert_common_output_keys(
    out: dict,
    *,
    B: int,
    T: int,
    decoder_channels: int,
    expect_temporal_weights: bool,
) -> None:
    required = {
        "mask_features",
        "multi_scale_memory",
        "multi_scale_features",
        "multi_scale_hw",
        "multi_scale_stages",
        "temporal_weights",
        "side_temporal_weights",
    }
    assert required.issubset(out.keys()), out.keys()

    mask_features = out["mask_features"]
    H1, W1 = STAGE_CONFIG["stage1"][:2]
    assert mask_features.shape == (B, decoder_channels, H1, W1), mask_features.shape

    stages = out["multi_scale_stages"]
    assert stages == ["stage4", "stage3", "stage2", "stage1"]
    assert len(out["multi_scale_memory"]) == 4
    assert len(out["multi_scale_features"]) == 4
    assert len(out["multi_scale_hw"]) == 4

    for stage, memory, feat, hw in zip(
        stages,
        out["multi_scale_memory"],
        out["multi_scale_features"],
        out["multi_scale_hw"],
    ):
        H, W = STAGE_CONFIG[stage][:2]
        assert hw == (H, W), (stage, hw)
        assert feat.shape == (B, decoder_channels, H, W), (stage, feat.shape)
        assert memory.shape == (B, H * W, decoder_channels), (stage, memory.shape)
        rebuilt = memory.transpose(1, 2).reshape(B, decoder_channels, H, W)
        torch.testing.assert_close(rebuilt, feat)

    if expect_temporal_weights:
        assert out["temporal_weights"] is not None
        assert out["temporal_weights"].shape == (B, 1, T, H1, W1)
        assert set(out["side_temporal_weights"]) == set(stages)
    else:
        assert out["temporal_weights"] is None
        assert out["side_temporal_weights"] == {}


def test_pixel_decoder_shapes_and_fpn(
    *,
    B: int = 2,
    T: int = 4,
    decoder_channels: int = 64,
) -> None:
    stage_channels = {stage: cfg[2] for stage, cfg in STAGE_CONFIG.items()}
    features = _make_features(B=B, T=T)
    decoder = PixelDecoder(
        stage_channels=stage_channels,
        decoder_channels=decoder_channels,
        temporal_aggregation="learned_all_frames",
        dropout=0.0,
        pixel_decoder_type="fpn",
    )
    assert decoder.pixel_decoder_type == "fpn"
    assert decoder.deformable_layers is None
    assert decoder.lateral_convs is not None

    out = decoder(features)
    _assert_common_output_keys(
        out,
        B=B,
        T=T,
        decoder_channels=decoder_channels,
        expect_temporal_weights=True,
    )

    loss = out["mask_features"].sum()
    loss.backward()
    assert decoder.input_projs["stage1"][0].weight.grad is not None
    assert torch.isfinite(decoder.input_projs["stage1"][0].weight.grad).all()
    assert decoder.mask_feature_head.conv.weight.grad is not None


def test_pixel_decoder_shapes_and_deformable(
    *,
    B: int = 2,
    T: int = 2,
    decoder_channels: int = 64,
) -> None:
    stage_channels = {stage: cfg[2] for stage, cfg in STAGE_CONFIG.items()}
    features = _make_features(B=B, T=T)
    decoder = PixelDecoder(
        stage_channels=stage_channels,
        decoder_channels=decoder_channels,
        temporal_aggregation="learned_all_frames",
        dropout=0.0,
        pixel_decoder_type="deformable",
        num_pixel_decoder_layers=3,
    )
    assert decoder.pixel_decoder_type == "deformable"
    assert decoder.deformable_layers is not None
    assert len(decoder.deformable_layers) == 3
    assert decoder.lateral_convs is None

    out = decoder(features)
    _assert_common_output_keys(
        out,
        B=B,
        T=T,
        decoder_channels=decoder_channels,
        expect_temporal_weights=True,
    )
    # stage1 multi-scale map must be the deformable mask_features.
    torch.testing.assert_close(out["multi_scale_features"][-1], out["mask_features"])

    loss = out["mask_features"].sum()
    loss.backward()
    assert decoder.input_projs["stage1"][0].weight.grad is not None
    assert decoder.deformable_layers[0].deform_attn.sampling_offsets.weight.grad is not None


def test_pixel_decoder_temporal_modes() -> None:
    stage_channels = {stage: cfg[2] for stage, cfg in STAGE_CONFIG.items()}
    features = _make_features(B=1, T=3)

    for mode in ("learned_all_frames", "mean", "last"):
        for pixel_decoder_type in ("fpn", "deformable"):
            decoder = PixelDecoder(
                stage_channels=stage_channels,
                decoder_channels=32,
                temporal_aggregation=mode,
                pixel_decoder_type=pixel_decoder_type,
                num_pixel_decoder_layers=1,
            )
            out = decoder(features)
            assert out["mask_features"].shape[1] == 32
            if mode == "learned_all_frames":
                assert out["temporal_weights"] is not None
            else:
                assert out["temporal_weights"] is None
                assert out["side_temporal_weights"] == {}


def test_pixel_decoder_no_concept_parameters() -> None:
    stage_channels = {stage: cfg[2] for stage, cfg in STAGE_CONFIG.items()}
    for pixel_decoder_type in ("fpn", "deformable"):
        decoder = PixelDecoder(
            stage_channels=stage_channels,
            decoder_channels=48,
            pixel_decoder_type=pixel_decoder_type,
            num_pixel_decoder_layers=1,
        )
        for name, _ in decoder.named_parameters():
            assert "concept" not in name.lower(), name


def main() -> None:
    test_pixel_decoder_shapes_and_fpn()
    print("PixelDecoder shapes/FPN/grads: OK")
    test_pixel_decoder_shapes_and_deformable()
    print("PixelDecoder shapes/deformable/grads: OK")
    test_pixel_decoder_temporal_modes()
    print("PixelDecoder temporal modes: OK")
    test_pixel_decoder_no_concept_parameters()
    print("PixelDecoder concept-free: OK")


if __name__ == "__main__":
    main()
