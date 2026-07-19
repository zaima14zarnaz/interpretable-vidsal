"""Independent tests for MaskPredictionHead."""

from __future__ import annotations

import torch

from model.saliency_prediction import MaskPredictionHead


def test_mask_prediction_head_shapes_and_grads() -> None:
    B, Q, D, H, W = 2, 8, 64, 28, 28
    H_out, W_out = 224, 384
    head = MaskPredictionHead(d_model=D)
    queries = torch.randn(B, Q, D, requires_grad=True)
    mask_features = torch.randn(B, D, H, W, requires_grad=True)

    out = head(queries, mask_features, output_size=(H_out, W_out))
    saliency_logits = out["saliency_logits"]
    per_query = out["per_query_mask_logits"]
    weights = out["query_weights"]

    assert saliency_logits.shape == (B, 1, H_out, W_out), saliency_logits.shape
    assert per_query.shape == (B, Q, H, W), per_query.shape
    assert weights.shape == (B, Q), weights.shape
    assert torch.allclose(weights.sum(dim=1), torch.ones(B), atol=1e-5)
    assert torch.isfinite(saliency_logits).all()

    # Aggregation at pixel res should match einsum reconstruction.
    pixel_agg = torch.einsum("bq,bqhw->bhw", weights, per_query).unsqueeze(1)
    up = torch.nn.functional.interpolate(
        pixel_agg, size=(H_out, W_out), mode="bilinear", align_corners=False
    )
    torch.testing.assert_close(saliency_logits, up, rtol=1e-4, atol=1e-4)

    loss = saliency_logits.sum()
    loss.backward()
    assert head.mask_embed.layers[0].weight.grad is not None
    assert head.pixel_proj.weight.grad is not None
    assert head.query_weight_head.weight.grad is not None
    assert queries.grad is not None and queries.grad.abs().sum() > 0
    assert mask_features.grad is not None and mask_features.grad.abs().sum() > 0


def test_mask_prediction_head_no_upsample() -> None:
    head = MaskPredictionHead(d_model=32)
    out = head(torch.randn(1, 4, 32), torch.randn(1, 32, 7, 9))
    assert out["saliency_logits"].shape == (1, 1, 7, 9)
    assert out["per_query_mask_logits"].shape == (1, 4, 7, 9)


def main() -> None:
    test_mask_prediction_head_shapes_and_grads()
    print("MaskPredictionHead shapes/grads: OK")
    test_mask_prediction_head_no_upsample()
    print("MaskPredictionHead no-upsample: OK")


if __name__ == "__main__":
    main()
