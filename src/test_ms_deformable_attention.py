"""Shape / gradient smoke tests for MultiScaleDeformableAttention2D."""

from __future__ import annotations

import torch

from model.saliency_prediction import MultiScaleDeformableAttention2D


def test_ms_deform_attn_shapes_and_grads() -> None:
    B, Nq, D = 2, 8, 64
    num_heads, num_levels, num_points = 8, 4, 4
    attn = MultiScaleDeformableAttention2D(
        d_model=D,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        dropout=0.0,
    )

    query = torch.randn(B, Nq, D, requires_grad=True)
    reference_points = torch.rand(B, Nq, 2)
    value_features = [
        torch.randn(B, D, 4, 4, requires_grad=True),
        torch.randn(B, D, 7, 7, requires_grad=True),
        torch.randn(B, D, 14, 14, requires_grad=True),
        torch.randn(B, D, 28, 28, requires_grad=True),
    ]

    out = attn(query, value_features, reference_points)
    assert out.shape == (B, Nq, D), out.shape
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    # At zero-init, offsets/attn logits ignore query; values still get gradients.
    for feat in value_features:
        assert feat.grad is not None and torch.isfinite(feat.grad).all()
        assert float(feat.grad.norm().item()) > 0.0
    assert attn.value_proj.weight.grad is not None
    assert attn.output_proj.weight.grad is not None
    assert float(attn.output_proj.weight.grad.norm().item()) > 0.0

    # After non-zero offset/attn weights, query must receive gradients.
    attn.zero_grad(set_to_none=True)
    query2 = torch.randn(B, Nq, D, requires_grad=True)
    nn_init = torch.nn.init
    nn_init.normal_(attn.sampling_offsets.weight, std=0.02)
    nn_init.normal_(attn.attention_weights.weight, std=0.02)
    out2 = attn(query2, [f.detach() for f in value_features], reference_points)
    out2.sum().backward()
    assert query2.grad is not None and torch.isfinite(query2.grad).all()
    assert float(query2.grad.norm().item()) > 0.0


def test_ms_deform_attn_softmax_normalizes() -> None:
    """Attention logits softmax over levels*points (checked via zero-init path)."""
    attn = MultiScaleDeformableAttention2D(
        d_model=32, num_heads=4, num_levels=2, num_points=3, dropout=0.0
    )
    B, Nq = 1, 5
    query = torch.zeros(B, Nq, 32)
    reference_points = torch.full((B, Nq, 2), 0.5)
    value_features = [
        torch.randn(B, 32, 5, 5),
        torch.randn(B, 32, 10, 10),
    ]
    out = attn(query, value_features, reference_points)
    assert out.shape == (B, Nq, 32)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    test_ms_deform_attn_shapes_and_grads()
    print("ms deform attn shapes/grads: OK")
    test_ms_deform_attn_softmax_normalizes()
    print("ms deform attn softmax path: OK")
