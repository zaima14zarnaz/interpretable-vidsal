"""Independent tests for SaliencyQueryTransformerDecoder."""

from __future__ import annotations

from typing import List

import torch

from model.saliency_prediction import SaliencyQueryTransformerDecoder


def _make_multiscale_memory(
    *,
    B: int = 2,
    D: int = 64,
    hw_list=((4, 4), (7, 7), (14, 14), (28, 28)),
) -> List[torch.Tensor]:
    memories: List[torch.Tensor] = []
    for H, W in hw_list:
        memories.append(torch.randn(B, H * W, D))
    return memories


def test_query_decoder_output_shape_and_grads() -> None:
    B, Q, D = 2, 8, 64
    decoder = SaliencyQueryTransformerDecoder(
        d_model=D,
        num_queries=Q,
        num_layers=4,
        nhead=8,
        dropout=0.0,
        num_feature_levels=4,
        use_concept_cross_attention=False,
    )
    memory = _make_multiscale_memory(B=B, D=D)
    out = decoder(memory)
    query_out = out["query_embeddings"]
    assert query_out.shape == (B, Q, D), query_out.shape
    assert out["concept_query_attention"] is None
    assert torch.isfinite(query_out).all()

    loss = query_out.sum()
    loss.backward()
    assert decoder.query_feat.weight.grad is not None
    assert decoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert torch.isfinite(decoder.query_feat.weight.grad).all()
    gnorm = float(decoder.query_feat.weight.grad.norm().item())
    assert gnorm > 0.0


def test_query_decoder_default_num_queries() -> None:
    decoder = SaliencyQueryTransformerDecoder(d_model=32)
    assert decoder.num_queries == 8
    assert decoder.num_layers == 6
    assert decoder.use_concept_cross_attention is False
    memory = [torch.randn(1, 16, 32) for _ in range(3)]
    out = decoder(memory)
    assert out["query_embeddings"].shape == (1, 8, 32)


def test_query_decoder_cycles_levels() -> None:
    """With 2 levels and 4 layers, levels should be used as 0,1,0,1."""
    D = 32
    decoder = SaliencyQueryTransformerDecoder(
        d_model=D,
        num_queries=4,
        num_layers=4,
        nhead=4,
        num_feature_levels=2,
        use_concept_cross_attention=False,
    )
    mem0 = torch.zeros(1, 9, D)
    mem1 = torch.ones(1, 16, D)
    out = decoder([mem0, mem1])
    assert out["query_embeddings"].shape == (1, 4, D)
    assert torch.isfinite(out["query_embeddings"]).all()


def test_query_decoder_with_optional_pos() -> None:
    B, D = 2, 48
    memory = _make_multiscale_memory(B=B, D=D, hw_list=((3, 5), (6, 10)))
    pos = [torch.randn_like(m) for m in memory]
    decoder = SaliencyQueryTransformerDecoder(
        d_model=D,
        num_queries=8,
        num_layers=2,
        nhead=8,
        num_feature_levels=2,
        use_concept_cross_attention=False,
    )
    out = decoder(memory, multi_scale_pos=pos)
    assert out["query_embeddings"].shape == (B, 8, D)


def test_concept_cross_attention_shapes_and_flag() -> None:
    B, Q, D, S = 2, 8, 64, 4
    decoder = SaliencyQueryTransformerDecoder(
        d_model=D,
        num_queries=Q,
        num_layers=3,
        nhead=8,
        use_concept_cross_attention=True,
        max_concept_tokens=S,
    )
    memory = _make_multiscale_memory(B=B, D=D)
    concept_tokens = torch.randn(B, S, D, requires_grad=True)
    out = decoder(
        memory,
        concept_tokens=concept_tokens,
        return_details=True,
    )
    query_out = out["query_embeddings"]
    attn = out["concept_query_attention"]
    assert query_out.shape == (B, Q, D)
    assert attn is not None
    assert attn.shape == (B, Q, S), attn.shape
    # Attention weights should be a distribution over concept tokens.
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, Q), atol=1e-4)

    loss = query_out.sum()
    loss.backward()
    assert concept_tokens.grad is not None
    assert concept_tokens.grad.abs().sum() > 0
    assert decoder.layers[0].concept_cross_attn.in_proj_weight.grad is not None

    # Flag off: no concept attn even if tokens are provided.
    decoder_off = SaliencyQueryTransformerDecoder(
        d_model=D,
        num_queries=Q,
        num_layers=2,
        use_concept_cross_attention=False,
    )
    out_off = decoder_off(
        memory,
        concept_tokens=concept_tokens.detach(),
        return_details=True,
    )
    assert out_off["concept_query_attention"] is None
    assert out_off["query_embeddings"].shape == (B, Q, D)


def main() -> None:
    test_query_decoder_default_num_queries()
    print("default num_queries=8: OK")
    test_query_decoder_output_shape_and_grads()
    print("shape + grads: OK")
    test_query_decoder_cycles_levels()
    print("multiscale cycling: OK")
    test_query_decoder_with_optional_pos()
    print("optional positional encodings: OK")
    test_concept_cross_attention_shapes_and_flag()
    print("concept cross-attention: OK")


if __name__ == "__main__":
    main()
