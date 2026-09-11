"""Forward/backward smoke tests for concept-patch prioritization at stages 3 and 4."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import compute_patch_priority_map_loss
from model.saliency_prediction import (
    ConceptGatedMultiScaleSaliencyDecoder,
    SpatioTemporalConceptGatedFusionBlock,
    _PRIORITY_EPS,
    _scatter_last_frame_update,
)

STAGE_CONFIG: Dict[str, Tuple[int, int, int]] = {
    "stage1": (8, 8, 32),
    "stage2": (4, 4, 32),
    "stage3": (4, 4, 32),
    "stage4": (2, 2, 32),
}
OUTPUT_SIZE = (16, 16)
LEGACY_STATE_TOKENS = (
    "pairwise_mlp",
    "relevance_gate_mlp",
    "distance_mlp",
    "distance_rbf",
    "visual_mask_queries",
    "motion_concept",
    "centroid",
    "compactness",
)
LEGACY_OUTPUT_KEYS = (
    "concept_compactness_loss",
    "stage_visual_group_centroids",
    "stage_concept_group_validity",
    "stage_motion_concept_volumes",
)
EXPECTED_UNUSED_PREFIXES = (
    "fusion_blocks.stage4.prev_proj.",
    "fusion_blocks.stage4.prev_upsample.",
    "fusion_blocks.stage4.prev_scale",
)


def _make_concept_out(
    *,
    B: int,
    T: int,
    H: int,
    W: int,
    C: int,
    concept_dim: int,
    K: int,
    device: torch.device,
    validity: torch.Tensor | None = None,
) -> Dict[str, Any]:
    if validity is None:
        validity = torch.ones(B, 1, H, W, K, dtype=torch.bool, device=device)
    return {
        "visual_concept_representation": torch.randn(
            B * H * W, concept_dim, device=device
        ),
        "visual_metadata": {
            "feature_shape": {"B": B, "C": C, "T": 1, "H": H, "W": W},
            "window_T": T,
            "assignment_time_index": T - 1,
        },
        "active_visual_prototypes": torch.randn(
            B, 1, H, W, K, concept_dim, device=device
        ),
        "visual_validity_mask": validity,
    }


def _make_decoder_inputs(
    *,
    B: int = 1,
    T: int = 2,
    concept_dim: int = 16,
    K: int = 3,
    device: torch.device | None = None,
    validity: torch.Tensor | None = None,
    invalidate_last_concept: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor], Dict[str, int]]:
    if device is None:
        device = torch.device("cpu")
    concept_outs: Dict[str, Dict[str, Any]] = {}
    features_dict: Dict[str, torch.Tensor] = {}
    stage_channels: Dict[str, int] = {}
    for stage, (H, W, C) in STAGE_CONFIG.items():
        stage_channels[stage] = C
        stage_validity = validity
        if invalidate_last_concept:
            stage_validity = torch.ones(B, 1, H, W, K, dtype=torch.bool, device=device)
            stage_validity[..., -1] = False
        concept_outs[stage] = _make_concept_out(
            B=B,
            T=T,
            H=H,
            W=W,
            C=C,
            concept_dim=concept_dim,
            K=K,
            device=device,
            validity=stage_validity,
        )
        features_dict[stage] = torch.randn(B, C, T, H, W, device=device)
    return concept_outs, features_dict, stage_channels


def _clone_decoder_inputs(
    concept_outs: Dict[str, Dict[str, Any]],
    features_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
    cloned_concepts = {
        stage: {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in payload.items()
        }
        for stage, payload in concept_outs.items()
    }
    cloned_features = {stage: tensor.clone() for stage, tensor in features_dict.items()}
    return cloned_concepts, cloned_features


def _finite(*tensors: torch.Tensor) -> bool:
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def _expected_guided_features(
    block: SpatioTemporalConceptGatedFusionBlock,
    film: torch.Tensor,
    mask_out: Dict[str, torch.Tensor],
    prototypes: torch.Tensor,
) -> torch.Tensor:
    alpha = block.max_strength * torch.sigmoid(block.raw_strength)
    film_last = film[:, :, -1:, :, :]
    _, _, concept_conditioned_update_last = block._build_concept_conditioned_update(
        film_last,
        mask_out["concept_priorities"],
        mask_out["patch_priority_distribution"],
        prototypes,
        mask_out["concept_validity"],
    )
    concept_conditioned_update = _scatter_last_frame_update(
        torch.zeros_like(film),
        concept_conditioned_update_last,
    )
    return film + alpha * (mask_out["priority_mask"] * concept_conditioned_update)


def _film_features(
    block: SpatioTemporalConceptGatedFusionBlock,
    features: torch.Tensor,
    concept_volume: torch.Tensor,
) -> torch.Tensor:
    feature_proj = block.feature_proj(features)
    concept_proj = block.concept_proj(concept_volume)
    gamma, beta = block.film(concept_proj).chunk(2, dim=1)
    gamma = 1.0 + 0.1 * torch.tanh(gamma)
    beta = 0.1 * beta
    return feature_proj * gamma + beta


def _factorized_pair_score(
    block: SpatioTemporalConceptGatedFusionBlock,
    queries: torch.Tensor,
    keys: torch.Tensor,
    n: int,
    m: int,
) -> torch.Tensor:
    scale = block.factor_scale
    return scale * (
        (queries[..., n, :] * keys[..., m, :]).sum(dim=-1)
        - (queries[..., m, :] * keys[..., n, :]).sum(dim=-1)
    )


def _explicit_spatial_context(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    eta: torch.Tensor,
    rank: int,
    eps: float = _PRIORITY_EPS,
) -> torch.Tensor:
    """Explicit ``[B,T,N,N]`` reference for spatially modulated antisymmetric context."""
    content_nm = torch.einsum("btnr,btmr->btnm", q, k)
    content_pairwise = (
        content_nm - content_nm.transpose(-1, -2)
    ) / math.sqrt(rank)
    spatial_kernel = eta @ eta.transpose(-1, -2)
    explicit = content_pairwise * spatial_kernel
    weighted = explicit * weights.unsqueeze(2)
    numerator = weighted.sum(dim=-1)
    total_weight = weights.sum(dim=2)
    denominator = total_weight.unsqueeze(-1) - weights
    context = numerator / denominator.clamp_min(eps)
    context = torch.where(
        denominator > eps,
        context,
        torch.zeros_like(context),
    )
    return context


def _reference_spatial_context(
    block: SpatioTemporalConceptGatedFusionBlock,
    tokens: torch.Tensor,
    activations: torch.Tensor,
    candidate_valid: torch.Tensor,
    x_coords: torch.Tensor,
    y_coords: torch.Tensor,
) -> torch.Tensor:
    """Global-summary reference matching ``_factorized_pairwise_context``."""
    valid = candidate_valid[:, None, :].to(dtype=tokens.dtype)
    weights = activations.clamp_min(0.0) * valid
    q = block.factor_query(tokens)
    k = block.factor_key(tokens)
    eta, _ = block._build_spatial_factors(
        x_coords, y_coords, dtype=tokens.dtype, device=tokens.device
    )
    q_global = torch.einsum("btn,btnr,ns->btrs", weights, q, eta)
    k_global = torch.einsum("btn,btnr,ns->btrs", weights, k, eta)
    q_against_k = torch.einsum("btnr,ns,btrs->btn", q, eta, k_global)
    q_against_q = torch.einsum("btrs,btnr,ns->btn", q_global, k, eta)
    numerator = (q_against_k - q_against_q) / math.sqrt(block.factorized_rank)
    denominator = weights.sum(dim=-1, keepdim=True) - weights
    context = numerator / denominator.clamp_min(_PRIORITY_EPS)
    context = torch.where(
        denominator > _PRIORITY_EPS,
        context,
        torch.zeros_like(context),
    )
    return context * valid


def _priority_inputs_from_block(
    block: SpatioTemporalConceptGatedFusionBlock,
    features: torch.Tensor,
    concept_volume: torch.Tensor,
    prototypes: torch.Tensor,
    validity: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    film = _film_features(block, features, concept_volume)
    film_last = film[:, :, -1:, :, :]
    visual_out = block._compute_soft_activations(film_last, prototypes, validity)
    part = block._build_activation_tokens(
        visual_out["projected_features"],
        visual_out["projected_prototypes"],
        visual_out["activations"],
    )
    B, T, N, _ = part["tokens"].shape
    H, W = part["spatial_hw"]
    candidate_valid = validity.reshape(B, N)
    return {
        "film": film,
        "activations_map": visual_out["activations"],
        **part,
        "candidate_valid": candidate_valid,
    }


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, msg: str) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=1e-5):
        delta = (actual - expected).abs().max().item()
        raise AssertionError(f"{msg}: max abs diff {delta} > atol {atol}")


def test_fusion_block() -> Dict[str, Any]:
    torch.manual_seed(0)
    B, T, H, W, C, D, K = 2, 2, 3, 4, 16, 16, 3
    rank = 8
    block = SpatioTemporalConceptGatedFusionBlock(
        feature_channels=C,
        concept_dim=D,
        decoder_channels=16,
        dropout=0.0,
        enable_patch_prioritization=True,
        factorized_rank=rank,
    )
    block.train()
    features = torch.randn(B, C, T, H, W, requires_grad=True)
    concept_volume = torch.randn(B, D, T, H, W, requires_grad=True)
    prototypes = torch.randn(B, 1, H, W, K, D, requires_grad=True)
    validity = torch.ones(B, 1, H, W, K, dtype=torch.bool)
    validity[..., -1] = False
    K_valid = K - 1
    n_rect = K * H * W
    n_valid = K_valid * H * W
    n_pairs = n_valid * (n_valid - 1)

    captured: Dict[str, torch.Tensor] = {}

    def _capture_refine_input(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["fused"] = inputs[0]

    handle = block.refine1.register_forward_hook(_capture_refine_input)
    t0 = time.perf_counter()
    decoded, mask_out = block(
        features,
        concept_volume,
        active_visual_prototypes=prototypes,
        visual_validity_mask=validity,
    )
    fusion_s = time.perf_counter() - t0
    handle.remove()

    priority_mask = mask_out["priority_mask"]
    priorities = mask_out["concept_priorities"]
    distribution = mask_out["patch_priority_distribution"]
    gate = mask_out["patch_priority_gate"]
    patch_map = mask_out["patch_priority_map"]
    conditional_weights = mask_out["conditional_concept_weights"]
    concept_context = mask_out["priority_weighted_concept_context"]
    activations = mask_out["concept_patch_activations"]
    film = _film_features(block, features, concept_volume)
    film_last = film[:, :, -1:, :, :]
    alpha = block.max_strength * torch.sigmoid(block.raw_strength)
    expected_guided = _expected_guided_features(block, film, mask_out, prototypes)

    T_prio = 1
    assert decoded.shape == (B, 16, T, H, W)
    assert priority_mask.shape == (B, 1, T, H, W)
    assert priorities.shape == (B, T_prio, K, H, W)
    assert distribution.shape == (B, T_prio, H, W)
    assert gate.shape == (B, T_prio, H, W)
    assert patch_map.shape == (B, T_prio, H, W)
    assert conditional_weights.shape == (B, T_prio, K, H, W)
    assert concept_context.shape == (B, 16, T_prio, H, W)
    assert activations.shape == (B, T_prio, K, H, W)
    assert _finite(
        decoded,
        priority_mask,
        priorities,
        distribution,
        gate,
        patch_map,
        conditional_weights,
        concept_context,
        activations,
    )
    _assert_close(priorities.sum(dim=(2, 3, 4)), torch.ones(B, T_prio), 1e-5, "priorities sum")
    _assert_close(priorities[:, :, -1], torch.zeros(B, T_prio, H, W), 1e-6, "padded concept priority")
    _assert_close(distribution, priorities.sum(dim=2), 1e-6, "distribution vs concept sum")
    _assert_close(patch_map, distribution, 1e-6, "patch map alias")
    _assert_close(distribution.sum(dim=(-2, -1)), torch.ones(B, T_prio), 1e-5, "distribution sums to 1")
    _assert_close(gate.mean(dim=(-2, -1)), torch.ones(B, T_prio), 1e-5, "gate mean is 1")
    valid = mask_out["concept_validity"].permute(0, 1, 4, 2, 3)
    _assert_close(
        (conditional_weights * valid.to(dtype=conditional_weights.dtype)).sum(dim=2),
        torch.ones(B, T_prio, H, W),
        1e-5,
        "conditional weights sum over valid concepts",
    )
    assert float(priority_mask.detach().min()) >= 0.0
    assert torch.all(activations[:, :, -1] == 0)
    assert torch.all(activations.sum(dim=2) <= 1.0 + 1e-5)
    _assert_close(captured["fused"], expected_guided, 1e-5, "residual FiLM+concept branch gating")
    assert not torch.allclose(captured["fused"], priority_mask * film, atol=1e-4)
    assert not torch.allclose(captured["fused"], film + alpha * (priority_mask * film), atol=1e-4)

    mutated_priorities = priorities.clone()
    swap = mutated_priorities[:, :, 0].clone()
    mutated_priorities[:, :, 0] = mutated_priorities[:, :, 1]
    mutated_priorities[:, :, 1] = swap
    fixed_distribution = distribution.detach()
    with torch.no_grad():
        _, context_a, _ = block._build_concept_conditioned_update(
            film_last,
            priorities,
            fixed_distribution,
            prototypes,
            mask_out["concept_validity"],
        )
        _, context_b, _ = block._build_concept_conditioned_update(
            film_last,
            mutated_priorities,
            fixed_distribution,
            prototypes,
            mask_out["concept_validity"],
        )
    assert not torch.allclose(context_a, context_b, atol=1e-4), (
        "concept context must change when concept mix changes at fixed distribution"
    )

    internals = _priority_inputs_from_block(
        block, features, concept_volume, prototypes, validity
    )
    t1 = time.perf_counter()
    context, factorized_stats = block._factorized_pairwise_context(
        internals["tokens"],
        internals["activations"],
        internals["candidate_valid"],
        internals["x_coords"],
        internals["y_coords"],
    )
    pairwise_s = time.perf_counter() - t1
    ref_context = _reference_spatial_context(
        block,
        internals["tokens"],
        internals["activations"],
        internals["candidate_valid"],
        internals["x_coords"],
        internals["y_coords"],
    )
    valid = internals["candidate_valid"][:, None, :].to(dtype=internals["tokens"].dtype)
    weights = internals["activations"].clamp_min(0.0) * valid
    q = block.factor_query(internals["tokens"])
    k = block.factor_key(internals["tokens"])
    eta, _ = block._build_spatial_factors(
        internals["x_coords"],
        internals["y_coords"],
        dtype=internals["tokens"].dtype,
        device=internals["tokens"].device,
    )
    explicit_context = _explicit_spatial_context(
        q, k, weights, eta, block.factorized_rank
    )
    _assert_close(context, ref_context, 1e-5, "global vs closed-form context")
    _assert_close(ref_context, explicit_context * valid, 1e-5, "global vs explicit context")

    queries = block.factor_query(internals["tokens"])
    keys = block.factor_key(internals["tokens"])
    N = queries.shape[2]
    for n in range(N):
        self_score = _factorized_pair_score(block, queries, keys, n, n)
        _assert_close(self_score, torch.zeros_like(self_score), 1e-6, f"self score n={n}")
        for m in range(n + 1, N):
            r_nm = _factorized_pair_score(block, queries, keys, n, m)
            r_mn = _factorized_pair_score(block, queries, keys, m, n)
            _assert_close(r_nm, -r_mn, 1e-6, f"antisymmetry n={n}, m={m}")

    assert not torch.allclose(
        block.factor_query[-1].weight,
        block.factor_key[-1].weight,
    )

    features.grad = None
    concept_volume.grad = None
    prototypes.grad = None
    block.zero_grad(set_to_none=True)
    loss = decoded.square().mean() + patch_map[:, -1].square().mean()
    loss.backward()

    def _has_grad(param: torch.Tensor | None) -> bool:
        return param is not None and param.grad is not None and float(param.grad.abs().sum()) > 0

    grad_report = {
        "features": features.grad is not None and float(features.grad.abs().sum()) > 0,
        "prototypes": _has_grad(prototypes),
        "unary_priority_mlp": _has_grad(block.unary_priority_mlp[0].weight),
        "factor_query_first": _has_grad(block.factor_query[0].weight),
        "factor_query_last": _has_grad(block.factor_query[-1].weight),
        "factor_key_first": _has_grad(block.factor_key[0].weight),
        "factor_key_last": _has_grad(block.factor_key[-1].weight),
        "raw_strength": _has_grad(block.raw_strength),
        "raw_pairwise_strength": _has_grad(block.raw_pairwise_strength),
        "feature_token_proj": _has_grad(block.feature_token_proj.weight),
        "concept_proto_proj": _has_grad(block.concept_proto_proj.weight),
        "null_prototype": _has_grad(block.null_prototype),
        "film": _has_grad(block.film.weight),
        "mask_feature_branch": _has_grad(block.mask_feature_branch.conv.weight),
        "priority_application_proto_proj": _has_grad(
            block.priority_application_proto_proj.weight
        ),
    }

    block.zero_grad(set_to_none=True)
    block.eval()
    features2 = features.detach().requires_grad_(True)
    concept2 = concept_volume.detach().requires_grad_(True)
    proto2 = prototypes.detach().requires_grad_(True)
    _, mask2 = block(
        features2,
        concept2,
        active_visual_prototypes=proto2,
        visual_validity_mask=validity,
    )
    aux = mask2["patch_priority_map"].square().mean()
    aux.backward()
    early_grad = float(features2.grad[:, :, 0].abs().sum())
    last_grad = float(features2.grad[:, :, -1].abs().sum())
    last_frame_only = (
        features2.grad is not None
        and last_grad > 0
        and last_grad > early_grad * 3.0
    )
    block.train()

    with torch.no_grad():
        fused_early, fused_late = captured["fused"][:, :, 0], captured["fused"][:, :, -1]
        film_early, film_late = film[:, :, 0], film[:, :, -1]
    frame_isolated = torch.allclose(
        fused_early,
        film_early,
        atol=1e-5,
    ) and not torch.allclose(
        fused_late,
        film_late,
        atol=1e-4,
    )

    return {
        "decoded_shape": tuple(decoded.shape),
        "priority_mask_shape": tuple(priority_mask.shape),
        "priorities_shape": tuple(priorities.shape),
        "activations_shape": tuple(activations.shape),
        "candidate_count_rect": n_rect,
        "candidate_count_valid": n_valid,
        "directed_pairs_per_frame": n_pairs,
        "factorized_rank": rank,
        "fusion_forward_s": fusion_s,
        "pairwise_context_s": pairwise_s,
        "grads": grad_report,
        "last_frame_aux_isolates_t": last_frame_only,
        "frames_not_mixed": frame_isolated,
        "self_pairs_zero": True,
        "antisymmetric_pairs": True,
        "residual_not_replace": True,
        "state_has_legacy": any(
            token in key for key in block.state_dict() for token in LEGACY_STATE_TOKENS
        ),
        "factor_query_last_has_bias": block.factor_query[-1].bias is not None,
        "factor_key_last_has_bias": block.factor_key[-1].bias is not None,
        "unary_last_has_bias": block.unary_priority_mlp[-1].bias is not None,
        "factorized_stats": {
            k: float(v.detach().mean()) if torch.is_tensor(v) and v.numel() > 1
            else float(v.detach()) if torch.is_tensor(v)
            else float(v)
            for k, v in factorized_stats.items()
        },
    }


def test_stage12_without_priority() -> Dict[str, Any]:
    torch.manual_seed(2)
    B, T, H, W, C, D, K = 1, 2, 4, 4, 16, 16, 2
    block = SpatioTemporalConceptGatedFusionBlock(
        feature_channels=C,
        concept_dim=D,
        decoder_channels=16,
        dropout=0.0,
        enable_patch_prioritization=False,
    )
    features = torch.randn(B, C, T, H, W)
    concept_volume = torch.randn(B, D, T, H, W)
    prototypes = torch.randn(B, 1, H, W, K, D)
    validity = torch.ones(B, 1, H, W, K, dtype=torch.bool)
    captured: Dict[str, torch.Tensor] = {}

    def _capture(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["fused"] = inputs[0]

    handle = block.refine1.register_forward_hook(_capture)
    decoded, mask_out = block(
        features,
        concept_volume,
        active_visual_prototypes=prototypes,
        visual_validity_mask=validity,
    )
    handle.remove()
    film = _film_features(block, features, concept_volume)
    _assert_close(captured["fused"], film, 1e-5, "stage without priority is FiLM only")
    assert mask_out == {}
    assert block.unary_priority_mlp is None
    assert block.factor_query is None
    assert block.factor_key is None
    assert block.priority_application_proto_proj is None
    assert block.raw_spatial_frequency_weights is None
    assert block.raw_spatial_strength is None
    assert block.mask_feature_branch is None
    assert _finite(decoded)
    return {"decoded_shape": tuple(decoded.shape), "mask_outputs": mask_out}


def test_decoder() -> Dict[str, Any]:
    torch.manual_seed(1)
    B, T, K, concept_dim = 1, 2, 3, 16
    concept_outs, features_dict, stage_channels = _make_decoder_inputs(
        B=B,
        T=T,
        concept_dim=concept_dim,
        K=K,
        invalidate_last_concept=True,
    )
    for tensor in features_dict.values():
        tensor.requires_grad_(True)
    decoder = ConceptGatedMultiScaleSaliencyDecoder(
        stage_channels=stage_channels,
        concept_dim=concept_dim,
        decoder_channels=16,
        dropout=0.0,
        output_activation="sigmoid",
    )
    decoder.train()
    rank = decoder.fusion_blocks["stage3"].factorized_rank

    timings: Dict[str, float] = {}
    memory: Dict[str, Dict[str, int]] = {}
    for stage in ("stage3", "stage4"):
        H, W, _ = STAGE_CONFIG[stage]
        n_rect = K * H * W
        n_valid = 2 * H * W
        memory[stage] = {
            "candidates_rect": n_rect,
            "candidates_valid": n_valid,
            "directed_pairs": n_valid * (n_valid - 1),
            "factorized_rank": rank,
            "query_key_tensor_elems": B * T * n_rect * rank * 2,
        }

    t0 = time.perf_counter()
    out = decoder(
        concept_outs=concept_outs,
        features_dict=features_dict,
        output_size=OUTPUT_SIZE,
        return_details=True,
    )
    timings["decoder_forward_s"] = time.perf_counter() - t0

    gt = torch.rand(B, 1, *OUTPUT_SIZE)
    maps = out["stage_patch_priority_maps"]
    gates = out["stage_patch_priority_gates"]
    distributions = out["stage_patch_priority_distributions"]
    for stage, priority_map in maps.items():
        priority_map.retain_grad()
    priority_losses = compute_patch_priority_map_loss(out, gt)
    priority_losses["loss_priority_map"].backward(retain_graph=True)
    last_frame_grads = {
        stage: {
            "last": float(maps[stage].grad[:, -1].abs().sum()),
            "only_frame": int(maps[stage].shape[1]) == 1,
        }
        for stage in ("stage3", "stage4")
        if maps[stage].grad is not None
    }
    decoder.zero_grad(set_to_none=True)

    side_terms = [logit.mean() for logit in out["side_saliency_logits"].values()]
    side_patch_terms = [
        logit.mean() for logit in out["side_patch_saliency_logits"].values()
    ]
    loss = (
        out["saliency_map"].mean()
        + out["patch_saliency_logits"].mean()
        + sum(side_terms)
        + sum(side_patch_terms)
        + 0.05 * priority_losses["loss_priority_map"]
    )
    assert torch.isfinite(loss)
    loss.backward()

    unused: List[str] = []
    expected_unused: List[str] = []
    for name, param in decoder.named_parameters():
        if not param.requires_grad:
            continue
        missing = param.grad is None or float(param.grad.abs().sum()) == 0.0
        if not missing:
            continue
        if name.startswith(EXPECTED_UNUSED_PREFIXES) or name == "fusion_blocks.stage4.prev_scale":
            expected_unused.append(name)
        else:
            unused.append(name)

    motion_modules = [
        name
        for name, _ in decoder.named_modules()
        if "motion" in name.lower()
    ]
    motion_params = [
        name
        for name, _ in decoder.named_parameters()
        if "motion" in name.lower()
    ]

    T_prio = 1
    stage3_p = out["stage_concept_priorities"]["stage3"]
    stage4_p = out["stage_concept_priorities"]["stage4"]
    _assert_close(stage3_p.sum(dim=(2, 3, 4)), torch.ones(B, T_prio), 1e-4, "stage3 softmax")
    _assert_close(stage4_p.sum(dim=(2, 3, 4)), torch.ones(B, T_prio), 1e-4, "stage4 softmax")
    _assert_close(stage3_p[:, :, -1], torch.zeros_like(stage3_p[:, :, -1]), 1e-6, "stage3 padded")
    for stage in ("stage3", "stage4"):
        dist = distributions[stage]
        gate = gates[stage]
        _assert_close(dist.sum(dim=(-2, -1)), torch.ones(B, T_prio), 1e-4, f"{stage} distribution sum")
        _assert_close(gate.mean(dim=(-2, -1)), torch.ones(B, T_prio), 1e-4, f"{stage} gate mean")
        _assert_close(maps[stage], dist, 1e-6, f"{stage} supervised map alias")
    _assert_close(
        maps["stage3"],
        out["stage_concept_priorities"]["stage3"].sum(dim=2),
        1e-5,
        "stage3 map sum",
    )

    decoder.eval()
    with torch.no_grad():
        base_concepts, base_features = _clone_decoder_inputs(concept_outs, features_dict)
        base = decoder(base_concepts, base_features, OUTPUT_SIZE, return_details=True)
        mutated_s3, feat_s3 = _clone_decoder_inputs(concept_outs, features_dict)
        feat_s3["stage3"].add_(1.0)
        mut3 = decoder(mutated_s3, feat_s3, OUTPUT_SIZE, return_details=True)
        mutated_s4, feat_s4 = _clone_decoder_inputs(concept_outs, features_dict)
        feat_s4["stage4"].add_(1.0)
        mut4 = decoder(mutated_s4, feat_s4, OUTPUT_SIZE, return_details=True)

    stages_not_mixed = torch.allclose(
        base["stage_concept_priorities"]["stage4"],
        mut3["stage_concept_priorities"]["stage4"],
        atol=1e-5,
    ) and torch.allclose(
        base["stage_concept_priorities"]["stage3"],
        mut4["stage_concept_priorities"]["stage3"],
        atol=1e-5,
    ) and not torch.allclose(
        base["stage_concept_priorities"]["stage3"],
        mut3["stage_concept_priorities"]["stage3"],
        atol=1e-4,
    ) and not torch.allclose(
        base["stage_concept_priorities"]["stage4"],
        mut4["stage_concept_priorities"]["stage4"],
        atol=1e-4,
    )

    side_shapes = {
        stage: tuple(logit.shape) for stage, logit in out["side_saliency_logits"].items()
    }
    return {
        "saliency_map_shape": tuple(out["saliency_map"].shape),
        "patch_logits_shape": tuple(out["patch_saliency_logits"].shape),
        "side_logit_shapes": side_shapes,
        "priority_map_stages": sorted(maps.keys()),
        "distribution_stages": sorted(distributions.keys()),
        "gate_stages": sorted(gates.keys()),
        "priority_stages": sorted(out["stage_concept_priorities"].keys()),
        "stage3_priority_shape": tuple(stage3_p.shape),
        "stage4_priority_shape": tuple(stage4_p.shape),
        "stage1_enabled": decoder.fusion_blocks["stage1"].enable_patch_prioritization,
        "stage2_enabled": decoder.fusion_blocks["stage2"].enable_patch_prioritization,
        "stage3_enabled": decoder.fusion_blocks["stage3"].enable_patch_prioritization,
        "stage4_enabled": decoder.fusion_blocks["stage4"].enable_patch_prioritization,
        "legacy_output_keys": [key for key in LEGACY_OUTPUT_KEYS if key in out],
        "legacy_state": any(
            token in key for key in decoder.state_dict() for token in LEGACY_STATE_TOKENS
        ),
        "motion_modules": motion_modules,
        "motion_params": motion_params,
        "priority_kl": float(priority_losses["loss_priority_map"].detach()),
        "last_frame_map_grads": last_frame_grads,
        "stages_not_mixed": stages_not_mixed,
        "unexpected_unused_params": unused,
        "expected_unused_stage4_prev": expected_unused,
        "pairwise_memory": memory,
        "factorized_rank": rank,
        "timings": timings,
        "finite_saliency": bool(torch.isfinite(out["saliency_map"]).all()),
    }


def main() -> None:
    fusion = test_fusion_block()
    stage12 = test_stage12_without_priority()
    decoder = test_decoder()

    assert fusion["priority_mask_shape"][1] == 1
    assert all(fusion["grads"].values()), fusion["grads"]
    assert fusion["antisymmetric_pairs"]
    assert fusion["self_pairs_zero"]
    assert fusion["last_frame_aux_isolates_t"]
    assert fusion["frames_not_mixed"]
    assert not fusion["state_has_legacy"]
    assert not fusion["factor_query_last_has_bias"]
    assert not fusion["factor_key_last_has_bias"]
    assert not fusion["unary_last_has_bias"]
    assert stage12["mask_outputs"] == {}
    assert decoder["priority_stages"] == ["stage3", "stage4"]
    assert decoder["priority_map_stages"] == ["stage3", "stage4"]
    assert decoder["distribution_stages"] == ["stage3", "stage4"]
    assert decoder["gate_stages"] == ["stage3", "stage4"]
    assert decoder["saliency_map_shape"] == (1, 1, *OUTPUT_SIZE)
    assert all(shape == (1, 1, *OUTPUT_SIZE) for shape in decoder["side_logit_shapes"].values())
    assert not decoder["stage1_enabled"] and not decoder["stage2_enabled"]
    assert decoder["stage3_enabled"] and decoder["stage4_enabled"]
    assert decoder["legacy_output_keys"] == []
    assert not decoder["legacy_state"]
    assert decoder["motion_modules"] == []
    assert decoder["motion_params"] == []
    assert decoder["stages_not_mixed"]
    assert decoder["unexpected_unused_params"] == [], decoder["unexpected_unused_params"]
    for stage, grads in decoder["last_frame_map_grads"].items():
        assert grads["last"] > 0, stage
        assert grads["only_frame"], (stage, grads)

    print("fusion block:")
    for key, value in fusion.items():
        print(f"  {key}: {value}")
    print("stage1/2 FiLM-only:")
    for key, value in stage12.items():
        print(f"  {key}: {value}")
    print("decoder:")
    for key, value in decoder.items():
        print(f"  {key}: {value}")
    print("concept-patch prioritization smoke test: OK")


if __name__ == "__main__":
    main()
