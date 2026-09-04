"""Unit tests for spatially modulated factorized antisymmetric concept-patch comparison."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from model.saliency_prediction import (
    SpatioTemporalConceptGatedFusionBlock,
    _PRIORITY_EPS,
)

_FORBIDDEN_SOURCE_PATTERNS = (
    r"\bpairwise_mlp\b",
    r"\brelevance_gate_mlp\b",
    r"\bdistance_mlp\b",
    r"\bdistance_rbf\b",
    r"\b_directed_pair_logits\b",
    r"\b_pairwise_context_impl\b",
    r"torch\.utils\.checkpoint.*pairwise",
    r"for\s+\w+\s+in\s+range\s*\(\s*N\s*\).*for\s+\w+\s+in\s+range\s*\(\s*N\s*\)",
)
_SALIENCY_SOURCE = Path(__file__).resolve().parent / "model" / "saliency_prediction.py"

_TRAIN_STAGE3 = {"B": 2, "T": 8, "K": 512, "H": 14, "W": 24, "C": 384, "D": 128}
_TRAIN_STAGE4 = {"B": 2, "T": 8, "K": 512, "H": 7, "W": 12, "C": 768, "D": 128}


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    msg: str,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(f"{msg}: {exc}") from exc


def _make_coords(N: int, *, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    side = max(2, int(math.ceil(math.sqrt(N))))
    ys = (torch.arange(side, device=device, dtype=dtype) + 0.5) / float(side)
    xs = (torch.arange(side, device=device, dtype=dtype) + 0.5) / float(side)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    flat_x = grid_x.reshape(-1)[:N]
    flat_y = grid_y.reshape(-1)[:N]
    return flat_x, flat_y


def _global_spatial_context(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    eta: torch.Tensor,
    rank: int,
    eps: float = _PRIORITY_EPS,
) -> torch.Tensor:
    q_global = torch.einsum("btn,btnr,ns->btrs", weights, q, eta)
    k_global = torch.einsum("btn,btnr,ns->btrs", weights, k, eta)
    q_against_k = torch.einsum("btnr,ns,btrs->btn", q, eta, k_global)
    q_against_q = torch.einsum("btrs,btnr,ns->btn", q_global, k, eta)
    numerator = (q_against_k - q_against_q) / math.sqrt(rank)
    denominator = weights.sum(dim=-1, keepdim=True) - weights
    context = numerator / denominator.clamp_min(eps)
    return torch.where(
        denominator > eps,
        context,
        torch.zeros_like(context),
    )


def _explicit_spatial_context(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    eta: torch.Tensor,
    rank: int,
    eps: float = _PRIORITY_EPS,
) -> torch.Tensor:
    content_nm = torch.einsum("btnr,btmr->btnm", q, k)
    content_pairwise = (
        content_nm - content_nm.transpose(-1, -2)
    ) / math.sqrt(rank)
    spatial_kernel = eta @ eta.transpose(-1, -2)
    explicit_pairwise = content_pairwise * spatial_kernel
    weighted = explicit_pairwise * weights.unsqueeze(2)
    numerator = weighted.sum(dim=-1)
    denominator = weights.sum(dim=-1, keepdim=True) - weights
    context = numerator / denominator.clamp_min(eps)
    return torch.where(
        denominator > eps,
        context,
        torch.zeros_like(context),
    )


def _content_only_global_context(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    rank: int,
    eps: float = _PRIORITY_EPS,
) -> torch.Tensor:
    q_global = (weights.unsqueeze(-1) * q).sum(dim=2)
    k_global = (weights.unsqueeze(-1) * k).sum(dim=2)
    numerator = (
        (q * k_global.unsqueeze(2)).sum(dim=-1)
        - (k * q_global.unsqueeze(2)).sum(dim=-1)
    ) / math.sqrt(rank)
    denominator = weights.sum(dim=-1, keepdim=True) - weights
    context = numerator / denominator.clamp_min(eps)
    return torch.where(
        denominator > eps,
        context,
        torch.zeros_like(context),
    )


def _make_block(*, rank: int = 8) -> SpatioTemporalConceptGatedFusionBlock:
    return SpatioTemporalConceptGatedFusionBlock(
        feature_channels=16,
        concept_dim=16,
        decoder_channels=16,
        dropout=0.0,
        enable_patch_prioritization=True,
        factorized_rank=rank,
    )


def test_explicit_matches_factorized_context() -> Dict[str, float]:
    torch.manual_seed(7)
    B, T, N, R = 2, 3, 7, 4
    block = _make_block(rank=R)
    block.eval()

    tokens = torch.randn(B, T, N, block.token_dim)
    activations = torch.rand(B, T, N)
    candidate_valid = torch.ones(B, N, dtype=torch.bool)
    candidate_valid[:, -2:] = False
    x_coords, y_coords = _make_coords(N, device=tokens.device, dtype=tokens.dtype)

    valid = candidate_valid[:, None, :].to(dtype=tokens.dtype)
    weights = activations.clamp_min(0.0) * valid
    q = block.factor_query(tokens)
    k = block.factor_key(tokens)
    eta, rho = block._build_spatial_factors(
        x_coords, y_coords, dtype=tokens.dtype, device=tokens.device
    )

    global_ctx = _global_spatial_context(q, k, weights, eta, R)
    explicit_ctx = _explicit_spatial_context(q, k, weights, eta, R)
    block_ctx, stats = block._factorized_pairwise_context(
        tokens,
        activations,
        candidate_valid,
        x_coords,
        y_coords,
    )

    content_nm = torch.einsum("btnr,btmr->btnm", q, k)
    content_pairwise = (
        content_nm - content_nm.transpose(-1, -2)
    ) / math.sqrt(R)
    spatial_kernel = eta @ eta.transpose(-1, -2)

    _assert_close(
        content_pairwise,
        -content_pairwise.transpose(-1, -2),
        rtol=1e-5,
        atol=1e-6,
        msg="explicit antisymmetry",
    )
    diag = torch.diagonal(content_pairwise, dim1=-2, dim2=-1)
    _assert_close(diag, torch.zeros_like(diag), rtol=0.0, atol=1e-6, msg="zero diagonal")

    _assert_close(
        spatial_kernel,
        spatial_kernel.transpose(-1, -2),
        rtol=1e-5,
        atol=1e-6,
        msg="spatial kernel symmetry",
    )
    kernel_diag = torch.diagonal(spatial_kernel, dim1=-2, dim2=-1)
    _assert_close(
        kernel_diag,
        torch.ones_like(kernel_diag),
        rtol=1e-4,
        atol=1e-5,
        msg="spatial kernel diagonal is one",
    )
    assert torch.isfinite(spatial_kernel).all()
    assert float(spatial_kernel.detach().min()) >= -1e-5
    assert float(spatial_kernel.detach().max()) <= 1.0 + 1e-5

    explicit_pairwise = content_pairwise * spatial_kernel
    _assert_close(
        explicit_pairwise,
        -explicit_pairwise.transpose(-1, -2),
        rtol=1e-5,
        atol=1e-6,
        msg="modulated pairwise antisymmetry",
    )

    _assert_close(
        global_ctx * valid,
        explicit_ctx * valid,
        rtol=1e-4,
        atol=1e-5,
        msg="global vs explicit context",
    )
    _assert_close(
        block_ctx,
        global_ctx * valid,
        rtol=1e-4,
        atol=1e-5,
        msg="block vs global context",
    )

    assert "spatial_strength" in stats
    assert "spatial_frequency_weights" in stats
    assert "spatial_frequency_weight_entropy" in stats

    return {
        "B": float(B),
        "T": float(T),
        "N": float(N),
        "rank": float(R),
        "eta_shape_S": float(eta.shape[-1]),
        "explicit_tensor_elems": float(B * T * N * N),
    }


def test_lambda_zero_matches_content_only() -> Dict[str, float]:
    torch.manual_seed(19)
    B, T, N, R = 1, 2, 6, 4
    block = _make_block(rank=R)
    block.eval()
    with torch.no_grad():
        block.raw_spatial_strength.fill_(-30.0)

    tokens = torch.randn(B, T, N, block.token_dim)
    activations = torch.rand(B, T, N)
    candidate_valid = torch.ones(B, N, dtype=torch.bool)
    x_coords, y_coords = _make_coords(N, device=tokens.device, dtype=tokens.dtype)

    valid = candidate_valid[:, None, :].to(dtype=tokens.dtype)
    weights = activations.clamp_min(0.0) * valid
    q = block.factor_query(tokens)
    k = block.factor_key(tokens)

    spatial_ctx, _ = block._factorized_pairwise_context(
        tokens,
        activations,
        candidate_valid,
        x_coords,
        y_coords,
    )
    content_ctx = _content_only_global_context(q, k, weights, R)
    _assert_close(
        spatial_ctx,
        content_ctx * valid,
        rtol=1e-4,
        atol=1e-5,
        msg="lambda=0 matches content-only comparison",
    )
    return {"spatial_strength": float(block.spatial_strength.detach())}


def test_gradient_flow() -> Dict[str, bool]:
    torch.manual_seed(11)
    B, T, N, R = 1, 2, 5, 3
    block = _make_block(rank=R)
    block.train()

    tokens = torch.randn(B, T, N, block.token_dim, requires_grad=True)
    activations = torch.rand(B, T, N, requires_grad=True)
    candidate_valid = torch.ones(B, N, dtype=torch.bool)
    x_coords, y_coords = _make_coords(N, device=tokens.device, dtype=tokens.dtype)

    context, _ = block._factorized_pairwise_context(
        tokens,
        activations,
        candidate_valid,
        x_coords,
        y_coords,
    )
    context.sum().backward()

    grad_report = {
        "activations": activations.grad is not None
        and torch.isfinite(activations.grad).all()
        and float(activations.grad.abs().sum()) > 0,
        "tokens": tokens.grad is not None
        and torch.isfinite(tokens.grad).all()
        and float(tokens.grad.abs().sum()) > 0,
        "factor_query": block.factor_query[0].weight.grad is not None
        and float(block.factor_query[0].weight.grad.abs().sum()) > 0,
        "factor_key": block.factor_key[0].weight.grad is not None
        and float(block.factor_key[0].weight.grad.abs().sum()) > 0,
        "raw_spatial_frequency_weights": block.raw_spatial_frequency_weights.grad is not None
        and float(block.raw_spatial_frequency_weights.grad.abs().sum()) > 0,
        "raw_spatial_strength": block.raw_spatial_strength.grad is not None
        and float(block.raw_spatial_strength.grad.abs().sum()) > 0,
    }
    if not all(grad_report.values()):
        raise AssertionError(f"missing spatial comparison grads: {grad_report}")
    return grad_report


def test_invalid_candidates_are_zero() -> None:
    torch.manual_seed(5)
    B, T, N, R = 1, 1, 5, 4
    block = _make_block(rank=R)
    block.eval()

    tokens = torch.randn(B, T, N, block.token_dim)
    activations = torch.rand(B, T, N)
    candidate_valid = torch.tensor([[True, True, True, False, False]])
    x_coords, y_coords = _make_coords(N, device=tokens.device, dtype=tokens.dtype)

    context, _ = block._factorized_pairwise_context(
        tokens,
        activations,
        candidate_valid,
        x_coords,
        y_coords,
    )
    _assert_close(
        context[:, :, 3:],
        torch.zeros(B, T, 2, dtype=context.dtype),
        rtol=0.0,
        atol=1e-7,
        msg="invalid candidate context is zero",
    )

    valid_only = torch.tensor([[True, True, True, False, False]])
    mutated_activations = activations.clone()
    mutated_activations[:, :, 3:] = 10.0
    context_mut, _ = block._factorized_pairwise_context(
        tokens,
        mutated_activations,
        valid_only,
        x_coords,
        y_coords,
    )
    _assert_close(
        context[:, :, :3],
        context_mut[:, :, :3],
        rtol=1e-5,
        atol=1e-6,
        msg="invalid candidates do not affect valid contexts",
    )


def test_prioritization_properties() -> Dict[str, Tuple[int, ...]]:
    torch.manual_seed(3)
    B, T, H, W, C, D, K = 2, 2, 3, 4, 16, 16, 4
    block = _make_block(rank=6)
    block.train()

    features = torch.randn(B, C, T, H, W, requires_grad=True)
    concept_volume = torch.randn(B, D, T, H, W, requires_grad=True)
    prototypes = torch.randn(B, K, D, requires_grad=True)
    validity = torch.ones(B, K, dtype=torch.bool)
    validity[:, -1] = False

    decoded, mask_out = block(
        features,
        concept_volume,
        active_visual_prototypes=prototypes,
        visual_validity_mask=validity,
    )
    priorities = mask_out["concept_priorities"]
    patch_map = mask_out["patch_priority_map"]

    _assert_close(
        priorities.sum(dim=(2, 3, 4)),
        torch.ones(B, T),
        rtol=1e-5,
        atol=1e-5,
        msg="priority softmax",
    )
    _assert_close(
        priorities[:, :, -1],
        torch.zeros(B, T, H, W),
        rtol=0.0,
        atol=1e-6,
        msg="padded concept priority",
    )
    _assert_close(
        patch_map,
        priorities.sum(dim=2),
        rtol=0.0,
        atol=1e-6,
        msg="patch map from concept sum",
    )
    return {
        "priorities_shape": tuple(priorities.shape),
        "patch_map_shape": tuple(patch_map.shape),
        "priority_mask_shape": tuple(mask_out["priority_mask"].shape),
    }


def test_no_legacy_pairwise_code() -> Dict[str, int]:
    source = _SALIENCY_SOURCE.read_text(encoding="utf-8")
    hits: Dict[str, int] = {}
    for pattern in _FORBIDDEN_SOURCE_PATTERNS:
        matches = re.findall(pattern, source)
        if matches:
            hits[pattern] = len(matches)
    if hits:
        raise AssertionError(f"legacy pairwise artifacts remain: {hits}")
    return {"forbidden_patterns_found": 0}


def _benchmark_pairwise_context(
    cfg: Dict[str, int],
    *,
    rank: int,
    device: torch.device,
) -> Dict[str, float]:
    B, T, K, H, W = cfg["B"], cfg["T"], cfg["K"], cfg["H"], cfg["W"]
    N = K * H * W
    block = SpatioTemporalConceptGatedFusionBlock(
        feature_channels=16,
        concept_dim=16,
        decoder_channels=16,
        dropout=0.0,
        enable_patch_prioritization=True,
        factorized_rank=rank,
    ).to(device)
    block.train()

    tokens = torch.randn(B, T, N, block.token_dim, device=device)
    activations = torch.rand(B, T, N, device=device, requires_grad=True)
    candidate_valid = torch.ones(B, N, dtype=torch.bool, device=device)
    x_coords, y_coords = _make_coords(N, device=device, dtype=tokens.dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    context, _ = block._factorized_pairwise_context(
        tokens,
        activations,
        candidate_valid,
        x_coords,
        y_coords,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_s = time.perf_counter() - t0
    forward_peak = (
        float(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0.0
    )

    loss = context.square().mean()
    backward_peak = forward_peak
    backward_s = 0.0
    try:
        t1 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_s = time.perf_counter() - t1
        backward_peak = (
            float(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0.0
        )
    except torch.OutOfMemoryError:
        backward_peak = float("nan")

    total_s = forward_s + backward_s
    peak_bytes = backward_peak if math.isfinite(backward_peak) else forward_peak

    legacy_nn_elems = float(B * T * N * N)
    spatial_dim = 1 + (1 + block.pos_dim)
    factorized_elems = float(B * T * N * (rank * 2 + spatial_dim))
    return {
        "N_candidates": float(N),
        "pairwise_forward_s": forward_s,
        "pairwise_backward_s": backward_s,
        "pairwise_total_s": total_s,
        "peak_cuda_bytes": peak_bytes,
        "forward_peak_cuda_bytes": forward_peak,
        "legacy_nn_tensor_elems": legacy_nn_elems,
        "factorized_tensor_elems": factorized_elems,
        "legacy_nn_tensor_gib": legacy_nn_elems * 4.0 / (1024.0**3),
    }


def test_training_scale_benchmark() -> Dict[str, Dict[str, float]]:
    if not torch.cuda.is_available():
        return {"skipped": {"cuda": 0.0}}

    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    rank = 32
    stage3 = _benchmark_pairwise_context(_TRAIN_STAGE3, rank=rank, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stage4 = _benchmark_pairwise_context(_TRAIN_STAGE4, rank=rank, device=device)
    return {"stage3": stage3, "stage4": stage4, "factorized_rank": {"rank": float(rank)}}


def test_training_forward_backward_step() -> Dict[str, object]:
    if not torch.cuda.is_available():
        return {"skipped": 1.0}

    import sys

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from torch.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Subset

    from model.losses import compute_total_loss
    from model.model import ExplainableVidSalModel
    from pre_process.collate import video_saliency_collate_fn
    from pre_process.dataloader import DatasetLoader
    from train import (
        BACKBONE_GRADIENT_CHECKPOINTING,
        FREEZE_BACKBONE,
        MICRO_BATCH_SIZE,
        TRAIN_DATASET_DIR,
        USE_AMP,
        VISUAL_CONCEPT_LOGIT_SCALE,
        VISUAL_CONCEPT_ON,
        WINDOW_LEN,
        _effective_loss_lambda,
    )

    train_dataset = DatasetLoader(TRAIN_DATASET_DIR, window_len=WINDOW_LEN)
    loader = DataLoader(
        Subset(train_dataset, range(MICRO_BATCH_SIZE)),
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=video_saliency_collate_fn,
    )
    batch = next(iter(loader))
    _, rgb_batch, sal_batch, fix_batch, _, _ = batch

    backbone_device = torch.device(
        "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    )
    head_device = backbone_device
    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=False,
        freeze_backbone=FREEZE_BACKBONE,
        backbone_gradient_checkpointing=BACKBONE_GRADIENT_CHECKPOINTING,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=128,
        num_concepts=512,
        concept_hidden_dim=256,
        saliency_hidden_dim=256,
        visual_concept_on=VISUAL_CONCEPT_ON,
        temporal_concepts_on=False,
        visual_concept_logit_scale=VISUAL_CONCEPT_LOGIT_SCALE,
        output_activation="none",
        return_details=True,
    ).to_split_devices(backbone_device, head_device)
    model.train()

    fix_batch = (fix_batch > 0).float()
    rgb_batch, sal_batch, fix_batch = model.prepare_training_batch(
        rgb_batch,
        sal_batch,
        fix_batch,
    )

    scaler = GradScaler("cuda", enabled=USE_AMP)
    torch.cuda.reset_peak_memory_stats(backbone_device)
    torch.cuda.synchronize(backbone_device)
    t0 = time.perf_counter()

    with autocast("cuda", enabled=USE_AMP):
        model_out = model(
            rgb_batch,
            saliency_maps=sal_batch,
            return_details=True,
            return_concept_losses=True,
            return_decoder_diagnostics=True,
        )
        loss_dict = compute_total_loss(
            model_out,
            sal_batch,
            fixation_maps=fix_batch,
            enable_side_aux=True,
            **_effective_loss_lambda(epoch=1),
        )
        loss = loss_dict["loss_total"]

    scaler.scale(loss).backward()
    torch.cuda.synchronize(backbone_device)
    elapsed = time.perf_counter() - t0
    peak_bytes = torch.cuda.max_memory_allocated(backbone_device)

    pred_out = model_out.get("prediction_out")
    stage_shapes: Dict[str, Tuple[int, ...]] = {}
    if isinstance(pred_out, dict):
        priorities = pred_out.get("stage_concept_priorities", {})
        if isinstance(priorities, dict):
            for stage in ("stage3", "stage4"):
                pri = priorities.get(stage)
                if pri is not None:
                    stage_shapes[stage] = tuple(pri.shape)

    return {
        "forward_backward_s": elapsed,
        "peak_cuda_bytes": float(peak_bytes),
        "loss_total": float(loss.detach().cpu()),
        "stage3_priority_shape": stage_shapes.get("stage3"),
        "stage4_priority_shape": stage_shapes.get("stage4"),
    }


def main() -> None:
    tiny = test_explicit_matches_factorized_context()
    lambda_zero = test_lambda_zero_matches_content_only()
    grads = test_gradient_flow()
    test_invalid_candidates_are_zero()
    shapes = test_prioritization_properties()
    legacy = test_no_legacy_pairwise_code()

    print("explicit vs factorized:")
    for key, value in tiny.items():
        print(f"  {key}: {value}")
    print("lambda=0 content-only:", lambda_zero)
    print("gradient flow:", grads)
    print("prioritization shapes:", shapes)
    print("legacy scan:", legacy)

    bench = test_training_scale_benchmark()
    print("training-scale pairwise benchmark:")
    for stage, stats in bench.items():
        print(f"  {stage}: {stats}")

    train_step = test_training_forward_backward_step()
    print("training micro-batch step:")
    for key, value in train_step.items():
        print(f"  {key}: {value}")
    print("spatial factorized pairwise context unit tests: OK")


if __name__ == "__main__":
    main()
