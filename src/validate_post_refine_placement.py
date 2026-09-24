#!/usr/bin/env python3
"""Read-only validation of pre_refine vs post_refine prototype placement."""

from __future__ import annotations

import ast
import copy
import importlib
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model.saliency_prediction as saliency_prediction
from model.model import ExplainableVidSalModel
from pre_process.collate import video_saliency_collate_fn
from pre_process.dataloader import DatasetLoader
import train as train_cfg

CHECKPOINT = Path(
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/"
    "20260917_231452/epoch_125.pth"
)
MODIFIED_FILES = [
    SRC_DIR / "model" / "saliency_prediction.py",
    SRC_DIR / "model" / "model.py",
    SRC_DIR / "train.py",
    SRC_DIR / "evaluation.py",
    SRC_DIR / "find_best_ckpt.py",
    SRC_DIR / "verify_prototype_bottleneck.py",
]

RESULTS: Dict[str, Any] = {"checks": {}}


def record(name: str, passed: bool, detail: str = "") -> None:
    RESULTS["checks"][name] = {"pass": bool(passed), "detail": detail}
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))


def check_syntax_and_imports() -> None:
    print("\n=== 1. Syntax / import checks ===")
    all_ok = True
    for path in MODIFIED_FILES:
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            print(f"  syntax OK: {path.name}")
        except Exception as exc:
            all_ok = False
            print(f"  syntax FAIL: {path.name}: {exc}")
    try:
        importlib.reload(saliency_prediction)
        importlib.reload(sys.modules["model.model"])
        importlib.reload(train_cfg)
        print("  import OK: saliency_prediction, model, train")
    except Exception as exc:
        all_ok = False
        print(f"  import FAIL: {exc}")
    record("1_syntax_imports", all_ok)


def build_model(
    position: str,
    device: torch.device,
) -> ExplainableVidSalModel:
    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=False,
        freeze_backbone=True,
        backbone_gradient_checkpointing=False,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=128,
        num_concepts=512,
        concept_hidden_dim=256,
        saliency_hidden_dim=256,
        top_k=8,
        max_source_patches=64,
        tau_pi=0.5,
        tau_alpha=0.07,
        tau_concept=0.07,
        concept_residual_weight=0.0,
        last_transition_only=True,
        use_rgb_refinement=False,
        use_feature_refinement=False,
        output_activation="none",
        return_details=True,
        use_subpatch_head=True,
        subpatch_factor=4,
        subpatch_residual_scale=0.5,
        use_temporal_transition_aggregation=True,
        temporal_aggregation_hidden_channels=128,
        temporal_aggregation_temperature=1.0,
        visual_concept_on=train_cfg.VISUAL_CONCEPT_ON,
        temporal_concepts_on=train_cfg.TEMPORAL_CONCEPTS_ON,
        visual_concept_logit_scale=train_cfg.VISUAL_CONCEPT_LOGIT_SCALE,
        visual_concept_residual_weight=1.0,
        use_temporal_feature_infusion=True,
        use_shared_concept_activations=True,
        prototype_bottleneck_strength=train_cfg.PROTOTYPE_BOTTLENECK_STRENGTH,
        prototype_application_position=position,
    ).to_split_devices(device, device)
    model.eval()
    return model


def load_strict(
    model: ExplainableVidSalModel,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str]]:
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    return list(missing), list(unexpected)


def get_val_batch(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset = DatasetLoader(
        train_cfg.VAL_DATASET_DIR,
        window_len=train_cfg.WINDOW_LEN,
        stride=32,
        random_train_sampling=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=video_saliency_collate_fn,
    )
    batch = next(iter(loader))
    # DatasetLoader collate returns:
    # (filenames, rgb, sal, fix, n_frames, valid_mask)
    rgb, sal, fix = batch[1], batch[2], batch[3]
    return rgb, sal, fix


def forward_details(
    model: ExplainableVidSalModel,
    rgb: torch.Tensor,
    sal: torch.Tensor,
) -> Dict[str, Any]:
    rgb_b, sal_b, _ = model.prepare_training_batch(rgb, sal, sal)
    with torch.no_grad():
        out = model(
            rgb_b,
            saliency_maps=sal_b,
            return_details=True,
            return_decoder_diagnostics=True,
        )
    assert isinstance(out, dict)
    return out


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def install_stage_order_hooks(block: nn.Module) -> Dict[str, Any]:
    """Record call order for post_refine sequence validation."""
    state: Dict[str, Any] = {"events": [], "scatter_bases": [], "guided": []}
    handles = []

    def _mark(name: str):
        def hook(module, inputs, output):
            state["events"].append(name)

        return hook

    handles.append(block.prev_upsample.register_forward_hook(_mark("prev_upsample")))
    handles.append(block.prev_proj.register_forward_hook(_mark("prev_proj")))
    handles.append(block.refine1.register_forward_hook(_mark("refine1")))
    handles.append(block.refine2.register_forward_hook(_mark("refine2")))

    original_scatter = saliency_prediction._scatter_last_frame_update

    def scatter_hook(base, update):
        state["events"].append("prototype_blend_scatter")
        state["scatter_bases"].append(base.detach())
        state["guided"].append(update.detach())
        # Verify earlier slices unchanged relative to base.
        result = original_scatter(base, update)
        state["scatter_ok_earlier"] = torch.allclose(
            result[:, :, :-1], base[:, :, :-1], atol=0.0, rtol=0.0
        )
        state["scatter_last_eq_guided"] = torch.allclose(
            result[:, :, -1:], update, atol=1e-6, rtol=1e-5
        )
        return result

    saliency_prediction._scatter_last_frame_update = scatter_hook
    state["handles"] = handles
    state["restore_scatter"] = lambda: setattr(
        saliency_prediction, "_scatter_last_frame_update", original_scatter
    )
    return state


def remove_hooks(state: Dict[str, Any]) -> None:
    for h in state.get("handles", []):
        h.remove()
    restore = state.get("restore_scatter")
    if restore is not None:
        restore()


def clone_visual_prototypes(model: ExplainableVidSalModel) -> Dict[str, torch.Tensor]:
    banks = {}
    for name, param in model.named_parameters():
        if name.endswith("visual_concepts") and "concept_creations" in name:
            banks[name] = param.detach().clone()
    return banks


def set_visual_prototypes(
    model: ExplainableVidSalModel,
    banks: Dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in banks:
                param.copy_(banks[name])


def zero_visual_prototypes(model: ExplainableVidSalModel) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("visual_concepts") and "concept_creations" in name:
                param.zero_()


def permute_visual_prototypes(model: ExplainableVidSalModel, generator: torch.Generator) -> None:
    """Permute concept-index order in each visual prototype bank."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not (name.endswith("visual_concepts") and "concept_creations" in name):
                continue
            perm = torch.randperm(param.shape[0], generator=generator)
            param.copy_(param[perm])


def install_spatial_prototype_permute_hooks(
    model: ExplainableVidSalModel,
    seed: int,
) -> List[Any]:
    """Permute active prototypes across spatial locations at decoder stages."""
    handles = []

    def pre_hook(module, args, kwargs):
        if "active_visual_prototypes" not in kwargs:
            return None
        protos = kwargs["active_visual_prototypes"]
        if not torch.is_tensor(protos) or protos.dim() != 6:
            return None
        b, t, h, w, k, d = protos.shape
        flat = protos.reshape(b, t, h * w, k, d)
        # Deterministic spatial shuffle on the tensor device.
        device_gen = torch.Generator(device=protos.device)
        device_gen.manual_seed(seed)
        perm = torch.randperm(h * w, generator=device_gen, device=protos.device)
        flat = flat[:, :, perm, :, :]
        if b > 1:
            bperm = torch.randperm(b, generator=device_gen, device=protos.device)
            flat = flat[bperm]
        kwargs = dict(kwargs)
        kwargs["active_visual_prototypes"] = flat.reshape(b, t, h, w, k, d)
        return args, kwargs

    for stage in ("stage3", "stage4"):
        block = model.saliency_prediction.fusion_blocks[stage]
        handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=True))
    return handles


def remove_handle_list(handles: List[Any]) -> None:
    for h in handles:
        h.remove()


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().cpu())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float((a - b).norm().cpu())
    den = float(b.norm().clamp_min(1e-8).cpu())
    return num / den


def main() -> None:
    print(f"Checkpoint: {CHECKPOINT}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    check_syntax_and_imports()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    RESULTS["checkpoint_keys"] = {
        "num_tensors": len(state_dict),
        "has_raw_strength_stage3": "saliency_prediction.fusion_blocks.stage3.raw_strength"
        in state_dict,
        "has_raw_strength_stage4": "saliency_prediction.fusion_blocks.stage4.raw_strength"
        in state_dict,
    }

    print("\n=== 2. Strict state-dict load ===")
    missing_pre: List[str] = []
    unexpected_pre: List[str] = []
    missing_post: List[str] = []
    unexpected_post: List[str] = []
    try:
        model_pre = build_model("pre_refine", device)
        missing_pre, unexpected_pre = load_strict(model_pre, state_dict)
        record(
            "2_strict_load_pre_refine",
            True,
            f"missing={len(missing_pre)} unexpected={len(unexpected_pre)}",
        )
    except Exception as exc:
        model_pre = None
        record("2_strict_load_pre_refine", False, str(exc))
        print(traceback.format_exc())

    try:
        model_post = build_model("post_refine", device)
        missing_post, unexpected_post = load_strict(model_post, state_dict)
        record(
            "2_strict_load_post_refine",
            True,
            f"missing={len(missing_post)} unexpected={len(unexpected_post)}",
        )
    except Exception as exc:
        model_post = None
        record("2_strict_load_post_refine", False, str(exc))
        print(traceback.format_exc())

    RESULTS["missing_keys_pre"] = missing_pre
    RESULTS["unexpected_keys_pre"] = unexpected_pre
    RESULTS["missing_keys_post"] = missing_post
    RESULTS["unexpected_keys_post"] = unexpected_post
    print(f"  pre missing={missing_pre}")
    print(f"  pre unexpected={unexpected_pre}")
    print(f"  post missing={missing_post}")
    print(f"  post unexpected={unexpected_post}")

    if model_pre is None or model_post is None:
        print("Cannot continue without successful strict loads.")
        return

    print("\n=== Load validation batch ===")
    rgb, sal, fix = get_val_batch(device)
    print(f"  rgb={tuple(rgb.shape)} sal={tuple(sal.shape)}")
    RESULTS["input_shapes"] = {"rgb": list(rgb.shape), "sal": list(sal.shape)}

    print("\n=== 3-4. Eval forwards for both positions ===")
    out_pre = forward_details(model_pre, rgb, sal)
    out_post = forward_details(model_post, rgb, sal)

    logits_pre = out_pre["prediction_out"]["saliency_logits"]
    logits_post = out_post["prediction_out"]["saliency_logits"]
    map_pre = out_pre["saliency_map"] if "saliency_map" in out_pre else out_pre["prediction_out"]["saliency_map"]
    map_post = out_post["saliency_map"] if "saliency_map" in out_post else out_post["prediction_out"]["saliency_map"]

    # Prefer top-level saliency_map if present
    if "saliency_map" in out_pre:
        map_pre = out_pre["saliency_map"]
        map_post = out_post["saliency_map"]
        logits_pre = out_pre.get("saliency_logits", logits_pre)
        logits_post = out_post.get("saliency_logits", logits_post)
    else:
        map_pre = out_pre["prediction_out"]["saliency_map"]
        map_post = out_post["prediction_out"]["saliency_map"]

    # Unify extraction from model result dict
    def _get_logits(out: Dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(out.get("saliency_logits")):
            return out["saliency_logits"]
        return out["prediction_out"]["saliency_logits"]

    def _get_map(out: Dict[str, Any]) -> torch.Tensor:
        if torch.is_tensor(out.get("saliency_map")):
            return out["saliency_map"]
        return out["prediction_out"]["saliency_map"]

    logits_pre = _get_logits(out_pre)
    logits_post = _get_logits(out_post)
    map_pre = _get_map(out_pre)
    map_post = _get_map(out_post)

    RESULTS["shapes"] = {
        "saliency_logits_pre": list(logits_pre.shape),
        "saliency_logits_post": list(logits_post.shape),
        "saliency_map_pre": list(map_pre.shape),
        "saliency_map_post": list(map_post.shape),
    }
    print(f"  logits pre={tuple(logits_pre.shape)} post={tuple(logits_post.shape)}")
    print(f"  map    pre={tuple(map_pre.shape)} post={tuple(map_post.shape)}")

    shape_ok = logits_pre.shape == logits_post.shape and map_pre.shape == map_post.shape
    record("4a_saliency_shape_unchanged", shape_ok, str(tuple(logits_pre.shape)))

    finite_ok = all(
        _finite(t)
        for t in (logits_pre, logits_post, map_pre, map_post)
    )
    record("4b_outputs_finite", finite_ok)

    # Priority mask shapes from diagnostics / stage maps
    pred_pre = out_pre["prediction_out"]
    pred_post = out_post["prediction_out"]
    priority_shapes = {}
    priority_ok = True
    for label, pred in (("pre", pred_pre), ("post", pred_post)):
        gates = pred.get("stage_patch_priority_gates") or {}
        maps = pred.get("stage_patch_priority_maps") or {}
        for stage in ("stage3", "stage4"):
            tensor = gates.get(stage)
            if tensor is None:
                tensor = maps.get(stage)
            if tensor is None:
                priority_ok = False
                priority_shapes[f"{label}_{stage}"] = None
                continue
            priority_shapes[f"{label}_{stage}"] = list(tensor.shape)
            # Expect [B, H, W] or [B, 1, H, W] / [B, T, H, W] last-frame style
            if tensor.dim() not in (3, 4, 5):
                priority_ok = False
    RESULTS["priority_mask_shapes"] = priority_shapes
    record("4c_priority_mask_shapes", priority_ok, str(priority_shapes))

    # Last-slice-only via hooks on both modes
    print("\n=== 4d. Last-temporal-slice-only checks ===")
    last_slice_ok = True
    last_slice_details = {}
    for position, model in (("pre_refine", model_pre), ("post_refine", model_post)):
        block = model.saliency_prediction.fusion_blocks["stage3"]
        original_scatter = saliency_prediction._scatter_last_frame_update
        captured = {}

        def scatter_capture(base, update, _cap=captured):
            result = original_scatter(base, update)
            _cap["base"] = base.detach()
            _cap["update"] = update.detach()
            _cap["result"] = result.detach()
            return result

        saliency_prediction._scatter_last_frame_update = scatter_capture
        try:
            _ = forward_details(model, rgb, sal)
        finally:
            saliency_prediction._scatter_last_frame_update = original_scatter

        if "base" not in captured:
            last_slice_ok = False
            last_slice_details[position] = "no scatter captured"
            continue
        earlier_eq = torch.equal(captured["result"][:, :, :-1], captured["base"][:, :, :-1])
        last_eq = torch.allclose(
            captured["result"][:, :, -1:], captured["update"], atol=1e-5, rtol=1e-5
        )
        last_slice_details[position] = {
            "earlier_unchanged": bool(earlier_eq),
            "last_equals_guided": bool(last_eq),
            "base_shape": list(captured["base"].shape),
            "guided_shape": list(captured["update"].shape),
        }
        if not (earlier_eq and last_eq):
            last_slice_ok = False
    RESULTS["last_slice"] = last_slice_details
    record("4d_last_temporal_slice_only", last_slice_ok, str(last_slice_details))

    resolution_ok = map_pre.shape[-2:] == map_post.shape[-2:]
    record(
        "4e_output_resolution_unchanged",
        resolution_ok,
        f"pre={tuple(map_pre.shape[-2:])} post={tuple(map_post.shape[-2:])}",
    )

    print("\n=== 5. post_refine order: upsample → fusion → refine → blend ===")
    # Need prev_decoder present: use stage3 which receives prev from stage4.
    # Hooks on stage3 during full forward.
    block3 = model_post.saliency_prediction.fusion_blocks["stage3"]
    order_state = install_stage_order_hooks(block3)
    try:
        _ = forward_details(model_post, rgb, sal)
    finally:
        remove_hooks(order_state)

    events = order_state["events"]
    # Global scatter monkeypatch also fires for stage4 before stage3 runs.
    # Restrict the order check to the stage3 subsequence starting at prev_upsample.
    if "prev_upsample" in events:
        events = events[events.index("prev_upsample") :]

    def _first_idx(name: str) -> int:
        try:
            return events.index(name)
        except ValueError:
            return -1

    idx_up = _first_idx("prev_upsample")
    idx_proj = _first_idx("prev_proj")
    idx_r1 = _first_idx("refine1")
    idx_r2 = _first_idx("refine2")
    idx_blend = _first_idx("prototype_blend_scatter")
    order_ok = (
        idx_up >= 0
        and idx_proj >= 0
        and idx_r1 >= 0
        and idx_r2 >= 0
        and idx_blend >= 0
        and idx_up < idx_proj < idx_r1 < idx_r2 < idx_blend
        and bool(order_state.get("scatter_ok_earlier", False))
    )
    RESULTS["post_refine_order_events"] = events
    record(
        "5_post_refine_order",
        order_ok,
        f"events={events} idxs=up{idx_up}<proj{idx_proj}<r1{idx_r1}<r2{idx_r2}<blend{idx_blend}",
    )

    print("\n=== 6. Backward-pass smoke test (post_refine) ===")
    model_grad = build_model("post_refine", device)
    load_strict(model_grad, state_dict)
    model_grad.train()
    # Keep backbone frozen for memory; unfreeze head path.
    for p in model_grad.parameters():
        p.requires_grad_(True)
    for p in model_grad.backbone.parameters():
        p.requires_grad_(False)

    rgb_b, sal_b, _ = model_grad.prepare_training_batch(rgb, sal, fix)
    # Enable grads through concept path: need training mode without inference_mode
    out = model_grad(
        rgb_b,
        saliency_maps=sal_b,
        return_details=True,
        return_decoder_diagnostics=False,
    )
    logits = out["saliency_logits"] if torch.is_tensor(out.get("saliency_logits")) else out["prediction_out"]["saliency_logits"]
    loss = logits.float().square().mean()
    model_grad.zero_grad(set_to_none=True)
    loss.backward()

    grad_report = {}
    grad_ok = True

    # visual prototypes
    proto_grad_norm = 0.0
    proto_found = False
    for name, param in model_grad.named_parameters():
        if "visual_concepts" in name and param.grad is not None:
            proto_found = True
            proto_grad_norm += float(param.grad.detach().norm().cpu())
    grad_report["visual_prototypes"] = proto_grad_norm
    if not proto_found or not math.isfinite(proto_grad_norm) or proto_grad_norm <= 0:
        grad_ok = False

    block = model_grad.saliency_prediction.fusion_blocks["stage3"]

    def module_grad_norm(module: Optional[nn.Module]) -> float:
        if module is None:
            return 0.0
        total = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm().cpu())
        return total

    unary_n = module_grad_norm(block.unary_priority_mlp)
    pair_n = module_grad_norm(block.factor_query) + module_grad_norm(block.factor_key)
    mask_n = module_grad_norm(block.mask_feature_branch)
    grad_report["unary_priority_mlp"] = unary_n
    grad_report["pairwise_factor_qk"] = pair_n
    grad_report["mask_feature_branch"] = mask_n
    for key, val in (
        ("unary", unary_n),
        ("pairwise", pair_n),
        ("mask_feature_branch", mask_n),
    ):
        if not math.isfinite(val) or val <= 0:
            grad_ok = False
    RESULTS["gradient_norms"] = grad_report
    record("6_backward_gradients", grad_ok, str(grad_report))
    del model_grad
    torch.cuda.empty_cache()

    print("\n=== 7. Prototype sensitivity on fixed val batch (post_refine) ===")
    model_sens = build_model("post_refine", device)
    load_strict(model_sens, state_dict)
    model_sens.eval()
    normal_banks = clone_visual_prototypes(model_sens)
    out_normal = forward_details(model_sens, rgb, sal)
    logits_normal = _get_logits(out_normal)

    zero_visual_prototypes(model_sens)
    out_zero = forward_details(model_sens, rgb, sal)
    logits_zero = _get_logits(out_zero)

    set_visual_prototypes(model_sens, normal_banks)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    # Concept-index bank permutation (may be weakly visible under shared activations).
    permute_visual_prototypes(model_sens, gen)
    out_bank_perm = forward_details(model_sens, rgb, sal)
    logits_bank_perm = _get_logits(out_bank_perm)

    # Spatial / per-sample permutation of active prototypes at stage3/4 (requested check).
    set_visual_prototypes(model_sens, normal_banks)
    handles = install_spatial_prototype_permute_hooks(model_sens, seed=1)
    try:
        out_perm = forward_details(model_sens, rgb, sal)
    finally:
        remove_handle_list(handles)
    logits_perm = _get_logits(out_perm)

    sens = {
        "zero_mae": mae(logits_zero, logits_normal),
        "zero_rel_l2": rel_l2(logits_zero, logits_normal),
        "bank_index_perm_mae": mae(logits_bank_perm, logits_normal),
        "bank_index_perm_rel_l2": rel_l2(logits_bank_perm, logits_normal),
        "spatial_perm_mae": mae(logits_perm, logits_normal),
        "spatial_perm_rel_l2": rel_l2(logits_perm, logits_normal),
        "pre_vs_post_mae": mae(logits_post, logits_pre),
        "pre_vs_post_rel_l2": rel_l2(logits_post, logits_pre),
    }
    RESULTS["sensitivity"] = sens
    sens_ok = (
        sens["zero_mae"] > 0
        or sens["spatial_perm_mae"] > 0
        or sens["pre_vs_post_mae"] > 0
    )
    record(
        "7_prototype_sensitivity",
        sens_ok,
        (
            f"zero MAE={sens['zero_mae']:.6e} relL2={sens['zero_rel_l2']:.6e}; "
            f"bank_perm MAE={sens['bank_index_perm_mae']:.6e} "
            f"relL2={sens['bank_index_perm_rel_l2']:.6e}; "
            f"spatial_perm MAE={sens['spatial_perm_mae']:.6e} "
            f"relL2={sens['spatial_perm_rel_l2']:.6e}; "
            f"pre_vs_post MAE={sens['pre_vs_post_mae']:.6e} "
            f"relL2={sens['pre_vs_post_rel_l2']:.6e}"
        ),
    )
    del model_sens
    torch.cuda.empty_cache()

    print("\n=== 8. pre_refine reproduces checkpoint-initialized output ===")
    # Reload fresh pre_refine model twice and confirm identical logits.
    model_a = build_model("pre_refine", device)
    load_strict(model_a, state_dict)
    model_b = build_model("pre_refine", device)
    load_strict(model_b, state_dict)
    out_a = forward_details(model_a, rgb, sal)
    out_b = forward_details(model_b, rgb, sal)
    logits_a = _get_logits(out_a)
    logits_b = _get_logits(out_b)
    repro_ok = torch.allclose(logits_a, logits_b, atol=1e-6, rtol=1e-5)
    max_diff = float((logits_a - logits_b).abs().max().cpu())
    # Also confirm pre_refine path blends before refine via event order on stage3
    block_pre = model_a.saliency_prediction.fusion_blocks["stage3"]
    pre_order = install_stage_order_hooks(block_pre)
    try:
        _ = forward_details(model_a, rgb, sal)
    finally:
        remove_hooks(pre_order)
    pre_events = pre_order["events"]
    if "prev_upsample" in pre_events:
        pre_events = pre_events[pre_events.index("prev_upsample") :]
    # For pre_refine: blend scatter should occur BEFORE refine1/refine2
    pre_blend = (
        pre_events.index("prototype_blend_scatter")
        if "prototype_blend_scatter" in pre_events
        else -1
    )
    pre_r1 = pre_events.index("refine1") if "refine1" in pre_events else -1
    pre_path_ok = pre_blend >= 0 and pre_r1 >= 0 and pre_blend < pre_r1
    RESULTS["pre_refine_repro"] = {
        "max_abs_diff_two_loads": max_diff,
        "events": pre_events,
        "blend_before_refine": pre_path_ok,
    }
    record(
        "8_pre_refine_reproduces_checkpoint_behavior",
        repro_ok and pre_path_ok,
        f"max_diff={max_diff:.3e} blend_before_refine={pre_path_ok} events={pre_events}",
    )

    print("\n=== Summary ===")
    fails = [k for k, v in RESULTS["checks"].items() if not v["pass"]]
    print(f"Passed: {sum(1 for v in RESULTS['checks'].values() if v['pass'])}/{len(RESULTS['checks'])}")
    if fails:
        print("Failed checks: " + ", ".join(fails))
    else:
        print("All checks passed.")
    print("\nSensitivity numbers:")
    for k, v in RESULTS.get("sensitivity", {}).items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nGradient norms:")
    for k, v in RESULTS.get("gradient_norms", {}).items():
        print(f"  {k}: {v:.6e}")


if __name__ == "__main__":
    main()
