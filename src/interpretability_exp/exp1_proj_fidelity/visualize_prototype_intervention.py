#!/usr/bin/env python3
"""Trace and intervene on one visual prototype for one validation window.

Place alongside create_random_prototype_checkpoint.py in
src/interpretability_exp/exp1_proj_fidelity/ and run from any directory.
The existing checkpoint script supplies the exact training-model constructor,
checkpoint loader, and deterministic/device settings.

This intervention masks one global prototype ID out of the selected stage's
priority competition. It does not change the concept encoder, its top-k search,
the decoder weights, or the other stages. The mask is recomputed from the
remaining valid candidates, including pairwise context at stages 3 and 4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any


STAGES = ("stage1", "stage2", "stage3", "stage4")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--helper-script", type=Path,
                   default=Path(__file__).with_name("create_random_prototype_checkpoint.py"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--random-checkpoint", type=Path, default=None,
                   help="Random prototype checkpoint (default: helper's DEFAULT_OUTPUT).")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--window-len", type=int, default=None)
    p.add_argument("--sample-index", type=int, default=1000)
    p.add_argument("--stage", choices=STAGES, default="stage3")
    p.add_argument("--prototype-id", type=int, default=None,
                   help="Global bank index. Default: highest total priority at this stage/frame.")
    p.add_argument("--top-patches", type=int, default=3)
    p.add_argument("--prototype-patch", type=Path, default=None,
                   help="Optional image of this prototype's projected TRAINING patch.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("random_prot_exp_qual"))
    args = p.parse_args()
    if args.sample_index < 0 or args.top_patches < 1:
        p.error("sample-index must be nonnegative and top-patches must be positive")
    return args


def load_helper(path: Path) -> Any:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostic helper not found: {path}")
    spec = importlib.util.spec_from_file_location("prototype_checkpoint_diagnostic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("build_diagnostic_model", "load_model_checkpoint", "_set_deterministic",
                 "_resolve_device", "_ensure_src_on_path"):
        if not hasattr(module, name):
            raise RuntimeError(f"{path} is missing {name}; use your diagnostic checkpoint script")
    module._ensure_src_on_path()
    return module


@contextmanager
def capture_decoder(model: Any):
    """Retain the decoder's actual inputs/outputs, regardless of outer model layout."""
    import torch

    decoder = model.saliency_prediction
    captured: dict[str, Any] = {}

    def before(_module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        concept_outs = inputs[0] if inputs else kwargs.get("concept_outs")
        if not isinstance(concept_outs, dict):
            raise RuntimeError("Decoder did not receive a concept_outs dict")
        captured["concept_outs"] = concept_outs

    def after(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, dict) or not torch.is_tensor(output.get("saliency_map")):
            raise RuntimeError("Decoder did not return a dict with saliency_map")
        captured["output"] = output

    first = decoder.register_forward_pre_hook(before, with_kwargs=True)
    last = decoder.register_forward_hook(after)
    try:
        yield captured
    finally:
        first.remove()
        last.remove()


def stage_tensors(captured: dict[str, Any], stage: str) -> dict[str, Any]:
    import torch

    out = captured["output"]
    concept = captured["concept_outs"][stage]
    indices = concept["active_visual_prototype_indices"]
    valid = concept["visual_validity_mask"]
    priorities = out["stage_concept_priorities"][stage]
    masks = out["stage_priority_masks"][stage]
    unary = out["stage_unary_scores"][stage]
    if not all(torch.is_tensor(t) for t in (indices, valid, priorities, masks, unary)):
        raise RuntimeError("Missing per-stage prototype diagnostic tensors; use return_details=True")
    if indices.ndim != 5 or valid.shape != indices.shape:
        raise RuntimeError(f"Expected [B,T,H,W,K] indices/validity, got {indices.shape}")
    if priorities.ndim != 5 or priorities.shape[2] != indices.shape[-1]:
        raise RuntimeError(f"Expected [B,T,K,H,W] priorities, got {priorities.shape}")
    # Some model revisions retain full-window concept matches although the
    # priority mask only applies to the final temporal slice.
    indices, valid = indices[:, -1:], valid[:, -1:]
    priorities, unary = priorities[:, -1:], unary[:, -1:]
    if tuple(indices.shape[:2]) != tuple(priorities.shape[:2]) or \
            tuple(indices.shape[2:4]) != tuple(priorities.shape[3:5]):
        raise RuntimeError(f"Indices {indices.shape} and priorities {priorities.shape} misaligned")
    if int(indices.shape[0]) != 1 or masks.ndim != 5:
        raise RuntimeError("This diagnostic requires batch size one and [B,1,T,H,W] masks")
    return {"indices": indices, "valid": valid, "priorities": priorities,
            "masks": masks, "unary": unary,
            "context": out.get("stage_pairwise_context", {}).get(stage),
            "activations": out.get("stage_concept_patch_activations", {}).get(stage),
            "concept": concept}


def choose_prototype(data: dict[str, Any], chosen: int | None) -> int:
    import torch

    indices = data["indices"][0, -1]  # [H,W,K]
    valid = data["valid"][0, -1].bool()
    priority = data["priorities"][0, -1].permute(1, 2, 0)
    if chosen is not None:
        if not bool(((indices == chosen) & valid).any()):
            raise ValueError(f"Prototype {chosen} is not active on this stage's last frame")
        return chosen
    active_indices = torch.unique(indices[valid]).tolist()
    if not active_indices:
        raise RuntimeError("No active visual prototypes in the selected frame")
    return int(max(active_indices, key=lambda i: float(
        priority[(indices == i) & valid].sum().item())))


@contextmanager
def exclude_from_stage_priority(model: Any, stage: str, prototype_id: int,
                                concept_outs: dict[str, Any]):
    """Remove this ID from scoring/softmax/context; restore the original method."""
    import torch

    block = model.saliency_prediction.fusion_blocks[stage]
    original = block._prioritize_concept_patch_activations
    concept = concept_outs[stage]
    indices = concept["active_visual_prototype_indices"]
    eligible = concept["visual_validity_mask"].bool()
    selected = torch.zeros_like(eligible)
    selected[:, -1:] = (indices[:, -1:] == prototype_id) & eligible[:, -1:]
    count = int(selected.sum().item())
    if count == 0:
        raise RuntimeError("Selected prototype has no eligible occurrences at this stage")
    remaining = eligible & ~selected
    # _masked_softmax scores all concept/patch candidates together; a patch
    # with no remaining candidates gets zero priority without NaNs. The
    # decoder's concept mixture also uses a clamped denominator in this case.
    if not bool(remaining.any(dim=(-1, -2, -3)).all()):
        raise RuntimeError("Removing this prototype leaves the entire frame without candidates")
    empty_patches = int((~remaining[:, -1:].any(dim=-1)).sum().item())
    calls = {"count": 0, "removed": count, "patches_without_remaining": empty_patches}
    previous_instance_method = block.__dict__.get("_prioritize_concept_patch_activations")

    def patched(_self: Any, modalities: list[dict[str, Any]]) -> dict[str, Any]:
        if len(modalities) != 1:
            raise RuntimeError("Expected exactly one visual modality for the selected stage")
        visual = modalities[0]
        target = selected if visual["validity_mask"].shape == selected.shape else selected[:, -1:]
        if visual["validity_mask"].shape != target.shape:
            raise RuntimeError("Concept IDs do not align with priority inputs")
        updated = dict(visual)
        updated["validity_mask"] = visual["validity_mask"].bool() & ~target
        updated["activations"] = visual["activations"].masked_fill(
            target.permute(0, 1, 4, 2, 3), 0.0)
        calls["count"] += 1
        return original([updated])

    block._prioritize_concept_patch_activations = MethodType(patched, block)
    try:
        yield calls
    finally:
        if previous_instance_method is None:
            del block._prioritize_concept_patch_activations
        else:
            block._prioritize_concept_patch_activations = previous_instance_method


def get_map(captured: dict[str, Any]):
    import torch

    x = captured["output"]["saliency_map"]
    if not torch.is_tensor(x) or x.ndim != 4 or x.shape[0:2] != (1, 1):
        raise RuntimeError(f"Expected saliency_map [1,1,H,W], got {getattr(x, 'shape', None)}")
    return x.detach().cpu()


def metric_numbers(metrics: dict[str, Any]) -> dict[str, float]:
    import torch

    result = {}
    for name in ("CC", "SIM", "NSS"):
        matches = [v for k, v in metrics.items() if str(k).upper() == name]
        if len(matches) != 1:
            raise KeyError(f"Expected {name} in metrics: {list(metrics)}")
        value = matches[0]
        result[name] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
    return result


def comparison(a: Any, b: Any) -> dict[str, float | None]:
    import torch

    a, b = a.float().reshape(-1), b.float().reshape(-1)
    d = a - b
    denom = max(float(a.norm()), float(b.norm()), 1e-12)
    corr = (float(torch.corrcoef(torch.stack((a, b)))[0, 1])
            if a.numel() > 1 and float(a.std()) > 1e-12 and float(b.std()) > 1e-12
            else None)
    return {"mae": float(d.abs().mean()), "relative_L2": float(d.norm()) / denom,
            "map_correlation": corr}


def image_array(frame: Any):
    """Display raw last RGB frame (not model-normalized/resized tensor)."""
    import numpy as np

    arr = frame.detach().float().cpu().permute(1, 2, 0).numpy()
    if arr.min() < 0:
        arr = arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    elif arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0, 1)


def display_map(tensor: Any):
    import numpy as np
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError("Expected tensor for display")
    return np.asarray(tensor.detach().float().cpu().squeeze().numpy())


def training_patch(path: Path | None):
    if path is None:
        return None
    from PIL import Image
    if not path.is_file():
        raise FileNotFoundError(f"Projected training patch not found: {path}")
    return Image.open(path).convert("RGB")


def save_prediction_files(directory: Path, prefix: str, original: Any,
                          random_map: Any, prepare: Any) -> dict[str, str]:
    """Save raw arrays and evaluation-prepared 0–1 grayscale PNGs."""
    import numpy as np
    from PIL import Image

    files: dict[str, str] = {}
    for label, tensor in (("original", original), ("random", random_map)):
        raw = display_map(tensor)
        raw_file = directory / f"{prefix}_{label}_raw.npy"
        np.save(raw_file, raw)
        files[f"{label}_raw_numpy"] = str(raw_file)
        prepared = display_map(prepare(tensor))
        if prepared.ndim != 2 or not np.isfinite(prepared).all():
            raise RuntimeError(f"{label} eval-prepared prediction is invalid")
        prepared_file = directory / f"{prefix}_{label}_eval.npy"
        np.save(prepared_file, prepared)
        files[f"{label}_eval_numpy"] = str(prepared_file)
        png = directory / f"{prefix}_{label}_eval.png"
        Image.fromarray(np.round(np.clip(prepared, 0, 1) * 255).astype(np.uint8),
                        mode="L").save(png)
        files[f"{label}_eval_png"] = str(png)
    return files


def last_rgb(rgb: Any):
    import torch
    if not torch.is_tensor(rgb) or rgb.ndim != 5 or rgb.shape[0] != 1:
        raise RuntimeError(f"Expected raw RGB batch [1,T,3,H,W], got {getattr(rgb, 'shape', None)}")
    if rgb.shape[2] == 3:
        return image_array(rgb[0, -1])
    if rgb.shape[1] == 3:
        return image_array(rgb[0, :, -1])
    raise RuntimeError("Cannot locate RGB channel dimension in collated validation batch")


def patch_records(data: dict[str, Any], prototype_id: int, bank: Any,
                  limit: int, frame: Any) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    idx = data["indices"][0, -1].detach().cpu()
    valid = data["valid"][0, -1].detach().cpu().bool()
    priorities = data["priorities"][0, -1].detach().cpu().permute(1, 2, 0)
    unary = data["unary"][0, -1].detach().cpu().permute(1, 2, 0)
    ctx = data["context"]
    if torch.is_tensor(ctx):
        ctx = ctx[0, -1].detach().cpu().permute(1, 2, 0)
    activation = data["activations"]
    if torch.is_tensor(activation):
        activation = activation[0, -1].detach().cpu().permute(1, 2, 0)
    chosen = (idx == prototype_id) & valid
    coords = chosen.nonzero(as_tuple=False)
    coords = sorted((tuple(int(v) for v in coord) for coord in coords),
                    key=lambda pos: float(priorities[pos]), reverse=True)[:limit]
    h, w, _ = idx.shape
    image_h, image_w = frame.shape[:2]
    emb = data["concept"].get("visual_patch_embeddings")
    if torch.is_tensor(emb):
        emb = emb.detach().cpu()
        # Concept creation stores flattened [B*T*H*W,D] embeddings.
        if emb.numel() % (h * w) != 0 or emb.ndim not in (2, 5):
            raise RuntimeError(f"Unexpected visual_patch_embeddings shape {emb.shape}")
        emb = emb[-h * w:].reshape(h, w, -1)
    prototype = bank[prototype_id].detach().float().cpu()
    records = []
    for y, x, slot in coords:
        box = [int(x * image_w / w), int(y * image_h / h),
               max(1, int((x + 1) * image_w / w)),
               max(1, int((y + 1) * image_h / h))]
        rec = {"patch_yx": [y, x], "topk_slot": slot,
               "image_box_xyxy_approx": box,
               "prototype_priority": float(priorities[y, x, slot]),
               "unary_score": float(unary[y, x, slot]),
               "context_score": float(ctx[y, x, slot]) if ctx is not None else None,
               "concept_activation": float(activation[y, x, slot])
               if activation is not None else None}
        if emb is not None:
            vector = emb[y, x].float()
            if vector.numel() == prototype.numel():
                rec["cosine_patch_to_learned_prototype"] = float(
                    F.cosine_similarity(vector[None], prototype[None], dim=-1)[0])
            else:
                raise RuntimeError("Patch embedding and learned prototype dimensions differ")
        records.append(rec)
    return records


def save_figure(path: Path, frame: Any, gt: Any, base_map: Any,
                changed_map: Any, base_data: dict[str, Any],
                changed_data: dict[str, Any], prototype_id: int,
                prototype_vector: Any, patches: list[dict[str, Any]],
                metrics: dict[str, dict[str, float]], projected_patch: Any) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(21, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 5, height_ratios=[1, 1, .8])
    orig = display_map(base_map)
    intervention = display_map(changed_map)
    lo, hi = float(min(orig.min(), intervention.min())), float(max(orig.max(), intervention.max()))

    image_ax = fig.add_subplot(grid[0, 0]); image_ax.imshow(frame)
    image_ax.set_title("Input: final validation frame")
    colors = ("lime", "cyan", "yellow", "magenta")
    for j, rec in enumerate(patches):
        x0, y0, x1, y1 = rec["image_box_xyxy_approx"]
        image_ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                     fill=False, ec=colors[j % 4], lw=2))
        image_ax.text(x0, y0, f"{j+1}", color="black", weight="bold",
                      bbox={"facecolor": colors[j % 4], "pad": 1})
    image_ax.axis("off")

    def panel(position, array, title, *, cmap="magma", vmin=None, vmax=None):
        ax = fig.add_subplot(grid[position]); im = ax.imshow(array, cmap=cmap,
                                                            vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=15); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=.044, pad=.02)
        return ax

    panel((0, 1), display_map(gt), "Ground-truth density")
    a, b = metrics["original"], metrics["intervention"]
    panel((0, 2), orig, f"Original\nCC {a['CC']:.3f}, SIM {a['SIM']:.3f}, NSS {a['NSS']:.3f}",
          vmin=lo, vmax=hi)
    panel((0, 3), intervention,
          f"Excluded\nCC {b['CC']:.3f}, SIM {b['SIM']:.3f}, NSS {b['NSS']:.3f}",
          vmin=lo, vmax=hi)
    ax = fig.add_subplot(grid[0, 4]); ax.axis("off")
    ax.text(.05, .6, f"Prototype #{prototype_id}\n"
            f"CC change: {b['CC'] - a['CC']:+.3f}\n"
            f"SIM change: {b['SIM'] - a['SIM']:+.3f}\n"
            f"NSS change: {b['NSS'] - a['NSS']:+.3f}",
            transform=ax.transAxes, fontsize=15)

    gate_a = display_map(base_data["masks"][0, :, -1])
    gate_b = display_map(changed_data["masks"][0, :, -1])
    gate_lo, gate_hi = float(min(gate_a.min(), gate_b.min())), float(max(gate_a.max(), gate_b.max()))
    panel((1, 0), gate_a, "Original stage priority mask", vmin=gate_lo, vmax=gate_hi)
    panel((1, 1), gate_b, "Mask after excluding prototype", vmin=gate_lo, vmax=gate_hi)
    panel((1, 2), gate_b - gate_a, "Priority mask change", cmap="coolwarm")
    panel((1, 3), intervention - orig, "Final prediction change", cmap="coolwarm")
    # Keep integer prototype IDs and Boolean validity: display_map casts to
    # float for plotting and floats cannot be combined with '&' here.
    local_indices = base_data["indices"][0, -1].detach().cpu().numpy()
    local_valid = base_data["valid"][0, -1].detach().cpu().numpy().astype(bool)
    local_priority = display_map(base_data["priorities"][0, -1]).transpose(1, 2, 0)
    contribution = np.where((local_indices == prototype_id) & local_valid,
                            local_priority, 0).sum(axis=-1)
    panel((1, 4), contribution, "Selected prototype's priority")

    for j in range(3):
        ax = fig.add_subplot(grid[2, j])
        if j < len(patches):
            rec = patches[j]; x0, y0, x1, y1 = rec["image_box_xyxy_approx"]
            ax.imshow(frame[y0:y1, x0:x1])
            sim = rec.get("cosine_patch_to_learned_prototype")
            sim_label = f"{sim:.3f}" if sim is not None else "n/a"
            context_label = (f"{rec['context_score']:.3f}"
                             if rec["context_score"] is not None else "n/a")
            ax.set_title(f"Matched input patch {j+1} | cos {sim_label}\n"
                         f"unary {rec['unary_score']:.3f}, "
                         f"context {context_label}\n"
                         f"priority {rec['prototype_priority']:.4f}", fontsize=15)
        ax.axis("off")
    ax = fig.add_subplot(grid[2, 3])
    v = display_map(prototype_vector).reshape(1, -1)
    ax.imshow(v, cmap="coolwarm", aspect="auto", vmin=-max(abs(v.min()), abs(v.max())),
              vmax=max(abs(v.min()), abs(v.max())))
    ax.set_yticks([]); ax.set_xlabel("Embedding dimension")
    ax.set_title(f"Learned prototype #{prototype_id}\n(feature vector, not RGB)")
    ax = fig.add_subplot(grid[2, 4]); ax.axis("off")
    if projected_patch is not None:
        ax.imshow(projected_patch)
        ax.set_title("Projected TRAINING patch (supplied)")
    else:
        ax.text(.05, .55, "Training-patch image unavailable.\n"
                "Use --prototype-patch with the saved\n"
                "projection image for this prototype.", transform=ax.transAxes)
    fig.suptitle("Prototype contribution to one saliency prediction", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = arguments()
    helper = load_helper(args.helper_script)
    import torch
    import train as train_cfg
    from metrics import compute_saliency_metrics, prepare_prediction_map
    from pre_process.collate import video_saliency_collate_fn
    from pre_process.dataloader import DatasetLoader

    helper._set_deterministic(args.seed)
    device = helper._resolve_device(args.device)
    checkpoint = (args.checkpoint or helper.DEFAULT_CHECKPOINT).expanduser().resolve()
    random_checkpoint = (args.random_checkpoint or helper.DEFAULT_OUTPUT).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not random_checkpoint.is_file():
        raise FileNotFoundError(f"Random checkpoint not found: {random_checkpoint}; "
                                "generate it first or pass --random-checkpoint")
    dataset_dir = (args.dataset_dir or Path(train_cfg.VAL_DATASET_DIR)).expanduser().resolve()
    window_len = args.window_len or int(train_cfg.WINDOW_LEN)
    dataset = DatasetLoader(str(dataset_dir), window_len=window_len, stride=32)
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample-index must be < {len(dataset)}")
    sample = dataset[args.sample_index]
    _, rgb, sal, fix, n_frames, _ = video_saliency_collate_fn([sample])
    if sal is None or fix is None:
        raise ValueError("Selected sample has no saliency/fixation ground truth")
    model = helper.build_diagnostic_model(device)
    helper.load_model_checkpoint(model, checkpoint)
    model.eval()
    rgb_ready, sal_ready, fix_ready = model.prepare_training_batch(rgb, sal, fix)
    if sal_ready is None or fix_ready is None:
        raise RuntimeError("Preprocessing did not return ground truth")
    fixation = (fix_ready.detach().cpu() > 0).float()
    gt = sal_ready.detach().cpu()
    frame = last_rgb(rgb)
    forward_kwargs = {"saliency_maps": sal_ready, "return_details": True,
                      "return_concept_losses": False,
                      "return_decoder_diagnostics": True}

    with torch.inference_mode():
        with capture_decoder(model) as base:
            model(rgb_ready, **forward_kwargs)
    initial = stage_tensors(base, args.stage)
    prototype_id = choose_prototype(initial, args.prototype_id)
    bank = model.concept_creations[args.stage].visual_concepts
    if not (0 <= prototype_id < bank.shape[0]):
        raise IndexError(f"Prototype {prototype_id} outside bank {bank.shape}")
    vector = bank[prototype_id].detach().cpu()
    records = patch_records(initial, prototype_id, bank, args.top_patches, frame)

    helper._set_deterministic(args.seed)
    with exclude_from_stage_priority(model, args.stage, prototype_id,
                                     base["concept_outs"]) as changed_calls:
        with torch.inference_mode():
            with capture_decoder(model) as changed:
                model(rgb_ready, **forward_kwargs)
    if changed_calls["count"] != 1:
        raise RuntimeError(f"Expected one selected-stage intervention, got {changed_calls}")
    modified = stage_tensors(changed, args.stage)
    original_map, changed_map = get_map(base), get_map(changed)
    if original_map.shape != changed_map.shape:
        raise RuntimeError("Predicted map shapes do not match")
    orig_gate = initial["masks"][:, :, -1].detach().cpu()
    changed_gate = modified["masks"][:, :, -1].detach().cpu()
    gate_max_delta = float((orig_gate - changed_gate).abs().max())
    suppressed_slots = ((initial["indices"][0, -1] == prototype_id)
                        & initial["valid"][0, -1].bool())
    modified_priorities = modified["priorities"][0, -1].permute(1, 2, 0)
    if not bool((modified_priorities[suppressed_slots] == 0).all()):
        raise RuntimeError("Selected prototype still has nonzero priority after exclusion")

    metric_args = {"fixation_target": fixation,
                   "allow_pseudo_fixations": False, "dh1k_exact": True}
    before_metrics = metric_numbers(compute_saliency_metrics(original_map, gt, **metric_args))
    after_metrics = metric_numbers(compute_saliency_metrics(changed_map, gt, **metric_args))
    result = {
        "checkpoint": str(checkpoint), "dataset_dir": str(dataset_dir),
        "sample_index": args.sample_index, "video": str(dataset.windows[args.sample_index][0]),
        "start_frame": int(dataset.windows[args.sample_index][1]),
        "n_frames": int(n_frames[0]), "stage": args.stage,
        "prototype_id": prototype_id,
        "prototype_vector_l2": float(vector.float().norm()),
        "intervention": "exclude selected global prototype ID from this stage's priority competition",
        "removed_candidate_occurrences": changed_calls["removed"],
        "patches_without_remaining_prototypes": changed_calls["patches_without_remaining"],
        "priority_mask_max_abs_change": gate_max_delta,
        "metrics": {"original": before_metrics, "intervention": after_metrics,
                    "delta_intervention_minus_original": {
                        name: after_metrics[name] - before_metrics[name]
                        for name in ("CC", "SIM", "NSS")}},
        "map_comparison_evaluation_prepared": comparison(
            prepare_prediction_map(original_map), prepare_prediction_map(changed_map)),
        "top_matched_patches": records,
        "projected_training_patch": str(args.prototype_patch.expanduser().resolve())
        if args.prototype_patch else None,
        "note": "Displayed input crops approximate feature-grid cells, not full CNN receptive fields."
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"sample_{args.sample_index:05d}_{args.stage}_prototype_{prototype_id}"
    figure = args.output_dir / f"{prefix}.png"
    report = args.output_dir / f"{prefix}.json"
    save_figure(figure, frame, gt, original_map, changed_map, initial, modified,
                prototype_id, vector, records, result["metrics"],
                training_patch(args.prototype_patch.expanduser().resolve()
                               if args.prototype_patch else None))

    # Release detailed stage tensors and the first model before loading another
    # full copy of the backbone on GPU.
    del initial, modified, base, changed, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    helper._set_deterministic(args.seed)
    random_model = helper.build_diagnostic_model(device)
    helper.load_model_checkpoint(random_model, random_checkpoint)
    random_model.eval()
    with torch.inference_mode():
        with capture_decoder(random_model) as random_capture:
            random_model(rgb_ready, **forward_kwargs)
    random_map = get_map(random_capture)
    if random_map.shape != original_map.shape:
        raise RuntimeError("Random checkpoint returned a different saliency map size")
    random_metrics = metric_numbers(compute_saliency_metrics(random_map, gt, **metric_args))
    result["random_checkpoint"] = str(random_checkpoint)
    result["metrics"]["random"] = random_metrics
    result["map_comparison_original_vs_random_evaluation_prepared"] = comparison(
        prepare_prediction_map(original_map), prepare_prediction_map(random_map))
    result["prediction_files"] = save_prediction_files(
        args.output_dir, prefix, original_map, random_map, prepare_prediction_map)
    result["prediction_file_note"] = (
        "_raw.npy holds unprocessed model outputs; _eval.npy and _eval.png "
        "hold independently evaluation-prepared predictions on a common [0,1] scale."
    )
    report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(figure), "report": str(report),
                      "prediction_files": result["prediction_files"],
                      "metrics": result["metrics"],
                      "priority_mask_max_abs_change": gate_max_delta}, indent=2))


if __name__ == "__main__":
    main()
