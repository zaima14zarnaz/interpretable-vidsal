"""Generate prototype-level natural-language descriptions for a concept model.

This script:
  1. Loads the trained ExplainableVidSalModel from a checkpoint.
  2. Runs the model over a dataset and retrieves top-K patch-sequence examples for
     every concept prototype using activations from model_out["concept_out"].
  3. Optionally sends the retrieved examples to a multimodal LLM and writes a structured
     per-concept explanation.
  4. Regenerates global summary files by scanning all existing concept directories.

Output structure
----------------
Under --concepts-root (default: dh1k/concepts):

  {concepts_root}/
    concept_summary.json          # list of all concept explanations (regenerated each run)
    concept_summary.csv           # same fields in CSV form
    c_tr_000/
      explanation.json            # LLM-generated concept description (if --skip-llm not set)
      top_examples/
        examples_metadata.json    # ranks, scores, retrieval metadata (no image paths except context panels)
        example_000_context_panel.png
        ...
        contact_sheet.png

During LLM explanation, each cluster sends one contact_sheet.png image (3 samples per row).
The concept-level sheet lives under top_examples/; per-cluster subsets are written under
cluster_explanations/ when clustering is enabled.

Top examples are retrieved using saliency-aware scoring by default: concept activations
restricted to the top --saliency-top-percent salient patches per sample (predicted saliency
by default; use --saliency-source gt for ground-truth maps). Use --saliency-filter-mode
none to recover activation-only behavior. Metadata includes patch_pred_saliency (patch
saliency score used for filtering), activation_score, retrieval_score, and
is_salient_region.

    c_per_000/
      ...

Per-concept explanation.json contains concept_id, global_concept_index, concept_type,
concept_number, candidate_name, explanation, confidence, saliency retrieval metadata,
llm_model, top_k, raw_llm_output, parse_success, and related fields.

Under --output-dir (default: explanation_outputs):

  retrieved_examples.json         # serializable copy of newly saved top examples
  concept_descriptions.json       # incremental aggregate of explanations generated this run

Pipeline
--------
For each selected concept, the script retrieves top examples across the dataset (heap
updates defer image cropping until save time), then writes top_examples to disk once at
the end (or every --save-every-n-batches). Then it optionally runs the LLM and writes
per-concept explanation.json plus a global concept_summary.json / .csv.

Use --max-concepts N to limit processing to global concept indices 0..N-1 and stop.

Resumability
------------
By default (--skip-existing), the script reuses existing top_examples/ files.
explanation.json is always regenerated (overwritten) when the LLM step runs.
Pass --overwrite to also regenerate top_examples. Use --skip-llm to retrieve top examples only.

Discover activation key
-----------------------
The per-concept activation tensor lives inside model_out["concept_out"]. If the key is
unknown, discover it first with --dry-run-discover:

  python explanation_generation.py \\
    --checkpoint training_outputs/best_checkpoint.pth \\
    --dataset-dir /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/testing \\
    --dry-run-discover

Full run (retrieval + LLM explanations)
---------------------------------------
  python explanation_generation.py \\
    --checkpoint training_outputs/best_checkpoint.pth \\
    --dataset-dir /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/testing \\
    --concepts-root /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/concepts \\
    --activation-key stage4.concept_activations \\
    --top-k 8 \\
    --llm-name qwen3

Retrieval only (no LLM)
-----------------------
  python explanation_generation.py \\
    --checkpoint training_outputs/best_checkpoint.pth \\
    --dataset-dir /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/testing \\
    --concepts-root /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/concepts \\
    --activation-key stage4.concept_activations \\
    --top-k 8 \\
    --skip-llm
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from model.llm import LLMHandle

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cluster_concepts import (
    cluster_examples_by_clip_features,
    cluster_examples_by_patch_difference,
    load_clip_model,
)
from pre_process.collate import video_saliency_collate_fn
from pre_process.dataloader import DatasetLoader
from model.model import ExplainableVidSalModel

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover - optional dependency
    process_vision_info = None


# -----------------------------
# Reproducibility / model setup
# -----------------------------

DEFAULT_CHECKPOINT = "/home/zaimaz/Desktop/research1/ExplainableVidSal/src/training_outputs/ckpts/20260611_035436/epoch_018.pth"
DEFAULT_CONCEPTS_ROOT = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/concepts"
)
NUM_EXAMPLE_FRAMES = 5
CONTACT_SHEET_COLS = 3


def resolve_checkpoint_path(checkpoint_path: str) -> Path:
    """Resolve a checkpoint path from CWD or src/ when not absolute."""

    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return path.resolve()

    candidates: List[Path] = []
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(_SRC_ROOT / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = [str(path), *[str(candidate) for candidate in candidates]]
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}\n"
        "Searched:\n  " + "\n  ".join(searched)
    )


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory if needed and return its Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_concept_type_and_local_index(
    global_concept_idx: int,
    num_transition_concepts: int,
    num_persistence_concepts: int,
) -> Tuple[str, int]:
    """
    Map a global concept index to (concept_type, local_index).

    Indices 0..num_transition_concepts-1 map to transition concepts.
    Indices num_transition_concepts..num_transition+num_persistence-1 map to persistence.
    """
    if global_concept_idx < 0:
        raise ValueError(f"global_concept_idx must be >= 0, got {global_concept_idx}")

    if global_concept_idx < num_transition_concepts:
        return "tr", global_concept_idx

    local_idx = global_concept_idx - num_transition_concepts
    if local_idx < num_persistence_concepts:
        return "per", local_idx

    total = num_transition_concepts + num_persistence_concepts
    raise ValueError(
        f"global_concept_idx {global_concept_idx} out of range for "
        f"{total} concepts ({num_transition_concepts} transition + "
        f"{num_persistence_concepts} persistence)"
    )


def get_concept_dir(
    concepts_root: Path,
    concept_type: str,
    concept_number: int,
) -> Path:
    """Return concepts_root/c_{tr|per}_{number:03d}."""
    if concept_type not in {"tr", "per"}:
        raise ValueError(f"concept_type must be 'tr' or 'per', got {concept_type!r}")
    return concepts_root / f"c_{concept_type}_{concept_number:03d}"


def format_concept_id(
    global_concept_idx: int,
    num_transition_concepts: int,
    num_persistence_concepts: int,
) -> str:
    concept_type, local_idx = get_concept_type_and_local_index(
        global_concept_idx,
        num_transition_concepts,
        num_persistence_concepts,
    )
    return f"c_{concept_type}_{local_idx:03d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device: torch.device) -> ExplainableVidSalModel:
    """Build the model exactly like the provided training script."""

    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=True,
        freeze_backbone=True,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=256,
        num_concepts=1024,
        concept_hidden_dim=256,
        saliency_hidden_dim=256,
        top_k=3,
        max_source_patches=64,
        tau_pi=0.5,
        tau_alpha=0.07,
        tau_concept=0.2,
        concept_residual_weight=0.0,
        use_rgb_refinement=False,
        use_feature_refinement=False,
        output_activation="sigmoid",
        return_details=True,
    ).to(device)
    model.eval()
    return model


def load_saliency_model(checkpoint_path: str, device: torch.device) -> ExplainableVidSalModel:
    resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
    model = build_model(device)
    ckpt = torch.load(resolved_checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: missing keys while loading checkpoint: {len(missing)}")
        print("  first missing keys:", missing[:10])
    if unexpected:
        print(f"WARNING: unexpected keys while loading checkpoint: {len(unexpected)}")
        print("  first unexpected keys:", unexpected[:10])
    model.eval()
    return model


# -----------------------------
# Tensor discovery and reshaping
# -----------------------------


def iter_named_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    """Recursively yield nested tensor names from dict/list/tuple outputs."""

    if torch.is_tensor(obj):
        yield prefix.rstrip("."), obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_named_tensors(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from iter_named_tensors(v, f"{prefix}{i}.")


def get_by_key_path(obj: Any, key_path: str) -> Any:
    """Retrieve nested objects using dot-separated paths."""

    cur = obj
    visited: List[str] = []
    for part in key_path.split("."):
        visited.append(part)
        if isinstance(cur, dict):
            if part not in cur:
                partial = ".".join(visited)
                raise KeyError(
                    f"Key {part!r} not found at path {partial!r}. "
                    f"Available keys: {sorted(cur.keys())}"
                )
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit():
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)} with part={part!r}")
    return cur


def _find_stage_with_activations(concept_out: dict) -> str:
    """Return the deepest stage that exposes trajectory concept activations."""

    for stage in reversed(("stage4", "stage3", "stage2", "stage1")):
        stage_out = concept_out.get(stage)
        if not isinstance(stage_out, dict):
            continue
        if {
            "metadata",
            "transition_activations",
            "persistence_activations",
        }.issubset(stage_out.keys()):
            return stage

    available = {
        stage: sorted(stage_out.keys())
        for stage, stage_out in concept_out.items()
        if isinstance(stage_out, dict)
    }
    raise KeyError(
        "No stage with transition_activations, persistence_activations, and metadata. "
        f"Available stage keys: {available}"
    )


def build_patch_concept_activations_from_stage(
    stage_out: Dict[str, Any],
    num_transition_concepts: int,
    num_persistence_concepts: int,
    *,
    use_transition: bool = True,
    use_persistence: bool = True,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Scatter trajectory activations onto target patches as [B, N, C]."""

    metadata = stage_out["metadata"]
    feature_shape = metadata["feature_shape"]
    batch_size = int(feature_shape["B"])
    grid_h = int(feature_shape["H"])
    grid_w = int(feature_shape["W"])
    num_patches = grid_h * grid_w
    total_concepts = num_transition_concepts + num_persistence_concepts

    transition_activations = stage_out["transition_activations"]
    persistence_activations = stage_out["persistence_activations"]
    device = transition_activations.device
    dtype = transition_activations.dtype

    num_transition = 0
    num_persistence = 0
    chunks: List[torch.Tensor] = []
    if use_transition:
        num_transition = min(num_transition_concepts, transition_activations.shape[-1])
        if num_transition > 0:
            chunks.append(transition_activations[:, :num_transition])
    if use_persistence:
        num_persistence = min(num_persistence_concepts, persistence_activations.shape[-1])
        if num_persistence > 0:
            chunks.append(persistence_activations[:, :num_persistence])

    if not chunks:
        raise ValueError("No concept activation columns selected for patch aggregation.")

    combined = torch.cat(chunks, dim=-1).float()
    activations_flat = torch.zeros(
        batch_size * num_patches,
        total_concepts,
        device=device,
        dtype=combined.dtype,
    )

    batch_idx = metadata["batch_idx"].to(device=device, dtype=torch.long)
    target_idx = metadata["target_idx"].to(device=device, dtype=torch.long)
    flat_patch = batch_idx * num_patches + target_idx
    scatter_index = flat_patch.unsqueeze(-1).expand(-1, combined.shape[-1])

    if use_transition and num_transition > 0:
        transition_index = scatter_index[:, :num_transition]
        activations_flat[:, :num_transition].scatter_reduce_(
            0,
            transition_index,
            combined[:, :num_transition],
            reduce="amax",
            include_self=True,
        )

    if use_persistence and num_persistence > 0:
        persistence_start = num_transition_concepts
        persistence_index = scatter_index[:, num_transition : num_transition + num_persistence]
        activations_flat[:, persistence_start : persistence_start + num_persistence].scatter_reduce_(
            0,
            persistence_index,
            combined[:, num_transition : num_transition + num_persistence],
            reduce="amax",
            include_self=True,
        )

    activations = activations_flat.view(batch_size, num_patches, total_concepts)
    return activations, (grid_h, grid_w)


def resolve_activation_tensor(
    model_out: dict,
    activation_key: Optional[str],
    num_transition_concepts: int,
    num_persistence_concepts: int,
    num_concepts: int,
) -> Tuple[str, torch.Tensor, Optional[Tuple[int, int]]]:
    """Resolve an activation key to a patch-grid tensor [B, N, C]."""

    concept_out = model_out.get("concept_out", {})
    if not isinstance(concept_out, dict) or not concept_out:
        raise KeyError("model_out does not contain a non-empty 'concept_out' dict.")

    combined_keys = {
        "concept_activations",
        "combined_patch_activations",
        "patch_concept_activations",
    }

    if activation_key is None:
        stage = _find_stage_with_activations(concept_out)
        activations, grid_hw = build_patch_concept_activations_from_stage(
            concept_out[stage],
            num_transition_concepts,
            num_persistence_concepts,
        )
        selected_name = f"{stage}.concept_activations"
        print(f"Auto-built patch concept activations from {selected_name}")
        return selected_name, activations, grid_hw

    if "." not in activation_key:
        raise ValueError(
            f"activation_key must look like 'stage4.concept_activations', got {activation_key!r}"
        )

    stage, field = activation_key.split(".", 1)
    if stage not in concept_out:
        raise KeyError(
            f"Stage {stage!r} not found in concept_out. "
            f"Available stages: {sorted(concept_out.keys())}"
        )

    stage_out = concept_out[stage]
    if not isinstance(stage_out, dict):
        raise KeyError(f"concept_out[{stage!r}] is not a dict.")

    if field in combined_keys:
        activations, grid_hw = build_patch_concept_activations_from_stage(
            stage_out,
            num_transition_concepts,
            num_persistence_concepts,
        )
        return activation_key, activations, grid_hw

    if field == "transition_activations":
        activations, grid_hw = build_patch_concept_activations_from_stage(
            stage_out,
            num_transition_concepts,
            num_persistence_concepts,
            use_transition=True,
            use_persistence=False,
        )
        return activation_key, activations, grid_hw

    if field == "persistence_activations":
        activations, grid_hw = build_patch_concept_activations_from_stage(
            stage_out,
            num_transition_concepts,
            num_persistence_concepts,
            use_transition=False,
            use_persistence=True,
        )
        return activation_key, activations, grid_hw

    raw = get_by_key_path(concept_out, activation_key)
    if not torch.is_tensor(raw):
        raise TypeError(
            f"Activation path {activation_key!r} resolved to {type(raw)}, expected a tensor."
        )
    activations, grid_hw = normalize_activation_tensor(raw, num_concepts)
    return activation_key, activations, grid_hw


def infer_grid_from_n(n_patches: int) -> Optional[Tuple[int, int]]:
    """Infer a plausible HxW grid from a flattened patch count."""

    root = int(math.sqrt(n_patches))
    if root * root == n_patches:
        return root, root

    # Common grids in this project after resizing to 224x384.
    common = [(28, 28), (14, 14), (8, 28), (7, 7), (14, 24), (7, 12), (28, 48)]
    for h, w in common:
        if h * w == n_patches:
            return h, w
    return None


def normalize_activation_tensor(
    x: torch.Tensor,
    num_concepts: int,
) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
    """Convert concept activation tensors to [B, N, C].

    Supported shapes:
      [B, N, C]
      [B, C, N]
      [B, C, H, W]
      [B, H, W, C]

    Returns:
      activations: float tensor [B, N, C]
      grid_hw: optional spatial grid if known
    """

    if x.dim() == 4:
        if x.shape[1] == num_concepts:
            # [B, C, H, W] -> [B, H*W, C]
            b, c, h, w = x.shape
            return x.permute(0, 2, 3, 1).reshape(b, h * w, c).float(), (h, w)
        if x.shape[-1] == num_concepts:
            # [B, H, W, C] -> [B, H*W, C]
            b, h, w, c = x.shape
            return x.reshape(b, h * w, c).float(), (h, w)

    if x.dim() == 3:
        if x.shape[-1] == num_concepts:
            b, n, c = x.shape
            return x.float(), infer_grid_from_n(n)
        if x.shape[1] == num_concepts:
            b, c, n = x.shape
            return x.permute(0, 2, 1).float(), infer_grid_from_n(n)

    raise ValueError(
        f"Could not normalize activation tensor with shape {tuple(x.shape)} "
        f"for num_concepts={num_concepts}."
    )


def auto_find_activation_tensor(
    model_out: dict,
    num_concepts: int,
) -> Tuple[str, torch.Tensor]:
    """Find a likely per-concept activation tensor inside model_out['concept_out']."""

    concept_out = model_out.get("concept_out")
    if concept_out is None:
        raise KeyError("model_out does not contain 'concept_out'.")

    candidates: List[Tuple[int, str, torch.Tensor]] = []
    good_words = ("activation", "activ", "score", "sim", "similarity", "prob", "logit")
    bad_words = ("loss", "debug", "saliency_map", "saliency_logits")

    for name, tensor in iter_named_tensors(concept_out):
        shape = tuple(tensor.shape)
        if tensor.dim() not in (3, 4):
            continue
        if num_concepts not in shape:
            continue
        lname = name.lower()
        if any(w in lname for w in bad_words):
            continue
        score = 0
        if "concept" in lname:
            score += 2
        if any(w in lname for w in good_words):
            score += 3
        if "stage4" in lname:
            score += 1
        candidates.append((score, name, tensor))

    if not candidates:
        available = [f"{n}: {tuple(t.shape)}" for n, t in iter_named_tensors(concept_out)]
        msg = "Could not auto-find a concept activation tensor. Available tensors:\n"
        msg += "\n".join(available[:80])
        raise RuntimeError(msg)

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_name, best_tensor = candidates[0]
    print(f"Auto-selected activation tensor: {best_name} shape={tuple(best_tensor.shape)} score={best_score}")
    print("Other candidate tensors:")
    for score, name, tensor in candidates[1:10]:
        print(f"  {name}: shape={tuple(tensor.shape)} score={score}")
    return best_name, best_tensor


def discover_concept_tensors(model_out: dict) -> None:
    """Print all tensors under concept_out for debugging the activation-key."""

    concept_out = model_out.get("concept_out")
    print("\nTensors found under model_out['concept_out']:")
    if concept_out is None:
        print("  None")
        return
    for name, tensor in iter_named_tensors(concept_out):
        print(f"  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


# -----------------------------
# Image/crop utilities
# -----------------------------


def rgb_video_to_btchw(rgb_batch: torch.Tensor) -> torch.Tensor:
    """Convert RGB batch to [B, T, C, H, W] float in [0, 1] when possible."""

    x = rgb_batch.detach().float().cpu()
    if x.dim() != 5:
        raise ValueError(f"Expected rgb_batch to be 5D, got {tuple(x.shape)}")

    if x.shape[2] == 3:
        # [B, T, C, H, W]
        out = x
    elif x.shape[-1] == 3:
        # [B, T, H, W, C]
        out = x.permute(0, 1, 4, 2, 3)
    else:
        raise ValueError(f"Unsupported rgb_batch shape {tuple(x.shape)}")

    if out.numel() > 0 and float(out.max()) > 1.0:
        out = out / 255.0
    return out.clamp(0.0, 1.0)


def tensor_chw_to_pil(x: torch.Tensor) -> Image.Image:
    arr = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def patch_bounds_from_index(
    patch_idx: int,
    grid_hw: Optional[Tuple[int, int]],
    image_hw: Tuple[int, int],
    pad_ratio: float = 0.0,
) -> Tuple[int, int, int, int]:
    """
    Convert flattened patch index into pixel bounds (x0, y0, x1, y1)
    for an image with size image_hw=(height, width).
    """
    if grid_hw is None:
        grid_hw = (1, 1)
    gh, gw = grid_hw
    h, w = image_hw
    row = int(patch_idx) // gw
    col = int(patch_idx) % gw
    row = max(0, min(row, gh - 1))
    col = max(0, min(col, gw - 1))

    y0 = int(round(row * h / gh))
    y1 = int(round((row + 1) * h / gh))
    x0 = int(round(col * w / gw))
    x1 = int(round((col + 1) * w / gw))

    if pad_ratio > 0:
        pad_y = int(round((y1 - y0) * pad_ratio))
        pad_x = int(round((x1 - x0) * pad_ratio))
        y0 = max(0, y0 - pad_y)
        y1 = min(h, y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(w, x1 + pad_x)

    return x0, y0, x1, y1


def draw_patch_rectangle(
    frame_img: Image.Image,
    box_xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 4,
) -> Image.Image:
    """Return a copy of frame_img with a rectangle drawn around the patch."""
    boxed = frame_img.copy()
    draw = ImageDraw.Draw(boxed)
    x0, y0, x1, y1 = box_xyxy
    for offset in range(width):
        draw.rectangle(
            (x0 - offset, y0 - offset, x1 + offset, y1 + offset),
            outline=color,
        )
    return boxed


def resize_with_aspect(
    image: Image.Image,
    max_size: Tuple[int, int],
) -> Image.Image:
    """Return a resized copy of image preserving aspect ratio."""
    resized = image.copy()
    resized.thumbnail(max_size, Image.Resampling.BICUBIC)
    return resized


def _paste_centered(canvas: Image.Image, image: Image.Image, cell_box: Tuple[int, int, int, int]) -> None:
    cx0, cy0, cx1, cy1 = cell_box
    cell_w = cx1 - cx0
    cell_h = cy1 - cy0
    paste_x = cx0 + (cell_w - image.width) // 2
    paste_y = cy0 + (cell_h - image.height) // 2
    canvas.paste(image, (paste_x, paste_y))


def make_context_panel_sequence(
    full_first_boxed: Image.Image,
    full_last_boxed: Image.Image,
    crop_frames: Sequence[Image.Image],
) -> Image.Image:
    """
    Create a 2-row panel:
    top row: cropped patch across consecutive frames
    bottom row: first and last full frames with the activated patch boxed
    """
    spacing = 6
    label_h = 16
    num_frames = len(crop_frames)
    patch_cell = (72, 72)
    full_cell = (220, 140)

    panel_w = patch_cell[0] * num_frames + spacing * max(0, num_frames - 1)
    panel_h = label_h + patch_cell[1] + spacing + label_h + full_cell[1] + 4
    canvas = Image.new("RGB", (panel_w, panel_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    top_y = label_h
    bottom_y = top_y + patch_cell[1] + spacing + label_h
    full_left_x = 0
    full_right_x = panel_w // 2 + spacing // 2

    for frame_idx, crop_frame in enumerate(crop_frames):
        cell_x0 = frame_idx * (patch_cell[0] + spacing)
        draw.text((cell_x0 + 4, 2), f"t+{frame_idx}", fill=(0, 0, 0))
        _paste_centered(
            canvas,
            resize_with_aspect(crop_frame, patch_cell),
            (cell_x0, top_y, cell_x0 + patch_cell[0], top_y + patch_cell[1]),
        )

    draw.text((full_left_x + 4, bottom_y - label_h + 2), "full frame t", fill=(0, 0, 0))
    draw.text(
        (full_right_x + 4, bottom_y - label_h + 2),
        f"full frame t+{num_frames - 1}",
        fill=(0, 0, 0),
    )
    _paste_centered(
        canvas,
        resize_with_aspect(full_first_boxed, full_cell),
        (full_left_x, bottom_y, full_left_x + panel_w // 2 - spacing // 2, bottom_y + full_cell[1]),
    )
    _paste_centered(
        canvas,
        resize_with_aspect(full_last_boxed, full_cell),
        (full_right_x, bottom_y, panel_w, bottom_y + full_cell[1]),
    )
    return canvas


def _compose_frame_t_and_t4_strip(
    frame_t: Image.Image,
    frame_t4: Image.Image,
) -> Image.Image:
    """Lay out cropped patches at frame t and frame t+4 side by side (no labels)."""
    spacing = 12
    frame_t_disp = resize_with_aspect(frame_t, (128, 128))
    frame_t4_disp = resize_with_aspect(frame_t4, (128, 128))
    strip_w = frame_t_disp.width + frame_t4_disp.width + spacing
    strip_h = max(frame_t_disp.height, frame_t4_disp.height)
    strip_img = Image.new("RGB", (strip_w, strip_h), color=(255, 255, 255))
    strip_img.paste(frame_t_disp, (0, 0))
    strip_img.paste(frame_t4_disp, (frame_t_disp.width + spacing, 0))
    return strip_img


def _extract_frame_t_and_t4_from_context_panel(
    context_panel: Image.Image,
    num_frames: int = NUM_EXAMPLE_FRAMES,
) -> Optional[Image.Image]:
    """Crop the first and last patch cells from a context panel's top row (no text)."""
    spacing = 6
    label_h = 16
    patch_cell = (72, 72)
    top_y = label_h
    last_x0 = (num_frames - 1) * (patch_cell[0] + spacing)
    try:
        first_crop = context_panel.crop((0, top_y, patch_cell[0], top_y + patch_cell[1]))
        last_crop = context_panel.crop(
            (last_x0, top_y, last_x0 + patch_cell[0], top_y + patch_cell[1])
        )
    except (ValueError, OSError):
        return None
    return _compose_frame_t_and_t4_strip(first_crop, last_crop)


def _heap_item_contact_sheet_image(item: HeapItem) -> Optional[np.ndarray]:
    """Build the contact-sheet cell image using only frame t and frame t+4 patches."""
    if item.frame_t_image is not None and item.frame_t1_image is not None:
        return np.asarray(_compose_frame_t_and_t4_strip(item.frame_t_image, item.frame_t1_image))
    if len(item.frame_sequence_images) >= 2:
        return np.asarray(
            _compose_frame_t_and_t4_strip(
                item.frame_sequence_images[0],
                item.frame_sequence_images[-1],
            )
        )
    return None


def _compose_patch_sequence_strip(crop_frames: Sequence[Image.Image]) -> Image.Image:
    """Lay out consecutive cropped patches side by side with frame labels."""
    spacing = 8
    label_h = 24
    display_frames = [resize_with_aspect(frame, (96, 96)) for frame in crop_frames]
    strip_w = sum(frame.width for frame in display_frames) + spacing * max(0, len(display_frames) - 1)
    strip_h = max(frame.height for frame in display_frames) + label_h
    sequence_img = Image.new("RGB", (strip_w, strip_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(sequence_img)

    x_offset = 0
    for frame_idx, frame_img in enumerate(display_frames):
        sequence_img.paste(frame_img, (x_offset, label_h))
        draw.text((x_offset + 4, 4), f"t+{frame_idx}", fill=(0, 0, 0))
        x_offset += frame_img.width + spacing
    return sequence_img


def crop_patch_sequence_images(
    rgb_batch: torch.Tensor,
    b_idx: int,
    patch_idx: int,
    grid_hw: Optional[Tuple[int, int]],
    pad_ratio: float = 0.10,
    num_frames: int = NUM_EXAMPLE_FRAMES,
) -> Tuple[
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    List[Image.Image],
]:
    """Crop the same patch across consecutive frames and build display assets."""

    video = rgb_video_to_btchw(rgb_batch)  # [B,T,C,H,W]
    b, t, c, h, w = video.shape
    if t < 2:
        raise ValueError("Need at least two frames to form a patch sequence example.")

    if grid_hw is None:
        grid_hw = infer_grid_from_n(max(patch_idx + 1, 1)) or (1, 1)

    use_frames = min(num_frames, t)
    frame_indices = list(range(t - use_frames, t))

    box_x0, box_y0, box_x1, box_y1 = patch_bounds_from_index(
        patch_idx, grid_hw, (h, w), pad_ratio=0.0
    )
    crop_x0, crop_y0, crop_x1, crop_y1 = patch_bounds_from_index(
        patch_idx, grid_hw, (h, w), pad_ratio=pad_ratio
    )

    crop_frames: List[Image.Image] = []
    for frame_idx in frame_indices:
        full_pil = tensor_chw_to_pil(video[b_idx, frame_idx])
        crop_frames.append(full_pil.crop((crop_x0, crop_y0, crop_x1, crop_y1)))

    frame_t_img = resize_with_aspect(crop_frames[0], (128, 128))
    frame_t1_img = resize_with_aspect(crop_frames[-1], (128, 128))
    sequence_img = _compose_patch_sequence_strip(crop_frames)

    full_first_pil = tensor_chw_to_pil(video[b_idx, frame_indices[0]])
    full_last_pil = tensor_chw_to_pil(video[b_idx, frame_indices[-1]])
    full_t_boxed_img = draw_patch_rectangle(
        full_first_pil,
        (box_x0, box_y0, box_x1, box_y1),
    )
    full_t1_boxed_img = draw_patch_rectangle(
        full_last_pil,
        (box_x0, box_y0, box_x1, box_y1),
    )
    context_panel_img = make_context_panel_sequence(
        full_t_boxed_img,
        full_t1_boxed_img,
        crop_frames,
    )
    return (
        frame_t_img,
        frame_t1_img,
        sequence_img,
        full_t_boxed_img,
        full_t1_boxed_img,
        context_panel_img,
        crop_frames,
    )


def crop_patch_pair_images(
    rgb_batch: torch.Tensor,
    b_idx: int,
    patch_idx: int,
    grid_hw: Optional[Tuple[int, int]],
    pad_ratio: float = 0.10,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    """Backward-compatible wrapper around the multi-frame patch sequence cropper."""
    (
        frame_t_img,
        frame_t1_img,
        sequence_img,
        full_t_boxed_img,
        full_t1_boxed_img,
        context_panel_img,
        _crop_frames,
    ) = crop_patch_sequence_images(
        rgb_batch,
        b_idx,
        patch_idx,
        grid_hw,
        pad_ratio=pad_ratio,
    )
    return (
        frame_t_img,
        frame_t1_img,
        sequence_img,
        full_t_boxed_img,
        full_t1_boxed_img,
        context_panel_img,
    )


def crop_patch_pair(
    rgb_batch: torch.Tensor,
    b_idx: int,
    patch_idx: int,
    grid_hw: Optional[Tuple[int, int]],
    pad_ratio: float = 0.10,
) -> Image.Image:
    """Crop the same spatial cell across consecutive frames and concatenate side by side."""
    _, _, sequence_img, _, _, _ = crop_patch_pair_images(
        rgb_batch, b_idx, patch_idx, grid_hw, pad_ratio=pad_ratio
    )
    return sequence_img


def extract_gt_saliency_batch(sal_batch: torch.Tensor) -> torch.Tensor:
    """Return ground-truth saliency maps from the dataset batch ([B, T, H, W])."""
    if not torch.is_tensor(sal_batch):
        raise ValueError("sal_batch must be a tensor with ground-truth saliency maps.")
    return sal_batch


def extract_predicted_patch_saliency(
    model_out: dict,
    grid_hw: Optional[Tuple[int, int]],
) -> torch.Tensor:
    """
    Return predicted per-patch saliency scores as [B, N] on the concept patch grid.
    """
    patch_logits = model_out.get("patch_saliency_logits")
    if patch_logits is not None and torch.is_tensor(patch_logits):
        scores = patch_logits.detach().float().cpu()
        if scores.dim() == 4:
            scores = scores[:, 0] if scores.shape[1] == 1 else scores.mean(dim=1)
        elif scores.dim() != 3:
            raise ValueError(
                f"Unexpected patch_saliency_logits shape {tuple(patch_logits.shape)}"
            )

        if grid_hw is not None and tuple(scores.shape[-2:]) != tuple(grid_hw):
            scores = F.interpolate(
                scores.unsqueeze(1),
                size=grid_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if float(scores.max()) > 1.0 or float(scores.min()) < 0.0:
            scores = torch.sigmoid(scores)
        return scores.flatten(1).clamp(0.0, 1.0)

    saliency_map = model_out.get("saliency_map")
    if saliency_map is None or not torch.is_tensor(saliency_map):
        raise ValueError(
            "model_out is missing predicted saliency outputs "
            "(patch_saliency_logits or saliency_map)."
        )
    return compute_patch_saliency_grid(saliency_map, grid_hw)


def get_target_saliency_frame(saliency_maps: torch.Tensor) -> torch.Tensor:
    """
    Return target saliency frame as [B, H, W] float tensor on CPU.
    """
    target = saliency_maps.detach().float().cpu()
    if target.dim() == 5:
        if target.shape[1] == 1:
            # [B, 1, T, H, W] -> channel 0, last temporal frame
            target = target[:, 0, -1]
        elif target.shape[2] == 1:
            # [B, T, 1, H, W] -> last temporal frame, channel 0
            target = target[:, -1, 0]
        else:
            # [B, T, H, W] (no explicit channel dim in dim 1/2)
            target = target[:, -1]
    elif target.dim() == 4:
        if target.shape[1] == 1:
            # [B, 1, H, W]
            target = target[:, 0]
        else:
            # [B, T, H, W]
            target = target[:, -1]
    elif target.dim() != 3:
        raise ValueError(f"Unsupported saliency_maps shape {tuple(saliency_maps.shape)}")

    if target.numel() > 0 and float(target.max()) > 1.0:
        target = target / 255.0
    return target.clamp(0.0, 1.0)


def compute_patch_saliency_grid(
    saliency_maps: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]],
) -> torch.Tensor:
    """
    Convert saliency maps to patch-level saliency.

    Returns:
        patch_saliency: [B, N] mean saliency inside each patch cell.
    """
    target = get_target_saliency_frame(saliency_maps)
    if grid_hw is None:
        grid_hw = (1, 1)

    target_4d = target.unsqueeze(1)
    pooled = F.adaptive_avg_pool2d(target_4d, output_size=grid_hw)
    return pooled.flatten(1).float().cpu()


def _ensure_sample_salient_patch(
    salient_mask: torch.Tensor,
    patch_saliency: torch.Tensor,
) -> torch.Tensor:
    """If a sample has no salient patches, keep its highest-saliency patch."""
    updated = salient_mask.clone()
    for b in range(patch_saliency.shape[0]):
        if not bool(updated[b].any()):
            updated[b, patch_saliency[b].argmax()] = True
    return updated


def build_saliency_retrieval_scores(
    activations: torch.Tensor,
    patch_saliency: torch.Tensor,
    mode: str,
    saliency_threshold: float,
    saliency_top_percent: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build retrieval scores for concept examples.

    Returns:
        retrieval_scores: [B*N, C]
        salient_mask_flat: [B*N] bool
    """
    acts = activations.detach().float().cpu()
    bsz, n_patches, num_concepts = acts.shape
    concept_flat = acts.reshape(bsz * n_patches, num_concepts)
    sal_flat = patch_saliency.reshape(bsz * n_patches)
    salient_mask = torch.zeros(bsz, n_patches, dtype=torch.bool)

    if mode == "none":
        salient_mask[:] = True
        retrieval_scores = concept_flat.clone()
    elif mode == "gt_threshold":
        salient_mask = patch_saliency >= saliency_threshold
        salient_mask = _ensure_sample_salient_patch(salient_mask, patch_saliency)
        retrieval_scores = concept_flat.clone()
    elif mode == "gt_top_percent":
        top_percent = min(max(float(saliency_top_percent), 1e-6), 1.0)
        k_keep = max(1, int(math.ceil(n_patches * top_percent)))
        for b in range(bsz):
            _, top_idx = torch.topk(patch_saliency[b], k=k_keep, largest=True)
            salient_mask[b, top_idx] = True
        retrieval_scores = concept_flat.clone()
    elif mode == "gt_weighted":
        if saliency_threshold > 0:
            salient_mask = patch_saliency >= saliency_threshold
            salient_mask = _ensure_sample_salient_patch(salient_mask, patch_saliency)
        else:
            salient_mask[:] = True
        retrieval_scores = concept_flat * sal_flat.unsqueeze(1)
    else:
        raise ValueError(f"Unknown saliency filter mode: {mode}")

    salient_mask_flat = salient_mask.reshape(bsz * n_patches)
    if mode != "none":
        retrieval_scores = retrieval_scores.clone()
        retrieval_scores[~salient_mask_flat, :] = float("-inf")

    return retrieval_scores, salient_mask_flat


# -----------------------------
# Retrieval data structures
# -----------------------------


@dataclass(order=True)
class HeapItem:
    score: float
    serial: int
    concept_idx: int
    batch_index: int
    patch_index: int
    video_name: str
    grid_hw: Optional[Tuple[int, int]]
    image: Any = field(compare=False, default=None)
    frame_t_image: Any = field(compare=False, default=None)
    frame_t1_image: Any = field(compare=False, default=None)
    frame_sequence_images: List[Any] = field(compare=False, default_factory=list)
    full_t_boxed_image: Any = field(compare=False, default=None)
    full_t1_boxed_image: Any = field(compare=False, default=None)
    context_panel_image: Any = field(compare=False, default=None)
    rgb_window: Any = field(compare=False, default=None)
    activation_score: float = field(compare=False, default=0.0)
    patch_pred_saliency: float = field(compare=False, default=0.0)
    retrieval_score: float = field(compare=False, default=0.0)
    is_salient_region: bool = field(compare=False, default=False)


@dataclass
class SavedExample:
    concept_idx: int
    rank: int
    score: float
    video_name: str
    batch_index: int
    patch_index: int
    grid_hw: Optional[Tuple[int, int]]
    image_path: str
    frame_t_path: str
    frame_t1_path: str
    pair_path: str
    sequence_path: str = ""
    frame_paths: List[str] = field(default_factory=list)
    num_frames: int = NUM_EXAMPLE_FRAMES
    full_t_boxed_path: str = ""
    full_t1_boxed_path: str = ""
    context_panel_path: str = ""
    activation_score: float = 0.0
    patch_pred_saliency: float = 0.0
    retrieval_score: float = 0.0
    is_salient_region: bool = False


def resolve_concept_indices(
    num_concepts: int,
    max_concepts: Optional[int] = None,
) -> List[int]:
    """Return global concept indices to process, in ascending order."""

    if max_concepts is None:
        return list(range(num_concepts))
    if max_concepts <= 0:
        raise ValueError(f"max_concepts must be positive, got {max_concepts}")
    return list(range(min(max_concepts, num_concepts)))


def populate_heap_item_images(item: HeapItem) -> None:
    """Crop display images once from the cached RGB window (deferred cropping)."""
    if item.image is not None:
        return
    if item.rgb_window is None:
        return

    window_batch = item.rgb_window.unsqueeze(0)
    try:
        (
            frame_t_img,
            frame_t1_img,
            sequence_img,
            full_t_boxed_img,
            full_t1_boxed_img,
            context_panel_img,
            crop_frames,
        ) = crop_patch_sequence_images(
            window_batch,
            0,
            item.patch_index,
            item.grid_hw,
        )
    except Exception as exc:
        print(f"WARNING: failed to crop patch sequence for {item.video_name}: {exc}")
        return

    item.frame_t_image = frame_t_img
    item.frame_t1_image = frame_t1_img
    item.frame_sequence_images = list(crop_frames)
    item.image = sequence_img
    item.full_t_boxed_image = full_t_boxed_img
    item.full_t1_boxed_image = full_t1_boxed_img
    item.context_panel_image = context_panel_img


def update_heaps_from_activations(
    heaps: Dict[int, List[HeapItem]],
    activations: torch.Tensor,
    rgb_batch: torch.Tensor,
    video_filenames: Sequence[Any],
    grid_hw: Optional[Tuple[int, int]],
    top_k: int,
    serial_start: int,
    concept_indices: Optional[Sequence[int]] = None,
    max_candidates_per_concept_per_batch: int = 4,
    saliency_filter_mode: str = "gt_top_percent",
    saliency_threshold: float = 0.0,
    saliency_top_percent: float = 0.20,
    saliency_maps: Optional[torch.Tensor] = None,
    patch_saliency: Optional[torch.Tensor] = None,
) -> int:
    """Update per-concept heaps using saliency-aware retrieval scores [B,N,C]."""

    acts = activations.detach().float().cpu()
    bsz, n_patches, num_concepts = acts.shape
    activation_flat = acts.reshape(bsz * n_patches, num_concepts)
    if patch_saliency is not None:
        patch_sal = patch_saliency.detach().float().cpu()
    elif saliency_maps is not None:
        patch_sal = compute_patch_saliency_grid(saliency_maps, grid_hw)
    else:
        raise ValueError("Either patch_saliency or saliency_maps must be provided.")
    patch_saliency = patch_sal
    sal_flat = patch_saliency.reshape(bsz * n_patches)
    retrieval_scores, salient_mask_flat = build_saliency_retrieval_scores(
        activations=acts,
        patch_saliency=patch_saliency,
        mode=saliency_filter_mode,
        saliency_threshold=saliency_threshold,
        saliency_top_percent=saliency_top_percent,
    )
    serial = serial_start
    indices = (
        list(concept_indices)
        if concept_indices is not None
        else list(range(num_concepts))
    )
    if not indices:
        return serial

    scores_sub = retrieval_scores[:, indices]
    k_local = min(max_candidates_per_concept_per_batch, scores_sub.shape[0])
    if k_local <= 0:
        return serial

    batch_values, batch_flat_idx = torch.topk(scores_sub, k=k_local, dim=0, largest=True)

    for col_j, c_idx in enumerate(indices):
        for row in range(k_local):
            value = batch_values[row, col_j]
            if not torch.isfinite(value):
                continue

            flat_idx = int(batch_flat_idx[row, col_j].item())
            b_idx = flat_idx // n_patches
            p_idx = flat_idx % n_patches
            video_name = str(video_filenames[b_idx])

            item = HeapItem(
                score=float(value),
                serial=serial,
                concept_idx=c_idx,
                batch_index=b_idx,
                patch_index=p_idx,
                video_name=video_name,
                grid_hw=grid_hw,
                rgb_window=rgb_batch[b_idx].detach().cpu().clone(),
                activation_score=float(activation_flat[flat_idx, c_idx]),
                patch_pred_saliency=float(sal_flat[flat_idx]),
                retrieval_score=float(value),
                is_salient_region=bool(salient_mask_flat[flat_idx]),
            )
            serial += 1
            heap = heaps.setdefault(c_idx, [])
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif item.score > heap[0].score:
                heapq.heapreplace(heap, item)
    return serial


def should_skip_existing_artifacts(skip_existing: bool, overwrite: bool) -> bool:
    """Return True when existing per-concept artifacts should be reused."""
    return skip_existing and not overwrite


def _should_write(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.exists()


def _save_example_image(
    image: Optional[Image.Image],
    path: Path,
    size: Tuple[int, int],
    overwrite: bool,
) -> bool:
    if not _should_write(path, overwrite):
        return False
    if image is not None:
        image.save(path)
        return True
    placeholder = Image.new("RGB", size, color=(240, 240, 240))
    placeholder.save(path)
    return True


def _relative_to_concept_dir(concept_dir: Path, path: Path) -> str:
    return str(path.relative_to(concept_dir))


def contact_sheet_grid_shape(num_examples: int) -> Tuple[int, int]:
    """Return (rows, cols) for a contact sheet grid (three samples per row)."""

    if num_examples <= 0:
        return 1, 1
    cols = CONTACT_SHEET_COLS
    rows = max(1, math.ceil(num_examples / cols))
    return rows, cols


def _render_contact_sheet_grid(
    cells: Sequence[Optional[np.ndarray]],
    path: Path,
) -> bool:
    """Render a text-free contact-sheet PNG from display arrays."""
    if not cells:
        return False

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    num_examples = len(cells)
    rows, cols = contact_sheet_grid_shape(num_examples)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, max(rows, 1) * 2.2))
    if rows * cols == 1:
        flat_axes = [axes]
    else:
        flat_axes = list(np.asarray(axes).reshape(-1))

    for cell_idx, ax in enumerate(flat_axes):
        ax.set_xticks([])
        ax.set_yticks([])
        if cell_idx >= num_examples:
            ax.axis("off")
            continue

        image_arr = cells[cell_idx]
        if image_arr is not None:
            ax.imshow(image_arr)
        else:
            ax.imshow(np.full((128, 512, 3), 0.94, dtype=np.float32))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def save_contact_sheet(
    ranked: Sequence[HeapItem],
    path: Path,
    overwrite: bool,
) -> bool:
    """Save a text-free 3-column contact sheet of frame t / frame t+4 strips."""

    if not ranked or not _should_write(path, overwrite):
        return False

    cells: List[Optional[np.ndarray]] = []
    for item in ranked:
        cells.append(_heap_item_contact_sheet_image(item))
    return _render_contact_sheet_grid(cells, path)


def _load_contact_sheet_cell_image(
    concept_dir: Path,
    example: SavedExample,
) -> Optional[np.ndarray]:
    """Load a text-free frame t / frame t+4 strip for one contact-sheet cell."""
    image_path = _example_source_path(
        concept_dir,
        example,
        "context_panel_path",
        "context_panel",
    )
    if image_path is None:
        return None
    try:
        context_panel = Image.open(image_path).convert("RGB")
        strip_img = _extract_frame_t_and_t4_from_context_panel(
            context_panel,
            num_frames=example.num_frames or NUM_EXAMPLE_FRAMES,
        )
        if strip_img is None:
            return None
        return np.asarray(strip_img)
    except (OSError, ValueError):
        return None


def save_contact_sheet_from_examples(
    concept_dir: Path,
    examples: Sequence[SavedExample],
    path: Path,
    overwrite: bool = True,
) -> bool:
    """Save a contact sheet for a subset of saved examples."""
    if not examples:
        return False
    if not overwrite and path.exists():
        return True

    cells: List[Optional[np.ndarray]] = []
    for example in sorted(examples, key=lambda item: item.rank):
        cells.append(_load_contact_sheet_cell_image(concept_dir, example))
    return _render_contact_sheet_grid(cells, path)


def _prune_top_examples_dir(top_examples_dir: Path) -> None:
    """Remove legacy image files; keep context panels, contact sheet, and metadata."""
    for path in top_examples_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in {"contact_sheet.png", "examples_metadata.json"}:
            continue
        if path.name.endswith("_context_panel.png"):
            continue
        path.unlink(missing_ok=True)


def load_saved_examples_from_metadata(
    concept_dir: Path,
    metadata_path: Path,
    concept_idx: int,
) -> List[SavedExample]:
    """Load SavedExample entries from an existing top_examples metadata file."""

    entries = json.loads(metadata_path.read_text(encoding="utf-8"))
    examples: List[SavedExample] = []
    for entry in entries:
        context_panel_path = str(entry.get("context_panel_path", ""))
        activation_score = float(entry.get("activation_score", entry.get("score", 0.0)))
        patch_pred_saliency = float(
            entry.get(
                "patch_pred_saliency",
                entry.get("patch_gt_saliency", 0.0),
            )
        )
        retrieval_score = float(
            entry.get(
                "retrieval_score",
                entry.get("activation_score", entry.get("score", 0.0)),
            )
        )
        is_salient_region = bool(
            entry.get("is_salient_region", patch_pred_saliency > 0.0)
        )
        examples.append(
            SavedExample(
                concept_idx=concept_idx,
                rank=int(entry["rank"]),
                score=retrieval_score,
                video_name=str(entry.get("video_filename", "")),
                batch_index=int(entry.get("batch_index", -1)),
                patch_index=int(entry.get("sample_index", entry.get("patch_index", -1))),
                grid_hw=None,
                image_path=context_panel_path,
                frame_t_path="",
                frame_t1_path="",
                pair_path="",
                sequence_path="",
                frame_paths=[],
                num_frames=int(entry.get("num_frames", NUM_EXAMPLE_FRAMES)),
                full_t_boxed_path="",
                full_t1_boxed_path="",
                context_panel_path=context_panel_path,
                activation_score=activation_score,
                patch_pred_saliency=patch_pred_saliency,
                retrieval_score=retrieval_score,
                is_salient_region=is_salient_region,
            )
        )
    return sorted(examples, key=lambda e: e.rank)


def save_single_concept_heap(
    c_idx: int,
    heap: Sequence[HeapItem],
    concepts_root: Path,
    num_transition_concepts: int,
    num_persistence_concepts: int,
    overwrite: bool = True,
) -> List[SavedExample]:
    """Write one concept's current top examples to disk."""

    concept_type, local_idx = get_concept_type_and_local_index(
        c_idx,
        num_transition_concepts,
        num_persistence_concepts,
    )
    concept_dir = ensure_dir(get_concept_dir(concepts_root, concept_type, local_idx))
    top_examples_dir = ensure_dir(concept_dir / "top_examples")
    metadata_path = top_examples_dir / "examples_metadata.json"

    ranked = sorted(heap, key=lambda x: x.score, reverse=True)
    saved_examples: List[SavedExample] = []
    metadata_entries: List[dict] = []

    if overwrite:
        _prune_top_examples_dir(top_examples_dir)

    for rank, item in enumerate(ranked):
        populate_heap_item_images(item)
        prefix = f"example_{rank:03d}"
        context_panel_abs = top_examples_dir / f"{prefix}_context_panel.png"

        _save_example_image(
            item.context_panel_image, context_panel_abs, size=(540, 340), overwrite=overwrite
        )

        context_panel_rel = _relative_to_concept_dir(concept_dir, context_panel_abs)

        saved_examples.append(
            SavedExample(
                concept_idx=c_idx,
                rank=rank,
                score=item.retrieval_score,
                video_name=item.video_name,
                batch_index=item.batch_index,
                patch_index=item.patch_index,
                grid_hw=item.grid_hw,
                image_path=context_panel_rel,
                frame_t_path="",
                frame_t1_path="",
                pair_path="",
                sequence_path="",
                frame_paths=[],
                num_frames=NUM_EXAMPLE_FRAMES,
                full_t_boxed_path="",
                full_t1_boxed_path="",
                context_panel_path=context_panel_rel,
                activation_score=item.activation_score,
                patch_pred_saliency=item.patch_pred_saliency,
                retrieval_score=item.retrieval_score,
                is_salient_region=item.is_salient_region,
            )
        )
        metadata_entries.append(
            {
                "rank": rank,
                "score": float(item.score),
                "activation_score": float(item.activation_score),
                "patch_pred_saliency": float(item.patch_pred_saliency),
                "retrieval_score": float(item.retrieval_score),
                "is_salient_region": bool(item.is_salient_region),
                "batch_index": int(item.batch_index),
                "sample_index": int(item.patch_index),
                "global_concept_index": int(c_idx),
                "concept_type": concept_type,
                "concept_number": int(local_idx),
                "video_filename": item.video_name,
                "num_frames": NUM_EXAMPLE_FRAMES,
                "context_panel_path": context_panel_rel,
            }
        )

    if metadata_entries:
        metadata_path.write_text(
            json.dumps(metadata_entries, indent=2),
            encoding="utf-8",
        )
        save_contact_sheet(
            ranked,
            top_examples_dir / "contact_sheet.png",
            overwrite=overwrite,
        )

    return saved_examples


def partition_concepts_for_retrieval(
    concept_indices: Sequence[int],
    concepts_root: Path,
    num_transition_concepts: int,
    num_persistence_concepts: int,
    skip_existing: bool,
    overwrite: bool,
) -> Tuple[List[int], Dict[int, List[SavedExample]]]:
    """Split concepts into those needing retrieval vs. reused on-disk examples."""

    saved: Dict[int, List[SavedExample]] = {}
    retrieve_indices: List[int] = []
    skip_artifacts = should_skip_existing_artifacts(skip_existing, overwrite)

    for c_idx in concept_indices:
        concept_type, local_idx = get_concept_type_and_local_index(
            c_idx,
            num_transition_concepts,
            num_persistence_concepts,
        )
        concept_dir = get_concept_dir(concepts_root, concept_type, local_idx)
        metadata_path = concept_dir / "top_examples" / "examples_metadata.json"
        if skip_artifacts and metadata_path.exists():
            saved[c_idx] = load_saved_examples_from_metadata(
                concept_dir,
                metadata_path,
                c_idx,
            )
            continue
        retrieve_indices.append(c_idx)

    return retrieve_indices, saved


def write_retrieved_examples_aggregate(
    saved_examples: Dict[int, List[SavedExample]],
    output_dir: Path,
) -> None:
    metadata_path = output_dir / "retrieved_examples.json"
    serializable = {str(k): [asdict(e) for e in v] for k, v in saved_examples.items()}
    metadata_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def save_retrieved_examples(
    heaps: Dict[int, List[HeapItem]],
    concepts_root: Path,
    num_transition_concepts: int,
    num_persistence_concepts: int,
    concept_indices: Optional[Sequence[int]] = None,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
    skip_existing: bool = True,
) -> Dict[int, List[SavedExample]]:
    ensure_dir(concepts_root)
    saved: Dict[int, List[SavedExample]] = {}
    skip_artifacts = should_skip_existing_artifacts(skip_existing, overwrite)
    indices = (
        sorted(heaps.keys())
        if concept_indices is None
        else list(concept_indices)
    )

    for c_idx in indices:
        heap = heaps.get(c_idx, [])
        concept_type, local_idx = get_concept_type_and_local_index(
            c_idx,
            num_transition_concepts,
            num_persistence_concepts,
        )
        concept_dir = get_concept_dir(concepts_root, concept_type, local_idx)
        metadata_path = concept_dir / "top_examples" / "examples_metadata.json"

        if metadata_path.exists() and skip_artifacts:
            saved[c_idx] = load_saved_examples_from_metadata(
                concept_dir,
                metadata_path,
                c_idx,
            )
            continue

        saved[c_idx] = save_single_concept_heap(
            c_idx,
            heap,
            concepts_root=concepts_root,
            num_transition_concepts=num_transition_concepts,
            num_persistence_concepts=num_persistence_concepts,
            overwrite=overwrite or not skip_artifacts,
        )

    if output_dir is not None:
        write_retrieved_examples_aggregate(saved, output_dir)

    return saved


# -----------------------------
# LLM prompting / parsing
# -----------------------------



def build_concept_explanation_prompt(concept_id: str, top_k: int) -> str:
    return f"""You are viewing {top_k} top-activated examples for one learned visual prototype.

The task is to describe the common visual pattern across these samples. Assess why eye attention remains fixated on these patches over the other regions.

Example common patterns:
- Common pattern: sharp edges and green textures with gradual motion to the right.
- Common pattern: misshaped blobs with static motion across frames.
- Common pattern: Two types of samples: hand movements and object interactions.

Return this JSON:
{{
  "concept_id": "{concept_id}",
  "candidate_name": "...",
  "explanation": "...",
  "confidence": "high | medium | low"
}}



""".strip()


def parse_json_from_text(text: str) -> dict:
    """Recover a JSON object from raw LLM text, tolerating fences and surrounding prose."""

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty LLM output")

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {text[:500]}")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : idx + 1]
                return json.loads(candidate)

    raise ValueError(f"Unbalanced JSON object in output: {text[:500]}")


EXPLANATION_RECORD_KEYS = (
    "concept_id",
    "global_concept_index",
    "concept_type",
    "concept_number",
    "candidate_name",
    "explanation",
    "confidence",
    "num_salient_examples",
    "mean_patch_pred_saliency",
    "mean_activation_score",
    "mean_retrieval_score",
    "saliency_filter_mode",
    "saliency_threshold",
    "saliency_top_percent",
    "low_saliency_coverage",
    "llm_model",
    "top_k",
    "top_examples_dir",
    "top_examples_metadata",
    "raw_llm_output",
    "parse_success",
    "parse_error",
)


def build_explanation_record(
    *,
    concept_id: str,
    global_concept_index: int,
    concept_type: str,
    concept_number: int,
    parsed_llm: Optional[dict],
    raw_llm_output: str,
    parse_success: bool,
    llm_model: str,
    top_k: int,
    num_salient_examples: int,
    mean_patch_pred_saliency: float,
    mean_activation_score: float,
    mean_retrieval_score: float,
    saliency_filter_mode: str,
    saliency_threshold: float,
    saliency_top_percent: float,
    low_saliency_coverage: bool = False,
    parse_error: Optional[str] = None,
) -> dict:
    """Build the final explanation.json payload for one concept."""

    record = {
        "concept_id": concept_id,
        "global_concept_index": global_concept_index,
        "concept_type": concept_type,
        "concept_number": concept_number,
        "llm_model": llm_model,
        "top_k": top_k,
        "num_salient_examples": num_salient_examples,
        "mean_patch_pred_saliency": mean_patch_pred_saliency,
        "mean_activation_score": mean_activation_score,
        "mean_retrieval_score": mean_retrieval_score,
        "saliency_filter_mode": saliency_filter_mode,
        "saliency_threshold": saliency_threshold,
        "saliency_top_percent": saliency_top_percent,
        "low_saliency_coverage": low_saliency_coverage,
        "top_examples_dir": "top_examples",
        "top_examples_metadata": "top_examples/examples_metadata.json",
        "raw_llm_output": raw_llm_output,
        "parse_success": parse_success,
    }

    if parse_success and parsed_llm is not None:
        explanation_text = (
            parsed_llm.get("explanation")
            or parsed_llm.get("saliency_shift_explanation")
            or parsed_llm.get("trajectory_explanation")
            or parsed_llm.get("visual_description")
            or parsed_llm.get("common_visual_pattern")
            or parsed_llm.get("evidence")
            or "unclear"
        )
        record.update(
            {
                "candidate_name": str(parsed_llm.get("candidate_name", "unclear")),
                "explanation": str(explanation_text),
                "confidence": str(parsed_llm.get("confidence", "low")),
            }
        )
    else:
        record.update(
            {
                "candidate_name": "unclear",
                "explanation": "unclear",
                "confidence": "low",
            }
        )
        if parse_error:
            record["parse_error"] = parse_error

    return {key: record[key] for key in EXPLANATION_RECORD_KEYS if key in record}


CONFIDENCE_RANK = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "unclear": 0,
}


def _example_source_path(
    concept_dir: Path,
    example: SavedExample,
    path_attr: str,
    fallback_suffix: str,
) -> Optional[Path]:
    rel_path = getattr(example, path_attr, "") or ""
    if rel_path:
        candidate = concept_dir / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        if candidate.exists():
            return candidate

    fallback = concept_dir / "top_examples" / f"example_{example.rank:03d}_{fallback_suffix}.png"
    if fallback.exists():
        return fallback
    return None


def collect_cluster_contact_sheet_path(
    concept_dir: Path,
    cluster_id: int,
    examples: Sequence[SavedExample],
) -> Tuple[str, str]:
    """
    Build one contact-sheet image for a cluster and return its absolute/relative paths.
    """
    contact_sheet_abs = (
        concept_dir
        / "cluster_explanations"
        / f"cluster_{int(cluster_id):02d}_contact_sheet.png"
    )
    save_contact_sheet_from_examples(
        concept_dir,
        examples,
        contact_sheet_abs,
        overwrite=True,
    )
    return str(contact_sheet_abs), _relative_to_concept_dir(concept_dir, contact_sheet_abs)


def collect_llm_contact_sheet_path(
    concept_dir: Path,
    cluster_examples_list: Sequence[SavedExample],
    all_examples: Sequence[SavedExample],
    cluster_id: int,
) -> Optional[str]:
    """Return the contact-sheet image path to send to the LLM (one image per cluster)."""
    concept_contact_sheet = concept_dir / "top_examples" / "contact_sheet.png"
    cluster_ranks = {example.rank for example in cluster_examples_list}
    all_ranks = {example.rank for example in all_examples}
    if cluster_ranks == all_ranks and concept_contact_sheet.exists():
        return str(concept_contact_sheet)

    contact_sheet_abs, _ = collect_cluster_contact_sheet_path(
        concept_dir,
        cluster_id,
        cluster_examples_list,
    )
    if Path(contact_sheet_abs).exists():
        return contact_sheet_abs

    print(
        f"WARNING: missing contact sheet for cluster {cluster_id} "
        f"under {concept_dir}"
    )
    return None


def compute_example_statistics(
    examples: Sequence[SavedExample],
    min_salient_examples_per_concept: int,
) -> Tuple[int, float, float, float, bool]:
    num_salient_examples = sum(1 for example in examples if example.is_salient_region)
    mean_patch_pred_saliency = float(
        np.mean([example.patch_pred_saliency for example in examples]) if examples else 0.0
    )
    mean_activation_score = float(
        np.mean([example.activation_score for example in examples]) if examples else 0.0
    )
    mean_retrieval_score = float(
        np.mean([example.retrieval_score for example in examples]) if examples else 0.0
    )
    low_saliency_coverage = num_salient_examples < min_salient_examples_per_concept
    return (
        num_salient_examples,
        mean_patch_pred_saliency,
        mean_activation_score,
        mean_retrieval_score,
        low_saliency_coverage,
    )


def select_representative_cluster_record(cluster_records: Sequence[dict]) -> dict:
    """Pick the concept-level representative from per-cluster LLM records."""
    if not cluster_records:
        return {
            "candidate_name": "unclear",
            "explanation": "unclear",
            "confidence": "low",
            "parse_success": False,
            "raw_llm_output": "",
        }

    def sort_key(record: dict) -> Tuple[int, int, int]:
        confidence = str(record.get("confidence", "low")).lower()
        return (
            int(record.get("cluster_size", 0)),
            1 if record.get("parse_success") else 0,
            CONFIDENCE_RANK.get(confidence, 0),
        )

    return max(cluster_records, key=sort_key)


def _build_clustering_metadata(
    *,
    cluster_examples: bool,
    cluster_feature: str,
    max_clusters: int,
    diff_image_size: int,
    clip_model_name: str,
    clip_image_source: str = "large_crop_t",
    clip_crop_scale: float = 3.0,
) -> dict:
    if not cluster_examples:
        return {"enabled": False}

    if cluster_feature == "clip":
        return {
            "enabled": True,
            "feature_type": "clip",
            "feature": "clip_visual_embedding",
            "clip_model": clip_model_name,
            "clip_image_source": clip_image_source,
            "clip_crop_scale": clip_crop_scale,
            "cluster_algorithm": "AgglomerativeClustering cosine average linkage",
            "max_clusters": max_clusters,
        }

    return {
        "enabled": True,
        "feature_type": "diff",
        "feature": "abs(frame_t1 - frame_t) grayscale 2D flattened",
        "max_clusters": max_clusters,
        "diff_image_size": diff_image_size,
    }


def build_combined_concept_explanation(
    *,
    concept_id: str,
    global_concept_index: int,
    concept_type: str,
    concept_number: int,
    llm_model: str,
    top_k: int,
    cluster_records: Sequence[dict],
    representative: dict,
    all_examples: Sequence[SavedExample],
    min_salient_examples_per_concept: int,
    saliency_filter_mode: str,
    saliency_threshold: float,
    saliency_top_percent: float,
    difference_clustering_enabled: bool,
    max_diff_clusters: int,
    diff_image_size: int,
    cluster_feature: str = "clip",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    clip_image_source: str = "large_crop_t",
    clip_crop_scale: float = 3.0,
) -> dict:
    (
        num_salient_examples,
        mean_patch_pred_saliency,
        mean_activation_score,
        mean_retrieval_score,
        low_saliency_coverage,
    ) = compute_example_statistics(all_examples, min_salient_examples_per_concept)

    return {
        "concept_id": concept_id,
        "global_concept_index": global_concept_index,
        "concept_type": concept_type,
        "concept_number": concept_number,
        "llm_model": llm_model,
        "top_k": top_k,
        "num_clusters": len(cluster_records),
        "difference_clustering_enabled": difference_clustering_enabled,
        "cluster_explanations": list(cluster_records),
        "candidate_name": str(representative.get("candidate_name", "unclear")),
        "explanation": str(representative.get("explanation", "unclear")),
        "confidence": str(representative.get("confidence", "low")),
        "parse_success": bool(representative.get("parse_success", False)),
        "raw_llm_output": str(representative.get("raw_llm_output", "")),
        "num_salient_examples": num_salient_examples,
        "mean_patch_pred_saliency": mean_patch_pred_saliency,
        "mean_activation_score": mean_activation_score,
        "mean_retrieval_score": mean_retrieval_score,
        "saliency_filter_mode": saliency_filter_mode,
        "saliency_threshold": saliency_threshold,
        "saliency_top_percent": saliency_top_percent,
        "low_saliency_coverage": low_saliency_coverage,
        "top_examples_dir": "top_examples",
        "top_examples_metadata": "top_examples/examples_metadata.json",
        "clustering": _build_clustering_metadata(
            cluster_examples=difference_clustering_enabled,
            cluster_feature=cluster_feature,
            max_clusters=max_diff_clusters,
            diff_image_size=diff_image_size,
            clip_model_name=clip_model_name,
            clip_image_source=clip_image_source,
            clip_crop_scale=clip_crop_scale,
        ),
        "difference_clustering": {
            "enabled": difference_clustering_enabled,
            "feature": "abs(frame_t1 - frame_t) grayscale 2D flattened",
            "max_clusters": max_diff_clusters,
            "diff_image_size": diff_image_size,
        },
    }


def generate_qwen3_response(llm: LLMHandle, prompt: str, image_paths: Sequence[str], max_new_tokens: int = 256) -> str:
    """Generate one structured response using Qwen3-VL."""

    content = []
    for p in image_paths:
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    processor = llm.processor
    model = llm.model

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        # Fallback for processors that accept PIL images directly.
        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(
            text=[text],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Remove prompt tokens before decoding when possible.
    input_len = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def describe_concepts_with_llm(
    saved_examples: Dict[int, List[SavedExample]],
    concepts_root: Path,
    num_transition_concepts: int,
    num_persistence_concepts: int,
    output_dir: Path,
    llm_name: str,
    max_new_tokens: int,
    top_k: int,
    saliency_filter_mode: str,
    saliency_threshold: float,
    saliency_top_percent: float,
    min_salient_examples_per_concept: int = 1,
    concept_indices: Optional[Sequence[int]] = None,
    cluster_examples: bool = True,
    max_diff_clusters: int = 4,
    diff_image_size: int = 64,
    cluster_feature: str = "clip",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    clip_batch_size: int = 16,
    clip_image_source: str = "large_crop_t",
    clip_crop_scale: float = 3.0,
) -> List[dict]:
    llm: Optional[LLMHandle] = None
    clip_model = None
    clip_processor = None
    clip_device: Optional[torch.device] = None
    descriptions: List[dict] = []
    ensure_dir(concepts_root)
    indices = (
        list(concept_indices)
        if concept_indices is not None
        else list(range(num_transition_concepts + num_persistence_concepts))
    )

    for c_idx in tqdm(indices, desc="LLM concept descriptions"):
        concept_type, local_idx = get_concept_type_and_local_index(
            c_idx,
            num_transition_concepts,
            num_persistence_concepts,
        )
        concept_id = format_concept_id(
            c_idx,
            num_transition_concepts,
            num_persistence_concepts,
        )
        concept_dir = ensure_dir(get_concept_dir(concepts_root, concept_type, local_idx))
        explanation_path = concept_dir / "explanation.json"
        metadata_path = concept_dir / "top_examples" / "examples_metadata.json"

        examples = saved_examples.get(c_idx)
        if not examples and metadata_path.exists():
            examples = load_saved_examples_from_metadata(
                concept_dir,
                metadata_path,
                c_idx,
            )
        if not examples:
            continue

        if llm is None:
            from model.llm import load_llm

            llm = load_llm(llm_name)

        if cluster_examples and cluster_feature == "clip" and clip_model is None:
            clip_model, clip_processor, clip_device = load_clip_model(
                model_name=clip_model_name,
            )

        image_size = (diff_image_size, diff_image_size)
        if cluster_examples:
            if cluster_feature == "clip":
                clustered_examples = cluster_examples_by_clip_features(
                    concept_dir=concept_dir,
                    examples=examples,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=clip_device,
                    max_clusters=max_diff_clusters,
                    clip_batch_size=clip_batch_size,
                    model_name=clip_model_name,
                    clip_image_source=clip_image_source,
                    clip_crop_scale=clip_crop_scale,
                )
            else:
                clustered_examples = cluster_examples_by_patch_difference(
                    concept_dir=concept_dir,
                    examples=examples,
                    max_clusters=max_diff_clusters,
                    image_size=image_size,
                    save_diff_images=True,
                )
        else:
            clustered_examples = {0: list(examples)}

        cluster_explanations_dir = ensure_dir(concept_dir / "cluster_explanations")
        cluster_records: List[dict] = []

        for cluster_id, cluster_examples_list in sorted(clustered_examples.items()):
            if not cluster_examples_list:
                continue

            (
                num_salient_examples,
                mean_patch_pred_saliency,
                mean_activation_score,
                mean_retrieval_score,
                low_saliency_coverage,
            ) = compute_example_statistics(
                cluster_examples_list,
                min_salient_examples_per_concept,
            )

            prompt = build_concept_explanation_prompt(
                concept_id,
                len(cluster_examples_list),
            )
            llm_contact_sheet_path = collect_llm_contact_sheet_path(
                concept_dir,
                cluster_examples_list,
                examples,
                cluster_id,
            )
            if llm_contact_sheet_path is None:
                print(
                    f"WARNING: skipping LLM for {concept_id} cluster {cluster_id}: "
                    "no contact sheet available"
                )
                continue

            raw_text = generate_qwen3_response(
                llm,
                prompt,
                [llm_contact_sheet_path],
                max_new_tokens=max_new_tokens,
            )

            parse_success = False
            parsed_llm: Optional[dict] = None
            parse_error: Optional[str] = None
            try:
                parsed_llm = parse_json_from_text(raw_text)
                parse_success = True
            except Exception as exc:
                parse_error = str(exc)
                print(
                    f"WARNING: failed to parse LLM JSON for {concept_id} "
                    f"cluster {cluster_id}: {parse_error}"
                )

            cluster_record = build_explanation_record(
                concept_id=concept_id,
                global_concept_index=c_idx,
                concept_type=concept_type,
                concept_number=local_idx,
                parsed_llm=parsed_llm,
                raw_llm_output=raw_text,
                parse_success=parse_success,
                llm_model=llm.model_name,
                top_k=top_k,
                num_salient_examples=num_salient_examples,
                mean_patch_pred_saliency=mean_patch_pred_saliency,
                mean_activation_score=mean_activation_score,
                mean_retrieval_score=mean_retrieval_score,
                saliency_filter_mode=saliency_filter_mode,
                saliency_threshold=saliency_threshold,
                saliency_top_percent=saliency_top_percent,
                low_saliency_coverage=low_saliency_coverage,
                parse_error=parse_error,
            )
            cluster_record.update(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": len(cluster_examples_list),
                    "cluster_examples": [
                        {"rank": example.rank, "pair_path": example.pair_path}
                        for example in cluster_examples_list
                    ],
                    "llm_contact_sheet_path": _relative_to_concept_dir(
                        concept_dir,
                        Path(llm_contact_sheet_path),
                    ),
                    "clustering": _build_clustering_metadata(
                        cluster_examples=cluster_examples,
                        cluster_feature=cluster_feature,
                        max_clusters=max_diff_clusters,
                        diff_image_size=diff_image_size,
                        clip_model_name=clip_model_name,
                        clip_image_source=clip_image_source,
                        clip_crop_scale=clip_crop_scale,
                    ),
                }
            )
            cluster_records.append(cluster_record)

            cluster_path = (
                cluster_explanations_dir / f"cluster_{int(cluster_id):02d}_explanation.json"
            )
            cluster_path.write_text(json.dumps(cluster_record, indent=2), encoding="utf-8")

        representative = select_representative_cluster_record(cluster_records)
        explanation = build_combined_concept_explanation(
            concept_id=concept_id,
            global_concept_index=c_idx,
            concept_type=concept_type,
            concept_number=local_idx,
            llm_model=llm.model_name,
            top_k=top_k,
            cluster_records=cluster_records,
            representative=representative,
            all_examples=examples,
            min_salient_examples_per_concept=min_salient_examples_per_concept,
            saliency_filter_mode=saliency_filter_mode,
            saliency_threshold=saliency_threshold,
            saliency_top_percent=saliency_top_percent,
            difference_clustering_enabled=cluster_examples,
            max_diff_clusters=max_diff_clusters,
            diff_image_size=diff_image_size,
            cluster_feature=cluster_feature,
            clip_model_name=clip_model_name,
            clip_image_source=clip_image_source,
            clip_crop_scale=clip_crop_scale,
        )
        descriptions.append(explanation)

        explanation_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")

        # Incremental aggregate save so progress is not lost if generation stops.
        out_path = output_dir / "concept_descriptions.json"
        out_path.write_text(json.dumps(descriptions, indent=2), encoding="utf-8")

    return descriptions


# -----------------------------
# Main retrieval loop
# -----------------------------


def retrieve_and_save_top_examples(
    model: ExplainableVidSalModel,
    loader: DataLoader,
    device: torch.device,
    concepts_root: Path,
    output_dir: Path,
    num_concepts: int,
    num_transition_concepts: int,
    num_persistence_concepts: int,
    top_k: int,
    activation_key: Optional[str],
    max_batches: Optional[int],
    max_concepts: Optional[int],
    dry_run_discover: bool,
    overwrite: bool,
    skip_existing: bool,
    saliency_filter_mode: str,
    saliency_threshold: float,
    saliency_top_percent: float,
    saliency_source: str = "predicted",
    use_amp: bool = False,
    save_every_n_batches: int = 0,
) -> Dict[int, List[SavedExample]]:
    """Retrieve top examples; defer disk writes until the dataset pass completes."""

    model.eval()
    ensure_dir(concepts_root)
    concept_indices = resolve_concept_indices(num_concepts, max_concepts)
    retrieve_indices, saved_examples = partition_concepts_for_retrieval(
        concept_indices=concept_indices,
        concepts_root=concepts_root,
        num_transition_concepts=num_transition_concepts,
        num_persistence_concepts=num_persistence_concepts,
        skip_existing=skip_existing,
        overwrite=overwrite,
    )
    heaps: Dict[int, List[HeapItem]] = {c_idx: [] for c_idx in retrieve_indices}
    serial = 0

    if max_concepts is not None:
        print(
            f"Processing {len(concept_indices)} concept(s) "
            f"(global indices {concept_indices[0]}..{concept_indices[-1]})."
        )
    if retrieve_indices:
        save_note = (
            f"every {save_every_n_batches} batch(es)"
            if save_every_n_batches > 0
            else "once at end of dataset pass"
        )
        amp_note = "on" if use_amp and device.type == "cuda" else "off"
        print(
            f"Retrieving top examples for {len(retrieve_indices)} concept(s); "
            f"saving {save_note}; AMP {amp_note}."
        )
    elif concept_indices:
        print("All requested concepts already have saved top_examples; skipping retrieval.")

    amp_enabled = use_amp and device.type == "cuda"
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Retrieving top concept examples")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if not retrieve_indices:
                break

            video_filenames, rgb_batch, sal_batch, fix_batch, n_frames, valid_mask = batch
            rgb_device = rgb_batch.to(device, non_blocking=True)
            sal_device = (
                sal_batch.to(device, non_blocking=True) if torch.is_tensor(sal_batch) else None
            )

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                model_out = model(
                    rgb_device,
                    saliency_maps=sal_device,
                    return_details=True,
                    return_concept_losses=False,
                )

            if batch_idx == 0 and dry_run_discover:
                discover_concept_tensors(model_out)
                raise SystemExit(
                    "Dry-run discovery complete. Pass --activation-key or rerun without --dry-run-discover."
                )

            selected_name, activations, grid_hw = resolve_activation_tensor(
                model_out,
                activation_key=activation_key,
                num_transition_concepts=num_transition_concepts,
                num_persistence_concepts=num_persistence_concepts,
                num_concepts=num_concepts,
            )
            if batch_idx == 0:
                print(
                    f"Using activation tensor '{selected_name}' with shape "
                    f"{tuple(activations.shape)} grid={grid_hw}"
                )
                if saliency_source == "predicted":
                    predicted_patch_sal = extract_predicted_patch_saliency(
                        model_out,
                        grid_hw,
                    )
                    print(
                        "Saliency retrieval (predicted patch saliency): "
                        f"mode={saliency_filter_mode} "
                        f"threshold={saliency_threshold} "
                        f"top_percent={saliency_top_percent} "
                        f"shape={tuple(predicted_patch_sal.shape)}"
                    )
                else:
                    gt_saliency_maps = extract_gt_saliency_batch(sal_batch)
                    print(
                        "Saliency retrieval (ground-truth saliency map): "
                        f"mode={saliency_filter_mode} "
                        f"threshold={saliency_threshold} "
                        f"top_percent={saliency_top_percent} "
                        f"shape={tuple(gt_saliency_maps.shape)}"
                    )

            heap_kwargs: Dict[str, Any] = {
                "heaps": heaps,
                "activations": activations,
                "rgb_batch": rgb_batch,
                "video_filenames": video_filenames,
                "grid_hw": grid_hw,
                "top_k": top_k,
                "serial_start": serial,
                "concept_indices": retrieve_indices,
                "saliency_filter_mode": saliency_filter_mode,
                "saliency_threshold": saliency_threshold,
                "saliency_top_percent": saliency_top_percent,
            }
            if saliency_source == "predicted":
                heap_kwargs["patch_saliency"] = extract_predicted_patch_saliency(
                    model_out,
                    grid_hw,
                )
            else:
                heap_kwargs["saliency_maps"] = extract_gt_saliency_batch(sal_batch)

            serial = update_heaps_from_activations(**heap_kwargs)

            if (
                save_every_n_batches > 0
                and retrieve_indices
                and (batch_idx + 1) % save_every_n_batches == 0
            ):
                interim_saved = save_retrieved_examples(
                    heaps=heaps,
                    concepts_root=concepts_root,
                    num_transition_concepts=num_transition_concepts,
                    num_persistence_concepts=num_persistence_concepts,
                    concept_indices=retrieve_indices,
                    overwrite=True,
                    skip_existing=False,
                )
                saved_examples.update(interim_saved)

            del model_out, rgb_device, sal_device

    if retrieve_indices:
        newly_saved = save_retrieved_examples(
            heaps=heaps,
            concepts_root=concepts_root,
            num_transition_concepts=num_transition_concepts,
            num_persistence_concepts=num_persistence_concepts,
            concept_indices=retrieve_indices,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        saved_examples.update(newly_saved)

    write_retrieved_examples_aggregate(saved_examples, output_dir)
    return saved_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve and verbalize learned concept prototypes.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to best_checkpoint.pth or last_checkpoint.pth (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument("--dataset-dir", required=True, help="Dataset split directory to scan")
    parser.add_argument("--output-dir", default="explanation_outputs", help="Output directory")
    parser.add_argument(
        "--concepts-root",
        default=DEFAULT_CONCEPTS_ROOT,
        help="Root directory for per-concept explanations and top examples",
    )
    parser.add_argument(
        "--num-transition-concepts",
        type=int,
        default=128,
        help="Number of transition concepts (global indices 0..N-1)",
    )
    parser.add_argument(
        "--num-persistence-concepts",
        type=int,
        default=128,
        help="Number of persistence concepts (global indices N..N+M-1)",
    )
    parser.add_argument("--window-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-concepts", type=int, default=512)
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help=(
            "If set, only process the first N global concept indices "
            "(retrieve/save examples, run LLM, write summary) and then stop."
        ),
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of examples per concept")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional subset size")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional max batches for quick testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--activation-key",
        default=None,
        help=(
            "Dot path inside model_out['concept_out'] for per-concept activation tensor, "
            "e.g. stage4.concept_activations. If omitted, script attempts auto-discovery."
        ),
    )
    parser.add_argument("--dry-run-discover", action="store_true", help="Print concept_out tensors and exit")
    parser.add_argument("--skip-llm", action="store_true", help="Only retrieve top examples; do not run the LLM")
    parser.add_argument("--llm-name", default="qwen3", help="Short LLM name passed to load_llm")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--cluster-examples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cluster top examples before sending each cluster to the LLM."
        ),
    )
    parser.add_argument(
        "--cluster-feature",
        choices=["diff", "clip"],
        default="clip",
        help="Feature type used to cluster top examples before LLM explanation.",
    )
    parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model used for visual feature clustering.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=16,
        help="Batch size for CLIP visual embedding extraction.",
    )
    parser.add_argument(
        "--clip-image-source",
        choices=["frame_t", "large_crop_t"],
        default="large_crop_t",
        help="Image source used only for CLIP feature extraction during clustering.",
    )
    parser.add_argument(
        "--clip-crop-scale",
        type=float,
        default=3.0,
        help="Scale factor for enlarged crop around the activated patch for CLIP clustering.",
    )
    parser.add_argument(
        "--max-diff-clusters",
        type=int,
        default=4,
        help="Maximum number of clusters per concept.",
    )
    parser.add_argument(
        "--diff-image-size",
        type=int,
        default=64,
        help=(
            "Resize frame_t/frame_t1 patches to this square size before "
            "computing difference features."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing top_examples when present (default: True). "
            "explanation.json is always regenerated when the LLM step runs. "
            "--overwrite takes priority for top_examples."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate top examples and explanation.json even when they already exist",
    )
    parser.add_argument(
        "--saliency-source",
        choices=["predicted", "gt"],
        default="predicted",
        help=(
            "Saliency signal used to define salient patches during retrieval: "
            "model-predicted patch saliency (default) or ground-truth maps."
        ),
    )
    parser.add_argument(
        "--saliency-filter-mode",
        choices=["none", "gt_threshold", "gt_top_percent", "gt_weighted"],
        default="gt_top_percent",
        help="How to restrict top examples to salient patches (uses --saliency-source maps)",
    )
    parser.add_argument(
        "--saliency-threshold",
        type=float,
        default=0.0,
        help="Saliency threshold for gt_threshold / optional gt_weighted filtering",
    )
    parser.add_argument(
        "--saliency-top-percent",
        type=float,
        default=0.20,
        help="Keep top fraction of salient patches per sample (gt_top_percent mode)",
    )
    parser.add_argument(
        "--min-salient-examples-per-concept",
        type=int,
        default=1,
        help="Flag low_saliency_coverage when fewer salient examples are retrieved",
    )
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use autocast during model forward (default: on when CUDA is available)",
    )
    parser.add_argument(
        "--save-every-n-batches",
        type=int,
        default=0,
        help="Flush top examples to disk every N batches (0 = only at end of dataset pass)",
    )
    return parser.parse_args()


CONCEPT_DIR_PATTERN = re.compile(r"^c_(tr|per)_(\d+)$")

CONCEPT_SUMMARY_CSV_COLUMNS = [
    "concept_id",
    "global_concept_index",
    "concept_type",
    "concept_number",
    "candidate_name",
    "explanation",
    "confidence",
    "num_salient_examples",
    "mean_patch_pred_saliency",
    "mean_activation_score",
    "mean_retrieval_score",
    "saliency_filter_mode",
    "parse_success",
    "explanation_json",
    "top_examples_dir",
]


def build_concept_summary_entry(
    concept_dir: Path,
    explanation: dict,
    num_transition_concepts: int,
) -> dict:
    """Build one concept_summary record from a concept directory and explanation.json."""

    concept_id = concept_dir.name
    match = CONCEPT_DIR_PATTERN.match(concept_id)
    if not match:
        raise ValueError(f"Unexpected concept directory name: {concept_id}")

    concept_type = str(explanation.get("concept_type", match.group(1)))
    concept_number = int(explanation.get("concept_number", int(match.group(2))))
    global_concept_index = int(
        explanation.get(
            "global_concept_index",
            concept_number
            if concept_type == "tr"
            else num_transition_concepts + concept_number,
        )
    )

    return {
        "concept_id": str(explanation.get("concept_id", concept_id)),
        "global_concept_index": global_concept_index,
        "concept_type": concept_type,
        "concept_number": concept_number,
        "candidate_name": str(explanation.get("candidate_name", "unclear")),
        "explanation": str(
            explanation.get("explanation")
            or explanation.get("saliency_shift_explanation", "unclear")
        ),
        "confidence": str(explanation.get("confidence", "low")),
        "num_salient_examples": int(explanation.get("num_salient_examples", 0)),
        "mean_patch_pred_saliency": float(
            explanation.get(
                "mean_patch_pred_saliency",
                explanation.get("mean_patch_gt_saliency", 0.0),
            )
        ),
        "mean_activation_score": float(explanation.get("mean_activation_score", 0.0)),
        "mean_retrieval_score": float(explanation.get("mean_retrieval_score", 0.0)),
        "saliency_filter_mode": str(explanation.get("saliency_filter_mode", "none")),
        "parse_success": bool(explanation.get("parse_success", False)),
        "explanation_json": f"{concept_id}/explanation.json",
        "top_examples_dir": f"{concept_id}/top_examples",
    }


def scan_concept_summaries(
    concepts_root: Path,
    num_transition_concepts: int,
    concept_indices: Optional[Sequence[int]] = None,
) -> List[dict]:
    """Scan concept directories and collect summary records from explanation.json files."""

    summaries: List[dict] = []
    allowed_indices = set(concept_indices) if concept_indices is not None else None

    for concept_dir in sorted(concepts_root.iterdir()):
        if not concept_dir.is_dir() or not CONCEPT_DIR_PATTERN.match(concept_dir.name):
            continue

        explanation_path = concept_dir / "explanation.json"
        if not explanation_path.exists():
            continue

        explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
        if (
            allowed_indices is not None
            and int(explanation.get("global_concept_index", -1)) not in allowed_indices
        ):
            continue

        summaries.append(
            build_concept_summary_entry(
                concept_dir,
                explanation,
                num_transition_concepts,
            )
        )

    summaries.sort(key=lambda entry: entry["global_concept_index"])
    return summaries


def write_concept_summary(
    concepts_root: Path,
    num_transition_concepts: int,
    concept_indices: Optional[Sequence[int]] = None,
) -> List[dict]:
    """Regenerate concept_summary.json and concept_summary.csv under concepts_root."""

    summaries = scan_concept_summaries(
        concepts_root,
        num_transition_concepts,
        concept_indices=concept_indices,
    )

    json_path = concepts_root / "concept_summary.json"
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    csv_path = concepts_root / "concept_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONCEPT_SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for entry in summaries:
            writer.writerow(
                {
                    **entry,
                    "parse_success": str(entry["parse_success"]).lower(),
                }
            )

    return summaries


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concepts_root = ensure_dir(args.concepts_root)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    dataset = DatasetLoader(args.dataset_dir, window_len=args.window_len, stride=args.stride)
    if args.max_samples is not None and args.max_samples < len(dataset):
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(dataset), generator=generator)[: args.max_samples].tolist()
        dataset = Subset(dataset, indices)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": video_saliency_collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "shuffle": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(dataset, batch_size=args.batch_size, **loader_kwargs)
    print(f"Dataset windows: {len(dataset)} | batch_size={args.batch_size}")

    checkpoint_path = str(resolve_checkpoint_path(args.checkpoint or DEFAULT_CHECKPOINT))
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_saliency_model(checkpoint_path, device=device)

    concept_indices = resolve_concept_indices(args.num_concepts, args.max_concepts)

    use_amp = args.use_amp if args.use_amp is not None else device.type == "cuda"
    saved_examples = retrieve_and_save_top_examples(
        model=model,
        loader=loader,
        device=device,
        concepts_root=concepts_root,
        output_dir=output_dir,
        num_concepts=args.num_concepts,
        num_transition_concepts=args.num_transition_concepts,
        num_persistence_concepts=args.num_persistence_concepts,
        top_k=args.top_k,
        activation_key=args.activation_key,
        max_batches=args.max_batches,
        max_concepts=args.max_concepts,
        dry_run_discover=args.dry_run_discover,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        saliency_filter_mode=args.saliency_filter_mode,
        saliency_threshold=args.saliency_threshold,
        saliency_top_percent=args.saliency_top_percent,
        saliency_source=args.saliency_source,
        use_amp=use_amp,
        save_every_n_batches=args.save_every_n_batches,
    )
    print(f"Saved concept top examples under: {concepts_root}")
    print(f"Saved metadata to: {output_dir / 'retrieved_examples.json'}")

    if args.skip_llm:
        print("Skipping LLM generation because --skip-llm was set.")
    else:
        descriptions = describe_concepts_with_llm(
            saved_examples=saved_examples,
            concepts_root=concepts_root,
            num_transition_concepts=args.num_transition_concepts,
            num_persistence_concepts=args.num_persistence_concepts,
            output_dir=output_dir,
            llm_name=args.llm_name,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            saliency_filter_mode=args.saliency_filter_mode,
            saliency_threshold=args.saliency_threshold,
            saliency_top_percent=args.saliency_top_percent,
            min_salient_examples_per_concept=args.min_salient_examples_per_concept,
            concept_indices=concept_indices,
            cluster_examples=args.cluster_examples,
            max_diff_clusters=args.max_diff_clusters,
            diff_image_size=args.diff_image_size,
            cluster_feature=args.cluster_feature,
            clip_model_name=args.clip_model_name,
            clip_batch_size=args.clip_batch_size,
            clip_image_source=args.clip_image_source,
            clip_crop_scale=args.clip_crop_scale,
        )
        print(f"Saved {len(descriptions)} concept explanations under: {concepts_root}")
        print(f"Saved aggregate descriptions to: {output_dir / 'concept_descriptions.json'}")

    summaries = write_concept_summary(
        concepts_root=concepts_root,
        num_transition_concepts=args.num_transition_concepts,
        concept_indices=concept_indices,
    )
    print(f"Saved concept summary ({len(summaries)} concepts) to: {concepts_root / 'concept_summary.json'}")
    print(f"Saved concept summary CSV to: {concepts_root / 'concept_summary.csv'}")


if __name__ == "__main__":
    main()
