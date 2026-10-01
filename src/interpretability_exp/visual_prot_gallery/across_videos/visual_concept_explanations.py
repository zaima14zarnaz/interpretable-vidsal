"""Generate prototype-level natural-language descriptions for a concept model.

This script:
  1. Loads the trained ExplainableVidSalModel from a checkpoint.
  2. Runs the model over a dataset and retrieves top-K target-frame patch examples for
     every visual concept prototype using visual_concept_logits from
     model_out["concept_out"] with target-step decoder top-k assignment masking
     (patch-level appearance concepts only).
  3. Optionally runs a two-stage LLM pipeline and writes per-sample and per-concept
     explanations.
  4. Regenerates global summary files by scanning all existing concept directories.

Output structure
----------------
Under --concepts-root (default: dh1k/concepts):

  {concepts_root}/
    concept_summary.json          # list of all concept explanations (regenerated each run)
    concept_summary.csv           # same fields in CSV form
    c_tr_000/
      top_examples/
        examples_metadata.json
        contact_sheet.jpg           # top-K individual patch crops
        gallery.jpg                 # the same K full frames side by side, red patch boxes
        retrieval_settings.json
      sample_descriptions/
        sample_000_description.json
        sample_001_description.json
        ...
      explanation.json

During LLM explanation, the script uses a two-stage no-clustering pipeline:
  1. Each top activated example is sent individually to the multimodal LLM using its
     single top-activated patch crop at the target frame, extracted in memory.
  2. The per-sample descriptions are aggregated by the LLM in a text-only second pass to
     produce the concept-level explanation.

Notes:
  - gallery.jpg preserves each selected full frame and marks the matching patch in red.
  - No individual example panel or crop-pair image files are created.
  - The optional LLM receives only the selected patch cell extracted in memory
    from contact_sheet.jpg, not the entire contact sheet or gallery.
  - No clustering is used in this multi-stage file.
  - The concept-level explanation is based on the aggregation of individual per-sample
    descriptions.

Top examples use target-step decoder top-k assignments and one strongest match
per distinct video. --salient-patches-only defaults to false: all assigned
matches are eligible, both salient and non-salient. Set --salient-patches-only
(or --salient-patches-only true) to restrict to the top --saliency-top-percent
salient patches per sample (20 percent by default; predicted saliency by
default, --saliency-source gt selects ground truth). An explicit false disables
filtering. No four-candidate batch cap is used.

When multi-index top-k assignments are exported, they are used directly;
otherwise membership is reconstructed from the full visual_concept_logits
bank with MODEL_TOP_K, the same setting passed to build_model(). Target-step
filtering uses local visual_metadata.time_idx and feature_shape.T. Each entry
is associated with the last RGB frame in its input window. video_filenames
must contain stable video identities, not frame/window filenames. Gallery crops
and red boxes use backbone-resized RGB (same trilinear spatial resize as training,
without ImageNet normalization). Metadata records gallery_input_hw,
native_input_hw, activated_patch_box (gallery pixels), and
activated_patch_box_native when mapped back to dataloader resolution.

Examples from earlier retrieval versions or different settings are regenerated
automatically. --skip-llm still leaves all LLM generation disabled.

    c_vis_000/
      ...

Per-concept explanation.json contains the aggregated concept description plus
sample_descriptions, sample_description_paths, saliency retrieval metadata, llm_model,
top_k, raw_llm_output, parse_success, and related fields.

Under --output-dir (default: explanation_outputs):

  retrieved_examples.json         # serializable copy of newly saved top examples
  concept_descriptions.json       # incremental aggregate of explanations generated this run

Pipeline
--------
For each selected concept, the script retrieves top examples across the dataset (heap
updates defer image cropping until save time), then writes top_examples to disk once at
the end (or every --save-every-n-batches). Then it optionally runs the two-stage LLM
pipeline and writes sample_descriptions/, explanation.json, plus a global
concept_summary.json / .csv.

Use --max-concepts N to limit processing to global concept indices 0..N-1 and stop.

Resumability
------------
By default (--skip-existing), the script reuses existing top_examples/ files.
With --reuse-sample-descriptions (default), existing sample_descriptions/ files are
reused and only the aggregation step is rerun. explanation.json is always regenerated
when the LLM step runs. Pass --overwrite to also regenerate top_examples. Use --skip-llm
to retrieve top examples only.

Discover activation key
-----------------------
The per-concept activation tensor lives inside model_out["concept_out"] under each stage's
visual_concept_logits, visual_concept_indices, and visual_metadata. If the key is unknown,
discover it first with --dry-run-discover:

  python explanation_generation_multi_stage.py \\
    --checkpoint training_outputs/best_checkpoint.pth \\
    --dataset-dir /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/testing \\
    --dry-run-discover

Full run (retrieval + two-stage LLM explanations)
-------------------------------------------------
  python explanation_generation_multi_stage.py \\
    --checkpoint training_outputs/best_checkpoint.pth \\
    --dataset-dir /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/testing \\
    --concepts-root /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/concepts \\
    --activation-key stage4.visual_concept_logits \\
    --top-k 8 \\
    --llm-name qwen3 \\
    --sample-max-new-tokens 256 \\
    --aggregate-max-new-tokens 512

Retrieval only (no LLM)
-----------------------
  python interpretability_exp/visual_prot_gallery/within_videos/visual_concept_explanations.py \\
    --checkpoint training_outputs/saved_weights/best_dhf1k.pth \\
    --dataset-dir /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/testing \\
    --concepts-root /data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/concepts \\
    --activation-key stage4.visual_concept_logits \\
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

# This file lives at src/interpretability_exp/visual_prot_gallery/within_videos/.
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from pre_process.collate import video_saliency_collate_fn
from pre_process.dataloader import DatasetLoader
from model.model import ExplainableVidSalModel
import train as train_cfg

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover - optional dependency
    process_vision_info = None


# -----------------------------
# Reproducibility / model setup
# -----------------------------

DEFAULT_CHECKPOINT = "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/best_dhf1k.pth"
DEFAULT_CONCEPTS_ROOT = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/concepts"
)
NUM_EXAMPLE_FRAMES = 5
CONTACT_SHEET_COLS = 3
CONTACT_SHEET_CELL_SIZE = (128, 128)
GALLERY_CELL_SIZE = (384, 256)
GALLERY_SPACING = 12
JPEG_QUALITY = 95
STAGE1_SAMPLE_MAX_NEW_TOKENS = 256
STAGE1_SAMPLE_RETRY_MAX_NEW_TOKENS = 512
STAGE2_AGGREGATE_MAX_NEW_TOKENS = 512
STAGE2_AGGREGATE_RETRY_MAX_NEW_TOKENS = 1024

# Prototypes kept per patch for decoder priority scoring. Not stored in the
# checkpoint, so it must match the value hard-coded in train.py.
MODEL_TOP_K = 8


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
    *,
    concept_source: str = "visual",
    num_concepts: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Map a global concept index to (concept_type, local_index).

    When concept_source is "visual" (default), indices map to visual concepts
    c_vis_{idx}. When "trajectory", indices map to transition/persistence concepts.
    """
    if global_concept_idx < 0:
        raise ValueError(f"global_concept_idx must be >= 0, got {global_concept_idx}")

    if concept_source == "visual":
        total = (
            num_concepts
            if num_concepts is not None
            else num_transition_concepts + num_persistence_concepts
        )
        if global_concept_idx >= total:
            raise ValueError(
                f"global_concept_idx {global_concept_idx} out of range for "
                f"{total} visual concepts"
            )
        return "vis", global_concept_idx

    if concept_source != "trajectory":
        raise ValueError(
            f"concept_source must be 'visual' or 'trajectory', got {concept_source!r}"
        )

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
    """Return concepts_root/c_{tr|per|vis}_{number:03d}."""
    if concept_type not in {"tr", "per", "vis"}:
        raise ValueError(
            f"concept_type must be 'tr', 'per', or 'vis', got {concept_type!r}"
        )
    return concepts_root / f"c_{concept_type}_{concept_number:03d}"


def format_concept_id(
    global_concept_idx: int,
    num_transition_concepts: int,
    num_persistence_concepts: int,
    *,
    concept_source: str = "visual",
    num_concepts: Optional[int] = None,
) -> str:
    concept_type, local_idx = get_concept_type_and_local_index(
        global_concept_idx,
        num_transition_concepts,
        num_persistence_concepts,
        concept_source=concept_source,
        num_concepts=num_concepts,
    )
    return f"c_{concept_type}_{local_idx:03d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device: torch.device) -> ExplainableVidSalModel:
    """Build the model with the same architecture arguments as train.py."""

    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=True,
        freeze_backbone=True,
        backbone_gradient_checkpointing=False,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=128,
        num_concepts=512,
        concept_hidden_dim=256,
        saliency_hidden_dim=256,
        top_k=MODEL_TOP_K,
        max_source_patches=64,
        tau_pi=0.5,
        tau_alpha=0.07,
        tau_concept=0.07,
        concept_residual_weight=0.0,
        last_transition_only=True,
        use_rgb_refinement=False,
        use_feature_refinement=False,
        # Training uses "none"; activation has no parameters, and sigmoid makes
        # saliency_map a [0, 1] map for the saliency-filter fallback.
        output_activation="sigmoid",
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
        prototype_application_position=train_cfg.PROTOTYPE_APPLICATION_POSITION,
        fine_unary_mask_strength=train_cfg.FINE_UNARY_MASK_STRENGTH,
        stage2_backbone_fusion_enabled=train_cfg.STAGE2_BACKBONE_FUSION_ENABLED,
    ).to_split_devices(device, device)
    model.eval()
    return model


def _filter_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str]]:
    """Keep checkpoint tensors whose keys and shapes match the live model."""
    model_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    shape_mismatch: List[str] = []
    unexpected: List[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append(key)
            continue
        compatible[key] = value
    missing = [
        key
        for key in model_state
        if key not in compatible and key not in shape_mismatch
    ]
    return compatible, missing, unexpected, shape_mismatch


def load_saliency_model(checkpoint_path: str, device: torch.device) -> ExplainableVidSalModel:
    resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
    model = build_model(device)
    ckpt = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    compatible, missing, unexpected, shape_mismatch = _filter_compatible_state_dict(
        model,
        state_dict,
    )
    model.load_state_dict(compatible, strict=False)
    if shape_mismatch:
        print(
            f"WARNING: skipped {len(shape_mismatch)} checkpoint tensors with "
            "incompatible shapes"
        )
        print("  first shape mismatches:", shape_mismatch[:10])
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


def _find_stage_with_visual_concepts(concept_out: dict) -> str:
    """Return the deepest stage that exposes visual concept logits and metadata."""

    for stage in ("stage4", "stage3", "stage2", "stage1"):
        stage_out = concept_out.get(stage)
        if not isinstance(stage_out, dict):
            continue
        if {
            "visual_metadata",
            "visual_concept_logits",
            "visual_concept_indices",
        }.issubset(stage_out.keys()):
            return stage

    available = {
        stage: sorted(stage_out.keys())
        for stage, stage_out in concept_out.items()
        if isinstance(stage_out, dict)
    }
    raise KeyError(
        "No stage with visual_concept_logits, visual_concept_indices, and visual_metadata. "
        f"Available stage keys: {available}"
    )


def _find_stage_with_activations(concept_out: dict) -> str:
    """Return the deepest stage that exposes trajectory concept activations."""

    for stage in ("stage4", "stage3", "stage2", "stage1"):
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


def _visual_target_rows_and_topk_mask(
    stage_out: Dict[str, Any],
    decoder_top_k: int = MODEL_TOP_K,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """Return full-bank logits, target rows, and decoder top-k eligibility.

    Exported multi-index assignments are preferred. A one-dimensional hard
    assignment is insufficient: reconstruct top-k from the full logits bank
    using the same top_k supplied to build_model(). Do not compute membership
    after truncating the concept bank to --num-concepts.

    time_idx is local to the visual subset; feature_shape.T describes that
    subset. The last local step therefore corresponds to the target step.
    """
    metadata = stage_out["visual_metadata"]
    shape = metadata["feature_shape"]
    bsz, gh, gw = int(shape["B"]), int(shape["H"]), int(shape["W"])
    logits = stage_out["visual_concept_logits"].float()
    if logits.dim() != 2 or logits.shape[1] == 0:
        raise ValueError("visual_concept_logits must have shape [rows, full_concept_bank].")
    device = logits.device
    batch_idx = metadata["batch_idx"].to(device=device, dtype=torch.long)
    patch_idx = metadata["patch_idx"].to(device=device, dtype=torch.long)
    time_idx = metadata["time_idx"].to(device=device, dtype=torch.long)
    for label, index in (("batch_idx", batch_idx), ("patch_idx", patch_idx), ("time_idx", time_idx)):
        if index.dim() != 1 or index.numel() != logits.shape[0]:
            raise ValueError(f"visual_metadata.{label} must have one entry per logit row.")
    if bsz <= 0 or gh <= 0 or gw <= 0:
        raise ValueError("Visual feature dimensions must be positive.")
    if bool(((batch_idx < 0) | (batch_idx >= bsz)).any()):
        raise ValueError("Visual batch indices are outside feature_shape.B.")
    if bool(((patch_idx < 0) | (patch_idx >= gh * gw)).any()):
        raise ValueError("Visual patch indices are outside the spatial grid.")
    if "T" not in shape:
        raise ValueError("visual_metadata.feature_shape.T is required for target-step filtering.")
    feature_t = int(shape["T"])
    if feature_t <= 0 or bool(((time_idx < 0) | (time_idx >= feature_t)).any()):
        raise ValueError("Visual time indices must be local to feature_shape.T.")
    target_rows = time_idx == feature_t - 1
    window_t = int(metadata.get("window_T", feature_t))
    if window_t < feature_t:
        raise ValueError("window_T cannot be smaller than visual feature_shape.T.")

    exported = None
    for key in ("visual_concept_topk_indices", "visual_concept_top_k_indices", "visual_topk_indices"):
        if key in stage_out:
            exported = stage_out[key]
            break
    if exported is None:
        hard_or_topk = stage_out.get("visual_concept_indices")
        if torch.is_tensor(hard_or_topk) and hard_or_topk.dim() == 2 and hard_or_topk.shape[1] > 1:
            exported = hard_or_topk
    if exported is not None:
        if not torch.is_tensor(exported) or exported.dim() != 2 or exported.shape[0] != logits.shape[0]:
            raise ValueError("Exported visual top-k indices must have shape [rows, K].")
        chosen = exported.to(device=device, dtype=torch.long)
        if bool((chosen >= logits.shape[1]).any()) or bool((chosen < -1).any()):
            raise ValueError("Exported top-k indices must be valid bank indices or -1 padding.")
    else:
        if decoder_top_k <= 0:
            raise ValueError("decoder_top_k must be positive.")
        # Use the full bank and the usual decoder torch.topk operation, even
        # when retrieval saves only an initial subset of concepts. Exported
        # indices remain preferable when ties or custom decoder rules exist.
        chosen = torch.topk(logits, k=min(decoder_top_k, logits.shape[1]), dim=-1).indices
    eligible = torch.zeros_like(logits, dtype=torch.bool)
    rows = torch.arange(logits.shape[0], device=device).unsqueeze(1).expand_as(chosen)
    valid = chosen >= 0
    eligible[rows[valid], chosen[valid]] = True
    eligible &= target_rows.unsqueeze(1) & torch.isfinite(logits)
    return logits, eligible, batch_idx, patch_idx, (bsz, gh, gw)


def build_patch_visual_concept_activations_from_stage(
    stage_out: Dict[str, Any],
    num_visual_concepts: int,
    *,
    use_logits: bool = True,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Scatter target-step decoder-top-k visual scores onto [B, N, C]."""
    logits, eligible, batch_idx, patch_idx, shape = _visual_target_rows_and_topk_mask(stage_out)
    bsz, gh, gw = shape
    if num_visual_concepts <= 0:
        raise ValueError("num_visual_concepts must be positive.")
    raw_values = logits if use_logits else stage_out["visual_activations"].float().to(logits.device)
    if raw_values.shape != logits.shape:
        raise ValueError("visual_activations must have the same row/concept layout as visual_concept_logits.")
    count = min(num_visual_concepts, logits.shape[1])
    values = raw_values[:, :count].masked_fill(~eligible[:, :count], float("-inf"))
    flat = torch.full((bsz * gh * gw, num_visual_concepts), float("-inf"), device=logits.device, dtype=values.dtype)
    index = (batch_idx * gh * gw + patch_idx).unsqueeze(1).expand(-1, count)
    flat[:, :count].scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    return flat.view(bsz, gh * gw, num_visual_concepts), (gh, gw)


def compute_visual_peak_time_indices(
    stage_out: Dict[str, Any],
    num_visual_concepts: int,
    num_rgb_frames: Optional[int] = None,
) -> torch.Tensor:
    """Return target RGB frame indices for finite target-step top-k matches.

    This is a representative target frame, not a claim that a video feature's
    temporal receptive field consists of that frame alone. Earlier feature
    time steps never participate in ranking.
    """
    scores, _ = build_patch_visual_concept_activations_from_stage(stage_out, num_visual_concepts)
    metadata = stage_out["visual_metadata"]
    frames = int(num_rgb_frames) if num_rgb_frames is not None else int(metadata.get("window_T", metadata["feature_shape"]["T"]))
    if frames <= 0:
        raise ValueError("The RGB window must contain at least one frame.")
    result = torch.full(scores.shape, -1, dtype=torch.long, device=scores.device)
    result[torch.isfinite(scores)] = frames - 1
    return result


def resolve_visual_peak_time_indices(
    model_out: dict,
    activation_key: Optional[str],
    num_concepts: int,
    concept_source: str,
    num_rgb_frames: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Resolve per-patch peak activation RGB frames for visual concept retrieval."""

    if concept_source != "visual":
        return None

    concept_out = model_out.get("concept_out", {})
    if not isinstance(concept_out, dict) or not concept_out:
        return None

    if activation_key is not None and "." in activation_key:
        stage = activation_key.split(".", 1)[0]
    else:
        stage = _find_stage_with_visual_concepts(concept_out)

    stage_out = concept_out.get(stage)
    if not isinstance(stage_out, dict):
        return None

    return compute_visual_peak_time_indices(
        stage_out, num_concepts, num_rgb_frames=num_rgb_frames
    ).cpu()


def resolve_activation_tensor(
    model_out: dict,
    activation_key: Optional[str],
    num_transition_concepts: int,
    num_persistence_concepts: int,
    num_concepts: int,
    *,
    concept_source: str = "visual",
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
    visual_keys = {
        "visual_concept_logits",
        "visual_activations",
        "visual_concepts",
        "patch_visual_concept_activations",
    }

    if concept_source == "visual":
        if activation_key is None:
            stage = _find_stage_with_visual_concepts(concept_out)
            activations, grid_hw = build_patch_visual_concept_activations_from_stage(
                concept_out[stage],
                num_concepts,
                use_logits=True,
            )
            selected_name = f"{stage}.visual_concept_logits"
            print(f"Auto-built patch visual concept activations from {selected_name}")
            return selected_name, activations, grid_hw

        if "." not in activation_key:
            raise ValueError(
                "activation_key must look like 'stage4.visual_concept_logits', "
                f"got {activation_key!r}"
            )

        stage, field = activation_key.split(".", 1)
        if field == "visual_concept_representation":
            raise ValueError(
                "visual_concept_representation is a concept_dim feature vector, not a "
                "per-concept activation tensor. Use stage4.visual_concept_logits with "
                "target-step decoder top-k assignment masking."
            )

        if stage not in concept_out:
            raise KeyError(
                f"Stage {stage!r} not found in concept_out. "
                f"Available stages: {sorted(concept_out.keys())}"
            )

        stage_out = concept_out[stage]
        if not isinstance(stage_out, dict):
            raise KeyError(f"concept_out[{stage!r}] is not a dict.")

        if field in visual_keys:
            use_logits = field != "visual_activations"
            activations, grid_hw = build_patch_visual_concept_activations_from_stage(
                stage_out,
                num_concepts,
                use_logits=use_logits,
            )
            return activation_key, activations, grid_hw

        raise ValueError(
            "concept_source='visual' requires a visual activation key such as "
            f"{sorted(visual_keys)}, got {activation_key!r}"
        )

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
    activations, grid_hw = normalize_activation_tensor(
        raw,
        num_transition_concepts + num_persistence_concepts,
    )
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


def resolve_backbone_spatial_hw(model: ExplainableVidSalModel) -> Tuple[int, int]:
    """Return (H, W) spatial size applied inside the Video Swin backbone."""
    resize_to = getattr(getattr(model, "backbone", None), "resize_to", None)
    if resize_to is None:
        return (224, 384)
    if isinstance(resize_to, int):
        return (int(resize_to), int(resize_to))
    return (int(resize_to[0]), int(resize_to[1]))


def preprocess_rgb_video_like_backbone(
    video_btchw: torch.Tensor,
    resize_to: Tuple[int, int],
) -> torch.Tensor:
    """
    Apply the same spatial resize as ``VideoSwinBackbone._prepare_input`` (no ImageNet norm).

    Input/output layout: [B, T, C, H, W], float in [0, 1], matching dataloader/collate output
    before the model forward pass.
    """
    if video_btchw.dim() != 5:
        raise ValueError(
            f"Expected video_btchw shape [B, T, C, H, W], got {tuple(video_btchw.shape)}"
        )
    target_h, target_w = int(resize_to[0]), int(resize_to[1])
    if tuple(video_btchw.shape[-2:]) == (target_h, target_w):
        return video_btchw
    x = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
    x = F.interpolate(
        x,
        size=(int(video_btchw.shape[1]), target_h, target_w),
        mode="trilinear",
        align_corners=False,
    )
    return x.permute(0, 2, 1, 3, 4).contiguous()


def gallery_video_btchw_from_window(
    rgb_window: torch.Tensor,
    gallery_input_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Convert a dataloader RGB window to backbone-resized [1, T, C, H, W] for gallery crops."""
    native = rgb_video_to_btchw(rgb_window.unsqueeze(0))
    native_hw = (int(native.shape[-2]), int(native.shape[-1]))
    gallery = preprocess_rgb_video_like_backbone(native, gallery_input_hw)
    return gallery, native_hw


def map_patch_box_xyxy_to_native(
    box_xyxy: Tuple[int, int, int, int],
    gallery_hw: Tuple[int, int],
    native_hw: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Map a patch box from gallery (backbone-resized) pixels to native dataloader pixels."""
    gh, gw = gallery_hw
    nh, nw = native_hw
    if gh <= 0 or gw <= 0:
        raise ValueError(f"Invalid gallery_hw {gallery_hw}")
    x0, y0, x1, y1 = box_xyxy
    scale_y = nh / float(gh)
    scale_x = nw / float(gw)
    return (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )


def prepare_saliency_maps_for_concept_grid(
    saliency_maps: torch.Tensor,
    gallery_input_hw: Tuple[int, int],
) -> torch.Tensor:
    """Resize last-frame saliency to backbone spatial size before patch pooling."""
    target = get_target_saliency_frame(saliency_maps)
    if tuple(target.shape[-2:]) == tuple(gallery_input_hw):
        return target.unsqueeze(1)
    resized = F.interpolate(
        target.unsqueeze(1),
        size=gallery_input_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized


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
    # Patch bounds use exclusive right/bottom endpoints. Draw the outline
    # inward so patches touching a frame edge still show all four box sides.
    x0 = max(0, min(x0, boxed.width - 1))
    y0 = max(0, min(y0, boxed.height - 1))
    x1 = max(x0, min(x1 - 1, boxed.width - 1))
    y1 = max(y0, min(y1 - 1, boxed.height - 1))
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
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








def crop_activated_patch_image(
    rgb_batch: torch.Tensor,
    b_idx: int,
    patch_idx: int,
    grid_hw: Optional[Tuple[int, int]],
    frame_idx: int,
    pad_ratio: float = 0.10,
    display_size: Tuple[int, int] = (128, 128),
) -> Image.Image:
    """Crop one patch from RGB already in gallery/backbone spatial resolution."""

    video = rgb_video_to_btchw(rgb_batch)
    _, t, _, h, w = video.shape
    frame_idx = max(0, min(int(frame_idx), t - 1))

    if grid_hw is None:
        grid_hw = infer_grid_from_n(max(patch_idx + 1, 1)) or (1, 1)

    crop_x0, crop_y0, crop_x1, crop_y1 = patch_bounds_from_index(
        patch_idx,
        grid_hw,
        (h, w),
        pad_ratio=pad_ratio,
    )
    full_pil = tensor_chw_to_pil(video[b_idx, frame_idx])
    crop = full_pil.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    return resize_with_aspect(crop, display_size)


def _heap_item_activated_sample_image(item: HeapItem) -> Optional[Image.Image]:
    """Return the target-frame patch crop, without writing an individual file."""
    populate_heap_item_images(item)
    return item.activated_sample_image





def _heap_item_contact_sheet_image(item: HeapItem) -> Optional[np.ndarray]:
    """Build the contact-sheet cell from the top-activated patch crop only."""
    activated_sample_img = _heap_item_activated_sample_image(item)
    if activated_sample_img is None:
        return None
    return np.asarray(activated_sample_img)




















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
    prediction_out = model_out.get("prediction_out")
    patch_logits = None
    if isinstance(prediction_out, dict):
        patch_logits = prediction_out.get("patch_saliency_logits")
    if patch_logits is None:
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

        return torch.sigmoid(scores).flatten(1)

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
        # Saliency is metadata only here; both salient and non-salient scores
        # remain eligible. The mask is a top-percent diagnostic annotation.
        k_keep = max(1, int(math.ceil(n_patches * min(max(float(saliency_top_percent), 1e-6), 1.0))))
        for b in range(bsz):
            top_idx = torch.topk(patch_saliency[b], k=k_keep, largest=True).indices
            salient_mask[b, top_idx] = True
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
        retrieval_scores = torch.where(
            torch.isfinite(concept_flat),
            concept_flat * sal_flat.unsqueeze(1),
            torch.full_like(concept_flat, float("-inf")),
        )
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
    activated_sample_image: Any = field(compare=False, default=None)
    full_activated_frame_image: Any = field(compare=False, default=None)
    activated_patch_box: Optional[Tuple[int, int, int, int]] = field(compare=False, default=None)
    activated_patch_box_native: Optional[Tuple[int, int, int, int]] = field(
        compare=False, default=None
    )
    peak_time_idx: Optional[int] = field(compare=False, default=None)
    rgb_window: Any = field(compare=False, default=None)
    gallery_input_hw: Optional[Tuple[int, int]] = field(compare=False, default=None)
    native_input_hw: Optional[Tuple[int, int]] = field(compare=False, default=None)
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
    context_panel_path: str = ""  # Legacy field; no panel file is generated.
    contact_sheet_path: str = ""
    gallery_path: str = ""
    contact_sheet_cell_box: Optional[Tuple[int, int, int, int]] = None
    gallery_cell_box: Optional[Tuple[int, int, int, int]] = None
    activated_patch_box: Optional[Tuple[int, int, int, int]] = None
    activated_patch_box_native: Optional[Tuple[int, int, int, int]] = None
    gallery_input_hw: Optional[Tuple[int, int]] = None
    native_input_hw: Optional[Tuple[int, int]] = None
    peak_time_idx: Optional[int] = None
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
    """Prepare patch crop and full frame in backbone-resized RGB space (matches model input)."""
    if item.activated_sample_image is not None and item.full_activated_frame_image is not None:
        return
    if item.rgb_window is None:
        raise ValueError(f"Missing RGB window for retrieved example {item.video_name!r}.")
    if item.gallery_input_hw is None:
        raise ValueError(
            f"Missing gallery_input_hw for retrieved example {item.video_name!r}. "
            "Re-run retrieval with a loaded ExplainableVidSalModel so backbone resize is recorded."
        )
    video, native_hw = gallery_video_btchw_from_window(item.rgb_window, item.gallery_input_hw)
    item.native_input_hw = native_hw
    _, frames, _, height, width = video.shape
    target = frames - 1 if item.peak_time_idx is None else int(item.peak_time_idx)
    if target < 0 or target >= frames:
        raise ValueError(f"Target frame {target} is outside the RGB window for {item.video_name}.")
    if item.grid_hw is None:
        raise ValueError("The selected prototype's spatial grid is required to draw its patch box.")
    gh, gw = item.grid_hw
    if item.patch_index < 0 or item.patch_index >= gh * gw:
        raise ValueError("The selected patch index is outside the prototype grid.")
    full_frame = tensor_chw_to_pil(video[0, target])
    box = patch_bounds_from_index(item.patch_index, item.grid_hw, (height, width), pad_ratio=0.0)
    item.activated_sample_image = crop_activated_patch_image(
        video, 0, item.patch_index, item.grid_hw, target,
        pad_ratio=0.10, display_size=CONTACT_SHEET_CELL_SIZE,
    )
    item.activated_patch_box = box
    item.activated_patch_box_native = map_patch_box_xyxy_to_native(
        box, (height, width), native_hw
    )
    # Draw on the exact same RGB frame used to crop the patch contact sheet.
    # Every pixel of the full frame is preserved; resizing maintains aspect ratio.
    line_width = max(3, int(round(min(height, width) * 0.012)))
    item.full_activated_frame_image = draw_patch_rectangle(full_frame, box, color=(255, 0, 0), width=line_width)


def update_heaps_from_activations(
    heaps: Dict[int, List[HeapItem]],
    activations: torch.Tensor,
    rgb_batch: torch.Tensor,
    video_filenames: Sequence[Any],
    grid_hw: Optional[Tuple[int, int]],
    top_k: int,
    serial_start: int,
    concept_indices: Optional[Sequence[int]] = None,
    saliency_filter_mode: str = "none",
    saliency_threshold: float = 0.0,
    saliency_top_percent: float = 0.20,
    saliency_maps: Optional[torch.Tensor] = None,
    patch_saliency: Optional[torch.Tensor] = None,
    peak_time_indices: Optional[torch.Tensor] = None,
    gallery_input_hw: Optional[Tuple[int, int]] = None,
) -> int:
    """Keep exact top-K distinct-video winners per concept across all batches.

    Reduce all windows and patches of each video in this batch before heap
    insertion. No candidate cap is used. If a retained video appears again,
    replace its entry only when the new patch/window has a higher score.
    A non-retained video may re-enter on a later, stronger match. This bounded
    heap is equivalent to ranking each video's best match after the full pass.

    video_filenames MUST identify videos, not individual frames or windows.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    acts = activations.detach().float().cpu()
    bsz, n_patches, num_concepts = acts.shape
    if len(video_filenames) != bsz or rgb_batch.shape[0] != bsz:
        raise ValueError("Video names, RGB windows and activations must have matching batch sizes.")
    if patch_saliency is not None:
        patch_sal = patch_saliency.detach().float().cpu()
    elif saliency_maps is not None:
        if gallery_input_hw is not None:
            sal_for_grid = prepare_saliency_maps_for_concept_grid(
                saliency_maps, gallery_input_hw
            )
            patch_sal = compute_patch_saliency_grid(sal_for_grid, grid_hw)
        else:
            patch_sal = compute_patch_saliency_grid(saliency_maps, grid_hw)
    elif saliency_filter_mode == "none":
        patch_sal = torch.zeros((bsz, n_patches))
    else:
        raise ValueError("Salient-only retrieval requires patch_saliency or saliency_maps.")
    if tuple(patch_sal.shape) != (bsz, n_patches):
        raise ValueError("Patch saliency must have shape [B, N] on the selected concept grid.")
    if peak_time_indices is not None and tuple(peak_time_indices.shape) != tuple(acts.shape):
        raise ValueError("Target RGB frame indices must have the same shape as activations.")
    retrieval_scores, salient_mask_flat = build_saliency_retrieval_scores(
        acts, patch_sal, saliency_filter_mode, saliency_threshold, saliency_top_percent,
    )
    scores = retrieval_scores.view(bsz, n_patches, num_concepts)
    indices = list(concept_indices) if concept_indices is not None else list(range(num_concepts))
    if any(c < 0 or c >= num_concepts for c in indices):
        raise ValueError("Requested concept index is outside the activation bank.")
    if not indices:
        return serial_start
    groups: Dict[str, List[int]] = {}
    for b_idx, value in enumerate(video_filenames):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("Each video_filenames entry must be a nonempty video ID/path, not window metadata.")
        name = os.path.normpath(str(value))
        groups.setdefault(name, []).append(b_idx)
    serial = serial_start
    window_cache: Dict[int, torch.Tensor] = {}
    for video_name, batch_members in groups.items():
        # Search ALL eligible patches from ALL occurrences of this video.
        video_scores = scores[batch_members][:, :, indices].reshape(-1, len(indices))
        values, locations = video_scores.max(dim=0)
        for column, c_idx in enumerate(indices):
            value = float(values[column])
            if not math.isfinite(value):
                continue
            heap = heaps.setdefault(c_idx, [])
            existing = next((i for i, old in enumerate(heap) if old.video_name == video_name), None)
            if existing is not None:
                if value <= heap[existing].score:
                    continue
            elif len(heap) >= top_k and value <= heap[0].score:
                continue
            local_index = int(locations[column])
            b_idx = batch_members[local_index // n_patches]
            p_idx = local_index % n_patches
            # Share one immutable CPU window among accepted concepts; rejected
            # candidates never trigger an expensive RGB clone.
            if b_idx not in window_cache:
                window_cache[b_idx] = rgb_batch[b_idx].detach().cpu().clone()
            frame_idx = int(peak_time_indices[b_idx, p_idx, c_idx]) if peak_time_indices is not None else int(rgb_video_to_btchw(rgb_batch[b_idx:b_idx+1]).shape[1]) - 1
            if frame_idx < 0:
                raise ValueError("A finite target-step match has no target RGB frame index.")
            flat_idx = b_idx * n_patches + p_idx
            item = HeapItem(
                score=value, serial=serial, concept_idx=c_idx,
                batch_index=b_idx, patch_index=p_idx, video_name=video_name,
                grid_hw=grid_hw, rgb_window=window_cache[b_idx],
                gallery_input_hw=gallery_input_hw,
                peak_time_idx=frame_idx, activation_score=float(acts[b_idx, p_idx, c_idx]),
                patch_pred_saliency=float(patch_sal[b_idx, p_idx]),
                retrieval_score=value, is_salient_region=bool(salient_mask_flat[flat_idx]),
            )
            serial += 1
            if existing is not None:
                heap[existing] = item
                heapq.heapify(heap)
            elif len(heap) < top_k:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)
    return serial


def should_skip_existing_artifacts(skip_existing: bool, overwrite: bool) -> bool:
    """Return True when existing per-concept artifacts should be reused."""
    return skip_existing and not overwrite


def _should_write(path: Path, overwrite: bool) -> bool:
    return overwrite or not path.exists()





def _relative_to_concept_dir(concept_dir: Path, path: Path) -> str:
    return str(path.relative_to(concept_dir))


def contact_sheet_grid_shape(num_examples: int) -> Tuple[int, int]:
    """Return (rows, cols) for a contact sheet grid (three samples per row)."""

    if num_examples <= 0:
        return 1, 1
    cols = CONTACT_SHEET_COLS
    rows = max(1, math.ceil(num_examples / cols))
    return rows, cols


def contact_sheet_cell_box(rank: int) -> Tuple[int, int, int, int]:
    """Pixel bounds of a rank's cell in the deterministic patch contact sheet."""
    cell_w, cell_h = CONTACT_SHEET_CELL_SIZE
    row, col = divmod(rank, CONTACT_SHEET_COLS)
    x = GALLERY_SPACING + col * (cell_w + GALLERY_SPACING)
    y = GALLERY_SPACING + row * (cell_h + GALLERY_SPACING)
    return x, y, x + cell_w, y + cell_h


def gallery_cell_box(rank: int) -> Tuple[int, int, int, int]:
    """Pixel bounds of a rank's full-frame cell in the single-row gallery."""
    cell_w, cell_h = GALLERY_CELL_SIZE
    x = GALLERY_SPACING + rank * (cell_w + GALLERY_SPACING)
    return x, GALLERY_SPACING, x + cell_w, GALLERY_SPACING + cell_h


def _render_contact_sheet_grid(
    cells: Sequence[Optional[np.ndarray]],
    path: Path,
) -> bool:
    """Render a text-free JPEG patch contact sheet with reproducible cell bounds."""
    if not cells:
        return False
    rows, cols = contact_sheet_grid_shape(len(cells))
    cell_w, cell_h = CONTACT_SHEET_CELL_SIZE
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * GALLERY_SPACING,
                               rows * cell_h + (rows + 1) * GALLERY_SPACING), "white")
    for rank, cell in enumerate(cells):
        if cell is None:
            raise ValueError(f"Cannot render contact sheet: missing crop at rank {rank}.")
        crop = Image.fromarray(np.asarray(cell, dtype=np.uint8)).convert("RGB")
        _paste_centered(canvas, resize_with_aspect(crop, CONTACT_SHEET_CELL_SIZE), contact_sheet_cell_box(rank))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    return True


def save_frame_gallery(ranked: Sequence[HeapItem], path: Path, overwrite: bool) -> bool:
    """Save full target frames side by side, with each selected patch boxed red."""
    if not ranked or not _should_write(path, overwrite):
        return False
    cell_w, cell_h = GALLERY_CELL_SIZE
    canvas = Image.new("RGB", (len(ranked) * cell_w + (len(ranked) + 1) * GALLERY_SPACING,
                               cell_h + 2 * GALLERY_SPACING), "white")
    for rank, item in enumerate(ranked):
        populate_heap_item_images(item)
        frame = resize_with_aspect(item.full_activated_frame_image, GALLERY_CELL_SIZE)
        _paste_centered(canvas, frame, gallery_cell_box(rank))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    return True


def save_contact_sheet(
    ranked: Sequence[HeapItem],
    path: Path,
    overwrite: bool,
) -> bool:
    """Save a text-free 3-column contact sheet of top-activated patch crops."""

    if not ranked or not _should_write(path, overwrite):
        return False

    cells: List[Optional[np.ndarray]] = []
    for item in ranked:
        cells.append(_heap_item_contact_sheet_image(item))
    return _render_contact_sheet_grid(cells, path)


def _prune_top_examples_dir(top_examples_dir: Path) -> None:
    """Clear aggregate images and obsolete per-example image assets."""
    for path in top_examples_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in {"contact_sheet.png", "contact_sheet.jpg", "gallery.jpg", "examples_metadata.json", "retrieval_settings.json"} or (
            (path.name.startswith("example_") or path.name.startswith("crop_pair"))
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ):
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
        contact_sheet_path = str(entry.get("contact_sheet_path", "top_examples/contact_sheet.jpg"))
        gallery_path = str(entry.get("gallery_path", "top_examples/gallery.jpg"))
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
        peak_time_raw = entry.get("peak_time_idx")
        peak_time_idx = (
            int(peak_time_raw)
            if peak_time_raw is not None and int(peak_time_raw) >= 0
            else None
        )
        examples.append(
            SavedExample(
                concept_idx=concept_idx,
                rank=int(entry["rank"]),
                score=retrieval_score,
                video_name=str(entry.get("video_filename", "")),
                batch_index=int(entry.get("batch_index", -1)),
                patch_index=int(entry.get("sample_index", entry.get("patch_index", -1))),
                grid_hw=tuple(entry["grid_hw"]) if entry.get("grid_hw") is not None else None,
                image_path=contact_sheet_path,
                frame_t_path="",
                frame_t1_path="",
                pair_path="",
                sequence_path="",
                frame_paths=[],
                num_frames=int(entry.get("num_frames", NUM_EXAMPLE_FRAMES)),
                full_t_boxed_path="",
                full_t1_boxed_path="",
                contact_sheet_path=contact_sheet_path,
                gallery_path=gallery_path,
                contact_sheet_cell_box=tuple(entry["contact_sheet_cell_box"]) if entry.get("contact_sheet_cell_box") is not None else None,
                gallery_cell_box=tuple(entry["gallery_cell_box"]) if entry.get("gallery_cell_box") is not None else None,
                activated_patch_box=tuple(entry["activated_patch_box"]) if entry.get("activated_patch_box") is not None else None,
                activated_patch_box_native=(
                    tuple(entry["activated_patch_box_native"])
                    if entry.get("activated_patch_box_native") is not None
                    else None
                ),
                gallery_input_hw=(
                    tuple(entry["gallery_input_hw"])
                    if entry.get("gallery_input_hw") is not None
                    else None
                ),
                native_input_hw=(
                    tuple(entry["native_input_hw"])
                    if entry.get("native_input_hw") is not None
                    else None
                ),
                peak_time_idx=peak_time_idx,
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
    *,
    concept_source: str = "visual",
    num_concepts: Optional[int] = None,
) -> List[SavedExample]:
    """Write only contact_sheet.jpg, gallery.jpg and metadata for this prototype."""
    concept_type, local_idx = get_concept_type_and_local_index(
        c_idx, num_transition_concepts, num_persistence_concepts,
        concept_source=concept_source, num_concepts=num_concepts,
    )
    concept_dir = ensure_dir(get_concept_dir(concepts_root, concept_type, local_idx))
    top_examples_dir = ensure_dir(concept_dir / "top_examples")
    metadata_path = top_examples_dir / "examples_metadata.json"
    ranked = sorted(heap, key=lambda item: item.score, reverse=True)
    if overwrite:
        _prune_top_examples_dir(top_examples_dir)
    contact_rel = "top_examples/contact_sheet.jpg"
    gallery_rel = "top_examples/gallery.jpg"
    saved_examples: List[SavedExample] = []
    metadata_entries: List[dict] = []
    for rank, item in enumerate(ranked):
        populate_heap_item_images(item)
        crop_box = contact_sheet_cell_box(rank)
        frame_box = gallery_cell_box(rank)
        saved_examples.append(SavedExample(
            concept_idx=c_idx, rank=rank, score=item.retrieval_score,
            video_name=item.video_name, batch_index=item.batch_index,
            patch_index=item.patch_index, grid_hw=item.grid_hw,
            image_path=contact_rel, frame_t_path="", frame_t1_path="", pair_path="",
            num_frames=1, contact_sheet_path=contact_rel, gallery_path=gallery_rel,
            contact_sheet_cell_box=crop_box, gallery_cell_box=frame_box,
            activated_patch_box=item.activated_patch_box,
            activated_patch_box_native=item.activated_patch_box_native,
            gallery_input_hw=item.gallery_input_hw,
            native_input_hw=item.native_input_hw,
            peak_time_idx=item.peak_time_idx, activation_score=item.activation_score,
            patch_pred_saliency=item.patch_pred_saliency, retrieval_score=item.retrieval_score,
            is_salient_region=item.is_salient_region,
        ))
        metadata_entries.append({
            "rank": rank, "score": float(item.score),
            "activation_score": float(item.activation_score),
            "patch_pred_saliency": float(item.patch_pred_saliency),
            "retrieval_score": float(item.retrieval_score),
            "is_salient_region": bool(item.is_salient_region),
            "batch_index": int(item.batch_index), "sample_index": int(item.patch_index),
            "global_concept_index": int(c_idx), "concept_type": concept_type,
            "concept_number": int(local_idx), "video_filename": item.video_name,
            "grid_hw": list(item.grid_hw) if item.grid_hw is not None else None,
            "num_frames": 1, "peak_time_idx": item.peak_time_idx,
            "contact_sheet_path": contact_rel, "gallery_path": gallery_rel,
            "contact_sheet_cell_box": list(crop_box), "gallery_cell_box": list(frame_box),
            "activated_patch_box": list(item.activated_patch_box),
            "activated_patch_box_native": (
                list(item.activated_patch_box_native)
                if item.activated_patch_box_native is not None
                else None
            ),
            "gallery_input_hw": (
                list(item.gallery_input_hw) if item.gallery_input_hw is not None else None
            ),
            "native_input_hw": (
                list(item.native_input_hw) if item.native_input_hw is not None else None
            ),
        })
    if ranked:
        save_contact_sheet(ranked, top_examples_dir / "contact_sheet.jpg", overwrite=overwrite)
        save_frame_gallery(ranked, top_examples_dir / "gallery.jpg", overwrite=overwrite)
    metadata_path.write_text(json.dumps(metadata_entries, indent=2), encoding="utf-8")
    return saved_examples


def partition_concepts_for_retrieval(
    concept_indices: Sequence[int],
    concepts_root: Path,
    num_transition_concepts: int,
    num_persistence_concepts: int,
    skip_existing: bool,
    overwrite: bool,
    *,
    concept_source: str = "visual",
    num_concepts: Optional[int] = None,
    retrieval_settings: Optional[dict] = None,
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
            concept_source=concept_source,
            num_concepts=num_concepts,
        )
        concept_dir = get_concept_dir(concepts_root, concept_type, local_idx)
        metadata_path = concept_dir / "top_examples" / "examples_metadata.json"
        settings_path = concept_dir / "top_examples" / "retrieval_settings.json"
        settings_match = False
        if retrieval_settings is not None and settings_path.is_file():
            try:
                settings_match = json.loads(settings_path.read_text(encoding="utf-8")) == retrieval_settings
            except (ValueError, OSError):
                pass
        if skip_artifacts and settings_match and metadata_path.exists():
            saved[c_idx] = load_saved_examples_from_metadata(
                concept_dir,
                metadata_path,
                c_idx,
            )
            names = [example.video_name for example in saved[c_idx]]
            images_ready = not names or all(
                (concept_dir / rel).is_file()
                for rel in ("top_examples/contact_sheet.jpg", "top_examples/gallery.jpg")
            )
            if len(names) == len(set(names)) and images_ready:
                continue
            del saved[c_idx]
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
    *,
    concept_source: str = "visual",
    num_concepts: Optional[int] = None,
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
            concept_source=concept_source,
            num_concepts=num_concepts,
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
            concept_source=concept_source,
            num_concepts=num_concepts,
        )

    if output_dir is not None:
        write_retrieved_examples_aggregate(saved, output_dir)

    return saved


# -----------------------------
# LLM prompting / parsing
# -----------------------------



def build_single_example_description_prompt(concept_id: str, example: SavedExample) -> str:
    # concept_id and example are intentionally omitted from the prompt text.
    _ = (concept_id, example)
    return """You are viewing one top-activated example for a learned concept in a video saliency model.

The image shows a cropped patch from the same activated region.
Your task is to describe the patch in detail in less than 50 tokens.

Rules:
- Focus only on what is visible in the cropped patch.
- Keep every field to one short phrase or sentence. Be compact.
- Report low_level_features (color, edges, texture, shape, blur, etc.) and semantic_features
  (object/part/category cues such as face, hand, text, vehicle) separately. Use "none" for
  semantic_features when no clear semantic category is visible.
- For edge_shape, report edge geometry when visible (rounded, curved, horizontal, vertical,
  diagonal, etc.); otherwise use "none".
- Return valid JSON only. Do not add text before or after the JSON object.

Return JSON only:
{
  "detailed_description": "...",
  "low_level_features": "...",
  "edge_shape": "...",
  "semantic_features": "...",
  "possible_reason_for_saliency": "...",
  "confidence": "high | medium | low"
}
""".strip()


def build_aggregate_concept_prompt(
    concept_id: str,
    sample_descriptions: Sequence[dict],
) -> str:
    descriptions_json = json.dumps(list(sample_descriptions), indent=2)
    return f"""You are given individual descriptions of top-activated examples for one learned concept.

Identify the common pattern across the majority using each sample's low_level_features, edge_shape,
semantic_features, and possible_reason_for_saliency.
- Find the pattern across the majority of the samples, doesn't necessarily need to be all of them.
- If the shared pattern is semantic categories, name it as semantic.
- If the shared pattern is low-level, name it as low-level.
- If no consistent pattern exists, say unclear.
- Mention exceptions.
- Candidate name should be specific.
- Explanation should not be detailed and accurate. 

Per-sample descriptions:
{descriptions_json}

Return JSON only:
{{
  "concept_id": "{concept_id}",
  "candidate_name": "...",
  "pattern_type": "low-level | semantic | mixed | unclear",
  "common_pattern": "...",
  "why_it_may_contribute_to_saliency": "...",
  "exceptions": "...",
  "confidence": "high | medium | low"
}}
""".strip()


def _json_object_opens_unclosed_string(fragment: str) -> bool:
    in_string = False
    escape = False
    for ch in fragment:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
    return in_string


def _repair_truncated_json_object(cleaned: str) -> Optional[dict]:
    """Best-effort repair when generation stops before the JSON object closes."""
    start = cleaned.find("{")
    if start == -1:
        return None

    fragment = cleaned[start:].rstrip().rstrip(",")
    if _json_object_opens_unclosed_string(fragment):
        fragment += '"'

    open_braces = fragment.count("{") - fragment.count("}")
    if open_braces > 0:
        fragment += "}" * open_braces

    try:
        loaded = json.loads(fragment)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        return None


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

    repaired = _repair_truncated_json_object(cleaned)
    if repaired is not None:
        return repaired

    raise ValueError(f"Unbalanced JSON object in output: {text[:500]}")


def generate_and_parse_llm_json(
    llm: LLMHandle,
    prompt: str,
    image_paths: Sequence[Union[str, Image.Image]],
    max_new_tokens: int,
    *,
    retry_max_new_tokens: Optional[int] = None,
) -> Tuple[Optional[dict], str, bool, Optional[str]]:
    """Generate LLM output and parse JSON, retrying once with more tokens if needed."""
    raw_text = generate_qwen3_response(
        llm,
        prompt,
        image_paths,
        max_new_tokens=max_new_tokens,
    )
    try:
        return parse_json_from_text(raw_text), raw_text, True, None
    except Exception as first_exc:
        retry_tokens = retry_max_new_tokens or max(
            max_new_tokens * 2,
            STAGE1_SAMPLE_RETRY_MAX_NEW_TOKENS,
        )
        if retry_tokens <= max_new_tokens:
            return None, raw_text, False, str(first_exc)

        retry_text = generate_qwen3_response(
            llm,
            prompt,
            image_paths,
            max_new_tokens=retry_tokens,
        )
        try:
            return parse_json_from_text(retry_text), retry_text, True, None
        except Exception as second_exc:
            return None, retry_text, False, str(second_exc)


MULTI_STAGE_EXPLANATION_RECORD_KEYS = (
    "concept_id",
    "global_concept_index",
    "concept_type",
    "concept_number",
    "candidate_name",
    "pattern_type",
    "explanation",
    "common_pattern",
    "why_it_may_contribute_to_saliency",
    "exceptions",
    "confidence",
    "sample_descriptions",
    "sample_description_paths",
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


def build_sample_description_record(
    *,
    concept_id: str,
    global_concept_index: int,
    concept_type: str,
    concept_number: int,
    example: SavedExample,
    parsed_llm: Optional[dict],
    raw_llm_output: str,
    parse_success: bool,
    parse_error: Optional[str] = None,
) -> dict:
    """Build one per-sample description JSON record."""

    record = {
        "concept_id": concept_id,
        "global_concept_index": global_concept_index,
        "concept_type": concept_type,
        "concept_number": concept_number,
        "rank": int(example.rank),
        "video_name": example.video_name,
        "patch_index": int(example.patch_index),
        "activation_score": float(example.activation_score),
        "patch_pred_saliency": float(example.patch_pred_saliency),
        "retrieval_score": float(example.retrieval_score),
        "is_salient_region": bool(example.is_salient_region),
        "contact_sheet_path": example.contact_sheet_path,
        "contact_sheet_cell_box": example.contact_sheet_cell_box,
        "gallery_path": example.gallery_path,
        "gallery_cell_box": example.gallery_cell_box,
        "raw_llm_output": raw_llm_output,
        "parse_success": parse_success,
    }

    if parse_success and parsed_llm is not None:
        record.update(
            {
                "detailed_description": str(parsed_llm.get("detailed_description", "unclear")),
                "low_level_features": str(parsed_llm.get("low_level_features", "unclear")),
                "edge_shape": str(parsed_llm.get("edge_shape", "unclear")),
                "semantic_features": str(parsed_llm.get("semantic_features", "unclear")),
                "possible_reason_for_saliency": str(
                    parsed_llm.get("possible_reason_for_saliency", "unclear")
                ),
                "confidence": str(parsed_llm.get("confidence", "low")),
            }
        )
    else:
        record.update(
            {
                "detailed_description": "unclear",
                "low_level_features": "unclear",
                "edge_shape": "unclear",
                "semantic_features": "unclear",
                "possible_reason_for_saliency": "unclear",
                "confidence": "low",
            }
        )
        if parse_error:
            record["parse_error"] = parse_error

    return record


def save_sample_description(concept_dir: Path, sample_record: dict, rank: int) -> str:
    """Save one per-sample description JSON and return its path relative to concept_dir."""
    sample_dir = ensure_dir(concept_dir / "sample_descriptions")
    sample_path = sample_dir / f"sample_{rank:03d}_description.json"
    sample_path.write_text(json.dumps(sample_record, indent=2), encoding="utf-8")
    return _relative_to_concept_dir(concept_dir, sample_path)


def load_sample_description(concept_dir: Path, rank: int) -> Optional[dict]:
    """Load an existing per-sample description JSON if present."""
    sample_path = concept_dir / "sample_descriptions" / f"sample_{rank:03d}_description.json"
    if not sample_path.exists():
        return None
    try:
        loaded = json.loads(sample_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: failed to load sample description for rank {rank} "
            f"under {concept_dir}: {exc}"
        )
        return None


def _sample_descriptions_for_aggregation(sample_records: Sequence[dict]) -> List[dict]:
    summaries: List[dict] = []
    for record in sample_records:
        summaries.append(
            {
                "rank": record.get("rank"),
                "video_name": record.get("video_name"),
                "detailed_description": record.get("detailed_description", "unclear"),
                "low_level_features": record.get("low_level_features", "unclear"),
                "edge_shape": record.get("edge_shape", "unclear"),
                "semantic_features": record.get("semantic_features", "unclear"),
                "possible_reason_for_saliency": record.get(
                    "possible_reason_for_saliency",
                    "unclear",
                ),
                "confidence": record.get("confidence", "low"),
            }
        )
    return summaries


def _format_aggregate_explanation(parsed_aggregate: dict) -> str:
    """Combine aggregate fields into one readable explanation string."""
    segments: List[str] = []
    field_labels = (
        ("common_pattern", "Common pattern"),
        ("why_it_may_contribute_to_saliency", "Why it may contribute to saliency"),
        ("exceptions", "Exceptions"),
    )
    for field_name, label in field_labels:
        value = str(parsed_aggregate.get(field_name, "")).strip()
        if value and value != "unclear":
            segments.append(f"{label}: {value}")
    return " ".join(segments) if segments else "unclear"


def build_aggregated_explanation_record(
    *,
    concept_id: str,
    global_concept_index: int,
    concept_type: str,
    concept_number: int,
    llm_model: str,
    top_k: int,
    examples: Sequence[SavedExample],
    sample_records: Sequence[dict],
    sample_description_paths: Sequence[str],
    parsed_aggregate: Optional[dict],
    raw_aggregate_output: str,
    aggregate_parse_success: bool,
    aggregate_parse_error: Optional[str] = None,
    saliency_filter_mode: str,
    saliency_threshold: float,
    saliency_top_percent: float,
    min_salient_examples_per_concept: int,
) -> dict:
    """Build the aggregated concept-level explanation.json payload."""

    (
        num_salient_examples,
        mean_patch_pred_saliency,
        mean_activation_score,
        mean_retrieval_score,
        low_saliency_coverage,
    ) = compute_example_statistics(examples, min_salient_examples_per_concept)

    record = {
        "concept_id": concept_id,
        "global_concept_index": global_concept_index,
        "concept_type": concept_type,
        "concept_number": concept_number,
        "llm_model": llm_model,
        "top_k": top_k,
        "sample_descriptions": list(sample_records),
        "sample_description_paths": list(sample_description_paths),
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
        "raw_llm_output": raw_aggregate_output,
        "parse_success": aggregate_parse_success,
    }

    if aggregate_parse_success and parsed_aggregate is not None:
        record.update(
            {
                "candidate_name": str(parsed_aggregate.get("candidate_name", "unclear")),
                "pattern_type": str(parsed_aggregate.get("pattern_type", "unclear")),
                "common_pattern": str(parsed_aggregate.get("common_pattern", "unclear")),
                "why_it_may_contribute_to_saliency": str(
                    parsed_aggregate.get("why_it_may_contribute_to_saliency", "unclear")
                ),
                "exceptions": str(parsed_aggregate.get("exceptions", "unclear")),
                "confidence": str(parsed_aggregate.get("confidence", "low")),
                "explanation": _format_aggregate_explanation(parsed_aggregate),
            }
        )
    else:
        record.update(
            {
                "candidate_name": "unclear",
                "pattern_type": "unclear",
                "common_pattern": "unclear",
                "why_it_may_contribute_to_saliency": "unclear",
                "exceptions": "unclear",
                "explanation": "unclear",
                "confidence": "low",
            }
        )
        if aggregate_parse_error:
            record["parse_error"] = aggregate_parse_error

    return {
        key: record[key]
        for key in MULTI_STAGE_EXPLANATION_RECORD_KEYS
        if key in record
    }





def collect_example_patch_image(
    concept_dir: Path,
    example: SavedExample,
) -> Optional[Image.Image]:
    """Read one saved contact-sheet cell in memory; never create a crop-pair file."""
    rel_path = example.contact_sheet_path or "top_examples/contact_sheet.jpg"
    path = Path(rel_path) if Path(rel_path).is_absolute() else concept_dir / rel_path
    box = example.contact_sheet_cell_box
    if box is None:
        print(f"WARNING: missing contact-sheet cell metadata for rank {example.rank} ({example.video_name})")
        return None
    try:
        with Image.open(path) as sheet:
            if box[0] < 0 or box[1] < 0 or box[2] > sheet.width or box[3] > sheet.height:
                raise ValueError("Contact-sheet cell bounds exceed the image.")
            return sheet.convert("RGB").crop(box)
    except (OSError, ValueError) as exc:
        print(f"WARNING: cannot load patch for rank {example.rank} ({example.video_name}): {exc}")
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


def generate_qwen3_response(
    llm: LLMHandle,
    prompt: str,
    image_paths: Sequence[Union[str, Image.Image]] = (),
    max_new_tokens: int = STAGE1_SAMPLE_MAX_NEW_TOKENS,
) -> str:
    """Generate one structured response using Qwen3-VL (multimodal or text-only)."""

    processor = llm.processor
    model = llm.model

    if image_paths:
        content: List[dict] = []
        for path in image_paths:
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

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
            pil_images = [path.convert("RGB") if isinstance(path, Image.Image) else Image.open(path).convert("RGB") for path in image_paths]
            inputs = processor(
                text=[text],
                images=pil_images,
                padding=True,
                return_tensors="pt",
            )
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text],
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
    sample_max_new_tokens: int = 256,
    aggregate_max_new_tokens: int = 512,
    reuse_sample_descriptions: bool = True,
    *,
    concept_source: str = "visual",
    num_concepts: Optional[int] = None,
) -> List[dict]:
    llm: Optional[LLMHandle] = None
    descriptions: List[dict] = []
    ensure_dir(concepts_root)
    default_count = (
        num_concepts
        if concept_source == "visual"
        else num_transition_concepts + num_persistence_concepts
    )
    indices = (
        list(concept_indices)
        if concept_indices is not None
        else list(range(default_count))
    )

    for c_idx in tqdm(indices, desc="Two-stage LLM concept descriptions"):
        concept_type, local_idx = get_concept_type_and_local_index(
            c_idx,
            num_transition_concepts,
            num_persistence_concepts,
            concept_source=concept_source,
            num_concepts=num_concepts,
        )
        concept_id = format_concept_id(
            c_idx,
            num_transition_concepts,
            num_persistence_concepts,
            concept_source=concept_source,
            num_concepts=num_concepts,
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

        ensure_dir(concept_dir / "sample_descriptions")
        sample_records: List[dict] = []
        sample_description_paths: List[str] = []

        for example in sorted(examples, key=lambda item: item.rank):
            if reuse_sample_descriptions:
                cached_record = load_sample_description(concept_dir, example.rank)
                if cached_record is not None:
                    sample_records.append(cached_record)
                    sample_description_paths.append(
                        f"sample_descriptions/sample_{example.rank:03d}_description.json"
                    )
                    continue

            patch_image = collect_example_patch_image(concept_dir, example)
            if patch_image is None:
                sample_record = build_sample_description_record(
                    concept_id=concept_id,
                    global_concept_index=c_idx,
                    concept_type=concept_type,
                    concept_number=local_idx,
                    example=example,
                    parsed_llm=None,
                    raw_llm_output="",
                    parse_success=False,
                    parse_error="missing contact-sheet patch image",
                )
            else:
                sample_prompt = build_single_example_description_prompt(concept_id, example)
                (
                    sample_parsed_llm,
                    sample_raw_text,
                    sample_parse_success,
                    sample_parse_error,
                ) = generate_and_parse_llm_json(
                    llm,
                    sample_prompt,
                    [patch_image],
                    sample_max_new_tokens,
                )
                if not sample_parse_success:
                    print(
                        f"WARNING: failed to parse Stage 1 JSON for {concept_id} "
                        f"rank {example.rank}: {sample_parse_error}"
                    )

                sample_record = build_sample_description_record(
                    concept_id=concept_id,
                    global_concept_index=c_idx,
                    concept_type=concept_type,
                    concept_number=local_idx,
                    example=example,
                    parsed_llm=sample_parsed_llm,
                    raw_llm_output=sample_raw_text,
                    parse_success=sample_parse_success,
                    parse_error=sample_parse_error,
                )

            sample_description_paths.append(
                save_sample_description(concept_dir, sample_record, example.rank)
            )
            sample_records.append(sample_record)

        aggregate_prompt = build_aggregate_concept_prompt(
            concept_id,
            _sample_descriptions_for_aggregation(sample_records),
        )
        agg_raw_text = generate_qwen3_response(
            llm,
            aggregate_prompt,
            [],
            max_new_tokens=aggregate_max_new_tokens,
        )

        agg_parse_success = False
        agg_parsed_llm: Optional[dict] = None
        agg_parse_error: Optional[str] = None
        try:
            agg_parsed_llm = parse_json_from_text(agg_raw_text)
            agg_parse_success = True
        except Exception as exc:
            agg_parse_error = str(exc)
            print(
                f"WARNING: failed to parse Stage 2 JSON for {concept_id}: {agg_parse_error}"
            )

        explanation = build_aggregated_explanation_record(
            concept_id=concept_id,
            global_concept_index=c_idx,
            concept_type=concept_type,
            concept_number=local_idx,
            llm_model=llm.model_name,
            top_k=top_k,
            examples=examples,
            sample_records=sample_records,
            sample_description_paths=sample_description_paths,
            parsed_aggregate=agg_parsed_llm,
            raw_aggregate_output=agg_raw_text,
            aggregate_parse_success=agg_parse_success,
            aggregate_parse_error=agg_parse_error,
            saliency_filter_mode=saliency_filter_mode,
            saliency_threshold=saliency_threshold,
            saliency_top_percent=saliency_top_percent,
            min_salient_examples_per_concept=min_salient_examples_per_concept,
        )
        descriptions.append(explanation)

        explanation_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")

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
    *,
    concept_source: str = "visual",
) -> Dict[int, List[SavedExample]]:
    """Retrieve top examples; defer disk writes until the dataset pass completes."""

    model.eval()
    gallery_input_hw = resolve_backbone_spatial_hw(model)
    ensure_dir(concepts_root)
    concept_indices = resolve_concept_indices(num_concepts, max_concepts)
    checkpoint_signature = getattr(model, "_gallery_checkpoint_signature", None)
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        subset_indices = list(dataset.indices)
        dataset = dataset.dataset
    else:
        subset_indices = None
    retrieval_settings = {
        "retrieval_version": 4,
        "gallery_format": "aggregate_jpeg_only",
        "gallery_rgb_space": "backbone_spatial_resize",
        "gallery_input_hw": list(gallery_input_hw),
        "unique_videos": True, "target_step_only": True,
        "assignment": "decoder_top_k", "decoder_top_k": MODEL_TOP_K,
        "top_k": top_k, "activation_key": activation_key,
        "concept_source": concept_source, "num_concepts": num_concepts,
        "saliency_filter_mode": saliency_filter_mode, "saliency_source": saliency_source,
        "saliency_threshold": saliency_threshold, "saliency_top_percent": saliency_top_percent,
        "max_batches": max_batches, "use_amp": bool(use_amp),
        "checkpoint": checkpoint_signature,
        "dataset": getattr(model, "_gallery_dataset_signature", None),
        "subset_indices": subset_indices,
    }
    retrieve_indices, saved_examples = partition_concepts_for_retrieval(
        concept_indices=concept_indices,
        concepts_root=concepts_root,
        num_transition_concepts=num_transition_concepts,
        num_persistence_concepts=num_persistence_concepts,
        skip_existing=skip_existing,
        overwrite=overwrite,
        concept_source=concept_source,
        num_concepts=num_concepts,
        retrieval_settings=retrieval_settings,
    )
    # Invalidate cache completion markers before intermediate writes. A partial
    # or interrupted run must never look like a completed matching retrieval.
    for c_idx in retrieve_indices:
        concept_dir = get_concept_dir(concepts_root, *get_concept_type_and_local_index(
            c_idx, num_transition_concepts, num_persistence_concepts,
            concept_source=concept_source, num_concepts=num_concepts,
        ))
        (concept_dir / "top_examples" / "retrieval_settings.json").unlink(missing_ok=True)
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
        amp_note = "on" if use_amp and device.type == "cuda:1" else "off"
        print(
            f"Retrieving top examples for {len(retrieve_indices)} concept(s); "
            f"saving {save_note}; AMP {amp_note}."
        )
    elif concept_indices:
        print("All requested concepts already have saved top_examples; skipping retrieval.")

    amp_enabled = use_amp and device.type == "cuda:1"
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
                concept_source=concept_source,
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
                "gallery_input_hw": gallery_input_hw,
            }
            if saliency_source == "predicted":
                heap_kwargs["patch_saliency"] = extract_predicted_patch_saliency(
                    model_out,
                    grid_hw,
                )
            else:
                heap_kwargs["saliency_maps"] = extract_gt_saliency_batch(sal_batch)

            peak_time_indices = resolve_visual_peak_time_indices(
                model_out,
                activation_key=activation_key,
                num_concepts=num_concepts,
                concept_source=concept_source,
                num_rgb_frames=int(rgb_video_to_btchw(rgb_batch).shape[1]),
            )
            if peak_time_indices is not None:
                heap_kwargs["peak_time_indices"] = peak_time_indices

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
                    concept_source=concept_source,
                    num_concepts=num_concepts,
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
            overwrite=True,
            skip_existing=False,
            concept_source=concept_source,
            num_concepts=num_concepts,
        )
        saved_examples.update(newly_saved)
        for c_idx in retrieve_indices:
            concept_dir = get_concept_dir(concepts_root, *get_concept_type_and_local_index(
                c_idx, num_transition_concepts, num_persistence_concepts,
                concept_source=concept_source, num_concepts=num_concepts,
            ))
            ensure_dir(concept_dir / "top_examples").joinpath("retrieval_settings.json").write_text(
                json.dumps(retrieval_settings, indent=2), encoding="utf-8",
            )

    write_retrieved_examples_aggregate(saved_examples, output_dir)
    return saved_examples


def parse_bool_argument(value: str) -> bool:
    """Accept --salient-patches-only, --salient-patches-only false, or =false."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


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
        "--concept-source",
        choices=("visual", "trajectory"),
        default="visual",
        help=(
            "Which concept bank to retrieve from in concept_out. "
            "'visual' uses patch-level visual_concept_logits with decoder top-k assignment "
            "masking on the target feature step (default). "
            "'trajectory' uses transition/persistence activations."
        ),
    )
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
            "e.g. stage4.visual_concept_logits for visual concepts or "
            "stage4.concept_activations for trajectory concepts. "
            "If omitted, auto-discovers based on --concept-source."
        ),
    )
    parser.add_argument("--dry-run-discover", action="store_true", help="Print concept_out tensors and exit")
    parser.add_argument("--skip-llm", action="store_true", help="Only retrieve top examples; do not run the LLM")
    parser.add_argument("--llm-name", default="qwen3", help="Short LLM name passed to load_llm")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=STAGE1_SAMPLE_MAX_NEW_TOKENS,
        help=(
            "Legacy max-new-tokens fallback for Stage 1 when --sample-max-new-tokens "
            "is not set."
        ),
    )
    parser.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum new tokens for each single-example VLM description. "
            f"Default: --max-new-tokens or {STAGE1_SAMPLE_MAX_NEW_TOKENS}."
        ),
    )
    parser.add_argument(
        "--aggregate-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum new tokens for the text-only concept aggregation step. "
            f"Default: {STAGE2_AGGREGATE_MAX_NEW_TOKENS}."
        ),
    )
    parser.add_argument(
        "--reuse-sample-descriptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing sample_descriptions/sample_XXX_description.json files "
            "when present."
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
        "--salient-patches-only",
        type=parse_bool_argument,
        nargs="?",
        const=True,
        default=False,
        help=("Restrict retrieval to salient patches (default: false). A bare flag means true; "
              "an explicit false disables filtering regardless of --saliency-filter-mode."),
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
        help="Saliency filter used ONLY when --salient-patches-only is true (default: top 20 percent)",
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


def resolve_llm_max_new_tokens(args: argparse.Namespace) -> Tuple[int, int]:
    """Resolve Stage 1/2 token limits, preferring stage-specific CLI args."""
    sample_max_new_tokens = (
        args.sample_max_new_tokens
        if args.sample_max_new_tokens is not None
        else args.max_new_tokens
    )
    aggregate_max_new_tokens = (
        args.aggregate_max_new_tokens
        if args.aggregate_max_new_tokens is not None
        else STAGE2_AGGREGATE_MAX_NEW_TOKENS
    )
    return sample_max_new_tokens, aggregate_max_new_tokens


CONCEPT_DIR_PATTERN = re.compile(r"^c_(tr|per|vis)_(\d+)$")

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
            if concept_type in {"tr", "vis"}
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
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")
    if args.salient_patches_only and args.saliency_filter_mode == "none":
        raise SystemExit("--salient-patches-only true requires a saliency filter mode other than none.")
    if not args.salient_patches_only:
        args.saliency_filter_mode = "none"
    print(f"Salient patches only: {args.salient_patches_only}; effective filter: {args.saliency_filter_mode}")
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

    if args.concept_source == "trajectory" and not train_cfg.TEMPORAL_CONCEPTS_ON:
        raise SystemExit(
            "--concept-source trajectory requires temporal concepts, but the current "
            "architecture is built with TEMPORAL_CONCEPTS_ON=False (train.py). "
            "Use --concept-source visual."
        )

    checkpoint_path = str(resolve_checkpoint_path(args.checkpoint or DEFAULT_CHECKPOINT))
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_saliency_model(checkpoint_path, device=device)
    checkpoint_stat = Path(checkpoint_path).stat()
    model._gallery_checkpoint_signature = {
        "path": checkpoint_path, "size": checkpoint_stat.st_size,
        "mtime_ns": checkpoint_stat.st_mtime_ns,
    }
    model._gallery_dataset_signature = {
        "path": str(Path(args.dataset_dir).resolve()),
        "window_len": args.window_len, "stride": args.stride,
        "batch_size": args.batch_size, "seed": args.seed,
    }

    concept_indices = resolve_concept_indices(args.num_concepts, args.max_concepts)

    use_amp = args.use_amp if args.use_amp is not None else device.type == "cuda:1"
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
        concept_source=args.concept_source,
    )
    print(f"Saved concept top examples under: {concepts_root}")
    print(f"Saved metadata to: {output_dir / 'retrieved_examples.json'}")

    if args.skip_llm:
        print("Skipping LLM generation because --skip-llm was set.")
    else:
        sample_max_new_tokens, aggregate_max_new_tokens = resolve_llm_max_new_tokens(args)
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
            sample_max_new_tokens=sample_max_new_tokens,
            aggregate_max_new_tokens=aggregate_max_new_tokens,
            reuse_sample_descriptions=args.reuse_sample_descriptions,
            concept_source=args.concept_source,
            num_concepts=args.num_concepts,
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
