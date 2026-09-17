#!/usr/bin/env python3
"""Project learned visual prototypes onto their nearest dataset patch embeddings.

This script is designed for the ExplainableVidSalModel project.  It performs an
online exhaustive nearest-neighbour search for every *stage-specific* visual
prototype, replaces each prototype with its closest encoded patch, saves a
projected checkpoint, and writes auditable provenance/visualizations.

Important experimental rule
---------------------------
Projection must use the TRAINING split.  The evaluation split may be mirrored
to ``proto_replaced`` for the later projection-fidelity evaluation, but it must
not be used to select prototype patches.  A deliberate override is available
only for debugging.

The RGB evaluation frames are copied unchanged.  Prototype projection changes
model parameters, not input frames.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm exists in the project environment
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


DEFAULT_PROJECTION_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k/train")
DEFAULT_EVAL_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k/val")
DEFAULT_OUTPUT_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k/proto_replaced")
DEFAULT_CHECKPOINT = Path(
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/best_dhf1k.pth"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLIT_DIR_NAMES = {"train", "training", "val", "validation", "test", "testing", "proto_assign", "proto_replaced"}


@dataclass
class PatchMatch:
    stage: str
    prototype_index: int
    cosine_similarity: float
    video_name: str
    window_start_index: int
    feature_time_index: int
    feature_window_time: int
    assignment_feature_time_index: int
    feature_height: int
    feature_width: int
    patch_row: int
    patch_col: int
    source_frame_start_index: int
    source_frame_end_index: int
    source_center_frame_index: int
    source_frame_start_name: str
    source_frame_end_name: str
    source_center_frame_name: str
    pixel_x0: int
    pixel_y0: int
    pixel_x1: int
    pixel_y1: int
    source_image_width: int
    source_image_height: int
    patch_image: str = ""
    overlay_image: str = ""


@dataclass
class StageSearchState:
    scores: torch.Tensor
    embeddings: torch.Tensor
    matches: List[Optional[PatchMatch]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Project visual prototype banks onto nearest real video-patch embeddings.",
    )
    parser.add_argument(
        "--projection-root",
        type=Path,
        default=DEFAULT_PROJECTION_ROOT,
        help=(
            "Dataset split searched for nearest patches (TRAINING split for reported results). "
            "Must contain video folders like <root>/<video>/{images,maps,fixation}/."
        ),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Evaluation split whose images/maps/fixations are mirrored to output-root.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--projected-checkpoint",
        type=Path,
        default=None,
        help="Output checkpoint. Defaults to output-root/projection/projected_best_dhf1k.pth.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        # Script lives at interpretability_exp/exp1_proj_fidelity/; model/ is under src/.
        default=Path(__file__).resolve().parents[2],
        help="Project src directory containing model/ and pre_process/.",
    )
    parser.add_argument(
        "--model-factory",
        default="model.model:ExplainableVidSalModel",
        help="Import path MODULE:CALLABLE used to construct the model.",
    )
    parser.add_argument(
        "--model-kwargs-json",
        type=Path,
        default=None,
        help="Optional JSON object overriding auto-inferred model constructor arguments.",
    )
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Window stride. Keep 1 for an exhaustive sliding-window search; larger values are an approximation.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for compatibility; search is deterministic and indexed directly.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-chunk-size", type=int, default=65536)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to mirror evaluation images/maps/fixations.",
    )
    parser.add_argument("--skip-mirror", action="store_true")
    parser.add_argument("--overwrite-mirror", action="store_true")
    parser.add_argument(
        "--allow-evaluation-split-projection",
        action="store_true",
        help="Unsafe for reported fidelity results; permits projection-root == evaluation-root for debugging only.",
    )
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys. Strict loading is strongly recommended.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Debug only. Limiting windows invalidates exhaustive grounding.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10000,
        help=(
            "Save intermediate grounding results (matches CSV/JSON, current-best "
            "patch/overlay images, and a resumable search-state checkpoint) every "
            "N processed windows. Set <=0 to disable intermediate saves."
        ),
    )
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def same_path(a: Path, b: Path) -> bool:
    return a.expanduser().resolve() == b.expanduser().resolve()


def _looks_like_video_dataset_root(root: Path) -> bool:
    """True if root has at least one child with images/maps/fixation."""
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (
            (child / "images").is_dir()
            and (child / "maps").is_dir()
            and (child / "fixation").is_dir()
        ):
            return True
    return False


def _dataset_layout_hint(root: Path) -> str:
    child_names = {p.name.lower() for p in root.iterdir() if p.is_dir()}
    overlap = sorted(child_names & SPLIT_DIR_NAMES)
    if overlap:
        return (
            f"{root} looks like a dataset parent containing splits {overlap}. "
            f"Pass a split directory instead, e.g. {root / 'train'} or {root / 'val'}."
        )
    return (
        f"{root} must contain video folders of the form "
        f"<root>/<video_name>/{{images,maps,fixation}}/<frame>.png"
    )


def validate_dataset_root(root: Path, name: str) -> None:
    if not root.is_dir():
        raise NotADirectoryError(f"{name} dataset not found: {root}")
    if not _looks_like_video_dataset_root(root):
        raise RuntimeError(
            f"No valid video folders found under {name}={root}. {_dataset_layout_hint(root)}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    validate_dataset_root(args.projection_root, "projection-root")
    if not args.skip_mirror:
        validate_dataset_root(args.evaluation_root, "evaluation-root")
    if args.window_len < 1 or args.stride < 1 or args.batch_size < 1:
        raise ValueError("window-len, stride, and batch-size must all be positive")
    if args.similarity_chunk_size < 1:
        raise ValueError("similarity-chunk-size must be positive")
    if same_path(args.projection_root, args.evaluation_root) and not args.allow_evaluation_split_projection:
        raise RuntimeError(
            "Refusing to select prototypes from the evaluation split. Pass the DHF1K training "
            "directory with --projection-root. For a non-reportable smoke test only, add "
            "--allow-evaluation-split-projection."
        )
    if args.max_windows is not None and args.max_windows < 1:
        raise ValueError("max-windows must be positive")


def serialize_search_states(states: Mapping[str, StageSearchState]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for stage, state in states.items():
        payload[stage] = {
            "scores": state.scores.detach().cpu(),
            "embeddings": state.embeddings.detach().cpu(),
            "matches": [
                None if match is None else asdict(match) for match in state.matches
            ],
        }
    return payload


def save_grounding_results(
    *,
    states: Mapping[str, StageSearchState],
    projection_root: Path,
    artifact_root: Path,
    args: argparse.Namespace,
    windows_processed: int,
    total_windows: int,
    projected_checkpoint: Optional[Path] = None,
    partial: bool = False,
) -> List[PatchMatch]:
    """
    Persist current best grounding results.

    For intermediate snapshots, prototypes still without a match are skipped.
    """
    artifact_root.mkdir(parents=True, exist_ok=True)
    matches = save_match_visualizations(
        states,
        projection_root,
        artifact_root,
        allow_incomplete=partial,
    )

    status = {
        "windows_processed": int(windows_processed),
        "total_windows": int(total_windows),
        "partial": bool(partial),
        "prototypes_with_matches": len(matches),
        "prototypes_total": sum(len(state.matches) for state in states.values()),
    }
    with (artifact_root / "grounding_progress.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)

    torch.save(
        {
            "windows_processed": int(windows_processed),
            "total_windows": int(total_windows),
            "partial": bool(partial),
            "states": serialize_search_states(states),
        },
        artifact_root / "search_state.pt",
    )

    if matches:
        write_metadata(
            matches=matches,
            states=states,
            args=args,
            projected_checkpoint=projected_checkpoint
            if projected_checkpoint is not None
            else artifact_root / "projected_checkpoint_pending.pth",
            artifact_root=artifact_root,
            partial=partial,
            windows_processed=windows_processed,
            total_windows=total_windows,
        )
    print(
        f"Saved grounding results after {windows_processed}/{total_windows} windows "
        f"({len(matches)} prototype matches) -> {artifact_root}"
    )
    return matches


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    # Explicit weights_only=False is required because training checkpoints often
    # include optimizer/scaler metadata. Older PyTorch versions lack this kwarg.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and bool(obj) and all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    )


def extract_state_dict(checkpoint: Any) -> Tuple[Mapping[str, torch.Tensor], Optional[str]]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict(), "__module__"
    if is_state_dict(checkpoint):
        return checkpoint, None
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "network", "net"):
            value = checkpoint.get(key)
            if is_state_dict(value):
                return value, key
    raise TypeError(
        "Could not find a model state_dict in the checkpoint. Expected a raw state_dict "
        "or one of: model_state_dict/state_dict/model/network/net."
    )


def remove_uniform_prefix(
    state_dict: Mapping[str, torch.Tensor], prefix: str = "module."
) -> Tuple[Dict[str, torch.Tensor], bool]:
    keys = list(state_dict)
    had_prefix = bool(keys) and all(key.startswith(prefix) for key in keys)
    if not had_prefix:
        return dict(state_dict), False
    return {key[len(prefix):]: value for key, value in state_dict.items()}, True


def infer_constructor_kwargs(state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer the core shape-defining constructor arguments from a checkpoint."""
    stage_pattern = re.compile(r"^concept_creations\.([^.]+)\.visual_concepts$")
    stage_tensors: List[Tuple[str, torch.Tensor]] = []
    for key, tensor in state.items():
        match = stage_pattern.match(key)
        if match:
            stage_tensors.append((match.group(1), tensor))
    if not stage_tensors:
        raise KeyError("Checkpoint has no concept_creations.<stage>.visual_concepts tensors")

    natural_stage_key = lambda item: int(re.search(r"\d+", item[0]).group()) if re.search(r"\d+", item[0]) else item[0]
    stage_tensors.sort(key=natural_stage_key)
    stages = tuple(stage for stage, _ in stage_tensors)
    prototype_counts = {int(t.shape[0]) for _, t in stage_tensors}
    concept_dims = {int(t.shape[1]) for _, t in stage_tensors}
    if len(prototype_counts) != 1 or len(concept_dims) != 1:
        raise ValueError("All visual prototype banks must have common K and D for this model constructor")

    kwargs: Dict[str, Any] = {
        "backbone_stages": stages,
        "pretrained_backbone": False,
        "freeze_backbone": False,
        "input_format": "BTCHW",
        "resize_to": (224, 384),
        "concept_dim": concept_dims.pop(),
        "num_concepts": prototype_counts.pop(),
        "top_k": 8,
        "return_details": True,
    }

    hidden_key = f"concept_creations.{stages[0]}.visual_encoder.0.weight"
    if hidden_key in state:
        kwargs["concept_hidden_dim"] = int(state[hidden_key].shape[0])

    motion_banks = [
        tensor for key, tensor in state.items()
        if key.startswith("motion_concept_creations.") and key.endswith(".motion_concepts")
    ]
    kwargs["motion_concepts_on"] = bool(motion_banks)
    if motion_banks:
        kwargs["num_motion_concepts"] = int(motion_banks[0].shape[0])

    # A stage-1 1x1x1 feature projection normally has [decoder_C, 96, 1, 1, 1].
    # Use it only when the match is unambiguous; otherwise retain constructor default.
    decoder_widths = {
        int(t.shape[0])
        for key, t in state.items()
        if key.startswith("saliency_prediction.")
        and "stage1" in key
        and t.ndim == 5
        and int(t.shape[1]) == 96
        and int(t.shape[0]) > 1
        and tuple(t.shape[-3:]) == (1, 1, 1)
    }
    if len(decoder_widths) == 1:
        kwargs["saliency_hidden_dim"] = decoder_widths.pop()

    # Match training architecture flags from checkpoint contents.
    kwargs["use_shared_concept_activations"] = any(
        key.endswith(".shared_patch_proj.weight") for key in state
    )
    kwargs["use_temporal_feature_infusion"] = any(
        key.startswith("temporal_feature_infusers.") for key in state
    )
    kwargs["visual_concept_on"] = True
    kwargs["temporal_concepts_on"] = any(".temporal_concepts" in key for key in state)
    return kwargs


def import_factory(spec: str, repo_root: Path) -> Callable[..., nn.Module]:
    if ":" not in spec:
        raise ValueError("model-factory must have the form MODULE:CALLABLE")
    module_name, attr_name = spec.split(":", 1)
    sys.path.insert(0, str(repo_root.expanduser().resolve()))
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"Model factory is not callable: {spec}")
    return factory


def filter_supported_kwargs(factory: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(factory)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_model(
    args: argparse.Namespace,
) -> Tuple[nn.Module, Any, Optional[str], bool, Dict[str, torch.Tensor]]:
    checkpoint = torch_load(args.checkpoint, "cpu")
    raw_state, container_key = extract_state_dict(checkpoint)
    clean_state, had_module_prefix = remove_uniform_prefix(raw_state)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        factory = import_factory(args.model_factory, args.repo_root)
        kwargs = infer_constructor_kwargs(clean_state)
        if args.model_kwargs_json is not None:
            with args.model_kwargs_json.open("r", encoding="utf-8") as handle:
                overrides = json.load(handle)
            if not isinstance(overrides, dict):
                raise TypeError("model-kwargs-json must contain one JSON object")
            kwargs.update(overrides)
        kwargs = filter_supported_kwargs(factory, kwargs)
        print("Model constructor arguments:", json.dumps(kwargs, indent=2, default=list))
        model = factory(**kwargs)

    incompatible = model.load_state_dict(clean_state, strict=not args.non_strict_checkpoint)
    if args.non_strict_checkpoint:
        if incompatible.missing_keys:
            print("WARNING missing checkpoint keys:", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print("WARNING unexpected checkpoint keys:", incompatible.unexpected_keys)

    model.to(torch.device(args.device)).eval()
    return model, checkpoint, container_key, had_module_prefix, clean_state


def import_dataset_api(repo_root: Path) -> Tuple[type, Callable[..., Any]]:
    sys.path.insert(0, str(repo_root.expanduser().resolve()))
    dataloader_module = importlib.import_module("pre_process.dataloader")
    collate_module = importlib.import_module("pre_process.collate")
    return dataloader_module.DatasetLoader, collate_module.video_saliency_collate_fn


def list_frame_names(images_dir: Path) -> List[str]:
    names = sorted(
        item.name for item in images_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not names:
        raise RuntimeError(f"No RGB frames found in {images_dir}")
    return names


def initialize_search_states(model: nn.Module) -> Dict[str, StageSearchState]:
    if not hasattr(model, "concept_creations"):
        raise AttributeError("Model has no concept_creations module dictionary")
    states: Dict[str, StageSearchState] = {}
    for stage, module in model.concept_creations.items():
        if not hasattr(module, "visual_concepts"):
            raise AttributeError(f"concept_creations[{stage!r}] has no visual_concepts")
        bank = module.visual_concepts
        if bank.ndim != 2:
            raise ValueError(f"{stage} visual_concepts must be [K,D], got {tuple(bank.shape)}")
        k, d = map(int, bank.shape)
        states[stage] = StageSearchState(
            scores=torch.full((k,), -torch.inf, dtype=torch.float32),
            embeddings=torch.empty((k, d), dtype=torch.float32),
            matches=[None] * k,
        )
    return states


def source_interval(feature_index: int, feature_size: int, source_size: int) -> Tuple[int, int, int]:
    start = int(math.floor(feature_index * source_size / feature_size))
    end_exclusive = int(math.ceil((feature_index + 1) * source_size / feature_size))
    start = max(0, min(start, source_size - 1))
    end = max(start, min(end_exclusive - 1, source_size - 1))
    center = (start + end) // 2
    return start, end, center


def pixel_interval(feature_index: int, feature_size: int, source_size: int) -> Tuple[int, int]:
    start = int(math.floor(feature_index * source_size / feature_size))
    end = int(math.ceil((feature_index + 1) * source_size / feature_size))
    start = max(0, min(start, source_size - 1))
    end = max(start + 1, min(end, source_size))
    return start, end


def make_match(
    *,
    stage: str,
    prototype_index: int,
    score: float,
    flat_embedding_index: int,
    feature_shape: Sequence[int],
    feature_window_t: int,
    assignment_time_index: int,
    video_names: Sequence[str],
    window_starts: Sequence[int],
    n_frames: Sequence[int],
    frame_names_by_video: Mapping[str, List[str]],
    projection_root: Path,
) -> PatchMatch:
    batch_size, _, feature_t, feature_h, feature_w = map(int, feature_shape)
    per_time = feature_h * feature_w
    per_sample = feature_t * per_time
    batch_index = flat_embedding_index // per_sample
    remainder = flat_embedding_index % per_sample
    time_index = remainder // per_time
    patch_index = remainder % per_time
    row, col = divmod(patch_index, feature_w)
    if not 0 <= batch_index < batch_size:
        raise IndexError("Decoded patch batch index is out of range")

    video_name = video_names[batch_index]
    window_start = int(window_starts[batch_index])
    valid_frames = int(n_frames[batch_index])
    # The latest VisualConceptCreation assigns only the final feature-time slice,
    # so feature_shape has T=1. Map its recorded index in the full backbone
    # feature window to the corresponding raw-frame interval.
    if feature_t != 1 or time_index != 0:
        raise ValueError(
            "Expected latest-model last-frame visual embeddings with T=1; "
            f"received feature_shape={tuple(feature_shape)} and time_index={time_index}"
        )
    local_t0, local_t1, local_center = source_interval(
        assignment_time_index, feature_window_t, valid_frames
    )
    global_t0 = window_start + local_t0
    global_t1 = window_start + local_t1
    global_center = window_start + local_center
    frame_names = frame_names_by_video[video_name]
    center_path = projection_root / video_name / "images" / frame_names[global_center]
    with Image.open(center_path) as image:
        width, height = image.size
    x0, x1 = pixel_interval(col, feature_w, width)
    y0, y1 = pixel_interval(row, feature_h, height)

    return PatchMatch(
        stage=stage,
        prototype_index=prototype_index,
        cosine_similarity=score,
        video_name=video_name,
        window_start_index=window_start,
        feature_time_index=time_index,
        feature_window_time=feature_window_t,
        assignment_feature_time_index=assignment_time_index,
        feature_height=feature_h,
        feature_width=feature_w,
        patch_row=row,
        patch_col=col,
        source_frame_start_index=global_t0,
        source_frame_end_index=global_t1,
        source_center_frame_index=global_center,
        source_frame_start_name=frame_names[global_t0],
        source_frame_end_name=frame_names[global_t1],
        source_center_frame_name=frame_names[global_center],
        pixel_x0=x0,
        pixel_y0=y0,
        pixel_x1=x1,
        pixel_y1=y1,
        source_image_width=width,
        source_image_height=height,
    )


def update_stage_maxima(
    *,
    stage: str,
    embeddings: torch.Tensor,
    feature_shape: Sequence[int],
    feature_window_t: int,
    assignment_time_index: int,
    prototype_bank: torch.Tensor,
    state: StageSearchState,
    video_names: Sequence[str],
    window_starts: Sequence[int],
    n_frames: Sequence[int],
    frame_names_by_video: Mapping[str, List[str]],
    projection_root: Path,
    chunk_size: int,
) -> None:
    embeddings = F.normalize(embeddings.detach().float(), dim=-1)
    prototypes = F.normalize(prototype_bank.detach().float(), dim=-1)
    for chunk_start in range(0, embeddings.shape[0], chunk_size):
        chunk = embeddings[chunk_start:chunk_start + chunk_size]
        similarities = chunk @ prototypes.T
        chunk_scores, chunk_indices = similarities.max(dim=0)
        chunk_scores_cpu = chunk_scores.cpu()
        better = chunk_scores_cpu > state.scores
        if not bool(better.any()):
            continue
        for prototype_index in better.nonzero(as_tuple=False).flatten().tolist():
            local_index = int(chunk_indices[prototype_index].item())
            flat_index = chunk_start + local_index
            score = float(chunk_scores_cpu[prototype_index].item())
            state.scores[prototype_index] = score
            state.embeddings[prototype_index].copy_(embeddings[flat_index].cpu())
            state.matches[prototype_index] = make_match(
                stage=stage,
                prototype_index=prototype_index,
                score=score,
                flat_embedding_index=flat_index,
                feature_shape=feature_shape,
                feature_window_t=feature_window_t,
                assignment_time_index=assignment_time_index,
                video_names=video_names,
                window_starts=window_starts,
                n_frames=n_frames,
                frame_names_by_video=frame_names_by_video,
                projection_root=projection_root,
            )


@torch.inference_mode()
def search_nearest_patches(
    model: nn.Module,
    dataset: Any,
    collate_fn: Callable[..., Any],
    args: argparse.Namespace,
    artifact_root: Optional[Path] = None,
) -> Dict[str, StageSearchState]:
    required_model_api = ("_normalize_video_layout", "_resize_feature_for_concepts", "backbone")
    for name in required_model_api:
        if not hasattr(model, name):
            raise AttributeError(f"Model lacks required grounding API: {name}")

    states = initialize_search_states(model)
    frame_names_by_video = {
        video_name: list_frame_names(args.projection_root / video_name / "images")
        for video_name in dataset.video_dirs
    }
    total_windows = len(dataset)
    if args.max_windows is not None:
        total_windows = min(total_windows, args.max_windows)

    device = torch.device(args.device)
    batch_starts = range(0, total_windows, args.batch_size)
    progress = tqdm(batch_starts, total=math.ceil(total_windows / args.batch_size), desc="Grounding")
    windows_processed = 0
    save_every = int(getattr(args, "save_every", 0) or 0)
    snapshot_root = None
    if artifact_root is not None and save_every > 0:
        snapshot_root = artifact_root / "grounding_snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)

    for first_index in progress:
        indices = list(range(first_index, min(first_index + args.batch_size, total_windows)))
        samples = [dataset[index] for index in indices]
        # Never mix different temporal lengths in one grounding batch. Collate pads
        # short tail windows; because concepts use the final time slice, a padded
        # sample would otherwise be grounded to padding rather than a real frame.
        groups: Dict[int, List[Tuple[int, Any]]] = {}
        for index, sample in zip(indices, samples):
            # RGB is item 1 in both the fixation-aware and legacy dataset APIs.
            # Its leading axis is the real (unpadded) window length.
            sample_length = int(sample[1].shape[0])
            groups.setdefault(sample_length, []).append((index, sample))

        for group in groups.values():
            group_indices = [item[0] for item in group]
            group_samples = [item[1] for item in group]
            batch = collate_fn(group_samples)
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise TypeError("Collate function must return at least (video_names, rgb_batch, ...)")
            video_names = list(batch[0])
            rgb_batch = batch[1].to(device, non_blocking=True)
            valid_lengths = [int(sample[1].shape[0]) for sample in group_samples]
            window_starts = [int(dataset.windows[index][1]) for index in group_indices]

            normalized_video = model._normalize_video_layout(rgb_batch)
            features_dict = model.backbone.forward_features(normalized_video)
            for stage, module in model.concept_creations.items():
                features = model._resize_feature_for_concepts(features_dict[stage], stage)
                if features.ndim != 5:
                    raise ValueError(f"{stage} features must be [B,C,T,H,W], got {tuple(features.shape)}")
                if not hasattr(module, "_build_visual_concepts_from_patches"):
                    raise AttributeError(
                        f"{stage} concept module must expose _build_visual_concepts_from_patches "
                        "so grounding uses the exact latest-model embedding path"
                    )
                # This is intentionally the same helper used by
                # VisualConceptCreation.forward. In the latest model it selects
                # features[:, :, -1:] before encoding.
                visual_out = module._build_visual_concepts_from_patches(features)
                embeddings = visual_out["visual_patch_embeddings"]
                metadata = visual_out["visual_metadata"]
                shape_dict = metadata["feature_shape"]
                embedding_feature_shape = (
                    int(shape_dict["B"]),
                    int(shape_dict["C"]),
                    int(shape_dict["T"]),
                    int(shape_dict["H"]),
                    int(shape_dict["W"]),
                )
                update_stage_maxima(
                    stage=stage,
                    embeddings=embeddings,
                    feature_shape=embedding_feature_shape,
                    feature_window_t=int(metadata["window_T"]),
                    assignment_time_index=int(metadata["assignment_time_index"]),
                    prototype_bank=module.visual_concepts,
                    state=states[stage],
                    video_names=video_names,
                    window_starts=window_starts,
                    n_frames=valid_lengths,
                    frame_names_by_video=frame_names_by_video,
                    projection_root=args.projection_root,
                    chunk_size=args.similarity_chunk_size,
                )

            del rgb_batch, normalized_video, features_dict

        windows_processed += len(indices)
        progress.set_postfix(windows=windows_processed)
        if (
            snapshot_root is not None
            and save_every > 0
            and (
                windows_processed % save_every == 0
                or windows_processed >= total_windows
            )
        ):
            save_grounding_results(
                states=states,
                projection_root=args.projection_root,
                artifact_root=snapshot_root / f"windows_{windows_processed:08d}",
                args=args,
                windows_processed=windows_processed,
                total_windows=total_windows,
                partial=windows_processed < total_windows,
            )
            # Keep a rolling "latest" copy for easy inspection/resume.
            save_grounding_results(
                states=states,
                projection_root=args.projection_root,
                artifact_root=snapshot_root / "latest",
                args=args,
                windows_processed=windows_processed,
                total_windows=total_windows,
                partial=windows_processed < total_windows,
            )

    incomplete = {
        stage: [i for i, match in enumerate(state.matches) if match is None]
        for stage, state in states.items()
    }
    incomplete = {stage: indices for stage, indices in incomplete.items() if indices}
    if incomplete:
        raise RuntimeError(f"No source patch was found for some prototypes: {incomplete}")
    return states


def apply_projection(model: nn.Module, states: Mapping[str, StageSearchState]) -> None:
    with torch.no_grad():
        for stage, state in states.items():
            parameter = model.concept_creations[stage].visual_concepts
            projected = F.normalize(state.embeddings, dim=-1).to(
                device=parameter.device, dtype=parameter.dtype
            )
            if projected.shape != parameter.shape:
                raise ValueError(
                    f"Projected bank shape {tuple(projected.shape)} does not match "
                    f"{stage} bank {tuple(parameter.shape)}"
                )
            parameter.copy_(projected)


def add_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {f"module.{key}": value for key, value in state.items()}


def save_projected_checkpoint(
    *,
    model: nn.Module,
    original_checkpoint: Any,
    container_key: Optional[str],
    had_module_prefix: bool,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    projected_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if had_module_prefix:
        projected_state = add_module_prefix(projected_state)

    if container_key is None:
        payload: Any = projected_state
    elif container_key == "__module__":
        # Saving the full pickled module is brittle; use a standard checkpoint dict.
        payload = {"model_state_dict": projected_state}
    else:
        payload = dict(original_checkpoint)
        payload[container_key] = projected_state
        payload.pop("optimizer_state_dict", None)
        payload.pop("optimizer", None)
        payload.pop("scaler_state_dict", None)
        payload.pop("scaler", None)
    torch.save(payload, destination)


def save_match_visualizations(
    states: Mapping[str, StageSearchState],
    projection_root: Path,
    artifact_root: Path,
    allow_incomplete: bool = False,
) -> List[PatchMatch]:
    all_matches: List[PatchMatch] = []
    for stage, state in states.items():
        patch_dir = artifact_root / "patches" / stage
        overlay_dir = artifact_root / "overlays" / stage
        patch_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        for match_optional in state.matches:
            if match_optional is None:
                if allow_incomplete:
                    continue
                raise AssertionError(f"Missing match for stage={stage}")
            match = match_optional
            image_path = (
                projection_root / match.video_name / "images" / match.source_center_frame_name
            )
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            crop = image.crop((match.pixel_x0, match.pixel_y0, match.pixel_x1, match.pixel_y1))
            patch_path = patch_dir / f"prototype_{match.prototype_index:04d}.png"
            crop.save(patch_path)

            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            line_width = max(2, round(min(image.size) / 180))
            draw.rectangle(
                (match.pixel_x0, match.pixel_y0, match.pixel_x1 - 1, match.pixel_y1 - 1),
                outline=(255, 0, 0),
                width=line_width,
            )
            overlay_path = overlay_dir / f"prototype_{match.prototype_index:04d}.png"
            overlay.save(overlay_path)

            match.patch_image = str(patch_path.relative_to(artifact_root))
            match.overlay_image = str(overlay_path.relative_to(artifact_root))
            all_matches.append(match)
    return all_matches


def write_metadata(
    *,
    matches: Sequence[PatchMatch],
    states: Mapping[str, StageSearchState],
    args: argparse.Namespace,
    projected_checkpoint: Path,
    artifact_root: Path,
    partial: bool = False,
    windows_processed: Optional[int] = None,
    total_windows: Optional[int] = None,
) -> None:
    if not matches:
        raise ValueError("Cannot write grounding metadata with zero matches")
    rows = [asdict(match) for match in matches]
    csv_path = artifact_root / "prototype_matches.csv"
    json_path = artifact_root / "prototype_matches.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stage_stats: Dict[str, Any] = {}
    for stage, state in states.items():
        found = [m for m in state.matches if m is not None]
        if found:
            found_scores = torch.tensor(
                [m.cosine_similarity for m in found], dtype=torch.float32
            )
            stage_stats[stage] = {
                "num_prototypes": len(state.matches),
                "num_matched": len(found),
                "cosine_min": float(found_scores.min().item()),
                "cosine_mean": float(found_scores.mean().item()),
                "cosine_max": float(found_scores.max().item()),
            }
        else:
            stage_stats[stage] = {
                "num_prototypes": len(state.matches),
                "num_matched": 0,
                "cosine_min": None,
                "cosine_mean": None,
                "cosine_max": None,
            }

    report = {
        "checkpoint_before_projection": str(args.checkpoint),
        "checkpoint_after_projection": str(projected_checkpoint),
        "projection_root": str(args.projection_root),
        "evaluation_root_mirrored": None if args.skip_mirror else str(args.evaluation_root),
        "window_len": args.window_len,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "exhaustive": args.max_windows is None and not partial,
        "partial_snapshot": bool(partial),
        "windows_processed": windows_processed,
        "total_windows": total_windows,
        "evaluation_split_override_used": same_path(args.projection_root, args.evaluation_root),
        "stages": stage_stats,
        "notes": [
            "Each stage has an independent prototype bank and was projected independently.",
            "The latest VisualConceptCreation assigns visual concepts only on the final backbone time slice; grounding uses that identical code path.",
            "The saved center frame is a visualization anchor. A 3D Video Swin feature may depend on the recorded temporal interval and broader receptive-field context.",
            "The pixel box is the feature-grid cell mapped proportionally to the raw frame; it is not an exact analytical receptive-field boundary.",
            "Evaluation RGB frames were mirrored unchanged; only the checkpoint's visual prototype tensors were replaced.",
        ],
        "matches": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def copy_one(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        if destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(destination)
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:  # pragma: no cover
        raise ValueError(mode)


def mirror_evaluation_tree(
    source_root: Path,
    destination_root: Path,
    mode: str,
    overwrite: bool,
) -> None:
    video_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    for video_dir in tqdm(video_dirs, desc="Mirroring evaluation split"):
        for subdirectory in ("images", "maps", "fixation"):
            source_dir = video_dir / subdirectory
            if not source_dir.is_dir():
                raise FileNotFoundError(
                    f"Required evaluation directory is missing: {source_dir}"
                )
            for source_file in source_dir.iterdir():
                if source_file.is_file():
                    destination = destination_root / video_dir.name / subdirectory / source_file.name
                    copy_one(source_file, destination, mode, overwrite)


def main() -> None:
    args = parse_args()
    args.projection_root = args.projection_root.expanduser().resolve()
    args.evaluation_root = args.evaluation_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.projected_checkpoint is None:
        args.projected_checkpoint = args.output_root / "projection" / "projected_best_dhf1k.pth"
    else:
        args.projected_checkpoint = args.projected_checkpoint.expanduser().resolve()

    validate_args(args)
    set_determinism(args.seed)
    print(f"Projection/search split: {args.projection_root}")
    print(f"Evaluation split to mirror: {args.evaluation_root}")
    if same_path(args.projection_root, args.evaluation_root):
        print("WARNING: evaluation-split projection override is active; do not report these fidelity results.")

    model, checkpoint, container_key, had_module_prefix, _ = load_model(args)
    DatasetLoader, collate_fn = import_dataset_api(args.repo_root)
    dataset = DatasetLoader(
        str(args.projection_root),
        window_len=args.window_len,
        stride=args.stride,
    )
    print(f"Searching {len(dataset)} windows across {len(dataset.video_dirs)} videos")
    artifact_root = args.output_root / "projection"
    artifact_root.mkdir(parents=True, exist_ok=True)
    states = search_nearest_patches(
        model,
        dataset,
        collate_fn,
        args,
        artifact_root=artifact_root,
    )
    apply_projection(model, states)
    save_projected_checkpoint(
        model=model,
        original_checkpoint=checkpoint,
        container_key=container_key,
        had_module_prefix=had_module_prefix,
        destination=args.projected_checkpoint,
    )

    matches = save_grounding_results(
        states=states,
        projection_root=args.projection_root,
        artifact_root=artifact_root,
        args=args,
        windows_processed=len(dataset) if args.max_windows is None else min(len(dataset), args.max_windows),
        total_windows=len(dataset) if args.max_windows is None else min(len(dataset), args.max_windows),
        projected_checkpoint=args.projected_checkpoint,
        partial=False,
    )

    if not args.skip_mirror:
        mirror_evaluation_tree(
            args.evaluation_root,
            args.output_root,
            mode=args.copy_mode,
            overwrite=args.overwrite_mirror,
        )

    print(f"Projected checkpoint: {args.projected_checkpoint}")
    print(f"Prototype provenance: {artifact_root / 'prototype_matches.csv'}")
    print(f"Projection report: {artifact_root / 'prototype_matches.json'}")
    print("Done. Use the projected checkpoint—not modified RGB frames—in proj_fid.py.")


if __name__ == "__main__":
    main()
