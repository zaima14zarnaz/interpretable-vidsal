#!/usr/bin/env python3
"""Create a control checkpoint whose prototypes are random training patches.

This deliberately does *not* find nearest patches.  For every stage-specific
prototype, it samples a random DHF1K training window and a random spatial patch,
runs that patch through the model's own visual encoder, L2-normalizes the result,
and copies it into ``visual_concepts``.  All other learned tensors are preserved.

Keep this script beside ``patch_to_prototype_assign.py`` in the repository's
``src`` directory; it reuses that script's checkpoint/model loading helpers.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from patch_to_prototype_assign import (
    import_dataset_api,
    load_model,
    save_projected_checkpoint,
    set_determinism,
)


DEFAULT_TRAIN_ROOT = Path(
    "/data/quantization/zaima/videosal_datasets/dhf1k/train"
)
DEFAULT_CHECKPOINT = Path(
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/best_dhf1k.pth"
)
DEFAULT_OUTPUT = Path(
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/random_patch_ckpt.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace every visual prototype with a random TRAINING-patch embedding."
    )
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--repo-root",
        type=Path,
        # Script lives at interpretability_exp/exp1_proj_fidelity/; model/ is under src/.
        default=Path(__file__).resolve().parents[2],
        help="Project src directory containing model/ and pre_process/.",
    )
    parser.add_argument("--model-factory", default="model.model:ExplainableVidSalModel")
    parser.add_argument("--model-kwargs-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument(
        "--allow-non-training-root",
        action="store_true",
        help="Permit a root not named train/training (for debugging only).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.train_root = args.train_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.metadata is None:
        args.metadata = args.output.with_suffix(".metadata.json")
    else:
        args.metadata = args.metadata.expanduser().resolve()

    if not args.train_root.is_dir():
        raise NotADirectoryError(f"Training split not found: {args.train_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {args.checkpoint}")
    if args.window_len < 1 or args.stride < 1:
        raise ValueError("window-len and stride must be positive")
    if args.train_root.name.lower() not in {"train", "training"} and not args.allow_non_training_root:
        raise RuntimeError(
            f"Refusing to sample patches from {args.train_root}. Use the DHF1K train split; "
            "pass --allow-non-training-root only for a non-reportable smoke test."
        )


def visual_prototype_keys(model: nn.Module) -> set[str]:
    return {
        f"concept_creations.{stage}.visual_concepts"
        for stage in model.concept_creations.keys()
    }


def verify_only_prototypes_changed(
    model: nn.Module,
    original_state: Mapping[str, torch.Tensor],
) -> None:
    allowed = visual_prototype_keys(model)
    current = model.state_dict()
    if set(current) != set(original_state):
        missing = sorted(set(original_state) - set(current))
        extra = sorted(set(current) - set(original_state))
        raise RuntimeError(f"State-dict keys changed; missing={missing}, extra={extra}")
    changed_allowed = set()
    for key, tensor in current.items():
        if key in allowed:
            if not torch.equal(tensor.detach().cpu(), original_state[key].detach().cpu()):
                changed_allowed.add(key)
        elif not torch.equal(tensor.detach().cpu(), original_state[key].detach().cpu()):
            raise RuntimeError(f"Unexpected non-prototype tensor change: {key}")
    if changed_allowed != allowed:
        missing = sorted(allowed - changed_allowed)
        raise RuntimeError(f"These prototype banks were not changed: {missing}")


def select_window_indices(
    model: nn.Module,
    dataset_size: int,
    rng: random.Random,
) -> Dict[int, List[Tuple[str, int]]]:
    """Map dataset-window index -> requested (stage, prototype) assignments."""
    if dataset_size < 1:
        raise RuntimeError("The training dataset produced no temporal windows")
    requests: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for stage, module in model.concept_creations.items():
        count = int(module.visual_concepts.shape[0])
        if dataset_size >= count:
            indices = rng.sample(range(dataset_size), count)
        else:
            indices = [rng.randrange(dataset_size) for _ in range(count)]
        for prototype_index, dataset_index in enumerate(indices):
            requests[dataset_index].append((stage, prototype_index))
    return dict(requests)


@torch.inference_mode()
def collect_random_patch_embeddings(
    model: nn.Module,
    dataset: Any,
    collate_fn: Any,
    requests: Mapping[int, List[Tuple[str, int]]],
    rng: random.Random,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    assignments = {
        stage: torch.empty_like(module.visual_concepts, device="cpu", dtype=torch.float32)
        for stage, module in model.concept_creations.items()
    }
    records: List[Dict[str, Any]] = []

    for dataset_index in tqdm(sorted(requests), desc="Sampling random training patches"):
        sample = dataset[dataset_index]
        batch = collate_fn([sample])
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise TypeError("Collate function must return at least (video_names, rgb_batch, ...)")
        video_name = str(batch[0][0])
        rgb = batch[1].to(device, non_blocking=True)
        normalized = model._normalize_video_layout(rgb)
        features_by_stage = model.backbone.forward_features(normalized)

        requests_by_stage: Dict[str, List[int]] = defaultdict(list)
        for stage, prototype_index in requests[dataset_index]:
            requests_by_stage[stage].append(prototype_index)

        for stage, prototype_indices in requests_by_stage.items():
            module = model.concept_creations[stage]
            features = model._resize_feature_for_concepts(features_by_stage[stage], stage)
            visual_out = module._build_visual_concepts_from_patches(features)
            embeddings = F.normalize(
                visual_out["visual_patch_embeddings"].detach().float(), dim=-1
            )
            metadata = visual_out["visual_metadata"]
            feature_shape = metadata["feature_shape"]
            height = int(feature_shape["H"])
            width = int(feature_shape["W"])
            if int(feature_shape["B"]) != 1:
                raise RuntimeError("Random-patch extraction requires batch size 1")

            patch_count = int(embeddings.shape[0])
            if patch_count >= len(prototype_indices):
                patch_indices = rng.sample(range(patch_count), len(prototype_indices))
            else:
                patch_indices = [rng.randrange(patch_count) for _ in prototype_indices]

            original_bank = F.normalize(module.visual_concepts.detach().float(), dim=-1)
            window_start = int(dataset.windows[dataset_index][1])
            for prototype_index, patch_index in zip(prototype_indices, patch_indices):
                selected = embeddings[patch_index]
                assignments[stage][prototype_index].copy_(selected.cpu())
                records.append(
                    {
                        "stage": stage,
                        "prototype_index": prototype_index,
                        "dataset_window_index": dataset_index,
                        "video_name": video_name,
                        "window_start": window_start,
                        "window_length": int(sample[1].shape[0]),
                        "assignment_time_index": int(metadata["assignment_time_index"]),
                        "patch_index": patch_index,
                        "patch_row": patch_index // width,
                        "patch_col": patch_index % width,
                        "feature_grid_height": height,
                        "feature_grid_width": width,
                        "cosine_to_original_prototype": float(
                            torch.dot(original_bank[prototype_index], selected).item()
                        ),
                    }
                )

        del rgb, normalized, features_by_stage

    expected = sum(int(m.visual_concepts.shape[0]) for m in model.concept_creations.values())
    if len(records) != expected:
        raise RuntimeError(f"Collected {len(records)} assignments; expected {expected}")
    return assignments, records


def apply_assignments(model: nn.Module, assignments: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for stage, values in assignments.items():
            parameter = model.concept_creations[stage].visual_concepts
            replacement = F.normalize(values, dim=-1).to(parameter.device, parameter.dtype)
            if replacement.shape != parameter.shape:
                raise ValueError(
                    f"{stage}: replacement {tuple(replacement.shape)} != bank {tuple(parameter.shape)}"
                )
            parameter.copy_(replacement)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_determinism(args.seed)
    rng = random.Random(args.seed)

    model, checkpoint, container_key, had_module_prefix, original_state = load_model(args)
    required = ("concept_creations", "backbone", "_normalize_video_layout", "_resize_feature_for_concepts")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(f"Model lacks required random-patch API: {missing}")

    DatasetLoader, collate_fn = import_dataset_api(args.repo_root)
    dataset = DatasetLoader(str(args.train_root), window_len=args.window_len, stride=args.stride)
    requests = select_window_indices(model, len(dataset), rng)
    assignments, records = collect_random_patch_embeddings(
        model, dataset, collate_fn, requests, rng, torch.device(args.device)
    )
    apply_assignments(model, assignments)
    verify_only_prototypes_changed(model, original_state)
    save_projected_checkpoint(
        model=model,
        original_checkpoint=checkpoint,
        container_key=container_key,
        had_module_prefix=had_module_prefix,
        destination=args.output,
    )

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "control": "random_training_patch_prototypes",
        "seed": args.seed,
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(args.output),
        "training_root": str(args.train_root),
        "window_len": args.window_len,
        "stride": args.stride,
        "num_assignments": len(records),
        "assignments": sorted(records, key=lambda row: (row["stage"], row["prototype_index"])),
    }
    args.metadata.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved random-patch checkpoint: {args.output}")
    print(f"Saved assignment metadata: {args.metadata}")


if __name__ == "__main__":
    main()
