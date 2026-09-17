#!/usr/bin/env python3
"""Create a control checkpoint with random normalized prototype vectors.

Only ``concept_creations.<stage>.visual_concepts`` tensors are replaced.  Every
other model tensor is copied unchanged from the source checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


DEFAULT_CHECKPOINT = Path(
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/best_dhf1k.pth"
)
DEFAULT_OUTPUT = Path(
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/random_prot_ckpt.pth"
)
PROTOTYPE_PATTERN = re.compile(
    r"^(?:module\.)?concept_creations\.([^.]+)\.visual_concepts$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace only the learned visual prototype banks with random unit vectors."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_state_dict(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    )


def extract_state_dict(
    checkpoint: Any,
) -> Tuple[Mapping[str, torch.Tensor], Optional[str]]:
    if is_state_dict(checkpoint):
        return checkpoint, None
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "network", "net"):
            value = checkpoint.get(key)
            if is_state_dict(value):
                return value, key
    raise TypeError(
        "Could not find a model state_dict. Expected a raw state_dict or one of "
        "model_state_dict/state_dict/model/network/net."
    )


def random_unit_vectors_like(
    tensor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"Prototype bank must have shape [K,D], got {tuple(tensor.shape)}")
    vectors = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    return vectors.to(dtype=tensor.dtype)


def replace_prototypes(
    state: Mapping[str, torch.Tensor],
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], list[Dict[str, Any]]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    output = dict(state)
    records: list[Dict[str, Any]] = []

    for key, original in state.items():
        match = PROTOTYPE_PATTERN.match(key)
        if match is None:
            continue
        replacement = random_unit_vectors_like(original, generator)
        output[key] = replacement
        records.append(
            {
                "stage": match.group(1),
                "state_dict_key": key,
                "shape": list(original.shape),
                "dtype": str(original.dtype),
                "mean_cosine_to_original": float(
                    torch.nn.functional.cosine_similarity(
                        original.float(), replacement.float(), dim=-1
                    ).mean().item()
                ),
            }
        )

    if not records:
        raise KeyError(
            "No concept_creations.<stage>.visual_concepts tensors were found in the checkpoint"
        )

    # Strict invariant: all non-prototype tensors must remain bit-identical.
    for key, original in state.items():
        if PROTOTYPE_PATTERN.match(key) is None and not torch.equal(output[key], original):
            raise RuntimeError(f"Unexpected non-prototype tensor change: {key}")
    return output, records


def build_output_checkpoint(
    checkpoint: Any,
    container_key: Optional[str],
    state: Mapping[str, torch.Tensor],
) -> Any:
    if container_key is None:
        return dict(state)
    payload = dict(checkpoint)
    payload[container_key] = dict(state)
    # Optimizer/scaler moments no longer correspond to the altered parameters.
    for key in ("optimizer_state_dict", "optimizer", "scaler_state_dict", "scaler"):
        payload.pop(key, None)
    return payload


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.metadata is None:
        args.metadata = args.output.with_suffix(".metadata.json")
    else:
        args.metadata = args.metadata.expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {args.checkpoint}")

    checkpoint = torch_load(args.checkpoint)
    state, container_key = extract_state_dict(checkpoint)
    randomized_state, records = replace_prototypes(state, args.seed)
    output_checkpoint = build_output_checkpoint(checkpoint, container_key, randomized_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            {
                "control": "random_normalized_prototype_vectors",
                "seed": args.seed,
                "source_checkpoint": str(args.checkpoint),
                "output_checkpoint": str(args.output),
                "prototype_banks": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved random-prototype checkpoint: {args.output}")
    print(f"Saved metadata: {args.metadata}")


if __name__ == "__main__":
    main()
