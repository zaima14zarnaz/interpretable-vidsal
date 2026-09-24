#!/usr/bin/env python3
"""Create a control checkpoint with all visual prototype embeddings zeroed.

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
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20260921_220540/epoch_175.pth"
)
DEFAULT_OUTPUT = Path(
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/empty_prot_ckpt.pth"
)
PROTOTYPE_PATTERN = re.compile(
    r"^(?:module\.)?concept_creations\.([^.]+)\.visual_concepts$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace only the learned visual prototype banks with zero vectors."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=None)
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


def zero_prototypes_like(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(
            f"Prototype bank must have shape [K,D], got {tuple(tensor.shape)}"
        )
    return torch.zeros_like(tensor)


def prototype_difference(
    original: torch.Tensor,
    replacement: torch.Tensor,
) -> Dict[str, float]:
    """Summarize how far zeroed prototypes diverge from the originals."""
    orig = original.float()
    repl = replacement.float()
    delta = repl - orig
    cosine = torch.nn.functional.cosine_similarity(orig, repl, dim=-1)
    per_proto_l2 = delta.norm(dim=-1)
    return {
        "mean_cosine_to_original": float(cosine.mean().item()),
        "min_cosine_to_original": float(cosine.min().item()),
        "max_cosine_to_original": float(cosine.max().item()),
        "mean_l2_diff": float(per_proto_l2.mean().item()),
        "max_l2_diff": float(per_proto_l2.max().item()),
        "frobenius_diff": float(delta.norm().item()),
        "mean_abs_diff": float(delta.abs().mean().item()),
        "max_abs_diff": float(delta.abs().max().item()),
    }


def replace_prototypes(
    state: Mapping[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], list[Dict[str, Any]]]:
    output = dict(state)
    records: list[Dict[str, Any]] = []

    for key, original in state.items():
        match = PROTOTYPE_PATTERN.match(key)
        if match is None:
            continue
        replacement = zero_prototypes_like(original)
        output[key] = replacement
        records.append(
            {
                "stage": match.group(1),
                "state_dict_key": key,
                "shape": list(original.shape),
                "dtype": str(original.dtype),
                **prototype_difference(original, replacement),
            }
        )

    if not records:
        raise KeyError(
            "No concept_creations.<stage>.visual_concepts tensors were found "
            "in the checkpoint"
        )

    # Strict invariant: all non-prototype tensors must remain bit-identical.
    for key, original in state.items():
        if PROTOTYPE_PATTERN.match(key) is None and not torch.equal(
            output[key], original
        ):
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
    zeroed_state, records = replace_prototypes(state)
    output_checkpoint = build_output_checkpoint(
        checkpoint, container_key, zeroed_state
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(
            {
                "control": "zero_prototype_embeddings",
                "source_checkpoint": str(args.checkpoint),
                "output_checkpoint": str(args.output),
                "prototype_banks": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved empty-prototype checkpoint: {args.output}")
    print(f"Saved metadata: {args.metadata}")
    print("Prototype difference (original vs zeroed):")
    for record in records:
        print(
            f"  {record['stage']}: shape={record['shape']} "
            f"mean_cos={record['mean_cosine_to_original']:.4f} "
            f"(min={record['min_cosine_to_original']:.4f}, "
            f"max={record['max_cosine_to_original']:.4f}) "
            f"mean_l2={record['mean_l2_diff']:.4f} "
            f"frobenius={record['frobenius_diff']:.4f} "
            f"mean_abs={record['mean_abs_diff']:.6f} "
            f"max_abs={record['max_abs_diff']:.6f}"
        )


if __name__ == "__main__":
    main()
