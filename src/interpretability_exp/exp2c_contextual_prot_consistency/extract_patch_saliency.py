#!/usr/bin/env python3
"""Regenerate patch_pred_saliency and patch_gt_saliency in patch_data.json from a checkpoint.

Uses the same final-map + adaptive pool path as repair_patch_saliency.py and
find_patterns.py. Prototype / top-20 fields are unchanged. After updating JSON,
delete or invalidate contextual_prot_consistency/patch_archive.bin and .meta so
generate_candidate_pairs re-ingests saliency bands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, TypeVar

T = TypeVar("T")

_TQDM_BAR = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_SRC = _SCRIPT_DIR.parents[1]  # .../src
_DEFAULT_PATCH_DATA = Path(
    "/data/quantization/zaima/videosal_datasets/dhf1k/prototype_consistency_sal/patch_data.json"
)
_DEFAULT_VAL = Path("/data/quantization/zaima/videosal_datasets/dhf1k/val")


def tqdm_wrap(iterable: Iterable[T], desc: str, *, total: Optional[int] = None,
              disable: bool = False) -> Iterable[T]:
    if disable:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        unit="patch",
        total=total,
        dynamic_ncols=True,
        smoothing=0.05,
        bar_format=_TQDM_BAR,
    )


def expected_patch_total(patch_data: Path) -> Optional[int]:
    meta = patch_data.parent / "experiment_metadata.json"
    if not meta.is_file():
        return None
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        count = payload.get("collection", {}).get("patch_count")
        return int(count) if count else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--patch-data", type=Path, default=_DEFAULT_PATCH_DATA,
                        help=f"Input JSON array (default: {_DEFAULT_PATCH_DATA}).")
    parser.add_argument("--src-dir", type=Path, default=_DEFAULT_SRC,
                        help=f"ExplainableSaliency src root (default: {_DEFAULT_SRC}).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path,
                       help="Write here instead of input (default: <stem>_corrected.json beside input).")
    group.add_argument("--in-place", action="store_true",
                       help="Replace patch-data atomically; creates patch_data.json.bak first.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing --output file.")
    parser.add_argument("--evaluation-module", default="evaluation")
    parser.add_argument("--val-dataset-dir", type=Path, default=_DEFAULT_VAL)
    parser.add_argument("--window-len", type=int, default=None,
                        help="Temporal window length; default evaluation.WINDOW_LEN.")
    parser.add_argument("--device", help="e.g. cuda:0 or cpu")
    parser.add_argument("--progress-every", type=int, default=100_000,
                        help="Periodic logs when --no-tqdm; ignored when tqdm is active.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args(argv)
    if args.progress_every < 0 or (args.window_len is not None and args.window_len < 1):
        parser.error("--progress-every must be nonnegative and --window-len positive when set.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    import repair_patch_saliency as repair

    repair_args = argparse.Namespace(
        patch_data=args.patch_data,
        output=args.output,
        in_place=args.in_place,
        overwrite=args.overwrite,
        checkpoint=args.checkpoint,
        src_dir=args.src_dir,
        evaluation_module=args.evaluation_module,
        val_dataset_dir=args.val_dataset_dir,
        window_len=args.window_len,
        device=args.device,
        progress_every=args.progress_every,
        frame_index_mode="index0",
    )
    source = repair_args.patch_data.expanduser().resolve()
    if not source.is_file():
        raise ValueError("Input not found: {}.".format(source))
    destination = (
        source if repair_args.in_place
        else (repair_args.output.expanduser().resolve() if repair_args.output
              else source.with_name(source.stem + "_corrected.json"))
    )
    if destination == source and not repair_args.in_place:
        raise ValueError("Use --in-place to replace the input file.")
    print("Checkpoint: {}".format(repair_args.checkpoint), flush=True)
    print("Patch data: {}".format(source), flush=True)
    print("Output: {}".format(destination), flush=True)
    scorer = repair.FinalMapScorer(repair_args)
    patch_total = expected_patch_total(source)
    progress_every = repair_args.progress_every if args.no_tqdm else 0

    def progress_iter(records: Iterable[dict]) -> Iterable[dict]:
        return tqdm_wrap(records, "Regenerate patch saliency", total=patch_total, disable=args.no_tqdm)

    repair.repair_file(source, destination, scorer, repair_args.overwrite,
                       repair_args.in_place, progress_every, progress_iter=progress_iter)
    print("Done. If bands or archives used old saliency, remove patch_archive.bin and .meta, then re-run exp2c ingest.",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit("Error: {}".format(exc))
