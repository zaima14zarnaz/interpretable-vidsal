#!/usr/bin/env python3
"""Precompute frozen VideoSwin last-slice features for DHF1K windows.

Uses the independent frozen reference backbone from
``model.backbone_feature_cache`` (not the trainable prediction backbone).

Default enumeration matches training-eligible clip starts:
  DatasetLoader(..., window_len=32, stride=1, random_train_sampling=False)

Examples:
  # Dry-run estimate on a tiny subset
  python cache_backbone_features.py --split train --max-windows 8 --dry-run

  # Small pilot write + verification
  python cache_backbone_features.py --split train --max-windows 8 --device cuda:0

  # Full training-split build (large; do not launch casually)
  python cache_backbone_features.py --split train --device cuda:0 --storage-dtype float16
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from model.backbone_feature_cache import (
    CACHE_ROOT,
    DEFAULT_REFERENCE_CHECKPOINT,
    DEFAULT_RESIZE_TO,
    FrozenBackboneFeatureCache,
    STAGES,
    STAGE_CHANNELS,
    fingerprint_rgb_clip,
    make_identity_from_loader_sample,
)
from pre_process.dataloader import DatasetLoader

DEFAULT_DATASET_ROOT = Path("/data/quantization/zaima/videosal_datasets/dhf1k")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen VideoSwin last-slice features."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="DHF1K root containing train/ and val/.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split subdirectory under dataset-root.",
    )
    parser.add_argument("--window-len", type=int, default=32)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Deterministic window stride (default 1 covers training starts).",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=CACHE_ROOT,
        help="Feature cache root.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path(DEFAULT_REFERENCE_CHECKPOINT),
        help="Frozen reference VideoSwin weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="On-disk feature precision.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Loader batch size for enumeration (features still keyed per sample).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate storage and count windows without writing features.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite existing valid entries.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After writing/loading, compare against a fresh reference forward.",
    )
    return parser.parse_args()


def _split_dir(dataset_root: Path, split: str) -> Path:
    path = dataset_root.expanduser().resolve() / split
    if not path.is_dir():
        raise FileNotFoundError(f"Split directory not found: {path}")
    return path


def main() -> None:
    args = _parse_args()
    if args.window_len < 1 or args.stride < 1:
        raise SystemExit("window-len and stride must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be >= 1")

    split_dir = _split_dir(args.dataset_root, args.split)
    dataset = DatasetLoader(
        str(split_dir),
        window_len=args.window_len,
        stride=args.stride,
        random_train_sampling=False,
    )
    total_windows = len(dataset)
    limit = total_windows if args.max_windows is None else min(total_windows, args.max_windows)

    print(
        f"Dataset {split_dir} | windows={total_windows:,} | "
        f"processing={limit:,} | window_len={args.window_len} | stride={args.stride}",
        flush=True,
    )
    print(
        "Note: exhaustive stride-1 caching of all four stages is large "
        "(hundreds of GB possible in float32; float16 roughly halves that).",
        flush=True,
    )

    cache = FrozenBackboneFeatureCache(
        root=args.cache_root,
        checkpoint=str(args.reference_checkpoint),
        device=args.device,
        storage_dtype=args.storage_dtype,
        resize_to=DEFAULT_RESIZE_TO,
        normalize=True,
    )
    bytes_per = cache.estimate_entry_bytes()
    print(
        f"Reference fingerprint={cache.reference_fingerprint[:16]}… | "
        f"legacy_compatible={cache.legacy_reference_sha256 is not None} | "
        f"estimate≈{bytes_per / 1024**2:.2f} MiB / window "
        f"({args.storage_dtype})",
        flush=True,
    )
    print(
        f"Projected for {limit:,} windows ≈ "
        f"{limit * bytes_per / 1024**3:.2f} GiB (feature tensors only).",
        flush=True,
    )
    if args.dry_run:
        print("Dry-run complete; no features written.", flush=True)
        return

    # Enumerate deterministic windows one-by-one so each cache key uses the
    # exact DatasetLoader start/frame list (not a collated padded batch view).
    hits = misses = regenerated = verify_failures = 0
    t0 = time.time()
    for index in range(limit):
        video_id, start = dataset.windows[index]
        frame_names = dataset.video_frame_names[video_id][
            start : start + args.window_len
        ]
        sample = dataset[index]
        sample_video, rgb_np, _, _, n_frames = sample
        if sample_video != video_id:
            raise RuntimeError("Dataset window identity mismatch.")
        identity, rgb = make_identity_from_loader_sample(
            dataset_root=str(split_dir),
            video_id=video_id,
            window_start=start,
            frame_names=frame_names,
            rgb_thwc_or_tchw=torch.as_tensor(rgb_np),
            split=args.split,
            stride=args.stride,
            view_id="identity",
            preprocess=cache.preprocess,
        )
        if int(n_frames) != args.window_len or rgb.shape[0] != args.window_len:
            raise RuntimeError(
                f"Unexpected window length for {video_id}@{start}: "
                f"n_frames={n_frames}, rgb_t={rgb.shape[0]}"
            )

        existed = cache.entry_path(identity.content_sha256).is_file()
        before_hits = cache.stats["hits"]
        features = cache.get_or_compute(
            rgb, identity=identity, force=args.force
        )
        if cache.stats["hits"] > before_hits and not args.force:
            hits += 1
        else:
            misses += 1
            if args.force and existed:
                regenerated += 1

        if args.verify:
            fresh = cache.compute_last_slice_features(rgb)
            for stage in STAGES:
                if features[stage].shape != fresh[stage].shape:
                    verify_failures += 1
                    print(
                        f"VERIFY FAIL shape {video_id}@{start} {stage}: "
                        f"{tuple(features[stage].shape)} vs {tuple(fresh[stage].shape)}",
                        flush=True,
                    )
                    break
                if not torch.allclose(
                    features[stage].float(),
                    fresh[stage].float(),
                    rtol=1e-3 if args.storage_dtype == "float16" else 1e-5,
                    atol=1e-3 if args.storage_dtype == "float16" else 1e-5,
                ):
                    verify_failures += 1
                    print(
                        f"VERIFY FAIL values {video_id}@{start} {stage}",
                        flush=True,
                    )
                    break
            # Stable key / last-slice checks
            again = fingerprint_rgb_clip(rgb)
            if again != identity.content_sha256:
                verify_failures += 1
                print(f"VERIFY FAIL unstable key {video_id}@{start}", flush=True)
            for stage in STAGES:
                if features[stage].shape[0] != STAGE_CHANNELS[stage]:
                    verify_failures += 1
                if features[stage].shape[1] != 1:
                    verify_failures += 1
                    print(
                        f"VERIFY FAIL temporal slice {stage} T="
                        f"{features[stage].shape[1]}",
                        flush=True,
                    )

            # Distinct keys for a flipped view
            flipped = torch.flip(rgb, dims=[-1])
            flip_id, flip_rgb = make_identity_from_loader_sample(
                dataset_root=str(split_dir),
                video_id=video_id,
                window_start=start,
                frame_names=frame_names,
                rgb_thwc_or_tchw=flipped.permute(0, 2, 3, 1),
                split=args.split,
                stride=args.stride,
                view_id="hflip",
                preprocess=cache.preprocess,
            )
            if flip_id.content_sha256 == identity.content_sha256:
                verify_failures += 1
                print(
                    f"VERIFY FAIL flip collision {video_id}@{start}",
                    flush=True,
                )
            else:
                cache.get_or_compute(flip_rgb, identity=flip_id, force=False)

        if (index + 1) % 10 == 0 or (index + 1) == limit:
            elapsed = time.time() - t0
            print(
                f"[{index + 1:,}/{limit:,}] hits={hits} misses={misses} "
                f"regenerated={regenerated} writes={cache.stats['writes']} "
                f"rejected={cache.stats['rejected']} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    # Second pass over the same subset: valid entries should be skipped.
    second_hits = 0
    for index in range(limit):
        video_id, start = dataset.windows[index]
        frame_names = dataset.video_frame_names[video_id][
            start : start + args.window_len
        ]
        _, rgb_np, _, _, _ = dataset[index]
        identity, rgb = make_identity_from_loader_sample(
            dataset_root=str(split_dir),
            video_id=video_id,
            window_start=start,
            frame_names=frame_names,
            rgb_thwc_or_tchw=torch.as_tensor(rgb_np),
            split=args.split,
            stride=args.stride,
            view_id="identity",
            preprocess=cache.preprocess,
        )
        before = cache.stats["hits"]
        cache.get_or_compute(rgb, identity=identity, force=False)
        if cache.stats["hits"] > before:
            second_hits += 1

    print(
        f"Done. first_pass hits={hits} misses={misses} regenerated={regenerated} | "
        f"second_pass hits={second_hits}/{limit} | "
        f"verify_failures={verify_failures} | cache_stats={cache.stats}",
        flush=True,
    )
    if args.verify and verify_failures:
        raise SystemExit(f"Verification failed on {verify_failures} checks.")
    if second_hits != limit:
        raise SystemExit(
            f"Expected all {limit} entries to hit on second pass; got {second_hits}."
        )


if __name__ == "__main__":
    main()
