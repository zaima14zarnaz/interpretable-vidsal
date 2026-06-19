"""Redistribute DH1K processed videos into fixed train/test splits by video id."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

DEFAULT_DH1K_ROOT = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k"
)
DEFAULT_TRAIN_DIR = f"{DEFAULT_DH1K_ROOT}/training"
DEFAULT_TEST_DIR = f"{DEFAULT_DH1K_ROOT}/testing"
DEFAULT_EXCLUDED_DIR = f"{DEFAULT_DH1K_ROOT}/excluded"
DEFAULT_STAGING_DIR = f"{DEFAULT_DH1K_ROOT}/_resplit_staging"

TRAIN_ID_MIN = 1
TRAIN_ID_MAX = 600
TEST_ID_MIN = 601
TEST_ID_MAX = 700


def parse_video_id(entry: Path) -> Optional[int]:
    """Return numeric video id from a directory or video file name."""
    stem = entry.stem if entry.suffix.lower() in VIDEO_EXTENSIONS else entry.name
    if stem.isdigit():
        return int(stem)
    return None


def list_split_entries(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    entries = [
        p
        for p in root.iterdir()
        if p.name.startswith(".") is False and parse_video_id(p) is not None
    ]
    entries.sort(key=lambda p: (parse_video_id(p), p.name))
    return entries


def destination_for_id(
    video_id: int,
    train_dir: Path,
    test_dir: Path,
    excluded_dir: Path,
) -> Path:
    if TRAIN_ID_MIN <= video_id <= TRAIN_ID_MAX:
        return train_dir
    if TEST_ID_MIN <= video_id <= TEST_ID_MAX:
        return test_dir
    return excluded_dir


def _move_entry(src: Path, dst: Path, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Destination already exists: {dst}")
    if dry_run:
        print(f"[dry-run] move {src} -> {dst}")
        return
    shutil.move(str(src), str(dst))


def collect_entries(
    source_roots: Iterable[Path],
) -> Dict[int, List[Path]]:
    """Group filesystem entries by numeric video id across source roots."""
    by_id: Dict[int, List[Path]] = {}
    for root in source_roots:
        for entry in list_split_entries(root):
            video_id = parse_video_id(entry)
            if video_id is None:
                continue
            by_id.setdefault(video_id, []).append(entry)
    return by_id


def stage_all_entries(
    source_roots: Iterable[Path],
    staging_dir: Path,
    dry_run: bool,
) -> Dict[int, List[Path]]:
    """Move every train/test entry into a flat staging directory."""
    if staging_dir.exists():
        if any(staging_dir.iterdir()):
            raise RuntimeError(
                f"Staging directory is not empty: {staging_dir}. "
                "Remove it or choose another --staging-dir before re-running."
            )
    else:
        if not dry_run:
            staging_dir.mkdir(parents=True, exist_ok=True)

    staged_by_id: Dict[int, List[Path]] = {}
    for root in source_roots:
        for entry in list_split_entries(root):
            video_id = parse_video_id(entry)
            if video_id is None:
                continue
            staged_path = staging_dir / entry.name
            _move_entry(entry, staged_path, dry_run=dry_run)
            staged_by_id.setdefault(video_id, []).append(
                staged_path if not dry_run else entry
            )
    return staged_by_id


def distribute_from_staging(
    staging_dir: Path,
    train_dir: Path,
    test_dir: Path,
    excluded_dir: Path,
    dry_run: bool,
) -> Tuple[dict, dict]:
    """Move staged entries into train/test/excluded destinations."""
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    excluded_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "test": 0, "excluded": 0, "entries": 0}
    by_id: Dict[int, List[str]] = {"train": [], "test": [], "excluded": []}

    for entry in list_split_entries(staging_dir):
        video_id = parse_video_id(entry)
        if video_id is None:
            continue

        if TRAIN_ID_MIN <= video_id <= TRAIN_ID_MAX:
            split = "train"
            dest_root = train_dir
        elif TEST_ID_MIN <= video_id <= TEST_ID_MAX:
            split = "test"
            dest_root = test_dir
        else:
            split = "excluded"
            dest_root = excluded_dir

        dst = dest_root / entry.name
        _move_entry(entry, dst, dry_run=dry_run)
        counts[split] += 1
        counts["entries"] += 1
        if video_id not in by_id[split]:
            by_id[split].append(video_id)

    for key in by_id:
        by_id[key] = sorted(by_id[key])

    return counts, by_id


def redistribute_dh1k_split(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    test_dir: str | Path = DEFAULT_TEST_DIR,
    excluded_dir: str | Path = DEFAULT_EXCLUDED_DIR,
    staging_dir: str | Path = DEFAULT_STAGING_DIR,
    dry_run: bool = False,
) -> dict:
    """
    Collect all entries from current train/test folders and redistribute:

      train: ids 001-600
      test:  ids 601-700
      excluded: ids outside that range (e.g. 701-1000)
    """
    train_root = Path(train_dir)
    test_root = Path(test_dir)
    excluded_root = Path(excluded_dir)
    staging_root = Path(staging_dir)

    before = collect_entries([train_root, test_root])
    print(
        f"Found {sum(len(v) for v in before.values())} entries "
        f"across {len(before)} unique video ids in train+test."
    )

    if dry_run:
        counts = {"train": 0, "test": 0, "excluded": 0, "entries": 0}
        video_ids = {"train": set(), "test": set(), "excluded": set()}
        for video_id, entries in sorted(before.items()):
            dest_root = destination_for_id(
                video_id, train_root, test_root, excluded_root
            )
            if dest_root == train_root:
                split = "train"
            elif dest_root == test_root:
                split = "test"
            else:
                split = "excluded"
            for entry in entries:
                dst = dest_root / entry.name
                print(f"[dry-run] move {entry} -> {dst}")
                counts[split] += 1
                counts["entries"] += 1
            video_ids[split].add(video_id)
        return {
            "train_entries": counts["train"],
            "test_entries": counts["test"],
            "excluded_entries": counts["excluded"],
            "total_entries_moved": counts["entries"],
            "train_video_ids": len(video_ids["train"]),
            "test_video_ids": len(video_ids["test"]),
            "excluded_video_ids": len(video_ids["excluded"]),
            "train_dir": str(train_root),
            "test_dir": str(test_root),
            "excluded_dir": str(excluded_root),
        }

    stage_all_entries([train_root, test_root], staging_root, dry_run=False)
    counts, by_id = distribute_from_staging(
        staging_root,
        train_root,
        test_root,
        excluded_root,
        dry_run=False,
    )

    if staging_root.exists() and not any(staging_root.iterdir()):
        staging_root.rmdir()

    return {
        "train_entries": counts["train"],
        "test_entries": counts["test"],
        "excluded_entries": counts["excluded"],
        "total_entries_moved": counts["entries"],
        "train_video_ids": len(by_id["train"]),
        "test_video_ids": len(by_id["test"]),
        "excluded_video_ids": len(by_id["excluded"]),
        "train_dir": str(train_root),
        "test_dir": str(test_root),
        "excluded_dir": str(excluded_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Redistribute DH1K processed videos: train ids 001-600, "
            "test ids 601-700."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help="Destination directory for training videos.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=DEFAULT_TEST_DIR,
        help="Destination directory for testing videos.",
    )
    parser.add_argument(
        "--excluded-dir",
        type=str,
        default=DEFAULT_EXCLUDED_DIR,
        help="Directory for ids outside 001-700.",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=DEFAULT_STAGING_DIR,
        help="Temporary staging directory used during the move.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying the filesystem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = redistribute_dh1k_split(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        excluded_dir=args.excluded_dir,
        staging_dir=args.staging_dir,
        dry_run=args.dry_run,
    )

    print("\n=== DH1K split redistribution ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        print("\nDry run only. Re-run without --dry-run to apply moves.")


if __name__ == "__main__":
    main()
