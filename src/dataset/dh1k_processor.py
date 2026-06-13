"""Split DH1K raw videos into 80/20 training and testing directories."""

from __future__ import annotations

import os
import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from video_to_frame_converter import convert_videos_to_frames

DEFAULT_SOURCE_DIR = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/video"
)
DEFAULT_OUTPUT_DIR = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k"
)
DEFAULT_ANN_DIR = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/annotation"
)
DEFAULT_VID_DIR = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/training"
)
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FIXATION_DIR_NAMES = ("fixations", "fixation")
MAP_DIR_NAME = "maps"

def list_videos(source_dir: str | Path) -> List[Path]:
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {root}")

    videos = [
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda p: p.name)
    if not videos:
        raise RuntimeError(f"No video files found in {root}")
    return videos

def split_videos(
    videos: Sequence[Path],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    items = list(videos)
    rng = random.Random(seed)
    rng.shuffle(items)

    n_train = int(len(items) * train_ratio)
    if n_train <= 0 or n_train >= len(items):
        raise RuntimeError(
            f"Invalid split for {len(items)} videos with train_ratio={train_ratio}"
        )

    return items[:n_train], items[n_train:]

def _place_videos(videos: Iterable[Path], dest_dir: Path, method: str) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for src in videos:
        dst = dest_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if method == "symlink":
            dst.symlink_to(src.resolve())
        elif method == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown method: {method}")

        count += 1

    return count


def _list_video_ids(vid_dir: Path) -> List[str]:
    """Collect unique video ids from video files and subdirectories."""
    video_ids = set()
    for entry in vid_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            video_ids.add(entry.stem)
        elif entry.is_dir():
            video_ids.add(entry.name)
    return sorted(video_ids)


def _find_annotation_folder(ann_dir: Path, vid_filename: str) -> Path | None:
    """
    Resolve the annotation folder for a training video id.

    DH1K annotations use 4-digit ids (e.g. 0001) while training folders may
    use shorter stems (e.g. 001).
    """
    candidates = [
        ann_dir / vid_filename,
        ann_dir / vid_filename.zfill(4),
    ]
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir():
            return candidate
    return None


def _find_child_dir(parent: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        child = parent / name
        if child.is_dir():
            return child
    return None


def _move_dir(src: Path, dst: Path, replace: bool = True) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not replace:
            raise FileExistsError(f"Destination already exists: {dst}")
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    shutil.move(str(src), str(dst))


def attach_annotations_to_videos(
    ann_dir: str | Path = DEFAULT_ANN_DIR,
    vid_dir: str | Path = DEFAULT_VID_DIR,
    replace_existing: bool = True,
) -> dict:
    """
    Move fixation and saliency map folders from annotations into video dirs.

    For each ``vid_filename`` under ``vid_dir``, locate the matching folder in
    ``ann_dir`` and move:

        <ann_dir>/<vid_id>/fixation|fixations  -> <vid_dir>/<vid_filename>/fixation
        <ann_dir>/<vid_id>/maps                -> <vid_dir>/<vid_filename>/maps

    Args:
        ann_dir: root directory containing per-video annotation folders.
        vid_dir: root directory containing training videos / frame folders.
        replace_existing: overwrite existing fixation/maps folders in vid_dir.

    Returns:
        Summary dict with moved, skipped, and missing counts.
    """
    ann_root = Path(ann_dir)
    vid_root = Path(vid_dir)
    if not ann_root.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_root}")
    if not vid_root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {vid_root}")

    moved = 0
    skipped = 0
    missing_annotation = []
    missing_subdirs = []
    details = []

    for vid_filename in _list_video_ids(vid_root):
        ann_folder = _find_annotation_folder(ann_root, vid_filename)
        if ann_folder is None:
            missing_annotation.append(vid_filename)
            continue

        dest_video_dir = vid_root / vid_filename
        if not dest_video_dir.is_dir():
            dest_video_dir.mkdir(parents=True, exist_ok=True)

        moved_any = False
        for src_dir, dest_name, label in (
            (_find_child_dir(ann_folder, FIXATION_DIR_NAMES), "fixation", "fixation"),
            (_find_child_dir(ann_folder, (MAP_DIR_NAME,)), MAP_DIR_NAME, "maps"),
        ):
            if src_dir is None:
                missing_subdirs.append(f"{vid_filename}:{label}")
                continue

            dest_dir = dest_video_dir / dest_name
            if dest_dir.exists() and not replace_existing:
                skipped += 1
                details.append(
                    {
                        "vid_filename": vid_filename,
                        "kind": label,
                        "status": "skipped",
                        "source": str(src_dir),
                        "dest": str(dest_dir),
                    }
                )
                continue

            _move_dir(src_dir, dest_dir, replace=replace_existing)
            moved += 1
            moved_any = True
            details.append(
                {
                    "vid_filename": vid_filename,
                    "kind": label,
                    "status": "moved",
                    "source": str(src_dir),
                    "dest": str(dest_dir),
                }
            )

        if not moved_any:
            skipped += 1

    return {
        "ann_dir": str(ann_root.resolve()),
        "vid_dir": str(vid_root.resolve()),
        "video_count": len(_list_video_ids(vid_root)),
        "moved_dirs": moved,
        "skipped": skipped,
        "missing_annotation": missing_annotation,
        "missing_subdirs": missing_subdirs,
        "details": details,
    }


def modify_frame_name(stem: str) -> str:
    """
    Keep the last four characters of *stem*, parse as an integer, add one,
    and return a zero-padded 4-digit string.

    Example: ``"000000"`` -> ``"0001"`` (last four ``"0000"``, then +1).
    """
    if not stem:
        raise ValueError("Filename stem is empty")

    suffix = stem[-4:] if len(stem) >= 4 else stem.zfill(4)
    return f"{int(suffix) + 1:04d}"


def format_frame_name(
    folder: str | Path,
    *,
    dry_run: bool = False,
) -> dict:
    """
    Rename image files in *folder* using the last four stem digits + 1.

    Each file stem is truncated to its last four characters, incremented by
    one, and written back as a 4-digit name while preserving the extension.
    For example, ``000000.png`` becomes ``0001.png``.

    Renaming is done in two passes via temporary names so existing targets
    are not overwritten mid-run.

    Args:
        folder: Directory containing image files to rename.
        dry_run: If True, only report planned renames without changing files.

    Returns:
        Summary dict with ``folder``, ``renamed_count``, and ``renames`` list.
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    images = sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No image files found in {root}")

    planned: List[Tuple[Path, str]] = []
    target_names: dict[str, Path] = {}

    for src in images:
        new_stem = modify_frame_name(src.stem)
        new_name = f"{new_stem}{src.suffix.lower()}"
        if new_name in target_names and target_names[new_name] != src:
            raise ValueError(
                f"Collision: {src.name} and {target_names[new_name].name} "
                f"both map to {new_name}"
            )
        target_names[new_name] = src
        planned.append((src, new_name))

    renames = [
        {"from": src.name, "to": new_name}
        for src, new_name in planned
        if src.name != new_name
    ]

    if dry_run:
        return {
            "folder": str(root.resolve()),
            "renamed_count": len(renames),
            "renames": renames,
            "dry_run": True,
        }

    temp_suffix = ".__dh1k_rename_tmp__"
    temp_paths: List[Tuple[Path, Path]] = []

    for src, new_name in planned:
        if src.name == new_name:
            continue
        temp_path = root / f"{src.name}{temp_suffix}"
        if temp_path.exists():
            raise FileExistsError(f"Temporary rename path already exists: {temp_path}")
        src.rename(temp_path)
        temp_paths.append((temp_path, root / new_name))

    for temp_path, final_path in temp_paths:
        if final_path.exists():
            raise FileExistsError(
                f"Cannot rename {temp_path.name} -> {final_path.name}: "
                "destination already exists"
            )
        temp_path.rename(final_path)

    return {
        "folder": str(root.resolve()),
        "renamed_count": len(renames),
        "renames": renames,
        "dry_run": False,
    }


def split_dh1k_videos(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.8,
    seed: int = 42,
    method: str = "symlink",
) -> dict:
    videos = list_videos(source_dir)
    train_videos, test_videos = split_videos(
        videos, train_ratio=train_ratio, seed=seed
    )

    output_root = Path(output_dir)
    train_dir = output_root / "training"
    test_dir = output_root / "testing"

    n_train = _place_videos(train_videos, train_dir, method=method)
    n_test = _place_videos(test_videos, test_dir, method=method)

    return {
        "source_dir": str(Path(source_dir).resolve()),
        "output_dir": str(output_root.resolve()),
        "train_dir": str(train_dir.resolve()),
        "test_dir": str(test_dir.resolve()),
        "total_videos": len(videos),
        "train_count": n_train,
        "test_count": n_test,
        "train_ratio": train_ratio,
        "seed": seed,
        "method": method,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split DH1K videos into training/testing directories."
    )
    parser.add_argument("--source-dir", type=str, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    # summary = split_dh1k_videos(
    #     source_dir=args.source_dir,
    #     output_dir=args.output_dir,
    #     train_ratio=args.train_ratio,
    #     seed=args.seed,
    #     method=args.method,
    # )
    # convert_videos_to_frames(args.output_dir + "/training", frame_name_width=6)
    # convert_videos_to_frames(args.output_dir + "/testing", frame_name_width=6)

    # attach_annotations_to_videos(DEFAULT_ANN_DIR, args.output_dir + "/training")
    # attach_annotations_to_videos(DEFAULT_ANN_DIR, args.output_dir + "/testing")
    
    videos = os.listdir(args.output_dir + "/training")
    for video in videos:
        if os.path.isdir(args.output_dir + "/training/" + video):
            format_frame_name(args.output_dir + "/training/" + video + "/images")
    videos = os.listdir(args.output_dir + "/testing")
    for video in videos:
        if os.path.isdir(args.output_dir + "/testing/" + video):
            format_frame_name(args.output_dir + "/testing/" + video + "/images")
       

    # print("DH1K split complete.")
    # print(f"  source:      {summary['source_dir']}")
    # print(f"  training:    {summary['train_dir']} ({summary['train_count']} videos)")
    # print(f"  testing:     {summary['test_dir']} ({summary['test_count']} videos)")
    # print(f"  total:       {summary['total_videos']}")
    # print(f"  train_ratio: {summary['train_ratio']}")
    # print(f"  seed:        {summary['seed']}")
    # print(f"  method:      {summary['method']}")

if __name__ == "__main__":
    main()