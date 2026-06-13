"""
Extract frames from AVI videos into per-video PNG directories.

Input layout:
    x/001.AVI
    x/002.AVI
    ...

Output layout:
    x/001/000000.png
    x/001/000001.png
    x/002/000000.png
    ...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


PathLike = Union[str, Path]
VIDEO_EXTENSIONS = {".avi"}


def _list_avi_videos(video_dir: Path) -> List[Path]:
    videos = [
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    videos.sort(key=lambda p: p.name.lower())
    return videos


def _save_frame_png(frame_bgr, out_path: Path) -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for video frame extraction. "
            "Install it with: pip install opencv-python"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame_bgr):
        raise RuntimeError(f"Failed to write frame image: {out_path}")


def extract_video_frames(
    video_path: PathLike,
    output_dir: PathLike,
    frame_name_width: int = 6,
) -> int:
    """
    Extract all frames from one AVI video into ``output_dir`` as PNG images.

    Args:
        video_path: path to a single .avi file.
        output_dir: directory where PNG frames are written.
        frame_name_width: zero-padding width for frame filenames.

    Returns:
        Number of frames written.
    """
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for video frame extraction. "
            "Install it with: pip install opencv-python"
        )

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_name = f"{frame_idx:0{frame_name_width}d}.png"
            _save_frame_png(frame_bgr, output_dir / frame_name)
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    return frame_idx


def convert_videos_to_frames(
    video_dir: PathLike,
    frame_name_width: int = 6,
    video_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Convert every AVI in ``video_dir`` into a frame sequence.

    For a video file ``x/vid_filename.AVI``, frames are saved under:
        x/vid_filename/000000.png
        x/vid_filename/000001.png
        ...

    Args:
        video_dir: directory containing only .avi videos.
        frame_name_width: zero-padding width for frame filenames.
        video_names: optional subset of video filenames to process.

    Returns:
        Mapping from video filename to number of extracted frames.
    """
    root = Path(video_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {root}")

    videos = _list_avi_videos(root)
    if video_names is not None:
        wanted = {name.lower() for name in video_names}
        videos = [p for p in videos if p.name.lower() in wanted]

    if not videos:
        raise RuntimeError(f"No .avi videos found in {root}")

    results: Dict[str, int] = {}
    for video_path in videos:
        vid_filename = video_path.stem
        output_dir = root / vid_filename / "images"
        n_frames = extract_video_frames(
            video_path=video_path,
            output_dir=output_dir,
            frame_name_width=frame_name_width,
        )
        results[video_path.name] = n_frames

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PNG frames from AVI videos in a directory."
    )
    parser.add_argument(
        "video_dir",
        type=str,
        help="Directory containing .avi videos.",
    )
    parser.add_argument(
        "--frame-name-width",
        type=int,
        default=6,
        help="Zero-padding width for frame filenames (default: 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = convert_videos_to_frames(
        video_dir=args.video_dir,
        frame_name_width=args.frame_name_width,
    )

    total_frames = sum(results.values())
    print(f"Converted {len(results)} videos ({total_frames} frames total).")
    for video_name, n_frames in results.items():
        print(f"  {video_name}: {n_frames} frames")


if __name__ == "__main__":
    main()
