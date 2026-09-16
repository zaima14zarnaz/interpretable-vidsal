#!/usr/bin/env python3
"""Generate per-frame fixation and continuous saliency maps for Hollywood-2.

Expected layout
---------------
ROOT/{training,testing}/VIDEO/images/FRAME.png
ROOT/{training,testing}/VIDEO/s0_VIDEO.coord
ROOT/{training,testing}/VIDEO/s0_VIDEO.sacc

Outputs
-------
ROOT/{training,testing}/VIDEO/fixation/FRAME.png
ROOT/{training,testing}/VIDEO/maps/FRAME.png

The .coord files contain 1000-Hz gaze samples. Samples with insufficient
confidence or timestamps inside a .sacc interval are discarded. For each
subject and video frame, the remaining samples during that frame's exposure
are reduced to one representative location (median by default). Locations from
all available subjects form a binary fixation map. A Gaussian is placed at
each fixation and the summed image is normalized to [0, 255] to form the
continuous saliency map.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mpeg", ".mpg", ".mov", ".mkv"}
COMMON_FPS = np.asarray(
    [15.0, 23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CoordData:
    width: int
    height: int
    timestamps_us: np.ndarray
    x: np.ndarray
    y: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class VideoJob:
    video_dir: str
    video_file: str | None
    fps_arg: str
    min_confidence: float
    representative: str
    sigma: float
    kernel_size: int
    overwrite: bool
    dry_run: bool


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def parse_coord(path: Path) -> CoordData:
    width = height = None
    rows: list[tuple[int, float, float, float]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "gaze":
                if len(fields) != 3:
                    raise ValueError(f"{path}:{line_number}: malformed gaze header")
                width, height = int(fields[1]), int(fields[2])
                continue
            if fields[0] == "geometry":
                continue
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected four gaze-sample columns"
                )
            try:
                rows.append(
                    (int(fields[0]), float(fields[1]),
                     float(fields[2]), float(fields[3]))
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc

    if width is None or height is None:
        raise ValueError(f"{path}: missing 'gaze WIDTH HEIGHT' header")
    if not rows:
        raise ValueError(f"{path}: contains no gaze samples")

    values = np.asarray(rows, dtype=np.float64)
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    return CoordData(
        width=width,
        height=height,
        timestamps_us=values[:, 0].astype(np.int64),
        x=values[:, 1],
        y=values[:, 2],
        confidence=values[:, 3],
    )


def parse_saccades(path: Path) -> tuple[np.ndarray, np.ndarray]:
    intervals: list[tuple[int, int]] = []
    if not path.exists():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "saccades":
                continue
            if len(fields) != 6:
                raise ValueError(
                    f"{path}:{line_number}: expected six saccade columns"
                )
            try:
                onset, offset = int(fields[0]), int(fields[3])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid timestamp") from exc
            if offset < onset:
                raise ValueError(f"{path}:{line_number}: offset precedes onset")
            intervals.append((onset, offset))

    if not intervals:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    intervals.sort()
    values = np.asarray(intervals, dtype=np.int64)
    return values[:, 0], values[:, 1]


def samples_inside_saccades(
    timestamps_us: np.ndarray,
    onsets_us: np.ndarray,
    offsets_us: np.ndarray,
) -> np.ndarray:
    if onsets_us.size == 0:
        return np.zeros(timestamps_us.shape, dtype=bool)
    interval_index = np.searchsorted(onsets_us, timestamps_us, side="right") - 1
    has_previous = interval_index >= 0
    safe_index = np.maximum(interval_index, 0)
    return has_previous & (timestamps_us <= offsets_us[safe_index])


def representative_points(
    coord: CoordData,
    onsets_us: np.ndarray,
    offsets_us: np.ndarray,
    fps: float,
    frame_count: int,
    output_width: int,
    output_height: int,
    min_confidence: float,
    method: str,
) -> np.ndarray:
    in_saccade = samples_inside_saccades(
        coord.timestamps_us, onsets_us, offsets_us
    )
    valid = (
        (coord.confidence >= min_confidence)
        & ~in_saccade
        & np.isfinite(coord.x)
        & np.isfinite(coord.y)
        & (coord.x >= 0)
        & (coord.x < coord.width)
        & (coord.y >= 0)
        & (coord.y < coord.height)
    )

    timestamps = coord.timestamps_us[valid]
    xs = coord.x[valid]
    ys = coord.y[valid]
    frame_indices = np.floor(timestamps.astype(np.float64) * fps / 1_000_000.0)
    frame_indices = frame_indices.astype(np.int64)
    inside_video = (frame_indices >= 0) & (frame_indices < frame_count)
    timestamps = timestamps[inside_video]
    xs = xs[inside_video]
    ys = ys[inside_video]
    frame_indices = frame_indices[inside_video]

    output = np.full((frame_count, 2), np.nan, dtype=np.float32)
    if frame_indices.size == 0:
        return output

    boundaries = np.flatnonzero(np.diff(frame_indices)) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, frame_indices.size]

    scale_x = (output_width - 1) / max(coord.width - 1, 1)
    scale_y = (output_height - 1) / max(coord.height - 1, 1)

    for start, end in zip(starts, ends):
        frame_index = int(frame_indices[start])
        if method == "mean":
            x_value = float(np.mean(xs[start:end]))
            y_value = float(np.mean(ys[start:end]))
        elif method == "midpoint":
            midpoint_us = (frame_index + 0.5) * 1_000_000.0 / fps
            local = int(np.argmin(np.abs(timestamps[start:end] - midpoint_us)))
            x_value = float(xs[start + local])
            y_value = float(ys[start + local])
        else:
            x_value = float(np.median(xs[start:end]))
            y_value = float(np.median(ys[start:end]))
        output[frame_index] = (x_value * scale_x, y_value * scale_y)

    return output


def parse_fraction(value: str) -> float:
    value = value.strip()
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            raise ValueError("zero denominator")
        return float(numerator) / denominator_value
    return float(value)


def probe_fps(video_file: Path) -> float | None:
    for field in ("avg_frame_rate", "r_frame_rate"):
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", f"stream={field}",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_file),
        ]
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            fps = parse_fraction(result.stdout.splitlines()[0])
            if math.isfinite(fps) and fps > 0:
                return fps
        except (FileNotFoundError, subprocess.CalledProcessError,
                IndexError, ValueError):
            continue
    return None


def estimate_fps(coord_files: Sequence[Path], frame_count: int) -> tuple[float, float]:
    durations: list[float] = []
    for coord_file in coord_files:
        coord = parse_coord(coord_file)
        differences = np.diff(coord.timestamps_us)
        positive_differences = differences[differences > 0]
        sample_period = (
            float(np.median(positive_differences))
            if positive_differences.size else 1000.0
        )
        durations.append((float(coord.timestamps_us[-1]) + sample_period) / 1e6)
    duration = float(np.median(durations))
    if duration <= 0:
        raise ValueError("cannot estimate FPS from a non-positive gaze duration")
    raw_fps = frame_count / duration
    nearest = float(COMMON_FPS[np.argmin(np.abs(COMMON_FPS - raw_fps))])
    relative_error = abs(nearest - raw_fps) / nearest
    if relative_error > 0.05:
        raise ValueError(
            f"inferred FPS {raw_fps:.5f} is not near a common frame rate; "
            "supply --fps explicitly or provide the source videos with --video-root"
        )
    return nearest, raw_fps


def resolve_fps(
    fps_arg: str,
    video_file: Path | None,
    coord_files: Sequence[Path],
    frame_count: int,
) -> tuple[float, str]:
    if fps_arg.lower() != "auto":
        fps = float(fps_arg)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("--fps must be positive")
        return fps, "command line"
    if video_file is not None:
        fps = probe_fps(video_file)
        if fps is not None:
            return fps, f"ffprobe:{video_file.name}"
    fps, raw_fps = estimate_fps(coord_files, frame_count)
    return fps, f"gaze-duration estimate {raw_fps:.5f}, snapped"


def gaussian_kernel(
    sigma_x: float,
    sigma_y: float,
    kernel_size_x: int,
    kernel_size_y: int,
) -> np.ndarray:
    kernel_size_x = max(1, int(kernel_size_x))
    kernel_size_y = max(1, int(kernel_size_y))
    # Half-pixel centers for even sizes match fspecial('gaussian',[H W],sigma).
    x = np.arange(kernel_size_x, dtype=np.float32) - (kernel_size_x - 1) / 2.0
    y = np.arange(kernel_size_y, dtype=np.float32) - (kernel_size_y - 1) / 2.0
    kernel = np.exp(
        -0.5 * ((x[None, :] / sigma_x) ** 2 + (y[:, None] / sigma_y) ** 2)
    )
    return kernel.astype(np.float32)


def render_maps(
    points: np.ndarray,
    height: int,
    width: int,
    kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fixation = np.zeros((height, width), dtype=np.uint8)
    saliency = np.zeros((height, width), dtype=np.float32)
    kernel_height, kernel_width = kernel.shape
    # MATLAB imfilter's anchor for an even kernel is the lower-indexed of the
    # two central samples (index 74 for a 150-wide kernel).
    anchor_y = (kernel_height - 1) // 2
    anchor_x = (kernel_width - 1) // 2

    for x_float, y_float in points:
        if not (np.isfinite(x_float) and np.isfinite(y_float)):
            continue
        x = int(np.clip(np.rint(x_float), 0, width - 1))
        y = int(np.clip(np.rint(y_float), 0, height - 1))
        fixation[y, x] = 255

        full_x0 = x - anchor_x
        full_y0 = y - anchor_y
        image_x0 = max(0, full_x0)
        image_x1 = min(width, full_x0 + kernel_width)
        image_y0 = max(0, full_y0)
        image_y1 = min(height, full_y0 + kernel_height)
        kernel_x0 = image_x0 - full_x0
        kernel_x1 = kernel_x0 + (image_x1 - image_x0)
        kernel_y0 = image_y0 - full_y0
        kernel_y1 = kernel_y0 + (image_y1 - image_y0)
        saliency[image_y0:image_y1, image_x0:image_x1] += (
            kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1]
        )

    maximum = float(saliency.max())
    if maximum > 0:
        saliency_uint8 = np.rint(saliency * (255.0 / maximum)).astype(np.uint8)
    else:
        saliency_uint8 = np.zeros((height, width), dtype=np.uint8)
    return fixation, saliency_uint8


def matching_coord_files(video_dir: Path) -> list[Path]:
    expected_suffix = f"_{video_dir.name}.coord"
    matching = sorted(
        (path for path in video_dir.glob("*.coord")
         if path.name.endswith(expected_suffix)),
        key=natural_key,
    )
    if matching:
        return matching
    return sorted(video_dir.glob("*.coord"), key=natural_key)


def process_video(job: VideoJob) -> dict[str, object]:
    video_dir = Path(job.video_dir)
    image_dir = video_dir / "images"
    frames = sorted(
        (path for path in image_dir.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    if not frames:
        raise ValueError(f"{image_dir}: no frame images found")
    coord_files = matching_coord_files(video_dir)
    if not coord_files:
        raise ValueError(f"{video_dir}: no .coord files found")

    with Image.open(frames[0]) as first_frame:
        width, height = first_frame.size
    video_file = Path(job.video_file) if job.video_file else None
    fps, fps_source = resolve_fps(job.fps_arg, video_file, coord_files, len(frames))

    fixation_dir = video_dir / "fixation"
    maps_dir = video_dir / "maps"
    if job.dry_run:
        return {
            "video": video_dir.name, "frames": len(frames),
            "subjects": len(coord_files), "fps": fps,
            "fps_source": fps_source, "written": 0,
        }
    fixation_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    subject_points: list[np.ndarray] = []
    source_widths: list[int] = []
    source_heights: list[int] = []
    for coord_file in coord_files:
        coord = parse_coord(coord_file)
        sacc_file = coord_file.with_suffix(".sacc")
        if not sacc_file.exists():
            raise ValueError(
                f"{coord_file.name}: matching saccade file is missing: {sacc_file.name}"
            )
        onsets, offsets = parse_saccades(sacc_file)
        subject_points.append(
            representative_points(
                coord, onsets, offsets, fps, len(frames), width, height,
                job.min_confidence, job.representative,
            )
        )
        source_widths.append(coord.width)
        source_heights.append(coord.height)

    all_points = np.stack(subject_points, axis=0)  # subjects, frames, xy
    reference_width = float(np.median(source_widths))
    reference_height = float(np.median(source_heights))
    scale_x = width / reference_width
    scale_y = height / reference_height
    sigma_x = max(0.1, job.sigma * scale_x)
    sigma_y = max(0.1, job.sigma * scale_y)
    kernel = gaussian_kernel(
        sigma_x, sigma_y,
        max(3, int(round(job.kernel_size * scale_x))),
        max(3, int(round(job.kernel_size * scale_y))),
    )

    written = 0
    for frame_index, frame_path in enumerate(frames):
        output_name = f"{frame_path.stem}.png"
        fixation_path = fixation_dir / output_name
        map_path = maps_dir / output_name
        if not job.overwrite and fixation_path.exists() and map_path.exists():
            continue
        fixation, saliency = render_maps(
            all_points[:, frame_index, :], height, width, kernel
        )
        Image.fromarray(fixation, mode="L").save(fixation_path, optimize=False)
        Image.fromarray(saliency, mode="L").save(map_path, optimize=False)
        written += 1

    return {
        "video": video_dir.name,
        "frames": len(frames),
        "subjects": len(coord_files),
        "fps": fps,
        "fps_source": fps_source,
        "written": written,
    }


def build_video_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for current_root, directory_names, file_names in os.walk(root):
        directory_names[:] = [
            name for name in directory_names
            if name not in {"images", "maps", "fixation"}
        ]
        current = Path(current_root)
        for file_name in file_names:
            candidate = current / file_name
            if candidate.suffix.lower() in VIDEO_EXTENSIONS:
                index.setdefault(candidate.stem, candidate)
    return index


def discover_video_dirs(
    root: Path,
    splits: Iterable[str],
    only: set[str],
) -> list[Path]:
    directories: list[Path] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            print(f"WARNING: split directory not found: {split_dir}", file=sys.stderr)
            continue
        for candidate in sorted(split_dir.iterdir(), key=natural_key):
            if not candidate.is_dir() or not (candidate / "images").is_dir():
                continue
            if only and candidate.name not in only:
                continue
            directories.append(candidate)
    return directories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Hollywood-2 fixation and saliency maps from .coord/.sacc files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/quantization/zaima/videosal_datasets/hollywood2/videos"),
        help="dataset root containing training/ and testing/",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["training", "testing"],
        help="split directory names to process",
    )
    parser.add_argument(
        "--only", nargs="*", default=[], metavar="VIDEO",
        help="optional video-directory names to process",
    )
    parser.add_argument(
        "--video-root", type=Path,
        help="optional directory containing source videos for exact ffprobe FPS",
    )
    parser.add_argument(
        "--fps", default="auto",
        help="frame rate, or 'auto' (ffprobe when possible, otherwise infer and snap)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.75,
        help=("minimum .coord confidence (default: 0.75, allowing one valid eye; "
              "use 1.0 for fully valid binocular samples only)"),
    )
    parser.add_argument(
        "--representative", choices=["median", "mean", "midpoint"],
        default="median",
        help="reduce non-saccadic samples within a frame to one point per subject",
    )
    parser.add_argument(
        "--sigma", type=float, default=20.0,
        help="Gaussian sigma at the gaze-file resolution (default: 20 pixels)",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=150,
        help="Gaussian kernel size at gaze-file resolution (default: 150 pixels)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="number of videos processed concurrently (default: 4)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="regenerate frames whose map and fixation outputs already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate discovery and timing without writing images",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: dataset root does not exist: {root}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("ERROR: --workers must be at least 1", file=sys.stderr)
        return 2
    if args.sigma <= 0 or args.kernel_size <= 0:
        print("ERROR: --sigma and --kernel-size must be positive", file=sys.stderr)
        return 2

    video_dirs = discover_video_dirs(root, args.splits, set(args.only))
    if not video_dirs:
        print("ERROR: no matching video directories containing images/", file=sys.stderr)
        return 2

    index_root = args.video_root.expanduser().resolve() if args.video_root else root
    video_index = build_video_index(index_root)
    jobs = [
        VideoJob(
            video_dir=str(video_dir),
            video_file=str(video_index[video_dir.name])
            if video_dir.name in video_index else None,
            fps_arg=args.fps,
            min_confidence=args.min_confidence,
            representative=args.representative,
            sigma=args.sigma,
            kernel_size=args.kernel_size,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        for video_dir in video_dirs
    ]

    print(
        f"Found {len(jobs)} video directories; workers={args.workers}; "
        f"fps={args.fps}; dry_run={args.dry_run}"
    )
    completed = failed = total_written = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(process_video, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                completed += 1
                total_written += int(result["written"])
                print(
                    f"[{completed + failed}/{len(jobs)}] {result['video']}: "
                    f"frames={result['frames']}, subjects={result['subjects']}, "
                    f"fps={result['fps']:.5g} ({result['fps_source']}), "
                    f"written={result['written']}",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                print(f"FAILED {Path(job.video_dir).name}: {exc}", file=sys.stderr, flush=True)

    print(
        f"Finished: completed={completed}, failed={failed}, "
        f"output frame pairs written={total_written}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
