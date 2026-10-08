#!/usr/bin/env python3
"""Stream a single signature-pair CSV; discover ordering and confirm on disjoint videos.
No witness rows are held in memory. Memory scales with distinct frames. Split
membership is printed so it can be frozen and reused for other signature pairs.
Only videos represented in this CSV can be split. For multiple pairs, supply a
common --discovery-videos list chosen before inspecting their results.
"""
from __future__ import annotations
import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
def progress(rows, description, disabled):
    if disabled:
        return rows
    try:
        from tqdm import tqdm
    except ImportError:
        return rows
    return tqdm(rows, desc=description, unit="row", dynamic_ncols=True)
@dataclass
class Moments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
    @property
    def std(self):
        return math.sqrt(max(0.0, self.m2 / (self.n - 1))) if self.n > 1 else 0.0
@dataclass
class Signal:
    a_higher: int = 0
    b_higher: int = 0
    ties: int = 0
    differences: Moments = field(default_factory=Moments)
    def add(self, a, b, epsilon):
        difference = a - b
        self.differences.add(difference)
        if difference > epsilon:
            self.a_higher += 1
        elif difference < -epsilon:
            self.b_higher += 1
        else:
            self.ties += 1
    def rate(self, direction="B", exclude_ties=False):
        denominator = self.a_higher + self.b_higher if exclude_ties else self.differences.n
        numerator = self.b_higher if direction == "B" else self.a_higher
        return numerator / denominator if denominator else math.nan
@dataclass
class Counts:
    pred: Signal = field(default_factory=Signal)
    gt: Signal = field(default_factory=Signal)
    def add(self, values, epsilon):
        pa, pb, ga, gb = values
        self.pred.add(pa, pb, epsilon)
        self.gt.add(ga, gb, epsilon)
def average(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.mean(values) if values else math.nan
def percent(value):
    return "N/A" if not math.isfinite(value) else f"{100 * value:.4f}%"
def resolve_frame_data_path(input_dir):
    path = input_dir.expanduser().resolve()
    if path.is_dir():
        path /= "frame_data.csv"
    if not path.is_file():
        raise ValueError(f"CSV not found: {path}")
    return path
def resolve_column(names, supplied, candidates, kind):
    if supplied:
        if supplied not in names:
            raise ValueError(f"Missing {kind} column {supplied!r}.")
        return supplied
    matches = [c for c in candidates if c in names]
    if len(matches) != 1:
        raise ValueError(f"Specify --{kind}-field. Available columns: {', '.join(names)}")
    return matches[0]
def identity(row, name, line):
    value = row.get(name)
    if value is None or not value.strip():
        raise ValueError(f"CSV line {line}: missing {name}.")
    return value.strip()
def number(row, name, line):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"CSV line {line}: invalid {name}: {row.get(name)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"CSV line {line}: {name} must be finite.")
    return value
def inspect_videos(path, args):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        required = {"pred_sal_a", "pred_sal_b", "gt_sal_a", "gt_sal_b"}
        if not required.issubset(names):
            raise ValueError(f"Missing saliency columns: {', '.join(sorted(required - set(names)))}")
        video = resolve_column(names, args.video_field,
                               ("video_fname", "video_id", "video_name", "video", "video_filename"), "video")
        frame = resolve_column(names, args.frame_field,
                               ("frame_id", "frame_idx", "absolute_frame_index", "frame_filename", "frame_no", "frame_index", "frame"), "frame")
        videos = set()
        for row in progress(reader, "Find videos", args.no_tqdm):
            videos.add(identity(row, video, reader.line_num))
            identity(row, frame, reader.line_num)
    return videos, video, frame
def split_videos(videos, args):
    if len(videos) < 2:
        raise ValueError("At least two videos are needed for disjoint discovery and confirmation subsets.")
    if args.discovery_videos:
        discovery = set(args.discovery_videos)
        # A reusable global split may contain videos absent from this pair CSV.
        discovery &= videos
    else:
        ordered = sorted(videos)
        random.Random(args.split_seed).shuffle(ordered)
        n = min(len(ordered) - 1, max(1, int(len(ordered) * args.discovery_fraction + 0.5)))
        discovery = set(ordered[:n])
    confirmation = videos - discovery
    if not discovery or not confirmation:
        raise ValueError("Both subsets must contain at least one video represented in this CSV.")
    return discovery, confirmation
def load_reports(path, args, video_field, frame_field, discovery):
    totals = {name: Counts() for name in ("discovery", "confirmation")}
    frames = {name: {} for name in totals}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in progress(reader, "Analyze patch pairs", args.no_tqdm):
            video = identity(row, video_field, reader.line_num)
            frame = identity(row, frame_field, reader.line_num)
            subset = "discovery" if video in discovery else "confirmation"
            values = tuple(number(row, name, reader.line_num) for name in
                           ("pred_sal_a", "pred_sal_b", "gt_sal_a", "gt_sal_b"))
            totals[subset].add(values, args.tie_epsilon)
            counts = frames[subset].setdefault((video, frame), Counts())
            counts.add(values, args.tie_epsilon)
    return totals, frames
def macro_rate(frames, signal, direction="B", exclude_ties=False):
    return average(getattr(c, signal).rate(direction, exclude_ties) for c in frames.values())
def video_rates(frames, signal, direction="B", exclude_ties=False):
    by_video = defaultdict(list)
    for (video, _), counts in frames.items():
        by_video[video].append(getattr(counts, signal).rate(direction, exclude_ties))
    return {video: average(rates) for video, rates in by_video.items()}
def print_report(label, total, frames, direction, args):
    print(f"\n{'=' * 72}\n{label.upper()}")
    print(f"Witness rows (patch pairs): {total.pred.differences.n:,}")
    print(f"Distinct frames: {len(frames):,}; distinct videos: {len({v for v, _ in frames}):,}")
    print("\nPatch B saliency greater than patch A:")
    for title, signal in (("Predicted", total.pred), ("Ground truth", total.gt)):
        print(f"  {title}: {signal.b_higher:,} / {signal.differences.n:,} ({percent(signal.rate())})")
        print(f"    A higher: {signal.a_higher:,}; ties: {signal.ties:,}; B higher among non-ties: {percent(signal.rate(exclude_ties=True))}")
    print("\nDifference (patch A minus patch B) per row; sample standard deviation:")
    for title, signal in (("Predicted", total.pred), ("Ground truth", total.gt)):
        print(f"  {title}: mean {signal.differences.mean:.8g}; std {signal.differences.std:.8g}")
    print("  Positive mean => patch A higher on average.")
    print("\nConsistency with equal weight per frame and per video:")
    for title, signal in (("Predicted", "pred"), ("Ground truth", "gt")):
        print(f"  {title} B-higher rate: frame mean {percent(macro_rate(frames, signal))}; "
              f"video mean {percent(average(video_rates(frames, signal).values()))}")
        print(f"    Among non-ties: frame mean {percent(macro_rate(frames, signal, exclude_ties=True))}; "
              f"video mean {percent(average(video_rates(frames, signal, exclude_ties=True).values()))}")
    print(f"\nFrozen predicted ordering: patch {direction} higher (selected using discovery frames only).")
    frame_rates = [c.pred.rate(direction, True) for c in frames.values()]
    usable = [r for r in frame_rates if math.isfinite(r)]
    videos = video_rates(frames, "pred", direction, True)
    usable_videos = [r for r in videos.values() if math.isfinite(r)]
    print(f"  Pooled agreement among non-ties: {percent(total.pred.rate(direction, True))}")
    print(f"  Frame mean agreement among non-ties: {percent(average(usable))}")
    print(f"  Video mean agreement (each video averages its frames): {percent(average(usable_videos))}")
    print(f"  Frames with >= {100 * args.consistency_threshold:g}% agreement: "
          f"{sum(r >= args.consistency_threshold for r in usable):,} / {len(usable):,}; "
          f"all-tied frames excluded: {len(frame_rates) - len(usable):,}")
    print(f"  Videos with >= {100 * args.consistency_threshold:g}% mean agreement: "
          f"{sum(r >= args.consistency_threshold for r in usable_videos):,} / {len(usable_videos):,}; "
          f"all-tied videos excluded: {len(videos) - len(usable_videos):,}")
    print("\nPer-video statistics (frame rates exclude ties):")
    print("  video | frames | rows | pred B>A pooled | GT B>A pooled | pred frozen-order frame mean | GT B>A frame mean | pred diff mean/std | GT diff mean/std")
    grouped = defaultdict(list)
    for (video, _), counts in frames.items():
        grouped[video].append(counts)
    for video in sorted(grouped):
        counts = grouped[video]
        pooled = Counts()
        for c in counts:
            for name in ("pred", "gt"):
                source, target = getattr(c, name), getattr(pooled, name)
                target.a_higher += source.a_higher
                target.b_higher += source.b_higher
                target.ties += source.ties
                merge_moments(target.differences, source.differences)
        print(f"  {video} | {len(counts):,} | {pooled.pred.differences.n:,} | "
              f"{percent(pooled.pred.rate())} | {percent(pooled.gt.rate())} | {percent(videos[video])} | "
              f"{percent(average(c.gt.rate(exclude_ties=True) for c in counts))} | "
              f"{pooled.pred.differences.mean:.8g}/{pooled.pred.differences.std:.8g} | "
              f"{pooled.gt.differences.mean:.8g}/{pooled.gt.differences.std:.8g}")
def merge_moments(target, source):
    if not source.n:
        return
    n = target.n + source.n
    delta = source.mean - target.mean
    target.m2 += source.m2 + delta * delta * target.n * source.n / n
    target.mean += delta * source.n / n
    target.n = n
def contact_sheet_video_ids(videos, args):
    """Sorted CSV video IDs for contact sheets: [start, start + count)."""
    ordered = sorted(videos)
    start = args.contact_video_start
    count = args.contact_video_count
    if start < 0:
        raise ValueError("--contact-video-start must be >= 0.")
    if count < 1:
        raise ValueError("--contact-video-count must be >= 1.")
    if start >= len(ordered):
        raise ValueError("--contact-video-start {:,} out of range; CSV has {:,} videos (0..{:,})."
                         .format(start, len(ordered), max(0, len(ordered) - 1)))
    chosen = ordered[start:start + count]
    if len(chosen) < count:
        print("Contact sheet: only {:,} videos from index {:,} (requested {:,}); using {}."
              .format(len(chosen), start, count, ", ".join(chosen)), flush=True)
    return chosen


def choose_contact_frames(frames, videos, count, contact_videos):
    """Evenly spaced eligible frames in each contact-sheet video."""
    selected = set()
    for video in contact_videos:
        available = {frame for group in frames.values() for v, frame in group if v == video}
        ordered = sorted(available, key=lambda x: (0, int(x)) if x.isdecimal() else (1, x))
        n = min(count, len(ordered))
        indices = [len(ordered) // 2] if n == 1 else [round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)]
        selected.update((video, ordered[i]) for i in indices)
    return selected
def contact_witnesses(path, video_field, frame_field, selected, args):
    """Choose a saliency-independent witness by seeded hash; do not filter reversals."""
    import hashlib
    witnesses = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {'patch_idx_a', 'patch_idx_b'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Contact sheets require columns: {', '.join(sorted(missing))}")
        for row in progress(reader, "Choose contact witnesses", args.no_tqdm):
            key = (identity(row, video_field, reader.line_num), identity(row, frame_field, reader.line_num))
            if key not in selected:
                continue
            try:
                a, b = int(row['patch_idx_a']), int(row['patch_idx_b'])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Line {reader.line_num}: invalid patch indices") from exc
            if a < 0 or b < 0 or a == b:
                raise ValueError(f"Line {reader.line_num}: expected two distinct nonnegative patch indices")
            digest = hashlib.sha256(f"{args.split_seed}|{key!r}|{a}|{b}".encode()).digest()
            if key not in witnesses or digest < witnesses[key][0]:
                witnesses[key] = (digest, dict(row))
    if set(witnesses) != selected:
        raise ValueError("Some chosen frames have no CSV witness rows.")
    return {key: value[1] for key, value in witnesses.items()}
def target_frame_position(names, frame, mode):
    if mode in ('index0', 'index1'):
        index = int(frame) - (mode == 'index1')
        if not 0 <= index < len(names):
            raise ValueError(f"Frame {frame} outside video under {mode} indexing")
        return index
    matches = [i for i, name in enumerate(names)
               if Path(name).stem == str(frame) or
               (str(frame).isdecimal() and Path(name).stem.isdecimal() and int(Path(name).stem) == int(frame))]
    if len(matches) != 1:
        raise ValueError(f"Frame {frame}: expected one matching filename, found {len(matches)}")
    return matches[0]
def patch_box(index, grid, size):
    h, w = grid
    width, height = size
    if not 0 <= index < h * w:
        raise ValueError(f"Patch {index} outside stage-4 grid {h} x {w}")
    row, col = divmod(index, w)
    return (col * width / w, row * height / h, (col + 1) * width / w, (row + 1) * height / h)
def draw_witness_patch_boxes(image, boxes, colors, scale):
    """Outline witness patches A/B on RGB frame or saliency heatmap (no fill; scale maps native box coords)."""
    from PIL import ImageDraw
    outlines = {'red': 'red', 'blue': 'blue', 'gray': 'gray'}
    draw = ImageDraw.Draw(image)
    for label, box in boxes.items():
        key = colors[label]
        scaled = tuple(value * scale for value in box)
        draw.rectangle(scaled, outline=outlines[key], width=3)
def align_gt_map(ground_truth, shape):
    """Align GT to prediction resolution with bilinear interpolation; no blur."""
    import numpy as np
    from PIL import Image
    gt = np.asarray(ground_truth, dtype=np.float32)
    if gt.ndim != 2 or not np.isfinite(gt).all() or (gt < 0).any():
        raise ValueError("Expected finite nonnegative GT map [H,W]")
    if tuple(gt.shape) != tuple(shape):
        gt = np.asarray(Image.fromarray(gt).resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
    return gt


def map_agreement(prediction, ground_truth):
    """Pearson CC and SIM on original dense maps; undefined metrics reject frames."""
    import numpy as np
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.ndim != 2 or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("Expected finite nonnegative predicted map [H,W]")
    gt = align_gt_map(ground_truth, pred.shape).astype(np.float64)
    x, y = pred.ravel(), gt.ravel()
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    cc = float(np.clip(np.dot(xc, yc) / denominator, -1, 1)) if denominator > 0 else None
    sx, sy = float(x.sum()), float(y.sum())
    sim = float(np.minimum(x / sx, y / sy).sum()) if sx > 0 and sy > 0 else None
    return {'cc': cc, 'sim': sim, 'resolution_hw': list(pred.shape),
            'gt_alignment': 'bilinear to predicted map resolution; no smoothing'}


def display_map(values, size, sigma, gamma):
    """Render a separate copy, applying Gaussian blur and optional gamma only for display."""
    import numpy as np
    from PIL import Image, ImageFilter
    values = np.asarray(values, dtype=np.float32)
    image = Image.fromarray(np.round(np.clip(values, 0, 1) * 255).astype('uint8'))
    image = image.resize(size, Image.Resampling.BILINEAR)
    if sigma > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if gamma != 1:
        pixels = np.asarray(image, dtype=np.float32) / 255
        image = Image.fromarray(np.round(np.clip(pixels ** gamma, 0, 1) * 255).astype('uint8'))
    return image.convert('RGB')


def select_top_patch_witnesses(path, frames, videos, video_field, frame_field, args, scorer):
    """First qualifying frame in CSV order per video; stop inference for completed videos.
    Top band contains ceil(percent * N / 100) patches; ties at its cutoff
    are included. B must also strictly exceed A by more than tie_epsilon.
    Only gallery examples are filtered; statistical reports use all rows.
    Videos are taken in sorted CSV order from --contact-video-start; if a video
    has no qualifying frame, the next CSV video is tried until --contact-video-count
    videos each have a witness (or the CSV list is exhausted).
    """
    ordered = sorted(videos)
    start = args.contact_video_start
    target = args.contact_video_count
    if start >= len(ordered):
        contact_sheet_video_ids(videos, args)
    tail = ordered[start:]
    print("Contact sheet: need {:,} video(s) with witnesses; scanning from CSV index {:,} ({}) onward."
          .format(target, start, tail[0] if tail else "n/a"), flush=True)
    selected = {}
    counts = defaultdict(int)
    checked = defaultdict(set)
    rejected_quality = defaultdict(set)
    active = set()
    skipped = []
    ptr = 0
    cache = {}
    scorer.contact_inferences = {}
    scorer.contact_metrics = {}
    last_inference = None

    def gallery_complete():
        return sum(1 for v, n in counts.items() if n >= args.frames_per_video) >= target

    def refill_active():
        nonlocal ptr
        while not gallery_complete() and len(active) < target and ptr < len(tail):
            video = tail[ptr]
            ptr += 1
            if counts[video] >= args.frames_per_video or video in skipped:
                continue
            active.add(video)

    def finalize_pass():
        for video in list(active):
            if counts[video] >= args.frames_per_video:
                active.discard(video)
                continue
            active.discard(video)
            if video not in skipped:
                skipped.append(video)
                print("Gallery video {}: no witness meets top-band B>A and raw-map CC/SIM; trying next CSV video."
                      .format(video), flush=True)
        refill_active()

    def scan_csv_pass():
        nonlocal last_inference, cache
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = {'patch_idx_a', 'patch_idx_b'} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Contact sheets require columns: {', '.join(sorted(missing))}")
            for row in progress(reader, "Find first top-band B witness per video", args.no_tqdm):
                if gallery_complete():
                    break
                video = identity(row, video_field, reader.line_num)
                if video not in active or counts[video] >= args.frames_per_video:
                    continue
                frame = identity(row, frame_field, reader.line_num)
                key = (video, frame)
                if key in selected:
                    continue
                if key not in cache:
                    result = scorer.infer(video, frame)
                    last_inference = (key, result)
                    pred, gt = result[:2]
                    pred = tuple(float(x) for x in pred)
                    gt = tuple(float(x) for x in gt)
                    if not pred or not all(math.isfinite(x) for x in pred):
                        raise ValueError(f"Invalid predicted patch grid for {video}/{frame}")
                    n_top = max(1, math.ceil(len(pred) * args.contact_top_percent / 100.0))
                    cutoff = sorted(pred, reverse=True)[n_top - 1]
                    cache[key] = (pred, gt, cutoff)
                    checked[video].add(frame)
                pred, gt, cutoff = cache[key]
                if frame in rejected_quality[video]:
                    continue
                try:
                    a, b = int(row['patch_idx_a']), int(row['patch_idx_b'])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Line {reader.line_num}: invalid patch indices") from exc
                if a == b or not (0 <= a < len(pred) and 0 <= b < len(pred)):
                    raise ValueError(f"Line {reader.line_num}: invalid distinct-patch pair")
                pa, pb = pred[a], pred[b]
                if pb < cutoff or pb - pa <= args.tie_epsilon:
                    continue
                if key not in scorer.contact_metrics:
                    if last_inference is not None and last_inference[0] == key:
                        quality_result = last_inference[1]
                    else:
                        quality_result = scorer.infer(video, frame)
                        last_inference = (key, quality_result)
                    metrics = map_agreement(quality_result[2], quality_result[7])
                    scorer.contact_metrics[key] = metrics
                    if (metrics["cc"] is None or metrics["sim"] is None or
                            metrics["cc"] < args.contact_min_cc or metrics["sim"] < args.contact_min_sim):
                        rejected_quality[video].add(frame)
                        continue
                if frame in rejected_quality[video]:
                    continue
                witness = dict(row)
                witness.update(pred_sal_a=str(pa), pred_sal_b=str(pb),
                               gt_sal_a=str(gt[a]), gt_sal_b=str(gt[b]))
                selected[key] = witness
                if last_inference is not None and last_inference[0] == key:
                    scorer.contact_inferences[key] = last_inference[1]
                counts[video] += 1
                print(f"Gallery {video}/{frame}: B={pb:.6g} >= top-{args.contact_top_percent:g}% cutoff {cutoff:.6g}; A={pa:.6g}")
                if counts[video] >= args.frames_per_video:
                    cache = {k: v for k, v in cache.items() if k[0] != video}
                    active.discard(video)
                    refill_active()

    refill_active()
    while not gallery_complete() and active:
        scan_csv_pass()
        if gallery_complete():
            break
        finalize_pass()
    gallery_videos = sorted(
        {v for v, n in counts.items() if n >= args.frames_per_video},
        key=lambda v: ordered.index(v))
    if len(gallery_videos) < target:
        print("Contact sheet: only {:,} of {:,} requested videos have witnesses (CSV exhausted from index {:,})."
              .format(len(gallery_videos), target, start), flush=True)
    elif gallery_videos:
        print("Contact sheet gallery videos: {}".format(", ".join(gallery_videos)), flush=True)
    report = []
    for video in gallery_videos:
        chosen = [f for v, f in selected if v == video]
        report.append({'video': video, 'frames_checked': len(checked[video]),
                       'quality_checked_frames': sum(v == video for v, _ in scorer.contact_metrics),
                       'quality_rejected_frames': len(rejected_quality[video]),
                       'selected_frames': chosen, 'stopped_early': counts[video] >= args.frames_per_video})
    for video in skipped:
        if video in gallery_videos:
            continue
        report.append({'video': video, 'frames_checked': len(checked[video]),
                       'quality_checked_frames': sum(v == video for v, _ in scorer.contact_metrics),
                       'quality_rejected_frames': len(rejected_quality[video]),
                       'selected_frames': [], 'skipped_no_witness': True})
    return selected, report, gallery_videos
def generate_contact_sheets(path, frames, videos, discovery, video_field, frame_field, args, scorer=None):
    import importlib
    import os
    import sys
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    sys.path.insert(0, str(args.src_dir.expanduser().resolve()))
    import torch
    from PIL import Image, ImageDraw
    import numpy as np
    if scorer is None:
        scorer = FinalMapScorer(args)
    dataset = scorer.dataset
    checkpoint = args.checkpoint.expanduser().resolve()
    witnesses, selection_report, gallery_videos = select_top_patch_witnesses(
        path, frames, videos, video_field, frame_field, args, scorer)
    selected = set(witnesses)
    lookup = { (str(video), int(start)): i for i, (video, start) in enumerate(dataset.windows) }
    output = args.output_dir or path.parent / 'contact_sheets_quality'
    output.mkdir(parents=True, exist_ok=True)
    panels = []
    metadata = []
    for video, frame in sorted(selected):
        if video not in dataset.video_frame_names:
            raise ValueError(f"Video {video} absent from DatasetLoader")
        names = dataset.video_frame_names[video]
        position = target_frame_position(names, frame, args.frame_index_mode)
        start = position - dataset.window_len + 1
        if (video, start) not in lookup:
            raise ValueError(f"No valid window ending at {video}/{names[position]}; check --frame-index-mode and --window-len")
        pred_grid, gt_grid, prediction, grid, position, filename, start, dense_gt = (scorer.contact_inferences[(video, frame)] if (video, frame) in scorer.contact_inferences else scorer.infer(video, frame))
        frame_path = args.val_dataset_dir / video / 'images' / names[position]
        with Image.open(frame_path) as handle:
            original = handle.convert('RGB')
        row = witnesses[(video, frame)]
        a, b = int(row['patch_idx_a']), int(row['patch_idx_b'])
        if not (0 <= a < len(pred_grid) and 0 <= b < len(pred_grid)):
            raise ValueError('Contact witness indices outside stage-4 grid')
        pa, pb = float(pred_grid[a]), float(pred_grid[b])
        ga, gb = float(gt_grid[a]), float(gt_grid[b])
        row.update(pred_sal_a=str(pa), pred_sal_b=str(pb), gt_sal_a=str(ga), gt_sal_b=str(gb))
        boxes = { 'A': patch_box(a, grid, original.size), 'B': patch_box(b, grid, original.size) }
        # Colors use pooled scores from the displayed final predicted map.
        colors = {'A': 'red' if pa > pb else 'blue', 'B': 'red' if pb > pa else 'blue'}
        if abs(pa - pb) <= args.tie_epsilon:
            colors = {'A': 'gray', 'B': 'gray'}
        width = args.panel_width
        height = round(original.height * width / original.width)
        box_scale = width / original.width
        annotated = original.resize((width, height))
        draw_witness_patch_boxes(annotated, boxes, colors, box_scale)
        # Display transformations never feed into selection, patch scores, or metrics.
        metrics = scorer.contact_metrics[(video, frame)]
        map_image = display_map(prediction, (width, height), args.display_blur_sigma, args.display_gamma)
        gt_aligned = align_gt_map(dense_gt, prediction.shape)
        gt_max = float(gt_aligned.max())
        gt_display = gt_aligned / gt_max if gt_max > 0 else gt_aligned
        gt_image = display_map(gt_display, (width, height), 0.0, 1.0)
        draw_witness_patch_boxes(map_image, boxes, colors, box_scale)
        draw_witness_patch_boxes(gt_image, boxes, colors, box_scale)
        header = 100
        panel = Image.new('RGB', (3 * width + 24, height + header), 'white')
        panel.paste(annotated, (0, header))
        panel.paste(map_image, (width + 12, header))
        panel.paste(gt_image, (2 * width + 24, header))
        draw = ImageDraw.Draw(panel)
        subset = 'discovery' if video in discovery else 'confirmation'
        draw.text((6, 4), f"Video {video}; frame {frame} ({names[position]}); {subset}; selected illustration", fill='black')
        draw.text((6, 20), f"Original predicted: A={pa:.5g}, B={pb:.5g}; blue=A, red=B", fill='black')
        draw.text((6, 36), f"Dataset GT: A={ga:.5g}, B={gb:.5g}; raw-map CC={metrics['cc']:.3f}, SIM={metrics['sim']:.3f}", fill='black')
        draw.text((6, 52), f"B in top {args.contact_top_percent:g}%; display-only blur sigma={args.display_blur_sigma:g}px, gamma={args.display_gamma:g}", fill='black')
        draw.text((6, 80), 'RGB frame', fill='black')
        draw.text((width + 18, 80), 'Predicted map (display postprocessed)', fill='black')
        draw.text((2 * width + 30, 80), 'Ground truth (max-scaled for display)', fill='black')
        # Numeric originals and an unsmoothed PNG make the display auditable.
        sample_id = f"sample_{len(metadata) + 1:03d}"
        np.savez_compressed(output / f'{sample_id}_raw_maps.npz', predicted=prediction, ground_truth=dense_gt)
        display_map(prediction, (width, height), 0.0, 1.0).save(output / f'{sample_id}_predicted_raw.png')
        display_map(prediction, (width, height), args.display_blur_sigma, args.display_gamma).save(output / f'{sample_id}_predicted_display.png')
        panels.append((video, panel))
        metadata.append({'video': video, 'csv_frame': frame, 'frame_filename': names[position],
                         'absolute_frame_index': position, 'window_start_index': start,
                         'window_length': dataset.window_len, 'grid_hw': grid,
                         'raw_map_agreement': metrics, 'raw_maps_file': f'{sample_id}_raw_maps.npz',
                         'display_blur_sigma_px': args.display_blur_sigma, 'display_gamma': args.display_gamma,
                         'boxes_native_xyxy': boxes, 'top_band_cutoff': float(sorted(pred_grid, reverse=True)[max(1, math.ceil(len(pred_grid) * args.contact_top_percent / 100.0)) - 1]), 'witness': row})
    def save_sheet(items, filename):
        sheet_width = max(p.width for p in items)
        sheet = Image.new('RGB', (sheet_width, sum(p.height + 8 for p in items)), 'white')
        y = 0
        for panel in items:
            sheet.paste(panel, (0, y));y += panel.height + 8
        sheet.save(output / filename)
    if panels:
        save_sheet([panel for _, panel in panels], 'contact_sheet_first5.png')
    for index, video in enumerate(sorted({v for v, _ in panels}), 1):
        save_sheet([p for v, p in panels if v == video], f'contact_sheet_video_{index:02d}.png')
    import json
    with (output / 'contact_sheet_metadata.json').open('w', encoding='utf-8') as handle:
        json.dump({'checkpoint': str(checkpoint), 'frame_index_mode': args.frame_index_mode,
                   'contact_video_start': args.contact_video_start,
                   'contact_video_count': args.contact_video_count,
                   'contact_videos': gallery_videos,
                   'contact_videos_skipped_no_witness': [r['video'] for r in selection_report if r.get('skipped_no_witness')],
                   'selection': 'saliency-selected illustrative examples: first distinct frame per video in sorted CSV order (from --contact-video-start) where B is in the frame-wide top band and exceeds A and maps meet CC/SIM thresholds; advance to next CSV video when a video has no witness until --contact-video-count videos are filled',
                   'selection_report': selection_report, 'contact_top_percent': args.contact_top_percent,
                   'contact_min_cc': args.contact_min_cc, 'contact_min_sim': args.contact_min_sim,
                   'display_only': {'blur_sigma_px': args.display_blur_sigma, 'gamma': args.display_gamma},
                   'samples': metadata}, handle, indent=2)
    print(f"Saved {len(panels)} frame panels and contact sheets to {output}")
class FinalMapScorer:
    """Infer each represented temporal window once; retain only CPU patch grids.
    Predicted map: final model saliency_map, min/max normalized exactly as in
    train.prepare_pred_for_visualization. GT: dataset saliency, NOT fixation.
    Both maps are average pooled to the stage-4 grid, with row-major patch IDs.
    """
    def __init__(self, args):
        import importlib
        import os
        import sys
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        sys.path.insert(0, str(args.src_dir.expanduser().resolve()))
        import torch
        from pre_process.dataloader import DatasetLoader
        from pre_process.collate import video_saliency_collate_fn
        self.torch = torch
        self.collate = video_saliency_collate_fn
        self.args = args
        evaluation = importlib.import_module(args.evaluation_module)
        self.train = evaluation.train_cfg
        self.device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        checkpoint = args.checkpoint.expanduser().resolve()
        if checkpoint.is_dir():
            options = sorted(list(checkpoint.glob('*.pth')) + list(checkpoint.glob('*.pt')))
            if len(options) != 1:
                raise ValueError('Supply a checkpoint file, or a directory containing exactly one .pth/.pt checkpoint')
            checkpoint = options[0]
        if not checkpoint.is_file():
            raise ValueError(f'Checkpoint not found: {checkpoint}')
        self.train.set_seed(getattr(self.train, 'SEED', 42))
        self.model = evaluation.build_model(self.device, self.device)
        evaluation.load_checkpoint(self.model, str(checkpoint))
        self.model.eval()
        self.dataset = DatasetLoader(str(args.val_dataset_dir), window_len=args.window_len or evaluation.WINDOW_LEN,
                                     stride=1, random_train_sampling=False)
        self.lookup = {(str(v), int(start)): i for i, (v, start) in enumerate(self.dataset.windows)}
        self.cache = {}
    def infer(self, video, frame):
        torch = self.torch
        if video not in self.dataset.video_frame_names:
            raise ValueError(f'Video {video} absent from dataset')
        names = self.dataset.video_frame_names[video]
        position = target_frame_position(names, frame, self.args.frame_index_mode)
        start = position - self.dataset.window_len + 1
        if (video, start) not in self.lookup:
            raise ValueError(f'No valid window for {video}/{frame}; check frame-index mode and window length')
        _, rgb, gt_saliency, fixation, _, _ = self.collate([self.dataset[self.lookup[(video, start)]]])
        # Collate's third item is GT saliency; fourth is fixation.
        with torch.inference_mode():
            prepared_rgb, prepared_sal, _ = self.model.prepare_training_batch(rgb, gt_saliency, fixation)
            with torch.autocast(self.device.type, dtype=self.train._amp_dtype(self.device),
                                enabled=self.train._amp_enabled(self.device)):
                out = self.model(prepared_rgb, saliency_maps=prepared_sal, return_details=True,
                                 return_concept_losses=False, return_decoder_diagnostics=False)
            final_map = self.train.prepare_pred_for_visualization(out['saliency_map'])
            if final_map.ndim != 2 or not torch.isfinite(final_map).all():
                raise ValueError('Expected a finite final predicted map [H,W]')
            gt = gt_saliency.detach().float()
            if gt.ndim != 4 or tuple(gt.shape[:2]) != (1, 1):
                raise ValueError('Expected GT saliency from collate as [1,1,H,W]')
            if not torch.isfinite(gt).all() or bool((gt < 0).any()) or bool((gt > 1).any()):
                raise ValueError('GT saliency must be finite and in [0,1]; do not silently renormalize it')
            shape = out['concept_out']['stage4']['visual_metadata']['feature_shape']
            grid = (int(shape['H']), int(shape['W']))
            pred_grid = torch.nn.functional.adaptive_avg_pool2d(final_map[None, None], grid).flatten().cpu().numpy()
            gt_grid = torch.nn.functional.adaptive_avg_pool2d(gt, grid).flatten().cpu().numpy()
        return pred_grid, gt_grid, final_map.cpu().numpy(), grid, position, names[position], start, gt[0, 0].cpu().numpy()
    def values(self, row, video, frame, line):
        key = (video, frame)
        if key not in self.cache:
            pred, gt, *_ = self.infer(video, frame)
            self.cache[key] = (pred, gt)
            if len(self.cache) % 100 == 0:
                print(f'  Recomputed {len(self.cache):,} distinct frames', flush=True)
        pred, gt = self.cache[key]
        try:
            a, b = int(row['patch_idx_a']), int(row['patch_idx_b'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Line {line}: invalid patch_idx_a/b') from exc
        if a == b or not (0 <= a < len(pred) and 0 <= b < len(pred)):
            raise ValueError(f'Line {line}: patch indices outside stage-4 grid or not distinct')
        return float(pred[a]), float(pred[b]), float(gt[a]), float(gt[b])
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="CSV path or directory containing frame_data.csv")
    parser.add_argument("--video-field", help="CSV video identifier column; otherwise detected")
    parser.add_argument("--frame-field", help="CSV frame identifier column; otherwise detected")
    parser.add_argument("--discovery-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--discovery-videos", nargs="+", help="Explicit frozen discovery video IDs; other represented videos confirm")
    parser.add_argument("--consistency-threshold", type=float, default=0.95, help="Agreement fraction for reporting consistent frames/videos (default .95)")
    parser.add_argument("--tie-epsilon", type=float, default=0.0, help="Absolute A-B difference treated as tie; default 0 preserves strict comparisons")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Enable contact sheets using a checkpoint file or directory with one checkpoint")
    parser.add_argument("--src-dir", type=Path, help="Project src directory containing evaluation.py, train.py and pre_process/")
    parser.add_argument("--evaluation-module", default="evaluation", help="Module exporting build_model and load_checkpoint")
    parser.add_argument("--val-dataset-dir", type=Path, default=Path("/data/quantization/zaima/videosal_datasets/dhf1k/val"))
    parser.add_argument("--frame-index-mode", choices=("index0", "index1", "filename"), help="Meaning of CSV frame_idx: zero/one-based frame-list index or numeric filename stem")
    parser.add_argument("--window-len", type=int, help="Override evaluation.WINDOW_LEN to match the exporter")
    parser.add_argument("--frames-per-video", type=int, default=1, help="Maximum qualifying frames per video; default 1")
    parser.add_argument("--contact-video-start", type=int, default=0,
                        help="0-based index into sorted CSV video IDs for contact sheets (default: 0).")
    parser.add_argument("--contact-video-count", type=int, default=5,
                        help="Gallery size: videos with a witness, advancing in sorted CSV order past videos with none (default: 5).")
    parser.add_argument("--contact-top-percent", type=float, default=3.0, help="Within-frame top predicted patch-saliency percentage for patch B; default 3 (cutoff ties included)")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, help="Contact-sheet destination (default: beside CSV/contact_sheets_top_band)")
    parser.add_argument("--device", help="Inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--contact-min-cc", type=float, default=0.60, help="Minimum original dense predicted/GT map Pearson CC for gallery (default .60)")
    parser.add_argument("--contact-min-sim", type=float, default=0.40, help="Minimum sum-normalized histogram intersection for gallery (default .40)")
    parser.add_argument("--display-blur-sigma", type=float, default=2.0, help="Display-only Gaussian sigma in panel pixels; 0 disables (default 2)")
    parser.add_argument("--display-gamma", type=float, default=1.0, help="Display-only power transform; 1 is neutral, <1 brightens (default 1)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.contact_min_cc) or not -1 <= args.contact_min_cc <= 1:
        parser.error("--contact-min-cc must be in [-1,1]")
    if not math.isfinite(args.contact_min_sim) or not 0 <= args.contact_min_sim <= 1:
        parser.error("--contact-min-sim must be in [0,1]")
    if not math.isfinite(args.display_blur_sigma) or args.display_blur_sigma < 0:
        parser.error("--display-blur-sigma must be finite and nonnegative")
    if not math.isfinite(args.display_gamma) or args.display_gamma <= 0:
        parser.error("--display-gamma must be finite and positive")
    if args.checkpoint and (args.src_dir is None or args.frame_index_mode is None):
        parser.error("--checkpoint requires --src-dir and --frame-index-mode")
    if args.frames_per_video < 1 or args.panel_width < 200 or (args.window_len is not None and args.window_len < 1):
        parser.error("Frames/window length must be positive and panel width at least 200")
    if args.contact_video_start < 0 or args.contact_video_count < 1:
        parser.error("--contact-video-start must be >= 0 and --contact-video-count >= 1")
    if not 0 < args.discovery_fraction < 1:
        parser.error("--discovery-fraction must be strictly between 0 and 1")
    if not 0.5 < args.consistency_threshold <= 1:
        parser.error("--consistency-threshold must be greater than .5 and at most 1")
    if not math.isfinite(args.contact_top_percent) or not 0 < args.contact_top_percent <= 100:
        parser.error("--contact-top-percent must be finite and in (0,100]")
    if not math.isfinite(args.tie_epsilon) or args.tie_epsilon < 0:
        parser.error("--tie-epsilon must be finite and nonnegative")
    return args
def main(argv=None):
    args = parse_args(argv)
    path = resolve_frame_data_path(args.input_dir)
    before = path.stat()
    videos, video_field, frame_field = inspect_videos(path, args)
    discovery, confirmation = split_videos(videos, args)
    print(f"Input: {path}\nVideo column: {video_field}; frame column: {frame_field}")
    print(f"Split seed: {args.split_seed}; discovery fraction: {args.discovery_fraction}")
    print("Discovery videos: " + ", ".join(sorted(discovery)))
    print("Confirmation videos: " + ", ".join(sorted(confirmation)))
    totals, frames = load_reports(path, args, video_field, frame_field, discovery)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed during analysis; rerun on a stable CSV.")
    rate = macro_rate(frames["discovery"], "pred", exclude_ties=True)
    if not math.isfinite(rate) or rate == 0.5:
        raise ValueError("Discovery frame mean has no preferred predicted ordering; no rule to confirm.")
    direction = "B" if rate > 0.5 else "A"
    print("Ordering selected from discovery mean of within-frame non-tied predicted comparisons.")
    for subset in ("discovery", "confirmation"):
        print_report(subset, totals[subset], frames[subset], direction, args)
    print("\nThese statistics describe this preselected signature pair; they do not independently discover signatures.")
    print("If this pair was selected using all validation videos, this split is a robustness check, not untouched confirmation.")
    if args.checkpoint:
        generate_contact_sheets(path, frames, videos, discovery, video_field, frame_field, args)
    return 0
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Error: {exc}")
