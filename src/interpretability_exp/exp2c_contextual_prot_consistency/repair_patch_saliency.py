#!/usr/bin/env python3
"""Repair patch_data.json using final checkpoint predictions and true GT saliency.

Standalone script: requires your project evaluation.py, train.py, dataloader and
collate. Final predicted maps are min/max normalized per frame, exactly like
train.prepare_pred_for_visualization and find_patterns.py (score-source maps).
GT comes from collate's THIRD output (saliency), not FOURTH (fixation).
Both are adaptively average pooled onto the stage-4 patch grid. Existing patch
identities and prototype activation fields are preserved. Each distinct window
is inferred once, with only CPU patch score grids retained in memory.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

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
        return pred_grid, gt_grid, final_map.cpu().numpy(), grid, position, names[position], start


def stream_records(path, chunk_size=1024 * 1024):
    """Incrementally parse one JSON array without loading millions of records."""
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8-sig') as handle:
        buffer = ''
        eof = False
        def fill():
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            if not chunk:
                eof = True
            buffer += chunk
        def trim():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()
        trim()
        if not buffer.startswith('['):
            raise ValueError('Input must be a JSON array of patch records')
        buffer = buffer[1:]
        first = True
        while True:
            trim()
            if buffer.startswith(']'):
                buffer = buffer[1:]
                break
            if not first:
                if not buffer.startswith(','):
                    raise ValueError('Expected comma between patch records')
                buffer = buffer[1:]
                trim()
                if buffer.startswith(']'):
                    raise ValueError('Trailing comma in input JSON')
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError('Invalid or truncated input JSON') from exc
                    fill()
            if not isinstance(record, dict):
                raise ValueError('Every JSON array element must be a patch object')
            buffer = buffer[end:]
            first = False
            yield record
        # Check the full remaining input; another JSON value is not permitted.
        if buffer.strip() or any(chunk.strip() for chunk in iter(lambda: handle.read(chunk_size), '')):
            raise ValueError('Unexpected data after input JSON array')


def integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    try:
        number = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be an integer: {value!r}') from exc
    if isinstance(value, float) and value != number:
        raise ValueError(f'{name} must be an integer: {value!r}')
    return number


def repair_record(record, scorer, cache):
    video = str(record.get('video_fname', '')).strip()
    if not video:
        raise ValueError('Missing video_fname')
    if 'absolute_frame_index' not in record:
        raise ValueError('Missing absolute_frame_index; do not guess frame_no indexing')
    position = integer(record['absolute_frame_index'], 'absolute_frame_index')
    if position < 0:
        raise ValueError('absolute_frame_index must be nonnegative')
    window_length = scorer.dataset.window_len
    if 'window_length' in record and integer(record['window_length'], 'window_length') != window_length:
        raise ValueError(f'Exported window length differs from {window_length}; supply --window-len matching the export')
    if 'window_start_index' in record and integer(record['window_start_index'], 'window_start_index') != position-window_length+1:
        raise ValueError('window_start_index disagrees with absolute_frame_index and window length')
    key = (video, position)
    if key not in cache:
        pred, gt, _, grid, actual, filename, _ = scorer.infer(video, str(position))
        if actual != position:
            raise ValueError('Inferred frame index differs from exported index')
        cache[key] = (pred, gt, grid, filename)
    pred, gt, grid, filename = cache[key]
    if 'frame_filename' in record and str(record['frame_filename']) != str(filename):
        raise ValueError(f'Frame filename mismatch for {video}/{position}: export {record["frame_filename"]!r}, dataset {filename!r}')
    if 'feature_grid_hw' in record and list(record['feature_grid_hw']) != list(grid):
        raise ValueError(f'Exported patch grid differs from model stage-4 grid {grid}')
    patch = integer(record.get('patch_index'), 'patch_index')
    if not 0 <= patch < len(pred):
        raise ValueError(f'Patch index {patch} outside model grid {grid}')
    for field, expected in [('grid_row', patch // grid[1]), ('grid_column', patch % grid[1])]:
        if field in record and integer(record[field], field) != expected:
            raise ValueError(f'{field} disagrees with row-major patch_index')
    pa, ga = float(pred[patch]), float(gt[patch])
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (pa, ga)):
        raise ValueError('Recomputed scores must be finite and in [0,1]')
    result = dict(record)
    result['patch_pred_saliency'] = pa
    result['patch_gt_saliency'] = ga
    # Old bins were calculated from old predictions and must be recomputed.
    result['saliency_bin'] = None
    result['saliency_score_source'] = {
        'predicted': 'final_saliency_map_per_frame_minmax_then_adaptive_average_pool',
        'ground_truth': 'dataset_gt_saliency_then_adaptive_average_pool',
    }
    return result


def repair_file(source, destination, scorer, overwrite=False, in_place=False, progress_every=100000,
                progress_iter=None):
    if destination.exists() and not overwrite and not in_place:
        raise ValueError(f'Output exists: {destination}; use --overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    fd, temp_name = tempfile.mkstemp(prefix='.'+destination.name+'.', suffix='.tmp', dir=destination.parent)
    cache = {}
    count = 0
    ranges = {'pred': [math.inf, -math.inf], 'gt': [math.inf, -math.inf]}
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            output.write('[\n')
            records = stream_records(source)
            if progress_iter is not None:
                records = progress_iter(records)
            for count, record in enumerate(records, 1):
                try:
                    updated = repair_record(record, scorer, cache)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'Patch record {count}: {exc}') from exc
                if count > 1:
                    output.write(',\n')
                json.dump(updated, output, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                for name, field in [('pred','patch_pred_saliency'),('gt','patch_gt_saliency')]:
                    ranges[name][0] = min(ranges[name][0], updated[field])
                    ranges[name][1] = max(ranges[name][1], updated[field])
                if progress_iter is None and progress_every and count % progress_every == 0:
                    print(f'Updated {count:,} patches from {len(cache):,} distinct frames', flush=True)
            if not count:
                raise ValueError('Input has no patch records')
            output.write('\n]\n')
            output.flush()
            os.fsync(output.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source changed during repair; output was not committed')
        if in_place:
            backup = source.with_name(source.name + '.bak')
            if backup.exists():
                raise ValueError(f'Backup exists: {backup}; preserve or rename it before another in-place repair')
            shutil.copy2(source, backup)
            print(f'Original backed up to {backup}')
        os.replace(temp_name, destination)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    print(f'Saved {count:,} patches, {len(cache):,} distinct frames: {destination}')
    print(f'Predicted range: {ranges["pred"]}; GT range: {ranges["gt"]}')
    return count, len(cache)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch-data', required=True, type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--output', type=Path, help='Default: patch_data_corrected.json beside input')
    group.add_argument('--in-place', action='store_true', help='Replace input atomically after creating patch_data.json.bak')
    p.add_argument('--overwrite', action='store_true', help='Allow replacing an existing separate output file')
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--src-dir', required=True, type=Path)
    p.add_argument('--evaluation-module', default='evaluation')
    p.add_argument('--val-dataset-dir', type=Path, default=Path('/data/quantization/zaima/videosal_datasets/dhf1k/val'))
    p.add_argument('--window-len', type=int, help='Must match exported temporal window length; default evaluation.WINDOW_LEN')
    p.add_argument('--device', help='e.g. cuda:0 or cpu')
    p.add_argument('--progress-every', type=int, default=100000)
    args = p.parse_args(argv)
    if args.progress_every < 0 or (args.window_len is not None and args.window_len < 1):
        p.error('Progress interval must be nonnegative and window length positive')
    # Export records explicitly provide zero-based absolute_frame_index.
    args.frame_index_mode = 'index0'
    return args


def main(argv=None):
    args = parse_args(argv)
    source = args.patch_data.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f'Input not found: {source}')
    destination = source if args.in_place else (args.output.expanduser().resolve() if args.output else source.with_name(source.stem+'_corrected.json'))
    if destination == source and not args.in_place:
        raise ValueError('Use --in-place to replace the input')
    scorer = FinalMapScorer(args)
    repair_file(source, destination, scorer, args.overwrite, args.in_place, args.progress_every)
    print('Prototype fields preserved. Regenerate saliency bins, candidates and witness CSVs from the corrected export.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f'Error: {exc}')
