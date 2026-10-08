#!/usr/bin/env python3
"""Export validation-window stage-4 saliency and actual prototype activations.

The gallery adapter is exp2a_visual_gallery/across_videos/prototype_to_patch.py
(or pass --gallery-script). It supplies the existing model, DatasetLoader,
collate function and saliency-grid helpers. Pass the VALIDATION directory to
--dataset-dir.

Each window is inferred once. All spatial patches of its last frame are saved
to OUTPUT_DIR/patch_data.json, a JSON array. There are no appearance, saliency,
target-reference, or candidate-count filters. Padded last frames are skipped.
Records keep predicted and ground-truth patch scores; overlap_analysis.py assigns
the saliency bins. Run settings are saved in experiment_metadata.json.

Top-20 means up to 20 prototype slots per patch, ranked by descending
forward_activation when multiple prototypes are active; otherwise by descending
assignment_probability (hard-eval one-hot forward activations). Ties break on
ascending prototype ID. actual_active_prototype_count still counts prototypes
with forward_activation above the model validity epsilon. The model's top_k is
unchanged.
Each exported slot stores prototype_index, rank, and cosine_similarity rounded to
four decimal places. Assignment scores are used only for ranking and are not saved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


TOP_K = 20
VALIDITY_EPS = 1e-6
OUTPUT_DIR = Path('prototype_consistency_sal_outputs')


def load_gallery_adapter(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Gallery adapter not found: {path}')
    spec = importlib.util.spec_from_file_location('stage4_gallery_adapter', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import gallery adapter: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ('set_seed', 'resolve_checkpoint_path', 'load_saliency_model',
                'DatasetLoader', 'video_saliency_collate_fn', 'rgb_video_to_btchw',
                'resolve_backbone_spatial_hw', 'extract_predicted_patch_saliency',
                'extract_gt_saliency_batch', 'prepare_saliency_maps_for_concept_grid',
                'compute_patch_saliency_grid')
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ValueError(f'Gallery adapter is missing helpers: {missing}')
    return module


def verify_checkpoint_parameters(model, checkpoint):
    """Retain the source script's checkpoint/model compatibility guard."""
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    if not isinstance(state, dict):
        raise ValueError('Expected a state dict or model_state_dict checkpoint.')
    bad = [name for name, param in model.named_parameters()
           if name not in state or not torch.is_tensor(state[name])
           or state[name].shape != param.shape]
    if bad:
        raise ValueError(f'Checkpoint/model parameter mismatch: {bad[:10]}')


def bounds(patch_index, grid_hw, image_hw):
    gh, gw = map(int, grid_hw)
    height, width = map(int, image_hw)
    if min(gh, gw, height, width) < 1 or not 0 <= patch_index < gh * gw:
        raise ValueError('Invalid image/grid dimensions or patch index.')
    row, column = divmod(patch_index, gw)
    return [round(column * width / gw), round(row * height / gh),
            round((column + 1) * width / gw), round((row + 1) * height / gh)]


def concept_row_metadata(stage, rows):
    meta = stage['visual_metadata']
    shape = {key: int(meta['feature_shape'][key]) for key in ('B', 'T', 'H', 'W')}
    if min(shape.values()) < 1:
        raise ValueError('Concept grid dimensions must be positive.')
    values = []
    for key, upper in (('batch_idx', shape['B']), ('time_idx', shape['T']),
                       ('patch_idx', shape['H'] * shape['W'])):
        value = meta[key]
        if not torch.is_tensor(value) or value.ndim != 1 or value.numel() != rows:
            raise ValueError(f'{key} must be a one-dimensional tensor of {rows} rows.')
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f'{key} must contain integer indices.')
        value = value.detach().cpu().long()
        if ((value < 0) | (value >= upper)).any():
            raise ValueError(f'{key} contains out-of-range indices.')
        values.append(value)
    return shape, values


def valid_active_matrix(stage, rows, prototype_count, device):
    """Recover true assignments from the model exports, including 5D layouts."""
    indices = stage.get('active_visual_prototype_indices')
    valid = stage.get('visual_validity_mask')
    if not torch.is_tensor(indices) or not torch.is_tensor(valid) or valid.dtype != torch.bool:
        raise ValueError('Model must export active_visual_prototype_indices and bool visual_validity_mask.')
    if indices.shape != valid.shape or indices.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Invalid active index/mask shape or dtype.')
    shape, (b, t, p) = concept_row_metadata(stage, rows)
    if indices.ndim == 5:
        if tuple(indices.shape[:4]) != tuple(shape[key] for key in ('B', 'T', 'H', 'W')):
            raise ValueError('Active exports disagree with the concept grid.')
        b, t, p = b.to(indices.device), t.to(indices.device), p.to(indices.device)
        indices = indices[b, t, p // shape['W'], p % shape['W']]
        valid = valid[b, t, p // shape['W'], p % shape['W']]
    if indices.ndim != 2 or indices.shape[0] != rows or indices.shape[1] < 1:
        raise ValueError('Active exports must be [B,T,H,W,K] or [rows,K].')
    indices, valid = indices.to(device=device, dtype=torch.long), valid.to(device)
    if (((indices < 0) | (indices >= prototype_count)) & valid).any():
        raise ValueError('Valid prototype index outside the bank.')
    result = torch.zeros(rows, prototype_count, dtype=torch.bool, device=device)
    row_ids = torch.arange(rows, device=device)[:, None].expand_as(indices)
    result[row_ids[valid], indices[valid]] = True
    return result


def last_frame_rows(stage, rows):
    """Use explicit metadata; never assume the concept rows' flattening order."""
    shape, (b, t, p) = concept_row_metadata(stage, rows)
    n = shape['H'] * shape['W']
    selected = (t == shape['T'] - 1).nonzero(as_tuple=False).flatten()
    keys = b[selected] * n + p[selected]
    if keys.unique().numel() != selected.numel():
        raise ValueError('Duplicate last-frame concept rows for the same patch.')
    lookup = torch.full((shape['B'], n), -1, dtype=torch.long)
    lookup[b[selected], p[selected]] = selected
    if (lookup < 0).any():
        raise ValueError('Missing last-frame concept rows; cannot export every spatial patch.')
    return shape, lookup


def prototype_measurements(stage, prototype_bank):
    q = stage['visual_patch_embeddings'].detach().float()
    bank = prototype_bank.detach().to(q.device).float()
    if q.ndim != 2 or bank.ndim != 2 or q.shape[1] != bank.shape[1] or bank.shape[0] < 1:
        raise ValueError('Patch/prototype embedding dimensions do not match.')
    if not torch.isfinite(q).all() or not torch.isfinite(bank).all():
        raise ValueError('Nonfinite patch or prototype embeddings.')
    cosine = F.normalize(q, dim=-1) @ F.normalize(bank, dim=-1).T
    values = {'cos_sim': cosine,
              'scaled_logit': stage['visual_concept_logits'].detach().float(),
              'assignment_probability': stage['visual_assignment_probs'].detach().float(),
              'forward_activation': stage['visual_activations'].detach().float()}
    for name, value in values.items():
        if value.shape != cosine.shape or not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite or misaligned {name}.')
    active = valid_active_matrix(stage, q.shape[0], bank.shape[0], q.device)
    return {name: value.cpu() for name, value in values.items()}, active.cpu()


def ranking_scores(measurements, row):
    """Pick the score vector used to order exported top-k prototype slots."""
    forward = measurements['forward_activation'][row]
    if int((forward > VALIDITY_EPS).sum()) <= 1:
        return measurements['assignment_probability'][row]
    return forward


def export_cosine_similarity(value):
    return round(float(value), 4)


def top_active_prototypes(measurements, row):
    """Export up to TOP_K ranked slots with prototype index, rank, and cosine only."""
    scores = ranking_scores(measurements, row)
    count = int(scores.shape[0])
    selected = sorted(range(count), key=lambda prototype: (-float(scores[prototype]), prototype))[:min(TOP_K, count)]
    forward = measurements['forward_activation'][row]
    active_count = int((forward > VALIDITY_EPS).sum())
    cosine = measurements['cos_sim'][row]
    return [{'prototype_index': prototype, 'rank': rank,
             'cosine_similarity': export_cosine_similarity(cosine[prototype])}
            for rank, prototype in enumerate(selected, 1)], active_count


def compute_patch_scores(adapter, out, sal_batch, resize_to, grid_hw, batch_size):
    # Reuse exactly the predicted/GT extraction paths in the supplied script.
    predicted = adapter.extract_predicted_patch_saliency(out, grid_hw)
    gt_maps = adapter.extract_gt_saliency_batch(sal_batch)
    prepared_gt = adapter.prepare_saliency_maps_for_concept_grid(gt_maps, resize_to)
    ground_truth = adapter.compute_patch_saliency_grid(prepared_gt, grid_hw)
    expected = (batch_size, grid_hw[0] * grid_hw[1])
    for name, value in (('predicted', predicted), ('ground truth', ground_truth)):
        if not torch.is_tensor(value) or tuple(value.shape) != expected:
            raise ValueError(f'{name} patch saliency must have shape {expected}.')
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite {name} patch saliency.')
    return predicted.detach().cpu().float(), ground_truth.detach().cpu().float()


@torch.inference_mode()
def collect_patch_data(model, loader, adapter, out):
    if 'stage4' not in model.concept_creations:
        raise ValueError('The loaded model has no stage4 concept module.')
    concept = model.concept_creations['stage4']
    if not isinstance(loader.dataset, Subset):
        raise TypeError('Use a sequential Subset loader to recover source window indices.')
    dataset = loader.dataset.dataset
    resize_to = tuple(map(int, adapter.resolve_backbone_spatial_hw(model)))
    stats = {'windows_seen': 0, 'windows_exported': 0,
             'invalid_target_windows_skipped': 0, 'patch_count': 0,
             'patches_with_fewer_than_20_active_prototypes': 0,
             'predicted_patch_saliency_min': None, 'predicted_patch_saliency_max': None}
    active_counts = Counter()
    videos = set()
    cursor = 0
    first_record = True
    model.eval()
    out.write('[\n')
    for batch_number, batch in enumerate(loader):
        if not isinstance(batch, (tuple, list)) or len(batch) != 6:
            raise ValueError('Expected six-field video_saliency_collate_fn output.')
        names, rgb_batch, _, sal_batch, _, padding_valid = batch
        rgb = adapter.rgb_video_to_btchw(rgb_batch)
        if rgb.ndim != 5 or len(names) != rgb.shape[0]:
            raise ValueError('Expected RGB [B,T,C,H,W] aligned with the video names.')
        batch_size, window_len = int(rgb.shape[0]), int(rgb.shape[1])
        if torch.is_tensor(padding_valid):
            if tuple(padding_valid.shape) != (batch_size, window_len):
                raise ValueError('Padding validity mask must match RGB [B,T].')
            valid_window = padding_valid[:, -1].detach().cpu().bool()
        elif padding_valid is None:
            valid_window = torch.ones(batch_size, dtype=torch.bool)
        else:
            raise ValueError('Padding validity must be a tensor or None.')
        stats['windows_seen'] += batch_size
        stats['invalid_target_windows_skipped'] += int((~valid_window).sum())
        if not valid_window.any():
            cursor += batch_size
            continue
        model_input = rgb_batch.to(model.input_device, non_blocking=True)
        model_out = model(model_input, return_details=True, return_concept_losses=False)
        stage = model_out['concept_out']['stage4']
        measurements, active = prototype_measurements(stage, concept.visual_concepts)
        shape, row_of_patch = last_frame_rows(stage, active.shape[0])
        if shape['B'] != batch_size:
            raise ValueError('Concept batch size disagrees with the RGB batch size.')
        grid_hw = (shape['H'], shape['W'])
        predicted, ground_truth = compute_patch_scores(
            adapter, model_out, sal_batch, resize_to, grid_hw, batch_size)
        for b in range(batch_size):
            if not bool(valid_window[b]):
                continue
            dataset_index = int(loader.dataset.indices[cursor + b])
            video = str(names[b])
            source_video, start = dataset.windows[dataset_index]
            if not video.strip() or str(source_video) != video:
                raise ValueError('Loader video name disagrees with the dataset window.')
            start = int(start)
            absolute_frame_index = start + window_len - 1
            frame_names = dataset.video_frame_names[video]
            if not 0 <= absolute_frame_index < len(frame_names):
                raise ValueError('Last-frame index outside the source video.')
            frame_filename = str(frame_names[absolute_frame_index])
            label = Path(frame_filename).stem
            frame_no = int(label) if label.isdecimal() else label
            videos.add(video)
            stats['windows_exported'] += 1
            for patch in range(grid_hw[0] * grid_hw[1]):
                row = int(row_of_patch[b, patch])
                prototypes, active_count = top_active_prototypes(measurements, row)
                active_counts[active_count] += 1
                stats['patches_with_fewer_than_20_active_prototypes'] += int(active_count < TOP_K)
                score = float(predicted[b, patch])
                if stats['patch_count'] == 0:
                    stats['predicted_patch_saliency_min'] = score
                    stats['predicted_patch_saliency_max'] = score
                else:
                    stats['predicted_patch_saliency_min'] = min(stats['predicted_patch_saliency_min'], score)
                    stats['predicted_patch_saliency_max'] = max(stats['predicted_patch_saliency_max'], score)
                record = {
                    'video_fname': video, 'frame_no': frame_no,
                    'frame_filename': frame_filename, 'absolute_frame_index': absolute_frame_index,
                    'dataset_index': dataset_index, 'window_start_index': start,
                    'window_length': window_len, 'target_window_offset': window_len - 1, 'stage': 4,
                    'patch_index': patch, 'grid_row': patch // grid_hw[1],
                    'grid_column': patch % grid_hw[1], 'feature_grid_hw': list(grid_hw),
                    'box_model_xyxy': bounds(patch, grid_hw, resize_to),
                    'box_native_xyxy': bounds(patch, grid_hw, rgb.shape[-2:]),
                    'patch_pred_saliency': score, 'patch_gt_saliency': float(ground_truth[b, patch]),
                    'top_20_activated_prototype_indices': [item['prototype_index'] for item in prototypes],
                    'top_20_activated_prototypes': prototypes,
                }
                if not first_record:
                    out.write(',\n')
                out.write(json.dumps(record, allow_nan=False, separators=(',', ':')))
                first_record = False
                stats['patch_count'] += 1
        cursor += batch_size
        if batch_number == 0 or (batch_number + 1) % 20 == 0:
            print(f'Windows {cursor}/{len(loader.dataset)}; exported patches {stats["patch_count"]}', flush=True)
        del model_out, model_input, stage, measurements, active, predicted, ground_truth
    if cursor != len(loader.dataset):
        raise RuntimeError('Loader did not traverse every selected window.')
    out.write('\n]\n')
    if stats['patch_count'] == 0:
        raise ValueError('No patches exported: dataset is empty or all target frames are padding.')
    stats['videos_exported'] = len(videos)
    stats['active_prototype_count_histogram'] = dict(sorted(active_counts.items()))
    return stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--gallery-script',
        default=str(Path(__file__).resolve().parents[1]
                    / 'exp2a_visual_gallery' / 'across_videos' / 'prototype_to_patch.py'),
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-dir', required=True, help='Validation dataset directory (not the train directory).')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR))
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--window-len', type=int, default=32)
    parser.add_argument('--stride', type=int, default=1, help='1 scans every available sliding window.')
    parser.add_argument('--max-windows', type=int, default=0, help='0 = all; positive values are for smoke runs.')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite-results', action='store_true', help='Replace only this experiment\'s two JSON outputs.')
    args = parser.parse_args(argv)
    if min(args.window_len, args.stride, args.batch_size) < 1:
        parser.error('window-len, stride and batch-size must be positive.')
    if args.max_windows < 0 or args.num_workers < 0:
        parser.error('max-windows and num-workers must be nonnegative.')
    return args


def main():
    args = parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_path, metadata_path = root / 'patch_data.json', root / 'experiment_metadata.json'
    if not args.overwrite_results and any(path.exists() for path in (data_path, metadata_path)):
        raise SystemExit('Experiment output exists. Use another --output-dir or --overwrite-results.')
    adapter = load_gallery_adapter(args.gallery_script)
    adapter.set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = Path(adapter.resolve_checkpoint_path(args.checkpoint)).resolve()
    model = adapter.load_saliency_model(str(checkpoint), device)
    verify_checkpoint_parameters(model, checkpoint)
    dataset = adapter.DatasetLoader(args.dataset_dir, window_len=args.window_len, stride=args.stride)
    if not hasattr(dataset, 'windows') or not hasattr(dataset, 'video_frame_names'):
        raise TypeError('DatasetLoader must expose windows and video_frame_names.')
    count = min(args.max_windows, len(dataset)) if args.max_windows else len(dataset)
    if count == 0:
        raise ValueError('Validation dataset contains no windows.')
    subset = Subset(dataset, range(count))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, collate_fn=adapter.video_saliency_collate_fn,
                        pin_memory=device.type == 'cuda')
    print(f'Scanning {count}/{len(dataset)} validation windows; stride={args.stride}; '
          'all stage-4 last-frame patches.', flush=True)
    with tempfile.TemporaryDirectory(prefix='.prototype_consistency_sal_', dir=root) as tmp:
        tmp = Path(tmp)
        with (tmp / 'patch_data.json').open('w', encoding='utf-8') as out:
            stats = collect_patch_data(model, loader, adapter, out)
        concept = model.concept_creations['stage4']
        metadata = {
            'checkpoint': str(checkpoint), 'checkpoint_file_size': checkpoint.stat().st_size,
            'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns,
            'dataset_dir': str(Path(args.dataset_dir).expanduser().resolve()),
            'gallery_script': str(Path(args.gallery_script).expanduser().resolve()),
            'dataset_length': len(dataset), 'selected_window_count': count,
            'selected_indices_description': f'range(0, {count})',
            'settings': vars(args), 'collection': stats, 'stage': 4,
            'sampling': 'all_selected_loader_windows_last_frame',
            'saliency_extraction': 'existing gallery adapter predicted/GT concept-grid helpers',
            'saliency_binning': 'assigned later by overlap_analysis.py from patch_pred_saliency',
            'top_k': TOP_K,
            'prototype_ranking': 'descending forward_activation when multiple are active; otherwise descending assignment_probability',
            'prototype_tie_break': 'ascending prototype_index',
            'exported_top_k_rule': f'up to {TOP_K} ranked prototype slots per patch in top_20_activated_prototypes',
            'model_assignment_mode': str(getattr(concept, 'visual_assignment_mode', 'unknown')),
            'model_top_k_export_slots': int(getattr(concept, 'top_k', 0)),
            'model_assignment_temperature': float(getattr(concept, 'visual_assignment_temperature', 0)),
            'frame_no_definition': 'numeric filename stem when possible; otherwise string stem',
            'absolute_frame_index_definition': 'zero-based position in dataset.video_frame_names',
            'box_definition': 'nominal grid cell; xyxy with exclusive right/bottom, not full receptive field',
            'model_top_k_changed': False,
        }
        (tmp / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
        # Each file is replaced only after the full export succeeds.
        (tmp / 'patch_data.json').replace(data_path)
        (tmp / 'experiment_metadata.json').replace(metadata_path)
    print(f'Saved {stats["patch_count"]} patches to {data_path}', flush=True)
    fewer = stats['patches_with_fewer_than_20_active_prototypes']
    if fewer:
        print(f'{fewer} patches have fewer than 20 valid active prototypes; '
              'their lists contain only actual assignments. See experiment_metadata.json.', flush=True)


if __name__ == '__main__':
    main()
