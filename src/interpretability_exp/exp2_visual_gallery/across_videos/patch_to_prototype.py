#!/usr/bin/env python3
"""Find stage-4 patches matching manually selected semantic targets.

Scans every loader window (each window's last frame is a candidate time step),
restricted to highly salient grid cells by default, and compares cells to
group_{id}/target.jpg references. A video may contribute several patches, from
different frames. Each frame contributes at most one patch to a group: the
best-scoring cell on that frame.
Use --frame-sampling middle to restrict search to one middle-frame window per video.

Place beside prototype_to_patch.py, which supplies the project's model
and dataloader configuration. No model weights are changed. Every group member
must satisfy the similarity gates against the target. One frame cannot supply
two patches to the same group.
Targets require patch_selector.py EXIF metadata (or a target.json sidecar).
The original temporal window is rerun to recover each target's stage4 features.
Top-k cosine matches are explicitly distinct from actual forward activations.

Write results to the same directory that holds group_*/target.jpg seeds (or set
--target-groups-dir separately). Re-run with --overwrite-results to refresh exports.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


TARGET_GROUPS_DIR = '/data/quantization/zaima/videosal_datasets/dhf1k/visual_gallery/concepts_v3'


@dataclass
class Settings:
    num_groups: int = 4
    min_group_size: int = 4
    max_group_size: int = 8
    min_stage4_cosine: float = 0.90
    max_rgb_mae: float = 0.10
    min_rgb_structure_cosine: float = 0.90
    min_crop_spatial_std: float = 0.025
    descriptor_size: int = 16
    max_candidates: int = 4000
    patches_per_window: int = 12
    max_candidates_per_video: int = 64
    max_seed_trials: int = 500
    top_k: int = 10
    prototype_high_cosine: float = 0.70
    comparison_block: int = 256
    appearance_pair_chunk: int = 2048
    seed: int = 42
    salient_patches_only: bool = True
    saliency_source: str = 'predicted'
    saliency_filter_mode: str = 'gt_top_percent'
    saliency_threshold: float = 0.0
    saliency_top_percent: float = 0.30

    def validate(self):
        if not 2 <= self.min_group_size <= self.max_group_size:
            raise ValueError('Require 2 <= min_group_size <= max_group_size.')
        if not -1 <= self.min_stage4_cosine <= 1 or not -1 <= self.min_rgb_structure_cosine <= 1:
            raise ValueError('Cosine thresholds must be in [-1,1].')
        if not -1 <= self.prototype_high_cosine <= 1:
            raise ValueError('prototype_high_cosine must be in [-1,1].')
        if not 0 <= self.max_rgb_mae <= 1 or not 0 <= self.min_crop_spatial_std <= 1:
            raise ValueError('RGB thresholds must be in [0,1].')
        for key in ('num_groups','descriptor_size','max_candidates','patches_per_window',
                    'max_candidates_per_video','max_seed_trials','top_k',
                    'comparison_block','appearance_pair_chunk'):
            if getattr(self,key) < 1:
                raise ValueError(f'{key} must be positive.')
        if self.saliency_source not in ('predicted','gt'):
            raise ValueError('saliency_source must be predicted or gt.')
        if self.saliency_filter_mode not in ('none','gt_threshold','gt_top_percent','gt_weighted'):
            raise ValueError('Unknown saliency_filter_mode.')
        if not 0 < self.saliency_top_percent <= 1:
            raise ValueError('saliency_top_percent must be in (0,1].')
        if self.salient_patches_only and self.saliency_filter_mode == 'none':
            raise ValueError('salient_patches_only requires a saliency filter mode other than none.')


def compute_batch_patch_saliency(adapter,out,sal_batch,resize_to,grid_hw,saliency_source):
    """Patch-level saliency [B,N] on the stage-4 concept grid (last target frame)."""
    if saliency_source == 'predicted':
        return adapter.extract_predicted_patch_saliency(out,grid_hw)
    sal_maps = adapter.extract_gt_saliency_batch(sal_batch)
    sal_for_grid = adapter.prepare_saliency_maps_for_concept_grid(sal_maps,resize_to)
    return adapter.compute_patch_saliency_grid(sal_for_grid,grid_hw)


def salient_patch_mask(adapter,patch_saliency,settings):
    """Bool mask [B,N] using the same rules as prototype gallery retrieval."""
    b,n = patch_saliency.shape
    mode = settings.saliency_filter_mode if settings.salient_patches_only else 'none'
    dummy = torch.zeros(b,n,1)
    _, flat = adapter.build_saliency_retrieval_scores(
        dummy,patch_saliency,mode,settings.saliency_threshold,settings.saliency_top_percent)
    return flat.reshape(b,n)


def select_middle_frame_window_indices(dataset, window_len: int) -> list[int]:
    """
    One dataset index per video: a window whose last frame is the video middle frame.

    Visual concepts are assigned on the window's last backbone time step, so the
    loader window must end on the middle frame index M = (num_frames - 1) // 2.
    Requires stride=1 (or a window list that includes start = M - window_len + 1).
    """
    windows = getattr(dataset, 'windows', None)
    frame_names = getattr(dataset, 'video_frame_names', None)
    if windows is None or frame_names is None:
        raise TypeError('dataset must be a DatasetLoader with windows and video_frame_names.')

    chosen: dict[str, int] = {}
    for idx, (video_name, start) in enumerate(windows):
        names = frame_names.get(video_name)
        if not names:
            continue
        num_frames = len(names)
        if num_frames < window_len:
            continue
        middle = (num_frames - 1) // 2
        if int(start) + window_len - 1 != middle:
            continue
        if video_name not in chosen:
            chosen[video_name] = idx

    if not chosen:
        raise ValueError(
            'No loader windows end on each video middle frame. '
            f'Use --stride 1 and ensure window_len <= middle_frame_index + 1 '
            '(middle start = M - window_len + 1).'
        )
    return sorted(chosen.values())


def load_gallery_adapter(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('stage4_gallery_adapter', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import gallery adapter: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def bounds(patch_index, grid_hw, image_hw):
    gh, gw = grid_hw
    height, width = image_hw
    if not 0 <= patch_index < gh*gw:
        raise ValueError('Patch index outside feature grid.')
    row, column = divmod(patch_index, gw)
    return (round(column*width/gw), round(row*height/gh),
            round((column+1)*width/gw), round((row+1)*height/gh))


def crop_descriptor(frame, patch_index, grid_hw, size):
    x0,y0,x1,y1 = bounds(patch_index,grid_hw,frame.shape[-2:])
    crop = frame[:,y0:y1,x0:x1]
    if crop.numel() == 0:
        raise ValueError('Empty crop: grid exceeds frame resolution.')
    thumbnail = F.interpolate(crop[None].float(),size=(size,size),mode='bilinear',
                              align_corners=False)[0]
    # Per-channel spatial variance excludes spatially blank colored surfaces.
    spatial_std = thumbnail.flatten(1).std(dim=1,unbiased=False).mean().item()
    centered = thumbnail - thumbnail.mean(dim=(-2,-1),keepdim=True)
    structural = F.normalize(centered.flatten(),dim=0)
    return thumbnail.flatten(), structural, spatial_std


def batch_crop_descriptors(frames, grid_hw, size):
    """RGB thumbnails for every stage-4 cell.

    frames: [B, 3, H, W]. Returns flat RGB [B, N, 3*S*S], L2 structure [B, N, 3*S*S],
    and per-cell spatial std [B, N]. Same bilinear resize as crop_descriptor, batched
    over cells that share a crop size.
    """
    if frames.dim() != 4 or int(frames.shape[1]) != 3:
        raise ValueError(f'Expected frames [B,3,H,W], got {tuple(frames.shape)}.')
    size = int(size)
    if size < 1:
        raise ValueError('descriptor size must be positive.')
    batch, _, height, width = frames.shape
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    n_patches = gh * gw
    boxes = [bounds(patch, (gh, gw), (height, width)) for patch in range(n_patches)]
    groups: dict[tuple[int, int], list[int]] = {}
    for patch, (x0, y0, x1, y1) in enumerate(boxes):
        groups.setdefault((y1 - y0, x1 - x0), []).append(patch)
    flat = frames.new_empty(batch, n_patches, 3 * size * size)
    for (crop_h, crop_w), patches in groups.items():
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError('Empty crop: grid exceeds frame resolution.')
        crops = torch.stack(
            [frames[:, :, boxes[patch][1]:boxes[patch][3], boxes[patch][0]:boxes[patch][2]]
             for patch in patches],
            dim=1,
        )
        thumbs = F.interpolate(
            crops.reshape(batch * len(patches), 3, crop_h, crop_w).float(),
            size=(size, size), mode='bilinear', align_corners=False,
        )
        flat[:, patches] = thumbs.reshape(batch, len(patches), -1).to(dtype=flat.dtype)
    thumbs = flat.reshape(batch, n_patches, 3, size, size)
    spatial_std = thumbs.reshape(batch, n_patches, 3, size * size).std(dim=-1, unbiased=False).mean(-1)
    centered = thumbs - thumbs.mean(dim=(-2, -1), keepdim=True)
    structure = F.normalize(centered.reshape(batch, n_patches, -1), dim=-1)
    return flat, structure, spatial_std


def frame_key(metadata):
    """Video plus absolute frame index. One group may take only one patch from this key."""
    return (str(metadata['video_id']), int(metadata['absolute_frame_index']))


def reference_bank(references, device):
    """Stack target features once for a batched CUDA (or CPU) gate."""
    return {
        'feature': torch.stack([item['feature'] for item in references]).to(device),
        'rgb': torch.stack([item['rgb'] for item in references]).to(device),
        'structure': torch.stack([item['structure'] for item in references]).to(device),
        'video_id': [item['metadata']['video_id'] for item in references],
        'frame_key': [frame_key(item['metadata']) for item in references],
    }


def gate_against_references(features, rgb, structure, bank, settings):
    """Appearance gates for every cell against every target.

    features/rgb/structure are [B, N, D]. Returns ok [B, N, R] and finite [B, N].
    Same-video rejection is applied by the caller.
    """
    finite = torch.isfinite(features).all(dim=-1) & (features.norm(dim=-1) >= 1e-8)
    cosine = F.normalize(features, dim=-1) @ bank['feature'].T
    mae = (rgb.unsqueeze(2) - bank['rgb']).abs().mean(dim=-1)
    struct = structure @ bank['structure'].T
    ok = ((cosine >= settings.min_stage4_cosine)
          & (mae <= settings.max_rgb_mae)
          & (struct >= settings.min_rgb_structure_cosine)
          & finite.unsqueeze(-1))
    return ok, finite


def valid_active_matrix(stage, rows, prototype_count, device):
    """Actual active indices plus boolean mask, never inferred from top-k scores."""
    indices = stage.get('active_visual_prototype_indices')
    valid = stage.get('visual_validity_mask')
    if not torch.is_tensor(indices) or not torch.is_tensor(valid) or valid.dtype != torch.bool:
        raise ValueError('Model must export active_visual_prototype_indices and bool visual_validity_mask.')
    if indices.shape != valid.shape or indices.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8):
        raise ValueError('Invalid active index/mask export shape or index dtype.')
    meta = stage['visual_metadata']
    shape = meta['feature_shape']
    if indices.dim() == 5:
        if tuple(indices.shape[:4]) != tuple(shape[k] for k in ('B','T','H','W')):
            raise ValueError('Active exports do not align with the concept feature grid.')
        b = meta['batch_idx'].to(indices.device).long()
        t = meta['time_idx'].to(indices.device).long()
        p = meta['patch_idx'].to(indices.device).long()
        indices = indices[b,t,p//shape['W'],p%shape['W']]
        valid = valid[b,t,p//shape['W'],p%shape['W']]
    if indices.dim()!=2 or indices.shape[0]!=rows or indices.shape[1]<1:
        raise ValueError('Active exports must be [B,T,H,W,K] or [rows,K].')
    indices,valid = indices.to(device,dtype=torch.long),valid.to(device)
    if (((indices<0)|(indices>=prototype_count)) & valid).any():
        raise ValueError('Valid prototype index outside the bank.')
    result = torch.zeros(rows,prototype_count,dtype=torch.bool,device=device)
    row_ids = torch.arange(rows,device=device)[:,None].expand_as(indices)
    result[row_ids[valid],indices[valid]] = True
    return result


def prototype_measurements(stage, prototype_bank):
    """Return true cosine, scaled logits, probabilities, forward activations, validity."""
    q = stage['visual_patch_embeddings'].detach().float()
    bank = prototype_bank.detach().to(q.device).float()
    if q.dim()!=2 or bank.dim()!=2 or q.shape[1]!=bank.shape[1]:
        raise ValueError('Patch/prototype embedding dimensions do not match.')
    cosine = F.normalize(q,dim=-1) @ F.normalize(bank,dim=-1).T
    logits = stage['visual_concept_logits'].detach().float()
    probabilities = stage['visual_assignment_probs'].detach().float()
    activation = stage['visual_activations'].detach().float()
    for value in (cosine,logits,probabilities,activation):
        if value.shape != cosine.shape or not torch.isfinite(value).all():
            raise ValueError('Nonfinite or misaligned prototype measurements.')
    valid = valid_active_matrix(stage,q.shape[0],bank.shape[0],q.device)
    return {name:value.cpu() for name,value in (
        ('cosine',cosine),('logits',logits),('probabilities',probabilities),
        ('activation',activation),('valid_active',valid))}


class CandidatePool:
    """Stack retained patch data using the original analysis tensor layout."""
    def tensors(self):
        if not self.items:
            raise ValueError('No usable candidates; check inputs/crop-variance filter.')
        fields = ('feature','rgb','structure','cosine','logits','probabilities','activation','valid_active')
        result = {field:torch.stack([item[field] for item in self.items]) for field in fields}
        result['metadata'] = [item['metadata'] for item in self.items]
        return result


def discover_target_groups(root):
    """Read every immediate group_<integer>/target.jpg, in numeric ID order."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Target groups directory does not exist: {root}')
    folders = [(int(match.group(1)), path) for path in root.iterdir()
               if path.is_dir() and (match := re.fullmatch(r'group_(\d+)', path.name))]
    folders.sort(key=lambda pair: (pair[0], pair[1].name))
    if not folders:
        raise ValueError(f'No group_<integer> folders found in {root}')
    if len({number for number, _ in folders}) != len(folders):
        raise ValueError('Duplicate numeric group IDs (for example group_1 and group_01).')
    targets = []
    for number, folder in folders:
        path = folder / 'target.jpg'
        if not path.is_file():
            raise FileNotFoundError(f'Missing target image: {path}')
        with Image.open(path) as image:
            embedded = image.getexif().get(270)
        sidecar = folder / 'target.json'
        if embedded:
            metadata = json.loads(embedded)
        elif sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding='utf-8'))
        else:
            raise ValueError(f'{path} has no patch_selector EXIF metadata. Copy the original '
                             'JPG without re-encoding, or put its metadata record in target.json.')
        if not isinstance(metadata, dict) or metadata.get('stage') != 4:
            raise ValueError(f'{path}: target metadata must describe a stage-4 patch.')
        required = {'patch_index', 'feature_grid_hw', 'model_input_hw'}
        if not required.issubset(metadata) or not (metadata.get('video_id') or metadata.get('video_fname')):
            raise ValueError(f'{path}: incomplete source patch metadata.')
        if not metadata.get('frame_filename') and metadata.get('frame_no') is None:
            raise ValueError(f'{path}: metadata must identify the source frame filename/number.')
        targets.append({'group_id': number, 'group_name': folder.name,
                        'target_path': str(path), 'selector_metadata': metadata})
    return targets


TARGET_SEED_FILES = frozenset({'target.jpg', 'target.json'})
EXPERIMENT_ROOT_FILES = frozenset({
    'index.html', 'experiment_summary.json', 'target_groups.json',
    'patch_prototype_activations.csv', 'candidate_metadata.json',
    'topk_overlap_baseline.json', 'candidate_features_and_scores.pt',
})
def validate_output_workspace(root, group_names, *, overwrite_results=False):
    """Allow output roots that only contain target seeds (group_*/target.jpg)."""
    root = Path(root)
    if not root.exists():
        return
    allowed_dirs = set(group_names)
    for entry in root.iterdir():
        if entry.is_dir() and entry.name in allowed_dirs:
            group_dir = entry
            for child in group_dir.iterdir():
                if child.name in TARGET_SEED_FILES:
                    continue
                if overwrite_results:
                    continue
                hint = ' Pass --overwrite-results to replace prior group exports.'
                if child.is_dir():
                    raise SystemExit(f'{group_dir} already contains {child.name}.{hint}')
                raise SystemExit(f'{group_dir} already contains {child.name}.{hint}')
            continue
        if entry.name in EXPERIMENT_ROOT_FILES:
            if overwrite_results:
                continue
            raise SystemExit(
                f'Prior experiment output exists: {entry}. Pass --overwrite-results or use another directory.'
            )
        if entry.is_dir():
            raise SystemExit(
                f'Unexpected directory in output root: {entry.name}. '
                'Only group_<integer>/ seed folders are allowed before the first run.'
            )
        raise SystemExit(
            f'Unexpected file in output root: {entry.name}. '
            'Only group_<integer>/ seed folders are allowed before the first run.'
        )


def prepare_group_output_dir(group_dir, reference, *, overwrite_results=False):
    """Ensure group folder exists; preserve existing target.jpg when it is the source."""
    group_dir = Path(group_dir)
    group_dir.mkdir(parents=True, exist_ok=True)
    target_path = (group_dir / 'target.jpg').resolve()
    source = Path(reference['selected_target_path']).resolve()
    if source != target_path:
        if target_path.is_file() and target_path.read_bytes() != source.read_bytes():
            raise ValueError(
                f'{target_path} differs from the selected target source {source}. '
                'Remove the file or point --target-groups-dir at the correct seed.'
            )
        if not target_path.is_file():
            shutil.copyfile(source, target_path)
    if overwrite_results:
        for child in list(group_dir.iterdir()):
            if child.name == 'target.jpg':
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


def resolve_target_window(dataset, metadata, window_len):
    """Pick a loader window that contains the target frame (any temporal offset).

    Prefers a window whose last frame equals the target (training-style alignment).
    Otherwise uses the window with the latest start index that still contains the frame.
    """
    video = str(metadata.get('video_id') or metadata['video_fname'])
    names = dataset.video_frame_names.get(video)
    if not names:
        raise ValueError(f'Target video {video!r} is not in the supplied dataset.')
    filename = metadata.get('frame_filename')
    label = str(metadata.get('frame_no', ''))
    matches = [i for i, name in enumerate(names)
               if (Path(str(name)).name == filename if filename else Path(str(name)).stem == label)]
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely resolve target frame {filename or label!r} in video {video}.')
    frame_index = matches[0]
    candidates: list[tuple[tuple, int, int, int]] = []
    for dataset_index, (name, offset) in enumerate(dataset.windows):
        if str(name) != video:
            continue
        start = int(offset)
        if not (start <= frame_index < start + window_len):
            continue
        target_offset = frame_index - start
        prefer_last = target_offset == window_len - 1
        candidates.append(((prefer_last, start, -dataset_index), dataset_index, target_offset, start))
    if not candidates:
        raise ValueError(
            f'No loader window of length {window_len} contains {video}/{filename or label}. '
            f'Ensure the frame is in the dataset and has saliency/fixation maps for some valid window.'
        )
    candidates.sort(key=lambda row: row[0], reverse=True)
    _, dataset_index, target_offset, _start = candidates[0]
    return dataset_index, frame_index, target_offset


def source_metadata(dataset, dataset_index, video_id, rgb_window, patch, grid_hw,
                    resize_to, crop_std, saliency, is_salient, rgb_window_hash=None,
                    target_window_offset=None):
    start = int(dataset.windows[dataset_index][1])
    offset = (int(rgb_window.shape[0]) - 1 if target_window_offset is None
              else int(target_window_offset))
    absolute_frame_index = start + offset
    gh, gw = grid_hw
    y, x = divmod(patch, gw)
    return {'video_id': video_id, 'dataset_index': dataset_index,
            'sample_id': f'{video_id}::frame={absolute_frame_index}',
            'target_window_offset': offset,
            'absolute_frame_index': absolute_frame_index,
            'frame_filename': str(dataset.video_frame_names[video_id][absolute_frame_index]),
            'stage': 4, 'video_frame_count': len(dataset.video_frame_names[video_id]),
            'window_length': int(rgb_window.shape[0]), 'native_input_shape': list(rgb_window.shape),
            'rgb_window_sha256': rgb_window_hash or tensor_digest(rgb_window), 'feature_grid_hw': [gh, gw],
            'patch_index': patch, 'grid_row': y, 'grid_column': x,
            'box_model_xyxy': list(bounds(patch, grid_hw, resize_to)),
            'box_native_xyxy': list(bounds(patch, grid_hw, rgb_window.shape[-2:])),
            'crop_spatial_std': crop_std, 'patch_pred_saliency': saliency,
            'is_salient_region': is_salient}


@torch.inference_mode()
def load_target_references(model, dataset, adapter, settings, specs, window_len):
    """Compute targets in full-frame temporal context, never from cropped JPGs."""
    concept = model.concept_creations['stage4']
    resize_to = tuple(adapter.resolve_backbone_spatial_hw(model))
    references, captured = [], {}
    capture_time = {'offset': 0}
    def capture(module, args):
        captured['grid'] = args[0][:, :, capture_time['offset']].detach().float().cpu()
    hook = concept.register_forward_pre_hook(capture)
    try:
        model.eval()
        for spec in specs:
            selected = spec['selector_metadata']
            index, frame_index, target_offset = resolve_target_window(dataset, selected, window_len)
            names, batch, _, sal_batch, _, padding_valid = adapter.video_saliency_collate_fn([dataset[index]])
            rgb = adapter.rgb_video_to_btchw(batch)
            if (torch.is_tensor(padding_valid) and padding_valid.shape[1] > target_offset
                    and not bool(padding_valid[0, target_offset])):
                raise ValueError(f'{spec["group_name"]}: target frame is padding.')
            resized = adapter.preprocess_rgb_video_like_backbone(rgb, resize_to)
            capture_time['offset'] = target_offset
            out = model(batch.to(model.input_device), return_details=True,
                        return_concept_losses=False, use_reference_cache=False)
            grid = captured.pop('grid')
            stage = out['concept_out']['stage4']
            shape = stage['visual_metadata']['feature_shape']
            grid_hw = (int(shape['H']), int(shape['W']))
            if (tuple(grid.shape[-2:]) != grid_hw or tuple(selected['feature_grid_hw']) != grid_hw
                    or tuple(selected['model_input_hw']) != resize_to):
                raise ValueError(f'{spec["group_name"]}: selector grid/input size disagrees with loaded model.')
            patch = int(selected['patch_index'])
            box = bounds(patch, grid_hw, resize_to)
            if selected.get('box_model_xyxy') is not None and list(box) != selected['box_model_xyxy']:
                raise ValueError(f'{spec["group_name"]}: selector crop coordinates disagree with its patch index.')
            if selected.get('native_image_hw') is not None and list(rgb.shape[-2:]) != selected['native_image_hw']:
                raise ValueError(f'{spec["group_name"]}: source frame resolution changed.')
            actual_crop = image_from_tensor(resized[0, target_offset]).crop(box)
            with Image.open(spec['target_path']) as image:
                selected_crop = image.convert('RGB')
                if selected_crop.size != actual_crop.size:
                    raise ValueError(f'{spec["group_name"]}: target.jpg dimensions do not match its source cell.')
                mae = float(np.abs(np.asarray(selected_crop, dtype=np.float32) -
                                   np.asarray(actual_crop, dtype=np.float32)).mean() / 255)
            if mae > 0.05:
                raise ValueError(f'{spec["group_name"]}: target.jpg does not match the source cell '
                                 f'(RGB MAE={mae:.4f}). Check source metadata/preprocessing.')
            meta = stage['visual_metadata']
            row = ((meta['batch_idx'].cpu() == 0) &
                   (meta['time_idx'].cpu() == target_offset) &
                   (meta['patch_idx'].cpu() == patch)).nonzero(as_tuple=False).flatten()
            if row.numel() != 1:
                raise ValueError('Target patch must have exactly one concept row at the selected time step.')
            row = int(row[0])
            y, x = divmod(patch, grid_hw[1])
            feature = grid[0, :, y, x]
            descriptor, structure, std = crop_descriptor(resized[0, target_offset], patch, grid_hw,
                                                         settings.descriptor_size)
            if not torch.isfinite(feature).all() or feature.norm() < 1e-8 or structure.norm() < 1e-8:
                raise ValueError(f'{spec["group_name"]}: target has invalid/zero features or flat RGB structure.')
            measurements = prototype_measurements(stage, concept.visual_concepts)
            saliency = compute_batch_patch_saliency(adapter, out, sal_batch, resize_to, grid_hw,
                                                   settings.saliency_source)
            salient = salient_patch_mask(adapter, saliency, settings)
            source = source_metadata(dataset, index, str(names[0]), rgb[0], patch, grid_hw,
                                     resize_to, std, float(saliency[0, patch]), bool(salient[0, patch]),
                                     target_window_offset=target_offset)
            if source['absolute_frame_index'] != frame_index:
                raise ValueError('Resolved window does not align with the selected source frame index.')
            source.update({'is_target_reference': True, 'source_group_id': spec['group_id'],
                           'source_group_name': spec['group_name'], 'selected_target_path': spec['target_path'],
                           'selector_metadata': selected, 'selected_jpeg_source_mae': mae})
            references.append({'feature': F.normalize(feature, dim=0), 'rgb': descriptor,
                               'structure': structure, **{k: v[row].clone() for k, v in measurements.items()},
                               'metadata': source})
            print(f'Loaded {spec["group_name"]}: {source["video_id"]}/{source["frame_filename"]}, '
                  f'patch {patch}', flush=True)
    finally:
        hook.remove()
    keys = [(r['metadata']['dataset_index'], r['metadata']['patch_index']) for r in references]
    if len(set(keys)) != len(keys):
        raise ValueError('The same source patch was selected for multiple groups.')
    return references


def target_similarity(item, reference):
    return {'stage4_cosine': float(item['feature'] @ reference['feature']),
            'rgb_mae': float((item['rgb'] - reference['rgb']).abs().mean()),
            'rgb_structure_cosine': float(item['structure'] @ reference['structure'])}


def passes_target_gates(scores, settings):
    return (scores['stage4_cosine'] >= settings.min_stage4_cosine and
            scores['rgb_mae'] <= settings.max_rgb_mae and
            scores['rgb_structure_cosine'] >= settings.min_rgb_structure_cosine)


def similarity_sort_key(scores):
    return (scores['stage4_cosine'], -scores['rgb_mae'], scores['rgb_structure_cosine'])


class TargetCandidatePool:
    """Best matching cell per frame; a video may contribute one patch from each frame.

    Scan every eligible grid cell in each loader window (last-frame stage-4 grid).
    Per target, track the best match for each frame. The target's own frame is
    skipped for that target because the group already includes it. max_candidates
    caps how many distinct frames are retained per target before global dedup.
    """
    tensors = CandidatePool.tensors

    def __init__(self, settings, references):
        self.settings, self.references = settings, references
        self.limit = settings.max_candidates // len(references)
        if self.limit < settings.max_group_size - 1:
            raise ValueError('Increase --max-candidates to at least number_of_targets * (max_group_size - 1).')
        self.matches = [{} for _ in references]
        self.seen = 0
        self.target_keys = {(r['metadata']['dataset_index'], r['metadata']['patch_index']) for r in references}

    def add(self, item):
        if (item['metadata']['dataset_index'], item['metadata']['patch_index']) in self.target_keys:
            return
        key = frame_key(item['metadata'])
        accepted = False
        for reference_index, reference in enumerate(self.references):
            if key == frame_key(reference['metadata']):
                continue
            scores = target_similarity(item, reference)
            if not passes_target_gates(scores, self.settings):
                continue
            accepted = True
            rank = (*similarity_sort_key(scores), -item['metadata']['dataset_index'], -item['metadata']['patch_index'])
            matches = self.matches[reference_index]
            if key not in matches or rank > matches[key][0]:
                stored = {**item, 'metadata': {**item['metadata'],
                                                'best_target_reference_index': reference_index,
                                                **{f'target_{k}': v for k, v in scores.items()}}}
                matches[key] = (rank, stored)
                if len(matches) > self.limit:
                    del matches[min(matches, key=lambda name: matches[name][0])]
        self.seen += int(accepted)

    @property
    def items(self):
        result = list(self.references)
        used_keys = set(self.target_keys)
        ranked = []
        for matches in self.matches:
            ranked.extend(matches.values())
        for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True):
            key = (item['metadata']['dataset_index'], item['metadata']['patch_index'])
            if key in used_keys:
                continue
            result.append(item)
            used_keys.add(key)
        return result


def select_target_groups(pool, settings, reference_count):
    """Target is member zero. Several frames from one video are allowed; one patch per frame."""
    used = set(range(reference_count))
    groups, reports = [], []
    for reference_index in range(reference_count):
        reference = {key: pool[key][reference_index] for key in ('feature', 'rgb', 'structure')}
        source = pool['metadata'][reference_index]
        ranked = []
        for index in range(reference_count, len(pool['metadata'])):
            item = {key: pool[key][index] for key in ('feature', 'rgb', 'structure')}
            scores = target_similarity(item, reference)
            if passes_target_gates(scores, settings):
                ranked.append((similarity_sort_key(scores), index))
        ranked.sort(key=lambda pair: (pair[0], -pair[1]), reverse=True)
        members, frames = [reference_index], {frame_key(source)}
        for _, index in ranked:
            key = frame_key(pool['metadata'][index])
            if index in used or key in frames:
                continue
            members.append(index)
            frames.add(key)
            if len(members) == settings.max_group_size:
                break
        formed = len(members) >= settings.min_group_size
        reports.append({'group_id': source['source_group_id'], 'group_name': source['source_group_name'],
                        'target_path': source['selected_target_path'], 'target_candidate_index': reference_index,
                        'threshold_matching_candidates': len(ranked), 'selected_member_count': len(members),
                        'group_formed': formed, 'members': members if formed else [],
                        'reason': None if formed else 'Too few available matching frames after disjoint selection.'})
        if formed:
            validate_group(pool, members, settings)
            groups.append(members)
            used.update(members)
        print(f'{source["source_group_name"]}: {len(ranked)} matching candidates; '
              f'{len(members)} members including target; formed={formed}', flush=True)
    return groups, reports


@torch.inference_mode()
def collect_candidates(model,loader,adapter,settings,references,sampling='one_middle_frame_per_video',
                       similarity_device=None):
    if 'stage4' not in model.concept_creations:
        raise ValueError('The loaded model has no stage4 concept module.')
    concept = model.concept_creations['stage4']
    if not isinstance(loader.dataset,Subset):
        raise TypeError('Use a Subset loader so source dataset indices remain recoverable.')
    sim_device = torch.device(model.input_device if similarity_device in (None, 'auto') else similarity_device)
    bank = reference_bank(references, sim_device)
    print(f'Matching retained patches on {sim_device}.', flush=True)
    captured = {}
    def capture(module,args):
        # Exactly the backbone feature grid used for stage4 concept assignment,
        # before the trainable concept encoder and before prototype comparison.
        value = args[0]
        captured['stage4'] = value[:,:, -1].detach().to(device=sim_device, dtype=torch.float32).clone()
    hook = concept.register_forward_pre_hook(capture)
    pool = TargetCandidatePool(settings,references)
    cursor = 0
    stats = {'videos_seen':0,'windows_seen':0,'invalid_target_windows_skipped':0,'candidate_patches_scanned':0,
             'nonsalient_patches_skipped':0,'low_variance_crops_skipped':0,
             'sampling':sampling, 'selection':'one_best_patch_per_frame',
             'all_eligible_grid_cells_scanned':True, 'target_reference_count':len(references),
             'one_patch_per_frame_per_group':True, 'multiple_frames_per_video':True,
             'similarity_device':str(sim_device)}
    resize_to = adapter.resolve_backbone_spatial_hw(model)
    videos_seen = set()
    try:
        model.eval()
        for batch_number,batch in enumerate(loader):
            if not isinstance(batch,(tuple,list)) or len(batch)!=6:
                raise ValueError('Expected existing six-field video_saliency_collate_fn output.')
            names,rgb_batch,_,sal_batch,n_frames,padding_valid = batch
            rgb = adapter.rgb_video_to_btchw(rgb_batch)
            if len(names)!=rgb.shape[0]:
                raise ValueError('Video names and RGB batch size differ.')
            resized = adapter.preprocess_rgb_video_like_backbone(rgb,resize_to)
            model_input = rgb_batch.to(model.input_device,non_blocking=True)
            out = model(model_input,return_details=True,return_concept_losses=False,
                        use_reference_cache=False)
            if 'stage4' not in captured:
                raise RuntimeError('Stage4 concept input capture did not run.')
            feature_grid = captured.pop('stage4')
            stage = out['concept_out']['stage4']
            meta = stage['visual_metadata']
            shape = meta['feature_shape']
            B,H,W = int(shape['B']),int(shape['H']),int(shape['W'])
            if feature_grid.shape[0]!=B or tuple(feature_grid.shape[-2:])!=(H,W):
                raise ValueError('Captured stage4 and concept grids do not align.')
            grid_hw = (H,W)
            n_patches = H*W
            patch_saliency = compute_batch_patch_saliency(
                adapter,out,sal_batch,resize_to,grid_hw,settings.saliency_source)
            if patch_saliency.shape != (B,n_patches):
                raise ValueError('Patch saliency grid does not match stage-4 feature grid.')
            salient_mask = salient_patch_mask(adapter,patch_saliency,settings)
            batch_ids = meta['batch_idx'].detach().cpu().long()
            time_ids = meta['time_idx'].detach().cpu().long()
            patch_ids = meta['patch_idx'].detach().cpu().long()
            target_rows = time_ids == int(shape['T'])-1
            row_of_patch = torch.full((B, n_patches), -1, dtype=torch.long)
            row_index = target_rows.nonzero(as_tuple=False).flatten()
            row_of_patch[batch_ids[row_index], patch_ids[row_index]] = row_index
            valid_window = torch.ones(B, dtype=torch.bool)
            for b in range(B):
                stats['windows_seen'] += 1
                if (torch.is_tensor(padding_valid) and padding_valid.shape==(B,rgb.shape[1])
                        and not bool(padding_valid[b,-1])):
                    stats['invalid_target_windows_skipped'] += 1
                    valid_window[b] = False
                    continue
                video_id = str(names[b])
                videos_seen.add(video_id)
                if not video_id.strip():
                    raise ValueError('Video IDs must be nonempty and stable across windows.')
            present = (row_of_patch >= 0) & valid_window[:, None]
            if settings.salient_patches_only:
                salient = salient_mask.bool()
                stats['nonsalient_patches_skipped'] += int((present & ~salient).sum())
                scanned = present & salient
            else:
                scanned = present
            stats['candidate_patches_scanned'] += int(scanned.sum())
            frames = resized[:, -1].to(device=sim_device, dtype=torch.float32, non_blocking=True)
            rgb_desc, structure, spatial_std = batch_crop_descriptors(frames, grid_hw, settings.descriptor_size)
            features = feature_grid.permute(0, 2, 3, 1).reshape(B, n_patches, -1)
            low_var = (spatial_std < settings.min_crop_spatial_std) | (structure.norm(dim=-1) < 1e-8)
            ok, finite = gate_against_references(features, rgb_desc, structure, bank, settings)
            scanned_dev = scanned.to(device=sim_device)
            stats['low_variance_crops_skipped'] += int((scanned_dev & finite & low_var).sum())
            ok = ok & scanned_dev.unsqueeze(-1) & ~low_var.unsqueeze(-1)
            search_frames = []
            for b in range(B):
                if not bool(valid_window[b]):
                    search_frames.append(None)
                    continue
                dataset_index = int(loader.dataset.indices[cursor+b])
                start = int(loader.dataset.dataset.windows[dataset_index][1])
                search_frames.append((str(names[b]), start + int(rgb.shape[1]) - 1))
            for reference_index, ref_frame in enumerate(bank['frame_key']):
                same_frame = torch.tensor([search_frames[b] == ref_frame for b in range(B)], device=sim_device)
                ok[same_frame, :, reference_index] = False
            hit = ok.any(dim=-1).detach().cpu()
            if bool(hit.any()):
                measurement = prototype_measurements(stage, concept.visual_concepts)
                rgb_hash = {}
                hit_b, hit_p = hit.nonzero(as_tuple=False).unbind(dim=1)
                for b_tensor, patch_tensor in zip(hit_b.tolist(), hit_p.tolist()):
                    b, patch = int(b_tensor), int(patch_tensor)
                    row = int(row_of_patch[b, patch])
                    dataset_index = int(loader.dataset.indices[cursor+b])
                    if b not in rgb_hash:
                        rgb_hash[b] = tensor_digest(rgb[b])
                    feature = features[b, patch]
                    item = {'feature': F.normalize(feature, dim=0).cpu(),
                            'rgb': rgb_desc[b, patch].cpu(),
                            'structure': structure[b, patch].cpu(),
                            **{key: value[row].clone() for key, value in measurement.items()},
                            'metadata': source_metadata(
                                loader.dataset.dataset, dataset_index, str(names[b]), rgb[b],
                                patch, grid_hw, resize_to, float(spatial_std[b, patch]),
                                float(patch_saliency[b, patch]), bool(salient_mask[b, patch]), rgb_hash[b])}
                    pool.add(item)
                del measurement
            cursor += B
            if batch_number==0 or (batch_number+1)%20==0:
                print(f'Windows {cursor}/{len(loader.dataset)}; retained patches {len(pool.items)}',flush=True)
            del out,model_input,feature_grid,rgb_desc,structure,ok
    finally:
        hook.remove()
    stats['target_matching_patches_seen'] = pool.seen
    stats['videos_seen'] = len(videos_seen)
    stats['retained_match_count_per_target'] = [len(matches) for matches in pool.matches]
    stats['retained_patches'] = len(pool.items)
    stats['retained_distinct_videos'] = len({i['metadata']['video_id'] for i in pool.items})
    stats['retained_distinct_frames'] = len({frame_key(i['metadata']) for i in pool.items})
    return pool.tensors(),stats


def group_pairwise_metrics(pool,members):
    feature = F.normalize(pool['feature'][members].float(),dim=-1)
    rgb = pool['rgb'][members].float()
    structure = F.normalize(pool['structure'][members].float(),dim=-1)
    return {'stage4_cosine':feature@feature.T,
            'rgb_mae':(rgb[:,None,:]-rgb[None,:,:]).abs().mean(-1),
            'rgb_structure_cosine':structure@structure.T}


def validate_group(pool,members,settings):
    if len(members)<settings.min_group_size:
        raise ValueError('Undersized patch group.')
    if len(set(members))!=len(members):
        raise ValueError('Duplicate patches in group.')
    frames = [frame_key(pool['metadata'][i]) for i in members]
    if len(set(frames))!=len(frames):
        raise ValueError('Group must contain one patch per frame.')
    metrics = group_pairwise_metrics(pool,members)
    # Semantic groups are anchored on the selected target, member zero.
    # Pairwise member-to-member metrics remain available as descriptive output.
    mask = torch.zeros(len(members),len(members),dtype=torch.bool)
    mask[0,1:] = True
    if not pool['metadata'][members[0]].get('is_target_reference'):
        raise ValueError('The first group member must be its selected target.')
    if (metrics['stage4_cosine'][mask].min()<settings.min_stage4_cosine-1e-6
            or metrics['rgb_mae'][mask].max()>settings.max_rgb_mae+1e-6
            or metrics['rgb_structure_cosine'][mask].min()<settings.min_rgb_structure_cosine-1e-6):
        raise ValueError('Group fails the target-to-member similarity thresholds.')
    return metrics


def prototype_summary(pool,members,settings):
    cosine = pool['cosine'][members]
    n,P = cosine.shape
    order = torch.argsort(cosine,dim=1,descending=True,stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1,order,torch.arange(1,P+1)[None].expand(n,-1))
    k = min(settings.top_k,P)
    # Full candidate-pool frequency helps identify ubiquitous prototype matches.
    all_order = torch.argsort(pool['cosine'],dim=1,descending=True,stable=True)
    records = []
    for prototype in range(P):
        record = {'prototype_index':prototype,'top1_count':int((ranks[:,prototype]==1).sum()),
                  'top8_count':int((ranks[:,prototype]<=min(8,P)).sum()),
                  'top10_count':int((ranks[:,prototype]<=min(10,P)).sum()),
                  'top_k_count':int((ranks[:,prototype]<=k).sum()),
                  'top_k_fraction':float((ranks[:,prototype]<=k).float().mean()),
                  'mean_rank':float(ranks[:,prototype].float().mean()),
                  'min_cosine':float(cosine[:,prototype].min()),
                  'mean_cosine':float(cosine[:,prototype].mean()),
                  'max_cosine':float(cosine[:,prototype].max()),
                  'above_high_cosine_count':int((cosine[:,prototype]>=settings.prototype_high_cosine).sum()),
                  'actual_valid_active_count':int(pool['valid_active'][members,prototype].sum()),
                  'mean_forward_activation':float(pool['activation'][members,prototype].mean()),
                  'mean_soft_probability':float(pool['probabilities'][members,prototype].mean()),
                  'candidate_pool_top_k_fraction':float((all_order[:,:k]==prototype).any(1).float().mean())}
        record['shared_by_all_in_top_k'] = record['top_k_count']==n
        record['shared_by_all_with_high_cosine'] = (record['shared_by_all_in_top_k']
                                                  and record['above_high_cosine_count']==n)
        records.append(record)
    records.sort(key=lambda r:(-r['top_k_count'],-r['min_cosine'],r['mean_rank'],r['prototype_index']))
    return records,ranks


def overlap_statistics(pool,groups,settings,random_pairs=2000):
    k = min(settings.top_k,pool['cosine'].shape[1])
    top = torch.argsort(pool['cosine'],dim=1,descending=True,stable=True)[:,:k]
    def overlap(a,b):
        intersection = int((top[a,:,None]==top[b,None,:]).any(1).sum())
        return intersection,intersection/(2*k-intersection)
    grouped = []
    excluded = set()
    for group in groups:
        for i,a in enumerate(group):
            for b in group[i+1:]:
                grouped.append(overlap(a,b))
                excluded.add(tuple(sorted((a,b))))
    rng = random.Random(settings.seed+1)
    baseline = []
    used = set()
    n = len(pool['metadata'])
    for _ in range(random_pairs*20):
        if len(baseline)>=random_pairs or n<2:
            break
        a,b = sorted(rng.sample(range(n),2))
        if ((a,b) in excluded or (a,b) in used
                or pool['metadata'][a]['video_id']==pool['metadata'][b]['video_id']):
            continue
        used.add((a,b))
        baseline.append(overlap(a,b))
    def summarize(values):
        return {'pair_count':len(values),
                'mean_common_top_k_prototypes':float(np.mean([v[0] for v in values])) if values else None,
                'mean_top_k_jaccard':float(np.mean([v[1] for v in values])) if values else None}
    return {'top_k':k,'within_groups':summarize(grouped),
            'random_cross_video_candidate_pairs':summarize(baseline),
            'note':'Descriptive comparison only; pair observations are not independent.'}


def write_csv(path,records):
    if not records:
        path.write_text('',encoding='utf-8')
        return
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def image_from_tensor(frame):
    data = (frame.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    return Image.fromarray(data)


def save_montage(images,path,cell_hw=(180,240),columns=4,footers=None):
    if not images:
        return
    base_h,width = cell_hw
    header,footer = 22,(24 if footers else 0)
    image_max_h = base_h-header-6
    cell_h = base_h+footer
    columns = min(columns,len(images))
    rows = math.ceil(len(images)/columns)
    canvas = Image.new('RGB',(columns*width,rows*cell_h),'white')
    draw = ImageDraw.Draw(canvas)
    for index,image in enumerate(images):
        copy = image.copy()
        copy.thumbnail((width-12,image_max_h))
        x,y = (index%columns)*width,(index//columns)*cell_h
        canvas.paste(copy,(x+(width-copy.width)//2,y+header+(image_max_h-copy.height)//2))
        draw.text((x+8,y+4),f'Patch {index:02d}',fill='black')
        if footers:
            draw.text((x+8,y+base_h),str(footers[index]),fill='black')
    canvas.save(path)


def write_heatmap(pool,members,summary,path,top_k):
    prototypes = [record['prototype_index'] for record in summary if record['top_k_count']>0]
    cell_w,cell_h,left,top = 66,38,82,90
    width,height = left+cell_w*len(prototypes)+20,top+cell_h*len(members)+50
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="12" y="22" font-family="sans-serif" font-size="16">Patch–prototype cosine similarities</text>',
             f'<text x="12" y="44" font-family="sans-serif" font-size="12">Union of top-{top_k} matches. * = actual valid forward assignment.</text>']
    for column,prototype in enumerate(prototypes):
        x = left+column*cell_w
        lines.append(f'<text x="{x+8}" y="{top-12}" font-family="sans-serif" font-size="12">c_{prototype}</text>')
        for row,index in enumerate(members):
            cosine = float(pool['cosine'][index,prototype])
            active = bool(pool['valid_active'][index,prototype])
            strength = max(0,min(1,(cosine+1)/2))
            color = f'rgb({int(245-180*strength)},{int(248-110*strength)},{int(255-40*strength)})'
            y = top+row*cell_h
            tooltip = html.escape(f'patch={row}, prototype={prototype}, cosine={cosine:.6f}, '
                                  f'soft_probability={float(pool["probabilities"][index,prototype]):.6f}, '
                                  f'forward_activation={float(pool["activation"][index,prototype]):.6f}')
            lines += [f'<g><title>{tooltip}</title><rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color}"/>',
                      f'<text x="{x+5}" y="{y+24}" font-family="sans-serif" font-size="12">{cosine:.3f}{"*" if active else ""}</text></g>']
    for row in range(len(members)):
        lines.append(f'<text x="10" y="{top+row*cell_h+24}" font-family="sans-serif" font-size="12">Patch {row:02d}</text>')
    lines.append('</svg>')
    path.write_text('\n'.join(lines),encoding='utf-8')


def export_results(pool,groups,dataset,adapter,model,settings,root,manifest,*,overwrite_results=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if overwrite_results:
        for name in EXPERIMENT_ROOT_FILES:
            path = root / name
            if path.is_file():
                path.unlink()
    resize_to = adapter.resolve_backbone_spatial_hw(model)
    # Validate the complete selection before writing any group outputs.
    if len({index for group in groups for index in group}) != sum(map(len,groups)):
        raise ValueError('Patch groups must be disjoint.')
    for group in groups:
        validate_group(pool,group,settings)
    all_rows = []
    reports = []
    for members in groups:
        reference = pool['metadata'][members[0]]
        group_number = reference['source_group_id']
        group_dir = root/reference['source_group_name']
        prepare_group_output_dir(group_dir, reference, overwrite_results=overwrite_results)
        (group_dir/'target_metadata.json').write_text(json.dumps(reference,indent=2),encoding='utf-8')
        metrics = validate_group(pool,members,settings)
        summary,ranks = prototype_summary(pool,members,settings)
        write_csv(group_dir/'prototype_summary.csv',summary)
        write_heatmap(pool,members,summary,group_dir/'activation_heatmap.svg',min(settings.top_k,ranks.shape[1]))
        metadata = []
        crops,frames,structure_footers = [],[],[]
        for local_index,candidate_index in enumerate(members):
            source = pool['metadata'][candidate_index]
            single = adapter.video_saliency_collate_fn([dataset[source['dataset_index']]])
            names,rgb_batch,_,_,_,_ = single
            rgb = adapter.rgb_video_to_btchw(rgb_batch)
            if str(names[0])!=source['video_id'] or tensor_digest(rgb[0])!=source['rgb_window_sha256']:
                raise RuntimeError('Dataset window changed during reload. Disable stochastic transforms and rerun.')
            resized = adapter.preprocess_rgb_video_like_backbone(rgb,resize_to)[0]
            patch_dir = group_dir/f'patch_{local_index:02d}'
            patch_dir.mkdir()
            time_idx = int(source.get('target_window_offset', len(resized) - 1))
            frame = image_from_tensor(resized[time_idx])
            box = source['box_model_xyxy']
            crop = frame.crop(tuple(box))
            boxed = frame.copy()
            ImageDraw.Draw(boxed).rectangle((box[0],box[1],box[2]-1,box[3]-1),outline='red',width=2)
            frame.save(patch_dir/'target_frame.png')
            image_from_tensor(rgb[0, time_idx]).save(patch_dir/'target_frame_native.png')
            boxed.save(patch_dir/'target_frame_boxed.png')
            crop.save(patch_dir/'patch.png')
            torch.save({'rgb_model_input':resized.float().contiguous(),
                        'input_space':'backbone spatial resize, before ImageNet normalization',
                        'metadata':source},patch_dir/'temporal_window.pt')
            sequence_dir = patch_dir/'window_frames'
            sequence_dir.mkdir()
            sequence = []
            for offset,temporal_frame in enumerate(resized):
                image = image_from_tensor(temporal_frame)
                image.save(sequence_dir/f'frame_{offset:03d}.png')
                sequence.append(image)
            sequence[0].save(patch_dir/'temporal_window.gif',save_all=True,append_images=sequence[1:],
                             duration=100,loop=0)
            record = {**source,'candidate_index':candidate_index,'group_patch_index':local_index,
                      'is_target_reference':local_index==0,
                      'source_group_id':group_number,
                      'target_similarity':{key:float(value[0,local_index]) for key,value in metrics.items()},
                      'patch_directory':str(patch_dir.relative_to(root)),
                      'model_input_hw':list(resize_to),
                      'nominal_cell_not_receptive_field':True,
                      'gif_frame_duration_ms':100,'gif_timing_is_original_video_fps':False,
                      'top_k_cosine_matches':[]}
            prototype_order = torch.argsort(pool['cosine'][candidate_index],descending=True,stable=True)
            for prototype in prototype_order.tolist():
                row = {'group':group_number,'patch':local_index,'candidate_index':candidate_index,
                       'is_target_reference':local_index==0,
                       'video_id':source['video_id'],'dataset_index':source['dataset_index'],
                       'prototype_index':prototype,'rank':int(ranks[local_index,prototype]),
                       'cosine_similarity':float(pool['cosine'][candidate_index,prototype]),
                       'scaled_logit':float(pool['logits'][candidate_index,prototype]),
                       'soft_assignment_probability':float(pool['probabilities'][candidate_index,prototype]),
                       'forward_activation':float(pool['activation'][candidate_index,prototype]),
                       'actual_valid_active':bool(pool['valid_active'][candidate_index,prototype]),
                       'in_top8':bool(ranks[local_index,prototype]<=min(8,ranks.shape[1])),
                       'in_top10':bool(ranks[local_index,prototype]<=min(10,ranks.shape[1])),
                       'in_requested_top_k':bool(ranks[local_index,prototype]<=min(settings.top_k,ranks.shape[1]))}
                all_rows.append(row)
                if row['in_requested_top_k']:
                    record['top_k_cosine_matches'].append(row)
            metadata.append(record)
            crops.append(crop)
            frames.append(boxed)
            structure_footers.append(
                f'rgb-structure-cosine {float(metrics["rgb_structure_cosine"][0, local_index]):.3f}')
        (group_dir/'patch_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        save_montage(crops,group_dir/'contact_sheet.png',footers=structure_footers)
        save_montage(frames,group_dir/'frame_gallery.png')
        mask = ~torch.eye(len(members),dtype=torch.bool)
        pair_records = []
        for a in range(len(members)):
            for b in range(a+1,len(members)):
                pair_records.append({'patch_a':a,'patch_b':b,
                                     **{key:float(value[a,b]) for key,value in metrics.items()}})
        write_csv(group_dir/'pairwise_similarity.csv',pair_records)
        shared = [r for r in summary if r['shared_by_all_in_top_k']]
        report = {'group':group_number,'member_count':len(members),
                  'target_candidate_index':members[0],
                  'target_path':reference['selected_target_path'],
                  'similarity_gate_scope':'target_to_member',
                  'minimum_target_stage4_cosine':float(metrics['stage4_cosine'][0,1:].min()),
                  'maximum_target_rgb_mae':float(metrics['rgb_mae'][0,1:].max()),
                  'minimum_target_rgb_structure_cosine':float(metrics['rgb_structure_cosine'][0,1:].min()),
                  'distinct_videos':len({pool['metadata'][i]['video_id'] for i in members}),
                  'distinct_frames':len({frame_key(pool['metadata'][i]) for i in members}),
                  'minimum_pairwise_stage4_cosine':float(metrics['stage4_cosine'][mask].min()),
                  'maximum_pairwise_rgb_mae':float(metrics['rgb_mae'][mask].max()),
                  'minimum_pairwise_rgb_structure_cosine':float(metrics['rgb_structure_cosine'][mask].min()),
                  'shared_top_k_prototypes':[r['prototype_index'] for r in shared],
                  'shared_top_k_high_cosine_prototypes':[r['prototype_index'] for r in shared if r['shared_by_all_with_high_cosine']],
                  'directory':group_dir.name}
        reports.append(report)
    write_csv(root/'patch_prototype_activations.csv',all_rows)
    overlap = overlap_statistics(pool,groups,settings)
    (root/'topk_overlap_baseline.json').write_text(json.dumps(overlap,indent=2),encoding='utf-8')
    torch.save({key:value for key,value in pool.items() if key!='metadata'},root/'candidate_features_and_scores.pt')
    (root/'candidate_metadata.json').write_text(json.dumps(pool['metadata'],indent=2),encoding='utf-8')
    result = {**manifest,'settings':asdict(settings),'groups_requested':settings.num_groups,
              'groups_found':len(groups),'groups':reports,
              'thresholds_automatically_relaxed':False,
              'grouping_uses_prototype_scores':False,
              'group_feature_space':'live stage4 concept input before visual_encoder',
              'nominal_cells_are_not_full_receptive_fields':True,
              'absolute_frame_numbers_available':True,
              'sampling':manifest['sampling'],
              'group_search':'target_reference_similarity_ranked_one_patch_per_frame',
              'similarity_gate_scope':'target_to_member',
              'pairwise_member_metrics_are_descriptive':True,
              'candidate_pool_is_target_filtered':True,
              'topk_definition':'descending patch–prototype cosine, not decoder-active membership',
              'overlap':overlap}
    (root/'experiment_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (root/'target_groups.json').write_text(json.dumps(manifest['target_group_selection'],indent=2),encoding='utf-8')
    sections = []
    for r in reports:
        name = r['directory']
        sections.append(f'<section><h2>Group {r["group"]}: {r["member_count"]} patches</h2>'
                        f'<p>Shared top-{settings.top_k} prototypes: {html.escape(str(r["shared_top_k_prototypes"]))}</p>'
                        '<p>Patch 00 is the selected target. Similarity thresholds apply to each match against that target. '
                        'A group may include several frames from one video, with one patch per frame.</p>'
                        f'<img src="{name}/target.jpg" alt="Selected target"/><br/>'
                        f'<img src="{name}/contact_sheet.png"/><br/><img src="{name}/frame_gallery.png"/>'
                        f'<p><a href="{name}/activation_heatmap.svg">Activation heatmap</a> · '
                        f'<a href="{name}/prototype_summary.csv">Prototype summary</a> · '
                        f'<a href="{name}/patch_metadata.json">Frames, windows, and coordinates</a></p></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Stage-4 similar patch groups</title>'
            '<style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 20px}img{max-width:100%}section{margin:36px 0}</style>'
            f'<h1>Stage-4 target patch groups</h1><p>{len(groups)} of {settings.num_groups} target groups met the target-to-member thresholds and minimum size.</p>'
            '<p>Grouping uses stage-4 features plus RGB appearance, independently of prototype ranks. '
            'High cosine rank and actual forward activation are reported separately. '
            'The random overlap baseline uses the retained target-filtered candidate pool.</p>'
            '<p><a href="target_groups.json">Status of every selected target, including unfilled groups</a></p>'
            '<p><a href="experiment_summary.json">Experiment summary</a> · '
            '<a href="patch_prototype_activations.csv">Every patch–prototype score</a> · '
            '<a href="topk_overlap_baseline.json">Random cross-video overlap baseline</a></p>'+''.join(sections))
    (root/'index.html').write_text(page,encoding='utf-8')
    return result


def verify_checkpoint_parameters(model,checkpoint):
    ckpt = torch.load(checkpoint,map_location='cpu',weights_only=False)
    state = ckpt.get('model_state_dict',ckpt)
    bad = [name for name,param in model.named_parameters()
           if name not in state or not torch.is_tensor(state[name]) or state[name].shape!=param.shape]
    if bad:
        raise ValueError(f'Checkpoint/model parameters do not match; aborting rather than analyzing random weights: {bad[:10]}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery-script',default=str(Path(__file__).with_name('prototype_to_patch.py')))
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--dataset-dir',required=True)
    parser.add_argument('--target-groups-dir',default=None,
                        help='Read group_<integer>/target.jpg here (default: same as --output-dir).')
    parser.add_argument('--output-dir',default=TARGET_GROUPS_DIR,
                        help='Write results here; may already contain group_*/target.jpg seeds only.')
    parser.add_argument('--overwrite-results',action=argparse.BooleanOptionalAction,default=False,
                        help='Replace prior experiment exports. Inside each group_<id>/, delete every file and folder except target.jpg.')
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--similarity-device',default='auto',
                        help='Device for patch-to-target gates (auto = --device).')
    parser.add_argument('--window-len',type=int,default=32)
    parser.add_argument('--stride',type=int,default=32,
                        help='Loader stride between window starts (1 = every valid window).')
    parser.add_argument('--max-windows',type=int,default=0,
                        help='Cap loader windows to scan; 0 = all windows in the dataset.')
    parser.add_argument('--frame-sampling',choices=('middle','all'),default='all',
                        help='all = every loader window; middle = one middle-frame window per video.')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--num-workers',type=int,default=0)
    skip = {'salient_patches_only','num_groups','patches_per_window','max_candidates_per_video','max_seed_trials'}
    for name,default in asdict(Settings()).items():
        if name in skip:
            continue
        parser.add_argument('--'+name.replace('_','-'),type=type(default),default=default)
    # Accept legacy flags so an existing command still runs; target-based search
    # does not use random seed trials or preselect only a few patches per window.
    for name in ('num_groups','patches_per_window','max_candidates_per_video','max_seed_trials'):
        parser.add_argument('--'+name.replace('_','-'),type=int,default=getattr(Settings(),name),
                            help=argparse.SUPPRESS)
    parser.add_argument('--salient-patches-only',action=argparse.BooleanOptionalAction,default=True,
                        help='Keep only salient stage-4 patches when building groups (default: true).')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    target_root = Path(args.target_groups_dir).expanduser().resolve() if args.target_groups_dir else root
    specs = discover_target_groups(target_root)
    validate_output_workspace(root, [spec['group_name'] for spec in specs],
                              overwrite_results=args.overwrite_results)
    settings = Settings(**{key:getattr(args,key) for key in asdict(Settings())})
    settings.num_groups = len(specs)
    settings.salient_patches_only = args.salient_patches_only
    if not settings.salient_patches_only:
        settings.saliency_filter_mode = 'none'
    settings.validate()
    print(f'Salient patches only: {settings.salient_patches_only}; '
          f'filter={settings.saliency_filter_mode} source={settings.saliency_source} '
          f'top_percent={settings.saliency_top_percent}',flush=True)
    if args.max_windows<0 or args.window_len<1 or args.stride<1 or args.batch_size<1 or args.num_workers<0:
        raise SystemExit('Invalid window/loader settings.')
    adapter = load_gallery_adapter(args.gallery_script)
    adapter.set_seed(settings.seed)
    device = torch.device(args.device)
    checkpoint = adapter.resolve_checkpoint_path(args.checkpoint)
    model = adapter.load_saliency_model(str(checkpoint),device)
    verify_checkpoint_parameters(model,checkpoint)
    dataset = adapter.DatasetLoader(args.dataset_dir,window_len=args.window_len,stride=args.stride)
    references = load_target_references(model,dataset,adapter,settings,specs,args.window_len)
    sampling = 'one_middle_frame_per_video' if args.frame_sampling=='middle' else 'all_available_dataset_windows'
    indices = (select_middle_frame_window_indices(dataset,args.window_len)
               if args.frame_sampling=='middle' else list(range(len(dataset))))
    if args.max_windows:
        indices = indices[: min(args.max_windows, len(indices))]
    if not indices:
        raise ValueError('The supplied dataset contains no usable search windows.')
    print(
        f'Using {len(indices)} search windows; sampling={sampling}; targets={len(references)}. '
        'A video may contribute one patch from each frame; a group keeps at most one patch per frame.',
        flush=True,
    )
    subset = Subset(dataset, indices)
    loader = DataLoader(subset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                        collate_fn=adapter.video_saliency_collate_fn,pin_memory=device.type=='cuda')
    pool,stats = collect_candidates(model,loader,adapter,settings,references,sampling,
                                    similarity_device=args.similarity_device)
    print(f'Comparing {len(pool["metadata"])} patches across {stats["retained_distinct_videos"]} videos '
          f'and {stats["retained_distinct_frames"]} frames.',flush=True)
    groups,target_selection = select_target_groups(pool,settings,len(references))
    manifest = {'checkpoint':str(checkpoint),'checkpoint_file_size':checkpoint.stat().st_size,
                'checkpoint_mtime_ns':checkpoint.stat().st_mtime_ns,
                'dataset_dir':str(Path(args.dataset_dir).resolve()),'dataset_length':len(dataset),
                'sampled_dataset_indices':indices,'candidate_collection':stats,
                'sampling':sampling,'target_groups_dir':str(target_root),
                'target_group_selection':target_selection,
                'targets_included_in_prototype_analysis':True,
                'target_references_bypass_saliency_and_crop_variance_filters':True,
                'window_len':args.window_len,'stride':args.stride,
                'model_assignment_mode':str(getattr(model.concept_creations['stage4'],'visual_assignment_mode','unknown')),
                'model_top_k_export_slots':int(getattr(model.concept_creations['stage4'],'top_k',0)),
                'model_assignment_temperature':float(getattr(model.concept_creations['stage4'],'visual_assignment_temperature',0))}
    result = export_results(pool,groups,dataset,adapter,model,settings,root,manifest,
                            overwrite_results=args.overwrite_results)
    print(f'Found {result["groups_found"]}/{settings.num_groups} target groups. Open {root / "index.html"}',flush=True)
    if len(groups)<settings.num_groups:
        print('Thresholds were not relaxed. See target_groups.json; try --no-salient-patches-only '
              'or relax --min-stage4-cosine / RGB gates to broaden the search.',flush=True)


if __name__=='__main__':
    main()
