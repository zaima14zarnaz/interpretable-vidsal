#!/usr/bin/env python3
"""Group stage-4 patches before examining prototype ranks.

Place beside visual_concept_explanations.py, which supplies the project's model
and dataloader configuration. No model weights are changed. All group members
must be from distinct videos and satisfy every pairwise similarity gate.
Top-k cosine matches are explicitly distinct from actual forward activations.
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
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


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
    """Bounded seeded reservoir. Stores descriptors/scores, not full RGB windows."""
    def __init__(self, settings):
        self.settings = settings
        self.random = random.Random(settings.seed)
        self.items = []
        self.video_seen = {}
        self.seen = 0

    def add(self,item):
        video = item['metadata']['video_id']
        seen = self.video_seen.get(video,0)
        if seen >= self.settings.max_candidates_per_video:
            return
        self.video_seen[video] = seen+1
        self.seen += 1
        if len(self.items)<self.settings.max_candidates:
            self.items.append(item)
        else:
            index = self.random.randrange(self.seen)
            if index < self.settings.max_candidates:
                self.items[index] = item

    def tensors(self):
        if not self.items:
            raise ValueError('No usable candidates; check inputs/crop-variance filter.')
        fields = ('feature','rgb','structure','cosine','logits','probabilities','activation','valid_active')
        result = {field:torch.stack([item[field] for item in self.items]) for field in fields}
        result['metadata'] = [item['metadata'] for item in self.items]
        return result


@torch.inference_mode()
def collect_candidates(model,loader,adapter,settings):
    if 'stage4' not in model.concept_creations:
        raise ValueError('The loaded model has no stage4 concept module.')
    concept = model.concept_creations['stage4']
    if not isinstance(loader.dataset,Subset):
        raise TypeError('Use a Subset loader so source dataset indices remain recoverable.')
    captured = {}
    def capture(module,args):
        # Exactly the backbone feature grid used for stage4 concept assignment,
        # before the trainable concept encoder and before prototype comparison.
        value = args[0]
        captured['stage4'] = value[:,:, -1].detach().float().cpu()
    hook = concept.register_forward_pre_hook(capture)
    pool = CandidatePool(settings)
    generator = torch.Generator().manual_seed(settings.seed)
    cursor = 0
    stats = {'windows_seen':0,'invalid_target_windows_skipped':0,'candidate_patches_scanned':0,
             'low_variance_crops_skipped':0}
    resize_to = adapter.resolve_backbone_spatial_hw(model)
    try:
        model.eval()
        for batch_number,batch in enumerate(loader):
            if not isinstance(batch,(tuple,list)) or len(batch)!=6:
                raise ValueError('Expected existing six-field video_saliency_collate_fn output.')
            names,rgb_batch,_,_,n_frames,padding_valid = batch
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
            measurement = prototype_measurements(stage,concept.visual_concepts)
            batch_ids = meta['batch_idx'].detach().cpu().long()
            time_ids = meta['time_idx'].detach().cpu().long()
            patch_ids = meta['patch_idx'].detach().cpu().long()
            target_rows = time_ids == int(shape['T'])-1
            for b in range(B):
                dataset_index = int(loader.dataset.indices[cursor+b])
                stats['windows_seen'] += 1
                if (torch.is_tensor(padding_valid) and padding_valid.shape==(B,rgb.shape[1])
                        and not bool(padding_valid[b,-1])):
                    stats['invalid_target_windows_skipped'] += 1
                    continue
                video_id = str(names[b])
                if not video_id.strip():
                    raise ValueError('Video IDs must be nonempty and stable across windows.')
                rows = ((batch_ids==b)&target_rows).nonzero(as_tuple=False).flatten()
                rows = rows[torch.randperm(rows.numel(),generator=generator)]
                rgb_hash = tensor_digest(rgb[b])
                kept = 0
                for row_tensor in rows:
                    row = int(row_tensor)
                    patch = int(patch_ids[row])
                    y,x = divmod(patch,W)
                    feature = feature_grid[b,:,y,x]
                    stats['candidate_patches_scanned'] += 1
                    if not torch.isfinite(feature).all() or feature.norm()<1e-8:
                        continue
                    descriptor,structure,std = crop_descriptor(resized[b,-1],patch,(H,W),
                                                                settings.descriptor_size)
                    if std < settings.min_crop_spatial_std or structure.norm()<1e-8:
                        stats['low_variance_crops_skipped'] += 1
                        continue
                    item = {'feature':F.normalize(feature,dim=0),'rgb':descriptor,'structure':structure,
                            **{key:value[row].clone() for key,value in measurement.items()},
                            'metadata':{'video_id':video_id,'dataset_index':dataset_index,
                                        'sample_id':f'{video_id}::dataset_index={dataset_index}',
                                        'target_window_offset':int(rgb.shape[1])-1,
                                        'absolute_frame_indices':None,
                                        'window_length':int(rgb.shape[1]),
                                        'native_input_shape':list(rgb[b].shape),
                                        'rgb_window_sha256':rgb_hash,
                                        'feature_grid_hw':[H,W],'patch_index':patch,
                                        'grid_row':y,'grid_column':x,
                                        'box_model_xyxy':list(bounds(patch,(H,W),resize_to)),
                                        'box_native_xyxy':list(bounds(patch,(H,W),rgb.shape[-2:])),
                                        'crop_spatial_std':std}}
                    pool.add(item)
                    kept += 1
                    if kept>=settings.patches_per_window:
                        break
            cursor += B
            if batch_number==0 or (batch_number+1)%20==0:
                print(f'Windows {cursor}/{len(loader.dataset)}; retained patches {len(pool.items)}',flush=True)
            del out,measurement,model_input
    finally:
        hook.remove()
    stats['reservoir_eligible_patches'] = pool.seen
    stats['retained_patches'] = len(pool.items)
    stats['retained_distinct_videos'] = len({i['metadata']['video_id'] for i in pool.items})
    return pool.tensors(),stats


@torch.inference_mode()
def build_similarity_graph(pool,settings,device='cpu'):
    """Edges require all three visual checks and different video identities."""
    device = torch.device(device)
    features = F.normalize(pool['feature'].float().to(device),dim=-1)
    rgb = pool['rgb'].float().to(device)
    structure = F.normalize(pool['structure'].float().to(device),dim=-1)
    n = features.shape[0]
    adjacency = torch.zeros(n,n,dtype=torch.bool)
    video_codes = {}
    codes = torch.tensor([video_codes.setdefault(m['video_id'],len(video_codes))
                          for m in pool['metadata']],device=device)
    for start in range(0,n,settings.comparison_block):
        end = min(start+settings.comparison_block,n)
        feature_cosine = features[start:end] @ features.T
        shape_cosine = structure[start:end] @ structure.T
        preliminary = ((feature_cosine >= settings.min_stage4_cosine)
                       & (shape_cosine >= settings.min_rgb_structure_cosine)
                       & (codes[start:end,None] != codes[None,:]))
        pairs = preliminary.nonzero(as_tuple=False)
        block = torch.zeros(end-start,n,dtype=torch.bool,device=device)
        for offset in range(0,len(pairs),settings.appearance_pair_chunk):
            selected = pairs[offset:offset+settings.appearance_pair_chunk]
            a,b = selected[:,0]+start,selected[:,1]
            mae = (rgb[a]-rgb[b]).abs().mean(dim=1)
            accepted = mae <= settings.max_rgb_mae
            block[selected[accepted,0],selected[accepted,1]] = True
        adjacency[start:end] = block.cpu()
    # Require both directions to avoid threshold-rounding asymmetry.
    adjacency &= adjacency.T.clone()
    adjacency.fill_diagonal_(False)
    return adjacency


def select_strict_groups(pool,adjacency,settings):
    """Greedy clique search: each added patch must match EVERY member.

    This is a bounded heuristic, not an exhaustive maximum-clique algorithm.
    It may find fewer groups even when a different search could find more.
    Never relax thresholds to fill the requested group count.
    """
    n = adjacency.shape[0]
    available = torch.ones(n,dtype=torch.bool)
    groups = []
    for _ in range(settings.num_groups):
        degree = (adjacency & available[None,:]).sum(dim=1)
        seeds = sorted((available & (degree >= settings.min_group_size-1)).nonzero(as_tuple=False).flatten().tolist(),
                       key=lambda index:(-int(degree[index]),index))
        best = None
        for seed in seeds[:settings.max_seed_trials]:
            members = [seed]
            eligible = available & adjacency[seed]
            while eligible.any() and len(members)<settings.max_group_size:
                candidates = eligible.nonzero(as_tuple=False).flatten().tolist()
                # Favor candidates that leave room for more pairwise matches.
                chosen = max(candidates,key=lambda i:(int((adjacency[i]&eligible).sum()),
                                                       float(pool['feature'][i]@pool['feature'][seed]),-i))
                members.append(chosen)
                eligible &= adjacency[chosen]
                eligible[members] = False
            if len(members)>=settings.min_group_size:
                if best is None or len(members)>len(best):
                    best = members
                if len(members)==settings.max_group_size:
                    break
        if best is None:
            break
        validate_group(pool,best,settings)
        groups.append(best)
        available[best] = False
    return groups


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
    videos = [pool['metadata'][i]['video_id'] for i in members]
    if len(set(videos))!=len(videos):
        raise ValueError('Group must contain different videos.')
    metrics = group_pairwise_metrics(pool,members)
    mask = ~torch.eye(len(members),dtype=torch.bool)
    if (metrics['stage4_cosine'][mask].min()<settings.min_stage4_cosine-1e-6
            or metrics['rgb_mae'][mask].max()>settings.max_rgb_mae+1e-6
            or metrics['rgb_structure_cosine'][mask].min()<settings.min_rgb_structure_cosine-1e-6):
        raise ValueError('Group fails the all-pairs similarity thresholds.')
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


def save_montage(images,path,cell_hw=(180,240),columns=4):
    if not images:
        return
    height,width = cell_hw
    columns = min(columns,len(images))
    rows = math.ceil(len(images)/columns)
    canvas = Image.new('RGB',(columns*width,rows*height),'white')
    draw = ImageDraw.Draw(canvas)
    for index,image in enumerate(images):
        copy = image.copy()
        copy.thumbnail((width-12,height-28))
        x,y = (index%columns)*width,(index//columns)*height
        canvas.paste(copy,(x+(width-copy.width)//2,y+20+(height-28-copy.height)//2))
        draw.text((x+8,y+5),f'Patch {index:02d}',fill='black')
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


def export_results(pool,groups,dataset,adapter,model,settings,root,manifest):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new/empty output directory to avoid mixing experiment runs.')
    root.mkdir(parents=True,exist_ok=True)
    resize_to = adapter.resolve_backbone_spatial_hw(model)
    # Validate the complete selection before writing any group outputs.
    if len({index for group in groups for index in group}) != sum(map(len,groups)):
        raise ValueError('Patch groups must be disjoint.')
    for group in groups:
        validate_group(pool,group,settings)
    all_rows = []
    reports = []
    for group_number,members in enumerate(groups):
        group_dir = root/f'group_{group_number:02d}'
        group_dir.mkdir()
        metrics = validate_group(pool,members,settings)
        summary,ranks = prototype_summary(pool,members,settings)
        write_csv(group_dir/'prototype_summary.csv',summary)
        write_heatmap(pool,members,summary,group_dir/'activation_heatmap.svg',min(settings.top_k,ranks.shape[1]))
        metadata = []
        crops,frames = [],[]
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
            frame = image_from_tensor(resized[-1])
            box = source['box_model_xyxy']
            crop = frame.crop(tuple(box))
            boxed = frame.copy()
            ImageDraw.Draw(boxed).rectangle((box[0],box[1],box[2]-1,box[3]-1),outline='red',width=2)
            frame.save(patch_dir/'target_frame.png')
            image_from_tensor(rgb[0,-1]).save(patch_dir/'target_frame_native.png')
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
                      'patch_directory':str(patch_dir.relative_to(root)),
                      'model_input_hw':list(resize_to),
                      'nominal_cell_not_receptive_field':True,
                      'gif_frame_duration_ms':100,'gif_timing_is_original_video_fps':False,
                      'top_k_cosine_matches':[]}
            prototype_order = torch.argsort(pool['cosine'][candidate_index],descending=True,stable=True)
            for prototype in prototype_order.tolist():
                row = {'group':group_number,'patch':local_index,'candidate_index':candidate_index,
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
        (group_dir/'patch_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        save_montage(crops,group_dir/'contact_sheet.png')
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
                  'distinct_videos':len({pool['metadata'][i]['video_id'] for i in members}),
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
              'absolute_frame_numbers_available':False,
              'group_search':'bounded greedy clique search, not exhaustive',
              'topk_definition':'descending patch–prototype cosine, not decoder-active membership',
              'overlap':overlap}
    (root/'experiment_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    sections = []
    for r in reports:
        name = r['directory']
        sections.append(f'<section><h2>Group {r["group"]}: {r["member_count"]} patches</h2>'
                        f'<p>Shared top-{settings.top_k} prototypes: {html.escape(str(r["shared_top_k_prototypes"]))}</p>'
                        f'<img src="{name}/contact_sheet.png"/><br/><img src="{name}/frame_gallery.png"/>'
                        f'<p><a href="{name}/activation_heatmap.svg">Activation heatmap</a> · '
                        f'<a href="{name}/prototype_summary.csv">Prototype summary</a> · '
                        f'<a href="{name}/patch_metadata.json">Frames, windows, and coordinates</a></p></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Stage-4 similar patch groups</title>'
            '<style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 20px}img{max-width:100%}section{margin:36px 0}</style>'
            f'<h1>Stage-4 similar patch groups</h1><p>{len(groups)} of {settings.num_groups} requested groups met the strict thresholds.</p>'
            '<p>Grouping uses stage-4 features plus RGB appearance, independently of prototype ranks. '
            'High cosine rank and actual forward activation are reported separately.</p>'
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
    parser.add_argument('--gallery-script',default=str(Path(__file__).with_name('visual_concept_explanations.py')))
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--dataset-dir',required=True)
    parser.add_argument('--output-dir',default='stage4_patch_groups')
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--similarity-device',default='cpu')
    parser.add_argument('--window-len',type=int,default=16)
    parser.add_argument('--stride',type=int,default=16)
    parser.add_argument('--max-windows',type=int,default=512)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--num-workers',type=int,default=0)
    for name,default in asdict(Settings()).items():
        parser.add_argument('--'+name.replace('_','-'),type=type(default),default=default)
    return parser.parse_args()


def main():
    args = parse_args()
    settings = Settings(**{key:getattr(args,key) for key in asdict(Settings())})
    settings.validate()
    if args.max_windows<1 or args.window_len<1 or args.stride<1 or args.batch_size<1 or args.num_workers<0:
        raise SystemExit('Invalid window/loader settings.')
    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit('Use a new/empty output directory for this experiment.')
    adapter = load_gallery_adapter(args.gallery_script)
    adapter.set_seed(settings.seed)
    device = torch.device(args.device)
    checkpoint = adapter.resolve_checkpoint_path(args.checkpoint)
    model = adapter.load_saliency_model(str(checkpoint),device)
    verify_checkpoint_parameters(model,checkpoint)
    dataset = adapter.DatasetLoader(args.dataset_dir,window_len=args.window_len,stride=args.stride)
    generator = torch.Generator().manual_seed(settings.seed)
    indices = torch.randperm(len(dataset),generator=generator)[:min(args.max_windows,len(dataset))].tolist()
    subset = Subset(dataset,indices)
    loader = DataLoader(subset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                        collate_fn=adapter.video_saliency_collate_fn,pin_memory=device.type=='cuda')
    pool,stats = collect_candidates(model,loader,adapter,settings)
    print(f'Comparing {len(pool["metadata"])} patches across {stats["retained_distinct_videos"]} videos.',flush=True)
    adjacency = build_similarity_graph(pool,settings,args.similarity_device)
    groups = select_strict_groups(pool,adjacency,settings)
    manifest = {'checkpoint':str(checkpoint),'checkpoint_file_size':checkpoint.stat().st_size,
                'checkpoint_mtime_ns':checkpoint.stat().st_mtime_ns,
                'dataset_dir':str(Path(args.dataset_dir).resolve()),'dataset_length':len(dataset),
                'sampled_dataset_indices':indices,'candidate_collection':stats,
                'window_len':args.window_len,'stride':args.stride,
                'model_assignment_mode':str(getattr(model.concept_creations['stage4'],'visual_assignment_mode','unknown')),
                'model_top_k_export_slots':int(getattr(model.concept_creations['stage4'],'top_k',0)),
                'model_assignment_temperature':float(getattr(model.concept_creations['stage4'],'visual_assignment_temperature',0))}
    result = export_results(pool,groups,dataset,adapter,model,settings,root,manifest)
    print(f'Found {result["groups_found"]}/{settings.num_groups} strict groups. Open {root / "index.html"}',flush=True)
    if len(groups)<settings.num_groups:
        print('Thresholds were not relaxed. More sampled windows/candidates or a different seed may help.',flush=True)


if __name__=='__main__':
    main()
