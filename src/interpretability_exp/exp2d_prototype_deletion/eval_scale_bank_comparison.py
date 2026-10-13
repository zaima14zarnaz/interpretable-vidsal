"""Paired feature-preserving prototype bypass at each scale: CC, SIM, NSS.

Run: python eval_scale_bank_comparison.py
Pilot: python eval_scale_bank_comparison.py --max-windows 20
Place anywhere under src/ or pass --src-root /path/to/ExplainableSaliency/src.

Six conditions on EVERY evaluated validation window, using best_dhf1k.pth:
  baseline_all_stages
  bypass_stage1_prototypes
  bypass_stage2_prototypes
  bypass_stage3_prototypes
  bypass_stage4_prototypes
  bypass_all_stage_prototypes  (feature-only decoder)

Only the selected fusion block temporarily uses a feature-only forward:
  feature projection -> configured previous-stage fusion -> refine1 -> refine2.
Previous-stage projection/learned upsampling and its original prev_scale are
retained. include_backbone_in_fusion is respected (including the stage2 option).
The block does NOT read its concept volume, active prototype vectors or weights;
concept projection, FiLM, unary/pairwise priority scoring and prototype-conditioned
updates are skipped. This implements gamma=1, beta=0, identity spatial gating,
and F_out=F instead of the prototype mixture. It DOES NOT zero the feature stream.
Main and side heads, subsequent upsampling and temporal aggregation stay intact.

Baseline concept assignments and feature tensors are computed once per window
and cached for all conditions. Unbypassed stages retain their baseline prototype
evidence and original forward methods; downstream decoder outputs can change.
No prototype-ID selection, deletion, bank sharing, retraining, spatial shuffling
or top-5% ROI. Assignment tensors remain unchanged: baseline evidence at the
bypassed stage is UNUSED, not deleted. This measures dependence on the trained
prototype-conditioning pathway; it is not a trained architecture comparison.
FiLM removal also removes conditioning carried by the patch-embedding residual
inside the concept volume, so this is broader than deleting bank lookup alone.

Default inference: topk_soft, top_k=8, output_activation=none, stride32,
--metric-input model_map, matching the preceding evaluator. These settings may
differ from checkpoint training mode; both baseline and bypass use the same ones.
Full-map CC/SIM use human density; NSS uses true fixations at the last REAL frame.
No-fixation NSS is undefined/excluded. Means use common finite windows across
all conditions per metric; paired CIs resample whole videos and recompute the
equal-window statistic. Negative delta means lower metric performance.

Outputs in evaluation_outputs/stage_prototype_bypass/:
  summary.csv: Condition, CC, SIM, NSS
  paired_comparisons.csv: metric deltas and whole-video bootstrap 95% CIs
  window_results.csv, video_results.csv
  bypass_evidence_summary.csv: baseline active slots/mass unused at each stage
  stage_prototype_bypass_results.json: settings, metrics, audits and exclusions

Checkpoint weights/model source are unchanged. Reversible runtime overrides
restore the original forward after each condition, including errors. Cached
no-bypass replay is checked against baseline. Source guards require the reviewed
fusion/decoder forward implementation; update the adapter if that code changes.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Tuple

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
STAGES=('stage1','stage2','stage3','stage4')
METRIC_NAMES=('CC','SIM','NSS')
CHECKPOINT_PATH='/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/best_dhf1k.pth'
VAL_DATASET_DIR='/data/quantization/zaima/videosal_datasets/dhf1k/val'
GPU_ID=1
WINDOW_LEN=BATCH_SIZE=NUM_WORKERS=None
WINDOW_STRIDE=32
MODEL_TOP_K=8
MODEL_ASSIGNMENT_MODE='topk_soft'
METRIC_INPUT='model_map'

def _resolve_devices() -> Tuple[torch.device, torch.device]:
    if torch.cuda.is_available():
        if GPU_ID >= torch.cuda.device_count() or GPU_ID < 0:
            raise ValueError(f"GPU_ID={GPU_ID} is unavailable.")
        device = torch.device(f"cuda:{GPU_ID}")
        return device, device
    return torch.device("cpu"), torch.device("cpu")

def build_model(backbone_device, head_device):
    import train as train_cfg
    from model.model import ExplainableVidSalModel

    # Matches the user's supplied baseline construction, including deprecated args.
    return ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=False, freeze_backbone=train_cfg.FREEZE_BACKBONE,
        backbone_gradient_checkpointing=False, input_format="BTCHW", resize_to=(224, 384),
        concept_dim=128, num_concepts=512, concept_hidden_dim=256, saliency_hidden_dim=256,
        top_k=MODEL_TOP_K, max_source_patches=64, tau_pi=0.5, tau_alpha=0.07, tau_concept=0.07,
        concept_residual_weight=0.0, last_transition_only=True,
        use_rgb_refinement=False, use_feature_refinement=False, output_activation="none",
        return_details=True, use_subpatch_head=True, subpatch_factor=4,
        subpatch_residual_scale=0.5, use_temporal_transition_aggregation=True,
        temporal_aggregation_hidden_channels=128, temporal_aggregation_temperature=1.0,
        visual_concept_on=train_cfg.VISUAL_CONCEPT_ON,
        temporal_concepts_on=train_cfg.TEMPORAL_CONCEPTS_ON,
        visual_concept_logit_scale=train_cfg.VISUAL_CONCEPT_LOGIT_SCALE,
        visual_concept_residual_weight=1.0, use_temporal_feature_infusion=True,
        visual_assignment_mode=MODEL_ASSIGNMENT_MODE,
        use_shared_concept_activations=True,
        prototype_bottleneck_strength=train_cfg.PROTOTYPE_BOTTLENECK_STRENGTH,
        prototype_application_position=train_cfg.PROTOTYPE_APPLICATION_POSITION,
        fine_unary_mask_strength=getattr(train_cfg, "FINE_UNARY_MASK_STRENGTH", 1.0),
        stage2_backbone_fusion_enabled=getattr(train_cfg, "STAGE2_BACKBONE_FUSION_ENABLED", True),
    ).to_split_devices(backbone_device, head_device)

def video_id(filename):
    return Path(str(filename)).stem

def read_video_ids(path):
    values = json.loads(Path(path).read_text())
    if not isinstance(values, list) or not values:
        raise ValueError(f'{path} must be a nonempty JSON list of video IDs.')
    ids = [video_id(v) for v in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path} contains duplicate video IDs.')
    return set(ids)

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return index, self.dataset[index]

def indexed_collate(samples):
    from pre_process.collate import video_saliency_collate_fn
    indices, items = zip(*samples)
    return video_saliency_collate_fn(items), torch.tensor(indices, dtype=torch.long)

def build_loader():
    import train as train_cfg
    from pre_process.dataloader import DatasetLoader
    window_len = train_cfg.WINDOW_LEN if WINDOW_LEN is None else WINDOW_LEN
    batch_size = train_cfg._dataloader_batch_size() if BATCH_SIZE is None else BATCH_SIZE
    workers = train_cfg.NUM_WORKERS if NUM_WORKERS is None else NUM_WORKERS
    dataset = DatasetLoader(VAL_DATASET_DIR, window_len=window_len, stride=WINDOW_STRIDE)
    options = dict(num_workers=workers, collate_fn=indexed_collate,
                   pin_memory=torch.cuda.is_available())
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=False, **options), dataset, window_len

def last_real_map(target, length, indices):
    if not isinstance(target,torch.Tensor):
        raise ValueError('Density and true fixation maps must be supplied by the loader.')
    x=target[indices]
    if x.ndim==3:return x.unsqueeze(1)
    if x.ndim==4:
        if x.shape[1]==1:return x
        if x.shape[1]<length:raise ValueError('Target time axis is shorter than the real RGB window.')
        return x[:,length-1:length]
    if x.ndim==5:
        if x.shape[1]==1:
            if x.shape[2]<length:raise ValueError('Target time axis is too short.')
            return x[:,:,length-1]
        if x.shape[2]==1:
            if x.shape[1]<length:raise ValueError('Target time axis is too short.')
            return x[:,length-1]
    raise ValueError(f'Unsupported density/fixation layout: {tuple(x.shape)}.')

def evaluation_batches(loader, allowed, maximum):
    count=0
    for batch_index,(batch,dataset_indices) in enumerate(tqdm(loader,desc='Paired validation',unit='batch')):
        if len(batch)!=6:raise ValueError('Expected filenames, RGB, density, fixations, n_frames, valid_mask.')
        names,rgb,density,fixation,lengths,valid=batch
        candidates=[i for i,n in enumerate(names) if allowed is None or video_id(n) in allowed]
        for length in sorted({int(lengths[i]) for i in candidates}):
            if length<1:raise ValueError('Empty RGB window.')
            idx=[i for i in candidates if int(lengths[i])==length]
            if valid is not None:
                if valid.ndim==1:idx=[i for i in idx if bool(valid[i])]
                elif valid.ndim==2:idx=[i for i in idx if bool(valid[i,length-1])]
                else:raise ValueError('Expected a [B,T] temporal validity mask or [B] last-frame validity.')
            if maximum is not None:idx=idx[:max(0,maximum-count)]
            if not idx:
                if maximum is not None and count>=maximum:return
                continue
            count+=len(idx)
            yield batch_index,[video_id(names[i]) for i in idx],[int(dataset_indices[i]) for i in idx],rgb[idx,:length],last_real_map(density,length,idx),last_real_map(fixation,length,idx),length
            if maximum is not None and count>=maximum:return

def load_metric_functions():
    # Reuse project implementations, including their resizing/normalization.
    try:module=importlib.import_module('model.metrics')
    except ModuleNotFoundError as exc:
        if exc.name!='model.metrics':raise
        module=importlib.import_module('metrics')
    return {name:getattr(module,fn) for name,fn in [('CC','cc_score'),('NSS','nss_score'),('SIM','sim_score')]}

def prediction_for_metrics(prediction):
    source=prediction['saliency_logits'] if METRIC_INPUT=='sigmoid' else prediction['saliency_map']
    source=source.float()
    if source.ndim!=4 or source.shape[1]!=1 or not torch.isfinite(source).all():
        raise FloatingPointError('Expected finite [B,1,H,W] prediction maps.')
    return source.sigmoid() if METRIC_INPUT=='sigmoid' else source

def per_sample_metrics(prediction, density, fixations, functions):
    pred = prediction_for_metrics(prediction)
    density = density.to(pred.device)
    fixations = fixations.to(pred.device)
    for label, tensor in [('density', density), ('fixations', fixations)]:
        if not torch.isfinite(tensor).all() or (tensor < 0).any():
            raise ValueError(f'{label} ground truth must be finite and nonnegative.')
    binary = (fixations > 0).float()
    rows = []
    for i in range(len(pred)):
        row = {'fixation_pixels': int(binary[i].sum())}
        for metric in METRIC_NAMES:
            if metric == 'NSS' and not row['fixation_pixels']:
                row[metric] = None
                continue
            target = binary[i:i+1] if metric == 'NSS' else density[i:i+1]
            value = float(functions[metric](pred[i:i+1], target).detach().cpu())
            row[metric] = value if math.isfinite(value) else None
        rows.append(row)
    return rows

def bootstrap_ci(deltas, videos, repeats, seed):
    sums, counts = defaultdict(float), defaultdict(int)
    for d, v in zip(deltas, videos):
        sums[v] += d
        counts[v] += 1
    ids = sorted(sums)
    if len(ids) < 2:
        return None
    a = np.array([sums[v] for v in ids], dtype=float)
    b = np.array([counts[v] for v in ids], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, repeats, 1000):
        draws = rng.integers(len(ids), size=(min(1000, repeats-start), len(ids)))
        values.extend((a[draws].sum(1)/b[draws].sum(1)).tolist())
    return np.quantile(values, [.025, .975]).tolist()

def summarize(rows, conditions, repeats, seed):
    by_window = defaultdict(dict)
    for r in rows:
        key = (r['video'], r['dataset_index'])
        if r['condition'] in by_window[key]:
            raise ValueError('Duplicate dataset window/condition in evaluation.')
        by_window[key][r['condition']] = r
    if any(set(c) != set(conditions) for c in by_window.values()):
        raise ValueError('Every window must have all paired conditions.')
    summary = {c: {'Condition': c} for c in conditions}
    contrasts, video_rows, exclusions = [], [], {}
    for metric in METRIC_NAMES:
        paired = [p for p in by_window.values()
                  if all(p[c][metric] is not None and math.isfinite(p[c][metric]) for c in conditions)]
        exclusions[metric] = {'included_windows': len(paired),
                              'excluded_windows': len(by_window)-len(paired),
                              'included_videos': len({p[conditions[0]]['video'] for p in paired})}
        for c in conditions:
            values = [p[c][metric] for p in paired]
            summary[c][metric] = float(np.mean(values)) if values else None
            per_video = defaultdict(list)
            for p in paired:
                per_video[p[c]['video']].append(p[c][metric])
            for v, vs in sorted(per_video.items()):
                video_rows.append({'video': v, 'condition': c, 'metric': metric,
                                   'mean': float(np.mean(vs)), 'windows': len(vs)})
        for c in conditions[1:]:
            ds = [p[c][metric]-p[conditions[0]][metric] for p in paired]
            vs = [p[c]['video'] for p in paired]
            ci = bootstrap_ci(ds, vs, repeats, seed) if ds else None
            contrasts.append({'condition': c, 'reference': conditions[0], 'metric': metric,
                              'mean_delta': float(np.mean(ds)) if ds else None,
                              'ci95_low': ci[0] if ci else None, 'ci95_high': ci[1] if ci else None,
                              'windows': len(ds), 'videos': len(set(vs))})
    return list(summary.values()), contrasts, video_rows, exclusions

def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

SUPPORTED_METHODS = {
    "SpatioTemporalConceptGatedFusionBlock._factorized_pairwise_context": "e4948ffedb9e720e552a83d8c91421586d1f8153a587a06ae91e927d18eb684b",
    "SpatioTemporalConceptGatedFusionBlock._prioritize_concept_patch_activations": "601fbc2c50cf1c41229630346508da0b8a29e6a1bef2db12c326e23acf123a32",
    "SpatioTemporalConceptGatedFusionBlock._build_concept_conditioned_update": "db502d16ba5f4c1478fa25bcf3729357dd23bcf26f2f202d77fb8367506e2e80",
    "SpatioTemporalConceptGatedFusionBlock.forward": "b4411e755f433b9fe46934041627a8fd5db831ec81daf189e7b63888d0be27be",
    "SpatioTemporalConceptGatedFusionBlock._resolve_visual_activations": "9bf5ceb9fd0eb7854d86fc71680c47ed382909c7c61d3657ce250fca9fd310f0",
    "ConceptGatedMultiScaleSaliencyDecoder.forward": "fd96d87d2fc28369533230d50867777f59724b6dd05a23f338bbff6a7eafdccf",
    "VisualConceptCreation._extract_active_prototypes": "ab1d18f2719c62c4b46b1c64611d017f554b4fdb0fbf4a74e372a9ea77501b58",
    "VisualConceptCreation._compute_visual_assignments": "4187aaecef2eba3f87de1d59c6afb254282ad7d22f226f4964deccc3844eceb0",
    "VisualConceptCreation._build_visual_concepts_from_patches": "64b5bee95eda0527fe1dd996ec9e423ac749366f847c7609e5a7e4b8e3ff2bce",
    "VisualConceptCreation.forward": "bbdc2c86be23db5e044cd14645d20dbc7a77d7be9a64a25b02ac49f5f96e1b5e",
    "ExplainableVidSalModel.forward": "b83b145cdf670d557516d8cdc01a88fbc47e173657af941f56b6a18a0f823e37"
}

def _check_method(obj: Any, class_name: str, name: str) -> None:
    # Parse full module so dedent cannot alter multiline string constants.
    tree = ast.parse(inspect.getsource(inspect.getmodule(type(obj))))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == type(obj).__name__)
    node = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    for child in ast.walk(node):
        if hasattr(child, "type_params") and not child.type_params:
            delattr(child, "type_params")
    key = f"{class_name}.{name}"
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if digest != SUPPORTED_METHODS[key]:
        raise RuntimeError(f"Unsupported inference implementation: {key}. Adapt to the current model; do not bypass.")

@contextmanager
def inference_context(model, enabled, dtype):
    with torch.inference_mode(), torch.autocast(model.output_device.type, enabled=enabled, dtype=dtype):
        yield

def capture_baseline(model, rgb, enabled=False, dtype=torch.bfloat16):
    captured = []
    signature = inspect.signature(model.saliency_prediction.forward)
    def hook(module, args, kwargs):
        captured.append(dict(signature.bind(*args, **kwargs).arguments))
    handle = model.saliency_prediction.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with inference_context(model,enabled,dtype):
            baseline = model(rgb, return_details=True, return_concept_losses=False,
                             return_decoder_diagnostics=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one baseline decoder invocation.')
    cache = captured[0]
    if not {'concept_outs','features_dict','output_size'} <= set(cache):
        raise RuntimeError('Unsupported decoder signature; baseline cache is incomplete.')
    return baseline, cache

def cached_prediction(model, cache, concept_outs, enabled=False, dtype=torch.bfloat16):
    kwargs = dict(cache)
    kwargs['concept_outs'] = concept_outs
    kwargs['return_details'] = False
    with inference_context(model,enabled,dtype):
        return model.saliency_prediction(**kwargs)

def validate_baseline(concept, original):
    indices=original['active_visual_prototype_indices']
    valid=original['visual_validity_mask'].bool()
    if indices.ndim!=5 or indices.shape[1]!=1 or valid.shape!=indices.shape:
        raise ValueError('Expected last-frame [B,1,H,W,K] assignments.')
    a=original['visual_activations']
    rows=valid.numel()//valid.shape[-1]
    bank=int(concept.visual_concepts.shape[0])
    if a.shape!=(rows,bank) or not torch.isfinite(a).all() or (a<0).any():
        raise ValueError('Baseline sparse activation shape/values are invalid.')
    expected=1 if concept.visual_assignment_mode=='hard_eval' else min(concept.top_k,bank)
    if not valid.sum(-1).eq(expected).all():
        raise ValueError(f'Expected {expected} valid original slots per patch.')
    if not torch.allclose(a.sum(-1),torch.ones_like(a[:,0]),rtol=1e-5,atol=1e-6):
        raise ValueError('Original baseline assignment weights must sum to one.')
    mapped=torch.zeros_like(a,dtype=torch.long)
    mapped.scatter_add_(1,indices.reshape(rows,-1),valid.reshape(rows,-1).long())
    # Very small selected weights can underflow to zero; their top-k slots
    # remain valid. Require unique selected IDs and no mass outside them.
    if (mapped>1).any() or a.masked_select(mapped.eq(0)).ne(0).any():
        raise ValueError('Original valid slots must be distinct and contain all assignment evidence.')

def load_checkpoint(model, path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    before=path.stat()
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('model_state_dict'),dict):
        raise ValueError('Expected a project checkpoint containing model_state_dict.')
    state=checkpoint['model_state_dict']
    expected=model.state_dict()
    missing=sorted(set(expected)-set(state))
    wrong=[k for k in expected if k in state and expected[k].shape!=state[k].shape]
    if missing or wrong:
        raise ValueError(f'Checkpoint/model mismatch: missing={missing[:8]}, wrong shapes={wrong[:8]}')
    model.load_state_dict({k:state[k] for k in expected},strict=True)
    model.eval();model.allow_eval_concept_losses=False
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed while loading.')
    return {'path':str(path),'epoch':checkpoint.get('epoch'),
            'size_bytes':after.st_size,'mtime_ns':after.st_mtime_ns,
            'ignored_unused_keys':sorted(set(state)-set(expected))}


def feature_only_forward(block, features, concept_volume, prev_decoder=None, *,
                         active_visual_prototypes, visual_validity_mask, concept_out=None):
    """Original feature path, with every concept-dependent operation bypassed.

    Concept-related arguments retain the original signature but are never read.
    This overrides gating with an identity operation; no feature tensor is zeroed.
    """
    feature_proj=block.feature_proj(features)
    prev_up=None
    if prev_decoder is not None:
        prev_up=block.prev_upsample(prev_decoder,target_thw=feature_proj.shape[-3:])
        prev_up=block.prev_proj(prev_up)
    if block.include_backbone_in_fusion:
        fused=feature_proj
        if prev_up is not None:
            fused=fused+block.prev_scale*prev_up
    else:
        if prev_up is None:
            raise ValueError('A decoder stage without backbone fusion requires a previous stage.')
        fused=block.prev_scale*prev_up
    decoded=block.refine2(block.refine1(fused))
    b,_,t,h,w=decoded.shape
    # Report the effective identity gate, not a fabricated learned priority map.
    mask=decoded.new_ones((b,1,t,h,w))
    return decoded,{'priority_mask':mask,
                    'prototype_path_bypassed':True,
                    'priority_mode':'feature_only_bypass',
                    'pairwise_context_used':False,
                    'prototype_application_position':'bypassed',
                    'prototype_bottleneck_strength':decoded.new_tensor(0.),
                    'decoded_base_last_norm':decoded[:,:,-1:].detach().norm(),
                    'prototype_branch_norm':decoded.new_tensor(0.),
                    'prototype_to_base_norm_ratio':decoded.new_tensor(0.)}


class StagePrototypeBypass:
    """Temporarily replace exactly the requested fusion blocks' forward methods."""
    def __init__(self,model,stages):
        self.model=model
        self.stages=tuple(stages)
        if not self.stages or len(set(self.stages))!=len(self.stages):
            raise ValueError('Bypass needs distinct stage names.')
        self.calls={s:0 for s in self.stages}
        self.restores=[]
        self.active=False
        for s in self.stages:
            if s not in model.saliency_prediction.fusion_blocks:
                raise ValueError(f'Unknown stage {s}.')
            block=model.saliency_prediction.fusion_blocks[s]
            _check_method(block,'SpatioTemporalConceptGatedFusionBlock','forward')
        _check_method(model.saliency_prediction,'ConceptGatedMultiScaleSaliencyDecoder','forward')

    def __enter__(self):
        if self.model.training:
            raise RuntimeError('Prototype bypass is inference-only; call model.eval().')
        if self.active:
            raise RuntimeError('Bypass context cannot be entered twice.')
        self.active=True
        self.calls={s:0 for s in self.stages}
        try:
            for s in self.stages:
                block=self.model.saliency_prediction.fusion_blocks[s]
                self.restores.append((block,'forward' in block.__dict__,block.__dict__.get('forward')))
                def wrapped(block,*args,_stage=s,**kwargs):
                    self.calls[_stage]+=1
                    return feature_only_forward(block,*args,**kwargs)
                block.forward=MethodType(wrapped,block)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        for block,own,original in reversed(self.restores):
            if own:
                block.forward=original
            else:
                delattr(block,'forward')
        self.restores.clear()
        self.active=False

    def __exit__(self,exc_type,*exc):
        self._restore()
        if exc_type is None and any(n!=1 for n in self.calls.values()):
            raise RuntimeError(f'Expected one bypassed block call per decoder replay: {self.calls}')


def bypass_conditions():
    conditions={'baseline_all_stages':()}
    conditions.update({f'bypass_{s}_prototypes':(s,) for s in STAGES})
    conditions['bypass_all_stage_prototypes']=STAGES
    return conditions


def original_evidence_counts(concept_out):
    valid=concept_out['visual_validity_mask'].bool()
    b=valid.shape[0]
    slots=valid.sum((1,2,3,4))
    mass=concept_out['visual_activations'].reshape(b,-1).double().sum(1)
    return [{'original_active_slots':int(slots[i]),'original_assignment_mass':float(mass[i])}
            for i in range(b)]


def evaluate(args):
    import train as cfg
    cfg.set_seed(args.seed)
    model=build_model(*_resolve_devices())
    checkpoint=load_checkpoint(model,args.checkpoint)
    if set(model.concept_creations)!=set(STAGES):
        raise ValueError('Expected concept creation modules for all four stages.')
    _check_method(model,'ExplainableVidSalModel','forward')
    for s,concept in model.concept_creations.items():
        if concept.top_k!=MODEL_TOP_K or concept.visual_assignment_mode!=MODEL_ASSIGNMENT_MODE:
            raise ValueError(f'{s}: assignment settings differ from requested baseline.')
    conditions=bypass_conditions()
    adapters={c:StagePrototypeBypass(model,stages) for c,stages in conditions.items() if stages}
    loader,dataset,window_len=build_loader()
    allowed=read_video_ids(args.evaluation_video_ids) if args.evaluation_video_ids else None
    functions=load_metric_functions()
    enabled=not args.no_amp and cfg._amp_enabled(model.output_device)
    dtype=cfg._amp_dtype(model.output_device)
    out=Path(args.output_dir).expanduser();out.mkdir(parents=True,exist_ok=True)
    result_path=out/'stage_prototype_bypass_results.json'
    result_path.write_text(json.dumps({'status':'incomplete','checkpoint':checkpoint['path']})+'\n')
    rows=[];seen=set();replay_error=None
    totals={c:{s:defaultdict(float) for s in STAGES} for c in conditions}
    call_totals={c:{s:0 for s in STAGES} for c in conditions}
    fields=['video','dataset_index','loader_batch','window_length','condition',*METRIC_NAMES,
            'fixation_pixels',*(f'delta_{m}' for m in METRIC_NAMES),'prototype_bypass_details']
    print(f'Checkpoint: {checkpoint["path"]}\nFeature-preserving prototype bypass, including all-stage feature-only; '
          f'mode={MODEL_ASSIGNMENT_MODE}, top_k={MODEL_TOP_K}; metric input={METRIC_INPUT}',flush=True)
    with (out/'window_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for batch,names,indices,rgb,density,fixations,length in evaluation_batches(loader,allowed,args.max_windows):
            baseline,cache=capture_baseline(model,rgb,enabled,dtype)
            original=baseline['concept_out']
            for s in STAGES:
                validate_baseline(model.concept_creations[s],original[s])
            if replay_error is None:
                replay=cached_prediction(model,cache,original,enabled,dtype)
                torch.testing.assert_close(replay['saliency_logits'],baseline['saliency_logits'],rtol=1e-5,atol=1e-6)
                replay_error=float((replay['saliency_logits'].float()-baseline['saliency_logits'].float()).abs().max())
                del replay
            with torch.inference_mode():
                base_scores=per_sample_metrics(baseline,density,fixations,functions)
                evidence={s:original_evidence_counts(original[s]) for s in STAGES}
            for condition,bypassed in conditions.items():
                if not bypassed:
                    scores=base_scores
                else:
                    adapter=adapters[condition]
                    with adapter:
                        # Every condition receives the identical baseline concept
                        # dictionaries and backbone/temporal feature tensors.
                        prediction=cached_prediction(model,cache,original,enabled,dtype)
                    for s,n in adapter.calls.items():
                        call_totals[condition][s]+=n
                    with torch.inference_mode():
                        scores=per_sample_metrics(prediction,density,fixations,functions)
                    del prediction
                for i,(v,index) in enumerate(zip(names,indices)):
                    details={s:{**evidence[s][i],
                                'prototype_path_bypassed':s in bypassed,
                                'unused_active_slots':evidence[s][i]['original_active_slots'] if s in bypassed else 0,
                                'unused_assignment_mass':evidence[s][i]['original_assignment_mass'] if s in bypassed else 0.}
                             for s in STAGES}
                    row={'video':v,'dataset_index':index,'loader_batch':batch,'window_length':length,
                         'condition':condition,**scores[i],
                         'prototype_bypass_details':json.dumps(details,sort_keys=True)}
                    for m in METRIC_NAMES:
                        base,value=base_scores[i][m],scores[i][m]
                        row[f'delta_{m}']=value-base if base is not None and value is not None else None
                    writer.writerow(row);rows.append(row)
                    for s,info in details.items():
                        totals[condition][s]['evaluated_windows']+=1
                        for key in ('original_active_slots','original_assignment_mass','unused_active_slots','unused_assignment_mass'):
                            totals[condition][s][key]+=info[key]
            seen.update(names);handle.flush()
            del baseline,cache,original
    if not rows:
        raise RuntimeError('No valid labeled validation windows were evaluated.')
    if allowed is not None and args.max_windows is None and seen!=allowed:
        raise ValueError(f'Videos missing valid windows: {sorted(allowed-seen)}')
    checkpoint_stat=Path(checkpoint['path']).stat()
    if (checkpoint_stat.st_size,checkpoint_stat.st_mtime_ns)!=(checkpoint['size_bytes'],checkpoint['mtime_ns']):
        raise RuntimeError('Checkpoint changed during evaluation; results are not finalized.')
    if any('forward' in model.saliency_prediction.fusion_blocks[s].__dict__ for s in STAGES):
        # Reviewed blocks use class methods; each runtime override must be gone.
        raise RuntimeError('A prototype-bypass forward override was not restored.')
    summary,contrasts,video_rows,exclusions=summarize(rows,list(conditions),args.bootstrap_repeats,args.seed)
    write_csv(out/'summary.csv',summary,['Condition',*METRIC_NAMES])
    write_csv(out/'paired_comparisons.csv',contrasts)
    write_csv(out/'video_results.csv',video_rows)
    evidence_summary=[]
    for c,stages in totals.items():
        for s,counts in stages.items():
            evidence_summary.append({'condition':c,'stage':s,'prototype_path_bypassed':s in conditions[c],
                                     **counts,'unused_slots_percent':100*counts['unused_active_slots']/counts['original_active_slots'],
                                     'unused_mass_percent':100*counts['unused_assignment_mass']/counts['original_assignment_mass']})
    write_csv(out/'bypass_evidence_summary.csv',evidence_summary)
    print('\nCondition                         CC          SIM          NSS')
    for row in summary:
        values=['undefined' if row[m] is None else f'{row[m]:.6f}' for m in METRIC_NAMES]
        print(f"{row['Condition']:<29} "+'  '.join(f'{v:>11}' for v in values))
    print('\nPaired delta = bypass minus baseline; whole-video bootstrap 95% CI')
    for row in contrasts:
        delta='undefined' if row['mean_delta'] is None else f"{row['mean_delta']:+.6f}"
        ci='unavailable' if row['ci95_low'] is None else f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]"
        print(f"{row['condition']}: {row['metric']} {delta}, {ci}; windows={row['windows']}, videos={row['videos']}")
    sources={}
    for name,fn in functions.items():
        src=inspect.getsourcefile(fn)
        sources[name]={'callable':f'{fn.__module__}.{fn.__name__}','source':src,
                       'source_sha256':hashlib.sha256(Path(src).read_bytes()).hexdigest() if src else None}
    result={'status':'complete' if args.max_windows is None else 'complete_pilot_subset',
            'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':checkpoint,
            'experiment':'feature_preserving_stage_prototype_path_bypass',
            'checkpoint_files_modified':False,'prototype_assignments_modified':False,
            'conditions':{c:{'bypassed_stages':list(stages),
                             'prototype_enabled_stages':[s for s in STAGES if s not in stages],
                             'decoder_uses_only_feature_maps':len(stages)==4} for c,stages in conditions.items()},
            'bypass_policy':{'film':'gamma=1, beta=0; concept projection/FiLM skipped',
                             'priority':'identity spatial gating; unary/pairwise scoring skipped',
                             'prototype_conditioned_update':'skipped; ordinary feature output retained',
                             'retained':'feature projection, configured backbone/previous-stage fusion, prev_scale, refinement, main/side heads, temporal aggregation and upsampling'},
            'scope_note':'Bypasses the complete concept-conditioning path including FiLM conditioning derived from the patch-embedding residual; this is broader than bank-lookup deletion.',
            'unbypassed_evidence':'exact baseline concept dictionaries and original forward methods',
            'baseline_decoder_replay_max_abs_logit_error':replay_error,
            'bypassed_block_call_counts':call_totals,
            'model_config':{'assignment_mode':MODEL_ASSIGNMENT_MODE,'top_k':MODEL_TOP_K,
                            'output_activation':'none','resize_to':[224,384],
                            'prototype_bottleneck_strength':cfg.PROTOTYPE_BOTTLENECK_STRENGTH,
                            'prototype_application_position':cfg.PROTOTYPE_APPLICATION_POSITION,
                            'fine_unary_mask_strength':getattr(cfg,'FINE_UNARY_MASK_STRENGTH',1.0),
                            'stage2_backbone_fusion_enabled':getattr(cfg,'STAGE2_BACKBONE_FUSION_ENABLED',True)},
            'metric_input':METRIC_INPUT,'metric_sources':sources,
            'aggregation':'equal-window mean on common finite windows per metric across all conditions',
            'confidence_intervals':'whole-video cluster bootstrap of equal-window paired differences',
            'bootstrap_repeats':args.bootstrap_repeats,'seed':args.seed,
            'validation_dir':str(Path(VAL_DATASET_DIR).resolve()),'stride':WINDOW_STRIDE,
            'window_len':window_len,'max_windows':args.max_windows,
            'evaluated_windows':len(rows)//len(conditions),'evaluated_videos':sorted(seen),
            'undefined_metric_exclusions':exclusions,'summary':summary,'paired_comparisons':contrasts,
            'bypass_evidence_summary':evidence_summary,
            'evidence_totals_note':'Baseline evidence UNUSED by bypassed blocks, not physically deleted; counts include all evaluated windows.'}
    temporary=result_path.with_suffix('.json.part')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(result_path)
    print(f'\nResults saved to {out.resolve()}')
    return result


def configure_cli():
    global VAL_DATASET_DIR,GPU_ID,WINDOW_LEN,BATCH_SIZE,NUM_WORKERS
    global WINDOW_STRIDE,MODEL_TOP_K,MODEL_ASSIGNMENT_MODE,METRIC_INPUT
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src-root',type=Path,help='Project src folder containing train.py, model/, pre_process/.')
    p.add_argument('--checkpoint',default=CHECKPOINT_PATH)
    p.add_argument('--val-dir',default=VAL_DATASET_DIR)
    p.add_argument('--output-dir',default='evaluation_outputs/stage_prototype_bypass')
    p.add_argument('--gpu-id',type=int,default=GPU_ID)
    p.add_argument('--window-len',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--num-workers',type=int)
    p.add_argument('--stride',type=int,default=WINDOW_STRIDE)
    p.add_argument('--top-k',type=int,default=MODEL_TOP_K)
    p.add_argument('--assignment-mode',choices=('topk_soft','hard_eval'),default=MODEL_ASSIGNMENT_MODE)
    p.add_argument('--metric-input',choices=('model_map','sigmoid'),default=METRIC_INPUT)
    p.add_argument('--evaluation-video-ids',help='Optional JSON video list restricting all conditions identically.')
    p.add_argument('--max-windows',type=int,help='Pilot subset only; omit for full validation.')
    p.add_argument('--bootstrap-repeats',type=int,default=10000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-amp',action='store_true')
    args=p.parse_args()
    for name in ('window_len','batch_size','stride','top_k','max_windows','bootstrap_repeats'):
        value=getattr(args,name)
        if value is not None and value<1:
            p.error(f'--{name.replace("_","-")} must be positive.')
    if args.num_workers is not None and args.num_workers<0:
        p.error('--num-workers must be nonnegative.')
    if args.top_k>512:
        p.error('--top-k exceeds the configured 512 prototypes.')
    root=args.src_root
    if root is None:
        candidates=[*Path(__file__).resolve().parents,Path.cwd(),Path.cwd()/'src']
        root=next((x for x in candidates if (x/'train.py').is_file()),None)
    if root is None or not (root/'train.py').is_file():
        p.error('Cannot find train.py; pass --src-root /path/to/ExplainableSaliency/src.')
    sys.path.insert(0,str(root.resolve()))
    VAL_DATASET_DIR=args.val_dir;GPU_ID=args.gpu_id;WINDOW_LEN=args.window_len
    BATCH_SIZE=args.batch_size;NUM_WORKERS=args.num_workers;WINDOW_STRIDE=args.stride
    MODEL_TOP_K=args.top_k;MODEL_ASSIGNMENT_MODE=args.assignment_mode;METRIC_INPUT=args.metric_input
    return args


if __name__=='__main__':
    evaluate(configure_cli())
