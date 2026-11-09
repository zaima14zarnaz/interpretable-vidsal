"""Paired percentage deletion of active prototype evidence: CC, NSS and SIM.

Place in src/ or src/interpretability_exp/exp2d_prototype_deletion/.
Requires your current reviewed topk_soft concept/decoder files. Standalone: no
other deletion evaluator, prototype target list, or selection manifest required.

Run: python eval_percentage_prototype_deletion.py
Single dose: python eval_percentage_prototype_deletion.py --percentages 50
Pilot: python eval_percentage_prototype_deletion.py --percentages 50 --repeats 1 --max-windows 20
Stage ablation: python eval_percentage_prototype_deletion.py --stages stage4 --percentages 25 50 75 100
Hard top-1 baseline: python eval_percentage_prototype_deletion.py --assignment-mode hard_eval
Most-used first: python eval_percentage_prototype_deletion.py --order highest_usage
Tests: python eval_percentage_prototype_deletion.py --self-test

Percent means percent of ORIGINAL VALID assigned prototype SLOTS per sample at
EACH selected stage, not percent of bank IDs or assignment weight. Default:
random deletion at all four stages; 25/50/75/100 percent, three random repeats.
Masks are nested within a repeat and keyed by video, dataset index and stage,
so batching does not change the intervention. Count rounding: floor(N*p/100+.5).
Actual removed slots, weights and empty patches are saved per stage/window.

--order highest_usage ranks prototype IDs by their number of ORIGINAL VALID
patch assignments within EACH validation window and stage. Usage is a count,
not assignment weight or learned priority. All occurrences of the most-used ID
are ordered first, then the next ID. Equal usage ties use ascending prototype
ID; occurrences of the same ID use original patch/slot raster order. The requested
percentage STILL counts valid SLOTS, so a cutoff may remove only some occurrences
of the boundary ID. IDs are not removed globally. Counts come from the original
baseline and are NOT recomputed after deletion. Masks are deterministic/nested,
so one repeat is evaluated per percentage. This is per-window usage, not a
dataset-wide usage ranking. Use a separate output directory when comparing
orders to preserve each run's results.

Baseline and deletion use identical weights/model settings/windows. Original
assignments are frozen: removed evidence is zeroed only at its patch, surviving
weights/indices and patch-embedding residuals are preserved, priorities and
representations are recomputed, and empty patches use the reviewed zero-priority
policy with unchanged residual coefficients. Checkpoint and model files are not
modified. Backbone features are cached once; a no-deletion decoder replay is
checked against the full baseline before any intervention.

Default model settings match the current deletion evaluator (topk_soft, k=8).
Use hard_eval to evaluate hard top-1 assignments with the SAME current source.
Do not compare a different inference mode to a baseline from another run.

CC/NSS/SIM call your project's cc_score/nss_score/sim_score on last REAL frame
labels, never padded frames. Default --metric-input model_map matches the
reviewed training metric call: it passes model saliency_map as returned, without
extra sigmoid; CC/NSS/SIM apply their own normalization. --metric-input sigmoid
explicitly evaluates sigmoid(logits) instead. The choice is recorded. NSS uses
true binary fixation labels, never pseudo-fixations from saliency density.

Outputs: percentage_deletion_results.json, summary.csv, window_results.csv,
video_results.csv, in evaluation_outputs/percentage_prototype_deletion.
Console/summary window means weight each window equally (repeat effects averaged
first). Also report equal-video means and paired video-bootstrap delta CIs,
conditional on sampled masks. Delta = deleted minus baseline: negative CC/NSS/SIM
delta means worse. Undefined metrics are recorded as null and paired exclusions
are counted; they are never silently replaced with good scores.
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
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Sequence, Tuple

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
_SRC_ROOT = next((p for p in Path(__file__).resolve().parents if (p/'train.py').is_file()), None)
if _SRC_ROOT is not None:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CHECKPOINT_PATH = '/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/best_dhf1k.pth'
VAL_DATASET_DIR = '/data/quantization/zaima/videosal_datasets/dhf1k/val'
GPU_ID = 0
WINDOW_LEN = None
BATCH_SIZE = None
NUM_WORKERS = None
SEED = None
MODEL_TOP_K = 8
MODEL_ASSIGNMENT_MODE = 'topk_soft'
WINDOW_STRIDE = 32
OUTPUT_DIR = 'evaluation_outputs/percentage_prototype_deletion'
TARGET_STAGES = ('stage4','stage3','stage2','stage1')
DELETION_PERCENTAGES = (25.,50.,75.,100.)
DELETION_ORDER = 'random'  # random, highest_weight, or highest_usage (baseline valid-slot count per ID)
DELETION_REPEATS = 3
DELETION_SEED = 42
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 2026
EVALUATION_VIDEO_IDS_PATH = None
MAX_WINDOWS = None
METRIC_INPUT = 'model_map'  # same input convention as reviewed train.py
USE_TRAIN_AMP = True
METRIC_NAMES = ('CC','NSS','SIM')

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

SUPPORTED_DECODER_FUNCTIONS = {
    "_gather_active_visual_activations": "722da9a748a0258123ab832564c67486d58d17da3757f9b24444a17c59a75782",
    "_build_shared_visual_modality_from_concept_out": "71715fe923a2ac731e7cd23363ab815ee64363d333f17e3bcd33d092e9dbf21c",
    "_masked_softmax": "74966968817883fb18b0341cc60f637f936f3bd1b09b6f053ccbfbd3be8cf1ac"
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

def _check_decoder_functions():
    import model.saliency_prediction as decoder_module
    tree = ast.parse(inspect.getsource(decoder_module))
    for name, expected in SUPPORTED_DECODER_FUNCTIONS.items():
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        for child in ast.walk(node):
            if hasattr(child, "type_params") and not child.type_params:
                delattr(child, "type_params")
        actual = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unsupported decoder helper: {name}. Install the reviewed top-k files; do not bypass the check.")

def _allow_empty_visual_prototypes(block: Any, validity: torch.Tensor) -> None:
    # Preserve shape validation; intentionally relax the every-patch-valid check.
    if validity.dim() != 5:
        raise ValueError("visual_validity_mask must have shape [B,T,H,W,K].")

def _prioritize_allowing_empty(
    block: Any, modality_outs: list[Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """Original visual-only prioritization, allowing empty patches after deletion.

    Invalid slots are absent as priority candidates and as competitors, including
    the denominator of the factorized comparison. The original helper safely
    returns zeros if the entire sample has no surviving candidates.
    """
    import model.saliency_prediction as decoder_module

    if len(modality_outs) != 1:
        raise ValueError("This adapter supports the supplied visual-only model.")
    modality = modality_outs[0]
    part = block._build_activation_tokens(
        modality["projected_features"], modality["projected_prototypes"],
        modality["activations"], concept_id_offset=0,
    )
    tokens = part["tokens"]
    activations = part["activations"]
    B, T, N, _ = tokens.shape
    H, W = part["spatial_hw"]
    K = N // (H * W)
    validity_mask = modality["validity_mask"]
    if validity_mask.shape != (B, T, H, W, K):
        raise ValueError("Prototype token/validity dimensions do not match.")
    validity = validity_mask.reshape(B, T, N)
    unary = block.unary_priority_mlp(tokens).squeeze(-1)
    context = factorized_stats = None
    if block.priority_mode == "unary_mask":
        scores = unary
    elif block.priority_mode == "pairwise":
        context, factorized_stats = block._factorized_pairwise_context(
            tokens, activations, validity, part["x_coords"], part["y_coords"],
        )
        # Original norm diagnostics are undefined when ALL candidates are absent.
        # Keep empty-evidence diagnostics finite; this does not change prediction.
        for key in ("query_norm_mean", "key_norm_mean"):
            factorized_stats[key] = torch.nan_to_num(factorized_stats[key], nan=0.0)
        scores = unary + block.pairwise_strength * context
    else:
        raise ValueError(f"Unsupported priority_mode: {block.priority_mode}")
    logits = (scores / block.priority_temperature
              + torch.log(activations.clamp_min(0.0) + decoder_module._PRIORITY_EPS))
    priorities_flat = decoder_module._masked_softmax(logits, validity, dim=-1)
    priorities = priorities_flat.view(B, T, H, W, K).permute(0, 1, 4, 2, 3).contiguous()
    distribution = priorities.sum(dim=2)
    spatial_mean = distribution.mean(dim=(-2, -1), keepdim=True)
    gate = distribution / spatial_mean.clamp_min(decoder_module._PRIORITY_EPS)
    result = {
        "priority_mask": gate.unsqueeze(1),
        "patch_priority_gate": gate,
        "patch_priority_distribution": distribution,
        "patch_priority_map": distribution,
        "concept_priorities": priorities,
        "concept_validity": validity_mask,
        "unary_scores": unary.view(B, T, H, W, K).permute(0, 1, 4, 2, 3).contiguous(),
        "priority_mode": block.priority_mode,
        "pairwise_context_used": block.priority_mode == "pairwise",
    }
    if context is not None:
        result["pairwise_context"] = context.view(B, T, H, W, K).permute(0, 1, 4, 2, 3).contiguous()
    if factorized_stats is not None:
        result["factorized_comparison_stats"] = factorized_stats
    return result

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
        pretrained_backbone=True, freeze_backbone=train_cfg.FREEZE_BACKBONE,
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

def load_checkpoint(model, checkpoint_path: str) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint is missing required model weights: {missing[:5]}")
    if unexpected:
        print(f"Warning: ignored {len(unexpected)} unused checkpoint keys.")
    return checkpoint

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

class LocalSlotDeletionAdapter:
    """Allow empty patches while changing only frozen patch-local evidence."""
    def __init__(self, model, stage):
        self.model, self.stage = model, stage
        self.restores = []
        if stage not in model.concept_creations:
            raise ValueError(f'Unknown target stage {stage}.')
        concept = model.concept_creations[stage]
        block = model.saliency_prediction.fusion_blocks[stage]
        if not block.use_shared_concept_activations:
            raise RuntimeError('Local no-replacement deletion requires shared activations.')
        _check_decoder_functions()
        for name in ('forward','_build_visual_concepts_from_patches','_compute_visual_assignments','_extract_active_prototypes'):
            _check_method(concept,'VisualConceptCreation',name)
        for name in ('forward','_resolve_visual_activations','_prioritize_concept_patch_activations',
                     '_factorized_pairwise_context','_build_concept_conditioned_update'):
            _check_method(block,'SpatioTemporalConceptGatedFusionBlock',name)
        _check_method(model.saliency_prediction,'ConceptGatedMultiScaleSaliencyDecoder','forward')

    def __enter__(self):
        if self.model.training:
            raise RuntimeError('Local deletion is inference-only; call model.eval().')
        block = self.model.saliency_prediction.fusion_blocks[self.stage]
        for name, replacement in [('_assert_required_visual_prototypes',_allow_empty_visual_prototypes),
                                  ('_prioritize_concept_patch_activations',_prioritize_allowing_empty)]:
            own = name in block.__dict__
            self.restores.append((block,name,own,block.__dict__.get(name)))
            setattr(block,name,MethodType(replacement,block))
        return self

    def __exit__(self,*exc):
        for obj,name,own,old in reversed(self.restores):
            if own:
                setattr(obj,name,old)
            else:
                delattr(obj,name)
        self.restores.clear()

def delete_local_slots(concept, original, slot_mask):
    """Mask slot evidence only at its ORIGINAL patch; never delete an ID globally."""
    out = dict(original)
    indices = original['active_visual_prototype_indices']
    valid = original['visual_validity_mask'].bool()
    if slot_mask.shape != valid.shape or slot_mask.dtype != torch.bool:
        raise ValueError('Deletion mask must be boolean [B,1,H,W,K].')
    deleted = slot_mask & valid
    rows, bank_size = original['visual_activations'].shape
    # scatter_add handles invalid padded duplicate IDs safely (they add zero).
    removed_columns = torch.zeros((rows,bank_size),device=indices.device,dtype=torch.long)
    removed_columns.scatter_add_(1,indices.reshape(rows,-1),deleted.reshape(rows,-1).long())
    removed_columns = removed_columns > 0
    a = original['visual_activations'].masked_fill(removed_columns,0)
    out['visual_activations'] = a
    out['visual_validity_mask'] = valid & ~deleted
    out['active_visual_prototypes'] = original['active_visual_prototypes'].masked_fill(deleted.unsqueeze(-1),0)
    # Keep original indices, including at deleted slots, for auditability.
    q = original['visual_patch_embeddings']
    bank = F.normalize(concept.visual_concepts,dim=-1)
    representation = a @ bank
    if concept.visual_concept_residual_weight > 0:
        representation = representation + concept.visual_concept_residual_weight*q
    zero = ~representation.ne(0).any(-1,keepdim=True)
    safe = torch.where(zero,torch.ones_like(representation),representation)
    out['visual_concept_representation'] = F.normalize(safe,dim=-1).masked_fill(zero,0)
    agreement = F.cosine_similarity(q,out['visual_concept_representation'],dim=-1)
    out['visual_feature_concept_agreement'] = agreement.masked_fill(zero.squeeze(-1),0)
    out['visual_assignment_probs'] = original['visual_assignment_probs'].masked_fill(removed_columns,0)
    out['visual_concept_logits'] = original['visual_concept_logits'].masked_fill(removed_columns,-torch.inf)
    winner = original['visual_concept_indices']
    deleted_winner = removed_columns.gather(1,winner.unsqueeze(1)).squeeze(1)
    out['visual_concept_indices'] = winner.masked_fill(deleted_winner,-1)
    probs = out['visual_assignment_probs']
    out['visual_assignment_usage'] = probs.mean(0)
    out['visual_assignment_entropy'] = -(probs*(probs+1e-8).log()).sum(-1).mean()
    out['losses'] = {}
    b,t,h,w,k = valid.shape
    original_weights = original['visual_activations'].reshape(b,t,h,w,bank_size).gather(-1,indices)
    removed_weights = original_weights*deleted
    stats = {'removed_slots':deleted.sum((1,2,3,4)),
             'deleted_patches':deleted.any(-1).sum((1,2,3)),
             'removed_assignment_weight':removed_weights.double().sum((1,2,3,4)),
             'empty_patches':(~out['visual_validity_mask'].any(-1)).sum((1,2,3))}
    return out, stats

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
    if not torch.equal(mapped.gt(0),a.gt(0)):
        raise ValueError('Valid baseline slots must exactly match positive assignment evidence.')


def deletion_slot_mask(original, percentage, names, dataset_indices, stage, repeat):
    valid=original['visual_validity_mask'].bool()
    b=valid.shape[0]
    valid_cpu=valid.reshape(b,-1).cpu().numpy()
    indices=original['active_visual_prototype_indices']
    a=original['visual_activations']
    weights=a.reshape(*indices.shape[:-1],a.shape[-1]).gather(-1,indices)
    weights=weights.float().reshape(b,-1).cpu().numpy()
    masks=np.zeros_like(valid_cpu)
    for i,(name,index) in enumerate(zip(names,dataset_indices)):
        positions=np.flatnonzero(valid_cpu[i])
        count=int(math.floor(len(positions)*percentage/100.+.5))
        if DELETION_ORDER=='random':
            key=f'{DELETION_SEED}|{name}|{index}|{stage}|{repeat}'.encode()
            seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
            order=np.random.default_rng(seed).permutation(positions)
        elif DELETION_ORDER=='highest_weight':
            order=positions[np.argsort(-weights[i,positions],kind='stable')]
        elif DELETION_ORDER=='highest_usage':
            # A usage occurrence is one ORIGINAL valid patch-prototype slot.
            # Counts and ranking are local to this window/stage, never batch-wide.
            slot_ids=indices[i].reshape(-1).detach().cpu().numpy()[positions]
            usage=np.bincount(slot_ids,minlength=a.shape[-1])
            # Primary: descending usage. Secondary: ascending prototype ID.
            # Within an ID: original raster slot order. Weights do not rank usage.
            order=positions[np.lexsort((positions,slot_ids,-usage[slot_ids]))]
        else:
            raise ValueError(f'Unsupported deletion order: {DELETION_ORDER}')
        masks[i,order[:count]]=True
    return torch.from_numpy(masks.reshape(valid.shape)).to(valid.device)


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


def per_sample_metrics(prediction,density,fixations,functions):
    pred=prediction_for_metrics(prediction)
    density=density.to(pred.device)
    fixations=fixations.to(pred.device)
    if not torch.isfinite(density).all() or (density<0).any():
        raise ValueError('Ground-truth density must be finite and nonnegative.')
    if not torch.isfinite(fixations).all() or (fixations<0).any():
        raise ValueError('True fixation map must be finite and nonnegative.')
    binary=(fixations>0).float()
    scores=[]
    for i in range(len(pred)):
        row={}
        for metric,fn in functions.items():
            target=binary[i:i+1] if metric=='NSS' else density[i:i+1]
            value=float(fn(pred[i:i+1],target).detach().cpu())
            row[metric]=value if math.isfinite(value) else None
        row['fixation_pixels']=int(binary[i].sum())
        scores.append(row)
    return scores


def summarize_metrics(rows):
    by_window=defaultdict(dict)
    for row in rows:
        key=(row['video'],row['dataset_index'])
        by_window[key][row['condition']]=row
    videos=[]
    dose_records={p:[] for p in (0.,*DELETION_PERCENTAGES)}
    for (video,index),conditions in sorted(by_window.items()):
        baseline=conditions['baseline']
        for p in dose_records:
            dose=[baseline] if p==0 else [r for r in conditions.values() if r['percentage']==p]
            if not dose:raise RuntimeError('Incomplete paired percentage evaluation.')
            record={'video':video,'dataset_index':index}
            for metric in METRIC_NAMES:
                vals=[r[metric] for r in dose]
                deleted=float(np.mean(vals)) if all(v is not None for v in vals) else None
                base=baseline[metric]
                record[metric]=deleted
                record[f'baseline_{metric}']=base
                record[f'delta_{metric}']=deleted-base if deleted is not None and base is not None else None
            dose_records[p].append(record)
    summaries={}
    for p,records in dose_records.items():
        summary={'window_count':len(records),'window_mean':{},'video_mean':{},'paired_comparison':{}}
        for metric in METRIC_NAMES:
            values=[r[metric] for r in records if r[metric] is not None]
            summary['window_mean'][metric]=float(np.mean(values)) if values else None
            summary.setdefault('valid_window_counts',{})[metric]=len(values)
            groups=defaultdict(list)
            for r in records:groups[r['video']].append(r)
            video_values=[];paired_base=[];paired_deleted=[];paired_delta=[]
            valid_pairs=[r for r in records if r[f'delta_{metric}'] is not None]
            for video,items in sorted(groups.items()):
                finite=[r[metric] for r in items if r[metric] is not None]
                if finite:video_values.append(float(np.mean(finite)))
                pairs=[r for r in items if r[f'delta_{metric}'] is not None]
                if pairs:
                    bv=float(np.mean([r[f'baseline_{metric}'] for r in pairs]))
                    dv=float(np.mean([r[metric] for r in pairs]))
                    paired_base.append(bv);paired_deleted.append(dv);paired_delta.append(dv-bv)
            summary['video_mean'][metric]=float(np.mean(video_values)) if video_values else None
            ci=None
            if len(paired_delta)>=2:
                effect=np.asarray(paired_delta)
                rng=np.random.default_rng(BOOTSTRAP_SEED)
                means=[]
                for start in range(0,BOOTSTRAP_REPEATS,1000):
                    n=min(1000,BOOTSTRAP_REPEATS-start)
                    draw=rng.integers(len(effect),size=(n,len(effect)))
                    means.extend(effect[draw].mean(1).tolist())
                ci=np.quantile(means,[.025,.975]).tolist()
            summary['paired_comparison'][metric]={
                'paired_windows':len(valid_pairs),'excluded_windows':len(records)-len(valid_pairs),
                'paired_videos':len(paired_delta),
                'baseline_window_mean':float(np.mean([r[f'baseline_{metric}'] for r in valid_pairs])) if valid_pairs else None,
                'deleted_window_mean':float(np.mean([r[metric] for r in valid_pairs])) if valid_pairs else None,
                'delta_window_mean':float(np.mean([r[f'delta_{metric}'] for r in valid_pairs])) if valid_pairs else None,
                'baseline_video_mean':float(np.mean(paired_base)) if paired_base else None,
                'deleted_video_mean':float(np.mean(paired_deleted)) if paired_deleted else None,
                'delta_video_mean':float(np.mean(paired_delta)) if paired_delta else None,
                'paired_video_bootstrap_95_ci':ci}
        for video,items in sorted(groups.items()):
            record={'video':video,'percentage':p,'window_count':len(items)}
            for metric in METRIC_NAMES:
                finite=[r[metric] for r in items if r[metric] is not None]
                pairs=[r for r in items if r[f'delta_{metric}'] is not None]
                record[metric]=float(np.mean(finite)) if finite else None
                record[f'paired_windows_{metric}']=len(pairs)
                record[f'delta_{metric}']=float(np.mean([r[f'delta_{metric}'] for r in pairs])) if pairs else None
            videos.append(record)
        summaries[f'{p:g}']=summary
    return summaries,videos


def write_csv(path,rows):
    if not rows:path.write_text('');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def display_and_save_summary(output,summaries):
    def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
    print('\nPercentage removed |       CC |      NSS |      SIM |      dCC |     dNSS |     dSIM',flush=True)
    table=[]
    for p,summary in summaries.items():
        row={'percentage':float(p),**summary['window_mean']}
        row.update({f'delta_{m}':summary['paired_comparison'][m]['delta_window_mean'] for m in METRIC_NAMES})
        row.update({f'paired_windows_{m}':summary['paired_comparison'][m]['paired_windows'] for m in METRIC_NAMES})
        table.append(row)
        nums=[row[k] for k in (*METRIC_NAMES,*(f'delta_{m}' for m in METRIC_NAMES))]
        print(f'{float(p):17g}% | '+' | '.join(f'{fmt(v):>8}' for v in nums),flush=True)
    write_csv(output/'summary.csv',table)


def evaluate():
    import train as train_cfg
    output=Path(OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    result_path=output/'percentage_deletion_results.json'
    result_path.write_text(json.dumps({'status':'incomplete','checkpoint':CHECKPOINT_PATH})+'\n')
    train_cfg.set_seed(train_cfg.SEED if SEED is None else SEED)
    model=build_model(*_resolve_devices())
    checkpoint=load_checkpoint(model,CHECKPOINT_PATH)
    model.eval();model.allow_eval_concept_losses=False
    _check_method(model,'ExplainableVidSalModel','forward')
    for stage,c in model.concept_creations.items():
        if c.visual_assignment_mode!=MODEL_ASSIGNMENT_MODE or c.top_k!=MODEL_TOP_K:
            raise RuntimeError('Baseline configuration differs from the requested assignment settings.')
    adapters={stage:LocalSlotDeletionAdapter(model,stage) for stage in TARGET_STAGES}
    loader,dataset,window_len=build_loader()
    allowed=read_video_ids(EVALUATION_VIDEO_IDS_PATH) if EVALUATION_VIDEO_IDS_PATH else None
    enabled=USE_TRAIN_AMP and train_cfg._amp_enabled(model.output_device)
    dtype=train_cfg._amp_dtype(model.output_device)
    functions=load_metric_functions()
    conditions=[(p,r,f'delete_{p:g}_repeat_{r:03d}') for p in DELETION_PERCENTAGES
                for r in range(1 if p==100 or DELETION_ORDER!='random' else DELETION_REPEATS)]
    seen=set();rows=[];replay_error=None
    stat_totals={name:{s:defaultdict(float) for s in TARGET_STAGES} for _,_,name in conditions}
    stage_audit={}
    fields=['video','dataset_index','loader_batch','window_length','condition','percentage','repeat',
            *METRIC_NAMES,'fixation_pixels',*(f'baseline_{m}' for m in METRIC_NAMES),
            *(f'delta_{m}' for m in METRIC_NAMES),'deletion_details']
    print(f'Checkpoint: {CHECKPOINT_PATH}\nStages: {TARGET_STAGES}; slot deletion percentages: {DELETION_PERCENTAGES}; '
          f'mode={MODEL_ASSIGNMENT_MODE}, top_k={MODEL_TOP_K}; metric input={METRIC_INPUT}',flush=True)
    with (output/'window_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for batch_index,names,indices,rgb,density,fixations,length in evaluation_batches(loader,allowed,MAX_WINDOWS):
            baseline,cache=capture_baseline(model,rgb,enabled,dtype)
            concept_outs=baseline['concept_out']
            for stage in TARGET_STAGES:
                validate_baseline(model.concept_creations[stage],concept_outs[stage])
                c=concept_outs[stage];valid=c['visual_validity_mask'].sum(-1)
                audit=stage_audit.setdefault(stage,{'slots_per_patch_min':int(valid.min()),'slots_per_patch_max':int(valid.max())})
                audit['slots_per_patch_min']=min(audit['slots_per_patch_min'],int(valid.min()))
                audit['slots_per_patch_max']=max(audit['slots_per_patch_max'],int(valid.max()))
            if replay_error is None:
                replay=cached_prediction(model,cache,concept_outs,enabled,dtype)
                torch.testing.assert_close(replay['saliency_logits'],baseline['saliency_logits'],rtol=1e-5,atol=1e-6)
                replay_error=float((replay['saliency_logits'].float()-baseline['saliency_logits'].float()).abs().max())
                del replay
            with torch.inference_mode():base_scores=per_sample_metrics(baseline,density,fixations,functions)
            def append(condition,percentage,repeat,scores,details):
                for i,(video,index) in enumerate(zip(names,indices)):
                    row={'video':video,'dataset_index':index,'loader_batch':batch_index,'window_length':length,
                         'condition':condition,'percentage':percentage,'repeat':repeat,**scores[i],
                         'deletion_details':json.dumps(details[i],sort_keys=True)}
                    for m in METRIC_NAMES:
                        base=base_scores[i][m];value=scores[i][m]
                        row[f'baseline_{m}']=base
                        row[f'delta_{m}']=value-base if value is not None and base is not None else None
                    writer.writerow(row);rows.append(row)
                handle.flush()
            append('baseline',0.,-1,base_scores,[{} for _ in names]);seen.update(names)
            for percentage,repeat,name in conditions:
                changed_concepts=dict(concept_outs);details=[{} for _ in names]
                with inference_context(model,enabled,dtype):
                    for stage in TARGET_STAGES:
                        original=concept_outs[stage]
                        mask=deletion_slot_mask(original,percentage,names,indices,stage,repeat)
                        # A rounded zero dose must be a true identity intervention.
                        if mask.any():
                            changed_concepts[stage],stats=delete_local_slots(model.concept_creations[stage],original,mask)
                        else:
                            stats={'removed_slots':torch.zeros(len(names),device=mask.device,dtype=torch.long),
                                   'deleted_patches':torch.zeros(len(names),device=mask.device,dtype=torch.long),
                                   'removed_assignment_weight':torch.zeros(len(names),device=mask.device),
                                   'empty_patches':(~original['visual_validity_mask'].any(-1)).sum((1,2,3))}
                        original_count=original['visual_validity_mask'].sum((1,2,3,4))
                        original_weight=original['visual_activations'].reshape(len(names),-1).double().sum(1)
                        for i in range(len(names)):
                            item={k:float(v[i]) if v.dtype.is_floating_point else int(v[i]) for k,v in stats.items()}
                            item.update({'original_slots':int(original_count[i]),'original_assignment_weight':float(original_weight[i])})
                            item['actual_removed_slot_percent']=100*item['removed_slots']/item['original_slots']
                            item['actual_removed_weight_percent']=100*item['removed_assignment_weight']/item['original_assignment_weight']
                            details[i][stage]=item
                            for k,v in item.items():
                                if not k.endswith('_percent'):stat_totals[name][stage][k]+=v
                    with ExitStack() as stack:
                        for adapter in adapters.values():stack.enter_context(adapter)
                        changed=cached_prediction(model,cache,changed_concepts,enabled,dtype)
                    scores=per_sample_metrics(changed,density,fixations,functions)
                append(name,percentage,repeat,scores,details)
                del changed,changed_concepts
            # print(f'Batch {batch_index}: paired baseline + {len(conditions)} deletions; windows={len(rows)//(1+len(conditions))}',flush=True)
            del baseline,cache,concept_outs
    if not rows:raise RuntimeError('No valid labeled validation windows were evaluated.')
    if allowed is not None and MAX_WINDOWS is None and seen!=allowed:
        raise ValueError(f'Requested evaluation videos missing valid windows: {sorted(allowed-seen)}')
    summaries,video_rows=summarize_metrics(rows)
    write_csv(output/'video_results.csv',video_rows)
    display_and_save_summary(output,summaries)
    metric_provenance={}
    for key,fn in functions.items():
        source=inspect.getsourcefile(fn)
        metric_provenance[key]={'callable':f'{fn.__module__}.{fn.__name__}',
                               'source_path':source,
                               'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest() if source and Path(source).is_file() else None}
    result={'status':'complete' if MAX_WINDOWS is None else 'complete_pilot_subset',
            'checkpoint':CHECKPOINT_PATH,'checkpoint_epoch':checkpoint.get('epoch'),'checkpoint_weights_modified':False,
            'intervention':'patch_local_percentage_slot_deletion_post_assignment_no_replacement',
            'percentage_definition':'fraction of baseline VALID prototype slots per window at EACH selected stage; nearest integer floor(N*p/100+.5)',
            'stages':list(TARGET_STAGES),'percentages':list(DELETION_PERCENTAGES),'deletion_order':DELETION_ORDER,
            'random_seed':DELETION_SEED,'requested_repeats':DELETION_REPEATS,
            'usage_order_scope':'per_window_per_stage' if DELETION_ORDER=='highest_usage' else None,
            'usage_order_definition':'count of original valid patch-prototype assignments per ID' if DELETION_ORDER=='highest_usage' else None,
            'usage_order_ties':'ascending prototype ID, then raster patch-slot order' if DELETION_ORDER=='highest_usage' else None,
            'usage_order_cutoff':'valid-slot percentage; boundary ID may be partly removed' if DELETION_ORDER=='highest_usage' else None,
            'conditions':[{'percentage':p,'repeat':r,'condition':n} for p,r,n in conditions],
            'nested_masks':True,'empty_patch_policy':'zero prototype priority/mixture; retain existing feature and patch-embedding residual coefficients',
            'model_config':{'top_k':MODEL_TOP_K,'assignment_mode':MODEL_ASSIGNMENT_MODE,
                            'output_activation':model.saliency_prediction.output_activation},
            'baseline_replay_max_abs_logit_error':replay_error,'baseline_assignment_audit':stage_audit,
            'metric_input':METRIC_INPUT,'metric_functions':metric_provenance,
            'metric_normalization':'project metric helpers; prediction resized to target by reviewed functions; true fixation labels for NSS',
            'window_stride':WINDOW_STRIDE,'window_len':window_len,'dataset_windows':len(dataset),
            'evaluated_windows':len(rows)//(1+len(conditions)),'evaluation_videos':sorted(seen),
            'windows_without_fixations':sum(r['fixation_pixels']==0 for r in rows if r['condition']=='baseline'),
            'aggregation':'repeat means per window, then equal windows; equal-video results also provided; undefined metrics/pairs excluded with counts',
            'delta_sign':'deleted minus baseline; negative CC/NSS/SIM delta means worse',
            'uncertainty':'paired video bootstrap conditional on sampled masks; repeats averaged before resampling videos',
            'bootstrap_repeats':BOOTSTRAP_REPEATS,'bootstrap_seed':BOOTSTRAP_SEED,
            'deletion_stats':{n:{s:dict(v) for s,v in stages.items()} for n,stages in stat_totals.items()},
            'summary':summaries,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'evaluated_at':datetime.now(timezone.utc).isoformat(timespec='seconds')}
    result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Results: {result_path}',flush=True)
    return result


def configure_cli():
    global CHECKPOINT_PATH,VAL_DATASET_DIR,GPU_ID,OUTPUT_DIR,TARGET_STAGES,DELETION_PERCENTAGES
    global DELETION_ORDER,DELETION_REPEATS,DELETION_SEED,WINDOW_STRIDE,MAX_WINDOWS
    global EVALUATION_VIDEO_IDS_PATH,MODEL_ASSIGNMENT_MODE,MODEL_TOP_K,METRIC_INPUT
    global BATCH_SIZE,NUM_WORKERS,WINDOW_LEN,USE_TRAIN_AMP
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',default=CHECKPOINT_PATH)
    parser.add_argument('--val-dir',default=VAL_DATASET_DIR)
    parser.add_argument('--output-dir',default=OUTPUT_DIR)
    parser.add_argument('--gpu-id',type=int,default=GPU_ID)
    parser.add_argument('--percentages',type=float,nargs='+',default=list(DELETION_PERCENTAGES))
    parser.add_argument('--stages',nargs='+',choices=['stage1','stage2','stage3','stage4'],default=list(TARGET_STAGES))
    parser.add_argument('--order',choices=['random','highest_weight','highest_usage'],default=DELETION_ORDER)
    parser.add_argument('--repeats',type=int,default=DELETION_REPEATS)
    parser.add_argument('--deletion-seed',type=int,default=DELETION_SEED)
    parser.add_argument('--stride',type=int,default=WINDOW_STRIDE)
    parser.add_argument('--max-windows',type=int,default=MAX_WINDOWS)
    parser.add_argument('--video-ids',default=EVALUATION_VIDEO_IDS_PATH)
    parser.add_argument('--assignment-mode',choices=['topk_soft','hard_eval'],default=MODEL_ASSIGNMENT_MODE)
    parser.add_argument('--top-k',type=int,default=MODEL_TOP_K)
    parser.add_argument('--metric-input',choices=['model_map','sigmoid'],default=METRIC_INPUT)
    parser.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    parser.add_argument('--workers',type=int,default=NUM_WORKERS)
    parser.add_argument('--window-len',type=int,default=WINDOW_LEN)
    parser.add_argument('--no-amp',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if any(not math.isfinite(p) or not 0<=p<=100 for p in args.percentages):parser.error('Percentages must lie in [0,100].')
    for field in ['repeats','stride','top_k']:
        if getattr(args,field)<1:parser.error(f'{field} must be positive.')
    for field in ['max_windows','batch_size','window_len']:
        value=getattr(args,field)
        if value is not None and value<1:parser.error(f'{field} must be positive.')
    if args.workers is not None and args.workers<0:parser.error('workers must be nonnegative.')
    CHECKPOINT_PATH,VAL_DATASET_DIR,GPU_ID,OUTPUT_DIR=args.checkpoint,args.val_dir,args.gpu_id,args.output_dir
    TARGET_STAGES=tuple(dict.fromkeys(args.stages))
    DELETION_PERCENTAGES=tuple(sorted(set(p for p in args.percentages if p>0)))
    DELETION_ORDER,DELETION_REPEATS,DELETION_SEED=args.order,args.repeats,args.deletion_seed
    WINDOW_STRIDE,MAX_WINDOWS,EVALUATION_VIDEO_IDS_PATH=args.stride,args.max_windows,args.video_ids
    MODEL_ASSIGNMENT_MODE,MODEL_TOP_K,METRIC_INPUT=args.assignment_mode,args.top_k,args.metric_input
    BATCH_SIZE,NUM_WORKERS,WINDOW_LEN=args.batch_size,args.workers,args.window_len
    USE_TRAIN_AMP=not args.no_amp
    return args


def self_test():
    """Tests real reviewed concept/decoder code, without checkpoint or dataset."""
    from model.concept_creation import VisualConceptCreation
    from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder
    torch.manual_seed(42);torch.set_num_threads(1)
    class Harness(torch.nn.Module):
        def __init__(self,mode,residual):
            super().__init__();self.output_device=torch.device('cpu')
            self.concept_creations=torch.nn.ModuleDict({s:VisualConceptCreation(
                in_channels=8,concept_dim=8,num_concepts=16,hidden_dim=16,top_k=8,
                visual_assignment_mode=mode,visual_concept_residual_weight=residual)
                for s in ('stage4','stage3','stage2','stage1')})
            self.saliency_prediction=ConceptGatedMultiScaleSaliencyDecoder(
                stage_channels={s:8 for s in self.concept_creations},concept_dim=8,
                decoder_channels=8,dropout=0,use_shared_concept_activations=True,
                temporal_aggregation='last',output_activation='none',stage2_backbone_fusion_enabled=False)
        def forward(self,features,**kwargs):
            concepts={s:c(features[s],return_losses=False) for s,c in self.concept_creations.items()}
            pred=self.saliency_prediction(concepts,features,output_size=(32,32),return_details=False)
            return {**pred,'concept_out':concepts}
    features={s:torch.randn(2,8,2,n,n) for s,n in [('stage4',2),('stage3',4),('stage2',8),('stage1',16)]}
    for mode in ('topk_soft','hard_eval'):
        for residual in (0.,1.):
            model=Harness(mode,residual).eval()
            weights={k:v.clone() for k,v in model.state_dict().items()}
            with torch.no_grad():
                baseline,cache=capture_baseline(model,features)
                replay=cached_prediction(model,cache,baseline['concept_out'])
                torch.testing.assert_close(replay['saliency_logits'],baseline['saliency_logits'],rtol=0,atol=0)
                adapters={s:LocalSlotDeletionAdapter(model,s) for s in model.concept_creations}
                changed=dict(baseline['concept_out'])
                for s,original in baseline['concept_out'].items():
                    validate_baseline(model.concept_creations[s],original)
                    m25=deletion_slot_mask(original,25.,['a','b'],[3,9],s,0)
                    m50=deletion_slot_mask(original,50.,['a','b'],[3,9],s,0)
                    assert not (m25 & ~m50).any()
                    assert torch.equal(m25,deletion_slot_mask(original,25.,['a','b'],[3,9],s,0))
                    valid=original['visual_validity_mask']
                    n=valid.sum((1,2,3,4))
                    assert torch.equal(m50.sum((1,2,3,4)),(n*.5+.5).floor().long())
                    assert not (m50 & ~valid).any()
                    # The same video/window mask is unchanged when processed alone.
                    per_sample={'active_visual_prototype_indices':original['active_visual_prototype_indices'][1:2],
                                'visual_validity_mask':valid[1:2],
                                'visual_activations':original['visual_activations'].reshape(2,-1,16)[1].reshape(-1,16)}
                    assert torch.equal(m50[1:2],deletion_slot_mask(per_sample,50.,['b'],[9],s,0))
                    m100=deletion_slot_mask(original,100.,['a','b'],[3,9],s,0)
                    assert torch.equal(m100,valid)
                    changed[s],stats=delete_local_slots(model.concept_creations[s],original,m100)
                    assert not changed[s]['visual_activations'].count_nonzero()
                    assert not changed[s]['visual_validity_mask'].count_nonzero()
                    assert torch.equal(changed[s]['active_visual_prototype_indices'],original['active_visual_prototype_indices'])
                    torch.testing.assert_close(stats['removed_assignment_weight'],original['visual_activations'].reshape(2,-1).double().sum(1))
                    assert not deletion_slot_mask(original,0.,['a','b'],[3,9],s,0).any()
                    # Partial deletion leaves every surviving original weight intact.
                    partial,_=delete_local_slots(model.concept_creations[s],original,m50)
                    remaining=partial['visual_activations']>0
                    torch.testing.assert_close(partial['visual_activations'][remaining],original['visual_activations'][remaining],rtol=0,atol=0)
                with ExitStack() as stack:
                    for adapter in adapters.values():stack.enter_context(adapter)
                    pred=cached_prediction(model,cache,changed)
                    assert torch.isfinite(pred['saliency_logits']).all()
                    for dtype in (torch.float16,torch.bfloat16):
                        amp=cached_prediction(model,cache,changed,True,dtype)
                        assert torch.isfinite(amp['saliency_logits']).all()
                restored=cached_prediction(model,cache,baseline['concept_out'])
                torch.testing.assert_close(restored['saliency_logits'],baseline['saliency_logits'],rtol=0,atol=0)
                assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
    # Last-real labels for both temporal layouts; padding must never be used.
    target=torch.arange(2*3*2*2).reshape(2,3,2,2).float()
    expected=target[1:2,1:2]
    torch.testing.assert_close(last_real_map(target,2,[1]),expected)
    torch.testing.assert_close(last_real_map(target.unsqueeze(1),2,[1]),expected)
    torch.testing.assert_close(last_real_map(target.unsqueeze(2),2,[1]),expected)
    # Project metric definitions on a perfect map, plus explicit undefined CC.
    functions=load_metric_functions()
    density=torch.tensor([[[[0.,.2],[.5,1.]]]])
    fixation=torch.tensor([[[[0.,0.],[0.,1.]]]])
    prediction={'saliency_map':density,'saliency_logits':torch.logit(density.clamp(.001,.999))}
    scores=per_sample_metrics(prediction,density,fixation,functions)[0]
    if METRIC_INPUT=='model_map':assert np.isclose(scores['CC'],1.) and np.isclose(scores['SIM'],1.)
    assert scores['NSS']>0
    flat={'saliency_map':torch.ones_like(density)*.5,'saliency_logits':torch.zeros_like(density)}
    assert per_sample_metrics(flat,density,fixation,functions)[0]['CC'] is None
    # Paired aggregation averages random repeats BEFORE giving videos equal weight.
    test_rows=[]
    for v in ['a','b']:
        for condition,percentage,value in [('baseline',0.,1.),('delete_50_repeat_000',50.,.6),('delete_50_repeat_001',50.,.8)]:
            test_rows.append({'video':v,'dataset_index':0,'condition':condition,'percentage':percentage,'CC':value,'NSS':value,'SIM':value})
    previous=globals()['DELETION_PERCENTAGES'];globals()['DELETION_PERCENTAGES']=(50.,)
    try:
        summary,_=summarize_metrics(test_rows)
        assert np.isclose(summary['50']['window_mean']['CC'],.7)
        assert np.isclose(summary['50']['paired_comparison']['CC']['delta_video_mean'],-.3)
        assert np.allclose(summary['50']['paired_comparison']['CC']['paired_video_bootstrap_95_ci'],[-.3,-.3])
    finally:globals()['DELETION_PERCENTAGES']=previous
    print('PASS: hard top-1/top-8, exact nested slot counts, batch-invariant random masks, surviving weights/indices, cached baseline replay, all-stage empty evidence, AMP, residual policies, unchanged weights/restoration, real-frame labels, project metrics, undefined CC and paired repeat/video aggregation.')


if __name__=='__main__':
    args=configure_cli()
    if args.self_test:self_test()
    else:evaluate()
