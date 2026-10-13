"""Delete local prototype evidence at each window's baseline top-5% patches.

Place beside train.py or in src/interpretability_exp/exp2d_prototype_deletion/.
Requires the reviewed topk_soft concept_creation.py and saliency_prediction.py.
Run: python eval_window_patch_deletion.py
Pilot: python eval_window_patch_deletion.py --max-windows 20 --controls 2
Dose study: python eval_window_patch_deletion.py --doses 1 2 4 8
Baseline-only comparison: python eval_window_patch_deletion.py --controls 0
Tests: python eval_window_patch_deletion.py --self-test

Default: stage4, top_k=8, all validation windows produced with stride=32,
5% of baseline patch-averaged logits, dose=8, five spatial-control repeats.
No fixed prototype-ID targets, selection manifest, or selection split is needed.
Optional --video-ids restricts evaluation to an existing JSON list of video IDs.

Each original prototype SLOT is deleted only at its selected patch. The same
prototype remains available elsewhere. Indices are frozen, surviving weights
are unchanged, patch-embedding residuals are retained, and downstream priorities,
representations, main/side prediction are recomputed. Empty patches use the
reviewed zero-priority/zero-mixture policy with existing residual coefficients.

One full baseline forward per batch captures prototype-independent decoder
features. Conditions replay the decoder from those exact cached features and
original assignments, without repeating the backbone or reassigning prototypes.
A no-deletion replay is checked against baseline on the first batch.

At dose=8, spatial controls remove all eight slots at an equal number of other
patches in the SAME window; each patch has unit assignment weight, so total
removed weight matches up to floating-point rounding. At smaller doses, spatial
controls approximately match removed weight and random-slot controls delete the
same number of slots at the SAME HIGH locations. All comparisons measure the
same frozen HIGH locations. Control matching uses baseline weights only.

Results are window_results.csv, video_results.csv, baseline_audit.jsonl,
optional patch_results.csv, and window_deletion_results.json in OUTPUT_DIR.
Delta = deletion minus baseline. Negative score/mass/rank deltas mean decreases;
positive HIGH-minus-control fractions mean more frequent decreases. Ranks are
midrank percentiles on [0,1], with 1 the highest. Video bootstrap CIs condition
on the sampled control masks; videos receive equal weight. This tests local
prototype-evidence necessity, not the semantic role of fixed prototype IDs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
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
from typing import Any, Dict, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_SRC_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "train.py").is_file()), None)
if _SRC_ROOT is not None:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CHECKPOINT_PATH = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/best_dhf1k.pth"
)

VAL_DATASET_DIR = "/data/quantization/zaima/videosal_datasets/dhf1k/val"

GPU_ID = 1

WINDOW_LEN = None

BATCH_SIZE = None

NUM_WORKERS = None

SEED = None

MODEL_TOP_K = 8

MODEL_ASSIGNMENT_MODE = "topk_soft"

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

# Experiment configuration. CLI options below override these constants.
TARGET_STAGE = "stage4"
TOP_PERCENT = 5.0
SELECTION_SCORE = "logit"  # "logit" or "mass"; choose before evaluation
WINDOW_STRIDE = 32
DELETION_COUNTS = (8,)
CONTROL_REPEATS = 5
CONTROL_NEAREST_POOL = 5
CONTROL_SEED = 42
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 2026
USE_TRAIN_AMP = True
EVALUATION_VIDEO_IDS_PATH = None  # None evaluates all available validation videos
MAX_WINDOWS = None
OUTPUT_DIR = "evaluation_outputs/window_high_patch_deletion"
SAVE_PATCH_ROWS = True

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

def _validate_model_configuration(model):
    for stage, concept in model.concept_creations.items():
        if concept.visual_assignment_mode != MODEL_ASSIGNMENT_MODE or concept.top_k != MODEL_TOP_K:
            raise RuntimeError(f"{stage}: expected {MODEL_ASSIGNMENT_MODE}, top_k={MODEL_TOP_K}; "
                               f"got {concept.visual_assignment_mode}, top_k={concept.top_k}.")
        if not model.saliency_prediction.fusion_blocks[stage].use_shared_concept_activations:
            raise RuntimeError(f"{stage}: shared concept activations are required for frozen-evidence deletion.")

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

def pool_map(x, hw):
    # Current decoder sizes are integer multiples of every concept grid.
    # This gives nonoverlapping cells with exactly equal area, without smoothing.
    h, w = hw
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] % h or x.shape[-1] % w:
        raise ValueError('Prediction dimensions must be divisible by the concept grid.')
    return F.avg_pool2d(x.float(), (x.shape[-2] // h, x.shape[-1] // w))[:, 0]

def validate_sparse_baseline(c, concept):
    """Validate ORIGINAL assignments only; deletion intentionally breaks these sums/counts."""
    indices = c['active_visual_prototype_indices']
    valid = c['visual_validity_mask'].bool()
    if indices.ndim != 5 or indices.shape[1] != 1 or valid.shape != indices.shape:
        raise ValueError('Expected original last-frame [B,1,H,W,K] indices and validity.')
    b, t, h, w, slots = indices.shape
    bank_size = int(concept.visual_concepts.shape[0])
    k = min(int(concept.top_k), bank_size)
    if slots != concept.top_k or not (valid.sum(-1) == k).all().item():
        raise ValueError(f'Original baseline must have exactly {k} valid top-k slots per patch; '
                         'install the new top-k model and check MODEL_ASSIGNMENT_MODE.')
    a = c['visual_activations'].reshape(b*t*h*w, bank_size)
    if not torch.isfinite(a).all().item() or (a < 0).any().item():
        raise ValueError('Original sparse activations must be finite and nonnegative.')
    if not (a.gt(0).sum(-1) == k).all().item():
        raise ValueError('Original activations must have exactly k positive entries per patch.')
    if not torch.allclose(a.sum(-1), torch.ones_like(a[:, 0]), rtol=1e-5, atol=1e-6):
        raise ValueError('Original top-k weights must sum to one. Surviving deletion weights are not checked here.')
    actual = indices.reshape(-1, slots)[:, :k]
    expected = c['visual_concept_logits'].topk(k, dim=-1).indices
    if not torch.equal(actual, expected) or not valid[..., :k].all().item():
        raise ValueError('Original prototype slots do not match the actual top-k similarity ranking.')
    if k < slots and valid[..., k:].any().item():
        raise ValueError('Bank-size padding must remain invalid.')
    if not a.gather(1, actual).gt(0).all().item():
        raise ValueError('Every selected original prototype must have a positive weight.')

def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

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


def input_batches(loader, allowed, max_windows):
    yielded = 0
    for batch_index, (batch, dataset_indices) in enumerate(tqdm(loader, desc='Validation', unit='batch')):
        if len(batch) != 6:
            raise ValueError('Expected filenames, RGB, saliency, fixation, n_frames, valid_mask.')
        names, rgb, _, _, lengths, _ = batch
        selected = [i for i,n in enumerate(names) if allowed is None or video_id(n) in allowed]
        # Split by actual temporal length so predictions never use a padded last frame.
        for length in sorted({int(lengths[i]) for i in selected}):
            if length < 1:
                raise ValueError('Empty input window.')
            idx = [i for i in selected if int(lengths[i]) == length]
            if max_windows is not None:
                idx = idx[:max(0, max_windows-yielded)]
            if not idx:
                return
            yielded += len(idx)
            yield batch_index, [video_id(names[i]) for i in idx], [int(dataset_indices[i]) for i in idx], rgb[idx,:length], length
            if max_windows is not None and yielded >= max_windows:
                return


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


def midrank_percentiles(scores):
    flat = scores.float().flatten(1).contiguous()
    if flat.shape[1] == 1:
        return torch.full_like(scores.float(),.5)
    sorted_scores = flat.sort(1).values.contiguous()
    left = torch.searchsorted(sorted_scores,flat,right=False)
    right = torch.searchsorted(sorted_scores,flat,right=True)
    rank = (left.float()+right.float()-1)*.5/(flat.shape[1]-1)
    return rank.view_as(scores)


def patch_maps(logits, hw):
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError('Nonfinite saliency logits.')
    logit = pool_map(logits,hw)
    density = logits.float().flatten(2).softmax(-1).view_as(logits)
    area = (logits.shape[-2]//hw[0])*(logits.shape[-1]//hw[1])
    mass = pool_map(density,hw)*area
    return {'logit':logit,'mass':mass,'logit_rank':midrank_percentiles(logit),
            'mass_rank':midrank_percentiles(mass)}


def high_patch_mask(scores, percent):
    b,h,w = scores.shape
    count = max(1,int(math.ceil(h*w*percent/100)))
    flat = scores.flatten(1)
    usable = flat.max(1).values > flat.min(1).values
    chosen = flat.argsort(dim=1,descending=True,stable=True)[:,:count]
    mask = torch.zeros_like(flat,dtype=torch.bool).scatter(1,chosen,True)
    mask = mask & usable[:,None]
    return mask.view(b,h,w),usable


def original_slot_weights(c):
    indices = c['active_visual_prototype_indices']
    b,t,h,w,k = indices.shape
    return c['visual_activations'].reshape(b,t,h,w,-1).gather(-1,indices)[:,0]


def condition_rng(video, dataset_index, dose, family, repeat):
    key = f'{CONTROL_SEED}|{video}|{dataset_index}|{dose}|{family}|{repeat}'.encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'little'))


def make_conditions(c, high, names, dataset_indices, doses, controls):
    """All masks are selected from baseline evidence before any deletion runs."""
    weights = original_slot_weights(c).detach().float().cpu().numpy()
    b,h,w,k = weights.shape
    high_cpu = high.cpu().numpy().reshape(b,-1)
    flat = weights.reshape(b,h*w,k)
    orders = np.argsort(-flat,axis=-1,kind='stable')
    conditions = []
    for dose in doses:
        families = [('high',1),('spatial',controls)]
        if dose < k:
            families.append(('random_slots',controls))
        for family,repeats in families:
            for repeat in range(repeats):
                mask = np.zeros((b,h*w,k),dtype=bool)
                errors = np.zeros(b)
                for i in range(b):
                    targets = np.flatnonzero(high_cpu[i])
                    if not len(targets):
                        continue
                    rng = condition_rng(names[i],dataset_indices[i],dose,family,repeat)
                    if family == 'high':
                        for p in targets:
                            mask[i,p,orders[i,p,:dose]]=True
                    elif family == 'random_slots':
                        for p in targets:
                            mask[i,p,rng.choice(k,size=dose,replace=False)]=True
                    else:
                        pool = np.flatnonzero(~high_cpu[i])
                        if len(pool) < len(targets):
                            raise ValueError('Insufficient non-HIGH patches for equal-count spatial controls.')
                        candidates = []
                        for p in rng.permutation(targets):
                            if dose == k:
                                pos = int(rng.integers(len(pool)))
                            else:
                                wanted = flat[i,p,orders[i,p,:dose]].sum()
                                candidate_mass = np.take_along_axis(flat[i,pool],orders[i,pool,:dose],axis=-1).sum(-1)
                                distance = np.abs(candidate_mass-wanted)
                                nearest = np.argsort(distance,kind='stable')[:min(CONTROL_NEAREST_POOL,len(pool))]
                                pos = int(rng.choice(nearest))
                                errors[i] += float(distance[pos])/len(targets)
                            candidates.append(int(pool[pos]))
                            pool = np.delete(pool,pos)
                        for p in candidates:
                            mask[i,p,orders[i,p,:dose]]=True
                condition = f'high_top{dose}' if family=='high' else f'{family}_top{dose}_{repeat:03d}'
                conditions.append({'name':condition,'family':family,'dose':dose,'repeat':repeat,
                    'slot_mask':torch.from_numpy(mask.reshape(b,1,h,w,k)).to(high.device),
                    'matching_weight_error':errors})
    return conditions
def region_measurements(base, changed, mask):
    n = int(mask.sum())
    if n == 0:
        return None
    result = {'target_patch_count':n}
    for metric in ('logit','mass','logit_rank','mass_rank'):
        delta = changed[metric]-base[metric]
        result[f'baseline_{metric}'] = float(base[metric][mask].mean())
        result[f'deleted_{metric}'] = float(changed[metric][mask].mean())
        result[f'delta_{metric}'] = float(delta[mask].mean())
        result[f'fraction_decreased_{metric}'] = float((delta[mask]<0).float().mean())
        result[f'outside_delta_{metric}'] = float(delta[~mask].mean()) if (~mask).any() else 0.0
        result[f'inside_minus_outside_{metric}'] = result[f'delta_{metric}']-result[f'outside_delta_{metric}']
    result['baseline_total_mass'] = float(base['mass'][mask].sum())
    result['deleted_total_mass'] = float(changed['mass'][mask].sum())
    result['delta_total_mass'] = result['deleted_total_mass']-result['baseline_total_mass']
    return result


IDENTITY_FIELDS = ['video','dataset_index','loader_batch','window_length','stage','condition','family','dose','repeat']
STAT_FIELDS = ['deleted_patches','removed_slots','removed_assignment_weight','empty_patches','matching_weight_error']
METRIC_FIELDS = ['target_patch_count']+[f'{prefix}_{metric}' for metric in ('logit','mass','logit_rank','mass_rank')
    for prefix in ('baseline','deleted','delta','fraction_decreased','outside_delta','inside_minus_outside')]+[
    'baseline_total_mass','deleted_total_mass','delta_total_mass']
SUMMARY_METRICS = [f'delta_{metric}' for metric in ('logit','mass','logit_rank','mass_rank')]+[
    'delta_total_mass','fraction_decreased_logit','fraction_decreased_mass','fraction_decreased_mass_rank',
    'inside_minus_outside_logit','inside_minus_outside_mass_rank']


def bootstrap_mean(values):
    values = np.asarray(values,dtype=float)
    result = {'n_videos':len(values),'mean':float(values.mean()),'video_std':float(values.std(ddof=1)) if len(values)>1 else None,
              'paired_video_bootstrap_95_ci':None}
    if len(values)>1:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means = []
        for start in range(0,BOOTSTRAP_REPEATS,1000):
            count = min(1000,BOOTSTRAP_REPEATS-start)
            idx = rng.integers(len(values),size=(count,len(values)))
            means.extend(values[idx].mean(1).tolist())
        result['paired_video_bootstrap_95_ci'] = np.quantile(means,[.025,.975]).tolist()
    return result


def summarize(video_rows,doses):
    summary = {}
    for dose in doses:
        rows = [r for r in video_rows if r['dose']==dose]
        grouped = defaultdict(dict)
        for row in rows:
            grouped[row['video']][row['condition']]=row
        key=f'high_top{dose}'
        videos=sorted(grouped)
        result={'n_videos':len(videos),'video_ids':videos,'high_deletion':{},
                'note':'All conditions measure the same frozen baseline HIGH locations.'}
        for metric in SUMMARY_METRICS:
            high=np.array([grouped[v][key][metric] for v in videos])
            result['high_deletion'][metric]=bootstrap_mean(high)
            for family in ('spatial','random_slots'):
                controls=sorted({r['condition'] for r in rows if r['family']==family})
                if not controls:
                    continue
                if not all(all(k in grouped[v] for k in controls) for v in videos):
                    raise RuntimeError('Incomplete within-video paired controls.')
                matrix=np.array([[grouped[v][k][metric] for k in controls] for v in videos])
                control=matrix.mean(1)
                result.setdefault(f'{family}_controls_mean',{})[metric]=bootstrap_mean(control)
                contrast=bootstrap_mean(high-control)
                contrast['high_mean']=float(high.mean())
                contrast['control_mean']=float(control.mean())
                contrast['control_set_video_mean_range']=[float(matrix.mean(0).min()),float(matrix.mean(0).max())]
                contrast['negative_means_larger_HIGH_decrease']=metric.startswith('delta_') or metric.startswith('inside_minus_outside_')
                result.setdefault(f'high_minus_{family}_controls',{})[metric]=contrast
        summary[f'dose_{dose}']=result
    return summary


def audit_record(c,high,conditions,i,video,dataset_index,length,base_maps):
    indices=c['active_visual_prototype_indices'][i,0].detach().cpu()
    weights=original_slot_weights(c)[i].detach().float().cpu()
    rows=[]
    for y,x in high[i].nonzero().tolist():
        rows.append({'row':y,'column':x,'prototype_ids':indices[y,x].tolist(),
                     'weights':weights[y,x].tolist(),
                     'baseline_logit':float(base_maps['logit'][i,y,x]),
                     'baseline_mass':float(base_maps['mass'][i,y,x])})
    deletions={}
    for condition in conditions:
        mask=condition['slot_mask'][i,0].cpu()
        cells=[]
        for y,x in mask.any(-1).nonzero().tolist():
            slots=mask[y,x].nonzero().flatten()
            cells.append({'row':y,'column':x,'slots':slots.tolist(),
                          'removed_ids':indices[y,x,slots].tolist(),
                          'removed_weights':weights[y,x,slots].tolist()})
        deletions[condition['name']]=cells
    return {'video':video,'dataset_index':dataset_index,'window_length':length,'stage':TARGET_STAGE,
            'selection_score':SELECTION_SCORE,'flat_baseline_skipped':not bool(high[i].any()),
            'baseline_high_patches':rows,'deletions':deletions}


@torch.no_grad()
def evaluate():
    import train as train_cfg
    if not 0 < TOP_PERCENT <= 50 or SELECTION_SCORE not in ('logit','mass'):
        raise ValueError('Use 0<TOP_PERCENT<=50 and SELECTION_SCORE logit or mass.')
    if not DELETION_COUNTS or any(r<1 or r>MODEL_TOP_K for r in DELETION_COUNTS):
        raise ValueError('Deletion doses must be between 1 and MODEL_TOP_K.')
    if CONTROL_REPEATS<0 or CONTROL_NEAREST_POOL<1 or WINDOW_STRIDE<1:
        raise ValueError('Control repeats must be nonnegative; nearest pool and stride positive.')
    if MAX_WINDOWS is not None and MAX_WINDOWS<1:
        raise ValueError('MAX_WINDOWS must be positive or None.')
    allowed=None if EVALUATION_VIDEO_IDS_PATH is None else read_video_ids(EVALUATION_VIDEO_IDS_PATH)
    output=Path(OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat(timespec='seconds')
    result_path=output/'window_deletion_results.json'
    config={'stage':TARGET_STAGE,'top_k':MODEL_TOP_K,'assignment_mode':MODEL_ASSIGNMENT_MODE,
            'top_percent':TOP_PERCENT,'selection_score':SELECTION_SCORE,'window_stride':WINDOW_STRIDE,
            'doses':list(DELETION_COUNTS),'control_repeats':CONTROL_REPEATS,'control_seed':CONTROL_SEED,
            'max_windows':MAX_WINDOWS,'evaluation_video_ids_path':EVALUATION_VIDEO_IDS_PATH}
    result_path.write_text(json.dumps({'status':'incomplete','started_at':started,'config':config},indent=2)+'\n')
    seed=train_cfg.SEED if SEED is None else SEED
    train_cfg.set_seed(seed)
    backbone_device,head_device=_resolve_devices()
    model=build_model(backbone_device,head_device)
    _check_method(model,'ExplainableVidSalModel','forward')
    checkpoint=load_checkpoint(model,CHECKPOINT_PATH)
    model.eval();model.allow_eval_concept_losses=False
    _validate_model_configuration(model)
    adapter=LocalSlotDeletionAdapter(model,TARGET_STAGE)
    loader,dataset,window_len=build_loader()
    enabled=USE_TRAIN_AMP and train_cfg._amp_enabled(head_device)
    dtype=train_cfg._amp_dtype(head_device)
    accumulator={}
    seen=set();total_windows=0;used_windows=0;flat_skipped=0
    replay_error=None
    patch_columns=IDENTITY_FIELDS+['row','column']+[f'{p}_{m}' for m in ('logit','mass','logit_rank','mass_rank') for p in ('baseline','deleted','delta')]
    windows_file=(output/'window_results.csv').open('w',newline='')
    audit_file=(output/'baseline_audit.jsonl').open('w')
    patch_file=(output/'patch_results.csv').open('w',newline='') if SAVE_PATCH_ROWS else None
    writer=csv.DictWriter(windows_file,fieldnames=IDENTITY_FIELDS+STAT_FIELDS+METRIC_FIELDS);writer.writeheader()
    patch_writer=csv.DictWriter(patch_file,fieldnames=patch_columns) if patch_file else None
    if patch_writer:patch_writer.writeheader()
    try:
        with adapter:
            for batch_index,names,dataset_indices,rgb,length in input_batches(loader,allowed,MAX_WINDOWS):
                baseline,cache=capture_baseline(model,rgb,enabled,dtype)
                original=cache['concept_outs'][TARGET_STAGE]
                validate_sparse_baseline(original,model.concept_creations[TARGET_STAGE])
                hw=original['visual_validity_mask'].shape[2:4]
                base_maps=patch_maps(baseline['saliency_logits'],hw)
                high,usable=high_patch_mask(base_maps[SELECTION_SCORE],TOP_PERCENT)
                seen.update(names);total_windows+=len(names)
                used_windows+=int(usable.sum());flat_skipped+=int((~usable).sum())
                conditions=make_conditions(original,high,names,dataset_indices,DELETION_COUNTS,CONTROL_REPEATS)
                # Same dtype, same inputs, same masked-priority equations.
                if replay_error is None:
                    replay=cached_prediction(model,cache,cache['concept_outs'],enabled,dtype)
                    replay_error=float((replay['saliency_logits'].float()-baseline['saliency_logits'].float()).abs().max())
                    torch.testing.assert_close(replay['saliency_logits'],baseline['saliency_logits'],rtol=1e-5,atol=1e-6)
                    del replay
                for i,v in enumerate(names):
                    audit_file.write(json.dumps(audit_record(original,high,conditions,i,v,dataset_indices[i],length,base_maps),allow_nan=False)+'\n')
                for condition in tqdm(conditions,desc='Cached decoder conditions',unit='condition',leave=False):
                    with inference_context(model,enabled,dtype):
                        changed,stats=delete_local_slots(model.concept_creations[TARGET_STAGE],original,condition['slot_mask'])
                    concept_outs={**cache['concept_outs'],TARGET_STAGE:changed}
                    prediction=cached_prediction(model,cache,concept_outs,enabled,dtype)
                    new_maps=patch_maps(prediction['saliency_logits'],hw)
                    for i,v in enumerate(names):
                        if not usable[i]:continue
                        identity={'video':v,'dataset_index':dataset_indices[i],'loader_batch':batch_index,
                            'window_length':length,'stage':TARGET_STAGE,'condition':condition['name'],
                            'family':condition['family'],'dose':condition['dose'],'repeat':condition['repeat']}
                        values=region_measurements({k:x[i] for k,x in base_maps.items()},
                                                   {k:x[i] for k,x in new_maps.items()},high[i])
                        record={**identity,**values,
                            'deleted_patches':int(stats['deleted_patches'][i]),
                            'removed_slots':int(stats['removed_slots'][i]),
                            'removed_assignment_weight':float(stats['removed_assignment_weight'][i]),
                            'empty_patches':int(stats['empty_patches'][i]),
                            'matching_weight_error':float(condition['matching_weight_error'][i])}
                        if record['deleted_patches']!=record['target_patch_count'] or record['removed_slots']!=record['target_patch_count']*condition['dose']:
                            raise AssertionError('Condition did not delete the prescribed local patch/slot count.')
                        writer.writerow(record)
                        key=(v,condition['name'])
                        if key not in accumulator:
                            accumulator[key]={'video':v,'condition':condition['name'],'family':condition['family'],
                                              'dose':condition['dose'],'window_count':0,
                                              'sums':{k:0. for k in STAT_FIELDS+METRIC_FIELDS}}
                        item=accumulator[key];item['window_count']+=1
                        for k in item['sums']:item['sums'][k]+=record[k]
                        if patch_writer:
                            for y,x in high[i].nonzero().tolist():
                                patch_record={**identity,'row':y,'column':x}
                                for metric in ('logit','mass','logit_rank','mass_rank'):
                                    bval=float(base_maps[metric][i,y,x]);dval=float(new_maps[metric][i,y,x])
                                    patch_record.update({f'baseline_{metric}':bval,f'deleted_{metric}':dval,f'delta_{metric}':dval-bval})
                                patch_writer.writerow(patch_record)
                    del changed,prediction,concept_outs,new_maps
                windows_file.flush();audit_file.flush()
                if patch_file:patch_file.flush()
                print(f'Batch {batch_index}: {len(names)} windows; baseline + {len(conditions)} cached decoder deletions',flush=True)
                del baseline,cache,original,conditions,base_maps
        if not accumulator:
            raise RuntimeError('No non-flat HIGH patch regions were evaluated.')
        if allowed is not None and MAX_WINDOWS is None and seen!=allowed:
            raise ValueError(f'Requested videos missing from validation loader: {sorted(allowed-seen)}')
        video_rows=[]
        for item in accumulator.values():
            count=item['window_count']
            video_rows.append({k:v for k,v in item.items() if k!='sums'} | {k:v/count for k,v in item['sums'].items()})
        write_csv(output/'video_results.csv',video_rows)
        summary=summarize(video_rows,DELETION_COUNTS)
        result={'status':'complete','started_at':started,'completed_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'checkpoint':CHECKPOINT_PATH,'checkpoint_epoch':checkpoint.get('epoch'),'weights_modified':False,
            'intervention':'patch_local_post_assignment_no_replacement','config':config,
            'num_dataset_windows':len(dataset),'num_evaluated_windows':total_windows,'num_used_windows':used_windows,
            'num_flat_windows_skipped':flat_skipped,'window_len':window_len,'video_ids':sorted(seen),
            'partial_dataset':MAX_WINDOWS is not None,'baseline_replay_max_abs_logit_error':replay_error,
            'selection':'top ceil(5% * patch_count), using configured percent/score; ties by raster order; flat grids skipped',
            'measurement':'same frozen HIGH cells for baseline, local deletion and all controls; equal-area pooling',
            'rank_definition':'midrank percentile in [0,1], 1 highest; computed independently for logits and mass',
            'aggregation':'equal windows within a video, then equal videos',
            'confidence_intervals':'paired video bootstrap, conditional on sampled baseline control masks',
            'empty_patch_policy':'zero prototype priority/mixture, preserve original patch-embedding residual and decoder coefficients',
            'full_dose_spatial_matching':'equal patch count and equal total assignment weight up to rounding',
            'partial_dose_spatial_matching':'approximate nearest removed-assignment-weight matching; report matching_weight_error',
            'random_slot_controls':'only for doses below k; same patches and slot count, weights may differ',
            'supported_method_fingerprints':SUPPORTED_METHODS,'supported_decoder_helper_fingerprints':SUPPORTED_DECODER_FUNCTIONS,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summary':summary}
        result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        for dose,sections in summary.items():
            high=sections['high_deletion']['delta_logit']
            print(f'{dose}: HIGH logit delta={high["mean"]:.6g}; 95% CI={high["paired_video_bootstrap_95_ci"]}')
            contrast=sections.get('high_minus_spatial_controls',{}).get('delta_logit')
            if contrast:print(f'  HIGH-spatial control={contrast["mean"]:.6g}; 95% CI={contrast["paired_video_bootstrap_95_ci"]}')
        print(f'Results: {result_path}',flush=True)
        return result
    except Exception as exc:
        result_path.write_text(json.dumps({'status':'failed','started_at':started,'config':config,
            'error':str(exc),'num_evaluated_windows':total_windows},indent=2)+'\n')
        raise
    finally:
        windows_file.close();audit_file.close()
        if patch_file:patch_file.close()
@torch.no_grad()
def self_test():
    from model.concept_creation import VisualConceptCreation
    from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder
    torch.manual_seed(42);torch.set_num_threads(1)
    scores=torch.tensor([[[3.,1.,1.]],[[0.,0.,0.]]])
    torch.testing.assert_close(midrank_percentiles(scores)[0],torch.tensor([[1.,.25,.25]]))
    mask,usable=high_patch_mask(scores,5)
    assert mask[0].sum()==1 and mask[1].sum()==0 and usable.tolist()==[True,False]
    logits=torch.tensor([[[[2.,2.,0.,0.],[2.,2.,0.,0.]]]])
    maps=patch_maps(logits,(1,2));torch.testing.assert_close(maps['mass'].sum(),torch.tensor(1.))
    zero=region_measurements({k:v[0] for k,v in maps.items()},{k:v[0] for k,v in maps.items()},torch.tensor([[True,False]]))
    assert all(zero[k]==0 for k in zero if k.startswith('delta_'))

    class Harness(torch.nn.Module):
        def __init__(self,residual,position):
            super().__init__();self.output_device=torch.device('cpu')
            self.concept_creations=torch.nn.ModuleDict({s:VisualConceptCreation(
                in_channels=8,concept_dim=8,num_concepts=16,hidden_dim=16,top_k=8,
                visual_assignment_mode='topk_soft',visual_concept_residual_weight=residual)
                for s in ('stage4','stage3','stage2','stage1')})
            self.saliency_prediction=ConceptGatedMultiScaleSaliencyDecoder(
                stage_channels={s:8 for s in self.concept_creations},concept_dim=8,decoder_channels=8,
                dropout=0,use_shared_concept_activations=True,prototype_application_position=position,
                temporal_aggregation='last',output_activation='none',stage2_backbone_fusion_enabled=False)
        def forward(self,features,**kwargs):
            concepts={s:c(features[s],return_losses=False) for s,c in self.concept_creations.items()}
            out=self.saliency_prediction(concept_outs=concepts,features_dict=features,
                                        output_size=(32,32),return_details=False)
            return {'concept_out':concepts,'saliency_logits':out['saliency_logits']}

    features={s:torch.randn(2,8,2,h,h) for s,h in (('stage4',2),('stage3',4),('stage2',8),('stage1',16))}
    # Identical stage4 embeddings guarantee the same IDs occur elsewhere: local
    # deletion must keep those exact IDs and their weights outside the chosen cell.
    features['stage4']=torch.randn(2,8,2,1,1).expand(2,8,2,2,2).contiguous()
    for residual in (0.,1.):
        for position in ('pre_refine','post_refine'):
            model=Harness(residual,position).eval()
            before={k:v.clone() for k,v in model.state_dict().items()}
            baseline,cache=capture_baseline(model,features)
            c=cache['concept_outs']['stage4'];validate_sparse_baseline(c,model.concept_creations['stage4'])
            high=torch.zeros((2,2,2),dtype=torch.bool);high[:,0,0]=True
            conditions=make_conditions(c,high,['a','b'],[5,9],[1,2,4,8],2)
            assert len(conditions)==18
            repeat=make_conditions(c,high,['a','b'],[5,9],[1,2,4,8],2)
            assert all(torch.equal(x['slot_mask'],y['slot_mask']) for x,y in zip(conditions,repeat))
            assert len(make_conditions(c,high,['a','b'],[5,9],[8],0))==1
            with LocalSlotDeletionAdapter(model,'stage4'):
                replay=cached_prediction(model,cache,cache['concept_outs'])
                torch.testing.assert_close(replay['saliency_logits'],baseline['saliency_logits'],rtol=0,atol=0)
                for condition in conditions:
                    changed,stats=delete_local_slots(model.concept_creations['stage4'],c,condition['slot_mask'])
                    assert torch.equal(changed['active_visual_prototype_indices'],c['active_visual_prototype_indices'])
                    dense_untouched=~condition['slot_mask'].any(-1).reshape(-1)
                    torch.testing.assert_close(changed['visual_activations'][dense_untouched],c['visual_activations'][dense_untouched],rtol=0,atol=0)
                    assert (stats['removed_slots']==condition['dose']).all()
                    if condition['family']=='spatial':
                        assert not (condition['slot_mask'][:,0].any(-1)&high).any()
                    elif condition['family']=='random_slots':
                        assert torch.equal(condition['slot_mask'][:,0].any(-1),high)
                    if condition['dose']==8:
                        torch.testing.assert_close(stats['removed_assignment_weight'],torch.ones(2,dtype=torch.float64),rtol=1e-5,atol=1e-6)
                # Delete all slots at one cell in sample 0; sample 1 stays untouched.
                local=torch.zeros_like(c['visual_validity_mask']);local[0,0,0,0,:]=True
                changed,stats=delete_local_slots(model.concept_creations['stage4'],c,local)
                torch.testing.assert_close(changed['visual_activations'][1:],c['visual_activations'][1:],rtol=0,atol=0)
                assert stats['removed_slots'].tolist()==[8,0]
                prediction=cached_prediction(model,cache,{**cache['concept_outs'],'stage4':changed})
                assert not torch.equal(prediction['saliency_logits'][0],baseline['saliency_logits'][0])
                torch.testing.assert_close(prediction['saliency_logits'][1],baseline['saliency_logits'][1],rtol=0,atol=0)
                # FP16/BF16 empty evidence uses the original residual policy safely.
                for dtype in (torch.float16,torch.bfloat16):
                    amp_base,amp_cache=capture_baseline(model,features,True,dtype)
                    original=amp_cache['concept_outs']['stage4']
                    with inference_context(model,True,dtype):
                        empty,_=delete_local_slots(model.concept_creations['stage4'],original,torch.ones_like(original['visual_validity_mask']))
                    amp_prediction=cached_prediction(model,amp_cache,{**amp_cache['concept_outs'],'stage4':empty},True,dtype)
                    assert torch.isfinite(amp_prediction['saliency_logits']).all()
                    assert torch.isfinite(empty['visual_concept_representation']).all()
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
            restored,_=capture_baseline(model,features)
            torch.testing.assert_close(restored['saliency_logits'],baseline['saliency_logits'],rtol=0,atol=0)
    # Paired summaries including a zero-controls run must still give HIGH deltas.
    rows=[{'video':v,'dose':8,'condition':condition,'family':family,
           **{m:(value if condition=='high_top8' else 0.) for m in SUMMARY_METRICS}}
          for v,value in [('a',-1.),('b',-3.)]
          for condition,family in [('high_top8','high'),('spatial_top8_000','spatial')]]
    result=summarize(rows,[8])['dose_8']
    assert result['high_deletion']['delta_logit']['mean']==-2
    assert result['high_minus_spatial_controls']['delta_logit']['mean']==-2
    solo=summarize([r for r in rows if r['family']=='high'],[8])['dose_8']
    assert solo['high_deletion']['delta_logit']['mean']==-2 and 'high_minus_spatial_controls' not in solo
    print('PASS: exact cached baseline replay, patch-local deletion, same IDs retained elsewhere, sample isolation, dose/control masks, equal full-dose weight, ties/ranks, zero controls, paired summaries, FP16/BF16 empty evidence, restoration and unchanged parameters.')


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--max-windows',type=int,default=MAX_WINDOWS)
    parser.add_argument('--controls',type=int,default=CONTROL_REPEATS)
    parser.add_argument('--doses',type=int,nargs='+',default=list(DELETION_COUNTS))
    parser.add_argument('--video-ids',default=EVALUATION_VIDEO_IDS_PATH)
    parser.add_argument('--stage',choices=['stage1','stage2','stage3','stage4'],default=TARGET_STAGE)
    parser.add_argument('--stride',type=int,default=WINDOW_STRIDE)
    parser.add_argument('--top-percent',type=float,default=TOP_PERCENT)
    parser.add_argument('--selection-score',choices=['logit','mass'],default=SELECTION_SCORE)
    parser.add_argument('--output-dir',default=OUTPUT_DIR)
    parser.add_argument('--gpu-id',type=int,default=GPU_ID)
    return parser.parse_args()


if __name__=='__main__':
    args=parse_args()
    if args.self_test:
        self_test()
    else:
        MAX_WINDOWS=args.max_windows;CONTROL_REPEATS=args.controls
        DELETION_COUNTS=tuple(sorted(set(args.doses)))
        EVALUATION_VIDEO_IDS_PATH=args.video_ids;TARGET_STAGE=args.stage
        WINDOW_STRIDE=args.stride;TOP_PERCENT=args.top_percent;SELECTION_SCORE=args.selection_score
        OUTPUT_DIR=args.output_dir;GPU_ID=args.gpu_id
        evaluate()
