"""Remove baseline prototype evidence only at human-fixation patches, all stages.

Run: python fixated_regions_removal.py
Pilot: python fixated_regions_removal.py --max-windows 20
Place under src/ or pass --src-root /path/to/ExplainableSaliency/src.

Two PAIRED conditions on each validation window:
  baseline_all_stages: the original model, with evidence at every patch
  fixation_regions_deleted_all_stages: remove ALL valid prototype slots at
    fixation-containing patches at stages1,2,3,4 simultaneously.

The model first constructs its normal top-k assignments for ALL patches, without
using ground-truth labels. Regions are subsequently defined from the last REAL
frame's TRUE BINARY human fixation map: a patch is selected iff at least one
fixation pixel center maps into its spatial grid cell. Each fixation is mapped
directly to every stage's grid; fixation maps are never interpolated/downsampled
to make the mask. This also works when a patch grid is finer than the label map.
Input labels and RGB must already describe the same spatial field of view; the
project's standard validation resizing and the model's resize_to are retained.
Do not supply a smoothed saliency density map in place of a binary fixation map.

Only those patches lose sparse assignment weights, validity and active prototype
vectors. The same prototype ID remains available at other patches. Survivor
indices/weights, backbone/temporal features and patch-embedding residuals are
retained. Priority distributions, representations and main/side predictions
are recomputed without prototype replacement. The original empty-evidence
policy gives zero prototype priority/mixture and preserves configured residual
coefficients; it is NOT the feature-preserving bypass experiment.

One full baseline forward caches prototype-independent decoder features and
original assignments. All four masks use that baseline; no reassignment, top-5%
selection, fixed-ID list or random-control deletion. A no-deletion replay is
checked against baseline. Checkpoint weights and project source are unchanged.

CC, SIM, NSS are FULL-MAP metrics using the project metric functions. CC/SIM use
human saliency density; NSS uses true fixation pixels at the last real frame.
Default metric input is model_map with output_activation=none, matching earlier
evaluators; --metric-input sigmoid explicitly applies sigmoid(logits) once.
No-fixation windows have no deletion and remain in CC/SIM; their NSS is undefined
and excluded. Common finite paired windows are used per metric. Means weight
windows equally; 95% CIs resample whole videos and recompute the paired
equal-window mean. Delta = deleted minus baseline.

Outputs in evaluation_outputs/fixation_region_prototype_deletion/:
  summary.csv (Condition, CC, SIM, NSS, four regional effects and coverage)
  paired_comparisons.csv, window_results.csv, video_results.csv
  removed_evidence_summary.csv (actual slots/mass/patch coverage per stage)
  baseline_audit.jsonl (fixed selected patch indices for every window/stage)
  fixation_region_deletion_results.json (configuration, scores, CIs, exclusions)
All stages are required. Uses the reviewed top-k concept/decoder implementation;
source guards reject incompatible forward methods before intervening.
Regional intervention reports are also saved in region_error_window_results.csv,
region_error_summary.csv and region_error_comparisons.csv (existing filenames).
The four columns are delta_bias_removed, delta_mae_removed, delta_bias_retained,
and delta_mae_retained; summary.csv and window_results.csv include them too.
For each region R (actual fixation-patch deletion union or its complement):
  delta_bias = mean_R(P_intact - P_deleted)
  delta_mae = mean_R(abs(GT-P_deleted) - abs(GT-P_intact)).
Positive delta_bias means decreased saliency; positive delta_mae means increased
GT discrepancy. Values are UNSCALED and use the SAME masks for both predictions.
Baseline effects are zero in valid nonempty regions. Empty regions and zero-mass
GT have undefined effects (CSV blank, JSON null, console N/A).

Regional normalization MATCHES the high-priority deletion evaluator: resize
prediction bilinearly to GT resolution (align_corners=False), no SIM blur, shift
negative prediction minima to zero, normalize each whole map to unit mass, and
use uniform density for zero-mass predictions. GT is normalized by its sum.
CC/SIM/NSS still use the original project metric functions and retain their
existing paired comparisons. The four new effects have whole-video bootstrap CIs.
No old signed GT-Pred regional columns or computations remain.

Removed regions are the UNION of the fixed fixation-selected patch footprints
across all stages, mapped onto GT pixel centers. Its complement retains local
prototype evidence at every stage. Both regions are defined before intervention;
outside predictions may still change through normalization/context/decoding.
Coverage is saved on the GT pixel grid. A full-image signed change is zero by
unit-mass construction, even when the full-image MAE change is nonzero.


Save five reproducibly sampled evaluated windows (seed 42) under the directory
containing this .py: sal_maps/fixated_r/sample_<i>/, with i=1..5. Each contains
pred_map_org_<i>.png, pred_map_intercepted_<i>.png, gt_map_<i>.png and
original_rgb_<i>.png (last-frame RGB of the evaluated window, uint8, unsmoothed),
plus metadata.
Saved map PNGs use the same display post-processing as train.save_batch_maps:
each map is independently min-max normalized to [0, 1], then written with
matplotlib gray colormap (vmin=0, vmax=1). No unit-mass renormalization and no
Gaussian smoothing are applied to display copies. Metrics remain unsmoothed.
Sampling uses a separate reservoir RNG, adds no forward passes, and includes
all evaluated windows (including no-fixation/no-op windows). Fewer than five
windows save all available samples and print the actual count.
--sal-maps-dir overrides the sal_maps root; --sal-map-seed changes sampling.
Metadata is saved under sal_maps/fixated_r/manifest.json, separately from
priority-removal exports.
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

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import numpy as np
def unit_mass_prediction(prediction):
    """Use the same whole-map normalization for intact and intervened predictions."""
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.ndim != 2 or pred.size == 0 or not np.isfinite(pred).all():
        raise ValueError('Expected a finite nonempty [H,W] prediction map.')
    pred = pred - min(0., float(pred.min()))
    total = pred.sum()
    return pred / total if total > 0 else np.full_like(pred, 1. / pred.size)

def last_window_rgb_uint8(rgb):
    """RGB window [B,T,3,H,W] float in [0,1] -> [B,H,W,3] uint8 last frame."""
    import torch
    if not isinstance(rgb, torch.Tensor) or rgb.ndim != 5 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError('Expected nonempty [B,T,3,H,W] RGB windows.')
    frame = rgb[:, -1].detach().float().cpu().numpy()
    if not np.isfinite(frame).all():
        raise ValueError('RGB frames must be finite.')
    # Collate keeps RGB in [0,1]; clamp in case of slight overshoot.
    return np.transpose(np.clip(np.rint(frame * 255.), 0, 255).astype(np.uint8), (0, 2, 3, 1))

def minmax_01_for_display(array):
    """Match train.prepare_pred_for_visualization / prepare_last_saliency_for_visualization."""
    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 2 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty [H,W] map for display.')
    x = x - float(x.min())
    return x / (float(x.max()) + 1e-8)

def save_map_png_train_style(array, path):
    """Match train.save_map_png: per-map min-max to [0,1], gray PNG vmin/vmax=0/1."""
    import matplotlib.pyplot as plt
    display = minmax_01_for_display(array)
    plt.imsave(path, display, cmap='gray', vmin=0.0, vmax=1.0)
    return display

def regional_intervention_changes(prediction, baseline_prediction, ground_truth, marked):
    """Unscaled changes from intact baseline on the SAME removed/retained masks."""
    pred = np.asarray(prediction, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    marked = np.asarray(marked, dtype=bool)
    if pred.ndim != 2 or pred.shape != baseline.shape or pred.shape != gt.shape or pred.shape != marked.shape or pred.size == 0:
        raise ValueError('Predictions, GT and marked mask must share a nonempty [H,W] grid.')
    if not np.isfinite(pred).all() or not np.isfinite(baseline).all() or not np.isfinite(gt).all() or (gt < 0).any():
        raise ValueError('Regional effects require finite predictions and finite nonnegative GT.')
    inside_count = int(marked.sum())
    result = {'marked_regions_percent': 100. * inside_count / marked.size,
              'marked_pixels': inside_count, 'outside_pixels': marked.size - inside_count,
              'delta_bias_removed': None, 'delta_mae_removed': None,
              'delta_bias_retained': None, 'delta_mae_retained': None}
    if gt.sum() == 0:
        return result
    pred = unit_mass_prediction(pred)
    baseline = unit_mass_prediction(baseline)
    gt = gt / gt.sum()
    bias_change = baseline - pred  # GT cancels in the change of signed GT-Pred bias.
    mae_change = np.abs(gt - pred) - np.abs(gt - baseline)
    for region, name in ((marked, 'removed'), (~marked, 'retained')):
        if region.any():
            result[f'delta_bias_{name}'] = float(bias_change[region].mean())
            result[f'delta_mae_{name}'] = float(mae_change[region].mean())
    return result

class FixationSaliencyMapSampler:
    """Uniform reservoir over evaluated windows; memory holds at most five cases."""
    def __init__(self, seed=42, count=5):
        if count < 1:
            raise ValueError('The sample count must be positive.')
        self.rng = np.random.default_rng(seed)
        self.seed, self.count, self.seen = int(seed), int(count), 0
        self.samples = []

    def consider_batch(self, names, indices, batch, length, originals, ground_truth, rgb_frames):
        """Return batch index -> still-selected record after all reservoir updates."""
        if (len(names) != len(indices) or len(names) != len(originals)
                or len(names) != len(ground_truth) or len(names) != len(rgb_frames)):
            raise ValueError('Sample names, indices, maps and RGB frames must have equal batch size.')
        selected_slots = {}
        for i, (video, index) in enumerate(zip(names, indices)):
            self.seen += 1
            slot = len(self.samples) if len(self.samples) < self.count else int(self.rng.integers(self.seen))
            if slot >= self.count:
                continue
            original = np.asarray(originals[i], dtype=np.float64).copy()
            gt = np.asarray(ground_truth[i], dtype=np.float64).copy()
            rgb = np.asarray(rgb_frames[i])
            if original.ndim != 2 or gt.shape != original.shape:
                raise ValueError('Sample maps must share a [H,W] GT-aligned grid.')
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError('Expected [H,W,3] uint8 RGB frames.')
            record = {'video': str(video), 'dataset_index': int(index),
                      'loader_batch': int(batch), 'window_length': int(length),
                      'stream_index': self.seen - 1, 'original': original,
                      'gt': gt, 'rgb': rgb.copy(), 'intervened': {0.: original.copy()}}
            if slot == len(self.samples):
                self.samples.append(record)
            else:
                self.samples[slot] = record
            # An earlier sample from this same batch may have been replaced.
            selected_slots[slot] = i
        return {i: self.samples[slot] for slot, i in selected_slots.items()}

    def save(self, root):
        from PIL import Image
        root = (Path(root).expanduser() / 'fixated_r').resolve()
        doses = (1.,)  # the paired fixation-deletion prediction
        records = sorted(self.samples, key=lambda item: item['stream_index'])
        # Validate completeness before writing any images.
        for record in records:
            if any(p not in record['intervened'] for p in doses):
                raise RuntimeError('A sampled window is missing a deletion condition.')
        manifest = {'directory': str(root), 'requested_samples': self.count,
                    'saved_samples': len(records), 'evaluated_windows': self.seen,
                    'seed': self.seed, 'sampling': 'uniform reservoir without replacement over evaluated windows',
                    'condition': 'fixated_r', 'paired_original_and_intervened': True,
                    'intervention': 'fixation_regions_deleted_all_stages',
                    'normalization': 'per-map min-max to [0,1]; matches train.prepare_*_for_visualization',
                    'display_smoothing': None,
                    'display': 'matplotlib gray PNG vmin=0 vmax=1; matches train.save_map_png',
                    'rgb': 'last-frame uint8 RGB of the evaluated window; unsmoothed; may differ in H×W from GT-aligned maps',
                    'samples': []}
        for number, record in enumerate(records, 1):
            gt = np.asarray(record['gt'], dtype=np.float64)
            rgb = np.asarray(record['rgb'])
            if not np.isfinite(gt).all() or (gt < 0).any():
                raise ValueError('Cannot save nonfinite or negative GT density.')
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError('Cannot save missing or invalid RGB frame.')
            original = np.asarray(record['original'], dtype=np.float64)
            intervened = {p: np.asarray(record['intervened'][p], dtype=np.float64) for p in doses}
            metadata = {key: record[key] for key in
                        ('video', 'dataset_index', 'loader_batch', 'window_length', 'stream_index')}
            metadata.update({'sample': number, 'map_shape': list(gt.shape),
                             'rgb_shape': list(rgb.shape),
                             'display_min': 0., 'display_max': 1.,
                             'display_normalization': 'independent per-map min-max',
                             'files': {}})
            for percentage in doses:
                directory = root / f'sample_{number}'
                directory.mkdir(parents=True, exist_ok=True)
                paths = {}
                for name, array in ((f'pred_map_org_{number}', original),
                                    (f'pred_map_intercepted_{number}', intervened[percentage]),
                                    (f'gt_map_{number}', gt)):
                    path = directory / f'{name}.png'
                    save_map_png_train_style(array, path)
                    paths[name] = str(path.relative_to(root))
                rgb_name = f'original_rgb_{number}'
                rgb_path = directory / f'{rgb_name}.png'
                Image.fromarray(rgb, mode='RGB').save(rgb_path)
                paths[rgb_name] = str(rgb_path.relative_to(root))
                metadata['files']['fixated_r'] = paths
                sample_metadata = {key: value for key, value in metadata.items() if key != 'files'}
                sample_metadata.update({'condition': 'fixated_r'})
                (directory / 'sample_metadata.json').write_text(json.dumps(
                    sample_metadata, indent=2, allow_nan=False) + '\n')
            manifest['samples'].append(metadata)
        root.mkdir(parents=True, exist_ok=True)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(f'Saved {len(records)} random fixation-removal saliency-map triplets (+ original RGB) to {root}', flush=True)
        if len(records) < self.count:
            print(f'Only {self.seen} windows were evaluated; requested {self.count} samples.', flush=True)
        return manifest

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
STAGES=('stage1','stage2','stage3','stage4')
METRIC_NAMES=('CC','SIM','NSS')
REGION_EFFECT_FIELDS=('delta_bias_removed','delta_mae_removed','delta_bias_retained','delta_mae_retained')
REGION_COVERAGE_FIELDS=('marked_regions_percent','marked_pixels','outside_pixels')
CHECKPOINT_PATH='/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/best_dhf1k.pth'
VAL_DATASET_DIR='/data/quantization/zaima/videosal_datasets/dhf1k/val'
GPU_ID=1
WINDOW_LEN=BATCH_SIZE=NUM_WORKERS=None
WINDOW_STRIDE=32
MODEL_TOP_K=8
MODEL_ASSIGNMENT_MODE='topk_soft'
METRIC_INPUT='model_map'

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
    deleted_rows=deleted.reshape(rows,-1).any(-1)
    recomputed=F.normalize(safe,dim=-1).masked_fill(zero,0)
    out['visual_concept_representation'] = torch.where(
        deleted_rows.unsqueeze(-1),recomputed,original['visual_concept_representation'])
    agreement = F.cosine_similarity(q,out['visual_concept_representation'],dim=-1)
    out['visual_feature_concept_agreement'] = torch.where(
        deleted_rows,agreement.masked_fill(zero.squeeze(-1),0),original['visual_feature_concept_agreement'])
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


def validate_original_assignments(original, concept):
    indices=original['active_visual_prototype_indices']
    valid=original['visual_validity_mask'].bool()
    if indices.ndim!=5 or indices.shape[1]!=1 or valid.shape!=indices.shape:
        raise ValueError('Expected last-frame assignments with shape [B,1,H,W,K].')
    bank=int(concept.visual_concepts.shape[0])
    if (indices<0).any() or (indices>=bank).any():
        raise ValueError('Baseline prototype indices must be within the bank.')
    k=1 if concept.visual_assignment_mode=='hard_eval' else min(concept.top_k,bank)
    if not valid.sum(-1).eq(k).all():
        raise ValueError(f'Expected {k} valid original slots per patch.')
    a=original['visual_activations']
    rows=valid.numel()//valid.shape[-1]
    if a.shape!=(rows,bank) or not torch.isfinite(a).all() or (a<0).any():
        raise ValueError('Invalid baseline sparse assignment weights.')
    if not torch.allclose(a.sum(-1),torch.ones_like(a[:,0]),rtol=1e-5,atol=1e-6):
        raise ValueError('Original sparse weights must sum to one at every patch.')
    mapped=torch.zeros_like(a,dtype=torch.long)
    mapped.scatter_add_(1,indices.reshape(rows,-1),valid.reshape(rows,-1).long())
    if (mapped>1).any() or a.masked_select(mapped.eq(0)).ne(0).any():
        raise ValueError('Valid original slots must be unique and contain all assignment mass.')
    expected=original['visual_concept_logits'].topk(k,dim=-1).indices
    actual=indices.reshape(rows,-1)[:,:k]
    if not torch.equal(expected,actual) or not valid[...,:k].all():
        raise ValueError('Baseline slots differ from the original top-k similarity selection.')


def fixation_patch_mask(fixations,validity):
    """Point-preserving mapping from binary fixation pixel centers to patch cells.

    For source pixel x, patch column = floor((x+0.5)*W_patch/W_label),
    with the analogous row mapping. No interpolation, dilation or percentile.
    """
    if not isinstance(fixations,torch.Tensor) or fixations.ndim!=4 or fixations.shape[1]!=1:
        raise ValueError('Expected true last-frame fixation maps [B,1,H_label,W_label].')
    if validity.ndim!=5 or validity.shape[1]!=1 or validity.shape[0]!=fixations.shape[0]:
        raise ValueError('Expected last-frame patch validity [B,1,H_patch,W_patch,K].')
    if not torch.isfinite(fixations).all() or (fixations<0).any():
        raise ValueError('Fixation annotations must be finite and nonnegative.')
    if not ((fixations==0)|(fixations==1)|(fixations==255)).all():
        raise ValueError('Fixation regions require a binary 0/1 or 0/255 annotation map. '
                         'Do not use smoothed density maps or interpolate sparse fixation maps.')
    b,_,h,w,_=validity.shape
    lh,lw=fixations.shape[-2:]
    if min(h,w,lh,lw)<1:
        raise ValueError('Empty spatial grid.')
    points=fixations.to(validity.device).gt(0).nonzero(as_tuple=False)
    mask=torch.zeros((b,1,h,w),device=validity.device,dtype=torch.bool)
    if len(points):
        py=((2*points[:,2]+1)*h)//(2*lh)
        px=((2*points[:,3]+1)*w)//(2*lw)
        mask[points[:,0],0,py,px]=True
    return mask


def deletion_audit(original,changed,patch_mask,stats,stage):
    """Verify local deletion and freeze all evidence outside the human ROI."""
    valid=original['visual_validity_mask'].bool()
    expected=patch_mask.unsqueeze(-1).expand_as(valid)&valid
    if not torch.equal(changed['visual_validity_mask'],valid&~expected):
        raise RuntimeError(f'{stage}: deletion validity differs from the fixed fixation mask.')
    if changed['active_visual_prototypes'].masked_select(expected.unsqueeze(-1).expand_as(changed['active_visual_prototypes'])).ne(0).any():
        raise RuntimeError(f'{stage}: selected slots still contain prototype vectors.')
    if not torch.equal(changed['active_visual_prototype_indices'],original['active_visual_prototype_indices']):
        raise RuntimeError(f'{stage}: original prototype indices changed.')
    b,t,h,w,k=valid.shape
    a=original['visual_activations'];bank=a.shape[-1]
    patch_rows=patch_mask.reshape(-1)
    if changed['visual_activations'][patch_rows].ne(0).any():
        raise RuntimeError(f'{stage}: fixation patches still contain assignment evidence.')
    if not torch.equal(changed['visual_activations'][~patch_rows],a[~patch_rows]):
        raise RuntimeError(f'{stage}: assignment weights changed outside fixation patches.')
    if not torch.equal(changed['visual_concept_representation'][~patch_rows],original['visual_concept_representation'][~patch_rows]):
        raise RuntimeError(f'{stage}: concept representation changed outside fixation patches.')
    untouched=~expected.unsqueeze(-1).expand_as(changed['active_visual_prototypes'])
    if not torch.equal(changed['active_visual_prototypes'][untouched],original['active_visual_prototypes'][untouched]):
        raise RuntimeError(f'{stage}: active prototype vectors changed outside selected slots.')
    selected_patches=patch_mask.sum((1,2,3))
    all_slots=valid.sum((1,2,3,4))
    mass=a.reshape(b,t,h,w,bank).double().sum(-1)
    all_mass=mass.sum((1,2,3))
    selected_mass=(mass*patch_mask).sum((1,2,3))
    removed_slots=expected.sum((1,2,3,4))
    if not torch.equal(stats['removed_slots'],removed_slots):
        raise RuntimeError(f'{stage}: removed-slot count mismatch.')
    torch.testing.assert_close(stats['removed_assignment_weight'],selected_mass,rtol=1e-7,atol=1e-7)
    return [{'patch_grid':[h,w],'total_patches':h*w,
             'fixation_selected_patches':int(selected_patches[i]),
             'selected_patch_percent':100*int(selected_patches[i])/(h*w),
             'original_active_slots':int(all_slots[i]),'removed_active_slots':int(removed_slots[i]),
             'original_assignment_mass':float(all_mass[i]),'removed_assignment_mass':float(selected_mass[i]),
             'selected_patch_indices':patch_mask[i].reshape(-1).nonzero(as_tuple=False).flatten().cpu().tolist()}
            for i in range(b)]


def comparison_distributions(prediction,density):
    """Raw GT-grid maps for the SAME regional normalization as priority deletion.

    Project CC/SIM/NSS functions remain unchanged. These diagnostics use bilinear
    resizing without SIM blur; the shared regional helper normalizes each whole
    map to unit mass (shifting negative prediction minima, using uniform density
    for zero-mass predictions, and excluding zero-mass GT).
    """
    pred=prediction_for_metrics(prediction)
    if density.ndim!=4 or density.shape[1]!=1 or len(density)!=len(pred):
        raise ValueError('Expected one [B,1,H,W] GT density map per prediction.')
    if pred.shape[-2:]!=density.shape[-2:]:
        pred=F.interpolate(pred,size=density.shape[-2:],mode='bilinear',align_corners=False)
    return (pred.detach().double().cpu().numpy()[:,0],
            density.detach().double().cpu().numpy()[:,0])

def pixel_region_masks(patch_masks,target_hw):
    """Map fixed patch footprints onto GT pixel centers, then form their union."""
    lh,lw=target_hw
    result={}
    for stage,patch in patch_masks.items():
        if patch.ndim!=4 or patch.shape[1]!=1:
            raise ValueError('Expected fixed [B,1,H_patch,W_patch] ROI masks.')
        h,w=patch.shape[-2:]
        ys=((2*torch.arange(lh,device=patch.device)+1)*h)//(2*lh)
        xs=((2*torch.arange(lw,device=patch.device)+1)*w)//(2*lw)
        result[stage]=patch[:,:,ys[:,None],xs[None,:]]
    union=torch.stack(list(result.values()),dim=0).any(0)
    return {'joint_union':union,**result}

def regional_effect_rows(distributions,baseline_distributions,roi_masks,names,indices,batch,length,condition):
    """One row per window: changed/intact maps on identical union/complement."""
    pred,gt=distributions
    baseline,baseline_gt=baseline_distributions
    if pred.shape!=gt.shape or pred.shape!=baseline.shape or not np.array_equal(gt,baseline_gt):
        raise ValueError('Changed/intact predictions must share identical GT grids and labels.')
    if set(roi_masks)!={'joint_union'}:
        raise ValueError('Regional effects require the joint all-stage deletion footprint.')
    marked=roi_masks['joint_union'].detach().cpu().numpy()[:,0].astype(bool)
    if marked.shape!=pred.shape or len(names)!=len(pred) or len(indices)!=len(pred):
        raise ValueError('One actual deletion footprint is required per prediction.')
    rows=[]
    for i,(video,index) in enumerate(zip(names,indices)):
        effects=regional_intervention_changes(pred[i],baseline[i],gt[i],marked[i])
        status=('zero_mass_gt' if gt[i].sum()==0 else
                'empty_removed_region' if not marked[i].any() else
                'empty_retained_region' if marked[i].all() else 'ok')
        rows.append({'video':video,'dataset_index':index,'loader_batch':batch,
                     'window_length':length,'condition':condition,**effects,'status':status})
    return rows


def summarize_region_effects(rows,conditions,repeats,seed):
    """Common finite paired windows per effect; whole-video bootstrap of effects."""
    by_window=defaultdict(dict)
    for row in rows:
        key=(row['video'],row['dataset_index'])
        if row['condition'] in by_window[key]:
            raise ValueError('Duplicate window/condition in regional effects.')
        by_window[key][row['condition']]=row
    if any(set(window)!=set(conditions) for window in by_window.values()):
        raise ValueError('Regional effects require all paired conditions per window.')
    summaries={c:{'Condition':c} for c in conditions}
    comparisons=[];exclusions={}
    for c in conditions:
        for field in REGION_COVERAGE_FIELDS:
            summaries[c][field]=float(np.mean([window[c][field] for window in by_window.values()])) if by_window else None
    for field in REGION_EFFECT_FIELDS:
        paired=[window for window in by_window.values()
                if all(window[c][field] is not None and math.isfinite(window[c][field]) for c in conditions)]
        exclusions[field]={'included_windows':len(paired),'excluded_windows':len(by_window)-len(paired),
                           'included_videos':len({window[conditions[0]]['video'] for window in paired})}
        for c in conditions:
            values=[window[c][field] for window in paired]
            summaries[c][field]=float(np.mean(values)) if values else None
            summaries[c][f'valid_windows_{field}']=len(values)
            if c!=conditions[0]:
                videos=[window[c]['video'] for window in paired]
                ci=bootstrap_ci(values,videos,repeats,seed) if values else None
                comparisons.append({'condition':c,'reference':conditions[0],'metric':field,
                                    'mean_effect':summaries[c][field],
                                    'ci95_low':ci[0] if ci else None,'ci95_high':ci[1] if ci else None,
                                    'windows':len(values),'videos':len(set(videos))})
    return list(summaries.values()),comparisons,exclusions


def evaluate(args):
    import train as cfg
    importlib.import_module('PIL.Image')  # check PNG support before model evaluation
    saliency_sampler=FixationSaliencyMapSampler(seed=args.sal_map_seed,count=5)
    cfg.set_seed(cfg.SEED if args.seed is None else args.seed)
    model=build_model(*_resolve_devices())
    checkpoint=load_checkpoint(model,args.checkpoint)
    _check_method(model,'ExplainableVidSalModel','forward')
    if set(model.concept_creations)!=set(STAGES):
        raise ValueError('This experiment requires all four stages.')
    for s,c in model.concept_creations.items():
        if c.top_k!=MODEL_TOP_K or c.visual_assignment_mode!=MODEL_ASSIGNMENT_MODE:
            raise ValueError(f'{s}: assignment settings differ from the requested baseline.')
    adapters={s:LocalSlotDeletionAdapter(model,s) for s in STAGES}
    loader,dataset,window_len=build_loader()
    allowed=read_video_ids(args.video_ids) if args.video_ids else None
    enabled=not args.no_amp and cfg._amp_enabled(model.output_device)
    dtype=cfg._amp_dtype(model.output_device)
    functions=load_metric_functions()
    conditions=['baseline_all_stages','fixation_regions_deleted_all_stages']
    out=Path(args.output_dir).expanduser();out.mkdir(parents=True,exist_ok=True)
    result_path=out/'fixation_region_deletion_results.json'
    result_path.write_text(json.dumps({'status':'incomplete','checkpoint':checkpoint['path']})+'\n')
    totals={s:defaultdict(float) for s in STAGES}
    rows=[];region_rows=[];seen=set();replay_error=None;no_fixation_windows=0;intervened_windows=0
    joint_removed_pixels=joint_total_pixels=0
    fields=['video','dataset_index','loader_batch','window_length','condition',*METRIC_NAMES,
            'fixation_pixels',*REGION_COVERAGE_FIELDS,*REGION_EFFECT_FIELDS,*(f'delta_{m}' for m in METRIC_NAMES)]
    print(f'Checkpoint: {checkpoint["path"]}\nHuman fixation-patch evidence deletion at ALL stages; '
          f'mode={MODEL_ASSIGNMENT_MODE}, top_k={MODEL_TOP_K}; metric input={METRIC_INPUT}',flush=True)
    with (out/'window_results.csv').open('w',newline='') as handle, \
         (out/'baseline_audit.jsonl').open('w') as audit_file, \
         (out/'region_error_window_results.csv').open('w',newline='') as region_handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        region_fields=['video','dataset_index','loader_batch','window_length','condition',
                       *REGION_COVERAGE_FIELDS,*REGION_EFFECT_FIELDS,'status']
        region_writer=csv.DictWriter(region_handle,fieldnames=region_fields);region_writer.writeheader()
        for batch,names,indices,rgb,density,fixations,length in evaluation_batches(loader,allowed,args.max_windows):
            baseline,cache=capture_baseline(model,rgb,enabled,dtype)
            original=cache['concept_outs']
            patch_masks={}
            for s in STAGES:
                validate_original_assignments(original[s],model.concept_creations[s])
                patch_masks[s]=fixation_patch_mask(fixations,original[s]['visual_validity_mask'])
            if replay_error is None:
                replay=cached_prediction(model,cache,original,enabled,dtype)
                torch.testing.assert_close(replay['saliency_logits'],baseline['saliency_logits'],rtol=1e-5,atol=1e-6)
                replay_error=float((replay['saliency_logits'].float()-baseline['saliency_logits'].float()).abs().max())
                del replay
            with inference_context(model,enabled,dtype):
                base_scores=per_sample_metrics(baseline,density,fixations,functions)
                union=pixel_region_masks(patch_masks,density.shape[-2:])['joint_union']
                roi_masks={'joint_union':union}
                joint_removed_pixels+=int(union.sum());joint_total_pixels+=union.numel()
                base_distributions=comparison_distributions(baseline,density)
                base_region_rows=regional_effect_rows(base_distributions,base_distributions,roi_masks,names,indices,batch,length,conditions[0])
                changed=dict(original);details={}
                for s in STAGES:
                    mask=patch_masks[s].unsqueeze(-1).expand_as(original[s]['visual_validity_mask'])
                    if mask.any():
                        changed[s],stats=delete_local_slots(model.concept_creations[s],original[s],mask)
                    else:
                        stats={'removed_slots':torch.zeros(len(names),device=mask.device,dtype=torch.long),
                               'removed_assignment_weight':torch.zeros(len(names),device=mask.device,dtype=torch.float64)}
                    details[s]=deletion_audit(original[s],changed[s],patch_masks[s],stats,s)
            has_fixations=fixations.gt(0).reshape(len(names),-1).any(-1)
            no_fixation_windows+=int((~has_fixations).sum())
            intervened_windows+=int(has_fixations.sum())
            if has_fixations.any():
                with ExitStack() as stack:
                    for s in STAGES:
                        stack.enter_context(adapters[s])
                    prediction=cached_prediction(model,cache,changed,enabled,dtype)
                no_fix_idx=(~has_fixations).to(baseline['saliency_logits'].device)
                if no_fix_idx.any():
                    torch.testing.assert_close(prediction['saliency_logits'][no_fix_idx],baseline['saliency_logits'][no_fix_idx],rtol=1e-5,atol=1e-6)
                with torch.inference_mode():
                    deleted_scores=per_sample_metrics(prediction,density,fixations,functions)
                    deleted_distributions=comparison_distributions(prediction,density)
                    if not np.array_equal(deleted_distributions[1],base_distributions[1]):
                        raise RuntimeError('GT must remain identical across paired conditions.')
                del prediction
            else:
                deleted_scores=base_scores
                deleted_distributions=base_distributions
            sampled_records=saliency_sampler.consider_batch(
                names,indices,batch,length,base_distributions[0],base_distributions[1],
                last_window_rgb_uint8(rgb))
            for i,record in sampled_records.items():
                record['intervened'][1.]=deleted_distributions[0][i].copy()
            with torch.inference_mode():
                deleted_region_rows=regional_effect_rows(deleted_distributions,base_distributions,roi_masks,names,indices,batch,length,conditions[1])
            for region_row in base_region_rows+deleted_region_rows:
                region_writer.writerow(region_row);region_rows.append(region_row)
            effect_rows={conditions[0]:base_region_rows,conditions[1]:deleted_region_rows}
            for i,(v,index) in enumerate(zip(names,indices)):
                audit={'video':v,'dataset_index':index,'loader_batch':batch,'window_length':length,
                       'fixation_label_shape':list(fixations.shape[-2:]),'fixation_pixels':base_scores[i]['fixation_pixels'],
                       'stages':{s:details[s][i] for s in STAGES}}
                audit_file.write(json.dumps(audit,allow_nan=False)+'\n')
                for s,info in audit['stages'].items():
                    totals[s]['evaluated_windows']+=1
                    for key in ('total_patches','fixation_selected_patches','original_active_slots','removed_active_slots',
                                'original_assignment_mass','removed_assignment_mass'):
                        totals[s][key]+=info[key]
                for condition,scores in zip(conditions,(base_scores,deleted_scores)):
                    row={'video':v,'dataset_index':index,'loader_batch':batch,'window_length':length,
                         'condition':condition,**scores[i],
                         **{key:effect_rows[condition][i][key] for key in (*REGION_COVERAGE_FIELDS,*REGION_EFFECT_FIELDS)}}
                    for m in METRIC_NAMES:
                        base,value=base_scores[i][m],scores[i][m]
                        row[f'delta_{m}']=value-base if base is not None and value is not None else None
                    writer.writerow(row);rows.append(row)
            seen.update(names);handle.flush();audit_file.flush();region_handle.flush()
            del baseline,cache,original,changed,patch_masks,base_distributions,deleted_distributions,roi_masks,union
    if not rows:
        raise RuntimeError('No valid labeled validation windows were evaluated.')
    if allowed is not None and args.max_windows is None and seen!=allowed:
        raise ValueError(f'Requested videos missing valid windows: {sorted(allowed-seen)}')
    now=Path(checkpoint['path']).stat()
    if (now.st_size,now.st_mtime_ns)!=(checkpoint['size_bytes'],checkpoint['mtime_ns']):
        raise RuntimeError('Checkpoint changed during evaluation; results are not finalized.')
    summary,contrasts,video_rows,exclusions=summarize(rows,conditions,args.bootstrap_repeats,args.bootstrap_seed)
    write_csv(out/'paired_comparisons.csv',contrasts)
    write_csv(out/'video_results.csv',video_rows)
    region_summary,region_contrasts,region_exclusions=summarize_region_effects(
        region_rows,conditions,args.bootstrap_repeats,args.bootstrap_seed)
    effects_by_condition={row['Condition']:row for row in region_summary}
    for row in summary:
        row.update({key:value for key,value in effects_by_condition[row['Condition']].items() if key!='Condition'})
    write_csv(out/'summary.csv',summary)
    write_csv(out/'region_error_summary.csv',region_summary)
    write_csv(out/'region_error_comparisons.csv',region_contrasts)
    evidence_summary=[]
    for s,counts in totals.items():
        evidence_summary.append({'stage':s,**counts,
                                 'patches_percent':100*counts['fixation_selected_patches']/counts['total_patches'],
                                 'slots_percent':100*counts['removed_active_slots']/counts['original_active_slots'],
                                 'mass_percent':100*counts['removed_assignment_mass']/counts['original_assignment_mass']})
    write_csv(out/'removed_evidence_summary.csv',evidence_summary)
    print('\nCondition                                   CC          SIM          NSS  | ΔBias removed | ΔMAE removed | ΔBias retained | ΔMAE retained')
    for row in summary:
        values=['undefined' if row[m] is None else f'{row[m]:.6f}' for m in METRIC_NAMES]
        effects=' | '.join('N/A'.rjust(14) if row[key] is None else f'{row[key]:14.6e}' for key in REGION_EFFECT_FIELDS)
        print(f"{row['Condition']:<39} "+'  '.join(f'{v:>11}' for v in values)+' | '+effects)
    print('\nPaired delta = deleted minus baseline; whole-video bootstrap 95% CI')
    for row in contrasts:
        delta='undefined' if row['mean_delta'] is None else f"{row['mean_delta']:+.6f}"
        ci='unavailable' if row['ci95_low'] is None else f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]"
        print(f"{row['metric']}: {delta}, {ci}; windows={row['windows']}, videos={row['videos']}")
    print('\nRemoved prototype evidence over all evaluated windows (percent of baseline)')
    print('Stage     patches_percent  slots_percent  mass_percent')
    for row in evidence_summary:
        print(f"{row['stage']:<8} {row['patches_percent']:>16.4f} {row['slots_percent']:>14.4f} {row['mass_percent']:>13.4f}")
    print(f'Joint union of deletion footprints: {100*joint_removed_pixels/joint_total_pixels:.6f}% of GT-grid pixels')
    print('Regional effects are unscaled changes from the intact baseline on the SAME masks.')
    print('Positive ΔBias = decreased predicted saliency; positive ΔMAE = worse agreement with GT.')
    print('Removed = union of fixation-selected patch footprints; retained = its complement.')
    print('Baseline has zero effects on each nonempty, valid region; empty regions are N/A.')
    sal_maps_root=args.sal_maps_dir if args.sal_maps_dir is not None else Path(__file__).resolve().parent/'sal_maps'
    saliency_manifest=saliency_sampler.save(sal_maps_root)
    sources={}
    for name,fn in functions.items():
        src=inspect.getsourcefile(fn)
        sources[name]={'callable':f'{fn.__module__}.{fn.__name__}','source':src,
                       'source_sha256':hashlib.sha256(Path(src).read_bytes()).hexdigest() if src else None}
    result={'status':'complete' if args.max_windows is None else 'complete_pilot_subset',
            'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':checkpoint,
            'experiment':'joint_all_stage_human_fixation_patch_prototype_evidence_removal',
            'stages':list(STAGES),'conditions':conditions,'checkpoint_files_modified':False,
            'baseline_uses_ground_truth':False,'prototype_assignments_recomputed_after_deletion':False,
            'prototype_replacement':False,'prototype_IDs_removed_globally':False,
            'region_definition':'a stage patch containing at least one binary human fixation pixel center from the last real frame',
            'coordinate_mapping':'floor((coordinate+0.5)*patch_grid_size/fixation_label_size); no interpolation or dilation',
            'spatial_alignment_assumption':'Loader RGB/density/fixation annotations share the same field of view; standard validation/model resizing retained.',
            'deletion_policy':'all original valid slots at selected patches only; survivor weights/indices and feature residual retained; priorities/representations recomputed',
            'empty_patch_policy':'zero prototype priority/mixture; configured gate and residual coefficients retained',
            'feature_preserving_bypass':False,'baseline_decoder_replay_max_abs_logit_error':replay_error,
            'model_config':{'assignment_mode':MODEL_ASSIGNMENT_MODE,'top_k':MODEL_TOP_K,'output_activation':'none','resize_to':[224,384],
                            'prototype_bottleneck_strength':cfg.PROTOTYPE_BOTTLENECK_STRENGTH,
                            'prototype_application_position':cfg.PROTOTYPE_APPLICATION_POSITION,
                            'fine_unary_mask_strength':getattr(cfg,'FINE_UNARY_MASK_STRENGTH',1.0),
                            'stage2_backbone_fusion_enabled':getattr(cfg,'STAGE2_BACKBONE_FUSION_ENABLED',True)},
            'metric_region':'overall full map','metric_input':METRIC_INPUT,'metric_sources':sources,
            'aggregation':'equal-window paired means; common finite windows per metric',
            'confidence_intervals':'whole-video cluster bootstrap of equal-window paired differences',
            'bootstrap_repeats':args.bootstrap_repeats,'bootstrap_seed':args.bootstrap_seed,
            'seed':cfg.SEED if args.seed is None else args.seed,'validation_dir':str(Path(VAL_DATASET_DIR).resolve()),
            'stride':WINDOW_STRIDE,'window_len':window_len,'max_windows':args.max_windows,
            'video_ids_path':args.video_ids,'evaluated_windows':len(rows)//2,'evaluated_videos':sorted(seen),
            'windows_with_fixations':intervened_windows,'windows_without_fixations':no_fixation_windows,
            'no_fixation_policy':'no evidence removed; windows retained for CC/SIM; NSS undefined/excluded',
            'undefined_metric_exclusions':exclusions,'summary':summary,'paired_comparisons':contrasts,
            'removed_evidence_summary':evidence_summary,
            'regional_effect_normalization':'Same as high-priority evaluator: bilinear prediction resize to GT grid (align_corners=False), no SIM blur; whole-map unit mass; shift prediction minimum only when negative; zero-mass prediction is uniform; GT normalized by sum; zero-mass GT excluded.',
            'regional_effect_units':'unscaled unit-mass density per pixel; no display multipliers',
            'regional_effect_definitions':{'delta_bias_removed':'mean_R(P_intact-P_deleted)',
                                           'delta_mae_removed':'mean_R(abs(GT-P_deleted)-abs(GT-P_intact))',
                                           'delta_bias_retained':'mean_complement_R(P_intact-P_deleted)',
                                           'delta_mae_retained':'mean_complement_R(abs(GT-P_deleted)-abs(GT-P_intact))',
                                           'bias_sign':'positive = decreased predicted saliency, negative = increased predicted saliency',
                                           'mae_sign':'positive = worse agreement with GT, negative = improved agreement'},
            'regional_effect_masks':'Fixed fixation-selected patch footprints mapped to GT pixel centers. Removed is joint union across stages; retained is its complement. The same masks evaluate intact and changed predictions.',
            'regional_effect_scope_note':'Evidence is removed at at least one scale inside the union; the complement retains local evidence at every scale. Retained local evidence does not imply unchanged output saliency.',
            'regional_effect_baseline':'zero change on valid nonempty masks; empty regions undefined',
            'regional_effect_full_frame_note':'Full-frame signed bias change is zero by unit-mass construction; MAE change remains informative.',
            'joint_union_pixel_coverage':{'selected_pixels':joint_removed_pixels,'total_pixels':joint_total_pixels,
                                         'percent':100*joint_removed_pixels/joint_total_pixels},
            'regional_effect_summary':region_summary,'regional_effect_confidence_intervals':region_contrasts,
            'regional_effect_exclusions':region_exclusions,
            'saliency_map_samples':saliency_manifest,
            'evidence_totals_note':'All evaluated windows; metric-specific exclusions may differ.'}
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
    p.add_argument('--output-dir',default='evaluation_outputs/fixation_region_prototype_deletion')
    p.add_argument('--gpu-id',type=int,default=GPU_ID)
    p.add_argument('--window-len',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--workers',type=int)
    p.add_argument('--stride',type=int,default=WINDOW_STRIDE)
    p.add_argument('--top-k',type=int,default=MODEL_TOP_K)
    p.add_argument('--assignment-mode',choices=('topk_soft','hard_eval'),default=MODEL_ASSIGNMENT_MODE)
    p.add_argument('--metric-input',choices=('model_map','sigmoid'),default=METRIC_INPUT)
    p.add_argument('--video-ids',help='Optional JSON video list restricting both conditions identically.')
    p.add_argument('--max-windows',type=int,help='Pilot subset; omit for full validation.')
    p.add_argument('--bootstrap-repeats',type=int,default=10000)
    p.add_argument('--bootstrap-seed',type=int,default=2026)
    p.add_argument('--seed',type=int)
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--sal-maps-dir',type=Path,help='Root for sal_maps; exports go into fixated_r below it.')
    p.add_argument('--sal-map-seed',type=int,default=42,help='Separate RNG seed for five sampled PNG triplets.')
    args=p.parse_args()
    for name in ('window_len','batch_size','stride','top_k','max_windows','bootstrap_repeats'):
        value=getattr(args,name)
        if value is not None and value<1:
            p.error(f'--{name.replace("_","-")} must be positive.')
    if args.workers is not None and args.workers<0:
        p.error('--workers must be nonnegative.')
    if args.sal_map_seed<0:
        p.error('--sal-map-seed must be nonnegative.')
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
    BATCH_SIZE=args.batch_size;NUM_WORKERS=args.workers;WINDOW_STRIDE=args.stride
    MODEL_TOP_K=args.top_k;MODEL_ASSIGNMENT_MODE=args.assignment_mode;METRIC_INPUT=args.metric_input
    return args


if __name__=='__main__':
    evaluate(configure_cli())
