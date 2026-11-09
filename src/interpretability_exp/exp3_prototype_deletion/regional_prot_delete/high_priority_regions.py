"""Highest-total-priority PATCH deletion of prototype evidence: CC, NSS and SIM.

Place in src/ or src/interpretability_exp/exp2d_prototype_deletion/.
Requires the same reviewed model/concept/decoder source as the supplied evaluator.

Run:   python eval_priority_patch_prototype_deletion.py --percentages 25 50
Pilot: python eval_priority_patch_prototype_deletion.py --percentages 25 50 --max-windows 20
Matched random-patch control:
       python eval_priority_patch_prototype_deletion.py --order random_patches --percentages 25 50 --output-dir evaluation_outputs/random_patch_control
Selection tests without PyTorch or project files:
       python eval_priority_patch_prototype_deletion.py --selection-self-test
Regional intervention tests without PyTorch or project files:
       python eval_priority_patch_prototype_deletion.py --regional-self-test
Full project tests:
       python eval_priority_patch_prototype_deletion.py --self-test

DEFAULT EXPERIMENT
For each original validation window and EACH selected stage (all four by default),
rank valid last-frame patches using the original assignment-weighted mean of
per-slot TOTAL learned scores. Total = unary + context priority, where the
model's context priority is pairwise_strength * pairwise_context. Stages in
unary_mask mode have zero context. Thus the patch score is
    sum_k a_k * (unary_k + pairwise_strength * context_k) / sum_k a_k.
This uses raw learned scores, before temperature, log-activation addition,
softmax, or spatial gate normalization. It is NOT prototype usage or similarity.

Remove ALL valid matches at the highest-scoring percentage of patches per stage.
Percentages count patches, not individual slots; round floor(N*p/100 + .5).
Rankings are frozen from the intact baseline, nested across doses, deterministic,
and tied by ascending patch raster index. Stages are ranked separately, never
pooled across incompatible score scales or resolutions. All selected stages are
intervened on together for each dose. Original surviving assignment weights,
prototype indices, backbone features, patch residuals and residual coefficients
are retained; priorities/representations are recomputed after deletion. No
replacement matching occurs. Selection uses no GT or perturbed-model outputs.

--order random_patches uses the SAME whole-patch budget for a matched control.
Legacy --order random/highest_weight/highest_usage retain the original SLOT-based
experiment for reproducibility; their percentages must not be called patch doses.
Deterministic orders use one repeat; random orders use --repeats (except 100%).
Each batch's no-deletion decoder replay with the empty-patch adapter is checked
against the unmodified baseline before any evidence is removed. Source checksum
checks remain enabled. No checkpoint or project model source is modified.

CC/NSS/SIM reuse project metric functions and last REAL-frame labels. NSS uses
true fixation labels. --metric-input model_map preserves the supplied evaluator's
normalization; sigmoid explicitly applies sigmoid to logits. Outputs include
percentage_deletion_results.json, summary.csv, window_results.csv, video_results.csv,
and patch_selection_audit.jsonl (ranked patch IDs and baseline score components).
Default output: evaluation_outputs/highest_priority_patch_prototype_deletion.
Regional intervention effects use each condition's actual removed/retained mask
for BOTH the intact prediction P0 and the intervened prediction Pd:
    delta_bias = mean_R(P0 - Pd)
    delta_mae  = mean_R(abs(GT - Pd) - abs(GT - P0)).
Positive delta_bias means predicted saliency decreased; negative means increased.
Positive delta_mae means agreement with GT worsened; negative means improved.
These four quantities are UNSCALED (no x100000/x10000 factors). Prediction is
bilinearly resized to the GT grid; each WHOLE map is normalized to unit mass
before regional averaging. Intact maps are cached once per evaluation batch.
Negative predictions are shifted up by their minimum; a zero-mass prediction
uses a uniform density. A zero-mass GT has undefined regional effects.
Marked regions are the UNION of selected patches across intervened stages,
projected to the GT grid by nearest-neighbor resizing. For legacy slot deletion,
a patch is marked when any valid slot is deleted. This is a patch footprint,
not a decoder receptive-field attribution. Actual marked image coverage is
reported separately from the requested per-stage removal percentage.
Empty regions are undefined (CSV blank, JSON null, console N/A). At zero dose
the removed region is empty and retained-region changes are zero; at full image
coverage the retained region is empty. A full-image signed change is zero by
unit-mass construction, even if predictions differ; MAE change can be nonzero.
Random repeats are averaged per window before window/video aggregation.
No CC/NSS/SIM changes relative to baseline are calculated or reported.

Five evaluated windows are sampled uniformly without replacement by reservoir
sampling with a separate RNG (default seed 42). The SAME windows are used at all
removal percentages, including the 0% baseline. No extra model forwards are run.
For random deletion repeats, PNGs show repeat 0; metric aggregation is unchanged.
After evaluation, save at <directory containing this .py>/sal_maps/<percentage>/
sample_<i>/{pred_map_org_<i>,pred_map_intercepted_<i>,gt_map_<i>,original_rgb_<i>}.png, i=1..5.
original_rgb is the last-frame RGB of the evaluated window (uint8, unsmoothed).
Override the root/seed with --sal-maps-dir and --sal-map-seed. Fewer than five
evaluated windows produce that many samples, with an explicit message.
Saved map PNGs use the same display post-processing as train.save_batch_maps:
each map is independently min-max normalized to [0, 1], then written with
matplotlib gray colormap (vmin=0, vmax=1). No unit-mass renormalization and no
Gaussian smoothing are applied to display copies. Evaluation metrics and
regional effects continue to use the original unsmoothed maps.
Sample metadata and a manifest identify the exact video/window.
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


def project_marked_patches(patch_mask, output_size):
    """Nearest-neighbor projection of [B,1,H,W] or [B,H,W] patch flags."""
    mask = np.asarray(patch_mask, dtype=bool)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3 or min(mask.shape) < 1:
        raise ValueError('Expected nonempty [B,1,H,W] or [B,H,W] patch flags.')
    height, width = map(int, output_size)
    if height < 1 or width < 1:
        raise ValueError('Output spatial dimensions must be positive.')
    # Match interpolate(mode="nearest"): floor(out_index * input_size / output_size).
    rows = np.arange(height, dtype=np.int64) * mask.shape[1] // height
    cols = np.arange(width, dtype=np.int64) * mask.shape[2] // width
    return mask[:, rows[:, None], cols[None, :]]


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


class SaliencyMapSampler:
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

    def save(self, root, percentages):
        from PIL import Image
        root = Path(root).expanduser().resolve()
        doses = tuple(dict.fromkeys(float(p) for p in (0., *percentages)))
        records = sorted(self.samples, key=lambda item: item['stream_index'])
        # Validate completeness before writing any images.
        for record in records:
            if any(p not in record['intervened'] for p in doses):
                raise RuntimeError('A sampled window is missing a deletion condition.')
        manifest = {'directory': str(root), 'requested_samples': self.count,
                    'saved_samples_per_percentage': len(records), 'evaluated_windows': self.seen,
                    'seed': self.seed, 'sampling': 'uniform reservoir without replacement over evaluated windows',
                    'same_windows_across_percentages': True, 'percentages': list(doses),
                    'deletion_repeat': 0,
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
                directory = root / f'{percentage:g}' / f'sample_{number}'
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
                metadata['files'][f'{percentage:g}'] = paths
                sample_metadata = {key: value for key, value in metadata.items() if key != 'files'}
                sample_metadata.update({'percentage': percentage, 'deletion_repeat': 0})
                (directory / 'sample_metadata.json').write_text(json.dumps(
                    sample_metadata, indent=2, allow_nan=False) + '\n')
            manifest['samples'].append(metadata)
        root.mkdir(parents=True, exist_ok=True)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(f'Saved {len(records)} random saliency-map triplets (+ original RGB) per percentage to {root}', flush=True)
        if len(records) < self.count:
            print(f'Only {self.seen} windows were evaluated; requested {self.count} samples.', flush=True)
        return manifest


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


def regional_self_test():
    gt = np.array([[3., 1.], [0., 0.]])
    pred = np.array([[1., 1.], [1., 1.]])
    marked = np.array([[True, False], [False, False]])
    result = regional_intervention_changes(pred, gt, gt, marked)
    assert np.isclose(result['delta_bias_removed'], .5)
    assert np.isclose(result['delta_mae_removed'], .5)
    assert np.isclose(result['delta_bias_retained'], -1. / 6.)
    assert np.isclose(result['delta_mae_retained'], 1. / 6.)
    assert result['marked_pixels'] == 1 and result['outside_pixels'] == 3
    assert result['marked_regions_percent'] == 25.
    scaled = regional_intervention_changes(pred * 7., gt * 3., gt * 11., marked)
    assert scaled == result  # whole-map scale invariance; no regional renormalization
    for flags, empty_name, full_name in ((np.zeros_like(marked), 'removed', 'retained'),
                                       (np.ones_like(marked), 'retained', 'removed')):
        actual = regional_intervention_changes(pred, gt, gt, flags)
        assert actual[f'delta_bias_{empty_name}'] is None
        assert actual[f'delta_mae_{empty_name}'] is None
        assert np.isclose(actual[f'delta_bias_{full_name}'], 0.)
        assert actual[f'delta_mae_{full_name}'] > 0
    assert regional_intervention_changes(pred, gt, np.zeros_like(gt), marked)['delta_bias_removed'] is None
    assert regional_intervention_changes(np.zeros_like(pred), gt, gt, marked) == result
    assert regional_intervention_changes(pred - 2., gt, gt, marked) == result
    unchanged = regional_intervention_changes(pred, pred, gt, marked)
    assert all(unchanged[f'delta_{metric}_{region}'] == 0.
               for metric in ('bias', 'mae') for region in ('removed', 'retained'))
    improved = regional_intervention_changes(gt, pred, gt, marked)
    for region in ('removed', 'retained'):
        for metric in ('bias', 'mae'):
            assert np.isclose(improved[f'delta_{metric}_{region}'], -result[f'delta_{metric}_{region}'])
    # A decrease in saliency can correct pre-existing OVERprediction: bias and
    # MAE changes must not be conflated.
    corrected = regional_intervention_changes(gt, np.array([[4., 0.], [0., 0.]]), gt, marked)
    assert corrected['delta_bias_removed'] > 0 and corrected['delta_mae_removed'] < 0
    coarse = project_marked_patches(np.array([[[[1, 0]]]]), (2, 4))
    fine = project_marked_patches(np.array([[[0, 0, 0, 1], [0, 0, 0, 0]]]), (2, 4))
    union = coarse | fine
    assert coarse.shape == (1, 2, 4) and coarse.sum() == 4 and union.sum() == 5
    assert not project_marked_patches(np.zeros((2, 1, 2, 2)), (3, 5)).any()
    assert project_marked_patches(np.ones((2, 1, 2, 2)), (3, 5)).all()
    print('PASS: matched-baseline bias/MAE changes, signs, unscaled units, '
          'identity/improvement interventions, whole-map normalization, empty regions/GT, '
          'zero/negative predictions, and multiscale mask projection/union.')


if __name__ == '__main__' and '--regional-self-test' in sys.argv:
    regional_self_test()
    raise SystemExit(0)


def rank_patch_positions(scores, valid_patches):
    """Descending finite score, with raster-index ties; never pool samples/stages."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_patches, dtype=bool).reshape(-1)
    if scores.shape != valid.shape:
        raise ValueError("Patch scores and patch validity must have identical shapes.")
    positions = np.flatnonzero(valid)
    if not np.isfinite(scores[positions]).all():
        raise ValueError("Valid baseline patch priorities must be finite.")
    return positions[np.argsort(-scores[positions], kind='stable')]


def select_patch_positions(scores, valid_patches, percentage, random_seed=None):
    """Select a nested whole-patch prefix; randomness changes only its ordering."""
    if not math.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError("Percentage must be finite and in [0,100].")
    order = rank_patch_positions(scores, valid_patches)
    if random_seed is not None:
        order = np.random.default_rng(random_seed).permutation(np.sort(order))
    count = int(math.floor(len(order) * percentage / 100. + .5))
    return order[:count]


def selection_self_test():
    """Synthetic behavioral tests; no model, checkpoint, dataset or PyTorch needed."""
    scores = np.array([.1, 9., 5., 5., -2., 100.])
    valid = np.array([1, 1, 1, 1, 1, 0], dtype=bool)
    assert rank_patch_positions(scores, valid).tolist() == [1, 2, 3, 0, 4]
    previous = set()
    for pct in (0., 10., 25., 50., 75., 100.):
        selected = select_patch_positions(scores, valid, pct)
        assert len(selected) == int(math.floor(valid.sum()*pct/100.+.5))
        assert previous <= set(selected)
        previous = set(selected)
        assert valid[selected].all()
    assert select_patch_positions(scores, valid, 50.).tolist() == [1, 2, 3]
    # All evidence slots of each selected patch are removed together, even for K=1.
    for k in (1, 8):
        slots = np.repeat(valid[:, None], k, axis=1)
        patch_mask = np.zeros_like(valid)
        patch_mask[select_patch_positions(scores, valid, 50.)] = True
        removed = slots & patch_mask[:, None]
        assert np.array_equal(removed.any(-1), patch_mask)
        assert np.all(removed.sum(-1)[patch_mask] == k)
        assert not removed[~patch_mask].any()
    # Total score ranking must respond to context, not just unary scores.
    unary = np.array([10., 2.]); context = np.array([-20., 5.])
    assert select_patch_positions(unary + context, [True, True], 50.).tolist() == [1]
    # Weights distinguish a weighted mean from max/sum of prototype scores.
    weights = np.array([[.9, .1], [.5, .5]])
    total = np.array([[0., 10.], [2., 2.]])
    weighted = (weights*total).sum(-1)/weights.sum(-1)
    assert rank_patch_positions(weighted, [True, True]).tolist() == [1, 0]
    # Random whole-patch control is deterministic, nested, and batch independent.
    for seed in (12, 99):
        a = select_patch_positions(scores, valid, 25., seed)
        b = select_patch_positions(scores, valid, 50., seed)
        assert set(a) <= set(b)
        assert np.array_equal(b, select_patch_positions(scores, valid, 50., seed))
    for bad in (np.nan, np.inf, -1., 101.):
        try:
            select_patch_positions(scores, valid, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid percentage was accepted.")
    try:
        rank_patch_positions([np.nan], [True])
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite valid score was accepted.")
    assert select_patch_positions([np.nan], [False], 100.).size == 0
    print("PASS: total-priority ranking, weighted reduction, stable ties, patch budgets, "
          "all-slot deletion, nested doses, random control, and invalid input rejection.")


if __name__ == '__main__' and '--selection-self-test' in sys.argv:
    selection_self_test()
    raise SystemExit(0)

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
OUTPUT_DIR = 'evaluation_outputs/highest_priority_patch_prototype_deletion'
TARGET_STAGES = ('stage4','stage3','stage2','stage1')
DELETION_PERCENTAGES = (25.,50.,75.,100.)
DELETION_ORDER = 'highest_priority'  # whole patches; random_patches is the matched control
DELETION_REPEATS = 3  # only used by random/random_patches
DELETION_SEED = 42
EVALUATION_VIDEO_IDS_PATH = None
MAX_WINDOWS = None
METRIC_INPUT = 'model_map'  # same input convention as reviewed train.py
USE_TRAIN_AMP = True
SAL_MAPS_DIR = None  # default: directory containing this .py / sal_maps
SAL_MAP_SEED = 42
METRIC_NAMES = ('CC','NSS','SIM')
REGIONAL_NAMES = ('marked_regions_percent', 'marked_pixels', 'outside_pixels',
                  'delta_bias_removed', 'delta_mae_removed',
                  'delta_bias_retained', 'delta_mae_retained')
REPORT_NAMES = METRIC_NAMES + REGIONAL_NAMES

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
        "total_priority_scores": scores.view(B, T, H, W, K).permute(0, 1, 4, 2, 3).contiguous(),
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
    def __init__(self, model, stage, priority_sink=None):
        self.model, self.stage = model, stage
        self.priority_sink = priority_sink
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
        def prioritize_and_capture(block, modality_outs):
            result = _prioritize_allowing_empty(block, modality_outs)
            if self.priority_sink is not None:
                if self.stage in self.priority_sink:
                    raise RuntimeError(f"Stage {self.stage} prioritized more than once in baseline replay.")
                self.priority_sink[self.stage] = {
                    key: result[key].detach().float().clone()
                    for key in ('total_priority_scores', 'unary_scores', 'pairwise_context')
                    if key in result
                }
            return result
        for name, replacement in [('_assert_required_visual_prototypes',_allow_empty_visual_prototypes),
                                  ('_prioritize_concept_patch_activations',prioritize_and_capture)]:
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


def baseline_patch_priorities(original, captured, block):
    """Original activation-weighted mean of raw (unary + scaled context) scores."""
    valid = original['visual_validity_mask'].bool()
    indices = original['active_visual_prototype_indices']
    a = original['visual_activations']
    weights = a.reshape(*indices.shape[:-1], a.shape[-1]).gather(-1, indices).float()
    weights = weights.masked_fill(~valid, 0.)
    def reduce(name):
        if name not in captured:
            raise RuntimeError(f"Missing baseline priority component: {name}")
        # Decoder diagnostics are [B,T,K,H,W]; concept assignments are [B,T,H,W,K].
        values = captured[name].permute(0, 1, 3, 4, 2)
        if values.shape != valid.shape:
            raise ValueError("Captured baseline priority shape does not match concept assignments.")
        if not torch.isfinite(values[valid]).all():
            raise FloatingPointError("Nonfinite valid baseline priority score.")
        values = values.masked_fill(~valid, 0.)
        return (values * weights).sum(-1) / weights.sum(-1).clamp_min(1e-12)
    unary = reduce('unary_scores')
    if block.priority_mode == 'pairwise':
        context = float(block.pairwise_strength) * reduce('pairwise_context')
    elif block.priority_mode == 'unary_mask':
        context = torch.zeros_like(unary)
    else:
        raise ValueError(f"Unsupported priority_mode: {block.priority_mode}")
    # Reduce in float32, then add the actual context contribution. Reconstructing
    # this sum avoids AMP rounding of the native per-slot sum changing ranks.
    total = unary + context
    return {'total': total.detach(), 'unary': unary.detach(), 'context': context.detach()}


def capture_baseline_patch_priorities(model, cache, concept_outs, baseline, enabled, dtype):
    """Replay the INTACT decoder; scores are cached once, before any deletions."""
    sink = {}
    with ExitStack() as stack:
        for stage in TARGET_STAGES:
            stack.enter_context(LocalSlotDeletionAdapter(model, stage, priority_sink=sink))
        replay = cached_prediction(model, cache, concept_outs, enabled, dtype)
    torch.testing.assert_close(replay['saliency_logits'], baseline['saliency_logits'], rtol=1e-5, atol=1e-6)
    error = float((replay['saliency_logits'].float()-baseline['saliency_logits'].float()).abs().max())
    if set(sink) != set(TARGET_STAGES):
        raise RuntimeError("Could not capture original priorities at every selected stage.")
    scores = {stage: baseline_patch_priorities(concept_outs[stage], sink[stage],
              model.saliency_prediction.fusion_blocks[stage]) for stage in TARGET_STAGES}
    return scores, error


def deletion_patch_mask(original, percentage, names, dataset_indices, stage, repeat, patch_scores):
    """Select whole baseline patches, then expand selection to ALL their valid slots."""
    valid = original['visual_validity_mask'].bool()
    patch_valid = valid.any(-1)
    if patch_scores is None or patch_scores.shape != patch_valid.shape:
        raise ValueError("Whole-patch deletion requires matching ORIGINAL baseline patch priorities.")
    if len(names) != valid.shape[0] or len(dataset_indices) != valid.shape[0]:
        raise ValueError("Batch metadata must match patch-priority batch size.")
    selected = np.zeros(tuple(patch_valid.shape), dtype=bool)
    score_cpu = patch_scores.detach().float().cpu().numpy()
    valid_cpu = patch_valid.cpu().numpy()
    for i, (name, index) in enumerate(zip(names, dataset_indices)):
        seed = None
        if DELETION_ORDER == 'random_patches':
            key = f'{DELETION_SEED}|{name}|{index}|{stage}|{repeat}'.encode()
            seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        positions = select_patch_positions(score_cpu[i], valid_cpu[i], percentage, seed)
        selected[i].reshape(-1)[positions] = True
    patch_mask = torch.from_numpy(selected).to(valid.device)
    return patch_mask.unsqueeze(-1).expand_as(valid) & valid


def deletion_slot_mask(original, percentage, names, dataset_indices, stage, repeat, patch_scores=None):
    if DELETION_ORDER in ('highest_priority', 'random_patches'):
        return deletion_patch_mask(original, percentage, names, dataset_indices, stage, repeat, patch_scores)
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


def prediction_maps_on_gt_grid(prediction, density):
    """Resize once to the GT grid, then expose float64 CPU maps for diagnostics."""
    pred = prediction_for_metrics(prediction)
    if density.ndim != 4 or density.shape[1] != 1 or len(density) != len(pred):
        raise ValueError('Expected one [B,1,H,W] GT density map per prediction.')
    if pred.shape[-2:] != density.shape[-2:]:
        pred = F.interpolate(pred, size=density.shape[-2:], mode='bilinear', align_corners=False)
    return pred.detach().double().cpu().numpy()[:, 0]


def per_sample_regional_changes(prediction, baseline_maps, density, marked_regions):
    """Compare changed/intact maps on identical actual deletion footprints."""
    pred_np = prediction_maps_on_gt_grid(prediction, density)
    gt_np = density.detach().double().cpu().numpy()[:, 0]
    baseline_maps = np.asarray(baseline_maps, dtype=np.float64)
    marked_regions = np.asarray(marked_regions, dtype=bool)
    if marked_regions.shape != gt_np.shape or baseline_maps.shape != gt_np.shape:
        raise ValueError('Intact maps and deletion footprints must have shape [B,GT_H,GT_W].')
    return [regional_intervention_changes(p, b, g, m)
            for p, b, g, m in zip(pred_np, baseline_maps, gt_np, marked_regions)]


def summarize_metrics(rows):
    """Average repeats per window, then report equal-window/equal-video means."""
    by_window=defaultdict(dict)
    for row in rows:
        key=(row['video'],row['dataset_index'])
        by_window[key][row['condition']]=row
    videos=[]
    dose_records={p:[] for p in (0.,*DELETION_PERCENTAGES)}
    for (video,index),conditions in sorted(by_window.items()):
        for p in dose_records:
            dose=[conditions['baseline']] if p==0 else [r for r in conditions.values() if r['percentage']==p]
            if not dose:raise RuntimeError('Incomplete paired percentage evaluation.')
            record={'video':video,'dataset_index':index}
            for metric in REPORT_NAMES:
                vals=[r[metric] for r in dose]
                # Require all repeats to be defined so a window's repeat budget
                # cannot silently change for an undefined region or metric.
                record[metric]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            dose_records[p].append(record)
    summaries={}
    for p,records in dose_records.items():
        summary={'window_count':len(records),'window_mean':{},'video_mean':{},'valid_window_counts':{}}
        groups=defaultdict(list)
        for r in records:groups[r['video']].append(r)
        for metric in REPORT_NAMES:
            values=[r[metric] for r in records if r[metric] is not None]
            summary['window_mean'][metric]=float(np.mean(values)) if values else None
            summary.setdefault('valid_window_counts',{})[metric]=len(values)
            video_values=[]
            for video,items in sorted(groups.items()):
                finite=[r[metric] for r in items if r[metric] is not None]
                if finite:video_values.append(float(np.mean(finite)))
            summary['video_mean'][metric]=float(np.mean(video_values)) if video_values else None
        for video,items in sorted(groups.items()):
            record={'video':video,'percentage':p,'window_count':len(items)}
            for metric in REPORT_NAMES:
                finite=[r[metric] for r in items if r[metric] is not None]
                record[metric]=float(np.mean(finite)) if finite else None
                record[f'valid_windows_{metric}']=len(finite)
            videos.append(record)
        summaries[f'{p:g}']=summary
    return summaries,videos


def write_csv(path,rows):
    if not rows:path.write_text('');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def display_and_save_summary(output,summaries):
    def fmt(x):return 'N/A' if x is None else f'{x:.6f}'
    print('\nRegions Removed (%) |       CC |      NSS |      SIM | '
          'ΔBias removed | ΔMAE removed | ΔBias retained | ΔMAE retained',flush=True)
    table=[]
    for p,summary in summaries.items():
        row={'percentage':float(p),**summary['window_mean']}
        row.update({f'valid_windows_{m}':summary['valid_window_counts'][m] for m in REPORT_NAMES})
        table.append(row)
        regular=' | '.join(f'{fmt(row[k]):>8}' for k in METRIC_NAMES)
        changes=' | '.join('N/A'.rjust(14) if row[k] is None else f'{row[k]:14.6e}'
                           for k in ('delta_bias_removed','delta_mae_removed',
                                     'delta_bias_retained','delta_mae_retained'))
        print(f'{float(p):18g}% | {regular} | {changes}',flush=True)
    print('Marked footprint = union across stages; actual image coverage is saved as '
          'marked_regions_percent. Regional effects use the intact baseline on each condition\'s '
          'own mask; values are unscaled. Positive ΔBias = decreased saliency; '
          'positive ΔMAE = worse agreement with GT.',flush=True)
    write_csv(output/'summary.csv',table)


def evaluate():
    import train as train_cfg
    importlib.import_module('PIL.Image')  # fail before model evaluation if PNG support is missing
    output=Path(OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    saliency_sampler=SaliencyMapSampler(seed=SAL_MAP_SEED,count=5)
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
                for r in range(1 if p==100 or DELETION_ORDER not in ('random','random_patches') else DELETION_REPEATS)]
    seen=set();rows=[];replay_error=None
    stat_totals={name:{s:defaultdict(float) for s in TARGET_STAGES} for _,_,name in conditions}
    stage_audit={}
    fields=['video','dataset_index','loader_batch','window_length','condition','percentage','repeat',
            *REPORT_NAMES,'fixation_pixels','deletion_details']
    unit='patches' if DELETION_ORDER in ('highest_priority','random_patches') else 'slots'
    print(f'Checkpoint: {CHECKPOINT_PATH}\nStages: {TARGET_STAGES}; deletion percentages ({unit}): {DELETION_PERCENTAGES}; '
          f'mode={MODEL_ASSIGNMENT_MODE}, top_k={MODEL_TOP_K}; metric input={METRIC_INPUT}',flush=True)
    with (output/'window_results.csv').open('w',newline='') as handle, (output/'patch_selection_audit.jsonl').open('w') as audit_writer:
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
            patch_priorities, batch_replay_error = capture_baseline_patch_priorities(
                model, cache, concept_outs, baseline, enabled, dtype)
            replay_error = max(replay_error or 0., batch_replay_error)
            # JSONL records the intact rank/score components once per stage and window.
            for stage in TARGET_STAGES:
                patch_valid = concept_outs[stage]['visual_validity_mask'].any(-1)
                components = patch_priorities[stage]
                for i, (video, index) in enumerate(zip(names, indices)):
                    scores_np = components['total'][i].cpu().numpy()
                    order = rank_patch_positions(scores_np, patch_valid[i].cpu().numpy())
                    audit_writer.write(json.dumps({
                        'video': video, 'dataset_index': index, 'stage': stage,
                        'patch_shape': list(patch_valid.shape[1:]),
                        'priority_mode': model.saliency_prediction.fusion_blocks[stage].priority_mode,
                        'context_multiplier': float(model.saliency_prediction.fusion_blocks[stage].pairwise_strength)
                            if model.saliency_prediction.fusion_blocks[stage].priority_mode == 'pairwise' else 0.,
                        'descending_patch_raster_indices': order.tolist(),
                        **{key: val[i].reshape(-1)[torch.as_tensor(order, device=val.device)].cpu().tolist()
                           for key, val in components.items()}
                    }, allow_nan=False) + '\n')
            audit_writer.flush()
            with torch.inference_mode():
                base_scores=per_sample_metrics(baseline,density,fixations,functions)
                baseline_maps=prediction_maps_on_gt_grid(baseline,density)
                empty_marked=np.zeros((len(names),*density.shape[-2:]),dtype=bool)
                base_regional=per_sample_regional_changes(baseline,baseline_maps,density,empty_marked)
                for score,regional in zip(base_scores,base_regional):score.update(regional)
            sampled_records=saliency_sampler.consider_batch(
                names,indices,batch_index,length,baseline_maps,
                density.detach().double().cpu().numpy()[:,0],
                last_window_rgb_uint8(rgb))
            def append(condition,percentage,repeat,scores,details):
                for i,(video,index) in enumerate(zip(names,indices)):
                    row={'video':video,'dataset_index':index,'loader_batch':batch_index,'window_length':length,
                         'condition':condition,'percentage':percentage,'repeat':repeat,**scores[i],
                         'deletion_details':json.dumps(details[i],sort_keys=True)}
                    writer.writerow(row);rows.append(row)
                handle.flush()
            append('baseline',0.,-1,base_scores,[{} for _ in names]);seen.update(names)
            for percentage,repeat,name in conditions:
                changed_concepts=dict(concept_outs);details=[{} for _ in names]
                marked_regions=np.zeros((len(names),*density.shape[-2:]),dtype=bool)
                with inference_context(model,enabled,dtype):
                    for stage in TARGET_STAGES:
                        original=concept_outs[stage]
                        mask=deletion_slot_mask(original,percentage,names,indices,stage,repeat,patch_priorities[stage]['total'])
                        marked_regions |= project_marked_patches(
                            mask.any(-1).detach().cpu().numpy(),density.shape[-2:])
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
                            item.update({'original_slots':int(original_count[i]),'original_assignment_weight':float(original_weight[i]),
                                         'original_patches':int(original['visual_validity_mask'][i].any(-1).sum())})
                            item['actual_removed_patch_percent']=100*item['deleted_patches']/item['original_patches']
                            item['actual_removed_slot_percent']=100*item['removed_slots']/item['original_slots']
                            item['actual_removed_weight_percent']=100*item['removed_assignment_weight']/item['original_assignment_weight']
                            details[i][stage]=item
                            for k,v in item.items():
                                if not k.endswith('_percent'):stat_totals[name][stage][k]+=v
                    with ExitStack() as stack:
                        for adapter in adapters.values():stack.enter_context(adapter)
                        changed=cached_prediction(model,cache,changed_concepts,enabled,dtype)
                    scores=per_sample_metrics(changed,density,fixations,functions)
                    regional_scores=per_sample_regional_changes(changed,baseline_maps,density,marked_regions)
                    for score,regional in zip(scores,regional_scores):score.update(regional)
                    if sampled_records and repeat==0:
                        changed_maps=prediction_maps_on_gt_grid(changed,density)
                        for i,record in sampled_records.items():
                            record['intervened'][float(percentage)]=changed_maps[i].copy()
                        del changed_maps
                append(name,percentage,repeat,scores,details)
                del changed,changed_concepts
            # print(f'Batch {batch_index}: paired baseline + {len(conditions)} deletions; windows={len(rows)//(1+len(conditions))}',flush=True)
            del baseline,cache,concept_outs,patch_priorities,baseline_maps
    if not rows:raise RuntimeError('No valid labeled validation windows were evaluated.')
    if allowed is not None and MAX_WINDOWS is None and seen!=allowed:
        raise ValueError(f'Requested evaluation videos missing valid windows: {sorted(allowed-seen)}')
    summaries,video_rows=summarize_metrics(rows)
    write_csv(output/'video_results.csv',video_rows)
    display_and_save_summary(output,summaries)
    sal_maps_root=Path(SAL_MAPS_DIR) if SAL_MAPS_DIR is not None else Path(__file__).resolve().parent/'sal_maps'
    saliency_manifest=saliency_sampler.save(sal_maps_root,DELETION_PERCENTAGES)
    metric_provenance={}
    for key,fn in functions.items():
        source=inspect.getsourcefile(fn)
        metric_provenance[key]={'callable':f'{fn.__module__}.{fn.__name__}',
                               'source_path':source,
                               'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest() if source and Path(source).is_file() else None}
    result={'status':'complete' if MAX_WINDOWS is None else 'complete_pilot_subset',
            'checkpoint':CHECKPOINT_PATH,'checkpoint_epoch':checkpoint.get('epoch'),'checkpoint_weights_modified':False,
            'intervention':('whole_patch_percentage_deletion_post_assignment_no_replacement'
                if DELETION_ORDER in ('highest_priority','random_patches') else 'patch_local_percentage_slot_deletion_post_assignment_no_replacement'),
            'percentage_definition':('fraction of baseline valid PATCHES per window at EACH selected stage; all valid slots removed; floor(N*p/100+.5)'
                if DELETION_ORDER in ('highest_priority','random_patches') else 'fraction of baseline VALID prototype slots per window at EACH selected stage; floor(N*p/100+.5)'),
            'priority_ranking':{
                'scope':'per_window_per_stage; all selected stages deleted simultaneously',
                'slot_score':'unary + context_priority; context_priority = pairwise_strength * pairwise_context',
                'patch_reduction':'original-assignment-weighted mean across valid prototype slots',
                'unary_only_stages':'context_priority is zero',
                'score_space':'raw learned total; before temperature, log activations, softmax or gate normalization',
                'source':'intact baseline decoder replay; frozen across all deletion doses',
                'ties':'ascending patch raster index',
                'selection_audit':'patch_selection_audit.jsonl'},
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
            'regional_intervention_effects':{
                'delta_bias':'mean_R(P_intact - P_deleted); equivalent to change in signed GT-Pred bias',
                'delta_mae':'mean_R(abs(GT - P_deleted) - abs(GT - P_intact))',
                'comparison':'intact and intervened predictions evaluated on identical removed/retained masks for each condition',
                'bias_sign':'positive = decreased predicted saliency; negative = increased predicted saliency',
                'mae_sign':'positive = increased error relative to GT; negative = reduced error',
                'scale':1,'units':'unscaled unit-mass density per pixel',
                'prediction_input':METRIC_INPUT,
                'prediction_resize':'bilinear to GT spatial grid; align_corners=False',
                'normalization':'each whole map independently normalized to unit mass; never normalize inside/outside separately',
                'negative_prediction_policy':'subtract minimum only when negative, then normalize',
                'zero_mass_prediction_policy':'uniform density',
                'zero_mass_gt_policy':'all four regional effects undefined',
                'marked_region':'union of selected patch footprints across intervened stages, nearest-neighbor projection to GT grid; any deleted slot marks a patch in legacy slot modes',
                'coverage':'percentage is requested per-stage patch/slot dose; marked_regions_percent is actual union coverage on GT grid',
                'empty_region_policy':'null in JSON, blank in CSV, N/A in console',
                'full_frame_signed_change':'zero by unit-mass construction; full-frame MAE change remains informative'},
            'window_stride':WINDOW_STRIDE,'window_len':window_len,'dataset_windows':len(dataset),
            'evaluated_windows':len(rows)//(1+len(conditions)),'evaluation_videos':sorted(seen),
            'windows_without_fixations':sum(r['fixation_pixels']==0 for r in rows if r['condition']=='baseline'),
            'aggregation':'repeat means per window (all repeats must be defined), then equal windows; equal-video results also provided; undefined values excluded with counts',
            'deletion_stats':{n:{s:dict(v) for s,v in stages.items()} for n,stages in stat_totals.items()},
            'summary':summaries,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'saliency_map_samples':saliency_manifest,
            'evaluated_at':datetime.now(timezone.utc).isoformat(timespec='seconds')}
    result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Results: {result_path}',flush=True)
    return result


def configure_cli():
    global CHECKPOINT_PATH,VAL_DATASET_DIR,GPU_ID,OUTPUT_DIR,TARGET_STAGES,DELETION_PERCENTAGES
    global DELETION_ORDER,DELETION_REPEATS,DELETION_SEED,WINDOW_STRIDE,MAX_WINDOWS
    global EVALUATION_VIDEO_IDS_PATH,MODEL_ASSIGNMENT_MODE,MODEL_TOP_K,METRIC_INPUT
    global BATCH_SIZE,NUM_WORKERS,WINDOW_LEN,USE_TRAIN_AMP
    global SAL_MAPS_DIR,SAL_MAP_SEED
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint',default=CHECKPOINT_PATH)
    parser.add_argument('--val-dir',default=VAL_DATASET_DIR)
    parser.add_argument('--output-dir',default=OUTPUT_DIR)
    parser.add_argument('--gpu-id',type=int,default=GPU_ID)
    parser.add_argument('--percentages',type=float,nargs='+',default=list(DELETION_PERCENTAGES))
    parser.add_argument('--stages',nargs='+',choices=['stage1','stage2','stage3','stage4'],default=list(TARGET_STAGES))
    parser.add_argument('--order',choices=['highest_priority','random_patches','random','highest_weight','highest_usage'],default=DELETION_ORDER)
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
    parser.add_argument('--sal-maps-dir',type=Path,default=SAL_MAPS_DIR,
                        help='PNG root; default is sal_maps beside this .py file.')
    parser.add_argument('--sal-map-seed',type=int,default=SAL_MAP_SEED,
                        help='Separate reproducible RNG seed for five sampled saliency-map windows.')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--selection-self-test',action='store_true',help='Test patch selection without project files or PyTorch.')
    parser.add_argument('--regional-self-test',action='store_true',help='Test matched-baseline regional bias/MAE changes without project files or PyTorch.')
    args=parser.parse_args()
    if any(not math.isfinite(p) or not 0<=p<=100 for p in args.percentages):parser.error('Percentages must lie in [0,100].')
    for field in ['repeats','stride','top_k']:
        if getattr(args,field)<1:parser.error(f'{field} must be positive.')
    for field in ['max_windows','batch_size','window_len']:
        value=getattr(args,field)
        if value is not None and value<1:parser.error(f'{field} must be positive.')
    if args.workers is not None and args.workers<0:parser.error('workers must be nonnegative.')
    if args.sal_map_seed<0:parser.error('sal-map-seed must be nonnegative.')
    CHECKPOINT_PATH,VAL_DATASET_DIR,GPU_ID,OUTPUT_DIR=args.checkpoint,args.val_dir,args.gpu_id,args.output_dir
    TARGET_STAGES=tuple(dict.fromkeys(args.stages))
    DELETION_PERCENTAGES=tuple(sorted(set(p for p in args.percentages if p>0)))
    if not DELETION_PERCENTAGES:
        parser.error('At least one strictly positive deletion percentage is required.')
    DELETION_ORDER,DELETION_REPEATS,DELETION_SEED=args.order,args.repeats,args.deletion_seed
    WINDOW_STRIDE,MAX_WINDOWS,EVALUATION_VIDEO_IDS_PATH=args.stride,args.max_windows,args.video_ids
    MODEL_ASSIGNMENT_MODE,MODEL_TOP_K,METRIC_INPUT=args.assignment_mode,args.top_k,args.metric_input
    BATCH_SIZE,NUM_WORKERS,WINDOW_LEN=args.batch_size,args.workers,args.window_len
    USE_TRAIN_AMP=not args.no_amp
    SAL_MAPS_DIR,SAL_MAP_SEED=args.sal_maps_dir,args.sal_map_seed
    return args


def self_test():
    """Tests real reviewed concept/decoder code, without checkpoint or dataset."""
    from model.concept_creation import VisualConceptCreation
    from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder
    selection_self_test()
    regional_self_test()
    global TARGET_STAGES
    TARGET_STAGES = ('stage4','stage3','stage2','stage1')
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
                patch_priorities, _ = capture_baseline_patch_priorities(
                    model, cache, baseline['concept_out'], baseline, False, torch.bfloat16)
                adapters={s:LocalSlotDeletionAdapter(model,s) for s in model.concept_creations}
                changed=dict(baseline['concept_out'])
                for s,original in baseline['concept_out'].items():
                    validate_baseline(model.concept_creations[s],original)
                    m25=deletion_slot_mask(original,25.,['a','b'],[3,9],s,0,patch_priorities[s]['total'])
                    m50=deletion_slot_mask(original,50.,['a','b'],[3,9],s,0,patch_priorities[s]['total'])
                    assert not (m25 & ~m50).any()
                    assert torch.equal(m25,deletion_slot_mask(original,25.,['a','b'],[3,9],s,0,patch_priorities[s]['total']))
                    valid=original['visual_validity_mask']
                    n=valid.sum((1,2,3,4))
                    if DELETION_ORDER in ('highest_priority','random_patches'):
                        patch_count = valid.any(-1).sum((1,2,3))
                        assert torch.equal(m50.any(-1).sum((1,2,3)),(patch_count*.5+.5).floor().long())
                        assert torch.equal(m50, m50.any(-1).unsqueeze(-1).expand_as(valid) & valid)
                    else:
                        assert torch.equal(m50.sum((1,2,3,4)),(n*.5+.5).floor().long())
                    assert not (m50 & ~valid).any()
                    # The same video/window mask is unchanged when processed alone.
                    per_sample={'active_visual_prototype_indices':original['active_visual_prototype_indices'][1:2],
                                'visual_validity_mask':valid[1:2],
                                'visual_activations':original['visual_activations'].reshape(2,-1,16)[1].reshape(-1,16)}
                    assert torch.equal(m50[1:2],deletion_slot_mask(per_sample,50.,['b'],[9],s,0,patch_priorities[s]['total'][1:2]))
                    m100=deletion_slot_mask(original,100.,['a','b'],[3,9],s,0,patch_priorities[s]['total'])
                    assert torch.equal(m100,valid)
                    changed[s],stats=delete_local_slots(model.concept_creations[s],original,m100)
                    assert not changed[s]['visual_activations'].count_nonzero()
                    assert not changed[s]['visual_validity_mask'].count_nonzero()
                    assert torch.equal(changed[s]['active_visual_prototype_indices'],original['active_visual_prototype_indices'])
                    torch.testing.assert_close(stats['removed_assignment_weight'],original['visual_activations'].reshape(2,-1).double().sum(1))
                    assert not deletion_slot_mask(original,0.,['a','b'],[3,9],s,0,patch_priorities[s]['total']).any()
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
    # Aggregation averages random repeats BEFORE giving videos equal weight.
    test_rows=[]
    for v in ['a','b']:
        for condition,percentage,value in [('baseline',0.,1.),('delete_50_repeat_000',50.,.6),('delete_50_repeat_001',50.,.8)]:
            test_rows.append({'video':v,'dataset_index':0,'condition':condition,'percentage':percentage,
                              **{key:value for key in REPORT_NAMES}})
    previous=globals()['DELETION_PERCENTAGES'];globals()['DELETION_PERCENTAGES']=(50.,)
    try:
        summary,_=summarize_metrics(test_rows)
        assert np.isclose(summary['50']['window_mean']['CC'],.7)
        assert np.isclose(summary['50']['video_mean']['CC'],.7)
        assert np.isclose(summary['50']['window_mean']['delta_bias_removed'],.7)
        assert 'paired_comparison' not in summary['50']
    finally:globals()['DELETION_PERCENTAGES']=previous
    print('PASS: hard top-1/top-8, exact nested slot counts, batch-invariant random masks, surviving weights/indices, cached baseline replay, all-stage empty evidence, AMP, residual policies, unchanged weights/restoration, real-frame labels, project metrics, undefined CC, matched-baseline regional effects and repeat/video aggregation.')


if __name__=='__main__':
    args=configure_cli()
    if args.self_test:self_test()
    else:evaluate()
