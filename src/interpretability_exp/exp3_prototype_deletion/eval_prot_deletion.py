"""Paired local HIGH prototype deletion with selection-matched controls.

Put this file beside train.py or under src/interpretability_exp/exp2d_prototype_deletion/.
For every stage N, HIGH targets are the HIGH_TARGET_COUNT highest-E(G) IDs of
stage N's own bank among IDs observed in the high band, including E(G)<=1
(high_prototypes.json, per-stage scores). Video lists come from
finalized_targets.json. All four stages are deleted jointly, each with its own
IDs. After selection-only profiling, each stage keeps the largest E(G)-ranked
prefix that leaves equal-count distinct used non-target controls; the actual
counts are recorded. Actual removal on evaluation windows is recorded.
Set SELECTION_EXP_DIR (or PROTO_SELECTION_DIR) to the regenerated top-k selection.
MODEL_TOP_K=8 and MODEL_ASSIGNMENT_MODE="topk_soft" configure the modified model.
Run: python eval_prot_deletion.py
Optional local/model tests (no checkpoint/dataset; project model imports required):
    python eval_prot_deletion.py --self-test
Selection-only support diagnostic:
    python eval_prot_deletion.py --selection-diagnostic 5

Intervention: compute original assignments/top-k, remove selected evidence with
NO replacement, rebuild FiLM representation, recompute priorities/context and
mixtures, then rerun main/side prediction. Original surviving sparse top-k assignment weights
are retained; downstream normalization is recomputed. This is an intervention on
the original evidence, NOT bank exclusion before assignment/reassignment.

Generic backbone features and the existing patch-embedding residual are retained.
An empty patch has zero prototype priority/mixture. Existing decoder residual
coefficients are retained (not replaced with an identity gate).
No parameters, core source files, or checkpoint files are changed.

Measurement: replace saliency-value effects with CC, NSS and SIM against GT.
Overall scores use the project metric functions and METRIC_INPUT='model_map',
matching the reviewed training metric call. Top-5% masks are selected ONCE from
baseline sigmoid patch means at each stage, expanded into pixel ROIs, and frozen
for baseline/HIGH/control outputs. Stage rows measure the same joint all-stage
intervention at different grids, not independent stage ablations.

Regional scores are CONDITIONAL within the selected ROI: each metric normalizes
inside the ROI. Outside pixels are excluded, never zero-padded into the score.
NSS uses true fixation labels and is undefined if none occur within the ROI.
Constant/degenerate CC is undefined. Undefined values and paired exclusions are
reported; no pseudo-fixations or fabricated scores are used. Metric delta is
DELETED minus BASELINE; negative CC/NSS/SIM delta means worse performance.

Results: metric_summary.csv, frame_results.csv, video_results.csv, and
local_deletion_results.json under LOCAL_OUTPUT_DIR. Console scores give paired
equal-window means; JSON also gives equal-video means and paired-bootstrap CIs
for HIGH versus baseline and HIGH versus sampled controls.
End-of-run tables and confidence_intervals.csv report HIGH versus baseline,
the mean control, and each control, using equal-window effects with a paired
whole-video cluster bootstrap. Existing equal-video intervals remain in JSON.
deletion_evidence_summary.csv reports configured deleted ID counts, actually
removed valid slots and original sparse assignment mass for each condition/stage,
plus fractions of the stage's original slots/mass on the evaluation windows.

"""
from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import importlib
import math
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Works beside train.py or under src/interpretability_exp/exp2d_prototype_deletion/.
_SRC_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "train.py").is_file()), None)
if _SRC_ROOT is not None:
    sys.path.insert(0, str(_SRC_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# EDIT THESE CONSTANTS. No argparse is required for evaluation.
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/"
    "training_outputs/saved_weights/best_dhf1k.pth"
)
VAL_DATASET_DIR = "/data/quantization/zaima/videosal_datasets/dhf1k/val"
OUTPUT_DIR = "evaluation_outputs/high_prototype_deletion"
GPU_ID = 1
SAVE_VAL_MAPS = False
RUN_BASELINE_FIRST = False  # Optional extra full validation pass for metric deltas.

# None uses the SAME settings as train.py / the supplied baseline evaluation.py.
WINDOW_LEN = None
BATCH_SIZE = None
NUM_WORKERS = None
SEED = None

# Point this to the regenerated selection for the modified top-k model.
# The attached script's existing path is preserved; it currently names top1.
SELECTION_EXP_DIR = os.environ.get(
    "PROTO_SELECTION_DIR",
    "/data/quantization/zaima/videosal_datasets/dhf1k/deletion_exp",
)
FINALIZED_TARGETS_PATH = f"{SELECTION_EXP_DIR}/finalized_targets.json"
HIGH_PROTOTYPES_PATH = f"{SELECTION_EXP_DIR}/high_prototypes.json"
HIGH_TARGET_COUNT = 50
FINALIZED_STAGE_TO_MODEL_STAGE = {
    1: "stage1", 2: "stage2", 3: "stage3", 4: "stage4",
    "1": "stage1", "2": "stage2", "3": "stage3", "4": "stage4",
    "stage1": "stage1", "stage2": "stage2", "stage3": "stage3", "stage4": "stage4",
}
MODEL_TOP_K = 8
MODEL_ASSIGNMENT_MODE = "topk_soft"
DELETION_STAGES = ("stage1", "stage2", "stage3", "stage4")


def high_scores(stage):
    scores = json.loads(Path(HIGH_PROTOTYPES_PATH).read_text(encoding="utf-8"))
    stages = scores.get("stages")
    if not isinstance(stages, dict) or stage not in stages:
        raise ValueError(f"{HIGH_PROTOTYPES_PATH}: missing per-stage scores for {stage}; "
                         "rerun select_target_prototypes.py with independent stage ranking.")
    return stages[stage]


def ranked_by_eg(scores, ids):
    return sorted(ids, key=lambda j: (-scores[str(j)]["E(G)"], -scores[str(j)]["high_patch_count"], int(j)))


def load_high_targets():
    """Per stage: top HIGH_TARGET_COUNT IDs of that stage's own bank by E(G), among IDs observed in the high band."""
    targets = {}
    for stage in DELETION_STAGES:
        scores = high_scores(stage)
        # Unobserved IDs all share the smoothed-prior E(G); they carry no evidence.
        observed = [int(k) for k, v in scores.items() if v["high_patch_count"] > 0]
        if not observed:
            raise ValueError(f"{HIGH_PROTOTYPES_PATH}: no {stage} prototypes observed in the high band.")
        targets[stage] = ranked_by_eg(scores, observed)[:HIGH_TARGET_COUNT]
    return _normalized_targets(targets)


# AST fingerprints of the inference methods in the three supplied model files.
# Whitespace/comments/file renaming do not affect these checks. If your inference
# code differs, adapt this script to that version rather than bypassing the check.
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


def _validate_model_configuration(model):
    for stage, concept in model.concept_creations.items():
        if concept.visual_assignment_mode != MODEL_ASSIGNMENT_MODE or concept.top_k != MODEL_TOP_K:
            raise RuntimeError(f"{stage}: expected {MODEL_ASSIGNMENT_MODE}, top_k={MODEL_TOP_K}; "
                               f"got {concept.visual_assignment_mode}, top_k={concept.top_k}.")
        if not model.saliency_prediction.fusion_blocks[stage].use_shared_concept_activations:
            raise RuntimeError(f"{stage}: shared concept activations are required for frozen-evidence deletion.")


def _normalized_targets(targets: Dict[str, Sequence[int]]) -> Dict[str, list[int]]:
    unknown = set(targets) - {"stage1", "stage2", "stage3", "stage4"}
    if unknown:
        raise ValueError(f"Unknown target stages: {sorted(unknown)}")
    result = {}
    for stage, ids in targets.items():
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError(f"{stage}: prototype IDs must be nonnegative integers.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{stage}: duplicate prototype IDs; fix the editable list.")
        result[stage] = sorted(ids)
    return result


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


class PrototypeEvidenceDeletionAdapter:
    """Reversible, inference-only instance overrides; state_dict is unchanged."""

    def __init__(self, model: torch.nn.Module, targets: Dict[str, Sequence[int]]):
        self.model = model
        self.targets = _normalized_targets(targets)
        self._restores: list[Tuple[Any, str, bool, Any]] = []
        self._installed = False
        self.stats = {}
        _check_decoder_functions()
        for stage, ids in self.targets.items():
            if not ids:
                continue
            if stage not in model.concept_creations or stage not in model.saliency_prediction.fusion_blocks:
                raise ValueError(f"Target stage {stage} is not present in this model.")
            concept = model.concept_creations[stage]
            block = model.saliency_prediction.fusion_blocks[stage]
            size = int(concept.visual_concepts.shape[0])
            if ids[-1] >= size:
                raise ValueError(f"{stage}: ID {ids[-1]} exceeds bank range 0..{size - 1}.")
            if not block.use_shared_concept_activations:
                raise RuntimeError(
                    "This no-replacement adapter requires use_shared_concept_activations=True, "
                    "as in your supplied baseline evaluation. Independent decoder "
                    "reassignment is not supported by this experiment."
                )
            for name in ("forward", "_build_visual_concepts_from_patches",
                         "_compute_visual_assignments", "_extract_active_prototypes"):
                _check_method(concept, "VisualConceptCreation", name)
            for name in ("forward", "_resolve_visual_activations",
                         "_prioritize_concept_patch_activations", "_factorized_pairwise_context",
                         "_build_concept_conditioned_update"):
                _check_method(block, "SpatioTemporalConceptGatedFusionBlock", name)
            self.stats[stage] = {
                "prototype_ids": ids, "forward_calls": 0, "patches": 0,
                "patches_with_target_assignment": 0, "removed_valid_slots": 0,
                "patches_empty_after_deletion": 0, "removed_assignment_mass": 0.0,
                "original_valid_slots": 0, "original_assignment_mass": 0.0,
                "removed_mass_per_prototype": {str(i): 0.0 for i in ids},
                "valid_slots_per_patch_before_min": None,
                "valid_slots_per_patch_before_max": None,
            }
        _check_method(model.saliency_prediction, "ConceptGatedMultiScaleSaliencyDecoder", "forward")

    def _override(self, obj: Any, name: str, method: Any) -> None:
        # Restore whether the attribute was inherited or previously instance-bound.
        own = name in obj.__dict__
        self._restores.append((obj, name, own, obj.__dict__.get(name)))
        setattr(obj, name, MethodType(method, obj))

    def _delete_output(self, stage: str, concept: Any, original: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(original)
        ids = torch.tensor(self.targets[stage], device=out["visual_activations"].device)
        original_a = out["visual_activations"]
        removed_a = original_a.index_select(1, ids)
        a = original_a.clone()
        a[:, ids] = 0
        out["visual_activations"] = a

        # Sparse original top-k evidence feeds FiLM. No replacement or weight renormalization here.
        bank = F.normalize(concept.visual_concepts, dim=-1)
        q = out["visual_patch_embeddings"]
        representation = a @ bank
        if concept.visual_concept_residual_weight > 0:
            representation = representation + concept.visual_concept_residual_weight * q
        # FP16 cannot represent F.normalize's default 1e-12 epsilon. Normalize
        # a safe placeholder only for exactly empty rows, then return exact zeros.
        # Nonempty rows retain the model's original arithmetic and dtype.
        zero_rows = ~representation.ne(0).any(dim=-1, keepdim=True)
        safe_representation = torch.where(zero_rows, torch.ones_like(representation), representation)
        out["visual_concept_representation"] = F.normalize(
            safe_representation, dim=-1,
        ).masked_fill(zero_rows, 0)
        agreement = F.cosine_similarity(
            q, out["visual_concept_representation"], dim=-1,
        )
        out["visual_feature_concept_agreement"] = agreement.masked_fill(zero_rows.squeeze(-1), 0)

        indices = out["active_visual_prototype_indices"]  # Keep original top-k slots.
        deleted_slots = torch.zeros_like(indices, dtype=torch.bool)
        for prototype_id in self.targets[stage]:
            deleted_slots |= indices == prototype_id
        valid_before = out["visual_validity_mask"].bool()
        valid_after = valid_before & ~deleted_slots
        out["visual_validity_mask"] = valid_after
        out["active_visual_prototypes"] = out["active_visual_prototypes"].masked_fill(
            deleted_slots.unsqueeze(-1), 0,
        )
        # Indices deliberately retain original IDs for auditing; validity decides use.
        probs = out["visual_assignment_probs"].clone()
        probs[:, ids] = 0  # Retain surviving baseline weights; do not renormalize here.
        out["visual_assignment_probs"] = probs
        logits = out["visual_concept_logits"].clone()
        logits[:, ids] = -torch.inf  # Diagnostic only; assignments are already frozen.
        out["visual_concept_logits"] = logits
        winner = out["visual_concept_indices"].clone()
        for prototype_id in self.targets[stage]:
            winner.masked_fill_(winner == prototype_id, -1)  # No replacement winner.
        out["visual_concept_indices"] = winner
        out["visual_assignment_usage"] = probs.mean(dim=0)
        out["visual_assignment_entropy"] = -(probs * (probs + 1e-8).log()).sum(-1).mean()
        out["losses"] = {}  # Inference ablation; no stale auxiliary concept losses.

        stats = self.stats[stage]
        stats["forward_calls"] += 1
        stats["patches"] += int(a.shape[0])
        stats["patches_with_target_assignment"] += int((removed_a.sum(-1) > 0).sum().item())
        stats["removed_valid_slots"] += int((valid_before & deleted_slots).sum().item())
        stats["patches_empty_after_deletion"] += int((~valid_after.any(-1)).sum().item())
        masses = removed_a.double().sum(0).tolist()
        stats["removed_assignment_mass"] += sum(masses)
        stats["original_valid_slots"] += int(valid_before.sum().item())
        stats["original_assignment_mass"] += float(original_a.double().sum().item())
        for prototype_id, mass in zip(self.targets[stage], masses):
            stats["removed_mass_per_prototype"][str(prototype_id)] += mass
        counts = valid_before.sum(-1)
        minimum, maximum = int(counts.min().item()), int(counts.max().item())
        for key, value, combine in (
            ("valid_slots_per_patch_before_min", minimum, min),
            ("valid_slots_per_patch_before_max", maximum, max),
        ):
            stats[key] = value if stats[key] is None else combine(stats[key], value)

        # Cheap first-call checks: deleted columns/slots cannot carry evidence.
        if stats["forward_calls"] == 1:
            if torch.count_nonzero(a[:, ids]).item() or (valid_after & deleted_slots).any().item():
                raise AssertionError("Target evidence was not fully masked.")
            if not torch.isfinite(out["visual_concept_representation"]).all().item():
                raise FloatingPointError("Nonfinite concept representation after deletion.")
        return out

    def install(self) -> "PrototypeEvidenceDeletionAdapter":
        if self._installed:
            raise RuntimeError("Adapter already installed.")
        if self.model.training:
            raise RuntimeError("Call model.eval() before installing this inference-only adapter.")
        try:
            for stage, ids in self.targets.items():
                if not ids:
                    continue
                concept = self.model.concept_creations[stage]
                block = self.model.saliency_prediction.fusion_blocks[stage]
                original_forward = concept.forward
                signature = inspect.signature(original_forward)

                def deleted_forward(module, *args, _stage=stage, _original=original_forward,
                                    _signature=signature, **kwargs):
                    if module.training:
                        raise RuntimeError("Prototype deletion is inference-only.")
                    bound = _signature.bind(*args, **kwargs)
                    bound.arguments["return_losses"] = False
                    original = _original(*bound.args, **bound.kwargs)
                    return self._delete_output(_stage, module, original)

                self._override(concept, "forward", deleted_forward)
                self._override(block, "_assert_required_visual_prototypes", _allow_empty_visual_prototypes)
                self._override(block, "_prioritize_concept_patch_activations", _prioritize_allowing_empty)
            self._installed = True
        except Exception:
            self.remove()
            raise
        return self

    def remove(self) -> None:
        for obj, name, had_own_attribute, previous in reversed(self._restores):
            if had_own_attribute:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self._restores.clear()
        self._installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *_exc):
        self.remove()


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


def build_val_loader():
    import train as train_cfg
    from pre_process.collate import video_saliency_collate_fn
    from pre_process.dataloader import DatasetLoader

    window_len = train_cfg.WINDOW_LEN if WINDOW_LEN is None else WINDOW_LEN
    batch_size = train_cfg._dataloader_batch_size() if BATCH_SIZE is None else BATCH_SIZE
    workers = train_cfg.NUM_WORKERS if NUM_WORKERS is None else NUM_WORKERS
    dataset = DatasetLoader(VAL_DATASET_DIR, window_len=window_len, stride=32)
    kwargs = dict(num_workers=workers, collate_fn=video_saliency_collate_fn,
                  pin_memory=torch.cuda.is_available())
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs), dataset, window_len


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON.")



# ---------------------------------------------------------------------------
# LOCAL EXPERIMENT SETTINGS. Targets and video lists must share one selection run.
# The two JSON files contain lists of video identifiers matching loader filenames
# after Path(filename).stem. Use your existing discovery/confirmation split.
# Do not let videos used to select HIGH targets enter EVALUATION_VIDEO_IDS_PATH.
# ---------------------------------------------------------------------------
SELECTION_VIDEO_IDS_PATH = f'{SELECTION_EXP_DIR}/selection_video_ids.json'
EVALUATION_VIDEO_IDS_PATH = f'{SELECTION_EXP_DIR}/evaluation_video_ids.json'
# Example file contents: ["001", "002", "003"]. Use actual loader video IDs.
# Keep the target-selection videos in the selection list. The evaluation list
# must contain only videos not used to select targets or tune the protocol.
LOCAL_OUTPUT_DIR = 'evaluation_outputs/selected_id_deletion_metrics_all_stages'
SELECTION_DIAGNOSTIC_OUTPUT_DIR = 'evaluation_outputs/selection_support_diagnostic_topk8_top50eg'
CONTROL_REPEATS = 2
CONTROL_SEED = 42
CONTROL_NEAREST_POOL = 5  # randomize among closest remaining candidates
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 2026
USE_TRAIN_AMP = True
METRIC_INPUT = 'model_map'  # same as project training metrics; 'sigmoid' is explicit alternative
METRIC_NAMES = ('CC','NSS','SIM')


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


def subset_batches(loader, allowed, seen, desc="Selection/Evaluation"):
    for batch_index, batch in enumerate(tqdm(loader, desc=desc, unit="batch")):
        if len(batch) != 6:
            raise ValueError('Expected current six-item collate output: filenames, RGB, saliency, fixation, n_frames, valid_mask.')
        names, rgb, _, _, n_frames, _ = batch
        ids = [video_id(n) for n in names]
        selected = [i for i, v in enumerate(ids) if v in allowed]
        if not selected:
            continue
        # Evaluate each temporal length separately so the final slice is real,
        # never a padded frame. Keep the same window for baseline and deletion.
        for length in sorted({int(n_frames[i]) for i in selected}):
            if length <= 0:
                raise ValueError('Found an empty input window.')
            idx = [i for i in selected if int(n_frames[i]) == length]
            seen.update(ids[i] for i in idx)
            yield batch_index, [ids[i] for i in idx], rgb[idx, :length], length


def run_forward(model, rgb, train_cfg):
    device = model.output_device
    enabled = USE_TRAIN_AMP and train_cfg._amp_enabled(device)
    with torch.autocast(device.type, dtype=train_cfg._amp_dtype(device), enabled=enabled):
        return model(rgb, return_details=True, return_concept_losses=False,
                     return_decoder_diagnostics=False)


def patch_membership(concept, ids):
    indices = concept['active_visual_prototype_indices']
    valid = concept['visual_validity_mask'].bool()
    if indices.ndim != 5 or indices.shape[1] != 1:
        raise ValueError('This experiment requires last-frame-only [B,1,H,W,K] assignments.')
    wanted = torch.tensor(ids, device=indices.device, dtype=indices.dtype)
    return (torch.isin(indices, wanted) & valid).any(-1)[:, 0]


def pool_map(x, hw):
    # Current decoder sizes are integer multiples of every concept grid.
    # This gives nonoverlapping cells with exactly equal area, without smoothing.
    h, w = hw
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2] % h or x.shape[-1] % w:
        raise ValueError('Prediction dimensions must be divisible by the concept grid.')
    return F.avg_pool2d(x.float(), (x.shape[-2] // h, x.shape[-1] // w))[:, 0]


def measurement_saliency_map(prediction, output_activation):
    # Preserve model construction/forward. This evaluator requests raw outputs
    # with activation="none", so activate only the measurement copy. Never
    # apply a second sigmoid to a decoder that already returns sigmoid maps.
    if output_activation == 'none':
        logits = prediction['saliency_logits'].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite prediction logits.')
        return logits.sigmoid()
    if output_activation == 'sigmoid':
        return prediction['saliency_map'].float()
    raise ValueError(f'Unsupported measurement output activation: {output_activation!r}.')


def patch_maps(saliency_map, hw):
    if not torch.isfinite(saliency_map).all():
        raise FloatingPointError('Nonfinite prediction saliency map.')
    # Activation has been applied once per pixel, before patch averaging.
    return {'saliency': pool_map(saliency_map, hw)}


def baseline_high_region(maps):
    # Exactly ceil(5% of BASELINE grid cells), with stable raster-order ties.
    # Call once before deletion; reuse this mask for ALL conditions.
    scores = maps['saliency']
    high_band = torch.zeros_like(scores, dtype=torch.bool).flatten(1)
    count = max(1, int(np.ceil(high_band.shape[1] * .05)))
    rank = torch.argsort(scores.flatten(1), dim=1, descending=True, stable=True)
    high_band.scatter_(1, rank[:, :count], True)
    return high_band.view_as(scores)


def read_split():
    selection = read_video_ids(SELECTION_VIDEO_IDS_PATH)
    evaluation = read_video_ids(EVALUATION_VIDEO_IDS_PATH)
    if selection & evaluation:
        raise ValueError('Selection and evaluation video IDs must be disjoint.')
    finalized = json.loads(Path(FINALIZED_TARGETS_PATH).read_text(encoding='utf-8'))
    for name, ids in (('selection_videos', selection), ('evaluation_videos', evaluation)):
        values = finalized.get(name)
        if not isinstance(values, list) or {video_id(v) for v in values} != ids:
            raise ValueError(f'{name} in {FINALIZED_TARGETS_PATH} differs from the video ID file; '
                             'targets and split must come from the same selection run.')
    return selection, evaluation


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


def selection_support_report(model, targets, result, slots, seen, selected, partial, processed):
    report = {'checkpoint': CHECKPOINT_PATH, 'finalized_targets': FINALIZED_TARGETS_PATH,
              'target_selection':'independent per-stage E(G) ranking',
              'deletion_stages':list(DELETION_STAGES),
              'selection_videos_expected': len(selected), 'selection_videos_processed': len(seen),
              'processed_selection_batches': processed, 'partial_pass': partial,
              'model_config': {'top_k': MODEL_TOP_K, 'assignment_mode': MODEL_ASSIGNMENT_MODE},
              'stages': {}}
    print(f'\nSelection support | checkpoint={CHECKPOINT_PATH}')
    print(f'  selection videos processed: {len(seen)}/{len(selected)}'
          + (' (PARTIAL diagnostic pass)' if partial else ''))
    for stage, ids in targets.items():
        if not ids:
            continue
        concept = model.concept_creations[stage]
        s = slots[stage]
        mean_slots = s['sum'] / s['patches'] if s['patches'] else None
        zero = [int(j) for j in ids if result[stage][j, 0] <= 0]
        rows = {str(j): {'valid_patches': int(result[stage][j, 0]),
                         'activation_mass': float(result[stage][j, 1]),
                         'video_support': int(result[stage][j, 2])} for j in ids}
        report['stages'][stage] = {
            'assignment_mode': concept.visual_assignment_mode, 'model_training': concept.training,
            'configured_top_k': int(concept.top_k), 'patches': s['patches'],
            'valid_slots_per_patch': {'min': s['min'], 'mean': mean_slots, 'max': s['max'],
                                      'histogram': {str(k): v for k, v in sorted(s['hist'].items())}},
            'targets': rows, 'zero_support_ids': zero}
        shown_mean = 'n/a' if mean_slots is None else f'{mean_slots:.3f}'
        print(f'  {stage}: assignment_mode={concept.visual_assignment_mode} | '
              f'training={concept.training} | configured top_k={concept.top_k} | '
              f'valid slots/patch min/mean/max={s["min"]}/{shown_mean}/{s["max"]} '
              f'over {s["patches"]} patches')
        print(f'    {"ID":>5} {"valid patches":>14} {"activation mass":>16} {"videos":>7}')
        for j in ids:
            r = rows[str(j)]
            print(f'    {j:>5} {r["valid_patches"]:>14} {r["activation_mass"]:>16.6g} {r["video_support"]:>7}')
        print(f'    zero-support IDs: {zero if zero else "none"}')
    return report


@torch.no_grad()
def selection_profiles(model, loader, selected, targets, train_cfg, max_batches=None):
    if max_batches is not None and (isinstance(max_batches, bool) or max_batches < 1):
        raise ValueError('Diagnostic max_batches must be a positive integer.')
    profiles, slots = {}, {}
    for stage, ids in targets.items():
        if ids:
            m = int(model.concept_creations[stage].visual_concepts.shape[0])
            profiles[stage] = {'patches': np.zeros(m), 'mass': np.zeros(m),
                               'videos': [set() for _ in range(m)]}
            slots[stage] = {'patches': 0, 'sum': 0, 'min': None, 'max': None, 'hist': {}}
    seen, processed = set(), 0
    for batch_index, names, rgb, length in subset_batches(loader, selected, seen, 'Selection profiles'):
        out = run_forward(model, rgb, train_cfg)
        for stage, profile in profiles.items():
            c = out['concept_out'][stage]
            validate_sparse_baseline(c, model.concept_creations[stage])
            indices, valid = c['active_visual_prototype_indices'], c['visual_validity_mask'].bool()
            b, t, h, w, k = indices.shape
            per_patch = valid.sum(-1).flatten().cpu()
            s = slots[stage]
            s['patches'] += int(per_patch.numel())
            s['sum'] += int(per_patch.sum())
            lo, hi = int(per_patch.min()), int(per_patch.max())
            s['min'] = lo if s['min'] is None else min(s['min'], lo)
            s['max'] = hi if s['max'] is None else max(s['max'], hi)
            for value, count in zip(*np.unique(per_patch.numpy(), return_counts=True)):
                s['hist'][int(value)] = s['hist'].get(int(value), 0) + int(count)
            # This is actual sparse prediction evidence, NOT full-bank soft
            # visual_assignment_probs retained for the auxiliary regularizers.
            a = c['visual_activations'].reshape(b, h*w, -1)
            profile['mass'] += a.double().sum((0, 1)).cpu().numpy()
            for i, v in enumerate(names):
                active = indices[i][valid[i]].cpu().numpy()
                counts = np.bincount(active, minlength=len(profile['patches']))
                profile['patches'] += counts
                for j in np.flatnonzero(counts):
                    profile['videos'][j].add(v)
        del out
        processed += 1
        if max_batches is not None and processed >= max_batches:
            break
    # A capped diagnostic cannot establish full support, even if every video
    # appeared in its first few windows.
    partial = max_batches is not None
    result = {stage: np.column_stack((p['patches'], p['mass'], [len(v) for v in p['videos']]))
              for stage, p in profiles.items()}
    report = selection_support_report(model, targets, result, slots, seen, selected, partial, processed)
    directory = Path(SELECTION_DIAGNOSTIC_OUTPUT_DIR if partial else LOCAL_OUTPUT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    name = 'selection_support_diagnostic.json' if partial else 'selection_support_report.json'
    (directory / name).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if partial:
        return result, report
    if seen != selected:
        raise ValueError(f'Selection videos missing from loader: {sorted(selected-seen)}')
    for stage in profiles:
        zero = report['stages'][stage]['zero_support_ids']
        if zero:
            raise ValueError(f'{stage}: HIGH targets have no valid selection assignments '
                             f'(zero-support IDs: {zero}). '
                             'Regenerate selection for this top-k model; no targets were silently removed.')
    return result


@torch.no_grad()
def selection_diagnostic(max_batches):
    """Selection-only support pass: no deletion or evaluation-video ranking."""
    import train as train_cfg
    if max_batches < 1:
        raise ValueError('Diagnostic batch limit must be positive.')
    targets = load_high_targets()
    selection, _ = read_split()
    seed = train_cfg.SEED if SEED is None else SEED
    train_cfg.set_seed(seed)
    backbone_device, head_device = _resolve_devices()
    model = build_model(backbone_device, head_device)
    _check_method(model, 'ExplainableVidSalModel', 'forward')
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()
    model.allow_eval_concept_losses = False
    _validate_model_configuration(model)
    PrototypeEvidenceDeletionAdapter(model, targets)
    loader, _, _ = build_val_loader()
    _, report = selection_profiles(model, loader, selection, targets, train_cfg, max_batches=max_batches)
    print(f'Saved diagnostic: {Path(SELECTION_DIAGNOSTIC_OUTPUT_DIR) / "selection_support_diagnostic.json"}', flush=True)
    return report



def fit_targets_to_control_pool(profiles, targets):
    """Per stage, keep the largest E(G)-ranked prefix with equal-count used non-target controls."""
    fitted, adjustment = {}, {}
    for stage, original in targets.items():
        if not original:
            continue
        ranked = ranked_by_eg(high_scores(stage), original)
        pool = {int(j) for j in np.flatnonzero(profiles[stage][:, 0] > 0)}
        missing = sorted(set(ranked) - pool)
        if missing:
            raise ValueError(f'{stage}: HIGH targets lack selection support: {missing}.')
        count = next((n for n in range(len(ranked), 0, -1)
                      if len(pool - set(ranked[:n])) >= n), 0)
        if not count:
            raise ValueError(f'{stage}: no nonempty E(G)-ranked target prefix has equal-count '
                             f'distinct used non-target controls (used prototypes={len(pool)}).')
        kept = ranked[:count]
        fitted[stage] = kept
        adjustment[stage] = {
            'requested_count': len(original),
            'total_selection_used_prototypes': len(pool),
            'actual_high_count': count,
            'available_used_non_target_controls': len(pool - set(kept)),
            'original_candidate_ids_ranked': ranked,
            'retained_ids_ranked': kept,
            'dropped_ids_ranked': ranked[count:],
            'rule': 'largest per-stage E(G)-ranked prefix with equal-count distinct used non-target controls',
            'uses_evaluation_outputs': False,
        }
        print(f'{stage}: HIGH target count {len(original)} -> {count}; '
              f'distinct used non-target controls={len(pool-set(kept))}', flush=True)
    return _normalized_targets(fitted), adjustment


def match_controls(profiles, targets):
    if CONTROL_REPEATS < 2 or CONTROL_NEAREST_POOL < 1:
        raise ValueError('Use at least two control repeats and a positive nearest pool.')
    controls, diagnostics = {}, []
    for repeat in range(CONTROL_REPEATS):
        rng = np.random.default_rng(CONTROL_SEED + repeat)
        condition = f'control_{repeat:03d}'
        matched = {}
        for stage, high in targets.items():
            if not high:
                continue
            raw = profiles[stage]
            pool = np.array([j for j in range(len(raw)) if j not in high and raw[j,0] > 0], dtype=int)
            if len(pool) < len(high):
                raise ValueError(f'{stage}: insufficient used non-target prototypes for equal-count controls.')
            transformed = np.log1p(raw)
            scale = transformed[np.r_[pool, high]].std(axis=0)
            scale[scale < 1e-12] = 1.0
            z = transformed / scale
            assigned = {}
            # Unique controls within a set; no test saliency or deletion effects
            # are consulted. Match stage, counts, patch usage, mass, video support.
            for target in rng.permutation(high):
                distance = np.linalg.norm(z[pool] - z[target], axis=1)
                nearest = np.argsort(distance, kind='stable')[:min(CONTROL_NEAREST_POOL, len(pool))]
                pos = int(rng.choice(nearest))
                control = int(pool[pos])
                assigned[int(target)] = control
                diagnostics.append({'condition': condition, 'stage': stage,
                    'target': int(target), 'control': control,
                    'standardized_log_distance': float(distance[pos]),
                    'target_patch_count': float(raw[target,0]), 'control_patch_count': float(raw[control,0]),
                    'target_assignment_mass': float(raw[target,1]), 'control_assignment_mass': float(raw[control,1]),
                    'target_video_support': int(raw[target,2]), 'control_video_support': int(raw[control,2])})
                pool = np.delete(pool, pos)
            matched[stage] = sorted(assigned.values())
        controls[condition] = matched
    return controls, diagnostics








def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)






def last_real_map(target, length, indices):
    if not isinstance(target,torch.Tensor):
        raise ValueError('Loader must supply ground-truth density and true fixation maps.')
    x=target[indices]
    if x.ndim==3:return x.unsqueeze(1)
    if x.ndim==4:
        if x.shape[1]==1:return x
        if x.shape[1]<length:raise ValueError('Target time axis is shorter than the RGB window.')
        return x[:,length-1:length]
    if x.ndim==5:
        if x.shape[1]==1:
            if x.shape[2]<length:raise ValueError('Target time axis is too short.')
            return x[:,:,length-1]
        if x.shape[2]==1:
            if x.shape[1]<length:raise ValueError('Target time axis is too short.')
            return x[:,length-1]
    raise ValueError(f'Unsupported target layout: {tuple(x.shape)}.')


def metric_batches(loader, allowed, seen):
    for batch_index,batch in enumerate(tqdm(loader,desc='Paired metric evaluation',unit='batch')):
        if len(batch)!=6:raise ValueError('Expected filenames, RGB, density, fixations, n_frames, valid_mask.')
        names,rgb,density,fixations,lengths,valid=batch
        ids=[video_id(n) for n in names]
        selected=[i for i,v in enumerate(ids) if v in allowed]
        for length in sorted({int(lengths[i]) for i in selected}):
            if length<1:raise ValueError('Empty RGB window.')
            idx=[i for i in selected if int(lengths[i])==length]
            if valid is not None:
                if valid.ndim==1:idx=[i for i in idx if bool(valid[i])]
                elif valid.ndim==2:idx=[i for i in idx if bool(valid[i,length-1])]
                else:raise ValueError('Expected [B,T] temporal validity or [B] last-frame validity.')
            if not idx:continue
            seen.update(ids[i] for i in idx)
            yield batch_index,[ids[i] for i in idx],rgb[idx,:length],last_real_map(density,length,idx),last_real_map(fixations,length,idx),length


def load_metric_functions():
    try:module=importlib.import_module('model.metrics')
    except ModuleNotFoundError as exc:
        if exc.name!='model.metrics':raise
        module=importlib.import_module('metrics')
    return {name:getattr(module,fn) for name,fn in [('CC','cc_score'),('NSS','nss_score'),('SIM','sim_score')]}


def metric_prediction(prediction):
    x=prediction['saliency_logits'] if METRIC_INPUT=='sigmoid' else prediction['saliency_map']
    x=x.float()
    if x.ndim!=4 or x.shape[1]!=1 or not torch.isfinite(x).all():
        raise FloatingPointError('Expected finite [B,1,H,W] prediction maps.')
    return x.sigmoid() if METRIC_INPUT=='sigmoid' else x


def region_pixel_mask(patch_mask, prediction_hw, target_hw):
    # Expand into exactly the original patch footprints first. Then nearest
    # mapping to the label resolution keeps masks binary and spatially fixed.
    h,w=patch_mask.shape[-2:];ph,pw=prediction_hw
    if ph%h or pw%w:raise ValueError('Prediction must be divisible by the concept grid.')
    mask=patch_mask.float().unsqueeze(1)
    mask=F.interpolate(mask,size=prediction_hw,mode='nearest')
    if tuple(target_hw)!=tuple(prediction_hw):mask=F.interpolate(mask,size=target_hw,mode='nearest')
    return mask[:,0].bool()


def score_maps(pred, density, fixations, functions, patch_mask=None):
    """Per-window whole-map or conditional within-ROI metrics.

    Restricted pixels are gathered, never zero-padded outside the ROI. CC and
    SIM use the density resolution; NSS uses the true fixation resolution.
    Prediction is bilinearly resized first, matching the project helper's
    resize direction. Inside ROI, the SAME project metric functions operate on
    [1,1,1,N] vectors: CC centers within ROI, NSS z-scores within ROI, and SIM
    min-max/sum-normalizes within ROI. NSS without ROI fixations is undefined.
    """
    if density.ndim!=4 or fixations.ndim!=4 or density.shape[1]!=1 or fixations.shape[1]!=1:
        raise ValueError('Expected last-real density/fixation maps [B,1,H,W].')
    density=density.to(pred.device);fixations=fixations.to(pred.device)
    if not torch.isfinite(density).all() or (density<0).any():raise ValueError('Invalid ground-truth density.')
    if not torch.isfinite(fixations).all() or (fixations<0).any():raise ValueError('Invalid true fixation map.')
    binary=(fixations>0).float()
    pred_density=pred if pred.shape[-2:]==density.shape[-2:] else F.interpolate(pred,size=density.shape[-2:],mode='bilinear',align_corners=False)
    pred_fix=pred if pred.shape[-2:]==fixations.shape[-2:] else F.interpolate(pred,size=fixations.shape[-2:],mode='bilinear',align_corners=False)
    if patch_mask is None:
        density_mask=torch.ones_like(density[:,0],dtype=torch.bool)
        fixation_mask=torch.ones_like(binary[:,0],dtype=torch.bool)
    else:
        density_mask=region_pixel_mask(patch_mask,pred.shape[-2:],density.shape[-2:])
        fixation_mask=region_pixel_mask(patch_mask,pred.shape[-2:],fixations.shape[-2:])
    rows=[]
    for i in range(len(pred)):
        dm,fm=density_mask[i],fixation_mask[i]
        fixation_count=int(binary[i,0][fm].sum())
        row={'density_pixels':int(dm.sum()),'fixation_region_pixels':int(fm.sum()),'fixation_pixels':fixation_count}
        for metric,fn in functions.items():
            mask=fm if metric=='NSS' else dm
            values=pred_fix if metric=='NSS' else pred_density
            target=binary if metric=='NSS' else density
            if not mask.any() or (metric=='NSS' and fixation_count==0):
                row[metric]=None
                row[f'undefined_{metric}']='no_true_fixations' if metric=='NSS' else 'empty_region'
                continue
            if patch_mask is None:
                value=float(fn(pred[i:i+1],target[i:i+1]).detach().cpu())
            else:
                p=values[i,0][mask].view(1,1,1,-1)
                q=target[i,0][mask].view(1,1,1,-1)
                if p.numel()<2:
                    row[metric]=None;row[f'undefined_{metric}']='fewer_than_two_region_pixels';continue
                value=float(fn(p,q).detach().cpu())
            row[metric]=value if math.isfinite(value) else None
            row[f'undefined_{metric}']='nonfinite_metric_constant_or_degenerate_map' if row[metric] is None else ''
        rows.append(row)
    return rows


def bootstrap_ci(effects):
    if len(effects)<2:return None
    effect=np.asarray(effects,dtype=float)
    rng=np.random.default_rng(BOOTSTRAP_SEED);means=[]
    for start in range(0,BOOTSTRAP_REPEATS,1000):
        count=min(1000,BOOTSTRAP_REPEATS-start)
        indices=rng.integers(len(effect),size=(count,len(effect)))
        means.extend(effect[indices].mean(1).tolist())
    return np.quantile(means,[.025,.975]).tolist()


def summarize_metric_rows(frame_rows, controls):
    summaries={'overall':{},'baseline_top5pct':{}}
    video_rows=[];table_rows=[]
    groups={}
    for row in frame_rows:
        key=(row['stage'],row['region'],row['video'],row['loader_batch'],row['sample'],row['window_length'])
        groups.setdefault(key,{})[row['condition']]=row
    scopes=sorted({(k[0],k[1]) for k in groups})
    required=['baseline','high_deleted',*controls]
    for stage,region in scopes:
        all_windows=[(k,d) for k,d in groups.items() if k[:2]==(stage,region)]
        if any(set(d)!=set(required) for _,d in all_windows):raise RuntimeError('Incomplete paired baseline/HIGH/control windows.')
        scope=summaries['overall'] if region=='overall' else summaries['baseline_top5pct'].setdefault(stage,{})
        for metric in METRIC_NAMES:
            # One common valid frame set across baseline/HIGH/ALL controls.
            valid=[(k,d) for k,d in all_windows if all(d[c][metric] is not None for c in required)]
            by_video={}
            for k,d in valid:by_video.setdefault(k[2],[]).append(d)
            window_means={c:float(np.mean([d[c][metric] for _,d in valid])) if valid else None for c in required}
            video_means={c:[] for c in required}
            for video,items in sorted(by_video.items()):
                record={'video':video,'stage':stage,'region':region,'metric':metric,'paired_window_count':len(items)}
                for c in required:
                    mean=float(np.mean([d[c][metric] for d in items]));video_means[c].append(mean);record[c]=mean
                record['control_mean']=float(np.mean([record[c] for c in controls]))
                record['HIGH_delta']=record['high_deleted']-record['baseline']
                record['control_delta']=record['control_mean']-record['baseline']
                record['HIGH_minus_control_delta']=record['HIGH_delta']-record['control_delta']
                video_rows.append(record)
            vb=np.asarray(video_means['baseline']);vh=np.asarray(video_means['high_deleted'])
            vc=np.mean([video_means[c] for c in controls],axis=0) if len(vb) else np.array([])
            high_delta=vh-vb;control_delta=vc-vb;contrast=vh-vc
            summary={
                'evaluated_windows':len(all_windows),'paired_windows':len(valid),'excluded_windows':len(all_windows)-len(valid),
                'paired_videos':len(by_video),'video_ids':sorted(by_video),
                'undefined_window_counts':{c:sum(d[c][metric] is None for _,d in all_windows) for c in required},
                'baseline_window_mean':window_means['baseline'],'high_deleted_window_mean':window_means['high_deleted'],
                'control_window_mean':float(np.mean([window_means[c] for c in controls])) if valid else None,
                'baseline_video_mean':float(vb.mean()) if len(vb) else None,
                'high_deleted_video_mean':float(vh.mean()) if len(vb) else None,
                'control_video_mean':float(vc.mean()) if len(vb) else None,
                'HIGH_delta_video_mean':float(high_delta.mean()) if len(vb) else None,
                'control_delta_video_mean':float(control_delta.mean()) if len(vb) else None,
                'HIGH_minus_control_delta_video_mean':float(contrast.mean()) if len(vb) else None,
                'HIGH_delta_paired_video_bootstrap_95_ci':bootstrap_ci(high_delta),
                'HIGH_minus_control_delta_paired_video_bootstrap_95_ci':bootstrap_ci(contrast),
                'control_set_video_mean_range':[float(min(np.mean(video_means[c]) for c in controls)),float(max(np.mean(video_means[c]) for c in controls))] if len(vb) else None}
            scope[metric]=summary
            for c in required:
                mean=window_means[c];base=window_means['baseline']
                table_rows.append({'stage':stage,'region':region,'metric':metric,'condition':c,
                                   'window_mean':mean,'delta_window_mean':mean-base if mean is not None and base is not None else None,
                                   'video_mean':float(np.mean(video_means[c])) if len(vb) else None,
                                   'paired_windows':len(valid),'excluded_windows':len(all_windows)-len(valid)})
    return summaries,video_rows,table_rows


def print_metric_summary(table_rows):
    def fmt(x):return 'undefined' if x is None else f'{x:.6f}'
    print('\nCC/NSS/SIM on the same joint selected-ID deletion (equal-window paired means):',flush=True)
    grouped={}
    for row in table_rows:grouped.setdefault((row['stage'],row['region'],row['condition']),{})[row['metric']]=row
    print('Region / grid                 Condition        CC          NSS          SIM         dCC         dNSS         dSIM',flush=True)
    for (stage,region,condition),metrics in grouped.items():
        values=[metrics[m]['window_mean'] for m in METRIC_NAMES]
        deltas=[metrics[m]['delta_window_mean'] for m in METRIC_NAMES]
        label='overall / joint' if region=='overall' else f'baseline top5% / {stage}'
        print(f'{label:29} {condition:14} '+' '.join(f'{fmt(v):>11}' for v in values+deltas),flush=True)


def cluster_window_bootstrap_ci(video_effects, video_window_counts):
    """Paired whole-video resampling for the existing equal-window estimand.

    A selected video brings all its common finite paired windows with it. Its
    summed effect is its mean effect times its window count; the denominator is
    recomputed for every draw. This is not a bootstrap of independent windows.
    """
    effects = np.asarray(video_effects, dtype=float)
    counts = np.asarray(video_window_counts, dtype=float)
    if effects.shape != counts.shape or effects.ndim != 1:
        raise ValueError('Expected one paired effect and window count per video.')
    if not np.isfinite(effects).all() or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError('Cluster bootstrap requires finite effects and positive window counts.')
    if len(effects) < 2:
        return None
    if BOOTSTRAP_REPEATS < 1:
        raise ValueError('BOOTSTRAP_REPEATS must be positive.')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sums = effects * counts
    means = []
    for start in range(0, BOOTSTRAP_REPEATS, 1000):
        size = min(1000, BOOTSTRAP_REPEATS - start)
        indices = rng.integers(len(effects), size=(size, len(effects)))
        means.extend((sums[indices].sum(1) / counts[indices].sum(1)).tolist())
    return np.quantile(means, [.025, .975]).tolist()


def paired_confidence_interval_rows(summaries, video_rows, controls):
    """HIGH versus baseline, mean control and each fixed sampled control set."""
    grouped = {}
    for row in video_rows:
        grouped.setdefault((row['stage'], row['region'], row['metric']), []).append(row)
    scopes = [('joint', 'overall', summaries['overall'])]
    scopes.extend((stage, 'baseline_top5pct', metrics)
                  for stage, metrics in summaries['baseline_top5pct'].items())
    reports = []
    for stage, region, metrics in scopes:
        for metric in METRIC_NAMES:
            summary = metrics[metric]
            items = grouped.get((stage, region, metric), [])
            counts = np.asarray([r['paired_window_count'] for r in items], dtype=float)
            comparisons = [('baseline', 'HIGH_delta'), ('mean_control', 'HIGH_minus_control_delta')]
            comparisons.extend((c, None) for c in controls)
            for reference, field in comparisons:
                effects = np.asarray([r[field] if field else r['high_deleted'] - r[reference]
                                      for r in items], dtype=float)
                ci = cluster_window_bootstrap_ci(effects, counts)
                reports.append({
                    'stage': stage, 'region': region, 'metric': metric,
                    'condition': 'high_deleted', 'reference': reference,
                    'delta_window_mean': float(np.dot(effects, counts) / counts.sum()) if items else None,
                    'ci_95_low': ci[0] if ci is not None else None,
                    'ci_95_high': ci[1] if ci is not None else None,
                    'paired_windows': summary['paired_windows'],
                    'paired_videos': summary['paired_videos'],
                    'excluded_windows': summary['excluded_windows'],
                    'aggregation': 'equal-window paired mean',
                    'bootstrap_unit': 'whole video with all paired windows',
                    'bootstrap_repeats': BOOTSTRAP_REPEATS, 'bootstrap_seed': BOOTSTRAP_SEED,
                    'ci_status': 'available' if ci is not None else 'fewer_than_two_paired_videos',
                })
    return reports


def print_paired_confidence_intervals(rows):
    print('\nPaired 95% confidence intervals (equal-window effects; whole-video cluster bootstrap):', flush=True)
    print('Delta = HIGH deletion minus reference; negative means worse CC/NSS/SIM.', flush=True)
    print('Region / grid                 Metric Reference           Mean delta                  95% CI   Windows Videos', flush=True)
    for row in rows:
        label = 'overall / joint' if row['region'] == 'overall' else f'baseline top5% / {row["stage"]}'
        delta = 'undefined' if row['delta_window_mean'] is None else f'{row["delta_window_mean"]:+.6f}'
        ci = 'unavailable (<2 videos)' if row['ci_95_low'] is None else f'[{row["ci_95_low"]:+.6f}, {row["ci_95_high"]:+.6f}]'
        print(f'{label:29} {row["metric"]:6} {row["reference"]:17} {delta:>11} {ci:>25} '
              f'{row["paired_windows"]:9} {row["paired_videos"]:6}', flush=True)
    print('All comparisons use each metric\'s common finite baseline/HIGH/all-control window set. '
          'Control intervals are conditional on the sampled fixed control sets.', flush=True)


def deletion_evidence_rows(stats, conditions):
    """Aggregate actual evaluation removal, never selection-support estimates."""
    rows = []
    for stage in DELETION_STAGES:
        for condition, targets in conditions.items():
            s = stats[condition].get(stage)
            if s is None:
                raise RuntimeError(f'Missing evaluation deletion statistics for {condition}/{stage}.')
            slots = s['original_valid_slots']
            mass = s['original_assignment_mass']
            rows.append({
                'stage': stage, 'condition': condition,
                'deleted_id_count': len(set(targets.get(stage, []))),
                'removed_active_slots': s['removed_valid_slots'],
                'removed_assignment_mass': s['removed_assignment_mass'],
                'original_active_slots': slots, 'original_assignment_mass': mass,
                'removed_active_slot_percent': 100. * s['removed_valid_slots'] / slots if slots else None,
                'removed_assignment_mass_percent': 100. * s['removed_assignment_mass'] / mass if mass else None,
                'evaluated_windows': s['evaluated_windows'], 'patches': s['patches'],
                'patches_empty_after_deletion': s['patches_empty_after_deletion'],
            })
    return rows


def print_deletion_evidence_summary(rows):
    print('\nActual evidence removed on evaluation windows, per stage:', flush=True)
    for stage in DELETION_STAGES:
        print(f'\n{stage}', flush=True)
        print('Condition       Deleted ID count  Removed active slots  Removed assignment mass     Slots %      Mass %', flush=True)
        for row in rows:
            if row['stage'] != stage:
                continue
            label = 'HIGH deletion' if row['condition'] == 'high_deleted' else row['condition'].replace('control_', 'Control ')
            pct = lambda x: 'undefined' if x is None else f'{x:.6f}'
            print(f'{label:15} {row["deleted_id_count"]:16} {row["removed_active_slots"]:21} '
                  f'{row["removed_assignment_mass"]:24.6f} {pct(row["removed_active_slot_percent"]):>11} '
                  f'{pct(row["removed_assignment_mass_percent"]):>11}', flush=True)
    print('Deleted ID count includes inactive IDs. Slots count patch-prototype occurrences; '
          'mass sums their original sparse assignment weights before deletion/downstream normalization. '
          'Percentages use each condition/stage\'s original evidence on the same evaluated windows; '
          'windows excluded from individual undefined metrics remain included in these removal totals.', flush=True)


@torch.no_grad()
def evaluate_local():
    import train as train_cfg
    targets=load_high_targets();selection,evaluation=read_split()
    output=Path(LOCAL_OUTPUT_DIR);output.mkdir(parents=True,exist_ok=True)
    (output/'local_deletion_results.json').write_text(json.dumps({'status':'incomplete','targets':targets})+'\n')
    train_cfg.set_seed(train_cfg.SEED if SEED is None else SEED)
    model=build_model(*_resolve_devices());_check_method(model,'ExplainableVidSalModel','forward')
    checkpoint=load_checkpoint(model,CHECKPOINT_PATH)
    model.eval();model.allow_eval_concept_losses=False;_validate_model_configuration(model)
    PrototypeEvidenceDeletionAdapter(model,targets)
    loader,dataset,window_len=build_val_loader()
    profiles=selection_profiles(model,loader,selection,targets,train_cfg)
    targets,target_count_adjustment=fit_targets_to_control_pool(profiles,targets)
    (output/'target_count_adjustment.json').write_text(json.dumps(target_count_adjustment,indent=2)+'\n')
    support_path=output/'selection_support_report.json';support_report=json.loads(support_path.read_text())
    support_report['deletion_targets']=targets;support_report['target_count_adjustment']=target_count_adjustment
    support_path.write_text(json.dumps(support_report,indent=2)+'\n')
    controls,matching=match_controls(profiles,targets)
    conditions={'high_deleted':targets,**controls}
    write_csv(output/'control_matching.csv',matching)
    (output/'fixed_controls.json').write_text(json.dumps(conditions,indent=2)+'\n')
    functions=load_metric_functions()
    frame_rows=[];seen=set();stats={name:{} for name in conditions}
    fields=['video','loader_batch','sample','window_length','stage','region','condition',*METRIC_NAMES,
            *(f'baseline_{m}' for m in METRIC_NAMES),*(f'delta_{m}' for m in METRIC_NAMES),
            'density_pixels','fixation_region_pixels','fixation_pixels',*(f'undefined_{m}' for m in METRIC_NAMES)]
    roi_audit={'selection':'baseline sigmoid map patch means; ceil(5% of each stage grid), stable raster ties',
               'mask_mapping':'original patch footprints at prediction resolution, then nearest mapping to each label grid',
               'frozen_for_all_conditions':True,'stages':{}}
    with (output/'frame_results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for batch_index,names,rgb,density,fixations,length in metric_batches(loader,evaluation,seen):
            baseline=run_forward(model,rgb,train_cfg)
            base_pred=metric_prediction(baseline)
            base_sigmoid=measurement_saliency_map(baseline,model.saliency_prediction.output_activation)
            masks={}
            for stage in targets:
                c=baseline['concept_out'][stage];validate_sparse_baseline(c,model.concept_creations[stage])
                hw=c['visual_validity_mask'].shape[2:4]
                masks[stage]=baseline_high_region(patch_maps(base_sigmoid,hw))
                roi_audit['stages'][stage]={'patch_grid':list(hw),'selected_patches_per_window':int(masks[stage][0].sum())}
            base_scores={('joint','overall'):score_maps(base_pred,density,fixations,functions)}
            base_scores.update({(s,'baseline_top5pct'):score_maps(base_pred,density,fixations,functions,mask) for s,mask in masks.items()})
            def append(condition,scores_by_region):
                for (stage,region),scores in scores_by_region.items():
                    for i,video in enumerate(names):
                        values=scores[i];base=base_scores[(stage,region)][i]
                        row={'video':video,'loader_batch':batch_index,'sample':i,'window_length':length,
                             'stage':stage,'region':region,'condition':condition,**values}
                        for metric in METRIC_NAMES:
                            row[f'baseline_{metric}']=base[metric]
                            row[f'delta_{metric}']=values[metric]-base[metric] if values[metric] is not None and base[metric] is not None else None
                        writer.writerow(row);frame_rows.append(row)
                handle.flush()
            append('baseline',base_scores)
            del baseline,base_pred,base_sigmoid
            for condition,deleted_ids in tqdm(conditions.items(),desc='Joint deletion conditions',leave=False):
                adapter=PrototypeEvidenceDeletionAdapter(model,deleted_ids)
                with adapter:changed=run_forward(model,rgb,train_cfg)
                pred=metric_prediction(changed)
                scores={('joint','overall'):score_maps(pred,density,fixations,functions)}
                scores.update({(s,'baseline_top5pct'):score_maps(pred,density,fixations,functions,mask) for s,mask in masks.items()})
                append(condition,scores)
                for stage,s in adapter.stats.items():
                    totals=stats[condition].setdefault(stage,{'patches':0,'removed_valid_slots':0,'patches_empty_after_deletion':0,
                        'removed_assignment_mass':0.,'original_valid_slots':0,'original_assignment_mass':0.,'evaluated_windows':0})
                    for k in totals:
                        if k!='evaluated_windows':totals[k]+=s[k]
                    totals['evaluated_windows']+=len(names)
                del changed,pred
            print(f'Batch {batch_index}: paired baseline/HIGH/{len(controls)} controls, overall and four fixed-region metric sets.',flush=True)
    if seen!=evaluation:raise ValueError(f'Evaluation videos missing valid labeled windows: {sorted(evaluation-seen)}')
    if not frame_rows:raise RuntimeError('No paired metric windows were evaluated.')
    summaries,video_rows,table_rows=summarize_metric_rows(frame_rows,list(controls))
    write_csv(output/'video_results.csv',video_rows);write_csv(output/'metric_summary.csv',table_rows)
    ci_rows=paired_confidence_interval_rows(summaries,video_rows,list(controls))
    evidence_rows=deletion_evidence_rows(stats,conditions)
    write_csv(output/'confidence_intervals.csv',ci_rows)
    write_csv(output/'deletion_evidence_summary.csv',evidence_rows)
    print_metric_summary(table_rows)
    print_paired_confidence_intervals(ci_rows)
    print_deletion_evidence_summary(evidence_rows)
    provenance={}
    for name,fn in functions.items():
        source=inspect.getsourcefile(fn)
        provenance[name]={'callable':f'{fn.__module__}.{fn.__name__}','source_path':source,
                          'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest() if source and Path(source).is_file() else None}
    result={'status':'complete','checkpoint':CHECKPOINT_PATH,'checkpoint_epoch':checkpoint.get('epoch'),
        'weights_modified':False,'intervention':'post_assignment_no_replacement',
        'intervention_scope':'joint deletion of each stage\'s own top E(G) HIGH IDs at all four stages; regional stage labels are measurement grids, not separate ablations',
        'target_selection':'independent per-stage E(G) ranking','deletion_stages':list(DELETION_STAGES),
        'stage_identity_note':'independent learned banks; each stage\'s HIGH targets are selected from that stage\'s own scores',
        'measurement':'CC/NSS/SIM against true last-real density and fixation maps; overall and conditional baseline top5% regions',
        'metric_input':METRIC_INPUT,'metric_functions':provenance,
        'regional_metric_definition':{'CC':'project CC on paired pixels inside the frozen ROI; centered/normalized inside ROI',
                                      'NSS':'project NSS with prediction z-scored inside ROI, evaluated at true fixation pixels inside ROI; undefined without fixations',
                                      'SIM':'project SIM on paired ROI pixels, min-max and sum normalized inside ROI; conditional similarity, not full-map mass contribution'},
        'undefined_metric_policy':'null per-window values; each metric summary uses the common finite window set across baseline, HIGH, and ALL controls; exclusions counted',
        'roi_audit':roi_audit,'effect_sign':'deleted minus baseline; negative CC/NSS/SIM delta means worse',
        'output_activation':getattr(model.saliency_prediction,'output_activation',None),
        'aggregation':'paired equal-window means for console and confidence_intervals (whole-video cluster bootstrap); existing summary also retains equal-video means and paired video-bootstrap delta intervals',
        'uncertainty':'paired video bootstrap conditional on sampled control sets; existing summary intervals target equal-video means; confidence_intervals target equal-window means with whole-video cluster resampling',
        'matching':'same stage and ID count; nearest standardized log1p patch usage, assignment mass, video support',
        'matching_is_approximate':True,'selection_video_ids':sorted(selection),'evaluation_video_ids':sorted(evaluation),
        'targets':targets,'controls':controls,'control_seed':CONTROL_SEED,'bootstrap_seed':BOOTSTRAP_SEED,
        'control_repeats':CONTROL_REPEATS,'window_len':window_len,'dataset_windows':len(dataset),
        'deletion_stats':stats,'summary':summaries,
        'confidence_intervals':ci_rows,'deletion_evidence_summary':evidence_rows,
        'deletion_evidence_definition':'actual evaluation-window removal at all patches; configured unique ID count includes inactive IDs; active slots count valid patch-prototype occurrences; mass is summed original sparse assignment weight before deletion/downstream normalization',
        'model_config':{'top_k':MODEL_TOP_K,'assignment_mode':MODEL_ASSIGNMENT_MODE},
        'finalized_targets_path':FINALIZED_TARGETS_PATH,'high_prototypes_path':HIGH_PROTOTYPES_PATH,
        'high_target_count':HIGH_TARGET_COUNT,'actual_high_target_counts':{s:len(ids) for s,ids in targets.items()},
        'target_count_adjustment':target_count_adjustment,
        'high_selection_rule':'per stage: top high_target_count by E(G) among prototypes with high_patch_count > 0',
        'supported_method_fingerprints':SUPPORTED_METHODS,'supported_decoder_helper_fingerprints':SUPPORTED_DECODER_FUNCTIONS,
        'selection_profiles':{s:p.tolist() for s,p in profiles.items()},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluated_at':datetime.now(timezone.utc).isoformat(timespec='seconds')}
    (output/'local_deletion_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Complete: {output / "local_deletion_results.json"}',flush=True)
    return result


def local_self_test():
    torch.set_num_threads(1)
    functions=load_metric_functions()
    # The new peak moves outside H, but the baseline ROI is reused exactly.
    base_map=torch.tensor([[[[.9,.8,.7,.1,.2,.3]]]])
    changed=torch.tensor([[[[.7,.8,.9,.99,.98,.97]]]])
    mask=torch.tensor([[[True,True,True,False,False,False]]])
    gt=torch.tensor([[[[.9,.8,.7,.1,.2,.3]]]])
    fix=torch.tensor([[[[1.,0.,0.,0.,0.,0.]]]])
    b=score_maps(base_map,gt,fix,functions,mask)[0]
    d=score_maps(changed,gt,fix,functions,mask)[0]
    assert np.isclose(b['CC'],1.) and np.isclose(d['CC'],-1.)
    assert b['NSS']>0 and d['NSS']<0 and d['SIM']<b['SIM']
    # Changing outside pixels cannot affect conditional ROI metrics.
    outside=changed.clone();outside[:,:,:,3:]=-100.
    o=score_maps(outside,gt,fix,functions,mask)[0]
    assert all(np.isclose(o[m],d[m]) for m in METRIC_NAMES)
    # No ROI fixations is undefined, without inventing a fixation or a zero score.
    no_fix=fix.clone();no_fix.zero_()
    assert score_maps(changed,gt,no_fix,functions,mask)[0]['NSS'] is None
    flat=changed.clone();flat[:,:,:,:3]=.5
    assert score_maps(flat,gt,fix,functions,mask)[0]['CC'] is None
    # Pixel masks correspond to original patch footprints and stay binary.
    patch=torch.tensor([[[True,False],[False,False]]])
    expanded=region_pixel_mask(patch,(4,4),(8,8))
    assert expanded.shape==(1,8,8) and expanded.sum()==16
    assert expanded[0,:4,:4].all() and not expanded[0,4:].any()
    # True last frame for temporal layouts, including padded shorter windows.
    target=torch.arange(2*3*4*4).reshape(2,3,4,4).float()
    expected=target[1:2,1:2]
    torch.testing.assert_close(last_real_map(target,2,[1]),expected)
    torch.testing.assert_close(last_real_map(target.unsqueeze(1),2,[1]),expected)
    torch.testing.assert_close(last_real_map(target.unsqueeze(2),2,[1]),expected)
    # Separate global / ROI summaries with known paired video contrasts.
    rows=[]
    for video in ['a','b']:
        for condition,value in [('baseline',1.),('high_deleted',.5),('control_000',.9),('control_001',.8)]:
            rows.append({'video':video,'loader_batch':0,'sample':0,'window_length':1,'stage':'joint','region':'overall',
                         'condition':condition,'CC':value,'NSS':value,'SIM':value})
    summary,_,_=summarize_metric_rows(rows,['control_000','control_001'])
    assert np.isclose(summary['overall']['CC']['HIGH_delta_video_mean'],-.5)
    assert np.isclose(summary['overall']['CC']['HIGH_minus_control_delta_video_mean'],-.35)
    assert np.allclose(summary['overall']['CC']['HIGH_delta_paired_video_bootstrap_95_ci'],[-.5,-.5])
    rows[-1]['CC']=None
    summary,_,_=summarize_metric_rows(rows,['control_000','control_001'])
    assert summary['overall']['CC']['excluded_windows']==1
    assert summary['overall']['NSS']['paired_windows']==2
    # Window-weighted effects must stay consistent with the console, while the
    # bootstrap resamples videos rather than treating their windows as iid.
    uneven=[]
    for video,count,effect in [('a',1,-.8),('b',3,-.2)]:
        for sample in range(count):
            for condition,value in [('baseline',1.),('high_deleted',1.+effect),('control_000',.9),('control_001',.8)]:
                uneven.append({'video':video,'loader_batch':0,'sample':sample,'window_length':1,
                               'stage':'joint','region':'overall','condition':condition,
                               'CC':value,'NSS':value,'SIM':value})
    paired,per_video,_=summarize_metric_rows(uneven,['control_000','control_001'])
    ci_rows=paired_confidence_interval_rows(paired,per_video,['control_000','control_001'])
    cc={r['reference']:r for r in ci_rows if r['metric']=='CC'}
    assert np.isclose(cc['baseline']['delta_window_mean'],-.35)
    assert np.isclose(paired['overall']['CC']['HIGH_delta_video_mean'],-.5)
    assert np.allclose([cc['baseline']['ci_95_low'],cc['baseline']['ci_95_high']],[-.8,-.2])
    assert np.isclose(cc['mean_control']['delta_window_mean'],-.2)
    assert np.isclose(cc['control_000']['delta_window_mean'],-.25)
    assert np.isclose(cc['control_001']['delta_window_mean'],-.15)
    assert cc['baseline']['paired_windows']==4 and cc['baseline']['paired_videos']==2
    # Fewer than two videos cannot produce a confidence interval. Undefined
    # metrics still receive explicit rows rather than disappearing from reports.
    one=[r for r in uneven if r['video']=='a']
    paired,per_video,_=summarize_metric_rows(one,['control_000','control_001'])
    assert all(r['ci_95_low'] is None for r in paired_confidence_interval_rows(paired,per_video,['control_000','control_001']))
    for r in one:r['NSS']=None
    paired,per_video,_=summarize_metric_rows(one,['control_000','control_001'])
    missing=[r for r in paired_confidence_interval_rows(paired,per_video,['control_000','control_001']) if r['metric']=='NSS']
    assert all(r['delta_window_mean'] is None and r['paired_windows']==0 for r in missing)
    print('PASS: frozen ROI, actual project CC/NSS/SIM, conditional pixel-only metrics, shifted peaks, outside independence, undefined NSS/CC, mask mapping, real-frame labels, paired deltas/CIs and common-window exclusions.')


def adapter_self_test():
    """Exercise the reviewed concept/decoder code without a dataset or checkpoint."""
    from model.concept_creation import VisualConceptCreation
    from model.saliency_prediction import ConceptGatedMultiScaleSaliencyDecoder

    class Harness(torch.nn.Module):
        def __init__(self, position, residual):
            super().__init__()
            self.concept_creations = torch.nn.ModuleDict({
                s: VisualConceptCreation(in_channels=8, concept_dim=8, num_concepts=16,
                     hidden_dim=16, top_k=8, visual_assignment_mode='topk_soft',
                     visual_concept_residual_weight=residual)
                for s in ('stage4','stage3','stage2','stage1')})
            self.saliency_prediction = ConceptGatedMultiScaleSaliencyDecoder(
                stage_channels={s:8 for s in self.concept_creations}, concept_dim=8,
                decoder_channels=8, dropout=0, use_shared_concept_activations=True,
                prototype_application_position=position, temporal_aggregation='last',
                output_activation='none', stage2_backbone_fusion_enabled=False)

        def forward(self, features):
            concepts = {s:c(features[s], return_losses=False) for s,c in self.concept_creations.items()}
            prediction = self.saliency_prediction(concepts, features, output_size=(32,32), return_details=True)
            return concepts, prediction

    torch.manual_seed(42)
    torch.set_num_threads(1)
    features = {s:torch.randn(1,8,2,h,h) for s,h in (('stage4',2),('stage3',4),('stage2',8),('stage1',16))}
    for position in ('pre_refine','post_refine'):
        for residual in (0.0,1.0):
            model = Harness(position, residual).eval()
            original, baseline = model(features)
            weights_before = {k:v.clone() for k,v in model.state_dict().items()}
            for stage, c in original.items():
                validate_sparse_baseline(c, model.concept_creations[stage])
                assert (baseline['stage_concept_priorities'][stage].gt(0).sum(2)==8).all()
                block = model.saliency_prediction.fusion_blocks[stage]
                projected = block.feature_proj(features[stage])[:, :, -1:]
                modality = block._resolve_visual_activations(projected,c['active_visual_prototypes'],c['visual_validity_mask'],c)
                p1 = block._prioritize_concept_patch_activations([modality])
                p2 = _prioritize_allowing_empty(block,[modality])
                torch.testing.assert_close(p1['concept_priorities'],p2['concept_priorities'],rtol=0,atol=0)
                torch.testing.assert_close(p1['patch_priority_gate'],p2['patch_priority_gate'],rtol=0,atol=0)
            with PrototypeEvidenceDeletionAdapter(model, {}):
                _, identity = model(features)
            torch.testing.assert_close(baseline['saliency_map'],identity['saliency_map'],rtol=0,atol=0)
            unused = [j for j in range(16) if not original['stage4']['visual_activations'][:,j].count_nonzero()]
            if unused:
                with PrototypeEvidenceDeletionAdapter(model, {'stage4':[unused[0]]}):
                    _, unchanged = model(features)
                torch.testing.assert_close(baseline['saliency_map'],unchanged['saliency_map'],rtol=0,atol=0)
            # Delete the original rank-8 ID at one source patch. Its evidence is
            # removed everywhere it originally occurred, without choosing a new ID.
            selected_id = int(original['stage4']['active_visual_prototype_indices'][0,0,0,0,-1])
            with PrototypeEvidenceDeletionAdapter(model, {'stage4':[selected_id]}) as adapter:
                removed, prediction = model(features)
                c = removed['stage4']; old = original['stage4']
                assert torch.equal(c['active_visual_prototype_indices'],old['active_visual_prototype_indices'])
                torch.testing.assert_close(c['visual_activations'][:,[j for j in range(16) if j!=selected_id]],
                    old['visual_activations'][:,[j for j in range(16) if j!=selected_id]],rtol=0,atol=0)
                assert not c['visual_activations'][:,selected_id].count_nonzero()
                assert torch.equal(c['visual_validity_mask'],old['visual_validity_mask'] & (old['active_visual_prototype_indices'] != selected_id))
                assert not torch.equal(prediction['saliency_map'],baseline['saliency_map'])
                assert adapter.stats['stage4']['removed_valid_slots']>0
                removal=adapter.stats['stage4']
                assert removal['removed_valid_slots']==int((old['visual_validity_mask'].bool() & (old['active_visual_prototype_indices']==selected_id)).sum())
                assert removal['original_valid_slots']==int(old['visual_validity_mask'].sum())
                assert np.isclose(removal['removed_assignment_mass'],float(old['visual_activations'][:,selected_id].double().sum()))
                assert np.isclose(removal['original_assignment_mass'],float(old['visual_activations'].double().sum()))
                expected=F.normalize(c['visual_activations']@F.normalize(model.concept_creations['stage4'].visual_concepts,dim=-1)+residual*c['visual_patch_embeddings'],dim=-1)
                torch.testing.assert_close(c['visual_concept_representation'],expected)
                # This intervention remains finite and freezes the same IDs under AMP.
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    amp_c,amp_prediction=model(features)
                assert torch.isfinite(amp_prediction['saliency_map']).all()
                assert not amp_c['stage4']['visual_activations'][:,selected_id].count_nonzero()
            # Remove all evidence: empty patches keep the original residual policy.
            all_ids={s:list(range(16)) for s in model.concept_creations}
            with PrototypeEvidenceDeletionAdapter(model,all_ids):
                empty, prediction = model(features)
                assert torch.isfinite(prediction['saliency_map']).all()
                assert all(torch.isfinite(x).all() for x in prediction['side_saliency_logits'].values())
                assert all(not x.count_nonzero() for x in prediction['stage_patch_priority_gates'].values())
                for s,c in empty.items():
                    assert not c['visual_activations'].count_nonzero()
                    expected=F.normalize(residual*c['visual_patch_embeddings'],dim=-1)
                    torch.testing.assert_close(c['visual_concept_representation'],expected)
                for dtype in (torch.float16, torch.bfloat16):
                    with torch.autocast('cpu',dtype=dtype):
                        amp_empty, amp_prediction = model(features)
                    assert torch.isfinite(amp_prediction['saliency_map']).all()
                    assert all(torch.isfinite(c['visual_concept_representation']).all()
                               and torch.isfinite(c['visual_feature_concept_agreement']).all()
                               for c in amp_empty.values())
            _, restored = model(features)
            torch.testing.assert_close(baseline['saliency_map'],restored['saliency_map'],rtol=0,atol=0)
            assert all(torch.equal(weights_before[k],v) for k,v in model.state_dict().items())
    print('PASS: reviewed fingerprints, eight valid slots, priority parity, frozen indices/weights, rank-8 deletion influence, AMP, empty patches, residual policy, main/side finiteness, restoration and unchanged weights.')


if __name__ == '__main__':
    if sys.argv[1:]==['--self-test']:
        with torch.no_grad():
            local_self_test()
            adapter_self_test()
    elif len(sys.argv) == 3 and sys.argv[1] == '--selection-diagnostic' and sys.argv[2].isdigit():
        selection_diagnostic(int(sys.argv[2]))
    elif sys.argv[1:]:
        raise SystemExit('Edit constants and run without arguments, or use --self-test or --selection-diagnostic N.')
    else:
        evaluate_local()
