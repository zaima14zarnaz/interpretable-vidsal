"""
Evaluate saved checkpoints on the validation set.

Default workflow:
1. Read training_outputs/val_metrics.csv for epochs with CC > 0.54 and SIM > 0.41.
2. Evaluate the matching checkpoints from checkpoint_dir.
3. Write evaluation_outputs/best_results.csv with CC, SIM, and NSS.

Run:

    python evaluation.py
"""

from __future__ import annotations

import csv
import json
import os
import re

# Reduce CUDA fragmentation (must be set before the first CUDA allocation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
from torch.utils.data import DataLoader

from model.model import ExplainableVidSalModel
from pre_process.collate import video_saliency_collate_fn
from pre_process.dataloader import DatasetLoader

import train as train_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Edit these directly (no argparse).
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20260907_031811/epoch_135.pth"
)
VAL_METRICS_CSV = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/val_metrics.csv"
)
CHECKPOINT_DIR = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20260907_182210"
)
CC_THRESHOLD = 0.54
SIM_THRESHOLD = 0.41
SKIP_EPOCHS = {63, 80, 106}
VAL_DATASET_DIR = train_cfg.VAL_DATASET_DIR
WINDOW_LEN = train_cfg.WINDOW_LEN
BATCH_SIZE = train_cfg._dataloader_batch_size()
NUM_WORKERS = train_cfg.NUM_WORKERS
SEED = train_cfg.SEED
OUTPUT_DIR = "evaluation_outputs"
BEST_RESULTS_CSV = os.path.join(OUTPUT_DIR, "best_results.csv")
SAVE_VAL_MAPS = False
MAP_SAVE_INTERVAL = train_cfg.MAP_SAVE_INTERVAL if SAVE_VAL_MAPS else 10**9
CHECKPOINT_COLUMN = "Checkpoint file dir (checkpoint_dir/epoch_{epoch_no}.pth)"
_EPOCH_PATH_RE = re.compile(r"epoch_(\d+)\.pth$")


def _resolve_devices() -> Tuple[torch.device, torch.device]:
    """Match the backbone/head split used in train.py."""
    if torch.cuda.is_available():
        backbone_device = torch.device("cuda:0")
        head_device = torch.device(
            "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        )
    else:
        backbone_device = torch.device("cpu")
        head_device = torch.device("cpu")
    return backbone_device, head_device


def build_model(
    backbone_device: torch.device,
    head_device: torch.device,
) -> ExplainableVidSalModel:
    """Match the architecture used in train.py."""
    return ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=True,
        freeze_backbone=train_cfg.FREEZE_BACKBONE,
        backbone_gradient_checkpointing=False,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=128,
        num_concepts=512,
        concept_hidden_dim=256,
        saliency_hidden_dim=256,
        top_k=8,
        max_source_patches=64,
        tau_pi=0.5,
        tau_alpha=0.07,
        tau_concept=0.07,
        concept_residual_weight=0.0,
        last_transition_only=True,
        use_rgb_refinement=False,
        use_feature_refinement=False,
        output_activation="none",
        return_details=True,
        use_subpatch_head=True,
        subpatch_factor=4,
        subpatch_residual_scale=0.5,
        use_temporal_transition_aggregation=True,
        temporal_aggregation_hidden_channels=128,
        temporal_aggregation_temperature=1.0,
        visual_concept_on=train_cfg.VISUAL_CONCEPT_ON,
        temporal_concepts_on=train_cfg.TEMPORAL_CONCEPTS_ON,
        visual_concept_logit_scale=train_cfg.VISUAL_CONCEPT_LOGIT_SCALE,
        visual_concept_residual_weight=1.0,
    ).to_split_devices(backbone_device, head_device)


def _filter_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str]]:
    """Keep checkpoint tensors whose keys and shapes match the live model."""
    model_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    shape_mismatch: List[str] = []
    unexpected: List[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatch.append(key)
            continue
        compatible[key] = value
    missing = [
        key
        for key in model_state
        if key not in compatible and key not in shape_mismatch
    ]
    return compatible, missing, unexpected, shape_mismatch


def load_checkpoint(
    model: ExplainableVidSalModel,
    checkpoint_path: str,
) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model_state_dict'."
        )

    compatible, missing, unexpected, shape_mismatch = _filter_compatible_state_dict(
        model,
        state_dict,
    )
    model.load_state_dict(compatible, strict=False)

    if shape_mismatch:
        preview = ", ".join(shape_mismatch[:5])
        suffix = " ..." if len(shape_mismatch) > 5 else ""
        print(
            f"Warning: skipped {len(shape_mismatch)} checkpoint tensors with "
            f"incompatible shapes (left at current init): {preview}{suffix}"
        )
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        print(
            f"Warning: {len(missing)} model parameters were not in the checkpoint "
            f"(left at current init): {preview}{suffix}"
        )
    if unexpected:
        print(
            f"Warning: ignored {len(unexpected)} checkpoint keys not used by the "
            "current model (likely from an older architecture variant)."
        )
    return checkpoint


def build_val_loader() -> Tuple[DataLoader, DatasetLoader]:
    val_dataset = DatasetLoader(
        VAL_DATASET_DIR,
        window_len=WINDOW_LEN,
        stride=1,
    )

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "collate_fn": video_saliency_collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return val_loader, val_dataset


def find_qualifying_epochs(
    val_metrics_csv: str,
    cc_threshold: float = CC_THRESHOLD,
    sim_threshold: float = SIM_THRESHOLD,
) -> List[int]:
    """Return epoch numbers where CC and SIM exceed thresholds.

    val_metrics.csv is written one row per epoch starting at epoch 1, so the
    first data row after the header maps to epoch_001.pth.
    """
    if not os.path.isfile(val_metrics_csv):
        raise FileNotFoundError(f"Validation metrics CSV not found: {val_metrics_csv}")

    qualifying_epochs: List[int] = []
    with open(val_metrics_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for epoch_no, row in enumerate(reader, start=1):
            cc = float(row["CC"])
            sim = float(row["SIM"])
            if cc > cc_threshold and sim > sim_threshold:
                qualifying_epochs.append(epoch_no)

    return qualifying_epochs


def checkpoint_path_for_epoch(checkpoint_dir: str, epoch_no: int) -> str:
    return os.path.join(checkpoint_dir, f"epoch_{epoch_no:03d}.pth")


@torch.no_grad()
def run_validation(
    model: ExplainableVidSalModel,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float]]:
    original_map_interval = train_cfg.MAP_SAVE_INTERVAL
    train_cfg.MAP_SAVE_INTERVAL = MAP_SAVE_INTERVAL
    try:
        val_loss, val_metrics = train_cfg.validate_one_epoch(
            model,
            val_loader,
            device,
            epoch=epoch,
            output_dir=OUTPUT_DIR,
        )
    finally:
        train_cfg.MAP_SAVE_INTERVAL = original_map_interval

    if val_metrics is None:
        raise RuntimeError(f"Validation metrics were not computed for epoch {epoch}.")
    return float(val_loss), val_metrics


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str = CHECKPOINT_PATH,
    save_json: bool = True,
) -> Dict[str, Any]:
    train_cfg.set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    backbone_device, head_device = _resolve_devices()
    device = head_device
    print(f"Backbone device: {backbone_device} | Head device: {head_device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = build_model(backbone_device, head_device)
    checkpoint = load_checkpoint(model, checkpoint_path)
    val_loader, val_dataset = build_val_loader()

    print(
        f"Val: {len(val_dataset)} windows ({VAL_DATASET_DIR}) | "
        f"window_len={WINDOW_LEN} | batch_size={BATCH_SIZE}"
    )

    val_loss, val_metrics = run_validation(
        model,
        val_loader,
        device,
        epoch=int(checkpoint.get("epoch", 0)),
    )

    results: Dict[str, Any] = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "val_dir": VAL_DATASET_DIR,
        "num_samples": len(val_dataset),
        "num_windows": len(val_dataset),
        "window_len": WINDOW_LEN,
        "batch_size": BATCH_SIZE,
        "backbone_device": str(backbone_device),
        "head_device": str(head_device),
        "mean_loss": float(val_loss),
        "metrics": val_metrics,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if save_json:
        results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results: {results_path}")

    print(f"\nEvaluation complete.")
    print(f"  Mean val loss: {val_loss:.6f}")
    print(
        "  Val metrics   | "
        f"CC: {val_metrics['CC']:.4f} | "
        f"SIM: {val_metrics['SIM']:.4f} | "
        f"NSS: {val_metrics['NSS']:.4f}"
    )

    return results


def epoch_from_checkpoint_path(checkpoint_path: str) -> int | None:
    match = _EPOCH_PATH_RE.search(checkpoint_path)
    if match is None:
        return None
    return int(match.group(1))


def load_existing_best_results(
    output_path: str = BEST_RESULTS_CSV,
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """Load prior evaluation rows and the epoch numbers already saved."""
    if not os.path.isfile(output_path):
        return [], set()

    results_rows: List[Dict[str, Any]] = []
    completed_epochs: Set[int] = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or CHECKPOINT_COLUMN not in reader.fieldnames:
            return [], set()

        for row in reader:
            checkpoint_path = row.get(CHECKPOINT_COLUMN, "").strip()
            if not checkpoint_path:
                continue
            epoch_no = epoch_from_checkpoint_path(checkpoint_path)
            row_dict = {
                "checkpoint_path": checkpoint_path,
                "epoch": epoch_no,
                "CC": float(row["CC"]),
                "SIM": float(row["SIM"]),
                "NSS": float(row["NSS"]),
            }
            results_rows.append(row_dict)
            if epoch_no is not None:
                completed_epochs.add(epoch_no)

    return results_rows, completed_epochs


def append_best_result_row(
    row: Dict[str, Any],
    output_path: str = BEST_RESULTS_CSV,
) -> None:
    """Append one evaluation row without modifying existing CSV contents."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_header = not os.path.isfile(output_path) or os.path.getsize(output_path) == 0
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[CHECKPOINT_COLUMN, "CC", "SIM", "NSS"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                CHECKPOINT_COLUMN: row["checkpoint_path"],
                "CC": row["CC"],
                "SIM": row["SIM"],
                "NSS": row["NSS"],
            }
        )


def evaluate_qualifying_checkpoints(
    val_metrics_csv: str = VAL_METRICS_CSV,
    checkpoint_dir: str = CHECKPOINT_DIR,
    cc_threshold: float = CC_THRESHOLD,
    sim_threshold: float = SIM_THRESHOLD,
    output_csv: str = BEST_RESULTS_CSV,
) -> List[Dict[str, Any]]:
    train_cfg.set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_rows, completed_epochs = load_existing_best_results(output_csv)
    if completed_epochs:
        print(
            f"Resuming from {output_csv}: skipping epochs already evaluated "
            f"({len(completed_epochs)}): {sorted(completed_epochs)}"
        )

    all_qualifying_epochs = find_qualifying_epochs(
        val_metrics_csv,
        cc_threshold=cc_threshold,
        sim_threshold=sim_threshold,
    )
    skipped_epochs = [epoch_no for epoch_no in all_qualifying_epochs if epoch_no in SKIP_EPOCHS]
    qualifying_epochs = [
        epoch_no
        for epoch_no in all_qualifying_epochs
        if epoch_no not in SKIP_EPOCHS and epoch_no not in completed_epochs
    ]
    print(
        f"Found {len(all_qualifying_epochs)} qualifying epochs in {val_metrics_csv} "
        f"(CC > {cc_threshold}, SIM > {sim_threshold}): {all_qualifying_epochs}"
    )
    if skipped_epochs:
        print(f"Skipping epochs: {skipped_epochs}")
    if not qualifying_epochs:
        if results_rows:
            print(f"All remaining checkpoints already evaluated. Results in: {output_csv}")
            return results_rows
        print("No qualifying checkpoints to evaluate.")
        return []

    backbone_device, head_device = _resolve_devices()
    device = head_device
    print(f"Backbone device: {backbone_device} | Head device: {head_device}")

    model = build_model(backbone_device, head_device)
    val_loader, val_dataset = build_val_loader()
    print(
        f"Val: {len(val_dataset)} windows ({VAL_DATASET_DIR}) | "
        f"window_len={WINDOW_LEN} | batch_size={BATCH_SIZE}"
    )

    results_rows: List[Dict[str, Any]] = list(results_rows)
    total_target = len(results_rows) + len(qualifying_epochs)
    for epoch_no in qualifying_epochs:
        checkpoint_path = checkpoint_path_for_epoch(checkpoint_dir, epoch_no)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Expected checkpoint for epoch {epoch_no} not found: {checkpoint_path}"
            )

        print(f"\nEvaluating epoch {epoch_no}: {checkpoint_path}")
        load_checkpoint(model, checkpoint_path)
        val_loss, val_metrics = run_validation(model, val_loader, device, epoch=epoch_no)

        row = {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "epoch": epoch_no,
            "CC": float(val_metrics["CC"]),
            "SIM": float(val_metrics["SIM"]),
            "NSS": float(val_metrics["NSS"]),
            "mean_loss": float(val_loss),
        }
        results_rows.append(row)
        print(
            f"  Mean val loss: {val_loss:.6f} | "
            f"CC: {row['CC']:.4f} | SIM: {row['SIM']:.4f} | NSS: {row['NSS']:.4f}"
        )
        append_best_result_row(row, output_csv)
        print(f"  Appended progress ({len(results_rows)}/{total_target}) to: {output_csv}")

    print(f"\nFinished evaluating {len(results_rows)} checkpoints.")
    print(f"Final results: {output_csv}")
    return results_rows


def main() -> None:
    evaluate_qualifying_checkpoints()


if __name__ == "__main__":
    main()
