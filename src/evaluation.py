"""
Evaluate a saved checkpoint on the validation set.

Edit the module-level constants below (especially CHECKPOINT_PATH) and run:

    python evaluation.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from model.model import ExplainableVidSalModel
from pre_process.collate import video_saliency_collate_fn
from pre_process.dataloader import DatasetLoader

import train as train_cfg

# ---------------------------------------------------------------------------
# Edit these directly (no argparse).
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20260710_173733/epoch_002.pth"
)
VAL_DATASET_DIR = train_cfg.VAL_DATASET_DIR
WINDOW_LEN = train_cfg.WINDOW_LEN
BATCH_SIZE = train_cfg._dataloader_batch_size()
NUM_WORKERS = train_cfg.NUM_WORKERS
SEED = train_cfg.SEED
OUTPUT_DIR = "evaluation_outputs"
SAVE_VAL_MAPS = False
MAP_SAVE_INTERVAL = train_cfg.MAP_SAVE_INTERVAL if SAVE_VAL_MAPS else 10**9


def build_model(device: torch.device) -> ExplainableVidSalModel:
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
        saliency_hidden_dim=96,
        top_k=5,
        max_source_patches=128,
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
        num_motion_concepts=128,
    ).to(device)


def load_checkpoint(
    model: ExplainableVidSalModel,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model_state_dict'."
        )

    model.load_state_dict(state_dict)
    return checkpoint


def build_val_loader() -> Tuple[DataLoader, DatasetLoader]:
    val_dataset = DatasetLoader(VAL_DATASET_DIR, window_len=WINDOW_LEN, stride=16)

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


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str = CHECKPOINT_PATH,
) -> Dict[str, Any]:
    train_cfg.set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = build_model(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    val_loader, val_dataset = build_val_loader()

    print(
        f"Val: {len(val_dataset)} windows ({VAL_DATASET_DIR}) | "
        f"window_len={WINDOW_LEN} | batch_size={BATCH_SIZE}"
    )

    # Reuse the validation loop from train.py.
    original_map_interval = train_cfg.MAP_SAVE_INTERVAL
    train_cfg.MAP_SAVE_INTERVAL = MAP_SAVE_INTERVAL
    try:
        val_loss, val_metrics = train_cfg.validate_one_epoch(
            model,
            val_loader,
            device,
            epoch=int(checkpoint.get("epoch", 0)),
            output_dir=OUTPUT_DIR,
        )
    finally:
        train_cfg.MAP_SAVE_INTERVAL = original_map_interval

    results: Dict[str, Any] = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "val_dir": VAL_DATASET_DIR,
        "num_samples": len(val_dataset),
        "num_windows": len(val_dataset),
        "window_len": WINDOW_LEN,
        "batch_size": BATCH_SIZE,
        "mean_loss": float(val_loss),
        "metrics": val_metrics or {},
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete.")
    print(f"  Mean val loss: {val_loss:.6f}")
    if val_metrics is not None:
        print(
            "  Val metrics   | "
            f"CC: {val_metrics['CC']:.4f} | "
            f"SIM: {val_metrics['SIM']:.4f} | "
            f"AUC: {val_metrics['AUC']:.4f} | "
            f"sAUC: {val_metrics['sAUC']:.4f} | "
            f"NSS: {val_metrics['NSS']:.4f}"
        )
    print(f"  Saved results: {results_path}")

    return results


def main() -> None:
    evaluate_checkpoint(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
