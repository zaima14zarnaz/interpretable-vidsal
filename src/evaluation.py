"""
Evaluate a saved checkpoint on the validation set.

Edit the module-level constants below (especially CHECKPOINT_PATH) and run:

    python evaluation.py
"""

from __future__ import annotations

import json
import os

# Reduce CUDA fragmentation (must be set before the first CUDA allocation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    # "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/empty_prot_ckpt.pth"
    # "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/random_prot_ckpt.pth"
    # "/data/quantization/zaima/videosal_datasets/dhf1k/proto_replaced/projection/projected_best_dhf1k.pth"
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/best_dhf1k.pth"
    # "/data/quantization/zaima/videosal_datasets/dhf1k/proto_replaced/projection/projected_best_dhf1k.pth"
)
VAL_DATASET_DIR = "/data/quantization/zaima/videosal_datasets/dhf1k/val"  # "/data/quantization/zaima/videosal_datasets/ucfsports/testing"
WINDOW_LEN = train_cfg.WINDOW_LEN
BATCH_SIZE = train_cfg._dataloader_batch_size()
NUM_WORKERS = train_cfg.NUM_WORKERS
SEED = train_cfg.SEED
OUTPUT_DIR = "evaluation_outputs"
EVALUATION_GPU_ID = 0  # Keep the original evaluation GPU selection explicit.
SAVE_VAL_MAPS = False
MAP_SAVE_INTERVAL = train_cfg.MAP_SAVE_INTERVAL if SAVE_VAL_MAPS else 10**9


def _resolve_devices() -> Tuple[torch.device, torch.device]:
    """Place the backbone and head together on the selected evaluation GPU."""
    gpu_id = EVALUATION_GPU_ID
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"EVALUATION_GPU_ID={gpu_id} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)."
            )
        device = torch.device(f"cuda:{gpu_id}")
        return device, device
    return torch.device("cpu"), torch.device("cpu")


def build_model(
    backbone_device: torch.device,
    head_device: torch.device,
    model_config: Optional[Dict[str, Any]] = None,
) -> ExplainableVidSalModel:
    """Use the checkpoint's architecture; fall back to train.py for older weights."""
    config = train_cfg.get_model_kwargs()
    if model_config is not None:
        if not isinstance(model_config, dict):
            raise TypeError("checkpoint['model_config'] must be a dictionary")
        config.update(model_config)
    # The full backbone weights are loaded from the evaluated checkpoint.
    config["pretrained_backbone"] = False
    config["backbone_gradient_checkpointing"] = False
    return train_cfg.build_model(
        backbone_device,
        head_device,
        model_kwargs=config,
    )


def load_checkpoint(
    model: ExplainableVidSalModel,
    checkpoint_path: str,
    *,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint is None:
        checkpoint = train_cfg._read_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model_state_dict'."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise RuntimeError(
            "Checkpoint is missing weights required by the current model: "
            f"{preview}{suffix}"
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
        stride=32,
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


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str = CHECKPOINT_PATH,
) -> Dict[str, Any]:
    train_cfg.set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    backbone_device, head_device = _resolve_devices()
    device = head_device
    print(f"Backbone device: {backbone_device} | Head device: {head_device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = train_cfg._read_checkpoint(checkpoint_path)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        print(
            "Checkpoint has no model_config; using train.py configuration "
            f"(top_k={train_cfg.MODEL_TOP_K}). For older checkpoints, set "
            "MODEL_TOP_K to the value used during their training."
        )
    model = build_model(backbone_device, head_device, model_config=model_config)
    load_checkpoint(model, checkpoint_path, checkpoint=checkpoint)
    print(f"Evaluating with top_k={model.concept_creations['stage4'].top_k}")
    val_loader, val_dataset = build_val_loader()

    print(
        f"Val: {len(val_dataset)} windows ({VAL_DATASET_DIR}) | "
        f"window_len={WINDOW_LEN} | batch_size={BATCH_SIZE}"
    )

    # Reuse train.py validation: CC/SIM/NSS with empty fixation maps excluded
    # from metric averaging (sum / valid_count, not sum / batch_size).
    original_map_interval = train_cfg.MAP_SAVE_INTERVAL
    train_cfg.MAP_SAVE_INTERVAL = MAP_SAVE_INTERVAL
    try:
        val_loss, val_metrics, _ = train_cfg.validate_one_epoch(
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
        "backbone_device": str(backbone_device),
        "head_device": str(head_device),
        "mean_loss": float(val_loss),
        "metrics": val_metrics or {},
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete.")
    print(f"  Mean val loss: {val_loss:.6f}")
    if val_metrics is not None:
        print(
            "  Val metrics   | "
            f"CC: {val_metrics['CC']:.4f} | "
            f"SIM: {val_metrics['SIM']:.4f} | "
            # f"AUC: {val_metrics['AUC']:.4f} | "
            # f"sAUC: {val_metrics['sAUC']:.4f} | "
            f"NSS: {val_metrics['NSS']:.4f}"
        )
    print(f"  Saved results: {results_path}")

    return results


def main() -> None:
    evaluate_checkpoint(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
