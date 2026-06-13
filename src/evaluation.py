"""
Evaluate a trained ExplainableVidSalModel on the UCF-style test split.

Loads a checkpoint, runs inference without GT in the forward pass, and reports
loss (fidelity + any concept regularizers present) and saliency metrics (CC, SIM,
AUC, sAUC, NSS) on the last frame.
"""

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm

from model.losses import compute_total_loss
from model.metrics import MetricAverager, compute_saliency_metrics
from model.model import ExplainableVidSalModel
from pre_process.collate import video_saliency_collate_fn
from pre_process.dataloader import DatasetLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Defaults aligned with train.py
DEFAULT_TRAIN_DIR = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/training"
)
DEFAULT_TEST_DIR = (
    "/data/research/zaima/dataset/Dataset/VideoSaliencyDatasets/dh1k/testing"
)
DEFAULT_CHECKPOINT = "/home/zaimaz/Desktop/research1/ExplainableVidSal/src/training_outputs/ckpts/20260611_035436/epoch_003.pth"
DEFAULT_OUTPUT_DIR = "evaluation_outputs"
MAX_SAMPLES = 100

WINDOW_LEN = 32
BATCH_SIZE = 4
NUM_WORKERS = 4

FIXATION_THRESHOLD = 0.5
TOP_PERCENT = 0.05

LOSS_LAMBDA = {
    "lambda_delta": 0.0,
    "lambda_dense": 0.5,
    "lambda_bce": 0.0,
    "lambda_kl": 10.0,
    "lambda_align": 0.0,
    "lambda_sparse": 0.0,
    "lambda_div": 0.0,
    "lambda_gate": 0.0,
    "patch_from_logits": True,
}


def build_model(device: torch.device) -> ExplainableVidSalModel:
    """Same architecture / hyperparameters as train.py."""
    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=True,
        freeze_backbone=True,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=256,
        num_concepts=1024,
        concept_hidden_dim=256,
        saliency_hidden_dim=256,
        top_k=3,
        max_source_patches=64,
        tau_pi=0.5,
        tau_alpha=0.07,
        tau_concept=0.2,
        concept_residual_weight=0.0,
        use_rgb_refinement=False,   
        use_feature_refinement=False,
        output_activation="sigmoid",
        return_details=True,
    )
    return model.to(device)


def load_checkpoint(
    model: ExplainableVidSalModel,
    checkpoint_path: str,
    device: torch.device,
) -> Dict:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return checkpoint


def _make_test_loader(
    dataset_dir: str,
    batch_size: int,
    num_workers: int,
    stride: int = 16,
) -> DataLoader:
    dataset = DatasetLoader(dataset_dir, window_len=WINDOW_LEN, stride=stride)
    # n_samples = min(MAX_SAMPLES, len(full_dataset))
    # dataset = Subset(full_dataset, range(n_samples))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": video_saliency_collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


def _compute_batch_loss(model_out: dict, sal_batch: torch.Tensor) -> dict:
    return compute_total_loss(model_out, sal_batch, **LOSS_LAMBDA)


def _compute_batch_metrics(
    model_out: dict, sal_batch: torch.Tensor
) -> Dict[str, torch.Tensor]:
    return compute_saliency_metrics(
        model_out["saliency_map"],
        sal_batch,
        fixation_target=None,
        fixation_threshold=FIXATION_THRESHOLD,
        top_percent=TOP_PERCENT,
    )


@torch.no_grad()
def evaluate(
    model: ExplainableVidSalModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float], int]:
    """
    Run test evaluation.

    Forward does not pass GT saliency into the model (inference-only).
    GT is used only for metrics and external loss computation.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    num_samples = 0
    metric_averager = MetricAverager()

    pbar = tqdm(loader, desc="Test evaluation", leave=True)
    for video_filenames, rgb_batch, sal_batch, n_frames, valid_mask in pbar:
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        if not torch.is_tensor(sal_batch):
            raise ValueError("sal_batch must be a torch.Tensor.")
        sal_batch = sal_batch.to(device, non_blocking=True)

        model_out = model(
            rgb_batch,
            saliency_maps=None,
            return_details=True,
        )

        # loss_dict = _compute_batch_loss(model_out, sal_batch)
        # batch_loss = loss_dict["loss_total"].item()
        # running_loss += batch_loss
        num_batches += 1
        num_samples += rgb_batch.shape[0]

        metric_dict = _compute_batch_metrics(model_out, sal_batch)
        metric_averager.update(metric_dict, batch_size=rgb_batch.shape[0])

    mean_loss = running_loss / max(num_batches, 1)
    return mean_loss, metric_averager.mean(), num_samples


def _print_results(
    mean_loss: float,
    metrics: Dict[str, float],
    num_samples: int,
    checkpoint_path: str,
    test_dir: str,
    checkpoint_meta: Optional[Dict] = None,
) -> None:
    print("\n=== Test evaluation ===")
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  Test data:    {test_dir}")
    print(f"  Samples:      {num_samples}")
    if checkpoint_meta is not None and "epoch" in checkpoint_meta:
        print(f"  Checkpoint epoch: {checkpoint_meta['epoch']}")
    print(f"  Mean loss:    {mean_loss:.6f}")
    print(
        f"  CC:   {metrics['CC']:.4f} | SIM: {metrics['SIM']:.4f} | "
        f"AUC: {metrics['AUC']:.4f} | sAUC: {metrics['sAUC']:.4f} | "
        f"NSS: {metrics['NSS']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ExplainableVidSal on the test set."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=DEFAULT_TEST_DIR,
        help="Path to UCF-style test split (videos with images/ and maps/).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to .pth checkpoint (best_checkpoint.pth or last_checkpoint.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save evaluation_results.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=DEFAULT_TRAIN_DIR,
        help="Training dir used only to derive default test dir if --test-dir omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    test_dir = args.test_dir
    if test_dir == DEFAULT_TEST_DIR and not os.path.isdir(test_dir):
        derived = args.train_dir.replace("/training", "/testing")
        if os.path.isdir(derived):
            test_dir = derived

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test dataset directory not found: {test_dir}\n"
            "Pass --test-dir to your UCF testing split."
        )

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Test directory: {test_dir}")

    test_loader = _make_test_loader(
        test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Test windows: {len(test_loader.dataset)} | batches: {len(test_loader)}")

    model = build_model(device)
    checkpoint_meta = load_checkpoint(model, args.checkpoint, device)
    model.optimize_for_inference()
    print(f"Loaded checkpoint: {args.checkpoint}")

    mean_loss, metrics, num_samples = evaluate(model, test_loader, device)
    _print_results(
        mean_loss,
        metrics,
        num_samples,
        args.checkpoint,
        test_dir,
        checkpoint_meta,
    )

    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "test_dir": os.path.abspath(test_dir),
        "num_samples": num_samples,
        "num_windows": len(test_loader.dataset),
        "window_len": WINDOW_LEN,
        "batch_size": args.batch_size,
        "mean_loss": mean_loss,
        "metrics": metrics,
    }
    if isinstance(checkpoint_meta, dict) and "epoch" in checkpoint_meta:
        results["checkpoint_epoch"] = checkpoint_meta["epoch"]

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
