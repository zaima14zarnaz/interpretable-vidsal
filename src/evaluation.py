"""
Evaluate a trained ExplainableVidSalModel on the UCF-style test split.

Loads a checkpoint, runs inference without GT in the forward pass, and reports
saliency metrics on the last frame. CC/SIM use continuous saliency density maps;
AUC, sAUC, and NSS use binary fixation maps from ``fix_batch``.
"""

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
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
DEFAULT_CHECKPOINT = "/home/zaimaz/Desktop/research1/ExplainableVidSal/src/training_outputs/ckpts/20260616_014755/epoch_001.pth"
DEFAULT_OUTPUT_DIR = "evaluation_outputs"
MAX_SAMPLES = 10000000

WINDOW_LEN = 16
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


def _infer_subpatch_factor_from_state(state: dict) -> int:
    """Infer subpatch_factor from the last Conv2d in subpatch_saliency_head."""
    key = (
        "saliency_prediction.stage_predictors.stage1"
        ".subpatch_saliency_head.4.weight"
    )
    weight = state.get(key)
    if weight is None:
        return 16
    out_ch = int(weight.shape[0])
    factor = int(round(out_ch**0.5))
    if factor * factor != out_ch:
        raise ValueError(
            f"Cannot infer subpatch_factor from subpatch head out_channels={out_ch}"
        )
    return factor


def build_model(
    device: torch.device,
    subpatch_factor: int = 16,
) -> ExplainableVidSalModel:
    """Same architecture / hyperparameters as train.py."""
    model = ExplainableVidSalModel(
        backbone_stages=("stage1", "stage2", "stage3", "stage4"),
        pretrained_backbone=True,
        freeze_backbone=False,
        input_format="BTCHW",
        resize_to=(224, 384),
        concept_dim=256,
        num_concepts=1024,
        concept_hidden_dim=256,
        saliency_hidden_dim=256,
        top_k=1,
        max_source_patches=64,
        tau_pi=0.5,
        tau_alpha=0.07,
        tau_concept=0.2,
        concept_residual_weight=0.0,
        last_transition_only=True,
        use_rgb_refinement=False,
        use_feature_refinement=False,
        output_activation="sigmoid",
        return_details=True,
        use_subpatch_head=False,
        subpatch_factor=4,
        subpatch_residual_scale=0.5,
        use_temporal_transition_aggregation=True,
        temporal_aggregation_hidden_channels=128,
        temporal_aggregation_temperature=1.0,
    ).to(device)
    return model


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


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ExplainableVidSalModel, Dict]:
    """Build model with subpatch_factor matching the checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    subpatch_factor = _infer_subpatch_factor_from_state(state)
    model = build_model(device, subpatch_factor=subpatch_factor)
    model.load_state_dict(state)
    return model, checkpoint


def _make_test_loader(
    dataset_dir: str,
    batch_size: int,
    num_workers: int,
    stride: int = 16,
) -> DataLoader:
    dataset = DatasetLoader(dataset_dir, window_len=WINDOW_LEN, stride=stride)
    n_samples = min(MAX_SAMPLES, len(dataset))
    dataset = Subset(dataset, range(n_samples))
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


def _prepare_last_map_for_visualization(batch: torch.Tensor) -> torch.Tensor:
    """First sample, last temporal frame -> [H, W] in [0, 1] for display."""
    x = batch.detach().float()
    if x.dim() == 3:
        x = x[0]
    elif x.dim() == 4:
        x = x[0, -1] if x.shape[1] != 1 else x[0, 0]
    elif x.dim() == 5:
        if x.shape[1] == 1:
            x = x[0, 0, -1]
        elif x.shape[2] == 1:
            x = x[0, -1, 0]
        else:
            raise ValueError(f"Unsupported 5D map shape {tuple(batch.shape)}")
    else:
        raise ValueError(f"Unsupported map shape {tuple(batch.shape)}")

    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    x = x - x.min()
    return x / (x.max() + 1e-8)


def _prepare_pred_for_visualization(pred_saliency: torch.Tensor) -> torch.Tensor:
    x = pred_saliency.detach().float()
    if x.dim() == 4:
        x = x[0, 0] if x.shape[1] == 1 else x[0]
    elif x.dim() == 3:
        x = x[0]
    else:
        raise ValueError(f"Unsupported pred shape {tuple(pred_saliency.shape)}")
    x = x - x.min()
    return x / (x.max() + 1e-8)


def _prepare_last_rgb_for_visualization(rgb_batch: torch.Tensor) -> torch.Tensor:
    x = rgb_batch.detach().float()
    if x.dim() == 5 and x.shape[-1] == 3:
        x = x[0, -1]
    elif x.dim() == 4 and x.shape[-1] == 3:
        x = x[0]
    else:
        raise ValueError(f"Unsupported rgb shape {tuple(rgb_batch.shape)}")
    if x.numel() > 0 and x.max() > 2.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def _save_map_png(map_tensor: torch.Tensor, save_path: str) -> None:
    arr = map_tensor.detach().cpu().numpy()
    plt.imsave(save_path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def _save_rgb_png(rgb_tensor: torch.Tensor, save_path: str) -> None:
    arr = rgb_tensor.detach().cpu().numpy()
    plt.imsave(save_path, arr)


def _save_eval_maps(
    model_out: dict,
    sal_batch: torch.Tensor,
    fix_batch: torch.Tensor,
    rgb_batch: torch.Tensor,
    video_filenames,
    maps_dir: str,
    sample_offset: int,
) -> int:
    batch_size = rgb_batch.shape[0]
    for i in range(batch_size):
        vis_dir = os.path.join(maps_dir, f"sample_{sample_offset + i:06d}")
        os.makedirs(vis_dir, exist_ok=True)

        gt_sal = _prepare_last_map_for_visualization(sal_batch[i : i + 1])
        gt_fix = _prepare_last_map_for_visualization(fix_batch[i : i + 1])
        pred = _prepare_pred_for_visualization(model_out["saliency_map"][i : i + 1])
        rgb_frame = _prepare_last_rgb_for_visualization(rgb_batch[i : i + 1])

        _save_map_png(gt_sal, os.path.join(vis_dir, "gt_sal_map.png"))
        _save_map_png(gt_fix, os.path.join(vis_dir, "gt_fixation_map.png"))
        _save_map_png(pred, os.path.join(vis_dir, "pred_sal_map.png"))
        _save_rgb_png(rgb_frame, os.path.join(vis_dir, "rgb_frame.png"))

        with open(os.path.join(vis_dir, "video.txt"), "w", encoding="utf-8") as f:
            f.write(str(video_filenames[i]))

    return sample_offset + batch_size


def _compute_batch_loss(model_out: dict, sal_batch: torch.Tensor) -> dict:
    return compute_total_loss(model_out, sal_batch, **LOSS_LAMBDA)


def _compute_batch_metrics(
    model_out: dict,
    sal_batch: torch.Tensor,
    fix_batch: torch.Tensor,
    allow_pseudo_fixations: bool = False,
) -> Dict[str, torch.Tensor]:
    return compute_saliency_metrics(
        model_out["saliency_map"],
        target_density=sal_batch,
        fixation_target=fix_batch,
        fixation_threshold=0.5,
        top_percent=None,
        allow_pseudo_fixations=allow_pseudo_fixations,
    )


@torch.no_grad()
def evaluate(
    model: ExplainableVidSalModel,
    loader: DataLoader,
    device: torch.device,
    maps_dir: Optional[str] = None,
    allow_pseudo_fixations: bool = False,
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
    sample_offset = 0
    if maps_dir is not None:
        os.makedirs(maps_dir, exist_ok=True)

    pbar = tqdm(loader, desc="Test evaluation", leave=True)
    for batch_idx, (
        video_filenames,
        rgb_batch,
        sal_batch,
        fix_batch,
        n_frames,
        valid_mask,
    ) in enumerate(pbar):
        rgb_batch = rgb_batch.to(device, non_blocking=True)
        if not torch.is_tensor(sal_batch):
            raise ValueError("sal_batch must be a torch.Tensor.")
        sal_batch = sal_batch.to(device, non_blocking=True)
        fix_batch = fix_batch.to(device, non_blocking=True)
        fix_batch = (fix_batch > 0).float()

        if batch_idx == 0:
            target_fix_last = fix_batch[:, -1:].float()
            print(
                "Eval DEBUG fixation pixels per sample:",
                target_fix_last.flatten(1).sum(dim=1).detach().cpu().tolist(),
            )

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

        metric_dict = _compute_batch_metrics(
            model_out,
            sal_batch,
            fix_batch,
            allow_pseudo_fixations=allow_pseudo_fixations,
        )
        metric_averager.update(metric_dict, batch_size=rgb_batch.shape[0])
        pbar.set_postfix(NSS=f"{metric_averager.mean()['NSS']:.4f}")

        if maps_dir is not None:
            sample_offset = _save_eval_maps(
                model_out,
                sal_batch,
                fix_batch,
                rgb_batch,
                video_filenames,
                maps_dir,
                sample_offset,
            )

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
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Save predicted saliency, GT density, GT fixation, and RGB frames.",
    )
    parser.add_argument(
        "--maps-dir",
        type=str,
        default=None,
        help="Directory for saved evaluation maps (default: <output-dir>/maps).",
    )
    parser.add_argument(
        "--allow-pseudo-fixations",
        action="store_true",
        help="Use saliency-derived pseudo-fixations for AUC/sAUC/NSS (debug only).",
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

    model, checkpoint_meta = build_model_from_checkpoint(args.checkpoint, device)
    model.optimize_for_inference()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"subpatch_factor: {int(model.saliency_prediction.stage_predictors['stage1'].subpatch_factor)}")

    maps_dir = None
    if args.save_maps:
        maps_dir = args.maps_dir or os.path.join(args.output_dir, "maps")
        print(f"Saving evaluation maps to: {maps_dir}")

    if args.allow_pseudo_fixations:
        print(
            "WARNING: --allow-pseudo-fixations enabled; "
            "AUC/sAUC/NSS are not true fixation-based metrics."
        )

    mean_loss, metrics, num_samples = evaluate(
        model,
        test_loader,
        device,
        maps_dir=maps_dir,
        allow_pseudo_fixations=args.allow_pseudo_fixations,
    )
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
