#!/usr/bin/env python3
"""Count stage-4 top-8 prototype activation frequency on DHF1K train and val.

Loads ``best_dhf1k.pth``, runs one eval pass over the train and val splits, and
tallies how often each visual prototype appears in each patch's top-8 active
list. Writes a CSV and a bar chart under ``--output-dir``.

Example:

    cd ExplainableSaliency/src
    python interpretability_exp/exp2a_visual_gallery/prototype_activation_freq.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluation as evaluation_cfg
import train as train_cfg
from pre_process.collate import video_saliency_collate_fn
from pre_process.dataloader import DatasetLoader

DEFAULT_CHECKPOINT = (
    "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/"
    "saved_weights/best_dhf1k.pth"
)
DEFAULT_TRAIN_DIR = "/data/quantization/zaima/videosal_datasets/dhf1k/train"
DEFAULT_VAL_DIR = "/data/quantization/zaima/videosal_datasets/dhf1k/val"
DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parent / "prototype_activation_freq_outputs")
TOP_K = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-len", type=int, default=train_cfg.WINDOW_LEN)
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Sliding-window stride for both splits (default: 32, matching val eval).",
    )
    parser.add_argument("--batch-size", type=int, default=train_cfg._dataloader_batch_size())
    parser.add_argument("--num-workers", type=int, default=train_cfg.NUM_WORKERS)
    parser.add_argument("--device", default=None, help="e.g. cuda:0; default uses evaluation.py GPU setup.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=train_cfg.SEED)
    return parser.parse_args()


def build_loader(dataset_dir: str, *, window_len: int, stride: int, batch_size: int, num_workers: int) -> tuple[DataLoader, DatasetLoader]:
    dataset = DatasetLoader(
        dataset_dir,
        window_len=window_len,
        stride=stride,
        random_train_sampling=False,
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": video_saliency_collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return loader, dataset


def extract_topk_indices(model_out: dict, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flat valid top-k prototype indices for every patch in the batch."""
    concept_out = model_out.get("concept_out")
    if not isinstance(concept_out, dict) or "stage4" not in concept_out:
        raise ValueError("Model output is missing concept_out['stage4'].")
    stage = concept_out["stage4"]
    indices = stage.get("active_visual_prototype_indices")
    valid = stage.get("visual_validity_mask")
    if not torch.is_tensor(indices) or not torch.is_tensor(valid):
        raise ValueError("stage4 must export active_visual_prototype_indices and visual_validity_mask.")
    if valid.dtype != torch.bool:
        raise ValueError("visual_validity_mask must be bool.")
    if indices.shape != valid.shape:
        raise ValueError(f"Index/mask shape mismatch: {tuple(indices.shape)} vs {tuple(valid.shape)}")

    if indices.dim() == 5:
        # [B, T, H, W, K] — flatten patches, keep K.
        b, t, h, w, k = indices.shape
        indices = indices.reshape(b * t * h * w, k)
        valid = valid.reshape(b * t * h * w, k)
    elif indices.dim() == 2:
        pass
    else:
        raise ValueError(f"Unexpected active index rank {indices.dim()}; expected 2 or 5.")

    if indices.shape[1] < top_k:
        raise ValueError(f"Model exported K={indices.shape[1]} but --top-k={top_k}.")
    indices = indices[:, :top_k].detach().long().cpu()
    valid = valid[:, :top_k].detach().cpu()
    return indices, valid


@torch.inference_mode()
def count_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    split_name: str,
    top_k: int,
    num_prototypes: int,
) -> dict:
    counts = torch.zeros(num_prototypes, dtype=torch.long)
    patch_count = 0
    window_count = 0
    assignment_slots = 0

    pbar = tqdm(loader, desc=f"Eval {split_name}", leave=True)
    for (
        _video_filenames,
        rgb_batch,
        sal_batch,
        fix_batch,
        _n_frames,
        _valid_mask,
    ) in pbar:
        rgb_batch, sal_batch, fix_batch = model.prepare_training_batch(
            rgb_batch, sal_batch, fix_batch
        )
        with torch.autocast(
            device.type,
            dtype=train_cfg._amp_dtype(device),
            enabled=train_cfg._amp_enabled(device),
        ):
            model_out = model(
                rgb_batch,
                saliency_maps=sal_batch,
                return_details=True,
                return_concept_losses=False,
                return_decoder_diagnostics=False,
            )
        indices, valid = extract_topk_indices(model_out, top_k)
        chosen = indices[valid]
        if chosen.numel():
            if int(chosen.min()) < 0 or int(chosen.max()) >= num_prototypes:
                raise ValueError(
                    f"Prototype index out of range on {split_name}: "
                    f"[{int(chosen.min())}, {int(chosen.max())}] vs bank size {num_prototypes}"
                )
            counts += torch.bincount(chosen, minlength=num_prototypes)
        patch_count += int(indices.shape[0])
        assignment_slots += int(valid.sum())
        window_count += int(rgb_batch.shape[0])
        pbar.set_postfix(patches=patch_count, windows=window_count)

    total = int(counts.sum())
    frequencies = (counts.float() / total).tolist() if total else [0.0] * num_prototypes
    return {
        "split": split_name,
        "window_count": window_count,
        "patch_count": patch_count,
        "assignment_slots": assignment_slots,
        "top_k": top_k,
        "num_prototypes": num_prototypes,
        "counts": counts.tolist(),
        "frequencies": frequencies,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    num_prototypes = results[0]["num_prototypes"]
    fieldnames = ["prototype_index"]
    for result in results:
        name = result["split"]
        fieldnames.extend([f"{name}_count", f"{name}_frequency"])
    fieldnames.extend(["combined_count", "combined_frequency"])

    combined = [0] * num_prototypes
    for result in results:
        for index, count in enumerate(result["counts"]):
            combined[index] += count
    combined_total = sum(combined) or 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prototype_index in range(num_prototypes):
            row = {"prototype_index": prototype_index}
            for result in results:
                name = result["split"]
                row[f"{name}_count"] = result["counts"][prototype_index]
                row[f"{name}_frequency"] = f"{result['frequencies'][prototype_index]:.8f}"
            row["combined_count"] = combined[prototype_index]
            row["combined_frequency"] = f"{combined[prototype_index] / combined_total:.8f}"
            writer.writerow(row)


def write_plot(path: Path, results: list[dict], top_k: int) -> None:
    num_prototypes = results[0]["num_prototypes"]
    x = list(range(num_prototypes))
    if not results:
        raise ValueError("No results to plot.")
    fig, axes = plt.subplots(
        len(results) + 1,
        1,
        figsize=(14, 3.2 * (len(results) + 1)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)

    combined = [0] * num_prototypes
    for result, axis in zip(results, axes):
        axis.bar(x, result["counts"], width=1.0, color="#146d68", edgecolor="none")
        axis.set_ylabel("Count")
        axis.set_title(
            f"{result['split']} · top-{top_k} activations · "
            f"{result['patch_count']} patches · {result['window_count']} windows"
        )
        for index, count in enumerate(result["counts"]):
            combined[index] += count

    axes[-1].bar(x, combined, width=1.0, color="#182d3c", edgecolor="none")
    axes[-1].set_ylabel("Count")
    axes[-1].set_xlabel("Prototype index")
    axes[-1].set_title("Combined train + val")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive.")
    if args.window_len < 1 or args.stride < 1 or args.batch_size < 1:
        raise SystemExit("window-len, stride, and batch-size must be positive.")

    train_cfg.set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
        backbone_device = head_device = device
    else:
        backbone_device, head_device = evaluation_cfg._resolve_devices()
        device = head_device

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    model = evaluation_cfg.build_model(backbone_device, head_device)
    checkpoint = evaluation_cfg.load_checkpoint(model, args.checkpoint)
    model.eval()

    if "stage4" not in model.concept_creations:
        raise RuntimeError("Loaded model has no stage4 concept module.")
    concept = model.concept_creations["stage4"]
    num_prototypes = int(concept.visual_concepts.shape[0])
    model_top_k = int(getattr(concept, "top_k", args.top_k))
    if model_top_k < args.top_k:
        raise SystemExit(f"Model top_k={model_top_k} is smaller than requested --top-k={args.top_k}.")
    print(f"Prototype bank size: {num_prototypes} | model top_k: {model_top_k} | counting top-{args.top_k}")

    splits = (
        ("train", args.train_dir),
        ("val", args.val_dir),
    )
    results = []
    for split_name, dataset_dir in splits:
        loader, dataset = build_loader(
            dataset_dir,
            window_len=args.window_len,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            f"{split_name}: {len(dataset)} windows from {dataset_dir} "
            f"(window_len={args.window_len}, stride={args.stride})"
        )
        result = count_split(
            model,
            loader,
            device,
            split_name=split_name,
            top_k=args.top_k,
            num_prototypes=num_prototypes,
        )
        results.append(result)
        top = Counter({i: c for i, c in enumerate(result["counts"]) if c}).most_common(5)
        print(
            f"  patches={result['patch_count']} assignment_slots={result['assignment_slots']} "
            f"top prototypes={top}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"prototype_activation_freq_{stamp}.csv"
    plot_path = output_dir / f"prototype_activation_freq_{stamp}.png"
    summary_path = output_dir / f"prototype_activation_freq_{stamp}.json"
    write_csv(csv_path, results)
    write_plot(plot_path, results, args.top_k)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "window_len": args.window_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "num_prototypes": num_prototypes,
        "device": str(device),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "splits": [
            {
                "split": result["split"],
                "window_count": result["window_count"],
                "patch_count": result["patch_count"],
                "assignment_slots": result["assignment_slots"],
            }
            for result in results
        ],
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote meta: {summary_path}")


if __name__ == "__main__":
    main()
