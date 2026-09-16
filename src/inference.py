"""
Run checkpoint inference on DHF1K test frames and save per-frame saliency maps.

Uses the same model construction / checkpoint loading as evaluation.py.

Default layout:
    Input:  /data/quantization/zaima/dhf1k/test/<video>/images/<frame>.png
    Output: /data/quantization/zaima/dhf1k/test_inference/ProtoVSal/<video>/<frame>.png

Run:
    python inference.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import train as train_cfg
from evaluation import CHECKPOINT_PATH, build_model, load_checkpoint
from model.model import ExplainableVidSalModel
from pre_process.collate import _rgb_clip_to_tchw, _supervision_map_to_hw

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Edit these directly.
# ---------------------------------------------------------------------------
TEST_DATASET_DIR = "/data/quantization/zaima/dhf1k/test"
OUTPUT_ROOT = "/data/quantization/zaima/dhf1k/test_inference/ProtoVSal"
CHECKPOINT = CHECKPOINT_PATH
WINDOW_LEN = train_cfg.WINDOW_LEN
STRIDE = 1
BATCH_SIZE = train_cfg._dataloader_batch_size()
NUM_WORKERS = train_cfg.NUM_WORKERS
SEED = train_cfg.SEED
SKIP_EXISTING = True
# Use a single GPU for the full model (backbone + head on the same device).
GPU_ID = 0


def _resolve_single_device(gpu_id: int = GPU_ID) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"GPU_ID={gpu_id} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)."
            )
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def _natural_key(filename: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


def _list_image_names(images_dir: str) -> List[str]:
    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    image_files.sort(key=_natural_key)
    return image_files


class TestFrameInferenceDataset(Dataset):
    """
    One sample per target frame (stride=1).

    Each window ends at the target frame. Early frames are padded by repeating
    the first available frame so every frame receives a prediction.
    """

    IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        dataset_dir: str,
        window_len: int,
        stride: int = 1,
        output_root: str | None = None,
        skip_existing: bool = False,
    ):
        if window_len < 1:
            raise ValueError(f"window_len must be >= 1, got {window_len}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.dataset_dir = os.path.abspath(dataset_dir)
        self.window_len = window_len
        self.stride = stride
        self.output_root = os.path.abspath(output_root) if output_root else None
        self.skip_existing = skip_existing

        self.video_frame_names: Dict[str, List[str]] = {}
        self.windows: List[Tuple[str, int, str]] = []
        self.num_total_frames = 0
        self.num_skipped_existing = 0

        for name in sorted(os.listdir(self.dataset_dir)):
            video_path = os.path.join(self.dataset_dir, name)
            if not os.path.isdir(video_path):
                continue

            images_dir = os.path.join(video_path, "images")
            if not os.path.isdir(images_dir):
                continue

            frame_names = _list_image_names(images_dir)
            if not frame_names:
                continue

            self.video_frame_names[name] = frame_names
            for end_idx in range(0, len(frame_names), stride):
                self.num_total_frames += 1
                if self._should_skip(name, frame_names[end_idx]):
                    self.num_skipped_existing += 1
                    continue
                self.windows.append((name, end_idx, frame_names[end_idx]))

        if self.num_total_frames == 0:
            raise RuntimeError(
                f"No inference frames found under {self.dataset_dir}. "
                "Expected subdirs like '<video>/images/*.png'."
            )

    def _should_skip(self, video_name: str, frame_name: str) -> bool:
        if not self.skip_existing or self.output_root is None:
            return False
        save_path = os.path.join(self.output_root, video_name, frame_name)
        return os.path.isfile(save_path)

    @staticmethod
    def _load_rgb_pil(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB").copy()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        video_filename, end_idx, frame_name = self.windows[idx]
        frame_names = self.video_frame_names[video_filename]
        images_dir = os.path.join(self.dataset_dir, video_filename, "images")

        start_idx = max(0, end_idx - self.window_len + 1)
        window_frames = frame_names[start_idx : end_idx + 1]
        if len(window_frames) < self.window_len:
            pad_count = self.window_len - len(window_frames)
            window_frames = [window_frames[0]] * pad_count + window_frames

        rgb_imgs = [
            self._load_rgb_pil(os.path.join(images_dir, fname))
            for fname in window_frames
        ]
        rgb_frames = [np.asarray(img, dtype=np.uint8) for img in rgb_imgs]
        rgb_frame_set = np.stack(rgb_frames, axis=0)

        h, w = rgb_frame_set.shape[1:3]
        dummy_map = np.zeros((h, w), dtype=np.float32)

        return (
            video_filename,
            frame_name,
            rgb_frame_set,
            dummy_map,
            dummy_map,
            rgb_frame_set.shape[0],
        )


def inference_collate_fn(batch):
    (
        video_filenames,
        frame_names,
        rgb_list,
        sal_list,
        fix_list,
        n_frames_list,
    ) = zip(*batch)

    b = len(batch)
    t_max = max(int(n) for n in n_frames_list)
    h_max = max(_rgb_clip_to_tchw(rgb).shape[2] for rgb in rgb_list)
    w_max = max(_rgb_clip_to_tchw(rgb).shape[3] for rgb in rgb_list)

    rgb_batch = torch.zeros(b, t_max, 3, h_max, w_max, dtype=torch.float32)
    sal_batch = torch.zeros(b, 1, h_max, w_max, dtype=torch.float32)
    fix_batch = torch.zeros(b, 1, h_max, w_max, dtype=torch.float32)
    n_frames = torch.tensor(n_frames_list, dtype=torch.int64)

    for i, (rgb, sal, fix, n) in enumerate(
        zip(rgb_list, sal_list, fix_list, n_frames_list)
    ):
        n = int(n)
        rgb = _rgb_clip_to_tchw(rgb)
        _, _, h, w = rgb.shape
        rgb_batch[i, :n, :, :h, :w] = rgb[:n]

        sal_map = _supervision_map_to_hw(sal)
        fix_map = _supervision_map_to_hw(fix)
        sh, sw = sal_map.shape
        sal_batch[i, 0, :sh, :sw] = sal_map
        fix_batch[i, 0, :sh, :sw] = fix_map

    return (
        list(video_filenames),
        list(frame_names),
        rgb_batch,
        sal_batch,
        fix_batch,
        n_frames,
    )


def _normalize_prediction(pred_hw: torch.Tensor) -> torch.Tensor:
    """Map a single [H, W] prediction to [0, 1] for saving."""
    x = pred_hw.detach().float()
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def _save_saliency_png(pred_hw: torch.Tensor, save_path: str) -> None:
    arr = (_normalize_prediction(pred_hw).cpu().numpy() * 255.0).clip(0, 255)
    arr_u8 = arr.astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(arr_u8, mode="L").save(save_path)


def build_test_loader(
    output_root: str = OUTPUT_ROOT,
) -> Tuple[DataLoader | None, TestFrameInferenceDataset]:
    dataset = TestFrameInferenceDataset(
        TEST_DATASET_DIR,
        window_len=WINDOW_LEN,
        stride=STRIDE,
        output_root=output_root,
        skip_existing=SKIP_EXISTING,
    )

    if not dataset.windows:
        return None, dataset

    loader_kwargs: Dict[str, Any] = {
        "num_workers": NUM_WORKERS,
        "collate_fn": inference_collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    return loader, dataset


@torch.no_grad()
def run_inference(
    checkpoint_path: str = CHECKPOINT,
    output_root: str = OUTPUT_ROOT,
) -> Dict[str, Any]:
    train_cfg.set_seed(SEED)
    os.makedirs(output_root, exist_ok=True)

    device = _resolve_single_device(GPU_ID)
    print(f"Using single device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = build_model(device, device)
    checkpoint = load_checkpoint(model, checkpoint_path)
    model.eval()

    loader, dataset = build_test_loader(output_root=output_root)
    print(
        f"Test inference: {dataset.num_total_frames} total frames | "
        f"{dataset.num_skipped_existing} already saved | "
        f"{len(dataset)} remaining | "
        f"dir={TEST_DATASET_DIR} | window_len={WINDOW_LEN} | "
        f"stride={STRIDE} | batch_size={BATCH_SIZE}"
    )
    print(f"Saving maps to: {output_root}")

    if loader is None:
        summary = {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "test_dir": TEST_DATASET_DIR,
            "output_root": str(Path(output_root).resolve()),
            "device": str(device),
            "num_frames_total": dataset.num_total_frames,
            "num_frames_skipped_existing": dataset.num_skipped_existing,
            "num_frames_scheduled": 0,
            "num_frames_saved": 0,
            "window_len": WINDOW_LEN,
            "stride": STRIDE,
            "batch_size": BATCH_SIZE,
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
        print("\nAll saliency maps already exist. Nothing to run.")
        return summary

    num_saved = 0
    pbar = tqdm(
        loader,
        desc="Inference",
        leave=True,
        # disable=not train_cfg.SHOW_PROGRESS_BAR,
    )

    for (
        video_filenames,
        frame_names,
        rgb_batch,
        sal_batch,
        fix_batch,
        _n_frames,
    ) in pbar:
        rgb_batch, sal_batch, fix_batch = model.prepare_training_batch(
            rgb_batch,
            sal_batch,
            fix_batch,
        )

        with autocast(
            device.type,
            dtype=train_cfg._amp_dtype(device),
            enabled=train_cfg._amp_enabled(device),
        ):
            model_out = model(
                rgb_batch,
                saliency_maps=None,
                return_details=True,
                return_concept_losses=False,
                return_decoder_diagnostics=False,
            )

        pred_batch = model_out["saliency_map"]
        if pred_batch.dim() != 4 or pred_batch.shape[1] != 1:
            raise ValueError(
                f"Expected saliency_map [B,1,H,W], got {tuple(pred_batch.shape)}"
            )

        pred_batch = torch.sigmoid(pred_batch.float())
        _, _, _, in_h, in_w = rgb_batch.shape

        for i, (video_name, frame_name) in enumerate(
            zip(video_filenames, frame_names)
        ):
            save_path = os.path.join(output_root, video_name, frame_name)
            if SKIP_EXISTING and os.path.isfile(save_path):
                continue

            pred_i = F.interpolate(
                pred_batch[i : i + 1],
                size=(in_h, in_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            _save_saliency_png(pred_i, save_path)
            num_saved += 1

        pbar.set_postfix(saved=num_saved, remaining=max(len(dataset) - num_saved, 0))

    summary = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "test_dir": TEST_DATASET_DIR,
        "output_root": str(Path(output_root).resolve()),
        "device": str(device),
        "num_frames_total": dataset.num_total_frames,
        "num_frames_skipped_existing": dataset.num_skipped_existing,
        "num_frames_scheduled": len(dataset),
        "num_frames_saved": num_saved,
        "window_len": WINDOW_LEN,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    print(
        f"\nInference complete. Saved {num_saved} new saliency maps "
        f"({dataset.num_skipped_existing} were already present) under "
        f"{output_root}"
    )
    return summary


def main() -> None:
    run_inference(CHECKPOINT, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
