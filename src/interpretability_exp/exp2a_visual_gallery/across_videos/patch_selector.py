#!/usr/bin/env python3
"""Export every stage-4 image cell for manual semantic seed selection.

Place beside prototype_to_patch.py for automatic grid discovery. Either edit
the settings below and run `python patch_selector.py`, or use command-line flags:

    python patch_selector.py --frame /data/val/601/images/0300.png \
        --checkpoint /path/to/epoch_125.pth --output-root selected_patches

If you already KNOW the concept grid and backbone input dimensions, no model
or PyTorch is needed (Pillow only):

    python patch_selector.py --frame /path/to/frame.png \
        --grid-hw GRID_HEIGHT GRID_WIDTH --model-input-hw INPUT_HEIGHT INPUT_WIDTH

Grid dimensions must match the stage4 concept input in patch_to_prototype.py.
Each crop is a nominal spatial grid cell, not its full backbone receptive field.
No features, prototype scores, or semantic labels are computed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image


# ------------------------- EDIT THESE SETTINGS -------------------------
frame_dir = "/data/quantization/zaima/videosal_datasets/dhf1k/train/001/images/0001.png"  # A single frame FILE, despite the name.
OUT_DIR_ROOT = "/data/quantization/zaima/videosal_datasets/dhf1k/patches/"
CHECKPOINT = "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/ckpts/20261001_040032/epoch_068.pth"  # e.g. "/path/to/epoch_125.pth"; needed for automatic mode.
GALLERY_SCRIPT = str(Path(__file__).with_name("prototype_to_patch.py"))
WINDOW_LEN = 32

# Leave both None for automatic model-based dimension discovery.
# Otherwise set both to the verified (height, width) used by your model.
STAGE4_GRID_HW = None
MODEL_INPUT_HW = None

# Optional overrides. Default video name: parent directory, or parent of
# an images/frames/rgb directory. Default frame number: filename without suffix.
VIDEO_FNAME = "001"
FRAME_NO = "0001"
STAGE = 4
# ----------------------------------------------------------------------


def bounds(patch_index, grid_hw, image_hw):
    """Exactly the proportional, rounded cell mapping in patch_to_prototype.py."""
    gh, gw = grid_hw
    height, width = image_hw
    if not 0 <= patch_index < gh * gw:
        raise ValueError("Patch index outside feature grid.")
    row, column = divmod(patch_index, gw)
    return (round(column * width / gw), round(row * height / gh),
            round((column + 1) * width / gw), round((row + 1) * height / gh))


def load_adapter(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gallery adapter not found: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("patch_selector_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    return adapter


def model_frame_and_grid(image, args):
    """Discover spatial geometry; repeated frames are used ONLY for grid shape."""
    import numpy as np
    import torch

    adapter = load_adapter(args.gallery_script)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = adapter.resolve_checkpoint_path(args.checkpoint)
    model = adapter.load_saliency_model(str(checkpoint), device)
    model.eval()
    if "stage4" not in model.concept_creations:
        raise ValueError("The model has no stage4 concept module.")
    input_hw = tuple(int(v) for v in adapter.resolve_backbone_spatial_hw(model))
    frame = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    rgb = frame[None, None].repeat(1, args.window_len, 1, 1, 1)  # [B,T,C,H,W]
    captured = {}

    def capture_grid(module, inputs):
        value = inputs[0]  # Stage4 concept input: [B,C,T,H,W].
        if value.ndim != 5:
            raise ValueError("Expected a [B,C,T,H,W] stage4 concept input.")
        captured["grid_hw"] = tuple(int(v) for v in value.shape[-2:])

    hook = model.concept_creations["stage4"].register_forward_pre_hook(capture_grid)
    try:
        with torch.inference_mode():
            output = model(rgb.to(model.input_device), return_details=True,
                           return_concept_losses=False, use_reference_cache=False)
            if "grid_hw" not in captured:
                raise RuntimeError("Stage4 concept input hook did not run.")
            shape = output["concept_out"]["stage4"]["visual_metadata"]["feature_shape"]
            grid_hw = captured["grid_hw"]
            if grid_hw != (int(shape["H"]), int(shape["W"])):
                raise ValueError("Stage4 input and concept metadata grid dimensions disagree.")
            # Same pre-normalization resizing and RGB rounding as the existing exporter.
            resized = adapter.preprocess_rgb_video_like_backbone(rgb, input_hw)[0, -1]
            pixels = (resized.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255)
            model_image = Image.fromarray(pixels.round().astype(np.uint8))
    finally:
        hook.remove()
    if (model_image.height, model_image.width) != input_hw:
        raise ValueError("Preprocessed image does not match the backbone input size.")
    return model_image, grid_hw, input_hw, {
        "grid_source": "stage4_concept_input_hook",
        "checkpoint": str(checkpoint),
        "gallery_script": str(Path(args.gallery_script).expanduser().resolve()),
        "shape_probe_window_len": args.window_len,
        "shape_probe_uses_repeated_frame": True,
        "preprocessing": "preprocess_rgb_video_like_backbone",
    }


def safe_component(value, label):
    value = str(value)
    if not value.strip() or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single nonempty directory name.")
    return value


def export_patches(image, model_image, grid_hw, input_hw, args, provenance):
    frame_path = Path(args.frame).expanduser().resolve()
    parent = frame_path.parent
    if parent.name.lower() in {"images", "frames", "rgb"}:
        parent = parent.parent
    video_fname = safe_component(args.video_fname or parent.name, "video_fname")
    frame_no = safe_component(args.frame_no or frame_path.stem, "frame_no")
    out_dir = Path(args.output_root).expanduser().resolve() / video_fname / frame_no
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output frame directory is not empty: {out_dir}. Use a new output root.")
    native_hw = (image.height, image.width)
    for label, hw in (("grid_hw", grid_hw), ("model_input_hw", input_hw)):
        if len(hw) != 2 or any(v < 1 for v in hw):
            raise ValueError(f"{label} must contain two positive integers.")
    if grid_hw[0] > input_hw[0] or grid_hw[1] > input_hw[1]:
        raise ValueError("Grid exceeds model image dimensions; some crops would be empty.")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "schema_version": 1,
        "video_fname": video_fname,
        "video_id": video_fname,
        "frame_no": frame_no,  # Filename label; NOT an assumed zero-based dataset index.
        "frame_filename": frame_path.name,
        "frame_path": str(frame_path),
        "source_frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        "stage": STAGE,
        "stage_name": "stage4",
        "feature_grid_hw": list(grid_hw),
        "model_input_hw": list(input_hw),
        "native_image_hw": list(native_hw),
        "patch_index_order": "zero_based_row_major",
        "coordinate_convention": "xyxy; left/top inclusive, right/bottom exclusive",
        "nominal_cell_not_receptive_field": True,
        **provenance,
    }
    patches = []
    gh, gw = grid_hw
    for patch_index in range(gh * gw):
        row, column = divmod(patch_index, gw)
        patch_id = f"{patch_index:04d}"
        model_box = bounds(patch_index, grid_hw, input_hw)
        native_box = bounds(patch_index, grid_hw, native_hw)
        record = {
            **source,
            "patch_id": patch_id,
            "patch_index": patch_index,
            "grid_row": row,
            "grid_column": column,
            "box_model_xyxy": list(model_box),
            "box_native_xyxy": list(native_box),
            "patch_filename": f"{patch_id}.jpg",
            "patch_path": str(out_dir / f"{patch_id}.jpg"),
        }
        # Metadata survives a normal file copy or move into group_{group_id}.
        # The later grouping script can read Image.open(path).getexif()[270].
        exif = Image.Exif()
        exif[270] = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        model_image.crop(model_box).save(out_dir / record["patch_filename"],
                                        quality=95, subsampling=0, exif=exif)
        patches.append(record)
    metadata_dir = out_dir / "metadata.json"  # Requested metadata FILE location.
    metadata_dir.write_text(json.dumps({**source, "patch_count": len(patches),
                                         "patches": patches}, indent=2) + "\n", encoding="utf-8")
    # CSV counterpart: array-valued columns are JSON strings.
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patches[0]))
        writer.writeheader()
        for record in patches:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in record.items()})
    print(f"Saved all {len(patches)} stage4 patches ({gh} x {gw}) to {out_dir}")
    print(f"Metadata: {metadata_dir} and {out_dir / 'metadata.csv'}")
    print("Copy one selected JPG per group into group_{group_id}; keep the original exports.")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", default=frame_dir)
    parser.add_argument("--output-root", default=OUT_DIR_ROOT)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--gallery-script", default=GALLERY_SCRIPT)
    parser.add_argument("--window-len", type=int, default=WINDOW_LEN)
    parser.add_argument("--device", default=None)
    parser.add_argument("--grid-hw", nargs=2, type=int, default=STAGE4_GRID_HW, metavar=("H", "W"))
    parser.add_argument("--model-input-hw", nargs=2, type=int, default=MODEL_INPUT_HW, metavar=("H", "W"))
    parser.add_argument("--video-fname", default=VIDEO_FNAME)
    parser.add_argument("--frame-no", default=FRAME_NO)
    args = parser.parse_args()
    if args.window_len < 1:
        parser.error("--window-len must be positive.")
    if (args.grid_hw is None) != (args.model_input_hw is None):
        parser.error("Supply both --grid-hw and --model-input-hw, or neither.")
    if args.grid_hw is None and not args.checkpoint:
        parser.error("Set CHECKPOINT/--checkpoint for automatic mode, or supply both known dimensions.")
    if args.grid_hw is not None and any(v < 1 for v in (*args.grid_hw, *args.model_input_hw)):
        parser.error("Grid and model input dimensions must be positive.")
    return args


def main():
    args = parse_args()
    frame_path = Path(args.frame).expanduser().resolve()
    if not frame_path.is_file():
        raise FileNotFoundError(f"Frame image not found: {frame_path}")
    with Image.open(frame_path) as opened:
        image = opened.convert("RGB")
    if args.grid_hw is None:
        model_image, grid_hw, input_hw, provenance = model_frame_and_grid(image, args)
    else:
        grid_hw, input_hw = tuple(args.grid_hw), tuple(args.model_input_hw)
        model_image = image.resize((input_hw[1], input_hw[0]), Image.Resampling.BILINEAR)
        provenance = {"grid_source": "user_supplied_verified_dimensions",
                      "preprocessing": "Pillow_bilinear_resize",
                      "note": "Manual resize may differ slightly from the adapter's tensor interpolation."}
    export_patches(image, model_image, grid_hw, input_hw, args, provenance)


if __name__ == "__main__":
    main()
