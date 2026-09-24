#!/usr/bin/env python3
"""Add full-validation CSV reporting to create_random_prototype_checkpoint.py.

The conversation attachment was a Markdown rendering rather than executable
Python, so this updater applies the change to the original .py in your repo.
It writes a backup before replacing the file and refuses unknown layouts.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


VALIDATION_FUNCTION = '''\
def run_validation_csv(
    *,
    original_checkpoint: Path,
    random_checkpoint: Path,
    csv_path: Path,
    dataset_dir: Optional[Path],
    window_len: Optional[int],
    device: torch.device,
    seed: int,
) -> None:
    """Evaluate every validation window under both checkpoints, one row each."""
    _ensure_src_on_path()
    import csv
    import train as train_cfg
    from pre_process.collate import video_saliency_collate_fn
    from pre_process.dataloader import DatasetLoader
    from metrics import compute_saliency_metrics, prepare_prediction_map

    _set_deterministic(seed)
    ds_dir = Path(dataset_dir) if dataset_dir is not None else Path(train_cfg.VAL_DATASET_DIR)
    win = int(window_len) if window_len is not None else int(train_cfg.WINDOW_LEN)
    dataset = DatasetLoader(str(ds_dir), window_len=win, stride=32)
    if len(dataset) == 0:
        raise ValueError(f"Validation dataset is empty: {ds_dir}")

    verification = verify_checkpoint_delta(original_checkpoint, random_checkpoint)
    if not verification["ok"]:
        raise RuntimeError("Original/random checkpoint verification failed")
    model_orig = build_diagnostic_model(device)
    load_model_checkpoint(model_orig, original_checkpoint)
    model_orig.eval()
    model_rand = build_diagnostic_model(device)
    load_model_checkpoint(model_rand, random_checkpoint)
    model_rand.eval()

    fieldnames = [
        "sample_index", "video", "start_frame", "n_frames",
        "original_CC", "random_CC", "original_SIM", "random_SIM",
        "original_NSS", "random_NSS", "mae", "relative_L2",
        "map_correlation", "cc_comp",
    ]

    def metric_value(metrics: Mapping[str, Any], name: str) -> float:
        matches = [value for key, value in metrics.items() if str(key).upper() == name]
        if len(matches) != 1:
            raise KeyError(f"Expected one {name} metric; got keys {list(metrics)}")
        value = matches[0]
        number = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite {name} metric: {number}")
        return number

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = csv_path.with_name(csv_path.name + ".part")
    # The .part file remains available if a long evaluation is interrupted.
    with part_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for sample_index in range(len(dataset)):
                sample = dataset[sample_index]
                _, rgb, sal, fix, n_frames, _ = video_saliency_collate_fn([sample])
                if sal is None or fix is None:
                    raise ValueError(f"Missing saliency or fixation target at index {sample_index}")
                # Preprocess once, then feed identical tensors into both models.
                rgb_ready, sal_ready, fix_ready = model_orig.prepare_training_batch(
                    rgb, sal, fix
                )
                if sal_ready is None or fix_ready is None:
                    raise ValueError(f"Missing prepared targets at index {sample_index}")
                fix_ready = (fix_ready > 0).float()
                forward_args = dict(
                    saliency_maps=sal_ready,
                    return_details=False,
                    return_concept_losses=False,
                    return_decoder_diagnostics=False,
                )
                out_orig = model_orig(rgb_ready, **forward_args)
                out_rand = model_rand(rgb_ready, **forward_args)
                # ExplainableVidSalModel returns a tensor with return_details=False
                # and a dict containing saliency_map with return_details=True.
                map_orig = (out_orig if torch.is_tensor(out_orig) else out_orig["saliency_map"]).detach().cpu()
                map_rand = (out_rand if torch.is_tensor(out_rand) else out_rand["saliency_map"]).detach().cpu()
                if map_orig.ndim != 4 or map_rand.shape != map_orig.shape or map_orig.shape[1] != 1:
                    raise ValueError(
                        f"Expected matching [B,1,H,W] maps, got {tuple(map_orig.shape)} "
                        f"and {tuple(map_rand.shape)}"
                    )

                sal_cpu = sal_ready.cpu()
                fix_cpu = fix_ready.cpu()
                metric_args = dict(
                    fixation_target=fix_cpu,
                    allow_pseudo_fixations=False,
                    dh1k_exact=True,
                )
                original_metrics = compute_saliency_metrics(map_orig, sal_cpu, **metric_args)
                random_metrics = compute_saliency_metrics(map_rand, sal_cpu, **metric_args)
                # Compare eval-prepared predictions, so MAE has the same scale
                # used by the repository's saliency metric preparation.
                comparison = compare_final_maps(
                    prepare_prediction_map(map_orig),
                    prepare_prediction_map(map_rand),
                )
                original_cc = metric_value(original_metrics, "CC")
                random_cc = metric_value(random_metrics, "CC")
                if random_cc > original_cc:
                    cc_comp = "better"
                elif random_cc == original_cc:
                    cc_comp = "equal"
                else:
                    cc_comp = "worse"

                video, start_frame = dataset.windows[sample_index]
                row = {
                    "sample_index": sample_index,
                    "video": video,
                    "start_frame": start_frame,
                    "n_frames": int(n_frames[0]),
                    "original_CC": original_cc,
                    "random_CC": random_cc,
                    "original_SIM": metric_value(original_metrics, "SIM"),
                    "random_SIM": metric_value(random_metrics, "SIM"),
                    "original_NSS": metric_value(original_metrics, "NSS"),
                    "random_NSS": metric_value(random_metrics, "NSS"),
                    "mae": comparison["mae"],
                    "relative_L2": comparison["relative_l2"],
                    "map_correlation": comparison["map_correlation"],
                    "cc_comp": cc_comp,
                }
                writer.writerow(row)
                if (sample_index + 1) % 100 == 0 or sample_index + 1 == len(dataset):
                    file.flush()
                    print(f"Validation: {sample_index + 1}/{len(dataset)} windows")

    part_path.replace(csv_path)
    print(f"Saved {len(dataset)} paired validation rows: {csv_path}")


'''


def replace_once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise ValueError(f"Expected exactly one {label} anchor, found {count}")
    return source.replace(before, after, 1)


def update_source(source: str) -> str:
    if "def run_validation_csv(" in source or "--validation-csv" in source:
        raise ValueError("Validation CSV support appears to be present already")
    for symbol in (
        "def verify_checkpoint_delta(",
        "def build_diagnostic_model(",
        "def load_model_checkpoint(",
        "def compare_final_maps(",
        "def main() -> None:",
    ):
        if symbol not in source:
            raise ValueError(f"Expected diagnostic script symbol is missing: {symbol}")

    source = replace_once(
        source,
        '    parser.add_argument("--atol", type=float, default=1e-5)',
        '    parser.add_argument(\n'
        '        "--validation-csv", type=Path, default=None,\n'
        '        help="Evaluate every validation window and write paired metrics to CSV.",\n'
        '    )\n'
        '    parser.add_argument("--atol", type=float, default=1e-5)',
        "argument parser",
    )
    source = replace_once(
        source, "def main() -> None:\n", VALIDATION_FUNCTION + "def main() -> None:\n", "main function"
    )
    source = replace_once(
        source,
        "    if not args.diagnostic:\n        return\n",
        "    if args.validation_csv is not None:\n"
        "        csv_path = args.validation_csv.expanduser().resolve()\n"
        "        ds_dir = args.dataset_dir.expanduser().resolve() if args.dataset_dir is not None else None\n"
        "        run_validation_csv(\n"
        "            original_checkpoint=args.checkpoint, random_checkpoint=args.output,\n"
        "            csv_path=csv_path, dataset_dir=ds_dir, window_len=args.window_len,\n"
        "            device=_resolve_device(args.device), seed=args.seed,\n"
        "        )\n\n"
        "    if not args.diagnostic:\n        return\n",
        "main diagnostic branch",
    )
    ast.parse(source)
    return source


def repair_existing_source(source: str) -> str:
    """Repair the original CSV addition that assumed dictionary model outputs."""
    original_line = '                map_orig = out_orig["saliency_map"].detach().cpu()'
    random_line = '                map_rand = out_rand["saliency_map"].detach().cpu()'
    if source.count(original_line) != 1 or source.count(random_line) != 1:
        raise ValueError("Expected CSV output lines not found; inspect the current file before editing")
    source = source.replace(
        original_line,
        '                map_orig = (out_orig if torch.is_tensor(out_orig) else out_orig["saliency_map"]).detach().cpu()',
        1,
    )
    source = source.replace(
        random_line,
        '                map_rand = (out_rand if torch.is_tensor(out_rand) else out_rand["saliency_map"]).detach().cpu()\n'
        '                if map_orig.ndim != 4 or map_rand.shape != map_orig.shape or map_orig.shape[1] != 1:\n'
        '                    raise ValueError(\n'
        '                        f"Expected matching [B,1,H,W] maps, got {tuple(map_orig.shape)} "\n'
        '                        f"and {tuple(map_rand.shape)}"\n'
        '                    )',
        1,
    )
    ast.parse(source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="Path to the original executable .py file")
    args = parser.parse_args()
    script = args.script.expanduser().resolve()
    source = script.read_text(encoding="utf-8")
    already_patched = "def run_validation_csv(" in source
    updated = repair_existing_source(source) if already_patched else update_source(source)
    backup_suffix = ".before_validation_csv_repair" if already_patched else ".before_validation_csv"
    backup = script.with_suffix(script.suffix + backup_suffix)
    if backup.exists():
        raise FileExistsError(f"Backup already exists: {backup}")
    backup.write_text(source, encoding="utf-8")
    script.write_text(updated, encoding="utf-8")
    print(f"Updated: {script}")
    print(f"Backup: {backup}")
    print("Use --validation-csv PATH with the checkpoint creation script.")


if __name__ == "__main__":
    main()
