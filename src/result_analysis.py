"""
Parse training metrics from out.log and plot CC vs epoch.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAIN_CC_RE = re.compile(r"Train metrics \| CC:\s*([\d.]+)")
VAL_CC_RE = re.compile(r"Val metrics\s+\| CC:\s*([\d.]+)")


def _read_log_text(log_path: Path) -> str:
    raw = log_path.read_bytes()
    # tqdm progress lines use \r; normalize so regex can scan the full log.
    return raw.decode("utf-8", errors="replace").replace("\r", "\n")


def parse_cc_metrics(log_path: Path) -> Tuple[List[float], List[float]]:
    """
    Extract per-epoch train and validation CC from lines like:
      Train metrics | CC: 0.8569 | SIM: ...
      Val metrics   | CC: 0.6354 | SIM: ...
    """
    text = _read_log_text(log_path)
    train_cc = [float(v) for v in TRAIN_CC_RE.findall(text)]
    val_cc = [float(v) for v in VAL_CC_RE.findall(text)]
    if not train_cc or not val_cc:
        raise ValueError(
            f"No CC metrics found in {log_path}. "
            "Expected 'Train metrics | CC: ...' and 'Val metrics   | CC: ...' lines."
        )
    if len(train_cc) != len(val_cc):
        n = min(len(train_cc), len(val_cc))
        train_cc = train_cc[:n]
        val_cc = val_cc[:n]
    return train_cc, val_cc


def plot_cc_progress(
    train_cc: List[float],
    val_cc: List[float],
    output_path: Path,
) -> None:
    epochs = np.arange(1, len(train_cc) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_cc, label="Train CC", marker="o")
    ax.plot(epochs, val_cc, label="Val CC", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CC")
    ax.set_title("Correlation coefficient (CC) over training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot CC curves from training out.log")
    parser.add_argument(
        "--log",
        type=Path,
        default=script_dir / "out.log",
        help="Path to training log (default: src/out.log)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "training_outputs" / "cc_progress.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    log_path = args.log.resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    train_cc, val_cc = parse_cc_metrics(log_path)
    plot_cc_progress(train_cc, val_cc, args.output.resolve())
    print(f"Parsed {len(train_cc)} epochs from {log_path}")
    print(f"Saved CC curve to {args.output.resolve()}")


if __name__ == "__main__":
    main()
