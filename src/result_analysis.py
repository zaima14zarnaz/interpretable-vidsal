"""
Analyze validation metrics and plot CC curves from training logs.

Find the best validation epoch by maximizing CC + SIM + NSS in val_metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAIN_CC_RE = re.compile(r"Train metrics \| CC:\s*([\d.]+)")
VAL_CC_RE = re.compile(r"Val metrics\s+\| CC:\s*([\d.]+)")


@dataclass(frozen=True)
class ValMetricsRow:
    epoch: int
    cc: float
    sim: float
    nss: float

    @property
    def combined_score(self) -> float:
        return self.cc + self.sim + self.nss


def load_val_metrics(csv_path: Path) -> List[ValMetricsRow]:
    """
    Load per-epoch validation metrics from val_metrics.csv.

    Rows are written one per epoch starting at epoch 1 (see train.py).
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Validation metrics CSV not found: {csv_path}")

    rows: List[ValMetricsRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"CC", "SIM", "NSS"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{csv_path} must contain columns CC, SIM, and NSS; "
                f"got {reader.fieldnames!r}"
            )

        for epoch_no, row in enumerate(reader, start=1):
            rows.append(
                ValMetricsRow(
                    epoch=epoch_no,
                    cc=float(row["CC"]),
                    sim=float(row["SIM"]),
                    nss=float(row["NSS"]),
                )
            )

    if not rows:
        raise ValueError(f"No validation metric rows found in {csv_path}")

    return rows


def find_best_epoch_by_combined_score(rows: List[ValMetricsRow]) -> ValMetricsRow:
    """Return the epoch with the highest CC + SIM + NSS."""
    return max(rows, key=lambda row: row.combined_score)


def print_best_epoch_summary(best: ValMetricsRow, csv_path: Path) -> None:
    print(f"Best epoch from {csv_path}:")
    print(f"  epoch: {best.epoch}")
    print(f"  CC:    {best.cc:.6f}")
    print(f"  SIM:   {best.sim:.6f}")
    print(f"  NSS:   {best.nss:.6f}")
    print(f"  CC+SIM+NSS: {best.combined_score:.6f}")


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
    parser = argparse.ArgumentParser(
        description=(
            "Find the best validation epoch by CC+SIM+NSS and optionally "
            "plot CC curves from out.log."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=script_dir / "training_outputs" / "val_metrics.csv",
        help="Path to val_metrics.csv (default: training_outputs/val_metrics.csv)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional training log for CC curve plot (default: skip plotting)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "training_outputs" / "cc_progress.png",
        help="Output PNG path when --log is provided",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    rows = load_val_metrics(csv_path)
    best = find_best_epoch_by_combined_score(rows)
    print_best_epoch_summary(best, csv_path)

    if args.log is not None:
        log_path = args.log.resolve()
        if not log_path.is_file():
            raise FileNotFoundError(f"Log file not found: {log_path}")
        train_cc, val_cc = parse_cc_metrics(log_path)
        plot_cc_progress(train_cc, val_cc, args.output.resolve())
        print(f"Parsed {len(train_cc)} epochs from {log_path}")
        print(f"Saved CC curve to {args.output.resolve()}")


if __name__ == "__main__":
    main()
