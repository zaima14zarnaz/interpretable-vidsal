#!/usr/bin/env python3
"""Summarize mean predicted patch saliency per visual concept prototype."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _default_json_path() -> Path:
    return Path(__file__).resolve().parent / "explanation_outputs" / "retrieved_examples.json"


def load_retrieved_examples(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {path}, got {type(data).__name__}")
    return data


def patch_pred_saliency_values_for_concept(
    examples: List[Dict[str, Any]],
) -> List[float]:
    """Collect patch_pred_saliency for each retrieved example of one prototype."""
    values: List[float] = []
    for entry in examples:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("patch_pred_saliency")
        if raw is None:
            continue
        values.append(float(raw))
    return values


def patch_pred_saliency_variance(values: List[float]) -> Optional[float]:
    """Sample variance of patch_pred_saliency across retrieved examples."""
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    return statistics.variance(values)


def patch_pred_saliency_stats_for_concept(
    examples: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], int, List[float]]:
    """Return (mean, variance, count, values) for one prototype."""
    values = patch_pred_saliency_values_for_concept(examples)
    if not values:
        return None, None, 0, values
    return (
        statistics.fmean(values),
        patch_pred_saliency_variance(values),
        len(values),
        values,
    )


def summarize_by_prototype(
    retrieved: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(retrieved.keys(), key=lambda k: int(k)):
        examples = retrieved[key]
        if not isinstance(examples, list):
            raise ValueError(f"Concept {key!r} must map to a list, got {type(examples).__name__}")
        mean_sal, var_sal, count, saliency_list = patch_pred_saliency_stats_for_concept(
            examples
        )
        concept_idx = int(key)
        if examples and isinstance(examples[0], dict) and "concept_idx" in examples[0]:
            concept_idx = int(examples[0]["concept_idx"])
        rows.append(
            {
                "concept_idx": concept_idx,
                "num_examples": count,
                "mean": mean_sal,
                "variance": var_sal,
                "saliency_list": saliency_list,
            }
        )
    return rows


def _format_saliency_list(values: List[float]) -> str:
    inner = ", ".join(f"{v:.3f}" for v in values)
    return f"[{inner}]"


def print_summary(rows: List[Dict[str, Any]], *, only_with_examples: bool) -> None:
    display = [r for r in rows if r["num_examples"] > 0] if only_with_examples else rows
    print(
        f"{'concept':>8}  {'n':>4}  {'mean':>13}  "
        f"{'variance':>13}  saliency list"
    )
    print("-" * 72)
    for row in display:
        mean_sal = row["mean"]
        var_sal = row.get("variance")
        if mean_sal is None:
            continue
        mean_str = f"{mean_sal:.3f}" if mean_sal is not None else "—"
        var_str = f"{var_sal:.3f}" if var_sal is not None else "—"
        sal_list = row.get("saliency_list") or []
        sal_list_str = _format_saliency_list(sal_list) if sal_list else "[]"
        print(
            f"{row['concept_idx']:8d}  {row['num_examples']:4d}  {mean_str:>13}  "
            f"{var_str:>13}  {sal_list_str}"
        )

    with_examples = [r for r in rows if r["mean"] is not None]
    empty = len(rows) - len(with_examples)
    print()
    print(f"Prototypes total: {len(rows)}")
    print(f"Prototypes with retrieved examples: {len(with_examples)}")
    print(f"Prototypes with no examples: {empty}")
    if with_examples:
        means = [float(r["mean"]) for r in with_examples]
        print(f"Mean of per-prototype averages: {statistics.fmean(means):.6f}")
        print(f"Median of per-prototype averages: {statistics.median(means):.6f}")


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "concept_idx",
                "num_examples",
                "mean",
                "variance",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = {
                "concept_idx": row["concept_idx"],
                "num_examples": row["num_examples"],
                "mean": row["mean"],
                "variance": row.get("variance"),
            }
            if out["mean"] is None:
                out["mean"] = ""
            if out["variance"] is None:
                out["variance"] = ""
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read retrieved_examples.json and report the average patch_pred_saliency "
            "for each visual concept prototype."
        )
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=_default_json_path(),
        help="Path to retrieved_examples.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write per-prototype summary CSV",
    )
    parser.add_argument(
        "--only-with-examples",
        action="store_true",
        help="Print only prototypes that have at least one retrieved example",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = args.json.expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing retrieved examples file: {json_path}")

    retrieved = load_retrieved_examples(json_path)
    rows = summarize_by_prototype(retrieved)
    print(f"Source: {json_path}")
    print_summary(rows, only_with_examples=args.only_with_examples)

    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        write_csv(rows, csv_path)
        print(f"Wrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
