"""Synthetic aggregation checks for train.py appearance CSV columns (no full train)."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd
import torch

import train as train_mod


def _stage_out(
    *,
    evaluated: float,
    total: float,
    rgb: float,
    feature: float,
    app_loss: float,
) -> Dict[str, Any]:
    return {
        "visual_appearance_diagnostics": {
            "app_loss_patch_count": torch.tensor(evaluated),
            "app_loss_patch_count_total": torch.tensor(total),
            "app_loss_rgb_patch_count": torch.tensor(rgb),
            "app_loss_feature_patch_count": torch.tensor(feature),
        },
        "losses": {
            "loss_visual_appearance": torch.tensor(app_loss, requires_grad=True),
        },
    }


def _model_out(stages: Dict[str, Dict[str, Any]]) -> dict:
    return {"concept_out": stages}


def test_unequal_stages_and_microbatches():
    """Unequal stage grids + smaller final batch + grad-accum-style microbatches."""
    acc = train_mod._new_appearance_epoch_accumulator()

    # Microbatch 0: full history. stage3 grid 8, stage4 grid 2 (unequal).
    b0 = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage3": _stage_out(evaluated=8, total=8, rgb=8, feature=6, app_loss=0.40),
                "stage4": _stage_out(evaluated=2, total=2, rgb=2, feature=0, app_loss=0.20),
            }
        )
    )
    # Equal-stage mean: (0.40 + 0.20) / 2 = 0.30; patch sum = 10
    assert abs(b0["app_loss"] - 0.30) < 1e-6
    assert abs(b0["App_loss_patch_count"] - 10.0) < 1e-6
    train_mod._accumulate_appearance_epoch_stats(acc, b0)

    # Microbatch 1 (smaller final batch): stage3 4 patches, stage4 1.
    b1 = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage3": _stage_out(evaluated=4, total=4, rgb=4, feature=4, app_loss=0.10),
                "stage4": _stage_out(evaluated=1, total=1, rgb=1, feature=1, app_loss=0.30),
            }
        )
    )
    assert abs(b1["app_loss"] - 0.20) < 1e-6
    assert abs(b1["App_loss_patch_count"] - 5.0) < 1e-6
    train_mod._accumulate_appearance_epoch_stats(acc, b1)

    # Do NOT divide patch counts by accum_steps (simulate accum=2 by leaving sums).
    final = train_mod._finalize_appearance_epoch_stats(acc)
    assert abs(final["App_loss_patch_count"] - 15.0) < 1e-6  # 10+5, not 15/2
    # Epoch mean like train loss: mean over microbatches of equal-stage means.
    assert abs(final["app_loss"] - 0.25) < 1e-6  # (0.30 + 0.20) / 2

    assert final["stages"]["stage3"]["evaluated"] == 12.0
    assert final["stages"]["stage3"]["total"] == 12.0
    assert final["stages"]["stage3"]["rgb"] == 12.0
    assert final["stages"]["stage3"]["feature"] == 10.0
    assert final["stages"]["stage4"]["feature"] == 1.0
    print("PASS unequal stages / smaller final batch / no accum division")


def test_warmup_then_populated():
    acc = train_mod._new_appearance_epoch_accumulator()
    warm = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage3": _stage_out(evaluated=0, total=8, rgb=0, feature=0, app_loss=0.0),
                "stage4": _stage_out(evaluated=0, total=2, rgb=0, feature=0, app_loss=0.0),
            }
        )
    )
    assert warm["App_loss_patch_count"] == 0.0
    train_mod._accumulate_appearance_epoch_stats(acc, warm)

    populated = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage3": _stage_out(evaluated=8, total=8, rgb=8, feature=8, app_loss=0.5),
                "stage4": _stage_out(evaluated=2, total=2, rgb=2, feature=2, app_loss=0.1),
            }
        )
    )
    train_mod._accumulate_appearance_epoch_stats(acc, populated)
    final = train_mod._finalize_appearance_epoch_stats(acc)
    assert final["App_loss_patch_count"] == 10.0
    # Mean includes warmup zero microbatch: (0.0 + 0.3) / 2
    assert abs(final["app_loss"] - 0.15) < 1e-6
    print("PASS warmup then populated")


def test_epoch_with_no_evaluated_queries():
    acc = train_mod._new_appearance_epoch_accumulator()
    for _ in range(3):
        batch = train_mod._collect_main_forward_appearance_stats(
            _model_out(
                {
                    "stage3": _stage_out(evaluated=0, total=8, rgb=0, feature=0, app_loss=0.0),
                    "stage4": _stage_out(evaluated=0, total=2, rgb=0, feature=0, app_loss=0.0),
                }
            )
        )
        train_mod._accumulate_appearance_epoch_stats(acc, batch)
    final = train_mod._finalize_appearance_epoch_stats(acc)
    assert final["App_loss_patch_count"] == 0.0
    assert math.isnan(final["app_loss"])
    row = train_mod._format_val_metrics_csv_row(
        1, None, None, appearance_stats=final
    )
    assert row["App_loss_patch_count"] == 0.0
    assert row["app_loss"] == ""
    print("PASS no-evaluated-queries → empty app_loss")


def test_exclude_flipped_forward():
    """Only main forward stats are accumulated; flipped outputs are ignored by API use."""
    main = train_mod._collect_main_forward_appearance_stats(
        _model_out({"stage3": _stage_out(evaluated=4, total=4, rgb=4, feature=0, app_loss=0.2)})
    )
    flipped = train_mod._collect_main_forward_appearance_stats(
        _model_out({"stage3": _stage_out(evaluated=4, total=4, rgb=4, feature=4, app_loss=9.0)})
    )
    acc = train_mod._new_appearance_epoch_accumulator()
    train_mod._accumulate_appearance_epoch_stats(acc, main)
    # Caller must not pass flipped; verify main-only finalize.
    final = train_mod._finalize_appearance_epoch_stats(acc)
    assert final["App_loss_patch_count"] == 4.0
    assert abs(final["app_loss"] - 0.2) < 1e-6
    assert abs(flipped["app_loss"] - 9.0) < 1e-6  # would pollute if included
    print("PASS flipped-forward exclusion (caller collects main only)")


def test_csv_header_and_resume_empty_cells():
    old_row = {"Epoch_no": 1, "Val Loss": 0.5, "CC": 0.1, "SIM": 0.2, "NSS": 0.3}
    new_stats = {
        "App_loss_patch_count": 15.0,
        "app_loss": 0.25,
        "stages": {},
    }
    new_row = train_mod._format_val_metrics_csv_row(
        2, 0.4, {"CC": 0.11, "SIM": 0.22, "NSS": 0.33}, appearance_stats=new_stats
    )
    df = pd.DataFrame([old_row, new_row], columns=list(train_mod.VAL_METRICS_CSV_COLUMNS))
    assert list(df.columns) == [
        "Epoch_no",
        "Val Loss",
        "CC",
        "SIM",
        "NSS",
        "App_loss_patch_count",
        "app_loss",
    ]
    assert pd.isna(df.loc[0, "App_loss_patch_count"]) or df.loc[0, "App_loss_patch_count"] == ""
    assert pd.isna(df.loc[0, "app_loss"]) or df.loc[0, "app_loss"] == ""
    assert float(df.loc[1, "App_loss_patch_count"]) == 15.0
    assert abs(float(df.loc[1, "app_loss"]) - 0.25) < 1e-8

    csv_text = df.to_csv(index=False, na_rep="")
    print("--- CSV header + example rows ---")
    print(csv_text.strip())
    header, row1, row2 = csv_text.strip().splitlines()
    assert header == "Epoch_no,Val Loss,CC,SIM,NSS,App_loss_patch_count,app_loss"
    # Older row: trailing empty appearance cells
    assert row1.endswith(",,") or row1.count(",") == 6
    assert "15.0" in row2 and "0.25" in row2
    print("PASS CSV header / resume empty cells / example row")


def test_combined_full_feature_missing():
    """Combined coverage can be complete via RGB while feature coverage is zero."""
    stats = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage3": _stage_out(evaluated=8, total=8, rgb=8, feature=0, app_loss=0.3),
            }
        )
    )
    assert stats["stages"]["stage3"]["evaluated"] == 8
    assert stats["stages"]["stage3"]["rgb"] == 8
    assert stats["stages"]["stage3"]["feature"] == 0
    print("PASS combined-complete / feature-missing coverage split")


if __name__ == "__main__":
    test_unequal_stages_and_microbatches()
    test_warmup_then_populated()
    test_epoch_with_no_evaluated_queries()
    test_exclude_flipped_forward()
    test_csv_header_and_resume_empty_cells()
    test_combined_full_feature_missing()
    print("ALL TRAIN APPEARANCE CSV TESTS PASSED")
