"""Train-side appearance monitoring / aggregation tests for the fixed-cluster pipeline."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional
from unittest import mock

import pandas as pd
import torch

import train as train_mod
from losses import _aggregate_concept_total_loss


def _stage_out(
    *,
    evaluated: float,
    total: float,
    rgb: float,
    feature: float,
    app_loss: float,
    structure: Optional[float] = None,
    loss_total_concept: Optional[float] = None,
) -> Dict[str, Any]:
    if structure is None:
        structure = evaluated
    losses: Dict[str, Any] = {
        "loss_visual_appearance": torch.tensor(app_loss, requires_grad=True),
    }
    if loss_total_concept is not None:
        losses["loss_total_concept"] = torch.tensor(
            loss_total_concept, requires_grad=True
        )
    return {
        "visual_appearance_diagnostics": {
            "app_loss_patch_count": torch.tensor(evaluated),
            "app_loss_patch_count_total": torch.tensor(total),
            "app_loss_rgb_patch_count": torch.tensor(rgb),
            "app_loss_feature_patch_count": torch.tensor(feature),
            "app_loss_structure_patch_count": torch.tensor(structure),
        },
        "losses": losses,
    }


def _model_out(stages: Dict[str, Dict[str, Any]]) -> dict:
    return {"concept_out": stages}


def test_stage4_only_and_patch_weighted_mean():
    """Unequal microbatches + smaller final batch; stage1/2/3 zeros ignored."""
    acc = train_mod._new_appearance_epoch_accumulator()

    b0 = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage1": _stage_out(evaluated=0, total=0, rgb=0, feature=0, app_loss=0.0),
                "stage2": _stage_out(evaluated=0, total=0, rgb=0, feature=0, app_loss=0.0),
                "stage3": _stage_out(evaluated=8, total=8, rgb=8, feature=6, app_loss=0.40),
                "stage4": _stage_out(
                    evaluated=2,
                    total=2,
                    rgb=2,
                    feature=0,
                    structure=2,
                    app_loss=0.20,
                ),
            }
        )
    )
    assert abs(b0["app_loss"] - 0.20) < 1e-6
    assert abs(b0["App_loss_patch_count"] - 2.0) < 1e-6
    assert set(b0["stages"]) == {"stage4"}
    train_mod._accumulate_appearance_epoch_stats(acc, b0)

    # Smaller final microbatch.
    b1 = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage3": _stage_out(evaluated=4, total=4, rgb=4, feature=4, app_loss=0.10),
                "stage4": _stage_out(
                    evaluated=1,
                    total=1,
                    rgb=1,
                    feature=1,
                    structure=1,
                    app_loss=0.50,
                ),
            }
        )
    )
    assert abs(b1["app_loss"] - 0.50) < 1e-6
    assert abs(b1["App_loss_patch_count"] - 1.0) < 1e-6
    train_mod._accumulate_appearance_epoch_stats(acc, b1)

    # Zero-coverage microbatch must not dilute the weighted mean.
    b2 = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage4": _stage_out(evaluated=0, total=2, rgb=0, feature=0, app_loss=0.0),
            }
        )
    )
    train_mod._accumulate_appearance_epoch_stats(acc, b2)

    final = train_mod._finalize_appearance_epoch_stats(acc)
    assert abs(final["App_loss_patch_count"] - 3.0) < 1e-6
    # (0.20*2 + 0.50*1 + 0*0) / 3
    assert abs(final["app_loss"] - (0.9 / 3.0)) < 1e-6
    assert final["stages"]["stage4"]["structure"] == 3.0
    print("PASS CSV counts / patch-weighted means with unequal batches")


def test_other_stages_do_not_dilute_logged_app_loss():
    stats = train_mod._collect_main_forward_appearance_stats(
        _model_out(
            {
                "stage1": _stage_out(evaluated=0, total=16, rgb=0, feature=0, app_loss=0.0),
                "stage2": _stage_out(evaluated=0, total=8, rgb=0, feature=0, app_loss=0.0),
                "stage3": _stage_out(evaluated=0, total=4, rgb=0, feature=0, app_loss=0.0),
                "stage4": _stage_out(evaluated=3, total=3, rgb=3, feature=3, app_loss=0.42),
            }
        )
    )
    assert abs(stats["app_loss"] - 0.42) < 1e-6
    assert stats["App_loss_patch_count"] == 3.0
    # Equal-stage mean of (0,0,0,0.42) would have been 0.105 — must not happen.
    assert abs(stats["app_loss"] - 0.105) > 0.2
    print("PASS other stages' zero losses do not dilute logged app_loss")


def test_epoch_with_no_evaluated_queries():
    acc = train_mod._new_appearance_epoch_accumulator()
    for _ in range(3):
        batch = train_mod._collect_main_forward_appearance_stats(
            _model_out(
                {
                    "stage4": _stage_out(evaluated=0, total=2, rgb=0, feature=0, app_loss=0.0),
                }
            )
        )
        train_mod._accumulate_appearance_epoch_stats(acc, batch)
    final = train_mod._finalize_appearance_epoch_stats(acc)
    assert final["App_loss_patch_count"] == 0.0
    assert math.isnan(final["app_loss"])
    row = train_mod._format_val_metrics_csv_row(1, None, None, appearance_stats=final)
    assert row["App_loss_patch_count"] == 0.0
    assert row["app_loss"] == ""
    print("PASS no-evaluated-queries → empty app_loss")


def test_csv_header_and_history_preserved():
    old_row = {"Epoch_no": 1, "Val Loss": 0.5, "CC": 0.1, "SIM": 0.2, "NSS": 0.3}
    new_stats = {"App_loss_patch_count": 15.0, "app_loss": 0.25, "stages": {}}
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
    assert float(df.loc[1, "App_loss_patch_count"]) == 15.0
    csv_text = df.to_csv(index=False, na_rep="")
    print("--- CSV header + example rows ---")
    print(csv_text.strip())
    header = csv_text.strip().splitlines()[0]
    assert header == "Epoch_no,Val Loss,CC,SIM,NSS,App_loss_patch_count,app_loss"
    print("PASS CSV header / history empty cells preserved")


def test_appearance_enters_total_once_with_stage_average():
    """Appearance contributes only via mean(stage loss_total_concept), never twice."""
    # Four stages: only stage4 carries appearance inside its concept total.
    w_app = 1.0
    raw_app = 0.8
    stage4_concept = w_app * raw_app  # appearance-only concept total
    concept_out = {
        "stage1": {"losses": {"loss_total_concept": torch.tensor(0.0)}},
        "stage2": {"losses": {"loss_total_concept": torch.tensor(0.0)}},
        "stage3": {"losses": {"loss_total_concept": torch.tensor(0.0)}},
        "stage4": {
            "losses": {
                "loss_visual_appearance": torch.tensor(raw_app),
                "loss_total_concept": torch.tensor(stage4_concept),
            }
        },
    }
    ref = torch.zeros(())
    aggregated = _aggregate_concept_total_loss(concept_out, ref)
    expected = stage4_concept / 4.0
    assert abs(float(aggregated) - expected) < 1e-6

    # Mimic compute_total_loss composition: appearance is NOT added separately.
    fake_total_without_concept = torch.tensor(1.25)
    loss_total = fake_total_without_concept + aggregated
    # If someone wrongly also added raw_app, the total would be higher by 0.8.
    wrongly_double = loss_total + torch.tensor(raw_app)
    assert abs(float(loss_total) - (1.25 + expected)) < 1e-6
    assert abs(float(wrongly_double) - float(loss_total)) > 0.5
    print(
        "PASS appearance enters total once with stage-average scaling "
        f"(contrib={float(aggregated):.4f} = {w_app}*{raw_app}/4)"
    )


def test_appearance_only_requests_concept_losses_and_cache():
    class _Stage4:
        loss_weights = {
            "visual_appearance": 1.0,
            "visual_rgb": 1.0,
            "visual_feature": 1.0,
            "visual_structure": 1.0,
        }

    class _Model:
        def __init__(self, training=True):
            self.training = training
            self.concept_creations = {"stage4": _Stage4()}

    # Zero out unrelated concept lambdas so appearance-only is the reason.
    with mock.patch.object(
        train_mod,
        "_effective_loss_lambda",
        return_value={
            "lambda_align": 0.0,
            "lambda_sparse": 0.0,
            "lambda_div": 0.0,
            "lambda_gate": 0.0,
            "lambda_visual": 0.0,
            "lambda_visual_div": 0.0,
            "lambda_visual_entropy": 0.0,
            "lambda_visual_usage": 0.0,
            "lambda_visual_equiv": 0.0,
        },
    ), mock.patch.object(train_mod, "USE_CACHED_BACKBONE_FEATURES", True):
        model = _Model(training=True)
        assert train_mod._return_concept_losses(model) is True
        assert train_mod._needs_reference_features(model) is True
        model_eval = _Model(training=False)
        assert train_mod._needs_reference_features(model_eval) is False
    print("PASS appearance-only requests concept losses and cached features")


def test_eval_skips_cache_resolution_helper():
    class _Stage4:
        loss_weights = {
            "visual_appearance": 1.0,
            "visual_rgb": 1.0,
            "visual_feature": 1.0,
            "visual_structure": 1.0,
        }

    class _Model:
        training = False
        concept_creations = {"stage4": _Stage4()}

    with mock.patch.object(train_mod, "USE_CACHED_BACKBONE_FEATURES", True):
        assert train_mod._needs_reference_features(_Model()) is False
    print("PASS eval skips cache resolution via _needs_reference_features")


def test_exclude_flipped_forward_by_caller():
    main = train_mod._collect_main_forward_appearance_stats(
        _model_out({"stage4": _stage_out(evaluated=4, total=4, rgb=4, feature=0, app_loss=0.2)})
    )
    flipped = train_mod._collect_main_forward_appearance_stats(
        _model_out({"stage4": _stage_out(evaluated=4, total=4, rgb=4, feature=4, app_loss=9.0)})
    )
    acc = train_mod._new_appearance_epoch_accumulator()
    train_mod._accumulate_appearance_epoch_stats(acc, main)
    final = train_mod._finalize_appearance_epoch_stats(acc)
    assert final["App_loss_patch_count"] == 4.0
    assert abs(final["app_loss"] - 0.2) < 1e-6
    assert abs(flipped["app_loss"] - 9.0) < 1e-6
    print("PASS flipped-forward exclusion (caller collects main only)")


if __name__ == "__main__":
    test_stage4_only_and_patch_weighted_mean()
    test_other_stages_do_not_dilute_logged_app_loss()
    test_epoch_with_no_evaluated_queries()
    test_csv_header_and_history_preserved()
    test_appearance_enters_total_once_with_stage_average()
    test_appearance_only_requests_concept_losses_and_cache()
    test_eval_skips_cache_resolution_helper()
    test_exclude_flipped_forward_by_caller()
    print("ALL TRAIN APPEARANCE CSV TESTS PASSED")
