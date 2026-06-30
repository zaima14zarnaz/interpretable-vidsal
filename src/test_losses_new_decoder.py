"""Regression: compute_total_loss works without legacy trajectory-head keys."""

from __future__ import annotations

import torch

from losses import _inject_feature_shape, _is_new_decoder, compute_total_loss


def test_inject_feature_shape_without_selected_metadata() -> None:
    prediction_out = {
        "saliency_map": torch.rand(2, 1, 224, 224),
        "saliency_logits": torch.randn(2, 1, 224, 224),
        "patch_saliency_logits": torch.randn(2, 1, 28, 28),
    }
    out = _inject_feature_shape(prediction_out, concept_out=None)
    assert out is prediction_out
    assert _is_new_decoder(prediction_out)


def test_compute_total_loss_new_decoder() -> None:
    B, T, H, W = 2, 4, 224, 224
    prediction_out = {
        "saliency_map": torch.rand(B, 1, H, W),
        "saliency_logits": torch.randn(B, 1, H, W),
        "patch_saliency_logits": torch.randn(B, 1, 28, 28),
        "coarse_saliency_logits": torch.randn(B, 1, H, W),
    }
    model_out = {
        "saliency_map": prediction_out["saliency_map"],
        "saliency_logits": prediction_out["saliency_logits"],
        "prediction_out": prediction_out,
        "concept_out": {
            "stage1": {
                "concept_representation": torch.randn(16, 64),
                "metadata": {
                    "feature_shape": {"B": B, "C": 96, "T": T, "H": 28, "W": 28},
                    "batch_idx": torch.zeros(16, dtype=torch.long),
                    "time_idx": torch.zeros(16, dtype=torch.long),
                    "source_idx": torch.zeros(16, dtype=torch.long),
                    "target_idx": torch.zeros(16, dtype=torch.long),
                },
                "losses": {
                    "loss_align": torch.tensor(0.1),
                    "loss_sparse": torch.tensor(0.2),
                    "loss_div": torch.tensor(0.05),
                    "loss_gate": torch.tensor(0.0),
                },
            }
        },
    }
    saliency_maps = torch.rand(B, T, H, W)
    fixation_maps = (torch.rand(B, T, H, W) > 0.95).float()

    loss_dict = compute_total_loss(
        model_out,
        saliency_maps,
        lambda_delta=0.0,
        lambda_dense=0.25,
        lambda_kl=1.0,
        lambda_cc=1.0,
        lambda_nss=1.0,
        fixation_maps=fixation_maps,
    )

    assert torch.isfinite(loss_dict["loss_total"])
    assert loss_dict["loss_delta"].detach().item() == 0.0


def main() -> None:
    test_inject_feature_shape_without_selected_metadata()
    test_compute_total_loss_new_decoder()
    print("OK")


if __name__ == "__main__":
    main()
