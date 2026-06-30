"""Regression test: eval predictions must not depend on GT saliency maps."""

from __future__ import annotations

import torch

from model.model import ExplainableVidSalModel


def assert_eval_prediction_invariant_to_gt_saliency(
    model: ExplainableVidSalModel,
    video: torch.Tensor,
    saliency_maps: torch.Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """
    In eval mode, final saliency outputs must match with or without GT saliency
    when concept losses are disabled.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out_without_gt = model(
                video,
                saliency_maps=None,
                return_details=True,
                return_concept_losses=False,
            )
            out_with_gt = model(
                video,
                saliency_maps=saliency_maps,
                return_details=True,
                return_concept_losses=False,
            )
    finally:
        model.train(was_training)

    for key in ("saliency_map", "coarse_saliency_logits"):
        torch.testing.assert_close(
            out_without_gt[key],
            out_with_gt[key],
            atol=atol,
            rtol=rtol,
            msg=f"{key} changed when GT saliency was passed in eval mode",
        )


def _build_test_model() -> ExplainableVidSalModel:
    return ExplainableVidSalModel(
        backbone_stages=("stage2",),
        pretrained_backbone=False,
        freeze_backbone=True,
        resize_to=(224, 224),
        concept_dim=64,
        num_concepts=8,
        concept_hidden_dim=64,
        saliency_hidden_dim=64,
        top_k=3,
        max_source_patches=16,
        use_feature_refinement=False,
        use_gated_trajectory_head=False,
        use_subpatch_head=False,
        return_details=True,
        visual_concept_on=True,
        temporal_concepts_on=True,
        allow_eval_concept_losses=False,
    )


def main() -> None:
    torch.manual_seed(0)
    model = _build_test_model()
    model.eval()

    batch_size, num_frames, height, width = 1, 4, 224, 224
    video = torch.rand(batch_size, num_frames, 3, height, width)
    saliency_maps = torch.rand(batch_size, num_frames, height, width)

    assert_eval_prediction_invariant_to_gt_saliency(model, video, saliency_maps)
    print("OK: eval saliency outputs are invariant to GT saliency maps")


if __name__ == "__main__":
    main()
