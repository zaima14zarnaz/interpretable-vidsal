"""
Stage-2 backbone-stream ablation configuration.

Baseline (train.STAGE2_BACKBONE_FUSION_ENABLED=True):
    stage2: fused = film_features + prev_scale * prev_up

Ablation (this config: False):
    stage2: fused = prev_scale * prev_up

Stage-2 backbone / FiLM features are still computed for unary mask scoring
(prototype matching on film_last). Only the direct addition into the decoded
feature stream is removed. Stages 4/3/1 are unchanged. No new parameters —
the same checkpoint loads for a controlled inference comparison.

Usage (evaluation example)::

    import train as train_cfg
    from interpretability_exp.exp_stage2_backbone_ablation import config as abl

    abl.apply(train_cfg)
    # then build model via evaluation.build_model / train construction
"""

from __future__ import annotations

from typing import Any

# Ablation: do not add stage-2 backbone/FiLM features into the decoder stream.
STAGE2_BACKBONE_FUSION_ENABLED = False

EXPERIMENT_NAME = "stage2_no_backbone_stream_fusion"
EXPERIMENT_DESCRIPTION = (
    "At stage2 only, set fused = prev_scale * prev_up before refine1/refine2. "
    "Keep stage2 FiLM features for unary priority-mask scoring. "
    "Stages 4/3/1, side heads, temporal aggregation, and losses unchanged."
)


def apply(train_cfg: Any) -> None:
    """Overlay this experiment's flags onto an imported train module."""
    train_cfg.STAGE2_BACKBONE_FUSION_ENABLED = STAGE2_BACKBONE_FUSION_ENABLED
