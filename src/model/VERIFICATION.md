# Verification of the consolidated model

The refactor passed 32 test groups. Original-versus-refactored comparisons used `rtol=0, atol=0` (with equal NaNs accepted for existing diagnostic statistics).

**29,064 tensor comparisons matched exactly**, including parameter states, outputs, losses, gradients, and updated weights across the tested configurations.

## Environment and limits

- PyTorch `2.5.1+cpu` on CPU.
- One CPU thread, deterministic algorithms enabled, autograd multithreading disabled.
- The unprovided custom Video Swin implementation was replaced only in the verification harness by the same deterministic synthetic backbone for both versions.
- The final test uses the real Swin-T stage channel widths (96/192/384/768) and original default concept, decoder, and temporal-infusion dimensions; the backbone remains synthetic.
- Fixed-bank tests construct actual NPY/NPZ/SQLite/manifest fixtures and run both supplied and refactored loaders. Cache tests exercise actual disk writes and hits.
- Neither the real Video Swin, an actual saved training checkpoint, DHF1K data, the external training driver, nor a GPU was available for integration testing.
- Wider CPU models can show floating-point accumulation differences under parallel execution; exact comparisons here use the deterministic settings above.

## What was checked

- Identical named learned parameters, registration order, shapes, initial values, requires-grad flags, and registered module hierarchy for paired configurations.
- Strict state_dict loading in both directions.
- Prediction maps/logits, side outputs, masks, unary/context scores, prototype assignments/representations, intermediate diagnostics, and every active loss term.
- Parameter/input gradients and equivalent SGD updates, including learned prototype updates, for the full-model scenarios.
- Training and evaluation, default and explicitly requested losses, disabled losses, and absent supervision.
- Mixed precision with CPU bfloat16 autocast.
- Fixed-reference tensors, fingerprints, member counts, explicit capped subsets, zero reference-query masking, flat RGB patches, and FP32 reference retention under `.half()`.
- RGB and feature appearance costs also checked against direct all-reference mean calculations, independently of the original code.
- Missing required cached features still raise the existing error.
- Diagnostic traces and CPU split-device placement.
- Frozen-feature cache miss/write/hit behavior, sample fingerprints, and feature provenance.
- Python compilation; Ruff undefined-name/import and unused-argument checks.

Always-None trajectory/gate outputs, duplicate appearance-diagnostic aliases, the disabled graph diagnostic, and two always-zero graph loss keys are intentionally absent and excluded from structural output-key equality. All remaining compared values match.

## Test groups

| Test group | Result |
| --- | --- |
| `default_unshared_training` | PASS |
| `shared_topk_training` | PASS |
| `infusion_disabled` | PASS |
| `stage2_backbone_disabled` | PASS |
| `side_fusion_disabled` | PASS |
| `last_only_infusion` | PASS |
| `no_temporal_difference` | PASS |
| `padded_topk` | PASS |
| `hard_assignments` | PASS |
| `backbone_trainable` | PASS |
| `raw_output` | PASS |
| `BCTHW_input` | PASS |
| `no_visual_residual` | PASS |
| `mask_strengths_zero` | PASS |
| `single_stage4` | PASS |
| `single_stage1` | PASS |
| `eval_losses_enabled` | PASS |
| `training_without_supervision_and_explicit_loss_skip` | PASS |
| `CPU_bfloat16_autocast_outputs_losses_gradients` | PASS |
| `diagnostic_trace_and_CPU_device_placement` | PASS |
| `concept_topk_soft_cap0` | PASS |
| `concept_topk_soft_cap2` | PASS |
| `concept_soft_cap0` | PASS |
| `concept_soft_cap2` | PASS |
| `concept_straight_through_cap0` | PASS |
| `concept_straight_through_cap2` | PASS |
| `concept_hard_eval_cap0` | PASS |
| `concept_hard_eval_cap2` | PASS |
| `appearance_costs_vs_explicit_reference_means` | PASS |
| `missing_cache_error_preserved` | PASS |
| `cache_miss_write_hit_and_provenance` | PASS |
| `Swin_T_channel_widths_and_default_head_dimensions` | PASS |

## Size and retained functionality

The six source files totaled 5,933 lines / 258,615 bytes. The two Python deliverables total 5,101 lines / 222,039 bytes. The reduction is deliberately conservative: reference-bank integrity checks, cache/provenance handling, supported ablations, and explanation diagnostics are functional code and were retained.

No learned layer was pruned. The unused nonpersistent temperature buffer, obsolete arguments/attributes, no-op methods, aliases, dead temporal-weight branches, duplicate group-selection helpers, zero graph losses, and redundant outputs were removed. Loss methods and their weighted sum moved to `losses.py`.

## Verified deliverable hashes

| File | SHA-256 |
| --- | --- |
| `model.py` | `ae9cb226ebba46bc658a04eb7862933bdd93bb0727ac0274b853fca8a59e1658` |
| `losses.py` | `48ec96e37a11ae23cb04253e10328c47b91da55224ddc9f23ccbd642b6b79e46` |

## Supplied-source hashes

| Source | SHA-256 |
| --- | --- |
| `backbone_feature_cache.py` | `674d430397b5230eec6157f8ffc9e2c8cc4ec377a513bc382de296f2705d665a` |
| `concept_creation(20261124-205929).py` | `a4b2cffc1e5e7c908d74383736e88368f6a67001ae91700fce30e38051a8e546` |
| `diagnostic_trace.py` | `eb8b43d879a0f19b0752a73ea014d65796b0f17c608112aecb29da39ff4493c1` |
| `model(20261124-205930).py` | `a80b9b4f8d944ecaac8549b5673e0ba36c72d655e3c550f87471739f7c989f30` |
| `saliency_prediction(20261124-205932).py` | `9b3567b9b34158f6f89ea940ae08200bceb3408294236311b044eab73e2eb3a8` |
| `temporal_feature_infusion(2).py` | `336f86f29e1e7df82c3601532f8b182ec000181d7ff728235c3e81f0c4fb1757` |
