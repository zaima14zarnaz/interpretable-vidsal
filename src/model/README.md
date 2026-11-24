# Explainable Video Saliency: consolidated model

The six supplied Python files are replaced by **model.py + losses.py**. The custom Video Swin backbone remains in its existing location, as requested.

The active model architecture, learned parameter registration/order, initialization, tensor operations, loss formulas/weights/reductions, and default prediction behavior are preserved. Obsolete interfaces were removed deliberately; update callers using the migration notes below. This is not an architecture redesign.

## Install

1. Put these files at `model/model.py` and `model/losses.py` in the existing project.
2. Keep the existing `model/backbones/video_swin_custom.py` and its dependencies, weights, and package initialization files.
3. Update imports in training, evaluation, explanation, and cache-generation scripts before removing the old module files.
4. Keep the existing fixed patch-cluster bank and cache. Their formats and defaults are retained.

The model still needs PyTorch, NumPy, and the existing backbone dependencies. SQLite support comes from Python's standard library.

```python
from model.model import ExplainableVidSalModel

model = ExplainableVidSalModel(
    visual_cluster_dir="/path/to/the/existing/patch_cluster",
    # Keep your existing values for the remaining active options.
)
```

For inference, the public `forward` contract remains:

```python
model.eval()
saliency = model(video)  # [B, 1, H, W]

# Existing diagnostic maps and explanation tensors remain available.
details = model(
    video,
    return_details=True,
    return_decoder_diagnostics=True,
)
```

For training, preserve the existing optimizer, scheduler, target preparation, and saliency objectives in your training script:

```python
model.train()
details = model(
    video,
    saliency_maps=ground_truth,
    return_details=True,
    cached_backbone_features=reference_features,
)
concept_losses = {
    stage: output["losses"] for stage, output in details["concept_out"].items()
}
```

As before, stage4 appearance supervision requires the loaded fixed reference bank and detached reference features when its feature term is enabled. The model does not create the cache or cluster bank during its prediction forward pass.

## Code organization

| File / section | Responsibility |
| --- | --- |
| `model.py`: `ExplainableVidSalModel` | Public model, devices, input handling, backbone, per-stage execution |
| `model.py`: temporal infusion | Existing factorized temporal/spatial 3D residual blocks |
| `model.py`: `VisualConceptCreation` | Patch encoding, prototype matching, immutable reference loading and identity |
| `model.py`: decoder | FiLM, multiscale fusion/refinement, unary/pairwise scores, masks and prediction heads |
| `model.py`: diagnostic tracing | Optional intermediate tensor capture |
| `model.py`: reference cache | Frozen backbone cache, sample identity, provenance, checkpoint hashing |
| `losses.py`: `VisualConceptLossMixin` | Every loss calculation present in the uploaded code, associated target descriptors and diagnostics, and weighted concept total |

The loss mixin has **no registered neural layers or parameters**. `visual_saliency_head` remains registered under each concept module because moving its weights into another module would change checkpoint names. Its evaluation for auxiliary supervision and BCE calculation occur in `losses.py`.

Fixed cluster arrays are training targets, not the learned prototypes. Their validation and FP32/device handling remain necessary, and their memberships remain unchanged during forward/backward. Learned prototypes still update through the existing gradients and optimizer.

## Preserved computation

- The backbone's raw features feed visual concepts; the existing temporal infusion feeds decoder features.
- Visual assignments use the last feature slice. Full-bank probabilities feed the existing regularizers and appearance loss; sparse top-k assignments feed prototype representations and prediction.
- Existing assignment aliases, explicit hard assignment mode, top-k padding, validity masks, and shared/non-shared decoder assignments are retained.
- Existing reconstruction, prototype diversity, assignment entropy/usage, auxiliary saliency alignment, and stage4 appearance terms keep their original arithmetic and weighting.
- Appearance supervision retains exact mean RGB MAE using sorted values/prefix sums, mean normalized-feature cosine costs, centered RGB structure costs, and exclusion of invalid cached queries.
- Decoder features still pass through projection, prototype-derived FiLM, previous-stage fusion, 3D refinement, and residual last-slice spatial masks. Stages3/4 use unary + pairwise priorities; stages1/2 use unary priorities.
- Main and side heads still read the last decoder slice. Side-logit fusion and stage2 backbone-fusion ablations remain available.
- The same learned layers are instantiated in the same order, including layers used for auxiliary outputs or supported ablations.

## Import migration and obsolete files

After updating callers, the following separate files are no longer needed **for the supplied model**. Their retained functionality now lives in `model.py` and `losses.py`:

| Old import | New import |
| --- | --- |
| `model.model.ExplainableVidSalModel` | unchanged |
| `model.concept_creation.VisualConceptCreation` | `model.model.VisualConceptCreation` |
| `model.concept_creation.ConceptCreation` | `model.model.VisualConceptCreation` |
| `model.saliency_prediction.ConceptGatedMultiScaleSaliencyDecoder` | `model.model.ConceptGatedMultiScaleSaliencyDecoder` |
| `model.temporal_feature_infusion.SpatioTemporal3DFeatureInfusion` | `model.model.SpatioTemporal3DFeatureInfusion` |
| `model.temporal_feature_infusion.SameLocationPatchTemporalTransformer` | `model.model.SpatioTemporal3DFeatureInfusion` |
| `model.backbone_feature_cache.<name>` | `model.model.<name>` |
| `model.diagnostic_trace.capture_diagnostic_trace` | `model.model.capture_diagnostic_trace` |
| `model.diagnostic_trace.diagnostic_trace_enabled` | `model.model.diagnostic_trace_enabled` |
| `model.diagnostic_trace.record` | `model.model._diag_record` (internal tracing helper) |
| `model.concept_creation.build_patch_cluster_reference_checkpoint_metadata` | `model.model.build_patch_cluster_reference_checkpoint_metadata` |

This removes the need for the old `concept_creation.py`, `saliency_prediction.py`, `temporal_feature_infusion.py`, `diagnostic_trace.py`, and `backbone_feature_cache.py`. No repository or training scripts were supplied, so imports outside these six files have not been edited. Original uploaded source files have not been overwritten by this bundle.

## Removed model options

Remove the following keyword arguments from `ExplainableVidSalModel(...)` calls/config dictionaries. They did not select a different active computation in the supplied code:

- `max_source_patches`, `tau_pi`, `tau_alpha`, `tau_concept`, `concept_residual_weight`
- `last_transition_only`, `use_temporal_transition_aggregation`
- `use_feature_refinement`, `feature_refine_channels`, `use_rgb_refinement`
- `use_gated_trajectory_head`, `gated_trajectory_residual_scale`
- `use_subpatch_head`, `subpatch_factor`, `subpatch_hidden_dim`, `subpatch_residual_scale`
- `visual_concept_on`, `temporal_concepts_on` (the supplied implementation always constructed/executed its visual concept modules)
- `decoder_temporal_aggregation` (the supplied decoder always selected the last slice)
- `prototype_application_position` (the supplied decoder always masked after refinement)

The permissive `**_deprecated_saliency_kwargs` sink is gone: unknown options now raise `TypeError` rather than being ignored. Call arguments should use keywords; removing obsolete positional arguments changes positional indices.

`collect_gate_debug` was removed from the model and concept-module forward signatures; it had no effect. `backbone_stage` remains because it selects the stage when `backbone_stages=None`.

Direct `VisualConceptCreation(...)` callers must also remove old `eps_*`, `gate_temp_*`, `gate_min_conf`, `use_target_centric`, `visual_graph_*`, `visual_queue_size`, and `visual_queue_min_confidence` options. Direct decoder callers must remove `feature_residual_scale`, `tau_pi`, `distance_gate_range`, `temporal_aggregation`, and `prototype_application_position`. Direct fusion-block callers must remove `max_strength` and `prototype_application_position`. Direct temporal-infusion callers must remove `num_heads`, `max_temporal_len`, and arbitrary deprecated kwargs. `estimate_entry_bytes()` no longer accepts the ignored `sample_hw` argument; its original fixed-grid estimate is unchanged.

The unused nonpersistent `inv_tau_concept` buffer and redundant configuration attributes were removed. No learned parameters were removed. The all-None concept-grid resize hook was removed; the supplied configuration never resized any concept grid.

## Reference-helper migration

FIFO-named aliases were removed. Use the fixed-reference names:

| Old name | Retained name |
| --- | --- |
| `clear_prototype_queue` | `reset_appearance_forward_counter` |
| `prototype_queue_state` | `fixed_reference_identity_state` |
| `load_prototype_queue_state` | `load_fixed_reference_identity_state` |
| `save_prototype_queue` | `save_fixed_reference_identity` |
| `save_prototype_queues` | `save_fixed_reference_identities` |
| `load_prototype_queues` | `load_fixed_reference_identities` |
| `DEFAULT_PROTOTYPE_QUEUE_DIR` | `DEFAULT_REFERENCE_SNAPSHOT_DIR` |

These snapshot helpers keep their original default directory/filenames and formats, validate reference identity, and restore counters only. They do not restore or mutate reference patches. Compatibility with existing on-disk feature-cache formats/fingerprints is retained because it affects whether cached targets are accepted.

Removed no-op methods: `configure_visual_graph`, `_enqueue_rgb_descriptors`, and `summarize_gate_debug`. The unused `_visual_appearance_loss` compatibility wrapper is removed; the active component calculation is `_visual_appearance_losses` in `losses.py`.

## Returned-output migration

Remove logger/report accesses to these retired concept output keys:

- Always-None trajectory/gate placeholders: `trajectory_vectors`, `trajectory_embeddings`, `concept_representation`, `transition_activations`, `persistence_activations`, `gate_probs`, `metadata`. Use `visual_metadata` for the active visual grid.
- The disabled `visual_graph_diagnostics` entry.
- Duplicate `visual_queue_diagnostics` and `visual_patch_cluster_diagnostics` aliases. Use `visual_appearance_diagnostics`.
- Always-zero `loss_visual_graph_assignment` and `loss_visual_graph_feature`. Their removal does not change `loss_total_concept`.

Active prediction/priority/side-output keys and `loss_total_concept` are preserved. Decoder diagnostic aliases were retained to avoid unnecessary changes to the explanation/report interface.

## Checkpoint compatibility and verification

Use existing **state_dict** checkpoints with the same active dimensions and options. Learned parameter keys/shapes/order remain unchanged. Load through your existing checkpoint extraction code with `strict=True`. Whole-object pickles can still refer to removed Python module paths; the compatibility claim concerns state_dict weights.

`VERIFICATION.md` records the completed equivalence checks and their limits. The custom Video Swin source/weights, real training driver, dataset, and GPU were not available, so those end-to-end integrations were not executed here. No external saliency-training objective was invented or changed: `losses.py` contains the losses present in the supplied model files.
