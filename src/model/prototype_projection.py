"""Project all concept banks using a fixed, single-process training-data pass.

Default batches: {'rgb': tensor, 'saliency_maps': density_tensor,
                  'sample_ids': list_of_unique_window_ids,
                  'frame_indices': optional_list_of_target_frame_numbers}.
Pass batch_adapter(batch) to translate a custom loader's batch format.
Use deterministic preprocessing matching the gallery. No optimizer steps may
run concurrently with this sweep. Under DDP, run separately in one process,
save the projected checkpoint, then reload it in every training process.
"""

import torch


@torch.no_grad()
def project_prototypes(model, training_loader, optimizer=None, batch_adapter=None,
                       chunk_size=2048):
    """Return per-scale source reports and mutate only the prototype banks.

    References are unnecessary for projection: actual live encoded patches are
    compared with prototypes within their GT attention roles. Project after
    training (or between optimizer steps), never during a gradient-bearing pass.
    The caller saves reports together with model.state_dict() afterward.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        raise RuntimeError("Run projection in one process over the full training set.")
    if hasattr(model, "module"):
        model = model.module
    if not hasattr(model, "concept_creations"):
        raise TypeError("Expected ExplainableVidSalModel with per-scale concept_creations.")
    if any(module._projection_state is not None for module in model.concept_creations.values()):
        raise RuntimeError("An existing projection sweep is already active.")
    modes = [(module, module.training) for module in model.modules()]
    versions = [(parameter, parameter._version) for parameter in model.parameters()]
    rollback = {stage: module.visual_concepts.detach().clone()
                for stage, module in model.concept_creations.items()}
    # Momentum is only cleared after all banks have committed successfully.
    committed = False
    try:
        model.eval()
        for module in model.concept_creations.values():
            module.begin_prototype_projection()
        batch_count = 0
        for raw_batch in training_loader:
            batch = raw_batch if batch_adapter is None else batch_adapter(raw_batch)
            if not isinstance(batch, dict) or not all(
                    name in batch for name in ("rgb", "saliency_maps", "sample_ids")):
                raise ValueError("Projection batches need rgb, saliency_maps, sample_ids.")
            if isinstance(batch["sample_ids"], (str, bytes)):
                raise ValueError("sample_ids must be a sequence of unique window identifiers.")
            if any(parameter._version != version for parameter, version in versions):
                raise RuntimeError("Model weights changed during the projection sweep.")
            result = model(batch["rgb"], return_details=True,
                           return_concept_losses=False, use_reference_cache=False)
            for stage, module in model.concept_creations.items():
                module.update_prototype_projection(
                    result["concept_out"][stage], batch["saliency_maps"],
                    batch["sample_ids"], frame_indices=batch.get("frame_indices"),
                    chunk_size=chunk_size)
            batch_count += 1
        if batch_count == 0:
            raise ValueError("Projection training_loader was empty.")
        if any(parameter._version != version for parameter, version in versions):
            raise RuntimeError("Model weights changed during the projection sweep.")
        reports = {stage: module.finalize_prototype_projection()
                   for stage, module in model.concept_creations.items()}
        committed = True
        if optimizer is not None:
            for stage, module in model.concept_creations.items():
                matched = torch.tensor([source is not None for source in reports[stage]["sources"]],
                                       device=module.visual_concepts.device)
                for value in optimizer.state.get(module.visual_concepts, {}).values():
                    if torch.is_tensor(value) and value.shape == module.visual_concepts.shape:
                        value[matched.to(value.device)] = 0
        return reports
    except Exception:
        if not committed:
            for stage, module in model.concept_creations.items():
                module.visual_concepts.copy_(rollback[stage])
        raise
    finally:
        for module in model.concept_creations.values():
            module.cancel_prototype_projection()
        # Restore individual mode flags, including an intentionally frozen stage.
        for module, training in modes:
            module.training = training
