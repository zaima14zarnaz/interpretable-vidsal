"""Coverage diagnostics + loss/grad/queue identity checks for visual appearance."""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# concept_creation uses a relative import for diagnostic_trace; load as a package.
if "model" not in sys.modules:
    import types
    _pkg = types.ModuleType("model")
    _pkg.__path__ = [str(_HERE)]
    sys.modules["model"] = _pkg
_diag_spec = importlib.util.spec_from_file_location(
    "model.diagnostic_trace", _HERE / "diagnostic_trace.py"
)
_diag = importlib.util.module_from_spec(_diag_spec)
sys.modules["model.diagnostic_trace"] = _diag
_diag_spec.loader.exec_module(_diag)
_cc_spec = importlib.util.spec_from_file_location(
    "model.concept_creation", _HERE / "concept_creation.py"
)
_cc = importlib.util.module_from_spec(_cc_spec)
sys.modules["model.concept_creation"] = _cc
_cc_spec.loader.exec_module(_cc)
VisualConceptCreation = _cc.VisualConceptCreation


def _make_module(**kwargs) -> VisualConceptCreation:
    defaults = dict(
        in_channels=8,
        concept_dim=16,
        num_concepts=4,
        num_visual_concepts=4,
        hidden_dim=32,
        top_k=2,
        visual_queue_size=4,
        visual_queue_min_confidence=0.0,
        visual_appearance_patch_size=2,
        use_visual_saliency_alignment=False,
        visual_saliency_align_weight=0.0,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return VisualConceptCreation(**defaults)


def _inputs(module: VisualConceptCreation, B=2, H=3, W=4, T=3, device="cpu"):
    C = module.in_channels
    features = torch.randn(B, C, T, H, W, device=device, requires_grad=False)
    rgb = torch.rand(B, 3, H * 8, W * 8, device=device)
    cached = torch.randn(B, C, 1, H, W, device=device)
    return features, rgb, cached


def _diag_counts(module: VisualConceptCreation) -> Dict[str, float]:
    d = module.visual_appearance_diagnostics
    return {
        "total": float(d["app_loss_patch_count_total"]),
        "evaluated": float(d["app_loss_patch_count"]),
        "rgb": float(d["app_loss_rgb_patch_count"]),
        "feature": float(d["app_loss_feature_patch_count"]),
    }


def _queue_fingerprint(module: VisualConceptCreation) -> Dict[str, torch.Tensor]:
    return {
        "rgb": module._prototype_queue.detach().clone(),
        "rgb_sum": module._prototype_queue_sum.detach().clone(),
        "rgb_sq": module._prototype_queue_squared_norm_sum.detach().clone(),
        "rgb_count": module._prototype_queue_count.detach().clone(),
        "rgb_ptr": module._prototype_queue_pointer.detach().clone(),
        "feat": module._prototype_feature_queue.detach().clone(),
        "feat_sum": module._prototype_feature_queue_sum.detach().clone(),
        "feat_count": module._prototype_feature_count.detach().clone(),
        "feat_valid": module._prototype_feature_valid.detach().clone(),
        "step": int(module._prototype_queue_step),
    }


def _assert_queues_equal(a: Dict[str, Any], b: Dict[str, Any], label: str) -> None:
    for k in a:
        if k == "step":
            assert a[k] == b[k], f"{label}: step mismatch {a[k]} vs {b[k]}"
            continue
        assert torch.equal(a[k], b[k]) or torch.allclose(a[k].float(), b[k].float(), atol=0, rtol=0), (
            f"{label}: queue field {k} diverged"
        )


def _uninstrumented_appearance_losses(
    module: VisualConceptCreation,
    visual_out: Dict[str, Any],
    rgb_last_frame: Optional[torch.Tensor],
    cached_backbone_features: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Exact pre-diagnostics loss body (no coverage counters)."""
    logits = visual_out["visual_concept_logits"]
    zero = logits.sum() * 0.0
    result = {"rgb": zero, "feature": zero, "combined": zero}
    w_rgb, w_feature = module.loss_weights["visual_rgb"], module.loss_weights["visual_feature"]
    reason = None
    if not module.training:
        reason = "evaluation"
    elif not torch.is_grad_enabled():
        reason = "no_grad"
    elif module.loss_weights["visual_appearance"] == 0 or (w_rgb == 0 and w_feature == 0):
        reason = "zero_weight"
    elif rgb_last_frame is None:
        reason = "no_rgb"
    elif w_rgb == 0 and cached_backbone_features is None:
        reason = "no_cached_features"
    if reason is not None:
        return result
    shape = visual_out["visual_metadata"]["feature_shape"]
    descriptors = module._rgb_appearance_descriptors(rgb_last_frame, int(shape["H"]), int(shape["W"]))
    feature_descriptors = feature_valid = None
    if w_feature > 0 and cached_backbone_features is not None:
        feature_descriptors, feature_valid = module._cached_feature_descriptors(cached_backbone_features)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        if w_rgb > 0:
            divisor = module._prototype_queue_count.clamp_min(1).float()
            means = module._prototype_queue_sum / divisor[:, None]
            mean_squared_norm = module._prototype_queue_squared_norm_sum / divisor
            rgb_costs = (descriptors.square().sum(-1, keepdim=True)
                         - 2.0 * descriptors @ means.T + mean_squared_norm[None, :])
            rgb_costs = (rgb_costs / module.visual_appearance_dim).clamp_min(0.0).detach()
            weights, has_history = module._queue_assignment_weights(logits, module._prototype_queue_count > 0)
            result["rgb"] = (weights * rgb_costs).sum(-1).mean() * has_history
        if feature_descriptors is not None:
            means = module._prototype_feature_queue_sum / module._prototype_feature_count.clamp_min(1).float()[:, None]
            feature_costs = (1.0 - feature_descriptors @ means.T).clamp(0.0, 2.0).detach()
            feature_weights, feature_history = module._queue_assignment_weights(
                logits, module._prototype_feature_count > 0
            )
            query_valid = feature_valid.to(torch.float32)
            result["feature"] = ((feature_weights * feature_costs).sum(-1) * query_valid).sum() / query_valid.sum().clamp_min(1.0)
            result["feature"] = result["feature"] * feature_history
        result["combined"] = w_rgb * result["rgb"] + w_feature * result["feature"]
    module._enqueue_rgb_descriptors(visual_out, descriptors, feature_descriptors, feature_valid)
    return result


def _clone_module_state(src: VisualConceptCreation) -> VisualConceptCreation:
    dst = _make_module(
        in_channels=src.in_channels,
        concept_dim=src.concept_dim,
        num_concepts=src.num_concepts,
        num_visual_concepts=src.num_visual_concepts,
        hidden_dim=src.hidden_dim,
        top_k=src.top_k,
        visual_queue_size=src.visual_queue_size,
        visual_queue_min_confidence=src.visual_queue_min_confidence,
        visual_appearance_patch_size=src.visual_appearance_patch_size,
        use_visual_saliency_alignment=False,
        visual_saliency_align_weight=0.0,
        loss_weights=dict(src.loss_weights),
    )
    dst.load_state_dict(src.state_dict())
    dst.load_prototype_queue_state(src.prototype_queue_state())
    return dst


def test_empty_history_warmup():
    m = _make_module()
    m.train()
    features, rgb, cached = _inputs(m)
    out = m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    n = 2 * 3 * 4
    c = _diag_counts(m)
    assert c["total"] == n
    assert c["evaluated"] == 0 and c["rgb"] == 0 and c["feature"] == 0
    assert abs(float(out["losses"]["loss_visual_appearance"].detach())) == 0.0
    assert m._prototype_queue_count.sum().item() > 0  # enqueue still ran
    print("PASS empty-history warmup")


def test_populated_history_all_queries():
    m = _make_module()
    m.train()
    features, rgb, cached = _inputs(m)
    # Warm queues
    m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    out = m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    n = 2 * 3 * 4
    c = _diag_counts(m)
    assert c["total"] == n
    assert c["rgb"] == n
    assert c["feature"] == n  # all cached vectors nonzero from randn
    assert c["evaluated"] == n
    assert float(out["losses"]["loss_visual_appearance"].detach()) >= 0.0
    print("PASS populated history covering all queries")


def test_missing_cached_features():
    m = _make_module()
    m.train()
    features, rgb, cached = _inputs(m)
    m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    out = m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=None)
    n = 2 * 3 * 4
    c = _diag_counts(m)
    assert c["total"] == n
    assert c["rgb"] == n
    assert c["feature"] == 0
    assert c["evaluated"] == n
    assert abs(float(out["losses"]["loss_visual_feature"].detach())) == 0.0
    print("PASS missing cached features")


def test_zero_norm_feature_queries():
    m = _make_module()
    m.train()
    features, rgb, cached = _inputs(m)
    m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    cached_zero = torch.zeros_like(cached)
    # Keep half the patches nonzero so feature history exists from warmup and
    # only zero-norm live queries are excluded.
    cached_zero[:, :, :, :2, :] = cached[:, :, :, :2, :]
    out = m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached_zero)
    n = 2 * 3 * 4
    # Valid live queries: B * 2 * W = 2*2*4 = 16? H=3 rows; first 2 rows kept -> 2*2*4=16
    expected_feat = 2 * 2 * 4
    c = _diag_counts(m)
    assert c["total"] == n
    assert c["rgb"] == n
    assert c["feature"] == expected_feat
    assert c["evaluated"] == n  # RGB covers all
    assert "loss_visual_feature" in out["losses"]
    print("PASS zero-norm feature queries")


def test_zero_loss_nonzero_coverage():
    """Constant RGB → ~0 MSE while coverage remains full (do not infer from loss!=0)."""
    m = _make_module(loss_weights={"visual_feature": 0.0})
    m.train()
    B, H, W, T = 1, 2, 2, 2
    features = torch.randn(B, m.in_channels, T, H, W)
    rgb = torch.full((B, 3, H * 8, W * 8), 0.4)
    m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=None)
    out = m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=None)
    n = B * H * W
    c = _diag_counts(m)
    assert c["rgb"] == n and c["evaluated"] == n and c["feature"] == 0
    rgb_loss = float(out["losses"]["loss_visual_appearance_rgb"].detach())
    assert abs(rgb_loss) < 1e-6, rgb_loss
    assert c["evaluated"] > 0  # coverage independent of near-zero loss
    print("PASS legitimate zero loss with nonzero coverage")


def test_disabled_and_eval():
    m = _make_module()
    m.train()
    features, rgb, cached = _inputs(m)
    m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    n = 2 * 3 * 4

    m.eval()
    m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    c = _diag_counts(m)
    assert c["evaluated"] == 0 and c["rgb"] == 0 and c["feature"] == 0 and c["total"] == n
    assert m.visual_appearance_diagnostics["reason"] == "evaluation"

    m.train()
    with torch.no_grad():
        m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    c = _diag_counts(m)
    assert c["evaluated"] == 0 and m.visual_appearance_diagnostics["reason"] == "no_grad"

    m.loss_weights["visual_appearance"] = 0.0
    m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    c = _diag_counts(m)
    assert c["evaluated"] == 0 and m.visual_appearance_diagnostics["reason"] == "zero_weight"

    m.loss_weights["visual_appearance"] = 1.0
    m(features, return_losses=False, rgb_last_frame=rgb, cached_backbone_features=cached)
    c = _diag_counts(m)
    assert c["evaluated"] == 0 and m.visual_appearance_diagnostics["reason"] == "return_losses_false"
    assert c["total"] == n
    print("PASS disabled losses and evaluation")


def test_instrumented_matches_uninstrumented():
    """Same visual_out + queue start → identical losses, grads, and queue writes."""
    m_inst = _make_module()
    m_ref = _clone_module_state(m_inst)
    m_inst.train()
    m_ref.train()
    features, rgb, cached = _inputs(m_inst)
    # Warm queues once, then sync.
    m_inst(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=cached)
    m_ref.load_prototype_queue_state(m_inst.prototype_queue_state())
    pre = _queue_fingerprint(m_inst)

    def _restore(mod, fp):
        mod._prototype_queue.copy_(fp["rgb"])
        mod._prototype_queue_sum.copy_(fp["rgb_sum"])
        mod._prototype_queue_squared_norm_sum.copy_(fp["rgb_sq"])
        mod._prototype_queue_count.copy_(fp["rgb_count"])
        mod._prototype_queue_pointer.copy_(fp["rgb_ptr"])
        mod._prototype_feature_queue.copy_(fp["feat"])
        mod._prototype_feature_queue_sum.copy_(fp["feat_sum"])
        mod._prototype_feature_count.copy_(fp["feat_count"])
        mod._prototype_feature_valid.copy_(fp["feat_valid"])
        mod._prototype_queue_step = fp["step"]

    # Shared live assignments (bypass dropout nondeterminism between modules).
    m_inst.train()
    m_ref.train()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        visual_out = m_inst._build_visual_concepts_from_patches(features)
    shared = {
        "visual_concept_logits": visual_out["visual_concept_logits"].detach(),
        "visual_assignment_probs": visual_out["visual_assignment_probs"].detach(),
        "visual_concept_indices": visual_out["visual_concept_indices"].detach(),
        "visual_metadata": visual_out["visual_metadata"],
        "visual_patch_coords": visual_out["visual_patch_coords"],
    }

    _restore(m_inst, pre)
    _restore(m_ref, pre)

    logits_i = shared["visual_concept_logits"].detach().requires_grad_(True)
    shared_i = dict(shared)
    shared_i["visual_concept_logits"] = logits_i
    app_i = m_inst._visual_appearance_losses(shared_i, rgb, cached)
    app_i["combined"].backward()
    post_i = _queue_fingerprint(m_inst)
    grad_i = logits_i.grad.detach().clone()

    _restore(m_ref, pre)
    logits_r = shared["visual_concept_logits"].detach().requires_grad_(True)
    shared_r = dict(shared)
    shared_r["visual_concept_logits"] = logits_r
    app_r = _uninstrumented_appearance_losses(m_ref, shared_r, rgb, cached)
    app_r["combined"].backward()
    post_r = _queue_fingerprint(m_ref)
    grad_r = logits_r.grad.detach().clone()

    assert torch.allclose(app_i["combined"], app_r["combined"])
    assert torch.allclose(app_i["rgb"], app_r["rgb"])
    assert torch.allclose(app_i["feature"], app_r["feature"])
    assert torch.allclose(grad_i, grad_r), "logits grads diverged"
    _assert_queues_equal(post_i, post_r, "instrumented vs uninstrumented")
    assert "app_loss_patch_count" in m_inst.visual_appearance_diagnostics
    print("PASS instrumented ≡ uninstrumented (loss/grad/queue)")


if __name__ == "__main__":
    test_empty_history_warmup()
    test_populated_history_all_queries()
    test_missing_cached_features()
    test_zero_norm_feature_queries()
    test_zero_loss_nonzero_coverage()
    test_disabled_and_eval()
    test_instrumented_matches_uninstrumented()
    print("ALL TESTS PASSED")
else:
    pass
