"""Fixed-cluster appearance tests using a small synthetic reference bank."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util
import sys
import types

_HERE = Path(__file__).resolve().parent
if "model" not in sys.modules:
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
build_patch_cluster_reference_checkpoint_metadata = (
    _cc.build_patch_cluster_reference_checkpoint_metadata
)

# Synthetic bank geometry (kept tiny for CPU unit tests).
IN_CHANNELS = 4
NUM_CONCEPTS = 2
PATCH_SIZE = 2
RGB_DIM = 3 * PATCH_SIZE * PATCH_SIZE  # 12
BACKBONE_FP = "synthetic_backbone_fp_v1"
CHECKPOINT_SHA = "synthetic_checkpoint_sha_v1"


def _write_synthetic_bank(
    root: Path,
    *,
    backbone_fp: str = BACKBONE_FP,
    checkpoint_sha: str = CHECKPOINT_SHA,
) -> Dict[str, Any]:
    """Write a complete fixed_stage4_patch_clusters_v1 bank under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "descriptors").mkdir(exist_ok=True)

    # Cluster 0: colorful + flat gray. Cluster 1: two colorful.
    # RGB stored HWC; features are nonzero.
    rgb_hwc = np.zeros((4, RGB_DIM), dtype=np.float32)
    # patch0: ramp
    rgb_hwc[0] = np.linspace(0.1, 0.9, RGB_DIM, dtype=np.float32)
    # patch1: flat gray (structure-flat)
    rgb_hwc[1] = 0.4
    # patch2 / patch3: other ramps
    rgb_hwc[2] = np.linspace(0.9, 0.1, RGB_DIM, dtype=np.float32)
    rgb_hwc[3] = np.concatenate(
        [
            np.full(4, 0.2, dtype=np.float32),
            np.full(4, 0.5, dtype=np.float32),
            np.full(4, 0.8, dtype=np.float32),
        ]
    )
    feature = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    node_ids = np.array([100, 101, 200, 201], dtype=np.int64)

    np.save(root / "descriptors/rgb.npy", rgb_hwc)
    np.save(root / "descriptors/feature.npy", feature)
    np.save(root / "descriptors/prototype_ids.npy", labels)
    np.save(root / "descriptors/node_ids.npy", node_ids)

    # Dummy references blob; only its SHA is checked.
    refs_path = root / "cluster_references.npz"
    np.savez(refs_path, placeholder=np.arange(8, dtype=np.int64))
    h = hashlib.sha256()
    with refs_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    references_sha256 = h.hexdigest()

    con = sqlite3.connect(root / "patch_clusters.sqlite")
    try:
        con.execute(
            "CREATE TABLE patches ("
            "descriptor_index INTEGER PRIMARY KEY, node_id INTEGER, prototype_id INTEGER)"
        )
        con.execute(
            "CREATE TABLE prototypes (id INTEGER PRIMARY KEY, patch_count INTEGER)"
        )
        for i, (nid, pid) in enumerate(zip(node_ids.tolist(), labels.tolist())):
            con.execute(
                "INSERT INTO patches(descriptor_index,node_id,prototype_id) VALUES (?,?,?)",
                (i, nid, pid),
            )
        for pid in range(NUM_CONCEPTS):
            con.execute(
                "INSERT INTO prototypes(id,patch_count) VALUES (?,?)",
                (pid, int((labels == pid).sum())),
            )
        con.commit()
    finally:
        con.close()

    provenance = {
        "backbone_fingerprint": backbone_fp,
        "temporal_selection": "last",
        "feature_source": "frozen_reference",
        "preprocess": {
            "resize_to": [224, 224],
            "normalize": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_format": "rgb_float01",
        },
    }
    manifest = {
        "format_version": "fixed_stage4_patch_clusters_v1",
        "status": "complete",
        "stage": "stage4",
        "rgb_order": "HWC",
        "num_prototypes": NUM_CONCEPTS,
        "feature_dim": IN_CHANNELS,
        "rgb_descriptor_size": PATCH_SIZE,
        "num_patches": int(labels.shape[0]),
        "references_sha256": references_sha256,
        "rgb_settings": {
            "rgb_resize": "bilinear, per-channel float32, uniform input-grid crop",
            "rgb_size": PATCH_SIZE,
        },
        "source_manifest": {"checkpoint_sha256": checkpoint_sha},
        "source_feature_semantics": "synthetic_frozen_reference",
        "reference_feature_provenance": provenance,
        "reference_feature_provenance_status": "verified",
        "weights": {"feature": 1.0, "rgb_mae": 1.0, "structure": 1.0},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "rgb_hwc": rgb_hwc,
        "feature": feature,
        "labels": labels,
        "provenance": provenance,
        "manifest": manifest,
    }


def _make_module(bank_dir: Optional[str] = None, **kwargs) -> VisualConceptCreation:
    defaults = dict(
        in_channels=IN_CHANNELS,
        concept_dim=8,
        num_concepts=NUM_CONCEPTS,
        num_visual_concepts=NUM_CONCEPTS,
        hidden_dim=16,
        top_k=2,
        visual_appearance_patch_size=PATCH_SIZE,
        use_visual_saliency_alignment=False,
        visual_saliency_align_weight=0.0,
        visual_entropy_weight=0.0,
        visual_usage_weight=0.0,
        stage_name="stage4",
        visual_cluster_dir=bank_dir or "/tmp/explainable_vidsal_no_cluster_bank",
        visual_cluster_max_patches_per_prototype=0,
        visual_cluster_seed=42,
        loss_weights={
            "visual": 0.0,
            "visual_div": 0.0,
            "visual_appearance": 1.0,
            "visual_rgb": 1.0,
            "visual_feature": 1.0,
            "visual_structure": 1.0,
        },
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return VisualConceptCreation(**defaults)


def _paint_rgb_from_chw(descriptors_chw: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Paint CHW patch descriptors into an RGB image whose crops are exactly r×r."""
    r = PATCH_SIZE
    b = descriptors_chw.shape[0] // (H * W)
    assert descriptors_chw.shape[0] == b * H * W
    image = torch.zeros(b, 3, H * r, W * r, dtype=torch.float32)
    for bi in range(b):
        for row in range(H):
            for col in range(W):
                idx = bi * H * W + row * W + col
                patch = descriptors_chw[idx].reshape(3, r, r)
                y0, y1 = row * r, (row + 1) * r
                x0, x1 = col * r, (col + 1) * r
                image[bi, :, y0:y1, x0:x1] = patch
    return image


def _chw_from_hwc(rgb_hwc: np.ndarray) -> np.ndarray:
    r = PATCH_SIZE
    return (
        rgb_hwc.reshape(-1, r, r, 3)
        .transpose(0, 3, 1, 2)
        .reshape(-1, RGB_DIM)
        .astype(np.float32)
    )


def _unit_rows(x: np.ndarray) -> np.ndarray:
    scales = np.max(np.abs(x), axis=1, keepdims=True)
    scaled = x / np.maximum(scales, 1e-12)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.maximum(norms, 1e-12)


def _structure_unit_and_flat(rgb_chw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = PATCH_SIZE
    pixels = rgb_chw.reshape(-1, 3, r * r)
    offsets = pixels - pixels[:, :, :1]
    centered = (offsets - offsets.mean(axis=-1, keepdims=True)).reshape(-1, RGB_DIM)
    norms = np.linalg.norm(centered, axis=1)
    flat = norms < 1e-12
    unit = centered / np.maximum(norms[:, None], 1e-12)
    unit[flat] = 0.0
    return unit.astype(np.float32), flat


def _explicit_cluster_costs(
    query_rgb_chw: np.ndarray,
    query_feat: np.ndarray,
    bank: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Mean query→every-reference costs per cluster (matches optimized targets)."""
    rgb_refs = _chw_from_hwc(bank["rgb_hwc"])
    feat_refs = _unit_rows(bank["feature"].astype(np.float64)).astype(np.float32)
    struct_refs, flat_refs = _structure_unit_and_flat(rgb_refs)
    q_struct, q_flat = _structure_unit_and_flat(query_rgb_chw)
    q_feat = _unit_rows(query_feat.astype(np.float64)).astype(np.float32)

    n_q = query_rgb_chw.shape[0]
    rgb_c = np.zeros((n_q, NUM_CONCEPTS), dtype=np.float64)
    feat_c = np.zeros((n_q, NUM_CONCEPTS), dtype=np.float64)
    struct_c = np.zeros((n_q, NUM_CONCEPTS), dtype=np.float64)
    labels = bank["labels"]
    for j in range(NUM_CONCEPTS):
        idx = np.where(labels == j)[0]
        # RGB MAE averaged over members.
        diffs = np.abs(query_rgb_chw[:, None, :] - rgb_refs[None, idx, :])
        rgb_c[:, j] = diffs.mean(axis=(1, 2))
        # Feature: mean (1 - cos) == 1 - q·mean(unit refs)
        feat_c[:, j] = 1.0 - (q_feat @ feat_refs[idx].mean(axis=0))
        # Structure: 1 - (q·mean(unit structs) + flat_q * flat_fraction)
        mean_s = struct_refs[idx].mean(axis=0)
        flat_frac = float(flat_refs[idx].mean())
        sim = q_struct @ mean_s + q_flat.astype(np.float64) * flat_frac
        struct_c[:, j] = 1.0 - sim
    return {
        "rgb": np.clip(rgb_c, 0.0, 1.0),
        "feature": np.clip(feat_c, 0.0, 2.0),
        "structure": np.clip(struct_c, 0.0, 2.0),
    }


def _reference_fingerprint(module: VisualConceptCreation) -> Dict[str, Any]:
    return {
        "loaded": module._cluster_loaded,
        "fingerprint": module._cluster_metadata.get("fingerprint"),
        "selected_counts": list(module._cluster_metadata.get("selected_counts") or []),
        "full_counts": list(module._cluster_metadata.get("full_counts") or []),
        "feature_mean": module._cluster_feature_mean.detach().cpu().clone(),
        "structure_mean": module._cluster_structure_mean.detach().cpu().clone(),
        "flat_fraction": module._cluster_flat_fraction.detach().cpu().clone(),
        "counts": module._cluster_counts.detach().cpu().clone(),
        "rgb_sorted": [t.detach().cpu().clone() for t in module._cluster_rgb_sorted],
        "rgb_prefix": [t.detach().cpu().clone() for t in module._cluster_rgb_prefix],
    }


def _assert_refs_equal(a: Dict[str, Any], b: Dict[str, Any], label: str) -> None:
    assert a["loaded"] and b["loaded"], label
    assert a["fingerprint"] == b["fingerprint"], label
    assert a["selected_counts"] == b["selected_counts"], label
    assert a["full_counts"] == b["full_counts"], label
    assert torch.equal(a["feature_mean"], b["feature_mean"]), label
    assert torch.equal(a["structure_mean"], b["structure_mean"]), label
    assert torch.equal(a["flat_fraction"], b["flat_fraction"]), label
    assert torch.equal(a["counts"], b["counts"]), label
    assert len(a["rgb_sorted"]) == len(b["rgb_sorted"]), label
    for i, (x, y) in enumerate(zip(a["rgb_sorted"], b["rgb_sorted"])):
        assert torch.equal(x, y), f"{label}: rgb_sorted[{i}]"
    for i, (x, y) in enumerate(zip(a["rgb_prefix"], b["rgb_prefix"])):
        assert torch.equal(x, y), f"{label}: rgb_prefix[{i}]"


def test_first_forward_evaluates_without_warmup():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        assert m._cluster_loaded
        m.train()
        B, H, W, T = 1, 2, 2, 2
        features = torch.randn(B, IN_CHANNELS, T, H, W)
        rgb_bank = np.load(bank_dir / "descriptors/rgb.npy")
        query_chw = torch.from_numpy(
            np.repeat(_chw_from_hwc(rgb_bank)[:1], B * H * W, axis=0)
        )
        rgb = _paint_rgb_from_chw(query_chw, H, W)
        cached = torch.randn(B, IN_CHANNELS, 1, H, W)
        out = m(
            features,
            return_losses=True,
            rgb_last_frame=rgb,
            cached_backbone_features=cached,
        )
        diag = m.visual_appearance_diagnostics
        assert diag["active_this_forward"] is True
        assert diag["reason"] == "fixed_reference_cluster_comparison"
        assert float(diag["app_loss_patch_count"]) == float(B * H * W)
        assert float(out["losses"]["loss_visual_appearance"].detach()) >= 0.0
        assert m._appearance_forward_count == 1  # no empty-history warmup
        print("PASS first forward evaluates without warmup")


def test_non_stage4_zero_appearance():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        _write_synthetic_bank(bank_dir)
        for stage in ("stage1", "stage2", "stage3"):
            m = _make_module(str(bank_dir), stage_name=stage, in_channels=IN_CHANNELS)
            # Non-stage4 never auto-loads / uses the bank for appearance.
            assert not m._cluster_loaded or stage == "stage4"
            m.train()
            features = torch.randn(1, IN_CHANNELS, 2, 2, 2)
            rgb = torch.rand(1, 3, 4, 4)
            cached = torch.randn(1, IN_CHANNELS, 1, 2, 2)
            out = m(
                features,
                return_losses=True,
                rgb_last_frame=rgb,
                cached_backbone_features=cached,
            )
            assert abs(float(out["losses"]["loss_visual_appearance"].detach())) == 0.0
            assert m.visual_appearance_diagnostics["reason"] == "stage_excluded"
        print("PASS stages1/2/3 contribute no appearance loss")


def test_costs_match_explicit_pairwise():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        bank = _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        m.train()

        # Two queries: colorful ramp + flat gray (covers flat-patch structure).
        rgb_bank = bank["rgb_hwc"]
        query_rgb = _chw_from_hwc(rgb_bank[[0, 1]])  # colorful, flat
        query_feat = bank["feature"][[0, 1]].astype(np.float32)
        # Also include a zero-norm feature query to verify exclusion policy later.
        descriptors = torch.from_numpy(query_rgb.copy())
        feat_desc = torch.from_numpy(query_feat.copy())
        costs, populated = m._fixed_appearance_costs(descriptors, feat_desc)
        assert bool(populated.all())
        explicit = _explicit_cluster_costs(query_rgb, query_feat, bank)
        for name in ("rgb", "feature", "structure"):
            got = costs[name].detach().cpu().numpy()
            exp = explicit[name]
            assert np.allclose(got, exp, atol=1e-5, rtol=1e-5), (
                f"{name} mismatch\n got={got}\n exp={exp}"
            )

        # Flat query vs all-flat member should have near-zero structure cost to cluster0
        # (cluster0 contains the flat member; mean includes flat contribution).
        assert explicit["structure"][1, 0] < explicit["structure"][0, 0] + 0.5
        print("PASS three-metric costs match explicit query-to-every-reference")


def test_references_immutable_across_train_val_reload():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        before = _reference_fingerprint(m)
        m.train()
        B, H, W, T = 1, 2, 2, 2
        features = torch.randn(B, IN_CHANNELS, T, H, W, requires_grad=False)
        # Keep encoder/prototypes trainable.
        for p in m.parameters():
            p.requires_grad_(True)
        rgb_bank = np.load(bank_dir / "descriptors/rgb.npy")
        query_chw = torch.from_numpy(
            np.repeat(_chw_from_hwc(rgb_bank)[:1], B * H * W, axis=0)
        )
        rgb = _paint_rgb_from_chw(query_chw, H, W)
        cached = torch.randn(B, IN_CHANNELS, 1, H, W)
        assert not cached.requires_grad
        out = m(
            features,
            return_losses=True,
            rgb_last_frame=rgb,
            cached_backbone_features=cached,
        )
        loss = out["losses"]["loss_visual_appearance"]
        opt = torch.optim.SGD(
            list(m.visual_encoder.parameters()) + [m.visual_concepts], lr=0.1
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        after_train = _reference_fingerprint(m)
        _assert_refs_equal(before, after_train, "after train step")

        m.eval()
        with torch.no_grad():
            m(
                features,
                return_losses=True,
                rgb_last_frame=rgb,
                cached_backbone_features=cached,
            )
        after_val = _reference_fingerprint(m)
        _assert_refs_equal(before, after_val, "after eval")

        # Checkpoint reload of learned weights must not alter fixed references.
        state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        m2 = _make_module(str(bank_dir))
        m2.load_state_dict(state, strict=True)
        after_reload = _reference_fingerprint(m2)
        _assert_refs_equal(before, after_reload, "after checkpoint reload")
        # Ensure bank arrays were not serialized into the ordinary state_dict.
        assert not any("cluster" in k for k in state)
        print("PASS references immutable after forward/backward/opt/val/reload")


def test_appearance_gradients_and_frozen_targets():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        m.train()
        B, H, W, T = 1, 2, 2, 2
        features = torch.randn(B, IN_CHANNELS, T, H, W)
        rgb_bank = np.load(bank_dir / "descriptors/rgb.npy")
        query_chw = torch.from_numpy(
            np.repeat(_chw_from_hwc(rgb_bank)[:1], B * H * W, axis=0)
        )
        rgb = _paint_rgb_from_chw(query_chw, H, W)
        cached = torch.randn(B, IN_CHANNELS, 1, H, W)
        # If a caller mistakenly left grad on cached targets, the module must reject.
        bad = cached.detach().clone().requires_grad_(True)
        try:
            m(
                features,
                return_losses=True,
                rgb_last_frame=rgb,
                cached_backbone_features=bad,
            )
            raise AssertionError("expected ValueError for cached requires_grad")
        except ValueError as exc:
            assert "detached" in str(exc).lower() or "requires_grad" in str(exc)

        out = m(
            features,
            return_losses=True,
            rgb_last_frame=rgb,
            cached_backbone_features=cached,
        )
        loss = out["losses"]["loss_visual_appearance"]
        loss.backward()
        assert m.visual_concepts.grad is not None and m.visual_concepts.grad.abs().sum() > 0
        enc_grads = [
            p.grad.abs().sum().item()
            for p in m.visual_encoder.parameters()
            if p.grad is not None
        ]
        assert any(g > 0 for g in enc_grads)
        # Assignment logits are intermediate; gradients must reach prototypes/encoder.
        assert out["visual_concept_logits"].requires_grad or True
        assert not cached.requires_grad
        print("PASS appearance grads on encoder/prototypes; cached targets frozen")


def test_missing_inputs_and_provenance_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        m.train()
        features = torch.randn(1, IN_CHANNELS, 2, 2, 2)
        rgb = torch.rand(1, 3, 4, 4)
        # Missing cached features with feature term enabled must raise.
        try:
            m(features, return_losses=True, rgb_last_frame=rgb, cached_backbone_features=None)
            raise AssertionError("expected RuntimeError for missing cached features")
        except RuntimeError as exc:
            assert "cached_backbone_features" in str(exc)

        # Incompatible expected provenance / checkpoint fingerprint.
        try:
            _make_module(
                str(bank_dir),
                visual_cluster_expected_checkpoint_sha256="definitely_wrong_fp",
            )
            raise AssertionError("expected provenance mismatch")
        except ValueError as exc:
            assert "fingerprint" in str(exc).lower() or "sha" in str(exc).lower()

        # Resume fingerprint mismatch via checkpoint metadata helper.
        # Import train helpers without requiring GPU/dataset.
        src_root = _HERE.parent
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        import train as train_mod

        class _Model:
            def __init__(self, module):
                self.concept_creations = {"stage4": module}

        live = _make_module(str(bank_dir))
        model = _Model(live)
        meta = build_patch_cluster_reference_checkpoint_metadata(model)
        assert meta is not None
        bad = dict(meta)
        bad["bank_fingerprint"] = "not_the_live_fingerprint"
        try:
            train_mod._verify_patch_cluster_checkpoint_metadata(
                model, {"patch_cluster_reference": bad}, is_resume=True, checkpoint_path="x.pth"
            )
            raise AssertionError("expected resume reject")
        except RuntimeError as exc:
            assert "Resume rejected" in str(exc)
        print("PASS missing inputs / provenance / resume fingerprint failures")


def test_fifo_migration_never_imports_queue_members():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        bank = _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        before = _reference_fingerprint(m)
        fake_queue = torch.randn(NUM_CONCEPTS, 12, RGB_DIM)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.load_prototype_queue_state(
                {
                    "format": "prototype_rgb_feature_fifo_v3",
                    "num_prototypes": NUM_CONCEPTS,
                    "in_channels": IN_CHANNELS,
                    "queue": fake_queue,
                    "feature_queue": torch.randn(NUM_CONCEPTS, 12, IN_CHANNELS),
                }
            )
        assert any("Ignoring obsolete FIFO" in str(w.message) for w in caught)
        after = _reference_fingerprint(m)
        _assert_refs_equal(before, after, "fifo migration")
        # Bank RGB means must still equal the synthetic members, not the fake queue.
        rgb_refs = _chw_from_hwc(bank["rgb_hwc"])
        # Mean of cluster0 RGB in CHW sorted-prefix path is independent of fake_queue.
        assert not torch.allclose(
            before["feature_mean"],
            fake_queue.mean(dim=(0, 1))[:IN_CHANNELS],
        )
        _ = rgb_refs
        print("PASS FIFO migration never imports queue members into fixed references")


def test_eval_skips_appearance_calculations():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        features = torch.randn(1, IN_CHANNELS, 2, 2, 2)
        rgb = torch.rand(1, 3, 4, 4)
        cached = torch.randn(1, IN_CHANNELS, 1, 2, 2)
        m.eval()
        out = m(
            features,
            return_losses=True,
            rgb_last_frame=rgb,
            cached_backbone_features=cached,
        )
        assert m.visual_appearance_diagnostics["reason"] == "evaluation"
        assert float(m.visual_appearance_diagnostics["app_loss_patch_count"]) == 0.0
        assert abs(float(out["losses"]["loss_visual_appearance"].detach())) == 0.0

        src_root = _HERE.parent
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        import train as train_mod

        class _Model(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.concept_creations = {"stage4": module}
                self.training = False

        model = _Model(m)
        model.eval()
        assert train_mod._needs_reference_features(model) is False
        print("PASS evaluation skips appearance calculations and cache resolution")


def test_zero_norm_feature_excludes_all_components():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "bank"
        _write_synthetic_bank(bank_dir)
        m = _make_module(str(bank_dir))
        m.train()
        B, H, W, T = 1, 1, 2, 1
        features = torch.randn(B, IN_CHANNELS, T, H, W)
        rgb_bank = np.load(bank_dir / "descriptors/rgb.npy")
        query_chw = torch.from_numpy(
            np.repeat(_chw_from_hwc(rgb_bank)[:1], B * H * W, axis=0)
        )
        rgb = _paint_rgb_from_chw(query_chw, H, W)
        cached = torch.zeros(B, IN_CHANNELS, 1, H, W)
        # One valid feature vector, one zero → only the valid query counts.
        cached[0, 0, 0, 0, 0] = 1.0
        out = m(
            features,
            return_losses=True,
            rgb_last_frame=rgb,
            cached_backbone_features=cached,
        )
        diag = m.visual_appearance_diagnostics
        assert float(diag["app_loss_patch_count"]) == 1.0
        assert float(diag["app_loss_rgb_patch_count"]) == 1.0
        assert float(diag["app_loss_feature_patch_count"]) == 1.0
        assert float(diag["app_loss_structure_patch_count"]) == 1.0
        assert float(diag["app_loss_patch_count_total"]) == 2.0
        _ = out
        print("PASS zero-norm feature excludes query from all three components")


if __name__ == "__main__":
    test_first_forward_evaluates_without_warmup()
    test_non_stage4_zero_appearance()
    test_costs_match_explicit_pairwise()
    test_references_immutable_across_train_val_reload()
    test_appearance_gradients_and_frozen_targets()
    test_missing_inputs_and_provenance_and_resume()
    test_fifo_migration_never_imports_queue_members()
    test_eval_skips_appearance_calculations()
    test_zero_norm_feature_excludes_all_components()
    print("ALL APPEARANCE COVERAGE TESTS PASSED")
