#!/usr/bin/env python3
"""Synthetic fixture for the graph-free cache→cluster pipeline."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import form_patch_clusters as fpc
from model.backbone_feature_cache import (
    build_reference_feature_provenance,
    compare_reference_feature_provenance,
    preprocess_config,
)
from model.concept_creation import VisualConceptCreation


FEATURE_DIM = 8
GRID_H, GRID_W = 2, 3
RESIZE = (32, 48)
RGB_SIZE = 4
WINDOW_LEN = 4
SYNTH_FP = "synthetic_backbone_fingerprint_cache_mode_v1"


def _write_tiny_train_split(root: Path, *, n_frames: int = 6) -> Path:
    """Minimal DHF1K-like layout that DatasetLoader accepts."""
    train = root / "train"
    video = train / "vid0001"
    for sub in ("images", "maps", "fixation"):
        (video / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        name = f"{i:04d}.png"
        rgb = (rng.random((40, 60, 3)) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(video / "images" / name)
        Image.fromarray(np.full((40, 60), 128, dtype=np.uint8)).save(video / "maps" / name)
        Image.fromarray(np.zeros((40, 60), dtype=np.uint8)).save(video / "fixation" / name)
    return train


class _FakeCache:
    """Stand-in for FrozenBackboneFeatureCache (no Kinetics load)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.reference_fingerprint = SYNTH_FP
        self.checkpoint_path = str(fpc.DEFAULT_REFERENCE_CHECKPOINT)
        self.checkpoint_sha256 = "0" * 64
        self.preprocess = preprocess_config(resize_to=RESIZE, normalize=True)
        self._writes = 0
        self._hits = 0

    def get_or_compute(self, rgb, *, identity=None, force=False, allow_compute=True):
        # Deterministic features from content hash so reuse is stable.
        seed = int(identity.content_sha256[:8], 16) if identity is not None else 0
        rng = np.random.default_rng(seed)
        feat = rng.standard_normal((FEATURE_DIM, 1, GRID_H, GRID_W)).astype(np.float32)
        # Keep nonzero channels for normalize_feature.
        feat += 0.1
        if allow_compute:
            self._writes += 1
        else:
            self._hits += 1
        return {
            "stage1": torch.zeros(96, 1, 1, 1),
            "stage2": torch.zeros(192, 1, 1, 1),
            "stage3": torch.zeros(384, 1, 1, 1),
            "stage4": torch.from_numpy(feat),
        }


def test_cache_mode_synthetic_bank_and_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dataset_root = tmp / "dhf1k"
        _write_tiny_train_split(dataset_root)
        out_dir = tmp / "patch_cluster_frozen_ref"
        cache_root = tmp / "cached_feats"
        cache_root.mkdir()

        fake = _FakeCache(cache_root)
        provenance = build_reference_feature_provenance(
            backbone_fingerprint=SYNTH_FP,
            checkpoint_path=fake.checkpoint_path,
            checkpoint_sha256=fake.checkpoint_sha256,
            weight_scope="FrozenBackboneFeatureCache.reference.state_dict",
            preprocess=fake.preprocess,
            temporal_selection="last_backbone_step",
            feature_module="backbone",
            feature_output_path=["stage1", "stage2", "stage3", "stage4"],
            feature_layout="BCTHW_last_slice_per_stage",
            spatial_grid="native_stage_resolutions_after_resize_to",
            verified=True,
        )

        args = SimpleNamespace(
            source_mode="cache",
            dataset_root=dataset_root,
            split="train",
            window_len=WINDOW_LEN,
            stride=1,
            cache_root=cache_root,
            reference_checkpoint=fpc.DEFAULT_REFERENCE_CHECKPOINT,
            device="cpu",
            storage_dtype="float32",
            resize_to=list(RESIZE),
            rgb_size=RGB_SIZE,
            max_windows=2,
            allow_compute=True,
            cache_read_only=False,
            num_prototypes=2,
            fit_sample_size=8,
            fit_max_per_video=8,
            iterations=2,
            restarts=1,
            medoid_candidates=4,
            batch_size=16,
            seed=0,
            feature_weight=1.0,
            rgb_weight=1.0,
            structure_weight=1.0,
            overwrite=False,
            output_dir=out_dir,
        )

        with mock.patch.object(fpc, "FrozenBackboneFeatureCache", create=True), mock.patch(
            "model.backbone_feature_cache.FrozenBackboneFeatureCache",
            return_value=fake,
        ), mock.patch(
            "model.backbone_feature_cache.provenance_from_cache",
            return_value=provenance,
        ):
            # extract_from_training_cache imports FrozenBackboneFeatureCache locally.
            import model.backbone_feature_cache as bfc

            with mock.patch.object(bfc, "FrozenBackboneFeatureCache", return_value=fake), mock.patch.object(
                bfc, "provenance_from_cache", return_value=provenance
            ):
                weights = (1.0, 1.0, 1.0)
                out_dir = fpc.prepare_output_directory(out_dir, overwrite=False)
                manifest = fpc.build_from_cache(args, out_dir, weights)

        assert manifest["status"] == "complete"
        assert manifest["source_mode"] == "cache"
        assert manifest["graph_sqlite_used"] is False
        assert manifest["model_assignments_used_for_clustering"] is False
        assert manifest["reference_feature_provenance_status"] == "verified"
        assert manifest["reference_feature_provenance"]["backbone_fingerprint"] == SYNTH_FP
        assert manifest["num_prototypes"] == 2
        assert manifest["window_stride"] == 1
        assert Path(out_dir / "manifest.json").is_file()
        assert Path(out_dir / "patch_clusters.sqlite").is_file()
        assert Path(out_dir / "cluster_references.npz").is_file()
        for name in ("feature", "rgb", "structure", "flat", "node_ids", "prototype_ids"):
            assert Path(out_dir / "descriptors" / f"{name}.npy").is_file()

        # Per-member arrays (not medoids-only).
        labels = np.load(out_dir / "descriptors/prototype_ids.npy")
        features = np.load(out_dir / "descriptors/feature.npy")
        assert labels.shape[0] == features.shape[0] == manifest["num_patches"]
        assert features.shape[1] == FEATURE_DIM
        assert set(labels.tolist()) == {0, 1}

        # train.py acceptance helper.
        compare_reference_feature_provenance(
            manifest["reference_feature_provenance"], provenance
        )

        # concept_creation loader accepts the bank.
        module = VisualConceptCreation(
            in_channels=FEATURE_DIM,
            concept_dim=8,
            num_concepts=2,
            num_visual_concepts=2,
            hidden_dim=16,
            top_k=2,
            stage_name="stage4",
            visual_appearance_patch_size=RGB_SIZE,
            visual_cluster_dir=str(out_dir),
            use_visual_saliency_alignment=False,
            visual_saliency_align_weight=0.0,
            visual_entropy_weight=0.0,
            visual_usage_weight=0.0,
        )
        assert module._cluster_loaded
        assert module._cluster_metadata["fingerprint"]
        assert (
            module._cluster_metadata["reference_feature_provenance"]["backbone_fingerprint"]
            == SYNTH_FP
        )
        print("PASS synthetic cache→cluster bank + provenance + concept_creation load")


def test_parser_exposes_stride_and_cluster_count():
    p = fpc.parser()
    args = p.parse_args(
        [
            "--source-mode",
            "cache",
            "--stride",
            "2",
            "--num-prototypes",
            "256",
            "--output-dir",
            str(fpc.DEFAULT_CACHE_OUTPUT_DIR),
        ]
    )
    assert args.stride == 2
    assert args.num_prototypes == 256
    assert args.allow_compute is True
    assert Path(args.reference_checkpoint) == fpc.DEFAULT_REFERENCE_CHECKPOINT
    print("PASS parser exposes stride, cluster count, Kinetics default, allow-compute")


if __name__ == "__main__":
    test_parser_exposes_stride_and_cluster_count()
    test_cache_mode_synthetic_bank_and_provenance()
    print("ALL CACHE-MODE FIXTURE TESTS PASSED")
