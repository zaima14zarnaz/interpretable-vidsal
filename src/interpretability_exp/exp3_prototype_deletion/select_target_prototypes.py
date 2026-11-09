#!/usr/bin/env python3
r"""Select individual prototype deletion targets from exported patch_data.json.

Usage:
    # Extract once from the checkpoint, then select IDs. Later runs reuse JSON:
    python select_target_prototypes.py --output-dir prototype_selection
    # Override extraction settings on the FIRST run:
    python select_target_prototypes.py --output-dir prototype_selection \
        --checkpoint /path/to/best_dhf1k.pth --val-dir /path/to/dhf1k/val --gpu-id 1
    # Optional: score an existing all-stage export directly:
    python select_target_prototypes.py --patch-data patch_data.json \
        --output-dir prototype_selection
    # Reuse previously computed scores/support and preserve the existing split:
    python select_target_prototypes.py --finalize-existing prototype_selection
    # Export video lists from the saved split without recomputing targets:
    python select_target_prototypes.py --export-video-ids-only prototype_selection

Defaults: video-disjoint 50/50 split (seed 42), frame-local top 5% and
bottom 70%, first eight ranked exported IDs, independent banks of 512 IDs at
stage1, stage2, stage3 and stage4. Mixed-stage exports are processed in ONE pass.
Alternatively pass --stage-patch-data stage1=PATH stage2=PATH stage3=PATH stage4=PATH.
Every stage uses the SAME selection/evaluation video split. An existing output
split_manifest.json is reused automatically; --split-manifest selects another
saved split. Evaluation videos are never used to compute scores or candidates.
Selection uses only selection videos. Evaluation videos are recorded, not scored.
Targets must have E(G) > 1, >=100 target-band patches and >=5 selection videos.
Filter first, then rerank: retain up to 15 HIGH and 18 LOW targets by default.
Thresholds are configurable practical settings, not formal validity guarantees.

Expected input: a JSON ARRAY of patch dictionaries, as in the supplied script:
    video_fname (or video_id/video_name)
    absolute_frame_index (or frame_filename/frame_no)
    patch_pred_saliency
    top_20_activated_prototype_indices: ranked list of IDs
    and/or top_20_activated_prototypes: dictionaries with prototype_index or
        prot_idx, optionally rank (1-based).
Records need stage (1..4 or stage1..stage4), unless their input file is explicitly
labelled with --stage-patch-data or a single-bank --stage was requested.
Optional patch_index and window-context fields are validated if present.
The first eight exported IDs are assumed to be actual activated assignments;
this script cannot recover model validity masks absent from the export.

E_high(j) = P(j active | high) / P(j active | low).
E_low(j) is the reverse ratio. Each patch counts each ID at most once.
Default Beta(0.5, 0.5) smoothing makes ratios finite:
    P_smoothed = (patch_support + alpha) / (band_patch_count + 2*alpha).
Raw probabilities and raw ratios are also saved (undefined ratios use null).
This is single-ID selectivity, not subset/coactivation-group selectivity.

Band sizes use ceil(percent * frame_patch_count / 100). Cutoff ties are
broken by patch_index, or input order when patch_index is unavailable.
Flat frames are skipped. If high and low cutoff scores coincide, patches at
that shared score are excluded from BOTH bands to avoid arbitrary labels;
actual counts can therefore be smaller than requested. Tiny frames whose
requested counts overlap are skipped. Partial exported frames are treated
as the full available patch population; export complete grids for experiments.

Files:
    high_prototypes.json, low_prototypes.json: stage-keyed score dictionaries
        under "stages", including all bank IDs and independently computed E(G).
    high_prototypes.csv, low_prototypes.csv: the same score rows with stage/ID.
    split_manifest.json: exact video/frame membership and frame band cutoffs.
    selection_video_ids.json, evaluation_video_ids.json: the exact saved video
        lists, exported for evaluation_high_local_deletion.py.
    selection_metadata.json: formulas, settings, totals and ranking summaries.
    finalized_targets.json: "stages" maps each bank to its own HIGH/LOW candidates,
        eligible targets, selected targets, and equal-count HIGH/LOW deletion sets.
    finalized_targets.csv: all prototypes with support/eligibility/selection flags.
Candidates are ALL bank prototypes, ranked by descending E(G) (candidate_rank),
including unobserved and E(G)<=1 IDs; "enriched" marks observed E(G)>threshold.
Score files list prototypes in the same descending-E(G) order.
The existing support checks and 15 HIGH/18 LOW selection caps remain configurable.
JSON schema version 3 requires consumers to read each stage's own IDs; never copy
stage4 IDs into another bank. --finalize-existing can rerank saved all-stage scores
but cannot recover missing stages from old stage4-only scores: rescan all-stage data.

With no input-source flag, extraction/cache reuse is automatic. The cache is
OUTPUT_DIR/stage_top8_activations.json (override with --activation-json). It is a
streamed JSON ARRAY containing actual valid top-8 IDs, original assignment weights,
and baseline sigmoid saliency patch means at every stage/last real window frame.
An adjacent *.metadata.json stores checkpoint/model/dataset provenance. Complete
files are installed atomically; failed extraction never publishes a partial JSON.
If the activation JSON exists, it is reused without loading torch/model/checkpoint
or running inference. Extraction flags affect only a newly generated cache;
choose another directory/cache filename to extract a different checkpoint/config.
Scoring still validates all required stage records before producing target files.
Physical last-frame duplicates are exported once (first dataset window wins).
The existing frame-local band/tie rules and support/selection thresholds are retained.
Extraction requires your repository's train.py, model and pre_process modules;
--src-root points to src if the script is located elsewhere. Cache scoring needs
only the Python standard library. No evaluator script is required.

Terminal percentages are cumulative RANK fractions of eligible prototypes,
not fractions of summed E(G). Equal E(G) values are ordered by numeric ID.
Eligibility requires both target-band support and E(G) > --min-selectivity.
Rejected IDs remain in the score dictionaries with explicit failure reasons.
Existing-score mode uses high_prototypes.json, low_prototypes.json,
selection_metadata.json and split_manifest.json; it never resplits videos.
Probabilities pool selected patches; long videos can contribute more patches.

Only Python 3.8+ standard library is required. Input is streamed and compact
records are temporarily indexed on disk, so unordered frames are supported.
No training, deletion, or evaluation-set ranking is performed.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from decimal import Decimal, ROUND_CEILING
import json
import hashlib
import math
import os
from pathlib import Path
import random
import sqlite3
import struct
import sys
import tempfile
from datetime import datetime, timezone

STAGES = ("stage1", "stage2", "stage3", "stage4")
DEFAULT_CHECKPOINT = "/home/z/zaimazarnaz/research1/ExplainableSaliency/src/training_outputs/saved_weights/best_dhf1k.pth"
DEFAULT_VAL_DIR = "/data/quantization/zaima/videosal_datasets/dhf1k/val"


class IndexedSamples:
    """Keep stable DatasetLoader indices alongside the existing project collate."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def extraction_collate(samples):
    from pre_process.collate import video_saliency_collate_fn
    indices, items = zip(*samples)
    return video_saliency_collate_fn(items), list(indices)


def build_extraction_runtime(args):
    """Lazy imports: cache reuse/scoring must never initialize the model."""
    roots = [args.src_root] if args.src_root else list(Path(__file__).resolve().parents) + [Path.cwd(), Path.cwd() / "src"]
    src = next((p.resolve() for p in roots if (p / "train.py").is_file()), None)
    if src is None:
        raise ValueError("Extraction needs your project src containing train.py; pass --src-root /path/to/ExplainableSaliency/src.")
    sys.path.insert(0, str(src))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import train as train_cfg
    from torch.utils.data import DataLoader
    from model.model import ExplainableVidSalModel
    from pre_process.dataloader import DatasetLoader

    if not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(args.checkpoint))
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError("GPU {} unavailable; pass --gpu-id for a visible device.".format(args.gpu_id))
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device("cpu")
    train_cfg.set_seed(args.seed)
    # Same inference architecture/settings as the reviewed deletion evaluator.
    # All required checkpoint keys are loaded; no pretrained download is needed.
    model = ExplainableVidSalModel(
        backbone_stages=STAGES, pretrained_backbone=False,
        freeze_backbone=train_cfg.FREEZE_BACKBONE, backbone_gradient_checkpointing=False,
        input_format="BTCHW", resize_to=(224, 384), concept_dim=128,
        num_concepts=args.num_prototypes, concept_hidden_dim=256, saliency_hidden_dim=256,
        top_k=8, max_source_patches=64, tau_pi=0.5, tau_alpha=0.07, tau_concept=0.07,
        concept_residual_weight=0.0, last_transition_only=True,
        use_rgb_refinement=False, use_feature_refinement=False, output_activation="none",
        return_details=True, use_subpatch_head=True, subpatch_factor=4,
        subpatch_residual_scale=0.5, use_temporal_transition_aggregation=True,
        temporal_aggregation_hidden_channels=128, temporal_aggregation_temperature=1.0,
        visual_concept_on=train_cfg.VISUAL_CONCEPT_ON, temporal_concepts_on=train_cfg.TEMPORAL_CONCEPTS_ON,
        visual_concept_logit_scale=train_cfg.VISUAL_CONCEPT_LOGIT_SCALE,
        visual_concept_residual_weight=1.0, use_temporal_feature_infusion=True,
        visual_assignment_mode="topk_soft", use_shared_concept_activations=True,
        prototype_bottleneck_strength=train_cfg.PROTOTYPE_BOTTLENECK_STRENGTH,
        prototype_application_position=train_cfg.PROTOTYPE_APPLICATION_POSITION,
        fine_unary_mask_strength=getattr(train_cfg, "FINE_UNARY_MASK_STRENGTH", 1.0),
        stage2_backbone_fusion_enabled=getattr(train_cfg, "STAGE2_BACKBONE_FUSION_ENABLED", True),
    ).to_split_devices(device, device)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict, as in best_dhf1k.pth.")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing:
        raise ValueError("Checkpoint is missing required model weights: {}".format(missing[:5]))
    if unexpected:
        print("Warning: ignored {} unused checkpoint keys.".format(len(unexpected)))
    epoch = checkpoint.get("epoch")
    if hasattr(epoch, "item"):
        epoch = epoch.item()
    del checkpoint
    model.eval()
    model.allow_eval_concept_losses = False
    for stage in STAGES:
        concept = model.concept_creations[stage]
        if concept.top_k != 8 or concept.visual_assignment_mode != "topk_soft":
            raise ValueError(stage + ": extraction requires the updated topk_soft model with eight valid slots.")
        if not model.saliency_prediction.fusion_blocks[stage].use_shared_concept_activations:
            raise ValueError(stage + ": prediction must use the same shared concept assignments being exported.")
    window_len = train_cfg.WINDOW_LEN if args.window_len is None else args.window_len
    batch_size = train_cfg._dataloader_batch_size() if args.batch_size is None else args.batch_size
    workers = train_cfg.NUM_WORKERS if args.num_workers is None else args.num_workers
    dataset = DatasetLoader(str(args.val_dir), window_len=window_len, stride=args.stride)
    if not hasattr(dataset, "windows"):
        raise ValueError("DatasetLoader must expose windows as (video_filename, start_index) to export true frame identities.")
    options = dict(num_workers=workers, collate_fn=extraction_collate, pin_memory=torch.cuda.is_available())
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(IndexedSamples(dataset), batch_size=batch_size, shuffle=False, **options)
    config = {"checkpoint_epoch": epoch, "window_len": window_len, "batch_size": batch_size,
              "num_workers": workers, "stride": args.stride, "device": str(device),
              "assignment_mode": "topk_soft", "top_k": 8, "num_prototypes": args.num_prototypes,
              "output_activation": "none", "resize_to": [224, 384], "src_root": str(src)}
    config["source_sha256"] = {name: hashlib.sha256((src / name).read_bytes()).hexdigest()
                               for name in ("model/model.py", "model/concept_creation.py", "model/saliency_prediction.py", "train.py")
                               if (src / name).is_file()}
    return model, train_cfg, loader, dataset, config


def extraction_window_identity(dataset, index, name, length):
    window = dataset.windows[index]
    if not isinstance(window, (tuple, list)) or len(window) < 2:
        raise ValueError("DatasetLoader.windows entries must begin with (video_filename, start_index).")
    video = identifier(window[0], "dataset window video")
    start = integer(window[1], "window start index")
    if Path(video).stem != Path(str(name)).stem:
        raise ValueError("Collated video and indexed dataset window disagree.")
    return {"video_fname": video, "absolute_frame_index": start + length - 1,
            "dataset_index": index, "window_start_index": start, "window_length": length}


def stage_activation_records(prediction, model, identities, grid_audit):
    """Export actual model slots, not a new ranking of full-bank posterior IDs."""
    import torch
    import torch.nn.functional as F
    activation = model.saliency_prediction.output_activation
    if activation == "none":
        saliency = prediction["saliency_logits"].float().sigmoid()
    elif activation == "sigmoid":
        saliency = prediction["saliency_map"].float()
    else:
        raise ValueError("Unsupported baseline saliency activation: " + str(activation))
    if saliency.ndim != 4 or saliency.shape[1] != 1 or not torch.isfinite(saliency).all():
        raise ValueError("Expected finite baseline [B,1,H,W] saliency maps.")
    for stage in STAGES:
        c = prediction["concept_out"][stage]
        indices, valid = c["active_visual_prototype_indices"], c["visual_validity_mask"].bool()
        if indices.ndim != 5 or indices.shape[1] != 1 or indices.shape[-1] != 8 or valid.shape != indices.shape or not valid.all():
            raise ValueError(stage + ": expected actual last-frame [B,1,H,W,8] slots, all valid; install updated topk_soft model code.")
        b, _, h, w, k = indices.shape
        if b != len(identities) or saliency.shape[0] != b:
            raise ValueError("Prediction batch and exported frame identities disagree.")
        bank = int(model.concept_creations[stage].visual_concepts.shape[0])
        flat = indices.reshape(-1, k)
        if (flat < 0).any() or (flat >= bank).any() or not (flat.sort(-1).values.diff(dim=-1) > 0).all():
            raise ValueError(stage + ": invalid/duplicate top-8 IDs.")
        logits = c["visual_concept_logits"].reshape(b * h * w, bank)
        if not torch.equal(flat, logits.topk(8, dim=-1).indices):
            raise ValueError(stage + ": exported slots do not match original top-8 similarity assignments.")
        a = c["visual_activations"].reshape(b * h * w, bank).float()
        weights = a.gather(1, flat)
        if not torch.isfinite(a).all() or (a < 0).any() or not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), rtol=1e-5, atol=1e-6):
            raise ValueError(stage + ": original sparse top-8 assignment weights must be finite, nonnegative and sum to one.")
        if not torch.allclose(a.sum(-1), weights.sum(-1), rtol=1e-5, atol=1e-6):
            raise ValueError(stage + ": extra evidence exists outside the exported top-8 slots.")
        ph, pw = saliency.shape[-2:]
        if ph % h or pw % w:
            raise ValueError("Baseline output must be divisible by each concept grid for exact patch averaging.")
        scores = F.avg_pool2d(saliency, (ph // h, pw // w))[:, 0].reshape(b, h * w).cpu().tolist()
        ids = indices[:, 0].reshape(b, h * w, 8).cpu().tolist()
        masses = weights.reshape(b, h * w, 8).cpu().tolist()
        grid = [h, w]
        if grid not in grid_audit.setdefault(stage, []):
            grid_audit[stage].append(grid)
        for i, identity in enumerate(identities):
            for patch in range(h * w):
                yield {**identity, "stage": stage, "patch_index": patch,
                       "patch_pred_saliency": scores[i][patch],
                       "top_8_activated_prototype_indices": ids[i][patch],
                       "top_8_activation_weights": masses[i][patch]}


def extract_activation_json(args, path):
    import torch
    from tqdm import tqdm
    checkpoint_stat = args.checkpoint.stat()
    model, train_cfg, loader, dataset, config = build_extraction_runtime(args)
    seen_frames, seen_videos, grid_audit = set(), set(), {}
    row_counts = Counter()
    windows, duplicate_windows, invalid_windows = 0, 0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    meta_temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=path.name + ".", suffix=".part", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("[\n")
            first = True
            with torch.inference_mode():
                for collated, indices in tqdm(loader, desc="Extracting all-stage top-8", unit="batch"):
                    if len(collated) not in (5, 6):
                        raise ValueError("Expected current six-item collate (or five-item version without fixation labels).")
                    names, rgb = collated[:2]
                    lengths, valid = collated[-2:]
                    groups = {}
                    for i, index in enumerate(indices):
                        length = int(lengths[i])
                        if length < 1 or length > rgb.shape[1]:
                            raise ValueError("Invalid real RGB window length.")
                        if valid is not None:
                            if valid.ndim == 1:
                                is_valid = bool(valid[i])
                            elif valid.ndim == 2:
                                is_valid = bool(valid[i, length - 1])
                            else:
                                raise ValueError("Expected [B] or [B,T] temporal validity.")
                            if not is_valid:
                                invalid_windows += 1
                                continue
                        identity = extraction_window_identity(dataset, int(index), names[i], length)
                        frame = (Path(identity["video_fname"]).stem, identity["absolute_frame_index"])
                        if frame in seen_frames:
                            duplicate_windows += 1
                            continue
                        # Reserve in dataset order BEFORE grouping by length, so
                        # duplicate physical frames keep the first input window.
                        seen_frames.add(frame)
                        groups.setdefault(length, []).append((i, identity))
                    for length, items in sorted(groups.items()):
                        positions = [i for i, _ in items]
                        identities = [identity for _, identity in items]
                        device = model.output_device
                        enabled = not args.no_amp and train_cfg._amp_enabled(device)
                        with torch.autocast(device.type, dtype=train_cfg._amp_dtype(device), enabled=enabled):
                            prediction = model(rgb[positions, :length], return_details=True,
                                return_concept_losses=False, return_decoder_diagnostics=False)
                        for record in stage_activation_records(prediction, model, identities, grid_audit):
                            if not first:
                                handle.write(",\n")
                            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
                            first = False
                            row_counts[record["stage"]] += 1
                        windows += len(identities)
                        seen_videos.update(Path(identity["video_fname"]).stem for identity in identities)
                        del prediction
            if first:
                raise ValueError("No valid windows were exported.")
            wanted_videos = {Path(str(window[0])).stem for window in dataset.windows}
            if seen_videos != wanted_videos:
                raise ValueError("Videos with no valid exported windows: {}".format(sorted(wanted_videos - seen_videos)))
            final_checkpoint_stat = args.checkpoint.stat()
            if (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns) != (final_checkpoint_stat.st_size, final_checkpoint_stat.st_mtime_ns):
                raise ValueError("Checkpoint changed during extraction; incomplete cache was discarded.")
            handle.write("\n]\n")
            handle.flush()
            os.fsync(handle.fileno())
        metadata = {"status": "complete", "schema_version": 1, "stages": list(STAGES),
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_size_bytes": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns, "val_dir": str(args.val_dir.resolve()),
            "model_config": config, "dataset_windows": len(dataset), "exported_windows": windows,
            "duplicate_last_frame_windows_skipped": duplicate_windows, "invalid_windows_skipped": invalid_windows,
            "video_ids": sorted(seen_videos), "patch_grid_shapes": grid_audit,
            "patch_records_per_stage": dict(row_counts), "patch_records": sum(row_counts.values()),
            "top_k": 8, "assignment_mode": "topk_soft", "all_exported_slots_valid": True,
            "saliency_measurement": "sigmoid once per baseline pixel, then exact nonoverlapping patch mean",
            "frame_identity": "zero-based DatasetLoader.windows start_index + real_length - 1",
            "selection_evaluation_split": "applied after extraction; only selection videos are scored",
            "activation_json_size_bytes": temporary.stat().st_size,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        meta_path = path.with_suffix(".metadata.json")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=meta_path.name + ".", suffix=".part", dir=path.parent, delete=False) as handle:
            meta_temporary = Path(handle.name)
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
        os.replace(str(meta_temporary), str(meta_path))
        meta_temporary = None
        print("Saved {:,} patch records from {} real last-frame windows to {}".format(metadata["patch_records"], windows, path))
        return metadata
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if meta_temporary is not None:
            meta_temporary.unlink(missing_ok=True)


def ensure_activation_cache(args):
    path = args.activation_json
    meta_path = path.with_suffix(".metadata.json")
    if path.exists():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Existing activation cache is not a nonempty file: " + str(path))
        metadata = read_json(meta_path) if meta_path.is_file() else None
        if metadata:
            if metadata.get("status") != "complete" or metadata.get("top_k") != 8 or set(metadata.get("stages", [])) != set(STAGES):
                raise ValueError("Existing activation cache metadata is incomplete or not all-stage top-8; no inference was rerun.")
            if metadata.get("activation_json_size_bytes") != path.stat().st_size:
                raise ValueError("Existing activation cache size differs from its metadata; no inference was rerun.")
        print("Reusing existing activation JSON {}; model/checkpoint inference skipped.".format(path.resolve()))
        if metadata:
            print("Cached checkpoint: {} | extracted {}".format(metadata.get("checkpoint"), metadata.get("extracted_at")))
        return {"path": str(path.resolve()), "reused": True, "provenance": metadata}
    metadata = extract_activation_json(args, path)
    return {"path": str(path.resolve()), "reused": False, "provenance": metadata}


def canonical_stage(value):
    if isinstance(value, bool):
        raise ValueError("Stage must identify stage1, stage2, stage3 or stage4.")
    name = str(value).strip().lower()
    if name in ("1", "2", "3", "4"):
        name = "stage" + name
    if name not in STAGES:
        raise ValueError("Unknown stage {!r}; expected 1..4 or stage1..stage4.".format(value))
    return name


def expected_stages(args):
    return (args.stage,) if args.stage else STAGES


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(str(path) + " must be a JSON dictionary.")
    return result


def split_lists(manifest):
    """Validate the saved split without changing IDs or writing any files."""
    result = {}
    for split in ("selection", "evaluation"):
        values = manifest.get(split + "_videos")
        if not isinstance(values, list) or not values:
            raise ValueError("Manifest needs nonempty " + split + "_videos.")
        for value in values:
            identifier(value, split + " video ID")
        normalized = [Path(str(value)).stem for value in values]
        if any(not v for v in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("Video IDs must be unique after filename normalization.")
        result[split] = set(normalized)
    if result["selection"] & result["evaluation"]:
        raise ValueError("Selection/evaluation videos overlap after filename normalization.")
    return result


def iter_records(path, chunk_size=1 << 20):
    """Incrementally decode a JSON array without optional dependencies."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as handle:
        buffer, position, eof = "", 0, False

        def refill():
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            chunk = handle.read(chunk_size)
            buffer += chunk
            eof = not chunk

        def peek():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return None
                refill()

        if peek() != "[":
            raise ValueError("Input must be a JSON array of patch objects.")
        position += 1
        first = True
        while True:
            char = peek()
            if char == "]":
                position += 1
                break
            if not first:
                if char != ",":
                    raise ValueError("Expected a comma between patch records.")
                position += 1
                char = peek()
            if char != "{":
                raise ValueError("Expected a patch object; input may be truncated.")
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError("Invalid or truncated patch JSON.") from exc
                    refill()
            yield record
            first = False
        if peek() is not None:
            raise ValueError("Unexpected content after the JSON array.")


def integer(value, label, minimum=0):
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}.".format(label, minimum))
    return value


def identifier(value, label):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("{} must be a nonempty string or integer.".format(label))
    return str(value)


def ranked_ids(record, k, bank_size):
    ids = record.get("top_20_activated_prototype_indices", record.get("top_8_activated_prototype_indices"))
    details = record.get("top_20_activated_prototypes", record.get("top_8_activated_prototypes"))
    if details is not None:
        if not isinstance(details, list) or any(not isinstance(x, dict) for x in details):
            raise ValueError("top_20_activated_prototypes must be a list of objects.")
        if any("rank" in x for x in details):
            ranks = [integer(x.get("rank"), "rank", 1) for x in details]
            if len(set(ranks)) != len(ranks):
                raise ValueError("Duplicate prototype ranks.")
            details = sorted(details, key=lambda x: x["rank"])
            if [integer(x["rank"], "rank", 1) for x in details[:k]] != list(range(1, k + 1)):
                raise ValueError("Prototype ranks must start at 1 without gaps through top-k.")
        extracted = []
        for item in details:
            values = [integer(item[name], name) for name in ("prototype_index", "prot_idx") if name in item]
            if not values or len(set(values)) != 1:
                raise ValueError("Prototype details need consistent prototype_index/prot_idx.")
            extracted.append(values[0])
        if ids is not None:
            if not isinstance(ids, list) or [integer(x, "prototype ID") for x in ids] != extracted:
                raise ValueError("Ranked details and prototype index list disagree.")
        ids = extracted
    if not isinstance(ids, list) or len(ids) < k:
        raise ValueError("Expected at least {} ranked prototype IDs; no padding is used.".format(k))
    ids = [integer(x, "prototype ID") for x in ids[:k]]
    if len(set(ids)) != k:
        raise ValueError("Top-k IDs must be distinct.")
    if any(x >= bank_size for x in ids):
        raise ValueError("Prototype ID exceeds --num-prototypes ({}).".format(bank_size))
    return ids


def ingest(paths, db, args):
    db.executescript("""
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE patches(frame_id INTEGER, tie INTEGER, score REAL, ids BLOB);
    """)
    frames, schemas, count = {}, {}, 0
    print("Reading and validating independent stage patch records...", flush=True)
    pack = struct.Struct("<" + "I" * args.top_k)
    global_row = 0
    for forced_stage, path in paths:
        for row_number, record in enumerate(iter_records(path), 1):
            global_row += 1
            try:
                tag = record.get("stage", forced_stage or args.stage)
                if tag is None:
                    raise ValueError("Missing stage: use a stage-labelled export or --stage-patch-data STAGE=PATH.")
                stage = canonical_stage(tag)
                if forced_stage and stage != forced_stage:
                    raise ValueError("Record stage disagrees with the explicit input-file bank label.")
                if args.stage and stage != args.stage:
                    continue
                if stage not in schemas:
                    video_field = args.video_field or next((x for x in ("video_fname", "video_id", "video_name") if x in record), None)
                    frame_field = args.frame_field or next((x for x in ("absolute_frame_index", "frame_filename", "frame_no") if x in record), None)
                    if not video_field or not frame_field:
                        raise ValueError("Missing video/frame identifier fields.")
                    schemas[stage] = {"video_field": video_field, "frame_field": frame_field,
                        "stage": stage, "has_patch_index": "patch_index" in record,
                        "context_fields": [x for x in ("dataset_index", "window_start_index", "window_length") if x in record]}
                schema = schemas[stage]
                video = identifier(record.get(schema["video_field"]), schema["video_field"])
                frame = identifier(record.get(schema["frame_field"]), schema["frame_field"])
                fields = [x for x in ("dataset_index", "window_start_index", "window_length") if x in record]
                if fields != schema["context_fields"]:
                    raise ValueError("Inconsistent window-context fields within " + stage)
                context = []
                for field in fields:
                    if isinstance(record[field], (dict, list)):
                        raise ValueError("Non-scalar window-context field: " + field)
                    context.append(record[field])
                # The stage belongs in the frame key: IDs and patch indices at
                # different resolutions must NEVER share a prototype bank/grid.
                key = (stage, video, frame)
                if key not in frames:
                    frames[key] = {"id": len(frames), "context": context, "patch_count": 0}
                info = frames[key]
                if info["context"] != context:
                    raise ValueError("Physical stage/frame {} has multiple window contexts.".format(key))
                if ("patch_index" in record) != schema["has_patch_index"]:
                    raise ValueError("Inconsistent patch_index availability within " + stage)
                tie = integer(record["patch_index"], "patch_index") if schema["has_patch_index"] else global_row
                score = record.get(args.saliency_field)
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(args.saliency_field + " must be finite numeric saliency.")
                ids = ranked_ids(record, args.top_k, args.num_prototypes)
                db.execute("INSERT INTO patches VALUES (?,?,?,?)", (info["id"], tie, float(score), pack.pack(*ids)))
                info["patch_count"] += 1
                count += 1
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("{} record {}: {}".format(path, row_number, exc)) from exc
            if count % 50000 == 0:
                db.commit()
            if args.progress_every and count % args.progress_every == 0:
                print("  {:,} patches, {:,} stage/frames".format(count, len(frames)), flush=True)
    missing = set(expected_stages(args)) - set(schemas)
    if missing:
        raise ValueError("Missing exports for {}. Supply top-8 patch activations and saliency for every stage; stage4-only scores cannot infer other banks. Use --stage stage4 only for an intentional single-bank run.".format(sorted(missing)))
    if not count:
        raise ValueError("No patch records found.")
    db.commit()
    try:
        db.execute("CREATE UNIQUE INDEX frame_patch ON patches(frame_id, tie)")
    except sqlite3.IntegrityError as exc:
        raise ValueError("Duplicate patch_index within the same stage/frame; deduplicate the export first.") from exc
    for stage, schema in schemas.items():
        if not schema["has_patch_index"]:
            print(stage + ": no patch_index; ties use input order and duplicate spatial patches cannot be detected.")
    return frames, schemas, count, pack


def ceil_percent(n, percent):
    return int((Decimal(str(percent)) * n / Decimal(100)).to_integral_value(rounding=ROUND_CEILING))


def frame_bands(rows, top_percent, bottom_percent):
    ordered = sorted(rows, key=lambda row: (row[1], row[0]))
    n = len(ordered)
    high_n, low_n = ceil_percent(n, top_percent), ceil_percent(n, bottom_percent)
    info = {"requested_high_patches": high_n, "requested_low_patches": low_n}
    if high_n + low_n > n:
        return [], [], dict(info, skip_reason="too_few_patches")
    if ordered[0][1] == ordered[-1][1]:
        return [], [], dict(info, skip_reason="constant_saliency")
    high_cutoff, low_cutoff = ordered[n - high_n][1], ordered[low_n - 1][1]
    ambiguous = lambda row: high_cutoff <= row[1] <= low_cutoff
    # High ties prefer smaller patch_index (same convention as low ties).
    high_order = sorted(rows, key=lambda row: (-row[1], row[0]))
    high = [row for row in high_order[:high_n] if not ambiguous(row)]
    low = [row for row in ordered[:low_n] if not ambiguous(row)]
    info.update(high_cutoff=high_cutoff, low_cutoff=low_cutoff,
                ambiguous_patches=sum(ambiguous(row) for row in rows),
                high_cutoff_ties=sum(row[1] == high_cutoff for row in rows),
                low_cutoff_ties=sum(row[1] == low_cutoff for row in rows))
    if not high or not low:
        return [], [], dict(info, skip_reason="empty_band_after_tie_exclusion")
    return high, low, info


def collect(db, frames, selection, args, pack):
    counts = {band: Counter() for band in ("high", "low")}
    frame_counts = {band: Counter() for band in counts}
    video_sets = {band: [set() for _ in range(args.num_prototypes)] for band in counts}
    totals, skipped = Counter(), Counter()
    manifest = {"selection": [], "evaluation": []}
    for number, ((video, frame), source) in enumerate(sorted(frames.items()), 1):
        split = "selection" if Path(video).stem in selection else "evaluation"
        item = {"video_id": video, "frame_id": frame, "patch_count": source["patch_count"]}
        if source["context"]:
            item["window_context"] = source["context"]
        if split == "selection":
            rows = list(db.execute("SELECT tie,score,ids FROM patches WHERE frame_id=?", (source["id"],)))
            high, low, info = frame_bands(rows, args.top_percent, args.bottom_percent)
            item.update(info)
            item["high_patches"], item["low_patches"] = len(high), len(low)
            if "skip_reason" in info:
                skipped[info["skip_reason"]] += 1
            else:
                totals["eligible_frames"] += 1
            totals["ambiguous_patches"] += info.get("ambiguous_patches", 0)
            for band, selected_rows in (("high", high), ("low", low)):
                totals[band] += len(selected_rows)
                in_frame = set()
                for _, _, blob in selected_rows:
                    ids = pack.unpack(blob)
                    counts[band].update(ids)
                    in_frame.update(ids)
                frame_counts[band].update(in_frame)
                for proto in in_frame:
                    video_sets[band][proto].add(video)
        manifest[split].append(item)
        if number % 10000 == 0:
            print("  Processed {:,}/{:,} frames".format(number, len(frames)), flush=True)
    if not totals["high"] or not totals["low"]:
        raise ValueError("Selection split has no usable high/low patches. Inspect flat/tied or tiny frames.")
    return counts, frame_counts, video_sets, totals, skipped, manifest


def output_scores(counts, frame_counts, video_sets, totals, args):
    tables = {"high": {}, "low": {}}
    for proto in range(args.num_prototypes):
        h, l = counts["high"][proto], counts["low"][proto]
        ph, pl = h / totals["high"], l / totals["low"]
        sh = (h + args.smoothing) / (totals["high"] + 2 * args.smoothing)
        sl = (l + args.smoothing) / (totals["low"] + 2 * args.smoothing)
        base = {"high_patch_count": h, "low_patch_count": l,
                "high_frame_count": frame_counts["high"][proto], "low_frame_count": frame_counts["low"][proto],
                "high_video_count": len(video_sets["high"][proto]), "low_video_count": len(video_sets["low"][proto]),
                "p_high": ph, "p_low": pl, "p_high_smoothed": sh, "p_low_smoothed": sl}
        for band in tables:
            high = band == "high"
            probability, other = (ph, pl) if high else (pl, ph)
            e = sh / sl if high else sl / sh
            tables[band][str(proto)] = dict(base, **{"E(G)": e, "E_raw": probability / other if other else None,
                                                   "eligible": False, "rank": None})
    return tables


def rank_and_finalize(tables, args, metadata, manifest):
    """Apply the same selection-only checks to fresh or previously saved scores."""
    selection_videos = manifest.get("selection_videos", [])
    evaluation_videos = manifest.get("evaluation_videos", [])
    if not selection_videos or not evaluation_videos or set(selection_videos) & set(evaluation_videos):
        raise ValueError("Split manifest must contain nonempty, disjoint selection/evaluation video lists.")
    if set(tables["high"]) != set(tables["low"]):
        raise ValueError("HIGH and LOW score files must describe the same prototype bank.")
    finalized = {"schema_version": 1, "stage": metadata["input_schema"].get("stage"),
                 "selection_videos": selection_videos, "evaluation_videos": evaluation_videos,
                 "selection_rule": {"minimum_target_band_patches": args.min_band_patches,
                                    "minimum_target_band_videos": args.min_band_videos,
                                    "selectivity_strictly_greater_than": args.min_selectivity,
                                    "ranking": "descending smoothed E(G), ties by numeric prototype ID",
                                    "filter_before_ranking": True},
                 "high": {}, "low": {}}
    summaries = {}
    for band, entries in tables.items():
        failures = Counter()
        observed, enriched, supported = 0, 0, 0
        for proto, entry in entries.items():
            if not proto.isdecimal():
                raise ValueError("Prototype dictionary keys must be nonnegative numeric IDs.")
            # Both files must contain identical counts from the same selection run.
            for name in ("high_patch_count", "low_patch_count", "high_video_count", "low_video_count"):
                value = integer(entry.get(name), name)
                if value != tables["low" if band == "high" else "high"][proto].get(name):
                    raise ValueError("HIGH/LOW support disagrees for prototype {}.".format(proto))
            patch_count = entry[band + "_patch_count"]
            video_count = entry[band + "_video_count"]
            if video_count > len(selection_videos) or video_count > patch_count:
                raise ValueError("Impossible selection-video support for prototype {}.".format(proto))
            e = entry.get("E(G)")
            if isinstance(e, bool) or not isinstance(e, (int, float)) or not math.isfinite(e) or e <= 0:
                raise ValueError("E(G) must be positive and finite for prototype {}.".format(proto))
            other_e = tables["low" if band == "high" else "high"][proto].get("E(G)")
            if isinstance(other_e, bool) or not isinstance(other_e, (int, float)) or not math.isfinite(other_e) or not math.isclose(e * other_e, 1.0, rel_tol=1e-8):
                raise ValueError("HIGH/LOW selectivities are not reciprocals for prototype {}.".format(proto))
            reasons = []
            if patch_count < args.min_band_patches:
                reasons.append("insufficient_target_band_patches")
            if video_count < args.min_band_videos:
                reasons.append("insufficient_target_band_videos")
            if e <= args.min_selectivity:
                reasons.append("insufficient_selectivity")
            support_pass = (patch_count >= args.min_band_patches and video_count >= args.min_band_videos)
            entry.update(support_pass=support_pass, selectivity_pass=e > args.min_selectivity,
                         eligible=not reasons, exclusion_reasons=reasons, rank=None,
                         selected=False, target_rank=None,
                         enriched=patch_count > 0 and e > args.min_selectivity,
                         candidate=True, candidate_rank=None)
            failures.update(reasons)
            observed += patch_count > 0
            enriched += patch_count > 0 and e > args.min_selectivity
            supported += support_pass
        ranked = sorted((proto for proto in entries if entries[proto]["eligible"]),
                        key=lambda proto: (-entries[proto]["E(G)"], int(proto)))
        candidates = sorted(entries, key=lambda proto: (-entries[proto]["E(G)"], int(proto)))
        for rank, proto in enumerate(candidates, 1):
            entries[proto]["candidate_rank"] = rank
        for rank, proto in enumerate(ranked, 1):
            entries[proto]["rank"] = rank
        requested_count = args.high_target_count if band == "high" else args.low_target_count
        chosen = ranked[:requested_count]
        for rank, proto in enumerate(chosen, 1):
            entries[proto]["selected"] = True
            entries[proto]["target_rank"] = rank
        finalized[band] = {"requested_count": requested_count, "eligible_count": len(ranked),
                           "selected_count": len(chosen), "shortfall": max(0, requested_count - len(chosen)),
                           "prototype_ids": [int(proto) for proto in chosen],
                           "candidate_count": len(candidates),
                           "candidate_prototype_ids": [int(proto) for proto in candidates],
                           "eligible_prototype_ids": [int(proto) for proto in ranked],
                           "prototypes": {proto: dict(entries[proto]) for proto in chosen}}
        # All IDs in descending E(G) order, keeping direct ID lookup.
        tables[band] = {proto: entries[proto] for proto in candidates}
        print("\n{}: {} observed; {} enriched; {} pass support; {} pass all checks.".format(
            band.upper(), observed, enriched, supported, len(ranked)))
        print("  Checks: >= {} target-band patches, >= {} selection videos, E(G) > {}".format(
            args.min_band_patches, args.min_band_videos, args.min_selectivity))
        print("  Cumulative rank fraction     Count     Minimum E(G)")
        fractions = {}
        for percent in (20, 40, 60, 80, 100):
            n = ceil_percent(len(ranked), percent)
            cutoff = entries[ranked[n - 1]]["E(G)"] if n else None
            at_boundary = sum(entries[p]["E(G)"] == cutoff for p in ranked) if n else 0
            fractions[str(percent)] = {"count": n, "minimum_E(G)": cutoff, "prototypes_tied_at_cutoff": at_boundary}
            print("  Top {:3d}%                 {:6d}     {}".format(percent, n, "{:.6g}".format(cutoff) if n else "n/a"))
        summaries[band] = {"observed_prototypes": observed, "enriched_prototypes": enriched,
                           "support_pass_prototypes": supported, "eligible_prototypes": len(ranked),
                           "failure_counts_nonexclusive": dict(failures), "top_rank_percentages": fractions}
        print("  FINAL {}: {}/{} targets; IDs = {}".format(band.upper(), len(chosen), requested_count, [int(p) for p in chosen]))
        if len(chosen) < requested_count:
            print("  Shortfall: keeping only qualified IDs; no unsupported IDs added.")
    common_count = min(finalized["high"]["selected_count"], finalized["low"]["selected_count"])
    finalized["equal_count_deletion_groups"] = [
        {"count": count, "high_prototype_ids": finalized["high"]["prototype_ids"][:count],
         "low_prototype_ids": finalized["low"]["prototype_ids"][:count]}
        for count in args.deletion_counts if count <= common_count]
    finalized["unavailable_deletion_counts"] = [count for count in args.deletion_counts if count > common_count]
    finalized["maximum_equal_count_available"] = common_count
    print("\nEqual-count HIGH/LOW deletion doses: {}".format(
        [g["count"] for g in finalized["equal_count_deletion_groups"]]))
    return summaries, finalized


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(["stage", "prototype_id"] + [key for row in rows for key in row]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def export_video_ids(manifest, output_dir):
    """Export the saved split exactly; never shuffle, resplit, or alter targets."""
    lists = {}
    normalized = {}
    for split in ("selection", "evaluation"):
        values = manifest.get(split + "_videos")
        if not isinstance(values, list) or not values:
            raise ValueError("Manifest needs a nonempty " + split + "_videos list.")
        for value in values:
            identifier(value, split + " video ID")
        # The local evaluator compares Path(filename).stem. Catch collisions
        # here instead of silently dropping duplicates or leaking videos.
        ids = [Path(str(value)).stem for value in values]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(split + " IDs must be unique after Path(filename).stem normalization.")
        lists[split] = values
        normalized[split] = set(ids)
    if normalized["selection"] & normalized["evaluation"]:
        raise ValueError("Selection/evaluation videos overlap after filename normalization.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, values in lists.items():
        write_json(output_dir / (split + "_video_ids.json"), values)
    print("Exported {} selection / {} evaluation video IDs from the existing split to {}".format(
        len(lists["selection"]), len(lists["evaluation"]), output_dir.resolve()))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--extract-activations", action="store_true", help="Extract or reuse all-stage activation JSON, then score/select (also the default with no input-source flag).")
    inputs.add_argument("--patch-data", type=Path, nargs="+", help="One mixed-stage JSON array, or several stage-labelled JSON arrays.")
    inputs.add_argument("--stage-patch-data", nargs="+", metavar="STAGE=PATH", help="Explicit bank labels for separate exports, e.g. stage1=patch_stage1.json through stage4=patch_stage4.json.")
    inputs.add_argument("--finalize-existing", type=Path, help="Reuse existing scores, support counts and split from this directory; no patch-data scan.")
    inputs.add_argument("--export-video-ids-only", type=Path,
                        help="Read split_manifest.json in this directory and export its two video lists; do not recompute scores or targets.")
    parser.add_argument("--output-dir", type=Path, help="Default: existing-score directory, or prototype_selection for a fresh run.")
    parser.add_argument("--activation-json", type=Path, help="Cache path; default output-dir/stage_top8_activations.json. Existing JSON is always reused.")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT), help="Checkpoint for first-time extraction only.")
    parser.add_argument("--val-dir", type=Path, default=Path(DEFAULT_VAL_DIR), help="Validation dataset directory for first-time extraction only.")
    parser.add_argument("--src-root", type=Path, help="Repository src directory containing train.py; normally found through script parents.")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--window-len", type=int, help="Default: train.WINDOW_LEN.")
    parser.add_argument("--batch-size", type=int, help="Default: train._dataloader_batch_size().")
    parser.add_argument("--num-workers", type=int, help="Default: train.NUM_WORKERS.")
    parser.add_argument("--stride", type=int, default=32, help="Extraction window stride, matching the deletion evaluator default.")
    parser.add_argument("--no-amp", action="store_true", help="Disable training AMP settings for extraction.")
    parser.add_argument("--selection-fraction", type=float, default=0.5, help="Fraction of videos assigned to selection (default 0.5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=Path, help="Reuse this saved video split when computing fresh all-stage scores. Default: reuse output-dir/split_manifest.json if present.")
    parser.add_argument("--top-percent", type=float, default=5.0)
    parser.add_argument("--bottom-percent", type=float, default=70.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-prototypes", type=int, default=512)
    parser.add_argument("--smoothing", type=float, default=0.5, help="Positive Beta prior pseudocount (default 0.5).")
    parser.add_argument("--min-band-patches", type=int, default=100)
    parser.add_argument("--min-band-videos", type=int, default=5)
    parser.add_argument("--min-selectivity", type=float, default=1.0, help="Require strictly greater E(G) (default 1).")
    parser.add_argument("--high-target-count", type=int, default=15)
    parser.add_argument("--low-target-count", type=int, default=18)
    parser.add_argument("--deletion-counts", type=int, nargs="+", default=[1, 2, 4, 8, 15])
    parser.add_argument("--saliency-field", default="patch_pred_saliency")
    parser.add_argument("--video-field")
    parser.add_argument("--frame-field")
    parser.add_argument("--stage", help="Optional single-bank mode (1..4 or stage1..stage4); default requires all four stages.")
    parser.add_argument("--temp-dir", type=Path, help="Directory with space for the temporary compact SQLite archive.")
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args(argv)
    if args.gpu_id < 0 or args.stride < 1 or (args.num_workers is not None and args.num_workers < 0):
        parser.error("GPU/workers must be nonnegative; stride must be positive.")
    if any(value is not None and value < 1 for value in (args.window_len, args.batch_size)):
        parser.error("Window length and batch size must be positive.")
    if not math.isfinite(args.selection_fraction) or not 0 < args.selection_fraction < 1:
        parser.error("--selection-fraction must be between 0 and 1.")
    if not all(math.isfinite(p) and 0 < p < 100 for p in (args.top_percent, args.bottom_percent)) or args.top_percent + args.bottom_percent > 100:
        parser.error("Band percentages must be positive, below 100, and sum to <= 100.")
    if args.top_k != 8 or not args.top_k <= args.num_prototypes <= 2**32:
        parser.error("This selector requires --top-k 8; --num-prototypes must cover it (max 2**32).")
    if not math.isfinite(args.smoothing) or args.smoothing <= 0:
        parser.error("--smoothing must be positive and finite.")
    if min(args.min_band_patches, args.min_band_videos) < 1 or args.progress_every < 0:
        parser.error("Minimum support must be >=1; progress interval must be >=0.")
    if not math.isfinite(args.min_selectivity) or args.min_selectivity < 1:
        parser.error("--min-selectivity must be finite and >=1.")
    if min(args.high_target_count, args.low_target_count, *args.deletion_counts) < 1:
        parser.error("Target counts and deletion counts must be positive.")
    args.deletion_counts = sorted(set(args.deletion_counts))
    try:
        if args.stage is not None:
            args.stage = canonical_stage(args.stage)
        if args.stage_patch_data:
            labelled = []
            for value in args.stage_patch_data:
                name, sep, path = value.partition("=")
                if not sep or not path:
                    raise ValueError("--stage-patch-data entries must be STAGE=PATH.")
                labelled.append((canonical_stage(name), Path(path)))
            if len({name for name, _ in labelled}) != len(labelled):
                raise ValueError("--stage-patch-data must specify each stage only once.")
            args.stage_patch_data = labelled
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir is None:
        args.output_dir = args.export_video_ids_only or args.finalize_existing or Path("prototype_selection")
    if args.activation_json is None:
        args.activation_json = args.output_dir / "stage_top8_activations.json"
    return args


def compute_from_patches(args):
    paths = [(stage, path.resolve()) for stage, path in args.stage_patch_data] if args.stage_patch_data else [(None, p.resolve()) for p in args.patch_data]
    initial_stats = {path: path.stat() for _, path in paths}
    tables, stage_metadata, frame_manifest = {}, {}, {"selection": [], "evaluation": []}
    with tempfile.TemporaryDirectory(prefix="prototype_selection_", dir=args.temp_dir) as tmp:
        db = sqlite3.connect(str(Path(tmp) / "patches.sqlite"))
        try:
            frames, schemas, count, pack = ingest(paths, db, args)
            for path, initial in initial_stats.items():
                final = path.stat()
                if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                    raise ValueError("Input changed while being read: " + str(path))
            videos = sorted({video for _, video, _ in frames})
            if len(videos) < 2:
                raise ValueError("At least two videos are required for a video-disjoint split.")
            normalized = [Path(v).stem for v in videos]
            if len(normalized) != len(set(normalized)):
                raise ValueError("Source videos collide after filename normalization.")
            available = set(normalized)
            for stage in expected_stages(args):
                stage_videos = {Path(v).stem for s, v, _ in frames if s == stage}
                if stage_videos != available:
                    raise ValueError("{} is missing videos {}. Export the same video set at every stage.".format(stage, sorted(available - stage_videos)))
            saved_split = args.split_manifest
            if saved_split is None and (args.output_dir / "split_manifest.json").is_file():
                saved_split = args.output_dir / "split_manifest.json"
            if saved_split:
                manifest = read_json(saved_split)
                lists = split_lists(manifest)
                if lists["selection"] | lists["evaluation"] != available:
                    raise ValueError("Saved split and exported video set differ; supply all saved selection/evaluation videos at every stage.")
                # Lists and their order are copied exactly, not resampled.
                manifest = dict(manifest)
                selection = lists["selection"]
                print("Reusing saved shared video split from " + str(saved_split))
            else:
                shuffled = videos[:]
                random.Random(args.seed).shuffle(shuffled)
                n_selection = min(len(videos) - 1, max(1, math.floor(len(videos) * args.selection_fraction + 0.5)))
                manifest = {"schema_version": 3, "split_unit": "video", "seed": args.seed,
                    "selection_fraction_requested": args.selection_fraction,
                    "selection_videos": sorted(shuffled[:n_selection]), "evaluation_videos": sorted(shuffled[n_selection:])}
                selection = split_lists(manifest)["selection"]
            print("Shared split: {} selection / {} evaluation videos.".format(len(manifest["selection_videos"]), len(manifest["evaluation_videos"])))
            for stage in expected_stages(args):
                print("\nComputing {} top-5%/low bands on selection videos only...".format(stage), flush=True)
                stage_frames = {(video, frame): value for (s, video, frame), value in frames.items() if s == stage}
                try:
                    counts, frame_counts, video_sets, totals, skipped, stage_manifest = collect(db, stage_frames, selection, args, pack)
                except ValueError as exc:
                    raise ValueError(stage + ": " + str(exc)) from exc
                tables[stage] = output_scores(counts, frame_counts, video_sets, totals, args)
                stage_metadata[stage] = {"input_schema": schemas[stage], "totals": dict(totals),
                    "skipped_selection_frames": dict(skipped), "source_patch_count": sum(v["patch_count"] for v in stage_frames.values())}
                for split in frame_manifest:
                    frame_manifest[split].extend(dict(item, stage=stage) for item in stage_manifest[split])
                print("{}: {:,} HIGH / {:,} LOW patches from {:,} usable selection frames.".format(stage, totals["high"], totals["low"], totals["eligible_frames"]))
        finally:
            db.close()
    metadata = {"schema_version": 3, "stages": stage_metadata,
        "sources": [{"path": str(path), "stage_label": stage, "size_bytes": initial_stats[path].st_size, "mtime_ns": initial_stats[path].st_mtime_ns} for stage, path in paths],
        "source_patch_count": count, "settings": json.loads(json.dumps(vars(args), default=str)),
        "selectivity_unit": "individual prototype ID within its own independent stage bank",
        "high_formula": "P(j active | high) / P(j active | low)",
        "low_formula": "P(j active | low) / P(j active | high)",
        "probability_estimator": "(support + smoothing) / (band_patches + 2*smoothing)",
        "aggregation": "patch-pooled within each stage across selection videos; each ID once per patch",
        "band_rule": "frame-local within each stage; ceil counts; stable ties; shared cutoff scores excluded from both bands",
        "top_k": args.top_k, "top_percent": args.top_percent, "bottom_percent": args.bottom_percent}
    manifest.update(schema_version=3, stages=list(expected_stages(args)), frames=frame_manifest,
                    shared_across_stages=True, split_reused_from=str(saved_split.resolve()) if saved_split else None)
    return tables, metadata, manifest


def load_existing(args):
    """Rerank stage-separated saved scores without changing the shared split."""
    root = args.finalize_existing
    scores = {band: read_json(root / (band + "_prototypes.json")) for band in ("high", "low")}
    metadata = read_json(root / "selection_metadata.json")
    manifest = read_json(root / "split_manifest.json")
    split_lists(manifest)
    if "stages" in scores["high"] and "stages" in scores["low"]:
        high, low = scores["high"]["stages"], scores["low"]["stages"]
        if set(high) != set(low):
            raise ValueError("HIGH/LOW files disagree on their stage banks.")
        tables = {stage: {"high": high[stage], "low": low[stage]} for stage in high}
        stored_k = metadata.get("top_k", metadata.get("settings", {}).get("top_k"))
    else:
        # Old score dictionaries contain ONE bank, not interchangeable IDs at
        # all stages. Import only in an explicit single-stage run.
        if not args.stage:
            raise ValueError("Existing files contain only one stage. --finalize-existing cannot create missing stage scores; rescan all-stage --patch-data using the existing split.")
        stage = canonical_stage(metadata.get("input_schema", {}).get("stage"))
        if stage != args.stage:
            raise ValueError("Requested --stage differs from the saved bank.")
        stored_k = metadata.get("settings", {}).get("top_k")
        tables = {stage: scores}
        metadata = {**metadata, "stages": {stage: {"input_schema": metadata["input_schema"],
                    "totals": metadata.get("totals", {}), "skipped_selection_frames": metadata.get("skipped_selection_frames", {})}},
                    "top_k": stored_k, "top_percent": metadata.get("settings", {}).get("top_percent"),
                    "bottom_percent": metadata.get("settings", {}).get("bottom_percent")}
    if stored_k != 8:
        raise ValueError("Saved scores must identify top_k=8; recompute from top-8 exports.")
    missing = set(expected_stages(args)) - set(tables)
    if missing:
        raise ValueError("Saved scores are missing banks {}. Rescan all-stage patch exports.".format(sorted(missing)))
    tables = {s: tables[s] for s in expected_stages(args)}
    for stage, bands in tables.items():
        if stage not in metadata.get("stages", {}):
            raise ValueError("Missing stage metadata for " + stage)
        for entries in bands.values():
            if not isinstance(entries, dict) or any(not isinstance(entry, dict) for entry in entries.values()):
                raise ValueError("Stage scores must contain prototype-ID-keyed dictionaries.")
    print("Reusing stage-specific selection scores from {}; preserving saved video split.".format(root.resolve()))
    return tables, metadata, manifest


def main(argv=None):
    args = parse_args(argv)
    if args.export_video_ids_only:
        manifest = read_json(args.export_video_ids_only / "split_manifest.json")
        export_video_ids(manifest, args.output_dir)
        return 0
    cache_metadata = None
    if args.extract_activations or not (args.patch_data or args.stage_patch_data or args.finalize_existing):
        cache_metadata = ensure_activation_cache(args)
        args.patch_data = [args.activation_json]
        provenance = cache_metadata.get('provenance')
        if provenance:
            cached_bank_size = provenance.get('model_config', {}).get('num_prototypes')
            if cached_bank_size is not None:
                args.num_prototypes = integer(cached_bank_size, 'cached prototype bank size', 8)
    tables, metadata, manifest = load_existing(args) if args.finalize_existing else compute_from_patches(args)
    if cache_metadata is not None:
        metadata['activation_cache'] = cache_metadata
    finalized = {"schema_version": 3, "stage": args.stage,
        "selection_videos": manifest["selection_videos"], "evaluation_videos": manifest["evaluation_videos"],
        "top_k": metadata["top_k"], "top_percent": metadata["top_percent"],
        "bank_identity": "independent stage banks; targets are selected separately, never copied across stages",
        "stages": {}}
    summaries = {}
    all_rows = {"high": [], "low": []}
    for stage, bands in tables.items():
        print("\n========== {}: independent candidate ranking ==========".format(stage))
        stage_meta = metadata["stages"][stage]
        summary, chosen = rank_and_finalize(bands, args, stage_meta, manifest)
        chosen.update(schema_version=3, stage=stage)
        finalized["stages"][stage] = chosen
        summaries[stage] = summary
        for band, entries in bands.items():
            for proto, entry in entries.items():
                all_rows[band].append({"stage": stage, "prototype_id": int(proto), **entry})
    finalized["high_prototype_ids"] = {s: f["high"]["prototype_ids"] for s, f in finalized["stages"].items()}
    finalized["high_candidate_prototype_ids"] = {s: f["high"]["candidate_prototype_ids"] for s, f in finalized["stages"].items()}
    finalized["low_prototype_ids"] = {s: f["low"]["prototype_ids"] for s, f in finalized["stages"].items()}
    metadata.update(schema_version=3, rank_summaries=summaries,
        ranking="independent within-stage candidates: all prototypes by descending smoothed E(G); targets: support and E(G)>threshold filter then descending smoothed E(G), ties by numeric ID",
        finalization_settings={**next(iter(finalized["stages"].values()))["selection_rule"],
            "high_target_count": args.high_target_count, "low_target_count": args.low_target_count,
            "deletion_counts": args.deletion_counts})
    if args.finalize_existing:
        metadata["finalization_input_directory"] = str(args.finalize_existing.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for band in ("high", "low"):
        write_json(args.output_dir / (band + "_prototypes.json"),
                   {"schema_version": 3, "top_k": metadata["top_k"], "stages": {s: b[band] for s, b in tables.items()}})
        write_csv(args.output_dir / (band + "_prototypes.csv"), all_rows[band])
    candidate_rows = [dict(row, band=band.upper()) for band, rows in all_rows.items() for row in rows if row["candidate"]]
    candidate_rows.sort(key=lambda row: (STAGES.index(row["stage"]), row["band"], row["candidate_rank"]))
    write_csv(args.output_dir / "finalized_targets.csv", candidate_rows)
    write_json(args.output_dir / "split_manifest.json", manifest)
    write_json(args.output_dir / "selection_metadata.json", metadata)
    write_json(args.output_dir / "finalized_targets.json", finalized)
    export_video_ids(manifest, args.output_dir)
    print("Wrote per-stage candidate scores, CSVs, shared split and finalized targets to {}".format(args.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, sqlite3.Error, struct.error) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
