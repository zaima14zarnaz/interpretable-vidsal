#!/usr/bin/env python3
"""Find recurring visual patterns using the prototype graph and one Gemini request per salient anchor.

Place beside generate_prot_graph.py. No visual-similarity graph/script is
needed. The prototype graph must have the new all-prototype activation cache,
ranked prototype_patches index, visual_descriptors and generated neighborhood.json.
Run ACTION='cache_similarity' and then ACTION='view'/'serve' in the prototype
generator once if upgrading an older database.

Example:
  python find_vis_pattern.py --video-id 001 --frame-name YOUR_FRAME.png \
      --prototype-graph-dir /path/prototype_graph_outputs --prepare-only

For each salient stage4 anchor, rank ALL native visual_activations and retain
its top 8 prototype IDs (or the whole bank if smaller). For each candidate,
retrieve exactly its N connected graph patches using indexed prototype/rank
lookups. N comes from the generated graph snapshot, not the ranking index's
larger cache capacity. Mean visual similarity is evaluated separately against
every graph member using stage4 cosine + (1 - RGB MAE) + RGB structure cosine.
Choose the prototype with the HIGHEST mean (ties: lowest prototype ID).

The winning prototype's complete group becomes the report's retrieved matches;
there is no cross-video deduplication, similarity threshold or --top-k cap.
The anchor participates in the group mean if it is already a graph member,
but is removed from matches so it appears exactly once in the model's inputs.
One Gemini request receives the anchor first plus all other winning-group
patches, in their graph activation-rank order, using the unchanged prompt.
Candidate IDs, activations, group scores and winner are saved in report.json;
prototype overlap still uses the original stored top-k candidate IDs.

The HTML template, visible wording, context-crop prompt, image preprocessing,
rate limits, retry/resume behavior and saliency overlay remain unchanged.
--patch-grid-cols controls layout (default graph N + 1 on new retrieval runs).
--top-k and old visual-graph/threshold options are accepted for command-line
compatibility but do not limit or filter the new prototype-group retrieval.
--min-group-size still controls the minimum number of retrieved matches needed
to query Gemini; the anchor is not counted toward that minimum.

Graph files remain read-only. Only the selected frame's original full temporal
window is replayed; raw stage4 features, loader-RGB descriptors, activations,
assignments, patch saliency, checkpoint and evaluation provenance are checked
against the prototype cache. Groups/descriptors are reused across anchors.

Set GEMINI_API_KEY in the environment or privately below before requesting
answers. --prepare-only sends no API requests. --describe-only --output RUN_DIR
resumes saved queries without inference/retrieval; --reset-descriptions archives
old answers before regenerating them. --view-only --output RUN_DIR rebuilds and
serves the saved report at http://127.0.0.1:8765/report.html (no inference/API
requests). Use --no-browser or --port as before. Existing reports retain their
saved retrieval; use a new output directory for prototype-based retrieval.

Outputs: report.html, report.md, report.json, raw prediction arrays, frame.png,
predicted_saliency.png, salient_anchors.png, crops/, api_crops/ and frames/.
Dependencies: existing saliency environment, NumPy and Pillow. Viewing/resuming
saved reports does not require either graph generator or PyTorch inference.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gc
import html
from io import BytesIO
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import webbrowser
import zlib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from email.utils import parsedate_to_datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
# Prefer the GEMINI_API_KEY environment variable. Optionally paste a local key
# here for private runs; never commit a real key. Not saved in reports.
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-3.5-flash-lite"
MIN_REQUEST_INTERVAL_SECONDS = 65.0
SALIENCY_PERCENTILE = 0.97
TOP_K = 30  # Legacy CLI default; prototype groups are never truncated.
ANCHOR_PROTOTYPE_COUNT = 8
PROTOTYPE_RETRIEVAL_MODE = "top8_activation_best_mean_visual_group_v1"
PATCH_GRID_COLS = None  # None selects top_k + 1; CLI can override with an integer.
MIN_GROUP_SIZE = 2
STAGE4_COSINE_RELAXATION = 0.2
MAX_RGB_MAE = 0.6
MIN_RGB_STRUCTURE_COSINE = 0.2
API_IMAGE_MAX_SIDE = 256
API_JPEG_QUALITY = 90
API_CONTEXT_PADDING = 0.5  # Fraction of original patch width/height on EACH side.
API_CONTEXT_BOX_WIDTH = 2  # Drawn in transmitted-image pixels, after downscaling.
API_CONTEXT_MODE = "padded_patch_red_box_v1"
CONTEXT_INSTRUCTIONS = """Each image is a context crop. The RED RECTANGLE marks the original target patch.
Compare only the visual content INSIDE the rectangles. Use the surroundings
only to help identify what the boxed content depicts. Ignore the red outlines
themselves. Distinguish visible appearance from uncertain semantic inferences.
The first image is the ANCHOR PATCH; subsequent images are numbered matches.

"""
HEATMAP_ALPHA = 0.5  # Overlay opacity for predicted saliency on the RGB frame.
HEATMAP_DISPLAY_FORMAT = "jet_overlay_v1"
GROUPS = ("all", "salient", "non_salient")
GROUP_LABELS = {"all": "All patches", "salient": "Salient patches only", "non_salient": "Non-salient patches"}
METRICS = ("stage4_cosine", "rgb_mae", "rgb_structure_cosine")


def jet_colormap(values):
    """Map normalized values in [0, 1] to an RGB JET heatmap (float32 in [0, 1])."""
    t = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    # Piecewise OpenCV-style JET: dark blue → cyan → green → yellow → red.
    r = np.clip(1.5 - 4.0 * np.abs(t - 0.75), 0.0, 1.0)
    g = np.clip(1.5 - 4.0 * np.abs(t - 0.5), 0.0, 1.0)
    b = np.clip(1.5 - 4.0 * np.abs(t - 0.25), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def normalize_saliency_for_display(prediction):
    """Per-frame min-max normalize a single-channel map to [0, 1] for visualization."""
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("Predicted saliency must be a non-empty 2-D map.")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Predicted saliency contains no finite values.")
    cleaned = np.where(finite, values, np.nan)
    low = float(np.nanmin(cleaned))
    high = float(np.nanmax(cleaned))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Predicted saliency extrema are non-finite.")
    if high == low:
        normalized = np.full(values.shape, 0.5, dtype=np.float64)
        constant = True
    else:
        normalized = (cleaned - low) / (high - low)
        normalized = np.where(finite, normalized, 0.5)
        constant = False
    return np.clip(normalized, 0.0, 1.0), low, high, constant


def resize_saliency_bilinear(prediction, width, height):
    """Bilinearly resize a float saliency map to (height, width) display pixels."""
    values = np.asarray(prediction, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Predicted saliency must be 2-D before resizing.")
    if width < 1 or height < 1:
        raise ValueError("Display frame dimensions must be positive.")
    if values.shape == (height, width):
        return values.astype(np.float64, copy=False)
    image = Image.fromarray(values, mode="F")
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
    resized = image.resize((int(width), int(height)), resample=resample)
    return np.asarray(resized, dtype=np.float64)


def saliency_heatmap_overlay(frame_rgb, prediction, alpha=HEATMAP_ALPHA):
    """Blend a JET heatmap of predicted saliency onto an RGB frame.

    frame_rgb: HxWx3 uint8 or float RGB in [0,1]/[0,255]
    prediction: 2-D single-channel map (any resolution); resized to the frame.
    Returns (overlay_uint8_rgb, metadata). Does not mutate prediction.npy.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Heatmap opacity must be a finite value in [0, 1].")
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must be an HxWx3 RGB image.")
    height, width = frame.shape[:2]
    if frame.dtype == np.uint8:
        original = frame.astype(np.float32) / 255.0
    else:
        original = np.asarray(frame, dtype=np.float32)
        if original.max() > 1.5:
            original = original / 255.0
        original = np.clip(original, 0.0, 1.0)

    resized = resize_saliency_bilinear(prediction, width, height)
    normalized, low, high, constant = normalize_saliency_for_display(resized)
    heatmap = jet_colormap(normalized)
    overlay = (1.0 - alpha) * original + alpha * heatmap
    overlay_u8 = np.rint(np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)
    meta = {
        "minimum": low,
        "maximum": high,
        "constant": constant,
        "alpha": alpha,
        "colormap": "jet",
        "format": HEATMAP_DISPLAY_FORMAT,
        "prediction_shape": [int(n) for n in np.asarray(prediction).shape],
        "frame_shape": [int(height), int(width), 3],
        "scaling": "per-frame min-max to [0,1] for display only; bilinear resize to frame size",
    }
    return overlay_u8, meta


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_module(path, name):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Place this script beside both graph generators, or pass their paths.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def select_frame(con, video, dataset=None, frame_name=None, frame_no=None):
    conditions, values = ["video=?"], [video]
    if dataset is not None:
        conditions.append("dataset=?"); values.append(dataset)
    if frame_name is not None:
        conditions.append("target_name=?"); values.append(frame_name)
    else:
        if frame_no is None or frame_no < 1:
            raise ValueError("--frame-no must be a one-based frame position >= 1.")
        conditions.append("target_index=?"); values.append(frame_no - 1)
    rows = con.execute("SELECT * FROM frames WHERE " + " AND ".join(conditions), values).fetchall()
    if not rows:
        raise ValueError("Requested target frame is absent from the graph. It must be an extracted target frame with a full temporal history; check --frame-name, --frame-no, --dataset and graph extraction stride/coverage. No nearby frame will be substituted.")
    if len(rows) != 1:
        raise ValueError("Frame is ambiguous across graph datasets/windows. Specify --dataset or use a graph with unique target windows.")
    return dict(rows[0])


def sort_key(hit):
    return (-hit["stage4_cosine"], hit["rgb_mae"], -hit["rgb_structure_cosine"], hit["node_id"])


def prepare_api_image(path, max_side=API_IMAGE_MAX_SIDE, jpeg_quality=API_JPEG_QUALITY):
    """Downscale/re-encode one crop in memory for the Gemini request only.

    Returns (mime_type, encoded_bytes, stats). Disk files are never modified.
    """
    path = Path(path)
    original_bytes = path.read_bytes()
    with Image.open(BytesIO(original_bytes)) as image:
        original_size = image.size
        rgb = image.convert("RGB")
    width, height = rgb.size
    longest = max(width, height)
    resized = False
    if longest > max_side:
        scale = max_side / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)
        rgb = rgb.resize(new_size, resample)
        resized = True
    transmitted_size = rgb.size

    png_buffer = BytesIO()
    rgb.save(png_buffer, format="PNG", optimize=True)
    png_bytes = png_buffer.getvalue()

    jpeg_buffer = BytesIO()
    rgb.save(jpeg_buffer, format="JPEG", quality=int(jpeg_quality), optimize=True)
    jpeg_bytes = jpeg_buffer.getvalue()

    candidates = [("image/png", png_bytes), ("image/jpeg", jpeg_bytes)]
    if not resized and original_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        # Keep original PNG bytes when they are already smaller than re-encodes.
        candidates.append(("image/png", original_bytes))
    mime_type, encoded = min(candidates, key=lambda item: (len(item[1]), 0 if item[0] == "image/png" else 1))
    stats = {
        "path": str(path),
        "original_size": list(original_size),
        "transmitted_size": list(transmitted_size),
        "original_bytes": len(original_bytes),
        "transmitted_bytes": len(encoded),
        "encoding": "jpeg" if mime_type == "image/jpeg" else "png",
        "mime_type": mime_type,
        "resized": resized,
        "retained_original": (not resized) and encoded is original_bytes,
    }
    return mime_type, encoded, stats


def llm_selection(args):
    return "gemini", getattr(args, "gemini_model", GEMINI_MODEL)


class PrototypeGroupRetriever:
    """Read-only access to the exact groups displayed by the prototype graph.

    No visual-similarity graph, materialized patch edges or global patch search.
    Native activations, indexed group memberships and descriptors are reused.
    """
    def __init__(self, helper, directory):
        self.helper = helper
        self.directory = self.source_dir = Path(directory).resolve()
        required = ("connect_db", "get_node", "require_dense_cache", "load_descriptors",
                    "visual_metrics", "Stage4Capture", "rgb_descriptor", "extract_stage")
        missing = [name for name in required if not hasattr(helper, name)]
        if missing:
            raise ValueError(f"Prototype generator is outdated (missing {missing}). Use the updated generate_prot_graph.py.")
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("status") != "complete":
            raise ValueError("Prototype graph extraction is not complete.")
        self.manifest_sha256 = helper.sha256_file(self.directory / "manifest.json")
        self.con = helper.connect_db(self.directory / "graph.sqlite", readonly=True)
        try:
            self.prototype_count = helper.require_dense_cache(self.con)
            setting = self.con.execute("SELECT value FROM similarity_metadata WHERE key='settings'").fetchone()
            if not setting:
                raise ValueError("Prototype graph is missing its visual descriptor settings.")
            self.settings = json.loads(setting[0])
            if self.settings.get("version") != helper.SIMILARITY_VERSION:
                raise ValueError("Prototype descriptor version differs from the loaded generator. Use the matching prototype script.")
            # Replay the descriptor source recorded at extraction time.
            helper.STAGE4_FEATURE_MODULE = self.settings["feature_module"]
            helper.STAGE4_FEATURE_OUTPUT_PATH = tuple(self.settings["feature_output_path"])
            helper.STAGE4_FEATURE_LAYOUT = self.settings["feature_layout"]
            helper.RGB_DESCRIPTOR_SIZE = int(self.settings["rgb_size"])
            helper.TOP_K = int(self.manifest["top_k"])
            if helper.similarity_settings() != self.settings:
                raise ValueError("Prototype descriptor semantics differ from the loaded generator.")
            snapshot_path = self.directory / "neighborhood.json"
            if snapshot_path.is_file():
                snapshot = json.loads(snapshot_path.read_text())
                if snapshot.get("graph_version") != getattr(helper, "GRAPH_VERSION", None):
                    raise ValueError("Graph snapshot is not the new prototype/patch graph. Regenerate it with ACTION='view'.")
                self.group_size = int(snapshot["patches_per_prototype"])
                if snapshot.get("total_prototypes") != self.prototype_count:
                    raise ValueError("Prototype snapshot bank size differs from graph.sqlite. Regenerate ACTION='view'.")
                self.group_size_source = "neighborhood.json:patches_per_prototype"
            elif self.manifest.get("patches_per_prototype") is not None:
                self.group_size = int(self.manifest["patches_per_prototype"])
                self.group_size_source = "manifest.json:patches_per_prototype"
            else:
                raise ValueError("Cannot determine the generated graph's N. Run ACTION='view' in the prototype generator to write neighborhood.json.")
            if self.group_size < 1:
                raise ValueError("Prototype graph N must be positive.")
            rank_row = self.con.execute("SELECT value FROM prototype_graph_metadata WHERE key='ranking'").fetchone()
            if not rank_row:
                raise ValueError("Prototype ranking cache is invalid/missing. Run ACTION='view' in the prototype generator.")
            self.ranking_metadata = json.loads(rank_row[0])
            if self.ranking_metadata.get("capacity", 0) < self.group_size:
                raise ValueError("Prototype index contains fewer than the generated N matches. Regenerate ACTION='view'.")
            patch_count = int(self.con.execute("SELECT COUNT(*) FROM nodes WHERE stage='stage4'").fetchone()[0])
            if (self.ranking_metadata.get("patch_count") != patch_count or
                    self.ranking_metadata.get("prototype_count") != self.prototype_count):
                raise ValueError("Prototype index is stale. Regenerate ACTION='view' before retrieving report patches.")
            self.expected_group_size = min(self.group_size, patch_count)
            self.source = "prototype_patches_sqlite"
            self._nodes, self._descriptors, self._groups, self._activation_frames = {}, {}, {}, {}
        except BaseException:
            self.con.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.con.close()

    def node(self, node_id):
        node_id = int(node_id)
        if node_id not in self._nodes:
            self._nodes[node_id] = self.helper.get_node(self.con, node_id)
        return dict(self._nodes[node_id])

    def descriptors(self, node_ids):
        ids = list(dict.fromkeys(map(int, node_ids)))
        missing = [nid for nid in ids if nid not in self._descriptors]
        if missing:
            self._descriptors.update(self.helper.load_descriptors(self.con, missing))
        unavailable = [nid for nid in ids if nid not in self._descriptors]
        if unavailable:
            raise ValueError(f"Visual descriptors missing for graph patches {unavailable[:10]}. Complete ACTION='cache_similarity' before making this report.")
        return {nid: self._descriptors[nid] for nid in ids}

    def frame_activations(self, frame_id, node_ids):
        frame_id = int(frame_id)
        if frame_id not in self._activation_frames:
            row = self.con.execute("SELECT * FROM prototype_activation_frames WHERE frame_id=? AND stage='stage4'", (frame_id,)).fetchone()
            if row is None:
                raise ValueError(f"No all-prototype activations for frame {frame_id}. Run ACTION='cache_similarity'.")
            ids = np.frombuffer(zlib.decompress(row["node_ids"]), dtype='<i8')
            values = np.frombuffer(zlib.decompress(row["activations"]), dtype='<f4').reshape(row["patch_count"], row["prototype_count"])
            if (len(ids) != len(values) or len(set(ids.tolist())) != len(ids)
                    or values.shape[1] != self.prototype_count or not np.isfinite(values).all()):
                raise ValueError("Corrupt all-prototype activation frame cache.")
            self._activation_frames[frame_id] = ({int(nid): i for i, nid in enumerate(ids)}, values)
        positions, values = self._activation_frames[frame_id]
        try:
            return values[[positions[int(nid)] for nid in node_ids]]
        except KeyError as error:
            raise ValueError(f"Patch {error.args[0]} is absent from its frame activation block.") from error

    def members(self, prototype_id):
        prototype_id = int(prototype_id)
        if not 0 <= prototype_id < self.prototype_count:
            raise ValueError(f"Unknown stage4 prototype {prototype_id}.")
        if prototype_id not in self._groups:
            rows = list(self.con.execute("""SELECT p.node_id,p.rank,p.activation,n.stage FROM prototype_patches p
                JOIN nodes n ON n.id=p.node_id
                WHERE p.stage='stage4' AND p.prototype_id=? AND p.rank<=? ORDER BY p.rank""",
                (prototype_id, self.group_size)))
            if (len(rows) != self.expected_group_size or
                    [int(row["rank"]) for row in rows] != list(range(1, len(rows)+1)) or
                    any(row["stage"] != "stage4" for row in rows) or
                    len({int(row["node_id"]) for row in rows}) != len(rows)):
                raise ValueError(f"Prototype {prototype_id} does not have its complete generated graph group. Regenerate ACTION='view'.")
            self._groups[prototype_id] = [dict(row) for row in rows]
        return self._groups[prototype_id]

    def select(self, anchor):
        """Highest mean over all graph members among the anchor's top-8 activations.

        Self-membership contributes its actual similarity to that group's mean,
        as for every other graph member. It is removed only from output matches
        because the anchor is already image 1 of the unchanged Gemini prompt.
        """
        if anchor.get("stage") != "stage4":
            raise ValueError("Prototype-group retrieval is defined for stage4 anchors.")
        weights = self.frame_activations(anchor["frame_id"], [anchor["id"]])[0]
        candidates = np.lexsort((np.arange(self.prototype_count), -weights))[:min(ANCHOR_PROTOTYPE_COUNT, self.prototype_count)]
        groups = {int(pid): self.members(pid) for pid in candidates}
        needed = [anchor["id"], *[row["node_id"] for group in groups.values() for row in group]]
        descriptors = self.descriptors(needed)
        scores, candidate_hits = [], {}
        pair_cache = {}
        for candidate_rank, pid in enumerate(candidates, 1):
            pid = int(pid)
            group = groups[pid]
            hits = []
            for row in group:
                nid = int(row["node_id"])
                if nid not in pair_cache:
                    pair_cache[nid] = self.helper.visual_metrics(descriptors[int(anchor["id"])], descriptors[nid])
                metrics = pair_cache[nid]
                score = metrics.get("visual_score")
                if score is None or not math.isfinite(score):
                    raise ValueError(f"Cannot score every member of prototype {pid}: patch {nid}: {metrics.get('similarity_missing_reason')}. Repair the descriptor cache before retrieval.")
                hits.append({"node_id": nid, **metrics,
                             "prototype_graph_rank": int(row["rank"]), "prototype_activation": float(row["activation"])})
            mean_score = float(np.mean([hit["visual_score"] for hit in hits])) if hits else None
            scores.append({"prototype_id": pid, "anchor_activation_rank": candidate_rank,
                           "anchor_activation": float(weights[pid]), "graph_patch_count": len(group),
                           "graph_patch_ids": [hit["node_id"] for hit in hits],
                           "mean_visual_similarity": mean_score,
                           "anchor_is_graph_member": any(hit["node_id"] == anchor["id"] for hit in hits)})
            candidate_hits[pid] = hits
        eligible = [score for score in scores if score["mean_visual_similarity"] is not None]
        winner = min(eligible, key=lambda score: (-score["mean_visual_similarity"], score["prototype_id"])) if eligible else None
        selected_id = winner["prototype_id"] if winner else None
        hits = [dict(hit) for hit in candidate_hits.get(selected_id, []) if hit["node_id"] != anchor["id"]]
        trace = {"mode": PROTOTYPE_RETRIEVAL_MODE,
                 "top_activated_prototype_ids": [int(p) for p in candidates],
                 "activation_source": "native cached visual_activations, full bank; ties by prototype ID",
                 "visual_similarity_formula": self.helper.SIMILARITY_FORMULA,
                 "selection": "maximum mean anchor-to-group visual similarity; lowest prototype ID breaks ties",
                 "group_mean_scope": "every generated graph member, including anchor self-membership when present",
                 "graph_group_size": self.group_size, "group_size_source": self.group_size_source,
                 "candidate_scores": scores, "selected_prototype_id": selected_id,
                 "selected_mean_visual_similarity": winner["mean_visual_similarity"] if winner else None,
                 "selected_graph_patch_ids": winner["graph_patch_ids"] if winner else [],
                 "returned_match_ids": [hit["node_id"] for hit in hits],
                 "anchor_removed_from_match_list": bool(winner and winner["anchor_is_graph_member"])}
        return hits, trace


def validate_retrieved_matches(anchor, matches, report):
    if report.get("protocol", {}).get("retrieval_mode") != PROTOTYPE_RETRIEVAL_MODE:
        # Resuming old saved reports preserves their previous retrieval contract.
        return validate_match_videos(anchor, matches)
    ids = [int(node["id"]) for node in matches]
    if len(ids) != len(set(ids)) or int(anchor["id"]) in ids:
        raise ValueError("Prototype-group matches must be unique patches distinct from the anchor.")
    for node in matches:
        if node.get("stage") != "stage4":
            raise ValueError("Prototype-group matches must be stage4 patches.")
        video_key(node)  # Metadata is still needed by the existing report.

def video_key(node):
    """Video IDs are scoped by their graph dataset, as in cross-video retrieval."""
    if node.get("dataset") is None or node.get("video") is None:
        raise ValueError("Missing dataset/video metadata; cannot guarantee distinct-video matches. Prepare a new report.")
    return node["dataset"], node["video"]


def unique_video_hits(hits, anchor, top_k, key_for_hit):
    seen = {video_key(anchor)}
    selected = []
    for hit in hits:
        key = key_for_hit(hit)
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)
        if len(selected) >= top_k:
            break
    return selected


def validate_match_videos(anchor, matches):
    seen = {video_key(anchor)}
    for node in matches:
        key = video_key(node)
        if key in seen:
            raise ValueError("Matches must come from distinct other videos (one patch per video per anchor). "
                             "This saved report uses the old retrieval policy; prepare a new report with a new --output directory.")
        seen.add(key)


def split_matches(matches, cutoff):
    for node in matches:
        value = node.get("pred_percentile")
        if value is None or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid cached saliency percentile for node {node.get('id')}.")
    return {"all": matches,
            "salient": [n for n in matches if n["pred_percentile"] >= cutoff],
            "non_salient": [n for n in matches if n["pred_percentile"] < cutoff]}


def crop_box(node, size):
    w, h = size
    return (round(node["col"] * w / node["grid_w"]), round(node["row"] * h / node["grid_h"]),
            round((node["col"]+1) * w / node["grid_w"]), round((node["row"]+1) * h / node["grid_h"]))


def save_api_context_crop(node, outdir, max_side=API_IMAGE_MAX_SIDE):
    """Create API-only padded/boxed RGB from the saved full model-input frame.

    Original tight crops, graph assets and saliency measurements stay intact.
    Coordinates use Pillow's half-open crop convention; rectangle endpoints
    are converted to inclusive pixels after resizing.
    """
    outdir = Path(outdir)
    relative = node.get("frame_file")
    if not relative or not (outdir / relative).is_file():
        raise FileNotFoundError(f"Node {node.get('id')}: saved full-frame RGB is missing; "
                                "restore frames/ and frame.png or prepare a new report. A tight crop cannot supply context.")
    with Image.open(outdir / relative) as source:
        frame = source.convert("RGB")
    left, top, right, bottom = crop_box(node, frame.size)
    if not (0 <= left < right <= frame.width and 0 <= top < bottom <= frame.height):
        raise ValueError(f"Node {node.get('id')}: invalid original patch boundary.")
    px, py = API_CONTEXT_PADDING * (right-left), API_CONTEXT_PADDING * (bottom-top)
    context_box = (max(0, math.floor(left-px)), max(0, math.floor(top-py)),
                   min(frame.width, math.ceil(right+px)), min(frame.height, math.ceil(bottom+py)))
    crop = frame.crop(context_box)
    context_size = crop.size
    target_box = (left-context_box[0], top-context_box[1],
                  right-context_box[0], bottom-context_box[1])
    if max(crop.size) > max_side:
        scale = max_side / float(max(crop.size))
        size = tuple(max(1, round(value*scale)) for value in crop.size)
        crop = crop.resize(size, Image.Resampling.LANCZOS)
    sx, sy = crop.width/context_size[0], crop.height/context_size[1]
    x0, y0, x1, y1 = target_box
    # Use actual post-rounding scale, including independently rounded dimensions.
    x0 = min(crop.width-1, max(0, round(x0*sx)))
    y0 = min(crop.height-1, max(0, round(y0*sy)))
    x1 = min(crop.width-1, max(x0, round(x1*sx)-1))
    y1 = min(crop.height-1, max(y0, round(y1*sy)-1))
    box_width = max(1, min(API_CONTEXT_BOX_WIDTH, (x1-x0+1)//2, (y1-y0+1)//2))
    ImageDraw.Draw(crop).rectangle((x0, y0, x1, y1), outline="red", width=box_width)
    filename = Path("api_crops") / Path(node["crop_file"]).name
    (outdir / filename).parent.mkdir(parents=True, exist_ok=True)
    crop.save(outdir / filename)
    node["api_context"] = {"mode": API_CONTEXT_MODE, "file": filename.as_posix(),
        "padding_fraction_per_side": API_CONTEXT_PADDING, "frame_size": list(frame.size),
        "original_patch_box": [left, top, right, bottom], "context_crop_box": list(context_box),
        "context_size": list(context_size), "transmitted_size": list(crop.size),
        "red_box_inclusive": [x0, y0, x1, y1], "box_width": box_width}
    return outdir / filename


def save_match_assets(node, source_dir, outdir):
    relative = node.get("thumb_path")
    if not relative or not (source_dir / relative).is_file():
        raise FileNotFoundError(f"Node {node['id']}: cached model-input frame thumbnail is missing. Restore prototype_graph_outputs/frames; original RGB files will not be substituted silently.")
    frame_path = Path("frames") / f"frame_{node['frame_id']:09d}.jpg"
    if not (outdir / frame_path).exists():
        shutil.copyfile(source_dir / relative, outdir / frame_path)
    with Image.open(outdir / frame_path) as source:
        image = source.convert("RGB")
    crop_path = Path("crops") / f"node_{node['id']}.png"
    image.crop(crop_box(node, image.size)).save(outdir / crop_path)
    node.update(crop_file=crop_path.as_posix(), frame_file=frame_path.as_posix(),
                crop_source="cached model-input RGB thumbnail; uniform stage4 cell, not a receptive-field boundary")
    return node


def replay_frame(args, helper, graph, frame, manifest, outdir):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    evaluation = Path(args.evaluation or manifest["evaluation_file"]).resolve()
    checkpoint = Path(args.checkpoint or manifest["checkpoint"]).resolve()
    if helper.sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the graph's checkpoint. Regenerate the prototype graph for that checkpoint before comparing cached saliency labels.")
    if helper.sha256_file(evaluation) != manifest["evaluation_sha256"]:
        raise ValueError("evaluation.py differs from the graph's extraction version. Restore it, or regenerate graphs with the current code.")
    helper.SRC_DIR, helper.EVALUATION_FILE = evaluation.parent, evaluation
    ev = helper.load_evaluation()
    ev.train_cfg.set_seed(int(manifest["seed"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select an available GPU or pass --device cpu.")
    root = manifest["dataset_roots"][frame["dataset"]]
    length = int(manifest["window_len"])
    dataset = ev.DatasetLoader(root, window_len=length, stride=int(manifest["stride"]))
    index = int(frame["dataset_index"])
    video, start = dataset.windows[index]
    names = helper.frame_names_for_video(dataset, root, str(video))
    expected = json.loads(frame["window_names"])
    if str(video) != frame["video"] or int(start) != frame["window_start"] or names[int(start):int(start)+length] != expected:
        raise ValueError("Dataset window ordering differs from graph extraction; refusing a different temporal context.")
    batch = next(iter(DataLoader(Subset(dataset, [index]), batch_size=1, shuffle=False,
                                 num_workers=0, collate_fn=ev.video_saliency_collate_fn)))
    labels, rgb, sal, lengths, valid = helper.unpack_batch(batch)
    if labels != [frame["video"]] or int(lengths[0]) != length or not bool(valid.all()):
        raise ValueError("Replay did not return the exact full target window.")
    model = ev.build_model(device, device)
    ev.load_checkpoint(model, str(checkpoint)); model.eval()
    capture = helper.Stage4Capture(model)
    capture.reset()
    try:
        amp_dtype = getattr(torch, str(manifest.get("amp_dtype") or "torch.float16").split(".")[-1])
        with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype, enabled=bool(manifest.get("amp_enabled"))):
            result = model(rgb.to(device), return_details=True, return_concept_losses=False, return_decoder_diagnostics=False)
        if not isinstance(result, dict) or "saliency_map" not in result:
            raise ValueError("Model must expose saliency_map in return_details=True mode.")
        prediction = result["saliency_map"].detach().float().cpu()
        if prediction.ndim != 4 or tuple(prediction.shape[:2]) != (1, 1) or not bool(torch.isfinite(prediction).all()):
            raise ValueError("Expected a finite predicted saliency_map [1,1,H,W].")
        detail = result.get("concept_out", {}).get("stage4")
        if detail is None:
            raise ValueError("Replay must expose native concept_out['stage4'].")
        info = helper.extract_stage(detail, 1, "stage4")
        h, w = info["h"], info["w"]
        if info["m"] != graph.prototype_count or info["weights"] is None:
            raise ValueError("Replay must expose native activations for the same full prototype bank.")
        live_features = capture.last_frame(info, 1)[0].reshape(h*w, -1)
        records = graph.con.execute("SELECT id,row,col,grid_h,grid_w,pred_score FROM nodes WHERE frame_id=? AND stage='stage4' ORDER BY row,col", (frame["id"],)).fetchall()
        if len(records) != h*w or any((r["grid_h"], r["grid_w"], r["row"], r["col"]) != (h, w, i//w, i%w) for i, r in enumerate(records)):
            raise ValueError("Replay stage4 grid differs from the graph.")
        node_ids = [int(row["id"]) for row in records]
        descriptors = graph.descriptors(node_ids)
        cached = np.stack([descriptors[nid][0] for nid in node_ids])
        cached_rgb = np.stack([descriptors[nid][1] for nid in node_ids])
        live = live_features
        if live.shape != cached.shape or not np.allclose(live, cached, atol=args.replay_atol, rtol=.005):
            raise ValueError("Replay descriptors differ from cached graph descriptors. Check model source, checkpoint and preprocessing; do not mix graph versions.")
        cached_activations = graph.frame_activations(frame["id"], node_ids)
        live_activations = info["weights"][0].reshape(h*w, info["m"])
        if not np.allclose(live_activations, cached_activations, atol=args.replay_atol, rtol=.005):
            raise ValueError("Replay prototype activations differ from the graph cache. Regenerate the prototype graph or restore the original model configuration.")
        for i, nid in enumerate(node_ids):
            row, col = divmod(i, w)
            node = graph.node(nid)
            top = list(map(int, info["top"][0, row, col]))
            active = sorted(set(map(int, info["active"][0, row, col][info["mask"][0, row, col]])))
            if top != node["topk_ids"] or active != node["active_ids"]:
                raise ValueError(f"Replay assignments differ for patch {nid}; refusing mixed prototype-overlap metadata.")
        scores = F.adaptive_avg_pool2d(prediction, (h, w))[0, 0].numpy()
        cached_scores = np.array([r["pred_score"] for r in records]).reshape(h, w)
        if not np.allclose(scores, cached_scores, atol=args.replay_score_atol, rtol=.005):
            raise ValueError("Replayed saliency differs from cached saliency. Check model source/configuration and graph provenance.")
        percentiles = helper.midrank_percentiles(scores).reshape(h, w)
        if rgb.shape[2] == 3:
            last_rgb = rgb[0, -1].detach().float().cpu()
        elif rgb.shape[-1] == 3:
            last_rgb = rgb[0, -1].detach().float().cpu().permute(2, 0, 1)
        else:
            raise ValueError("Unsupported RGB layout.")
        if last_rgb.max() > 2:
            last_rgb = last_rgb / 255
        if not bool(torch.isfinite(last_rgb).all()) or last_rgb.min() < -.01 or last_rgb.max() > 1.01:
            raise ValueError("Expected unnormalized RGB in [0,1] or [0,255].")
        if tuple(last_rgb.shape[-2:]) != (frame["input_h"], frame["input_w"]):
            raise ValueError("Replay RGB dimensions differ from the graph.")
        rgb_hwc = last_rgb.permute(1, 2, 0).numpy()
        live_rgb = np.stack([helper.rgb_descriptor(rgb_hwc, r, c, h, w)
                             for r in range(h) for c in range(w)])
        if not np.allclose(live_rgb, cached_rgb, atol=1e-5, rtol=1e-5):
            raise ValueError("Replay RGB descriptors differ from prototype graph extraction preprocessing.")
        image = Image.fromarray((last_rgb.permute(1, 2, 0).numpy().clip(0, 1)*255).astype(np.uint8))
        image.save(outdir / "frame.png")
        np.save(outdir / "prediction.npy", prediction[0, 0].numpy(), allow_pickle=False)
        np.save(outdir / "patch_scores.npy", scores, allow_pickle=False)
        np.save(outdir / "patch_percentiles.npy", percentiles, allow_pickle=False)
        anchors = []
        for r in records:
            row, col = r["row"], r["col"]
            if percentiles[row, col] < args.saliency_percentile:
                continue
            node = graph.node(r["id"])
            node.update(replayed_pred_score=float(scores[row, col]), replayed_pred_percentile=float(percentiles[row, col]))
            crop = Path("crops") / f"anchor_{node['id']}.png"
            image.crop(crop_box(node, image.size)).save(outdir / crop)
            node.update(crop_file=crop.as_posix(), frame_file="frame.png", crop_source="replayed model-input RGB")
            anchors.append(node)
        anchors.sort(key=lambda n: (-n["replayed_pred_percentile"], -n["replayed_pred_score"], n["id"]))
        overview = image.copy(); draw = ImageDraw.Draw(overview)
        for node in anchors:
            box = crop_box(node, overview.size)
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0]+2, box[1]+2), str(node["id"]), fill="red", stroke_width=1, stroke_fill="white")
        overview.save(outdir / "salient_anchors.png")
        return anchors, {"checkpoint": str(checkpoint), "checkpoint_sha256": manifest["checkpoint_sha256"],
            "evaluation_file": str(evaluation), "evaluation_sha256": manifest["evaluation_sha256"],
            "window_names": expected, "window_len": length, "grid": [h, w], "device": str(device),
            "max_feature_difference": float(np.max(np.abs(live-cached))),
            "max_score_difference": float(np.max(np.abs(scores-cached_scores))),
            "max_rgb_descriptor_difference": float(np.max(np.abs(live_rgb-cached_rgb))),
            "max_activation_difference": float(np.max(np.abs(live_activations-cached_activations)))}
    finally:
        capture.close()
        # No returned object retains GPU tensors. Release saliency before API requests.
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def prompt_for(count):
    schema = {
        "Description of each patch": [
            {label: {"Low-level visual features": "...", "Semantic object or part": "..."}}
            for label in ["Anchor", *[f"Match {i}" for i in range(1, count + 1)]]],
        "Description of the common visual pattern": "A concise 1–3 sentence description.",
        "Visual pattern title": "A short title, or None if there is no majority pattern.",
        "Supporting patches": [],
        "Non-supporting patches": [],
    }
    return CONTEXT_INSTRUCTIONS + f"""Your task is to identify the visual pattern shared by most of the given patches. Observe each patch, and report the visual pattern present in STRICTLY MORE THAN HALF of the {count + 1} patches as the common visual pattern.

Each image contains two regions:

- **Target patch:** ONLY the content inside the red rectangle. This is the region you must describe and compare.
- **Surrounding context:** The content outside the rectangle. Use it only to help recognize what the patch depicts.

The first image is the anchor. Subsequent images are labeled Match 1, Match 2, and so on. Use these labels in your answer.

**Rules:**

1. Describe only what is visible INSIDE each red rectangle. Do not describe the entire crop.
2. Context may help identify a boxed object part, but your description must remain specific to that part. For example, if the box contains dark hair and a small area of skin, describe those features—not the person's face, clothing, or surroundings outside the box.
3. If an object extends outside the box, describe only its visible portion inside the box. Do not attribute its unseen features to the patch.
4. Identify shared colors, textures, edge orientations, shapes, contrasts, or semantic objects or parts. 
6. Ignore the red rectangles themselves. Do not count features outside the boxes as supporting evidence.
7. If there is no clear pattern shared by most target patches, state this explicitly.

Before answering, check that every described feature belongs inside a target box.

Return ONLY one valid JSON object with the following schema (replace the placeholders):
{json.dumps(schema, ensure_ascii=False, indent=2)}

Use arrays of exact image labels for Supporting patches and Non-supporting patches,
for example ["Anchor", "Match 2"]. Every label must appear in exactly one of these
two arrays. Include one description for every image, in the given image order.
If no majority pattern is found, set Visual pattern title to "None", explain why
in the common-pattern description, use an empty supporting array, and put all
labels in the non-supporting array. Never invent semantic identities; state
uncertainty when a visible object or part cannot be confidently identified."""


def new_anchor_query(anchor, matches, minimum):
    count = len(matches)
    return {"status": "skipped_empty" if not count else "skipped_small_group" if count < minimum else "pending",
            "anchor_id": anchor["id"], "node_ids": [n["id"] for n in matches],
            "includes_anchor": True, "matched_patch_count": count,
            "image_context_mode": API_CONTEXT_MODE,
            "image_order": [{"role": "anchor", "node_id": anchor["id"]}] +
                           [{"role": "match", "node_id": n["id"]} for n in matches],
            "prompt": prompt_for(count) if count >= minimum else None}


def wait_seconds(seconds):
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 30.0))


def retry_delay_seconds(headers, payload):
    """Honor both standard Retry-After and Google RPC RetryInfo delays."""
    delays = []
    if headers:
        value = headers.get("Retry-After")
        if value:
            try:
                delays.append(float(value))
            except ValueError:
                try:
                    delays.append(parsedate_to_datetime(value).timestamp()-time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    for detail in error.get("details", []):
        value = detail.get("retryDelay") if isinstance(detail, dict) else None
        if isinstance(value, str) and value.endswith("s"):
            try:
                delays.append(float(value[:-1]))
            except ValueError:
                pass
    return max([0.] + [n for n in delays if math.isfinite(n)])


def is_daily_or_zero_quota(payload):
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    text = json.dumps(error).lower().replace("_", "").replace("-", "").replace(" ", "")
    return ("perday" in text or "daily" in text or '"quotavalue":"0"' in text
            or '"quotavalue":0' in text or "limit:0" in text)


class GeminiDescriber:
    """One independent REST prompt per anchor, with pacing and bounded retries."""
    def __init__(self, args, outdir):
        self.args = args
        self.api_key = (GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError('Add your key to GEMINI_API_KEY = "" near the top of find_vis_pattern.py, or set the GEMINI_API_KEY environment variable. Prepared matches are saved and can be resumed.')
        self.model = args.gemini_model.removeprefix("models/")
        if not self.model or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in self.model):
            raise ValueError("--gemini-model must be a Gemini model ID, not a URL.")
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.state_path = Path(outdir) / "gemini_rate_state.json"
        self.last_attempt_at = None
        if self.state_path.exists():
            self.last_attempt_at = float(json.loads(self.state_path.read_text())["last_attempt_at"])
        self.metadata = {"provider": "gemini", "model_id": self.model, "api": "generateContent REST v1beta",
            "max_new_tokens": args.max_new_tokens, "min_request_interval_seconds": args.min_request_interval,
            "max_retries": args.max_retries, "max_retry_wait_seconds": args.max_retry_wait,
            "anchor_in_prompt": True, "saliency_labels_in_prompt": False,
            "image_input": "anchor context crop first, followed by every retrieved match context crop; original patch boxed in red",
            "image_context_mode": API_CONTEXT_MODE,
            "context_padding_fraction_per_side": API_CONTEXT_PADDING,
            "context_box_color": "red", "context_box_width_pixels": API_CONTEXT_BOX_WIDTH,
            "api_image_max_side": int(args.api_image_max_side),
            "api_jpeg_quality": int(args.api_jpeg_quality),
            "queries_per_anchor": 1, "independent_conversations": True}

    def _pace(self):
        if self.last_attempt_at is not None:
            delay = max(0., self.args.min_request_interval-(time.time()-self.last_attempt_at))
            if delay:
                print(f'Waiting {delay:.1f}s before the next Gemini request…', flush=True)
                wait_seconds(delay)
        self.last_attempt_at = time.time()
        dump_json(self.state_path, {"last_attempt_at": self.last_attempt_at, "model_id": self.model})

    def _send(self, body):
        request = Request(self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        with urlopen(request, timeout=self.args.request_timeout) as response:
            return json.loads(response.read())

    def _build_request_body(self, paths, prompt):
        """Prepare one logical request: anchor first, then every match, with compressed inline images."""
        parts = []
        image_stats = []
        original_parts = []
        for number, path in enumerate(paths):
            label = "ANCHOR PATCH:" if number == 0 else f"Match {number}:"
            mime_type, encoded, stats = prepare_api_image(
                path, max_side=self.args.api_image_max_side, jpeg_quality=self.args.api_jpeg_quality)
            parts.extend([{"text": label},
                {"inlineData": {"mimeType": mime_type,
                                "data": base64.b64encode(encoded).decode("ascii")}}])
            original_bytes = Path(path).read_bytes()
            original_parts.extend([{"text": label},
                {"inlineData": {"mimeType": "image/png",
                                "data": base64.b64encode(original_bytes).decode("ascii")}}])
            image_stats.append(stats)
        parts.append({"text": prompt})
        original_parts.append({"text": prompt})
        config = {"maxOutputTokens": self.args.max_new_tokens}
        if self.model.startswith("gemini-3"):
            config["thinkingConfig"] = {"thinkingLevel": "medium"}
        body = json.dumps({"contents": [{"role": "user", "parts": parts}], "generationConfig": config}).encode()
        original_body = json.dumps({"contents": [{"role": "user", "parts": original_parts}],
                                    "generationConfig": config}).encode()
        return body, original_body, image_stats

    def _log_payload_stats(self, paths, image_stats, body, original_body):
        original_total = len(original_body)
        transmitted_total = len(body)
        saved = original_total - transmitted_total
        reduction = (100.0 * saved / original_total) if original_total else 0.0
        print(f"Gemini image payload: {len(paths)} images in 1 request "
              f"(anchor first, then {len(paths)-1} matches).", flush=True)
        for index, stats in enumerate(image_stats):
            role = "anchor" if index == 0 else f"match {index}"
            ow, oh = stats["original_size"]
            tw, th = stats["transmitted_size"]
            print(f"  [{role}] {ow}x{oh} -> {tw}x{th} px; "
                  f"{stats['original_bytes']:,} -> {stats['transmitted_bytes']:,} bytes; "
                  f"encoding={stats['encoding']}"
                  f"{'; retained original PNG' if stats['retained_original'] else ''}"
                  f"{'; resized' if stats['resized'] else ''}", flush=True)
        if saved > 0:
            print(f"  Request JSON+Base64: {original_total:,} -> {transmitted_total:,} bytes "
                  f"({reduction:.1f}% smaller).", flush=True)
        elif saved < 0:
            print(f"  Request JSON+Base64: {original_total:,} -> {transmitted_total:,} bytes "
                  f"({abs(reduction):.1f}% larger; compression provided no saving).", flush=True)
        else:
            print(f"  Request JSON+Base64: {transmitted_total:,} bytes "
                  f"(0.0% change; compression provided no saving).", flush=True)

    def describe(self, paths, prompt):
        if len(paths)-1 < self.args.min_group_size:
            raise ValueError(f"At least {self.args.min_group_size} retrieved matches are required in addition to the anchor.")
        body, original_body, image_stats = self._build_request_body(paths, prompt)
        self._log_payload_stats(paths, image_stats, body, original_body)
        if len(body) > 18*1024*1024:
            raise ValueError("Inline image request exceeds this script's 18 MiB safety margin. Lower --top-k; no patches were silently dropped.")
        for attempt in range(self.args.max_retries+1):
            self._pace()
            try:
                payload = self._send(body)
                break
            except HTTPError as error:
                try:
                    payload = json.loads(error.read())
                except (ValueError, UnicodeDecodeError):
                    payload = {"error": {"message": f"HTTP {error.code}"}}
                message = str(payload.get("error", {}).get("message", f"HTTP {error.code}")).replace(self.api_key, "[REDACTED]")
                if error.code == 429 and is_daily_or_zero_quota(payload):
                    raise RuntimeError("Gemini daily or zero quota exhausted. Progress is saved; check AI Studio limits and resume later. " + message) from None
                if error.code not in (429, 500, 502, 503, 504) or attempt >= self.args.max_retries:
                    raise RuntimeError(f"Gemini HTTP {error.code}; progress is saved. {message}") from None
                delay = max(15.*2**attempt, retry_delay_seconds(error.headers, payload))
                if delay > self.args.max_retry_wait:
                    raise RuntimeError(f"Gemini requests a {delay:.1f}s retry delay, exceeding --max-retry-wait. Progress is saved; resume later.") from None
                print(f'Gemini HTTP {error.code}; retry {attempt+1}/{self.args.max_retries} after at least {delay:.1f}s…', flush=True)
                wait_seconds(delay)
            except (URLError, TimeoutError) as error:
                message = str(error).replace(self.api_key, "[REDACTED]")
                if attempt >= self.args.max_retries:
                    raise RuntimeError("Gemini network request failed; progress is saved. " + message) from None
                delay = min(self.args.max_retry_wait, 15. * 2 ** attempt)
                print(f'Gemini network/timeout error; retry {attempt+1}/{self.args.max_retries} after {delay:.1f}s… ({message})', flush=True)
                wait_seconds(delay)
        else:
            raise RuntimeError("Gemini network request failed after retries; progress is saved.")
        candidates = payload.get("candidates") or []
        if not candidates:
            reason = payload.get("promptFeedback", {}).get("blockReason", "no candidate")
            raise RuntimeError(f"Gemini returned no candidate ({reason}); progress is saved.")
        candidate = candidates[0]
        text = "\n".join(p["text"] for p in candidate.get("content", {}).get("parts", [])
                         if p.get("text") and not p.get("thought", False)).strip()
        if not text:
            raise RuntimeError(f"Gemini returned no answer text ({candidate.get('finishReason')}). Increase --max-new-tokens if needed and resume.")
        usage = payload.get("usageMetadata", {})
        return {"status": "complete", "text": text, "raw_response": payload,
            "prompt_mode": "gemini_single_anchor_all_matches", "response_model": payload.get("modelVersion", self.model),
            "input_tokens": usage.get("promptTokenCount"), "generated_tokens": usage.get("candidatesTokenCount"),
            "thinking_tokens": usage.get("thoughtsTokenCount"), "finish_reason": candidate.get("finishReason"),
            "api_attempts": attempt+1, "possibly_truncated": candidate.get("finishReason") == "MAX_TOKENS",
            "api_image_payload": {
                "image_count": len(paths),
                "images": image_stats,
                "original_request_bytes": len(original_body),
                "transmitted_request_bytes": len(body),
                "reduction_percent": (100.0 * (len(original_body) - len(body)) / len(original_body)) if original_body else 0.0,
            }}


def query_status_text(query, minimum):
    if query["status"] == "skipped_empty":
        return "No retrieved matches; anchor query skipped."
    if query["status"] == "skipped_small_group":
        return f"Only {len(query['node_ids'])} retrieved matches; at least {minimum} required in addition to the anchor. Query skipped."
    return query.get("text") or query.get("error") or "Pending Gemini inference."


PATTERN_FIELDS = (
    "Description of each patch", "Description of the common visual pattern",
    "Visual pattern title", "Supporting patches", "Non-supporting patches",
)


def extract_pattern_analysis(text, match_count):
    """Normalize JSON or the legacy JSON-like answer without executing model text.

    Match indices always refer to the original request, even after UI filtering.
    The anchor has index 0 and its own explicit label. Missing membership is
    unknown, never silently inferred from predicted saliency or a missing list.
    """
    labels = ["Anchor", *[f"Match {i}" for i in range(1, match_count + 1)]]
    decoder = json.JSONDecoder()
    source = text.strip() if isinstance(text, str) else ""
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source, flags=re.I)
        source = re.sub(r"\s*```$", "", source)
    fields, warnings = {}, []
    if isinstance(text, dict):
        fields = text
    else:
        try:
            value = json.loads(source)
            if isinstance(value, dict):
                fields = value
        except (ValueError, TypeError):
            pass
    if not fields:
        # Older answers omit outer braces/commas between top-level properties.
        # Decode each known property independently; no eval or code execution.
        keys = "|".join(map(re.escape, (*PATTERN_FIELDS, "Unsupporting patches")))
        headers = list(re.finditer(
            rf'''(?im)^\s*\**["']?({keys})["']?\**\s*:\s*''', source))
        for i, header in enumerate(headers):
            fragment = source[header.end():headers[i+1].start() if i+1 < len(headers) else len(source)].strip()
            try:
                fields[header[1]] = decoder.raw_decode(fragment)[0]
            except ValueError:
                # Plain prose remains readable; incomplete JSON stays explicit.
                if fragment and fragment[0] not in '[{"':
                    fields[header[1]] = fragment.rstrip(",\n ")
        if fields:
            warnings.append("Fields recovered from a legacy response format.")

    def label_of(value):
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            return None
        value = str(value).strip()
        if value.lower() == "anchor":
            return "Anchor"
        match = re.fullmatch(r"(?:match\s*)?(\d+)", value, flags=re.I)
        if match and 1 <= int(match[1]) <= match_count:
            return f"Match {int(match[1])}"
        return None

    def membership(value):
        if value is None:
            return None
        if isinstance(value, str):
            # Reasons after ':' can mention other labels; do not count those.
            prefix = value.split(":", 1)[0].strip()
            if prefix.lower().strip(". ") in ("none", "no patches", "n/a", "[]", ""):
                return []
            values = re.findall(r"\bAnchor\b|\bMatch\s*\d+\b", prefix, flags=re.I)
            if not values:
                warnings.append("A support list could not be interpreted.")
                return None
        elif isinstance(value, list):
            values = value
        else:
            return None
        result = []
        for item in values:
            label = label_of(item)
            if label is None:
                warnings.append("An unrecognized patch label was ignored.")
            elif label not in result:
                result.append(label)
        return [label for label in labels if label in result]

    supporting = membership(fields.get("Supporting patches"))
    nonsupporting = membership(fields.get("Non-supporting patches", fields.get("Unsupporting patches")))
    conflicts = set(supporting or []) & set(nonsupporting or [])
    if conflicts:
        warnings.append("Conflicting support labels are shown as unknown and excluded from counts.")
        supporting = [label for label in supporting if label not in conflicts]
        nonsupporting = [label for label in nonsupporting if label not in conflicts]

    descriptions = {}
    raw = fields.get("Description of each patch")
    groups = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    for group in groups:
        if isinstance(group, dict):
            for label, details in group.items():
                canonical = label_of(label)
                if canonical and isinstance(details, dict):
                    descriptions[canonical] = details
    # Also recover complete individual descriptions if an old list was truncated
    # or used List(...) instead of valid JSON brackets.
    if not descriptions:
        for header in re.finditer(r'''["'](Anchor|Match\s*\d+)["']\s*:\s*''', source, re.I):
            try:
                details = decoder.raw_decode(source[header.end():])[0]
            except ValueError:
                continue
            label = label_of(header[1])
            if label and isinstance(details, dict):
                descriptions[label] = details

    def as_text(value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    patches = []
    for index, label in enumerate(labels):
        details = descriptions.get(label, {})
        support = ("supporting" if label in (supporting or []) else
                   "non-supporting" if label in (nonsupporting or []) else "unknown")
        patches.append({"index": index, "label": label, "support": support,
            "low_level_visual_features": as_text(details.get("Low-level visual features")),
            "semantic_object_or_part": as_text(details.get("Semantic object or part"))})
    unknown = [p["label"] for p in patches if p["support"] == "unknown"]
    if fields and unknown:
        warnings.append("Support status was not available for: " + ", ".join(unknown) + ".")
    return {"schema_version": 1, "available": bool(fields or descriptions),
        "title": as_text(fields.get("Visual pattern title")),
        "description": as_text(fields.get("Description of the common visual pattern")),
        "supporting_patches": supporting, "supporting_count": None if supporting is None else len(supporting),
        "nonsupporting_patches": nonsupporting,
        "nonsupporting_count": None if nonsupporting is None else len(nonsupporting),
        "unknown_patches": unknown, "patches": patches, "warnings": list(dict.fromkeys(warnings))}


REPORT_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual pattern explorer</title>
<style>
:root{color-scheme:light;--ink:#172b3c;--muted:#657382;--line:#dce3e8;--paper:#fff;--bg:#f3f5f6;--accent:#146b67;--warm:#e3a144;--shadow:0 8px 28px #172b3c08}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,select{font:inherit;color:inherit}button{cursor:pointer}button:disabled{cursor:default;opacity:.4}button:focus-visible,select:focus-visible{outline:3px solid #359eac;outline-offset:3px}[hidden]{display:none!important}.page{max-width:1440px;margin:0 auto;padding:30px 32px 48px}.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:10px;font-weight:750;color:var(--accent)}header{display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:24px}h1{font-size:30px;letter-spacing:-.04em;line-height:1.2;margin:5px 0 8px;font-weight:650}h2{font-size:17px;letter-spacing:-.02em;margin:0}h3{font-size:14px;margin:0;font-weight:650}p{margin:0}.muted{color:var(--muted)}.subline{font-size:13px}.status-badge,.tag{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:20px;background:white;padding:6px 11px;font-size:11px;white-space:nowrap}.status-dot{width:6px;height:6px;border-radius:50%;background:var(--accent)}.top-meta{display:flex;flex-direction:column;align-items:flex-end;gap:9px}.paired{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);overflow:hidden}.card-head{padding:15px 18px;display:flex;align-items:center;justify-content:space-between;gap:10px}.card-title{font-size:13px;font-weight:650}.card-note{font-size:11px;color:var(--muted)}.viewport{position:relative;aspect-ratio:var(--frame-ratio,16/9);background:#111820;overflow:hidden}.viewport>img{display:block;width:100%;height:100%;object-fit:fill}.card-foot{padding:11px 18px;min-height:45px;display:flex;align-items:center;justify-content:space-between;gap:14px;font-size:11px;color:var(--muted)}.legend-key{display:inline-block;width:10px;height:10px;border:1.5px solid var(--warm);vertical-align:-1px;margin-right:6px}.scale{display:flex;align-items:center;gap:8px;font-size:10px}.scale-ramp{height:7px;width:92px;background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000,#7f0000);border:1px solid #c6cdd4;border-radius:2px}.patch-box{position:absolute;border:1.5px solid #ffbf58;background:#ffc5670b;padding:0;min-width:0;transition:background .15s,border .15s;z-index:1}.patch-box>span{position:absolute;top:0;left:0;background:#ffbf58;color:#202d35;font-size:9px;font-weight:800;line-height:1.4;padding:1px 4px;pointer-events:none}.patch-box:hover,.patch-box.is-hover{background:#ffd27335;border-color:white;z-index:3}.patch-box.is-active{border:2px solid #5bfff0;background:#5bfff027;z-index:2;box-shadow:0 0 0 1px #103e3b}.patch-box.is-active>span{background:#5bfff0}.map-empty{height:100%;display:grid;place-items:center;padding:30px;color:#b3bfcc;text-align:center}.anchors-section{margin-top:28px}.section-heading{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:13px}.section-heading p{font-size:12px;color:var(--muted);margin-top:3px}.count{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}.anchor-strip{display:flex;gap:12px;overflow-x:auto;padding:3px 3px 12px;scroll-snap-type:x proximity}.anchor-card{flex:0 0 139px;scroll-snap-align:start;background:white;border:1px solid var(--line);padding:9px;border-radius:10px;text-align:left;box-shadow:0 2px 6px #172b3c03;transition:background .15s,border .15s}.anchor-card:hover,.anchor-card.is-hover{border-color:#7da4a1;background:#f6fbfa}.anchor-card.is-active{border:2px solid var(--accent);padding:8px;background:#f0f8f6}.anchor-card img{display:block;width:100%;height:78px;object-fit:contain;background:#edf0f2;border-radius:5px;margin-bottom:8px}.anchor-title{display:flex;justify-content:space-between;gap:5px;align-items:center;font-size:12px;font-weight:700}.anchor-score{font-size:10px;color:var(--accent);font-weight:650}.anchor-card small{display:block;margin-top:3px;font-size:10px;color:var(--muted)}.selection-prompt{margin-top:12px;border:1px dashed #bdccd2;border-radius:10px;padding:24px;text-align:center;color:var(--muted);background:#ffffff66;font-size:13px}.detail-panel{scroll-margin-top:18px;margin-top:18px}.detail-header{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:18px 20px;border-bottom:1px solid var(--line);background:white}.detail-ident{display:flex;align-items:center;gap:13px;min-width:0}.detail-ident img{width:50px;height:50px;object-fit:contain;background:#eef1f3;border-radius:6px}.detail-ident p{font-size:11px;color:var(--muted);margin-top:3px}.controls{display:flex;align-items:center;gap:7px}.btn{border:1px solid var(--line);background:white;border-radius:7px;min-height:32px;padding:5px 10px;font-size:12px}.btn:hover:not(:disabled){background:#f3f7f7;border-color:#a5bcbb}.panel-body{display:grid;grid-template-columns:minmax(0,1fr) 340px;align-items:start}.matches-pane{padding:18px 20px;min-width:0}.pane-heading{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:15px}.pane-heading label{font-size:11px;color:var(--muted);display:flex;gap:7px;align-items:center}.pane-heading select{font-size:11px;padding:5px 8px;border:1px solid var(--line);background:white;border-radius:6px;max-width:140px}.match-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(156px,1fr));gap:13px}.match-card{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:white;min-width:0}.match-rank{display:flex;justify-content:space-between;gap:4px;padding:7px 8px;font-size:10px;font-weight:650}.match-label{font-size:8px;font-weight:600;color:#687583;background:#eef1f4;border-radius:4px;padding:1px 4px;align-self:center}.match-label.salient{color:#146b67;background:#e2f1ed}.match-crop{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#edf0f2}.context-frame{position:relative;margin:6px 6px 0;background:#111820}.context-frame>img{display:block;width:100%;height:auto}.context-box{position:absolute;border:1.5px solid #ffc264;box-sizing:border-box}.match-caption{padding:7px 8px;font-size:10px;line-height:1.6;color:var(--muted);overflow-wrap:anywhere}.match-caption strong{font-weight:600;color:#344b5c}.analysis-pane{padding:20px 22px;border-left:1px solid var(--line);background:#fafcfb;align-self:stretch;min-width:0}.analysis-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}.analysis-badge{font-size:9px;border-radius:5px;padding:3px 6px;background:#e4f2ec;color:#246854}.analysis-summary{font-size:11px;color:var(--muted);padding-bottom:16px;border-bottom:1px solid #e1e9e5;margin-bottom:16px}.analysis-copy{font-size:13px;line-height:1.8;overflow-wrap:anywhere}.analysis-copy p{margin:0 0 14px;white-space:pre-wrap}.analysis-copy h3{margin:18px 0 8px;font-size:14px}.analysis-copy strong{color:#173f3c;font-weight:650}.analysis-footnote{font-size:10px;line-height:1.6;color:#7c8984;margin-top:22px}.error-banner{padding:12px 16px;border:1px solid #edcfb2;border-radius:8px;color:#8d5329;background:#fff7ef;margin-bottom:18px;font-size:12px;overflow-wrap:anywhere}.footer-note{font-size:10px;line-height:1.6;color:var(--muted);margin-top:25px}.empty-matches{color:var(--muted);font-size:12px;padding:24px 0}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(min-width:1150px){.analysis-pane{position:sticky;top:0}}@media(max-width:1050px){.page{padding:22px}.panel-body{grid-template-columns:minmax(0,1fr) 300px}.match-grid{grid-template-columns:repeat(auto-fill,minmax(115px,1fr))}}@media(max-width:760px){.page{padding:18px 14px 30px}header{align-items:flex-start;gap:10px}h1{font-size:25px}.top-meta .card-note{display:none}.paired{grid-template-columns:1fr;gap:14px}.card-head{padding:12px 14px}.card-foot{padding:10px 14px}.section-heading{align-items:flex-start}.panel-body{grid-template-columns:1fr}.analysis-pane{border-left:0;border-top:1px solid var(--line)}.detail-header{padding:14px;flex-wrap:wrap}.detail-ident{flex:1}.controls{margin-left:auto}.matches-pane{padding:16px 14px}.anchor-card{flex-basis:126px}.anchor-card img{height:68px}.match-grid{grid-template-columns:repeat(auto-fill,minmax(112px,1fr))}.pane-heading{flex-wrap:wrap}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
.prototype-summary{padding:15px 20px;border-bottom:1px solid var(--line);display:grid;grid-template-columns:1fr 1fr;gap:10px 25px;background:#fcfdfd}.prototype-summary h3{font-size:11px;color:var(--muted);font-weight:600;margin-bottom:6px}.prototype-pills{display:flex;flex-wrap:wrap;gap:4px}.proto-pill{display:inline-flex;padding:2px 6px;border:1px solid #c5dcd7;background:#edf7f3;color:#255e52;border-radius:4px;font-size:10px;font-weight:650;line-height:1.5}.proto-pill.recurring{border-color:#e4d9bf;background:#faf6ec;color:#79613d}.prototype-coverage{grid-column:1/-1;font-size:10px;color:var(--muted)}.prototype-empty{font-size:10px;color:var(--muted)}.match-prototypes{padding:0 8px 10px}.prototype-row{margin-top:7px}.prototype-row>small{display:block;font-size:9px;color:var(--muted);margin-bottom:4px}.match-prototypes .proto-pill{font-size:10px;padding:1px 5px}@media(max-width:760px){.prototype-summary{padding:14px;grid-template-columns:1fr}.detail-ident{flex:1 1 100%;width:100%}.controls{width:100%;justify-content:flex-end}.analysis-pane{grid-row:1;border-top:0;border-bottom:1px solid var(--line)}}
.panel-body{grid-template-columns:minmax(0,1fr) 310px}
.analysis-pane{grid-column:2;grid-row:1;border-right:0;border-left:1px solid var(--line);padding:20px 18px}
.matches-pane{grid-column:1;grid-row:1}.match-grid{grid-template-columns:repeat(var(--patch-grid-cols,1),minmax(120px,1fr));align-items:start}
.match-grid-scroll{overflow-x:auto;padding-bottom:10px}
.analysis-summary{line-height:1.6}.pattern-box{border:1px solid var(--line);border-radius:9px;background:white;padding:13px 14px;margin-bottom:12px;overflow-wrap:anywhere}
.pattern-box h4{margin:0 0 8px;font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.pattern-box p{font-size:12px;line-height:1.7;white-space:pre-wrap;margin:0}.pattern-box.title-box p{font-size:18px;font-weight:650;line-height:1.4;color:#174942}
.pattern-total{font-size:23px;font-weight:700;line-height:1.2}.pattern-total small{font-size:11px;font-weight:400;color:var(--muted);margin-left:7px}
.patch-indices{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}.index-chip{font-size:10px;padding:2px 7px;border-radius:4px;background:#eef3f2;border:1px solid #d6e2df}
.pattern-box.supporting{border-top:3px solid #2a8170}.pattern-box.non-supporting{border-top:3px solid #bd8154}
.patch-analysis{padding:10px;border-top:1px solid var(--line)}.patch-analysis dl{margin:9px 0 0}.patch-analysis dt{font-size:10px;color:var(--muted);font-weight:650;margin-top:10px}.patch-analysis dd{margin:3px 0 0;font-size:11px;line-height:1.65;overflow-wrap:anywhere;white-space:pre-wrap}
.support-badge{display:inline-flex;font-size:10px;font-weight:650;line-height:1.4;border:1px solid #d4dce0;padding:4px 7px;border-radius:5px;background:#f2f4f6;color:#52616d}
.support-badge.supporting{color:#1c6656;background:#e9f6ef;border-color:#bddbcb}.support-badge.non-supporting{color:#85552f;background:#fff3e8;border-color:#edd3bb}
.anchor-evidence{border-color:#7caea4}.anchor-evidence .match-rank{background:#edf7f3}.analysis-raw{margin-top:15px;font-size:11px;color:var(--muted)}.analysis-raw summary{cursor:pointer}.analysis-raw pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.6;font:11px/1.6 inherit}
.analysis-notice{font-size:11px;color:var(--muted);line-height:1.6;margin-top:10px}.analysis-notice.warning{color:#8d5329}
@media(max-width:1050px){.panel-body{grid-template-columns:minmax(0,1fr) 280px}}
@media(max-width:760px){.panel-body{grid-template-columns:1fr}.analysis-pane{grid-column:1;grid-row:2;border-left:0;border-right:0;border-top:1px solid var(--line);border-bottom:0;position:static}.matches-pane{grid-column:1;grid-row:1}}
</style>
</head>
<body><main class="page">
<header><div><p class="eyebrow">Explainable video saliency</p><h1>Visual pattern explorer</h1><p class="subline muted" id="frame-meta"></p></div><div class="top-meta"><span class="status-badge"><span class="status-dot"></span><span id="report-status"></span></span><span class="card-note" id="model-name"></span></div></header>
<p id="error-banner" class="error-banner" hidden></p>
<section class="paired" aria-label="Selected frame and prediction">
<article class="card"><div class="card-head"><span class="card-title">Selected frame</span><span class="card-note" id="frame-caption"></span></div><div class="viewport" id="frame-view"><img id="selected-frame" alt="Selected input frame with clickable salient patch boxes"></div><div class="card-foot"><span><i class="legend-key"></i>Salient patch locations</span><span>Click a box to inspect</span></div></article>
<article class="card"><div class="card-head"><span class="card-title">Predicted saliency overlay</span><span class="card-note" id="map-caption"></span></div><div class="viewport" id="map-view"><img id="saliency-map" alt="Predicted saliency as a JET heatmap overlaid on the original frame"><div id="map-empty" class="map-empty" hidden></div></div><div class="card-foot"><span id="map-range-note">Blue = lower predicted saliency; red = higher predicted saliency. Colors are normalized within each frame.</span><span class="scale">Low<span class="scale-ramp"></span>High</span></div></article>
</section>
<section class="anchors-section" aria-labelledby="anchors-heading"><div class="section-heading"><div><h2 id="anchors-heading">Salient patches</h2><p>Choose a patch to explore its visual matches and Gemini analysis.</p></div><span class="count" id="anchor-count"></span></div><div class="anchor-strip" id="anchor-strip" aria-label="Select a salient anchor patch"></div></section>
<div class="selection-prompt" id="selection-prompt">Select a salient patch above to open its detail panel.</div>
<section class="detail-panel card" id="detail-panel" aria-labelledby="detail-title" hidden>
<div class="detail-header"><div class="detail-ident"><img id="detail-crop" alt="Selected salient anchor"><div><h2 id="detail-title" tabindex="-1"></h2><p id="detail-meta"></p></div></div><div class="controls"><button class="btn" id="previous" aria-label="Previous salient patch">← Previous</button><button class="btn" id="next" aria-label="Next salient patch">Next →</button><button class="btn" id="close-panel" aria-label="Close patch detail panel">Close ×</button></div></div>
<div class="prototype-summary"><div><h3>Anchor top-k prototypes</h3><div id="anchor-prototypes" class="prototype-pills"></div></div><div><h3>Common to every matched patch</h3><div id="common-prototypes" class="prototype-pills"></div></div><p class="prototype-coverage" id="prototype-coverage"></p></div>
<div class="panel-body"><div class="matches-pane"><div class="pane-heading"><div><h3>Anchor and similar patches</h3><p class="card-note" id="matches-summary" aria-live="polite"></p></div><label>Show<select id="match-filter"><option value="all">All matches</option><option value="salient">Salient matches</option><option value="non-salient">Non-salient matches</option></select></label></div><div class="match-grid-scroll"><div class="match-grid" id="match-grid"></div></div><p class="empty-matches" id="empty-matches" hidden></p></div><aside class="analysis-pane" aria-label="Gemini pattern analysis"><div class="analysis-heading"><h3>Gemini analysis</h3><span class="analysis-badge" id="analysis-status"></span></div><p class="analysis-summary" id="analysis-summary"></p><div class="analysis-copy" id="analysis-output"></div><p class="analysis-footnote">This is a description of shared appearance. It does not establish why the model assigned saliency.</p></aside></div>
</section>
<p class="footer-note">Patch boxes locate cells on the stage-4 grid. Their visual tokens may encode a wider spatial and temporal context. The predicted overlay is normalized only for display; selection uses the saved model outputs.</p>
<p id="selection-announcement" class="sr-only" aria-live="polite"></p>
<noscript><p>Enable JavaScript to select anchors and explore the saved matches. The Markdown report contains the same analysis.</p></noscript>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
'use strict';
const REPORT=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id),anchors=REPORT.anchors;
$('match-grid').style.setProperty('--patch-grid-cols',String(REPORT.patch_grid_cols||(REPORT.top_k+1)));
let selected=-1,returnFocus=null;
const make=(tag,className,text)=>{const element=document.createElement(tag);if(className)element.className=className;if(text!=null)element.textContent=text;return element;};
const image=(src,alt,className)=>{const element=make('img',className);element.src=src;element.alt=alt;return element;};
const percent=value=>`${(100*value).toFixed(1)}%`;
function geometry(element,node){element.style.left=`${100*node.col/node.grid_w}%`;element.style.top=`${100*node.row/node.grid_h}%`;element.style.width=`${100/node.grid_w}%`;element.style.height=`${100/node.grid_h}%`;}
function statusName(status){return({complete:'Ready',prepared:'Prepared',describing:'In progress',incomplete:'Incomplete',pending:'Pending',failed:'Request failed',skipped_empty:'No matches',skipped_small_group:'Skipped'})[status]||status;}
$('frame-meta').textContent=`Video ${REPORT.video} · ${REPORT.frame_name} · frame ${REPORT.frame_number}`;
$('frame-caption').textContent=`Video ${REPORT.video} · Frame ${REPORT.frame_number}`;
$('report-status').textContent=statusName(REPORT.status);
$('model-name').textContent=REPORT.model;
$('selected-frame').src=REPORT.frame_file;
for(const id of ['frame-view','map-view'])$(id).style.setProperty('--frame-ratio',`${REPORT.width}/${REPORT.height}`);
if(REPORT.map.available){$('saliency-map').src=REPORT.map.image;$('map-caption').textContent=`Video ${REPORT.video} · Frame ${REPORT.frame_number}`;$('map-range-note').textContent='Blue = lower predicted saliency; red = higher predicted saliency. Colors are normalized within each frame.';$('map-range-note').title=`Display range: ${REPORT.map.minimum} to ${REPORT.map.maximum}. Overlay opacity ${REPORT.map.alpha!=null?REPORT.map.alpha:0.5}. Original prediction.npy is unchanged.`;}
else{$('saliency-map').hidden=true;$('map-empty').hidden=false;$('map-empty').textContent=REPORT.map.reason||'Predicted saliency overlay is unavailable for this run.';$('map-caption').textContent='Unavailable';}
if(REPORT.error){$('error-banner').hidden=false;$('error-banner').textContent=REPORT.error;}
$('anchor-count').textContent=`${anchors.length} ${anchors.length===1?'patch':'patches'} · ≥ ${percent(REPORT.cutoff)}`;
function hoverAnchor(index,on){document.querySelectorAll(`[data-anchor-index="${index}"]`).forEach(element=>element.classList.toggle('is-hover',on));}
anchors.forEach((entry,index)=>{
 const anchor=entry.anchor,number=String(index+1).padStart(2,'0');
 const box=make('button','patch-box');box.type='button';box.dataset.anchorIndex=index;box.setAttribute('aria-label',`Select patch ${index+1}, node ${anchor.id}, predicted percentile ${percent(anchor.percentile)}`);box.setAttribute('aria-pressed','false');box.append(make('span','',number));geometry(box,anchor);$('frame-view').append(box);
 const card=make('button','anchor-card');card.type='button';card.dataset.anchorIndex=index;card.setAttribute('aria-label',`Select patch ${index+1}, node ${anchor.id}`);card.setAttribute('aria-pressed','false');card.append(image(anchor.crop_file,`Salient patch ${index+1}`));const title=make('span','anchor-title');title.append(make('span','',`Patch ${number}`),make('span','anchor-score',percent(anchor.percentile)));card.append(title,make('small','',`Video ${anchor.video} · Frame ${anchor.frame_number}`),make('small','',`Cell ${anchor.row}, ${anchor.col} · ${entry.matches.length} matches`));$('anchor-strip').append(card);
 for(const element of [box,card]){element.onclick=()=>{returnFocus=element;selectAnchor(index,true);};element.onmouseenter=()=>hoverAnchor(index,true);element.onmouseleave=()=>hoverAnchor(index,false);}
});
if(!anchors.length){$('selection-prompt').textContent='No patches meet the saliency cutoff for this frame.';}
function appendRichText(target,text){
 let start=0;for(const match of text.matchAll(/\*\*([^*\n]+)\*\*/g)){target.append(document.createTextNode(text.slice(start,match.index)),make('strong','',match[1]));start=match.index+match[0].length;}target.append(document.createTextNode(text.slice(start)));
}
function renderAnalysis(entry){
 const data=entry.pattern_analysis,output=$('analysis-output');output.replaceChildren();
 function box(title,text,className=''){
  const item=make('section',`pattern-box ${className}`);item.append(make('h4','',title),make('p','',text||'Not available in the saved response.'));output.append(item);return item;
 }
 box('Visual pattern title',data.title,'title-box');
 box('Visual pattern description',data.description);
 for(const [title,labels,count,kind] of [
  ['Supporting patches',data.supporting_patches,data.supporting_count,'supporting'],
  ['Unsupporting patches',data.nonsupporting_patches,data.nonsupporting_count,'non-supporting']]){
  const item=make('section',`pattern-box ${kind}`);item.append(make('h4','',title));
  const total=make('div','pattern-total',count==null?'—':String(count));total.append(make('small','',count==null?'count unavailable':`of ${entry.matches.length+1} patches`));item.append(total);
  const indices=make('div','patch-indices');
  if(labels?.length){for(const label of labels)indices.append(make('span','index-chip',label));}
  else indices.append(make('span','muted',labels==null?'Indices unavailable':'None'));
  item.append(indices);output.append(item);
 }
 if(!data.available)output.append(make('p','analysis-notice','Structured fields are not available yet. The saved response or request status is shown below.'));
 if(data.unknown_patches.length&&data.available)output.append(make('p','analysis-notice',`Unclassified: ${data.unknown_patches.join(', ')}`));
 for(const warning of data.warnings)output.append(make('p','analysis-notice',warning));
 if(entry.truncated)output.append(make('p','analysis-notice warning','This response reached the output-token limit. Some fields may be missing.'));
 const raw=make('details','analysis-raw');raw.append(make('summary','','Saved response / request status'),make('pre','',entry.analysis_text));output.append(raw);
}
function patchAnalysis(card,description){
 const area=make('div','patch-analysis');
 const status=description.support;
 const text=({supporting:'Supporting','non-supporting':'Non-supporting',unknown:'Support not reported'})[status];
 area.append(make('span',`support-badge ${status}`,text));
 const list=make('dl');
 for(const [title,value] of [['Low-level visual features',description.low_level_visual_features],['Semantic object or part',description.semantic_object_or_part]]){
  list.append(make('dt','',title),make('dd','',value||'Not provided in the saved response.'));
 }
 area.append(list);card.append(area);
}
function prototypePills(target,values,recurring=false){
 target.replaceChildren();if(values==null){target.append(make('span','prototype-empty','Unavailable'));return;}if(!values.length){target.append(make('span','prototype-empty','None'));return;}
 for(const value of values){const pill=make('span',`proto-pill${recurring?' recurring':''}`,recurring?`${value.id} · ${value.count}/${anchors[selected].matches.length}`:String(value));if(recurring)pill.title=`Prototype ${value.id} is observed in ${value.count} of ${anchors[selected].matches.length} retrieved matches`;target.append(pill);}
}
function frameIdentity(node){return `Video ${node.video} · Frame ${node.frame_number==null?node.target_name:node.frame_number}`;}
function renderMatches(){
 const entry=anchors[selected];if(!entry)return;const filter=$('match-filter').value;
 const list=entry.matches.map((node,index)=>({node,index})).filter(({node})=>filter==='all'||node.label===filter);
 $('matches-summary').textContent=`Anchor + ${list.length} of ${entry.matches.length} matches · original match indices retained`;
 $('match-grid').replaceChildren();$('empty-matches').hidden=!!list.length;
 if(!list.length)$('empty-matches').textContent=entry.matches.length?'No matches have this saliency label.':'No visual matches passed the retrieval thresholds.';
 const nodes=[{node:entry.anchor,index:-1},...list];
 for(const {node,index} of nodes){
  const isAnchor=index===-1,label=isAnchor?'Anchor':`Match ${index+1}`;
  const card=make('article',`match-card${isAnchor?' anchor-evidence':''}`),rank=make('div','match-rank');
  card.dataset.patchLabel=label;
  rank.append(make('span','',label),make('span',`match-label ${isAnchor||node.label==='salient'?'salient':''}`,isAnchor?'salient anchor':node.label));
  card.append(rank,image(node.crop_file,`${label}, node ${node.id}`,'match-crop'));
  const frameSource=isAnchor?REPORT.frame_file:node.frame_file;
  if(frameSource){const frame=make('div','context-frame'),img=image(frameSource,`Full frame for ${label}`);img.loading='lazy';const marker=make('span','context-box');geometry(marker,node);frame.append(img,marker);card.append(frame);}
  const caption=make('div','match-caption');
  if(!isAnchor)caption.append(make('strong','',`Cosine ${node.stage4_cosine.toFixed(3)}`));
  caption.append(make('div','',frameIdentity(node)),make('div','',node.target_name));
  card.append(caption);
  patchAnalysis(card,entry.pattern_analysis.patches[index+1]);
  if(!isAnchor){const overlaps=make('div','match-prototypes'),row=make('div','prototype-row'),pills=make('div','prototype-pills');row.append(make('small','','Shared with anchor'));prototypePills(pills,node.shared_with_anchor);row.append(pills);overlaps.append(row);card.append(overlaps);}
  $('match-grid').append(card);
 }
}
function selectAnchor(index,scroll){
 if(index<0||index>=anchors.length)return;selected=index;const entry=anchors[index],anchor=entry.anchor;
 document.querySelectorAll('[data-anchor-index]').forEach(element=>{const active=Number(element.dataset.anchorIndex)===index;element.classList.toggle('is-active',active);element.setAttribute('aria-pressed',String(active));});
 $('selection-prompt').hidden=true;$('detail-panel').hidden=false;$('detail-crop').src=anchor.crop_file;
 $('detail-title').textContent=`Patch ${String(index+1).padStart(2,'0')} · node ${anchor.id}`;
 $('detail-meta').textContent=`${frameIdentity(anchor)} (${anchor.target_name}) · cell ${anchor.row}, ${anchor.col} · percentile ${percent(anchor.percentile)}`;
 $('previous').disabled=index===0;$('next').disabled=index===anchors.length-1;
 prototypePills($('anchor-prototypes'),entry.prototypes.anchor_ids);prototypePills($('common-prototypes'),entry.prototypes.common_to_all);
 $('prototype-coverage').textContent=`Stage-4 top-k candidate overlap across all ${entry.matches.length} retrieved matches. Metadata available for ${entry.prototypes.matches_with_metadata}/${entry.matches.length} matches. Candidate overlap does not establish forward-pass participation.`;
 $('match-filter').value='all';renderMatches();
 $('analysis-status').textContent=statusName(entry.query_status);$('analysis-summary').textContent=`Anchor + ${entry.matches.length} matches. Counts include the anchor; support refers to the common visual pattern, independently of saliency.`;
 renderAnalysis(entry);
 $('selection-announcement').textContent=`Patch ${index+1} selected. Showing ${entry.matches.length} similar patches and its saved Gemini analysis.`;
 if(scroll){const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;$('detail-panel').scrollIntoView({behavior:reduced?'auto':'smooth',block:'start'});}
}
function closePanel(){selected=-1;$('detail-panel').hidden=true;$('selection-prompt').hidden=false;document.querySelectorAll('[data-anchor-index]').forEach(element=>{element.classList.remove('is-active');element.setAttribute('aria-pressed','false');});if(returnFocus?.isConnected)returnFocus.focus({preventScroll:true});}
$('previous').onclick=()=>selectAnchor(selected-1,false);$('next').onclick=()=>selectAnchor(selected+1,false);$('close-panel').onclick=closePanel;$('match-filter').onchange=renderMatches;
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&!$('detail-panel').hidden){event.preventDefault();closePanel();}});
</script></body></html>
'''


def saliency_display(report, outdir, alpha=HEATMAP_ALPHA):
    """Write a JET heatmap overlay PNG without changing predictions or selection."""
    source = outdir / "prediction.npy"
    frame_path = outdir / "frame.png"
    output = outdir / "predicted_saliency.png"
    previous = report.get("prediction_display", {})
    alpha = float(alpha)
    if not source.is_file():
        if (previous.get("available") and previous.get("format") == HEATMAP_DISPLAY_FORMAT
                and previous.get("alpha") == alpha and output.is_file()):
            return previous
        return {"available": False, "reason": "No saved prediction.npy was found for this run."}
    if not frame_path.is_file():
        return {"available": False, "reason": "No saved frame.png was found for overlay display."}
    modified = source.stat().st_mtime_ns
    frame_modified = frame_path.stat().st_mtime_ns
    if (previous.get("available") and previous.get("source_mtime_ns") == modified
            and previous.get("frame_mtime_ns") == frame_modified
            and previous.get("format") == HEATMAP_DISPLAY_FORMAT
            and previous.get("alpha") == alpha and output.is_file()):
        return previous
    try:
        values = np.asarray(np.load(source, allow_pickle=False), dtype=np.float64)
        if values.ndim != 2 or not values.size:
            return {"available": False, "reason": "The saved prediction must be a non-empty two-dimensional map."}
        if not np.isfinite(values).any():
            return {"available": False, "reason": "The saved prediction contains no finite values."}
        frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        overlay, meta = saliency_heatmap_overlay(frame_rgb, values, alpha=alpha)
        Image.fromarray(overlay, mode="RGB").save(output)
    except (OSError, ValueError) as error:
        return {"available": False, "reason": f"Could not build saliency overlay: {error}"}
    return {"available": True, "image": output.name, "minimum": meta["minimum"], "maximum": meta["maximum"],
            "constant": meta["constant"], "alpha": meta["alpha"], "colormap": meta["colormap"],
            "format": meta["format"], "source_mtime_ns": modified, "frame_mtime_ns": frame_modified,
            "prediction_shape": meta["prediction_shape"], "frame_shape": meta["frame_shape"],
            "scaling": meta["scaling"]}


def prototype_overlap_data(anchor, matches):
    """Use the same top-k candidate IDs as prototype-overlap graph metadata."""
    def ids(node):
        value = node.get("topk_ids")
        return None if value is None else sorted(set(map(int, value)))
    anchor_ids = ids(anchor)
    match_ids = [ids(node) for node in matches]
    counts = Counter(p for group in match_ids if group is not None for p in group)
    known = sum(group is not None for group in match_ids)
    common = (sorted(p for p, count in counts.items() if count == len(matches))
              if known == len(matches) else None)
    per_match = []
    for node, group in zip(matches, match_ids):
        overlap = None if group is None or anchor_ids is None else sorted(set(group) & set(anchor_ids))
        recurring = None if group is None else [{"id": p, "count": counts[p]} for p in group if counts[p] >= 2]
        per_match.append({"node_id": node["id"], "topk_ids": group,
                          "with_anchor": overlap, "across_matches": recurring})
    return {"definition": "stage4 top-k prototype candidate IDs; not forward-valid participation",
            "anchor_ids": anchor_ids, "match_count": len(matches), "matches_with_metadata": known,
            "common_to_all": common,
            "recurring": [{"id": p, "count": count} for p, count in sorted(counts.items()) if count >= 2],
            "per_match": per_match}


def report_view_payload(report):
    meta, frame = report["protocol"], report["frame"]
    cutoff = meta["saliency_percentile"]
    entries = []
    for entry in report["anchors"]:
        anchor, query = entry["anchor"], entry["query"]
        anchor_data = {key: anchor[key] for key in ("id", "row", "col", "grid_h", "grid_w", "crop_file")}
        anchor_data["percentile"] = anchor.get("replayed_pred_percentile", anchor.get("pred_percentile", 0.))
        anchor_data.update(video=frame["video"], frame_number=frame["target_index"]+1,
                           target_name=frame["target_name"])
        prototypes = entry.get("prototype_overlap") or prototype_overlap_data(anchor, entry["matches"])
        per_match = {p["node_id"]: p for p in prototypes["per_match"]}
        matches = []
        for node in entry["matches"]:
            item = {key: node[key] for key in ("id", "row", "col", "grid_h", "grid_w", "crop_file", "video", "target_name", "stage4_cosine", "rgb_mae", "rgb_structure_cosine")}
            item.update(frame_file=node.get("frame_file"), percentile=node["pred_percentile"],
                        frame_number=node["target_index"]+1 if node.get("target_index") is not None else None,
                        label="salient" if node["pred_percentile"] >= cutoff else "non-salient",
                        shared_with_anchor=per_match[node["id"]]["with_anchor"],
                        recurring_with_matches=per_match[node["id"]]["across_matches"])
            matches.append(item)
        entries.append({"anchor": anchor_data, "matches": matches,
            "matched_video_count": entry["matched_video_count"],
            "query_status": query["status"], "truncated": bool(query.get("possibly_truncated")),
            "pattern_analysis": extract_pattern_analysis(query.get("text", ""), len(matches)),
            "analysis_text": query_status_text(query, meta.get("min_group_size", MIN_GROUP_SIZE))})
        entries[-1]["prototypes"] = {key: value for key, value in prototypes.items() if key != "per_match"}
    return {"video": frame["video"], "frame_name": frame["target_name"], "frame_number": frame["target_index"]+1,
            "width": frame.get("input_w", 640), "height": frame.get("input_h", 360),
            "frame_file": "frame.png", "cutoff": cutoff, "top_k": meta["top_k"],
            "status": report["status"], "error": report.get("error"), "map": report["prediction_display"],
            "model": report.get("llm", {}).get("model_id") or meta.get("llm_model") or GEMINI_MODEL,
            "patch_grid_cols": int(meta.get("patch_grid_cols") or (meta["top_k"] + 1)),
            "anchors": entries}


def write_reports(report, outdir, heatmap_alpha=None):
    outdir = Path(outdir)
    if heatmap_alpha is None:
        heatmap_alpha = report.get("protocol", {}).get("heatmap_alpha", HEATMAP_ALPHA)
    heatmap_alpha = float(heatmap_alpha)
    report.setdefault("protocol", {})["heatmap_alpha"] = heatmap_alpha
    report["prediction_display"] = saliency_display(report, outdir, alpha=heatmap_alpha)
    for entry in report["anchors"]:
        entry["prototype_overlap"] = prototype_overlap_data(entry["anchor"], entry["matches"])
        entry["pattern_analysis"] = extract_pattern_analysis(
            entry["query"].get("text", ""), len(entry["matches"]))
    dump_json(outdir / "report.json", report)
    payload = json.dumps(report_view_payload(report), allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    (outdir / "report.html").write_text(REPORT_HTML.replace("__REPORT_DATA__", payload), encoding="utf-8")
    frame, meta = report["frame"], report["protocol"]
    md = ['# Visual pattern report', '',
          f"Video {frame['video']} · {frame['target_name']} · frame position {frame['target_index']+1} · {report['status']}", '',
          '![Selected frame and salient anchors](salient_anchors.png)', '']
    if report["prediction_display"].get("available"):
        md += ['![Predicted saliency overlay on the original frame](predicted_saliency.png)', '',
               'Blue = lower predicted saliency; red = higher predicted saliency. Colors are normalized within each frame.', '']
    if report.get("error"):
        md += [report["error"], '']
    for entry in report["anchors"]:
        anchor, query = entry["anchor"], entry["query"]
        md += [f"## Anchor {anchor['id']}: cell ({anchor['row']}, {anchor['col']})", '',
               f"Video {frame['video']} · frame {frame['target_index']+1} ({frame['target_name']})", '',
               f"![Anchor]({anchor['crop_file']})", '',
               query_status_text(query, meta.get("min_group_size", MIN_GROUP_SIZE)), '',
               f"Anchor top-k prototypes: {entry['prototype_overlap']['anchor_ids']}", '',
               f"Prototypes common to every retrieved match: {entry['prototype_overlap']['common_to_all']}", '',
               '| Match number | Node | Video / frame | Saliency label | Percentile | Stage-4 cosine |',
               '| --- | --- | --- | --- | --- | --- |']
        for number, node in enumerate(entry["matches"], 1):
            label = "salient" if node["pred_percentile"] >= meta["saliency_percentile"] else "non-salient"
            identity = (str(node["video"])+" / "+str(node["target_name"])).replace('|', '\\|')
            md.append(f"| {number} | {node['id']} | {identity} | {label} | {node['pred_percentile']:.3f} | {node['stage4_cosine']:.4f} |")
        md.append('')
        for overlap in entry["prototype_overlap"]["per_match"]:
            md += [f"Node {overlap['node_id']} — shared with anchor: {overlap['with_anchor']}; recurring across matches (ID/count): {overlap['across_matches']}", '']
    (outdir / "report.md").write_text('\n'.join(md), encoding="utf-8")


def prepare(args):
    helper = load_module(args.prototype_script, "_pattern_prototype_graph")
    source_dir = Path(args.prototype_graph_dir or helper.OUTPUT_DIR).resolve()
    frame_label = args.frame_name or f"frame_{args.frame_no}"
    outdir = Path(args.output or SCRIPT_DIR / "pattern_report_outputs" /
        (args.video_id + '_' + frame_label)).resolve()
    if outdir.exists():
        raise FileExistsError(f"Output already exists: {outdir}. Use --describe-only to finish an existing report, or choose a new --output.")
    with PrototypeGroupRetriever(helper, source_dir) as graph:
        manifest = graph.manifest
        frame = select_frame(graph.con, args.video_id, args.dataset, args.frame_name, args.frame_no)
        provider, llm_model = llm_selection(args)
        minimum = getattr(args, "min_group_size", MIN_GROUP_SIZE)
        print(f"Prototype graph: {source_dir}; N={graph.group_size} connected patches per prototype. "
              f"Each anchor compares its top-{min(ANCHOR_PROTOTYPE_COUNT, graph.prototype_count)} activated prototype groups.", flush=True)
        if any(getattr(args, name, None) is not None for name in ("visual_graph_dir", "min_stage4_cosine")):
            print("Prototype-group retrieval does not use visual-graph files or patch-edge thresholds.", flush=True)
        if args.top_k != TOP_K:
            print("--top-k does not truncate prototype groups; all winning-group patches are retained. Use --patch-grid-cols for row size.", flush=True)
        outdir.mkdir(parents=True)
        (outdir / "crops").mkdir(); (outdir / "frames").mkdir()
        print(f'Replaying {frame["video"]}/{frame["target_name"]} with the original temporal context…', flush=True)
        anchors, replay = replay_frame(args, helper, graph, frame, manifest, outdir)
        # GPU references from replay_frame have now left scope.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        report = {"schema_version": 2, "status": "prepared", "created_at": datetime.now(timezone.utc).isoformat(),
            "frame": frame, "replay": replay, "protocol": {"stage": "stage4", "saliency_percentile": args.saliency_percentile,
            "percentile_definition": "ascending midrank of mean raw predicted saliency_map per stage4 grid cell within its own frame; ties preserved",
            "retrieval_mode": PROTOTYPE_RETRIEVAL_MODE,
            "anchor_prototype_count": ANCHOR_PROTOTYPE_COUNT,
            "prototype_selection": "highest mean anchor-to-member visual similarity among top-8 activated prototype groups",
            "visual_similarity_formula": helper.SIMILARITY_FORMULA,
            "graph_group_size": graph.group_size, "graph_group_size_source": graph.group_size_source,
            "top_k": graph.group_size, "legacy_requested_top_k": args.top_k,
            "top_k_before_saliency_split": True, "thresholds": None,
            "cached_graph_thresholds": None, "min_group_size": minimum,
            "heatmap_alpha": float(getattr(args, "heatmap_alpha", HEATMAP_ALPHA)),
            "patch_grid_cols": int(getattr(args, "patch_grid_cols", PATCH_GRID_COLS) or (graph.group_size + 1)),
            "heatmap_display": HEATMAP_DISPLAY_FORMAT,
            "llm_provider": provider, "llm_model": llm_model,
            "other_videos_only": False, "max_matches_per_video": None,
            "video_identity": "(dataset, video)", "top_k_after_video_deduplication": False,
            "ranking": "select prototype by maximum mean visual similarity; return its complete connected group in graph activation-rank order",
            "source_graph": str(source_dir), "prototype_graph": str(source_dir),
            "source_manifest_sha256": graph.manifest_sha256, "edge_source": graph.source,
            "anchor_in_llm_inputs": True, "queries_per_anchor": 1, "query_scope": "anchor_plus_all_matches",
            "anchor_self_membership": "included in group mean; omitted from matches to avoid duplicate input",
            "empty_group_policy": "skip without refilling", "small_group_policy": "skip anchors below minimum retrieved matches"}, "anchors": []}
        for anchor in anchors:
            print(f'Selecting the best prototype group for salient node {anchor["id"]}…', flush=True)
            hits, selection = graph.select(anchor)
            matches = []
            for hit in hits:
                node = graph.node(hit["node_id"])
                node.update({key: float(hit[key]) for key in (*METRICS, "visual_score")})
                node.update(selected_prototype_id=selection["selected_prototype_id"],
                            prototype_graph_rank=hit["prototype_graph_rank"],
                            prototype_activation=hit["prototype_activation"])
                matches.append(save_match_assets(node, source_dir, outdir))
            validate_retrieved_matches(anchor, matches, report)
            groups = split_matches(matches, args.saliency_percentile)
            entry = {"anchor": anchor, "matches": matches, "prototype_selection": selection,
                "matched_video_count": len({(n["dataset"], n["video"]) for n in matches}),
                "groups": {name: [n["id"] for n in members] for name, members in groups.items()},
                "query": new_anchor_query(anchor, matches, minimum)}
            report["anchors"].append(entry)
            print(f'Anchor {anchor["id"]}: prototype {selection["selected_prototype_id"]}, '
                  f'mean similarity {selection["selected_mean_visual_similarity"]}, {len(matches)} matches.', flush=True)
            write_reports(report, outdir)
        write_reports(report, outdir)
    return report, outdir


def archive_descriptions(report, reason):
    snapshot = {"reason": reason, "archived_at": datetime.now(timezone.utc).isoformat(),
                "llm": report.get("llm") or report.get("gemma"), "anchors": []}
    for entry in report["anchors"]:
        old = entry.get("queries") if "queries" in entry else entry.get("query")
        if old is not None:
            snapshot["anchors"].append({"anchor_id": entry["anchor"]["id"],
                                        "responses": json.loads(json.dumps(old))})
    report.setdefault("previous_llm_runs", []).append(snapshot)


def normalize_anchor_queries(report, minimum, reset=False):
    legacy = any("queries" in entry for entry in report["anchors"])
    if legacy or reset:
        archive_descriptions(report, "migrated_from_three_group_queries" if legacy else "requested_regeneration")
        report.pop("gemma", None); report.pop("llm", None)
    for entry in report["anchors"]:
        matches = entry["matches"]
        ids = [n["id"] for n in matches]
        if len(ids) != len(set(ids)) or entry["anchor"]["id"] in ids:
            raise ValueError("Retrieved matches must be unique nodes distinct from the anchor.")
        entry.pop("queries", None)
        if legacy or reset or "query" not in entry:
            entry["query"] = new_anchor_query(entry["anchor"], matches, minimum)
        query = entry["query"]
        if query["node_ids"] != ids or query.get("anchor_id") != entry["anchor"]["id"] or not query.get("includes_anchor"):
            raise ValueError("Saved query does not match this anchor's current retrieved patches. Use --reset-descriptions to regenerate it.")
        if not matches:
            query["status"] = "skipped_empty"
        elif len(matches) < minimum:
            query["status"] = "skipped_small_group"
        else:
            if query["status"] in ("skipped_empty", "skipped_small_group"):
                query["status"] = "pending"
            if not query.get("prompt"):
                query["prompt"] = prompt_for(len(matches))
        if "groups" not in entry:
            groups = split_matches(matches, report["protocol"]["saliency_percentile"])
            entry["groups"] = {name: [n["id"] for n in members] for name, members in groups.items()}
    report["schema_version"] = 2
    report["protocol"].update(min_group_size=minimum, queries_per_anchor=1,
        anchor_in_llm_inputs=True, query_scope="anchor_plus_all_matches")
    report["protocol"].pop("anchor_in_gemma_inputs", None)
    report["protocol"].pop("gemma_model", None)


def query_execution_summary(report):
    minimum = report["protocol"].get("min_group_size", MIN_GROUP_SIZE)
    queries = [entry["query"] for entry in report["anchors"]]
    return {"mode": "sequential_one_query_per_anchor", "queries_per_anchor": 1,
            "conversation_scope": "fresh_per_anchor", "anchor_in_prompt": True,
            "min_retrieved_matches": minimum, "salient_anchors": len(queries),
            "eligible_anchors": sum(len(q["node_ids"]) >= minimum for q in queries),
            "completed_model_queries": sum(len(q["node_ids"]) >= minimum and q["status"] == "complete" for q in queries),
            "skipped_empty_anchors": sum(not q["node_ids"] for q in queries),
            "skipped_small_anchors": sum(0 < len(q["node_ids"]) < minimum for q in queries)}


def describe_report(args, report, outdir):
    # Check the entire saved retrieval before starting any external requests.
    for entry in report["anchors"]:
        validate_retrieved_matches(entry["anchor"], entry["matches"], report)
    minimum = args.min_group_size
    normalize_anchor_queries(report, minimum, reset=getattr(args, "reset_descriptions", False))
    report["query_execution"] = summary = query_execution_summary(report)
    prior = report.get("llm")
    model_id = args.gemini_model.removeprefix("models/")
    if prior and summary["completed_model_queries"]:
        if prior.get("provider") != "gemini" or prior.get("model_id") != model_id or prior.get("max_new_tokens") != args.max_new_tokens:
            raise ValueError("Saved answers use another model/output budget. Use --reset-descriptions with --describe-only to archive them before rerunning.")
    if any(entry["query"]["status"] == "complete" and
           entry["query"].get("image_context_mode") != API_CONTEXT_MODE
           for entry in report["anchors"]):
        raise ValueError("Saved answers used the previous image input format. Use --describe-only --reset-descriptions "
                         "to archive and regenerate them with padded, boxed context crops.")
    report["protocol"].update(llm_provider="gemini", llm_model=model_id,
        video_identity="(dataset, video)",
        image_context_mode=API_CONTEXT_MODE, context_padding_fraction_per_side=API_CONTEXT_PADDING,
        context_target="original patch inside red box; surrounding pixels provide context only")
    pending_count = summary["eligible_anchors"]-summary["completed_model_queries"]
    if not pending_count:
        report["status"] = "complete"; report.pop("error", None); write_reports(report, outdir)
        return
    print(f'{summary["salient_anchors"]} salient anchors; {pending_count} Gemini queries pending. '
          f'Each query contains one anchor plus all its matches. '
          f'{summary["skipped_small_anchors"]+summary["skipped_empty_anchors"]} anchors have fewer than {minimum} matches and are skipped.', flush=True)
    try:
        report["status"] = "describing"; report.pop("error", None); write_reports(report, outdir)
        describer = GeminiDescriber(args, outdir)
        report["llm"] = describer.metadata
        write_reports(report, outdir)
        index = 0
        for anchor_index, entry in enumerate(report["anchors"], 1):
            anchor, matches, query = entry["anchor"], entry["matches"], entry["query"]
            query["anchor_index"] = anchor_index
            if len(matches) < minimum:
                print(f'Anchor {anchor["id"]}: {len(matches)} matches; requires ≥{minimum}; skipped.', flush=True)
                continue
            if query["status"] == "complete":
                print(f'Anchor {anchor["id"]}: already complete.', flush=True)
                continue
            index += 1
            paths = [save_api_context_crop(node, outdir, max_side=args.api_image_max_side)
                     for node in [anchor, *matches]]
            if not query.get("prompt", "").startswith(CONTEXT_INSTRUCTIONS):
                query["prompt"] = CONTEXT_INSTRUCTIONS + (query.get("prompt") or prompt_for(len(matches)))
            query.update(submitted_image_count=len(paths), submitted_match_count=len(matches),
                         image_context_mode=API_CONTEXT_MODE,
                         submitted_image_files=[path.relative_to(outdir).as_posix() for path in paths])
            write_reports(report, outdir)
            print(f'Query {index}/{pending_count} · salient anchor {anchor_index}/{summary["salient_anchors"]} '
                  f'· node {anchor["id"]} · anchor + {len(matches)} matches…', flush=True)
            try:
                answer = describer.describe(paths, query["prompt"])
            except Exception as error:
                query.update(status="failed", error=str(error)); write_reports(report, outdir)
                raise
            query.update(answer); query.pop("error", None)
            report["query_execution"] = query_execution_summary(report)
            write_reports(report, outdir)
        report["status"] = "complete"; write_reports(report, outdir)
    except Exception as error:
        report.update(status="incomplete", error=str(error)); write_reports(report, outdir)
        raise


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video-id')
    frame = parser.add_mutually_exclusive_group()
    frame.add_argument('--frame-no', type=int, help='One-based position in ordered loader frames, not a parsed filename number')
    frame.add_argument('--frame-name', help='Exact frame filename including extension')
    parser.add_argument('--dataset')
    parser.add_argument('--checkpoint', help='Defaults to graph checkpoint; must have the same SHA256')
    parser.add_argument('--evaluation', help='Defaults to evaluation.py path in graph manifest')
    parser.add_argument('--visual-script', type=Path, default=SCRIPT_DIR/'generate_vis_sim_graph.py', help=argparse.SUPPRESS)
    parser.add_argument('--prototype-script', type=Path, default=SCRIPT_DIR/'generate_prot_graph.py')
    parser.add_argument('--visual-graph-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--prototype-graph-dir', type=Path, help='Prototype graph output directory (defaults to the prototype generator OUTPUT_DIR)')
    parser.add_argument('--top-k', type=int, default=TOP_K, help='Legacy compatibility option; new retrieval uses every patch connected to the chosen prototype. Use --patch-grid-cols for layout.')
    parser.add_argument('--min-group-size', type=int, default=MIN_GROUP_SIZE, help=f'Minimum RETRIEVED matches per anchor (at least {MIN_GROUP_SIZE}); anchor not counted')
    parser.add_argument('--stage4-cosine-relaxation', type=float, default=STAGE4_COSINE_RELAXATION,
                        help='Subtract from the cached stage4 cosine cutoff (default STAGE4_COSINE_RELAXATION)')
    parser.add_argument('--min-stage4-cosine', type=float, help='Absolute cosine cutoff; overrides default relaxation')
    parser.add_argument('--max-rgb-mae', type=float, default=MAX_RGB_MAE,
                        help='Absolute RGB MAE cutoff (default MAX_RGB_MAE)')
    parser.add_argument('--min-rgb-structure-cosine', type=float, default=MIN_RGB_STRUCTURE_COSINE,
                        help='Absolute RGB structure cosine cutoff (default MIN_RGB_STRUCTURE_COSINE)')
    parser.add_argument('--saliency-percentile', type=float, default=SALIENCY_PERCENTILE)
    parser.add_argument('--patch-grid-cols', '--patch_grid_cols', dest='patch_grid_cols',
                        type=int, default=PATCH_GRID_COLS,
                        help='Patch cards per row, including the anchor (default top_k + 1; uses saved top_k in view/describe-only mode)')
    parser.add_argument('--heatmap-alpha', type=float, default=HEATMAP_ALPHA,
                        help='Opacity of the JET saliency heatmap blended onto the RGB frame (default 0.5)')
    parser.add_argument('--device', default='cuda:0', help='Saliency model device; Gemini runs remotely')
    parser.add_argument('--gemini-model', default=GEMINI_MODEL)
    parser.add_argument('--min-request-interval', type=float, default=MIN_REQUEST_INTERVAL_SECONDS, help='Minimum seconds between HTTP attempts, including retries; state persists in this run directory')
    parser.add_argument('--max-retries', type=int, default=3, help='Bounded retries for temporary 429/5xx errors; daily/zero quotas stop')
    parser.add_argument('--max-retry-wait', type=float, default=300.)
    parser.add_argument('--request-timeout', type=float, default=300.,
                        help='Per-request socket timeout in seconds for Gemini (default 300)')
    parser.add_argument('--api-image-max-side', type=int, default=API_IMAGE_MAX_SIDE,
                        help='Longest side in pixels for Gemini inline crops; never upscales (default 256)')
    parser.add_argument('--api-jpeg-quality', type=int, default=API_JPEG_QUALITY,
                        help='JPEG quality when encoding Gemini inline crops (default 90)')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--replay-atol', type=float, default=.005)
    parser.add_argument('--replay-score-atol', type=float, default=1e-5)
    parser.add_argument('--edge-source', choices=('auto','implicit'), default='auto')
    parser.add_argument('--output', type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare-only', action='store_true')
    action.add_argument('--describe-only', action='store_true')
    action.add_argument('--view-only', action='store_true', help='Rebuild and serve the saved report over local HTTP, then open the browser; Ctrl+C stops serving; no inference/API calls')
    parser.add_argument('--no-browser', action='store_true', help='Serve the report without automatically launching a browser in --view-only mode')
    parser.add_argument('--port', type=int, default=8765, help='Local report HTTP port for --view-only (default 8765; 0 selects a free port)')
    parser.add_argument('--reset-descriptions', action='store_true', help='Archive prior answers before intentionally regenerating them; requires --describe-only')
    args = parser.parse_args()
    if args.patch_grid_cols is not None and args.patch_grid_cols < 1:
        parser.error('--patch-grid-cols must be a positive integer.')
    if not 0 <= args.port <= 65535:
        parser.error('--port must be in [0,65535].')
    if args.describe_only or args.view_only:
        if args.output is None:
            parser.error('--describe-only/--view-only requires --output RUN_DIR.')
    elif args.video_id is None or (args.frame_name is None and args.frame_no is None):
        parser.error('Pass --video-id and either --frame-no or --frame-name.')
    if args.min_group_size < MIN_GROUP_SIZE or args.top_k < 1 or args.max_new_tokens < 1 or args.max_retries < 0:
        parser.error(f'Minimum group size is {MIN_GROUP_SIZE}; top-k/max-new-tokens must be positive and retries nonnegative.')
    for name in ('stage4_cosine_relaxation','min_request_interval','max_retry_wait','replay_atol','replay_score_atol'):
        value = getattr(args,name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{name.replace("_","-")} must be finite and nonnegative.')
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error('--request-timeout must be finite and positive.')
    if args.api_image_max_side < 1:
        parser.error('--api-image-max-side must be a positive integer.')
    if not 1 <= args.api_jpeg_quality <= 100:
        parser.error('--api-jpeg-quality must be in [1,100].')
    for name,low,high in [('saliency_percentile',0,1),('min_stage4_cosine',-1,1),('max_rgb_mae',0,1),('min_rgb_structure_cosine',-1,1),('heatmap_alpha',0,1)]:
        value = getattr(args,name)
        if value is not None and not low <= value <= high:
            parser.error(f'--{name.replace("_","-")} must be in [{low},{high}].')
    if args.reset_descriptions and not args.describe_only:
        parser.error('--reset-descriptions requires --describe-only.')
    return args


def open_report_in_browser(url):
    """Request a browser tab for the HTTP report; leave manual access available."""
    try:
        opened = webbrowser.open(url, new=2)
    except (webbrowser.Error, OSError) as error:
        print(f"Could not open the default browser: {error}", flush=True)
        opened = False
    if opened:
        print(f"Requested browser tab: {url}", flush=True)
    else:
        print(f"Automatic browser opening was unavailable. Paste this address into your browser: {url}\n"
              "Browser opening runs on the machine executing Python. If using SSH, "
              "forward the report port to your computer first.", flush=True)
    return bool(opened)


class ReportRequestHandler(SimpleHTTPRequestHandler):
    """Serve saved report assets without stale browser caches or noisy logs."""
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, format, *args):
        pass


def serve_report(outdir, port=8765, open_browser=True):
    outdir = Path(outdir).resolve()
    if not (outdir / 'report.html').is_file():
        raise FileNotFoundError(f"Report has not been generated: {outdir / 'report.html'}")
    handler = partial(ReportRequestHandler, directory=str(outdir))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    except OSError as error:
        raise RuntimeError(f"Could not start the local report server on port {port}: {error}. "
                           "Try --port 0 to select a free port.") from error
    with server:
        url = f"http://127.0.0.1:{server.server_address[1]}/report.html"
        print(f"Open report: {url}\nKeep this terminal running. Press Ctrl+C to stop.", flush=True)
        if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_TTY'):
            actual_port = server.server_address[1]
            print(f"Remote session detected. Forward port {actual_port} in your SSH client "
                  f"(ssh -N -L {actual_port}:127.0.0.1:{actual_port} user@remote-host), "
                  "then open the HTTP address on your computer.", flush=True)
        if open_browser:
            # Some browser controllers block until their window closes. Keep
            # serving on the main thread regardless of the controller behavior.
            Thread(target=open_report_in_browser, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\nReport server stopped.', flush=True)


def main():
    args = arguments()
    if args.describe_only or args.view_only:
        outdir = args.output.resolve()
        report = json.loads((outdir/'report.json').read_text())
    else:
        report, outdir = prepare(args)
    report.setdefault('protocol', {})['heatmap_alpha'] = float(args.heatmap_alpha)
    report['protocol']['heatmap_display'] = HEATMAP_DISPLAY_FORMAT
    report['protocol']['patch_grid_cols'] = (
        args.patch_grid_cols or (int(report['protocol'].get('top_k', args.top_k)) + 1))
    if args.view_only:
        legacy = any('queries' in entry for entry in report['anchors'])
        normalize_anchor_queries(report, report['protocol'].get('min_group_size', MIN_GROUP_SIZE))
        if legacy:
            report['status'] = 'prepared'
        report['query_execution'] = query_execution_summary(report)
        write_reports(report, outdir, heatmap_alpha=args.heatmap_alpha)
    elif not args.prepare_only:
        describe_report(args, report, outdir)
    print(f"Report: {outdir/'report.html'}\nData: {outdir/'report.json'}", flush=True)
    for entry in report['anchors']:
        print(f"Anchor {entry['anchor']['id']}: {query_status_text(entry['query'],args.min_group_size)}")
    if args.view_only:
        serve_report(outdir, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        sys.exit(f'{type(error).__name__}: {error}')
