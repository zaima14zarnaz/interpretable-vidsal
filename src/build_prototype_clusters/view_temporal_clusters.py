#!/usr/bin/env python3
"""Inspect temporal clusters in one layout: local/mean arrows and patch clips.

  python view_temporal_clusters.py --cluster-dir /path/to/temporal_bank
  python view_temporal_clusters.py --prototype 12 --columns 3 --no-browser
  python view_temporal_clusters.py --fps 30 --clip-frames 16

Dependencies: numpy, opencv-python[-headless], Pillow, and system FFmpeg with
H.264 encoding. Torch is needed only to read legacy v1 .pt flow caches.

Each card has one fixed layout: smooth side-by-side local and mean SVG arrows,
a silent looping cropped RGB window clip, metadata, a compact dark-gray compass,
and a vertical static-mass bar on its right. The reference-panel magnitude
heatmap and shared speed scale are retained. No alternative display modes,
raw-flow images, RGB still gallery, overlays, direction wheel, or mode routing.

Local and mean arrows use the selected frame gap and the same target-aligned,
camera-relative cached flow. Local arrows summarize a 4x4 spatial grid; the mean
arrow summarizes the whole patch. Color is mean speed over valid pixels in
pixels/frame, on one shared log/TURBO scale. Neutral symbols indicate static,
invalid, or mixed local flow. The compass pools all usable recorded gaps.
Its arrow lengths use directional mean speed, the shared scale, and a minimum
nonzero length (--min-arrow-fraction .25 by default). Zero flow has no arrow.
Static mass excludes unreliable pixels and is a separate vertical 0..100% bar.

Diagnostic clips contain exactly target_index-gap through target_index,
inclusive, for the same selected gap as both arrow panels. For example a target
of 351 and gap 6 produces frames 345..351 (seven images). --fps is the assumed
source FPS; the MP4 itself is encoded at fps*--slow-motion (.25 by default), so
both the viewer and downloaded clips play slowly. --clip-frames is deprecated.

Each clip has Slow loop and Blink endpoints controls. Blinking alternates two
lossless PNG crops of the actual endpoint source frames at --blink-ms (default
500), with the frame index outside the image. Both occupy exactly the same
screen rectangle; no text overlays, crossfade, frame interpolation, H.264 seek,
or motion compensation are applied to the RGB comparison. Playback and blinking
pause when offscreen or the document is hidden. The crop is a fixed spatial cell,
not an object track. Flow arrows may be camera-relative; RGB is the source view.

The bank and flow caches remain read-only. Generated silent H.264 MP4s go in a
separate cache outside the bank. No flow estimation, backbone/image feature
extraction, prototype reassignment or training occurs. Missing flow entries are
reported, never recomputed. Keep original RGB frames and recorded flow caches
accessible; --image-root OLD_ROOT=NEW_ROOT remaps moved paths, and
--flow-cache-dir overrides the exact flow namespace directory.

Supports fixed_stage4_temporal_clusters_v2_flow_only and legacy v1 banks.
Legacy assignments may also depend on feature changes that this flow display
does not explain. The server binds only to 127.0.0.1. For remote use:
  ssh -L 8768:127.0.0.1:8768 user@server
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
from contextlib import closing
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import sqlite3
import sys
from threading import BoundedSemaphore, Timer
from urllib.parse import parse_qs, urlparse
import webbrowser

import cv2
import numpy as np
from PIL import Image

DEFAULT_CLUSTER_DIR = Path(
    "/data/quantization/zaima/videosal_datasets/dhf1k/concept_clusters/temporal_clusters_flow_only_v2"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
FORMATS = {
    "fixed_stage4_temporal_clusters_v1",
    "fixed_stage4_temporal_clusters_v2_flow_only",
}


def readonly(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def target_validity(forward, backward, absolute, relative):
    h, w = backward.shape[:2]
    y, x = np.mgrid[:h, :w].astype(np.float32)
    px, py = x + backward[..., 0], y + backward[..., 1]
    sampled = cv2.remap(
        forward, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    error = np.linalg.norm(backward + sampled, axis=-1)
    tolerance = absolute + relative * (
        np.linalg.norm(backward, axis=-1) + np.linalg.norm(sampled, axis=-1)
    )
    return (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1) & (error <= tolerance)


def recorded_camera_field(shape, camera):
    """Replay the recorded camera fit; never refit or rerun a flow estimator."""
    if camera.get("used") == "none" or camera.get("used") == "no_valid_flow":
        return np.zeros(shape, np.float32)
    if "affine" in camera and camera.get("used") == "affine":
        affine = np.asarray(camera["affine"], np.float32)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise ValueError("Invalid recorded camera affine matrix.")
        y, x = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
        xy = np.stack((x, y), axis=-1)
        grid = np.concatenate((xy, np.ones((*shape[:2], 1), np.float32)), axis=-1)
        return (grid @ affine.T - xy).astype(np.float32)
    if "vector" in camera:
        vector = np.asarray(camera["vector"], np.float32)
        if vector.shape != (2,) or not np.isfinite(vector).all():
            raise ValueError("Invalid recorded camera vector.")
        return np.broadcast_to(vector, shape)
    raise ValueError(
        "Recorded camera model missing; refusing to guess a different compensation."
    )


def magnitude_colors(speed, scale):
    """One shared log-speed map for arrows, heatmaps and legend."""
    normalized = np.clip(np.log1p(speed) / np.log1p(scale), 0, 1)
    values = np.round(255 * normalized).astype(np.uint8)
    return cv2.cvtColor(
        cv2.applyColorMap(values, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )


def magnitude_rgb(velocity, valid, scale, static_threshold):
    speed = np.linalg.norm(velocity, axis=-1)
    rgb = magnitude_colors(speed, scale)
    rgb[valid & (speed <= static_threshold)] = 0
    rgb[~valid] = (155, 164, 179)
    return rgb


def summarize_patch_motion(
    velocity, valid, threshold, min_valid=0.2, static_fraction=0.9
):
    """Summarize a selected patch without presenting missing data as movement.

    Orientation is the mean of valid MOVING vectors. Color is mean speed over
    ALL valid pixels, matching the per-gap descriptor's speed definition.
    Coherence is ||sum(v)|| / sum(||v||) for moving vectors. Opposing motion
    therefore remains explicitly mixed instead of producing an arbitrary arrow.
    """
    fraction = float(valid.mean())
    result = dict(
        valid_fraction=fraction,
        speed=0.0,
        coherence=0.0,
        angle=0.0,
        moving_fraction=0.0,
        status="invalid",
        label="No reliable flow",
    )
    if fraction < min_valid or not valid.any():
        return result
    vectors = velocity[valid]
    speeds = np.linalg.norm(vectors, axis=-1)
    moving = speeds > threshold
    result.update(speed=float(speeds.mean()), moving_fraction=float(moving.mean()))
    if not moving.any() or float(moving.mean()) <= 1 - static_fraction:
        result.update(status="static", label="Mostly static")
        return result
    selected = vectors[moving]
    mean = selected.mean(axis=0)
    coherence = float(
        np.linalg.norm(selected.sum(axis=0))
        / np.maximum(np.linalg.norm(selected, axis=-1).sum(), 1e-12)
    )
    angle = float(np.arctan2(mean[1], mean[0]))
    result.update(coherence=coherence, angle=angle)
    if coherence < 0.35:
        result.update(status="mixed", label="Mixed directions")
    else:
        labels = (
            "Rightward",
            "Down-right",
            "Downward",
            "Down-left",
            "Leftward",
            "Up-left",
            "Upward",
            "Up-right",
        )
        result.update(
            status="moving",
            label=labels[int(np.floor((angle % (2 * np.pi)) / (np.pi / 4) + 0.5)) % 8],
        )
    return result


def arrow_color(summary, scale):
    return tuple(
        int(c)
        for c in magnitude_colors(np.array([[summary["speed"]]], np.float32), scale)[
            0, 0
        ]
    )


def centered_arrow_svg(summary, scale):
    """Vector glyph drawn in display coordinates, never enlarged from flow pixels."""
    color = "#%02x%02x%02x" % arrow_color(summary, scale)
    angle = np.degrees(summary["angle"])
    if summary["status"] == "moving":
        glyph = f'<g transform="translate(120 120) scale(.8) translate(-120 -120) rotate({angle:.4f} 120 120) translate(0 8)"><path d="M 54 103 L 148 103 L 148 84 L 186 112 L 148 140 L 148 121 L 54 121 Z" fill="{color}" stroke="#24344b" stroke-width="1.4" stroke-linejoin="round"/></g>'
        detail = f"{summary['speed']:.3f} px/frame"
    elif summary["status"] == "static":
        glyph = '<circle cx="120" cy="112" r="19" fill="#e8edf4" stroke="#67778c" stroke-width="2"/><circle cx="120" cy="112" r="5" fill="#67778c"/>'
        detail = f"{summary['speed']:.3f} px/frame"
    elif summary["status"] == "mixed":
        glyph = '<g fill="none" stroke="#64748b" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"><path d="M 75 100 H 165 M 150 86 L 165 100 L 150 114"/><path d="M 165 129 H 75 M 90 115 L 75 129 L 90 143"/></g>'
        detail = f"{summary['speed']:.3f} px/frame · mixed motion"
    else:
        glyph = '<circle cx="120" cy="112" r="25" fill="#f1f4f8" stroke="#94a3b8" stroke-width="2"/><path d="M 104 128 L 136 96" stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>'
        detail = "Insufficient valid measurements"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" role="img" aria-label="{summary["label"]}" preserveAspectRatio="xMidYMid meet">'
        '<rect width="240" height="240" fill="#ffffff"/>'
        '<text x="120" y="22" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" font-weight="600" letter-spacing="1.5" fill="#7b899b">MEAN FLOW</text>'
        + glyph
        + f'<text x="120" y="193" text-anchor="middle" font-family="system-ui,sans-serif" font-size="17" font-weight="650" fill="#20314a">{summary["label"]}</text>'
        f'<text x="120" y="214" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="#64748b">{detail}</text>'
        f'<text x="120" y="234" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" fill="#7b899b">Valid flow: {100*summary["valid_fraction"]:.0f}%</text></svg>'
    ).encode()


def compass_flow_stats(samples, threshold, min_valid=0.2, min_valid_gaps=0.5):
    """Pool rates from actual cached flows; histogram mass is NOT speed.

    Each sample is (target-aligned residual velocity [H,W,2], validity [H,W]).
    Soft circular membership matches the eight compass centers. Average each
    gap's contributions over its valid pixels, then pool gaps with equal weight.
    Directional mean speed = pooled speed sum / pooled direction membership.
    No image features, flow estimation, per-card scaling or clustering occurs.
    """
    numerator = np.zeros(8, np.float64)
    mass = np.zeros(8, np.float64)
    fractions = []
    static_mass = 0.0
    usable = 0
    for velocity, valid in samples:
        fraction = float(valid.mean())
        fractions.append(fraction)
        if fraction < min_valid or not valid.any():
            continue
        usable += 1
        vectors = velocity[valid].astype(np.float64)
        speed = np.linalg.norm(vectors, axis=-1)
        moving = speed > threshold
        static_mass += float((~moving).mean())
        if not moving.any():
            continue
        vectors = vectors[moving]
        speed = speed[moving]
        bins = (
            np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi)
            * 8
            / (2 * np.pi)
        )
        base = np.floor(bins).astype(int)
        fraction = bins - base
        for indices, weights in ((base % 8, 1 - fraction), ((base + 1) % 8, fraction)):
            mass += np.bincount(indices, weights=weights, minlength=8) / int(
                valid.sum()
            )
            numerator += np.bincount(
                indices, weights=weights * speed, minlength=8
            ) / int(valid.sum())
    good = usable >= math.ceil(len(samples) * min_valid_gaps) and usable > 0
    speeds = np.divide(numerator, mass, out=np.zeros(8), where=mass > 1e-12)
    return dict(
        speeds=speeds.tolist(),
        fractions=(mass / max(usable, 1)).tolist(),
        valid=good,
        valid_fraction=float(np.mean(fractions)) if fractions else 0.0,
        static_fraction=static_mass / max(usable, 1),
        usable_gaps=usable,
        total_gaps=len(samples),
    )


def compass_svg(data, scale, min_arrow_fraction=0.25):
    """Compass with a shared magnitude scale and an explicit visibility floor."""
    if not math.isfinite(min_arrow_fraction) or not 0 <= min_arrow_fraction <= 1:
        raise ValueError("Minimum arrow fraction must be in 0..1.")
    center = 120.0
    radius = 80.0
    labels = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 286" role="img" aria-label="Pooled compass flow magnitudes" preserveAspectRatio="xMidYMid meet">',
        '<rect width="240" height="286" fill="white"/>',
        '<circle cx="120" cy="120" r="80" fill="none" stroke="#b7c3d3" stroke-width="1.3"/>',
        '<circle cx="120" cy="120" r="40" fill="none" stroke="#e3e9f1" stroke-width="1" stroke-dasharray="3 3"/>',
    ]
    for i, label in enumerate(labels):
        angle = i * np.pi / 4
        dx, dy = np.cos(angle), np.sin(angle)
        parts.append(
            f'<line x1="120" y1="120" x2="{center+radius*dx:.3f}" y2="{center+radius*dy:.3f}" stroke="#e6ebf2" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{center+104*dx:.3f}" y="{center+104*dy:.3f}" text-anchor="middle" dominant-baseline="central" font-family="system-ui,sans-serif" font-size="13" font-weight="600" fill="#34465f">{label}</text>'
        )
    drawn = 0
    if data["valid"]:
        for i, (speed, mass) in enumerate(zip(data["speeds"], data["fractions"])):
            if mass <= 1e-8 or speed <= 0:
                continue
            drawn += 1
            angle = i * np.pi / 4
            direction = np.array([np.cos(angle), np.sin(angle)])
            magnitude_length = radius * min(speed / scale, 1.0)
            length = max(radius * min_arrow_fraction, magnitude_length)
            tip = np.array([center, center]) + direction * length
            head = min(12.0, length * 0.4)
            perpendicular = np.array([-direction[1], direction[0]])
            left = tip - direction * head + perpendicular * head * 0.55
            right = tip - direction * head - perpendicular * head * 0.55
            capped = "; at scale limit" if speed > scale else ""
            if magnitude_length < radius * min_arrow_fraction:
                capped += "; enlarged to minimum display length"
            parts.append(
                f'<g data-direction="{labels[i]}" data-speed="{speed:.9g}" data-radius="{length:.6f}" data-magnitude-radius="{magnitude_length:.6f}"><title>{labels[i]}: {speed:.4f} px/frame; {100*mass:.1f}% of valid flow{capped}</title>'
            )
            parts.append(
                f'<line x1="120" y1="120" x2="{tip[0]:.6f}" y2="{tip[1]:.6f}" stroke="#374151" stroke-width="3" stroke-linecap="round"/>'
            )
            parts.append(
                f'<polygon points="{tip[0]:.6f},{tip[1]:.6f} {left[0]:.6f},{left[1]:.6f} {right[0]:.6f},{right[1]:.6f}" fill="#374151"/>'
            )
            if speed > scale:
                # Hollow marker explicitly flags a clipped value at the rim.
                parts.append(
                    f'<circle cx="{tip[0]:.6f}" cy="{tip[1]:.6f}" r="3.2" fill="white" stroke="#374151" stroke-width="1.2"/>'
                )
            parts.append("</g>")
    if drawn:
        parts.append('<circle cx="120" cy="120" r="1.7" fill="#374151"/>')
    else:
        label = "No reliable flow" if not data["valid"] else "Mostly static"
        parts.append(
            f'<text x="120" y="120" text-anchor="middle" dominant-baseline="central" font-family="system-ui,sans-serif" font-size="12" fill="#67778d">{label}</text>'
        )
    parts.append(
        f'<text x="120" y="246" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" fill="#5c6c81">Rim = {scale:.3f} px/frame</text>'
    )
    parts.append(
        f'<text x="120" y="261" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" fill="#7b899b">Valid flow: {100*data["valid_fraction"]:.0f}%</text>'
    )
    parts.append(
        f'<text x="120" y="279" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" fill="#7b899b">Min visible arrow = {100*min_arrow_fraction:.0f}% of radius</text></svg>'
    )
    return "".join(parts).encode()


def static_bar_svg(data):
    """Vertical static-mass graph: 0% at bottom, 100% at top."""
    available = bool(data["valid"])
    fraction = float(np.clip(data["static_fraction"], 0, 1))
    label = f"{100*fraction:.1f}%" if available else "N/A"
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 76 286" role="img" aria-label="Vertical static flow fraction" preserveAspectRatio="xMidYMid meet">',
        '<rect width="76" height="286" fill="white"/>',
        "<title>Static mass: fraction of valid pixels below the motion threshold. Unreliable flow is excluded.</title>",
        '<text x="30" y="17" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" font-weight="600" fill="#34465f">Static</text>',
        '<rect x="17" y="40" width="20" height="160" rx="3" fill="#edf1f6" stroke="#c9d3df" stroke-width=".8"/>',
    ]
    if available:
        height = 160 * fraction
        parts.append(
            f'<rect data-static-fraction="{fraction:.9g}" x="17" y="{200-height:.6f}" width="20" height="{height:.6f}" rx="2" fill="#475569"/>'
        )
    for y, text in ((40, "100%"), (120, "50%"), (200, "0%")):
        parts.append(
            f'<line x1="39" y1="{y}" x2="43" y2="{y}" stroke="#9aa9bc" stroke-width=".8"/><text x="46" y="{y}" dominant-baseline="central" font-family="system-ui,sans-serif" font-size="8" fill="#728197">{text}</text>'
        )
    parts.append(
        f'<text x="30" y="222" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" font-weight="600" fill="#34465f">{label}</text>'
    )
    if not available:
        parts.append(
            '<text x="30" y="239" text-anchor="middle" font-family="system-ui,sans-serif" font-size="8" fill="#728197">Unavailable</text>'
        )
    parts.append("</svg>")
    return "".join(parts).encode()


def local_flow_svg(velocity, valid, threshold, scale, grid=4, min_valid=0.2):
    """Smooth local arrows drawn directly in display-space SVG coordinates.

    Each of 4x4 spatial cells summarizes the same selected-gap residual flow
    and valid pixels as the mean-arrow view. Arrow direction is its valid
    moving-vector mean; color is mean speed over all valid local pixels.
    Fixed display lengths show direction clearly. Static/unknown/mixed cells
    have neutral symbols, not enlarged pixel masks or invented flow arrows.
    """
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240" role="img" aria-label="Local optical flow arrows" preserveAspectRatio="xMidYMid meet">',
        '<rect width="240" height="240" fill="white"/>',
        '<text x="120" y="22" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" font-weight="600" letter-spacing="1.5" fill="#7b899b">LOCAL FLOW</text>',
    ]
    h, w = valid.shape
    for row in range(grid):
        for col in range(grid):
            sl = np.s_[
                row * h // grid : (row + 1) * h // grid,
                col * w // grid : (col + 1) * w // grid,
            ]
            cell_valid = valid[sl]
            cx = 22 + (col + 0.5) * 196 / grid
            cy = 38 + (row + 0.5) * 174 / grid
            if (
                not cell_valid.size
                or not cell_valid.any()
                or cell_valid.mean() < min_valid
            ):
                parts.append(
                    f'<g data-cell="{row},{col}" data-state="invalid"><title>Insufficient valid flow</title><path d="M {cx-3:.3f} {cy-3:.3f} L {cx+3:.3f} {cy+3:.3f} M {cx-3:.3f} {cy+3:.3f} L {cx+3:.3f} {cy-3:.3f}" stroke="#c1cad6" stroke-width="1.3" stroke-linecap="round"/></g>'
                )
                continue
            vectors = velocity[sl][cell_valid]
            speeds = np.linalg.norm(vectors, axis=-1)
            moving = speeds > threshold
            if not moving.any():
                parts.append(
                    f'<g data-cell="{row},{col}" data-state="static"><title>Static flow</title><circle cx="{cx:.3f}" cy="{cy:.3f}" r="2.2" fill="#c1cad6"/></g>'
                )
                continue
            selected = vectors[moving].astype(np.float64)
            mean = selected.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            coherence = float(
                np.linalg.norm(selected.sum(axis=0))
                / max(np.linalg.norm(selected, axis=-1).sum(), 1e-12)
            )
            if coherence < 0.35 or norm <= 1e-12:
                parts.append(
                    f'<g data-cell="{row},{col}" data-state="mixed"><title>Mixed local directions</title><circle cx="{cx:.3f}" cy="{cy:.3f}" r="5" fill="none" stroke="#a7b4c4" stroke-width="1.4"/></g>'
                )
                continue
            speed = float(speeds.mean())
            color = "#%02x%02x%02x" % tuple(
                int(c)
                for c in magnitude_colors(np.array([[speed]], np.float32), scale)[0, 0]
            )
            angle = float(np.arctan2(mean[1], mean[0]))
            degrees = np.degrees(angle)
            length = min(35.0, 196 / grid * 0.76)
            # Vector coordinates are in screen space, independent of the source
            # patch's pixel resolution. Browser scaling preserves smooth edges.
            unit = np.array([np.cos(angle), np.sin(angle)])
            parts.append(
                f'<g data-cell="{row},{col}" data-state="moving" data-dx="{unit[0]:.8f}" data-dy="{unit[1]:.8f}" data-speed="{speed:.8f}" transform="translate({cx:.3f} {cy:.3f}) rotate({degrees:.4f})"><title>Local mean: {speed:.3f} px/frame; valid {100*cell_valid.mean():.0f}%</title>'
            )
            parts.append(
                f'<path d="M {-length/2:.3f} -3 L {length/2-10:.3f} -3 L {length/2-10:.3f} -8 L {length/2:.3f} 0 L {length/2-10:.3f} 8 L {length/2-10:.3f} 3 L {-length/2:.3f} 3 Z" fill="{color}" stroke="#24344b" stroke-width=".8" stroke-linejoin="round"/></g>'
            )
    parts.append(
        '<text x="120" y="231" text-anchor="middle" font-family="system-ui,sans-serif" font-size="8.5" fill="#7b899b">dot: static · cross: invalid · ring: mixed</text></svg>'
    )
    return "".join(parts).encode()


def encode_png(image):
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


class PatchClipStore:
    def __init__(
        self,
        directory,
        *,
        image_roots=(),
        cache_dir=None,
        fps=30,
        clip_frames=0,
        slow_motion=0.25,
        ffmpeg="ffmpeg",
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if (
            self.manifest.get("status") != "complete"
            or self.manifest.get("format_version") not in FORMATS
        ):
            raise ValueError(
                "Expected a completed temporal bank from form_temporal_clusters.py."
            )
        self.db_path = self.directory / "temporal_clusters.sqlite"
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Missing database: {self.db_path}")
        self.fps = float(fps)
        self.slow_motion = float(slow_motion)
        if not math.isfinite(self.slow_motion) or not 0 < self.slow_motion <= 1:
            raise ValueError("Slow-motion factor must be in (0,1].")
        self.playback_fps = self.fps * self.slow_motion
        self.gaps = sorted(
            set(
                int(g)
                for g in self.manifest["descriptor"].get(
                    "flow_frame_gaps", [self.manifest["frame_span"]]
                )
            )
        )
        self.default_gap = 6 if 6 in self.gaps else self.gaps[0]
        self.clip_frames = int(clip_frames)
        if self.clip_frames:
            print(
                "NOTE: --clip-frames is ignored; diagnostic clips use exactly the selected flow interval.",
                file=sys.stderr,
            )
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 120:
            raise ValueError("Preview FPS must be in 1..120.")
        self.ffmpeg = shutil.which(str(ffmpeg))
        if not self.ffmpeg:
            raise ValueError(
                "FFmpeg is required. Install it and put it on PATH, or use --ffmpeg /path/to/ffmpeg."
            )
        self.cache_dir = (
            Path(cache_dir or Path(tempfile.gettempdir()) / "temporal_cluster_clips_v1")
            .expanduser()
            .resolve()
        )
        if self.cache_dir == self.directory or self.directory in self.cache_dir.parents:
            raise ValueError(
                "The clip cache must be outside the cluster bank directory."
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_roots = []
        for mapping in image_roots:
            old, sep, new = mapping.partition("=")
            if not sep or not old or not new:
                raise ValueError("--image-root must be OLD_ROOT=NEW_ROOT.")
            self.image_roots.append((Path(old).expanduser(), Path(new).expanduser()))
        self.image_roots.sort(key=lambda pair: len(str(pair[0])), reverse=True)
        self.encoders = BoundedSemaphore(2)
        with closing(readonly(self.db_path)) as con:
            self.prototypes = [
                dict(row) for row in con.execute("SELECT * FROM prototypes ORDER BY id")
            ]
        if (
            not self.prototypes
            or len(self.prototypes) != self.manifest["num_prototypes"]
        ):
            raise ValueError(
                "Temporal prototype table does not match the completed manifest."
            )

    def resolve_image(self, value):
        original = Path(value).expanduser()
        for old, new in self.image_roots:
            try:
                mapped = new / original.relative_to(old)
            except ValueError:
                continue
            if mapped.is_file():
                return mapped
        if original.is_file():
            return original
        raise FileNotFoundError(
            f"Source frame unavailable: {original}. Use --image-root OLD_ROOT=NEW_ROOT if it moved."
        )

    def patch(self, node):
        with closing(readonly(self.db_path)) as con:
            row = con.execute(
                """SELECT p.*,f.dataset,f.video,f.window_start,f.target_index,
                f.target_name,f.previous_index,f.previous_name,f.previous_slice,f.current_slice,
                f.input_h,f.input_w,f.source_path,f.previous_source_path
                FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.node_id=?""",
                (node,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown patch node {node}.")
        return dict(row)

    def clip_sources(self, patch, gap=None):
        target = self.resolve_image(patch["source_path"])
        previous = self.resolve_image(patch["previous_source_path"])
        if previous.parent != target.parent:
            raise ValueError(
                "Descriptor endpoint frames must be in the same source directory."
            )
        files = sorted(
            (
                p
                for p in target.parent.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=natural_key,
        )
        # Reproduce DatasetLoader's natural order and verify recorded indices.
        end, before, start = (
            patch["target_index"],
            patch["previous_index"],
            patch["window_start"],
        )
        if not (0 <= start <= before < end < len(files)):
            raise ValueError("Invalid recorded frame indices or missing source frames.")
        if (
            files[end].name != patch["target_name"]
            or files[before].name != patch["previous_name"]
        ):
            raise ValueError(
                "Source frame order no longer matches the bank. Restore the original frame directory."
            )
        gap = self.default_gap if gap is None else int(gap)
        if gap not in self.gaps:
            raise ValueError(f"Gap must be one of {self.gaps}.")
        start = end - gap
        if start < patch["window_start"]:
            raise ValueError("Selected flow endpoint precedes the recorded window.")
        return files[start : end + 1], start

    def render_frame(self, path, patch, frame_index):
        width, height = patch["input_w"], patch["input_h"]
        if not (1 <= width <= 2048 and 1 <= height <= 2048):
            raise ValueError("Unsupported canvas dimensions.")
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        x0 = round(patch["col"] * width / patch["grid_w"])
        x1 = round((patch["col"] + 1) * width / patch["grid_w"])
        y0 = round(patch["row"] * height / patch["grid_h"])
        y1 = round((patch["row"] + 1) * height / patch["grid_h"])
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Invalid stored patch geometry.")
        image = image.crop((x0, y0, x1, y1)).resize(
            (192, 192), Image.Resampling.BILINEAR
        )
        # yuv420p needs even dimensions; do not change the grid's aspect ratio.
        if image.width % 2 or image.height % 2:
            padded = Image.new(
                "RGB", (image.width + image.width % 2, image.height + image.height % 2)
            )
            padded.paste(image)
            image = padded
        return image

    def clip(self, node, gap=None):
        gap = self.default_gap if gap is None else int(gap)
        patch = self.patch(node)
        sources, start = self.clip_sources(patch, gap)
        identity = {
            "renderer": "flow_interval_slow_clip_v3",
            "flow_gap": gap,
            "slow_motion": self.slow_motion,
            "bank": str(self.directory),
            "patch": patch,
            "fps": self.fps,
            "start": start,
            "files": [
                (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources
            ],
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        destination = self.cache_dir / f"{key}.mp4"
        if destination.is_file() and destination.stat().st_size:
            return destination
        with self.encoders:
            if destination.is_file() and destination.stat().st_size:
                return destination
            images = [
                self.render_frame(p, patch, start + i) for i, p in enumerate(sources)
            ]
            width, height = images[0].size
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", dir=self.cache_dir, delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                command = [
                    self.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(self.playback_fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ]
                result = subprocess.run(
                    command,
                    check=False,
                    input=b"".join(image.tobytes() for image in images),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=120,
                )
                if result.returncode:
                    raise RuntimeError(
                        "FFmpeg encoding failed: "
                        + result.stderr.decode(errors="replace")[-1500:]
                    )
                if temporary.stat().st_size == 0:
                    raise RuntimeError("FFmpeg produced an empty clip.")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return destination

    @lru_cache(maxsize=512)
    def endpoint(self, node, gap, which):
        if which not in ("previous", "target"):
            raise ValueError("Endpoint must be previous or target.")
        patch = self.patch(node)
        sources, start = self.clip_sources(patch, gap)
        index = 0 if which == "previous" else len(sources) - 1
        return encode_png(self.render_frame(sources[index], patch, start + index))


def byte_range(value, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
    if not match or not any(match.groups()):
        raise ValueError("Invalid byte range.")
    left, right = match.groups()
    if not left:
        length = int(right)
        if length <= 0:
            raise ValueError("Invalid suffix range.")
        return max(0, size - length), size - 1
    start, end = int(left), min(int(right), size - 1) if right else size - 1
    if start >= size or end < start:
        raise ValueError("Unsatisfiable byte range.")
    return start, end


class ClusterStore:
    def __init__(
        self,
        directory,
        image_roots=(),
        flow_cache_dir=None,
        flow_scale=None,
        min_arrow_fraction=0.25,
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if (
            self.manifest.get("status") != "complete"
            or self.manifest.get("format_version") not in FORMATS
        ):
            raise ValueError(
                "Expected a complete v1 or v2 temporal cluster bank, not a spatial patch bank."
            )
        self.db_path = self.directory / self.manifest.get(
            "database", "temporal_clusters.sqlite"
        )
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Temporal database missing: {self.db_path}")
        self.clip_options = {}
        self.clip_store = None
        self.clip_errors = {}
        self.blink_ms = 500
        self.image_root_options = tuple(image_roots)
        self.v2 = self.manifest["format_version"].endswith("v2_flow_only")
        self.desc = self.manifest["descriptor"]
        self.gaps = sorted(
            set(
                int(g)
                for g in self.desc.get("flow_frame_gaps", [self.manifest["frame_span"]])
            )
        )
        if not self.gaps or min(self.gaps) <= 0:
            raise ValueError("Frame gaps must be positive.")
        self.default_gap = 6 if 6 in self.gaps else self.gaps[0]
        self.image_roots = []
        for mapping in image_roots:
            old, sep, new = mapping.partition("=")
            if not sep or not old or not new:
                raise ValueError("--image-root must have the form OLD_ROOT=NEW_ROOT.")
            self.image_roots.append((Path(old).expanduser(), Path(new).expanduser()))
        self.image_roots.sort(key=lambda p: len(str(p[0])), reverse=True)
        recorded_cache = self.manifest.get("flow_cache_directory")
        if flow_cache_dir is None and not recorded_cache:
            raise ValueError(
                "No flow cache location recorded; supply --flow-cache-dir with the namespace directory."
            )
        self.flow_cache = (
            Path(flow_cache_dir).expanduser().resolve()
            if flow_cache_dir
            else self.remap(recorded_cache)
        )
        scale = float(
            self.manifest.get("normalization", {}).get("flow_magnitude_scale", 1)
        )
        if not self.v2:
            scale /= self.gaps[0]
        self.min_arrow_fraction = float(min_arrow_fraction)
        if (
            not math.isfinite(self.min_arrow_fraction)
            or not 0 <= self.min_arrow_fraction <= 1
        ):
            raise ValueError("Minimum arrow fraction must be in 0..1.")
        self.scale = float(flow_scale if flow_scale is not None else scale)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(
                "Flow display scale must be finite and positive, in pixels/frame."
            )
        self.threshold = float(self.desc.get("flow_static_threshold", 0.05))
        if not self.v2:
            self.threshold /= self.gaps[0]
        self.hist = np.load(
            self.directory / "descriptors/flow_hist.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        role_path = self.directory / "descriptors/role.npy"
        self.roles = (
            np.load(role_path, mmap_mode="r", allow_pickle=False)
            if role_path.is_file()
            else None
        )
        self.cluster_metadata = {
            int(c["prototype_id"]): c for c in self.manifest.get("clusters", [])
        }

    def remap(self, path):
        path = Path(path).expanduser()
        for old, new in self.image_roots:
            try:
                mapped = new / path.relative_to(old)
            except ValueError:
                continue
            if mapped.exists():
                return mapped
        return path

    def validate_prototype(self, prototype):
        if not 0 <= prototype < int(self.manifest["num_prototypes"]):
            raise ValueError(
                f'Prototype must be in 0..{self.manifest["num_prototypes"]-1}.'
            )

    def cluster(
        self, prototype, offset=0, limit=40, order="distance", unique_videos=False
    ):
        self.validate_prototype(prototype)
        if offset < 0 or not 1 <= limit <= 200 or order not in ("distance", "node"):
            raise ValueError(
                "Invalid pagination/order (limit 1..200; distance or node)."
            )
        with closing(readonly(self.db_path)) as con:
            summary = con.execute(
                "SELECT * FROM prototypes WHERE id=?", (prototype,)
            ).fetchone()
            if summary is None:
                raise ValueError("Missing prototype record.")
            summary = dict(summary)
            summary.update(self.cluster_metadata.get(prototype, {}))
            sort = (
                "CASE WHEN p.node_id=? THEN 0 ELSE 1 END,p.distance,p.node_id"
                if order == "distance"
                else "p.node_id"
            )
            params = (
                [prototype]
                + ([summary["medoid_node_id"]] if order == "distance" else [])
                + [limit, offset]
            )
            if unique_videos:
                ranking = (
                    "CASE WHEN p.node_id=? THEN 0 ELSE 1 END,p.distance,p.node_id"
                    if order == "distance"
                    else "p.node_id"
                )
                rank_params = [summary["medoid_node_id"]] if order == "distance" else []
                cte = f"""WITH candidates AS (SELECT p.*,f.video,f.target_name,f.window_start,f.target_index,
                    ROW_NUMBER() OVER (PARTITION BY f.dataset,f.video ORDER BY {ranking}) AS rn
                    FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.prototype_id=?)"""
                sort_outer = (
                    "CASE WHEN node_id=? THEN 0 ELSE 1 END,distance,node_id"
                    if order == "distance"
                    else "node_id"
                )
                tail = ([summary["medoid_node_id"]] if order == "distance" else []) + [
                    limit,
                    offset,
                ]
                rows = con.execute(
                    cte
                    + f" SELECT * FROM candidates WHERE rn=1 ORDER BY {sort_outer} LIMIT ? OFFSET ?",
                    rank_params + [prototype] + tail,
                ).fetchall()
                total = con.execute(
                    "SELECT COUNT(DISTINCT f.video) FROM patches p JOIN frames f ON f.id=p.frame_id WHERE p.prototype_id=?",
                    (prototype,),
                ).fetchone()[0]
            else:
                rows = con.execute(
                    f"""SELECT p.*,f.video,f.target_name,f.window_start,f.target_index
                    FROM patches p JOIN frames f ON f.id=p.frame_id
                    WHERE p.prototype_id=? ORDER BY {sort} LIMIT ? OFFSET ?""",
                    params,
                ).fetchall()
                total = summary["patch_count"]
        patches = []
        for row in rows:
            p = dict(row)
            p["is_medoid"] = p["node_id"] == summary["medoid_node_id"]
            p["histogram"] = (
                np.asarray(self.hist[p["descriptor_index"]]).astype(float).tolist()
            )
            p["role"] = (
                ("invalid", "static", "moving")[int(self.roles[p["descriptor_index"]])]
                if self.roles is not None
                else ("legacy" if p["flow_valid"] else "invalid")
            )
            patches.append(p)
        return dict(
            prototype_id=prototype,
            offset=offset,
            limit=limit,
            total=total,
            summary=summary,
            patches=patches,
        )

    @lru_cache(maxsize=256)
    def patch(self, node):
        with closing(readonly(self.db_path)) as con:
            row = con.execute(
                "SELECT * FROM patches WHERE node_id=?", (node,)
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown patch node {node}.")
        return dict(row)

    @lru_cache(maxsize=32)
    def frame(self, frame_id):
        with closing(readonly(self.db_path)) as con:
            row = con.execute("SELECT * FROM frames WHERE id=?", (frame_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown source frame.")
        return dict(row)

    @lru_cache(maxsize=48)
    def flow_fields(self, frame_id, gap):
        if gap not in self.gaps:
            raise ValueError(f"Gap must be one of {self.gaps}.")
        frame = self.frame(frame_id)
        model = json.loads(frame["global_model_json"])
        if self.v2:
            models, keys = model["pairs"], model["pair_keys"]
            if len(models) != len(keys):
                raise ValueError("Flow pair provenance has inconsistent lengths.")
            matches = [
                (camera, key)
                for camera, key in zip(models, keys)
                if int(camera["frame_gap"]) == gap
            ]
            if len(matches) != 1:
                raise ValueError("Cannot identify the recorded flow pair for this gap.")
            camera, key = matches[0]
        else:
            camera, key = model, frame["pair_sha256"]
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid recorded flow cache key.")
        suffix = ".npz" if self.v2 else ".pt"
        path = self.flow_cache / "entries" / key[:2] / (key + suffix)
        if not path.is_file():
            raise FileNotFoundError(
                "Recorded optical-flow cache entry unavailable. Set --flow-cache-dir to the namespace directory containing manifest.json and entries/. Flow is not recomputed."
            )
        if self.v2:
            with np.load(path, allow_pickle=False) as payload:
                if (
                    str(payload["pair_sha256"]) != key
                    or str(payload["namespace"])
                    != self.manifest["flow_cache_namespace"]
                ):
                    raise ValueError("Cached flow identity mismatch.")
                forward, backward = payload["forward"].astype(np.float32), payload[
                    "backward"
                ].astype(np.float32)
        else:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload["pair_sha256"] != key
                or payload["namespace"] != self.manifest["flow_cache_namespace"]
            ):
                raise ValueError("Cached flow identity mismatch.")
            forward, backward = (
                payload["forward"].float().numpy(),
                payload["backward"].float().numpy(),
            )
        expected = (frame["input_h"], frame["input_w"], 2)
        if any(
            x.shape != expected or not np.isfinite(x).all() for x in (forward, backward)
        ):
            raise ValueError("Invalid cached flow geometry/values.")
        valid = target_validity(
            forward,
            backward,
            float(self.desc.get("forward_backward_absolute", 1)),
            float(self.desc.get("forward_backward_relative", 0.05)),
        )
        residual = (-backward - recorded_camera_field(backward.shape, camera)) / gap
        return residual, valid

    @lru_cache(maxsize=512)
    def arrow_summary(self, node, gap):
        patch = self.patch(node)
        residual, valid = self.flow_fields(patch["frame_id"], gap)
        h, w = valid.shape
        box = (
            patch["col"] * w // patch["grid_w"],
            patch["row"] * h // patch["grid_h"],
            (patch["col"] + 1) * w // patch["grid_w"],
            (patch["row"] + 1) * h // patch["grid_h"],
        )
        sl = np.s_[box[1] : box[3], box[0] : box[2]]
        return summarize_patch_motion(
            residual[sl],
            valid[sl],
            self.threshold,
            float(self.desc.get("min_flow_valid_fraction", 0.2)),
            float(self.desc.get("static_patch_fraction", 0.9)),
        )

    @lru_cache(maxsize=512)
    def compass_data(self, node):
        patch = self.patch(node)
        samples = []
        for gap in self.gaps:
            residual, valid = self.flow_fields(patch["frame_id"], gap)
            h, w = valid.shape
            sl = np.s_[
                patch["row"]
                * h
                // patch["grid_h"] : (patch["row"] + 1)
                * h
                // patch["grid_h"],
                patch["col"]
                * w
                // patch["grid_w"] : (patch["col"] + 1)
                * w
                // patch["grid_w"],
            ]
            samples.append((residual[sl], valid[sl]))
        return compass_flow_stats(
            samples,
            self.threshold,
            float(self.desc.get("min_flow_valid_fraction", 0.2)),
            float(self.desc.get("min_valid_gap_fraction", 0.5)),
        )

    def compass(self, node):
        return compass_svg(self.compass_data(node), self.scale, self.min_arrow_fraction)

    def static_bar(self, node):
        return static_bar_svg(self.compass_data(node))

    @lru_cache(maxsize=256)
    def local_arrows(self, node, gap):
        patch = self.patch(node)
        residual, valid = self.flow_fields(patch["frame_id"], gap)
        h, w = valid.shape
        sl = np.s_[
            patch["row"]
            * h
            // patch["grid_h"] : (patch["row"] + 1)
            * h
            // patch["grid_h"],
            patch["col"]
            * w
            // patch["grid_w"] : (patch["col"] + 1)
            * w
            // patch["grid_w"],
        ]
        return local_flow_svg(
            residual[sl],
            valid[sl],
            self.threshold,
            self.scale,
            min_valid=float(self.desc.get("min_flow_valid_fraction", 0.2)),
        )

    def arrow_svg(self, node, gap):
        return centered_arrow_svg(self.arrow_summary(node, gap), self.scale)

    def ensure_clip_store(self):
        if self.clip_store is None:
            self.clip_store = PatchClipStore(
                self.directory, image_roots=self.image_root_options, **self.clip_options
            )
        return self.clip_store

    def patch_clip(self, node, gap=None):
        gap = self.default_gap if gap is None else int(gap)
        try:
            path = self.ensure_clip_store().clip(node, gap)
            self.clip_errors.pop((node, gap), None)
            return path
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self.clip_errors[(node, gap)] = str(exc)
            raise

    def endpoint_image(self, node, gap, which):
        return self.ensure_clip_store().endpoint(node, gap, which)

    def magnitude_image(self, node, gap):
        patch = self.patch(node)
        residual, valid = self.flow_fields(patch["frame_id"], gap)
        h, w = valid.shape
        sl = np.s_[
            patch["row"]
            * h
            // patch["grid_h"] : (patch["row"] + 1)
            * h
            // patch["grid_h"],
            patch["col"]
            * w
            // patch["grid_w"] : (patch["col"] + 1)
            * w
            // patch["grid_w"],
        ]
        rgb = magnitude_rgb(residual[sl], valid[sl], self.scale, self.threshold)
        return encode_png(
            Image.fromarray(rgb).resize((192, 192), Image.Resampling.BILINEAR)
        )

    def public_manifest(self):
        return dict(
            num_prototypes=self.manifest["num_prototypes"],
            num_patches=self.manifest["num_patches"],
            num_videos=self.manifest.get("num_videos"),
            gaps=self.gaps,
            default_gap=self.default_gap,
            min_arrow_fraction=self.min_arrow_fraction,
            magnitude_ticks=np.expm1(np.linspace(0, np.log1p(self.scale), 5)).tolist(),
            scale=self.scale,
            threshold=self.threshold,
            flow_only=self.v2,
            cache_available=self.flow_cache.is_dir(),
            preview_fps=self.clip_options.get("fps", 30),
            slow_motion=self.clip_options.get("slow_motion", 0.25),
            blink_ms=self.blink_ms,
        )

    def magnitude_legend(self):
        speed = np.expm1(np.linspace(0, np.log1p(self.scale), 256))[None, :]
        rgb = magnitude_colors(np.repeat(speed, 18, axis=0), self.scale)
        return encode_png(Image.fromarray(rgb))


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Temporal flow clusters</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17263b;background:#f3f6fa}*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0}
header{background:white;border-bottom:1px solid #dae2ed;padding:22px 30px}h1{margin:0 0 8px;font-size:25px}p{line-height:1.5;margin:8px 0}.muted{color:#5c6c81;font-size:12px}
.controls{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin-top:16px}label{display:grid;gap:5px;font-size:12px;font-weight:600}input,select,button{font:inherit;padding:8px 10px;border:1px solid #bbc9da;border-radius:7px;background:white;color:inherit}input[type=number]{width:110px}.check{display:flex;align-items:center;align-self:center;gap:5px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.primary{background:#214e78;color:white}
main{padding:22px 30px;max-width:1900px;margin:auto}.layout{display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:22px}.toolbar{display:flex;justify-content:space-between;gap:12px;margin-bottom:16px;flex-wrap:wrap}.pager{display:flex;gap:8px;align-items:center;font-size:12px}.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:12px}
.card{background:white;border:1px solid #d7e1ed;border-radius:9px;overflow:hidden;min-width:0}.card.medoid{border:2px solid #d2922b}.inspect{display:block;width:100%;padding:0;border:0;background:white;border-radius:0;text-align:left}.inspect:focus-visible{outline:3px solid #214e78;outline-offset:-3px}
.motion-pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:6px;background:white;padding:4px}.motion-pair img{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain;background:white;min-width:0}
.clip-stage{position:relative;width:100%;aspect-ratio:1;background:#111827}.patch-clip{display:block;width:100%;height:100%;object-fit:contain;background:#111827}.endpoint-pair{position:absolute;inset:0}.endpoint-pair img{position:absolute;inset:0;display:block;width:100%;height:100%;object-fit:contain}.clip-tools{display:flex;gap:5px;justify-content:center;flex-wrap:wrap;padding:7px 5px 0}.clip-tools button{font-size:10px;padding:5px 7px}.clip-tools button.active{background:#e6eef8;border-color:#607d9f;color:#183e66}.clip-meta{height:22px;text-align:center;font-size:10px;color:#5c6c81;line-height:22px;white-space:nowrap}.clip-error{margin:0;padding:10px;font-size:11px;line-height:1.5;color:#a02b36;background:#fff1f0}.caption{padding:9px;font-size:11px;line-height:1.6}.caption strong{display:block;font-size:12px}.line{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tag{display:inline-block;background:#fff0cc;color:#835100;padding:2px 4px;border-radius:4px}.detail-button{margin-top:8px;padding:5px 8px;font-size:11px}
.flow-charts{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,.32fr);gap:8px;align-items:start;width:100%;max-width:266px;margin:10px auto 0}.compass{display:block;width:100%;height:auto;aspect-ratio:240/286;object-fit:contain;background:white;min-width:0}.static-bar{display:block;width:100%;height:auto;aspect-ratio:76/286;object-fit:contain;background:white;min-width:0}.chart-label{text-align:center;font-size:9px;color:#5c6c81;margin-top:2px}
.failed{height:100px;aspect-ratio:auto;font-size:11px;color:#a02b36}.error{color:#a02b36}aside{background:white;padding:17px;border:1px solid #d7e1ed;border-radius:10px;height:fit-content;position:sticky;top:16px}aside h2{font-size:19px;margin:0 0 10px}aside h3{font-size:13px}.reference{width:100%;max-height:180px;object-fit:contain;background:#edf2f8}dl{margin:0}dt{font-size:11px;color:#65758a;margin-top:12px}dd{margin:3px 0 0;font-size:16px;font-weight:600}.note{border-top:1px solid #dbe3ed;padding-top:12px;margin-top:14px}.badge{padding:5px 8px;background:#e8eef7;border-radius:5px;font-size:11px;display:inline-block}.warning{background:#fff1d4;padding:10px;border-radius:6px;font-size:12px;margin-top:10px}
dialog{width:min(1100px,95vw);border:1px solid #ccd6e3;border-radius:12px;padding:20px;color:inherit}dialog::backdrop{background:#0b1c30aa}.modal-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.modal-head h2{font-size:18px}.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.detail-grid h3{font-size:13px}.detail-grid .patch-clip{max-height:500px}#status{font-size:13px}#empty{padding:25px;background:white}
@media(max-width:1100px){.grid{grid-template-columns:repeat(4,minmax(0,1fr))}.layout{grid-template-columns:minmax(0,1fr) 230px}}@media(max-width:700px){header,main{padding:18px}.layout{display:flex;flex-direction:column}aside{position:static;order:-1}.grid{grid-template-columns:repeat(3,minmax(0,1fr))}.detail-grid{grid-template-columns:1fr}}@media(max-width:450px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body>
<header><h1>Temporal flow clusters</h1><p class="muted">Compare local flow arrows, mean direction, and the exact matching flow interval. Clips loop in slow motion; Blink endpoints alternates the previous and target frames. Arrow colors show speed. The compass and static bar pool the usable training gaps.</p>
<form id="controls" class="controls">
<label>Prototype / cluster<input id="prototype" type="number" min="0" step="1" required value="__PROTOTYPE__"></label>
<label>Frame gap<select id="gap"></select></label>
<label>Order<select id="order"><option value="distance">Nearest reference first</option><option value="node">Source node ID</option></select></label>
<label>Per page<input id="limit" type="number" min="1" max="200" step="1" value="__LIMIT__"></label>
<label class="check"><input id="unique" type="checkbox">One patch per video</label>
<label class="check"><input id="autoplay" type="checkbox" checked>Autoplay loops</label>
<button class="primary" type="submit">Load cluster</button></form><div id="warning" class="warning" hidden></div></header>
<main><div class="toolbar"><span id="status" role="status" aria-live="polite">Loading…</span><div class="pager"><button id="previous" disabled>Previous</button><span id="page"></span><button id="next" disabled>Next</button></div></div>
<div class="layout"><section><div id="grid" class="grid" style="--cols:__COLUMNS__"></div><p id="empty" hidden>No members on this page.</p></section>
<aside><h2 id="summaryTitle">Reference</h2><div id="summary"></div><section class="note"><h3>Magnitude heatmap · reference patch</h3><img id="magnitudeReference" class="reference" alt="Reference patch movement magnitude"><p class="muted">Movement speed · pixels/frame</p><img src="/magnitude-scale.png" style="display:block;width:100%;height:18px" alt="Shared low-to-high magnitude color scale"><div id="magnitudeTicks" style="display:flex;justify-content:space-between;font-size:9px;margin-top:4px"></div><p class="muted" id="scale"></p><p class="muted">Heatmap: black = near-static, gray = unreliable. Local and mean arrow colors show speed; their display lengths are fixed.</p></section><p class="muted note">Compass plots pool cached flow across usable training gaps. Dark-gray arrows show mean speed in each direction. Nonzero arrows have a minimum display length; zero-flow directions have no arrow. The vertical static bar excludes unreliable pixels.</p><p class="muted" id="playbackInfo"></p></aside></div></main>
<dialog id="detail"><div class="modal-head"><h2 id="detailTitle"></h2><button id="close">Close</button></div><div class="detail-grid"><section><h3>Local and mean flow · selected gap</h3><div id="detailMotionPair"></div></section><section><h3>Flow interval · slow loop / endpoint blink</h3><div id="detailClip"></div></section></div><p id="detailInfo" class="muted"></p></dialog>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const score=x=>x==null?'unavailable':Number(x).toFixed(4),count=x=>Number(x).toLocaleString();
let manifest=null,result=null,offset=0,requestId=0;
async function json(url){const r=await fetch(url),d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d;}
function flowQuery(node){return new URLSearchParams({node,gap:$('gap').value});}
function motionPair(node){const q=flowQuery(node);return `<div class="motion-pair"><img loading="lazy" src="/local-flow.svg?${q}" alt="Smooth local flow arrows"><img loading="lazy" src="/arrow.svg?${q}" alt="Mean flow direction"></div>`;}
function compass(node){return `<div class="flow-charts"><img class="compass" loading="lazy" src="/compass.svg?node=${node}" alt="Pooled compass flow magnitudes"><img class="static-bar" loading="lazy" src="/static.svg?node=${node}" alt="Vertical static mass bar"></div>`;}
function clipHTML(p){const gap=Number($('gap').value),start=p.target_index-gap;return `<div class="clip-widget" data-node="${p.node_id}" data-gap="${gap}" data-start="${start}" data-end="${p.target_index}" data-mode="loop"><div class="clip-stage"><video class="patch-clip" data-src="/clip?${flowQuery(p.node_id)}" controls loop muted playsinline preload="none" aria-label="Slow-motion frames ${start} to ${p.target_index}"></video><div class="endpoint-pair" hidden><img data-endpoint="previous" alt="Previous flow endpoint"><img data-endpoint="target" alt="Target flow endpoint"></div></div><p class="clip-error" hidden></p><div class="clip-tools"><button type="button" data-action="loop" class="active" aria-pressed="true">Slow loop</button><button type="button" data-action="blink" aria-pressed="false">Blink endpoints</button></div><div class="clip-meta">Frames ${start}–${p.target_index} · ${manifest.slow_motion}×</div></div>`;}
function bindImageErrors(root){root.querySelectorAll('img:not([data-endpoint])').forEach(img=>img.onerror=()=>{img.classList.add('failed');img.alt='Flow image unavailable: check the recorded flow-cache path.';});}
function play(video){video.muted=true;const pending=video.play();if(pending&&pending.catch)pending.catch(()=>{});}
function loadVideo(video){if(!video.getAttribute('src')){video.src=video.dataset.src;video.load();}}
function stopBlink(w){if(w._blinkTimer){clearInterval(w._blinkTimer);w._blinkTimer=null;}}
function showEndpoint(w,index){w.querySelectorAll('[data-endpoint]').forEach((img,i)=>img.style.visibility=i===index?'visible':'hidden');w.querySelector('.clip-meta').textContent=(index===0?'PREVIOUS · frame ':'TARGET · frame ')+(index===0?w.dataset.start:w.dataset.end);}
function startBlink(w){stopBlink(w);if(w.dataset.mode!=='blink'||!w._endpointsReady||!w._inViewport||document.hidden)return;w._blinkIndex=0;showEndpoint(w,0);w._blinkTimer=setInterval(()=>{w._blinkIndex=1-w._blinkIndex;showEndpoint(w,w._blinkIndex);},manifest.blink_ms);}
async function prepareEndpoints(w){if(w._endpointsReady)return;const images=Array.from(w.querySelectorAll('[data-endpoint]'));images.forEach(img=>{if(!img.getAttribute('src'))img.src='/endpoint.png?'+new URLSearchParams({node:w.dataset.node,gap:w.dataset.gap,which:img.dataset.endpoint});});await Promise.all(images.map(img=>img.decode()));w._endpointsReady=true;}
function widgetError(w,message){if(w.dataset.released==='1'||!w.isConnected)return;const box=w.querySelector('.clip-error');box.hidden=false;box.textContent=message;}
async function setClipMode(w,mode){stopBlink(w);w.dataset.mode=mode;const v=w.querySelector('video');w.querySelectorAll('[data-action]').forEach(b=>{b.classList.toggle('active',b.dataset.action===mode);b.setAttribute('aria-pressed',String(b.dataset.action===mode));});v.hidden=mode==='blink';w.querySelector('.endpoint-pair').hidden=mode!=='blink';if(mode==='blink'){v.pause();showEndpoint(w,0);try{await prepareEndpoints(w);startBlink(w);}catch(e){widgetError(w,'Endpoint images unavailable. Check the original RGB frame paths.');}}else{w.querySelector('.clip-meta').textContent=`Frames ${w.dataset.start}–${w.dataset.end} · ${manifest.slow_motion}×`;loadVideo(v);if(!document.hidden)play(v);}}
function activateWidget(w){if(w.dataset.mode==='blink'){if(w._endpointsReady)startBlink(w);}else{const v=w.querySelector('video');loadVideo(v);if($('autoplay').checked&&!document.hidden)play(v);}}
const clipObserver='IntersectionObserver' in window?new IntersectionObserver(entries=>{for(const e of entries){const w=e.target;w._inViewport=e.isIntersecting;if(e.isIntersecting)activateWidget(w);else{w.querySelector('video').pause();stopBlink(w);}}},{rootMargin:'0px',threshold:.05}):null;
function bindVideos(root){root.querySelectorAll('.clip-widget').forEach(w=>{w.dataset.released='0';const v=w.querySelector('video');v.muted=true;v.addEventListener('error',async()=>{let error='Clip unavailable. Check the source frame paths and FFmpeg.';try{const d=await json('/api/clip-status?'+new URLSearchParams({node:w.dataset.node,gap:w.dataset.gap}));error=d.error||error;}catch{}widgetError(w,error);},{once:true});w.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>setClipMode(w,b.dataset.action));if(clipObserver)clipObserver.observe(w);else{w._inViewport=true;activateWidget(w);}});}
function releaseVideos(root){root.querySelectorAll('.clip-widget').forEach(w=>{w.dataset.released='1';if(clipObserver)clipObserver.unobserve(w);stopBlink(w);const v=w.querySelector('video');v.pause();v.removeAttribute('src');v.load();w.querySelectorAll('[data-endpoint]').forEach(img=>img.removeAttribute('src'));});}
function updateReference(){const s=result.summary;$('summaryTitle').textContent='Prototype '+s.id;$('summary').innerHTML=`<span class="badge">${esc(s.role||'Legacy temporal')}</span>${motionPair(s.medoid_node_id)}<p class="muted">Medoid node ${s.medoid_node_id}</p><dl><dt>Assigned patches / videos</dt><dd>${count(s.patch_count)} / ${count(s.video_count)}</dd><dt>Mean distance to medoid</dt><dd>${score(s.mean_distance)}</dd><dt>Audit silhouette</dt><dd>${score(s.silhouette_audit)}</dd></dl>`;$('magnitudeReference').src='/magnitude.png?'+flowQuery(s.medoid_node_id);bindImageErrors($('summary'));}
function render(){if(!result)return;releaseVideos($('grid'));const d=result;
$('grid').innerHTML=d.patches.map((p,i)=>`<article class="card${p.is_medoid?' medoid':''}"><button class="inspect" type="button" data-index="${i}" aria-label="Inspect patch ${p.node_id}">${motionPair(p.node_id)}</button>${clipHTML(p)}<div class="caption"><strong>Patch ${p.node_id} · ${esc(p.role)}</strong><span class="line" title="${esc(p.video+' / '+p.target_name)}">${esc(p.video)} / ${esc(p.target_name)}</span><span class="line">Cell ${p.row},${p.col} · gap ${$('gap').value}</span><span class="line">Distance ${score(p.distance)}</span><span class="line">Valid ${(p.flow_valid_fraction*100).toFixed(0)}% · pooled speed ${Number(p.flow_magnitude/(manifest.flow_only?1:manifest.gaps[0])).toFixed(3)} px/frame</span>${compass(p.node_id)}<div class="chart-label">Pooled across training gaps</div>${p.is_medoid?'<span class="tag">REFERENCE MEDOID</span>':''}<button class="detail-button" type="button" data-index="${i}">Inspect patch</button></div></article>`).join('');
$('grid').querySelectorAll('[data-index]').forEach(b=>b.onclick=()=>showPatch(d.patches[Number(b.dataset.index)]));bindImageErrors($('grid'));bindVideos($('grid'));$('empty').hidden=d.patches.length>0;updateReference();}
function showPatch(p){$('detailTitle').textContent=`Patch ${p.node_id} · prototype ${p.prototype_id}`;$('detailMotionPair').innerHTML=motionPair(p.node_id);releaseVideos($('detailClip'));$('detailClip').innerHTML=clipHTML(p);$('detailInfo').textContent=`${p.video} / ${p.target_name} · frames ${p.target_index-Number($('gap').value)}–${p.target_index} · cell ${p.row},${p.col} · distance ${score(p.distance)}. Slow loop shows only the selected flow interval at ${manifest.slow_motion}× speed. Blink endpoints alternates the original endpoint crops every ${manifest.blink_ms} ms. The RGB crop is not camera-stabilized.`;$('detail').showModal();bindImageErrors($('detail'));bindVideos($('detailClip'));}
async function load(){const id=Number($('prototype').value),request=++requestId;try{if(!Number.isInteger(id)||id<0||id>=manifest.num_prototypes)throw Error(`Prototype must be in 0..${manifest.num_prototypes-1}.`);$('status').className='';$('status').textContent='Loading cluster…';const d=await json('/api/cluster?'+new URLSearchParams({prototype:id,offset,limit:$('limit').value,order:$('order').value,unique:$('unique').checked?'1':'0'}));if(request!==requestId)return;result=d;$('status').textContent=`Prototype ${id} · ${count(d.total)} patches · showing ${d.patches.length?d.offset+1:0}–${d.offset+d.patches.length}`;$('page').textContent=`Page ${Math.floor(d.offset/d.limit)+1} / ${Math.max(1,Math.ceil(d.total/d.limit))}`;$('previous').disabled=d.offset===0;$('next').disabled=d.offset+d.patches.length>=d.total;render();}catch(e){if(request===requestId){$('status').className='error';$('status').textContent=e.message;}}}
$('controls').onsubmit=e=>{e.preventDefault();offset=0;load();};$('unique').onchange=()=>{offset=0;load();};$('gap').onchange=render;$('previous').onclick=()=>{offset=Math.max(0,result.offset-result.limit);load();};$('next').onclick=()=>{offset=result.offset+result.limit;load();};$('close').onclick=()=>$('detail').close();$('detail').addEventListener('close',()=>releaseVideos($('detailClip')));
$('autoplay').onchange=()=>document.querySelectorAll('.clip-widget').forEach(w=>{const v=w.querySelector('video');if(w.dataset.mode==='loop'){if($('autoplay').checked&&w._inViewport&&!document.hidden){loadVideo(v);play(v);}else v.pause();}});document.addEventListener('visibilitychange',()=>{document.querySelectorAll('.clip-widget').forEach(w=>{if(document.hidden){w.querySelector('video').pause();stopBlink(w);}else if(w._inViewport&&w.dataset.released!=='1')activateWidget(w);});});
json('/api/manifest').then(d=>{manifest=d;$('prototype').max=d.num_prototypes-1;$('gap').innerHTML=d.gaps.map(g=>`<option value="${g}" ${g===d.default_gap?'selected':''}>${g} frames</option>`).join('');$('scale').textContent=`Shared log-speed scale: 0–${d.scale.toFixed(3)} px/frame. Static threshold: ${d.threshold.toFixed(3)} px/frame.`;$('magnitudeTicks').innerHTML=d.magnitude_ticks.map((x,i)=>`<span>${i===4?'≥':''}${Number(x).toFixed(2)}</span>`).join('');$('playbackInfo').textContent=`Source: ${d.preview_fps} FPS (assumed). Loop: ${d.slow_motion}× speed. Blink: ${d.blink_ms} ms per endpoint. Only the selected flow interval is shown.`;let warnings=[];if(!d.flow_only)warnings.push('Legacy bank: feature changes may also have contributed to clustering.');if(!d.cache_available)warnings.push('Flow cache unavailable. Set --flow-cache-dir to the recorded namespace directory.');if(warnings.length){$('warning').hidden=false;$('warning').textContent=warnings.join(' ');}load();}).catch(e=>{$('status').textContent=e.message;$('status').className='error';});

</script></body></html>"""


def make_server(store, port=8768, prototype=0, columns=8, top_k=12):
    store.validate_prototype(prototype)
    html = (
        HTML.replace("__PROTOTYPE__", str(prototype))
        .replace("__COLUMNS__", str(columns))
        .replace("__LIMIT__", str(top_k))
        .encode()
    )

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content, kind, headers=()):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        def video(self, path, download, node):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            if self.headers.get("Range"):
                try:
                    start, end = byte_range(self.headers["Range"], size)
                    status = 206
                except ValueError:
                    self.reply(
                        416, b"", "video/mp4", [("Content-Range", f"bytes */{size}")]
                    )
                    return
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "private, max-age=3600")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="temporal_patch_{node}.mp4"',
                )
            self.end_headers()
            if self.command == "HEAD":
                return
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = stream.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = parse_qs(url.query)
                if url.path in ("/", "/index.html"):
                    self.reply(200, html, "text/html; charset=utf-8")
                    return
                if url.path == "/static.svg":
                    self.reply(
                        200,
                        store.static_bar(int(q.get("node", ["-1"])[0])),
                        "image/svg+xml; charset=utf-8",
                    )
                    return
                if url.path == "/compass.svg":
                    self.reply(
                        200,
                        store.compass(int(q.get("node", ["-1"])[0])),
                        "image/svg+xml; charset=utf-8",
                    )
                    return
                if url.path == "/local-flow.svg":
                    self.reply(
                        200,
                        store.local_arrows(
                            int(q.get("node", ["-1"])[0]),
                            int(q.get("gap", [str(store.default_gap)])[0]),
                        ),
                        "image/svg+xml; charset=utf-8",
                    )
                    return
                if url.path == "/arrow.svg":
                    self.reply(
                        200,
                        store.arrow_svg(
                            int(q.get("node", ["-1"])[0]),
                            int(q.get("gap", [str(store.default_gap)])[0]),
                        ),
                        "image/svg+xml; charset=utf-8",
                    )
                    return
                if url.path == "/magnitude-scale.png":
                    self.reply(200, store.magnitude_legend(), "image/png")
                    return
                if url.path == "/api/manifest":
                    payload = store.public_manifest()
                elif url.path == "/api/cluster":
                    payload = store.cluster(
                        int(q.get("prototype", ["0"])[0]),
                        int(q.get("offset", ["0"])[0]),
                        int(q.get("limit", ["40"])[0]),
                        q.get("order", ["distance"])[0],
                        q.get("unique", ["0"])[0] == "1",
                    )
                elif url.path == "/clip":
                    node = int(q.get("node", ["-1"])[0])
                    self.video(
                        store.patch_clip(
                            node, int(q.get("gap", [str(store.default_gap)])[0])
                        ),
                        q.get("download", ["0"])[0] == "1",
                        node,
                    )
                    return
                elif url.path == "/endpoint.png":
                    self.reply(
                        200,
                        store.endpoint_image(
                            int(q.get("node", ["-1"])[0]),
                            int(q.get("gap", [str(store.default_gap)])[0]),
                            q.get("which", ["previous"])[0],
                        ),
                        "image/png",
                    )
                    return
                elif url.path == "/magnitude.png":
                    self.reply(
                        200,
                        store.magnitude_image(
                            int(q.get("node", ["-1"])[0]),
                            int(q.get("gap", [str(store.default_gap)])[0]),
                        ),
                        "image/png",
                    )
                    return
                elif url.path == "/api/clip-status":
                    node = int(q.get("node", ["-1"])[0])
                    store.patch(node)
                    payload = {
                        "error": store.clip_errors.get(
                            (node, int(q.get("gap", [str(store.default_gap)])[0])),
                            "The clip could not be played. Check the source-frame paths and FFmpeg H.264 support.",
                        )
                    }
                elif url.path == "/favicon.ico":
                    self.reply(204, b"", "image/x-icon")
                    return
                else:
                    self.reply(404, b'{"error":"Not found"}', "application/json")
                    return
                self.reply(
                    200,
                    json.dumps(payload, allow_nan=False).encode(),
                    "application/json; charset=utf-8",
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (
                ValueError,
                KeyError,
                OSError,
                RuntimeError,
                ImportError,
                sqlite3.Error,
                cv2.error,
                subprocess.TimeoutExpired,
            ) as exc:
                print(f"Viewer request failed: {exc}", file=sys.stderr)
                try:
                    self.reply(
                        400,
                        json.dumps({"error": str(exc)}).encode(),
                        "application/json",
                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith(
                ("GET /magnitude", "GET /local-flow", "GET /arrow", "GET /clip")
            ):
                return
            super().log_message(fmt, *args)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    p.add_argument("--prototype", type=int, default=0)
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--columns", type=int, default=8)
    p.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Initial nearest members per page, 1..200.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Assumed original/source FPS; diagnostic playback multiplies this by --slow-motion.",
    )
    p.add_argument(
        "--clip-frames",
        type=int,
        default=0,
        help="Deprecated: diagnostic clips always use the selected flow gap.",
    )
    p.add_argument(
        "--slow-motion",
        type=float,
        default=0.25,
        help="Playback factor relative to --fps, default .25 (4x slower).",
    )
    p.add_argument(
        "--blink-ms",
        type=int,
        default=500,
        help="Hold each endpoint for this many milliseconds when blinking.",
    )
    p.add_argument("--cache-dir", type=Path, help="Separate generated RGB MP4 cache.")
    p.add_argument(
        "--ffmpeg", default="ffmpeg", help="System FFmpeg for the patch clips."
    )
    p.add_argument(
        "--flow-cache-dir",
        type=Path,
        help="Exact namespace directory containing entries/ and manifest.json; optional relocation override.",
    )
    p.add_argument(
        "--flow-scale",
        type=float,
        help="Shared saturation speed, in canvas pixels/frame. Default: recorded training p95.",
    )
    p.add_argument(
        "--min-arrow-fraction",
        type=float,
        default=0.25,
        help="Minimum displayed arrow length as fraction of compass radius, 0..1. Default .25; zero flow is never drawn.",
    )
    p.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="OLD_ROOT=NEW_ROOT",
        help="Remap dataset or flow-cache paths after moving them.",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=2,
        help="OpenCV CPU threads (no flow estimation is performed).",
    )
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.columns <= 16 or args.threads < 1:
        p.error("Port must be in 1..65535, columns in 1..16, and threads positive.")
    if (
        not 1 <= args.top_k <= 200
        or not math.isfinite(args.fps)
        or not 1 <= args.fps <= 120
        or not math.isfinite(args.slow_motion)
        or not 0 < args.slow_motion <= 1
        or not 100 <= args.blink_ms <= 5000
    ):
        p.error(
            "Require top-k 1..200, source FPS 1..120, slow-motion factor (0,1], and blink-ms 100..5000."
        )
    try:
        cv2.setNumThreads(args.threads)
        store = ClusterStore(
            args.cluster_dir,
            args.image_root,
            args.flow_cache_dir,
            args.flow_scale,
            args.min_arrow_fraction,
        )
        store.clip_options = dict(
            fps=args.fps,
            clip_frames=args.clip_frames,
            slow_motion=args.slow_motion,
            cache_dir=args.cache_dir,
            ffmpeg=args.ffmpeg,
        )
        store.blink_ms = args.blink_ms
        # Clips are part of the only layout: fail early if FFmpeg/cache setup is invalid.
        store.ensure_clip_store()
        server = make_server(store, args.port, args.prototype, args.columns, args.top_k)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_port}"
    print(
        f"Open {url} — optical flow only uses existing caches; stop with Ctrl+C.",
        flush=True,
    )
    if not args.no_browser:
        timer = Timer(0.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
