#!/usr/bin/env python3
"""Local patch subgroup curator. Python 3.9+, standard library only.

Run on the machine containing the exported gallery:
    python3 patch_subgroup_curator.py --root /path/to/concepts_v4
Open http://127.0.0.1:8765 . For a remote server use an SSH port forward.
Saved JSON lives in a sibling <gallery-name>_manual_subgroups directory by default.
Source images, patch_metadata.json, and model weights are never modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

GROUP_RE = re.compile(r"group_\d+\Z")
SAVE_RE = re.compile(r"subgroup_\d{4,}\.json\Z")
DEFAULT_ACTIVATION_FREQ_DIR = (
    Path(__file__).resolve().parents[3] / "prototype_activation_freq_outputs"
)


class Problem(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise Problem(f"{name} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        raise Problem(f"{name} is outside its allowed range.")
    return value


def top_matches(patch, k):
    # Same ranking rule as the supplied top_matches(): stable descending cosine.
    return sorted(patch["top_k_cosine_matches"],
                  key=lambda match: match["cosine_similarity"], reverse=True)[:k]


def overlap_report(patches, k):
    """All-selected intersection, frequency membership, cosines, and pairwise overlaps.

    Never enumerate the powerset: its size grows exponentially with selections.
    IDs in intersections follow the first selected patch's cosine rank order.
    """
    selected = [{"group_patch_index": p["group_patch_index"],
                 "available_match_count": len(p["top_k_cosine_matches"]),
                 "top_k_cosine_matches": copy.deepcopy(top_matches(p, k))}
                for p in patches]
    lists = [[m["prototype_index"] for m in p["top_k_cosine_matches"]] for p in selected]
    sets = [set(ids) for ids in lists]
    shared = [pid for pid in lists[0] if all(pid in s for s in sets[1:])] if lists else []
    support = {}
    for p in selected:
        for match in p["top_k_cosine_matches"]:
            pid = match["prototype_index"]
            entry = support.setdefault(pid, {"prototype_index": pid, "patch_count": 0,
                                            "patch_ids": [], "cosine_by_patch": {}})
            entry["patch_count"] += 1
            entry["patch_ids"].append(p["group_patch_index"])
            entry["cosine_by_patch"][str(p["group_patch_index"])] = match["cosine_similarity"]
    pairwise = []
    for i, a in enumerate(selected):
        for j in range(i + 1, len(selected)):
            b = selected[j]
            ids = [pid for pid in lists[i] if pid in sets[j]]
            union = sets[i] | sets[j]
            pairwise.append({"patch_a": a["group_patch_index"], "patch_b": b["group_patch_index"],
                             "shared_count": len(ids), "shared_prototype_ids": ids,
                             "jaccard": len(ids) / len(union) if union else None})
    return {"requested_top_k": k, "selected_patch_count": len(patches),
            "reference_patch_index": patches[0]["group_patch_index"] if patches else None,
            "shared_by_all_prototype_ids": shared,
            "prototype_support": sorted(support.values(), key=lambda r: (-r["patch_count"], r["prototype_index"])),
            "per_patch": selected, "pairwise": pairwise}


def resolve_activation_freq_csv(path=None):
    """Return the newest prototype_activation_freq_*.csv, or an explicit path."""
    if path is not None:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise Problem(f"Activation-frequency CSV not found: {candidate}")
        return candidate
    folder = DEFAULT_ACTIVATION_FREQ_DIR
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob("prototype_activation_freq_*.csv"))
    return matches[-1] if matches else None


def load_activation_frequencies(csv_path):
    """Parse prototype_activation_freq.py CSV into a prototype-index map."""
    path = Path(csv_path).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "prototype_index" not in reader.fieldnames:
            raise Problem(f"Activation-frequency CSV is missing prototype_index: {path}")
        records = {}
        for row_number, row in enumerate(reader, start=2):
            try:
                prototype_index = int(row["prototype_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise Problem(f"Invalid prototype_index on line {row_number} of {path}: {exc}") from exc

            def number(field, cast=float, default=0):
                raw = row.get(field, "")
                if raw is None or str(raw).strip() == "":
                    return default
                try:
                    return cast(raw)
                except (TypeError, ValueError) as exc:
                    raise Problem(f"Invalid {field} on line {row_number} of {path}: {exc}") from exc

            records[str(prototype_index)] = {
                "prototype_index": prototype_index,
                "train_count": number("train_count", int, 0),
                "train_frequency": number("train_frequency", float, 0.0),
                "val_count": number("val_count", int, 0),
                "val_frequency": number("val_frequency", float, 0.0),
                "combined_count": number("combined_count", int, 0),
                "combined_frequency": number("combined_frequency", float, 0.0),
            }
    return {"source_csv": str(path), "prototypes": records}


class Curator:
    def __init__(self, root, output, activation_freq_csv=None):
        self.root = Path(root).expanduser().resolve()
        self.output = Path(output).expanduser().resolve()
        if not self.root.is_dir():
            raise Problem(f"Gallery directory does not exist: {self.root}")
        if self.output == self.root or self.root in self.output.parents:
            raise Problem("Choose --output-dir outside the gallery root, so gallery re-exports cannot erase subgroups.")
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache = {}
        self.token = secrets.token_urlsafe(32)
        self._process_lock = None
        csv_path = resolve_activation_freq_csv(activation_freq_csv)
        self.activation_frequencies = load_activation_frequencies(csv_path) if csv_path else None

    def acquire_process_lock(self):
        # Serialize writers across server processes on Linux/macOS.
        try:
            import fcntl
        except ImportError:
            raise Problem("Run this server on Linux/macOS (including WSL on Windows).")
        handle = (self.output / ".curator.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise Problem("Another curator is already using this output directory.")
        self._process_lock = handle

    def close(self):
        if self._process_lock:
            self._process_lock.close()
            self._process_lock = None

    def group_dir(self, name):
        if not isinstance(name, str) or not GROUP_RE.fullmatch(name):
            raise Problem("Invalid source group.")
        folder = (self.root / name).resolve()
        if folder.parent != self.root or not folder.is_dir():
            raise Problem("Source group not found.", 404)
        return folder

    def load(self, name):
        path = self.group_dir(name) / "patch_metadata.json"
        if not path.is_file():
            raise Problem(f"{name}/patch_metadata.json is missing.", 404)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            cached = self.cache.get(name)
            if cached and cached[0] == digest:
                return cached
            try:
                patches = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise Problem(f"Cannot parse {name}/patch_metadata.json: {exc}")
            if not isinstance(patches, list):
                raise Problem(f"{name}/patch_metadata.json must contain a list of patch records.")
            seen = set()
            for patch in patches:
                if not isinstance(patch, dict):
                    raise Problem("Each patch metadata entry must be an object.")
                pid = integer(patch.get("group_patch_index"), "group_patch_index")
                if pid in seen:
                    raise Problem(f"Duplicate group_patch_index {pid} in {name}.")
                seen.add(pid)
                matches = patch.get("top_k_cosine_matches")
                if not isinstance(matches, list):
                    raise Problem(f"Patch {pid} is missing top_k_cosine_matches.")
                prototypes = set()
                for match in matches:
                    if not isinstance(match, dict):
                        raise Problem(f"Patch {pid} has an invalid cosine match.")
                    proto = integer(match.get("prototype_index"), "prototype_index")
                    if proto in prototypes:
                        raise Problem(f"Patch {pid} repeats prototype {proto}.")
                    prototypes.add(proto)
                    score = match.get("cosine_similarity")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise Problem(f"Patch {pid} has a nonfinite or invalid cosine similarity.")
            patches.sort(key=lambda p: p["group_patch_index"])
            self.cache[name] = (digest, patches)
            return digest, patches

    def saved(self, name):
        self.group_dir(name)
        records = []
        for path in sorted((self.output / name).glob("subgroup_*.json")):
            if not SAVE_RE.fullmatch(path.name):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(record, dict) or record.get("schema_version") != 1
                        or record.get("source_group") != name or record.get("subgroup_id") != path.stem
                        or not isinstance(record.get("patch_ids"), list)
                        or len(set(record["patch_ids"])) != len(record["patch_ids"])):
                    raise ValueError("Invalid saved subgroup schema")
                for pid in record["patch_ids"]:
                    integer(pid, "saved patch index")
                records.append(record)
            except (ValueError, OSError, Problem) as exc:
                raise Problem(f"Cannot read saved subgroup {path}: {exc}. Restore or move this file before continuing.", 500)
        return records

    @staticmethod
    def saved_summary(record):
        return {key: record.get(key) for key in ("subgroup_id", "name", "description", "created_at",
                "source_group", "source_metadata_sha256", "patch_ids", "requested_top_k",
                "distinct_video_count", "shared_by_all_prototype_ids")}

    def media(self, name, patch):
        # patch_directory is relative to the export root in the supplied code.
        # Fall back to the exporter naming convention for relocated exports.
        folder_name = patch.get("patch_directory") or f"{name}/patch_{patch['group_patch_index']:02d}"
        supplied = Path(folder_name)
        folder = (self.root / supplied).resolve() if not supplied.is_absolute() else supplied.resolve()
        group = self.group_dir(name)
        if not folder.is_dir() or group not in folder.parents:
            folder = (group / f"patch_{patch['group_patch_index']:02d}").resolve()
        if group not in folder.parents:
            raise Problem("Patch directory points outside its source group.")
        def url(filename):
            path = folder / filename
            if not path.is_file():
                return None
            return "/media/" + quote(path.relative_to(self.root).as_posix(), safe="/")
        frame = url("target_frame_boxed.png") or url("target_frame.png") or url("target_frame_native.png")
        return {"frame": frame, "patch": url("patch.png"), "temporal": url("temporal_window.gif")}

    def group(self, name):
        digest, patches = self.load(name)
        with self.lock:
            saved = self.saved(name)
        summaries = [self.saved_summary(r) for r in saved]
        locked = {}
        for r in saved:
            for pid in r["patch_ids"]:
                if str(pid) in locked:
                    raise Problem(f"Patch {pid} occurs in multiple saved subgroups.", 500)
                locked[str(pid)] = r["subgroup_id"]
        items = [{"metadata": p, "media": self.media(name, p)} for p in patches]
        return {"name": name, "metadata_sha256": digest, "patches": items,
                "saved_subgroups": summaries, "locked_patch_ids": locked,
                "stale_saved_subgroups": [r["subgroup_id"] for r in saved
                                          if r.get("source_metadata_sha256") != digest]}

    def groups(self):
        folders = sorted((p for p in self.root.iterdir() if p.is_dir() and GROUP_RE.fullmatch(p.name)),
                         key=lambda p: (int(p.name.split("_")[1]), p.name))
        if not folders:
            raise Problem("No group_<number> folders found in this gallery.")
        results = []
        for folder in folders:
            try:
                _, patches = self.load(folder.name)
                with self.lock:
                    saved = self.saved(folder.name)
                results.append({"name": folder.name, "patch_count": len(patches),
                                "saved_count": len(saved), "error": None})
            except (Problem, OSError) as exc:
                results.append({"name": folder.name, "patch_count": 0, "saved_count": 0, "error": str(exc)})
        return {"groups": results, "root": str(self.root), "output_dir": str(self.output),
                "csrf_token": self.token,
                "activation_frequencies": self.activation_frequencies}

    def form(self, body):
        if not isinstance(body, dict):
            raise Problem("Request must be an object.")
        name = body.get("source_group")
        k = integer(body.get("k"), "Top-k", 1, 100000)
        ids = body.get("patch_ids")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 512:
            raise Problem("Select between 1 and 512 patches.")
        for pid in ids:
            integer(pid, "Patch index")
        if len(set(ids)) != len(ids):
            raise Problem("Duplicate patch selections.")
        label = body.get("name", "")
        description = body.get("description", "")
        if not isinstance(label, str) or len(label) > 160 or not isinstance(description, str) or len(description) > 4000:
            raise Problem("Subgroup name or description is too long or invalid.")
        with self.lock:
            digest, patches = self.load(name)
            if body.get("metadata_sha256") != digest:
                raise Problem("Source metadata changed. Reload the page before forming a subgroup.", 409)
            existing = self.saved(name)
            locked = {pid for record in existing for pid in record["patch_ids"]}
            if locked.intersection(ids):
                raise Problem("Some selected patches are already saved in a subgroup. Reload this group.", 409)
            by_id = {p["group_patch_index"]: p for p in patches}
            if any(pid not in by_id for pid in ids):
                raise Problem("A selected patch is not in this source group.")
            selected = [by_id[pid] for pid in ids]
            report = overlap_report(selected, k)
            folder = self.output / name
            folder.mkdir(parents=True, exist_ok=True)
            sequence = max([int(r["subgroup_id"].split("_")[1]) for r in existing] + [0]) + 1
            sid = f"subgroup_{sequence:04d}"
            record = {"schema_version": 1, "subgroup_id": sid, "source_group": name,
                      "name": label.strip() or sid.replace("_", " ").title(),
                      "description": description.strip(),
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "source_root": str(self.root),
                      "source_metadata_file": f"{name}/patch_metadata.json",
                      "source_metadata_sha256": digest,
                      "ranking_basis": "top_k_cosine_matches, descending cosine_similarity; not inferred forward activation",
                      "patch_ids": ids, "distinct_video_count": len({str(p.get("video_id", "")) for p in selected}),
                      **report,
                      "patches": [{"metadata": copy.deepcopy(p),
                                   "selected_top_k_cosine_matches": copy.deepcopy(top_matches(p, k)),
                                   "effective_top_k": min(k, len(p["top_k_cosine_matches"]))}
                                  for p in selected]}
            destination = folder / (sid + ".json")
            encoded = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
            fd, tempname = tempfile.mkstemp(prefix=".pending_", suffix=".json", dir=folder)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tempname, destination)
            finally:
                if os.path.exists(tempname):
                    os.unlink(tempname)
            return {"subgroup": self.saved_summary(record), "saved_path": str(destination),
                    "download_url": f"/api/download/{name}/{sid}.json"}


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patchwork · Prototype subgroup curator</title>
<style>
:root{--bg:#f5f6f8;--paper:#fff;--ink:#172438;--muted:#68758a;--line:#dfe5ec;--blue:#295be7;--soft:#eef3ff;--green:#167c61;--orange:#b76a25;--radius:12px;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg);font-size:13px}
*{box-sizing:border-box}body{margin:0}button,input,select,textarea{font:inherit}button,a,input,textarea,select{outline-offset:3px}button{cursor:pointer;color:inherit;background:white;border:1px solid var(--line);border-radius:8px;padding:8px 11px;font-weight:550}button:hover:enabled{border-color:#a4b5d7;background:#f5f8ff}button:disabled{cursor:default;opacity:.45}button.primary{background:var(--blue);color:white;border-color:var(--blue)}button.primary:hover:enabled{background:#204dcc}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}.muted{color:var(--muted)}.eyebrow{font-size:10px;letter-spacing:1.8px;text-transform:uppercase;font-weight:700;color:var(--muted)}
header{height:78px;padding:0 25px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);background:white;gap:20px}.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;grid-template-columns:repeat(2,9px);gap:3px;transform:rotate(-6deg)}.mark i{display:block;width:9px;height:9px;border-radius:2px;background:var(--blue)}.mark i:nth-child(2){opacity:.35}.mark i:nth-child(3){opacity:.65}.brand h1{font-size:21px;letter-spacing:-.6px;margin:0}.brand p{margin:3px 0 0;color:var(--muted);font-size:12px}.local{display:flex;align-items:center;gap:7px;color:var(--green);font-size:12px}.local:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green)}
.shell{padding:23px;max-width:2100px;margin:auto}.heading{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:18px}.heading h2{font-size:24px;letter-spacing:-.7px;margin:6px 0}.heading p{margin:0;color:var(--muted);line-height:1.55}.guide{font-size:12px;color:var(--muted);max-width:300px;text-align:right;line-height:1.5}.tabs{display:flex;gap:6px;overflow-x:auto;margin:0 0 20px;padding-bottom:5px}.tab{white-space:nowrap;padding:11px 16px;background:transparent;border-color:transparent;color:var(--muted)}.tab[aria-selected=true]{background:white;color:var(--blue);border-color:var(--line);box-shadow:0 2px 4px #243a6010}.tab small{font-size:10px;margin-left:7px;background:#e9edf4;border-radius:5px;padding:2px 5px}.tab[aria-selected=true] small{background:var(--soft);color:var(--blue)}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:20px;align-items:start}.gallery{min-width:0}.toolbar{background:white;border:1px solid var(--line);border-radius:var(--radius);padding:13px 15px;display:flex;justify-content:space-between;align-items:center;gap:15px;margin-bottom:12px;flex-wrap:wrap}.toolbar strong{font-size:14px}.toolbar .muted{font-size:11px;margin-top:4px}.tools{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.tools label{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted)}input[type=checkbox]{accent-color:var(--blue)}.stage-label{font-size:10px;color:var(--orange);background:#fff4e8;border-radius:5px;padding:3px 6px;margin-left:7px}.grid-nav{display:grid;grid-template-columns:28px minmax(0,1fr) 28px;gap:7px;align-items:start}.group-arrow{position:sticky;top:25px;height:70px;padding:0;font-size:23px;margin-top:45px;color:var(--muted);background:#fff}.grid-scroll{overflow-x:auto;padding:2px}.patch-grid{display:grid;grid-template-columns:repeat(8,minmax(90px,1fr));gap:8px;min-width:800px;align-items:stretch}.card{border:1px solid var(--line);border-radius:9px;background:white;overflow:hidden;position:relative;transition:border-color .12s,box-shadow .12s}.card.selected{border-color:var(--blue);box-shadow:0 0 0 2px #295be720}.card.locked{background:#f0f3f7;border-color:#dbe2eb}.card.locked .media-button{opacity:.62}.card-head{display:flex;justify-content:space-between;align-items:center;padding:7px 7px 6px;gap:3px}.patch-id{font-size:11px;font-weight:700}.pick{width:19px;height:19px;margin:0;appearance:none;border:1.5px solid #b6c1d2;border-radius:50%;background:white;cursor:pointer;display:grid;place-content:center;flex-shrink:0}.pick:checked{border-color:var(--blue);background:var(--blue)}.pick:checked:before{content:"✓";color:white;font-size:12px;font-weight:800}.pick:disabled{background:#e2e8ef;border-color:#c5d0df;cursor:default}.pick:disabled:before{content:"−";color:#7b899b;font-size:13px}.media-button{display:block;width:100%;border:0;padding:0;border-radius:0;background:#edf0f5;position:relative;overflow:hidden}.media-button:hover:enabled{background:#edf0f5;border:0}.media-button img{display:block;width:100%;object-fit:contain;background:#e8ecf2}.frame-img{aspect-ratio:16/9}.patch-img{aspect-ratio:1/1;object-fit:contain!important}.image-caption{display:flex;align-items:center;justify-content:space-between;font-size:9px;color:var(--muted);padding:4px 7px;text-transform:uppercase;letter-spacing:.8px}.target{color:var(--orange);font-weight:650}.patch-wrap{margin:0 6px 6px;border-radius:5px;overflow:hidden;background:#edf0f5}.card-foot{font-size:10px;padding:0 7px 7px;line-height:1.5;color:var(--muted);overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.lock-badge{font-size:9px;color:var(--green);padding:3px 6px;background:#e6f1ec;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.pagination{display:flex;justify-content:center;align-items:center;gap:12px;margin:18px 0}.pagination span{color:var(--muted);font-size:12px}.empty{padding:60px 20px;text-align:center;background:white;border:1px dashed var(--line);border-radius:var(--radius);color:var(--muted);line-height:1.7}.statusline{min-height:20px;font-size:11px;color:var(--muted);margin:9px 38px}.warning{border:1px solid #ebcc9c;background:#fff8ea;color:#946018;padding:10px 12px;border-radius:8px;line-height:1.5;font-size:12px;margin-bottom:12px}.error{border-color:#efc0c0;background:#fff2f2;color:#a63232}.hidden{display:none!important}
.sidebar{position:sticky;top:15px;max-height:calc(100vh - 30px);overflow-y:auto;background:white;border:1px solid var(--line);border-radius:var(--radius);box-shadow:0 3px 16px #243a6005}.panel-head{padding:18px 18px 14px;border-bottom:1px solid var(--line)}.panel-head h3{font-size:16px;margin:5px 0 7px;letter-spacing:-.3px}.panel-head p{margin:0;font-size:11px;color:var(--muted);line-height:1.6}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:15px}.stat{background:var(--bg);padding:10px 8px;border-radius:7px;text-align:center}.stat strong{display:block;font-size:21px;font-weight:650;font-variant-numeric:tabular-nums}.stat span{font-size:10px;color:var(--muted)}.panel-section{padding:15px 18px;border-bottom:1px solid var(--line)}.row{display:flex;align-items:center;justify-content:space-between;gap:8px}.row h4{margin:0;font-size:12px}.k-input{width:65px;border:1px solid var(--line);padding:6px 8px;border-radius:6px}.hint{font-size:11px;line-height:1.6;color:var(--muted);margin:8px 0 0}.chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:11px}.chip{font-size:11px;font-family:ui-monospace,SFMono-Regular,monospace;color:var(--blue);background:var(--soft);padding:4px 7px;border-radius:5px;display:inline-block}.chip.green{background:#e7f5ee;color:var(--green)}.selection-list{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}.remove-patch{font-size:11px;padding:4px 7px;border-radius:5px}.remove-patch span{margin-left:7px;color:#8d9bb0}.support-table{width:100%;border-collapse:collapse;font-size:11px;margin-top:10px}.support-table th{text-align:left;font-size:10px;font-weight:500;color:var(--muted);padding:6px 3px;border-bottom:1px solid var(--line)}.support-table td{padding:6px 3px;border-bottom:1px solid #f0f2f6;font-variant-numeric:tabular-nums}.support-table td:last-child{text-align:right}.bar{width:55px;height:4px;background:#e8edf6;border-radius:3px;margin-top:4px}.bar i{display:block;height:4px;background:var(--blue);border-radius:3px}.scroll-table{max-height:210px;overflow:auto}.shared-detail{font-size:10px;line-height:1.6;padding:6px 0;border-bottom:1px solid #f0f2f6;word-break:break-word}.shared-detail strong{color:var(--blue)}details{margin-top:10px}summary{cursor:pointer;font-size:11px;font-weight:600;color:var(--muted);padding:4px 0}.per-patch{font-size:11px;line-height:1.7;margin:9px 0}.per-patch b{display:block}.idlist{font-family:ui-monospace,SFMono-Regular,monospace;font-size:10px;color:var(--muted);word-break:break-word}.form-field{display:block;font-size:11px;font-weight:600;margin:12px 0 5px}.text-input{width:100%;border:1px solid var(--line);border-radius:7px;padding:9px 10px;color:var(--ink);background:white}.text-input:focus{border-color:var(--blue)}textarea.text-input{resize:vertical;min-height:60px}.form-button{width:100%;margin-top:12px;padding:11px}.saved-item{padding:10px 0;border-bottom:1px solid #eff2f6}.saved-item:last-child{border-bottom:0}.saved-item strong{font-size:12px}.saved-item p{font-size:11px;line-height:1.5;color:var(--muted);margin:4px 0;word-break:break-word}.saved-item a{font-size:11px}.subtle-btn{border:0;background:transparent;padding:0;color:var(--blue);font-size:11px}.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#183a30;color:white;padding:13px 20px;border-radius:10px;box-shadow:0 5px 30px #0002;z-index:20;max-width:80vw;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}.toast.bad{background:#872c32}footer{font-size:11px;color:var(--muted);padding:20px 0 3px;overflow-wrap:anywhere;line-height:1.6}dialog{border:1px solid var(--line);border-radius:15px;padding:18px;max-width:95vw;width:920px;max-height:94vh;box-shadow:0 20px 80px #0003}dialog::backdrop{background:#172438a0;backdrop-filter:blur(3px)}.dialog-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:15px}.dialog-images{display:grid;grid-template-columns:2fr 1fr;gap:18px;align-items:start}.dialog-images img{width:100%;max-height:64vh;object-fit:contain;background:#edf0f5;border-radius:8px}.dialog-images p{font-size:11px;color:var(--muted)}.dialog-meta{font-size:12px;color:var(--muted);line-height:1.8;margin-top:12px;overflow-wrap:anywhere}.pairwise-list{max-height:220px;overflow:auto}.pairwise-row{border-bottom:1px solid #eff2f6;padding:7px 0;font-size:11px}.pairwise-row .idlist{margin-top:4px}.busy{opacity:.6;pointer-events:none}
@media(min-width:1800px){.workspace{grid-template-columns:minmax(0,1fr) 390px}.patch-grid{gap:12px}.card-head{padding:10px}.patch-id{font-size:12px}}
@media(max-width:1100px){.workspace{grid-template-columns:minmax(0,1fr) 310px}.shell{padding:15px}.heading h2{font-size:21px}.guide{display:none}}
@media(max-width:800px){header{padding:0 15px;height:70px}.workspace{grid-template-columns:1fr}.sidebar{position:static;max-height:none}.local{font-size:10px}.grid-nav{grid-template-columns:25px minmax(0,1fr) 25px}.shell{padding:12px}.brand h1{font-size:19px}.heading p{font-size:12px}.dialog-images{grid-template-columns:1fr}.dialog-images img{max-height:38vh}}
</style></head><body>
<header><div class="brand"><div class="mark" aria-hidden="true"><i></i><i></i><i></i><i></i></div><div><h1>Patchwork</h1><p>Prototype subgroup curator</p></div></div><div class="local">Local workspace · JSON saved to disk</div></header>
<div class="shell"><div class="heading"><div><div class="eyebrow">Visual concept analysis</div><h2>Find the pattern. Form a subgroup.</h2><p>Select patches that share a visual pattern. Inspect prototype overlap as you go.</p></div><div class="guide">01 Select patches &nbsp; 02 Inspect overlap<br>03 Name and save your subgroup</div></div>
<nav class="tabs" id="tabs" role="tablist" aria-label="Source groups"></nav><div id="global-error" class="warning error hidden" role="alert"></div>
<div class="workspace"><main class="gallery" id="gallery"><div class="toolbar"><div><strong id="group-title">Loading gallery…</strong><span class="stage-label">Stage 4</span><div class="muted" id="group-description">Reading patch metadata</div></div><div class="tools"><label><input id="hide-locked" type="checkbox">Hide saved</label><button id="reload" title="Reload this group's source metadata and saved subgroups">↻ Reload</button></div></div>
<div id="stale-warning" class="warning hidden"></div><div class="grid-nav"><button class="group-arrow" id="prev-group" aria-label="Previous source group" title="Previous group">‹</button><div class="grid-scroll"><div class="patch-grid" id="grid" role="tabpanel" aria-label="Patch gallery"></div><div id="empty-grid" class="empty hidden"></div></div><button class="group-arrow" id="next-group" aria-label="Next source group" title="Next group">›</button></div>
<div class="statusline" id="grid-status"></div><div class="pagination"><button id="prev-page">← Previous page</button><span id="page-label">Page 1 of 1</span><button id="next-page">Next page →</button></div></main>
<aside class="sidebar" aria-label="Subgroup selection and prototype overlap"><div class="panel-head"><div class="eyebrow">Live selection</div><h3>Prototype overlap</h3><p>Cosine-ranked matches from patch_metadata.json.<br>Updates immediately with every selection.</p><div class="stats"><div class="stat"><strong id="stat-patches">0</strong><span>patches</span></div><div class="stat"><strong id="stat-videos">0</strong><span>videos</span></div><div class="stat"><strong id="stat-shared">0</strong><span>shared IDs</span></div></div></div>
<div class="panel-section"><div class="row"><h4>Cosine matches per patch</h4><label for="top-k">Top-k <input class="k-input" id="top-k" type="number" value="20" min="1" max="100000"></label></div><p class="hint" id="k-warning">Ranks are computed from the matches stored in each patch record.</p><div class="row" style="margin-top:15px"><h4>Selected patches</h4><button class="subtle-btn" id="clear">Clear selection</button></div><div class="selection-list" id="selection-list"></div></div>
<div class="panel-section"><h4 style="margin:0;font-size:12px" id="shared-title">Shared by all selected patches</h4><div id="shared-ids" class="chips"></div><p class="hint" id="shared-hint">Select at least two patches to compare their prototype IDs.</p><details id="cosine-details"><summary>Cosines for shared prototypes</summary><div id="shared-cosines" class="scroll-table"></div></details><details id="support-details"><summary>Prototype frequency across selection</summary><div class="scroll-table"><table class="support-table"><thead><tr><th>Prototype</th><th>Coverage</th><th>Mean cosine</th></tr></thead><tbody id="support-body"></tbody></table></div></details><details id="patch-details"><summary>Top-k IDs and cosines per patch</summary><div id="per-patch" class="scroll-table"></div></details><details id="pair-details"><summary>Pairwise overlaps</summary><div id="pairwise" class="pairwise-list"></div></details><details id="activation-freq-details"><summary>Activation frequency</summary><div id="activation-freq" class="scroll-table"><p class="hint" id="activation-freq-hint">Select shared prototypes to inspect dataset-wide activation frequency.</p><table class="support-table hidden" id="activation-freq-table"><thead><tr><th>Prototype</th><th>Combined</th><th>Train</th><th>Val</th><th>Count</th></tr></thead><tbody id="activation-freq-body"></tbody></table></div></details></div>
<div class="panel-section"><div class="eyebrow">Save subgroup</div><label class="form-field" for="subgroup-name">Subgroup name <span class="muted">(optional)</span></label><input class="text-input" id="subgroup-name" maxlength="160" placeholder="e.g. Light diagonal edge on green"><label class="form-field" for="subgroup-description">Visual pattern / notes <span class="muted">(optional)</span></label><textarea class="text-input" id="subgroup-description" maxlength="4000" placeholder="Describe the common appearance…"></textarea><button class="primary form-button" id="form-group" disabled>Form group · 0 patches</button><p class="hint">Saves full metadata, ranked matches, shared IDs, and overlaps as JSON. Saved patches are locked within this source group.</p></div>
<div class="panel-section"><div class="row"><h4>Saved subgroups</h4><span class="muted" id="saved-count">0</span></div><div id="saved-list"><p class="hint">No subgroups formed yet.</p></div></div></aside></div><footer id="footer"></footer></div>
<div id="toast" class="toast hidden" role="status" aria-live="polite"></div><dialog id="preview"><div class="dialog-head"><strong id="preview-title">Patch preview</strong><button id="close-preview">Close ×</button></div><div class="dialog-images"><div><img id="preview-frame" alt="Full frame with patch location"><p>Associated frame · red box marks the nominal patch cell</p><button id="toggle-temporal" class="hidden">Play temporal window</button></div><div><img id="preview-patch" alt="Selected patch"><p>Patch crop</p></div></div><div class="dialog-meta" id="preview-meta"></div></dialog>
<script>
'use strict';
const $ = id => document.getElementById(id);
const state = {groups:[], active:null, data:null, drafts:new Map(), pages:new Map(), k:20, token:'', loading:false, saving:false, loadGeneration:0, activationFrequencies:null};
const PAGE_SIZE=64;
function node(tag, cls, value){const n=document.createElement(tag);if(cls)n.className=cls;if(value!==undefined)n.textContent=String(value);return n;}
function toast(message,bad=false){const n=$('toast');n.textContent=message;n.classList.toggle('bad',bad);n.classList.remove('hidden');clearTimeout(toast.timer);toast.timer=setTimeout(()=>n.classList.add('hidden'),bad?10000:6000);}
function issue(message){$('global-error').textContent=message;$('global-error').classList.toggle('hidden',!message);}
async function api(path, options={}){const r=await fetch(path,{cache:'no-store',...options});const b=await r.json();if(!r.ok)throw new Error(b.error||`HTTP ${r.status}`);return b;}
function draft(){if(!state.drafts.has(state.active))state.drafts.set(state.active,{ids:new Set(),name:'',description:''});return state.drafts.get(state.active);}
function ranked(p,k){return p.top_k_cosine_matches.map((m,i)=>({...m,_order:i})).sort((a,b)=>b.cosine_similarity-a.cosine_similarity||a._order-b._order).slice(0,k);}
function compute(patches,k){const rows=patches.map(p=>({id:p.group_patch_index,matches:ranked(p,k)}));const sets=rows.map(r=>new Set(r.matches.map(m=>m.prototype_index)));const shared=rows.length?rows[0].matches.map(m=>m.prototype_index).filter(id=>sets.every(s=>s.has(id))):[];const support=new Map();rows.forEach(r=>r.matches.forEach(m=>{if(!support.has(m.prototype_index))support.set(m.prototype_index,{id:m.prototype_index,entries:[]});support.get(m.prototype_index).entries.push({patch:r.id,cosine:m.cosine_similarity});}));return {rows,sets,shared,support:[...support.values()].sort((a,b)=>b.entries.length-a.entries.length||a.id-b.id)};}
function selectedPatches(){if(!state.data)return [];const byId=new Map(state.data.patches.map(p=>[p.metadata.group_patch_index,p.metadata]));return [...draft().ids].map(id=>byId.get(id)).filter(Boolean);}
function persistDraft(){if(!state.data)return;const d=draft();d.name=$('subgroup-name').value;d.description=$('subgroup-description').value;try{sessionStorage.setItem('patchwork:'+state.root+':'+state.active,JSON.stringify({digest:state.data.metadata_sha256,ids:[...d.ids],name:d.name,description:d.description}));}catch(e){/* in-memory drafts still work */}}
function restoreDraft(){const d=draft();if(d.digest&&d.digest!==state.data.metadata_sha256){d.ids.clear();d.name='';d.description='';toast('Source metadata changed. The previous draft selection has been cleared.',true);}if(!d.restored){d.restored=true;try{const raw=JSON.parse(sessionStorage.getItem('patchwork:'+state.root+':'+state.active));if(raw&&raw.digest===state.data.metadata_sha256){d.ids=new Set(raw.ids);d.name=raw.name||'';d.description=raw.description||'';}}catch(e){}}d.digest=state.data.metadata_sha256;}
function renderTabs(){const tabs=$('tabs');tabs.replaceChildren();state.groups.forEach((g,i)=>{const b=node('button','tab');b.setAttribute('role','tab');b.setAttribute('aria-selected',String(g.name===state.active));b.id='tab-'+g.name;b.textContent='Group '+g.name.split('_')[1];b.append(node('small','',g.error?'!':g.patch_count));b.addEventListener('click',()=>openGroup(g.name));tabs.append(b);});const index=state.groups.findIndex(g=>g.name===state.active);$('prev-group').disabled=index<=0||state.loading;$('next-group').disabled=index<0||index>=state.groups.length-1||state.loading;}
async function openGroup(name){if(state.saving)return;persistDraft();const generation=++state.loadGeneration;state.active=name;state.data=null;state.loading=true;issue('');renderTabs();$('grid').replaceChildren();$('empty-grid').textContent='Loading patches…';$('empty-grid').classList.remove('hidden');$('group-title').textContent=name;$('group-description').textContent='Reading metadata…';$('form-group').disabled=true;clearPanel();try{const data=await api('/api/group?name='+encodeURIComponent(name));if(generation!==state.loadGeneration)return;state.data=data;state.loading=false;restoreDraft();const d=draft();const available=new Set(data.patches.map(p=>p.metadata.group_patch_index));d.ids=new Set([...d.ids].filter(id=>available.has(id)&&!Object.hasOwn(data.locked_patch_ids,String(id))));$('subgroup-name').value=d.name;$('subgroup-description').value=d.description;renderTabs();renderGrid();renderSelection();renderSaved();const stale=data.stale_saved_subgroups;$('stale-warning').classList.toggle('hidden',!stale.length);$('stale-warning').textContent=stale.length?'Source metadata changed since some subgroups were saved. Existing patch locks remain visible; you can still form new subgroups from unlocked patches.':'';persistDraft();}catch(e){if(generation!==state.loadGeneration)return;state.loading=false;issue(e.message);$('empty-grid').textContent='This group could not be loaded. Check the error above, then reload.';renderTabs();}}
function clearPanel(){$('stat-patches').textContent='0';$('stat-videos').textContent='0';$('stat-shared').textContent='0';for(const id of ['selection-list','shared-ids','shared-cosines','support-body','per-patch','pairwise','activation-freq-body','saved-list'])$(id).replaceChildren();$('saved-count').textContent='0';$('subgroup-name').value='';$('subgroup-description').value='';$('shared-hint').textContent='Select at least two patches to compare their prototype IDs.';$('activation-freq-hint').textContent='Select shared prototypes to inspect dataset-wide activation frequency.';$('activation-freq-hint').classList.remove('hidden');$('activation-freq-table').classList.add('hidden');}
function mediaButton(url,cls,alt,item){const b=node('button','media-button');b.type='button';b.title='Open enlarged frame and patch';b.setAttribute('aria-label',alt+' — enlarge');if(url){const im=node('img',cls);im.src=url;im.alt=alt;im.loading='lazy';im.decoding='async';im.addEventListener('error',()=>{im.replaceWith(node('div','hint','Image unavailable'));});b.append(im);}else{b.append(node('div','hint','Image missing'));}b.addEventListener('click',()=>showPreview(item));return b;}
function renderGrid(){if(!state.data)return;const data=state.data;const ids=draft().ids;const filter=$('hide-locked').checked;const items=data.patches.filter(p=>!filter||!Object.hasOwn(data.locked_patch_ids,String(p.metadata.group_patch_index)));const pages=Math.max(1,Math.ceil(items.length/PAGE_SIZE));const page=Math.max(0,Math.min(state.pages.get(state.active)||0,pages-1));state.pages.set(state.active,page);$('grid').setAttribute('aria-labelledby','tab-'+state.active);$('grid').replaceChildren();items.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE).forEach(item=>{const p=item.metadata,id=p.group_patch_index,locked=data.locked_patch_ids[String(id)];const card=node('article','card'+(ids.has(id)?' selected':'')+(locked?' locked':''));card.dataset.patchId=id;const head=node('div','card-head');head.append(node('span','patch-id','P'+String(id).padStart(2,'0')));const input=node('input','pick');input.type='checkbox';input.checked=ids.has(id);input.disabled=!!locked||state.saving;input.setAttribute('aria-label',`Select patch ${id}`);input.title=locked?`Saved in ${locked}`:'Select / unselect patch';input.addEventListener('change',()=>toggle(id));head.append(input);card.append(head);card.append(mediaButton(item.media.frame,'frame-img',`Frame for patch ${id}`,item));const cap=node('div','image-caption');cap.append(node('span','','Patch'));if(p.is_target_reference)cap.append(node('span','target','Target'));card.append(cap);const wrap=node('div','patch-wrap');wrap.append(mediaButton(item.media.patch,'patch-img',`Patch ${id}`,item));card.append(wrap);const foot=node('div','card-foot',`V ${p.video_id??'—'} · r${p.grid_row??'?'} c${p.grid_column??'?'}`);foot.title=`Video ${p.video_id??'—'} · ${p.frame_filename??'unknown frame'}`;card.append(foot);if(locked)card.append(node('div','lock-badge','✓ '+locked));$('grid').append(card);});$('empty-grid').classList.toggle('hidden',items.length>0);$('empty-grid').textContent=filter?'All patches in this group have been saved. Uncheck “Hide saved” to review them.':'No patch records in this group.';$('group-title').textContent='Group '+state.active.split('_')[1];const lockedCount=data.patches.filter(p=>Object.hasOwn(data.locked_patch_ids,String(p.metadata.group_patch_index))).length;$('group-description').textContent=`${data.patches.length} patches · ${data.patches.length-lockedCount} available · ${data.saved_subgroups.length} subgroups formed`;$('page-label').textContent=`Page ${page+1} of ${pages}`;$('prev-page').disabled=page===0;$('next-page').disabled=page===pages-1;$('grid-status').textContent=items.length?`Showing ${page*PAGE_SIZE+1}–${Math.min((page+1)*PAGE_SIZE,items.length)} of ${items.length} · 8 columns × up to 8 rows · selection spans pages`:'';}
function toggle(id){if(!state.data||state.saving||Object.hasOwn(state.data.locked_patch_ids,String(id)))return;const ids=draft().ids;if(ids.has(id))ids.delete(id);else if(ids.size<512)ids.add(id);else toast('A subgroup can contain at most 512 patches.',true);const card=$('grid').querySelector(`[data-patch-id="${id}"]`);if(card){card.classList.toggle('selected',ids.has(id));card.querySelector('input').checked=ids.has(id);}renderSelection();persistDraft();}
function renderSelection(){if(!state.data)return;const patches=selectedPatches(),count=patches.length,report=compute(patches,state.k);state.report=report;$('stat-patches').textContent=count;$('stat-videos').textContent=new Set(patches.map(p=>String(p.video_id??''))).size;$('stat-shared').textContent=count>=2?report.shared.length:'—';$('shared-title').textContent=count===1?'Top-k IDs for the selected patch':`Shared by all ${count||''} selected patches`;$('selection-list').replaceChildren();patches.forEach(p=>{const b=node('button','remove-patch','P'+p.group_patch_index);b.append(node('span','','×'));b.setAttribute('aria-label','Unselect patch '+p.group_patch_index);b.disabled=state.saving;b.addEventListener('click',()=>toggle(p.group_patch_index));$('selection-list').append(b);});$('shared-ids').replaceChildren();report.shared.forEach(id=>$('shared-ids').append(node('span','chip green',id)));$('shared-hint').textContent=count===0?'Select at least two patches to compare their prototype IDs.':count===1?'One patch selected. Add another to calculate shared IDs.':report.shared.length?`${report.shared.length} IDs occur in every selected patch’s stored top-${state.k} list. Ordered by the first selected patch’s cosine rank.`:'No prototype IDs are shared by every selected patch. You can still form this visual subgroup.';const short=patches.filter(p=>p.top_k_cosine_matches.length<state.k);$('k-warning').textContent=short.length?`${short.length} selected patch(es) store fewer than ${state.k} matches. Using all available matches for those patches; no missing ranks are inferred.`:'Ranks use descending cosine similarity. These are distinct from actual forward activations.';$('k-warning').style.color=short.length?'var(--orange)':'';$('shared-cosines').replaceChildren();report.shared.forEach(id=>{const d=node('div','shared-detail');d.append(node('strong','','Prototype '+id+' · '));d.append(document.createTextNode(report.rows.map(r=>`P${r.id}: ${r.matches.find(m=>m.prototype_index===id).cosine_similarity.toFixed(4)}`).join(' · ')));$('shared-cosines').append(d);});$('support-body').replaceChildren();report.support.forEach(entry=>{const row=node('tr');row.append(node('td','mono',entry.id));const coverage=node('td','',`${entry.entries.length}/${count}`);const bar=node('div','bar'),fill=node('i');fill.style.width=(100*entry.entries.length/count)+'%';bar.append(fill);coverage.append(bar);coverage.title=entry.entries.map(e=>'P'+e.patch).join(', ');row.append(coverage);row.append(node('td','mono',(entry.entries.reduce((s,e)=>s+e.cosine,0)/entry.entries.length).toFixed(4)));$('support-body').append(row);});$('per-patch').replaceChildren();report.rows.forEach(r=>{const div=node('div','per-patch');div.append(node('b','','Patch '+r.id));div.append(node('div','idlist',r.matches.map(m=>`${m.prototype_index} (${m.cosine_similarity.toFixed(4)})`).join(', ')||'No matches stored'));$('per-patch').append(div);});$('pairwise').replaceChildren();if($('pair-details').open)renderPairwise();renderActivationFreq();$('form-group').textContent=state.saving?'Saving subgroup…':`Form group · ${count} patch${count===1?'':'es'}`;$('form-group').disabled=count===0||state.saving;$('clear').disabled=count===0||state.saving;}
function renderPairwise(){const r=state.report;if(!r)return;$('pairwise').replaceChildren();let displayed=0;outer:for(let i=0;i<r.rows.length;i++)for(let j=i+1;j<r.rows.length;j++){if(displayed++>=500){$('pairwise').append(node('p','hint','Showing the first 500 pairs. All pairwise overlaps are included in the saved JSON.'));break outer;}const shared=r.rows[i].matches.map(m=>m.prototype_index).filter(id=>r.sets[j].has(id));const div=node('div','pairwise-row');div.append(node('b','',`P${r.rows[i].id} & P${r.rows[j].id} · ${shared.length} shared`));div.append(node('div','idlist',shared.join(', ')||'No shared IDs'));$('pairwise').append(div);}if(r.rows.length<2)$('pairwise').append(node('p','hint','Select two or more patches.'));}
function fmtFreq(value){return Number.isFinite(value)?(100*value).toFixed(3)+'%':'—';}
function renderActivationFreq(){const hint=$('activation-freq-hint'),table=$('activation-freq-table'),body=$('activation-freq-body');body.replaceChildren();const report=state.report;if(!state.activationFrequencies){hint.textContent='No activation-frequency CSV loaded. Run prototype_activation_freq.py, or pass --activation-freq-csv.';hint.classList.remove('hidden');table.classList.add('hidden');return;}if(!report||report.shared.length===0){hint.textContent=(!report||report.rows.length<2)?'Select at least two patches that share prototype IDs.':'No prototypes are shared by every selected patch.';hint.classList.remove('hidden');table.classList.add('hidden');return;}const freq=state.activationFrequencies.prototypes||{};const missing=[];const rows=report.shared.map(id=>{const entry=freq[String(id)];if(!entry){missing.push(id);return null;}return entry;}).filter(Boolean).sort((a,b)=>b.combined_frequency-a.combined_frequency||a.prototype_index-b.prototype_index);if(!rows.length){hint.textContent='Shared prototypes were not found in the activation-frequency CSV.';hint.classList.remove('hidden');table.classList.add('hidden');return;}hint.textContent=missing.length?`Dataset-wide top-8 activation rates for shared prototypes. Missing from CSV: ${missing.join(', ')}.`:'Dataset-wide top-8 activation rates for prototypes shared by every selected patch.';hint.classList.remove('hidden');table.classList.remove('hidden');rows.forEach(entry=>{const row=node('tr');row.append(node('td','mono',entry.prototype_index));row.append(node('td','mono',fmtFreq(entry.combined_frequency)));row.append(node('td','mono',fmtFreq(entry.train_frequency)));row.append(node('td','mono',fmtFreq(entry.val_frequency)));row.append(node('td','mono',entry.combined_count));body.append(row);});}
function renderSaved(){$('saved-list').replaceChildren();$('saved-count').textContent=state.data.saved_subgroups.length;if(!state.data.saved_subgroups.length)$('saved-list').append(node('p','hint','No subgroups formed yet.'));state.data.saved_subgroups.forEach(s=>{const div=node('div','saved-item');div.append(node('strong','',s.name));div.append(node('p','',`${s.subgroup_id} · ${s.patch_ids.length} patches · ${s.distinct_video_count} videos`));div.append(node('p','mono',s.patch_ids.map(id=>'P'+id).join(', ')));if(s.description)div.append(node('p','',s.description));div.append(node('p','',`Top-${s.requested_top_k} shared IDs: ${s.shared_by_all_prototype_ids.join(', ')||'none'}`));const link=node('a','','Download JSON ↗');link.href=`/api/download/${state.active}/${s.subgroup_id}.json`;link.download=s.subgroup_id+'.json';div.append(link);$('saved-list').append(div);});}
async function formGroup(){if(!state.data||state.saving||!draft().ids.size)return;persistDraft();const source=state.active,d=draft();state.saving=true;renderGrid();renderSelection();$('top-k').disabled=true;$('subgroup-name').disabled=true;$('subgroup-description').disabled=true;try{const result=await api('/api/subgroups',{method:'POST',headers:{'Content-Type':'application/json','X-Curator-Token':state.token},body:JSON.stringify({source_group:source,patch_ids:[...d.ids],k:state.k,name:d.name,description:d.description,metadata_sha256:state.data.metadata_sha256})});d.ids.clear();d.name='';d.description='';$('subgroup-name').value='';$('subgroup-description').value='';persistDraft();toast('Saved '+result.subgroup.name+'\n'+result.saved_path);state.saving=false;await openGroup(source);}catch(e){toast(e.message,true);}finally{state.saving=false;$('top-k').disabled=false;$('subgroup-name').disabled=false;$('subgroup-description').disabled=false;renderGrid();renderSelection();}}
function showPreview(item){const p=item.metadata;$('preview-title').textContent=`${state.active} · Patch ${p.group_patch_index}`;for(const [id,url] of [['preview-frame',item.media.frame],['preview-patch',item.media.patch]]){const img=$(id);if(url){img.src=url;img.classList.remove('hidden');}else{img.removeAttribute('src');img.classList.add('hidden');}}$('preview-meta').textContent=`Video ${p.video_id??'—'} · Frame ${p.frame_filename??'—'} · Grid row ${p.grid_row??'—'}, column ${p.grid_column??'—'} · Candidate ${p.candidate_index??'—'}`;const toggle=$('toggle-temporal');toggle.classList.toggle('hidden',!item.media.temporal);toggle.textContent='Play temporal window';toggle.onclick=()=>{const play=toggle.textContent==='Play temporal window';$('preview-frame').src=play?item.media.temporal:item.media.frame;toggle.textContent=play?'Show boxed frame':'Play temporal window';};$('preview').showModal();}
function moveGroup(delta){const i=state.groups.findIndex(g=>g.name===state.active),next=state.groups[i+delta];if(next)openGroup(next.name);}
$('prev-group').onclick=()=>moveGroup(-1);$('next-group').onclick=()=>moveGroup(1);$('prev-page').onclick=()=>{state.pages.set(state.active,(state.pages.get(state.active)||0)-1);renderGrid();};$('next-page').onclick=()=>{state.pages.set(state.active,(state.pages.get(state.active)||0)+1);renderGrid();};$('hide-locked').onchange=()=>{state.pages.set(state.active,0);renderGrid();};$('reload').onclick=()=>{if(state.active)openGroup(state.active);else init();};$('clear').onclick=()=>{draft().ids.clear();renderGrid();renderSelection();persistDraft();};$('top-k').addEventListener('input',()=>{const k=Number($('top-k').value);if(Number.isInteger(k)&&k>=1&&k<=100000){state.k=k;renderSelection();}});$('top-k').addEventListener('blur',()=>{$('top-k').value=state.k;});$('pair-details').addEventListener('toggle',()=>{if($('pair-details').open)renderPairwise();});$('activation-freq-details').addEventListener('toggle',()=>{if($('activation-freq-details').open)renderActivationFreq();});$('form-group').onclick=formGroup;$('subgroup-name').oninput=persistDraft;$('subgroup-description').oninput=persistDraft;$('close-preview').onclick=()=>$('preview').close();$('preview').addEventListener('click',e=>{if(e.target===$('preview')){const b=$('preview').getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)$('preview').close();}});
async function init(){try{const config=await api('/api/groups');state.groups=config.groups;state.root=config.root;state.token=config.csrf_token;state.activationFrequencies=config.activation_frequencies||null;const freqNote=state.activationFrequencies?`  ·  Activation freq: ${state.activationFrequencies.source_csv}`:'  ·  Activation freq: not loaded';$('footer').textContent=`Source: ${config.root}  ·  Saved JSON: ${config.output_dir}`+freqNote;await openGroup(state.active||state.groups[0].name);}catch(e){issue(e.message);$('group-description').textContent='Gallery unavailable';}}
init();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    server_version = "Patchwork/1.0"

    @property
    def curator(self):
        return self.server.curator

    def send_bytes(self, data, content_type, status=200, attachment=None, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=60" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "same-origin")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, obj, status=200):
        self.send_bytes(json.dumps(obj, ensure_ascii=False, allow_nan=False).encode(),
                        "application/json; charset=utf-8", status)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            route = urlsplit(self.path)
            path = unquote(route.path)
            if path == "/":
                self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            elif path == "/api/groups":
                self.send_json(self.curator.groups())
            elif path == "/api/group":
                self.send_json(self.curator.group(parse_qs(route.query).get("name", [None])[0]))
            elif path.startswith("/api/download/"):
                parts = path.removeprefix("/api/download/").split("/")
                if len(parts) != 2 or not SAVE_RE.fullmatch(parts[1]):
                    raise Problem("Subgroup not found.", 404)
                self.curator.group_dir(parts[0])
                file = (self.curator.output / parts[0] / parts[1]).resolve()
                if self.curator.output not in file.parents or not file.is_file():
                    raise Problem("Subgroup not found.", 404)
                self.send_bytes(file.read_bytes(), "application/json", attachment=file.name)
            elif path.startswith("/media/"):
                relative = Path(path.removeprefix("/media/"))
                if not relative.parts or not GROUP_RE.fullmatch(relative.parts[0]):
                    raise Problem("Image not found.", 404)
                self.curator.group_dir(relative.parts[0])
                file = (self.curator.root / relative).resolve()
                if (self.curator.root not in file.parents or not file.is_file()
                        or file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}):
                    raise Problem("Image not found.", 404)
                self.send_bytes(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "image/png", cache=True)
            elif path == "/favicon.ico":
                self.send_bytes(b"", "image/x-icon", 204)
            else:
                raise Problem("Not found.", 404)
        except (Problem, OSError, ValueError) as exc:
            self.send_json({"error": str(exc)}, getattr(exc, "status", 500))

    def do_POST(self):
        try:
            if urlsplit(self.path).path != "/api/subgroups":
                raise Problem("Not found.", 404)
            if not secrets.compare_digest(self.headers.get("X-Curator-Token", ""), self.curator.token):
                raise Problem("Invalid session token. Reload the page.", 403)
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise Problem("Expected JSON.", 415)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1000000:
                raise Problem("Invalid request length.", 413)
            try:
                body = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeError):
                raise Problem("Invalid JSON.")
            self.send_json(self.curator.form(body), 201)
        except (Problem, OSError, ValueError) as exc:
            self.send_json({"error": str(exc)}, getattr(exc, "status", 500))

    def log_message(self, message, *args):
        if self.command == "POST" or (args and str(args[1:2]) not in ("('200',)", "('304',)")):
            super().log_message(message, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="Gallery containing group_XX/patch_metadata.json")
    parser.add_argument("--output-dir", type=Path, help="JSON output directory outside the gallery (default: sibling <root>_manual_subgroups)")
    parser.add_argument(
        "--activation-freq-csv",
        type=Path,
        help="CSV from prototype_activation_freq.py (default: newest file in prototype_activation_freq_outputs/)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output_dir or root.with_name(root.name + "_manual_subgroups")
    curator = None
    try:
        curator = Curator(root, output, activation_freq_csv=args.activation_freq_csv)
        curator.groups()
        curator.acquire_process_lock()
        with ThreadingHTTPServer((args.host, args.port), Handler) as server:
            server.curator = curator
            print(f"Patchwork: http://{args.host}:{server.server_port}", flush=True)
            print(f"Source: {curator.root}\nSaved JSON: {curator.output}", flush=True)
            if curator.activation_frequencies:
                print(f"Activation freq: {curator.activation_frequencies['source_csv']}", flush=True)
            else:
                print("Activation freq: not loaded (run prototype_activation_freq.py or pass --activation-freq-csv)", flush=True)
            print("Press Ctrl+C to stop.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    except (Problem, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    finally:
        if curator:
            curator.close()


if __name__ == "__main__":
    main()
