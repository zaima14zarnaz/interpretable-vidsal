#!/usr/bin/env python3
"""Reassign compact temporal windows to a new frozen bank, without optical flow.

Place beside cache_optical_flow.py. Example:
  python reassign_temporal_cache.py --source-index /old/namespace/index.sqlite \
      --bank-dir /new/temporal_cluster --cache-root /cached_temporal_feats \
      --expected-clusters 1042 --workers 4

Only saved unnormalized patch summaries are read. No images, dense flow,
backbone, flow estimator, GPU context, or learned encoder is used. All source
splits, window starts and views are retained. Extraction settings must match;
new training normalization, distance weights and prototypes are applied.
Original training assignments in the ncd ew bank are retained when their exact
frame-pair hashes match (including medoid ownership on ties). Other windows
receive role-routed nearest-medoid assignments using the cache builder's math.

The output is a separate compact namespace, printed when the job starts and
finishes. Rerun the same command to resume. The original cache and bank are
opened read-only. Existing output files are checked and never overwritten.
Conversion cannot fill missing windows or views; source coverage is preserved.
CPU threads process bounded groups of windows; --workers controls concurrency.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np

CONVERSION_SCHEMA = "reassigned_compact_temporal_windows_v1"
RAW_FIELDS = {
    "flow_hist": (9,), "flow_magnitude": (), "flow_mean": (2,),
    "flow_spread": (), "flow_variance": (), "global_motion_raw": (2,),
    "flow_valid_fraction": (), "flow_valid": (), "role": (),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def load_helper(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Place this file beside cache_optical_flow.py, or pass --cache-script: {path}")
    spec = importlib.util.spec_from_file_location("_temporal_conversion_cache_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("Bank", "load_core", "assign_frozen", "init_index", "digest", "file_sha",
                "save_npz_once", "atomic_json", "readonly", "now", "SCHEMA", "BANK_FORMAT")
    if any(not hasattr(module, name) for name in required):
        raise ValueError("--cache-script must be the compact v2 flow-only cache_optical_flow.py builder.")
    return module, path


def load_source(index, helper):
    index = Path(index).expanduser().resolve()
    if index.is_dir():
        index = index / "index.sqlite"
    if not index.is_file() or index.name != "index.sqlite":
        raise FileNotFoundError("--source-index must name an existing namespace/index.sqlite.")
    manifest = read_json(index.parent / "manifest.json")
    if manifest.get("schema") != helper.SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Stop the source builder and finish its cache first; a complete source cache is required.")
    with closing(helper.readonly(index)) as con:
        row = con.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if row is None:
            raise ValueError("Source index has no contract.")
        contract = json.loads(row[0])
        if contract != manifest.get("contract") or contract.get("schema") != helper.SCHEMA:
            raise ValueError("Source manifest and index contracts disagree.")
        if index.parent.name != contract.get("namespace"):
            raise ValueError("Source namespace directory and contract disagree.")
        count = con.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
        if count == 0:
            raise ValueError("Source cache is empty.")
        if con.execute("SELECT 1 FROM windows GROUP BY split,video,window_start,augmentation HAVING COUNT(*)>1 LIMIT 1").fetchone():
            raise ValueError("Source index has ambiguous duplicate sample/view entries.")
        if con.execute("SELECT 1 FROM windows WHERE augmentation NOT IN ('original','hflip') LIMIT 1").fetchone():
            raise ValueError("Only original/hflip cached views can be converted.")
    return index, manifest, contract, count


def check_compatible(source_index, source, bank):
    expected = dict(
        descriptor=bank.desc, flow_config=bank.flow_config,
        flow_frame_gaps=bank.gaps, canvas_hw=list(bank.canvas),
        patch_size=list(bank.patch), grid_hw=list(bank.grid),
    )
    changed = [name for name, value in expected.items() if source.get(name) != value]
    if changed:
        raise ValueError("Cached extraction settings differ from the new bank: " + ", ".join(changed)
                         + ". Pooled summaries cannot reconstruct changed extraction settings.")
    if source.get("bank_format") != bank.manifest["format_version"]:
        raise ValueError("Source and target bank formats differ.")
    if source.get("window_len") != bank.window_len:
        raise ValueError("Source context length cannot change in a descriptor-only conversion.")
    if bank.desc.get("camera_motion") == "affine":
        snapshot = source_index.parent / "bank_contract.json"
        if not snapshot.is_file():
            raise ValueError("Affine provenance needs the source bank_contract.json.")
        original = read_json(snapshot)["manifest"]
        old_seed = original.get("viewer_camera_seed", original.get("seed"))
        new_seed = bank.manifest.get("viewer_camera_seed", bank.manifest.get("seed"))
        if old_seed != new_seed:
            raise ValueError("Source/target affine camera-fit seeds differ.")


def renormalize(raw, bank):
    """Mirror merge_pairs after its float32 pooling boundary; never repool flow."""
    arrays = {name: value.copy() for name, value in raw.items()}
    norm = bank.norm

    def transform(value, scale):
        value = np.maximum(value, 0)
        result = value / scale if norm["magnitude_transform"] == "clip" else np.log1p(value) / np.log1p(scale)
        return np.clip(result, 0, 1)

    magnitude = np.stack([
        transform(arrays["flow_magnitude"], norm["flow_magnitude_scale"]),
        transform(np.sqrt(arrays["flow_variance"]), norm["variation_std_scale"]),
    ], axis=-1).astype(np.float32)
    global_motion = arrays["global_motion_raw"] / norm["global_motion_scale"]
    global_motion /= np.maximum(1, np.linalg.norm(global_motion, axis=1, keepdims=True))
    compact = np.column_stack([
        np.clip(arrays["flow_mean"] / norm["flow_magnitude_scale"], -1, 1),
        transform(arrays["flow_spread"], norm["flow_magnitude_scale"]),
    ]).astype(np.float32)
    arrays.update(magnitude=magnitude, compact=compact, global_motion=global_motion.astype(np.float32))
    parts = []
    for name, _ in bank.components:
        if name == "normalized_flow_magnitude":
            value = magnitude[:, :1]
        elif name == "normalized_cross_gap_std":
            value = magnitude[:, 1:2]
        elif name in ("flow_valid", "flow_valid_fraction"):
            value = np.asarray(arrays[name], np.float32)[:, None]
        else:
            value = arrays[name]
        parts.append(value)
    arrays["encoder_inputs"] = np.concatenate(parts, axis=-1).astype(np.float32)
    arrays["encoder_inputs"][~arrays["flow_valid"], :-2] = 0
    return arrays


class Converter:
    def __init__(self, source_index, source_contract, bank, core, helper, directory, contract):
        self.index, self.source, self.bank, self.core, self.helper = source_index, source_contract, bank, core, helper
        self.directory, self.contract = directory, contract
        self.local = threading.local()
        self.connections, self.connection_lock = [], threading.Lock()
        self.frame_map = {}
        self.groups = {}
        for role in (0, 1, 2):
            columns = np.flatnonzero(bank.references["role"] == role)
            self.groups[role] = SimpleNamespace(
                ids=bank.ids[columns], weights=bank.weights, metric=bank.metric,
                references={name: bank.references[name][columns] for name in
                            ("flow_hist", "magnitude", "compact", "global_motion", "flow_valid", "role")})
        if bank.db is not None:
            print("Indexing saved target-bank training assignments (read-only)...", flush=True)
            for row in bank.db.execute("SELECT id,video,window_start,target_index,global_model_json FROM frames WHERE dataset='train'"):
                key = (row["video"], row["window_start"], row["target_index"])
                if key in self.frame_map:
                    raise ValueError("Target bank contains ambiguous duplicate training frames.")
                self.frame_map[key] = (row["id"], row["global_model_json"])
            print(f"Indexed {len(self.frame_map):,} target-bank windows.", flush=True)

    def close(self):
        for con in self.connections:
            con.close()

    def saved_training_labels(self, meta):
        if meta["split"] != "train" or meta["augmentation"] != "original":
            return None
        target = meta["target_index"]
        frame = self.frame_map.get((meta["video"], target - self.bank.bank_window_len + 1, target))
        if frame is None or json.loads(frame[1]).get("pair_keys") != [p["raw_pair_key"] for p in meta["pairs"]]:
            return None
        if not hasattr(self.local, "bank_db"):
            con = sqlite3.connect(self.bank.db_path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
            self.local.bank_db = con
            with self.connection_lock:
                self.connections.append(con)
        rows = self.local.bank_db.execute(
            "SELECT row,col,prototype_id,distance FROM patches WHERE frame_id=? ORDER BY row,col", (frame[0],)).fetchall()
        gh, gw = self.bank.grid
        if len(rows) != gh * gw or any((r[0], r[1]) != (i // gw, i % gw) or r[2] is None or r[3] is None
                                      for i, r in enumerate(rows)):
            raise ValueError("Target bank has incomplete saved training assignments.")
        labels = np.array([r[2] for r in rows], np.int32)
        distances = np.array([r[3] for r in rows], np.float32)
        return labels, distances, np.isfinite(distances)

    def nearest(self, arrays):
        n = len(arrays["role"])
        labels = np.full(n, -1, np.int32)
        distances = np.full(n, np.inf, np.float32)
        valid = np.zeros(n, bool)
        for role, group in self.groups.items():
            selected = np.flatnonzero(arrays["role"] == role)
            if not len(selected) or not len(group.ids):
                continue
            subset = {name: value[selected] for name, value in arrays.items()}
            labels[selected], distances[selected], valid[selected] = self.helper.assign_frozen(subset, group, self.core)
        return labels, distances, valid

    def validate_output(self, arrays, meta, key, source_hash):
        n = int(np.prod(self.bank.grid))
        if (meta.get("cache_key") != key or meta.get("namespace") != self.contract["namespace"]
                or meta.get("bank_identity") != self.bank.identity
                or meta.get("conversion_source_npz_sha256") != source_hash
                or meta.get("dense_saved") is not False):
            raise ValueError("Existing converted window has incompatible metadata; it was preserved.")
        if (arrays["encoder_inputs"].shape != (n, sum(d for _, d in self.bank.components))
                or not np.isfinite(arrays["encoder_inputs"]).all()
                or arrays["cluster_ids"].shape != (n,)
                or np.any((arrays["cluster_ids"] < -1) | (arrays["cluster_ids"] >= self.bank.count))
                or arrays["cluster_distances"].shape != (n,)
                or arrays["assignment_valid"].shape != (n,) or arrays["assignment_valid"].dtype != bool
                or arrays["temporal_train_mask"].shape != (n,) or arrays["temporal_train_mask"].dtype != bool
                or not np.array_equal(arrays["assignment_valid"], arrays["cluster_ids"] >= 0)
                or not np.isfinite(arrays["cluster_distances"][arrays["assignment_valid"]]).all()
                or not np.array_equal(arrays["temporal_train_mask"], arrays["assignment_valid"] & arrays["flow_valid"])):
            raise ValueError("Invalid converted compact window; existing file was preserved.")

    def convert(self, row):
        path = (self.index.parent / row["path"]).resolve()
        if not path.is_relative_to(self.index.parent):
            raise ValueError("Source index points outside its namespace.")
        content = path.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        with np.load(io.BytesIO(content), allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            expected = dict(cache_key=row["cache_key"], namespace=self.source["namespace"],
                            bank_identity=self.source["bank_identity"], split=row["split"], video=row["video"],
                            window_start=row["window_start"], target_index=row["target_index"],
                            augmentation=row["augmentation"])
            expected.update({name: self.source[name] for name in
                             ("grid_hw", "patch_size", "canvas_hw", "flow_frame_gaps", "window_len")})
            if any(meta.get(name) != value for name, value in expected.items()):
                raise ValueError(f"Source window identity/geometry differs from its contract: {path}")
            if meta["target_index"] != meta["window_start"] + meta["window_len"] - 1:
                raise ValueError("Source window target is not its last frame.")
            pairs = meta["pairs"]
            if ([p["gap"] for p in pairs] != self.bank.gaps
                    or any(p["target_index"] != meta["target_index"]
                           or p["previous_index"] != meta["target_index"] - p["gap"] for p in pairs)):
                raise ValueError("Source window frame pairs differ from its contract.")
            raw = {}
            n = int(np.prod(self.bank.grid))
            for name, tail in RAW_FIELDS.items():
                if name not in z:
                    raise ValueError(f"Source window lacks unnormalized {name}: {path}")
                value = z[name]
                if value.shape != (n, *tail) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid unnormalized {name}: {path}")
                raw[name] = value
        if (raw["flow_valid"].dtype != bool or np.any((raw["role"] < 0) | (raw["role"] > 2))
                or not np.array_equal(raw["flow_valid"], raw["role"] != 0)
                or np.any(raw["flow_variance"] < 0)):
            raise ValueError("Invalid source roles, validity or cross-gap variance.")
        key = self.helper.digest(dict(namespace=self.contract["namespace"], split=meta["split"], video=meta["video"],
                                      start=meta["window_start"], target=meta["target_index"], augmentation=meta["augmentation"],
                                      pair_keys=[p["raw_pair_key"] for p in pairs]))
        destination = self.directory / "windows" / key[:2] / (key + ".npz")
        hit = destination.is_file()
        if hit:
            arrays = self.helper.load_npz(destination)
            new_meta = json.loads(str(arrays.pop("metadata")))
            self.validate_output(arrays, new_meta, key, source_hash)
            if any(not np.array_equal(arrays[name], value) for name, value in raw.items()):
                raise ValueError("Existing converted raw summaries differ from the source; file preserved.")
        else:
            arrays = renormalize(raw, self.bank)
            saved = self.saved_training_labels(meta)
            labels, distances, valid = saved if saved is not None else self.nearest(arrays)
            arrays.update(cluster_ids=labels, cluster_distances=distances, assignment_valid=valid,
                          temporal_train_mask=valid & arrays["flow_valid"])
            new_meta = dict(meta, cache_key=key, bank_identity=self.bank.identity, namespace=self.contract["namespace"],
                            encoder_input_components=self.bank.components, dense_saved=False,
                            assignment_source="exact_original_bank_training_assignments" if saved is not None else "frozen_role_routed_nearest_medoid",
                            conversion_schema=CONVERSION_SCHEMA, conversion_source_cache_key=row["cache_key"],
                            conversion_source_namespace=self.source["namespace"], conversion_source_npz_sha256=source_hash)
            new_meta["pairs"] = [dict(p, raw_path=None, processed_path=None) for p in pairs]
            self.validate_output(arrays, new_meta, key, source_hash)
            if not self.helper.save_npz_once(destination, metadata=np.array(json.dumps(new_meta, sort_keys=True, allow_nan=False)), **arrays):
                raise RuntimeError("Concurrent output write detected. Stop other converters and rerun.")
        return key, str(destination.relative_to(self.directory)), new_meta, hit, arrays["cluster_ids"], arrays["assignment_valid"], arrays["temporal_train_mask"]


def bounded_results(executor, function, iterable, maximum):
    iterator, pending = iter(iterable), deque()
    for _ in range(maximum):
        item = next(iterator, None)
        if item is None:
            break
        pending.append(executor.submit(function, item))
    while pending:
        yield pending.popleft().result()
        item = next(iterator, None)
        if item is not None:
            pending.append(executor.submit(function, item))


def build(options):
    helper, helper_path = load_helper(options.cache_script)
    core, _ = helper.load_core()
    source_index, source_manifest, source, count = load_source(options.source_index, helper)
    source_manifest_sha = helper.file_sha(source_index.parent / "manifest.json")
    bank_options = SimpleNamespace(expected_clusters=options.expected_clusters, window_len=source["window_len"],
                                   raft_weights=None, flow_device=None, device="cpu")
    bank = helper.Bank(options.bank_dir, core, bank_options)
    con, converter, run_id = None, None, None
    try:
        check_compatible(source_index, source, bank)
        root = options.cache_root.expanduser().resolve()
        if root == bank.directory or root in bank.directory.parents or bank.directory in root.parents:
            raise ValueError("Output cache root must not overlap the target bank.")
        if root == source_index.parent or source_index.parent in root.parents:
            raise ValueError("Output cache root must not be inside the source namespace.")
        recipe = dict(source)
        for obsolete in ("namespace", "bank_flow_config", "flow_backend_override"):
            recipe.pop(obsolete, None)
        recipe.update(bank_identity=bank.identity, bank_reference_sha256=bank.reference_sha,
                      num_prototypes=bank.count, normalization=bank.norm, descriptor=bank.desc,
                      weights=bank.manifest["weights"], flow_config=bank.flow_config,
                      bank_window_len=bank.bank_window_len, encoder_input_components=bank.components,
                      core_sha256=helper.file_sha(helper_path), implementation_sha256=helper.file_sha(__file__),
                      conversion_schema=CONVERSION_SCHEMA, conversion_source_contract_sha256=helper.digest(source))
        contract = json.loads(json.dumps(dict(recipe, namespace=helper.digest(recipe))))
        directory = root / helper.SCHEMA / contract["namespace"]
        if directory == source_index.parent:
            raise ValueError("Conversion output must differ from its source namespace.")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".conversion.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another converter is writing this output namespace.") from exc
            manifest_path = directory / "manifest.json"
            if manifest_path.exists() and read_json(manifest_path).get("contract") != contract:
                raise ValueError("Existing output contract differs; output preserved.")
            frozen = directory / "bank_references.npz"
            if not frozen.exists():
                shutil.copyfile(bank.reference_path, frozen)
            if helper.file_sha(frozen) != bank.reference_sha:
                raise ValueError("Output bank snapshot differs; output preserved.")
            helper.atomic_json(directory / "bank_contract.json", dict(bank_identity=bank.identity, manifest=bank.manifest))
            con = helper.init_index(directory / "index.sqlite", contract)
            run_id = uuid.uuid4().hex
            settings = dict(source_manifest.get("settings", {}), storage_mode="compact_only",
                            conversion_source_index=str(source_index), workers=options.workers)
            started = helper.now()
            con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?)", (run_id, started, None, "running", json.dumps(settings), None))
            con.commit()
            manifest = dict(schema=helper.SCHEMA, status="running", contract=contract, index="index.sqlite",
                            bank_directory=str(bank.directory), bank_manifest_sha256=helper.file_sha(bank.manifest_path),
                            source_builder=str(helper_path), raw_cache_directory=None, raw_cache_sources=[],
                            raw_flow_saved=False, dense_processed_pairs_saved=False, flow_backend_changed=False,
                            flow_estimator=bank.estimator_config["estimator"], last_run_id=run_id,
                            last_run_started_at=started, settings=settings, conversion_schema=CONVERSION_SCHEMA,
                            conversion_source_index=str(source_index), conversion_source_status=source_manifest["status"],
                            conversion_source_manifest_sha256=source_manifest_sha,
                            conversion_source_bank_identity=source["bank_identity"],
                            dataset_sampling_note="Exactly preserves source-index splits, views and starts; no missing coverage is generated.",
                            unsupported_role_policy="cluster_ids=-1, assignment_valid=False; never invent a different-role target",
                            training_encoder_note="Run the trainable temporal encoder live on encoder_inputs; cluster_ids are targets.")
            helper.atomic_json(manifest_path, manifest)
            print(f"Converting {count:,} cached views: {source['num_prototypes']} -> {bank.count} clusters", flush=True)
            print(f"Output: {directory / 'index.sqlite'}", flush=True)
            print("Saved summaries only; flow estimator calls=0; RGB decodes=0.", flush=True)
            stats = dict(cache_windows=0, window_hits=0, window_writes=0, original_training_label_windows=0,
                         unassigned_patches=0, supervised_patches=0, prototype_counts_by_split_view={},
                         flow_estimator_calls=0, rgb_decodes=0,
                         raw_flow=dict(hits=0, misses=0, writes=0, estimator_calls=0),
                         processed_pairs=dict(hits=0, writes=0))
            converter = Converter(source_index, source, bank, core, helper, directory, contract)
            began, last_report = time.monotonic(), time.monotonic()
            executor = ThreadPoolExecutor(max_workers=options.workers)
            try:
                with closing(helper.readonly(source_index)) as source_con:
                    source_con.execute("BEGIN")
                    rows = source_con.execute("SELECT * FROM windows ORDER BY split,video,window_start,augmentation")
                    for key, path, meta, hit, labels, valid, train_mask in bounded_results(executor, converter.convert, rows, options.workers * 2):
                        con.execute("INSERT INTO windows VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET updated_at=excluded.updated_at",
                                    (key, meta["split"], meta["video"], meta["window_start"], meta["target_index"], meta["augmentation"], path, helper.now()))
                        con.execute("INSERT OR IGNORE INTO run_windows VALUES (?,?)", (run_id, key))
                        stats["cache_windows"] += 1
                        stats["window_hits" if hit else "window_writes"] += 1
                        stats["original_training_label_windows"] += int(meta["assignment_source"] == "exact_original_bank_training_assignments")
                        stats["unassigned_patches"] += int(np.count_nonzero(~valid))
                        stats["supervised_patches"] += int(np.count_nonzero(train_mask))
                        category = meta["split"] + "/" + meta["augmentation"]
                        if category not in stats["prototype_counts_by_split_view"]:
                            stats["prototype_counts_by_split_view"][category] = np.zeros(bank.count, np.int64)
                        stats["prototype_counts_by_split_view"][category] += np.bincount(labels[valid], minlength=bank.count)
                        done = stats["cache_windows"]
                        if done % 100 == 0:
                            con.commit()
                        elapsed = time.monotonic() - began
                        if time.monotonic() - last_report >= 5 or done == count:
                            speed = done / max(elapsed, 1e-9)
                            print(f"Reassign cached views: {done:,}/{count:,} ({done/count:.1%}) | {speed:.1f} views/s | ETA {(count-done)/max(speed,1e-9)/60:.1f} min | reused {stats['window_hits']:,}", flush=True)
                            last_report = time.monotonic()
                executor.shutdown(wait=True)
                if helper.file_sha(source_index.parent / "manifest.json") != source_manifest_sha:
                    raise RuntimeError("Source manifest changed during conversion; stop its builder and rerun.")
                if helper.file_sha(bank.reference_path) != bank.reference_sha or helper.file_sha(bank.manifest_path) != manifest["bank_manifest_sha256"]:
                    raise RuntimeError("Target bank changed during conversion; output is incomplete.")
                if stats["cache_windows"] != count or con.execute("SELECT COUNT(*) FROM windows").fetchone()[0] != count:
                    raise RuntimeError("Converted index coverage differs from the source index.")
                if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise RuntimeError("Converted index failed its integrity check.")
                stats["prototype_counts_by_split_view"] = {k: v.tolist() for k, v in stats["prototype_counts_by_split_view"].items()}
                completed = helper.now()
                con.execute("UPDATE runs SET completed_at=?,status=?,stats_json=? WHERE id=?", (completed, "complete", json.dumps(stats, allow_nan=False), run_id))
                con.commit()
                manifest.update(status="complete", last_run_completed_at=completed, stats=stats)
                helper.atomic_json(manifest_path, manifest)
            except BaseException as exc:
                executor.shutdown(wait=True, cancel_futures=True)
                con.rollback()
                con.execute("UPDATE runs SET completed_at=?,status=?,stats_json=? WHERE id=?", (helper.now(), "failed", json.dumps(dict(error=str(exc))), run_id))
                con.commit()
                manifest.update(status="failed", error=str(exc), failed_at=helper.now())
                helper.atomic_json(manifest_path, manifest)
                raise
            print(f"Complete: {directory / 'index.sqlite'}", flush=True)
            print(f"Train with --temporal-cache-dir {directory} --temporal-cluster-dir {bank.directory} --temporal-expected-clusters {bank.count}", flush=True)
            return directory, manifest
    finally:
        if converter is not None:
            converter.close()
        if con is not None:
            con.close()
        bank.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-script", type=Path, default=Path(__file__).with_name("cache_optical_flow.py"))
    parser.add_argument("--expected-clusters", type=int)
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args(argv)
    if options.workers < 1:
        parser.error("--workers must be positive.")
    return options


def main():
    try:
        build(parse_args())
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
