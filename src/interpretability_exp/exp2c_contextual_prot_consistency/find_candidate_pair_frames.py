#!/usr/bin/env python3
"""Find frames where two activation signatures co-occur on distinct patches.

Single-pair mode (--patch-sign-a / --patch-sign-b):
  Scan patch_archive.bin; write OUTPUT_DIR/{patch_a}_{patch_b}/frame_data.csv with video,
  frame, patch indices, and predicted/GT saliency from --patch-data.
  Signs are comma-separated prototype IDs (e.g. 12,45,67,89,120) or S000042 with
  --activation-signatures.

Bulk mode (default):
  Read high/low opposing rows from candidate_pairs.csv, export patch-level witnesses and
  optional saliency from patch_data.json to candidate_pair_sal_data.csv.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

PatchKey = Tuple[str, str, int]  # video, frame id string, patch_index

_gcp = None  # set in main for format_activation_sign


def _load_generate_module(script_dir: Path):
    path = script_dir / "generate_candidate_pairs.py"
    spec = importlib.util.spec_from_file_location("generate_candidate_pairs", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load generate_candidate_pairs.py from {}.".format(script_dir))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_signature_index(signature_id: str) -> int:
    token = signature_id.strip()
    if not token.startswith("S") or not token[1:].isdigit():
        raise ValueError("Invalid signature_id: {}.".format(signature_id))
    return int(token[1:]) - 1


def prototype_ids_to_bits(prototype_ids: str) -> int:
    ids = [int(x.strip()) for x in prototype_ids.split(",") if x.strip()]
    if not ids:
        raise ValueError("Empty prototype_ids.")
    return sum(1 << i for i in ids)


def parse_patch_sign(label: str, text: str, bits_by_index: Optional[Dict[int, int]]) -> Tuple[int, str]:
    """Resolve --patch-sign-* to (signature bits, canonical comma-separated prototype ids)."""
    token = text.strip()
    if token.startswith("S"):
        if not bits_by_index:
            raise ValueError("{}: signature id {} requires --activation-signatures."
                             .format(label, token))
        index = parse_signature_index(token)
        if index not in bits_by_index:
            raise ValueError("{}: {} not found in activation_signatures.csv.".format(label, token))
        bits = bits_by_index[index]
    else:
        bits = prototype_ids_to_bits(token)
    return bits, format_activation_sign(bits)


def format_activation_sign(bits: int) -> str:
    return ",".join(str(x) for x in _gcp.signature_ids(bits))


def load_signature_bits(path: Path) -> Dict[int, int]:
    bits_by_index: Dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            index = parse_signature_index(row["signature_id"])
            bits_by_index[index] = prototype_ids_to_bits(row["prototype_ids"])
    if not bits_by_index:
        raise ValueError("No signatures in {}.".format(path))
    return bits_by_index


def is_opposing_high_low(origin_a: str, origin_b: str) -> bool:
    a, b = origin_a.strip(), origin_b.strip()
    return (a == "high" and b == "low") or (a == "low" and b == "high")


def load_opposing_pairs(path: Path, bits_by_index: Dict[int, int]
                        ) -> List[Tuple[int, int]]:
    """Return (high_sig_index, low_sig_index) per CSV row."""
    pairs: List[Tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not is_opposing_high_low(row["discovery_origin_a"], row["discovery_origin_b"]):
                continue
            idx_a = parse_signature_index(row["signature_id_a"])
            idx_b = parse_signature_index(row["signature_id_b"])
            if idx_a not in bits_by_index or idx_b not in bits_by_index:
                raise ValueError("Missing prototype definition for pair {} / {}."
                                 .format(row["signature_id_a"], row["signature_id_b"]))
            if row["discovery_origin_a"].strip() == "high":
                pairs.append((idx_a, idx_b))
            else:
                pairs.append((idx_b, idx_a))
    if not pairs:
        raise ValueError("No high/low opposing pairs in {}.".format(path))
    return pairs


@dataclass(frozen=True)
class PairIndex:
    """Opposing high/low pairs indexed for per-frame lookup (not O(#pairs) per frame)."""
    pairs_by_hi: Dict[int, Tuple[int, ...]]
    sign_str: Dict[int, str]
    sig_bits: Dict[int, int]


def build_single_pair_index(bits_a: int, bits_b: int, sign_a: str, sign_b: str) -> PairIndex:
    """PairIndex for one unordered signature pair (internal keys 0 and 1)."""
    return PairIndex(
        pairs_by_hi={0: (1,)},
        sign_str={0: sign_a, 1: sign_b},
        sig_bits={0: bits_a, 1: bits_b},
    )


def build_pair_index(opposing_pairs: List[Tuple[int, int]], bits_by_index: Dict[int, int]) -> PairIndex:
    by_hi: Dict[int, List[int]] = defaultdict(list)
    used: Set[int] = set()
    for hi_idx, lo_idx in opposing_pairs:
        by_hi[hi_idx].append(lo_idx)
        used.add(hi_idx)
        used.add(lo_idx)
    pairs_by_hi = {hi: tuple(los) for hi, los in by_hi.items()}
    sign_str = {idx: format_activation_sign(bits_by_index[idx]) for idx in used}
    sig_bits = {idx: bits_by_index[idx] for idx in used}
    return PairIndex(pairs_by_hi=pairs_by_hi, sign_str=sign_str, sig_bits=sig_bits)


def signature_on_patch(sig_bits: int, txn_bits: int) -> bool:
    return (sig_bits & txn_bits) == sig_bits


def sig_patches_for_frame(
        patches: Sequence[Tuple[int, int]],
        transaction_bits: Sequence[int],
        sig_bits: Dict[int, int],
) -> Dict[int, Set[int]]:
    """Map signature index -> patch ids matching in this frame (one pass over distinct gids)."""
    by_gid: Dict[int, Set[int]] = {}
    for patch_id, gid in patches:
        by_gid.setdefault(gid, set()).add(patch_id)
    sig_patches: Dict[int, Set[int]] = {}
    for gid, patch_ids in by_gid.items():
        txn = transaction_bits[gid]
        for sig_idx, bits in sig_bits.items():
            if signature_on_patch(bits, txn):
                bucket = sig_patches.get(sig_idx)
                if bucket is None:
                    sig_patches[sig_idx] = set(patch_ids)
                else:
                    bucket.update(patch_ids)
    return sig_patches


def rows_for_frame(
        video_fname: str,
        frame_idx: str,
        patches: Sequence[Tuple[int, int]],
        transaction_bits: Sequence[int],
        pair_index: PairIndex,
) -> List[dict]:
    sig_patches = sig_patches_for_frame(patches, transaction_bits, pair_index.sig_bits)
    if len(sig_patches) < 2:
        return []
    rows: List[dict] = []
    sign_str = pair_index.sign_str
    for hi_idx, lo_indices in pair_index.pairs_by_hi.items():
        hi_patches = sig_patches.get(hi_idx)
        if not hi_patches:
            continue
        hi_sign = sign_str[hi_idx]
        for lo_idx in lo_indices:
            lo_patches = sig_patches.get(lo_idx)
            if not lo_patches:
                continue
            lo_sign = sign_str[lo_idx]
            for hi_patch in hi_patches:
                for lo_patch in lo_patches:
                    if hi_patch == lo_patch:
                        continue
                    rows.append({
                        "activation_sign_1": hi_sign,
                        "activation_sign_2": lo_sign,
                        "patch_idx_1": hi_patch,
                        "patch_idx_2": lo_patch,
                        "frame_idx": frame_idx,
                        "video_fname": video_fname,
                        "_patch_key_1": (video_fname, frame_idx, hi_patch),
                        "_patch_key_2": (video_fname, frame_idx, lo_patch),
                    })
    return rows


# --- multiprocessing (archive scan) ---

_scan_worker_ctx: dict = {}


def _init_scan_worker(ctx: dict) -> None:
    global _scan_worker_ctx
    loaded = dict(ctx)
    loaded["gcp"] = _load_generate_module(Path(ctx["script_dir"]))
    _scan_worker_ctx = loaded


def _scan_chunk(task: Tuple[Path, List[Tuple[int, int, int]]]) -> Tuple[List[dict], int]:
    archive_path, frame_specs = task
    ctx = _scan_worker_ctx
    number_to_key = ctx["number_to_key"]
    transaction_bits = ctx["transaction_bits"]
    pair_index = ctx["pair_index"]
    gcp = ctx["gcp"]

    rows: List[dict] = []
    archive = gcp.PatchArchive(archive_path)
    with archive_path.open("rb") as handle:
        for _fn, _vi, offset in frame_specs:
            frame_number, _video_index, patches = archive.read_frame(handle, offset)
            key = number_to_key.get(frame_number)
            if key is None:
                continue
            video_fname, frame_idx = key[0], key[1]
            rows.extend(rows_for_frame(
                video_fname, frame_idx, patches, transaction_bits, pair_index))
    return rows, len(frame_specs)


def _task_chunk_size(n_frames: int, workers: int) -> int:
    target_tasks = max(workers * 32, workers)
    return max(128, min(2048, (n_frames + target_tasks - 1) // target_tasks))


def collect_patch_pair_rows(
        ingest,
        gcp,
        pair_index: PairIndex,
        args,
) -> List[dict]:
    plan = ingest.plan
    number_to_key = {scan.number: key for key, scan in plan.frames.items()}
    transaction_bits = tuple(ingest.global_index.transaction_bits)
    frame_specs = ingest.archive.frame_index
    workers = max(1, int(getattr(args, "workers", 1)))

    if workers == 1:
        rows: List[dict] = []
        frame_iter = gcp.tqdm_wrap(frame_specs, "Scan archive for patch pairs", args,
                                   unit="frame", total=len(frame_specs))
        archive = ingest.archive
        with archive.path.open("rb") as handle:
            for _frame_number, _video_bit, offset in frame_iter:
                frame_number, _video_index, patches = archive.read_frame(handle, offset)
                key = number_to_key.get(frame_number)
                if key is None:
                    continue
                rows.extend(rows_for_frame(key[0], key[1], patches, transaction_bits, pair_index))
        return rows

    chunk_size = _task_chunk_size(len(frame_specs), workers)
    tasks = []
    for start in range(0, len(frame_specs), chunk_size):
        tasks.append((ingest.archive.path, frame_specs[start:start + chunk_size]))

    ctx = {
        "script_dir": str(Path(__file__).resolve().parent),
        "number_to_key": number_to_key,
        "transaction_bits": transaction_bits,
        "pair_index": pair_index,
    }
    print("  Archive scan: {:,} frames, {:,} workers, ~{:,} frames/task ({} tasks)."
          .format(len(frame_specs), workers, chunk_size, len(tasks)), flush=True)

    rows: List[dict] = []
    pbar = gcp.tqdm_create("Scan archive for patch pairs", args, unit="frame", total=len(frame_specs))
    with mp.Pool(workers, initializer=_init_scan_worker, initargs=(ctx,)) as pool:
        for chunk_rows, done in pool.imap_unordered(_scan_chunk, tasks):
            rows.extend(chunk_rows)
            if pbar is not None:
                pbar.update(done)
                pbar.set_postfix(rows=len(rows), refresh=False)
    if pbar is not None:
        pbar.close()
    return rows


# --- saliency join (optional parallel partition by video) ---

_saliency_worker_ctx: dict = {}


def _init_saliency_worker(ctx: dict) -> None:
    global _saliency_worker_ctx
    loaded = dict(ctx)
    loaded["gcp"] = _load_generate_module(Path(ctx["script_dir"]))
    _saliency_worker_ctx = loaded


def _saliency_scan_chunk(keys: Set[PatchKey]) -> Dict[PatchKey, Tuple[float, float]]:
    ctx = _saliency_worker_ctx
    gcp = ctx["gcp"]
    patch_data = ctx["patch_data"]
    pred_field = ctx["pred_field"]
    gt_field = ctx["gt_field"]
    videos_needed = ctx["videos_needed"]

    saliency: Dict[PatchKey, Tuple[float, float]] = {}
    resolver = None
    for record in gcp.iter_patch_records(Path(patch_data)):
        if resolver is None:
            resolver = gcp.FrameResolver(record, ctx.get("video_field"), ctx.get("frame_field"))
        key_frame = resolver.key(record)
        if key_frame[0] not in videos_needed:
            continue
        patch_index = gcp.integer(record["patch_index"], "patch_index")
        patch_key: PatchKey = (key_frame[0], key_frame[1], patch_index)
        if patch_key not in keys:
            continue
        saliency[patch_key] = (
            gcp.saliency(record, pred_field),
            gcp.saliency(record, gt_field),
        )
        if len(saliency) == len(keys):
            break
    return saliency


def load_saliency_for_keys(
        patch_data: Path,
        gcp,
        keys: Set[PatchKey],
        pred_field: str,
        gt_field: str,
        args,
) -> Dict[PatchKey, Tuple[float, float]]:
    workers = max(1, int(getattr(args, "workers", 1)))
    if workers == 1 or len(keys) < 50_000:
        return _load_saliency_single(patch_data, gcp, keys, pred_field, gt_field, args)

    key_list = sorted(keys)
    chunk_count = min(workers, max(1, len(key_list) // 10_000))
    chunks: List[Set[PatchKey]] = [set() for _ in range(chunk_count)]
    for index, key in enumerate(key_list):
        chunks[index % chunk_count].add(key)

    videos_needed = {k[0] for k in keys}
    ctx = {
        "script_dir": str(Path(__file__).resolve().parent),
        "patch_data": str(patch_data),
        "pred_field": pred_field,
        "gt_field": gt_field,
        "video_field": args.video_field,
        "frame_field": args.frame_field,
        "videos_needed": videos_needed,
    }
    print("  Saliency join: {:,} patch keys, {:,} JSON workers.".format(len(keys), chunk_count), flush=True)
    merged: Dict[PatchKey, Tuple[float, float]] = {}
    with mp.Pool(chunk_count, initializer=_init_saliency_worker, initargs=(ctx,)) as pool:
        for part in pool.imap_unordered(_saliency_scan_chunk, chunks):
            merged.update(part)
    missing = keys - merged.keys()
    if missing:
        sample = next(iter(missing))
        raise ValueError("Missing saliency for {:,} patches (example: {}).".format(len(missing), sample))
    return merged


def _load_saliency_single(
        patch_data: Path,
        gcp,
        keys: Set[PatchKey],
        pred_field: str,
        gt_field: str,
        args,
) -> Dict[PatchKey, Tuple[float, float]]:
    saliency: Dict[PatchKey, Tuple[float, float]] = {}
    records = gcp.iter_patch_records(patch_data)
    record_iter = gcp.tqdm_wrap(records, "Join patch saliency", args, unit="patch")
    resolver = None
    for record in record_iter:
        if resolver is None:
            resolver = gcp.FrameResolver(record, args.video_field, args.frame_field)
        key_frame = resolver.key(record)
        if not resolver.has_patch_index:
            raise ValueError("patch_data.json must include patch_index for this export.")
        patch_index = gcp.integer(record["patch_index"], "patch_index")
        patch_key: PatchKey = (key_frame[0], key_frame[1], patch_index)
        if patch_key not in keys:
            continue
        saliency[patch_key] = (
            gcp.saliency(record, pred_field),
            gcp.saliency(record, gt_field),
        )
        if len(saliency) == len(keys):
            break
    missing = keys - saliency.keys()
    if missing:
        sample = next(iter(missing))
        raise ValueError("Missing saliency for {:,} patches (example: {}).".format(len(missing), sample))
    return saliency


def distinct_frames(rows: List[dict]) -> int:
    return len({(row["video_fname"], row["frame_idx"]) for row in rows})


def patch_sign_dir_label(raw_sign: str) -> str:
    """Filesystem-safe directory segment from --patch-sign-* (keeps S000042; protos use dashes)."""
    token = raw_sign.strip()
    if token.startswith("S") and token[1:].isdigit():
        return token
    label = token.replace(" ", "").replace(",", "-")
    for char in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        label = label.replace(char, "_")
    if not label:
        raise ValueError("Empty patch sign label.")
    return label


def single_pair_frame_data_path(output_dir: Path, raw_sign_a: str, raw_sign_b: str) -> Path:
    folder = "{}_{}".format(patch_sign_dir_label(raw_sign_a), patch_sign_dir_label(raw_sign_b))
    return output_dir / folder / "frame_data.csv"


def _patch_saliencies(row: dict, saliency: Dict[PatchKey, Tuple[float, float]]
                      ) -> Tuple[float, float, float, float]:
    pred_a, gt_a = saliency[row["_patch_key_1"]]
    pred_b, gt_b = saliency[row["_patch_key_2"]]
    return pred_a, gt_a, pred_b, gt_b


def print_cooccurrence_list(rows: List[dict], sign_a: str, sign_b: str, limit: int,
                            saliency: Dict[PatchKey, Tuple[float, float]]) -> None:
    frames = distinct_frames(rows)
    print("\nSignature A: {}".format(sign_a), flush=True)
    print("Signature B: {}".format(sign_b), flush=True)
    print("Distinct frames with both signatures on separate patches: {:,}.".format(frames), flush=True)
    print("Patch witness rows (video, frame, patch_a, patch_b): {:,}.".format(len(rows)), flush=True)
    if not rows:
        return
    print("\nvideo_fname  frame_idx  patch_idx_a  patch_idx_b  pred_sal_a  pred_sal_b  gt_sal_a  gt_sal_b",
          flush=True)
    stop = len(rows) if limit == 0 else min(limit, len(rows))
    for row in rows[:stop]:
        pred_a, gt_a, pred_b, gt_b = _patch_saliencies(row, saliency)
        print("{}  {}  {}  {}  {:g}  {:g}  {:g}  {:g}".format(
            row["video_fname"], row["frame_idx"], row["patch_idx_1"], row["patch_idx_2"],
            pred_a, pred_b, gt_a, gt_b), flush=True)
    if limit and len(rows) > limit:
        print("  Showing {:,} of {:,} rows; use --report-limit 0 for all.".format(limit, len(rows)), flush=True)


def write_frame_data_csv(path: Path, rows: List[dict],
                         saliency: Dict[PatchKey, Tuple[float, float]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError("Refusing to overwrite {}.".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "video_fname", "frame_idx", "patch_idx_a", "patch_idx_b",
        "pred_sal_a", "pred_sal_b", "gt_sal_a", "gt_sal_b",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pred_a, gt_a, pred_b, gt_b = _patch_saliencies(row, saliency)
            writer.writerow({
                "video_fname": row["video_fname"],
                "frame_idx": row["frame_idx"],
                "patch_idx_a": row["patch_idx_1"],
                "patch_idx_b": row["patch_idx_2"],
                "pred_sal_a": pred_a,
                "pred_sal_b": pred_b,
                "gt_sal_a": gt_a,
                "gt_sal_b": gt_b,
            })


def write_output(path: Path, rows: List[dict], saliency: Dict[PatchKey, Tuple[float, float]],
                 overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError("Refusing to overwrite {}.".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "activation_sign_1", "activation_sign_2",
        "patch_idx_1", "patch_idx_2", "frame_idx", "video_fname",
        "pred_sal", "gt_sal",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pred1, gt1 = saliency[row["_patch_key_1"]]
            pred2, gt2 = saliency[row["_patch_key_2"]]
            writer.writerow({
                "activation_sign_1": row["activation_sign_1"],
                "activation_sign_2": row["activation_sign_2"],
                "patch_idx_1": row["patch_idx_1"],
                "patch_idx_2": row["patch_idx_2"],
                "frame_idx": row["frame_idx"],
                "video_fname": row["video_fname"],
                "pred_sal": "{:g};{:g}".format(pred1, pred2),
                "gt_sal": "{:g};{:g}".format(gt1, gt2),
            })


def parse_args(argv=None):
    default_workers = max(1, (os.cpu_count() or 4) - 1)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch-sign-a", default=None,
                        help="First signature: comma-separated prototype IDs or S000042 with "
                             "--activation-signatures.")
    parser.add_argument("--patch-sign-b", default=None,
                        help="Second signature (same format as --patch-sign-a).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output root. Single-pair writes OUTPUT_DIR/{patch_a}_{patch_b}/frame_data.csv.")
    parser.add_argument("--frame-data-out", type=Path, default=None,
                        help="Single-pair: override default frame_data.csv path.")
    parser.add_argument("--candidate-pairs", type=Path, default=None,
                        help="Default: OUTPUT_DIR/candidate_pairs.csv")
    parser.add_argument("--activation-signatures", type=Path, default=None,
                        help="Default: OUTPUT_DIR/activation_signatures.csv")
    parser.add_argument("--patch-cache", type=Path, required=True,
                        help="patch_archive.bin path (.meta sidecar required).")
    parser.add_argument("--patch-data", type=Path, default=None,
                        help="patch_data.json for predicted/GT saliency (required in both modes).")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help="CPU workers for archive scan and large saliency joins (default: CPU count - 1).")
    parser.add_argument("--report-limit", type=int, default=50,
                        help="Single-pair stdout rows; 0 prints all witness rows.")
    parser.add_argument("--pred-saliency-field", default="patch_pred_saliency")
    parser.add_argument("--gt-saliency-field", default="patch_gt_saliency")
    parser.add_argument("--video-field", default=None)
    parser.add_argument("--frame-field", default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1.")
    if args.report_limit < 0:
        parser.error("--report-limit must be >= 0.")
    single = args.patch_sign_a is not None or args.patch_sign_b is not None
    if single and (args.patch_sign_a is None or args.patch_sign_b is None):
        parser.error("Single-pair mode requires both --patch-sign-a and --patch-sign-b.")
    if args.patch_data is None:
        parser.error("--patch-data is required.")
    if args.output_dir is None:
        parser.error("--output-dir is required.")
    args.single_pair_mode = single
    return args


def _open_ingest(gcp, args):
    meta_path = Path(str(args.patch_cache) + ".meta")
    if not meta_path.is_file():
        raise ValueError("Missing ingest metadata: {}.".format(meta_path))
    ingest = gcp.load_ingest_cache(meta_path)
    if ingest.archive.path.resolve() != args.patch_cache.resolve():
        print("Note: meta archive_path is {}; using --patch-cache {}."
              .format(ingest.archive.path, args.patch_cache), flush=True)
        ingest.archive = gcp.PatchArchive.open(args.patch_cache)
    return ingest


def run_single_pair(args, gcp) -> int:
    output_dir = args.output_dir
    sig_path = args.activation_signatures
    if sig_path is None and output_dir is not None:
        sig_path = output_dir / "activation_signatures.csv"
    bits_by_index: Optional[Dict[int, int]] = None
    if sig_path is not None and sig_path.is_file():
        bits_by_index = load_signature_bits(sig_path)

    bits_a, sign_a = parse_patch_sign("--patch-sign-a", args.patch_sign_a, bits_by_index)
    bits_b, sign_b = parse_patch_sign("--patch-sign-b", args.patch_sign_b, bits_by_index)
    pair_index = build_single_pair_index(bits_a, bits_b, sign_a, sign_b)

    ingest = _open_ingest(gcp, args)
    rows = collect_patch_pair_rows(ingest, gcp, pair_index, args)
    if not rows:
        # print_cooccurrence_list(rows, sign_a, sign_b, args.report_limit, {})
        return 0

    keys: Set[PatchKey] = set()
    for row in rows:
        keys.add(row["_patch_key_1"])
        keys.add(row["_patch_key_2"])
    saliency = load_saliency_for_keys(
        args.patch_data, gcp, keys, args.pred_saliency_field, args.gt_saliency_field, args)

    # print_cooccurrence_list(rows, sign_a, sign_b, args.report_limit, saliency)

    if args.frame_data_out is not None:
        out_path = args.frame_data_out
    else:
        out_path = single_pair_frame_data_path(output_dir, args.patch_sign_a, args.patch_sign_b)
    write_frame_data_csv(out_path, rows, saliency, args.overwrite)
    print("Wrote {:,} rows to {}.".format(len(rows), out_path), flush=True)
    return 0


def run_bulk_export(args, gcp) -> int:
    output_dir = args.output_dir
    pairs_path = args.candidate_pairs or (output_dir / "candidate_pairs.csv")
    sig_path = args.activation_signatures or (output_dir / "activation_signatures.csv")

    bits_by_index = load_signature_bits(sig_path)
    opposing_pairs = load_opposing_pairs(pairs_path, bits_by_index)
    pair_index = build_pair_index(opposing_pairs, bits_by_index)
    print("Loaded {:,} high/low opposing signature pairs ({:,} unique high signatures)."
          .format(len(opposing_pairs), len(pair_index.pairs_by_hi)), flush=True)

    ingest = _open_ingest(gcp, args)
    rows = collect_patch_pair_rows(ingest, gcp, pair_index, args)
    print("Found {:,} high-vs-low patch witness pairs in archive frames.".format(len(rows)), flush=True)
    if not rows:
        return 0

    keys: Set[PatchKey] = set()
    for row in rows:
        keys.add(row["_patch_key_1"])
        keys.add(row["_patch_key_2"])

    saliency = load_saliency_for_keys(
        args.patch_data, gcp, keys, args.pred_saliency_field, args.gt_saliency_field, args)

    out_path = output_dir / "candidate_pair_sal_data.csv"
    write_output(out_path, rows, saliency, args.overwrite)
    print("Wrote {:,} rows to {}.".format(len(rows), out_path), flush=True)
    return 0


def main(argv=None) -> int:
    global _gcp
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    gcp = _load_generate_module(script_dir)
    _gcp = gcp

    if args.single_pair_mode:
        return run_single_pair(args, gcp)
    return run_bulk_export(args, gcp)


if __name__ == "__main__":
    raise SystemExit(main())
