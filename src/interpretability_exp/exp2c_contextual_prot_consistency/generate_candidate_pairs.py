#!/usr/bin/env python3
r"""Mine recurring prototype subsets and report same-frame candidate frequencies.

Discovery:
  1. Select the top 3% and bottom 50% of predicted saliency within each frame.
     Use --percentile-scope dataset for pooled dataset percentiles instead.
  2. Independently mine recurring five-ID subsets of the patches' top-20 IDs
     in each band. Recurring means present in at least two selected patches
     (--min-subset-patches) AND at least --min-frame-occurrence distinct physical
     frames globally (--min-frame-occurrence, default 1). Discovery patch support
     counts selected-band PATCHES only; global frame support counts distinct
     (video, frame) identifiers across ALL patches where the full subset appears
     together on one patch's top-20 list. Both checks run inside mining recursion.
  3. Deduplicate the union into potential_candidates. A signature found in
     both bands has one ID and retains both discovery origins. A patch matches
     EVERY mined subset contained in its top-20 list; membership is not exclusive.
  4. Pass 3 walks physical frames: match every patch top-20 against all retained
     signatures, then count only signature pairs actually observed together in
     a frame (unordered pairs of signatures present on distinct patches unless
     both are sole witnesses of the same patch). Each valid pair gains at most
     one count per frame. high/high, low/low, and cross-origin pairs are allowed.
     Only observed pairs are stored; the global N*(N-1)/2 space is never enumerated.

Use --min-frame-occurrence (default 1) during subset mining (global frame
support) and again when reporting valid signature-pair co-occurrence frames.
Optional --output-dir writes candidate_pairs.csv and activation_signatures.csv
for signature pairs whose valid same-frame co-occurrence count meets
--min-frame-occurrence. These are discovery frequencies, not saliency-ordering
results.
Discovery origins do not restrict the saliency of later matching patches.

All recurring subsets are retained by default. --max-signatures-per-band N
optionally keeps the N most frequent subsets from EACH band, with deterministic
ties. A nonzero limit changes the candidate pool and is reported explicitly.
--report-limit limits PRINTED rows only; it never changes mining or counting.
Use --report-limit 0 to print every signature and positive-frequency pair.
Frequency summaries cover observed pairs; unobserved combinations are reported
only as an aggregate subtracted from the theoretical total.

The five-ID signature is a 512-position binary vector represented compactly as
an integer; as_binary_vector() exposes the vector. The input must contain exactly
20 distinct valid prototype IDs per patch. Short exports are rejected, not padded.
--overlap-percent 25 is retained as an alias for a five-ID SUBSET SIZE; it no
longer means pairwise overlap clustering. --subset-size overrides the default.

Percentile counts use ceil(percent * population / 100). By default, cutoff ties
are broken by input order; --tie-policy include keeps all cutoff ties. Scores at
overlapping high/low cutoffs are excluded from discovery (including flat frames).
Their patches remain eligible when counting occurrences of discovered signatures.

Implementation: one JSON ingest (ijson when installed) writing a binary patch archive;
weighted vertical mining; inverted top-20->signature table; pair counting replays
the archive only (optional --patch-cache reuse; --pair-workers for parallel frames).
Python 3.8+, standard library plus optional ijson.

Example (counts only):
  python generate_candidiate_pairs.py --patch-data /path/to/patch_data.json
  python generate_candidiate_pairs.py --patch-data /path/to/patch_data.json \
      --percentile-scope dataset --report-limit 0
"""
from __future__ import annotations

import argparse
from array import array
import csv
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
from functools import lru_cache
import heapq
import json
import math
import multiprocessing as mp
import pickle
from pathlib import Path
import shelve
import struct
import sys
import tempfile
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union


T = TypeVar("T")


_TQDM_BAR = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"


def tqdm_wrap(iterable: Iterable[T], desc: str, args, unit: str = "it",
              total: Optional[int] = None) -> Union[Iterable[T], Iterator[T]]:
    """Progress bar for major passes; no-op when --no-tqdm or tqdm is missing."""
    if getattr(args, "no_tqdm", False):
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        dynamic_ncols=True,
        smoothing=0.05,
        bar_format=_TQDM_BAR,
    )


def tqdm_create(desc: str, args, unit: str = "it", total: Optional[int] = None):
    """Manual tqdm bar (for updates that are not one-per-iteration)."""
    if getattr(args, "no_tqdm", False):
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        smoothing=0.05,
        bar_format=_TQDM_BAR,
    )


NUM_PROTOTYPES = 512
EXPORT_TOP_K = 20  # patch_data.json ranked slot count (top_20_* fields)
# Cover essentially all distinct activation lists in typical exports (~20k) without unbounded RAM.
DEFAULT_MATCH_CACHE_SIZE = 32768
FrameKey = Tuple[str, str]
RecordFactory = Callable[[], Iterable[dict]]


def popcount(bits: int) -> int:
    method = getattr(int, "bit_count", None)
    return method(bits) if method is not None else bin(bits).count("1")


def signature_ids(bits: int) -> Tuple[int, ...]:
    ids = []
    while bits:
        lowest = bits & -bits
        ids.append(lowest.bit_length() - 1)
        bits ^= lowest
    return tuple(ids)


def as_binary_vector(bits: int) -> Tuple[bool, ...]:
    """Expose the exact 512-position representation without storing padded arrays."""
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 0 or bits.bit_length() > NUM_PROTOTYPES:
        raise ValueError("Signature must be a nonnegative 512-bit integer.")
    return tuple(bool(bits & (1 << i)) for i in range(NUM_PROTOTYPES))


def iter_json_array(handle, chunk_size: int = 1 << 20) -> Iterable[dict]:
    """Stream a JSON array, including pretty-printed exports, without loading it."""
    decoder = json.JSONDecoder()
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
                raise ValueError("Expected a comma between patch objects.")
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


def iter_patch_records(path: Path) -> Iterable[dict]:
    """Stream patch objects; use ijson when installed for faster incremental parsing."""
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None
    if ijson is not None:
        with path.open("rb") as handle:
            for item in ijson.items(handle, "item"):
                if not isinstance(item, dict):
                    raise ValueError("Each patch entry must be a JSON object.")
                yield item
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        yield from iter_json_array(handle)


def integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}.".format(label, minimum))
    return value


def detail_prototype_index(item: dict) -> int:
    values = [integer(item[k], "prototype index") for k in ("prototype_index", "prot_idx") if k in item]
    if not values:
        raise ValueError("Prototype detail must contain prototype_index or prot_idx.")
    if len(set(values)) != 1:
        raise ValueError("prototype_index and prot_idx disagree.")
    return values[0]


def saliency(record: dict, field_name: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("{} must be a finite number.".format(field_name))
    return float(value)


def activation_signature(record: dict, top_k: int) -> int:
    """Read the first top_k ranked prototype IDs (from up to 20 exported slots)."""
    if not 1 <= top_k <= EXPORT_TOP_K:
        raise ValueError("top_k must be in [1, {}].".format(EXPORT_TOP_K))
    ids = record.get("top_20_activated_prototype_indices")
    details = record.get("top_20_activated_prototypes")
    if details is not None:
        if not isinstance(details, list) or any(not isinstance(x, dict) for x in details):
            raise ValueError("top_20_activated_prototypes must be a list of objects.")
        if any("rank" in x for x in details):
            ranks = [integer(x.get("rank"), "prototype rank", 1) for x in details]
            if len(set(ranks)) != len(ranks):
                raise ValueError("Duplicate prototype ranks in one patch.")
            details = sorted(details, key=lambda x: integer(x["rank"], "prototype rank", 1))
        extracted = [detail_prototype_index(x) for x in details]
        if ids is not None:
            if not isinstance(ids, list) or [integer(x, "prototype index") for x in ids] != extracted:
                raise ValueError("Ranked prototype details and index list disagree.")
        ids = extracted
    if not isinstance(ids, list):
        raise ValueError("Missing exported top-20 prototype list.")
    ids = [integer(x, "prototype index") for x in ids]
    if len(ids) < top_k:
        raise ValueError("Expected at least {:,} ranked prototype IDs, found {:,}. "
                         "This script does not pad short lists.".format(top_k, len(ids)))
    ids = ids[:top_k]
    if len(set(ids)) != top_k:
        raise ValueError("Expected {:,} distinct prototype IDs in the top-{:,} ranks, found {:,} distinct."
                         .format(top_k, top_k, len(set(ids))))
    if any(x >= NUM_PROTOTYPES for x in ids):
        raise ValueError("Prototype indices must be in [0, 511].")
    return sum(1 << x for x in ids)


class FrameResolver:
    """Identify physical source frames and prevent mixing stages/window contexts."""
    def __init__(self, record: dict, video_field=None, frame_field=None):
        self.video_field = video_field or next(
            (x for x in ("video_fname", "video_id", "video_name") if x in record), None)
        self.frame_field = frame_field or next(
            (x for x in ("absolute_frame_index", "frame_filename", "frame_no") if x in record), None)
        if self.video_field is None or self.frame_field is None:
            raise ValueError("Patch records need video and frame identifiers.")
        self.context_fields = tuple(x for x in
            ("dataset_index", "window_start_index", "window_length") if x in record)
        self.has_stage = "stage" in record
        self.stage = record.get("stage")
        self.has_patch_index = "patch_index" in record

    def key(self, record: dict) -> FrameKey:
        if ("stage" in record) != self.has_stage or record.get("stage") != self.stage:
            raise ValueError("Input mixes prototype stages; analyze one export/stage at a time.")
        values = []
        for name in (self.video_field, self.frame_field):
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
                raise ValueError("{} must be a nonempty video/frame identifier.".format(name))
            values.append(str(value))
        return values[0], values[1]

    def context(self, record: dict) -> tuple:
        values = []
        for name in self.context_fields:
            if name not in record or isinstance(record[name], (dict, list)):
                raise ValueError("Missing or non-scalar frame-context field {}.".format(name))
            values.append(record[name])
        return tuple(values)


@dataclass
class FrameScan:
    context: tuple
    number: int
    count: int = 0
    patch_bits: int = 0


def percent_count(count: int, percent: float) -> int:
    return int((Decimal(str(percent)) * count / Decimal(100)).to_integral_value(rounding=ROUND_CEILING))


@dataclass
class BandBoundary:
    high_cutoff: float
    low_cutoff: float
    high_requested: int
    low_requested: int
    high_ties: int
    low_ties: int
    high_ties_keep: int
    low_ties_keep: int
    high_ties_seen: int = 0
    low_ties_seen: int = 0

    @classmethod
    def from_scores(cls, scores, top_percent: float, bottom_percent: float):
        values = sorted(scores)
        n = len(values)
        high_n, low_n = percent_count(n, top_percent), percent_count(n, bottom_percent)
        high, low = values[n - high_n], values[low_n - 1]
        high_left, high_right = bisect_left(values, high), bisect_right(values, high)
        low_left, low_right = bisect_left(values, low), bisect_right(values, low)
        return cls(high, low, high_n, low_n, high_right - high_left,
                   low_right - low_left, high_n - (n - high_right), low_n - low_left)

    def classify(self, score: float, tie_policy: str) -> Optional[str]:
        if self.high_cutoff <= score <= self.low_cutoff:
            return "ambiguous"
        if tie_policy == "include":
            high, low = score >= self.high_cutoff, score <= self.low_cutoff
        else:
            high, low = score > self.high_cutoff, score < self.low_cutoff
            if score == self.high_cutoff:
                high = self.high_ties_seen < self.high_ties_keep
                self.high_ties_seen += 1
            if score == self.low_cutoff:
                low = self.low_ties_seen < self.low_ties_keep
                self.low_ties_seen += 1
        return "high" if high else "low" if low else None


@dataclass
class SelectionPlan:
    resolver: FrameResolver
    frames: Dict[FrameKey, FrameScan]
    boundaries: Dict[Optional[FrameKey], BandBoundary]
    patch_count: int
    minimum: float
    maximum: float


def build_selection(records: Iterable[dict], args) -> SelectionPlan:
    resolver = None
    frames = {}
    scores = defaultdict(lambda: array("d"))
    count = 0
    minimum, maximum = float("inf"), -float("inf")
    print("Pass 1/3: reading saliency and physical frame identifiers...", flush=True)
    for number, record in enumerate(records, 1):
        try:
            if resolver is None:
                resolver = FrameResolver(record, args.video_field, args.frame_field)
            key, context = resolver.key(record), resolver.context(record)
            if key not in frames:
                frames[key] = FrameScan(context, len(frames))
            scan = frames[key]
            if scan.context != context:
                raise ValueError("Physical frame appears with different window contexts: {}.".format(key))
            if ("patch_index" in record) != resolver.has_patch_index:
                raise ValueError("Inconsistent patch_index availability.")
            if resolver.has_patch_index:
                index = integer(record["patch_index"], "patch_index")
                if index > 1_000_000:
                    raise ValueError("Unreasonably large patch_index.")
                bit = 1 << index
                if scan.patch_bits & bit:
                    raise ValueError("Duplicate patch_index {} in physical frame {}.".format(index, key))
                scan.patch_bits |= bit
            value = saliency(record, args.saliency_field)
            scan.count += 1
            scores[key if args.percentile_scope == "frame" else None].append(value)
            minimum, maximum = min(minimum, value), max(maximum, value)
            count += 1
        except (ValueError, TypeError) as exc:
            raise ValueError("Record {}: {}".format(number, exc)) from exc
        if args.progress_every and number % args.progress_every == 0:
            print("  Read {:,} patches in {:,} frames.".format(number, len(frames)), flush=True)
    if resolver is None:
        raise ValueError("Input contains no patch records.")
    print("Computing {} percentile cutoffs...".format(args.percentile_scope), flush=True)
    boundaries = {k: BandBoundary.from_scores(v, args.top_percent, args.bottom_percent) for k, v in scores.items()}
    return SelectionPlan(resolver, frames, boundaries, count, minimum, maximum)


def checked_frame(record: dict, plan: SelectionPlan) -> FrameKey:
    key = plan.resolver.key(record)
    if key not in plan.frames or plan.frames[key].context != plan.resolver.context(record):
        raise ValueError("Frame/context changed between input passes.")
    return key


def check_pass_counts(counts: Counter, plan: SelectionPlan):
    if sum(counts.values()) != plan.patch_count or any(counts[k] != f.count for k, f in plan.frames.items()):
        raise ValueError("Patch counts changed between input passes.")


@dataclass
class SignatureBand:
    name: str
    selected: int = 0
    frequencies: Counter = field(default_factory=Counter)
    minimum: float = float("inf")
    maximum: float = -float("inf")

    def add(self, bits: int, value: float):
        self.selected += 1
        self.frequencies[bits] += 1
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)


@dataclass
class GlobalTransactionIndex:
    """All patches: unique top-20 lists with distinct physical-frame bitsets."""
    transaction_bits: List[int]
    bits_to_gid: Dict[int, int]
    frame_masks: List[int]
    postings: Dict[int, frozenset]

    @property
    def transaction_count(self) -> int:
        return len(self.transaction_bits)


def build_global_postings(transaction_bits: List[int]) -> Dict[int, frozenset]:
    buckets = defaultdict(set)
    for gid, bits in enumerate(transaction_bits):
        for prototype in signature_ids(bits):
            buckets[prototype].add(gid)
    return {prototype: frozenset(gids) for prototype, gids in buckets.items()}


PATCH_ARCHIVE_MAGIC = b"ESPC"
PATCH_ARCHIVE_VERSION = 1
_PATCH_FRAME = struct.Struct("<IHI")
_PATCH_ENTRY = struct.Struct("<II")
_PATCH_HEADER = struct.Struct("<4sIQQ")


@dataclass
class PatchArchive:
    """Columnar on-disk frame blocks: patch_id + global top-20 transaction id."""
    path: Path
    patch_count: int = 0
    frame_count: int = 0
    frame_index: List[Tuple[int, int, int]] = field(default_factory=list)  # frame_no, video_idx, offset

    def write_frame(self, handle, frame_number: int, video_index: int,
                    patches: Sequence[Tuple[int, int]]) -> None:
        offset = handle.tell()
        handle.write(_PATCH_FRAME.pack(frame_number, video_index, len(patches)))
        for patch_id, gid in patches:
            handle.write(_PATCH_ENTRY.pack(patch_id, gid))
        self.frame_index.append((frame_number, video_index, offset))
        self.patch_count += len(patches)
        self.frame_count += 1

    @classmethod
    def create(cls, path: Path) -> "PatchArchive":
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("wb")
        handle.write(_PATCH_HEADER.pack(PATCH_ARCHIVE_MAGIC, PATCH_ARCHIVE_VERSION, 0, 0))
        archive = cls(path)
        archive._handle = handle
        return archive

    def finalize(self) -> None:
        handle = self._handle
        handle.flush()
        end = handle.tell()
        handle.seek(0)
        handle.write(_PATCH_HEADER.pack(PATCH_ARCHIVE_MAGIC, PATCH_ARCHIVE_VERSION,
                                       self.frame_count, self.patch_count))
        handle.seek(end)
        handle.close()
        del self._handle

    @classmethod
    def open(cls, path: Path) -> "PatchArchive":
        if not path.is_file():
            raise ValueError("Patch archive does not exist: {}.".format(path))
        with path.open("rb") as handle:
            magic, version, frame_count, patch_count = _PATCH_HEADER.unpack(handle.read(_PATCH_HEADER.size))
        if magic != PATCH_ARCHIVE_MAGIC or version != PATCH_ARCHIVE_VERSION:
            raise ValueError("Unsupported or corrupt patch archive: {}.".format(path))
        archive = cls(path, patch_count=int(patch_count), frame_count=int(frame_count))
        archive._rebuild_index()
        return archive

    def _rebuild_index(self) -> None:
        self.frame_index = []
        with self.path.open("rb") as handle:
            handle.seek(_PATCH_HEADER.size)
            while handle.tell() < self.path.stat().st_size:
                offset = handle.tell()
                frame_number, video_index, n_patches = _PATCH_FRAME.unpack(handle.read(_PATCH_FRAME.size))
                handle.seek(n_patches * _PATCH_ENTRY.size, 1)
                self.frame_index.append((frame_number, video_index, offset))

    def read_frame(self, handle, offset: int) -> Tuple[int, int, List[Tuple[int, int]]]:
        handle.seek(offset)
        frame_number, video_index, n_patches = _PATCH_FRAME.unpack(handle.read(_PATCH_FRAME.size))
        patches = [_PATCH_ENTRY.unpack(handle.read(_PATCH_ENTRY.size)) for _ in range(n_patches)]
        return frame_number, video_index, patches

    def iter_frames(self) -> Iterable[Tuple[int, int, List[Tuple[int, int]]]]:
        with self.path.open("rb") as handle:
            for _, _, offset in self.frame_index:
                yield self.read_frame(handle, offset)


@dataclass
class IngestResult:
    plan: SelectionPlan
    high: SignatureBand
    low: SignatureBand
    ambiguous: int
    global_index: GlobalTransactionIndex
    archive: PatchArchive
    video_index: Dict[str, int]


def _classify_bands_from_rows(rows, plan: SelectionPlan, args) -> Tuple[SignatureBand, SignatureBand, int]:
    high, low = SignatureBand("high"), SignatureBand("low")
    ambiguous = 0
    for bits, value, boundary_key in rows:
        boundary = plan.boundaries[boundary_key]
        band = boundary.classify(value, args.tie_policy)
        if band == "ambiguous":
            ambiguous += 1
        elif band is not None:
            (high if band == "high" else low).add(bits, value)
    return high, low, ambiguous


def expected_patch_total(meta_path: Optional[Path]) -> Optional[int]:
    """Use prior ingest metadata so JSON ingest tqdm can show ETA."""
    if meta_path is None or not meta_path.is_file():
        return None
    try:
        with meta_path.open("rb") as handle:
            payload = pickle.load(handle)
        return int(payload["plan"].patch_count)
    except (KeyError, OSError, TypeError, ValueError):
        return None


def ingest_patches_single_pass(records: Iterable[dict], args, archive_path: Path,
                               expected_patches: Optional[int] = None) -> IngestResult:
    """One JSON read: percentile plan, global index, discovery bands, and binary patch archive."""
    resolver = None
    frames: Dict[FrameKey, FrameScan] = {}
    scores = defaultdict(lambda: array("d"))
    global_bits: List[int] = []
    bits_to_gid: Dict[int, int] = {}
    frame_masks: List[int] = []
    minimum, maximum = float("inf"), -float("inf")
    pending: Dict[FrameKey, List[Tuple[int, int, float, Optional[FrameKey]]]] = {}
    patch_total = 0
    record_iter = tqdm_wrap(records, "Pass 1/2 JSON ingest", args, unit="patch",
                            total=expected_patches)
    for number, record in enumerate(record_iter, 1):
        try:
            if resolver is None:
                resolver = FrameResolver(record, args.video_field, args.frame_field)
            key = resolver.key(record)
            context = resolver.context(record)
            if key not in frames:
                frames[key] = FrameScan(context, len(frames))
            scan = frames[key]
            if scan.context != context:
                raise ValueError("Physical frame appears with different window contexts: {}.".format(key))
            if ("patch_index" in record) != resolver.has_patch_index:
                raise ValueError("Inconsistent patch_index availability.")
            if resolver.has_patch_index:
                patch_id = integer(record["patch_index"], "patch_index")
                if patch_id > 1_000_000:
                    raise ValueError("Unreasonably large patch_index.")
                bit = 1 << patch_id
                if scan.patch_bits & bit:
                    raise ValueError("Duplicate patch_index {} in physical frame {}.".format(patch_id, key))
                scan.patch_bits |= bit
            else:
                patch_id = len(pending.get(key, []))
            scan.count += 1
            patch_total += 1
            value = saliency(record, args.saliency_field)
            scores[key if args.percentile_scope == "frame" else None].append(value)
            minimum, maximum = min(minimum, value), max(maximum, value)
            bits = activation_signature(record, args.top_k)
            frame_number = scan.number
            gid = bits_to_gid.get(bits)
            if gid is None:
                gid = len(global_bits)
                bits_to_gid[bits] = gid
                global_bits.append(bits)
                frame_masks.append(0)
            frame_masks[gid] |= 1 << frame_number
            boundary_key = key if args.percentile_scope == "frame" else None
            pending.setdefault(key, []).append((patch_id, gid, value, boundary_key))
        except (ValueError, TypeError) as exc:
            raise ValueError("Record {}: {}".format(number, exc)) from exc
        if getattr(args, "no_tqdm", False) and args.progress_every and number % args.progress_every == 0:
            print("  Read {:,} patches in {:,} frames.".format(number, len(frames)), flush=True)
    if resolver is None:
        raise ValueError("Input contains no patch records.")
    completed_frames: List[Tuple[FrameKey, int, List[Tuple[int, int, float, Optional[FrameKey]]]]] = []
    for key, scan in frames.items():
        patches = pending.get(key, [])
        if len(patches) != scan.count:
            raise ValueError("Frame {} has {:,} buffered patches but {:,} were counted."
                             .format(key, len(patches), scan.count))
        completed_frames.append((key, scan.number, patches))
    pending.clear()
    print("Computing {} percentile cutoffs...".format(args.percentile_scope), flush=True)
    boundaries = {k: BandBoundary.from_scores(v, args.top_percent, args.bottom_percent) for k, v in scores.items()}
    plan = SelectionPlan(resolver, frames, boundaries, patch_total, minimum, maximum)
    for boundary in plan.boundaries.values():
        boundary.high_ties_seen = boundary.low_ties_seen = 0
    band_rows = [(global_bits[gid], sal, bkey)
                 for _, _, patches in completed_frames for _, gid, sal, bkey in patches]
    high, low, ambiguous = _classify_bands_from_rows(band_rows, plan, args)
    global_index = GlobalTransactionIndex(global_bits, bits_to_gid, frame_masks,
                                          build_global_postings(global_bits))
    occupied_frames = 0
    for mask in frame_masks:
        occupied_frames |= mask
    print("  Global index: {:,} unique top-{} activation transactions; {:,} physical frames in index."
          .format(len(global_bits), args.top_k, popcount(occupied_frames)), flush=True)
    video_index = {video: bit for bit, video in enumerate(sorted({k[0] for k in plan.frames}))}
    archive = PatchArchive.create(archive_path)
    handle = archive._handle
    frame_rows = sorted(completed_frames, key=lambda row: row[1])
    for key, frame_number, patches in tqdm_wrap(
            frame_rows, "Writing patch archive", args, unit="frame", total=len(frame_rows)):
        archive.write_frame(handle, frame_number, video_index[key[0]],
                              [(patch_id, gid) for patch_id, gid, _, _ in patches])
    archive.finalize()
    if args.progress_every:
        print("  Scanned {:,}; high {:,}, low {:,}.".format(patch_total, high.selected, low.selected), flush=True)
    return IngestResult(plan, high, low, ambiguous, global_index, archive, video_index)


class _IngestMetaUnpickler(pickle.Unpickler):
    """Meta is often written with classes pickling as __main__.* when this file is executed directly."""

    def find_class(self, module: str, name: str):
        if module == "__main__":
            for candidate in (__name__, "generate_candidate_pairs", "__main__"):
                try:
                    return super().find_class(candidate, name)
                except AttributeError:
                    continue
            raise AttributeError("Can't get attribute {!r} from ingest meta (tried __main__ aliases)."
                                 .format(name))
        return super().find_class(module, name)


def load_ingest_cache(meta_path: Path) -> Optional[IngestResult]:
    if not meta_path.is_file():
        return None
    with meta_path.open("rb") as handle:
        payload = _IngestMetaUnpickler(handle).load()
    archive = PatchArchive.open(payload["archive_path"])
    return IngestResult(payload["plan"], payload["high"], payload["low"], payload["ambiguous"],
                        payload["global_index"], archive, payload["video_index"])


def save_ingest_cache(meta_path: Path, ingest: IngestResult, source_stat, top_k: int) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "top_k": top_k,
        "archive_path": ingest.archive.path,
        "plan": ingest.plan,
        "high": ingest.high,
        "low": ingest.low,
        "ambiguous": ingest.ambiguous,
        "global_index": ingest.global_index,
        "video_index": ingest.video_index,
    }
    with meta_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cache_valid(meta_path: Path, source_path: Path, top_k: int) -> bool:
    if not meta_path.is_file():
        return False
    with meta_path.open("rb") as handle:
        payload = pickle.load(handle)
    archive_path = Path(payload["archive_path"])
    if not archive_path.is_file():
        return False
    stat = source_path.stat()
    return (payload.get("source_size") == stat.st_size
            and payload.get("source_mtime_ns") == stat.st_mtime_ns
            and payload.get("top_k", EXPORT_TOP_K) == top_k)


def resolve_ingest(args, source_path: Path) -> IngestResult:
    archive_path = args.patch_cache
    meta_path = Path(str(archive_path) + ".meta") if archive_path else None
    if archive_path and meta_path and cache_valid(meta_path, source_path, args.top_k):
        print("Reusing patch archive cache {} (source JSON and --top-k unchanged).".format(archive_path),
              flush=True)
        return load_ingest_cache(meta_path)
    if archive_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix="patch_archive_", suffix=".bin", delete=False)
        archive_path = Path(tmp.name)
        tmp.close()
        meta_path = None
    patch_total_hint = expected_patch_total(meta_path)
    ingest = ingest_patches_single_pass(iter_patch_records(source_path), args, archive_path,
                                        expected_patches=patch_total_hint)
    if args.patch_cache and meta_path:
        save_ingest_cache(meta_path, ingest, source_path.stat(), args.top_k)
    return ingest


@dataclass
class MiningStats:
    prefixes_examined: int = 0
    rejected_discovery: int = 0
    rejected_global: int = 0
    completed_retained: int = 0


@dataclass
class MiningResult:
    frequencies: Dict[int, int]
    recurring_count: int
    stats: MiningStats = field(default_factory=MiningStats)


def global_frames_meets_threshold(gids: frozenset, frame_masks: List[int], threshold: int) -> bool:
    """Early stop once OR'd frame support reaches threshold (not an exact count)."""
    mask = 0
    for gid in gids:
        mask |= frame_masks[gid]
        if popcount(mask) >= threshold:
            return True
    return popcount(mask) >= threshold


def make_global_frame_counter(frame_masks: List[int]):
    @lru_cache(maxsize=16384)
    def exact(gids: frozenset) -> int:
        mask = 0
        for gid in gids:
            mask |= frame_masks[gid]
        return popcount(mask)

    return exact


def mine_subsets(band: SignatureBand, subset_size: int, min_patch_support: int,
                 min_frame_support: int, global_index: GlobalTransactionIndex,
                 limit: int = 0, progress_every: int = 0) -> MiningResult:
    """Weighted vertical mining with discovery patch support and global frame support.

    Discovery intersections use band-local transaction IDs. Global intersections
    use dataset-wide transaction IDs; frame support ORs masks of matching patches.
    """
    transactions = sorted(band.frequencies)
    weights = [band.frequencies[bits] for bits in transactions]
    band_postings = defaultdict(set)
    for tid, bits in enumerate(transactions):
        for prototype in signature_ids(bits):
            band_postings[prototype].add(tid)
    unit_weights = all(w == 1 for w in weights)
    global_postings = global_index.postings
    frame_masks = global_index.frame_masks
    global_frames_exact = make_global_frame_counter(frame_masks)

    def discovery_support(tids: frozenset) -> int:
        return len(tids) if unit_weights else sum(weights[t] for t in tids)

    stats = MiningStats()
    roots = []
    for prototype, tids in sorted(band_postings.items()):
        frozen = frozenset(tids)
        patch_n = discovery_support(frozen)
        gids = global_postings.get(prototype, frozenset())
        if patch_n < min_patch_support:
            stats.rejected_discovery += 1
            continue
        if not global_frames_meets_threshold(gids, frame_masks, min_frame_support):
            stats.rejected_global += 1
            continue
        roots.append((prototype, frozen, patch_n, gids))
    retained, heap = {}, []
    recurring = last_update = 0

    def retain(bits: int, patch_n: int):
        nonlocal recurring
        recurring += 1
        stats.completed_retained += 1
        if not limit:
            retained[bits] = patch_n
        else:
            entry = (patch_n, -bits, bits)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    def visit(prefix: int, remaining: int, candidates):
        nonlocal recurring, last_update
        if len(candidates) < remaining:
            return
        for position, (prototype, discovery_tids, patch_n, global_gids) in enumerate(candidates):
            if len(candidates) - position < remaining:
                break
            stats.prefixes_examined += 1
            if progress_every and stats.prefixes_examined % 4096 == 0 and time.monotonic() - last_update >= 20:
                print("  {}: {:,} prefixes examined, {:,} subsets retained."
                      .format(band.name, stats.prefixes_examined, stats.completed_retained), flush=True)
                last_update = time.monotonic()
            bits = prefix | (1 << prototype)
            if remaining == 1:
                if patch_n < min_patch_support:
                    stats.rejected_discovery += 1
                    continue
                if global_frames_exact(global_gids) < min_frame_support:
                    stats.rejected_global += 1
                    continue
                retain(bits, patch_n)
                continue
            children = []
            for other, other_discovery, _, other_global in candidates[position + 1:]:
                shared_discovery = discovery_tids & other_discovery
                if not shared_discovery:
                    continue
                shared_global = global_gids & other_global
                if not shared_global:
                    stats.rejected_global += 1
                    continue
                total_patch = discovery_support(shared_discovery)
                if total_patch < min_patch_support:
                    stats.rejected_discovery += 1
                    continue
                if not global_frames_meets_threshold(shared_global, frame_masks, min_frame_support):
                    stats.rejected_global += 1
                    continue
                children.append((other, shared_discovery, total_patch, shared_global))
            visit(bits, remaining - 1, children)

    visit(0, subset_size, roots)
    if limit:
        retained = {bits: n for n, _, bits in heap}
    return MiningResult(retained, recurring, stats)


@dataclass(frozen=True)
class PotentialSignature:
    bits: int
    high_patches: int
    low_patches: int
    discovery_bands: Optional[str] = None

    @property
    def origins(self) -> str:
        if self.discovery_bands is not None:
            return self.discovery_bands
        return "+".join(x for x, n in (("high", self.high_patches), ("low", self.low_patches)) if n)


def potential_candidates(high: MiningResult, low: MiningResult,
                         high_band: Optional[SignatureBand] = None,
                         low_band: Optional[SignatureBand] = None) -> List[PotentialSignature]:
    masks = set(high.frequencies) | set(low.frequencies)
    order = sorted(masks, key=lambda x: (-high.frequencies.get(x, 0) - low.frequencies.get(x, 0), x))
    signatures = [PotentialSignature(x, high.frequencies.get(x, 0), low.frequencies.get(x, 0),
                     "+".join(name for name, mining in (("high", high), ("low", low)) if x in mining.frequencies))
                  for x in order]
    # Absence from a band's retained mining output does not mean zero patches:
    # it may have one occurrence, or fall outside an explicit top-N pool limit.
    counts = [[s.high_patches for s in signatures], [s.low_patches for s in signatures]]
    for position, (band, mining) in enumerate(((high_band, high), (low_band, low))):
        if band is None:
            continue
        missing = [i for i, s in enumerate(signatures) if s.bits not in mining.frequencies]
        if missing:
            matcher = SignatureMatcher([signatures[i] for i in missing])
            for bits, frequency in band.frequencies.items():
                for local_index in matcher.matches(bits):
                    counts[position][missing[local_index]] += frequency
    result = [PotentialSignature(s.bits, counts[0][i], counts[1][i], s.origins)
              for i, s in enumerate(signatures)]
    return sorted(result, key=lambda s: (-s.high_patches - s.low_patches, s.bits))


class SignatureMatcher:
    """Return all contained signatures using a trie and a bounded per-list cache."""
    def __init__(self, signatures: List[PotentialSignature],
                 match_cache_size: int = DEFAULT_MATCH_CACHE_SIZE):
        self.root = {}
        for index, signature in enumerate(signatures):
            node = self.root
            for prototype in signature_ids(signature.bits):
                node = node.setdefault(prototype, {})
            node[-1] = index
        self.matches = lru_cache(maxsize=max(1024, match_cache_size))(self._matches)

    def _matches(self, bits: int) -> Tuple[int, ...]:
        ids = signature_ids(bits)
        found = []

        def visit(node, start):
            if -1 in node:
                found.append(node[-1])
                return
            for position in range(start, len(ids)):
                child = node.get(ids[position])
                if child is not None:
                    visit(child, position + 1)

        visit(self.root, 0)
        return tuple(sorted(found))


PAIR_SPILL_THRESHOLD = 8_000_000


def encode_pair_key(a: int, b: int, n_signatures: int) -> int:
    if a > b:
        a, b = b, a
    return a * n_signatures + b


def decode_pair_key(key: int, n_signatures: int) -> Tuple[int, int]:
    a, b = divmod(key, n_signatures)
    return a, b


class ObservedPairStore:
    """Compact frame/video tallies for observed signature pairs only."""

    def __init__(self, spill_threshold: int = PAIR_SPILL_THRESHOLD):
        self._memory: Dict[int, Tuple[int, int]] = {}
        self._spill: Optional[shelve.Shelf] = None
        self._spill_path: Optional[str] = None
        self._spill_threshold = spill_threshold

    def close(self):
        if self._spill is not None:
            self._spill.close()
            self._spill = None
        if self._spill_path is not None:
            try:
                Path(self._spill_path).unlink()
            except OSError:
                pass
            self._spill_path = None

    def _migrate_to_spill(self):
        if self._spill is not None:
            return
        handle = tempfile.NamedTemporaryFile(prefix="observed_pairs_", suffix=".db", delete=False)
        self._spill_path = handle.name
        handle.close()
        self._spill = shelve.open(self._spill_path, writeback=False)
        for key, value in self._memory.items():
            self._spill[str(key)] = value
        self._memory.clear()
        print("  Observed-pair map spilled to disk at {:,} entries.".format(len(self._spill)), flush=True)

    def increment(self, key: int, video_bit: int):
        self.merge_delta(key, 1, video_bit)

    def merge_delta(self, key: int, frame_count: int, video_bit: int) -> None:
        if self._spill is None:
            previous = self._memory.get(key)
            if previous is None:
                self._memory[key] = (frame_count, video_bit)
                if len(self._memory) >= self._spill_threshold:
                    self._migrate_to_spill()
                return
            frames, videos = previous
            self._memory[key] = (frames + frame_count, videos | video_bit)
            return
        previous = self._spill.get(str(key))
        if previous is None:
            self._spill[str(key)] = (frame_count, video_bit)
        else:
            frames, videos = previous
            self._spill[str(key)] = (frames + frame_count, videos | video_bit)

    def items(self) -> Iterable[Tuple[int, Tuple[int, int]]]:
        if self._spill is None:
            yield from self._memory.items()
            return
        for key, value in self._spill.items():
            yield int(key), value

    def __len__(self) -> int:
        return len(self._spill) if self._spill is not None else len(self._memory)


@dataclass
class FramePairCountResult:
    pair_store: ObservedPairStore
    n_signatures: int
    signature_patch_counts: List[int]
    signature_frame_masks: List[int]
    video_index: Dict[str, int]
    frames_processed: int
    max_signatures_in_frame: int
    local_combinations_examined: int

    def qualifying_pairs(self, min_frames: int) -> List[Tuple[int, int, int, int]]:
        rows = []
        for key, (frames, videos) in self.pair_store.items():
            if frames >= min_frames:
                a, b = decode_pair_key(key, self.n_signatures)
                rows.append((a, b, frames, popcount(videos)))
        rows.sort(key=lambda row: (-row[2], row[0], row[1]))
        return rows


def iter_valid_signature_pairs_in_frame(sig_patches: Dict[int, set], matching_patches: set
                                        ) -> Iterable[Tuple[int, int]]:
    if len(matching_patches) < 2:
        return
    present = sorted(sig_patches)
    for i, a in enumerate(present):
        pa = sig_patches[a]
        for b in present[i + 1:]:
            pb = sig_patches[b]
            if len(pa) == 1 and len(pb) == 1 and next(iter(pa)) == next(iter(pb)):
                continue
            yield a, b


def count_frame_local_combinations(sig_patches: Dict[int, set], matching_patches: set) -> int:
    if len(matching_patches) < 2:
        return 0
    present = sorted(sig_patches)
    n = len(present)
    return n * (n - 1) // 2


def build_inverted_signature_index(signatures: List[PotentialSignature],
                                   global_index: GlobalTransactionIndex,
                                   args=None) -> Tuple[Tuple[int, ...], ...]:
    """Precompute signature matches for each global top-20 transaction id (table lookup in Pass 2)."""
    matcher = SignatureMatcher(signatures)
    bits_iter = tqdm_wrap(global_index.transaction_bits, "Inverted top-20 index", args,
                          unit="txn", total=global_index.transaction_count)
    return tuple(matcher.matches(bits) for bits in bits_iter)


def build_sig_patches_for_frame_patches(patches: Sequence[Tuple[int, int]],
                                        inverted: Sequence[Tuple[int, ...]]
                                        ) -> Tuple[Dict[int, set], set]:
    """Group by global transaction id; one inverted-index lookup per distinct gid in the frame."""
    patches_by_gid: Dict[int, set] = {}
    for patch_id, gid in patches:
        patches_by_gid.setdefault(gid, set()).add(patch_id)
    sig_patches: Dict[int, set] = {}
    matching_patches = set()
    for gid, patch_ids in patches_by_gid.items():
        matched = inverted[gid]
        if not matched:
            continue
        matching_patches |= patch_ids
        for sig in matched:
            bucket = sig_patches.get(sig)
            if bucket is None:
                sig_patches[sig] = set(patch_ids)
            else:
                bucket.update(patch_ids)
    return sig_patches, matching_patches


def _process_one_frame(frame_number: int, video_index: int, patches: Sequence[Tuple[int, int]],
                       inverted: Sequence[Tuple[int, ...]], n_signatures: int
                       ) -> Tuple[Dict[int, Tuple[int, int]], List[int], List[int], int, int]:
    """Return pair deltas, per-signature patch counts, frame masks, max sigs, local combos."""
    sig_patches, matching_patches = build_sig_patches_for_frame_patches(patches, inverted)
    frame_bit = 1 << frame_number
    video_bit = 1 << video_index
    sig_patch_delta = [0] * n_signatures
    sig_frame_delta = [0] * n_signatures
    for sig, patch_ids in sig_patches.items():
        sig_patch_delta[sig] += len(patch_ids)
        sig_frame_delta[sig] |= frame_bit
    local_combos = count_frame_local_combinations(sig_patches, matching_patches)
    pair_delta: Dict[int, Tuple[int, int]] = {}
    for a, b in iter_valid_signature_pairs_in_frame(sig_patches, matching_patches):
        key = encode_pair_key(a, b, n_signatures)
        previous = pair_delta.get(key)
        if previous is None:
            pair_delta[key] = (1, video_bit)
        else:
            pair_delta[key] = (previous[0] + 1, previous[1] | video_bit)
    return pair_delta, sig_patch_delta, sig_frame_delta, len(sig_patches), local_combos


def _merge_pair_delta(store: ObservedPairStore, delta: Dict[int, Tuple[int, int]]) -> None:
    for key, (frames, videos) in delta.items():
        store.merge_delta(key, frames, videos)


_PAIR_WORKER_INVERTED: Optional[Tuple[Tuple[int, ...], ...]] = None


def _init_pair_count_worker(inverted: Tuple[Tuple[int, ...], ...]) -> None:
    global _PAIR_WORKER_INVERTED
    _PAIR_WORKER_INVERTED = inverted


def _pair_task_chunk_size(n_frames: int, workers: int) -> int:
    """Smaller tasks than one chunk per worker so tqdm updates during long pair passes."""
    target_tasks = max(workers * 32, workers)
    return max(128, min(2048, (n_frames + target_tasks - 1) // target_tasks))


def _pair_count_worker(task: Tuple[Path, List[Tuple[int, int, int]], int]
                       ) -> Tuple[Dict[int, Tuple[int, int]], List[int], List[int], int, int, int]:
    archive_path, frame_specs, n_signatures = task
    inverted = _PAIR_WORKER_INVERTED
    if inverted is None:
        raise RuntimeError("Pair worker missing inverted index (pool initializer not run).")
    pair_acc: Dict[int, Tuple[int, int]] = {}
    sig_patches = [0] * n_signatures
    sig_frames = [0] * n_signatures
    max_sigs = 0
    local_combos = 0
    frames_done = 0
    with archive_path.open("rb") as handle:
        archive = PatchArchive(archive_path)
        for frame_number, video_index, offset in frame_specs:
            _, _, patches = archive.read_frame(handle, offset)
            delta, patch_d, frame_d, max_sig, combos = _process_one_frame(
                frame_number, video_index, patches, inverted, n_signatures)
            for key, (frames, videos) in delta.items():
                prev = pair_acc.get(key)
                if prev is None:
                    pair_acc[key] = (frames, videos)
                else:
                    pair_acc[key] = (prev[0] + frames, prev[1] | videos)
            for i, n in enumerate(patch_d):
                sig_patches[i] += n
            for i, mask in enumerate(frame_d):
                sig_frames[i] |= mask
            max_sigs = max(max_sigs, max_sig)
            local_combos += combos
            frames_done += 1
    return pair_acc, sig_patches, sig_frames, max_sigs, local_combos, frames_done


def count_observed_signature_pairs_from_archive(ingest: IngestResult, signatures: List[PotentialSignature],
                                                args) -> FramePairCountResult:
    n_signatures = len(signatures)
    inverted = build_inverted_signature_index(signatures, ingest.global_index, args)
    pair_store = ObservedPairStore()
    signature_patch_counts = [0] * n_signatures
    signature_frame_masks = [0] * n_signatures
    max_signatures_in_frame = 0
    local_combinations_examined = 0
    frames_processed = 0
    workers = max(1, int(getattr(args, "pair_workers", 1)))
    frame_specs = ingest.archive.frame_index
    last_update = time.monotonic()
    if workers == 1:
        frame_iter = tqdm_wrap(frame_specs, "Pass 2/2 pair counting", args,
                               unit="frame", total=len(frame_specs))
        with ingest.archive.path.open("rb") as handle:
            for frame_number, video_index, offset in frame_iter:
                _, _, patches = ingest.archive.read_frame(handle, offset)
                delta, patch_d, frame_d, max_sig, combos = _process_one_frame(
                    frame_number, video_index, patches, inverted, n_signatures)
                _merge_pair_delta(pair_store, delta)
                for i, n in enumerate(patch_d):
                    signature_patch_counts[i] += n
                for i, mask in enumerate(frame_d):
                    signature_frame_masks[i] |= mask
                max_signatures_in_frame = max(max_signatures_in_frame, max_sig)
                local_combinations_examined += combos
                frames_processed += 1
                if getattr(args, "no_tqdm", False) and progress_every_frames(
                        frames_processed, last_update, args.progress_every):
                    print("  {:,} frames; {:,} observed pairs; max {:,} signatures/frame."
                          .format(frames_processed, len(pair_store), max_signatures_in_frame), flush=True)
                    last_update = time.monotonic()
    else:
        chunk_size = _pair_task_chunk_size(len(frame_specs), workers)
        tasks = []
        for start in range(0, len(frame_specs), chunk_size):
            chunk = [(fn, vi, off) for fn, vi, off in frame_specs[start:start + chunk_size]]
            tasks.append((ingest.archive.path, chunk, n_signatures))
        print("  Pair pass: {:,} frames, {:,} workers, ~{:,} frames/task ({} tasks)."
              .format(len(frame_specs), workers, chunk_size, len(tasks)), flush=True)
        pbar = tqdm_create("Pass 2/2 pair counting", args, unit="frame", total=len(frame_specs))
        with mp.Pool(workers, initializer=_init_pair_count_worker, initargs=(inverted,)) as pool:
            for pair_acc, patch_d, frame_d, max_sig, combos, done in pool.imap_unordered(_pair_count_worker, tasks):
                _merge_pair_delta(pair_store, pair_acc)
                for i, n in enumerate(patch_d):
                    signature_patch_counts[i] += n
                for i, mask in enumerate(frame_d):
                    signature_frame_masks[i] |= mask
                max_signatures_in_frame = max(max_signatures_in_frame, max_sig)
                local_combinations_examined += combos
                frames_processed += done
                if pbar is not None:
                    pbar.update(done)
                    pbar.set_postfix(observed_pairs=len(pair_store), max_sig=max_signatures_in_frame, refresh=False)
                elif progress_every_frames(frames_processed, last_update, args.progress_every):
                    print("  {:,} frames; {:,} observed pairs; max {:,} signatures/frame."
                          .format(frames_processed, len(pair_store), max_signatures_in_frame), flush=True)
                    last_update = time.monotonic()
        if pbar is not None:
            pbar.close()
    print("  Finished {:,} frames; {:,} distinct observed signature pairs; "
          "max {:,} signatures in one frame; {:,} local combinations examined."
          .format(frames_processed, len(pair_store), max_signatures_in_frame, local_combinations_examined),
          flush=True)
    return FramePairCountResult(pair_store, n_signatures, signature_patch_counts, signature_frame_masks,
                                ingest.video_index, frames_processed, max_signatures_in_frame,
                                local_combinations_examined)


def progress_every_frames(frames_processed: int, last_update: float, progress_every: int) -> bool:
    if not progress_every:
        return False
    return frames_processed % 4096 == 0 and time.monotonic() - last_update >= 20


def signature_id(index: int) -> str:
    return "S{:06d}".format(index + 1)


@dataclass
class CandidateReport:
    histogram: Counter
    rows: List[Tuple[int, int, int, int]]  # signature A, signature B, frame count, video count
    observed_pairs: int
    qualifying: int = 0
    rejected_by_threshold: int = 0
    theoretical_combinations: int = 0
    zero_support_aggregate: int = 0


def write_result_csvs(output_dir: Path, signatures: List[PotentialSignature],
                      result: FramePairCountResult, qualifying_rows: List[Tuple[int, int, int, int]],
                      overwrite: bool) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "candidate_pairs.csv"
    signatures_path = output_dir / "activation_signatures.csv"
    for path in (pairs_path, signatures_path):
        if path.exists() and not overwrite:
            raise ValueError("Refusing to overwrite existing file: {}.".format(path))
    used_indices = set()
    with pairs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "signature_id_a", "signature_id_b", "same_frame_count", "video_count",
            "discovery_origin_a", "discovery_origin_b"))
        writer.writeheader()
        for a, b, frame_count, videos in qualifying_rows:
            writer.writerow({
                "signature_id_a": signature_id(a),
                "signature_id_b": signature_id(b),
                "same_frame_count": frame_count,
                "video_count": videos,
                "discovery_origin_a": signatures[a].origins,
                "discovery_origin_b": signatures[b].origins,
            })
            used_indices.add(a)
            used_indices.add(b)
    sig_rows = 0
    with signatures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "signature_id", "discovery_origin", "high_patch_count", "low_patch_count",
            "all_patch_count", "all_frame_count", "prototype_ids"))
        writer.writeheader()
        for index in sorted(used_indices):
            sig = signatures[index]
            writer.writerow({
                "signature_id": signature_id(index),
                "discovery_origin": sig.origins,
                "high_patch_count": sig.high_patches,
                "low_patch_count": sig.low_patches,
                "all_patch_count": result.signature_patch_counts[index],
                "all_frame_count": popcount(result.signature_frame_masks[index]),
                "prototype_ids": ",".join(str(x) for x in signature_ids(sig.bits)),
            })
            sig_rows += 1
    print("Wrote {:,} qualifying signature-pair rows to {}.".format(len(qualifying_rows), pairs_path), flush=True)
    print("Wrote {:,} activation signature definitions to {}.".format(sig_rows, signatures_path), flush=True)
    return len(qualifying_rows), sig_rows


def report_observed_pairs(result: FramePairCountResult, signatures: List[PotentialSignature],
                          limit: int, min_frames: int) -> CandidateReport:
    histogram = Counter()
    for _, (frames, _) in result.pair_store.items():
        histogram[frames] += 1
    qualifying_rows = result.qualifying_pairs(min_frames)
    observed = len(result.pair_store)
    qualifying = len(qualifying_rows)
    rejected = observed - qualifying
    theoretical = (result.n_signatures * (result.n_signatures - 1) // 2
                   if result.n_signatures >= 2 else 0)
    zero_support = theoretical - observed if theoretical else 0
    print("\nObserved signature-pair counting complete "
          "(minimum distinct frames per reported pair: {:,}).".format(min_frames), flush=True)
    print("Physical frames processed: {:,}.".format(result.frames_processed))
    print("Distinct observed signature pairs (>=1 valid frame): {:,}.".format(observed))
    print("Qualifying pairs (>= {:,} frames): {:,}.".format(min_frames, qualifying))
    print("Observed pairs below final frame threshold: {:,}.".format(rejected))
    if theoretical:
        print("Theoretical unordered signature combinations (not enumerated): {:,}.".format(theoretical))
        print("Aggregate unobserved / zero-support combinations (by subtraction): {:,}.".format(zero_support))
    heap = []
    for a, b, count, videos in qualifying_rows:
        if limit == 0:
            continue
        entry = (count, -a, -b, a, b, videos)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heappushpop(heap, entry)
    rows = [(a, b, count, videos) for count, _, _, a, b, videos in sorted(heap, reverse=True)]
    if limit == 0:
        print("Qualifying pairs (neutral signature IDs; A/B by index order):")
        print("signature A  signature B   same frames  videos")
        for a, b, count, videos in qualifying_rows:
            print("{}    {}  {:12,d} {:7,d}".format(signature_id(a), signature_id(b), count, videos))
    elif qualifying:
        print("Highest-frequency qualifying pairs (display limit {}):".format(limit))
        print(" rank  signature A  signature B   same frames  videos")
        for rank, (a, b, count, videos) in enumerate(rows, 1):
            print("{:5,d}  {}    {}  {:12,d} {:7,d}"
                  .format(rank, signature_id(a), signature_id(b), count, videos))
        if qualifying > limit:
            print("  Showing {} of {:,}; --report-limit 0 prints all qualifying pairs."
                  .format(limit, qualifying))
    print("\nFrequency guide (observed pairs only):")
    print(" minimum distinct frames    signature pairs")
    for threshold in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000):
        n = sum(count for support, count in histogram.items() if support >= threshold)
        print("{:24,d} {:18,d}".format(threshold, n))
    print("\nExact frame-count distribution (observed pairs):")
    print(" distinct frames    signature pairs")
    distribution = sorted(histogram.items())
    shown = distribution if not limit else distribution[:limit]
    for support, count in shown:
        print("{:16,d} {:18,d}".format(support, count))
    if limit and len(distribution) > limit:
        print("  Showing {} of {:,} support values; --report-limit 0 prints all.".format(limit, len(distribution)))
    print("\nEach frame contributes at most one occurrence per signature pair, with distinct patch witnesses.")
    print("A patch may match several signatures; pair counts can therefore reuse patches and frames.")
    print("Discovery origins are metadata; co-occurrence counting includes every saliency band.")
    print("No ordering consistency was evaluated.", flush=True)
    return CandidateReport(histogram, rows, observed, qualifying, rejected, theoretical, zero_support)


def print_band(band: SignatureBand, total: int, label: str, top_k: int):
    print("\n{}: {:,} patches ({:.4f}% of dataset).".format(label, band.selected, 100 * band.selected / total))
    if band.selected:
        print("  Selected saliency range: {:.8g} to {:.8g}.".format(band.minimum, band.maximum))
    print("  Distinct full top-{} activation lists: {:,}; every validated list has {} IDs."
          .format(top_k, len(band.frequencies), top_k))


def print_signatures(signatures: List[PotentialSignature], result: FramePairCountResult, limit: int):
    print("\nPooled potential_candidates: {:,} exact subsets.".format(len(signatures)))
    print("signature   discovery  high patches  low patches   all patches  all frames  prototype IDs")
    stop = len(signatures) if not limit else min(limit, len(signatures))
    for index in range(stop):
        sig = signatures[index]
        print("{}  {:9s} {:12,d} {:12,d} {:13,d} {:11,d}  {}"
              .format(signature_id(index), sig.origins, sig.high_patches, sig.low_patches,
                      result.signature_patch_counts[index],
                      popcount(result.signature_frame_masks[index]), signature_ids(sig.bits)))
    if limit and len(signatures) > limit:
        print("  Showing {} of {:,} signatures; --report-limit 0 prints all.".format(limit, len(signatures)))
    print("  High/low patch counts are actual band support; discovery records which band retained the subset.")


@dataclass
class AnalysisResult:
    plan: SelectionPlan
    high: SignatureBand
    low: SignatureBand
    high_mining: MiningResult
    low_mining: MiningResult
    potential_candidates: List[PotentialSignature]
    pair_counts: FramePairCountResult
    report: CandidateReport


def analyze(factory: RecordFactory, args) -> AnalysisResult:
    del factory  # Ingest reads via resolve_ingest / patch archive cache.
    ingest = resolve_ingest(args, args.patch_data)
    plan = ingest.plan
    high, low, ambiguous, global_index = ingest.high, ingest.low, ingest.ambiguous, ingest.global_index
    print("Loaded {:,} patches, {:,} physical frames, {:,} videos; predicted saliency {:.8g} to {:.8g}."
          .format(plan.patch_count, len(plan.frames), len({k[0] for k in plan.frames}), plan.minimum, plan.maximum))
    print("Frame identity: ({}, {}); percentile scope: {}; cutoff tie policy: {}."
          .format(plan.resolver.video_field, plan.resolver.frame_field, args.percentile_scope, args.tie_policy))
    if not plan.resolver.has_patch_index:
        print("No patch_index: rows are treated as distinct patches; spatial duplicate checking is unavailable.")
    boundaries = list(plan.boundaries.values())
    for label, attribute in (("High", "high_cutoff"), ("Low", "low_cutoff")):
        values = [getattr(b, attribute) for b in boundaries]
        print("{} cutoff range: {:.8g} to {:.8g}.".format(label, min(values), max(values)))
    print("Patches tied at high/low cutoffs: {:,} / {:,}."
          .format(sum(b.high_ties for b in boundaries), sum(b.low_ties for b in boundaries)))
    print("Activation context per patch: top-{} of exported ranked prototype slots (--top-k)."
          .format(args.top_k), flush=True)
    print_band(high, plan.patch_count, "Highest {:.6g}%".format(args.top_percent), args.top_k)
    print_band(low, plan.patch_count, "Lowest {:.6g}%".format(args.bottom_percent), args.top_k)
    if ambiguous:
        print("Excluded {:,} patches at indistinguishable/overlapping discovery cutoffs.".format(ambiguous))
    print("\nMinimum frame-occurrence threshold (global, enforced during mining): {:,}."
          .format(args.min_frame_occurrence))
    print("Mining exact {}-prototype subsets: >= {:,} discovery PATCHES per band AND "
          ">= {:,} global distinct frames..."
          .format(args.subset_size, args.min_subset_patches, args.min_frame_occurrence), flush=True)
    mined = []
    for band in tqdm_wrap((high, low), "Mining discovery bands", args, unit="band", total=2):
        result = mine_subsets(band, args.subset_size, args.min_subset_patches, args.min_frame_occurrence,
                              global_index, args.max_signatures_per_band, args.progress_every)
        mined.append(result)
        st = result.stats
        print("  {}: {:,} prefixes examined; {:,} branches rejected for patch support; "
              "{:,} rejected for global frame support; {:,} completed subsets retained."
              .format(band.name, st.prefixes_examined, st.rejected_discovery, st.rejected_global,
                      len(result.frequencies)), flush=True)
    high_mining, low_mining = mined
    signatures = potential_candidates(high_mining, low_mining, high, low)
    both = sum(s.origins == "high+low" for s in signatures)
    print("Deduplicated union: {:,} signatures; {:,} discovered in both bands.".format(len(signatures), both))
    if args.max_signatures_per_band:
        print("Pool limited explicitly to the most frequent {:,} subsets PER BAND."
              .format(args.max_signatures_per_band))
    else:
        print("All subsets passing both mining thresholds retained; report limits only affect display.")
    signature_count = len(signatures)
    theoretical = signature_count * (signature_count - 1) // 2 if signature_count >= 2 else 0
    if theoretical:
        print("Theoretical unordered signature combinations (not enumerated): {:,}.".format(theoretical))
    pair_counts = count_observed_signature_pairs_from_archive(ingest, signatures, args)
    print_signatures(signatures, pair_counts, args.report_limit)
    if signature_count < 2:
        print("\nFewer than two mined signatures; skipping pair reporting.")
        empty = CandidateReport(Counter(), [], 0)
        pair_counts.pair_store.close()
        return AnalysisResult(plan, high, low, high_mining, low_mining, signatures, pair_counts, empty)
    report = report_observed_pairs(pair_counts, signatures, args.report_limit, args.min_frame_occurrence)
    qualifying_rows = pair_counts.qualifying_pairs(args.min_frame_occurrence)
    if args.output_dir is not None:
        write_result_csvs(args.output_dir, signatures, pair_counts, qualifying_rows, args.overwrite_results)
    pair_counts.pair_store.close()
    return AnalysisResult(plan, high, low, high_mining, low_mining, signatures, pair_counts, report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch-data", "--patch_data", required=True, type=Path)
    parser.add_argument("--top-percent", type=float, default=3.0)
    parser.add_argument("--bottom-percent", type=float, default=50.0)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--subset-size", type=int, default=None, help="IDs per exact subset signature (default: 5).")
    size.add_argument("--overlap-percent", "--min-overlap-percent", type=float, default=None,
                      help="Legacy alias: ceil(20 * percent / 100) IDs per SUBSET, not overlap grouping.")
    parser.add_argument("--min-subset-patches", type=int, default=2,
                        help="Recurrence support in selected patches, per band (default: 2).")
    parser.add_argument("--max-signatures-per-band", type=int, default=1000,
                        help="Keep only the N most frequent subsets per band; 0 retains all (default).")
    parser.add_argument("--top-k", type=int, default=EXPORT_TOP_K,
                        help="Use the first K ranked prototypes per patch from top_20_* export fields "
                             "(default: 20). Extra exported ranks are ignored; shorter lists are rejected.")
    parser.add_argument("--percentile-scope", choices=("frame", "dataset"), default="frame",
                        help="Percentile population (default: frame; dataset uses pooled patch scores).")
    parser.add_argument("--tie-policy", choices=("exact", "include"), default="exact")
    parser.add_argument("--saliency-field", default="patch_pred_saliency")
    parser.add_argument("--video-field", default=None)
    parser.add_argument("--frame-field", default=None)
    parser.add_argument("--report-limit", type=int, default=50,
                        help="Printed rows per table; 0 prints all. Does not filter candidates.")
    parser.add_argument("--min-frame-occurrence", "--min-frame-occurance", type=int, default=1,
                        help="Minimum distinct physical frames globally for a mined subset and for a "
                             "reported signature pair (default: 1). Enforced during mining, not post-hoc.")
    parser.add_argument("--progress-every", type=int, default=250000,
                        help="Record progress interval; 0 also disables periodic mining/counting updates.")
    parser.add_argument("--patch-cache", type=Path, default=None,
                        help="Binary patch archive path (.meta sidecar). Reused when JSON is unchanged; "
                             "skips a full JSON re-ingest on reruns.")
    parser.add_argument("--pair-workers", type=int, default=1,
                        help="Parallel workers for archive pair counting (default: 1).")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable tqdm progress bars (keeps periodic --progress-every logs).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write candidate_pairs.csv and activation_signatures.csv here.")
    parser.add_argument("--overwrite-results", dest="overwrite_results", action="store_true", default=True,
                        help="Overwrite CSV outputs in --output-dir (default: on).")
    parser.add_argument("--no-overwrite-results", dest="overwrite_results", action="store_false",
                        help="Refuse to overwrite existing CSV outputs.")
    parser.add_argument("--min-same-frame-occurrences", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("top_percent", "bottom_percent"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 100:
            parser.error("--{} must be in (0, 100].".format(name.replace("_", "-")))
    if args.top_percent + args.bottom_percent > 100:
        parser.error("Top and bottom percentages must sum to at most 100.")
    if args.overlap_percent is not None:
        if not math.isfinite(args.overlap_percent) or not 0 < args.overlap_percent <= 100:
            parser.error("--overlap-percent must be in (0, 100].")
        args.subset_size = percent_count(args.top_k, args.overlap_percent)
    elif args.subset_size is None:
        args.subset_size = 5
    if not 1 <= args.top_k <= EXPORT_TOP_K:
        parser.error("--top-k must be in [1, {}].".format(EXPORT_TOP_K))
    if not 1 <= args.subset_size <= args.top_k:
        parser.error("--subset-size must be in [1, --top-k].")
    if args.min_subset_patches < 2:
        parser.error("--min-subset-patches must be >= 2 to select recurring subsets.")
    if args.min_frame_occurrence < 1:
        parser.error("--min-frame-occurrence must be a positive integer.")
    if args.pair_workers < 1:
        parser.error("--pair-workers must be >= 1.")
    if any(getattr(args, name) < 0 for name in ("max_signatures_per_band", "report_limit", "progress_every")):
        parser.error("Signature limits, report-limit and progress-every must be nonnegative.")
    if args.min_same_frame_occurrences is not None:
        parser.error("Use --min-frame-occurrence instead of --min-same-frame-occurrences.")
    return args


def run_synthetic_tests() -> None:
    """Quick checks for global frame support, mining prune, and frame-based pairs."""
    test_top_k = 20
    full_top20 = sum(1 << i for i in range(test_top_k))

    # Distinct top-20 within a frame: one trie lookup fans out to many patch IDs.
    sig_a = PotentialSignature(sum(1 << i for i in range(5)), 1, 0)
    sig_b = PotentialSignature(sum(1 << i for i in range(5, 10)), 1, 0)
    matcher = SignatureMatcher([sig_a, sig_b], match_cache_size=128)
    class _NoTqdm:
        no_tqdm = True
    inverted = build_inverted_signature_index([sig_a, sig_b],
                                              GlobalTransactionIndex([full_top20], {full_top20: 0}, [1],
                                                                       build_global_postings([full_top20])),
                                              _NoTqdm())
    entries = [(10, 0), (11, 0), (12, 0)]
    sig_patches, matching = build_sig_patches_for_frame_patches(entries, inverted)
    assert matching == {10, 11, 12}
    assert len(sig_patches) >= 1

    # Frame-local pair validity.
    one_patch = {0: {7}, 1: {7}, 2: {7}}
    assert list(iter_valid_signature_pairs_in_frame(one_patch, {7})) == []
    two_patches = {0: {1}, 1: {2}}
    assert list(iter_valid_signature_pairs_in_frame(two_patches, {1, 2})) == [(0, 1)]
    same_sole = {0: {3}, 1: {3}}
    assert list(iter_valid_signature_pairs_in_frame(same_sole, {3})) == []

    n = 4
    store = ObservedPairStore(spill_threshold=10_000)
    store.increment(encode_pair_key(0, 1, n), 1)
    store.increment(encode_pair_key(0, 1, n), 2)
    store.increment(encode_pair_key(1, 2, n), 1)
    assert decode_pair_key(encode_pair_key(2, 0, n), n) == (0, 2)
    result = FramePairCountResult(store, n, [0] * n, [0] * n, {"v": 0}, 1, 2, 1)
    rows = result.qualifying_pairs(2)
    assert len(rows) == 1 and rows[0][:3] == (0, 1, 2)
    store.close()

    gindex = GlobalTransactionIndex([full_top20], {full_top20: 0}, [0b011],
                                    build_global_postings([full_top20]))
    assert global_frames_meets_threshold(frozenset([0]), gindex.frame_masks, 2)
    exact = make_global_frame_counter(gindex.frame_masks)
    assert exact(frozenset([0])) == 2

    band = SignatureBand("high")
    band.frequencies[full_top20] = 2
    kept = mine_subsets(band, 5, 2, 2, gindex, limit=0, progress_every=0)
    subset_key = sum(1 << i for i in range(5))
    assert subset_key in kept.frequencies

    print("run_synthetic_tests: all checks passed.", flush=True)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--self-test" in argv:
        run_synthetic_tests()
        return 0
    args = parse_args(argv)
    try:
        if not args.patch_data.is_file():
            raise ValueError("Input file does not exist: {}".format(args.patch_data))
        before = args.patch_data.stat()

        def records():
            current = args.patch_data.stat()
            if (current.st_size, current.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise ValueError("Input changed during analysis; rerun on a stable export.")
            return iter_patch_records(args.patch_data)

        analyze(records, args)
        return 0
    except (ValueError, OSError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        raise SystemExit(0)
