#!/usr/bin/env python3
"""Discover contextual prototype-signature comparison candidates; print only.

Input: the JSON array written by prototype_consistency_sal.py. No model, GPU,
third-party packages, CSV, JSON output, or temporary result files are needed.

Default experiment:
  * select the highest 3% and lowest 50% of patch_pred_saliency across the dataset;
  * encode exported top-20 IDs as a 512-bit binary activation signature;
  * group similar signatures independently within the high and low bands;
  * require at least ceil(20 * overlap_percent / 100) shared IDs between EVERY
    pair of signatures within a group (25% means five shared IDs);
  * count high-group/low-group co-occurrences within physical source frames.

The packed signature is exactly equivalent to a length-512 boolean vector.
as_binary_vector() exposes that vector when needed. Actual lists are never
padded. Rank selects membership; strength and order are not signature features.

Grouping is deterministic greedy complete-link grouping: signatures are visited
by decreasing patch frequency, ties by packed signature. A signature joins a
group only if it matches every existing member. Groups are disjoint within each
band; partial-match chains cannot merge signatures that fail the threshold.
This is not an exhaustive enumeration of every possible overlapping subgroup.

Dataset percentiles match the previous pooled analyses. --percentile-scope frame
selects the requested percentages independently within each frame instead.
Counts use ceil(percent * population / 100). The default exact tie policy uses
input order to break cutoff ties and reports their extent. --tie-policy include
keeps all cutoff ties, so selected percentages can increase. A score equal to
BOTH band cutoffs is excluded from both bands under either policy: it provides
no high-versus-low contrast. This also removes flat frames in frame mode.

A candidate is a pair of high/low GROUPS present together in at least one frame.
For each candidate, report distinct frames, distinct videos, participating high
and low patches, and sum(n_high_in_frame * n_low_in_frame). No frequency cutoff
is applied; the user can choose one after inspecting these frequencies. These
are discovery counts, not tests of ordering consistency or causal contribution.

Example:
  python contextual_prototype_consistency.py \
      --patch-data /path/to/patch_data.json --overlap-percent 25

Python 3.8+; standard library only. Two streaming passes retain numerical scores
and selected signature counts, rather than loading millions of JSON objects.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
import json
import math
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple


NUM_PROTOTYPES = 512
TOP_K = 20
FrameKey = Tuple[str, str]
RecordFactory = Callable[[], Iterable[dict]]


def popcount(bits: int) -> int:
    method = getattr(int, "bit_count", None)
    return method(bits) if method is not None else bin(bits).count("1")


def signature_ids(bits: int) -> Tuple[int, ...]:
    ids = []
    while bits:
        lowest_bit = bits & -bits
        ids.append(lowest_bit.bit_length() - 1)
        bits ^= lowest_bit
    return tuple(ids)


def as_binary_vector(bits: int) -> Tuple[bool, ...]:
    """Return the requested 512-position vector; packed storage avoids padding RAM."""
    if not isinstance(bits, int) or bits < 0 or bits.bit_length() > NUM_PROTOTYPES:
        raise ValueError("Signature must be a nonnegative 512-bit integer.")
    return tuple(bool(bits & (1 << index)) for index in range(NUM_PROTOTYPES))


def iter_json_array(handle, chunk_size: int = 1 << 20) -> Iterable[dict]:
    """Read a JSON array incrementally, including pretty-printed exports."""
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
    with path.open("r", encoding="utf-8-sig") as handle:
        yield from iter_json_array(handle)


def integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}.".format(label, minimum))
    return value


def saliency(record: dict, field_name: str) -> float:
    value = record.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a finite number.".format(field_name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("{} must be finite.".format(field_name))
    return value


def activation_signature(record: dict) -> int:
    """Use exported ranks/IDs, never invent additional matches for short lists."""
    ids = record.get("top_20_activated_prototype_indices")
    details = record.get("top_20_activated_prototypes")
    if details is not None:
        if not isinstance(details, list) or any(not isinstance(x, dict) for x in details):
            raise ValueError("top_20_activated_prototypes must be a list of objects.")
        if any("rank" in x for x in details):
            ranks = [integer(x.get("rank"), "prototype rank", 1) for x in details]
            if len(set(ranks)) != len(ranks):
                raise ValueError("Duplicate prototype ranks in one patch.")
            details = sorted(details, key=lambda x: x["rank"])
        extracted = [integer(x.get("prototype_index"), "prototype index") for x in details]
        if ids is not None and ids != extracted:
            raise ValueError("Ranked prototype details and index list disagree.")
        ids = extracted
    if not isinstance(ids, list):
        raise ValueError("Missing exported top-20 prototype list.")
    ids = [integer(x, "prototype index") for x in ids]
    if len(ids) > TOP_K or len(ids) != len(set(ids)):
        raise ValueError("Expected at most 20 distinct prototype IDs per patch.")
    if any(x >= NUM_PROTOTYPES for x in ids):
        raise ValueError("Prototype indices must be in [0, 511].")
    bits = 0
    for prototype in ids:
        bits |= 1 << prototype
    return bits


class FrameResolver:
    """Resolve fields once, then prevent mixed stages/contexts within a frame."""
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
            raise ValueError("Input mixes prototype stages; analyze one stage/export at a time.")
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
            if name not in record:
                raise ValueError("Missing frame-context field {}.".format(name))
            value = record[name]
            if isinstance(value, (dict, list)):
                raise ValueError("Frame-context identifiers must be scalar values.")
            values.append(value)
        return tuple(values)


@dataclass
class FrameScan:
    context: tuple
    count: int = 0
    patch_bits: int = 0


def percent_count(count: int, percent: float) -> int:
    value = Decimal(str(percent)) * count / Decimal(100)
    return int(value.to_integral_value(rounding=ROUND_CEILING))


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
        if not n:
            raise ValueError("Cannot select percentiles from an empty population.")
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
        if high and low:
            return "ambiguous"
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
    print("Pass 1/2: reading saliency and source-frame identifiers...", flush=True)
    for number, record in enumerate(records, 1):
        try:
            if resolver is None:
                resolver = FrameResolver(record, args.video_field, args.frame_field)
            key = resolver.key(record)
            context = resolver.context(record)
            scan = frames.setdefault(key, FrameScan(context))
            if scan.context != context:
                raise ValueError("Same physical frame appears with different window contexts: {}."
                                 .format(key))
            if ("patch_index" in record) != resolver.has_patch_index:
                raise ValueError("Inconsistent patch_index availability.")
            if resolver.has_patch_index:
                patch_index = integer(record["patch_index"], "patch_index")
                if patch_index > 1_000_000:
                    raise ValueError("Unreasonably large patch_index.")
                bit = 1 << patch_index
                if scan.patch_bits & bit:
                    raise ValueError("Duplicate patch_index {} in source frame {}."
                                     .format(patch_index, key))
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
    boundaries = {key: BandBoundary.from_scores(values, args.top_percent, args.bottom_percent)
                  for key, values in scores.items()}
    return SelectionPlan(resolver, frames, boundaries, count, minimum, maximum)


@dataclass
class SignatureBand:
    name: str
    selected: int = 0
    eligible: int = 0
    size_histogram: Counter = field(default_factory=Counter)
    frequencies: Counter = field(default_factory=Counter)
    by_frame: dict = field(default_factory=lambda: defaultdict(Counter))
    minimum: float = float("inf")
    maximum: float = -float("inf")

    def add(self, bits: int, frame_key: FrameKey, value: float, min_shared: int):
        self.selected += 1
        size = popcount(bits)
        self.size_histogram[size] += 1
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)
        self.frequencies[bits] += 1
        if size >= min_shared:
            self.eligible += 1
            self.by_frame[frame_key][bits] += 1


def collect_signatures(records: Iterable[dict], plan: SelectionPlan, args, min_shared: int):
    high, low = SignatureBand("high"), SignatureBand("low")
    ambiguous = count = 0
    frame_counts = Counter()
    print("Pass 2/2: constructing selected 512-bit signatures...", flush=True)
    for number, record in enumerate(records, 1):
        try:
            key = plan.resolver.key(record)
            if key not in plan.frames or plan.frames[key].context != plan.resolver.context(record):
                raise ValueError("Frame/context changed between input passes.")
            frame_counts[key] += 1
            value = saliency(record, args.saliency_field)
            boundary = plan.boundaries[key if args.percentile_scope == "frame" else None]
            band = boundary.classify(value, args.tie_policy)
            if band == "ambiguous":
                ambiguous += 1
            elif band is not None:
                bits = activation_signature(record)
                (high if band == "high" else low).add(bits, key, value, min_shared)
            count += 1
        except (ValueError, TypeError) as exc:
            raise ValueError("Record {}: {}".format(number, exc)) from exc
        if args.progress_every and number % args.progress_every == 0:
            print("  Scanned {:,}; high {:,}, low {:,}."
                  .format(number, high.selected, low.selected), flush=True)
    if count != plan.patch_count or any(frame_counts[k] != x.count for k, x in plan.frames.items()):
        raise ValueError("Patch counts changed between input passes.")
    return high, low, ambiguous


@dataclass
class SignatureGroup:
    representative: int
    members: List[int]
    common_bits: int
    patches: int


def group_signatures(band: SignatureBand, min_shared: int, progress_every: int = 0):
    """Greedy complete-link groups; inverted representative postings prune search."""
    masks = sorted((x for x in band.frequencies if popcount(x) >= min_shared),
                   key=lambda x: (-band.frequencies[x], x))
    groups = []
    membership = {}
    representative_postings = defaultdict(list)
    for visited, bits in enumerate(masks, 1):
        ids = signature_ids(bits)
        overlaps = Counter(g for p in ids for g in representative_postings[p])
        possible = sorted((g for g, common in overlaps.items() if common >= min_shared),
                          key=lambda g: (-overlaps[g], -groups[g].patches, g))
        chosen = None
        for group_id in possible:
            group = groups[group_id]
            if popcount(bits & group.common_bits) >= min_shared or all(
                    popcount(bits & member) >= min_shared for member in group.members):
                chosen = group_id
                break
        if chosen is None:
            chosen = len(groups)
            groups.append(SignatureGroup(bits, [bits], bits, band.frequencies[bits]))
            for p in ids:
                representative_postings[p].append(chosen)
        else:
            group = groups[chosen]
            group.members.append(bits)
            group.common_bits &= bits
            group.patches += band.frequencies[bits]
        membership[bits] = chosen
        if progress_every and visited % progress_every == 0:
            print("  {}: grouped {:,} signatures into {:,} groups."
                  .format(band.name, visited, len(groups)), flush=True)
    return groups, membership


def group_frame_counts(band: SignatureBand, membership: dict):
    result = {}
    for key, signatures in band.by_frame.items():
        counts = Counter()
        for bits, n in signatures.items():
            counts[membership[bits]] += n
        result[key] = counts
    return result


@dataclass
class Candidate:
    high_group: int
    low_group: int
    frame_count: int = 0
    videos: Set[str] = field(default_factory=set)
    high_patches: int = 0
    low_patches: int = 0
    patch_pairs: int = 0


def count_candidates(high_frames: dict, low_frames: dict):
    candidates = {}
    shared_frames = 0
    for key, high_counts in high_frames.items():
        low_counts = low_frames.get(key)
        if not low_counts:
            continue
        shared_frames += 1
        for high_id, high_n in high_counts.items():
            for low_id, low_n in low_counts.items():
                pair = high_id, low_id
                candidate = candidates.get(pair)
                if candidate is None:
                    candidate = candidates[pair] = Candidate(*pair)
                candidate.frame_count += 1
                candidate.videos.add(key[0])
                candidate.high_patches += high_n
                candidate.low_patches += low_n
                candidate.patch_pairs += high_n * low_n
    return candidates, shared_frames


def displayed(items: list, limit: int):
    return items if limit == 0 else items[:limit]


def format_ids(bits: int) -> str:
    return "{" + ",".join(str(x) for x in signature_ids(bits)) + "}"


def print_band(band: SignatureBand, total: int, label: str, min_shared: int):
    print("\n{}: {:,} patches ({:.4f}% of dataset)."
          .format(label, band.selected, 100 * band.selected / total))
    if band.selected:
        print("  Selected saliency range: {:.8g} to {:.8g}.".format(band.minimum, band.maximum))
    histogram = ", ".join("{} IDs: {:,}".format(k, v)
                          for k, v in sorted(band.size_histogram.items())) or "empty"
    print("  Actual prototype-list sizes: " + histogram)
    print("  Exact signatures: {:,}; patches with >= {} IDs: {:,}; excluded short/empty: {:,}."
          .format(len(band.frequencies), min_shared, band.eligible, band.selected - band.eligible))


def print_groups(groups: list, frames: dict, prefix: str, limit: int):
    frame_support, video_support = Counter(), defaultdict(set)
    for key, counts in frames.items():
        for group_id in counts:
            frame_support[group_id] += 1
            video_support[group_id].add(key[0])
    order = sorted(range(len(groups)), key=lambda g: (-groups[g].patches, g))
    print("\n{} groups: {:,}; each member pair passes the prototype overlap threshold."
          .format("High" if prefix == "H" else "Low", len(groups)))
    print("group  signatures        patches       frames  videos  representative IDs / IDs shared by ALL members")
    for group_id in displayed(order, limit):
        g = groups[group_id]
        print("{}{:04d} {:10,d} {:14,d} {:12,d} {:7,d}  {} / {}"
              .format(prefix, group_id + 1, len(g.members), g.patches, frame_support[group_id],
                      len(video_support[group_id]), format_ids(g.representative), format_ids(g.common_bits)))
    if limit and len(order) > limit:
        print("  Showing {} of {:,} groups; --report-limit 0 prints all.".format(limit, len(order)))
    print("  The common-ID intersection can be smaller than the pairwise overlap threshold.")


def print_candidates(candidates: dict, shared_frames: int, limit: int):
    rows = sorted(candidates.values(), key=lambda c:
                  (-c.frame_count, -len(c.videos), -c.patch_pairs, c.high_group, c.low_group))
    print("\nPotential comparison candidates: {:,}; frames with both eligible bands: {:,}."
          .format(len(rows), shared_frames))
    print("No minimum candidate frequency has been applied.")
    print(" rank   high    low   same frames  videos   high patches    low patches          patch pairs")
    for rank, c in enumerate(displayed(rows, limit), 1):
        print("{:5,d}  H{:04d}  L{:04d} {:12,d} {:7,d} {:14,d} {:14,d} {:20,d}"
              .format(rank, c.high_group + 1, c.low_group + 1, c.frame_count, len(c.videos),
                      c.high_patches, c.low_patches, c.patch_pairs))
    if limit and len(rows) > limit:
        print("  Showing {} of {:,} candidates; --report-limit 0 prints all.".format(limit, len(rows)))
    print("\nCandidate-frequency guide (cumulative; no candidates are filtered):")
    print(" minimum same-frame occurrences     candidate combinations")
    support_histogram = Counter(c.frame_count for c in rows)
    for threshold in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000):
        n = sum(count for support, count in support_histogram.items() if support >= threshold)
        print("{:31,d} {:26,d}".format(threshold, n))
    print("\n'same frames' counts distinct physical (video, frame) observations.")
    print("'high/low patches' count patches in those co-occurrence frames for that candidate.")
    print("'patch pairs' sums high_count * low_count within each shared frame.")
    print("Rows can reuse frames and patches; do not sum them as independent observations.")
    print("These are potential candidates only; no saliency-ordering pattern is validated yet.")
    print("All reports were printed to the terminal. No result files were written.")


def analyze(factory: RecordFactory, args):
    min_shared = percent_count(TOP_K, args.overlap_percent)
    plan = build_selection(factory(), args)
    print("Loaded {:,} patches, {:,} frames, {:,} videos; saliency range {:.8g} to {:.8g}."
          .format(plan.patch_count, len(plan.frames), len({k[0] for k in plan.frames}),
                  plan.minimum, plan.maximum), flush=True)
    print("Frame identity: ({}, {}); percentile scope: {}; tie policy: {}."
          .format(plan.resolver.video_field, plan.resolver.frame_field,
                  args.percentile_scope, args.tie_policy))
    if not plan.resolver.has_patch_index:
        print("No patch_index field: duplicate spatial-patch checking is unavailable.")
    boundaries = list(plan.boundaries.values())
    for label, attr in (("High", "high_cutoff"), ("Low", "low_cutoff")):
        values = [getattr(x, attr) for x in boundaries]
        print("{} cutoff{}: {:.8g} to {:.8g}."
              .format(label, " range" if len(values) > 1 else "", min(values), max(values)))
    high_ties = sum(x.high_ties for x in boundaries)
    low_ties = sum(x.low_ties for x in boundaries)
    print("Patches tied at high/low cutoffs: {:,} / {:,}.".format(high_ties, low_ties))
    if args.tie_policy == "exact":
        print("Cutoff ties are broken by input order to retain the requested number.")
    else:
        print("All cutoff ties are included; actual selected fractions can exceed the requested fractions.")
    print("Minimum overlap: {:.6g}% of requested top-20 = at least {} shared IDs."
          .format(args.overlap_percent, min_shared))
    high, low, ambiguous = collect_signatures(factory(), plan, args, min_shared)
    print_band(high, plan.patch_count, "Highest {:.6g}%".format(args.top_percent), min_shared)
    print_band(low, plan.patch_count, "Lowest {:.6g}%".format(args.bottom_percent), min_shared)
    if ambiguous:
        print("Excluded {:,} patches at overlapping/indistinguishable high-low cutoffs."
              .format(ambiguous))
    print("\nGrouping signatures independently in the two bands...", flush=True)
    high_groups, high_map = group_signatures(high, min_shared, args.progress_every)
    low_groups, low_map = group_signatures(low, min_shared, args.progress_every)
    high_frames = group_frame_counts(high, high_map)
    low_frames = group_frame_counts(low, low_map)
    print_groups(high_groups, high_frames, "H", args.report_limit)
    print_groups(low_groups, low_frames, "L", args.report_limit)
    if not high_groups or not low_groups:
        print("\nNo candidates can be formed: at least one band lacks signatures with {} active IDs."
              .format(min_shared))
        print("Short lists are not padded. Check the export or explicitly change --overlap-percent.")
    print("Counting same-frame group combinations...", flush=True)
    candidates, shared_frames = count_candidates(high_frames, low_frames)
    print_candidates(candidates, shared_frames, args.report_limit)
    return plan, high, low, high_groups, low_groups, candidates


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch-data", "--patch_data", required=True, type=Path,
                        help="Path to the exported patch_data.json; input is never modified.")
    parser.add_argument("--top-percent", type=float, default=3.0, help="Highest-saliency percentage (default: 3).")
    parser.add_argument("--bottom-percent", type=float, default=50.0, help="Lowest-saliency percentage (default: 50).")
    parser.add_argument("--overlap-percent", "--min-overlap-percent", type=float, default=25.0,
                        help="Shared-ID percentage of requested top-20 (default: 25 means 5 IDs).")
    parser.add_argument("--percentile-scope", choices=("dataset", "frame"), default="dataset",
                        help="dataset matches pooled tables; frame selects within each frame.")
    parser.add_argument("--tie-policy", choices=("exact", "include"), default="exact",
                        help="exact breaks cutoff ties by input order; include retains all ties.")
    parser.add_argument("--saliency-field", default="patch_pred_saliency")
    parser.add_argument("--video-field", default=None, help="Override automatic video-identifier field detection.")
    parser.add_argument("--frame-field", default=None, help="Override automatic frame-identifier field detection.")
    parser.add_argument("--report-limit", type=int, default=50,
                        help="Printed rows per group/candidate table; 0 prints all. Counting is always exact.")
    parser.add_argument("--progress-every", type=int, default=250000,
                        help="Progress interval in records/signatures; 0 disables periodic updates.")
    args = parser.parse_args(argv)
    for name in ("top_percent", "bottom_percent", "overlap_percent"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= 100:
            parser.error("--{} must be in (0, 100].".format(name.replace("_", "-")))
    if args.top_percent + args.bottom_percent > 100:
        parser.error("Top and bottom percentages must sum to at most 100.")
    if args.report_limit < 0 or args.progress_every < 0:
        parser.error("report-limit and progress-every must be nonnegative.")
    return args


def main(argv=None) -> int:
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
