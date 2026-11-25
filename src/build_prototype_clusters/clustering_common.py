"""Shared sampled-FINCH and medoid clustering.

Descriptor providers implement take(indices), distance(a, b, weights), and node_ids.
All returned assignments use real medoids and deterministic medoid self-ownership.
The FINCH score is the existing simplified medoid silhouette on discovery labels.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm


def assign(data, rows, medoids, weights, batch_size):
    """Closest-medoid assignments with deterministic ties and self ownership."""
    rows = np.asarray(rows, dtype=np.int64)
    refs = data.take(medoids)
    labels = np.empty(len(rows), dtype=np.int32)
    distances = np.empty(len(rows), dtype=np.float64)
    for start in tqdm(
        range(0, len(rows), batch_size), desc="Assign sample", unit="batch", leave=False
    ):
        ids = rows[start : start + batch_size]
        costs = data.distance(data.take(ids), refs, weights)
        chosen = costs.argmin(axis=1)
        # Each distinct medoid owns itself. This only resolves zero-distance
        # ties; it prevents identical input patches from emptying K-1 clusters.
        for j, medoid in enumerate(medoids):
            own = ids == medoid
            chosen[own] = j
            costs[own, j] = 0.0
        labels[start : start + len(ids)] = chosen
        distances[start : start + len(ids)] = costs[np.arange(len(ids)), chosen]
    return labels, distances


def choose_fit_sample(con, size, per_video, rng):
    """Reservoir sampling per video, then cap the combined discovery sample."""
    reservoirs, seen = {}, {}
    sql = """SELECT p.descriptor_index,f.dataset,f.video FROM patches p
             JOIN frames f ON f.id=p.frame_id ORDER BY p.descriptor_index"""
    for row in tqdm(con.execute(sql), desc="Fit-sample reservoir", unit="patch"):
        key = (row["dataset"], row["video"])
        count = seen.get(key, 0) + 1
        seen[key] = count
        bucket = reservoirs.setdefault(key, [])
        if per_video == 0 or len(bucket) < per_video:
            bucket.append(int(row["descriptor_index"]))
        else:
            position = int(rng.integers(count))
            if position < per_video:
                bucket[position] = int(row["descriptor_index"])
    pool = np.array(
        [i for key in sorted(reservoirs) for i in reservoirs[key]], dtype=np.int64
    )
    if len(pool) > size:
        pool = rng.choice(pool, size=size, replace=False)
    return np.sort(pool), len(seen)


def initialize_medoids(data, sample, k, weights, rng, batch_size):
    # Distance-weighted seeding for this composite distance, not an assumption
    # that Euclidean k-means minimizes a cosine+MAE objective.
    first = int(rng.choice(sample))
    medoids = [first]
    closest = np.full(len(sample), np.inf)
    for j in tqdm(range(k), desc="Seed medoids", unit="medoid", leave=False):
        for start in range(0, len(sample), batch_size):
            ids = sample[start : start + batch_size]
            d = data.distance(data.take(ids), data.take([medoids[-1]]), weights)[:, 0]
            closest[start : start + len(ids)] = np.minimum(
                closest[start : start + len(ids)], d
            )
        if j + 1 == k:
            break
        available = ~np.isin(sample, medoids)
        probabilities = np.where(available, closest, 0.0)
        total = float(probabilities.sum())
        next_id = (
            int(rng.choice(sample, p=probabilities / total))
            if total > 1e-12
            else int(rng.choice(sample[available]))
        )
        medoids.append(next_id)
    return np.array(medoids, dtype=np.int64)


def fit_medoids(data, sample, args, weights):
    best = None
    history = []
    for restart in tqdm(range(args.restarts), desc="K-medoids restarts", unit="restart"):
        rng = np.random.default_rng(args.seed + 1009 * (restart + 1))
        medoids = initialize_medoids(
            data, sample, args.num_prototypes, weights, rng, args.batch_size
        )
        for iteration in tqdm(
            range(args.iterations + 1),
            desc=f"Restart {restart + 1}/{args.restarts}",
            unit="iter",
            leave=False,
        ):
            labels, distances = assign(data, sample, medoids, weights, args.batch_size)
            objective = float(distances.mean())
            history.append(
                {"restart": restart, "iteration": iteration, "mean_distance": objective}
            )
            tqdm.write(
                f"Fit {restart + 1}/{args.restarts}, iteration {iteration}: mean distance {objective:.6f}"
            )
            if iteration == args.iterations:
                break
            updated = medoids.copy()
            for j in range(args.num_prototypes):
                members = sample[labels == j]
                if not len(members):
                    raise RuntimeError("Internal error: medoid self-ownership failed.")
                other = members[members != medoids[j]]
                if len(other) > args.medoid_candidates - 1:
                    other = rng.choice(
                        other, size=args.medoid_candidates - 1, replace=False
                    )
                candidates = np.sort(np.append(other, medoids[j]))
                costs = np.zeros(len(candidates), dtype=np.float64)
                for start in range(0, len(members), args.batch_size):
                    costs += data.distance(
                        data.take(candidates),
                        data.take(members[start : start + args.batch_size]),
                        weights,
                    ).sum(axis=1)
                # Keeping the current medoid among candidates makes each
                # update non-worsening up to floating-point roundoff.
                updated[j] = candidates[int(costs.argmin())]
            if np.array_equal(updated, medoids):
                break
            medoids = updated
        # Objective above always corresponds to the current medoids.
        key = (objective, tuple(sorted(int(x) for x in data.node_ids[medoids])))
        if best is None or key < best[0]:
            best = (key, medoids.copy())
    medoids = best[1]
    # Stable vocabulary ordering: ascending source node ID of each medoid.
    medoids = medoids[np.argsort(data.node_ids[medoids], kind="stable")]
    return medoids, history, best[0][0]


def first_neighbors(data, reps, weights, batch_size):
    """Exact first neighbour (self excluded) of every representative under the composite distance."""
    refs = data.take(reps)
    nn = np.empty(len(reps), dtype=np.int64)
    for start in tqdm(
        range(0, len(reps), batch_size),
        desc="FINCH neighbors",
        unit="batch",
        leave=False,
    ):
        stop = min(start + batch_size, len(reps))
        costs = data.distance(tuple(a[start:stop] for a in refs), refs, weights)
        costs[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nn[start:stop] = costs.argmin(axis=1)
    return nn


def cluster_representative(data, members, weights, rng, candidates, cap):
    """Real patch (descriptor index) of a cluster with the lowest summed distance to its members.

    Exact for small clusters; large clusters use sampled candidates / sampled members.
    """
    if len(members) == 1:
        return int(members[0])
    cand = (
        members
        if len(members) <= candidates
        else rng.choice(members, size=candidates, replace=False)
    )
    pool = (
        members if len(members) <= cap else rng.choice(members, size=cap, replace=False)
    )
    costs = data.distance(data.take(cand), data.take(pool), weights).sum(axis=1)
    return int(cand[int(costs.argmin())])


def finch_partitions(data, sample, weights, args, rng):
    """FINCH hierarchy (Sarfraz et al., CVPR 2019) with the composite distance.

    Level 0 links every sample patch to its first (nearest) neighbour; connected
    components of that graph are the first partition. Each later level repeats this
    on one representative patch per cluster. Representatives are real medoid patches
    (the composite distance has no meaningful mean), not feature means as in the
    original FINCH. Returns partitions finest to coarsest.
    """
    labels = np.arange(len(sample), dtype=np.int32)
    reps = np.asarray(sample, dtype=np.int64).copy()
    partitions = []
    with tqdm(desc="FINCH levels", unit="level") as pbar:
        while len(reps) > 1:
            m = len(reps)
            nn = first_neighbors(data, reps, weights, args.batch_size)
            # Edge i - nn[i]; components of this graph equal FINCH's adjacency
            # (i->j, j->i, or shared first neighbour) components.
            graph = coo_matrix(
                (np.ones(m, dtype=np.int8), (np.arange(m), nn)), shape=(m, m)
            )
            k, comp = connected_components(graph, directed=False)
            labels = comp[labels].astype(np.int32)
            order = np.argsort(labels, kind="stable")
            bounds = np.flatnonzero(np.diff(labels[order])) + 1
            groups = np.split(np.asarray(sample)[order], bounds)  # groups[j] <-> label j
            reps = np.array(
                [
                    cluster_representative(
                        data, g, weights, rng, args.medoid_candidates, 512
                    )
                    for g in tqdm(
                        groups, desc="Cluster reps", unit="cluster", leave=False
                    )
                ],
                dtype=np.int64,
            )
            partitions.append({"k": int(k), "labels": labels.copy(), "reps": reps.copy()})
            pbar.set_postfix(k=f"{k:,}")
            pbar.update(1)
            tqdm.write(f"FINCH level {len(partitions) - 1}: {k:,} clusters")
            if k >= m:  # defensive; cannot happen for m >= 2
                break
    return partitions


def score_partition(data, sample, labels, reps, weights, batch_size):
    """Mean distance to own representative and a medoid-based simplified silhouette."""
    refs = data.take(reps)
    own_sum, sil_sum = 0.0, 0.0
    for start in tqdm(
        range(0, len(sample), batch_size),
        desc="Score partition",
        unit="batch",
        leave=False,
    ):
        stop = min(start + batch_size, len(sample))
        costs = data.distance(data.take(sample[start:stop]), refs, weights)
        pos = np.arange(stop - start)
        own = costs[pos, labels[start:stop]].copy()
        costs[pos, labels[start:stop]] = np.inf
        other = costs.min(axis=1)
        denom = np.maximum(own, other)
        sil_sum += float(
            np.where(
                denom > 1e-12, (other - own) / np.where(denom > 1e-12, denom, 1.0), 0.0
            ).sum()
        )
        own_sum += float(own.sum())
    n = len(sample)
    return own_sum / n, sil_sum / n


def fit_finch(data, sample, args, weights):
    """Run FINCH and pick one partition. Returns (medoids, history, mean_distance, info)."""
    rng = np.random.default_rng(args.seed + 1009)
    parts = finch_partitions(data, sample, weights, args, rng)
    if not parts:
        raise ValueError("FINCH produced no partitions (fit sample too small).")
    history = []
    for level, p in tqdm(list(enumerate(parts)), desc="Score FINCH levels", unit="level"):
        row = {
            "level": level,
            "num_clusters": p["k"],
            "mean_distance": None,
            "silhouette": None,
        }
        if p["k"] >= 2:
            row["mean_distance"], row["silhouette"] = score_partition(
                data, sample, p["labels"], p["reps"], weights, args.batch_size
            )
        history.append(row)
    max_k = args.max_clusters if args.max_clusters else math.inf
    if args.finch_level is not None:
        if not 0 <= args.finch_level < len(parts):
            raise ValueError(
                f"--finch-level must be in [0, {len(parts) - 1}] (0 = finest); "
                f"cluster counts per level: {[p['k'] for p in parts]}"
            )
        chosen, rule = args.finch_level, "explicit_level"
    else:
        eligible = [
            r
            for r in history
            if r["silhouette"] is not None
            and args.min_clusters <= r["num_clusters"] <= max_k
        ]
        if not eligible:
            raise ValueError(
                f"No FINCH level has {args.min_clusters}..{args.max_clusters or 'inf'} clusters; "
                f"counts per level: {[p['k'] for p in parts]}. Adjust --min/--max-clusters or --finch-level."
            )
        # Highest silhouette; ties go to fewer clusters.
        chosen = max(
            eligible, key=lambda r: (round(r["silhouette"], 9), -r["num_clusters"])
        )["level"]
        rule = "max_simplified_silhouette_within_bounds"
    for row in history:
        row["selected"] = row["level"] == chosen
        print(
            "FINCH level {level}: k={num_clusters:,} mean_dist={mean_distance} silhouette={silhouette}{mark}".format(
                mark="  <== selected" if row["selected"] else "", **row
            ),
            flush=True,
        )
    medoids = parts[chosen]["reps"]
    medoids = medoids[
        np.argsort(data.node_ids[medoids], kind="stable")
    ]  # stable vocabulary order
    mean_distance = history[chosen]["mean_distance"]
    info = {
        "selection_rule": rule,
        "selected_level": chosen,
        "num_clusters": int(len(medoids)),
        "min_clusters": args.min_clusters,
        "max_clusters": args.max_clusters,
        "cluster_counts_per_level": [p["k"] for p in parts],
    }
    return (
        medoids,
        history,
        (0.0 if mean_distance is None else float(mean_distance)),
        info,
    )
