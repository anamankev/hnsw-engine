"""Derived comparisons quoted in BENCHMARKS.md, computed from the raw result files.

For each dataset: the largest recall difference between our engines and FAISS / hnswlib at equal
efSearch, the range of our distance computations relative to FAISS, throughput ratios at equal
efSearch and at recall targets, build-time ratios, and whether the two engines agree exactly
(runs without the pure-Python engine, such as SIFT 1M, skip the engine comparisons). Also the
change from SIFT 100K to SIFT 1M for every index measured at both sizes.
Writes ``results/ann/derived_stats.json``.

Usage: ``python benchmarks/derive_stats.py``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hnsw_engine.bench.report import load_results

RUNS = {
    "sift100k": "results/ann/sift-128-euclidean-n100000-main",
    "fmnist60k": "results/ann/fashion-mnist-784-euclidean-n60000-main",
    "glove100k": "results/ann/glove-100-angular-n100000-main",
    "sift1m": "results/ann/sift-128-euclidean-nfull-main",
}
TARGETS = (0.90, 0.95, 0.99)


def by_ef(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Points of one index keyed by query-parameter value."""
    return {p["value"]: p for p in result["points"]}


def best_qps(result: dict[str, Any], recall: float) -> float | None:
    """Highest measured QPS with recall at least ``recall``."""
    ok = [p["qps"] for p in result["points"] if p["recall"] >= recall]
    return max(ok) if ok else None


def ratios_at_targets(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """QPS of FAISS and hnswlib divided by ours at each recall target (both must reach it)."""
    out: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        ours = best_qps(results["hnsw-engine-numba"], target)
        ratios = {}
        for other in ("faiss-hnsw", "hnswlib"):
            theirs = best_qps(results[other], target)
            if ours is not None and theirs is not None:
                ratios[f"{other}_over_numba"] = theirs / ours
        out[f"{target:.2f}"] = ratios
    return out


def scaling(small: dict[str, dict[str, Any]], large: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """How build time and QPS at recall >= 0.95 change from SIFT 100K to SIFT 1M."""
    common = sorted(set(small) & set(large))
    qps = {k: (best_qps(small[k], 0.95), best_qps(large[k], 0.95)) for k in common}
    return {
        "build_seconds_ratio_1m_over_100k": {
            k: large[k]["build_seconds"] / small[k]["build_seconds"]
            for k in common
            if small[k]["build_seconds"] > 0.1
        },
        "qps_at_0.95_ratio_100k_over_1m": {
            k: a / b for k, (a, b) in qps.items() if a is not None and b is not None
        },
        "numba_over_flat_at_0.95": {
            "sift100k": qps["hnsw-engine-numba"][0] / qps["faiss-flat"][0],  # type: ignore[operator]
            "sift1m": qps["hnsw-engine-numba"][1] / qps["faiss-flat"][1],  # type: ignore[operator]
        },
    }


def main() -> None:
    out: dict[str, Any] = {}
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    for name, run in RUNS.items():
        if not Path(run).is_dir():  # SIFT 1M is a Tier 3 run (make bench-sift1m)
            print(f"skipping {name}: {run} not found")
            continue
        results = {r["index"]["name"]: r for r in load_results([Path(run)])}
        loaded[name] = results
        ours = by_ef(results["hnsw-engine-numba"])
        faiss, hnswlib = by_ef(results["faiss-hnsw"]), by_ef(results["hnswlib"])
        efs = sorted(ours)
        dist_ratio = [
            ours[e]["distance_computations_per_query"] / faiss[e]["distance_computations_per_query"]
            for e in efs
        ]
        stats: dict[str, Any] = {
            "max_abs_recall_diff_vs_faiss": max(
                abs(ours[e]["recall"] - faiss[e]["recall"]) for e in efs
            ),
            "max_abs_recall_diff_vs_hnswlib": max(
                abs(ours[e]["recall"] - hnswlib[e]["recall"]) for e in efs
            ),
            "fewer_distances_than_faiss_min_pct": 100 * (1 - max(dist_ratio)),
            "fewer_distances_than_faiss_max_pct": 100 * (1 - min(dist_ratio)),
            "qps_ratio_faiss_over_numba_by_ef": {e: faiss[e]["qps"] / ours[e]["qps"] for e in efs},
            "qps_ratio_hnswlib_over_numba_by_ef": {
                e: hnswlib[e]["qps"] / ours[e]["qps"] for e in efs
            },
            "recall_numba_minus_faiss_by_ef": {
                e: ours[e]["recall"] - faiss[e]["recall"] for e in efs
            },
            "recall_numba_minus_hnswlib_by_ef": {
                e: ours[e]["recall"] - hnswlib[e]["recall"] for e in efs
            },
            "qps_ratios_at_recall_targets": ratios_at_targets(results),
            "build_seconds": {k: r["build_seconds"] for k, r in results.items()},
            "build_ratio_numba_over_faiss": results["hnsw-engine-numba"]["build_seconds"]
            / results["faiss-hnsw"]["build_seconds"],
            "build_ratio_numba_over_hnswlib": results["hnsw-engine-numba"]["build_seconds"]
            / results["hnswlib"]["build_seconds"],
            "qps_at_recall_0.95": {k: best_qps(r, 0.95) for k, r in results.items()},
            "max_recall": {k: max(p["recall"] for p in r["points"]) for k, r in results.items()},
        }
        q = stats["qps_at_recall_0.95"]
        stats["at_0.95_faiss_over_numba"] = q["faiss-hnsw"] / q["hnsw-engine-numba"]
        stats["at_0.95_hnswlib_over_numba"] = q["hnswlib"] / q["hnsw-engine-numba"]
        stats["at_0.95_numba_over_flat"] = q["hnsw-engine-numba"] / q["faiss-flat"]
        if "hnsw-engine-reference" in results:  # absent from the SIFT 1M run (D18)
            ref = by_ef(results["hnsw-engine-reference"])
            stats["engines_identical"] = all(
                ours[e]["recall"] == ref[e]["recall"]
                and ours[e]["distance_computations_per_query"]
                == ref[e]["distance_computations_per_query"]
                for e in efs
            )
            stats["speedup_numba_over_reference_by_ef"] = {
                e: ours[e]["qps"] / ref[e]["qps"] for e in efs
            }
            stats["build_speedup_numba_over_reference"] = (
                results["hnsw-engine-reference"]["build_seconds"]
                / results["hnsw-engine-numba"]["build_seconds"]
            )
            stats["at_0.95_numba_over_reference"] = (
                q["hnsw-engine-numba"] / q["hnsw-engine-reference"]
            )
        out[name] = stats
    if {"sift100k", "sift1m"} <= loaded.keys():
        out["scaling_sift100k_to_sift1m"] = scaling(loaded["sift100k"], loaded["sift1m"])
    path = Path("results/ann/derived_stats.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
