"""Verify downloaded datasets and cross-check our exact oracle against the provided ground truth.

For each dataset: file size, SHA-256, shapes, dtypes, non-finite values and zero rows; then the
agreement between our oracle's top-10 and the ANN-Benchmarks top-10 for the first ``--queries``
test vectors on the *full* training set (the only setting where the provided ground truth is
valid). Writes ``results/datasets/<name>.json``.

Usage::

    python benchmarks/verify_datasets.py sift-128-euclidean --queries 1000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from hnsw_engine import exact_distances
from hnsw_engine.bench.env import capture_environment
from hnsw_engine.datasets import DATASETS, dataset_path, load, sha256_file, verify


def cross_check(name: str, data_dir: Path, queries: int, k: int = 10) -> dict[str, object]:
    """Compare our exact top-k with the provided top-k on the full training set."""
    cached = any((data_dir / "ground_truth").glob(f"{name}-n*-q{queries}-k{k}-*.npz"))
    start = time.perf_counter()
    ours = load(name, data_dir, n_train=None, n_test=queries, k=k)
    seconds = time.perf_counter() - start
    with h5py.File(dataset_path(name, data_dir), "r") as handle:
        provided_ids = handle["neighbors"][:queries, :k].astype(np.int64)
        provided_dist = handle["distances"][:queries, :k].astype(np.float64)
    ours_dist = np.sqrt(ours.distances) if ours.metric == "l2" else ours.distances
    same_list = (ours.neighbors == provided_ids).all(axis=1)
    same_members = np.array(
        [
            set(a.tolist()) == set(b.tolist())
            for a, b in zip(ours.neighbors, provided_ids, strict=True)
        ]
    )
    overlap = [len(set(a) & set(b)) / k for a, b in zip(ours.neighbors, provided_ids, strict=True)]
    kth_close = np.isclose(ours_dist[:, -1], provided_dist[:, -1], rtol=1e-5, atol=1e-6)
    # Re-score the provided ids with our exact float64 distances: if every provided neighbour is
    # within our exact k-th distance and both lists have the same sorted distances, the two lists
    # are equally valid exact answers that differ only in which tied points they include.
    rescored = exact_distances(ours.train, ours.test, provided_ids, ours.metric)
    tol = 1e-9 * np.maximum(1.0, np.abs(ours.distances[:, -1:]))
    provided_valid = (rescored <= ours.distances[:, -1:] + tol).all(axis=1)
    same_profile = (np.abs(np.sort(rescored, axis=1) - ours.distances) <= tol).all(axis=1)
    excess = np.maximum(rescored - ours.distances[:, -1:], 0.0).max()
    return {
        "queries": queries,
        "k": k,
        "ground_truth_was_cached": cached,
        "load_and_ground_truth_seconds": round(seconds, 3),
        "identical_top_k_lists": float(same_list.mean()),
        "non_identical_same_members_different_order": int((~same_list & same_members).sum()),
        "non_identical_different_members": int((~same_members).sum()),
        "mean_top_k_overlap": float(np.mean(overlap)),
        "kth_distance_agrees": float(kth_close.mean()),
        "provided_lists_valid_under_exact_rescoring": float(provided_valid.mean()),
        "same_sorted_distances_after_rescoring": float(same_profile.mean()),
        "max_provided_excess_over_exact_kth": float(excess),
        "max_abs_distance_difference": float(np.abs(ours_dist - provided_dist).max()),
        "tolerance": "1e-9 * max(1, exact k-th distance), in our distance units",
        "note": "provided euclidean distances are L2; ours are squared L2 (compared after sqrt)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", default=sorted(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("results/datasets"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        path = dataset_path(name, args.data_dir)
        if not path.exists():
            print(f"skip {name}: {path} not found (run python -m hnsw_engine.datasets download)")
            continue
        info = DATASETS[name]
        record = {
            "dataset": name,
            "url": info.url,
            "metric": info.metric,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "verification": verify(path),
            "cross_check": cross_check(name, args.data_dir, args.queries),
            "environment": capture_environment(),
        }
        out = args.out / f"{name}.json"
        out.write_text(json.dumps(record, indent=2))
        print(json.dumps({k: record[k] for k in ("dataset", "size_bytes", "sha256")}))
        print(json.dumps(record["cross_check"], indent=2))


if __name__ == "__main__":
    main()
