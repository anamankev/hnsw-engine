"""How much recall moves between graphs built with different random seeds (SIFT 100K).

HNSW draws every node's top layer at random, so two builds with different seeds are different
graphs. The benchmarks build each library once; this script measures, for our engine, the spread
of recall@10 over five seeds at each efSearch (M = 16, efConstruction = 200, 1,000 queries), which
is the yardstick for recall differences between libraries.

Writes ``results/studies/seed_variation.json``; ``--inject`` writes the summary table.

Usage: ``python benchmarks/run_seed_variation.py [--inject docs/BENCHMARKS.md]``
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from hnsw_engine import HNSWIndex, exact_distances
from hnsw_engine.bench.env import capture_environment, ensure_single_thread
from hnsw_engine.bench.metrics import recall_at_k
from hnsw_engine.bench.report import inject
from hnsw_engine.datasets import load

SEEDS = (0, 1, 2, 3, 4)
EF_SWEEP = (10, 20, 40, 80, 160)
K = 10
OUT = Path("results/studies/seed_variation.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--inject", type=Path, help="write the summary table into this file")
    parser.add_argument("--report-only", action="store_true", help="use the saved JSON")
    args = parser.parse_args()
    if not args.report_only:
        ensure_single_thread()
        run()
    payload = json.loads(OUT.read_text())
    print(table(payload["recall"]))
    if args.inject:
        inject(args.inject, "seed_variation", table(payload["recall"]))


def run() -> None:
    data = load("sift-128-euclidean", "data", n_train=100_000, n_test=1000, k=K)
    recall: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        index = HNSWIndex(128, M=16, ef_construction=200, seed=seed)
        index.add(data.train)
        recall[str(seed)] = {}
        for ef in EF_SWEEP:
            ids, _ = index.batch_search(data.test, K, ef=ef)
            exact = exact_distances(data.train, data.test, ids, "l2")
            value = recall_at_k(ids, exact, data.neighbors, data.distances, K)["recall"]
            recall[str(seed)][str(ef)] = value
        print(seed, recall[str(seed)], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": "sift-128-euclidean",
                    "n_train": 100_000,
                    "n_queries": 1000,
                    "M": 16,
                    "ef_construction": 200,
                    "seeds": list(SEEDS),
                    "k": K,
                },
                "recall": recall,
                "environment": capture_environment(),
            },
            indent=2,
        )
    )


def table(recall: dict[str, dict[str, float]]) -> str:
    """Recall@10 range and standard deviation over the seeds, per efSearch."""
    lines = [
        "| efSearch | Min recall@10 | Max recall@10 | Range | Standard deviation |",
        "|---:|---:|---:|---:|---:|",
    ]
    efs = sorted({int(ef) for per_seed in recall.values() for ef in per_seed})
    for ef in efs:
        values: list[Any] = [per_seed[str(ef)] for per_seed in recall.values()]
        lines.append(
            f"| {ef} | {min(values):.4f} | {max(values):.4f} | {max(values) - min(values):.4f} "
            f"| {statistics.stdev(values):.4f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
