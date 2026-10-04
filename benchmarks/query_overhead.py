"""Split per-query latency into compiled search work and Python call overhead (milestone A3).

Times the same queries three ways on the Numba engine: one ``HNSWIndex.search`` call per query
(public API, validation included), one engine call per query (no validation), and one
``batch_search`` call for all queries (compiled loop, no per-query Python). Writes
``results/profile/query_overhead.json``.

Usage: ``python benchmarks/query_overhead.py --n 100000 --ef 40``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hnsw_engine import HNSWIndex
from hnsw_engine.bench.env import capture_environment
from hnsw_engine.datasets import load
from hnsw_engine.metrics import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--ef", type=int, nargs="+", default=[10, 40, 160])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    data = load("sift-128-euclidean", "data", n_train=args.n, n_test=args.queries, k=10)
    index = HNSWIndex(128, M=16, ef_construction=200, seed=0, engine="numba")
    index.add(data.train)
    prepared = prepare(data.test, "l2")
    engine = index._engine
    rows = []
    for ef in args.ef:
        index.batch_search(data.test[:10], 10, ef=ef)  # warm-up
        timings: dict[str, list[float]] = {"public_api": [], "engine_call": [], "batch": []}
        for _ in range(args.repetitions):
            t = time.perf_counter()
            for q in data.test:
                index.search(q, 10, ef=ef)
            timings["public_api"].append((time.perf_counter() - t) / args.queries)
            t = time.perf_counter()
            for q in prepared:
                engine.search(q, 10, ef)
            timings["engine_call"].append((time.perf_counter() - t) / args.queries)
            t = time.perf_counter()
            index.batch_search(data.test, 10, ef=ef)
            timings["batch"].append((time.perf_counter() - t) / args.queries)
        row = {"ef_search": ef, **{k: float(np.median(v)) * 1e6 for k, v in timings.items()}}
        row["python_overhead_us"] = row["public_api"] - row["batch"]
        rows.append(row)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()})
    out = Path("results/profile")
    out.mkdir(parents=True, exist_ok=True)
    (out / "query_overhead.json").write_text(
        json.dumps(
            {
                "unit": "microseconds per query (median of repetitions)",
                "n": args.n,
                "rows": rows,
                "environment": capture_environment(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
