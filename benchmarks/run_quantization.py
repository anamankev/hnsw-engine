"""Tier 3: int8 scalar quantization, recall against memory (``docs/DESIGN.md`` 9.2).

On the three benchmark subsets (SIFT 100K, Fashion-MNIST 60K, GloVe 100K) the float32 graph is
built once (M = 16, efConstruction = 200, seed 0) and searched three ways for each efSearch:

* ``float32``: the normal search;
* ``int8``: the same graph traversed with distances to one-byte codes, top 10 by code distance;
* ``int8_rerank``: traversal with codes, then the ``ef`` candidates the beam returns are
  re-scored with exact float32 distances.

Recall@10 is tie-aware against the exact float32 ground truth. Timing: one compiled batch over
the 1,000 queries on one thread, median of three repetitions. Memory: bytes of the vectors the
traversal reads (float32 or codes) and of the graph (link slots and levels). Re-ranking adds
``ef`` exact distance computations per query, recorded as ``reranked_per_query``.

Writes ``results/studies/quantization.json``; ``--inject`` writes the summary table.

Usage: ``python benchmarks/run_quantization.py [--inject docs/STUDIES.md]``
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from hnsw_engine import HNSWIndex, QuantizedIndex, exact_distances
from hnsw_engine.bench.env import capture_environment, ensure_single_thread
from hnsw_engine.bench.metrics import recall_at_k
from hnsw_engine.bench.report import inject
from hnsw_engine.datasets import load

DATASETS = (
    ("sift-128-euclidean", 100_000),
    ("fashion-mnist-784-euclidean", 60_000),
    ("glove-100-angular", 100_000),
)
EF_SWEEP = (10, 20, 40, 80, 160, 320)
MODES = ("float32", "int8", "int8_rerank")
K = 10
REPETITIONS = 3
OUT = Path("results/studies/quantization.json")
T = TypeVar("T")


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Result of the last of ``REPETITIONS`` calls and their median wall time in seconds."""
    seconds: list[float] = []
    for _ in range(REPETITIONS):
        start = time.perf_counter()
        result = fn()
        seconds.append(time.perf_counter() - start)
    return result, statistics.median(seconds)


def search_call(
    mode: str, ef: int, index: HNSWIndex, quantized: QuantizedIndex, queries: np.ndarray
) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """One batch search over every query in the given mode."""
    if mode == "float32":
        return functools.partial(index.batch_search, queries, K, ef=ef)
    rerank = ef if mode == "int8_rerank" else 0
    return functools.partial(quantized.batch_search, queries, K, ef=ef, rerank=rerank)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--inject", type=Path, help="write the summary table into this file")
    parser.add_argument("--report-only", action="store_true", help="use the saved JSON")
    args = parser.parse_args()
    if not args.report_only:
        ensure_single_thread()
        run()
    payload = json.loads(OUT.read_text())
    print(table(payload["rows"]))
    if args.inject:
        inject(args.inject, "study_quantization", table(payload["rows"]))


def run() -> None:
    rows: list[dict[str, Any]] = []
    for name, n_train in DATASETS:
        data = load(name, "data", n_train=n_train, n_test=1000, k=K)
        train, queries, truth = data.train, data.test, (data.neighbors, data.distances)
        index = HNSWIndex(train.shape[1], metric=data.metric, M=16, ef_construction=200, seed=0)
        index.add(train)
        quantized = QuantizedIndex(index)
        stored = index._numba_engine("study").native_arrays()["vectors"]
        decoded = quantized.quantizer.decode(quantized.codes)
        stats = index.stats()
        vector_bytes = quantized.vector_bytes
        for mode in MODES:  # compile every code path before timing
            search_call(mode, EF_SWEEP[0], index, quantized, queries)()
        row: dict[str, Any] = {
            "dataset": name,
            "n": int(train.shape[0]),
            "dim": int(train.shape[1]),
            "metric": data.metric,
            "step": quantized.quantizer.step,
            "lossless": bool(np.array_equal(decoded, stored)),
            "max_abs_reconstruction_error": float(np.abs(decoded - stored).max()),
            "vector_bytes_float32": vector_bytes,
            "vector_bytes_int8": quantized.code_bytes,
            "graph_bytes": stats.memory_bytes - vector_bytes,  # link slots and levels
        }
        for mode in MODES:
            points = []
            for ef in EF_SWEEP:
                index.reset_counters()
                (ids, _), seconds = timed(search_call(mode, ef, index, quantized, queries))
                exact = exact_distances(train, queries, ids, data.metric)
                points.append(
                    {
                        "ef_search": ef,
                        "recall": recall_at_k(ids, exact, truth[0], truth[1], K)["recall"],
                        "qps": len(queries) / seconds,
                        "distance_computations_per_query": index.stats().distance_computations
                        / (REPETITIONS * len(queries)),
                        "reranked_per_query": ef if mode == "int8_rerank" else 0,
                    }
                )
            row[mode] = points
        rows.append(row)
        print(
            name,
            f"step {row['step']:.6g} lossless {row['lossless']}",
            {m: [round(p["recall"], 4) for p in row[m]] for m in MODES},
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "config": {
                    "M": 16,
                    "ef_construction": 200,
                    "seed": 0,
                    "k": K,
                    "n_queries": 1000,
                    "repetitions": REPETITIONS,
                    "rerank_depth": "efSearch (every candidate the beam returns)",
                    "timing": "median of repetitions of one batch over all queries, one thread",
                },
                "rows": rows,
                "environment": capture_environment(),
            },
            indent=2,
        )
    )


def table(rows: list[dict[str, Any]]) -> str:
    """Markdown summary: memory, recall at efSearch 40 and 160, the largest recall drop over the
    sweep, and int8 throughput relative to float32 (range over the sweep)."""
    header = [
        "Dataset",
        "Codes lossless",
        "Vectors: float32 / int8 (MB)",
        "Graph links (MB)",
        "Recall@10 ef 40: float32 / int8 / int8 + re-rank",
        "Recall@10 ef 160: float32 / int8 / int8 + re-rank",
        "Largest recall drop vs float32: int8 / int8 + re-rank",
        "QPS vs float32: int8 / int8 + re-rank",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * 7) + "|"]
    for row in rows:

        def at(mode: str, ef: int, key: str, r: dict[str, Any] = row) -> float:
            return float(next(p[key] for p in r[mode] if p["ef_search"] == ef))

        def drop(mode: str, r: dict[str, Any] = row) -> float:
            pairs = zip(r["float32"], r[mode], strict=True)
            return float(max(a["recall"] - b["recall"] for a, b in pairs))

        def speed(mode: str, r: dict[str, Any] = row) -> str:
            ratios = [b["qps"] / a["qps"] for a, b in zip(r["float32"], r[mode], strict=True)]
            return f"{min(ratios):.2f}-{max(ratios):.2f}x"

        error = row["max_abs_reconstruction_error"]
        cells = [
            f"{row['dataset']} ({row['n']:,} x {row['dim']})",
            "yes" if row["lossless"] else f"no (max error {error:.4f})",
            f"{row['vector_bytes_float32'] / 1e6:.1f} / {row['vector_bytes_int8'] / 1e6:.1f}",
            f"{row['graph_bytes'] / 1e6:.1f}",
            " / ".join(f"{at(m, 40, 'recall'):.4f}" for m in MODES),
            " / ".join(f"{at(m, 160, 'recall'):.4f}" for m in MODES),
            f"{drop('int8'):+.4f} / {drop('int8_rerank'):+.4f}",
            f"{speed('int8')} / {speed('int8_rerank')}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
