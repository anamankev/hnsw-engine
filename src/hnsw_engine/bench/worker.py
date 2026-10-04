"""Run one benchmark configuration in a fresh process and write a JSON result.

Isolation gives a clean memory measurement (resident set size before and after the build) and
contains crashes. The parent sets single-thread environment variables before this process starts,
so NumPy, FAISS, hnswlib and Numba all run on one thread.

Protocol for each query-parameter value (ANN-Benchmarks "single query" mode):
warm-up queries (untimed), then ``repetitions`` timed passes over all queries, one query per
call, timed with ``perf_counter_ns``. Recall is computed from the first pass; later passes must
return identical ids (recorded as ``deterministic_across_repetitions``).

Usage: ``python -m hnsw_engine.bench.worker config.json``
"""

from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import load
from ..oracle import exact_distances
from .adapters import make_adapter
from .env import capture_environment
from .metrics import latency_summary, median_of_repetitions, recall_at_k


def _rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # Linux reports KiB


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute one configuration; returns the result record (also written to ``config['out']``)."""
    started = time.perf_counter()
    wall_started = time.time()  # wall clock keeps running while the machine sleeps...
    thermal_before = capture_environment().get("power")
    k = int(config["k"])
    data = load(
        config["dataset"],
        config["data_dir"],
        n_train=config["n_train"],
        n_test=config["n_test"],
        k=config["k_ground_truth"],
    )
    adapter = make_adapter(config["index"], data.metric, config["params"])
    adapter.warm_up(data.train.shape[1])

    gc.collect()
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    adapter.build(data.train)
    build_seconds = time.perf_counter() - t0
    gc.collect()
    rss_after = _rss_bytes()

    sweep: list[int | None] = list(config["sweep"]) or [None]
    points: list[dict[str, Any]] = []
    truncated = False
    n_queries = data.test.shape[0]
    for value in sweep:
        if value is not None:
            adapter.set_query_param(value)
        for i in range(min(int(config["warmup_queries"]), n_queries)):
            adapter.query(data.test[i], k)
        repetitions: list[dict[str, float]] = []
        first_ids: np.ndarray | None = None
        deterministic = True
        counters = None
        for rep in range(int(config["repetitions"])):
            if rep == 0:
                adapter.reset_counters()
            latencies = np.empty(n_queries, dtype=np.int64)
            ids = np.empty((n_queries, k), dtype=np.int64)
            for i in range(n_queries):
                start = time.perf_counter_ns()
                found = adapter.query(data.test[i], k)
                latencies[i] = time.perf_counter_ns() - start
                ids[i] = found[:k]
            if rep == 0:
                counters = adapter.counters()
                first_ids = ids
            else:
                assert first_ids is not None
                deterministic &= bool(np.array_equal(first_ids, ids))
            repetitions.append(latency_summary(latencies))
        assert first_ids is not None
        exact = exact_distances(data.train, data.test, first_ids, data.metric)
        point: dict[str, Any] = {
            "query_param": adapter.query_param,
            "value": value,
            **recall_at_k(first_ids, exact, data.neighbors, data.distances, k),
            **median_of_repetitions(repetitions),
            "repetitions": repetitions,
            "deterministic_across_repetitions": deterministic,
        }
        if counters is not None:
            point["distance_computations_per_query"] = counters["distance_computations"] / n_queries
            point["hops_per_query"] = counters["hops"] / n_queries
        points.append(point)
        if time.perf_counter() - started > float(config["time_budget_seconds"]):
            truncated = value != sweep[-1]
            break

    record = {
        "config": config,
        "index": adapter.describe(),
        "dataset": {
            "name": data.name,
            "metric": data.metric,
            "n_train": int(data.train.shape[0]),
            "n_test": n_queries,
            "dim": int(data.train.shape[1]),
            "source_sha256": data.source_sha256,
        },
        "build_seconds": build_seconds,
        "index_rss_bytes": rss_after - rss_before,
        "peak_rss_bytes": _peak_rss_bytes(),
        "logical_bytes": adapter.logical_bytes(),
        "points": points,
        "truncated_by_time_budget": truncated,
        "worker_seconds": time.perf_counter() - started,
        # ...while perf_counter (mach_absolute_time on macOS) does not: a gap means the machine
        # slept during the run, so its timings cannot be trusted and the runner retries it.
        "suspended_seconds": max(
            0.0, (time.time() - wall_started) - (time.perf_counter() - started)
        ),
        "power_before": thermal_before,
        "environment": capture_environment(),
    }
    out = Path(config["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    """Entry point: ``python -m hnsw_engine.bench.worker config.json``."""
    config = json.loads(Path(sys.argv[1]).read_text())
    run(config)


if __name__ == "__main__":
    main()
