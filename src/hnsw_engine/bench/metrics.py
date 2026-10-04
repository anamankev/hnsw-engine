"""Benchmark metrics: tie-aware recall and latency summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

RELATIVE_TIE_TOLERANCE = 1e-9


def recall_at_k(
    found_ids: NDArray[np.int64],
    found_exact_dist: NDArray[np.float64],
    true_ids: NDArray[np.int64],
    true_dist: NDArray[np.float64],
    k: int,
) -> dict[str, float]:
    """Recall@k for a batch of queries, in two flavours.

    * ``recall``: a returned id counts if it is a valid id, not a repeat, and its exact distance
      is at most the exact k-th neighbour distance (relative tolerance ``1e-9``). Equally close
      points are interchangeable, which matters on integer-valued data such as SIFT where exact
      ties are common. This follows the ANN-Benchmarks definition.
    * ``recall_ids``: plain id overlap with the oracle's top-k (ties broken by lower id).

    Args:
        found_ids: ``(nq, >=k)`` ids returned by the index (``-1`` for missing).
        found_exact_dist: exact distances of ``found_ids`` (``inf`` for missing).
        true_ids: ``(nq, >=k)`` exact neighbour ids.
        true_dist: ``(nq, >=k)`` exact neighbour distances.
        k: cut-off.
    """
    found = found_ids[:, :k]
    dist = found_exact_dist[:, :k]
    kth = true_dist[:, k - 1 : k]
    threshold = kth + RELATIVE_TIE_TOLERANCE * np.maximum(1.0, np.abs(kth))
    hits = np.zeros(found.shape[0])
    id_hits = np.zeros(found.shape[0])
    for row in range(found.shape[0]):
        seen: set[int] = set()
        for col in range(k):
            node = int(found[row, col])
            if node < 0 or node in seen:
                continue
            seen.add(node)
            if dist[row, col] <= threshold[row, 0]:
                hits[row] += 1
        id_hits[row] = len(seen & set(true_ids[row, :k].tolist()))
    return {"recall": float(hits.mean() / k), "recall_ids": float(id_hits.mean() / k)}


def latency_summary(latencies_ns: NDArray[np.int64]) -> dict[str, float]:
    """Mean and percentile latencies in milliseconds, plus queries per second (1 / mean)."""
    ms = latencies_ns.astype(np.float64) / 1e6
    mean = float(ms.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
        "qps": 1000.0 / mean,
    }


def median_of_repetitions(summaries: list[dict[str, float]]) -> dict[str, Any]:
    """Median of each statistic across repetitions (robust to one noisy repetition)."""
    keys = summaries[0].keys()
    return {key: float(np.median([s[key] for s in summaries])) for key in keys}
