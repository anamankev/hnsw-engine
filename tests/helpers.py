"""Shared test helpers, including an independent pure-Python k-NN implementation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def pure_python_knn(
    data: list[list[float]], query: list[float], k: int, metric: str
) -> list[tuple[float, int]]:
    """Textbook k-NN with Python floats only: no NumPy, so it shares no code with the oracle."""
    scored: list[tuple[float, int]] = []
    q_norm = math.sqrt(sum(v * v for v in query))
    for i, row in enumerate(data):
        if metric == "l2":
            dist = sum((a - b) ** 2 for a, b in zip(row, query, strict=True))
        elif metric == "ip":
            dist = -sum(a * b for a, b in zip(row, query, strict=True))
        else:
            dot = sum(a * b for a, b in zip(row, query, strict=True))
            dist = 1.0 - dot / (math.sqrt(sum(v * v for v in row)) * q_norm)
        scored.append((dist, i))
    scored.sort()
    return scored[:k]


def recall(found: NDArray[np.int64], truth: NDArray[np.int64]) -> float:
    """Mean fraction of true neighbours recovered per query (id-based)."""
    hits = [
        len(set(f[f >= 0].tolist()) & set(t.tolist())) for f, t in zip(found, truth, strict=True)
    ]
    return float(np.mean(hits) / truth.shape[1])


def assert_same_neighbors(
    ids_a: NDArray[np.int64],
    dist_a: NDArray[np.float64],
    ids_b: NDArray[np.int64],
    dist_b: NDArray[np.float64],
    *,
    atol: float = 1e-9,
) -> None:
    """Assert two k-NN results agree up to floating-point near-ties.

    Distances must match position by position. Ids strictly closer than the k-th distance must
    be the same sets; ids tied (within ``atol``) with the k-th distance may legitimately differ,
    because which member of a tie makes the cut depends on last-bit rounding.
    """
    np.testing.assert_allclose(dist_a, dist_b, rtol=0, atol=atol)
    for row in range(ids_a.shape[0]):
        boundary = max(dist_a[row, -1], dist_b[row, -1]) - atol
        inside_a = set(ids_a[row][dist_a[row] < boundary].tolist())
        inside_b = set(ids_b[row][dist_b[row] < boundary].tolist())
        assert inside_a == inside_b, row
