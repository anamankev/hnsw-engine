"""Exact brute-force k-nearest-neighbour search: the oracle all approximate results are scored by.

Cost is ``O(n * d)`` per query; the job of this module is to be *exact* and still fast enough to
produce ground truth for million-vector datasets.

Algorithm (two passes, both vectorised):

1. **Filter in float32.** Distances between a block of queries and a chunk of data are computed
   with one matrix multiply (``|q - x|^2 = |q|^2 - 2 q.x + |x|^2`` for L2). Float32 rounding can
   reorder near-ties, so instead of trusting this ranking we keep every point whose approximate
   distance is within ``margin`` of the k-th smallest one. ``margin = 2 * e`` where ``e`` is a
   bound on the float32 error, derived from the standard dot-product error bound
   ``|fl(x.y) - x.y| <= d * u * |x| * |y|`` (``u = 2**-24``) with a safety factor of 4.
2. **Rank exactly in float64.** The surviving candidates are re-scored in float64 from the
   original input values and ordered by ``(distance, id)``, so ties go to the lower id.

Why this is exact: if a true top-k point ``j`` were filtered out, then ``k`` points would have
approximate distance more than ``margin`` below ``j``'s, hence exact distance more than
``margin - 2e >= 0`` below it, so ``j`` could not be in the top-k. Data is processed in chunks,
so memory stays bounded and no float64 copy of the dataset is ever made.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_matrix, check_int, check_k
from .metrics import MetricName, validate_metric

_UNIT_ROUNDOFF_F32 = 2.0**-24
_SAFETY = 4.0


def exact_search(
    data: ArrayLike,
    queries: ArrayLike,
    k: int,
    metric: str = "l2",
    *,
    query_block: int = 256,
    data_block: int = 65_536,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return the exact ``k`` nearest neighbours of each query.

    Args:
        data: ``(n, d)`` array of indexed vectors.
        queries: ``(nq, d)`` array of query vectors.
        k: number of neighbours, ``1 <= k <= n``.
        metric: ``"l2"`` (squared Euclidean), ``"cosine"`` (``1 - cos``) or ``"ip"``
            (negative inner product).
        query_block: queries processed per matrix multiply.
        data_block: data rows processed per matrix multiply.

    Returns:
        ``(ids, distances)``, each of shape ``(nq, k)``, sorted by ascending distance with ties
        broken by ascending id. Distances are computed in float64.

    Raises:
        ValueError: on empty data, shape or dimension mismatch, NaN or infinite values, zero
            vectors under ``cosine``, or ``k`` outside ``[1, n]``.
    """
    metric_name = validate_metric(metric)
    x = as_matrix(data, "data")
    n, dim = x.shape
    if n == 0 or dim == 0:
        raise ValueError(f"data must be non-empty with at least one dimension, got {x.shape}")
    q = as_matrix(queries, "queries", dim=dim)
    k = check_k(k, n)
    query_block = check_int(query_block, "query_block", minimum=1)
    data_block = check_int(data_block, "data_block", minimum=1)

    data_norms = _row_norms(x, data_block)
    query_norms = _row_norms(q, query_block)
    if metric_name == "cosine":
        _reject_zero_rows(data_norms, "data")
        _reject_zero_rows(query_norms, "queries")
    margins = 2.0 * _error_bound(metric_name, dim, query_norms, float(data_norms.max()))

    data_sq = (
        _squared_norms_f32(x, data_block, metric_name, data_norms) if metric_name == "l2" else None
    )
    out_ids = np.empty((q.shape[0], k), dtype=np.int64)
    out_dist = np.empty((q.shape[0], k), dtype=np.float64)

    for q_start in range(0, q.shape[0], query_block):
        q_stop = min(q_start + query_block, q.shape[0])
        q32 = _to_f32(q[q_start:q_stop], metric_name, query_norms[q_start:q_stop])
        q_sq = np.einsum("ij,ij->i", q32, q32) if metric_name == "l2" else None
        block_margins = margins[q_start:q_stop]
        pool_ids: list[list[NDArray[np.int64]]] = [[] for _ in range(q_stop - q_start)]
        pool_dist: list[list[NDArray[np.float32]]] = [[] for _ in range(q_stop - q_start)]

        for d_start in range(0, n, data_block):
            d_stop = min(d_start + data_block, n)
            x32 = _to_f32(x[d_start:d_stop], metric_name, data_norms[d_start:d_stop])
            approx = _approx_distances(
                metric_name,
                q32,
                x32,
                q_sq,
                None if data_sq is None else data_sq[d_start:d_stop],
            )
            kk = min(k, d_stop - d_start)
            if kk < d_stop - d_start:
                kth = np.partition(approx, kk - 1, axis=1)[:, kk - 1]
            else:
                kth = approx.max(axis=1)
            keep = approx <= (kth + block_margins)[:, None]
            for row in range(q_stop - q_start):
                cols = np.flatnonzero(keep[row])
                pool_ids[row].append(cols.astype(np.int64) + d_start)
                pool_dist[row].append(approx[row, cols])

        for row in range(q_stop - q_start):
            cand_ids = np.concatenate(pool_ids[row])
            cand_approx = np.concatenate(pool_dist[row])
            if cand_ids.size > k:
                kth_global = np.partition(cand_approx, k - 1)[k - 1]
                cand_ids = cand_ids[cand_approx <= kth_global + block_margins[row]]
            exact = _exact_distances(metric_name, x, q[q_start + row], cand_ids)
            order = np.lexsort((cand_ids, exact))[:k]
            out_ids[q_start + row] = cand_ids[order]
            out_dist[q_start + row] = exact[order]

    return out_ids, out_dist


def exact_distances(
    data: ArrayLike, queries: ArrayLike, ids: ArrayLike, metric: str = "l2"
) -> NDArray[np.float64]:
    """Float64 distances between each query and the data rows named in ``ids``.

    ``ids`` has shape ``(nq, m)``; entries equal to ``-1`` (a missing result) map to ``+inf``.
    Used to score approximate results with the same distance definition as ``exact_search``.
    """
    metric_name = validate_metric(metric)
    x = as_matrix(data, "data")
    q = as_matrix(queries, "queries", dim=x.shape[1])
    id_array = np.asarray(ids, dtype=np.int64)
    if id_array.ndim != 2 or id_array.shape[0] != q.shape[0]:
        raise ValueError(f"ids must have shape (n_queries, m), got {id_array.shape}")
    if ((id_array < -1) | (id_array >= x.shape[0])).any():
        raise ValueError("ids contains values outside [-1, n)")
    out = np.full(id_array.shape, np.inf, dtype=np.float64)
    for row in range(q.shape[0]):
        valid = id_array[row] >= 0
        if valid.any():
            out[row, valid] = _exact_distances(metric_name, x, q[row], id_array[row, valid])
    return out


def _row_norms(x: NDArray[np.generic], block: int) -> NDArray[np.float64]:
    norms = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], block):
        rows = np.asarray(x[start : start + block], dtype=np.float64)
        norms[start : start + block] = np.linalg.norm(rows, axis=1)
    return norms


def _reject_zero_rows(norms: NDArray[np.float64], name: str) -> None:
    zero_rows = np.flatnonzero(norms == 0.0)
    if zero_rows.size:
        raise ValueError(
            f"cosine distance is undefined for zero vectors ({name} rows {zero_rows[:5].tolist()})"
        )


def _error_bound(
    metric: MetricName, dim: int, query_norms: NDArray[np.float64], max_data_norm: float
) -> NDArray[np.float64]:
    """Per-query upper bound on ``|float32 distance - float64 distance|``."""
    scale = _SAFETY * _UNIT_ROUNDOFF_F32
    if metric == "l2":
        return scale * (dim + 4) * (query_norms + max_data_norm) ** 2
    if metric == "ip":
        return scale * (dim + 4) * query_norms * max_data_norm
    return np.full(query_norms.shape, scale * (dim + 8), dtype=np.float64)


def _to_f32(
    rows: NDArray[np.generic], metric: MetricName, norms: NDArray[np.float64]
) -> NDArray[np.float32]:
    if metric == "cosine":
        rows64 = np.asarray(rows, dtype=np.float64) / norms[:, None]
        return np.ascontiguousarray(rows64, dtype=np.float32)
    return np.ascontiguousarray(rows, dtype=np.float32)


def _squared_norms_f32(
    x: NDArray[np.generic], block: int, metric: MetricName, norms: NDArray[np.float64]
) -> NDArray[np.float32]:
    out = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], block):
        rows = _to_f32(x[start : start + block], metric, norms[start : start + block])
        out[start : start + block] = np.einsum("ij,ij->i", rows, rows)
    return out


def _approx_distances(
    metric: MetricName,
    q32: NDArray[np.float32],
    x32: NDArray[np.float32],
    q_sq: NDArray[np.float32] | None,
    x_sq: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    dots = q32 @ x32.T
    if metric == "l2":
        assert q_sq is not None
        assert x_sq is not None
        return q_sq[:, None] + x_sq[None, :] - 2.0 * dots
    if metric == "ip":
        return -dots
    return 1.0 - dots


def _exact_distances(
    metric: MetricName, x: NDArray[np.generic], query: NDArray[np.generic], ids: NDArray[np.int64]
) -> NDArray[np.float64]:
    rows = np.asarray(x[ids], dtype=np.float64)
    q64 = np.asarray(query, dtype=np.float64)
    if metric == "l2":
        diff = rows - q64
        squared: NDArray[np.float64] = np.einsum("ij,ij->i", diff, diff)
        return squared
    if metric == "ip":
        return -(rows @ q64)
    rows = rows / np.linalg.norm(rows, axis=1)[:, None]
    return 1.0 - rows @ (q64 / np.linalg.norm(q64))
