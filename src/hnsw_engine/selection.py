"""Neighbour selection: which candidates become a node's links (paper Algorithms 3 and 4).

Both functions are pure: they receive candidates as ``(distance_to_base, id)`` pairs sorted in
ascending order (ties by id) and, for the heuristic, a callable that measures distances between
graph nodes. The optional ``extend_candidates`` step of Algorithm 4 needs the graph, so the
engines perform it before calling :func:`select_heuristic`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

Candidate = tuple[float, int]
PairDistance = Callable[[int, list[int]], NDArray[np.float64]]


def select_simple(candidates: Sequence[Candidate], M: int) -> list[Candidate]:
    """Algorithm 3: the ``M`` candidates closest to the base element."""
    return list(candidates[:M])


def select_heuristic(
    candidates: Sequence[Candidate],
    M: int,
    pair_distance: PairDistance,
    *,
    keep_pruned_connections: bool = False,
) -> list[Candidate]:
    """Algorithm 4: pick up to ``M`` candidates that point in *different directions*.

    Candidates are examined closest-first. A candidate ``e`` is kept only if it is strictly
    closer to the base element than to every candidate already kept; otherwise some kept
    neighbour ``r`` is at least as close to ``e``, so ``e`` is reachable through ``r`` and its
    slot is better spent elsewhere. On clustered data this keeps long links between clusters
    that "M closest" would drop, which preserves graph connectivity.

    Args:
        candidates: ``(distance_to_base, id)`` pairs, ascending, unique ids, base excluded.
        M: maximum number of neighbours to return.
        pair_distance: ``pair_distance(e, ids)`` returns distances from node ``e`` to ``ids``.
        keep_pruned_connections: fill unused slots with discarded candidates, closest first.

    Returns:
        Kept candidates in examination order, followed by any filled-in discarded ones.
    """
    kept: list[Candidate] = []
    kept_ids: list[int] = []
    discarded: list[Candidate] = []
    for candidate in candidates:
        if len(kept) >= M:
            break
        dist_to_base, node = candidate
        if kept_ids and not bool(np.all(dist_to_base < pair_distance(node, kept_ids))):
            discarded.append(candidate)
            continue
        kept.append(candidate)
        kept_ids.append(node)
    if keep_pruned_connections:
        kept.extend(discarded[: max(0, M - len(kept))])
    return kept
