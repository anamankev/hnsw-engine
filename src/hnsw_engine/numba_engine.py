"""Optimized HNSW engine: the reference algorithm compiled with Numba over flat arrays.

The algorithm, comparison rules and tie-breaking are exactly those of ``reference.py`` (see
``docs/DESIGN.md`` section 5); only the data layout and the execution model change:

* vectors in one contiguous ``float32`` array;
* layer 0 links in a fixed-width ``int32`` array ``(capacity, 2M + 1)`` with a count per node;
  links of the upper layers in a second ``(rows, M + 1)`` array, where a node of level ``L`` owns
  ``L`` consecutive rows starting at ``upper_offset[node]``. The extra column holds the one link
  that may overflow a node before it is shrunk back to its cap;
* a preallocated ``uint32`` visited array stamped with an epoch counter;
* compiled distance kernels (float64 accumulation, as in the reference) and compiled search,
  insertion and selection routines; heaps are Python-semantics ``heapq`` lists of
  ``(distance, id)`` tuples, so ties behave exactly as in the reference.

An equivalence test builds the same data with both engines and requires identical graphs and
search results.
"""

from __future__ import annotations

import heapq
from typing import Any, TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray

from .metrics import F32Array, MetricName
from .persistence import GraphState

_METRIC_CODES = {"l2": 0, "ip": 1, "cosine": 2}
_SELECTION_CODES = {"simple": 0, "heuristic": 1}
_INITIAL_CAPACITY = 1024
_NO_FILTER = np.ones(0, dtype=np.bool_)  # placeholder mask for unfiltered searches
_STORAGE = (  # every array that holds the graph, in the order of the native format
    "vectors",
    "levels",
    "links0",
    "count0",
    "upper_offset",
    "links_up",
    "count_up",
)
ScalarT = TypeVar("ScalarT", bound=np.generic)


# --------------------------------------------------------------------------- kernels


@njit(cache=True, inline="always")
def _as_id(value: int) -> int:
    """Widen a stored int32 link to int64, so every heap holds one tuple type."""
    return np.int64(value)  # type: ignore[return-value]  # Numba's int() keeps int32


@njit(cache=True, inline="always")
def _distance(metric: int, q: NDArray[np.float64], vectors: F32Array, j: int) -> float:
    """Distance from float64 query ``q`` to stored vector ``j`` with float64 accumulation."""
    acc = 0.0
    if metric == 0:
        for i in range(q.shape[0]):
            d = np.float64(vectors[j, i]) - q[i]
            acc += d * d
        return acc
    for i in range(q.shape[0]):
        acc += np.float64(vectors[j, i]) * q[i]
    if metric == 1:
        return -acc
    return 1.0 - acc


@njit(cache=True, inline="always")
def _links(
    node: int,
    layer: int,
    links0: NDArray[np.int32],
    count0: NDArray[np.int32],
    links_up: NDArray[np.int32],
    count_up: NDArray[np.int32],
    upper_offset: NDArray[np.int64],
) -> NDArray[np.int32]:
    if layer == 0:
        return links0[node, : count0[node]]
    row = upper_offset[node] + layer - 1
    return links_up[row, : count_up[row]]


@njit(cache=True)
def _next_epoch(state: NDArray[np.int64], visited: NDArray[np.uint32]) -> int:
    state[2] += 1
    if state[2] >= 4_294_967_295:  # wrap-around: reset every stamp
        visited[:] = 0
        state[2] = 1
    return int(state[2])


@njit(cache=True)
def _search_layer(
    q: NDArray[np.float64],
    entry: list[tuple[float, int]],
    ef: int,
    layer: int,
    metric: int,
    vectors: F32Array,
    links0: NDArray[np.int32],
    count0: NDArray[np.int32],
    links_up: NDArray[np.int32],
    count_up: NDArray[np.int32],
    upper_offset: NDArray[np.int64],
    visited: NDArray[np.uint32],
    state: NDArray[np.int64],
    counters: NDArray[np.int64],
) -> list[tuple[float, int]]:
    """Algorithm 2: best-first search keeping the ``ef`` closest nodes, sorted ascending."""
    epoch = _next_epoch(state, visited)
    candidates = [(d, e) for d, e in entry]
    heapq.heapify(candidates)
    results = [(-d, -e) for d, e in entry]
    heapq.heapify(results)
    while len(results) > ef:
        heapq.heappop(results)
    for _, e in entry:
        visited[e] = epoch

    while len(candidates) > 0:
        d_c, c = heapq.heappop(candidates)
        if d_c > -results[0][0]:
            break
        counters[1] += 1
        neighbours = _links(c, layer, links0, count0, links_up, count_up, upper_offset)
        worst = -results[0][0]
        for idx in range(neighbours.shape[0]):
            e = _as_id(neighbours[idx])
            if visited[e] == epoch:
                continue
            visited[e] = epoch
            d = _distance(metric, q, vectors, e)
            counters[0] += 1
            if len(results) < ef or d < worst:
                heapq.heappush(candidates, (d, e))
                heapq.heappush(results, (-d, -e))
                if len(results) > ef:
                    heapq.heappop(results)
                worst = -results[0][0]
    out = [(-nd, -ne) for nd, ne in results]
    out.sort()
    return out


@njit(cache=True)
def _search_layer0_filtered(
    q: NDArray[np.float64],
    entry: list[tuple[float, int]],
    ef: int,
    allowed: NDArray[np.bool_],
    metric: int,
    vectors: F32Array,
    links0: NDArray[np.int32],
    count0: NDArray[np.int32],
    visited: NDArray[np.uint32],
    state: NDArray[np.int64],
    counters: NDArray[np.int64],
) -> list[tuple[float, int]]:
    """Algorithm 2 on layer 0 returning only allowed nodes (as ``_search_layer_filtered``)."""
    epoch = _next_epoch(state, visited)
    candidates = [(d, e) for d, e in entry]
    heapq.heapify(candidates)
    results = [(-d, -e) for d, e in entry if allowed[e]]
    heapq.heapify(results)
    while len(results) > ef:
        heapq.heappop(results)
    for _, e in entry:
        visited[e] = epoch

    while len(candidates) > 0:
        d_c, c = heapq.heappop(candidates)
        if len(results) == ef and d_c > -results[0][0]:
            break
        counters[1] += 1
        worst = -results[0][0] if len(results) > 0 else np.inf
        for idx in range(count0[c]):
            e = _as_id(links0[c, idx])
            if visited[e] == epoch:
                continue
            visited[e] = epoch
            d = _distance(metric, q, vectors, e)
            counters[0] += 1
            if len(results) < ef or d < worst:
                heapq.heappush(candidates, (d, e))
                if allowed[e]:
                    heapq.heappush(results, (-d, -e))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]
    out = [(-nd, -ne) for nd, ne in results]
    out.sort()
    return out


@njit(cache=True)
def _select(
    base: int,
    candidates: list[tuple[float, int]],
    m: int,
    layer: int,
    metric: int,
    selection: int,
    extend: bool,
    keep_pruned: bool,
    vectors: F32Array,
    links0: NDArray[np.int32],
    count0: NDArray[np.int32],
    links_up: NDArray[np.int32],
    count_up: NDArray[np.int32],
    upper_offset: NDArray[np.int64],
    visited: NDArray[np.uint32],
    state: NDArray[np.int64],
    counters: NDArray[np.int64],
) -> list[tuple[float, int]]:
    """SELECT-NEIGHBORS: Algorithm 3 (simple) or Algorithm 4 (heuristic)."""
    if selection == 0:
        return candidates[:m]
    base_vec = vectors[base].astype(np.float64)
    work = candidates
    if extend:  # Algorithm 4, lines 3-7
        epoch = _next_epoch(state, visited)
        for _, e in candidates:
            visited[e] = epoch
        visited[base] = epoch
        work = [(d, e) for d, e in candidates]
        for _, e in candidates:
            adjacent = _links(e, layer, links0, count0, links_up, count_up, upper_offset)
            for idx in range(adjacent.shape[0]):
                a = _as_id(adjacent[idx])
                if visited[a] != epoch:
                    visited[a] = epoch
                    counters[0] += 1
                    work.append((_distance(metric, base_vec, vectors, a), a))
        work.sort()
    kept = [(0.0, 0) for _ in range(0)]
    discarded = [(0.0, 0) for _ in range(0)]
    for d_base, e in work:
        if len(kept) >= m:
            break
        e_vec = vectors[e].astype(np.float64)
        good = True
        for _, r in kept:
            counters[0] += 1
            if not d_base < _distance(metric, e_vec, vectors, r):
                good = False
                break
        if good:
            kept.append((d_base, e))
        else:
            discarded.append((d_base, e))
    if keep_pruned:
        for item in discarded:
            if len(kept) >= m:
                break
            kept.append(item)
    return kept


@njit(cache=True)
def _set_links(
    node: int,
    layer: int,
    chosen: list[tuple[float, int]],
    links0: NDArray[np.int32],
    count0: NDArray[np.int32],
    links_up: NDArray[np.int32],
    count_up: NDArray[np.int32],
    upper_offset: NDArray[np.int64],
) -> None:
    if layer == 0:
        for i in range(len(chosen)):
            links0[node, i] = chosen[i][1]
        count0[node] = len(chosen)
    else:
        row = upper_offset[node] + layer - 1
        for i in range(len(chosen)):
            links_up[row, i] = chosen[i][1]
        count_up[row] = len(chosen)


@njit(cache=True)
def _insert_batch(
    start: int,
    stop: int,
    levels: NDArray[np.int64],
    m: int,
    ef_construction: int,
    metric: int,
    selection: int,
    extend: bool,
    keep_pruned: bool,
    vectors: F32Array,
    links0: NDArray[np.int32],
    count0: NDArray[np.int32],
    links_up: NDArray[np.int32],
    count_up: NDArray[np.int32],
    upper_offset: NDArray[np.int64],
    visited: NDArray[np.uint32],
    state: NDArray[np.int64],
    counters: NDArray[np.int64],
) -> None:
    """Algorithm 1 for nodes ``start..stop-1`` (their vectors, levels and rows are in place)."""
    m0 = 2 * m
    for node in range(start, stop):
        level = levels[node]
        if state[0] < 0:  # first element: nothing to link to
            state[0] = node
            state[1] = level
            continue
        q = vectors[node].astype(np.float64)
        ep = state[0]
        entry = [(_distance(metric, q, vectors, ep), int(ep))]
        counters[0] += 1
        max_level = state[1]
        for layer in range(max_level, level, -1):
            entry = _search_layer(
                q,
                entry,
                1,
                layer,
                metric,
                vectors,
                links0,
                count0,
                links_up,
                count_up,
                upper_offset,
                visited,
                state,
                counters,
            )
        for layer in range(min(level, max_level), -1, -1):
            candidates = _search_layer(
                q,
                entry,
                ef_construction,
                layer,
                metric,
                vectors,
                links0,
                count0,
                links_up,
                count_up,
                upper_offset,
                visited,
                state,
                counters,
            )
            chosen = _select(
                node,
                candidates,
                m,
                layer,
                metric,
                selection,
                extend,
                keep_pruned,
                vectors,
                links0,
                count0,
                links_up,
                count_up,
                upper_offset,
                visited,
                state,
                counters,
            )
            _set_links(node, layer, chosen, links0, count0, links_up, count_up, upper_offset)
            cap = m0 if layer == 0 else m
            for _, e in chosen:
                # append the reverse link (the extra column absorbs a temporary overflow)
                if layer == 0:
                    links0[e, count0[e]] = node
                    count0[e] += 1
                    size = count0[e]
                else:
                    row = upper_offset[e] + layer - 1
                    links_up[row, count_up[row]] = node
                    count_up[row] += 1
                    size = count_up[row]
                if size > cap:  # shrink with the same selection strategy
                    e_vec = vectors[e].astype(np.float64)
                    current = _links(e, layer, links0, count0, links_up, count_up, upper_offset)
                    scored = [(0.0, 0) for _ in range(0)]
                    for idx in range(current.shape[0]):
                        other = _as_id(current[idx])
                        counters[0] += 1
                        scored.append((_distance(metric, e_vec, vectors, other), other))
                    scored.sort()
                    kept = _select(
                        e,
                        scored,
                        cap,
                        layer,
                        metric,
                        selection,
                        extend,
                        keep_pruned,
                        vectors,
                        links0,
                        count0,
                        links_up,
                        count_up,
                        upper_offset,
                        visited,
                        state,
                        counters,
                    )
                    _set_links(e, layer, kept, links0, count0, links_up, count_up, upper_offset)
            entry = candidates
        if level > max_level:
            state[0] = node
            state[1] = level


@njit(cache=True)
def _search_batch(
    queries: F32Array,
    k: int,
    ef: int,
    metric: int,
    vectors: F32Array,
    links0: NDArray[np.int32],
    count0: NDArray[np.int32],
    links_up: NDArray[np.int32],
    count_up: NDArray[np.int32],
    upper_offset: NDArray[np.int64],
    visited: NDArray[np.uint32],
    state: NDArray[np.int64],
    counters: NDArray[np.int64],
    allowed: NDArray[np.bool_],
    filtered: bool,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Algorithm 5 for every query row; results padded with ``-1`` / ``inf``.

    With ``filtered``, layer 0 returns only nodes whose ``allowed`` entry is true.
    """
    nq = queries.shape[0]
    ids = np.full((nq, k), -1, dtype=np.int64)
    dists = np.full((nq, k), np.inf, dtype=np.float64)
    for row in range(nq):
        q = queries[row].astype(np.float64)
        ep = state[0]
        entry = [(_distance(metric, q, vectors, ep), int(ep))]
        counters[0] += 1
        for layer in range(state[1], 0, -1):
            entry = _search_layer(
                q,
                entry,
                1,
                layer,
                metric,
                vectors,
                links0,
                count0,
                links_up,
                count_up,
                upper_offset,
                visited,
                state,
                counters,
            )
        if filtered:
            found = _search_layer0_filtered(
                q,
                entry,
                max(ef, k),
                allowed,
                metric,
                vectors,
                links0,
                count0,
                visited,
                state,
                counters,
            )
        else:
            found = _search_layer(
                q,
                entry,
                max(ef, k),
                0,
                metric,
                vectors,
                links0,
                count0,
                links_up,
                count_up,
                upper_offset,
                visited,
                state,
                counters,
            )
        for i in range(min(k, len(found))):
            dists[row, i] = found[i][0]
            ids[row, i] = found[i][1]
    return ids, dists


@njit(cache=True)
def _rerank_batch(
    queries: F32Array,
    vectors: F32Array,
    candidates: NDArray[np.int64],
    k: int,
    metric: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Exact distances to each row's candidate ids (``-1`` skipped); best ``k`` by distance, id."""
    nq = queries.shape[0]
    ids = np.full((nq, k), -1, dtype=np.int64)
    dists = np.full((nq, k), np.inf, dtype=np.float64)
    for row in range(nq):
        q = queries[row].astype(np.float64)
        scored = [(0.0, np.int64(0))]
        scored.pop()
        for j in range(candidates.shape[1]):
            e = candidates[row, j]
            if e >= 0:
                scored.append((_distance(metric, q, vectors, e), e))
        scored.sort()
        for i in range(min(k, len(scored))):
            dists[row, i] = scored[i][0]
            ids[row, i] = scored[i][1]
    return ids, dists


# --------------------------------------------------------------------------- engine


class NumbaEngine:
    """Compiled HNSW core with the same interface and results as ``ReferenceEngine``."""

    name = "numba"

    def __init__(
        self,
        dim: int,
        metric: MetricName,
        M: int,
        ef_construction: int,
        selection: str,
        extend_candidates: bool,
        keep_pruned_connections: bool,
    ) -> None:
        self.dim = dim
        self.metric: MetricName = metric
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.selection = selection
        self.extend_candidates = extend_candidates
        self.keep_pruned_connections = keep_pruned_connections
        self._metric_code = _METRIC_CODES[metric]
        self._selection_code = _SELECTION_CODES[selection]
        self._size = 0
        self._upper_rows = 0
        self._vectors: F32Array = np.empty((_INITIAL_CAPACITY, dim), dtype=np.float32)
        self._levels = np.zeros(_INITIAL_CAPACITY, dtype=np.int64)
        self._links0 = np.full((_INITIAL_CAPACITY, self.M0 + 1), -1, dtype=np.int32)
        self._count0 = np.zeros(_INITIAL_CAPACITY, dtype=np.int32)
        self._upper_offset = np.full(_INITIAL_CAPACITY, -1, dtype=np.int64)
        self._links_up = np.full((max(_INITIAL_CAPACITY // 8, 1), M + 1), -1, dtype=np.int32)
        self._count_up = np.zeros(self._links_up.shape[0], dtype=np.int32)
        self._visited = np.zeros(_INITIAL_CAPACITY, dtype=np.uint32)
        self._state = np.array([-1, -1, 0], dtype=np.int64)  # entry point, max level, epoch
        self._counters = np.zeros(2, dtype=np.int64)  # distance computations, hops

    # -- engine interface ------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of nodes in the graph."""
        return self._size

    @property
    def entry_point(self) -> int:
        """Id of the entry point (-1 when empty)."""
        return int(self._state[0])

    @property
    def max_level(self) -> int:
        """Top layer (-1 when empty)."""
        return int(self._state[1])

    @property
    def distance_computations(self) -> int:
        """Cumulative distance evaluations."""
        return int(self._counters[0])

    @distance_computations.setter
    def distance_computations(self, value: int) -> None:
        self._counters[0] = value

    @property
    def hops(self) -> int:
        """Cumulative node expansions."""
        return int(self._counters[1])

    @hops.setter
    def hops(self, value: int) -> None:
        self._counters[1] = value

    def insert(self, vectors: F32Array, levels: NDArray[np.int64]) -> None:
        """Insert prepared vectors with pre-drawn levels; ids continue from the current size."""
        self._ensure_writable()
        count = vectors.shape[0]
        start, stop = self._size, self._size + count
        self._reserve_nodes(stop)
        self._vectors[start:stop] = vectors
        self._levels[start:stop] = levels
        new_rows = int(levels.sum())
        self._reserve_upper_rows(self._upper_rows + new_rows)
        for offset, level in enumerate(levels.tolist()):
            if level > 0:
                self._upper_offset[start + offset] = self._upper_rows
                self._upper_rows += level
        _insert_batch(
            start,
            stop,
            self._levels,
            self.M,
            self.ef_construction,
            self._metric_code,
            self._selection_code,
            self.extend_candidates,
            self.keep_pruned_connections,
            self._vectors,
            self._links0,
            self._count0,
            self._links_up,
            self._count_up,
            self._upper_offset,
            self._visited,
            self._state,
            self._counters,
        )
        self._size = stop

    def search(
        self, query: F32Array, k: int, ef: int, allowed: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Algorithm 5 for one prepared query; results padded to length ``k``."""
        ids, dists = self.batch_search(query[None, :], k, ef, allowed)
        return ids[0], dists[0]

    def batch_search(
        self, queries: F32Array, k: int, ef: int, allowed: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Search every row of ``queries`` inside one compiled loop (optionally filtered)."""
        return _search_batch(
            np.ascontiguousarray(queries, dtype=np.float32),
            k,
            ef,
            self._metric_code,
            self._vectors,
            self._links0,
            self._count0,
            self._links_up,
            self._count_up,
            self._upper_offset,
            self._visited,
            self._state,
            self._counters,
            _NO_FILTER if allowed is None else np.ascontiguousarray(allowed, dtype=np.bool_),
            allowed is not None,
        )

    def batch_search_codes(
        self, codes: NDArray[np.uint8], queries: NDArray[np.float64], k: int, ef: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Algorithm 5 over this graph with distances to ``codes`` instead of the stored vectors.

        ``codes`` has one row per node; ``queries`` are already in code space
        (``ScalarQuantizer.transform_queries``). Numba compiles a ``uint8`` version of the same
        search code, so the traversal rules are exactly those of :meth:`batch_search`.
        """
        if codes.shape != (self._size, self.dim) or codes.dtype != np.uint8:
            raise ValueError(f"codes must be uint8 with shape {(self._size, self.dim)}")
        return _search_batch(
            np.ascontiguousarray(queries, dtype=np.float64),
            k,
            ef,
            self._metric_code,
            np.ascontiguousarray(codes),
            self._links0,
            self._count0,
            self._links_up,
            self._count_up,
            self._upper_offset,
            self._visited,
            self._state,
            self._counters,
            _NO_FILTER,
            False,
        )

    def rerank(
        self, queries: F32Array, candidates: NDArray[np.int64], k: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Re-score each query's candidate ids with exact distances; keep the best ``k``."""
        return _rerank_batch(
            np.ascontiguousarray(queries, dtype=np.float32),
            self._vectors,
            np.ascontiguousarray(candidates, dtype=np.int64),
            k,
            self._metric_code,
        )

    # -- storage ------------------------------------------------------------------------

    def _reserve_nodes(self, capacity: int) -> None:
        current = self._vectors.shape[0]
        if capacity <= current:
            return
        new = max(capacity, 2 * current)
        self._vectors = _grow(self._vectors, new, 0.0)
        self._levels = _grow(self._levels, new, 0)
        self._links0 = _grow(self._links0, new, -1)
        self._count0 = _grow(self._count0, new, 0)
        self._upper_offset = _grow(self._upper_offset, new, -1)
        self._visited = _grow(self._visited, new, 0)

    def _reserve_upper_rows(self, rows: int) -> None:
        current = self._links_up.shape[0]
        if rows <= current:
            return
        new = max(rows, 2 * current)
        self._links_up = _grow(self._links_up, new, -1)
        self._count_up = _grow(self._count_up, new, 0)

    def memory_bytes(self) -> int:
        """Bytes of every array the engine holds (including unused capacity)."""
        arrays = (
            self._vectors,
            self._levels,
            self._links0,
            self._count0,
            self._upper_offset,
            self._links_up,
            self._count_up,
            self._visited,
        )
        return int(sum(a.nbytes for a in arrays))

    # -- snapshots ------------------------------------------------------------------------

    def export_state(self) -> GraphState:
        """Snapshot of the graph in the engine-independent format."""
        n = self._size
        levels = self._levels[:n].copy()
        top = int(levels.max()) if n else -1
        indptr: list[NDArray[np.int64]] = []
        indices: list[NDArray[np.int64]] = []
        for layer in range(top + 1):
            nodes = np.flatnonzero(levels >= layer)
            if layer == 0:
                counts = self._count0[nodes].astype(np.int64)
                rows = [self._links0[v, : self._count0[v]] for v in nodes]
            else:
                row_ids = self._upper_offset[nodes] + layer - 1
                counts = self._count_up[row_ids].astype(np.int64)
                rows = [self._links_up[r, : self._count_up[r]] for r in row_ids]
            ptr = np.zeros(nodes.size + 1, dtype=np.int64)
            np.cumsum(counts, out=ptr[1:])
            flat = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64)
            indptr.append(ptr)
            indices.append(flat)
        return GraphState(
            self._vectors[:n].copy(),
            levels,
            self.entry_point,
            self.max_level,
            tuple(indptr),
            tuple(indices),
        )

    def load_state(self, state: GraphState) -> None:
        """Replace the engine contents with ``state``."""
        self._ensure_writable()
        n = state.size
        self._size = 0
        self._upper_rows = 0
        self._reserve_nodes(max(n, 1))
        self._vectors[:n] = state.vectors
        self._levels[:n] = state.levels
        self._count0[:] = 0
        self._upper_offset[:] = -1
        self._reserve_upper_rows(int(state.levels.sum()) if n else 0)
        for node, level in enumerate(state.levels.tolist()):
            if level > 0:
                self._upper_offset[node] = self._upper_rows
                self._upper_rows += level
        for layer in range(state.num_layers):
            for node, links in state.adjacency(layer).items():
                arr = np.asarray(links, dtype=np.int32)
                if layer == 0:
                    self._links0[node, : arr.size] = arr
                    self._count0[node] = arr.size
                else:
                    row = self._upper_offset[node] + layer - 1
                    self._links_up[row, : arr.size] = arr
                    self._count_up[row] = arr.size
        self._state[0], self._state[1] = state.entry_point, state.max_level
        self._size = n

    # -- native arrays (memory-mappable persistence) ---------------------------------------

    def native_arrays(self) -> dict[str, NDArray[Any]]:
        """The arrays that hold the graph, trimmed to the rows in use (views, not copies)."""
        n, rows = self._size, self._upper_rows
        return {
            "vectors": self._vectors[:n],
            "levels": self._levels[:n],
            "links0": self._links0[:n],
            "count0": self._count0[:n],
            "upper_offset": self._upper_offset[:n],
            "links_up": self._links_up[:rows],
            "count_up": self._count_up[:rows],
        }

    def attach_native(
        self,
        arrays: dict[str, NDArray[Any]],
        entry_point: int,
        max_level: int,
        *,
        validate: bool = True,
    ) -> None:
        """Use ``arrays`` (from :meth:`native_arrays`, possibly read-only memory maps) in place.

        Searches read them directly; the first insertion copies them into memory, so the
        arrays themselves are never written. ``validate`` checks every link before use (see
        :func:`_validate_native`); skip it only for files you produced yourself.
        """
        n = arrays["vectors"].shape[0]
        rows = arrays["links_up"].shape[0]
        expected = {
            "vectors": (np.float32, (n, self.dim)),
            "levels": (np.int64, (n,)),
            "links0": (np.int32, (n, self.M0 + 1)),
            "count0": (np.int32, (n,)),
            "upper_offset": (np.int64, (n,)),
            "links_up": (np.int32, (rows, self.M + 1)),
            "count_up": (np.int32, (rows,)),
        }
        for name, (dtype, shape) in expected.items():
            array = arrays[name]
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(
                    f"{name} has dtype {array.dtype} and shape {array.shape}, expected "
                    f"{np.dtype(dtype)} and {shape}"
                )
        if int(arrays["levels"].sum()) != rows:
            raise ValueError("upper-layer rows do not match the node levels")
        if validate:
            _validate_native(arrays, entry_point, max_level, self.M0, self.M)
        for name in _STORAGE:
            setattr(self, f"_{name}", arrays[name])
        self._visited = np.zeros(n, dtype=np.uint32)
        self._size = n
        self._upper_rows = rows
        self._state[:] = (entry_point, max_level, 0)

    def _ensure_writable(self) -> None:
        """Copy read-only (memory-mapped) storage into memory before the graph changes."""
        for name in _STORAGE:
            array = getattr(self, f"_{name}")
            if not array.flags.writeable:
                setattr(self, f"_{name}", np.array(array))


def _validate_native(
    arrays: dict[str, NDArray[Any]], entry_point: int, max_level: int, m0: int, m: int
) -> None:
    """Raise ``ValueError`` unless the native arrays describe a well-formed graph.

    The compiled kernels do not bounds-check, so every link a search can follow must be a valid
    node id, every node's upper-layer rows must lie inside ``links_up`` without overlapping, and
    upper-layer links must point to nodes present on that layer. Vectorised: one pass over the
    link arrays.
    """
    levels = arrays["levels"]
    n, rows = levels.shape[0], arrays["links_up"].shape[0]
    if n == 0:
        if rows or entry_point != -1 or max_level != -1:
            raise ValueError("an empty graph has no entry point and no layers")
        return
    if (levels < 0).any():
        raise ValueError("negative node level")
    top = int(levels.max())
    if max_level != top or not 0 <= entry_point < n or int(levels[entry_point]) != top:
        raise ValueError("the entry point is not a node of the top layer")
    _check_links(arrays["links0"], arrays["count0"], n, m0, "layer 0")
    upper = np.flatnonzero(levels > 0)
    offsets = arrays["upper_offset"][upper]
    order = np.argsort(offsets, kind="stable")
    starts, sizes = offsets[order], levels[upper][order]
    ends = starts + sizes
    tiles = starts.size == 0 or (
        starts[0] == 0 and ends[-1] == rows and bool((starts[1:] == ends[:-1]).all())
    )
    if not tiles or (starts.size == 0 and rows):
        raise ValueError("upper-layer rows of the nodes overlap or leave gaps")
    count_up = arrays["count_up"]
    targets = _check_links(arrays["links_up"], count_up, n, m, "upper layers")
    layer_of_row = np.arange(rows) - np.repeat(starts, sizes) + 1
    if (levels[targets] < np.repeat(layer_of_row, count_up)).any():
        raise ValueError("an upper-layer link points to a node absent from that layer")


def _check_links(
    links: NDArray[np.int32], counts: NDArray[np.int32], n: int, cap: int, where: str
) -> NDArray[np.int32]:
    """Used link slots (row by row) after checking counts and ids; raises ``ValueError``."""
    if (counts < 0).any() or (counts > cap).any():
        raise ValueError(f"{where}: link counts outside [0, {cap}]")
    used = np.arange(links.shape[1])[None, :] < counts[:, None]
    values = links[used]
    if values.size and ((values < 0).any() or (values >= n).any()):
        raise ValueError(f"{where}: a link points outside the graph")
    return values


def _grow(array: NDArray[ScalarT], rows: int, fill: float) -> NDArray[ScalarT]:
    """Copy ``array`` into a larger array (first axis ``rows``), filling the new part."""
    out = np.full((rows, *array.shape[1:]), fill, dtype=array.dtype)
    out[: array.shape[0]] = array
    return out
