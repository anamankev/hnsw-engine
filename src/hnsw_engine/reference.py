"""Reference HNSW engine: the paper's algorithms in readable Python and NumPy.

Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical
Navigable Small World graphs" (arXiv:1603.09320). Comments cite the paper's algorithm numbers.
The exact tie-breaking and comparison rules are fixed in ``docs/DESIGN.md`` section 5; the
optimized engine must follow them too, which is what makes the two comparable bit for bit.

This engine favours clarity over speed. Its only concession to performance is that the distances
from the query to all unvisited neighbours of an expanded node are computed in one NumPy call.

Data layout:

* ``_vectors``: ``float32`` array, grown by doubling; row ``i`` is node ``i``.
* ``_layers[l]``: ``dict`` mapping each node on layer ``l`` to its list of neighbour ids.
* ``_visited``: one ``uint32`` stamp per node. A search marks nodes with a fresh epoch number,
  so "clearing" the visited set between searches costs nothing (the trick hnswlib uses).
"""

from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

from .metrics import F32Array, F64Array, MetricName, distances_to_many
from .persistence import GraphState
from .selection import Candidate, select_heuristic, select_simple

_INITIAL_CAPACITY = 1024


class ReferenceEngine:
    """Pure Python HNSW core. Receives prepared ``float32`` vectors and pre-drawn levels."""

    name = "reference"

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

        self._vectors: F32Array = np.empty((_INITIAL_CAPACITY, dim), dtype=np.float32)
        self._size = 0
        self._levels: list[int] = []
        self._layers: list[dict[int, list[int]]] = []
        self.entry_point = -1
        self.max_level = -1
        self._visited = np.zeros(_INITIAL_CAPACITY, dtype=np.uint32)
        self._epoch = 0
        self.distance_computations = 0
        self.hops = 0

    @property
    def size(self) -> int:
        """Number of nodes in the graph."""
        return self._size

    # ------------------------------------------------------------------ insertion

    def insert(self, vectors: F32Array, levels: NDArray[np.int64]) -> None:
        """Insert prepared vectors one by one; node ids continue from the current size."""
        count = vectors.shape[0]
        self._reserve(self._size + count)
        self._vectors[self._size : self._size + count] = vectors
        for i in range(count):
            self._insert_one(self._size, int(levels[i]))
            self._size += 1

    def _insert_one(self, node: int, level: int) -> None:
        """Algorithm 1 (INSERT)."""
        query = self._vectors[node].astype(np.float64)
        self._levels.append(level)
        while len(self._layers) <= level:
            self._layers.append({})
        for layer in range(level + 1):
            self._layers[layer][node] = []

        if self.entry_point < 0:  # first element: nothing to link to
            self.entry_point, self.max_level = node, level
            return

        entry = [(float(self._distances(query, [self.entry_point])[0]), self.entry_point)]
        # Phase 1: greedy descent (ef = 1) through the layers above the new node's level.
        for layer in range(self.max_level, level, -1):
            entry = self._search_layer(query, entry, 1, layer)

        # Phase 2: on every layer the node joins, link it to M selected candidates.
        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(query, entry, self.ef_construction, layer)
            chosen = self._select(node, query, candidates, self.M, layer)
            neighbours = [e for _, e in chosen]
            self._layers[layer][node] = neighbours
            cap = self.M0 if layer == 0 else self.M
            for e in neighbours:
                links = self._layers[layer][e]
                links.append(node)
                if len(links) > cap:  # shrink with the same selection strategy
                    self._shrink(e, layer, cap)
            entry = candidates  # the whole candidate list seeds the next layer

        if level > self.max_level:
            self.entry_point, self.max_level = node, level

    def _shrink(self, node: int, layer: int, cap: int) -> None:
        """Re-select the links of ``node`` after it exceeded its degree cap."""
        links = self._layers[layer][node]
        base = self._vectors[node].astype(np.float64)
        dists = self._distances(base, links)
        candidates = sorted(zip(dists.tolist(), links, strict=True))
        chosen = self._select(node, base, candidates, cap, layer)
        self._layers[layer][node] = [e for _, e in chosen]

    def _select(
        self, base: int, base_vector: F64Array, candidates: list[Candidate], M: int, layer: int
    ) -> list[Candidate]:
        """SELECT-NEIGHBORS: Algorithm 3 (simple) or Algorithm 4 (heuristic)."""
        if self.selection == "simple":
            return select_simple(candidates, M)
        if self.extend_candidates:
            candidates = self._extend(base, base_vector, candidates, layer)
        return select_heuristic(
            candidates,
            M,
            self._pair_distance,
            keep_pruned_connections=self.keep_pruned_connections,
        )

    def _extend(
        self, base: int, base_vector: F64Array, candidates: list[Candidate], layer: int
    ) -> list[Candidate]:
        """Algorithm 4, lines 3-7: add the candidates' neighbours to the working set."""
        seen = {e for _, e in candidates}
        seen.add(base)
        extra: list[int] = []
        for _, e in candidates:
            for adjacent in self._layers[layer][e]:
                if adjacent not in seen:
                    seen.add(adjacent)
                    extra.append(adjacent)
        if not extra:
            return candidates
        dists = self._distances(base_vector, extra)
        return sorted(candidates + list(zip(dists.tolist(), extra, strict=True)))

    def _pair_distance(self, node: int, others: list[int]) -> F64Array:
        return self._distances(self._vectors[node].astype(np.float64), others)

    # --------------------------------------------------------------------- search

    def search(
        self, query: F32Array, k: int, ef: int, allowed: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[np.int64], F64Array]:
        """Algorithm 5 (K-NN-SEARCH) for one prepared query; results padded to length ``k``.

        With ``allowed`` (one boolean per node), only allowed nodes are returned: the descent
        through the upper layers is unchanged and layer 0 uses :meth:`_search_layer_filtered`.
        """
        q = query.astype(np.float64)
        entry = [(float(self._distances(q, [self.entry_point])[0]), self.entry_point)]
        for layer in range(self.max_level, 0, -1):
            entry = self._search_layer(q, entry, 1, layer)
        if allowed is None:
            found = self._search_layer(q, entry, max(ef, k), 0)[:k]
        else:
            found = self._search_layer_filtered(q, entry, max(ef, k), allowed)[:k]
        ids = np.full(k, -1, dtype=np.int64)
        dists = np.full(k, np.inf, dtype=np.float64)
        ids[: len(found)] = [e for _, e in found]
        dists[: len(found)] = [d for d, _ in found]
        return ids, dists

    def batch_search(
        self, queries: F32Array, k: int, ef: int, allowed: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Run :meth:`search` for each row of ``queries``."""
        ids = np.empty((queries.shape[0], k), dtype=np.int64)
        dists = np.empty((queries.shape[0], k), dtype=np.float64)
        for row in range(queries.shape[0]):
            ids[row], dists[row] = self.search(queries[row], k, ef, allowed)
        return ids, dists

    def _search_layer(
        self, query: F64Array, entry: list[Candidate], ef: int, layer: int
    ) -> list[Candidate]:
        """Algorithm 2 (SEARCH-LAYER): best-first search keeping the ``ef`` closest nodes.

        ``candidates`` is a min-heap of nodes still to expand, keyed by ``(distance, id)``.
        ``results`` is a max-heap (keys negated) of the best ``ef`` nodes found so far. The
        search ends when the closest unexpanded candidate is farther than the worst result:
        expanding it could not improve the result set.
        """
        epoch = self._next_epoch()
        visited = self._visited
        links_of = self._layers[layer]

        candidates = list(entry)
        heapq.heapify(candidates)
        results = [(-d, -node) for d, node in entry]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)
        for _, node in entry:
            visited[node] = epoch

        while candidates:
            dist, node = heapq.heappop(candidates)
            if dist > -results[0][0]:
                break
            self.hops += 1
            links = links_of[node]
            if not links:
                continue
            neighbours = np.asarray(links, dtype=np.int64)
            neighbours = neighbours[visited[neighbours] != epoch]
            if neighbours.size == 0:
                continue
            visited[neighbours] = epoch
            dists = self._distances(query, neighbours)
            worst = -results[0][0]
            for d, e in zip(dists.tolist(), neighbours.tolist(), strict=True):
                if len(results) < ef or d < worst:
                    heapq.heappush(candidates, (d, e))
                    heapq.heappush(results, (-d, -e))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]
        return sorted((-d, -node) for d, node in results)

    def _search_layer_filtered(
        self, query: F64Array, entry: list[Candidate], ef: int, allowed: NDArray[np.bool_]
    ) -> list[Candidate]:
        """Algorithm 2 on layer 0 returning only allowed nodes (design rule 9, as in hnswlib).

        Disallowed nodes are still expanded, so the search can travel through them, but only
        allowed nodes enter ``results``. Because ``results`` may hold fewer than ``ef`` nodes
        (or none: the worst distance is then infinite), the search stops only once it is full
        and the closest candidate is farther than its worst element.
        """
        epoch = self._next_epoch()
        visited = self._visited
        links_of = self._layers[0]

        candidates = list(entry)
        heapq.heapify(candidates)
        results = [(-d, -node) for d, node in entry if allowed[node]]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)
        for _, node in entry:
            visited[node] = epoch

        while candidates:
            dist, node = heapq.heappop(candidates)
            if len(results) == ef and dist > -results[0][0]:
                break
            self.hops += 1
            links = links_of[node]
            if not links:
                continue
            neighbours = np.asarray(links, dtype=np.int64)
            neighbours = neighbours[visited[neighbours] != epoch]
            if neighbours.size == 0:
                continue
            visited[neighbours] = epoch
            dists = self._distances(query, neighbours)
            worst = -results[0][0] if results else np.inf
            for d, e in zip(dists.tolist(), neighbours.tolist(), strict=True):
                if len(results) < ef or d < worst:
                    heapq.heappush(candidates, (d, e))
                    if allowed[e]:
                        heapq.heappush(results, (-d, -e))
                        if len(results) > ef:
                            heapq.heappop(results)
                        worst = -results[0][0]
        return sorted((-d, -node) for d, node in results)

    # -------------------------------------------------------------------- helpers

    def _distances(self, query: F64Array, ids: list[int] | NDArray[np.int64]) -> F64Array:
        self.distance_computations += len(ids)
        return distances_to_many(self.metric, query, self._vectors[ids])

    def _next_epoch(self) -> int:
        self._epoch += 1
        if self._epoch == np.iinfo(np.uint32).max:  # wrap-around: reset all stamps
            self._visited[:] = 0
            self._epoch = 1
        return self._epoch

    def _reserve(self, capacity: int) -> None:
        current = self._vectors.shape[0]
        if capacity <= current:
            return
        new_capacity = max(capacity, 2 * current)
        vectors = np.empty((new_capacity, self.dim), dtype=np.float32)
        vectors[: self._size] = self._vectors[: self._size]
        self._vectors = vectors
        visited = np.zeros(new_capacity, dtype=np.uint32)
        visited[:current] = self._visited
        self._visited = visited

    # ---------------------------------------------------------------- snapshots

    def export_state(self) -> GraphState:
        """Snapshot of the graph (copies, safe to keep after further inserts)."""
        return GraphState.from_adjacency(
            self._vectors[: self._size].copy(),
            np.asarray(self._levels, dtype=np.int64),
            self.entry_point,
            self.max_level,
            self._layers,
        )

    def load_state(self, state: GraphState) -> None:
        """Replace the engine contents with ``state``."""
        self._size = 0
        self._reserve(max(state.size, 1))
        self._vectors[: state.size] = state.vectors
        self._size = state.size
        self._levels = state.levels.tolist()
        self._layers = [
            {v: list(e) for v, e in state.adjacency(layer).items()}
            for layer in range(state.num_layers)
        ]
        self.entry_point, self.max_level = state.entry_point, state.max_level
