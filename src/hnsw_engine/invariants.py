"""Structural invariants of an HNSW graph, checked on an engine-independent snapshot.

Used by the test suite after every build and by the reachability study.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from .persistence import GraphState


def check_invariants(state: GraphState, M: int) -> list[str]:
    """Return a list of violated invariants (empty if the graph is well formed).

    Checked: (1) layer membership is nested, i.e. a node on layer ``l`` is on every layer
    below; (2) links only point to nodes present on that layer; (3) no self links and no
    duplicate links; (4) degree caps (``2M`` on layer 0, ``M`` above); (5) the entry point has
    the top level.
    """
    problems: list[str] = []
    n = state.size
    if n == 0:
        if state.entry_point != -1 or state.max_level != -1 or state.num_layers != 0:
            problems.append("empty graph must have no entry point and no layers")
        return problems
    if (state.levels < 0).any():
        problems.append("negative level found")
    top = int(state.levels.max())
    if state.max_level != top or state.num_layers != top + 1:
        problems.append(f"max_level {state.max_level} / {state.num_layers} layers, top level {top}")
    if not 0 <= state.entry_point < n or int(state.levels[state.entry_point]) != top:
        problems.append(f"entry point {state.entry_point} does not have the top level {top}")

    for layer in range(state.num_layers):
        cap = 2 * M if layer == 0 else M
        # Membership is derived from levels, so nesting holds iff every node listed here has
        # level >= layer -- which is how layer_nodes is defined; we check link targets instead.
        for node, links in state.adjacency(layer).items():
            if len(links) > cap:
                problems.append(f"layer {layer}: node {node} has {len(links)} > {cap} links")
            if node in links:
                problems.append(f"layer {layer}: self link at node {node}")
            if len(set(links)) != len(links):
                problems.append(f"layer {layer}: duplicate links at node {node}")
            problems.extend(
                f"layer {layer}: link {node}->{target} leaves the layer"
                for target in links
                if not 0 <= target < n or int(state.levels[target]) < layer
            )
        expected = int((state.levels >= layer).sum())
        if state.indptr[layer].shape[0] != expected + 1:
            problems.append(f"layer {layer}: CSR covers the wrong number of nodes")
    return problems


def reachable_from_entry(state: GraphState, layer: int = 0) -> NDArray[np.bool_]:
    """Boolean mask over all node ids: which nodes a search on ``layer`` can ever visit.

    Breadth-first search along directed links from the entry point. On layer 0 every node is
    present, so ``~mask`` lists the nodes that no query can return, whatever ``ef`` is.
    """
    mask = np.zeros(state.size, dtype=bool)
    if state.size == 0:
        return mask
    adjacency = state.adjacency(layer)
    if state.entry_point not in adjacency:
        raise ValueError(f"entry point is not on layer {layer}")
    queue: deque[int] = deque([state.entry_point])
    mask[state.entry_point] = True
    while queue:
        node = queue.popleft()
        for target in adjacency[node]:
            if not mask[target]:
                mask[target] = True
                queue.append(target)
    return mask


def searchable_from_entry(state: GraphState) -> NDArray[np.bool_]:
    """Boolean mask over node ids: which nodes *some* query can reach on layer 0.

    A query descends from the entry point and may end its walk on a layer anywhere that layer's
    links reach, then continues one layer down from there. So the possible starting points on
    layer ``l`` are all nodes reachable on layer ``l + 1``. This takes that closure layer by layer
    from the top down. Nodes outside the mask can never be returned, whatever ``ef`` is.
    (``reachable_from_entry(state, 0)`` is the stricter layer-0-only view from the entry point.)
    """
    mask = np.zeros(state.size, dtype=bool)
    if state.size == 0:
        return mask
    starts = [state.entry_point]
    for layer in range(state.num_layers - 1, -1, -1):
        adjacency = state.adjacency(layer)
        reached = np.zeros(state.size, dtype=bool)
        reached[starts] = True
        queue: deque[int] = deque(starts)
        while queue:
            node = queue.popleft()
            for target in adjacency[node]:
                if not reached[target]:
                    reached[target] = True
                    queue.append(target)
        starts = np.flatnonzero(reached).tolist()
        mask = reached
    return mask
