"""Neighbour selection on hand-made 2-D points (squared L2 distances, base at the origin)."""

import numpy as np

from hnsw_engine.selection import select_heuristic, select_simple

POINTS = {
    0: (1.0, 0.0),  # A: closest to the base
    1: (1.1, 0.1),  # B: right next to A, same direction
    2: (0.0, 1.2),  # C: a different direction
    3: (-1.5, 0.0),  # D: the opposite direction, farthest
}


def _pair_distance(points: dict[int, tuple[float, float]]):
    def distance(node: int, others: list[int]) -> np.ndarray:
        p = np.array(points[node])
        return np.array([float(np.sum((p - np.array(points[o])) ** 2)) for o in others])

    return distance


def _candidates(points: dict[int, tuple[float, float]], base=(0.0, 0.0)):
    b = np.array(base)
    pairs = [(float(np.sum((np.array(p) - b) ** 2)), i) for i, p in points.items()]
    return sorted(pairs)


def test_candidate_order_used_by_tests() -> None:
    # d(A)=1.0 < d(B)=1.22 < d(C)=1.44 < d(D)=2.25
    assert [i for _, i in _candidates(POINTS)] == [0, 1, 2, 3]


def test_simple_selection_takes_closest() -> None:
    cands = _candidates(POINTS)
    assert [i for _, i in select_simple(cands, 2)] == [0, 1]
    assert select_simple(cands, 10) == cands


def test_heuristic_prunes_redundant_direction() -> None:
    # B is 0.02 from A but 1.22 from the base: reachable through A, so it is pruned.
    cands = _candidates(POINTS)
    chosen = select_heuristic(cands, 3, _pair_distance(POINTS))
    assert [i for _, i in chosen] == [0, 2, 3]


def test_heuristic_stops_at_m() -> None:
    chosen = select_heuristic(_candidates(POINTS), 2, _pair_distance(POINTS))
    assert [i for _, i in chosen] == [0, 2]


def test_heuristic_can_return_fewer_than_m() -> None:
    # Without keep_pruned_connections, pruning can leave slots empty.
    chosen = select_heuristic(_candidates(POINTS), 4, _pair_distance(POINTS))
    assert [i for _, i in chosen] == [0, 2, 3]


def test_keep_pruned_connections_fills_free_slots_in_order() -> None:
    chosen = select_heuristic(
        _candidates(POINTS), 4, _pair_distance(POINTS), keep_pruned_connections=True
    )
    assert [i for _, i in chosen] == [0, 2, 3, 1]


def test_keep_pruned_never_exceeds_m() -> None:
    chosen = select_heuristic(
        _candidates(POINTS), 3, _pair_distance(POINTS), keep_pruned_connections=True
    )
    assert [i for _, i in chosen] == [0, 2, 3]


def test_heuristic_condition_is_strict() -> None:
    # E is exactly as far from A as from the base (x = 0.5), so it is discarded.
    points = {0: (1.0, 0.0), 1: (0.5, 2.0)}
    chosen = select_heuristic(_candidates(points), 2, _pair_distance(points))
    assert [i for _, i in chosen] == [0]


def test_heuristic_keeps_first_of_duplicate_candidates() -> None:
    # Duplicates are at distance 0 from each other, so only the first one survives.
    points = {0: (1.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0)}
    chosen = select_heuristic(_candidates(points), 3, _pair_distance(points))
    assert [i for _, i in chosen] == [0, 3]


def test_duplicate_of_base_starves_the_node() -> None:
    # The duplicate-vector risk: a candidate identical to the base has distance 0 to it. Once
    # kept, every other candidate e has d(e, duplicate) == d(e, base), and the strict rule
    # requires d(e, base) < d(e, duplicate), so *everything* else is pruned. The node ends up
    # linked only to its duplicate. keep_pruned_connections restores the other links.
    points = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (-1.0, 0.0)}
    cands = _candidates(points)
    assert [i for _, i in select_heuristic(cands, 3, _pair_distance(points))] == [0]
    filled = select_heuristic(cands, 3, _pair_distance(points), keep_pruned_connections=True)
    assert [i for _, i in filled] == [0, 1, 2]


def test_empty_candidates() -> None:
    assert select_simple([], 3) == []
    assert select_heuristic([], 3, _pair_distance(POINTS)) == []
