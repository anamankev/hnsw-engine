from pathlib import Path

import numpy as np
import pytest

from hnsw_engine import (
    GraphState,
    HNSWIndex,
    check_invariants,
    exact_search,
    reachable_from_entry,
    searchable_from_entry,
)
from hnsw_engine.persistence import diff_states
from hnsw_engine.synthetic import clustered, gaussian, small_integers

from helpers import assert_same_neighbors, recall


@pytest.fixture(params=["reference", "numba"], autouse=True)
def _every_engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every test in this module once per engine (explicit ``engine=`` arguments still win)."""
    original = HNSWIndex.__init__

    def init(self: HNSWIndex, *args: object, engine: str = request.param, **kwargs: object) -> None:
        original(self, *args, engine=engine, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(HNSWIndex, "__init__", init)
    return str(request.param)


# ----------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"dim": 0}, ValueError),
        ({"dim": 2.0}, TypeError),
        ({"M": 1}, ValueError),
        ({"ef_construction": 0}, ValueError),
        ({"ef_search": 0}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": 1.5}, TypeError),
        ({"metric": "hamming"}, ValueError),
        ({"selection": "random"}, ValueError),
        ({"selection": "simple", "extend_candidates": True}, ValueError),
        ({"selection": "simple", "keep_pruned_connections": True}, ValueError),
        ({"extend_candidates": 1}, TypeError),
        ({"m_L": -0.1}, ValueError),
        ({"m_L": float("nan")}, ValueError),
        ({"m_L": float("inf")}, ValueError),
        ({"m_L": True}, TypeError),
        ({"engine": "gpu"}, ValueError),
    ],
)
def test_parameter_validation(kwargs: dict, error: type[Exception]) -> None:
    params = {"dim": 4, **kwargs}
    with pytest.raises(error):
        HNSWIndex(**params)


def test_defaults_follow_the_paper() -> None:
    index = HNSWIndex(dim=3, M=16)
    assert index.m_L == pytest.approx(1 / np.log(16))
    assert index.selection == "heuristic"
    assert index.stats().max_links_layer0 == 32


def test_add_validation() -> None:
    index = HNSWIndex(dim=3, metric="cosine")
    with pytest.raises(ValueError, match="dimension 2, expected 3"):
        index.add(np.ones((2, 2)))
    with pytest.raises(ValueError, match="2-D"):
        index.add(np.ones(3))
    with pytest.raises(ValueError, match="NaN"):
        index.add(np.array([[1.0, np.nan, 0.0]]))
    with pytest.raises(ValueError, match="zero vectors"):
        index.add(np.zeros((1, 3)))
    assert len(index) == 0
    assert index.add(np.empty((0, 3))).shape == (0,)


def test_search_validation() -> None:
    index = HNSWIndex(dim=2)
    with pytest.raises(ValueError, match="empty index"):
        index.search(np.zeros(2), k=1)
    index.add(np.eye(2))
    for bad_k, error in [(0, ValueError), (3, ValueError), (1.0, TypeError)]:
        with pytest.raises(error):
            index.search(np.zeros(2), k=bad_k)
    with pytest.raises(ValueError, match=">= 1"):
        index.search(np.zeros(2), k=1, ef=0)
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.zeros(3), k=1)
    with pytest.raises(ValueError, match="1-D"):
        index.search(np.zeros((1, 2)), k=1)
    with pytest.raises(ValueError, match="NaN"):
        index.search(np.array([np.nan, 0.0]), k=1)
    with pytest.raises(ValueError, match=">= 1"):
        index.ef_search = 0


def test_add_returns_consecutive_ids() -> None:
    index = HNSWIndex(dim=2, seed=0)
    np.testing.assert_array_equal(index.add(np.ones((3, 2))), [0, 1, 2])
    np.testing.assert_array_equal(index.add(np.zeros((2, 2))), [3, 4])
    assert len(index) == 5


# ----------------------------------------------------------------- invariants

CONFIGS = [
    {"selection": "simple"},
    {"selection": "heuristic"},
    {"selection": "heuristic", "extend_candidates": True},
    {"selection": "heuristic", "keep_pruned_connections": True},
    {"selection": "heuristic", "extend_candidates": True, "keep_pruned_connections": True},
]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_invariants_hold_after_build(config: dict, metric: str) -> None:
    data = gaussian(400, 8, seed=1)
    index = HNSWIndex(dim=8, metric=metric, M=6, ef_construction=30, seed=2, **config)
    index.add(data)
    assert check_invariants(index.graph_state(), M=6) == []
    assert index.max_level >= 1  # the hierarchy was actually exercised


@pytest.mark.parametrize("config", CONFIGS)
def test_invariants_hold_with_duplicates_and_ties(config: dict) -> None:
    data = small_integers(300, 3, seed=4, high=3)  # 27 distinct points: heavy duplication
    index = HNSWIndex(dim=3, M=4, ef_construction=20, seed=5, **config)
    index.add(data)
    assert check_invariants(index.graph_state(), M=4) == []


def test_m_l_zero_collapses_to_single_layer() -> None:
    index = HNSWIndex(dim=8, M=8, ef_construction=40, m_L=0.0, seed=3)
    index.add(gaussian(500, 8, seed=3))
    state = index.graph_state()
    assert index.max_level == 0
    assert state.num_layers == 1
    assert (state.levels == 0).all()
    assert check_invariants(state, M=8) == []
    queries = gaussian(30, 8, seed=30)
    truth, _ = exact_search(state.vectors, queries, k=5)
    found, _ = index.batch_search(queries, k=5, ef=100)
    assert recall(found, truth) >= 0.9


# ------------------------------------------------------------------ exactness


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_no_pruning_and_full_ef_equals_oracle(metric: str) -> None:
    # Simple selection with M >= n never prunes: every node links to every earlier node, the
    # graph is complete, and a search with ef = n visits everything -> exact results.
    n = 60
    data = small_integers(n, 4, seed=8, high=5)
    queries = small_integers(25, 4, seed=9, high=5)
    index = HNSWIndex(dim=4, metric=metric, M=n, ef_construction=n, selection="simple", seed=1)
    index.add(data)
    assert reachable_from_entry(index.graph_state()).all()
    found_ids, found_dist = index.batch_search(queries, k=10, ef=n)
    true_ids, true_dist = exact_search(data, queries, k=10, metric=metric)
    np.testing.assert_array_equal(found_ids, true_ids)
    np.testing.assert_array_equal(found_dist, true_dist)


def test_no_pruning_and_full_ef_equals_oracle_cosine() -> None:
    n = 50
    data = small_integers(n, 4, seed=10, high=5) + 1.0  # no zero vectors
    queries = small_integers(20, 4, seed=11, high=5) + 1.0
    index = HNSWIndex(dim=4, metric="cosine", M=n, ef_construction=n, selection="simple", seed=1)
    index.add(data)
    found = index.batch_search(queries, k=8, ef=n)
    truth = exact_search(data, queries, k=8, metric="cosine")
    # float32 storage of normalised vectors vs float64 oracle: equal up to rounding.
    assert_same_neighbors(found[0], found[1], truth[0], truth[1], atol=1e-6)


# ----------------------------------------------------------- entry point / levels


def test_entry_point_updates_with_forced_levels() -> None:
    index = HNSWIndex(dim=2, M=4, seed=0)
    data = gaussian(6, 2, seed=1)
    expected = [(0, 0), (1, 2), (1, 2), (3, 3), (3, 3), (3, 3)]  # (entry point, max level)
    for i, level in enumerate([0, 2, 1, 3, 3, 0]):
        index._add_with_levels(data[i : i + 1], [level])
        assert (index.entry_point, index.max_level) == expected[i], i
    state = index.graph_state()
    np.testing.assert_array_equal(state.levels, [0, 2, 1, 3, 3, 0])
    assert check_invariants(state, M=4) == []
    assert state.adjacency(3) == {3: [4], 4: [3]}


def test_forced_levels_validation() -> None:
    index = HNSWIndex(dim=2)
    with pytest.raises(ValueError, match="levels"):
        index._add_with_levels(np.ones((2, 2)), [0])
    with pytest.raises(ValueError, match="levels"):
        index._add_with_levels(np.ones((1, 2)), [-1])


# ---------------------------------------------------------------- determinism


def test_same_seed_same_graph_and_results() -> None:
    data, queries = clustered(600, 6, seed=1), clustered(40, 6, seed=2)
    a = HNSWIndex(dim=6, M=8, ef_construction=50, seed=42)
    b = HNSWIndex(dim=6, M=8, ef_construction=50, seed=42)
    a.add(data)
    b.add(data)
    assert diff_states(a.graph_state(), b.graph_state()) == []
    for x, y in zip(a.batch_search(queries, 10), b.batch_search(queries, 10), strict=True):
        np.testing.assert_array_equal(x, y)


def test_different_seed_changes_levels() -> None:
    data = gaussian(300, 4, seed=1)
    a, b = HNSWIndex(dim=4, seed=1), HNSWIndex(dim=4, seed=2)
    a.add(data)
    b.add(data)
    assert not np.array_equal(a.graph_state().levels, b.graph_state().levels)


def test_chunked_add_equals_single_add() -> None:
    data = gaussian(500, 5, seed=3)
    whole = HNSWIndex(dim=5, M=6, ef_construction=40, seed=7)
    whole.add(data)
    chunked = HNSWIndex(dim=5, M=6, ef_construction=40, seed=7)
    for start in range(0, 500, 123):
        chunked.add(data[start : start + 123])
    assert diff_states(whole.graph_state(), chunked.graph_state()) == []


# ---------------------------------------------------------------- persistence


def test_save_load_roundtrip(tmp_path: Path) -> None:
    data, queries = gaussian(400, 6, seed=5), gaussian(20, 6, seed=6)
    index = HNSWIndex(dim=6, metric="cosine", M=5, ef_construction=30, ef_search=40, seed=3)
    index.add(data)
    path = tmp_path / "index.npz"
    index.save(path)
    loaded = HNSWIndex.load(path)
    assert repr(loaded) == repr(index)
    assert diff_states(index.graph_state(), loaded.graph_state()) == []
    for x, y in zip(index.batch_search(queries, 5), loaded.batch_search(queries, 5), strict=True):
        np.testing.assert_array_equal(x, y)


def test_load_then_add_continues_the_random_stream(tmp_path: Path) -> None:
    data = gaussian(600, 4, seed=8)
    uninterrupted = HNSWIndex(dim=4, M=6, ef_construction=30, seed=11)
    uninterrupted.add(data)
    first = HNSWIndex(dim=4, M=6, ef_construction=30, seed=11)
    first.add(data[:250])
    first.save(tmp_path / "half.npz")
    resumed = HNSWIndex.load(tmp_path / "half.npz")
    resumed.add(data[250:])
    assert diff_states(uninterrupted.graph_state(), resumed.graph_state()) == []


def test_load_rejects_foreign_files(tmp_path: Path) -> None:
    path = tmp_path / "other.npz"
    np.savez(path, x=np.zeros(3))
    with pytest.raises(ValueError, match="not an hnsw-engine index"):
        HNSWIndex.load(path)


def test_empty_index_roundtrip(tmp_path: Path) -> None:
    index = HNSWIndex(dim=3, seed=1)
    index.save(tmp_path / "empty.npz")
    loaded = HNSWIndex.load(tmp_path / "empty.npz")
    assert len(loaded) == 0
    loaded.add(np.ones((2, 3)))
    assert len(loaded) == 2


# ------------------------------------------------------------- search quality


@pytest.mark.parametrize("generator", [gaussian, clustered])
def test_recall_is_high_on_moderate_data(generator) -> None:
    data, queries = generator(2000, 16, seed=1), generator(100, 16, seed=2)
    index = HNSWIndex(dim=16, M=16, ef_construction=100, seed=0)
    index.add(data)
    truth, _ = exact_search(data, queries, k=10)
    found, _ = index.batch_search(queries, k=10, ef=100)
    assert recall(found, truth) >= 0.95


def test_indexed_points_find_themselves() -> None:
    data = gaussian(1000, 12, seed=4)
    index = HNSWIndex(dim=12, M=12, ef_construction=80, seed=0)
    index.add(data)
    found, dist = index.batch_search(data, k=1, ef=40)
    assert (found[:, 0] == np.arange(1000)).mean() >= 0.99
    assert (dist[found[:, 0] == np.arange(1000), 0] == 0).all()


def test_batch_search_equals_individual_searches() -> None:
    data, queries = gaussian(300, 5, seed=1), gaussian(10, 5, seed=2)
    index = HNSWIndex(dim=5, M=6, seed=0)
    index.add(data)
    batch_ids, batch_dist = index.batch_search(queries, k=4, ef=20)
    for row, q in enumerate(queries):
        ids, dist = index.search(q, k=4, ef=20)
        np.testing.assert_array_equal(ids, batch_ids[row])
        np.testing.assert_array_equal(dist, batch_dist[row])


def test_results_are_sorted_and_correctly_scored() -> None:
    data = gaussian(500, 7, seed=2)
    index = HNSWIndex(dim=7, M=8, seed=0)
    index.add(data)
    ids, dist = index.search(gaussian(1, 7, seed=3)[0], k=20, ef=50)
    assert (np.diff(dist) >= 0).all()
    assert len(set(ids.tolist())) == 20
    expected = (
        (data[ids].astype(np.float64) - gaussian(1, 7, seed=3)[0].astype(np.float64)) ** 2
    ).sum(1)
    np.testing.assert_allclose(dist, expected, rtol=1e-12)


def test_k_equal_to_size_returns_every_node() -> None:
    data = gaussian(40, 3, seed=1)
    # M = 20 gives layer 0 a cap of 40 >= n - 1: no link is ever pruned, so all links are
    # bidirectional and every node is reachable.
    index = HNSWIndex(dim=3, M=20, ef_construction=40, selection="simple", seed=0)
    index.add(data)
    ids, _ = index.search(data[0], k=40, ef=40)
    assert sorted(ids.tolist()) == list(range(40))


def test_unreachable_nodes_are_padded() -> None:
    # Hand-built graph where node 2 has no incoming links: a search can return only 2 nodes.
    index = HNSWIndex(dim=2, M=2, seed=0)
    vectors = np.array([[0, 0], [1, 0], [5, 5]], dtype=np.float32)
    state = GraphState.from_adjacency(
        vectors, np.zeros(3, dtype=np.int64), 0, 0, [{0: [1], 1: [0], 2: [0]}]
    )
    index._engine.load_state(state)
    ids, dist = index.search(np.array([5.0, 5.0]), k=3, ef=10)
    np.testing.assert_array_equal(ids, [1, 0, -1])
    assert dist[2] == np.inf
    assert not reachable_from_entry(index.graph_state())[2]


# --------------------------------------------------------------- stats & misc


def test_stats_are_consistent() -> None:
    index = HNSWIndex(dim=4, M=4, ef_construction=20, seed=1)
    index.add(gaussian(200, 4, seed=1))
    stats = index.stats()
    state = index.graph_state()
    assert stats.size == 200
    assert stats.nodes_per_layer[0] == 200
    assert stats.links_per_layer == tuple(len(ix) for ix in state.indices)
    assert stats.max_level == state.max_level
    link_slots = sum(n * ((8 if i == 0 else 4) + 1) for i, n in enumerate(stats.nodes_per_layer))
    assert stats.memory_bytes == 200 * 4 * 4 + link_slots * 4 + 200 * 4
    assert stats.as_dict()["metric"] == "l2"
    before = stats.distance_computations
    index.search(np.zeros(4), k=3)
    assert index.stats().distance_computations > before
    index.reset_counters()
    assert index.stats().distance_computations == 0
    assert index.stats().hops == 0


def test_neighbors_accessor() -> None:
    index = HNSWIndex(dim=2, M=2, seed=0)
    index._add_with_levels(np.array([[0, 0], [1, 0], [2, 0]]), [1, 0, 0])
    assert index.neighbors(0, layer=0) == index.graph_state().adjacency(0)[0]
    with pytest.raises(ValueError, match="not present"):
        index.neighbors(1, layer=1)
    with pytest.raises(ValueError, match="not present"):
        index.neighbors(5)


def test_ef_search_property_is_used() -> None:
    index = HNSWIndex(dim=3, ef_search=1, seed=0)
    index.add(gaussian(100, 3, seed=1))
    index.ef_search = 64
    assert index.ef_search == 64
    ids_default, _ = index.search(np.zeros(3), k=5)
    ids_explicit, _ = index.search(np.zeros(3), k=5, ef=64)
    np.testing.assert_array_equal(ids_default, ids_explicit)


def test_searchable_counts_nodes_reachable_through_upper_layers() -> None:
    # Node 2 has no incoming layer-0 link from the entry point's side, but a query can descend on
    # layer 1 to node 1 and continue on layer 0 from there.
    vectors = np.array([[0, 0], [9, 9], [9, 8]], dtype=np.float32)
    state = GraphState.from_adjacency(
        vectors,
        np.array([1, 1, 0]),
        0,
        1,
        [{0: [], 1: [2], 2: [1]}, {0: [1], 1: [0]}],
    )
    assert reachable_from_entry(state).tolist() == [True, False, False]
    assert searchable_from_entry(state).tolist() == [True, True, True]


# ------------------------------------------------------------- filtered search


def test_filter_validation() -> None:
    index = HNSWIndex(dim=2)
    index.add(np.eye(2))
    with pytest.raises(TypeError, match="boolean"):
        index.search(np.zeros(2), k=1, filter=np.array([1, 0]))
    with pytest.raises(ValueError, match="one entry per indexed vector"):
        index.search(np.zeros(2), k=1, filter=np.array([True]))
    with pytest.raises(ValueError, match="one entry per indexed vector"):
        index.batch_search(np.zeros((1, 2)), k=1, filter=np.ones((2, 1), dtype=bool))


def test_all_true_filter_equals_unfiltered_search() -> None:
    # With every node allowed, the filtered layer-0 search takes exactly the same steps.
    data, queries = clustered(800, 6, seed=1), clustered(30, 6, seed=2)
    index = HNSWIndex(dim=6, M=6, ef_construction=40, seed=0)
    index.add(data)
    index.reset_counters()
    plain = index.batch_search(queries, k=10, ef=30)
    plain_stats = index.stats()
    index.reset_counters()
    filtered = index.batch_search(queries, k=10, ef=30, filter=np.ones(800, dtype=bool))
    for a, b in zip(plain, filtered, strict=True):
        np.testing.assert_array_equal(a, b)
    assert index.stats().distance_computations == plain_stats.distance_computations
    assert index.stats().hops == plain_stats.hops


@pytest.mark.parametrize("selectivity", [0.5, 0.1, 0.02])
def test_filter_returns_only_allowed_nodes_with_high_recall(selectivity: float) -> None:
    data, queries = gaussian(2000, 8, seed=3), gaussian(50, 8, seed=4)
    index = HNSWIndex(dim=8, M=8, ef_construction=60, seed=0)
    index.add(data)
    allowed = np.random.default_rng(5).random(2000) < selectivity
    ids, dist = index.batch_search(queries, k=10, ef=64, filter=allowed)
    assert (ids >= 0).all()
    assert allowed[ids].all()
    assert (np.diff(dist, axis=1) >= 0).all()
    subset = np.flatnonzero(allowed)
    truth, _ = exact_search(data[subset], queries, 10)
    assert recall(ids, subset[truth]) >= 0.9


def test_filter_with_full_ef_is_exact_over_allowed_nodes() -> None:
    # Complete graph (no pruning) and ef = n: the result set can never fill early, so the search
    # visits every node and must equal brute force over the allowed nodes, ties included.
    n = 60
    data = small_integers(n, 4, seed=12, high=5)
    queries = small_integers(20, 4, seed=13, high=5)
    index = HNSWIndex(dim=4, M=n, ef_construction=n, selection="simple", seed=1)
    index.add(data)
    allowed = np.random.default_rng(0).random(n) < 0.4
    subset = np.flatnonzero(allowed)
    assert subset.size >= 8
    found_ids, found_dist = index.batch_search(queries, k=8, ef=n, filter=allowed)
    true_ids, true_dist = exact_search(data[subset], queries, k=8)
    np.testing.assert_array_equal(found_ids, subset[true_ids])
    np.testing.assert_array_equal(found_dist, true_dist)


def test_filter_with_fewer_allowed_nodes_than_k_pads() -> None:
    data = gaussian(300, 4, seed=6)
    index = HNSWIndex(dim=4, M=6, seed=0)
    index.add(data)
    allowed = np.zeros(300, dtype=bool)
    allowed[[7, 70, 170]] = True
    ids, dist = index.search(data[70], k=5, ef=10, filter=allowed)
    assert ids[0] == 70
    assert dist[0] == 0
    assert sorted(ids[:3].tolist()) == [7, 70, 170]
    np.testing.assert_array_equal(ids[3:], [-1, -1])
    assert np.isinf(dist[3:]).all()


def test_filter_allowing_nothing_returns_padding_without_searching() -> None:
    index = HNSWIndex(dim=3, M=4, seed=0)
    index.add(gaussian(50, 3, seed=1))
    index.reset_counters()
    nothing = np.zeros(50, dtype=bool)
    ids, dist = index.batch_search(gaussian(4, 3, seed=2), k=3, filter=nothing)
    assert (ids == -1).all()
    assert np.isinf(dist).all()
    one_ids, _ = index.search(gaussian(1, 3, seed=3)[0], k=2, filter=nothing)
    assert (one_ids == -1).all()
    assert index.stats().distance_computations == 0


def test_filtered_batch_search_equals_individual_searches() -> None:
    data, queries = gaussian(400, 5, seed=1), gaussian(10, 5, seed=2)
    index = HNSWIndex(dim=5, M=6, seed=0)
    index.add(data)
    allowed = np.random.default_rng(1).random(400) < 0.2
    batch_ids, batch_dist = index.batch_search(queries, k=4, ef=20, filter=allowed)
    for row, q in enumerate(queries):
        ids, dist = index.search(q, k=4, ef=20, filter=allowed)
        np.testing.assert_array_equal(ids, batch_ids[row])
        np.testing.assert_array_equal(dist, batch_dist[row])


@pytest.mark.parametrize("bad", [10**6, -3])
def test_load_rejects_out_of_range_links(tmp_path: Path, bad: int) -> None:
    # A damaged .npz must fail before any engine (the compiled one does not bounds-check) uses it.
    index = HNSWIndex(dim=3, M=4, seed=0)
    index.add(gaussian(200, 3, seed=1))
    index.save(tmp_path / "good.npz")
    with np.load(tmp_path / "good.npz") as data:
        arrays = {name: data[name] for name in data.files}
    arrays["indices_0"][0] = bad
    with open(tmp_path / "bad.npz", "wb") as handle:
        np.savez(handle, **arrays)
    with pytest.raises(ValueError, match="outside the layer"):
        HNSWIndex.load(tmp_path / "bad.npz")
