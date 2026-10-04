"""The optimized Numba engine must reproduce the reference engine exactly.

On integer-valued data every distance is computed exactly in any summation order, so graphs and
search results must be bit-identical. On continuous data the two engines sum in different orders
(last-bit differences of about 1e-16 relative); graphs and returned ids must still be identical
and distances equal to 1e-12.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hnsw_engine import GraphState, HNSWIndex, check_invariants, exact_search, reachable_from_entry
from hnsw_engine.persistence import diff_states
from hnsw_engine.synthetic import clustered, gaussian, small_integers

from helpers import recall

CONFIGS = [
    {"selection": "simple"},
    {"selection": "heuristic"},
    {"selection": "heuristic", "extend_candidates": True},
    {"selection": "heuristic", "keep_pruned_connections": True},
    {"selection": "heuristic", "extend_candidates": True, "keep_pruned_connections": True},
]


def _pair(dim: int, **kwargs) -> tuple[HNSWIndex, HNSWIndex]:
    return (HNSWIndex(dim, engine="reference", **kwargs), HNSWIndex(dim, engine="numba", **kwargs))


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_bit_identical_on_integer_data(config: dict, metric: str) -> None:
    data = small_integers(1200, 6, seed=1, high=5)
    queries = small_integers(60, 6, seed=2, high=5)
    ref, fast = _pair(6, metric=metric, M=6, ef_construction=40, seed=3, **config)
    ref.add(data)
    fast.add(data)
    assert diff_states(ref.graph_state(), fast.graph_state()) == []
    for a, b in zip(
        ref.batch_search(queries, 10, ef=40), fast.batch_search(queries, 10, ef=40), strict=True
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_identical_graphs_on_continuous_data(config: dict, metric: str) -> None:
    data, queries = clustered(1500, 10, seed=4), clustered(80, 10, seed=5)
    ref, fast = _pair(10, metric=metric, M=8, ef_construction=50, seed=6, **config)
    ref.add(data)
    fast.add(data)
    assert diff_states(ref.graph_state(), fast.graph_state()) == []
    ids_a, dist_a = ref.batch_search(queries, 10, ef=50)
    ids_b, dist_b = fast.batch_search(queries, 10, ef=50)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_allclose(dist_a, dist_b, rtol=0, atol=1e-12)


@given(
    n=st.integers(1, 60),
    dim=st.integers(1, 5),
    M=st.integers(2, 6),
    efc=st.integers(1, 20),
    config=st.sampled_from(CONFIGS),
    seed=st.integers(0, 2**16),
    data_seed=st.integers(0, 2**16),
)
def test_equivalence_property(
    n: int, dim: int, M: int, efc: int, config: dict, seed: int, data_seed: int
) -> None:
    data = small_integers(n, dim, seed=data_seed, high=4)
    ref, fast = _pair(dim, M=M, ef_construction=efc, seed=seed, **config)
    ref.add(data)
    fast.add(data)
    assert diff_states(ref.graph_state(), fast.graph_state()) == []
    k = min(5, n)
    for a, b in zip(ref.batch_search(data[:5], k), fast.batch_search(data[:5], k), strict=True):
        np.testing.assert_array_equal(a, b)


def test_single_search_matches_batch_search() -> None:
    data = gaussian(500, 6, seed=1)
    index = HNSWIndex(6, M=6, seed=0, engine="numba")
    index.add(data)
    ids, dist = index.batch_search(data[:20], k=5, ef=30)
    for row in range(20):
        a, b = index.search(data[row], k=5, ef=30)
        np.testing.assert_array_equal(a, ids[row])
        np.testing.assert_array_equal(b, dist[row])


def test_growth_beyond_initial_capacity_and_chunked_adds() -> None:
    # Initial capacity is 1,024 nodes; upper-layer rows also grow.
    data = gaussian(3000, 4, seed=7)
    whole = HNSWIndex(4, M=4, ef_construction=20, seed=1, engine="numba")
    whole.add(data)
    chunked = HNSWIndex(4, M=4, ef_construction=20, seed=1, engine="numba")
    for start in range(0, 3000, 700):
        chunked.add(data[start : start + 700])
    assert diff_states(whole.graph_state(), chunked.graph_state()) == []
    reference = HNSWIndex(4, M=4, ef_construction=20, seed=1, engine="reference")
    reference.add(data)
    assert diff_states(whole.graph_state(), reference.graph_state()) == []
    assert check_invariants(whole.graph_state(), 4) == []


def test_cross_engine_persistence(tmp_path: Path) -> None:
    data = small_integers(800, 5, seed=3, high=6)
    fast = HNSWIndex(5, M=5, ef_construction=30, seed=2, engine="numba")
    fast.add(data[:500])
    fast.save(tmp_path / "fast.npz")
    as_reference = HNSWIndex.load(tmp_path / "fast.npz", engine="reference")
    as_fast = HNSWIndex.load(tmp_path / "fast.npz")
    assert as_fast.engine == "numba"
    for other in (as_reference, as_fast):
        assert diff_states(fast.graph_state(), other.graph_state()) == []
        other.add(data[500:])
    assert diff_states(as_reference.graph_state(), as_fast.graph_state()) == []


def test_exactness_regime_numba() -> None:
    n = 50
    data = small_integers(n, 4, seed=11, high=5)
    index = HNSWIndex(4, M=n, ef_construction=n, selection="simple", seed=0, engine="numba")
    index.add(data)
    found = index.batch_search(data[:20], k=8, ef=n)
    truth = exact_search(data, data[:20], 8)
    np.testing.assert_array_equal(found[0], truth[0])
    np.testing.assert_array_equal(found[1], truth[1])


def test_recall_and_invariants_numba() -> None:
    data, queries = clustered(3000, 16, seed=1), clustered(100, 16, seed=2)
    index = HNSWIndex(16, M=16, ef_construction=100, seed=0, engine="numba")
    index.add(data)
    assert check_invariants(index.graph_state(), 16) == []
    truth, _ = exact_search(data, queries, 10)
    found, _ = index.batch_search(queries, 10, ef=100)
    assert recall(found, truth) >= 0.95


def test_padding_and_counters_numba() -> None:
    index = HNSWIndex(2, M=2, seed=0, engine="numba")
    vectors = np.array([[0, 0], [1, 0], [5, 5]], dtype=np.float32)
    state = GraphState.from_adjacency(
        vectors, np.zeros(3, dtype=np.int64), 0, 0, [{0: [1], 1: [0], 2: [0]}]
    )
    index._engine.load_state(state)
    ids, dist = index.search(np.array([5.0, 5.0]), k=3, ef=10)
    np.testing.assert_array_equal(ids, [1, 0, -1])
    assert dist[2] == np.inf
    assert not reachable_from_entry(index.graph_state())[2]
    assert index.stats().distance_computations > 0
    index.reset_counters()
    assert index.stats().distance_computations == 0


def test_visited_epoch_wraparound() -> None:
    data = gaussian(300, 3, seed=1)
    index = HNSWIndex(3, M=4, seed=0, engine="numba")
    index.add(data)
    before = index.batch_search(data[:10], k=3)
    index._engine._state[2] = 4_294_967_290  # force the epoch counter to wrap during the searches
    after = index.batch_search(data[:10], k=3)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)
    assert index._engine._state[2] < 1000


def test_forced_levels_entry_point_numba() -> None:
    index = HNSWIndex(2, M=4, seed=0, engine="numba")
    data = gaussian(6, 2, seed=1)
    expected = [(0, 0), (1, 2), (1, 2), (3, 3), (3, 3), (3, 3)]
    for i, level in enumerate([0, 2, 1, 3, 3, 0]):
        index._add_with_levels(data[i : i + 1], [level])
        assert (index.entry_point, index.max_level) == expected[i]
    assert check_invariants(index.graph_state(), 4) == []


@pytest.mark.parametrize("selectivity", [1.0, 0.5, 0.1, 0.01])
@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_filtered_search_bit_identical(selectivity: float, metric: str) -> None:
    data = small_integers(1500, 6, seed=21, high=5)
    queries = small_integers(50, 6, seed=22, high=5)
    ref, fast = _pair(6, metric=metric, M=6, ef_construction=40, seed=23)
    ref.add(data)
    fast.add(data)
    allowed = np.random.default_rng(24).random(1500) < selectivity
    for ef in (10, 50):
        ref.reset_counters()
        fast.reset_counters()
        a = ref.batch_search(queries, 10, ef=ef, filter=allowed)
        b = fast.batch_search(queries, 10, ef=ef, filter=allowed)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert ref.stats().distance_computations == fast.stats().distance_computations
        assert ref.stats().hops == fast.stats().hops


@given(
    n=st.integers(1, 60),
    dim=st.integers(1, 4),
    M=st.integers(2, 6),
    seed=st.integers(0, 2**16),
    mask_seed=st.integers(0, 2**16),
    share=st.floats(0.0, 1.0),
)
def test_filtered_equivalence_property(
    n: int, dim: int, M: int, seed: int, mask_seed: int, share: float
) -> None:
    data = small_integers(n, dim, seed=seed, high=4)
    ref, fast = _pair(dim, M=M, ef_construction=10, seed=seed)
    ref.add(data)
    fast.add(data)
    allowed = np.random.default_rng(mask_seed).random(n) < share
    k = min(4, n)
    a = ref.batch_search(data[:5], k, ef=6, filter=allowed)
    b = fast.batch_search(data[:5], k, ef=6, filter=allowed)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ------------------------------------------------------- native (memory-mapped) files


def _digests(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_native_roundtrip_is_memory_mapped_and_identical(tmp_path: Path) -> None:
    data, queries = clustered(2500, 8, seed=1), clustered(40, 8, seed=2)
    index = HNSWIndex(8, M=6, ef_construction=40, seed=3, engine="numba")
    index.add(data)
    index.save_native(tmp_path / "idx")
    opened = HNSWIndex.load_native(tmp_path / "idx")
    assert opened.engine == "numba"
    assert not opened._engine._vectors.flags.writeable  # used in place, read-only
    assert diff_states(index.graph_state(), opened.graph_state()) == []
    allowed = np.random.default_rng(0).random(2500) < 0.2
    for mask in (None, allowed):
        for a, b in zip(
            index.batch_search(queries, 10, ef=40, filter=mask),
            opened.batch_search(queries, 10, ef=40, filter=mask),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)


def test_native_add_after_open_copies_and_leaves_files_untouched(tmp_path: Path) -> None:
    data = gaussian(1500, 5, seed=4)
    original = HNSWIndex(5, M=5, ef_construction=30, seed=1, engine="numba")
    original.add(data[:1000])
    original.save_native(tmp_path / "idx")
    before = _digests(tmp_path / "idx")
    opened = HNSWIndex.load_native(tmp_path / "idx")
    opened.add(data[1000:])  # the saved generator state continues the same random stream
    original.add(data[1000:])
    assert opened._engine._vectors.flags.writeable
    assert diff_states(original.graph_state(), opened.graph_state()) == []
    assert _digests(tmp_path / "idx") == before


def test_native_load_into_memory(tmp_path: Path) -> None:
    data = gaussian(800, 4, seed=5)
    index = HNSWIndex(4, M=4, seed=2, engine="numba")
    index.add(data)
    index.save_native(tmp_path / "idx")
    loaded = HNSWIndex.load_native(tmp_path / "idx", mmap=False)
    assert loaded._engine._vectors.flags.writeable
    assert diff_states(index.graph_state(), loaded.graph_state()) == []


def test_native_empty_index_roundtrip(tmp_path: Path) -> None:
    index = HNSWIndex(4, seed=0, engine="numba")
    index.save_native(tmp_path / "empty")
    opened = HNSWIndex.load_native(tmp_path / "empty")
    assert len(opened) == 0
    data = gaussian(30, 4, seed=1)
    opened.add(data)
    index.add(data)
    assert diff_states(index.graph_state(), opened.graph_state()) == []


def test_native_format_needs_the_numba_engine(tmp_path: Path) -> None:
    index = HNSWIndex(3, seed=0, engine="reference")
    index.add(gaussian(10, 3, seed=0))
    with pytest.raises(ValueError, match="engine='numba'"):
        index.save_native(tmp_path / "idx")


def test_native_rejects_foreign_or_inconsistent_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an hnsw-engine native index"):
        HNSWIndex.load_native(tmp_path)
    index = HNSWIndex(3, M=4, seed=0, engine="numba")
    index.add(gaussian(50, 3, seed=1))
    root = tmp_path / "idx"
    index.save_native(root)
    header = json.loads((root / "header.json").read_text())

    (root / "header.json").write_text(json.dumps({**header, "format": "other"}))
    with pytest.raises(ValueError, match="not an hnsw-engine native index"):
        HNSWIndex.load_native(root)
    (root / "header.json").write_text(json.dumps({**header, "format_version": 99}))
    with pytest.raises(ValueError, match="version"):
        HNSWIndex.load_native(root)
    wrong_m = {**header, "params": {**header["params"], "M": 5}}
    (root / "header.json").write_text(json.dumps(wrong_m))
    with pytest.raises(ValueError, match="links0"):
        HNSWIndex.load_native(root)
    (root / "header.json").write_text(json.dumps(header))
    np.save(root / "count0.npy", np.zeros(49, dtype=np.int32))
    with pytest.raises(ValueError, match="count0"):
        HNSWIndex.load_native(root)


def _corrupt(root: Path, name: str, change) -> None:  # type: ignore[no-untyped-def]
    array = np.load(root / f"{name}.npy")
    change(array)
    np.save(root / f"{name}.npy", array)


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("links0", lambda a: a.__setitem__((0, 0), 10**6), "outside the graph"),
        ("links0", lambda a: a.__setitem__((0, 0), -5), "outside the graph"),
        ("count0", lambda a: a.__setitem__(0, 99), "link counts"),
        ("upper_offset", lambda a: a.__setitem__(np.flatnonzero(a >= 0)[0], 10**6), "overlap"),
    ],
)
def test_native_load_rejects_corrupted_links(
    tmp_path: Path, name: str, change, message: str
) -> None:  # type: ignore[no-untyped-def]
    # The compiled search does not bounds-check, so a damaged file must fail at load time.
    index = HNSWIndex(4, M=4, seed=0, engine="numba")
    index.add(gaussian(400, 4, seed=1))
    root = tmp_path / "idx"
    index.save_native(root)
    _corrupt(root, name, change)
    with pytest.raises(ValueError, match=message):
        HNSWIndex.load_native(root)
    HNSWIndex.load_native(root, validate=False)  # trusted mode skips the check (do not search)


def test_native_load_rejects_upper_link_to_a_node_absent_from_the_layer(tmp_path: Path) -> None:
    index = HNSWIndex(4, M=4, seed=0, engine="numba")
    index.add(gaussian(400, 4, seed=1))
    root = tmp_path / "idx"
    index.save_native(root)
    levels = np.load(root / "levels.npy")
    ground = int(np.flatnonzero(levels == 0)[0])
    _corrupt(root, "links_up", lambda a: a.__setitem__((0, 0), ground))
    with pytest.raises(ValueError, match="absent from that layer"):
        HNSWIndex.load_native(root)


def test_native_load_rejects_a_bad_entry_point(tmp_path: Path) -> None:
    index = HNSWIndex(4, M=4, seed=0, engine="numba")
    index.add(gaussian(100, 4, seed=1))
    root = tmp_path / "idx"
    index.save_native(root)
    header = json.loads((root / "header.json").read_text())
    (root / "header.json").write_text(json.dumps({**header, "entry_point": 10**6}))
    with pytest.raises(ValueError, match="entry point"):
        HNSWIndex.load_native(root)
