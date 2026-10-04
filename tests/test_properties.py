"""Property-based tests: random small datasets and parameters, invariants that must always hold.

Data are small integers so that exact ties and duplicates are common (the hard cases for
tie-breaking) and float64 distance sums are exact.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from hnsw_engine import HNSWIndex, check_invariants, exact_distances, exact_search
from hnsw_engine.persistence import diff_states


@pytest.fixture(params=["reference", "numba"], autouse=True)
def _every_engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every test in this module once per engine (explicit ``engine=`` arguments still win)."""
    original = HNSWIndex.__init__

    def init(self: HNSWIndex, *args: object, engine: str = request.param, **kwargs: object) -> None:
        original(self, *args, engine=engine, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(HNSWIndex, "__init__", init)
    return str(request.param)


@st.composite
def datasets(draw: st.DrawFn, *, nonzero: bool = False) -> np.ndarray:
    n = draw(st.integers(1, 40))
    dim = draw(st.integers(1, 5))
    low = 1 if nonzero else 0
    values = draw(st.lists(st.integers(low, 3), min_size=n * dim, max_size=n * dim))
    return np.array(values, dtype=np.float32).reshape(n, dim)


@st.composite
def index_params(draw: st.DrawFn) -> dict:
    selection = draw(st.sampled_from(["simple", "heuristic"]))
    heuristic = selection == "heuristic"
    return {
        "M": draw(st.integers(2, 6)),
        "ef_construction": draw(st.integers(1, 16)),
        "selection": selection,
        "extend_candidates": heuristic and draw(st.booleans()),
        "keep_pruned_connections": heuristic and draw(st.booleans()),
        "seed": draw(st.integers(0, 2**16)),
    }


@given(data=datasets(), params=index_params(), metric=st.sampled_from(["l2", "ip"]))
def test_invariants_always_hold(data: np.ndarray, params: dict, metric: str) -> None:
    index = HNSWIndex(dim=data.shape[1], metric=metric, **params)
    index.add(data)
    assert check_invariants(index.graph_state(), params["M"]) == []


@given(data=datasets(nonzero=True), params=index_params())
def test_invariants_always_hold_cosine(data: np.ndarray, params: dict) -> None:
    index = HNSWIndex(dim=data.shape[1], metric="cosine", **params)
    index.add(data)
    assert check_invariants(index.graph_state(), params["M"]) == []


@given(
    data=datasets(),
    params=index_params(),
    metric=st.sampled_from(["l2", "ip"]),
    k=st.integers(1, 40),
    ef=st.integers(1, 50),
)
def test_search_results_are_valid(
    data: np.ndarray, params: dict, metric: str, k: int, ef: int
) -> None:
    k = min(k, data.shape[0])
    index = HNSWIndex(dim=data.shape[1], metric=metric, **params)
    index.add(data)
    ids, dist = index.search(data[0], k=k, ef=ef)
    assert ids.shape == (k,)
    found = ids >= 0
    # Padding, if any, only at the tail; real results unique, in range, sorted, correctly scored.
    assert (found[: found.sum()]).all()
    real = ids[found]
    assert len(set(real.tolist())) == real.size
    assert ((real >= 0) & (real < data.shape[0])).all()
    assert (np.diff(dist[found]) >= 0).all()
    rescored = exact_distances(data, data[:1], real[None, :], metric)[0]
    np.testing.assert_array_equal(dist[found], rescored)
    assert (dist[~found] == np.inf).all()


@given(data=datasets(), metric=st.sampled_from(["l2", "ip"]), seed=st.integers(0, 2**16))
def test_complete_graph_search_equals_oracle(data: np.ndarray, metric: str, seed: int) -> None:
    n = data.shape[0]
    M = max(n, 2)
    index = HNSWIndex(
        dim=data.shape[1], metric=metric, M=M, ef_construction=n, selection="simple", seed=seed
    )
    index.add(data)
    k = min(5, n)
    found_ids, found_dist = index.batch_search(data[:3], k=k, ef=n)
    true_ids, true_dist = exact_search(data, data[:3], k=k, metric=metric)
    np.testing.assert_array_equal(found_ids, true_ids)
    np.testing.assert_array_equal(found_dist, true_dist)


@given(data=datasets(), params=index_params())
def test_build_is_deterministic(data: np.ndarray, params: dict) -> None:
    a = HNSWIndex(dim=data.shape[1], **params)
    b = HNSWIndex(dim=data.shape[1], **params)
    a.add(data)
    b.add(data)
    assert diff_states(a.graph_state(), b.graph_state()) == []


@given(
    data=datasets(),
    params=index_params(),
    metric=st.sampled_from(["l2", "ip"]),
    k=st.integers(1, 40),
    ef=st.integers(1, 50),
    mask_seed=st.integers(0, 2**16),
)
def test_filtered_search_results_are_valid(
    data: np.ndarray, params: dict, metric: str, k: int, ef: int, mask_seed: int
) -> None:
    n = data.shape[0]
    k = min(k, n)
    allowed = np.random.default_rng(mask_seed).random(n) < 0.5
    index = HNSWIndex(dim=data.shape[1], metric=metric, **params)
    index.add(data)
    ids, dist = index.search(data[0], k=k, ef=ef, filter=allowed)
    found = ids >= 0
    assert (found[: found.sum()]).all()
    real = ids[found]
    assert allowed[real].all()
    assert len(set(real.tolist())) == real.size
    assert real.size <= allowed.sum()
    assert (np.diff(dist[found]) >= 0).all()
    rescored = exact_distances(data, data[:1], real[None, :], metric)[0]
    np.testing.assert_array_equal(dist[found], rescored)
    assert (dist[~found] == np.inf).all()


@given(data=datasets(), metric=st.sampled_from(["l2", "ip"]), mask_seed=st.integers(0, 2**16))
def test_complete_graph_filtered_search_equals_oracle_on_allowed_nodes(
    data: np.ndarray, metric: str, mask_seed: int
) -> None:
    n = data.shape[0]
    allowed = np.random.default_rng(mask_seed).random(n) < 0.5
    subset = np.flatnonzero(allowed)
    index = HNSWIndex(
        dim=data.shape[1],
        metric=metric,
        M=max(n, 2),
        ef_construction=n,
        selection="simple",
        seed=mask_seed,
    )
    index.add(data)
    k = min(5, n)
    found_ids, found_dist = index.batch_search(data[:3], k=k, ef=n, filter=allowed)
    if subset.size == 0:
        assert (found_ids == -1).all()
        return
    kk = min(k, subset.size)
    true_ids, true_dist = exact_search(data[subset], data[:3], k=kk, metric=metric)
    np.testing.assert_array_equal(found_ids[:, :kk], subset[true_ids])
    np.testing.assert_array_equal(found_dist[:, :kk], true_dist)
    assert (found_ids[:, kk:] == -1).all()
