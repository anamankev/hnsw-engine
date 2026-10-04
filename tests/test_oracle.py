import numpy as np
import pytest

from hnsw_engine import exact_distances, exact_search

from helpers import assert_same_neighbors, pure_python_knn


def test_hand_checkable_l2() -> None:
    data = np.array([[0, 0], [1, 0], [0, 2], [3, 3]], dtype=np.float32)
    ids, dist = exact_search(data, np.array([[0.0, 0.0]]), k=3)
    np.testing.assert_array_equal(ids, [[0, 1, 2]])
    np.testing.assert_array_equal(dist, [[0.0, 1.0, 4.0]])


def test_hand_checkable_ip_and_cosine() -> None:
    data = np.array([[1, 0], [0, 3], [2, 2]], dtype=np.float32)
    query = np.array([[1.0, 1.0]])
    ids, dist = exact_search(data, query, k=3, metric="ip")
    np.testing.assert_array_equal(ids, [[2, 1, 0]])  # dots 4, 3, 1
    np.testing.assert_array_equal(dist, [[-4.0, -3.0, -1.0]])
    ids, dist = exact_search(data, query, k=3, metric="cosine")
    np.testing.assert_array_equal(ids, [[2, 0, 1]])  # (2,2) is parallel; (1,0) and (0,3) tie
    np.testing.assert_allclose(dist[0], [0.0, 1 - 2**-0.5, 1 - 2**-0.5], atol=1e-12)


def test_duplicates_are_ordered_by_id() -> None:
    data = np.array([[5, 5], [1, 1], [5, 5], [1, 1], [9, 9]], dtype=np.float32)
    ids, dist = exact_search(data, np.array([[1.0, 1.0], [5.0, 5.0]]), k=4)
    np.testing.assert_array_equal(ids, [[1, 3, 0, 2], [0, 2, 1, 3]])
    np.testing.assert_array_equal(dist[:, :2], 0.0)


def test_ties_are_broken_by_lower_id() -> None:
    # Four points at distance 1 from the query, stored in shuffled order.
    data = np.array([[9, 9], [0, -1], [1, 0], [-1, 0], [0, 1]], dtype=np.float32)
    ids, dist = exact_search(data, np.zeros((1, 2)), k=4)
    np.testing.assert_array_equal(ids, [[1, 2, 3, 4]])
    np.testing.assert_array_equal(dist, [[1.0, 1.0, 1.0, 1.0]])


def test_tie_at_kth_position_picks_lower_id() -> None:
    data = np.array([[2, 0], [0, 0], [-2, 0], [0, 2]], dtype=np.float32)
    ids, _ = exact_search(data, np.array([[0.0, 0.0]]), k=2)
    np.testing.assert_array_equal(ids, [[1, 0]])  # ids 0, 2, 3 all tie for second place


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_agrees_with_pure_python_on_integer_data(metric: str) -> None:
    rng = np.random.default_rng(7)
    data = rng.integers(1, 5, size=(80, 4)).astype(np.float32)  # many exact ties
    queries = rng.integers(1, 5, size=(15, 4)).astype(np.float32)
    ids, dist = exact_search(data, queries, k=10, metric=metric)
    for row, q in enumerate(queries):
        expected = pure_python_knn(data.tolist(), q.tolist(), 10, metric)
        exp_ids = np.array([[i for _, i in expected]])
        exp_dist = np.array([[d for d, _ in expected]])
        if metric == "cosine":  # sqrt/normalisation differ in the last bits
            assert_same_neighbors(ids[row : row + 1], dist[row : row + 1], exp_ids, exp_dist)
        else:  # integer arithmetic is exact, so ids must match exactly
            np.testing.assert_array_equal(ids[row], exp_ids[0])
            np.testing.assert_array_equal(dist[row], exp_dist[0])


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_agrees_with_pure_python_on_float_data(metric: str) -> None:
    rng = np.random.default_rng(11)
    data = rng.standard_normal((200, 6)).astype(np.float32)
    queries = rng.standard_normal((10, 6)).astype(np.float32)
    ids, dist = exact_search(data, queries, k=7, metric=metric)
    for row, q in enumerate(queries):
        expected = pure_python_knn(
            data.astype(np.float64).tolist(), q.astype(np.float64).tolist(), 7, metric
        )
        assert_same_neighbors(
            ids[row : row + 1],
            dist[row : row + 1],
            np.array([[i for _, i in expected]]),
            np.array([[d for d, _ in expected]]),
        )


def test_exact_under_catastrophic_cancellation() -> None:
    # Large offsets make |q|^2 - 2q.x + |x|^2 lose all precision in float32; the exact float64
    # re-ranking step must still return the true neighbours.
    rng = np.random.default_rng(3)
    data = (1e4 + rng.integers(0, 4, size=(300, 8))).astype(np.float32)
    queries = (1e4 + rng.integers(0, 4, size=(20, 8)) + 0.5).astype(np.float32)
    ids, dist = exact_search(data, queries, k=5)
    for row, q in enumerate(queries):
        expected = pure_python_knn(
            data.astype(np.float64).tolist(), q.astype(np.float64).tolist(), 5, "l2"
        )
        np.testing.assert_array_equal(ids[row], [i for _, i in expected])
        np.testing.assert_array_equal(dist[row], [d for d, _ in expected])


def test_block_sizes_do_not_change_results() -> None:
    rng = np.random.default_rng(5)
    data = rng.integers(0, 3, size=(101, 5)).astype(np.float32)
    queries = rng.integers(0, 3, size=(13, 5)).astype(np.float32)
    reference = exact_search(data, queries, k=9)
    for qb, db in [(1, 1), (3, 7), (13, 101), (4, 1000)]:
        got = exact_search(data, queries, k=9, query_block=qb, data_block=db)
        np.testing.assert_array_equal(got[0], reference[0])
        np.testing.assert_array_equal(got[1], reference[1])


def test_accepts_integer_dtypes_and_k_equal_n() -> None:
    data = np.array([[0, 0], [3, 4], [1, 1]], dtype=np.uint8)
    ids, dist = exact_search(data, np.array([[0, 0]], dtype=np.uint8), k=3)
    np.testing.assert_array_equal(ids, [[0, 2, 1]])
    np.testing.assert_array_equal(dist, [[0.0, 2.0, 25.0]])
    assert ids.dtype == np.int64
    assert dist.dtype == np.float64


def test_zero_queries_returns_empty_arrays() -> None:
    ids, dist = exact_search(np.ones((4, 2)), np.empty((0, 2)), k=2)
    assert ids.shape == (0, 2)
    assert dist.shape == (0, 2)


@pytest.mark.parametrize(
    ("data", "queries", "k", "metric", "message"),
    [
        (np.ones((3, 2)), np.ones((1, 2)), 0, "l2", ">= 1"),
        (np.ones((3, 2)), np.ones((1, 2)), 4, "l2", "exceeds"),
        (np.ones((3, 2)), np.ones((1, 2)), -1, "l2", ">= 1"),
        (np.ones((3, 2)), np.ones((1, 3)), 1, "l2", "dimension 3, expected 2"),
        (np.ones(3), np.ones((1, 3)), 1, "l2", "2-D"),
        (np.ones((3, 2)), np.ones(2), 1, "l2", "2-D"),
        (np.empty((0, 2)), np.ones((1, 2)), 1, "l2", "non-empty"),
        (np.array([[1.0, np.nan]]), np.ones((1, 2)), 1, "l2", "NaN"),
        (np.ones((3, 2)), np.array([[np.inf, 0.0]]), 1, "l2", "NaN or infinite"),
        (np.array([[0.0, 0.0], [1.0, 1.0]]), np.ones((1, 2)), 1, "cosine", "zero vectors"),
        (np.ones((3, 2)), np.zeros((1, 2)), 1, "cosine", "zero vectors"),
        (np.ones((3, 2)), np.ones((1, 2)), 1, "hamming", "unknown metric"),
    ],
)
def test_invalid_inputs_raise_value_error(
    data: np.ndarray, queries: np.ndarray, k: int, metric: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        exact_search(data, queries, k, metric)


@pytest.mark.parametrize("k", [1.5, "2", True, None])
def test_non_integer_k_raises_type_error(k: object) -> None:
    with pytest.raises(TypeError):
        exact_search(np.ones((3, 2)), np.ones((1, 2)), k)  # type: ignore[arg-type]


def test_exact_distances_scores_ids_and_marks_missing() -> None:
    data = np.array([[0, 0], [3, 4]], dtype=np.float32)
    out = exact_distances(data, np.array([[0.0, 0.0]]), np.array([[1, -1, 0]]))
    np.testing.assert_array_equal(out, [[25.0, np.inf, 0.0]])
    with pytest.raises(ValueError, match="outside"):
        exact_distances(data, np.array([[0.0, 0.0]]), np.array([[2]]))
    with pytest.raises(ValueError, match="shape"):
        exact_distances(data, np.array([[0.0, 0.0]]), np.array([1]))
