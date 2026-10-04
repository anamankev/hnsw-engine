import numpy as np
import pytest

from hnsw_engine._validation import as_matrix, as_vector, check_int, check_k
from hnsw_engine.metrics import distances_to_many, prepare, validate_metric


def test_validate_metric_accepts_known_names() -> None:
    for name in ("l2", "cosine", "ip"):
        assert validate_metric(name) == name


@pytest.mark.parametrize("bad", ["L2", "euclidean", "", None, 3])
def test_validate_metric_rejects_unknown(bad: object) -> None:
    with pytest.raises(ValueError, match="unknown metric"):
        validate_metric(bad)


def test_prepare_returns_contiguous_float32() -> None:
    x = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]  # non-contiguous view
    out = prepare(x, "l2")
    assert out.dtype == np.float32
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, x.astype(np.float32))


def test_prepare_cosine_normalises_rows() -> None:
    out = prepare(np.array([[3.0, 4.0], [0.0, -2.0]]), "cosine")
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, -1.0]], rtol=1e-7)


def test_prepare_cosine_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        prepare(np.array([[1.0, 0.0], [0.0, 0.0]]), "cosine")


def test_prepare_rejects_float32_overflow() -> None:
    with pytest.raises(ValueError, match="overflow"):
        prepare(np.array([[1e39, 0.0]]), "l2")


def test_distances_match_hand_computed_values() -> None:
    q = np.array([1.0, 2.0])
    x = np.array([[1.0, 2.0], [4.0, 6.0], [-1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(distances_to_many("l2", q, x), [0.0, 25.0, 8.0])
    np.testing.assert_allclose(distances_to_many("ip", q, x), [-5.0, -16.0, 1.0])
    unit = prepare(x, "cosine")
    qn = q / np.linalg.norm(q)
    expected = [1.0 - float(np.dot(qn, r / np.linalg.norm(r))) for r in x.astype(np.float64)]
    np.testing.assert_allclose(distances_to_many("cosine", qn, unit), expected, atol=1e-7)


def test_check_int() -> None:
    assert check_int(5, "x", minimum=1) == 5
    assert check_int(np.int64(3), "x") == 3
    with pytest.raises(ValueError, match=">= 1"):
        check_int(0, "x", minimum=1)
    for bad in (1.5, "3", None, True):
        with pytest.raises(TypeError):
            check_int(bad, "x")


def test_as_matrix_and_vector_validation() -> None:
    assert as_matrix([[1, 2]], "m").shape == (1, 2)
    with pytest.raises(ValueError, match="2-D"):
        as_matrix([1, 2], "m")
    with pytest.raises(ValueError, match="dimension 2, expected 3"):
        as_matrix([[1, 2]], "m", dim=3)
    with pytest.raises(ValueError, match="NaN"):
        as_matrix([[1.0, np.nan]], "m")
    with pytest.raises(ValueError, match="NaN or infinite"):
        as_matrix([[1.0, np.inf]], "m")
    with pytest.raises(ValueError, match="real numbers"):
        as_matrix(np.array([[1 + 2j]]), "m")
    with pytest.raises(ValueError, match="real numbers"):
        as_matrix(np.array([[True]]), "m")
    assert as_vector([1.0, 2.0], "v", dim=2).shape == (2,)
    with pytest.raises(ValueError, match="1-D"):
        as_vector([[1.0, 2.0]], "v", dim=2)


def test_check_k() -> None:
    assert check_k(3, 3) == 3
    with pytest.raises(ValueError, match="exceeds"):
        check_k(4, 3)
    with pytest.raises(ValueError, match=">= 1"):
        check_k(0, 3)


def test_prepare_direct_cast_matches_float64_route() -> None:
    rng = np.random.default_rng(0)
    for values in (
        rng.normal(size=(50, 7)),
        rng.normal(size=(50, 7)).astype(np.float32),
        rng.integers(-1000, 1000, size=(50, 7)),
    ):
        expected = np.asarray(values, dtype=np.float64).astype(np.float32)
        for metric in ("l2", "ip"):
            np.testing.assert_array_equal(prepare(values, metric), expected)
