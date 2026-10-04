"""int8 scalar quantization: lossless on 8-bit data, at most half a step of error otherwise, and
a code-space search that ranks candidates exactly as the decoded vectors would."""

import numpy as np
import pytest

from hnsw_engine import HNSWIndex, QuantizedIndex, ScalarQuantizer, exact_distances, exact_search
from hnsw_engine.synthetic import clustered, gaussian, small_integers

from helpers import recall


def test_integer_data_spanning_256_values_is_lossless() -> None:
    data = small_integers(500, 6, seed=1, high=256) + 7.0
    quantizer = ScalarQuantizer.fit(data)
    assert quantizer.step == 1.0
    np.testing.assert_array_equal(quantizer.decode(quantizer.encode(data)), data)


def test_continuous_data_error_is_at_most_half_a_step() -> None:
    data = gaussian(1000, 8, seed=2) * 3.0
    quantizer = ScalarQuantizer.fit(data)
    widest = float((data.max(axis=0).astype(np.float64) - data.min(axis=0)).max())
    assert quantizer.step == pytest.approx(widest / 255)
    error = np.abs(quantizer.decode(quantizer.encode(data)) - data)
    assert error.max() <= quantizer.step / 2 + 1e-9


def test_integers_spanning_more_than_256_values_are_scaled() -> None:
    data = np.array([[0.0], [300.0]])
    assert ScalarQuantizer.fit(data).step == pytest.approx(300 / 255)


def test_out_of_range_values_are_clipped() -> None:
    quantizer = ScalarQuantizer.fit(np.array([[0.0, 0.0], [255.0, 10.0]]))
    np.testing.assert_array_equal(quantizer.encode(np.array([[-5.0, 300.0]])), [[0, 255]])


def test_fit_validation() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        ScalarQuantizer.fit(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="NaN"):
        ScalarQuantizer.fit(np.array([[np.nan, 1.0]]))
    assert ScalarQuantizer.fit(np.full((4, 2), 0.5)).step == 1.0


@pytest.mark.parametrize("metric", ["l2", "ip", "cosine"])
def test_code_space_distances_convert_to_decoded_distances(metric: str) -> None:
    data, queries = gaussian(300, 5, seed=3).astype(np.float64), gaussian(20, 5, seed=4)
    if metric == "cosine":
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    quantizer = ScalarQuantizer.fit(data)
    codes = quantizer.encode(data).astype(np.float64)
    code_q = quantizer.transform_queries(queries, metric)  # type: ignore[arg-type]
    decoded = quantizer.decode(quantizer.encode(data))
    q64 = queries.astype(np.float64)
    if metric == "l2":
        code_d = ((code_q[:, None, :] - codes[None]) ** 2).sum(-1)
        true_d = ((q64[:, None, :] - decoded[None]) ** 2).sum(-1)
    elif metric == "ip":
        code_d, true_d = -(code_q @ codes.T), -(q64 @ decoded.T)
    else:
        code_d, true_d = 1.0 - code_q @ codes.T, 1.0 - q64 @ decoded.T
    converted = quantizer.to_distance(code_d, q64, metric)  # type: ignore[arg-type]
    np.testing.assert_allclose(converted, true_d, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_lossless_codes_reproduce_the_float32_search_exactly(metric: str) -> None:
    data = small_integers(1500, 6, seed=4, high=5)
    queries = small_integers(40, 6, seed=5, high=5)
    index = HNSWIndex(6, metric=metric, M=6, ef_construction=40, seed=6)
    index.add(data)
    quantized = QuantizedIndex(index)
    assert quantized.quantizer.step == 1.0
    for a, b in zip(
        index.batch_search(queries, 10, ef=40),
        quantized.batch_search(queries, 10, ef=40),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_rerank_returns_exact_distances_and_recovers_float32_recall() -> None:
    data, queries = clustered(3000, 16, seed=7), clustered(60, 16, seed=8)
    index = HNSWIndex(16, M=12, ef_construction=80, seed=0)
    index.add(data)
    quantized = QuantizedIndex(index)
    truth, _ = exact_search(data, queries, 10)
    float_ids, _ = index.batch_search(queries, 10, ef=120)
    plain_ids, _ = quantized.batch_search(queries, 10, ef=120)
    ids, dist = quantized.batch_search(queries, 10, ef=120, rerank=40)
    np.testing.assert_allclose(dist, exact_distances(data, queries, ids), rtol=1e-12)
    # The 40 candidates contain the plain top 10, so exact re-scoring cannot lose a neighbour.
    assert recall(ids, truth) >= recall(plain_ids, truth)
    # Re-ranking recovers the quality of the float32 search on the same graph.
    assert recall(ids, truth) >= recall(float_ids, truth) - 0.01
    assert recall(ids, truth) >= 0.95


def test_cosine_quantized_search_with_rerank() -> None:
    data, queries = gaussian(2000, 12, seed=9), gaussian(50, 12, seed=10)
    index = HNSWIndex(12, metric="cosine", M=8, ef_construction=60, seed=0)
    index.add(data)
    truth, _ = exact_search(data, queries, 10, metric="cosine")
    ids, _ = QuantizedIndex(index).batch_search(queries, 10, ef=80, rerank=50)
    assert recall(ids, truth) >= 0.9


def test_memory_accounting() -> None:
    index = HNSWIndex(32, seed=0)
    index.add(gaussian(100, 32, seed=0))
    quantized = QuantizedIndex(index)
    assert quantized.code_bytes == 100 * 32
    assert quantized.vector_bytes == 4 * quantized.code_bytes


def test_quantized_index_validation() -> None:
    reference = HNSWIndex(4, seed=0, engine="reference")
    reference.add(gaussian(20, 4, seed=1))
    with pytest.raises(ValueError, match="engine='numba'"):
        QuantizedIndex(reference)
    with pytest.raises(ValueError, match="empty"):
        QuantizedIndex(HNSWIndex(4, seed=0))
    index = HNSWIndex(4, seed=0)
    index.add(gaussian(20, 4, seed=1))
    quantized = QuantizedIndex(index)
    with pytest.raises(ValueError, match=">= 0"):
        quantized.batch_search(gaussian(2, 4, seed=2), k=3, rerank=-1)
    index.add(gaussian(5, 4, seed=3))
    with pytest.raises(ValueError, match="changed"):
        quantized.batch_search(gaussian(2, 4, seed=2), k=3)
