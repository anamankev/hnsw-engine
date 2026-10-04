import math

import numpy as np
import pytest

from hnsw_engine.levels import default_m_l, sample_levels


def test_default_m_l() -> None:
    assert default_m_l(16) == pytest.approx(1 / math.log(16))


@pytest.mark.parametrize("M", [4, 16, 32])
def test_level_distribution_matches_geometric_law(M: int) -> None:
    # P(level >= l) = exp(-l / mL) = M ** -l for mL = 1/ln(M).
    n = 400_000
    levels = sample_levels(np.random.default_rng(1234), n, default_m_l(M))
    for level in (1, 2, 3):
        p = float(M) ** -level
        observed = float((levels >= level).mean())
        sigma = math.sqrt(p * (1 - p) / n)
        assert abs(observed - p) < 5 * sigma + 1e-9, (level, observed, p)


def test_batch_draw_equals_sequential_draws() -> None:
    batch = sample_levels(np.random.default_rng(9), 500, 0.5)
    rng = np.random.default_rng(9)
    sequential = np.concatenate([sample_levels(rng, 1, 0.5) for _ in range(500)])
    np.testing.assert_array_equal(batch, sequential)


def test_m_l_zero_gives_single_layer() -> None:
    assert (sample_levels(np.random.default_rng(0), 10_000, 0.0) == 0).all()


class _FixedGenerator:
    """Stand-in generator returning chosen values from ``random``."""

    def __init__(self, values: list[float]) -> None:
        self.values = np.array(values)

    def random(self, n: int) -> np.ndarray:
        return self.values[:n]


def test_extreme_uniform_draws_stay_finite() -> None:
    # rng.random() == 0 maps to U = 1 (level 0); the largest draw below 1 maps to a large but
    # finite level. U = 0 can never occur, so -ln(U) is never infinite.
    largest_below_one = 1.0 - 2.0**-53
    levels = sample_levels(_FixedGenerator([0.0, largest_below_one]), 2, 1.0)  # type: ignore[arg-type]
    assert levels[0] == 0
    assert levels[1] == math.floor(-math.log(2.0**-53))


def test_levels_are_non_negative_int64() -> None:
    levels = sample_levels(np.random.default_rng(0), 1000, default_m_l(8))
    assert levels.dtype == np.int64
    assert (levels >= 0).all()
