"""Synthetic datasets for tests and controlled studies.

All generators are deterministic given ``seed`` and return ``float32`` arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._validation import check_int


def uniform(n: int, dim: int, seed: int) -> NDArray[np.float32]:
    """``n`` points drawn uniformly from the unit hypercube ``[0, 1)^dim``."""
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    return rng.random(
        (check_int(n, "n", minimum=0), check_int(dim, "dim", minimum=1)), dtype=np.float32
    )


def gaussian(n: int, dim: int, seed: int) -> NDArray[np.float32]:
    """``n`` points from a standard normal distribution."""
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    shape = (check_int(n, "n", minimum=0), check_int(dim, "dim", minimum=1))
    return rng.standard_normal(shape, dtype=np.float32)


def clustered(
    n: int, dim: int, seed: int, *, n_clusters: int = 10, spread: float = 0.05
) -> NDArray[np.float32]:
    """Gaussian mixture: ``n_clusters`` centres uniform in ``[0, 1)^dim``, points around them
    with standard deviation ``spread``. Tight, well-separated clusters are the case where
    simple neighbour selection tends to disconnect the graph."""
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    n = check_int(n, "n", minimum=0)
    dim = check_int(dim, "dim", minimum=1)
    n_clusters = check_int(n_clusters, "n_clusters", minimum=1)
    centres = rng.random((n_clusters, dim))
    labels = rng.integers(0, n_clusters, size=n)
    points = centres[labels] + spread * rng.standard_normal((n, dim))
    return points.astype(np.float32)


def with_duplicates(
    base: NDArray[np.float32], duplicate_fraction: float, seed: int
) -> NDArray[np.float32]:
    """Replace a fraction of the rows of ``base`` with exact copies of other rows."""
    if not 0.0 <= duplicate_fraction < 1.0:
        raise ValueError("duplicate_fraction must be in [0, 1)")
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    out = base.copy()
    n = base.shape[0]
    n_dup = round(duplicate_fraction * n)
    if n_dup == 0:
        return out
    targets = rng.choice(n, size=n_dup, replace=False)
    sources = rng.integers(0, n, size=n_dup)
    out[targets] = base[sources]
    return out


def small_integers(n: int, dim: int, seed: int, *, high: int = 4) -> NDArray[np.float32]:
    """Integer-valued points in ``[0, high)``: produces many exact ties and duplicates, and makes
    float64 distance sums exact in any summation order (used by equivalence tests)."""
    rng = np.random.default_rng(check_int(seed, "seed", minimum=0))
    shape = (check_int(n, "n", minimum=0), check_int(dim, "dim", minimum=1))
    return rng.integers(0, check_int(high, "high", minimum=1), size=shape).astype(np.float32)
