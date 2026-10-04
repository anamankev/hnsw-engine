"""Input validation shared by every public entry point.

All checks raise ``ValueError`` (or ``TypeError`` for wrong Python types) with a message that
names the offending argument, so callers get the same behaviour from the oracle, the index and
the benchmark tools.
"""

from __future__ import annotations

import numbers

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_int(value: object, name: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as ``int`` after checking it is an integer ``>= minimum``.

    Booleans are rejected even though ``bool`` subclasses ``int``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def as_matrix(values: ArrayLike, name: str, *, dim: int | None = None) -> NDArray[np.generic]:
    """Validate a 2-D array of finite real numbers, optionally with a fixed column count.

    The array is returned without changing its dtype, so callers decide on precision.
    """
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}")
    _check_real_and_finite(array, name)
    if dim is not None and array.shape[1] != dim:
        raise ValueError(f"{name} has dimension {array.shape[1]}, expected {dim}")
    return array


def as_vector(values: ArrayLike, name: str, *, dim: int) -> NDArray[np.generic]:
    """Validate a 1-D array of ``dim`` finite real numbers."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    _check_real_and_finite(array, name)
    if array.shape[0] != dim:
        raise ValueError(f"{name} has dimension {array.shape[0]}, expected {dim}")
    return array


def check_k(k: object, n: int) -> int:
    """Validate the number of neighbours requested against the number of indexed vectors."""
    k_int = check_int(k, "k", minimum=1)
    if k_int > n:
        raise ValueError(f"k={k_int} exceeds the number of indexed vectors ({n})")
    return k_int


def _check_real_and_finite(array: NDArray[np.generic], name: str) -> None:
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"{name} must contain real numbers, got dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
