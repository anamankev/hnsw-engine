"""Random level assignment (paper section 4, Algorithm 1 line 4).

Each inserted element gets a top layer ``l = floor(-ln(U) * mL)`` with ``U`` uniform on
``(0, 1]``. This makes ``P(l >= L) = exp(-L / mL)``; with the default ``mL = 1 / ln(M)`` that is
``M ** -L``, so every layer holds about ``1/M`` of the elements of the layer below it, exactly
like the levels of a skip list.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def default_m_l(M: int) -> float:
    """The paper's recommended normalisation factor ``mL = 1 / ln(M)``."""
    return 1.0 / math.log(M)


def sample_levels(rng: np.random.Generator, n: int, m_l: float) -> NDArray[np.int64]:
    """Draw ``n`` levels from ``rng``, one per element, in insertion order.

    ``rng.random`` returns values in ``[0, 1)``, so ``U = 1 - rng.random()`` lies in ``(0, 1]``
    and ``-ln(U)`` is always finite. Drawing ``n`` values at once consumes the generator exactly
    like ``n`` single draws, so adding vectors in one batch or in several produces the same
    levels.
    """
    u = 1.0 - rng.random(n)
    return np.floor(-np.log(u) * m_l).astype(np.int64)
