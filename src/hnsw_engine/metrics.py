"""Distance metrics.

Every metric is expressed as a *distance*, where smaller means more similar, so the graph code
uses one ordering convention for all of them:

=========  ===========================  ==========================================
name       distance                     notes
=========  ===========================  ==========================================
``l2``     ``sum((x - q)**2)``          squared Euclidean; the square root is
                                        monotonic, so it never changes a ranking
``cosine`` ``1 - cos(x, q)``            vectors are L2-normalised once on insert,
                                        after which this is ``1 - x.q``
``ip``     ``-x.q``                     maximum inner product search; not a true
                                        metric (no triangle inequality)
=========  ===========================  ==========================================

Precision contract (see ``docs/DESIGN.md`` section 5): vectors are stored as ``float32`` and
distances are accumulated in ``float64``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

MetricName = Literal["l2", "cosine", "ip"]
METRICS: tuple[MetricName, ...] = ("l2", "cosine", "ip")

F32Array = NDArray[np.float32]
F64Array = NDArray[np.float64]


def validate_metric(metric: object) -> MetricName:
    """Return ``metric`` if it names a supported metric, else raise ``ValueError``."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    return metric


def prepare(vectors: NDArray[np.generic], metric: MetricName) -> F32Array:
    """Convert validated vectors to the stored representation: contiguous ``float32`` rows.

    Under ``cosine`` each row is divided by its norm in ``float64`` first. Zero vectors have no
    direction, so they are rejected rather than silently mapped to an arbitrary distance.
    """
    if metric != "cosine":
        # Direct cast: identical to casting via float64, without a float64 copy of the data.
        with np.errstate(over="ignore"):  # overflow is detected and reported just below
            direct = np.ascontiguousarray(vectors, dtype=np.float32)
        if not np.isfinite(direct).all():
            raise ValueError("vectors overflow float32 (values beyond about 3.4e38)")
        return direct
    as64 = np.asarray(vectors, dtype=np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(as64, axis=1)
        zero_rows = np.flatnonzero(norms == 0.0)
        if zero_rows.size:
            raise ValueError(
                f"cosine distance is undefined for zero vectors (rows {zero_rows[:5].tolist()})"
            )
        as64 = as64 / norms[:, None]
    with np.errstate(over="ignore"):  # overflow is detected and reported just below
        out = np.ascontiguousarray(as64, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("vectors overflow float32 (values beyond about 3.4e38)")
    return out


def distances_to_many(metric: MetricName, query: F64Array, vectors: F32Array) -> F64Array:
    """Distances from one ``float64`` query to each row of ``vectors`` (``float32``).

    This is the reference engine's hot path: one vectorised call per expanded graph node.
    """
    rows = vectors.astype(np.float64)
    if metric == "l2":
        diff = rows - query
        squared: F64Array = np.einsum("ij,ij->i", diff, diff)
        return squared
    dots = rows @ query
    if metric == "ip":
        return -dots
    return 1.0 - dots
