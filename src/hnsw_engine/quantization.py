"""int8 scalar quantization: one byte per dimension (``docs/DESIGN.md``, section 9.2).

:class:`ScalarQuantizer` maps a vector to codes ``round((x - offset) / step)`` clipped to
``[0, 255]``, with a per-dimension ``offset`` and one ``step`` shared by every dimension. The
shared step is what lets the compiled search run unchanged over the codes: for L2 the distance
between the transformed query ``(q - offset) / step`` and the codes is the distance to the
decoded vectors divided by ``step ** 2``, and for inner product and cosine the code-space
distance with the unchanged query is an increasing affine function of the decoded one. Either
way, candidates rank exactly as they would against the decoded vectors.

:class:`QuantizedIndex` traverses an existing graph (built from the ``float32`` vectors) with
distances to the codes, and can re-score the best candidates with exact ``float32`` distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_matrix, check_int
from .index import HNSWIndex
from .metrics import MetricName

CODE_LEVELS = 256


@dataclass(frozen=True)
class ScalarQuantizer:
    """Per-dimension offset and one shared step; codes are ``uint8``."""

    offset: NDArray[np.float64]
    step: float

    @classmethod
    def fit(cls, vectors: ArrayLike) -> ScalarQuantizer:
        """Offsets are the per-dimension minima; the step spreads the widest range over 255.

        If every value is an integer and every dimension spans at most 256 values, the step is 1
        and the codes are lossless (SIFT and Fashion-MNIST are 8-bit data).
        """
        x = np.asarray(as_matrix(vectors, "vectors"), dtype=np.float64)
        if x.shape[0] == 0:
            raise ValueError("cannot fit a quantizer on zero vectors")
        low = x.min(axis=0)
        widest = float((x.max(axis=0) - low).max())
        integer_valued = bool(np.array_equal(x, np.rint(x)))
        if (integer_valued and widest <= CODE_LEVELS - 1) or widest == 0.0:
            step = 1.0
        else:
            step = widest / (CODE_LEVELS - 1)
        return cls(low, step)

    def encode(self, vectors: ArrayLike) -> NDArray[np.uint8]:
        """Codes of ``vectors``; values outside the fitted range are clipped."""
        x = np.asarray(vectors, dtype=np.float64)
        codes = np.rint((x - self.offset) / self.step)
        return np.ascontiguousarray(np.clip(codes, 0, CODE_LEVELS - 1), dtype=np.uint8)

    def decode(self, codes: NDArray[np.uint8]) -> NDArray[np.float64]:
        """The vectors the codes stand for: ``offset + step * code``."""
        return self.offset + self.step * codes.astype(np.float64)

    def transform_queries(self, queries: ArrayLike, metric: MetricName) -> NDArray[np.float64]:
        """Queries in code space, so that ranking against codes equals ranking against decoded
        vectors: shifted and scaled for L2, unchanged for inner product and cosine."""
        q = np.asarray(queries, dtype=np.float64)
        if metric == "l2":
            return np.ascontiguousarray((q - self.offset) / self.step)
        return np.ascontiguousarray(q)

    def to_distance(
        self, code_distances: NDArray[np.float64], queries: ArrayLike, metric: MetricName
    ) -> NDArray[np.float64]:
        """Convert code-space distances (one row per query) to distances to decoded vectors."""
        if metric == "l2":
            return code_distances * self.step**2
        shift = np.asarray(queries, dtype=np.float64) @ self.offset  # offset . q, per query
        if metric == "ip":  # -(offset.q + step * c.q), and the code distance is -c.q
            return self.step * code_distances - shift[:, None]
        # cosine: 1 - offset.q - step * c.q, and the code distance is 1 - c.q
        return 1.0 - shift[:, None] - self.step + self.step * code_distances


class QuantizedIndex:
    """Searches the graph of an :class:`HNSWIndex` with distances to int8 codes.

    The graph is the one built from the ``float32`` vectors; only the distances computed during
    the search change. With ``rerank = R`` the best ``R`` candidates are re-scored with exact
    ``float32`` distances. Requires ``engine="numba"`` (the compiled search runs over the codes).
    The codes are a snapshot: build a new ``QuantizedIndex`` after adding vectors.
    """

    def __init__(self, index: HNSWIndex) -> None:
        if len(index) == 0:
            raise ValueError("cannot quantize an empty index")
        self._index = index
        self._engine = index._numba_engine("QuantizedIndex")
        vectors = self._engine.native_arrays()["vectors"]
        self.quantizer = ScalarQuantizer.fit(vectors)
        self.codes = self.quantizer.encode(vectors)

    @property
    def code_bytes(self) -> int:
        """Bytes of the codes (one per dimension per vector)."""
        return int(self.codes.nbytes)

    @property
    def vector_bytes(self) -> int:
        """Bytes of the ``float32`` vectors the codes replace during traversal."""
        return len(self._index) * self._index.dim * 4

    def batch_search(
        self, queries: ArrayLike, k: int = 10, ef: int | None = None, rerank: int = 0
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Approximate ``k`` nearest neighbours of each query row.

        Without re-ranking the distances are those to the decoded vectors. With ``rerank = R``
        the search collects ``max(R, k)`` candidates (and uses at least that ``ef``), then returns
        the best ``k`` by exact distance, ties to the lower id.
        """
        if len(self._index) != self.codes.shape[0]:
            raise ValueError("the index changed after quantization; build a new QuantizedIndex")
        k_valid, ef_valid = self._index._check_search_args(k, ef)
        r = check_int(rerank, "rerank", minimum=0)
        metric = self._index.metric
        prepared = self._index._prepare(queries)
        fetch = min(max(r, k_valid), len(self._index))
        ids, code_dist = self._engine.batch_search_codes(
            self.codes,
            self.quantizer.transform_queries(prepared, metric),
            fetch,
            max(ef_valid, fetch),
        )
        if r == 0:
            return ids, self.quantizer.to_distance(code_dist, prepared, metric)
        return self._engine.rerank(prepared, ids, k_valid)
