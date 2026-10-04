"""Uniform adapters over the indexes being compared.

Every adapter builds from the same ``float32`` training matrix, answers one query at a time
(ANN-Benchmarks "single query" mode), and exposes the same knobs. Cosine is handled the way each
library expects: ours and hnswlib normalise internally; FAISS indexes receive L2-normalised
vectors and use inner product. Query-side normalisation happens inside ``query`` so its cost is
counted for every library.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..index import HNSWIndex
from ..metrics import MetricName

F32 = NDArray[np.float32]


class Adapter(ABC):
    """Common interface. ``query_param`` names the knob swept at query time (``None`` if none)."""

    name: str
    query_param: str | None = None

    def __init__(self, metric: MetricName, **params: Any) -> None:
        self.metric = metric
        self.params = params

    def warm_up(self, dim: int) -> None:  # noqa: B027 - optional hook
        """Run once before memory is measured (e.g. to trigger JIT compilation)."""

    @abstractmethod
    def build(self, train: F32) -> None:
        """Index ``train``."""

    def set_query_param(self, value: int) -> None:  # noqa: B027 - optional hook
        """Set the query-time knob (``ef_search`` or ``nprobe``)."""

    @abstractmethod
    def query(self, vector: F32, k: int) -> NDArray[np.int64]:
        """Ids of the approximate ``k`` nearest neighbours of one query."""

    def logical_bytes(self) -> int | None:
        """Size of the index data structure in bytes (library-specific measure)."""
        return None

    def reset_counters(self) -> None:  # noqa: B027 - optional hook
        """Zero algorithmic work counters, if the library has them."""

    def counters(self) -> dict[str, int] | None:
        """Distance computations and hops since the last reset, if available."""
        return None

    def describe(self) -> dict[str, Any]:
        """Configuration recorded in results files."""
        return {"name": self.name, "metric": self.metric, **self.params}


class HnswEngineAdapter(Adapter):
    """This project's index (``engine`` = ``reference`` or ``numba``)."""

    query_param = "ef_search"

    def __init__(
        self, metric: MetricName, *, M: int, ef_construction: int, engine: str, seed: int = 0
    ) -> None:
        super().__init__(metric, M=M, ef_construction=ef_construction, engine=engine, seed=seed)
        self.name = f"hnsw-engine-{engine}"
        self.index: HNSWIndex | None = None

    def warm_up(self, dim: int) -> None:
        rng = np.random.default_rng(0)
        data = rng.random((64, dim), dtype=np.float32) + 0.1
        tiny = HNSWIndex(
            dim, metric=self.metric, M=4, ef_construction=8, engine=self.params["engine"]
        )
        tiny.add(data)
        tiny.search(data[0], k=1)
        tiny.batch_search(data[:2], k=1)

    def build(self, train: F32) -> None:
        self.index = HNSWIndex(
            train.shape[1],
            metric=self.metric,
            M=self.params["M"],
            ef_construction=self.params["ef_construction"],
            seed=self.params["seed"],
            engine=self.params["engine"],
        )
        self.index.add(train)

    def set_query_param(self, value: int) -> None:
        assert self.index is not None
        self.index.ef_search = value

    def query(self, vector: F32, k: int) -> NDArray[np.int64]:
        assert self.index is not None
        return self.index.search(vector, k)[0]

    def logical_bytes(self) -> int | None:
        assert self.index is not None
        return self.index.stats().memory_bytes

    def reset_counters(self) -> None:
        assert self.index is not None
        self.index.reset_counters()

    def counters(self) -> dict[str, int] | None:
        assert self.index is not None
        stats = self.index.stats()
        return {"distance_computations": stats.distance_computations, "hops": stats.hops}


class _FaissAdapter(Adapter):
    def __init__(self, metric: MetricName, **params: Any) -> None:
        super().__init__(metric, **params)
        import faiss

        self.faiss = faiss
        faiss.omp_set_num_threads(1)
        self.index: Any = None

    def _faiss_metric(self) -> int:
        return int(self.faiss.METRIC_L2 if self.metric == "l2" else self.faiss.METRIC_INNER_PRODUCT)

    def _prepare(self, vectors: F32) -> F32:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == "cosine":
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors

    def query(self, vector: F32, k: int) -> NDArray[np.int64]:
        q = self._prepare(vector[None, :])
        _, ids = self.index.search(q, k)
        return np.asarray(ids[0], dtype=np.int64)

    def logical_bytes(self) -> int | None:
        return int(self.faiss.serialize_index(self.index).nbytes)


class FaissHnswAdapter(_FaissAdapter):
    """FAISS ``IndexHNSWFlat``: M links above layer 0, 2M on layer 0, heuristic selection."""

    name = "faiss-hnsw"
    query_param = "ef_search"

    def __init__(self, metric: MetricName, *, M: int, ef_construction: int) -> None:
        super().__init__(metric, M=M, ef_construction=ef_construction)

    def build(self, train: F32) -> None:
        self.index = self.faiss.IndexHNSWFlat(
            train.shape[1], self.params["M"], self._faiss_metric()
        )
        self.index.hnsw.efConstruction = self.params["ef_construction"]
        self.index.add(self._prepare(train))

    def set_query_param(self, value: int) -> None:
        self.index.hnsw.efSearch = value

    def reset_counters(self) -> None:
        self.faiss.cvar.hnsw_stats.reset()

    def counters(self) -> dict[str, int] | None:
        stats = self.faiss.cvar.hnsw_stats
        return {"distance_computations": int(stats.ndis), "hops": int(stats.nhops)}


class FaissFlatAdapter(_FaissAdapter):
    """FAISS exact search (``IndexFlatL2`` / ``IndexFlatIP``): the speed of brute force."""

    name = "faiss-flat"

    def build(self, train: F32) -> None:
        cls = self.faiss.IndexFlatL2 if self.metric == "l2" else self.faiss.IndexFlatIP
        self.index = cls(train.shape[1])
        self.index.add(self._prepare(train))


class FaissIvfAdapter(_FaissAdapter):
    """FAISS ``IndexIVFFlat``: k-means partitions, scan the ``nprobe`` closest lists."""

    name = "faiss-ivf"
    query_param = "nprobe"

    def __init__(self, metric: MetricName, *, nlist: int) -> None:
        super().__init__(metric, nlist=nlist)

    def build(self, train: F32) -> None:
        dim = train.shape[1]
        cls = self.faiss.IndexFlatL2 if self.metric == "l2" else self.faiss.IndexFlatIP
        self._quantizer = cls(dim)
        self.index = self.faiss.IndexIVFFlat(
            self._quantizer, dim, self.params["nlist"], self._faiss_metric()
        )
        prepared = self._prepare(train)
        self.index.train(prepared)
        self.index.add(prepared)

    def set_query_param(self, value: int) -> None:
        self.index.nprobe = value


class HnswlibAdapter(Adapter):
    """hnswlib: the reference C++ HNSW implementation by the paper's first author."""

    name = "hnswlib"
    query_param = "ef_search"

    def __init__(self, metric: MetricName, *, M: int, ef_construction: int, seed: int = 0) -> None:
        super().__init__(metric, M=M, ef_construction=ef_construction, seed=seed)
        self.index: Any = None

    def build(self, train: F32) -> None:
        import hnswlib

        space = {"l2": "l2", "cosine": "cosine", "ip": "ip"}[self.metric]
        self.index = hnswlib.Index(space=space, dim=train.shape[1])
        self.index.init_index(
            max_elements=train.shape[0],
            ef_construction=self.params["ef_construction"],
            M=self.params["M"],
            random_seed=self.params["seed"],
        )
        self.index.set_num_threads(1)
        self.index.add_items(train, np.arange(train.shape[0]), num_threads=1)

    def set_query_param(self, value: int) -> None:
        self.index.set_ef(value)

    def query(self, vector: F32, k: int) -> NDArray[np.int64]:
        labels, _ = self.index.knn_query(vector, k=k, num_threads=1)
        return np.asarray(labels[0], dtype=np.int64)

    def logical_bytes(self) -> int | None:
        fd, path = tempfile.mkstemp(suffix=".hnswlib")
        os.close(fd)
        try:
            self.index.save_index(path)
            return os.path.getsize(path)
        finally:
            os.unlink(path)


def make_adapter(name: str, metric: MetricName, params: dict[str, Any]) -> Adapter:
    """Construct an adapter from its registry name and parameters."""
    if name in ("hnsw-engine-reference", "hnsw-engine-numba"):
        engine = name.removeprefix("hnsw-engine-")
        return HnswEngineAdapter(metric, engine=engine, **params)
    factories: dict[str, type[Adapter]] = {
        "faiss-hnsw": FaissHnswAdapter,
        "faiss-flat": FaissFlatAdapter,
        "faiss-ivf": FaissIvfAdapter,
        "hnswlib": HnswlibAdapter,
    }
    if name not in factories:
        raise ValueError(f"unknown index {name!r}")
    return factories[name](metric, **params)
