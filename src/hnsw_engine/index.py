"""Public API: :class:`HNSWIndex`.

The facade owns everything that must be identical across engines -- validation, metric
preparation, the seeded random generator and level sampling -- and delegates graph
construction and search to an engine (``"reference"`` or ``"numba"``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_matrix, as_vector, check_int, check_k
from .levels import default_m_l, sample_levels
from .metrics import F32Array, MetricName, prepare, validate_metric
from .persistence import GraphState, load_native, load_state, save_native, save_state

if TYPE_CHECKING:
    from .numba_engine import NumbaEngine

EngineName = Literal["reference", "numba"]
SelectionName = Literal["simple", "heuristic"]
ENGINES: tuple[EngineName, ...] = ("reference", "numba")
SELECTIONS: tuple[SelectionName, ...] = ("simple", "heuristic")


class Engine(Protocol):
    """Interface every engine implements (see ``reference.ReferenceEngine``)."""

    name: str
    distance_computations: int
    hops: int

    @property
    def size(self) -> int: ...

    @property
    def entry_point(self) -> int: ...

    @property
    def max_level(self) -> int: ...

    def insert(self, vectors: F32Array, levels: NDArray[np.int64]) -> None: ...

    def search(
        self, query: F32Array, k: int, ef: int, allowed: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

    def batch_search(
        self, queries: F32Array, k: int, ef: int, allowed: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

    def export_state(self) -> GraphState: ...

    def load_state(self, state: GraphState) -> None: ...


@dataclass(frozen=True)
class IndexStats:
    """Summary of an index. ``memory_bytes`` is the size of a compact fixed-width C layout
    (float32 vectors; int32 link slots, ``2M + 1`` per node on layer 0 and ``M + 1`` per node on
    each upper layer; one int32 level per node), comparable to how FAISS and hnswlib store
    graphs. Real process memory is measured separately by the benchmark harness."""

    size: int
    dim: int
    metric: str
    engine: str
    M: int
    max_links_layer0: int
    ef_construction: int
    ef_search: int
    m_L: float
    selection: str
    extend_candidates: bool
    keep_pruned_connections: bool
    max_level: int
    entry_point: int
    nodes_per_layer: tuple[int, ...]
    links_per_layer: tuple[int, ...]
    mean_degree_per_layer: tuple[float, ...]
    memory_bytes: int
    distance_computations: int
    hops: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary of all fields."""
        return asdict(self)


class HNSWIndex:
    """Hierarchical Navigable Small World index for approximate nearest neighbour search.

    Args:
        dim: vector dimensionality.
        metric: ``"l2"`` (squared Euclidean), ``"cosine"`` (``1 - cos``) or ``"ip"`` (negative
            inner product). Distances are always "smaller is closer".
        M: links per node on layers above 0; layer 0 allows ``2 * M``.
        ef_construction: candidate list size while inserting (higher: better graph, slower).
        ef_search: default candidate list size while searching (higher: better recall, slower).
        seed: seed for level sampling; the same seed and data give the same graph.
        selection: ``"heuristic"`` (paper Algorithm 4, default) or ``"simple"`` (Algorithm 3).
        extend_candidates: heuristic option, also consider the candidates' neighbours.
        keep_pruned_connections: heuristic option, fill free slots with pruned candidates.
        m_L: level normalisation factor; defaults to ``1 / ln(M)``. ``0`` gives one layer.
        engine: ``"numba"`` (compiled, default) or ``"reference"`` (pure Python, readable). Both
            build identical graphs; see ``tests/test_numba_engine.py``.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "l2",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int | None = None,
        selection: str = "heuristic",
        extend_candidates: bool = False,
        keep_pruned_connections: bool = False,
        m_L: float | None = None,
        engine: str = "numba",
    ) -> None:
        self._dim = check_int(dim, "dim", minimum=1)
        self._metric: MetricName = validate_metric(metric)
        self._M = check_int(M, "M", minimum=2)
        self._ef_construction = check_int(ef_construction, "ef_construction", minimum=1)
        self._ef_search = check_int(ef_search, "ef_search", minimum=1)
        self._seed = None if seed is None else check_int(seed, "seed", minimum=0)
        if selection not in SELECTIONS:
            raise ValueError(f"unknown selection {selection!r}; expected one of {SELECTIONS}")
        self._selection: SelectionName = selection
        self._extend_candidates = _check_bool(extend_candidates, "extend_candidates")
        self._keep_pruned = _check_bool(keep_pruned_connections, "keep_pruned_connections")
        if self._selection == "simple" and (self._extend_candidates or self._keep_pruned):
            raise ValueError(
                "extend_candidates and keep_pruned_connections apply to selection='heuristic'"
            )
        self._m_L = default_m_l(self._M) if m_L is None else _check_m_l(m_L)
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
        self._engine_name: EngineName = engine
        self._rng = np.random.default_rng(self._seed)
        self._engine: Engine = self._make_engine()

    # ----------------------------------------------------------------- properties

    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim

    @property
    def metric(self) -> MetricName:
        """Distance metric name."""
        return self._metric

    @property
    def M(self) -> int:
        """Maximum links per node above layer 0 (``2 * M`` on layer 0)."""
        return self._M

    @property
    def ef_construction(self) -> int:
        """Candidate list size used during insertion."""
        return self._ef_construction

    @property
    def ef_search(self) -> int:
        """Default candidate list size used by :meth:`search`."""
        return self._ef_search

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self._ef_search = check_int(value, "ef_search", minimum=1)

    @property
    def m_L(self) -> float:
        """Level normalisation factor."""
        return self._m_L

    @property
    def selection(self) -> SelectionName:
        """Neighbour selection strategy."""
        return self._selection

    @property
    def engine(self) -> EngineName:
        """Engine name."""
        return self._engine_name

    @property
    def entry_point(self) -> int:
        """Id of the entry point (``-1`` when empty)."""
        return self._engine.entry_point

    @property
    def max_level(self) -> int:
        """Top layer index (``-1`` when empty)."""
        return self._engine.max_level

    def __len__(self) -> int:
        return self._engine.size

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(dim={self._dim}, metric={self._metric!r}, M={self._M}, "
            f"ef_construction={self._ef_construction}, ef_search={self._ef_search}, "
            f"selection={self._selection!r}, engine={self._engine_name!r}, size={len(self)})"
        )

    # ------------------------------------------------------------------- building

    def add(self, vectors: ArrayLike) -> NDArray[np.int64]:
        """Insert a ``(n, dim)`` array of vectors; returns their (consecutive) ids."""
        prepared = self._prepare(vectors)
        levels = sample_levels(self._rng, prepared.shape[0], self._m_L)
        return self._insert(prepared, levels)

    def _add_with_levels(self, vectors: ArrayLike, levels: ArrayLike) -> NDArray[np.int64]:
        """Insert with explicit levels instead of random ones. Intended for tests."""
        prepared = self._prepare(vectors)
        level_array = np.asarray(levels, dtype=np.int64)
        if level_array.shape != (prepared.shape[0],) or (level_array < 0).any():
            raise ValueError("levels must be a non-negative integer array, one per vector")
        return self._insert(prepared, level_array)

    def _insert(self, prepared: F32Array, levels: NDArray[np.int64]) -> NDArray[np.int64]:
        start = len(self)
        if prepared.shape[0]:
            self._engine.insert(prepared, levels)
        return np.arange(start, start + prepared.shape[0], dtype=np.int64)

    def _prepare(self, vectors: ArrayLike) -> F32Array:
        return prepare(as_matrix(vectors, "vectors", dim=self._dim), self._metric)

    # --------------------------------------------------------------------- search

    def search(
        self,
        query: ArrayLike,
        k: int = 10,
        ef: int | None = None,
        filter: ArrayLike | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Approximate ``k`` nearest neighbours of one query.

        Args:
            query: ``(dim,)`` vector.
            k: number of neighbours, ``1 <= k <= len(self)``.
            ef: candidate list size on layer 0; defaults to ``ef_search``; raised to ``k`` if
                smaller.
            filter: optional boolean array with one entry per indexed vector; only vectors
                marked ``True`` are returned. The graph search passes through the others
                (``docs/DESIGN.md``, section 5, rule 9), so a very selective filter makes the
                search visit many nodes.

        Returns:
            ``(ids, distances)`` of length ``k``, closest first. Padded with ``-1`` / ``inf``
            if fewer than ``k`` qualifying nodes are reachable (a disconnected graph, or a
            filter that allows fewer than ``k`` vectors).
        """
        k_valid, ef_valid = self._check_search_args(k, ef)
        allowed = self._check_filter(filter)
        prepared = self._prepare(as_vector(query, "query", dim=self._dim)[None, :])[0]
        if allowed is not None and not allowed.any():
            return np.full(k_valid, -1, dtype=np.int64), np.full(k_valid, np.inf)
        return self._engine.search(prepared, k_valid, ef_valid, allowed)

    def batch_search(
        self,
        queries: ArrayLike,
        k: int = 10,
        ef: int | None = None,
        filter: ArrayLike | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """:meth:`search` for each row of a ``(nq, dim)`` array; returns ``(nq, k)`` arrays.

        ``filter`` (one boolean per indexed vector) applies to every query.
        """
        k_valid, ef_valid = self._check_search_args(k, ef)
        allowed = self._check_filter(filter)
        prepared = self._prepare(queries)
        if allowed is not None and not allowed.any():
            shape = (prepared.shape[0], k_valid)
            return np.full(shape, -1, dtype=np.int64), np.full(shape, np.inf)
        return self._engine.batch_search(prepared, k_valid, ef_valid, allowed)

    def _check_search_args(self, k: int, ef: int | None) -> tuple[int, int]:
        if len(self) == 0:
            raise ValueError("cannot search an empty index")
        k_valid = check_k(k, len(self))
        ef_valid = self._ef_search if ef is None else check_int(ef, "ef", minimum=1)
        return k_valid, max(ef_valid, k_valid)

    def _check_filter(self, filter: ArrayLike | None) -> NDArray[np.bool_] | None:
        if filter is None:
            return None
        mask = np.asarray(filter)
        if mask.dtype != np.bool_:
            raise TypeError(f"filter must be a boolean array, got dtype {mask.dtype}")
        if mask.shape != (len(self),):
            raise ValueError(
                f"filter must have one entry per indexed vector ({len(self)}), got shape "
                f"{mask.shape}"
            )
        return np.ascontiguousarray(mask)

    # --------------------------------------------------------------- inspection

    def graph_state(self) -> GraphState:
        """Engine-independent snapshot of the graph (used for invariants and equivalence)."""
        return self._engine.export_state()

    def neighbors(self, node: int, layer: int = 0) -> list[int]:
        """Links of ``node`` on ``layer``."""
        state = self.graph_state()
        node = check_int(node, "node", minimum=0)
        layer = check_int(layer, "layer", minimum=0)
        if node >= state.size or layer > int(state.levels[node]):
            raise ValueError(f"node {node} is not present on layer {layer}")
        return state.adjacency(layer)[node]

    def stats(self) -> IndexStats:
        """Sizes, layer structure, degrees, memory estimate and cumulative work counters."""
        state = self.graph_state()
        nodes = tuple(int((state.levels >= layer).sum()) for layer in range(state.num_layers))
        links = tuple(int(state.indices[layer].shape[0]) for layer in range(state.num_layers))
        degrees = tuple(lk / nd if nd else 0.0 for lk, nd in zip(links, nodes, strict=True))
        link_slots = sum(
            nd * ((2 * self._M if layer == 0 else self._M) + 1) for layer, nd in enumerate(nodes)
        )
        memory = state.size * self._dim * 4 + link_slots * 4 + state.size * 4
        return IndexStats(
            size=state.size,
            dim=self._dim,
            metric=self._metric,
            engine=self._engine_name,
            M=self._M,
            max_links_layer0=2 * self._M,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
            m_L=self._m_L,
            selection=self._selection,
            extend_candidates=self._extend_candidates,
            keep_pruned_connections=self._keep_pruned,
            max_level=state.max_level,
            entry_point=state.entry_point,
            nodes_per_layer=nodes,
            links_per_layer=links,
            mean_degree_per_layer=degrees,
            memory_bytes=memory,
            distance_computations=self._engine.distance_computations,
            hops=self._engine.hops,
        )

    def reset_counters(self) -> None:
        """Zero the cumulative distance-computation and hop counters."""
        self._engine.distance_computations = 0
        self._engine.hops = 0

    # --------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Write the index (parameters, generator state and graph) to ``path`` as ``.npz``."""
        meta = {"params": self._params(), "rng_state": self._rng.bit_generator.state}
        save_state(path, meta, self.graph_state())

    @classmethod
    def load(cls, path: str | Path, engine: str | None = None) -> HNSWIndex:
        """Load an index written by :meth:`save`, optionally into a different engine."""
        meta, state = load_state(path)
        params = dict(meta["params"])
        if engine is not None:
            params["engine"] = engine
        index = cls(**params)
        if state.vectors.shape[1:] != (index.dim,):
            raise ValueError(
                f"stored vectors have shape {state.vectors.shape}, expected dim {index.dim}"
            )
        index._rng.bit_generator.state = meta["rng_state"]
        index._engine.load_state(state)
        return index

    def save_native(self, directory: str | Path) -> None:
        """Write the compiled engine's arrays to ``directory`` for :meth:`load_native`.

        One ``.npy`` file per array plus ``header.json`` (``docs/DESIGN.md``, section 9.3).
        Requires ``engine="numba"``; the reference engine has no native arrays (use :meth:`save`).
        """
        engine = self._numba_engine("save_native")
        meta = {
            "params": self._params(),
            "rng_state": self._rng.bit_generator.state,
            "entry_point": engine.entry_point,
            "max_level": engine.max_level,
        }
        save_native(directory, meta, engine.native_arrays())

    @classmethod
    def load_native(
        cls, directory: str | Path, mmap: bool = True, validate: bool = True
    ) -> HNSWIndex:
        """Open an index written by :meth:`save_native` (always with the Numba engine).

        With ``mmap`` (the default) the arrays are read-only memory maps of the files: the
        operating system reads pages as searches touch them, processes that open the same files
        share them, and adding vectors copies the arrays into memory first, so the files are
        never modified. ``validate`` (the default) checks every link id before the compiled code
        uses it, which reads the link arrays once; ``validate=False`` opens in constant time and
        is safe only for files you produced yourself.
        """
        meta, arrays = load_native(directory, mmap=mmap)
        params = {**meta["params"], "engine": "numba"}
        index = cls(**params)
        index._rng.bit_generator.state = meta["rng_state"]
        index._numba_engine("load_native").attach_native(
            arrays, int(meta["entry_point"]), int(meta["max_level"]), validate=validate
        )
        return index

    def _numba_engine(self, operation: str) -> NumbaEngine:
        if self._engine_name != "numba":
            raise ValueError(
                f"{operation} needs engine='numba' (the reference engine has no native arrays); "
                "use save() and load()"
            )
        return cast("NumbaEngine", self._engine)

    def _params(self) -> dict[str, Any]:
        return {
            "dim": self._dim,
            "metric": self._metric,
            "M": self._M,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "seed": self._seed,
            "selection": self._selection,
            "extend_candidates": self._extend_candidates,
            "keep_pruned_connections": self._keep_pruned,
            "m_L": self._m_L,
            "engine": self._engine_name,
        }

    def _make_engine(self) -> Engine:
        args = (
            self._dim,
            self._metric,
            self._M,
            self._ef_construction,
            self._selection,
            self._extend_candidates,
            self._keep_pruned,
        )
        if self._engine_name == "numba":
            from .numba_engine import NumbaEngine  # imported lazily: loading Numba takes time

            return NumbaEngine(*args)
        from .reference import ReferenceEngine

        return ReferenceEngine(*args)


def _check_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _check_m_l(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"m_L must be a number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"m_L must be finite and >= 0, got {value}")
    return result
