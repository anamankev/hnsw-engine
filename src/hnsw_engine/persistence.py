"""Engine-independent graph snapshot and the on-disk index format.

A :class:`GraphState` is a plain-array snapshot of an index: vectors, levels, entry point and,
for each layer, the links of every node in compressed sparse row (CSR) form. Both engines can
export and import it, which gives three things for free: persistence, engine-agnostic invariant
checks, and an exact way to compare the graphs two engines built.

Files are written with ``numpy.savez`` without pickle, so loading cannot execute code embedded
in the file, and every load validates the graph (:func:`validate_state`), because the compiled
engine does not bounds-check and an out-of-range link in a damaged file could otherwise make a
search access memory outside its arrays.

The native format (:func:`save_native`, :func:`load_native`) instead stores the compiled engine's
own arrays as separate ``.npy`` files next to a JSON header, so they can be memory-mapped and
used in place: opening an index then costs almost nothing, whatever its size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FORMAT_VERSION = 1
NATIVE_FORMAT = "hnsw-engine-native"
NATIVE_FORMAT_VERSION = 1
NATIVE_HEADER = "header.json"


@dataclass(frozen=True)
class GraphState:
    """Snapshot of an HNSW graph.

    Nodes present on layer ``l`` are those with ``levels >= l``, in ascending id order; the
    links of the ``i``-th such node are ``indices[l][indptr[l][i]:indptr[l][i + 1]]``, in the
    order the engine stores them.
    """

    vectors: NDArray[np.float32]
    levels: NDArray[np.int64]
    entry_point: int
    max_level: int
    indptr: tuple[NDArray[np.int64], ...]
    indices: tuple[NDArray[np.int64], ...]

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return int(self.levels.shape[0])

    @property
    def num_layers(self) -> int:
        """Number of layers (``max_level + 1``; zero for an empty graph)."""
        return len(self.indptr)

    def layer_nodes(self, layer: int) -> NDArray[np.int64]:
        """Ids of the nodes present on ``layer``, ascending."""
        return np.flatnonzero(self.levels >= layer).astype(np.int64)

    def adjacency(self, layer: int) -> dict[int, list[int]]:
        """Links of every node on ``layer`` as a ``{node: [neighbours]}`` dictionary."""
        nodes = self.layer_nodes(layer)
        ptr, idx = self.indptr[layer], self.indices[layer]
        return {int(v): idx[ptr[i] : ptr[i + 1]].tolist() for i, v in enumerate(nodes)}

    @classmethod
    def from_adjacency(
        cls,
        vectors: NDArray[np.float32],
        levels: NDArray[np.int64],
        entry_point: int,
        max_level: int,
        layers: list[dict[int, list[int]]],
    ) -> GraphState:
        """Build a snapshot from per-layer ``{node: [neighbours]}`` dictionaries."""
        indptr: list[NDArray[np.int64]] = []
        indices: list[NDArray[np.int64]] = []
        for layer, links_of in enumerate(layers):
            nodes = np.flatnonzero(levels >= layer)
            counts = np.fromiter((len(links_of[int(v)]) for v in nodes), np.int64, nodes.size)
            ptr = np.zeros(nodes.size + 1, dtype=np.int64)
            np.cumsum(counts, out=ptr[1:])
            flat = [e for v in nodes for e in links_of[int(v)]]
            indptr.append(ptr)
            indices.append(np.asarray(flat, dtype=np.int64))
        return cls(vectors, levels, entry_point, max_level, tuple(indptr), tuple(indices))


def diff_states(a: GraphState, b: GraphState) -> list[str]:
    """Human-readable differences between two snapshots (empty list if identical)."""
    problems: list[str] = []
    if a.vectors.shape != b.vectors.shape or not np.array_equal(a.vectors, b.vectors):
        problems.append("vectors differ")
    if not np.array_equal(a.levels, b.levels):
        problems.append("levels differ")
    if (a.entry_point, a.max_level) != (b.entry_point, b.max_level):
        problems.append(
            f"entry point/max level differ: {(a.entry_point, a.max_level)} vs "
            f"{(b.entry_point, b.max_level)}"
        )
    if a.num_layers != b.num_layers:
        problems.append(f"layer count differs: {a.num_layers} vs {b.num_layers}")
        return problems
    for layer in range(a.num_layers):
        if np.array_equal(a.indptr[layer], b.indptr[layer]) and np.array_equal(
            a.indices[layer], b.indices[layer]
        ):
            continue
        adj_a, adj_b = a.adjacency(layer), b.adjacency(layer)
        bad = [v for v in adj_a if adj_a[v] != adj_b.get(v)]
        problems.append(f"layer {layer}: links differ for {len(bad)} nodes, first {bad[:5]}")
    return problems


def validate_state(state: GraphState) -> None:
    """Raise ``ValueError`` unless ``state`` is internally consistent and every id is in range.

    The compiled engine does not bounds-check array accesses, so a damaged or hostile file with an
    out-of-range link could make a search read or write outside its arrays. Every load runs this
    check (vectorised, so cheap next to the load itself).
    """
    n = state.size
    if state.vectors.ndim != 2 or state.vectors.shape[0] != n:
        raise ValueError("vectors and levels disagree on the number of nodes")
    if n == 0:
        if state.entry_point != -1 or state.max_level != -1 or state.num_layers != 0:
            raise ValueError("an empty graph has no entry point and no layers")
        return
    if (state.levels < 0).any():
        raise ValueError("negative node level")
    top = int(state.levels.max())
    if state.max_level != top or state.num_layers != top + 1 or len(state.indices) != top + 1:
        raise ValueError("the number of layers does not match the node levels")
    if not 0 <= state.entry_point < n or int(state.levels[state.entry_point]) != top:
        raise ValueError("the entry point is not a node of the top layer")
    for layer in range(top + 1):
        count = int((state.levels >= layer).sum())
        ptr, idx = state.indptr[layer], state.indices[layer]
        if (
            ptr.shape != (count + 1,)
            or ptr[0] != 0
            or ptr[-1] != idx.shape[0]
            or (np.diff(ptr) < 0).any()
        ):
            raise ValueError(f"layer {layer}: malformed link offsets")
        if idx.size and ((idx < 0).any() or (idx >= n).any() or (state.levels[idx] < layer).any()):
            raise ValueError(f"layer {layer}: a link points to a node outside the layer")


def save_state(path: str | Path, meta: dict[str, Any], state: GraphState) -> None:
    """Write ``meta`` (JSON-serialisable) and ``state`` to ``path`` exactly (no suffix added)."""
    arrays: dict[str, NDArray[Any]] = {
        "meta": np.array(json.dumps({"format_version": FORMAT_VERSION, **meta})),
        "vectors": state.vectors,
        "levels": state.levels,
        "entry": np.array([state.entry_point, state.max_level], dtype=np.int64),
    }
    for layer in range(state.num_layers):
        arrays[f"indptr_{layer}"] = state.indptr[layer]
        arrays[f"indices_{layer}"] = state.indices[layer]
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)  # type: ignore[arg-type]


def load_state(path: str | Path) -> tuple[dict[str, Any], GraphState]:
    """Read a file written by :func:`save_state`; returns ``(meta, state)``."""
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data.files:
            raise ValueError(f"{path} is not an hnsw-engine index file")
        meta: dict[str, Any] = json.loads(str(data["meta"]))
        version = meta.pop("format_version", None)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported index format version {version!r}")
        entry_point, max_level = (int(v) for v in data["entry"])
        num_layers = max_level + 1
        state = GraphState(
            vectors=np.ascontiguousarray(data["vectors"], dtype=np.float32),
            levels=data["levels"].astype(np.int64),
            entry_point=entry_point,
            max_level=max_level,
            indptr=tuple(data[f"indptr_{i}"].astype(np.int64) for i in range(num_layers)),
            indices=tuple(data[f"indices_{i}"].astype(np.int64) for i in range(num_layers)),
        )
    validate_state(state)
    return meta, state


def save_native(
    directory: str | Path, meta: dict[str, Any], arrays: dict[str, NDArray[Any]]
) -> None:
    """Write ``arrays`` as ``<name>.npy`` files and ``meta`` as ``header.json`` in ``directory``.

    The header records each array's dtype and shape, so :func:`load_native` can reject files that
    do not belong together. The header is written last: a directory without one is incomplete.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / NATIVE_HEADER).unlink(missing_ok=True)
    layout = {}
    for name, array in arrays.items():
        np.save(root / f"{name}.npy", np.ascontiguousarray(array), allow_pickle=False)
        layout[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
    header = {
        "format": NATIVE_FORMAT,
        "format_version": NATIVE_FORMAT_VERSION,
        "arrays": layout,
        **meta,
    }
    (root / NATIVE_HEADER).write_text(json.dumps(header, indent=2))


def load_native(
    directory: str | Path, *, mmap: bool = True
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Read a directory written by :func:`save_native`; returns ``(meta, arrays)``.

    With ``mmap`` the arrays are read-only memory maps of the files (nothing is read until it is
    used); otherwise they are loaded into memory.
    """
    root = Path(directory)
    header_path = root / NATIVE_HEADER
    if not header_path.is_file():
        raise ValueError(f"{root} is not an hnsw-engine native index (no {NATIVE_HEADER})")
    meta: dict[str, Any] = json.loads(header_path.read_text())
    if meta.pop("format", None) != NATIVE_FORMAT:
        raise ValueError(f"{root} is not an hnsw-engine native index")
    version = meta.pop("format_version", None)
    if version != NATIVE_FORMAT_VERSION:
        raise ValueError(f"unsupported native format version {version!r}")
    arrays: dict[str, NDArray[Any]] = {}
    for name, expected in meta.pop("arrays").items():
        array = np.load(root / f"{name}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
        if str(array.dtype) != expected["dtype"] or list(array.shape) != expected["shape"]:
            raise ValueError(
                f"{name}.npy has dtype {array.dtype} and shape {array.shape}, header says "
                f"{expected['dtype']} and {tuple(expected['shape'])}"
            )
        arrays[name] = np.asarray(array)  # plain ndarray view: Numba does not need the subclass
    return meta, arrays
