"""ANN-Benchmarks datasets: download, verification, subsets and cached exact ground truth.

Files use the ANN-Benchmarks HDF5 layout (https://github.com/erikbern/ann-benchmarks): ``train``
(vectors to index), ``test`` (queries), ``neighbors`` and ``distances`` (the true 100 nearest
training vectors of each query). That ground truth is only valid for the *full* training set,
so this module computes exact ground truth with :func:`hnsw_engine.oracle.exact_search` for every
configuration (full or subset) and caches it under ``data/ground_truth/``. For full datasets the
provided ground truth is used as an independent cross-check (see ``benchmarks/verify_datasets.py``).

Command line::

    python -m hnsw_engine.datasets download sift-128-euclidean --data-dir data
    python -m hnsw_engine.datasets verify sift-128-euclidean --data-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from ._validation import check_int
from .metrics import MetricName
from .oracle import exact_search

_CHUNK = 1 << 20


@dataclass(frozen=True)
class DatasetInfo:
    """Registry entry for a downloadable dataset.

    ``sha256`` is the checksum observed on first download (2026-10-02); ANN-Benchmarks does not
    publish checksums, so this pins the exact file every result was produced from.
    """

    name: str
    url: str
    metric: MetricName
    dim: int
    n_train: int
    n_test: int
    size_bytes: int
    sha256: str | None


DATASETS: dict[str, DatasetInfo] = {
    info.name: info
    for info in (
        DatasetInfo(
            name="sift-128-euclidean",
            url="https://ann-benchmarks.com/sift-128-euclidean.hdf5",
            metric="l2",
            dim=128,
            n_train=1_000_000,
            n_test=10_000,
            size_bytes=525_128_288,
            sha256="dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984",
        ),
        DatasetInfo(
            name="fashion-mnist-784-euclidean",
            url="https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
            metric="l2",
            dim=784,
            n_train=60_000,
            n_test=10_000,
            size_bytes=227_528_288,
            sha256="7f24e5122b71346346686a6ea80e08d9145b9c020f9f2db4c4c799dfcfc96a07",
        ),
        DatasetInfo(
            name="glove-100-angular",
            url="https://ann-benchmarks.com/glove-100-angular.hdf5",
            metric="cosine",
            dim=100,
            n_train=1_183_514,
            n_test=10_000,
            size_bytes=485_413_888,
            sha256="544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4",
        ),
    )
}


@dataclass(frozen=True)
class Dataset:
    """A loaded benchmark dataset with exact ground truth for exactly these vectors."""

    name: str
    metric: MetricName
    train: NDArray[np.float32]
    test: NDArray[np.float32]
    neighbors: NDArray[np.int64]
    distances: NDArray[np.float64]
    source_sha256: str


def get_info(name: str) -> DatasetInfo:
    """Registry lookup with a helpful error."""
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}") from None


def dataset_path(name: str, data_dir: str | Path) -> Path:
    """Where the HDF5 file for ``name`` lives inside ``data_dir``."""
    return Path(data_dir) / f"{get_info(name).name}.hdf5"


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, cached in a ``.sha256`` sidecar keyed by size and modification time."""
    path = Path(path)
    stat = path.stat()
    sidecar = path.with_name(path.name + ".sha256")
    stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    if sidecar.exists():
        cached = sidecar.read_text().split()
        if len(cached) == 2 and cached[1] == stamp:
            return cached[0]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    value = digest.hexdigest()
    sidecar.write_text(f"{value} {stamp}\n")
    return value


def download(name: str, data_dir: str | Path, *, url: str | None = None) -> Path:
    """Download ``name`` into ``data_dir`` unless already present; verify its checksum.

    The file is streamed to ``<name>.hdf5.part`` and renamed only after the download completes,
    so an interrupted download never looks like a valid file. ``url`` overrides the registry URL
    (tests use ``file://`` URLs).
    """
    info = get_info(name)
    target = dataset_path(name, data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url or info.url) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
        os.replace(partial, target)
    actual = sha256_file(target)
    if info.sha256 is not None and actual != info.sha256:
        raise ValueError(f"checksum mismatch for {target}: {actual} != {info.sha256}")
    return target


def verify(path: str | Path) -> dict[str, Any]:
    """Inspect an ANN-Benchmarks file: keys, shapes, dtypes, non-finite values and zero rows.

    Returns a JSON-serialisable summary whose ``problems`` list is empty for a valid file.
    """
    summary: dict[str, Any] = {"path": str(path), "problems": []}
    problems: list[str] = summary["problems"]
    with h5py.File(path, "r") as handle:
        summary["attrs"] = {k: _jsonable(v) for k, v in handle.attrs.items()}
        for key in ("train", "test", "neighbors", "distances"):
            if key not in handle:
                problems.append(f"missing dataset '{key}'")
                continue
            values = handle[key][:]
            entry: dict[str, Any] = {"shape": list(values.shape), "dtype": str(values.dtype)}
            if np.issubdtype(values.dtype, np.floating):
                entry["non_finite"] = int((~np.isfinite(values)).sum())
                if entry["non_finite"]:
                    problems.append(f"'{key}' contains {entry['non_finite']} non-finite values")
            if key in ("train", "test"):
                entry["zero_rows"] = int((~values.any(axis=1)).sum())
            summary[key] = entry
    if not problems:
        dim = summary["train"]["shape"][1]
        if summary["test"]["shape"][1] != dim:
            problems.append("train and test dimensions differ")
        if summary["neighbors"]["shape"][0] != summary["test"]["shape"][0]:
            problems.append("neighbors rows do not match test rows")
        if summary["neighbors"]["shape"] != summary["distances"]["shape"]:
            problems.append("neighbors and distances shapes differ")
    return summary


def load(
    name: str,
    data_dir: str | Path,
    *,
    n_train: int | None = None,
    n_test: int = 1000,
    k: int = 100,
    path: str | Path | None = None,
) -> Dataset:
    """Load the first ``n_train`` training vectors and ``n_test`` queries with exact ground truth.

    Args:
        name: registry name, e.g. ``"sift-128-euclidean"``.
        data_dir: folder holding the HDF5 file and the ground-truth cache.
        n_train: number of training vectors (prefix); ``None`` for all of them.
        n_test: number of queries (prefix of ``test``).
        k: ground-truth depth.
        path: explicit HDF5 path (defaults to ``dataset_path(name, data_dir)``).
    """
    info = get_info(name)
    file_path = Path(path) if path is not None else dataset_path(name, data_dir)
    n_test = check_int(n_test, "n_test", minimum=1)
    k = check_int(k, "k", minimum=1)
    with h5py.File(file_path, "r") as handle:
        total = handle["train"].shape[0]
        n = total if n_train is None else check_int(n_train, "n_train", minimum=1)
        if n > total or n_test > handle["test"].shape[0]:
            raise ValueError(f"requested {n} train / {n_test} test vectors; file has fewer")
        train = np.ascontiguousarray(handle["train"][:n], dtype=np.float32)
        test = np.ascontiguousarray(handle["test"][:n_test], dtype=np.float32)
    source_sha = sha256_file(file_path)
    ids, dists = ground_truth(
        train,
        test,
        k,
        info.metric,
        cache_dir=Path(data_dir) / "ground_truth",
        cache_key=f"{name}-n{n}-q{n_test}-k{k}-{source_sha[:12]}",
    )
    return Dataset(name, info.metric, train, test, ids, dists, source_sha)


def ground_truth(
    train: NDArray[np.float32],
    test: NDArray[np.float32],
    k: int,
    metric: MetricName,
    *,
    cache_dir: Path,
    cache_key: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Exact ground truth from the oracle, cached as ``<cache_dir>/<cache_key>.npz``."""
    cache = cache_dir / f"{cache_key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return data["ids"].astype(np.int64), data["distances"].astype(np.float64)
    ids, dists = exact_search(train, test, min(k, train.shape[0]), metric)
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(cache.name + ".part")
    with open(partial, "wb") as handle:
        np.savez(handle, ids=ids, distances=dists)
    os.replace(partial, cache)
    return ids, dists


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: ``download`` or ``verify`` a registered dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "verify"])
    parser.add_argument("name", choices=sorted(DATASETS))
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)
    path = download(args.name, args.data_dir)
    if args.command == "verify":
        print(json.dumps(verify(path), indent=2))
    else:
        print(f"{path} sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
