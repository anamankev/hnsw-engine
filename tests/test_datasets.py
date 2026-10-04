"""Dataset loader tests on tiny synthetic files in the ANN-Benchmarks layout (no network)."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from hnsw_engine import datasets, exact_search
from hnsw_engine.datasets import DATASETS, DatasetInfo, load, sha256_file, verify


def _write_ann_file(path: Path, train: np.ndarray, test: np.ndarray, k: int = 10) -> None:
    ids, dist = exact_search(train, test, k)
    with h5py.File(path, "w") as handle:
        handle.attrs["distance"] = "euclidean"
        handle["train"] = train
        handle["test"] = test
        handle["neighbors"] = ids.astype(np.int32)
        handle["distances"] = np.sqrt(dist).astype(np.float32)


@pytest.fixture
def tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rng = np.random.default_rng(0)
    path = tmp_path / "tiny-8-euclidean.hdf5"
    _write_ann_file(
        path,
        rng.normal(size=(300, 8)).astype(np.float32),
        rng.normal(size=(40, 8)).astype(np.float32),
    )
    info = DatasetInfo("tiny-8-euclidean", "unused", "l2", 8, 300, 40, path.stat().st_size, None)
    monkeypatch.setitem(DATASETS, info.name, info)
    return path


def test_registry_contains_the_mission_datasets() -> None:
    assert DATASETS["sift-128-euclidean"].metric == "l2"
    assert DATASETS["fashion-mnist-784-euclidean"].dim == 784
    assert DATASETS["glove-100-angular"].metric == "cosine"
    with pytest.raises(ValueError, match="unknown dataset"):
        datasets.get_info("mnist")


def test_verify_accepts_valid_file(tiny: Path) -> None:
    summary = verify(tiny)
    assert summary["problems"] == []
    assert summary["train"]["shape"] == [300, 8]
    assert summary["neighbors"]["dtype"] == "int32"
    assert summary["train"]["zero_rows"] == 0


def test_verify_reports_nan_and_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as handle:
        handle["train"] = np.array([[1.0, np.nan]], dtype=np.float32)
        handle["test"] = np.zeros((1, 2), dtype=np.float32)
    problems = verify(path)["problems"]
    assert any("non-finite" in p for p in problems)
    assert any("missing dataset 'neighbors'" in p for p in problems)


def test_full_load_ground_truth_matches_provided(tiny: Path) -> None:
    data = load("tiny-8-euclidean", tiny.parent, n_test=40, k=10, path=tiny)
    with h5py.File(tiny, "r") as handle:
        np.testing.assert_array_equal(data.neighbors, handle["neighbors"][:].astype(np.int64))
    assert data.train.dtype == np.float32
    assert data.train.shape == (300, 8)


def test_subset_ground_truth_matches_brute_force_and_is_cached(
    tiny: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = load("tiny-8-euclidean", tiny.parent, n_train=120, n_test=25, k=7, path=tiny)
    ids, dist = exact_search(data.train, data.test, 7)
    np.testing.assert_array_equal(data.neighbors, ids)
    np.testing.assert_array_equal(data.distances, dist)
    assert data.train.shape == (120, 8)
    assert list((tiny.parent / "ground_truth").glob("tiny-8-euclidean-n120-q25-k7-*.npz"))

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("ground truth should come from the cache")

    monkeypatch.setattr(datasets, "exact_search", fail)
    again = load("tiny-8-euclidean", tiny.parent, n_train=120, n_test=25, k=7, path=tiny)
    np.testing.assert_array_equal(again.neighbors, ids)


def test_subset_ground_truth_differs_from_full_ground_truth(tiny: Path) -> None:
    # The reason subsets need their own ground truth: neighbours outside the subset vanish.
    full = load("tiny-8-euclidean", tiny.parent, n_test=40, k=10, path=tiny)
    subset = load("tiny-8-euclidean", tiny.parent, n_train=100, n_test=40, k=10, path=tiny)
    assert (full.neighbors >= 100).any()
    assert (subset.neighbors < 100).all()


def test_load_rejects_oversized_requests(tiny: Path) -> None:
    with pytest.raises(ValueError, match="fewer"):
        load("tiny-8-euclidean", tiny.parent, n_train=301, path=tiny)
    with pytest.raises(ValueError, match="fewer"):
        load("tiny-8-euclidean", tiny.parent, n_test=41, path=tiny)


def test_download_from_file_url_and_checksum(
    tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "downloads"
    path = datasets.download("tiny-8-euclidean", target_dir, url=tiny.as_uri())
    assert path.read_bytes() == tiny.read_bytes()
    assert sha256_file(path) == sha256_file(tiny)
    assert not list(target_dir.glob("*.part"))
    bad = DatasetInfo("tiny-8-euclidean", "unused", "l2", 8, 300, 40, 0, "0" * 64)
    monkeypatch.setitem(DATASETS, bad.name, bad)
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets.download("tiny-8-euclidean", target_dir, url=tiny.as_uri())


def test_sha256_sidecar_is_invalidated_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    first = sha256_file(path)
    assert first == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    path.write_bytes(b"abcd")
    assert sha256_file(path) != first
