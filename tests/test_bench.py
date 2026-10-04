"""Benchmark harness tests: metrics, reporting, adapters and a full worker run on tiny data."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from hnsw_engine import datasets, exact_search
from hnsw_engine.bench import report
from hnsw_engine.bench.adapters import make_adapter
from hnsw_engine.bench.env import capture_environment, single_thread_environment
from hnsw_engine.bench.metrics import latency_summary, median_of_repetitions, recall_at_k
from hnsw_engine.bench.worker import run
from hnsw_engine.datasets import DATASETS, DatasetInfo
from hnsw_engine.synthetic import gaussian

# -------------------------------------------------------------------- metrics


def test_recall_counts_tied_alternatives_as_correct() -> None:
    true_ids = np.array([[0, 1, 2]])
    true_dist = np.array([[1.0, 2.0, 3.0]])
    # id 9 is tied with the 3rd neighbour (distance 3.0): correct under the tie-aware definition.
    found = np.array([[0, 1, 9]])
    found_dist = np.array([[1.0, 2.0, 3.0]])
    out = recall_at_k(found, found_dist, true_ids, true_dist, k=3)
    assert out == {"recall": 1.0, "recall_ids": pytest.approx(2 / 3)}


def test_recall_ignores_missing_and_repeated_ids() -> None:
    true_ids = np.array([[0, 1, 2], [0, 1, 2]])
    true_dist = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    found = np.array([[0, 0, 0], [0, -1, 7]])
    found_dist = np.array([[1.0, 1.0, 1.0], [1.0, np.inf, 5.0]])
    out = recall_at_k(found, found_dist, true_ids, true_dist, k=3)
    assert out["recall"] == pytest.approx(1 / 3)
    assert out["recall_ids"] == pytest.approx(1 / 3)


def test_recall_uses_only_the_first_k_columns() -> None:
    true_ids = np.array([[0, 1, 2, 3]])
    true_dist = np.array([[1.0, 2.0, 3.0, 4.0]])
    found = np.array([[0, 5, 1, 2]])
    found_dist = np.array([[1.0, 9.0, 2.0, 3.0]])
    assert recall_at_k(found, found_dist, true_ids, true_dist, k=2)["recall"] == 0.5


def test_latency_summary_and_median() -> None:
    summary = latency_summary(np.array([1_000_000, 2_000_000, 3_000_000], dtype=np.int64))
    assert summary["mean_ms"] == pytest.approx(2.0)
    assert summary["p50_ms"] == pytest.approx(2.0)
    assert summary["qps"] == pytest.approx(500.0)
    med = median_of_repetitions([{"qps": 1.0}, {"qps": 9.0}, {"qps": 3.0}])
    assert med == {"qps": 3.0}


# ---------------------------------------------------------------- environment


def test_environment_capture_is_json_serialisable() -> None:
    env = capture_environment()
    json.dumps(env)
    assert env["python"]
    assert "numpy" in env["packages"]
    assert set(single_thread_environment().values()) == {"1"}


# ----------------------------------------------------------------- reporting


def test_markdown_table_formatting() -> None:
    rows = [{"index": "x", "recall": 0.95, "qps": 1234.5, "mean_ms": 0.81, "flag": None}]
    table = report.markdown_table(
        rows, {"index": "Index", "recall": "R", "qps": "QPS", "mean_ms": "ms", "flag": "F"}
    )
    assert table.splitlines() == [
        "| Index | R | QPS | ms | F |",
        "|---|---:|---:|---:|---:|",
        "| x | 0.9500 | 1,234 | 0.8100 | - |",
    ]


def test_inject_replaces_only_the_marked_section(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("intro\n<!-- table:t -->\nold\n<!-- /table:t -->\noutro\n")
    report.inject(doc, "t", "new\ncontent")
    assert doc.read_text() == "intro\n<!-- table:t -->\nnew\ncontent\n<!-- /table:t -->\noutro\n"
    with pytest.raises(ValueError, match="markers"):
        report.inject(doc, "missing", "x")


# ------------------------------------------------------------------ adapters


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("hnsw-engine-reference", {"M": 8, "ef_construction": 40}),
        ("faiss-hnsw", {"M": 8, "ef_construction": 40}),
        ("hnswlib", {"M": 8, "ef_construction": 40}),
        ("faiss-flat", {}),
        ("faiss-ivf", {"nlist": 8}),
    ],
)
@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_adapters_find_true_neighbours(name: str, params: dict, metric: str) -> None:
    data, queries = gaussian(500, 8, seed=1), gaussian(20, 8, seed=2)
    adapter = make_adapter(name, metric, params)
    adapter.warm_up(8)
    adapter.build(data)
    if adapter.query_param == "ef_search":
        adapter.set_query_param(100)
    elif adapter.query_param == "nprobe":
        adapter.set_query_param(8)
    truth, _ = exact_search(data, queries, 5, metric)
    hits = [
        len(set(adapter.query(q, 5).tolist()) & set(t.tolist()))
        for q, t in zip(queries, truth, strict=True)
    ]
    assert np.mean(hits) / 5 >= 0.9
    assert adapter.logical_bytes() is None or adapter.logical_bytes() > 0
    assert adapter.describe()["name"] == adapter.name


def test_unknown_adapter() -> None:
    with pytest.raises(ValueError, match="unknown index"):
        make_adapter("annoy", "l2", {})


# --------------------------------------------------------------- worker run


def test_worker_produces_complete_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 6)).astype(np.float32)
    test = rng.normal(size=(30, 6)).astype(np.float32)
    path = tmp_path / "tiny-6-euclidean.hdf5"
    ids, dist = exact_search(train, test, 10)
    with h5py.File(path, "w") as handle:
        handle["train"], handle["test"] = train, test
        handle["neighbors"], handle["distances"] = ids.astype(np.int32), np.sqrt(dist)
    info = DatasetInfo("tiny-6-euclidean", "unused", "l2", 6, 400, 30, 0, None)
    monkeypatch.setitem(DATASETS, info.name, info)
    datasets.load(info.name, tmp_path, n_train=300, n_test=30, k=10)  # parent caches truth

    config = {
        "dataset": info.name,
        "data_dir": str(tmp_path),
        "n_train": 300,
        "n_test": 30,
        "k": 10,
        "k_ground_truth": 10,
        "index": "hnsw-engine-reference",
        "params": {"M": 8, "ef_construction": 50},
        "sweep": [10, 50],
        "repetitions": 2,
        "warmup_queries": 5,
        "time_budget_seconds": 600,
        "out": str(tmp_path / "out.json"),
    }
    record = run(config)
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["dataset"]["n_train"] == 300
    assert [p["value"] for p in record["points"]] == [10, 50]
    for point in record["points"]:
        assert 0.0 <= point["recall"] <= 1.0
        assert point["qps"] > 0
        assert len(point["repetitions"]) == 2
        assert point["deterministic_across_repetitions"]
        assert point["distance_computations_per_query"] > 0
    assert record["points"][1]["recall"] >= record["points"][0]["recall"]
    assert record["build_seconds"] > 0
    assert not record["truncated_by_time_budget"]
    assert record["environment"]["packages"]["numpy"]
    assert 0.0 <= record["suspended_seconds"] < 1.0


def test_runner_retries_a_run_during_which_the_machine_slept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hnsw_engine.bench import runner

    train = gaussian(200, 4, seed=0)
    test = gaussian(10, 4, seed=1)
    path = tmp_path / "tiny-4-euclidean.hdf5"
    with h5py.File(path, "w") as handle:
        handle["train"], handle["test"] = train, test
        handle["neighbors"] = np.zeros((10, 10), dtype=np.int32)
        handle["distances"] = np.zeros((10, 10), dtype=np.float32)
    info = DatasetInfo("tiny-4-euclidean", "unused", "l2", 4, 200, 10, 0, None)
    monkeypatch.setitem(DATASETS, info.name, info)
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        config = json.loads(Path(command[-1]).read_text())
        calls.append(config["index"])
        suspended = 900.0 if len(calls) == 1 else 0.0  # the first attempt "slept"
        Path(config["out"]).write_text(json.dumps({"suspended_seconds": suspended}))

        class Done:
            returncode = 0
            stderr = ""

        return Done()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    suite = runner.Suite(
        info.name, tmp_path, 100, 10, 5, (runner.IndexConfig("faiss-flat"),), cooldown_seconds=0.0
    )
    run_dir = runner.run_suite(suite, tmp_path / "results", label="t")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [r["status"] for r in manifest["runs"]] == ["suspended", "ok"]
    assert len(calls) == 2


def test_page_cache_residency_and_eviction(tmp_path: Path) -> None:
    from hnsw_engine.bench import pagecache

    path = tmp_path / "blob.bin"
    path.write_bytes(os.urandom(8 * 1024 * 1024))
    pagecache.warm(path)
    assert pagecache.resident_fraction(path) > 0.9
    pagecache.evict(path)
    assert pagecache.resident_fraction(path) < 0.1
    assert path.stat().st_size == 8 * 1024 * 1024  # evicting never changes the file
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    pagecache.evict(empty)
    assert pagecache.resident_fraction(empty) == 1.0
