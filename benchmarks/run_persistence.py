"""Tier 3: opening a saved index, ``.npz`` against memory-mapped native files (DESIGN.md 9.3).

Builds an index of SIFT vectors (all 1M by default; M = 16, efConstruction = 200, seed 0), saves
it in both formats, then opens it in fresh processes, three times per mode:

* ``npz``: ``HNSWIndex.load`` (reads the archive and rebuilds the engine arrays);
* ``native-memory``: ``HNSWIndex.load_native(mmap=False)`` (reads the arrays into memory);
* ``native-mmap``: ``HNSWIndex.load_native(mmap=True)`` (maps the files; validating the links
  reads the link arrays once);
* ``native-mmap-trusted``: the same with ``validate=False`` (nothing is read until searched).

Each process first imports the libraries and runs one search on a tiny index with the same kind
of storage (in memory or memory-mapped), so that neither the import of Numba nor the first call
into its compiled code is counted. It then reports the time to open the index, its resident
memory after opening and after searching, and the time of two consecutive batches of 1,000
queries (efSearch 80, one thread). The first batch after a memory-mapped open includes the page
faults that bring the touched pages in. By default the files were just written, so they sit in
the operating system's page cache (a warm cache). With ``--cache both`` every repetition measures
each mode twice: once after reading the index files into the cache, once after evicting them
from it (``hnsw_engine.bench.pagecache``: no administrator rights needed, residency checked with
``mincore``), recording the share of pages resident at the start and the pages read from disk
during the run (macOS ``vm_stat``, system-wide). Index files are deleted at the end.

Writes ``results/studies/persistence.json`` (``--cache both``: ``persistence_cache.json``);
``--inject`` writes the summary table.

Usage: ``python benchmarks/run_persistence.py [--n-train N] [--cache both] [--inject FILE]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hnsw_engine import HNSWIndex, exact_distances
from hnsw_engine.bench import pagecache
from hnsw_engine.bench.env import capture_environment, ensure_single_thread, machine_activity
from hnsw_engine.bench.metrics import recall_at_k
from hnsw_engine.bench.report import inject
from hnsw_engine.datasets import load

MODES = ("npz", "native-memory", "native-mmap", "native-mmap-trusted")
EF = 80
K = 10
REPETITIONS = 3
WORK = Path("results/persistence")
OUT = Path("results/studies/persistence.json")
OUT_CACHE = Path("results/studies/persistence_cache.json")
CACHE_STATES = ("warm", "cold")


def rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def open_index(mode: str) -> HNSWIndex:
    if mode == "npz":
        return HNSWIndex.load(WORK / "index.npz")
    return HNSWIndex.load_native(
        WORK / "native", mmap=mode.startswith("native-mmap"), validate=not mode.endswith("trusted")
    )


def warm_up(mode: str, queries: np.ndarray) -> None:
    """Load Numba and the compiled search for this kind of storage on a tiny index."""
    tiny = HNSWIndex(queries.shape[1], M=16, ef_construction=32, seed=0)
    tiny.add(np.random.default_rng(0).random((256, queries.shape[1]), dtype=np.float32))
    if mode.startswith("native-mmap"):  # read-only arrays: Numba uses another specialisation
        path = WORK / f"warmup-{os.getpid()}"
        tiny.save_native(path)
        tiny = HNSWIndex.load_native(path, mmap=True)
    tiny.batch_search(queries[:2], K, ef=EF)


def worker(mode: str, warmup: bool) -> None:
    """Runs in a fresh process: open, search twice, report one JSON line on stdout."""
    queries = np.load(WORK / "queries.npy")
    if warmup:
        open_index(mode).batch_search(queries[:5], K, ef=EF)
        return
    warm_up(mode, queries)
    baseline = rss_bytes()
    start = time.perf_counter()
    index = open_index(mode)
    open_seconds = time.perf_counter() - start
    after_open = rss_bytes()
    timings = []
    for _ in range(2):
        start = time.perf_counter()
        ids, _ = index.batch_search(queries, K, ef=EF)
        timings.append(time.perf_counter() - start)
    np.save(WORK / f"ids-{mode}.npy", ids)
    record = {
        "mode": mode,
        "activity": machine_activity(),
        "open_seconds": open_seconds,
        "rss_baseline_bytes": baseline,
        "rss_after_open_bytes": after_open,
        "rss_after_queries_bytes": rss_bytes(),
        "first_batch_seconds": timings[0],
        "second_batch_seconds": timings[1],
        "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
    }
    print(json.dumps(record))


def spawn(mode: str, warmup: bool = False) -> dict[str, Any] | None:
    args = [sys.executable, __file__, "--worker", mode] + (["--warmup"] if warmup else [])
    done = subprocess.run(args, capture_output=True, text=True, check=True, env=os.environ)
    if warmup:
        return None
    return dict(json.loads(done.stdout.strip().splitlines()[-1]))


def index_files(mode: str) -> list[Path]:
    """The files a mode opens."""
    return [WORK / "index.npz"] if mode == "npz" else sorted((WORK / "native").iterdir())


def set_cache(mode: str, state: str) -> float:
    """Warm or evict the mode's index files; returns the resident share of their bytes."""
    files = index_files(mode)
    for path in files:
        pagecache.warm(path) if state == "warm" else pagecache.evict(path)
    pagecache.warm(WORK / "queries.npy")  # the queries are always in memory
    sizes = [path.stat().st_size for path in files]
    resident = sum(pagecache.resident_fraction(p) * n for p, n in zip(files, sizes, strict=True))
    return resident / sum(sizes)


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n-train", type=int, default=None, help="default: all vectors")
    parser.add_argument("--inject", type=Path, help="write the summary table into this file")
    parser.add_argument("--report-only", action="store_true", help="use the saved JSON")
    parser.add_argument(
        "--cache",
        choices=("warm", "both"),
        default="warm",
        help="'both' also measures each mode after evicting the index files from the cache",
    )
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.warmup)
        return
    if not args.report_only:
        ensure_single_thread()
        run(args.n_train, args.cache)
    if args.cache == "both":
        markdown, marker = cache_table(json.loads(OUT_CACHE.read_text())), "persistence_cache"
    else:
        markdown, marker = table(json.loads(OUT.read_text())), "persistence"
    print(markdown)
    if args.inject:
        inject(args.inject, marker, markdown)


def run(n_train: int | None, cache: str = "warm") -> None:
    data = load("sift-128-euclidean", "data", n_train=n_train, n_test=1000, k=K)
    train, queries = data.train, data.test
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / "queries.npy", queries)
    index = HNSWIndex(train.shape[1], M=16, ef_construction=200, seed=0)
    start = time.perf_counter()
    index.add(train)
    build_seconds = time.perf_counter() - start
    saves = {}
    start = time.perf_counter()
    index.save(WORK / "index.npz")
    saves["npz"] = time.perf_counter() - start
    shutil.rmtree(WORK / "native", ignore_errors=True)
    start = time.perf_counter()
    index.save_native(WORK / "native")
    saves["native"] = time.perf_counter() - start
    sizes = {"npz": size_bytes(WORK / "index.npz"), "native": size_bytes(WORK / "native")}
    del index
    for mode in MODES:
        spawn(mode, warmup=True)
    if cache == "both":
        compare_cache(train, queries, data, build_seconds, saves, sizes)
        shutil.rmtree(WORK)
        return
    runs: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for _ in range(REPETITIONS):
        for mode in MODES:
            record = spawn(mode)
            assert record is not None
            runs[mode].append(record)
            print(mode, {k: v for k, v in record.items() if k != "ids_sha256"}, flush=True)
    digests = {r["ids_sha256"] for rs in runs.values() for r in rs}
    ids = np.load(WORK / "ids-native-mmap.npy")
    exact = exact_distances(train, queries, ids, "l2")
    recall = recall_at_k(ids, exact, data.neighbors, data.distances, K)["recall"]
    summary = {
        mode: {
            key: statistics.median(r[key] for r in rs)
            for key in (
                "open_seconds",
                "rss_after_open_bytes",
                "rss_after_queries_bytes",
                "rss_baseline_bytes",
                "first_batch_seconds",
                "second_batch_seconds",
            )
        }
        for mode, rs in runs.items()
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": "sift-128-euclidean",
                    "n_train": int(train.shape[0]),
                    "M": 16,
                    "ef_construction": 200,
                    "seed": 0,
                    "ef_search": EF,
                    "k": K,
                    "n_queries": int(queries.shape[0]),
                    "repetitions": REPETITIONS,
                    "page_cache": "warm (files just written; dropping the cache needs sudo)",
                },
                "build_seconds": build_seconds,
                "save_seconds": saves,
                "file_bytes": sizes,
                "identical_results_across_modes": len(digests) == 1,
                "recall_at_10": recall,
                "summary_median": summary,
                "runs": runs,
                "environment": capture_environment(),
            },
            indent=2,
        )
    )
    shutil.rmtree(WORK)  # the index files are large and fully described by the JSON


def compare_cache(
    train: np.ndarray,
    queries: np.ndarray,
    data: Any,
    build_seconds: float,
    saves: dict[str, float],
    sizes: dict[str, int],
) -> None:
    """Every repetition: each mode with a warm and with an evicted (cold) index."""
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        state: {mode: [] for mode in MODES} for state in CACHE_STATES
    }
    for _ in range(REPETITIONS):
        for mode in MODES:
            for state in CACHE_STATES:
                resident = set_cache(mode, state)
                before = pagecache.pageins()
                record = spawn(mode)
                after = pagecache.pageins()
                assert record is not None
                record["resident_fraction_at_start"] = resident
                record["pageins_during_run"] = (
                    None if before is None or after is None else after - before
                )
                runs[state][mode].append(record)
                print(state, mode, {k: v for k, v in record.items() if k != "ids_sha256"})
    digests = {r["ids_sha256"] for modes in runs.values() for rs in modes.values() for r in rs}
    ids = np.load(WORK / "ids-native-mmap.npy")
    exact = exact_distances(train, queries, ids, "l2")
    keys = (
        "open_seconds",
        "first_batch_seconds",
        "second_batch_seconds",
        "resident_fraction_at_start",
        "pageins_during_run",
    )
    summary = {
        state: {
            mode: {key: statistics.median(r[key] for r in rs) for key in keys}
            for mode, rs in modes.items()
        }
        for state, modes in runs.items()
    }
    OUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    OUT_CACHE.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": "sift-128-euclidean",
                    "n_train": int(train.shape[0]),
                    "M": 16,
                    "ef_construction": 200,
                    "seed": 0,
                    "ef_search": EF,
                    "k": K,
                    "n_queries": int(queries.shape[0]),
                    "repetitions": REPETITIONS,
                    "page_size_bytes": mmap.PAGESIZE,
                    "cache": "warm: index files read into the cache first; cold: evicted first",
                },
                "build_seconds": build_seconds,
                "save_seconds": saves,
                "file_bytes": sizes,
                "identical_results_across_runs": len(digests) == 1,
                "recall_at_10": recall_at_k(ids, exact, data.neighbors, data.distances, K)[
                    "recall"
                ],
                "summary_median": summary,
                "runs": runs,
                "environment": capture_environment(),
            },
            indent=2,
        )
    )


def cache_table(payload: dict[str, Any]) -> str:
    """Markdown summary: warm against cold, per mode (medians)."""
    labels = {
        "npz": "`load` (.npz archive)",
        "native-memory": "`load_native(mmap=False)`",
        "native-mmap": "`load_native(mmap=True)`",
        "native-mmap-trusted": "`load_native(mmap=True, validate=False)`",
    }
    page = payload["config"]["page_size_bytes"]
    header = [
        "Open with",
        "Open: warm / cold (s)",
        "First 1,000 queries: warm / cold (s)",
        "Next 1,000 queries: warm / cold (s)",
        "Read from disk during the run: warm / cold (MB)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * 4) + "|"]
    warm, cold = payload["summary_median"]["warm"], payload["summary_median"]["cold"]
    for mode in MODES:
        w, c = warm[mode], cold[mode]
        cells = [
            labels[mode],
            f"{w['open_seconds']:.3f} / {c['open_seconds']:.3f}",
            f"{w['first_batch_seconds']:.3f} / {c['first_batch_seconds']:.3f}",
            f"{w['second_batch_seconds']:.3f} / {c['second_batch_seconds']:.3f}",
            f"{w['pageins_during_run'] * page / 1e6:,.1f} / "
            f"{c['pageins_during_run'] * page / 1e6:,.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table(payload: dict[str, Any]) -> str:
    """Markdown summary of the median measurements per mode."""
    labels = {
        "npz": "`load` (.npz archive)",
        "native-memory": "`load_native(mmap=False)`",
        "native-mmap": "`load_native(mmap=True)`",
        "native-mmap-trusted": "`load_native(mmap=True, validate=False)`",
    }
    header = [
        "Open with",
        "File size (MB)",
        "Open (s)",
        "Memory after open (MB)",
        "First 1,000 queries (s)",
        "Next 1,000 queries (s)",
        "Memory after queries (MB)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * 6) + "|"]
    for mode in MODES:
        s = payload["summary_median"][mode]
        size = payload["file_bytes"]["npz" if mode == "npz" else "native"]
        base = s["rss_baseline_bytes"]
        cells = [
            labels[mode],
            f"{size / 1e6:,.1f}",
            f"{s['open_seconds']:.3f}",
            f"{(s['rss_after_open_bytes'] - base) / 1e6:,.1f}",
            f"{s['first_batch_seconds']:.3f}",
            f"{s['second_batch_seconds']:.3f}",
            f"{(s['rss_after_queries_bytes'] - base) / 1e6:,.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
