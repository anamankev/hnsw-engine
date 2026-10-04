"""Tier 3: filtered search with metadata predicates, on SIFT 100K (``docs/DESIGN.md`` 9.1).

A predicate is a boolean mask over the indexed vectors, the same for every query of a setting.
Two kinds, each at selectivities from 50% down to 1% of the vectors:

* ``random``: each vector is allowed independently with probability ``selectivity``;
* ``correlated``: the ``selectivity`` share of vectors with the largest projection on a fixed
  random direction, so the allowed vectors sit in one region of the space and many queries are
  far from all of them (the hard case for graph search).

Ground truth is the exact top 10 among the allowed vectors. Four strategies, each timed as one
batch over the 1,000 queries on one thread (compiled loop or C++), median of three repetitions:

* ``in_graph``: ``HNSWIndex.batch_search(..., filter=mask)`` with hnswlib's semantics (design
  rule 9), efSearch swept;
* ``post_filter``: an unfiltered search for ``fetch = m * k / selectivity`` results
  (m = 1, 2, 4, 8; at most n), keeping the allowed ones;
* ``faiss_filtered``: FAISS ``IndexHNSWFlat`` (same M and efConstruction) searched with an
  ``IDSelectorBitmap``, efSearch swept;
* ``exact``: exact search over the allowed vectors (compiled loop, float64 accumulation).

Writes ``results/studies/filtered.json`` and ``docs/figures/study_filtered.png``; ``--inject``
writes the summary tables into a Markdown document.

Usage: ``python benchmarks/run_filtered.py [--plot-only] [--inject docs/STUDIES.md]``
"""

from __future__ import annotations

import argparse
import functools
import heapq
import importlib
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numba import njit
from numpy.typing import NDArray

from hnsw_engine import HNSWIndex, exact_distances, exact_search
from hnsw_engine.bench.env import capture_environment, ensure_single_thread
from hnsw_engine.bench.metrics import recall_at_k
from hnsw_engine.bench.report import STYLE, THEMES, _style_axes, inject
from hnsw_engine.datasets import load

SELECTIVITIES = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
PREDICATES = ("random", "correlated")
EF_SWEEP = (10, 20, 40, 80, 160, 320)
FETCH_MULTIPLIERS = (1, 2, 4, 8)
K = 10
REPETITIONS = 3
N_TRAIN = 100_000
M, EF_CONSTRUCTION = 16, 200
TARGET_RECALL = 0.95
OUT = Path("results/studies/filtered.json")
FIGURE = Path("docs/figures/study_filtered.png")
T = TypeVar("T")


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Result of the last of ``REPETITIONS`` calls and their median wall time in seconds."""
    seconds: list[float] = []
    for _ in range(REPETITIONS):
        start = time.perf_counter()
        result = fn()
        seconds.append(time.perf_counter() - start)
    return result, statistics.median(seconds)


@njit(cache=True)
def exact_subset_search(
    queries: NDArray[np.float32], vectors: NDArray[np.float32], subset: NDArray[np.int64], k: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Exact L2 top ``k`` among ``vectors[subset]`` for every query (ascending ``subset``).

    float64 accumulation and ties to the lower id, as in the oracle and the engines.
    """
    nq, dim = queries.shape
    ids = np.full((nq, k), -1, dtype=np.int64)
    dists = np.full((nq, k), np.inf, dtype=np.float64)
    for row in range(nq):
        q = queries[row].astype(np.float64)
        heap = [(0.0, np.int64(0))]  # max-heap of the best k as (-distance, -id)
        heap.pop()
        for j in range(subset.shape[0]):
            v = subset[j]
            acc = 0.0
            for i in range(dim):
                diff = np.float64(vectors[v, i]) - q[i]
                acc += diff * diff
            if len(heap) < k:
                heapq.heappush(heap, (-acc, -v))
            elif acc < -heap[0][0]:  # equal distance: the earlier (lower) id stays
                heapq.heapreplace(heap, (-acc, -v))
        found = [(-nd, -nv) for nd, nv in heap]
        found.sort()
        for i in range(len(found)):
            dists[row, i] = found[i][0]
            ids[row, i] = found[i][1]
    return ids, dists


def predicate(kind: str, selectivity: float, train: NDArray[np.float32]) -> NDArray[np.bool_]:
    """Boolean mask allowing about ``selectivity`` of the vectors (deterministic per setting)."""
    n = train.shape[0]
    rng = np.random.default_rng([int(selectivity * 1000), PREDICATES.index(kind)])
    if kind == "random":
        return np.asarray(rng.random(n) < selectivity)
    direction = np.random.default_rng(7).standard_normal(train.shape[1])
    projection = train.astype(np.float64) @ direction
    allowed = np.zeros(n, dtype=bool)
    allowed[np.argsort(-projection, kind="stable")[: max(1, round(selectivity * n))]] = True
    return allowed


def score(
    ids: NDArray[np.int64],
    train: NDArray[np.float32],
    queries: NDArray[np.float32],
    truth: tuple[NDArray[np.int64], NDArray[np.float64]],
    allowed: NDArray[np.bool_],
) -> dict[str, float]:
    """Tie-aware recall@10, share of full result lists, and a check that every id is allowed."""
    exact = exact_distances(train, queries, ids, "l2")
    valid = ids >= 0
    if not allowed[ids[valid]].all():
        raise AssertionError("a filtered search returned a vector the predicate excludes")
    return {
        "recall": recall_at_k(ids, exact, truth[0], truth[1], K)["recall"],
        "full_result_lists": float(valid.all(axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plot-only", action="store_true", help="redraw from the saved JSON")
    parser.add_argument("--inject", type=Path, help="write the summary tables into this file")
    args = parser.parse_args()
    if not args.plot_only:
        ensure_single_thread()
        run()
    payload = json.loads(OUT.read_text())
    plot(payload["rows"])
    if args.inject:
        for kind in PREDICATES:
            inject(args.inject, f"study_filtered_{kind}", table(payload["rows"], kind))


def run() -> None:
    faiss: Any = importlib.import_module("faiss")  # untyped use of the SWIG bindings
    faiss.omp_set_num_threads(1)
    data = load("sift-128-euclidean", "data", n_train=N_TRAIN, n_test=1000, k=K)
    train, queries = data.train, data.test
    n = train.shape[0]
    index = HNSWIndex(train.shape[1], M=M, ef_construction=EF_CONSTRUCTION, seed=0)
    start = time.perf_counter()
    index.add(train)
    build_seconds = time.perf_counter() - start
    faiss_index = faiss.IndexHNSWFlat(train.shape[1], M)
    faiss_index.hnsw.efConstruction = EF_CONSTRUCTION
    start = time.perf_counter()
    faiss_index.add(train)
    faiss_build_seconds = time.perf_counter() - start
    # Warm-up: compile every code path before anything is timed.
    index.batch_search(queries[:5], K, ef=10, filter=np.ones(n, dtype=bool))
    exact_subset_search(queries[:2], train, np.arange(50, dtype=np.int64), K)

    rows: list[dict[str, Any]] = []
    for kind in PREDICATES:
        for selectivity in SELECTIVITIES:
            allowed = predicate(kind, selectivity, train)
            subset = np.flatnonzero(allowed).astype(np.int64)
            sub_ids, sub_dist = exact_search(train[subset], queries, K)
            truth = (subset[sub_ids], sub_dist)
            row: dict[str, Any] = {
                "predicate": kind,
                "selectivity": selectivity,
                "allowed": int(subset.size),
            }
            (ids, _), seconds = timed(
                functools.partial(exact_subset_search, queries, train, subset, K)
            )
            row["exact"] = {
                "qps": len(queries) / seconds,
                **score(ids, train, queries, truth, allowed),
            }
            row["in_graph"] = []
            for ef in EF_SWEEP:
                index.reset_counters()
                (ids, _), seconds = timed(
                    functools.partial(index.batch_search, queries, K, ef=ef, filter=allowed)
                )
                row["in_graph"].append(
                    {
                        "ef_search": ef,
                        "qps": len(queries) / seconds,
                        "distance_computations_per_query": index.stats().distance_computations
                        / (REPETITIONS * len(queries)),
                        **score(ids, train, queries, truth, allowed),
                    }
                )
            row["post_filter"] = []
            for multiplier in FETCH_MULTIPLIERS:
                fetch = min(n, math.ceil(multiplier * K / selectivity))

                def post_filter(f: int = fetch, a: NDArray[np.bool_] = allowed) -> NDArray[Any]:
                    found, _ = index.batch_search(queries, f, ef=f)
                    kept = np.full((len(queries), K), -1, dtype=np.int64)
                    for i, candidates in enumerate(found):
                        good = candidates[(candidates >= 0) & a[np.maximum(candidates, 0)]][:K]
                        kept[i, : good.size] = good
                    return kept

                ids, seconds = timed(post_filter)
                row["post_filter"].append(
                    {
                        "multiplier": multiplier,
                        "fetch": fetch,
                        "qps": len(queries) / seconds,
                        **score(ids, train, queries, truth, allowed),
                    }
                )
            bitmap = np.packbits(allowed, bitorder="little")
            selector = faiss.IDSelectorBitmap(n, faiss.swig_ptr(bitmap))
            row["faiss_filtered"] = []
            for ef in EF_SWEEP:
                params = faiss.SearchParametersHNSW()
                params.sel = selector
                params.efSearch = ef
                faiss.cvar.hnsw_stats.reset()
                (_, found), seconds = timed(
                    functools.partial(faiss_index.search, queries, K, params=params)
                )
                row["faiss_filtered"].append(
                    {
                        "ef_search": ef,
                        "qps": len(queries) / seconds,
                        "distance_computations_per_query": faiss.cvar.hnsw_stats.ndis
                        / (REPETITIONS * len(queries)),
                        **score(found.astype(np.int64), train, queries, truth, allowed),
                    }
                )
            rows.append(row)
            print(
                f"{kind:10s} {selectivity:5.2f} exact {row['exact']['qps']:8.0f} qps | in-graph "
                + " ".join(f"{p['recall']:.3f}" for p in row["in_graph"])
                + " | post "
                + " ".join(f"{p['recall']:.3f}" for p in row["post_filter"])
                + " | faiss "
                + " ".join(f"{p['recall']:.3f}" for p in row["faiss_filtered"]),
                flush=True,
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "config": {
                    "dataset": "sift-128-euclidean",
                    "n_train": n,
                    "n_queries": len(queries),
                    "k": K,
                    "M": M,
                    "ef_construction": EF_CONSTRUCTION,
                    "seed": 0,
                    "repetitions": REPETITIONS,
                    "timing": "median of repetitions of one batch over all queries, one thread",
                    "build_seconds": build_seconds,
                    "faiss_build_seconds": faiss_build_seconds,
                },
                "rows": rows,
                "environment": capture_environment(),
            },
            indent=2,
        )
    )


def best_at_target(points: list[dict[str, Any]], knob: str) -> str:
    """Highest QPS among settings with recall >= the target, with its knob value."""
    good = [p for p in points if p["recall"] >= TARGET_RECALL]
    if not good:
        return f"not reached (max recall {max(p['recall'] for p in points):.3f})"
    best = max(good, key=lambda p: p["qps"])
    return f"{best['qps']:,.0f} ({knob} {best[knob]:,})"


def table(rows: list[dict[str, Any]], kind: str) -> str:
    """Markdown summary for one predicate kind."""
    header = [
        "Allowed vectors",
        "Exact search QPS",
        "In-graph filter: best QPS at recall >= 0.95",
        "Post-filter: best QPS at recall >= 0.95",
        "FAISS filtered: best QPS at recall >= 0.95",
        "In-graph distances per query (ef 40)",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * 5) + "|"]
    for row in (r for r in rows if r["predicate"] == kind):
        ef40 = next(p for p in row["in_graph"] if p["ef_search"] == 40)
        cells = [
            f"{row['allowed']:,} ({row['selectivity']:.0%})",
            f"{row['exact']['qps']:,.0f}",
            best_at_target(row["in_graph"], "ef_search"),
            best_at_target(row["post_filter"], "fetch"),
            best_at_target(row["faiss_filtered"], "ef_search"),
            f"{ef40['distance_computations_per_query']:,.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines).replace("ef_search", "ef")


def plot(rows: list[dict[str, Any]]) -> None:
    """Recall versus QPS per predicate (rows) and selectivity (columns)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEMES["light"]
    series = (
        ("in_graph", STYLE["hnsw-engine-numba"][0], "o", "hnsw-engine, in-graph filter"),
        ("post_filter", STYLE["hnsw-engine-reference"][0], "D", "hnsw-engine, post-filter"),
        ("faiss_filtered", STYLE["faiss-hnsw"][0], "s", "FAISS HNSW, IDSelector"),
    )
    exact_colour = STYLE["faiss-flat"][0]
    fig, axes = plt.subplots(
        len(PREDICATES),
        len(SELECTIVITIES),
        figsize=(2.75 * len(SELECTIVITIES) + 2.2, 3.0 * len(PREDICATES) + 0.6),
        dpi=150,
        sharey=True,
    )
    fig.patch.set_facecolor(theme["surface"])
    for r, kind in enumerate(PREDICATES):
        for c, selectivity in enumerate(SELECTIVITIES):
            ax = axes[r, c]
            _style_axes(ax, theme)
            row = next(
                x for x in rows if x["predicate"] == kind and x["selectivity"] == selectivity
            )
            for key, colour, marker, label in series:
                pts = sorted(row[key], key=lambda p: p["qps"], reverse=True)
                ax.plot(
                    [p["recall"] for p in pts],
                    [p["qps"] for p in pts],
                    color=colour,
                    marker=marker,
                    markersize=4.5,
                    linewidth=1.6,
                    label=label,
                    markeredgecolor=theme["surface"],
                    markeredgewidth=0.7,
                )
            ax.axhline(
                row["exact"]["qps"],
                color=exact_colour,
                linewidth=1.2,
                linestyle=(0, (4, 2)),
                label="exact search over allowed vectors",
            )
            ax.set_yscale("log")
            ax.set_xlim(-0.03, 1.03)
            ax.set_title(
                f"{kind}, {selectivity:.0%} allowed", fontsize=9, color=theme["ink"], loc="left"
            )
            if r == len(PREDICATES) - 1:
                ax.set_xlabel("Recall@10", color=theme["ink2"], fontsize=8.5)
        axes[r, 0].set_ylabel("Queries per second (log)", color=theme["ink2"], fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.80, 0.98),
        frameon=False,
        fontsize=8.5,
    )
    for text in legend.get_texts():
        text.set_color(theme["ink2"])
    fig.suptitle(
        "Filtered search on SIFT 100K (M = 16, one thread, 1,000 queries)",
        x=0.01,
        ha="left",
        color=theme["ink"],
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.80, 0.95))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=theme["surface"])
    plt.close(fig)


if __name__ == "__main__":
    main()
