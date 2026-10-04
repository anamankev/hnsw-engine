"""Milestone A6: controlled studies, each changing one variable.

Studies (all with the Numba engine, which builds the same graphs as the reference):

* ``selection``   simple vs heuristic neighbour selection on uniform and clustered data
* ``m``           effect of M on SIFT 100K (efConstruction fixed)
* ``efc``         effect of efConstruction on SIFT 100K (M fixed)
* ``reachability`` nodes no search can reach after pruning, with and without
                  ``keep_pruned_connections``
* ``duplicates``  exact duplicate vectors under the strict heuristic
* ``dimension``   Gaussian data of increasing dimension

Each study writes ``results/studies/<study>.json`` (configs, every measured point, environment)
and a figure ``docs/figures/study_<study>.png``. Latency is the mean per query of one compiled
``batch_search`` call (no per-query Python overhead), so absolute values are lower than in
``BENCHMARKS.md``; comparisons within a study are like for like.

Usage: ``python benchmarks/run_studies.py [selection m efc reachability duplicates dimension]``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hnsw_engine import (
    HNSWIndex,
    exact_distances,
    exact_search,
    reachable_from_entry,
    searchable_from_entry,
)
from hnsw_engine.bench.env import capture_environment
from hnsw_engine.bench.metrics import recall_at_k
from hnsw_engine.datasets import load
from hnsw_engine.synthetic import clustered, gaussian, uniform, with_duplicates

EF_SWEEP = (10, 20, 40, 80, 160, 320)
K = 10
OUT = Path("results/studies")
FIGURES = Path("docs/figures")


def measure(
    index: HNSWIndex,
    train: np.ndarray,
    queries: np.ndarray,
    truth: tuple[np.ndarray, np.ndarray],
    metric: str = "l2",
    sweep: tuple[int, ...] = EF_SWEEP,
) -> list[dict[str, float]]:
    """Recall@10 and mean latency (compiled batch path) for each efSearch value."""
    points = []
    index.batch_search(queries[:10], K, ef=sweep[0])  # warm-up
    for ef in sweep:
        index.reset_counters()
        start = time.perf_counter()
        ids, _ = index.batch_search(queries, K, ef=ef)
        seconds = time.perf_counter() - start
        exact = exact_distances(train, queries, ids, metric)
        rec = recall_at_k(ids, exact, truth[0], truth[1], K)
        points.append(
            {
                "ef_search": ef,
                "recall": rec["recall"],
                "mean_ms": 1000 * seconds / queries.shape[0],
                "qps": queries.shape[0] / seconds,
                "distance_computations_per_query": index.stats().distance_computations
                / len(queries),
            }
        )
    return points


def build(train: np.ndarray, **params: Any) -> tuple[HNSWIndex, float]:
    """Build an index (Numba engine) and return it with its build time in seconds."""
    index = HNSWIndex(train.shape[1], engine="numba", seed=params.pop("seed", 0), **params)
    start = time.perf_counter()
    index.add(train)
    return index, time.perf_counter() - start


def unreachable(index: HNSWIndex) -> int:
    """Nodes that no query can reach (closure through every layer, from the top down)."""
    return int((~searchable_from_entry(index.graph_state())).sum())


def unreachable_layer0(index: HNSWIndex) -> int:
    """Nodes that layer 0 alone cannot reach from the entry point (a stricter view)."""
    return int((~reachable_from_entry(index.graph_state())).sum())


def save(study: str, payload: dict[str, Any]) -> None:
    """Write a study's results with the environment."""
    OUT.mkdir(parents=True, exist_ok=True)
    payload["environment"] = capture_environment()
    (OUT / f"{study}.json").write_text(json.dumps(payload, indent=2))


def plot(study: str, panels: dict[str, dict[str, list[dict[str, float]]]], title: str) -> None:
    """One panel per dataset: recall (x) vs QPS (log y), one curve per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hnsw_engine.bench.report import THEMES, _style_axes

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    markers = ["o", "s", "^", "D", "v", "P"]
    theme = THEMES["light"]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.4 * len(panels) + 1.6, 4.4), dpi=150, squeeze=False
    )
    fig.patch.set_facecolor(theme["surface"])
    for ax, (panel, curves) in zip(axes[0], panels.items(), strict=True):
        _style_axes(ax, theme)
        for slot, (label, points) in enumerate(curves.items()):
            pts = sorted(points, key=lambda p: p["recall"])
            ax.plot(
                [p["recall"] for p in pts],
                [p["qps"] for p in pts],
                color=palette[slot],
                marker=markers[slot],
                markersize=5,
                linewidth=1.8,
                label=label,
                markeredgecolor=theme["surface"],
                markeredgewidth=0.8,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Recall@10", color=theme["ink2"], fontsize=9)
        ax.set_title(panel, color=theme["ink"], fontsize=10, loc="left")
    axes[0][0].set_ylabel(
        "Queries per second (compiled batch, log)", color=theme["ink2"], fontsize=9
    )
    legend = axes[0][-1].legend(
        loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=8.5
    )
    for text in legend.get_texts():
        text.set_color(theme["ink2"])
    fig.suptitle(title, x=0.01, ha="left", color=theme["ink"], fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"study_{study}.png", facecolor=theme["surface"])
    plt.close(fig)


def synthetic(name: str, n: int, dim: int, seed: int) -> np.ndarray:
    """The synthetic datasets used by several studies."""
    if name == "uniform":
        return uniform(n, dim, seed)
    if name == "clustered":
        return clustered(n, dim, seed, n_clusters=100, spread=0.02)
    return gaussian(n, dim, seed)


def study_selection() -> None:
    """Simple vs heuristic selection on uniform and clustered data (n = 20,000, d = 8)."""
    panels: dict[str, dict[str, list[dict[str, float]]]] = {}
    rows = []
    for data_name in ("uniform", "clustered"):
        train = synthetic(data_name, 20_000, 8, seed=1)
        queries = synthetic(data_name, 1000, 8, seed=2)
        truth = exact_search(train, queries, K)
        panels[f"{data_name} (n=20,000, d=8)"] = {}
        for selection in ("simple", "heuristic"):
            index, seconds = build(train, M=8, ef_construction=100, selection=selection)
            points = measure(index, train, queries, truth)
            panels[f"{data_name} (n=20,000, d=8)"][selection] = points
            rows.append(
                {
                    "data": data_name,
                    "selection": selection,
                    "build_seconds": seconds,
                    "unreachable_nodes": unreachable(index),
                    "unreachable_layer0_from_entry": unreachable_layer0(index),
                    "points": points,
                }
            )
            print(
                "selection",
                data_name,
                selection,
                "unreachable",
                rows[-1]["unreachable_nodes"],
                "recall",
                [round(p["recall"], 4) for p in points],
                flush=True,
            )
    save(
        "selection",
        {
            "config": {
                "M": 8,
                "ef_construction": 100,
                "n": 20_000,
                "dim": 8,
                "clusters": 100,
                "spread": 0.02,
            },
            "rows": rows,
        },
    )
    plot("selection", panels, "Neighbour selection: simple vs heuristic (M = 8)")


def _sift() -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    data = load("sift-128-euclidean", "data", n_train=100_000, n_test=1000, k=100)
    return data.train, data.test, (data.neighbors, data.distances)


def study_m() -> None:
    """Effect of M on SIFT 100K, efConstruction = 200."""
    train, queries, truth = _sift()
    curves, rows = {}, []
    for m in (4, 8, 16, 32, 48):
        index, seconds = build(train, M=m, ef_construction=200)
        points = measure(index, train, queries, truth)
        curves[f"M = {m}"] = points
        rows.append(
            {
                "M": m,
                "build_seconds": seconds,
                "memory_bytes": index.stats().memory_bytes,
                "unreachable_nodes": unreachable(index),
                "unreachable_layer0_from_entry": unreachable_layer0(index),
                "points": points,
            }
        )
        print("M", m, round(seconds, 1), "s", [round(p["recall"], 4) for p in points], flush=True)
    save(
        "m",
        {
            "config": {"dataset": "sift-128-euclidean", "n": 100_000, "ef_construction": 200},
            "rows": rows,
        },
    )
    plot("m", {"SIFT 100K, efConstruction = 200": curves}, "Effect of M")


def study_efc() -> None:
    """Effect of efConstruction on SIFT 100K, M = 16."""
    train, queries, truth = _sift()
    curves, rows = {}, []
    for efc in (16, 50, 100, 200, 400):
        index, seconds = build(train, M=16, ef_construction=efc)
        points = measure(index, train, queries, truth)
        curves[f"efConstruction = {efc}"] = points
        rows.append(
            {
                "ef_construction": efc,
                "build_seconds": seconds,
                "unreachable_nodes": unreachable(index),
                "unreachable_layer0_from_entry": unreachable_layer0(index),
                "points": points,
            }
        )
        print(
            "efc", efc, round(seconds, 1), "s", [round(p["recall"], 4) for p in points], flush=True
        )
    save("efc", {"config": {"dataset": "sift-128-euclidean", "n": 100_000, "M": 16}, "rows": rows})
    plot("efc", {"SIFT 100K, M = 16": curves}, "Effect of efConstruction")


def study_reachability() -> None:
    """Unreachable nodes after pruning: selection x keep_pruned_connections, small M."""
    rows = []
    configs = [("simple", False), ("heuristic", False), ("heuristic", True)]
    for data_name in ("uniform", "clustered"):
        train = synthetic(data_name, 20_000, 8, seed=1)
        for m in (2, 4, 8):
            for selection, keep in configs:
                for seed in (0, 1, 2):
                    index, _ = build(
                        train,
                        M=m,
                        ef_construction=50,
                        selection=selection,
                        keep_pruned_connections=keep,
                        seed=seed,
                    )
                    rows.append(
                        {
                            "data": data_name,
                            "M": m,
                            "selection": selection,
                            "keep_pruned_connections": keep,
                            "seed": seed,
                            "unreachable_nodes": unreachable(index),
                            "unreachable_layer0_from_entry": unreachable_layer0(index),
                            "mean_degree_layer0": index.stats().mean_degree_per_layer[0],
                        }
                    )
                print(
                    "reach",
                    data_name,
                    m,
                    selection,
                    keep,
                    [r["unreachable_nodes"] for r in rows[-3:]],
                    flush=True,
                )
    save(
        "reachability",
        {
            "config": {"n": 20_000, "dim": 8, "ef_construction": 50, "seeds": [0, 1, 2]},
            "rows": rows,
        },
    )


def study_duplicates() -> None:
    """Exact duplicates (fraction of rows replaced by copies of others) under each selection."""
    base = gaussian(20_000, 8, seed=1)
    queries = gaussian(1000, 8, seed=2)
    rows: list[dict[str, Any]] = []
    panels: dict[str, dict[str, list[dict[str, float]]]] = {}
    configs = [("simple", False), ("heuristic", False), ("heuristic", True)]
    for fraction in (0.0, 0.25, 0.5):
        train = with_duplicates(base, fraction, seed=3)
        truth = exact_search(train, queries, K)
        label = f"{int(fraction * 100)}% duplicates"
        panels[label] = {}
        for selection, keep in configs:
            index, seconds = build(
                train, M=8, ef_construction=100, selection=selection, keep_pruned_connections=keep
            )
            points = measure(index, train, queries, truth)
            name = selection + (" + keep pruned" if keep else "")
            panels[label][name] = points
            rows.append(
                {
                    "duplicate_fraction": fraction,
                    "selection": selection,
                    "keep_pruned_connections": keep,
                    "build_seconds": seconds,
                    "unreachable_nodes": unreachable(index),
                    "unreachable_layer0_from_entry": unreachable_layer0(index),
                    "mean_degree_layer0": index.stats().mean_degree_per_layer[0],
                    "points": points,
                }
            )
            print(
                "dup",
                fraction,
                name,
                "unreachable",
                rows[-1]["unreachable_nodes"],
                "degree",
                round(rows[-1]["mean_degree_layer0"], 2),
                [round(p["recall"], 4) for p in points],
                flush=True,
            )
    save(
        "duplicates",
        {"config": {"n": 20_000, "dim": 8, "M": 8, "ef_construction": 100}, "rows": rows},
    )
    plot("duplicates", panels, "Exact duplicate vectors (M = 8)")


def study_dimension() -> None:
    """Gaussian data of increasing dimension (n = 20,000, M = 16, efConstruction = 100)."""
    curves, rows = {}, []
    for dim in (4, 16, 64, 256):
        train, queries = gaussian(20_000, dim, seed=1), gaussian(1000, dim, seed=2)
        truth = exact_search(train, queries, K)
        index, seconds = build(train, M=16, ef_construction=100)
        points = measure(index, train, queries, truth)
        curves[f"d = {dim}"] = points
        rows.append({"dim": dim, "build_seconds": seconds, "points": points})
        print("dim", dim, [round(p["recall"], 4) for p in points], flush=True)
    save("dimension", {"config": {"n": 20_000, "M": 16, "ef_construction": 100}, "rows": rows})
    plot("dimension", {"Gaussian, n = 20,000": curves}, "Effect of dimension")


def replot() -> None:
    """Redraw every study figure from the saved JSON (no re-running)."""

    def load_rows(study: str) -> list[dict[str, Any]]:
        return list(json.loads((OUT / f"{study}.json").read_text())["rows"])

    rows = load_rows("selection")
    plot(
        "selection",
        {
            f"{data} (n=20,000, d=8)": {
                r["selection"]: r["points"] for r in rows if r["data"] == data
            }
            for data in ("uniform", "clustered")
        },
        "Neighbour selection: simple vs heuristic (M = 8)",
    )
    plot(
        "m",
        {"SIFT 100K, efConstruction = 200": {f"M = {r['M']}": r["points"] for r in load_rows("m")}},
        "Effect of M",
    )
    plot(
        "efc",
        {
            "SIFT 100K, M = 16": {
                f"efConstruction = {r['ef_construction']}": r["points"] for r in load_rows("efc")
            }
        },
        "Effect of efConstruction",
    )
    rows = load_rows("duplicates")
    plot(
        "duplicates",
        {
            f"{int(f * 100)}% duplicates": {
                r["selection"] + (" + keep pruned" if r["keep_pruned_connections"] else ""): r[
                    "points"
                ]
                for r in rows
                if r["duplicate_fraction"] == f
            }
            for f in (0.0, 0.25, 0.5)
        },
        "Exact duplicate vectors (M = 8)",
    )
    plot(
        "dimension",
        {"Gaussian, n = 20,000": {f"d = {r['dim']}": r["points"] for r in load_rows("dimension")}},
        "Effect of dimension",
    )


STUDIES = {
    "selection": study_selection,
    "m": study_m,
    "efc": study_efc,
    "reachability": study_reachability,
    "duplicates": study_duplicates,
    "dimension": study_dimension,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("studies", nargs="*", help=f"any of {list(STUDIES)} (default: all)")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from saved JSON")
    args = parser.parse_args()
    if args.plot_only:
        replot()
        return
    unknown = set(args.studies) - set(STUDIES)
    if unknown:
        parser.error(f"unknown studies {sorted(unknown)}; choose from {list(STUDIES)}")
    args.studies = args.studies or list(STUDIES)
    HNSWIndex(2, M=2, engine="numba").add(np.eye(2))  # compile before timing anything
    for name in args.studies:
        STUDIES[name]()


if __name__ == "__main__":
    main()
