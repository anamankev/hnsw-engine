"""Turn benchmark JSON files into figures and tables.

* Recall-vs-QPS figures (ANN-Benchmarks view: recall on x, queries per second on a log y axis),
  rendered in light and dark variants.
* CSV and Markdown tables of every measured point and a per-index summary.
* Optional injection of generated Markdown tables into documents between
  ``<!-- table:NAME -->`` and ``<!-- /table:NAME -->`` markers, so documents always show exactly
  what the results files contain.

Colours follow the entity, never its rank: each index keeps the same colour and marker in every
figure (validated default categorical palette, fixed slot order, with distinct markers as a
second, colour-independent encoding).
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# (light, dark) colours by fixed slot, marker, display label.
STYLE: dict[str, tuple[str, str, str, str]] = {
    "hnsw-engine-numba": ("#2a78d6", "#3987e5", "o", "hnsw-engine (Numba)"),
    "faiss-hnsw": ("#eb6834", "#d95926", "s", "FAISS IndexHNSWFlat"),
    "hnswlib": ("#1baf7a", "#199e70", "^", "hnswlib"),
    "hnsw-engine-reference": ("#eda100", "#c98500", "D", "hnsw-engine (pure Python)"),
    "faiss-ivf": ("#e87ba4", "#d55181", "v", "FAISS IndexIVFFlat"),
    "faiss-flat": ("#008300", "#008300", "P", "FAISS IndexFlat (exact)"),
}
ORDER = list(STYLE)

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


def load_results(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load worker result files from run directories or explicit file paths."""
    results = []
    for path in paths:
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for file in files:
            if file.name.endswith(".config.json") or file.name == "manifest.json":
                continue
            results.append(json.loads(file.read_text()))
    results.sort(
        key=lambda r: ORDER.index(r["index"]["name"]) if r["index"]["name"] in ORDER else 99
    )
    return results


def points_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (index, query-parameter value)."""
    return [
        {
            "index": result["index"]["name"],
            "param": p["query_param"] or "-",
            "value": "-" if p["value"] is None else p["value"],
            "recall": p["recall"],
            "recall_ids": p["recall_ids"],
            "qps": p["qps"],
            "mean_ms": p["mean_ms"],
            "p50_ms": p["p50_ms"],
            "p99_ms": p["p99_ms"],
            "dist_per_query": p.get("distance_computations_per_query"),
            "hops_per_query": p.get("hops_per_query"),
            "deterministic": p["deterministic_across_repetitions"],
        }
        for result in results
        for p in result["points"]
    ]


def summary_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-index build cost, memory and the best measured QPS at recall thresholds.

    "QPS at recall >= t" is the highest QPS among measured points with recall >= t (no
    interpolation); ``None`` when no point reaches t.
    """
    rows = []
    for result in results:
        points = result["points"]
        row: dict[str, Any] = {
            "index": result["index"]["name"],
            "build_s": result["build_seconds"],
            "index_rss_mb": result["index_rss_bytes"] / 1e6,
            "logical_mb": None
            if result["logical_bytes"] is None
            else result["logical_bytes"] / 1e6,
            "max_recall": max(p["recall"] for p in points),
        }
        for threshold in (0.90, 0.95, 0.99):
            ok = [p["qps"] for p in points if p["recall"] >= threshold]
            row[f"qps_at_{threshold:.2f}"] = max(ok) if ok else None
        row["truncated"] = result["truncated_by_time_budget"]
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write ``rows`` (dictionaries with identical keys) as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, key: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if key.startswith("qps"):
            return f"{value:,.0f}"
        if key.startswith("recall") or key == "max_recall":
            return f"{value:.4f}"
        if key.endswith("_ms"):
            return f"{value:.4f}" if value < 1 else f"{value:.3f}"
        if key in ("dist_per_query", "hops_per_query"):
            return f"{value:,.1f}"
        if key.endswith("_mb"):
            return f"{value:,.1f}"
        return f"{value:,.2f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: dict[str, str]) -> str:
    """Render ``rows`` as a Markdown table; ``columns`` maps keys to header labels."""
    header = "| " + " | ".join(columns.values()) + " |"
    rule = "|" + "|".join("---" if k == "index" else "---:" for k in columns) + "|"
    body = ["| " + " | ".join(_fmt(r[k], k) for k in columns) + " |" for r in rows]
    return "\n".join([header, rule, *body])


SUMMARY_COLUMNS = {
    "index": "Index",
    "build_s": "Build (s)",
    "index_rss_mb": "Index RSS (MB)",
    "logical_mb": "Index size (MB)",
    "max_recall": "Max recall@10",
    "qps_at_0.90": "QPS @ recall>=0.90",
    "qps_at_0.95": "QPS @ recall>=0.95",
    "qps_at_0.99": "QPS @ recall>=0.99",
}
POINT_COLUMNS = {
    "index": "Index",
    "param": "Param",
    "value": "Value",
    "recall": "Recall@10",
    "qps": "QPS",
    "mean_ms": "Mean (ms)",
    "p50_ms": "p50 (ms)",
    "p99_ms": "p99 (ms)",
    "dist_per_query": "Distances/query",
}


HNSW_INDEXES = ("hnsw-engine-numba", "hnsw-engine-reference", "faiss-hnsw", "hnswlib")
SHORT = {
    "hnsw-engine-numba": "ours (Numba)",
    "hnsw-engine-reference": "ours (Python)",
    "faiss-hnsw": "FAISS",
    "hnswlib": "hnswlib",
}


def equal_parameter_markdown(results: list[dict[str, Any]]) -> str:
    """HNSW implementations side by side at identical ``ef_search`` (same M, efConstruction).

    Equal recall and equal distance computations at equal parameters show the algorithm is the
    same; the QPS columns then isolate implementation speed.
    """
    by_index = {
        r["index"]["name"]: {p["value"]: p for p in r["points"]}
        for r in results
        if r["index"]["name"] in HNSW_INDEXES
    }
    names = [n for n in HNSW_INDEXES if n in by_index]
    if not names:
        return ""
    common = sorted(set.intersection(*(set(by_index[n]) for n in names)))
    with_dist = [n for n in names if "distance_computations_per_query" in by_index[n][common[0]]]
    headers = (
        ["efSearch"]
        + [f"Recall: {SHORT[n]}" for n in names]
        + [f"Distances/query: {SHORT[n]}" for n in with_dist]
        + [f"QPS: {SHORT[n]}" for n in names]
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---:"] * len(headers)) + "|"]
    for ef in common:
        cells = [str(ef)]
        cells += [f"{by_index[n][ef]['recall']:.4f}" for n in names]
        cells += [f"{by_index[n][ef]['distance_computations_per_query']:,.1f}" for n in with_dist]
        cells += [f"{by_index[n][ef]['qps']:,.0f}" for n in names]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def environment_markdown(results: list[dict[str, Any]]) -> str:
    """Hardware and software of the run (taken from the first result file)."""
    env = results[0]["environment"]
    pkgs = env["packages"]
    ram = env["ram_bytes"]
    rows = [
        (
            "CPU",
            f"{env['cpu_model']} ({env['performance_cores']} performance + "
            f"{env['efficiency_cores']} efficiency cores)",
        ),
        ("RAM", f"{ram / 2**30:.0f} GiB" if ram else "unknown"),
        ("OS", f"macOS {env.get('macos_version', '')} ({env['platform']})"),
        ("Python", env["python"]),
        ("Libraries", ", ".join(f"{k} {v}" for k, v in pkgs.items() if v)),
        (
            "Threads",
            "1 for every library ("
            + ", ".join(f"{k}={v}" for k, v in env["thread_env"].items() if v)
            + ")",
        ),
        ("Run date (UTC)", env["timestamp_utc"]),
    ]
    lines = ["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines)


def _style_axes(ax: Any, theme: dict[str, str]) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, which="major", color=theme["grid"], linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(which="both", colors=theme["muted"], labelsize=8.5)


def plot_recall_qps(results: list[dict[str, Any]], title: str, out_base: Path) -> list[Path]:
    """Recall (x) vs QPS (log y), one curve per index, writes ``<out_base>_{light,dark}.png``.

    Two panels share the y axis: the full recall range and a zoom on recall >= 0.98, where the
    implementations are hardest to tell apart.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for theme_name, theme in THEMES.items():
        fig, (ax_full, ax_zoom) = plt.subplots(
            1, 2, figsize=(11.6, 4.9), dpi=160, sharey=True, gridspec_kw={"width_ratios": [3, 2]}
        )
        fig.patch.set_facecolor(theme["surface"])
        all_qps = [p["qps"] for r in results for p in r["points"]]
        low_points = [p for r in results for p in r["points"] if p["recall"] < 0.5]
        for ax in (ax_full, ax_zoom):
            _style_axes(ax, theme)
            for result in results:
                name = result["index"]["name"]
                light, dark, marker, label = STYLE.get(name, ("#52514e", "#c3c2b7", "x", name))
                pts = sorted(result["points"], key=lambda p: p["recall"])
                xs = [p["recall"] for p in pts]
                ax.plot(
                    xs,
                    [p["qps"] for p in pts],
                    color=light if theme_name == "light" else dark,
                    linewidth=1.8,
                    marker=marker,
                    markersize=6,
                    linestyle="-" if len(xs) > 1 else "none",
                    markeredgecolor=theme["surface"],
                    markeredgewidth=0.8,
                    label=label,
                    zorder=3,
                )
            ax.axhline(1000, color=theme["muted"], linewidth=0.8, zorder=1)
        ax_full.set_yscale("log")
        ax_full.set_ylim(min(min(all_qps), 1000) / 2.0, max(all_qps) * 1.6)
        ax_full.set_xlim(0.5, 1.005)
        ax_zoom.set_xlim(0.98, 1.0008)
        ax_zoom.set_xticks([0.98, 0.985, 0.99, 0.995, 1.0])
        ax_zoom.set_xticklabels(["0.980", "0.985", "0.990", "0.995", "1.000"])
        ax_full.axvline(0.95, color=theme["muted"], linewidth=0.8, zorder=1)
        ax_full.text(0.505, 1000 * 1.1, "1 ms mean latency", color=theme["muted"], fontsize=7.5)
        ax_full.text(
            0.947,
            ax_full.get_ylim()[0] * 1.2,
            "recall 0.95",
            color=theme["muted"],
            fontsize=7.5,
            ha="right",
        )
        ax_full.set_xlabel("Recall@10 (higher is better)", color=theme["ink2"], fontsize=9.5)
        ax_zoom.set_xlabel("Recall@10, zoom on 0.98 to 1.0", color=theme["ink2"], fontsize=9.5)
        ax_full.set_ylabel(
            "Queries per second, one thread (log scale)", color=theme["ink2"], fontsize=9.5
        )
        ax_full.set_title(title, color=theme["ink"], fontsize=11, loc="left", pad=10)
        legend = ax_zoom.legend(
            loc="upper left",
            bbox_to_anchor=(1.03, 1.0),
            fontsize=8.5,
            frameon=False,
            borderaxespad=0.0,
        )
        for text in legend.get_texts():
            text.set_color(theme["ink2"])
        if low_points:
            fig.text(
                0.99,
                0.01,
                "Points with recall < 0.5 are off-chart; see the tables.",
                color=theme["muted"],
                fontsize=7,
                ha="right",
            )
        fig.tight_layout()
        path = out_base.parent / f"{out_base.name}_{theme_name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=theme["surface"])
        plt.close(fig)
        written.append(path)
    return written


def inject(document: Path, name: str, content: str) -> None:
    """Replace the text between ``<!-- table:name -->`` and ``<!-- /table:name -->``."""
    text = document.read_text()
    marker = re.escape(name)
    pattern = re.compile(rf"(<!-- table:{marker} -->).*?(<!-- /table:{marker} -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"markers for table {name!r} not found in {document}")
    document.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", text))
