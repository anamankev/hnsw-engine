"""Render the study results (``results/studies/*.json``) as Markdown tables in STUDIES.md.

Each table replaces the ``<!-- table:study_<name> -->`` section of the document.

Usage: ``python benchmarks/report_studies.py --inject docs/STUDIES.md``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hnsw_engine.bench.report import inject

OUT = Path("results/studies")
EFS = (10, 40, 160, 320)


def rows(study: str) -> list[dict[str, Any]]:
    """Saved rows of one study."""
    return list(json.loads((OUT / f"{study}.json").read_text())["rows"])


def recall_cells(points: list[dict[str, Any]]) -> list[str]:
    """Recall at the reported efSearch values, then QPS at efSearch 40."""
    by_ef = {p["ef_search"]: p for p in points}
    return [f"{by_ef[e]['recall']:.4f}" for e in EFS] + [f"{by_ef[40]['qps']:,.0f}"]


def table(header: list[str], body: list[list[str]]) -> str:
    """Markdown table; the first column is left-aligned."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


RECALL_HEADER = [f"Recall@10 (ef {e})" for e in EFS] + ["QPS (ef 40)"]


def selection() -> str:
    body = [
        [r["data"], r["selection"], f"{r['unreachable_nodes']:,}", *recall_cells(r["points"])]
        for r in rows("selection")
    ]
    return table(["Data", "Selection", "Unreachable by any query", *RECALL_HEADER], body)


def m_study() -> str:
    body = [
        [
            str(r["M"]),
            f"{r['build_seconds']:.1f}",
            f"{r['memory_bytes'] / 1e6:.1f}",
            f"{r['unreachable_nodes']:,}",
            *recall_cells(r["points"]),
        ]
        for r in rows("m")
    ]
    return table(
        ["M", "Build (s)", "Index size (MB)", "Unreachable by any query", *RECALL_HEADER], body
    )


def efc_study() -> str:
    body = [
        [
            str(r["ef_construction"]),
            f"{r['build_seconds']:.1f}",
            f"{r['unreachable_nodes']:,}",
            *recall_cells(r["points"]),
        ]
        for r in rows("efc")
    ]
    return table(["efConstruction", "Build (s)", "Unreachable by any query", *RECALL_HEADER], body)


def reachability() -> str:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for r in rows("reachability"):
        name = r["selection"] + (" + keep pruned" if r["keep_pruned_connections"] else "")
        groups.setdefault((r["data"], r["M"], name), []).append(r)
    body = []
    for (data, m, name), group in groups.items():
        ordered = sorted(group, key=lambda g: g["seed"])
        any_query = " / ".join(f"{g['unreachable_nodes']:,}" for g in ordered)
        layer0 = " / ".join(f"{g['unreachable_layer0_from_entry']:,}" for g in ordered)
        degree = sum(g["mean_degree_layer0"] for g in group) / len(group)
        body.append([data, str(m), name, any_query, layer0, f"{degree:.2f}"])
    return table(
        [
            "Data",
            "M",
            "Selection",
            "Unreachable by any query (seeds 0 / 1 / 2)",
            "Unreachable on layer 0 from the entry point",
            "Mean layer-0 degree",
        ],
        body,
    )


def duplicates() -> str:
    body = []
    for r in rows("duplicates"):
        name = r["selection"] + (" + keep pruned" if r["keep_pruned_connections"] else "")
        body.append(
            [
                f"{int(r['duplicate_fraction'] * 100)}%",
                name,
                f"{r['unreachable_nodes']:,}",
                f"{r['mean_degree_layer0']:.2f}",
                *recall_cells(r["points"]),
            ]
        )
    return table(
        [
            "Duplicates",
            "Selection",
            "Unreachable by any query",
            "Mean layer-0 degree",
            *RECALL_HEADER,
        ],
        body,
    )


def dimension() -> str:
    body = []
    for r in rows("dimension"):
        by_ef = {p["ef_search"]: p for p in r["points"]}
        body.append(
            [
                str(r["dim"]),
                f"{r['build_seconds']:.1f}",
                *recall_cells(r["points"]),
                f"{by_ef[40]['distance_computations_per_query']:,.0f}",
            ]
        )
    return table(["Dimension", "Build (s)", *RECALL_HEADER, "Distances/query (ef 40)"], body)


TABLES = {
    "selection": selection,
    "m": m_study,
    "efc": efc_study,
    "reachability": reachability,
    "duplicates": duplicates,
    "dimension": dimension,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inject", type=Path, action="append", default=[])
    args = parser.parse_args()
    for name, render in TABLES.items():
        markdown = render()
        print(f"## {name}\n{markdown}\n")
        for document in args.inject:
            inject(document, f"study_{name}", markdown)


if __name__ == "__main__":
    main()
