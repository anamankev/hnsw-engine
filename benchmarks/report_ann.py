"""Generate figures and tables from benchmark run directories.

Usage::

    python benchmarks/report_ann.py results/ann/sift-128-euclidean-n100000-main \
        --name sift100k --title "SIFT 100K (128-d, L2)" --inject docs/BENCHMARKS.md

Writes ``docs/figures/<name>_recall_qps_{light,dark}.png`` and, under ``docs/tables/``, the
``<name>_points``, ``<name>_summary``, ``<name>_equal_ef`` and ``<name>_environment`` tables. With
``--inject``, each Markdown table replaces the matching ``<!-- table:<name>_<kind> -->`` section of
the document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hnsw_engine.bench.report import (
    POINT_COLUMNS,
    SUMMARY_COLUMNS,
    environment_markdown,
    equal_parameter_markdown,
    inject,
    load_results,
    markdown_table,
    plot_recall_qps,
    points_table,
    summary_table,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run directories or result files")
    parser.add_argument("--name", required=True, help="short name used in output file names")
    parser.add_argument("--title", required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--inject", type=Path, action="append", default=[])
    args = parser.parse_args()

    results = load_results(args.runs)
    tables = args.docs / "tables"
    points = points_table(results)
    summary = summary_table(results)
    write_csv(points, tables / f"{args.name}_points.csv")
    write_csv(summary, tables / f"{args.name}_summary.csv")
    points_md = markdown_table(points, POINT_COLUMNS)
    summary_md = markdown_table(summary, SUMMARY_COLUMNS)
    equal_md = equal_parameter_markdown(results)
    env_md = environment_markdown(results)
    (tables / f"{args.name}_points.md").write_text(points_md + "\n")
    (tables / f"{args.name}_summary.md").write_text(summary_md + "\n")
    (tables / f"{args.name}_equal_ef.md").write_text(equal_md + "\n")
    (tables / f"{args.name}_environment.md").write_text(env_md + "\n")
    figures = plot_recall_qps(
        results, args.title, args.docs / "figures" / f"{args.name}_recall_qps"
    )
    tables = {
        "summary": summary_md,
        "points": points_md,
        "equal_ef": equal_md,
        "environment": env_md,
    }
    for document in args.inject:
        text = document.read_text()
        present = [kind for kind in tables if f"<!-- table:{args.name}_{kind} -->" in text]
        if not present:
            raise SystemExit(f"no <!-- table:{args.name}_* --> markers in {document}")
        for kind in present:  # documents choose which tables they show
            inject(document, f"{args.name}_{kind}", tables[kind])
    print(summary_md)
    print("figures:", *figures)


if __name__ == "__main__":
    main()
