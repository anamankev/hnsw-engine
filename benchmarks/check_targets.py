"""Check the milestone A3 targets against a benchmark result and report them as a table.

Targets: build 100K SIFT vectors in under 10 minutes; recall@10 of at least 0.95 at under 1 ms
mean latency, single thread. Reads the compiled engine's SIFT 100K result file and, with
``--inject``, writes the table into the ``targets`` section of a document.

Usage::

    python benchmarks/check_targets.py \
        results/ann/sift-128-euclidean-n100000-main/hnsw-engine-numba-M16-ef_construction200.json \
        --inject docs/PERFORMANCE.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hnsw_engine.bench.report import inject

BUILD_LIMIT_S = 600.0
RECALL_TARGET = 0.95
LATENCY_LIMIT_MS = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result", type=Path)
    parser.add_argument("--inject", type=Path, action="append", default=[])
    args = parser.parse_args()
    record = json.loads(args.result.read_text())
    build = float(record["build_seconds"])
    eligible = [p for p in record["points"] if p["recall"] >= RECALL_TARGET]
    best = min(eligible, key=lambda p: p["mean_ms"]) if eligible else None
    rows = [
        "| Target | Measured | Met |",
        "|---|---|:---:|",
        f"| Build {record['dataset']['n_train']:,} SIFT vectors in under 10 minutes | "
        f"{build:.1f} s | {'yes' if build < BUILD_LIMIT_S else 'no'} |",
    ]
    if best is None:
        rows.append(
            "| Recall@10 >= 0.95 at under 1 ms mean latency | recall 0.95 not reached | no |"
        )
    else:
        met = best["mean_ms"] < LATENCY_LIMIT_MS
        rows.append(
            f"| Recall@10 >= 0.95 at under 1 ms mean latency | recall {best['recall']:.4f} at "
            f"{best['mean_ms']:.4f} ms (efSearch {best['value']}) | {'yes' if met else 'no'} |"
        )
    table = "\n".join(rows)
    note = (
        f"\nSource: `{args.result}` (engine `{record['index']['engine']}`, one thread, "
        f"public `HNSWIndex.search` per query)."
    )
    for document in args.inject:
        inject(document, "targets", table + "\n" + note)
    print(table)


if __name__ == "__main__":
    main()
