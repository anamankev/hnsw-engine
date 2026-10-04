"""Profile index construction and search with cProfile (milestone A3).

Builds an index over the first ``--n`` SIFT vectors and runs ``--queries`` searches, profiling each
phase separately. Writes the full sorted profiles and a JSON summary of the top functions to
``results/profile/<engine>/``.

Usage: ``python benchmarks/profile_engine.py --engine reference --n 10000 --queries 500``
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

from hnsw_engine import HNSWIndex
from hnsw_engine.bench.env import capture_environment
from hnsw_engine.datasets import load


def top_functions(profile: cProfile.Profile, limit: int = 15) -> list[dict[str, object]]:
    """The ``limit`` functions with the largest self time (tottime)."""
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), (_cc, nc, tottime, cumtime, _) in stats.stats.items():  # type: ignore[attr-defined]
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}({name})",
                "calls": nc,
                "self_seconds": tottime,
                "cumulative_seconds": cumtime,
            }
        )
    rows.sort(key=lambda r: -float(r["self_seconds"]))
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="reference")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--ef", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/profile"))
    args = parser.parse_args()
    data = load("sift-128-euclidean", args.data_dir, n_train=args.n, n_test=args.queries, k=10)
    out = args.out / args.engine
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "engine": args.engine,
        "n": args.n,
        "queries": args.queries,
        "ef_search": args.ef,
        "M": 16,
        "ef_construction": 200,
    }
    index = HNSWIndex(128, M=16, ef_construction=200, seed=0, engine=args.engine)
    if args.engine != "reference":  # compile outside the profile
        warm = HNSWIndex(128, M=4, ef_construction=8, engine=args.engine)
        warm.add(data.train[:64])
        warm.search(data.train[0], k=1)

    for phase in ("build", "search"):
        profile = cProfile.Profile()
        start = time.perf_counter()
        profile.enable()
        if phase == "build":
            index.add(data.train)
        else:
            for q in data.test:
                index.search(q, k=10, ef=args.ef)
        profile.disable()
        seconds = time.perf_counter() - start
        text = io.StringIO()
        pstats.Stats(profile, stream=text).sort_stats("tottime").print_stats(30)
        (out / f"{phase}_profile.txt").write_text(text.getvalue())
        summary[phase] = {
            "seconds_under_profiler": seconds,
            "top_self_time": top_functions(profile),
        }
        print(f"{phase}: {seconds:.2f} s under cProfile")
        for row in summary[phase]["top_self_time"][:8]:  # type: ignore[index]
            print(f"   {row['self_seconds']:7.2f}s  {row['calls']:>9}  {row['function']}")
    summary["environment"] = capture_environment()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
