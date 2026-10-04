"""Run the ANN benchmark suite (ANN-Benchmarks methodology) on one dataset configuration.

Examples::

    # quick smoke run (a few minutes): 10K SIFT vectors, 200 queries, one repetition
    python benchmarks/run_ann.py --dataset sift-128-euclidean --quick

    # published configuration: 100K SIFT vectors, 1,000 queries, 3 repetitions
    python benchmarks/run_ann.py --dataset sift-128-euclidean --n-train 100000 \
        --indexes hnsw-engine-reference faiss-hnsw hnswlib faiss-flat faiss-ivf

Every index runs in its own process, single-threaded; results go to
``results/ann/<dataset>-n<size>-<label>/`` as one JSON file per index.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from hnsw_engine.bench.runner import IndexConfig, Suite, run_suite

EF_SWEEP = (10, 20, 40, 80, 120, 200, 400, 800)
NPROBE_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# The long pure-Python run goes last so it does not heat the machine before the others.
ALL_INDEXES = (
    "faiss-hnsw",
    "hnswlib",
    "hnsw-engine-numba",
    "faiss-flat",
    "faiss-ivf",
    "hnsw-engine-reference",
)


def index_configs(
    names: list[str], n_train: int, M: int, ef_construction: int, ef_sweep: tuple[int, ...]
) -> tuple[IndexConfig, ...]:
    """Build configurations with identical graph parameters for every HNSW implementation."""
    nlist = 2 ** round(math.log2(4 * math.sqrt(n_train)))  # FAISS guideline: ~4 sqrt(n) lists
    configs = []
    for name in names:
        if name.startswith("hnsw-engine") or name in ("faiss-hnsw", "hnswlib"):
            params = {"M": M, "ef_construction": ef_construction}
            configs.append(IndexConfig(name, params, ef_sweep))
        elif name == "faiss-ivf":
            sweep = tuple(p for p in NPROBE_SWEEP if p <= nlist)
            configs.append(IndexConfig(name, {"nlist": nlist}, sweep))
        elif name == "faiss-flat":
            configs.append(IndexConfig(name))
        else:
            raise SystemExit(f"unknown index {name}; choose from {ALL_INDEXES}")
    return tuple(configs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/ann"))
    parser.add_argument("--n-train", type=int, default=None, help="prefix size (default: all)")
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--indexes", nargs="+", default=list(ALL_INDEXES))
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-sweep", type=int, nargs="+", default=list(EF_SWEEP))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-queries", type=int, default=100)
    parser.add_argument("--time-budget", type=float, default=1200.0, help="seconds per index")
    parser.add_argument("--label", default=None, help="run directory suffix (default: timestamp)")
    parser.add_argument("--cooldown", type=float, default=30.0, help="seconds between workers")
    parser.add_argument("--quick", action="store_true", help="10K vectors, 200 queries, 1 rep")
    args = parser.parse_args()

    if args.quick:
        args.n_train, args.n_test, args.repetitions = 10_000, 200, 1
        args.ef_sweep = [10, 40, 160]
        args.cooldown = 0.0
        args.label = args.label or "quick"
    n_train = args.n_train
    if n_train is None:
        from hnsw_engine.datasets import get_info

        n_train = get_info(args.dataset).n_train
    suite = Suite(
        dataset=args.dataset,
        data_dir=args.data_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        k=args.k,
        indexes=index_configs(
            args.indexes, n_train, args.M, args.ef_construction, tuple(args.ef_sweep)
        ),
        repetitions=args.repetitions,
        warmup_queries=args.warmup_queries,
        time_budget_seconds=args.time_budget,
        cooldown_seconds=args.cooldown,
    )
    run_dir = run_suite(suite, args.results_dir, label=args.label)
    print(f"results in {run_dir}")


if __name__ == "__main__":
    main()
