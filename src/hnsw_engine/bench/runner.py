"""Run a benchmark suite: one isolated worker process per index configuration, sequentially.

Configurations run one after another (never in parallel) so they do not compete for memory
bandwidth, caches or thermal headroom. Exact ground truth is computed once in the parent and
cached, so workers only read it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datasets import load
from .env import capture_environment, single_thread_environment


@dataclass(frozen=True)
class IndexConfig:
    """One index to benchmark: adapter name, build parameters and the query-time sweep."""

    index: str
    params: dict[str, Any] = field(default_factory=dict)
    sweep: tuple[int, ...] = ()

    @property
    def slug(self) -> str:
        """Filesystem-friendly identifier, e.g. ``faiss-hnsw-M16-ef_construction200``."""
        parts = [self.index] + [f"{k}{v}" for k, v in sorted(self.params.items())]
        return "-".join(parts)


@dataclass(frozen=True)
class Suite:
    """A full benchmark run on one dataset configuration."""

    dataset: str
    data_dir: Path
    n_train: int | None
    n_test: int
    k: int
    indexes: tuple[IndexConfig, ...]
    repetitions: int = 3
    warmup_queries: int = 100
    time_budget_seconds: float = 1200.0
    k_ground_truth: int = 100
    cooldown_seconds: float = 30.0
    max_attempts: int = 3
    suspension_tolerance_seconds: float = 1.0


def run_suite(suite: Suite, results_dir: Path, *, label: str | None = None) -> Path:
    """Run every configuration of ``suite``; returns the run directory with one JSON each."""
    load(
        suite.dataset,
        suite.data_dir,
        n_train=suite.n_train,
        n_test=suite.n_test,
        k=suite.k_ground_truth,
    )  # compute and cache exact ground truth once
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    size = "full" if suite.n_train is None else str(suite.n_train)
    run_dir = results_dir / f"{suite.dataset}-n{size}-{label or stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "suite": {
            **suite.__dict__,
            "data_dir": str(suite.data_dir),
            "indexes": [c.__dict__ for c in suite.indexes],
        },
        "environment": capture_environment(),
        "runs": [],
    }
    env = {**os.environ, **single_thread_environment()}
    for cfg in suite.indexes:
        out = run_dir / f"{cfg.slug}.json"
        config = {
            "dataset": suite.dataset,
            "data_dir": str(suite.data_dir),
            "n_train": suite.n_train,
            "n_test": suite.n_test,
            "k": suite.k,
            "k_ground_truth": suite.k_ground_truth,
            "index": cfg.index,
            "params": cfg.params,
            "sweep": list(cfg.sweep),
            "repetitions": suite.repetitions,
            "warmup_queries": suite.warmup_queries,
            "time_budget_seconds": suite.time_budget_seconds,
            "out": str(out),
        }
        config_path = run_dir / f"{cfg.slug}.config.json"
        config_path.write_text(json.dumps(config, indent=2))
        for attempt in range(1, suite.max_attempts + 1):
            if suite.cooldown_seconds > 0:  # let a fanless laptop shed heat between runs
                time.sleep(suite.cooldown_seconds)
            print(
                f"[{dt.datetime.now():%H:%M:%S}] running {cfg.slug} (attempt {attempt})", flush=True
            )
            out.unlink(missing_ok=True)  # a stale file from an earlier run must never count
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-m", "hnsw_engine.bench.worker", str(config_path)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - t0
            status = "ok" if proc.returncode == 0 and out.exists() else "failed"
            suspended = 0.0
            if status == "ok":
                suspended = float(json.loads(out.read_text()).get("suspended_seconds", 0.0))
                if suspended > suite.suspension_tolerance_seconds:
                    status = "suspended"
            manifest["runs"].append(
                {
                    "slug": cfg.slug,
                    "attempt": attempt,
                    "status": status,
                    "seconds": elapsed,
                    "suspended_seconds": suspended,
                    "returncode": proc.returncode,
                    "stderr_tail": proc.stderr[-2000:],
                }
            )
            print(f"    {status} in {elapsed:.1f}s", flush=True)
            if status == "failed":
                print(proc.stderr[-2000:], file=sys.stderr)
            if status != "suspended":
                break
            print(f"    machine slept for {suspended:.0f}s during the run; retrying", flush=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return run_dir
