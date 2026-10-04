"""Capture the hardware and software environment that produced a result.

Every results file embeds this dictionary, so each number can be traced to the machine, library
versions and thread settings it came from.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)

PACKAGES = (
    "numpy",
    "numba",
    "llvmlite",
    "faiss-cpu",
    "hnswlib",
    "h5py",
    "psutil",
    "threadpoolctl",
    "hnsw-engine",
)


def single_thread_environment() -> dict[str, str]:
    """Environment variables that cap every common threading runtime at one thread.

    They must be set before NumPy, FAISS or Numba are imported, so workers receive them from the
    parent process.
    """
    return dict.fromkeys(THREAD_VARIABLES, "1")


def ensure_single_thread() -> None:
    """Re-execute the running script with :func:`single_thread_environment` if it is not set.

    The caps only take effect when set before NumPy loads its BLAS, so a script that times
    NumPy or FAISS code calls this first thing in ``main``. It returns at once when every cap is
    already in place (for example in the re-executed process).
    """
    wanted = single_thread_environment()
    if all(os.environ.get(name) == value for name, value in wanted.items()):
        return
    os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, **wanted})


def capture_environment(extra_packages: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return a JSON-serialisable description of the current machine and process."""
    env: dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "physical_cores": _sysctl_int("hw.physicalcpu"),
        "performance_cores": _sysctl_int("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl_int("hw.perflevel1.physicalcpu"),
        "ram_bytes": _ram_bytes(),
        "packages": {name: _version(name) for name in PACKAGES + extra_packages},
        "thread_env": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "threadpools": _threadpools(),
    }
    if platform.system() == "Darwin":
        env["macos_version"] = platform.mac_ver()[0]
        env["power"] = power_state()
        env["activity"] = machine_activity()
    return env


def machine_activity() -> dict[str, Any]:
    """How busy the machine was: load averages and seconds since the last keyboard or mouse
    input (macOS ``ioreg``). Timings taken while someone uses the laptop run slower, and this
    record makes that visible."""
    idle = None
    report = _run(["ioreg", "-c", "IOHIDSystem"])
    if report:
        for line in report.splitlines():
            if '"HIDIdleTime"' in line:
                idle = int(line.split("=")[-1].strip()) / 1e9
                break
    return {"load_average_1_5_15_min": list(os.getloadavg()), "user_idle_seconds": idle}


def power_state() -> dict[str, str | None]:
    """Power source, Low Power Mode and thermal/performance warnings (macOS ``pmset``).

    Laptop timings depend on these: a fanless machine slows down when hot, and results are only
    comparable when taken under similar conditions.
    """
    batt = _run(["pmset", "-g", "batt"])
    settings = _run(["pmset", "-g"])
    therm = _run(["pmset", "-g", "therm"])
    source = None
    if batt:
        first = batt.splitlines()[0]
        source = first.split("'")[1] if "'" in first else first
    low_power = None
    if settings:
        for line in settings.splitlines():
            if "lowpowermode" in line:
                low_power = line.split()[-1]
    return {
        "source": source,
        "low_power_mode": low_power,
        "thermal": " | ".join(line.strip() for line in therm.splitlines()) if therm else None,
    }


def _run(command: list[str]) -> str | None:
    try:
        out = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _sysctl(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _sysctl_int(name: str) -> int | None:
    value = _sysctl(name)
    return int(value) if value is not None and value.isdigit() else None


def _cpu_model() -> str:
    model = _sysctl("machdep.cpu.brand_string")
    if model:
        return model
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    value = _sysctl_int("hw.memsize")
    if value is not None:
        return value
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def _threadpools() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    keep = ("user_api", "internal_api", "num_threads", "version")
    return [{k: info.get(k) for k in keep} for info in threadpool_info()]
