"""Control and observe the operating system's page cache for one file, without root.

Cold-cache timings need files that are not in memory. Dropping the whole cache needs
administrator rights (``sudo purge`` on macOS), but a process may evict the pages of a file it can
read: on Linux with ``posix_fadvise(POSIX_FADV_DONTNEED)``, on macOS with
``msync(MS_INVALIDATE)`` over a mapping of the file (what ``vmtouch -e`` does there). ``mincore``
reports which pages of a mapping are resident, so the effect can be checked rather than assumed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
_PROT_READ, _MAP_SHARED, _MS_INVALIDATE = 1, 1, 2
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def _with_mapping(path: Path, action: Callable[[int, int], T]) -> T:
    size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        address = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if address is None or address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), f"mmap failed for {path}")
        try:
            return action(address, size)
        finally:
            _libc.munmap(address, size)
    finally:
        os.close(fd)


def resident_fraction(path: str | Path) -> float:
    """Share of the file's pages currently in memory (``mincore``); 1.0 for an empty file."""
    path = Path(path)
    if path.stat().st_size == 0:
        return 1.0

    def count(address: int, size: int) -> float:
        pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        flags = ctypes.create_string_buffer(pages)
        if _libc.mincore(address, size, flags) != 0:
            raise OSError(ctypes.get_errno(), f"mincore failed for {path}")
        return sum(byte & 1 for byte in flags.raw) / pages

    return _with_mapping(path, count)


def evict(path: str | Path) -> None:
    """Ask the operating system to drop the file's pages from its cache.

    The file is flushed first, so that only clean pages are dropped and no write can be lost.
    """
    path = Path(path)
    if path.stat().st_size == 0:
        return
    fadvise = getattr(os, "posix_fadvise", None)  # Linux; absent on macOS
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        if fadvise is not None and dontneed is not None:
            fadvise(fd, 0, 0, dontneed)
            return
    finally:
        os.close(fd)

    def invalidate(address: int, size: int) -> None:
        if _libc.msync(address, size, _MS_INVALIDATE) != 0:
            raise OSError(ctypes.get_errno(), f"msync failed for {path}")

    _with_mapping(path, invalidate)


def warm(path: str | Path) -> None:
    """Read the whole file once, so its pages are in the cache."""
    with open(path, "rb") as handle:
        while handle.read(1 << 24):
            pass


def pageins() -> int | None:
    """Pages read from disk into memory since boot, system-wide (macOS ``vm_stat``)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Pageins:\s+(\d+)", out)
    return int(match.group(1)) if match else None
