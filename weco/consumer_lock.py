"""A per-working-tree lock guaranteeing a single evaluation consumer.

The optimization loop evaluates candidates by swapping their code into the
working directory, running the eval command, then restoring the originals
(``run_evaluation_with_files_swap``). Two consumers doing that concurrently in
the same tree clobber each other's files mid-eval. With parallel derived runs
feeding one lineage queue, the invariant we need is: **at most one consumer per
working tree.**

This module enforces it with an advisory ``flock`` on ``.weco/consumer.lock``
in the working directory. Acquisition is non-blocking (try-lock): a second
consumer that finds the lock held does *not* start a competing loop — it leaves
the work to the consumer already draining the lineage.

POSIX only. On platforms without ``fcntl`` the lock is a best-effort no-op
(``acquired=True``) — parallel derive there falls back to today's behavior.
"""

from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from typing import Iterator, Optional

try:
    import fcntl  # POSIX
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


# Sentinel handle returned when advisory locking is unavailable (non-POSIX):
# we report "acquired" so behavior falls back to today's, but hold no real lock.
_NOOP_HANDLE = -1


def _lock_path(workdir: Optional[str]) -> pathlib.Path:
    base = pathlib.Path(workdir) if workdir else pathlib.Path.cwd()
    return base / ".weco" / "consumer.lock"


def try_acquire(workdir: Optional[str] = None) -> Optional[int]:
    """Non-blocking acquire of the working-tree consumer lock.

    Returns an opaque handle if this process now holds the lock, or ``None`` if
    another consumer already holds it. Pass the handle to :func:`release`. On
    platforms without ``fcntl`` this always "succeeds" (best-effort no-op).
    """
    if fcntl is None:
        return _NOOP_HANDLE
    path = _lock_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return None
    return fd


def release(handle: Optional[int]) -> None:
    """Release a lock handle returned by :func:`try_acquire`. Safe on None."""
    if handle is None or handle == _NOOP_HANDLE or fcntl is None:
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        try:
            os.close(handle)
        except OSError:
            pass


@contextmanager
def consumer_lock(workdir: Optional[str] = None) -> Iterator[bool]:
    """Context-manager form of :func:`try_acquire`.

    Yields ``True`` if this process now holds the lock (and should consume), or
    ``False`` if another consumer already holds it. Released on exit. Never blocks.
    """
    handle = try_acquire(workdir)
    try:
        yield handle is not None
    finally:
        release(handle)
