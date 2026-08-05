"""Global thread pool registry for GhostAP.

Consolidates the ~15 scattered ThreadPoolExecutor instances into 3 shared pools:
- io_pool: Card delivery, Feishu API calls, notification sends
- compute_pool: ACP calls, AI inference waiting, classification
- background_pool: TTL cleanup, GC, health checks, heartbeats

Modules should use these shared pools via get_*_pool() instead of creating
their own ThreadPoolExecutor instances. Temporary fan-out pools (council and
perspective review) remain as-is since they're short-lived.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

_T = TypeVar("_T")

_IO_POOL: ThreadPoolExecutor | None = None
_COMPUTE_POOL: ThreadPoolExecutor | None = None
_BACKGROUND_POOL: ThreadPoolExecutor | None = None
_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

# Defaults — can be overridden via configure() before first use
_IO_WORKERS = 16
_COMPUTE_WORKERS = 8
_BACKGROUND_WORKERS = 4


def configure(
    *,
    io_workers: int | None = None,
    compute_workers: int | None = None,
    background_workers: int | None = None,
) -> None:
    """Configure pool sizes before first access. Must be called before any get_*_pool()."""
    global _IO_WORKERS, _COMPUTE_WORKERS, _BACKGROUND_WORKERS
    if io_workers is not None:
        _IO_WORKERS = io_workers
    if compute_workers is not None:
        _COMPUTE_WORKERS = compute_workers
    if background_workers is not None:
        _BACKGROUND_WORKERS = background_workers


def get_io_pool() -> ThreadPoolExecutor:
    """Pool for I/O-bound work: card delivery, Feishu API, notifications."""
    global _IO_POOL
    if _IO_POOL is None:
        with _LOCK:
            if _IO_POOL is None:
                _IO_POOL = ThreadPoolExecutor(
                    max_workers=_IO_WORKERS,
                    thread_name_prefix="ghostap-io",
                )
    return _IO_POOL


def get_compute_pool() -> ThreadPoolExecutor:
    """Pool for compute-bound waiting: ACP sessions, AI inference, NLI."""
    global _COMPUTE_POOL
    if _COMPUTE_POOL is None:
        with _LOCK:
            if _COMPUTE_POOL is None:
                _COMPUTE_POOL = ThreadPoolExecutor(
                    max_workers=_COMPUTE_WORKERS,
                    thread_name_prefix="ghostap-compute",
                )
    return _COMPUTE_POOL


def get_background_pool() -> ThreadPoolExecutor:
    """Pool for background housekeeping: TTL cleanup, GC, health checks."""
    global _BACKGROUND_POOL
    if _BACKGROUND_POOL is None:
        with _LOCK:
            if _BACKGROUND_POOL is None:
                _BACKGROUND_POOL = ThreadPoolExecutor(
                    max_workers=_BACKGROUND_WORKERS,
                    thread_name_prefix="ghostap-bg",
                )
    return _BACKGROUND_POOL


def submit_io(fn: Callable[..., _T], *args, **kwargs) -> Future[_T]:
    """Submit I/O-bound work to the shared pool."""
    return get_io_pool().submit(fn, *args, **kwargs)


def submit_compute(fn: Callable[..., _T], *args, **kwargs) -> Future[_T]:
    """Submit compute-waiting work to the shared pool."""
    return get_compute_pool().submit(fn, *args, **kwargs)


def submit_background(fn: Callable[..., _T], *args, **kwargs) -> Future[_T]:
    """Submit background housekeeping to the shared pool."""
    return get_background_pool().submit(fn, *args, **kwargs)


def shutdown_all(wait: bool = False) -> None:
    """Shut down all pools. Called during application exit."""
    global _IO_POOL, _COMPUTE_POOL, _BACKGROUND_POOL
    for pool in (_IO_POOL, _COMPUTE_POOL, _BACKGROUND_POOL):
        if pool is not None:
            pool.shutdown(wait=wait)
    _IO_POOL = None
    _COMPUTE_POOL = None
    _BACKGROUND_POOL = None
