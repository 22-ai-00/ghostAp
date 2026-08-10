"""Shared executor for the production I/O call site."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

_T = TypeVar("_T")

_IO_POOL: ThreadPoolExecutor | None = None
_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock


def get_io_pool() -> ThreadPoolExecutor:
    """Pool for I/O-bound work: card delivery, Feishu API, notifications."""
    global _IO_POOL
    if _IO_POOL is None:
        with _LOCK:
            if _IO_POOL is None:
                _IO_POOL = ThreadPoolExecutor(
                    max_workers=16,
                    thread_name_prefix="ghostap-io",
                )
    return _IO_POOL


def submit_io(fn: Callable[..., _T], *args, **kwargs) -> Future[_T]:
    return get_io_pool().submit(fn, *args, **kwargs)
