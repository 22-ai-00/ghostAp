"""Cross-platform runtime memory tuning and heap-trimming helpers.

Designed for memory-constrained Linux and macOS deployments (e.g. 2C8G VMs):
- Caps default Python worker thread stack size to 1 MiB (down from typical
  8 MiB OS defaults) to reduce virtual address reservation and RSS pressure.
- On Linux glibc, bounds malloc arenas via ``mallopt(M_ARENA_MAX, ...)`` and
  periodically/explicitly returns freed heap pages via ``malloc_trim(0)``.
- On macOS or non-glibc platforms, OS-specific glibc calls degrade cleanly to
  harmless no-ops without errors.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default 1 MiB thread stack (safe well above Linux/macOS PTHREAD_STACK_MIN).
DEFAULT_THREAD_STACK_BYTES = 1024 * 1024
DEFAULT_MALLOC_ARENA_MAX = 2

# glibc <malloc.h> M_ARENA_MAX constant
_GLIBC_M_ARENA_MAX = -8

_STATE_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_LIBC_HANDLE: Any = None
_LIBC_PROBED = False
_LAST_PERIODIC_TRIM_AT: Optional[float] = None


def _parse_int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value >= minimum else default


def _get_glibc() -> Any:
    """Lazily probe Linux glibc handle cleanly; returns ``None`` on macOS/non-glibc."""
    global _LIBC_HANDLE, _LIBC_PROBED
    if not sys.platform.startswith("linux"):
        return None
    if _LIBC_PROBED:
        return _LIBC_HANDLE
    with _STATE_LOCK:
        if _LIBC_PROBED:
            return _LIBC_HANDLE
        try:
            libc = ctypes.CDLL("libc.so.6")
        except OSError:
            libc = None
        _LIBC_HANDLE = libc
        _LIBC_PROBED = True
        return _LIBC_HANDLE


def configure_runtime_memory_tuning(
    *,
    stack_size_bytes: Optional[int] = None,
    max_arenas: Optional[int] = None,
) -> dict[str, object]:
    """Apply cross-platform thread-stack and Linux glibc arena limits.

    Intended to be called early in process startup before worker threads spawn.
    Returns a small status dict describing which tunings were applied.
    """
    target_stack = (
        stack_size_bytes
        if stack_size_bytes is not None
        else _parse_int_env(
            "GHOSTAP_THREAD_STACK_BYTES",
            DEFAULT_THREAD_STACK_BYTES,
            minimum=0,
        )
    )
    stack_applied = False
    try:
        threading.stack_size(target_stack)
        stack_applied = True
    except (ValueError, RuntimeError, threading.ThreadError):
        logger.debug("Skipping custom threading.stack_size(%s)", target_stack, exc_info=True)

    target_arenas = (
        max_arenas
        if max_arenas is not None
        else _parse_int_env(
            "GHOSTAP_MALLOC_ARENA_MAX",
            DEFAULT_MALLOC_ARENA_MAX,
            minimum=1,
        )
    )
    mallopt_applied = False
    libc = _get_glibc()
    if libc is not None:
        mallopt_fn = getattr(libc, "mallopt", None)
        if callable(mallopt_fn):
            try:
                mallopt_fn.argtypes = [ctypes.c_int, ctypes.c_int]
                mallopt_fn.restype = ctypes.c_int
                mallopt_applied = bool(mallopt_fn(_GLIBC_M_ARENA_MAX, target_arenas))
            except Exception:
                logger.debug("Skipping glibc mallopt tuning", exc_info=True)

    return {
        "platform": sys.platform,
        "stack_size_bytes": target_stack if stack_applied else threading.stack_size(),
        "stack_size_applied": stack_applied,
        "max_arenas": target_arenas if mallopt_applied else None,
        "mallopt_applied": mallopt_applied,
    }


def trim_process_heap() -> bool:
    """Release unused heap pages back to the OS on Linux glibc; no-op elsewhere."""
    libc = _get_glibc()
    if libc is None:
        return False
    trim_fn = getattr(libc, "malloc_trim", None)
    if not callable(trim_fn):
        return False
    try:
        trim_fn.argtypes = [ctypes.c_size_t]
        trim_fn.restype = ctypes.c_int
        return bool(trim_fn(0))
    except Exception:
        return False


def maybe_trim_process_heap(*, min_interval_s: float = 60.0) -> bool:
    """Rate-limited heap trim suitable for post-task or idle-drain hooks."""
    global _LAST_PERIODIC_TRIM_AT
    now = time.monotonic()
    with _STATE_LOCK:
        if (
            min_interval_s > 0
            and _LAST_PERIODIC_TRIM_AT is not None
            and (now - _LAST_PERIODIC_TRIM_AT) < min_interval_s
        ):
            return False
        _LAST_PERIODIC_TRIM_AT = now
    return trim_process_heap()


__all__ = [
    "DEFAULT_MALLOC_ARENA_MAX",
    "DEFAULT_THREAD_STACK_BYTES",
    "configure_runtime_memory_tuning",
    "maybe_trim_process_heap",
    "trim_process_heap",
]
