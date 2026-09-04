"""Runtime lock-ordering violation detector.

Implements the partial order defined in ``docs/adr-lock-ordering.md``::

    BaseEngineManager._lock  (level -1, outermost)
      → BaseEngine._lock  (level 0)
        → SystemHandler._acp_activation_lock  (level 1)
          → ProjectManager._lock  (level 2)
            → ProjectContext._chat_lock  (level 3)
              → ChatLockManager._mu  (level 4)
                → RepoLockManager._mu  (level 5, innermost)

Each thread tracks the *highest* level it currently holds.  Acquiring a lock
whose level is ≤ the current max constitutes a potential deadlock and is
logged as a warning.

Usage::

    from src.utils.lock_order import ordered_lock, LockLevel

    class MyManager:
        def __init__(self):
            self._mu = ordered_lock(LockLevel.REPO_LOCK)

        def do_work(self):
            with self._mu:
                ...

The wrapper is zero-cost when ``GHOSTAP_LOCK_ORDER_CHECK`` env-var is unset
or falsy (the default).  Set it to ``1`` to enable checking in development /
CI.
"""

from __future__ import annotations

import logging
import os
import threading
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lock levels — lower numeric value = outer (acquired first).
# ---------------------------------------------------------------------------


class LockLevel(IntEnum):
    """Canonical lock ordering levels.

    Values correspond to the ADR-defined partial order.  Lower = outer.
    """

    ENGINE_MANAGER = -1  # BaseEngineManager._lock (Lock)
    ENGINE_INSTANCE = 0  # BaseEngine._lock (RLock)
    ACP_ACTIVATION = 1  # SystemHandler._acp_activation_lock (RLock)
    PROJECT_MANAGER = 2  # ProjectManager._lock (RLock)
    CHAT_LOCK_CTX = 3  # ProjectContext._chat_lock
    CHAT_LOCK_MGR = 4  # ChatLockManager._mu
    REPO_LOCK = 5  # RepoLockManager._mu


# ---------------------------------------------------------------------------
# Thread-local bookkeeping
# ---------------------------------------------------------------------------

_tls = threading.local()

_ENABLED: Optional[bool] = None
_STRICT: Optional[bool] = None


def _is_enabled() -> bool:
    """Return True when lock-order checking is active (cached).

    NOTE: Intentionally uses os.environ.get instead of Settings — this module
    is imported at lock-definition time, potentially before Settings is initialized.
    """
    global _ENABLED
    if _ENABLED is None:
        val = os.environ.get("GHOSTAP_LOCK_ORDER_CHECK", "").strip().lower()
        _ENABLED = val in {"1", "true", "yes", "strict"}
    return _ENABLED


def _is_strict() -> bool:
    """Return True when strict mode is active (violations raise RuntimeError)."""
    global _STRICT
    if _STRICT is None:
        val = os.environ.get("GHOSTAP_LOCK_ORDER_CHECK", "").strip().lower()
        _STRICT = val == "strict"
    return _STRICT


def enable_lock_order_check(*, strict: bool = False) -> None:
    """Programmatically enable lock-order checking (useful for tests)."""
    global _ENABLED, _STRICT
    _ENABLED = True
    _STRICT = strict


def disable_lock_order_check() -> None:
    """Programmatically disable lock-order checking."""
    global _ENABLED, _STRICT
    _ENABLED = False
    _STRICT = False


def _get_held() -> list[int]:
    """Return the per-thread held-levels stack."""
    held = getattr(_tls, "held", None)
    if held is None:
        held = []
        _tls.held = held
    return held


def _on_acquire(level: int, name: str) -> None:
    """Called just before a lock is acquired."""
    held = _get_held()
    if held:
        max_held = max(held)
        if level <= max_held:
            msg = (
                f"Lock ordering violation: acquiring {name} (level={level}) while holding level={max_held} "
                f"(thread={threading.current_thread().name}). See docs/adr-lock-ordering.md."
            )
            if _is_strict():
                raise RuntimeError(msg)
            logger.warning(msg)
    held.append(level)


def _on_release(level: int) -> None:
    """Called just after a lock is released."""
    held = _get_held()
    try:
        held.remove(level)
    except ValueError:
        logger.debug("lock level %d not found in held list", level, exc_info=True)


# ---------------------------------------------------------------------------
# Ordered lock wrapper
# ---------------------------------------------------------------------------


class _OrderedLock:
    """Thin wrapper around ``threading.Lock`` with ordering checks."""

    __slots__ = ("_lock", "_level", "_name")

    def __init__(self, level: LockLevel, name: str = ""):
        self._lock = threading.Lock()
        self._level = int(level)
        self._name = name or level.name

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if _is_enabled():
            _on_acquire(self._level, self._name)
        return self._lock.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        try:
            self._lock.release()
        finally:
            if _is_enabled():
                _on_release(self._level)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

    # threading.Lock protocol
    def locked(self) -> bool:
        return self._lock.locked()


class _OrderedRLock:
    """Thin wrapper around ``threading.RLock`` with ordering checks.

    Only checks ordering on the *first* (outermost) acquisition per thread.
    """

    __slots__ = ("_lock", "_level", "_name", "_owners")

    def __init__(self, level: LockLevel, name: str = ""):
        self._lock = threading.RLock()
        self._level = int(level)
        self._name = name or level.name
        self._owners: dict[int, int] = {}  # thread-ident → depth

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        tid = threading.get_ident()
        depth = self._owners.get(tid, 0)
        if depth == 0 and _is_enabled():
            _on_acquire(self._level, self._name)
        result = self._lock.acquire(blocking=blocking, timeout=timeout)
        if result:
            self._owners[tid] = depth + 1
        return result

    def release(self) -> None:
        tid = threading.get_ident()
        depth = self._owners.get(tid, 1)
        try:
            self._lock.release()
        finally:
            new_depth = depth - 1
            if new_depth <= 0:
                self._owners.pop(tid, None)
                if _is_enabled():
                    _on_release(self._level)
            else:
                self._owners[tid] = new_depth

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


def ordered_lock(level: LockLevel, *, name: str = "") -> _OrderedLock:
    """Create a ``threading.Lock``-compatible object with ordering checks."""
    return _OrderedLock(level, name)


def ordered_rlock(level: LockLevel, *, name: str = "") -> _OrderedRLock:
    """Create a ``threading.RLock``-compatible object with ordering checks."""
    return _OrderedRLock(level, name)
