"""Global bounded ThreadPoolExecutor for card delivery I/O.

Provides a shared pool so that CardSession.dispatch() can submit delivery
work without blocking the caller thread. The pool is lazily initialised on
first access and gracefully shut down via shutdown hooks.
"""

from __future__ import annotations

import atexit
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_pool: ThreadPoolExecutor | None = None
_shutting_down = False


class _ShutdownSafePoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that degrades gracefully if submitted during shutdown."""

    def submit(self, fn, /, *args, **kwargs) -> Future[Any]:  # type: ignore[override]
        try:
            return super().submit(fn, *args, **kwargs)
        except RuntimeError as exc:
            logger.debug(
                "delivery pool submit skipped after shutdown (%s); running inline",
                exc,
            )
            future: Future[Any] = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as inner_exc:
                future.set_exception(inner_exc)
            else:
                future.set_result(result)
            return future


def get_delivery_pool() -> ThreadPoolExecutor:
    """Return the global delivery thread-pool (lazy init)."""
    global _pool
    if _pool is not None:
        return _pool
    with _lock:
        if _shutting_down:
            raise RuntimeError("delivery pool has been shut down")
        if _pool is not None:
            return _pool
        from src.config import get_settings

        max_workers = get_settings().card.delivery_pool_max_workers
        _pool = _ShutdownSafePoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="card-delivery",
        )
        logger.info("delivery pool created (max_workers=%d)", max_workers)
        return _pool


def shutdown_delivery_pool(wait: bool = True) -> None:
    """Shut down the global pool. Safe to call multiple times."""
    global _pool, _shutting_down
    with _lock:
        if _pool is None or _shutting_down:
            return
        _shutting_down = True
        pool = _pool
        _pool = None
    logger.info("shutting down delivery pool (wait=%s)", wait)
    pool.shutdown(wait=wait, cancel_futures=False)
    logger.info("delivery pool shutdown complete")


# Bounded-wait timeout for atexit safety net (consistent with
# DeliveryRegistry.drain_in_flight default timeout).
ATEXIT_TIMEOUT_SECONDS: float = 5.0


def _atexit_bounded_shutdown() -> None:
    """Safety-net shutdown with bounded wait to avoid hanging the process."""
    import threading as _th

    done = _th.Event()

    def _do_shutdown():
        shutdown_delivery_pool(wait=True)
        done.set()

    t = _th.Thread(target=_do_shutdown, daemon=True, name="delivery-pool-atexit")
    t.start()
    done.wait(timeout=ATEXIT_TIMEOUT_SECONDS)
    if not done.is_set():
        logger.warning(
            "delivery pool atexit: shutdown did not complete within %.1fs, giving up",
            ATEXIT_TIMEOUT_SECONDS,
        )


# Register atexit as a safety net (primary shutdown via utils/shutdown.py hooks)
atexit.register(_atexit_bounded_shutdown)
