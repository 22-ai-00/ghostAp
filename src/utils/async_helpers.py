"""Async helper utilities — safe wrappers around asyncio primitives."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Awaitable, TypeVar

__all__ = ["run_async", "safe_wait_for"]

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Thread-safe async bridge — replaces scattered _get_shared_loop() / asyncio.run()
# ---------------------------------------------------------------------------
#
# There are two distinct needs:
#
# 1. ``run_async`` — run a coroutine to completion from a *synchronous* worker
#    thread and block for the result. Each worker thread gets its OWN persistent
#    event loop (thread-local) and drives it with ``run_until_complete``. This
#    restores the pre-refactor parallelism (the old ``asyncio.run()`` created a
#    fresh loop per call in the caller's own thread) WITHOUT the earlier
#    single-shared-loop bottleneck, where independent operations — ACP model
#    probes, coco probes, slock autonomous-resolve — serialized head-of-line
#    behind one loop. Loops are still centralized and reused (one per worker
#    thread), so we don't reintroduce "scattered competing loops".
#
# 2. ``_get_bridge_loop`` — a single persistent background loop for callers that
#    submit work from another thread via ``run_coroutine_threadsafe`` (slock NLI
#    classification, slock engine autonomous resolve). These need a loop that is
#    *already running* in a separate thread.

_BRIDGE_LOOP: asyncio.AbstractEventLoop | None = None
_BRIDGE_THREAD: threading.Thread | None = None
_BRIDGE_PID: int | None = None
_BRIDGE_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
_BRIDGE_START_TIMEOUT_SECONDS = 1.0

_THREAD_LOCAL = threading.local()


def _reset_async_state_after_fork() -> None:
    """Discard thread-bound async state inherited by a forked child."""

    global _BRIDGE_LOOP, _BRIDGE_THREAD, _BRIDGE_PID, _BRIDGE_LOCK, _THREAD_LOCAL
    _BRIDGE_LOOP = None
    _BRIDGE_THREAD = None
    _BRIDGE_PID = None
    # A lock held by a vanished parent thread can never be released in the
    # child, so it must be recreated alongside the loop metadata.
    _BRIDGE_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
    _THREAD_LOCAL = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_async_state_after_fork)


def _get_bridge_loop() -> asyncio.AbstractEventLoop:
    """Get or create the singleton bridge event loop running in a daemon thread.

    Intended for ``run_coroutine_threadsafe`` callers that need a persistent,
    already-running loop in a separate thread. NOT used by ``run_async`` (which
    uses a per-thread loop to avoid head-of-line blocking).
    """
    global _BRIDGE_LOOP, _BRIDGE_THREAD, _BRIDGE_PID, _BRIDGE_LOCK
    current_pid = os.getpid()

    # Defensive fallback for runtimes that do not invoke register_at_fork
    # hooks.  This check intentionally happens before acquiring the inherited
    # lock, which may have been held by a parent thread at fork time.
    if _BRIDGE_PID is not None and _BRIDGE_PID != current_pid:
        _BRIDGE_LOOP = None
        _BRIDGE_THREAD = None
        _BRIDGE_PID = None
        _BRIDGE_LOCK = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock

    with _BRIDGE_LOCK:
        if (
            _BRIDGE_PID == current_pid
            and _BRIDGE_LOOP is not None
            and _BRIDGE_THREAD is not None
            and _BRIDGE_THREAD.is_alive()
            and _BRIDGE_LOOP.is_running()
        ):
            return _BRIDGE_LOOP

        loop = asyncio.new_event_loop()
        started = threading.Event()
        abort_startup = threading.Event()

        def _run():
            asyncio.set_event_loop(loop)
            # Queue the readiness signal before run_forever so startup does not
            # depend on the cross-thread self-pipe wakeup path.
            loop.call_soon(started.set)
            loop.call_soon(
                lambda: loop.stop() if abort_startup.is_set() else None
            )
            try:
                loop.run_forever()
            finally:
                if abort_startup.is_set() and not loop.is_closed():
                    loop.close()

        t = threading.Thread(target=_run, name="async-bridge", daemon=True)
        t.start()

        if (
            not started.wait(timeout=_BRIDGE_START_TIMEOUT_SECONDS)
            or not t.is_alive()
            or not loop.is_running()
        ):
            abort_startup.set()
            if loop.is_running():
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass
            t.join(timeout=0.05)
            if not t.is_alive():
                if not loop.is_closed():
                    loop.close()
            raise RuntimeError("async bridge event loop failed to start")

        _BRIDGE_LOOP = loop
        _BRIDGE_THREAD = t
        _BRIDGE_PID = current_pid
        return loop


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent event loop bound to the current thread.

    Reused across calls on the same thread to avoid per-call loop
    creation/teardown, while keeping loops isolated between threads so that
    concurrent ``run_async`` callers never serialize behind one another.
    """
    loop = getattr(_THREAD_LOCAL, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _THREAD_LOCAL.loop = loop
    return loop


def run_async(coro: Awaitable[_T], *, timeout: float | None = None) -> _T:
    """Run an async coroutine from synchronous code, thread-safely.

    Drives a per-thread event loop with ``run_until_complete`` so that
    independent worker threads execute their coroutines concurrently instead of
    serializing on a single shared loop.

    Must NOT be called from a thread that already has a running event loop.

    Parameters
    ----------
    coro:
        The coroutine to execute.
    timeout:
        Optional timeout in seconds. Raises TimeoutError if exceeded.
    """
    loop = _get_thread_loop()

    if timeout is None:
        return loop.run_until_complete(coro)  # type: ignore[arg-type]

    async def _with_timeout() -> _T:
        return await asyncio.wait_for(coro, timeout=timeout)

    try:
        return loop.run_until_complete(_with_timeout())
    except (asyncio.TimeoutError, TimeoutError) as exc:
        msg = str(exc).strip() or f"操作超时 ({timeout}s)"
        raise TimeoutError(msg) from exc


async def safe_wait_for(
    coro: "asyncio.coroutines" | asyncio.Future,  # type: ignore[type-arg]
    timeout: float,
    *,
    action: str = "",
) -> _T:
    """``asyncio.wait_for`` wrapper that guarantees a non-empty TimeoutError message.

    ``asyncio.wait_for`` raises ``asyncio.TimeoutError()`` with **no message**
    (``str(e) == ""``).  Down-stream formatting helpers (``fmt_error``,
    ``get_error_detail``) already guard against this, but wrapping at source is
    the cheapest defence.

    Parameters
    ----------
    coro:
        Awaitable to run with a deadline.
    timeout:
        Seconds before cancellation.
    action:
        Human-readable label injected into the ``TimeoutError`` message when
        the original exception carries no text.  Example: ``"ACP 健康检查"``.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        msg = str(exc).strip()
        if not msg:
            label = action or "操作"
            msg = f"{label}超时 ({timeout}s)"
        raise TimeoutError(msg) from exc
