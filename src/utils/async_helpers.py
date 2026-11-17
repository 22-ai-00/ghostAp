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
# There is one core need:
# ``run_async`` — run a coroutine to completion from a *synchronous* worker
# thread and block for the result, using one persistent event loop per thread.

_THREAD_LOCAL = threading.local()


def _reset_async_state_after_fork() -> None:
    """Discard thread-bound async state inherited by a forked child."""

    global _THREAD_LOCAL
    _THREAD_LOCAL = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_async_state_after_fork)


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
