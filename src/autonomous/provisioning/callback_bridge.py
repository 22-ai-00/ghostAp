"""Non-blocking synchronous callback bridge for async hire activities."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ConcurrentFuture
from typing import Any

from .external_mutation_gate import external_task_name


class CallbackBridgeError(RuntimeError):
    """A bridged callback failed without exposing callback arguments."""


Callback = Callable[..., object | Awaitable[object]]


class AsyncCallbackBridge:
    """Schedule synchronous SDK callbacks and durably drain them before return.

    Lark invokes registration callbacks synchronously.  The bridge therefore
    only enqueues callback work from that stack; the owning async activity
    later calls :meth:`drain` before it returns.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._mutex = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._pending: set[asyncio.Future[Any] | ConcurrentFuture[Any]] = set()

    def callback(
        self,
        target: Callback | None,
        *prefix: object,
    ) -> Callable[..., None]:
        """Return a synchronous, non-blocking callback accepted by the SDK."""

        def enqueue(*args: object) -> None:
            if target is None:
                return
            self._submit(target, *prefix, *args)

        return enqueue

    def _submit(self, target: Callback, *args: object) -> None:
        async def invoke() -> None:
            if inspect.iscoroutinefunction(target):
                result = target(*args)
            else:
                result = await asyncio.to_thread(target, *args)
            if inspect.isawaitable(result):
                await result

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            future: asyncio.Future[Any] | ConcurrentFuture[Any] = (
                self._loop.create_task(
                    invoke(),
                    name=external_task_name("registration-callback"),
                )
            )
        else:
            proxy: ConcurrentFuture[Any] = ConcurrentFuture()

            def start() -> None:
                coroutine = invoke()
                try:
                    task = self._loop.create_task(
                        coroutine,
                        name=external_task_name("registration-callback"),
                    )
                except BaseException as exc:
                    coroutine.close()
                    proxy.set_exception(exc)
                    return

                def completed(done: asyncio.Task[Any]) -> None:
                    if proxy.done():
                        return
                    try:
                        proxy.set_result(done.result())
                    except asyncio.CancelledError:
                        proxy.cancel()
                    except BaseException as exc:
                        proxy.set_exception(exc)

                task.add_done_callback(completed)

            future = proxy
            with self._mutex:
                self._pending.add(future)
            try:
                self._loop.call_soon_threadsafe(start)
            except BaseException:
                with self._mutex:
                    self._pending.discard(future)
                raise
            return
        with self._mutex:
            self._pending.add(future)

    async def drain(self) -> None:
        """Wait for every callback queued before the activity completes."""

        failed = False
        while True:
            with self._mutex:
                pending = tuple(self._pending)
                self._pending.clear()
            if not pending:
                if failed:
                    raise CallbackBridgeError(
                        "registration callback failed"
                    ) from None
                return
            awaitables = [
                future
                if isinstance(future, asyncio.Future)
                else asyncio.wrap_future(future, loop=self._loop)
                for future in pending
            ]
            results = await asyncio.gather(
                *awaitables,
                return_exceptions=True,
            )
            failed = failed or any(
                isinstance(result, BaseException) for result in results
            )


__all__ = ["AsyncCallbackBridge", "CallbackBridgeError"]
