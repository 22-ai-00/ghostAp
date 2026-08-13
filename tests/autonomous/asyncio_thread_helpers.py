"""Python 3.13-safe thread helpers for asyncio lifecycle tests."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


async def run_in_isolated_thread(
    func: Callable[..., T],
    /,
    *args: object,
    **kwargs: object,
) -> T:
    """Run blocking test work without retaining asyncio's default executor."""

    context = contextvars.copy_context()
    result: list[T] = []
    errors: list[BaseException] = []
    done = threading.Event()

    def invoke() -> None:
        try:
            result.append(context.run(func, *args, **kwargs))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(
        target=invoke,
        name="test-asyncio-isolated-thread",
    )
    worker.start()
    while not done.is_set():
        await asyncio.sleep(0.01)
    worker.join(timeout=1.0)
    if worker.is_alive():
        raise TimeoutError("isolated test thread did not stop")
    if errors:
        raise errors[0]
    return result[0]
