"""Shared ACP transport lifecycle helpers."""

from __future__ import annotations

from typing import Any

from acp.task import InMemoryMessageQueue


class LateFrameTolerantMessageQueue(InMemoryMessageQueue):
    """Drop transport frames after connection shutdown begins.

    The Python SDK stops its dispatcher queue before it stops the receive loop.
    A provider can therefore deliver a final frame after the queue has closed.
    Those frames cannot be consumed and should not turn routine connection
    teardown into a receive-loop error.
    """

    def __init__(self, *, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._accepting = True

    async def publish(self, task: Any) -> None:
        if not self._accepting:
            return
        try:
            await super().publish(task)
        except RuntimeError:
            if not self._accepting:
                return
            raise

    async def close(self) -> None:
        self._accepting = False
        await super().close()
