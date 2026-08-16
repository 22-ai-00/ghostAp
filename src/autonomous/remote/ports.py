"""Backend-neutral ports for remote Agent dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .models import (
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteSnapshot,
    RemoteTaskHandle,
)


class RemoteAgentDispatchPort(Protocol):
    """Operations required from a durable remote-Agent transport adapter."""

    def dispatch(
        self,
        request: RemoteDispatchRequest,
    ) -> AsyncIterator[RemoteObservation]: ...

    async def get_task(self, handle: RemoteTaskHandle) -> RemoteSnapshot: ...

    def subscribe(
        self,
        handle: RemoteTaskHandle,
    ) -> AsyncIterator[RemoteObservation]: ...

    async def cancel(self, handle: RemoteTaskHandle) -> RemoteSnapshot: ...


__all__ = ["RemoteAgentDispatchPort"]
