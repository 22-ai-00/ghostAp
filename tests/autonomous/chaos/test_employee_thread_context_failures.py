"""Fault injection for employee Thread Context and the zero-dispatch gate."""

from __future__ import annotations

import time
from contextlib import contextmanager

import pytest

from src.autonomous.context import (
    ContextUnavailableError,
    ContextUnavailableReason,
    MessagePage,
    ThreadContextConfig,
)
from tests.autonomous.helpers import (
    FakeEmployeeMessageSource as _FakeSource,
)
from tests.autonomous.helpers import (
    make_context_message as _msg,
)
from tests.autonomous.integration.test_employee_context_service import (
    _composition,
    _GroupBackend,
    _request,
)


class _SourceFactory:
    def __init__(self, source) -> None:
        self.source = source
        self.calls = []
        self.close_calls = 0

    @contextmanager
    def open(self, *, scope, principal):
        self.calls.append((scope, principal))
        self.source.scope = scope
        try:
            yield self.source
        finally:
            self.close_calls += 1


class _SlowSource(_FakeSource):
    def list_thread_messages(self, **kwargs):
        time.sleep(0.02)
        return super().list_thread_messages(**kwargs)


def test_page_timeout_fails_closed() -> None:
    thread = [
        _msg("om_root", "root", create=1_000, position=0),
        _msg("om_current", "current", create=3_000, position=1),
    ]
    source_factory = _SourceFactory(
        _SlowSource(
            traversals=[
                [MessagePage(tuple(thread), False)],
                [MessagePage(tuple(thread), False)],
            ]
        )
    )
    built = _composition(
        source_factory=source_factory,
        config=ThreadContextConfig(fetch_timeout_seconds=0.001),
    )

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(_request())

    assert raised.value.reason is ContextUnavailableReason.DEADLINE
    assert built.source_factory.close_calls == 1


class _FailingGroupBackend(_GroupBackend):
    def read_group_memory(self, chat_id: str) -> str:
        super().read_group_memory(chat_id)
        raise RuntimeError("unsafe group backend detail")


def test_group_read_failure_stops_before_source() -> None:
    source_factory = _SourceFactory(_FakeSource(traversals=[]))
    built = _composition(
        source_factory=source_factory,
        backend=_FailingGroupBackend(),
    )

    with pytest.raises(ContextUnavailableError) as raised:
        built.service.assemble(_request())

    assert raised.value.reason is ContextUnavailableReason.MEMORY
    assert built.source_factory.calls == []
