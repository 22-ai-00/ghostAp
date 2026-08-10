"""Fault injection for employee Thread Context and the zero-dispatch gate."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import replace

import pytest

from src.autonomous.context import (
    ContextPreparingExecutionPort,
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
    _Delegate,
    _Fence,
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


class _DispatchSpies(_Delegate):
    def __init__(self) -> None:
        super().__init__()
        self.task_commits = []
        self.acp_calls = []

    def execute(self, execution_input):
        self.task_commits.append(execution_input.request.current_message_id)
        self.acp_calls.append(execution_input.request.current_message_id)
        return super().execute(execution_input)


def _port(*, source_factory, backend=None, config=None):
    built = _composition(
        source_factory=source_factory,
        backend=backend,
        config=config,
    )
    delegate = _DispatchSpies()
    port = ContextPreparingExecutionPort(
        context_service=built.service,
        authority_fence=_Fence(),
        delegate=delegate,
    )
    return built, delegate, port


def _assert_zero_dispatch(delegate: _DispatchSpies) -> None:
    assert delegate.calls == []
    assert delegate.task_commits == []
    assert delegate.acp_calls == []


class _InTraversalMutationSource(_FakeSource):
    """Mutate the source after page one and before page two is returned."""

    def __init__(self, kind: str) -> None:
        super().__init__(traversals=[])
        self._kind = kind
        self._traversal = -1
        self.events: list[tuple[int, str]] = []
        self._root = _msg("om_root", "root", create=1_000, position=0)
        self._before = _msg("om_before", "before", create=2_000, position=1)
        self._current = _msg("om_current", "current", create=4_000, position=3)

    def list_thread_messages(
        self,
        *,
        page_token: str = "",
        page_size: int = 50,
    ) -> MessagePage:
        del page_size
        self.thread_calls += 1
        if not page_token:
            self._traversal += 1
            self.events.append((self._traversal, "page1"))
            return MessagePage((self._root, self._before), True, "next")

        assert page_token == "next"
        inject_mutation = self._traversal % 2 == 0
        messages = []
        if inject_mutation:
            self.events.append((self._traversal, self._kind))
            if self._kind == "insert":
                messages.append(
                    _msg("om_inserted", "inserted", create=3_000, position=2)
                )
            elif self._kind == "edit":
                messages.append(
                    replace(
                        self._before,
                        text="edited",
                        update_time_ms=2_500,
                        edited=True,
                    )
                )
            else:
                messages.append(
                    replace(
                        self._before,
                        text="",
                        update_time_ms=2_500,
                        deleted=True,
                    )
                )
        else:
            self.events.append((self._traversal, "baseline"))
        messages.append(self._current)
        return MessagePage(tuple(messages), False)


@pytest.mark.parametrize("mutation", ["insert", "edit", "delete"])
def test_paging_mutation_fails_revision_and_never_dispatches(
    mutation: str,
) -> None:
    source = _InTraversalMutationSource(mutation)
    source_factory = _SourceFactory(source)
    built, delegate, port = _port(source_factory=source_factory)

    with pytest.raises(ContextUnavailableError) as raised:
        port.execute(_request(), tool="codex", model="gpt", effort="high")

    assert raised.value.reason is ContextUnavailableReason.REVISION
    _assert_zero_dispatch(delegate)
    assert built.source_factory.close_calls == 1
    assert source.thread_calls == 8
    assert source.events == [
        (0, "page1"),
        (0, mutation),
        (1, "page1"),
        (1, "baseline"),
        (2, "page1"),
        (2, mutation),
        (3, "page1"),
        (3, "baseline"),
    ]


class _SlowSource(_FakeSource):
    def list_thread_messages(self, **kwargs):
        time.sleep(0.02)
        return super().list_thread_messages(**kwargs)


def test_page_timeout_never_dispatches() -> None:
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
    _built, delegate, port = _port(
        source_factory=source_factory,
        config=ThreadContextConfig(fetch_timeout_seconds=0.001),
    )

    with pytest.raises(ContextUnavailableError) as raised:
        port.execute(_request(), tool="codex", model="gpt", effort="high")

    assert raised.value.reason is ContextUnavailableReason.DEADLINE
    _assert_zero_dispatch(delegate)


class _FailingGroupBackend(_GroupBackend):
    def read_group_memory(self, chat_id: str) -> str:
        super().read_group_memory(chat_id)
        raise RuntimeError("unsafe group backend detail")


def test_group_read_failure_stops_before_source_and_dispatch() -> None:
    source_factory = _SourceFactory(_FakeSource(traversals=[]))
    built, delegate, port = _port(
        source_factory=source_factory,
        backend=_FailingGroupBackend(),
    )

    with pytest.raises(ContextUnavailableError) as raised:
        port.execute(_request(), tool="codex", model="gpt", effort="high")

    assert raised.value.reason is ContextUnavailableReason.MEMORY
    assert built.source_factory.calls == []
    _assert_zero_dispatch(delegate)
