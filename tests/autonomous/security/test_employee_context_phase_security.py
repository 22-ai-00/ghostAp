"""Phase-2 security evidence through the real Context execution gate."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest

from src.autonomous.context import (
    ContextPreparingExecutionPort,
    ContextUnavailableError,
    ContextUnavailableReason,
    ThreadContextConfig,
)
from tests.autonomous.helpers import (
    FakeEmployeeMessageSource as _FakeSource,
)
from tests.autonomous.helpers import (
    make_context_message as _msg,
)
from tests.autonomous.helpers import (
    message_pages as _pages,
)
from tests.autonomous.integration.test_employee_context_service import (
    _composition,
    _Delegate,
    _Fence,
    _request,
)


class _SourceFactory:
    def __init__(self, source: _FakeSource) -> None:
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


def _execution_port(source_factory: _SourceFactory, *, config=None):
    built = _composition(source_factory=source_factory, config=config)
    delegate = _Delegate()
    port = ContextPreparingExecutionPort(
        context_service=built.service,
        authority_fence=_Fence(),
        delegate=delegate,
    )
    return built, delegate, port


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            {"sender_tenant_key": "tenant_2"},
            ContextUnavailableReason.SCOPE,
        ),
        ({"chat_id": "oc_other"}, ContextUnavailableReason.SCOPE),
        (
            {"thread_id": "omt_other"},
            ContextUnavailableReason.ROOT_THREAD_BINDING,
        ),
        (
            {"root_id": "om_other"},
            ContextUnavailableReason.ROOT_THREAD_BINDING,
        ),
    ],
)
def test_cross_tenant_chat_thread_returns_fail_closed_and_never_dispatch(
    mutation: dict[str, str],
    reason: ContextUnavailableReason,
) -> None:
    root = _msg("om_root", "root", create=1_000, position=0)
    current = replace(
        _msg("om_current", "current", create=3_000, position=1),
        **mutation,
    )
    source_factory = _SourceFactory(
        _FakeSource(
            traversals=[
                _pages([root, current]),
                _pages([root, current]),
            ]
        )
    )
    built, delegate, port = _execution_port(source_factory)

    with pytest.raises(ContextUnavailableError) as raised:
        port.execute(_request(), tool="codex", model="gpt", effort="high")

    assert raised.value.reason is reason
    assert delegate.calls == []
    assert built.source_factory.close_calls == 1


@pytest.mark.parametrize(
    "config",
    [
        ThreadContextConfig(max_context_chars=3),
        ThreadContextConfig(
            max_context_chars=100,
            max_context_tokens=4,
            tokens_per_char=1,
        ),
        ThreadContextConfig(
            max_context_chars=100,
            max_context_tokens=5,
            tokens_per_char=1,
        ),
    ],
)
def test_oversized_protected_current_fails_budget_and_never_dispatches(
    config: ThreadContextConfig,
) -> None:
    root = _msg("om_root", "", create=1_000, position=0)
    current = _msg("om_current", "12345", create=3_000, position=1)
    source_factory = _SourceFactory(
        _FakeSource(
            traversals=[
                _pages([root, current]),
                _pages([root, current]),
            ]
        )
    )
    built, delegate, port = _execution_port(
        source_factory,
        config=config,
    )

    with pytest.raises(ContextUnavailableError) as raised:
        port.execute(_request(), tool="codex", model="gpt", effort="high")

    assert raised.value.reason is ContextUnavailableReason.BUDGET
    assert delegate.calls == []
    assert built.source_factory.close_calls == 1
