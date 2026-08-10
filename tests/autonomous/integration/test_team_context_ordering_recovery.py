"""Regression evidence for the v1 Team/context ordering failure."""

from __future__ import annotations

import pytest

from src.autonomous.authorization import EmployeeAuthorizationScope
from src.autonomous.context import (
    ContextMessage,
    ContextUnavailableError,
    ContextUnavailableReason,
    EmployeeMessageScope,
    EmployeeThreadContext,
    MessagePage,
    ResolvedThread,
)


def _message(message_id: str, *, position: int, create_time_ms: int) -> ContextMessage:
    return ContextMessage(
        message_id=message_id,
        sender_id="ou_requester",
        sender_type="user",
        text=message_id,
        timestamp=create_time_ms / 1000,
        chat_id="oc_team",
        thread_id="omt_team",
        root_id="" if message_id == "om_root" else "om_root",
        parent_id="" if message_id == "om_root" else "om_root",
        sender_id_type="open_id",
        sender_tenant_key="tenant_1",
        create_time_ms=create_time_ms,
        update_time_ms=create_time_ms,
        message_position=position,
        thread_message_position=position,
    )


class _PositionCollisionLarkSource:
    scope = EmployeeMessageScope(
        tenant_key="tenant_1",
        agent_id="agt_worker",
        bot_principal_id="bot_worker",
        app_id="cli_worker",
        chat_id="oc_team",
        thread_root_message_id="om_root",
        current_message_id="om_current",
        source_requester_principal_id="ou_requester",
        authorization_scope=EmployeeAuthorizationScope.MANAGED_GROUP,
        feishu_thread_id="omt_team",
    )

    def resolve_thread(self) -> ResolvedThread:
        return ResolvedThread("om_root", "omt_team", "om_current")

    def list_thread_messages(self, *, page_token: str = "", page_size: int = 50):
        del page_token, page_size
        return MessagePage(
            (
                _message("om_root", position=0, create_time_ms=1_000),
                _message("om_current", position=2, create_time_ms=3_000),
            ),
            False,
        )

    def list_chat_messages(self, *, page_token: str = "", page_size: int = 20):
        del page_token, page_size
        # Lark can expose a group-window item with the same position as the
        # current thread item but a different message id. v1 rejects the whole
        # employee assignment before backend execution.
        return MessagePage(
            (_message("om_collision", position=2, create_time_ms=3_000),),
            False,
        )

    def reset_chat_traversal(self) -> None:
        pass


def test_v1_team_assignment_ordering_failure_prevents_backend_execution() -> None:
    source = _PositionCollisionLarkSource()
    backend_calls: list[str] = []

    with pytest.raises(ContextUnavailableError) as raised:
        EmployeeThreadContext(message_source=source).assemble()
        backend_calls.append("execute")

    assert raised.value.reason is ContextUnavailableReason.ORDERING
    assert backend_calls == []


def test_v2_contract_classifies_ordering_as_partial_not_team_terminal() -> None:
    from src.autonomous.gateway.coordinator import _TRANSIENT_CONTEXT_REASONS

    assert ContextUnavailableReason.ORDERING in _TRANSIENT_CONTEXT_REASONS
    assert ContextUnavailableReason.SCOPE not in _TRANSIENT_CONTEXT_REASONS
    assert ContextUnavailableReason.PERMISSION not in _TRANSIENT_CONTEXT_REASONS
    assert ContextUnavailableReason.CURRENT_MESSAGE not in _TRANSIENT_CONTEXT_REASONS
