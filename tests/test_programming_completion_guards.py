"""Regression contracts for ordinary programming task completion."""

from __future__ import annotations

import pytest

from src.acp.models import (
    ACPEvent,
    ACPEventType,
    ACPImageInfo,
    PlanEntryInfo,
    PlanInfo,
    PromptResult,
    ToolCallInfo,
)
from src.acp.outcome import PromptOutcome, classify_prompt_result
from src.card.delivery.engine import CardDelivery
from src.card.programming_adapter import (
    ProgrammingCardSession,
    build_programming_metadata,
)
from src.card.session import CardSession
from src.card.session.config import SessionConfig


class _CardClient:
    def __init__(self) -> None:
        self.created = 0

    def create_card(
        self,
        chat_id,
        card_json,
        *,
        reply_to=None,
        reply_in_thread=None,
        idempotency_key=None,
    ):
        self.created += 1
        return f"msg-{self.created}", f"card-{self.created}"

    def update_card(self, card_id, card_json, *, sequence=0) -> None:
        return None

    def update_element(self, card_id, element_id, content, *, sequence=0) -> None:
        return None


def _card_session(
    delivery: CardDelivery,
    *,
    session_id: str,
    metadata=None,
) -> CardSession:
    return CardSession(
        chat_id="chat-programming",
        config=SessionConfig(
            metadata=metadata or build_programming_metadata("codex"),
            reply_to="origin-message",
            sync_delivery=True,
        ),
        delivery=delivery,
        session_id=session_id,
    )


def _agent_event(event_type: ACPEventType, *, status: str) -> ACPEvent:
    return ACPEvent(
        event_type=event_type,
        tool_call=ToolCallInfo(
            id="agent-call-1",
            title="agent",
            kind="other",
            status=status,
            content="分析剩余实现\n子代理：Explore",
        ),
    )


def test_agent_completion_updates_main_summary_without_completing_parent() -> None:
    """A subtask TOOL_CALL_DONE only updates the main-card summary."""
    client = _CardClient()
    delivery = CardDelivery(client)
    metadata = build_programming_metadata(
        "codex",
        project_name="ghostAp",
        working_dir="/repo",
    )
    parent = _card_session(delivery, session_id="parent", metadata=metadata)
    programming = ProgrammingCardSession(
        parent,
        base_metadata=metadata,
    )
    try:
        programming.start()
        programming.on_event(
            _agent_event(ACPEventType.TOOL_CALL_START, status="in_progress")
        )
        programming.on_event(
            _agent_event(ACPEventType.TOOL_CALL_DONE, status="completed")
        )

        assert client.created == 1
        assert parent.state is not None
        assert parent.state.terminal == "running"
        assert parent.state.metadata.subagents[0]["status"] == "completed"

        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TEXT_CHUNK,
                text="父任务继续输出",
            )
        )
        programming._flush_now()
        assert "父任务继续输出" in programming.get_final_text()
    finally:
        programming.abort()


def test_agent_call_uses_main_summary_not_tool_or_child_card() -> None:
    client = _CardClient()
    delivery = CardDelivery(client)
    parent = _card_session(delivery, session_id="parent-no-factory")
    programming = ProgrammingCardSession(parent)
    try:
        programming.start()
        programming.on_event(
            _agent_event(ACPEventType.TOOL_CALL_START, status="in_progress")
        )
        programming.on_event(
            _agent_event(ACPEventType.TOOL_CALL_DONE, status="completed")
        )

        assert parent.state is not None
        assert parent.state.terminal == "running"
        assert client.created == 1
        assert not any(block.kind == "tool_call" for block in parent.state.blocks)
        assert parent.state.metadata.subagents[0]["status"] == "completed"
    finally:
        programming.abort()


def test_execute_output_containing_subagent_marker_stays_parent_tool() -> None:
    """Source text mentioning the marker must not turn an exec into a child task."""
    delivery = CardDelivery(_CardClient())
    parent = _card_session(delivery, session_id="parent-exec-marker")
    programming = ProgrammingCardSession(parent)
    try:
        programming.start()
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=ToolCallInfo(
                    id="exec-reading-source",
                    title="exec",
                    kind="execute",
                    status="in_progress",
                    content="sed -n '1,460p' src/acp/client.py",
                ),
            )
        )
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_DONE,
                tool_call=ToolCallInfo(
                    id="exec-reading-source",
                    title="exec",
                    kind="execute",
                    status="completed",
                    content='parts.append(f"子代理：{subagent_type}")',
                ),
            )
        )

        assert parent.state is not None
        assert parent.state.terminal == "running"
        assert any(block.kind == "tool_call" for block in parent.state.blocks)
    finally:
        programming.abort()


def test_known_agent_call_keeps_routing_when_terminal_shape_changes() -> None:
    """Once summarized as a subtask, later events route by the stable call id."""
    delivery = CardDelivery(_CardClient())
    parent = _card_session(delivery, session_id="parent-shape-change")
    programming = ProgrammingCardSession(parent)
    try:
        programming.start()
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=ToolCallInfo(
                    id="agent-shape-change",
                    title="worker",
                    kind="other",
                    status="in_progress",
                    content="分析实现\n子代理：Explore",
                ),
            )
        )
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_DONE,
                tool_call=ToolCallInfo(
                    id="agent-shape-change",
                    title="shell",
                    kind="execute",
                    status="failed",
                    content="boom",
                ),
            )
        )

        assert parent.state is not None
        assert parent.state.terminal == "running"
        assert parent.state.metadata.subagents[0]["status"] == "failed"
    finally:
        programming.abort()


def test_agent_image_routes_to_main_card_with_task_attribution() -> None:
    client = _CardClient()
    delivery = CardDelivery(client)
    parent = _card_session(delivery, session_id="parent-child-image")
    programming = ProgrammingCardSession(
        parent,
        image_uploader=lambda _: "img_child",
    )
    try:
        programming.start()
        programming.on_event(
            _agent_event(ACPEventType.TOOL_CALL_START, status="in_progress")
        )
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.IMAGE_CHUNK,
                image=ACPImageInfo(
                    image_id="sha256:child-image",
                    mime_type="image/png",
                    data="aW1hZ2U=",
                    name="child.png",
                ),
                source_id="agent-call-1",
            )
        )

        assert parent.state is not None
        assert client.created == 1
        image_block = next(
            block for block in parent.state.blocks if block.kind == "image"
        )
        assert "分析剩余实现" in image_block.alt
    finally:
        programming.abort()


def test_late_agent_image_stays_in_main_card_with_task_attribution() -> None:
    delivery = CardDelivery(_CardClient())
    parent = _card_session(delivery, session_id="parent-late-child-image")
    programming = ProgrammingCardSession(
        parent,
        image_uploader=lambda _: "img_late_child",
    )
    try:
        programming.start()
        programming.on_event(
            _agent_event(ACPEventType.TOOL_CALL_START, status="in_progress")
        )
        programming.on_event(
            _agent_event(ACPEventType.TOOL_CALL_DONE, status="completed")
        )
        task_label = parent.state.metadata.subagents[0]["label"]

        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.IMAGE_CHUNK,
                image=ACPImageInfo(
                    image_id="sha256:late-child-image",
                    mime_type="image/png",
                    data="aW1hZ2U=",
                    name="late.png",
                ),
                source_id="agent-call-1",
            )
        )

        assert parent.state is not None
        image_block = next(
            block for block in parent.state.blocks if block.kind == "image"
        )
        assert task_label
        assert task_label in image_block.alt
    finally:
        programming.abort()


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", PromptOutcome.COMPLETED),
        ("cancelled", PromptOutcome.CANCELLED),
        ("canceled", PromptOutcome.CANCELLED),
        ("max_turn_requests", PromptOutcome.INCOMPLETE),
        ("max_tokens", PromptOutcome.INCOMPLETE),
        ("refusal", PromptOutcome.INCOMPLETE),
        ("failed", PromptOutcome.INCOMPLETE),
        ("error", PromptOutcome.INCOMPLETE),
        ("timeout", PromptOutcome.INCOMPLETE),
        ("", PromptOutcome.INCOMPLETE),
    ],
)
def test_prompt_stop_reason_is_not_implicitly_successful(
    stop_reason: str,
    expected: PromptOutcome,
) -> None:
    assessment = classify_prompt_result(PromptResult(stop_reason=stop_reason))
    assert assessment.outcome is expected


def test_end_turn_with_pending_plan_is_incomplete() -> None:
    result = PromptResult(
        stop_reason="end_turn",
        plan=PlanInfo(
            entries=[
                PlanEntryInfo(content="已完成", status="completed"),
                PlanEntryInfo(content="仍需真实链路验证", status="in_progress"),
            ]
        ),
    )

    assessment = classify_prompt_result(result)

    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert "计划" in assessment.detail


def test_end_turn_with_active_tool_is_incomplete() -> None:
    result = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="tool-1",
                title="pytest",
                kind="execute",
                status="in_progress",
            )
        ],
    )

    assessment = classify_prompt_result(result)

    assert assessment.outcome is PromptOutcome.INCOMPLETE
    assert "工具" in assessment.detail


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", PromptOutcome.COMPLETED),
        ("failed", PromptOutcome.COMPLETED),
        ("pending", PromptOutcome.INCOMPLETE),
        ("in_progress", PromptOutcome.INCOMPLETE),
        ("future_pending_state", PromptOutcome.INCOMPLETE),
        ("", PromptOutcome.INCOMPLETE),
        (None, PromptOutcome.INCOMPLETE),
    ],
)
def test_end_turn_requires_tool_calls_to_be_terminal(
    status: str | None,
    expected: PromptOutcome,
) -> None:
    result = PromptResult(
        stop_reason="end_turn",
        tool_calls=[
            ToolCallInfo(
                id="tool-status-contract",
                title="contract",
                kind="other",
                status=status,
            )
        ],
    )

    assessment = classify_prompt_result(result)

    assert assessment.outcome is expected
