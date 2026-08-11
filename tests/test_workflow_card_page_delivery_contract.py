"""RED contract tests for paged Workflow progress-card delivery."""

from __future__ import annotations

import dataclasses
import threading
from typing import Any
from unittest.mock import MagicMock, patch

from src.card.state.models import ContentBlock
from src.feishu.handlers.workflow import WorkflowHandler
from src.feishu.handlers.workflow_card_pages import WorkflowCardPageDelivery
from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer import WorkflowProgressRenderer


def _renderer_page(
    label: str,
    *,
    include_stop: bool = False,
    page_key: tuple[str, int | str, int] | None = None,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = [
        {"tag": "markdown", "content": label},
    ]
    if include_stop:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "Stop"},
                        "value": {"action": "workflow_stop_running"},
                    }
                ],
            }
        )
    page = {
        "header": {
            "title": {"tag": "plain_text", "content": label},
            "template": "blue",
        },
        "elements": elements,
    }
    if page_key is not None:
        page["_workflow_page_key"] = page_key
    return page


def _make_handler(*, continuation_results: list[object]) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(side_effect=continuation_results)
    handler.reply_text = MagicMock()
    handler._resolve_origin = MagicMock(return_value="origin-message")
    handler._workflow_is_running_for_card = MagicMock(return_value=True)
    handler._send_workflow_completion_report = MagicMock(return_value={})
    handler._reply_workflow_completion_fallback = MagicMock()
    return handler


def _build_callbacks(handler: WorkflowHandler) -> Any:
    project = MagicMock(project_id="project-1")
    return WorkflowHandler._build_workflow_callbacks(
        handler,
        "status-message",
        "chat-1",
        project,
    )


def _updated_message_ids(handler: WorkflowHandler) -> list[str]:
    return [call.args[0] for call in handler.update_card.call_args_list]


def _sent_payload_labels(handler: WorkflowHandler) -> list[str]:
    return [str(call.args[1]) for call in handler.send_card_to_chat.call_args_list]


def test_paged_callbacks_replace_status_append_results_reuse_pages_and_finalize() -> None:
    handler = _make_handler(
        continuation_results=["result-message-1", "result-message-2"],
    )
    callbacks = _build_callbacks(handler)

    callbacks.on_progress(
        [
            _renderer_page("status-v1"),
            _renderer_page("result-1-v1"),
            _renderer_page("result-2-v1"),
        ]
    )

    assert _updated_message_ids(handler) == ["status-message"]
    assert handler.send_card_to_chat.call_count == 2
    first_sent = _sent_payload_labels(handler)
    assert "result-1-v1" in first_sent[0]
    assert "result-2-v1" in first_sent[1]

    handler.update_card.reset_mock()
    handler.send_card_to_chat.reset_mock()
    callbacks.on_progress(
        [
            _renderer_page("status-v2"),
            _renderer_page("result-1-v2"),
            _renderer_page("result-2-v2"),
        ]
    )

    assert _updated_message_ids(handler) == [
        "status-message",
        "result-message-1",
        "result-message-2",
    ]
    handler.send_card_to_chat.assert_not_called()

    handler.update_card.reset_mock()
    terminal_pages = [
        _renderer_page("terminal-status", include_stop=True),
        _renderer_page("terminal-result-1", include_stop=True),
        _renderer_page("terminal-result-2", include_stop=True),
    ]
    done_project = WorkflowProject(
        name="paged workflow",
        requirement="deliver every result",
        status=WorkflowStatus.COMPLETED,
        result='{"final_report": "complete"}',
    )
    with patch(
        "src.workflow_engine.renderer.render_completion_cards",
        return_value=terminal_pages,
    ):
        callbacks.on_done(done_project)

    assert _updated_message_ids(handler) == [
        "status-message",
        "result-message-1",
        "result-message-2",
    ]
    handler.send_card_to_chat.assert_not_called()
    handler._reply_workflow_completion_fallback.assert_not_called()
    for update_call in handler.update_card.call_args_list:
        assert "workflow_stop_running" not in str(update_call.args[1])


def test_terminal_page_failure_uses_report_then_exposes_retryable_double_failure() -> None:
    handler = _make_handler(continuation_results=[None, None])
    handler.update_card.return_value = False
    handler._send_workflow_completion_report.return_value = {
        "generated": True,
        "attachment_sent": False,
        "html_filename": "workflow-report.html",
        "error": "attachment unavailable",
    }
    callbacks = _build_callbacks(handler)
    done_project = WorkflowProject(
        name="paged workflow",
        requirement="deliver every result",
        status=WorkflowStatus.COMPLETED,
        result='{"final_report": "complete"}',
    )
    terminal_pages = [
        _renderer_page("terminal-status"),
        _renderer_page("terminal-result"),
    ]

    with patch(
        "src.workflow_engine.renderer.render_completion_cards",
        return_value=terminal_pages,
    ):
        callbacks.on_done(done_project)

    handler._send_workflow_completion_report.assert_called_once()
    handler._reply_workflow_completion_fallback.assert_called_once()
    fallback_kwargs = handler._reply_workflow_completion_fallback.call_args.kwargs
    assert fallback_kwargs["failed_page_indexes"]


def test_paged_callbacks_keep_existing_pages_when_continuation_create_fails() -> None:
    handler = _make_handler(
        continuation_results=[
            "result-message-1",
            RuntimeError("continuation create failed"),
            "result-message-2",
        ],
    )
    callbacks = _build_callbacks(handler)

    callbacks.on_progress(
        [
            _renderer_page("status-v1"),
            _renderer_page("result-1-v1"),
            _renderer_page("result-2-v1"),
        ]
    )

    assert _updated_message_ids(handler) == ["status-message"]
    assert handler.send_card_to_chat.call_count == 2

    handler.update_card.reset_mock()
    callbacks.on_progress(
        [
            _renderer_page("status-v2"),
            _renderer_page("result-1-v2"),
            _renderer_page("result-2-v2"),
        ]
    )

    assert _updated_message_ids(handler) == ["status-message", "result-message-1"]
    updated_payloads = {
        call.args[0]: str(call.args[1])
        for call in handler.update_card.call_args_list
    }
    assert "status-v2" in updated_payloads["status-message"]
    assert "result-1-v2" in updated_payloads["result-message-1"]
    assert "result-2-v2" not in updated_payloads["status-message"]
    assert handler.send_card_to_chat.call_count == 3
    assert "result-2-v2" in str(handler.send_card_to_chat.call_args.args[1])


def test_keyed_pages_do_not_overwrite_later_agent_when_earlier_agent_appears() -> None:
    handler = _make_handler(
        continuation_results=["agent-a2-message", "agent-a1-message"],
    )
    callbacks = _build_callbacks(handler)

    callbacks.on_progress(
        [
            _renderer_page("status-v1", page_key=("status", -1, 0)),
            _renderer_page("agent-a2-v1", page_key=("agent", 1, 0)),
        ]
    )

    handler.update_card.reset_mock()
    handler.send_card_to_chat.reset_mock()
    callbacks.on_progress(
        [
            _renderer_page("status-v2", page_key=("status", -1, 0)),
            _renderer_page("agent-a1-v1", page_key=("agent", 0, 0)),
            _renderer_page("agent-a2-v2", page_key=("agent", 1, 0)),
        ]
    )

    updated = {
        call.args[0]: str(call.args[1])
        for call in handler.update_card.call_args_list
    }
    assert "agent-a2-v2" in updated["agent-a2-message"]
    assert "agent-a1-v1" not in updated["agent-a2-message"]
    handler.send_card_to_chat.assert_called_once()
    assert "agent-a1-v1" in str(handler.send_card_to_chat.call_args.args[1])


def test_keyed_pages_do_not_overwrite_next_agent_when_call_gains_a_page() -> None:
    handler = _make_handler(
        continuation_results=[
            "agent-a1-page-1-message",
            "agent-a2-message",
            "agent-a1-page-2-message",
        ],
    )
    callbacks = _build_callbacks(handler)

    callbacks.on_progress(
        [
            _renderer_page("status-v1", page_key=("status", -1, 0)),
            _renderer_page("agent-a1-page-1-v1", page_key=("agent", 0, 0)),
            _renderer_page("agent-a2-v1", page_key=("agent", 1, 0)),
        ]
    )

    handler.update_card.reset_mock()
    handler.send_card_to_chat.reset_mock()
    callbacks.on_progress(
        [
            _renderer_page("status-v2", page_key=("status", -1, 0)),
            _renderer_page("agent-a1-page-1-v2", page_key=("agent", 0, 0)),
            _renderer_page("agent-a1-page-2-v1", page_key=("agent", 0, 1)),
            _renderer_page("agent-a2-v2", page_key=("agent", 1, 0)),
        ]
    )

    updated = {
        call.args[0]: str(call.args[1])
        for call in handler.update_card.call_args_list
    }
    assert "agent-a1-page-1-v2" in updated["agent-a1-page-1-message"]
    assert "agent-a2-v2" in updated["agent-a2-message"]
    assert "agent-a1-page-2-v1" not in updated["agent-a2-message"]
    handler.send_card_to_chat.assert_called_once()
    assert "agent-a1-page-2-v1" in str(handler.send_card_to_chat.call_args.args[1])


def test_keyed_terminal_refresh_retains_missing_known_page_and_strips_stop() -> None:
    handler = _make_handler(
        continuation_results=["agent-a2-message", "agent-a1-message"],
    )
    callbacks = _build_callbacks(handler)
    callbacks.on_progress(
        [
            _renderer_page(
                "status-progress",
                include_stop=True,
                page_key=("status", -1, 0),
            ),
            _renderer_page(
                "agent-a2-progress",
                include_stop=True,
                page_key=("agent", 1, 0),
            ),
        ]
    )

    handler.update_card.reset_mock()
    handler.send_card_to_chat.reset_mock()
    terminal_pages = [
        _renderer_page(
            "status-terminal",
            include_stop=True,
            page_key=("status", -1, 0),
        ),
        _renderer_page(
            "agent-a1-terminal",
            include_stop=True,
            page_key=("agent", 0, 0),
        ),
    ]
    done_project = WorkflowProject(
        name="keyed terminal",
        status=WorkflowStatus.COMPLETED,
        result='{"final_report": "complete"}',
    )

    with patch(
        "src.workflow_engine.renderer.render_completion_cards",
        return_value=terminal_pages,
    ):
        callbacks.on_done(done_project)

    updated = {
        call.args[0]: str(call.args[1])
        for call in handler.update_card.call_args_list
    }
    assert "status-terminal" in updated["status-message"]
    assert "agent-a2-progress" in updated["agent-a2-message"]
    assert "agent-a1-terminal" not in updated["agent-a2-message"]
    assert all(
        "workflow_stop_running" not in payload
        for payload in updated.values()
    )
    handler.send_card_to_chat.assert_called_once()
    assert "agent-a1-terminal" in str(handler.send_card_to_chat.call_args.args[1])
    assert "workflow_stop_running" not in str(handler.send_card_to_chat.call_args.args[1])


def test_keyed_delivery_skips_only_successfully_delivered_unchanged_payloads() -> None:
    delivery = WorkflowCardPageDelivery(["status-message"])
    created_message_ids = iter(["agent-a1-message", "agent-a2-message"])
    wire_calls: list[tuple[str | None, str]] = []
    fail_next_a1_patch = False

    def replace_or_send(**kwargs: Any) -> str | None:
        nonlocal fail_next_a1_patch
        message_id = kwargs["card_message_id"]
        payload = str(kwargs["card_data"])
        wire_calls.append((message_id, payload))
        if fail_next_a1_patch and "agent-a1-v2" in payload:
            fail_next_a1_patch = False
            return None
        return message_id or next(created_message_ids)

    first = delivery.deliver(
        [
            _renderer_page("status-v1", page_key=("status", -1, 0)),
            _renderer_page("agent-a1-v1", page_key=("agent", 0, 0)),
            _renderer_page("agent-a2-v1", page_key=("agent", 1, 0)),
        ],
        replace_or_send=replace_or_send,
        chat_id="chat-1",
    )
    assert first.fully_delivered

    wire_calls.clear()
    fail_next_a1_patch = True
    failed_patch = delivery.deliver(
        [
            _renderer_page("status-v1", page_key=("status", -1, 0)),
            _renderer_page("agent-a1-v2", page_key=("agent", 0, 0)),
            _renderer_page("agent-a2-v1", page_key=("agent", 1, 0)),
        ],
        replace_or_send=replace_or_send,
        chat_id="chat-1",
    )

    assert len(wire_calls) == 1
    assert wire_calls[0][0] == "agent-a1-message"
    assert "agent-a1-v2" in wire_calls[0][1]
    assert failed_patch.failed_page_indexes == (1,)
    assert failed_patch.delivered_page_indexes == (0, 2)

    wire_calls.clear()
    retried = delivery.deliver(
        [
            _renderer_page("status-v1", page_key=("status", -1, 0)),
            _renderer_page("agent-a1-v2", page_key=("agent", 0, 0)),
            _renderer_page("agent-a2-v1", page_key=("agent", 1, 0)),
        ],
        replace_or_send=replace_or_send,
        chat_id="chat-1",
    )

    assert len(wire_calls) == 1
    assert wire_calls[0][0] == "agent-a1-message"
    assert retried.fully_delivered
    assert retried.delivered_page_indexes == (0, 1, 2)


def test_keyed_delivery_accepts_deterministic_string_call_identity() -> None:
    delivery = WorkflowCardPageDelivery(["status-message"])

    result = delivery.deliver(
        [
            _renderer_page("status", page_key=("status", -1, 0)),
            _renderer_page(
                "agent",
                page_key=("agent", "call:deterministic-identity", 0),
            ),
        ],
        replace_or_send=lambda **kwargs: (
            kwargs["card_message_id"] or "agent-message"
        ),
        chat_id="chat-1",
    )

    assert result.fully_delivered
    assert delivery.page_message_ids == ("status-message", "agent-message")


def test_failure_terminal_delivery_flushes_latest_execution_on_stable_pages() -> None:
    handler = _make_handler(
        continuation_results=["agent-a1-message", "ledger-message"],
    )
    callbacks = _build_callbacks(handler)
    running_agent = AgentProgress(
        label="implementation",
        agent_id="A1",
        tool="codex",
        model="model-a",
        status=AgentStatus.RUNNING,
        started_at=100.0,
        call_index=0,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="answer",
                content="FLUSHED_PREFIX",
                status="active",
            )
        ],
    )
    completed_agent = AgentProgress(
        label="analysis",
        agent_id="A2",
        tool="claude",
        status=AgentStatus.DONE,
        result="analysis complete",
        started_at=90.0,
        finished_at=99.0,
        call_index=1,
    )
    running_project = WorkflowProject(
        name="failure-terminal-stream",
        status=WorkflowStatus.RUNNING,
        phases=[
            PhaseProgress(
                title="build",
                agents=[running_agent, completed_agent],
            )
        ],
    )
    callbacks.on_progress(
        WorkflowProgressRenderer(running_project).render_progress_cards(
            running_project
        )
    )

    assert _updated_message_ids(handler) == ["status-message"]
    assert handler.send_card_to_chat.call_count == 1

    handler.update_card.reset_mock()
    handler.send_card_to_chat.reset_mock()
    failed_agent = running_agent.model_copy(deep=True)
    failed_agent.status = AgentStatus.FAILED
    failed_agent.error = "runtime bridge failed"
    failed_agent.finished_at = 101.0
    failed_agent.execution_blocks[0] = dataclasses.replace(
        failed_agent.execution_blocks[0],
        content="FLUSHED_PREFIX\nLAST_UNFLUSHED_STREAM_MARKER",
    )
    failed_project = WorkflowProject(
        name="failure-terminal-stream",
        status=WorkflowStatus.FAILED,
        error="runtime bridge failed",
        phases=[
            PhaseProgress(
                title="build",
                agents=[failed_agent, completed_agent.model_copy(deep=True)],
            )
        ],
    )

    callbacks.on_error("runtime bridge failed", failed_project)

    assert _updated_message_ids(handler) == [
        "status-message",
        "agent-a1-message",
    ]
    handler.send_card_to_chat.assert_called_once()
    assert "analysis complete" in str(handler.send_card_to_chat.call_args.args[1])
    updated_payloads = {
        call.args[0]: str(call.args[1])
        for call in handler.update_card.call_args_list
    }
    agent_payload = updated_payloads["agent-a1-message"]
    assert "LAST_UNFLUSHED_STREAM_MARKER" in agent_payload
    assert "当前进展" not in agent_payload
    assert "'template': 'red'" in agent_payload
    assert all(
        "workflow_stop_running" not in payload
        for payload in updated_payloads.values()
    )


def test_cancelled_terminal_delivery_bypasses_owner_stop_fence() -> None:
    handler = _make_handler(
        continuation_results=["agent-message", "ledger-message"],
    )
    lifecycle_owner = MagicMock()
    lifecycle_owner.delivery_lock = threading.RLock()
    lifecycle_owner.stop_event = threading.Event()
    project_context = MagicMock(project_id="project-1")
    callbacks = WorkflowHandler._build_workflow_callbacks(
        handler,
        "status-message",
        "chat-1",
        project_context,
        lifecycle_owner=lifecycle_owner,
    )
    running_agent = AgentProgress(
        label="implementation",
        agent_id="A1",
        tool="codex",
        status=AgentStatus.RUNNING,
        started_at=100.0,
        call_index=0,
        execution_blocks=[
            ContentBlock(
                kind="text",
                block_id="answer",
                content="BEFORE_CANCEL",
                status="active",
            )
        ],
    )
    running_project = WorkflowProject(
        name="cancel-terminal-stream",
        status=WorkflowStatus.RUNNING,
        phases=[PhaseProgress(title="build", agents=[running_agent])],
    )
    callbacks.on_progress(
        WorkflowProgressRenderer(running_project).render_progress_cards(
            running_project
        )
    )

    lifecycle_owner.stop_event.set()
    handler.update_card.reset_mock()
    handler.send_card_to_chat.reset_mock()
    cancelled_agent = running_agent.model_copy(deep=True)
    cancelled_agent.status = AgentStatus.CANCELLED
    cancelled_agent.error = "Workflow cancelled"
    cancelled_agent.finished_at = 101.0
    cancelled_agent.execution_blocks[0] = dataclasses.replace(
        cancelled_agent.execution_blocks[0],
        content="BEFORE_CANCEL\nLAST_CANCELLED_STREAM_FRAME",
    )
    cancelled_project = WorkflowProject(
        name="cancel-terminal-stream",
        status=WorkflowStatus.CANCELLED,
        error="Workflow cancelled",
        phases=[PhaseProgress(title="build", agents=[cancelled_agent])],
    )

    callbacks.on_cancelled(cancelled_project)

    updated_payloads = {
        call.args[0]: str(call.args[1])
        for call in handler.update_card.call_args_list
    }
    assert "LAST_CANCELLED_STREAM_FRAME" in updated_payloads["agent-message"]
    assert "'template': 'grey'" in updated_payloads["agent-message"]
    assert all(
        "workflow_stop_running" not in payload
        for payload in updated_payloads.values()
    )
