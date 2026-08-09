"""RED contract tests for paged Workflow progress-card delivery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.workflow import WorkflowHandler
from src.feishu.handlers.workflow_script import WorkflowScriptMixin
from src.workflow_engine.models import WorkflowProject, WorkflowStatus

_CALLBACK_BUILDERS = (
    pytest.param(WorkflowHandler._build_workflow_callbacks, id="workflow-handler"),
    pytest.param(WorkflowScriptMixin._build_workflow_callbacks, id="workflow-script-mixin"),
)


def _renderer_page(label: str, *, include_stop: bool = False) -> dict[str, Any]:
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
    return {
        "header": {
            "title": {"tag": "plain_text", "content": label},
            "template": "blue",
        },
        "elements": elements,
    }


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


def _build_callbacks(
    builder: Callable[..., Any],
    handler: WorkflowHandler,
) -> Any:
    project = MagicMock(project_id="project-1")
    return builder(handler, "status-message", "chat-1", project)


def _updated_message_ids(handler: WorkflowHandler) -> list[str]:
    return [call.args[0] for call in handler.update_card.call_args_list]


def _sent_payload_labels(handler: WorkflowHandler) -> list[str]:
    return [str(call.args[1]) for call in handler.send_card_to_chat.call_args_list]


@pytest.mark.parametrize("builder", _CALLBACK_BUILDERS)
def test_paged_callbacks_replace_status_append_results_reuse_pages_and_finalize(
    builder: Callable[..., Any],
) -> None:
    handler = _make_handler(
        continuation_results=["result-message-1", "result-message-2"],
    )
    callbacks = _build_callbacks(builder, handler)

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
        "src.workflow_engine.renderer.render_completion_card",
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


@pytest.mark.parametrize("builder", _CALLBACK_BUILDERS)
def test_paged_callbacks_keep_existing_pages_when_continuation_create_fails(
    builder: Callable[..., Any],
) -> None:
    handler = _make_handler(
        continuation_results=[
            "result-message-1",
            RuntimeError("continuation create failed"),
            "result-message-2",
        ],
    )
    callbacks = _build_callbacks(builder, handler)

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
