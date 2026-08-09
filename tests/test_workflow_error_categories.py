"""Tests for Workflow callback error categories (AC19).

Validates that all callback validation failures return standardized error cards
via _reply_workflow_error with the correct error category.

Error categories covered:
- session_expired: missing engine, session key mismatch
- invalid_state: wrong workflow status for the operation
- invalid_argument: invalid parameter values
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.feishu.handlers.workflow import WorkflowHandler
from src.workflow_engine.models import WorkflowProject, WorkflowStatus


def _create_mock_handler():
    """Create a mock WorkflowHandler with basic dependencies."""
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.reply_card = MagicMock()
    handler.reply_text = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler._get_root_path = MagicMock(return_value="/tmp/project")
    handler._resolve_project_from_id = MagicMock(return_value=None)
    return handler


# ---------------------------------------------------------------------------
# Tool selection callback error tests
# ---------------------------------------------------------------------------

def test_tool_select_missing_engine_returns_session_expired():
    """AC19: 工具选择回调 missing engine 返回 session_expired 错误。"""
    handler = _create_mock_handler()
    handler.ctx.workflow_engine_manager.get = MagicMock(return_value=None)

    handler.handle_workflow_select_tool(
        message_id="msg_123",
        chat_id="chat_123",
        project_id="",
        value={"action": "workflow_select_tool", "tool_name": "coco"},
    )

    # Verify _reply_workflow_error was called with session_expired
    handler._reply_workflow_error.assert_called_once()
    call_args = handler._reply_workflow_error.call_args
    assert call_args[0][1] == "session_expired", \
        f"Expected 'session_expired', got '{call_args[0][1]}'"


def test_tool_select_wrong_state_returns_invalid_state():
    """AC19: 工具选择回调 wrong state 返回 invalid_state 错误。"""
    handler = _create_mock_handler()

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.RUNNING,  # Wrong state - should be AWAITING_*
        pending=None,
    )
    handler.ctx.workflow_engine_manager.get = MagicMock(return_value=mock_engine)

    handler.handle_workflow_select_tool(
        message_id="msg_123",
        chat_id="chat_123",
        project_id="",
        value={"action": "workflow_select_tool", "tool_name": "coco"},
    )

    handler._reply_workflow_error.assert_called_once()
    call_args = handler._reply_workflow_error.call_args
    assert call_args[0][1] == "invalid_state", \
        f"Expected 'invalid_state', got '{call_args[0][1]}'"


@patch("src.thread.get_current_sender_id", return_value="user_123")
def test_tool_select_session_key_mismatch_returns_session_expired(mock_sender):
    """AC19: 工具选择回调 session key 不匹配返回 session_expired 错误。"""
    from src.workflow_engine.models import PendingConfirmation

    handler = _create_mock_handler()

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            engine_session_key="correct_key",
            initiator_user_id="user_123",
            selected_tools=["coco"],
        ),
    )
    handler.ctx.workflow_engine_manager.get = MagicMock(return_value=mock_engine)

    handler.handle_workflow_select_tool(
        message_id="msg_123",
        chat_id="chat_123",
        project_id="",
        value={
            "action": "workflow_select_tool",
            "tool_name": "coco",
            "engine_session_key": "wrong_key",
        },
    )

    handler._reply_workflow_error.assert_called_once()
    call_args = handler._reply_workflow_error.call_args
    assert call_args[0][1] == "session_expired"


@patch("src.thread.get_current_sender_id", return_value="user_456")
def test_tool_select_wrong_initiator_returns_forbidden(mock_sender):
    """AC19: 工具选择回调非发起者返回 forbidden 错误。"""
    from src.workflow_engine.models import PendingConfirmation

    handler = _create_mock_handler()

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            engine_session_key="valid_key",
            initiator_user_id="user_123",  # Different from current user
            selected_tools=["coco"],
        ),
    )
    handler.ctx.workflow_engine_manager.get = MagicMock(return_value=mock_engine)

    handler.handle_workflow_select_tool(
        message_id="msg_123",
        chat_id="chat_123",
        project_id="",
        value={
            "action": "workflow_select_tool",
            "tool_name": "coco",
            "engine_session_key": "valid_key",
        },
    )

    handler._reply_workflow_error.assert_called_once()
    call_args = handler._reply_workflow_error.call_args
    assert call_args[0][1] == "forbidden"


@patch("src.thread.get_current_sender_id", return_value="user_123")
def test_tool_select_missing_tool_name_returns_invalid_argument(mock_sender):
    """AC19: 工具选择回调缺少 tool_name 返回 invalid_argument 错误。"""
    from src.workflow_engine.models import PendingConfirmation

    handler = _create_mock_handler()

    mock_engine = MagicMock()
    mock_engine.project = WorkflowProject(
        workflow_id="test_wf",
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            engine_session_key="valid_key",
            initiator_user_id="user_123",
            selected_tools=["coco"],
        ),
    )
    handler.ctx.workflow_engine_manager.get = MagicMock(return_value=mock_engine)

    handler.handle_workflow_select_tool(
        message_id="msg_123",
        chat_id="chat_123",
        project_id="",
        value={
            "action": "workflow_select_tool",
            "engine_session_key": "valid_key",
            # Missing tool_name
        },
    )

    handler._reply_workflow_error.assert_called_once()
    call_args = handler._reply_workflow_error.call_args
    assert call_args[0][1] == "invalid_argument"


# ---------------------------------------------------------------------------
# Budget selection removal verification
# ---------------------------------------------------------------------------


def test_budget_selection_removed():
    """Verify budget selection feature has been removed.

    Budget selection has been removed in favor of 2-step orchestrator+review selection.
    This test documents that budget-related actions should no longer be registered.
    """
    from src.card.actions.dispatch import build_common_action_registry

    common_registry = build_common_action_registry()
    all_actions = set(common_registry.keys())

    # Verify no budget-related actions are registered
    budget_actions = [a for a in all_actions if "budget" in a.lower()]
    assert len(budget_actions) == 0, f"Found unexpected budget actions: {budget_actions}"


# ---------------------------------------------------------------------------
# Error card visibility and UI text tests
# ---------------------------------------------------------------------------

def test_error_card_is_visible_to_user():
    """AC19: 错误卡片通过 reply_card 发送给用户，可见。"""

    handler = _create_mock_handler()

    # Call the actual _reply_workflow_error method
    handler._reply_workflow_error = WorkflowHandler._reply_workflow_error.__get__(handler)

    # Mock the card builder
    with patch.object(handler, 'reply_card') as mock_reply:
        handler._reply_workflow_error("msg_123", "session_expired")

        # Verify reply_card was called (error is visible to user)
        mock_reply.assert_called_once()
        card = mock_reply.call_args[0][1]
        assert isinstance(card, dict)
        assert "body" in card
        assert "elements" in card["body"]


def test_all_error_categories_have_ui_text():
    """AC19: 所有错误类别都有对应的 UI 文本。"""
    from src.card.ui_text import UI_TEXT

    error_categories = [
        "session_expired",
        "invalid_state",
        "invalid_argument",
        "forbidden",
        "runtime_timeout",
    ]

    for category in error_categories:
        ui_key = f"workflow_error_{category}_title"
        assert ui_key in UI_TEXT, f"Missing UI text for error category: {category}"


def test_error_card_uses_correct_ui_text():
    """AC19: 错误卡片使用正确的 UI 文本。"""
    from src.card.ui_text import UI_TEXT

    handler = _create_mock_handler()
    handler._reply_workflow_error = WorkflowHandler._reply_workflow_error.__get__(handler)

    with patch.object(handler, 'reply_card') as mock_reply:
        # Test session_expired
        handler._reply_workflow_error("msg_123", "session_expired")
        card = mock_reply.call_args[0][1]
        assert card["header"]["title"]["content"] == UI_TEXT["workflow_error_session_expired_title"]

        mock_reply.reset_mock()

        # Test invalid_state
        handler._reply_workflow_error("msg_123", "invalid_state")
        card = mock_reply.call_args[0][1]
        assert card["header"]["title"]["content"] == UI_TEXT["workflow_error_invalid_state_title"]

        mock_reply.reset_mock()

        # Test invalid_argument with detail
        handler._reply_workflow_error("msg_123", "invalid_argument", detail="测试错误详情")
        card = mock_reply.call_args[0][1]
        assert card["header"]["title"]["content"] == UI_TEXT["workflow_error_invalid_argument_title"]
        body = card["body"]["elements"][0]["content"]
        assert "测试错误详情" in body

        mock_reply.reset_mock()

        # Test forbidden
        handler._reply_workflow_error("msg_123", "forbidden")
        card = mock_reply.call_args[0][1]
        assert card["header"]["title"]["content"] == UI_TEXT["workflow_error_forbidden_title"]

        mock_reply.reset_mock()

        # Test runtime_timeout
        handler._reply_workflow_error(
            "msg_123",
            "runtime_timeout",
            detail="Workflow execution exceeded total timeout of 1800s",
        )
        card = mock_reply.call_args[0][1]
        assert card["header"]["title"]["content"] == UI_TEXT["workflow_error_runtime_timeout_title"]
        assert card["header"]["title"]["content"] != UI_TEXT["workflow_error_invalid_state_title"]
        body = card["body"]["elements"][0]["content"]
        assert "1800s" in body


def test_runtime_timeout_callback_is_not_reported_as_invalid_state():
    """Runtime timeout is a workflow execution failure, not a callback state mismatch."""

    handler = _create_mock_handler()
    handler._resolve_origin = MagicMock(return_value="origin_msg")
    handler._build_error_card = MagicMock(return_value={"error": "timeout"})
    handler._replace_or_send_workflow_card = MagicMock()

    callbacks = WorkflowHandler._build_workflow_callbacks(
        handler,
        message_id="msg_123",
        chat_id="chat_123",
        project=None,
    )

    callbacks.on_error("Workflow execution exceeded total timeout of 1800s")

    handler._build_error_card.assert_called_once_with(
        "runtime_timeout",
        detail="Workflow execution exceeded total timeout of 1800s",
    )
    handler._replace_or_send_workflow_card.assert_called_once()


def test_cancelled_reviewer_is_reported_as_review_failure():
    """Nested reviewer cancellation is not a top-level lifecycle mismatch."""
    handler = _create_mock_handler()
    handler._resolve_origin = MagicMock(return_value="origin_msg")
    handler._build_error_card = MagicMock(return_value={"error": "review"})
    handler._replace_or_send_workflow_card = MagicMock()
    callbacks = WorkflowHandler._build_workflow_callbacks(
        handler,
        message_id="msg_123",
        chat_id="chat_123",
        project=None,
    )

    detail = (
        "Independent review failed: traex/model: Reviewer did not complete "
        "normally (stop_reason=cancelled)"
    )
    callbacks.on_error(detail)

    handler._build_error_card.assert_called_once_with(
        "review_failed",
        detail=detail,
    )
