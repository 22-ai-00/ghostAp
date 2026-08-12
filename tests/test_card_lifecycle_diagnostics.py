"""Terminal card diagnostics must match the actions actually rendered."""

from src.card.actions import dispatch as action_ids
from src.card.error_diagnostics import render_error_diagnostic
from src.card.events import CardEvent
from src.card.state.models import CardMetadata
from src.card.state.reducer import reduce_card_state


def _started_state():
    return reduce_card_state(
        None,
        CardEvent.started(),
        CardMetadata(mode_name="Codex", tool_name="codex"),
    )


def _block_text(state) -> str:
    return "\n".join(
        str(getattr(block, "content", "") or "") for block in state.blocks
    )


def test_failed_card_only_promises_details_when_detail_action_exists() -> None:
    state = reduce_card_state(_started_state(), CardEvent.failed("boom"))

    assert "点击“查看详情”" not in _block_text(state)
    assert all(button.text != "查看详情" for button in state.buttons)


def test_failed_card_renders_bound_diagnostic_action() -> None:
    state = reduce_card_state(
        _started_state(),
        CardEvent.failed(
            "boom",
            details="safe structured diagnostic",
            detail_action={
                "action": action_ids.SHOW_ERROR_DETAILS,
                "chat_id": "chat-1",
                "origin_message_id": "message-1",
            },
        ),
    )

    detail_button = next(button for button in state.buttons if button.text == "查看详情")
    assert detail_button.action_id == action_ids.SHOW_ERROR_DETAILS
    assert detail_button.value is not None
    diagnostic_token = detail_button.value.get("diagnostic_token")
    assert diagnostic_token
    assert "点击“查看详情”" in _block_text(state)
    rendered = render_error_diagnostic(
        diagnostic_token,
        chat_id="chat-1",
        origin_message_id="message-1",
    )
    assert "safe structured diagnostic" in rendered
    assert "已脱敏" in rendered


def test_completed_card_can_surface_non_blocking_reconciliation_diagnostic() -> None:
    state = reduce_card_state(
        _started_state(),
        CardEvent.completed(
            warning="历史子代理状态已由最终权威快照完成对账，不影响结果。",
            details="pending_plan_entries=0; incomplete_outer_tool_calls=0",
            detail_action={
                "action": action_ids.SHOW_ERROR_DETAILS,
                "chat_id": "chat-1",
                "origin_message_id": "message-1",
            },
        ),
    )

    assert state.terminal == "completed"
    assert state.footer.warning_type == "warning"
    assert "不影响结果" in str(state.footer.warning_banner)
    assert [button.text for button in state.buttons] == ["查看详情"]
