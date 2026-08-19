"""Terminal card diagnostics must match the actions actually rendered."""

import pytest

from src.card.actions import dispatch as action_ids
from src.card.error_diagnostics import render_error_diagnostic
from src.card.events import CardEvent, CardEventType
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


@pytest.mark.parametrize("engine_type", [None, "spec"], ids=["ordinary", "engine"])
def test_keep_alive_warning_upserts_and_clear_removes_singleton_button(
    engine_type: str | None,
) -> None:
    metadata = CardMetadata(
        mode_name="Codex" if engine_type is None else "Spec",
        tool_name="codex",
        engine_type=engine_type,
    )
    state = reduce_card_state(None, CardEvent.started(), metadata)
    if engine_type is not None:
        state = reduce_card_state(
            state,
            CardEvent.criteria_updated("验收标准", 0, 1),
        )
        assert state.engine_ext is not None
    else:
        assert state.engine_ext is None
    non_keep_alive_buttons = tuple(
        button for button in state.buttons if button.action_id != "ttl_keep_alive"
    )

    state = reduce_card_state(
        state,
        CardEvent.warning_updated(
            "即将超时",
            show_keep_alive_btn=True,
            keep_alive_minutes=7,
        ),
    )
    state = reduce_card_state(
        state,
        CardEvent.warning_updated(
            "超时窗口已更新",
            show_keep_alive_btn=True,
            keep_alive_minutes=12,
        ),
    )

    keep_alive_buttons = tuple(
        button for button in state.buttons if button.action_id == "ttl_keep_alive"
    )
    assert len(keep_alive_buttons) == 1
    assert "12" in keep_alive_buttons[0].text
    assert tuple(
        button for button in state.buttons if button.action_id != "ttl_keep_alive"
    ) == non_keep_alive_buttons
    assert state.footer.warning_banner == "超时窗口已更新"

    state = reduce_card_state(state, CardEvent.warning_updated(""))

    assert state.footer.warning_banner is None
    assert state.footer.warning_type is None
    assert all(button.action_id != "ttl_keep_alive" for button in state.buttons)
    if engine_type is not None:
        assert state.engine_ext is not None
    else:
        assert state.engine_ext is None


def test_normal_cancel_clears_transient_running_warning() -> None:
    state = reduce_card_state(
        _started_state(),
        CardEvent(
            type=CardEventType.WARNING_UPDATED,
            payload={
                "warning": (
                    "COT 过程通道暂不可用；完整过程已切换到主卡，"
                    "任务继续执行，无需重试。"
                ),
                "warning_type": "info",
            },
        ),
    )

    state = reduce_card_state(state, CardEvent.cancelled())

    assert state.terminal == "cancelled"
    assert state.footer.warning_banner is None
    assert state.footer.warning_type is None
    assert state.footer.persistent_warning is False


@pytest.mark.parametrize(
    "cancel_event",
    [
        CardEvent.cancelled(reason="ttl_expired"),
        CardEvent(
            type=CardEventType.CANCELLED,
            payload={"reason": "cancelled", "persistent_warning": True},
        ),
    ],
    ids=["ttl-expired", "explicit-persistent"],
)
def test_cancel_preserves_only_contractually_persistent_warning(
    cancel_event: CardEvent,
) -> None:
    state = reduce_card_state(
        _started_state(),
        CardEvent.warning_updated(
            "⏰ 会话关闭提示",
            show_keep_alive_btn=True,
            keep_alive_minutes=7,
        ),
    )

    state = reduce_card_state(state, cancel_event)

    assert state.footer.warning_banner == "⏰ 会话关闭提示"
    assert state.footer.warning_type == "warning"
    assert state.footer.persistent_warning is True
    assert all(button.action_id != "ttl_keep_alive" for button in state.buttons)
