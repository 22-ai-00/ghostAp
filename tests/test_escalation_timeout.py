"""Unit tests for escalation timeout auto-abort flow.

Covers: card update, text notification, dirty setter, agent resume,
deadlock prevention, configurable timeout, and card_message_id serialization.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

from src.slock_engine.card_templates import build_escalation_card
from src.slock_engine.escalation_manager import EscalationManager
from src.slock_engine.models import (
    EscalationLevel,
    EscalationRequest,
)


def _make_manager(
    tmp_path=None,
    *,
    escalation_timeout_s: int = 30 * 60,
    update_card_fn=None,
    send_text_fn=None,
) -> tuple[EscalationManager, dict]:
    """Create a minimal EscalationManager with mock callbacks."""
    from src.slock_engine.task_router import TaskRouter

    lock = threading.RLock()
    escalations: list[EscalationRequest] = []
    retry_counts: dict[str, int] = {}
    mocks = {
        "dirty_setter": MagicMock(),
        "transition_agent": MagicMock(),
        "flush_if_dirty": MagicMock(),
        "execute_task_fn": MagicMock(return_value=None),
        "rollback_task_fn": MagicMock(),
        "force_complete_task_fn": MagicMock(),
    }

    router = TaskRouter()

    context = MagicMock()
    context.channel = None
    context.chat_id = "test_chat_id"
    context.dirty = False
    context.set_dirty = mocks["dirty_setter"]

    mgr = EscalationManager(
        lock=lock,
        escalations=escalations,
        retry_counts=retry_counts,
        context=context,
        router=router,
        transition_agent=mocks["transition_agent"],
        flush_if_dirty=mocks["flush_if_dirty"],
        update_card_fn=update_card_fn,
        send_text_fn=send_text_fn,
        escalation_timeout_s=escalation_timeout_s,
    )
    mgr.set_task_callbacks(
        execute_task_fn=mocks["execute_task_fn"],
        rollback_task_fn=mocks["rollback_task_fn"],
        force_complete_task_fn=mocks["force_complete_task_fn"],
    )
    return mgr, mocks


def _make_escalation(
    agent_name: str = "TestAgent",
    task_id: str = "task-001",
    card_message_id: str = "msg_abc123",
) -> EscalationRequest:
    return EscalationRequest(
        agent_id="agent-001",
        agent_name=agent_name,
        task_id=task_id,
        level=EscalationLevel.BLOCKED,
        reason="Cannot access resource",
        context="Error details here",
        options=["重试", "跳过", "中止"],
        card_message_id=card_message_id,
    )


class TestTimeoutAutoAbort:
    """Tests for _timeout_auto_abort method."""

    def test_timeout_auto_abort_updates_card(self):
        """AC-1: Timeout updates card to resolved state with green header."""
        update_card_fn = MagicMock(return_value=True)
        mgr, mocks = _make_manager(update_card_fn=update_card_fn)

        esc = _make_escalation()
        mgr._escalations.append(esc)

        mgr._timeout_auto_abort(esc.escalation_id)
        mgr._io_executor.shutdown(wait=True)

        update_card_fn.assert_called_once()
        call_args = update_card_fn.call_args
        assert call_args[0][0] == "msg_abc123"  # message_id
        card_json = call_args[0][1]
        card = json.loads(card_json)
        assert "[已解决]" in card["header"]["title"]["content"]
        assert card["header"]["template"] == "green"

    def test_timeout_auto_abort_sends_text(self):
        """AC-2: Timeout sends text notification with agent name and reason."""
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(send_text_fn=send_text_fn)

        esc = _make_escalation(agent_name="CoderBot")
        mgr._escalations.append(esc)

        mgr._timeout_auto_abort(esc.escalation_id)
        mgr._io_executor.shutdown(wait=True)

        send_text_fn.assert_called_once()
        call_args = send_text_fn.call_args
        assert call_args[0][0] == "test_chat_id"  # chat_id
        text = call_args[0][1]
        assert "CoderBot" in text
        assert "超时" in text
        assert "Cannot access resource" in text

    def test_timeout_auto_abort_calls_dirty_setter(self):
        """AC-4: Timeout triggers dirty flag for status panel refresh."""
        mgr, mocks = _make_manager()

        esc = _make_escalation()
        mgr._escalations.append(esc)

        mgr._timeout_auto_abort(esc.escalation_id)

        mocks["dirty_setter"].assert_called_with(True)

    def test_timeout_auto_abort_resumes_agent(self):
        """AC-3: Timeout triggers resume_after_escalation (abort branch sets dirty)."""
        mgr, mocks = _make_manager()

        esc = _make_escalation(task_id="task-xyz")
        mgr._escalations.append(esc)

        mgr._timeout_auto_abort(esc.escalation_id)
        mgr._io_executor.shutdown(wait=True)

        # Abort branch in resume_after_escalation calls set_dirty(True)
        mocks["dirty_setter"].assert_called_with(True)

    def test_timeout_marks_resolved(self):
        """Timeout marks escalation as resolved with resolution='中止'."""
        mgr, mocks = _make_manager()

        esc = _make_escalation()
        mgr._escalations.append(esc)

        mgr._timeout_auto_abort(esc.escalation_id)

        assert esc.resolved is True
        assert esc.resolution == "中止"
        assert esc.resolved_at is not None

    def test_timeout_already_resolved_noop(self):
        """Already-resolved escalation: timeout does nothing."""
        update_card_fn = MagicMock(return_value=True)
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(
            update_card_fn=update_card_fn, send_text_fn=send_text_fn,
        )

        esc = _make_escalation()
        esc.resolved = True
        esc.resolution = "跳过"
        mgr._escalations.append(esc)

        mgr._timeout_auto_abort(esc.escalation_id)

        update_card_fn.assert_not_called()
        send_text_fn.assert_not_called()
        mocks["dirty_setter"].assert_not_called()
        mocks["force_complete_task_fn"].assert_not_called()

    def test_timeout_no_deadlock(self):
        """Timer thread completes without deadlock using real RLock."""
        mgr, mocks = _make_manager(escalation_timeout_s=1)  # 1 second

        esc = _make_escalation()
        mgr._escalations.append(esc)

        # Start timer and wait for it to fire
        mgr._start_timeout_timer(esc)

        # Wait up to 3 seconds for the timer to fire
        deadline = time.time() + 3
        while not esc.resolved and time.time() < deadline:
            time.sleep(0.1)

        assert esc.resolved is True
        assert esc.resolution == "中止"

    def test_timeout_card_update_failure_does_not_block(self):
        """Card update failure doesn't prevent text notification or agent resume."""
        update_card_fn = MagicMock(side_effect=Exception("API error"))
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(
            update_card_fn=update_card_fn, send_text_fn=send_text_fn,
        )

        esc = _make_escalation()
        mgr._escalations.append(esc)

        # Should not raise
        mgr._timeout_auto_abort(esc.escalation_id)
        mgr._io_executor.shutdown(wait=True)

        # Text and resume still happen despite card failure
        send_text_fn.assert_called_once()
        mocks["dirty_setter"].assert_called_with(True)

    def test_timeout_no_card_message_id_skips_update(self):
        """When card_message_id is None, card update is skipped but text is sent."""
        update_card_fn = MagicMock(return_value=True)
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(
            update_card_fn=update_card_fn, send_text_fn=send_text_fn,
        )

        esc = _make_escalation(card_message_id=None)
        mgr._escalations.append(esc)

        mgr._timeout_auto_abort(esc.escalation_id)
        mgr._io_executor.shutdown(wait=True)

        update_card_fn.assert_not_called()  # skipped
        send_text_fn.assert_called_once()  # still fires

    def test_timeout_send_text_failure_still_resumes_agent(self):
        """send_text_fn exception doesn't block resume_after_escalation."""
        update_card_fn = MagicMock(return_value=True)
        send_text_fn = MagicMock(side_effect=Exception("Network timeout"))
        mgr, mocks = _make_manager(
            update_card_fn=update_card_fn, send_text_fn=send_text_fn,
        )

        esc = _make_escalation(task_id="task-send-fail")
        mgr._escalations.append(esc)

        # Should not raise
        mgr._timeout_auto_abort(esc.escalation_id)
        mgr._io_executor.shutdown(wait=True)

        # send_text_fn was called (and raised)
        send_text_fn.assert_called_once()
        # Abort branch sets dirty despite send_text_fn failure
        mocks["dirty_setter"].assert_called_with(True)

    def test_timeout_abort_branch_sets_dirty_and_logs(self):
        """Abort branch in resume_after_escalation sets dirty and logs info."""
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(send_text_fn=send_text_fn)

        esc = _make_escalation(task_id="task-abort-test")
        mgr._escalations.append(esc)

        with patch("src.slock_engine.escalation_manager.logger") as mock_logger:
            mgr._timeout_auto_abort(esc.escalation_id)
            mgr._io_executor.shutdown(wait=True)

            # Abort branch sets dirty
            mocks["dirty_setter"].assert_called_with(True)
            # Info log about abort
            mock_logger.info.assert_called()
            info_calls = [str(c) for c in mock_logger.info.call_args_list]
            assert any("Abort" in c or "abandoned" in c for c in info_calls)


class TestBuildEscalationCardTimeout:
    """Tests for timeout hint in escalation card."""

    def test_card_has_timeout_hint(self):
        """AC-5: Card includes timeout hint with configured minutes."""
        esc = _make_escalation()
        card = build_escalation_card(esc, channel_id="ch1", timeout_minutes=30)

        card_str = json.dumps(card, ensure_ascii=False)
        assert "30 分钟后自动中止" in card_str

    def test_card_no_timeout_hint_when_none(self):
        """No timeout hint when timeout_minutes is None."""
        esc = _make_escalation()
        card = build_escalation_card(esc, channel_id="ch1", timeout_minutes=None)

        card_str = json.dumps(card, ensure_ascii=False)
        assert "分钟后自动中止" not in card_str

    def test_card_custom_timeout_value(self):
        """AC-6: Custom timeout value is correctly displayed."""
        esc = _make_escalation()
        card = build_escalation_card(esc, channel_id="ch1", timeout_minutes=10)

        card_str = json.dumps(card, ensure_ascii=False)
        assert "10 分钟后自动中止" in card_str


class TestConfigurableTimeout:
    """Tests for configurable escalation timeout."""

    def test_custom_timeout_used_in_timer(self):
        """AC-6: Custom timeout_s is passed to Timer."""
        mgr, mocks = _make_manager(escalation_timeout_s=600)

        esc = _make_escalation()
        mgr._escalations.append(esc)

        with patch("src.slock_engine.escalation_manager._threading.Timer") as MockTimer:
            mock_timer = MagicMock()
            MockTimer.return_value = mock_timer
            mgr._start_timeout_timer(esc)
            # First call is the full timeout timer
            first_call = MockTimer.call_args_list[0]
            assert first_call[0][0] == 600
            assert first_call[0][1] == mgr._timeout_auto_abort

    def test_default_timeout_is_1800(self):
        """AC-7: Default timeout remains 30 minutes (1800s)."""
        mgr, mocks = _make_manager()
        assert mgr._escalation_timeout_s == 1800

    def test_get_escalation_card_passes_timeout_minutes(self):
        """get_escalation_card passes correct timeout_minutes to card builder."""
        mgr, mocks = _make_manager(escalation_timeout_s=600)

        esc = _make_escalation()
        card = mgr.get_escalation_card(esc)

        card_str = json.dumps(card, ensure_ascii=False)
        assert "10 分钟后自动中止" in card_str


class TestCardMessageIdSerialization:
    """Tests for card_message_id field in EscalationRequest."""

    def test_to_dict_includes_card_message_id(self):
        """AC-9: to_dict includes card_message_id."""
        esc = _make_escalation(card_message_id="msg_xyz")
        d = esc.to_dict()
        assert d["card_message_id"] == "msg_xyz"

    def test_to_dict_card_message_id_none(self):
        """to_dict handles None card_message_id."""
        esc = _make_escalation(card_message_id=None)
        d = esc.to_dict()
        assert d["card_message_id"] is None

    def test_from_dict_with_card_message_id(self):
        """from_dict restores card_message_id."""
        esc = _make_escalation(card_message_id="msg_round_trip")
        d = esc.to_dict()
        restored = EscalationRequest.from_dict(d)
        assert restored.card_message_id == "msg_round_trip"

    def test_from_dict_without_card_message_id(self):
        """from_dict handles missing card_message_id (backward compat)."""
        d = {
            "escalation_id": "test-id",
            "agent_id": "a1",
            "agent_name": "Bot",
            "level": "blocked",
            "reason": "test",
        }
        esc = EscalationRequest.from_dict(d)
        assert esc.card_message_id is None


class TestHalfTimeReminder:
    """Tests for _half_time_reminder method."""

    def test_half_time_reminder_sends_notification(self):
        """Half-time reminder sends text notification with remaining time."""
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(
            escalation_timeout_s=1800, send_text_fn=send_text_fn,
        )

        esc = _make_escalation(agent_name="WorkerBot")
        mgr._escalations.append(esc)

        mgr._half_time_reminder(esc.escalation_id)

        send_text_fn.assert_called_once()
        call_args = send_text_fn.call_args
        assert call_args[0][0] == "test_chat_id"
        text = call_args[0][1]
        assert "WorkerBot" in text
        assert "即将超时" in text

    def test_half_time_reminder_skips_resolved(self):
        """Half-time reminder does nothing if escalation is already resolved."""
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(send_text_fn=send_text_fn)

        esc = _make_escalation()
        esc.resolved = True
        esc.resolution = "跳过"
        mgr._escalations.append(esc)

        mgr._half_time_reminder(esc.escalation_id)

        send_text_fn.assert_not_called()

    def test_half_time_reminder_skips_missing_escalation(self):
        """Half-time reminder does nothing if escalation_id not found."""
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(send_text_fn=send_text_fn)

        mgr._half_time_reminder("nonexistent-id")

        send_text_fn.assert_not_called()

    def test_start_timeout_creates_both_timers(self):
        """_start_timeout_timer creates full timeout and half-time timers."""
        mgr, mocks = _make_manager(escalation_timeout_s=600)
        esc = _make_escalation()
        mgr._escalations.append(esc)

        with patch("src.slock_engine.escalation_manager._threading.Timer") as MockTimer:
            mock_timer = MagicMock()
            MockTimer.return_value = mock_timer
            mgr._start_timeout_timer(esc)

            # Should be called twice: once for full timeout, once for half-time
            assert MockTimer.call_count == 2
            calls = MockTimer.call_args_list
            # Full timeout: 600s, target=_timeout_auto_abort
            assert calls[0][0][0] == 600
            assert calls[0][0][1] == mgr._timeout_auto_abort
            # Half-time: 300s, target=_half_time_reminder
            assert calls[1][0][0] == 300.0
            assert calls[1][0][1] == mgr._half_time_reminder

    def test_cancel_timeout_cancels_both_timers(self):
        """_cancel_timeout_timer cancels both timeout and half-time timers."""
        mgr, mocks = _make_manager(escalation_timeout_s=600)
        esc = _make_escalation()
        mgr._escalations.append(esc)

        mock_full_timer = MagicMock()
        mock_half_timer = MagicMock()
        mgr._timeout_timers[esc.escalation_id] = mock_full_timer
        mgr._half_timers[esc.escalation_id] = mock_half_timer

        mgr._cancel_timeout_timer(esc.escalation_id)

        mock_full_timer.cancel.assert_called_once()
        mock_half_timer.cancel.assert_called_once()
        assert esc.escalation_id not in mgr._timeout_timers
        assert esc.escalation_id not in mgr._half_timers


class TestTimeoutReasonPropagation:
    """Tests for reason kwarg passing through force_complete_task."""

    def test_abort_branch_passes_reason(self):
        """resume_after_escalation abort branch sets dirty and logs abandon."""
        mgr, mocks = _make_manager()

        esc = _make_escalation(task_id="task-abort")
        esc.resolved = True
        esc.resolution = "中止"
        mgr._escalations.append(esc)

        mgr.resume_after_escalation(esc)

        # Abort branch sets dirty (task marked DONE/abandoned)
        mocks["dirty_setter"].assert_called_with(True)

    def test_retry_limit_passes_reason(self):
        """Retry limit exceeded passes reason='重试次数超限'."""
        mgr, mocks = _make_manager()

        esc = _make_escalation(task_id="task-retry")
        esc.resolved = True
        esc.resolution = "重试"
        mgr._escalations.append(esc)

        # Exhaust retries
        retry_key = f"esc_retry:{esc.escalation_id}"
        mgr._escalation_retry_counts[retry_key] = mgr._MAX_ESCALATION_RETRIES

        mgr.resume_after_escalation(esc)

        mocks["force_complete_task_fn"].assert_called_once_with(
            "task-retry", reason="重试次数超限", actor_id="system:escalation",
        )


class TestResumeExceptionAlert:
    """Tests for _do_timeout_io fallback when resume_after_escalation raises."""

    def test_resume_failure_triggers_fallback_dirty_and_alert(self):
        """When resume raises, fallback sets dirty and sends alert text."""
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(send_text_fn=send_text_fn)

        esc = _make_escalation(task_id="task-resume-fail")
        mgr._escalations.append(esc)

        # Make resume_after_escalation raise
        with patch.object(mgr, "resume_after_escalation", side_effect=RuntimeError("boom")):
            mgr._timeout_auto_abort(esc.escalation_id)
            mgr._io_executor.shutdown(wait=True)

        # Fallback sets dirty
        mocks["dirty_setter"].assert_called_with(True)
        # Alert text is sent (second call after timeout notification)
        assert send_text_fn.call_count == 2
        alert_text = send_text_fn.call_args_list[1][0][1]
        assert "系统告警" in alert_text or "🚨" in alert_text

    def test_resume_failure_sends_alert_text(self):
        """When resume raises, alert text is sent to chat."""
        send_text_fn = MagicMock()
        mgr, mocks = _make_manager(send_text_fn=send_text_fn)

        esc = _make_escalation(task_id="task-alert", agent_name="FailBot")
        mgr._escalations.append(esc)

        with patch.object(mgr, "resume_after_escalation", side_effect=RuntimeError("boom")):
            mgr._timeout_auto_abort(esc.escalation_id)
            mgr._io_executor.shutdown(wait=True)

        # send_text_fn called twice: timeout notification + alert
        assert send_text_fn.call_count == 2
        alert_call = send_text_fn.call_args_list[1]
        alert_text = alert_call[0][1]
        assert "系统告警" in alert_text or "🚨" in alert_text
        assert "FailBot" in alert_text

    def test_resume_failure_both_fallbacks_fail_no_crash(self):
        """When resume, fallback force_complete, AND alert all fail, no exception propagates."""
        send_text_fn = MagicMock()
        # First call succeeds (timeout text), second call fails (alert)
        send_text_fn.side_effect = [None, Exception("send failed")]
        mgr, mocks = _make_manager(send_text_fn=send_text_fn)
        mocks["force_complete_task_fn"].side_effect = Exception("force_complete failed")

        esc = _make_escalation(task_id="task-cascade-fail")
        mgr._escalations.append(esc)

        with patch.object(mgr, "resume_after_escalation", side_effect=RuntimeError("boom")):
            # Should not raise
            mgr._timeout_auto_abort(esc.escalation_id)
            mgr._io_executor.shutdown(wait=True)
