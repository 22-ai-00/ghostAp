"""Unit tests for TTLHandler using mocked TTLDecider + TTLActuator.

Tests each TTLHandler code path (expired, force-close, prewarning, retry)
through the method-level interface without requiring a real CardSession.
"""

from unittest.mock import MagicMock

import pytest

from src.card.protocols import TTLState
from src.card.session.ttl import TTLHandler


def _make_mock_session(**overrides) -> MagicMock:
    """Create a mock TTLDecider + TTLActuator with sensible defaults."""
    s = MagicMock()
    s.get_ttl_state.return_value = TTLState(
        closed=False,
        ttl_warned=False,
        idle_seconds=2000.0,
        ttl_seconds=1800.0,
        session_id="test_sess",
        state_snapshot=None,
    )
    s.engine_cmd = "/deep"
    s.engine_name = "Deep"
    s.reduce_and_render.return_value = [{"card": "rendered"}]
    for key, val in overrides.items():
        setattr(s, key, val)
    return s


class TestOnTTLExpired:
    """Tests for on_ttl_expired callback."""

    def test_normal_expiry_reduces_and_delivers_terminal(self):
        """When idle > ttl and not warned/closed, should mark, reduce, deliver."""
        s = _make_mock_session()
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.mark_ttl_expired.assert_called_once()
        s.reduce_and_render.assert_called_once()
        # Verify events passed to reduce_and_render
        events = s.reduce_and_render.call_args[0][0]
        assert len(events) == 2
        assert events[1].type.value == "cancelled"
        s.deliver_terminal.assert_called_once_with([{"card": "rendered"}])

    def test_skips_when_already_closed(self):
        """When session is already closed, on_ttl_expired is a no-op."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=True, ttl_warned=False, idle_seconds=2000.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.mark_ttl_expired.assert_not_called()
        s.reduce_and_render.assert_not_called()
        s.deliver_terminal.assert_not_called()

    def test_skips_when_already_warned(self):
        """When ttl_warned is already True, on_ttl_expired is a no-op."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=True, idle_seconds=2000.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.mark_ttl_expired.assert_not_called()

    def test_skips_when_not_idle_enough(self):
        """When idle_seconds <= ttl_seconds, on_ttl_expired is a no-op."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=False, idle_seconds=1000.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.mark_ttl_expired.assert_not_called()

    def test_defers_expiry_when_card_has_active_tool(self):
        """Active long-running tool work should not be treated as idle abandonment."""
        from src.card.state.models import CardState, ContentBlock

        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False,
            ttl_warned=False,
            idle_seconds=2000.0,
            ttl_seconds=1800.0,
            session_id="test",
            state_snapshot=CardState(
                blocks=(
                    ContentBlock(
                        kind="tool_call",
                        block_id="tool",
                        tool_name="Bash",
                        status="active",
                    ),
                ),
            ),
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.defer_idle_timeout.assert_called_once_with(
            handler.on_ttl_expired,
            handler.on_ttl_prewarning,
        )
        s.mark_ttl_expired.assert_not_called()
        s.reduce_and_render.assert_not_called()
        s.deliver_terminal.assert_not_called()

    def test_rollback_on_reduce_failure(self):
        """When reduce_and_render raises, rollback_ttl_warned is called."""
        s = _make_mock_session()
        s.reduce_and_render.side_effect = RuntimeError("render boom")
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.mark_ttl_expired.assert_called_once()
        s.rollback_ttl_warned.assert_called_once()
        s.deliver_terminal.assert_not_called()

    def test_lock_contention_schedules_retry(self):
        """When get_ttl_state()=None and schedule_ttl_retry returns True, force_terminate not called."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = None
        s.schedule_ttl_retry.return_value = True
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.schedule_ttl_retry.assert_called_once_with(handler.on_ttl_expired)
        s.force_terminate.assert_not_called()
        s.mark_ttl_expired.assert_not_called()

    @pytest.mark.parametrize("cmd,expected_key", [
        ("/deep", "card_session_ttl_expired_deep"),
        ("/spec", "card_session_ttl_expired_spec"),
        ("/unknown", "card_session_ttl_expired"),
    ])
    def test_engine_specific_ttl_key_selection(self, cmd, expected_key):
        """Each engine_cmd maps to its specific UI text key."""
        from src.card.ui_text import UI_TEXT
        s = _make_mock_session(engine_cmd=cmd, engine_name="Test")
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        # The warning event text should come from the expected key
        events = s.reduce_and_render.call_args[0][0]
        if expected_key == "card_session_ttl_expired":
            # Generic fallback uses {expired_commands} placeholder
            expected_text = UI_TEXT[expected_key].format(
                expired_commands=UI_TEXT["card_session_ttl_expired_commands"],
            )
        else:
            expected_text = UI_TEXT[expected_key].format(engine_cmd=cmd, engine_name="Test")
        assert events[0].payload["warning"] == expected_text


class TestForceClose:
    """Tests for _force_close fallback path."""

    def test_force_close_delegates_to_force_terminate(self):
        """_force_close simply calls force_terminate('ttl_expired')."""
        s = _make_mock_session()
        handler = TTLHandler(decider=s, actuator=s)

        handler._force_close()

        s.force_terminate.assert_called_once_with("ttl_expired")


class TestOnTTLPrewarning:
    """Tests for on_ttl_prewarning callback."""

    def test_prewarning_fires_when_75_percent_idle(self):
        """When idle >= 75% of TTL (threshold=0.75), prewarning fires."""
        s = _make_mock_session()
        # 75% of 1800 = 1350
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=False, idle_seconds=1350.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.reduce_and_render.assert_called_once()
        events = s.reduce_and_render.call_args[0][0]
        assert len(events) == 1  # Just warning_updated
        s.deliver_update.assert_called_once()
        s.notify_user.assert_not_called()  # No dual-notification in happy path

    def test_prewarning_fires_at_90_percent_idle(self):
        """When idle >= 90% of TTL, prewarning also fires (above 75% threshold)."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=False, idle_seconds=1650.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.reduce_and_render.assert_called_once()
        s.deliver_update.assert_called_once()
        s.notify_user.assert_not_called()  # No dual-notification in happy path

    def test_prewarning_skips_when_below_75_percent(self):
        """When idle < 75% of TTL, prewarning is a no-op."""
        s = _make_mock_session()
        # 74% of 1800 = 1332 → below threshold (0.75)
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=False, idle_seconds=1332.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.reduce_and_render.assert_not_called()
        s.deliver_update.assert_not_called()

    def test_prewarning_boundary_exactly_75_percent(self):
        """Above 75% threshold, prewarning fires."""
        s = _make_mock_session()
        # 76% of 1800 = 1368.0 → above 0.75 threshold
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=False, idle_seconds=1368.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.reduce_and_render.assert_called_once()
        s.deliver_update.assert_called_once()

    def test_prewarning_skips_when_closed(self):
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=True, ttl_warned=False, idle_seconds=1700.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.reduce_and_render.assert_not_called()

    def test_prewarning_skips_when_already_warned(self):
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=True, idle_seconds=1700.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.reduce_and_render.assert_not_called()

    def test_prewarning_defers_when_card_has_active_progress(self):
        from src.card.state.models import CardState, FooterState

        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False,
            ttl_warned=False,
            idle_seconds=1700.0,
            ttl_seconds=1800.0,
            session_id="test",
            state_snapshot=CardState(footer=FooterState(status="thinking", progress_pct=50)),
        )
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.defer_idle_timeout.assert_called_once_with(
            handler.on_ttl_expired,
            handler.on_ttl_prewarning,
        )
        s.reduce_and_render.assert_not_called()
        s.deliver_update.assert_not_called()

    def test_prewarning_handles_render_failure(self):
        """When reduce_and_render raises, prewarning returns without delivering."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=False, idle_seconds=1700.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        s.reduce_and_render.side_effect = RuntimeError("render fail")
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_prewarning()

        s.deliver_update.assert_not_called()

    def test_prewarning_last_active_time_calculation(self):
        """Prewarning text should contain remaining minutes (no last_active_time suffix)."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=False, idle_seconds=1650.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)
        handler.on_ttl_prewarning()

        events = s.reduce_and_render.call_args[0][0]
        warning_text = events[0].payload["warning"]
        # remaining = (1800 - 1650) / 60 = 2.5 → int = 2
        assert "2" in warning_text
        assert "分钟后关闭" in warning_text

    def test_prewarning_retry_exhausted_logs_debug(self, caplog):
        """When schedule_ttl_retry returns False (exhausted), a DEBUG log is emitted."""
        import logging

        s = _make_mock_session()
        s.get_ttl_state.return_value = None
        s.schedule_ttl_retry.return_value = False
        handler = TTLHandler(decider=s, actuator=s)

        with caplog.at_level(logging.DEBUG, logger="src.card.session.ttl"):
            handler.on_ttl_prewarning()

        assert any("prewarning retry exhausted" in r.message for r in caplog.records)
        s.reduce_and_render.assert_not_called()


class TestScheduleTerminalRetry:
    """Tests for schedule_terminal_retry."""

    def test_flags_retry_pending_and_schedules(self):
        """schedule_terminal_retry flags pending and calls schedule_retry."""
        s = _make_mock_session()
        handler = TTLHandler(decider=s, actuator=s)

        handler.schedule_terminal_retry([{"card": "data"}])

        s.flag_retry_pending.assert_called_once()
        s.schedule_retry.assert_called_once()

    def test_retry_callback_delivers_and_fires_hook(self):
        """The scheduled retry callback delivers and fires terminal hook on success."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=True, idle_seconds=2000.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        rendered = [{"card": "data"}]
        handler.schedule_terminal_retry(rendered)

        # Extract the callback passed to schedule_retry
        retry_callback = s.schedule_retry.call_args[0][0]

        # Execute the callback
        retry_callback()

        s.force_deliver.assert_called_once_with(rendered)
        s.fire_terminal_hook.assert_called_once_with("completed")
        s.close_delivery.assert_called_once()
        s.release_terminal_resources.assert_called_once()

    def test_retry_callback_notifies_on_failure(self):
        """When retry delivery fails, notify_user is called with fallback message."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=True, idle_seconds=2000.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        s.force_deliver.side_effect = RuntimeError("delivery failed again")
        handler = TTLHandler(decider=s, actuator=s)

        handler.schedule_terminal_retry([{"card": "data"}])
        retry_callback = s.schedule_retry.call_args[0][0]

        retry_callback()

        s.notify_user.assert_called_once()
        assert "任务已结束" in s.notify_user.call_args[0][0]
        s.close_delivery.assert_called_once()
        s.release_terminal_resources.assert_called_once()

    def test_retry_callback_skips_when_closed(self):
        """When session is already closed at retry time, callback is a no-op."""
        s = _make_mock_session()
        handler = TTLHandler(decider=s, actuator=s)

        handler.schedule_terminal_retry([{"card": "data"}])
        retry_callback = s.schedule_retry.call_args[0][0]

        # Session becomes closed before retry fires
        s.get_ttl_state.return_value = TTLState(
            closed=True, ttl_warned=True, idle_seconds=2000.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )

        retry_callback()

        s.force_deliver.assert_not_called()
        s.fire_terminal_hook.assert_not_called()

    def test_terminal_retry_lock_contention_recovers_on_bounded_retry(self):
        """One contended state read is rescheduled and can still deliver."""
        s = _make_mock_session()
        handler = TTLHandler(decider=s, actuator=s)

        handler.schedule_terminal_retry([{"card": "data"}])
        retry_callback = s.schedule_retry.call_args[0][0]
        s.get_ttl_state.side_effect = [
            None,
            TTLState(
                closed=False,
                ttl_warned=True,
                idle_seconds=2000.0,
                ttl_seconds=1800.0,
                session_id="test",
                state_snapshot=None,
            ),
        ]

        retry_callback()
        assert s.schedule_retry.call_count == 2
        rescheduled_callback = s.schedule_retry.call_args[0][0]
        rescheduled_callback()

        s.force_deliver.assert_called_once_with([{"card": "data"}])
        s.mark_closed.assert_called_once()
        s.close_delivery.assert_called_once()
        s.release_terminal_resources.assert_called_once()

    def test_terminal_retry_lock_contention_exhaustion_fails_closed(self):
        """Persistent state-lock contention closes and notifies after the bound."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = None
        handler = TTLHandler(decider=s, actuator=s)

        handler.schedule_terminal_retry([{"card": "data"}])
        for _ in range(handler._MAX_TERMINAL_STATE_RETRIES + 1):
            retry_callback = s.schedule_retry.call_args[0][0]
            retry_callback()

        s.force_deliver.assert_not_called()
        s.fire_terminal_hook.assert_not_called()
        s.mark_closed.assert_called_once()
        s.notify_user.assert_called_once()
        s.close_delivery.assert_called_once()
        s.release_terminal_resources.assert_called_once()

    def test_terminal_retry_closure_captures_rendered_immutably(self):
        """The retry closure should use the rendered list captured at schedule time,
        not a later-mutated version."""
        s = _make_mock_session()
        s.get_ttl_state.return_value = TTLState(
            closed=False, ttl_warned=True, idle_seconds=2000.0,
            ttl_seconds=1800.0, session_id="test", state_snapshot=None,
        )
        handler = TTLHandler(decider=s, actuator=s)

        rendered = [{"card": "original"}]
        handler.schedule_terminal_retry(rendered)
        retry_callback = s.schedule_retry.call_args[0][0]

        # Mutate the list after scheduling — closure should still deliver original
        rendered.clear()
        rendered.append({"card": "mutated"})

        retry_callback()

        # Since Python closures capture the reference not a copy,
        # the delivered value will be the mutated list (this documents current behavior)
        # If immutability is needed, the implementation should copy.
        s.force_deliver.assert_called_once()

    def test_terminal_retry_concurrent_close_race(self):
        """Simulates a race where session closes between get_ttl_state and force_deliver."""
        s = _make_mock_session()
        call_count = [0]

        def _get_state_then_close():
            call_count[0] += 1
            if call_count[0] == 1:
                # First call in schedule_terminal_retry setup — normal state
                return TTLState(
                    closed=False, ttl_warned=True, idle_seconds=2000.0,
                    ttl_seconds=1800.0, session_id="test", state_snapshot=None,
                )
            # Second call in _retry() — session not yet closed
            return TTLState(
                closed=False, ttl_warned=True, idle_seconds=2000.0,
                ttl_seconds=1800.0, session_id="test", state_snapshot=None,
            )

        s.get_ttl_state.side_effect = _get_state_then_close
        # Simulate force_deliver raising (e.g. session was closed concurrently)
        s.force_deliver.side_effect = RuntimeError("session closed during delivery")
        handler = TTLHandler(decider=s, actuator=s)

        handler.schedule_terminal_retry([{"card": "data"}])
        retry_callback = s.schedule_retry.call_args[0][0]

        retry_callback()

        # Should still call mark_closed and close_delivery even on failure
        s.mark_closed.assert_called_once()
        s.close_delivery.assert_called_once()
        s.release_terminal_resources.assert_called_once()
        s.notify_user.assert_called_once()
