"""Tests for DispatchDeliveryCoordinator deliver/failure/rejected paths."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.card.delivery.tracker import DeliveryTracker
from src.card.dispatch_coordinator import DispatchDeliveryCoordinator


def _make_coordinator(
    *,
    notify_callback=None,
    cancel_callback=None,
    reply_text_fn=None,
    reply_to=None,
):
    delivery = MagicMock()
    tracker = DeliveryTracker()
    hook_firer = MagicMock()
    ttl_handler = MagicMock()
    coord = DispatchDeliveryCoordinator(
        session_id="sess-1",
        chat_id="chat-1",
        delivery=delivery,
        tracker=tracker,
        hook_firer=hook_firer,
        ttl_handler=ttl_handler,
        notify_callback=notify_callback,
        cancel_callback=cancel_callback,
        reply_text_fn=reply_text_fn,
        reply_to=reply_to,
    )
    return coord, delivery, tracker, hook_firer, ttl_handler


class TestDeliverSuccess:
    def test_deliver_calls_delivery_engine(self):
        coord, delivery, tracker, _, _ = _make_coordinator()
        delivery.deliver.return_value = [{"kind": "ok"}]
        result = coord.deliver([{"card": "data"}])
        assert result == [{"kind": "ok"}]
        delivery.deliver.assert_called_once_with(
            session_id="sess-1",
            chat_id="chat-1",
            rendered=[{"card": "data"}],
            reply_to=None,
            is_terminal=False,
        )

    def test_deliver_propagates_is_terminal(self):
        """Terminal deliveries must flag is_terminal so the delivery engine
        stops freezing history pages (first card message reaches final state)."""
        coord, delivery, _, _, _ = _make_coordinator()
        delivery.deliver.return_value = []
        coord.deliver([{"card": "data"}], is_terminal=True)
        delivery.deliver.assert_called_once_with(
            session_id="sess-1",
            chat_id="chat-1",
            rendered=[{"card": "data"}],
            reply_to=None,
            is_terminal=True,
        )

    def test_on_success_resets_failure_count(self):
        coord, _, tracker, _, _ = _make_coordinator()
        # Simulate a failure first
        tracker.on_failure()
        assert tracker.delivery_failures == 1
        coord.on_success(is_terminal=False)
        assert tracker.delivery_failures == 0

    def test_on_success_terminal(self):
        coord, _, tracker, _, _ = _make_coordinator()
        tracker.on_failure()
        coord.on_success(is_terminal=True)
        assert tracker.delivery_failures == 0


class TestDeliverFailure:
    def test_on_failure_increments_failure_count(self):
        coord, _, tracker, _, _ = _make_coordinator()
        coord.on_failure(RuntimeError("net"), rendered=[], is_terminal=False)
        assert tracker.delivery_failures == 1

    def test_on_failure_terminal_schedules_retry(self):
        coord, _, _, _, ttl_handler = _make_coordinator()
        rendered = [{"card": "final"}]
        coord.on_failure(RuntimeError("net"), rendered=rendered, is_terminal=True)
        ttl_handler.schedule_terminal_retry.assert_called_once_with(rendered)

    def test_on_failure_non_terminal_no_retry(self):
        coord, _, _, _, ttl_handler = _make_coordinator()
        coord.on_failure(RuntimeError("net"), rendered=[], is_terminal=False)
        ttl_handler.schedule_terminal_retry.assert_not_called()


class TestNotifyRejected:
    def test_notify_rejected_calls_callback(self):
        notify = MagicMock()
        coord, _, _, _, _ = _make_coordinator(notify_callback=notify)
        result = coord.notify_rejected(engine_cmd="/deep")
        assert result is None
        notify.assert_called_once()
        args = notify.call_args[0]
        assert args[0] == "chat-1"
        assert "/deep" in args[1]

    def test_notify_rejected_deduplicates_within_window(self):
        notify = MagicMock()
        coord, _, _, _, _ = _make_coordinator(notify_callback=notify)
        coord.notify_rejected(engine_cmd="/deep")
        result = coord.notify_rejected(engine_cmd="/deep")
        # Second call within 60s returns throttle toast
        assert result is not None
        assert "toast" in result
        # notify_callback called only once
        assert notify.call_count == 1

    def test_notify_rejected_falls_back_to_reply_text(self):
        reply_fn = MagicMock()
        coord, _, _, _, _ = _make_coordinator(reply_text_fn=reply_fn, reply_to="msg-123")
        coord.notify_rejected(engine_cmd="/deep")
        reply_fn.assert_called_once()
        args = reply_fn.call_args[0]
        assert args[0] == "msg-123"
        assert "/deep" in args[1]


class TestFinalizeTerminal:
    def test_finalize_closes_delivery_and_fires_hooks(self):
        coord, delivery, _, hook_firer, _ = _make_coordinator()
        coord.finalize_terminal(state=None, terminal_reason="completed")
        delivery.close.assert_called_once_with("sess-1")
        hook_firer.fire_terminal.assert_called_once_with(None, "completed")

    def test_finalize_cancelled_calls_cancel_callback(self):
        cancel_cb = MagicMock()
        coord, _, _, _, _ = _make_coordinator(cancel_callback=cancel_cb)
        coord.finalize_terminal(state=None, terminal_reason="cancelled")
        cancel_cb.assert_called_once()

    def test_finalize_non_cancelled_no_cancel_callback(self):
        cancel_cb = MagicMock()
        coord, _, _, _, _ = _make_coordinator(cancel_callback=cancel_cb)
        coord.finalize_terminal(state=None, terminal_reason="completed")
        cancel_cb.assert_not_called()
