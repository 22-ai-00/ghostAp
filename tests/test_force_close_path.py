"""Tests for TTL force-close path when lock acquisition retries are exhausted.

Validates that CardSession properly force-closes when _on_ttl_expired
cannot acquire the session lock after max retries.
"""

import threading
import time
from unittest.mock import patch

import pytest

from src.card.delivery.engine import CardDelivery
from src.card.events import CardEvent
from src.card.session import CardSession
from src.card.session._ttl_mixin import TTLActuator
from src.card.session.config import SessionCallbacks, SessionConfig
from src.card.state.models import CardMetadata
from src.card.types import RenderedCard


@pytest.fixture(autouse=True)
def _fast_ttl_lock_timeout(monkeypatch):
    """Reduce TTL lock acquire timeout from 1s to 0.01s to speed up tests."""
    monkeypatch.setattr(TTLActuator, "_LOCK_ACQUIRE_TIMEOUT", 0.01)

class _MockClient:
    """Minimal mock Feishu client for CardSession tests."""

    def __init__(self):
        self.created = []
        self.updated = []

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self.created.append((chat_id, card_json))
        return ("msg_1", "card_1")

    def update_card(self, card_id, card_json, *, sequence=0):
        self.updated.append((card_id, card_json))


class TestForceClosePath:
    """TTL force-close when lock cannot be acquired."""

    def test_force_close_renders_and_reports_frozen_session_duration(self):
        clock = [100.0]
        terminal_states = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                terminal_states.append((state, reason))

        client = _MockClient()
        delivery = CardDelivery(client)
        config = SessionConfig(
            metadata=CardMetadata(
                engine_type="deep",
                mode_name="Deep",
                mode_emoji="🔍",
            ),
            clock=lambda: clock[0],
            sync_delivery=True,
        )
        session = CardSession(
            chat_id="chat_elapsed",
            config=config,
            delivery=delivery,
            session_id="force_close_elapsed",
            hooks=(TrackingHook(),),
        )
        session.dispatch(CardEvent.started())
        clock[0] = 158.0

        session._lock.acquire()
        try:
            session._ttl_actuator.force_terminate("ttl_expired")
        finally:
            session._lock.release()

        assert client.updated
        assert "⏱ 用时 00:00:58" in str(client.updated[-1][1])
        terminal_state, terminal_reason = terminal_states[-1]
        assert terminal_reason == "ttl_expired"
        assert terminal_state.terminal == "cancelled"
        assert terminal_state.footer.duration_seconds == 58.0

    def test_force_close_sets_closed_flag(self):
        """When TTL retries are exhausted, session is marked closed."""
        client = _MockClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.1)
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="force_close_test",
        )
        # Start session
        session.dispatch(CardEvent.started())

        # Simulate lock being held forever to trigger force-close
        # Acquire the session lock and hold it
        session._lock.acquire()

        # Manually trigger TTL expired (simulating timer fire)
        # First call will fail to acquire lock and schedule retry
        session._ttl_handler.on_ttl_expired()

        # Timer manager should have scheduled a retry
        assert session._timers._ttl_retry_count == 1

        # Exhaust retries
        from src.card.timers.manager import _MAX_TTL_RETRIES
        for _ in range(_MAX_TTL_RETRIES - 1):
            session._ttl_handler.on_ttl_expired()

        # Next call should trigger force-close
        session._ttl_handler.on_ttl_expired()

        # Release the lock (cleanup)
        session._lock.release()

        # Session should be marked as closed
        assert session._closed.is_set()

    def test_force_close_calls_delivery_close(self):
        """Force-close path calls delivery.close()."""
        client = _MockClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🚀")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.1)
        session = CardSession(
            chat_id="chat_2",
            config=config,
            delivery=delivery,
            session_id="force_close_delivery",
        )
        session.dispatch(CardEvent.started())

        # Hold the lock
        session._lock.acquire()

        # Exhaust all retries
        from src.card.timers.manager import _MAX_TTL_RETRIES
        for _ in range(_MAX_TTL_RETRIES + 1):
            session._ttl_handler.on_ttl_expired()

        session._lock.release()

        # Verify delivery is closed: deliver returns [] and create_card is not called again
        rendered = [RenderedCard(_card_json={"body": {}}, structure_signature="sig1", page_index=0)]
        created_before = len(client.created)
        result = delivery.deliver("force_close_delivery", "chat_2", rendered)
        assert result == []
        assert len(client.created) == created_before  # no new API calls

    def test_force_close_fires_terminal_hooks(self):
        """Force-close path fires terminal hooks with 'ttl_expired' reason."""
        client = _MockClient()
        delivery = CardDelivery(client)
        hook_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                hook_calls.append(reason)

        metadata = CardMetadata(engine_type="spec", mode_name="Spec", mode_emoji="📋")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.1)
        session = CardSession(
            chat_id="chat_3",
            config=config,
            delivery=delivery,
            session_id="force_close_hooks",
            hooks=(TrackingHook(),),
        )
        session.dispatch(CardEvent.started())

        # Hold the lock
        session._lock.acquire()

        # Exhaust all retries
        from src.card.timers.manager import _MAX_TTL_RETRIES
        for _ in range(_MAX_TTL_RETRIES + 1):
            session._ttl_handler.on_ttl_expired()

        session._lock.release()

        assert "ttl_expired" in hook_calls

    def test_force_close_survives_delivery_exception(self):
        """Force-close doesn't crash even if delivery.close() raises."""
        class BrokenClient:
            def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
                return ("msg_1", "card_1")
            def update_card(self, card_id, card_json, *, sequence=0):
                pass

        delivery = CardDelivery(BrokenClient())
        metadata = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.1)
        session = CardSession(
            chat_id="chat_4",
            config=config,
            delivery=delivery,
            session_id="force_close_broken",
        )
        session.dispatch(CardEvent.started())

        # Patch delivery.close to raise
        with patch.object(delivery, 'close', side_effect=RuntimeError("delivery broken")):
            session._lock.acquire()

            from src.card.timers.manager import _MAX_TTL_RETRIES
            for _ in range(_MAX_TTL_RETRIES + 1):
                session._ttl_handler.on_ttl_expired()

            session._lock.release()

        # Should still be marked closed despite delivery failure
        assert session._closed.is_set()


class TestForceCloseRealConcurrency:
    """True concurrent force-close tests with Barrier synchronization."""

    def test_concurrent_dispatch_and_force_close(self):
        """One thread dispatches continuously while another triggers force-close.

        Validates: session._closed ends up True with no exceptions raised.
        """
        client = _MockClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.5)
        session = CardSession(
            chat_id="c_conc_1", config=config, delivery=delivery,
            session_id="concurrent_fc_1",
        )
        session.dispatch(CardEvent.started())

        barrier = threading.Barrier(2, timeout=10)
        dispatch_started = threading.Event()
        errors: list[Exception] = []

        def dispatch_loop():
            try:
                barrier.wait()
                for i in range(50):
                    try:
                        session.dispatch(CardEvent.text_delta("x"))
                    except Exception:
                        pass  # Session may be closed mid-dispatch
                    if i == 0:
                        dispatch_started.set()
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        def force_close_trigger():
            try:
                barrier.wait()
                dispatch_started.wait(timeout=5)  # Wait for first dispatch
                # Hold the lock then exhaust retries to trigger force-close
                session._lock.acquire()
                try:
                    from src.card.timers.manager import _MAX_TTL_RETRIES
                    for _ in range(_MAX_TTL_RETRIES + 1):
                        session._ttl_handler.on_ttl_expired()
                finally:
                    session._lock.release()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=dispatch_loop)
        t2 = threading.Thread(target=force_close_trigger)
        t1.start(); t2.start()
        t1.join(timeout=30); t2.join(timeout=30)

        assert not errors, f"Concurrent errors: {errors}"
        assert session._closed.is_set()

    def test_close_and_force_close_hooks_idempotent(self):
        """close() and force-close triggered concurrently: fire_terminal called exactly once.

        Uses Barrier to synchronize both paths firing at the same time.
        """
        client = _MockClient()
        delivery = CardDelivery(client)
        terminal_calls: list[str] = []
        call_lock = threading.Lock()

        class CountingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                with call_lock:
                    terminal_calls.append(reason)

        metadata = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.5)
        session = CardSession(
            chat_id="c_idem_1", config=config, delivery=delivery,
            session_id="idem_fc_1",
            hooks=(CountingHook(),),
        )
        session.dispatch(CardEvent.started())

        barrier = threading.Barrier(2, timeout=10)
        errors: list[Exception] = []

        def normal_close():
            try:
                barrier.wait()
                session.close()
            except Exception as exc:
                errors.append(exc)

        def force_close():
            try:
                barrier.wait()
                # Directly set force-close flags (simulating exhausted retries)
                session._closed.set()
                session._timers.cancel_all()
                try:
                    delivery.close(session._session_id)
                except Exception:
                    pass
                session._hook_firer.fire_terminal(session._state, "ttl_expired")
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=normal_close)
        t2 = threading.Thread(target=force_close)
        t1.start(); t2.start()
        t1.join(timeout=30); t2.join(timeout=30)

        assert not errors, f"Concurrent errors: {errors}"
        assert session._closed.is_set()
        # fire_terminal is now exactly-once (HookFirer._fired guard)
        assert len(terminal_calls) == 1


class TestNotifyCallbackFallback:
    """Tests for notify_callback invocation during force-close."""

    def test_notify_callback_called_on_force_close(self):
        """When force-close triggers and lightweight delivery fails, notify_callback is invoked."""
        client = _MockClient()
        delivery = CardDelivery(client)
        notify_calls = []

        def notify_cb(chat_id, text):
            notify_calls.append((chat_id, text))

        metadata = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.1)
        callbacks = SessionCallbacks(notify_callback=notify_cb)
        session = CardSession(
            chat_id="chat_notify",
            config=config,
            delivery=delivery,
            session_id="notify_test",
            callbacks=callbacks,
        )
        session.dispatch(CardEvent.started())

        # Patch render_card to raise so lightweight delivery fails
        with patch("src.card.session.core.render_card", side_effect=RuntimeError("render failed")):
            # Hold lock and exhaust retries
            session._lock.acquire()
            from src.card.timers.manager import _MAX_TTL_RETRIES
            for _ in range(_MAX_TTL_RETRIES + 1):
                session._ttl_handler.on_ttl_expired()
            session._lock.release()

        assert session._closed.is_set()
        # notify_callback should have been called since lightweight delivery failed
        assert len(notify_calls) >= 1
        assert notify_calls[0][0] == "chat_notify"


class TestRenderCardFailureFallback:
    """Tests for render_card failure during force-close."""

    def test_force_close_survives_render_failure(self):
        """If render_card raises during force-close, session still closes cleanly."""
        client = _MockClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍")
        config = SessionConfig(metadata=metadata, ttl_seconds=0.1)
        session = CardSession(
            chat_id="chat_render_fail",
            config=config,
            delivery=delivery,
            session_id="render_fail_test",
        )
        session.dispatch(CardEvent.started())

        # Patch render_card to raise during the force-close terminal render
        with patch("src.card.session.core.render_card", side_effect=RuntimeError("render boom")):
            session._lock.acquire()
            from src.card.timers.manager import _MAX_TTL_RETRIES
            for _ in range(_MAX_TTL_RETRIES + 1):
                session._ttl_handler.on_ttl_expired()
            session._lock.release()

        # Session should still be closed despite render failure
        assert session._closed.is_set()


class TestTTLPrewarningToExpiredSequence:
    """Full TTL sequence: prewarning → expired → close → hooks.

    Uses injected clock and direct callback invocation for deterministic timing.
    """

    def _make_session(self, clock, *, ttl_seconds=100.0, hooks=(), notify_cb=None):
        client = _MockClient()
        delivery = CardDelivery(client)
        config = SessionConfig(
            metadata=CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔍"),
            ttl_seconds=ttl_seconds,
            clock=lambda: clock[0],
        )
        callbacks = SessionCallbacks(notify_callback=notify_cb) if notify_cb else None
        session = CardSession(
            chat_id="chat_ttl_seq",
            config=config,
            delivery=delivery,
            session_id="ttl_seq_test",
            hooks=hooks,
            callbacks=callbacks,
        )
        # Cancel real timers — we invoke callbacks directly
        session._timers.cancel_all()
        return session, client

    def test_prewarning_then_expired_full_sequence(self):
        """Full sequence: started → prewarning → expired → hooks fired once."""
        clock = [0.0]
        hook_calls = []
        notify_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                hook_calls.append(reason)

        def notify_cb(chat_id, text):
            notify_calls.append(text)

        session, client = self._make_session(
            clock, ttl_seconds=100.0,
            hooks=(TrackingHook(),), notify_cb=notify_cb,
        )

        # Step 1: Start session
        session.dispatch(CardEvent.started())
        assert not session.closed
        assert session.state.terminal == "running"

        # Step 2: Advance to 90% TTL → trigger prewarning
        clock[0] = 91.0
        session._ttl_handler.on_ttl_prewarning()
        assert not session.closed  # Still alive
        # Prewarning should set warning_banner on state
        assert session.state.footer.warning_banner is not None
        assert "分钟后关闭" in session.state.footer.warning_banner
        # Prewarning no longer sends separate chat notify (dual-notification removed);
        # only card banner is updated in the happy path
        prewarning_notifies = [t for t in notify_calls if "超时" in t]
        assert prewarning_notifies == []

        # Step 3: Advance to 100% TTL → trigger expired
        clock[0] = 101.0
        session._ttl_handler.on_ttl_expired()
        assert session.closed
        assert session._terminal_reason == "ttl_expired"

        # Step 4: Verify hooks fired exactly once
        assert len(hook_calls) == 1
        assert hook_calls[0] == "ttl_expired"

        # Step 5: Verify card state is terminal
        assert session.state.terminal == "cancelled"

    def test_prewarning_skipped_when_activity_resumes(self):
        """If user dispatches before prewarning fires, prewarning is a no-op."""
        clock = [0.0]
        session, _ = self._make_session(clock, ttl_seconds=100.0)
        session.dispatch(CardEvent.started())

        # Advance to 90% TTL...
        clock[0] = 91.0
        # ...but user dispatches just before (resets last_dispatch_time)
        session.dispatch(CardEvent.text_started("b1"))
        # Now _last_dispatch_time is ~91.0 so prewarning should see idle < 90%
        session._ttl_handler.on_ttl_prewarning()
        # No warning set because idle time is now near 0
        # (prewarning checks idle >= ttl * 0.9)
        assert not session.closed

    def test_expired_skipped_when_already_completed(self):
        """If session completed before TTL fires, expired callback is a no-op."""
        clock = [0.0]
        hook_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                hook_calls.append(reason)

        session, _ = self._make_session(
            clock, ttl_seconds=100.0, hooks=(TrackingHook(),),
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())
        assert session.closed

        # Advance clock and trigger expired — should be no-op
        clock[0] = 200.0
        session._ttl_handler.on_ttl_expired()

        # Hook was already called during completed(), not again for ttl_expired
        assert len(hook_calls) == 1
        assert hook_calls[0] == "completed"

    def test_double_expired_fires_hooks_once(self):
        """Calling on_ttl_expired twice fires hooks only once (exactly-once)."""
        clock = [0.0]
        hook_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                hook_calls.append(reason)

        session, _ = self._make_session(
            clock, ttl_seconds=100.0, hooks=(TrackingHook(),),
        )
        session.dispatch(CardEvent.started())

        clock[0] = 101.0
        session._ttl_handler.on_ttl_expired()
        session._ttl_handler.on_ttl_expired()  # Second call should be no-op

        assert session.closed
        assert len(hook_calls) == 1
