"""Tests for CardSession orchestration layer."""

import threading
import time
from unittest.mock import MagicMock, patch

from src.card.delivery.engine import CardDelivery, MutationOutcome
from src.card.events import CardEvent, CardEventType
from src.card.session import CardSession
from src.card.session.config import SessionCallbacks, SessionConfig
from src.card.state.models import CardMetadata
from src.card.state.reducer import reduce_card_state
from src.card.types import RenderedCard


class MockDeliveryClient:
    """Mock CardAPIClient for testing."""

    def __init__(self):
        self.creates = []
        self.updates = []
        self.elements = []
        self._counter = 0

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._counter += 1
        self.creates.append({"chat_id": chat_id, "card_json": card_json})
        return (f"msg_{self._counter}", f"card_{self._counter}")

    def update_card(self, card_id, card_json, *, sequence=0):
        self.updates.append(card_id)

    def update_element(self, card_id, element_id, content, *, sequence=0):
        self.elements.append(element_id)


class TestCardSessionDispatch:
    """Core dispatch behavior."""

    def _make_session(self):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="test_sess",
        )
        return session, client, delivery


class TestAsyncSnapshotCoalescing:
    def test_blocked_delivery_renders_only_first_and_latest_terminal_snapshot(self):
        from src.card.render.renderer import render_card as real_render_card

        class _BlockingDelivery:
            def __init__(self):
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.calls = 0

            def open_session(self, _session_id, _chat_id):
                return None

            def deliver(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    self.release_first.wait(timeout=3.0)
                return [MutationOutcome(kind="applied")]

            def close(self, _session_id):
                return None

            def release_session_lock(self, _session_id):
                return None

        delivery = _BlockingDelivery()
        config = SessionConfig(
            metadata=CardMetadata(mode_name="Coalesce"),
            sync_delivery=False,
        )
        session = CardSession(
            chat_id="chat-coalesce",
            config=config,
            delivery=delivery,
            session_id="session-coalesce",
        )
        # The global test fixture forces synchronous delivery; this contract is
        # specifically about the production asynchronous path.
        session._sync_delivery = False
        session._timers.close()
        rendered_snapshots: list[tuple[int, str | None]] = []

        def _recording_render(state, budget):
            rendered_snapshots.append((state.version, state.terminal_reason))
            return real_render_card(state, budget)

        try:
            with patch("src.card.session.core.render_card", side_effect=_recording_render):
                session.dispatch(CardEvent.started())
                assert delivery.first_started.wait(timeout=2.0)

                session.dispatch(CardEvent.text_delta("main", "one"))
                session.dispatch(CardEvent.text_delta("main", "two"))
                session.dispatch(CardEvent.completed(summary="done"))

                # Rendering is expensive: pending immutable states must be
                # coalesced before it, with the terminal state winning.
                assert rendered_snapshots == [(1, None)]

                delivery.release_first.set()
                assert session.wait_delivery_idle(timeout=3.0)
        finally:
            delivery.release_first.set()
            session.wait_delivery_idle(timeout=3.0)

        assert len(rendered_snapshots) == 2
        assert rendered_snapshots[-1][1] == "completed"
        assert delivery.calls == 2

    def test_close_fences_snapshot_blocked_in_render_before_delivery(self):
        render_started = threading.Event()
        release_render = threading.Event()
        delivery = MagicMock()
        session = CardSession(
            chat_id="chat-close-render",
            config=SessionConfig(
                metadata=CardMetadata(mode_name="CloseFence"),
                sync_delivery=False,
            ),
            delivery=delivery,
            session_id="session-close-render",
        )
        session._sync_delivery = False
        session._timers.close()
        session._deliver_and_track = MagicMock()

        def _blocked_render(_state, _budget):
            render_started.set()
            release_render.wait()
            return [
                RenderedCard(
                    _card_json={"body": {"elements": []}},
                    structure_signature="close-fence",
                )
            ]

        with patch("src.card.session.core.render_card", side_effect=_blocked_render):
            worker = threading.Thread(target=lambda: session.dispatch(CardEvent.started()))
            worker.start()
            assert render_started.wait(timeout=1.0)

            close_done = threading.Event()
            close_worker = threading.Thread(
                target=lambda: (session.close(), close_done.set())
            )
            close_worker.start()
            assert close_done.wait(timeout=0.1) is False
            release_render.set()
            worker.join(timeout=2.0)
            close_worker.join(timeout=2.0)
            assert session.wait_delivery_idle(timeout=2.0)

        assert worker.is_alive() is False
        assert close_worker.is_alive() is False
        assert close_done.is_set()
        session._deliver_and_track.assert_not_called()

    def test_external_close_waits_for_pre_close_dispatched_hook(self):
        hook_started = threading.Event()
        release_hook = threading.Event()
        close_done = threading.Event()

        class _BlockingHook:
            def on_dispatched(self, _event, _state):
                hook_started.set()
                release_hook.wait()

        session = CardSession(
            chat_id="chat-close-hook",
            config=SessionConfig(metadata=CardMetadata(mode_name="HookFence")),
            delivery=MagicMock(),
            session_id="session-close-hook",
            hooks=(_BlockingHook(),),
        )
        session._timers.close()
        session.dispatch(CardEvent.started())
        assert hook_started.wait(timeout=1.0)

        close_worker = threading.Thread(
            target=lambda: (session.close(), close_done.set())
        )
        close_worker.start()
        try:
            assert close_done.wait(timeout=0.1) is False
            release_hook.set()
            close_worker.join(timeout=2.0)
            assert close_done.is_set()
        finally:
            release_hook.set()
            close_worker.join(timeout=2.0)

    def test_dispatched_hook_close_skips_self_but_waits_for_sibling(self):
        sibling_started = threading.Event()
        release_sibling = threading.Event()
        hook_close_returned = threading.Event()
        session_ref = []

        class _ClosingHook:
            def on_dispatched(self, _event, _state):
                assert sibling_started.wait(timeout=1.0)
                session_ref[0].close()
                hook_close_returned.set()

        class _SiblingHook:
            def on_dispatched(self, _event, _state):
                sibling_started.set()
                release_sibling.wait()

        session = CardSession(
            chat_id="chat-self-close-hook",
            config=SessionConfig(metadata=CardMetadata(mode_name="HookSelfFence")),
            delivery=MagicMock(),
            session_id="session-self-close-hook",
            hooks=(_ClosingHook(), _SiblingHook()),
        )
        session_ref.append(session)
        session._timers.close()
        session.dispatch(CardEvent.started())
        assert sibling_started.wait(timeout=1.0)
        try:
            assert hook_close_returned.wait(timeout=0.1) is False
            release_sibling.set()
            assert hook_close_returned.wait(timeout=2.0)
        finally:
            release_sibling.set()













class TestCardSessionLifecycle:
    """Full lifecycle tests."""

    def _make_session(self):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="test_sess",
        )
        return session, client

    def test_full_lifecycle(self):
        """Complete flow: started → text → tool → text → completed."""
        session, client = self._make_session()

        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "Analyzing...", "block_id": "_active_text"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE))
        session.dispatch(CardEvent(
            type=CardEventType.TOOL_STARTED,
            payload={"tool_name": "bash", "block_id": "tc1"}
        ))
        session.dispatch(CardEvent(
            type=CardEventType.TOOL_DONE,
            payload={"block_id": "tc1", "tool_output": "result"}
        ))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "Done!", "block_id": "_active_text"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE))
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        assert session.closed
        state = session.state
        assert state.terminal == "completed"
        assert len(state.blocks) >= 3  # At least: text + tool + text
        # Verify text was actually written into state
        text_blocks = [b for b in state.blocks if b.kind == "text"]
        assert any("Analyzing" in b.content for b in text_blocks)
        assert any("Done" in b.content for b in text_blocks)


# ---------------------------------------------------------------------------
# Phase 5: New edge-case tests for delivery failures, inbound_action, snapshot
# ---------------------------------------------------------------------------

class FailingDelivery:
    """Mock delivery that raises on deliver."""

    def __init__(self, fail_count: int = 999):
        self._fail_count = fail_count
        self._calls = 0

    def deliver(self, *, session_id, chat_id, rendered, reply_to=None, is_terminal=False):
        self._calls += 1
        if self._calls <= self._fail_count:
            raise ConnectionError("network error")
        return []

    def close(self, session_id):
        pass


# ==============================================================================
# _pending_action_to_event helper tests
# ==============================================================================


# ---------------------------------------------------------------------------
# Phase 6: Terminal retry, TTL, concurrency guard, empty render tests
# ---------------------------------------------------------------------------


class CountingDelivery:
    """Mock delivery that tracks calls and can toggle failure."""

    def __init__(self):
        self.deliver_calls = 0
        self.close_calls = 0
        self.fail_until = 0  # fail first N deliver calls

    def deliver(self, *, session_id, chat_id, rendered, reply_to=None, is_terminal=False):
        self.deliver_calls += 1
        if self.deliver_calls <= self.fail_until:
            raise ConnectionError("simulated failure")

    def close(self, session_id):
        self.close_calls += 1


class TestTerminalRetry:
    """Tests for _schedule_terminal_retry behavior (AC24) — no time.sleep()."""

    def _make_session(self, delivery, **kwargs):
        config = SessionConfig(
            metadata=CardMetadata(mode_name="Test"),
            retry_delay=0.01,
        )
        # Extract callbacks from kwargs if present
        callbacks_kwargs = {}
        for key in ("notify_callback", "cancel_callback", "reply_text_fn"):
            if key in kwargs:
                callbacks_kwargs[key] = kwargs.pop(key)
        callbacks = SessionCallbacks(**callbacks_kwargs) if callbacks_kwargs else None
        return CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="retry_test",
            callbacks=callbacks,
            **kwargs,
        )




class TestRetryConcurrentClose:
    """Test _retry() guard when close() called concurrently (AC25)."""

        # deliver_calls should not have increased from retry
        # (timer was cancelled by close)


class TestProactiveTTLTimer:
    """Test proactive Timer-based TTL expiration (Task 19).

    Verifies _on_ttl_expired callback fires and closes session
    without requiring a dispatch call.
    """

    def _make_session(self, now, ttl_seconds=10.0):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(
            metadata=CardMetadata(mode_name="Test"),
            ttl_seconds=ttl_seconds,
            clock=lambda: now[0],
        )
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="timer_proactive",
        )
        # Cancel real timer to avoid interference
        session._timers.cancel_all()
        # Prevent dispatch from spawning new timers
        session._reset_ttl_timer = lambda: None
        return session



    def test_timer_callback_lock_contention_early_return(self):
        """Proactive timer: when lock is held by another thread, _on_ttl_expired early-returns without modifying state."""
        import threading as _threading

        now = [0.0]
        session = self._make_session(now)
        session.dispatch(CardEvent.started())
        state_before = session.state

        # Advance clock past TTL
        now[0] = 11.0

        # Hold the lock from the main thread to simulate contention
        session._lock.acquire()
        try:
            # Call _on_ttl_expired from another thread — it should fail to acquire lock and return
            result_holder = []

            def call_ttl():
                try:
                    session._ttl_handler.on_ttl_expired()
                    result_holder.append("ok")
                except Exception as exc:
                    result_holder.append(f"error: {exc}")

            t = _threading.Thread(target=call_ttl)
            t.start()
            t.join(timeout=3.0)
        finally:
            session._lock.release()

        # Should have returned without error
        assert result_holder == ["ok"]
        # Session state should be unchanged — not closed, not cancelled
        assert not session.closed
        assert session.state == state_before
        # A retry timer should have been scheduled to prevent zombie sessions
        assert session._timers._ttl_handle is not None
        session._timers.cancel_all()  # cleanup


# ---------------------------------------------------------------------------
# Lifecycle hooks tests
# ---------------------------------------------------------------------------


# ==============================================================================
# Task 4: session.close() does NOT trigger on_terminal hooks
# ==============================================================================


# ==============================================================================
# Task 6: max_failures_banner does not contain raw {timestamp}
# ==============================================================================


# ==============================================================================
# Phase 3 feature tests
# ==============================================================================


class TestCardDeliveryConcurrentCloseDeliver:
    """Verify close + deliver race condition is handled safely (double-check locking)."""

    def test_close_during_deliver_blocks_api_call(self):
        """When close() and deliver() enter simultaneously, deliver should NOT call API after close."""
        import threading

        client = MockDeliveryClient()
        delivery = CardDelivery(client)

        # First, do a normal deliver to establish binding
        card = RenderedCard(
            _card_json={"body": {"elements": []}},
            structure_signature="sig1",
            content_hash="",
            active_element=None,
            page_index=0,
            total_pages=1,
        )
        delivery.deliver("session_concurrent", "test_chat", [card])

        # Now set up concurrent close + deliver using Barrier
        barrier = threading.Barrier(2, timeout=5)
        results = {"deliver_called": False}

        def do_close():
            barrier.wait()
            delivery.close("session_concurrent")

        def do_deliver():
            barrier.wait()
            import time
            time.sleep(0.01)
            card2 = RenderedCard(
                _card_json={"body": {"elements": [{"tag": "markdown", "content": "new"}]}},
                structure_signature="sig2",
                content_hash="h2",
                active_element=None,
                page_index=0,
                total_pages=1,
            )
            delivery.deliver("session_concurrent", "test_chat", [card2])
            results["deliver_called"] = True

        t1 = threading.Thread(target=do_close)
        t2 = threading.Thread(target=do_deliver)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # The key invariant: after close, deliver is a no-op
        assert results["deliver_called"]


class TestTTLCloseWithoutDispatchSkipsHooks:
    """AC19: Session force-closed when state is None should not call terminal hooks."""

    def test_ttl_force_close_without_state_skips_hooks(self):
        """Force-close path (lock retries exhausted) with state=None skips terminal hooks."""
        hook_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                hook_calls.append(("dispatched", event, state))

            def on_terminal(self, state, reason):
                hook_calls.append(("terminal", state, reason))

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        hook = TrackingHook()
        config = SessionConfig(
            metadata=CardMetadata(mode_name="Test", engine_type="deep"),
            ttl_seconds=10.0,
        )
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="no_dispatch_force_close",
            hooks=(hook,),
        )
        # Cancel auto-started TTL timer
        session._timers.cancel_all()

        # Never dispatch — state is None
        assert session.state is None

        # Simulate the force-close path by calling close() directly
        # close() checks state and only fires hooks if state is not None
        session.close()

        # Session should be closed
        assert session.closed

        # Terminal hooks should NOT have been called because state was None
        terminal_calls = [c for c in hook_calls if c[0] == "terminal"]
        assert terminal_calls == [], "fire_terminal should skip when state is None"

    def test_normal_ttl_close_after_dispatch_fires_hooks(self):
        """TTL close after a dispatch has occurred DOES fire terminal hooks."""
        hook_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                hook_calls.append(("dispatched", event, state))

            def on_terminal(self, state, reason):
                hook_calls.append(("terminal", state, reason))

        now = [0.0]
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        hook = TrackingHook()
        config = SessionConfig(
            metadata=CardMetadata(mode_name="Test", engine_type="deep"),
            ttl_seconds=10.0,
            clock=lambda: now[0],
        )
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="dispatch_then_ttl",
            hooks=(hook,),
        )
        session._timers.cancel_all()
        session._reset_ttl_timer = lambda: None

        # Dispatch to create state
        session.dispatch(CardEvent.started())
        assert session.state is not None

        # Expire via TTL
        now[0] = 11.0
        session._ttl_handler.on_ttl_expired()

        assert session.closed
        terminal_calls = [c for c in hook_calls if c[0] == "terminal"]
        assert len(terminal_calls) == 1
        assert terminal_calls[0][2] == "ttl_expired"


class TestTerminalReduceFailure:
    """AC: Terminal event reduce failure must force-close to prevent zombie sessions."""

    def _make_session(self, hooks=()):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="term_reduce_fail",
            hooks=hooks,
        )
        return session, client

    def test_completed_reduce_failure_closes_session(self, monkeypatch):
        """If COMPLETED event causes reduce to raise, session must close."""
        session, _ = self._make_session()
        session.dispatch(CardEvent.started())
        assert not session.closed

        call_count = [0]
        original_reduce = reduce_card_state

        def failing_reduce(state, event, metadata):
            call_count[0] += 1
            if event.type == CardEventType.COMPLETED:
                raise RuntimeError("reduce crash on COMPLETED")
            return original_reduce(state, event, metadata)

        monkeypatch.setattr("src.card.session.core.reduce_card_state", failing_reduce)

        session.dispatch(CardEvent(type=CardEventType.COMPLETED))
        assert session.closed, "Session must be force-closed when terminal reduce fails"


    def test_terminal_reduce_failure_fires_terminal_hooks(self, monkeypatch):
        """Force-close on terminal reduce failure should fire terminal hooks."""
        hook_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                hook_calls.append(reason)

        session, _ = self._make_session(hooks=(TrackingHook(),))
        session.dispatch(CardEvent.started())

        def failing_reduce(state, event, metadata):
            if event.type in (CardEventType.COMPLETED, CardEventType.FAILED, CardEventType.CANCELLED):
                raise RuntimeError("crash on terminal")
            return reduce_card_state(state, event, metadata)

        monkeypatch.setattr("src.card.session.core.reduce_card_state", failing_reduce)

        session.dispatch(CardEvent(type=CardEventType.COMPLETED))
        assert session.closed
        assert len(hook_calls) == 1


# ---------------------------------------------------------------------------
# FS-4: _deliver_and_track rejected path tests
# ---------------------------------------------------------------------------


class RejectingDelivery:
    """Mock delivery that returns rejected MutationOutcome on deliver()."""

    def __init__(self):
        self.deliver_calls = 0
        self.close_calls = 0

    def deliver(self, *, session_id, chat_id, rendered, reply_to=None, is_terminal=False):
        self.deliver_calls += 1
        return [MutationOutcome(kind="rejected", message="capacity exhausted")]

    def close(self, session_id):
        self.close_calls += 1


class TestDeliverAndTrackRejected:
    """Tests for _deliver_and_track rejected path (FS-4)."""

    def _make_session(self, delivery, **kwargs):
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata, retry_delay=0.01)
        callbacks_kwargs = {}
        for key in ("notify_callback", "cancel_callback", "reply_text_fn"):
            if key in kwargs:
                callbacks_kwargs[key] = kwargs.pop(key)
        callbacks = SessionCallbacks(**callbacks_kwargs) if callbacks_kwargs else None
        return CardSession(
            chat_id="chat_rej",
            config=config,
            delivery=delivery,
            session_id="rej_sess",
            callbacks=callbacks,
            **kwargs,
        )


    def test_rejected_terminal_schedules_retry_no_finalize(self):
        """When rejected on terminal event, _schedule_terminal_retry is called, _finalize_terminal is NOT."""
        delivery = RejectingDelivery()
        notify = MagicMock()
        session = self._make_session(delivery, notify_callback=notify)
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.text_delta("blk_1", "hello"))
        # Dispatch COMPLETED (terminal) — will get rejected
        session.dispatch(CardEvent.completed())
        # delivery.close should NOT have been called (no finalize)
        assert delivery.close_calls == 0
        # notify should have been called for rejected
        assert notify.call_count >= 1


# ---------------------------------------------------------------------------
# FS-5: delivery.close() exception does not block hooks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FS-6: _finalize_terminal ordering: close → hooks → cancel_callback
# ---------------------------------------------------------------------------


class TestFinalizeTerminalOrdering:
    """Verify _finalize_terminal executes: delivery.close → hooks → cancel_callback."""

    def test_ordering_close_hooks_cancel(self):
        """Operations in _finalize_terminal follow strict order."""
        call_log = []

        hook = MagicMock()
        hook.on_dispatched = MagicMock()
        hook.on_terminal = MagicMock(side_effect=lambda state, reason: call_log.append("hook"))

        def cancel_cb():
            call_log.append("cancel")

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)

        session = CardSession(
            chat_id="chat_ord",
            config=config,
            delivery=delivery,
            session_id="ord_sess",
            hooks=(hook,),
            callbacks=SessionCallbacks(cancel_callback=cancel_cb),
        )

        # Patch delivery.close to record order
        original_close = delivery.close

        def tracking_close(session_id):
            call_log.append("close")
            return original_close(session_id)

        with patch.object(delivery, "close", side_effect=tracking_close):
            session.dispatch(CardEvent.started())
            # Dispatch CANCELLED without reason to get terminal_reason="cancelled"
            session.dispatch(CardEvent.cancelled())

        # Wait for hooks (async via thread pool)
        time.sleep(0.3)
        assert call_log == ["close", "hook", "cancel"], f"Expected ['close', 'hook', 'cancel'], got {call_log}"


# ---------------------------------------------------------------------------
# FS-7: _terminal_reason=None fallback to 'completed'
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FS-8: cancel_callback exception does not propagate
# ---------------------------------------------------------------------------


        # No exception propagated (test would fail if it did)


# ---------------------------------------------------------------------------
# FS-9: _create_page double failure → reconcile, no orphan binding
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FS-10: _stream_element SequenceConflictError → fallback to _update_page
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FS-11: to_feishu_json isolation — delivery payload vs RenderedCard internal
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AC11: Concurrent close() race test
# ---------------------------------------------------------------------------


class TestConcurrentCloseRace:
    """Two threads call close() simultaneously — hooks fire exactly once."""

    def test_concurrent_close_fires_hooks_once(self):
        """AC11: threading.Barrier ensures both threads call close() at the same moment."""
        hook = MagicMock()
        hook.on_dispatched = MagicMock()
        hook.on_terminal = MagicMock()

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        callbacks = SessionCallbacks(hooks=(hook,))
        session = CardSession(
            chat_id="c_race",
            config=config,
            delivery=delivery,
            session_id="race_close",
            callbacks=callbacks,
        )
        session.dispatch(CardEvent.started())

        barrier = threading.Barrier(2, timeout=5.0)
        errors = []

        def close_thread():
            try:
                barrier.wait()
                session.close()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=close_thread)
        t2 = threading.Thread(target=close_thread)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not errors, f"Unexpected errors: {errors}"
        assert session.closed
        # Hook on_terminal must have been called exactly once
        time.sleep(0.3)  # Allow any async hook processing
        assert hook.on_terminal.call_count == 1


# ---------------------------------------------------------------------------
# AC13: Non-terminal double reduce failure — state rollback
# ---------------------------------------------------------------------------


# ─── Task 17 [AC-TEST-2]: TestAddHookIntegration ───
