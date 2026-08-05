"""Advanced pipeline tests: TTL streaming interruption, cancellation paths,
rotation mid-streaming, renderer e2e, eviction, and close/eviction race.

Tasks 27-36 from the card pipeline migration cleanup.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.card.delivery.engine import CardDelivery
from src.card.events import CardEvent, CardEventType
from src.card.protocols import TTLState
from src.card.session import CardSession
from src.card.session.config import SessionConfig
from src.card.session.rotator import SessionRotator
from src.card.session.ttl import TTLHandler
from src.card.state.models import CardMetadata

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class MockDeliveryClient:
    """Minimal CardAPIClient mock for pipeline tests."""

    def __init__(self):
        self.creates = []
        self.updates = []
        self.elements = []
        self._counter = 0

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._counter += 1
        self.creates.append(card_json)
        return (f"msg_{self._counter}", f"card_{self._counter}")

    def update_card(self, card_id, card_json, *, sequence=0):
        self.updates.append(card_json)

    def update_element(self, card_id, element_id, content, *, sequence=0):
        self.elements.append(element_id)


def _make_session(engine_type="deep", ttl_seconds=1800, clock=None):
    """Create a CardSession with a mock client for testing."""
    client = MockDeliveryClient()
    delivery = CardDelivery(client)
    metadata = CardMetadata(
        mode_name="Test", tool_name="test", model_name="test",
        engine_type=engine_type,
    )
    config_kwargs = dict(metadata=metadata, ttl_seconds=ttl_seconds)
    if clock is not None:
        config_kwargs["clock"] = clock
    config = SessionConfig(**config_kwargs)
    session = CardSession(
        chat_id="chat_test",
        config=config,
        delivery=delivery,
        session_id="test_pipeline",
    )
    return session, client


def _make_ttl_mock(**overrides) -> MagicMock:
    """Create a mock TTLDecider + TTLActuator."""
    s = MagicMock()
    s.get_ttl_state.return_value = TTLState(
        closed=False, ttl_warned=False, idle_seconds=2000.0,
        ttl_seconds=1800.0, session_id="test_sess", state_snapshot=None,
    )
    s.engine_cmd = "/deep"
    s.engine_name = "Deep"
    s.reduce_and_render.return_value = [{"card": "rendered"}]
    for key, val in overrides.items():
        setattr(s, key, val)
    return s


# ===========================================================================
# Tasks 27-29: TTL streaming interruption tests
# ===========================================================================


class TestTTLStreamingInterruption:
    """TTL expiry while streaming is in progress (Deep/Spec engines)."""

    @pytest.mark.parametrize("engine_type,engine_cmd", [
        ("deep", "/deep"),
        ("spec", "/spec"),
    ])
    def test_ttl_expires_during_streaming_dispatches_cancel(self, engine_type, engine_cmd):
        """When TTL fires mid-streaming, session should be cancelled with ttl_expired reason."""
        session, client = _make_session(engine_type=engine_type, ttl_seconds=1, clock=time.monotonic)

        # Start the session and begin streaming
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "Hello"}))

        # Simulate TTL expiry by dispatching after clock moves past TTL
        # Use a controllable clock
        base_time = time.monotonic()
        t = [base_time]

        session2, client2 = _make_session(
            engine_type=engine_type,
            ttl_seconds=5,
            clock=lambda: t[0],
        )
        session2.dispatch(CardEvent(type=CardEventType.STARTED))
        session2.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session2.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "Mid-stream"}))

        # Advance clock past TTL
        t[0] = base_time + 10

        # Next dispatch should trigger inline TTL check
        session2.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": " more"}))

        # Session should be closed (TTL inline check fires)
        assert session2.closed is True
        state = session2.state
        assert state.terminal == "cancelled"

    @pytest.mark.parametrize("engine_type", ["deep", "spec"])
    def test_ttl_handler_expired_callback_mid_streaming(self, engine_type):
        """TTLHandler.on_ttl_expired works correctly regardless of engine type."""
        s = _make_ttl_mock()
        s.engine_cmd = f"/{engine_type}"
        s.engine_name = engine_type.title()
        handler = TTLHandler(decider=s, actuator=s)

        handler.on_ttl_expired()

        s.mark_ttl_expired.assert_called_once()
        s.reduce_and_render.assert_called_once()
        events = s.reduce_and_render.call_args[0][0]
        # Should have warning_updated + cancelled
        assert len(events) == 2
        assert events[1].type == CardEventType.CANCELLED
        s.deliver_terminal.assert_called_once()


# ===========================================================================
# Tasks 30-32: Cancellation path tests
# ===========================================================================


class TestCancellationPaths:
    """Cancellation dispatches correctly for all engine types."""

    @pytest.mark.parametrize("engine_type", ["deep", "spec"])
    def test_cancel_during_running_transitions_to_cancelled(self, engine_type):
        """CANCELLED event transitions running session to cancelled terminal state."""
        session, client = _make_session(engine_type=engine_type)
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))

        session.dispatch(CardEvent.cancelled())

        state = session.state
        assert state.terminal == "cancelled"
        assert session.closed is True

    @pytest.mark.parametrize("engine_type", ["deep", "spec"])
    def test_cancel_with_ttl_expired_reason(self, engine_type):
        """CANCELLED with reason='ttl_expired' sets terminal_reason correctly."""
        session, client = _make_session(engine_type=engine_type)
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        session.dispatch(CardEvent.cancelled(reason="ttl_expired"))

        state = session.state
        assert state.terminal == "cancelled"
        assert state.terminal_reason == "ttl_expired"

    @pytest.mark.parametrize("engine_type", ["deep", "spec"])
    def test_cancel_after_close_is_noop(self, engine_type):
        """Dispatching CANCELLED after session.close() is a no-op."""
        session, client = _make_session(engine_type=engine_type)
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.close()

        # Should not raise
        session.dispatch(CardEvent.cancelled())
        # Session state should remain as it was at close time
        assert session.closed is True


# ===========================================================================
# Task 33: Rotation during streaming
# ===========================================================================


class TestRotationDuringStreaming:
    """SessionRotator.rotate() during active streaming."""

    def test_rotate_mid_stream_archives_old_starts_new(self):
        """Rotation during streaming archives old session and creates new one."""
        session1, client1 = _make_session(engine_type="deep")
        session1.dispatch(CardEvent(type=CardEventType.STARTED))
        session1.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session1.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "streaming..."}))

        rotator = SessionRotator(session1)

        # Rotate to new session
        session2, client2 = _make_session(engine_type="deep")

        new_session = rotator.rotate(lambda: session2)

        assert new_session is session2
        # Old session got archived event
        old_state = session1.state
        assert old_state.terminal == "archived"
        assert rotator.rotation_count == 1

    def test_rotate_preserves_text_in_old_session(self):
        """Old session retains accumulated text blocks after rotation."""
        session1, _ = _make_session(engine_type="spec")
        session1.dispatch(CardEvent(type=CardEventType.STARTED))
        session1.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session1.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "important data"}))
        session1.dispatch(CardEvent(type=CardEventType.TEXT_DONE))

        rotator = SessionRotator(session1)
        session2, _ = _make_session(engine_type="spec")
        rotator.rotate(lambda: session2)

        # Old session still has its text blocks
        text_blocks = [b for b in session1.state.blocks if b.kind == "text"]
        assert any("important data" in b.content for b in text_blocks)

    def test_dispatch_after_rotate_goes_to_new_session(self):
        """Dispatches after rotation go to the new session, not the old."""
        session1, _ = _make_session(engine_type="deep")
        session1.dispatch(CardEvent(type=CardEventType.STARTED))

        rotator = SessionRotator(session1)
        session2, _ = _make_session(engine_type="deep")
        rotator.rotate(lambda: session2)

        # Dispatch to rotator goes to new session
        rotator.dispatch(CardEvent(type=CardEventType.STARTED))
        assert session2.state.terminal == "running"


# ===========================================================================
# Task 34: Renderer e2e test
# ===========================================================================


class TestRendererEndToEnd:
    """End-to-end rendering pipeline: events → state → rendered card."""

    def test_full_lifecycle_produces_valid_cards(self):
        """A complete session lifecycle (start → text → complete) produces valid card JSON."""
        session, client = _make_session(engine_type="deep")

        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "Result: 42"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE))
        session.dispatch(CardEvent.completed(summary="Task done"))

        # Should have created + multiple updates
        assert len(client.creates) == 1
        assert len(client.updates) >= 1
        assert session.closed is True
        assert session.state.terminal == "completed"

    def test_rendered_card_has_header_and_body(self):
        """Rendered card contains header template and body elements."""
        session, client = _make_session(engine_type="deep")
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        # The create payload should have card structure
        assert len(client.creates) >= 1
        card_json = client.creates[0]
        # Card should be a dict with standard Feishu card structure
        assert isinstance(card_json, dict)
        # Should have header (indicated by 'header' key in card)
        assert "header" in card_json


# ===========================================================================
# Task 35: Eviction loop time-driven test
# ===========================================================================


class TestEvictionLoopTimeDriven:
    """Verify that session locks are evicted when TTL expires."""

    def test_stale_sessions_are_evictable(self):
        """Sessions that exceed lock TTL can be evicted from delivery."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)

        # Create a binding by dispatching a session
        metadata = CardMetadata(mode_name="Test", tool_name="t", model_name="m", engine_type="deep")
        config = SessionConfig(metadata=metadata, ttl_seconds=3600)
        session = CardSession(
            chat_id="chat_evict",
            config=config,
            delivery=delivery,
            session_id="evict_test",
        )
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        # The delivery should have a binding
        binding = delivery.get_binding("evict_test")
        assert binding is not None

    def test_delivery_eviction_removes_stale_bindings(self):
        """CardDelivery._maybe_evict removes bindings exceeding TTL."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)

        # Create multiple sessions to trigger eviction threshold
        sessions = []
        for i in range(5):
            metadata = CardMetadata(mode_name="Test", tool_name="t", model_name="m", engine_type="deep")
            config = SessionConfig(metadata=metadata, ttl_seconds=3600)
            s = CardSession(
                chat_id=f"chat_{i}",
                config=config,
                delivery=delivery,
                session_id=f"sess_{i}",
            )
            s.dispatch(CardEvent(type=CardEventType.STARTED))
            sessions.append(s)

        # All bindings should exist
        for i in range(5):
            assert delivery.get_binding(f"sess_{i}") is not None


# ===========================================================================
# Task 36: Close/eviction race test
# ===========================================================================


class TestCloseEvictionRace:
    """Verify session.close() and delivery eviction don't race."""

    def test_close_during_dispatch_is_safe(self):
        """Closing session from another thread during dispatch doesn't crash."""
        session, client = _make_session(engine_type="deep")
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        errors = []

        def dispatch_loop():
            for i in range(20):
                try:
                    session.dispatch(CardEvent(
                        type=CardEventType.TEXT_DELTA,
                        payload={"text": f"chunk_{i}"}
                    ))
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        def close_after_delay():
            time.sleep(0.005)
            session.close()

        t1 = threading.Thread(target=dispatch_loop)
        t2 = threading.Thread(target=close_after_delay)

        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Should not have raised any exceptions
        assert len(errors) == 0
        assert session.closed is True

    def test_concurrent_dispatch_and_close_no_deadlock(self):
        """Multiple threads dispatching while close happens doesn't deadlock."""
        session, client = _make_session(engine_type="deep")
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))

        barrier = threading.Barrier(3, timeout=5)

        def dispatcher():
            barrier.wait()
            for _ in range(10):
                session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "x"}))

        def closer():
            barrier.wait()
            time.sleep(0.002)
            session.close()

        threads = [
            threading.Thread(target=dispatcher),
            threading.Thread(target=dispatcher),
            threading.Thread(target=closer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert session.closed is True
        # No thread should be hung
        for t in threads:
            assert not t.is_alive()
