"""End-to-end integration tests: simulates Handler → CardSession → API flow."""

import threading

from src.card.delivery.engine import CardDelivery
from src.card.events import CardEvent, CardEventType
from src.card.render.budget import RenderBudget
from src.card.session import CardSession
from src.card.session.config import SessionCallbacks, SessionConfig
from src.card.state.models import CardMetadata, CardState


class RecordingClient:
    """Records all API calls for verification."""

    def __init__(self):
        self.calls: list[dict] = []
        self._counter = 0

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._counter += 1
        self.calls.append({"op": "create", "chat_id": chat_id, "json": card_json})
        return (f"msg_{self._counter}", f"card_{self._counter}")

    def update_card(self, card_id, card_json, *, sequence=0):
        self.calls.append({"op": "update", "card_id": card_id, "seq": sequence})

    def update_element(self, card_id, element_id, content, *, sequence=0):
        self.calls.append({"op": "element", "card_id": card_id, "element_id": element_id, "content": content})


def _make_session(client: RecordingClient, **meta_kw) -> CardSession:
    delivery = CardDelivery(client)
    metadata = CardMetadata(
        mode_name=meta_kw.get("mode_name", "Coco"),
        tool_name=meta_kw.get("tool_name", "coco"),
        model_name=meta_kw.get("model_name", "gpt-4o"),
    )
    config = SessionConfig(metadata=metadata)
    session = CardSession(
        chat_id="chat_test",
        config=config,
        delivery=delivery,
        session_id="e2e_sess",
    )
    return session


class TestDeepEngineFlow:
    """Simulates a typical Deep Engine flow."""

    def test_started_text_tool_text_completed(self):
        client = RecordingClient()
        session = _make_session(client)

        # Started
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        assert any(c["op"] == "create" for c in client.calls)

        # Text streaming
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED, payload={"block_id": "t1"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "t1", "text": "Analyzing requirements..."}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE, payload={"block_id": "t1"}))

        # Tool call
        session.dispatch(CardEvent(type=CardEventType.TOOL_STARTED, payload={"tool_name": "bash", "block_id": "tc1"}))
        session.dispatch(CardEvent(type=CardEventType.TOOL_DONE, payload={"block_id": "tc1", "tool_output": "OK"}))

        # Final text
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED, payload={"block_id": "t2"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "t2", "text": "Done!"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE, payload={"block_id": "t2"}))

        # Completed
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        assert session.closed
        state = session.state
        assert state.terminal == "completed"
        assert len(state.blocks) >= 3

        # Verify card was created then updated multiple times
        creates = [c for c in client.calls if c["op"] == "create"]
        updates = [c for c in client.calls if c["op"] == "update"]
        assert len(creates) >= 1
        assert len(updates) >= 1  # At least one structural update (tool change)


class TestMultiPageFlow:
    """Large content triggers pagination."""

    def test_multi_page_on_large_content(self):
        client = RecordingClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco")
        # Very small budget to force pagination
        budget = RenderBudget(byte_budget=2000)
        config = SessionConfig(metadata=metadata, budget=budget)
        session = CardSession(
            chat_id="chat_test",
            config=config,
            delivery=delivery,
            session_id="page_test",
        )

        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED, payload={"block_id": "big"}))
        # Send a large text delta
        big_text = "x" * 10000
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "big", "text": big_text}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE, payload={"block_id": "big"}))
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        # Should have created multiple cards (one per page)
        creates = [c for c in client.calls if c["op"] == "create"]
        assert len(creates) >= 2


class TestToolBlockTracking:
    """Multiple completed tool blocks are tracked in card state."""

    def test_completed_tools_tracked_in_state(self):
        client = RecordingClient()
        session = _make_session(client)

        session.dispatch(CardEvent(type=CardEventType.STARTED))

        # 4 completed tools (above fold threshold of 3)
        for i in range(4):
            session.dispatch(CardEvent(
                type=CardEventType.TOOL_STARTED,
                payload={"tool_name": f"tool_{i}", "block_id": f"tc{i}"}
            ))
            session.dispatch(CardEvent(
                type=CardEventType.TOOL_DONE,
                payload={"block_id": f"tc{i}", "tool_output": f"result_{i}"}
            ))

        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        state = session.state
        tool_blocks = [b for b in state.blocks if b.kind == "tool_call"]
        assert len(tool_blocks) == 4
        assert all(b.status == "completed" for b in tool_blocks)


class TestHeaderSubtitle:
    """Model/tool changes update header subtitle."""

    def test_tool_model_changed_updates_subtitle(self):
        client = RecordingClient()
        session = _make_session(client, tool_name="coco", model_name="gpt-4o")

        session.dispatch(CardEvent(type=CardEventType.STARTED))

        # Change model
        session.dispatch(CardEvent(
            type=CardEventType.TOOL_MODEL_CHANGED,
            payload={"tool_name": "claude", "model_name": "claude-4-sonnet"}
        ))

        state = session.state
        assert state.metadata.tool_name == "claude"
        assert state.metadata.model_name == "claude-4-sonnet"
        # Tool/model info is now shown in footer, not header subtitle
        assert state.header.subtitle is None


class TestSpecEngineFlow:
    """Simulates a Spec Engine flow: STARTED → PHASE → CRITERIA → COMPLETED."""

    def test_spec_phase_criteria_completed(self):
        client = RecordingClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(
            engine_type="spec",
            mode_name="Spec · Coco",
            mode_emoji="📋",
        )
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_spec",
            config=config,
            delivery=delivery,
            session_id="e2e_spec",
        )

        # Started
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        assert any(c["op"] == "create" for c in client.calls)

        # Phase started
        session.dispatch(CardEvent(type=CardEventType.PHASE_STARTED, payload={
            "phase": "Plan", "phase_index": 1, "total_phases": 5,
        }))

        # Criteria update
        session.dispatch(CardEvent(type=CardEventType.CRITERIA_UPDATED, payload={
            "satisfied_count": 2, "total_count": 5, "content": "AC-1 ✅\nAC-2 ✅\nAC-3 ❌\nAC-4 ❌\nAC-5 ❌",
        }))

        # Phase done
        session.dispatch(CardEvent(type=CardEventType.PHASE_DONE, payload={
            "phase": "Plan", "phase_index": 1, "total_phases": 5,
        }))

        # Completed
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        assert session.closed
        state = session.state
        assert state.terminal == "completed"
        # Verify criteria were recorded
        assert state.engine_ext is not None
        assert state.engine_ext.criteria_total == 5
        assert state.engine_ext.criteria_satisfied == 2


class _SpyHook:
    """Test spy implementing SessionHook protocol for verifying hook calls."""

    def __init__(self):
        self.dispatched_events: list[CardEvent] = []
        self.terminal_calls: list[tuple[CardState, str]] = []
        self._terminal_event = threading.Event()

    def on_dispatched(self, event: CardEvent, state: CardState) -> None:
        self.dispatched_events.append(event)

    def on_terminal(self, state: CardState, reason: str) -> None:
        self.terminal_calls.append((state, reason))
        self._terminal_event.set()

    def wait_terminal(self, timeout: float = 5.0) -> bool:
        return self._terminal_event.wait(timeout)


class TestSessionHookIntegration:
    """Verify SessionHook lifecycle integration in the full dispatch pipeline."""

    def test_hooks_fire_on_completed(self):
        """Hooks.on_terminal must be called when session reaches COMPLETED."""
        client = RecordingClient()
        spy = _SpyHook()

        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_hook_test",
            config=config,
            delivery=delivery,
            session_id="hook_sess",
            hooks=(spy,),
        )

        # Dispatch lifecycle
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED, payload={"block_id": "t1"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "t1", "text": "Hello"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE, payload={"block_id": "t1"}))
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        # Wait for async hook execution
        assert spy.wait_terminal(timeout=5.0), "on_terminal was not called within timeout"
        assert len(spy.terminal_calls) == 1
        final_state, reason = spy.terminal_calls[0]
        assert reason == "completed"
        assert final_state.terminal == "completed"

    def test_hooks_fire_on_failed(self):
        """Hooks.on_terminal must also fire for FAILED events."""
        client = RecordingClient()
        spy = _SpyHook()

        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_hook_fail",
            config=config,
            delivery=delivery,
            session_id="hook_fail_sess",
            hooks=(spy,),
        )

        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.FAILED, payload={"error": "timeout"}))

        assert spy.wait_terminal(timeout=5.0), "on_terminal was not called for FAILED"
        assert len(spy.terminal_calls) == 1
        _, reason = spy.terminal_calls[0]
        assert reason == "failed"


class TestTTLAutoClose:
    """Verify TTL expiry auto-closes session and notifies user."""

    def test_ttl_expiry_closes_session(self):
        """When idle time exceeds TTL, session should close with cancelled state."""
        client = RecordingClient()
        notifications: list[str] = []

        def notify_cb(chat_id: str, text: str) -> None:
            notifications.append(text)

        # Use a very short TTL for testing
        delivery = CardDelivery(client)
        metadata = CardMetadata(
            mode_name="Deep",
            tool_name="coco",
            model_name="gpt-4o",
            engine_type="deep",
        )

        # Controllable clock: starts at 0, advances when we want
        clock_value = [0.0]

        def fake_clock() -> float:
            return clock_value[0]

        config = SessionConfig(
            metadata=metadata,
            ttl_seconds=10.0,  # 10 second TTL
            warn_before_seconds=3.0,  # warn at 7s
            clock=fake_clock,
        )
        session = CardSession(
            chat_id="chat_ttl_test",
            config=config,
            delivery=delivery,
            session_id="ttl_sess",
            callbacks=SessionCallbacks(notify_callback=notify_cb),
        )

        # Start the session
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        assert not session.closed

        # Advance clock past TTL
        clock_value[0] = 15.0

        # Trigger TTL check manually (the timer handler)
        ttl_handler = session._ttl_handler
        ttl_handler.on_ttl_expired()

        # Session should now be closed
        assert session.closed
        state = session.state
        assert state.terminal == "cancelled"


class TestTTLExpiredRestart:
    """Verify TTL expired CANCELLED state renders a restart button."""

    def test_ttl_expired_cancelled_has_restart_button(self):
        """CANCELLED(reason='ttl_expired') should produce a restart button in state."""
        client = RecordingClient()

        delivery = CardDelivery(client)
        metadata = CardMetadata(
            mode_name="Deep",
            tool_name="coco",
            model_name="gpt-4o",
            engine_type="deep",
        )

        clock_value = [0.0]

        def fake_clock() -> float:
            return clock_value[0]

        config = SessionConfig(
            metadata=metadata,
            ttl_seconds=10.0,
            warn_before_seconds=3.0,
            clock=fake_clock,
        )
        session = CardSession(
            chat_id="chat_ttl_restart",
            config=config,
            delivery=delivery,
            session_id="ttl_restart_sess",
            callbacks=SessionCallbacks(notify_callback=lambda _c, _t: None),
        )

        # Start session
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        assert not session.closed

        # Advance clock past TTL
        clock_value[0] = 15.0

        # Trigger TTL expiry
        ttl_handler = session._ttl_handler
        ttl_handler.on_ttl_expired()

        # Verify closed with cancelled/ttl_expired
        assert session.closed
        state = session.state
        assert state.terminal == "cancelled"
        assert state.terminal_reason == "ttl_expired"

        # Key assertion: restart button should be present
        assert len(state.buttons) >= 1
        restart_btn = state.buttons[0]
        assert restart_btn.type == "primary"
        assert "intent.deep.resume" in restart_btn.action_id
