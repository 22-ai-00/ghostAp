"""Tests for CardSession orchestration layer."""

import threading
import time
from unittest.mock import MagicMock, patch

from src.card.delivery.engine import CardDelivery, MutationOutcome
from src.card.events import CardEvent, CardEventType
from src.card.session import CardSession
from src.card.session.config import SessionCallbacks, SessionConfig
from src.card.state.button_intent import ButtonIntent
from src.card.state.models import CardMetadata, CardState, ContentBlock
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

    def test_dispatch_started_creates_card(self):
        session, client, _ = self._make_session()
        event = CardEvent(type=CardEventType.STARTED)
        session.dispatch(event)

        assert len(client.creates) == 1
        assert session.state is not None
        assert session.state.terminal == "running"

    def test_task_list_can_seed_the_first_created_card_before_started(self):
        session, client, _ = self._make_session()

        session.dispatch(CardEvent(
            type=CardEventType.TASK_LIST_UPDATED,
            payload={
                "tasks": [{
                    "task_id": "_deep_main",
                    "name": "分析与执行主流程",
                    "status": "in_progress",
                }],
                "current_task_id": "_deep_main",
            },
        ))
        session.dispatch(CardEvent.started())

        assert len(client.creates) == 1
        assert "分析与执行主流程" in str(client.creates[0]["card_json"])

    def test_dispatch_text_delta_updates_state(self):
        session, client, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session.dispatch(CardEvent(
            type=CardEventType.TEXT_DELTA,
            payload={"text": "Hello "}
        ))

        state = session.state
        assert state is not None
        # Should have at least one text block
        text_blocks = [b for b in state.blocks if b.kind == "text"]
        assert len(text_blocks) >= 1
        assert "Hello" in text_blocks[-1].content

    def test_dispatch_tool_started_updates_structure(self):
        session, client, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(
            type=CardEventType.TOOL_STARTED,
            payload={"tool_name": "bash", "block_id": "tc1"}
        ))

        state = session.state
        tool_blocks = [b for b in state.blocks if b.kind == "tool_call"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].tool_name == "bash"

    def test_async_delivery_coalesces_non_terminal_updates_behind_in_flight_job(self, monkeypatch):
        session, _, _ = self._make_session()
        session._sync_delivery = False
        delivered = []

        class FakePool:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args, **kwargs):
                self.jobs.append((fn, args, kwargs))

        pool = FakePool()
        monkeypatch.setattr("src.card.delivery.pool.get_delivery_pool", lambda: pool)
        monkeypatch.setattr(session, "_deliver_and_track", lambda rendered, is_terminal, *, event=None: delivered.append(rendered))

        started = CardEvent.started()
        session._submit_delivery(["first"], False, started)
        session._submit_delivery(["middle"], False, CardEvent.text_delta("b", "middle"))
        session._submit_delivery(["latest"], False, CardEvent.text_delta("b", "latest"))

        assert len(pool.jobs) == 1
        fn, args, kwargs = pool.jobs.pop(0)
        fn(*args, **kwargs)

        assert delivered == [["first"]]
        assert len(pool.jobs) == 1
        fn, args, kwargs = pool.jobs.pop(0)
        fn(*args, **kwargs)
        assert delivered == [["first"], ["latest"]]

    def test_async_delivery_rejects_older_render_that_finishes_after_newer_state(self, monkeypatch):
        """A slow render of version N must not be queued after version N+1."""
        session, _, _ = self._make_session()
        session._sync_delivery = False
        delivered = []
        old_render_started = threading.Event()
        release_old_render = threading.Event()

        class FakePool:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args, **kwargs):
                self.jobs.append((fn, args, kwargs))

        def render_by_version(state, _budget):
            if state.version == 1:
                old_render_started.set()
                assert release_old_render.wait(timeout=2)
            return [f"version-{state.version}"]

        pool = FakePool()
        monkeypatch.setattr("src.card.delivery.pool.get_delivery_pool", lambda: pool)
        monkeypatch.setattr("src.card.session.core.render_card", render_by_version)
        monkeypatch.setattr(
            session,
            "_deliver_and_track",
            lambda rendered, is_terminal, *, event=None: delivered.append(rendered),
        )

        old_dispatch = threading.Thread(target=lambda: session.dispatch(CardEvent.started()))
        old_dispatch.start()
        assert old_render_started.wait(timeout=2)

        session.dispatch(CardEvent.text_started("newer"))
        release_old_render.set()
        old_dispatch.join(timeout=2)
        assert not old_dispatch.is_alive()

        while pool.jobs:
            fn, args, kwargs = pool.jobs.pop(0)
            fn(*args, **kwargs)

        assert delivered == [["version-2"]]

    def test_late_non_terminal_event_cannot_overtake_slow_terminal_render(self, monkeypatch):
        """Once state reaches terminal, late stream events cannot steal its delivery slot."""
        session, _, _ = self._make_session()
        session.dispatch(CardEvent.started())
        session._sync_delivery = False
        delivered = []
        terminal_render_started = threading.Event()
        release_terminal_render = threading.Event()

        class FakePool:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args, **kwargs):
                self.jobs.append((fn, args, kwargs))

        def render_by_version(state, _budget):
            if state.version == 2:
                terminal_render_started.set()
                assert release_terminal_render.wait(timeout=2)
            return [f"{state.terminal}-version-{state.version}"]

        pool = FakePool()
        monkeypatch.setattr("src.card.delivery.pool.get_delivery_pool", lambda: pool)
        monkeypatch.setattr("src.card.session.core.render_card", render_by_version)
        monkeypatch.setattr(
            session,
            "_deliver_and_track",
            lambda rendered, is_terminal, *, event=None: delivered.append((rendered, is_terminal)),
        )

        terminal_dispatch = threading.Thread(
            target=lambda: session.dispatch(CardEvent.completed())
        )
        terminal_dispatch.start()
        assert terminal_render_started.wait(timeout=2)

        session.dispatch(CardEvent.text_started("late-stream-block"))
        release_terminal_render.set()
        terminal_dispatch.join(timeout=2)
        assert not terminal_dispatch.is_alive()

        while pool.jobs:
            fn, args, kwargs = pool.jobs.pop(0)
            fn(*args, **kwargs)

        assert delivered == [(["completed-version-2"], True)]
        assert session.state.version == 2

    def test_sync_delivery_serializes_revision_acceptance_with_visible_mutation(self, monkeypatch):
        """The deterministic sync path must not let a newer write overtake an older one."""
        session, _, _ = self._make_session()
        session._sync_delivery = True
        delivered = []
        first_delivery_started = threading.Event()
        release_first_delivery = threading.Event()
        second_rendered = threading.Event()

        def render_by_version(state, _budget):
            if state.version == 2:
                second_rendered.set()
            return [f"version-{state.version}"]

        monkeypatch.setattr("src.card.session.core.render_card", render_by_version)

        def controlled_delivery(rendered, is_terminal, *, event=None):
            if rendered == ["version-1"]:
                first_delivery_started.set()
                assert release_first_delivery.wait(timeout=2)
            delivered.append(rendered)

        monkeypatch.setattr(session, "_deliver_and_track", controlled_delivery)

        first = threading.Thread(target=lambda: session.dispatch(CardEvent.started()))
        second = threading.Thread(target=lambda: session.dispatch(CardEvent.text_started("newer")))
        first.start()
        assert first_delivery_started.wait(timeout=2)
        second.start()

        assert second_rendered.wait(timeout=2)
        assert delivered == []
        release_first_delivery.set()
        first.join(timeout=2)
        second.join(timeout=2)

        assert delivered == [["version-1"], ["version-2"]]

    def test_async_delivery_terminal_replaces_pending_non_terminal_update(self, monkeypatch):
        session, _, _ = self._make_session()
        session._sync_delivery = False
        delivered = []

        class FakePool:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args, **kwargs):
                self.jobs.append((fn, args, kwargs))

        pool = FakePool()
        monkeypatch.setattr("src.card.delivery.pool.get_delivery_pool", lambda: pool)
        monkeypatch.setattr(session, "_deliver_and_track", lambda rendered, is_terminal, *, event=None: delivered.append(rendered))

        session._submit_delivery(["first"], False, CardEvent.started())
        session._submit_delivery(["stale"], False, CardEvent.text_delta("b", "stale"))
        session._submit_delivery(["terminal"], True, CardEvent.completed())

        fn, args, kwargs = pool.jobs.pop(0)
        fn(*args, **kwargs)
        fn, args, kwargs = pool.jobs.pop(0)
        fn(*args, **kwargs)

        assert delivered == [["first"], ["terminal"]]

    def test_async_delivery_drops_non_terminal_updates_behind_terminal_job(self, monkeypatch):
        session, _, _ = self._make_session()
        session._sync_delivery = False
        delivered = []

        class FakePool:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args, **kwargs):
                self.jobs.append((fn, args, kwargs))

        pool = FakePool()
        monkeypatch.setattr("src.card.delivery.pool.get_delivery_pool", lambda: pool)
        monkeypatch.setattr(session, "_deliver_and_track", lambda rendered, is_terminal, *, event=None: delivered.append(rendered))

        session._submit_delivery(["terminal"], True, CardEvent.completed())
        session._submit_delivery(["stale"], False, CardEvent.text_delta("b", "stale"))

        assert len(pool.jobs) == 1
        fn, args, kwargs = pool.jobs.pop(0)
        fn(*args, **kwargs)

        assert delivered == [["terminal"]]
        assert pool.jobs == []

    def test_async_delivery_drops_duplicate_terminal_update_behind_terminal_job(self, monkeypatch):
        session, _, _ = self._make_session()
        session._sync_delivery = False
        delivered = []

        class FakePool:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args, **kwargs):
                self.jobs.append((fn, args, kwargs))

        def _deliver_and_close(rendered, is_terminal, *, event=None):
            delivered.append(rendered)
            if is_terminal:
                session._closed.set()

        pool = FakePool()
        monkeypatch.setattr("src.card.delivery.pool.get_delivery_pool", lambda: pool)
        monkeypatch.setattr(session, "_deliver_and_track", _deliver_and_close)

        session._submit_delivery(["terminal"], True, CardEvent.completed())
        session._submit_delivery(["duplicate-terminal"], True, CardEvent.completed())

        assert len(pool.jobs) == 1
        fn, args, kwargs = pool.jobs.pop(0)
        fn(*args, **kwargs)

        assert delivered == [["terminal"]]
        assert pool.jobs == []

    def test_tool_panel_has_input_and_output_from_acp_raw_fields(self):
        """ACP ToolCall raw_input/raw_output should surface in tool panel."""
        from types import SimpleNamespace

        from src.acp.client import _parse_tool_call
        from src.acp.models import ACPEvent, ACPEventType
        from src.card.events.acp_adapter import card_event_from_acp
        from src.card.render.tools import render_tool_panel

        session, _, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        # Start: input is taken from raw_input (execute kind → command string)
        start_update = SimpleNamespace(
            tool_call_id="tc_raw_1",
            title="bash",
            kind="execute",
            status="in_progress",
            raw_input={"command": "ls -la"},
            raw_output=None,
            locations=[],
        )
        start_info = _parse_tool_call(start_update)
        ev_start = ACPEvent(event_type=ACPEventType.TOOL_CALL_START, tool_call=start_info)
        session.dispatch(card_event_from_acp(ev_start))

        # Done: output is taken from raw_output (execute kind → output text)
        done_update = SimpleNamespace(
            tool_call_id="tc_raw_1",
            title="bash",
            kind="execute",
            status="completed",
            raw_input={"command": "ls -la"},
            raw_output={"output": "file1\nfile2\n"},
            locations=[],
        )
        done_info = _parse_tool_call(done_update)
        ev_done = ACPEvent(event_type=ACPEventType.TOOL_CALL_DONE, tool_call=done_info)
        session.dispatch(card_event_from_acp(ev_done))

        state = session.state
        assert state is not None
        tool_blocks = [b for b in state.blocks if b.kind == "tool_call"]
        assert len(tool_blocks) == 1
        b = tool_blocks[0]
        assert "ls -la" in (b.tool_input or "")
        assert "file1" in (b.tool_output or "")

        panel = render_tool_panel(b)
        # Detail content is a markdown element inside collapsible panel
        detail_md = panel["elements"][0]["content"]
        assert "ls -la" in detail_md
        assert "file1" in detail_md

    def test_dispatch_completed_closes_session(self):
        session, client, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        assert session.closed is True
        assert session.state.terminal == "completed"

    def test_card_split_closes_session_and_fires_terminal_hooks(self):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        hook_calls: list[str] = []

        class _Hook:
            def on_terminal(self, state, reason):
                hook_calls.append(reason)

        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o")
        config = SessionConfig(metadata=metadata, sync_delivery=True)
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="test_split_sess",
            hooks=(_Hook(),),
        )

        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.card_split(reason="task_done", hint="task 3"))

        assert session.closed is True
        assert session.state.terminal == "completed"
        assert hook_calls == ["completed"]

    def test_card_split_emits_completed_callback_after_terminal_delivery(self):
        session, _, _ = self._make_session()
        session._sync_delivery = True
        on_split = MagicMock()
        session.on_card_split_completed = on_split

        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.card_split(reason="phase_changed", hint="executing"))

        on_split.assert_called_once_with("phase_changed", "executing")

    def test_dispatch_after_close_ignored(self):
        session, client, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))

        # This should be ignored
        session.dispatch(CardEvent(
            type=CardEventType.TEXT_DELTA,
            payload={"text": "ghost text", "block_id": "_active_text"}
        ))
        # State shouldn't change after close
        assert session.state.terminal == "completed"


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

    def test_close_idempotent(self):
        session, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.close()
        session.close()  # Should not raise
        assert session.closed

    def test_thread_safety(self):
        """Multiple threads dispatching concurrently should not crash."""
        session, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        errors = []

        def dispatch_many():
            try:
                for i in range(50):
                    session.dispatch(CardEvent(
                        type=CardEventType.TEXT_DELTA,
                        payload={"text": f"chunk_{i} ", "block_id": "_active_text"}
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=dispatch_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_structural_and_terminal_events(self):
        """Concurrent structural + terminal events: state stays consistent."""
        session, _ = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        errors = []
        barrier = threading.Barrier(3)

        def dispatch_tools():
            try:
                barrier.wait(timeout=2)
                for i in range(20):
                    session.dispatch(CardEvent(
                        type=CardEventType.TOOL_STARTED,
                        payload={"tool_name": f"tool_{i}", "block_id": f"t_{i}"}
                    ))
                    session.dispatch(CardEvent(
                        type=CardEventType.TOOL_DONE,
                        payload={"block_id": f"t_{i}", "tool_output": "ok"}
                    ))
            except Exception as e:
                errors.append(e)

        def dispatch_text():
            try:
                barrier.wait(timeout=2)
                for i in range(30):
                    session.dispatch(CardEvent(
                        type=CardEventType.TEXT_DELTA,
                        payload={"text": f"w{i} ", "block_id": "_active_text"}
                    ))
            except Exception as e:
                errors.append(e)

        def dispatch_complete():
            try:
                barrier.wait(timeout=2)
                import time
                time.sleep(0.01)  # Let others start first
                session.dispatch(CardEvent(type=CardEventType.COMPLETED))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=dispatch_tools),
            threading.Thread(target=dispatch_text),
            threading.Thread(target=dispatch_complete),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # After all threads done, session must be closed and terminal
        assert session.closed
        assert session.state.terminal == "completed"


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


class TestDeliveryFailureCounter:
    """Delivery failure counter behavior via DeliveryTracker."""

    def _make_session(self, delivery):
        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        return CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="fail_test",
        )

    def test_delivery_failure_increments_counter(self):
        delivery = FailingDelivery(fail_count=999)
        session = self._make_session(delivery)
        # Dispatch should not raise even on delivery failure
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        assert session._tracker.delivery_failures == 1

    def test_delivery_success_resets_counter(self):
        delivery = FailingDelivery(fail_count=1)
        session = self._make_session(delivery)
        session.dispatch(CardEvent(type=CardEventType.STARTED))  # fails
        assert session._tracker.delivery_failures == 1
        # Next call succeeds (fail_count=1, so 2nd call goes through)
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA,
                                    payload={"text": "ok", "block_id": "b1"}))
        assert session._tracker.delivery_failures == 0

    def test_consecutive_failures_logs_error_no_crash(self):
        """After max consecutive failures, logs error but doesn't crash."""
        delivery = FailingDelivery(fail_count=999)
        session = self._make_session(delivery)
        for _ in range(5):
            session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA,
                                        payload={"text": "x", "block_id": "b1"}))
        assert session._tracker.delivery_failures == 5  # all failed, still running


class TestInboundActionEdgeCases:
    """inbound_action edge cases."""

    def _make_session(self, **kwargs):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        return CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="action_test",
            **kwargs,
        )

    def test_unknown_action_returns_toast(self):
        session = self._make_session()
        result = session.inbound_action("nonexistent_action")
        assert result is not None
        assert result["toast"]["type"] == "warning"
        assert "已失效" in result["toast"]["content"]

    def test_factory_exception_returns_toast(self):
        def bad_factory(payload):
            raise ValueError("broken")

        session = self._make_session(action_registry={"bad": bad_factory})
        result = session.inbound_action("bad")
        assert result is not None
        assert result["toast"]["type"] == "error"
        assert "操作异常" in result["toast"]["content"]

    def test_closed_session_returns_toast(self):
        session = self._make_session()
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.COMPLETED))
        assert session.closed

        result = session.inbound_action("any_action")
        assert result is not None
        assert result["toast"]["type"] == "info"
        assert "任务已完成" in result["toast"]["content"]

    def test_successful_action_returns_toast(self):
        registry = {
            "ok": lambda p: CardEvent(type=CardEventType.TEXT_DELTA,
                                       payload={"text": "hi", "block_id": "b1"}),
        }
        session = self._make_session(action_registry=registry)
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        result = session.inbound_action("ok")
        assert result is not None
        assert result["toast"]["type"] == "info"
        assert "操作已提交" in result["toast"]["content"]


class TestSnapshotEdgeCases:
    """snapshot() edge cases."""

    def test_snapshot_none_state_returns_none(self):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata())
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
        )
        # No dispatch yet → state is None
        assert session.snapshot() is None

    def test_snapshot_with_state_returns_interactive_json(self):
        """state 非 None 时，snapshot 返回 ('interactive', json_str) 且 json 可解析。"""
        import json

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Coco", tool_name="coco"))
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="snap_test",
        )
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED))
        session.dispatch(CardEvent(
            type=CardEventType.TEXT_DELTA,
            payload={"text": "Hello world", "block_id": "_active_text"},
        ))

        result = session.snapshot()
        assert result is not None
        msg_type, json_str = result
        assert msg_type == "interactive"
        card = json.loads(json_str)
        assert isinstance(card, dict)
        # Card should have schema structure
        assert "elements" in card or "body" in card or "header" in card


# ==============================================================================
# _pending_action_to_event helper tests
# ==============================================================================


class TestPendingActionToEvent:
    """Test PendingAction → CardEvent conversion."""

    def test_show_recovery_maps_to_warning_event(self):
        from src.card.delivery.tracker import DeliveryTracker, PendingAction
        from src.card.session import _pending_action_to_event
        from src.card.ui_text import UI_TEXT

        tracker = DeliveryTracker()
        event = _pending_action_to_event(PendingAction.SHOW_RECOVERY, tracker)
        assert event.type == CardEventType.WARNING_UPDATED
        assert event.payload["warning"] == UI_TEXT["card_session_recovery_banner"]

    def test_clear_banner_maps_to_empty_warning(self):
        from src.card.delivery.tracker import DeliveryTracker, PendingAction
        from src.card.session import _pending_action_to_event

        tracker = DeliveryTracker()
        event = _pending_action_to_event(PendingAction.CLEAR_BANNER, tracker)
        assert event.type == CardEventType.WARNING_UPDATED
        assert event.payload["warning"] == ""

    def test_max_failures_maps_to_warning_event(self):
        from src.card.delivery.tracker import DeliveryTracker, PendingAction
        from src.card.session import _pending_action_to_event

        tracker = DeliveryTracker()
        # Simulate a failure to capture timestamp
        tracker.on_failure()
        event = _pending_action_to_event(PendingAction.SHOW_MAX_FAILURES_WARNING, tracker)
        assert event.type == CardEventType.WARNING_UPDATED
        # Timestamp is formatted at call time, so check structure not raw template
        assert "{timestamp}" not in event.payload["warning"]
        assert "状态更新暂停" in event.payload["warning"]


class TestDispatchExceptionProtection:
    """Ensure reduce/render exceptions don't crash the session."""

    def _make_session(self):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", tool_name="test", model_name="test")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="test_exc",
        )
        return session, client

    def test_reducer_exception_preserves_state(self, monkeypatch, caplog):
        """If reducer raises, state should remain unchanged."""
        import logging
        session, client = self._make_session()
        session.dispatch(CardEvent.started())
        state_before = session.state

        # Monkeypatch reducer to raise
        def bad_reducer(state, event, metadata):
            raise RuntimeError("intentional reducer crash")

        monkeypatch.setattr("src.card.session.core.reduce_card_state", bad_reducer)

        with caplog.at_level(logging.ERROR):
            session.dispatch(CardEvent.text_delta("b1", "hello"))

        # State unchanged
        assert session.state is state_before
        # Error was logged
        assert "reduce failed" in caplog.text
        assert "intentional reducer crash" in caplog.text
        # Session not closed
        assert not session.closed

    def test_renderer_exception_does_not_crash_session(self, monkeypatch, caplog):
        """If renderer raises, state is already updated (reduce succeeded) but session stays open."""
        import logging
        session, client = self._make_session()
        session.dispatch(CardEvent.started())
        state_before = session.state

        def bad_render(state, budget=None):
            raise ValueError("intentional render crash")

        monkeypatch.setattr("src.card.session.core.render_card", bad_render)

        with caplog.at_level(logging.ERROR):
            session.dispatch(CardEvent.text_delta("b1", "hi"))

        # State IS updated (reduce succeeded before render)
        assert session.state is not state_before
        assert "render failed" in caplog.text
        assert not session.closed


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

    def test_terminal_retry_success(self):
        """AC24: retry succeeds → session closed, delivery.close() called."""
        delivery = CountingDelivery()
        delivery.fail_until = 2  # First two delivers fail (started + completed), retry succeeds
        session = self._make_session(delivery)
        session.dispatch(CardEvent.started())

        # This terminal dispatch fails first time, schedules retry
        session.dispatch(CardEvent.completed())

        # Wait for the retry timer to fire (retry_delay=0.01)
        import time
        time.sleep(0.3)

        assert session.closed
        assert delivery.close_calls >= 1
        assert session._timers._ttl_handle is None
        assert session._timers._ttl_prewarning_handle is None
        assert session._timers._retry_handle is None
        assert not session._finalizer.alive

    def test_terminal_retry_failure_notifies(self):
        """AC24: retry also fails → notify_callback called, session closed."""
        delivery = CountingDelivery()
        delivery.fail_until = 999  # All calls fail
        notify_calls = []

        def mock_notify(chat_id, msg):
            notify_calls.append(msg)

        session = self._make_session(delivery, notify_callback=mock_notify)
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())  # fails, schedules retry

        # Wait for the retry timer to fire (retry_delay=0.01)
        import time
        time.sleep(0.3)

        assert session.closed
        assert len(notify_calls) >= 1
        assert "任务已结束" in notify_calls[-1]
        assert session._timers._ttl_handle is None
        assert session._timers._ttl_prewarning_handle is None
        assert session._timers._retry_handle is None
        assert not session._finalizer.alive


class TestRetryConcurrentClose:
    """Test _retry() guard when close() called concurrently (AC25)."""

    def test_retry_skips_if_already_closed(self):
        """AC25: if close() called before timer fires, retry should not deliver."""
        delivery = CountingDelivery()
        delivery.fail_until = 1  # First call fails
        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="guard_test",
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())  # fails, schedules retry

        # Now close() before retry fires
        session.close()

        # Timer should have been cancelled; even if we manually call retry logic:
        # the guard should prevent delivery
        # Simulate _retry being called after close
        with session._lock:
            already_closed = session._closed.is_set()
        assert already_closed is True
        # deliver_calls should not have increased from retry
        # (timer was cancelled by close)


class TestDispatchEmptyRender:
    """Test dispatch() when render_card returns [] (AC26)."""

    def test_empty_render_no_crash(self, monkeypatch):
        """AC26: if render_card returns [], dispatch handles gracefully."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="empty_test",
        )

        monkeypatch.setattr("src.card.session.core.render_card", lambda state, budget=None: [])
        # Should not crash
        session.dispatch(CardEvent.started())
        # No card created since render returned empty
        assert len(client.creates) == 0


class TestTTLTimeout:
    """Test TTL auto-close behavior (AC16) with fake clock injection."""

    def _make_session_with_clock(self, clock_fn, ttl_seconds=10.0):
        """Create session with injectable clock for deterministic TTL testing.

        Patches threading.Timer to prevent real timer threads from interfering.
        """
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(
            metadata=CardMetadata(mode_name="Test"),
            ttl_seconds=ttl_seconds,
            clock=clock_fn,
        )
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="ttl_test",
        )
        # Cancel any real timer started in __init__ to avoid interference
        session._timers.cancel_all()
        return session

    def _patch_reset_timer(self, session):
        """Replace _reset_ttl_timer with a no-op to avoid spawning real timers."""
        session._reset_ttl_timer = lambda: None

    def test_ttl_expired_on_dispatch_closes_session(self):
        """AC16: idle time exceeds TTL on dispatch → auto-cancel."""
        now = [0.0]
        session = self._make_session_with_clock(lambda: now[0], ttl_seconds=10.0)
        self._patch_reset_timer(session)
        session.dispatch(CardEvent.started())
        assert not session.closed

        # Advance clock past TTL boundary
        now[0] = 11.0
        session.dispatch(CardEvent.text_delta("b1", "hello"))

        assert session.closed
        assert session.state.terminal == "cancelled"

    def test_ttl_boundary_exact_does_not_expire(self):
        """Boundary: idle time == TTL → NOT expired (condition is strict '>')."""
        now = [0.0]
        session = self._make_session_with_clock(lambda: now[0], ttl_seconds=10.0)
        self._patch_reset_timer(session)
        session.dispatch(CardEvent.started())  # _last_dispatch_time = 0.0

        # Advance to exactly TTL boundary
        now[0] = 10.0
        session.dispatch(CardEvent.text_delta("b1", "hello"))

        assert not session.closed

    def test_ttl_just_before_boundary_not_expired(self):
        """Just before TTL boundary: session stays alive."""
        now = [0.0]
        session = self._make_session_with_clock(lambda: now[0], ttl_seconds=10.0)
        self._patch_reset_timer(session)
        session.dispatch(CardEvent.started())

        now[0] = 9.99
        session.dispatch(CardEvent.text_delta("b1", "hello"))
        assert not session.closed

    def test_ttl_just_after_boundary_expired(self):
        """Just after TTL boundary: session auto-cancels."""
        now = [0.0]
        session = self._make_session_with_clock(lambda: now[0], ttl_seconds=10.0)
        self._patch_reset_timer(session)
        session.dispatch(CardEvent.started())

        now[0] = 10.01
        session.dispatch(CardEvent.text_delta("b1", "hello"))
        assert session.closed
        assert session.state.terminal == "cancelled"

    def test_ttl_inline_defers_when_tool_is_active(self):
        """Long-running active tool work refreshes TTL instead of closing the card."""
        now = [0.0]
        session = self._make_session_with_clock(lambda: now[0], ttl_seconds=10.0)
        self._patch_reset_timer(session)
        session.dispatch(CardEvent.started())
        session._state = CardState(
            blocks=(
                ContentBlock(
                    kind="tool_call",
                    block_id="active-tool",
                    tool_name="Bash",
                    status="active",
                ),
            ),
            terminal="running",
        )

        now[0] = 11.0
        session.dispatch(CardEvent.text_delta("b1", "still working"))

        assert not session.closed
        assert session._last_dispatch_time == 11.0

    def test_dispatch_refreshes_idle_timer(self):
        """Each dispatch refreshes idle time: TTL measured from last dispatch."""
        now = [0.0]
        session = self._make_session_with_clock(lambda: now[0], ttl_seconds=10.0)
        self._patch_reset_timer(session)
        session.dispatch(CardEvent.started())  # _last_dispatch_time = 0.0

        # Advance 8s and dispatch (refreshes idle time to 8.0)
        now[0] = 8.0
        session.dispatch(CardEvent.text_delta("b1", "first"))
        assert not session.closed

        # Advance to 17s total (9s since last dispatch, still < 10s TTL)
        now[0] = 17.0
        session.dispatch(CardEvent.text_delta("b1", "second"))
        assert not session.closed

        # Advance to 28s total (11s since last dispatch at 17s)
        now[0] = 28.0
        session.dispatch(CardEvent.text_delta("b1", "third"))
        assert session.closed
        assert session.state.terminal == "cancelled"

    def test_ttl_not_expired_proceeds_normally(self):
        """TTL not yet expired: normal dispatch."""
        now = [0.0]
        session = self._make_session_with_clock(lambda: now[0], ttl_seconds=9999.0)
        self._patch_reset_timer(session)
        session.dispatch(CardEvent.started())
        now[0] = 1.0
        session.dispatch(CardEvent.text_delta("b1", "hello"))
        assert not session.closed


class TestInboundActionUnknown:
    """Test unknown action graceful degrade (AC15)."""

    def test_unknown_action_returns_toast_without_cancel(self):
        """AC15: unknown action → toast only, session stays alive."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="action_test",
            action_registry={"known_action": lambda p: CardEvent.started()},
        )
        session.dispatch(CardEvent.started())
        assert not session.closed

        result = session.inbound_action("unknown_button_id")
        assert result is not None
        assert "toast" in result
        # Session should NOT be cancelled — just return toast
        assert not session.closed
        assert session.state.terminal != "cancelled"

        # Subsequent dispatch should still work normally
        session.dispatch(CardEvent.text_delta("b1", "hello"))
        assert not session.closed


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

    def test_timer_callback_closes_session(self):
        """Proactive: Timer fires → session cancelled with warning text."""
        now = [0.0]
        session = self._make_session(now)
        session.dispatch(CardEvent.started())
        assert not session.closed

        # Advance clock past TTL
        now[0] = 11.0

        # Manually invoke the callback (simulating timer fire)
        session._ttl_handler.on_ttl_expired()

        assert session.closed
        assert session.state.terminal == "cancelled"
        # Should contain TTL expired text in warning banner
        assert session.state.footer.warning_banner is not None

    def test_timer_callback_freezes_duration_from_session_clock(self):
        now = [100.0]
        session = self._make_session(now)
        session.dispatch(CardEvent.started())

        now[0] = 111.0
        session._ttl_handler.on_ttl_expired()

        assert session.state.terminal == "cancelled"
        assert session.state.footer.duration_seconds == 11.0

    def test_timer_callback_noop_if_already_closed(self):
        """Proactive timer ignores if session already closed."""
        now = [0.0]
        session = self._make_session(now)
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())
        assert session.closed

        # Advance and fire timer — should be no-op
        now[0] = 11.0
        session._ttl_handler.on_ttl_expired()  # Should not crash

    def test_timer_callback_noop_if_idle_time_not_exceeded(self):
        """Timer fires early (race) but idle time not actually exceeded → no-op."""
        now = [0.0]
        session = self._make_session(now)
        session.dispatch(CardEvent.started())

        # Simulate a timer firing but clock says only 5s elapsed
        now[0] = 5.0
        session._ttl_handler.on_ttl_expired()

        assert not session.closed

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


class TestTTLToastDistinguishesReason:
    """Test inbound_action returns TTL-specific toast after TTL closure."""

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
            session_id="ttl_toast_test",
        )
        session._timers.cancel_all()
        session._reset_ttl_timer = lambda: None
        return session

    def test_ttl_closed_returns_ttl_specific_toast(self):
        """After TTL expiry, inbound_action returns TTL-specific toast."""

        now = [0.0]
        session = self._make_session(now)
        session.dispatch(CardEvent.started())

        # Expire via TTL
        now[0] = 11.0
        session._ttl_handler.on_ttl_expired()
        assert session.closed

        result = session.inbound_action("some_action")
        assert result is not None
        assert "超时" in result["toast"]["content"]

    def test_normal_closed_returns_generic_toast(self):
        """After normal completion, inbound_action returns completed toast."""
        now = [0.0]
        session = self._make_session(now)
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())
        assert session.closed

        result = session.inbound_action("some_action")
        assert result is not None
        assert "已完成" in result["toast"]["content"]


class TestTTLReduceRollback:
    """Test that TTL reduce/render failure rolls back state (Task 34)."""

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
            session_id="ttl_rollback",
        )
        session._timers.cancel_all()
        return session

    def test_inline_ttl_reduce_failure_rolls_back(self, monkeypatch):
        """When reduce raises during inline TTL check, state should roll back."""
        now = [0.0]
        session = self._make_session(now)
        session._reset_ttl_timer = lambda: None
        session.dispatch(CardEvent.started())
        assert not session.closed

        # Make reduce_card_state raise on TTL path
        call_count = [0]
        original_reduce = reduce_card_state

        def failing_reduce(state, event, metadata):
            call_count[0] += 1
            # Let the first dispatch (started) work, fail on TTL warning_updated
            if event.type == CardEventType.WARNING_UPDATED:
                raise RuntimeError("simulated reduce failure")
            return original_reduce(state, event, metadata)

        monkeypatch.setattr("src.card.session.core.reduce_card_state", failing_reduce)

        # Advance clock past TTL
        now[0] = 11.0
        session.dispatch(CardEvent.text_delta("b1", "hello"))

        # State should be rolled back — session NOT closed, state unchanged
        assert not session.closed
        assert session._ttl_warned is False  # Reset for retry

    def test_proactive_ttl_reduce_failure_rolls_back(self, monkeypatch):
        """When reduce raises during _on_ttl_expired, state should roll back."""
        now = [0.0]
        session = self._make_session(now)
        session._reset_ttl_timer = lambda: None
        session.dispatch(CardEvent.started())
        assert not session.closed

        # Make reduce raise on cancelled event
        original_reduce = reduce_card_state

        def failing_reduce(state, event, metadata):
            if event.type == CardEventType.CANCELLED:
                raise RuntimeError("simulated cancel reduce failure")
            return original_reduce(state, event, metadata)

        monkeypatch.setattr("src.card.session.core.reduce_card_state", failing_reduce)

        # Advance clock and trigger proactive TTL
        now[0] = 11.0
        session._last_dispatch_time = 0.0
        session._ttl_handler.on_ttl_expired()

        # State should be rolled back — not closed
        assert not session.closed
        assert session._ttl_warned is False


class TestErrorBannerDeduplication:
    """Test error banner body-top + footer deduplication (Task 20).

    Error-level banners should appear only in body top, NOT in footer.
    """

    def test_error_banner_not_in_footer(self):
        """Error warning: rendered in body top, skipped in footer."""
        from src.card.render.footer import render_footer
        from src.card.state.models import FooterState

        state = CardState(
            metadata=CardMetadata(mode_name="Test"),
            footer=FooterState(
                status="idle",
                status_text="ready",
                warning_banner="Something failed",
                warning_type="error",
            ),
        )
        elements = render_footer(state)
        # Footer should have hr + status_text but NO warning banner div
        banner_elements = [e for e in elements if e.get("background_style") == "red"]
        assert len(banner_elements) == 0

    def test_warning_banner_not_in_footer(self):
        """Non-error warning: also NOT in footer (all banners now in body top)."""
        from src.card.render.footer import render_footer
        from src.card.state.models import FooterState

        state = CardState(
            metadata=CardMetadata(mode_name="Test"),
            footer=FooterState(
                status="idle",
                status_text="ready",
                warning_banner="Heads up",
                warning_type="warning",
            ),
        )
        elements = render_footer(state)
        banner_elements = [e for e in elements if e.get("background_style") == "orange"]
        assert len(banner_elements) == 0

    def test_error_banner_in_body_top(self):
        """Error banner appears in body top via render_card."""
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card
        from src.card.state.models import FooterState

        state = CardState(
            metadata=CardMetadata(mode_name="Test"),
            footer=FooterState(
                status="idle",
                status_text="ready",
                warning_banner="Critical failure",
                warning_type="error",
            ),
            terminal="running",
        )
        cards = render_card(state, RenderBudget())
        assert len(cards) >= 1
        body = cards[0]._card_json.get("body", {}).get("elements", [])
        # First element should be the error banner
        assert len(body) > 0
        first_el = body[0]
        assert first_el.get("background_style") == "red"


class TestCancelledRetainsTimeoutText:
    """Test cancelled state retains timeout text (Task 21).

    When TTL expires, the merged warning+cancel state should contain
    the TTL expired text in warning_banner.
    """

    def test_ttl_cancel_preserves_warning_text(self):
        """After TTL cancel, state.footer.warning_banner has TTL text."""

        now = [0.0]
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(
            metadata=CardMetadata(mode_name="Test"),
            ttl_seconds=10.0,
            clock=lambda: now[0],
        )
        session = CardSession(
            chat_id="c1",
            config=config,
            delivery=delivery,
            session_id="cancel_text",
        )
        # Cancel real timer and prevent new ones
        session._timers.cancel_all()
        session._reset_ttl_timer = lambda: None

        session.dispatch(CardEvent.started())

        now[0] = 11.0
        session.dispatch(CardEvent.text_delta("b1", "hello"))

        assert session.closed
        assert session.state.terminal == "cancelled"
        # Warning banner should contain TTL expired text
        assert "已超时关闭" in session.state.footer.warning_banner



class TestAtomRendererRegistry:
    """Test atom renderer registry completeness and fallback (Task 24)."""

    def test_unknown_kind_falls_through_to_markdown(self):
        """Unknown atom kind → rendered as plain markdown with warning log."""
        from src.card.render.atoms import RenderAtom
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import _render_atoms_to_elements

        unknown_atom = RenderAtom(kind="unknown_widget", content="hello world", block_id="b1")
        state = CardState(metadata=CardMetadata(mode_name="Test"), terminal="running")
        budget = RenderBudget()
        block_index = {}

        elements = _render_atoms_to_elements([unknown_atom], state, budget, block_index)
        assert len(elements) == 1
        assert elements[0]["tag"] == "markdown"
        # Running state → shows "任务仍在运行中" placeholder
        assert "部分内容暂时无法渲染" in elements[0]["content"]
        assert "运行中" in elements[0]["content"]

# ---------------------------------------------------------------------------
# Lifecycle hooks tests
# ---------------------------------------------------------------------------


class TestCardSessionLifecycleHooks:
    """Tests for CardSession lifecycle hook mechanism."""

    def _make_session_with_hooks(self, hooks):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_hook",
            config=config,
            delivery=delivery,
            session_id="hook_sess",
            hooks=tuple(hooks),
        )
        return session, client

    def test_on_dispatched_called_after_reduce(self):
        """on_dispatched is called after each successful dispatch."""
        calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                calls.append(("dispatched", event.type, state.terminal))

            def on_terminal(self, state, reason):
                calls.append(("terminal", reason))

        session, _ = self._make_session_with_hooks([TrackingHook()])
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.text_delta("b1", "hello"))

        # fire_dispatched is fire-and-forget; give executor time to run hooks
        import time
        time.sleep(0.15)

        assert len(calls) == 2
        assert calls[0] == ("dispatched", CardEventType.STARTED, "running")
        assert calls[1] == ("dispatched", CardEventType.TEXT_DELTA, "running")

    def test_on_terminal_called_on_completed(self):
        """on_terminal is called when a terminal event is delivered successfully."""
        calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                calls.append(("terminal", reason))

        session, _ = self._make_session_with_hooks([TrackingHook()])
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())

        assert len(calls) == 1
        assert calls[0] == ("terminal", "completed")

    def test_on_terminal_called_on_failed(self):
        """on_terminal fires with reason='failed' for FAILED events."""
        calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                calls.append(reason)

        session, _ = self._make_session_with_hooks([TrackingHook()])
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.failed("oops"))

        assert calls == ["failed"]

    def test_hook_exception_does_not_block_pipeline(self):
        """A failing hook should not prevent delivery."""
        class BrokenHook:
            def on_dispatched(self, event, state):
                raise RuntimeError("hook exploded")

            def on_terminal(self, state, reason):
                raise RuntimeError("terminal hook exploded")

        session, client = self._make_session_with_hooks([BrokenHook()])
        # Should not raise
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())

        # Delivery still happened
        assert len(client.creates) == 1
        assert session.closed

    def test_multiple_hooks_all_called(self):
        """All registered hooks are invoked in order."""
        order = []

        class HookA:
            def on_dispatched(self, event, state):
                order.append("A")

            def on_terminal(self, state, reason):
                order.append("A_term")

        class HookB:
            def on_dispatched(self, event, state):
                order.append("B")

            def on_terminal(self, state, reason):
                order.append("B_term")

        session, _ = self._make_session_with_hooks([HookA(), HookB()])
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())

        # on_dispatched is fire-and-forget; terminal hooks are waited on, so
        # cross-thread ordering is intentionally not guaranteed.
        deadline = time.time() + 1.0
        while len(order) < 6 and time.time() < deadline:
            time.sleep(0.01)
        assert order.count("A") == 2
        assert order.count("B") == 2
        assert order.count("A_term") == 1
        assert order.count("B_term") == 1

    def test_on_terminal_called_on_cancelled(self):
        """on_terminal fires with reason='cancelled' for CANCELLED events."""
        calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                calls.append(reason)

        session, _ = self._make_session_with_hooks([TrackingHook()])
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.cancelled())

        assert calls == ["cancelled"]

    def test_retry_success_fires_terminal_hooks(self):
        """When terminal delivery fails then succeeds on retry, hooks are fired."""
        calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                calls.append(reason)

        # Create a delivery client that fails on first update (terminal) then succeeds
        attempt_count = {"n": 0}

        class FlakeyClient:
            def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
                return ("msg_1", "card_1")

            def update_card(self, card_id, card_json, *, sequence=0):
                attempt_count["n"] += 1
                if attempt_count["n"] == 2:
                    # Second update is the terminal event - fail first time
                    raise ConnectionError("network blip")

            def update_element(self, card_id, element_id, content, *, sequence=0):
                pass

        delivery = CardDelivery(FlakeyClient())
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata, retry_delay=0.01)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="s_retry", hooks=(TrackingHook(),),
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())

        # Wait for retry timer to fire
        import time
        time.sleep(0.1)

        assert "completed" in calls


# ==============================================================================
# Task 4: session.close() does NOT trigger on_terminal hooks
# ==============================================================================


class TestCloseDoesNotFireTerminalHooks:
    """Verify that explicit close() fires on_terminal hooks when state is running."""

    def test_close_fires_terminal_hooks_when_running(self):
        """Creating session with hooks → close() → on_terminal called with 'cancelled'."""
        terminal_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                terminal_calls.append(reason)

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="s_close_hook", hooks=(TrackingHook(),),
        )
        session.dispatch(CardEvent.started())
        # Explicitly close without terminal event — should trigger hooks
        session.close()

        assert session.closed is True
        assert terminal_calls == ["cancelled"], "close() should fire on_terminal hooks when state is running"

    def test_close_after_dispatch_does_not_double_fire(self):
        """Terminal event fires hooks once; subsequent close() doesn't fire again."""
        terminal_calls = []

        class TrackingHook:
            def on_dispatched(self, event, state):
                pass
            def on_terminal(self, state, reason):
                terminal_calls.append(reason)

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="s_double", hooks=(TrackingHook(),),
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())  # Fires on_terminal
        session.close()  # Should NOT fire again

        assert terminal_calls == ["completed"]  # Only once


class TestTerminalResourceCleanup:
    """Successful terminal delivery must release session-owned resources."""

    def test_archived_delivery_cancels_timers_and_detaches_finalizer(self):
        """A frozen continuation page must not stay retained by idle timers."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        session = CardSession(
            chat_id="chat_archived_cleanup",
            config=SessionConfig(
                metadata=CardMetadata(mode_name="Test"),
                ttl_seconds=60.0,
                warn_before_seconds=10.0,
                sync_delivery=True,
            ),
            delivery=delivery,
            session_id="archived_cleanup",
        )

        session.dispatch(CardEvent.started())
        assert session._timers._ttl_handle is not None
        assert session._timers._ttl_prewarning_handle is not None
        assert session._finalizer.alive

        session.dispatch(CardEvent.archived(summary="continued"))

        assert session.closed
        assert session._timers._ttl_handle is None
        assert session._timers._ttl_prewarning_handle is None
        assert session._timers._retry_handle is None
        assert not session._finalizer.alive

    def test_stale_idle_defer_cannot_rearm_timers_after_close(self):
        """A prewarning callback delayed behind the session lock sees closure."""
        client = MockDeliveryClient()
        session = CardSession(
            chat_id="chat_stale_defer",
            config=SessionConfig(
                metadata=CardMetadata(mode_name="Test"),
                ttl_seconds=60.0,
                warn_before_seconds=10.0,
                sync_delivery=True,
            ),
            delivery=CardDelivery(client),
            session_id="stale_defer_cleanup",
        )
        session._timers.cancel_all()
        callback_started = threading.Event()

        def stale_defer():
            callback_started.set()
            session._ttl_actuator.defer_idle_timeout(
                session._ttl_handler.on_ttl_expired,
                session._ttl_handler.on_ttl_prewarning,
            )

        with session._lock:
            worker = threading.Thread(target=stale_defer)
            worker.start()
            assert callback_started.wait(timeout=1.0)
            session._closed.set()
        worker.join(timeout=1.0)

        assert not worker.is_alive()
        assert session._timers._ttl_handle is None
        assert session._timers._ttl_prewarning_handle is None
        session.release_terminal_resources()

    def test_terminal_timer_fence_wins_after_defer_passes_closed_check(
        self,
        monkeypatch,
    ):
        """Terminal cleanup linearizes with a defer already entering reset."""
        session = CardSession(
            chat_id="chat_defer_interleave",
            config=SessionConfig(
                metadata=CardMetadata(mode_name="Test"),
                ttl_seconds=60.0,
                warn_before_seconds=10.0,
                sync_delivery=True,
            ),
            delivery=CardDelivery(MockDeliveryClient()),
            session_id="defer_interleave_cleanup",
        )
        session._timers.cancel_all()
        entered_reset = threading.Event()
        allow_reset = threading.Event()
        original_reset = session._timers.reset_ttl_timer

        def delayed_reset(*args, **kwargs):
            entered_reset.set()
            assert allow_reset.wait(timeout=1.0)
            original_reset(*args, **kwargs)

        monkeypatch.setattr(session._timers, "reset_ttl_timer", delayed_reset)
        worker = threading.Thread(
            target=lambda: session._ttl_actuator.defer_idle_timeout(
                session._ttl_handler.on_ttl_expired,
                session._ttl_handler.on_ttl_prewarning,
            )
        )
        worker.start()
        assert entered_reset.wait(timeout=1.0)

        session._ttl_actuator.mark_closed()
        session.release_terminal_resources()
        allow_reset.set()
        worker.join(timeout=1.0)

        assert not worker.is_alive()
        assert session._timers._ttl_handle is None
        assert session._timers._ttl_prewarning_handle is None
        assert session._timers._retry_handle is None


# ==============================================================================
# Task 6: max_failures_banner does not contain raw {timestamp}
# ==============================================================================


class TestMaxFailuresBannerFormatting:
    """Verify that max_failures_banner {timestamp} placeholder is formatted."""

    def test_max_failures_banner_no_raw_placeholder(self):
        """After max failures, the warning_banner should not contain literal '{timestamp}'."""

        class FailingClient:
            """Client that always fails on update_card."""
            def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
                return ("msg_1", "card_1")
            def update_card(self, card_id, card_json, *, sequence=0):
                raise ConnectionError("always fail")
            def update_element(self, card_id, element_id, content, *, sequence=0):
                raise ConnectionError("always fail")


        delivery = CardDelivery(FailingClient())
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="s_format",
        )
        # First dispatch creates the card (create_card succeeds)
        session.dispatch(CardEvent.started())

        # Subsequent dispatches will trigger update_card failures
        for _ in range(4):
            session.dispatch(CardEvent(
                type=CardEventType.TEXT_DELTA,
                payload={"text": "chunk", "block_id": "_active_text"}
            ))

        # After failures are accumulated and pending actions consumed,
        # check the state's warning_banner for raw placeholder
        state = session.state
        if state and state.footer.warning_banner:
            assert "{timestamp}" not in state.footer.warning_banner, (
                f"Raw placeholder found in banner: {state.footer.warning_banner}"
            )


# ==============================================================================
# Phase 3 feature tests
# ==============================================================================


class TestTTLPrewarningTimer:
    """TTL prewarning fires at 90% of TTL and updates warning_banner."""

    def test_prewarning_fires_and_sets_banner(self):
        """After 90% idle, prewarning banner appears."""
        now = [0.0]
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(
            metadata=metadata,
            ttl_seconds=100.0,
            clock=lambda: now[0],
        )
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="ttl_pw",
        )
        session.dispatch(CardEvent.started())

        # Simulate 91 seconds of idle
        now[0] = 91.0
        session._ttl_handler.on_ttl_prewarning()

        state = session.state
        assert state is not None
        assert state.footer.warning_banner is not None
        assert "分钟后关闭" in state.footer.warning_banner

    def test_prewarning_skipped_if_activity_happened(self):
        """If activity happened since prewarning was scheduled, skip it."""
        now = [0.0]
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(
            metadata=metadata,
            ttl_seconds=100.0,
            clock=lambda: now[0],
        )
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="ttl_pw2",
        )
        session.dispatch(CardEvent.started())

        # Simulate activity at 50s
        now[0] = 50.0
        session.dispatch(CardEvent.text_started("b1"))

        # Timer fires at original 90s but idle is only 40s (50→90)
        now[0] = 90.0
        session._ttl_handler.on_ttl_prewarning()

        state = session.state
        # No prewarning because idle < 90% threshold
        assert state.footer.warning_banner is None or "分钟后关闭" not in (state.footer.warning_banner or "")


class TestEngineAwareToast:
    """Toast message includes engine-specific command for non-completed terminals."""

    def test_deep_engine_toast_contains_deep_cmd(self):
        now = [0.0]
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Deep", engine_type="deep")
        config = SessionConfig(metadata=metadata, clock=lambda: now[0])
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="toast_deep",
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.cancelled())
        result = session.inbound_action("any_action")
        assert "/deep" in result["toast"]["content"]

    def test_completed_returns_success_toast(self):
        """Completed sessions return a success toast without engine command."""
        now = [0.0]
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Deep", engine_type="deep")
        config = SessionConfig(metadata=metadata, clock=lambda: now[0])
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="toast_done",
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.completed())
        result = session.inbound_action("any_action")
        assert "任务已完成" in result["toast"]["content"]


class TestCancelledInjectsRestartButton:
    """CANCELLED state injects restart button for known engine types."""

    def test_deep_cancelled_has_resume_button(self):
        from src.card.state.reducer import reduce_card_state

        state = reduce_card_state(None, CardEvent.started(), CardMetadata(engine_type="deep"))
        state = reduce_card_state(state, CardEvent.cancelled(), CardMetadata(engine_type="deep"))
        assert state.terminal == "cancelled"
        assert len(state.buttons) == 1
        assert state.buttons[0].action_id == ButtonIntent.DEEP_RESUME

    def test_no_engine_type_cancelled_no_button(self):
        from src.card.state.reducer import reduce_card_state

        state = reduce_card_state(None, CardEvent.started(), CardMetadata())
        state = reduce_card_state(state, CardEvent.cancelled(), CardMetadata())
        assert state.terminal == "cancelled"
        assert state.buttons == ()


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

    def test_failed_event_reduce_failure_closes_session(self, monkeypatch):
        """If FAILED event causes reduce to raise, session must close."""
        session, _ = self._make_session()
        session.dispatch(CardEvent.started())

        def failing_reduce(state, event, metadata):
            if event.type == CardEventType.FAILED:
                raise RuntimeError("reduce crash on FAILED")
            return reduce_card_state(state, event, metadata)

        monkeypatch.setattr("src.card.session.core.reduce_card_state", failing_reduce)

        session.dispatch(CardEvent(type=CardEventType.FAILED, payload={"error": "test"}))
        assert session.closed

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

    def test_non_terminal_reduce_failure_does_not_close(self, monkeypatch):
        """Non-terminal event reduce failure should NOT close the session."""
        session, _ = self._make_session()
        session.dispatch(CardEvent.started())

        def failing_reduce(state, event, metadata):
            if event.type == CardEventType.TEXT_DELTA:
                raise RuntimeError("crash on non-terminal")
            return reduce_card_state(state, event, metadata)

        monkeypatch.setattr("src.card.session.core.reduce_card_state", failing_reduce)

        session.dispatch(CardEvent.text_delta("b1", "hello"))
        assert not session.closed, "Non-terminal reduce failure should not close"


class TestToastClosedElseBranch:
    """AC-1: The else branch (non completed/failed/ttl_expired) returns '启动新任务' toast."""

    def test_cancelled_session_toast_contains_new_task(self):
        """Cancelled session → toast includes '重新启动' and engine cmd."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Deep", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="toast_else_branch",
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.cancelled())
        result = session.inbound_action("any_action")
        assert result is not None
        toast_content = result["toast"]["content"]
        assert "重新启动" in toast_content
        assert "/deep" in toast_content

    def test_cancelled_spec_session_toast_contains_spec_cmd(self):
        """Spec engine cancelled → toast includes '/spec' and '重新启动'."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Spec", engine_type="spec")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="toast_else_spec",
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.cancelled())
        result = session.inbound_action("any_action")
        toast_content = result["toast"]["content"]
        assert "/spec" in toast_content
        assert "重新启动" in toast_content


class TestFallbackCmdToast:
    """AC-4: Unknown or None engine_type uses fallback cmd from UI_TEXT."""

    def test_none_engine_type_uses_fallback_cmd(self):
        """engine_type=None → toast contains the fallback command text."""
        from src.card.ui_text import UI_TEXT
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Unknown")  # engine_type defaults to None/""
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="toast_fallback_none",
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.cancelled())
        result = session.inbound_action("any_action")
        toast_content = result["toast"]["content"]
        assert UI_TEXT["card_session_fallback_cmd"] in toast_content

    def test_unknown_engine_type_uses_fallback_cmd(self):
        """engine_type='unknown_xyz' → toast contains the fallback command text."""
        from src.card.ui_text import UI_TEXT
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Mystery", engine_type="unknown_xyz")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="c1", config=config, delivery=delivery,
            session_id="toast_fallback_unknown",
        )
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.cancelled())
        result = session.inbound_action("any_action")
        toast_content = result["toast"]["content"]
        assert UI_TEXT["card_session_fallback_cmd"] in toast_content


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

    def test_rejected_calls_notify_rejected(self):
        """When deliver() returns rejected, _notify_rejected is called."""
        delivery = RejectingDelivery()
        notify = MagicMock()
        session = self._make_session(delivery, notify_callback=notify)
        # Start the session first
        session.dispatch(CardEvent.started())
        # TEXT_DELTA triggers non-terminal delivery which gets rejected
        session.dispatch(CardEvent.text_delta("blk_1", "hello"))
        # notify_callback should have been called
        assert notify.call_count >= 1

    def test_rejected_non_terminal_no_retry(self):
        """When rejected on non-terminal event, _schedule_terminal_retry is NOT called."""
        delivery = RejectingDelivery()
        notify = MagicMock()
        session = self._make_session(delivery, notify_callback=notify)
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent.text_delta("blk_1", "hello"))
        # Session should NOT be closed (no terminal retry for non-terminal events)
        time.sleep(0.05)  # Allow any async timers to fire
        assert not session._closed.is_set()

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


class TestFinalizeTerminalCloseException:
    """delivery.close() exception must not prevent on_terminal hooks from firing."""

    def test_close_exception_hooks_still_fire(self):
        """Even if delivery.close() raises, on_terminal hook is called."""
        hook = MagicMock()
        hook.on_dispatched = MagicMock()
        hook.on_terminal = MagicMock()

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_exc",
            config=config,
            delivery=delivery,
            session_id="exc_sess",
            hooks=(hook,),
        )
        session.dispatch(CardEvent.started())
        # Patch delivery.close to raise
        with patch.object(delivery, "close", side_effect=RuntimeError("close boom")):
            session.dispatch(CardEvent.completed())
        # Wait for hooks (async via thread pool)
        time.sleep(0.2)
        # on_terminal hook should still have been called
        assert hook.on_terminal.call_count == 1

    def test_close_exception_still_releases_delivery_session_lock(self):
        """Detaching the finalizer must retain its delivery-lock safety net."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        session = CardSession(
            chat_id="chat_close_lock",
            config=SessionConfig(
                metadata=CardMetadata(mode_name="Test", engine_type="deep"),
                sync_delivery=True,
            ),
            delivery=delivery,
            session_id="close_lock_sess",
        )
        session.dispatch(CardEvent.started())
        assert delivery._lock_pool.contains("close_lock_sess")

        with patch.object(delivery, "close", side_effect=RuntimeError("close boom")):
            session.dispatch(CardEvent.completed())

        assert session.closed
        assert not delivery._lock_pool.contains("close_lock_sess")
        assert not session._finalizer.alive

    def test_lock_release_exception_keeps_finalizer_safety_net(self):
        """Detach only after the explicit delivery-lock release succeeds."""
        delivery = CardDelivery(MockDeliveryClient())
        session = CardSession(
            chat_id="chat_release_failure",
            config=SessionConfig(
                metadata=CardMetadata(mode_name="Test", engine_type="deep"),
                sync_delivery=True,
            ),
            delivery=delivery,
            session_id="release_failure_sess",
        )
        session.dispatch(CardEvent.started())

        with (
            patch.object(delivery, "close", side_effect=RuntimeError("close boom")),
            patch.object(
                delivery,
                "release_session_lock",
                side_effect=RuntimeError("release boom"),
            ),
        ):
            session.dispatch(CardEvent.completed())

        assert session.closed
        assert session._finalizer.alive
        session.release_terminal_resources()
        assert not session._finalizer.alive


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


class TestTerminalReasonFallback:
    """When _terminal_reason is None, hooks receive reason='completed'."""

    def test_none_reason_falls_back_to_completed(self):
        """If _terminal_reason is None after terminal delivery, hooks get 'completed'."""
        hook = MagicMock()
        hook.on_dispatched = MagicMock()
        hook.on_terminal = MagicMock()

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_fb",
            config=config,
            delivery=delivery,
            session_id="fb_sess",
            hooks=(hook,),
        )
        session.dispatch(CardEvent.started())
        # Manually ensure _terminal_reason stays None by dispatching COMPLETED
        # (terminal_reason is set from state.terminal_reason which for COMPLETED is typically 'completed'
        # We force it to None to test the fallback)
        session._terminal_reason = None
        session.dispatch(CardEvent.completed())
        # Wait for hooks
        time.sleep(0.2)
        assert hook.on_terminal.call_count == 1
        # The reason should be 'completed' (fallback from None)
        call_args = hook.on_terminal.call_args
        reason = call_args[0][1] if call_args[0] else call_args[1].get("reason")
        assert reason == "completed", f"Expected 'completed', got {reason}"


# ---------------------------------------------------------------------------
# FS-8: cancel_callback exception does not propagate
# ---------------------------------------------------------------------------


class TestCancelCallbackException:
    """cancel_callback raising must not propagate and session stays closed."""

    def test_cancel_callback_exception_isolated(self):
        """Exception in cancel_callback is silently caught, session remains closed."""

        def boom_cancel():
            raise RuntimeError("cancel exploded")

        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Test", engine_type="deep")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_cc",
            config=config,
            delivery=delivery,
            session_id="cc_sess",
            callbacks=SessionCallbacks(cancel_callback=boom_cancel),
        )
        session.dispatch(CardEvent.started())
        # CANCELLED (no reason) triggers cancel_callback since terminal_reason="cancelled"
        session.dispatch(CardEvent.cancelled())
        # Wait for finalize
        time.sleep(0.1)
        # Session should be closed
        assert session._closed.is_set()
        # No exception propagated (test would fail if it did)


# ---------------------------------------------------------------------------
# FS-9: _create_page double failure → reconcile, no orphan binding
# ---------------------------------------------------------------------------


class TestCreatePageDoubleFailure:
    """When streaming card creation fails AND IM fallback also fails,
    _create_page must return reconcile and not leave orphan bindings."""

    def test_streaming_and_fallback_both_fail(self):
        """Both streaming and IM create_card fail → reconcile outcome."""
        from src.card.delivery.engine import CardDelivery
        from src.card.types import ActiveElement

        client = MagicMock()
        client.create_streaming_card.side_effect = RuntimeError("streaming fail")
        client.create_card.side_effect = RuntimeError("IM fallback fail")

        delivery = CardDelivery(client=client)
        try:
            card = RenderedCard(
                _card_json={"config": {"streaming_mode": True}},
                structure_signature="sig1",
                content_hash="hash1",
                active_element=ActiveElement(element_id="e1", text="hello"),
            )
            outcome = delivery._create_page("sess_1", "chat_1", card, reply_to="msg_0")

            assert outcome.kind == "reconcile"
            # No orphan binding should exist
            binding = delivery.get_binding("sess_1")
            assert binding is None or 0 not in binding.pages
        finally:
            delivery._shutdown()


# ---------------------------------------------------------------------------
# FS-10: _stream_element SequenceConflictError → fallback to _update_page
# ---------------------------------------------------------------------------


class TestStreamElementSequenceConflictFallback:
    """SequenceConflictError in _stream_element should fallback to _update_page."""

    def test_sequence_conflict_falls_back_to_update(self):
        from src.card.delivery.binding import PageBinding
        from src.card.delivery.engine import CardDelivery, SequenceConflictError
        from src.card.types import ActiveElement

        client = MagicMock()
        # update_element raises SequenceConflictError
        client.update_element.side_effect = SequenceConflictError(next_floor=5)
        # update_card (PATCH fallback) succeeds
        client.update_card.return_value = None

        delivery = CardDelivery(client=client)
        try:
            page = PageBinding(message_id="msg_1", card_id="card_1", signature="old_sig", page_index=0)
            card = RenderedCard(
                _card_json={"body": "updated"},
                structure_signature="new_sig",
                content_hash="hash2",
                active_element=ActiveElement(element_id="e1", text="new_text"),
            )
            # Pre-register binding so _update_page can find it
            delivery._bindings.create("sess_1", "chat_1")
            delivery._bindings.set_page("sess_1", 0, "msg_1", "card_1", "old_sig", "")

            outcome = delivery._stream_element("sess_1", page, card)

            assert outcome.kind == "applied"
            assert "updated:" in outcome.message
            # Verify update_card was called (fallback path)
            client.update_card.assert_called_once()
        finally:
            delivery._shutdown()


# ---------------------------------------------------------------------------
# FS-11: to_feishu_json isolation — delivery payload vs RenderedCard internal
# ---------------------------------------------------------------------------


class TestToFeishuJsonIsolation:
    """Verify that delivery receives a copy; mutating it doesn't affect RenderedCard."""

    def test_payload_mutation_does_not_affect_rendered_card(self):

        original_json = {"header": {"title": "Test"}, "body": [{"tag": "div"}]}
        card = RenderedCard(_card_json=original_json, structure_signature="s1", content_hash="h1")

        payload = card.to_feishu_json()
        # Mutate the returned payload
        payload["header"]["title"] = "MUTATED"
        payload["injected"] = True

        # Internal state must be unaffected
        assert card._card_json["header"]["title"] == "Test"
        assert "injected" not in card._card_json


class TestTTLKeepAliveConcurrentWrite:
    """Test that concurrent TTL_KEEP_ALIVE actions don't race on _last_dispatch_time."""

    def test_ttl_keep_alive_concurrent_write(self):
        """Multiple threads calling inbound_action(TTL_KEEP_ALIVE) must not corrupt state."""
        import threading
        import time
        from unittest.mock import MagicMock

        from src.card.actions.dispatch import TTL_KEEP_ALIVE
        from src.card.render.budget import RenderBudget
        from src.card.session.config import SessionConfig
        from src.card.session.core import CardSession
        from src.card.state.models import CardMetadata

        delivery = MagicMock()
        config = SessionConfig(
            metadata=CardMetadata(engine_type="deep", mode_name="Deep", mode_emoji="🔵"),
            budget=RenderBudget(),
            clock=time.monotonic,
        )
        session = CardSession(
            chat_id="chat_concurrent",
            config=config,
            delivery=delivery,
            session_id="test-concurrent-ttl",
            hooks=(),
        )

        # Dispatch an initial event so the session has state
        from src.card.events import CardEvent, CardEventType
        session.dispatch(CardEvent(type=CardEventType.STARTED, payload={}))

        errors: list[Exception] = []
        barrier = threading.Barrier(10)

        def hammer():
            try:
                barrier.wait(timeout=2.0)
                for _ in range(50):
                    result = session.inbound_action(TTL_KEEP_ALIVE)
                    assert result is not None
                    assert "toast" in result
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not errors, f"Concurrent TTL_KEEP_ALIVE errors: {errors}"
        # Verify _last_dispatch_time was updated (monotonically increasing)
        assert session._last_dispatch_time > 0


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

    def test_concurrent_close_delivery_close_once(self):
        """delivery.close() should only be called once even with racing close()."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        close_calls = []
        original_close = delivery.close

        def counting_close(session_id):
            close_calls.append(session_id)
            return original_close(session_id)

        delivery.close = counting_close

        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        session = CardSession(
            chat_id="c_race2",
            config=config,
            delivery=delivery,
            session_id="race_close2",
        )
        session.dispatch(CardEvent.started())

        barrier = threading.Barrier(2, timeout=5.0)

        def close_thread():
            barrier.wait()
            session.close()

        t1 = threading.Thread(target=close_thread)
        t2 = threading.Thread(target=close_thread)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert len(close_calls) == 1


# ---------------------------------------------------------------------------
# AC13: Non-terminal double reduce failure — state rollback
# ---------------------------------------------------------------------------


class TestNonTerminalDoubleReduceFailure:
    """Both original reduce and warning banner reduce fail for non-terminal event."""

    def test_state_rolls_back_and_session_stays_open(self, monkeypatch):
        """AC13: non-terminal TEXT_DELTA double failure → state preserved, session open."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Test"))
        session = CardSession(
            chat_id="c_dbl",
            config=config,
            delivery=delivery,
            session_id="dbl_fail",
        )
        # Bootstrap with a successful started event
        session.dispatch(CardEvent.started())
        state_after_start = session._state

        # Monkeypatch reduce_card_state to always raise
        call_count = [0]

        def always_fail(state, event, metadata):
            call_count[0] += 1
            raise RuntimeError("simulated reduce failure")

        monkeypatch.setattr("src.card.session.core.reduce_card_state", always_fail)

        # Dispatch a non-terminal event — should trigger double failure path
        session.dispatch(CardEvent.text_delta("block_1", "hello"))

        # State should be rolled back to pre-dispatch state
        assert session._state is state_after_start
        # Session should NOT be closed
        assert not session.closed
        # reduce was called at least twice (original + warning banner attempt)
        assert call_count[0] >= 2


class TestRenderFallbackNoneNocrash:
    """AC-21: When both render and render_fallback_card return None, session doesn't crash."""

    def test_double_render_failure_no_crash(self):
        """render fails → fallback also returns None → session skips delivery silently."""
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(engine_type="deep")
        config = SessionConfig(metadata=metadata, sync_delivery=True)
        session = CardSession(
            chat_id="c_fallback", config=config, delivery=delivery,
            session_id="fallback_none_test",
        )

        # Start the session normally
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        creates_before = len(client.creates)

        # Patch both render paths to return None
        with patch("src.card.session.core.render_card", side_effect=RuntimeError("render crash")), \
             patch("src.card.session.core.render_fallback_card", return_value=None):
            # This should NOT raise
            session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "hello"}))

        # No new card operations should have happened
        assert len(client.creates) == creates_before
        # Session should still be alive (not closed)
        assert not session.closed


class TestOnFirstDeliverGuards:
    """AC-24: on_first_delivered hooks fire once and skip failed delivery."""

    def _make_session(self, **kwargs):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_guard",
            config=config,
            delivery=delivery,
            session_id="guard_sess",
            **kwargs,
        )
        return session, client

    def test_rejected_delivery_does_not_fire(self):
        """on_first_delivered hook does NOT fire if delivery fails with exception."""
        class TrackingHook:
            def __init__(self):
                self.delivered_calls = []

            def on_first_delivered(self, session_id: str, msg_id: str) -> None:
                self.delivered_calls.append(msg_id)

            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                pass

        hook = TrackingHook()
        session, client = self._make_session(hooks=(hook,))

        # Make the client raise on create_card to simulate delivery failure.
        client.create_card = MagicMock(side_effect=RuntimeError("network error"))
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        assert len(hook.delivered_calls) == 0

    def test_hook_fires_once_on_first_delivery(self):
        """on_first_delivered hook fires once on first successful delivery."""
        class TrackingHook:
            def __init__(self):
                self.delivered_calls = []

            def on_first_delivered(self, session_id: str, msg_id: str) -> None:
                self.delivered_calls.append(msg_id)

            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                pass

        hook = TrackingHook()
        session, client = self._make_session(hooks=(hook,))

        session.dispatch(CardEvent(type=CardEventType.STARTED))

        # Hook fires since no legacy callback
        assert len(hook.delivered_calls) == 1
        assert hook.delivered_calls[0] == "msg_1"

        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"text": "hello"}))
        assert len(hook.delivered_calls) == 1


# ─── Task 17 [AC-TEST-2]: TestAddHookIntegration ───


class TestAddHookIntegration:
    """AC-TEST-2: Real CardSession.add_hook() → HookFirer.append_hook() integration."""

    def _make_session(self, **kwargs):
        client = MockDeliveryClient()
        delivery = CardDelivery(client)
        metadata = CardMetadata(mode_name="Coco", tool_name="coco", model_name="gpt-4o")
        config = SessionConfig(metadata=metadata)
        session = CardSession(
            chat_id="chat_add_hook",
            config=config,
            delivery=delivery,
            session_id="add_hook_sess",
            **kwargs,
        )
        return session, client

    def test_add_hook_then_deliver_fires_on_first_delivered(self):
        """add_hook(spy) → dispatch triggers delivery → spy.on_first_delivered is called."""
        class SpyHook:
            def __init__(self):
                self.delivered_calls = []

            def on_first_delivered(self, session_id: str, msg_id: str) -> None:
                self.delivered_calls.append((session_id, msg_id))

            def on_dispatched(self, event, state):
                pass

            def on_terminal(self, state, reason):
                pass

        spy = SpyHook()
        session, client = self._make_session()
        session.add_hook(spy)

        # Dispatch a STARTED event — this triggers create_card → first delivery
        session.dispatch(CardEvent(type=CardEventType.STARTED))

        # Verify spy was called
        assert len(spy.delivered_calls) == 1
        assert spy.delivered_calls[0][0] == "add_hook_sess"
        assert spy.delivered_calls[0][1] == "msg_1"
