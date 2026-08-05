"""Integration tests: event → reduce → render pipeline end-to-end.

Verifies that dispatching events through the reducer produces state that
renders correctly to card JSON.
"""
import json

from src.card.events import CardEvent, CardEventType
from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.session.config import SessionConfig
from src.card.state.button_intent import ButtonIntent
from src.card.state.models import CardMetadata, CardState
from src.card.state.reducer import reduce_card_state


def _dispatch_sequence(events: list[CardEvent], metadata: CardMetadata | None = None) -> CardState:
    """Dispatch a sequence of events and return final state."""
    meta = metadata or CardMetadata()
    state = None
    for event in events:
        state = reduce_card_state(state, event, meta)
    return state


class TestFailedPipeline:
    """FAILED event produces visible error + retry button in rendered card."""

    def test_failed_with_error_renders_error_text(self):
        meta = CardMetadata(engine_type="deep", mode_name="Deep Agent")
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.text_started("b1"),
            CardEvent.text_delta("b1", "Working..."),
            CardEvent.failed("Connection timeout after 30s"),
        ], meta)

        assert state.terminal == "failed"
        # Error block exists
        error_blocks = [b for b in state.blocks if b.block_id == "_error"]
        assert len(error_blocks) == 1
        assert "Connection timeout" in error_blocks[0].content
        # Retry button exists
        assert any(b.action_id == ButtonIntent.DEEP_RESUME for b in state.buttons)

        # Renders without error
        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1
        card_json = rendered[0]._card_json
        # Card JSON should be a valid dict
        assert isinstance(card_json, dict)
        # Serialize to verify JSON compliance
        json_str = json.dumps(card_json, ensure_ascii=False)
        assert "Connection timeout" in json_str

class TestCompletedPipeline:
    """COMPLETED event produces summary block in rendered card."""

    def test_completed_with_summary_renders_summary(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.text_started("b1"),
            CardEvent.text_delta("b1", "Done"),
            CardEvent.text_done("b1"),
            CardEvent.completed(summary="执行完成：调用了 5 个工具"),
        ])

        assert state.terminal == "completed"
        summary_blocks = [b for b in state.blocks if b.block_id == "_summary"]
        assert len(summary_blocks) == 1
        assert "5 个工具" in summary_blocks[0].content

        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1
        json_str = json.dumps(rendered[0]._card_json, ensure_ascii=False)
        assert "5 个工具" in json_str

    def test_completed_without_summary_no_extra_block(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.completed(),
        ])

        assert state.terminal == "completed"
        summary_blocks = [b for b in state.blocks if b.block_id == "_summary"]
        assert len(summary_blocks) == 0


class TestProgressPipeline:
    """PROGRESS_UPDATED event produces progress bar in footer."""

    def test_progress_updated_renders_progress_bar(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.progress_updated(current=3, total=10, label="步骤 3"),
        ])

        assert state.footer.progress is not None
        assert state.footer.progress_pct == 30
        assert "3/10" in state.footer.progress

        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1
        json_str = json.dumps(rendered[0]._card_json, ensure_ascii=False)
        assert "▰" in json_str  # progress bar chars

    def test_progress_zero_total_no_bar(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.progress_updated(current=0, total=0),
        ])
        assert state.footer.progress is None


class TestCriteriaUpdatedPipeline:
    """CRITERIA_UPDATED event inserts ContentBlock and renders criteria section."""

    def test_criteria_updated_creates_block_and_renders(self):
        meta = CardMetadata(engine_type="spec", mode_name="Spec")
        content = "✅ 条件1：通过\n❌ 条件2：未通过\n✅ 条件3：通过"
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.criteria_updated(content, satisfied_count=2, total_count=3),
        ], meta)

        # Verify block was created
        criteria_blocks = [b for b in state.blocks if b.block_id == "criteria_section"]
        assert len(criteria_blocks) == 1
        assert criteria_blocks[0].kind == "criteria"
        assert "条件1" in criteria_blocks[0].content
        # Verify engine_ext updated
        assert state.engine_ext.criteria_satisfied == 2
        assert state.engine_ext.criteria_total == 3

        # Verify renders without error
        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1
        json_str = json.dumps(rendered[0]._card_json, ensure_ascii=False)
        assert "条件1" in json_str

    def test_criteria_updated_replaces_existing_block(self):
        meta = CardMetadata(engine_type="spec", mode_name="Spec")
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.criteria_updated("v1", satisfied_count=0, total_count=2),
            CardEvent.criteria_updated("v2-updated", satisfied_count=1, total_count=2),
        ], meta)

        criteria_blocks = [b for b in state.blocks if b.block_id == "criteria_section"]
        assert len(criteria_blocks) == 1
        assert criteria_blocks[0].content == "v2-updated"
        assert state.engine_ext.criteria_satisfied == 1


class TestCyclePhasePipeline:
    """Cycle/Phase events produce correct state and render."""

    def test_cycle_phase_sequence(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.cycle_started(cycle_num=1, max_cycles=3),
            CardEvent.phase_started(cycle_num=1, phase="planning"),
            CardEvent.phase_done(cycle_num=1, phase="planning", output="Plan complete"),
            CardEvent.phase_started(cycle_num=1, phase="coding"),
        ])

        assert state.engine_ext.cycle_num == 1
        assert state.engine_ext.phase_info == "coding"
        # The card keeps one current phase-progress panel per cycle, so old
        # phase snapshots do not pin a stale status line at the top.
        phase_blocks = [b for b in state.blocks if b.kind == "phase"]
        assert len(phase_blocks) == 1
        assert phase_blocks[0].phase_name == "coding"
        assert phase_blocks[0].status == "active"

        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1


class TestCancelledPipeline:
    """Task 24: CANCELLED event produces correct terminal state."""

    def test_cancelled_clears_buttons_and_sets_terminal(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.text_started("b1"),
            CardEvent.text_delta("b1", "Working on it..."),
            CardEvent.cancelled(),
        ])

        assert state.terminal == "cancelled"
        assert state.buttons == ()

    def test_cancelled_renders_without_error(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.cancelled(),
        ])

        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1
        card_json = rendered[0]._card_json
        assert isinstance(card_json, dict)
        json_str = json.dumps(card_json, ensure_ascii=False)
        assert len(json_str) > 0

    def test_cancelled_after_progress_clears_progress(self):
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.progress_updated(current=5, total=10),
            CardEvent.cancelled(),
        ])

        assert state.terminal == "cancelled"
        # Footer should reflect terminal state
        assert state.footer.status is None or state.footer.status == "idle"


class TestDeepProgressVisualBar:
    """Deep engine progress_updated SHOULD produce progress_pct for ▰▱ bar."""

    def test_deep_progress_uses_visual_bar(self):
        meta = CardMetadata(engine_type="deep", mode_name="Deep")
        events = [
            CardEvent(type=CardEventType.STARTED, payload={}),
            CardEvent(type=CardEventType.PROGRESS_UPDATED, payload={"current": 2, "total": 4}),
        ]
        state = _dispatch_sequence(events, meta)
        assert state.footer.progress_pct == 50
        assert "步骤 2/4" in state.footer.progress


class TestEndToEndDeepSmoke:
    """End-to-end smoke test: CardSession → Reducer → Render → CardDelivery → API mock.

    Verifies the full pipeline from event dispatch to Feishu API call.
    """

    def test_deep_session_dispatches_to_api(self):
        """A complete Deep session lifecycle triggers create_card and update_card on the API client."""
        from unittest.mock import MagicMock

        from src.card.delivery.engine import CardDelivery
        from src.card.session import CardSession

        # Mock API client
        client = MagicMock()
        client.create_card.return_value = ("msg_001", "card_001")
        client.update_card.return_value = None

        delivery = CardDelivery(client)
        meta = CardMetadata(engine_type="deep", mode_name="Deep Agent", mode_emoji="🤖")
        config = SessionConfig(metadata=meta)
        session = CardSession(
            chat_id="chat_test",
            config=config,
            delivery=delivery,
            session_id="sess_e2e",
        )

        # Simulate deep engine lifecycle
        session.dispatch(CardEvent.started())
        session.dispatch(CardEvent(type=CardEventType.TEXT_STARTED, payload={"block_id": "b1"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "b1", "text": "Hello world"}))
        session.dispatch(CardEvent(type=CardEventType.TEXT_DONE, payload={"block_id": "b1"}))
        session.dispatch(CardEvent.completed(summary="Done"))

        # Verify API was called: at least one create and one update
        assert client.create_card.call_count >= 1, "Expected at least one create_card call"
        assert client.update_card.call_count >= 1, "Expected at least one update_card call"

        # Verify session is closed after COMPLETED
        assert session.closed


class TestSpecEngineE2ESmokeTest:
    """Spec engine full lifecycle: started → phases → criteria → completed."""

    def test_spec_full_lifecycle_renders(self):
        meta = CardMetadata(engine_type="spec", mode_name="Spec Engine")
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.cycle_started(cycle_num=1, max_cycles=10),
            CardEvent.phase_started(cycle_num=1, phase="spec"),
            CardEvent.phase_done(cycle_num=1, phase="spec", output="Spec written"),
            CardEvent.phase_started(cycle_num=1, phase="planning"),
            CardEvent.phase_done(cycle_num=1, phase="planning", output="Plan ready"),
            CardEvent.phase_started(cycle_num=1, phase="coding"),
            CardEvent.text_started("b1"),
            CardEvent.text_delta("b1", "Building module..."),
            CardEvent.text_done("b1"),
            CardEvent.phase_done(cycle_num=1, phase="coding", output="Module built"),
            CardEvent.phase_started(cycle_num=1, phase="review"),
            CardEvent.criteria_updated(
                "✅ Correctness\n❌ Performance", satisfied_count=1, total_count=2
            ),
            CardEvent.phase_done(cycle_num=1, phase="review", output="Needs perf fix"),
            CardEvent.cycle_started(cycle_num=2, max_cycles=10),
            CardEvent.phase_started(cycle_num=2, phase="coding"),
            CardEvent.phase_done(cycle_num=2, phase="coding", output="Perf optimized"),
            CardEvent.phase_started(cycle_num=2, phase="review"),
            CardEvent.criteria_updated(
                "✅ Correctness\n✅ Performance", satisfied_count=2, total_count=2
            ),
            CardEvent.phase_done(cycle_num=2, phase="review", output="All pass"),
            CardEvent.completed(summary="Spec complete in 2 cycles"),
        ], meta)

        assert state.terminal == "completed"
        assert state.engine_ext.cycle_num == 2
        assert state.engine_ext.criteria_satisfied == 2
        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1

    def test_spec_cancelled_with_retry_button(self):
        meta = CardMetadata(engine_type="spec", mode_name="Spec Engine")
        state = _dispatch_sequence([
            CardEvent.started(),
            CardEvent.cycle_started(cycle_num=1, max_cycles=10),
            CardEvent.cancelled(reason="ttl_expired"),
        ], meta)

        assert state.terminal == "cancelled"
        assert state.terminal_reason == "ttl_expired"
        # ttl_expired now also injects restart button + show_status button
        assert len(state.buttons) == 2
        assert "重新开始" in state.buttons[0].text
        assert "查看状态" in state.buttons[1].text
        rendered = render_card(state, RenderBudget())
        assert len(rendered) >= 1


# ---------------------------------------------------------------------------
# Retry pending banner integration (flag_retry_pending → dispatch enrichment)
# ---------------------------------------------------------------------------


class TestRetryPendingBannerIntegration:
    """Verify flag_retry_pending triggers SHOW_RETRY_PENDING banner in pipeline."""

    def test_flag_retry_pending_injects_banner_on_next_dispatch(self):
        """flag_retry_pending() → next dispatch() → warning_updated with retry text."""
        from src.card.delivery.engine import CardDelivery
        from src.card.session import CardSession

        class _Client:
            def create_card(self, *a, **kw):
                return ("msg_1", "card_1")
            def update_card(self, *a, **kw):
                pass
            def update_element(self, *a, **kw):
                pass

        client = _Client()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Test", engine_type="deep"))
        session = CardSession(
            chat_id="chat_retry",
            config=config,
            delivery=delivery,
            session_id="retry_sess",
        )

        # Bootstrap session
        session.dispatch(CardEvent.started())
        assert (session.state.footer.warning_banner or "") == ""

        # Flag retry pending via tracker
        session._coordinator._tracker.flag_retry_pending()

        # Next dispatch should inject the retry pending banner
        session.dispatch(CardEvent.text_started("b1"))
        assert "更新中" in (session.state.footer.warning_banner or "")

    def test_retry_banner_consumed_only_once(self):
        """After consuming SHOW_RETRY_PENDING, subsequent dispatches have no banner."""
        from src.card.delivery.engine import CardDelivery
        from src.card.session import CardSession

        class _Client:
            def create_card(self, *a, **kw):
                return ("msg_1", "card_1")
            def update_card(self, *a, **kw):
                pass
            def update_element(self, *a, **kw):
                pass

        client = _Client()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(mode_name="Test", engine_type="deep"))
        session = CardSession(
            chat_id="chat_retry2",
            config=config,
            delivery=delivery,
            session_id="retry_sess2",
        )

        session.dispatch(CardEvent.started())
        session._coordinator._tracker.flag_retry_pending()
        session.dispatch(CardEvent.text_started("b1"))
        # Banner should be set
        assert (session.state.footer.warning_banner or "") != ""

        # Next dispatch should NOT re-inject the banner (consumed once)
        session.dispatch(CardEvent.text_delta("b1", "hello"))
        # warning_text remains from the previous reduce (warning_updated is sticky),
        # but no NEW pending action should have been consumed
        # Verify by checking tracker has no pending actions
        actions = session._coordinator._tracker.consume_pending_actions()
        assert len(actions) == 0
