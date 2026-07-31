"""Tests for worktree reducer and event dispatch flow."""
import json

from src.card.events import CardEvent, CardEventType
from src.card.events.worktree import (
    worktree_cleanup,
    worktree_completed_no_change,
    worktree_confirm,
    worktree_merge,
    worktree_progress,
    worktree_tool_select,
)
from src.card.render.footer import render_footer
from src.card.state.button_intent import ButtonIntent
from src.card.state.models import CardMetadata, CardState
from src.card.state.reducer import reduce_card_state
from src.card.state.reducers.worktree import reduce_worktree


def _base_state(**kwargs) -> CardState:
    return CardState(**kwargs)


def _parse_block_data(state: CardState, index: int = 0) -> dict:
    """Get structured data from a content block (prefers .data, falls back to JSON)."""
    block = state.blocks[index]
    if block.data is not None:
        return block.data
    return json.loads(block.content)


class TestReduceWorktreeToolSelect:
    def test_tool_select_stores_structured_data(self):
        state = _base_state()
        event = worktree_tool_select(
            tools=[
                {"id": "coco", "name": "Coco", "description": "AI assistant"},
                {"id": "bash", "name": "Bash", "description": "Shell executor"},
            ],
            selected=["coco"],
            message="请选择工具",
        )
        new = reduce_worktree(state, event)
        assert len(new.blocks) == 1
        assert new.blocks[0].kind == "worktree_tool_select"
        data = _parse_block_data(new)
        assert len(data["tools"]) == 2
        assert data["tools"][0]["name"] == "Coco"
        assert data["tools"][1]["name"] == "Bash"
        assert data["selected"] == ["coco"]
        assert data["message"] == "请选择工具"

    def test_tool_select_reducer_emits_no_buttons(self):
        """Reducer 不再产出 footer 按钮——render 层全权负责确认/移除/清空。"""
        state = _base_state()
        event = worktree_tool_select(
            tools=[{"id": "coco", "name": "Coco"}],
            selected=["coco"],
        )
        new = reduce_worktree(state, event)
        assert new.buttons == ()

    def test_tool_select_empty_selection_no_confirm_button(self):
        state = _base_state()
        event = worktree_tool_select(
            tools=[{"id": "coco", "name": "Coco"}],
            selected=[],
        )
        new = reduce_worktree(state, event)
        assert len(new.buttons) == 0


class TestReduceWorktreeConfirm:
    def test_confirm_stores_structured_data(self):
        state = _base_state()
        event = worktree_confirm(
            selected_items=[
                {"tool": "Coco", "model": "claude-sonnet"},
                {"tool": "Bash", "model": "none"},
            ],
            goal="实现搜索功能",
        )
        new = reduce_worktree(state, event)
        assert new.blocks[0].kind == "worktree_confirm"
        data = _parse_block_data(new)
        assert data["selected_items"][0]["tool"] == "Coco"
        assert data["goal"] == "实现搜索功能"
        assert not any(b.action_id == ButtonIntent.WORKTREE_CONFIRM_START for b in new.buttons)
        assert any(b.action_id == ButtonIntent.WORKTREE_SHOW_MENU for b in new.buttons)


class TestReduceWorktreeProgress:
    def test_progress_stores_unit_data(self):
        state = _base_state()
        units = [
            {"name": "Unit A", "status": "completed", "summary": "All good"},
            {"name": "Unit B", "status": "running"},
            {"name": "Unit C", "status": "failed"},
        ]
        event = worktree_progress(units, project_id="p1")
        new = reduce_worktree(state, event)
        assert new.blocks[0].kind == "worktree_units"
        data = _parse_block_data(new)
        assert len(data["units"]) == 3
        assert data["units"][0]["name"] == "Unit A"
        assert data["units"][0]["status"] == "completed"
        assert data["completed"] == 1
        assert data["total"] == 3

    def test_progress_records_iteration_for_header_context(self):
        state = _base_state(metadata=CardMetadata(engine_type="worktree", mode_name="Worktree"))
        event = worktree_progress([{"name": "Unit A", "status": "running"}], project_id="p1", iteration=2)

        new = reduce_worktree(state, event)

        assert new.metadata.iteration_index == 2
        assert "第 2 轮" in new.header.title

    def test_progress_with_failed_shows_retry_button(self):
        state = _base_state()
        units = [{"name": "U1", "status": "failed"}]
        event = worktree_progress(units)
        new = reduce_worktree(state, event)
        assert any(b.action_id == ButtonIntent.WORKTREE_RETRY_FAILED for b in new.buttons)

    def test_progress_without_failed_no_retry_button(self):
        state = _base_state()
        units = [{"name": "U1", "status": "completed"}]
        event = worktree_progress(units)
        new = reduce_worktree(state, event)
        # All completed with no failures → show retry_all + cancel buttons
        assert len(new.buttons) == 2
        assert new.buttons[0].action_id == ButtonIntent.WORKTREE_RETRY_ALL
        assert new.buttons[1].action_id == ButtonIntent.WORKTREE_CANCEL

    def test_progress_bar_calculation(self):
        state = _base_state()
        units = [
            {"name": "A", "status": "completed"},
            {"name": "B", "status": "completed"},
            {"name": "C", "status": "running"},
            {"name": "D", "status": "pending"},
        ]
        event = worktree_progress(units)
        new = reduce_worktree(state, event)
        assert new.footer.progress_pct == 50
        assert "2/4" in new.footer.progress

    def test_progress_empty_units_no_error(self):
        """Empty units list should not raise and should produce valid state."""
        state = _base_state()
        event = worktree_progress([], project_id="p1")
        new = reduce_worktree(state, event)
        assert new.blocks[0].kind == "worktree_units"
        data = _parse_block_data(new)
        assert data["units"] == []
        assert data["total"] == 0
        # progress_pct should be None or 0 (no division by zero)
        assert new.footer.progress_pct is None or new.footer.progress_pct == 0
        # Empty units must NOT produce "finishing" status_text
        assert new.footer.status_text is None or "收尾" not in new.footer.status_text


class TestReduceWorktreeMerge:
    def test_merge_stores_structured_data(self):
        state = _base_state()
        merge_notes = [
            {"branch": "feat/search", "status": "ready", "summary": "Search feature"},
            {"branch": "feat/auth", "status": "conflict", "summary": "Auth module"},
        ]
        event = worktree_merge(merge_notes, base_branch="develop")
        new = reduce_worktree(state, event)
        assert new.blocks[0].kind == "worktree_merge"
        data = _parse_block_data(new)
        assert data["base_branch"] == "develop"
        assert data["merge_notes"][0]["branch"] == "feat/search"
        assert data["merge_notes"][1]["status"] == "conflict"
        assert any(b.action_id == ButtonIntent.WORKTREE_MERGE for b in new.buttons)

    def test_merge_empty_notes_no_error(self):
        """Empty merge_notes list should not raise and should produce valid state."""
        state = _base_state()
        event = worktree_merge([], base_branch="main")
        new = reduce_worktree(state, event)
        assert new.blocks[0].kind == "worktree_merge"
        data = _parse_block_data(new)
        assert data["merge_notes"] == []
        assert data["base_branch"] == "main"


class TestReduceWorktreeCleanup:
    def test_cleanup_stores_structured_data(self):
        state = _base_state()
        merge_notes = [{"branch": "feat/a", "status": "merged"}]
        merge_results = [{"branch": "feat/a", "success": True}]
        event = worktree_cleanup(
            merge_notes, merge_results=merge_results, project_id="p1",
            cleanup_phase="actions",
        )
        new = reduce_worktree(state, event)
        assert new.blocks[0].kind == "worktree_cleanup"
        data = _parse_block_data(new)
        assert data["merge_notes"][0]["branch"] == "feat/a"
        assert data["merge_results"][0]["success"] is True
        assert any(b.action_id == ButtonIntent.WORKTREE_CLEANUP for b in new.buttons)

    def test_cleanup_with_failed_merge_shows_retry(self):
        state = _base_state()
        merge_notes = [{"branch": "feat/b", "status": "conflict"}]
        merge_results = [{"branch": "feat/b", "success": False}]
        event = worktree_cleanup(
            merge_notes, merge_results=merge_results,
        )
        new = reduce_worktree(state, event)
        assert any(b.action_id == ButtonIntent.WORKTREE_RETRY_FAILED for b in new.buttons)

    def test_cleanup_summary_phase_shows_merge_and_cancel_buttons(self):
        """cleanup_phase='summary' produces merge + cancel button group."""
        state = _base_state()
        merge_notes = [{"branch": "feat/x", "status": "ready"}]
        merge_results = [{"branch": "feat/x", "success": True}]
        event = worktree_cleanup(
            merge_notes, merge_results=merge_results,
            cleanup_phase="summary",
        )
        new = reduce_worktree(state, event)
        action_ids = [b.action_id for b in new.buttons]
        assert ButtonIntent.WORKTREE_MERGE in action_ids
        assert ButtonIntent.WORKTREE_CANCEL in action_ids
        # No retry button since all succeeded
        assert ButtonIntent.WORKTREE_RETRY_FAILED not in action_ids

    def test_cleanup_summary_phase_with_failures_shows_retry(self):
        """cleanup_phase='summary' with failed results shows retry button."""
        state = _base_state()
        merge_notes = [{"branch": "feat/y", "status": "conflict"}]
        merge_results = [{"branch": "feat/y", "success": False}]
        event = worktree_cleanup(
            merge_notes, merge_results=merge_results,
            cleanup_phase="summary",
        )
        new = reduce_worktree(state, event)
        action_ids = [b.action_id for b in new.buttons]
        assert ButtonIntent.WORKTREE_MERGE in action_ids
        assert ButtonIntent.WORKTREE_CANCEL in action_ids
        assert ButtonIntent.WORKTREE_RETRY_FAILED in action_ids

    def test_cleanup_completed_phase_has_no_buttons(self):
        """cleanup_phase='completed' is terminal after auto merge and cleanup."""
        state = _base_state()
        event = worktree_cleanup(
            [{"branch": "feat/z", "status": "ready"}],
            merge_results=[{"branch": "feat/z", "success": True}],
            cleanup_phase="completed",
        )
        new = reduce_worktree(state, event)
        assert new.blocks[0].kind == "worktree_cleanup"
        assert new.buttons == ()


class TestWorktreeReducerInMainPipeline:
    """Verify worktree events route correctly through main reducer."""

    def test_worktree_progress_via_main_reducer(self):
        meta = CardMetadata(engine_type="worktree")
        state = reduce_card_state(None, CardEvent.started(), meta)
        state = reduce_card_state(
            state,
            worktree_progress(
                [{"name": "U1", "status": "running"}], project_id="p1",
            ),
        )
        assert len(state.blocks) == 1
        assert state.blocks[0].kind == "worktree_units"
        data = _parse_block_data(state)
        assert data["units"][0]["name"] == "U1"

    def test_unrelated_event_passthrough(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "x", "text": "y"})
        new = reduce_worktree(state, event)
        assert new is state


# ---------------------------------------------------------------------------
# Phase 5: worktree_cancel action test
# ---------------------------------------------------------------------------
from src.card.delivery.engine import CardDelivery
from src.card.session import CardSession
from src.card.session.config import SessionConfig


class _MockClient:
    def __init__(self):
        self.creates = []
        self._counter = 0

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._counter += 1
        self.creates.append(card_json)
        return (f"msg_{self._counter}", f"card_{self._counter}")

    def update_card(self, card_id, card_json, *, sequence=0):
        pass

    def update_element(self, card_id, element_id, content, *, sequence=0):
        pass


class TestWorktreeCancelAction:
    """worktree_cancel triggers CANCELLED and closes session."""

    def test_cancel_action_closes_session(self):
        from src.card.actions.dispatch import build_worktree_action_registry

        registry = build_worktree_action_registry()
        client = _MockClient()
        delivery = CardDelivery(client)
        config = SessionConfig(metadata=CardMetadata(engine_type="worktree"))
        session = CardSession(
            chat_id="chat_1",
            config=config,
            delivery=delivery,
            session_id="wt_test",
            action_registry=registry,
        )
        # Start session first
        session.dispatch(CardEvent(type=CardEventType.STARTED))
        assert not session.closed

        # Trigger cancel
        result = session.inbound_action("worktree_cancel")
        assert result is not None  # successful dispatch returns ack toast
        assert result["toast"]["type"] == "info"
        assert session.closed  # CANCELLED is a terminal event


# ---------------------------------------------------------------------------
# Phase 6: WorktreeRuntimeState round-trip & reporter end-to-end (AC27, AC28)
# ---------------------------------------------------------------------------

from src.card.render.worktree import _render_worktree_merge
from src.worktree_engine.models import WorktreeRuntimeState, WorktreeUnit, WorktreeUnitStatus
from src.worktree_engine.reporter import WorktreeReporter


class TestWorktreeStateRoundTrip:
    """AC27: to_dict() → from_dict() round-trip preserves merge_notes."""

    def test_roundtrip_preserves_merge_notes(self):
        state = WorktreeRuntimeState()
        state.iteration_count = 2
        state.merge_notes = [
            {"branch": "feature/a", "status": "ready", "summary": "Feature A"},
            {"branch": "feature/b", "status": "conflict", "summary": "Feature B"},
        ]
        data = state.to_dict()
        restored = WorktreeRuntimeState.from_dict(data)
        assert restored.iteration_count == 2
        assert len(restored.merge_notes) == 2
        assert restored.merge_notes[0]["branch"] == "feature/a"
        assert restored.merge_notes[0]["summary"] == "Feature A"
        assert restored.merge_notes[1]["summary"] == "Feature B"

    def test_legacy_description_key_migrated(self):
        """AC27: old 'description' key is migrated to 'summary' on deserialization."""
        data = {
            "merge_notes": [
                {"branch": "b1", "status": "ready", "description": "old desc"},
            ]
        }
        state = WorktreeRuntimeState.from_dict(data)
        assert state.merge_notes[0]["summary"] == "old desc"
        assert "description" not in state.merge_notes[0]

    def test_legacy_string_list_migrated(self):
        """AC27: old list[str] format converted to dict."""
        data = {"merge_notes": ["bare note text", "another"]}
        state = WorktreeRuntimeState.from_dict(data)
        assert state.merge_notes[0]["summary"] == "bare note text"
        assert state.merge_notes[0]["branch"] == ""

    def test_invalid_iteration_count_falls_back_to_zero(self):
        state = WorktreeRuntimeState.from_dict({"iteration_count": "not-a-number"})

        assert state.iteration_count == 0


class TestReporterSummaryEndToEnd:
    """AC28: reporter summary field renders in final card output."""

    def test_merge_notes_summary_renders_in_card(self):
        """reporter.build_merge_notes → _render_worktree_merge → markdown contains summary text."""
        unit = WorktreeUnit(unit_id="unit-01")
        unit.display_name = "工作空间 A"
        unit.branch_name = "ghostap/wt/01-feature"
        unit.status = WorktreeUnitStatus.COMPLETED

        notes = WorktreeReporter().build_merge_notes([unit], "main")
        assert "summary" in notes[0]
        assert "ghostap/wt/01-feature" in notes[0]["summary"]

        # Now render
        data = {"merge_notes": notes, "base_branch": "main"}
        result = _render_worktree_merge(data)
        content = result["content"]
        assert "ghostap/wt/01-feature" in content
        # The summary text should appear after the branch name
        assert "工作空间 A" in content


# ---------------------------------------------------------------------------
# WORKTREE_COMPLETED_NO_CHANGE event tests
# ---------------------------------------------------------------------------


class TestReduceWorktreeCompletedNoChange:
    """Test the WORKTREE_COMPLETED_NO_CHANGE event and its reducer handler."""

    def test_completed_no_change_freezes_total_elapsed(self, monkeypatch):
        metadata = CardMetadata(
            engine_type="worktree",
            session_started_at=100.0,
        )
        state = reduce_card_state(None, CardEvent.started(), metadata)
        event = worktree_completed_no_change([], message="无变更")
        event = CardEvent(
            type=event.type,
            payload={**event.payload, "_now": 158.0},
        )

        state = reduce_card_state(state, event)
        monkeypatch.setattr(
            "src.card.render.footer.time.monotonic",
            lambda: 999.0,
        )
        content = "\n".join(
            element.get("content", "")
            for element in render_footer(state)
        )

        assert state.footer.duration_seconds == 58.0
        assert "⏱ 用时 00:00:58" in content

    def test_progress_resumes_completed_no_change_state(self):
        metadata = CardMetadata(
            engine_type="worktree",
            session_started_at=100.0,
        )
        state = reduce_card_state(None, CardEvent.started(), metadata)
        completed = worktree_completed_no_change([], message="无变更")
        state = reduce_card_state(
            state,
            CardEvent(
                type=completed.type,
                payload={**completed.payload, "_now": 158.0},
            ),
        )

        state = reduce_card_state(
            state,
            worktree_progress([{"name": "U1", "status": "running"}]),
        )

        assert state.terminal == "running"
        assert state.footer.duration_seconds is None

    def test_event_factory_creates_correct_type(self):
        event = worktree_completed_no_change(
            [{"name": "U1", "status": "completed"}], project_id="p1", message="No changes"
        )
        assert event.type == CardEventType.WORKTREE_COMPLETED_NO_CHANGE
        assert event.payload["units"][0]["name"] == "U1"
        assert event.payload["message"] == "No changes"

    def test_reducer_produces_completed_terminal(self):
        state = _base_state()
        event = worktree_completed_no_change(
            [{"name": "U1", "status": "completed"}], message="无变更"
        )
        new = reduce_worktree(state, event)
        assert new.terminal == "completed_empty"
        assert new.blocks[0].kind == "worktree_units"
        assert new.blocks[0].block_id == "worktree_no_change"

    def test_reducer_shows_retry_and_cancel_buttons(self):
        state = _base_state()
        event = worktree_completed_no_change(
            [{"name": "U1", "status": "completed"}]
        )
        new = reduce_worktree(state, event)
        action_ids = [b.action_id for b in new.buttons]
        assert ButtonIntent.WORKTREE_RETRY_ALL in action_ids
        assert ButtonIntent.WORKTREE_CANCEL in action_ids

    def test_reducer_uses_default_message_when_empty(self):
        state = _base_state()
        event = worktree_completed_no_change([], message="")
        new = reduce_worktree(state, event)
        # Should use the UI_TEXT default
        assert new.footer.status_text != ""

    def test_header_uses_wathet_template(self):
        state = _base_state()
        event = worktree_completed_no_change(
            [{"name": "U1", "status": "completed"}]
        )
        new = reduce_worktree(state, event)
        assert new.header.template == "wathet"
        # Header title reuses programming mode title; subtitle shows no-change message
        assert new.header.subtitle is not None


class TestWorktreeProgressSilentBranch:
    """Test that silent=True progress events use worktree_footer_silent text."""

    def test_silent_progress_uses_silent_footer(self):
        from src.card.events.worktree import worktree_progress
        from src.card.ui_text import UI_TEXT

        state = _base_state()
        event = worktree_progress(
            units=[
                {"name": "coco", "status": "running"},
                {"name": "bash", "status": "pending"},
            ],
            silent=True,
        )
        new = reduce_worktree(state, event)
        assert new.footer is not None
        assert new.footer.status_text == UI_TEXT["worktree_footer_silent"]

    def test_non_silent_progress_has_no_silent_footer(self):
        from src.card.events.worktree import worktree_progress
        from src.card.ui_text import UI_TEXT

        state = _base_state()
        event = worktree_progress(
            units=[
                {"name": "coco", "status": "running"},
                {"name": "bash", "status": "pending"},
            ],
            silent=False,
        )
        new = reduce_worktree(state, event)
        assert new.footer is not None
        # Non-silent should NOT use silent footer text
        assert new.footer.status_text != UI_TEXT["worktree_footer_silent"]

    def test_via_main_reducer_pipeline(self):
        """Verify routing through main reduce_card_state."""
        meta = CardMetadata(engine_type="worktree")
        state = reduce_card_state(None, CardEvent.started(), meta)
        state = reduce_card_state(
            state,
            worktree_completed_no_change(
                [{"name": "U1", "status": "completed"}], message="无变更"
            ),
        )
        assert state.terminal == "completed_empty"
        assert state.blocks[0].block_id == "worktree_no_change"

    def test_no_change_records_iteration_for_header_context(self):
        meta = CardMetadata(engine_type="worktree", mode_name="Worktree")
        state = reduce_card_state(None, CardEvent.started(), meta)
        state = reduce_card_state(
            state,
            worktree_completed_no_change(
                [{"name": "U1", "status": "completed"}], message="无变更", iteration=3
            ),
        )

        assert state.metadata.iteration_index == 3
        assert "第 3 轮" in state.header.title


class TestRetryButtonConfirmText:
    """Verify retry buttons carry the correct confirm text per retry mode."""

    def test_retry_failed_button_confirm_text(self):
        """When there are failed units, retry_failed button uses wt_btn_confirm_retry."""
        from src.card.ui_text import UI_TEXT

        state = _base_state()
        # Start with a progress event containing failures
        event = worktree_progress(
            units=[
                {"name": "coco", "status": "completed"},
                {"name": "bash", "status": "failed"},
            ],
            message="执行中",
        )
        new = reduce_worktree(state, event)
        retry_btns = [b for b in new.buttons if b.action_id == ButtonIntent.WORKTREE_RETRY_FAILED]
        assert len(retry_btns) == 1
        assert retry_btns[0].confirm == UI_TEXT["wt_btn_confirm_retry"]

    def test_retry_all_button_confirm_text(self):
        """When all units complete with no failures, retry_all button uses wt_btn_confirm_retry_all."""
        from src.card.ui_text import UI_TEXT

        state = _base_state()
        # All completed, no failures → retry_all
        event = worktree_progress(
            units=[
                {"name": "coco", "status": "completed"},
                {"name": "bash", "status": "completed"},
            ],
            message="完成",
        )
        new = reduce_worktree(state, event)
        retry_btns = [b for b in new.buttons if b.action_id == ButtonIntent.WORKTREE_RETRY_ALL]
        assert len(retry_btns) == 1
        assert retry_btns[0].confirm == UI_TEXT["wt_btn_confirm_retry_all"]
