"""Unit tests for card state reducers: criteria, cycle, phase, lifecycle."""
from dataclasses import replace

import pytest

from src.card.events import CardEvent, CardEventType
from src.card.state.button_intent import ButtonIntent
from src.card.state.models import (
    ButtonSpec,
    CardMetadata,
    CardState,
    ContentBlock,
    EngineExtState,
    FooterState,
)
from src.card.state.reducer import reduce_card_state
from src.card.state.reducers.criteria import reduce_criteria
from src.card.state.reducers.cycle import reduce_cycle
from src.card.state.reducers.lifecycle import reduce_lifecycle
from src.card.state.reducers.phase import reduce_phase


def _base_state(**kwargs) -> CardState:
    """Create a base state with sensible defaults for testing."""
    if "engine_ext" not in kwargs:
        kwargs["engine_ext"] = EngineExtState()
    return CardState(**kwargs)


# ==============================================================================
# reduce_criteria tests
# ==============================================================================


class TestReduceCriteria:
    def test_criteria_updated_sets_content_and_counts(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.CRITERIA_UPDATED, payload={
            "content": "## Criteria\n- [x] done\n- [ ] pending",
            "satisfied_count": 1,
            "total_count": 2,
        })
        new = reduce_criteria(state, event)
        assert new.engine_ext.criteria_section == "## Criteria\n- [x] done\n- [ ] pending"
        assert new.engine_ext.criteria_satisfied == 1
        assert new.engine_ext.criteria_total == 2

    def test_criteria_updated_preserves_existing_counts_if_not_provided(self):
        ext = EngineExtState(criteria_satisfied=3, criteria_total=5)
        state = _base_state(engine_ext=ext)
        event = CardEvent(type=CardEventType.CRITERIA_UPDATED, payload={"content": "updated"})
        new = reduce_criteria(state, event)
        assert new.engine_ext.criteria_section == "updated"
        assert new.engine_ext.criteria_satisfied == 3
        assert new.engine_ext.criteria_total == 5

    def test_warning_updated_sets_and_clears_banner(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.WARNING_UPDATED, payload={"warning": "⚠️ Token limit approaching"})
        new = reduce_criteria(state, event)
        assert new.footer.warning_banner == "⚠️ Token limit approaching"

        # Clear with empty
        state2 = _base_state(footer=FooterState(warning_banner="old warning"))
        event2 = CardEvent(type=CardEventType.WARNING_UPDATED, payload={"warning": ""})
        new2 = reduce_criteria(state2, event2)
        assert new2.footer.warning_banner is None

    def test_review_retry_waiting_status(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.REVIEW_RETRY, payload={
            "attempt": 2, "max_attempts": 3, "status": "waiting", "delay_sec": 10,
        })
        new = reduce_criteria(state, event)
        assert "等待 10 秒" in new.footer.status_text
        assert "2/3" in new.footer.status_text
        assert any(b.action_id == ButtonIntent.SPEC_STOP for b in new.buttons)

    def test_review_retry_exhausted_shows_resume_button(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.REVIEW_RETRY, payload={
            "attempt": 3, "max_attempts": 3, "status": "exhausted",
        })
        new = reduce_criteria(state, event)
        assert "重试已耗尽" in new.footer.status_text
        assert any(b.action_id == ButtonIntent.SPEC_RESUME for b in new.buttons)

    def test_review_retry_delay_sec_none_no_crash(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.REVIEW_RETRY, payload={
            "attempt": 1, "max_attempts": 2, "status": "waiting", "delay_sec": None,
        })
        new = reduce_criteria(state, event)
        assert "等待 0 秒" in new.footer.status_text

    def test_unrelated_event_returns_state_unchanged(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.STARTED)
        assert reduce_criteria(state, event) is state


# ==============================================================================
# reduce_cycle tests
# ==============================================================================


class TestReduceCycle:
    def test_cycle_started_sets_cycle_num_and_footer(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.CYCLE_STARTED, payload={"cycle_num": 2, "max_cycles": 5})
        new = reduce_cycle(state, event)
        assert new.engine_ext.cycle_num == 2
        assert new.engine_ext.max_cycles == 5
        assert "2/5" in new.footer.status_text
        assert new.terminal == "running"

    def test_cycle_done_clears_footer(self):
        ext = EngineExtState(cycle_num=2, max_cycles=5)
        state = _base_state(
            engine_ext=ext,
            footer=FooterState(status="tool_running", status_text="⏳ 迭代 2/5"),
        )
        event = CardEvent(type=CardEventType.CYCLE_DONE, payload={"cycle_num": 2})
        new = reduce_cycle(state, event)
        assert new.footer.status == "idle"
        assert new.footer.status_text is None

    def test_unrelated_event_returns_state_unchanged(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.COMPLETED)
        assert reduce_cycle(state, event) is state


# ==============================================================================
# reduce_phase tests
# ==============================================================================


class TestReducePhase:
    def test_phase_started_adds_block(self):
        ext = EngineExtState(cycle_num=1)
        state = _base_state(engine_ext=ext)
        event = CardEvent(type=CardEventType.PHASE_STARTED, payload={"cycle_num": 1, "phase": "planning"})
        new = reduce_phase(state, event)
        assert len(new.blocks) == 1
        assert new.blocks[0].block_id == "phase_1_planning"
        assert new.blocks[0].status == "active"
        assert new.engine_ext.phase_info == "planning"

    def test_phase_started_idempotent_replaces_existing(self):
        ext = EngineExtState(cycle_num=1)
        block = ContentBlock(
            kind="phase", block_id="phase_1_planning",
            content="planning", status="active", phase_name="planning", cycle_num=1,
        )
        state = _base_state(engine_ext=ext, blocks=(block,))
        event = CardEvent(type=CardEventType.PHASE_STARTED, payload={"cycle_num": 1, "phase": "planning"})
        new = reduce_phase(state, event)
        matching = [b for b in new.blocks if b.block_id == "phase_1_planning"]
        assert len(matching) == 1
        assert matching[0].status == "active"

    def test_phase_done_marks_completed(self):
        ext = EngineExtState(cycle_num=1, phase_info="planning")
        block = ContentBlock(
            kind="phase", block_id="phase_1_planning",
            content="planning", status="active", phase_name="planning", cycle_num=1,
        )
        state = _base_state(engine_ext=ext, blocks=(block,))
        event = CardEvent(type=CardEventType.PHASE_DONE, payload={
            "cycle_num": 1, "phase": "planning", "output": "Plan created",
        })
        new = reduce_phase(state, event)
        assert new.blocks[0].status == "completed"
        assert new.blocks[0].content == "Plan created"
        assert new.engine_ext.phase_info is None

    def test_phase_done_without_prior_start_returns_unchanged(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.PHASE_DONE, payload={"cycle_num": 1, "phase": "unknown"})
        new = reduce_phase(state, event)
        assert new is state

    def test_unrelated_event_returns_state_unchanged(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.STARTED)
        assert reduce_phase(state, event) is state


class TestRuntimeStats:
    def test_spec_runtime_stats_track_cycle_phase_and_elapsed(self):
        metadata = CardMetadata(mode_name="Spec", mode_emoji="📋", engine_type="spec", session_started_at=100.0)
        state = reduce_card_state(
            None, CardEvent(type=CardEventType.STARTED, payload={"_now": 101.0}), metadata=metadata,
        )
        state = reduce_card_state(
            state, CardEvent(type=CardEventType.CYCLE_STARTED, payload={"cycle_num": 2, "max_cycles": 500, "_now": 110.0}),
        )
        state = reduce_card_state(
            state, CardEvent(type=CardEventType.PHASE_STARTED, payload={"cycle_num": 2, "phase": "review", "_now": 125.0}),
        )
        assert state.runtime_stats.spec_cycle == 2
        assert state.runtime_stats.spec_perspective == "review"
        assert state.runtime_stats.elapsed_seconds == 25.0


# ==============================================================================
# reduce_lifecycle tests
# ==============================================================================


class TestReduceLifecycle:
    def test_started_sets_running(self):
        state = _base_state()
        event = CardEvent.started()
        new = reduce_lifecycle(state, event)
        assert new.terminal == "running"
        assert new.footer.status == "thinking"

    def test_completed_sets_terminal_and_clears_buttons(self):
        state = _base_state(buttons=(ButtonSpec(text="x", action_id="y"),))
        event = CardEvent.completed()
        new = reduce_lifecycle(state, event)
        assert new.terminal == "completed"
        assert new.buttons == ()

    def test_completed_with_summary_appends_block(self):
        state = _base_state()
        event = CardEvent.completed(summary="Done: 5 tools called")
        new = reduce_lifecycle(state, event)
        assert new.terminal == "completed"
        summary_blocks = [b for b in new.blocks if b.block_id == "_summary"]
        assert len(summary_blocks) == 1
        assert "5 tools called" in summary_blocks[0].content

    def test_failed_inserts_error_block(self):
        state = _base_state()
        event = CardEvent.failed("Connection timeout")
        new = reduce_lifecycle(state, event)
        assert new.terminal == "failed"
        error_blocks = [b for b in new.blocks if b.block_id == "_error"]
        assert len(error_blocks) == 1
        assert "Connection timeout" in error_blocks[0].content

    @pytest.mark.parametrize("engine_type,expected_btn", [
        ("deep", ButtonIntent.DEEP_RESUME),
        ("spec", ButtonIntent.SPEC_RESUME),
    ])
    def test_failed_injects_retry_button(self, engine_type, expected_btn):
        meta = CardMetadata(engine_type=engine_type)
        state = _base_state(metadata=meta)
        event = CardEvent.failed("error")
        new = reduce_lifecycle(state, event)
        assert any(b.action_id == expected_btn for b in new.buttons)

    def test_failed_no_retry_for_unknown_engine(self):
        meta = CardMetadata(engine_type="unknown")
        state = _base_state(metadata=meta)
        event = CardEvent.failed("error")
        new = reduce_lifecycle(state, event)
        assert not any(b.action_id in (ButtonIntent.DEEP_RESUME, ButtonIntent.SPEC_RESUME, ButtonIntent.WORKTREE_RETRY_FAILED) for b in new.buttons)
        assert any(b.action_id == ButtonIntent.SHOW_STATUS for b in new.buttons)

    def test_cancelled_clears_buttons(self):
        state = _base_state(buttons=(ButtonSpec(text="x", action_id="y"),))
        event = CardEvent.cancelled()
        new = reduce_lifecycle(state, event)
        assert new.terminal == "cancelled"
        assert new.buttons == ()

    def test_archived_can_freeze_full_card_without_appending_content(self):
        from src.card.state.models import TextBlock
        from src.card.state.reducer import MAX_TOTAL_BLOCKS

        state = _base_state(blocks=tuple(
            TextBlock(block_id=f"history-{index}", content=str(index))
            for index in range(MAX_TOTAL_BLOCKS)
        ))

        archived = reduce_lifecycle(
            state,
            CardEvent.archived(
                sequence=1,
                new_message_id="next-message",
                append_hint=False,
            ),
        )

        assert archived.metadata.frozen is True
        assert len(archived.blocks) == MAX_TOTAL_BLOCKS
        assert archived.blocks[0].block_id == "history-0"
        assert not any(block.block_id == "_archived_hint" for block in archived.blocks)
        assert archived.buttons[0].url.endswith("next-message")

    def test_unrelated_event_returns_state_unchanged(self):
        state = _base_state()
        event = CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "b", "text": "t"})
        assert reduce_lifecycle(state, event) is state


# ==============================================================================
# Stop button injection tests (P4)
# ==============================================================================


class TestLifecycleStopButtons:
    @pytest.mark.parametrize("engine_type,expected_action", [
        ("deep", ButtonIntent.ENGINE_STOP),
        ("spec", ButtonIntent.ENGINE_STOP),
        ("worktree", ButtonIntent.WORKTREE_CANCEL),
    ])
    def test_started_injects_stop_button(self, engine_type, expected_action):
        state = _base_state(metadata=CardMetadata(engine_type=engine_type))
        event = CardEvent.started()
        new = reduce_lifecycle(state, event)
        assert len(new.buttons) == 2
        assert new.buttons[0].action_id == ButtonIntent.MODE_COMPACT
        assert new.buttons[1].action_id == expected_action

    def test_started_no_buttons_for_none_engine(self):
        state = _base_state(metadata=CardMetadata(engine_type=None))
        event = CardEvent.started()
        new = reduce_lifecycle(state, event)
        assert new.buttons == ()

    def test_resumed_injects_stop_button(self):
        """RESUMED injects stop button for engine types (representative: deep)."""
        state = _base_state(metadata=CardMetadata(engine_type="deep"))
        event = CardEvent(type=CardEventType.RESUMED)
        new = reduce_lifecycle(state, event)
        assert len(new.buttons) == 2
        assert new.buttons[1].type == "default"


# ==============================================================================
# Warning banner semantic type tests (P2)
# ==============================================================================


class TestWarningBannerType:
    @pytest.mark.parametrize("text,expected_type", [
        ("✅ 完成", "success"),
        ("❌ 失败", "error"),
        ("普通文本", "warning"),
    ])
    def test_warning_type_inferred_from_prefix(self, text, expected_type):
        state = _base_state()
        event = CardEvent.warning_updated(text)
        new = reduce_criteria(state, event)
        assert new.footer.warning_type == expected_type

    def test_empty_warning_clears_type(self):
        state = _base_state(footer=FooterState(warning_banner="old", warning_type="error"))
        event = CardEvent.warning_updated("")
        new = reduce_criteria(state, event)
        assert new.footer.warning_banner is None
        assert new.footer.warning_type is None


# ==============================================================================
# ContentBlock factory parametrized tests
# ==============================================================================


class TestTextDeltaAutoCreate:
    def test_text_delta_auto_creates_block_and_sets_footer(self):
        from src.card.state.reducers.text import reduce_text
        state = _base_state()
        event = CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "new_block", "text": "hello"})
        new = reduce_text(state, event)
        assert len(new.blocks) == 1
        assert new.blocks[0].block_id == "new_block"
        assert new.blocks[0].content == "hello"
        assert new.blocks[0].kind == "text"
        assert new.footer.status == "thinking"
        assert "思考" in new.footer.status_text

    def test_text_delta_appends_to_existing_block(self):
        from src.card.state.models import TextBlock
        from src.card.state.reducers.text import reduce_text
        block = TextBlock(block_id="b1", content="hello ", status="active")
        state = _base_state(blocks=(block,))
        event = CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "b1", "text": "world"})
        new = reduce_text(state, event)
        assert new.blocks[0].content == "hello world"

    def test_text_delta_existing_does_not_update_footer(self):
        from src.card.state.models import TextBlock
        from src.card.state.reducers.text import reduce_text
        block = TextBlock(block_id="b1", content="x", status="active")
        state = _base_state(blocks=(block,), footer=FooterState(status="tool_running", status_text="🔧 running"))
        event = CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "b1", "text": "y"})
        new = reduce_text(state, event)
        assert new.footer.status == "tool_running"


class TestContentBlockFactory:
    """ContentBlock() factory creates correct subclass for each kind."""

    @pytest.mark.parametrize("kind", ["text", "tool_call", "phase", "worktree_units"])
    def test_factory_creates_correct_kind(self, kind):
        block = ContentBlock(kind=kind, block_id="test_1")
        assert block.kind == kind

    def test_factory_unknown_kind_defaults_to_text(self):
        block = ContentBlock(kind="unknown_xyz")
        assert block.kind == "text"

    def test_factory_tool_block_accepts_tool_fields(self):
        block = ContentBlock(kind="tool_call", block_id="t1", tool_name="read_file")
        assert block.kind == "tool_call"
        assert block.tool_name == "read_file"


# ==============================================================================
# reduce_worktree tests
# ==============================================================================


class TestReduceWorktree:
    @pytest.fixture
    def wt_state(self):
        return _base_state(metadata=CardMetadata(engine_type="worktree"))

    def test_tool_select_sets_block_no_reducer_buttons(self, wt_state):
        from src.card.state.reducers.worktree import reduce_worktree
        event = CardEvent(type=CardEventType.WORKTREE_TOOL_SELECT, payload={
            "tools": [{"name": "coco"}, {"name": "claude"}],
            "selected": ["coco"], "message": "请选择工具",
        })
        new = reduce_worktree(wt_state, event)
        assert len(new.blocks) == 1
        assert new.blocks[0].kind == "worktree_tool_select"
        assert new.blocks[0].data["selected"] == ["coco"]
        assert new.buttons == ()

    def test_confirm_sets_block_and_action_buttons(self, wt_state):
        from src.card.state.reducers.worktree import reduce_worktree
        event = CardEvent(type=CardEventType.WORKTREE_CONFIRM, payload={
            "selected_items": [{"tool": "coco", "model": "gpt-4o"}],
            "goal": "Build feature X", "message": "确认配置",
        })
        new = reduce_worktree(wt_state, event)
        assert new.blocks[0].kind == "worktree_confirm"
        assert len(new.buttons) == 2
        action_ids = [b.action_id for b in new.buttons]
        assert ButtonIntent.WORKTREE_SHOW_MENU in action_ids
        assert ButtonIntent.WORKTREE_CANCEL in action_ids

    @pytest.mark.parametrize("units,expected_pct,has_retry", [
        ([{"name": "A", "status": "running"}], 0, False),
        ([{"name": "A", "status": "completed"}, {"name": "B", "status": "failed"}], 50, True),
        ([{"name": "A", "status": "completed"}, {"name": "B", "status": "completed"}], 100, True),
    ])
    def test_progress_buttons_and_pct(self, wt_state, units, expected_pct, has_retry):
        from src.card.state.reducers.worktree import reduce_worktree
        event = CardEvent(type=CardEventType.WORKTREE_PROGRESS, payload={"units": units, "message": "progress"})
        new = reduce_worktree(wt_state, event)
        assert new.blocks[0].kind == "worktree_units"
        assert new.footer.progress_pct == expected_pct
        has_retry_btn = any(
            b.action_id in (ButtonIntent.WORKTREE_RETRY_FAILED, ButtonIntent.WORKTREE_RETRY_ALL)
            for b in new.buttons
        )
        assert has_retry_btn == has_retry

    def test_merge_sets_block_and_merge_button(self, wt_state):
        from src.card.state.reducers.worktree import reduce_worktree
        event = CardEvent(type=CardEventType.WORKTREE_MERGE, payload={
            "merge_notes": [{"branch": "feat/a", "status": "ready"}], "base_branch": "main",
        })
        new = reduce_worktree(wt_state, event)
        assert new.blocks[0].kind == "worktree_merge"
        assert any(b.action_id == ButtonIntent.WORKTREE_MERGE for b in new.buttons)

    def test_cleanup_summary_and_actions_phases(self, wt_state):
        from src.card.state.reducers.worktree import reduce_worktree

        base_payload = {
            "merge_notes": [{"branch": "feat/a", "status": "merged"}],
            "base_branch": "main",
            "merge_results": [{"branch": "feat/a", "success": True}],
        }

        # Summary phase
        event_summary = CardEvent(type=CardEventType.WORKTREE_CLEANUP, payload={**base_payload, "cleanup_phase": "summary"})
        new_summary = reduce_worktree(wt_state, event_summary)
        assert new_summary.blocks[0].kind == "worktree_cleanup"
        action_ids = [b.action_id for b in new_summary.buttons]
        assert ButtonIntent.WORKTREE_MERGE in action_ids
        assert ButtonIntent.WORKTREE_CANCEL in action_ids

        # Actions phase
        event_actions = CardEvent(type=CardEventType.WORKTREE_CLEANUP, payload={**base_payload, "cleanup_phase": "actions"})
        new_actions = reduce_worktree(wt_state, event_actions)
        action_ids = [b.action_id for b in new_actions.buttons]
        assert ButtonIntent.WORKTREE_CLEANUP in action_ids
        cleanup_btn = next(b for b in new_actions.buttons if b.action_id == ButtonIntent.WORKTREE_CLEANUP)
        assert cleanup_btn.type == "danger"

    def test_unrelated_event_returns_unchanged(self, wt_state):
        from src.card.state.reducers.worktree import reduce_worktree
        event = CardEvent(type=CardEventType.STARTED)
        assert reduce_worktree(wt_state, event) is wt_state


# ==============================================================================
# engine_ext=None defense tests (AC7)
# ==============================================================================


class TestEngineExtNoneDefense:
    @pytest.mark.parametrize("reducer,event_type,payload", [
        (reduce_criteria, CardEventType.CRITERIA_UPDATED, {"content": "test", "satisfied_count": 1, "total_count": 2}),
        (reduce_phase, CardEventType.PHASE_STARTED, {"cycle_num": 1, "phase": "planning"}),
        (reduce_cycle, CardEventType.CYCLE_STARTED, {"cycle_num": 1, "max_cycles": 3}),
    ])
    def test_returns_state_when_engine_ext_none(self, reducer, event_type, payload):
        state = CardState()
        event = CardEvent(type=event_type, payload=payload)
        result = reducer(state, event)
        assert result is state

    def test_criteria_total_zero(self):
        ext = EngineExtState(criteria_satisfied=0, criteria_total=0)
        state = _base_state(engine_ext=ext)
        event = CardEvent(type=CardEventType.CRITERIA_UPDATED, payload={
            "content": "No criteria defined", "satisfied_count": 0, "total_count": 0,
        })
        new = reduce_criteria(state, event)
        assert new.engine_ext.criteria_total == 0

    def test_phase_display_name_unknown_phase(self):
        ext = EngineExtState(cycle_num=1)
        state = _base_state(engine_ext=ext)
        event = CardEvent(type=CardEventType.PHASE_STARTED, payload={"cycle_num": 1, "phase": "unknown_phase_xyz"})
        new = reduce_phase(state, event)
        assert "进行中" in new.footer.status_text


# ==============================================================================
# Sliding-window conditional gating tests
# ==============================================================================


class TestSlidingWindowGating:
    def _state_with_excess_completed_tools(self) -> CardState:
        from src.card.state.models import ToolBlock
        from src.card.state.reducer import MAX_COMPLETED_TOOL_BLOCKS

        blocks = []
        for i in range(MAX_COMPLETED_TOOL_BLOCKS + 5):
            blocks.append(ToolBlock(
                kind="tool_call", block_id=f"tool_{i}",
                content=f"tool {i}", status="completed",
            ))
        blocks.append(ContentBlock(kind="text", block_id="text_1", content="hello"))
        return _base_state(blocks=tuple(blocks))

    def test_text_delta_does_not_trigger_trim(self):
        from src.card.state.reducer import reduce_card_state
        state = self._state_with_excess_completed_tools()
        initial_block_count = len(state.blocks)
        event = CardEvent(type=CardEventType.TEXT_DELTA, payload={"block_id": "text_1", "delta": " world"})
        new_state = reduce_card_state(state, event)
        assert len(new_state.blocks) == initial_block_count

    @pytest.mark.parametrize("event_type", [CardEventType.TOOL_DONE, CardEventType.TOOL_FAILED])
    def test_tool_completion_triggers_trim(self, event_type):
        from src.card.state.models import ToolBlock
        from src.card.state.reducer import MAX_COMPLETED_TOOL_BLOCKS, reduce_card_state

        state = self._state_with_excess_completed_tools()
        running_tool = ToolBlock(kind="tool_call", block_id="tool_running", content="running", status="running")
        state = replace(state, blocks=state.blocks + (running_tool,))

        payload = {"block_id": "tool_running"}
        if event_type == CardEventType.TOOL_DONE:
            payload["tool_output"] = "done"
        else:
            payload["error"] = "timeout"

        event = CardEvent(type=event_type, payload=payload)
        new_state = reduce_card_state(state, event)
        completed = [b for b in new_state.blocks if b.kind == "tool_call" and b.status == "completed"]
        assert len(completed) <= MAX_COMPLETED_TOOL_BLOCKS


class TestCardStateContinuationProjection:
    def test_exact_total_block_boundary_does_not_rotate_early(self):
        from src.card.state import reducer as reducer_module
        from src.card.state.models import TextBlock

        state = _base_state(blocks=tuple(
            TextBlock(block_id=f"text-{index}", content=str(index))
            for index in range(reducer_module.MAX_TOTAL_BLOCKS - 1)
        ))

        assert reducer_module.card_state_requires_continuation(
            state,
            CardEvent.text_started("exact-boundary"),
        ) is False

    def test_total_block_projection_detects_trim_without_mutating_state(self):
        from src.card.state import reducer as reducer_module
        from src.card.state.models import TextBlock

        assert hasattr(reducer_module, "card_state_requires_continuation")
        state = _base_state(blocks=tuple(
            TextBlock(block_id=f"text-{index}", content=str(index))
            for index in range(reducer_module.MAX_TOTAL_BLOCKS)
        ))
        original_blocks = state.blocks

        requires_continuation = reducer_module.card_state_requires_continuation(
            state,
            CardEvent.text_started("overflow-text"),
        )

        assert requires_continuation is True
        assert state.blocks is original_blocks
        assert len(state.blocks) == reducer_module.MAX_TOTAL_BLOCKS

    def test_completed_tool_projection_detects_fifty_first_completion(self):
        from src.card.state import reducer as reducer_module
        from src.card.state.models import ToolBlock

        assert hasattr(reducer_module, "card_state_requires_continuation")
        completed = tuple(
            ToolBlock(
                block_id=f"completed-{index}",
                status="completed",
                tool_name="read",
            )
            for index in range(reducer_module.MAX_COMPLETED_TOOL_BLOCKS)
        )
        active = ToolBlock(
            block_id="active-tool",
            status="active",
            tool_name="read",
        )
        state = _base_state(blocks=completed + (active,))

        assert reducer_module.card_state_requires_continuation(
            state,
            CardEvent.tool_done("active-tool", "done"),
        ) is True
        assert state.blocks[-1].status == "active"

    def test_exact_completed_tool_boundary_does_not_rotate_early(self):
        from src.card.state import reducer as reducer_module
        from src.card.state.models import ToolBlock

        completed = tuple(
            ToolBlock(
                block_id=f"completed-{index}",
                status="completed",
                tool_name="read",
            )
            for index in range(reducer_module.MAX_COMPLETED_TOOL_BLOCKS - 1)
        )
        active = ToolBlock(
            block_id="active-tool",
            status="active",
            tool_name="read",
        )
        state = _base_state(blocks=completed + (active,))

        assert reducer_module.card_state_requires_continuation(
            state,
            CardEvent.tool_done("active-tool", "done"),
        ) is False
