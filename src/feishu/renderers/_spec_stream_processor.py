"""SpecStreamProcessor — extracted state object for Spec Engine event callbacks.

Replaces the closure-heavy `SpecRenderer.create_spec_callbacks` approach with
an explicit state class. All mutable state that was previously captured via
`nonlocal` or list-wrapping patterns is now held as plain instance attributes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional

from ...acp import ACPEventType
from ...card.events import CardEvent, card_event_from_acp
from ...card.orchestrator import (
    UNCONFIRMED_SUBAGENT_SUMMARY,
    TaskOrchestrator,
)
from ...card.render.budget import RenderBudget
from ...card.render.build_heartbeat import BuildHeartbeat
from ...card.render.throttle import StreamThrottle
from ...card.state.models import CardMetadata
from ...card.stream_bridge import ACPStreamBridge
from ...card.ui_text import UI_TEXT
from ...spec_engine import SpecEngineCallbacks
from ...spec_engine.artifact_display import build_plan_display_payload, build_task_display_payloads
from ...spec_engine.models import (
    ReviewResult,
    SpecCycle,
    SpecPhase,
    SpecProject,
)
from ...spec_engine.retry_status import RetryEvent, RetryStatus
from ...spec_engine.review_display import build_review_role_payloads, format_review_overview
from ._base_stream_processor import BaseStreamProcessor
from .base import _dispatch_text_block

if TYPE_CHECKING:
    from ...card.session.rotator import SessionRotator
    from ...spec_engine.reporter import SpecReporter

logger = logging.getLogger(__name__)

# RetryStatus → UI_TEXT key mapping (class-level constant)
_RETRY_STATUS_TEXT: dict[RetryStatus, str] = {
    RetryStatus.WAITING: "retry_waiting",
    RetryStatus.EXECUTING: "retry_executing",
    RetryStatus.EXHAUSTED: "retry_exhausted",
    RetryStatus.NO_RETRY: "retry_no_retry",
}

_STRUCTURED_ARTIFACT_PHASES = frozenset({
    SpecPhase.SPEC,
    SpecPhase.PLAN,
    SpecPhase.TASK,
})


class _EngineContext(NamedTuple):
    """Lightweight return type for _get_engine_and_state()."""
    snap: object  # EngineSnapshot | None
    spec_project: "SpecProject | None"
    state: dict
    max_cycles: int


class SpecStreamProcessor(BaseStreamProcessor):
    """Holds mutable state and implements all Spec Engine event callbacks.

    Constructed once per `create_spec_callbacks` invocation.  The renderer
    delegates callback creation to this class via `build_callbacks()`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        rotator: "SessionRotator",
        reporter: "SpecReporter",
        metadata: CardMetadata,
        hooks: tuple,
        budget: RenderBudget,
        spec_project_id: str,
        message_id: str,
        chat_id: str,
        # Renderer/handler references (kept opaque to avoid circular import)
        renderer,  # SpecRenderer instance
        project_root_path: str,
        throttle: StreamThrottle,
    ) -> None:
        # Base class handles: rotator, renderer, message_id, chat_id,
        # _acp_renderer, _stream_bridge, _started_dispatched, _start_time
        super().__init__(rotator=rotator, renderer=renderer, message_id=message_id, chat_id=chat_id)

        # Immutable dependencies
        self._reporter = reporter
        # Capture the initial session's session_started_at into the metadata
        # template so that subsequent _rotate_session() calls propagate the
        # total start time rather than resetting to the current cycle's start.
        if metadata.session_started_at is None:
            from dataclasses import replace as _dc_replace
            initial_started = rotator.current.session_started_at
            metadata = _dc_replace(metadata, session_started_at=initial_started)
        self._metadata = metadata
        self._hooks = hooks
        self._budget = budget
        self._spec_project_id = spec_project_id
        self._project_root_path = project_root_path

        # Mutable state (previously closure-captured variables)
        self._max_cycles: int = 0
        self._throttle: StreamThrottle = throttle
        self._footer_status: Optional[str] = None
        self._last_phase_content: str = ""
        self._current_cycle: int = 0
        # Rotation guard: minimum dispatched events before allowing rotation
        self._events_since_rotation: int = 0
        # Build phase tool tracking
        self._build_tool_count: int = 0
        self._build_file_set: set[str] = set()
        self._build_tool_ids: set[str] = set()
        self._build_started_tool_ids: set[str] = set()
        self._build_heartbeat: BuildHeartbeat | None = None

        # TaskOrchestrator for optional task-level cards within a Spec cycle's Build phase
        def _task_session_creator(task_id: str):
            from dataclasses import replace as _replace
            task_item = self._orchestrator.registry.get(task_id)
            task_label = task_item.name if task_item else task_id
            task_metadata = _replace(
                metadata,
                unit_kind="task",
                unit_label=task_label,
                unit_id=task_id,
                iteration_index=self._current_cycle or None,
                iteration_total=self._max_cycles or None,
            )
            return renderer.create_session(chat_id, message_id, task_metadata, hooks=hooks, budget=budget)

        from ...config import get_settings
        self._multi_card_enabled = get_settings().card.task_level_cards_enabled

        self._orchestrator = TaskOrchestrator.from_settings(
            chat_id=chat_id,
            session_creator=_task_session_creator,
            thinking_session=self._rotator,
            bridge_class=lambda session: ACPStreamBridge(
                session,
                image_uploader=self._image_uploader,
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers (previously nested closures)
    # ------------------------------------------------------------------

    # Minimum events before allowing a card rotation (prevents ultra-short cards)
    _MIN_EVENTS_BEFORE_ROTATION: int = 3

    def _rotate_session(self, cycle_num: int) -> bool:
        """Conditionally rotate to a new session at cycle boundary.

        Skips rotation if the current session hasn't accumulated enough content
        (prevents very short cards being sent). In that case, just continues on
        the same session with a cycle_started event as visual separator.

        Returns True if rotation was performed, False if skipped.
        """
        current_session = self._rotator.current
        # Skip rotation if:
        # 1. Session hasn't been delivered to Feishu yet (card doesn't exist), OR
        # 2. Too few events have been dispatched since last rotation
        if not current_session.delivered_message_id or self._events_since_rotation < self._MIN_EVENTS_BEFORE_ROTATION:
            logger.debug(
                "SpecStreamProcessor: skipping rotation for cycle %d "
                "(delivered=%s, events_since=%d)",
                cycle_num, bool(current_session.delivered_message_id),
                self._events_since_rotation,
            )
            return False

        cont_meta = self._renderer.build_unit_metadata(
            self._metadata,
            unit_id=str(cycle_num),
            unit_kind="cycle",
            unit_label=UI_TEXT["spec_cycle_label"].format(cycle_num=cycle_num),
            iteration_index=cycle_num,
            iteration_total=self._max_cycles or None,
            continuation_seq=self._rotator.rotation_count + 1,
        )
        if self._renderer._pending_split_hint:
            from dataclasses import replace
            if isinstance(cont_meta, CardMetadata):
                cont_meta = replace(cont_meta, bridge_phrase="续接：")
            self._renderer._pending_split_hint = None
        old_msg_id = self._rotator.current.delivered_message_id or self._message_id
        renderer = self._renderer
        chat_id = self._chat_id
        hooks = self._hooks
        budget = self._budget
        self._rotator.rotate(
            lambda: renderer.create_session(chat_id, old_msg_id, cont_meta, hooks=hooks, budget=budget)
        )
        self._stream_bridge.bind(self._rotator)
        self._events_since_rotation = 0
        return True

    def _get_engine_and_state(self) -> _EngineContext:
        """Return structured engine context."""
        ctx = self._renderer.ctx
        snap = ctx.spec_engine_manager.snapshot(self._chat_id, self._project_root_path)
        spec_project = snap.ext.get("project") if snap else None
        state = self._renderer.get_ui_state(self._spec_project_id)
        max_c = self._max_cycles or (spec_project.cycle_count_total if spec_project else 10)
        return _EngineContext(snap=snap, spec_project=spec_project, state=state, max_cycles=max_c)

    # ------------------------------------------------------------------
    # Callback methods
    # ------------------------------------------------------------------

    def on_analyzing_start(self, requirement: str) -> None:
        self._renderer.update_ui_state(self._spec_project_id, view_mode="status", view_context={})
        self._dispatch_started_once()
        content = self._reporter.format_analyzing_start(requirement)
        self._rotator.dispatch(CardEvent.text_delta("_main", content))
        self._events_since_rotation += 1

    def on_analyzing_done(self, spec_project: SpecProject) -> None:
        self._renderer.update_ui_state(self._spec_project_id, view_mode="status", view_context={})
        self._dispatch_started_once()
        content = self._reporter.format_analyzing_done(spec_project)
        self._rotator.dispatch(CardEvent.text_delta("_main", content))
        self._events_since_rotation += 1

    def on_cycle_start(self, current: int, max_cycles: int) -> None:
        self._max_cycles = max_cycles
        self._current_cycle = current
        self._renderer.update_ui_state(self._spec_project_id, view_mode="status", view_context={})
        self._renderer.notify_cycle_change(current_cycle=current, perspective=None)
        self._orchestrator.reset()
        rotated = self._rotate_session(current)
        if rotated:
            self._rotator.dispatch(CardEvent.started())

        _, spec_project, state, _ = self._get_engine_and_state()

        self._rotator.dispatch(CardEvent.cycle_started(current, max_cycles))
        self._events_since_rotation += 1

        if spec_project:
            criteria_section = self._reporter.format_criteria_section(spec_project)
            self._rotator.dispatch(CardEvent.criteria_updated(
                criteria_section,
                satisfied_count=spec_project.satisfied_count,
                total_count=spec_project.total_criteria,
            ))

            warning = self._renderer.check_warning_banner(spec_project.duration())
            if warning:
                self._rotator.dispatch(CardEvent.warning_updated(warning))

    def on_cycle_done(self, cycle_num: int, cycle: SpecCycle) -> None:
        self._renderer.update_ui_state(
            self._spec_project_id, view_mode="cycle_done", view_context={"cycle_num": cycle_num}
        )
        self._stream_bridge.close_open_blocks()

        _, spec_project, state, _ = self._get_engine_and_state()
        if spec_project:
            self._rotator.dispatch(CardEvent.cycle_done(cycle_num))
            content = self._reporter.format_cycle_done(cycle_num, cycle)
            _dispatch_text_block(self._rotator, f"_cycle_done_{cycle_num}", content)

            criteria_section = self._reporter.format_criteria_section(spec_project)
            self._rotator.dispatch(CardEvent.criteria_updated(
                criteria_section,
                satisfied_count=spec_project.satisfied_count,
                total_count=spec_project.total_criteria,
            ))

    def on_review_done(self, cycle_num: int, review: ReviewResult) -> None:
        self._renderer.update_ui_state(
            self._spec_project_id, view_mode="review_done", view_context={"cycle_num": cycle_num}
        )

        content = format_review_overview(review, cycle_num)
        _, spec_project, state, max_c = self._get_engine_and_state()
        status_content = self._reporter.format_phase_done_content(
            cycle_num, SpecPhase.REVIEW, max_c, content
        )
        subtitle = self._reporter.format_phase_subtitle(
            cycle_num, max_c, SpecPhase.REVIEW, completed=True
        )
        self._rotator.dispatch(
            CardEvent.phase_done(
                cycle_num, SpecPhase.REVIEW.value, status_content, subtitle=subtitle
            )
        )
        _dispatch_text_block(self._rotator, f"_review_done_{cycle_num}", content)
        self._rotator.dispatch(
            CardEvent.review_result_updated(cycle_num, build_review_role_payloads(review, cycle_num))
        )

        if spec_project:
            criteria_section = self._reporter.format_criteria_section(spec_project)
            self._rotator.dispatch(CardEvent.criteria_updated(
                criteria_section,
                satisfied_count=spec_project.satisfied_count,
                total_count=spec_project.total_criteria,
            ))

            warning = self._renderer.check_warning_banner(spec_project.duration(), is_executing=True)
            if warning:
                self._rotator.dispatch(CardEvent.warning_updated(warning))

        if not review.all_passed:
            self._rotator.dispatch(CardEvent.review_retry(
                cycle_num=cycle_num,
                attempt=cycle_num,
                max_attempts=0,
                status="executing",
            ))

    def on_project_done(self, spec_project: SpecProject) -> None:
        self._renderer.update_ui_state(self._spec_project_id, view_mode="status", view_context={})
        self._stream_bridge.close_open_blocks()
        content = self._reporter.format_project_done(spec_project)
        _dispatch_text_block(self._rotator, "_project_done", content)

        criteria_section = self._reporter.format_criteria_section(spec_project)
        self._rotator.dispatch(CardEvent.criteria_updated(
            criteria_section,
            satisfied_count=spec_project.satisfied_count,
            total_count=spec_project.total_criteria,
        ))

        # Terminal event (hooks fire emoji automatically)
        self._orchestrator.finalize_unfinished_subagents(
            status="cancelled",
            summary=UNCONFIRMED_SUBAGENT_SUMMARY,
        )
        if spec_project.status.value == "completed":
            self._orchestrator.close(terminal_status="completed")
            self._rotator.dispatch(CardEvent.completed())
        else:
            self._orchestrator.close(
                terminal_status="failed",
                summary=UI_TEXT["card_project_failed"],
            )
            self._rotator.dispatch(CardEvent.failed(UI_TEXT["card_project_failed"]))
        self._renderer._current_session = None

    def on_error(self, error: str) -> None:
        self._renderer.update_ui_state(
            self._spec_project_id, view_mode="error", view_context={"error": error}
        )

        self._orchestrator.finalize_unfinished_subagents(
            status="cancelled",
            summary=UNCONFIRMED_SUBAGENT_SUMMARY,
        )
        self._orchestrator.close(
            terminal_status="failed",
            summary=error,
        )

        self._dispatch_failed(error)

    def on_phase_start(self, cycle_num: int, phase: SpecPhase) -> None:
        self._acp_renderer.reset()
        self._footer_status = "tool_running"
        phase_name = phase.value if hasattr(phase, "value") else str(phase)
        self._renderer.notify_cycle_change(current_cycle=cycle_num, perspective=phase_name)

        _, spec_project, state, max_c = self._get_engine_and_state()
        subtitle = self._reporter.format_phase_subtitle(cycle_num, max_c, phase, completed=False)
        content = self._reporter.format_phase_start_content(cycle_num, phase, max_c)

        self._rotator.dispatch(
            CardEvent.phase_started(cycle_num, phase_name, subtitle=subtitle, content=content)
        )
        self._events_since_rotation += 1

        if phase == SpecPhase.BUILD:
            # Build phase: use tool panels, no text content needed
            self._build_tool_count = 0
            self._build_file_set = set()
            self._build_tool_ids = set()
            self._build_started_tool_ids = set()
            self._start_build_heartbeat()

    def on_phase_event(self, cycle_num: int, phase: SpecPhase, event) -> None:
        """Real-time ACP event processing."""
        self._acp_renderer.process_event(event)

        # Track footer_status
        if event.event_type == ACPEventType.THOUGHT_CHUNK:
            self._footer_status = "thinking"
        elif event.event_type in (ACPEventType.TOOL_CALL_START, ACPEventType.TOOL_CALL_UPDATE):
            self._footer_status = "tool_running"
        elif event.event_type == ACPEventType.TEXT_CHUNK:
            self._footer_status = None

        # Reset heartbeat timer on every BUILD event
        if phase == SpecPhase.BUILD and self._build_heartbeat is not None:
            activity = "tool_running" if self._footer_status == "tool_running" else "thinking"
            self._build_heartbeat.reset(activity)

        # Codex child lifecycle is independent from whether the model emits a
        # PLAN_UPDATE. Once a stable child source is registered, keep its text
        # and thought stream on that child card in every Spec phase.
        source_id = str(getattr(event, "source_id", "") or "").strip()
        tool_call = getattr(event, "tool_call", None)
        is_dynamic_agent_event = (
            tool_call is not None
            and TaskOrchestrator.is_agent_task_event(event)
        )
        is_bound_child_event = bool(
            source_id and self._orchestrator.registry.get(source_id) is not None
        )
        if (
            is_dynamic_agent_event
            or is_bound_child_event
        ):
            self._orchestrator.route_acp_event(event, self._stream_bridge)
            return

        # SPEC/PLAN/TASK are structured artifact phases whose model output is
        # intentionally JSON-like for parsing. Streaming those chunks verbatim
        # makes the Feishu card unreadable; keep the raw output in the engine
        # tracker/artifact path and render a concise phase summary on done.
        if phase in _STRUCTURED_ARTIFACT_PHASES and event.event_type == ACPEventType.TEXT_CHUNK:
            return

        # Detect PLAN_UPDATE for optional task-level cards in BUILD phase
        if self._multi_card_enabled and event.event_type == ACPEventType.PLAN_UPDATE and phase == SpecPhase.BUILD:
            self._orchestrator.handle_plan_update(event, self._stream_bridge)

        if phase == SpecPhase.BUILD:
            # Build phase: route through orchestrator if multi-card enabled
            if self._orchestrator.has_plan and not self._orchestrator.is_fallback_mode:
                self._orchestrator.route_acp_event(event, self._stream_bridge)
            else:
                self._dispatch_build_event(event)
        else:
            self._stream_bridge.on_event(event)

    def _dispatch_build_event(self, event) -> None:
        """Dispatch Build-phase ACP events as native tool/plan CardEvents."""
        if event.event_type == ACPEventType.IMAGE_CHUNK:
            self._stream_bridge.on_event(event)
            return
        card_evt = card_event_from_acp(event)

        if event.event_type == ACPEventType.TOOL_CALL_START:
            self._mark_build_tool_started(event)
            self._rotator.dispatch(card_evt)
            self._dispatch_build_progress(event)
            return

        if event.event_type in (ACPEventType.TOOL_CALL_UPDATE, ACPEventType.TOOL_CALL_DONE):
            self._dispatch_synthetic_tool_start_if_needed(event)
            self._dispatch_build_progress(event)

        self._rotator.dispatch(card_evt)

    def _mark_build_tool_started(self, event) -> None:
        tc = getattr(event, "tool_call", None)
        tool_id = str(getattr(tc, "id", "") or "").strip()
        if tool_id:
            self._build_started_tool_ids.add(tool_id)

    def _dispatch_synthetic_tool_start_if_needed(self, event) -> None:
        tc = getattr(event, "tool_call", None)
        tool_id = str(getattr(tc, "id", "") or "").strip()
        if not tool_id or tool_id in self._build_started_tool_ids:
            return
        tool_name = str(getattr(tc, "title", "") or getattr(tc, "kind", "") or "Tool")
        self._rotator.dispatch(CardEvent.tool_started(tool_id, tool_name, ""))
        self._build_started_tool_ids.add(tool_id)

    def _dispatch_build_progress(self, event) -> None:
        tc = getattr(event, "tool_call", None)
        tool_id = str(getattr(tc, "id", "") or "").strip()
        if not tool_id:
            return

        self._build_tool_ids.add(tool_id)
        self._build_tool_count = len(self._build_tool_ids)
        self._record_build_file_refs(tc)

        progress_text = UI_TEXT["spec_build_progress"].format(tool_count=self._build_tool_count)
        if self._build_file_set:
            progress_text += UI_TEXT["spec_build_progress_files"].format(file_count=len(self._build_file_set))
        self._rotator.dispatch(CardEvent.progress_updated(
            current=self._build_tool_count, total=0, label=progress_text
        ))

    def _record_build_file_refs(self, tc) -> None:
        for path in getattr(tc, "locations", None) or []:
            path_text = str(path or "").strip()
            if path_text:
                self._build_file_set.add(path_text)

        title_lower = str(getattr(tc, "title", "") or "").lower()
        if not any(kw in title_lower for kw in ("write", "edit", "create", "read")):
            return
        content = str(getattr(tc, "content", "") or "")
        if content and "/" in content:
            first_line = content.split("\n")[0].strip()
            if "/" in first_line and len(first_line) < 200:
                self._build_file_set.add(first_line)

    # ------------------------------------------------------------------
    # Build heartbeat — periodic footer progress refresh
    # ------------------------------------------------------------------

    def _start_build_heartbeat(self) -> None:
        self._stop_build_heartbeat()
        from ...config import get_settings
        interval = get_settings().card.build_heartbeat_interval
        self._build_heartbeat = BuildHeartbeat(
            session_id=self._rotator.session_id,
            on_tick=self._on_build_heartbeat_tick,
            interval=interval,
        )
        self._build_heartbeat.start()

    def _stop_build_heartbeat(self) -> None:
        hb = self._build_heartbeat
        self._build_heartbeat = None
        if hb is not None:
            hb.stop()

    def _on_build_heartbeat_tick(self, elapsed: float, activity: str) -> None:
        """Called by heartbeat timer — dispatch a progress_updated event."""
        seconds = int(elapsed)
        if seconds < 3:
            return

        if self._build_tool_count > 0:
            progress_text = UI_TEXT["spec_build_progress"].format(tool_count=self._build_tool_count)
            if self._build_file_set:
                progress_text += UI_TEXT["spec_build_progress_files"].format(file_count=len(self._build_file_set))
            suffix_key = f"spec_build_heartbeat_{activity}"
            progress_text += UI_TEXT.get(suffix_key, UI_TEXT["spec_build_heartbeat_thinking"]).format(seconds=seconds)
        else:
            progress_text = UI_TEXT["spec_build_heartbeat_idle"].format(seconds=seconds)

        self._rotator.dispatch(CardEvent.progress_updated(
            current=self._build_tool_count, total=0, label=progress_text
        ))

    def on_phase_done(self, cycle_num: int, phase: SpecPhase, output: str) -> None:
        self._stop_build_heartbeat()
        _, spec_project, state, max_c = self._get_engine_and_state()
        self._footer_status = None
        self._stream_bridge.close_open_blocks()
        status_content = self._reporter.format_phase_done_content(cycle_num, phase, max_c, output)
        subtitle = self._reporter.format_phase_subtitle(cycle_num, max_c, phase, completed=True)

        self._rotator.dispatch(CardEvent.phase_done(
            cycle_num,
            phase.value if hasattr(phase, 'value') else str(phase),
            status_content,
            subtitle=subtitle,
        ))
        self._events_since_rotation += 1

        if phase == SpecPhase.BUILD:
            # Build phase: tool panels already rendered, just show summary
            summary = self._reporter._extract_phase_summary(phase, output)
            done_text = UI_TEXT["spec_build_done"].format(summary=summary) if summary else UI_TEXT["spec_build_done_plain"]
            _dispatch_text_block(self._rotator, f"_phase_done_{cycle_num}_{phase.value}", done_text)
        elif phase == SpecPhase.PLAN:
            plan_payload = build_plan_display_payload(output, cycle_num)
            self._rotator.dispatch(CardEvent.spec_plan_updated(cycle_num, plan_payload))
        elif phase == SpecPhase.TASK:
            task_payloads = build_task_display_payloads(output, cycle_num)
            if task_payloads:
                self._rotator.dispatch(CardEvent.spec_tasks_updated(cycle_num, task_payloads))

    def on_phase_retry(self, attempt: int, max_attempts: int, detail: str) -> None:
        """Push phase-level retry status."""
        retry_text = UI_TEXT["phase_retry_progress"].format(attempt=attempt, max_attempts=max_attempts)
        if detail:
            retry_text += f" — {detail[:80]}"
        self._footer_status = retry_text

        self._rotator.dispatch(CardEvent.review_retry(
            cycle_num=0,
            attempt=attempt,
            max_attempts=max_attempts,
            status="executing",
        ))

    def on_review_retry(self, cycle: int, event: RetryEvent) -> None:
        """Push review-level retry status."""
        if event.status == RetryStatus.SUCCEEDED:
            return

        text_key = _RETRY_STATUS_TEXT.get(event.status, "retry_executing")
        if event.status == RetryStatus.WAITING:
            detail_msg = UI_TEXT[text_key].format(sec=int(event.delay_sec), i=event.attempt, n=event.max_attempts)
        elif event.status == RetryStatus.EXECUTING:
            detail_msg = UI_TEXT[text_key].format(i=event.attempt, n=event.max_attempts)
        elif event.status == RetryStatus.EXHAUSTED:
            detail_msg = UI_TEXT[text_key].format(n=event.max_attempts)
        elif event.status == RetryStatus.NO_RETRY:
            if event.max_attempts == 0:
                detail_msg = UI_TEXT["retry_no_retry_disabled"]
            else:
                detail_msg = UI_TEXT["retry_no_retry_budget"]
        else:
            detail_msg = UI_TEXT[text_key]
        self._footer_status = detail_msg

        self._rotator.dispatch(CardEvent.review_retry(
            cycle_num=cycle,
            attempt=event.attempt,
            max_attempts=event.max_attempts,
            status=event.status.value if hasattr(event.status, 'value') else str(event.status),
        ))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def build_callbacks(self) -> SpecEngineCallbacks:
        """Assemble a SpecEngineCallbacks bound to this processor's methods."""
        return SpecEngineCallbacks(
            on_analyzing_start=self.on_analyzing_start,
            on_analyzing_done=self.on_analyzing_done,
            on_cycle_start=self.on_cycle_start,
            on_phase_start=self.on_phase_start,
            on_phase_event=self.on_phase_event,
            on_phase_done=self.on_phase_done,
            on_cycle_done=self.on_cycle_done,
            on_review_done=self.on_review_done,
            on_project_done=self.on_project_done,
            on_error=self.on_error,
            on_phase_retry=self.on_phase_retry,
            on_review_retry=self.on_review_retry,
        )
