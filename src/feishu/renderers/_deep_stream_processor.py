"""Spec-style card processor for Deep engine callbacks."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from ...acp import ACPEvent, ACPEventType
from ...card.events import CardEvent, CardEventType
from ...card.orchestrator import TaskOrchestrator
from ...card.render.build_heartbeat import BuildHeartbeat
from ...card.task_registry import TaskRegistry, tasks_from_plan_entries
from ...card.ui_text import UI_TEXT
from ...config import get_settings
from ...deep_engine import DeepEngineCallbacks
from ...deep_engine.models import DeepProject, DeepProjectStatus
from ..emoji import EmojiReaction
from ._base_stream_processor import BaseStreamProcessor

if TYPE_CHECKING:
    from ...card.session.rotator import SessionRotator
    from ...project import ProjectContext
    from .deep_renderer import DeepRenderer

logger = logging.getLogger(__name__)


class DeepStreamProcessor(BaseStreamProcessor):
    """Render Deep engine progress through the shared Spec-style card pipeline."""

    _CYCLE = 1
    _MAIN_TASK_ID = "_deep_main"

    def __init__(
        self,
        *,
        rotator: "SessionRotator",
        renderer: "DeepRenderer",
        message_id: str,
        chat_id: str,
        root_path: str | None,
        project: "ProjectContext | None",
    ) -> None:
        super().__init__(rotator=rotator, renderer=renderer, message_id=message_id, chat_id=chat_id)
        self._root_path = root_path
        self._project = project
        self._task_registry = TaskRegistry()
        self._task_registry.register(
            task_id=self._MAIN_TASK_ID,
            name=UI_TEXT["deep_main_task"],
            status="in_progress",
        )
        self._tool_count = 0
        self._plan_steps = 0
        self._phase = "analyzing"
        self._current_task_id = self._MAIN_TASK_ID
        self._has_real_plan = False
        self._subagent_task_ids: set[str] = set()
        self._materialized_tool_ids: set[str] = set()

        settings = get_settings()
        self._heartbeat = BuildHeartbeat(
            session_id=rotator.session_id,
            on_tick=self._on_heartbeat_tick,
            interval=float(settings.card.build_heartbeat_interval),
        )

    def build_callbacks(self) -> DeepEngineCallbacks:
        return DeepEngineCallbacks(
            on_analyzing_start=self.on_analyzing_start,
            on_analyzing_done=self.on_analyzing_done,
            on_event=self.on_event,
            on_project_done=self.on_project_done,
            on_error=self.on_error,
        )

    def on_analyzing_start(self, _requirement_text: str) -> None:
        self._dispatch_start()

    def on_analyzing_done(self, deep_project: DeepProject) -> None:
        self._dispatch_start(deep_project)

    def _dispatch_start(self, deep_project: DeepProject | None = None) -> None:
        if self._started_dispatched:
            return
        self._started_dispatched = True

        project_name = self._resolve_project_name(deep_project)
        root_path = self._resolve_root_path(deep_project)
        # Seed sticky task state before STARTED creates the first Feishu message.
        # STARTED preserves existing blocks, so the first visible card already
        # satisfies the "task list on every main card" contract.
        self._dispatch_task_list(self._MAIN_TASK_ID)
        self._rotator.dispatch(CardEvent.started())
        self._rotator.dispatch(CardEvent.cycle_started(self._CYCLE, 1))
        self._rotator.dispatch(CardEvent.phase_started(
            self._CYCLE,
            "analyzing",
            subtitle=UI_TEXT["deep_spec_style_subtitle_analyzing"],
            content=UI_TEXT["deep_exec_start"].format(
                project_name=project_name,
                root_path=root_path,
            ),
        ))
        self._phase = "analyzing"
        self._heartbeat.start()

    def _resolve_project_name(self, deep_project: DeepProject | None) -> str:
        if deep_project and deep_project.name:
            return deep_project.name
        if self._project and getattr(self._project, "project_name", None):
            return self._project.project_name
        root_path = self._resolve_root_path(deep_project)
        if root_path:
            return Path(root_path).name or "deep"
        return "deep"

    def _resolve_root_path(self, deep_project: DeepProject | None) -> str:
        if deep_project and deep_project.root_path:
            return deep_project.root_path
        if self._root_path:
            return self._root_path
        if self._project and getattr(self._project, "root_path", None):
            return self._project.root_path
        return ""

    def on_event(self, event: ACPEvent) -> None:
        """Process ACP events and dispatch Spec-style card events."""
        self._acp_renderer.process_event(event)
        activity = (
            "tool_running"
            if event.event_type in {
                ACPEventType.TOOL_CALL_START,
                ACPEventType.TOOL_CALL_UPDATE,
            }
            else "thinking"
        )
        self._heartbeat.reset(activity)

        if event.event_type == ACPEventType.PLAN_UPDATE:
            if self._handle_plan_task_list(event):
                self._ensure_build_phase()
            return

        if event.event_type == ACPEventType.TOOL_CALL_START:
            self._tool_count += 1
            self._ensure_build_phase()

        is_subagent_wrapper = self._handle_agent_task_list(event)
        self._forward_main_stream_event(
            event,
            suppress_wrapper=is_subagent_wrapper,
        )

        if event.event_type == ACPEventType.TOOL_CALL_START:
            self._rotator.dispatch(CardEvent.progress_updated(
                current=self._tool_count,
                total=max(self._plan_steps, self._tool_count),
                label=UI_TEXT["deep_phase_executing"],
            ))

    def _forward_main_stream_event(
        self,
        event: ACPEvent,
        *,
        suppress_wrapper: bool,
    ) -> None:
        """Forward visible ACP output with programming-card tool pairing."""
        tool_call = event.tool_call
        tool_id = str(getattr(tool_call, "id", "") or "").strip()
        is_tool_terminal = event.event_type == ACPEventType.TOOL_CALL_DONE

        # A wrapper that was visible before later collaboration metadata arrived
        # still needs its terminal event; otherwise its tool panel stays active.
        if suppress_wrapper and not (
            is_tool_terminal and tool_id in self._materialized_tool_ids
        ):
            return

        if (
            event.event_type in {
                ACPEventType.TOOL_CALL_UPDATE,
                ACPEventType.TOOL_CALL_DONE,
            }
            and tool_id
            and tool_id not in self._materialized_tool_ids
        ):
            # Match ordinary programming cards: an orphan delta/terminal event
            # must materialize a tool block before the reducer can update it.
            self._rotator.dispatch(CardEvent.tool_started(
                tool_id,
                str(getattr(tool_call, "title", "") or "tool"),
            ))
            self._materialized_tool_ids.add(tool_id)

        self._stream_bridge.on_event(event)
        if event.event_type == ACPEventType.TOOL_CALL_START and tool_id:
            self._materialized_tool_ids.add(tool_id)

    def on_project_done(self, deep_project: DeepProject) -> None:
        self._heartbeat.stop()
        self._stream_bridge.close_open_blocks()
        self._complete_active_phase()
        self._rotator.dispatch(CardEvent.cycle_done(self._CYCLE))

        snap = self._renderer._get_engine(self._chat_id, self._root_path, self._project)
        tool_calls_count = snap.tool_calls_count if snap else self._tool_count
        summary = UI_TEXT["deep_exec_completed"].format(tool_calls_count=tool_calls_count)
        duration_seconds = self._project_duration(deep_project)

        if deep_project.status == DeepProjectStatus.COMPLETED:
            self._cancel_unfinished_subagent_tasks()
            self._finalize_main_tasks(success=True)
            self._rotator.dispatch(CardEvent.completed(
                summary=summary,
                duration_seconds=duration_seconds,
            ))
            self._renderer.handler.add_reaction(self._message_id, EmojiReaction.on_multi_task_done())
        else:
            tasks = self._task_payload()
            completed = sum(1 for task in tasks if task.get("status") == "completed")
            failure = UI_TEXT["deep_exec_incomplete"].format(
                completed=completed,
                total=len(tasks),
            )
            if deep_project.status == DeepProjectStatus.PAUSED:
                self._cancel_unfinished_subagent_tasks()
                self._rotator.dispatch(CardEvent.cancelled(reason=failure))
            else:
                self._cancel_unfinished_subagent_tasks()
                self._finalize_main_tasks(success=False)
                self._rotator.dispatch(CardEvent.failed(
                    failure,
                    duration_seconds=duration_seconds,
                ))
        self._renderer._current_session = None

    def _cancel_unfinished_subagent_tasks(self) -> None:
        """Settle unconfirmed child summaries without creating child sessions."""
        changed = False
        for task_id in self._subagent_task_ids:
            item = self._task_registry.get(task_id)
            if item is None or item.status in {"completed", "failed", "cancelled"}:
                continue
            self._task_registry.update_status(task_id, "cancelled", notify=False)
            changed = True
        if changed:
            self._dispatch_task_list()

    def on_error(self, error: str) -> None:
        self._heartbeat.stop()
        self._cancel_unfinished_subagent_tasks()
        self._finalize_main_tasks(success=False)
        self._dispatch_failed(
            error,
            duration_seconds=self._current_project_duration(),
        )

    def _current_project_duration(self) -> float | None:
        """Return the authoritative Deep duration without weakening failures."""
        try:
            snap = self._renderer._get_engine(
                self._chat_id,
                self._root_path,
                self._project,
            )
            ext = getattr(snap, "ext", None)
            deep_project = ext.get("project") if isinstance(ext, dict) else None
            return self._project_duration(deep_project)
        except Exception:
            logger.debug("Deep duration snapshot unavailable", exc_info=True)
            return None

    @staticmethod
    def _project_duration(deep_project: object) -> float | None:
        """Normalize a domain duration, falling back on invalid wall-clock data."""
        duration_fn = getattr(deep_project, "duration", None)
        try:
            duration = duration_fn() if callable(duration_fn) else None
        except Exception:
            return None
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            return None
        return float(duration)

    def _on_heartbeat_tick(self, _elapsed: float, activity: str) -> None:
        """Refresh elapsed time and the latest execution status during quiet gaps."""
        if self._phase == "done":
            return
        is_executing = self._phase == "build" or activity == "tool_running"
        self._rotator.dispatch(CardEvent.progress_updated(
            current=self._tool_count,
            total=max(self._plan_steps, self._tool_count),
            label=UI_TEXT[
                "deep_phase_executing" if is_executing else "deep_phase_planning"
            ],
        ))

    def _finalize_main_tasks(self, *, success: bool) -> None:
        """Finalize unfinished sticky tasks before delivering the terminal card."""
        terminal_status = "completed" if success else "failed"
        for task in self._task_registry.get_snapshot():
            if task.status in {"pending", "in_progress"}:
                self._task_registry.update_status(
                    task.task_id,
                    terminal_status,
                    notify=False,
                )
        self._dispatch_task_list()

    def _task_payload(self) -> list[dict]:
        return [
            {"task_id": task.task_id, "name": task.name, "status": task.status}
            for task in self._task_registry.get_snapshot()
            if not (self._has_real_plan and task.task_id == self._MAIN_TASK_ID)
        ]

    def _pick_current_task_id(self, preferred: str = "") -> str:
        snapshot = [
            task
            for task in self._task_registry.get_snapshot()
            if not (self._has_real_plan and task.task_id == self._MAIN_TASK_ID)
        ]
        if preferred:
            for item in snapshot:
                if item.task_id == preferred and item.status == "in_progress":
                    return preferred
        for item in reversed(snapshot):
            if item.status == "in_progress":
                return item.task_id
        return ""

    def _dispatch_task_list(self, preferred_current_id: str = "") -> None:
        tasks = self._task_payload()
        if not tasks:
            return
        current = self._pick_current_task_id(preferred_current_id or self._current_task_id)
        self._current_task_id = current
        self._rotator.dispatch(CardEvent(
            type=CardEventType.TASK_LIST_UPDATED,
            payload={"tasks": tasks, "current_task_id": current},
        ))

    def _upsert_task(self, task_id: str, name: str, status: str) -> None:
        task_id = str(task_id or "").strip()
        name = str(name or "").strip()
        if not task_id:
            return
        if status not in {"pending", "in_progress", "completed", "failed", "cancelled"}:
            status = "pending"

        existing = self._task_registry.get(task_id)
        if existing is None:
            self._task_registry.register(task_id=task_id, name=name or "子任务", status=status)
            return

        if (
            name
            and self._is_generic_agent_task_label(existing.name)
            and not self._is_generic_agent_task_label(name)
        ):
            self._task_registry.update_name(task_id, name)
        if existing.status in {"completed", "failed", "cancelled"}:
            return
        self._task_registry.update_status(task_id, status)

    @staticmethod
    def _is_generic_agent_task_label(value: str) -> bool:
        label = str(value or "").strip()
        if label.startswith("🧬 "):
            label = label[2:].strip()
        return TaskOrchestrator._is_generic_task_label(label)

    def _handle_plan_task_list(self, event: ACPEvent) -> bool:
        if event.event_type != ACPEventType.PLAN_UPDATE or not event.plan:
            return False
        tasks = tasks_from_plan_entries(event.plan.entries)
        if not tasks:
            return False

        self._has_real_plan = True
        self._plan_steps = len(tasks)
        current = ""
        for task in tasks:
            task_id = str(task.get("task_id") or "")
            status = str(task.get("status") or "pending")
            self._upsert_task(task_id, str(task.get("name") or ""), status)
            if not current and status == "in_progress":
                current = task_id
        self._dispatch_task_list(current)
        return True

    def _handle_agent_task_list(self, event: ACPEvent) -> bool:
        if event.event_type not in {
            ACPEventType.TOOL_CALL_START,
            ACPEventType.TOOL_CALL_UPDATE,
            ACPEventType.TOOL_CALL_DONE,
        }:
            return False
        tool_call = event.tool_call
        if tool_call is None:
            return False
        collaboration_route = self._handle_collaboration_task_states(tool_call)
        if collaboration_route is not None:
            return collaboration_route

        is_agent_task_event = TaskOrchestrator.is_agent_task_event(event)
        task_id = str(
            getattr(tool_call, "subagent_source_id", "")
            or getattr(tool_call, "id", "")
            or ""
        ).strip()
        if not task_id:
            return False
        if not is_agent_task_event and self._task_registry.get(task_id) is None:
            return False
        self._subagent_task_ids.add(task_id)
        activity = str(getattr(tool_call, "subagent_activity", "") or "").strip().lower()
        if activity in {"started", "interacted"}:
            status = "in_progress"
        elif activity == "interrupted":
            provider_status = str(getattr(tool_call, "status", "") or "").lower()
            status = (
                "cancelled"
                if provider_status == "completed"
                else "in_progress"
            )
        elif event.event_type == ACPEventType.TOOL_CALL_DONE:
            raw_status = str(getattr(tool_call, "status", "") or "").strip().lower()
            status = "failed" if raw_status == "failed" else "completed"
        else:
            status = "in_progress"
        label = TaskOrchestrator._extract_agent_task_label(tool_call)
        if is_agent_task_event and not label.startswith("🧬 "):
            label = f"🧬 {label}"
        self._upsert_task(task_id, label, status)
        self._dispatch_task_list(task_id)
        return True

    def _handle_collaboration_task_states(self, tool_call) -> bool | None:
        """Project child summaries and decide whether to hide the wrapper."""
        states = tuple(getattr(tool_call, "subagent_states", ()) or ())
        is_collaboration = bool(
            getattr(tool_call, "collaboration_tool", None)
            or getattr(tool_call, "collaboration_receivers", ())
            or states
        )
        if not is_collaboration:
            return None

        states_by_source = {
            str(state.get("source_id") or "").strip(): state
            for state in states
            if isinstance(state, dict)
            and str(state.get("source_id") or "").strip()
        }
        source_ids = [
            str(source_id or "").strip()
            for source_id in getattr(tool_call, "collaboration_receivers", ())
            if str(source_id or "").strip()
        ]
        source_ids.extend(
            source_id
            for source_id in states_by_source
            if source_id not in source_ids
        )
        provider_status = str(getattr(tool_call, "status", "") or "").lower()
        if not source_ids:
            return provider_status != "failed"

        status_map = {
            "pendinginit": "in_progress",
            "pending_init": "in_progress",
            "pending": "in_progress",
            "running": "in_progress",
            "completed": "completed",
            "shutdown": "completed",
            "errored": "failed",
            "failed": "failed",
            "notfound": "failed",
            "not_found": "failed",
            "interrupted": "cancelled",
        }
        label = TaskOrchestrator._extract_agent_task_label(tool_call)
        if not label.startswith("🧬 "):
            label = f"🧬 {label}"
        current_task_id = ""
        for task_id in source_ids:
            self._subagent_task_ids.add(task_id)
            state = states_by_source.get(task_id, {})
            raw_status = str(
                state.get("status")
                or ("failed" if provider_status == "failed" else "running")
            ).strip().lower()
            status = status_map.get(raw_status, "pending")
            existing = self._task_registry.get(task_id)
            name = existing.name if existing is not None else label
            self._upsert_task(task_id, name, status)
            if not current_task_id and status == "in_progress":
                current_task_id = task_id

        self._dispatch_task_list(current_task_id)
        return provider_status != "failed"

    def _ensure_build_phase(self) -> None:
        if self._phase == "build":
            return
        self._dispatch_phase_transition(
            cycle=self._CYCLE,
            from_phase="analyzing",
            to_phase="build",
            done_content=UI_TEXT["deep_spec_style_analyzing_done"],
            done_subtitle=UI_TEXT["deep_spec_style_subtitle_build"],
            started_subtitle=UI_TEXT["deep_spec_style_subtitle_build"],
            started_content=UI_TEXT["deep_phase_executing"],
        )
        self._phase = "build"

    def _complete_active_phase(self) -> None:
        if self._phase == "done":
            return
        output_key = (
            "deep_spec_style_build_done"
            if self._phase == "build"
            else "deep_spec_style_analyzing_done"
        )
        self._rotator.dispatch(CardEvent.phase_done(
            self._CYCLE,
            self._phase,
            UI_TEXT[output_key],
        ))
        self._phase = "done"
