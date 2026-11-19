"""Support helpers for programming mode handlers and execution."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...card.actions import dispatch as card_action_ids
from ...card.ui_text import UI_TEXT

if TYPE_CHECKING:
    from ...acp.collaboration import AuthoritativeChildLifecycleProof
    from ...acp.outcome import PromptAssessment
    from ...card.session.config import SessionCallbacks

logger = logging.getLogger("src.feishu.handlers.programming")

_BTW_EVENT_LIMIT = 48
_BTW_EVENT_TEXT_LIMIT = 600
_BTW_SNAPSHOT_LIMIT = 12_000


def _bounded_btw_text(value: object, limit: int = _BTW_EVENT_TEXT_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _format_btw_event(event: object) -> str:
    """Project one ACP event into a bounded, user-visible observation."""
    event_type = getattr(getattr(event, "event_type", None), "value", "")
    if event_type in {"text_chunk", "thought_chunk"}:
        label = "Agent 输出" if event_type == "text_chunk" else "Agent 分析"
        text = _bounded_btw_text(getattr(event, "text", ""))
        return f"{label}: {text}" if text else ""
    if event_type == "image_chunk":
        image = getattr(event, "image", None)
        name = _bounded_btw_text(getattr(image, "name", "") or "任务图片")
        source = _bounded_btw_text(getattr(image, "source_uri", ""), 300)
        return f"图片: {name}" + (f" ({source})" if source else "")
    if event_type in {"tool_call_start", "tool_call_update", "tool_call_done"}:
        tool_call = getattr(event, "tool_call", None)
        if tool_call is None:
            return ""
        status = _bounded_btw_text(getattr(tool_call, "status", "") or event_type)
        title = _bounded_btw_text(getattr(tool_call, "title", "") or "未命名操作", 300)
        content = _bounded_btw_text(getattr(tool_call, "content", ""), 300)
        line = f"工具[{status}]: {title}"
        return f"{line} — {content}" if content else line
    if event_type == "plan_update":
        plan = getattr(event, "plan", None)
        entries = getattr(plan, "entries", ()) or ()
        parts = []
        for entry in entries:
            content = _bounded_btw_text(getattr(entry, "content", ""), 240)
            if content:
                parts.append(f"[{getattr(entry, 'status', 'pending')}] {content}")
        return "计划: " + "; ".join(parts) if parts else ""
    return ""


@dataclass
class _ActiveProgrammingRun:
    task_text: str
    cwd: str
    agent_type: str
    model_name: str | None
    started_at: float = field(default_factory=time.monotonic)
    events: deque[str] = field(
        default_factory=lambda: deque(maxlen=_BTW_EVENT_LIMIT)
    )
    session: object | None = field(default=None, repr=False)
    prog_session: object | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _btw_inflight: bool = field(default=False, repr=False)

    def observe(self, event: object) -> None:
        line = _format_btw_event(event)
        if not line:
            return
        with self._lock:
            if (
                self.events
                and line.startswith(("Agent 输出:", "Agent 分析:"))
                and self.events[-1].partition(":")[0]
                == line.partition(":")[0]
            ):
                label, _, previous = self.events[-1].partition(":")
                _, _, addition = line.partition(":")
                combined = _bounded_btw_text(
                    f"{previous} {addition}",
                    _BTW_EVENT_TEXT_LIMIT,
                )
                self.events[-1] = f"{label}: {combined}"
                return
            self.events.append(line)

    def claim_btw(self) -> bool:
        with self._lock:
            if self._btw_inflight:
                return False
            self._btw_inflight = True
            return True

    def release_btw(self) -> None:
        with self._lock:
            self._btw_inflight = False

    def snapshot(self) -> str:
        with self._lock:
            events = tuple(self.events)
        elapsed = max(0, int(time.monotonic() - self.started_at))
        header = "\n".join(
            (
                f"主任务: {_bounded_btw_text(self.task_text, 2000)}",
                f"已运行: {elapsed} 秒",
                "最近执行事件:",
            )
        )
        event_text = "\n".join(f"- {line}" for line in events)
        if not events:
            event_text = "- 暂无可见执行事件"
        remaining = max(0, _BTW_SNAPSHOT_LIMIT - len(header) - 1)
        return f"{header}\n{event_text[-remaining:]}"

    def progress_fallback(self) -> str:
        with self._lock:
            events = tuple(self.events)[-4:]
        if not events:
            return "主任务仍在运行，但暂时还没有可用于回答的执行事件。"
        return "当前可见进展：\n" + "\n".join(f"- {line}" for line in events)


def _append_execution_notice(text: str, notice: str) -> str:
    """Append one terminal diagnostic without duplicating backend output."""
    content = str(text or "").strip()
    notice = str(notice or "").strip()
    if not notice or notice in content:
        return content
    return f"{content}\n\n{notice}" if content else notice


def _was_already_streamed_on_main(
    terminal_text: str,
    streamed_response: str,
) -> bool:
    """Return whether unstreamed-looking result text merely mirrors main stream."""
    norm_terminal = " ".join(str(terminal_text or "").split())
    norm_streamed = " ".join(str(streamed_response or "").split())
    if not norm_terminal or not norm_streamed:
        return False
    return (
        norm_terminal == norm_streamed
        or norm_streamed.endswith(norm_terminal)
    )


def _configured_finalization_reserve(settings: object) -> int:
    raw = getattr(settings, "programming_finalization_reserve_s", 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    return max(0, int(raw))


def _configured_execution_windows(settings: object) -> int:
    raw = getattr(settings, "programming_max_execution_windows", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 1
    return min(24, max(1, raw))


def _configured_programming_idle_timeout(
    settings: object,
) -> int | None:
    raw = getattr(settings, "programming_agent_idle_timeout_s", 0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = max(0, int(raw))
    return value or None


def _finalization_incomplete_reason(
    assessment: PromptAssessment,
) -> str:
    if assessment.pending_plan_entries > 0:
        return f"仍有 {assessment.pending_plan_entries} 个计划项未完成"
    if assessment.incomplete_tool_calls > 0:
        return (
            f"仍有 {assessment.incomplete_tool_calls} 个工具调用未进入终态"
        )
    return assessment.detail


def _incomplete_notice(execution: object) -> str:
    assessment = execution.assessment
    if getattr(execution, "window_limit_reached", False):
        return UI_TEXT["mode_exec_window_limit_incomplete_msg"].format(
            windows=max(1, int(getattr(execution, "execution_windows", 1))),
            reason=_finalization_incomplete_reason(assessment),
        )
    if not execution.entered_finalization:
        return UI_TEXT["mode_exec_incomplete_msg"].format(
            reason=assessment.detail,
        )
    goal = getattr(execution.result, "goal", None)
    key = (
        "mode_exec_finalization_paused_goal_msg"
        if goal is not None and goal.status == "paused"
        else "mode_exec_finalization_incomplete_msg"
    )
    return UI_TEXT[key].format(
        reason=_finalization_incomplete_reason(assessment),
    )


def _log_prompt_execution(mode_name: str, execution: object) -> None:
    assessment = execution.assessment
    goal = getattr(execution.result, "goal", None)
    raw_goal_status = getattr(goal, "status", None)
    goal_status = (
        raw_goal_status.strip().casefold()
        if isinstance(raw_goal_status, str)
        and raw_goal_status.strip().casefold()
        in {"active", "paused", "blocked", "completed"}
        else "unknown"
    )
    logger.info(
        "%s ACP执行判定: entered_finalization=%s execution_windows=%s "
        "window_limit_reached=%s goal_status=%s "
        "automatic_continuations=%s pending_plan_entries=%s "
        "incomplete_tool_calls=%s incomplete_outer_tool_calls=%s "
        "unresolved_child_tool_calls=%s unresolved_tools=%s",
        mode_name,
        execution.entered_finalization,
        getattr(execution, "execution_windows", 1),
        getattr(execution, "window_limit_reached", False),
        goal_status,
        execution.automatic_continuations,
        assessment.pending_plan_entries,
        assessment.incomplete_tool_calls,
        assessment.incomplete_outer_tool_calls,
        assessment.unresolved_child_tool_calls,
        ";".join(assessment.incomplete_tool_diagnostics) or "none",
    )


def _log_final_unresolved_tool_trace(
    mode_name: str,
    assessment: PromptAssessment,
) -> None:
    """Log Codex-only correlation data once the handler kept the failure."""
    if str(mode_name or "").strip().casefold() != "codex":
        return
    from ...acp.outcome import IncompleteToolTrace

    trace = tuple(
        observation
        for observation in assessment.incomplete_tool_traces
        if isinstance(observation, IncompleteToolTrace)
    )
    rendered = ";".join(
        "id_sha256={id_sha256},diagnostic={diagnostic},"
        "child_source_present={child_source_present},"
        "context_compaction={context_compaction},"
        "child_metadata_malformed={child_metadata_malformed}".format(
            id_sha256=observation.id_sha256,
            diagnostic=observation.diagnostic,
            child_source_present=int(observation.child_source_present),
            context_compaction=int(observation.context_compaction),
            child_metadata_malformed=int(
                observation.child_metadata_malformed
            ),
        )
        for observation in trace
    )
    logger.info(
        "%s ACP未终态工具关联证据: tool_count=%d traces=%s",
        mode_name,
        assessment.incomplete_tool_calls,
        rendered or "unavailable",
    )


def _execution_diagnostic_details(execution: object) -> str:
    """Return a bounded, provider-allowlisted diagnostic for card callbacks."""
    assessment = execution.assessment
    diagnostics = ";".join(assessment.incomplete_tool_diagnostics) or "none"
    return "\n".join(
        (
            f"outcome={assessment.outcome.value}",
            f"stop_reason={assessment.stop_reason or 'unknown'}",
            f"execution_windows={max(1, int(getattr(execution, 'execution_windows', 1)))}",
            f"entered_finalization={bool(execution.entered_finalization)}",
            f"automatic_continuations={max(0, int(execution.automatic_continuations))}",
            f"pending_plan_entries={assessment.pending_plan_entries}",
            f"incomplete_tool_calls={assessment.incomplete_tool_calls}",
            f"incomplete_outer_tool_calls={assessment.incomplete_outer_tool_calls}",
            f"unresolved_child_tool_calls={assessment.unresolved_child_tool_calls}",
            f"unresolved_tools={diagnostics}",
        )
    )


def _diagnostic_action(chat_id: str, message_id: str) -> dict[str, str]:
    return {
        "action": card_action_ids.SHOW_ERROR_DETAILS,
        "chat_id": chat_id,
        "origin_message_id": message_id,
    }


def _has_reconciled_stale_child_history(
    execution: object,
    child_lifecycle_proof: AuthoritativeChildLifecycleProof,
) -> bool:
    """Accept only a final explicit list_agents proof, never a display inference."""
    if execution.entered_finalization:
        return False
    if getattr(execution, "window_limit_reached", False):
        return False
    from ...acp.outcome import PromptOutcome

    assessment = execution.assessment
    if (
        assessment.outcome is not PromptOutcome.INCOMPLETE
        or assessment.stop_reason != "end_turn"
        or assessment.pending_plan_entries != 0
        or assessment.incomplete_outer_tool_calls != 0
        or assessment.unresolved_child_tool_calls <= 0
        or assessment.incomplete_tool_calls
        != assessment.unresolved_child_tool_calls
    ):
        return False
    goal = getattr(execution.result, "goal", None)
    raw_goal_status = getattr(goal, "status", "") if goal is not None else ""
    goal_status = str(getattr(raw_goal_status, "value", raw_goal_status) or "").strip().casefold()
    if goal is not None and goal_status != "completed":
        return False
    return child_lifecycle_proof.all_observed_children_terminal()


def _format_swarm_completion_notice(final_response: str) -> str:
    """Build a concise Feishu completion notice for finished Swarm runs."""
    cleaned = str(final_response or "").strip()
    if not cleaned or cleaned == UI_TEXT["mode_exec_complete"]:
        return UI_TEXT["mode_swarm_exec_complete_notice_empty"]
    if len(cleaned) > 300:
        summary = cleaned[:300].rstrip() + "…\n（完整结果见上方任务卡片）"
    else:
        summary = cleaned
    return UI_TEXT["mode_swarm_exec_complete_notice"].format(summary=summary)


def build_programming_session_callbacks(
    *,
    reply_text_fn: Callable[[str, str], object],
    add_reaction: Callable[[str, str], object],
    message_id: str,
    chat_id: str,
) -> SessionCallbacks:
    """Build callbacks for normal programming CardSession lifecycle."""
    from ...card.hooks import EmojiHook
    from ...card.session.config import SessionCallbacks

    return SessionCallbacks(
        reply_text_fn=reply_text_fn,
        hooks=(
            EmojiHook(
                add_reaction=add_reaction,
                message_id=message_id,
                chat_id=chat_id,
            ),
        ),
    )
