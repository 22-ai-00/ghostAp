"""Footer rendering: status line + progress + duration (banners rendered in body top by renderer.py)."""

from __future__ import annotations

import json
import math
import time

from src.card.state.models import CardState, ContentBlock
from src.card.tool_display import is_unhelpful_display_label
from src.card.ui_text import UI_TEXT
from src.utils.text import format_elapsed_clock

from .budget import RenderBudget
from .progress import MOBILE_SEGMENTS, render_progress_bar

_ACTIVE_TOOL_STATUSES = frozenset({"active"})
_FINAL_TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "completed_empty",
        "failed",
        "cancelled",
        "blocked",
        "archived",
        "ttl_expired",
    }
)


def _short_path_footer(path: str | None) -> str:
    """Shorten an absolute path to ~/relative for display in footer."""
    if not path:
        return ""
    from pathlib import Path as _Path
    try:
        resolved = _Path(path).expanduser().resolve()
        home = _Path.home().resolve()
        rel = resolved.relative_to(home)
        return f"~/{rel}"
    except (OSError, ValueError):
        return str(path)


def _render_context_line(state: CardState) -> str | None:
    """Build footer context line: working_dir + engine phase subtitle."""
    metadata = state.metadata
    parts: list[str] = []

    if metadata.working_dir:
        parts.append(f"📂 {_short_path_footer(metadata.working_dir)}")

    # Engine subtitle (phase info like "cycle 2/Build") from reducer
    if metadata.engine_type and state.header.subtitle:
        parts.append(state.header.subtitle)

    if not parts:
        return None
    return " · ".join(parts)


def _format_idle_timeout(seconds: int) -> str:
    """Format idle timeout seconds into human-friendly display (e.g. '30 分钟', '2 小时')."""
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600} 小时"
    if seconds >= 3600:
        hours = seconds / 3600
        return f"约 {hours:.1g} 小时"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} 分钟"


def _format_timestamp(raw: str) -> str:
    """Format a timestamp into relative time (e.g. '刚刚', '3 秒前', '5 分钟前').

    Input format: "MM-DD HH:MM" or "HH:MM:SS" (from session.py).
    Falls back to raw string if parsing fails.
    """
    if not raw:
        return raw
    import datetime

    now = time.time()
    today = datetime.date.today()

    try:
        # Try "MM-DD HH:MM" format first
        if len(raw) >= 11 and raw[5] == " ":
            month, day = int(raw[:2]), int(raw[3:5])
            hour, minute = int(raw[6:8]), int(raw[9:11])
            dt = datetime.datetime(today.year, month, day, hour, minute)
            ts = dt.timestamp()
        elif ":" in raw and len(raw) <= 8:
            # "HH:MM" or "HH:MM:SS"
            parts = raw.split(":")
            hour, minute = int(parts[0]), int(parts[1])
            second = int(parts[2]) if len(parts) > 2 else 0
            dt = datetime.datetime(today.year, today.month, today.day, hour, minute, second)
            ts = dt.timestamp()
        else:
            return raw
    except (ValueError, IndexError):
        return raw

    diff = int(now - ts)
    if diff < 0:
        diff = 0

    if diff < 5:
        return UI_TEXT["time_just_now"]
    if diff < 60:
        return UI_TEXT["time_secs_ago"].format(seconds=diff)
    minutes = diff // 60
    if minutes < 60:
        return UI_TEXT["time_mins_ago"].format(minutes=minutes)
    hours = minutes // 60
    remaining_mins = minutes % 60
    if hours < 24:
        if remaining_mins:
            return UI_TEXT["time_hours_mins_ago"].format(hours=hours, minutes=remaining_mins)
        return UI_TEXT["time_hours_ago"].format(hours=hours)
    days = hours // 24
    return UI_TEXT["time_days_ago"].format(days=days)


# Engine type → progress bar theme color
_ENGINE_PROGRESS_COLOR: dict[str, str] = {
    "deep": "violet",
    "spec": "green",
}

_TOOL_BRIEF = {
    "Read": lambda p: f"读取 {p.get('path', '...')}",
    "Edit": lambda p: f"写入 {p.get('path', '...')}",
    "Write": lambda p: f"创建 {p.get('path', '...')}",
    "Grep": lambda p: f"搜索 “{p.get('pattern') or p.get('query') or '...'}”",
    "Glob": lambda p: f"列出 {p.get('pattern', '...')}",
    "Bash": lambda p: f"执行 {_short_cmd(p.get('command') or p.get('cmd') or '')}",
}

_SUBTASK_TERMINAL_STATUS = {
    "completed": "✅ 已完成",
    "failed": "❌ 执行失败",
    "cancelled": "⚪ 已取消",
    "paused": "⏸ 已暂停",
    "archived": "📦 已封存",
    "blocked": "⛔ 已阻塞",
}


def _total_elapsed_from_session(state: CardState) -> float | None:
    """Return CardSession total elapsed seconds when a monotonic start exists."""
    metadata = state.metadata
    if metadata.frozen and metadata.frozen_total_elapsed is not None:
        return float(metadata.frozen_total_elapsed)
    if metadata.session_started_at is None:
        return None
    return max(0.0, time.monotonic() - float(metadata.session_started_at))


def _footer_elapsed_seconds(
    state: CardState,
    *,
    is_final_terminal: bool,
) -> float | None:
    """Select the live or frozen total elapsed value for the footer."""
    if is_final_terminal:
        if state.footer.duration_seconds is not None:
            return max(0.0, float(state.footer.duration_seconds))
        if state.metadata.frozen_total_elapsed is not None:
            return max(0.0, float(state.metadata.frozen_total_elapsed))
        return None

    total_elapsed = _total_elapsed_from_session(state)
    if total_elapsed is not None:
        return total_elapsed
    if state.footer.progress_started_at is None:
        return None
    return max(0.0, time.monotonic() - state.footer.progress_started_at)


def render_now_tool_hint(tool) -> str:
    """Render the v2 footer's one-line hint for the currently running tool."""
    if tool is None:
        return ""
    status = _tool_status(tool)
    if status not in _ACTIVE_TOOL_STATUSES:
        return ""
    name = _tool_name(tool)
    payload = _tool_payload(tool)
    brief_fn = _TOOL_BRIEF.get(name)
    brief = brief_fn(payload) if brief_fn else name
    if brief.casefold() == name.casefold():
        return f"⚙ **{name}**"
    return f"⚙ **{name}** · {brief}"


def render_footer(state: CardState, budget: RenderBudget | None = None) -> list[dict]:
    """Generate footer elements.

    Layout order:
      1. hr separator
      2. ⚙ tool hint (when an active tool exists on a main card)
      3. Status text + progress merged (notation size)
      4. Tool/model info line
      5. Duration (terminal states show final, running states show elapsed)

    Note: All warning banners (error/warning/info/success) are now rendered
    at body top by renderer.py for unified positioning.
    """
    elements: list[dict] = []
    is_final_terminal = state.terminal in _FINAL_TERMINAL_STATUSES
    elapsed_seconds = _footer_elapsed_seconds(
        state,
        is_final_terminal=is_final_terminal,
    )
    uses_unified_execution = (
        not state.metadata.is_subagent
        and state.metadata.engine_type is None
    )
    hide_tool_status = (
        uses_unified_execution
        and state.footer.status == "tool_running"
    )

    # Determine if we have any status/progress content to show
    has_status_content = (
        state.footer.status is not None and not hide_tool_status
    )
    # Also render footer for terminal states (tool/model/duration)
    has_meta_content = bool(
        state.metadata.tool_name
        or state.metadata.model_name
        or state.footer.duration_seconds is not None
        or elapsed_seconds is not None
    )
    # Check for active tool hint
    running_tool = (
        None if uses_unified_execution else _find_running_tool(state)
    )
    tool_hint = (
        render_now_tool_hint(running_tool)
        if running_tool and not state.metadata.is_subagent
        else ""
    )
    has_tool_hint = bool(tool_hint)
    # Check for context line (working_dir + engine phase info moved from header)
    context_line = (
        None
        if state.metadata.is_subagent
        else _render_context_line(state)
    )
    has_context = bool(context_line)

    if not has_status_content and not has_meta_content and not has_tool_hint and not has_context:
        return []

    elements.append({"tag": "hr"})

    # Context line: working_dir + engine phase info (moved from header subtitle)
    if context_line:
        elements.append(
            {"tag": "markdown", "content": context_line, "text_size": "notation"}
        )

    # ⚙ tool hint line (after context, before status)
    if tool_hint:
        elements.append(
            {"tag": "markdown", "content": tool_hint, "text_size": "notation"}
        )

    status_text = "" if hide_tool_status else (state.footer.status_text or "")
    if state.metadata.is_subagent:
        display_terminal = "archived" if state.metadata.frozen else state.terminal
        status_text = _SUBTASK_TERMINAL_STATUS.get(
            display_terminal,
            status_text,
        )
        if (
            display_terminal == "blocked"
            and state.engine_ext
            and state.engine_ext.blocked_reason
            and not is_unhelpful_display_label(state.engine_ext.blocked_reason)
        ):
            reason = state.engine_ext.blocked_reason
            if len(reason) > 60:
                reason = f"{reason[:59]}…"
            status_text = f"{status_text} · {reason}"
    show_progress = not state.metadata.is_subagent

    # Progress rendering: merge status + progress bar into one line (only when status is active)
    if has_status_content and show_progress and state.footer.progress_pct is not None:
        bar_color = _ENGINE_PROGRESS_COLOR.get(state.metadata.engine_type or "", "blue")
        mobile_segs = MOBILE_SEGMENTS if (budget is None or budget.mobile) else None
        bar_text = render_progress_bar(state.footer.progress_pct, color=bar_color, mobile_segments=mobile_segs)
        # Add semantic label prefix based on context
        prefix = ""
        if state.engine_ext and state.engine_ext.criteria_total > 0:
            prefix = f"{UI_TEXT['card_progress_criteria_label']}: "
        elif state.metadata.engine_type == "deep":
            prefix = f"{UI_TEXT['card_progress_tool_label']}: "
        # Merge status text + bar + progress count into single line
        parts = []
        if status_text:
            parts.append(status_text)
        bar_part = f"{prefix}{bar_text}"
        if state.footer.progress:
            bar_part = f"{bar_part}\u2003{state.footer.progress}"
        parts.append(bar_part)
        content = " · ".join(parts) if len(parts) > 1 else parts[0]
        elements.append(
            {"tag": "markdown", "content": content, "text_size": "notation"}
        )
    elif has_status_content and show_progress and state.footer.progress is not None:
        # Plain progress text merged with status
        if status_text:
            content = f"{status_text} · {state.footer.progress}"
        else:
            content = state.footer.progress
        elements.append(
            {"tag": "markdown", "content": content, "text_size": "notation"}
        )
    elif has_status_content and status_text:
        elements.append(
            {"tag": "markdown", "content": status_text, "text_size": "notation"}
        )
    elif not has_status_content and status_text:
        # Terminal states: show CTA text even without active status
        elements.append(
            {"tag": "markdown", "content": status_text, "text_size": "notation"}
        )

    # Tool/model info line + duration (combined into one line)
    meta_parts = []
    tool_name = state.metadata.tool_name
    model_name = state.metadata.model_name
    if tool_name and not (
        state.metadata.is_subagent
        and is_unhelpful_display_label(tool_name)
    ):
        meta_parts.append(f"🔧 {tool_name}")
    if model_name and not (
        state.metadata.is_subagent
        and is_unhelpful_display_label(model_name)
    ):
        meta_parts.append(f"🧩 {model_name}")

    if elapsed_seconds is not None:
        meta_parts.append(f"⏱ 用时 {format_elapsed_clock(elapsed_seconds)}")

    if meta_parts:
        elements.append(
            {"tag": "markdown", "content": " · ".join(meta_parts), "text_size": "notation"}
        )

    # Blocked reason as visible text below footer status
    if (
        not state.metadata.is_subagent
        and state.terminal == "blocked"
        and state.engine_ext
        and state.engine_ext.blocked_reason
    ):
        reason_text = UI_TEXT["card_lifecycle_blocked_reason_fmt"].format(reason=state.engine_ext.blocked_reason)
        elements.append(
            {"tag": "markdown", "content": reason_text, "text_size": "notation"}
        )

    # Idle timeout hint — only show when remaining time <= warn_before_seconds
    if (
        not state.metadata.is_subagent
        and state.terminal == "running"
        and state.metadata
        and state.metadata.idle_timeout_seconds
    ):
        warn_before = state.metadata.warn_before_seconds if hasattr(state.metadata, "warn_before_seconds") and state.metadata.warn_before_seconds else state.metadata.idle_timeout_seconds
        idle_remaining = getattr(state.footer, "idle_remaining_seconds", None)
        if idle_remaining is None or idle_remaining <= warn_before:
            timeout_display = _format_idle_timeout(state.metadata.idle_timeout_seconds)
            hint = UI_TEXT["card_footer_idle_timeout_hint"].format(timeout_display=timeout_display)
            elements.append(
                {"tag": "markdown", "content": hint, "text_size": "notation"}
            )

    # Last updated timestamp on non-terminal (active) cards
    if not state.terminal and state.footer.last_updated_at:
        _ts_display = _format_timestamp(state.footer.last_updated_at)
        elements.append(
            {"tag": "markdown", "content": UI_TEXT["card_footer_last_updated"].format(timestamp=_ts_display), "text_size": "notation"}
        )

    return elements


def _find_running_tool(state: CardState) -> ContentBlock | None:
    if state.terminal in _FINAL_TERMINAL_STATUSES:
        return None
    for block in reversed(state.blocks):
        if getattr(block, "kind", "") != "tool_call":
            continue
        if getattr(block, "status", "") in _ACTIVE_TOOL_STATUSES:
            return block
    return None


def _tool_name(tool) -> str:
    return str(getattr(tool, "tool_name", None) or getattr(tool, "name", None) or "tool")


def _tool_status(tool) -> str:
    return str(getattr(tool, "status", ""))


def _tool_payload(tool) -> dict:
    raw = getattr(tool, "tool_input", None)
    if raw is None:
        raw = getattr(tool, "input", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            if _tool_name(tool) == "Bash":
                return {"command": raw}
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _short_cmd(command: str) -> str:
    command = " ".join(command.split())
    if not command:
        return "..."
    return command[:80] + ("…" if len(command) > 80 else "")
