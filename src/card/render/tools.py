"""Tool call panel rendering."""

from __future__ import annotations

import json
import re

from src.card.render.atoms import RenderAtom, estimate_atom_size
from src.card.state.models import ContentBlock
from src.card.themes import PANEL_STYLES
from src.card.tool_display import (
    is_unhelpful_display_label,
    sanitize_subagent_display_text,
    sanitize_tool_failure_detail,
)
from src.card.ui_text import UI_TEXT

_MAX_OUTPUT_CHARS = 2000
_MAX_SUMMARY_CHARS = 80
_MAX_ACTIVITY_DETAILS = 12

_STATUS_ICONS = {
    "active": "⏳",
    "completed": "✅",
    "failed": "❌",
}

_BASH_TOOLS = {"bash", "shell", "run_command", "execute_command"}

_PATH_TOOLS = {"read", "write", "edit", "read_file", "write_file", "edit_file"}

_SEARCH_TOOLS = {"grep", "search", "find", "glob", "search_codebase"}

# Tools rendered as compact single-line summary (no input/output detail)
_COMPACT_TOOLS = {"task", "todowrite", "todo_write"}

_COMMAND_TOOLS = _BASH_TOOLS
_EDIT_TOOLS = {
    "write", "edit", "multi_edit", "write_file", "edit_file", "replace",
    "str_replace_editor", "create_file", "insert", "patch", "apply_diff",
    "delete_file",
}
_EXPLORE_TOOLS = {
    "read", "read_file", "grep", "search", "find", "glob",
    "search_codebase", "list", "ls", "list_dir",
    "cat", "head", "tail", "tree",
}
_SEARCH_ACTIVITY_TOOLS = {"grep", "search", "find", "glob", "search_codebase"}

_SUBAGENT_STATUS_ICONS = {
    "running": "🟠",
    "active": "🟠",
    "completed": "✅",
    "failed": "❌",
    "cancelled": "⚪",
}
_SUBAGENT_SEQUENCE_RE = re.compile(
    r"^[1-9]\d{0,5}(?:\.[A-Za-z0-9]{1,8}){0,4}$"
)


def _safe_subagent_metadata(
    value: object,
    *,
    fallback: str,
    max_chars: int,
) -> str:
    raw = str(value or "").strip()
    if (
        not raw
        or "![" in raw
        or is_unhelpful_display_label(raw)
    ):
        return fallback
    return sanitize_subagent_display_text(
        raw,
        fallback=fallback,
        max_chars=max_chars,
    )


def _safe_subagent_sequence(value: object) -> str:
    sequence = str(value or "").strip()
    return sequence if _SUBAGENT_SEQUENCE_RE.fullmatch(sequence) else ""


def _is_empty_data(value) -> bool:
    """Check if a tool input/output value is considered empty."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (dict, list)) and not value:
        return True
    return False


def render_tool_panel(block: ContentBlock) -> dict | None:
    """Render a single tool call as a collapsible_panel.

    Returns None if both tool_input and tool_output are empty.

    Design intent (AC7/AC8):
    - AC7: When both input AND output are empty → returns None (panel omitted entirely).
    - AC8: When only output is empty (input non-empty) → renders input section only,
            no empty output section appears in the card JSON.
    """
    # AC7: suppress entirely when both sides are empty
    if _is_empty_data(block.tool_input) and _is_empty_data(block.tool_output):
        return None

    icon = _STATUS_ICONS.get(block.status, "⏳")
    summary = generate_tool_summary(block)
    tool_name = block.tool_name or "tool"
    if _is_task_tool_name(tool_name) and summary:
        title_text = f"{icon} **{_format_task_label(summary)}**"
    else:
        title_text = f"{icon} **{tool_name}** — {summary}"

    expanded = _should_expand_tool(block)
    border_color = PANEL_STYLES["border_failed"] if block.status == "failed" else PANEL_STYLES["border_normal"]

    detail_content = _render_detail(block)

    return {
        "tag": "collapsible_panel",
        "expanded": expanded,
        "header": {
            "title": {"tag": "markdown", "content": title_text},
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "follow_text",
            "icon_expanded_angle": -180,
        },
        "border": {"color": border_color, "corner_radius": PANEL_STYLES["corner_radius"]},
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_standard"],
        "elements": [{"tag": "markdown", "content": detail_content}],
    }


def _should_expand_tool(block: ContentBlock) -> bool:
    """Only the latest currently-running tool may be expanded."""
    return block.status == "active" and bool(getattr(block, "is_latest_active", False))


def render_subagent_dispatch_panel(subagents: list[dict]) -> dict | None:
    """Render a compact summary panel for parallel subagent dispatch."""
    if not subagents:
        return None

    status_counts: dict[str, int] = {}
    lines: list[str] = []
    for idx, item in enumerate(subagents, start=1):
        status = str(item.get("status") or "running").strip().lower()
        status_counts[status] = status_counts.get(status, 0) + 1
        icon = _SUBAGENT_STATUS_ICONS.get(status, "🟠")
        raw_label = (
            item.get("label")
            or item.get("branch")
            or item.get("name")
            or f"子任务 {idx}"
        )
        label = sanitize_tool_failure_detail(
            raw_label,
            fallback=f"子任务 {idx}",
            max_chars=60,
        )
        tool = _safe_subagent_metadata(
            item.get("tool") or item.get("tool_name"),
            fallback="tool",
            max_chars=24,
        )
        model = _safe_subagent_metadata(
            item.get("model") or item.get("model_name"),
            fallback="",
            max_chars=60,
        )
        seq = _safe_subagent_sequence(
            item.get("sequence") or item.get("card_sequence")
        )
        seq_part = f" · #{seq}" if seq else ""
        model_part = f" · {model}" if model else ""
        lines.append(f"- {icon} {label}{seq_part} · {tool}{model_part}")
        if item.get("progress"):
            progress = _safe_subagent_metadata(
                item["progress"],
                fallback="",
                max_chars=180,
            )
            if progress:
                lines.append(f"  - 进展：{progress}")
        if status == "failed" and item.get("error"):
            error_detail = sanitize_tool_failure_detail(
                item["error"],
                fallback="",
            )
            if error_detail:
                lines.append(f"  - 原因：{error_detail}")

    running_count = status_counts.get("running", 0) + status_counts.get("active", 0)
    completed_count = status_counts.get("completed", 0)
    failed_count = status_counts.get("failed", 0)
    cancelled_count = status_counts.get("cancelled", 0)
    known_count = (
        running_count
        + completed_count
        + failed_count
        + cancelled_count
    )
    unknown_count = max(0, len(subagents) - known_count)
    summary_parts: list[str] = []
    if running_count:
        summary_parts.append(f"运行中 {running_count}")
    if completed_count:
        summary_parts.append(f"完成 {completed_count}")
    if failed_count:
        summary_parts.append(f"失败 {failed_count}")
    if cancelled_count:
        summary_parts.append(f"取消 {cancelled_count}")
    if unknown_count:
        summary_parts.append(f"未知 {unknown_count}")
    summary = " / ".join(summary_parts) if summary_parts else "暂无状态"
    header_icon = (
        "❌"
        if failed_count
        else (
            "🟠"
            if running_count or unknown_count
            else ("⚪" if cancelled_count else "✅")
        )
    )

    return {
        "tag": "collapsible_panel",
        "expanded": bool(running_count or failed_count or unknown_count),
        "header": {
            "title": {"tag": "markdown", "content": f"{header_icon} **并行子任务** · {len(subagents)} 个 · {summary}"},
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "follow_text",
            "icon_expanded_angle": -180,
        },
        "border": {
            "color": (
                PANEL_STYLES["border_failed"]
                if failed_count
                else "orange"
            ),
            "corner_radius": PANEL_STYLES["corner_radius"],
        },
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_standard"],
        "elements": [{"tag": "markdown", "content": "\n".join(lines)}],
    }


def build_subagent_dispatch_atom(subagents: list[dict]) -> RenderAtom | None:
    """Build a render atom for the parallel subagent dispatch summary."""
    panel = render_subagent_dispatch_panel(subagents)
    if panel is None:
        return None
    atom = RenderAtom(
        kind="subagent_dispatch",
        elements=[panel],
        block_id="_subagent_dispatch",
        content=str(panel),
        splittable=False,
        node_count=4,
    )
    atom.byte_size = estimate_atom_size(atom)
    return atom


def generate_tool_summary(block: ContentBlock) -> str:
    """Generate a short summary for a tool call."""
    tool_name = (block.tool_name or "").lower()
    tool_input = block.tool_input or ""

    if tool_name in _COMPACT_TOOLS:
        desc = _render_compact_detail(tool_input)
        if desc and not (_is_task_tool_name(tool_name) and _is_generic_task_summary(desc)):
            return desc
        if block.tool_summary and not (_is_task_tool_name(tool_name) and _is_generic_task_summary(block.tool_summary)):
            return block.tool_summary

    # bash → command text
    if tool_name in _BASH_TOOLS:
        line = tool_input.split("\n")[0].strip()
        if len(line) > _MAX_SUMMARY_CHARS:
            return line[:_MAX_SUMMARY_CHARS] + "…"
        return line if line else (block.tool_summary or block.tool_name or "")

    # read/write/edit → file path
    if tool_name in _PATH_TOOLS:
        path = _extract_json_field(tool_input, ("path", "file_path", "file"))
        if path:
            return path

    # grep/search → "query · path"
    if tool_name in _SEARCH_TOOLS:
        parts = []
        query = _extract_json_field(tool_input, ("query", "pattern", "keyword"))
        if query:
            parts.append(query)
        path = _extract_json_field(tool_input, ("path", "directory", "dir"))
        if path:
            parts.append(path)
        if parts:
            return " · ".join(parts)

    # generic → try to extract common fields
    for fields in [("path", "file_path", "file"), ("name", "id"), ("query",)]:
        val = _extract_json_field(tool_input, fields)
        if val:
            return val

    # Fallback
    if block.tool_summary:
        return block.tool_summary
    return block.tool_name or ""


def _activity_target(block: ContentBlock) -> str:
    tool_name = (block.tool_name or "").lower()
    tool_input = block.tool_input or ""

    if tool_name in _COMMAND_TOOLS:
        return _first_non_empty_line(tool_input) or generate_tool_summary(block)

    path = _extract_json_field(tool_input, ("path", "file_path", "file", "directory", "dir"))
    if path:
        return path

    return generate_tool_summary(block)


def _activity_detail(block: ContentBlock) -> str:
    tool_name = (block.tool_name or "").lower()
    target = _activity_target(block)
    target_part = f" `{_truncate_activity_target(target)}`" if target else ""
    label = _activity_label(block.tool_name or "工具", target)

    if getattr(block, "status", "") == "failed":
        return f"失败 {label}{target_part}"
    if tool_name in _SEARCH_ACTIVITY_TOOLS:
        return f"搜索{target_part}"
    if tool_name in _EXPLORE_TOOLS:
        return f"读取{target_part}"
    if tool_name in _EDIT_TOOLS:
        return f"编辑{target_part}"
    if tool_name in _COMMAND_TOOLS:
        return f"运行{target_part}"
    if tool_name in _COMPACT_TOOLS:
        return f"处理{target_part}"
    return f"调用 {label}{target_part}"


def _activity_label(label: str, target: str) -> str:
    """Remove target duplicated in the label when detail already renders it."""
    cleaned = str(label or "").strip() or "工具"
    target = str(target or "").strip()
    if not target:
        return cleaned
    for needle in (target, f"`{target}`"):
        cleaned = cleaned.replace(needle, " ")
    cleaned = " ".join(cleaned.split()).strip(" -:·")
    return cleaned or "工具"


def _truncate_activity_target(value: str) -> str:
    value = value.strip()
    if len(value) <= 120:
        return value
    return value[:117] + "…"


def _first_non_empty_line(value: str) -> str:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_json_field(text: str, fields: tuple[str, ...]) -> str:
    """Try to extract a field from JSON text."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for f in fields:
                if f in data and data[f]:
                    val = str(data[f])
                    if len(val) > _MAX_SUMMARY_CHARS:
                        return val[:_MAX_SUMMARY_CHARS] + "…"
                    return val
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return ""


def _truncate_output(output: str) -> str:
    """Truncate output to last _MAX_OUTPUT_CHARS with '...' prefix."""
    if len(output) <= _MAX_OUTPUT_CHARS:
        return output
    return "…" + output[-_MAX_OUTPUT_CHARS:]


def _render_detail(block: ContentBlock) -> str:
    """Render detail content based on tool type.

    AC8 enforcement: output_empty flag is passed to sub-renderers which skip the
    output section when True, ensuring no empty output block in the card JSON.
    """
    tool_name = (block.tool_name or "").lower()
    tool_input = block.tool_input or ""
    tool_output = block.tool_output or ""
    output_empty = _is_empty_data(block.tool_output)

    # Compact tools (task, todowrite): single-line description only
    if tool_name in _COMPACT_TOOLS:
        return _render_compact_detail(tool_input)

    if tool_name in _BASH_TOOLS:
        return _render_bash_detail(tool_input, tool_output, output_empty)
    return _render_generic_detail(tool_input, tool_output, output_empty)


def _render_bash_detail(tool_input: str, tool_output: str, output_empty: bool = False) -> str:
    """Render bash-specific detail."""
    parts = [f"{UI_TEXT['tool_label_command']}\n```bash\n{tool_input}\n```"]
    if not output_empty:
        output = _truncate_output(tool_output)
        parts.append(f"{UI_TEXT['tool_label_result']}\n```\n{output}\n```")
    return "\n".join(parts)


def _render_compact_detail(tool_input: str) -> str:
    """Render compact single-line detail for task/todowrite tools.

    Extracts 'description' or 'query' from JSON input, or shows first line.
    """
    desc = _extract_json_field(tool_input, ("description", "query", "content", "name"))
    if desc:
        return desc
    # Fallback: first non-empty line of raw input
    for line in tool_input.split("\n"):
        stripped = line.strip()
        if stripped and stripped not in ("{", "}", "[", "]"):
            return stripped[:_MAX_SUMMARY_CHARS] + ("…" if len(stripped) > _MAX_SUMMARY_CHARS else "")
    return ""


def _render_generic_detail(tool_input: str, tool_output: str, output_empty: bool = False) -> str:
    """Render generic tool detail."""
    parts = [f"{UI_TEXT['tool_label_input']}\n```json\n{tool_input}\n```"]
    if not output_empty:
        output = _truncate_output(tool_output)
        parts.append(f"{UI_TEXT['tool_label_output']}\n```\n{output}\n```")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Activity Digest: compact one-line summaries for turn-based rendering
# ---------------------------------------------------------------------------

def render_activity_digest_line(blocks: list[ContentBlock]) -> str:
    """Render a one-line activity digest for a group of completed/failed tool calls.

    Example output: '▣ **已探索 3 项, 已编辑 1 个文件, 已运行 2 条命令**'
    """
    if not blocks:
        return ""

    explored = 0
    searched = 0
    edited = 0
    commands = 0
    tool_calls = 0
    failed = 0

    for b in blocks:
        name = (getattr(b, "tool_name", "") or "").lower()
        if getattr(b, "status", "") == "failed":
            failed += 1
            continue
        if name in _SEARCH_ACTIVITY_TOOLS:
            searched += 1
        elif name in _EXPLORE_TOOLS:
            explored += 1
        elif name in _EDIT_TOOLS:
            edited += 1
        elif name in _COMMAND_TOOLS:
            commands += 1
        else:
            tool_calls += 1

    parts: list[str] = []
    if explored:
        parts.append(f"已探索 {explored} 项")
    if searched:
        parts.append(f"已搜索 {searched} 次")
    if edited:
        parts.append(f"已编辑 {edited} 个文件")
    if commands:
        parts.append(f"已运行 {commands} 条命令")
    if tool_calls:
        parts.append(f"{tool_calls} 次工具调用")
    if failed:
        parts.append(f"{failed} 项失败")

    return f"▣ **{', '.join(parts)}**" if parts else ""


def _activity_panel_border(blocks: list[ContentBlock]) -> str:
    if any(getattr(block, "status", "") == "failed" for block in blocks):
        return PANEL_STYLES["border_failed"]

    counts = {
        "edit": 0,
        "command": 0,
        "search": 0,
        "explore": 0,
    }
    for block in blocks:
        name = (getattr(block, "tool_name", "") or "").lower()
        if name in _EDIT_TOOLS:
            counts["edit"] += 1
        elif name in _COMMAND_TOOLS:
            counts["command"] += 1
        elif name in _SEARCH_ACTIVITY_TOOLS:
            counts["search"] += 1
        elif name in _EXPLORE_TOOLS:
            counts["explore"] += 1

    dominant = max(counts, key=counts.get)
    if counts[dominant] == 0:
        return PANEL_STYLES["border_history"]
    return {
        "edit": "green",
        "command": "wathet",
        "search": "indigo",
        "explore": PANEL_STYLES["border_history"],
    }[dominant]


def render_activity_digest_panel(blocks: list[ContentBlock]) -> dict | None:
    """Render completed tool calls as one compact aggregate panel.

    The panel intentionally summarizes inputs only. Tool outputs, especially
    file contents from read operations, stay out of the card to keep the
    Feishu message scannable and small.
    """
    summary = render_activity_digest_line(blocks)
    if not summary:
        return None

    details = [_activity_detail(block) for block in blocks[:_MAX_ACTIVITY_DETAILS]]
    remaining = len(blocks) - len(details)
    if remaining > 0:
        details.append(f"另有 {remaining} 项已折叠")
    detail_text = "\n".join(f"- {line}" for line in details if line.strip())

    return {
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {
            "title": {"tag": "markdown", "content": summary},
            "vertical_align": "center",
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "follow_text",
            "icon_expanded_angle": -180,
        },
        "border": {
            "color": _activity_panel_border(blocks),
            "corner_radius": PANEL_STYLES["corner_radius"],
        },
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_compact"],
        "elements": [
            {"tag": "markdown", "content": detail_text, "text_size": "normal"}
        ] if detail_text else [],
    }


def render_active_tool_line(block: ContentBlock) -> str:
    """Render a single running tool as a compact one-line indicator.

    Example output: '⏳ **Read** · src/card/render/tools.py'
    """
    tool_name = block.tool_name or "tool"
    summary = generate_tool_summary(block)
    if _is_task_tool_name(tool_name) and summary:
        return f"⏳ **{_format_task_label(summary)}**"
    if summary and summary != tool_name:
        return f"⏳ **{tool_name}** · {summary}"
    return f"⏳ **{tool_name}**"


def _is_task_tool_name(tool_name: str) -> bool:
    return str(tool_name or "").strip().lower() == "task"


def _is_generic_task_summary(value: str) -> bool:
    return str(value or "").strip().lower() in {"task", "任务"}


def _format_task_label(summary: str) -> str:
    summary = str(summary or "").strip()
    if not summary or _is_generic_task_summary(summary):
        return "task"
    if summary.startswith(("task:", "task：")):
        return summary.replace("task:", "task：", 1)
    return f"task：{summary}"
