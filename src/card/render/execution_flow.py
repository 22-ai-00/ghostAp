"""Chronological execution-flow projection for normal programming cards."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.card.render.tools import generate_tool_summary
from src.card.state.models import ContentBlock
from src.card.themes import PANEL_STYLES
from src.card.ui_text import UI_TEXT

MAX_HISTORY_ITEMS = 12
_SUMMARY_MAX_CHARS = 160
_COMMAND_MAX_CHARS = 1200
_COMMAND_TOOL_NAMES = {
    "bash",
    "shell",
    "run_command",
    "execute_command",
}
_COMMAND_FIRST_TOKENS = {
    "bash",
    "cargo",
    "git",
    "go",
    "make",
    "npm",
    "npx",
    "pnpm",
    "python",
    "python3",
    "pytest",
    "sh",
    "uv",
    "yarn",
    "zsh",
}


@dataclass(frozen=True)
class ExecutionFlow:
    """Rendered live action and completed history for one programming run."""

    current_element: dict | None = None
    current_text: str = ""
    history_element: dict | None = None
    history_text: str = ""
    history_count: int = 0


def render_execution_flow(
    blocks: list[ContentBlock],
    *,
    terminal: bool,
) -> ExecutionFlow:
    """Project ordered reasoning/tool blocks into current action and history."""
    process_blocks = [
        block
        for block in blocks
        if block.kind in {"reasoning", "tool_call"} and _has_visible_content(block)
    ]
    if not process_blocks:
        return ExecutionFlow()

    active_block = None if terminal else next(
        (block for block in reversed(process_blocks) if block.status == "active"),
        None,
    )
    history_blocks = [block for block in process_blocks if block is not active_block]

    current_text = _current_action_text(active_block) if active_block is not None else ""
    current_element = _current_action_element(current_text) if current_text else None
    history_text = _history_markdown(history_blocks)
    history_element = _history_panel(
        history_text,
        count=len(history_blocks),
        has_failure=any(block.status == "failed" for block in history_blocks),
    ) if history_text else None

    return ExecutionFlow(
        current_element=current_element,
        current_text=current_text,
        history_element=history_element,
        history_text=history_text,
        history_count=len(history_blocks),
    )


def _has_visible_content(block: ContentBlock) -> bool:
    if block.kind == "reasoning":
        return bool(str(block.content or "").strip())
    return bool(block.tool_name or block.tool_input or block.tool_summary)


def _one_line(value: str, *, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _tool_target(block: ContentBlock) -> str:
    summary = _one_line(generate_tool_summary(block))
    tool_name = _one_line(block.tool_name or "工具", max_chars=48)
    if summary.casefold() == tool_name.casefold():
        return ""
    return summary


def _payload_command(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    for key in ("command", "cmd", "shell_command"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            return " ".join(str(item) for item in value)
    return ""


def _looks_like_shell_command(value: str) -> bool:
    text = " ".join(str(value or "").split())
    if not text:
        return False
    first = text.split(" ", 1)[0]
    executable = first.rsplit("/", 1)[-1].casefold()
    return (
        executable in _COMMAND_FIRST_TOKENS
        or first.startswith(("./", "/"))
        or ("=" in first and not first.startswith("="))
        or any(marker in text for marker in (" && ", " || ", " | ", "$(", "; "))
    )


def _tool_command(block: ContentBlock) -> str:
    raw_input = str(block.tool_input or "").strip()
    payload_command = _payload_command(raw_input)
    if payload_command:
        return _one_line(payload_command, max_chars=_COMMAND_MAX_CHARS)

    tool_name = _one_line(block.tool_name or "", max_chars=_COMMAND_MAX_CHARS)
    raw_command = _one_line(raw_input, max_chars=_COMMAND_MAX_CHARS)
    if str(block.tool_name or "").strip().casefold() in _COMMAND_TOOL_NAMES:
        return raw_command
    if not raw_command:
        return tool_name if _looks_like_shell_command(tool_name) else ""
    normalized_name = tool_name.casefold()
    normalized_command = raw_command.casefold()
    same_evidence = (
        normalized_name == normalized_command
        or normalized_name in normalized_command
        or normalized_command in normalized_name
    )
    if same_evidence and (
        _looks_like_shell_command(tool_name)
        or _looks_like_shell_command(raw_command)
    ):
        return raw_command
    return ""


def _command_action_label(command: str) -> str:
    normalized = f" {' '.join(command.casefold().split())} "
    if " pytest " in normalized or " unittest " in normalized:
        return "运行测试"
    if " ruff " in normalized:
        return "检查代码规范"
    if " mypy " in normalized or " pyright " in normalized:
        return "检查类型"
    if any(
        token in normalized
        for token in (" git push ", " git pull ", " git fetch ")
    ):
        return "同步代码"
    if " git " in normalized:
        return "检查代码变更"
    return "执行命令"


def _current_action_text(block: ContentBlock) -> str:
    if block.kind == "reasoning":
        summary = _one_line(block.content or "")
        return UI_TEXT["execution_current_analysis"].format(summary=summary)

    command = _tool_command(block)
    if command:
        return (
            f"⚙️ **{_command_action_label(command)}**\n"
            f"🔧 **执行中**：{command}"
        )

    tool_name = _one_line(block.tool_name or "工具", max_chars=48)
    target = _tool_target(block)
    suffix = f" · {target}" if target else ""
    return UI_TEXT["execution_current_tool"].format(tool_name=tool_name, suffix=suffix)


def _history_line(block: ContentBlock) -> str:
    if block.kind == "reasoning":
        return f"🧠 分析 · {_one_line(block.content or '')}"

    icon = {"active": "⏳", "failed": "❌"}.get(block.status, "✅")
    command = _tool_command(block)
    if command:
        return (
            f"{icon} {_command_action_label(command)}"
            f" · {_one_line(command)}"
        )
    tool_name = _one_line(block.tool_name or "工具", max_chars=48)
    target = _tool_target(block)
    suffix = f" · {target}" if target else ""
    return f"{icon} {tool_name}{suffix}"


def _history_markdown(blocks: list[ContentBlock]) -> str:
    hidden_count = max(0, len(blocks) - MAX_HISTORY_ITEMS)
    visible_blocks = blocks[-MAX_HISTORY_ITEMS:]
    lines: list[str] = []
    if hidden_count:
        lines.append(UI_TEXT["execution_history_folded"].format(count=hidden_count))
    lines.extend(_history_line(block) for block in visible_blocks)
    return "\n".join(f"- {line}" for line in lines if line.strip())


def _current_action_element(content: str) -> dict:
    return {
        "tag": "column_set",
        "flex_mode": "stretch",
        "background_style": "wathet",
        "columns": [
            {
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "vertical_align": "center",
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content,
                        "text_align": "left",
                    }
                ],
            }
        ],
    }


def _history_panel(content: str, *, count: int, has_failure: bool) -> dict:
    return {
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {
            "title": {
                "tag": "markdown",
                "content": UI_TEXT["execution_history_title"].format(count=count),
            },
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
            "color": PANEL_STYLES["border_failed"] if has_failure else PANEL_STYLES["border_history"],
            "corner_radius": PANEL_STYLES["corner_radius"],
        },
        "vertical_spacing": PANEL_STYLES["vertical_spacing"],
        "padding": PANEL_STYLES["padding_compact"],
        "elements": [
            {
                "tag": "markdown",
                "content": content,
                "text_size": "normal",
                "text_align": "left",
            }
        ],
    }
