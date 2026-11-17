"""Native slash command metadata, prompt formatting, and SMART-mode routing helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .execution_instructions import AUTONOMOUS_EXECUTION_PROMPT
from .models import ACPAvailableCommand

# Commands that MUST NEVER be invoked from chat/cards to protect host CLI auth/process state.
BLOCKED_NATIVE_COMMANDS: frozenset[str] = frozenset({"logout"})

# Zero-arg native slash commands whose backend parser expects a bare command block
# without prepended or appended autonomous execution instructions.
ZERO_ARG_NATIVE_COMMANDS: frozenset[str] = frozenset(
    {
        "compact",
        "status",
        "mcp",
        "skills",
        "review",
        "init",
        "cost",
        "doctor",
        "memory",
        "agents",
        "diff",
        "config",
    }
)

# Priority order aligned with Workflow/GhostAP recommendations when multiple available
# tools support the same native command and no project tool preference is pinned.
DEFAULT_TOOL_PRIORITY: tuple[str, ...] = (
    "agy",
    "codex",
    "traex",
    "claude",
    "coco",
    "grok",
    "dsh",
    "aiden",
    "gemini",
    "claude_w",
    "codex_w",
)

# Featured commands highlighted first on ready cards and /cmds menus.
FEATURED_COMMAND_PRIORITY: tuple[str, ...] = (
    "swarm",
    "boost",
    "plan",
    "compact",
    "review",
    "review-branch",
    "review-commit",
    "grill-me",
    "teamwork",
    "goal",
    "skills",
    "mcp",
    "init",
    "status",
    "learn",
)

# Command names must start with an ASCII letter and contain no second '/'.
# This rejects filesystem paths like `/usr/bin/python` or `/data00/...`.
_SLASH_CMD_RE = re.compile(r"^/([A-Za-z][A-Za-z0-9_.:-]*)(?:\s+(.*))?$", re.DOTALL)

STATIC_TOOL_NATIVE_COMMANDS: dict[str, tuple[ACPAvailableCommand, ...]] = {
    "agy": (
        ACPAvailableCommand("swarm", "启动多 Agent 协同 Swarm", "<任务>"),
        ACPAvailableCommand("boost", "复杂编程/研究深度推进", "<需求>"),
        ACPAvailableCommand("plan", "分步架构规划与落地检查", "<目标>"),
        ACPAvailableCommand("compact", "压缩当前会话上下文", ""),
        ACPAvailableCommand("teamwork", "多智能体协作流程", "<任务>"),
        ACPAvailableCommand("goal", "长时自主推进直到目标完成", "<目标>"),
        ACPAvailableCommand("grill-me", "交互式深度追问敲定方案", "<方案>"),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
        ACPAvailableCommand("learn", "沉淀经验到持久化规则", ""),
        ACPAvailableCommand("browser", "浏览器自动化与网页检索", "<目标>"),
        ACPAvailableCommand("schedule", "定时或计划任务", "<任务>"),
    ),
    "codex": (
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("review-branch", "审查分支相对主分支改动", "<分支>"),
        ACPAvailableCommand("review-commit", "审查指定 Commit 变更", "<commit>"),
        ACPAvailableCommand("plan", "生成分步实施计划", "<目标>"),
        ACPAvailableCommand("compact", "压缩上下文窗口", ""),
        ACPAvailableCommand("status", "查看 Codex 内部会话状态", ""),
        ACPAvailableCommand("mcp", "查看已配置的 MCP 服务", ""),
        ACPAvailableCommand("skills", "查看可用 Skills（支持 $skill）", ""),
        ACPAvailableCommand("init", "初始化 AGENTS.md 指南", ""),
        ACPAvailableCommand("goal", "自主推进长时目标", "<目标>"),
    ),
    "traex": (
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("review-branch", "审查分支改动", "<分支>"),
        ACPAvailableCommand("review-commit", "审查指定 Commit 变更", "<commit>"),
        ACPAvailableCommand("compact", "压缩当前会话上下文", ""),
        ACPAvailableCommand("init", "初始化项目上下文指南", ""),
        ACPAvailableCommand("goal", "长时自主推进目标", "<目标>"),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
        ACPAvailableCommand("mcp", "查看 MCP 服务配置", ""),
    ),
    "coco": (
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("review-branch", "审查分支改动", "<分支>"),
        ACPAvailableCommand("review-commit", "审查指定 Commit 变更", "<commit>"),
        ACPAvailableCommand("compact", "压缩当前会话上下文", ""),
        ACPAvailableCommand("init", "初始化项目上下文指南", ""),
        ACPAvailableCommand("goal", "长时自主推进目标", "<目标>"),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
        ACPAvailableCommand("mcp", "查看 MCP 服务配置", ""),
    ),
    "claude": (
        ACPAvailableCommand("compact", "压缩当前会话上下文", "[重点]"),
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("plan", "规划架构与执行方案", "<目标>"),
        ACPAvailableCommand("init", "初始化 CLAUDE.md 项目指南", ""),
        ACPAvailableCommand("mcp", "查看 MCP 服务配置", ""),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
    ),
    "gemini": (
        ACPAvailableCommand("compact", "压缩当前上下文", ""),
        ACPAvailableCommand("plan", "制定分步实现计划", "<目标>"),
        ACPAvailableCommand("mcp", "查看 MCP 服务配置", ""),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
        ACPAvailableCommand("init", "初始化项目指令", ""),
    ),
    "aiden": (
        ACPAvailableCommand("compact", "压缩当前上下文", ""),
        ACPAvailableCommand("plan", "制定实现计划", "<目标>"),
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
    ),
    "grok": (
        ACPAvailableCommand("compact", "压缩当前上下文", ""),
        ACPAvailableCommand("plan", "制定实现计划", "<目标>"),
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
    ),
    "dsh": (
        ACPAvailableCommand("compact", "压缩当前上下文", ""),
        ACPAvailableCommand("plan", "制定实现计划", "<目标>"),
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
    ),
    "claude_w": (
        ACPAvailableCommand("compact", "压缩当前会话上下文", "[重点]"),
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("plan", "规划架构与执行方案", "<目标>"),
        ACPAvailableCommand("init", "初始化 CLAUDE.md 项目指南", ""),
        ACPAvailableCommand("mcp", "查看 MCP 服务配置", ""),
        ACPAvailableCommand("skills", "查看可用 Skills", ""),
    ),
    "codex_w": (
        ACPAvailableCommand("review", "代码审查当前改动", "[关注点]"),
        ACPAvailableCommand("compact", "压缩上下文窗口", ""),
        ACPAvailableCommand("plan", "生成分步实施计划", "<目标>"),
        ACPAvailableCommand("mcp", "查看已配置的 MCP 服务", ""),
        ACPAvailableCommand("skills", "查看可用 Skills（支持 $skill）", ""),
    ),
}


def normalize_native_tool_key(tool_name: str | None) -> str:
    """Normalize tool/mode names (e.g. ``claude-w`` -> ``claude_w``)."""
    return str(tool_name or "").strip().casefold().replace("-", "_")


@dataclass(frozen=True)
class SmartNativeRouteResult:
    """Resolution outcome when checking a slash command in SMART mode."""

    status: str  # "matched" | "blocked" | "unavailable" | "none"
    command_name: str = ""
    tool_name: str = ""
    matched_tools: tuple[str, ...] = ()


def parse_slash_command(text: str | None) -> tuple[str, str] | None:
    """Parse ``/command [args]`` while rejecting filesystem paths such as ``/usr/bin``."""
    stripped = str(text or "").strip()
    if not stripped.startswith("/") or stripped.startswith("//"):
        return None
    match = _SLASH_CMD_RE.match(stripped)
    if not match:
        return None
    name = match.group(1).strip().casefold()
    args = (match.group(2) or "").strip()
    return name, args


def is_blocked_native_command(command_or_text: str | None) -> bool:
    """Return True if *command_or_text* targets a blocked native command (e.g. ``/logout``)."""
    raw = str(command_or_text or "").strip()
    if not raw:
        return False
    parsed = parse_slash_command(raw)
    if parsed is not None:
        return parsed[0] in BLOCKED_NATIVE_COMMANDS
    return raw.lstrip("/").strip().split(maxsplit=1)[0].casefold() in BLOCKED_NATIVE_COMMANDS


def normalize_acp_available_commands(
    raw_commands: Iterable[Any] | None,
) -> tuple[ACPAvailableCommand, ...]:
    """Normalize schema/dict command descriptors and exclude blocked commands."""
    if not raw_commands:
        return ()
    result: list[ACPAvailableCommand] = []
    seen: set[str] = set()
    for item in raw_commands:
        if isinstance(item, ACPAvailableCommand):
            raw_name = item.name
            description = item.description
            input_hint = item.input_hint
        elif isinstance(item, dict):
            raw_name = item.get("name", "")
            description = item.get("description", "")
            raw_input = item.get("input")
            if isinstance(raw_input, dict):
                input_hint = raw_input.get("hint", "")
            else:
                input_hint = item.get("input_hint", "")
        else:
            raw_name = getattr(item, "name", "")
            description = getattr(item, "description", "")
            raw_input = getattr(item, "input", None)
            root_input = getattr(raw_input, "root", raw_input)
            input_hint = getattr(root_input, "hint", "") if root_input is not None else ""

        name = str(raw_name or "").strip().lstrip("/").strip()
        lowered = name.casefold()
        if not name or lowered in BLOCKED_NATIVE_COMMANDS or lowered in seen:
            continue
        seen.add(lowered)
        result.append(
            ACPAvailableCommand(
                name=name,
                description=str(description or "").strip(),
                input_hint=str(input_hint or "").strip(),
            )
        )
    return tuple(result)


def get_tool_native_commands(
    tool_name: str,
    *,
    live_commands: Iterable[Any] | None = None,
) -> tuple[tuple[ACPAvailableCommand, ...], str]:
    """Return ``(commands, source)`` where *source* is ``'live'`` or ``'static'``."""
    normalized_live = normalize_acp_available_commands(live_commands)
    if normalized_live:
        return normalized_live, "live"
    tool_key = normalize_native_tool_key(tool_name)
    static_cmds = normalize_acp_available_commands(STATIC_TOOL_NATIVE_COMMANDS.get(tool_key, ()))
    return static_cmds, "static"


def get_featured_native_commands(
    tool_name: str,
    *,
    live_commands: Iterable[Any] | None = None,
    limit: int = 6,
) -> tuple[ACPAvailableCommand, ...]:
    """Return top featured native commands for *tool_name*, keeping priority order first."""
    commands, _source = get_tool_native_commands(tool_name, live_commands=live_commands)
    if not commands or limit <= 0:
        return ()
    by_name = {cmd.name.casefold(): cmd for cmd in commands}
    featured: list[ACPAvailableCommand] = []
    used: set[str] = set()
    for preferred in FEATURED_COMMAND_PRIORITY:
        cmd = by_name.get(preferred)
        if cmd is not None:
            featured.append(cmd)
            used.add(preferred)
            if len(featured) >= limit:
                return tuple(featured)
    for cmd in commands:
        key = cmd.name.casefold()
        if key not in used:
            featured.append(cmd)
            used.add(key)
            if len(featured) >= limit:
                break
    return tuple(featured)


def get_zero_arg_quick_commands(
    commands: Sequence[ACPAvailableCommand],
    *,
    limit: int = 6,
) -> tuple[ACPAvailableCommand, ...]:
    """Select safe zero-arg commands for one-click card buttons."""
    if not commands or limit <= 0:
        return ()
    zero_arg = [cmd for cmd in commands if not cmd.input_hint.strip()]
    by_name = {cmd.name.casefold(): cmd for cmd in zero_arg}
    selected: list[ACPAvailableCommand] = []
    seen: set[str] = set()
    for preferred in FEATURED_COMMAND_PRIORITY:
        cmd = by_name.get(preferred)
        if cmd is not None:
            selected.append(cmd)
            seen.add(preferred)
            if len(selected) >= limit:
                return tuple(selected)
    for cmd in zero_arg:
        key = cmd.name.casefold()
        if key not in seen:
            selected.append(cmd)
            seen.add(key)
            if len(selected) >= limit:
                break
    return tuple(selected)


def is_zero_arg_native_command(command_name: str) -> bool:
    """Return True if *command_name* belongs to the zero-arg native slash command set."""
    return str(command_name or "").strip().lstrip("/").casefold() in ZERO_ARG_NATIVE_COMMANDS


def is_pure_zero_arg_native_slash(text: str | None) -> bool:
    """Return True when *text* is a bare zero-arg native slash command without arguments."""
    parsed = parse_slash_command(text)
    if parsed is None:
        return False
    name, args = parsed
    return not args and is_zero_arg_native_command(name)


def build_acp_prompt_texts(text: str) -> list[str]:
    """Order ACP prompt text blocks so native ``/commands`` always stay at ``prompt[0]``.

    - Pure zero-arg native slash commands (e.g. ``/compact``, ``/status``, ``/mcp``,
      ``/skills``): ``[stripped_text]`` so providers treat them purely as control/state commands.
    - Parameterized native slash commands (e.g. ``/swarm fix``, ``/review-branch main``):
      ``[stripped_text, AUTONOMOUS_EXECUTION_PROMPT]`` so ``prompt[0]`` starts with ``/``.
    - Regular prompts: ``[AUTONOMOUS_EXECUTION_PROMPT, text]``.
    """
    stripped = str(text or "").strip()
    parsed = parse_slash_command(stripped)
    if parsed is not None:
        if is_pure_zero_arg_native_slash(stripped):
            return [stripped]
        return [stripped, AUTONOMOUS_EXECUTION_PROMPT]
    return [AUTONOMOUS_EXECUTION_PROMPT, text]


def resolve_smart_mode_native_command(
    text: str | None,
    *,
    preferred_tool: str | None = None,
    available_tools: Iterable[str] | None = None,
    live_commands_by_tool: dict[str, Iterable[Any]] | None = None,
) -> SmartNativeRouteResult:
    """Resolve an unknown ``/command`` in SMART mode to an available programming tool."""
    parsed = parse_slash_command(text)
    if parsed is None:
        return SmartNativeRouteResult(status="none")

    cmd_name, _args = parsed
    if cmd_name in BLOCKED_NATIVE_COMMANDS:
        return SmartNativeRouteResult(status="blocked", command_name=cmd_name)

    live_map = live_commands_by_tool or {}
    matching_tools: list[str] = []
    for tool in DEFAULT_TOOL_PRIORITY:
        commands, _ = get_tool_native_commands(tool, live_commands=live_map.get(tool))
        if any(cmd.name.casefold() == cmd_name for cmd in commands):
            matching_tools.append(tool)

    if not matching_tools:
        return SmartNativeRouteResult(status="none", command_name=cmd_name)

    avail_set = (
        {normalize_native_tool_key(t) for t in available_tools if str(t).strip()}
        if available_tools is not None
        else set(STATIC_TOOL_NATIVE_COMMANDS.keys())
    )
    available_matches = [tool for tool in matching_tools if tool in avail_set]
    if not available_matches:
        return SmartNativeRouteResult(
            status="unavailable",
            command_name=cmd_name,
            matched_tools=tuple(matching_tools),
        )

    preferred = normalize_native_tool_key(preferred_tool)
    if preferred and preferred in available_matches:
        selected = preferred
    elif len(available_matches) == 1:
        selected = available_matches[0]
    else:
        # When multiple available tools share a command (e.g. /review, /compact),
        # prefer the first tool in priority order without adding a confirmation gate.
        selected = available_matches[0]

    return SmartNativeRouteResult(
        status="matched",
        command_name=cmd_name,
        tool_name=selected,
        matched_tools=tuple(available_matches),
    )
