"""WorkflowProgressRenderer — renders workflow progress tree for Feishu cards."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from collections.abc import Mapping
from typing import Any

from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.state.models import (
    CardMetadata,
    CardState,
    ContentBlock,
    HeaderState,
)
from src.card.state.runtime_stats import RuntimeStats
from src.card.tool_display import sanitize_tool_failure_detail
from src.utils.text import format_duration, format_elapsed_clock

from .errors import _strip_internal_details
from .models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    SubagentProgress,
    SubagentStatus,
    WorkflowProject,
    WorkflowStatus,
)
from .result_brief import (
    BriefItem,
    BriefSeverity,
    BriefVerdict,
    WorkflowResultBrief,
    build_result_brief,
    fit_result_brief,
)

# ---------------------------------------------------------------------------
# Internal: string helpers
# ---------------------------------------------------------------------------

# Keep phase/agent labels readable on narrow mobile screens without
# truncating important context (e.g. a task identifier at the end of a
# long title). A middle ellipsis keeps both the leading description and
# the trailing identifier visible.
_LABEL_TRUNCATION_LIMIT = 40
_ELLIPSIS = "…"


def _middle_ellipsis(text: str, limit: int = _LABEL_TRUNCATION_LIMIT) -> str:
    """Return ``text`` with middle characters replaced by an ellipsis when
    it exceeds ``limit`` characters. Preserves the head and tail so both
    human-readable descriptions and trailing identifiers survive.

    Examples::

        _middle_ellipsis("code-review: verify payment-gateway auth flow")
        # → "code-review: ver…t flow"  (when limit == 24)
        _middle_ellipsis("short")  # → "short"
    """
    if not text:
        return text
    if len(text) <= limit:
        return text
    # Reserve room for the ellipsis itself.
    available = max(limit - len(_ELLIPSIS), 4)
    head = available // 2 + available % 2
    tail = available // 2
    return f"{text[:head]}{_ELLIPSIS}{text[-tail:]}"


def _escape_md(text: str) -> str:
    """Escape markdown special characters in user-supplied text."""
    for ch in ("*", "_", "`", "|", "[", "]", "~"):
        text = text.replace(ch, "\\" + ch)
    return text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel marker list used by lint-level defensive checks.
# An empty tuple means "no markers configured → no checks applied". Tests can
# monkey-patch this to inject sentinel values and verify the defensive gate.
_AGENT_OUTPUT_FORBIDDEN_MARKERS: tuple[str, ...] = ()
_WORKFLOW_PAGE_KEY_FIELD = "_workflow_page_key"

STATUS_ICONS: dict[AgentStatus, str] = {
    AgentStatus.PENDING: "\u23f3",
    AgentStatus.RUNNING: "\U0001f504",
    AgentStatus.DONE: "\u2705",
    AgentStatus.FAILED: "\u274c",
    AgentStatus.CACHED: "\U0001f4e6",
    AgentStatus.CANCELLED: "⏹️",
}

WORKFLOW_STATUS_ICONS: dict[WorkflowStatus, str] = {
    WorkflowStatus.IDLE: "\u23f3",
    WorkflowStatus.GENERATING_SCRIPT: "\U0001f504",
    WorkflowStatus.RUNNING: "\U0001f504",
    WorkflowStatus.COMPLETED: "\u2705",
    WorkflowStatus.FAILED: "\u274c",
    WorkflowStatus.CANCELLED: "\u274c",
}

_CARD_MAX_BYTES = 27 * 1024  # Shared card payload budget with safety margin.
_SUBAGENT_DISPLAY_LIMIT = 12

_SUBAGENT_STATUS_META: dict[SubagentStatus, tuple[str, str]] = {
    SubagentStatus.RUNNING: ("🟠", "执行中"),
    SubagentStatus.COMPLETED: ("✅", "已完成"),
    SubagentStatus.FAILED: ("❌", "失败"),
    SubagentStatus.CANCELLED: ("⚪", "已取消"),
}


# ---------------------------------------------------------------------------
# Defensive (lint-level) helpers — no-op under normal operation
# ---------------------------------------------------------------------------


def _card_text_for_agent_output(
    elements: list[dict],
    forbidden_markers: tuple[str, ...],
) -> None:
    """Scan ``elements`` recursively for forbidden marker strings.

    Iterates through the element list and any nested ``dict``/``list`` values,
    looking for ``text`` / ``content`` string fields that contain any of the
    ``forbidden_markers``. If a match is found, raises
    :class:`RuntimeError` with the message ``"card leaked agent output"``.

    When ``forbidden_markers`` is empty, the function is a no-op — this is the
    normal production configuration. Tests monkey-patch
    ``_AGENT_OUTPUT_FORBIDDEN_MARKERS`` to inject sentinel strings and verify
    the gate trips when agent output accidentally leaks into card text.
    """
    if not forbidden_markers:
        return

    stack: list[Any] = list(elements)
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in ("text", "content") and isinstance(value, str):
                    for marker in forbidden_markers:
                        if marker and marker in value:
                            raise RuntimeError("card leaked agent output")
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)


# ---------------------------------------------------------------------------
# Helper builders for Feishu card elements
# ---------------------------------------------------------------------------


def _md_element(content: str, *, text_align: str | None = None) -> dict[str, Any]:
    """Create a markdown text element."""
    element: dict[str, Any] = {"tag": "markdown", "content": content}
    if text_align is not None:
        element["text_align"] = text_align
    return element


def _hr_element() -> dict[str, Any]:
    """Create a horizontal rule divider."""
    return {"tag": "hr"}


def _collapsible_panel(
    header: str | dict[str, Any],
    elements: list[dict[str, Any]],
    *,
    expanded: bool = False,
    template: str | None = None,
) -> dict[str, Any]:
    """Wrap elements in a Feishu collapsible_panel.

    The ``expanded`` flag matches the Feishu schema and the convention used
    by the rest of the codebase (see ``card/render/tools.py``). When ``expanded=True`` the panel is shown
    open on first render; when ``expanded=False`` it is collapsed by default.
    """
    if isinstance(header, str):
        header_obj: dict[str, Any] = {
            "title": {"tag": "plain_text", "content": header},
        }
    else:
        header_obj = header
        header_obj.pop("template", None)
    panel = {
        "tag": "collapsible_panel",
        "header": header_obj,
        "elements": elements,
        "expanded": expanded,
    }
    if template is not None:
        panel["border"] = {"color": template, "corner_radius": "8px"}
    return panel


def _column_set(columns: list[dict[str, Any]], *, flex_mode: str = "none") -> dict[str, Any]:
    """Create a column_set layout element."""
    return {
        "tag": "column_set",
        "flex_mode": flex_mode,
        "columns": columns,
    }


def _column(
    elements: list[dict[str, Any]],
    *,
    weight: int = 1,
    width: str = "weighted",
    vertical_align: str | None = None,
) -> dict[str, Any]:
    """Create a single column inside a column_set."""
    column: dict[str, Any] = {
        "tag": "column",
        "width": width,
        "weight": weight,
        "elements": elements,
    }
    if vertical_align is not None:
        column["vertical_align"] = vertical_align
    return column


def _pct(used: int, total: int) -> str:
    """Calculate percentage string: "63%"."""
    if total <= 0:
        return "0%"
    return f"{int(min(used / total, 1.0) * 100)}%"


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h{mins}m"


def _format_tokens(tokens: int) -> str:
    """Format token count with K/M suffix."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.0f}K"
    return str(tokens)


def _render_subagent_lines(subagents: list[SubagentProgress]) -> str:
    """Render bounded, explicitly non-authoritative ACP child observations."""
    if not subagents:
        return ""

    source_ids = tuple(item.source_id for item in subagents if item.source_id)
    indexed = list(enumerate(subagents, start=1))
    if len(indexed) > _SUBAGENT_DISPLAY_LIMIT:
        important = [
            item
            for item in indexed
            if item[1].status
            in {SubagentStatus.RUNNING, SubagentStatus.FAILED, SubagentStatus.CANCELLED}
        ]
        selected = important[:_SUBAGENT_DISPLAY_LIMIT]
        selected_ids = {index for index, _ in selected}
        if len(selected) < _SUBAGENT_DISPLAY_LIMIT:
            remaining = [item for item in indexed if item[0] not in selected_ids]
            selected.extend(remaining[-(_SUBAGENT_DISPLAY_LIMIT - len(selected)):])
        indexed = sorted(selected, key=lambda item: item[0])

    counts = {status: 0 for status in SubagentStatus}
    for child in subagents:
        counts[child.status] += 1
    summary_parts = [
        f"观测运行 {counts[SubagentStatus.RUNNING]}",
        f"观测完成 {counts[SubagentStatus.COMPLETED]}",
        f"观测失败 {counts[SubagentStatus.FAILED]}",
        f"观测取消 {counts[SubagentStatus.CANCELLED]}",
    ]
    lines = [
        f"    ↳ ACP 内部 Agent 观测（非权威，不参与终态判定）· {len(subagents)} 个 · "
        + " / ".join(part for part in summary_parts if not part.endswith(" 0"))
    ]
    for index, child in indexed:
        icon, status_text = _SUBAGENT_STATUS_META[child.status]
        model = sanitize_tool_failure_detail(
            child.model,
            fallback="",
            max_chars=60,
            opaque_ids=source_ids,
        )
        model_text = model or "默认模型"
        progress = sanitize_tool_failure_detail(
            _strip_internal_details(child.progress),
            fallback="",
            max_chars=180,
            opaque_ids=source_ids,
        )
        authority = "" if child.authoritative else "观测"
        latest = " ".join((progress or "暂无最新操作").split())
        lines.append(
            f"    - {icon} ACP 子 Agent {index} · {authority}{status_text} · "
            f"{model_text} · 最新操作：{latest}"
        )
    hidden = len(subagents) - len(indexed)
    if hidden > 0:
        lines.append(f"    - 另有 {hidden} 个非权威观测已折叠，不影响主 Agent 终态")
    return "\n" + "\n".join(lines)


def _unicode_progress_bar(ratio: float, *, length: int = 20) -> str:
    """Render a Unicode block progress bar.

    Args:
        ratio: Progress ratio (0.0 to 1.0), clamped automatically.
        length: Total number of block characters (default: 20).

    Returns:
        Progress bar string like "┃████████████░░░░░░┃"
    """
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * length)
    empty = length - filled
    return f"┃{'█' * filled}{'░' * empty}┃"


# ---------------------------------------------------------------------------
# WorkflowProgressRenderer
# ---------------------------------------------------------------------------


def _binding_value(binding: Any, *names: str, default: Any = None) -> Any:
    if isinstance(binding, dict):
        for name in names:
            if name in binding and binding[name] is not None:
                return binding[name]
        return default
    for name in names:
        value = getattr(binding, name, None)
        if value is not None:
            return value
    return default


def _binding_model_parts(binding: Any) -> tuple[str, str, str | None, str | None]:
    """Return a display-only tool/model/profile/effort tuple without duplication."""
    tool_name = str(
        _binding_value(binding, "tool_name", "toolName", "tool", default="?")
    )
    profile_value = _binding_value(binding, "profile")
    effort_value = _binding_value(binding, "effort")
    profile = str(profile_value) if profile_value else None
    effort = str(effort_value) if effort_value else None
    raw_model = str(
        _binding_value(
            binding,
            "model_name",
            "modelName",
            "model",
            default="default",
        )
        or "default"
    )

    # ACP model selections may encode profile/effort in model_name while also
    # exposing the structured fields. Strip only the exact structured suffix;
    # model families that legitimately contain slashes remain untouched.
    structured_suffix = "/".join(value for value in (profile, effort) if value)
    suffix = f"/{structured_suffix}" if structured_suffix else ""
    model_name = raw_model[: -len(suffix)] if suffix and raw_model.endswith(suffix) else raw_model
    return tool_name, model_name or "default", profile, effort


def _binding_config_markdown(binding: Any) -> str:
    tool_name, model_name, profile, effort = _binding_model_parts(binding)
    parts = [f"`{tool_name}`", f"模型 `{model_name}`"]
    if profile:
        parts.append(f"Profile `{profile}`")
    if effort:
        parts.append(f"Effort `{effort}`")
    return " · ".join(parts)


class WorkflowGenerationRenderer:
    """Render the immutable Agent Pool while the workflow script is generated."""

    def __init__(
        self,
        *,
        requirement: str,
        agent_pool: Any,
        orchestrator_agent_id: str,
        orchestrator_was_auto: bool = False,
    ) -> None:
        self.requirement = str(requirement or "")
        self.agent_pool = tuple(agent_pool or ())
        self.orchestrator_agent_id = str(orchestrator_agent_id or "")
        self.orchestrator_was_auto = bool(orchestrator_was_auto)

    @staticmethod
    def _one_line(value: Any, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def render(
        self,
        *,
        current_activity: str | None = None,
        elapsed_seconds: int = 0,
        terminal_status: str | None = None,
    ) -> dict[str, Any]:
        orchestrator = next(
            (
                binding
                for binding in self.agent_pool
                if str(_binding_value(binding, "agent_id", "agentId", default=""))
                == self.orchestrator_agent_id
            ),
            None,
        )
        orchestrator_label = (
            f"Auto → {self.orchestrator_agent_id}"
            if self.orchestrator_was_auto
            else self.orchestrator_agent_id
        ) or "未解析"
        orchestrator_name = str(
            _binding_value(
                orchestrator,
                "display_name",
                "displayName",
                default=_binding_value(orchestrator, "tool_name", "toolName", default=""),
            )
            or ""
        )
        is_cancelled = str(terminal_status or "").lower() == "cancelled"
        default_activity = (
            "Workflow 任务已由用户停止"
            if is_cancelled
            else
            f"{self.orchestrator_agent_id} 正在生成并验证 Workflow 编排脚本"
            if self.orchestrator_agent_id
            else "正在生成并验证 Workflow 编排脚本"
        )
        activity = self._one_line(current_activity or default_activity, limit=240)
        requirement = self._one_line(self.requirement, limit=600)
        elapsed = max(0, int(elapsed_seconds or 0))
        elapsed_text = format_duration(elapsed)

        pool_lines: list[str] = []
        for binding in self.agent_pool:
            agent_id = str(
                _binding_value(binding, "agent_id", "agentId", default="?")
            )
            display_name = str(
                _binding_value(
                    binding,
                    "display_name",
                    "displayName",
                    default=_binding_value(binding, "tool_name", "toolName", default="?"),
                )
            )
            role = " · 主编排" if agent_id == self.orchestrator_agent_id else ""
            pool_lines.append(
                f"- **{agent_id}** · {display_name}{role}\n  {_binding_config_markdown(binding)}"
            )

        status_content = (
            "**CANCELLED** · Workflow 已取消\n"
            f"**最终状态** · {activity}\n"
            f"⏱ 结束于启动后 {elapsed_text}"
            if is_cancelled
            else
            "**当前阶段** · 生成并验证编排脚本\n"
            f"**当前操作** · {activity}\n"
            f"⏱ 已等待 {elapsed_text}\n\n"
            "① 生成脚本（进行中） → ② 验证计划 → ③ 执行节点 → ④ 汇总结果"
        )
        elements: list[dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": status_content,
            },
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": f"**任务**\n{requirement or '未提供任务描述'}",
            },
            {"tag": "hr"},
            {
                "tag": "markdown",
                "content": (
                    f"**主编排 Agent** · {orchestrator_label}"
                    f"{f' · {orchestrator_name}' if orchestrator_name else ''}"
                ),
            },
            {
                "tag": "markdown",
                "content": (
                    f"**已锁定 Agent Pool** · {len(self.agent_pool)} 个\n"
                    + ("\n".join(pool_lines) if pool_lines else "暂无 Agent")
                    + "\n\n后续节点只会从此池分配；执行期间无需再次选择。"
                ),
            },
        ]
        return {
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "Workflow · 已取消" if is_cancelled else "Workflow · 生成编排中",
                },
                "subtitle": {
                    "tag": "plain_text",
                    "content": "执行已停止" if is_cancelled else "Agent Pool 已锁定",
                },
                "template": "grey" if is_cancelled else "blue",
            },
            "elements": elements,
        }


class WorkflowAgentSelectionRenderer:
    """Render the owner-bound Agent Pool selection control surface."""

    def __init__(
        self,
        pending: Any,
        *,
        project_id: str,
        tool_options: dict[str, str] | None = None,
        model_state: Any = None,
    ) -> None:
        self.pending = pending
        self.project_id = project_id
        self.tool_options = dict(tool_options or {})
        self.model_state = model_state

    def _value(self, action: str, **extra: Any) -> dict[str, Any]:
        return {
            "action": action,
            "project_id": self.project_id,
            "selection_session_key": self.pending.selection_session_key or "",
            **extra,
        }

    def _selection_signature(self) -> str:
        """Identify the rendered draft/pool state for action deduplication."""

        material = {
            "next_agent_sequence": int(self.pending.next_agent_sequence),
            "draft": [
                str(self.pending.draft_tool_name or "").strip().lower(),
                str(self.pending.draft_model_name or "").strip(),
                str(self.pending.draft_profile or "").strip().lower(),
                str(self.pending.draft_effort or "").strip().lower(),
            ],
            "pool": [
                [
                    str(binding.agent_id or ""),
                    str(binding.tool_name or "").strip().lower(),
                    str(binding.model_name or "").strip(),
                    str(binding.profile or "").strip().lower(),
                    str(binding.effort or "").strip().lower(),
                ]
                for binding in tuple(self.pending.agent_pool or ())
            ],
        }
        canonical = json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _button(text: str, value: dict[str, Any], *, kind: str = "default") -> dict[str, Any]:
        return {
            "tag": "button",
            "text": {"tag": "plain_text", "content": text},
            "type": kind,
            "value": value,
            "behaviors": [{"type": "callback", "value": value}],
        }

    def _select_static(
        self,
        *,
        name: str,
        placeholder: str,
        options: list[tuple[str, str]],
        selected: str | None,
        action: str,
    ) -> dict[str, Any]:
        callback_value = self._value(action)
        element: dict[str, Any] = {
            "tag": "select_static",
            "name": name,
            "placeholder": {"tag": "plain_text", "content": placeholder},
            "options": [
                {
                    "text": {"tag": "plain_text", "content": label},
                    "value": value,
                }
                for value, label in options
            ],
            "value": callback_value,
            "behaviors": [{"type": "callback", "value": callback_value}],
        }
        option_values = {value for value, _label in options}
        if selected in option_values:
            element["initial_option"] = selected
        return element

    def render(self) -> dict[str, Any]:
        from ..card.shared import build_responsive_layout
        from .agent_pool import select_auto_orchestrator

        pool = tuple(self.pending.agent_pool or ())
        draft_tool = self.pending.draft_tool_name or next(iter(self.tool_options), "")
        draft_model = self.pending.draft_model_name
        auto_binding = None
        if pool:
            auto_binding = select_auto_orchestrator(
                pool,
                recommendations=self.pending.recommended_agents or (),
            )
        if self.pending.orchestrator_was_auto and self.pending.orchestrator_agent_id:
            orchestrator = f"Auto → {self.pending.orchestrator_agent_id}"
        elif self.pending.orchestrator_agent_id:
            orchestrator = self.pending.orchestrator_agent_id
        elif auto_binding is not None:
            orchestrator = f"Auto → {auto_binding.agent_id}"
        else:
            orchestrator = "Auto"
        elements: list[dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": (
                    f"**需求**\n{self.pending.requirement or ''}\n\n"
                    f"**主编排 Agent**: {orchestrator}\n\n"
                    f"**并发 Agent Pool**: {len(pool)}/8（最多 8 个）"
                ),
            },
        ]
        if self.pending.selection_error:
            elements.append(
                {
                    "tag": "markdown",
                    "content": f"❌ **无法应用选择**\n{self.pending.selection_error}",
                }
            )

        tool_select = self._select_static(
            name="tool_name",
            placeholder="选择工具",
            options=list(self.tool_options.items()),
            selected=draft_tool,
            action="workflow_select_tool",
        )
        model_options = [("default", "Backend default")]
        if self.model_state is not None:
            model_options.extend(
                (name, name) for name in self.model_state.model_names
            )
        model_select = self._select_static(
            name="model_group",
            placeholder="选择模型族",
            options=model_options,
            selected=draft_model or "default",
            action="workflow_select_model",
        )
        add_button = self._button(
            "+ 添加 Agent",
            self._value(
                "workflow_add_agent",
                _selection_sig=self._selection_signature(),
            ),
        )
        elements.extend(
            [
                {"tag": "markdown", "content": "**工具**"},
                tool_select,
                {"tag": "markdown", "content": "**模型族**"},
                model_select,
            ]
        )
        draft_model_is_active = bool(
            draft_model
            and self.model_state is not None
            and draft_model in self.model_state.model_names
        )
        if draft_model_is_active and self.model_state is not None:
            if self.model_state.profiles:
                elements.extend(
                    [
                        {"tag": "markdown", "content": "**Profile**"},
                        self._select_static(
                            name="model_profile",
                            placeholder="选择 Profile",
                            options=[
                                (profile, profile)
                                for profile in self.model_state.profiles
                            ],
                            selected=self.model_state.selected_profile,
                            action="workflow_select_profile",
                        ),
                    ]
                )
            if self.model_state.efforts:
                elements.extend(
                    [
                        {"tag": "markdown", "content": "**Effort**"},
                        self._select_static(
                            name="model_effort",
                            placeholder="选择 Effort",
                            options=[
                                (effort, effort)
                                for effort in self.model_state.efforts
                            ],
                            selected=self.model_state.selected_effort,
                            action="workflow_select_effort",
                        ),
                    ]
                )
        elements.append(add_button)
        elements.extend(
            build_responsive_layout(
                [
                    self._button("使用推荐池", self._value("workflow_add_recommended_pool")),
                    self._button("清空", self._value("workflow_clear_agents")),
                ],
                layout="mobile",
            )
        )
        if not pool:
            elements.append(
                {
                    "tag": "markdown",
                    "content": "尚未添加 Agent。先选择工具和模型，再加入并发池。",
                }
            )
        for binding in pool:
            elements.append(
                {
                    "tag": "markdown",
                    "content": (
                        f"**{binding.agent_id}** · {binding.display_name}\n"
                        f"{_binding_config_markdown(binding)}"
                    ),
                }
            )
            elements.extend(
                build_responsive_layout(
                    [
                        self._button(
                            f"设为主编排 {binding.agent_id}",
                            self._value(
                                "workflow_set_orchestrator",
                                agent_id=binding.agent_id,
                            ),
                        ),
                        self._button(
                            f"移除 {binding.agent_id}",
                            self._value(
                                "workflow_remove_agent",
                                agent_id=binding.agent_id,
                            ),
                        ),
                    ]
                )
            )

        elements.extend(
            build_responsive_layout(
                [
                    self._button(
                        "Auto 分配主编排",
                        self._value("workflow_set_orchestrator", agent_id="auto"),
                    ),
                    self._button(
                        "使用此池开始编排",
                        self._value("workflow_confirm_agents"),
                        kind="primary",
                    ),
                ],
                layout="mobile",
            )
        )
        recommendations = list(self.pending.recommended_agents or [])
        if recommendations:
            elements.insert(
                1,
                {
                    "tag": "markdown",
                    "content": "**推荐池（仅点击后加入）**\n"
                    + "\n".join(
                        f"- {item.get('display_name') or item.get('tool_name')}"
                        for item in recommendations
                    ),
                },
            )
        return {
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow · 选择 Agent Pool"},
                "template": "blue",
            },
            "elements": elements,
        }


def _ordered_direct_agents(project: WorkflowProject) -> list[AgentProgress]:
    flattened = [agent for phase in project.phases for agent in phase.agents]
    return [
        agent
        for _, agent in sorted(
            enumerate(flattened),
            key=lambda pair: (
                int(getattr(pair[1], "call_index", pair[0]) or 0),
                pair[0],
            ),
        )
    ]


def _set_workflow_page_key(
    card: dict[str, Any],
    *,
    kind: str,
    page_identity: int | str,
    local_page_index: int,
) -> dict[str, Any]:
    """Attach delivery-only identity without changing the Feishu card body."""
    card[_WORKFLOW_PAGE_KEY_FIELD] = (
        kind,
        page_identity,
        int(local_page_index),
    )
    return card


def _identified_direct_agents(
    project: WorkflowProject,
) -> list[tuple[AgentProgress, str]]:
    """Pair calls with stable identities independent of mutable list position."""
    seen: dict[str, int] = {}
    identified: list[tuple[AgentProgress, str]] = []
    for agent in _ordered_direct_agents(project):
        identity_payload = json.dumps(
            {
                "agent_id": str(getattr(agent, "agent_id", None) or ""),
                "call_index": int(getattr(agent, "call_index", 0) or 0),
                "label": str(getattr(agent, "label", "") or ""),
                "model": str(getattr(agent, "model", None) or ""),
                "role": str(getattr(agent, "role", None) or ""),
                "started_at": getattr(agent, "started_at", None),
                "tool": str(getattr(agent, "tool", "") or ""),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        base_identity = "call:" + hashlib.sha256(
            identity_payload.encode("utf-8")
        ).hexdigest()
        occurrence = seen.get(base_identity, 0)
        seen[base_identity] = occurrence + 1
        page_identity = (
            base_identity
            if occurrence == 0
            else f"{base_identity}:{occurrence}"
        )
        identified.append((agent, page_identity))
    return identified


def _execution_content_block(raw: object):
    """Normalize live frozen blocks and JSON-restored block dictionaries."""
    if not isinstance(raw, Mapping):
        if hasattr(raw, "kind") and hasattr(raw, "block_id"):
            return raw
        return None
    payload = dict(raw)
    kind = str(payload.pop("kind", "text") or "text")
    return ContentBlock(kind=kind, **payload)


def _execution_terminal(agent: AgentProgress) -> str:
    return {
        AgentStatus.DONE: "completed",
        AgentStatus.CACHED: "completed",
        AgentStatus.FAILED: "failed",
        AgentStatus.CANCELLED: "cancelled",
    }.get(agent.status, "running")


def _execution_header_template(agent: AgentProgress) -> str:
    return {
        AgentStatus.DONE: "green",
        AgentStatus.CACHED: "turquoise",
        AgentStatus.FAILED: "red",
        AgentStatus.CANCELLED: "grey",
    }.get(agent.status, "blue")


def _terminalize_execution_blocks(
    blocks: tuple,
    agent: AgentProgress,
) -> tuple:
    """Ensure a terminal direct call has no misleading active block chrome."""
    if _execution_terminal(agent) == "running":
        return blocks
    normalized = []
    for block in blocks:
        if getattr(block, "status", None) != "active":
            normalized.append(block)
            continue
        changes: dict[str, Any] = {"status": "completed"}
        if getattr(block, "kind", None) == "tool_call":
            changes["status"] = (
                "failed" if agent.status == AgentStatus.FAILED else "completed"
            )
            changes["is_latest_active"] = False
        normalized.append(dataclasses.replace(block, **changes))
    return tuple(normalized)


def _render_agent_execution_cards(project: WorkflowProject) -> list[dict[str, Any]]:
    """Render one ordinary-format, losslessly paginated stream per direct call."""
    cards: list[dict[str, Any]] = []
    for agent, page_identity in _identified_direct_agents(project):
        raw_blocks = list(getattr(agent, "execution_blocks", ()) or ())
        blocks = tuple(
            block
            for raw in raw_blocks
            if (block := _execution_content_block(raw)) is not None
        )
        if not blocks:
            continue
        blocks = _terminalize_execution_blocks(blocks, agent)

        display_index = max(1, int(getattr(agent, "call_index", 0) or 0) + 1)
        agent_id = str(getattr(agent, "agent_id", None) or "unbound")
        label = str(agent.label or "agent")
        terminal = _execution_terminal(agent)
        elapsed = max(0.0, float(agent.duration_s or 0.0))
        if elapsed <= 0 and agent.started_at:
            elapsed = max(
                0.0,
                float((agent.finished_at or time.time()) - agent.started_at),
            )

        state = CardState(
            blocks=blocks,
            terminal=terminal,
            header=HeaderState(
                title=f"#{display_index} · {agent_id} · {label}",
                template=_execution_header_template(agent),
            ),
            metadata=CardMetadata(
                project_name=project.name or "Workflow",
                mode_name=agent.tool or "Agent",
                tool_name=agent.tool or None,
                model_name=agent.model,
                unit_id=str(display_index),
                unit_kind="workflow_agent",
                unit_label=f"{agent_id} · {label}",
                card_sequence=display_index,
                session_started_at=agent.started_at,
                programming_text_sections=True,
                full_execution_blocks=True,
            ),
            runtime_stats=RuntimeStats(elapsed_seconds=elapsed),
            version=len(blocks),
            structural_version=len(blocks),
        )
        rendered_pages = render_card(
            state,
            RenderBudget(engine_cmd="/wf"),
        )
        for local_page_index, rendered in enumerate(rendered_pages):
            card = rendered.to_feishu_json()
            _set_workflow_page_key(
                card,
                kind="agent",
                page_identity=page_identity,
                local_page_index=local_page_index,
            )
            body = card.get("body") or {}
            elements = body.get("elements") or []
            _card_text_for_agent_output(
                elements,
                _AGENT_OUTPUT_FORBIDDEN_MARKERS,
            )
            cards.append(card)
    return cards


class WorkflowProgressRenderer:
    """Renders workflow execution state into Feishu card-compatible JSON.

    Read-only: all state mutations happen through WorkflowStateManager.
    This class only reads the WorkflowProject to produce card elements.
    """

    def __init__(self, project: WorkflowProject) -> None:
        self._project = project
        self._start_time: float = project.started_at or time.time()
        self._render_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

    # ------------------------------------------------------------------
    # Rendering — produce Feishu card elements
    # ------------------------------------------------------------------

    def render_progress_card(self, project: WorkflowProject | None = None) -> dict[str, Any]:
        """Generate the full Feishu card JSON structure.

        Args:
            project: Optional snapshot to render; falls back to self._project.
                Callers under concurrent mutation MUST pass a snapshot() for safety.
        """
        with self._render_lock:
            if project is not None:
                saved = self._project
                self._project = project
                try:
                    return self._render_progress_card_impl()
                finally:
                    self._project = saved
            return self._render_progress_card_impl()

    def render_progress_cards(self, project: WorkflowProject | None = None) -> list[dict[str, Any]]:
        """Render status and complete direct-call streams while running.

        Result ledgers are terminal-only. Feishu messages cannot be inserted
        before an already-created ledger, so publishing one after the first
        completed call would force every later direct-call card below it.
        """
        with self._render_lock:
            target = project or self._project
            if project is not None:
                saved = self._project
                self._project = project
                try:
                    status_cards = self._render_progress_card_pages_impl()
                finally:
                    self._project = saved
            else:
                status_cards = self._render_progress_card_pages_impl()
            return [
                *[
                    _set_workflow_page_key(
                        card,
                        kind="status",
                        page_identity=-1,
                        local_page_index=local_page_index,
                    )
                    for local_page_index, card in enumerate(status_cards)
                ],
                *_render_agent_execution_cards(target),
            ]

    def _render_progress_card_impl(self) -> dict[str, Any]:
        return self._render_progress_card_pages_impl()[0]

    def _render_progress_card_pages_impl(self) -> list[dict[str, Any]]:
        elements: list[dict[str, Any]] = []

        # -- Current execution summary section (top) --
        summary = self._render_summary_section()
        if summary is not None:
            elements.append(summary)
            elements.append(_hr_element())

        agent_pool = self._render_agent_pool_section()
        if agent_pool is not None:
            elements.append(agent_pool)
            elements.append(_hr_element())

        # -- Progress bar section --
        elements.append(self._render_progress_bar_section())
        elements.append(_hr_element())

        # -- Phase tree --
        for idx, phase in enumerate(self._project.phases):
            elements.extend(self._render_phase_section(idx, phase))

        # -- Token usage section (informational, no budget limit) --
        elements.append(_hr_element())
        elements.append(self._render_token_usage_section())

        # -- Metrics footer --
        elements.append(_hr_element())
        elements.append(self._render_metrics_footer())

        # Defensive check: ensure no accidental agent-output sentinel leaks
        _card_text_for_agent_output(elements, _AGENT_OUTPUT_FORBIDDEN_MARKERS)

        return _paginate_progress_cards(self._render_header(), elements)

    def _render_agent_pool_section(self) -> dict[str, Any] | None:
        """Render the immutable A-id bindings and script-declared plan."""
        pending = self._project.pending
        raw_pool: list[Any] = []
        orchestrator_agent_id: str | None = None
        orchestrator_was_auto = False
        raw_plan: list[Any] = []

        if pending is not None:
            raw_pool = list(pending.agent_pool)
            orchestrator_agent_id = pending.orchestrator_agent_id
            orchestrator_was_auto = pending.orchestrator_was_auto
            pending_meta = pending.meta or {}
            raw_plan = list(
                pending_meta.get("agentPlan") or pending_meta.get("agent_plan") or []
            )
        elif self._project.run_spec is not None:
            run_spec = self._project.run_spec
            if isinstance(run_spec, dict):
                raw_pool = list(run_spec.get("agent_pool") or [])
                orchestrator_agent_id = run_spec.get("orchestrator_agent_id")
            else:
                raw_pool = list(run_spec.agent_pool)
                orchestrator_agent_id = run_spec.orchestrator_agent_id

        if not raw_plan and self._project.meta is not None:
            raw_plan = list(self._project.meta.agent_plan)

        if not raw_pool and not raw_plan:
            return None

        lines = ["**Agent pool**"]
        if orchestrator_agent_id:
            orchestrator_label = (
                f"Auto → {orchestrator_agent_id}"
                if orchestrator_was_auto
                else orchestrator_agent_id
            )
            lines.append(f"- 主编排: {orchestrator_label}")
        for binding in raw_pool:
            agent_id = str(
                _binding_value(binding, "agentId", "agent_id", default="?")
            )
            suffix = " (orchestrator)" if agent_id == orchestrator_agent_id else ""
            lines.append(
                f"- `{agent_id}` · {_binding_config_markdown(binding)}{suffix}"
            )

        if raw_plan:
            lines.append("\n**Agent plan**")
            for raw_node in raw_plan:
                if not isinstance(raw_node, dict):
                    raw_node = raw_node.model_dump(by_alias=True)
                node_id = (
                    raw_node.get("nodeId")
                    or raw_node.get("node_id")
                    or raw_node.get("node")
                    or "?"
                )
                role = raw_node.get("role") or "worker"
                static_id = raw_node.get("agentId") or raw_node.get("agent_id")
                candidates = raw_node.get("candidateAgentIds") or raw_node.get(
                    "candidate_agent_ids"
                )
                if raw_node.get("runtime") is True:
                    binding = "运行时分配 → 候选 [" + ", ".join(candidates or []) + "]"
                else:
                    binding = f"静态分配 → {static_id}" if static_id else "未绑定"
                lines.append(f"- `{node_id}` · {role} · {binding}")

        return _md_element("\n".join(lines))

    def _render_summary_section(self) -> dict[str, Any] | None:
        """Render a compact current/latest activity summary block."""
        # Find a running agent first; fall back to the most-recently changed
        # agent across all phases.
        running_agent: Any = None
        running_phase: PhaseProgress | None = None
        running_changed_at: float | None = None
        latest_agent: Any = None
        latest_agent_phase: PhaseProgress | None = None
        latest_changed_at: float | None = None
        latest_phase_only: PhaseProgress | None = None
        latest_phase_changed_at: float | None = None

        for phase in self._project.phases:
            phase_changed_at = getattr(phase, "finished_at", None) or getattr(phase, "started_at", None) or 0.0
            if latest_phase_changed_at is None or phase_changed_at > latest_phase_changed_at:
                latest_phase_only = phase
                latest_phase_changed_at = phase_changed_at
            for agent in phase.agents:
                activity_changed_at = (
                    getattr(agent, "activity_updated_at", None)
                    or getattr(agent, "started_at", None)
                    or 0.0
                )
                if agent.status == AgentStatus.RUNNING and (
                    running_changed_at is None or activity_changed_at >= running_changed_at
                ):
                    running_agent = agent
                    running_phase = phase
                    running_changed_at = activity_changed_at
                # Track most recently changed agent
                changed_at = (
                    getattr(agent, "finished_at", None)
                    or getattr(agent, "activity_updated_at", None)
                    or getattr(agent, "started_at", None)
                    or 0.0
                )
                if latest_changed_at is None or changed_at > latest_changed_at:
                    latest_agent = agent
                    latest_agent_phase = phase
                    latest_changed_at = changed_at

        active_agent = running_agent or latest_agent
        latest_phase = running_phase if running_agent is not None else latest_agent_phase
        if running_agent is not None:
            latest_changed_at = running_changed_at
        if active_agent is None and latest_phase is None:
            latest_phase = latest_phase_only
            latest_changed_at = latest_phase_changed_at

        requirement = str(self._project.requirement or "").strip()
        if active_agent is None and latest_phase is None and not requirement:
            return None

        # Compose the summary lines
        lines: list[str] = []
        if requirement:
            lines.append(f"🎯 **任务:** {_middle_ellipsis(requirement, 240)}")
        metrics = self._project.metrics
        total_phases = len(self._project.phases)
        completed_phases = min(
            total_phases,
            max(
                int(getattr(metrics, "phases_completed", 0) or 0),
                sum(1 for phase in self._project.phases if phase.finished_at is not None),
            ),
        )
        start = self._project.started_at or self._start_time
        end = self._project.finished_at or time.time()
        elapsed = max(0.0, end - start)
        lines.append(
            f"📊 **总览:** Phase {completed_phases}/{total_phases} · "
            f"Token {_format_tokens(metrics.total_tokens)} · 耗时 {format_elapsed_clock(elapsed)}"
        )
        terminal = self._project.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        )
        if self._project.status == WorkflowStatus.COMPLETED:
            summary_title = "执行已完成"
        elif self._project.status == WorkflowStatus.FAILED:
            summary_title = "执行已失败"
        elif self._project.status == WorkflowStatus.CANCELLED:
            summary_title = "执行已取消"
        else:
            summary_title = "当前执行中"

        # Phase
        phase_title = latest_phase.title if latest_phase is not None else "(暂无阶段)"
        phase_title = _middle_ellipsis(phase_title)
        phase_idx = self._project.phases.index(latest_phase) + 1 if latest_phase in self._project.phases else "—"
        lines.append(f"📌 **当前阶段:** 阶段 {phase_idx} · {phase_title}")

        # Active agent
        agent_label_key = "最近代理" if terminal else "当前代理"
        tool_label_key = "使用工具" if terminal else "正在使用"
        if active_agent is not None:
            agent_label = _middle_ellipsis(active_agent.label or "agent")
            agent_status_icon = STATUS_ICONS.get(active_agent.status, "⏳")
            lines.append(f"🤖 **{agent_label_key}:** {agent_status_icon} {agent_label}")
            if active_agent.tool:
                binding = f"`{active_agent.tool}`"
                binding += f" / `{active_agent.model or '默认模型'}`"
                lines.append(f"🛠 **{tool_label_key}:** {binding}")
            else:
                lines.append(f"🛠 **{tool_label_key}:** (未指定工具)")
            if active_agent.task_summary:
                lines.append(f"📋 **当前任务:** {_middle_ellipsis(active_agent.task_summary, 60)}")
            attempt = max(1, int(getattr(active_agent, "attempt", 1) or 1))
            lines.append(f"🔁 **Attempt:** {attempt}")
            activity = getattr(active_agent, "current_activity", "") or ""
            if activity and not terminal and active_agent.status == AgentStatus.RUNNING:
                lines.append(f"⚡ **正在:** {_middle_ellipsis(activity, 60)}")
        else:
            lines.append(f"🤖 **{agent_label_key}:** (无代理调用)")
            lines.append(f"🛠 **{tool_label_key}:** —")

        # Last-change time
        if latest_changed_at and latest_changed_at > 0:
            from datetime import datetime  # local import — keeps module import lean

            try:
                ts = datetime.fromtimestamp(latest_changed_at).strftime("%H:%M:%S")
            except (OSError, OverflowError, ValueError):
                ts = "—"
            lines.append(f"🕒 **最近变更:** {ts}")
        else:
            lines.append("🕒 **最近变更:** —")

        # Live elapsed for a genuinely running agent — mirrors the per-agent
        # elapsed suffix so the top summary keeps advancing during a long
        # blocking agent() call (only meaningful while non-terminal).
        if (
            not terminal
            and active_agent is not None
            and active_agent.status == AgentStatus.RUNNING
            and active_agent.started_at
        ):
            elapsed = time.time() - active_agent.started_at
            if elapsed > 0:
                lines.append(f"⏱ **已运行:** {_format_duration(elapsed)}")

        return _md_element(f"**{summary_title}**\n" + "\n".join(lines))

    def render_compact_status(self) -> str:
        """One-line text summary of workflow status.

        Example: "任务: code-audit | 阶段 2/3 | 7/12 代理 完成 | 450K tokens 消耗"
        """
        name = self._project.name or "workflow"
        total_phases = len(self._project.phases)
        current_phase = self._current_phase_index() + 1

        metrics = self._project.metrics
        completed = metrics.completed_agents
        total = metrics.total_agents

        tokens = _format_tokens(metrics.total_tokens if hasattr(metrics, "total_tokens") else 0)

        status_icon = WORKFLOW_STATUS_ICONS.get(self._project.status, "\u23f3")

        return (
            f"任务: {name} | 阶段 {current_phase}/{total_phases} | "
            f"{completed}/{total} 代理 {status_icon} | {tokens} tokens 消耗"
        )

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _render_header(self) -> dict[str, Any]:
        """Render card header with workflow name + status."""
        status = self._project.status
        icon = WORKFLOW_STATUS_ICONS.get(status, "\u23f3")

        # Map status to header template color
        if status == WorkflowStatus.COMPLETED:
            template = "green"
        elif status == WorkflowStatus.FAILED:
            template = "red"
        elif status == WorkflowStatus.RUNNING:
            template = "blue"
        else:
            template = "grey"

        title = f"{icon} {self._project.name or 'Workflow'}"

        return {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        }

    def _render_progress_bar_section(self) -> dict[str, Any]:
        """Render overall progress as compact "进度 M/N · Z%" line + bar."""
        metrics = self._project.metrics
        completed = metrics.completed_agents
        total = max(metrics.total_agents, 1)
        ratio = completed / total
        pct = _pct(completed, total)
        bar = _unicode_progress_bar(ratio)
        return _md_element(f"进度 {completed}/{total} · {pct}\n{bar}")

    def _render_phase_section(self, idx: int, phase: PhaseProgress) -> list[dict[str, Any]]:
        """Render a phase with agents grouped by status into collapsible panels."""
        elements: list[dict[str, Any]] = []

        agents = phase.agents
        total_agents = len(agents)
        completed_count = sum(1 for a in agents if a.status in (AgentStatus.DONE, AgentStatus.CACHED))
        phase_tokens = sum(max(0, int(getattr(a, "token_usage", 0) or 0)) for a in agents)

        # Phase header — row 1: title (middle-ellipsis); row 2: completion count + duration
        phase_status = self._get_phase_status_icon(phase)
        elements.append(_md_element(f"**{phase_status} 阶段 {idx + 1}: {_middle_ellipsis(phase.title)}**"))

        if phase.started_at and total_agents > 0:
            elapsed = (phase.finished_at or time.time()) - phase.started_at
            duration_text = _format_duration(elapsed)
            elements.append(
                _md_element(
                    f"已完成 {completed_count}/{total_agents} · "
                    f"Token {_format_tokens(phase_tokens)} · 耗时 {duration_text}"
                )
            )
        elif total_agents > 0:
            elements.append(
                _md_element(
                    f"已完成 {completed_count}/{total_agents} · Token {_format_tokens(phase_tokens)}"
                )
            )
        elif phase.finished_at:
            if phase.started_at:
                elapsed = phase.finished_at - phase.started_at
                elements.append(_md_element(f"已完成 0/0 · 耗时 {_format_duration(elapsed)}"))
            else:
                elements.append(_md_element("已完成 0/0"))
        elif phase.started_at:
            elements.append(_md_element("进行中 0/0"))
        else:
            elements.append(_md_element("等待中"))

        if not agents:
            return elements

        # Group agents by status buckets
        buckets: dict[str, list[AgentProgress]] = {
            "RUNNING": [],
            "FAILED": [],
            "DONE": [],
            "CACHED": [],
            "CANCELLED": [],
            "PENDING": [],
        }
        for agent in agents:
            raw = agent.status.value if hasattr(agent.status, "value") else str(agent.status)
            key = raw.upper()
            if key in buckets:
                buckets[key].append(agent)
            else:
                buckets["PENDING"].append(agent)

        # Status → label + color mapping for collapsible_panel headers
        status_meta: dict[str, tuple[str, str]] = {
            "RUNNING": ("执行中", "blue"),
            "FAILED": ("失败", "red"),
            "DONE": ("已完成", "green"),
            "CACHED": ("缓存", "turquoise"),
            "CANCELLED": ("已取消", "grey"),
            "PENDING": ("待执行", "grey"),
        }
        terminal_operations = {
            AgentStatus.DONE: "已完成，完整输出见结果账本",
            AgentStatus.CACHED: "已命中缓存，完整输出见结果账本",
            AgentStatus.FAILED: "执行失败，错误与已有输出见结果账本",
            AgentStatus.CANCELLED: "已取消",
            AgentStatus.PENDING: "等待调度",
        }

        # Render status groups as collapsible panels (RUNNING/FAILED expanded, rest collapsed)
        display_order = [
            ("RUNNING", True),
            ("FAILED", True),
            ("DONE", False),
            ("CACHED", False),
            ("CANCELLED", False),
            ("PENDING", False),
        ]
        display_groups = [
            (key, expanded, offset, buckets[key][offset : offset + 8])
            for key, expanded in display_order
            for offset in range(0, len(buckets[key]), 8)
        ]
        for key, expanded, group_offset, group in display_groups:
            label, color = status_meta[key]
            lines: list[str] = []
            for agent in group:
                tool_badge = f"`{agent.tool or '未指定工具'}` / `{agent.model or '默认模型'}`"
                display_label = _middle_ellipsis(agent.label or "agent")
                display_index = max(1, int(getattr(agent, "call_index", 0) or 0) + 1)
                attempt = max(1, int(getattr(agent, "attempt", 1) or 1))
                status_text = status_meta[key][0]
                task = " ".join((agent.task_summary or "未提供任务摘要").split())
                if agent.status == AgentStatus.RUNNING:
                    operation = " ".join(
                        str(getattr(agent, "current_activity", "") or "等待 Agent 返回").split()
                    )
                    elapsed_suffix = ""
                    if agent.started_at:
                        elapsed = time.time() - agent.started_at
                        if elapsed > 0:
                            elapsed_suffix = f" · 已运行 {_format_duration(elapsed)}"
                else:
                    operation = terminal_operations.get(agent.status, "等待调度")
                    elapsed_suffix = (
                        f" · 耗时 {_format_duration(agent.duration_s)}"
                        if agent.duration_s > 0
                        else ""
                    )
                row = (
                    f"{STATUS_ICONS.get(agent.status, '·')} #{display_index} {display_label} · "
                    f"{status_text} · Attempt {attempt} · {tool_badge}{elapsed_suffix}\n"
                    f"任务：{_middle_ellipsis(task, 100)}\n"
                    f"当前操作：{_middle_ellipsis(operation, 120)}"
                )
                row += _render_subagent_lines(agent.subagents)
                lines.append(row)
            header_obj: dict[str, Any] = {
                "title": {
                    "tag": "plain_text",
                    "content": (
                        f"{label} ({group_offset + 1}-{group_offset + len(group)}"
                        f"/{len(buckets[key])})"
                        if len(buckets[key]) > len(group)
                        else f"{label} ({len(group)})"
                    ),
                },
            }
            panel = _collapsible_panel(
                header_obj,
                [_md_element("\n".join(lines))],
                expanded=expanded,
                template=color,
            )
            elements.append(panel)

        return elements

    def _render_token_usage_section(self) -> dict[str, Any]:
        """Render token consumption as compact informational line (no budget limit)."""
        metrics = self._project.metrics
        total_tokens = metrics.total_tokens if hasattr(metrics, "total_tokens") else 0
        used_str = _format_tokens(total_tokens)
        return _md_element(f"Token 消耗: {used_str}")

    def _render_metrics_footer(self) -> dict[str, Any]:
        """Render metrics footer as a 2-column stretch layout: Agents/耗时 · 缓存/失败。"""
        metrics = self._project.metrics
        start = self._project.started_at or self._start_time
        end = self._project.finished_at or time.time()
        elapsed = end - start if end >= start else time.time() - self._start_time
        elapsed_str = format_elapsed_clock(elapsed)

        # Left column: Agents + 耗时
        left_content = [
            f"**代理:** {metrics.completed_agents}/{metrics.total_agents}",
            f"**耗时:** {elapsed_str}",
        ]
        # Right column: 缓存 + 失败
        right_content = []
        if metrics.cached_agents > 0:
            right_content.append(f"**缓存:** {metrics.cached_agents}")
        if metrics.failed_agents > 0:
            right_content.append(f"**失败:** {metrics.failed_agents}")
        if not right_content:
            right_content.append("**缓存:** 0")

        return _column_set(
            [
                _column(
                    [_md_element("\n".join(left_content), text_align="center")],
                    weight=1,
                    vertical_align="center",
                ),
                _column(
                    [_md_element("\n".join(right_content), text_align="center")],
                    weight=1,
                    vertical_align="center",
                ),
            ],
            flex_mode="stretch",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_phase_index(self) -> int:
        """Return index of the current (last) phase, or 0 if none."""
        if not self._project.phases:
            return 0
        return len(self._project.phases) - 1

    def _get_phase_status_icon(self, phase: PhaseProgress) -> str:
        """Determine the overall status icon for a phase."""
        if not phase.agents:
            if phase.finished_at:
                return "\u2705"
            return "\u23f3"

        has_running = any(a.status == AgentStatus.RUNNING for a in phase.agents)
        has_failed = any(a.status == AgentStatus.FAILED for a in phase.agents)
        all_done = all(a.status in (AgentStatus.DONE, AgentStatus.CACHED, AgentStatus.CANCELLED) for a in phase.agents)

        if has_running:
            return "\U0001f504"
        if all_done:
            return "\u2705"
        if has_failed:
            return "\u274c"
        return "\u23f3"


_VERDICT_LABELS = {
    BriefVerdict.PASSED: "通过",
    BriefVerdict.NEEDS_ATTENTION: "需处理",
    BriefVerdict.FAILED: "失败",
    BriefVerdict.UNKNOWN: "待确认",
}

_FINDING_ICONS = {
    BriefSeverity.HIGH: "🔴",
    BriefSeverity.MEDIUM: "🟡",
    BriefSeverity.LOW: "🔵",
    BriefSeverity.INFO: "•",
}


def _brief_section_markdown(
    title: str,
    items: list[BriefItem],
    *,
    omitted: int = 0,
    finding_icons: bool = False,
) -> str:
    """Render complete brief items and one semantic overflow counter."""
    lines = [f"**{title}**"]
    for item in items:
        prefix = _FINDING_ICONS[item.severity] if finding_icons else "-"
        lines.append(f"{prefix} {_escape_md(item.text)}")
    if omitted:
        lines.append(f"- 另有 {omitted} 条完整内容，详见报告")
    return "\n".join(lines)


def _render_result_brief_elements(brief: WorkflowResultBrief) -> list[dict[str, Any]]:
    """Render the fixed result-first information hierarchy."""
    elements: list[dict[str, Any]] = [
        _md_element(f"**结论**\n{_escape_md(brief.conclusion)}"),
    ]
    section_plan = [
        ("关键发现", "findings", brief.findings, True),
        ("验证", "verification", brief.verification, False),
        ("交付物", "deliverables", brief.deliverables, False),
        ("下一步", "next_steps", brief.next_steps, False),
    ]
    for title, key, items, finding_icons in section_plan:
        omitted = brief.omitted_counts.get(key, 0)
        if not items and not omitted:
            continue
        elements.append(
            _md_element(
                _brief_section_markdown(
                    title,
                    items,
                    omitted=omitted,
                    finding_icons=finding_icons,
                )
            )
        )
    return elements


def _card_text_for_result_brief(
    brief: WorkflowResultBrief,
    forbidden_markers: tuple[str, ...],
) -> None:
    """Run the defensive output gate before Markdown escaping changes text."""
    raw_elements = [_md_element(brief.conclusion)]
    for items in (brief.findings, brief.verification, brief.deliverables, brief.next_steps):
        raw_elements.extend(_md_element(item.text) for item in items)
    _card_text_for_agent_output(raw_elements, forbidden_markers)


def _complete_text_or_default(value: Any, *, max_bytes: int, default: str) -> str:
    """Keep a complete status value or replace it with a semantic fallback."""
    text = _strip_internal_details(str(value or "").strip())
    if not text:
        return default
    if len(text.encode("utf-8", errors="surrogatepass")) > max_bytes:
        return default
    return text


def _completion_process_markdown(
    project: WorkflowProject,
    *,
    completed_phases: int,
    total_phases: int,
    completed_agents: int,
    total_agents: int,
    failed_agents: int,
    cached_agents: int,
) -> str:
    """Build a compact run process summary for the completion card."""
    elapsed = 0.0
    if project.started_at:
        end_time = project.finished_at or time.time()
        elapsed = end_time - project.started_at

    # Header stats line — dense, single row
    stats_parts = [f"**阶段** {completed_phases}/{total_phases}"]
    agent_desc = f"{completed_agents}/{total_agents} 完成"
    if failed_agents:
        agent_desc += f"，{failed_agents} 失败"
    if cached_agents:
        agent_desc += f"，{cached_agents} 缓存"
    stats_parts.append(f"**代理** {agent_desc}")
    stats_parts.append(f"**耗时** {format_elapsed_clock(elapsed)}")

    lines = [" · ".join(stats_parts)]

    # Phase rows — compact, no bullets
    for idx, phase in enumerate(project.phases, 1):
        agents = phase.agents
        total = len(agents)
        done = sum(1 for agent in agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED))
        cancelled = sum(1 for agent in agents if agent.status == AgentStatus.CANCELLED)
        failed = sum(1 for agent in agents if agent.status == AgentStatus.FAILED)
        if failed:
            icon = "\u274c"
            state = f"{done}/{total} 完成，{failed} 失败"
        elif cancelled:
            icon = "⏹\ufe0f"
            state = f"{done}/{total} 完成，{cancelled} 已取消"
        else:
            icon = "\u2705"
            state = f"已完成 {done}/{total}"

        duration = ""
        if phase.started_at and phase.finished_at:
            duration = f" · {_format_duration(phase.finished_at - phase.started_at)}"
        lines.append(f"{icon} 阶段 {idx}: {_middle_ellipsis(phase.title)} — {state}{duration}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Completion card helper (module-level, used by WorkflowHandler on_done)
# ---------------------------------------------------------------------------


def _completion_report_status_markdown(report_status: dict[str, Any]) -> str:
    """Render the full-report attachment status for the compact completion card."""
    if report_status.get("attachment_sent"):
        return "📎 **完整 HTML 报告已发送**\n- 附件已回复到当前 Workflow 话题，包含完整结论、原始结果和执行详情。"

    if report_status.get("generated"):
        html_path = report_status.get("html_filename") or report_status.get("html_path") or ""
        markdown_path = report_status.get("markdown_filename") or report_status.get("markdown_path") or ""
        error = _complete_text_or_default(
            report_status.get("error"),
            max_bytes=600,
            default="附件发送失败，详情请查看服务日志",
        )
        lines = [
            "📎 **完整 HTML 报告已生成，附件发送失败**",
            f"- 原因: {_escape_md(str(error))}",
        ]
        if html_path:
            safe_html = _complete_text_or_default(
                html_path,
                max_bytes=800,
                default="本地 HTML 报告",
            )
            lines.append(f"- HTML: `{_escape_md(safe_html)}`")
        if markdown_path:
            safe_markdown = _complete_text_or_default(
                markdown_path,
                max_bytes=800,
                default="本地 Markdown 报告",
            )
            lines.append(f"- Markdown: `{_escape_md(safe_markdown)}`")
        return "\n".join(lines)

    error = _complete_text_or_default(
        report_status.get("error"),
        max_bytes=600,
        default="未知错误，详情请查看服务日志",
    )
    return f"📎 **HTML 报告生成失败**\n- 原因: {_escape_md(str(error))}"


def render_completion_card(
    project: WorkflowProject,
    *,
    report_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a result-first completion card without slicing result items."""
    status = project.status
    metrics = project.metrics
    brief = fit_result_brief(build_result_brief(project.result))
    _card_text_for_result_brief(brief, _AGENT_OUTPUT_FORBIDDEN_MARKERS)

    if status == WorkflowStatus.COMPLETED and brief.verdict == BriefVerdict.NEEDS_ATTENTION:
        template = "orange"
        icon = "⚠️"
        title_suffix = "需修正"
    elif status == WorkflowStatus.COMPLETED:
        template = "green"
        icon = "\u2705"
        title_suffix = "完成"
    elif status == WorkflowStatus.FAILED:
        template = "red"
        icon = "\u274c"
        title_suffix = "失败"
    elif status == WorkflowStatus.CANCELLED:
        template = "grey"
        icon = "\u274c"
        title_suffix = "已取消"
    else:
        template = "blue"
        icon = "\u2705"
        title_suffix = "完成"

    name = _middle_ellipsis(project.name or "Workflow", 32)
    header = {
        "title": {"tag": "plain_text", "content": f"{icon} {name} — {title_suffix}"},
        "template": template,
    }

    elements: list[dict[str, Any]] = []

    elapsed = 0.0
    if project.started_at:
        end_time = project.finished_at or time.time()
        elapsed = end_time - project.started_at

    task_text = _complete_text_or_default(
        project.requirement,
        max_bytes=600,
        default=project.name or "Workflow",
    )
    elements.append(_md_element(f"**任务**: {_escape_md(task_text)}"))

    total_phases = len(project.phases)
    completed_phases = sum(
        1
        for phase in project.phases
        if (
            all(a.status in (AgentStatus.DONE, AgentStatus.CACHED, AgentStatus.CANCELLED) for a in phase.agents)
            if phase.agents
            else bool(phase.finished_at)
        )
    )
    phase_agents = [agent for phase in project.phases for agent in phase.agents]
    total_agents_count = metrics.total_agents or len(phase_agents)
    completed_agents_count = metrics.completed_agents or sum(
        1 for agent in phase_agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED)
    )
    failed_agents_count = metrics.failed_agents or sum(1 for agent in phase_agents if agent.status == AgentStatus.FAILED)
    cached_agents_count = metrics.cached_agents or sum(1 for agent in phase_agents if agent.status == AgentStatus.CACHED)
    verdict_label = _VERDICT_LABELS[brief.verdict]
    high_risk_count = sum(1 for item in brief.findings if item.severity == BriefSeverity.HIGH)
    outcome_count = high_risk_count if high_risk_count else len(brief.deliverables)
    outcome_label = "高风险" if high_risk_count else "交付物"

    elements.append(
        _column_set(
            [
                _column(
                    [
                        _md_element(
                            f"**{format_elapsed_clock(elapsed)}**\n<font color='grey'>耗时</font>",
                            text_align="center",
                        )
                    ],
                    weight=1,
                    vertical_align="center",
                ),
                _column(
                    [_md_element(f"**{completed_phases}/{total_phases}**\n<font color='grey'>阶段</font>", text_align="center")],
                    weight=1,
                    vertical_align="center",
                ),
                _column(
                    [_md_element(f"**{verdict_label}**\n<font color='grey'>验证</font>", text_align="center")],
                    weight=1,
                    vertical_align="center",
                ),
                _column(
                    [_md_element(f"**{outcome_count}**\n<font color='grey'>{outcome_label}</font>", text_align="center")],
                    weight=1,
                    vertical_align="center",
                ),
            ],
            flex_mode="stretch",
        )
    )

    elements.append(_hr_element())
    elements.extend(_render_result_brief_elements(brief))

    if project.phases:
        elements.append(_hr_element())
        elements.append(
            _collapsible_panel(
                "执行过程",
                [
                    _md_element(
                        _completion_process_markdown(
                            project,
                            completed_phases=completed_phases,
                            total_phases=total_phases,
                            completed_agents=completed_agents_count,
                            total_agents=total_agents_count,
                            failed_agents=failed_agents_count,
                            cached_agents=cached_agents_count,
                        )
                    )
                ],
                expanded=False,
                template="grey",
            )
        )

    if report_status:
        elements.append(_hr_element())
        elements.append(_md_element(_completion_report_status_markdown(report_status)))

    if status == WorkflowStatus.FAILED and project.error:
        elements.append(_hr_element())
        safe_project_err = _complete_text_or_default(
            project.error,
            max_bytes=600,
            default="错误详情过长，请查看服务日志",
        )
        elements.append(_md_element(f"\u274c **错误**: {safe_project_err}"))

    _card_text_for_agent_output(elements, _AGENT_OUTPUT_FORBIDDEN_MARKERS)
    card = {"header": header, "elements": elements}
    if len(json.dumps(card, ensure_ascii=False).encode("utf-8", errors="surrogatepass")) <= _CARD_MAX_BYTES:
        return card

    hidden_items = sum(brief.omitted_counts.values()) + sum(
        len(items)
        for items in (brief.findings, brief.verification, brief.deliverables, brief.next_steps)
    )
    minimal_elements = [_md_element(f"**结论**\n{_escape_md(brief.conclusion)}")]
    if hidden_items:
        minimal_elements.append(_md_element(f"另有 {hidden_items} 条完整内容，详见报告"))
    if report_status:
        minimal_elements.append(_hr_element())
        minimal_elements.append(_md_element(_completion_report_status_markdown(report_status)))
    if status == WorkflowStatus.FAILED and project.error:
        minimal_elements.append(_hr_element())
        minimal_elements.append(_md_element(f"\u274c **错误**: {safe_project_err}"))
    _card_text_for_agent_output(minimal_elements, _AGENT_OUTPUT_FORBIDDEN_MARKERS)
    return {"header": header, "elements": minimal_elements}


def render_completion_cards(
    project: WorkflowProject,
    *,
    report_status: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Render terminal status, complete direct-call streams, then the ledger."""
    return [
        _set_workflow_page_key(
            render_completion_card(project, report_status=report_status),
            kind="status",
            page_identity=-1,
            local_page_index=0,
        ),
        *_render_agent_execution_cards(project),
        *[
            _set_workflow_page_key(
                card,
                kind="ledger",
                page_identity=-1,
                local_page_index=local_page_index,
            )
            for local_page_index, card in enumerate(
                _render_result_ledger_cards(project)
            )
        ],
    ]


def _result_ledger_entries(project: WorkflowProject) -> list[tuple[str, str, str]]:
    terminal = {
        AgentStatus.DONE,
        AgentStatus.CACHED,
        AgentStatus.FAILED,
        AgentStatus.CANCELLED,
    }
    flattened = [agent for phase in project.phases for agent in phase.agents]
    ordered_agents = [
        agent
        for _, agent in sorted(
            enumerate(flattened),
            key=lambda pair: (getattr(pair[1], "call_index", pair[0]), pair[0]),
        )
        if agent.status in terminal
    ]
    status_labels = {
        AgentStatus.DONE: "已完成",
        AgentStatus.CACHED: "缓存命中",
        AgentStatus.FAILED: "失败",
        AgentStatus.CANCELLED: "已取消",
    }
    entries: list[tuple[str, str, str]] = []
    for ordinal, agent in enumerate(ordered_agents, start=1):
        agent_id = str(getattr(agent, "agent_id", None) or "unbound")
        binding = agent.tool or "未指定工具"
        if agent.model:
            binding += f" / {agent.model}"
        operation = str(
            getattr(agent, "current_activity", None)
            or getattr(agent, "task_summary", None)
            or agent.label
            or "agent"
        ).strip()
        result_text = str(getattr(agent, "result", None) or "").strip()
        error_text = _strip_internal_details(str(agent.error or "")).strip()
        if result_text and error_text:
            body = f"{result_text}\n\n错误: {error_text}"
        elif result_text:
            body = result_text
        elif error_text:
            body = f"错误: {error_text}"
        else:
            body = "（无结果）"
        entries.append(
            (
                f"Agent 结果 {ordinal} · `{agent_id}` · {_middle_ellipsis(agent.label or 'agent', 60)}",
                (
                    f"状态: {status_labels[agent.status]} · "
                    f"工具: {_middle_ellipsis(binding, 80)} · "
                    f"操作: {_middle_ellipsis(operation, 80)}"
                ),
                body,
            )
        )

    for ordinal, evidence in enumerate(project.reviewer_evidence, start=1):
        output = str(evidence.output or "").strip()
        error = _strip_internal_details(str(evidence.error or "")).strip()
        if output and error:
            body = f"{output}\n\n错误: {error}"
        elif output:
            body = output
        elif error:
            body = f"错误: {error}"
        else:
            body = "（无结果）"
        display_name = evidence.display_name or f"Reviewer {ordinal}"
        entries.append(
            (
                f"评审结果 {ordinal} · {_middle_ellipsis(display_name, 60)}",
                f"状态: {evidence.status} · 工具: {_middle_ellipsis(evidence.tool, 80)}",
                body,
            )
        )

    workflow_terminal = project.status in (
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    )
    if project.result is not None or workflow_terminal:
        raw_result = str(project.result or "").strip()
        if raw_result:
            try:
                decoded = json.loads(raw_result)
            except (TypeError, ValueError, json.JSONDecodeError):
                body = raw_result
            else:
                body = (
                    decoded
                    if isinstance(decoded, str)
                    else json.dumps(decoded, ensure_ascii=False, indent=2, default=str)
                )
        elif project.error:
            body = f"错误: {_strip_internal_details(project.error)}"
        else:
            body = "（无结果）"
        entries.append(("Workflow 最终结果", f"状态: {project.status.value}", body))
    return entries


def _result_ledger_card(
    project: WorkflowProject,
    elements: list[dict[str, Any]],
    *,
    page: int = 0,
    total: int = 0,
) -> dict[str, Any]:
    name = _middle_ellipsis(project.name or "Workflow", 32)
    page_suffix = f" · {page}/{total}" if total > 1 else ""
    return {
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"📚 {name} · 结果账本{page_suffix}",
            },
            "template": "blue",
        },
        "elements": elements,
    }


def _result_card_fits(project: WorkflowProject, elements: list[dict[str, Any]]) -> bool:
    card = _result_ledger_card(project, elements)
    size = len(json.dumps(card, ensure_ascii=False).encode("utf-8", errors="surrogatepass"))
    return size <= _CARD_MAX_BYTES


def _ledger_markdown(title: str, metadata: str, body: str, *, continuation: bool = False) -> str:
    suffix = "（续）" if continuation else ""
    heading = f"**{_escape_md(title)}{suffix}**\n<font color='grey'>{_escape_md(metadata)}</font>"
    return f"{heading}\n\n{_escape_md(body)}"


def _largest_result_prefix(
    project: WorkflowProject,
    title: str,
    metadata: str,
    body: str,
    *,
    continuation: bool,
) -> int:
    low, high, best = 1, len(body), 0
    while low <= high:
        middle = (low + high) // 2
        element = _md_element(
            _ledger_markdown(
                title,
                metadata,
                body[:middle],
                continuation=continuation,
            )
        )
        if _result_card_fits(project, [element]):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def _render_result_ledger_cards(project: WorkflowProject) -> list[dict[str, Any]]:
    entries = _result_ledger_entries(project)
    if not entries:
        return []

    page_elements: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for title, metadata, body in entries:
        whole = _md_element(_ledger_markdown(title, metadata, body))
        if _result_card_fits(project, [*current, whole]):
            current.append(whole)
            continue
        if current and _result_card_fits(project, [whole]):
            page_elements.append(current)
            current = [whole]
            continue
        if current:
            page_elements.append(current)
            current = []

        remaining = body
        continuation = False
        while remaining:
            take = _largest_result_prefix(
                project,
                title,
                metadata,
                remaining,
                continuation=continuation,
            )
            if take <= 0:
                raise ValueError("Workflow result ledger entry exceeds card metadata budget")
            fragment = _md_element(
                _ledger_markdown(
                    title,
                    metadata,
                    remaining[:take],
                    continuation=continuation,
                )
            )
            current.append(fragment)
            remaining = remaining[take:]
            continuation = True
            if remaining:
                page_elements.append(current)
                current = []

    if current:
        page_elements.append(current)
    total = len(page_elements)
    return [
        _result_ledger_card(project, elements, page=index, total=total)
        for index, elements in enumerate(page_elements, start=1)
    ]


# ---------------------------------------------------------------------------
# Card size enforcement
# ---------------------------------------------------------------------------


def _paginate_progress_cards(
    header: dict[str, Any],
    elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split progress elements into cards without deleting scheduling rows."""

    def _card(page_elements: list[dict[str, Any]], page: int = 0, total: int = 0) -> dict[str, Any]:
        page_header = {
            **header,
            "title": dict(header.get("title") or {}),
        }
        if total > 1:
            title = str(page_header["title"].get("content") or "Workflow")
            page_header["title"]["content"] = f"{title} · {page + 1}/{total}"
        return {"header": page_header, "elements": page_elements}

    pages: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for element in elements:
        candidate = [*current, element]
        encoded = json.dumps(_card(candidate), ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
        if len(encoded) <= _CARD_MAX_BYTES:
            current = candidate
            continue
        if current:
            pages.append(current)
            current = []
        single = json.dumps(_card([element]), ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
        if len(single) > _CARD_MAX_BYTES:
            raise ValueError("Workflow progress element exceeds Feishu card limit")
        current = [element]
    if current or not pages:
        pages.append(current)
    total = len(pages)
    return [_card(page, index, total) for index, page in enumerate(pages)]
