"""Shared Feishu Card 2.0 elements, safety helpers, and constants for workflow rendering."""

from __future__ import annotations

import sys
from typing import Any

from src.card.shared.text_safety import neutralize_feishu_rich_text_controls
from src.card.thresholds import THRESHOLDS
from src.card.tool_display import sanitize_tool_failure_detail

from .errors import _strip_internal_details
from .models import AgentStatus, SubagentProgress, SubagentStatus, WorkflowStatus

_RENDERER_MODULE_NAME = "src.workflow_engine.renderer"
_MISSING = object()


def _renderer_module() -> Any:
    """Return ``src.workflow_engine.renderer`` when loaded in ``sys.modules``."""
    return sys.modules.get(_RENDERER_MODULE_NAME)


def _resolve_renderer_attr(name: str, default: Any) -> Any:
    """Resolve an attribute dynamically from the facade module so runtime patches apply."""
    renderer_mod = _renderer_module()
    if renderer_mod is not None:
        resolved = getattr(renderer_mod, name, _MISSING)
        if resolved is not _MISSING and resolved is not None:
            return resolved
    return default


def _utf8_safe_text(value: Any) -> str:
    """Replace lone surrogate code points before JSON/card serialization."""
    text = "" if value is None else str(value)
    return text.encode("utf-8", errors="replace").decode("utf-8")


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
    text = _utf8_safe_text(text)
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
    text = neutralize_feishu_rich_text_controls(_utf8_safe_text(text))
    for ch in ("*", "_", "`", "|", "[", "]", "~"):
        text = text.replace(ch, "\\" + ch)
    return text


def _safe_dynamic_text(
    value: Any,
    *,
    limit: int | None = None,
    one_line: bool = False,
) -> str:
    """Return inert UTF-8 text for model/user-controlled card fields."""
    text = neutralize_feishu_rich_text_controls(_utf8_safe_text(value))
    if one_line:
        text = " ".join(text.split())
    if limit is not None:
        text = _middle_ellipsis(text, limit)
    return text


def _safe_dynamic_markdown(
    value: Any,
    *,
    limit: int | None = None,
    one_line: bool = False,
) -> str:
    """Return an inert dynamic value ready for interpolation into Markdown."""
    return _escape_md(
        _safe_dynamic_text(value, limit=limit, one_line=one_line)
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

_CARD_MAX_BYTES = 27 * 1024  # Shared final-wire payload budget.
_WORKFLOW_WIRE_RESERVE_BYTES = 1_536
_WORKFLOW_WIRE_RESERVE_NODES = 16
_WORKFLOW_FRAGMENT_MAX_BYTES = _CARD_MAX_BYTES - _WORKFLOW_WIRE_RESERVE_BYTES
_WORKFLOW_FRAGMENT_MAX_NODES = (
    THRESHOLDS["CARD_NODE_BUDGET"] - _WORKFLOW_WIRE_RESERVE_NODES
)
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
    forbidden_markers: tuple[str, ...] | None = None,
) -> None:
    """Scan ``elements`` recursively for forbidden marker strings.

    Iterates through the element list and any nested ``dict``/``list`` values,
    looking for ``text`` / ``content`` string fields that contain any of the
    ``forbidden_markers``. If a match is found, raises
    :class:`RuntimeError` with the message ``"card leaked agent output"``.

    When ``forbidden_markers`` is ``None``, reads
    ``_AGENT_OUTPUT_FORBIDDEN_MARKERS`` dynamically from
    ``src.workflow_engine.renderer`` so runtime test patches apply.
    When empty, the function is a no-op — this is the normal production
    configuration. Tests monkey-patch ``_AGENT_OUTPUT_FORBIDDEN_MARKERS`` to
    inject sentinel strings and verify the gate trips when agent output
    accidentally leaks into card text.
    """
    markers = (
        _resolve_renderer_attr("_AGENT_OUTPUT_FORBIDDEN_MARKERS", ())
        if forbidden_markers is None
        else forbidden_markers
    )
    if not markers:
        return

    marker_variants = tuple(
        frozenset((marker, _escape_md(marker)))
        for marker in markers
    )
    stack: list[Any] = list(elements)
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in ("text", "content") and isinstance(value, str):
                    for variants in marker_variants:
                        if any(marker and marker in value for marker in variants):
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
    element: dict[str, Any] = {
        "tag": "markdown",
        "content": _utf8_safe_text(content),
    }
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
        header_obj = dict(header)
        header_obj.pop("template", None)
    header_obj.setdefault(
        "icon",
        {
            "tag": "standard_icon",
            "token": "down_outlined",
            "color": "grey",
        },
    )
    header_obj.setdefault("icon_position", "right")
    header_obj.setdefault("icon_expanded_angle", -180)
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


# ---------------------------------------------------------------------------
# Binding formatting helpers
# ---------------------------------------------------------------------------


def _binding_value(binding: Any, *names: str, default: Any = None) -> Any:
    """Return the first non-None attribute or dict key from ``names``."""
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


def _binding_model_parts(
    binding: Any,
    *,
    default_tool: str = "?",
    default_model: str = "default",
) -> tuple[str, str, str | None, str | None]:
    """Return a display-only tool/model/profile/effort tuple without duplication."""
    raw_tool = str(
        _binding_value(binding, "tool_name", "toolName", "tool", default="") or ""
    ).strip()
    tool_name = raw_tool or default_tool
    profile_value = _binding_value(binding, "profile")
    effort_value = _binding_value(binding, "effort")
    profile = str(profile_value).strip() or None if profile_value else None
    effort = str(effort_value).strip() or None if effort_value else None
    raw_model = str(
        _binding_value(
            binding,
            "model_name",
            "modelName",
            "model",
            default=default_model,
        )
        or default_model
    ).strip()

    # ACP model selections may encode profile/effort in model_name while also
    # exposing the structured fields. Strip only the exact structured suffix;
    # model families that legitimately contain slashes remain untouched.
    structured_suffix = "/".join(value for value in (profile, effort) if value)
    suffix = f"/{structured_suffix}" if structured_suffix else ""
    model_name = (
        raw_model[: -len(suffix)]
        if suffix and raw_model.endswith(suffix)
        else raw_model
    )
    return tool_name, model_name or default_model, profile, effort


def _binding_roles(binding: Any) -> tuple[str, ...]:
    """Return deduplicated normalized role strings from ``role`` and ``roles``."""
    collected: list[str] = []
    primary = _binding_value(binding, "role")
    if isinstance(primary, str) and primary.strip():
        collected.append(primary.strip())
    raw_roles = _binding_value(binding, "roles")
    if isinstance(raw_roles, str):
        if raw_roles.strip():
            collected.append(raw_roles.strip())
    elif raw_roles is not None:
        try:
            for item in raw_roles:
                text = str(item or "").strip()
                if text:
                    collected.append(text)
        except TypeError:
            pass
    return tuple(dict.fromkeys(collected))


def _binding_role_markdown(binding: Any) -> str | None:
    """Format a binding's role(s) as safe inline Markdown, or ``None`` if absent."""
    roles = _binding_roles(binding)
    if not roles:
        return None
    return f"角色 `{_safe_dynamic_markdown('/'.join(roles), one_line=True)}`"


def _binding_config_markdown(
    binding: Any,
    *,
    include_role: bool = True,
    default_tool: str = "?",
    default_model: str = "default",
) -> str:
    """Format a binding's tool, model, profile, effort, and optional role as Markdown."""
    tool_name, model_name, profile, effort = _binding_model_parts(
        binding,
        default_tool=default_tool,
        default_model=default_model,
    )
    parts = [
        f"`{_safe_dynamic_markdown(tool_name, one_line=True)}`",
        f"模型 `{_safe_dynamic_markdown(model_name, one_line=True)}`",
    ]
    if profile:
        parts.append(f"Profile `{_safe_dynamic_markdown(profile, one_line=True)}`")
    if effort:
        parts.append(f"Effort `{_safe_dynamic_markdown(effort, one_line=True)}`")
    if include_role:
        role_md = _binding_role_markdown(binding)
        if role_md:
            parts.append(role_md)
    return " · ".join(parts)
