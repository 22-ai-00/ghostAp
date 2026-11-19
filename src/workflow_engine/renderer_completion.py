"""Completion card rendering, result brief sanitization, and completion image helpers."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any

from src.card.shared.text_safety import (
    neutralize_feishu_rich_text_controls,
    sanitize_card_text_for_audit,
    sanitize_markdown_image_references,
)
from src.card.tool_display import (
    sanitize_full_tool_event_value,
    sanitize_tool_failure_detail,
)
from src.utils.text import (
    format_duration,
    format_elapsed_clock,
    sanitize_single_line_label,
)

from .errors import _strip_internal_details
from .models import AgentStatus, WorkflowProject, WorkflowStatus
from .renderer_elements import (
    _LABEL_TRUNCATION_LIMIT,
    _WORKFLOW_FRAGMENT_MAX_BYTES,
    _card_text_for_agent_output,
    _collapsible_panel,
    _column,
    _column_set,
    _escape_md,
    _hr_element,
    _md_element,
    _resolve_renderer_attr,
    _safe_dynamic_markdown,
    _safe_dynamic_text,
    _set_workflow_page_key,
    _utf8_safe_text,
)
from .renderer_execution_pages import (
    _execution_content_block,
    _ordered_direct_agents,
    _render_agent_execution_cards,
)
from .result_brief import (
    BriefItem,
    BriefSeverity,
    BriefVerdict,
    WorkflowResultBrief,
    build_result_brief,
    fit_result_brief,
)

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


# ---------------------------------------------------------------------------
# ImageBlock completion / status helpers
# ---------------------------------------------------------------------------


def _render_image_block_element(block: Any) -> dict[str, Any]:
    """Render one ImageBlock as a Feishu img element or non-fatal placeholder."""
    image_key = str(getattr(block, "image_key", "") or "").strip()
    status = str(getattr(block, "status", "completed") or "").strip().lower()
    if image_key and status != "failed":
        alt = sanitize_single_line_label(
            getattr(block, "alt", ""),
            fallback="任务图片",
            max_chars=120,
        )
        safe_alt = _safe_dynamic_text(alt, limit=120, one_line=True) or "任务图片"
        return {
            "tag": "img",
            "img_key": image_key,
            "alt": {"tag": "plain_text", "content": safe_alt},
        }
    return {
        "tag": "markdown",
        "content": "🖼️ 图片产物暂时无法展示；文本结果与本地文件不受影响。",
        "text_size": "notation",
    }


def _render_project_image_elements(project: WorkflowProject) -> list[dict[str, Any]]:
    """Collect image elements from direct-call execution blocks for status cards."""
    image_elements: list[dict[str, Any]] = []
    for agent in _ordered_direct_agents(project):
        raw_blocks = list(getattr(agent, "execution_blocks", ()) or ())
        for raw in raw_blocks:
            block = _execution_content_block(raw)
            if block is None or getattr(block, "kind", None) != "image":
                continue
            image_elements.append(_render_image_block_element(block))
    if not image_elements:
        return []
    return [_md_element("**图片产物**"), *image_elements]


def _append_fitting_image_elements(
    header: dict[str, Any],
    base_elements: list[dict[str, Any]],
    image_elements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append as many image elements as fit within _WORKFLOW_FRAGMENT_MAX_BYTES."""
    if not image_elements:
        return base_elements
    fitted = [*base_elements, _hr_element(), image_elements[0]]
    for elem in image_elements[1:]:
        trial = [*fitted, elem]
        encoded = json.dumps({"header": header, "elements": trial}, ensure_ascii=False).encode(
            "utf-8", errors="surrogatepass"
        )
        if len(encoded) > _WORKFLOW_FRAGMENT_MAX_BYTES:
            break
        fitted = trial
    encoded_fitted = json.dumps({"header": header, "elements": fitted}, ensure_ascii=False).encode(
        "utf-8", errors="surrogatepass"
    )
    return fitted if len(encoded_fitted) <= _WORKFLOW_FRAGMENT_MAX_BYTES else base_elements


# ---------------------------------------------------------------------------
# Result brief decoding / sanitizing helpers
# ---------------------------------------------------------------------------


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
    forbidden_markers: tuple[str, ...] | None = None,
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


def _safe_terminal_error(
    value: Any,
    *,
    default: str,
    max_chars: int = 600,
) -> str:
    """Return a bounded, secret-free, Markdown-neutral terminal error."""
    _card_text_for_agent_output([_md_element(str(value or ""))])
    return sanitize_tool_failure_detail(
        _strip_internal_details(str(value or "")),
        fallback=default,
        max_chars=max_chars,
    )


def _brief_for_workflow_status(
    brief: WorkflowResultBrief,
    status: WorkflowStatus,
) -> WorkflowResultBrief:
    """Make the persisted Workflow terminal status authoritative in UI copy."""
    if status == WorkflowStatus.FAILED:
        return brief.model_copy(
            update={
                "verdict": BriefVerdict.FAILED,
                "conclusion": "任务未完成，请根据错误信息处理后重试。",
            }
        )
    if status == WorkflowStatus.CANCELLED:
        return brief.model_copy(
            update={
                "verdict": BriefVerdict.UNKNOWN,
                "conclusion": "任务已取消，未生成完整最终结果。",
            }
        )
    return brief


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
            duration = f" · {format_duration(phase.finished_at - phase.started_at)}"
        phase_title = _safe_dynamic_markdown(
            phase.title,
            limit=_LABEL_TRUNCATION_LIMIT,
            one_line=True,
        )
        lines.append(f"{icon} 阶段 {idx}: {phase_title} — {state}{duration}")

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
    brief = fit_result_brief(
        _brief_for_workflow_status(_safe_result_brief(project.result), status)
    )
    _card_text_for_result_brief(brief)

    if status == WorkflowStatus.COMPLETED and brief.verdict == BriefVerdict.FAILED:
        template = "red"
        icon = "\u274c"
        title_suffix = "完成但验证失败"
    elif status == WorkflowStatus.COMPLETED and brief.verdict == BriefVerdict.NEEDS_ATTENTION:
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

    name = _safe_dynamic_text(
        project.name or "Workflow",
        limit=32,
        one_line=True,
    )
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

    image_elements = _render_project_image_elements(project)
    if image_elements:
        elements.append(_hr_element())
        elements.extend(image_elements)

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
        safe_project_err = _safe_terminal_error(
            project.error,
            default="错误详情过长，请查看服务日志",
        )
        elements.append(_md_element(f"\u274c **错误**: {_escape_md(safe_project_err)}"))

    _card_text_for_agent_output(elements)
    card = {"header": header, "elements": elements}
    if (
        len(
            json.dumps(card, ensure_ascii=False).encode(
                "utf-8",
                errors="surrogatepass",
            )
        )
        <= _WORKFLOW_FRAGMENT_MAX_BYTES
    ):
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
        minimal_elements.append(
            _md_element(f"\u274c **错误**: {_escape_md(safe_project_err)}")
        )
    minimal_elements = _append_fitting_image_elements(
        header, minimal_elements, image_elements
    )
    _card_text_for_agent_output(minimal_elements)
    return {"header": header, "elements": minimal_elements}


def render_completion_cards(
    project: WorkflowProject,
    *,
    report_status: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Render terminal status plus complete direct-call fallback cards."""
    completion_card_fn = _resolve_renderer_attr(
        "render_completion_card",
        render_completion_card,
    )
    agent_cards_fn = _resolve_renderer_attr(
        "_render_agent_execution_cards",
        _render_agent_execution_cards,
    )
    return [
        _set_workflow_page_key(
            completion_card_fn(project, report_status=report_status),
            kind="status",
            page_identity=-1,
            local_page_index=0,
        ),
        *agent_cards_fn(project),
    ]


_RESULT_JSON_PREFIX_RE = re.compile(
    r"^(?:result|output|final(?:\s+result)?|结果|最终结果)\s*[:：]\s*",
    re.IGNORECASE,
)


def _decode_result_layers(value: Any) -> Any:
    """Decode up to three JSON wrappers without assuming every string is JSON."""
    decoded = value
    for _ in range(3):
        if not isinstance(decoded, str):
            break
        text = decoded.strip()
        if not text:
            return ""
        prefixed = _RESULT_JSON_PREFIX_RE.sub("", text, count=1)
        if prefixed != text:
            text = prefixed.strip()
        fenced = _json_fence_body(text)
        if fenced is not None:
            text = fenced
        try:
            candidate = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
            break
        if candidate == decoded:
            break
        decoded = candidate
    return decoded


def _json_fence_body(text: str) -> str | None:
    """Return a complete JSON fence while respecting delimiter type/length."""
    first_line, separator, remainder = text.partition("\n")
    if not separator:
        return None
    opener = re.fullmatch(
        r"\s{0,3}(?P<fence>`{3,}|~{3,})\s*json\s*",
        first_line,
        re.IGNORECASE,
    )
    if opener is None:
        return None
    body, closing_separator, closing_line = remainder.rstrip().rpartition("\n")
    if not closing_separator:
        return None
    closer = re.fullmatch(r"\s{0,3}(?P<fence>`{3,}|~{3,})\s*", closing_line)
    if closer is None:
        return None
    opening_fence = opener.group("fence")
    closing_fence = closer.group("fence")
    if (
        opening_fence[0] != closing_fence[0]
        or len(closing_fence) < len(opening_fence)
    ):
        return None
    return body.strip()


def _safe_decoded_result(raw_result: Any) -> Any:
    """Decode result wrappers before recursively redacting their leaf values."""
    decoded = _decode_result_layers(raw_result)
    return sanitize_full_tool_event_value(decoded)


def _safe_result_brief(raw_result: str | None) -> WorkflowResultBrief:
    """Build the completion brief from the same safe decoded view as ledgers."""
    decoded = _safe_decoded_result(str(raw_result or "").strip())
    if isinstance(decoded, Mapping):
        normalized = json.dumps(
            decoded,
            ensure_ascii=False,
            default=str,
        )
        brief = build_result_brief(_utf8_safe_text(normalized))
    else:
        brief = build_result_brief(None)
    return _sanitize_result_brief(brief)


def _sanitize_result_display_text(value: Any) -> str:
    """Redact, UTF-8 normalize, and neutralize card control sequences in result text."""
    text = _utf8_safe_text(sanitize_full_tool_event_value(value))
    text = neutralize_feishu_rich_text_controls(text)
    text = sanitize_markdown_image_references(text)
    return sanitize_card_text_for_audit(text)


def _sanitize_result_brief(brief: WorkflowResultBrief) -> WorkflowResultBrief:
    """Sanitize all user-visible fields on a parsed :class:`WorkflowResultBrief`."""
    def clean_items(items: list[BriefItem]) -> list[BriefItem]:
        return [
            item.model_copy(
                update={"text": _sanitize_result_display_text(item.text)}
            )
            for item in items
        ]

    return brief.model_copy(
        update={
            "conclusion": _sanitize_result_display_text(brief.conclusion),
            "findings": clean_items(brief.findings),
            "verification": clean_items(brief.verification),
            "deliverables": clean_items(brief.deliverables),
            "next_steps": clean_items(brief.next_steps),
        }
    )
