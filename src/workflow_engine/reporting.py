"""Full Workflow execution report generation.

The Feishu card is a compact status surface; this module writes the full,
untruncated execution payload to local report files that can be sent as an IM
attachment.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.text import format_duration

from .models import AgentProgress, AgentStatus, WorkflowProject
from .report_markdown_html import render_markdown_to_html
from .report_normalizer import (
    NormalizedAgentView,
    NormalizedBlock,
    NormalizedSection,
    NormalizedWorkflowReport,
    normalize_agent_output,
    normalize_workflow_report,
)
from .storage import (
    DEFAULT_WORKFLOW_STORAGE_ROOT,
    workflow_reports_dir,
)

DEFAULT_WORKFLOW_CACHE_ROOT = DEFAULT_WORKFLOW_STORAGE_ROOT

_SEVERITY_VALUES = frozenset({"critical", "major", "high", "medium", "minor", "low", "info"})
_SEVERITY_COLUMN_NAMES = frozenset({"severity", "级别", "严重程度", "level"})


@dataclass(frozen=True)
class WorkflowReportFiles:
    """Paths for a generated Workflow report bundle."""

    run_id: str
    html_path: str
    markdown_path: str
    html_filename: str
    markdown_filename: str


def _safe_slug(value: str, *, default: str = "workflow") -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:80] or default


def _status_text(value: Any) -> str:
    """Return string representation for an Enum or raw status value."""
    return getattr(value, "value", str(value or ""))


def _pretty_json(value: Any) -> str:
    """Serialize any Python object as indented UTF-8 JSON."""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _parse_json_payload(raw_result: str | None) -> Any | None:
    """Parse a raw JSON object/array string, returning None on invalid input."""
    text = str(raw_result or "").strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _markdown_code_fence(text: str, lang: str = "") -> str:
    """Wrap text in a backtick fence longer than any backtick run inside it."""
    runs = re.findall(r"`+", text)
    fence_len = max([3, *(len(run) + 1 for run in runs)])
    fence = "`" * fence_len
    body = text.rstrip("\n")
    return f"{fence}{lang}\n{body}\n{fence}" if body else f"{fence}{lang}\n{fence}"


def _escape_md_table_cell(value: Any) -> str:
    """Escape pipe characters and newlines inside a Markdown table cell."""
    text = str(value if value is not None else "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("|", "\\|").replace("\n", "<br>")
    return text.strip() or "—"


def _format_md_list_item(item: str) -> str:
    """Format a potentially multi-line list entry as a Markdown bullet item."""
    lines = str(item or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    first = lines[0].strip()
    rest = [f"  {line}" if line.strip() else "" for line in lines[1:]]
    if rest:
        return "\n".join([f"- {first}", *rest]).rstrip()
    return f"- {first}"


def _block_to_markdown(block: NormalizedBlock) -> str:
    """Convert a NormalizedBlock into clean GitHub-Flavored Markdown."""
    kind = block.kind
    title_prefix = f"### {block.title}\n\n" if block.title else ""

    if kind in ("markdown", "prose"):
        text = str(block.text or "").strip()
        return f"{title_prefix}{text}" if text else ""

    if kind == "table":
        table = block.table
        if table is None or not table.columns:
            text = str(block.text or "").strip()
            return f"{title_prefix}{_markdown_code_fence(text, 'text')}" if text else ""
        header = "| " + " | ".join(_escape_md_table_cell(col) for col in table.columns) + " |"
        sep = "| " + " | ".join("---" for _ in table.columns) + " |"
        rows: list[str] = []
        col_count = len(table.columns)
        for row in table.rows:
            padded = list(row[:col_count]) + [""] * max(0, col_count - len(row))
            rows.append("| " + " | ".join(_escape_md_table_cell(cell) for cell in padded) + " |")
        return f"{title_prefix}" + "\n".join([header, sep, *rows])

    if kind == "kv":
        if not block.pairs:
            return ""
        lines: list[str] = []
        for k, v in block.pairs:
            val_text = str(v if v is not None else "").replace("\r\n", "\n").replace("\r", "\n").strip()
            if "\n" in val_text:
                val_lines = val_text.split("\n")
                first = val_lines[0]
                tail = "\n".join(f"  {line}" for line in val_lines[1:])
                lines.append(f"- **{k}**: {first}\n{tail}")
            else:
                lines.append(f"- **{k}**: {val_text}")
        return f"{title_prefix}" + "\n".join(lines)

    if kind == "list":
        items = [_format_md_list_item(item) for item in block.items if str(item or "").strip()]
        if not items:
            return ""
        return f"{title_prefix}" + "\n".join(items)

    if kind == "callout":
        level = str(block.level or "info").lower()
        alert_tag = {
            "warning": "[!WARNING]",
            "warn": "[!WARNING]",
            "error": "[!CAUTION]",
            "danger": "[!CAUTION]",
            "critical": "[!CAUTION]",
            "fail": "[!CAUTION]",
            "success": "[!TIP]",
            "ok": "[!TIP]",
            "passed": "[!TIP]",
            "tip": "[!TIP]",
        }.get(level, "[!NOTE]")
        quote_lines = [f"> {alert_tag}"]
        if block.title:
            quote_lines.append(f"> **{block.title}**")
        text = str(block.text or "").strip()
        if text:
            for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                quote_lines.append(f"> {line}" if line else ">")
        return "\n".join(quote_lines)

    if kind == "json":
        text = str(block.text or "").strip()
        return f"{title_prefix}{_markdown_code_fence(text, 'json')}" if text else ""

    text = str(block.text or "").strip()
    return f"{title_prefix}{text}" if text else ""


def _blocks_to_markdown(blocks: tuple[NormalizedBlock, ...] | list[NormalizedBlock]) -> str:
    """Join rendered Markdown blocks with blank lines."""
    rendered = [part for part in (_block_to_markdown(block) for block in blocks) if part]
    return "\n\n".join(rendered)


def _compute_metrics(project: WorkflowProject) -> dict[str, Any]:
    """Compute aggregate phase, agent, token, and duration metrics for a run."""
    metrics = project.metrics
    phase_agents = [agent for phase in project.phases for agent in phase.agents]
    total_agents = metrics.total_agents or len(phase_agents)
    completed_agents = metrics.completed_agents or sum(
        1 for agent in phase_agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED)
    )
    failed_agents = metrics.failed_agents or sum(1 for agent in phase_agents if agent.status == AgentStatus.FAILED)
    cached_agents = metrics.cached_agents or sum(1 for agent in phase_agents if agent.status == AgentStatus.CACHED)
    total_tokens = metrics.total_tokens or sum(agent.token_usage for agent in phase_agents)
    total_phases = len(project.phases)
    completed_phases = sum(
        1
        for phase in project.phases
        if all(a.status in (AgentStatus.DONE, AgentStatus.CACHED, AgentStatus.CANCELLED) for a in phase.agents)
    )
    elapsed = 0.0
    if project.started_at:
        end_time = project.finished_at or datetime.now(timezone.utc).timestamp()
        elapsed = max(0.0, end_time - project.started_at)
    success_rate = int((completed_agents / max(total_agents, 1)) * 100)
    return {
        "total_agents": total_agents,
        "completed_agents": completed_agents,
        "failed_agents": failed_agents,
        "cached_agents": cached_agents,
        "total_tokens": total_tokens,
        "total_phases": total_phases,
        "completed_phases": completed_phases,
        "elapsed": elapsed,
        "success_rate": success_rate,
    }


def _workflow_summary(project: WorkflowProject) -> str:
    """Render the Markdown summary bullet list including pool and orchestrator metadata."""
    stats = _compute_metrics(project)
    lines = [
        f"- 任务: {project.requirement or project.name or 'Workflow'}",
        f"- 名称: {project.name or 'Workflow'}",
        f"- 状态: {_status_text(project.status)}",
        f"- Workflow ID: {project.workflow_id or '(none)'}",
        f"- 耗时: {format_duration(stats['elapsed'])}",
        f"- 阶段: {stats['total_phases']}",
        (
            f"- 代理: {stats['completed_agents']}/{stats['total_agents']} 完成，"
            f"{stats['failed_agents']} 失败，{stats['cached_agents']} 缓存"
        ),
        f"- Token: {stats['total_tokens']}",
    ]
    run_spec = project.run_spec or {}
    orchestrator = run_spec.get("orchestrator") or {}
    if orchestrator:
        orchestrator_model = (
            "default"
            if orchestrator.get("use_default_model", True)
            else orchestrator.get("model_name") or "default"
        )
        orch_id = run_spec.get("orchestrator_agent_id")
        orch_id_part = f" [{orch_id}]" if orch_id else ""
        lines.append(
            f"- 主编排绑定:{orch_id_part} {orchestrator.get('tool_name') or '(none)'}/{orchestrator_model}"
        )
    pool_entries = run_spec.get("agent_pool") or []
    if pool_entries:
        pool_parts: list[str] = []
        for entry in pool_entries:
            if not isinstance(entry, dict):
                continue
            aid = entry.get("agent_id") or entry.get("agentId") or "?"
            tool = entry.get("tool_name") or entry.get("toolName") or "?"
            model = entry.get("model_name") or entry.get("modelName") or "default"
            meta_bits = [f"{aid}:{tool}/{model}"]
            if entry.get("profile"):
                meta_bits.append(f"profile={entry['profile']}")
            if entry.get("effort"):
                meta_bits.append(f"effort={entry['effort']}")
            if entry.get("role"):
                meta_bits.append(f"role={entry['role']}")
            pool_parts.append(" ".join(meta_bits))
        if pool_parts:
            lines.append("- Agent Pool: " + "; ".join(pool_parts))
    if run_spec:
        lines.append(
            "- 评审模式: "
            + ("Auto（无独立 Reviewer）" if run_spec.get("auto_reviewer") else "显式独立 Reviewer")
        )
    return "\n".join(lines)


def _format_agent_md_line(agent: AgentProgress) -> str:
    """Format one agent metadata summary line for the Markdown phase timeline."""
    display_index = max(1, int(getattr(agent, "call_index", 0) or 0) + 1)
    agent_line = f"- Agent #{display_index} [{_status_text(agent.status)}] {agent.label or 'agent'}"
    if getattr(agent, "agent_id", None):
        agent_line += f" agent_id={agent.agent_id}"
    agent_line += (
        f" tool={agent.tool or '(none)'}"
        f" model={agent.model or 'default'}"
        f" attempt={max(1, int(getattr(agent, 'attempt', 1) or 1))}"
    )
    if getattr(agent, "profile", None):
        agent_line += f" profile={agent.profile}"
    if getattr(agent, "effort", None):
        agent_line += f" effort={agent.effort}"
    if agent.role:
        agent_line += f" role={agent.role}"
    if getattr(agent, "effective_model", None) and agent.effective_model != agent.model:
        agent_line += f" effective_model={agent.effective_model}"
    if agent.duration_s:
        agent_line += f" duration={format_duration(agent.duration_s)}"
    if agent.token_usage:
        agent_line += f" tokens={agent.token_usage}"
    return agent_line


def _phase_process(
    project: WorkflowProject,
    *,
    agent_views: dict[tuple[int, int], NormalizedAgentView] | None = None,
) -> str:
    """Render all recorded phases and per-agent outputs in readable Markdown."""
    if not project.phases:
        return "本次 Workflow 没有记录阶段。"

    lines: list[str] = []
    for p_idx, phase in enumerate(project.phases):
        agents = phase.agents
        total = len(agents)
        done = sum(1 for agent in agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED))
        failed = sum(1 for agent in agents if agent.status == AgentStatus.FAILED)
        cancelled = sum(1 for agent in agents if agent.status == AgentStatus.CANCELLED)
        duration = ""
        if phase.started_at and phase.finished_at:
            duration = f"，耗时 {format_duration(max(0.0, phase.finished_at - phase.started_at))}"
        lines.append(f"### 阶段 {p_idx + 1}: {phase.title}")
        lines.append(f"- 汇总: {done}/{total} 完成，{failed} 失败，{cancelled} 已取消{duration}")
        for a_idx, agent in enumerate(agents):
            agent_line = _format_agent_md_line(agent)
            if agent.task_summary:
                agent_line += f"\n  - task: {agent.task_summary}"
            operation = str(agent.current_activity or "").strip()
            if operation:
                agent_line += f"\n  - latest operation: {operation}"
            if agent.error:
                agent_line += f"\n  - error: {agent.error}"
            if agent.result is not None:
                view = agent_views.get((p_idx, a_idx)) if agent_views is not None else None
                if view is not None and view.has_structured_view and view.summary_blocks:
                    rendered_blocks = _blocks_to_markdown(view.summary_blocks)
                    if rendered_blocks:
                        agent_line += f"\n  - structured output:\n\n{rendered_blocks}"
                    else:
                        agent_line += f"\n  - full result:\n{agent.result}"
                else:
                    agent_line += f"\n  - full result:\n{agent.result}"
            if agent.subagents:
                agent_line += "\n  - ACP internal Agent observations (non-authoritative):"
                for child_index, child in enumerate(agent.subagents, start=1):
                    child_status = _status_text(child.status)
                    child_progress = child.progress or "(no latest operation)"
                    agent_line += (
                        f"\n    - observation {child_index}: status={child_status} "
                        f"model={child.model or 'default'} latest={child_progress}"
                    )
            lines.append(agent_line)
        lines.append("")
    return "\n".join(lines).strip()


def _format_raw_result_for_display(raw_result: str | None) -> str:
    """Pretty-print single-line JSON payloads while preserving multi-line raw output."""
    raw_text = str(raw_result or "")
    if not raw_text:
        return ""
    parsed = _parse_json_payload(raw_text)
    if parsed is not None and "\n" not in raw_text.strip():
        return _pretty_json(parsed)
    return raw_text


def build_workflow_report_markdown(project: WorkflowProject) -> str:
    """Return a full, untruncated Markdown report for a Workflow run."""
    title = project.name or "Workflow"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    normalized = normalize_workflow_report(project)

    lines = [
        f"# {title} 完整报告",
        "",
        f"生成时间: {generated_at}",
        "",
        "## 运行摘要",
        _workflow_summary(project),
        "",
    ]

    if project.error:
        lines.extend(["## 错误信息", project.error, ""])

    if normalized.sections:
        for section in normalized.sections:
            section_body = _blocks_to_markdown(section.blocks)
            if not section_body:
                continue
            badge_note = f"> [!NOTE]\n> 状态标签: {section.badge}\n\n" if section.badge else ""
            lines.extend([f"## {section.title}", f"{badge_note}{section_body}".strip(), ""])
    elif not project.result:
        lines.extend(["## 执行结果", "本次 Workflow 没有返回最终结果。", ""])

    lines.extend(
        [
            "## 执行过程",
            _phase_process(project, agent_views=normalized.agent_views),
            "",
        ]
    )

    raw_result = _format_raw_result_for_display(project.result)
    raw_result_md = (
        _markdown_code_fence(raw_result, "json" if _parse_json_payload(raw_result) is not None else "text")
        if raw_result
        else ""
    )
    lines.extend(["## 原始结果 JSON", raw_result_md, ""])
    lines.extend(
        [
            "## 原始 Workflow 状态 JSON",
            _markdown_code_fence(_pretty_json(project.to_dict()), "json"),
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def _render_inline_markdown_html(text: str) -> str:
    """Render short Markdown fragments inline when possible while strictly escaping HTML."""
    raw = str(text if text is not None else "").strip()
    if not raw:
        return ""
    rendered = render_markdown_to_html(raw).strip()
    if rendered.startswith("<p>") and rendered.endswith("</p>") and rendered.count("<p>") == 1:
        return rendered[3:-4]
    return rendered


def _render_table_cell_html(column_name: str, cell_value: str) -> str:
    """Render one table cell with severity pill formatting or escaped inline Markdown."""
    col_lower = str(column_name or "").strip().lower()
    raw = str(cell_value if cell_value is not None else "").strip()
    if col_lower in _SEVERITY_COLUMN_NAMES and raw.lower() in _SEVERITY_VALUES:
        sev = raw.lower()
        return f'<span class="badge-sev-{html.escape(sev)}">{html.escape(raw)}</span>'
    return _render_inline_markdown_html(raw) or "—"


def _render_block_html(block: NormalizedBlock) -> str:
    """Render a NormalizedBlock to escaped semantic HTML."""
    kind = block.kind
    title_html = f'<h4 class="block-title">{html.escape(block.title)}</h4>' if block.title else ""

    if kind in ("markdown", "prose"):
        text = str(block.text or "").strip()
        if not text:
            return ""
        rendered = render_markdown_to_html(text)
        return f'<div class="semantic-block">{title_html}<div class="md-prose">{rendered}</div></div>'

    if kind == "table":
        table = block.table
        if table is None or not table.columns:
            text = str(block.text or "").strip()
            if not text:
                return ""
            return f'<div class="semantic-block">{title_html}<pre class="raw-pre">{html.escape(text)}</pre></div>'
        ths = "".join(f"<th>{html.escape(str(col))}</th>" for col in table.columns)
        col_count = len(table.columns)
        rows_html: list[str] = []
        for row in table.rows:
            padded = list(row[:col_count]) + [""] * max(0, col_count - len(row))
            tds = "".join(
                f"<td>{_render_table_cell_html(table.columns[idx], cell)}</td>"
                for idx, cell in enumerate(padded)
            )
            rows_html.append(f"<tr>{tds}</tr>")
        body_html = "".join(rows_html) if rows_html else f'<tr><td colspan="{col_count}">—</td></tr>'
        return (
            f'<div class="semantic-block">{title_html}'
            f'<div class="md-table-wrap"><table class="md-table">'
            f"<thead><tr>{ths}</tr></thead><tbody>{body_html}</tbody>"
            f"</table></div></div>"
        )

    if kind == "kv":
        if not block.pairs:
            return ""
        items = "".join(
            f'<div class="kv-item">'
            f'<div class="kv-key">{html.escape(str(k))}</div>'
            f'<div class="kv-val">{_render_inline_markdown_html(str(v)) or "—"}</div>'
            f"</div>"
            for k, v in block.pairs
        )
        return f'<div class="semantic-block">{title_html}<div class="kv-grid">{items}</div></div>'

    if kind == "callout":
        level = str(block.level or "info").lower()
        cls = {
            "warning": "callout-warning",
            "warn": "callout-warning",
            "error": "callout-error",
            "danger": "callout-error",
            "critical": "callout-error",
            "fail": "callout-error",
            "success": "callout-success",
            "ok": "callout-success",
            "passed": "callout-success",
            "tip": "callout-success",
        }.get(level, "callout-info")
        c_title = f'<div class="callout-title">{html.escape(block.title)}</div>' if block.title else ""
        c_body = render_markdown_to_html(str(block.text or "")) if str(block.text or "").strip() else ""
        return (
            f'<div class="semantic-block"><div class="callout-card {cls}">'
            f'{c_title}<div class="md-prose">{c_body}</div>'
            f"</div></div>"
        )

    if kind == "list":
        non_empty = [str(item) for item in block.items if str(item or "").strip()]
        if not non_empty:
            return ""
        lis = "".join(
            f'<li><span class="list-idx">{idx:02d}</span>'
            f'<div class="md-prose">{_render_inline_markdown_html(item)}</div></li>'
            for idx, item in enumerate(non_empty, 1)
        )
        return f'<div class="semantic-block">{title_html}<ul class="semantic-list">{lis}</ul></div>'

    if kind == "json":
        text = str(block.text or "").strip()
        if not text:
            return ""
        return f'<div class="semantic-block">{title_html}<pre class="raw-pre">{html.escape(text)}</pre></div>'

    text = str(block.text or "").strip()
    if not text:
        return ""
    return f'<div class="semantic-block">{title_html}<pre class="raw-pre">{html.escape(text)}</pre></div>'


def _render_blocks_html(blocks: tuple[NormalizedBlock, ...] | list[NormalizedBlock]) -> str:
    """Render a sequence of NormalizedBlock items into safe HTML."""
    return "".join(part for part in (_render_block_html(block) for block in blocks) if part)


def _section_search_keywords(section: NormalizedSection) -> str:
    """Build an escaped lowercase keyword string for client-side section filtering."""
    parts = [section.title, section.badge]
    for block in section.blocks:
        if block.title:
            parts.append(block.title)
        if block.table and block.table.columns:
            parts.extend(str(col) for col in block.table.columns)
    return html.escape(" ".join(p for p in parts if p).lower(), quote=True)


def _badge_css_class(tone: str) -> str:
    """Map semantic badge tone names to CSS pill classes."""
    return {
        "ok": "badge-ok",
        "success": "badge-ok",
        "passed": "badge-ok",
        "warn": "badge-warn",
        "warning": "badge-warn",
        "needs_attention": "badge-warn",
        "fail": "badge-fail",
        "error": "badge-fail",
        "failed": "badge-fail",
        "info": "badge-info",
    }.get(str(tone or "").lower(), "badge-neutral")


def _render_summary_stats_html(project: WorkflowProject) -> str:
    """Render a visual stats grid for the HTML report hero section."""
    stats = _compute_metrics(project)
    cards = [
        (format_duration(stats["elapsed"]), "总耗时", "#93c5fd"),
        (f"{stats['completed_phases']}/{stats['total_phases']}", "阶段完成", "#6ee7b7"),
        (f"{stats['completed_agents']}/{stats['total_agents']}", "代理完成", "#c4b5fd"),
        (f"{stats['success_rate']}%", "成功率", "#6ee7b7" if stats["success_rate"] >= 80 else "#fca5a5"),
        (str(stats["total_tokens"]), "Token 消耗", "#cbd5e1"),
        (str(stats["failed_agents"]), "失败", "#fca5a5" if stats["failed_agents"] > 0 else "#cbd5e1"),
        (str(stats["cached_agents"]), "缓存命中", "#67e8f9"),
    ]
    items = [
        f'<div class="stat-card">'
        f'<div class="stat-value" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f"</div>"
        for value, label, color in cards
    ]
    return f'<div class="stats-grid">{"".join(items)}</div>'


def _render_agent_pool_html(project: WorkflowProject) -> str:
    """Render escaped Agent Pool chips for the hero executive banner."""
    run_spec = project.run_spec or {}
    pool_entries = run_spec.get("agent_pool") or []
    if not pool_entries:
        return ""
    orch_id = run_spec.get("orchestrator_agent_id")
    chips: list[str] = []
    for entry in pool_entries:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("agent_id") or entry.get("agentId") or "?")
        tool = str(entry.get("tool_name") or entry.get("toolName") or "?")
        model = str(entry.get("model_name") or entry.get("modelName") or "default")
        bits = [f"{html.escape(tool)} / {html.escape(model)}"]
        if entry.get("profile"):
            bits.append(f"profile={html.escape(str(entry['profile']))}")
        if entry.get("effort"):
            bits.append(f"effort={html.escape(str(entry['effort']))}")
        if entry.get("role"):
            bits.append(f"role={html.escape(str(entry['role']))}")
        is_orch = bool(orch_id and aid == orch_id)
        if is_orch:
            bits.append("主编排")
        chip_cls = "pool-chip orch" if is_orch else "pool-chip"
        chips.append(
            f'<span class="{chip_cls}"><span class="aid">[{html.escape(aid)}]</span> {" · ".join(bits)}</span>'
        )
    if not chips:
        return ""
    return (
        '<div class="pool-strip">'
        '<span class="pool-strip-label">Agent Pool:</span>'
        f'{"".join(chips)}'
        "</div>"
    )


def _render_hero_html(
    project: WorkflowProject,
    normalized: NormalizedWorkflowReport,
    generated_at: str,
) -> str:
    """Render the hero banner with status, verdict, metrics, and executive summary."""
    title = project.name or "Workflow"
    status_text = _status_text(project.status)
    status_cls = {
        "completed": "status-ok",
        "failed": "status-fail",
        "cancelled": "status-cancel",
    }.get(status_text.lower(), "status-running")

    verdict_cls = {
        "passed": "verdict-pill-ok",
        "needs_attention": "verdict-pill-warn",
        "failed": "verdict-pill-fail",
    }.get(normalized.verdict.lower(), "verdict-pill-neutral")
    verdict_badge = (
        f'<span class="meta-pill {verdict_cls}">{html.escape(normalized.verdict_label)}</span>'
        if normalized.verdict_label
        else ""
    )

    meta_extras: list[str] = []
    run_spec = project.run_spec or {}
    if run_spec:
        review_label = "Auto（无独立 Reviewer）" if run_spec.get("auto_reviewer") else "显式独立 Reviewer"
        meta_extras.append(f"<span>评审模式: {html.escape(review_label)}</span>")
        orchestrator = run_spec.get("orchestrator") or {}
        if orchestrator:
            orch_model = (
                "default"
                if orchestrator.get("use_default_model", True)
                else orchestrator.get("model_name") or "default"
            )
            orch_id = run_spec.get("orchestrator_agent_id")
            orch_id_part = f"[{orch_id}] " if orch_id else ""
            orch_str = f"{orch_id_part}{orchestrator.get('tool_name') or '(none)'}/{orch_model}"
            meta_extras.append(f"<span>主编排: {html.escape(orch_str)}</span>")

    stats_html = _render_summary_stats_html(project)
    pool_html = _render_agent_pool_html(project)

    req_html = (
        f'<div class="exec-req"><strong>原始需求：</strong>{html.escape(project.requirement)}</div>'
        if project.requirement
        else ""
    )
    headline = str(normalized.headline_conclusion or "").strip()
    if not headline:
        if project.error:
            headline = f"Workflow 执行失败：{project.error}"
        elif project.result:
            headline = "Workflow 已执行完成，详情请查看下方结构化结果。"
        else:
            headline = "本次 Workflow 没有返回最终结果。"
    headline_html = render_markdown_to_html(headline)

    banner_badge_cls = {
        "passed": "badge-ok",
        "needs_attention": "badge-warn",
        "failed": "badge-fail",
    }.get(normalized.verdict.lower(), "badge-info")
    banner_badge = (
        f'<span class="section-badge {banner_badge_cls}">{html.escape(normalized.verdict_label)}</span>'
        if normalized.verdict_label
        else ""
    )

    return f"""
  <header class="page-header">
    <div class="wrap">
      <div class="hero-top">
        <div>
          <div class="eyebrow">Workflow Execution Report · GhostAP</div>
          <h1>{html.escape(title)} — Workflow 报告</h1>
          <div class="meta-row">
            <span class="meta-pill {status_cls}">{html.escape(status_text)}</span>
            {verdict_badge}
            <span>生成时间: {html.escape(generated_at)}</span>
            <span>ID: {html.escape(project.workflow_id or '—')}</span>
            {''.join(meta_extras)}
          </div>
        </div>
      </div>
      {stats_html}
      <section class="exec-banner" aria-label="核心结论摘要">
        <div class="exec-banner-head">
          <div class="exec-banner-title"><span>执行结论与团队概览</span></div>
          {banner_badge}
        </div>
        {req_html}
        <div class="md-prose">{headline_html}</div>
        {pool_html}
      </section>
    </div>
  </header>"""


def _render_agent_drawer_html(agent: AgentProgress, view: NormalizedAgentView | None) -> str:
    """Render a collapsible semantic output drawer and nested raw fallback for an agent."""
    if agent.result is None:
        return ""
    raw_text = view.raw_text if view is not None else str(agent.result or "")
    if not raw_text.strip():
        return ""

    effective_view = view if view is not None else normalize_agent_output(raw_text)
    blocks_html = _render_blocks_html(effective_view.summary_blocks)
    char_hint = f"{len(raw_text):,} 字符"
    kind_hint = f"已结构化解析 · {char_hint}" if effective_view.has_structured_view else char_hint

    raw_drawer = ""
    if effective_view.has_structured_view or len(raw_text) > 120:
        raw_drawer = (
            '<details class="agent-raw-drawer">'
            "<summary>查看原始 Agent 输出文本</summary>"
            f'<pre class="raw-pre">{html.escape(raw_text)}</pre>'
            "</details>"
        )

    return (
        '<details class="agent-output-drawer">'
        "<summary>"
        "<span>查看输出详情</span>"
        f'<span class="agent-drawer-meta">{html.escape(kind_hint)}</span>'
        "</summary>"
        f'<div class="agent-drawer-body">{blocks_html}{raw_drawer}</div>'
        "</details>"
    )


def _render_phases_html(
    project: WorkflowProject,
    *,
    agent_views: dict[tuple[int, int], NormalizedAgentView] | None = None,
) -> str:
    """Render a structured phases timeline with semantic agent output drawers."""
    if not project.phases:
        return '<p class="empty-hint">本次 Workflow 没有记录阶段。</p>'

    items: list[str] = []
    for p_idx, phase in enumerate(project.phases):
        agents = phase.agents
        total = len(agents)
        done = sum(1 for agent in agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED))
        failed = sum(1 for agent in agents if agent.status == AgentStatus.FAILED)
        cancelled = sum(1 for agent in agents if agent.status == AgentStatus.CANCELLED)
        duration = ""
        if phase.started_at and phase.finished_at:
            duration = format_duration(max(0.0, phase.finished_at - phase.started_at))

        if failed:
            badge_cls = "badge-fail"
            status_text = f"{done}/{total} 完成，{failed} 失败"
        elif cancelled:
            badge_cls = "badge-warn"
            status_text = f"{done}/{total} 完成，{cancelled} 已取消"
        elif done == total and total > 0:
            badge_cls = "badge-ok"
            status_text = f"已完成 {done}/{total}"
        else:
            badge_cls = "badge-pending"
            status_text = f"{done}/{total}"

        agent_rows: list[str] = []
        phase_search_bits = [f"阶段 {p_idx + 1}", phase.title]
        for a_idx, agent in enumerate(agents):
            a_status = _status_text(agent.status)
            a_cls = {
                "done": "agent-ok",
                "cached": "agent-cached",
                "failed": "agent-fail",
                "cancelled": "agent-cancel",
            }.get(a_status.lower(), "agent-pending")

            chips: list[str] = []
            if agent.agent_id:
                chips.append(f'<span class="agent-chip id">[{html.escape(agent.agent_id)}]</span>')
            if agent.tool:
                chips.append(f'<span class="agent-chip tool">{html.escape(agent.tool)}</span>')
            if agent.model:
                chips.append(f'<span class="agent-chip">{html.escape(agent.model)}</span>')
            if agent.profile:
                chips.append(f'<span class="agent-chip">profile={html.escape(agent.profile)}</span>')
            if agent.effort:
                chips.append(f'<span class="agent-chip">effort={html.escape(agent.effort)}</span>')
            if agent.role:
                chips.append(f'<span class="agent-chip">role={html.escape(agent.role)}</span>')
            if agent.effective_model and agent.effective_model != agent.model:
                chips.append(f'<span class="agent-chip">effective={html.escape(agent.effective_model)}</span>')
            attempt = max(1, int(getattr(agent, "attempt", 1) or 1))
            if attempt > 1:
                chips.append(f'<span class="agent-chip">attempt={attempt}</span>')
            if agent.duration_s:
                chips.append(f'<span class="agent-chip">{html.escape(format_duration(agent.duration_s))}</span>')
            if agent.token_usage:
                chips.append(f'<span class="agent-chip">{agent.token_usage} tok</span>')

            a_task = f'<div class="agent-task">{html.escape(agent.task_summary or "")}</div>' if agent.task_summary else ""
            operation = str(agent.current_activity or "").strip()
            a_act = f'<div class="agent-activity">最新动作：{html.escape(operation)}</div>' if operation else ""
            a_err = f'<div class="agent-error">{html.escape(agent.error or "")}</div>' if agent.error else ""

            view = agent_views.get((p_idx, a_idx)) if agent_views is not None else None
            drawer_html = _render_agent_drawer_html(agent, view)

            sub_html = ""
            if agent.subagents:
                sub_items = []
                for c_idx, child in enumerate(agent.subagents, 1):
                    c_status = _status_text(child.status)
                    c_prog = child.progress or "(no latest operation)"
                    sub_items.append(
                        f"<li>观测 #{c_idx}: [{html.escape(c_status)}] "
                        f"{html.escape(child.model or 'default')} — {html.escape(c_prog)}</li>"
                    )
                sub_html = f'<ul class="agent-subagents">{"".join(sub_items)}</ul>'

            agent_search_parts = [
                a_status,
                agent.label or "",
                agent.agent_id or "",
                agent.tool or "",
                agent.model or "",
                agent.profile or "",
                agent.effort or "",
                agent.role or "",
                agent.effective_model or "",
                agent.task_summary or "",
                operation,
                agent.error or "",
            ]
            agent_search = html.escape(" ".join(p for p in agent_search_parts if p).lower(), quote=True)
            phase_search_bits.extend(p for p in agent_search_parts if p)

            agent_rows.append(
                f'<div class="agent-row {a_cls}" data-search="{agent_search}">'
                f'<div class="agent-head">'
                f'<span class="agent-status">{html.escape(a_status)}</span>'
                f'<span class="agent-label">{html.escape(agent.label or "agent")}</span>'
                f'{"".join(chips)}'
                f"</div>"
                f"{a_task}{a_act}{a_err}{drawer_html}{sub_html}"
                f"</div>"
            )

        agents_html = "".join(agent_rows) if agent_rows else '<div class="empty-hint">无代理调用</div>'
        dur_html = f'<span class="phase-duration">{html.escape(duration)}</span>' if duration else ""
        phase_search = html.escape(" ".join(phase_search_bits).lower(), quote=True)

        items.append(
            f'<div class="phase-block" data-search="{phase_search}">'
            f'<div class="phase-header">'
            f'<span class="phase-num">阶段 {p_idx + 1}</span>'
            f'<span class="phase-title">{html.escape(phase.title)}</span>'
            f'<span class="badge {badge_cls}">{html.escape(status_text)}</span>'
            f"{dur_html}"
            f"</div>"
            f'<div class="phase-agents">{agents_html}</div>'
            f"</div>"
        )
    return "".join(items)


def _render_section_card_html(
    dom_id: str,
    body_dom_id: str,
    title: str,
    body_html: str,
    *,
    search_text: str,
    badge: str = "",
    badge_tone: str = "neutral",
    expanded: bool = True,
    extra_class: str = "primary-section",
) -> str:
    """Wrap semantic section HTML in a collapsible card with toggle and copy actions."""
    badge_html = (
        f'<span class="section-badge {_badge_css_class(badge_tone)}">{html.escape(badge)}</span>'
        if badge
        else ""
    )
    open_attr = " open" if expanded else ""
    return f"""
      <details class="report-section {extra_class}" id="{dom_id}" data-search="{search_text}"{open_attr}>
        <summary>
          <span class="section-title">{html.escape(title)}</span>
          {badge_html}
          <div class="section-actions">
            <button type="button" onclick="toggleSection('{dom_id}'); event.preventDefault();">折叠/展开</button>
            <button type="button" onclick="copyText('{body_dom_id}', this); event.preventDefault();">复制</button>
          </div>
        </summary>
        <div class="section-body-rich" id="{body_dom_id}">{body_html}</div>
      </details>"""


def build_workflow_report_html(project: WorkflowProject, markdown: str | None = None) -> str:
    """Return a self-contained, static HTML report for a Workflow run."""
    title = project.name or "Workflow"
    markdown = markdown if markdown is not None else build_workflow_report_markdown(project)
    normalized = normalize_workflow_report(project)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    primary_toc: list[str] = []
    appendix_toc: list[str] = []
    primary_sections_html: list[str] = []
    secondary_sections_html: list[str] = []

    if project.error:
        err_block = NormalizedBlock(
            kind="callout",
            title="Workflow 执行错误信息",
            text=str(project.error),
            level="error",
        )
        err_search = html.escape(f"错误信息 {project.error}".lower(), quote=True)
        primary_toc.append('<a href="#section-error">错误信息</a>')
        primary_sections_html.append(
            _render_section_card_html(
                "section-error",
                "section-body-error",
                "错误信息",
                _render_block_html(err_block),
                search_text=err_search,
                badge="error",
                badge_tone="fail",
                expanded=True,
                extra_class="primary-section",
            )
        )

    for idx, section in enumerate(normalized.sections):
        body_html = _render_blocks_html(section.blocks)
        if not body_html:
            continue
        dom_id = f"section-{idx}"
        body_dom_id = f"section-body-{idx}"
        search_attr = _section_search_keywords(section)
        if section.is_debug_raw:
            appendix_toc.append(f'<a href="#{dom_id}">{html.escape(section.title)}</a>')
            secondary_sections_html.append(
                _render_section_card_html(
                    dom_id,
                    body_dom_id,
                    section.title,
                    body_html,
                    search_text=search_attr,
                    badge=section.badge,
                    badge_tone=section.badge_tone,
                    expanded=False,
                    extra_class="appendix-section",
                )
            )
        elif section.expanded:
            primary_toc.append(f'<a href="#{dom_id}">{html.escape(section.title)}</a>')
            primary_sections_html.append(
                _render_section_card_html(
                    dom_id,
                    body_dom_id,
                    section.title,
                    body_html,
                    search_text=search_attr,
                    badge=section.badge,
                    badge_tone=section.badge_tone,
                    expanded=True,
                    extra_class="primary-section",
                )
            )
        else:
            appendix_toc.append(f'<a href="#{dom_id}">{html.escape(section.title)}</a>')
            secondary_sections_html.append(
                _render_section_card_html(
                    dom_id,
                    body_dom_id,
                    section.title,
                    body_html,
                    search_text=search_attr,
                    badge=section.badge,
                    badge_tone=section.badge_tone,
                    expanded=False,
                    extra_class="secondary-section",
                )
            )

    primary_toc.append('<a href="#section-phases">执行过程</a>')
    appendix_toc.extend(
        [
            '<a href="#section-raw-result">原始结果 JSON</a>',
            '<a href="#section-raw-state">原始 Workflow 状态 JSON</a>',
            '<a href="#section-raw-markdown">完整 Markdown 源文本</a>',
        ]
    )

    hero_html = _render_hero_html(project, normalized, generated_at)
    phases_html = _render_phases_html(project, agent_views=normalized.agent_views)
    total_phase_agents = sum(len(phase.agents) for phase in project.phases)

    raw_result_display = _format_raw_result_for_display(project.result)
    raw_state_json = _pretty_json(project.to_dict())

    raw_result_search = html.escape("原始结果 json result payload", quote=True)
    raw_state_search = html.escape("原始 workflow 状态 json state", quote=True)
    raw_md_search = html.escape("完整 markdown 源文本", quote=True)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} — Workflow 报告</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f9;
      --panel: #ffffff;
      --surface: #f7f8fa;
      --surface-alt: #f0f4ff;
      --text: #1f2329;
      --text-secondary: #4e5969;
      --muted: #646a73;
      --quiet: #8f959e;
      --line: #dfe3ea;
      --line-soft: #edf0f5;
      --accent: #245bdb;
      --accent-deep: #173f9f;
      --accent-soft: #eef3ff;
      --ok: #15803d;
      --ok-bg: #ecfdf3;
      --ok-border: #abe0bd;
      --warn: #a65d05;
      --warn-bg: #fff8eb;
      --warn-border: #f5d199;
      --fail: #c9352b;
      --fail-bg: #fff3f2;
      --fail-border: #f5b8b4;
      --info: #1d4ed8;
      --info-bg: #eff6ff;
      --info-border: #bfdbfe;
      --purple: #6d28d9;
      --code-bg: #f6f8fa;
      --code-border: #e2e6ee;
      --radius: 12px;
      --radius-sm: 8px;
      --shadow: 0 2px 8px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.04);
      --mono: ui-monospace, "SFMono-Regular", "Cascadia Code", Menlo, Consolas, monospace;
      --sans: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Segoe UI", "Microsoft YaHei", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.65 var(--sans);
      -webkit-font-smoothing: antialiased;
    }}
    button {{ font: inherit; }}

    /* Hero header */
    .page-header {{
      background: linear-gradient(135deg, #0f172a 0%, #172554 52%, #1e3a8a 100%);
      color: #fff;
      padding: 30px 0 26px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }}
    .page-header .wrap {{ max-width: 1140px; margin: 0 auto; padding: 0 24px; }}
    .hero-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
    .eyebrow {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #93c5fd;
      margin-bottom: 6px;
    }}
    .page-header h1 {{
      margin: 0 0 10px;
      font-size: clamp(22px, 3vw, 28px);
      font-weight: 700;
      line-height: 1.25;
      letter-spacing: -0.015em;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 14px;
      font-size: 12.5px;
      color: rgba(255, 255, 255, 0.78);
    }}
    .meta-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.18);
      color: #fff;
    }}
    .status-ok {{ background: rgba(22, 163, 74, 0.24); border-color: rgba(134, 239, 172, 0.4); color: #bbf7d0; }}
    .status-fail {{ background: rgba(220, 38, 38, 0.26); border-color: rgba(252, 165, 165, 0.45); color: #fecaca; }}
    .status-cancel {{ background: rgba(148, 163, 184, 0.24); border-color: rgba(203, 213, 225, 0.35); color: #e2e8f0; }}
    .status-running {{ background: rgba(59, 130, 246, 0.28); border-color: rgba(147, 197, 253, 0.45); color: #dbeafe; }}
    .verdict-pill-ok {{ background: rgba(16, 185, 129, 0.24); border-color: rgba(110, 231, 183, 0.45); color: #a7f3d0; }}
    .verdict-pill-warn {{ background: rgba(245, 158, 11, 0.24); border-color: rgba(252, 211, 77, 0.48); color: #fde68a; }}
    .verdict-pill-fail {{ background: rgba(239, 68, 68, 0.26); border-color: rgba(252, 165, 165, 0.45); color: #fecaca; }}
    .verdict-pill-neutral {{ background: rgba(255, 255, 255, 0.14); border-color: rgba(255, 255, 255, 0.24); color: #f1f5f9; }}

    /* Stats grid */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
      gap: 10px;
      margin-top: 20px;
    }}
    .stat-card {{
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 10px;
      padding: 12px;
      text-align: center;
      backdrop-filter: blur(4px);
    }}
    .stat-value {{ font-size: 20px; font-weight: 700; line-height: 1.25; letter-spacing: -0.01em; }}
    .stat-label {{ font-size: 11.5px; color: rgba(255, 255, 255, 0.68); margin-top: 3px; }}

    /* Executive banner */
    .exec-banner {{
      margin-top: 18px;
      background: rgba(255, 255, 255, 0.97);
      color: var(--text);
      border-radius: var(--radius);
      padding: 18px 20px;
      border: 1px solid rgba(255, 255, 255, 0.6);
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
    }}
    .exec-banner-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line-soft);
    }}
    .exec-banner-title {{ display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 700; }}
    .exec-req {{ font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }}
    .exec-req strong {{ color: var(--text-secondary); }}
    .pool-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line-soft);
      align-items: center;
    }}
    .pool-strip-label {{ font-size: 12px; font-weight: 700; color: var(--muted); margin-right: 4px; }}
    .pool-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--surface);
      border: 1px solid var(--line);
      font-size: 12px;
      color: var(--text-secondary);
    }}
    .pool-chip .aid {{ font-family: var(--mono); font-weight: 700; color: var(--accent-deep); }}
    .pool-chip.orch {{ background: var(--accent-soft); border-color: #bfd3ff; color: var(--accent-deep); }}

    /* Sticky toolbar + TOC */
    .toolbar-wrap {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .toolbar {{
      max-width: 1140px;
      margin: 0 auto;
      padding: 10px 24px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .toolbar-controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
    .toolbar input[type="search"] {{
      flex: 1;
      min-width: 220px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      padding: 7px 12px;
      font: inherit;
      background: #fff;
      color: var(--text);
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .toolbar input[type="search"]:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(36, 91, 219, 0.12);
    }}
    .filter-count {{ font-size: 12px; color: var(--muted); white-space: nowrap; }}
    .toolbar button {{
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #fff;
      color: var(--text);
      cursor: pointer;
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 500;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
    }}
    .toolbar button:hover {{ background: var(--surface); border-color: #c8cfd8; }}
    .toolbar button.primary {{
      border-color: #bfd3ff;
      background: var(--accent-soft);
      color: var(--accent-deep);
      font-weight: 600;
    }}
    .toolbar button.primary:hover {{ background: #dfe9ff; }}
    .toc-row {{ display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; }}
    .toc-group {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
    .toc-group-label {{
      font-size: 11px;
      font-weight: 700;
      color: var(--quiet);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .toc-group a {{
      color: var(--accent-deep);
      text-decoration: none;
      border: 1px solid #d6e4ff;
      border-radius: 999px;
      padding: 2px 10px;
      font-size: 12px;
      background: #f5f8ff;
      transition: background 0.15s, border-color 0.15s;
      white-space: nowrap;
    }}
    .toc-group a:hover {{ background: #e0ebff; border-color: #adc5fa; }}
    .toc-group.toc-appendix a {{
      color: var(--muted);
      border-color: var(--line);
      background: var(--surface);
    }}
    .toc-group.toc-appendix a:hover {{ color: var(--text); background: #ebedf2; }}

    /* Main */
    main {{ max-width: 1140px; margin: 0 auto; padding: 22px 24px 68px; }}
    .search-empty {{
      display: none;
      margin: 18px 0;
      padding: 24px;
      text-align: center;
      color: var(--muted);
      background: var(--panel);
      border: 1px dashed var(--line);
      border-radius: var(--radius);
    }}

    /* Report sections */
    details.report-section {{
      margin: 14px 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
      scroll-margin-top: 110px;
    }}
    details[hidden],
    .phase-block[hidden],
    .agent-row[hidden],
    .appendix-group[hidden],
    .phases-section[hidden] {{ display: none !important; }}
    details.report-section > summary {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 14px 18px;
      cursor: pointer;
      font-weight: 700;
      font-size: 15px;
      user-select: none;
      background: #fff;
      border-bottom: 1px solid transparent;
      list-style: none;
      transition: background 0.12s;
    }}
    details.report-section > summary::-webkit-details-marker {{ display: none; }}
    details.report-section > summary::before {{
      content: "▸";
      display: inline-block;
      color: var(--quiet);
      font-size: 12px;
      transition: transform 0.15s ease;
    }}
    details.report-section[open] > summary::before {{ transform: rotate(90deg); color: var(--accent); }}
    details.report-section[open] > summary {{ border-bottom-color: var(--line-soft); background: #fafbfd; }}
    .section-title {{ flex: 1; min-width: 120px; }}
    .section-badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 9px;
      border-radius: 999px;
      white-space: nowrap;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--muted);
    }}
    .badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 999px;
      white-space: nowrap;
      border: 1px solid transparent;
    }}
    .badge-ok {{ background: var(--ok-bg); color: var(--ok); border-color: var(--ok-border); }}
    .badge-warn {{ background: var(--warn-bg); color: var(--warn); border-color: var(--warn-border); }}
    .badge-fail {{ background: var(--fail-bg); color: var(--fail); border-color: var(--fail-border); }}
    .badge-info {{ background: var(--info-bg); color: var(--info); border-color: var(--info-border); }}
    .badge-pending, .badge-neutral {{ background: var(--surface); color: var(--muted); border-color: var(--line); }}
    .section-actions {{ display: flex; gap: 6px; }}
    .section-actions button {{
      font-size: 12px;
      font-weight: 500;
      padding: 3px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      color: var(--muted);
      transition: color 0.15s, border-color 0.15s, background 0.15s;
    }}
    .section-actions button:hover {{ color: var(--accent); border-color: #bfd3ff; background: var(--accent-soft); }}
    .section-body-rich {{
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    /* Semantic blocks & Markdown prose */
    .semantic-block {{ min-width: 0; }}
    .block-title {{
      margin: 0 0 8px;
      font-size: 13.5px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .md-prose {{ color: var(--text); font-size: 14px; line-height: 1.7; word-break: break-word; }}
    .md-prose > :first-child {{ margin-top: 0; }}
    .md-prose > :last-child {{ margin-bottom: 0; }}
    .md-prose p {{ margin: 0 0 10px; }}
    .md-prose .md-h1, .md-prose .md-h2, .md-prose .md-h3,
    .md-prose .md-h4, .md-prose .md-h5, .md-prose .md-h6 {{
      margin: 16px 0 8px;
      font-weight: 700;
      line-height: 1.35;
      color: var(--text);
    }}
    .md-prose .md-h1 {{ font-size: 18px; border-bottom: 1px solid var(--line-soft); padding-bottom: 6px; }}
    .md-prose .md-h2 {{ font-size: 16px; border-bottom: 1px solid var(--line-soft); padding-bottom: 4px; }}
    .md-prose .md-h3 {{ font-size: 14.5px; }}
    .md-prose .md-h4, .md-prose .md-h5, .md-prose .md-h6 {{ font-size: 13.5px; color: var(--text-secondary); }}
    .md-prose ul, .md-prose ol, .md-list {{ margin: 6px 0 12px; padding-left: 22px; }}
    .md-prose li, .md-list li {{ margin: 4px 0; }}
    .md-prose li > p, .md-list li > p {{ margin: 0 0 4px; }}
    .md-task-list {{ list-style: none; padding-left: 4px; }}
    .md-task-item {{ display: flex; align-items: flex-start; gap: 8px; }}
    .md-task-item input[type="checkbox"] {{ margin-top: 4px; accent-color: var(--accent); }}
    .md-prose hr, .md-hr {{ border: 0; border-top: 1px solid var(--line); margin: 14px 0; }}
    .md-prose a {{ color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(36, 91, 219, 0.25); }}
    .md-prose a:hover {{ border-bottom-color: var(--accent); }}
    .md-inline-code {{
      font-family: var(--mono);
      font-size: 0.88em;
      background: var(--code-bg);
      border: 1px solid var(--code-border);
      border-radius: 5px;
      padding: 1px 5px;
      color: #0f172a;
      word-break: break-word;
    }}
    .md-code-block {{
      margin: 10px 0;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      overflow: hidden;
      background: #0f172a;
      color: #e2e8f0;
    }}
    .md-code-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 5px 12px;
      background: #1e293b;
      border-bottom: 1px solid #334155;
      font-size: 11px;
      font-family: var(--mono);
      color: #94a3b8;
      text-transform: uppercase;
    }}
    .md-code-block pre {{
      margin: 0;
      padding: 12px 14px;
      overflow-x: auto;
      font: 12.5px/1.55 var(--mono);
      white-space: pre;
    }}
    .md-quote {{
      margin: 10px 0;
      padding: 8px 14px;
      border-left: 3px solid var(--accent);
      background: var(--surface);
      color: var(--text-secondary);
      border-radius: 0 6px 6px 0;
    }}
    .md-alert {{
      margin: 10px 0;
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--info-border);
      border-left-width: 4px;
      background: var(--info-bg);
    }}
    .md-alert-title {{
      font-size: 12.5px;
      font-weight: 700;
      margin-bottom: 4px;
      color: var(--text);
    }}
    .md-alert-body > :first-child {{ margin-top: 0; }}
    .md-alert-body > :last-child {{ margin-bottom: 0; }}
    .md-alert-warning {{ border-color: var(--warn-border); border-left-color: var(--warn); background: var(--warn-bg); }}
    .md-alert-caution {{ border-color: var(--fail-border); border-left-color: var(--fail); background: var(--fail-bg); }}
    .md-alert-tip {{ border-color: var(--ok-border); border-left-color: var(--ok); background: var(--ok-bg); }}
    .md-alert-important {{ border-color: #ddd6fe; border-left-color: var(--purple); background: #f5f3ff; }}

    /* Tables */
    .md-table-wrap {{
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #fff;
      margin: 6px 0;
    }}
    table.md-table {{ width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.55; }}
    table.md-table thead th {{
      background: #f6f8fb;
      color: var(--text-secondary);
      font-weight: 700;
      text-align: left;
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }}
    table.md-table tbody td {{
      padding: 9px 12px;
      border-bottom: 1px solid var(--line-soft);
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: normal;
    }}
    table.md-table td .md-list {{ margin: 2px 0; padding-left: 18px; }}
    table.md-table td p {{ margin: 0 0 4px; }}
    table.md-table td p:last-child {{ margin-bottom: 0; }}
    table.md-table tbody tr:nth-child(even) {{ background: #fafbfd; }}
    table.md-table tbody tr:last-child td {{ border-bottom: 0; }}
    table.md-table tbody tr:hover {{ background: #f3f7ff; }}

    /* KV Grid */
    .kv-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }}
    .kv-item {{
      background: var(--surface);
      border: 1px solid var(--line-soft);
      border-radius: var(--radius-sm);
      padding: 9px 12px;
      min-width: 0;
    }}
    .kv-key {{ font-size: 11.5px; font-weight: 700; color: var(--muted); margin-bottom: 2px; }}
    .kv-val {{ font-size: 13px; color: var(--text); overflow-wrap: anywhere; word-break: normal; }}

    /* Callouts */
    .callout-card {{
      border-radius: var(--radius-sm);
      border: 1px solid var(--info-border);
      border-left-width: 4px;
      padding: 11px 14px;
      background: var(--info-bg);
    }}
    .callout-card.callout-warning {{ background: var(--warn-bg); border-color: var(--warn-border); border-left-color: var(--warn); }}
    .callout-card.callout-error {{ background: var(--fail-bg); border-color: var(--fail-border); border-left-color: var(--fail); }}
    .callout-card.callout-success {{ background: var(--ok-bg); border-color: var(--ok-border); border-left-color: var(--ok); }}
    .callout-title {{ font-size: 13px; font-weight: 700; margin-bottom: 4px; color: var(--text); }}

    /* Semantic list */
    .semantic-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      border: 1px solid var(--line-soft);
      border-radius: var(--radius-sm);
      overflow: hidden;
      background: #fff;
    }}
    .semantic-list li {{
      padding: 9px 14px;
      border-bottom: 1px solid var(--line-soft);
      display: flex;
      gap: 10px;
      align-items: baseline;
    }}
    .semantic-list li:last-child {{ border-bottom: 0; }}
    .semantic-list li:nth-child(even) {{ background: #fbfcfe; }}
    .list-idx {{
      flex: 0 0 auto;
      font-family: var(--mono);
      font-size: 11.5px;
      font-weight: 700;
      color: var(--accent);
      background: var(--accent-soft);
      padding: 1px 6px;
      border-radius: 4px;
    }}

    /* Severity badges */
    .badge-sev-critical, .badge-sev-major, .badge-sev-high {{
      background: var(--fail-bg); color: var(--fail); border: 1px solid var(--fail-border);
      padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; white-space: nowrap;
    }}
    .badge-sev-minor, .badge-sev-medium {{
      background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-border);
      padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; white-space: nowrap;
    }}
    .badge-sev-low, .badge-sev-info {{
      background: var(--info-bg); color: var(--info); border: 1px solid var(--info-border);
      padding: 1px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; white-space: nowrap;
    }}

    /* Phases & agents */
    .phases-section {{ margin: 24px 0 18px; scroll-margin-top: 110px; }}
    .section-heading-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .section-heading-row h2 {{ margin: 0; font-size: 16.5px; font-weight: 700; }}
    .section-heading-sub {{ font-size: 12px; color: var(--muted); }}
    .phase-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      margin-bottom: 12px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .phase-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 16px;
      background: #fafbfd;
      border-bottom: 1px solid var(--line-soft);
      flex-wrap: wrap;
    }}
    .phase-num {{
      font-weight: 700;
      font-size: 12.5px;
      color: var(--accent-deep);
      background: var(--accent-soft);
      border-radius: 6px;
      padding: 2px 8px;
      white-space: nowrap;
    }}
    .phase-title {{ font-weight: 700; flex: 1; min-width: 120px; }}
    .phase-duration {{ font-size: 12px; color: var(--muted); white-space: nowrap; }}
    .phase-agents {{ padding: 8px 16px 12px; }}
    .agent-row {{ padding: 10px 0; border-bottom: 1px solid var(--line-soft); font-size: 13px; }}
    .agent-row:last-child {{ border-bottom: 0; }}
    .agent-head {{ display: flex; align-items: center; gap: 6px 8px; flex-wrap: wrap; }}
    .agent-status {{
      font-size: 11px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 5px;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .agent-ok .agent-status {{ background: var(--ok-bg); color: var(--ok); }}
    .agent-cached .agent-status {{ background: #e6fffb; color: #006d75; }}
    .agent-fail .agent-status {{ background: var(--fail-bg); color: var(--fail); }}
    .agent-cancel .agent-status {{ background: var(--surface); color: var(--muted); }}
    .agent-pending .agent-status {{ background: var(--surface); color: var(--quiet); }}
    .agent-label {{ font-weight: 700; color: var(--text); }}
    .agent-chip {{
      font-size: 11.5px;
      padding: 1px 7px;
      border-radius: 5px;
      background: var(--surface);
      border: 1px solid var(--line-soft);
      color: var(--text-secondary);
      white-space: nowrap;
    }}
    .agent-chip.tool {{ background: var(--accent-soft); border-color: #d6e4ff; color: var(--accent-deep); font-weight: 600; }}
    .agent-chip.id {{ font-family: var(--mono); font-weight: 700; color: var(--text); }}
    .agent-task {{ margin-top: 5px; font-size: 12.5px; color: var(--muted); }}
    .agent-activity {{ margin-top: 4px; font-size: 12px; color: var(--text-secondary); }}
    .agent-error {{
      margin-top: 5px;
      padding: 6px 10px;
      border-radius: 6px;
      background: var(--fail-bg);
      border: 1px solid var(--fail-border);
      font-size: 12.5px;
      color: var(--fail);
    }}
    .agent-subagents {{ margin: 6px 0 0; padding-left: 18px; font-size: 12px; color: var(--muted); }}
    .empty-hint {{ color: var(--muted); font-size: 13px; padding: 8px 0; }}

    details.agent-output-drawer {{
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #fcfdff;
      overflow: hidden;
    }}
    details.agent-output-drawer > summary {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      cursor: pointer;
      font-size: 12.5px;
      font-weight: 600;
      color: var(--accent-deep);
      background: #f5f8ff;
      user-select: none;
      list-style: none;
    }}
    details.agent-output-drawer > summary::-webkit-details-marker {{ display: none; }}
    details.agent-output-drawer > summary::before {{ content: "▸"; font-size: 11px; transition: transform 0.15s ease; }}
    details.agent-output-drawer[open] > summary::before {{ transform: rotate(90deg); }}
    details.agent-output-drawer[open] > summary {{ border-bottom: 1px solid var(--line); }}
    .agent-drawer-meta {{ margin-left: auto; font-size: 11px; font-weight: 500; color: var(--muted); }}
    .agent-drawer-body {{ padding: 12px 14px; display: flex; flex-direction: column; gap: 12px; }}
    details.agent-raw-drawer {{
      border: 1px solid var(--line-soft);
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }}
    details.agent-raw-drawer > summary {{
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      color: var(--muted);
      background: var(--surface);
      user-select: none;
    }}

    /* Appendix & raw monospace */
    .appendix-group {{
      margin-top: 28px;
      padding-top: 18px;
      border-top: 1px dashed var(--line);
      scroll-margin-top: 110px;
    }}
    .appendix-section > summary {{ background: #fafbfd; color: var(--text-secondary); font-size: 14px; }}
    .raw-pre {{
      margin: 0;
      padding: 14px 16px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12.5px/1.55 var(--mono);
      background: var(--code-bg);
      color: var(--text);
      max-height: 460px;
    }}

    @media (max-width: 720px) {{
      .page-header .wrap, .toolbar, main {{ padding-left: 14px; padding-right: 14px; }}
      .section-body-rich {{ padding: 14px; }}
      .exec-banner {{ padding: 14px; }}
      .kv-grid {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      .toolbar-wrap, .section-actions {{ display: none !important; }}
    }}
  </style>
</head>
<body>
  {hero_html}
  <div class="toolbar-wrap">
    <div class="toolbar">
      <div class="toolbar-controls">
        <input id="filter" type="search" placeholder="搜索报告章节、交付表、阶段或 Agent…" oninput="filterSections(this.value)">
        <span class="filter-count" id="filter-count"></span>
        <button class="primary" type="button" onclick="setAllSections(true)">全部展开</button>
        <button type="button" onclick="setAllSections(false)">全部折叠</button>
      </div>
      <nav class="toc-row" aria-label="报告导航">
        <div class="toc-group">
          <span class="toc-group-label">主要阅读</span>
          {''.join(primary_toc)}
        </div>
        <div class="toc-group toc-appendix">
          <span class="toc-group-label">附录 / 原始数据</span>
          {''.join(appendix_toc)}
        </div>
      </nav>
    </div>
  </div>
  <main>
    <div class="search-empty" id="search-empty">未找到匹配的报告内容，请尝试清空或缩短搜索关键词。</div>
    {''.join(primary_sections_html)}
    <section class="phases-section" id="section-phases">
      <div class="section-heading-row">
        <h2>执行过程</h2>
        <span class="section-heading-sub">{len(project.phases)} 个阶段 · {total_phase_agents} 次 Agent 调用 · 点击「查看输出详情」展开结构化结果</span>
      </div>
      {phases_html}
    </section>
    <section class="appendix-group" id="appendix">
      <div class="section-heading-row">
        <h2>原始数据与调试附录</h2>
        <span class="section-heading-sub">默认折叠，完整保留未经截断的原始结果与完整状态快照用于排障溯源</span>
      </div>
      {''.join(secondary_sections_html)}
      <details class="report-section appendix-section" id="section-raw-result" data-search="{raw_result_search}">
        <summary>
          <span class="section-title">原始结果 JSON</span>
          <span class="section-badge badge-neutral">调试附录</span>
          <div class="section-actions">
            <button type="button" onclick="toggleSection('section-raw-result'); event.preventDefault();">折叠/展开</button>
            <button type="button" onclick="copyText('section-body-raw-result', this); event.preventDefault();">复制</button>
          </div>
        </summary>
        <pre class="raw-pre" id="section-body-raw-result">{html.escape(raw_result_display)}</pre>
      </details>
      <details class="report-section appendix-section" id="section-raw-state" data-search="{raw_state_search}">
        <summary>
          <span class="section-title">原始 Workflow 状态 JSON</span>
          <span class="section-badge badge-neutral">调试附录</span>
          <div class="section-actions">
            <button type="button" onclick="toggleSection('section-raw-state'); event.preventDefault();">折叠/展开</button>
            <button type="button" onclick="copyText('section-body-raw-state', this); event.preventDefault();">复制</button>
          </div>
        </summary>
        <pre class="raw-pre" id="section-body-raw-state">{html.escape(raw_state_json)}</pre>
      </details>
      <details class="report-section appendix-section markdown-source" id="section-raw-markdown" data-search="{raw_md_search}">
        <summary>
          <span class="section-title">完整 Markdown 源文本</span>
          <span class="section-badge badge-neutral">调试附录</span>
          <div class="section-actions">
            <button type="button" onclick="toggleSection('section-raw-markdown'); event.preventDefault();">折叠/展开</button>
            <button type="button" onclick="copyText('markdown-source', this); event.preventDefault();">复制</button>
          </div>
        </summary>
        <pre class="raw-pre" id="markdown-source">{html.escape(markdown)}</pre>
      </details>
    </section>
  </main>
  <script>
    function toggleSection(id) {{
      const el = document.getElementById(id);
      if (el) el.open = !el.open;
    }}
    function setAllSections(open) {{
      document.querySelectorAll('details.report-section').forEach(s => {{
        if (!s.hidden) s.open = open;
      }});
    }}
    function filterSections(query) {{
      const q = (query || '').trim().toLowerCase();
      let matches = 0;

      document.querySelectorAll('details.report-section').forEach(s => {{
        const attr = s.getAttribute('data-search') || '';
        const hay = (attr + ' ' + (s.textContent || '')).toLowerCase();
        const visible = !q || hay.includes(q);
        s.hidden = !visible;
        if (visible && q) matches += 1;
      }});

      const phasesSection = document.getElementById('section-phases');
      if (phasesSection) {{
        let visiblePhases = 0;
        phasesSection.querySelectorAll('.phase-block').forEach(phase => {{
          const pAttr = phase.getAttribute('data-search') || '';
          const pHeader = phase.querySelector('.phase-header');
          const pTitleHay = (pAttr + ' ' + (pHeader ? pHeader.textContent : '')).toLowerCase();
          const phaseMatch = !q || pTitleHay.includes(q);
          let visibleAgents = 0;
          phase.querySelectorAll('.agent-row').forEach(agent => {{
            const aAttr = agent.getAttribute('data-search') || '';
            const aHay = (aAttr + ' ' + (agent.textContent || '')).toLowerCase();
            const showAgent = !q || aHay.includes(q) || phaseMatch;
            agent.hidden = !showAgent;
            if (showAgent) visibleAgents += 1;
          }});
          const showPhase = !q || phaseMatch || visibleAgents > 0;
          phase.hidden = !showPhase;
          if (showPhase) {{
            visiblePhases += 1;
            if (q) matches += 1;
          }}
        }});
        phasesSection.hidden = q.length > 0 && visiblePhases === 0;
      }}

      const appendixGroup = document.getElementById('appendix');
      if (appendixGroup) {{
        const anyApp = Array.from(appendixGroup.querySelectorAll('details.report-section')).some(el => !el.hidden);
        appendixGroup.hidden = q.length > 0 && !anyApp;
      }}

      const countNode = document.getElementById('filter-count');
      if (countNode) {{
        countNode.textContent = q ? ('匹配 ' + matches + ' 处') : '';
      }}
      const emptyNode = document.getElementById('search-empty');
      if (emptyNode) {{
        const anySection = Array.from(document.querySelectorAll('details.report-section')).some(el => !el.hidden);
        const phasesVisible = phasesSection && !phasesSection.hidden;
        emptyNode.style.display = (q && !anySection && !phasesVisible) ? 'block' : 'none';
      }}
    }}
    async function copyText(id, btn) {{
      const node = document.getElementById(id);
      if (!node) return;
      const text = node.innerText || node.textContent || '';
      let copied = false;
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          await navigator.clipboard.writeText(text);
          copied = true;
        }}
      }} catch (_err) {{}}
      if (!copied) {{
        try {{
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          copied = document.execCommand('copy');
          document.body.removeChild(ta);
        }} catch (_err) {{}}
      }}
      if (btn && copied) {{
        const orig = btn.textContent;
        btn.textContent = '已复制';
        setTimeout(() => {{ btn.textContent = orig; }}, 1400);
      }}
    }}
  </script>
</body>
</html>
"""


def write_workflow_report_files(
    project: WorkflowProject,
    root_path: str,
    *,
    cache_root: str | None = None,
) -> WorkflowReportFiles:
    """Write full Workflow report files under the mirrored GhostAP cache root."""
    report_dir = Path(workflow_reports_dir(root_path, cache_root))
    report_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_slug = _safe_slug(project.workflow_id or project.name or "workflow")
    suffix = uuid.uuid4().hex[:8] if not project.workflow_id else stamp
    run_id = _safe_slug(f"{base_slug}-{suffix}")
    html_filename = f"{run_id}.html"
    markdown_filename = f"{run_id}.md"
    html_path = report_dir / html_filename
    markdown_path = report_dir / markdown_filename

    markdown = build_workflow_report_markdown(project)
    html_report = build_workflow_report_html(project, markdown)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")

    return WorkflowReportFiles(
        run_id=run_id,
        html_path=str(html_path),
        markdown_path=str(markdown_path),
        html_filename=html_filename,
        markdown_filename=markdown_filename,
    )
