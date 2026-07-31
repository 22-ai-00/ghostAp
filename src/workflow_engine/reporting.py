"""Full Workflow execution report generation.

The Feishu card is a compact status surface; this module writes the full,
untruncated execution payload to local report files that can be sent as an IM
attachment.
"""

from __future__ import annotations

import html
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AgentStatus, WorkflowProject

DEFAULT_WORKFLOW_CACHE_ROOT = "~/.cache/ghostAp"
_REPORT_DIR = "workflow_reports"


@dataclass(frozen=True)
class WorkflowReportFiles:
    """Paths for a generated Workflow report bundle."""

    run_id: str
    html_path: str
    markdown_path: str
    html_filename: str
    markdown_filename: str


@dataclass(frozen=True)
class _ReportSection:
    title: str
    body: str
    expanded: bool = False


def _safe_slug(value: str, *, default: str = "workflow") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:80] or default


def workflow_cache_root(cache_root: str | None = None) -> str:
    """Return the local cache root shared by Workflow runtime artifacts."""
    root = cache_root or DEFAULT_WORKFLOW_CACHE_ROOT
    return os.path.abspath(os.path.expanduser(root))


def workflow_project_cache_root(root_path: str, cache_root: str | None = None) -> str:
    """Mirror an absolute project path under ``~/.cache/ghostAp``."""
    abs_project = os.path.abspath(os.path.expanduser(root_path or "."))
    drive, tail = os.path.splitdrive(abs_project)
    parts = [part for part in Path(tail).parts if part not in (os.sep, "")]
    if drive:
        parts.insert(0, drive.rstrip(":"))
    return os.path.join(workflow_cache_root(cache_root), *parts)


def _status_text(value: Any) -> str:
    return getattr(value, "value", str(value or ""))


def _format_duration(seconds: float) -> str:
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


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _parse_json_payload(raw_result: str | None) -> Any | None:
    text = str(raw_result or "").strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _format_full_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return _pretty_json(value)


def _humanize_key(key: str) -> str:
    labels = {
        "summary": "摘要",
        "final_report": "最终报告",
        "report": "报告",
        "result": "结果",
        "output": "输出",
        "status": "状态",
        "conclusion": "结论",
        "verification": "验证",
        "reviews": "评审",
        "risks": "风险",
        "risk": "风险",
        "findings": "发现",
        "recommendations": "建议",
        "next_steps": "后续",
        "worker_findings": "执行发现",
        "parallel_results": "并行结果",
        "results": "结果",
        "agent_outputs": "Agent 原始输出",
        "raw_outputs": "原始输出",
        "logs": "日志",
    }
    return labels.get(key, key.replace("_", " "))


def _workflow_summary(project: WorkflowProject) -> str:
    metrics = project.metrics
    phase_agents = [agent for phase in project.phases for agent in phase.agents]
    total_agents = metrics.total_agents or len(phase_agents)
    completed_agents = metrics.completed_agents or sum(
        1 for agent in phase_agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED)
    )
    failed_agents = metrics.failed_agents or sum(1 for agent in phase_agents if agent.status == AgentStatus.FAILED)
    cached_agents = metrics.cached_agents or sum(1 for agent in phase_agents if agent.status == AgentStatus.CACHED)
    total_tokens = metrics.total_tokens or sum(agent.token_usage for agent in phase_agents)

    elapsed = 0.0
    if project.started_at:
        end_time = project.finished_at or datetime.now(timezone.utc).timestamp()
        elapsed = max(0.0, end_time - project.started_at)

    lines = [
        f"- 任务: {project.requirement or project.name or 'Workflow'}",
        f"- 名称: {project.name or 'Workflow'}",
        f"- 状态: {_status_text(project.status)}",
        f"- Workflow ID: {project.workflow_id or '(none)'}",
        f"- 耗时: {_format_duration(elapsed)}",
        f"- 阶段: {len(project.phases)}",
        f"- 代理: {completed_agents}/{total_agents} 完成，{failed_agents} 失败，{cached_agents} 缓存",
        f"- Token: {total_tokens}",
    ]
    run_spec = project.run_spec or {}
    orchestrator = run_spec.get("orchestrator") or {}
    if orchestrator:
        orchestrator_model = (
            "default"
            if orchestrator.get("use_default_model", True)
            else orchestrator.get("model_name") or "default"
        )
        lines.append(
            f"- 主编排绑定: {orchestrator.get('tool_name') or '(none)'}/{orchestrator_model}"
        )
    if run_spec:
        lines.append(
            "- 评审模式: "
            + ("Auto（无独立 Reviewer）" if run_spec.get("auto_reviewer") else "显式独立 Reviewer")
        )
    return "\n".join(lines)


def _phase_process(project: WorkflowProject) -> str:
    if not project.phases:
        return "本次 Workflow 没有记录阶段。"

    lines: list[str] = []
    for idx, phase in enumerate(project.phases, 1):
        agents = phase.agents
        total = len(agents)
        done = sum(1 for agent in agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED))
        failed = sum(1 for agent in agents if agent.status == AgentStatus.FAILED)
        cancelled = sum(1 for agent in agents if agent.status == AgentStatus.CANCELLED)
        duration = ""
        if phase.started_at and phase.finished_at:
            duration = f"，耗时 {_format_duration(max(0.0, phase.finished_at - phase.started_at))}"
        lines.append(f"### 阶段 {idx}: {phase.title}")
        lines.append(f"- 汇总: {done}/{total} 完成，{failed} 失败，{cancelled} 已取消{duration}")
        for agent in agents:
            agent_line = (
                f"- [{_status_text(agent.status)}] {agent.label or 'agent'}"
                f" tool={agent.tool or '(none)'}"
            )
            if agent.model:
                agent_line += f" model={agent.model}"
            if agent.role:
                agent_line += f" role={agent.role}"
            if agent.duration_s:
                agent_line += f" duration={_format_duration(agent.duration_s)}"
            if agent.token_usage:
                agent_line += f" tokens={agent.token_usage}"
            if agent.task_summary:
                agent_line += f"\n  - task: {agent.task_summary}"
            if agent.error:
                agent_line += f"\n  - error: {agent.error}"
            lines.append(agent_line)
        lines.append("")
    return "\n".join(lines).strip()


def _reviewer_evidence(project: WorkflowProject) -> str:
    """Render only committed evidence; never infer a review from selection."""
    run_spec = project.run_spec or {}
    if run_spec.get("auto_reviewer"):
        return "Auto 模式：未承诺、也未执行独立 Reviewer 调用。"
    if not project.reviewer_evidence:
        return "未记录独立 Reviewer 调用证据；本次运行不得视为已完成评审。"

    lines: list[str] = []
    for evidence in project.reviewer_evidence:
        binding = f"{evidence.tool}/{evidence.model or 'default'}"
        lines.append(
            f"### Reviewer {evidence.reviewer_index}: {binding}\n"
            f"- 状态: {evidence.status}\n"
            f"- Selection key: {evidence.selection_key or '(none)'}\n"
            f"- Fresh call: {'否' if evidence.cached else '是'}\n"
            f"- Stop reason: {evidence.stop_reason or '(missing)'}\n"
            f"- Token: {evidence.token_usage}\n"
            f"- 耗时: {_format_duration(evidence.duration_s)}"
        )
        if evidence.error:
            lines.append(f"- 错误: {evidence.error}")
        if evidence.output:
            lines.append(f"- 输出:\n{evidence.output}")
        lines.append("")
    return "\n".join(lines).strip()


def _result_sections(project: WorkflowProject) -> list[_ReportSection]:
    payload = _parse_json_payload(project.result)
    sections: list[_ReportSection] = []

    if isinstance(payload, dict):
        priority = [
            "final_report",
            "report",
            "summary",
            "conclusion",
            "result",
            "output",
            "verification",
            "reviews",
            "risks",
            "findings",
            "recommendations",
            "next_steps",
        ]
        seen: set[str] = set()
        for key in priority:
            if key not in payload:
                continue
            body = _format_full_value(payload.get(key))
            if body:
                sections.append(_ReportSection(_humanize_key(key), body, expanded=True))
                seen.add(key)

        for key in ("agent_outputs", "raw_outputs", "worker_outputs", "parallel_results", "results"):
            if key in payload and key not in seen:
                body = _format_full_value(payload.get(key))
                if body:
                    sections.append(_ReportSection(_humanize_key(key), body, expanded=False))
                    seen.add(key)

        remaining = {key: value for key, value in payload.items() if key not in seen}
        if remaining:
            sections.append(_ReportSection("其他结果字段", _pretty_json(remaining), expanded=False))
    elif payload is not None:
        sections.append(_ReportSection("执行结果", _format_full_value(payload), expanded=True))
    elif project.result:
        sections.append(_ReportSection("执行结果", str(project.result), expanded=True))
    else:
        sections.append(_ReportSection("执行结果", "本次 Workflow 没有返回最终结果。", expanded=True))

    return sections


def _report_sections(project: WorkflowProject) -> list[_ReportSection]:
    sections = [
        _ReportSection("运行摘要", _workflow_summary(project), expanded=True),
        _ReportSection("独立评审证据", _reviewer_evidence(project), expanded=True),
        *_result_sections(project),
        _ReportSection("执行过程", _phase_process(project), expanded=False),
        _ReportSection("原始结果 JSON", project.result or "", expanded=False),
        _ReportSection("原始 Workflow 状态 JSON", _pretty_json(project.to_dict()), expanded=False),
    ]
    if project.error:
        sections.insert(1, _ReportSection("错误信息", project.error, expanded=True))
    return sections


def build_workflow_report_markdown(project: WorkflowProject) -> str:
    """Return a full, untruncated Markdown report for a Workflow run."""
    title = project.name or "Workflow"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# {title} 完整报告",
        "",
        f"生成时间: {generated_at}",
        "",
    ]
    for section in _report_sections(project):
        lines.append(f"## {section.title}")
        lines.append(section.body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_summary_stats_html(project: WorkflowProject) -> str:
    """Render a visual stats grid for the HTML report hero section."""
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

    cards = [
        (_format_duration(elapsed), "总耗时", "#3370ff"),
        (f"{completed_phases}/{total_phases}", "阶段完成", "#00b578"),
        (f"{completed_agents}/{total_agents}", "代理完成", "#722ed1"),
        (f"{success_rate}%", "成功率", "#00b578" if success_rate >= 80 else "#ff4d4f"),
        (str(total_tokens), "Token 消耗", "#646a73"),
        (str(failed_agents), "失败", "#ff4d4f" if failed_agents > 0 else "#646a73"),
        (str(cached_agents), "缓存命中", "#13c2c2"),
    ]
    items = []
    for value, label, color in cards:
        items.append(
            f'<div class="stat-card">'
            f'<div class="stat-value" style="color:{color}">{html.escape(value)}</div>'
            f'<div class="stat-label">{html.escape(label)}</div>'
            f'</div>'
        )
    return f'<div class="stats-grid">{"".join(items)}</div>'


def _render_phases_html(project: WorkflowProject) -> str:
    """Render a structured phases timeline for the HTML report."""
    if not project.phases:
        return '<p class="empty-hint">本次 Workflow 没有记录阶段。</p>'

    items: list[str] = []
    for idx, phase in enumerate(project.phases, 1):
        agents = phase.agents
        total = len(agents)
        done = sum(1 for agent in agents if agent.status in (AgentStatus.DONE, AgentStatus.CACHED))
        failed = sum(1 for agent in agents if agent.status == AgentStatus.FAILED)
        cancelled = sum(1 for agent in agents if agent.status == AgentStatus.CANCELLED)
        duration = ""
        if phase.started_at and phase.finished_at:
            duration = _format_duration(max(0.0, phase.finished_at - phase.started_at))

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

        agent_rows = []
        for agent in agents:
            a_status = _status_text(agent.status)
            a_cls = {
                "done": "agent-ok", "cached": "agent-cached",
                "failed": "agent-fail", "cancelled": "agent-cancel",
            }.get(a_status.lower(), "agent-pending")
            a_dur = f' <span class="agent-meta">{_format_duration(agent.duration_s)}</span>' if agent.duration_s else ""
            a_tok = f' <span class="agent-meta">{agent.token_usage} tok</span>' if agent.token_usage else ""
            a_tool = f' <span class="agent-tool">{html.escape(agent.tool or "")}</span>' if agent.tool else ""
            a_model = f' <span class="agent-tool">{html.escape(agent.model or "")}</span>' if agent.model else ""
            a_role = f' <span class="agent-meta">{html.escape(agent.role or "")}</span>' if agent.role else ""
            a_task = f'<div class="agent-task">{html.escape(agent.task_summary or "")}</div>' if agent.task_summary else ""
            a_err = f'<div class="agent-error">{html.escape(agent.error or "")}</div>' if agent.error else ""
            agent_rows.append(
                f'<div class="agent-row {a_cls}">'
                f'<span class="agent-status">{html.escape(a_status)}</span>'
                f'<span class="agent-label">{html.escape(agent.label or "agent")}</span>'
                f'{a_tool}{a_model}{a_role}{a_dur}{a_tok}'
                f'{a_task}{a_err}'
                f'</div>'
            )

        agents_html = "".join(agent_rows) if agent_rows else '<div class="empty-hint">无代理调用</div>'
        dur_html = f'<span class="phase-duration">{html.escape(duration)}</span>' if duration else ""

        items.append(
            f'<div class="phase-block">'
            f'<div class="phase-header">'
            f'<span class="phase-num">阶段 {idx}</span>'
            f'<span class="phase-title">{html.escape(phase.title)}</span>'
            f'<span class="badge {badge_cls}">{html.escape(status_text)}</span>'
            f'{dur_html}'
            f'</div>'
            f'<div class="phase-agents">{agents_html}</div>'
            f'</div>'
        )
    return "".join(items)


def build_workflow_report_html(project: WorkflowProject, markdown: str | None = None) -> str:
    """Return a self-contained, static HTML report for a Workflow run."""
    title = project.name or "Workflow"
    markdown = markdown if markdown is not None else build_workflow_report_markdown(project)
    sections = _report_sections(project)

    nav_links = "\n".join(
        f'<a href="#section-{idx}">{html.escape(section.title)}</a>' for idx, section in enumerate(sections)
    )
    section_html = []
    for idx, section in enumerate(sections):
        body = html.escape(section.body)
        search_text = html.escape(f"{section.title}\n{section.body}".lower(), quote=True)
        expanded_attr = " open" if section.expanded else ""
        section_html.append(
            f"""
      <details class="report-section" id="section-{idx}" data-search="{search_text}"{expanded_attr}>
        <summary>
          <span class="section-title">{html.escape(section.title)}</span>
          <div class="section-actions">
            <button type="button" onclick="toggleSection('section-{idx}'); event.preventDefault();">折叠/展开</button>
            <button type="button" onclick="copyText('section-body-{idx}'); event.preventDefault();">复制</button>
          </div>
        </summary>
        <div class="section-body" id="section-body-{idx}">{body}</div>
      </details>"""
        )

    escaped_markdown = html.escape(markdown)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    status_text = _status_text(project.status)
    status_cls = {"completed": "status-ok", "failed": "status-fail", "cancelled": "status-cancel"}.get(
        status_text.lower(), "status-running"
    )
    stats_html = _render_summary_stats_html(project)
    phases_html = _render_phases_html(project)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} — Workflow 报告</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #1f2329;
      --muted: #8f959e;
      --line: #e5e6eb;
      --accent: #3370ff;
      --ok: #00b578;
      --fail: #ff4d4f;
      --warn: #faad14;
      --purple: #722ed1;
      --code-bg: #fafbfc;
      --radius: 10px;
      --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font: 14px/1.6 -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif;
      -webkit-font-smoothing: antialiased;
    }}

    /* Header */
    .page-header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      color: #fff;
      padding: 36px 0 28px;
    }}
    .page-header .wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 28px; }}
    .page-header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 8px; }}
    .page-header .meta {{ color: rgba(255,255,255,.7); font-size: 13px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
    .status-badge {{ display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 99px; font-size: 12px; font-weight: 600; }}
    .status-ok {{ background: rgba(0,181,120,.2); color: #5eff9e; }}
    .status-fail {{ background: rgba(255,77,79,.2); color: #ff9b9c; }}
    .status-cancel {{ background: rgba(255,255,255,.15); color: rgba(255,255,255,.7); }}
    .status-running {{ background: rgba(51,112,255,.2); color: #8cb4ff; }}

    /* Stats grid */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 12px;
      margin-top: 24px;
    }}
    .stat-card {{
      background: rgba(255,255,255,.08);
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 8px;
      padding: 14px 12px;
      text-align: center;
    }}
    .stat-value {{ font-size: 22px; font-weight: 700; line-height: 1.2; }}
    .stat-label {{ font-size: 12px; color: rgba(255,255,255,.6); margin-top: 4px; }}

    /* Toolbar */
    .toolbar-wrap {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: var(--shadow);
    }}
    .toolbar {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 12px 28px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .toolbar input[type="search"] {{
      flex: 1;
      min-width: 200px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      font: inherit;
      outline: none;
      transition: border-color .15s;
    }}
    .toolbar input[type="search"]:focus {{ border-color: var(--accent); }}
    .toolbar button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      cursor: pointer;
      padding: 7px 14px;
      font: inherit;
      transition: background .15s, border-color .15s;
    }}
    .toolbar button:hover {{ background: var(--bg); }}
    .toolbar button.primary {{ border-color: var(--accent); color: var(--accent); }}
    .toolbar nav {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      width: 100%;
      margin-top: 4px;
    }}
    .toolbar nav a {{
      color: var(--accent);
      text-decoration: none;
      border: 1px solid #d6e4ff;
      border-radius: 99px;
      padding: 3px 10px;
      font-size: 12px;
      background: #f0f5ff;
      transition: background .15s;
    }}
    .toolbar nav a:hover {{ background: #d6e4ff; }}

    /* Main */
    main {{ max-width: 1080px; margin: 0 auto; padding: 24px 28px 60px; }}

    /* Phase timeline */
    .phases-section {{ margin-bottom: 28px; }}
    .phases-section > h2 {{ font-size: 16px; margin-bottom: 14px; }}
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
      border-bottom: 1px solid var(--line);
      flex-wrap: wrap;
    }}
    .phase-num {{
      font-weight: 700;
      font-size: 13px;
      color: var(--accent);
      white-space: nowrap;
    }}
    .phase-title {{ font-weight: 600; flex: 1; min-width: 100px; }}
    .badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 99px;
      white-space: nowrap;
    }}
    .badge-ok {{ background: #e6fff5; color: #00875a; }}
    .badge-fail {{ background: #fff1f0; color: #cf1322; }}
    .badge-warn {{ background: #fffbe6; color: #ad6800; }}
    .badge-pending {{ background: #f0f0f0; color: #595959; }}
    .phase-duration {{ font-size: 12px; color: var(--muted); white-space: nowrap; }}
    .phase-agents {{ padding: 10px 16px; }}
    .agent-row {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      padding: 5px 0;
      font-size: 13px;
      border-bottom: 1px solid #f5f5f5;
      flex-wrap: wrap;
    }}
    .agent-row:last-child {{ border-bottom: none; }}
    .agent-status {{
      font-size: 11px;
      font-weight: 600;
      padding: 1px 6px;
      border-radius: 4px;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .agent-ok .agent-status {{ background: #e6fff5; color: #00875a; }}
    .agent-cached .agent-status {{ background: #e6fffb; color: #006d75; }}
    .agent-fail .agent-status {{ background: #fff1f0; color: #cf1322; }}
    .agent-cancel .agent-status {{ background: #f0f0f0; color: #595959; }}
    .agent-pending .agent-status {{ background: #f0f0f0; color: #8c8c8c; }}
    .agent-label {{ font-weight: 500; }}
    .agent-tool {{
      font-size: 12px;
      background: #f0f5ff;
      color: var(--accent);
      padding: 1px 6px;
      border-radius: 4px;
    }}
    .agent-meta {{ font-size: 12px; color: var(--muted); }}
    .agent-task {{ width: 100%; font-size: 12px; color: var(--muted); padding-left: 60px; }}
    .agent-error {{ width: 100%; font-size: 12px; color: var(--fail); padding-left: 60px; }}
    .empty-hint {{ color: var(--muted); font-size: 13px; padding: 8px 0; }}

    /* Report sections */
    details.report-section {{
      margin: 12px 0;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    details[hidden] {{ display: none; }}
    summary {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 14px 16px;
      cursor: pointer;
      font-weight: 600;
      user-select: none;
      border-bottom: 1px solid transparent;
      transition: background .1s;
    }}
    details[open] > summary {{ border-bottom-color: var(--line); background: #fafbfc; }}
    summary .section-title {{ flex: 1; }}
    summary .section-actions {{ display: flex; gap: 6px; }}
    summary .section-actions button {{
      font-size: 12px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      color: var(--muted);
      transition: color .15s, border-color .15s;
    }}
    summary .section-actions button:hover {{ color: var(--accent); border-color: var(--accent); }}
    .section-body {{
      padding: 16px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font: 13px/1.65 ui-monospace, "SFMono-Regular", "Cascadia Code", Menlo, Consolas, monospace;
      background: var(--code-bg);
      max-height: 600px;
    }}
    .markdown-source {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <div class="page-header">
    <div class="wrap">
      <h1>{html.escape(title)} — Workflow 报告</h1>
      <div class="meta">
        <span class="status-badge {status_cls}">{html.escape(status_text)}</span>
        <span>生成时间: {generated_at}</span>
        <span>ID: {html.escape(project.workflow_id or '—')}</span>
      </div>
      {stats_html}
    </div>
  </div>
  <div class="toolbar-wrap">
    <div class="toolbar">
      <input id="filter" type="search" placeholder="搜索报告内容…" oninput="filterSections(this.value)">
      <button class="primary" type="button" onclick="setAllSections(true)">全部展开</button>
      <button type="button" onclick="setAllSections(false)">全部折叠</button>
      <nav>{nav_links}</nav>
    </div>
  </div>
  <main>
    <div class="phases-section">
      <h2>执行过程</h2>
      {phases_html}
    </div>
    {''.join(section_html)}
    <details class="report-section markdown-source">
      <summary>
        <span class="section-title">完整 Markdown 源文本</span>
        <div class="section-actions"><button type="button" onclick="copyText('markdown-source'); event.preventDefault();">复制</button></div>
      </summary>
      <div class="section-body" id="markdown-source">{escaped_markdown}</div>
    </details>
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
      document.querySelectorAll('details.report-section').forEach(s => {{
        const hay = s.getAttribute('data-search') || s.textContent.toLowerCase();
        s.hidden = q.length > 0 && !hay.includes(q);
      }});
    }}
    async function copyText(id) {{
      const node = document.getElementById(id);
      if (!node || !navigator.clipboard) return;
      await navigator.clipboard.writeText(node.innerText || node.textContent || '');
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
    root = Path(workflow_project_cache_root(root_path, cache_root))
    report_dir = root / _REPORT_DIR
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
