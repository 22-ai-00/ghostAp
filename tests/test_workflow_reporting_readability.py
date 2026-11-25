"""Readability, semantic rendering, and XSS safety tests for Workflow reports."""

from __future__ import annotations

import json
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path

from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    SubagentProgress,
    SubagentStatus,
    WorkflowMetrics,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.reporting import (
    DEFAULT_WORKFLOW_CACHE_ROOT,
    WorkflowReportFiles,
    build_workflow_report_html,
    build_workflow_report_markdown,
    write_workflow_report_files,
)


def _build_cliproxy_nested_project() -> WorkflowProject:
    """Create a realistic Teamwork payload modeled after CLIProxyAPI run 4486548ffb98."""
    handoff = {
        "observation": "本地仓库位于 `feat/kiro-provider` 分支（提交 `40c3289e`），是 Go 1.26+ 多协议 AI 代理网关。",
        "logic_chain": [
            "1. 确认本地仓库版本基线：检出分支为 `feat/kiro-provider`。",
            "2. 逐项核对下游接入面：区分真实可用 API 与 HTTP 501 桩接口。",
        ],
        "caveats": [
            "评审流程状态：第 2 轮自动评审结果为 accepted=false，综合器已对照源码修订。",
            "Realtime / SIP 部分路由返回 HTTP 501。",
        ],
        "conclusion": (
            "当前本地仓库同时提供丰富的下游协议接入与多通道上游供应商接入：\n"
            "1. 下游入口：覆盖 `POST /v1/chat/completions`、`POST /v1/messages`、`POST /v1/responses`。\n"
            "2. 上游接入：支持 OAuth 登录、`--kiro-import` 与 `/v1/ws` 反向中继。"
        ),
        "verification": {
            "status": "remediated_by_synthesizer_from_codebase",
            "rounds": 2,
            "accepted": False,
            "round_1_summary": "failed: Claude 空闲超时。",
            "round_2_summary": "rejected (6 major, 4 minor): 初步回答存在路由及执行器归类偏差。",
            "synthesizer_reverification": "综合器直接核查源码并完成了全量修正。",
            "resolved_issues": [
                {
                    "severity": "major",
                    "issue": "误将 `oauth_scope` 列为上游执行器",
                    "resolution": "已移除 `oauth_scope`，并补齐 `sdk/cliproxy/service_executors.go:199-384`。",
                },
                {
                    "severity": "minor",
                    "issue": "默认凭据目录误写为 `auths/`",
                    "resolution": "已更正为 `~/.cli-proxy-api`。",
                },
            ],
        },
    }
    deliverables = [
        {
            "name": "一、下游接入方式（客户端 → 本地代理）",
            "items": [
                {
                    "category": "OpenAI Chat / Completions",
                    "endpoints_or_entry": "`POST /v1/chat/completions`、`POST /v1/completions`",
                    "evidence": "`internal/api/server_routes.go:74-77`",
                },
                {
                    "category": "Anthropic Messages API",
                    "endpoints_or_entry": "`POST /v1/messages`、`POST /v1/messages/count_tokens`",
                    "evidence": "`internal/api/server_routes.go:85-86`",
                },
            ],
        },
        {
            "name": "二、上游接入方式（本地代理 → 供应商）",
            "items": [
                {
                    "category": "Kiro 无交互导入",
                    "endpoints_or_entry": "`--kiro-import <json>` 与 `POST /v8/management/credentials`",
                    "evidence": "`cmd/server/main.go:158`",
                }
            ],
        },
    ]
    inner_output = {
        **handoff,
        "deliverables": deliverables,
        "next_steps": [
            "替换模板示例 `access.api-keys` 后启动服务。",
            "将 Claude Code 指向 `/v1/messages`。",
        ],
    }
    raw_result_payload = {
        "card_summary": {
            "verdict": "needs_attention",
            "conclusion": handoff["conclusion"],
            "findings": [],
            "verification": [
                {"status": "warning", "text": "[major] 误将 `oauth_scope` 列为上游执行器"}
            ],
            "deliverables": [],
            "next_steps": [],
        },
        "result": {
            "preset": "recommended",
            "handoff": handoff,
            "output": json.dumps(inner_output, ensure_ascii=False, indent=2),
            "verified_output": (
                "## 修订后的回答（旧初稿）\n\n"
                "| 接入方式 | 入口 |\n|---|---|\n| 旧执行器 | `oauth_scope` |\n"
            ),
            "verification": handoff["verification"],
        },
        "verification": handoff["verification"],
    }

    synth_agent_json = json.dumps(inner_output, ensure_ascii=False, indent=2)
    verify_r2_fenced = (
        "```json\n"
        + json.dumps(
            {
                "issues": [
                    {
                        "severity": "major",
                        "description": "将 `/v1/ws` 误归类为下游入口",
                        "evidence": "internal/wsrelay/manager.go:16-17",
                    }
                ],
                "approve": False,
            },
            ensure_ascii=False,
        )
        + "\n```"
    )

    return WorkflowProject(
        workflow_id="4486548ffb98",
        name="Workflow",
        requirement="帮我看下本地的这个都支持哪些接入方式",
        status=WorkflowStatus.COMPLETED,
        started_at=1000.0,
        finished_at=1907.0,
        metrics=WorkflowMetrics(
            total_agents=4,
            completed_agents=4,
            failed_agents=0,
            cached_agents=0,
            total_tokens=0,
            phases_completed=3,
        ),
        run_spec={
            "orchestrator_agent_id": "A1",
            "orchestrator": {
                "tool_name": "agy",
                "model_name": "gemini-pro-orchestrator-neon/xhigh/xhigh",
                "use_default_model": False,
            },
            "auto_reviewer": True,
            "agent_pool": [
                {
                    "agent_id": "A1",
                    "tool_name": "agy",
                    "model_name": "gemini-pro-orchestrator-neon",
                    "profile": "xhigh",
                    "effort": "xhigh",
                    "role": "scout",
                },
                {
                    "agent_id": "A2",
                    "tool_name": "claude",
                    "model_name": None,
                    "role": "worker",
                },
            ],
        },
        phases=[
            PhaseProgress(
                title="Scout",
                started_at=1000.0,
                finished_at=1177.0,
                agents=[
                    AgentProgress(
                        label="teamwork-scout",
                        agent_id="A1",
                        tool="agy",
                        model="gemini-pro-orchestrator-neon",
                        profile="xhigh",
                        effort="xhigh",
                        role="scout",
                        effective_model="gemini-pro-orchestrator-neon/xhigh/xhigh",
                        task_summary="Explore repository access methods.",
                        status=AgentStatus.DONE,
                        duration_s=177.0,
                        result="### 1. observation\n\n- 入口：`cmd/server/main.go`\n\n| 模块 | 说明 |\n|---|---|\n| `kiro` | 无交互导入 |\n",
                    )
                ],
            ),
            PhaseProgress(
                title="Verify",
                started_at=1200.0,
                finished_at=1524.0,
                agents=[
                    AgentProgress(
                        label="verify-r2-0",
                        agent_id="A1",
                        tool="agy",
                        model="gemini-pro-orchestrator-neon",
                        profile="xhigh",
                        effort="xhigh",
                        role="reviewer",
                        effective_model="gemini-pro-orchestrator-neon/xhigh/xhigh",
                        task_summary="Adversarial review",
                        status=AgentStatus.DONE,
                        duration_s=324.0,
                        result=verify_r2_fenced,
                    )
                ],
            ),
            PhaseProgress(
                title="Synthesize",
                started_at=1600.0,
                finished_at=1907.0,
                agents=[
                    AgentProgress(
                        label="teamwork-synthesize",
                        agent_id="A1",
                        tool="agy",
                        model="gemini-pro-orchestrator-neon",
                        profile="xhigh",
                        effort="xhigh",
                        role="synthesizer",
                        effective_model="gemini-pro-orchestrator-neon/xhigh/xhigh",
                        task_summary="Synthesize verified deliverables.",
                        status=AgentStatus.DONE,
                        duration_s=166.0,
                        result=synth_agent_json,
                        subagents=[
                            SubagentProgress(
                                source_id="sub-1",
                                status=SubagentStatus.COMPLETED,
                                progress="checked routes",
                                model="gemini",
                            )
                        ],
                    )
                ],
            ),
        ],
        result=json.dumps(raw_result_payload, ensure_ascii=False),
    )


def test_4486548_shaped_nested_payload_renders_readable_html_and_markdown() -> None:
    """Nested double-encoded Teamwork results should render cleanly without raw JSON walls."""
    project = _build_cliproxy_nested_project()
    md = build_workflow_report_markdown(project)
    html_out = build_workflow_report_html(project, md)

    # Markdown primary sections are structured and readable.
    for heading in (
        "## 运行摘要",
        "## 核心结论与摘要",
        "## 交付成果与详细内容",
        "## 分析推导与注意事项",
        "## 评审与质量验证",
        "## 执行过程",
        "## 原始结果 JSON",
        "## 原始 Workflow 状态 JSON",
    ):
        assert heading in md

    assert "| 分类 | 入口与说明 | 源码依据 |" in md or "| --- | --- |" in md
    assert "POST /v1/chat/completions" in md
    # Primary Markdown section before ## 原始结果 JSON must not dump double-encoded output JSON.
    primary_md = md.split("## 原始结果 JSON")[0]
    assert '"output": "{\\n' not in primary_md

    # HTML layout checks.
    assert '<section class="exec-banner"' in html_out
    assert '<div class="pool-strip">' in html_out
    assert "[A1]" in html_out
    assert '<table class="md-table">' in html_out
    assert '<code class="md-inline-code">POST /v1/chat/completions</code>' in html_out
    assert "主要阅读" in html_out
    assert "附录 / 原始数据" in html_out
    assert "原始数据与调试附录" in html_out
    assert 'class="agent-output-drawer"' in html_out
    assert 'class="agent-raw-drawer"' in html_out
    assert "function toggleSection" in html_out

    # Outdated verified_output draft must be flagged and collapsed, NOT expanded in primary flow.
    match_draft = re.search(
        r'<details class="report-section[^"]*" id="section-\d+"[^>]*>\s*<summary>\s*'
        r'<span class="section-title">补充参考：评审阶段初稿',
        html_out,
    )
    assert match_draft is not None
    assert " open>" not in match_draft.group(0)

    # Raw debug appendices must be collapsed by default (no open attribute).
    for raw_id in ("section-raw-result", "section-raw-state", "section-raw-markdown"):
        m_raw = re.search(rf'<details class="report-section[^"]*" id="{raw_id}"([^>]*)>', html_out)
        assert m_raw is not None
        assert "open" not in m_raw.group(1)

    # Primary HTML body (inside <main>, before phases and appendix) must not dump raw JSON or unrendered tables.
    main_html = html_out.split("<main>", 1)[1]
    before_phases_html = main_html.split('id="section-phases"')[0]
    primary_main_html = main_html.split('id="appendix"')[0]
    assert '"output": "{\\n' not in primary_main_html
    assert "|---|" not in before_phases_html
    assert "补充参考：评审阶段初稿" not in primary_main_html
    assert "补充参考：评审阶段初稿" in html_out.split('id="appendix"')[1]
    assert "max-height: 600px" not in html_out


class _XSSAuditParser(HTMLParser):
    """Verify every HTML tag and attribute is strictly from trusted report output."""

    ALLOWED_TAGS = frozenset(
        {
            "html",
            "head",
            "meta",
            "title",
            "style",
            "body",
            "header",
            "main",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "span",
            "section",
            "nav",
            "a",
            "input",
            "button",
            "details",
            "summary",
            "p",
            "ul",
            "ol",
            "li",
            "pre",
            "code",
            "table",
            "thead",
            "tbody",
            "tr",
            "th",
            "td",
            "blockquote",
            "strong",
            "em",
            "del",
            "hr",
            "br",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.tags: list[str] = []
        self.unsafe_attrs: list[tuple[str, str, str | None]] = []
        self.unsafe_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low_tag = tag.lower()
        self.tags.append(low_tag)
        for name, val in attrs:
            low_name = name.lower()
            if low_name.startswith("on"):
                if not (
                    (low_tag == "button" and low_name == "onclick")
                    or (low_tag == "input" and low_name == "oninput")
                ):
                    self.unsafe_attrs.append((low_tag, low_name, val))
            if low_name == "href" and val:
                decoded = val.strip().lower()
                if decoded.startswith(("javascript:", "data:", "vbscript:", "file:")):
                    self.unsafe_hrefs.append(val)


def test_html_escapes_xss_across_every_surface() -> None:
    """All untrusted payloads across Markdown, tables, KV, badges, and attributes must be escaped."""
    xss_script = "<script>alert('xss')</script>"
    xss_img = '<img src=x onerror="alert(2)">'
    xss_svg = "<svg onload=alert(3)>"
    xss_attr = '" onmouseover="alert(4)'
    xss_link = "[evil](javascript:alert(5))"
    xss_data_link = "[evil2](data:text/html,<script>alert(6)</script>)"

    payload = {
        "conclusion": f"Summary {xss_script} {xss_link} {xss_data_link}",
        "observation": f"Obs {xss_img}",
        "logic_chain": [f"Step 1 {xss_svg}"],
        "caveats": [f"Caveat {xss_img}"],
        "deliverables": [
            {
                "name": f"Group {xss_script}",
                "items": [
                    {
                        "category": f"Cat {xss_attr}",
                        "endpoints_or_entry": f"{xss_script} and {xss_link}",
                        "evidence": xss_svg,
                    }
                ],
            }
        ],
        "verification": {
            "status": f"bad{xss_script}",
            "rounds": 1,
            "accepted": False,
            "resolved_issues": [
                {
                    "severity": f"major{xss_script}",
                    "issue": xss_img,
                    "resolution": xss_svg,
                }
            ],
        },
        "next_steps": [f"Next {xss_script}"],
        "extra_meta": {"key<script>": f"val{xss_img}"},
    }

    project = WorkflowProject(
        workflow_id=f"wf-{xss_attr}",
        name=f"Title {xss_script}",
        requirement=f"Req {xss_img}",
        status=WorkflowStatus.FAILED,
        error=f"Err {xss_script} {xss_svg}",
        run_spec={
            "orchestrator_agent_id": f"A1{xss_attr}",
            "orchestrator": {
                "tool_name": f"tool{xss_script}",
                "model_name": f"model{xss_img}",
                "use_default_model": False,
            },
            "auto_reviewer": False,
            "agent_pool": [
                {
                    "agent_id": f"A1{xss_attr}",
                    "tool_name": f"tool{xss_script}",
                    "model_name": f"model{xss_img}",
                    "profile": f"prof{xss_svg}",
                    "effort": f"eff{xss_attr}",
                    "role": f"role{xss_script}",
                }
            ],
        },
        phases=[
            PhaseProgress(
                title=f"Phase {xss_script}",
                agents=[
                    AgentProgress(
                        label=f"agent{xss_img}",
                        agent_id="A1",
                        tool="codex",
                        model="gpt-5",
                        profile=f"max{xss_attr}",
                        effort="high",
                        role=f"worker{xss_script}",
                        effective_model=f"gpt-5/max{xss_svg}",
                        task_summary=f"Task {xss_img}",
                        current_activity=f"Act {xss_svg}",
                        error=f"AgentErr {xss_script}",
                        status=AgentStatus.FAILED,
                        result=f"# Heading {xss_script}\n\n{xss_link}\n\n| Col |\n|---|\n| {xss_img} |",
                    )
                ],
            )
        ],
        result=json.dumps(payload, ensure_ascii=False),
    )

    html_out = build_workflow_report_html(project)
    # Strip the single trusted <script>...</script> footer block at the end of the page.
    body_without_trusted_script = html_out.rsplit("<script>", 1)[0]

    parser = _XSSAuditParser()
    parser.feed(body_without_trusted_script)
    parser.close()

    assert "script" not in parser.tags
    assert "svg" not in parser.tags
    assert "img" not in parser.tags
    assert set(parser.tags) <= _XSSAuditParser.ALLOWED_TAGS
    assert parser.unsafe_attrs == []
    assert parser.unsafe_hrefs == []

    # Escaped forms must be present so content is preserved safely.
    assert "&lt;script&gt;alert" in body_without_trusted_script
    assert "&lt;img " in body_without_trusted_script
    assert "&lt;svg " in body_without_trusted_script


def test_empty_plain_and_failed_projects_render_cleanly() -> None:
    """Empty, plain-text, and failed reports must render gracefully in HTML and Markdown."""
    empty_proj = WorkflowProject(workflow_id="empty-1", name="Empty WF", status=WorkflowStatus.COMPLETED)
    empty_md = build_workflow_report_markdown(empty_proj)
    empty_html = build_workflow_report_html(empty_proj, empty_md)
    assert "本次 Workflow 没有返回最终结果" in empty_md
    assert "本次 Workflow 没有返回最终结果" in empty_html

    plain_proj = WorkflowProject(
        workflow_id="plain-1",
        name="Plain WF",
        status=WorkflowStatus.FAILED,
        error="执行失败 <script>bad</script>",
        result="Plain summary line\n\n- bullet 1\n- bullet 2",
    )
    plain_md = build_workflow_report_markdown(plain_proj)
    plain_html = build_workflow_report_html(plain_proj, plain_md)
    assert "## 错误信息" in plain_md
    assert "bullet 1" in plain_md
    assert "callout-error" in plain_html
    assert "&lt;script&gt;bad&lt;/script&gt;" in plain_html
    assert "<script>bad</script>" not in plain_html


def test_write_workflow_report_files_roundtrip() -> None:
    """Writing report files persists both HTML and Markdown under cache_root."""
    assert DEFAULT_WORKFLOW_CACHE_ROOT
    project = _build_cliproxy_nested_project()
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "repo"
        cache_dir = Path(tmpdir) / "cache"
        project_dir.mkdir()
        files: WorkflowReportFiles = write_workflow_report_files(
            project,
            root_path=str(project_dir),
            cache_root=str(cache_dir),
        )
        html_path = Path(files.html_path)
        md_path = Path(files.markdown_path)
        assert html_path.is_file()
        assert md_path.is_file()
        assert "POST /v1/chat/completions" in html_path.read_text(encoding="utf-8")
        assert "4486548ffb98" in md_path.read_text(encoding="utf-8")


def test_ux_workflow_report_preview_exists_and_aligns_with_renderer() -> None:
    """Offline UX preview artifact should exist and share key layout primitives."""
    preview_path = Path(__file__).resolve().parent.parent / "ux" / "workflow-report-preview.html"
    assert preview_path.is_file()
    preview_html = preview_path.read_text(encoding="utf-8")
    for marker in (
        'class="exec-banner"',
        'class="pool-strip"',
        'class="md-table"',
        'class="agent-output-drawer"',
        'id="appendix"',
        "function toggleSection",
    ):
        assert marker in preview_html

