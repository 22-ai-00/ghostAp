"""Tests for Workflow payload normalization and double-encoded JSON unwrapping."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from src.workflow_engine.models import (
    AgentProgress,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.report_normalizer import (
    NormalizedAgentView,
    NormalizedBlock,
    NormalizedSection,
    NormalizedWorkflowReport,
    StructuredTable,
    normalize_agent_output,
    normalize_workflow_report,
    unwrap_nested_json,
)


def _make_sample_teamwork_payload(*, accepted: bool = False, with_reverification: bool = True) -> dict[str, Any]:
    """Build a realistic CLIProxyAPI-style nested Teamwork payload."""
    resolved_issues = [
        {
            "severity": "major",
            "issue": "误将 `oauth_scope` 列为上游实际执行器",
            "resolution": "已从执行器清单移除 `oauth_scope`，并补齐 `service_executors.go:199-384`。",
        },
        {
            "severity": "minor",
            "issue": "默认凭据目录误写为 auths/",
            "resolution": "已更正为 `~/.cli-proxy-api`。",
        },
    ]
    inner_verification = {
        "status": "remediated_by_synthesizer_from_codebase" if with_reverification else "verified",
        "rounds": 2,
        "accepted": accepted,
        "round_1_summary": "failed (1 critical): 执行阶段空闲超时。",
        "round_2_summary": "rejected (1 major, 1 minor): 存在事实偏差。",
        "resolved_issues": resolved_issues,
    }
    if with_reverification:
        inner_verification["synthesizer_reverification"] = (
            "综合器直接核查源码并修正全部 2 项问题。"
        )

    deliverables = [
        {
            "name": "一、下游接入方式（客户端 → 本地代理）",
            "items": [
                {
                    "category": "OpenAI Chat",
                    "endpoints_or_entry": "`POST /v1/chat/completions`",
                    "evidence": "internal/api/server_routes.go:74-77",
                },
                {
                    "category": "Anthropic Messages",
                    "endpoints_or_entry": "`POST /v1/messages`",
                    "evidence": "internal/api/server_routes.go:85-86",
                },
            ],
        },
        {
            "name": "二、上游接入方式（本地代理 → 供应商）",
            "items": [
                {
                    "category": "Kiro 导入",
                    "endpoints_or_entry": "`--kiro-import <json>`",
                    "evidence": "cmd/server/main.go:158",
                },
            ],
        },
    ]

    conclusion = (
        "当前本地仓库同时提供丰富的下游协议接入、多通道上游供应商接入与插件扩展体系：\n"
        "1. 下游协议入口：覆盖 `/v1/chat/completions` 与 `/v1/messages`。\n"
        "2. 上游接入方式：支持 CLI OAuth 登录与 `--kiro-import` 导入。"
    )
    observation = "本地仓库位于 `feat/kiro-provider` 分支，经历 2 轮评审。"
    logic_chain = [
        "1. 确认本地分支版本基线。",
        "2. 核对下游路由与上游执行器。",
    ]
    caveats = [
        "评审流程状态：第 2 轮自动评审结果为 accepted=false，已由综合器修正。",
        "Kiro 仅支持 AWS Builder ID 与 IAM Identity Center。",
    ]
    next_steps = [
        "替换模板示例 `access.api-keys` 后启动代理。",
        "将 Claude Code 指向 `/v1/messages`。",
    ]

    handoff = {
        "observation": observation,
        "logic_chain": logic_chain,
        "caveats": caveats,
        "conclusion": conclusion,
        "verification": deepcopy(inner_verification),
    }

    output_dict = {
        "observation": observation,
        "logic_chain": logic_chain,
        "caveats": caveats,
        "conclusion": conclusion,
        "verification": deepcopy(inner_verification),
        "deliverables": deliverables,
        "next_steps": next_steps,
    }

    verified_output = "## 修订后的回答\n\n| 接入方式 | 入口 |\n|---|---|\n| OpenAI | `/v1/chat/completions` |"
    feedback = (
        "[major] 误将 `oauth_scope` 列为上游实际执行器\n"
        "Evidence: oauth_scope_executor.go 仅提供 ForAPIKey()"
    )
    runtime_verification = {
        "accepted": accepted,
        "output": verified_output,
        "feedback": feedback,
        "rounds": 2,
        "failures": [],
        "reviews": [
            {
                "round": 1,
                "verifier": "reviewer",
                "approve": False,
                "issues": [
                    {
                        "severity": "critical",
                        "description": "Output failed with timeout",
                        "evidence": "Claude 空闲超时",
                    }
                ],
            },
            {
                "round": 2,
                "verifier": "reviewer",
                "approve": accepted,
                "issues": [
                    {
                        "severity": "major",
                        "description": "误将 `oauth_scope` 列为上游实际执行器",
                        "evidence": "oauth_scope_executor.go 仅提供 ForAPIKey()",
                    }
                ],
            },
        ],
        "evidence": [
            {
                "round": 1,
                "verifier": "reviewer",
                "severity": "critical",
                "description": "Output failed with timeout",
                "evidence": "Claude 空闲超时",
            },
            {
                "round": 2,
                "verifier": "reviewer",
                "severity": "major",
                "description": "误将 `oauth_scope` 列为上游实际执行器",
                "evidence": "oauth_scope_executor.go 仅提供 ForAPIKey()",
            },
        ],
    }

    return {
        "card_summary": {
            "verdict": "passed" if accepted else "needs_attention",
            "conclusion": conclusion,
            "findings": [],
            "verification": [{"status": "passed" if accepted else "warning", "text": feedback}],
            "deliverables": [],
            "next_steps": [],
        },
        "result": {
            "preset": "recommended",
            "handoff": handoff,
            "output": json.dumps(output_dict, ensure_ascii=False, indent=2),
            "verified_output": verified_output,
            "scout": "## Scout 发现\n\n- 路由文件 `internal/api/server_routes.go`",
            "plan": None,
            "verification": deepcopy(runtime_verification),
        },
        "verification": deepcopy(runtime_verification),
    }


def test_unwrap_nested_json_handles_double_encoded_and_fenced_without_touching_prose():
    """Verify recursive unwrapping across raw JSON, fenced JSON, depth limits, and plain text."""
    nested = {
        "outer": json.dumps(
            {
                "inner": '```json\n{"deep": {"value": 42}}\n```',
                "text": "[INFO] keep literal bracket string",
                "bad": "{not valid json",
            },
            ensure_ascii=False,
        ),
        "arr": ['[1, 2, {"nested": "{\\"ok\\": true}"}]'],
        "markdown_with_fence": "## Title\n```json\n{\"stay_string\": 1}\n```",
    }
    original_snapshot = deepcopy(nested)

    unwrapped = unwrap_nested_json(nested, max_depth=4)
    assert nested == original_snapshot  # Pure function does not mutate input
    assert unwrapped["outer"]["inner"]["deep"]["value"] == 42
    assert unwrapped["outer"]["text"] == "[INFO] keep literal bracket string"
    assert unwrapped["outer"]["bad"] == "{not valid json"
    assert unwrapped["arr"] == [[1, 2, {"nested": {"ok": True}}]]
    assert unwrapped["markdown_with_fence"] == nested["markdown_with_fence"]

    # Depth limiting stops decoding at exact layer count
    triple = json.dumps({"a": json.dumps({"b": json.dumps({"c": 1})})})
    d0 = unwrap_nested_json(triple, max_depth=0)
    assert d0 == triple
    d1 = unwrap_nested_json(triple, max_depth=1)
    assert isinstance(d1, dict) and isinstance(d1["a"], str)
    d2 = unwrap_nested_json(triple, max_depth=2)
    assert isinstance(d2["a"], dict) and isinstance(d2["a"]["b"], str)
    d3 = unwrap_nested_json(triple, max_depth=3)
    assert d3["a"]["b"]["c"] == 1

    # Single-line and tilde JSON fences
    assert unwrap_nested_json("~~~JSON\n{\"k\": \"v\"}\n~~~") == {"k": "v"}
    assert unwrap_nested_json('```json {"x": [1, 2]} ```') == {"x": [1, 2]}


def test_normalize_agent_output_covers_all_payload_shapes():
    """Verify per-agent normalization across empty, error, markdown, fenced review, and handoff JSON."""
    empty_view = normalize_agent_output(None)
    assert empty_view == NormalizedAgentView(summary_blocks=(), raw_text="", has_structured_view=False)

    err_view = normalize_agent_output("❌ Claude 空闲超时")
    assert err_view.raw_text == "❌ Claude 空闲超时"
    assert err_view.has_structured_view is False
    assert err_view.summary_blocks == (
        NormalizedBlock(kind="callout", text="❌ Claude 空闲超时", level="error"),
    )

    md_raw = "## 修订后的回答\n\n| 列A | 列B |\n|---|---|\n| 1 | 2 |"
    md_view = normalize_agent_output(md_raw)
    assert md_view.has_structured_view is True
    assert md_view.summary_blocks[0].kind == "markdown"
    assert md_view.summary_blocks[0].text == md_raw

    fenced_review = (
        '```json\n'
        '{"issues": [{"severity": "major", "description": "遗漏插件", "evidence": "pluginapi"}], "approve": false}\n'
        '```'
    )
    review_view = normalize_agent_output(fenced_review)
    assert review_view.raw_text == fenced_review
    assert review_view.has_structured_view is True
    assert any(b.kind == "callout" and b.level == "warning" for b in review_view.summary_blocks)
    table_blocks = [b for b in review_view.summary_blocks if b.kind == "table"]
    assert len(table_blocks) == 1
    assert table_blocks[0].table == StructuredTable(
        columns=("级别", "问题描述", "源码依据"),
        rows=(("major", "遗漏插件", "pluginapi"),),
        title="发现问题",
    )

    # Double-encoded agent output
    double_agent = json.dumps(
        {
            "output": json.dumps(
                {
                    "conclusion": "分析完成，见下方建议。",
                    "next_steps": ["运行测试"],
                },
                ensure_ascii=False,
            )
        },
        ensure_ascii=False,
    )
    double_view = normalize_agent_output(double_agent)
    assert double_view.has_structured_view is True
    assert all(b.kind != "json" for b in double_view.summary_blocks)
    assert any("分析完成，见下方建议。" in b.text for b in double_view.summary_blocks)


def test_normalize_workflow_report_deduplicates_teamwork_and_collapses_superseded_draft():
    """Verify Teamwork unwrapping, deduplication, localized tables, and superseded verified_output."""
    payload = _make_sample_teamwork_payload(accepted=False, with_reverification=True)
    project = WorkflowProject(
        workflow_id="4486548ffb98",
        name="teamwork-recommended",
        status=WorkflowStatus.COMPLETED,
        requirement="帮我看下本地的这个都支持哪些接入方式",
        result=json.dumps(payload, ensure_ascii=False),
        phases=[
            PhaseProgress(
                title="Scout",
                agents=[
                    AgentProgress(label="teamwork-scout", result="## Scout\n\n- 路由文件"),
                ],
            ),
            PhaseProgress(
                title="Execute",
                agents=[
                    AgentProgress(label="teamwork-worker-1", result="❌ Claude 空闲超时"),
                ],
            ),
            PhaseProgress(
                title="Verify",
                agents=[
                    AgentProgress(
                        label="verify-r2-0",
                        result=(
                            '```json\n{"issues": [{"severity": "major", '
                            '"description": "d", "evidence": "e"}], "approve": false}\n```'
                        ),
                    ),
                ],
            ),
        ],
    )

    report = normalize_workflow_report(project)
    assert isinstance(report, NormalizedWorkflowReport)
    assert report.verdict == "needs_attention"
    assert report.verdict_label == "需处理"
    assert report.headline_conclusion
    assert report.headline_conclusion.count("`") % 2 == 0

    by_id = {sec.id: sec for sec in report.sections}
    assert list(by_id.keys()) == [
        "summary-conclusion",
        "deliverables-details",
        "reasoning-caveats",
        "verification-qa",
        "next-steps",
        "verified-output-draft",
        "agent-raw-extra",
    ]

    # 1. Conclusion & observation deduplicated across card_summary/handoff/output
    sec1_texts = [b.text for b in by_id["summary-conclusion"].blocks]
    assert sum("当前本地仓库同时提供" in t for t in sec1_texts) == 1
    assert sum("feat/kiro-provider" in t for t in sec1_texts) == 1

    # 2. Grouped deliverables turned into localized StructuredTable blocks
    deliv_sec = by_id["deliverables-details"]
    assert deliv_sec.expanded is True
    assert deliv_sec.badge == "2 组 / 3 项"
    tables = [b.table for b in deliv_sec.blocks if b.kind == "table"]
    assert len(tables) == 2
    assert tables[0] is not None
    assert tables[0].columns == ("分类", "入口与说明", "源码依据")
    assert len(tables[0].rows) == 2

    # 3. Superseded verified_output moved to collapsed warning section
    draft_sec = by_id["verified-output-draft"]
    assert draft_sec.title == "补充参考：评审阶段初稿（部分待修订）"
    assert draft_sec.expanded is False
    assert draft_sec.badge_tone == "warning"
    assert draft_sec.blocks[0].kind == "callout"
    assert draft_sec.blocks[0].level == "warning"
    assert any("修订后的回答" in b.text for b in draft_sec.blocks[1:])

    # 4. Verification section deduplicates top-level vs result.verification and feedback wall
    ver_sec = by_id["verification-qa"]
    assert ver_sec.expanded is True
    ver_kinds = [b.kind for b in ver_sec.blocks]
    assert "kv" in ver_kinds
    assert "callout" in ver_kinds  # synthesizer_reverification
    assert ver_kinds.count("table") == 2  # resolved_issues + evidence
    # Raw feedback wall is NOT repeated because evidence covers it
    assert not any(b.title == "评审反馈" for b in ver_sec.blocks)
    # Round summaries rendered cleanly
    assert any(b.title == "第 1 轮评审摘要" for b in ver_sec.blocks)
    assert any(b.title == "第 2 轮评审摘要" for b in ver_sec.blocks)

    # 5. Extra section collapsed with preset & scout, skipping null plan
    extra_sec = by_id["agent-raw-extra"]
    assert extra_sec.expanded is False
    assert any(b.kind == "markdown" and "Scout" in b.text for b in extra_sec.blocks)
    kv_block = next(b for b in extra_sec.blocks if b.kind == "kv")
    assert ("预设方案", "recommended") in kv_block.pairs
    assert all(k != "执行计划" for k, _ in kv_block.pairs)

    # 6. All (phase_idx, agent_idx) agent views populated
    assert set(report.agent_views.keys()) == {(0, 0), (1, 0), (2, 0)}
    assert report.agent_views[(0, 0)].summary_blocks[0].kind == "markdown"
    assert report.agent_views[(1, 0)].summary_blocks[0].level == "error"
    assert report.agent_views[(2, 0)].has_structured_view is True


def test_normalize_workflow_report_verified_output_accepted_and_identical_cases():
    """Verify verified_output stays expanded when accepted=True and is deduped when identical."""
    accepted_payload = _make_sample_teamwork_payload(accepted=True, with_reverification=False)
    project_ok = WorkflowProject(
        workflow_id="ok-1",
        status=WorkflowStatus.COMPLETED,
        result=json.dumps(accepted_payload, ensure_ascii=False),
    )
    report_ok = normalize_workflow_report(project_ok)
    assert report_ok.verdict == "passed"
    assert "verified-output-draft" not in {s.id for s in report_ok.sections}
    deliv_sec = next(s for s in report_ok.sections if s.id == "deliverables-details")
    assert any("修订后的回答" in b.text for b in deliv_sec.blocks)

    # When verified_output equals string output, do not duplicate it
    dup_payload = {
        "result": {
            "output": "## Same Report\n\nDetails",
            "verified_output": "## Same Report\n\nDetails",
            "verification": {"accepted": True, "rounds": 1},
        }
    }
    report_dup = normalize_workflow_report(
        WorkflowProject(status=WorkflowStatus.COMPLETED, result=json.dumps(dup_payload))
    )
    deliv_dup = next(s for s in report_dup.sections if s.id == "deliverables-details")
    assert len([b for b in deliv_dup.blocks if "Same Report" in b.text]) == 1


def test_normalize_workflow_report_preserves_legacy_extra_fields_and_uncovered_feedback():
    """Verify legacy report payloads, uncovered revision failures, and error states are preserved."""
    sentinel = "FINAL_SENTINEL_AFTER_LONG_CONTENT"
    legacy_payload = {
        "final_report": ("完整报告 " * 50) + sentinel,
        "verification": {
            "summary": "验证通过",
            "raw": "<script>alert('x')</script>",
            "feedback": "[major] Bad issue\nEvidence: bad.py:1\nRevision failed: reviser timeout",
            "evidence": [
                {
                    "round": 1,
                    "verifier": "v1",
                    "severity": "major",
                    "description": "Bad issue",
                    "evidence": "bad.py:1",
                }
            ],
            "reviews": [
                {
                    "round": 1,
                    "verifier": "v1",
                    "approve": False,
                    "conclusion": "Needs retry after fix",
                    "issues": [
                        {
                            "severity": "major",
                            "description": "Bad issue",
                            "evidence": "bad.py:1",
                        }
                    ],
                }
            ],
        },
        "agent_outputs": [
            {"label": "auditor", "output": "agent raw output " * 30},
        ],
        "custom_metrics": {"latency_ms": 120},
    }
    report = normalize_workflow_report(
        WorkflowProject(status=WorkflowStatus.COMPLETED, result=json.dumps(legacy_payload, ensure_ascii=False))
    )
    by_id = {s.id: s for s in report.sections}

    deliv_texts = "\n".join(b.text for b in by_id["deliverables-details"].blocks)
    assert sentinel in deliv_texts

    ver_sec = by_id["verification-qa"]
    ver_kv = next(b for b in ver_sec.blocks if b.kind == "kv")
    assert ("摘要", "验证通过") in ver_kv.pairs
    assert ("原始信息", "<script>alert('x')</script>") in ver_kv.pairs
    # Revision failure note from feedback is NOT dropped even when issue itself is in evidence
    assert any(b.title == "评审反馈" and "Revision failed: reviser timeout" in b.text for b in ver_sec.blocks)
    assert any(b.title == "各轮评审补充说明" and "Needs retry after fix" in "\n".join(b.items) for b in ver_sec.blocks)

    extra_sec = by_id["agent-raw-extra"]
    assert any("agent raw output" in b.text for b in extra_sec.blocks)
    assert any(b.kind == "kv" and ("latency ms", "120") in b.pairs for b in extra_sec.blocks)

    # Empty / failed / cancelled states
    failed_report = normalize_workflow_report(
        WorkflowProject(status=WorkflowStatus.FAILED, error="All pool fallbacks exhausted")
    )
    assert failed_report.verdict == "failed"
    assert failed_report.verdict_label == "失败"
    assert "All pool fallbacks exhausted" in failed_report.headline_conclusion

    cancelled_report = normalize_workflow_report(
        WorkflowProject(status=WorkflowStatus.CANCELLED, result=None)
    )
    assert cancelled_report.verdict == "cancelled"
    assert cancelled_report.verdict_label == "已取消"

    assert isinstance(StructuredTable(columns=("a",), rows=(("1",),)), StructuredTable)
    assert isinstance(NormalizedSection(id="s", title="t"), NormalizedSection)


def test_normalize_workflow_report_handles_non_dict_results_and_heterogeneous_tables():
    """Verify non-dict Markdown/list results and heterogeneous table columns."""
    md_report = normalize_workflow_report(
        WorkflowProject(
            status=WorkflowStatus.COMPLETED,
            result="# 部署文档\n\n- 步骤 1\n- 步骤 2",
        )
    )
    assert md_report.verdict == "passed"
    assert md_report.sections[0].id == "deliverables-details"
    assert md_report.sections[0].blocks[0].kind == "markdown"

    list_payload = [
        {"category": "API", "path": "/v1/models"},
        {"category": "CLI", "evidence": "cmd/main.go"},
    ]
    list_report = normalize_workflow_report(
        WorkflowProject(
            status=WorkflowStatus.COMPLETED,
            result=json.dumps(list_payload, ensure_ascii=False),
        )
    )
    assert list_report.sections[0].blocks[0].kind == "table"
    tbl = list_report.sections[0].blocks[0].table
    assert tbl is not None
    assert tbl.columns == ("分类", "路径", "源码依据")
    assert tbl.rows == (("API", "/v1/models", ""), ("CLI", "", "cmd/main.go"))

    # Mixed irregular array in agent output falls back to json block with has_structured_view=False
    irregular_agent = normalize_agent_output('[1, {"nested": [2, 3]}]')
    assert irregular_agent.has_structured_view is False
    assert irregular_agent.summary_blocks[0].kind == "json"
