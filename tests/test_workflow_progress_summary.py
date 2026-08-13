"""Tests for phase summaries, result delivery, and current operation display."""

from __future__ import annotations

import json

from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    ReviewerEvidence,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer import (
    WorkflowProgressRenderer,
    render_completion_cards,
)


def _make_agent(
    label: str,
    status: AgentStatus,
    *,
    tool: str = "coco",
    error: str | None = None,
    duration_s: float = 1.0,
    result: str | None = None,
    current_activity: str = "",
    activity_updated_at: float | None = None,
) -> AgentProgress:
    agent = AgentProgress(
        label=label,
        tool=tool,
        status=status,
        duration_s=duration_s,
        error=error,
        current_activity=current_activity,
    )
    # RED-contract fixture: production does not expose these result-ledger
    # fields yet. Bypass Pydantic's unknown-field filtering so this test can
    # describe the renderer's next public input contract without changing
    # production code during the RED phase.
    object.__setattr__(agent, "result", result)
    object.__setattr__(agent, "activity_updated_at", activity_updated_at)
    return agent


def _make_project(phase_title: str, agents: list[AgentProgress]) -> WorkflowProject:
    return WorkflowProject(
        name="test",
        phases=[PhaseProgress(title=phase_title, agents=agents, started_at=1000.0)],
    )


def _flatten_text(elements: list[dict]) -> str:
    """Recursively extract all string content from card elements (including dict headers)."""
    parts: list[str] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        content = el.get("content")
        if isinstance(content, str):
            parts.append(content)
        header = el.get("header")
        if isinstance(header, str):
            parts.append(header)
        elif isinstance(header, dict):
            # Structured collapsible_panel header: {"title": {"tag": ..., "content": ...}, "template": ...}
            title = header.get("title")
            if isinstance(title, dict):
                tc = title.get("content")
                if isinstance(tc, str):
                    parts.append(tc)
        # Column sets / columns recurse into their elements
        nested = el.get("elements")
        if isinstance(nested, list):
            parts.append(_flatten_text(nested))
        columns = el.get("columns")
        if isinstance(columns, list):
            for col in columns:
                if isinstance(col, dict):
                    col_els = col.get("elements")
                    if isinstance(col_els, list):
                        parts.append(_flatten_text(col_els))
    return "\n".join(parts)


def _markdown_blocks_containing(card: dict, needle: str) -> list[str]:
    matches: list[str] = []
    stack: list[object] = [card]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            content = node.get("content")
            if node.get("tag") == "markdown" and isinstance(content, str) and needle in content:
                matches.append(content)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return matches


def _card_size_bytes(card: dict) -> int:
    return len(json.dumps(card, ensure_ascii=False).encode("utf-8", errors="surrogatepass"))


def test_phase_header_has_completed_summary_with_large_phase() -> None:
    """Large phase (25 agents) — 已完成 M/N appears near header."""
    agents: list[AgentProgress] = []
    agents += [_make_agent(f"running-{i}", AgentStatus.RUNNING) for i in range(2)]
    agents.append(_make_agent("failed-0", AgentStatus.FAILED, error="boom"))
    agents += [_make_agent(f"done-{i}", AgentStatus.DONE) for i in range(18)]
    agents += [_make_agent(f"cached-{i}", AgentStatus.CACHED) for i in range(3)]
    agents.append(_make_agent("pending-0", AgentStatus.PENDING))

    project = _make_project("Large Phase", agents)
    renderer = WorkflowProgressRenderer(project)
    card = renderer.render_progress_card()

    text = _flatten_text(card["elements"])
    # 18 DONE + 3 CACHED = 21 completed / 25 total
    assert "已完成 21/25" in text, f"Expected '已完成 21/25' in: {text[:800]}"


def test_terminal_results_form_complete_ordered_ledger_across_cards() -> None:
    """10+ terminal results remain reconstructable instead of being hidden."""
    statuses = (AgentStatus.DONE, AgentStatus.CACHED, AgentStatus.FAILED)
    markers = [f"RESULT{i:02d}SAFE" for i in range(15)]
    agents = [
        _make_agent(
            f"agent-{i}",
            statuses[i % len(statuses)],
            error="agent failed" if statuses[i % len(statuses)] == AgentStatus.FAILED else None,
            result=marker,
        )
        for i, marker in enumerate(markers)
    ]
    project = _make_project("Big Phase", agents)
    project.status = WorkflowStatus.COMPLETED

    cards = render_completion_cards(project)

    assert len(cards) >= 2
    assert all(isinstance(card, dict) and "header" in card and "elements" in card for card in cards)
    status_text = _flatten_text(cards[0]["elements"])
    ledger_texts = [_flatten_text(card["elements"]) for card in cards[1:]]
    ledger_text = "\n".join(ledger_texts)

    assert "执行过程" in status_text
    assert all("进度 " not in text and "当前执行中" not in text for text in ledger_texts)
    assert all(marker not in status_text for marker in markers)
    assert all(ledger_text.count(marker) == 1 for marker in markers)
    positions = [ledger_text.index(marker) for marker in markers]
    assert positions == sorted(positions), "terminal results must retain agent call order"


def test_single_oversized_result_splits_without_losing_content() -> None:
    """One large terminal result spans result pages and every page stays bounded."""
    chunks = [f"RESULTCHUNK{i:03d} " + ("x" * 700) for i in range(72)]
    agent = _make_agent("large-result", AgentStatus.DONE, result="\n".join(chunks))
    project = _make_project("Large Result", [agent])
    project.status = WorkflowStatus.COMPLETED

    cards = render_completion_cards(project)

    assert len(cards[1:]) >= 2, "a single oversized result must span multiple ledger pages"
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])
    markers = [f"RESULTCHUNK{i:03d}" for i in range(72)]
    assert all(ledger_text.count(marker) == 1 for marker in markers)
    assert [ledger_text.index(marker) for marker in markers] == sorted(
        ledger_text.index(marker) for marker in markers
    )


def test_workflow_result_envelope_renders_readable_sections_without_raw_json() -> None:
    """Structured final output is unpacked instead of pretty-printed as JSON."""
    result = {
        "card_summary": {
            "verdict": "passed",
            "conclusion": "任务已完成，完整结果见报告。",
            "findings": [],
            "verification": [
                {"status": "passed", "text": "执行 Agent 已完成自检。"},
            ],
            "deliverables": [],
            "next_steps": [],
        },
        "result": (
            "先完成现状分析。\n\n"
            "## 架构结论\n\n"
            "- ENTRY_MARKER：统一安装入口。\n"
            "- API_MARKER：核心接口保持兼容。\n\n"
            + ("完整架构分析与验证证据。" * 120)
        ),
        "verification": [
            {"status": "passed", "text": "执行 Agent 已完成自检。"},
        ],
    }
    project = _make_project("Readable Final", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = json.dumps(result, ensure_ascii=False)

    cards = render_completion_cards(project)
    ledger_cards = [
        card for card in cards if card.get("_workflow_page_key", (None,))[0] == "ledger"
    ]
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in ledger_cards)

    assert "Workflow 最终结果" in ledger_text
    assert "已完成 · 验证通过" in ledger_text
    assert "**结论**" in ledger_text
    assert "**验证**" in ledger_text
    assert "## 架构结论" in ledger_text
    assert ledger_text.count("ENTRY_MARKER") == 1
    assert ledger_text.count("API_MARKER") == 1
    assert '"card_summary"' not in ledger_text
    assert '"verdict"' not in ledger_text
    assert "\\n" not in ledger_text
    assert "**关键发现**" not in ledger_text
    assert "**交付物**" not in ledger_text
    assert "**下一步**" not in ledger_text
    assert any("collapsible_panel" in str(card) for card in ledger_cards)
    assert all("down_outlined" in str(card) for card in ledger_cards)
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)


def test_workflow_result_envelope_object_body_never_falls_back_to_raw_json() -> None:
    """Object/list bodies remain readable without exposing serialization syntax."""
    result = {
        "card_summary": {
            "verdict": "needs_attention",
            "conclusion": "仍有一项需要处理。",
        },
        "result": {
            "summary": "OBJECT_SUMMARY_MARKER",
            "findings": [
                {"severity": "high", "text": "OBJECT_FINDING_MARKER"},
            ],
            "next_steps": ["OBJECT_NEXT_MARKER"],
        },
    }
    project = _make_project("Readable Object", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = json.dumps(result, ensure_ascii=False)

    cards = render_completion_cards(project)
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

    assert "需处理" in ledger_text
    assert "OBJECT_SUMMARY_MARKER" in ledger_text
    assert "OBJECT_FINDING_MARKER" in ledger_text
    assert "OBJECT_NEXT_MARKER" in ledger_text
    assert '"summary"' not in ledger_text
    assert '"findings"' not in ledger_text
    assert "{" not in ledger_text
    assert "}" not in ledger_text


def test_double_encoded_workflow_envelope_keeps_brief_and_list_result() -> None:
    """Bridge-compatible nested JSON wrappers keep their structured summary."""
    envelope = {
        "card_summary": {
            "verdict": "passed",
            "conclusion": "DOUBLECONCLUSIONMARKER",
            "verification": [
                {"status": "passed", "text": "DOUBLEVERIFYMARKER"},
            ],
        },
        "result": [
            "LISTRESULTONE",
            {"status": "passed", "text": "LISTRESULTTWO"},
        ],
    }
    project = _make_project("Double Encoded", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = json.dumps(
        json.dumps(envelope, ensure_ascii=False),
        ensure_ascii=False,
    )

    cards = render_completion_cards(project)
    status_text = _flatten_text(cards[0]["elements"])
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

    for marker in (
        "DOUBLECONCLUSIONMARKER",
        "DOUBLEVERIFYMARKER",
        "LISTRESULTONE",
        "LISTRESULTTWO",
    ):
        assert ledger_text.count(marker) == 1
    assert "DOUBLECONCLUSIONMARKER" in status_text
    assert "DOUBLEVERIFYMARKER" in status_text
    assert "待确认" not in status_text
    assert "验证通过" in ledger_text
    assert '"card_summary"' not in ledger_text
    assert "\\n" not in ledger_text


def test_agent_and_reviewer_envelopes_are_never_rendered_as_raw_json() -> None:
    """Every terminal result surface shares the readable envelope projection."""
    envelope = json.dumps(
        {
            "card_summary": {
                "verdict": "passed",
                "conclusion": "NESTEDRESULTCONCLUSION",
            },
            "result": "NESTEDRESULTBODY",
        },
        ensure_ascii=False,
    )
    agent = _make_agent("structured-agent", AgentStatus.DONE, result=envelope)
    project = _make_project("Structured Results", [agent])
    project.status = WorkflowStatus.COMPLETED
    project.result = envelope
    project.reviewer_evidence = [
        ReviewerEvidence(
            reviewer_index=1,
            display_name="Independent Review",
            tool="claude",
            status="passed",
            output=envelope,
        )
    ]

    cards = render_completion_cards(project)
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

    assert ledger_text.count("NESTEDRESULTCONCLUSION") == 3
    assert ledger_text.count("NESTEDRESULTBODY") == 3
    assert '"card_summary"' not in ledger_text
    assert '"verdict"' not in ledger_text
    assert '"result"' not in ledger_text
    assert "\\n" not in ledger_text
    assert "状态: 通过" in ledger_text


def test_fenced_result_envelope_is_unwrapped_and_redacted() -> None:
    secret = "sk-1234567890abcdefghijklmnop"
    envelope = {
        "card_summary": {
            "verdict": "passed",
            "conclusion": f"FENCEDCONCLUSION {secret}",
        },
        "result": "FENCEDBODYMARKER",
    }
    project = _make_project("Fenced Envelope", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = (
        "```json\n"
        + json.dumps(envelope, ensure_ascii=False)
        + "\n```"
    )

    cards = render_completion_cards(project)
    card_text = "\n".join(_flatten_text(card["elements"]) for card in cards)

    assert card_text.count("FENCEDCONCLUSION") == 2
    assert card_text.count("FENCEDBODYMARKER") == 1
    assert secret not in card_text
    assert r"<redacted:api\_key>" in card_text
    assert '"card_summary"' not in card_text
    assert '"verdict"' not in card_text


def test_prefixed_and_long_fenced_envelopes_are_unwrapped() -> None:
    envelope = json.dumps(
        {
            "card_summary": {
                "verdict": "passed",
                "conclusion": "WRAPPEDCONCLUSION",
            },
            "result": "WRAPPEDBODY",
        },
        ensure_ascii=False,
    )
    wrappers = (
        f"Result:\n{envelope}",
        f"Result:\n```json\n{envelope}\n```",
        f"````json\n{envelope}\n````",
        f"~~~~json\n{envelope}\n~~~~~",
    )
    for wrapper in wrappers:
        project = _make_project("Wrapped Envelope", [])
        project.status = WorkflowStatus.COMPLETED
        project.result = wrapper

        cards = render_completion_cards(project)
        card_text = "\n".join(_flatten_text(card["elements"]) for card in cards)

        assert card_text.count("WRAPPEDCONCLUSION") == 2
        assert card_text.count("WRAPPEDBODY") == 1
        assert '"card_summary"' not in card_text
        assert '"result"' not in card_text


def test_lone_surrogate_in_result_is_replaced_before_wire_serialization() -> None:
    project = _make_project("Surrogate Result", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = (
        '{"card_summary":{"verdict":"passed",'
        '"conclusion":"SURROGATECONCLUSION\\ud800"},'
        '"result":"SURROGATEBODY\\udfff"}'
    )

    cards = render_completion_cards(project)
    card_text = "\n".join(_flatten_text(card["elements"]) for card in cards)

    assert "SURROGATECONCLUSION" in card_text
    assert "SURROGATEBODY" in card_text
    assert "\ud800" not in card_text
    assert "\udfff" not in card_text
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)


def test_readable_projection_preserves_zero_and_string_scalars() -> None:
    project = _make_project("Scalar Result", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = json.dumps(
        {
            "count": 0,
            "ratio": 0.0,
            "flag": False,
            "literal_null": "null",
            "literal_zero": "0",
            "items": [0, 1],
        }
    )

    cards = render_completion_cards(project)
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

    assert "**count**\n0" in ledger_text
    assert "**ratio**\n0.0" in ledger_text
    assert "**flag**\n否" in ledger_text
    assert "**literal null**\nnull" in ledger_text
    assert "**literal zero**\n0" in ledger_text
    assert "- 0" in ledger_text
    assert "- 1" in ledger_text


def test_card_summary_extension_fields_remain_readable() -> None:
    project = _make_project("Legacy Summary", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = json.dumps(
        {
            "card_summary": {
                "status": "success",
                "message": "LEGACYMESSAGE",
                "details": "LEGACYDETAIL",
            },
            "result": "LEGACYBODY",
        }
    )

    cards = render_completion_cards(project)
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

    assert "摘要补充" in ledger_text
    assert "LEGACYMESSAGE" in ledger_text
    assert "LEGACYDETAIL" in ledger_text
    assert "LEGACYBODY" in ledger_text
    assert '"message"' not in ledger_text


def test_long_workflow_markdown_result_keeps_fences_balanced_across_pages() -> None:
    """Semantic pagination keeps every marker and repairs split code fences."""
    markers = [f"FENCED_RESULT_{index:03d}" for index in range(48)]
    result_body = (
        "```text\n"
        + "\n".join(f"{marker} " + ("x" * 900) for marker in markers)
        + "\n```\n\nFINAL_FENCE_MARKER"
    )
    project = _make_project("Fenced Final", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = json.dumps(
        {
            "card_summary": {
                "verdict": "passed",
                "conclusion": "长正文已完成。",
            },
            "result": result_body,
        },
        ensure_ascii=False,
    )

    cards = render_completion_cards(project)
    ledger_cards = [
        card for card in cards if card.get("_workflow_page_key", (None,))[0] == "ledger"
    ]
    ledger_texts = [_flatten_text(card["elements"]) for card in ledger_cards]
    ledger_text = "\n".join(ledger_texts)

    assert len(ledger_cards) >= 2
    assert all(ledger_text.count(marker) == 1 for marker in markers)
    assert ledger_text.count("FINAL_FENCE_MARKER") == 1
    assert all(text.count("```") % 2 == 0 for text in ledger_texts)
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)


def test_empty_workflow_results_use_one_clear_empty_state() -> None:
    """Empty JSON shapes never leak syntax or imply that content exists."""
    for raw_result in (None, "", "null", "[]", "{}"):
        project = _make_project("Empty Final", [])
        project.status = WorkflowStatus.COMPLETED
        project.result = raw_result

        cards = render_completion_cards(project)
        ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

        assert ledger_text.count("本次 Workflow 未返回可展示的最终内容") == 1
        assert "[]" not in ledger_text
        assert "{}" not in ledger_text
        assert "null" not in ledger_text


def test_failed_workflow_without_result_never_claims_completion() -> None:
    project = _make_project("Failed Final", [])
    project.status = WorkflowStatus.FAILED
    project.error = "FAILUREDETAILMARKER"
    project.result = None

    cards = render_completion_cards(project)
    status_text = _flatten_text(cards[0]["elements"])
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

    assert cards[0]["header"]["template"] == "red"
    assert "任务未完成" in status_text
    assert "任务已完成" not in status_text
    assert "FAILUREDETAILMARKER" in status_text
    assert "状态: 失败" in ledger_text
    assert "FAILUREDETAILMARKER" in ledger_text


def test_terminal_status_overrides_contradictory_result_verdicts() -> None:
    cases = (
        (
            WorkflowStatus.COMPLETED,
            "failed",
            "验证明确失败。",
            "red",
            "完成但验证失败",
            "验证失败",
        ),
        (
            WorkflowStatus.FAILED,
            "passed",
            "ALL GOOD",
            "red",
            "失败",
            "任务未完成",
        ),
        (
            WorkflowStatus.CANCELLED,
            "passed",
            "ALL GOOD",
            "grey",
            "已取消",
            "任务已取消",
        ),
    )
    for status, verdict, conclusion, template, title, expected in cases:
        project = _make_project("Contradictory Result", [])
        project.status = status
        project.result = json.dumps(
            {
                "card_summary": {
                    "verdict": verdict,
                    "conclusion": conclusion,
                },
                "result": "TERMINALSTATUSBODY",
            },
            ensure_ascii=False,
        )

        cards = render_completion_cards(project)
        status_text = _flatten_text(cards[0]["elements"])
        ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])

        assert cards[0]["header"]["template"] == template
        assert title in cards[0]["header"]["title"]["content"]
        assert expected in f"{cards[0]['header']['title']['content']}\n{status_text}"
        assert expected in ledger_text
        if status in {WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
            assert "ALL GOOD" not in status_text
            assert "ALL GOOD" not in ledger_text


def test_terminal_errors_are_redacted_and_cannot_inject_feishu_tags() -> None:
    secret = "sk-1234567890abcdefghijklmnop"
    injected = json.dumps(
        {
            "message": f"boom {secret} <at id=all></at>",
            "code": 500,
        }
    )
    agent = _make_agent(
        "failed-agent",
        AgentStatus.FAILED,
        error=injected,
    )
    project = _make_project("Safe Errors", [agent])
    project.status = WorkflowStatus.FAILED
    project.error = injected
    project.reviewer_evidence = [
        ReviewerEvidence(
            reviewer_index=1,
            display_name="reviewer",
            tool="codex",
            status="failed",
            error=injected,
        )
    ]

    cards = render_completion_cards(project)
    card_text = "\n".join(_flatten_text(card["elements"]) for card in cards)

    assert secret not in card_text
    assert "<at" not in card_text
    assert "id=all" in card_text
    assert "redacted" in card_text
    assert '"message"' not in card_text
    assert '"code"' not in card_text


def test_deeply_nested_result_degrades_safely_instead_of_raising() -> None:
    raw = '{"card_summary":{"verdict":"passed","conclusion":"DEEP"},"result":'
    raw += "[" * 1_100 + '"DEEPEST"' + "]" * 1_100 + "}"
    project = _make_project("Deep Result", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = raw

    cards = render_completion_cards(project)
    card_text = "\n".join(_flatten_text(card["elements"]) for card in cards)

    assert cards
    assert "Workflow 最终结果" in card_text
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)


def test_result_ledger_limits_collapsible_panels_per_card() -> None:
    agents = [
        _make_agent(
            f"panel-agent-{index}",
            AgentStatus.DONE,
            result=json.dumps(
                {
                    "card_summary": {
                        "verdict": "passed",
                        "conclusion": f"PANELCONCLUSION{index:02d}",
                    },
                    "result": f"PANELBODY{index:02d} " + ("x" * 1_400),
                }
            ),
        )
        for index in range(12)
    ]
    project = _make_project("Panel Limit", agents)
    project.status = WorkflowStatus.COMPLETED

    cards = render_completion_cards(project)
    ledger_cards = [
        card for card in cards if card.get("_workflow_page_key", (None,))[0] == "ledger"
    ]
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in ledger_cards)

    assert len(ledger_cards) >= 3
    assert all(str(card).count("'tag': 'collapsible_panel'") <= 5 for card in ledger_cards)
    assert all(ledger_text.count(f"PANELBODY{index:02d}") == 1 for index in range(12))


def test_current_step_uses_most_recent_running_activity() -> None:
    older = _make_agent(
        "older-running",
        AgentStatus.RUNNING,
        current_activity="OLDERACTIVITY",
        activity_updated_at=100.0,
    )
    newer = _make_agent(
        "newer-running",
        AgentStatus.RUNNING,
        current_activity="NEWESTACTIVITY",
        activity_updated_at=200.0,
    )
    project = _make_project("Concurrent", [older, newer])
    project.status = WorkflowStatus.RUNNING

    status_card = WorkflowProgressRenderer(project).render_progress_cards(project)[0]
    summaries = _markdown_blocks_containing(status_card, "当前执行中")

    assert len(summaries) == 1
    assert "NEWESTACTIVITY" in summaries[0]
    assert "OLDERACTIVITY" not in summaries[0]


def test_terminal_status_page_never_labels_stale_activity_as_running() -> None:
    agent = _make_agent(
        "finished-agent",
        AgentStatus.DONE,
        result="FINISHEDRESULT",
        current_activity="STALEACTIVITY",
        activity_updated_at=200.0,
    )
    project = _make_project("Finished", [agent])
    project.status = WorkflowStatus.COMPLETED

    status_card = WorkflowProgressRenderer(project).render_progress_cards(project)[0]
    status_text = _flatten_text(status_card["elements"])

    assert "⚡ **正在:**" not in status_text
    assert "STALEACTIVITY" not in status_text


def test_small_phase_renders_everything() -> None:
    """Small phase (5 agents, ≤ 8) — render everything unchanged."""
    agents: list[AgentProgress] = [
        _make_agent("a-0", AgentStatus.DONE),
        _make_agent("a-1", AgentStatus.DONE),
        _make_agent("a-2", AgentStatus.RUNNING),
        _make_agent("a-3", AgentStatus.PENDING),
        _make_agent("a-4", AgentStatus.FAILED, error="fail"),
    ]
    project = _make_project("Small Phase", agents)
    renderer = WorkflowProgressRenderer(project)
    card = renderer.render_progress_card()
    text = _flatten_text(card["elements"])

    assert "已完成 2/5" in text

    # Small phases should not have "条已完成/缓存（已折叠）" counter line
    assert "条已完成/缓存（已折叠）" not in text


def test_empty_phase_renders_summary_zero_over_zero() -> None:
    """Started empty phase — renders an explicit 0/0 in-progress state."""
    project = _make_project("Empty Phase", [])
    renderer = WorkflowProgressRenderer(project)
    card = renderer.render_progress_card()
    text = _flatten_text(card["elements"])

    assert "进行中 0/0" in text


# ---------------------------------------------------------------------------
# Phase collapsible-panel headers: structured with per-status color
# ---------------------------------------------------------------------------


def test_phase_collapsible_panel_headers_are_structured_with_border_colors() -> None:
    """Feishu accepts colors on panel border, not collapsible header.template."""
    agents: list[AgentProgress] = [
        _make_agent("r-0", AgentStatus.RUNNING),
        _make_agent("f-0", AgentStatus.FAILED, error="err"),
        _make_agent("d-0", AgentStatus.DONE),
        _make_agent("c-0", AgentStatus.CACHED),
        _make_agent("p-0", AgentStatus.PENDING),
    ]
    project = _make_project("Mixed Phase", agents)
    renderer = WorkflowProgressRenderer(project)
    card = renderer.render_progress_card()

    panels = [el for el in card["elements"] if isinstance(el, dict) and el.get("tag") == "collapsible_panel"]
    assert len(panels) >= 5, f"Expected 5 collapsible panels, got {len(panels)}"

    expected_colors = {
        "执行中": "blue",
        "失败": "red",
        "已完成": "green",
        "缓存": "turquoise",
        "待执行": "grey",
    }
    found_labels: set[str] = set()
    for panel in panels:
        header = panel.get("header")
        assert isinstance(header, dict), f"Expected dict header, got {type(header)}: {header}"
        title = header.get("title")
        assert isinstance(title, dict), f"Expected dict title, got {title}"
        assert title.get("tag") == "plain_text"
        assert header.get("icon") == {
            "tag": "standard_icon",
            "token": "down_outlined",
            "color": "grey",
        }
        assert header.get("icon_position") == "right"
        assert header.get("icon_expanded_angle") == -180
        content = str(title.get("content", ""))
        assert "template" not in header, f"collapsible_panel.header.template is invalid: {header}"
        border = panel.get("border")
        assert isinstance(border, dict), f"Expected dict border, got {border}"
        # Match the prefix label to verify the color mapping
        for prefix, color in expected_colors.items():
            if content.startswith(prefix):
                found_labels.add(prefix)
                assert border.get("color") == color, (
                    f"Expected border color '{color}' for {prefix!r}, got {border.get('color')}"
                )
    for expected_prefix in expected_colors:
        assert expected_prefix in found_labels, f"Missing panel for {expected_prefix!r}"
