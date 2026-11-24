"""Tests for phase summaries, result delivery, and current operation display."""

from __future__ import annotations

import json

from src.workflow_engine.models import (
    AgentProgress,
    AgentStatus,
    PhaseProgress,
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
    return AgentProgress(
        label=label,
        tool=tool,
        status=status,
        duration_s=duration_s,
        error=error,
        result=result,
        current_activity=current_activity,
        activity_updated_at=activity_updated_at,
    )


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

    assert "DOUBLECONCLUSIONMARKER" in status_text
    assert "DOUBLEVERIFYMARKER" in status_text
    assert "待确认" not in status_text
    assert '"card_summary"' not in status_text
    assert "\\n" not in status_text


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
    card_text = _flatten_text(cards[0]["elements"])

    assert "FENCEDCONCLUSION" in card_text
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
        card_text = _flatten_text(cards[0]["elements"])

        assert "WRAPPEDCONCLUSION" in card_text
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
    card_text = _flatten_text(cards[0]["elements"])

    assert "SURROGATECONCLUSION" in card_text
    assert "\ud800" not in card_text
    assert "\udfff" not in card_text
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)


def test_failed_workflow_without_result_never_claims_completion() -> None:
    project = _make_project("Failed Final", [])
    project.status = WorkflowStatus.FAILED
    project.error = "FAILUREDETAILMARKER"
    project.result = None

    cards = render_completion_cards(project)
    status_text = _flatten_text(cards[0]["elements"])

    assert cards[0]["header"]["template"] == "red"
    assert "任务未完成" in status_text
    assert "任务已完成" not in status_text
    assert "FAILUREDETAILMARKER" in status_text


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

        assert cards[0]["header"]["template"] == template
        assert title in cards[0]["header"]["title"]["content"]
        assert expected in f"{cards[0]['header']['title']['content']}\n{status_text}"
        if status in {WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
            assert "ALL GOOD" not in status_text


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

    cards = render_completion_cards(project)
    card_text = "\n".join(_flatten_text(card["elements"]) for card in cards)

    assert secret not in card_text
    assert "<at" not in card_text
    assert "id=all" in card_text
    assert "redacted" in card_text


def test_deeply_nested_result_degrades_safely_instead_of_raising() -> None:
    raw = '{"card_summary":{"verdict":"passed","conclusion":"DEEP"},"result":'
    raw += "[" * 1_100 + '"DEEPEST"' + "]" * 1_100 + "}"
    project = _make_project("Deep Result", [])
    project.status = WorkflowStatus.COMPLETED
    project.result = raw

    cards = render_completion_cards(project)

    assert cards
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)


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


def test_failed_brief_preserves_actionable_root_cause_and_rejects_passed_verdict() -> None:
    from src.workflow_engine.renderer_completion import _brief_for_workflow_status
    from src.workflow_engine.result_brief import (
        _FAILED_FALLBACK_CONCLUSION,
        _NEUTRAL_CONCLUSION,
        BriefVerdict,
        WorkflowResultBrief,
        build_result_brief,
        fit_result_brief,
    )

    # 1. Preserves actionable failed brief conclusion
    brief = WorkflowResultBrief(
        verdict=BriefVerdict.FAILED,
        conclusion="独立安全评审发现 OAuth state 仍可重复消费。",
    )
    updated = _brief_for_workflow_status(brief, WorkflowStatus.FAILED)
    assert updated.verdict == BriefVerdict.FAILED
    assert "OAuth state 仍可重复消费" in updated.conclusion
    assert "任务未完成" in updated.conclusion

    # 2. Rejects contradictory PASSED verdict and falls back to sanitized agent error
    contradictory = WorkflowResultBrief(
        verdict=BriefVerdict.PASSED,
        conclusion="ALL GOOD",
    )
    secret = "sk-1234567890abcdefghijklmnop"
    overridden = _brief_for_workflow_status(
        contradictory,
        WorkflowStatus.FAILED,
        failed_agent_errors=(f"codex 503 {secret} <at id=all></at>",),
    )
    assert "ALL GOOD" not in overridden.conclusion
    assert "codex 503" in overridden.conclusion
    assert secret not in overridden.conclusion
    assert "<at" not in overridden.conclusion

    # 3. fit_result_brief never reverts FAILED verdict to _NEUTRAL_CONCLUSION ("任务已完成...")
    oversized = WorkflowResultBrief(
        verdict=BriefVerdict.FAILED,
        conclusion="x" * 1500,
    )
    fitted = fit_result_brief(oversized, max_item_bytes=900)
    assert fitted.conclusion == _FAILED_FALLBACK_CONCLUSION
    assert fitted.conclusion != _NEUTRAL_CONCLUSION

    # 4. build_result_brief with {"error": "..."} uses error instead of _NEUTRAL_CONCLUSION
    err_brief = build_result_brief(json.dumps({"error": "All pool fallbacks exhausted: codex 503"}))
    assert err_brief.verdict == BriefVerdict.FAILED
    assert "codex 503" in err_brief.conclusion

    # 5. card_summary with verdict="failed" and no conclusion surfaces failed_agent_errors instead of generic fallback
    no_conclusion_failed = build_result_brief(
        json.dumps({"card_summary": {"verdict": "failed"}})
    )
    assert no_conclusion_failed.conclusion == _FAILED_FALLBACK_CONCLUSION
    surfaced = _brief_for_workflow_status(
        no_conclusion_failed,
        WorkflowStatus.FAILED,
        failed_agent_errors=("verifier timed out after 120s",),
    )
    assert "verifier timed out after 120s" in surfaced.conclusion
    assert surfaced.conclusion != _FAILED_FALLBACK_CONCLUSION
