"""Tests for phase summary, collapsible panel headers, and completion card layout."""

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
    render_completion_card,
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
    project.status = WorkflowStatus.RUNNING

    cards = WorkflowProgressRenderer(project).render_progress_cards(project)

    assert len(cards) >= 2
    assert all(isinstance(card, dict) and "header" in card and "elements" in card for card in cards)
    status_text = _flatten_text(cards[0]["elements"])
    ledger_texts = [_flatten_text(card["elements"]) for card in cards[1:]]
    ledger_text = "\n".join(ledger_texts)

    assert "进度 " in status_text
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
    project.status = WorkflowStatus.RUNNING

    cards = WorkflowProgressRenderer(project).render_progress_cards(project)

    assert len(cards[1:]) >= 2, "a single oversized result must span multiple ledger pages"
    assert all(_card_size_bytes(card) <= 28_000 for card in cards)
    ledger_text = "\n".join(_flatten_text(card["elements"]) for card in cards[1:])
    markers = [f"RESULTCHUNK{i:03d}" for i in range(72)]
    assert all(ledger_text.count(marker) == 1 for marker in markers)
    assert [ledger_text.index(marker) for marker in markers] == sorted(
        ledger_text.index(marker) for marker in markers
    )


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


def test_singular_progress_renderer_api_remains_available() -> None:
    project = _make_project("Compatible", [_make_agent("running", AgentStatus.RUNNING)])
    card = WorkflowProgressRenderer(project).render_progress_card(project)

    assert isinstance(card, dict)
    assert isinstance(card.get("header"), dict)
    assert isinstance(card.get("elements"), list)


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
# Completion card: stats column_set should be stretch + centered text
# ---------------------------------------------------------------------------


def _make_completion_project(status) -> WorkflowProject:
    return WorkflowProject(
        name="audit",
        status=status,
        started_at=1_700_000_000.0,
        finished_at=1_700_000_060.0,
        phases=[
            PhaseProgress(
                title="Analyze",
                agents=[
                    AgentProgress(label="scan", tool="coco", status=AgentStatus.DONE, duration_s=5.0),
                    AgentProgress(label="verify", tool="claude", status=AgentStatus.DONE, duration_s=5.0),
                ],
            )
        ],
    )


def test_completion_card_stats_use_stretch_flex_mode() -> None:
    """Stats column_sets in render_completion_card should use flex_mode='stretch'."""
    project = _make_completion_project(WorkflowStatus.COMPLETED)
    card = render_completion_card(project)
    stats_column_sets = [el for el in card["elements"] if isinstance(el, dict) and el.get("tag") == "column_set"]
    assert stats_column_sets, "Expected at least one column_set in completion card"
    for cs in stats_column_sets:
        assert cs.get("flex_mode") == "stretch", f"Expected flex_mode='stretch', got {cs.get('flex_mode')!r}"


def test_completion_card_stats_columns_centered_text() -> None:
    """Stat column markdown elements should set text_align='center'."""
    project = _make_completion_project(WorkflowStatus.COMPLETED)
    card = render_completion_card(project)
    stats_column_sets = [el for el in card["elements"] if isinstance(el, dict) and el.get("tag") == "column_set"]
    assert stats_column_sets, "Expected stats column_sets"
    for cs in stats_column_sets:
        for col in cs.get("columns", []):
            for inner in col.get("elements", []):
                if inner.get("tag") == "markdown":
                    assert inner.get("text_align") == "center", (
                        f"Expected text_align='center' on markdown element, got {inner}"
                    )


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
