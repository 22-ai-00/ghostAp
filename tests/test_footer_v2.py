import pytest

from src.card.render.footer import (
    build_footer_atoms,
    render_footer,
    render_now_tool_hint,
    render_subagent_badge,
)
from src.card.state.models import (
    CardMetadata,
    CardState,
    ContentBlock,
    EngineExtState,
    FooterState,
)
from src.utils.text import format_elapsed_clock


def _tool(name, status="active", tool_input=None):
    return ContentBlock(
        kind="tool_call",
        block_id=name,
        tool_name=name,
        status=status,
        tool_input=tool_input or "{}",
        tool_output="",
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00"),
        (59, "00:00:59"),
        (3661, "01:01:01"),
        (90061, "1天 01:01:01"),
    ],
)
def test_elapsed_clock_uses_zero_padded_hms_and_days(seconds, expected):
    assert format_elapsed_clock(seconds) == expected


def test_running_main_footer_is_always_visible_and_advances_on_each_render(
    monkeypatch,
):
    now = [220.0]
    monkeypatch.setattr(
        "src.card.render.footer.time.monotonic",
        lambda: now[0],
    )
    state = CardState(
        metadata=CardMetadata(session_started_at=100.0),
        footer=FooterState(),
    )

    first_content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
    )
    now[0] = 225.0
    second_content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
    )

    assert "⏱ 用时 00:02:00" in first_content
    assert "⏱ 用时 00:02:05" in second_content


def test_terminal_footer_freezes_authoritative_duration_in_clock_format(
    monkeypatch,
):
    now = [10_000.0]
    monkeypatch.setattr(
        "src.card.render.footer.time.monotonic",
        lambda: now[0],
    )
    state = CardState(
        terminal="completed",
        metadata=CardMetadata(session_started_at=100.0),
        footer=FooterState(duration_seconds=58),
    )

    first_content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
    )
    now[0] = 20_000.0
    second_content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
    )

    assert "⏱ 用时 00:00:58" in first_content
    assert second_content == first_content


def test_running_footer_uses_days_as_largest_unit(monkeypatch):
    monkeypatch.setattr(
        "src.card.render.footer.time.monotonic",
        lambda: 90_161.0,
    )
    state = CardState(
        metadata=CardMetadata(session_started_at=100.0),
        footer=FooterState(status="thinking", status_text="执行中"),
    )

    content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
    )

    assert "⏱ 用时 1天 01:01:01" in content


# --- render_now_tool_hint unit tests ---


def test_now_tool_hint_for_edit_path():
    line = render_now_tool_hint(_tool("Edit", tool_input='{"path": "src/router.py"}'))

    assert line == "⚙ **Edit** · 写入 src/router.py"


def test_now_tool_hint_for_grep_pattern():
    line = render_now_tool_hint(_tool("Grep", tool_input='{"pattern": "def route"}'))

    assert "⚙ **Grep**" in line
    assert "搜索" in line
    assert "def route" in line


def test_now_tool_hint_unknown_tool_falls_back_to_name():
    line = render_now_tool_hint(_tool("MysteryTool"))

    assert line == "⚙ **MysteryTool**"


def test_now_tool_hint_none_when_no_running_tool():
    assert render_now_tool_hint(_tool("Edit", status="completed")) == ""
    assert render_now_tool_hint(_tool("Edit", status="running")) == ""
    assert render_now_tool_hint(_tool("Edit", status="in_progress")) == ""
    assert render_now_tool_hint(None) == ""


# --- Footer renders ⚙ tool hint for active tools (v2) ---


def test_engine_footer_renders_now_tool_hint_for_running_tool():
    """Engine footer keeps its tool hint because it has no unified execution flow."""
    state = CardState(
        blocks=(_tool("Bash", tool_input='{"command": "uv run pytest"}'),),
        footer=FooterState(status="tool_running", status_text="执行中"),
        metadata=CardMetadata(engine_type="deep"),
    )

    elements = render_footer(state)

    assert any("⚙ **Bash**" in el.get("content", "") for el in elements)


def test_programming_footer_omits_current_tool_owned_by_execution_flow():
    command = "uv run python -m pytest tests/test_footer_v2.py -q"
    state = CardState(
        blocks=(_tool(command, tool_input=command),),
        footer=FooterState(
            status="tool_running",
            status_text=f"🔧 执行中: {command}",
        ),
        metadata=CardMetadata(
            tool_name="codex",
            model_name="gpt-5",
            engine_type=None,
        ),
    )

    content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
        if element.get("tag") == "markdown"
    )

    assert "⚙" not in content
    assert "执行中" not in content
    assert command not in content
    assert "🔧 codex · 🧩 gpt-5" in content


def test_terminal_footer_never_shows_stale_active_tool():
    state = CardState(
        blocks=(_tool("Context compacting", tool_input="Context compacting"),),
        terminal="failed",
        footer=FooterState(
            status="tool_running",
            status_text="🔧 执行中: Context compacting",
            duration_seconds=120,
        ),
        metadata=CardMetadata(tool_name="codex", engine_type=None),
    )

    content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
        if element.get("tag") == "markdown"
    )

    assert "Context compacting" not in content
    assert "执行中" not in content
    assert "⏱ 用时 00:02:00" in content


def test_frozen_footer_continuation():
    """Frozen card with continuation_seq shows UX-aligned continuation hint."""
    state = CardState(
        footer=FooterState(status="thinking", status_text="执行中"),
        metadata=CardMetadata(frozen=True, continuation_seq=2),
    )

    elements = render_footer(state)

    assert any(
        "本卡已停止更新" in el.get("content", "")
        and "续接" in el.get("content", "")
        and "#3" in el.get("content", "")
        and "↓" in el.get("content", "")
        for el in elements
    )


@pytest.mark.parametrize("seq,expected_next", [(1, "#2"), (2, "#3"), (5, "#6")])
def test_frozen_footer_continuation_boundary(seq, expected_next):
    """Frozen continuation hint produces correct next_seq for boundary values."""
    state = CardState(
        footer=FooterState(status="thinking", status_text="执行中"),
        metadata=CardMetadata(frozen=True, continuation_seq=seq),
    )

    elements = render_footer(state)

    assert any(expected_next in el.get("content", "") for el in elements)


def test_frozen_footer_no_continuation_when_seq_zero():
    """Frozen card with continuation_seq=0 should NOT show continuation hint."""
    state = CardState(
        footer=FooterState(status="thinking", status_text="执行中"),
        metadata=CardMetadata(frozen=True, continuation_seq=0),
    )

    elements = render_footer(state)

    assert not any("本卡已停止更新" in el.get("content", "") for el in elements)


def test_child_footer_keeps_one_status_and_metadata_line_without_duplicates(monkeypatch):
    monkeypatch.setattr("src.card.render.footer.time.monotonic", lambda: 146.0)
    state = CardState(
        blocks=(_tool("Bash", tool_input='{"command": "uv run pytest"}'),),
        footer=FooterState(
            status="tool_running",
            status_text="🔄 执行中",
            progress="工具调用 4/4",
            progress_started_at=100.0,
        ),
        metadata=CardMetadata(
            working_dir="/workspace/ghostAp",
            tool_name="codex",
            model_name="gpt-5",
            is_subagent=True,
            parent_card_seq="1",
        ),
    )

    elements = render_footer(state)
    content_lines = [
        element.get("content", "")
        for element in elements
        if element.get("tag") == "markdown"
    ]

    assert "🔄 执行中" in content_lines
    assert [
        line for line in content_lines if "codex" in line or "gpt-5" in line
    ] == ["🔧 codex · 🧩 gpt-5 · ⏱ 用时 00:00:46"]
    assert not any("📂" in line for line in content_lines)
    assert not any("⚙" in line for line in content_lines)
    assert not any("🧬" in line for line in content_lines)
    assert not any("sub" in line.lower() for line in content_lines)
    assert not any("model:" in line or "tool:" in line or "from #" in line for line in content_lines)


def test_running_child_footer_uses_session_start_when_progress_never_started(
    monkeypatch,
):
    monkeypatch.setattr("src.card.render.footer.time.monotonic", lambda: 146.0)
    state = CardState(
        footer=FooterState(
            status="thinking",
            status_text="🔄 执行中",
        ),
        metadata=CardMetadata(
            tool_name="codex",
            model_name="gpt-5",
            is_subagent=True,
            session_started_at=100.0,
        ),
    )

    content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
    )

    assert "🔧 codex · 🧩 gpt-5 · ⏱ 用时 00:00:46" in content


@pytest.mark.parametrize(
    "unsafe_tool",
    [
        "call_internal_123",
        '\\" not in ordinary_output\\",\\n',
        'Explore raw JSON: {"model":"x"}',
    ],
)
def test_child_footer_omits_unsafe_tool_metadata(unsafe_tool):
    state = CardState(
        footer=FooterState(status="thinking", status_text="执行中"),
        metadata=CardMetadata(
            tool_name=unsafe_tool,
            model_name="gpt-5",
            is_subagent=True,
        ),
    )

    content = "\n".join(
        element.get("content", "")
        for element in render_footer(state)
    )

    assert unsafe_tool not in content
    assert "🔧" not in content
    assert "🧩 gpt-5" in content


@pytest.mark.parametrize(
    ("terminal", "expected_status"),
    [
        ("completed", "✅ 已完成"),
        ("failed", "❌ 执行失败"),
    ],
)
def test_child_terminal_footer_projects_task_status(
    terminal,
    expected_status,
):
    state = CardState(
        terminal=terminal,
        footer=FooterState(
            status_text="发送 /deep 启动新任务",
            duration_seconds=18,
        ),
        metadata=CardMetadata(
            tool_name="codex",
            model_name="gpt-5",
            is_subagent=True,
        ),
    )

    content_lines = [
        element.get("content", "")
        for element in render_footer(state)
        if element.get("tag") == "markdown"
    ]

    assert content_lines[0] == expected_status
    assert not any("发送 /deep" in line for line in content_lines)


@pytest.mark.parametrize(
    "state",
    [
        CardState(
            terminal="blocked",
            footer=FooterState(duration_seconds=18),
            metadata=CardMetadata(
                tool_name="codex",
                model_name="gpt-5",
                is_subagent=True,
            ),
            engine_ext=EngineExtState(blocked_reason="等待依赖"),
        ),
        CardState(
            terminal="archived",
            footer=FooterState(duration_seconds=18),
            metadata=CardMetadata(
                tool_name="codex",
                model_name="gpt-5",
                is_subagent=True,
                frozen=True,
                continuation_seq=2,
            ),
        ),
        CardState(
            footer=FooterState(
                status="thinking",
                status_text="执行中",
            ),
            metadata=CardMetadata(
                tool_name="codex",
                model_name="gpt-5",
                is_subagent=True,
                idle_timeout_seconds=60,
            ),
        ),
    ],
    ids=["blocked", "archived", "idle-timeout"],
)
def test_child_footer_has_at_most_two_semantic_lines(state):
    content_lines = [
        element.get("content", "")
        for element in render_footer(state)
        if element.get("tag") == "markdown"
    ]

    assert len(content_lines) <= 2


def test_subagent_badge_helper_is_blank_for_main_card():
    assert render_subagent_badge(CardMetadata(tool_name="Codex")) == ""


def test_build_footer_atoms_omit_child_tool_hint_and_subagent_badge():
    """Child-only footer appendices must not reintroduce duplicate metadata."""
    state = CardState(
        blocks=(_tool("Edit", tool_input='{"path": "src/card/render/footer.py"}'),),
        footer=FooterState(status="tool_running", status_text="执行中"),
        metadata=CardMetadata(
            tool_name="Aiden",
            model_name="claude-haiku-4-5",
            is_subagent=True,
            parent_card_seq="5",
        ),
    )

    atoms = build_footer_atoms(state)

    assert not any("⚙ **Edit**" in atom.content for atom in atoms)
    assert not any("🧬 sub" in atom.content for atom in atoms)


def test_build_footer_atoms_omit_child_frozen_continuation():
    state = CardState(
        metadata=CardMetadata(
            is_subagent=True,
            frozen=True,
            continuation_seq=2,
        ),
    )

    atoms = build_footer_atoms(state)

    assert not any("本卡已停止更新" in atom.content for atom in atoms)


def test_build_footer_atoms_frozen_continuation():
    """build_footer_atoms should include frozen continuation hint (AC-11, AC-23)."""
    state = CardState(
        footer=FooterState(status="thinking", status_text="执行中"),
        metadata=CardMetadata(frozen=True, continuation_seq=2),
    )

    atoms = build_footer_atoms(state)

    assert any("本卡已停止更新" in atom.content and "续接" in atom.content and "#3" in atom.content for atom in atoms)


def test_build_footer_atoms_no_tool_hint_when_no_active_tool():
    """build_footer_atoms should not include tool hint when no active tool."""
    state = CardState(
        blocks=(_tool("Edit", status="completed", tool_input='{"path": "a.py"}'),),
        footer=FooterState(status="thinking", status_text="思考中"),
    )

    atoms = build_footer_atoms(state)

    assert not any("⚙" in atom.content for atom in atoms)


def test_footer_no_tool_hint_when_all_tools_completed():
    """Footer should NOT show ⚙ tool hint when all tools are completed."""
    state = CardState(
        blocks=(_tool("Edit", status="completed", tool_input='{"path": "a.py"}'),),
        footer=FooterState(status="thinking", status_text="思考中"),
    )

    elements = render_footer(state)

    assert not any("⚙ **Edit**" in el.get("content", "") for el in elements)


def test_footer_tool_hint_bash_shows_command():
    """Footer ⚙ hint for Bash shows the command."""
    state = CardState(
        blocks=(_tool("Bash", tool_input='{"command": "npm run build"}'),),
        footer=FooterState(status="tool_running", status_text="执行中"),
        metadata=CardMetadata(engine_type="deep"),
    )

    elements = render_footer(state)

    hints = [el.get("content", "") for el in elements if "⚙ **Bash**" in el.get("content", "")]
    assert len(hints) == 1
    assert "npm run build" in hints[0]


# --- Context line tests (working_dir + engine phase in footer) ---


def test_footer_context_line_with_working_dir():
    """Footer should show 📂 ~/path when working_dir is set."""
    state = CardState(
        footer=FooterState(status="thinking", status_text="思考中"),
        metadata=CardMetadata(working_dir="/home/user/projects/myapp", engine_type="spec"),
    )

    elements = render_footer(state)

    context_els = [el for el in elements if "📂" in el.get("content", "")]
    assert len(context_els) == 1
    assert "myapp" in context_els[0]["content"]


def test_footer_context_line_absent_when_no_working_dir():
    """Footer should NOT show 📂 context line when working_dir is None."""
    state = CardState(
        footer=FooterState(status="thinking", status_text="思考中"),
        metadata=CardMetadata(engine_type="spec"),
    )

    elements = render_footer(state)

    assert not any("📂" in el.get("content", "") for el in elements)


def test_footer_context_line_includes_engine_subtitle():
    """Footer context line includes engine phase from header subtitle."""
    from src.card.state.models import HeaderState

    state = CardState(
        header=HeaderState(subtitle="cycle 2 / Build"),
        footer=FooterState(status="thinking", status_text="思考中"),
        metadata=CardMetadata(
            working_dir="/home/user/work/ghostAp",
            engine_type="spec",
        ),
    )

    elements = render_footer(state)

    context_els = [el for el in elements if "📂" in el.get("content", "")]
    assert len(context_els) == 1
    assert "cycle 2 / Build" in context_els[0]["content"]


def test_spec_footer_uses_session_start_for_total_elapsed(monkeypatch):
    """Spec footer total runtime should not start at the first progress update."""
    monkeypatch.setattr("src.card.render.footer.time.monotonic", lambda: 220.0)
    state = CardState(
        metadata=CardMetadata(engine_type="spec", session_started_at=100.0),
        footer=FooterState(
            status="thinking",
            status_text="思考中",
            progress="1/3 通过",
            progress_started_at=190.0,
        ),
    )

    elements = render_footer(state)
    content = "\n".join(e.get("content", "") for e in elements)

    assert "⏱ 用时 00:02:00" in content
    assert "00:00:30" not in content


def test_spec_footer_renders_elapsed_even_without_status(monkeypatch):
    """Spec total runtime should make the footer visible by itself."""
    monkeypatch.setattr("src.card.render.footer.time.monotonic", lambda: 130.0)
    state = CardState(
        metadata=CardMetadata(engine_type="spec", session_started_at=100.0),
        footer=FooterState(),
    )

    elements = render_footer(state)
    content = "\n".join(e.get("content", "") for e in elements)

    assert elements[0] == {"tag": "hr"}
    assert "⏱ 用时 00:00:30" in content


def test_terminal_footer_places_duration_after_tool_model():
    state = CardState(
        terminal="completed",
        metadata=CardMetadata(tool_name="coco", model_name="Test-O-New-Thinking"),
        footer=FooterState(duration_seconds=58),
    )

    elements = render_footer(state)
    content_lines = [e.get("content", "") for e in elements if e.get("tag") == "markdown"]

    assert "🔧 coco · 🧩 Test-O-New-Thinking · ⏱ 用时 00:00:58" in content_lines
