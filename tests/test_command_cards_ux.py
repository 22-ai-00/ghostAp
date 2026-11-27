"""Comprehensive unit & integration tests for M1 command interactive card UX optimization (F1–F7)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.card.actions import dispatch as action_ids
from src.card.builders.core import CoreBuilder
from src.card.builders.project import ProjectBuilder
from src.card.builders.system import SystemBuilder
from src.card.shared.truncation import count_tagged_nodes
from src.card.thresholds import THRESHOLDS
from src.card.ui_text import UI_TEXT
from src.feishu.action_registry import init_action_registry
from src.feishu.handlers.project import ProjectHandler
from src.feishu.handlers.system import SystemHandler
from src.mode import InteractionMode
from src.project.context import ProjectContext, ProjectStatus


def _parse_and_validate_card(result: tuple[str, str], *, require_update_multi: bool = True) -> tuple[dict, str]:
    msg_type, raw = result
    assert msg_type == "interactive"
    assert len(raw.encode("utf-8")) <= THRESHOLDS["CARD_BYTE_BUDGET"]
    card = json.loads(raw)
    assert card["schema"] == "2.0"
    assert "elements" not in card
    assert isinstance(card["body"]["elements"], list)
    assert count_tagged_nodes(card) <= THRESHOLDS["CARD_NODE_BUDGET"]
    if require_update_multi:
        assert card["config"].get("update_multi") is True
    summary = card["config"].get("summary")
    if summary is not None:
        content = summary.get("content", "")
        assert 8 <= len(content) <= 60
    return card, raw


def _collect_buttons(node: object) -> list[dict]:
    buttons: list[dict] = []

    def _walk(cur: object) -> None:
        if isinstance(cur, dict):
            if cur.get("tag") == "button":
                buttons.append(cur)
                return
            for v in cur.values():
                _walk(v)
        elif isinstance(cur, list):
            for item in cur:
                _walk(item)

    _walk(node)
    return buttons


def _assert_callback_buttons_valid(card: dict) -> list[dict]:
    buttons = _collect_buttons(card)
    for btn in buttons:
        if btn.get("disabled"):
            assert "behaviors" not in btn, f"disabled button must not carry behaviors: {btn}"
            continue
        if "multi_url" in btn:
            continue
        behaviors = btn.get("behaviors")
        assert isinstance(behaviors, list) and behaviors, f"missing behaviors on {btn}"
        assert behaviors[0].get("type") == "callback"
        assert "value" in btn
        assert behaviors[0].get("value") == btn["value"]
    return buttons


def _collect_column_sets(card: dict) -> list[dict]:
    return [el for el in card["body"]["elements"] if isinstance(el, dict) and el.get("tag") == "column_set"]


def _make_project(
    project_id: str,
    name: str,
    root_path: str,
    *,
    last_active: float = 1700000000.0,
    bound_chat_id: str = "",
    bound_chat_name: str = "",
) -> ProjectContext:
    return ProjectContext(
        project_id=project_id,
        project_name=name,
        root_path=root_path,
        last_active=last_active,
        status=ProjectStatus.ACTIVE,
        bound_chat_id=bound_chat_id,
        bound_chat_name=bound_chat_name,
    )


# ---------------------------------------------------------------------------
# CoreBuilder helpers: _wrap_card & _compact_path
# ---------------------------------------------------------------------------


def test_core_wrap_card_summary_clamping_and_update_multi() -> None:
    short = CoreBuilder._wrap_card("标题", "blue", [], update_multi=True, summary="短")
    assert short["config"]["update_multi"] is True
    assert 8 <= len(short["config"]["summary"]["content"]) <= 60

    long_summary = "这是一个非常长非常长非常长非常长非常长非常长非常长非常长非常长非常长非常长的卡片摘要内容用于测试截断并且长度超过六十个字符的限制"
    clamped = CoreBuilder._wrap_card("标题", "blue", [], update_multi=False, summary=long_summary)
    assert "update_multi" not in clamped["config"]
    assert 8 <= len(clamped["config"]["summary"]["content"]) <= 60
    assert clamped["config"]["summary"]["content"].endswith("…")


def test_core_compact_path_cross_platform_and_home_error_guard(monkeypatch) -> None:
    fake_home = Path("/Users/alice") if Path.home().as_posix().startswith("/Users") else Path("/home/alice")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    short_home = (fake_home / "work" / "ghostAp").as_posix()
    assert CoreBuilder._compact_path(short_home, limit=44) == "~/work/ghostAp"

    deep_home = (fake_home / "work" / "very-long-org-name" / "services" / "api-gateway-service").as_posix()
    compacted = CoreBuilder._compact_path(deep_home, limit=34)
    assert compacted.startswith("~/…/")
    assert len(compacted) <= 34

    non_home = "/var/log/very/long/deployment/path/ghostap/worker-node-01"
    for exc in (RuntimeError("Could not determine home directory."), OSError("no home")):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls, _exc=exc: (_ for _ in ()).throw(_exc)))
        compacted_err = CoreBuilder._compact_path(non_home, limit=30)
        assert compacted_err.startswith("…")
        assert len(compacted_err) <= 30

    tiny = CoreBuilder._compact_path("/a/b/c/d/e/f", limit=1)
    assert len(tiny) <= 1
    assert tiny == "…"
    assert CoreBuilder._compact_path("/a/b/c/d/e/f", limit=0) == ""

    assert CoreBuilder._compact_path(MagicMock()) == "~"


# ---------------------------------------------------------------------------
# F1: /projects Status Board
# ---------------------------------------------------------------------------


def test_f1_status_board_empty_and_populated_cards(monkeypatch) -> None:
    fake_home = Path("/home/tester")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    empty_card, empty_raw = _parse_and_validate_card(ProjectBuilder.build_status_board_card([]))
    empty_btns = _assert_callback_buttons_valid(empty_card)
    assert [b["value"]["action"] for b in empty_btns] == ["new_project_prompt", "refresh_board"]
    assert empty_card["config"]["summary"]["content"] == "项目看板 · 暂无项目"
    assert any(len(cs["columns"]) == 2 for cs in _collect_column_sets(empty_card))

    cur = _make_project(
        "p1",
        "ghostAp",
        "/home/tester/workspace/very-long-organization/team/ghostAp",
        last_active=1700000100.0,
        bound_chat_id="oc_group_1",
    )
    cur.set_programming_mode("claude", True, chat_id="chat-1")
    cur.update_programming_snapshot("claude", query="q", query_count=4, session_id="s-claude", chat_id="chat-1")
    cur.set_acp_configuration("claude", "claude-sonnet-4-5", chat_id="chat-1")

    other = _make_project("p2", "api-svc", "/home/tester/code/api-svc", last_active=1700000050.0)
    other.set_acp_configuration("codex", "gpt-5.6-sol", chat_id="chat-1")

    card, raw = _parse_and_validate_card(
        ProjectBuilder.build_status_board_card([other, cur], current_project_id="p1", page=1, page_size=5, chat_id="chat-1")
    )
    buttons = _assert_callback_buttons_valid(card)

    # Header summary & preserved contract strings
    top_md = card["body"]["elements"][0]["content"]
    assert "共 **2** 个项目" in top_md
    assert "当前：**ghostAp**" in top_md
    assert "Claude(4)" in raw
    assert "claude-sonnet-4-5" in raw
    assert "Claude 模式中" not in raw
    assert "Codex/gpt-5.6-sol" in raw
    assert "~/…/team/ghostAp" in raw

    # Current project exposes 1-click cockpit buttons; non-current exposes switch_to + detail
    actions_by_pid: dict[str, list[str]] = {}
    for btn in buttons:
        val = btn.get("value") or {}
        pid = val.get("project_id")
        if pid:
            actions_by_pid.setdefault(pid, []).append(val["action"])

    assert actions_by_pid["p1"] == ["continue_dev", "open_tool_settings", "open_tool_usage", "show_detail"]
    assert actions_by_pid["p2"] == ["switch_to", "show_detail"]
    assert any(len(cs["columns"]) == 2 for cs in _collect_column_sets(card))


def test_f1_status_board_magicmock_bound_chat_id_guard() -> None:
    mock_proj = MagicMock(spec=ProjectContext)
    mock_proj.project_id = "p-mock"
    mock_proj.project_name = "MockProj"
    mock_proj.root_path = "/tmp/mock"
    mock_proj.last_active = 1000.0
    mock_proj.coco_mode = False
    mock_proj.claude_mode = False
    mock_proj.status = ProjectStatus.IDLE
    mock_proj.get_status_emoji.return_value = "📁"

    card, raw = _parse_and_validate_card(ProjectBuilder.build_status_board_card([mock_proj]))
    assert "MagicMock" not in raw
    assert not any("multi_url" in btn for btn in _collect_buttons(card))


# ---------------------------------------------------------------------------
# F2: Project Detail & switch_to bugfix
# ---------------------------------------------------------------------------


def test_f2_project_detail_fixes_switch_to_and_handles_active_state() -> None:
    non_active = _make_project("p2", "api-svc", "/tmp/api-svc")
    non_active.set_acp_configuration("codex", "gpt-5.6-sol", chat_id="chat-1")

    card_inactive, raw_inactive = _parse_and_validate_card(
        ProjectBuilder.build_project_status_report_card(
            non_active, "/tmp", chat_id="chat-1", is_current=False
        )
    )
    btns_inactive = _assert_callback_buttons_valid(card_inactive)
    actions_inactive = [b["value"]["action"] for b in btns_inactive if "value" in b]
    assert "switch_to" in actions_inactive
    assert "switch_project" not in actions_inactive
    assert "continue_dev" not in actions_inactive
    assert "open_tool_settings" in actions_inactive
    assert "open_tool_usage" in actions_inactive
    assert "show_tool_commands" in actions_inactive
    assert "show_board" in actions_inactive
    assert "默认工具" in raw_inactive
    assert "gpt-5.6-sol" in raw_inactive

    active = _make_project("p1", "ghostAp", "/tmp/ghostAp")
    card_active, raw_active = _parse_and_validate_card(
        ProjectBuilder.build_project_status_report_card(
            active, "/tmp", chat_id="chat-1", is_current=True
        )
    )
    btns_active = _assert_callback_buttons_valid(card_active)
    actions_active = [b["value"]["action"] for b in btns_active if "value" in b]
    assert "continue_dev" in actions_active
    assert "switch_to" not in actions_active
    assert "当前项目" in raw_active


def test_f2_project_handler_show_project_status_passes_is_current() -> None:
    p1 = _make_project("p1", "ghostAp", "/tmp/ghostAp")
    p2 = _make_project("p2", "other", "/tmp/other")

    ctx = MagicMock()
    ctx.settings = SimpleNamespace(default_working_dir="/tmp")
    ctx.project_manager.get_active_project.return_value = p1
    ctx.message_linker.query.return_value = None
    handler = ProjectHandler(ctx)
    handler.get_working_dir = MagicMock(return_value="/tmp")
    handler.reply_card = MagicMock(return_value="om_reply_1")
    handler.register_message_project = MagicMock()

    handler.show_project_status("om_1", "oc_1", p1)
    active_card_json = handler.reply_card.call_args_list[0][0][1]
    assert '"action": "continue_dev"' in active_card_json
    assert '"action": "switch_to"' not in active_card_json

    handler.show_project_status("om_2", "oc_1", p2)
    inactive_card_json = handler.reply_card.call_args_list[1][0][1]
    assert '"action": "switch_to"' in inactive_card_json
    assert '"action": "switch_project"' not in inactive_card_json


# ---------------------------------------------------------------------------
# F3: /new Project Created
# ---------------------------------------------------------------------------


def test_f3_project_created_card_has_2col_cockpit_and_quick_starts() -> None:
    proj = _make_project("p-new", "demo-app", "/tmp/demo-app")
    card, _ = _parse_and_validate_card(ProjectBuilder.build_project_created_card(proj))
    buttons = _assert_callback_buttons_valid(card)
    actions = [b["value"]["action"] for b in buttons]
    assert actions == [
        "open_tool_settings",
        "show_board",
        "enter_coco",
        "enter_claude",
        "enter_agy",
        "enter_codex",
    ]
    col_sets = _collect_column_sets(card)
    assert len(col_sets) == 3
    assert all(len(cs["columns"]) == 2 for cs in col_sets)


# ---------------------------------------------------------------------------
# F4: /menu Quick Command Menu
# ---------------------------------------------------------------------------


def test_f4_command_menu_card_and_handler(tmp_path: Path) -> None:
    # Without project
    card_empty, raw_empty = _parse_and_validate_card(SystemBuilder.build_command_menu_card(None))
    btns_empty = _assert_callback_buttons_valid(card_empty)
    assert "执行通道与完成度" not in raw_empty
    assert "developing" not in raw_empty
    assert len(btns_empty) == 8
    assert card_empty["body"]["elements"][-1].get("text_size") == "notation"

    # With active project & tool
    proj = _make_project("p1", "ghostAp", str(tmp_path / "ghostAp"))
    card, raw = _parse_and_validate_card(
        SystemBuilder.build_command_menu_card(
            proj,
            current_tool="agy",
            current_model="gemini-pro-orchestrator",
            active_mode_key="agy",
        )
    )
    btns = _assert_callback_buttons_valid(card)
    col_sets = _collect_column_sets(card)
    assert len(col_sets) == 4
    assert all(len(cs["columns"]) == 2 for cs in col_sets)
    assert "AGY" in raw
    assert "gemini-pro-orchestrator" in raw

    # Every button action in /menu must exist in action_registry
    mock_client = SimpleNamespace(
        _handler_ctx=SimpleNamespace(
            handlers={
                "coco": MagicMock(),
                "project": MagicMock(),
                "system": MagicMock(),
                "deep": MagicMock(),
                "spec": MagicMock(),
                "workflow": MagicMock(),
            }
        ),
        _project_manager=MagicMock(),
    )
    registry = init_action_registry(mock_client)
    for btn in btns:
        assert btn["value"]["action"] in registry

    # Handler invocation with and without active project
    ctx = MagicMock()
    ctx.settings = SimpleNamespace(default_working_dir=str(tmp_path))
    ctx.project_manager.get_active_project.return_value = proj
    ctx.mode_manager.get_mode.return_value = InteractionMode.SMART
    ctx.mode_manager.is_programming_mode.return_value = False
    ctx.thread_manager.get.return_value = None
    proj.set_acp_configuration("codex", "gpt-5.6-sol", chat_id="oc_1")

    sys_handler = SystemHandler(ctx)
    sys_handler.reply_card = MagicMock(return_value="om_menu")
    sys_handler.handle_menu_command("om_in", "oc_1")
    menu_raw = sys_handler.reply_card.call_args[0][1]
    assert "Codex" in menu_raw
    assert "gpt-5.6-sol" in menu_raw


# ---------------------------------------------------------------------------
# F5: /tools & /tools_status Cards
# ---------------------------------------------------------------------------


def test_f5_tools_list_and_status_cards() -> None:
    proj = _make_project("p1", "ghostAp", "/tmp/ghostAp")
    tools = [
        {"name": "coco", "emoji": "🤖", "description": "字节跳动 AI", "available": True, "last_used": "12 分钟前"},
        {"name": "claude", "emoji": "🔮", "description": "Anthropic AI", "available": True, "last_used": "1 小时前"},
        {"name": "aiden", "emoji": "🎯", "description": "Aiden CLI", "available": False, "last_used": "从未使用"},
        {"name": "agy", "emoji": "🪐", "description": "Google Antigravity", "available": True, "last_used": "刚刚"},
    ]

    list_card, list_raw = _parse_and_validate_card(
        SystemBuilder.build_tools_list_card(tools, proj, current_tool="agy")
    )
    list_btns = _assert_callback_buttons_valid(list_card)
    assert any(len(cs["columns"]) == 2 for cs in _collect_column_sets(list_card))
    agy_btn = next(b for b in list_btns if b["value"]["action"] == "enter_agy")
    assert agy_btn["type"] == "primary"
    assert "✓" in agy_btn["text"]["content"]
    aiden_btn = next(b for b in list_btns if b["value"]["action"] == "enter_aiden")
    assert aiden_btn["disabled"] is True
    assert "behaviors" not in aiden_btn
    list_actions = [b["value"]["action"] for b in list_btns]
    assert action_ids.OPEN_TOOL_SETTINGS in list_actions
    assert action_ids.OPEN_TOOL_USAGE in list_actions
    assert "可用工具: 3/4" in list_raw

    active_sessions = {
        "agy": {"chat_id": "oc_8a1234567890abcdef91c", "session_id": "s1", "message_count": 4}
    }
    status_card, status_raw = _parse_and_validate_card(
        SystemBuilder.build_tools_status_card(tools, active_sessions, proj, current_tool="agy")
    )
    status_btns = _assert_callback_buttons_valid(status_card)
    assert any(len(cs["columns"]) == 2 for cs in _collect_column_sets(status_card))
    assert "oc_8a12…f91c" in status_raw
    assert "oc_8a1234567890abcdef91c" not in status_raw
    assert "4条" in status_raw
    assert "未就绪工具" in status_raw
    status_actions = [b["value"]["action"] for b in status_btns]
    assert action_ids.OPEN_TOOL_SETTINGS in status_actions
    assert action_ids.OPEN_TOOL_USAGE in status_actions


# ---------------------------------------------------------------------------
# F6: Shared Response Card & /help Text
# ---------------------------------------------------------------------------


def test_f6_response_card_omits_duplicate_path_and_keeps_explicit_footer() -> None:
    proj = _make_project("p1", "ghostAp", "/tmp/unique-project-root")
    card_no_footer, raw_no_footer = _parse_and_validate_card(
        ProjectBuilder.build_project_response_card(
            proj, "目录信息", "已切换到最新分支", show_buttons=False, footer=None
        )
    )
    assert "📂 项目目录:" not in raw_no_footer
    assert raw_no_footer.count("/tmp/unique-project-root") == 1

    card_with_footer, raw_with_footer = _parse_and_validate_card(
        ProjectBuilder.build_project_response_card(
            proj, "Diff 报告", "变更内容", show_buttons=False, footer="用法：/diff"
        )
    )
    assert "用法：/diff" in raw_with_footer
    assert card_with_footer["body"]["elements"][-1]["text_size"] == "notation"


def test_f6_response_card_with_active_mode_emits_dual_field_footer_buttons() -> None:
    proj = _make_project("p1", "ghostAp", "/tmp/unique-project-root")
    proj.claude_mode = True

    card, _ = _parse_and_validate_card(
        ProjectBuilder.build_project_response_card(
            proj, "Title", "Body", show_buttons=True, footer=None
        )
    )
    buttons = _assert_callback_buttons_valid(card)
    assert [b["value"]["action"] for b in buttons] == ["exit", "switch_project"]
    assert all("value" in b and b["behaviors"][0]["value"] == b["value"] for b in buttons)
    assert buttons[0]["value"]["project_id"] == "p1"


def test_f6_help_card_mentions_cockpits_and_remains_buttonless() -> None:
    modes_body = UI_TEXT["system_help_section_modes_body"]
    assert "/usage [工具]" in modes_body
    assert "/settings [工具]" in modes_body
    assert "/tools" in modes_body
    assert "/tools_status" in modes_body

    card, _ = _parse_and_validate_card(
        SystemBuilder.build_help_card(
            None,
            session_idle_timeout=1800,
            session_idle_warn_at_remaining=300,
            lock_undo_window_seconds=30,
        ),
        require_update_multi=False,
    )
    assert not _collect_column_sets(card)
    assert not _collect_buttons(card)


# ---------------------------------------------------------------------------
# AST Import Boundary & F7 Preview Parity
# ---------------------------------------------------------------------------


def test_card_builders_respect_unidirectional_import_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "src/card/builders/core.py",
        "src/card/builders/project.py",
        "src/card/builders/system.py",
    ):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    assert "delivery" not in parts and "session" not in parts and "feishu" not in parts, (
                        f"{rel} imports forbidden module {alias.name}"
                    )
                    assert "tool_telemetry" not in parts, f"{rel} imports tool_telemetry"
            elif isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                assert "delivery" not in parts and "session" not in parts and "feishu" not in parts, (
                    f"{rel} imports forbidden module {node.module}"
                )
                assert "tool_telemetry" not in parts, f"{rel} imports tool_telemetry"


def test_f2_non_active_project_cockpit_actions_resolve_target_project() -> None:
    active = _make_project("p1", "ghostAp", "/tmp/ghostAp")
    active.set_acp_configuration("agy", None, chat_id="oc_1")
    non_active = _make_project("p2", "api-svc", "/tmp/api-svc")
    non_active.set_acp_configuration("codex", None, chat_id="oc_1")

    ctx = MagicMock()
    ctx.settings = SimpleNamespace(default_working_dir="/tmp")
    ctx.project_manager.get_active_project.return_value = active
    ctx.project_manager.get_project_for_chat.side_effect = (
        lambda pid, _cid: non_active if pid == "p2" else active
    )
    ctx.mode_manager.get_mode.return_value = InteractionMode.SMART
    ctx.mode_manager.is_programming_mode.return_value = False
    ctx.thread_manager.get.return_value = None

    sys_handler = SystemHandler(ctx)
    sys_handler.show_tool_settings = MagicMock()
    sys_handler.show_tool_usage = MagicMock()

    sys_handler.handle_tool_cockpit_action(
        "om_1", "oc_1", "p2", {"action": "open_tool_settings", "project_id": "p2"}
    )
    sys_handler.show_tool_settings.assert_called_once()
    assert sys_handler.show_tool_settings.call_args[0][2] == "codex"
    assert sys_handler.show_tool_settings.call_args[0][3] is non_active

    sys_handler.handle_tool_cockpit_action(
        "om_2", "oc_1", "p2", {"action": "open_tool_usage", "project_id": "p2"}
    )
    sys_handler.show_tool_usage.assert_called_once()
    assert sys_handler.show_tool_usage.call_args[0][2] == "codex"
    assert sys_handler.show_tool_usage.call_args[0][3] is non_active


def test_f7_ux_html_preview_parity() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "ux" / "project-and-navigation-cards.html").read_text(encoding="utf-8")
    for expected in (
        "375px",
        "项目看板",
        "切换到此项目",
        "新项目已创建",
        "选择工具与模型",
        "快捷菜单",
        "Workflow 配置",
        "工具选择",
        "工具状态",
    ):
        assert expected in html, f"missing {expected!r} in ux/project-and-navigation-cards.html"


# ---------------------------------------------------------------------------
# M1 + M2 Cross-Feature Integration: Command Cards with AGY-B
# ---------------------------------------------------------------------------


def test_m1_m2_projects_board_detail_menu_and_response_card_support_agy_b(tmp_path: Path) -> None:
    # 1. /projects status board: active AGY-B(3) + idle AGY-B/<model>
    active_p = _make_project("p1", "ghostAp", str(tmp_path / "ghostAp"), last_active=1700000200.0)
    active_p.set_programming_mode("agy_b", True, session_id="sess-b1", query_count=3, chat_id="oc_1")
    active_p.update_programming_snapshot(
        "agy_b", query="fix card", query_count=3, session_id="sess-b1", chat_id="oc_1"
    )
    active_p.set_acp_configuration("agy_b", "gemini-4-hydrogen/high", chat_id="oc_1")

    idle_p = _make_project("p2", "sidecar", str(tmp_path / "sidecar"), last_active=1700000100.0)
    idle_p.set_acp_configuration("agy_b", "gemini-4-hydrogen", chat_id="oc_1")

    board_card, board_raw = _parse_and_validate_card(
        ProjectBuilder.build_status_board_card(
            [idle_p, active_p], current_project_id="p1", page=1, page_size=5, chat_id="oc_1"
        )
    )
    board_btns = _assert_callback_buttons_valid(board_card)
    assert "🪐 AGY-B(3)" in board_raw
    assert "gemini-4-hydrogen/high" in board_raw
    assert "🪐 AGY-B/gemini-4-hydrogen" in board_raw
    p1_actions = [
        b["value"]["action"]
        for b in board_btns
        if (b.get("value") or {}).get("project_id") == "p1"
    ]
    assert p1_actions == ["continue_dev", "open_tool_settings", "open_tool_usage", "show_detail"]

    # 2. Project Detail (active & idle) + 1-click cockpit routing on active and non-active projects
    detail_active, detail_active_raw = _parse_and_validate_card(
        ProjectBuilder.build_project_status_report_card(
            active_p, str(tmp_path), chat_id="oc_1", is_current=True
        )
    )
    _assert_callback_buttons_valid(detail_active)
    assert "🪐 AGY-B: `sess-b1`" in detail_active_raw
    assert "对话数: 3" in detail_active_raw

    detail_idle, detail_idle_raw = _parse_and_validate_card(
        ProjectBuilder.build_project_status_report_card(
            idle_p, str(tmp_path), chat_id="oc_1", is_current=False
        )
    )
    detail_idle_btns = _assert_callback_buttons_valid(detail_idle)
    assert "默认工具: 🪐 AGY-B · 模型: `gemini-4-hydrogen`" in detail_idle_raw
    detail_idle_actions = [b["value"]["action"] for b in detail_idle_btns]
    assert "switch_to" in detail_idle_actions
    assert "open_tool_settings" in detail_idle_actions
    assert "open_tool_usage" in detail_idle_actions
    assert "show_tool_commands" in detail_idle_actions

    # Cockpit & /cmds routing for both active p1 and non-active p2
    ctx = MagicMock()
    ctx.settings = SimpleNamespace(default_working_dir=str(tmp_path))
    ctx.project_manager.get_active_project.return_value = active_p
    ctx.project_manager.get_project_for_chat.side_effect = (
        lambda pid, _cid: idle_p if pid == "p2" else active_p
    )
    ctx.mode_manager.get_mode.side_effect = lambda _cid, project_id=None: (
        InteractionMode.AGY_B if project_id == "p1" else InteractionMode.SMART
    )
    ctx.mode_manager.is_programming_mode.side_effect = lambda _cid, project_id=None: project_id == "p1"
    ctx.thread_manager.get.return_value = None

    sys_handler = SystemHandler(ctx)
    sys_handler.show_tool_settings = MagicMock()
    sys_handler.show_tool_usage = MagicMock()
    sys_handler.show_tool_commands = MagicMock()
    sys_handler.reply_card = MagicMock(return_value="om_menu_b")

    for target_proj in (active_p, idle_p):
        pid = target_proj.project_id
        sys_handler.show_tool_settings.reset_mock()
        sys_handler.show_tool_usage.reset_mock()
        sys_handler.show_tool_commands.reset_mock()

        sys_handler.handle_tool_cockpit_action(
            "om_s", "oc_1", pid, {"action": "open_tool_settings", "project_id": pid}
        )
        assert sys_handler.show_tool_settings.call_args[0][2] == "agy_b"
        assert sys_handler.show_tool_settings.call_args[0][3] is target_proj

        sys_handler.handle_tool_cockpit_action(
            "om_u", "oc_1", pid, {"action": "open_tool_usage", "project_id": pid}
        )
        assert sys_handler.show_tool_usage.call_args[0][2] == "agy_b"
        assert sys_handler.show_tool_usage.call_args[0][3] is target_proj

        sys_handler.handle_show_tool_commands(
            "om_c", "oc_1", pid, {"action": "show_tool_commands", "project_id": pid}
        )
        # show_tool_commands resolves active AGY_B on p1 and saved agy_b on p2
        sys_handler.show_tool_commands.assert_called_once()
        assert sys_handler.show_tool_commands.call_args[0][3] is target_proj
        assert (
            sys_handler._resolve_active_programming_mode_key("oc_1", target_proj)
            or sys_handler._resolve_current_acp_tool("oc_1", target_proj)
        ) == "agy_b"

    # 3. /menu (handle_menu_command) with active AGY_B and idle agy_b passes tool_name="agy_b"
    sys_handler.handle_menu_command("om_menu_1", "oc_1", project=active_p)
    menu_active_raw = sys_handler.reply_card.call_args[0][1]
    menu_active_card = json.loads(menu_active_raw)
    menu_active_btns = _assert_callback_buttons_valid(menu_active_card)
    assert "🪐 **AGY-B**" in menu_active_raw
    assert "gemini-4-hydrogen/high" in menu_active_raw
    assert "运行中" in menu_active_raw
    for action_name in ("open_tool_settings", "open_tool_usage", "show_tool_commands"):
        btn = next(b for b in menu_active_btns if b["value"]["action"] == action_name)
        assert btn["value"]["tool_name"] == "agy_b"

    sys_handler.handle_menu_command("om_menu_2", "oc_1", project=idle_p)
    menu_idle_raw = sys_handler.reply_card.call_args[0][1]
    menu_idle_card = json.loads(menu_idle_raw)
    menu_idle_btns = _assert_callback_buttons_valid(menu_idle_card)
    assert "🪐 **AGY-B**" in menu_idle_raw
    assert "已选默认" in menu_idle_raw
    for action_name in ("open_tool_settings", "open_tool_usage", "show_tool_commands"):
        btn = next(b for b in menu_idle_btns if b["value"]["action"] == action_name)
        assert btn["value"]["tool_name"] == "agy_b"

    # 4. Shared response card with agy_b_mode=True -> 🪐 <project> · AGY-B + dual-field buttons
    active_p.agy_b_mode = True
    resp_card, resp_raw = _parse_and_validate_card(
        ProjectBuilder.build_project_response_card(
            active_p, "运行状态", "AGY-B 执行完成", show_buttons=True, footer=None
        )
    )
    resp_btns = _assert_callback_buttons_valid(resp_card)
    assert resp_card["header"]["title"]["content"] == "🪐 ghostAp · AGY-B"
    assert [b["value"]["action"] for b in resp_btns] == ["exit", "switch_project"]
    assert all("value" in b and b["behaviors"][0]["value"] == b["value"] for b in resp_btns)
    assert "📂 项目目录:" not in resp_raw
