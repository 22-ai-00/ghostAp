from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.intent_recognizer import IntentRecognizer, IntentType
from src.card.builders.system import SystemBuilder
from src.feishu.slash_command_parser import SlashCommandParser


def _employee(
    name: str,
    state: str,
    *,
    agent_id: str,
    role: str = "工程师",
    tool: str = "codex",
    model: str = "gpt-5.6-sol",
) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        tenant_key="tenant-a",
        owner_principal_id="ou_owner_must_not_leak",
        bot_principal_id="bot_principal_must_not_leak",
        name=name,
        emoji="🤖",
        state=SimpleNamespace(value=state),
        role=role,
        tool=tool,
        model=model,
        profile="standard",
        effort="high",
        member_groups=("oc_secret_group",),
        persona="secret persona must not leak",
        permissions=("secret.permission",),
        created_at=1_786_000_000.0,
    )


def _handler(*, roster=(), error: Exception | None = None):
    from src.feishu.handlers.employee import EmployeeHandler

    ctx = MagicMock()
    ctx.tenant_key_resolver = MagicMock(return_value="tenant-a")
    ctx.employee_hire_service = MagicMock()
    if error is None:
        ctx.employee_hire_service.list_employee_roster.return_value = tuple(roster)
    else:
        ctx.employee_hire_service.list_employee_roster.side_effect = error
    handler = EmployeeHandler(ctx)
    handler.reply_card = MagicMock(return_value="om_roster_card")
    handler.reply_text = MagicMock(return_value="om_roster_text")
    return handler, ctx


@pytest.mark.parametrize("command", ["/employees", "/roster"])
def test_employee_roster_command_aliases_share_tenant_scoped_handler(command: str) -> None:
    match = SlashCommandParser.parse(command)
    assert match is not None
    assert match.command == "/employees"

    employee = _employee("Atlas", "active", agent_id="agt_atlas")
    handler, ctx = _handler(roster=(employee,))

    handler.list_employees_roster("om_request", "oc_chat")

    ctx.tenant_key_resolver.assert_called_once_with()
    ctx.employee_hire_service.list_employee_roster.assert_called_once_with(
        "tenant-a",
        include_archived=False,
    )
    handler.reply_card.assert_called_once()
    handler.reply_text.assert_not_called()


def test_employee_roster_empty_is_not_reported_as_unavailable() -> None:
    handler, ctx = _handler(roster=())

    handler.list_employees_roster("om_request", "oc_chat")

    ctx.employee_hire_service.list_employee_roster.assert_called_once_with(
        "tenant-a",
        include_archived=False,
    )
    handler.reply_card.assert_called_once()
    handler.reply_text.assert_not_called()
    rendered = json.dumps(handler.reply_card.call_args.args[1], ensure_ascii=False)
    assert "暂无" in rendered
    assert "暂不可用" not in rendered


def test_employee_roster_projection_error_is_visible_and_not_misreported_empty() -> None:
    handler, _ctx = _handler(error=RuntimeError("raw journal path and secret"))

    handler.list_employees_roster("om_request", "oc_chat")

    handler.reply_card.assert_not_called()
    handler.reply_text.assert_called_once()
    visible = handler.reply_text.call_args.args[1]
    assert "暂不可用" in visible
    assert "raw journal path" not in visible
    assert "secret" not in visible
    assert "暂无" not in visible


def _roster_entries() -> list[dict[str, object]]:
    return [
        {
            "agent_id": "agt_archived_zulu",
            "name": "Zulu",
            "emoji": "📦",
            "state": "archived",
            "role": "发布工程师",
            "tool": "traex",
            "model": "c_o_new_thinking",
            "profile": "max",
            "effort": "max",
            "group_count": 0,
            "tenant_key": "tenant-secret",
            "owner_principal_id": "ou-secret",
            "bot_principal_id": "bot-secret",
            "persona": "persona-secret",
            "permissions": ["permission-secret"],
        },
        {
            "agent_id": "agt_action_security",
            "name": "Security",
            "emoji": "🛡️",
            "state": "action_required",
            "role": "安全审计",
            "tool": "codex",
            "model": "gpt-5.6-sol",
            "profile": "standard",
            "effort": "high",
            "group_count": 2,
        },
        {
            "agent_id": "agt_archived_alpha",
            "name": "Alpha",
            "emoji": "📦",
            "state": "archived",
            "role": "前端工程师",
            "tool": "coco",
            "model": "default",
            "profile": "standard",
            "effort": "default",
            "group_count": 0,
        },
        {
            "agent_id": "agt_action_ops",
            "name": "Ops",
            "emoji": "🧰",
            "state": "action_required",
            "role": "平台工程师",
            "tool": "claude",
            "model": "default",
            "profile": "standard",
            "effort": "high",
            "group_count": 1,
        },
        {
            "agent_id": "agt_archived_gamma",
            "name": "Gamma",
            "emoji": "📦",
            "state": "archived",
            "role": "测试工程师",
            "tool": "gemini",
            "model": "default",
            "profile": "standard",
            "effort": "default",
            "group_count": 0,
        },
        {
            "agent_id": "agt_archived_beta",
            "name": "Beta",
            "emoji": "📦",
            "state": "archived",
            "role": "后端工程师",
            "tool": "aiden",
            "model": "default",
            "profile": "standard",
            "effort": "default",
            "group_count": 0,
        },
    ]


def test_employee_roster_card_counts_states_sorts_stably_and_redacts_authority() -> None:
    entries = _roster_entries()
    entries[3]["role"] = ""

    msg_type, content = SystemBuilder.build_employee_roster_card(entries)
    reverse_type, reverse_content = SystemBuilder.build_employee_roster_card(
        list(reversed(entries))
    )

    assert msg_type == reverse_type == "interactive"
    assert json.loads(content) == json.loads(reverse_content)
    card = json.loads(content)
    assert card["schema"] == "2.0"
    rendered = json.dumps(card, ensure_ascii=False)
    assert "当前员工 · 2" in rendered
    assert "共 2" in rendered
    assert "在职 0" in rendered
    assert "需处理 2" in rendered
    assert "已归档" not in rendered

    positions = [rendered.index(name) for name in ("Ops", "Security")]
    assert positions == sorted(positions)
    for archived_name in ("Alpha", "Beta", "Gamma", "Zulu"):
        assert archived_name not in rendered

    assert "/employee-role Ops 职责" in rendered
    assert "/employee-role Security 职责" not in rendered

    unescaped = rendered.replace("\\", "")
    for entry in entries:
        assert str(entry["agent_id"]) not in unescaped

    for forbidden in (
        "tenant-secret",
        "ou-secret",
        "bot-secret",
        "persona-secret",
        "permission-secret",
    ):
        assert forbidden not in rendered


def test_employee_roster_card_never_exposes_ids_for_duplicate_names() -> None:
    entries = [
        {
            "agent_id": "agt_same_name_alpha",
            "name": "同名员工",
            "state": "active",
            "role": "coder",
            "tool": "codex",
            "model": "gpt-5.6-sol",
        },
        {
            "agent_id": "agt_same_name_beta",
            "name": "同名员工",
            "state": "action_required",
            "role": "reviewer",
            "tool": "claude",
            "model": "default",
        },
    ]

    _msg_type, content = SystemBuilder.build_employee_roster_card(entries)

    rendered = json.dumps(json.loads(content), ensure_ascii=False)
    assert rendered.count("同名员工") == 2
    unescaped = rendered.replace("\\", "")
    for entry in entries:
        assert str(entry["agent_id"]) not in unescaped


def test_employee_roster_card_has_distinct_empty_state() -> None:
    msg_type, content = SystemBuilder.build_employee_roster_card([])

    assert msg_type == "interactive"
    rendered = json.dumps(json.loads(content), ensure_ascii=False)
    assert "共 0" in rendered
    assert "暂无" in rendered
    assert "暂不可用" not in rendered


@pytest.mark.parametrize(
    "text",
    [
        "员工列表",
        "数字员工列表",
        "员工目录",
        "有哪些员工",
        "都有哪些员工",
        "有哪些数字员工？",
        "查看员工列表",
        "列出所有员工",
    ],
)
def test_employee_roster_natural_language_uses_deterministic_intent(text: str) -> None:
    result = IntentRecognizer().recognize(text, "smart")

    assert result.primary_intent is IntentType.LIST_EMPLOYEES
    assert result.confidence >= 0.9


@pytest.mark.parametrize(
    "text",
    [
        "实现员工列表页面",
        "给员工列表增加分页",
        "修复员工目录组件的样式",
        "编写查询员工的数据库接口",
    ],
)
def test_employee_roster_natural_language_does_not_steal_programming_tasks(text: str) -> None:
    result = IntentRecognizer().recognize(text, "smart")

    assert result.primary_intent is not IntentType.LIST_EMPLOYEES
