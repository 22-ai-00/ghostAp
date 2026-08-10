from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.feishu.handlers.employee as employee_module
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.handlers.employee import EmployeeHandler
from src.feishu.handlers.system import SystemHandler
from src.feishu.product_catalog import resolve_command
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient, TrustActionDecision
from src.thread import set_current_sender_id
from src.trust.models import ActorKind, EffectiveTrust, TrustZone

COMMAND = "/employee-role"
USAGE = "/employee-role <员工名> <职责>"


@pytest.fixture(autouse=True)
def _reset_sender_context():
    set_current_sender_id(None)
    yield
    set_current_sender_id(None)


def _reply_content(handler: EmployeeHandler) -> str:
    calls = (*handler.reply_text.call_args_list, *handler.reply_error.call_args_list)
    return " ".join(str(value) for call in calls for value in (*call.args, *call.kwargs.values()))


def _employee_handler(
    *,
    tenant_key: str = "tenant_authorized",
    service: MagicMock | None = None,
) -> tuple[EmployeeHandler, MagicMock]:
    employee_service = service or MagicMock()
    ctx = MagicMock()
    ctx.tenant_key_resolver = MagicMock(return_value=tenant_key)
    ctx.employee_hire_service = employee_service
    handler = EmployeeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()
    return handler, employee_service


def _invoke(
    handler: EmployeeHandler,
    text: str,
    *,
    sender_id: str | None = "ou_admin",
    admin_ids: frozenset[str] = frozenset({"ou_admin"}),
) -> None:
    set_current_sender_id(sender_id)
    settings = SimpleNamespace(admin_user_ids=admin_ids)
    with patch.object(
        employee_module,
        "get_settings",
        return_value=settings,
        create=True,
    ):
        match = SlashCommandParser.parse(text)
        handler.update_employee_role(
            "om_role",
            "oc_owner",
            match.args if match is not None else "",
        )


def test_employee_role_command_is_catalogued_and_parsed_as_owner_control() -> None:
    match = SlashCommandParser.parse(f"{COMMAND} 柳神 负责核心开发")

    assert match is not None
    assert match.command == COMMAND
    assert match.args == "柳神 负责核心开发"
    resolved = resolve_command(match.command, match.args)
    assert resolved is not None
    assert resolved.action.command == COMMAND
    assert resolved.action.usage == USAGE
    assert resolved.action.owner_accessible is True


def test_system_handler_intercepts_and_routes_employee_role_prefix() -> None:
    handler = SystemHandler(MagicMock())
    target = MagicMock()
    handler.employee.update_employee_role = target
    text = f"{COMMAND} 柳神 负责核心开发"
    match = SlashCommandParser.parse(text)

    assert SystemHandler.is_interceptable_command_match(match) is True
    handler.handle_intercepted_command(
        "om_role",
        "oc_owner",
        text,
        command_match=match,
    )

    target.assert_called_once_with("om_role", "oc_owner", "柳神 负责核心开发")


@pytest.mark.parametrize("text", [COMMAND, f"{COMMAND} 只有员工名"])
def test_employee_role_usage_is_actionable_and_does_not_mutate(text: str) -> None:
    handler, service = _employee_handler()

    _invoke(handler, text)

    service.update_employee_role.assert_not_called()
    assert USAGE in _reply_content(handler)


@pytest.mark.parametrize(
    ("sender_id", "admin_ids", "tenant_key"),
    [
        (None, frozenset({"ou_admin"}), "tenant_authorized"),
        ("ou_other", frozenset({"ou_admin"}), "tenant_authorized"),
        ("ou_admin", frozenset({"ou_admin"}), ""),
    ],
)
def test_employee_role_fails_closed_without_sender_admin_or_tenant(
    sender_id: str | None,
    admin_ids: frozenset[str],
    tenant_key: str,
) -> None:
    handler, service = _employee_handler(tenant_key=tenant_key)

    _invoke(handler, f"{COMMAND} 柳神 负责核心开发", sender_id=sender_id, admin_ids=admin_ids)

    service.update_employee_role.assert_not_called()
    reply = _reply_content(handler)
    assert reply
    assert any(word in reply for word in ("管理员", "权限", "身份", "租户", "暂不可用"))


def test_employee_role_passes_authoritative_request_and_replies_success() -> None:
    handler, service = _employee_handler()
    service.update_employee_role.return_value = SimpleNamespace(
        employee_name="柳神",
        role="负责核心开发",
    )

    _invoke(handler, f"{COMMAND} 柳神 负责核心开发")

    service.update_employee_role.assert_called_once()
    request = service.update_employee_role.call_args.args[0]
    assert request.tenant_key == "tenant_authorized"
    assert request.employee == "柳神"
    assert request.role == "负责核心开发"
    assert request.requester_principal_id == "ou_admin"
    assert request.message_id == "om_role"
    reply = _reply_content(handler)
    assert "柳神" in reply
    assert "负责核心开发" in reply


def test_employee_role_service_error_is_replied_without_internal_detail() -> None:
    handler, service = _employee_handler()
    service.update_employee_role.side_effect = RuntimeError("internal-secret-detail")

    _invoke(handler, f"{COMMAND} 柳神 负责核心开发")

    reply = _reply_content(handler)
    assert reply
    assert "internal-secret-detail" not in reply
    assert any(word in reply for word in ("失败", "未找到", "暂不可用"))


def test_dispatcher_treats_employee_role_as_admin_mutation() -> None:
    client = MagicMock()
    client._handler_ctx.handlers = {
        "coco": MagicMock(),
        "system": MagicMock(),
        "project": MagicMock(),
    }
    dispatcher = MessageDispatcher(client)
    dispatcher._action_matrix_allows = MagicMock(return_value=False)
    text = f"{COMMAND} 柳神 负责核心开发"

    dispatcher.process_with_intent(
        "om_role",
        "oc_owner",
        text,
        command_match=SlashCommandParser.parse(text),
        effective_trust=MagicMock(),
    )

    assert dispatcher._action_matrix_allows.call_count == 1
    assert dispatcher._action_matrix_allows.call_args.kwargs == {
        "action_name": "grant_admin"
    }
    dispatcher.system.handle_intercepted_command.assert_not_called()


def test_ws_managed_ingress_classifies_employee_role_as_grant_admin() -> None:
    client = object.__new__(FeishuWSClient)
    trust = EffectiveTrust(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        actor=ActorKind.OWNER,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )
    text = f"{COMMAND} 柳神 负责核心开发"
    with patch("src.feishu.ws_client.ActionMatrix") as matrix:
        matrix.return_value.decide.return_value = TrustActionDecision.DENY

        allowed = client._managed_ingress_action_allowed(
            trust,
            text=text,
            command_match=SlashCommandParser.parse(text),
        )

    assert allowed is False
    request = matrix.return_value.decide.call_args.args[0]
    assert request.action.value == "grant_admin"
