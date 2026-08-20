from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.autonomous.data.query import AuditFailedError
from src.autonomous.provisioning.fire_service import EmployeeFireRequest
from src.autonomous.provisioning.fire_state import (
    FireCleanupMode,
    FireEffectState,
    FirePhase,
)
from src.autonomous.provisioning.hire_service import HireAdmissionError
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.handlers.employee import EmployeeHandler
from src.feishu.handlers.system import SystemHandler
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.thread import (
    set_current_is_p2p,
    set_current_sender_id,
    set_current_sender_union_id,
    set_current_tenant_key,
    set_current_thread_id,
)
from src.trust.models import ActorKind, EffectiveTrust, TrustZone


@pytest.fixture(autouse=True)
def _reset_request_context():
    set_current_sender_id(None)
    set_current_sender_union_id(None)
    set_current_tenant_key(None)
    set_current_is_p2p(False)
    set_current_thread_id(None)
    yield
    set_current_sender_id(None)
    set_current_sender_union_id(None)
    set_current_tenant_key(None)
    set_current_is_p2p(False)
    set_current_thread_id(None)


def _handler(*, admins: frozenset[str] = frozenset({"ou_admin"})):
    ctx = MagicMock()
    ctx.settings = SimpleNamespace(
        admin_user_ids=admins,
        app_id="cli_main_bot",
        default_acp_tool="",
    )
    ctx.tenant_key_resolver = MagicMock(return_value="tenant-a")
    ctx.employee_hire_service = MagicMock()
    ctx.employee_fire_service = MagicMock()
    ctx.employee_hire_readiness = MagicMock(
        return_value=SimpleNamespace(ready=True, blockers=())
    )
    ctx.employee_data_composition = MagicMock()
    ctx.employee_data_composition.service.shard_timezone = "UTC"
    handler = EmployeeHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()
    handler.reply_card = MagicMock()
    return handler, ctx


def _authorize(*, union_id: str | None = "on_admin", p2p: bool = True) -> None:
    set_current_sender_id("ou_admin")
    set_current_sender_union_id(union_id)
    set_current_tenant_key("tenant-a")
    set_current_is_p2p(p2p)


@pytest.mark.parametrize(
    ("text", "method_name", "expected_args"),
    [
        ("/hire Atlas", "hire_employee", "Atlas"),
        ("/h Atlas", "hire_employee", "Atlas"),
        ("/fire Atlas --drain", "fire_employee", "Atlas --drain"),
        ("/history Atlas", "show_employee_history", "Atlas"),
        ("/employee-memory Atlas", "show_employee_memory", "Atlas"),
    ],
)
def test_system_routes_every_employee_command(
    text: str,
    method_name: str,
    expected_args: str,
) -> None:
    handler = SystemHandler(MagicMock())
    target = MagicMock()
    setattr(handler.employee, method_name, target)
    match = SlashCommandParser.parse(text)

    assert match is not None
    assert SystemHandler.is_interceptable_command_match(match) is True

    handler.handle_intercepted_command(
        "om_command",
        "oc_admin_dm",
        text,
        command_match=match,
    )

    target.assert_called_once_with("om_command", "oc_admin_dm", expected_args)


@pytest.mark.parametrize(
    "text",
    [
        "/hire Atlas",
        "/fire Atlas",
        "/history Atlas",
        "/employee-memory Atlas",
    ],
)
@pytest.mark.parametrize(
    ("mode", "topic_context"),
    [
        (InteractionMode.SMART, False),
        (InteractionMode.COCO, False),
        (InteractionMode.CLAUDE, False),
        (InteractionMode.AIDEN, False),
        (InteractionMode.CODEX, False),
        (InteractionMode.GEMINI, False),
        (InteractionMode.TRAEX, False),
        (InteractionMode.GROK, False),
        (InteractionMode.DSH, False),
        (InteractionMode.SMART, True),
    ],
)
def test_dispatcher_intercepts_employee_commands_before_every_active_lane(
    text: str,
    mode: InteractionMode,
    topic_context: bool,
) -> None:
    client = MagicMock()
    active = MagicMock()
    system = MagicMock()
    system.is_interceptable_command_match.return_value = True
    client._handler_ctx.handlers = {
        "coco": active,
        "system": system,
        "project": MagicMock(),
    }
    client._get_effective_mode.return_value = (
        mode,
        mode is not InteractionMode.SMART,
    )
    client._is_topic_engine_context.return_value = topic_context
    client._current_trust_can_dispatch.return_value = True
    if topic_context:
        set_current_thread_id("omt_workflow")
        client._thread_manager.get.return_value = SimpleNamespace(mode="workflow")
    dispatcher = MessageDispatcher(client)
    match = SlashCommandParser.parse(text)

    dispatcher.process_with_intent(
        "om_command",
        "oc_admin_dm",
        text,
        command_match=match,
    )

    system.handle_intercepted_command.assert_called_once_with(
        "om_command",
        "oc_admin_dm",
        text,
        None,
        command_match=match,
    )
    active.handle_message.assert_not_called()
    client._intent_recognizer.recognize.assert_not_called()


@pytest.mark.parametrize(
    ("sender", "p2p", "tenant", "union_id"),
    [
        ("ou_intruder", True, "tenant-a", "on_intruder"),
        ("ou_admin", False, "tenant-a", "on_admin"),
        ("ou_admin", True, "", "on_admin"),
        ("ou_admin", True, "tenant-a", None),
    ],
)
def test_hire_denies_before_readiness_or_service(
    sender: str,
    p2p: bool,
    tenant: str,
    union_id: str | None,
) -> None:
    handler, ctx = _handler()
    set_current_sender_id(sender)
    set_current_sender_union_id(union_id)
    set_current_tenant_key(tenant or None)
    set_current_is_p2p(p2p)
    ctx.tenant_key_resolver.return_value = tenant

    handler.hire_employee("om_hire", "oc_admin_dm", "Atlas")

    ctx.employee_hire_readiness.assert_not_called()
    ctx.employee_hire_service.start_hire.assert_not_called()
    handler.reply_text.assert_called_once()


@pytest.mark.parametrize(
    ("sender", "p2p", "transport_tenant"),
    [
        ("ou_intruder", True, "tenant-a"),
        ("ou_admin", False, "tenant-a"),
        ("", True, "tenant-a"),
        ("ou_admin", True, ""),
    ],
    ids=("non-admin", "group", "missing-sender", "missing-tenant"),
)
def test_employee_roster_denies_before_tenant_resolution_or_projection(
    sender: str,
    p2p: bool,
    transport_tenant: str,
) -> None:
    handler, ctx = _handler()
    set_current_sender_id(sender or None)
    set_current_tenant_key(transport_tenant or None)
    set_current_is_p2p(p2p)

    handler.list_employees_roster("om_roster", "oc_admin_dm")

    ctx.tenant_key_resolver.assert_not_called()
    ctx.employee_hire_service.list_employee_roster.assert_not_called()
    handler.reply_card.assert_not_called()
    handler.reply_text.assert_called_once()


def test_employee_roster_allows_configured_admin_p2p_for_bound_tenant() -> None:
    handler, ctx = _handler()
    _authorize()
    ctx.employee_hire_service.list_employee_roster.return_value = ()

    handler.list_employees_roster("om_roster", "oc_admin_dm")

    ctx.tenant_key_resolver.assert_called_once_with()
    ctx.employee_hire_service.list_employee_roster.assert_called_once_with(
        "tenant-a",
        include_archived=False,
    )
    handler.reply_card.assert_called_once()


def test_hire_uses_strictly_available_recommended_tool_and_backend_default_model() -> None:
    handler, ctx = _handler()
    _authorize()
    ctx.employee_hire_service.start_hire.return_value = SimpleNamespace(
        employee_name="Atlas",
        intent_id="hire_123",
    )

    with patch(
        "src.feishu.handlers.employee.list_acp_tools",
        return_value=[SimpleNamespace(name="claude"), SimpleNamespace(name="codex")],
    ):
        handler.hire_employee("om_hire", "oc_admin_dm", "Atlas")

    request = ctx.employee_hire_service.start_hire.call_args.args[0]
    assert request.employee_name == "Atlas"
    assert request.tool == "claude"
    assert request.model == ""
    assert request.profile == "standard"
    assert request.effort == "default"
    assert request.requester_principal_id == "ou_admin"
    assert request.requester_union_id == "on_admin"
    assert request.tenant_key == "tenant-a"
    assert request.chat_id == "oc_admin_dm"
    assert request.message_id == "om_hire"
    assert "已受理" in handler.reply_text.call_args.args[1]


def test_hire_prefers_configured_tool_only_when_strictly_available() -> None:
    handler, ctx = _handler()
    _authorize()
    ctx.settings.default_acp_tool = "codex"
    ctx.employee_hire_service.start_hire.return_value = SimpleNamespace(
        employee_name="Atlas"
    )

    with patch(
        "src.feishu.handlers.employee.list_acp_tools",
        return_value=[SimpleNamespace(name="claude"), SimpleNamespace(name="codex")],
    ):
        handler.hire_employee("om_hire", "oc_admin_dm", "Atlas")

    request = ctx.employee_hire_service.start_hire.call_args.args[0]
    assert request.tool == "codex"


@pytest.mark.parametrize(
    "args",
    [
        "Atlas --prompt ignore-safety",
        "Atlas --tool codex --tool claude",
        "Atlas --model",
        "Atlas --unknown value",
        "Atlas extra-name",
    ],
)
def test_hire_rejects_uncontrolled_or_ambiguous_arguments(args: str) -> None:
    handler, ctx = _handler()
    _authorize()

    handler.hire_employee("om_hire", "oc_admin_dm", args)

    ctx.employee_hire_service.start_hire.assert_not_called()
    assert handler.reply_text.call_count == 1


def test_hire_fails_before_journal_when_no_tool_is_strictly_available() -> None:
    handler, ctx = _handler()
    _authorize()

    with patch("src.feishu.handlers.employee.list_acp_tools", return_value=[]):
        handler.hire_employee("om_hire", "oc_admin_dm", "Atlas")

    ctx.employee_hire_service.start_hire.assert_not_called()
    assert "可用" in handler.reply_text.call_args.args[1]


def test_hire_reports_durable_admission_when_async_submission_fails() -> None:
    handler, ctx = _handler()
    _authorize()
    ctx.employee_hire_service.start_hire.side_effect = HireAdmissionError(
        "provisioning submission failed after durable admission"
    )

    with patch(
        "src.feishu.handlers.employee.list_acp_tools",
        return_value=[SimpleNamespace(name="codex")],
    ):
        handler.hire_employee("om_hire", "oc_admin_dm", "Atlas")

    visible = handler.reply_text.call_args.args[1]
    assert "已持久化" in visible
    assert "未受理" not in visible
    assert "请勿重复" in visible


def test_fire_builds_authoritative_confirmation_request() -> None:
    handler, ctx = _handler()
    _authorize()
    ctx.employee_fire_service.confirm_external_disposition.return_value = SimpleNamespace(
        phase=FirePhase.ARCHIVED,
        cleanup_mode=FireCleanupMode.EXTERNAL_UNKNOWN,
        external_disposition_confirmed=True,
        effects=(),
        error_code="",
        app_id="cli_employee",
    )

    handler.fire_employee(
        "om_fire",
        "oc_admin_dm",
        "Atlas --confirm-app-disposed cli_employee",
    )

    ctx.employee_fire_service.start_fire.assert_not_called()
    ctx.employee_fire_service.confirm_external_disposition.assert_called_once()
    request, disposition = (
        ctx.employee_fire_service.confirm_external_disposition.call_args.args
    )
    assert isinstance(request, EmployeeFireRequest)
    assert request.employee == "Atlas"
    assert request.tenant_key == "tenant-a"
    assert request.requester_principal_id == "ou_admin"
    assert disposition == "cli_employee"
    reply = handler.reply_text.call_args.args[1]
    assert "处置确认" in reply
    assert "GhostAP 已删除" not in reply


def test_fire_drain_reports_automatic_completion_after_active_work() -> None:
    handler, ctx = _handler()
    _authorize()
    ctx.employee_fire_service.start_fire.return_value = SimpleNamespace(
        phase=FirePhase.RETIRING,
        drain=True,
        effects=(("execution_quiesce", FireEffectState.EXECUTING),),
        error_code="",
    )

    handler.fire_employee("om_fire", "oc_admin_dm", "Atlas --drain")

    request = ctx.employee_fire_service.start_fire.call_args.args[0]
    assert request.drain is True
    visible = handler.reply_text.call_args.args[1]
    assert "自然结束" in visible
    assert "自动继续" in visible
    assert "人工核对" not in visible


@pytest.mark.parametrize(
    "args",
    [
        "Atlas --drain --drain",
        "Atlas --confirm-app-disposed",
        "Atlas --confirm-app-disposed not-an-app",
        "Atlas --unknown",
        "Atlas Beacon",
    ],
)
def test_fire_rejects_ambiguous_or_uncontrolled_arguments(args: str) -> None:
    handler, ctx = _handler()
    _authorize()

    handler.fire_employee("om_fire", "oc_admin_dm", args)

    ctx.employee_fire_service.start_fire.assert_not_called()
    ctx.employee_fire_service.confirm_external_disposition.assert_not_called()


def test_fire_denies_group_admin_before_service() -> None:
    handler, ctx = _handler()
    _authorize(p2p=False)

    handler.fire_employee("om_fire", "oc_group", "Atlas")

    ctx.employee_fire_service.start_fire.assert_not_called()
    ctx.employee_fire_service.confirm_external_disposition.assert_not_called()


@pytest.mark.parametrize("method", ["show_employee_history", "show_employee_memory"])
def test_employee_data_commands_authorize_before_target_lookup(method: str) -> None:
    handler, ctx = _handler()
    set_current_sender_id("ou_intruder")
    set_current_is_p2p(True)
    set_current_tenant_key("tenant-a")

    getattr(handler, method)("om_read", "oc_admin_dm", "Atlas")

    ctx.employee_hire_service.list_employee_roster.assert_not_called()
    ctx.employee_data_composition.query.query.assert_not_called()
    ctx.employee_data_composition.memory_query.query.assert_not_called()


def test_history_uses_authenticated_tenant_query_and_only_safe_metadata() -> None:
    handler, ctx = _handler()
    _authorize()
    employee = SimpleNamespace(agent_id="agt_atlas", name="Atlas")
    ctx.employee_hire_service.list_employee_roster.return_value = (employee,)
    record = SimpleNamespace(
        ended_at="2026-08-12T12:34:56Z",
        status="completed",
        tool="codex",
        model="",
        effort="high",
        safe_summary_text="safe summary",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        blob_ref={"secret": "must-not-render"},
    )
    ctx.employee_data_composition.query.query.return_value = SimpleNamespace(
        records=(record,),
        total_available=1,
    )

    handler.show_employee_history("om_history", "oc_admin_dm", "atlas")

    request, spec = ctx.employee_data_composition.query.query.call_args.args
    assert request.principal_id == "ou_admin"
    assert request.tenant_key == "tenant-a"
    assert request.receiving_bot_app_id == "cli_main_bot"
    assert request.chat_id == "oc_admin_dm"
    assert request.chat_type == "p2p"
    assert request.requested_agent_id == "agt_atlas"
    assert spec.page_size == 50
    visible = handler.reply_text.call_args.args[1]
    assert "safe summary" in visible
    assert "codex" in visible
    assert "must-not-render" not in visible


def test_memory_uses_audited_full_l1_query_and_bounds_output() -> None:
    handler, ctx = _handler()
    _authorize()
    set_current_thread_id("omt_root")
    employee = SimpleNamespace(agent_id="agt_atlas", name="Atlas")
    ctx.employee_hire_service.list_employee_roster.return_value = (employee,)
    ctx.employee_data_composition.memory_query.query.return_value = SimpleNamespace(
        content="x" * 30_000,
        scope="full_l1",
    )

    handler.show_employee_memory("om_memory", "oc_admin_dm", "Atlas")

    request, spec = ctx.employee_data_composition.memory_query.query.call_args.args
    assert request.thread_root_id == "omt_root"
    assert request.requested_agent_id == "agt_atlas"
    assert spec.agent_id == "agt_atlas"
    assert spec.full_l1 is True
    visible = handler.reply_text.call_args.args[1]
    assert len(visible) < 21_000
    assert "截断" in visible


def test_memory_redacts_credentials_and_absolute_paths_before_delivery() -> None:
    handler, ctx = _handler()
    _authorize()
    employee = SimpleNamespace(agent_id="agt_atlas", name="Atlas")
    ctx.employee_hire_service.list_employee_roster.return_value = (employee,)
    ctx.employee_data_composition.memory_query.query.return_value = SimpleNamespace(
        content=(
            'api_key="plain-secret-value"\n'
            '{"nested":{"client_secret":"nested-secret-value"}}\n'
            "password='top secret phrase'\n"
            '{"client_secret":"unterminated-secret-value\n'
            "workspace=/data00/home/alice/private/project/config.yaml\n"
            "windows=C:\\Users\\alice\\private\\settings.json"
        ),
        scope="full_l1",
    )

    handler.show_employee_memory("om_memory", "oc_admin_dm", "Atlas")

    visible = handler.reply_text.call_args.args[1]
    for forbidden in (
        "plain-secret-value",
        "nested-secret-value",
        "top secret phrase",
        "secret phrase",
        "unterminated-secret-value",
        "/data00/home/alice/private/project/config.yaml",
        "C:\\Users\\alice\\private\\settings.json",
    ):
        assert forbidden not in visible
    assert "redacted" in visible


@pytest.mark.parametrize(
    ("method", "query_path"),
    [
        ("show_employee_history", "query"),
        ("show_employee_memory", "memory_query"),
    ],
)
def test_data_audit_failure_returns_no_read_content(
    method: str,
    query_path: str,
) -> None:
    handler, ctx = _handler()
    _authorize()
    employee = SimpleNamespace(agent_id="agt_atlas", name="Atlas")
    ctx.employee_hire_service.list_employee_roster.return_value = (employee,)
    query = getattr(ctx.employee_data_composition, query_path).query
    query.side_effect = AuditFailedError("secret-audit-backend-detail")

    getattr(handler, method)("om_read", "oc_admin_dm", "Atlas")

    visible = handler.reply_text.call_args.args[1]
    assert "secret-audit-backend-detail" not in visible
    assert "审计" in visible


@pytest.mark.parametrize("command", ["/history Atlas", "/employee-memory Atlas"])
def test_dispatcher_routes_sensitive_employee_read_without_action_policy(
    command: str,
) -> None:
    client = MagicMock()
    client._handler_ctx.handlers = {
        "coco": MagicMock(),
        "system": MagicMock(),
        "project": MagicMock(),
    }
    client._get_effective_mode.return_value = (InteractionMode.SMART, False)
    client._current_trust_can_dispatch.return_value = True
    client._handler_ctx.handlers[
        "system"
    ].is_interceptable_command_match.return_value = True
    dispatcher = MessageDispatcher(client)

    dispatcher.process_with_intent(
        "om_read",
        "oc_admin_dm",
        command,
        command_match=SlashCommandParser.parse(command),
        effective_trust=MagicMock(),
    )

    dispatcher.system.handle_intercepted_command.assert_called_once()


@pytest.mark.parametrize("command", ["/employees", "/roster"])
def test_dispatcher_routes_employee_roster_without_action_policy(
    command: str,
) -> None:
    client = MagicMock()
    client._handler_ctx.handlers = {
        "coco": MagicMock(),
        "system": MagicMock(),
        "project": MagicMock(),
    }
    client._get_effective_mode.return_value = (InteractionMode.SMART, False)
    client._current_trust_can_dispatch.return_value = True
    client._handler_ctx.handlers[
        "system"
    ].is_interceptable_command_match.return_value = True
    dispatcher = MessageDispatcher(client)
    command_match = SlashCommandParser.parse(command)

    assert command_match is not None
    assert command_match.command == "/employees"
    dispatcher.process_with_intent(
        "om_roster",
        "oc_admin_dm",
        command,
        command_match=command_match,
        effective_trust=MagicMock(),
    )

    dispatcher.system.handle_intercepted_command.assert_called_once()


@pytest.mark.parametrize("command", ["/history Atlas", "/employee-memory Atlas"])
def test_ws_ingress_allows_sensitive_employee_read_without_action_policy(
    command: str,
) -> None:
    client = object.__new__(FeishuWSClient)
    trust = EffectiveTrust(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        actor=ActorKind.OWNER,
        managed_group=MagicMock(revision=1),
        group_revision=1,
        grant_revision=1,
    )
    allowed = client._managed_ingress_action_allowed(
        trust,
        text=command,
        command_match=SlashCommandParser.parse(command),
    )

    assert allowed is True


@pytest.mark.parametrize("command", ["/employees", "/roster"])
def test_ws_ingress_allows_employee_roster_without_action_policy(
    command: str,
) -> None:
    client = object.__new__(FeishuWSClient)
    trust = EffectiveTrust(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        actor=ActorKind.OWNER,
        managed_group=MagicMock(revision=1),
        group_revision=1,
        grant_revision=1,
    )
    command_match = SlashCommandParser.parse(command)
    assert command_match is not None
    assert command_match.command == "/employees"
    allowed = client._managed_ingress_action_allowed(
        trust,
        text=command,
        command_match=command_match,
    )

    assert allowed is True
