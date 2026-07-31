from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.autonomous.domain import EmployeeState, WorkerType
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.supervisor.employee_channels import ChannelProcessState
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.ws_client import FeishuWSClient
from src.trust.models import ActorKind, EffectiveTrust, ManagedGroupOrigin, TrustZone
from src.trust.registry import ManagedGroupRegistry

OWNER_ID = "ou_owner"
GROUP_ID = "oc_managed"


def _registry(tmp_path) -> ManagedGroupRegistry:
    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    registry.register(
        chat_id=GROUP_ID,
        owner_id=OWNER_ID,
        origin=ManagedGroupOrigin.GHOSTAP_CREATED,
        receiving_bot_ref="cli_main_bot",
        project_id="project-1",
        canonical_root_ref="/srv/project-1",
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        backend_binding_ids=("codex",),
    )
    return registry


def _message(*, text: str = "implement the task", sender_id: str = OWNER_ID) -> MagicMock:
    data = MagicMock()
    data.header.tenant_key = "tenant-1"
    data.event.message.message_id = "om_managed"
    data.event.message.chat_id = GROUP_ID
    data.event.message.chat_type = "group"
    data.event.message.create_time = "9999999999999"
    data.event.message.message_type = "text"
    data.event.message.content = '{"text":"ignored by mock parser"}'
    data.event.message.parent_id = None
    data.event.message.root_id = None
    data.event.message.thread_id = None
    data.event.message.mentions = []
    data.event.sender.sender_id.open_id = sender_id
    data.event.sender.sender_id.union_id = "on_owner"
    data._task_text = text
    return data


def _client(registry: ManagedGroupRegistry, *, enforced: bool = True) -> FeishuWSClient:
    client = FeishuWSClient.__new__(FeishuWSClient)
    client.settings = SimpleNamespace(
        admin_user_ids=frozenset({OWNER_ID}),
        allowed_user_ids=frozenset(),
        allowed_chat_ids=frozenset(),
        ingress_access_mode="enforced" if enforced else "legacy_allow_all",
        admin_bootstrap_scope="p2p_only",
        thread_programming_enabled=False,
    )
    client._managed_group_registry = registry
    client._managed_group_owner_id = OWNER_ID
    client._employee_department_runtime = MagicMock()
    client._employee_department_runtime.trusted_employee_bot_open_ids.return_value = ()
    client._scheduler = MagicMock()
    client._scheduler.submit.return_value = SimpleNamespace(run_id="run-1")
    client._message_linker = MagicMock()
    client._message_linker.resolve_origin.return_value = None
    client._message_mapper = MagicMock()
    client._message_mapper.get_project_id.return_value = None
    client._project_manager = MagicMock()
    client._project_manager.get_active_project.return_value = None
    client._project_manager.get_project_for_chat.return_value = SimpleNamespace(
        project_id="project-1",
        root_path="/srv/project-1",
    )
    client._thread_manager = MagicMock()
    client._ensure_request_id = MagicMock(return_value="req-managed")
    client._extract_text_from_message = MagicMock(
        side_effect=lambda data: data._task_text,
    )
    client._is_exit_command = MagicMock(return_value=False)
    client._is_spec_command = MagicMock(return_value=False)
    client._build_control_queue_key = MagicMock(return_value=None)
    client._get_image_handler = MagicMock()
    client._get_image_handler.return_value.parse_message.side_effect = (
        lambda _message_type, _content: SimpleNamespace(
            text=client._current_worker_text,
            image_keys=[],
        )
    )
    client._clean_at_text = MagicMock(side_effect=lambda value: value)
    client._validate_message = MagicMock(return_value=True)
    client._chat_lock_gate = MagicMock()
    client._chat_lock_gate.check.return_value = False
    client._resolve_message_context = MagicMock(return_value=(None, None))
    client._dispatch_message_logic = MagicMock()
    client._dispatch_empty_text = MagicMock()
    client._get_api_client = MagicMock()
    client._system_handler = MagicMock()
    client._pending_image_lock = nullcontext()
    client._pending_image_keys = {}
    client._pending_image_only = set()
    return client


def test_registry_is_ready_before_first_ingress_event(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)

    client._handle_message(_message())

    client._scheduler.submit.assert_called_once()
    client._extract_text_from_message.assert_called_once()


def test_managed_group_repeated_tasks_emit_zero_permission_prompts(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)
    client._reply_text = MagicMock()
    client._reply_card = MagicMock()

    client._handle_message(_message(text="first task"))
    client._handle_message(_message(text="second task"))

    assert client._scheduler.submit.call_count == 2
    client._reply_text.assert_not_called()
    client._reply_card.assert_not_called()


@pytest.mark.parametrize("text", ["ls -la", "/access allow-chat", "/setadmin"])
def test_managed_group_cannot_enter_host_shell_or_grant_admin(
    tmp_path,
    text: str,
) -> None:
    client = _client(_registry(tmp_path), enforced=True)

    client._handle_message(_message(text=text))

    client._scheduler.submit.assert_not_called()


def test_owner_p2p_bypasses_legacy_chat_enrollment(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)
    data = _message(text="/status")
    data.event.message.chat_id = "oc_owner_dm"
    data.event.message.chat_type = "p2p"

    client._handle_message(data)

    client._scheduler.submit.assert_called_once()


def test_managed_group_scheduler_uses_registry_project_not_legacy_lookup(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)

    client._handle_message(_message())

    spec = client._scheduler.submit.call_args.args[0]
    assert spec.project_id == "project-1"
    client._message_mapper.get_project_id.assert_not_called()
    client._project_manager.get_active_project.assert_not_called()


@pytest.mark.parametrize("valid_continuation", [False, True])
def test_employee_bot_requires_server_side_outbox_continuation(
    tmp_path,
    valid_continuation: bool,
) -> None:
    employee_id = "ou_employee_bot"
    client = _client(_registry(tmp_path), enforced=True)
    client._employee_department_runtime.trusted_employee_bot_open_ids.return_value = (
        employee_id,
    )
    client._employee_department_runtime.is_valid_employee_continuation.return_value = (
        valid_continuation
    )

    data = _message(sender_id=employee_id)
    data.event.message.parent_id = "om_employee_card"
    client._handle_message(data)

    assert client._scheduler.submit.called is valid_continuation
    client._employee_department_runtime.is_valid_employee_continuation.assert_called_once_with(
        sender_open_id=employee_id,
        chat_id=GROUP_ID,
        message_id="om_employee_card",
    )


def test_employee_continuation_requires_current_channel_and_anchored_outbox() -> None:
    runtime = EmployeeDepartmentRuntime.__new__(EmployeeDepartmentRuntime)
    employee = SimpleNamespace(
        state=EmployeeState.ACTIVE,
        worker_type=WorkerType.VISIBLE,
        agent_id="agent-1",
        tenant_key="tenant-1",
        bot_principal_id="bot-1",
    )
    principal = SimpleNamespace(app_id="app-1")
    runtime._service = MagicMock()
    runtime._service.synchronize_projection.return_value = SimpleNamespace(
        employees={"agent-1": employee},
        bot_principals={"bot-1": principal},
    )
    runtime._channels = MagicMock()
    runtime._channels.status.return_value = SimpleNamespace(
        state=ChannelProcessState.READY,
        agent_id="agent-1",
        tenant_key="tenant-1",
        bot_principal_id="bot-1",
        app_id="app-1",
        generation=7,
        ready_metadata={"connection_id": "conn-1"},
        identity={"app_id": "app-1", "open_id": "ou_employee_bot"},
    )
    binding = SimpleNamespace(
        app_id="app-1",
        generation=7,
        connection_id="conn-1",
        message_id="om_employee_card",
    )
    record = SimpleNamespace(
        outbox_id="outbox-1",
        tenant_key="tenant-1",
        agent_id="agent-1",
        chat_id=GROUP_ID,
        binding=binding,
        latest=SimpleNamespace(state=SimpleNamespace(terminal=True)),
    )
    publication = SimpleNamespace(
        event_type="employee.outbox.collaboration_published",
        aggregate_id="outbox-1",
        payload={
            "tenant_key": "tenant-1",
            "chat_id": GROUP_ID,
            "agent_id": "agent-1",
            "app_id": "app-1",
            "generation": 7,
            "team_run_id": "run-1",
            "assignment_id": "assignment-1",
            "causal_event_id": "event-1",
        },
    )
    runtime._outbox = MagicMock()
    runtime._outbox.state.by_outbox_id = {"outbox-1": record}
    runtime._outbox._writer.replay.return_value = (
        SimpleNamespace(events=(publication,)),
    )

    assert runtime.is_valid_employee_continuation(
        sender_open_id="ou_employee_bot",
        chat_id=GROUP_ID,
        message_id="om_employee_card",
    )

    runtime._channels.status.return_value.generation = 8
    assert not runtime.is_valid_employee_continuation(
        sender_open_id="ou_employee_bot",
        chat_id=GROUP_ID,
        message_id="om_employee_card",
    )


def test_membership_event_requires_exact_registry_and_channel_generation(tmp_path) -> None:
    runtime = EmployeeDepartmentRuntime.__new__(EmployeeDepartmentRuntime)
    runtime._managed_group_registry = _registry(tmp_path)
    runtime._service = MagicMock()
    runtime._service.synchronize_projection.return_value = SimpleNamespace(
        employees={
            "agent-1": SimpleNamespace(
                state=EmployeeState.ACTIVE,
                worker_type=WorkerType.VISIBLE,
                tenant_key="tenant-1",
                agent_id="agent-1",
                bot_principal_id="bot-1",
                member_groups=(GROUP_ID,),
            )
        },
        bot_principals={
            "bot-1": SimpleNamespace(
                tenant_key="tenant-1",
                agent_id="agent-1",
                app_id="app-1",
            )
        },
    )
    runtime._channels = MagicMock()
    runtime._channels.status.return_value = SimpleNamespace(
        state=ChannelProcessState.READY,
        tenant_key="tenant-1",
        agent_id="agent-1",
        bot_principal_id="bot-1",
        app_id="app-1",
        generation=3,
        ready_metadata={"connection_id": "conn-1"},
        identity={"app_id": "app-1"},
    )
    metadata = SimpleNamespace(
        tenant_key="tenant-1",
        agent_id="agent-1",
        bot_principal_id="bot-1",
        app_id="app-1",
        channel_generation=3,
        connection_id="conn-1",
    )

    assert runtime._membership_event_transport_is_current(metadata, GROUP_ID)

    metadata.channel_generation = 2
    assert not runtime._membership_event_transport_is_current(metadata, GROUP_ID)


def test_startup_restores_registry_slock_employee_and_membership_in_order() -> None:
    client = FeishuWSClient.__new__(FeishuWSClient)
    order: list[str] = []
    client._reconcile_managed_groups_before_slock_restore = MagicMock(
        side_effect=lambda: order.append("registry")
    )
    client._managed_group_registry = MagicMock()
    client._managed_group_registry.active_record.return_value = object()
    client._slock_engine_manager = MagicMock()
    client._slock_engine_manager.restore_from_disk.side_effect = (
        lambda *_args, **_kwargs: order.append("slock") or 0
    )
    membership = SimpleNamespace(
        reconcile_projected_memberships=lambda: order.append("membership")
        or SimpleNamespace(removed=0, degraded=0)
    )
    client._employee_department_runtime = SimpleNamespace(
        recover=lambda: order.append("employee"),
        membership_service=membership,
    )
    client._reply_text = MagicMock()

    client._restore_trusted_ingress_dependencies("/srv/project")

    assert order == ["registry", "slock", "employee", "membership"]


def test_dispatcher_denies_external_before_mode_or_intent_lookup() -> None:
    client = MagicMock()
    dispatcher = MessageDispatcher(client)
    external = EffectiveTrust(
        zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
        actor=ActorKind.UNKNOWN,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )

    dispatcher.process_with_intent(
        "om_external",
        "oc_external",
        "implement this",
        effective_trust=external,
    )

    client._get_effective_mode.assert_not_called()
    client._intent_recognizer.recognize.assert_not_called()


@pytest.mark.parametrize(
    "text",
    [
        "/codex implement it",
        "/deep investigate it",
        "/spec define it",
        "/wt isolate it",
        "/wf orchestrate it",
        "/team ship it",
        "/slock coordinate it",
    ],
)
@patch("src.feishu.user_cache.resolve_display_name_nonblocking", return_value="Owner")
def test_existing_routes_bypass_legacy_enrollment_in_managed_group(
    _resolve_name: MagicMock,
    tmp_path,
    text: str,
) -> None:
    client = _client(_registry(tmp_path), enforced=True)
    client._current_worker_text = text

    client._process_message_async(_message(text=text))

    client._dispatch_message_logic.assert_called_once()
    client._system_handler.handle_intercepted_command.assert_not_called()
