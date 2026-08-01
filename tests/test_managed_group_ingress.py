from __future__ import annotations

import hashlib
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.intent_recognizer import IntentType
from src.autonomous.domain import EmployeeState, WorkerType
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.supervisor.employee_channels import ChannelProcessState
from src.feishu.dispatcher import MessageDispatcher
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
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


def test_employee_owner_p2p_status_survives_production_registry_gate(tmp_path) -> None:
    raw_chat_id = "chat-owner-employee-dm"
    raw_message_id = "om-owner-status"
    metadata = SimpleNamespace(
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        thread_root_message_id="",
    )
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "p2p",
                "content": {"text": "/status"},
                "sender_id": OWNER_ID,
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": "",
            },
        )
    )
    record = SimpleNamespace(metadata=metadata, disposition=None)
    ingress = MagicMock()
    ingress.state.by_acceptance_id = {"acceptance-1": record}
    ingress.get_payload.return_value = payload
    runtime = EmployeeDepartmentRuntime(
        managed_group_registry=_registry(tmp_path),
        managed_group_owner_id=OWNER_ID,
    )
    runtime._ingress = ingress
    runtime._router = MagicMock()
    runtime._router.claim_control.return_value = True
    runtime._handle_durable_activation_status = MagicMock(return_value=True)

    assert runtime._handle_control_ingress("acceptance-1") is True

    runtime._router.claim_control.assert_called_once_with(
        "acceptance-1",
        command="/status",
    )
    runtime._handle_durable_activation_status.assert_called_once_with("acceptance-1")
    ingress.record_disposition.assert_not_called()


@pytest.mark.parametrize(
    ("text", "expect_status"),
    [
        ("/status", True),
        ("implement project work", False),
    ],
)
def test_employee_owner_p2p_drain_only_admits_read_only_status(
    tmp_path,
    text: str,
    expect_status: bool,
) -> None:
    raw_chat_id = "chat-owner-employee-dm"
    raw_message_id = "om-owner-control"
    metadata = SimpleNamespace(
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        thread_root_message_id="",
    )
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "p2p",
                "content": {"text": text},
                "sender_id": OWNER_ID,
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": "",
            },
        )
    )
    record = SimpleNamespace(metadata=metadata, disposition=None)
    ingress = MagicMock()
    ingress.state.by_acceptance_id = {"acceptance-1": record}
    ingress.get_payload.return_value = payload
    ingress.gc_terminal_payloads.return_value = 0
    router = MagicMock()
    router.state.by_acceptance_id = {}
    router.claim_control.return_value = True
    dispatch = MagicMock()
    dispatch.employee_runtime = None
    dispatch.dispatch_next.return_value = None
    runtime = EmployeeDepartmentRuntime(
        managed_group_registry=_registry(tmp_path),
        managed_group_owner_id=OWNER_ID,
    )
    runtime._ingress = ingress
    runtime._router = router
    runtime._dispatch = dispatch
    runtime._handle_durable_activation_status = MagicMock(return_value=True)
    runtime._record_employee_ingress_group_event = MagicMock()
    runtime._reconcile_terminal_ingress = MagicMock(return_value=0)
    runtime._drain_employee_outbox_once = MagicMock(return_value=False)
    runtime._outbox = None

    assert runtime._drain_employee_dispatch_once() is True

    if expect_status:
        router.claim_control.assert_called_once_with(
            "acceptance-1",
            command="/status",
        )
        runtime._handle_durable_activation_status.assert_called_once_with(
            "acceptance-1"
        )
        ingress.record_disposition.assert_not_called()
    else:
        router.claim_control.assert_not_called()
        runtime._handle_durable_activation_status.assert_not_called()
        ingress.record_disposition.assert_called_once_with(
            "acceptance-1",
            state="ignored",
            reason_code="authority_denied",
        )
    router.route.assert_not_called()
    runtime._record_employee_ingress_group_event.assert_not_called()


def test_managed_topic_cannot_replace_registry_project(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)
    client.settings.thread_programming_enabled = True
    data = _message()
    data.event.message.root_id = "om_legacy_topic"
    client._thread_manager.get.return_value = SimpleNamespace(
        project_id="attacker-project",
        mode="deep",
        thread_root_id="om_legacy_topic",
    )
    client._current_worker_text = "implement the task"

    client._process_message_async(data)

    client._thread_manager.get.assert_not_called()
    project = client._dispatch_message_logic.call_args.args[3]
    assert project.project_id == "project-1"
    assert project.root_path == "/srv/project-1"


def test_managed_bound_chat_cannot_replace_registry_project(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)
    trusted_project = client._project_manager.get_project_for_chat.return_value
    client._project_manager.find_by_bound_chat_id.return_value = SimpleNamespace(
        project_id="attacker-project",
        root_path="/srv/attacker",
    )
    client._intent_recognizer = MagicMock()
    client._intent_recognizer.looks_like_shell.return_value = False
    client._process_with_intent = MagicMock()
    trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )

    FeishuWSClient._dispatch_message_logic(
        client,
        "om_managed",
        GROUP_ID,
        "implement the task",
        trusted_project,
        None,
        command_match=None,
        effective_trust=trust,
    )

    client._project_manager.find_by_bound_chat_id.assert_not_called()
    assert client._process_with_intent.call_args.args[3] is trusted_project


def test_managed_image_download_uses_registry_root_only(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)
    message = _message().event.message
    trusted_project = client._project_manager.get_project_for_chat.return_value
    client._resolve_message_context.return_value = (
        SimpleNamespace(project_id="attacker-project", root_path="/srv/attacker"),
        "deep",
    )
    client._get_image_handler.return_value.download_images.return_value = SimpleNamespace(
        saved_paths=[],
        failed_keys=[],
    )
    client._get_working_dir = MagicMock(return_value="/srv/default")

    with patch("src.feishu.ws_client.FeishuImageHandler.get_image_save_dir") as save_dir:
        client._handle_image_content(
            message,
            ["img-1"],
            "implement the task",
            "req-managed",
            None,
            trusted_project=trusted_project,
        )

    client._resolve_message_context.assert_not_called()
    save_dir.assert_called_once_with("/srv/project-1", client._get_working_dir(GROUP_ID))


def test_current_trust_rejects_project_mismatch(tmp_path) -> None:
    client = _client(_registry(tmp_path), enforced=True)
    trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )

    assert client._current_trust_can_dispatch(
        trust,
        project=SimpleNamespace(
            project_id="attacker-project",
            root_path="/srv/attacker",
        ),
    ) is False


def test_recognizer_rotation_fences_single_executor(tmp_path) -> None:
    registry = _registry(tmp_path)
    client = _client(registry, enforced=True)
    project = client._project_manager.get_project_for_chat.return_value
    trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    task = SimpleNamespace(intent=IntentType.COCO_MESSAGE, data={}, description="work")
    result = SimpleNamespace(
        primary_intent=IntentType.COCO_MESSAGE,
        confidence=1.0,
        tasks=[task],
        is_multi_task=False,
    )

    def recognize(*_args):
        registry.rotate_receiving_bot(
            chat_id=GROUP_ID,
            expected_bot_ref="cli_main_bot",
            new_bot_ref="cli_rotated_bot",
        )
        return result

    client._get_effective_mode = MagicMock(return_value=(InteractionMode.SMART, False))
    client._is_deep_command = MagicMock(return_value=False)
    client._is_spec_command = MagicMock(return_value=False)
    client._is_workflow_command = MagicMock(return_value=False)
    client._is_topic_engine_context = MagicMock(return_value=False)
    client._is_slock_command = MagicMock(return_value=False)
    client._is_slock_managed_chat = MagicMock(return_value=False)
    client._is_slock_active = MagicMock(return_value=False)
    client._is_interceptable_command_match = MagicMock(return_value=False)
    client._is_worktree_awaiting_goal = MagicMock(return_value=False)
    client._intent_recognizer = MagicMock()
    client._intent_recognizer.recognize.side_effect = recognize
    client._handle_coco_message = MagicMock()
    client._add_reaction = MagicMock()
    client.settings.slock_passive_mode = False

    MessageDispatcher(client).process_with_intent(
        "om_managed",
        GROUP_ID,
        "implement the task",
        project,
        command_match=None,
        effective_trust=trust,
    )

    client._handle_coco_message.assert_not_called()


def _slock_dispatch_client(
    registry: ManagedGroupRegistry,
) -> tuple[FeishuWSClient, object, EffectiveTrust]:
    client = _client(registry, enforced=True)
    project = client._project_manager.get_project_for_chat.return_value
    trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    client._get_effective_mode = MagicMock(
        return_value=(InteractionMode.SMART, False)
    )
    client._is_deep_command = MagicMock(return_value=False)
    client._is_spec_command = MagicMock(return_value=False)
    client._is_workflow_command = MagicMock(return_value=False)
    client._is_topic_engine_context = MagicMock(return_value=False)
    client._is_interceptable_command_match = MagicMock(return_value=False)
    client._is_worktree_awaiting_goal = MagicMock(return_value=False)
    client._is_exit_command = MagicMock(return_value=False)
    client._add_reaction = MagicMock()
    client._handle_slock_command = MagicMock()
    client._handle_slock_message = MagicMock()
    return client, project, trust


def test_explicit_slock_rechecks_trust_after_command_detection(tmp_path) -> None:
    registry = _registry(tmp_path)
    client, project, trust = _slock_dispatch_client(registry)

    def detect_and_rotate(*_args):
        registry.rotate_receiving_bot(
            chat_id=GROUP_ID,
            expected_bot_ref="cli_main_bot",
            new_bot_ref="cli_rotated_bot",
        )
        return True

    client._is_slock_command = MagicMock(side_effect=detect_and_rotate)

    MessageDispatcher(client).process_with_intent(
        "om_managed",
        GROUP_ID,
        "/slock status",
        project,
        effective_trust=trust,
    )

    client._handle_slock_command.assert_not_called()
    client._handle_slock_message.assert_not_called()


@pytest.mark.parametrize("passive_mode", [True, False], ids=["passive", "legacy"])
def test_slock_classifier_rotation_fences_real_message_handler(
    tmp_path,
    passive_mode: bool,
) -> None:
    registry = _registry(tmp_path)
    client, project, trust = _slock_dispatch_client(registry)
    client.settings.slock_passive_mode = passive_mode
    client._is_slock_command = MagicMock(return_value=False)
    client._is_slock_managed_chat = MagicMock(return_value=True)
    client._is_slock_active = MagicMock(return_value=True)

    def classify_and_rotate(*_args, **_kwargs):
        registry.rotate_receiving_bot(
            chat_id=GROUP_ID,
            expected_bot_ref="cli_main_bot",
            new_bot_ref="cli_rotated_bot",
        )
        return False, 1.0

    with patch(
        "src.slock_engine.task_classifier.TaskClassifier.classify",
        side_effect=classify_and_rotate,
    ):
        MessageDispatcher(client).process_with_intent(
            "om_managed",
            GROUP_ID,
            "Please coordinate the release plan",
            project,
            command_match=None,
            effective_trust=trust,
        )

    client._handle_slock_message.assert_not_called()
    client._handle_slock_command.assert_not_called()


def test_multi_task_rechecks_trust_before_every_step(tmp_path) -> None:
    registry = _registry(tmp_path)
    client = _client(registry, enforced=True)
    project = client._project_manager.get_project_for_chat.return_value
    trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    dispatcher = MessageDispatcher(client)
    dispatcher.execute_task_step = MagicMock()

    def first_step(*_args, **_kwargs):
        registry.rotate_receiving_bot(
            chat_id=GROUP_ID,
            expected_bot_ref="cli_main_bot",
            new_bot_ref="cli_rotated_bot",
        )
        return True

    dispatcher.execute_task_step.side_effect = first_step
    tasks = [
        SimpleNamespace(intent=IntentType.SHOW_HELP, data={}, description="one"),
        SimpleNamespace(intent=IntentType.SHOW_HELP, data={}, description="two"),
    ]
    client._reply_text = MagicMock()
    client._add_reaction = MagicMock()

    dispatcher.execute_multi_tasks(
        "om_managed",
        GROUP_ID,
        SimpleNamespace(tasks=tasks),
        project,
        effective_trust=trust,
    )

    dispatcher.execute_task_step.assert_called_once()


@pytest.mark.parametrize("mode", ["worktree", "deep", "spec", "workflow", "coco"])
def test_stale_trust_fences_direct_engine_and_programming_dispatch(
    tmp_path,
    mode: str,
) -> None:
    registry = _registry(tmp_path)
    client = _client(registry, enforced=True)
    project = client._project_manager.get_project_for_chat.return_value
    trust = client._resolve_effective_trust(
        sender_id=OWNER_ID,
        chat_id=GROUP_ID,
        chat_type="group",
    )
    registry.rotate_receiving_bot(
        chat_id=GROUP_ID,
        expected_bot_ref="cli_main_bot",
        new_bot_ref="cli_rotated_bot",
    )
    client._reply_if_topic_engine_switch_blocked = MagicMock(return_value=False)
    client._is_interceptable_command_match = MagicMock(return_value=False)
    client._is_programming_entry_command = MagicMock(return_value=False)
    client._is_deep_command = MagicMock(return_value=False)
    client._is_spec_command = MagicMock(return_value=False)
    client._is_workflow_command = MagicMock(return_value=False)
    client._add_reaction = MagicMock()
    client._handle_worktree_execute = MagicMock()
    client._start_deep_engine = MagicMock()
    client._start_spec_engine = MagicMock()
    client._workflow_handler = MagicMock()
    handler = MagicMock()
    client._get_mode_handler = MagicMock(return_value=handler)

    client._dispatch_message_logic(
        "om_managed",
        GROUP_ID,
        "implement the task",
        project,
        mode,
        command_match=None,
        effective_trust=trust,
    )

    client._handle_worktree_execute.assert_not_called()
    client._start_deep_engine.assert_not_called()
    client._start_spec_engine.assert_not_called()
    client._workflow_handler.handle_message.assert_not_called()
    handler.handle_message.assert_not_called()


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
