import asyncio
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.access_control import IngressAccessPolicy
from src.agent.intent_recognizer import IntentResult, IntentType, TaskStep
from src.card.ui_text import UI_TEXT
from src.config import IngressAccessMode
from src.feishu.image_handler import FeishuImageHandler, ImageDownloadResult
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.project import ProjectContext
from src.tasking import TaskPriority, TaskQueueFullError, TaskScheduler, TaskSpec, TaskStatus
from src.thread import get_current_thread_id, set_current_thread_id
from src.trust.models import ActorKind, EffectiveTrust, TrustZone


@pytest.fixture
def mock_ws_client():
    """Build the real routing graph without external sessions or worker threads."""
    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()

    with (
        patch("src.feishu.ws_client.get_settings") as mock_get_settings,
        patch("src.feishu.ws_client._build_task_scheduler", return_value=scheduler),
        patch("src.feishu.ws_client.ACPSessionManager"),
        patch("src.feishu.ws_client.IntentRecognizer"),
        patch("src.feishu.ws_client.ProjectManager"),
        patch("src.feishu.ws_client.MessageProjectMapper"),
        patch("src.feishu.ws_client.DeepEngineManager"),
        patch("src.feishu.ws_client.ProgressReporter"),
        patch("src.feishu.ws_client.SpecEngineManager"),
        patch("src.feishu.ws_client.SpecReporter"),
        patch("src.mode.ModeManager"),
        patch("src.workflow_engine.manager.WorkflowEngineManager"),
    ):
        settings = MagicMock()
        settings.app_id = "test_app_id"
        settings.app_secret = "test_app_secret"
        settings.admin_user_ids = ""
        settings.allowed_user_ids = ""
        settings.allowed_chat_ids = ""
        settings.ingress_access_mode = "legacy_allow_all"
        settings.admin_bootstrap_scope = "p2p_only"
        settings.streaming_enabled = False
        settings.task_scheduler_max_concurrent = 2
        settings.task_scheduler_per_key_concurrency = 1
        settings.system_command_concurrency = 10
        settings.message_cache_ttl = 300
        settings.message_cache_max_size = 1000
        settings.message_expire_seconds = 30
        settings.card.action_dedup_ttl = 1
        settings.card.action_dedup_max_size = 5000
        settings.spec_rate_limit_capacity = 100
        settings.spec_rate_limit_fill_rate = 50.0
        settings.spec_circuit_breaker_threshold = 10
        settings.spec_circuit_breaker_recovery = 5.0
        mock_get_settings.return_value = settings

        client = FeishuWSClient(MagicMock())
        client._main_bot_open_id = "ou_main_bot"
        client._project_manager.get_active_project.return_value = None
        try:
            yield client
        finally:
            client.close()




def create_mock_message(text: str, message_id="om_123", chat_id="oc_456", message_type="text"):
    data = MagicMock()
    data.header.tenant_key = "tenant_test"
    data.event.message.message_id = message_id
    data.event.message.chat_id = chat_id
    data.event.message.content = json.dumps({"text": text})
    data.event.message.message_type = message_type
    data.event.message.create_time = str(int(time.time() * 1000))
    # Reset parent/root
    data.event.message.parent_id = None
    data.event.message.root_id = None
    data.event.message.thread_id = None
    data.event.message.chat_type = "group"
    data.event.sender.sender_id.open_id = "ou_test"
    data.event.sender.sender_id.union_id = "on_test"
    return data


def _sdk_message_mention(
    *,
    key: str,
    open_id: str,
    name: str,
    tenant_key: str = "tenant_test",
):
    """Build the exact mention shape emitted by lark-channel-sdk 1.2.0."""

    from lark_channel.api.im.v1.model.mention_event import MentionEvent
    from lark_channel.api.im.v1.model.user_id import UserId

    return (
        MentionEvent.builder()
        .key(key)
        .id(UserId.builder().open_id(open_id).build())
        .name(name)
        .tenant_key(tenant_key)
        .build()
    )


def _mentioned_group_command(
    text: str,
    *mentions,
    message_id: str = "om_mentioned_command",
):
    message = create_mock_message(text, message_id=message_id)
    message.event.message.mentions = tuple(mentions)
    return message


def _pending_task_handle(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(run_id=run_id, add_done_callback=MagicMock())


def test_ordinary_message_reserves_dedup_before_worker_starts(
    mock_ws_client: FeishuWSClient,
) -> None:
    event = create_mock_message(
        "hello before worker",
        message_id="om_ordinary_admitted_not_started",
    )

    mock_ws_client._handle_message(event)
    mock_ws_client._handle_message(event)

    assert mock_ws_client._scheduler.submit.call_count == 1


@pytest.mark.parametrize(
    "admission_error",
    (
        RuntimeError("TaskScheduler admission is fenced"),
        ValueError("TaskScheduler rejected an invalid spec"),
    ),
    ids=("runtime", "invalid-spec"),
)
def test_scheduler_error_releases_its_message_reservation(
    mock_ws_client: FeishuWSClient,
    admission_error: Exception,
) -> None:
    event = create_mock_message(
        "hello after scheduler restart",
        message_id="om_scheduler_runtime_error",
    )
    mock_ws_client._scheduler.submit.side_effect = admission_error

    with pytest.raises(type(admission_error), match="TaskScheduler"):
        mock_ws_client._handle_message(event)

    mock_ws_client._scheduler.submit.side_effect = None
    mock_ws_client._scheduler.submit.return_value = _pending_task_handle(
        "run_after_restart"
    )
    mock_ws_client._handle_message(event)

    assert mock_ws_client._scheduler.submit.call_count == 2


def test_worker_finally_commits_the_original_reservation_after_event_drift(
    mock_ws_client: FeishuWSClient,
) -> None:
    original_message_id = "om_worker_event_drift"
    owner = mock_ws_client._message_ingress_guard.reserve(original_message_id)
    assert owner is not None
    event = create_mock_message(
        "mutated after admission",
        message_id=original_message_id,
    )
    event.event.message.message_id = ""

    mock_ws_client._process_message_async(
        event,
        message_reservation_id=original_message_id,
        message_reservation_owner=owner,
    )

    assert not mock_ws_client._message_ingress_guard.owns(
        original_message_id,
        owner,
    )
    assert mock_ws_client._message_ingress_guard.reserve(original_message_id) is None


@pytest.mark.parametrize(
    "reply_error",
    (RuntimeError("reply transport failed"), KeyError("reply contract failed")),
    ids=("transport", "unexpected"),
)
def test_ordinary_backpressure_reply_exception_releases_message_reservation(
    mock_ws_client: FeishuWSClient,
    reply_error: Exception,
) -> None:
    message_id = "om_backpressure_reply_raised"
    mock_ws_client._scheduler.submit.side_effect = TaskQueueFullError("normal", 1)
    mock_ws_client._handler_ctx.handlers["coco"].reply_text = MagicMock(
        side_effect=reply_error
    )

    with pytest.raises(type(reply_error)):
        mock_ws_client._handle_message(
            create_mock_message("hello", message_id=message_id)
        )

    retry_owner = mock_ws_client._message_ingress_guard.reserve(message_id)
    assert retry_owner is not None
    assert mock_ws_client._message_ingress_guard.release(message_id, retry_owner)


def test_scheduler_guard_failure_durably_warns_and_fences_message_origin(
    mock_ws_client: FeishuWSClient,
) -> None:
    @contextmanager
    def broken_guard():
        raise RuntimeError("restart gate admission failed")
        yield

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        system_concurrency=1,
        max_pending_normal=2,
        max_pending_system=2,
        max_terminal_history=10,
        run_guard=broken_guard,
    )
    previous_scheduler = mock_ws_client._scheduler
    mock_ws_client._scheduler = scheduler
    reply_warning = MagicMock(return_value=True)
    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    message_id = "om_restart_guard_failed_before_callback"
    try:
        mock_ws_client._handle_message(
            create_mock_message("hello", message_id=message_id)
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            states = scheduler.list_tasks(
                chat_id="oc_456",
                include_done=True,
                limit=10,
            )
            if states and states[0].status is TaskStatus.FAILED:
                break
            time.sleep(0.01)
        assert states and states[0].status is TaskStatus.FAILED

        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and not reply_warning.called
        ):
            time.sleep(0.01)
        reply_warning.assert_called_once_with(
            message_id,
            UI_TEXT["ws_message_prestart_terminal"],
            label="pre-start terminal warning",
        )
        assert mock_ws_client._message_ingress_guard.reserve(message_id) is None
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)
        mock_ws_client._scheduler = previous_scheduler


def test_scheduler_queued_cancel_durably_warns_and_fences_message_origin(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        system_concurrency=1,
        max_pending_normal=2,
        max_pending_system=2,
        max_terminal_history=10,
    )
    previous_scheduler = mock_ws_client._scheduler
    mock_ws_client._scheduler = scheduler
    reply_warning = MagicMock(return_value=True)
    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    scheduler.submit(
        TaskSpec(chat_id="oc_456", name="blocker"),
        lambda _ctx: (blocker_started.set(), release_blocker.wait(timeout=2)),
    )
    message_id = "om_queued_message_canceled"
    try:
        assert blocker_started.wait(timeout=1)
        mock_ws_client._handle_message(
            create_mock_message("hello", message_id=message_id)
        )
        queued = next(
            state
            for state in scheduler.list_tasks(
                chat_id="oc_456",
                include_done=False,
                limit=10,
            )
            if state.spec.message_id == message_id
        )
        assert scheduler.cancel(queued.run_id)

        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and not reply_warning.called
        ):
            time.sleep(0.01)
        reply_warning.assert_called_once_with(
            message_id,
            UI_TEXT["ws_message_prestart_terminal"],
            label="pre-start terminal warning",
        )
        assert mock_ws_client._message_ingress_guard.reserve(message_id) is None
    finally:
        release_blocker.set()
        scheduler.wait_for_idle(timeout=2)
        scheduler.stop(wait=True, shutdown_executor=True)
        mock_ws_client._scheduler = previous_scheduler


def test_terminal_before_reservation_bind_replays_after_scheduler_history_reap() -> None:
    from src.feishu.message_cache import MessageCache
    from src.feishu.ws_client import _MessageIngressReservation
    from src.feishu.ws_event_router import MessageIngressGuard

    @contextmanager
    def broken_guard():
        raise RuntimeError("restart gate admission failed before callback")
        yield

    scheduler = TaskScheduler(
        max_concurrent=1,
        per_key_concurrency=1,
        system_concurrency=1,
        max_pending_normal=2,
        max_pending_system=2,
        max_terminal_history=10,
        run_guard=broken_guard,
    )
    client = object.__new__(FeishuWSClient)
    client._scheduler = scheduler
    client._reply_text_with_bounded_retry = MagicMock(return_value=True)
    guard = MessageIngressGuard(
        message_cache=MessageCache(ttl=300, max_size=10),
        message_expire_seconds=30,
    )
    message_id = "om_terminal_before_reservation_bind"
    owner = guard.reserve(message_id)
    assert owner is not None
    reservation = _MessageIngressReservation(
        guard=guard,
        message_id=message_id,
        owner=owner,
    )
    callback_called = threading.Event()

    try:
        handle = scheduler.submit(
            TaskSpec(chat_id="replay-chat", name="never-started"),
            lambda _ctx: callback_called.set(),
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = scheduler.get_state(handle.run_id)
            if state is not None and state.status is TaskStatus.FAILED:
                break
            time.sleep(0.01)
        assert state is not None and state.status is TaskStatus.FAILED
        assert scheduler.wait_for_idle(timeout=2)
        with scheduler._cv:
            assert scheduler._reap_completed_states(max_age_seconds=0) == 1
        assert scheduler.get_state(handle.run_id) is None

        client._bind_message_ingress_reservation(
            handle,
            reservation,
            tenant_key="tenant-replay",
            chat_id="replay-chat",
        )

        assert guard.reserve(message_id) is None
        assert not callback_called.is_set()
    finally:
        scheduler.stop(wait=True, shutdown_executor=True)


def test_reservation_completion_callback_contains_cleanup_failure() -> None:
    class DeferredHandle:
        run_id = "run_completion_callback_failure"

        def __init__(self) -> None:
            self.callback = None

        def add_done_callback(self, callback) -> None:
            self.callback = callback

    client = object.__new__(FeishuWSClient)
    client._scheduler = MagicMock()
    client._reply_text_with_bounded_retry = MagicMock(return_value=True)
    reservation = MagicMock()
    reservation.owns.return_value = True
    reservation.claim_unstarted_terminal.return_value = True
    reservation.message_id = "om_completion_callback_failure"
    reservation.commit.side_effect = RuntimeError("reservation cleanup failed")
    handle = DeferredHandle()

    client._bind_message_ingress_reservation(
        handle,
        reservation,
        tenant_key="tenant-test",
        chat_id="oc-test",
    )

    assert handle.callback is not None
    handle.callback(
        SimpleNamespace(
            run_id=handle.run_id,
            status=TaskStatus.FAILED,
        )
    )
    reservation.commit.assert_called_once_with()
    client._scheduler.get_state.assert_not_called()


@pytest.mark.parametrize(
    ("command", "message_id"),
    (
        ("/stop_wf", "om_main_stop_wf"),
        ("/stop_workflow", "om_main_stop_workflow"),
    ),
)
def test_main_bot_mention_keeps_control_project_route_and_origin(
    mock_ws_client: FeishuWSClient,
    command: str,
    message_id: str,
) -> None:
    project = ProjectContext("project_main", "Main", "/tmp/main")
    mock_ws_client._project_manager.get_active_project.return_value = project
    mention = _sdk_message_mention(
        key="@_user_1",
        open_id="ou_main_bot",
        name="GhostAP",
    )
    event = _mentioned_group_command(
        f"@_user_1 {command}",
        mention,
        message_id=message_id,
    )
    mock_ws_client._validate_message = MagicMock(return_value=True)
    mock_ws_client._dispatch_message_logic = MagicMock()

    mock_ws_client._handle_message(event)

    spec, callback = mock_ws_client._scheduler.submit.call_args.args
    assert spec.task_type == "feishu_message"
    assert spec.is_system_command is True
    assert spec.project_id == "project_main"
    assert spec.queue_key == "oc_456:control:project_main"
    assert spec.origin_message_id == message_id
    assert mock_ws_client._message_linker.query(message_id)["project_id"] == (
        "project_main"
    )
    callback(MagicMock())
    dispatch = mock_ws_client._dispatch_message_logic.call_args
    assert dispatch.args[2] == command
    assert dispatch.kwargs["command_match"].command == "/stop_wf"


def test_main_bot_mention_uses_managed_trust_without_static_chat_allowlist(
    mock_ws_client: FeishuWSClient,
) -> None:
    owner_id = "ou_managed_owner"
    mock_ws_client._managed_group_owner_id = owner_id
    mock_ws_client._ingress_access_policy_provider.swap(
        IngressAccessPolicy(
            admin_ids=frozenset({owner_id}),
            allowed_user_ids=frozenset({owner_id}),
            allowed_chat_ids=frozenset(),
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope="p2p_only",
        )
    )
    mock_ws_client._resolve_effective_trust = MagicMock(
        return_value=SimpleNamespace(
            zone=TrustZone.MANAGED_AGENT_GROUP,
            actor=ActorKind.OWNER,
            managed_group=SimpleNamespace(
                chat_id="oc_456",
                project_id="project_managed",
            ),
        )
    )
    mock_ws_client._decide_ingress_access = MagicMock()
    event = _mentioned_group_command(
        "@_user_1 /stop_wf",
        _sdk_message_mention(
            key="@_user_1",
            open_id="ou_main_bot",
            name="GhostAP",
        ),
        message_id="om_main_managed_without_static_chat",
    )
    event.event.sender.sender_id.open_id = owner_id

    mock_ws_client._handle_message(event)

    mock_ws_client._resolve_effective_trust.assert_called_once()
    mock_ws_client._decide_ingress_access.assert_not_called()
    spec, _callback = mock_ws_client._scheduler.submit.call_args.args
    assert spec.task_type == "feishu_message"
    assert spec.project_id == "project_managed"
    assert spec.queue_key == "oc_456:control:project_managed"


def test_unknown_main_bot_identity_fails_closed_before_mentioned_command_admission(
    mock_ws_client: FeishuWSClient,
) -> None:
    mock_ws_client._main_bot_open_id = ""
    mock_ws_client._sync_main_bot_identity = MagicMock(return_value="")
    event = _mentioned_group_command(
        "@_user_1 /stop_wf",
        _sdk_message_mention(
            key="@_user_1",
            open_id="ou_main_bot",
            name="GhostAP",
        ),
        message_id="om_main_identity_startup_race",
    )
    mock_ws_client._validate_message = MagicMock(return_value=True)
    mock_ws_client._dispatch_message_logic = MagicMock()

    mock_ws_client._handle_message(event)

    mock_ws_client._scheduler.submit.assert_not_called()
    mock_ws_client._sync_main_bot_identity.assert_not_called()
    mock_ws_client._dispatch_message_logic.assert_not_called()


def test_main_bot_identity_lookup_validates_app_and_retries_after_transient_failure(
    mock_ws_client: FeishuWSClient,
) -> None:
    from lark_oapi.core.model.base_response import BaseResponse
    from lark_oapi.core.model.raw_response import RawResponse

    def response(payload: object, *, code: int = 0) -> BaseResponse:
        result = BaseResponse()
        result.code = code
        result.raw = RawResponse()
        result.raw.status_code = 200
        result.raw.content = json.dumps(payload).encode()
        return result

    request = MagicMock(
        side_effect=(
            OSError("temporary lookup failure"),
            response(
                {
                    "code": 0,
                    "bot": {
                        "app_id": "test_app_id",
                        "open_id": "ou_main_bot_refreshed",
                    },
                }
            ),
        )
    )
    mock_ws_client._api_client = SimpleNamespace(request=request)
    mock_ws_client._main_bot_open_id = ""

    assert mock_ws_client._sync_main_bot_identity() == ""
    mock_ws_client._main_bot_identity_next_retry_at = 0
    assert mock_ws_client._sync_main_bot_identity() == "ou_main_bot_refreshed"
    assert request.call_count == 2


def test_main_bot_identity_accepts_official_response_without_app_id(
    mock_ws_client: FeishuWSClient,
) -> None:
    """The tenant-token Bot info response does not expose an app_id field."""

    from lark_oapi.core.model.base_response import BaseResponse
    from lark_oapi.core.model.raw_response import RawResponse

    response = BaseResponse()
    response.code = 0
    response.raw = RawResponse()
    response.raw.status_code = 200
    response.raw.content = json.dumps(
        {
            "code": 0,
            "msg": "success",
            "bot": {
                "activate_status": 2,
                "app_name": "GhostAP",
                "avatar_url": "https://example.invalid/avatar.png",
                "ip_white_list": [],
                "open_id": "ou_main_bot_official_shape",
            },
        }
    ).encode()
    mock_ws_client._api_client = SimpleNamespace(
        request=MagicMock(return_value=response)
    )
    mock_ws_client._main_bot_open_id = ""

    assert (
        mock_ws_client._sync_main_bot_identity()
        == "ou_main_bot_official_shape"
    )
    assert mock_ws_client._main_bot_open_id == "ou_main_bot_official_shape"


def test_main_bot_identity_lookup_rejects_a_different_app_binding(
    mock_ws_client: FeishuWSClient,
) -> None:
    from lark_oapi.core.model.base_response import BaseResponse
    from lark_oapi.core.model.raw_response import RawResponse

    response = BaseResponse()
    response.code = 0
    response.raw = RawResponse()
    response.raw.status_code = 200
    response.raw.content = json.dumps(
        {
            "code": 0,
            "bot": {
                "app_id": "cli_other_app",
                "open_id": "ou_wrong_main_bot",
            },
        }
    ).encode()
    mock_ws_client._api_client = SimpleNamespace(
        request=MagicMock(return_value=response)
    )
    mock_ws_client._main_bot_open_id = ""

    assert mock_ws_client._sync_main_bot_identity() == ""
    assert mock_ws_client._main_bot_open_id == ""


def test_start_does_not_open_ws_intake_without_main_bot_identity(
    mock_ws_client: FeishuWSClient,
) -> None:
    mock_ws_client._main_bot_open_id = ""
    mock_ws_client.settings.feishu_ws_reconnect_delay_s = 0.0

    def identity_unavailable() -> str:
        mock_ws_client._closed = True
        return ""

    mock_ws_client._sync_main_bot_identity = MagicMock(
        side_effect=identity_unavailable
    )
    with (
        patch.object(mock_ws_client, "_publish_restart_participation"),
        patch.object(mock_ws_client, "_build_event_handler", return_value=object()),
        patch.object(mock_ws_client._message_cache, "start_cleanup_thread"),
        patch.object(mock_ws_client._card_event_cache, "start_cleanup_thread"),
        patch.object(mock_ws_client._ws_health_monitor, "start_watchdog"),
        patch.object(mock_ws_client, "_start_main_slash_command_sync"),
        patch.object(mock_ws_client, "_restore_trusted_ingress_dependencies"),
        patch("src.feishu.ws_client.ObservedLarkWSClient") as observed,
    ):
        observed.return_value.start.side_effect = lambda: setattr(
            mock_ws_client,
            "_closed",
            True,
        )
        mock_ws_client.start()

    mock_ws_client._sync_main_bot_identity.assert_called_once_with()
    observed.assert_not_called()


@pytest.mark.parametrize(
    "event",
    [
        create_mock_message(
            "/task unaddressed group task",
            message_id="om_unaddressed_task",
        ),
        _mentioned_group_command(
            "@_user_1 /task ambiguous target @_user_2",
            _sdk_message_mention(
                key="@_user_1",
                open_id="ou_employee_alpha",
                name="Employee Alpha",
            ),
            _sdk_message_mention(
                key="@_user_2",
                open_id="ou_employee_beta",
                name="Employee Beta",
            ),
            message_id="om_multiple_targets",
        ),
        _mentioned_group_command(
            "@_user_2 /task mismatched placeholder",
            _sdk_message_mention(
                key="@_user_1",
                open_id="ou_employee_alpha",
                name="Employee",
            ),
            message_id="om_mismatched_placeholder",
        ),
        _mentioned_group_command(
            "@_user_1 /stop_wf @_user_2",
            _sdk_message_mention(
                key="@_user_1",
                open_id="ou_employee_alpha",
                name="Employee",
            ),
            message_id="om_unbound_extra_placeholder",
        ),
        _mentioned_group_command(
            "@_user_1 ordinary conversation",
            _sdk_message_mention(
                key="@_user_1",
                open_id="ou_employee_alpha",
                name="Employee",
            ),
            message_id="om_non_command",
        ),
        _mentioned_group_command(
            "@_user_1 /task foreign tenant",
            _sdk_message_mention(
                key="@_user_1",
                open_id="ou_employee_alpha",
                name="Foreign Employee",
                tenant_key="tenant_other",
            ),
            message_id="om_foreign_tenant",
        ),
    ],
    ids=(
        "bare-task",
        "multiple",
        "placeholder-mismatch",
        "unbound-placeholder",
        "non-command",
        "foreign-tenant",
    ),
)
def test_unproven_group_task_target_uses_the_ordinary_main_route(
    mock_ws_client: FeishuWSClient,
    event,
) -> None:
    process_main_route = MagicMock()
    mock_ws_client._process_message_async = process_main_route

    mock_ws_client._handle_message(event)

    spec, callback = mock_ws_client._scheduler.submit.call_args.args
    assert spec.task_type == "feishu_message"
    assert spec.name == "process_message"
    callback(SimpleNamespace(run_id="run_main_route"))

    process_main_route.assert_called_once()


def test_uniquely_mentioned_other_bot_command_is_ignored_without_main_execution(
    mock_ws_client: FeishuWSClient,
) -> None:
    reply_text = MagicMock()
    mock_ws_client._handler_ctx.handlers["coco"].reply_text = reply_text
    mock_ws_client._dispatch_message_logic = MagicMock()
    event = _mentioned_group_command(
        "@_user_1 /stop_wf",
        _sdk_message_mention(
            key="@_user_1",
            open_id="ou_other_bot",
            name="Other Bot",
        ),
        message_id="om_other_bot_target",
    )

    mock_ws_client._handle_message(event)

    mock_ws_client._scheduler.submit.assert_not_called()
    reply_text.assert_not_called()
    mock_ws_client._dispatch_message_logic.assert_not_called()


def test_handle_message_system_command_routing(mock_ws_client: FeishuWSClient):
    """Test that system commands (like /help) bypass project queue and get HIGH priority."""
    msg = create_mock_message("/help")

    mock_ws_client._handle_message(msg)

    submit_mock = mock_ws_client._scheduler.submit
    assert submit_mock.call_count == 1
    spec, _ = submit_mock.call_args[0]

    assert spec.task_type == "system_help"
    assert spec.priority == TaskPriority.HIGH
    assert spec.is_system_command is True
    assert spec.tenant_key == "tenant_test"
    # System commands should not block behind regular project tasks (often goes to control queue or no strict project queue)


def test_default_open_ingress_routes_dsh_from_unknown_project_group(
    mock_ws_client: FeishuWSClient,
) -> None:
    """Open mode must cross every trust gate before /dsh dispatch."""
    unknown_group = EffectiveTrust(
        zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
        actor=ActorKind.UNKNOWN,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(
        return_value=unknown_group
    )
    system = mock_ws_client._handler_ctx.handlers["system"]
    system.handle_intercepted_command = MagicMock()
    event = create_mock_message(
        "/dsh",
        message_id="om_open_project_dsh",
        chat_id="oc_open_project",
    )

    mock_ws_client._handle_message(event)

    spec, callback = mock_ws_client._scheduler.submit.call_args.args
    assert spec.task_type == "feishu_message"
    assert spec.is_system_command is True
    callback(SimpleNamespace(run_id="run_open_project_dsh"))
    system.handle_intercepted_command.assert_called_once()
    assert system.handle_intercepted_command.call_args.args[:3] == (
        "om_open_project_dsh",
        "oc_open_project",
        "/dsh",
    )
    assert system.handle_intercepted_command.call_args.kwargs[
        "command_match"
    ].command == "/dsh"


def test_explicit_enforced_ingress_still_denies_unknown_project_group_dsh(
    mock_ws_client: FeishuWSClient,
) -> None:
    owner_id = "ou_owner"
    mock_ws_client._ingress_access_policy_provider.swap(
        IngressAccessPolicy(
            admin_ids=frozenset({owner_id}),
            allowed_user_ids=frozenset({owner_id}),
            allowed_chat_ids=frozenset({"oc_open_project"}),
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope="p2p_only",
        )
    )
    mock_ws_client._resolve_effective_trust = MagicMock(
        return_value=EffectiveTrust(
            zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
            actor=ActorKind.UNKNOWN,
            managed_group=None,
            group_revision=None,
            grant_revision=None,
        )
    )
    event = create_mock_message(
        "/dsh",
        message_id="om_enforced_project_dsh",
        chat_id="oc_open_project",
    )
    event.event.sender.sender_id.open_id = owner_id

    mock_ws_client._handle_message(event)

    mock_ws_client._scheduler.submit.assert_not_called()


def test_unconfigured_enforced_private_help_uses_narrow_bootstrap_route(
    mock_ws_client: FeishuWSClient,
) -> None:
    mock_ws_client._ingress_access_policy_provider.swap(
        IngressAccessPolicy(
            admin_ids=frozenset(),
            allowed_user_ids=frozenset(),
            allowed_chat_ids=frozenset(),
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope="p2p_only",
        )
    )
    event = create_mock_message(
        "/help",
        message_id="om_bootstrap_help",
        chat_id="oc_bootstrap_private",
    )
    event.event.message.chat_type = "p2p"
    unknown_private_trust = EffectiveTrust(
        zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
        actor=ActorKind.UNKNOWN,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(
        return_value=unknown_private_trust
    )
    system = mock_ws_client._handler_ctx.handlers["system"]
    system.show_bootstrap_help = MagicMock(return_value="om_help_reply")

    mock_ws_client._handle_message(event)

    spec, callback = mock_ws_client._scheduler.submit.call_args.args
    assert spec.task_type == "system_help"
    callback(SimpleNamespace(run_id="run_bootstrap_help"))
    system.show_bootstrap_help.assert_called_once_with(
        "om_bootstrap_help",
        "oc_bootstrap_private",
    )


def test_unconfigured_enforced_private_setadmin_uses_narrow_bootstrap_route(
    mock_ws_client: FeishuWSClient,
) -> None:
    mock_ws_client._ingress_access_policy_provider.swap(
        IngressAccessPolicy(
            admin_ids=frozenset(),
            allowed_user_ids=frozenset(),
            allowed_chat_ids=frozenset(),
            mode=IngressAccessMode.ENFORCED,
            admin_bootstrap_scope="p2p_only",
        )
    )
    event = create_mock_message(
        "/setadmin",
        message_id="om_bootstrap_admin",
        chat_id="oc_bootstrap_private",
    )
    event.event.message.chat_type = "p2p"
    unknown_private_trust = EffectiveTrust(
        zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
        actor=ActorKind.UNKNOWN,
        managed_group=None,
        group_revision=None,
        grant_revision=None,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(
        return_value=unknown_private_trust
    )
    system = mock_ws_client._handler_ctx.handlers["system"]
    system.handle_intercepted_command = MagicMock()

    mock_ws_client._handle_message(event)

    spec, callback = mock_ws_client._scheduler.submit.call_args.args
    assert spec.task_type == "feishu_message"
    callback(SimpleNamespace(run_id="run_bootstrap_admin"))
    system.handle_intercepted_command.assert_called_once()
    assert system.handle_intercepted_command.call_args.args[:3] == (
        "om_bootstrap_admin",
        "oc_bootstrap_private",
        "/setadmin",
    )


def test_handle_message_records_trusted_chat_origin(mock_ws_client: FeishuWSClient):
    """Message events are the authoritative source for DM provenance."""
    msg = create_mock_message("/projects", message_id="om_projects", chat_id="oc_dm")
    msg.event.message.chat_type = "p2p"
    msg.event.sender.sender_id.open_id = "ou_admin"
    msg.event.sender.sender_id.union_id = "on_admin"

    mock_ws_client._handle_message(msg)

    origin = mock_ws_client._message_linker.query("om_projects")
    assert origin is not None
    assert origin["chat_id"] == "oc_dm"
    assert origin["chat_type"] == "p2p"
    assert origin["sender_id"] == "ou_admin"
    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.sender_union_id == "on_admin"


def test_handle_message_shell_command_routing(mock_ws_client: FeishuWSClient):
    """Test that likely shell commands are fast-tracked to a shell-specific queue."""
    # Using 'ls -la' which is likely recognized as shell command by SystemHandler.is_likely_shell_command
    msg = create_mock_message("ls -la")

    mock_ws_client._handle_message(msg)

    submit_mock = mock_ws_client._scheduler.submit
    assert submit_mock.call_count == 1
    spec, _ = submit_mock.call_args[0]

    assert spec.task_type == "feishu_message"
    assert spec.priority == TaskPriority.NORMAL
    assert spec.is_system_command is False
    # Should use the fast-track shell queue
    assert spec.queue_key is not None
    assert ":shell:" in spec.queue_key


def test_handle_message_spec_command_routing(mock_ws_client: FeishuWSClient):
    """Test that spec commands use the spec rate limit configuration."""
    msg = create_mock_message("/spec do something")

    mock_ws_client._handle_message(msg)

    submit_mock = mock_ws_client._scheduler.submit
    assert submit_mock.call_count == 1
    spec, _ = submit_mock.call_args[0]

    assert spec.task_type == "spec_command"
    assert spec.is_system_command is True
    assert spec.priority == TaskPriority.HIGH
    assert spec.queue_key is not None
    assert ":control:" in spec.queue_key


@pytest.mark.parametrize(
    "text",
    [
        "/deep 恢复自主执行逻辑",
        "/spec 恢复规格闭环",
        "/wf 恢复工作流编排",
    ],
)
def test_handle_message_flat_post_engine_command_uses_system_priority(
    mock_ws_client: FeishuWSClient,
    text: str,
):
    """Flat rich-post slash commands must be classified before scheduler enqueue."""
    content_rows = [
        [{"tag": "text", "text": text, "style": []}],
        [{"tag": "img", "image_key": "img_v3_evidence"}],
    ]
    msg = create_mock_message("", message_type="post")
    msg.event.message.content = json.dumps(
        {"title": "", "content": content_rows, "content_v2": []}
    )

    mock_ws_client._handle_message(msg)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.priority is TaskPriority.HIGH
    assert spec.is_system_command is True


def test_handle_message_plain_message_does_not_fallback_to_recent_engine_topic(mock_ws_client: FeishuWSClient):
    """Plain chat messages must not continue a topic-bound engine without root_id."""
    mock_ws_client.settings.thread_programming_enabled = True
    mock_ws_client._thread_manager.register(
        "thread-deep",
        "chat_456",
        "proj_1",
        mode="deep",
    )
    msg = create_mock_message("继续")
    msg.event.message.root_id = None
    msg.event.message.parent_id = None

    mock_ws_client._handle_message(msg)

    spec, _ = mock_ws_client._scheduler.submit.call_args[0]
    assert spec.project_id is None
    assert not spec.queue_key or ":t:thread-deep" not in spec.queue_key




def test_explicit_engine_command_reaches_its_final_handler_in_every_programming_mode(
    mock_ws_client: FeishuWSClient,
):
    """Persistent programming state may not consume any explicit engine command."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._get_mode_handler = MagicMock()
    programming_modes = ("coco", "claude", "aiden", "codex", "gemini", "traex", "grok", "dsh", "agy")
    mock_ws_client._handler_ctx.handlers["coco"].add_reaction = MagicMock()
    engine_cases = (
        ("/deep 深入完成复杂任务", "deep", "handle_deep_command"),
        ("/spec 按规格迭代直到收敛", "spec", "handle_spec_command"),
        ("/wf 编排多个代理完成任务", "workflow", "handle_workflow_command"),
    )

    for text, handler_name, method_name in engine_cases:
        for programming_mode in programming_modes:
            target = MagicMock()
            setattr(
                mock_ws_client._handler_ctx.handlers[handler_name],
                method_name,
                target,
            )

            mock_ws_client._dispatch_message_logic(
                "msg_engine_final",
                "chat_456",
                text,
                project,
                programming_mode,
                command_match=SlashCommandParser.parse(text),
            )

            target.assert_called_once_with(
                "msg_engine_final", "chat_456", text, project
            )
            assert not mock_ws_client._get_mode_handler.called, (
                f"{text!r} fell back to {programming_mode!r}"
            )


@pytest.mark.parametrize(
    ("engine", "expected_method"),
    [
        ("deep", "start_deep_engine"),
        ("spec", "start_spec_engine"),
    ],
)
def test_deep_and_spec_topic_plain_text_keeps_engine_strategy(
    mock_ws_client: FeishuWSClient,
    engine: str,
    expected_method: str,
):
    """Deep/Spec topic continuation should not fall back to SMART intent routing."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    dispatch = MagicMock()
    mock_ws_client._message_dispatcher.process_with_intent = dispatch
    target = MagicMock()
    setattr(mock_ws_client._handler_ctx.handlers[engine], expected_method, target)
    mock_ws_client._handler_ctx.handlers["coco"].add_reaction = MagicMock()

    mock_ws_client._dispatch_message_logic(
        "msg_next",
        "chat_456",
        "继续按这个方向做",
        project,
        engine,
        command_match=None,
    )

    target.assert_called_once_with(
        "msg_next",
        "chat_456",
        "继续按这个方向做",
        project,
    )
    dispatch.assert_not_called()


def test_topic_engine_without_resolved_project_never_falls_back_to_smart(
    mock_ws_client: FeishuWSClient,
):
    """A topic-owned engine resolves/rejects its project instead of changing strategy."""
    dispatch = MagicMock()
    reply_text = MagicMock()
    mock_ws_client._message_dispatcher.process_with_intent = dispatch
    mock_ws_client._handler_ctx.handlers["coco"].reply_text = reply_text
    engine_cases = (
        ("deep", "start_deep_engine"),
        ("spec", "start_spec_engine"),
        ("workflow", "handle_message"),
    )
    slash_texts = {
        "deep": "/deep 继续执行",
        "spec": "/spec 继续执行",
        "workflow": "/wf 继续执行",
    }

    for engine, expected_method in engine_cases:
        for has_slash_command in (False, True):
            dispatch.reset_mock()
            reply_text.reset_mock()
            target = MagicMock()
            setattr(
                mock_ws_client._handler_ctx.handlers[engine],
                expected_method,
                target,
            )

            text = slash_texts[engine] if has_slash_command else "继续执行"
            command_match = (
                SlashCommandParser.parse(text) if has_slash_command else None
            )
            mock_ws_client._dispatch_message_logic(
                "msg_missing_project",
                "chat_456",
                text,
                None,
                engine,
                command_match=command_match,
            )

            assert not target.called, f"{engine!r} ran without a project"
            assert reply_text.call_count == 1, (
                f"{engine!r} did not explain the missing project"
            )
            assert "未执行" in reply_text.call_args.args[1]
            assert not dispatch.called, (
                f"{engine!r} fell back to SMART"
            )


def test_missing_topic_project_allows_safe_recovery_and_diagnostics_commands(
    mock_ws_client: FeishuWSClient,
):
    dispatch = MagicMock()
    reply_text = MagicMock()
    mock_ws_client._message_dispatcher.process_with_intent = dispatch
    mock_ws_client._handler_ctx.handlers["coco"].reply_text = reply_text
    commands = (
        "/projects",
        "/status",
        "/help",
        "/deep_status --all",
        "/stop_deep --all",
    )

    for text in commands:
        dispatch.reset_mock()
        reply_text.reset_mock()
        mock_ws_client._dispatch_message_logic(
            "msg_recover",
            "chat_456",
            text,
            None,
            "deep",
            command_match=SlashCommandParser.parse(text),
        )

        assert dispatch.call_count == 1, (
            f"{text!r} was not routed as a safe recovery command"
        )
        assert not reply_text.called, (
            f"{text!r} was rejected as an unsafe missing-project command"
        )


def test_process_message_async_auto_enter_mode(mock_ws_client: FeishuWSClient):
    """Test that an ongoing mode (auto_enter_mode) directly forwards to the respective handler."""
    msg = create_mock_message("hello")
    # Mock validation and parsing to skip actual processing overhead
    mock_ws_client._validate_message = MagicMock(return_value=True)

    # Mock resolving context to return a project and an auto-entered mode
    project = ProjectContext("proj_1", "Test", "/tmp")
    mock_ws_client._resolve_message_context = MagicMock(return_value=(project, "coco"))

    # Mock the mode handler
    mock_coco_handler = MagicMock()
    mock_ws_client._coco_handler = mock_coco_handler
    mock_ws_client._get_mode_handler = MagicMock(return_value=mock_coco_handler)

    # Execute the core async logic (synchronously in test)
    mock_ws_client._process_message_async(msg, task_ctx=MagicMock())

    # Since auto_enter_mode is 'coco', it should bypass intent recognition and call handle_message directly
    mock_ws_client._intent_recognizer.recognize.assert_not_called()
    mock_coco_handler.handle_message.assert_called_once_with(
        "om_123", "oc_456", "hello", project
    )




def test_flat_post_engine_command_reaches_dispatch_with_command_and_image(
    mock_ws_client: FeishuWSClient,
    monkeypatch,
    tmp_path: Path,
):
    """The production flat post shape must preserve slash routing at ingress."""
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))
    content_rows = [
        [{"tag": "text", "text": "/deep 恢复自主执行逻辑", "style": []}],
        [{"tag": "img", "image_key": "img_v3_evidence"}],
    ]
    msg = create_mock_message("", message_type="post")
    msg.event.message.content = json.dumps(
        {"title": "", "content": content_rows, "content_v2": content_rows}
    )
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    image_handler = FeishuImageHandler(MagicMock(), MagicMock())
    image_handler.download_images = MagicMock(
        return_value=ImageDownloadResult(saved_paths=["/tmp/evidence.png"])
    )
    mock_ws_client._get_image_handler = MagicMock(return_value=image_handler)
    mock_ws_client._validate_message = MagicMock(return_value=True)
    mock_ws_client._resolve_message_context = MagicMock(return_value=(project, "traex"))
    mock_ws_client._dispatch_message_logic = MagicMock()

    mock_ws_client._process_message_async(msg)

    args = mock_ws_client._dispatch_message_logic.call_args.args
    kwargs = mock_ws_client._dispatch_message_logic.call_args.kwargs
    assert args[0] == "om_123"
    assert args[1] == "oc_456"
    assert args[3] is project
    assert args[2].startswith("/deep 恢复自主执行逻辑")
    assert "/tmp/evidence.png" in args[2]
    assert args[4] == "traex"
    assert kwargs["command_match"].command == "/deep"
    image_handler.download_images.assert_called_once_with(
        "om_123",
        ["img_v3_evidence"],
        str(user_home / ".cache" / "ghostAp" / "picturechat"),
    )


def test_topic_bound_deep_blocks_spec_switch_command(mock_ws_client: FeishuWSClient):
    """A Deep topic must not be implicitly switched to Spec by a slash command."""
    project = ProjectContext("proj_1", "Test", "/tmp")
    reply_text = MagicMock()
    dispatch = MagicMock()
    mock_ws_client._handler_ctx.handlers["coco"].reply_text = reply_text
    mock_ws_client._message_dispatcher.process_with_intent = dispatch

    mock_ws_client._dispatch_message_logic(
        "msg_123",
        "chat_456",
        "/spec rewrite this",
        project,
        "deep",
        command_match=MagicMock(command="/spec"),
    )

    reply_text.assert_called_once()
    assert "DEEP" in reply_text.call_args.args[1]
    assert "SPEC" in reply_text.call_args.args[1]
    dispatch.assert_not_called()


def test_topic_bound_spec_allows_spec_command(mock_ws_client: FeishuWSClient):
    """Same-engine explicit commands remain available inside their topic."""
    project = ProjectContext("proj_1", "Test", "/tmp")
    reply_text = MagicMock()
    dispatch = MagicMock()
    mock_ws_client._handler_ctx.handlers["coco"].reply_text = reply_text
    mock_ws_client._message_dispatcher.process_with_intent = dispatch

    mock_ws_client._dispatch_message_logic(
        "msg_123",
        "chat_456",
        "/spec_status",
        project,
        "spec",
        command_match=MagicMock(command="/spec_status"),
    )

    reply_text.assert_not_called()
    dispatch.assert_called_once()


def _direct_card_action_data(
    *,
    chat_id: str,
    project_id: str,
    thread_root_id: str | None = None,
):
    value = {"action": "observe_thread_context", "project_id": project_id}
    if thread_root_id is not None:
        value["thread_root_id"] = thread_root_id
    return SimpleNamespace(
        event=SimpleNamespace(
            action=SimpleNamespace(
                value=value,
                behaviors=None,
                option=None,
                options=None,
                form_value=None,
                input_value=None,
            ),
            operator=SimpleNamespace(
                open_id="ou_card_operator",
                user_id=None,
                union_id=None,
            ),
            context=SimpleNamespace(
                open_message_id="om_card_callback",
                open_chat_id=chat_id,
                chat_type="group",
            ),
        )
    )


def _observe_card_thread_context(
    mock_ws_client: FeishuWSClient,
    data,
) -> str | None:
    observed: list[str | None] = []
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=None)
    mock_ws_client._current_trust_can_dispatch = MagicMock(return_value=True)
    mock_ws_client._chat_lock_gate.check_card_action = MagicMock(return_value=False)
    mock_ws_client._action_handlers["observe_thread_context"] = (
        lambda *_args: observed.append(get_current_thread_id())
    )
    mock_ws_client._process_card_action_async(data)
    assert len(observed) == 1
    return observed[0]


def test_card_callback_clears_inherited_thread_without_payload(
    mock_ws_client: FeishuWSClient,
):
    set_current_thread_id("om_stale_thread")
    try:
        observed = _observe_card_thread_context(
            mock_ws_client,
            _direct_card_action_data(chat_id="chat_456", project_id="proj_1"),
        )
    finally:
        set_current_thread_id(None)

    assert observed is None


def test_card_callback_restores_canonical_matching_thread(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._thread_manager.register(
        "om_canonical_root",
        "chat_456",
        "proj_1",
        alias_keys=["om_callback_alias"],
    )
    try:
        observed = _observe_card_thread_context(
            mock_ws_client,
            _direct_card_action_data(
                chat_id="chat_456",
                project_id="proj_1",
                thread_root_id="om_callback_alias",
            ),
        )
    finally:
        mock_ws_client._thread_manager.remove("om_canonical_root")

    assert observed == "om_canonical_root"


@pytest.mark.parametrize(
    ("callback_chat_id", "callback_project_id"),
    [("chat_other", "proj_1"), ("chat_456", "proj_other")],
)
def test_card_callback_rejects_cross_coordinate_thread(
    mock_ws_client: FeishuWSClient,
    callback_chat_id: str,
    callback_project_id: str,
):
    mock_ws_client._thread_manager.register(
        "om_coordinate_root",
        "chat_456",
        "proj_1",
        alias_keys=["om_coordinate_alias"],
    )
    set_current_thread_id("om_stale_thread")
    try:
        observed = _observe_card_thread_context(
            mock_ws_client,
            _direct_card_action_data(
                chat_id=callback_chat_id,
                project_id=callback_project_id,
                thread_root_id="om_coordinate_alias",
            ),
        )
    finally:
        set_current_thread_id(None)
        mock_ws_client._thread_manager.remove("om_coordinate_root")

    assert observed is None


def test_deep_start_binds_topic_context(mock_ws_client: FeishuWSClient):
    """Starting Deep registers the current Feishu topic as a Deep strategy context."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._thread_manager.remove("msg_deep")
    handler = mock_ws_client._handler_ctx.handlers["deep"]
    handler._submit_engine_task = MagicMock()
    handler.add_reaction = MagicMock()
    handler.ensure_request_id = MagicMock(return_value="req-1")
    handler.ctx.deep_engine_manager.get = MagicMock(return_value=None)
    handler.ctx.deep_engine_manager.get_or_create = MagicMock(return_value=MagicMock())

    set_current_thread_id(None)
    try:
        handler.start_deep_engine("msg_deep", "chat_456", "深入分析", project)
    finally:
        set_current_thread_id(None)

    ctx = mock_ws_client._thread_manager.get("msg_deep")
    assert ctx is not None
    assert ctx.mode == "deep"
    assert ctx.project_id == "proj_1"


def test_spec_start_binds_topic_context(mock_ws_client: FeishuWSClient):
    """Starting Spec registers the current Feishu topic as a Spec strategy context."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._thread_manager.remove("msg_spec")
    handler = mock_ws_client._handler_ctx.handlers["spec"]
    handler._submit_engine_task = MagicMock()
    handler.add_reaction = MagicMock()
    handler.ensure_request_id = MagicMock(return_value="req-1")
    handler.ctx.spec_engine_manager.get = MagicMock(return_value=None)
    handler.ctx.spec_engine_manager.get_or_create = MagicMock(return_value=MagicMock())

    set_current_thread_id(None)
    try:
        handler.start_spec_engine("msg_spec", "chat_456", "写清规格", project)
    finally:
        set_current_thread_id(None)

    ctx = mock_ws_client._thread_manager.get("msg_spec")
    assert ctx is not None
    assert ctx.mode == "spec"
    assert ctx.project_id == "proj_1"


def test_exit_in_engine_topic_unbinds_topic_strategy(mock_ws_client: FeishuWSClient):
    """In an engine-only topic, /exit exits the topic strategy instead of reporting SMART."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._thread_manager.register("thread-deep-exit", "chat_456", "proj_1", mode="deep")
    system = mock_ws_client._handler_ctx.handlers["system"]
    system.reply_text = MagicMock()
    mock_ws_client._control_plane.should_defer_exit = MagicMock(return_value=False)

    set_current_thread_id("thread-deep-exit")
    try:
        mock_ws_client._dispatch_message_logic(
            "msg_exit",
            "chat_456",
            "/exit",
            project,
            "deep",
            command_match=SlashCommandParser.parse("/exit"),
        )
    finally:
        set_current_thread_id(None)

    assert mock_ws_client._thread_manager.get("thread-deep-exit") is None
    system.reply_text.assert_called_once()


def test_dispatcher_processes_each_recognized_task(mock_ws_client: FeishuWSClient):
    """Test that intent recognizer correctly triggers multi-task execution."""
    project = ProjectContext("proj_1", "Test", "/tmp")
    mock_ws_client._get_effective_mode = MagicMock(return_value=(InteractionMode.SMART, False))

    # Mock the intent result to return a multi-task plan
    mock_intent_result = IntentResult(
        confidence=0.9,
        tasks=[
            TaskStep(intent=IntentType.CREATE_PROJECT, data={"name": "new_proj"}, description="Create project"),
            TaskStep(intent=IntentType.ENTER_COCO, data={}, description="Enter coco")
        ]
    )
    mock_ws_client._intent_recognizer.recognize.return_value = mock_intent_result

    execute_single_task = MagicMock()
    mock_ws_client._message_dispatcher.execute_single_task = execute_single_task
    mock_ws_client._handler_ctx.handlers["coco"].add_reaction = MagicMock()

    mock_ws_client._message_dispatcher.process_with_intent(
        "msg_123", "chat_456", "create a project and enter coco", project
    )

    assert execute_single_task.call_count == 2

    call_args_list = execute_single_task.call_args_list
    assert call_args_list[0][0][2].intent == IntentType.CREATE_PROJECT
    assert call_args_list[1][0][2].intent == IntentType.ENTER_COCO


def test_dispatcher_intercepts_engine_command_before_intent_recognition(
    mock_ws_client: FeishuWSClient,
):
    """Test that system commands bypass intent recognition completely during SMART mode."""
    project = ProjectContext("proj_1", "Test", "/tmp")
    mock_ws_client._get_effective_mode = MagicMock(return_value=(InteractionMode.SMART, False))

    handle_deep_command = MagicMock()
    mock_ws_client._handler_ctx.handlers["deep"].handle_deep_command = (
        handle_deep_command
    )
    mock_ws_client._handler_ctx.handlers["coco"].add_reaction = MagicMock()

    # Send a deep engine command
    mock_ws_client._message_dispatcher.process_with_intent(
        "msg_123", "chat_456", "/deep something", project
    )

    # Intent recognizer must not be called
    mock_ws_client._intent_recognizer.recognize.assert_not_called()
    # It should be directly routed to handle_deep_command
    handle_deep_command.assert_called_once_with(
        "msg_123", "chat_456", "/deep something", project
    )


def _card_action_data(*, event_id: str, message_id: str, chat_id: str, operator_id: str):
    data = MagicMock()
    data.schema = "2.0"
    data.header.event_id = event_id
    data.header.event_type = "card.action.trigger"
    data.header.tenant_key = "tenant_card"
    data.event.context.open_message_id = message_id
    data.event.context.open_chat_id = chat_id
    # The official card callback context has no chat_type field.
    del data.event.context.chat_type
    data.event.action.tag = "button"
    data.event.action.name = ""
    data.event.action.value = {"action": "show_status"}
    data.event.operator.open_id = operator_id
    data.event.operator.user_id = None
    data.event.operator.union_id = None
    return data


def test_card_action_rejects_old_callback_before_scheduling(
    mock_ws_client: FeishuWSClient,
) -> None:
    data = _card_action_data(
        event_id="evt_old_callback",
        message_id="msg_old_card",
        chat_id="chat_old",
        operator_id="ou_admin",
    )
    data.schema = "1.0"

    response = mock_ws_client._handle_card_action(data)

    assert response.__class__.__name__ == "P2CardActionTriggerResponse"
    mock_ws_client._scheduler.submit.assert_not_called()


def test_stale_managed_card_refresh_runs_only_in_system_follow_up(
    mock_ws_client: FeishuWSClient,
) -> None:
    trust = SimpleNamespace(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        managed_group=SimpleNamespace(chat_id="chat_managed", project_id="project_1"),
        group_revision=7,
        grant_revision=11,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=trust)
    mock_ws_client._managed_trust_access_decision = MagicMock(return_value=None)
    mock_ws_client._refresh_managed_card_revisions = MagicMock(return_value=True)
    submitted: list[tuple[object, object]] = []

    def submit(spec, callback):
        submitted.append((spec, callback))
        return SimpleNamespace(run_id="follow-up-1")

    mock_ws_client._scheduler.submit.side_effect = submit
    data = _card_action_data(
        event_id="evt_stale_managed",
        message_id="msg_stale_managed",
        chat_id="chat_managed",
        operator_id="ou_admin",
    )
    data.event.action.value.update({"group_revision": 6, "grant_revision": 11})

    response = mock_ws_client._handle_card_action_callback(data)

    assert response.__class__.__name__ == "P2CardActionTriggerResponse"
    mock_ws_client._refresh_managed_card_revisions.assert_not_called()
    assert submitted == []
    mock_ws_client._finish_card_advisories(True)
    assert len(submitted) == 1
    spec, callback = submitted[0]
    assert spec.name == "refresh_stale_managed_card"
    assert spec.task_type == "card_advisory_follow_up"
    assert spec.chat_id == "chat_managed"
    assert spec.is_system_command is True
    callback(SimpleNamespace())
    mock_ws_client._refresh_managed_card_revisions.assert_called_once_with(
        "msg_stale_managed", "chat_managed", trust
    )


def test_registered_card_callback_submits_advisory_only_after_inner_ack_returns(
    mock_ws_client: FeishuWSClient,
) -> None:
    """A real scheduler worker cannot race the stale refresh ahead of ACK creation."""
    from src.tasking import TaskScheduler

    trust = SimpleNamespace(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        managed_group=SimpleNamespace(chat_id="chat_managed", project_id="project_1"),
        group_revision=7,
        grant_revision=11,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=trust)
    mock_ws_client._managed_trust_access_decision = MagicMock(return_value=None)
    remote_started = threading.Event()

    def refresh(*_args):
        remote_started.set()
        return True

    mock_ws_client._refresh_managed_card_revisions = refresh
    real_scheduler = TaskScheduler(
        max_concurrent=1,
        system_concurrency=1,
        max_pending_normal=4,
        max_pending_system=4,
    )
    original_scheduler = mock_ws_client._scheduler
    mock_ws_client._scheduler = real_scheduler
    from lark_channel.ws.pb.pbbp2_pb2 import Frame

    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt_stale_real_worker",
            "event_type": "card.action.trigger",
            "tenant_key": "tenant_card",
        },
        "event": {
            "context": {
                "open_message_id": "msg_stale_real_worker",
                "open_chat_id": "chat_managed",
            },
            "operator": {"open_id": "ou_admin"},
            "action": {
                "tag": "button",
                "value": {
                    "action": "show_status",
                    "group_revision": 6,
                    "grant_revision": 11,
                },
            },
        },
    }
    frame = Frame()
    frame.SeqID = 1
    frame.LogID = 1
    frame.service = 1
    frame.method = 1
    for key, value in (
        ("type", "card"),
        ("message_id", "msg-stale-real-worker"),
        ("trace_id", "trace-stale-real-worker"),
        ("sum", "1"),
        ("seq", "0"),
    ):
        header = frame.headers.add()
        header.key = key
        header.value = value
    frame.payload = json.dumps(payload).encode()

    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    observed._on_activity = lambda _kind: None
    observed._on_response_written = mock_ws_client._finish_card_advisories
    observed._event_handler = mock_ws_client._build_event_handler()
    observed._conn_id = ""

    try:
        async def exercise() -> None:
            write_started = asyncio.Event()
            release_write = asyncio.Event()

            async def write_message(_raw):
                write_started.set()
                await release_write.wait()

            observed._write_message = write_message
            callback_task = asyncio.create_task(observed._handle_data_frame(frame))
            await asyncio.wait_for(write_started.wait(), timeout=2)
            assert remote_started.is_set() is False
            release_write.set()
            await asyncio.wait_for(callback_task, timeout=2)
            deadline = asyncio.get_running_loop().time() + 2
            while not remote_started.is_set() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert remote_started.is_set()

        asyncio.run(exercise())
    finally:
        real_scheduler.stop(wait=True)
        mock_ws_client._scheduler = original_scheduler


@pytest.mark.parametrize(
    "admission_error",
    [
        TaskQueueFullError("system", 1),
        RuntimeError("TaskScheduler admission is fenced"),
    ],
    ids=["system-lane-full", "admission-fenced"],
)
def test_stale_managed_card_follow_up_rejection_never_refreshes_synchronously(
    mock_ws_client: FeishuWSClient,
    admission_error: Exception,
) -> None:
    trust = SimpleNamespace(
        zone=TrustZone.MANAGED_AGENT_GROUP,
        managed_group=SimpleNamespace(chat_id="chat_managed", project_id="project_1"),
        group_revision=7,
        grant_revision=11,
    )
    mock_ws_client._resolve_effective_trust = MagicMock(return_value=trust)
    mock_ws_client._managed_trust_access_decision = MagicMock(return_value=None)
    mock_ws_client._refresh_managed_card_revisions = MagicMock(return_value=True)
    mock_ws_client._scheduler.submit.side_effect = admission_error
    data = _card_action_data(
        event_id="evt_stale_full",
        message_id="msg_stale_full",
        chat_id="chat_managed",
        operator_id="ou_admin",
    )
    data.event.action.value.update({"group_revision": 6, "grant_revision": 11})

    response = mock_ws_client._handle_card_action_callback(data)

    assert response.__class__.__name__ == "P2CardActionTriggerResponse"
    mock_ws_client._refresh_managed_card_revisions.assert_not_called()
    mock_ws_client._finish_card_advisories(True)
    assert mock_ws_client._scheduler.submit.call_count == 1


def test_close_shuts_down_card_registry_with_one_bounded_wave(
    mock_ws_client: FeishuWSClient,
) -> None:
    with (
        patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ) as shutdown_all,
        patch(
            "src.card.delivery.registry.delivery_registry.drain_in_flight",
            return_value=True,
        ) as drain_in_flight,
    ):
        assert mock_ws_client.close() is True

    drain_in_flight.assert_not_called()
    shutdown_all.assert_called_once()
    assert shutdown_all.call_args.kwargs["timeout"] > 0


def test_only_control_plane_scheduler_listener_registers_and_detaches_after_idle_close(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    control_plane_callbacks = [
        call.args[0]
        for call in scheduler.add_listener.call_args_list
        if getattr(call.args[0], "__self__", None) is mock_ws_client._control_plane
    ]

    assert len(control_plane_callbacks) == 1
    assert scheduler.add_listener.call_count == 1
    scheduler.remove_listener.reset_mock()

    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=True,
    ):
        assert mock_ws_client.close() is True

    scheduler.remove_listener.assert_called_once_with(control_plane_callbacks[0])


def test_close_keeps_scheduler_listener_until_running_callbacks_drain(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    scheduler.wait_for_idle.return_value = False
    scheduler.remove_listener.reset_mock()

    assert mock_ws_client.close() is False

    scheduler.remove_listener.assert_not_called()


def test_close_waits_for_completion_callbacks_before_manager_cleanup(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    scheduler.wait_for_idle.return_value = True
    cleanup.reset_mock()

    def wait_for_completions(*, timeout: float) -> bool:
        assert timeout > 0
        cleanup.assert_not_called()
        return True

    scheduler.wait_for_completion_callbacks.side_effect = wait_for_completions
    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=True,
    ):
        assert mock_ws_client.close() is True

    scheduler.wait_for_completion_callbacks.assert_called_once()
    assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)


def test_close_preserves_manager_cleanup_when_completion_callbacks_do_not_drain(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = False
    cleanup.reset_mock()

    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=True,
    ) as shutdown_all:
        assert mock_ws_client.close() is False

    shutdown_all.assert_not_called()
    cleanup.assert_not_called()


def test_close_waits_for_terminal_replay_to_anchor_prestart_warning(
    mock_ws_client: FeishuWSClient,
) -> None:
    from src.feishu.ws_client import _MessageIngressReservation

    original_scheduler = mock_ws_client._scheduler
    scheduler = TaskScheduler(max_concurrent=1, system_concurrency=1)
    mock_ws_client._scheduler = scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()
    message_id = "om_terminal_replay_shutdown"
    reservation_owner = mock_ws_client._message_ingress_guard.reserve(message_id)
    assert reservation_owner is not None
    reservation = _MessageIngressReservation(
        guard=mock_ws_client._message_ingress_guard,
        message_id=message_id,
        owner=reservation_owner,
    )
    terminal = threading.Event()
    warning_started = threading.Event()
    release_warning = threading.Event()
    warning_anchored = threading.Event()
    close_wait_started = threading.Event()
    cleanup_started = threading.Event()
    close_results: list[bool] = []

    scheduler.add_listener(
        lambda event: terminal.set()
        if event.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}
        else None
    )
    handle = scheduler.submit(
        TaskSpec(chat_id="oc_group", name="prestart-terminal"),
        lambda _ctx: None,
    )
    assert terminal.wait(timeout=1)

    def reply_warning(*_args, **_kwargs) -> bool:
        warning_started.set()
        release_warning.wait(timeout=3)
        warning_anchored.set()
        return True

    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    cleanup.side_effect = cleanup_started.set
    wait_for_completions = scheduler.wait_for_completion_callbacks

    def observe_completion_wait(*, timeout: float) -> bool:
        close_wait_started.set()
        return wait_for_completions(timeout=timeout)

    scheduler.wait_for_completion_callbacks = observe_completion_wait  # type: ignore[method-assign]
    registration_thread = threading.Thread(
        target=mock_ws_client._bind_message_ingress_reservation,
        args=(handle, reservation),
        kwargs={"tenant_key": "tenant-a", "chat_id": "oc_group"},
    )
    close_thread = threading.Thread(
        target=lambda: close_results.append(mock_ws_client.close())
    )

    try:
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            registration_thread.start()
            assert warning_started.wait(timeout=1)
            close_thread.start()
            assert close_wait_started.wait(timeout=1)
            assert cleanup_started.wait(timeout=0.2) is False

            release_warning.set()
            registration_thread.join(timeout=1)
            close_thread.join(timeout=2)

        assert warning_anchored.is_set()
        assert close_results == [True]
        assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)
        assert mock_ws_client._message_ingress_guard.reserve(message_id) is None
    finally:
        release_warning.set()
        registration_thread.join(timeout=1)
        close_thread.join(timeout=2)
        mock_ws_client._scheduler = original_scheduler
        scheduler.stop(wait=True, shutdown_executor=True)


def test_close_waits_for_dispatched_message_to_bind_prestart_callback(
    mock_ws_client: FeishuWSClient,
) -> None:
    """A Channel-dispatched handler must join shutdown before teardown."""

    class ReplayedTerminalHandle:
        run_id = "run_submit_bind_shutdown"

        @staticmethod
        def add_done_callback(callback) -> None:
            callback(
                SimpleNamespace(
                    run_id="run_submit_bind_shutdown",
                    status=TaskStatus.CANCELED,
                )
            )

    scheduler = mock_ws_client._scheduler
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    scheduler.submit.return_value = ReplayedTerminalHandle()
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = True
    reply_warning = MagicMock(return_value=True)
    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    cleanup.reset_mock()

    bind_entered = threading.Event()
    release_bind = threading.Event()
    cleanup_started = threading.Event()
    close_results: list[bool] = []
    original_bind = mock_ws_client._bind_message_ingress_reservation

    def block_between_submit_and_bind(*args, **kwargs) -> None:
        bind_entered.set()
        assert release_bind.wait(timeout=2)
        original_bind(*args, **kwargs)

    mock_ws_client._bind_message_ingress_reservation = block_between_submit_and_bind  # type: ignore[method-assign]
    cleanup.side_effect = cleanup_started.set
    handler_thread = threading.Thread(
        target=mock_ws_client._handle_message,
        args=(
            create_mock_message(
                "hello",
                message_id="om_submit_bind_shutdown",
                chat_id="oc_submit_bind_shutdown",
            ),
        ),
    )
    close_thread = threading.Thread(
        target=lambda: close_results.append(mock_ws_client.close())
    )

    try:
        assert mock_ws_client._begin_message_ingress_binding()
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            handler_thread.start()
            assert bind_entered.wait(timeout=1)
            close_thread.start()
            assert cleanup_started.wait(timeout=0.2) is False

            release_bind.set()
            handler_thread.join(timeout=1)
            mock_ws_client._finish_message_ingress_binding()
            close_thread.join(timeout=2)

        assert handler_thread.is_alive() is False
        assert close_thread.is_alive() is False
        assert close_results == [True]
        reply_warning.assert_called_once_with(
            "om_submit_bind_shutdown",
            UI_TEXT["ws_message_prestart_terminal"],
            label="pre-start terminal warning",
        )
        assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)
    finally:
        release_bind.set()
        handler_thread.join(timeout=1)
        if mock_ws_client._message_ingress_bindings_inflight:
            mock_ws_client._finish_message_ingress_binding()
        close_thread.join(timeout=2)


def test_close_timeout_preserves_cleanup_during_submit_bind_window(
    mock_ws_client: FeishuWSClient,
) -> None:
    scheduler = mock_ws_client._scheduler
    scheduler.submit.return_value = _pending_task_handle("run_bind_timeout")
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = True
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()

    bind_entered = threading.Event()
    release_bind = threading.Event()
    original_bind = mock_ws_client._bind_message_ingress_reservation

    def block_between_submit_and_bind(*args, **kwargs) -> None:
        bind_entered.set()
        assert release_bind.wait(timeout=2)
        original_bind(*args, **kwargs)

    mock_ws_client._bind_message_ingress_reservation = block_between_submit_and_bind  # type: ignore[method-assign]
    handler_thread = threading.Thread(
        target=mock_ws_client._handle_message,
        args=(
            create_mock_message(
                "hello",
                message_id="om_bind_timeout",
                chat_id="oc_bind_timeout",
            ),
        ),
    )

    try:
        assert mock_ws_client._begin_message_ingress_binding()
        handler_thread.start()
        assert bind_entered.wait(timeout=1)
        started = time.monotonic()
        with (
            patch("src.feishu.ws_client._SHUTDOWN_SCHEDULER_DRAIN_S", 0.05),
            patch(
                "src.card.delivery.registry.delivery_registry.shutdown_all",
                return_value=True,
            ) as shutdown_all,
        ):
            assert mock_ws_client.close() is False
        assert time.monotonic() - started < 0.5
        cleanup.assert_not_called()
        shutdown_all.assert_not_called()
    finally:
        release_bind.set()
        handler_thread.join(timeout=1)
        if mock_ws_client._message_ingress_bindings_inflight:
            mock_ws_client._finish_message_ingress_binding()


def test_observed_ws_tracks_scheduled_handler_before_business_entry(
    mock_ws_client: FeishuWSClient,
) -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    scheduler = mock_ws_client._scheduler
    scheduler.submit.return_value = SimpleNamespace(
        run_id="run_scheduled_before_entry",
        add_done_callback=lambda callback: callback(
            SimpleNamespace(
                run_id="run_scheduled_before_entry",
                status=TaskStatus.CANCELED,
            )
        ),
    )
    scheduler.wait_for_idle.return_value = True
    scheduler.wait_for_completion_callbacks.return_value = True
    reply_warning = MagicMock(return_value=True)
    mock_ws_client._reply_text_with_bounded_retry = reply_warning
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()

    observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    observed._handler_semaphore = None
    observed._on_handler_scheduled = mock_ws_client._begin_message_ingress_binding
    observed._on_handler_finished = mock_ws_client._finish_message_ingress_binding
    allow_business_entry: asyncio.Event
    cleanup_started = threading.Event()
    close_results: list[bool] = []
    cleanup.side_effect = cleanup_started.set
    message = create_mock_message(
        "hello",
        message_id="om_scheduled_before_entry",
        chat_id="oc_scheduled_before_entry",
    )

    async def exercise() -> None:
        nonlocal allow_business_entry
        allow_business_entry = asyncio.Event()

        async def delayed_handler(_raw) -> None:
            await allow_business_entry.wait()
            mock_ws_client._handle_message(message)

        observed._handle_message = delayed_handler  # type: ignore[method-assign]
        await observed._schedule_handle_message(b"raw-frame")
        close_thread = threading.Thread(
            target=lambda: close_results.append(mock_ws_client.close())
        )
        with patch(
            "src.card.delivery.registry.delivery_registry.shutdown_all",
            return_value=True,
        ):
            close_thread.start()
            deadline = asyncio.get_running_loop().time() + 0.2
            while not cleanup_started.is_set() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert cleanup_started.is_set() is False
            allow_business_entry.set()
            deadline = asyncio.get_running_loop().time() + 2
            while close_thread.is_alive() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
        assert close_thread.is_alive() is False

    asyncio.run(exercise())

    assert close_results == [True]
    reply_warning.assert_called_once_with(
        "om_scheduled_before_entry",
        UI_TEXT["ws_message_prestart_terminal"],
        label="pre-start terminal warning",
    )
    assert cleanup.call_count == len(mock_ws_client._handler_ctx.managers)


def test_observed_ws_rejects_handler_scheduled_after_shutdown_fence(
    mock_ws_client: FeishuWSClient,
) -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
    observed._handler_semaphore = None
    observed._on_handler_scheduled = mock_ws_client._begin_message_ingress_binding
    observed._on_handler_finished = mock_ws_client._finish_message_ingress_binding
    handled = MagicMock()

    async def handle_message(_raw) -> None:
        handled()

    observed._handle_message = handle_message  # type: ignore[method-assign]
    mock_ws_client._fence_message_ingress_bindings()

    asyncio.run(observed._schedule_handle_message(b"late-frame"))

    handled.assert_not_called()
    assert mock_ws_client._message_ingress_bindings_inflight == 0


def test_observed_ws_immediate_cancel_releases_handler_tracking() -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    scheduled = MagicMock(return_value=True)
    finished = MagicMock()
    handled = MagicMock()

    async def exercise() -> None:
        observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
        observed._handler_semaphore = None
        observed._on_handler_scheduled = scheduled
        observed._on_handler_finished = finished

        async def handle_message(_raw) -> None:
            handled()

        observed._handle_message = handle_message  # type: ignore[method-assign]
        before = asyncio.all_tasks()
        await observed._schedule_handle_message(b"cancel-before-start")
        created = asyncio.all_tasks() - before
        assert len(created) == 1
        task = created.pop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())

    scheduled.assert_called_once_with()
    finished.assert_called_once_with()
    handled.assert_not_called()


def test_observed_ws_immediate_cancel_releases_handler_semaphore() -> None:
    from src.feishu.ws_lifecycle import ObservedLarkWSClient

    scheduled = MagicMock(return_value=True)
    finished = MagicMock()
    handled = MagicMock()

    async def exercise() -> None:
        observed = ObservedLarkWSClient.__new__(ObservedLarkWSClient)
        observed._handler_semaphore = asyncio.Semaphore(1)
        observed._on_handler_scheduled = scheduled
        observed._on_handler_finished = finished

        async def handle_message(_raw) -> None:
            handled()

        observed._handle_message = handle_message  # type: ignore[method-assign]
        before = asyncio.all_tasks()
        await observed._schedule_handle_message(b"cancel-before-start")
        created = asyncio.all_tasks() - before
        assert len(created) == 1
        task = created.pop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.wait_for(observed._handler_semaphore.acquire(), timeout=0.1)

    asyncio.run(exercise())

    scheduled.assert_called_once_with()
    finished.assert_called_once_with()
    handled.assert_not_called()


def test_close_preserves_dependencies_when_card_registry_shutdown_times_out(
    mock_ws_client: FeishuWSClient,
) -> None:
    cleanup = mock_ws_client._handler_ctx.managers["coco"].cleanup_all
    cleanup.reset_mock()
    with patch(
        "src.card.delivery.registry.delivery_registry.shutdown_all",
        return_value=False,
    ):
        assert mock_ws_client.close() is False

    cleanup.assert_not_called()


def test_card_action_restores_p2p_from_trusted_message_origin(mock_ws_client: FeishuWSClient):
    """A DM selection flow must remain a DM after the callback hop."""
    mock_ws_client._message_linker.register_origin(
        "msg_origin",
        request_id="req_hire",
        chat_id="chat_dm",
        chat_type="p2p",
        sender_id="ou_admin",
    )
    mock_ws_client._message_linker.link_reply("msg_origin", "msg_card")
    data = _card_action_data(
        event_id="evt_hire_select",
        message_id="msg_card",
        chat_id="chat_dm",
        operator_id="ou_admin",
    )

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.origin_message_id == "msg_origin"
    assert spec.is_p2p is True


def test_card_action_after_restart_fails_closed_without_chat_api_lookup(
    mock_ws_client: FeishuWSClient,
):
    """The three-second callback path never performs a remote provenance lookup."""
    response = MagicMock()
    response.success.return_value = True
    response.data.chat_mode = "p2p"
    response.data.chat_type = "public"
    api_client = MagicMock()
    api_client.im.v1.chat.get.return_value = response
    mock_ws_client._get_api_client = MagicMock(return_value=api_client)

    data = _card_action_data(
        event_id="evt_after_restart",
        message_id="msg_card_after_restart",
        chat_id="chat_dm",
        operator_id="ou_admin",
    )
    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False
    api_client.im.v1.chat.get.assert_not_called()


def test_card_action_ignores_non_contract_callback_chat_type(mock_ws_client: FeishuWSClient):
    """An injected callback context.chat_type must not grant DM privileges."""
    response = MagicMock()
    response.success.return_value = True
    response.data.chat_mode = "group"
    response.data.chat_type = "private"
    api_client = MagicMock()
    api_client.im.v1.chat.get.return_value = response
    mock_ws_client._get_api_client = MagicMock(return_value=api_client)
    data = _card_action_data(
        event_id="evt_group",
        message_id="msg_group_card",
        chat_id="chat_group",
        operator_id="ou_admin",
    )
    data.event.context.chat_type = "p2p"

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False


def test_card_action_cross_operator_provenance_fails_closed(mock_ws_client: FeishuWSClient):
    mock_ws_client._message_linker.register_origin(
        "msg_origin",
        request_id="req_hire",
        chat_id="chat_dm",
        chat_type="p2p",
        sender_id="ou_original_admin",
    )
    mock_ws_client._message_linker.link_reply("msg_origin", "msg_card")
    mock_ws_client._get_api_client = MagicMock()
    data = _card_action_data(
        event_id="evt_other_operator",
        message_id="msg_card",
        chat_id="chat_dm",
        operator_id="ou_other_admin",
    )

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_partial_origin_provenance_fails_closed(mock_ws_client: FeishuWSClient):
    mock_ws_client._message_linker.register_origin(
        "msg_partial",
        request_id="req_partial",
        chat_id="chat_dm",
    )
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_partial",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_origin_query_error_fails_closed(mock_ws_client: FeishuWSClient):
    mock_ws_client._message_linker.query = MagicMock(side_effect=RuntimeError("cache unavailable"))
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_origin",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_origin_resolution_error_does_not_become_api_miss(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._message_linker.resolve_origin = MagicMock(
        side_effect=OSError("origin index unavailable")
    )
    mock_ws_client._get_api_client = MagicMock()
    data = _card_action_data(
        event_id="evt_origin_index_error",
        message_id="msg_card",
        chat_id="chat_dm",
        operator_id="ou_admin",
    )

    mock_ws_client._handle_card_action(data)

    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.is_p2p is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_api_fallback_rejects_empty_operator(mock_ws_client: FeishuWSClient):
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_card",
        open_chat_id="chat_dm",
        operator_id="",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_card_action_missing_origin_never_attempts_remote_provenance_write(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._get_api_client = MagicMock()
    mock_ws_client._message_linker = MagicMock()
    mock_ws_client._message_linker.query.return_value = None
    mock_ws_client._message_linker.register_trusted_origin_if_absent.return_value = None

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_card",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()
    mock_ws_client._message_linker.register_trusted_origin_if_absent.assert_not_called()


def test_card_action_rejects_provenance_for_different_origin(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._message_linker = MagicMock()
    mock_ws_client._message_linker.query.return_value = {
        "origin_message_id": "msg_other_origin",
        "chat_id": "chat_dm",
        "sender_id": "ou_admin",
        "chat_type": "p2p",
    }
    mock_ws_client._get_api_client = MagicMock()

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_expected_origin",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False
    mock_ws_client._get_api_client.assert_not_called()


def test_rejected_cross_chat_callback_cannot_rewrite_trusted_provenance(
    mock_ws_client: FeishuWSClient,
):
    assert mock_ws_client._message_linker.register_trusted_origin_if_absent(
        "msg_origin",
        chat_id="chat_dm",
        sender_id="ou_admin",
        chat_type="p2p",
    ) is True
    mock_ws_client._message_linker.link_reply("msg_origin", "msg_card")
    mock_ws_client._get_api_client = MagicMock()

    for event_id in ("evt_cross_chat_1", "evt_cross_chat_2"):
        data = _card_action_data(
            event_id=event_id,
            message_id="msg_card",
            chat_id="chat_other",
            operator_id="ou_admin",
        )
        mock_ws_client._handle_card_action(data)
        spec, _ = mock_ws_client._scheduler.submit.call_args.args
        assert spec.is_p2p is False
        assert mock_ws_client._message_linker.query("msg_origin")["chat_id"] == "chat_dm"

    mock_ws_client._get_api_client.assert_not_called()


# ---------------------------------------------------------------------------
# AC-18: chat-lock intercept card fallback on card send failure
# ---------------------------------------------------------------------------


class TestChatLockInterceptFallback:
    """AC-18: when the chat-lock intercept card fails to send, a plain text
    fallback message is delivered to the user.

    The card building + sending now lives in BaseHandler; ws_client delegates.
    """

    def test_fallback_text_on_card_build_failure(self, mock_ws_client):
        """Card build failure in handler → fallback plain text with lock icon."""
        from unittest.mock import MagicMock

        from src.feishu.handlers.lock_helper import LockHelper

        handler = MagicMock()

        # Simulate card build failure inside handler method
        clm = MagicMock()
        clm.get_lock_info.side_effect = RuntimeError("db error")

        # Use the real LockHelper with the mock handler
        lock_helper = LockHelper(handler)
        lock_helper.send_chat_lock_intercept_card("msg_1", "chat_1", clm)

        # Fallback should have been called via reply_text
        handler.reply_text.assert_called_once()
        args = handler.reply_text.call_args[0]
        assert args[0] == "msg_1"
        assert "🔒" in args[1] or "locked" in args[1].lower() or "锁定" in args[1]


def test_sync_main_bot_identity_releases_lock_during_http_and_singleflights(
    mock_ws_client: FeishuWSClient,
) -> None:
    """Slow HTTP bot identity lookup does not hold lock and coalesces callers."""
    from concurrent.futures import ThreadPoolExecutor

    from lark_oapi.core.model.base_response import BaseResponse
    from lark_oapi.core.model.raw_response import RawResponse

    entered = threading.Event()
    release = threading.Event()

    def slow_request(_request):
        entered.set()
        assert release.wait(timeout=5.0)
        result = BaseResponse()
        result.code = 0
        result.raw = RawResponse()
        result.raw.status_code = 200
        result.raw.content = json.dumps(
            {
                "code": 0,
                "bot": {
                    "app_id": "test_app_id",
                    "open_id": "ou_singleflight_bot",
                },
            }
        ).encode()
        return result

    mock_api_request = MagicMock(side_effect=slow_request)
    mock_ws_client._api_client = SimpleNamespace(request=mock_api_request)
    mock_ws_client._main_bot_open_id = ""
    mock_ws_client._main_bot_identity_next_retry_at = 0.0

    with ThreadPoolExecutor(max_workers=4) as pool:
        first_future = pool.submit(mock_ws_client._sync_main_bot_identity)
        assert entered.wait(timeout=2.0)

        # Lock MUST NOT be held across slow SDK I/O
        acquired = mock_ws_client._main_bot_identity_lock.acquire(timeout=0.5)
        assert acquired is True
        mock_ws_client._main_bot_identity_lock.release()

        # Concurrent callers wait on condition without issuing extra HTTP calls
        other_futures = [
            pool.submit(mock_ws_client._sync_main_bot_identity) for _ in range(3)
        ]
        release.set()

        results = [first_future.result(timeout=5.0)] + [
            fut.result(timeout=5.0) for fut in other_futures
        ]

    assert results == ["ou_singleflight_bot"] * 4
    assert mock_api_request.call_count == 1
