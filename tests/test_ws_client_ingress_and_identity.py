import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.access_control import IngressAccessPolicy
from src.card.ui_text import UI_TEXT
from src.config import IngressAccessMode
from src.feishu.ws_client import FeishuWSClient
from src.project import ProjectContext
from src.tasking import TaskQueueFullError, TaskScheduler, TaskSpec, TaskStatus
from src.trust.models import ActorKind, TrustZone
from tests.helpers.ws_routing_harness import (
    _mentioned_group_command,
    _pending_task_handle,
    _sdk_message_mention,
    create_mock_message,
    mock_ws_client,
)

__all__ = ["mock_ws_client"]


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
