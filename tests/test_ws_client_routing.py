import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.intent_recognizer import IntentResult, IntentType, TaskStep
from src.feishu.image_handler import FeishuImageHandler, ImageDownloadResult
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import (
    FeishuWSClient,
    _employee_hire_status_uuid,
    _main_bot_outbound_wiring,
    _visible_employee_runtime_requires_outbound_audit,
)
from src.mode import InteractionMode
from src.project import ProjectContext
from src.tasking import TaskPriority
from src.thread import get_current_thread_id, set_current_thread_id


@pytest.fixture
def mock_ws_client():
    """Build the real routing graph without external sessions or worker threads."""
    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()
    audit = SimpleNamespace(
        record_attempt=MagicMock(),
        mark_incomplete=MagicMock(),
    )

    def _runtime_from_settings(_settings, **kwargs):
        runtime = SimpleNamespace(
            hire_service=object(),
            fire_service=object(),
            membership_service=object(),
            data_composition=object(),
            team_service=object(),
            main_bot_outbound_audit=audit,
            readiness=lambda: SimpleNamespace(ready=True),
            _environment_provider=lambda _authority: SimpleNamespace(
                credential_env={},
                provider_files={
                    "traex_auth_json": str(
                        Path.home() / ".trae" / "cli" / "auth.json"
                    )
                },
            ),
            close=MagicMock(),
        )
        runtime._service = SimpleNamespace(
            _on_registration_status=kwargs["notification_status"],
        )
        return runtime

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
        patch(
            "src.autonomous.provisioning.composition.EmployeeDepartmentRuntime.from_settings",
            side_effect=_runtime_from_settings,
        ),
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
        settings.autonomous_visible_employee_limit = 1
        mock_get_settings.return_value = settings

        client = FeishuWSClient(MagicMock())
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


def test_handle_message_records_trusted_chat_origin(mock_ws_client: FeishuWSClient):
    """Message events are the authoritative source for DM provenance."""
    msg = create_mock_message("/hire 柳七月", message_id="om_hire", chat_id="oc_dm")
    msg.event.message.chat_type = "p2p"
    msg.event.sender.sender_id.open_id = "ou_admin"
    msg.event.sender.sender_id.union_id = "on_admin"

    mock_ws_client._handle_message(msg)

    origin = mock_ws_client._message_linker.query("om_hire")
    assert origin is not None
    assert origin["chat_id"] == "oc_dm"
    assert origin["chat_type"] == "p2p"
    assert origin["sender_id"] == "ou_admin"
    spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert spec.sender_union_id == "on_admin"


def test_employee_department_runtime_is_wired_and_enabled_by_default(
    mock_ws_client: FeishuWSClient,
) -> None:
    from src.autonomous.gateway.env_scope import EmployeeEnvironmentAuthority

    runtime = mock_ws_client._employee_department_runtime

    assert runtime is not None
    assert runtime.hire_service is not None
    assert runtime.readiness().ready is True
    assert mock_ws_client._handler_ctx.employee_hire_service is not None
    assert mock_ws_client._handler_ctx.employee_fire_service is not None
    assert mock_ws_client._handler_ctx.employee_hire_readiness().ready is True
    material = runtime._environment_provider(  # noqa: SLF001
        EmployeeEnvironmentAuthority("tenant_1", "agt_1", 1, "cred_1")
    )
    assert dict(material.credential_env) == {}
    assert material.provider_files["traex_auth_json"].endswith("/.trae/cli/auth.json")


def test_visible_employee_runtime_without_audit_fails_main_bot_outbound_closed() -> None:
    audit, failure = _main_bot_outbound_wiring(
        SimpleNamespace(main_bot_outbound_audit=None),
        required=True,
    )

    assert failure is None
    with pytest.raises(RuntimeError, match="audit.*unavailable"):
        audit("tenant-a", "reply", "om_message")


def test_dormant_employee_runtime_does_not_require_main_bot_outbound_audit() -> None:
    assert _main_bot_outbound_wiring(None, required=False) == (None, None)


def test_invalid_visible_employee_limit_shape_requires_outbound_audit() -> None:
    settings = SimpleNamespace(autonomous_visible_employee_limit=MagicMock())

    assert _visible_employee_runtime_requires_outbound_audit(settings) is True


def test_employee_registration_notifier_explains_pending_oauth_state(
    mock_ws_client: FeishuWSClient,
) -> None:
    runtime = mock_ws_client._employee_department_runtime
    state = SimpleNamespace(
        message_id="msg_hire",
        chat_id="chat_dm",
        requester_principal_id="ou_admin",
        tenant_key="tenant_test",
        employee_name="Atlas",
        intent_id="hire_intent_1",
    )
    reply_text = MagicMock()
    mock_ws_client._handler_ctx.handlers["coco"].reply_text = reply_text
    mock_ws_client._get_chat_mode = MagicMock(return_value="p2p")

    runtime._service._on_registration_status(state, "polling")

    reply_text.assert_called_once_with(
        "msg_hire",
        "独立飞书智能体注册请求已提交，正在等待你在上方链接中完成授权确认。"
        "确认前注册接口会持续返回 400 authorization_pending，这是设备授权"
        "流程的正常等待状态；请按链接完成授权，期间请勿重复发送 /hire。",
        idempotency_key=_employee_hire_status_uuid("hire_intent_1", "polling"),
    )


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
    programming_modes = ("coco", "claude", "aiden", "codex", "gemini", "traex")
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
):
    """The production flat post shape must preserve slash routing at ingress."""
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
