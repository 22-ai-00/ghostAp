import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.agent.intent_recognizer import IntentResult, IntentType, TaskStep
from src.config import get_settings
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
from src.thread import set_current_thread_id


@pytest.fixture
def mock_ws_client(tmp_path: Path):
    # Patch heavy components and side-effects to keep tests fast and isolated
    settings = get_settings().model_copy(deep=True)
    settings.autonomous_state_dir = str(tmp_path / "state")
    settings.autonomous_journal_dir = str(tmp_path / "journal")
    settings.autonomous_anchor_path = str(tmp_path / "journal.anchor")
    settings.autonomous_credential_dir = str(tmp_path / "credentials")
    settings.autonomous_data_blob_dir = str(tmp_path / "data-blobs")
    settings.autonomous_employee_ingress_blob_dir = str(tmp_path / "ingress-blobs")
    settings.autonomous_employee_outbox_blob_dir = str(tmp_path / "outbox-blobs")
    settings.autonomous_employee_attachment_staging_dir = str(tmp_path / "attachments")
    settings.autonomous_main_bot_audit_dir = str(tmp_path / "main-bot-audit")
    settings.autonomous_main_bot_audit_anchor_path = str(tmp_path / "main-bot-audit.anchor")
    settings.autonomous_visible_employee_limit = 8
    settings.autonomous_journal_hmac_key = SecretStr("")
    settings.autonomous_credential_keys = SecretStr("")
    settings.autonomous_credential_active_key_id = ""
    settings.autonomous_data_keys = SecretStr("")
    settings.autonomous_data_active_key_id = ""
    # Routing tests exercise downstream behavior, so their synthetic events
    # must cross the production deny-by-default ingress contract explicitly.
    settings.admin_user_ids = frozenset({"ou_admin"})
    settings.allowed_user_ids = frozenset({"ou_test", "ou_user"})
    settings.allowed_chat_ids = frozenset({"oc_456", "oc_dm"})

    with patch("src.feishu.ws_client.ACPSessionManager"), \
         patch("src.feishu.ws_client.configure_logging_with_trace"), \
         patch("src.feishu.ws_client.get_settings", return_value=settings), \
         patch(
             "src.autonomous.provisioning.composition.default_slock_storage_base",
             return_value=str(tmp_path / "slock"),
         ):

        def dummy_callback(*args, **kwargs):
            pass

        client = FeishuWSClient(message_callback=dummy_callback)

        # This fixture exercises the legacy downstream router in isolation.
        # Production startup performs durable recovery before opening WS, and
        # a real Registry is covered by managed-group ingress tests.
        client._recover_employee_runtime_after_handler_binding()
        client._managed_group_registry = None

        # Patch the intent recognizer dynamically for tests
        client._intent_recognizer = MagicMock()

        # Patch the scheduler to intercept task submissions without real execution
        client._scheduler.submit = MagicMock()

        # Mock out message duplicate check to always pass
        client._message_cache.is_duplicate = MagicMock(return_value=False)

        # Block real Feishu API calls (add_reaction triggers real HTTP requests)
        client._add_reaction = MagicMock()

        yield client
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
    mock_ws_client._reply_text = MagicMock()
    mock_ws_client._get_chat_mode = MagicMock(return_value="p2p")

    runtime._service._on_registration_status(state, "polling")

    mock_ws_client._reply_text.assert_called_once_with(
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


def test_project_chat_programming_mode_is_not_stolen_by_slock_managed_chat(mock_ws_client: FeishuWSClient):
    """项目群已在普通编程态时，Slock managed 标记不能抢走自由文本。"""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._project_manager.find_by_bound_chat_id = MagicMock(return_value=project)
    mock_ws_client._mode_manager.set_mode("chat_456", InteractionMode.COCO, project_id="proj_1")
    mock_ws_client._slock_engine_manager.register_managed_chat("chat_456")
    mock_ws_client._coco_handler = MagicMock()
    mock_ws_client._handle_slock_message = MagicMock()

    mock_ws_client._dispatch_message_logic(
        "msg_prog_slock",
        "chat_456",
        "继续修复项目群编程",
        project,
        None,
        command_match=None,
    )

    mock_ws_client._handle_slock_message.assert_not_called()
    mock_ws_client._coco_handler.handle_message.assert_called_once_with(
        "msg_prog_slock",
        "chat_456",
        "继续修复项目群编程",
        project,
    )


def test_explicit_engine_command_reaches_its_final_handler_in_every_programming_mode(
    mock_ws_client: FeishuWSClient,
):
    """Persistent programming state may not consume any explicit engine command."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._get_mode_handler = MagicMock()
    programming_modes = ("coco", "claude", "aiden", "codex", "gemini", "traex")
    engine_cases = (
        ("/deep 深入完成复杂任务", "_handle_deep_command"),
        ("/spec 按规格迭代直到收敛", "_handle_spec_command"),
        ("/wf 编排多个代理完成任务", "_handle_workflow_command"),
    )

    for text, expected_handler in engine_cases:
        for programming_mode in programming_modes:
            target = MagicMock()
            setattr(mock_ws_client, expected_handler, target)

            mock_ws_client._dispatch_message_logic(
                "msg_engine_final",
                "chat_456",
                text,
                project,
                programming_mode,
                command_match=SlashCommandParser.parse(text),
            )

            assert target.call_count == 1, (
                f"{text!r} did not reach {expected_handler!r} from "
                f"{programming_mode!r}"
            )
            assert not mock_ws_client._get_mode_handler.called, (
                f"{text!r} fell back to {programming_mode!r}"
            )


@pytest.mark.parametrize(
    ("engine", "expected_method"),
    [
        ("deep", "_start_deep_engine"),
        ("spec", "_start_spec_engine"),
    ],
)
def test_deep_and_spec_topic_plain_text_keeps_engine_strategy(
    mock_ws_client: FeishuWSClient,
    engine: str,
    expected_method: str,
):
    """Deep/Spec topic continuation should not fall back to SMART intent routing."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._process_with_intent = MagicMock()
    setattr(mock_ws_client, expected_method, MagicMock())

    mock_ws_client._dispatch_message_logic(
        "msg_next",
        "chat_456",
        "继续按这个方向做",
        project,
        engine,
        command_match=None,
    )

    getattr(mock_ws_client, expected_method).assert_called_once_with(
        "msg_next",
        "chat_456",
        "继续按这个方向做",
        project,
    )
    mock_ws_client._process_with_intent.assert_not_called()


def test_topic_engine_without_resolved_project_never_falls_back_to_smart(
    mock_ws_client: FeishuWSClient,
):
    """A topic-owned engine resolves/rejects its project instead of changing strategy."""
    mock_ws_client._process_with_intent = MagicMock()
    mock_ws_client._reply_text = MagicMock()
    engine_cases = (
        ("deep", "_start_deep_engine"),
        ("spec", "_start_spec_engine"),
        ("workflow", "_workflow_handler.handle_message"),
    )
    slash_texts = {
        "deep": "/deep 继续执行",
        "spec": "/spec 继续执行",
        "workflow": "/wf 继续执行",
    }

    for engine, expected_method in engine_cases:
        for has_slash_command in (False, True):
            mock_ws_client._process_with_intent.reset_mock()
            mock_ws_client._reply_text.reset_mock()
            if expected_method == "_workflow_handler.handle_message":
                target = MagicMock()
                mock_ws_client._workflow_handler.handle_message = target
            else:
                target = MagicMock()
                setattr(mock_ws_client, expected_method, target)

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
            assert mock_ws_client._reply_text.call_count == 1, (
                f"{engine!r} did not explain the missing project"
            )
            assert "未执行" in mock_ws_client._reply_text.call_args.args[1]
            assert not mock_ws_client._process_with_intent.called, (
                f"{engine!r} fell back to SMART"
            )


def test_missing_topic_project_allows_safe_recovery_and_diagnostics_commands(
    mock_ws_client: FeishuWSClient,
):
    mock_ws_client._process_with_intent = MagicMock()
    mock_ws_client._reply_text = MagicMock()
    commands = (
        "/projects",
        "/status",
        "/help",
        "/deep_status --all",
        "/stop_deep --all",
    )

    for text in commands:
        mock_ws_client._process_with_intent.reset_mock()
        mock_ws_client._reply_text.reset_mock()
        mock_ws_client._dispatch_message_logic(
            "msg_recover",
            "chat_456",
            text,
            None,
            "deep",
            command_match=SlashCommandParser.parse(text),
        )

        assert mock_ws_client._process_with_intent.call_count == 1, (
            f"{text!r} was not routed as a safe recovery command"
        )
        assert not mock_ws_client._reply_text.called, (
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


def test_group_ledger_publication_precedes_programming_mode_dispatch(
    mock_ws_client: FeishuWSClient,
) -> None:
    msg = create_mock_message("implement this")
    msg.event.message.chat_type = "group"
    msg.event.message.root_id = "om_topic_root"
    msg.event.message.thread_id = "omt_topic"
    mock_ws_client._validate_message = MagicMock(return_value=True)
    project = ProjectContext("proj_1", "Test", "/tmp")
    mock_ws_client._resolve_message_context = MagicMock(
        return_value=(project, "coco")
    )
    mock_coco_handler = MagicMock()
    mock_ws_client._get_mode_handler = MagicMock(return_value=mock_coco_handler)
    runtime = mock_ws_client._employee_department_runtime
    runtime.record_group_event = MagicMock(return_value=True)
    task_ctx = SimpleNamespace(
        run_id="run_group_ledger",
        spec=SimpleNamespace(
            sender_id="ou_user",
            sender_union_id="on_user",
            is_p2p=False,
            tenant_key="tenant_1",
        )
    )

    mock_ws_client._process_message_async(msg, task_ctx=task_ctx)

    runtime.record_group_event.assert_called_once_with(
        tenant_key="tenant_test",
        chat_id="oc_456",
        thread_id="omt_topic",
        message_id="om_123",
        sender_id="ou_test",
        text="implement this",
    )
    mock_coco_handler.handle_message.assert_called_once()


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
    mock_ws_client._employee_department_runtime.record_group_event = MagicMock(
        return_value=True
    )
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
    mock_ws_client._reply_text = MagicMock()
    mock_ws_client._process_with_intent = MagicMock()

    mock_ws_client._dispatch_message_logic(
        "msg_123",
        "chat_456",
        "/spec rewrite this",
        project,
        "deep",
        command_match=MagicMock(command="/spec"),
    )

    mock_ws_client._reply_text.assert_called_once()
    assert "Deep" in mock_ws_client._reply_text.call_args.args[1]
    assert "Spec" in mock_ws_client._reply_text.call_args.args[1]
    mock_ws_client._process_with_intent.assert_not_called()


def test_topic_bound_spec_allows_spec_command(mock_ws_client: FeishuWSClient):
    """Same-engine explicit commands remain available inside their topic."""
    project = ProjectContext("proj_1", "Test", "/tmp")
    mock_ws_client._reply_text = MagicMock()
    mock_ws_client._process_with_intent = MagicMock()
    mock_ws_client._is_interceptable_command_match = MagicMock(return_value=False)

    mock_ws_client._dispatch_message_logic(
        "msg_123",
        "chat_456",
        "/spec_status",
        project,
        "spec",
        command_match=MagicMock(command="/spec_status"),
    )

    mock_ws_client._reply_text.assert_not_called()
    mock_ws_client._process_with_intent.assert_called_once()


def test_deep_start_binds_topic_context(mock_ws_client: FeishuWSClient):
    """Starting Deep registers the current Feishu topic as a Deep strategy context."""
    project = ProjectContext("proj_1", "GhostAP", "/tmp")
    mock_ws_client._thread_manager.remove("msg_deep")
    mock_ws_client._deep_handler._submit_engine_task = MagicMock()
    mock_ws_client._deep_handler.add_reaction = MagicMock()
    mock_ws_client._deep_handler.ensure_request_id = MagicMock(return_value="req-1")
    mock_ws_client._deep_handler.ctx.deep_engine_manager.get = MagicMock(return_value=None)
    mock_ws_client._deep_handler.ctx.deep_engine_manager.get_or_create = MagicMock(return_value=MagicMock())

    set_current_thread_id(None)
    try:
        mock_ws_client._deep_handler.start_deep_engine("msg_deep", "chat_456", "深入分析", project)
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
    mock_ws_client._spec_handler._submit_engine_task = MagicMock()
    mock_ws_client._spec_handler.add_reaction = MagicMock()
    mock_ws_client._spec_handler.ensure_request_id = MagicMock(return_value="req-1")
    mock_ws_client._spec_handler.ctx.spec_engine_manager.get = MagicMock(return_value=None)
    mock_ws_client._spec_handler.ctx.spec_engine_manager.get_or_create = MagicMock(return_value=MagicMock())

    set_current_thread_id(None)
    try:
        mock_ws_client._spec_handler.start_spec_engine("msg_spec", "chat_456", "写清规格", project)
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
    mock_ws_client._system_handler.reply_text = MagicMock()
    mock_ws_client._control_plane.should_defer_exit = MagicMock(return_value=False)

    set_current_thread_id("thread-deep-exit")
    try:
        mock_ws_client._dispatch_message_logic(
            "msg_exit",
            "chat_456",
            "/exit",
            project,
            "deep",
            command_match=MagicMock(command="/exit"),
        )
    finally:
        set_current_thread_id(None)

    assert mock_ws_client._thread_manager.get("thread-deep-exit") is None
    mock_ws_client._system_handler.reply_text.assert_called_once()


def test_process_with_intent_multitask(mock_ws_client: FeishuWSClient):
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

    # Mock message reply and task steps
    mock_ws_client._reply_text = MagicMock()
    mock_ws_client._message_dispatcher.execute_task_step = MagicMock(return_value=True)

    mock_ws_client._process_with_intent("msg_123", "chat_456", "create a project and enter coco", project)

    # It should reply with a multi-task plan
    assert mock_ws_client._reply_text.call_count >= 1
    # It should have called _execute_task_step for each task
    assert mock_ws_client._message_dispatcher.execute_task_step.call_count == 2

    call_args_list = mock_ws_client._message_dispatcher.execute_task_step.call_args_list
    assert call_args_list[0][0][2].intent == IntentType.CREATE_PROJECT
    assert call_args_list[1][0][2].intent == IntentType.ENTER_COCO


def test_process_with_intent_system_command_interception(mock_ws_client: FeishuWSClient):
    """Test that system commands bypass intent recognition completely during SMART mode."""
    project = ProjectContext("proj_1", "Test", "/tmp")
    mock_ws_client._get_effective_mode = MagicMock(return_value=(InteractionMode.SMART, False))

    mock_ws_client._handle_deep_command = MagicMock()

    # Send a deep engine command
    mock_ws_client._process_with_intent("msg_123", "chat_456", "/deep something", project)

    # Intent recognizer must not be called
    mock_ws_client._intent_recognizer.recognize.assert_not_called()
    # It should be directly routed to handle_deep_command
    mock_ws_client._handle_deep_command.assert_called_once_with("msg_123", "chat_456", "/deep something", project)


def test_card_action_deduplication_and_routing(mock_ws_client: FeishuWSClient):
    """Test card action callback ignores duplicates and routes correctly via ActionDispatcher."""
    # Create fake card action data
    data = MagicMock()
    data.header.event_id = "event_001"
    data.header.tenant_key = "tenant_card"
    data.event.context.open_message_id = "msg_123"
    data.event.context.open_chat_id = "chat_456"
    data.event.action.value = '{"action": "show_status", "project_id": "proj_1"}'
    data.event.operator.open_id = "ou_test"
    data.event.operator.user_id = "u_test"
    data.event.operator.union_id = "on_test"

    # Mock deduplication cache to False
    mock_ws_client._card_event_cache.is_duplicate = MagicMock(return_value=False)

    # Inject action dispatcher spy
    mock_ws_client._action_dispatcher.dispatch = MagicMock(return_value=True)

    mock_ws_client._handle_card_action(data)

    # The action should be submitted as a task
    submit_mock = mock_ws_client._scheduler.submit
    assert submit_mock.call_count == 1
    spec, func = submit_mock.call_args[0]

    assert spec.task_type == "feishu_card_action"
    assert spec.tenant_key == "tenant_card"
    assert spec.project_id == "proj_1"
    assert spec.sender_union_id == "on_test"
    # System card actions like show_status are HIGH priority and is_system_command
    assert spec.priority == TaskPriority.HIGH
    assert spec.is_system_command is True

    # Now run the callback to verify dispatcher routing
    task_ctx = MagicMock()
    func(task_ctx)

    # ActionDispatcher should have received the decoded value
    mock_ws_client._action_dispatcher.dispatch.assert_called_once()
    args, kwargs = mock_ws_client._action_dispatcher.dispatch.call_args
    assert args[0] == "show_status"
    assert args[1] == "msg_123"
    assert args[2] == "chat_456"
    assert args[3] == "proj_1"
    assert args[4]["action"] == "show_status"


def _card_action_data(*, event_id: str, message_id: str, chat_id: str, operator_id: str):
    data = MagicMock()
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


def test_card_action_fallback_uses_chat_mode_not_visibility_chat_type(
    mock_ws_client: FeishuWSClient,
):
    """After an in-memory provenance miss, Chat API chat_mode is authoritative."""
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
    assert spec.is_p2p is True
    request = api_client.im.v1.chat.get.call_args.args[0]
    assert request.chat_id == "chat_dm"
    fallback_origin = mock_ws_client._message_linker.query("msg_card_after_restart")
    assert fallback_origin is not None
    assert fallback_origin["chat_id"] == "chat_dm"
    assert fallback_origin["sender_id"] == "ou_admin"
    assert fallback_origin["chat_type"] == "p2p"

    mock_ws_client._message_linker.link_reply(
        "msg_card_after_restart",
        "msg_next_hire_card",
    )
    api_client.im.v1.chat.get.reset_mock()
    next_data = _card_action_data(
        event_id="evt_after_restart_next_step",
        message_id="msg_next_hire_card",
        chat_id="chat_dm",
        operator_id="ou_admin",
    )
    mock_ws_client._handle_card_action(next_data)

    next_spec, _ = mock_ws_client._scheduler.submit.call_args.args
    assert next_spec.is_p2p is True
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


def test_card_action_api_fallback_requires_atomic_provenance_write(
    mock_ws_client: FeishuWSClient,
):
    response = MagicMock()
    response.success.return_value = True
    response.data.chat_mode = "p2p"
    api_client = MagicMock()
    api_client.im.v1.chat.get.return_value = response
    mock_ws_client._get_api_client = MagicMock(return_value=api_client)
    mock_ws_client._message_linker = MagicMock()
    mock_ws_client._message_linker.query.return_value = None
    mock_ws_client._message_linker.register_trusted_origin_if_absent.return_value = None

    assert mock_ws_client._resolve_card_is_p2p(
        origin_message_id="msg_card",
        open_chat_id="chat_dm",
        operator_id="ou_admin",
    ) is False


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
