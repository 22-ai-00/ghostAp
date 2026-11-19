import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.access_control import IngressAccessPolicy
from src.agent.intent_recognizer import IntentResult, IntentType, TaskStep
from src.config import IngressAccessMode
from src.feishu.image_handler import FeishuImageHandler, ImageDownloadResult
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.project import ProjectContext
from src.tasking import TaskPriority
from src.thread import set_current_thread_id
from src.trust.models import ActorKind, EffectiveTrust, TrustZone
from tests.helpers.ws_routing_harness import (
    create_mock_message,
    mock_ws_client,
)

__all__ = ["mock_ws_client"]


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
