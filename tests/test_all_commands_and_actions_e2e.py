"""Unmocked handler-layer E2E regression tests for slash commands and card actions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config.settings import Settings
from src.feishu.action_registry import init_action_registry
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.utils.signing import sign_command


def _collect_active_callback_buttons(node: object) -> list[dict]:
    """Return every enabled non-link button element inside a Card JSON 2.0 tree."""
    buttons: list[dict] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if cur.get("tag") == "button":
                behaviors = cur.get("behaviors")
                is_open_url = isinstance(behaviors, list) and any(
                    isinstance(b, dict) and b.get("type") == "open_url" for b in behaviors
                )
                if not cur.get("disabled") and "multi_url" not in cur and not is_open_url:
                    buttons.append(cur)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return buttons


def _build_real_ws_graph(tmp_path: Path, monkeypatch):
    """Instantiate a real FeishuWSClient graph stubbing only HTTP/runtime boundaries."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        app_id="cli_test_e2e",
        app_secret="secret_test_e2e",
        default_working_dir=str(repo_dir),
        projects_dir=str(tmp_path / "projects"),
        projects_config_file=str(fake_home / ".ghostap" / "projects.json"),
        autonomous_data_dir=str(tmp_path / "autonomous"),
    )
    deliveries: list[tuple[str, str, str]] = []

    def _run_submitted_task(_spec, callback):
        callback(SimpleNamespace(run_id="run_e2e_1"))
        return SimpleNamespace(run_id="run_e2e_1")

    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()
    scheduler.list_tasks = MagicMock(return_value=[])
    scheduler.get_state = MagicMock(return_value=None)
    scheduler.submit = MagicMock(side_effect=_run_submitted_task)

    audit = SimpleNamespace(record_attempt=MagicMock(), mark_incomplete=MagicMock())

    def _runtime_stub(_settings, **kwargs):
        rt = SimpleNamespace(
            hire_service=None,
            fire_service=None,
            membership_service=None,
            data_composition=None,
            main_bot_outbound_audit=audit,
            readiness=lambda: SimpleNamespace(ready=True),
            bind_main_bot_warning_transport=MagicMock(),
            queue_main_bot_warning=MagicMock(return_value=True),
            _environment_provider=lambda _a: SimpleNamespace(credential_env={}, provider_files={}),
            close=MagicMock(),
        )
        rt._service = SimpleNamespace(_on_registration_status=kwargs.get("notification_status"))
        return rt

    with (
        patch("src.feishu.ws_client.get_settings", return_value=settings),
        patch("src.config.get_settings", return_value=settings),
        patch("src.feishu.ws_client._build_task_scheduler", return_value=scheduler),
        patch(
            "src.autonomous.provisioning.composition.EmployeeDepartmentRuntime.from_settings",
            side_effect=_runtime_stub,
        ),
    ):
        client = FeishuWSClient(lambda *_a: None)

    fake_im = SimpleNamespace(
        reply_message=lambda _mid, content, msg_type="text", **_kw: (
            deliveries.append(("reply", msg_type, content))
            or SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="ok",
                data=SimpleNamespace(message_id=f"om_reply_{len(deliveries)}"),
            )
        ),
        send_message=lambda _rt, _rid, content, msg_type="text", **_kw: (
            deliveries.append(("send", msg_type, content))
            or SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="ok",
                data=SimpleNamespace(message_id=f"om_send_{len(deliveries)}"),
            )
        ),
        patch_message=lambda _mid, content, **_kw: (
            deliveries.append(("patch", "interactive", content))
            or SimpleNamespace(success=lambda: True, code=0, msg="ok")
        ),
        add_reaction=lambda *_a, **_kw: SimpleNamespace(
            success=lambda: True, data=SimpleNamespace(reaction_id="r1")
        ),
        delete_reaction=lambda *_a, **_kw: SimpleNamespace(success=lambda: True),
    )

    client._handler_ctx._im_client = fake_im
    for handler in client._handler_ctx.handlers.values():
        handler._im_client = fake_im
        if hasattr(handler, "im_client"):
            try:
                handler.im_client = fake_im
            except AttributeError:
                pass
    system_emp = getattr(client._handler_ctx.handlers["system"], "employee", None)
    if system_emp is not None:
        system_emp._im_client = fake_im

    client._project_manager.create_project("demo", "DemoProj", str(repo_dir), chat_id="oc_e2e")
    project = client._project_manager.get_project_for_chat("demo", "oc_e2e")
    assert project is not None
    client._project_manager.set_active_project("oc_e2e", project.project_id)
    return client, project, deliveries, settings, repo_dir


def _make_message_event(text: str, idx: int, chat_id: str = "oc_e2e"):
    return SimpleNamespace(
        header=SimpleNamespace(tenant_key="tenant_e2e", event_id=f"ev_{idx}"),
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(open_id="ou_e2e", union_id="on_e2e", user_id="u_e2e"),
                sender_type="user",
            ),
            message=SimpleNamespace(
                message_id=f"om_e2e_{idx}",
                chat_id=chat_id,
                chat_type="p2p",
                message_type="text",
                content=json.dumps({"text": text}, ensure_ascii=False),
                create_time=str(int(time.time() * 1000)),
                mentions=[],
                root_id=None,
                parent_id=None,
            ),
        ),
    )


def _assert_deliveries_valid(label: str, deliveries: list[tuple[str, str, str]]) -> None:
    assert deliveries, f"{label} produced 0 deliveries (silent failure)"
    for _kind, msg_type, payload in deliveries:
        if msg_type == "interactive":
            card = json.loads(payload)
            assert card.get("schema") == "2.0", f"{label} card not schema 2.0"
            assert "elements" not in card, f"{label} card used legacy root elements"
            assert isinstance(card.get("body", {}).get("elements"), list)
            for btn in _collect_active_callback_buttons(card):
                behaviors = btn.get("behaviors")
                assert isinstance(behaviors, list) and behaviors, (
                    f"{label} button missing Card 2.0 behaviors: {btn}"
                )
                assert any(
                    isinstance(b, dict) and b.get("type") == "callback" for b in behaviors
                ), f"{label} button missing callback behavior: {btn}"


def test_all_commands_and_card_actions_deliver_end_to_end(tmp_path: Path, monkeypatch) -> None:
    client, project, deliveries, settings, _repo = _build_real_ws_graph(tmp_path, monkeypatch)
    agy_handler = client._handler_ctx.handlers["agy"]
    original_show_info = agy_handler.show_info
    show_info_calls: list[str] = []

    def _spy_show_info(mid, cid, proj=None):
        show_info_calls.append(mid)
        return original_show_info(mid, cid, proj)

    agy_handler.show_info = _spy_show_info
    agy_handler.handle_message = lambda mid, cid, txt, proj: deliveries.append(("prog", "text", txt))
    client._handler_ctx.handlers["workflow"].handle_workflow_command = (
        lambda mid, cid, txt, proj=None: deliveries.append(("workflow", "text", txt))
    )
    fake_agy_session = SimpleNamespace(
        session_id="sess_e2e_1",
        message_count=0,
        _model_name="claude-sonnet-4",
        get_available_commands=lambda: [],
        get_usage_info=lambda: None,
    )
    client._handler_ctx.agy_manager.get_session = MagicMock(return_value=fake_agy_session)
    client._handler_ctx.agy_manager.get_session_info = MagicMock(
        return_value={"session_id": "sess_e2e_1", "message_count": 0, "model": "claude-sonnet-4"}
    )
    client._handler_ctx.agy_manager.end_session = MagicMock()
    client._handler_ctx.handlers["system"]._enter_mode_with_acp_model = MagicMock(return_value=True)

    try:
        with (
            patch("src.feishu.handlers.acp_activation.fetch_acp_models", return_value=[]),
            patch("src.acp.helper.is_programming_tool_available", return_value=True),
            patch("src.feishu.handlers.system.is_programming_tool_available", return_value=True),
        ):
            smart_commands = [
                "/status",
                "/status all",
                "/info",
                "/cmds",
                "/cmds agy",
                "/tools",
                "/tools_status",
                "/acp",
                "/acp agy",
                "/model list",
                "/model default",
                "/stop",
                "/sh pwd",
                "/projects",
                "/help",
                "/menu",
                "/exit",
                "/logout",
            ]
            for idx, cmd in enumerate(smart_commands, start=1):
                client._mode_manager.exit_to_smart("oc_e2e", project_id=project.project_id)
                deliveries.clear()
                client._process_message_async(_make_message_event(cmd, idx))
                _assert_deliveries_valid(f"SMART:{cmd}", deliveries)

            # Seed unified context & model selection, then verify /clear & /reset persist cleanly
            project.add_conversation("user", "old turn", "m0")
            client._project_manager.commit_acp_configuration(
                project,
                tool_name="agy",
                model_name="claude-sonnet-4",
            )
            client._context_manager.update_context(
                project.project_id,
                conversation={"role": "user", "content": "previous task context"},
                chat_id="oc_e2e",
            )
            client._mode_manager.enter_mode("oc_e2e", InteractionMode.AGY, project_id=project.project_id)
            for idx, cmd in enumerate(["/info", "/clear", "/reset"], start=100):
                deliveries.clear()
                client._process_message_async(_make_message_event(cmd, idx))
                _assert_deliveries_valid(f"AGY:{cmd}", deliveries)
            assert len(show_info_calls) == 1
            assert project.conversation_history == []

            reloaded = client._project_manager.__class__(str(client._project_manager._storage_path))
            saved_proj = reloaded.get_project_for_chat(project.project_id, "oc_e2e")
            assert saved_proj is not None
            assert saved_proj.acp_tool_name == "agy"
            assert saved_proj.acp_model_name == "claude-sonnet-4"
            assert saved_proj.conversation_history == []

            # Verify card callbacks via init_action_registry
            registry = init_action_registry(client)
            action_payloads = [
                ("show_tool_commands", {"action": "show_tool_commands", "tool_name": "agy", "project_id": project.project_id}),
                (
                    "run_native_tool_command",
                    {"action": "run_native_tool_command", "tool_name": "agy", "command": "/compact", "project_id": project.project_id},
                ),
                ("open_acp_model_picker", {"action": "open_acp_model_picker", "tool_name": "agy", "project_id": project.project_id}),
                ("programming_stop", {"action": "programming_stop", "tool_name": "agy", "project_id": project.project_id}),
                ("continue_dev", {"action": "continue_dev", "project_id": project.project_id}),
            ]
            for action_name, val in action_payloads:
                deliveries.clear()
                registry[action_name]("om_card_1", "oc_e2e", project.project_id, val)
                _assert_deliveries_valid(f"ACTION:{action_name}", deliveries)

            # Verify retry_command routes /wf build through _message_dispatcher, NOT shell
            deliveries.clear()
            system = client._handler_ctx.handlers["system"]
            system.submit_shell_command = MagicMock()
            registry["retry_command"](
                "om_card_retry",
                "oc_e2e",
                project.project_id,
                {"action": "retry_command", "_t": "/wf build", "_s": sign_command("/wf build", "oc_e2e")},
            )
            system.submit_shell_command.assert_not_called()
            assert any(kind == "workflow" and payload == "/wf build" for kind, _mt, payload in deliveries)
    finally:
        client.close()
