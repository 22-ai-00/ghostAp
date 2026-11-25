"""E2E tests for /usage, /settings, ToolCockpitMixin callbacks, TUI guard, and runtime settings."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.acp.options import ACPModelOption, ACPModelSelectionVariant
from src.config.settings import Settings
from src.feishu.action_registry import init_action_registry
from src.feishu.handlers.programming import AGYModeHandler, ProgrammingExecutionMixin
from src.feishu.handlers.system import SystemHandler
from src.feishu.handlers.tool_cockpit import (
    OPEN_TOOL_SETTINGS,
    OPEN_TOOL_USAGE,
    REFRESH_TOOL_SETTINGS,
    REFRESH_TOOL_USAGE,
    SETTINGS_CLEAR_SESSION,
    SETTINGS_SELECT_AGENT_MODE,
    SETTINGS_SELECT_MODEL_EFFORT,
    SETTINGS_SELECT_MODEL_GROUP,
    SETTINGS_SELECT_MODEL_PROFILE,
    SETTINGS_SELECT_TOOL,
    SETTINGS_SELECT_VERBOSITY,
    SETTINGS_TOGGLE_SHOW_THINKING,
    TOOL_COCKPIT_ACTION_TYPES,
)
from src.feishu.product_catalog import (
    HIDDEN_SLASH_COMMANDS,
    get_slash_discoverable_actions,
    resolve_command,
)
from src.feishu.slash_command_parser import SlashCommandParser
from src.feishu.ws_client import FeishuWSClient
from src.mode import InteractionMode
from src.project.context import ProjectContext


def _build_real_ws_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a real FeishuWSClient graph isolating filesystem and IM delivery."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("APP_SECRET", "test_secret_tool_cockpit")
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        app_id="cli_test_cockpit",
        app_secret="test_secret_tool_cockpit",
        default_working_dir=str(repo_dir),
        projects_dir=str(tmp_path / "projects"),
        projects_config_file=str(fake_home / ".ghostap" / "projects.json"),
    )
    deliveries: list[tuple[str, str, str, str]] = []

    def _run_submitted_task(_spec, callback):
        callback(SimpleNamespace(run_id="run_cockpit_1"))
        return SimpleNamespace(run_id="run_cockpit_1")

    scheduler = MagicMock()
    scheduler._restart_gate = MagicMock()
    scheduler.list_tasks = MagicMock(return_value=[])
    scheduler.get_state = MagicMock(return_value=None)
    scheduler.submit = MagicMock(side_effect=_run_submitted_task)

    with (
        patch("src.feishu.ws_client.get_settings", return_value=settings),
        patch("src.config.get_settings", return_value=settings),
        patch("src.feishu.ws_client._build_task_scheduler", return_value=scheduler),
    ):
        client = FeishuWSClient(lambda *_a: None)

    patch_should_fail = [False]
    fake_im = SimpleNamespace(
        reply_message=lambda mid, content, msg_type="text", **_kw: (
            deliveries.append(("reply", mid, msg_type, content))
            or SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="ok",
                data=SimpleNamespace(message_id=f"om_reply_{len(deliveries)}"),
            )
        ),
        send_message=lambda _rt, rid, content, msg_type="text", **_kw: (
            deliveries.append(("send", rid, msg_type, content))
            or SimpleNamespace(
                success=lambda: True,
                code=0,
                msg="ok",
                data=SimpleNamespace(message_id=f"om_send_{len(deliveries)}"),
            )
        ),
        patch_message=lambda mid, content, **_kw: (
            SimpleNamespace(success=lambda: False, code=400, msg="patch failed")
            if patch_should_fail[0]
            else (
                deliveries.append(("patch", mid, "interactive", content))
                or SimpleNamespace(success=lambda: True, code=0, msg="ok")
            )
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

    client._project_manager.create_project("demo", "DemoProj", str(repo_dir), chat_id="oc_cockpit")
    project = client._project_manager.get_project_for_chat("demo", "oc_cockpit")
    assert project is not None
    client._project_manager.set_active_project("oc_cockpit", project.project_id)
    return client, project, deliveries, patch_should_fail


def _make_message_event(text: str, idx: int, chat_id: str = "oc_cockpit"):
    return SimpleNamespace(
        header=SimpleNamespace(tenant_key="tenant_cockpit", event_id=f"ev_cockpit_{idx}"),
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(open_id="ou_cockpit", union_id="on_cockpit", user_id="u_cockpit"),
                sender_type="user",
            ),
            message=SimpleNamespace(
                message_id=f"om_cockpit_{idx}",
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


def _sample_agy_models() -> list[ACPModelOption]:
    return [
        ACPModelOption(
            name="gemini-pro-orchestrator-neon",
            description="AGY Orchestrator Neon",
            is_default=True,
            selection_variants=(
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator-neon",
                    model="gemini-pro-orchestrator-neon",
                    profile="default",
                    effort="default",
                    is_default=True,
                ),
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator-neon/high/medium",
                    model="gemini-pro-orchestrator-neon",
                    profile="high",
                    effort="medium",
                ),
                ACPModelSelectionVariant(
                    name="gemini-pro-orchestrator-neon/xhigh/high",
                    model="gemini-pro-orchestrator-neon",
                    profile="xhigh",
                    effort="high",
                ),
            ),
        ),
        ACPModelOption(
            name="claude-opus-4-6",
            description="Claude Opus 4.6",
            is_default=False,
            selection_variants=(
                ACPModelSelectionVariant(
                    name="claude-opus-4-6/high",
                    model="claude-opus-4-6",
                    profile=None,
                    effort="high",
                    is_default=True,
                ),
            ),
        ),
    ]


def _install_fake_slice_ab_modules(monkeypatch: pytest.MonkeyPatch):
    """Ensure Slice A telemetry and Slice B builder interfaces are callable on any branch."""
    telemetry_calls: list[dict] = []
    builder_calls: list[dict] = []

    fake_telemetry = ModuleType("src.acp.tool_telemetry")

    def _collect_usage(tool_name, *, cwd, model_name=None, session=None, timeout_s=5.0):
        telemetry_calls.append(
            {
                "fn": "usage",
                "tool_name": tool_name,
                "cwd": cwd,
                "model_name": model_name,
                "session": session,
            }
        )
        return SimpleNamespace(
            tool_name=tool_name,
            model_name=model_name,
            session_id=getattr(session, "session_id", None),
            message_count=int(getattr(session, "message_count", 0) or 0),
            quota_groups=(),
            remaining_credits=120,
            capability_note="ok",
            error=None,
        )

    def _collect_settings(
        tool_name,
        *,
        cwd,
        model_name=None,
        session=None,
        saved_runtime_settings=None,
        timeout_s=4.0,
    ):
        rt = dict(saved_runtime_settings or {})
        telemetry_calls.append(
            {
                "fn": "settings",
                "tool_name": tool_name,
                "cwd": cwd,
                "model_name": model_name,
                "saved_runtime_settings": rt,
            }
        )
        return SimpleNamespace(
            tool_name=tool_name,
            model_name=model_name,
            show_thinking=bool(rt.get("show_thinking", True)),
            verbosity=str(rt.get("verbosity") or "medium"),
            agent_mode=str(rt.get("agent_mode") or "auto"),
            config_options=(),
            backend_config={},
            error=None,
        )

    fake_telemetry.collect_tool_usage_snapshot = _collect_usage  # type: ignore[attr-defined]
    fake_telemetry.collect_tool_settings_snapshot = _collect_settings  # type: ignore[attr-defined]

    fake_builder_mod = ModuleType("src.card.builders.tool_cockpit")

    class _FakeToolCockpitBuilder:
        @staticmethod
        def build_tool_usage_card(
            tool_name: str,
            usage_snapshot,
            *,
            project_name=None,
            project_id=None,
            thread_root_id=None,
        ):
            builder_calls.append(
                {
                    "fn": "usage_card",
                    "tool_name": tool_name,
                    "usage_snapshot": usage_snapshot,
                    "project_name": project_name,
                    "project_id": project_id,
                    "thread_root_id": thread_root_id,
                }
            )
            card = {
                "schema": "2.0",
                "config": {"wide_screen_mode": True, "update_multi": True},
                "header": {"title": {"tag": "plain_text", "content": f"USAGE:{tool_name}"}, "template": "blue"},
                "body": {
                    "elements": [
                        {"tag": "markdown", "content": f"tool={tool_name} model={getattr(usage_snapshot, 'model_name', '')}"}
                    ]
                },
            }
            return "interactive", json.dumps(card, ensure_ascii=False)

        @staticmethod
        def build_tool_settings_card(
            tool_name: str,
            *,
            models,
            current_model,
            available_tools,
            settings_snapshot=None,
            project_name=None,
            project_id=None,
            thread_root_id=None,
            status_banner="",
        ):
            builder_calls.append(
                {
                    "fn": "settings_card",
                    "tool_name": tool_name,
                    "current_model": current_model,
                    "available_tools": list(available_tools),
                    "settings_snapshot": settings_snapshot,
                    "project_name": project_name,
                    "project_id": project_id,
                    "thread_root_id": thread_root_id,
                    "status_banner": status_banner,
                }
            )
            card = {
                "schema": "2.0",
                "config": {"wide_screen_mode": True, "update_multi": True},
                "header": {"title": {"tag": "plain_text", "content": f"SETTINGS:{tool_name}"}, "template": "blue"},
                "body": {
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": (
                                f"tool={tool_name} model={current_model or 'default'} "
                                f"banner={status_banner} "
                                f"thinking={getattr(settings_snapshot, 'show_thinking', None)} "
                                f"verbosity={getattr(settings_snapshot, 'verbosity', None)} "
                                f"agent_mode={getattr(settings_snapshot, 'agent_mode', None)}"
                            ),
                        }
                    ]
                },
            }
            return "interactive", json.dumps(card, ensure_ascii=False)

    fake_builder_mod.ToolCockpitBuilder = _FakeToolCockpitBuilder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.acp.tool_telemetry", fake_telemetry)
    monkeypatch.setitem(sys.modules, "src.card.builders.tool_cockpit", fake_builder_mod)
    return telemetry_calls, builder_calls


def _attach_runtime_settings_methods_if_missing(project: ProjectContext) -> None:
    """Attach Slice B runtime settings methods when running on unmerged Slice C branch."""
    if callable(getattr(project, "get_tool_runtime_settings", None)) and callable(
        getattr(project, "set_tool_runtime_settings", None)
    ):
        return
    store: dict[tuple[str, str], dict[str, object]] = {}

    def _get(chat_id: str | None, tool_name: str) -> dict[str, object]:
        return dict(store.get((str(chat_id or "").strip(), str(tool_name or "").strip().lower()), {}))

    def _set(chat_id: str | None, tool_name: str, settings: dict | None) -> dict[str, object]:
        key = (str(chat_id or "").strip(), str(tool_name or "").strip().lower())
        store[key] = dict(settings or {})
        return dict(store[key])

    project.get_tool_runtime_settings = _get  # type: ignore[attr-defined]
    project.set_tool_runtime_settings = _set  # type: ignore[attr-defined]


def test_product_catalog_panel_invariance_and_usage_settings_aliases() -> None:
    discoverable = {a.command for a in get_slash_discoverable_actions()}
    assert len(discoverable) == 23
    assert "/usage" not in discoverable
    assert "/settings" not in discoverable
    assert {"/usage", "/settings"} <= HIDDEN_SLASH_COMMANDS

    for alias in ("/usage", "/quota", "/credits", "/cost"):
        resolved = resolve_command(alias)
        assert resolved is not None
        assert resolved.action.command == "/usage"
        assert resolved.action.protects_from_auto_activation is True
        assert resolved.action.slash_discoverable is False

    for alias in ("/settings", "/config"):
        resolved = resolve_command(alias)
        assert resolved is not None
        assert resolved.action.command == "/settings"
        assert resolved.action.protects_from_auto_activation is True
        assert resolved.action.slash_discoverable is False


def test_slash_usage_and_settings_dispatch_smart_and_programming_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, project, deliveries, _patch_fail = _build_real_ws_graph(tmp_path, monkeypatch)
    telemetry_calls, builder_calls = _install_fake_slice_ab_modules(monkeypatch)
    agy_handler = client._handler_ctx.handlers["agy"]
    agy_handler.handle_message = MagicMock()
    fake_agy_session = SimpleNamespace(
        session_id="sess_agy_1",
        message_count=4,
        _model_name="gemini-pro-orchestrator-neon",
        get_usage_info=lambda: None,
    )
    client._handler_ctx.agy_manager.get_session = MagicMock(return_value=fake_agy_session)

    try:
        with (
            patch("src.feishu.handlers.acp_activation.fetch_acp_models", return_value=_sample_agy_models()),
            patch("src.feishu.handlers.system.is_programming_tool_available", return_value=True),
        ):
            # SMART mode: all aliases & explicit tool arg route to SystemHandler
            for idx, cmd in enumerate(("/usage", "/quota", "/credits", "/cost", "/usage agy"), start=1):
                client._mode_manager.exit_to_smart("oc_cockpit", project_id=project.project_id)
                deliveries.clear()
                client._process_message_async(_make_message_event(cmd, idx))
                assert len(deliveries) == 1
                assert deliveries[0][0] == "reply"
                assert deliveries[0][2] == "interactive"
                card = json.loads(deliveries[0][3])
                assert card["schema"] == "2.0"

            for idx, (cmd, expected_tool) in enumerate(
                (("/settings", "coco"), ("/config", "coco"), ("/settings agy", "agy")),
                start=20,
            ):
                client._mode_manager.exit_to_smart("oc_cockpit", project_id=project.project_id)
                deliveries.clear()
                client._process_message_async(_make_message_event(cmd, idx))
                assert len(deliveries) == 1
                card = json.loads(deliveries[0][3])
                assert f"SETTINGS:{expected_tool}" in card["header"]["title"]["content"]

            # Active AGY programming mode: intercepted by GhostAP, never forwarded to agy_handler
            client._mode_manager.enter_mode("oc_cockpit", InteractionMode.AGY, project_id=project.project_id)
            for idx, cmd in enumerate(("/usage", "/quota", "/cost", "/settings", "/config"), start=40):
                deliveries.clear()
                client._process_message_async(_make_message_event(cmd, idx))
                assert len(deliveries) == 1
            agy_handler.handle_message.assert_not_called()

            # Unsupported tool argument returns error card cleanly
            deliveries.clear()
            client._process_message_async(_make_message_event("/usage nonexistent_tool", 60))
            assert len(deliveries) == 1
            assert "不支持的 ACP 工具" in deliveries[0][3]
    finally:
        client.close()

    assert any(call["fn"] == "usage" for call in telemetry_calls)
    assert any(call["fn"] == "settings" for call in telemetry_calls)
    assert any(call["fn"] == "usage_card" for call in builder_calls)
    assert any(call["fn"] == "settings_card" for call in builder_calls)


def test_tool_cockpit_fallback_when_optional_slices_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ToolCockpitMixin works cleanly even if unmerged Slice A/B modules cannot import."""
    client, project, deliveries, _patch_fail = _build_real_ws_graph(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "src.acp.tool_telemetry", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "src.card.builders.tool_cockpit", None)  # type: ignore[arg-type]

    try:
        with (
            patch("src.feishu.handlers.acp_activation.fetch_acp_models", return_value=[]),
            patch("src.feishu.handlers.system.is_programming_tool_available", return_value=True),
        ):
            system = client._handler_ctx.handlers["system"]
            system.show_tool_usage("m_fb_1", "oc_cockpit", "agy", project)
            system.show_tool_settings("m_fb_2", "oc_cockpit", "agy", project)
            assert len(deliveries) == 2
            for _kind, _mid, msg_type, raw in deliveries:
                assert msg_type == "interactive"
                card = json.loads(raw)
                assert card["schema"] == "2.0"
                assert card["config"]["update_multi"] is True
                assert "body" in card and "elements" in card["body"]
    finally:
        client.close()


def test_all_12_tool_cockpit_actions_e2e_select_static_and_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, project, deliveries, patch_should_fail = _build_real_ws_graph(tmp_path, monkeypatch)
    _attach_runtime_settings_methods_if_missing(project)
    _telemetry_calls, builder_calls = _install_fake_slice_ab_modules(monkeypatch)

    live_agy = SimpleNamespace(
        session_id="sess_agy_live",
        message_count=2,
        _model_name="gemini-pro-orchestrator-neon",
        _is_agy_cli_session=False,
        _runtime_settings={"show_thinking": True},
        set_model=MagicMock(return_value=True),
        set_runtime_settings=MagicMock(
            side_effect=lambda cfg: live_agy._runtime_settings.update(cfg)
        ),
        get_runtime_settings=lambda: dict(live_agy._runtime_settings),
        get_usage_info=lambda: None,
    )
    client._handler_ctx.agy_manager.get_session = MagicMock(return_value=live_agy)
    client._handler_ctx.agy_manager.end_session = MagicMock()
    client._handler_ctx.agy_manager.retire_session = MagicMock()

    models_by_tool = {
        "agy": _sample_agy_models(),
        "codex": [
            ACPModelOption(
                name="gpt-5.6-sol",
                description="GPT 5.6",
                is_default=True,
                selection_variants=(
                    ACPModelSelectionVariant(
                        name="gpt-5.6-sol/medium",
                        model="gpt-5.6-sol",
                        effort="medium",
                        is_default=True,
                    ),
                    ACPModelSelectionVariant(
                        name="gpt-5.6-sol/high",
                        model="gpt-5.6-sol",
                        effort="high",
                    ),
                ),
            )
        ],
    }

    try:
        with (
            patch(
                "src.feishu.handlers.acp_activation.fetch_acp_models",
                side_effect=lambda tool, **_kw: models_by_tool.get(tool, []),
            ),
            patch("src.feishu.handlers.system.invalidate_acp_model_cache") as mock_invalidate,
            patch("src.feishu.handlers.system.is_programming_tool_available", return_value=True),
        ):
            registry = init_action_registry(client)
            assert set(TOOL_COCKPIT_ACTION_TYPES) <= set(registry)

            # 1. OPEN_TOOL_USAGE replies a card; REFRESH_TOOL_USAGE patches in-place (with fallback to reply)
            deliveries.clear()
            registry[OPEN_TOOL_USAGE](
                "om_card_u1",
                "oc_cockpit",
                project.project_id,
                {"action": OPEN_TOOL_USAGE, "tool_name": "agy", "project_id": project.project_id},
            )
            assert deliveries[-1][0] == "reply"

            deliveries.clear()
            registry[REFRESH_TOOL_USAGE](
                "om_card_u2",
                "oc_cockpit",
                project.project_id,
                {"action": REFRESH_TOOL_USAGE, "tool_name": "agy", "project_id": project.project_id},
            )
            assert deliveries[-1][0] == "patch"
            assert deliveries[-1][1] == "om_card_u2"

            patch_should_fail[0] = True
            deliveries.clear()
            registry[REFRESH_TOOL_USAGE](
                "om_card_u3",
                "oc_cockpit",
                project.project_id,
                {"action": REFRESH_TOOL_USAGE, "tool_name": "agy", "project_id": project.project_id},
            )
            assert deliveries[-1][0] == "reply"
            patch_should_fail[0] = False

            # 2. OPEN_TOOL_SETTINGS & REFRESH_TOOL_SETTINGS
            deliveries.clear()
            registry[OPEN_TOOL_SETTINGS](
                "om_card_s1",
                "oc_cockpit",
                project.project_id,
                {"action": OPEN_TOOL_SETTINGS, "tool_name": "agy", "project_id": project.project_id},
            )
            assert deliveries[-1][0] == "reply"

            deliveries.clear()
            registry[REFRESH_TOOL_SETTINGS](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {"action": REFRESH_TOOL_SETTINGS, "tool_name": "agy", "project_id": project.project_id},
            )
            assert deliveries[-1][0] == "patch"
            mock_invalidate.assert_called_once_with("agy", project.root_path)

            # 3. Model cascade select_static callbacks via value["_option"]
            deliveries.clear()
            registry[SETTINGS_SELECT_MODEL_GROUP](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_SELECT_MODEL_GROUP,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "_option": "claude-opus-4-6",
                },
            )
            assert project.get_acp_model_name("oc_cockpit", tool_name="agy") == "claude-opus-4-6/high"
            live_agy.set_model.assert_called_with("claude-opus-4-6/high")
            assert deliveries[-1][0] == "patch"

            deliveries.clear()
            registry[SETTINGS_SELECT_MODEL_GROUP](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_SELECT_MODEL_GROUP,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "_option": "gemini-pro-orchestrator-neon",
                },
            )
            registry[SETTINGS_SELECT_MODEL_PROFILE](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_SELECT_MODEL_PROFILE,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "model_group": "gemini-pro-orchestrator-neon",
                    "_option": "high",
                },
            )
            registry[SETTINGS_SELECT_MODEL_EFFORT](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_SELECT_MODEL_EFFORT,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "model_group": "gemini-pro-orchestrator-neon",
                    "model_profile": "xhigh",
                    "_option": "high",
                },
            )
            assert (
                project.get_acp_model_name("oc_cockpit", tool_name="agy")
                == "gemini-pro-orchestrator-neon/xhigh/high"
            )
            # Switching from ACP session to AGY CLI orchestrator retires the ACP session cleanly
            client._handler_ctx.agy_manager.retire_session.assert_called_with(
                "oc_cockpit",
                project_id=project.project_id,
            )

            # 4. Runtime settings: show_thinking toggle, verbosity, agent_mode
            registry[SETTINGS_TOGGLE_SHOW_THINKING](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {"action": SETTINGS_TOGGLE_SHOW_THINKING, "tool_name": "agy", "project_id": project.project_id},
            )
            assert project.get_tool_runtime_settings("oc_cockpit", "agy")["show_thinking"] is False
            live_agy.set_runtime_settings.assert_called()

            registry[SETTINGS_SELECT_VERBOSITY](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_SELECT_VERBOSITY,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "_option": "high",
                },
            )
            registry[SETTINGS_SELECT_AGENT_MODE](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_SELECT_AGENT_MODE,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "_option": "plan",
                },
            )
            rt = project.get_tool_runtime_settings("oc_cockpit", "agy")
            assert rt == {"show_thinking": False, "verbosity": "high", "agent_mode": "plan"}
            assert live_agy.get_runtime_settings() == rt

            # 5. SETTINGS_CLEAR_SESSION clears session & context while preserving tool/model/runtime settings
            client._context_manager.update_context(
                project.project_id,
                conversation={"role": "user", "content": "keep-out"},
                chat_id="oc_cockpit",
            )
            deliveries.clear()
            registry[SETTINGS_CLEAR_SESSION](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {"action": SETTINGS_CLEAR_SESSION, "tool_name": "agy", "project_id": project.project_id},
            )
            client._handler_ctx.agy_manager.end_session.assert_called()
            assert project.get_acp_tool_name("oc_cockpit") == "agy"
            assert project.get_acp_model_name("oc_cockpit", tool_name="agy") == "gemini-pro-orchestrator-neon/xhigh/high"
            assert project.get_tool_runtime_settings("oc_cockpit", "agy") == rt
            assert builder_calls[-1]["status_banner"] == "✅ 已清空当前会话上下文"
            assert deliveries[-1][0] == "patch"

            # 6. SETTINGS_SELECT_TOOL switches active tool + active programming mode without leaking AGY model
            client._mode_manager.enter_mode("oc_cockpit", InteractionMode.AGY, project_id=project.project_id)
            deliveries.clear()
            registry[SETTINGS_SELECT_TOOL](
                "om_card_s2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_SELECT_TOOL,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "_option": "codex",
                },
            )
            assert project.get_acp_tool_name("oc_cockpit") == "codex"
            assert project.get_acp_model_name("oc_cockpit", tool_name="codex") is None
            assert client._mode_manager.get_mode("oc_cockpit", project_id=project.project_id) == InteractionMode.CODEX
            assert builder_calls[-1]["tool_name"] == "codex"
            assert deliveries[-1][0] == "patch"

            # 7. Cross-chat unauthorized project_id fails closed
            deliveries.clear()
            registry[SETTINGS_SELECT_VERBOSITY](
                "om_card_foreign",
                "oc_cockpit",
                "foreign_project_id",
                {
                    "action": SETTINGS_SELECT_VERBOSITY,
                    "tool_name": "agy",
                    "project_id": "foreign_project_id",
                    "_option": "low",
                },
            )
            assert deliveries[-1][0] == "reply"
            assert "保存失败" in deliveries[-1][3]
    finally:
        client.close()


def test_programming_handler_applies_saved_runtime_settings_to_new_and_reused_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.thread.get_current_thread_id", lambda: None)

    project = ProjectContext(project_id="p_rt", project_name="demo", root_path=str(tmp_path))
    saved_rt = {"show_thinking": False, "verbosity": "low", "agent_mode": "plan"}
    project.get_tool_runtime_settings = (  # type: ignore[attr-defined]
        lambda chat_id, tool_name: dict(saved_rt) if chat_id == "c_rt" and tool_name == "agy" else {}
    )

    new_session = SimpleNamespace(
        session_id="sess_new",
        message_count=0,
        _model_name="gemini-pro-orchestrator-neon/high/high",
        set_runtime_settings=MagicMock(),
    )
    reused_session = SimpleNamespace(
        session_id="sess_reused",
        message_count=1,
        _model_name="gemini-pro-orchestrator-neon/high/high",
        set_runtime_settings=MagicMock(),
    )

    mgr = MagicMock()
    mgr.get_session.return_value = None
    mgr.ensure_session.return_value = new_session

    handler = object.__new__(AGYModeHandler)
    handler.ctx = SimpleNamespace(
        settings=SimpleNamespace(acp_startup_timeout=20, default_reply_mode="reply", ref_note_enabled=False),
        mode_manager=SimpleNamespace(
            get_mode=lambda *_a, **_kw: InteractionMode.SMART,
            enter_programming_mode=lambda *_a, **_kw: None,
        ),
        project_manager=SimpleNamespace(validate_project_path=lambda _pid: (True, str(tmp_path))),
        context_manager=SimpleNamespace(update_context=MagicMock(), store=SimpleNamespace(get=lambda *_a, **_kw: None)),
        agy_manager=mgr,
        handlers={},
    )
    handler._scoped_models = {}
    handler._scoped_models_lock = MagicMock()
    handler._active_programming_runs = {}
    handler._active_programming_runs_lock = MagicMock()
    handler.reply_text = MagicMock()
    handler.reply_card = MagicMock()
    handler.send_error_card = MagicMock()
    handler.add_reaction = MagicMock()
    handler.register_message_project = MagicMock()
    handler.get_working_dir = lambda _c: str(tmp_path)

    # 1. New session started via enter_mode receives saved runtime settings
    assert handler.enter_mode("m1", "c_rt", silent=True, project=project) is True
    new_session.set_runtime_settings.assert_called_once_with(saved_rt)

    # 2. Reused session in enter_mode receives saved runtime settings
    handler.ctx.mode_manager.get_mode = lambda *_a, **_kw: InteractionMode.AGY
    mgr.get_session.return_value = reused_session
    assert handler.enter_mode("m2", "c_rt", silent=True, project=project) is True
    reused_session.set_runtime_settings.assert_called_once_with(saved_rt)

    # 3. Reused session in handle_message receives saved runtime settings before execution
    reused_session.set_runtime_settings.reset_mock()
    with patch.object(
        ProgrammingExecutionMixin,
        "handle_message",
        return_value=None,
    ) as super_handle:
        handler.handle_message("m3", "c_rt", "hello", project)
        super_handle.assert_called_once()
    reused_session.set_runtime_settings.assert_called_once_with(saved_rt)


def test_select_static_option_extraction_accepts_all_callback_shapes() -> None:
    system = object.__new__(SystemHandler)
    assert system._extract_selected_option({"_option": "codex"}) == "codex"
    assert system._extract_selected_option({"_option": {"value": "agy"}}) == "agy"
    assert system._extract_selected_option({"selected_option": "high"}) == "high"
    assert system._extract_selected_option({"option": "plan"}) == "plan"
    assert system._extract_selected_option({}) is None
    assert SlashCommandParser.parse("/quota agy").command == "/usage"


def test_show_thinking_toggle_honors_card_payload_without_live_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, project, deliveries, _patch_should_fail = _build_real_ws_graph(tmp_path, monkeypatch)
    _attach_runtime_settings_methods_if_missing(project)
    _telemetry_calls, builder_calls = _install_fake_slice_ab_modules(monkeypatch)
    client._handler_ctx.agy_manager.get_session = MagicMock(return_value=None)
    client._handler_ctx.codex_manager.get_session = MagicMock(return_value=None)

    try:
        with (
            patch("src.feishu.handlers.acp_activation.fetch_acp_models", return_value=_sample_agy_models()),
            patch("src.feishu.handlers.system.is_programming_tool_available", return_value=True),
        ):
            registry = init_action_registry(client)

            # Case 1: card button sends legacy payload `show_thinking=True` without saved state or live session
            registry[SETTINGS_TOGGLE_SHOW_THINKING](
                "om_card_rt1",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_TOGGLE_SHOW_THINKING,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "show_thinking": True,
                },
            )
            assert project.get_tool_runtime_settings("oc_cockpit", "agy")["show_thinking"] is True
            assert builder_calls[-1]["status_banner"] == "✅ 已开启思考过程显示"
            assert deliveries[-1][0] == "patch"

            # Case 2: fresh untouched tool (`codex`) with payload `show_thinking=False` persists False cleanly
            registry[SETTINGS_TOGGLE_SHOW_THINKING](
                "om_card_rt2",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_TOGGLE_SHOW_THINKING,
                    "tool_name": "codex",
                    "project_id": project.project_id,
                    "show_thinking": False,
                },
            )
            assert project.get_tool_runtime_settings("oc_cockpit", "codex")["show_thinking"] is False
            assert builder_calls[-1]["status_banner"] == "✅ 已关闭思考过程显示"

            # Case 3: explicit next_show_thinking/current_show_thinking card payload toggles accurately
            registry[SETTINGS_TOGGLE_SHOW_THINKING](
                "om_card_rt3",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_TOGGLE_SHOW_THINKING,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "show_thinking": True,
                    "next_show_thinking": True,
                    "current_show_thinking": False,
                },
            )
            assert project.get_tool_runtime_settings("oc_cockpit", "agy")["show_thinking"] is True
            assert builder_calls[-1]["status_banner"] == "✅ 已开启思考过程显示"

            registry[SETTINGS_TOGGLE_SHOW_THINKING](
                "om_card_rt3",
                "oc_cockpit",
                project.project_id,
                {
                    "action": SETTINGS_TOGGLE_SHOW_THINKING,
                    "tool_name": "agy",
                    "project_id": project.project_id,
                    "show_thinking": False,
                    "next_show_thinking": False,
                    "current_show_thinking": True,
                },
            )
            assert project.get_tool_runtime_settings("oc_cockpit", "agy")["show_thinking"] is False
            assert builder_calls[-1]["status_banner"] == "✅ 已关闭思考过程显示"
    finally:
        client.close()
