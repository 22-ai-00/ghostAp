"""Tests for model switching when already in a programming mode.

Covers:
- switch_model() on ProgrammingModeHandler: ACP-protocol path (set_model succeeds)
- switch_model() fallback path (set_model fails → session restart)
- switch_model() no existing session (direct ensure_session)
- _enter_mode_with_acp_model() routes to switch_model() when mode is active
- _enter_mode_with_acp_model() routes to enter_mode() when not in mode
- ensure_session() restarts on model mismatch (non-TTADK)
- ACPSession.set_model() / SyncACPSession.set_model() delegation
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.project.context import ProjectContext


# ---------------------------------------------------------------------------
# ACPSession.set_model()
# ---------------------------------------------------------------------------
class TestACPSessionSetModel(unittest.IsolatedAsyncioTestCase):
    async def test_set_model_calls_conn(self):
        from src.acp.session import ACPSession

        session = ACPSession.__new__(ACPSession)
        session._agent_cmd = "coco"
        session._conn = AsyncMock()
        session._conn.set_config_option = None
        session._conn.setConfigOption = None
        session._conn.set_session_config_option = None
        session._conn.setSessionConfigOption = None
        session._conn._conn = None
        session._session_id = "sess-abc123"

        result = await session.set_model("gpt-5.2")

        session._conn.set_session_model.assert_awaited_once_with(
            model_id="gpt-5.2", session_id="sess-abc123"
        )
        assert result is True

    async def test_set_model_returns_false_on_error(self):
        from src.acp.session import ACPSession

        session = ACPSession.__new__(ACPSession)
        session._agent_cmd = "coco"
        session._conn = AsyncMock()
        session._conn.set_config_option = None
        session._conn.setConfigOption = None
        session._conn.set_session_config_option = None
        session._conn.setSessionConfigOption = None
        session._conn._conn = None
        session._conn.set_session_model.side_effect = RuntimeError("not supported")
        session._session_id = "sess-abc123"

        result = await session.set_model("gpt-5.2")
        assert result is False

    async def test_set_model_prefers_config_option_protocol(self):
        from src.acp.session import ACPSession

        session = ACPSession.__new__(ACPSession)
        session._agent_cmd = "npx"
        session._conn = AsyncMock()
        session._session_id = "sess-abc123"

        result = await session.set_model("gpt-5.6-sol")

        session._conn.set_config_option.assert_awaited_once_with(
            session_id="sess-abc123",
            config_id="model",
            value="gpt-5.6-sol",
        )
        session._conn.set_session_model.assert_not_awaited()
        assert result is True

    async def test_set_model_config_option_failure_does_not_fall_back_to_set_model(self):
        from src.acp.session import ACPSession

        session = ACPSession.__new__(ACPSession)
        session._agent_cmd = "npx"
        session._conn = AsyncMock()
        session._conn.set_config_option.side_effect = RuntimeError("Internal error")
        session._session_id = "sess-abc123"

        result = await session.set_model("gpt-5.6-sol")

        session._conn.set_config_option.assert_awaited_once()
        session._conn.set_session_model.assert_not_awaited()
        assert result is False

    async def test_set_config_option_uses_requested_config_id(self):
        from src.acp.session import ACPSession

        session = ACPSession.__new__(ACPSession)
        session._agent_cmd = "npx"
        session._conn = AsyncMock()
        session._session_id = "sess-abc123"

        with self.assertLogs("src.acp.session", level="INFO") as logs:
            result = await session.set_config_option(
                "future_sensitive_option",
                "secret-value",
            )

        session._conn.set_config_option.assert_awaited_once_with(
            session_id="sess-abc123",
            config_id="future_sensitive_option",
            value="secret-value",
        )
        assert result is True
        assert "future_sensitive_option" in "\n".join(logs.output)
        assert "secret-value" not in "\n".join(logs.output)

    async def test_set_model_raises_when_not_started(self):
        from src.acp.session import ACPSession

        session = ACPSession.__new__(ACPSession)
        session._agent_cmd = "coco"
        session._conn = None
        session._session_id = None

        with self.assertRaises(RuntimeError):
            await session.set_model("gpt-5.2")


# ---------------------------------------------------------------------------
# SyncACPSession.set_model()
# ---------------------------------------------------------------------------
class TestSyncACPSessionSetModel(unittest.TestCase):
    def _make_sync_session(self):
        from src.acp.sync_adapter import SyncACPSession

        s = SyncACPSession.__new__(SyncACPSession)
        s._acp_session = MagicMock()
        s._loop = asyncio.new_event_loop()
        return s

    def test_set_model_delegates_to_acp_session(self):
        from src.acp.sync_adapter import SyncACPSession

        s = SyncACPSession.__new__(SyncACPSession)
        s._acp_session = MagicMock()
        s._loop = None  # no loop → returns False
        result = s.set_model("claude-3.7-sonnet")
        assert result is False  # graceful fallback

    def test_set_model_returns_false_when_no_session(self):
        from src.acp.sync_adapter import SyncACPSession

        s = SyncACPSession.__new__(SyncACPSession)
        s._acp_session = None
        s._loop = None
        result = s.set_model("gpt-5.2")
        assert result is False

    def test_official_codex_set_model_applies_composite_selection(self):
        from src.acp.sync_adapter import SyncACPSession

        s = SyncACPSession.__new__(SyncACPSession)
        s._agent_type = "codex"
        s._agent_args = ["--yes", "@agentclientprotocol/codex-acp@1.1.2"]
        s._model_name = "gpt-5.6-sol/high"
        s._acp_session = MagicMock()
        s._acp_session.set_config_option = AsyncMock(return_value=True)
        s._loop = MagicMock()

        def run_immediately(coroutine, _loop):
            result = asyncio.run(coroutine)
            future = MagicMock()
            future.result.return_value = result
            return future

        with patch(
            "src.acp.sync_adapter.asyncio.run_coroutine_threadsafe",
            side_effect=run_immediately,
        ):
            assert s.set_model("gpt-5.6-sol/ultra") is True

        assert s._acp_session.set_config_option.await_args_list == [
            unittest.mock.call("model", "gpt-5.6-sol"),
            unittest.mock.call("reasoning_effort", "ultra"),
        ]
        assert s._model_name == "gpt-5.6-sol/ultra"

    def test_traex_set_model_applies_profile_then_effort(self):
        from src.acp.sync_adapter import SyncACPSession
        from src.acp.traex_selection import TraexRuntimeSelection

        s = SyncACPSession.__new__(SyncACPSession)
        s._agent_type = "traex"
        s._agent_args = ["acp", "serve"]
        s._model_name = "c_o_new_thinking/standard/high"
        s._acp_session = MagicMock()
        s._acp_session.set_config_option = AsyncMock(return_value=True)
        s._loop = MagicMock()

        def run_immediately(coroutine, _loop):
            result = asyncio.run(coroutine)
            future = MagicMock()
            future.result.return_value = result
            return future

        with (
            patch(
                "src.acp.traex_selection.resolve_traex_runtime_selection",
                return_value=TraexRuntimeSelection(
                    model_id="c_o_new_thinking",
                    backend_model_value="c_o_new_thinking__max",
                    profile="max",
                    effort="max",
                ),
            ),
            patch(
                "src.acp.sync_adapter.asyncio.run_coroutine_threadsafe",
                side_effect=run_immediately,
            ),
        ):
            assert s.set_model("c_o_new_thinking/max/max") is True

        assert s._acp_session.set_config_option.await_args_list == [
            unittest.mock.call("model", "c_o_new_thinking__max"),
            unittest.mock.call("reasoning_effort", "max"),
        ]
        assert s._model_name == "c_o_new_thinking/max/max"

    def test_traex_set_model_keeps_old_selection_when_effort_is_rejected(self):
        from src.acp.sync_adapter import SyncACPSession
        from src.acp.traex_selection import TraexRuntimeSelection

        s = SyncACPSession.__new__(SyncACPSession)
        s._agent_type = "traex"
        s._agent_args = ["acp", "serve"]
        s._model_name = "c_o_new_thinking/standard/high"
        s._acp_session = MagicMock()
        s._acp_session.set_config_option = AsyncMock(side_effect=[True, False])
        s._loop = MagicMock()

        def run_immediately(coroutine, _loop):
            result = asyncio.run(coroutine)
            future = MagicMock()
            future.result.return_value = result
            return future

        with (
            patch(
                "src.acp.traex_selection.resolve_traex_runtime_selection",
                return_value=TraexRuntimeSelection(
                    model_id="c_o_new_thinking",
                    backend_model_value="c_o_new_thinking__max",
                    profile="max",
                    effort="max",
                ),
            ),
            patch(
                "src.acp.sync_adapter.asyncio.run_coroutine_threadsafe",
                side_effect=run_immediately,
            ),
        ):
            assert s.set_model("c_o_new_thinking/max/max") is False

        assert s._model_name == "c_o_new_thinking/standard/high"


# ---------------------------------------------------------------------------
# ACPSessionManager.ensure_session() — model mismatch restart for non-TTADK
# ---------------------------------------------------------------------------
class TestEnsureSessionModelMismatch(unittest.TestCase):
    def _make_manager(self):
        from src.acp.manager import ACPSessionManager

        mgr = ACPSessionManager.__new__(ACPSessionManager)
        import threading
        mgr._lock = threading.Lock()
        mgr._sessions = {}
        mgr._closing_sessions = {}
        mgr._agent_type = "coco"
        mgr._session_timeout = 3600
        mgr._max_sessions = 10
        return mgr

    def test_model_mismatch_triggers_restart(self):
        """ensure_session with a new model_name restarts non-TTADK sessions."""
        import time

        from src.acp.manager import ACPSessionManager

        mgr = self._make_manager()

        # Inject a fake existing session with old model in its args
        fake_session = MagicMock()
        fake_session.last_active = time.time()
        fake_session.session_id = "old-sess"
        fake_session._agent_type = "coco"
        fake_session._agent_args = ["acp", "serve", "-c", "model.name=gpt-5.2"]
        fake_session.is_server_running.return_value = True
        fake_session.is_server_healthy.return_value = True

        key = ACPSessionManager._session_key("chat1")
        mgr._sessions[key] = fake_session

        # Patch start_session so we don't actually spawn a process
        new_session = MagicMock()
        new_session.session_id = "new-sess"
        with patch.object(mgr, "start_session", return_value=new_session) as mock_start:
            result = mgr.ensure_session(
                "chat1",
                cwd="/tmp",
                model_name="claude-3.7-sonnet",  # different from old model
            )

        mock_start.assert_called_once()
        assert result is new_session

    def test_same_model_no_restart(self):
        """ensure_session with the SAME model does NOT restart."""
        import time

        from src.acp.manager import ACPSessionManager

        mgr = self._make_manager()

        fake_session = MagicMock()
        fake_session.last_active = time.time()
        fake_session.session_id = "same-sess"
        fake_session._agent_type = "coco"
        fake_session._agent_args = ["acp", "serve", "-c", "model.name=gpt-5.2"]
        fake_session.is_server_running.return_value = True
        fake_session.is_server_healthy.return_value = True

        key = ACPSessionManager._session_key("chat1")
        mgr._sessions[key] = fake_session

        with patch.object(mgr, "start_session") as mock_start:
            result = mgr.ensure_session(
                "chat1",
                cwd="/tmp",
                model_name="gpt-5.2",  # same model
            )

        mock_start.assert_not_called()
        assert result is fake_session

    def test_official_codex_composite_model_reuses_matching_session(self):
        import time

        from src.acp.manager import ACPSessionManager

        mgr = self._make_manager()
        mgr._agent_type = "codex"

        fake_session = MagicMock()
        fake_session.last_active = time.time()
        fake_session.session_id = "codex-sess"
        fake_session._agent_type = "codex"
        fake_session._agent_args = [
            "--yes",
            "@agentclientprotocol/codex-acp@1.1.2",
        ]
        fake_session._model_name = "gpt-5.6-sol/max"
        fake_session.is_server_running.return_value = True

        key = ACPSessionManager._session_key("chat1")
        mgr._sessions[key] = fake_session

        with patch.object(mgr, "start_session") as mock_start:
            result = mgr.ensure_session(
                "chat1",
                cwd="/tmp",
                model_name="gpt-5.6-sol/max",
            )

        mock_start.assert_not_called()
        assert result is fake_session


# ---------------------------------------------------------------------------
# ProgrammingModeHandler.switch_model()
# ---------------------------------------------------------------------------
def _make_coco_handler():
    """Build a minimal CocoModeHandler with all deps mocked."""
    from src.feishu.handlers.programming import CocoModeHandler

    ctx = MagicMock()
    ctx.settings.thread_programming_enabled = False
    ctx.settings.acp_startup_timeout = 10
    ctx.mode_manager.is_coco_mode.return_value = True
    ctx.project_manager.get_active_project.return_value = None
    ctx.working_dirs = {}

    handler = CocoModeHandler(ctx)
    handler.reply_card = MagicMock()
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()
    handler.get_working_dir = MagicMock(return_value="/tmp")
    return handler


class TestSwitchModelACPPath(unittest.TestCase):
    """switch_model() succeeds via ACP set_model — no restart."""

    def test_acp_switch_success_sends_reply(self):
        handler = _make_coco_handler()

        fake_session = MagicMock()
        fake_session.set_model = MagicMock(return_value=True)

        mgr_mock = MagicMock()
        mgr_mock.get_session.return_value = fake_session

        project = MagicMock(spec=ProjectContext)
        project.project_id = "p1"
        project.coco_mode = True
        project.claude_mode = False
        project.ttadk_mode = False
        project.theme_color = "blue"
        project.root_path = "/tmp/p1"
        project.project_name = "P1"

        with patch.object(handler, "_get_session_manager", return_value=mgr_mock):
            result = handler.switch_model("msg1", "chat1", "claude-3.7-sonnet", project=project)

        assert result is True
        fake_session.set_model.assert_called_once_with("claude-3.7-sonnet")
        mgr_mock.end_session.assert_not_called()
        handler.reply_card.assert_called_once()
        card_str = handler.reply_card.call_args[0][1]
        assert "claude-3.7-sonnet" in card_str
        assert "对话上下文已保留" in card_str


class TestSwitchModelFallbackPath(unittest.TestCase):
    """switch_model() falls back to session restart when set_model fails."""

    def test_fallback_restarts_session_and_replies(self):
        handler = _make_coco_handler()

        fake_session = MagicMock()
        fake_session.set_model = MagicMock(return_value=False)  # ACP switch fails

        new_session = MagicMock()
        new_session.session_id = "new-sess"

        mgr_mock = MagicMock()
        mgr_mock.get_session.return_value = fake_session
        mgr_mock.ensure_session.return_value = new_session

        project = MagicMock(spec=ProjectContext)
        project.project_id = "p1"
        project.coco_mode = True
        project.claude_mode = False
        project.ttadk_mode = False
        project.theme_color = "blue"
        project.root_path = "/tmp/p1"
        project.project_name = "P1"

        with patch.object(handler, "_get_session_manager", return_value=mgr_mock):
            result = handler.switch_model("msg1", "chat1", "gpt-5.2", project=project)

        assert result is True
        mgr_mock.end_session.assert_called_once_with("chat1", project_id="p1")
        mgr_mock.ensure_session.assert_called_once()
        call_kwargs = mgr_mock.ensure_session.call_args
        assert call_kwargs[1].get("model_name") == "gpt-5.2" or "gpt-5.2" in str(call_kwargs)
        handler.reply_card.assert_called_once()
        card_str = handler.reply_card.call_args[0][1]
        assert "已重启会话" in card_str

    def test_fallback_on_set_model_exception(self):
        handler = _make_coco_handler()

        fake_session = MagicMock()
        fake_session.set_model = MagicMock(side_effect=RuntimeError("boom"))

        mgr_mock = MagicMock()
        mgr_mock.get_session.return_value = fake_session
        mgr_mock.ensure_session.return_value = MagicMock()

        with patch.object(handler, "_get_session_manager", return_value=mgr_mock):
            handler.switch_model("msg1", "chat1", "gpt-5.2")

        mgr_mock.end_session.assert_called_once()


class TestSwitchModelNoSession(unittest.TestCase):
    """switch_model() with no existing session goes straight to ensure_session."""

    def test_no_session_calls_ensure_session(self):
        handler = _make_coco_handler()

        mgr_mock = MagicMock()
        mgr_mock.get_session.return_value = None
        mgr_mock.ensure_session.return_value = MagicMock()

        with patch.object(handler, "_get_session_manager", return_value=mgr_mock):
            result = handler.switch_model("msg1", "chat1", "gpt-5.2")

        assert result is True
        mgr_mock.end_session.assert_not_called()
        mgr_mock.ensure_session.assert_called_once()

    def test_restart_failure_returns_false(self):
        handler = _make_coco_handler()
        mgr_mock = MagicMock()
        mgr_mock.get_session.return_value = None
        mgr_mock.ensure_session.side_effect = RuntimeError("startup failed")

        with patch.object(handler, "_get_session_manager", return_value=mgr_mock):
            result = handler.switch_model("msg1", "chat1", "gpt-5.2")

        assert result is False
        handler.reply_error.assert_called_once()


# ---------------------------------------------------------------------------
# SystemHandler._enter_mode_with_acp_model() routing
# ---------------------------------------------------------------------------
def _make_system_handler(is_coco_mode=False):
    from src.feishu.handlers.system import SystemHandler

    ctx = MagicMock()
    ctx.settings.app_id = "test"
    ctx.settings.app_secret = "secret"
    ctx.mode_manager.is_coco_mode.return_value = is_coco_mode
    ctx.mode_manager.is_claude_mode.return_value = False
    ctx.mode_manager.is_aiden_mode.return_value = False
    ctx.mode_manager.is_codex_mode.return_value = False
    ctx.mode_manager.is_gemini_mode.return_value = False
    ctx.project_manager.get_active_project.return_value = None
    ctx.working_dirs = {}

    mock_handlers = {
        "coco": MagicMock(spec=["enter_mode", "switch_model", "current_model"]),
        "claude": MagicMock(spec=["enter_mode", "switch_model", "current_model"]),
        "aiden": MagicMock(spec=["enter_mode", "switch_model", "current_model"]),
        "codex": MagicMock(spec=["enter_mode", "switch_model", "current_model"]),
        "gemini": MagicMock(spec=["enter_mode", "switch_model", "current_model"]),
    }
    ctx.handlers = mock_handlers

    handler = SystemHandler(ctx)
    handler.reply_text = MagicMock()
    handler.reply_error = MagicMock()
    return handler


class TestEnterModeWithAcpModelRouting(unittest.TestCase):
    def test_forged_callback_rejects_unsupported_claude_1m_before_scheduling(self):
        handler = _make_system_handler(is_coco_mode=False)
        project = ProjectContext(
            project_id="project-1",
            project_name="truthful-claude-model",
            root_path="/tmp",
        )
        project.acp_tool_name = "claude"
        project.acp_model_name = "claude-haiku-4-5"

        handler.handle_select_acp_model(
            "selection-card",
            "chat-1",
            "claude",
            "claude-haiku-4-5[1m]",
            project,
        )

        handler.reply_error.assert_called_once()
        assert "不支持 1M" in handler.reply_error.call_args.args[1]
        handler.ctx.scheduler.submit.assert_not_called()
        handler.ctx.handlers["claude"].enter_mode.assert_not_called()
        handler.ctx.handlers["claude"].switch_model.assert_not_called()
        handler.ctx.project_manager.commit_acp_programming_activation.assert_not_called()

    def test_model_command_rejects_unsupported_claude_1m_before_activation(self):
        handler = _make_system_handler(is_coco_mode=False)
        project = ProjectContext(
            project_id="project-1",
            project_name="truthful-claude-model",
            root_path="/tmp",
        )
        project.acp_tool_name = "claude"
        project.acp_model_name = "claude-haiku-4-5"

        handler.handle_model_command(
            "command-message",
            "chat-1",
            "/model claude-haiku-4-5[1m]",
            project,
        )

        handler.reply_error.assert_called_once()
        assert "不支持 1M" in handler.reply_error.call_args.args[1]
        handler.ctx.scheduler.submit.assert_not_called()
        handler.ctx.handlers["claude"].enter_mode.assert_not_called()
        handler.ctx.project_manager.commit_acp_programming_activation.assert_not_called()

    def test_failed_claude_restart_keeps_previous_project_selection(self):
        handler = _make_system_handler(is_coco_mode=False)
        project = ProjectContext(
            project_id="project-1",
            project_name="truthful-claude-model",
            root_path="/tmp",
        )
        project.acp_tool_name = "coco"
        project.acp_model_name = "previous-model"
        project.set_programming_mode("coco", True, "previous-session", 4)
        claude_handler = handler.ctx.handlers["claude"]
        claude_handler.current_model = "previous-handler-model"
        claude_handler.enter_mode.return_value = False
        handler.update_card = MagicMock(return_value=True)
        before = (
            project.acp_tool_name,
            project.acp_model_name,
            project.coco_mode,
            project.claude_mode,
            project.coco_session_snapshot,
            project.claude_session_snapshot,
        )

        handler.handle_select_acp_model(
            "selection-card",
            "chat-1",
            "claude",
            "claude-opus-4-8[1m]",
            project,
            run_immediately=True,
        )

        assert (
            project.acp_tool_name,
            project.acp_model_name,
            project.coco_mode,
            project.claude_mode,
            project.coco_session_snapshot,
            project.claude_session_snapshot,
        ) == before
        assert claude_handler.current_model == "previous-handler-model"
        handler.ctx.project_manager.commit_acp_programming_activation.assert_not_called()

    def test_calls_enter_mode_when_not_in_coco_mode(self):
        handler = _make_system_handler(is_coco_mode=False)
        handler._enter_mode_with_acp_model("msg1", "chat1", "coco", "gpt-5.2")
        handler.ctx.handlers["coco"].enter_mode.assert_called_once()
        handler.ctx.handlers["coco"].switch_model.assert_not_called()

    def test_calls_switch_model_when_already_in_coco_mode(self):
        handler = _make_system_handler(is_coco_mode=True)
        handler._enter_mode_with_acp_model("msg1", "chat1", "coco", "claude-3.7-sonnet")
        handler.ctx.handlers["coco"].switch_model.assert_called_once_with(
            "msg1", "chat1", "claude-3.7-sonnet", project=None
        )
        handler.ctx.handlers["coco"].enter_mode.assert_not_called()

    def test_calls_enter_mode_for_claude_when_not_in_claude_mode(self):
        handler = _make_system_handler(is_coco_mode=False)
        handler._enter_mode_with_acp_model("msg1", "chat1", "claude", "opus-4")
        handler.ctx.handlers["claude"].enter_mode.assert_called_once()
        handler.ctx.handlers["claude"].switch_model.assert_not_called()

    def test_error_for_unknown_tool(self):
        handler = _make_system_handler(is_coco_mode=False)
        handler._enter_mode_with_acp_model("msg1", "chat1", "unknown_tool", "gpt-5.2")
        handler.reply_error.assert_called_once()
        msg = handler.reply_error.call_args[0][1]
        assert "unknown_tool" in msg

    def test_passes_explicit_thread_id_and_returns_handler_failure(self):
        handler = _make_system_handler(is_coco_mode=False)
        handler.ctx.handlers["coco"].enter_mode.return_value = False

        result = handler._enter_mode_with_acp_model(
            "msg1",
            "chat1",
            "coco",
            "gpt-5.2",
            thread_id="thread-1",
        )

        assert result is False
        handler.ctx.handlers["coco"].enter_mode.assert_called_once_with(
            "msg1",
            "chat1",
            project=None,
            silent=True,
            thread_id="thread-1",
        )
