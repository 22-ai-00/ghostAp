"""Tests for diagnostics-level multi-chat isolation.

Validates that:
- TaskScheduler.get_state_by_task_id respects chat_id filtering.
- ACPSessionManager.list_active_sessions respects chat_id filtering.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tasking.scheduler import TaskRunState, TaskScheduler, TaskSpec


class TestSchedulerTaskIdIsolation:
    """get_state_by_task_id only returns tasks belonging to the requesting chat."""

    @pytest.fixture()
    def scheduler(self):
        s = TaskScheduler(max_concurrent=2)
        yield s
        s.stop(wait=False)

    def _inject_task(self, scheduler: TaskScheduler, chat_id: str, task_id: str):
        """Inject a task state directly into scheduler internals (avoids actually running it)."""
        spec = TaskSpec(chat_id=chat_id, name="test", task_id=task_id)
        run_id = f"run_{task_id}"
        state = TaskRunState(spec=spec, run_id=run_id, assigned_queue_key=f"{chat_id}:DEFAULT")
        with scheduler._lock:
            scheduler._states[run_id] = state
            scheduler._by_task_id[task_id] = run_id

    def test_same_chat_can_see_task(self, scheduler):
        self._inject_task(scheduler, "chatA", "task_001")
        result = scheduler.get_state_by_task_id("task_001", chat_id="chatA")
        assert result is not None
        assert result.spec.chat_id == "chatA"

    def test_different_chat_cannot_see_task(self, scheduler):
        self._inject_task(scheduler, "chatA", "task_002")
        result = scheduler.get_state_by_task_id("task_002", chat_id="chatB")
        assert result is None

    def test_no_chat_id_raises_type_error(self, scheduler):
        """chat_id is now required — omitting it must raise TypeError."""
        self._inject_task(scheduler, "chatA", "task_003")
        with pytest.raises(TypeError):
            scheduler.get_state_by_task_id("task_003")

    def test_partial_match_with_chat_isolation(self, scheduler):
        self._inject_task(scheduler, "chatA", "20250425_abcdef")
        # Partial match works for same chat
        result = scheduler.get_state_by_task_id("abcdef", chat_id="chatA")
        assert result is not None
        # But not for different chat
        result = scheduler.get_state_by_task_id("abcdef", chat_id="chatB")
        assert result is None


class TestACPSessionListIsolation:
    """list_active_sessions only returns sessions belonging to the requesting chat."""

    def test_filter_by_chat_id(self):
        from src.acp.manager import ACPSessionManager

        mgr = ACPSessionManager.__new__(ACPSessionManager)
        # Minimal initialization for list_active_sessions
        mgr._sessions = {}
        mgr._lock = __import__("threading").Lock()
        mgr._agent_type = "coco"

        # Mock the idle health service
        mock_health = MagicMock()
        mock_health.classify_session_idle_health.return_value = ("healthy", "active", 0.0, {})
        mgr._idle_health_service = mock_health

        # Create mock sessions keyed by SessionKeyCodec format: {chat_id}:{project_id}
        from src.acp.helper import SessionKeyCodec

        key_a = SessionKeyCodec.encode("chatA", project_id="proj1")
        key_b = SessionKeyCodec.encode("chatB", project_id="proj2")

        mock_session_a = MagicMock()
        mock_session_a.session_id = "sess_a"
        mock_session_a.last_active = 100.0
        mock_session_a.message_count = 5

        mock_session_b = MagicMock()
        mock_session_b.session_id = "sess_b"
        mock_session_b.last_active = 200.0
        mock_session_b.message_count = 3

        mgr._sessions = {key_a: mock_session_a, key_b: mock_session_b}

        # Mock _acquire_lock to use real lock
        from contextlib import contextmanager

        @contextmanager
        def _fake_lock():
            with mgr._lock:
                yield

        mgr._acquire_lock = _fake_lock

        # No filter: returns both
        all_sessions = mgr.list_active_sessions()
        assert len(all_sessions) == 2

        # Filter chatA: only one
        a_sessions = mgr.list_active_sessions(chat_id="chatA")
        assert len(a_sessions) == 1
        assert a_sessions[0]["session_key"] == key_a

        # Filter chatB: only one
        b_sessions = mgr.list_active_sessions(chat_id="chatB")
        assert len(b_sessions) == 1
        assert b_sessions[0]["session_key"] == key_b

        # Filter unknown chat: empty
        x_sessions = mgr.list_active_sessions(chat_id="chatX")
        assert len(x_sessions) == 0


class TestBuildLockStatusLines:
    """_build_lock_status_lines returns empty when idle, non-empty only with active locks."""

    @staticmethod
    def _make_handler(chat_lock_mgr=None, repo_lock_mgr=None):
        """Create a minimal DiagnosticsHandler mock with only what _build_lock_status_lines needs."""
        handler = MagicMock()
        handler.ctx = MagicMock()
        handler.ctx.chat_lock_manager = chat_lock_mgr
        handler.ctx.repo_lock_manager = repo_lock_mgr
        # Bind the real method to our mock
        from src.feishu.handlers.diagnostics import DiagnosticsHandler
        handler._build_lock_status_lines = DiagnosticsHandler._build_lock_status_lines.__get__(handler)
        return handler

    def test_placeholder_when_no_managers(self):
        handler = self._make_handler(chat_lock_mgr=None, repo_lock_mgr=None)
        result = handler._build_lock_status_lines("chat1")
        # F-18: no managers → empty string (hide lock section to reduce noise)
        assert result == ""

    def test_placeholder_when_managers_present_but_no_locks(self):
        chat_mgr = MagicMock()
        chat_mgr.get_lock_info.return_value = None
        repo_mgr = MagicMock()
        repo_mgr.get_lock_info.return_value = None
        project = MagicMock()
        project.root_path = "/some/path"
        handler = self._make_handler(chat_lock_mgr=chat_mgr, repo_lock_mgr=repo_mgr)
        result = handler._build_lock_status_lines("chat1", project=project)
        # F-11: when lock subsystem is enabled but no active locks, show "unlocked"
        assert "未锁定" in result

    def test_nonempty_when_chat_locked(self):
        import time as _time
        lock_info = MagicMock()
        lock_info.locked_by = "user_admin"
        lock_info.locked_by_name = "Admin"
        lock_info.locked_at_wall = _time.time()
        lock_info.locked_at = _time.monotonic()
        chat_mgr = MagicMock()
        chat_mgr.get_lock_info.return_value = lock_info
        handler = self._make_handler(chat_lock_mgr=chat_mgr)
        result = handler._build_lock_status_lines("chat1")
        assert result != ""
        assert "群锁定" in result
        assert "Admin" in result

    def test_chat_lock_cross_day_format(self):
        """F-19: cross-day lock shows MM-DD HH:MM format."""
        import time as _time
        lock_info = MagicMock()
        lock_info.locked_by = "user_admin"
        lock_info.locked_by_name = "Admin"
        # Set to yesterday
        lock_info.locked_at_wall = _time.time() - 86400 * 2
        lock_info.locked_at = _time.monotonic() - 86400 * 2
        chat_mgr = MagicMock()
        chat_mgr.get_lock_info.return_value = lock_info
        handler = self._make_handler(chat_lock_mgr=chat_mgr)
        result = handler._build_lock_status_lines("chat1")
        assert result != ""
        # Cross-day: should contain MM-DD format (e.g. "04-24")
        from datetime import datetime
        expected_date = datetime.fromtimestamp(lock_info.locked_at_wall).strftime("%m-%d")
        assert expected_date in result

    def test_nonempty_when_repo_locked(self):
        import time as _time
        repo_info = MagicMock()
        repo_info.chat_id = "chat1"
        repo_info.acquired_at = _time.monotonic() - 60
        repo_info.idle_seconds = 60.0
        repo_mgr = MagicMock()
        repo_mgr.get_lock_info.return_value = repo_info
        project = MagicMock()
        project.root_path = "/workspace/my-repo"
        handler = self._make_handler(repo_lock_mgr=repo_mgr)
        result = handler._build_lock_status_lines("chat1", project=project)
        assert result != ""
        assert "仓库锁" in result
        assert "my-repo" in result

    def test_repo_lock_shows_remaining_release_time(self):
        """F-11: repo lock section shows remaining auto-release time."""
        import time as _time
        repo_info = MagicMock()
        repo_info.chat_id = "chat1"
        repo_info.acquired_at = _time.monotonic() - 120
        repo_info.idle_seconds = 120.0  # idle for 2 min
        repo_mgr = MagicMock()
        repo_mgr.get_lock_info.return_value = repo_info
        project = MagicMock()
        project.root_path = "/workspace/my-repo"
        handler = self._make_handler(repo_lock_mgr=repo_mgr)
        with patch("src.config.get_settings") as mock_gs:
            mock_gs.return_value.repo_lock_idle_timeout = 300  # 5 min
            result = handler._build_lock_status_lines("chat1", project=project)
        assert "锁定时长" in result
        # 300 - 120 = 180s → 3 min remaining
        assert "3 分钟后自动释放" in result

    def test_repo_lock_shows_imminent_release(self):
        """F-11: when remaining time < 1min, show '即将自动释放'."""
        import time as _time
        repo_info = MagicMock()
        repo_info.chat_id = "other_chat"
        repo_info.acquired_at = _time.monotonic() - 290
        repo_info.idle_seconds = 290.0  # idle for 4m50s
        repo_mgr = MagicMock()
        repo_mgr.get_lock_info.return_value = repo_info
        project = MagicMock()
        project.root_path = "/workspace/my-repo"
        handler = self._make_handler(repo_lock_mgr=repo_mgr)
        with patch("src.config.get_settings") as mock_gs:
            mock_gs.return_value.repo_lock_idle_timeout = 300
            result = handler._build_lock_status_lines("chat1", project=project)
        assert "即将自动释放" in result

    # --- Admin differentiation tests (Task 15/16) ---

    def test_admin_chat_lock_shows_unlock_hint(self):
        """is_admin=True appends /unlock hint to chat lock line."""
        import time as _time
        lock_info = MagicMock()
        lock_info.locked_by = "user_admin"
        lock_info.locked_by_name = "Admin"
        lock_info.locked_at_wall = _time.time()
        lock_info.locked_at = _time.monotonic()
        chat_mgr = MagicMock()
        chat_mgr.get_lock_info.return_value = lock_info
        handler = self._make_handler(chat_lock_mgr=chat_mgr)
        result = handler._build_lock_status_lines("chat1", is_admin=True)
        assert "/unlock" in result
        assert "管理员" in result

    def test_nonadmin_chat_lock_no_unlock_hint(self):
        """is_admin=False (default) does NOT show /unlock hint."""
        import time as _time
        lock_info = MagicMock()
        lock_info.locked_by = "user_admin"
        lock_info.locked_by_name = "Admin"
        lock_info.locked_at_wall = _time.time()
        lock_info.locked_at = _time.monotonic()
        chat_mgr = MagicMock()
        chat_mgr.get_lock_info.return_value = lock_info
        handler = self._make_handler(chat_lock_mgr=chat_mgr)
        result = handler._build_lock_status_lines("chat1", is_admin=False)
        assert "/unlock" not in result

    def test_admin_repo_lock_other_chat_shows_force_release_hint(self):
        """is_admin=True + repo locked by another chat appends force-release hint."""
        import time as _time
        repo_info = MagicMock()
        repo_info.chat_id = "other_chat"
        repo_info.acquired_at = _time.monotonic() - 60
        repo_info.idle_seconds = 60.0
        repo_mgr = MagicMock()
        repo_mgr.get_lock_info.return_value = repo_info
        project = MagicMock()
        project.root_path = "/workspace/my-repo"
        handler = self._make_handler(repo_lock_mgr=repo_mgr)
        with patch("src.config.get_settings") as mock_gs:
            mock_gs.return_value.repo_lock_idle_timeout = 300
            result = handler._build_lock_status_lines("chat1", project=project, is_admin=True)
        assert "强制释放" in result

    def test_admin_repo_lock_same_chat_no_force_release_hint(self):
        """is_admin=True but repo locked by SAME chat should NOT show force-release hint."""
        import time as _time
        repo_info = MagicMock()
        repo_info.chat_id = "chat1"  # same chat
        repo_info.acquired_at = _time.monotonic() - 60
        repo_info.idle_seconds = 60.0
        repo_mgr = MagicMock()
        repo_mgr.get_lock_info.return_value = repo_info
        project = MagicMock()
        project.root_path = "/workspace/my-repo"
        handler = self._make_handler(repo_lock_mgr=repo_mgr)
        with patch("src.config.get_settings") as mock_gs:
            mock_gs.return_value.repo_lock_idle_timeout = 300
            result = handler._build_lock_status_lines("chat1", project=project, is_admin=True)
        assert "强制释放" not in result

    def test_nonadmin_repo_lock_other_chat_no_force_release_hint(self):
        """is_admin=False + repo locked by another chat should NOT show force-release hint."""
        import time as _time
        repo_info = MagicMock()
        repo_info.chat_id = "other_chat"
        repo_info.acquired_at = _time.monotonic() - 60
        repo_info.idle_seconds = 60.0
        repo_mgr = MagicMock()
        repo_mgr.get_lock_info.return_value = repo_info
        project = MagicMock()
        project.root_path = "/workspace/my-repo"
        handler = self._make_handler(repo_lock_mgr=repo_mgr)
        with patch("src.config.get_settings") as mock_gs:
            mock_gs.return_value.repo_lock_idle_timeout = 300
            result = handler._build_lock_status_lines("chat1", project=project, is_admin=False)
        assert "强制释放" not in result


class TestTraceProjectIsolation:
    """show_message_trace uses get_project_for_chat to prevent cross-chat leakage."""

    @staticmethod
    def _make_diagnostics_handler():
        import threading

        from src.feishu.handler_context import HandlerContext
        from src.feishu.handlers.diagnostics import DiagnosticsHandler

        ctx = HandlerContext(
            settings=MagicMock(),
            api_client_factory=MagicMock(),
            message_callback=MagicMock(),
            coco_manager=MagicMock(),
            claude_manager=MagicMock(),
            aiden_manager=MagicMock(),
            codex_manager=MagicMock(),
            gemini_manager=MagicMock(),
            traex_manager=MagicMock(),
            tui2acp_manager=MagicMock(),
            intent_recognizer=MagicMock(),
            scheduler=MagicMock(),
            project_manager=MagicMock(),
            message_mapper=MagicMock(),
            message_linker=MagicMock(),
            mode_manager=MagicMock(),
            context_manager=MagicMock(),
            deep_engine_manager=MagicMock(),
            progress_reporter=MagicMock(),
            spec_engine_manager=MagicMock(),
            spec_reporter=MagicMock(),
            thread_manager=MagicMock(),
            image_handler_factory=MagicMock(),
            working_dirs={},
            working_dir_lock=threading.Lock(),
            pending_image_keys={},
            pending_image_lock=threading.Lock(),
            enable_streaming=False,
            managers={},
            handlers={},
            slock_engine_manager=MagicMock(),
        )
        h = DiagnosticsHandler(ctx)
        h.reply_text = MagicMock()
        h.reply_card = MagicMock(return_value="resp1")
        return h, ctx

    def test_trace_uses_get_project_for_chat_not_get_project(self):
        """When trace data has a project_id, it must use get_project_for_chat."""
        h, ctx = self._make_diagnostics_handler()
        trace_data = {"chat_id": "chatA", "project_id": "proj1", "message_id": "m1"}
        ctx.message_linker.query.return_value = trace_data
        ctx.project_manager.get_project_for_chat.return_value = None

        from unittest.mock import patch
        with patch("src.feishu.handlers.diagnostics.CardBuilder") as mock_cb:
            mock_cb.build_message_trace_content.return_value = "trace content"
            mock_cb.build_smart_response_card.return_value = ("interactive", {"card": "data"})
            h.show_message_trace("m1", "chatA", "/trace m1", project=None)

        # Verify get_project_for_chat was called with (proj_id, chat_id)
        ctx.project_manager.get_project_for_chat.assert_called_once_with("proj1", "chatA")

    def test_trace_cross_chat_project_returns_none(self):
        """Cross-chat project_id in trace data should not leak project info."""
        h, ctx = self._make_diagnostics_handler()
        # Trace data belongs to chatA, but project belongs to chatA only
        trace_data = {"chat_id": "chatA", "project_id": "proj1", "message_id": "m1"}
        ctx.message_linker.query.return_value = trace_data
        # get_project_for_chat returns None for cross-chat
        ctx.project_manager.get_project_for_chat.return_value = None

        from unittest.mock import patch
        with patch("src.feishu.handlers.diagnostics.CardBuilder") as mock_cb:
            mock_cb.build_message_trace_content.return_value = "trace content"
            mock_cb.build_smart_response_card.return_value = ("interactive", {"card": "data"})
            h.show_message_trace("m1", "chatA", "/trace m1", project=None)

        # Should use smart_response_card (not project_response_card) since project is None
        mock_cb.build_smart_response_card.assert_called_once()
