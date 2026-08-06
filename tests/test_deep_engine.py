"""Tests for deep_engine — ACP-driven DeepEngine."""

import io
import logging
import os
import re
from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import PlanEntryInfo, PlanInfo, ToolCallInfo
from src.agent_session import ClaudeCLIConfig, SyncClaudeCLISession, create_engine_session
from src.deep_engine.engine import DeepEngine, DeepEngineCallbacks, DeepEngineManager
from src.deep_engine.models import DeepProject, DeepProjectStatus, EngineRunState
from src.deep_engine.progress import DeepProgress, _truncate_nested_data


class _DummySession:
    """Minimal session stub shared across TTADK startup/resume tests."""

    def __init__(self, *a, **k):
        self.session_id = "sid"
        self.created_at = 0.0
        self.last_active = 0.0
        self.message_count = 0
        self.last_query = ""
        self.is_resumed = False

    def describe_agent(self):
        return "dummy"

    def start(self, startup_timeout: float = 60, **kwargs):
        return "sid"

    def load_session(self, session_id: str, timeout: float = 60):
        del timeout
        return None

    def load_local_history(self, session_id=None, limit: int = 200):
        return []

    def cancel(self):
        return None

    def close(self):
        return None

    def to_snapshot(self):
        return {}

    def get_session_info(self):
        return ""

    def is_server_running(self):
        return True

    def is_server_healthy(self, healthcheck_timeout: float = 2.0):
        return True

    def send_prompt(self, *a, **k):
        return MagicMock(stop_reason="end_turn")


class _SessSettings:
    acp_startup_timeout = 20
    rate_limit_retry_enabled = False


class TestDeepEngine:
    @patch("src.engine_base.get_settings")
    def _make_engine(self, mock_settings, **kwargs):
        s = MagicMock()
        s.coco_execution_timeout = 300
        s.claude_execution_timeout = 600
        mock_settings.return_value = s
        return DeepEngine(chat_id="c1", root_path="/tmp/test", **kwargs)

    def test_initial_state(self):
        engine = self._make_engine()
        assert engine.run_state == EngineRunState.IDLE
        assert engine.project is None
        assert not engine.is_running

    def test_successful_retry_clears_previous_project_error(self):
        project = DeepProject.create("retry", "/tmp/retry")
        project.fail("first attempt failed")
        project.resume()

        project.complete()

        assert project.status is DeepProjectStatus.COMPLETED
        assert project.error is None

    def test_stop(self):
        engine = self._make_engine()
        engine._run_state = EngineRunState.RUNNING
        engine._session = MagicMock()
        engine.stop()
        assert engine.run_state == EngineRunState.STOPPING
        engine._session.cancel.assert_called_once()

    def test_pause(self):
        engine = self._make_engine()
        engine._project = MagicMock()
        engine._session = MagicMock()
        engine._run_state = EngineRunState.RUNNING
        engine.pause()
        engine._project.pause.assert_called_once()
        assert engine.run_state == EngineRunState.STOPPING

    def test_pause_holds_lock(self):
        """pause() must acquire self._lock when writing _run_state."""
        engine = self._make_engine()
        engine._project = MagicMock()
        engine._session = None
        engine._run_state = EngineRunState.RUNNING

        lock_acquired = False
        original_lock = engine._lock

        class TrackedLock:
            def __enter__(self_lock):
                nonlocal lock_acquired
                lock_acquired = True
                return original_lock.__enter__()

            def __exit__(self_lock, *args):
                return original_lock.__exit__(*args)

        engine._lock = TrackedLock()
        engine.pause()
        assert lock_acquired, "pause() should acquire self._lock"
        assert engine._run_state == EngineRunState.STOPPING

    def test_pause_session_cancel_raises(self):
        """pause() swallows exceptions from session.cancel() and still sets STOPPING."""
        engine = self._make_engine()
        engine._project = MagicMock()
        engine._run_state = EngineRunState.RUNNING

        session = MagicMock()
        session.cancel.side_effect = RuntimeError("connection lost")
        engine._session = session

        engine.pause()

        assert engine._run_state == EngineRunState.STOPPING
        session.cancel.assert_called_once()

    def test_pause_project_none(self):
        """pause() when _project is None must still set STOPPING without raising."""
        engine = self._make_engine()
        engine._project = None
        engine._session = MagicMock()
        engine._run_state = EngineRunState.RUNNING

        engine.pause()

        assert engine._run_state == EngineRunState.STOPPING
        engine._session.cancel.assert_called_once()

    def test_pause_concurrent(self):
        """Concurrent pause() calls must not raise and final state is STOPPING."""
        import threading

        engine = self._make_engine()
        engine._project = MagicMock()
        engine._session = MagicMock()
        engine._run_state = EngineRunState.RUNNING

        errors = []
        barrier = threading.Barrier(10, timeout=5)

        def call_pause():
            try:
                barrier.wait()
                engine.pause()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_pause) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent pause() raised: {errors}"
        assert engine._run_state == EngineRunState.STOPPING

    def test_cleanup(self):
        engine = self._make_engine()
        engine._session = MagicMock()
        engine._project = MagicMock()
        engine.cleanup()
        assert engine._session is None
        assert engine._project is None
        assert engine.run_state == EngineRunState.IDLE

    def test_build_deep_prompt(self):
        engine = self._make_engine()
        prompt = engine._build_deep_prompt("add login feature")
        assert "add login feature" in prompt
        assert "/tmp/test" in prompt
        assert "subagent / 子任务委托" in prompt
        assert "不会修改相同文件/接口契约/迁移配置" in prompt
        assert "哪些任务并行/委托执行" in prompt

    def test_build_deep_prompt_embeds_grill_me_auto_adoption(self):
        engine = self._make_engine()
        prompt = engine._build_deep_prompt("add login feature")

        assert "grill-me" in prompt
        assert "尖锐追问" in prompt
        assert "推荐答案" in prompt
        assert "自动采纳" in prompt
        assert "不要停下等待用户回答" in prompt

    def test_get_rendered_content(self):
        engine = self._make_engine()
        content = engine.get_rendered_content()
        assert isinstance(content, str)

    def test_save_state_no_project(self):
        engine = self._make_engine()
        with pytest.raises(ValueError):
            engine.save_state()

    def test_inject_guidance(self):
        engine = self._make_engine()
        engine._run_state = EngineRunState.RUNNING
        engine._session = MagicMock()
        engine.inject_guidance("test context")

    def test_get_progress_no_project(self):
        engine = self._make_engine()
        assert engine.get_progress() is None

    def test_get_task_summary_no_project(self):
        engine = self._make_engine()
        assert engine.get_task_summary() == "暂无任务"

    def test_error_callback_observes_frozen_failed_project(self):
        engine = self._make_engine()
        session = MagicMock()
        session.send_prompt_with_retry.side_effect = TimeoutError("deadline")
        observed: list[tuple[DeepProjectStatus, float | None, float | None]] = []

        callbacks = DeepEngineCallbacks(
            on_error=lambda _message: observed.append((
                engine.project.status,
                engine.project.completed_at,
                engine.project.duration(),
            )),
        )

        with patch(
            "src.deep_engine.engine.create_engine_session",
            return_value=session,
        ):
            result = engine.plan_and_execute("trigger timeout", callbacks)

        assert len(observed) == 1
        status, completed_at, duration = observed[0]
        assert status == DeepProjectStatus.FAILED
        assert completed_at is not None
        assert duration == result.duration()

    def test_ttadk_startup_model_log_uses_real_or_auto(self, caplog):
        """启动点日志语义：model 字段只能是真实名或 (auto)，不应等于输入友好名。"""
        engine = self._make_engine(agent_type="ttadk_codex", model_name="gpt-5.2")

        caplog.set_level(logging.INFO, logger="src.agent_session")

        with (
            patch("src.agent_session.factory.get_settings", return_value=_SessSettings()),
            patch("src.ttadk.startup_common.precheck_ttadk_startup_model") as mk_precheck,
            patch("src.agent_session.factory.SyncTTADKCLISession", return_value=_DummySession()),
        ):
            # SSOT：create_engine_session 统一走 start_agent_session；单测不应触发真实 ttadk/codex 探测。
            mk_precheck.return_value = {
                "tool": "codex",
                "input_model": "gpt-5.2",
                "model": "gpt-5.2-codex-ttadk",
                "validated": True,
                "source": "probe",
                "decision": "precheck_validated",
                "fail_phase": "",
                "warnings": [],
                "diagnostics": {"attempts": [{"phase": "precheck"}]},
            }
            caplog.clear()
            engine.plan_and_execute("do something")

        text = "\n".join([r.getMessage() for r in caplog.records])
        assert "[SessionFactory] ttadk cli startup:" in text
        m = re.search(r"\bmodel=([^\s]+)", text)
        assert m is not None
        assert m.group(1) == "gpt-5.2-codex-ttadk"
        assert m.group(1) != "gpt-5.2"

    def test_ttadk_resume_model_log_uses_real_or_auto(self, caplog):
        """恢复路径同样要求：model 字段只能是真实名或 (auto)。"""
        engine = self._make_engine(agent_type="ttadk_codex", model_name="gpt-5.2")
        engine._project = DeepProject.create(name="p", root_path="/tmp/test")
        engine._project.status = DeepProjectStatus.PAUSED

        caplog.set_level(logging.INFO, logger="src.agent_session")

        with (
            patch("src.agent_session.factory.get_settings", return_value=_SessSettings()),
            patch("src.ttadk.startup_common.precheck_ttadk_startup_model") as mk_precheck,
            patch("src.agent_session.factory.SyncTTADKCLISession", return_value=_DummySession()),
        ):
            mk_precheck.return_value = {
                "tool": "codex",
                "input_model": "gpt-5.2",
                "model": None,  # (auto)
                "validated": False,
                "source": "defaults",
                "decision": "precheck_auto",
                "fail_phase": "",
                "warnings": ["no_m_passthrough"],
                "diagnostics": {"attempts": [{"phase": "precheck"}]},
            }
            caplog.clear()
            engine.resume()

        text = "\n".join([r.getMessage() for r in caplog.records])
        assert "[SessionFactory] ttadk cli startup:" in text
        assert "model=(auto)" in text
        assert re.search(r"\bmodel=gpt-5\.2\b", text) is None


def test_ttadk_startup_log_semantics_consistent_between_create_sync_and_engine(monkeypatch, caplog):
    """跨入口一致性：create_sync_session 与 create_engine_session 的 TTADK 启动语义日志一致。"""
    import logging

    import src.agent_session as agent_session

    caplog.set_level(logging.INFO, logger="src.agent_session")

    info_ok = {
        "result": _DummySession(),
        "tool": "codex",
        "input_model": "gpt-5.2",
        "resolved_model": "gpt-5.2-codex-ttadk",
        "validated": True,
        "source": "probe",
        "decision": "precheck_validated",
        "fail_phase": "",
        "warnings": [],
        "degraded": False,
        "repaired": False,
        "diagnostics": {"attempts": [{"phase": "precheck"}]},
    }

    monkeypatch.setattr(agent_session.factory, "get_settings", lambda: _SessSettings())
    monkeypatch.setattr("src.ttadk.get_ttadk_manager", lambda *a, **k: MagicMock())
    monkeypatch.setattr("src.ttadk.manager.start_ttadk_engine_session", lambda **kw: dict(info_ok))
    monkeypatch.setattr(
        "src.ttadk.startup_common.precheck_ttadk_startup_model",
        lambda **kw: {
            "tool": "codex",
            "input_model": "gpt-5.2",
            "resolved_real_name": "gpt-5.2-codex-ttadk",
            "model": "gpt-5.2-codex-ttadk",
            "validated": True,
            "source": "probe",
            "decision": "precheck_validated",
            "fail_phase": "",
            "warnings": [],
            "diagnostics": {},
        },
    )

    # 避免触发真实 resolve_agent_spec 探测（codex 在本仓默认会抛 AgentSpecResolveError）。
    monkeypatch.setattr(agent_session.factory, "SyncACPSession", lambda *a, **k: _DummySession())

    caplog.clear()
    _ = agent_session.create_sync_session(agent_type="ttadk_codex", cwd="/tmp", model_name="gpt-5.2")
    _ = agent_session.create_engine_session(agent_type="ttadk_codex", cwd="/tmp", model_name="gpt-5.2")

    text = "\n".join([r.getMessage() for r in caplog.records])
    assert "[SessionFactory] ttadk precheck(startup):" in text
    # create_engine_session now uses CLI startup, while create_sync_session (via start_session_with_retry) might still log precheck
    # The key is to verify consistency in resolved model logic, even if log messages differ slightly due to backend differences.
    # We check for the CLI startup log for the engine session part.
    assert "[SessionFactory] ttadk cli startup:" in text or "[SessionFactory] ttadk startup:" in text
    # 允许存在 input_model=gpt-5.2；只需保证最终 model 不等于输入友好名。
    assert "model=gpt-5.2-codex-ttadk" in text
    assert "\n[SessionFactory] ttadk startup: tool=codex input_model=gpt-5.2 model=gpt-5.2 " not in text


class TestDeepEngineManager:
    def test_get_or_create(self):
        with patch("src.engine_base.get_settings") as mock:
            mock.return_value = MagicMock(coco_execution_timeout=300, claude_execution_timeout=600)
            mgr = DeepEngineManager()
            engine = mgr.get_or_create("c1", "/tmp/test")
            assert engine is not None
            engine2 = mgr.get_or_create("c1", "/tmp/test")
            assert engine is engine2

    def test_get_returns_none_when_missing(self):
        mgr = DeepEngineManager()
        assert mgr.get("nonexistent", "/tmp") is None

    def test_get_active_engine(self):
        with patch("src.engine_base.get_settings") as mock:
            mock.return_value = MagicMock(coco_execution_timeout=300, claude_execution_timeout=600)
            mgr = DeepEngineManager()
            engine = mgr.get_or_create("c1", "/tmp/test")
            assert mgr.get_active_engine("c1") is None
            engine._run_state = EngineRunState.RUNNING
            assert mgr.get_active_engine("c1") is engine

    def test_engine_name_switch(self):
        with patch("src.engine_base.get_settings") as mock:
            mock.return_value = MagicMock(coco_execution_timeout=300, claude_execution_timeout=600)
            mgr = DeepEngineManager()
            e1 = mgr.get_or_create("c1", "/tmp/test", engine_name="Coco")
            assert e1.engine_name == "Coco"
            e2 = mgr.get_or_create("c1", "/tmp/test", engine_name="Claude")
            assert e2.engine_name == "Claude"
            assert e1 is not e2

    def test_cleanup_all(self):
        with patch("src.engine_base.get_settings") as mock:
            mock.return_value = MagicMock(coco_execution_timeout=300, claude_execution_timeout=600)
            mgr = DeepEngineManager()
            mgr.get_or_create("c1", "/tmp/test")
            mgr.get_or_create("c2", "/tmp/test2")
            mgr.cleanup_all()
            assert mgr.get("c1", "/tmp/test") is None

    def test_cleanup_all_keeps_running_engine(self):
        with patch("src.engine_base.get_settings") as mock:
            mock.return_value = MagicMock(coco_execution_timeout=300, claude_execution_timeout=600)
            mgr = DeepEngineManager()
            engine = mgr.get_or_create("c1", "/tmp/test")
            engine._run_state = EngineRunState.RUNNING
            mgr.cleanup_all()
            assert mgr.get("c1", "/tmp/test") is engine
            assert engine.run_state == EngineRunState.STOPPING


class TestDeepProgress:
    def test_initial_state(self):
        p = DeepProgress()
        assert p.completed_steps == 0
        assert p.total_steps == 0
        assert p.progress_percent == 0
        assert p.tool_calls == []
        assert p.modified_files == set()

    def test_update_plan(self):
        p = DeepProgress()
        plan = PlanInfo(
            entries=[
                PlanEntryInfo(content="s1", status="completed"),
                PlanEntryInfo(content="s2", status="in_progress"),
                PlanEntryInfo(content="s3", status="pending"),
            ]
        )
        p.update_plan(plan)
        assert p.total_steps == 3
        assert p.completed_steps == 1

    def test_record_tool(self):
        p = DeepProgress()
        tc = ToolCallInfo(id="t1", title="Edit", kind="edit", status="completed", locations=["/a.py"])
        p.record_tool(tc)
        assert len(p.tool_calls) == 1
        assert "/a.py" in p.modified_files

    def test_append_text(self):
        p = DeepProgress()
        p.append_text("hello ")
        p.append_text("world")
        assert p.text_buffer == "hello world"

    def test_progress_bar(self):
        p = DeepProgress()
        plan = PlanInfo(
            entries=[
                PlanEntryInfo(content="s1", status="completed"),
                PlanEntryInfo(content="s2", status="completed"),
                PlanEntryInfo(content="s3", status="pending"),
                PlanEntryInfo(content="s4", status="pending"),
            ]
        )
        p.update_plan(plan)
        bar = p.progress_bar
        assert "50%" in bar


class TestClaudeCLISession:
    def test_resume_missing_conversation_fallback_to_new_session(self):
        class FakeProc:
            def __init__(self, stdout_text: str, stderr_text: str, returncode: int):
                self.stdout = io.StringIO(stdout_text)
                self.stderr = io.StringIO(stderr_text)
                self.returncode = returncode

            def wait(self, timeout: int = 0):
                return self.returncode

            def terminate(self):
                return None

        cfg = ClaudeCLIConfig(command="claude", add_dir=False, bypass_permissions=False)
        s = SyncClaudeCLISession(cwd="/tmp", config=cfg)
        s.session_id = "sid0"
        s.is_resumed = True

        procs = [
            FakeProc(stdout_text="", stderr_text="No conversation found with session ID: sid0\n", returncode=1),
            FakeProc(stdout_text="ok\n", stderr_text="", returncode=0),
        ]

        popen_calls = []

        def fake_popen(
            args,
            cwd,
            stdout,
            stderr,
            text,
            env=None,
            start_new_session=False,
        ):
            popen_calls.append((args, env))
            assert env is not None
            assert "CLAUDECODE" not in env
            assert start_new_session is (os.name == "posix")
            return procs.pop(0)

        with (
            patch.dict(os.environ, {"CLAUDECODE": "1"}),
            patch("src.agent_session.subprocess.Popen", side_effect=fake_popen),
            patch("src.agent_session.uuid.uuid4", return_value="sid_new"),
        ):
            events = []
            res = s.send_prompt("hi", on_event=lambda e: events.append(e), timeout=5)

        assert res.stop_reason == "end_turn"
        assert "ok" in res.text
        assert len(popen_calls) == 2
        assert "--resume" in popen_calls[0][0]
        assert "sid0" in popen_calls[0][0]
        assert "--session-id" in popen_calls[1][0]
        assert "sid_new" in popen_calls[1][0]


def test_create_engine_session_ttadk_invalid_model_autofix(monkeypatch):
    """当 TTADK ACP 启动因 Invalid model 失败时，应自动刷新并重试一次。"""
    from src.ttadk.manager import TTADKManager

    # 使用 ACP 可用的 tool（coco）覆盖 Invalid model 自动纠错闭环
    m = TTADKManager(default_tool="coco", default_model="bad")
    # 避免触碰真实 HOME 文件缓存
    monkeypatch.setattr(TTADKManager, "_load_cache_from_file", lambda self: None)
    monkeypatch.setattr(TTADKManager, "_save_cache_to_file", lambda self: None)
    # Mock refresh_models to avoid subprocess calls
    monkeypatch.setattr(TTADKManager, "refresh_models", lambda self, tool_name=None, cwd=None: None)
    monkeypatch.setattr("src.ttadk.get_ttadk_manager", lambda *a, **k: m)

    calls = {"n": 0}

    def fake_start_session_with_retry(agent_type, cwd, startup_timeout=60, model_name=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("✗ Error: Invalid model 'bad'. Available models: a, b")
        sess = MagicMock()
        sess._agent_type = agent_type
        sess._model_name = model_name
        return sess

    monkeypatch.setattr(
        "src.acp.sync_adapter.start_session_with_retry", lambda *a, **k: fake_start_session_with_retry(*a, **k)
    )

    # Mock precheck to avoid potential delays
    monkeypatch.setattr(
        "src.ttadk.startup_common.precheck_ttadk_startup_model",
        lambda **kw: {
            "tool": "codex",
            "input_model": kw.get("model_intent"),
            "model": "a",
            "resolved_model": "a",
            "validated": True,
            "source": "mock",
            "decision": "precheck_mock",
            "warnings": [],
            "diagnostics": {},
        },
    )

    # 触发 create_engine_session 的 TTADK 分支
    s = create_engine_session(agent_type="ttadk_coco", cwd="/tmp", model_name="bad")
    inner = getattr(s, "_inner", s)
    assert getattr(inner, "_model_name", None) == "a"


def test_resolve_ttadk_engine_startup_model_prefers_fast_path(monkeypatch):
    """启动阶段应优先使用 resolve_startup_model，且仅 validated=True 才透传真实模型名。"""
    from src.agent_session import resolve_ttadk_engine_startup_model

    class _Resolved:
        def __init__(self, real_name: str, validated: bool, source: str = "probe", warnings=None):
            self.real_name = real_name
            self.validated = validated
            self.source = source
            self.warnings = list(warnings or [])

    class _MgrFast:
        def get_current_model(self):
            return "gpt-5.2"

        def resolve_startup_model(self, model_name: str, *, tool_name: str, cwd: str):
            assert tool_name == "codex"
            assert cwd == "/tmp/test"
            assert model_name == "gpt-5.2"
            return _Resolved(real_name="gpt-5.2-codex-ttadk", validated=True, source="probe")

    monkeypatch.setattr("src.ttadk.get_ttadk_manager", lambda *a, **k: _MgrFast())

    info = resolve_ttadk_engine_startup_model(agent_type="ttadk_codex", cwd="/tmp/test", model_intent=None)
    assert info["tool"] == "codex"
    assert info["validated"] is True
    assert info["resolved_model"] == "gpt-5.2-codex-ttadk"


def test_resolve_ttadk_engine_startup_model_not_validated_returns_none(monkeypatch):
    """当无法确认真实模型名时，不应透传 -m（resolved_model=None）。"""
    from src.agent_session import resolve_ttadk_engine_startup_model

    class _Resolved:
        def __init__(self, real_name: str, validated: bool, source: str = "unknown", warnings=None):
            self.real_name = real_name
            self.validated = validated
            self.source = source
            self.warnings = list(warnings or [])

    class _MgrLegacy:
        def get_current_model(self):
            return "gpt-5.2"

        def resolve_and_ensure_valid_model(self, model_name: str, *, tool_name: str, cwd: str):
            return _Resolved(real_name=model_name, validated=False, source="defaults", warnings=["no_m_passthrough"])

    monkeypatch.setattr("src.ttadk.get_ttadk_manager", lambda *a, **k: _MgrLegacy())

    info = resolve_ttadk_engine_startup_model(agent_type="ttadk_codex", cwd="/tmp/test", model_intent="gpt-5.2")
    assert info["tool"] == "codex"
    assert info["validated"] is False
    assert info["resolved_model"] is None


def test_resolve_ttadk_engine_startup_model_includes_models_phase_diagnostics(monkeypatch):
    """集成断言：resolve_ttadk_engine_startup_model 返回 diagnostics.attempts，且包含 models 阶段记录。"""
    from src.agent_session import resolve_ttadk_engine_startup_model
    from src.ttadk.models import ResolvedModelResult

    class _Mgr:
        def resolve_startup_model_with_diagnostics(self, model_name: str, *, tool_name: str, cwd: str, timeout_s=None):
            # validated=False: should not passthrough -m
            resolved = ResolvedModelResult(
                tool_name=tool_name,
                input_name=model_name,
                real_name=model_name,
                source="defaults",
                validated=False,
                warnings=["no_m_passthrough"],
            )
            diag = {
                "attempts": [
                    {"phase": "quick", "validated": False, "source": "defaults"},
                    {
                        "phase": "models",
                        "ok": False,
                        "source": "defaults",
                        "count": 0,
                        "warnings": ["models_untrusted"],
                    },
                ]
            }
            return resolved, diag

    monkeypatch.setattr("src.ttadk.get_ttadk_manager", lambda *a, **k: _Mgr())

    info = resolve_ttadk_engine_startup_model(agent_type="ttadk_codex", cwd="/tmp/test", model_intent="gpt-5.2")
    assert info["validated"] is False
    assert info["resolved_model"] is None
    attempts = list((info.get("diagnostics") or {}).get("attempts") or [])
    assert any(a.get("phase") == "models" for a in attempts)


# ---------------------------------------------------------------------------
# Tests merged from test_deep_progress_recursion.py
# ---------------------------------------------------------------------------


def test_truncate_nested_data():
    # Construct a 15-level nested dict
    nested = {}
    current = nested
    for _ in range(15):
        current["child"] = {}
        current = current["child"]

    current["value"] = 42

    truncated = _truncate_nested_data(nested, max_depth=10)

    # Verify depth 10
    curr = truncated
    for _i in range(10):
        assert "child" in curr
        curr = curr["child"]

    assert curr == "[TRUNCATED: MAX DEPTH EXCEEDED]"


def test_deep_progress_record_tool_truncates():
    nested = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "level5": {"level6": {"level7": {"level8": {"level9": {"level10": {"level11": "too deep"}}}}}}
                    }
                }
            }
        }
    }
    tool_info = ToolCallInfo(id="t1", title="test", kind="read", status="completed", result=nested)

    progress = DeepProgress()
    progress.record_tool(tool_info)

    # Ensure no exception occurred and the result was truncated
    assert (
        progress.tool_calls[0].result["level1"]["level2"]["level3"]["level4"]["level5"]["level6"]["level7"]["level8"][
            "level9"
        ]["level10"]
        == "[TRUNCATED: MAX DEPTH EXCEEDED]"
    )
