"""Tests for deep_engine — ACP-driven DeepEngine."""

from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import PlanEntryInfo, PlanInfo, ToolCallInfo
from src.deep_engine.engine import DeepEngine, DeepEngineCallbacks, DeepEngineManager
from src.deep_engine.models import DeepProject, DeepProjectStatus, EngineRunState
from src.deep_engine.progress import DeepProgress, _truncate_nested_data


class _DummySession:
    """Minimal session stub shared by Deep startup tests."""

    def __init__(self, *a, **k):
        self.session_id = "sid"
        self.created_at = 0.0
        self.last_active = 0.0
        self.message_count = 0
        self.last_query = ""

    def describe_agent(self):
        return "dummy"

    def start(self, startup_timeout: float = 60, **kwargs):
        return "sid"


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
        project.start()

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

        class TimeoutSession:
            _force_dead = False

            def send_prompt_with_retry(self, *_args, **_kwargs):
                raise TimeoutError("deadline")

        session = TimeoutSession()
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
