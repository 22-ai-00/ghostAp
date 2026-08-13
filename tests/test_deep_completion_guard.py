"""Deep automatic progression and terminal-state contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.acp.models import PlanEntryInfo, PlanInfo, PromptResult
from src.acp.outcome import classify_prompt_result
from src.deep_engine.engine import DeepEngine
from src.deep_engine.models import DeepProjectStatus, EngineRunState


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        coco_execution_timeout=300,
        claude_execution_timeout=600,
        deep_memory_threshold=99.0,
        programming_finalization_reserve_s=0,
    )


class _Session:
    def __init__(self, *results: PromptResult):
        self._results = list(results)
        self._last = results[-1]
        self._force_dead = False
        self.prompts: list[str] = []

    def send_prompt_with_retry(self, text: str, **_kwargs) -> PromptResult:
        self.prompts.append(text)
        return self._results.pop(0) if self._results else self._last

    def cancel(self) -> None:
        return None


def _run(session: _Session):
    with (
        patch("src.engine_base.get_settings", return_value=_settings()),
        patch("src.deep_engine.engine.create_engine_session", return_value=session),
        patch("src.deep_engine.engine.get_gc_monitor"),
    ):
        engine = DeepEngine(
            chat_id="chat-1",
            root_path="/repo",
            agent_type="codex",
            engine_name="Codex",
        )
        return engine, engine.plan_and_execute("完成所有验收项")


def test_deep_automatically_continues_unfinished_plan() -> None:
    session = _Session(
        PromptResult(
            stop_reason="end_turn",
            plan=PlanInfo(entries=[PlanEntryInfo(content="运行验收", status="in_progress")]),
        ),
        PromptResult(
            stop_reason="end_turn",
            plan=PlanInfo(entries=[PlanEntryInfo(content="运行验收", status="completed")]),
        ),
    )

    _engine, project = _run(session)

    assert project.status is DeepProjectStatus.COMPLETED
    assert len(session.prompts) == 2
    assert "GhostAP 自动续做指令" in session.prompts[1]


def test_deep_automatically_recovers_blocked_goal() -> None:
    session = _Session(
        PromptResult(
            stop_reason="end_turn",
            goal=SimpleNamespace(status="blocked"),
        ),
        PromptResult(stop_reason="end_turn"),
    )

    _engine, project = _run(session)

    assert project.status is DeepProjectStatus.COMPLETED
    assert "GhostAP 自动恢复指令" in session.prompts[1]


def test_deep_automatically_chooses_safe_default_instead_of_waiting() -> None:
    session = _Session(
        PromptResult(stop_reason="end_turn", text="请选择推荐方案还是最小方案？"),
        PromptResult(stop_reason="end_turn", text="已采用推荐的安全可逆方案并完成。"),
    )

    _engine, project = _run(session)

    assert project.status is DeepProjectStatus.COMPLETED
    assert "GhostAP 自动续做默认决策" in session.prompts[1]
    assert "不得发布、部署、付费、删除数据" in session.prompts[1]


def test_deep_fails_after_bounded_automatic_continuations() -> None:
    incomplete = PromptResult(
        stop_reason="end_turn",
        plan=PlanInfo(entries=[PlanEntryInfo(content="仍未完成", status="in_progress")]),
    )
    session = _Session(incomplete)

    _engine, project = _run(session)

    assert project.status is DeepProjectStatus.FAILED
    assert project.error and "仍有 1 个计划项未完成" in project.error
    assert len(session.prompts) == 4


def test_deep_preserves_pending_context_when_follow_up_is_incomplete() -> None:
    follow_up_result = PromptResult(stop_reason="timeout")
    session = _Session(follow_up_result)
    with patch("src.engine_base.get_settings", return_value=_settings()):
        engine = DeepEngine(
            chat_id="chat-1",
            root_path="/repo",
            agent_type="codex",
            engine_name="Codex",
        )
    engine._run_state = EngineRunState.RUNNING
    engine._session = session
    engine._pending_context = ["不能丢失的新增验收条件"]

    previous_result = PromptResult(stop_reason="end_turn")
    result, assessment = engine._drain_pending_context(
        on_event=lambda _event: None,
        timeout=10,
        last_result=previous_result,
        last_assessment=classify_prompt_result(previous_result),
    )

    assert result is follow_up_result
    assert assessment.stop_reason == "timeout"
    assert engine._pending_context == ["不能丢失的新增验收条件"]


def test_deep_cancelled_transport_is_terminal_not_paused() -> None:
    _engine, project = _run(
        _Session(
            PromptResult(
                stop_reason="cancelled",
                cancellation_source="user",
            )
        )
    )

    assert project.status is DeepProjectStatus.CANCELLED
    assert project.completed_at is not None
