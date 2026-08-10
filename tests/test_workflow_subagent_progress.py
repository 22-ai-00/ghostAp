"""Workflow card synchronization for ACP-internal subagents."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.executor import AgentExecutor
from src.workflow_engine.models import (
    AgentCallParams,
    AgentCallResult,
    AgentProgress,
    AgentStatus,
    PhaseProgress,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.renderer import WorkflowProgressRenderer
from src.workflow_engine.state_manager import WorkflowStateManager


def _card_text(value: object) -> str:
    if isinstance(value, dict):
        return "\n".join(_card_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_card_text(item) for item in value)
    return value if isinstance(value, str) else ""


class _EventSession:
    def __init__(self, events: list[ACPEvent]) -> None:
        self._events = events

    def send_prompt(self, _text: str, *, on_event=None, **_kwargs):
        for event in self._events:
            if on_event is not None:
                on_event(event)
        return SimpleNamespace(
            text="outer agent completed",
            output_tokens=0,
            stop_reason="end_turn",
        )

    def cancel(self) -> None:
        return None

    def close(self) -> None:
        return None


def _tool_event(
    event_type: ACPEventType,
    *,
    call_id: str,
    source_id: str | None = None,
    activity: str | None = None,
    receivers: tuple[str, ...] = (),
    states: tuple[dict, ...] = (),
    status: str = "completed",
) -> ACPEvent:
    return ACPEvent(
        event_type=event_type,
        source_id=source_id,
        tool_call=ToolCallInfo(
            id=call_id,
            title="agent",
            kind="agent",
            status=status,
            subagent_source_id=source_id,
            subagent_activity=activity,
            collaboration_tool="spawn_agent" if receivers else None,
            collaboration_receivers=receivers,
            collaboration_model="gpt-5.6-sol/ultra" if receivers else None,
            subagent_states=states,
        ),
    )


def test_executor_preserves_legacy_positional_max_workers(tmp_path) -> None:
    def token_callback(_tokens):
        return None

    def activity_callback(_label, _activity):
        return None

    executor = AgentExecutor(
        str(tmp_path),
        threading.Event(),
        token_callback,
        activity_callback,
        3,
    )
    try:
        assert executor._session_pool._max_workers == 3
        assert executor.on_subagent_update is None
    finally:
        executor.shutdown(wait=True)


def test_executor_forwards_stable_subagent_updates_without_false_activity_completion(
    tmp_path,
) -> None:
    """Different tool-call IDs for one child must keep its thread identity.

    In particular, neither a completed ACP activity notification nor a nested
    ``agentsStates`` snapshot is authoritative evidence that the child finished.
    """
    thread_a = "0192a4a7-aaaa-7bbb-8ccc-111111111111"
    thread_b = "0192a4a7-bbbb-7ccc-8ddd-222222222222"
    events = [
        _tool_event(
            ACPEventType.TOOL_CALL_START,
            call_id="call_spawn",
            receivers=(thread_a, thread_b),
            states=(
                {
                    "source_id": thread_a,
                    "status": "running",
                    "message": "正在核查 Workflow 卡片",
                },
            ),
            status="in_progress",
        ),
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="call_interact",
            source_id=thread_a,
            activity="interacted",
            status="completed",
        ),
        _tool_event(
            ACPEventType.TOOL_CALL_START,
            call_id="call_interrupt_start",
            source_id=thread_b,
            activity="interrupted",
            status="in_progress",
        ),
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="call_interrupt_failed",
            source_id=thread_b,
            activity="interrupted",
            status="failed",
        ),
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="call_interrupt",
            source_id=thread_b,
            activity="interrupted",
            status="completed",
        ),
        _tool_event(
            ACPEventType.TOOL_CALL_DONE,
            call_id="call_wait",
            receivers=(thread_a,),
            states=(
                {
                    "source_id": thread_a,
                    "status": "completed",
                    "message": "Workflow 卡片核查完成",
                },
            ),
        ),
    ]
    updates: list[tuple[str, tuple[dict, ...]]] = []
    executor = AgentExecutor(
        cwd=str(tmp_path),
        cancel_event=threading.Event(),
        on_subagent_update=lambda label, batch: updates.append((label, batch)),
    )

    try:
        with patch(
            "src.agent_session.factory.create_engine_session",
            return_value=_EventSession(events),
        ):
            result = executor.execute(
                AgentCallParams(prompt="audit", tool="codex", label="outer-a")
            )
    finally:
        executor.shutdown(wait=True)

    assert result.error is None
    assert [label for label, _ in updates] == ["outer-a"] * 6
    assert updates[0][1] == (
        {
            "source_id": thread_a,
            "status": "running",
            "progress": "正在核查 Workflow 卡片",
            "model": "gpt-5.6-sol/ultra",
        },
        {
            "source_id": thread_b,
            "status": "running",
            "progress": "已启动",
            "model": "gpt-5.6-sol/ultra",
        },
    )
    assert updates[1][1] == (
        {
            "source_id": thread_a,
            "status": "running",
            "progress": "已与主 Agent 交互",
            "model": None,
        },
    )
    assert updates[2][1] == (
        {
            "source_id": thread_b,
            "status": "running",
            "progress": "正在中断",
            "model": None,
        },
    )
    assert updates[3][1] == (
        {
            "source_id": thread_b,
            "status": "running",
            "progress": "中断未完成",
            "model": None,
        },
    )
    assert updates[4][1][0]["status"] == "cancelled"
    assert updates[5][1][0]["status"] == "running"
    assert updates[5][1][0]["progress"] == "Workflow 卡片核查完成"


def test_state_manager_merges_subagent_updates_by_thread_and_deduplicates() -> None:
    project = WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
    manager = WorkflowStateManager(project)
    outer = manager.on_agent_started("outer", "codex", "执行")
    thread_id = "0192a4a7-aaaa-7bbb-8ccc-111111111111"

    assert manager.update_agent_subagents(
        outer,
        (
            {
                "source_id": thread_id,
                "status": "running",
                "progress": "已启动",
                "model": "gpt-5.6-sol/ultra",
            },
        ),
    )
    assert manager.update_agent_subagents(
        outer,
        (
            {
                "source_id": thread_id,
                "status": "running",
                "progress": "已与主 Agent 交互",
                "model": None,
            },
        ),
    )
    assert not manager.update_agent_subagents(
        outer,
        (
            {
                "source_id": thread_id,
                "status": "running",
                "progress": "已与主 Agent 交互",
                "model": None,
            },
        ),
    )

    agent = manager.snapshot().phases[0].agents[0]
    assert len(agent.subagents) == 1
    assert agent.subagents[0].source_id == thread_id
    assert agent.subagents[0].progress == "已与主 Agent 交互"


def test_duplicate_outer_labels_route_children_to_the_effective_agent(tmp_path) -> None:
    """A generated workflow may reuse labels; child activity must not cross."""
    engine = WorkflowEngine("chat", str(tmp_path), agent_type="codex")
    project = WorkflowProject(name="wf", status=WorkflowStatus.RUNNING)
    engine._project = project
    engine._state_manager = WorkflowStateManager(project)
    engine._journal = None
    engine._callbacks = None
    engine._run_spec = None

    seen = 0

    def execute(params, **_kwargs):
        nonlocal seen
        seen += 1
        engine._handle_agent_subagent_update(
            params.label,
            (
                {
                    "source_id": f"thread-{seen}",
                    "status": "running",
                    "progress": f"child-{seen}",
                    "model": None,
                },
            ),
        )
        return AgentCallResult(
            output="ok",
            stop_reason="end_turn",
            tool="codex",
        )

    engine._executor = SimpleNamespace(execute=execute)

    for _ in range(2):
        result = engine._handle_agent_call(
            AgentCallParams(
                prompt="audit",
                tool="codex",
                label="duplicate",
                phase="执行",
            ),
            allow_cache=False,
        )
        assert result.error is None

    agents = engine._state_manager.snapshot().phases[0].agents
    assert [agent.label for agent in agents] == ["duplicate", "duplicate #2"]
    assert [child.progress for child in agents[0].subagents] == ["child-1"]
    assert [child.progress for child in agents[1].subagents] == ["child-2"]


def test_renderer_shows_safe_subagent_status_without_internal_ids() -> None:
    thread_id = "0192a4a7-aaaa-7bbb-8ccc-111111111111"
    project = WorkflowProject(
        name="wf",
        status=WorkflowStatus.RUNNING,
        phases=[
            PhaseProgress(
                title="执行",
                agents=[
                    AgentProgress(
                        label="outer",
                        tool="codex",
                        status=AgentStatus.RUNNING,
                        subagents=[
                            {
                                "source_id": thread_id,
                                "status": "running",
                                "progress": (
                                    f"正在检查 {thread_id} TOKEN=super-secret "
                                    "/data00/home/user/private.py:12 ![x](file:///tmp/x.png)"
                                ),
                                "model": "gpt-5.6-sol/ultra",
                            },
                            {
                                "source_id": "0192a4a7-bbbb-7ccc-8ddd-222222222222",
                                "status": "cancelled",
                                "progress": "已中断",
                            },
                        ],
                    )
                ],
            )
        ],
    )

    text = _card_text(WorkflowProgressRenderer(project).render_progress_card())

    assert "子 Agent 1" in text
    assert "执行中" in text
    assert "子 Agent 2" in text
    assert "已取消" in text
    assert "非权威，不参与终态判定" in text
    assert "观测执行中" in text
    assert "当前操作：" in text
    assert "TOKEN=" in text
    assert "redacted" in text
    assert thread_id not in text
    assert "super-secret" not in text
    assert "/data00/home/user/private.py:12" not in text
    assert "![" not in text
