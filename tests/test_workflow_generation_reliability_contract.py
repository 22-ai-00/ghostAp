from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.acp.models import ACPEvent, ACPEventType, PromptResult
from src.feishu.handlers.workflow import WorkflowHandler, _WorkflowLifecycleOwner
from src.tasking.scheduler import TaskStatus
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus


def _binding(agent_id: str, tool: str, model: str) -> WorkflowAgentBinding:
    return WorkflowAgentBinding(
        agent_id=agent_id,
        tool_name=tool,
        model_name=model,
        display_name=f"{tool} {model}",
    )


def _pool() -> tuple[WorkflowAgentBinding, ...]:
    return (
        _binding("A-1", "codex", "fast"),
        _binding("A-2", "gemini", "pro"),
    )


def _script(agent_id: str, tool: str) -> str:
    return f'''
export const meta = {{
  name: "pool-generation",
  description: "pool-only generation",
  phases: [{{ title: "Run", detail: "Run work" }}],
  tools: ["{tool}"],
  agentPlan: [{{ node: "main", role: "lead", agentId: "{agent_id}" }}],
}};
export default async function main() {{
  const result = await agent({{
    prompt: "do work", agentId: "{agent_id}", label: "main", timeout: 120,
  }});
  if (result && result.error) throw new Error(result.error);
  return result;
}}
'''


class _Session:
    def __init__(
        self,
        outcome: PromptResult | BaseException,
        *,
        name: str,
        events: list[str] | None = None,
        close_error: BaseException | None = None,
        cancel_result: bool = True,
        on_send: Any = None,
        streamed_events: list[ACPEvent] | None = None,
    ) -> None:
        self.outcome = outcome
        self.name = name
        self.events = events if events is not None else []
        self.close_error = close_error
        self.cancel_result = cancel_result
        self.on_send = on_send
        self.streamed_events = streamed_events or []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.filter = None

    def set_tool_filter(self, callback) -> None:
        self.filter = callback

    def send_prompt(self, prompt: str, **kwargs: Any) -> PromptResult:
        self.calls.append((prompt, kwargs))
        if self.on_send:
            self.on_send()
        callback = kwargs.get("on_event")
        if callback:
            for event in self.streamed_events:
                callback(event)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool:
        self.events.append(f"{self.name}:cancel")
        return self.cancel_result

    def close(self) -> None:
        self.events.append(f"{self.name}:close")
        if self.close_error:
            raise self.close_error


def _handler() -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = SimpleNamespace(
        settings=SimpleNamespace(
            workflow_script_gen_timeout_s=5,
            admin_user_ids=[],
        )
    )
    return handler


def _engine(tmp_path, *, pool=None, orchestrator: str = "A-2") -> WorkflowEngine:
    engine = WorkflowEngine("chat-1", str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingWorkflow(
            requirement="build",
            initiator_user_id="user-1",
            engine_session_key="generation-1",
            project_id="project-1",
            agent_pool=pool or _pool(),
            orchestrator_agent_id=orchestrator,
        ),
    )
    engine._script_generation_owner = _WorkflowLifecycleOwner(
        "generation-1",
        "user-1",
    )
    return engine


def _generate(handler, engine, tmp_path, session_factory, progress_callback=None):
    output = tmp_path / ".ghostap" / "workflow_scripts" / "generated.js"
    with (
        patch("src.agent_session.create_engine_session", side_effect=session_factory),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "gemini": "Gemini", "rogue": "Rogue"},
        ),
    ):
        return handler._generate_script_via_ai(
            "build the workflow",
            str(tmp_path),
            ["rogue"],
            engine,
            progress_callback=progress_callback,
            output_path=str(output),
            cancel_event=threading.Event(),
        )


def test_generation_prompt_and_validator_use_only_confirmed_pool(tmp_path) -> None:
    session = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-2", "gemini")),
        name="A-2",
    )
    created: list[dict[str, Any]] = []

    def factory(**kwargs):
        created.append(kwargs)
        return session

    path, meta = _generate(_handler(), _engine(tmp_path), tmp_path, factory)

    assert path.endswith("generated.js")
    assert meta["agentPlan"][0]["agentId"] == "A-2"
    assert created[0]["agent_type"] == "gemini"
    assert created[0]["model_name"] == "pro"
    prompt = session.calls[0][0]
    assert "A-1" in prompt and "A-2" in prompt
    assert "rogue" not in prompt.lower()


@pytest.mark.parametrize(
    "first_outcome",
    [
        PromptResult(stop_reason="timeout", text=_script("A-2", "gemini")),
        ConnectionError("transport disconnected"),
        RuntimeError("rate limit exceeded"),
    ],
)
def test_timeout_transport_and_rate_limit_fallback_orchestrator_first_with_strict_close(
    tmp_path,
    first_outcome,
) -> None:
    lifecycle: list[str] = []
    first = _Session(first_outcome, name="A-2", events=lifecycle)
    second = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-1", "codex")),
        name="A-1",
        events=lifecycle,
    )
    sessions = iter([first, second])
    created: list[tuple[str, str | None]] = []

    def factory(**kwargs):
        created.append((kwargs["agent_type"], kwargs.get("model_name")))
        return next(sessions)

    _generate(_handler(), _engine(tmp_path), tmp_path, factory)

    assert created == [("gemini", "pro"), ("codex", "fast")]
    assert lifecycle[:2] == ["A-2:cancel", "A-2:close"]


def test_close_failure_blocks_binding_fallback(tmp_path) -> None:
    first = _Session(
        PromptResult(stop_reason="timeout", text=_script("A-2", "gemini")),
        name="A-2",
        close_error=RuntimeError("close uncertain"),
    )
    second = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-1", "codex")),
        name="A-1",
    )
    factory = MagicMock(side_effect=[first, second])

    with pytest.raises(RuntimeError, match="close|关闭|uncertain"):
        _generate(_handler(), _engine(tmp_path), tmp_path, factory)

    assert factory.call_count == 1


def test_cancel_uncertainty_still_closes_and_fences_new_workflow(tmp_path) -> None:
    engine = _engine(tmp_path)
    owner = _WorkflowLifecycleOwner("generation-1", "user-1")
    engine._script_generation_owner = owner
    session = _Session(
        PromptResult(stop_reason="timeout", text=_script("A-2", "gemini")),
        name="A-2",
        cancel_result=False,
    )
    handler = _handler()

    with pytest.raises(RuntimeError, match="cancel|close|uncertain"):
        _generate(handler, engine, tmp_path, MagicMock(return_value=session))

    assert session.events == ["A-2:cancel", "A-2:close"]
    assert owner.active_generation_session is session
    assert not owner.done_event.is_set()
    ok, error, _new_owner = handler._supersede_incomplete_workflow(
        engine,
        root_path=str(tmp_path),
        current_user="user-1",
    )
    assert ok is False
    assert error == "invalid_state"


def test_activity_aware_idle_and_shared_hard_cap_across_attempts(tmp_path) -> None:
    clock = [0.0]
    progress: list[str] = []

    def advance() -> None:
        clock[0] = 550.0

    first = _Session(
        PromptResult(stop_reason="end_turn", text="not javascript"),
        name="A-2",
        on_send=advance,
        streamed_events=[
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=""),
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="working"),
        ],
    )
    second = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-2", "gemini")),
        name="A-2-retry",
    )
    factory = MagicMock(side_effect=[first, second])

    with patch("src.feishu.handlers.workflow.time.monotonic", side_effect=lambda: clock[0]):
        _generate(
            _handler(),
            _engine(tmp_path),
            tmp_path,
            factory,
            progress_callback=progress.append,
        )

    first_kwargs = first.calls[0][1]
    second_kwargs = second.calls[0][1]
    assert 590 <= first_kwargs["timeout"] <= 600
    assert first_kwargs["idle_timeout"] == 120
    assert second_kwargs["timeout"] <= 50
    assert second_kwargs["idle_timeout"] <= 50
    assert any("A-2" in item and "working" in item for item in progress)
    assert not any("heartbeat" in item.lower() for item in progress)


@pytest.mark.parametrize("stop_reason", [None, "", "timeout", "cancelled", "error"])
def test_bad_stop_reason_rejects_valid_looking_script(tmp_path, stop_reason) -> None:
    only = (_binding("A-1", "codex", "fast"),)
    session = _Session(
        PromptResult(stop_reason=stop_reason, text=_script("A-1", "codex")),
        name="A-1",
    )

    with pytest.raises(RuntimeError, match="stop_reason|timeout|cancel|error|终止|失败"):
        _generate(
            _handler(),
            _engine(tmp_path, pool=only, orchestrator="A-1"),
            tmp_path,
            MagicMock(return_value=session),
        )


def test_stale_timeout_attempt_cannot_fallback_or_mutate_new_workflow(tmp_path) -> None:
    engine = _engine(tmp_path)
    replacement = PendingWorkflow(
        requirement="new workflow",
        initiator_user_id="user-1",
        engine_session_key="generation-2",
        project_id="project-1",
        agent_pool=(_binding("A-9", "codex", "new"),),
        orchestrator_agent_id="A-9",
    )

    def supersede() -> None:
        engine.project.pending = replacement

    first = _Session(
        PromptResult(stop_reason="timeout", text=_script("A-2", "gemini")),
        name="old",
        on_send=supersede,
    )
    factory = MagicMock(return_value=first)

    with pytest.raises(RuntimeError, match="stale|superseded|已替换|session"):
        _generate(_handler(), engine, tmp_path, factory)

    assert engine.project.pending is replacement
    assert factory.call_count == 1


def test_create_bind_race_rejects_new_workflow_and_keeps_original_prompt(tmp_path) -> None:
    handler = _handler()
    engine = _engine(tmp_path)
    old_owner = _WorkflowLifecycleOwner("generation-1", "user-1")
    engine._script_generation_owner = old_owner
    old_created = threading.Event()
    return_old = threading.Event()
    tracker_lock = threading.Lock()
    active_prompts = 0
    max_active_prompts = 0

    def track_old_send() -> None:
        nonlocal active_prompts, max_active_prompts
        with tracker_lock:
            active_prompts += 1
            max_active_prompts = max(max_active_prompts, active_prompts)
            active_prompts -= 1

    old_session = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-2", "gemini")),
        name="old",
        on_send=track_old_send,
    )
    outcomes: dict[str, object] = {}

    def factory(**kwargs):
        if "generation-1" in kwargs["thread_id"]:
            old_created.set()
            assert return_old.wait(timeout=5)
            return old_session
        raise AssertionError("a generating Workflow must not create a replacement session")

    def run_generation(label, owner, output_name) -> None:
        try:
            outcomes[label] = handler._generate_script_via_ai(
                "build",
                str(tmp_path),
                ["codex", "gemini"],
                engine,
                output_path=str(
                    tmp_path / ".ghostap" / "workflow_scripts" / output_name
                ),
                cancel_event=owner.stop_event,
            )
        except BaseException as exc:
            outcomes[label] = exc

    with (
        patch("src.agent_session.create_engine_session", side_effect=factory),
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"codex": "Codex", "gemini": "Gemini"},
        ),
    ):
        old_thread = threading.Thread(
            target=run_generation,
            args=("old", old_owner, "old.js"),
        )
        old_thread.start()
        assert old_created.wait(timeout=5)

        supersede_result = handler._supersede_incomplete_workflow(
            engine,
            root_path=str(tmp_path),
            current_user="user-1",
        )
        return_old.set()
        old_thread.join(timeout=5)
        assert not old_thread.is_alive()

        ok, error, admission_owner = supersede_result
        assert ok is False
        assert error == "invalid_state"
        assert admission_owner is None
        with engine._lock:
            assert engine._script_generation_owner is old_owner
            assert engine.project.pending.engine_session_key == "generation-1"

    assert not isinstance(outcomes["old"], BaseException)
    assert len(old_session.calls) == 1
    assert max_active_prompts == 1


def test_stop_cancels_current_local_generation_session_and_commits_cancelled(tmp_path) -> None:
    engine = _engine(tmp_path)
    session = _Session(
        PromptResult(stop_reason="cancelled"),
        name="active",
    )
    owner = _WorkflowLifecycleOwner(
        "generation-1",
        "user-1",
        active_generation_session=session,
    )
    engine._script_generation_owner = owner
    handler = _handler()
    handler.ctx.workflow_engine_manager = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    handler._remove_owned_workflow_artifact = MagicMock()
    handler.reply_text = MagicMock()

    with patch("src.thread.get_current_sender_id", return_value="user-1"):
        handler.stop_workflow("stop-1", "chat-1", None)

    assert session.events[:2] == ["active:cancel", "active:close"]
    assert engine.project.status is WorkflowStatus.CANCELLED
    assert engine.project.finished_at is not None
    handler.reply_text.assert_called_once_with("stop-1", "Workflow 任务已停止。")


def test_three_generation_failures_commit_failed_and_emit_one_error_card(tmp_path) -> None:
    engine = _engine(tmp_path)
    owner = _WorkflowLifecycleOwner("generation-1", "user-1")
    engine._script_generation_owner = owner
    handler = _handler()
    handler.ctx.workflow_engine_manager = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    queued: dict[str, Any] = {}
    handler._submit_engine_task = MagicMock(
        side_effect=lambda run_fn, *_args, **_kwargs: queued.setdefault("run", run_fn)
        or SimpleNamespace(run_id="run-1")
    )
    handler._generate_and_start_workflow = MagicMock(
        side_effect=RuntimeError("token=super-secret exhausted after 3 attempts")
    )
    handler._reply_workflow_error = MagicMock()

    handler._schedule_generate_and_start_workflow(
        message_id="origin-1",
        chat_id="chat-1",
        requirement="build",
        project=SimpleNamespace(
            project_id="project-1",
            project_name="Project",
            root_path=str(tmp_path),
        ),
        root_path=str(tmp_path),
        selected_tools=["codex", "gemini"],
        expected_session_key="generation-1",
        engine=engine,
    )
    queued["run"]()

    assert engine.project.status is WorkflowStatus.FAILED
    assert engine.project.finished_at is not None
    assert engine.project.error
    assert "super-secret" not in engine.project.error
    assert handler._reply_workflow_error.call_count == 1


def test_scheduler_pre_callback_failure_reconciles_generation_once(tmp_path) -> None:
    engine = _engine(tmp_path)
    owner = _WorkflowLifecycleOwner("generation-1", "user-1")
    engine._script_generation_owner = owner
    handler = _handler()
    handler.ctx.workflow_engine_manager = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.scheduler = MagicMock()
    handler.ctx.scheduler.get_state.return_value = SimpleNamespace(
        status=TaskStatus.FAILED,
        error="run guard failed",
    )
    handler._submit_engine_task = MagicMock(return_value=SimpleNamespace(run_id="run-1"))
    handler._reply_workflow_error = MagicMock()

    handler._schedule_generate_and_start_workflow(
        message_id="origin-1",
        chat_id="chat-1",
        requirement="build",
        project=SimpleNamespace(
            project_id="project-1",
            project_name="Project",
            root_path=str(tmp_path),
        ),
        root_path=str(tmp_path),
        selected_tools=["codex", "gemini"],
        expected_session_key="generation-1",
        engine=engine,
    )

    assert engine.project.status is WorkflowStatus.FAILED
    assert engine.project.finished_at is not None
    assert handler._reply_workflow_error.call_count == 1


def test_submit_engine_task_returns_scheduler_handle(tmp_path) -> None:
    handler = _handler()
    expected = SimpleNamespace(run_id="run-1")
    handler.ctx.scheduler = MagicMock()
    handler.ctx.scheduler.submit.return_value = expected
    handler.ctx.message_linker = MagicMock()
    project = SimpleNamespace(
        project_id="project-1",
        root_path=str(tmp_path),
    )

    handle = handler._submit_engine_task(
        lambda: None,
        "chat-1",
        "origin-1",
        project,
        request_id=None,
        task_id="task-1",
    )

    assert handle is expected
