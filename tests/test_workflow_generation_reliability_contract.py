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
        filter_error: BaseException | None = None,
        on_send: Any = None,
        streamed_events: list[ACPEvent] | None = None,
    ) -> None:
        self.outcome = outcome
        self.name = name
        self.events = events if events is not None else []
        self.close_error = close_error
        self.cancel_result = cancel_result
        self.filter_error = filter_error
        self.on_send = on_send
        self.streamed_events = streamed_events or []
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.filter = None
        self.event_callback = None

    def set_tool_filter(self, callback) -> None:
        if self.filter_error:
            raise self.filter_error
        self.filter = callback

    def send_prompt(self, prompt: str, **kwargs: Any) -> PromptResult:
        self.calls.append((prompt, kwargs))
        self.event_callback = kwargs.get("on_event")
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


def _generate(
    handler,
    engine,
    tmp_path,
    session_factory,
    progress_callback=None,
    *,
    cancel_event: threading.Event | None = None,
):
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
            cancel_event=cancel_event or threading.Event(),
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


def test_attempt_timeout_cancellation_does_not_set_workflow_stop_event(tmp_path) -> None:
    workflow_stop = threading.Event()
    created_cancel_events: list[threading.Event] = []
    outcomes = iter(
        [
            TimeoutError("attempt hard cap reached"),
            PromptResult(stop_reason="end_turn", text=_script("A-1", "codex")),
        ]
    )

    class _EventCoupledSession(_Session):
        def __init__(self, outcome, *, name: str, cancel_event: threading.Event):
            super().__init__(outcome, name=name)
            self.cancel_event = cancel_event

        def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool:
            self.cancel_event.set()
            return super().cancel(wait=wait, timeout=timeout)

    def factory(**kwargs):
        attempt_cancel = kwargs["cancel_event"]
        created_cancel_events.append(attempt_cancel)
        return _EventCoupledSession(
            next(outcomes),
            name=kwargs["agent_type"],
            cancel_event=attempt_cancel,
        )

    path, meta = _generate(
        _handler(),
        _engine(tmp_path),
        tmp_path,
        factory,
        cancel_event=workflow_stop,
    )

    assert path.endswith("generated.js")
    assert meta["agentPlan"][0]["agentId"] == "A-1"
    assert workflow_stop.is_set() is False
    assert len(created_cancel_events) == 2
    assert all(event is not workflow_stop for event in created_cancel_events)


def test_read_only_close_failure_is_quarantined_and_binding_fallback_continues(
    tmp_path,
) -> None:
    progress: list[str] = []
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

    engine = _engine(tmp_path)
    path, meta = _generate(
        _handler(),
        engine,
        tmp_path,
        factory,
        progress_callback=progress.append,
    )

    assert path.endswith("generated.js")
    assert meta["agentPlan"][0]["agentId"] == "A-1"
    assert factory.call_count == 2
    assert engine.has_uncertain_lifecycle_session() is False
    assert engine._script_generation_owner.active_generation_session is None

    assert first.event_callback is not None
    first.event_callback(
        ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="late stale activity")
    )
    assert not any("late stale activity" in item for item in progress)


def test_read_only_cancel_uncertainty_is_quarantined_and_does_not_fence(
    tmp_path,
) -> None:
    engine = _engine(tmp_path)
    owner = _WorkflowLifecycleOwner("generation-1", "user-1")
    engine._script_generation_owner = owner
    session = _Session(
        PromptResult(stop_reason="timeout", text=_script("A-2", "gemini")),
        name="A-2",
        cancel_result=False,
    )
    handler = _handler()

    fallback = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-1", "codex")),
        name="A-1",
    )
    factory = MagicMock(side_effect=[session, fallback])

    path, _meta = _generate(handler, engine, tmp_path, factory)

    assert path.endswith("generated.js")
    assert session.events == ["A-2:cancel", "A-2:close"]
    assert owner.active_generation_session is None
    assert engine.has_uncertain_lifecycle_session() is False


def test_unfenced_session_cleanup_uncertainty_still_blocks_fallback(tmp_path) -> None:
    engine = _engine(tmp_path)
    owner = engine._script_generation_owner
    session = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-2", "gemini")),
        name="A-2",
        filter_error=RuntimeError("filter installation failed"),
        close_error=RuntimeError("close uncertain"),
    )

    with pytest.raises(RuntimeError, match="close|uncertain"):
        _generate(_handler(), engine, tmp_path, MagicMock(return_value=session))

    assert owner.active_generation_session is session
    assert engine.has_uncertain_lifecycle_session() is True


def test_fair_member_slice_preserves_later_binding_budget_and_bounds_repairs(
    tmp_path,
) -> None:
    clock = [0.0]
    progress: list[str] = []
    handler = _handler()
    handler.ctx.settings.workflow_script_gen_timeout_s = 180

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
        PromptResult(stop_reason="end_turn", text=_script("A-1", "codex")),
        name="A-1",
    )
    factory = MagicMock(side_effect=[first, second])

    with patch("src.feishu.handlers.workflow.time.monotonic", side_effect=lambda: clock[0]):
        _generate(
            handler,
            _engine(tmp_path),
            tmp_path,
            factory,
            progress_callback=progress.append,
        )

    first_kwargs = first.calls[0][1]
    second_kwargs = second.calls[0][1]
    assert 170 <= first_kwargs["timeout"] <= 180
    assert first_kwargs["idle_timeout"] == 120
    assert second_kwargs["timeout"] <= 50
    assert second_kwargs["idle_timeout"] <= 50
    assert factory.call_args_list[0].kwargs["startup_log_failures"] is False
    assert factory.call_args_list[1].kwargs["startup_log_failures"] is False
    assert any("A-2" in item and "working" in item for item in progress)
    assert not any("heartbeat" in item.lower() for item in progress)


def test_unused_member_slice_rolls_forward_to_later_binding(tmp_path) -> None:
    clock = [0.0]
    handler = _handler()
    handler.ctx.settings.workflow_script_gen_timeout_s = 600

    def use_part_of_first_slice() -> None:
        clock[0] = 100.0

    first = _Session(
        PromptResult(stop_reason="timeout", text=""),
        name="A-2",
        on_send=use_part_of_first_slice,
    )
    second = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-1", "codex")),
        name="A-1",
    )

    with patch("src.feishu.handlers.workflow.time.monotonic", side_effect=lambda: clock[0]):
        _generate(
            handler,
            _engine(tmp_path),
            tmp_path,
            MagicMock(side_effect=[first, second]),
        )

    assert first.calls[0][1]["timeout"] == pytest.approx(300.0)
    assert second.calls[0][1]["timeout"] == pytest.approx(500.0)


def test_generation_activity_coalesces_token_chunks_into_complete_sentences(
    tmp_path,
) -> None:
    progress: list[str] = []
    session = _Session(
        PromptResult(stop_reason="end_turn", text=_script("A-2", "gemini")),
        name="A-2",
        streamed_events=[
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="首先"),
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="，"),
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="我会"),
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="分析"),
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="现有卡片。"),
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="然后"),
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="给出方案。"),
        ],
    )

    _generate(
        _handler(),
        _engine(tmp_path),
        tmp_path,
        MagicMock(return_value=session),
        progress_callback=progress.append,
    )

    live_activity = [item for item in progress if "last activity:" in item]
    assert live_activity == [
        "A-2 last activity: 首先，我会分析现有卡片。",
        "A-2 last activity: 然后给出方案。",
    ]


@pytest.mark.parametrize("stop_reason", [None, "", "timeout", "cancelled", "error"])
def test_bad_provider_stop_reason_uses_deterministic_pool_fallback(
    tmp_path,
    stop_reason,
) -> None:
    only = (_binding("A-1", "codex", "fast"),)
    session = _Session(
        PromptResult(stop_reason=stop_reason, text=_script("A-1", "codex")),
        name="A-1",
    )

    output_path, meta = _generate(
        _handler(),
        _engine(tmp_path, pool=only, orchestrator="A-1"),
        tmp_path,
        MagicMock(return_value=session),
    )

    content = (tmp_path / ".ghostap" / "workflow_scripts" / "generated.js").read_text()
    assert output_path.endswith("generated.js")
    assert meta["agentPlan"][0]["agentId"] == "A-1"
    assert 'agentId: "A-1"' in content
    assert "rogue" not in content


def test_deterministic_fallback_failure_is_typed_and_hides_cleanup_details(
    tmp_path,
) -> None:
    from src.feishu.handlers.workflow import _WorkflowGenerationExhausted

    first = _Session(
        PromptResult(stop_reason="timeout", text=""),
        name="A-2",
        close_error=RuntimeError("private close uncertain"),
    )
    second = _Session(
        PromptResult(stop_reason="error", text=""),
        name="A-1",
    )

    with patch(
        "src.workflow_engine.script_gen.generate_simple_script",
        return_value="not a valid workflow",
    ):
        with pytest.raises(_WorkflowGenerationExhausted) as raised:
            _generate(
                _handler(),
                _engine(tmp_path),
                tmp_path,
                MagicMock(side_effect=[first, second]),
            )

    assert "private" not in str(raised.value)
    assert "close" not in str(raised.value).lower()


def test_explicit_owner_stop_wins_before_deterministic_fallback_write(tmp_path) -> None:
    from src.feishu.handlers.workflow import _WorkflowGenerationCancelled

    only = (_binding("A-1", "codex", "fast"),)
    stop_event = threading.Event()
    session = _Session(
        PromptResult(stop_reason="cancelled", text=""),
        name="A-1",
    )

    def stop_then_generate(*args, **kwargs):
        del args, kwargs
        stop_event.set()
        return _script("A-1", "codex")

    with patch(
        "src.workflow_engine.script_gen.generate_simple_script",
        side_effect=stop_then_generate,
    ):
        with pytest.raises(_WorkflowGenerationCancelled):
            _generate(
                _handler(),
                _engine(tmp_path, pool=only, orchestrator="A-1"),
                tmp_path,
                MagicMock(return_value=session),
                cancel_event=stop_event,
            )

    assert not (
        tmp_path / ".ghostap" / "workflow_scripts" / "generated.js"
    ).exists()


def test_workflow_internal_error_card_never_tells_user_to_contact_admin() -> None:
    card = _handler()._build_error_card(
        "internal_error",
        detail=(
            "generation session cancellation was not confirmed; "
            "generation session close uncertain"
        ),
    )

    rendered = str(card)
    assert "联系管理员" not in rendered
    assert "generation session" not in rendered
    assert "/wf" in rendered


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
    handler._build_error_card = MagicMock(return_value={"schema": "2.0", "body": {"elements": []}})
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock()
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
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args.args[0] == "origin-1"
    handler.send_card_to_chat.assert_not_called()
    handler._reply_workflow_error.assert_not_called()


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
    handler._build_error_card = MagicMock(return_value={"schema": "2.0", "body": {"elements": []}})
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock()
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
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args.args[0] == "origin-1"
    handler.send_card_to_chat.assert_not_called()
    handler._reply_workflow_error.assert_not_called()


def test_generation_worker_cancellation_replaces_same_card_with_terminal_state(
    tmp_path,
) -> None:
    from src.feishu.handlers.workflow import _WorkflowGenerationCancelled

    engine = _engine(tmp_path)
    pending = engine.project.pending
    pending.engine_session_key = "generation-1"
    owner = _WorkflowLifecycleOwner(
        "generation-1",
        "user-1",
        chat_id="chat-1",
        project_id=pending.project_id or "project-1",
        root_path=str(tmp_path),
        source_script_path=str(tmp_path / "cancelled.js"),
    )
    engine._script_generation_owner = owner
    handler = _handler()
    handler.ctx.workflow_engine_manager = MagicMock()
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.get_engine_name = MagicMock(return_value="codex")
    handler._resolve_origin = MagicMock(return_value="origin-1")
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="unexpected-new-card")
    handler._remove_owned_workflow_artifact = MagicMock()
    handler._generate_script_via_ai = MagicMock(
        side_effect=_WorkflowGenerationCancelled("Workflow generation cancelled")
    )

    with patch("src.feishu.handlers.workflow.threading.Thread"):
        handler._generate_and_start_workflow(
            message_id="generation-card",
            chat_id="chat-1",
            requirement=pending.requirement or "build",
            project=SimpleNamespace(
                project_id=pending.project_id or "project-1",
                project_name="Project",
                root_path=str(tmp_path),
            ),
            root_path=str(tmp_path),
            selected_tools=list(pending.selected_tools or ["codex"]),
            expected_session_key="generation-1",
        )

    assert engine.project.status is WorkflowStatus.CANCELLED
    assert len(handler.update_card.call_args_list) == 2
    terminal_update = handler.update_card.call_args_list[-1]
    assert terminal_update.args[0] == "generation-card"
    assert "取消" in str(terminal_update.args[1])
    assert "生成编排中" not in str(terminal_update.args[1])
    handler.send_card_to_chat.assert_not_called()


def test_scheduler_cancellation_replaces_same_card_with_terminal_state(tmp_path) -> None:
    from src.feishu.handlers.workflow import _WorkflowGenerationCancelled

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
        side_effect=_WorkflowGenerationCancelled("Workflow generation cancelled")
    )
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock()
    handler._reply_workflow_error = MagicMock()

    handler._schedule_generate_and_start_workflow(
        message_id="generation-card",
        chat_id="chat-1",
        requirement="build",
        project=SimpleNamespace(
            project_id="project-1",
            project_name="Project",
            root_path=str(tmp_path),
        ),
        root_path=str(tmp_path),
        selected_tools=["codex"],
        expected_session_key="generation-1",
        engine=engine,
    )
    queued["run"]()

    assert engine.project.status is WorkflowStatus.CANCELLED
    handler.update_card.assert_called_once()
    assert handler.update_card.call_args.args[0] == "generation-card"
    assert "取消" in str(handler.update_card.call_args.args[1])
    assert "生成编排中" not in str(handler.update_card.call_args.args[1])
    handler.send_card_to_chat.assert_not_called()
    handler._reply_workflow_error.assert_not_called()


def test_generated_workflow_validation_failure_does_not_reply_a_second_card(
    tmp_path,
) -> None:
    engine = _engine(tmp_path)
    pending = engine.project.pending
    pending.script_path = None
    owner = _WorkflowLifecycleOwner(
        pending.engine_session_key or "generation-1",
        pending.initiator_user_id or "user-1",
        chat_id="chat-1",
        project_id=pending.project_id or "project-1",
        root_path=str(tmp_path),
    )
    pending.engine_session_key = owner.session_key
    engine._script_generation_owner = owner
    handler = _handler()
    handler._reply_workflow_error = MagicMock()

    handler._queue_generated_workflow(
        message_id="generation-card",
        chat_id="chat-1",
        project=SimpleNamespace(
            project_id=pending.project_id or "project-1",
            project_name="Project",
            root_path=str(tmp_path),
        ),
        root_path=str(tmp_path),
        engine=engine,
        generation_owner=owner,
    )

    handler._reply_workflow_error.assert_not_called()


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
