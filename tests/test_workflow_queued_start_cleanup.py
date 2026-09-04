"""Regression coverage for queued Workflow start cleanup linearization."""

from __future__ import annotations

import hashlib
import itertools
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.feishu.handlers.workflow import (
    WorkflowHandler,
    _WorkflowLifecycleOwner,
)
from src.workflow_engine.agent_pool import WorkflowAgentBinding
from src.workflow_engine.engine import WorkflowEngine, WorkflowEngineCallbacks
from src.workflow_engine.manager import WorkflowEngineManager
from src.workflow_engine.models import (
    PendingWorkflow,
    ReviewAgentBinding,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.storage import workflow_scripts_dir


def test_cleanup_waits_for_started_preclaim_closure_and_fences_repo_lock(
    tmp_path,
) -> None:
    """Cleanup must wait for a queued closure that has started but not claimed.

    The closure is paused after its initial cancellation check and before
    ``_safe_execute_engine`` can acquire the repository lock.  Cleanup must
    cancel it without forging ``done_event`` or reporting quiescence early.
    Once released, the stale closure must retire before touching the repo lock.
    """
    script_dir = Path(workflow_scripts_dir(str(tmp_path)))
    script_dir.mkdir(parents=True)
    script_path = script_dir / "generated-workflow-preclaim.js"
    script_text = (
        "export const meta = {\n"
        "  name: 'preclaim-cleanup',\n"
        "  description: 'queued start cleanup regression',\n"
        "  phases: [{ title: 'Run', detail: 'Do work' }],\n"
        "  tools: ['coco'],\n"
        "  agentPlan: [{ node: 'work', role: 'worker', agentId: 'A-1' }],\n"
        "};\n"
        "export default async function main() {\n"
        "  const result = await agent({\n"
        "    prompt: 'do work', agentId: 'A-1', label: 'work', timeout: 120\n"
        "  });\n"
        "  if (result && result.error) return result;\n"
        "  return result;\n"
        "}\n"
    )
    script_path.write_text(script_text, encoding="utf-8")
    script_hash = hashlib.sha256(script_text.encode("utf-8")).hexdigest()

    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingWorkflow(
            script_path=str(script_path),
            requirement="do work",
            meta={"name": "preclaim-cleanup", "tools": ["coco"]},
            initiator_user_id="user_1",
            engine_session_key="session_1",
            project_id="project_1",
            selected_tools=["coco"],
            agent_pool=(
                WorkflowAgentBinding(
                    agent_id="A-1",
                    tool_name="coco",
                    model_name=None,
                    display_name="Coco",
                ),
            ),
            orchestrator_agent_id="A-1",
            orchestrator_agent="coco",
            orchestrator_binding=ReviewAgentBinding(
                provider="cli",
                tool_name="coco",
                display_name="Coco",
                agent_type="coco",
                model_name=None,
                model_display_name=None,
                selection_key="coco:default",
                use_default_model=True,
            ),
            script_hash=script_hash,
        ),
    )

    ctx = MagicMock()
    ctx.progress_reporter = MagicMock()

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = ctx
    project = SimpleNamespace(
        project_id="project_1",
        project_name="project",
        root_path=str(tmp_path),
    )
    handler._reply_workflow_error = MagicMock()
    handler._resolve_origin = MagicMock(return_value="origin_1")
    handler._show_initial_workflow_progress_card = MagicMock(return_value="progress_1")
    handler.get_engine_name = MagicMock(return_value="coco")
    handler._build_workflow_callbacks = MagicMock(return_value=WorkflowEngineCallbacks())

    queued: dict[str, object] = {}
    handler._submit_engine_task = MagicMock(
        side_effect=lambda run_fn, *_args, **_kwargs: queued.setdefault(
            "run",
            run_fn,
        )
    )

    generation_owner = _WorkflowLifecycleOwner(
        "session_1",
        "user_1",
        chat_id="chat_1",
        project_id="project_1",
        root_path=str(tmp_path),
    )
    engine._script_generation_owner = generation_owner
    handler._queue_generated_workflow(
        message_id="generation_1",
        chat_id="chat_1",
        project=project,
        root_path=str(tmp_path),
        engine=engine,
        generation_owner=generation_owner,
    )

    assert "run" in queued
    owner = engine._workflow_start_owner
    assert owner is not None

    closure_preclaim = threading.Event()
    release_closure = threading.Event()
    repo_lock_attempted = threading.Event()

    def ensure_request_id(*_args, **_kwargs) -> str:
        closure_preclaim.set()
        if not release_closure.wait(timeout=5):
            raise TimeoutError("test did not release queued Workflow closure")
        return "request_1"

    class _ObservedLockHelper:
        @staticmethod
        def _with_repo_lock(_root_path, _chat_id, body):
            repo_lock_attempted.set()
            return body()

    handler.ensure_request_id = ensure_request_id
    handler.lock_helper = _ObservedLockHelper()
    engine.execute_workflow = MagicMock()

    cleanup_finished = threading.Event()
    cleanup_outcome: dict[str, object] = {}

    def run_cleanup() -> None:
        try:
            cleanup_outcome["result"] = engine.cleanup()
        except BaseException as exc:  # pragma: no cover - surfaced below
            cleanup_outcome["error"] = exc
        finally:
            cleanup_finished.set()

    closure_thread = threading.Thread(
        target=queued["run"],
        name="workflow-queued-preclaim",
    )
    cleanup_thread = threading.Thread(
        target=run_cleanup,
        name="workflow-cleanup",
    )

    closure_thread.start()
    assert closure_preclaim.wait(timeout=1)

    cleanup_thread.start()
    assert owner.stop_event.wait(timeout=1)

    # Snapshot the buggy state while the closure is still deterministically
    # blocked. A correct cleanup is waiting for the closure's real finally.
    worker_was_registered = owner.worker_started_event.is_set()
    done_was_forged = owner.done_event.wait(timeout=0.2)
    cleanup_returned_early = cleanup_finished.wait(timeout=0.2)

    release_closure.set()
    closure_thread.join(timeout=2)
    cleanup_thread.join(timeout=2)

    assert not closure_thread.is_alive()
    assert not cleanup_thread.is_alive()
    assert {
        "worker_registered": worker_was_registered,
        "done_forged": done_was_forged,
        "cleanup_returned_early": cleanup_returned_early,
        "repo_lock_attempted": repo_lock_attempted.is_set(),
    } == {
        "worker_registered": True,
        "done_forged": False,
        "cleanup_returned_early": False,
        "repo_lock_attempted": False,
    }
    assert cleanup_outcome == {"result": True}
    assert owner.done_event.is_set()
    engine.execute_workflow.assert_not_called()


def test_repeated_manager_cleanup_retains_preclaim_tombstone_until_done(
    tmp_path,
) -> None:
    """A cleanup timeout must remain observable on every subsequent retry."""
    manager = WorkflowEngineManager()
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    owner = _WorkflowLifecycleOwner("preclaim_session")
    owner.worker_started_event.set()
    object.__setattr__(owner, "worker_thread_id", -1)
    engine._workflow_start_owner = owner

    key = f"chat_1:{tmp_path}"
    manager._engines[key] = engine
    manager._add_index("chat_1", key)

    def remove_with_expired_deadline() -> None:
        clock = itertools.chain([0.0], itertools.repeat(31.0))
        with (
            patch(
                "src.workflow_engine.engine.time.monotonic",
                side_effect=clock,
            ),
            patch("src.engine_base.get_gc_monitor"),
        ):
            manager.remove("chat_1", str(tmp_path))

    # First cleanup sees the started worker and times out, so the manager
    # correctly keeps a tombstone.
    remove_with_expired_deadline()
    assert manager.get("chat_1", str(tmp_path)) is engine
    assert not owner.done_event.is_set()

    # A retry must still see the same unresolved worker. Losing its owner
    # reference after the first timeout would make this cleanup report a
    # false quiescence and remove the engine.
    remove_with_expired_deadline()
    assert manager.get("chat_1", str(tmp_path)) is engine
    assert not owner.done_event.is_set()

    owner.done_event.set()
    remove_with_expired_deadline()
    assert manager.get("chat_1", str(tmp_path)) is None


def test_repeated_abandoned_starts_release_retired_owner_tombstones(tmp_path) -> None:
    script_path = tmp_path / "generated-workflow.js"
    script_text = (
        "export const meta = {\n"
        "  name: 'abandon-cleanup',\n"
        "  description: 'owner release regression',\n"
        "  phases: [{ title: 'Run', detail: 'Do work' }],\n"
        "  tools: ['coco'],\n"
        "  agentPlan: [{ node: 'work', role: 'worker', agentId: 'A-1' }],\n"
        "};\n"
        "export default async function main() {\n"
        "  const result = await agent({\n"
        "    prompt: 'work', agentId: 'A-1', label: 'work', timeout: 120\n"
        "  });\n"
        "  if (result && result.error) return result;\n"
        "  return result;\n"
        "}\n"
    )
    script_path.write_text(script_text, encoding="utf-8")
    script_hash = hashlib.sha256(script_text.encode("utf-8")).hexdigest()
    project = SimpleNamespace(
        project_id="project_1",
        project_name="project",
        root_path=str(tmp_path),
    )

    for failure_mode in ("scheduler_rejected", "worker_not_admitted"):
        engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
        handler = WorkflowHandler.__new__(WorkflowHandler)
        handler.ctx = MagicMock(progress_reporter=MagicMock())
        handler._reply_workflow_error = MagicMock()
        handler._resolve_origin = MagicMock(return_value="origin_1")
        handler._show_initial_workflow_progress_card = MagicMock(
            return_value="progress_1"
        )
        handler.get_engine_name = MagicMock(return_value="coco")
        handler._build_error_card = MagicMock(return_value={"error": True})
        handler._replace_or_send_workflow_card = MagicMock()
        handler._remove_owned_workflow_artifact = MagicMock()

        def submit(run_fn, *_args, **_kwargs):
            if failure_mode == "scheduler_rejected":
                raise RuntimeError("scheduler unavailable")
            engine._closing = True
            try:
                run_fn()
            finally:
                engine._closing = False

        handler._submit_engine_task = MagicMock(side_effect=submit)

        for index in range(24):
            session_key = f"generation-{failure_mode}-{index}"
            generation_owner = _WorkflowLifecycleOwner(
                session_key,
                "user_1",
                chat_id="chat_1",
                project_id="project_1",
                root_path=str(tmp_path),
            )
            engine._project = WorkflowProject(
                status=WorkflowStatus.GENERATING_SCRIPT,
                pending=PendingWorkflow(
                    script_path=str(script_path),
                    requirement="do work",
                    meta={"name": "abandon-cleanup", "tools": ["coco"]},
                    initiator_user_id="user_1",
                    engine_session_key=session_key,
                    project_id="project_1",
                    selected_tools=["coco"],
                    agent_pool=(
                        WorkflowAgentBinding(
                            agent_id="A-1",
                            tool_name="coco",
                            model_name=None,
                            display_name="Coco",
                        ),
                    ),
                    orchestrator_agent_id="A-1",
                    orchestrator_agent="coco",
                    orchestrator_binding=ReviewAgentBinding(
                        provider="cli",
                        tool_name="coco",
                        display_name="Coco",
                        agent_type="coco",
                        model_name=None,
                        model_display_name=None,
                        selection_key="coco:default",
                        use_default_model=True,
                    ),
                    script_hash=script_hash,
                ),
            )
            engine._script_generation_owner = generation_owner

            handler._queue_generated_workflow(
                message_id=f"generation_{index}",
                chat_id="chat_1",
                project=project,
                root_path=str(tmp_path),
                engine=engine,
                generation_owner=generation_owner,
            )

            assert handler._submit_engine_task.call_count == index + 1
            generation_owner.done_event.set()
            engine.release_lifecycle_owner(generation_owner)
            assert engine._workflow_start_owner is None
            assert engine._retired_lifecycle_owners == []
