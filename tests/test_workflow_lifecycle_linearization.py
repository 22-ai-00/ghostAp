"""Deterministic regression tests for Workflow callback linearization."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.feishu.handlers.workflow import WorkflowHandler
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import (
    PendingConfirmation,
    WorkflowProject,
    WorkflowStatus,
)
from src.workflow_engine.selection_flow import SelectionFlowController


def _make_project(root_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        project_id="proj_1",
        project_name="ghostAp",
        root_path=root_path,
    )


def _make_handler(
    *,
    engine: WorkflowEngine,
    project: SimpleNamespace,
) -> WorkflowHandler:
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._resolve_project_from_id = MagicMock(return_value=project)
    handler._get_root_path = MagicMock(return_value=project.root_path)
    handler._reply_workflow_error = MagicMock()
    handler.update_card = MagicMock(return_value=True)
    handler._send_combined_selection_card = MagicMock()
    return handler


def test_review_finish_linearizes_with_concurrent_review_selection(tmp_path):
    """A finish callback must not schedule a snapshot older than the UI state."""
    project = _make_project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            requirement="review this change",
            initiator_user_id="user_1",
            engine_session_key="session_1",
            orchestrator_agent="coco",
            selected_tools=["coco"],
        ),
    )

    controller = SelectionFlowController(step=2)
    controller.add_or_update_selection(
        {
            "tool_name": "coco",
            "display_name": "Coco",
            "use_default_model": True,
        },
        is_review=True,
    )
    project._wf_selection_session_key = "session_1"
    project._wf_selection_snapshot = controller.snapshot()
    project._wf_selection_lock = threading.RLock()

    handler = _make_handler(engine=engine, project=project)
    handler._validate_tools_against_registry = MagicMock(
        return_value=(["claude"], []),
    )
    handler._schedule_generate_and_show_confirm_card = MagicMock()

    finish_snapshot_ready = threading.Event()
    release_finish = threading.Event()
    original_transaction = handler._selection_controller_transaction

    @contextmanager
    def transaction_with_finish_barrier(
        project_arg,
        *,
        session_key,
        **transaction_kwargs,
    ):
        with original_transaction(
            project_arg,
            session_key=session_key,
            **transaction_kwargs,
        ) as active_controller:
            yield active_controller
        if threading.current_thread().name == "workflow-review-finish":
            finish_snapshot_ready.set()
            assert release_finish.wait(timeout=2)

    handler._selection_controller_transaction = transaction_with_finish_barrier
    thread_errors: list[BaseException] = []

    def finish_review() -> None:
        try:
            handler.handle_workflow_review_finish(
                "card_1",
                "chat_1",
                "proj_1",
                {
                    "action": "workflow_review_finish",
                    "project_id": "proj_1",
                    "engine_session_key": "session_1",
                },
            )
        except BaseException as exc:  # pragma: no cover - assertion plumbing
            thread_errors.append(exc)

    def select_review_model() -> None:
        try:
            handler.handle_workflow_review_select_model(
                "card_1",
                "chat_1",
                "proj_1",
                {
                    "action": "workflow_review_select_model",
                    "project_id": "proj_1",
                    "engine_session_key": "session_1",
                    "tool_name": "claude",
                    "display_name": "Claude",
                    "use_default_model": True,
                },
            )
        except BaseException as exc:  # pragma: no cover - assertion plumbing
            thread_errors.append(exc)

    finish_thread = threading.Thread(
        name="workflow-review-finish",
        target=finish_review,
    )
    selection_thread = threading.Thread(
        name="workflow-review-select",
        target=select_review_model,
    )

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        finish_thread.start()
        assert finish_snapshot_ready.wait(timeout=1)
        selection_thread.start()
        # A correct implementation may serialize this callback behind the
        # finish transition. Releasing the barrier keeps either ordering
        # deterministic without making the test depend on one strategy.
        selection_thread.join(timeout=0.1)
        release_finish.set()
        finish_thread.join(timeout=2)
        selection_thread.join(timeout=2)

    assert not finish_thread.is_alive()
    assert not selection_thread.is_alive()
    assert not thread_errors
    handler._schedule_generate_and_show_confirm_card.assert_called_once()

    restored = SelectionFlowController()
    restored.restore(project._wf_selection_snapshot)
    controller_tools = {item.get("tool_name") for item in restored.snapshot()["review_selections"].values()}
    pending_tools = {agent.tool_name for agent in (engine.project.pending.review_agents or [])}
    scheduled_tools = set(handler._schedule_generate_and_show_confirm_card.call_args.kwargs["selected_tools"] or [])

    assert pending_tools == controller_tools
    assert scheduled_tools == controller_tools | {"coco"}


def test_legacy_confirm_tools_snapshots_selection_inside_transition(tmp_path):
    """Confirm must schedule the tool set current at its status transition."""
    project = _make_project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            requirement="run with every selected tool",
            initiator_user_id="user_1",
            engine_session_key="session_1",
            selected_tools=["coco"],
        ),
    )

    controller = SelectionFlowController(step=2)
    controller.add_or_update_selection(
        {
            "tool_name": "coco",
            "display_name": "Coco",
            "use_default_model": True,
        },
        is_review=True,
    )
    project._wf_selection_session_key = "session_1"
    project._wf_selection_snapshot = controller.snapshot()
    project._wf_selection_lock = threading.RLock()

    handler = _make_handler(engine=engine, project=project)
    handler._validate_tools_against_registry = MagicMock(
        return_value=(["claude"], []),
    )
    handler._schedule_generate_and_show_confirm_card = MagicMock()

    confirm_snapshot_started = threading.Event()
    release_confirm = threading.Event()

    class SnapshotBarrierList(list):
        def __iter__(self):
            if threading.current_thread().name == "workflow-confirm-tools":
                confirm_snapshot_started.set()
                assert release_confirm.wait(timeout=2)
            return super().__iter__()

    engine.project.pending.selected_tools = SnapshotBarrierList(["coco"])
    thread_errors: list[BaseException] = []

    def confirm_tools() -> None:
        try:
            handler.handle_workflow_confirm_tools(
                "card_1",
                "chat_1",
                "proj_1",
                {
                    "action": "workflow_confirm_tools",
                    "project_id": "proj_1",
                    "engine_session_key": "session_1",
                },
            )
        except BaseException as exc:  # pragma: no cover - assertion plumbing
            thread_errors.append(exc)

    def select_tool() -> None:
        try:
            handler.handle_workflow_select_tool(
                "card_1",
                "chat_1",
                "proj_1",
                {
                    "action": "workflow_select_tool",
                    "project_id": "proj_1",
                    "engine_session_key": "session_1",
                    "tool_name": "claude",
                    "display_name": "Claude",
                    "use_default_model": True,
                },
            )
        except BaseException as exc:  # pragma: no cover - assertion plumbing
            thread_errors.append(exc)

    confirm_thread = threading.Thread(
        name="workflow-confirm-tools",
        target=confirm_tools,
    )
    selection_thread = threading.Thread(
        name="workflow-select-tool",
        target=select_tool,
    )

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        confirm_thread.start()
        assert confirm_snapshot_started.wait(timeout=1)
        selection_thread.start()
        # If confirm snapshots outside the transition lock, selection wins
        # here. A linearized implementation may instead keep it blocked until
        # confirm commits; both orderings are valid as long as state/schedule
        # agree.
        selection_thread.join(timeout=0.1)
        release_confirm.set()
        confirm_thread.join(timeout=2)
        selection_thread.join(timeout=2)

    assert not confirm_thread.is_alive()
    assert not selection_thread.is_alive()
    assert not thread_errors
    handler._schedule_generate_and_show_confirm_card.assert_called_once()
    assert (
        handler._schedule_generate_and_show_confirm_card.call_args.kwargs["selected_tools"]
        == engine.project.pending.selected_tools
    )


def test_stale_selection_render_cannot_overwrite_newer_same_session_card(
    tmp_path,
):
    """A slow S1 render must not be delivered after a newer S2 card."""
    project = _make_project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            requirement="keep the newest selection visible",
            initiator_user_id="user_1",
            engine_session_key="session_1",
            selected_tools=["coco"],
        ),
    )

    controller = SelectionFlowController(step=2)
    controller.add_or_update_selection(
        {
            "tool_name": "coco",
            "display_name": "Coco",
            "use_default_model": True,
        },
        is_review=True,
    )
    project._wf_selection_session_key = "session_1"
    project._wf_selection_snapshot = controller.snapshot()
    project._wf_selection_lock = threading.RLock()

    handler = _make_handler(engine=engine, project=project)
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._send_combined_selection_card = WorkflowHandler._send_combined_selection_card.__get__(
        handler,
        WorkflowHandler,
    )
    first_render_started = threading.Event()
    release_first_render = threading.Event()
    render_count_lock = threading.Lock()
    render_count = 0

    def build_available_tools():
        nonlocal render_count
        with render_count_lock:
            render_count += 1
            call_number = render_count
        if call_number == 1:
            first_render_started.set()
            assert release_first_render.wait(timeout=2)
        return [
            {
                "tool_name": "coco",
                "display_name": "Coco",
                "supports_model": True,
            }
        ]

    handler._build_available_tools = MagicMock(
        side_effect=build_available_tools,
    )
    thread_errors: list[BaseException] = []

    def render_selection() -> None:
        try:
            handler._send_combined_selection_card(
                message_id="card_1",
                chat_id="chat_1",
                project=project,
                requirement="keep the newest selection visible",
                session_key="session_1",
                is_review=True,
            )
        except BaseException as exc:  # pragma: no cover - assertion plumbing
            thread_errors.append(exc)

    stale_render = threading.Thread(
        name="workflow-selection-render-s1",
        target=render_selection,
    )
    stale_render.start()
    assert first_render_started.wait(timeout=1)

    with handler._selection_controller_transaction(
        project,
        session_key="session_1",
        engine=engine,
        expected_status=WorkflowStatus.AWAITING_TOOL_SELECT,
        initiator_user_id="user_1",
    ) as active_controller:
        assert active_controller is not None
        active_controller.add_or_update_selection(
            {
                "tool_name": "claude",
                "display_name": "SECOND_SELECTION_MARKER",
                "use_default_model": True,
            },
            is_review=True,
        )

    newest_render = threading.Thread(
        name="workflow-selection-render-s2",
        target=render_selection,
    )
    newest_render.start()
    newest_render.join(timeout=2)
    assert not newest_render.is_alive()
    assert not thread_errors
    handler.update_card.assert_called_once()
    newest_card = handler.update_card.call_args.args[1]
    assert "SECOND_SELECTION_MARKER" in str(newest_card)

    release_first_render.set()
    stale_render.join(timeout=2)

    assert not stale_render.is_alive()
    assert not thread_errors
    assert handler.update_card.call_args.args[1] == newest_card
