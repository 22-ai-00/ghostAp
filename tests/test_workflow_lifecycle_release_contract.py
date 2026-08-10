from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.feishu.handlers.workflow import WorkflowHandler, _WorkflowLifecycleOwner
from src.workflow_engine.engine import WorkflowEngine
from src.workflow_engine.models import PendingWorkflow, WorkflowProject, WorkflowStatus


class _Session:
    def __init__(self, *, cancel_result=True, close_error=None) -> None:
        self.cancel_result = cancel_result
        self.close_error = close_error
        self.calls: list[str] = []

    def cancel(self, wait=True, timeout=2.0):
        del wait, timeout
        self.calls.append("cancel")
        return self.cancel_result

    def close(self):
        self.calls.append("close")
        if self.close_error:
            raise self.close_error


def _stop_context(tmp_path, session: _Session):
    engine = WorkflowEngine("chat-1", str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingWorkflow(
            requirement="build",
            initiator_user_id="user-1",
            engine_session_key="generation-1",
            project_id="project-1",
        ),
    )
    owner = _WorkflowLifecycleOwner(
        "generation-1",
        "user-1",
        chat_id="chat-1",
        project_id="project-1",
        root_path=str(tmp_path),
        active_generation_session=session,
    )
    engine._script_generation_owner = owner
    handler = WorkflowHandler.__new__(WorkflowHandler)
    manager = MagicMock()
    manager.get.return_value = engine
    handler.ctx = SimpleNamespace(
        workflow_engine_manager=manager,
        settings=SimpleNamespace(admin_user_ids=[]),
    )
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    handler._remove_owned_workflow_artifact = MagicMock()
    handler.reply_text = MagicMock()
    return handler, engine, owner


@pytest.mark.parametrize(
    "session",
    [
        _Session(cancel_result=False),
        _Session(close_error=RuntimeError("close uncertain")),
    ],
)
def test_stop_uncertainty_retains_fence_and_session_and_never_replies_success(tmp_path, session) -> None:
    handler, engine, owner = _stop_context(tmp_path, session)
    with patch("src.thread.get_current_sender_id", return_value="user-1"):
        handler.stop_workflow("stop-1", "chat-1", None)

    assert engine.project.status is WorkflowStatus.FAILED
    assert engine._script_generation_owner is owner
    assert owner.active_generation_session is session
    assert session.calls == ["cancel", "close"]
    handler.reply_text.assert_not_called()
    handler._reply_workflow_error.assert_called_once()

    ok, error, _new_owner = handler._supersede_incomplete_workflow(
        engine,
        root_path=str(tmp_path),
        current_user="user-1",
    )
    assert ok is False
    assert error == "invalid_state"


def test_confirmed_stop_releases_session_and_owner_resources(tmp_path) -> None:
    session = _Session()
    handler, engine, owner = _stop_context(tmp_path, session)
    with patch("src.thread.get_current_sender_id", return_value="user-1"):
        handler.stop_workflow("stop-1", "chat-1", None)

    assert engine.project.status is WorkflowStatus.CANCELLED
    assert owner.active_generation_session is None
    assert engine._script_generation_owner is None
    assert all(item is not owner for item in engine._retired_lifecycle_owners)
    handler.reply_text.assert_called_once()
    handler._reply_workflow_error.assert_not_called()


def test_released_lifecycle_owners_do_not_grow_across_many_workflows(tmp_path) -> None:
    engine = WorkflowEngine("chat-1", str(tmp_path))
    for index in range(32):
        owner = _WorkflowLifecycleOwner(f"session-{index}")
        engine.retire_lifecycle_owner(owner)
        owner.done_event.set()
        assert engine.release_lifecycle_owner(owner) is True
    assert engine._retired_lifecycle_owners == []


def test_generation_worker_finally_retains_uncertain_owner_and_session(tmp_path) -> None:
    session = _Session(cancel_result=False)
    handler, engine, owner = _stop_context(tmp_path, session)
    manager = MagicMock()
    manager.get_or_create.return_value = engine
    handler.ctx.workflow_engine_manager = manager
    handler.get_engine_name = MagicMock(return_value="codex")
    handler.send_card_to_chat = MagicMock(return_value="generation-card")
    handler.update_card = MagicMock()
    handler._build_error_card = MagicMock(return_value={"elements": []})
    handler._replace_or_send_workflow_card = MagicMock()
    handler._generate_script_via_ai = MagicMock(
        side_effect=RuntimeError("generation session close uncertain")
    )

    with patch("src.thread.get_current_sender_id", return_value="user-1"):
        handler._generate_and_start_workflow(
            message_id="origin-1",
            chat_id="chat-1",
            requirement="build",
            project=SimpleNamespace(project_id="project-1"),
            root_path=str(tmp_path),
            selected_tools=["codex"],
            expected_session_key="generation-1",
        )

    assert engine._script_generation_owner is owner
    assert owner.active_generation_session is session
    assert not owner.done_event.is_set()
    assert any(item is owner for item in engine._retired_lifecycle_owners)


def test_cleanup_retries_uncertain_session_before_done_and_release(tmp_path) -> None:
    session = _Session(cancel_result=False)
    _handler, engine, owner = _stop_context(tmp_path, session)

    assert engine.cleanup() is False
    assert session.calls == ["cancel", "close"]
    assert owner.active_generation_session is session
    assert not owner.done_event.is_set()
    assert any(item is owner for item in engine._retired_lifecycle_owners)

    session.calls.clear()
    session.cancel_result = True
    assert engine.cleanup() is True
    assert session.calls == ["cancel", "close"]
    assert owner.active_generation_session is None
    assert owner.done_event.is_set()
    assert all(item is not owner for item in engine._retired_lifecycle_owners)
