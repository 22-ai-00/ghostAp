"""Regression tests for Workflow state consistency across runs."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from src.engine_base import EngineRunState
from src.feishu.handlers.workflow import (
    WorkflowHandler,
    _WorkflowGenerationCancelled,
    _WorkflowLifecycleOwner,
)
from src.project.mapper import MessageLinker
from src.workflow_engine.engine import WorkflowEngine, WorkflowEngineCallbacks
from src.workflow_engine.models import PendingConfirmation, WorkflowProject, WorkflowStatus
from src.workflow_engine.selection_flow import SelectionFlowController


class _Project:
    def __init__(self, root_path: str) -> None:
        self.project_id = "proj_1"
        self.project_name = "ghostAp"
        self.root_path = root_path


def _walk_card_text(node):
    if isinstance(node, dict):
        text = node.get("text")
        if isinstance(text, dict):
            yield str(text.get("content", ""))
        if "content" in node:
            yield str(node.get("content", ""))
        for value in node.values():
            yield from _walk_card_text(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_card_text(item)


def test_workflow_engine_clears_stale_cancel_event_before_new_run(tmp_path):
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "smoke", description: "", phases: [], tools: [] };
export default async function workflow() { return "ok"; }
""",
        encoding="utf-8",
    )

    class FakeBridge:
        def __init__(self, *args, cancel_event, **kwargs):
            self.cancel_event = cancel_event

        @staticmethod
        def check_node_available():
            return True

        def start(self):
            return None

        def run(self):
            if self.cancel_event.is_set():
                raise RuntimeError("Workflow cancelled")
            return "ok"

        def stop(self):
            return None

    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine.cancel_event.set()

    with patch("src.workflow_engine.engine.RuntimeBridge", FakeBridge):
        project = engine.execute_workflow("run cleanly", str(script_path))

    assert project.status == WorkflowStatus.COMPLETED
    assert not engine.cancel_event.is_set()


def test_workflow_engine_preserves_cancelled_terminal_status(tmp_path):
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "cancel", description: "", phases: [], tools: [] };
export default async function workflow() { return "never"; }
""",
        encoding="utf-8",
    )

    class FakeBridge:
        def __init__(self, *args, cancel_event, **kwargs):
            self.cancel_event = cancel_event

        @staticmethod
        def check_node_available():
            return True

        def start(self):
            self.cancel_event.set()

        def run(self):
            raise RuntimeError("Workflow cancelled")

        def stop(self):
            return None

    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))

    with patch("src.workflow_engine.engine.RuntimeBridge", FakeBridge):
        project = engine.execute_workflow("cancel cleanly", str(script_path))

    assert project.status == WorkflowStatus.CANCELLED
    assert project.error == "Workflow cancelled"


def test_workflow_engine_rejects_cancelled_queued_start_atomically(tmp_path):
    """A queued start cancelled after its wrapper check must not claim the engine."""
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "guarded", description: "", phases: [], tools: [] };
export default async function workflow() { return "ok"; }
""",
        encoding="utf-8",
    )
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    original_project = engine.project
    original_callbacks = object()
    engine._callbacks = original_callbacks
    start_owner = _WorkflowLifecycleOwner("queued_session")
    engine._workflow_start_owner = start_owner
    start_owner.stop_event.set()

    with patch(
        "src.workflow_engine.engine.RuntimeBridge.check_node_available",
        side_effect=AssertionError("cancelled queued start reached runtime setup"),
    ):
        result = engine.execute_workflow(
            "must not run",
            str(script_path),
            start_owner=start_owner,
        )

    assert result is original_project
    assert engine.project is original_project
    assert engine._callbacks is original_callbacks
    assert engine.is_running is False
    assert engine._workflow_start_owner is None


def test_fresh_workflow_agent_selection_does_not_restore_previous_selection(tmp_path):
    project = _Project(str(tmp_path))

    stale = SelectionFlowController(step=2)
    stale.add_or_update_selection(
        {
            "tool_name": "coco",
            "display_name": "Coco",
            "model_name": "old-model",
            "supports_model": True,
        },
        is_review=False,
    )
    project._wf_selection_snapshot = stale.snapshot()

    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler._resolve_tool_lists = MagicMock(
        return_value=({"coco": "Coco tool"}, ["coco"], [], ["coco"])
    )
    handler._get_workflow_models_for_tool = MagicMock(return_value=[])
    handler.send_card_to_chat = MagicMock(return_value="card_msg_1")

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler._show_agent_selection_card(
            message_id="msg_1",
            chat_id="chat_1",
            requirement="build something new",
            project=project,
            root_path=str(tmp_path),
        )

    sent_card = handler.send_card_to_chat.call_args.args[1]
    rendered_text = "\n".join(_walk_card_text(sent_card))

    assert "old-model" not in rendered_text
    assert "**已选**" not in rendered_text
    assert project._wf_selection_snapshot["orchestrator_selections"] == {}
    assert project._wf_selection_snapshot["review_selections"] == {}


def test_first_workflow_selection_card_links_to_original_inbound_message(tmp_path):
    """The first outbound card must retain the trusted /wf origin in MessageLinker."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    linker = MessageLinker()
    linker.register_origin(
        "inbound_wf",
        request_id="request_1",
        chat_id="chat_1",
        chat_type="group",
        sender_id="user_1",
    )
    response = MagicMock()
    response.success.return_value = True
    response.data.message_id = "selection_card"

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.managed_group_registry = None
    handler.ctx.message_linker = linker
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.im_client = MagicMock()
    handler.im_client.send_message.return_value = response
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler._resolve_tool_lists = MagicMock(
        return_value=({"coco": "Coco tool"}, ["coco"], [], ["coco"])
    )
    handler._get_workflow_models_for_tool = MagicMock(return_value=[])
    handler._preheat_workflow_models = MagicMock()
    handler.format_ref_note = MagicMock(return_value="")

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler._show_agent_selection_card(
            message_id="inbound_wf",
            chat_id="chat_1",
            requirement="build something new",
            project=project,
            root_path=str(tmp_path),
        )

    assert linker.resolve_origin(reply_message_id="selection_card") == "inbound_wf"
    assert handler._resolve_origin("selection_card") == "inbound_wf"
    assert handler._reply_audit_aliases("inbound_wf") == ("chat_1",)

    patch_failure = MagicMock()
    patch_failure.success.return_value = False
    fallback_response = MagicMock()
    fallback_response.success.return_value = True
    fallback_response.data.message_id = "fallback_card"
    handler.im_client.patch_message.return_value = patch_failure
    handler.im_client.send_message.return_value = fallback_response

    fallback_id = handler._replace_or_send_workflow_rendered_card(
        card_message_id="selection_card",
        chat_id="chat_1",
        card_data={
            "header": {
                "title": {"tag": "plain_text", "content": "Workflow"},
                "template": "blue",
            },
            "elements": [],
        },
        origin_message_id=handler._resolve_origin("selection_card"),
    )

    assert fallback_id == "fallback_card"
    assert linker.resolve_origin(reply_message_id="fallback_card") == "inbound_wf"

    patch_success = MagicMock()
    patch_success.success.return_value = True
    handler.im_client.patch_message.return_value = patch_success
    assert handler.update_card(
        "fallback_card",
        {"schema": "2.0", "body": {"elements": []}},
    )
    assert handler.im_client.patch_message.call_args.kwargs["audit_aliases"] == (
        "chat_1",
    )


def test_concurrent_same_user_starts_publish_only_the_newest_selection_card(tmp_path):
    """An older card build that finishes late must fail its final owner CAS."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    first_resolver_entered = threading.Event()
    release_first_resolver = threading.Event()
    resolver_lock = threading.Lock()
    resolver_calls = 0

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler._get_workflow_models_for_tool = MagicMock(return_value=[])
    handler._preheat_workflow_models = MagicMock()
    handler._resolve_origin = MagicMock(side_effect=lambda message_id: message_id)
    handler.send_card_to_chat = MagicMock(return_value="selection_card")

    def resolve_tools():
        nonlocal resolver_calls
        with resolver_lock:
            resolver_calls += 1
            call_number = resolver_calls
        if call_number == 1:
            first_resolver_entered.set()
            assert release_first_resolver.wait(timeout=2)
        return {"coco": "Coco"}, ["coco"], [], ["coco"]

    handler._resolve_tool_lists = MagicMock(side_effect=resolve_tools)

    old_thread = threading.Thread(
        target=handler._show_agent_selection_card,
        kwargs={
            "message_id": "old_msg",
            "chat_id": "chat_1",
            "requirement": "old requirement",
            "project": project,
            "root_path": str(tmp_path),
        },
    )
    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        old_thread.start()
        assert first_resolver_entered.wait(timeout=1)
        handler._show_agent_selection_card(
            message_id="new_msg",
            chat_id="chat_1",
            requirement="new requirement",
            project=project,
            root_path=str(tmp_path),
        )
        release_first_resolver.set()
        old_thread.join(timeout=2)

    assert not old_thread.is_alive()
    handler.send_card_to_chat.assert_called_once()
    sent_card = handler.send_card_to_chat.call_args.args[1]
    assert "new requirement" in "\n".join(_walk_card_text(sent_card))
    assert engine.project.pending.requirement == "new requirement"


def test_old_script_generation_task_does_not_apply_after_session_changes(tmp_path):
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="old task",
            engine_session_key="old_session",
            initiator_user_id="user_1",
        ),
    )
    project = _Project(str(tmp_path))

    captured = {}

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._submit_engine_task = MagicMock(
        side_effect=lambda fn, *args, **kwargs: captured.setdefault("task", fn)
    )
    handler._generate_and_show_confirm_card = MagicMock()
    handler._reply_workflow_error = MagicMock()

    handler._schedule_generate_and_show_confirm_card(
        message_id="old_card",
        chat_id="chat_1",
        requirement="old task",
        project=project,
        root_path=str(tmp_path),
        selected_tools=["coco"],
        engine=engine,
        expected_session_key="old_session",
    )

    engine.project.status = WorkflowStatus.AWAITING_AGENT_SELECT
    engine.project.pending = PendingConfirmation(
        requirement="new task",
        engine_session_key="new_session",
        initiator_user_id="user_1",
    )

    captured["task"]()

    handler._generate_and_show_confirm_card.assert_not_called()
    assert engine.project.status == WorkflowStatus.AWAITING_AGENT_SELECT
    assert engine.project.pending.requirement == "new task"


def test_stale_generation_schedule_cannot_adopt_the_new_session(tmp_path):
    """A delayed old callback must not re-derive and capture the newest key."""
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="new task",
            engine_session_key="new_session",
            initiator_user_id="user_1",
        ),
    )
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._submit_engine_task = MagicMock()

    handler._schedule_generate_and_show_confirm_card(
        message_id="old_card",
        chat_id="chat_1",
        requirement="old task",
        project=_Project(str(tmp_path)),
        root_path=str(tmp_path),
        selected_tools=["coco"],
        engine=engine,
        expected_session_key="old_session",
    )

    handler._submit_engine_task.assert_not_called()
    assert engine.project.pending.requirement == "new task"
    assert engine.project.pending.engine_session_key == "new_session"


def test_cancel_drains_generation_owner_even_after_pending_key_rotates(tmp_path):
    """Generation CAS rotates the confirm key; cancel still owns the live token."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_CONFIRM,
        pending=PendingConfirmation(
            requirement="task",
            engine_session_key="confirm_session",
            initiator_user_id="user_1",
        ),
    )
    owner = _WorkflowLifecycleOwner("selection_session")
    engine._script_generation_owner = owner

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._resolve_project_from_id = MagicMock(return_value=project)
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    handler.update_card = MagicMock(return_value=True)

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_cancel(
            "confirm_card",
            "chat_1",
            "proj_1",
            {"engine_session_key": "confirm_session"},
        )

    assert owner.stop_event.is_set()
    assert engine._script_generation_owner is None
    assert engine.project.status == WorkflowStatus.IDLE
    handler._reply_workflow_error.assert_not_called()
    handler.update_card.assert_called_once()


def test_stale_review_finish_cannot_mutate_or_schedule_new_session(tmp_path):
    """A callback paused after validation must lose its final session CAS."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            requirement="old task",
            engine_session_key="old_session",
            initiator_user_id="user_1",
            orchestrator_agent="coco",
        ),
    )
    ctrl = SelectionFlowController(step=2)
    ctrl.add_or_update_selection(
        {
            "tool_name": "coco",
            "display_name": "Coco",
            "use_default_model": True,
        },
        is_review=True,
    )

    def replace_with_new_session(*, is_review):
        assert is_review is True
        engine._project = WorkflowProject(
            status=WorkflowStatus.AWAITING_TOOL_SELECT,
            pending=PendingConfirmation(
                requirement="new task",
                engine_session_key="new_session",
                initiator_user_id="user_1",
                orchestrator_agent="claude",
            ),
        )
        return True, ""

    ctrl.validate_non_empty = MagicMock(side_effect=replace_with_new_session)
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._resolve_project_from_id = MagicMock(return_value=project)
    handler._get_selection_controller = MagicMock(return_value=ctrl)
    handler._schedule_generate_and_show_confirm_card = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler.update_card = MagicMock()

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_review_finish(
            "old_card",
            "chat_1",
            "proj_1",
            {
                "project_id": "proj_1",
                "engine_session_key": "old_session",
            },
        )

    assert engine.project.status == WorkflowStatus.AWAITING_TOOL_SELECT
    assert engine.project.pending.requirement == "new task"
    assert engine.project.pending.orchestrator_agent == "claude"
    assert engine.project.pending.review_agents is None
    handler.update_card.assert_not_called()
    handler._schedule_generate_and_show_confirm_card.assert_not_called()


def test_script_generation_result_is_ignored_if_session_changes_mid_generation(tmp_path):
    script_path = tmp_path / "generated.js"
    script_path.write_text(
        """
export const meta = { name: "generated", description: "", phases: [], tools: ["coco"] };
export default async function workflow() { return await agent("do it", { tool: "coco" }); }
""",
        encoding="utf-8",
    )

    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="old task",
            engine_session_key="old_session",
            initiator_user_id="user_1",
            selected_tools=["coco"],
        ),
    )
    project = _Project(str(tmp_path))

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler.send_card_to_chat = MagicMock(return_value="generating_card")
    handler.update_card = MagicMock()
    handler._replace_or_send_workflow_card = MagicMock()

    def generation_finishes_after_cancel(*args, **kwargs):
        engine.project.status = WorkflowStatus.IDLE
        engine.project.pending = None
        return str(script_path), {"name": "generated", "tools": ["coco"]}, False

    handler._generate_script_via_ai = MagicMock(side_effect=generation_finishes_after_cancel)

    with (
        patch("src.thread.get_current_sender_id", return_value="user_1"),
        patch("src.workflow_engine.templates.discover_templates", return_value=[]),
    ):
        handler._generate_and_show_confirm_card(
            message_id="old_card",
            chat_id="chat_1",
            requirement="old task",
            project=project,
            root_path=str(tmp_path),
            selected_tools=["coco"],
            expected_session_key="old_session",
        )

    assert engine.project.status == WorkflowStatus.IDLE
    assert engine.project.pending is None
    handler._replace_or_send_workflow_card.assert_not_called()


def test_stale_generation_task_does_not_cancel_current_generation_heartbeat(tmp_path):
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="new task",
            engine_session_key="new_session",
            initiator_user_id="user_1",
        ),
    )
    current_generation_owner = _WorkflowLifecycleOwner("new_session")
    engine._script_generation_owner = current_generation_owner

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler.send_card_to_chat = MagicMock(return_value="stale_card")
    handler.update_card = MagicMock()

    with patch("threading.Thread") as thread_cls:
        handler._generate_and_show_confirm_card(
            message_id="old_card",
            chat_id="chat_1",
            requirement="old task",
            project=_Project(str(tmp_path)),
            root_path=str(tmp_path),
            selected_tools=["coco"],
            expected_session_key="old_session",
        )

    assert current_generation_owner.stop_event.is_set() is False
    assert engine._script_generation_owner is current_generation_owner
    handler.send_card_to_chat.assert_not_called()
    thread_cls.assert_not_called()


def test_generation_result_cas_cannot_commit_after_stop_acknowledgement(tmp_path):
    """The engine lock must order /stop_wf before a blocked generation commit."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="old task",
            engine_session_key="old_session",
            initiator_user_id="user_1",
        ),
    )
    owner = _WorkflowLifecycleOwner("old_session")
    engine._script_generation_owner = owner
    prepared_pending = PendingConfirmation(
        requirement="generated task",
        engine_session_key="next_session",
        initiator_user_id="user_1",
    )
    commit_attempted = threading.Event()
    result = {}

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler.reply_text = MagicMock()
    handler._reply_workflow_error = MagicMock()

    def commit_result():
        commit_attempted.set()
        result["value"] = handler._commit_generation_result_if_current(
            engine,
            owner,
            prepared_pending,
        )

    commit_thread = threading.Thread(target=commit_result)
    with engine._lock:
        commit_thread.start()
        assert commit_attempted.wait(timeout=1)
        with patch("src.thread.get_current_sender_id", return_value="user_1"):
            handler.stop_workflow("stop_msg", "chat_1", project)

    commit_thread.join(timeout=2)

    assert not commit_thread.is_alive()
    assert result["value"] is False
    assert engine.project.status == WorkflowStatus.IDLE
    assert engine.project.pending is None
    handler.reply_text.assert_called_once_with("stop_msg", "Workflow 任务已停止。")


def test_new_workflow_supersedes_owned_incomplete_generation(tmp_path):
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    old_script = tmp_path / ".ghostap" / "workflow_scripts" / "generated-workflow-old.js"
    old_script.parent.mkdir(parents=True)
    old_script.write_text("old", encoding="utf-8")
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="old task",
            script_path=str(old_script),
            engine_session_key="old_session",
            initiator_user_id="user_1",
        ),
    )
    old_owner = _WorkflowLifecycleOwner("old_session")
    engine._script_generation_owner = old_owner

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler.add_reaction = MagicMock()
    handler._ensure_topic_engine_context = MagicMock()
    handler._show_agent_selection_card = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler._ensure_project = MagicMock(return_value=project)

    with (
        patch("src.thread.get_current_sender_id", return_value="user_1"),
        patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True),
        patch("src.workflow_engine.templates.discover_templates", return_value=[]),
    ):
        handler.start_workflow("msg_1", "chat_1", "new task", project)

    assert old_owner.stop_event.is_set()
    assert engine._script_generation_owner is None
    assert not old_script.exists()
    handler._reply_workflow_error.assert_not_called()
    handler._show_agent_selection_card.assert_not_called()
    assert engine.project.status == WorkflowStatus.GENERATING_SCRIPT
    assert engine.project.pending is not None
    assert engine.project.pending.requirement == "new task"
    assert engine.project.pending.auto_reviewer is True
    assert engine.project.pending.orchestrator_agent


def test_new_workflow_cancels_old_queued_start_before_it_can_claim_engine(tmp_path):
    """Newest-wins must include the RUNNING-before-scheduler-claim window."""
    project = _Project(str(tmp_path))
    old_script = tmp_path / "ghostap-confirmed-old.js"
    old_script.write_text(
        """
export const meta = { name: "old", description: "", phases: [], tools: [] };
export default async function workflow() { return "old"; }
""",
        encoding="utf-8",
    )
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.RUNNING,
        requirement="old task",
        script_path=str(old_script),
        initiator_user_id="user_1",
    )
    old_owner = _WorkflowLifecycleOwner("old_session")
    engine._workflow_start_owner = old_owner

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler.add_reaction = MagicMock()
    handler._ensure_topic_engine_context = MagicMock()
    handler._show_agent_selection_card = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler._ensure_project = MagicMock(return_value=project)

    with (
        patch("src.thread.get_current_sender_id", return_value="user_1"),
        patch("src.workflow_engine.bridge.RuntimeBridge.check_node_available", return_value=True),
        patch("src.workflow_engine.templates.discover_templates", return_value=[]),
    ):
        handler.start_workflow("new_msg", "chat_1", "new task", project)

    assert old_owner.stop_event.is_set()
    assert engine._workflow_start_owner is None
    handler._show_agent_selection_card.assert_not_called()
    assert engine.project.status == WorkflowStatus.GENERATING_SCRIPT
    assert engine.project.pending is not None
    assert engine.project.pending.requirement == "new task"
    assert engine.project.pending.auto_reviewer is True

    with patch(
        "src.workflow_engine.engine.RuntimeBridge.check_node_available",
        side_effect=AssertionError("superseded queued workflow reached runtime"),
    ):
        result = engine.execute_workflow(
            "old task",
            str(old_script),
            start_owner=old_owner,
        )

    assert result.status != WorkflowStatus.RUNNING
    assert engine.project.status != WorkflowStatus.RUNNING


def test_new_workflow_cannot_supersede_another_users_pending_session(tmp_path):
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="owned task",
            engine_session_key="owned_session",
            initiator_user_id="user_1",
        ),
    )
    owner = _WorkflowLifecycleOwner("owned_session")
    engine._script_generation_owner = owner

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler._ensure_project = MagicMock(return_value=project)
    handler._reply_workflow_error = MagicMock()
    handler._show_agent_selection_card = MagicMock()

    with patch("src.thread.get_current_sender_id", return_value="user_2"):
        handler.start_workflow("msg_2", "chat_1", "new task", project)

    assert not owner.stop_event.is_set()
    assert engine._script_generation_owner is owner
    handler._reply_workflow_error.assert_called_once()
    handler._show_agent_selection_card.assert_not_called()


def test_runtime_stop_fences_all_late_progress_and_terminal_delivery(tmp_path):
    """No runtime callback may publish after the stop acknowledgement."""
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "stop", description: "", phases: [], tools: [] };
export default async function workflow() { return "never"; }
""",
        encoding="utf-8",
    )
    bridge_started = threading.Event()

    class FakeBridge:
        def __init__(self, *args, cancel_event, **kwargs):
            self.cancel_event = cancel_event

        @staticmethod
        def check_node_available():
            return True

        def start(self):
            bridge_started.set()

        def run(self):
            if not self.cancel_event.wait(timeout=2):
                raise AssertionError("workflow was not stopped")
            raise RuntimeError("Workflow cancelled")

        def stop(self):
            return None

    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    owner = _WorkflowLifecycleOwner("runtime_session")
    engine._workflow_start_owner = owner
    engine.save_state = MagicMock(return_value="")

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._resolve_origin = MagicMock(return_value="origin_msg")
    handler.update_card = MagicMock(return_value=True)
    handler.send_card_to_chat = MagicMock(return_value="new_card")
    handler._reply_workflow_error = MagicMock()
    handler.reply_text = MagicMock()
    callbacks = handler._build_workflow_callbacks(
        "progress_card",
        "chat_1",
        project,
        lifecycle_owner=owner,
    )

    execution_thread = threading.Thread(
        target=engine.execute_workflow,
        kwargs={
            "requirement": "stop me",
            "script_path": str(script_path),
            "callbacks": callbacks,
            "initiator_user_id": "user_1",
            "start_owner": owner,
        },
    )
    with (
        patch("src.workflow_engine.engine.RuntimeBridge", FakeBridge),
        patch("src.workflow_engine.engine.WorkflowHistory.record"),
    ):
        execution_thread.start()
        assert bridge_started.wait(timeout=1)
        handler.update_card.reset_mock()

        with patch("src.thread.get_current_sender_id", return_value="user_1"):
            handler.stop_workflow("stop_msg", "chat_1", project)

        updates_at_ack = handler.update_card.call_count
        errors_at_ack = handler._reply_workflow_error.call_count
        execution_thread.join(timeout=2)

    assert not execution_thread.is_alive()
    assert owner.stop_event.is_set()
    assert handler.update_card.call_count == updates_at_ack
    assert handler._reply_workflow_error.call_count == errors_at_ack == 0
    handler.reply_text.assert_called_once_with("stop_msg", "Workflow 任务已停止。")


def test_stop_workflow_clears_generating_script_state(tmp_path):
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="old task",
            engine_session_key="old_session",
            initiator_user_id="user_1",
        ),
    )

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler.reply_text = MagicMock()
    handler._reply_workflow_error = MagicMock()
    generation_owner = _WorkflowLifecycleOwner("old_session")
    engine._script_generation_owner = generation_owner

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.stop_workflow("msg_1", "chat_1", project)

    assert generation_owner.stop_event.is_set()
    assert engine.project.status == WorkflowStatus.IDLE
    assert engine.project.pending is None
    handler.reply_text.assert_called_once()
    handler._reply_workflow_error.assert_not_called()


def test_pending_workflow_stop_authorizes_the_pending_initiator_not_prior_run(tmp_path):
    """A reused engine must not apply the previous run's owner to a new pending task."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        initiator_user_id="previous_user",
        pending=PendingConfirmation(
            requirement="new task",
            engine_session_key="new_session",
            initiator_user_id="current_user",
        ),
    )
    owner = _WorkflowLifecycleOwner("new_session")
    engine._script_generation_owner = owner

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler.reply_text = MagicMock()
    handler._reply_workflow_error = MagicMock()

    with patch("src.thread.get_current_sender_id", return_value="current_user"):
        handler.stop_workflow("msg_1", "chat_1", project)

    assert owner.stop_event.is_set()
    handler.reply_text.assert_called_once()
    handler._reply_workflow_error.assert_not_called()


def test_stopped_generation_rejects_late_direct_progress_callback(tmp_path):
    """A model callback returning after /stop_wf must not PATCH the loading card."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.GENERATING_SCRIPT,
        pending=PendingConfirmation(
            requirement="old task",
            engine_session_key="old_session",
            initiator_user_id="user_1",
            selected_tools=["coco"],
        ),
    )
    callback_ready = threading.Event()
    callback_invoked = threading.Event()
    release_generation = threading.Event()

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.ctx.settings.admin_user_ids = []
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler._resolve_origin = MagicMock(return_value="origin_msg")
    handler.send_card_to_chat = MagicMock(return_value="loading_card")
    handler.update_card = MagicMock(return_value=True)
    handler.reply_text = MagicMock()
    handler._reply_workflow_error = MagicMock()
    handler._start_pending_workflow_execution = MagicMock(return_value=True)

    def block_generation(*_args, progress_callback, **_kwargs):
        callback_ready.set()
        if not release_generation.wait(timeout=2):
            raise AssertionError("timed out waiting to finish script generation")
        progress_callback("late model progress")
        callback_invoked.set()
        return str(tmp_path / "generated.js"), {"name": "generated", "tools": ["coco"]}, False

    handler._generate_script_via_ai = MagicMock(side_effect=block_generation)

    generation_thread = threading.Thread(
        target=handler._generate_and_show_confirm_card,
        kwargs={
            "message_id": "old_card",
            "chat_id": "chat_1",
            "requirement": "old task",
            "project": project,
            "root_path": str(tmp_path),
            "selected_tools": ["coco"],
            "expected_session_key": "old_session",
        },
    )
    generation_thread.start()
    assert callback_ready.wait(timeout=1)

    handler.update_card.reset_mock()
    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.stop_workflow("stop_msg", "chat_1", project)

    release_generation.set()
    generation_thread.join(timeout=2)

    assert not generation_thread.is_alive()
    assert callback_invoked.is_set()
    handler.update_card.assert_not_called()
    handler.reply_text.assert_called_once_with("stop_msg", "Workflow 任务已停止。")
    handler._reply_workflow_error.assert_not_called()


from src.workflow_engine.models import AgentStatus, WorkflowMetrics
from src.workflow_engine.state_manager import WorkflowStateManager


def _make_state_manager():
    """Create a minimal state manager with a single running agent."""
    project = WorkflowProject(
        workflow_id="test",
        status=WorkflowStatus.RUNNING,
        metrics=WorkflowMetrics(),
    )
    sm = WorkflowStateManager(project)
    sm.on_phase_changed("phase1")
    return sm


class TestStateManagerStickyTerminal:
    """Agent terminal states are final — no transition may overwrite another."""

    def test_done_not_overwritten_by_failed(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_done(label, {"token_usage": 10, "duration_s": 1.0})

        snap = sm.snapshot()
        agent = snap.phases[0].agents[0]
        assert agent.status == AgentStatus.DONE
        assert snap.metrics.completed_agents == 1
        assert snap.metrics.failed_agents == 0

        sm.on_agent_failed(label, "some error")

        snap = sm.snapshot()
        agent = snap.phases[0].agents[0]
        assert agent.status == AgentStatus.DONE, "DONE must not be overwritten by FAILED"
        assert agent.error is None or agent.error == ""
        assert snap.metrics.completed_agents == 1
        assert snap.metrics.failed_agents == 0, "failed_agents must not increment for already-done agent"

    def test_done_not_overwritten_by_cancelled(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_done(label, {"token_usage": 10, "duration_s": 1.0})

        sm.on_agent_aborted(label, "race loser")

        snap = sm.snapshot()
        agent = snap.phases[0].agents[0]
        assert agent.status == AgentStatus.DONE, "DONE must not be overwritten by CANCELLED"
        assert snap.metrics.completed_agents == 1

    def test_failed_not_overwritten_by_done(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_failed(label, "timeout")

        snap = sm.snapshot()
        assert snap.phases[0].agents[0].status == AgentStatus.FAILED
        assert snap.metrics.failed_agents == 1

        sm.on_agent_done(label, {"token_usage": 5, "duration_s": 2.0})

        snap = sm.snapshot()
        agent = snap.phases[0].agents[0]
        assert agent.status == AgentStatus.FAILED, "FAILED must not be overwritten by DONE"
        assert agent.error == "timeout"
        assert snap.metrics.failed_agents == 1
        assert snap.metrics.total_tokens == 0, "token_usage from late done must not be counted"

    def test_failed_not_overwritten_by_cancelled(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_failed(label, "connection error")

        sm.on_agent_aborted(label, "race loser")

        snap = sm.snapshot()
        agent = snap.phases[0].agents[0]
        assert agent.status == AgentStatus.FAILED, "FAILED must not be overwritten by CANCELLED"
        assert snap.metrics.failed_agents == 1

    def test_cancelled_not_overwritten_by_done(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_aborted(label, "race loser")

        snap = sm.snapshot()
        assert snap.phases[0].agents[0].status == AgentStatus.CANCELLED
        assert snap.metrics.completed_agents == 1

        sm.on_agent_done(label, {"token_usage": 10, "duration_s": 1.0})

        snap = sm.snapshot()
        agent = snap.phases[0].agents[0]
        assert agent.status == AgentStatus.CANCELLED, "CANCELLED must not be overwritten by DONE"
        assert snap.metrics.completed_agents == 1
        assert snap.metrics.total_tokens == 0

    def test_cancelled_not_overwritten_by_failed(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_aborted(label, "race loser")

        sm.on_agent_failed(label, "some error")

        snap = sm.snapshot()
        agent = snap.phases[0].agents[0]
        assert agent.status == AgentStatus.CANCELLED, "CANCELLED must not be overwritten by FAILED"
        assert snap.metrics.failed_agents == 0

    def test_done_idempotent(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_done(label, {"token_usage": 10, "duration_s": 1.0})
        sm.on_agent_done(label, {"token_usage": 20, "duration_s": 2.0})

        snap = sm.snapshot()
        assert snap.metrics.completed_agents == 1, "completed_agents must not double-count"
        assert snap.metrics.total_tokens == 10, "second done must not overwrite token count"

    def test_failed_idempotent(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_failed(label, "error1")
        sm.on_agent_failed(label, "error2")

        snap = sm.snapshot()
        assert snap.metrics.failed_agents == 1, "failed_agents must not double-count"
        assert snap.metrics.completed_agents == 1

    def test_aborted_idempotent(self):
        sm = _make_state_manager()
        label = sm.on_agent_started("agent1", "coco", "phase1")
        sm.on_agent_aborted(label, "reason1")
        sm.on_agent_aborted(label, "reason2")

        snap = sm.snapshot()
        assert snap.metrics.completed_agents == 1, "completed_agents must not double-count on abort"

    def test_workflow_failed_overwritten_by_cancelled(self):
        """User-initiated cancel takes precedence over a runtime failure.

        When the user stops a failing workflow, the final status should be
        CANCELLED — the user's explicit action takes precedence over the
        failure that was in progress.
        """
        sm = _make_state_manager()
        sm.on_workflow_failed("fatal error")

        sm.on_workflow_cancelled("user cancelled")

        snap = sm.snapshot()
        assert snap.status == WorkflowStatus.CANCELLED, (
            "FAILED workflow should be overwritten by CANCELLED "
            "(user stop takes precedence)"
        )
        assert snap.error == "user cancelled"


class TestStateManagerMetricsAtomicity:
    """Metrics counters must be consistent even under concurrent updates."""

    def test_concurrent_done_no_double_count(self):
        sm = _make_state_manager()
        labels = [sm.on_agent_started(f"agent{i}", "coco", "phase1") for i in range(20)]
        barrier = threading.Barrier(20)

        def mark_done(label):
            barrier.wait()
            sm.on_agent_done(label, {"token_usage": 5, "duration_s": 0.1})

        threads = [threading.Thread(target=mark_done, args=(lbl,)) for lbl in labels]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        snap = sm.snapshot()
        assert snap.metrics.total_agents == 20
        assert snap.metrics.completed_agents == 20, (
            f"completed_agents must be exactly 20, got {snap.metrics.completed_agents}"
        )
        assert snap.metrics.failed_agents == 0
        assert snap.metrics.total_tokens == 100  # 20 * 5

    def test_concurrent_mixed_statuses_consistent(self):
        """Concurrent done/failed/abort calls must produce consistent totals."""
        sm = _make_state_manager()
        n_done = 15
        n_failed = 10
        n_aborted = 5
        total = n_done + n_failed + n_aborted

        done_labels = [sm.on_agent_started(f"d{i}", "coco", "phase1") for i in range(n_done)]
        failed_labels = [sm.on_agent_started(f"f{i}", "coco", "phase1") for i in range(n_failed)]
        aborted_labels = [sm.on_agent_started(f"a{i}", "coco", "phase1") for i in range(n_aborted)]

        barrier = threading.Barrier(total)
        threads = []

        for lbl in done_labels:
            def _d(l=lbl):
                barrier.wait()
                sm.on_agent_done(l, {"token_usage": 2, "duration_s": 0.1})
            threads.append(threading.Thread(target=_d))

        for lbl in failed_labels:
            def _f(l=lbl):
                barrier.wait()
                sm.on_agent_failed(l, "fail")
            threads.append(threading.Thread(target=_f))

        for lbl in aborted_labels:
            def _a(l=lbl):
                barrier.wait()
                sm.on_agent_aborted(l, "abort")
            threads.append(threading.Thread(target=_a))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        snap = sm.snapshot()
        assert snap.metrics.total_agents == total
        assert snap.metrics.completed_agents == total, (
            f"all agents must be terminal, completed_agents={snap.metrics.completed_agents}"
        )
        assert snap.metrics.failed_agents == n_failed
        # done + cached count: n_done agents
        done_count = sum(
            1 for ph in snap.phases for a in ph.agents
            if a.status == AgentStatus.DONE or a.status == AgentStatus.CACHED
        )
        assert done_count == n_done
        assert snap.metrics.total_tokens == n_done * 2


def test_admission_token_rejects_older_request_that_reaches_selection_late(
    tmp_path,
):
    """The latest ingress owner wins even before either request builds a card."""
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.settings.admin_user_ids = []
    handler.ctx.workflow_engine_manager.get_or_create.return_value = engine
    handler.get_engine_name = MagicMock(return_value="Coco")
    handler._resolve_tool_lists = MagicMock(
        return_value=({"coco": "Coco"}, ["coco"], [], ["coco"])
    )
    handler._get_workflow_models_for_tool = MagicMock(return_value=[])
    handler._preheat_workflow_models = MagicMock()
    handler._resolve_origin = MagicMock(side_effect=lambda value: value)
    handler.send_card_to_chat = MagicMock(return_value="selection_card")

    ok, error, old_owner = handler._supersede_incomplete_workflow(
        engine,
        root_path=str(tmp_path),
        current_user="user_1",
    )
    assert ok and error is None and old_owner is not None
    ok, error, new_owner = handler._supersede_incomplete_workflow(
        engine,
        root_path=str(tmp_path),
        current_user="user_1",
    )
    assert ok and error is None and new_owner is not None
    assert old_owner.stop_event.is_set()

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler._show_agent_selection_card(
            message_id="new_msg",
            chat_id="chat_1",
            requirement="new requirement",
            project=project,
            root_path=str(tmp_path),
            admission_owner=new_owner,
        )
        handler._show_agent_selection_card(
            message_id="old_msg",
            chat_id="chat_1",
            requirement="old requirement",
            project=project,
            root_path=str(tmp_path),
            admission_owner=old_owner,
        )

    handler.send_card_to_chat.assert_called_once()
    assert engine.project.pending.requirement == "new requirement"


def test_stale_selection_controller_cannot_overwrite_new_snapshot(tmp_path):
    project = _Project(str(tmp_path))
    handler = WorkflowHandler.__new__(WorkflowHandler)

    new_controller = SelectionFlowController(step=1)
    new_controller.add_or_update_selection(
        {"tool_name": "coco", "use_default_model": True},
        is_review=False,
    )
    project._wf_selection_session_key = "new_session"
    project._wf_selection_snapshot = new_controller.snapshot()
    expected_snapshot = project._wf_selection_snapshot

    old_controller = handler._get_selection_controller(
        project,
        session_key="old_session",
    )
    old_controller.add_or_update_selection(
        {"tool_name": "claude", "use_default_model": True},
        is_review=False,
    )

    assert handler._persist_selection_controller(
        project,
        old_controller,
    ) is False
    assert project._wf_selection_snapshot == expected_snapshot


def test_stale_tool_callback_cannot_toggle_new_pending_session(tmp_path):
    project = _Project(str(tmp_path))
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_CONFIRM,
        pending=PendingConfirmation(
            engine_session_key="old_session",
            initiator_user_id="user_1",
            selected_tools=["coco"],
        ),
    )
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._resolve_project_from_id = MagicMock(return_value=project)
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    handler.update_card = MagicMock()

    def replace_session(_candidate_tools):
        engine._project = WorkflowProject(
            status=WorkflowStatus.AWAITING_CONFIRM,
            pending=PendingConfirmation(
                engine_session_key="new_session",
                initiator_user_id="user_1",
                selected_tools=["coco"],
            ),
        )
        return ["claude"], []

    handler._validate_tools_against_registry = MagicMock(
        side_effect=replace_session
    )
    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        handler.handle_workflow_select_tool(
            "old_card",
            "chat_1",
            "proj_1",
            {
                "engine_session_key": "old_session",
                "tool_name": "claude",
            },
        )

    assert engine.project.pending.engine_session_key == "new_session"
    assert engine.project.pending.selected_tools == ["coco"]
    handler.update_card.assert_not_called()


def test_cancelled_generator_does_not_recreate_owned_script(tmp_path):
    script_path = (
        tmp_path
        / ".ghostap"
        / "workflow_scripts"
        / "generated-workflow-cancelled.js"
    )
    prompt_started = threading.Event()
    release_prompt = threading.Event()
    owner = _WorkflowLifecycleOwner(
        "generation_session",
        source_script_path=str(script_path),
    )

    class FakeSession:
        def set_tool_filter(self, _tool_filter):
            return None

        def send_prompt(self, _prompt, *, timeout):
            assert timeout > 0
            prompt_started.set()
            assert release_prompt.wait(timeout=2)
            result = MagicMock()
            result.text = """
export const meta = { name: "late", phases: [], tools: ["coco"] };
export default async function workflow() {
  return await agent("late", { tool: "coco" });
}
"""
            return result

        def close(self):
            return None

    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.settings.workflow_script_gen_timeout_s = 30
    outcome = {}

    def generate():
        try:
            handler._generate_script_via_ai(
                "generate a cancellable workflow",
                str(tmp_path),
                ["coco"],
                output_path=str(script_path),
                cancel_event=owner.stop_event,
                artifact_lock=owner.delivery_lock,
            )
        except Exception as exc:  # noqa: BLE001
            outcome["error"] = exc

    with (
        patch(
            "src.workflow_engine.tool_registry.get_available_tools",
            return_value={"coco": "Coco"},
        ),
        patch(
            "src.agent_session.create_engine_session",
            return_value=FakeSession(),
        ),
    ):
        thread = threading.Thread(target=generate)
        thread.start()
        assert prompt_started.wait(timeout=1)
        owner.stop_event.set()
        with owner.delivery_lock:
            pass
        release_prompt.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), _WorkflowGenerationCancelled)
    assert not script_path.exists()


def test_engine_constructor_failure_does_not_reuse_prior_run_components(
    tmp_path,
):
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "fail", phases: [], tools: [] };
export default async function workflow() { return "never"; }
""",
        encoding="utf-8",
    )
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    prior_state_manager = MagicMock()
    engine._state_manager = prior_state_manager
    on_error = MagicMock()

    with (
        patch(
            "src.workflow_engine.engine.WorkflowJournal",
            side_effect=RuntimeError("journal unavailable"),
        ),
        patch("src.workflow_engine.engine.WorkflowHistory.record"),
    ):
        result = engine.execute_workflow(
            "fail during construction",
            str(script_path),
            callbacks=WorkflowEngineCallbacks(on_error=on_error),
        )

    assert result.status == WorkflowStatus.FAILED
    prior_state_manager.on_workflow_failed.assert_not_called()
    prior_state_manager.on_workflow_cancelled.assert_not_called()
    on_error.assert_called_once()


def test_engine_cleanup_removes_queued_owner_artifacts(tmp_path):
    source_path = (
        tmp_path
        / ".ghostap"
        / "workflow_scripts"
        / "generated-workflow-queued.js"
    )
    source_path.parent.mkdir(parents=True)
    source_path.write_text("source", encoding="utf-8")
    execution_path = tmp_path / "ghostap-confirmed-queued.js"
    execution_path.write_text("copy", encoding="utf-8")

    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    owner = _WorkflowLifecycleOwner(
        "queued_session",
        source_script_path=str(source_path),
        execution_script_path=str(execution_path),
    )
    engine._workflow_start_owner = owner

    engine.cleanup()

    assert owner.stop_event.is_set()
    assert owner.done_event.is_set()
    assert not source_path.exists()
    assert not execution_path.exists()


def test_engine_keeps_run_claimed_until_bridge_workers_quiesce(tmp_path):
    """An old bridge callback must finish before a new run can reuse fields."""
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "quiesce", phases: [], tools: [] };
export default async function workflow() { return "ok"; }
""",
        encoding="utf-8",
    )
    wait_started = threading.Event()
    release_workers = threading.Event()
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    bridge_instances = []

    class FakeBridge:
        def __init__(self, *args, **kwargs):
            bridge_instances.append(self)

        @staticmethod
        def check_node_available():
            return True

        def start(self):
            return None

        def run(self):
            return "ok"

        def stop(self):
            return None

        def wait_for_workers(self):
            wait_started.set()
            assert release_workers.wait(timeout=2)

    first_result = {}

    def execute_first():
        first_result["project"] = engine.execute_workflow(
            "first",
            str(script_path),
        )

    with (
        patch("src.workflow_engine.engine.RuntimeBridge", FakeBridge),
        patch("src.workflow_engine.engine.WorkflowHistory.record"),
    ):
        first_thread = threading.Thread(target=execute_first)
        first_thread.start()
        assert wait_started.wait(timeout=1)
        assert engine.is_running

        active_project = engine.project
        duplicate_result = engine.execute_workflow(
            "must not replace first",
            str(script_path),
        )

        assert duplicate_result is active_project
        assert engine.project is active_project
        assert len(bridge_instances) == 1
        assert engine.is_running

        release_workers.set()
        first_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert first_result["project"].status == WorkflowStatus.COMPLETED
    assert engine.is_running is False


def test_engine_fails_closed_when_bridge_cannot_quiesce(tmp_path):
    """A failed worker drain must retain a non-reusable engine tombstone."""
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "tombstone", phases: [], tools: [] };
export default async function workflow() { return "ok"; }
""",
        encoding="utf-8",
    )
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    owner = _WorkflowLifecycleOwner("runtime_session")
    engine._workflow_start_owner = owner
    fake_executor = MagicMock()

    class FakeBridge:
        @staticmethod
        def check_node_available():
            return True

        def __init__(self, *args, **kwargs):
            return None

        def start(self):
            return None

        def run(self):
            return "ok"

        def stop(self):
            return None

        def wait_for_workers(self):
            return False

    with (
        patch("src.workflow_engine.engine.RuntimeBridge", FakeBridge),
        patch(
            "src.workflow_engine.engine.AgentExecutor",
            return_value=fake_executor,
        ),
        patch("src.workflow_engine.engine.WorkflowHistory.record"),
    ):
        result = engine.execute_workflow(
            "do not reuse",
            str(script_path),
            start_owner=owner,
        )

    assert result.status == WorkflowStatus.COMPLETED
    assert engine.run_state == EngineRunState.STOPPING
    assert engine._closing is True
    assert engine._workflow_start_owner is owner
    assert owner.done_event.is_set()
    assert owner.stop_event.is_set()
    fake_executor.shutdown.assert_not_called()


def test_cleanup_of_orphaned_bridge_is_bounded_and_retained(tmp_path):
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    bridge = MagicMock()
    bridge.wait_for_workers.return_value = False
    engine._bridge = bridge

    assert engine.cleanup() is False

    bridge.stop.assert_called_once_with()
    bridge.wait_for_workers.assert_called_once()
    assert bridge.wait_for_workers.call_args.kwargs["timeout"] <= 30
    assert engine._bridge is bridge
    assert engine._closing is True


def test_stop_can_cancel_bridge_while_start_is_waiting(tmp_path):
    """Node readiness must not hold the engine lock needed by stop()."""
    script_path = tmp_path / "workflow.js"
    script_path.write_text(
        """
export const meta = { name: "starting", phases: [], tools: [] };
export default async function workflow() { return "never"; }
""",
        encoding="utf-8",
    )
    start_entered = threading.Event()
    allow_start_return = threading.Event()
    stop_done = threading.Event()
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    owner = _WorkflowLifecycleOwner("starting_session")
    engine._workflow_start_owner = owner

    class FakeBridge:
        def __init__(self, *args, cancel_event, **kwargs):
            self.cancel_event = cancel_event

        @staticmethod
        def check_node_available():
            return True

        def start(self):
            start_entered.set()
            assert allow_start_return.wait(timeout=2)

        def run(self):
            assert self.cancel_event.wait(timeout=1)
            raise RuntimeError("Workflow cancelled")

        def stop(self):
            self.cancel_event.set()
            allow_start_return.set()

        def wait_for_workers(self):
            return True

    def execute():
        engine.execute_workflow(
            "cancel while starting",
            str(script_path),
            start_owner=owner,
        )

    def stop():
        engine.stop()
        stop_done.set()

    with (
        patch("src.workflow_engine.engine.RuntimeBridge", FakeBridge),
        patch("src.workflow_engine.engine.WorkflowHistory.record"),
    ):
        execution_thread = threading.Thread(target=execute)
        execution_thread.start()
        assert start_entered.wait(timeout=1)
        stop_thread = threading.Thread(target=stop)
        stop_thread.start()
        stopped_while_start_waited = stop_done.wait(timeout=0.2)
        if not stopped_while_start_waited:
            allow_start_return.set()
        stop_thread.join(timeout=2)
        execution_thread.join(timeout=2)

    assert stopped_while_start_waited
    assert not stop_thread.is_alive()
    assert not execution_thread.is_alive()


def test_legacy_tool_callbacks_serialize_selection_snapshot_updates(tmp_path):
    """Two same-session callbacks must preserve both controller selections."""
    project = _Project(str(tmp_path))
    project._wf_selection_session_key = "same_session"
    project._wf_selection_snapshot = SelectionFlowController().snapshot()
    engine = WorkflowEngine(chat_id="chat_1", root_path=str(tmp_path))
    engine._project = WorkflowProject(
        status=WorkflowStatus.AWAITING_TOOL_SELECT,
        pending=PendingConfirmation(
            requirement="parallel selection",
            engine_session_key="same_session",
            initiator_user_id="user_1",
            selected_tools=["seed"],
        ),
    )
    handler = WorkflowHandler.__new__(WorkflowHandler)
    handler.ctx = MagicMock()
    handler.ctx.workflow_engine_manager.get.return_value = engine
    handler._resolve_project_from_id = MagicMock(return_value=project)
    handler._get_root_path = MagicMock(return_value=str(tmp_path))
    handler._reply_workflow_error = MagicMock()
    handler._validate_tools_against_registry = MagicMock(
        side_effect=lambda tools: (tools, []),
    )
    handler._send_combined_selection_card = MagicMock()

    first_persist_entered = threading.Event()
    allow_first_persist = threading.Event()
    second_done = threading.Event()
    persist_count = 0
    count_lock = threading.Lock()
    original_persist = handler._persist_selection_controller

    def delayed_persist(target_project, controller):
        nonlocal persist_count
        with count_lock:
            persist_count += 1
            call_number = persist_count
        if call_number == 1:
            first_persist_entered.set()
            assert allow_first_persist.wait(timeout=2)
        return original_persist(target_project, controller)

    handler._persist_selection_controller = delayed_persist

    def select(tool_name):
        handler.handle_workflow_select_tool(
            "card_1",
            "chat_1",
            "proj_1",
            {
                "engine_session_key": "same_session",
                "tool_name": tool_name,
            },
        )

    with patch("src.thread.get_current_sender_id", return_value="user_1"):
        first = threading.Thread(target=select, args=("coco",))
        first.start()
        assert first_persist_entered.wait(timeout=1)

        def select_second():
            select("claude")
            second_done.set()

        second = threading.Thread(target=select_second)
        second.start()
        second_completed_before_release = second_done.wait(timeout=0.1)
        allow_first_persist.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not second_completed_before_release
    assert not first.is_alive()
    assert not second.is_alive()
    selections = project._wf_selection_snapshot["review_selections"]
    assert {
        selection["tool_name"]
        for selection in selections.values()
    } == {"coco", "claude"}
