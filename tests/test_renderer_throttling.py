import json
from unittest.mock import MagicMock, patch

import pytest

from src.acp import ACPEventType
from src.acp.models import ACPEvent, ToolCallInfo
from src.card.builders.system import SystemBuilder
from src.deep_engine.models import DeepProjectStatus
from src.feishu.renderers.deep_renderer import DeepRenderer


class TestRendererThrottling:
    @pytest.fixture
    def mock_handler(self):
        handler = MagicMock()
        handler.ctx = MagicMock()
        handler.settings = MagicMock()

        # Configure settings for throttle testing
        handler.settings.deep_stream_interval = 0.0  # Instant for test
        handler.settings.deep_stream_min_chars = 0
        handler.settings.default_reply_mode = "thread"
        handler.settings.card.deep_compact_default = False

        # Mock reporter
        handler.ctx.progress_reporter.format_summary.return_value = "Mock Summary"

        # Mock engines
        mock_engine = MagicMock()
        mock_engine.project = MagicMock()
        mock_engine.project.duration.return_value = 10.0
        mock_engine.project.total_criteria = 10
        mock_engine.project.satisfied_count = 5
        mock_engine.progress = MagicMock()
        mock_engine.progress.format_summary.return_value = "Mock Summary"
        mock_engine.get_rendered_content.return_value = "Mock Content"
        handler.ctx.deep_engine_manager.get.return_value = mock_engine

        # Mock new card API methods
        handler.reply_card = MagicMock(return_value="reply_msg_id")
        handler.update_card = MagicMock(return_value=True)

        return handler

    def test_deep_renderer_throttling(self, mock_handler):
        """Test that DeepRenderer dispatches ACP events to CardSession"""
        mock_handler.settings.engine_collapsible_enabled = False
        renderer = DeepRenderer(mock_handler)

        mock_project = MagicMock()
        mock_project.project_id = "test_proj"

        with patch("src.feishu.renderers.base.BaseRenderer.create_session") as mock_create:
            mock_session = MagicMock()
            mock_create.return_value = mock_session

            callbacks = renderer.create_deep_callbacks(
                "msg_id", "chat_id", mock_project, initial_message_id="existing_msg_id"
            )

            mock_event = MagicMock()
            mock_event.event_type = ACPEventType.TEXT_CHUNK
            mock_event.text = "some text content"

            # Events should be dispatched to the CardSession
            callbacks.on_event(mock_event)
            assert mock_session.dispatch.called

            # Verify it dispatched a text_delta event
            calls = [c.args[0] for c in mock_session.dispatch.call_args_list]
            types = [c.type.value for c in calls]
            assert "text_delta" in types

    def test_deep_renderer_critical_update(self, mock_handler):
        """Test that critical updates (success/error) dispatch terminal events to CardSession"""
        mock_handler.settings.engine_collapsible_enabled = False
        renderer = DeepRenderer(mock_handler)
        mock_project = MagicMock()
        mock_project.project_id = "test_proj"

        with patch("src.feishu.renderers.base.BaseRenderer.create_session") as mock_create:
            mock_session = MagicMock()
            mock_create.return_value = mock_session

            callbacks = renderer.create_deep_callbacks(
                "msg_id", "chat_id", mock_project, initial_message_id="reply_msg_id"
            )

            mock_deep_project = MagicMock()
            mock_deep_project.status = DeepProjectStatus.COMPLETED

            callbacks.on_project_done(mock_deep_project)

            # Should dispatch completed event
            calls = [c.args[0] for c in mock_session.dispatch.call_args_list]
            types = [c.type.value for c in calls]
            assert "completed" in types

    def test_deep_subagent_events_update_main_card_task(self, mock_handler):
        renderer = DeepRenderer(mock_handler)
        project = MagicMock(project_id="test_proj")

        with patch("src.feishu.renderers.base.BaseRenderer.create_session") as create:
            session = create.return_value
            callbacks = renderer.create_deep_callbacks("msg_id", "chat_id", project)

            for event_type, path, status in (
                (ACPEventType.TOOL_CALL_START, None, "in_progress"),
                (ACPEventType.TOOL_CALL_UPDATE, "/root/card-audit", "in_progress"),
                (ACPEventType.TOOL_CALL_DONE, "/root/card-audit", "completed"),
            ):
                callbacks.on_event(ACPEvent(
                    event_type=event_type,
                    tool_call=ToolCallInfo(
                        id="call-1",
                        title="Task",
                        kind="agent",
                        status=status,
                        subagent_source_id="child-1",
                        subagent_path=path,
                    ),
                ))

            task_lists = [
                call.args[0].payload["tasks"]
                for call in session.dispatch.call_args_list
                if call.args[0].type.value == "task_list_updated"
            ]
            child_snapshots = [
                next((task for task in tasks if task["task_id"] == "child-1"), None)
                for tasks in task_lists
            ]
            child_snapshots = [task for task in child_snapshots if task is not None]
            assert [(task["name"], task["status"]) for task in child_snapshots] == [
                ("🧬 子任务", "in_progress"),
                ("🧬 card-audit", "in_progress"),
                ("🧬 card-audit", "completed"),
            ]
            assert all(
                task["task_id"] != "call-1"
                for tasks in task_lists for task in tasks
            )

    def test_deep_collaboration_snapshot_reaches_main_card(self, mock_handler):
        renderer = DeepRenderer(mock_handler)
        project = MagicMock(project_id="test_proj")

        with patch("src.feishu.renderers.base.BaseRenderer.create_session") as create:
            session = create.return_value
            callbacks = renderer.create_deep_callbacks("msg_id", "chat_id", project)

            for event_type, child_status, call_status in (
                (ACPEventType.TOOL_CALL_START, "running", "in_progress"),
                (ACPEventType.TOOL_CALL_DONE, "completed", "completed"),
            ):
                callbacks.on_event(ACPEvent(
                    event_type=event_type,
                    tool_call=ToolCallInfo(
                        id="collab-1",
                        title="Task",
                        kind="agent",
                        status=call_status,
                        collaboration_tool="spawn_agent",
                        collaboration_receivers=("child-1",),
                        subagent_states=({"source_id": "child-1", "status": child_status},),
                    ),
                ))

            task_lists = [
                call.args[0].payload["tasks"]
                for call in session.dispatch.call_args_list
                if call.args[0].type.value == "task_list_updated"
            ]
            assert [
                next(task["status"] for task in tasks if task["task_id"] == "child-1")
                for tasks in task_lists if any(task["task_id"] == "child-1" for task in tasks)
            ] == ["in_progress", "completed"]

    def test_help_card_structure(self):
        """Test that build_help_card returns valid structure"""
        msg_type, content_json = SystemBuilder.build_help_card(
            category="main",
            session_idle_timeout=600,
            session_idle_warn_at_remaining=120,
            lock_undo_window_seconds=300,
        )
        content = json.loads(content_json)

        assert msg_type == "interactive"
        assert "header" in content
        assert "body" in content

        # Help content stays compact; quick-action buttons live in /menu.
        elements = content["body"]["elements"]
        buttons_row = next((e for e in elements if e.get("tag") == "column_set"), None)
        assert buttons_row is None

        # Check for content text (now inside collapsible_panel elements)
        found = False
        for e in elements:
            if e.get("tag") == "collapsible_panel":
                for inner in e.get("elements", []):
                    if inner.get("tag") == "markdown" and "编程模式" in inner.get("content", ""):
                        found = True
                        break
            elif e.get("tag") == "markdown" and "编程模式" in e.get("content", ""):
                found = True
            if found:
                break
        assert found
