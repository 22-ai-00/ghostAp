"""Integration tests for DeepRenderer's single-main-card behavior.

Deep keeps plan and subagent progress in its sticky task list while all ACP
text, reasoning, tools, and images share the main execution stream. Only the
shared delivery capacity policy may append a continuation message.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.acp.models import ACPEvent, ACPEventType, PlanEntryInfo, PlanInfo, ToolCallInfo
from src.card.events import CardEventType
from src.card.state.reducer import reduce_card_state

# ---------------------------------------------------------------------------
# Fakes and Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeDeepProject:
    name: str = "test-project"
    root_path: str = "/tmp/test"
    project_id: str = "dp_test"
    status: object = None
    duration_seconds: float = 313.6

    def __post_init__(self):
        if self.status is None:
            from src.deep_engine.models import DeepProjectStatus
            self.status = DeepProjectStatus.COMPLETED

    def duration(self) -> float:
        return self.duration_seconds


class FakeHandler:
    """Minimal fake handler for DeepRenderer."""

    def __init__(self):
        self.settings = MagicMock()
        self.settings.deep_stream_interval = 0.1
        self.settings.deep_stream_min_chars = 10
        self.project_manager = MagicMock()
        self.context_manager = MagicMock()
        self.add_reaction = MagicMock()
        self.ctx = FakeRendererCtx()
        self._request_ids = {}

    def ensure_request_id(self, message_id, **kwargs):
        return f"req_{message_id}"

    def reply_text(self, message_id, text):
        pass

    def get_working_dir(self, chat_id):
        return "/tmp/test"

    def get_engine_name(self, chat_id, **kwargs):
        return "Coco"

    def get_card_delivery(self):
        return MagicMock()


class FakeRendererCtx:
    """Minimal renderer context."""

    def __init__(self):
        self.progress_reporter = MagicMock()
        self.deep_engine_manager = MagicMock()
        self.deep_engine_manager.snapshot.return_value = None
        self.deep_engine_manager.snapshot_active.return_value = []


class SessionTracker:
    """Tracks all sessions created (simulates create_card calls)."""

    def __init__(self):
        self.sessions_created: list[MagicMock] = []
        self._lock = threading.Lock()

    def create_session(self, *args, **kwargs):
        """Each create_session call represents a create_card call."""
        session = MagicMock()
        session.dispatch = MagicMock()
        session.closed = False
        session.sequence = len(self.sessions_created) + 1
        session.session_started_at = time.monotonic()
        if len(args) >= 3:
            session._metadata = args[2]
        session.delivered_message_id = f"msg_{len(self.sessions_created)}"
        with self._lock:
            self.sessions_created.append(session)
        return session

    @property
    def create_card_count(self) -> int:
        with self._lock:
            return len(self.sessions_created)


class FakeDeepHeartbeat:
    """Deterministic heartbeat used by Deep callback integration tests."""

    def __init__(self, *, session_id, on_tick, interval):
        self.session_id = session_id
        self.on_tick = on_tick
        self.interval = interval
        self.running = False
        self.activity = "thinking"

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def reset(self, activity="thinking"):
        self.activity = activity

    def fire(self, elapsed=5.0):
        if self.running:
            self.on_tick(elapsed, self.activity)


def _make_plan_event(entries: list[tuple[str, str]]) -> ACPEvent:
    """Create a PLAN_UPDATE ACPEvent with given (content, status) entries."""
    plan_entries = [PlanEntryInfo(content=c, status=s) for c, s in entries]
    return ACPEvent(
        event_type=ACPEventType.PLAN_UPDATE,
        plan=PlanInfo(entries=plan_entries),
    )


def _make_text_event(text: str = "hello") -> ACPEvent:
    return ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text=text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeepRendererSingleCard:
    """Verify Deep keeps one ordered main-card execution flow."""

    def _setup_renderer(self):
        """Create a DeepRenderer with mocked dependencies and session tracking."""
        from src.feishu.renderers.deep_renderer import DeepRenderer

        tracker = SessionTracker()
        handler = FakeHandler()

        renderer = DeepRenderer(handler)
        renderer.ctx = FakeRendererCtx()
        renderer._session_factory = MagicMock()

        # Patch create_session to track card creation
        renderer.create_session = tracker.create_session
        renderer._get_session_factory = lambda: MagicMock()
        renderer._build_hooks = lambda *a, **kw: ()
        renderer.check_warning_banner = lambda *a, **kw: None

        return renderer, tracker

    def _create_callbacks(
        self,
        renderer,
        *,
        project=None,
        engine_name: str = "Coco",
        requirement_text: str | None = None,
    ):
        mock_settings = MagicMock()
        mock_settings.card.build_heartbeat_interval = 5.0
        with patch(
            "src.feishu.renderers._deep_stream_processor.get_settings",
            return_value=mock_settings,
        ), patch(
            "src.feishu.renderers._deep_stream_processor.BuildHeartbeat",
            FakeDeepHeartbeat,
        ):
            return renderer.create_deep_callbacks(
                message_id="msg_1",
                chat_id="chat_1",
                project=project,
                engine_name=engine_name,
                requirement_text=requirement_text,
            )

    def test_deep_session_has_question_summary_before_first_dispatch(self):
        renderer, tracker = self._setup_renderer()

        self._create_callbacks(
            renderer,
            requirement_text="  优化Deep模式消息卡片标题并展示用户问题  ",
        )

        metadata = tracker.sessions_created[0]._metadata
        assert metadata.question_title == "优化Deep模式消息卡片标题…"
        assert len(metadata.question_title) <= 15

    def test_deep_session_without_requirement_has_stable_fallback_before_dispatch(self):
        renderer, tracker = self._setup_renderer()

        self._create_callbacks(renderer)

        assert tracker.sessions_created[0]._metadata.question_title == "Deep 任务"

    def test_deep_start_uses_spec_style_cycle_phase_events(self):
        """Deep cards should use the same cycle/phase structure as Spec cards."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        events = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type")
        ]
        event_types = [event.type for event in events]

        assert CardEventType.CYCLE_STARTED in event_types
        phase_events = [event for event in events if event.type == CardEventType.PHASE_STARTED]
        assert phase_events
        assert phase_events[-1].payload["phase"] == "analyzing"
        assert not any(
            event.type == CardEventType.TEXT_STARTED
            and event.payload.get("block_id") == "_main_text"
            for event in events
        )
        task_updates = [
            event for event in events if event.type == CardEventType.TASK_LIST_UPDATED
        ]
        assert task_updates[-1].payload == {
            "tasks": [
                {
                    "task_id": "_deep_main",
                    "name": "分析与执行主流程",
                    "status": "in_progress",
                },
            ],
            "current_task_id": "_deep_main",
        }

    def test_deep_start_card_dispatches_on_analyzing_start(self):
        """Deep should create the first visible card before waiting for model output."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)
        assert callbacks.on_analyzing_start is not None

        callbacks.on_analyzing_start("investigate startup latency")

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        events = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type")
        ]
        event_types = [event.type for event in events]

        assert CardEventType.STARTED in event_types
        assert event_types.index(CardEventType.TASK_LIST_UPDATED) < event_types.index(
            CardEventType.STARTED
        )
        assert CardEventType.CYCLE_STARTED in event_types
        assert CardEventType.PHASE_STARTED in event_types

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        events_after_done = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type")
        ]
        assert [event.type for event in events_after_done].count(CardEventType.STARTED) == 1
        assert [event.type for event in events_after_done].count(CardEventType.CYCLE_STARTED) == 1
        assert [event.type for event in events_after_done].count(CardEventType.PHASE_STARTED) == 1

    def test_deep_plan_update_transitions_to_spec_style_build_phase(self):
        """Deep plan updates should switch from analyzing to the shared build phase panel."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)
        callbacks.on_event(_make_plan_event([
            ("Analyze requirements", "completed"),
            ("Write implementation", "in_progress"),
        ]))

        main_session = tracker.sessions_created[0]
        events = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type")
        ]

        assert any(
            event.type == CardEventType.PHASE_DONE
            and event.payload["phase"] == "analyzing"
            for event in events
        )
        assert any(
            event.type == CardEventType.PHASE_STARTED
            and event.payload["phase"] == "build"
            for event in events
        )
        task_updates = [
            event for event in events if event.type == CardEventType.TASK_LIST_UPDATED
        ]
        assert task_updates[-1].payload["tasks"] == [
            {"task_id": "step_0", "name": "Analyze requirements", "status": "completed"},
            {"task_id": "step_1", "name": "Write implementation", "status": "in_progress"},
        ]

    def test_multi_task_plan_stays_single_card(self):
        """Deep ignores task-card fanout and renders plan tasks on the main card."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        assert tracker.create_card_count == 1

        plan_event = _make_plan_event([
            ("Analyze requirements", "in_progress"),
            ("Write implementation", "in_progress"),
            ("Run tests", "in_progress"),
        ])
        callbacks.on_event(plan_event)

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type") and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert task_updates
        assert task_updates[-1].payload["tasks"] == [
            {"task_id": "step_0", "name": "Analyze requirements", "status": "in_progress"},
            {"task_id": "step_1", "name": "Write implementation", "status": "in_progress"},
            {"task_id": "step_2", "name": "Run tests", "status": "in_progress"},
        ]

    def test_project_done_completes_main_card(self):
        """on_project_done completes the same Deep card."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        plan_event = _make_plan_event([
            ("Task A", "in_progress"),
            ("Task B", "in_progress"),
        ])
        callbacks.on_event(plan_event)
        assert tracker.create_card_count == 1

        dp.status = DeepProjectStatus.COMPLETED
        callbacks.on_project_done(dp)

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        completed_calls = [
            call for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type") and call.args[0].type == CardEventType.COMPLETED
        ]
        assert len(completed_calls) == 1
        assert "执行完成" in completed_calls[0].args[0].payload["summary"]
        renderer.handler.add_reaction.assert_called_once_with("msg_1", "PARTY")
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and hasattr(call.args[0], "type")
            and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert [task["status"] for task in task_updates[-1].payload["tasks"]] == [
            "completed",
            "completed",
        ]

    def test_no_plan_project_done_completes_placeholder_task(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)

        callbacks.on_analyzing_start("finish without a plan event")
        dp = FakeDeepProject()
        callbacks.on_project_done(dp)

        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and hasattr(call.args[0], "type")
            and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert task_updates[-1].payload["tasks"] == [{
            "task_id": "_deep_main",
            "name": "分析与执行主流程",
            "status": "completed",
        }]
        assert task_updates[-1].payload["current_task_id"] == ""

    def test_project_done_failed_formats_incomplete_summary(self):
        """Deep incomplete terminal text should not leak format placeholders."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        callbacks.on_event(_make_plan_event([
            ("Task A", "completed"),
            ("Task B", "in_progress"),
        ]))
        dp.status = DeepProjectStatus.FAILED
        callbacks.on_project_done(dp)

        main_session = tracker.sessions_created[0]
        failed_events = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type") and call.args[0].type == CardEventType.FAILED
        ]
        assert failed_events
        assert "{completed}" not in failed_events[-1].payload["error"]
        assert "已完成 1/2 步" in failed_events[-1].payload["error"]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and hasattr(call.args[0], "type")
            and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert [task["status"] for task in task_updates[-1].payload["tasks"]] == [
            "completed",
            "failed",
        ]

    def test_paused_project_cancels_unfinished_subagent_tasks_on_main_card(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("pause with mixed child states")

        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent_done",
                title="subagent",
                kind="execute",
                status="in_progress",
                content="finished child\n子代理：Explore",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="agent_done",
                title="subagent",
                kind="execute",
                status="completed",
                content="finished child",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent_active",
                title="subagent",
                kind="execute",
                status="in_progress",
                content="active child\n子代理：Write",
            ),
        ))

        from src.deep_engine.models import DeepProjectStatus
        callbacks.on_project_done(FakeDeepProject(status=DeepProjectStatus.PAUSED))

        assert tracker.create_card_count == 1
        terminal_types = [
            call.args[0].type
            for call in tracker.sessions_created[0].dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None)
            in {
                CardEventType.COMPLETED,
                CardEventType.FAILED,
                CardEventType.CANCELLED,
            }
        ]
        assert terminal_types == [CardEventType.CANCELLED]
        main_task_updates = [
            call.args[0]
            for call in tracker.sessions_created[0].dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.TASK_LIST_UPDATED
        ]
        by_id = {
            task["task_id"]: task["status"]
            for task in main_task_updates[-1].payload["tasks"]
        }
        assert by_id["agent_done"] == "completed"
        assert by_id["agent_active"] == "cancelled"
        processor = callbacks.on_event.__self__
        assert processor._task_registry.get("agent_active").status == "cancelled"

    def test_subagent_output_and_summary_stay_on_main_card(self):
        """Management wrappers stay compact while child output uses the main flow."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        callbacks.on_event(_make_plan_event([
            ("Task A", "in_progress"),
            ("Task B", "in_progress"),
        ]))
        assert tracker.create_card_count == 1

        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent_call_1",
                title="agent",
                kind="execute",
                status="in_progress",
                content="检查卡片路由\n子代理：Explore",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_UPDATE,
            tool_call=ToolCallInfo(
                id="agent_call_1",
                title="shell",
                kind="execute",
                status="in_progress",
                content="正在检查 Deep 子任务卡片",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TEXT_CHUNK,
            source_id="agent_call_1",
            text="子代理正在核对投递边界。",
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            source_id="agent_call_1",
            tool_call=ToolCallInfo(
                id="child_shell_1",
                title="shell",
                kind="execute",
                status="in_progress",
                content="uv run pytest tests/test_card_delivery_engine.py -q",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            source_id="agent_call_1",
            tool_call=ToolCallInfo(
                id="child_shell_1",
                title="shell",
                kind="execute",
                status="completed",
                content="all passed",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="agent_call_1",
                title="shell",
                kind="execute",
                status="completed",
                content="子任务完成",
            ),
        ))

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        main_event_types = [
            call.args[0].type
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type")
        ]
        assert CardEventType.TEXT_STARTED in main_event_types
        assert CardEventType.TEXT_DELTA in main_event_types
        assert CardEventType.TOOL_STARTED in main_event_types
        assert CardEventType.TOOL_DONE in main_event_types
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type") and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert task_updates[-1].payload["tasks"][-1] == {
            "task_id": "agent_call_1",
            "name": "🧬 检查卡片路由",
            "status": "completed",
        }

    def test_codex_activity_uses_stable_thread_id_without_false_completion(self):
        """Activity calls are operations on one child, not child terminal frames."""
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("track Codex child activity")

        source_id = "thread-stable-card-audit"
        for call_id, activity in (
            ("activity-call-started", "started"),
            ("activity-call-interacted", "interacted"),
        ):
            for event_type, status in (
                (ACPEventType.TOOL_CALL_START, "in_progress"),
                (ACPEventType.TOOL_CALL_DONE, "completed"),
            ):
                callbacks.on_event(ACPEvent(
                    event_type=event_type,
                    source_id=source_id,
                    tool_call=ToolCallInfo(
                        id=call_id,
                        title=f"Subagent {activity}",
                        kind="other",
                        status=status,
                        subagent_source_id=source_id,
                        subagent_path="/root/card-audit",
                        subagent_activity=activity,
                    ),
                ))

        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.TASK_LIST_UPDATED
        ]
        latest_tasks = task_updates[-1].payload["tasks"]
        child_tasks = [task for task in latest_tasks if task["task_id"] != "_deep_main"]

        assert child_tasks == [{
            "task_id": source_id,
            "name": "🧬 card-audit",
            "status": "in_progress",
        }]
        assert tracker.create_card_count == 1

    def test_failed_codex_interrupt_activity_keeps_deep_child_running(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("failed interrupt must not cancel child")

        source_id = "thread-stable-card-audit"
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            source_id=source_id,
            tool_call=ToolCallInfo(
                id="activity-call-interrupt",
                title="Subagent interrupted",
                kind="other",
                status="failed",
                subagent_source_id=source_id,
                subagent_path="/root/card-audit",
                subagent_activity="interrupted",
            ),
        ))

        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.TASK_LIST_UPDATED
        ]
        child_tasks = [
            task
            for task in task_updates[-1].payload["tasks"]
            if task["task_id"] != "_deep_main"
        ]
        assert child_tasks == [{
            "task_id": source_id,
            "name": "🧬 card-audit",
            "status": "in_progress",
        }]
        assert tracker.create_card_count == 1

    def test_codex_collaboration_state_finalizes_stable_deep_task(self):
        """The collaboration snapshot, not activity DONE, owns child terminal state."""
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("finish Codex child from collaboration state")

        source_id = "thread-stable-card-audit"
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            source_id=source_id,
            tool_call=ToolCallInfo(
                id="activity-call-started",
                title="Subagent started",
                kind="other",
                status="in_progress",
                subagent_source_id=source_id,
                subagent_path="/root/card-audit",
                subagent_activity="started",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="collaboration-call-wait",
                title="wait",
                kind="agent",
                status="completed",
                collaboration_tool="wait",
                collaboration_receivers=(source_id,),
                subagent_states=({
                    "source_id": source_id,
                    "status": "completed",
                    "message": "card audit complete",
                },),
            ),
        ))

        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.TASK_LIST_UPDATED
        ]
        latest_tasks = task_updates[-1].payload["tasks"]
        child_tasks = [task for task in latest_tasks if task["task_id"] != "_deep_main"]

        assert child_tasks == [{
            "task_id": source_id,
            "name": "🧬 card-audit",
            "status": "completed",
        }]
        assert tracker.create_card_count == 1

    def test_failed_collaboration_without_child_keeps_error_on_main_card(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("show failed spawn")

        failed_event = ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="spawn_failed",
                title="spawn_agent",
                kind="agent",
                status="failed",
                content="子代理启动失败：并发槽位不足",
                collaboration_tool="spawn_agent",
            ),
        )
        callbacks.on_event(failed_event)
        callbacks.on_event(failed_event)

        main_session = tracker.sessions_created[0]
        tool_events = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) in {
                CardEventType.TOOL_STARTED,
                CardEventType.TOOL_FAILED,
            }
        ]
        tool_event_types = [event.type for event in tool_events]
        state = None
        for event in tool_events:
            state = reduce_card_state(state, event)

        assert tracker.create_card_count == 1
        assert tool_event_types == [
            CardEventType.TOOL_STARTED,
            CardEventType.TOOL_FAILED,
            CardEventType.TOOL_FAILED,
        ]
        failed_blocks = [
            block
            for block in state.blocks
            if block.kind == "tool_call" and block.block_id == "spawn_failed"
        ]
        assert len(failed_blocks) == 1
        failed_block = failed_blocks[0]
        assert failed_block.status == "failed"
        assert "并发槽位不足" in failed_block.tool_output

    def test_collaboration_receivers_without_states_create_task_summary(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("track receiver-only spawn")

        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="spawn_receiver",
                title="spawn_agent",
                kind="agent",
                status="in_progress",
                content="检查投递边界\n子代理：Explore",
                collaboration_tool="spawn_agent",
                collaboration_receivers=("thread_receiver",),
            ),
        ))

        task_updates = [
            call.args[0]
            for call in tracker.sessions_created[0].dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None)
            == CardEventType.TASK_LIST_UPDATED
        ]
        receiver = next(
            task
            for task in task_updates[-1].payload["tasks"]
            if task["task_id"] == "thread_receiver"
        )
        assert tracker.create_card_count == 1
        assert receiver == {
            "task_id": "thread_receiver",
            "name": "🧬 检查投递边界",
            "status": "in_progress",
        }

    def test_completed_parent_does_not_fabricate_running_child_completion(self):
        from src.deep_engine.models import DeepProjectStatus

        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("running child terminal truth")
        source_id = "thread-still-running"
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_UPDATE,
            tool_call=ToolCallInfo(
                id="collaboration-call-wait",
                title="wait",
                kind="agent",
                status="completed",
                collaboration_tool="wait",
                collaboration_receivers=(source_id,),
                subagent_states=({
                    "source_id": source_id,
                    "status": "running",
                    "message": "still running",
                },),
            ),
        ))

        callbacks.on_project_done(
            FakeDeepProject(status=DeepProjectStatus.COMPLETED)
        )

        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.TASK_LIST_UPDATED
        ]
        child = next(
            task
            for task in task_updates[-1].payload["tasks"]
            if task["task_id"] == source_id
        )
        assert child["status"] == "cancelled"
        assert tracker.create_card_count == 1

    def test_execute_failure_with_subagent_marker_does_not_create_task_summary(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("verify ordinary execute routing")
        execute_call_id = "call_zXAT0JlJc0dqRewiUJK8nHYL"
        initial_card_count = tracker.create_card_count

        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id=execute_call_id,
                title="exec",
                kind="execute",
                status="in_progress",
                content="uv run python -m pytest tests/test_card_orchestrator.py -q",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id=execute_call_id,
                title="exec",
                kind="execute",
                status="failed",
                content='assert "子代理：" not in ordinary_output',
            ),
        ))

        assert tracker.create_card_count == initial_card_count
        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and hasattr(call.args[0], "type")
            and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert all(
            task.get("task_id") != execute_call_id
            for event in task_updates
            for task in event.payload["tasks"]
        )
        tool_events = [
            call.args[0].type
            for call in main_session.dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) in {
                CardEventType.TOOL_STARTED,
                CardEventType.TOOL_FAILED,
            }
        ]
        assert tool_events == [
            CardEventType.TOOL_STARTED,
            CardEventType.TOOL_FAILED,
        ]

    def test_parallel_subagents_stay_in_single_main_card(self):
        """Parallel child summaries never allocate independent Feishu cards."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent_call_1",
                title="agent",
                kind="execute",
                status="in_progress",
                content="梳理 Deep 卡片问题\n子代理：Explore",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent_call_2",
                title="subagent",
                kind="execute",
                status="in_progress",
                content="补充 Deep 回归测试\n子代理：Write",
            ),
        ))
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="agent_call_1",
                title="agent",
                kind="execute",
                status="completed",
                content="完成卡片路径梳理",
            ),
        ))

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type") and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert task_updates
        latest_tasks = task_updates[-1].payload["tasks"]
        assert latest_tasks == [
            {"task_id": "_deep_main", "name": "分析与执行主流程", "status": "in_progress"},
            {"task_id": "agent_call_1", "name": "🧬 梳理 Deep 卡片问题", "status": "completed"},
            {"task_id": "agent_call_2", "name": "🧬 补充 Deep 回归测试", "status": "in_progress"},
        ]
        main_event_types = [
            call.args[0].type
            for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type")
        ]
        assert CardEventType.TOOL_STARTED not in main_event_types
        assert CardEventType.TOOL_DONE not in main_event_types

    def test_deep_does_not_emit_long_running_retry_warning(self):
        renderer, tracker = self._setup_renderer()
        renderer.check_warning_banner = lambda *a, **kw: (
            "⚠️ 执行耗时较长，若无响应可尝试停止后重试"
        )

        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("long-running deep task")
        callbacks.on_event(_make_text_event("still making progress"))

        main_session = tracker.sessions_created[0]
        warnings = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and hasattr(call.args[0], "type")
            and call.args[0].type == CardEventType.WARNING_UPDATED
        ]
        assert warnings == []

    def test_deep_heartbeat_refreshes_latest_card_and_stops_on_terminal(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)

        callbacks.on_analyzing_start("keep elapsed time live")
        processor = callbacks.on_analyzing_start.__self__
        heartbeat = processor._heartbeat
        assert heartbeat.running is True

        main_session = tracker.sessions_created[0]
        before = main_session.dispatch.call_count
        heartbeat.fire(elapsed=5.0)
        heartbeat_event = main_session.dispatch.call_args_list[-1].args[0]

        assert main_session.dispatch.call_count == before + 1
        assert heartbeat_event.type == CardEventType.PROGRESS_UPDATED
        assert heartbeat_event.payload["label"] == "🧠 分析/规划中"

        callbacks.on_error("stop heartbeat")
        assert heartbeat.running is False

    def test_deep_terminal_event_uses_authoritative_project_duration(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        project = FakeDeepProject(duration_seconds=313.6)

        callbacks.on_analyzing_start("show whole deep runtime")
        callbacks.on_project_done(project)

        completed = [
            call.args[0]
            for call in tracker.sessions_created[0].dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.COMPLETED
        ]
        assert len(completed) == 1
        assert completed[0].payload["duration_seconds"] == 313.6

    def test_invalid_project_duration_falls_back_without_losing_terminal(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        project = FakeDeepProject(duration_seconds=-1.0)

        callbacks.on_analyzing_start("survive wall clock rollback")
        callbacks.on_project_done(project)

        completed = [
            call.args[0]
            for call in tracker.sessions_created[0].dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.COMPLETED
        ]
        assert len(completed) == 1
        assert "duration_seconds" not in completed[0].payload

    def test_deep_error_event_uses_authoritative_project_duration(self):
        renderer, tracker = self._setup_renderer()
        project = FakeDeepProject(duration_seconds=187.2)
        renderer._get_engine = MagicMock(
            return_value=SimpleNamespace(ext={"project": project})
        )
        callbacks = self._create_callbacks(renderer)

        callbacks.on_analyzing_start("preserve failed deep runtime")
        callbacks.on_error("boom")

        failed = [
            call.args[0]
            for call in tracker.sessions_created[0].dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None) == CardEventType.FAILED
        ]
        assert len(failed) == 1
        assert failed[0].payload["duration_seconds"] == 187.2

    def test_error_fails_main_card(self):
        """on_error fails the same Deep card."""
        renderer, tracker = self._setup_renderer()

        callbacks = self._create_callbacks(renderer)

        from src.deep_engine.models import DeepProjectStatus
        dp = FakeDeepProject(status=DeepProjectStatus.EXECUTING)
        callbacks.on_analyzing_done(dp)

        plan_event = _make_plan_event([
            ("Task A", "pending"),
            ("Task B", "pending"),
        ])
        callbacks.on_event(plan_event)

        callbacks.on_error("Something went wrong")

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        failed_calls = [
            call for call in main_session.dispatch.call_args_list
            if call.args and hasattr(call.args[0], "type") and call.args[0].type == CardEventType.FAILED
        ]
        assert len(failed_calls) == 1
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and hasattr(call.args[0], "type")
            and call.args[0].type == CardEventType.TASK_LIST_UPDATED
        ]
        assert [task["status"] for task in task_updates[-1].payload["tasks"]] == [
            "failed",
            "failed",
        ]

    def test_parent_error_cancels_unconfirmed_subagent_task_on_main_card(self):
        renderer, tracker = self._setup_renderer()
        callbacks = self._create_callbacks(renderer)
        callbacks.on_analyzing_start("fail active child")
        callbacks.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent_live",
                title="subagent",
                kind="execute",
                status="in_progress",
                content="active child\n子代理：Explore",
            ),
        ))

        callbacks.on_error("parent failed")

        assert tracker.create_card_count == 1
        main_session = tracker.sessions_created[0]
        task_updates = [
            call.args[0]
            for call in main_session.dispatch.call_args_list
            if call.args
            and getattr(call.args[0], "type", None)
            == CardEventType.TASK_LIST_UPDATED
        ]
        agent_task = next(
            task
            for task in task_updates[-1].payload["tasks"]
            if task["task_id"] == "agent_live"
        )
        assert agent_task["status"] == "cancelled"
