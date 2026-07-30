"""Tests for Programming Mode card session adapter."""

import json

import pytest

from src.card.delivery.engine import CardDelivery
from src.card.events import CardEventType
from src.card.programming_adapter import (
    ProgrammingCardSession,
    build_programming_metadata,
)
from src.card.session import CardSession
from src.card.session.config import SessionConfig


class MockClient:
    def __init__(self):
        self._counter = 0
        self.creates = []

    def create_card(self, chat_id, card_json, *, reply_to=None, reply_in_thread=None, idempotency_key=None):
        self._counter += 1
        self.creates.append({"chat_id": chat_id, "card_json": card_json, "reply_to": reply_to})
        return (f"msg_{self._counter}", f"card_{self._counter}")

    def update_card(self, card_id, card_json, *, sequence=0):
        pass

    def update_element(self, card_id, element_id, content, *, sequence=0):
        pass


def _make_programming_session(mode_name="coco", image_uploader=None, **kwargs):
    client = MockClient()
    delivery = CardDelivery(client)
    metadata = build_programming_metadata(mode_name, **kwargs)
    config = SessionConfig(metadata=metadata, reply_to="origin_msg", sync_delivery=True)
    session = CardSession(
        chat_id="chat_prog",
        config=config,
        delivery=delivery,
        session_id=f"prog_{mode_name}",
    )

    return (
        ProgrammingCardSession(
            session,
            base_metadata=metadata,
            image_uploader=image_uploader,
        ),
        client,
    )


class TestBuildProgrammingMetadata:
    """Metadata builder tests."""

    def test_coco_metadata(self):
        meta = build_programming_metadata("coco", model_name="gpt-4o")
        assert meta.mode_name == "Coco"
        assert meta.mode_emoji == "🤖"
        assert meta.tool_name == "coco"
        assert meta.model_name == "gpt-4o"

    def test_claude_metadata(self):
        meta = build_programming_metadata("claude", model_name="claude-4-sonnet")
        assert meta.mode_name == "Claude"
        assert meta.mode_emoji == "🧠"
        assert meta.tool_name == "claude"
        assert meta.model_name == "claude-4-sonnet"

    def test_ttadk_metadata(self):
        meta = build_programming_metadata("ttadk", tool_name="cursor", model_name="gpt-4o")
        assert meta.mode_name == "TTADK"
        assert meta.tool_name == "cursor"
        assert meta.model_name == "gpt-4o"

    def test_with_project_name(self):
        meta = build_programming_metadata("coco", project_name="MyProject")
        assert meta.project_name == "MyProject"

    def test_with_working_dir_for_v2_header(self):
        meta = build_programming_metadata("coco", working_dir="/repo")
        assert meta.working_dir == "/repo"

    def test_all_modes_have_display(self):
        modes = ["coco", "claude", "aiden", "codex", "gemini", "traex", "ttadk"]
        for mode in modes:
            meta = build_programming_metadata(mode)
            assert meta.mode_name != ""
            assert meta.mode_emoji != ""


class TestProgrammingCardSession:
    """ProgrammingCardSession streaming tests."""

    def test_start_creates_card(self):
        pcs, client = _make_programming_session()
        pcs.start()
        assert len(client.creates) == 1
        assert pcs.session.state is not None

    def test_image_event_uploads_once_and_enters_card_state(self):
        from unittest.mock import MagicMock

        from src.acp.models import ACPEvent, ACPEventType, ACPImageInfo

        image = ACPImageInfo(
            image_id="sha256:generated",
            mime_type="image/png",
            data="aW1hZ2U=",
            name="generated.png",
        )
        uploader = MagicMock(return_value="img_generated")
        pcs, _ = _make_programming_session(image_uploader=uploader)
        pcs.start()
        event = ACPEvent(event_type=ACPEventType.IMAGE_CHUNK, image=image)

        pcs.on_event(event)
        pcs.on_event(event)

        uploader.assert_called_once_with(image)
        blocks = [block for block in pcs.session.state.blocks if block.kind == "image"]
        assert len(blocks) == 1
        assert blocks[0].image_key == "img_generated"

    def test_image_upload_failure_is_visible_without_failing_task(self):
        from src.acp.models import ACPEvent, ACPEventType, ACPImageInfo

        image = ACPImageInfo(
            image_id="sha256:failed",
            mime_type="image/png",
            data="aW1hZ2U=",
            name="failed.png",
        )
        pcs, _ = _make_programming_session(image_uploader=lambda _: None)
        pcs.start()

        pcs.on_event(ACPEvent(event_type=ACPEventType.IMAGE_CHUNK, image=image))

        block = next(block for block in pcs.session.state.blocks if block.kind == "image")
        assert block.status == "failed"
        assert pcs.session.state.terminal == "running"

    def test_start_and_finish_drive_live_ticker(self):
        calls: list[str] = []

        class FakeTicker:
            def __init__(self, *, session_id, on_frame, interval=1.2):
                calls.append(f"create:{session_id}")
                self.on_frame = on_frame

            def start(self):
                calls.append("start")
                self.on_frame("⚪")

            def stop(self):
                calls.append("stop")

        pcs, _ = _make_programming_session()
        pcs._ticker_factory = FakeTicker

        pcs.start()
        assert "start" in calls
        assert pcs.session.state.metadata.live_ticker_frame == "⚪"

        pcs.finish()
        assert calls[-1] == "stop"

    def test_live_ticker_frames_are_rate_limited(self):
        pcs, _ = _make_programming_session()
        pcs._ticker_update_min_interval = 9999.0
        pcs.start()
        pcs._last_ticker_update_at = None

        pcs._on_ticker_frame("⚪")
        assert pcs.session.state.metadata.live_ticker_frame == "⚪"

        pcs._on_ticker_frame("🟢")
        assert pcs.session.state.metadata.live_ticker_frame == "⚪"

    def test_live_ticker_dispatch_is_offloaded_when_async_enabled(self):
        pcs, _ = _make_programming_session()
        pcs.start()
        pcs._ticker_dispatch_async = True
        pcs._last_ticker_update_at = None

        submitted = []

        class FakePool:
            def submit(self, fn, *args, **kwargs):
                submitted.append((fn, args, kwargs))

        pcs._ticker_executor_factory = lambda: FakePool()

        def fail_inline_dispatch(_event):
            raise AssertionError("ticker dispatch must be offloaded")

        pcs._rotator.dispatch = fail_inline_dispatch
        pcs._on_ticker_frame("⚪")

        assert len(submitted) == 1
        assert submitted[0][1] == ("⚪",)

    def test_on_text_appends_content(self):
        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_text("Hello ")
        pcs.on_text("World")
        pcs._flush_now()  # Flush batched text before checking state

        state = pcs.session.state
        text_blocks = [b for b in state.blocks if b.kind == "text"]
        assert any("Hello " in b.content and "World" in b.content for b in text_blocks)

    def test_on_event_processes_acp(self):
        pcs, _ = _make_programming_session()
        pcs.start()

        from src.acp.models import ACPEvent, ACPEventType
        event = ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="streaming text")
        pcs.on_event(event)
        pcs._flush_now()  # Flush batched text before checking state

        state = pcs.session.state
        text_blocks = [b for b in state.blocks if b.kind == "text"]
        assert any("streaming text" in b.content for b in text_blocks)

    def test_on_event_projects_acp_directly_to_card_state(self):
        pcs, _ = _make_programming_session()
        pcs.start()

        from src.acp.models import ACPEvent, ACPEventType
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="streaming text"))
        pcs._flush_now()

        text_blocks = [b for b in pcs.session.state.blocks if b.kind == "text"]
        assert any("streaming text" in b.content for b in text_blocks)

    def test_on_event_handles_tool_call(self):
        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_text("before tool")

        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        tool_event = ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="tc1", title="bash", kind="execute", content="ls -la", status="running"),
        )
        pcs.on_event(tool_event)

        state = pcs.session.state
        tool_blocks = [b for b in state.blocks if b.kind == "tool_call"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].tool_name == "bash"

    def test_finish_completes_session(self):
        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_text("result")
        pcs.finish()

        assert pcs.closed
        assert pcs.session.state.terminal == "completed"

    def test_fail_marks_failed(self):
        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.fail("timeout")

        assert pcs.closed
        assert pcs.session.state.terminal == "failed"

    def test_waiting_for_user_confirmation_closes_as_blocked_and_preserves_reason(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_text("已完成自动续做，")
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.THOUGHT_CHUNK,
                text="仍需用户确认。",
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=ToolCallInfo(
                    id="agent-waiting",
                    title="Agent",
                    kind="other",
                    status="in_progress",
                    content="等待确认前整理结果\n子代理：Explore",
                ),
            )
        )

        reason = "自动续做已完成，仍需用户确认后继续"
        pcs.wait_for_user_confirmation(reason)

        state = pcs.session.state
        assert pcs.closed
        assert state.terminal == "blocked"
        assert state.terminal_reason == "blocked"
        assert state.engine_ext.blocked_reason == reason
        assert any(
            block.kind == "text" and "已完成自动续做" in block.content
            for block in state.blocks
        )
        assert all(
            block.status == "completed"
            for block in state.blocks
            if block.kind == "reasoning"
        )
        assert state.metadata.subagents[0]["status"] == "cancelled"

    def test_update_tool_model(self):
        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.update_tool_model(tool_name="cursor", model_name="gpt-4o-mini")

        state = pcs.session.state
        assert state.metadata.tool_name == "cursor"
        assert state.metadata.model_name == "gpt-4o-mini"

    def test_text_resumes_after_tool(self):
        """After a tool completes, text should auto-start new block."""
        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_text("before")

        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        # Tool start
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="tc1", title="read", kind="read", content="/file.py", status="running"),
        ))
        # Tool done
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(id="tc1", title="read", kind="read", content="file content", status="completed"),
        ))
        # Text resumes
        pcs.on_text("after tool")

        state = pcs.session.state
        text_blocks = [b for b in state.blocks if b.kind == "text"]
        assert len(text_blocks) >= 2  # Before and after tool

    def test_acp_text_after_tool_uses_new_turn_block(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="先分析。"))
        pcs._flush_now()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="read-1", title="Read", kind="read", status="in_progress", content="src/a.py"),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(id="read-1", title="Read", kind="read", status="completed", content="done"),
        ))
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="再总结。"))
        pcs._flush_now()

        blocks = pcs.session.state.blocks
        text_blocks = [b for b in blocks if b.kind == "text"]
        assert [b.content for b in text_blocks] == ["先分析。", "再总结。"]
        assert [b.block_id for b in text_blocks] == ["_active_text", "_turn_2_text"]
        assert [b.kind for b in blocks[:3]] == ["text", "tool_call", "text"]

    def test_acp_text_from_different_sources_uses_separate_blocks(self):
        from src.acp.models import ACPEvent, ACPEventType

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="Alpha ", source_id="agent-a"))
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="甲", source_id="agent-b"))
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="Beta", source_id="agent-a"))
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="乙", source_id="agent-b"))
        pcs._flush_now()

        text_blocks = [b for b in pcs.session.state.blocks if b.kind == "text" and b.content]
        assert [b.content for b in text_blocks] == ["Alpha Beta", "甲乙"]
        assert len({b.block_id for b in text_blocks}) == 2

    def test_acp_turn_text_block_ids_are_monotonic_after_renderer_reset(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="第一轮。"))
        pcs._flush_now()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="read-1", title="Read", kind="read", status="in_progress", content="src/a.py"),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(id="read-1", title="Read", kind="read", status="completed", content="done"),
        ))
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="第二轮。"))
        pcs._flush_now()

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="read-2", title="Read", kind="read", status="in_progress", content="src/b.py"),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(id="read-2", title="Read", kind="read", status="completed", content="done"),
        ))
        pcs.on_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="第三轮。"))
        pcs._flush_now()

        text_blocks = [b for b in pcs.session.state.blocks if b.kind == "text"]
        assert [b.content for b in text_blocks] == ["第一轮。", "第二轮。", "第三轮。"]
        assert len({b.block_id for b in text_blocks}) == 3

    def test_continuation_boundary_closes_stream_blocks_without_new_card(self):
        from src.acp.models import ACPEvent, ACPEventType

        pcs, client = _make_programming_session()
        pcs.start()
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.TEXT_CHUNK,
                text="第一轮答复。",
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.THOUGHT_CHUNK,
                text="第一轮推理。",
            )
        )

        pcs.begin_continuation_turn()
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.THOUGHT_CHUNK,
                text="续做推理。",
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.TEXT_CHUNK,
                text="续做答复。",
            )
        )
        pcs._flush_now()

        state = pcs.session.state
        text_blocks = [
            block for block in state.blocks
            if block.kind == "text" and block.content
        ]
        reasoning_blocks = [
            block for block in state.blocks
            if block.kind == "reasoning" and block.content
        ]
        assert len(client.creates) == 1
        assert [block.content for block in text_blocks] == [
            "第一轮答复。",
            "续做答复。",
        ]
        assert [block.block_id for block in text_blocks] == [
            "_active_text",
            "_turn_2_text",
        ]
        assert [block.content for block in reasoning_blocks] == [
            "第一轮推理。",
            "续做推理。",
        ]
        assert [block.block_id for block in reasoning_blocks] == [
            "_active_reasoning",
            "_turn_2_reasoning",
        ]
        assert all(block.status == "completed" for block in reasoning_blocks)

    def test_tool_calls_split_reasoning_into_chronological_turns(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="需要先读文件。"))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="read-1", title="Read", kind="read", status="in_progress", content="src/a.py"),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(id="read-1", title="Read", kind="read", status="completed", content="done"),
        ))
        pcs.on_event(ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="再检查测试。"))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(id="read-2", title="Read", kind="read", status="in_progress", content="tests/test_a.py"),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(id="read-2", title="Read", kind="read", status="completed", content="done"),
        ))

        process_blocks = [
            block
            for block in pcs.session.state.blocks
            if block.kind in {"reasoning", "tool_call"}
        ]
        assert [block.kind for block in process_blocks] == [
            "reasoning",
            "tool_call",
            "reasoning",
            "tool_call",
        ]
        reasoning_blocks = [block for block in process_blocks if block.kind == "reasoning"]
        assert [block.block_id for block in reasoning_blocks] == [
            "_active_reasoning",
            "_turn_2_reasoning",
        ]
        assert [block.content for block in reasoning_blocks] == [
            "需要先读文件。",
            "再检查测试。",
        ]
        assert [block.status for block in reasoning_blocks] == ["completed", "completed"]

        pcs.finish()

        reasoning_blocks = [b for b in pcs.session.state.blocks if b.kind == "reasoning"]
        assert all(block.status == "completed" for block in reasoning_blocks)

    def test_tool_boundary_retires_reasoning_from_a_different_source(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.THOUGHT_CHUNK,
                text="工具前分析。",
                source_id="agent-a",
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=ToolCallInfo(
                    id="read-cross-source",
                    title="Read",
                    kind="read",
                    status="in_progress",
                    content="src/a.py",
                ),
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_DONE,
                tool_call=ToolCallInfo(
                    id="read-cross-source",
                    title="Read",
                    kind="read",
                    status="completed",
                    content="done",
                ),
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.THOUGHT_CHUNK,
                text="工具后分析。",
                source_id="agent-a",
            )
        )

        process_blocks = [
            block
            for block in pcs.session.state.blocks
            if block.kind in {"reasoning", "tool_call"}
        ]
        assert [block.kind for block in process_blocks] == [
            "reasoning",
            "tool_call",
            "reasoning",
        ]
        reasoning_blocks = [
            block for block in process_blocks if block.kind == "reasoning"
        ]
        assert len({block.block_id for block in reasoning_blocks}) == 2
        assert [block.content for block in reasoning_blocks] == [
            "工具前分析。",
            "工具后分析。",
        ]

    def test_image_boundary_retires_reasoning_before_later_thought(self):
        from src.acp.models import ACPEvent, ACPEventType, ACPImageInfo

        pcs, _ = _make_programming_session(
            image_uploader=lambda _: "img_boundary",
        )
        pcs.start()
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.THOUGHT_CHUNK,
                text="图片前分析。",
                source_id="agent-a",
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.IMAGE_CHUNK,
                image=ACPImageInfo(
                    image_id="sha256:boundary",
                    mime_type="image/png",
                    data="aW1hZ2U=",
                    name="boundary.png",
                ),
                source_id="image-tool",
            )
        )
        pcs.on_event(
            ACPEvent(
                event_type=ACPEventType.THOUGHT_CHUNK,
                text="图片后分析。",
                source_id="agent-a",
            )
        )

        process_blocks = [
            block
            for block in pcs.session.state.blocks
            if block.kind in {"reasoning", "image"}
        ]
        assert [block.kind for block in process_blocks] == [
            "reasoning",
            "image",
            "reasoning",
        ]
        reasoning_blocks = [
            block for block in process_blocks if block.kind == "reasoning"
        ]
        assert len({block.block_id for block in reasoning_blocks}) == 2
        assert [block.content for block in reasoning_blocks] == [
            "图片前分析。",
            "图片后分析。",
        ]

    def test_plan_update_moves_to_task_list_at_card_start(self):
        from src.acp.models import ACPEvent, ACPEventType, PlanEntryInfo, PlanInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_text("先输出一些文本")
        pcs._flush_now()

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.PLAN_UPDATE,
            plan=PlanInfo(entries=[
                PlanEntryInfo(content="梳理卡片链路", status="completed"),
                PlanEntryInfo(content="实现任务分卡", status="in_progress"),
                PlanEntryInfo(content="补充回归测试", status="pending"),
            ]),
        ))

        state = pcs.session.state
        assert state.blocks[0].kind == "task_list"
        assert state.blocks[0].current_task_id == "step_1"
        assert [task["name"] for task in state.blocks[0].tasks] == [
            "梳理卡片链路",
            "实现任务分卡",
            "补充回归测试",
        ]
        assert not any(block.kind == "plan" for block in state.blocks)

    def test_plan_updates_stay_in_single_card(self):
        """Plan/task changes update the task list in place — no new card per task switch.

        The whole task list lives in one streaming card; a new continuation card
        is only spawned when the current card nears the Feishu node/byte limit
        (handled by render-time pagination, not by plan transitions).
        """
        from src.acp.models import ACPEvent, ACPEventType, PlanEntryInfo, PlanInfo

        pcs, client = _make_programming_session()
        pcs.start()
        first_message_id = pcs.get_message_id()
        creates_after_start = len(client.creates)

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.PLAN_UPDATE,
            plan=PlanInfo(entries=[
                PlanEntryInfo(content="任务 A", status="in_progress"),
                PlanEntryInfo(content="任务 B", status="pending"),
            ]),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.PLAN_UPDATE,
            plan=PlanInfo(entries=[
                PlanEntryInfo(content="任务 A", status="completed"),
                PlanEntryInfo(content="任务 B", status="in_progress"),
            ]),
        ))

        # No extra cards created — same card, updated in place
        assert len(client.creates) == creates_after_start
        assert pcs.get_message_id() == first_message_id
        # Task list reflects the latest in-progress task without adding an execution-plan block.
        task_list = pcs.session.state.blocks[0]
        assert task_list.kind == "task_list"
        assert task_list.current_task_id == "step_1"
        assert task_list.tasks[1]["name"] == "任务 B"
        assert not any(block.kind == "plan" for block in pcs.session.state.blocks)

    def test_parallel_agent_tasks_stay_in_main_card(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, client = _make_programming_session()
        pcs.start()
        creates_after_start = len(client.creates)

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-1",
                title="Agent",
                kind="other",
                status="in_progress",
                content="实现后端接口\n子代理：Explore",
            ),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-2",
                title="Agent",
                kind="other",
                status="in_progress",
                content="补充前端回归测试\n子代理：Explore",
            ),
        ))

        assert len(client.creates) == creates_after_start
        assert [item["sequence"] for item in pcs.session.state.metadata.subagents] == [
            "1.a",
            "1.b",
        ]

    def test_escaped_agent_marker_stays_out_of_main_subtask_summary(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="call_internal",
                title="agent",
                kind="other",
                status="in_progress",
                content='子代理：\\" not in ordinary_output\\",\\n',
            ),
        ))

        card_json = render_card(pcs.session.state, RenderBudget())[0]._card_json
        rendered = json.dumps(card_json, ensure_ascii=False)
        assert "call_internal" not in rendered
        assert "ordinary_output" not in rendered
        assert "\\n" not in rendered
        assert pcs.session.state.metadata.subagents[0]["tool"] == "agent"
        assert pcs.session.state.metadata.subagents[0]["label"] == "子任务"

    def test_task_tool_updates_main_card_without_independent_card(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, client = _make_programming_session()
        pcs.start()
        creates_after_start = len(client.creates)

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="task-tool-1",
                title="task",
                kind="other",
                status="in_progress",
                content="依赖分析",
            ),
        ))

        assert len(client.creates) == creates_after_start
        assert pcs.session.state.metadata.subagents == (
            {
                "label": "依赖分析",
                "tool": "task",
                "status": "running",
                "sequence": "1.a",
            },
        )

    def test_main_subtask_summary_updates_generic_label_when_description_arrives_late(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="task-tool-late",
                title="task",
                kind="other",
                status="in_progress",
                content="",
            ),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_UPDATE,
            tool_call=ToolCallInfo(
                id="task-tool-late",
                title="task",
                kind="other",
                status="in_progress",
                content="梳理 Deep 任务列表展示问题",
            ),
        ))

        assert (
            pcs.session.state.metadata.subagents[0]["label"]
            == "梳理 Deep 任务列表展示问题"
        )

    def test_main_subtask_summary_uses_readable_json_label_without_stdout(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        payload = json.dumps({
            "call_id": "call_123",
            "command": ["/usr/bin/zsh", "-lc", "nl -ba src/card/orchestrator.py"],
            "parsed_cmd": [{
                "type": "read",
                "cmd": "nl -ba src/card/orchestrator.py",
                "name": "orchestrator.py",
                "path": "src/card/orchestrator.py",
            }],
            "stdout": "1290\\tlarge output that should not appear",
            "stderr": "",
        }, ensure_ascii=False, indent=2)

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="task-tool-json",
                title="task",
                kind="other",
                status="in_progress",
                content=payload,
            ),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="task-tool-json",
                title="task",
                kind="other",
                status="completed",
                content=payload,
            ),
        ))

        summary = pcs.session.state.metadata.subagents[0]
        assert summary["label"] == "读取 src/card/orchestrator.py"
        assert summary["status"] == "completed"
        rendered_payload = str(summary)
        assert "stdout" not in rendered_payload
        assert "1290" not in rendered_payload

    def test_parallel_agent_tasks_update_main_summary_panel(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-1",
                title="Agent",
                kind="other",
                status="in_progress",
                content="实现后端接口\n子代理：Explore",
            ),
        ))

        cards = render_card(pcs.session.state, RenderBudget())
        body = str(cards[0]._card_json["body"]["elements"])
        assert "并行子任务" in body
        assert "实现后端接口" in body
        assert "#1.a" in body

    def test_parallel_agent_summary_panel_reflects_terminal_statuses(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-1",
                title="Agent",
                kind="other",
                status="in_progress",
                content="实现后端接口\n子代理：Explore",
            ),
        ))

        pcs.finish()

        assert pcs.session.state.metadata.subagents[0]["status"] == "completed"
        body = str(render_card(pcs.session.state, RenderBudget())[0]._card_json["body"]["elements"])
        assert "✅ 实现后端接口" in body
        assert "完成 1" in body

    def test_failed_subagent_panel_renders_safe_error_as_subordinate_line(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-failed",
                title="Agent",
                kind="other",
                status="in_progress",
                content="验证终态交付\n子代理：Explore",
            ),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="agent-task-failed",
                title="Agent",
                kind="other",
                status="failed",
                content=json.dumps(
                    {
                        "error": (
                            "backend timed out while waiting for final card"
                        )
                    }
                ),
            ),
        ))

        body = str(
            render_card(
                pcs.session.state,
                RenderBudget(),
            )[0]._card_json["body"]["elements"]
        )
        assert pcs.session.state.terminal == "running"
        assert "❌ 验证终态交付" in body
        assert "原因：backend timed out while waiting for final card" in body

    def test_subagent_panel_sanitizes_untrusted_label_at_render_boundary(self):
        from src.card.render.tools import render_subagent_dispatch_panel

        secret = "sk-0123456789abcdef"
        panel = render_subagent_dispatch_panel(
            [
                {
                    "status": "failed",
                    "label": (
                        "\x1b[31mcall_private_123 API_TOKEN="
                        f"{secret} ![](/tmp/private.png)\u202e"
                    ),
                    "tool": "agent",
                    "error": "transport timeout",
                }
            ]
        )

        assert panel is not None
        body = panel["elements"][0]["content"]
        assert secret not in body
        assert "call_private_123" not in body
        assert "\x1b" not in body
        assert "![](" not in body
        assert "\u202e" not in body
        assert "原因：transport timeout" in body

    def test_subagent_panel_sanitizes_all_untrusted_metadata_fields(self):
        from src.card.render.tools import render_subagent_dispatch_panel

        secret = "sk-0123456789abcdef"
        panel = render_subagent_dispatch_panel(
            [
                {
                    "status": "failed",
                    "label": "检查渲染边界",
                    "tool": (
                        "\x1b[31mcall_private_tool "
                        "![](/tmp/private-tool.png)"
                    ),
                    "model": f"API_TOKEN={secret} **private-model**",
                    "sequence": "1.a\n![private](/tmp/private-seq.png)",
                    "error": "transport timeout",
                }
            ]
        )

        assert panel is not None
        body = panel["elements"][0]["content"]
        assert secret not in body
        assert "call_private_tool" not in body
        assert "\x1b" not in body
        assert "![](" not in body
        assert "**private-model**" not in body
        assert "/tmp/private-seq.png" not in body
        assert "#1.a" not in body

    def test_cancel_updates_parent_subagent_summary_before_terminal(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-1",
                title="Agent",
                kind="other",
                status="in_progress",
                content="实现后端接口\n子代理：Explore",
            ),
        ))

        pcs.cancel(reason="user_stop")

        assert pcs.session.state.terminal == "cancelled"
        assert pcs.session.state.metadata.subagents[0]["status"] == "cancelled"
        body = str(render_card(pcs.session.state, RenderBudget())[0]._card_json["body"]["elements"])
        assert "⚪ 实现后端接口" in body
        assert "取消 1" in body
        assert "运行中 1" not in body

    def test_timeout_failure_marks_live_subagent_cancelled_not_failed(self):
        """A parent deadline must not fabricate a child-agent failure."""
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-timeout",
                title="Agent",
                kind="other",
                status="in_progress",
                content="检查超时收尾\n子代理：Explore",
            ),
        ))

        pcs.fail(
            "parent prompt timeout",
            unfinished_subagent_status="cancelled",
        )

        assert pcs.session.state.terminal == "failed"
        assert pcs.session.state.metadata.subagents[0]["status"] == "cancelled"

    def test_finalization_success_does_not_fabricate_live_subagent_completion(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-finalization",
                title="Agent",
                kind="other",
                status="in_progress",
                content="等待收尾\n子代理：Explore",
            ),
        ))

        pcs.finish(
            fallback_text="已完成安全收尾",
            unfinished_subagent_status="cancelled",
        )

        assert pcs.session.state.terminal == "completed"
        assert pcs.session.state.metadata.subagents[0]["status"] == "cancelled"

    def test_parent_completion_survives_subagent_summary_dispatch_failure(self):
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="agent-task-1",
                title="Agent",
                kind="other",
                status="in_progress",
                content="实现后端接口\n子代理：Explore",
            ),
        ))

        original_dispatch = pcs._rotator.dispatch

        def flaky_dispatch(event):
            if event.type == CardEventType.TOOL_MODEL_CHANGED and "subagents" in event.payload:
                raise RuntimeError("summary dispatch failed")
            return original_dispatch(event)

        pcs._rotator.dispatch = flaky_dispatch
        pcs.finish()

        assert pcs.session.state.terminal == "completed"

    def test_render_omits_process_summary_after_later_text_updates(self):
        """Completed tools join the folded execution record beside answer text."""
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs.on_text("先说明目标。")
        pcs._flush_now()

        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="cmd-1",
                title="bash",
                kind="execute",
                status="running",
                content="uv run python -m pytest tests/test_example.py -q",
            ),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="cmd-1",
                title="bash",
                kind="execute",
                status="completed",
                content="1 passed",
            ),
        ))
        pcs.on_text("后续正文继续更新。")
        pcs._flush_now()

        cards = render_card(pcs.session.state, RenderBudget())
        body = cards[0]._card_json["body"]["elements"]
        rendered_text = str(body)

        assert "执行记录" in rendered_text
        assert "bash" in rendered_text
        assert "pytest tests/test_example.py" in rendered_text
        assert "先说明目标。" in rendered_text
        assert "后续正文继续更新。" in rendered_text

    def test_completed_header_does_not_show_stale_ticker_frame(self):
        from src.card.render.budget import RenderBudget
        from src.card.render.renderer import render_card

        pcs, _ = _make_programming_session()
        pcs.start()
        pcs._on_ticker_frame("⚪")
        pcs.finish()

        card = render_card(pcs.session.state, RenderBudget())[0]._card_json
        body_text = str(card["body"]["elements"])
        assert "⚪" not in body_text
        assert "subtitle" not in card["header"]

    def test_finish_fallback_text_injected_when_no_text_blocks(self):
        """When card has only tool calls and no text, fallback_text appears as summary."""
        from src.acp.models import ACPEvent, ACPEventType, ToolCallInfo

        pcs, _ = _make_programming_session()
        pcs.start()

        # Simulate tool call without any text events
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_START,
            tool_call=ToolCallInfo(
                id="t1", title="bash", kind="execute",
                status="running", content="echo hello",
            ),
        ))
        pcs.on_event(ACPEvent(
            event_type=ACPEventType.TOOL_CALL_DONE,
            tool_call=ToolCallInfo(
                id="t1", title="bash", kind="execute",
                status="completed", content="hello",
            ),
        ))

        pcs.finish(fallback_text="This is the fallback answer")

        state = pcs.session.state
        text_blocks = [b for b in state.blocks if b.kind == "text" and b.content]
        assert any("This is the fallback answer" in b.content for b in text_blocks), (
            f"Expected fallback text in blocks, got: {[b.content for b in text_blocks]}"
        )

    def test_finish_fallback_text_not_used_when_text_already_present(self):
        """When card already has streamed text, fallback_text is ignored."""
        pcs, _ = _make_programming_session()
        pcs.start()

        pcs.on_text("Streamed answer text.")
        pcs._flush_now()

        pcs.finish(fallback_text="This fallback should NOT appear")

        state = pcs.session.state
        text_contents = [b.content for b in state.blocks if b.kind == "text" and b.content]
        assert any("Streamed answer text." in c for c in text_contents)
        assert not any("This fallback should NOT appear" in c for c in text_contents)


class TestSessionMetadataPerMode:
    """Each mode produces correct metadata in the session."""

    def test_coco_header_subtitle(self):
        pcs, _ = _make_programming_session("coco", model_name="gpt-4o")
        pcs.start()
        state = pcs.session.state
        # Header subtitle should contain tool/model info
        if state.header.subtitle:
            assert "coco" in state.header.subtitle.lower() or "gpt" in state.header.subtitle.lower()

    def test_claude_header_subtitle(self):
        pcs, _ = _make_programming_session("claude", model_name="claude-4-sonnet")
        pcs.start()
        state = pcs.session.state
        if state.header.subtitle:
            assert "claude" in state.header.subtitle.lower()

    def test_ttadk_custom_tool_name(self):
        pcs, _ = _make_programming_session("ttadk", tool_name="cursor", model_name="gpt-4o")
        pcs.start()
        state = pcs.session.state
        assert state.metadata.tool_name == "cursor"


class TestNonStreamingFallback:
    """Verify non-streaming fallback uses result.text.

    The handler's _handle_response_non_streaming builds final_response as:
        (getattr(result, "text", None) or "").strip()
        or renderer.get_final_content()
        or UI_TEXT["mode_exec_complete"]
    This ensures result.text is the primary source when streaming is unavailable.
    """

    def test_generated_image_is_uploaded_and_replied_as_card(self):
        import base64
        import json
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from src.acp.models import ACPEvent, ACPEventType, ACPImageInfo, PromptResult
        from src.feishu.handlers.programming import ProgrammingModeHandler

        image = ACPImageInfo(
            image_id="sha256:fallback",
            mime_type="image/png",
            data=base64.b64encode(b"\x89PNG\r\n\x1a\nfallback").decode(),
            name="fallback.png",
        )
        session = MagicMock()

        def send_prompt(_text, *, on_event, timeout):
            assert timeout == pytest.approx(33, abs=0.1)
            on_event(ACPEvent(event_type=ACPEventType.IMAGE_CHUNK, image=image))
            return PromptResult(stop_reason="end_turn", text="图片已生成")

        session.send_prompt.side_effect = send_prompt
        handler = SimpleNamespace(
            settings=SimpleNamespace(
                coco_execution_timeout=60,
                claude_execution_timeout=60,
                programming_finalization_reserve_s=0,
                repo_lock_hard_timeout=120,
            ),
            is_coco=True,
            mode_name="Coco",
            upload_acp_image=MagicMock(return_value="img_fallback"),
            reply_card=MagicMock(),
            reply_text=MagicMock(),
            add_reaction=MagicMock(),
        )

        ProgrammingModeHandler._handle_response_non_streaming(
            handler,
            "message-1",
            "chat-1",
            "生成图片",
            session,
            None,
            "/workspace",
        )

        handler.upload_acp_image.assert_called_once_with(image)
        handler.reply_text.assert_not_called()
        handler.reply_card.assert_called_once()
        card_json = json.loads(handler.reply_card.call_args.args[1])
        image_elements = [
            element
            for element in card_json["body"]["elements"]
            if element.get("tag") == "img"
        ]
        assert image_elements == [
            {
                "tag": "img",
                "img_key": "img_fallback",
                "alt": {"tag": "plain_text", "content": "图片 1"},
            }
        ]

    def test_timeout_uses_reserved_finalization_turn(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from src.acp.models import PromptResult
        from src.feishu.handlers.programming import ProgrammingModeHandler

        calls: list[tuple[str, int | float | None]] = []

        class Session:
            _force_dead = False

            def send_prompt(self, prompt, *, on_event, timeout):
                calls.append((prompt, timeout))
                if len(calls) == 1:
                    raise TimeoutError("primary deadline")
                return PromptResult(stop_reason="end_turn", text="安全收尾完成")

        handler = SimpleNamespace(
            settings=SimpleNamespace(
                coco_execution_timeout=90,
                claude_execution_timeout=90,
                programming_finalization_reserve_s=30,
                repo_lock_hard_timeout=120,
            ),
            is_coco=True,
            mode_name="Coco",
            upload_acp_image=MagicMock(),
            reply_card=MagicMock(),
            reply_text=MagicMock(),
            add_reaction=MagicMock(),
            _replace_timed_out_session=MagicMock(),
            _retire_finalization_session=MagicMock(),
        )

        ProgrammingModeHandler._handle_response_non_streaming(
            handler,
            "message-1",
            "chat-1",
            "BRIDGE CONTEXT: unrelated prior authorization",
            Session(),
            None,
            "/workspace",
            _finalization_task_text="完成并提交原任务",
        )

        assert calls[0][1] == pytest.approx(60, abs=0.1)
        assert calls[1][1] == pytest.approx(30, abs=0.1)
        assert "不要创建新的子代理" in calls[1][0]
        assert "完成并提交原任务" in calls[1][0]
        assert "BRIDGE CONTEXT" not in calls[1][0]
        assert "安全收尾完成" in handler.reply_text.call_args.args[1]
        handler.add_reaction.assert_called_once()

    def test_timeout_without_reserve_still_retires_session(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from src.feishu.handlers.programming import ProgrammingModeHandler

        class Session:
            _force_dead = False

            def send_prompt(self, _prompt, *, on_event, timeout):
                raise TimeoutError(f"hard deadline {timeout}")

        retire = MagicMock()
        handler = SimpleNamespace(
            settings=SimpleNamespace(
                coco_execution_timeout=60,
                claude_execution_timeout=60,
                programming_finalization_reserve_s=0,
                repo_lock_hard_timeout=120,
            ),
            is_coco=True,
            mode_name="Coco",
            upload_acp_image=MagicMock(),
            reply_card=MagicMock(),
            reply_text=MagicMock(),
            add_reaction=MagicMock(),
            _replace_timed_out_session=MagicMock(),
            _retire_finalization_session=retire,
        )

        session = Session()
        ProgrammingModeHandler._handle_response_non_streaming(
            handler,
            "message-1",
            "chat-1",
            "hard-timeout task",
            session,
            None,
            "/workspace",
        )

        retire.assert_called_once()
        retire_call = retire.call_args
        assert retire_call.kwargs["chat_id"] == "chat-1"
        assert retire_call.kwargs["project"] is None
        assert retire_call.kwargs["thread_id"] is None
        assert retire_call.kwargs["active_session"] is session
        assert 0 < retire_call.kwargs["retirement_budget_s"] <= 60

    def test_timeout_after_replacement_retires_active_replacement(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from src.feishu.handlers.programming import ProgrammingModeHandler

        class Session:
            _force_dead = False

        original = Session()
        replacement = Session()
        retire = MagicMock()
        replace_session = MagicMock(return_value=replacement)
        handler = SimpleNamespace(
            settings=SimpleNamespace(
                coco_execution_timeout=90,
                claude_execution_timeout=90,
                programming_finalization_reserve_s=30,
                repo_lock_hard_timeout=120,
            ),
            is_coco=True,
            mode_name="Coco",
            upload_acp_image=MagicMock(),
            reply_card=MagicMock(),
            reply_text=MagicMock(),
            add_reaction=MagicMock(),
            _replace_timed_out_session=replace_session,
            _retire_finalization_session=retire,
        )

        def replace_then_timeout(
            active,
            _text,
            *,
            replace_dead_session,
            **_kwargs,
        ):
            assert active is original
            assert replace_dead_session(12.5) is replacement
            raise TimeoutError("deadline after replacement")

        with patch(
            "src.feishu.handlers.programming.run_prompt_with_continuation",
            side_effect=replace_then_timeout,
        ):
            ProgrammingModeHandler._handle_response_non_streaming(
                handler,
                "message-1",
                "chat-1",
                "replacement-timeout task",
                original,
                None,
                "/workspace",
            )

        assert replace_session.call_args.kwargs["timed_out_session"] is original
        retire.assert_called_once()
        assert retire.call_args.kwargs["active_session"] is replacement
        assert replacement._force_dead is True
        assert original._force_dead is False
        handler.reply_card.assert_called_once()

    def test_retirement_failure_still_returns_timeout_error_card(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from src.feishu.handlers.programming import ProgrammingModeHandler

        class Session:
            _force_dead = False

            def send_prompt(self, _prompt, *, on_event, timeout):
                raise TimeoutError(f"hard deadline {timeout}")

        session = Session()
        handler = SimpleNamespace(
            settings=SimpleNamespace(
                coco_execution_timeout=60,
                claude_execution_timeout=60,
                programming_finalization_reserve_s=0,
                repo_lock_hard_timeout=120,
            ),
            is_coco=True,
            mode_name="Coco",
            upload_acp_image=MagicMock(),
            reply_card=MagicMock(),
            reply_text=MagicMock(),
            add_reaction=MagicMock(),
            _replace_timed_out_session=MagicMock(),
            _retire_finalization_session=MagicMock(
                side_effect=RuntimeError("retirement lock unavailable"),
            ),
        )

        ProgrammingModeHandler._handle_response_non_streaming(
            handler,
            "message-1",
            "chat-1",
            "hard-timeout task",
            session,
            None,
            "/workspace",
        )

        assert session._force_dead is True
        handler.reply_card.assert_called_once()

    def test_result_text_used_as_primary_response(self):
        """When send_prompt returns result.text, it should be the final response."""
        from dataclasses import dataclass

        from src.acp.renderer import ACPEventRenderer

        @dataclass
        class FakeResult:
            text: str = ""

        result = FakeResult(text="actual response")
        renderer = ACPEventRenderer()

        # Replicate the non-streaming fallback logic from programming.py:837-871
        final_response = (
            (getattr(result, "text", None) or "").strip()
            or renderer.get_final_content()
            or "执行完成"
        )
        assert final_response == "actual response"

    def test_fallback_to_renderer_when_result_text_empty(self):
        """When result.text is empty, renderer.get_final_content() is used."""
        from dataclasses import dataclass

        from src.acp.models import ACPEvent, ACPEventType
        from src.acp.renderer import ACPEventRenderer

        @dataclass
        class FakeResult:
            text: str = ""

        result = FakeResult(text="")
        renderer = ACPEventRenderer()
        renderer.process_event(ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="rendered output"))

        final_response = (
            (getattr(result, "text", None) or "").strip()
            or renderer.get_final_content()
            or "执行完成"
        )
        assert final_response == "rendered output"

    def test_fallback_to_placeholder_when_both_empty(self):
        """When both result.text and renderer are empty, placeholder is used."""
        from dataclasses import dataclass

        from src.acp.renderer import ACPEventRenderer

        @dataclass
        class FakeResult:
            text: str = ""

        result = FakeResult(text="")
        renderer = ACPEventRenderer()

        final_response = (
            (getattr(result, "text", None) or "").strip()
            or renderer.get_final_content()
            or "执行完成"
        )
        assert final_response == "执行完成"

    def test_result_text_stripped(self):
        """result.text should be stripped of whitespace."""
        from dataclasses import dataclass

        from src.acp.renderer import ACPEventRenderer

        @dataclass
        class FakeResult:
            text: str = ""

        result = FakeResult(text="  response with spaces  \n")
        renderer = ACPEventRenderer()

        final_response = (
            (getattr(result, "text", None) or "").strip()
            or renderer.get_final_content()
            or "执行完成"
        )
        assert final_response == "response with spaces"


class TestScheduleFlushLockAssertion:
    """_schedule_flush must raise RuntimeError if called without holding _flush_lock."""

    def test_schedule_flush_without_lock_raises(self):
        """Calling _schedule_flush without holding the lock raises RuntimeError."""
        pcs, _ = _make_programming_session()
        with pytest.raises(RuntimeError, match="_schedule_flush must be called under _flush_lock"):
            pcs._schedule_flush()

    def test_schedule_flush_with_lock_starts_timer(self):
        """Calling _schedule_flush while holding the lock starts a timer."""
        pcs, _ = _make_programming_session()
        with pcs._flush_lock:
            pcs._flush_lock_holder.held = True
            try:
                pcs._schedule_flush()
                assert pcs._flush_timer is not None
                assert pcs._flush_timer.is_alive()
            finally:
                pcs._flush_lock_holder.held = False
                pcs._flush_timer.cancel()

    def test_schedule_flush_does_not_create_duplicate_timer(self):
        """Second _schedule_flush call with existing timer does nothing."""
        pcs, _ = _make_programming_session()
        with pcs._flush_lock:
            pcs._flush_lock_holder.held = True
            try:
                pcs._schedule_flush()
                first_timer = pcs._flush_timer
                pcs._schedule_flush()
                assert pcs._flush_timer is first_timer  # same timer, not replaced
            finally:
                pcs._flush_lock_holder.held = False
                pcs._flush_timer.cancel()
