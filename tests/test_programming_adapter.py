"""programming_adapter SectionLayout integration tests."""
from __future__ import annotations

import pytest

from src.acp.models import (
    ACPEvent,
    ACPEventType,
    PlanEntryInfo,
    PlanInfo,
    ToolCallInfo,
)
from src.card.delivery.engine import CardDelivery
from src.card.events import CardEvent, CardEventType
from src.card.programming_adapter import (
    ProgrammingCardSession,
    build_programming_metadata,
)
from src.card.render.budget import RenderBudget
from src.card.render.renderer import render_card
from src.card.session import CardSession
from src.card.session.config import SessionConfig
from src.card.state.models import (
    CardMetadata,
    CardState,
    HeaderState,
    ReasoningBlock,
    TaskListBlock,
    TextBlock,
    ToolBlock,
)
from src.card.state.runtime_stats import RuntimeStats


class _CardClient:
    def __init__(self) -> None:
        self.created = 0
        self.updated_cards: list[dict] = []

    def create_card(
        self,
        chat_id,
        card_json,
        *,
        reply_to=None,
        reply_in_thread=None,
        idempotency_key=None,
    ):
        self.created += 1
        return f"msg-{self.created}", f"card-{self.created}"

    def update_card(self, card_id, card_json, *, sequence=0) -> None:
        self.updated_cards.append(card_json)

    def update_element(self, card_id, element_id, content, *, sequence=0) -> None:
        return None


class _ProcessSink:
    def __init__(self, *, visible_check=None, start_error: Exception | None = None) -> None:
        self.started = False
        self.healthy = True
        self.message_id = "cot-message"
        self.events: list[CardEvent] = []
        self.completed: list[CardEvent] = []
        self.aborted = False
        self.abort_calls = 0
        self.visible_check = visible_check
        self.start_error = start_error
        self.fail_mode: str | None = None
        self.complete_mode: str | None = None

    def start(self) -> None:
        if self.visible_check is not None:
            self.visible_check()
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def emit(self, event: CardEvent) -> bool:
        if self.fail_mode == "return_false":
            return False
        self.events.append(event)
        if self.fail_mode == "async_unhealthy":
            self.healthy = False
        return True

    def complete(self, event: CardEvent) -> bool:
        self.completed.append(event)
        if self.complete_mode == "raise":
            raise RuntimeError("COT completion failed")
        if self.complete_mode == "return_false":
            return False
        return self.healthy

    def abort(self) -> None:
        self.abort_calls += 1
        self.aborted = True
        self.started = False


def _programming_session(
    *,
    process_sink: _ProcessSink | None = None,
) -> tuple[_CardClient, CardSession, ProgrammingCardSession]:
    client = _CardClient()
    metadata = build_programming_metadata("codex")
    card_session = CardSession(
        chat_id="chat-programming",
        config=SessionConfig(
            metadata=metadata,
            reply_to="origin-message",
            sync_delivery=True,
        ),
        delivery=CardDelivery(client),
        session_id="programming-adapter-test",
    )
    programming = ProgrammingCardSession(
        card_session,
        base_metadata=metadata,
        process_sink=process_sink,
    )
    return client, card_session, programming


def _tool_event(
    event_type: ACPEventType,
    *,
    status: str,
    content: str,
) -> ACPEvent:
    return ACPEvent(
        event_type=event_type,
        tool_call=ToolCallInfo(
            id="tool-1",
            title="Read",
            kind="read",
            status=status,
            content=content,
        ),
    )


def test_programming_direct_mode_omits_redundant_phase_banner():
    state = CardState(
        metadata=CardMetadata(
            mode_name="Programming",
            mode_emoji="💬",
            engine_type=None,
            tool_name="Coco",
        ),
        header=HeaderState(title="Programming"),
        blocks=(),
        terminal="running",
    )
    object.__setattr__(state, "runtime_stats", RuntimeStats(elapsed_seconds=32.0))

    pages = render_card(state, RenderBudget())

    assert len(pages) == 1
    body = pages[0]._card_json["body"]["elements"]
    body_text = str(body)
    assert "Programming · Coco · 进行中" not in body_text
    assert "Coco · 进行中" not in body_text
    panel_titles = [
        element.get("header", {}).get("title", {}).get("content", "")
        for element in body
        if element.get("tag") == "collapsible_panel"
    ]
    assert not any("任务列表" in title for title in panel_titles)


def test_cot_keeps_plan_on_card_and_publishes_only_terminal_conclusion() -> None:
    sink = _ProcessSink()
    client, card_session, programming = _programming_session(process_sink=sink)

    def assert_card_is_visible() -> None:
        assert client.created == 1
        assert card_session.state is not None
        assert card_session.state.terminal == "running"

    sink.visible_check = assert_card_is_visible

    try:
        programming.start()

        assert client.created == 1
        assert card_session.state is not None
        assert not any(
            isinstance(block, (TextBlock, ReasoningBlock, ToolBlock))
            for block in card_session.state.blocks
        )
        assert programming.activate_process_sink() is True

        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.PLAN_UPDATE,
                plan=PlanInfo(
                    entries=[
                        PlanEntryInfo(content="检查实现", status="completed"),
                        PlanEntryInfo(content="补回归测试", status="in_progress"),
                    ]
                ),
            )
        )
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="中间分析")
        )
        programming._flush_now()
        programming.on_event(
            ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="检查边界")
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content="src/card/programming_adapter.py",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content="读取完成",
            )
        )
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="最终结论")
        )
        programming._flush_now()

        assert card_session.state is not None
        assert any(isinstance(block, TaskListBlock) for block in card_session.state.blocks)
        assert not any(
            isinstance(block, (TextBlock, ReasoningBlock, ToolBlock))
            for block in card_session.state.blocks
        )
        assert {
            CardEventType.TEXT_DELTA,
            CardEventType.REASONING_DELTA,
            CardEventType.TOOL_STARTED,
            CardEventType.TOOL_DONE,
        }.issubset({event.type for event in sink.events})

        programming.finish()

        assert card_session.state is not None
        text_blocks = [
            block for block in card_session.state.blocks if isinstance(block, TextBlock)
        ]
        assert [(block.block_id, block.content) for block in text_blocks] == [
            ("_final_answer", "最终结论")
        ]
        assert not any(
            isinstance(block, (ReasoningBlock, ToolBlock))
            for block in card_session.state.blocks
        )
        assert card_session.state.terminal == "completed"
        assert [event.type for event in sink.completed] == [CardEventType.COMPLETED]
    finally:
        programming.abort()


def test_cot_start_failure_restores_complete_legacy_card_projection() -> None:
    sink = _ProcessSink(start_error=RuntimeError("COT unavailable"))
    client, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is False

        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="卡片正文")
        )
        programming._flush_now()
        programming.on_event(
            ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="卡片推理")
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content="src/main.py",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content="done",
            )
        )
        programming.finish()

        assert sink.aborted is True
        assert sink.events == []
        assert sink.completed == []
        assert card_session.state is not None
        assert card_session.state.terminal == "completed"
        assert card_session.state.engine_ext is None
        assert sink.abort_calls == 1
        assert any(
            "COT 过程通道暂不可用；完整过程已切换到主卡，"
            "任务继续执行，无需重试。" in str(card)
            for card in client.updated_cards
        )
        assert any(
            isinstance(block, TextBlock) and block.content == "卡片正文"
            for block in card_session.state.blocks
        )
        assert any(
            isinstance(block, ReasoningBlock) and block.content == "卡片推理"
            for block in card_session.state.blocks
        )
        assert any(
            isinstance(block, ToolBlock)
            and block.tool_input == "src/main.py"
            and block.tool_output == "done"
            and block.status == "completed"
            for block in card_session.state.blocks
        )
    finally:
        programming.abort()


def test_cot_activation_always_aborts_sink_if_cutover_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _ProcessSink(start_error=RuntimeError("COT unavailable"))
    _, _, programming = _programming_session(process_sink=sink)

    def fail_cutover(*, reason: str) -> bool:
        assert reason == "activation_failed"
        raise RuntimeError("card cutover failed")

    try:
        programming.start()
        monkeypatch.setattr(
            programming,
            "_cut_over_to_card_locked",
            fail_cutover,
        )

        with pytest.raises(RuntimeError, match="card cutover failed"):
            programming.activate_process_sink()

        assert sink.abort_calls == 1
        assert sink.aborted is True
    finally:
        programming.abort()


@pytest.mark.parametrize("complete_mode", ["return_false", "raise"])
def test_cot_complete_failure_replays_full_process_once(
    complete_mode: str,
) -> None:
    sink = _ProcessSink()
    sink.complete_mode = complete_mode
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True

        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="前置分析")
        )
        programming._flush_now()
        programming.on_event(
            ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="检查风险")
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content="src/contract.py",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content="工具结果",
            )
        )
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="最终结论")
        )
        programming._flush_now()

        assert card_session.state is not None
        assert not any(
            isinstance(block, (TextBlock, ReasoningBlock, ToolBlock))
            for block in card_session.state.blocks
        )

        programming.finish()

        assert sink.aborted is True
        assert [event.type for event in sink.completed] == [CardEventType.COMPLETED]
        assert card_session.state is not None
        assert card_session.state.terminal == "completed"
        process_blocks = [
            block
            for block in card_session.state.blocks
            if isinstance(block, (TextBlock, ReasoningBlock, ToolBlock))
        ]
        assert [block.block_id for block in process_blocks] == [
            "_active_text",
            "_active_reasoning",
            "tool-1",
            "_turn_2_text",
        ]
        assert [
            block.content for block in process_blocks if isinstance(block, TextBlock)
        ] == ["前置分析", "最终结论"]
        assert sum(
            isinstance(block, TextBlock) and block.content == "最终结论"
            for block in process_blocks
        ) == 1
        assert not any(
            isinstance(block, TextBlock) and block.block_id == "_final_answer"
            for block in process_blocks
        )
        assert isinstance(process_blocks[1], ReasoningBlock)
        assert process_blocks[1].content == "检查风险"
        assert isinstance(process_blocks[2], ToolBlock)
        assert process_blocks[2].tool_input == "src/contract.py"
        assert process_blocks[2].tool_output == "工具结果"
        assert process_blocks[2].status == "completed"
    finally:
        programming.abort()


def test_cot_success_keeps_parallel_agent_summaries_on_card() -> None:
    sink = _ProcessSink()
    _, card_session, programming = _programming_session(process_sink=sink)

    def agent_event(
        event_type: ACPEventType,
        *,
        call_id: str,
        label: str,
        status: str,
    ) -> ACPEvent:
        return ACPEvent(
            event_type=event_type,
            tool_call=ToolCallInfo(
                id=call_id,
                title="agent",
                kind="other",
                status=status,
                content=f"{label}\n子代理：Explore",
            ),
        )

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.on_event(
            agent_event(
                ACPEventType.TOOL_CALL_START,
                call_id="agent-a",
                label="检查事件映射",
                status="in_progress",
            )
        )
        programming.on_event(
            agent_event(
                ACPEventType.TOOL_CALL_START,
                call_id="agent-b",
                label="检查降级路径",
                status="in_progress",
            )
        )

        assert card_session.state is not None
        assert len(card_session.state.metadata.subagents) == 2
        assert {item["status"] for item in card_session.state.metadata.subagents} == {
            "running"
        }
        assert not any(event.type in {
            CardEventType.TOOL_STARTED,
            CardEventType.TOOL_DELTA,
            CardEventType.TOOL_DONE,
            CardEventType.TOOL_FAILED,
        } for event in sink.events)

        programming.on_event(
            agent_event(
                ACPEventType.TOOL_CALL_DONE,
                call_id="agent-a",
                label="检查事件映射",
                status="completed",
            )
        )
        programming.on_event(
            agent_event(
                ACPEventType.TOOL_CALL_DONE,
                call_id="agent-b",
                label="检查降级路径",
                status="completed",
            )
        )
        programming.on_text("并行检查完成")
        programming._flush_now()
        programming.finish()

        assert card_session.state is not None
        assert len(card_session.state.metadata.subagents) == 2
        assert {item["status"] for item in card_session.state.metadata.subagents} == {
            "completed"
        }
        assert {item["label"] for item in card_session.state.metadata.subagents} == {
            "检查事件映射",
            "检查降级路径",
        }
        assert [event.type for event in sink.completed] == [CardEventType.COMPLETED]
        assert any(
            isinstance(block, TextBlock) and block.content == "并行检查完成"
            for block in card_session.state.blocks
        )
    finally:
        programming.abort()


@pytest.mark.parametrize("failure_mode", ["return_false", "async_unhealthy"])
def test_cot_delivery_loss_atomically_replays_without_losing_event_shape(
    failure_mode: str,
) -> None:
    sink = _ProcessSink()
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True

        programming.dispatch(
            CardEvent.text_started(
                "child-text",
                source_kind="subagent",
                source_sequence="2.a",
                source_label="Reviewer",
                source_ref="opaque-child-ref",
            )
        )
        programming.dispatch(
            CardEvent.tool_started("tool-buffered", "Read", "src/contract.py")
        )
        programming.dispatch(CardEvent.tool_delta("tool-buffered", "partial"))
        programming.dispatch(
            CardEvent(
                type=CardEventType.TOOL_DONE,
                payload={
                    "block_id": "tool-buffered",
                    "tool_output": "complete output",
                    "tool_summary": "Read contract",
                },
            )
        )
        sink.fail_mode = failure_mode

        programming.dispatch(CardEvent.text_delta("child-text", "exactly once"))

        assert sink.aborted is True
        assert sink.abort_calls == 1
        assert card_session.state is not None
        assert card_session.state.terminal == "running"
        assert card_session.state.engine_ext is None
        assert card_session.state.footer.warning_banner == (
            "COT 过程通道暂不可用；完整过程已切换到主卡，"
            "任务继续执行，无需重试。"
        )
        assert card_session.state.footer.warning_type == "info"
        rendered = render_card(card_session.state, RenderBudget())
        rendered_payload = str(rendered[0]._card_json)
        assert rendered_payload.count(card_session.state.footer.warning_banner) == 1
        assert "ℹ️" in rendered_payload
        child = next(
            block
            for block in card_session.state.blocks
            if isinstance(block, TextBlock) and block.block_id == "child-text"
        )
        assert child.content == "exactly once"
        assert child.source_kind == "subagent"
        assert child.source_sequence == "2.a"
        assert child.source_label == "Reviewer"
        assert child.source_ref == "opaque-child-ref"

        tool = next(
            block
            for block in card_session.state.blocks
            if isinstance(block, ToolBlock) and block.block_id == "tool-buffered"
        )
        assert tool.tool_name == "Read"
        assert tool.tool_input == "src/contract.py"
        assert tool.tool_output == "complete output"
        assert tool.tool_summary == "Read contract"
        assert tool.status == "completed"

        programming.dispatch(CardEvent.text_done("child-text"))
        assert sink.abort_calls == 1
        assert programming._process_cutover_announced is True
    finally:
        programming.abort()


def test_cot_degraded_banner_is_cleared_when_parent_is_cancelled() -> None:
    sink = _ProcessSink()
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.dispatch(CardEvent.text_started("cancel-after-cutover"))
        sink.fail_mode = "return_false"
        programming.dispatch(
            CardEvent.text_delta("cancel-after-cutover", "过程仍在主卡")
        )

        assert card_session.state is not None
        assert "任务继续执行" in str(card_session.state.footer.warning_banner)

        programming.cancel()

        assert card_session.state is not None
        assert card_session.state.terminal == "cancelled"
        assert card_session.state.footer.warning_banner is None
        assert card_session.state.footer.warning_type is None
        rendered = render_card(card_session.state, RenderBudget())
        assert "任务继续执行" not in str(rendered[0]._card_json)
    finally:
        programming.abort()


def test_cot_replay_buffer_hard_limit_cuts_over_before_accepting_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.card.programming_adapter._PROCESS_REPLAY_MAX_EVENTS", 2)
    sink = _ProcessSink()
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.dispatch(CardEvent.text_started("bounded-text"))
        programming.dispatch(CardEvent.text_delta("bounded-text", "a"))

        # The third atom is not offered to COT: it forces an atomic replay and
        # is projected once into the authoritative card.
        programming.dispatch(CardEvent.text_delta("bounded-text", "b"))
        programming.dispatch(CardEvent.text_delta("bounded-text", "c"))

        assert sink.aborted is True
        assert sink.abort_calls == 1
        assert [event.type for event in sink.events] == [
            CardEventType.TEXT_STARTED,
            CardEventType.TEXT_DELTA,
        ]
        assert card_session.state is not None
        assert card_session.state.terminal == "running"
        assert card_session.state.footer.warning_type == "info"
        block = next(
            block
            for block in card_session.state.blocks
            if isinstance(block, TextBlock) and block.block_id == "bounded-text"
        )
        assert block.content == "abc"
    finally:
        programming.abort()


def test_cot_replay_buffer_byte_limit_cuts_over_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = CardEvent.text_started("byte-bounded-text")
    monkeypatch.setattr(
        "src.card.programming_adapter._PROCESS_REPLAY_MAX_BYTES",
        ProgrammingCardSession._process_event_size(start) + 4,
    )
    sink = _ProcessSink()
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.dispatch(start)
        programming.dispatch(
            CardEvent.text_delta("byte-bounded-text", "完整回放内容")
        )

        assert sink.aborted is True
        assert sink.abort_calls == 1
        assert sink.events == [start]
        assert card_session.state is not None
        assert card_session.state.terminal == "running"
        assert card_session.state.footer.warning_type == "info"
        block = next(
            block
            for block in card_session.state.blocks
            if isinstance(block, TextBlock)
            and block.block_id == "byte-bounded-text"
        )
        assert block.content == "完整回放内容"
    finally:
        programming.abort()


def test_terminal_fence_drops_late_text_and_tool_events() -> None:
    sink = _ProcessSink()
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.on_text("最终答案")
        programming._flush_now()
        programming.finish()

        assert card_session.state is not None
        blocks_at_terminal = card_session.state.blocks
        sink_events_at_terminal = tuple(sink.events)

        programming.on_text("迟到正文")
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="迟到文本事件")
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content="late input",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content="late output",
            )
        )

        assert card_session.state.blocks == blocks_at_terminal
        assert tuple(sink.events) == sink_events_at_terminal
        assert [event.type for event in sink.completed] == [CardEventType.COMPLETED]
        assert not any("迟到" in getattr(block, "content", "") for block in blocks_at_terminal)
        assert not any(
            isinstance(block, ToolBlock) and block.tool_input == "late input"
            for block in blocks_at_terminal
        )
    finally:
        programming.abort()
