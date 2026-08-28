"""programming_adapter SectionLayout integration tests."""
from __future__ import annotations

import json
import threading

import pytest
from lark_oapi.core.model.base_response import BaseResponse
from lark_oapi.core.model.raw_response import RawResponse

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
from src.card.protocols import ProcessSegmentRollover
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
from src.feishu.cot import FeishuCOTAPIClient, FeishuCOTStream


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
        self.rollover_mode: str | None = None
        self.rollover_calls = 0

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

    def rollover(self) -> ProcessSegmentRollover:
        self.rollover_calls += 1
        if self.rollover_mode == "success":
            return ProcessSegmentRollover(sealed=True, started=True)
        if self.rollover_mode == "success_with_anchor":
            anchor = next(
                event
                for event in reversed(self.events)
                if event.type is CardEventType.TEXT_STARTED
            )
            return ProcessSegmentRollover(
                sealed=True,
                started=True,
                replay_events=(anchor,),
            )
        return ProcessSegmentRollover(sealed=False, started=False)

    def abort(self) -> None:
        self.abort_calls += 1
        self.aborted = True
        self.started = False


class _COTRecordingClient:
    """Return valid COT API responses while retaining the real wire requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: list[object] = []

    def request(self, request: object) -> BaseResponse:
        with self._lock:
            self.requests.append(request)
        uri = getattr(request, "uri", "")
        body = getattr(request, "body", None)
        data: dict[str, str] = {}
        if (
            uri == "/open-apis/im/v1/message_cot"
            and isinstance(body, dict)
            and "receive_id" in body
        ):
            data = {
                "cot_id": "cot_adapter_lifecycle",
                "message_id": "om_adapter_lifecycle",
            }
        response = BaseResponse()
        response.code = 0
        response.raw = RawResponse()
        response.raw.status_code = 200
        response.raw.content = json.dumps(
            {"code": 0, "msg": "success", "data": data},
        ).encode("utf-8")
        return response


def _cot_wire_event_types(client: _COTRecordingClient) -> list[str]:
    event_types: list[str] = []
    for request in client.requests:
        body = getattr(request, "body", None)
        if not isinstance(body, dict):
            continue
        events = body.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict) and isinstance(event.get("event_type"), str):
                event_types.append(event["event_type"])
    return event_types


def _cot_wire_contents(client: _COTRecordingClient) -> list[dict[str, object]]:
    contents: list[dict[str, object]] = []
    for request in client.requests:
        body = getattr(request, "body", None)
        if not isinstance(body, dict):
            continue
        events = body.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            raw_content = event.get("content")
            if not isinstance(raw_content, str):
                continue
            content = json.loads(raw_content)
            if isinstance(content, dict):
                contents.append(content)
    return contents


def _cot_wire_events(
    client: _COTRecordingClient,
) -> list[tuple[str, dict[str, object]]]:
    events_with_contents: list[tuple[str, dict[str, object]]] = []
    for request in client.requests:
        body = getattr(request, "body", None)
        if not isinstance(body, dict):
            continue
        events = body.get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type")
            raw_content = event.get("content")
            if not isinstance(event_type, str) or not isinstance(raw_content, str):
                continue
            content = json.loads(raw_content)
            if isinstance(content, dict):
                events_with_contents.append((event_type, content))
    return events_with_contents


def _cot_request_json(client: _COTRecordingClient) -> str:
    return "\n".join(
        json.dumps(
            getattr(request, "body", None),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for request in client.requests
    )


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
    tool_id: str = "tool-1",
    title: str = "Read",
) -> ACPEvent:
    return ACPEvent(
        event_type=event_type,
        tool_call=ToolCallInfo(
            id=tool_id,
            title=title,
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


def test_cot_normalizes_missing_tool_start_without_cutting_over_main_card() -> None:
    """A summarized child tool must not make native COT abandon its process view."""
    cot_client = _COTRecordingClient()
    sink = FeishuCOTStream(
        FeishuCOTAPIClient(cot_client, timeout_seconds=0.2),
        chat_id="oc_adapter_cot",
        origin_message_id="om_adapter_origin",
        input_text="验证缺失工具开始帧",
        detail="detailed",
        flush_interval=0.005,
        request_timeout=0.2,
    )
    client, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True

        # Some providers summarize away TOOL_STARTED, then resume the normal
        # process feed at UPDATE/DONE.  The adapter must synthesize the missing
        # start before handing either atom to the strict native COT lifecycle.
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_UPDATE,
                status="in_progress",
                content="正在读取契约",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content="契约读取完成",
            )
        )

        assert sink.started is True
        assert sink.healthy is True
        assert programming._native_process_started is True
        assert programming._process_cutover_announced is False
        assert card_session.state is not None
        assert card_session.state.footer.warning_banner is None
        assert not any(
            "完整过程已切换到主卡" in str(card)
            for card in client.updated_cards
        )

        programming.on_text("后续过程仍由 COT 记录")
        programming._flush_now()
        programming.finish()

        assert card_session.state is not None
        assert card_session.state.terminal == "completed"
        assert card_session.state.footer.warning_banner is None
        assert any(
            isinstance(block, TextBlock)
            and block.block_id == "_final_answer"
            and block.content == "后续过程仍由 COT 记录"
            for block in card_session.state.blocks
        )

        wire_types = _cot_wire_event_types(cot_client)
        tool_start = wire_types.index("TOOL_CALL_START")
        tool_result = wire_types.index("TOOL_CALL_RESULT")
        assert wire_types[tool_start:tool_result + 1] == [
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "TOOL_CALL_RESULT",
        ]
        assert "TEXT_MESSAGE_CONTENT" in wire_types
        assert any(
            content.get("status") == "done"
            for content in _cot_wire_contents(cot_client)
        )
    finally:
        programming.abort()


def test_card_keeps_reused_provider_tool_ids_as_distinct_tool_blocks() -> None:
    """A provider's recycled raw ID still identifies two logical card tools."""
    raw_tool_id = "provider-raw-tool-id-9b3c"
    _, card_session, programming = _programming_session()

    try:
        programming.start()
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content=f"first-input marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read first",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content=f"first-result marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read first",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content=f"second-input marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read second",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_UPDATE,
                status="in_progress",
                content=f"second-update marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read second",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content=f"second-result marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read second",
            )
        )
        programming.finish()

        assert card_session.state is not None
        assert card_session.state.terminal == "completed"
        tool_blocks = [
            block
            for block in card_session.state.blocks
            if isinstance(block, ToolBlock)
        ]
        assert len(tool_blocks) == 2
        assert len({block.block_id for block in tool_blocks}) == 2
        assert raw_tool_id not in {block.block_id for block in tool_blocks}
        assert [block.tool_name for block in tool_blocks] == [
            "Read first",
            "Read second",
        ]
        assert all(block.status == "completed" for block in tool_blocks)
        assert "first-result marker" in str(tool_blocks[0].tool_output)
        assert "second-result marker" in str(tool_blocks[1].tool_output)
    finally:
        programming.abort()


def test_native_cot_rekeys_reused_provider_tool_ids_without_cutover_or_leak() -> None:
    """Native COT must isolate recycled provider IDs and redact their raw form."""
    raw_tool_id = "provider-raw-tool-id-9b3c"
    cot_client = _COTRecordingClient()
    sink = FeishuCOTStream(
        FeishuCOTAPIClient(cot_client, timeout_seconds=0.2),
        chat_id="oc_adapter_reused_tool",
        origin_message_id="om_adapter_reused_tool",
        input_text="验证重复工具标识",
        detail="detailed",
        flush_interval=0.005,
        request_timeout=0.2,
    )
    client, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content=f"first-input marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title=f"Read first {raw_tool_id}",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content=f"first-result marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title=f"Read first {raw_tool_id}",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content=f"second-input marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title=f"Read second {raw_tool_id}",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_UPDATE,
                status="in_progress",
                content=f"second-update marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title=f"Read second {raw_tool_id}",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content=f"second-result marker raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title=f"Read second {raw_tool_id}",
            )
        )

        assert sink.started is True
        assert sink.healthy is True
        assert programming._native_process_started is True
        assert programming._process_cutover_announced is False
        assert card_session.state is not None
        assert card_session.state.footer.warning_banner is None
        assert not any(
            "完整过程已切换到主卡" in str(card)
            for card in client.updated_cards
        )

        programming.finish()

        assert card_session.state is not None
        assert card_session.state.terminal == "completed"
        assert card_session.state.footer.warning_banner is None
        wire_events = _cot_wire_events(cot_client)
        start_ids: list[str] = []
        result_ids: list[str] = []
        for event_type, content in wire_events:
            if event_type == "TOOL_CALL_START":
                tool_call_id = content.get("toolCallId")
                assert isinstance(tool_call_id, str)
                start_ids.append(tool_call_id)
            if event_type == "TOOL_CALL_RESULT":
                tool_call_id = content.get("toolCallId")
                assert isinstance(tool_call_id, str)
                result_ids.append(tool_call_id)

        assert len(start_ids) == 2
        assert len(set(start_ids)) == 2
        assert len(result_ids) == 2
        assert set(result_ids) == set(start_ids)
        assert raw_tool_id not in _cot_request_json(cot_client)
        assert any(
            content.get("status") == "done"
            for content in _cot_wire_contents(cot_client)
        )
    finally:
        programming.abort()


def test_native_cot_ignores_late_frames_after_a_closed_tool_invocation() -> None:
    """DONE/DONE/UPDATE after one invocation must not create a phantom result."""
    raw_tool_id = "provider-raw-tool-id-late-4e2a"
    cot_client = _COTRecordingClient()
    sink = FeishuCOTStream(
        FeishuCOTAPIClient(cot_client, timeout_seconds=0.2),
        chat_id="oc_adapter_late_tool",
        origin_message_id="om_adapter_late_tool",
        input_text="验证迟到工具帧",
        detail="detailed",
        flush_interval=0.005,
        request_timeout=0.2,
    )
    client, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_START,
                status="in_progress",
                content=f"initial-input raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read once",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content=f"initial-result raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read once",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_DONE,
                status="completed",
                content=f"late-terminal raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read once",
            )
        )
        programming.on_event(
            _tool_event(
                ACPEventType.TOOL_CALL_UPDATE,
                status="in_progress",
                content=f"late-update raw={raw_tool_id}",
                tool_id=raw_tool_id,
                title="Read once",
            )
        )

        assert sink.started is True
        assert sink.healthy is True
        assert programming._process_cutover_announced is False
        assert card_session.state is not None
        assert card_session.state.footer.warning_banner is None
        assert not any(
            "完整过程已切换到主卡" in str(card)
            for card in client.updated_cards
        )

        programming.finish()

        wire_types = _cot_wire_event_types(cot_client)
        assert wire_types.count("TOOL_CALL_START") == 1
        assert wire_types.count("TOOL_CALL_RESULT") == 1
        assert raw_tool_id not in _cot_request_json(cot_client)
    finally:
        programming.abort()


def test_cot_reopens_reasoning_after_text_without_cutting_over_main_card() -> None:
    """Text between thoughts must start a fresh strict-COT reasoning lifecycle."""
    cot_client = _COTRecordingClient()
    sink = FeishuCOTStream(
        FeishuCOTAPIClient(cot_client, timeout_seconds=0.2),
        chat_id="oc_adapter_reasoning",
        origin_message_id="om_adapter_reasoning",
        input_text="验证推理续接",
        detail="detailed",
        flush_interval=0.005,
        request_timeout=0.2,
    )
    client, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True

        programming.on_event(
            ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="先检查边界")
        )
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="中间正文")
        )
        programming.on_event(
            ACPEvent(event_type=ACPEventType.THOUGHT_CHUNK, text="再检查终态")
        )

        assert sink.started is True
        assert sink.healthy is True
        assert programming._native_process_started is True
        assert programming._process_cutover_announced is False
        assert card_session.state is not None
        assert card_session.state.footer.warning_banner is None
        assert not any(
            "完整过程已切换到主卡" in str(card)
            for card in client.updated_cards
        )

        programming.finish()

        assert card_session.state is not None
        assert card_session.state.terminal == "completed"
        assert card_session.state.footer.warning_banner is None
        assert any(
            isinstance(block, TextBlock)
            and block.block_id == "_final_answer"
            and block.content == "中间正文"
            for block in card_session.state.blocks
        )

        reasoning_lifecycle_types = {
            "REASONING_START",
            "REASONING_MESSAGE_START",
            "REASONING_MESSAGE_CONTENT",
            "REASONING_MESSAGE_END",
            "REASONING_END",
        }
        reasoning_lifecycle: list[tuple[str, str]] = []
        for event_type, content in _cot_wire_events(cot_client):
            if event_type not in reasoning_lifecycle_types:
                continue
            message_id = content.get("messageId")
            assert isinstance(message_id, str)
            reasoning_lifecycle.append((event_type, message_id))

        reasoning_start_ids = [
            message_id
            for event_type, message_id in reasoning_lifecycle
            if event_type == "REASONING_START"
        ]
        assert len(reasoning_start_ids) == 2
        assert len(set(reasoning_start_ids)) == 2
        first_reasoning_id, second_reasoning_id = reasoning_start_ids
        assert reasoning_lifecycle == [
            ("REASONING_START", first_reasoning_id),
            ("REASONING_MESSAGE_START", first_reasoning_id),
            ("REASONING_MESSAGE_CONTENT", first_reasoning_id),
            ("REASONING_MESSAGE_END", first_reasoning_id),
            ("REASONING_END", first_reasoning_id),
            ("REASONING_START", second_reasoning_id),
            ("REASONING_MESSAGE_START", second_reasoning_id),
            ("REASONING_MESSAGE_CONTENT", second_reasoning_id),
            ("REASONING_MESSAGE_END", second_reasoning_id),
            ("REASONING_END", second_reasoning_id),
        ]
        assert any(
            content.get("status") == "done"
            for content in _cot_wire_contents(cot_client)
        )
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
            "_active_tool",
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


def test_cot_replay_budget_seals_and_continues_in_a_new_process_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.card.programming_adapter._PROCESS_REPLAY_MAX_EVENTS", 2)
    sink = _ProcessSink()
    sink.rollover_mode = "success"
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.dispatch(CardEvent.text_started("segmented-text"))
        programming.dispatch(CardEvent.text_delta("segmented-text", "a"))

        # The next atom starts a fresh COT segment instead of forcing the
        # long-running task's whole process stream back into the main card.
        programming.dispatch(CardEvent.text_delta("segmented-text", "b"))

        assert sink.rollover_calls == 1
        assert sink.abort_calls == 0
        assert [event.type for event in sink.events] == [
            CardEventType.TEXT_STARTED,
            CardEventType.TEXT_DELTA,
            CardEventType.TEXT_DELTA,
        ]
        assert card_session.state is not None
        assert card_session.state.terminal == "running"
        assert card_session.state.footer.warning_banner is None
        assert not any(
            isinstance(block, TextBlock) and block.block_id == "segmented-text"
            for block in card_session.state.blocks
        )
    finally:
        programming.abort()


def test_degradation_after_rollover_replays_only_the_open_segment_with_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.card.programming_adapter._PROCESS_REPLAY_MAX_EVENTS", 3)
    sink = _ProcessSink()
    sink.rollover_mode = "success_with_anchor"
    _, card_session, programming = _programming_session(process_sink=sink)

    try:
        programming.start()
        assert programming.activate_process_sink() is True
        programming.dispatch(CardEvent.text_started("segmented-text"))
        programming.dispatch(CardEvent.text_delta("segmented-text", "old-a"))
        programming.dispatch(CardEvent.text_delta("segmented-text", "old-b"))
        programming.dispatch(CardEvent.text_delta("segmented-text", "new-c"))

        sink.fail_mode = "async_unhealthy"
        programming.dispatch(CardEvent.text_delta("segmented-text", "new-d"))

        assert sink.rollover_calls == 1
        assert sink.abort_calls == 1
        assert card_session.state is not None
        assert card_session.state.footer.warning_type == "info"
        assert "已完成的过程分段仍保留" in str(
            card_session.state.footer.warning_banner
        )
        block = next(
            block
            for block in card_session.state.blocks
            if isinstance(block, TextBlock) and block.block_id == "segmented-text"
        )
        assert block.content == "new-cnew-d"
        assert "old-a" not in block.content
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
