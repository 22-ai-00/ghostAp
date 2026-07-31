from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import SimpleNamespace

import pytest
from lark_channel import OutboundCard, SendError, SendResult
from lark_channel.channel.errors import FeishuChannelErrorCode

from src.acp.models import ACPEvent, ACPEventType, PlanEntryInfo, PlanInfo, ToolCallInfo
from src.card.delivery.channel_client import LarkChannelCardAPIClient
from src.card.delivery.engine import CardDelivery
from src.card.delivery.types import SequenceConflictError, TransportError
from src.card.programming_adapter import ProgrammingCardSession, build_programming_metadata
from src.card.session import CardSession
from src.card.session.config import SessionConfig
from src.card.types import RenderedCard


class _CardResource:
    def __init__(self) -> None:
        self.requests = []
        self.response_code = 0
        self.response_msg = ""

    async def aupdate(self, request):
        self.requests.append(request)
        return SimpleNamespace(code=self.response_code, msg=self.response_msg)


class _ImmediateChannel:
    def __init__(self) -> None:
        self.scheduled = 0
        self.created_specs: list[dict] = []
        self.sent: list[tuple[str, object, dict]] = []
        self.element_updates: list[tuple[str, str, str, int]] = []
        self.finishes: list[tuple[str, int]] = []
        self.send_result = SendResult.ok(message_id="om_1")
        self.card_resource = _CardResource()
        self.client = SimpleNamespace(
            cardkit=SimpleNamespace(
                v1=SimpleNamespace(card=self.card_resource),
            )
        )

    def schedule(self, coro) -> Future:
        self.scheduled += 1
        future: Future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    async def create_card_instance(self, spec: dict) -> str:
        self.created_specs.append(spec)
        return "card_1"

    async def send(self, to: str, message, opts: dict) -> SendResult:
        self.sent.append((to, message, opts))
        return self.send_result

    async def update_card_element_content(
        self,
        card_id: str,
        element_id: str,
        content: str,
        sequence: int,
    ) -> None:
        self.element_updates.append((card_id, element_id, content, sequence))

    async def finish_streaming_card(self, card_id: str, sequence: int) -> None:
        self.finishes.append((card_id, sequence))


def test_create_card_sends_complete_card_with_idempotency_key() -> None:
    channel = _ImmediateChannel()
    client = LarkChannelCardAPIClient(channel)
    payload = {"schema": "2.0", "body": {"elements": [{"tag": "markdown", "content": "hi"}]}}

    message_id, card_id = client.create_card(
        "oc_1",
        payload,
        reply_to="om_origin",
        reply_in_thread=True,
        idempotency_key="idem-1",
    )

    assert (message_id, card_id) == ("om_1", "om_1")
    to, outbound, opts = channel.sent[0]
    assert to == "oc_1"
    assert isinstance(outbound, OutboundCard)
    assert outbound.card == payload
    assert opts == {
        "receive_id_type": "chat_id",
        "reply_to": "om_origin",
        "reply_in_thread": True,
        "reply_target_gone": "fail",
        "uuid": "idem-1",
    }


def test_streaming_create_and_reference_use_channel_cardkit() -> None:
    channel = _ImmediateChannel()
    client = LarkChannelCardAPIClient(channel, default_reply_in_thread=True)
    payload = {
        "schema": "2.0",
        "config": {"streaming_mode": True},
        "body": {"elements": [{"tag": "markdown", "element_id": "main", "content": "..."}]},
    }

    card_id = client.create_streaming_card(payload)
    message_id = client.send_card_reference(
        "oc_1",
        card_id,
        reply_to="om_origin",
        idempotency_key="idem-2",
    )

    assert card_id == "card_1"
    assert message_id == "om_1"
    assert channel.created_specs == [payload]
    _to, outbound, opts = channel.sent[0]
    assert isinstance(outbound, OutboundCard)
    assert outbound.card_id == "card_1"
    assert opts["uuid"] == "idem-2"
    assert opts["reply_target_gone"] == "fail"
    assert opts["reply_in_thread"] is True


def test_structure_update_puts_complete_card_entity_with_sequence() -> None:
    channel = _ImmediateChannel()
    client = LarkChannelCardAPIClient(channel)
    payload = {"schema": "2.0", "config": {"streaming_mode": True}, "body": {"elements": []}}

    client.update_card("card_1", payload, sequence=7)

    request = channel.card_resource.requests[0]
    assert request.card_id == "card_1"
    assert request.request_body.sequence == 7
    assert request.request_body.uuid
    assert request.request_body.card.type == "card_json"
    assert json.loads(request.request_body.card.data) == payload


def test_element_update_and_finish_use_channel_streaming_methods() -> None:
    channel = _ImmediateChannel()
    client = LarkChannelCardAPIClient(channel)

    client.update_element("card_1", "main", "hello", sequence=8)
    client.finish_streaming_card("card_1", sequence=9)

    assert channel.element_updates == [("card_1", "main", "hello", 8)]
    assert channel.finishes == [("card_1", 9)]


def test_send_failure_maps_raw_code_to_transport_error() -> None:
    channel = _ImmediateChannel()
    channel.send_result = SendResult.fail(
        SendError(
            code=FeishuChannelErrorCode.TARGET_REVOKED,
            retryable=False,
            hint="message missing",
            raw_code=99992354,
        )
    )
    client = LarkChannelCardAPIClient(channel)

    with pytest.raises(TransportError) as caught:
        client.create_card("oc_1", {"body": {}})

    assert caught.value.code == 99992354
    assert caught.value.needs_recreate is True


def test_reference_failure_closes_unreferenced_stream_without_inline_fallback() -> None:
    channel = _ImmediateChannel()
    channel.send_result = SendResult.fail(
        SendError(
            code=FeishuChannelErrorCode.TARGET_REVOKED,
            retryable=False,
            hint="message missing",
            raw_code=99992354,
        )
    )
    adapter = LarkChannelCardAPIClient(channel, preallocate_cards=True)
    delivery = CardDelivery(adapter)
    rendered = [
        RenderedCard(
            _card_json={"config": {"streaming_mode": True}, "body": {}},
            structure_signature="stream",
            page_index=0,
            total_pages=1,
        )
    ]

    try:
        outcomes = delivery.deliver("failed_reference", "oc_chat", rendered)

        assert outcomes[0].kind == "reconcile"
        assert channel.finishes == [("card_1", 1)]
        assert len(channel.sent) == 1
        assert delivery.get_binding("failed_reference").pages == {}
    finally:
        delivery._shutdown()


def test_cardkit_sequence_conflict_maps_to_domain_error() -> None:
    channel = _ImmediateChannel()
    channel.card_resource.response_code = 300317
    channel.card_resource.response_msg = "sequence must increase"
    client = LarkChannelCardAPIClient(channel)

    with pytest.raises(SequenceConflictError) as caught:
        client.update_card("card_1", {"body": {}}, sequence=12)

    assert caught.value.next_floor == 13


def test_element_sequence_conflict_keeps_requested_floor() -> None:
    channel = _ImmediateChannel()

    async def fail_element(*_args, **_kwargs):
        raise RuntimeError("raw_code=300317 sequence conflict")

    channel.update_card_element_content = fail_element
    client = LarkChannelCardAPIClient(channel)

    with pytest.raises(SequenceConflictError) as caught:
        client.update_element("card_1", "main", "hello", sequence=8)

    assert caught.value.next_floor == 9


def test_finish_sequence_conflict_keeps_requested_floor() -> None:
    channel = _ImmediateChannel()

    async def fail_finish(*_args, **_kwargs):
        raise RuntimeError("raw_code=300317 sequence conflict")

    channel.finish_streaming_card = fail_finish
    client = LarkChannelCardAPIClient(channel)

    with pytest.raises(SequenceConflictError) as caught:
        client.finish_streaming_card("card_1", sequence=11)

    assert caught.value.next_floor == 12


class _TimeoutChannel(_ImmediateChannel):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    async def create_card_instance(self, spec: dict) -> str:
        await asyncio.sleep(60)
        return "late"

    def schedule(self, coro) -> Future:
        self.scheduled += 1
        channel = self

        class _NeverFuture(Future):
            def result(self, timeout=None):
                raise FutureTimeoutError()

            def cancel(self):
                coro.close()
                channel.cancelled = True
                return True

        return _NeverFuture()


def test_timeout_cancels_channel_future() -> None:
    channel = _TimeoutChannel()
    client = LarkChannelCardAPIClient(channel, timeout_seconds=0.01)

    with pytest.raises(TimeoutError, match="timed out"):
        client.create_streaming_card({"body": {}})

    assert channel.cancelled is True


def test_audit_failure_blocks_channel_schedule() -> None:
    channel = _ImmediateChannel()
    failures: list[Exception] = []
    client = LarkChannelCardAPIClient(
        channel,
        outbound_audit=lambda *_args: (_ for _ in ()).throw(OSError("audit disk")),
        outbound_audit_failure=failures.append,
        outbound_target_aliases=lambda _target: ("oc_chat", "ou_sender"),
    )

    with pytest.raises(OSError, match="audit disk"):
        client.create_card("oc_1", {"body": {}})

    assert channel.scheduled == 0
    assert len(failures) == 1


def test_streaming_audit_failure_blocks_cardkit_create() -> None:
    channel = _ImmediateChannel()
    failures: list[Exception] = []
    client = LarkChannelCardAPIClient(
        channel,
        outbound_audit=lambda *_args: (_ for _ in ()).throw(OSError("audit disk")),
        outbound_audit_failure=failures.append,
        outbound_target_aliases=lambda _target: ("oc_chat", "ou_sender"),
    )

    with pytest.raises(OSError, match="audit disk"):
        client.create_streaming_card_for_target(
            {"body": {}},
            target="om_origin",
            operation="reply",
        )

    assert channel.scheduled == 0
    assert channel.created_specs == []
    assert len(failures) == 1


def test_programming_pipeline_preserves_rich_state_and_finishes_channel_stream() -> None:
    channel = _ImmediateChannel()
    adapter = LarkChannelCardAPIClient(channel, preallocate_cards=True)
    delivery = CardDelivery(adapter)
    metadata = build_programming_metadata(
        "codex",
        tool_name="Codex CLI",
        model_name="gpt-5",
        project_name="GhostAP",
        working_dir="/repo/ghostap",
    )
    card_session = CardSession(
        chat_id="oc_chat",
        config=SessionConfig(
            metadata=metadata,
            reply_to="om_origin",
        ),
        delivery=delivery,
        session_id="channel_programming",
    )
    programming = ProgrammingCardSession(card_session, base_metadata=metadata)

    class _NoopTicker:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    programming._ticker_factory = _NoopTicker

    try:
        programming.start()
        assert programming.wait_until_visible(timeout=3.0) is True
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.PLAN_UPDATE,
                plan=PlanInfo(
                    entries=[
                        PlanEntryInfo(content="检查通信链路", status="completed"),
                        PlanEntryInfo(content="实现流式更新", status="in_progress"),
                    ]
                ),
            )
        )
        assert card_session.wait_delivery_idle(timeout=3.0) is True
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="正在分析")
        )
        programming._flush_now()
        assert card_session.wait_delivery_idle(timeout=3.0) is True
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="普通编程模式。")
        )
        programming._flush_now()
        assert card_session.wait_delivery_idle(timeout=3.0) is True
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_START,
                tool_call=ToolCallInfo(
                    id="tool-1",
                    title="Read",
                    kind="read",
                    status="in_progress",
                    content="src/feishu/ws_client.py",
                ),
            )
        )
        assert card_session.wait_delivery_idle(timeout=3.0) is True
        programming.on_event(
            ACPEvent(
                event_type=ACPEventType.TOOL_CALL_DONE,
                tool_call=ToolCallInfo(
                    id="tool-1",
                    title="Read",
                    kind="read",
                    status="completed",
                    content="读取完成",
                ),
            )
        )
        assert card_session.wait_delivery_idle(timeout=3.0) is True
        programming.on_event(
            ACPEvent(event_type=ACPEventType.TEXT_CHUNK, text="流式更新已完成。")
        )
        binding = delivery.get_binding("channel_programming")
        assert binding is not None
        assert binding.pages[0].is_streaming is True
        programming.finish()
        assert card_session.wait_delivery_idle(timeout=3.0) is True

        entity_payloads = [
            json.loads(request.request_body.card.data)
            for request in channel.card_resource.requests
        ]
        rendered_history = json.dumps(
            [*channel.created_specs, *entity_payloads],
            ensure_ascii=False,
        )

        assert "GhostAP" in rendered_history
        assert "Codex CLI" in rendered_history
        assert "gpt-5" in rendered_history
        assert "检查通信链路" in rendered_history
        assert "实现流式更新" in rendered_history
        assert "Read" in rendered_history
        assert "正在分析普通编程模式。" in rendered_history
        assert "流式更新已完成。" in rendered_history
        assert "⏱ 用时 00:00:00" in rendered_history
        assert channel.element_updates
        assert channel.finishes

        all_sequences = [
            request.request_body.sequence for request in channel.card_resource.requests
        ]
        all_sequences.extend(update[3] for update in channel.element_updates)
        all_sequences.extend(sequence for _card_id, sequence in channel.finishes)
        assert len(all_sequences) == len(set(all_sequences))
        assert sorted(all_sequences) == list(range(1, max(all_sequences) + 1))
        assert entity_payloads[-1]["config"]["streaming_mode"] is True
        assert channel.finishes[-1][0] == "card_1"
    finally:
        delivery._shutdown()
