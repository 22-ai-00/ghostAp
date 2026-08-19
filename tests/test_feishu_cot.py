from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from lark_oapi.core.enum import AccessTokenType, HttpMethod
from lark_oapi.core.model.base_request import BaseRequest
from lark_oapi.core.model.base_response import BaseResponse
from lark_oapi.core.model.raw_response import RawResponse

from src.card.events import CardEvent, CardEventType
from src.feishu.cot import (
    COTTransportError,
    FeishuCOTAPIClient,
    FeishuCOTSession,
)

_CHAT_ID = "oc_cot_chat"
_ORIGIN_ID = "om_cot_origin"
_MESSAGE_ID = "om_cot_message"
_COT_ID = "cot_task_1"
_EMPTY = object()


def _response(
    data: object = _EMPTY,
    *,
    status_code: int = 200,
    code: object = 0,
    sdk_code: object = 0,
    raw_content: bytes | None = None,
) -> BaseResponse:
    payload: dict[str, object] = {"code": code, "msg": "success"}
    if data is not _EMPTY:
        payload["data"] = data
    response = BaseResponse()
    response.code = sdk_code
    response.raw = RawResponse()
    response.raw.status_code = status_code
    response.raw.content = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if raw_content is None
        else raw_content
    )
    return response


def _create_response() -> BaseResponse:
    return _response({"cot_id": _COT_ID, "message_id": _MESSAGE_ID})


def _empty_response() -> BaseResponse:
    return _response({})


class _RecordingClient:
    def __init__(self, *outcomes: object) -> None:
        self._outcomes = deque(outcomes)
        self._lock = threading.Lock()
        self.requests: list[BaseRequest] = []

    def request(self, request: BaseRequest) -> object:
        with self._lock:
            self.requests.append(request)
            if not self._outcomes:
                raise AssertionError("unexpected COT request")
            outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(request)
        return outcome


def _api(
    client: _RecordingClient,
    **kwargs: object,
) -> FeishuCOTAPIClient:
    return FeishuCOTAPIClient(
        client,
        timeout_seconds=kwargs.pop("timeout_seconds", 0.2),
        **kwargs,
    )


def _event_content(wire_event: dict[str, object]) -> dict[str, object]:
    content = wire_event["content"]
    assert isinstance(content, str)
    decoded = json.loads(content)
    assert isinstance(decoded, dict)
    return decoded


def _request_events(request: BaseRequest) -> list[dict[str, object]]:
    assert isinstance(request.body, dict)
    events = request.body["events"]
    assert isinstance(events, list)
    return events


def _wait_until(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def test_api_request_shapes_double_serialization_audit_and_revision_fence() -> None:
    client = _RecordingClient(_create_response(), _empty_response(), _empty_response())
    audit_calls: list[tuple[str, str, str]] = []
    revision_calls: list[str] = []

    def revision(chat_id: str) -> tuple[int, int]:
        revision_calls.append(chat_id)
        return (7, 11)

    api = _api(
        client,
        outbound_audit=lambda tenant, operation, target: audit_calls.append(
            (tenant, operation, target)
        ),
        tenant_key_resolver=lambda: "tenant-a",
        outbound_target_aliases=lambda _target: (_CHAT_ID, "om_root"),
        trust_revision_provider=revision,
    )
    binding = api.create(
        _CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        reply_in_thread=True,
    )
    api.append(
        binding,
        (
            {
                "event_type": "TEXT_MESSAGE_CONTENT",
                "content": {"messageId": "text_1", "delta": "hello\ud800"},
                "timestamp": 1234,
            },
        ),
    )
    api.complete(binding, reason="done")

    create, update, complete = client.requests
    assert create.http_method is HttpMethod.POST
    assert create.uri == "/open-apis/im/v1/message_cot"
    assert create.queries == [("receive_id_type", "chat_id")]
    assert create.paths == {}
    assert create.headers == {"Content-Type": "application/json; charset=utf-8"}
    assert create.token_types == {AccessTokenType.TENANT}
    assert create.body == {
        "receive_id": _CHAT_ID,
        "origin_message_id": _ORIGIN_ID,
        "reply_in_thread": True,
    }

    assert update.http_method is HttpMethod.PUT
    assert update.uri == "/open-apis/im/v1/message_cot"
    assert update.queries == []
    assert update.headers == {"Content-Type": "application/json; charset=utf-8"}
    assert update.token_types == {AccessTokenType.TENANT}
    assert update.body["message_id"] == _MESSAGE_ID
    assert update.body["cot_id"] == _COT_ID
    wire_event = _request_events(update)[0]
    assert wire_event["event_type"] == "TEXT_MESSAGE_CONTENT"
    assert wire_event["timestamp"] == 1234
    assert isinstance(wire_event["content"], str)
    assert _event_content(wire_event) == {
        "messageId": "text_1",
        "delta": "hello�",
    }

    assert complete.http_method is HttpMethod.POST
    assert complete.uri == "/open-apis/im/v1/message_cot/complete/:cot_id"
    assert complete.paths == {"cot_id": _COT_ID}
    assert complete.queries == [
        ("message_id", _MESSAGE_ID),
        ("reason", "done"),
    ]
    assert complete.headers == {"Content-Type": "application/json; charset=utf-8"}
    assert complete.token_types == {AccessTokenType.TENANT}
    assert complete.body is None

    create_scope = [_ORIGIN_ID, _CHAT_ID, "om_root"]
    patch_scope = [_MESSAGE_ID, _COT_ID, *create_scope]
    assert audit_calls == [
        *(("tenant-a", "reply", target) for target in create_scope),
        *(("tenant-a", "patch", target) for target in patch_scope),
        *(("tenant-a", "patch", target) for target in patch_scope),
    ]
    assert revision_calls == [_CHAT_ID] * 6
    assert binding.audit_targets == tuple(create_scope)
    assert binding.trust_revision == (7, 11)


@pytest.mark.parametrize(
    "outcome",
    [
        SimpleNamespace(raw=None, code=0),
        _response(
            {"cot_id": _COT_ID, "message_id": _MESSAGE_ID},
            status_code=500,
        ),
        _response(
            {"cot_id": _COT_ID, "message_id": _MESSAGE_ID},
            code=True,
        ),
        _response(
            {"cot_id": _COT_ID, "message_id": _MESSAGE_ID},
            sdk_code=True,
        ),
        _response(raw_content=b"not-json"),
        _response(),
        _response({"cot_id": "bad/id", "message_id": _MESSAGE_ID}),
    ],
)
def test_api_rejects_invalid_base_raw_business_and_create_schema(
    outcome: object,
) -> None:
    client = _RecordingClient(outcome)
    with pytest.raises(COTTransportError):
        _api(client).create(_CHAT_ID)


def test_create_response_accepts_forward_compatible_data_fields() -> None:
    client = _RecordingClient(
        _response(
            {
                "cot_id": _COT_ID,
                "message_id": _MESSAGE_ID,
                "server_extension": {"version": 2},
            }
        )
    )
    binding = _api(client).create(_CHAT_ID)
    assert binding.cot_id == _COT_ID
    assert binding.message_id == _MESSAGE_ID
    assert len(client.requests) == 1


def test_update_and_complete_require_empty_data_schema() -> None:
    client = _RecordingClient(
        _create_response(),
        _response({"unexpected": True}),
        _response({"unexpected": True}),
    )
    api = _api(client)
    binding = api.create(_CHAT_ID)
    event = {
        "event_type": "RUN_ERROR",
        "content": {"message": "failed"},
        "timestamp": 1,
    }
    with pytest.raises(COTTransportError, match="update response schema"):
        api.append(binding, (event,))
    with pytest.raises(COTTransportError, match="complete response schema"):
        api.complete(binding, reason="error")


def test_audit_aliases_and_revision_snapshot_fail_before_network() -> None:
    client = _RecordingClient(_create_response())
    audit_calls: list[tuple[str, str, str]] = []
    api = _api(
        client,
        outbound_audit=lambda *args: audit_calls.append(args),
        outbound_target_aliases=lambda _target: (),
    )
    with pytest.raises(COTTransportError, match="recipient scope"):
        api.create(_CHAT_ID, origin_message_id=_ORIGIN_ID)
    assert client.requests == []
    assert audit_calls == []

    mismatched = _api(
        client,
        outbound_audit=lambda *args: audit_calls.append(args),
        outbound_target_aliases=lambda _target: ("oc_other_chat",),
    )
    with pytest.raises(COTTransportError, match="chat scope"):
        mismatched.create(_CHAT_ID, origin_message_id=_ORIGIN_ID)
    assert client.requests == []
    assert audit_calls == []

    snapshot_client = _RecordingClient(_create_response())
    snapshot_api = _api(
        snapshot_client,
        trust_revision_provider=lambda _chat_id: (_ for _ in ()).throw(
            RuntimeError("snapshot unavailable")
        ),
    )
    with pytest.raises(COTTransportError, match="trust revision unavailable"):
        snapshot_api.create(_CHAT_ID)
    assert snapshot_client.requests == []


def test_revision_change_blocks_update_and_complete_without_audit_or_network() -> None:
    client = _RecordingClient(_create_response())
    current_revision = [(2, 3)]
    audit_calls: list[tuple[str, str, str]] = []
    api = _api(
        client,
        trust_revision_provider=lambda _chat: current_revision[0],
        outbound_audit=lambda *args: audit_calls.append(args),
    )
    binding = api.create(_CHAT_ID)
    assert len(client.requests) == 1
    assert audit_calls == [("", "create", _CHAT_ID)]

    current_revision[0] = (2, 4)
    event = {
        "event_type": "RUN_ERROR",
        "content": {"message": "failed"},
        "timestamp": 1,
    }
    with pytest.raises(COTTransportError, match="trust revision changed"):
        api.append(binding, (event,))
    with pytest.raises(COTTransportError, match="trust revision changed"):
        api.complete(binding, reason="error")
    assert len(client.requests) == 1
    assert audit_calls == [("", "create", _CHAT_ID)]


def test_api_hard_timeout_is_single_attempt() -> None:
    entered = threading.Event()
    release = threading.Event()

    def block(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        return _create_response()

    client = _RecordingClient(block)
    api = _api(client, timeout_seconds=0.02)
    try:
        started_at = time.monotonic()
        with pytest.raises(COTTransportError, match="TimeoutError"):
            api.create(_CHAT_ID, timeout_seconds=1.0)
        assert time.monotonic() - started_at < 0.2
        assert entered.is_set()
        assert len(client.requests) == 1
    finally:
        release.set()


def test_session_maps_ordered_text_reasoning_and_latest_safe_tool_result_in_one_batch() -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.1),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        input_text="修复登录\ud800",
        flush_interval=0.02,
    )
    session.start()
    assert session.started is True
    assert session.healthy is True
    assert session.message_id == _MESSAGE_ID
    assert session._worker is not None and session._worker.daemon is True

    assert session.emit(CardEvent(type=CardEventType.TASK_LIST_UPDATED)) is False
    assert session.emit(CardEvent.text_started("opaque-text-id")) is True
    assert session.emit(CardEvent.text_delta("opaque-text-id", "答\ud800案")) is True
    assert session.emit(CardEvent.text_done("opaque-text-id")) is True
    assert session.emit(CardEvent.reasoning_started("opaque-reason-id")) is True
    assert session.emit(
        CardEvent(
            type=CardEventType.REASONING_DELTA,
            payload={"block_id": "opaque-reason-id", "text": "分析"},
        )
    ) is True
    assert session.emit(CardEvent.reasoning_done("opaque-reason-id")) is True

    tool_id = "call_secretopaque"
    assert session.emit(
        CardEvent.tool_started(
            tool_id,
            "shell",
            '{"command":"echo ok","API_KEY":"supersecret"}',
        )
    ) is True
    assert session.emit(CardEvent.tool_delta(tool_id, "first result")) is True
    assert session.emit(
        CardEvent.tool_delta(
            tool_id,
            f"second result {tool_id} API_KEY=supersecret",
        )
    ) is True
    assert session.emit(
        CardEvent(
            type=CardEventType.TOOL_DONE,
            payload={"block_id": tool_id, "tool_output": ""},
        )
    ) is True
    assert session.emit(
        CardEvent(
            type=CardEventType.TOOL_DONE,
            payload={"block_id": tool_id, "tool_output": "duplicate"},
        )
    ) is True

    assert session.complete(CardEvent.completed()) is True
    assert session.complete(CardEvent.completed()) is True
    assert len(client.requests) == 5

    run_started = _request_events(client.requests[1])
    assert [event["event_type"] for event in run_started] == ["RUN_STARTED"]
    started_content = _event_content(run_started[0])
    assert started_content["input"]["query"] == "修复登录�"
    assert started_content["input"]["statusText"] == {
        "running": {"zh_cn": "过程记录中", "en_us": "Recording process"},
        "thinking": {"zh_cn": "正在记录过程", "en_us": "Recording process"},
        "done": {"zh_cn": "过程记录完成", "en_us": "Process recorded"},
        "error": {
            "zh_cn": "过程记录异常结束，请查看主卡",
            "en_us": "Process ended unexpectedly; see the main card",
        },
        "paused": {
            "zh_cn": "过程展示已切换到主卡",
            "en_us": "Process moved to the main card",
        },
        "interrupted": {
            "zh_cn": "过程记录已中断",
            "en_us": "Process interrupted",
        },
    }
    assert set(started_content) == {"threadId", "runId", "input"}

    batch = _request_events(client.requests[2])
    assert [event["event_type"] for event in batch] == [
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    ]
    contents = [_event_content(event) for event in batch]
    assert contents[0]["messageId"] == contents[1]["messageId"] == contents[2]["messageId"]
    assert contents[1]["delta"] == "答�案"
    assert "opaque-text-id" not in json.dumps(batch, ensure_ascii=False)
    assert contents[3]["messageId"] == contents[4]["messageId"]
    assert contents[4]["messageId"] == contents[5]["messageId"]
    assert contents[5]["messageId"] == contents[6]["messageId"] == contents[7]["messageId"]

    tool_start, tool_args, tool_end, tool_result = contents[-4:]
    assert tool_start["toolCallId"] == tool_args["toolCallId"]
    assert tool_args["toolCallId"] == tool_end["toolCallId"]
    assert tool_end["toolCallId"] == tool_result["toolCallId"]
    assert tool_start["toolCallName"] == "shell"
    assert set(tool_result) == {"messageId", "toolCallId", "content", "role"}
    assert tool_result["role"] == "tool"
    assert "second result" in tool_result["content"]
    assert "first result" not in tool_result["content"]
    assert tool_id not in json.dumps(batch, ensure_ascii=False)
    assert "supersecret" not in json.dumps(batch, ensure_ascii=False)

    terminal = _request_events(client.requests[3])[0]
    assert terminal["event_type"] == "RUN_FINISHED"
    terminal_content = _event_content(terminal)
    assert terminal_content == {
        "threadId": started_content["threadId"],
        "runId": started_content["runId"],
        "status": "done",
    }
    all_events = [*run_started, *batch, terminal]
    timestamps = [event["timestamp"] for event in all_events]
    assert all(isinstance(timestamp, int) for timestamp in timestamps)
    assert all(left < right for left, right in zip(timestamps, timestamps[1:]))


@pytest.mark.parametrize(
    (
        "terminal_event",
        "event_type",
        "status",
        "error_code",
        "complete_reason",
    ),
    [
        (CardEvent.completed(), "RUN_FINISHED", "done", None, "done"),
        (
            CardEvent.cancelled(reason="user_cancelled"),
            "RUN_FINISHED",
            "interrupted",
            None,
            "error",
        ),
        (
            CardEvent.cancelled(reason="external_timeout"),
            "RUN_ERROR",
            None,
            "TIMEOUT",
            "error",
        ),
        (
            CardEvent.failed("bad\ud800task"),
            "RUN_ERROR",
            None,
            "TASK_FAILED",
            "error",
        ),
    ],
)
def test_session_terminal_mapping_and_complete_reason(
    terminal_event: CardEvent,
    event_type: str,
    status: str | None,
    error_code: str | None,
    complete_reason: str,
) -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.01,
    )
    session.start()
    assert session.complete(terminal_event) is True

    started_content = _event_content(_request_events(client.requests[1])[0])
    terminal = _request_events(client.requests[2])[0]
    content = _event_content(terminal)
    assert terminal["event_type"] == event_type
    assert content["threadId"] == started_content["threadId"]
    assert content["runId"] == started_content["runId"]
    if status is not None:
        assert content["status"] == status
    else:
        assert content["code"] == error_code
        assert content["message"] == (
            "bad�task" if error_code == "TASK_FAILED" else "Task timed out"
        )
    assert client.requests[3].queries[-1] == ("reason", complete_reason)


def test_tool_failure_emits_one_error_result() -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.01,
    )
    session.start()
    assert session.emit(CardEvent.tool_started("call_failed", "shell", "{}"))
    assert session.emit(
        CardEvent(
            type=CardEventType.TOOL_FAILED,
            payload={
                "block_id": "call_failed",
                "error": "API_TOKEN=secret failure",
            },
        )
    )
    assert session.emit(
        CardEvent(
            type=CardEventType.TOOL_FAILED,
            payload={"block_id": "call_failed", "error": "duplicate"},
        )
    )
    assert session.complete(CardEvent.completed())

    batch = _request_events(client.requests[2])
    assert [event["event_type"] for event in batch] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    ]
    result = _event_content(batch[-1])
    assert set(result) == {"messageId", "toolCallId", "content", "role"}
    assert result["role"] == "tool"
    assert "secret" not in result["content"]


def test_large_text_delta_is_losslessly_chunked_into_bounded_update_requests() -> None:
    client = _RecordingClient(
        _create_response(),
        *(_empty_response() for _ in range(20)),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.1),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    session.start()
    text = ("中文-and-ascii-" * 2_000) + "终"
    assert session.emit(CardEvent.text_started("large-text")) is True
    assert session.emit(CardEvent.text_delta("large-text", text)) is True
    assert session.emit(CardEvent.text_done("large-text")) is True
    assert session.complete(CardEvent.completed()) is True

    update_requests = client.requests[2:-2]
    assert len(update_requests) > 1
    assert all(
        len(
            json.dumps(
                request.body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= 16 * 1024
        for request in update_requests
    )
    deltas = [
        _event_content(event)["delta"]
        for request in update_requests
        for event in _request_events(request)
        if event["event_type"] == "TEXT_MESSAGE_CONTENT"
    ]
    assert "".join(deltas) == text
    assert all(len(delta.encode("utf-8")) <= 4_000 for delta in deltas)


def test_large_tool_result_uses_exact_agui_schema_and_utf8_byte_cap() -> None:
    client = _RecordingClient(
        _create_response(),
        *(_empty_response() for _ in range(10)),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.1),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    session.start()
    assert session.emit(CardEvent.tool_started("large-tool", "shell", "{}"))
    assert session.emit(
        CardEvent(
            type=CardEventType.TOOL_DONE,
            payload={
                "block_id": "large-tool",
                "tool_output": "结果" * 10_000,
            },
        )
    )
    assert session.complete(CardEvent.completed()) is True

    result = next(
        _event_content(event)
        for request in client.requests[2:-2]
        for event in _request_events(request)
        if event["event_type"] == "TOOL_CALL_RESULT"
    )
    assert set(result) == {"messageId", "toolCallId", "content", "role"}
    assert result["role"] == "tool"
    assert len(str(result["content"]).encode("utf-8")) <= 4_000
    assert str(result["content"]).endswith("…")


def test_abort_neutrally_closes_process_sidecar_exactly_once() -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.01,
    )
    session.start()
    session.abort()
    session.abort()

    _wait_until(lambda: len(client.requests) == 4)
    assert len(client.requests) == 4
    terminal = _request_events(client.requests[2])[0]
    content = _event_content(terminal)
    started_content = _event_content(_request_events(client.requests[1])[0])
    assert terminal["event_type"] == "RUN_FINISHED"
    assert content == {
        "threadId": started_content["threadId"],
        "runId": started_content["runId"],
        "status": "paused",
    }
    assert client.requests[3].queries[-1] == ("reason", "done")
    assert session.healthy is False


def test_async_update_failure_is_detectable_without_retry_or_emit_exception() -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        OSError("network down"),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.01,
    )
    session.start()
    assert session.emit(CardEvent.text_started("text-one")) is True
    _wait_until(lambda: not session.healthy)
    assert session.emit(CardEvent.text_delta("text-one", "late")) is False
    assert session.complete(CardEvent.completed()) is False

    _wait_until(lambda: len(client.requests) == 5)
    terminal = _request_events(client.requests[3])[0]
    content = _event_content(terminal)
    started_content = _event_content(_request_events(client.requests[1])[0])
    assert terminal["event_type"] == "RUN_FINISHED"
    assert content["threadId"] == started_content["threadId"]
    assert content["runId"] == started_content["runId"]
    assert content["status"] == "paused"
    assert client.requests[4].queries[-1] == ("reason", "done")
    assert session._worker_done.wait(1.0)


def test_async_update_business_failure_logs_the_sanitized_error_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _response({}, code=230099),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    session.start()
    with caplog.at_level("WARNING", logger="src.feishu.cot"):
        assert session.emit(CardEvent.text_started("text-one")) is True
        _wait_until(lambda: not session.healthy)

    assert "stage=update" in caplog.text
    assert "code=230099" in caplog.text


@pytest.mark.parametrize(
    ("parent_terminal", "event_type", "status_or_code", "complete_reason"),
    [
        (CardEvent.completed(), "RUN_FINISHED", "done", "done"),
        (CardEvent.failed("parent failed"), "RUN_ERROR", "TASK_FAILED", "error"),
    ],
)
def test_queue_backpressure_does_not_override_real_parent_terminal(
    parent_terminal: CardEvent,
    event_type: str,
    status_or_code: str,
    complete_reason: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_update(_request: BaseRequest) -> BaseResponse:
        entered.set()
        assert release.wait(1.0)
        return _empty_response()

    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        blocking_update,
        _empty_response(),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.1),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.01,
        queue_capacity=1,
    )
    session.start()
    assert session.emit(CardEvent.text_started("text-one")) is True
    assert entered.wait(1.0)
    assert session.emit(CardEvent.text_delta("text-one", "queued")) is True
    assert session.emit(CardEvent.text_delta("text-one", "overflow")) is False
    assert session.healthy is False
    release.set()
    assert session.complete(parent_terminal) is False

    assert len(client.requests) == 6
    terminal = _request_events(client.requests[4])[0]
    assert terminal["event_type"] == event_type
    content = _event_content(terminal)
    assert (content.get("status") or content.get("code")) == status_or_code
    assert client.requests[5].queries[-1] == ("reason", complete_reason)


def test_drain_timeout_never_sends_concurrent_terminal_or_complete() -> None:
    client = _RecordingClient(_create_response(), _empty_response())
    api = _api(client, timeout_seconds=0.01)
    session = FeishuCOTSession(
        api,
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
        close_timeout=0.04,
    )
    session.start()

    entered = threading.Event()
    release = threading.Event()
    original_append = api.append

    def blocking_append(
        binding: object,
        events: object,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        del binding, events, timeout_seconds
        entered.set()
        release.wait(1.0)

    api.append = blocking_append  # type: ignore[method-assign]
    try:
        assert session.emit(CardEvent.text_started("text-one")) is True
        assert entered.wait(1.0)
        assert session.complete(CardEvent.completed()) is False
        assert len(client.requests) == 2
        assert session.healthy is False
    finally:
        release.set()
        api.append = original_append  # type: ignore[method-assign]
        assert session._worker_done.wait(1.0)


def test_sdk_update_timeout_never_sends_concurrent_terminal_or_complete() -> None:
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def blocking_update(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        exited.set()
        return _empty_response()

    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        blocking_update,
        _empty_response(),
        _empty_response(),
    )
    api = _api(client, timeout_seconds=0.02)
    session = FeishuCOTSession(
        api,
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    session.start()
    assert session.emit(CardEvent.text_started("text-one")) is True
    assert entered.wait(1.0)

    try:
        assert session.complete(CardEvent.completed()) is False
        assert session.healthy is False
        assert len(client.requests) == 3
    finally:
        release.set()
    assert exited.wait(1.0)
    _wait_until(lambda: len(client.requests) == 5)
    terminal = _request_events(client.requests[3])[0]
    assert terminal["event_type"] == "RUN_FINISHED"
    assert _event_content(terminal)["status"] == "paused"
    assert client.requests[4].queries[-1] == ("reason", "done")
    assert session._worker_done.wait(1.0)


def test_close_never_expands_the_per_request_hard_deadline() -> None:
    client = _RecordingClient(_create_response(), _empty_response())
    api = _api(client, timeout_seconds=0.05)
    session = FeishuCOTSession(
        api,
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.01,
    )
    session.start()
    observed_timeouts: list[float] = []

    def record_append(
        binding: object,
        events: object,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        del binding, events
        assert timeout_seconds is not None
        observed_timeouts.append(timeout_seconds)

    def record_complete(
        binding: object,
        *,
        reason: str,
        timeout_seconds: float | None = None,
    ) -> None:
        del binding, reason
        assert timeout_seconds is not None
        observed_timeouts.append(timeout_seconds)

    api.append = record_append  # type: ignore[method-assign]
    api.complete = record_complete  # type: ignore[method-assign]
    assert session.complete(CardEvent.completed()) is True
    assert len(observed_timeouts) == 2
    assert all(0 < timeout <= api.request_timeout for timeout in observed_timeouts)


def test_late_create_success_completes_without_unpaired_run_error() -> None:
    entered = threading.Event()
    release = threading.Event()

    def late_create(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        return _create_response()

    client = _RecordingClient(
        late_create,
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.01,
        flush_interval=0.005,
    )
    with pytest.raises(COTTransportError, match="TimeoutError"):
        session.start()
    assert entered.is_set()
    assert len(client.requests) == 1

    release.set()
    _wait_until(lambda: len(client.requests) == 3)
    repaired_lifecycle = _request_events(client.requests[1])
    assert [event["event_type"] for event in repaired_lifecycle] == [
        "RUN_STARTED",
        "RUN_FINISHED",
    ]
    assert _event_content(repaired_lifecycle[1])["status"] == "paused"
    assert client.requests[2].uri.endswith("/complete/:cot_id")
    assert client.requests[2].queries[-1] == ("reason", "done")


def test_late_create_sdk_failure_never_guesses_a_cleanup_binding() -> None:
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def late_failure(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        exited.set()
        raise ConnectionResetError("response lost")

    client = _RecordingClient(late_failure)
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.01,
        flush_interval=0.005,
    )
    with pytest.raises(COTTransportError, match="TimeoutError"):
        session.start()
    assert entered.is_set()
    release.set()
    assert exited.wait(1.0)
    time.sleep(0.02)
    assert len(client.requests) == 1


def test_late_run_started_success_closes_in_fifo_order() -> None:
    entered = threading.Event()
    release = threading.Event()

    def late_run_started(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        return _empty_response()

    client = _RecordingClient(
        _create_response(),
        late_run_started,
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.01,
        flush_interval=0.005,
    )
    with pytest.raises(COTTransportError, match="TimeoutError"):
        session.start()
    assert entered.is_set()
    assert len(client.requests) == 2

    release.set()
    _wait_until(lambda: len(client.requests) == 4)
    terminal = _request_events(client.requests[2])[0]
    assert terminal["event_type"] == "RUN_FINISHED"
    assert _event_content(terminal)["status"] == "paused"
    assert client.requests[3].queries[-1] == ("reason", "done")


@pytest.mark.parametrize(
    ("parent_terminal", "complete_reason"),
    [
        (CardEvent.completed(), "done"),
        (CardEvent.failed("parent failed"), "error"),
    ],
)
def test_late_terminal_success_preserves_parent_reason_after_the_barrier(
    parent_terminal: CardEvent,
    complete_reason: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def late_terminal(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        return _empty_response()

    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        late_terminal,
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.01,
        flush_interval=0.005,
    )
    session.start()
    assert session.complete(parent_terminal) is False
    assert entered.is_set()
    assert len(client.requests) == 3

    release.set()
    _wait_until(lambda: len(client.requests) == 4)
    assert client.requests[3].uri.endswith("/complete/:cot_id")
    assert client.requests[3].queries[-1] == ("reason", complete_reason)


def test_late_complete_success_is_not_retried() -> None:
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()

    def late_complete(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        exited.set()
        return _empty_response()

    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _empty_response(),
        late_complete,
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.01,
        flush_interval=0.005,
    )
    session.start()
    assert session.complete(CardEvent.completed()) is False
    assert entered.is_set()
    assert len(client.requests) == 4

    release.set()
    assert exited.wait(1.0)
    time.sleep(0.02)
    assert len(client.requests) == 4


def test_abort_during_start_never_enters_accepting_state() -> None:
    entered = threading.Event()
    release = threading.Event()
    errors: list[Exception] = []

    def blocking_create(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        return _create_response()

    client = _RecordingClient(
        blocking_create,
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.2,
        flush_interval=0.005,
    )

    def run_start() -> None:
        try:
            session.start()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_start, daemon=True)
    thread.start()
    assert entered.wait(1.0)
    session.abort()
    release.set()
    thread.join(1.0)
    _wait_until(lambda: len(client.requests) == 3)

    assert errors and isinstance(errors[0], COTTransportError)
    assert session.started is False
    assert session.emit(CardEvent.text_started("never")) is False
    repaired_lifecycle = _request_events(client.requests[1])
    assert [event["event_type"] for event in repaired_lifecycle] == [
        "RUN_STARTED",
        "RUN_FINISHED",
    ]
    assert _event_content(repaired_lifecycle[1])["status"] == "paused"
    assert client.requests[2].queries[-1] == ("reason", "done")


def test_session_close_budget_is_independent_and_capped() -> None:
    client = _RecordingClient(_create_response(), _empty_response())
    api = _api(client, timeout_seconds=35.0)
    session = FeishuCOTSession(
        api,
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
    )
    assert session._request_timeout == 5.0
    assert session._close_timeout == 4.5
    with pytest.raises(ValueError, match="6.0"):
        FeishuCOTSession(
            api,
            chat_id=_CHAT_ID,
            origin_message_id=_ORIGIN_ID,
            close_timeout=6.01,
        )


def test_trust_revision_change_during_audit_blocks_network() -> None:
    revision = [(4, 9)]
    audit_calls: list[tuple[str, str, str]] = []

    def audit(tenant: str, operation: str, target: str) -> None:
        audit_calls.append((tenant, operation, target))
        revision[0] = (5, 9)

    client = _RecordingClient(_create_response())
    api = _api(
        client,
        outbound_audit=audit,
        outbound_target_aliases=lambda _target: (_CHAT_ID,),
        trust_revision_provider=lambda _chat: revision[0],
    )
    with pytest.raises(COTTransportError, match="revision changed"):
        api.create(_CHAT_ID, origin_message_id=_ORIGIN_ID)
    assert audit_calls
    assert client.requests == []


def test_close_drains_multiple_batches_before_success_terminal() -> None:
    client = _RecordingClient(
        _create_response(),
        *(_empty_response() for _ in range(5)),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.1),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.1,
        queue_capacity=64,
    )
    session.start()
    for index in range(40):
        assert session.emit(CardEvent.text_started(f"text-{index}")) is True
    assert session.complete(CardEvent.completed()) is True

    wire_types = [
        event["event_type"]
        for request in client.requests[2:-2]
        for event in _request_events(request)
    ]
    assert wire_types == ["TEXT_MESSAGE_START"] * 40
    assert _request_events(client.requests[-2])[0]["event_type"] == "RUN_FINISHED"


def test_abort_returns_before_an_inflight_update_finishes() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_update(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        return _empty_response()

    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        blocking_update,
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.2,
        flush_interval=0.005,
    )
    session.start()
    assert session.emit(CardEvent.text_started("text-one")) is True
    assert entered.wait(1.0)

    started_at = time.monotonic()
    session.abort()
    assert time.monotonic() - started_at < 0.05
    assert len(client.requests) == 3
    release.set()
    _wait_until(lambda: len(client.requests) == 5)
    terminal = _request_events(client.requests[3])[0]
    assert terminal["event_type"] == "RUN_FINISHED"
    assert _event_content(terminal)["status"] == "paused"
    assert client.requests[4].queries[-1] == ("reason", "done")


def test_authoritative_close_returns_within_independent_total_budget() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_terminal(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(2.0)
        return _empty_response()

    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        blocking_terminal,
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=35.0),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=2.0,
        close_timeout=1.2,
        flush_interval=0.005,
    )
    session.start()
    started_at = time.monotonic()
    assert session.complete(CardEvent.completed()) is False
    elapsed = time.monotonic() - started_at
    assert entered.is_set()
    assert elapsed < 1.0
    assert len(client.requests) == 3

    release.set()
    _wait_until(lambda: len(client.requests) == 4)
    assert client.requests[3].queries[-1] == ("reason", "done")


def test_late_explicit_run_started_failure_gets_neutral_ordered_cleanup() -> None:
    entered = threading.Event()
    release = threading.Event()

    def late_business_failure(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        return _response({}, code=23)

    client = _RecordingClient(
        _create_response(),
        late_business_failure,
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.01,
        flush_interval=0.005,
    )
    with pytest.raises(COTTransportError, match="TimeoutError"):
        session.start()
    assert entered.is_set()
    release.set()
    _wait_until(lambda: len(client.requests) == 4)
    repair = _request_events(client.requests[2])
    assert [event["event_type"] for event in repair] == [
        "RUN_STARTED",
        "RUN_FINISHED",
    ]
    assert _event_content(repair[1])["status"] == "paused"
    assert client.requests[3].queries[-1] == ("reason", "done")


@pytest.mark.parametrize("operation", ["append", "complete"])
def test_mutation_revision_change_during_audit_blocks_network(
    operation: str,
) -> None:
    revision = [(8, 13)]
    mutate_during_patch = [False]

    def audit(_tenant: str, audit_operation: str, _target: str) -> None:
        if audit_operation == "patch" and mutate_during_patch[0]:
            revision[0] = (9, 13)

    client = _RecordingClient(_create_response())
    api = _api(
        client,
        outbound_audit=audit,
        outbound_target_aliases=lambda _target: (_CHAT_ID,),
        trust_revision_provider=lambda _chat: revision[0],
    )
    binding = api.create(_CHAT_ID, origin_message_id=_ORIGIN_ID)
    mutate_during_patch[0] = True
    event = {
        "event_type": "RUN_ERROR",
        "content": {"message": "failed"},
        "timestamp": 1,
    }

    with pytest.raises(COTTransportError, match="revision changed"):
        if operation == "append":
            api.append(binding, (event,))
        else:
            api.complete(binding, reason="error")
    assert len(client.requests) == 1


def test_worker_thread_construction_failure_closes_started_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_thread = threading.Thread

    def guarded_thread(*args: object, **kwargs: object) -> threading.Thread:
        name = kwargs.get("name")
        if isinstance(name, str) and name.startswith("feishu-cot-") and "-api-" not in name:
            raise RuntimeError("thread construction failed")
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", guarded_thread)
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    with pytest.raises(COTTransportError, match="worker start"):
        session.start()

    assert len(client.requests) == 4
    terminal = _request_events(client.requests[2])[0]
    assert terminal["event_type"] == "RUN_FINISHED"
    assert _event_content(terminal)["status"] == "paused"
    assert client.requests[3].queries[-1] == ("reason", "done")


def test_run_started_sdk_exception_gets_neutral_ordered_cleanup() -> None:
    client = _RecordingClient(
        _create_response(),
        ConnectionResetError("response lost"),
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    with pytest.raises(COTTransportError, match="ConnectionResetError"):
        session.start()

    _wait_until(lambda: len(client.requests) == 4)
    cleanup = _request_events(client.requests[2])
    # The server may have accepted RUN_STARTED before the connection failed,
    # so cleanup must not blindly send a second RUN_STARTED.
    assert [event["event_type"] for event in cleanup] == ["RUN_FINISHED"]
    terminal = cleanup[0]
    assert terminal["event_type"] == "RUN_FINISHED"
    assert _event_content(terminal)["status"] == "paused"
    assert client.requests[3].queries[-1] == ("reason", "done")


def test_terminal_sdk_exception_completes_only_after_call_returns() -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        ConnectionResetError("response lost"),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    session.start()
    assert session.complete(CardEvent.completed()) is False

    _wait_until(lambda: len(client.requests) == 4)
    assert client.requests[3].uri.endswith("/complete/:cot_id")
    assert client.requests[3].queries[-1] == ("reason", "done")


def test_complete_sdk_exception_is_never_retried() -> None:
    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        _empty_response(),
        ConnectionResetError("response lost"),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.05),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        flush_interval=0.005,
    )
    session.start()
    assert session.complete(CardEvent.completed()) is False
    time.sleep(0.02)
    assert len(client.requests) == 4


def test_worker_append_failure_exits_after_late_cleanup_without_caller_close() -> None:
    entered = threading.Event()
    release = threading.Event()

    def failing_update(_request: BaseRequest) -> BaseResponse:
        entered.set()
        release.wait(1.0)
        raise ConnectionResetError("response lost")

    client = _RecordingClient(
        _create_response(),
        _empty_response(),
        failing_update,
        _empty_response(),
        _empty_response(),
    )
    session = FeishuCOTSession(
        _api(client, timeout_seconds=0.2),
        chat_id=_CHAT_ID,
        origin_message_id=_ORIGIN_ID,
        request_timeout=0.2,
        flush_interval=0.005,
        queue_capacity=4,
    )
    session.start()
    assert session.emit(CardEvent.text_started("text-one")) is True
    assert entered.wait(1.0)
    assert session.emit(CardEvent.text_delta("text-one", "queued")) is True

    release.set()
    _wait_until(lambda: len(client.requests) == 5)
    _wait_until(lambda: session._closed)
    assert session._worker_done.wait(1.0)
    assert session._queue.empty()
    terminal = _request_events(client.requests[3])[0]
    assert terminal["event_type"] == "RUN_FINISHED"
    assert _event_content(terminal)["status"] == "paused"
    assert client.requests[4].queries[-1] == ("reason", "done")
