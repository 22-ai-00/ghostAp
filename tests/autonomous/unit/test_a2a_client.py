from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from src.autonomous.a2a.card import (
    PilotAgentRegistration,
    canonical_card_digest,
)
from src.autonomous.a2a.client import (
    A2AClientLimits,
    RemoteA2AClientError,
    RemoteA2ADispatchAdapter,
    _hardened_http_transport,
    _PublicOnlyNetworkBackend,
)
from src.autonomous.a2a.codec import MAX_OBSERVATION_BYTES, NormalizedA2AObservation
from src.autonomous.journal.blob_store import BlobRef
from src.autonomous.remote.models import (
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteObservation,
    RemoteProjection,
    RemoteSnapshot,
    RemoteTaskHandle,
    RemoteTaskState,
)

_CARD_URL = "https://cards.example.test/.well-known/agent-card.json"
_ENDPOINT_URL = "https://agent.example.test/a2a"
_TOKEN = "sentinel-a2a-token"


def _card_bytes() -> bytes:
    return json.dumps(
        {
            "name": "reviewer",
            "description": "Reviews an anchored change",
            "supportedInterfaces": [
                {
                    "url": _ENDPOINT_URL,
                    "protocolBinding": "JSONRPC",
                    "protocolVersion": "1.0",
                }
            ],
            "version": "1",
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "skills": [
                {
                    "id": "review",
                    "name": "Review",
                    "description": "Review an anchored change",
                    "tags": ["review"],
                }
            ],
        },
        separators=(",", ":"),
    ).encode()


def _registration(raw: bytes | None = None) -> PilotAgentRegistration:
    card = raw or _card_bytes()
    return PilotAgentRegistration(
        tenant_key="tenant-a",
        agent_id="agt_reviewer",
        card_url=_CARD_URL,
        endpoint_url=_ENDPOINT_URL,
        expected_card_digest=canonical_card_digest(card),
        credential_ref="cred_a2a_reviewer",
    )


class _Resolver:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls = 0

    async def resolve_bearer(self, _descriptor: object) -> str:
        self.calls += 1
        self.order.append("credential")
        return _TOKEN


class _TrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: Any) -> None:
        self._inner = httpx.MockTransport(handler)
        self.close_calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        self.close_calls += 1
        await self._inner.aclose()


class _FakeLedger:
    def __init__(self) -> None:
        self.snapshots: dict[str, RemoteSnapshot] = {}
        self.order: list[str] = []

    def projection(self) -> RemoteProjection:
        return RemoteProjection(
            by_key={snapshot.handle.key: snapshot for snapshot in self.snapshots.values()},
            by_acceptance_id={acceptance_id: snapshot.handle.key for acceptance_id, snapshot in self.snapshots.items()},
        )

    def snapshot(self, acceptance_id: str) -> RemoteSnapshot | None:
        return self.snapshots.get(acceptance_id)

    def prepare(self, request: RemoteDispatchRequest) -> RemoteTaskHandle:
        self.order.append("prepared")
        digest = hashlib.sha256(request.instruction.encode()).hexdigest()
        handle = RemoteTaskHandle(
            acceptance_id=request.acceptance_id,
            run_id=request.run_id,
            assignment_id=request.assignment_id,
            attempt_id=request.attempt_id,
            message_id=request.message_id,
            context_id=request.context_id,
            descriptor=request.descriptor,
            instruction_ref=BlobRef(blob_hash="a" * 64, payload_hash=digest),
        )
        self.snapshots[request.acceptance_id] = RemoteSnapshot(
            handle=handle,
            phase=RemoteAttemptPhase.PREPARED,
            state=RemoteTaskState.PREPARED,
        )
        return handle

    def mark_executing(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        self.order.append("executing")
        snapshot = replace(
            self.snapshots[handle.acceptance_id],
            phase=RemoteAttemptPhase.EXECUTING,
            state=RemoteTaskState.EXECUTING,
        )
        self.snapshots[handle.acceptance_id] = snapshot
        return snapshot

    def bind_task(self, handle: RemoteTaskHandle, task_id: str) -> RemoteTaskHandle:
        self.order.append("bound")
        bound = replace(handle, task_id=task_id)
        self.snapshots[handle.acceptance_id] = replace(
            self.snapshots[handle.acceptance_id],
            handle=bound,
            phase=RemoteAttemptPhase.TRACKING,
        )
        return bound

    def next_observation_sequence(self, handle: RemoteTaskHandle) -> int:
        return len(self.snapshots[handle.acceptance_id].observations)

    def record_observation(
        self,
        handle: RemoteTaskHandle,
        observation: NormalizedA2AObservation | RemoteObservation,
        payload: bytes | None = None,
        *,
        observed_state: RemoteTaskState | None = None,
        sequence: int | None = None,
    ) -> RemoteObservation:
        del payload
        self.order.append("recorded")
        if isinstance(observation, NormalizedA2AObservation):
            remote = observation.to_remote_observation(
                handle,
                content_ref=BlobRef(
                    blob_hash="b" * 64,
                    payload_hash=observation.payload_digest,
                ),
                observed_state=observed_state,
                sequence=sequence,
            )
        else:
            remote = observation
        snapshot = self.snapshots[handle.acceptance_id]
        duplicate = next(
            (item for item in snapshot.observations if item.observation_id == remote.observation_id),
            None,
        )
        if duplicate is not None:
            return duplicate
        phase = RemoteAttemptPhase.TERMINAL if remote.state.is_remote_terminal else snapshot.phase
        self.snapshots[handle.acceptance_id] = replace(
            snapshot,
            phase=phase,
            state=remote.state,
            observations=(*snapshot.observations, remote),
        )
        return remote

    def mark_send_uncertain(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        self.order.append("send_uncertain")
        snapshot = replace(
            self.snapshots[handle.acceptance_id],
            phase=RemoteAttemptPhase.SEND_UNCERTAIN,
            state=RemoteTaskState.OUTCOME_UNCERTAIN,
        )
        self.snapshots[handle.acceptance_id] = snapshot
        return snapshot

    def request_cancel(self, handle: RemoteTaskHandle) -> RemoteSnapshot:
        self.order.append("cancel_requested")
        snapshot = replace(
            self.snapshots[handle.acceptance_id],
            phase=RemoteAttemptPhase.CANCEL_REQUESTED,
            cancel_requested=True,
        )
        self.snapshots[handle.acceptance_id] = snapshot
        return snapshot


async def _public_dns(_host: str, _port: int) -> tuple[str, ...]:
    return ("8.8.8.8", "2606:4700:4700::1111")


async def _allow_mock_peer(_response: httpx.Response) -> None:
    return None


def _task(*, state: str = "TASK_STATE_COMPLETED") -> dict[str, Any]:
    return {
        "id": "task-1",
        "contextId": "ctx-1",
        "status": {"state": state},
    }


def _rpc_response(request_id: object, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _sse(request_id: object, result: dict[str, Any]) -> httpx.Response:
    payload = json.dumps(_rpc_response(request_id, result), separators=(",", ":"))
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {payload}\n\n".encode(),
    )


def _request(adapter: RemoteA2ADispatchAdapter) -> RemoteDispatchRequest:
    return RemoteDispatchRequest(
        acceptance_id="acceptance-1",
        run_id="run-1",
        assignment_id="assignment-1",
        attempt_id="attempt-1",
        message_id="message-1",
        context_id="ctx-1",
        instruction="Review the anchored change",
        descriptor=adapter.descriptor,
    )


@pytest.mark.asyncio
async def test_dispatch_validates_public_card_then_anchors_stream_before_yield() -> None:
    raw_card = _card_bytes()
    ledger = _FakeLedger()
    order: list[str] = []
    endpoint_requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            order.append("card")
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        body = json.loads(request.content)
        endpoint_requests.append(body)
        assert ledger.order[-1] == "executing"
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        return _sse(body["id"], {"task": _task()})

    resolver = _Resolver(order)
    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        resolver,
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    observations = [item async for item in adapter.dispatch(_request(adapter))]

    assert order == ["card", "credential"]
    assert ledger.order == ["prepared", "executing", "bound", "recorded"]
    assert [item.state for item in observations] == [RemoteTaskState.CLAIMED_COMPLETED]
    assert endpoint_requests[0]["method"] == "SendStreamingMessage"
    assert endpoint_requests[0]["params"]["message"] == {
        "messageId": "message-1",
        "contextId": "ctx-1",
        "role": "ROLE_USER",
        "parts": [{"text": "Review the anchored change", "mediaType": "text/plain"}],
    }
    assert "taskPushNotificationConfig" not in endpoint_requests[0]["params"]["configuration"]
    await adapter.close()


@pytest.mark.asyncio
async def test_known_task_restart_subscribes_without_resending() -> None:
    raw_card = _card_bytes()
    ledger = _FakeLedger()
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        body = json.loads(request.content)
        methods.append(body["method"])
        return _sse(
            body["id"],
            {
                "statusUpdate": {
                    "taskId": "task-1",
                    "contextId": "ctx-1",
                    "status": {"state": "TASK_STATE_COMPLETED"},
                }
            },
        )

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    request = _request(adapter)
    handle = ledger.prepare(request)
    ledger.mark_executing(handle)
    ledger.bind_task(handle, "task-1")

    observations = [item async for item in adapter.dispatch(request)]

    assert methods == ["SubscribeToTask"]
    assert observations[-1].state is RemoteTaskState.CLAIMED_COMPLETED
    assert "SendStreamingMessage" not in methods
    await adapter.close()


@pytest.mark.asyncio
async def test_executing_attempt_without_task_id_fails_closed_without_network() -> None:
    raw_card = _card_bytes()
    ledger = _FakeLedger()
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        methods.append(json.loads(request.content)["method"])
        raise AssertionError("unknown send outcome must not be resent")

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    request = _request(adapter)
    handle = ledger.prepare(request)
    ledger.mark_executing(handle)

    with pytest.raises(RemoteA2AClientError, match="send-outcome-unknown"):
        _ = [item async for item in adapter.dispatch(request)]
    assert methods == []
    assert ledger.snapshot("acceptance-1").phase is RemoteAttemptPhase.SEND_UNCERTAIN
    await adapter.close()


@pytest.mark.asyncio
async def test_private_dns_result_rejected_before_card_or_credential() -> None:
    handler_called = False
    resolver = _Resolver([])

    async def private_dns(_host: str, _port: int) -> tuple[str, ...]:
        return ("8.8.8.8", "127.0.0.1")

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal handler_called
        handler_called = True
        return httpx.Response(500)

    with pytest.raises(RemoteA2AClientError, match="endpoint-resolution-rejected"):
        await RemoteA2ADispatchAdapter.create(
            _registration(),
            _FakeLedger(),
            resolver,
            dns_resolver=private_dns,
            response_peer_verifier=_allow_mock_peer,
            transport=httpx.MockTransport(handler),
        )
    assert handler_called is False
    assert resolver.calls == 0


@pytest.mark.asyncio
async def test_private_endpoint_dns_is_rejected_before_credential_resolution() -> None:
    raw_card = _card_bytes()
    credential = _Resolver([])
    dns_calls = 0

    async def changing_dns(_host: str, _port: int) -> tuple[str, ...]:
        nonlocal dns_calls
        dns_calls += 1
        return ("8.8.8.8",) if dns_calls == 1 else ("10.0.0.8",)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=raw_card,
        )

    with pytest.raises(RemoteA2AClientError, match="endpoint-resolution-rejected"):
        await RemoteA2ADispatchAdapter.create(
            _registration(raw_card),
            _FakeLedger(),
            credential,
            dns_resolver=changing_dns,
            response_peer_verifier=_allow_mock_peer,
            transport=httpx.MockTransport(handler),
        )

    assert dns_calls == 2
    assert credential.calls == 0


@pytest.mark.asyncio
async def test_missing_actual_peer_address_fails_closed_before_credential() -> None:
    resolver = _Resolver([])

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=_card_bytes(),
        )

    with pytest.raises(RemoteA2AClientError, match="peer-address-unavailable"):
        await RemoteA2ADispatchAdapter.create(
            _registration(),
            _FakeLedger(),
            resolver,
            dns_resolver=_public_dns,
            transport=httpx.MockTransport(handler),
        )
    assert resolver.calls == 0


@pytest.mark.asyncio
async def test_connect_time_peer_check_rejects_rebinding_before_stream_is_used() -> None:
    class PrivateStream:
        closed = False

        @staticmethod
        def get_extra_info(_info: str) -> tuple[str, int]:
            return "127.0.0.1", 443

        async def aclose(self) -> None:
            self.closed = True

    class Backend:
        def __init__(self) -> None:
            self.stream = PrivateStream()

        async def connect_tcp(self, *_args: object, **_kwargs: object) -> PrivateStream:
            return self.stream

        async def sleep(self, _seconds: float) -> None:
            return None

    backend = Backend()
    guarded = _PublicOnlyNetworkBackend(backend)  # type: ignore[arg-type]

    with pytest.raises(RemoteA2AClientError, match="peer-address-rejected"):
        await guarded.connect_tcp("agent.example.test", 443)
    assert backend.stream.closed is True


@pytest.mark.asyncio
async def test_pinned_httpx_transport_keeps_connect_time_peer_guard() -> None:
    transport = _hardened_http_transport(A2AClientLimits())
    try:
        pool = getattr(transport, "_pool")
        assert isinstance(pool._network_backend, _PublicOnlyNetworkBackend)
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_invalid_card_never_resolves_or_reflects_credential() -> None:
    raw_card = _card_bytes()
    changed_card = raw_card.replace(b"anchored", b"untrusted")
    resolver = _Resolver([])

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=changed_card,
        )

    with pytest.raises(RemoteA2AClientError) as exc_info:
        await RemoteA2ADispatchAdapter.create(
            _registration(raw_card),
            _FakeLedger(),
            resolver,
            dns_resolver=_public_dns,
            response_peer_verifier=_allow_mock_peer,
            transport=httpx.MockTransport(handler),
        )
    assert resolver.calls == 0
    assert _TOKEN not in str(exc_info.value)


@pytest.mark.asyncio
async def test_create_cancellation_closes_owned_http_client() -> None:
    raw_card = _card_bytes()
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingResolver:
        async def resolve_bearer(self, _descriptor: object) -> str:
            entered.set()
            await release.wait()
            return _TOKEN

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=raw_card,
        )

    transport = _TrackingTransport(handler)
    create_task = asyncio.create_task(
        RemoteA2ADispatchAdapter.create(
            _registration(raw_card),
            _FakeLedger(),
            BlockingResolver(),
            dns_resolver=_public_dns,
            response_peer_verifier=_allow_mock_peer,
            transport=transport,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    create_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await create_task
    assert transport.close_calls == 1


@pytest.mark.asyncio
async def test_close_failure_remains_retryable() -> None:
    raw_card = _card_bytes()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=raw_card,
        )

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        _FakeLedger(),
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    owned_client = adapter._client

    class FlakyClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("synthetic close failure")

    flaky_client = FlakyClient()
    adapter._client = flaky_client  # type: ignore[assignment]
    try:
        with pytest.raises(RemoteA2AClientError, match="client-close-failed"):
            await adapter.close()
        assert "closed=False" in repr(adapter)

        await adapter.close()

        assert flaky_client.close_calls == 2
        assert "closed=True" in repr(adapter)
    finally:
        await owned_client.close()


@pytest.mark.asyncio
async def test_duplicate_jsonrpc_key_is_rejected_and_send_marked_uncertain() -> None:
    raw_card = _card_bytes()
    ledger = _FakeLedger()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        request_id = json.loads(request.content)["id"]
        duplicate = (
            '{"jsonrpc":"2.0","id":'
            + json.dumps(request_id)
            + ',"result":{"task":'
            + json.dumps(_task())
            + '},"result":{"task":'
            + json.dumps(_task())
            + "}}"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {duplicate}\n\n".encode(),
        )

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RemoteA2AClientError, match="jsonrpc-duplicate-key"):
        _ = [item async for item in adapter.dispatch(_request(adapter))]
    assert ledger.snapshot("acceptance-1").phase is RemoteAttemptPhase.SEND_UNCERTAIN
    assert _TOKEN not in repr(adapter)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error_code"),
    [
        (b'{"jsonrpc":"2.0","id":"wrong","result":NaN}', "jsonrpc-non-finite-number"),
        (b"x" * (MAX_OBSERVATION_BYTES + 1), "sse-line-too-large"),
    ],
)
async def test_stream_json_is_strict_and_each_sse_event_is_bounded(
    payload: bytes,
    error_code: str,
) -> None:
    raw_card = _card_bytes()
    ledger = _FakeLedger()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: " + payload + b"\n\n",
        )

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RemoteA2AClientError, match=error_code):
        _ = [item async for item in adapter.dispatch(_request(adapter))]
    assert ledger.snapshot("acceptance-1").phase is RemoteAttemptPhase.SEND_UNCERTAIN
    await adapter.close()


@pytest.mark.asyncio
async def test_unary_jsonrpc_body_is_bounded_before_parsing() -> None:
    raw_card = _card_bytes()
    ledger = _FakeLedger()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b"{" + b"x" * MAX_OBSERVATION_BYTES + b"}",
        )

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    request = _request(adapter)
    unbound = ledger.prepare(request)
    ledger.mark_executing(unbound)
    handle = ledger.bind_task(unbound, "task-1")

    with pytest.raises(RemoteA2AClientError, match="jsonrpc-response-too-large"):
        await adapter.get_task(handle)
    await adapter.close()


@pytest.mark.asyncio
async def test_cancel_intent_is_anchored_before_cancel_task() -> None:
    raw_card = _card_bytes()
    ledger = _FakeLedger()
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        body = json.loads(request.content)
        methods.append(body["method"])
        assert ledger.order[-1] == "cancel_requested"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=_rpc_response(body["id"], _task(state="TASK_STATE_CANCELED")),
        )

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )
    request = _request(adapter)
    unbound = ledger.prepare(request)
    ledger.mark_executing(unbound)
    handle = ledger.bind_task(unbound, "task-1")

    snapshot = await adapter.cancel(handle)

    assert methods == ["CancelTask"]
    assert snapshot.cancel_requested is True
    assert snapshot.state is RemoteTaskState.CANCELED
    await adapter.close()
