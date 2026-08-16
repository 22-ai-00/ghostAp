from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.autonomous.a2a.client import RemoteA2ADispatchAdapter
from src.autonomous.remote.models import RemoteTaskState
from tests.autonomous.unit.test_a2a_client import (
    _TOKEN,
    _allow_mock_peer,
    _card_bytes,
    _FakeLedger,
    _public_dns,
    _registration,
    _request,
    _Resolver,
    _rpc_response,
    _sse,
    _task,
)


@pytest.mark.asyncio
async def test_official_sdk_jsonrpc_1_0_wire_supports_required_pull_operations() -> None:
    """Exercise the real a2a-sdk ClientFactory/BaseClient transport contract."""

    raw_card = _card_bytes()
    ledger = _FakeLedger()
    requests: list[tuple[httpx.Request, dict[str, Any]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )

        body = json.loads(request.content)
        requests.append((request, body))
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        assert request.headers["a2a-version"] == "1.0"
        if body["method"] == "SendStreamingMessage":
            return _sse(
                body["id"],
                {"task": _task(state="TASK_STATE_SUBMITTED")},
            )
        if body["method"] == "GetTask":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json=_rpc_response(
                    body["id"],
                    _task(state="TASK_STATE_WORKING"),
                ),
            )
        if body["method"] == "SubscribeToTask":
            return _sse(
                body["id"],
                {
                    "statusUpdate": {
                        "taskId": "task-1",
                        "contextId": "ctx-1",
                        "status": {"state": "TASK_STATE_WORKING"},
                    }
                },
            )
        if body["method"] == "CancelTask":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json=_rpc_response(
                    body["id"],
                    _task(state="TASK_STATE_CANCELED"),
                ),
            )
        raise AssertionError(f"unexpected A2A method: {body['method']}")

    adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        ledger,
        _Resolver([]),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=httpx.MockTransport(handler),
    )

    dispatch_observations = [observation async for observation in adapter.dispatch(_request(adapter))]
    handle = ledger.snapshot("acceptance-1").handle
    subscribe_observations = [observation async for observation in adapter.subscribe(handle)]
    cancelled = await adapter.cancel(handle)

    methods = [body["method"] for _request_value, body in requests]
    assert methods == [
        "SendStreamingMessage",
        "GetTask",
        "SubscribeToTask",
        "CancelTask",
    ]
    assert all(body["jsonrpc"] == "2.0" for _request_value, body in requests)
    assert requests[1][1]["params"] == {"id": "task-1", "historyLength": 20}
    assert requests[2][1]["params"] == {"id": "task-1"}
    assert requests[3][1]["params"] == {"id": "task-1"}
    assert dispatch_observations[-1].state is RemoteTaskState.WORKING
    assert subscribe_observations[-1].state is RemoteTaskState.WORKING
    assert cancelled.state is RemoteTaskState.CANCELED
    assert not any("PushNotification" in method for method in methods)
    await adapter.close()
