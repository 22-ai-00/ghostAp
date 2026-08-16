from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.autonomous.a2a.client import RemoteA2ADispatchAdapter
from src.autonomous.a2a.journal import RemoteDispatchLedger
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.remote.models import (
    RemoteAttemptPhase,
    RemoteDispatchRequest,
    RemoteTaskState,
)
from src.autonomous.remote.projection import EVENT_OBSERVATION, EVENT_TASK_BOUND
from tests.autonomous.unit.test_a2a_client import (
    _TOKEN,
    _allow_mock_peer,
    _card_bytes,
    _public_dns,
    _registration,
    _rpc_response,
    _sse,
    _task,
)

_HMAC_KEY = b"a2a-client-integration-hmac-key-32"
_DATA_KEY = b"i" * 32


class _Resolver:
    async def resolve_bearer(self, _descriptor: object) -> str:
        return _TOKEN


def _open_stack(
    root: Path,
    anchor: FileAnchor,
    *,
    epoch: int,
) -> tuple[RemoteDispatchLedger, JournalWriter, BlobStore]:
    store = BlobStore(
        root / "blobs",
        AesGcmEncryptionProvider(lambda _key_ref: _DATA_KEY),
    )
    writer = JournalWriter.open(
        root / "journal",
        anchor=anchor,
        hmac_key=_HMAC_KEY,
        writer_epoch=epoch,
        blob_ref_validator=lambda ref: _is_published(store, ref),
    )
    return RemoteDispatchLedger(writer, store, "a2a-key-v1"), writer, store


def _is_published(store: BlobStore, ref: object) -> bool:
    try:
        store.read(ref)  # type: ignore[arg-type]
    except Exception:
        return False
    return True


@pytest.mark.asyncio
async def test_restart_with_bound_task_subscribes_without_resending(
    tmp_path: Path,
) -> None:
    raw_card = _card_bytes()
    anchor = FileAnchor(tmp_path / "anchor.json")
    methods: list[str] = []
    card_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            card_requests.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=raw_card,
            )
        body = json.loads(request.content)
        methods.append(body["method"])
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        if body["method"] == "SendStreamingMessage":
            return _sse(
                body["id"],
                {"task": _task(state="TASK_STATE_SUBMITTED")},
            )
        if body["method"] == "SubscribeToTask":
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
        if body["method"] == "GetTask":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json=_rpc_response(
                    body["id"],
                    _task(state="TASK_STATE_COMPLETED"),
                ),
            )
        raise AssertionError(f"unexpected method: {body['method']}")

    transport = httpx.MockTransport(handler)
    first_ledger, first_writer, first_store = _open_stack(
        tmp_path,
        anchor,
        epoch=1,
    )
    first_adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        first_ledger,
        _Resolver(),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=transport,
    )
    request = RemoteDispatchRequest(
        acceptance_id="acceptance-1",
        run_id="run-1",
        assignment_id="assignment-1",
        attempt_id="attempt-1",
        message_id="message-1",
        context_id="ctx-1",
        instruction="secret review instruction",
        descriptor=first_adapter.descriptor,
    )
    stream = first_adapter.dispatch(request)
    first = await anext(stream)
    assert first.state is RemoteTaskState.SUBMITTED
    await stream.aclose()
    await first_adapter.close()
    first_writer.close()
    first_store.close()

    second_ledger, second_writer, second_store = _open_stack(
        tmp_path,
        anchor,
        epoch=2,
    )
    second_adapter = await RemoteA2ADispatchAdapter.create(
        _registration(raw_card),
        second_ledger,
        _Resolver(),
        dns_resolver=_public_dns,
        response_peer_verifier=_allow_mock_peer,
        transport=transport,
    )
    try:
        recovered = [item async for item in second_adapter.dispatch(request)]
        snapshot = second_ledger.snapshot(request.acceptance_id)

        assert methods == ["SendStreamingMessage", "SubscribeToTask"]
        assert recovered[-1].state is RemoteTaskState.CLAIMED_COMPLETED
        assert snapshot is not None
        assert snapshot.phase is RemoteAttemptPhase.TERMINAL
        event_types = [event.event_type for frame in second_writer.replay() for event in frame.events]
        assert event_types.index(EVENT_TASK_BOUND) < event_types.index(EVENT_OBSERVATION)
        journal_bytes = second_writer.journal_path.read_bytes()
        assert request.instruction.encode() not in journal_bytes
        assert _TOKEN.encode() not in journal_bytes
        assert all("authorization" not in item.headers for item in card_requests)
    finally:
        await second_adapter.close()
        second_writer.close()
        second_store.close()
