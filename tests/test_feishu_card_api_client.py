"""Tests for FeishuCardAPIClient request mapping."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.card.delivery.feishu_client import FeishuCardAPIClient


class _FakeResponse:
    code = 0
    msg = ""
    data = SimpleNamespace(message_id="msg_1")

    def success(self) -> bool:
        return True


class _FakeMessageApi:
    def __init__(self) -> None:
        self.created_request = None
        self.replied_request = None

    def create(self, request):
        self.created_request = request
        return _FakeResponse()

    def reply(self, request):
        self.replied_request = request
        return _FakeResponse()


class _FakeCardApi:
    def __init__(self) -> None:
        self.updated_request = None

    def update(self, request):
        self.updated_request = request
        return _FakeResponse()


def _client_for(message_api: _FakeMessageApi):
    return SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=message_api)))


def test_create_card_sets_feishu_uuid_for_direct_message() -> None:
    message_api = _FakeMessageApi()
    client = FeishuCardAPIClient(_client_for(message_api))

    client.create_card("chat_1", {"body": {}}, idempotency_key="idem-1")

    body = message_api.created_request.request_body
    assert body.receive_id == "chat_1"
    assert body.msg_type == "interactive"
    assert body.uuid == "idem-1"


def test_create_card_sets_feishu_uuid_for_reply_message() -> None:
    message_api = _FakeMessageApi()
    client = FeishuCardAPIClient(_client_for(message_api))

    client.create_card("chat_1", {"body": {}}, reply_to="origin_1", idempotency_key="idem-2")

    assert message_api.replied_request.message_id == "origin_1"
    body = message_api.replied_request.request_body
    assert body.msg_type == "interactive"
    assert body.uuid == "idem-2"


def test_send_card_reference_sets_feishu_uuid() -> None:
    message_api = _FakeMessageApi()
    client = FeishuCardAPIClient(_client_for(message_api))

    client.send_card_reference("chat_1", "card_1", idempotency_key="idem-3")

    body = message_api.created_request.request_body
    assert json.loads(body.content) == {"type": "card", "data": {"card_id": "card_1"}}
    assert body.uuid == "idem-3"


def test_card_message_create_reply_and_patch_are_audited_before_sdk_dispatch() -> None:
    message_api = _FakeMessageApi()
    message_api.patch = lambda _request: _FakeResponse()
    events: list[tuple[str, str, str]] = []
    client = FeishuCardAPIClient(
        _client_for(message_api),
        outbound_audit=lambda tenant, operation, target: events.append(
            (tenant, operation, target)
        ),
        tenant_key_resolver=lambda: "tenant-a",
        outbound_target_aliases=lambda _target: ("chat_1", "ou_requester"),
    )

    client.create_card("chat_1", {"body": {}})
    client.create_card("chat_1", {"body": {}}, reply_to="origin_1")
    client.update_card("card_1", {"body": {}})
    client.send_card_reference("chat_1", "card_entity")
    client.send_card_reference("chat_1", "card_entity", reply_to="origin_2")

    assert events == [
        ("tenant-a", "create", "chat_1"),
        ("tenant-a", "reply", "origin_1"),
        ("tenant-a", "reply", "chat_1"),
        ("tenant-a", "reply", "ou_requester"),
        ("tenant-a", "patch", "card_1"),
        ("tenant-a", "patch", "chat_1"),
        ("tenant-a", "patch", "ou_requester"),
        ("tenant-a", "create", "chat_1"),
        ("tenant-a", "reply", "origin_2"),
        ("tenant-a", "reply", "chat_1"),
        ("tenant-a", "reply", "ou_requester"),
    ]


@pytest.mark.parametrize(
    "case",
    (
        "create",
        "reply",
        "patch",
        "element_patch",
        "reference_create",
        "reference_reply",
    ),
)
def test_card_audit_failure_blocks_every_message_sdk_dispatch(case: str) -> None:
    message_api = _FakeMessageApi()
    failures: list[Exception] = []
    client = FeishuCardAPIClient(
        _client_for(message_api),
        outbound_audit=lambda *_args: (_ for _ in ()).throw(OSError("audit disk")),
        outbound_audit_failure=failures.append,
        outbound_target_aliases=lambda _target: ("chat_1", "ou_requester"),
    )
    client._call_api = MagicMock(side_effect=AssertionError("SDK must not run"))

    with pytest.raises(OSError, match="audit disk"):
        if case == "create":
            client.create_card("chat_1", {"body": {}})
        elif case == "reply":
            client.create_card("chat_1", {"body": {}}, reply_to="origin_1")
        elif case == "patch":
            client.update_card("card_1", {"body": {}})
        elif case == "element_patch":
            client.update_element("card_1", "element_1", "text")
        elif case == "reference_create":
            client.send_card_reference("chat_1", "card_entity")
        else:
            client.send_card_reference(
                "chat_1",
                "card_entity",
                reply_to="origin_1",
            )

    client._call_api.assert_not_called()
    assert len(failures) == 1


def test_created_card_remembers_recipient_aliases_for_later_patch() -> None:
    message_api = _FakeMessageApi()
    message_api.patch = lambda _request: _FakeResponse()
    events: list[tuple[str, str]] = []
    resolver_calls = 0

    def resolve_aliases(_target: str) -> tuple[str, ...]:
        nonlocal resolver_calls
        resolver_calls += 1
        if resolver_calls > 1:
            raise RuntimeError("provenance store unavailable")
        return ("oc_requester_dm", "ou_requester")

    client = FeishuCardAPIClient(
        _client_for(message_api),
        outbound_audit=lambda _tenant, operation, target: events.append(
            (operation, target)
        ),
        outbound_target_aliases=resolve_aliases,
    )

    message_id, _card_id = client.create_card(
        "oc_requester_dm",
        {"body": {}},
        reply_to="om_origin",
    )
    client.update_card(message_id, {"body": {}})

    assert resolver_calls == 1
    assert events[-4:] == [
        ("patch", message_id),
        ("patch", "om_origin"),
        ("patch", "oc_requester_dm"),
        ("patch", "ou_requester"),
    ]


def test_streaming_card_entity_update_uses_cardkit_and_remembers_aliases() -> None:
    """CardKit structure updates target the entity, not its carrier message."""
    message_api = _FakeMessageApi()
    message_api.patch = MagicMock(return_value=_FakeResponse())
    card_api = _FakeCardApi()
    element_api = SimpleNamespace(content=lambda _request: _FakeResponse())
    sdk_client = _client_for(message_api)
    sdk_client.cardkit = SimpleNamespace(
        v1=SimpleNamespace(card=card_api, card_element=element_api),
    )
    events: list[tuple[str, str]] = []
    resolver_calls = 0

    def resolve_aliases(_target: str) -> tuple[str, ...]:
        nonlocal resolver_calls
        resolver_calls += 1
        if resolver_calls > 1:
            raise RuntimeError("provenance store unavailable")
        return ("oc_requester_dm", "ou_requester")

    client = FeishuCardAPIClient(
        sdk_client,
        outbound_audit=lambda _tenant, operation, target: events.append(
            (operation, target)
        ),
        outbound_target_aliases=resolve_aliases,
    )

    message_id = client.send_card_reference(
        "oc_requester_dm",
        "card_entity_1",
        reply_to="om_origin",
    )
    client.update_card("card_entity_1", {"body": {}})
    client.update_element("card_entity_1", "element_1", "done")

    assert message_id == "msg_1"
    assert card_api.updated_request.card_id == "card_entity_1"
    assert card_api.updated_request.request_body.sequence == 0
    assert json.loads(card_api.updated_request.request_body.card.data) == {"body": {}}
    message_api.patch.assert_not_called()
    assert resolver_calls == 1
    assert events[-4:] == [
        ("patch", "card_entity_1"),
        ("patch", "om_origin"),
        ("patch", "oc_requester_dm"),
        ("patch", "ou_requester"),
    ]


def test_call_api_times_out_without_waiting_for_stuck_sdk_call(monkeypatch) -> None:
    message_api = _FakeMessageApi()
    client = FeishuCardAPIClient(_client_for(message_api))
    release = threading.Event()

    monkeypatch.setattr(client, "_api_timeout_seconds", lambda: 0.01)

    started_at = time.monotonic()
    with pytest.raises(TimeoutError):
        client._call_api("slow.test", lambda: release.wait(timeout=1.0))

    assert time.monotonic() - started_at < 0.5
    release.set()


def test_call_api_rejects_when_worker_slots_are_exhausted(monkeypatch) -> None:
    message_api = _FakeMessageApi()
    client = FeishuCardAPIClient(_client_for(message_api))
    release = threading.Event()
    slots = threading.BoundedSemaphore(1)

    monkeypatch.setattr(FeishuCardAPIClient, "_worker_slots", slots)
    monkeypatch.setattr(client, "_api_timeout_seconds", lambda: 0.01)

    with pytest.raises(TimeoutError):
        client._call_api("slow.first", lambda: release.wait(timeout=1.0))

    started_at = time.monotonic()
    with pytest.raises(TimeoutError, match="worker slots exhausted"):
        client._call_api("slow.second", lambda: "late")
    assert time.monotonic() - started_at < 0.5
    release.set()
