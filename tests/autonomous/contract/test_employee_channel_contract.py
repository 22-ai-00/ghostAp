from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import textwrap
from datetime import UTC, datetime

import pytest

from src.autonomous.ingress.models import (
    EmployeeIngressAck,
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
    IngressAcceptance,
)
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning import channel_protocol as channel_protocol_module
from src.autonomous.provisioning.channel_protocol import (
    MAX_FRAME_BYTES,
    ChannelFrame,
    FrameType,
    ProtocolError,
    decode_frame,
    encode_frame,
)
from src.autonomous.provisioning.channel_worker import (
    WorkerSecurityError,
    _fetch_employee_bot_open_id,
    _handle_low_level_outbound,
    _normalize_sdk_ingress,
    run_low_level_employee_channel,
)
from src.autonomous.provisioning.channel_worker import (
    main as channel_worker_main,
)
from src.autonomous.supervisor.employee_channels import EmployeeChannelSupervisor


def test_sdk_p2p_message_preserves_transport_chat_type_and_union_identity() -> None:
    from types import SimpleNamespace

    event = SimpleNamespace(
        header=SimpleNamespace(
            event_id="evt_owner_p2p",
            event_type="im.message.receive_v1",
            tenant_key="tenant_1",
            app_id="cli_alpha",
            create_time="1783987200000",
        ),
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="om_owner_p2p",
                chat_id="oc_owner_p2p",
                root_id="",
                parent_id="",
                content='{"text":"run it"}',
                mentions=(),
                message_type="text",
                chat_type="p2p",
                thread_id="",
            ),
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(
                    open_id="ou_employee_app_owner",
                    union_id="on_owner",
                ),
                sender_type="user",
                tenant_key="tenant_1",
            ),
        ),
    )

    metadata, payload, _correlation = _normalize_sdk_ingress(
        event,
        kind="message",
        agent_id="agt_alpha",
        app_id="cli_alpha",
        generation=3,
        connection_id="conn_alpha",
        tenant_key="tenant_1",
        bot_principal_id="bot_alpha",
    )

    part = payload.normalized_parts[0]
    assert part["chat_type"] == "p2p"
    assert part["sender_union_id"] == "on_owner"
    assert part["sender_id"] == "ou_employee_app_owner"
    assert metadata.sender_principal_id == "ou_employee_app_owner"
    assert metadata.chat_id != "oc_owner_p2p"


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"code": 0, "bot": {"open_id": "invalid"}},
        {
            "code": 0,
            "bot": {"app_id": "cli_other", "open_id": "ou_employee"},
        },
    ),
)
def test_worker_fails_closed_on_untrusted_bot_identity(payload: object) -> None:
    from lark_oapi.core.model.base_response import BaseResponse
    from lark_oapi.core.model.raw_response import RawResponse

    response = BaseResponse()
    response.code = 0
    response.raw = RawResponse()
    response.raw.status_code = 200
    response.raw.content = json.dumps(payload).encode()
    client = type("Client", (), {"request": lambda _self, _request: response})()

    with pytest.raises(WorkerSecurityError, match="identity response is invalid"):
        _fetch_employee_bot_open_id(client, expected_app_id="cli_employee")


def test_parent_durable_ingress_call_graph_excludes_router_and_acp_execution() -> None:
    source = "\n".join(
        textwrap.dedent(inspect.getsource(method))
        for method in (
            EmployeeChannelSupervisor._accept_ingress,
            EmployeeIngressService.accept,
        )
    )
    tree = ast.parse(source)
    invoked = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }

    assert invoked.isdisjoint(
        {"route", "execute", "start_session", "ensure_session", "_run_acp_session"}
    )
    assert "src.acp" not in source
    assert "provisioning.router" not in source


@pytest.mark.parametrize(
    ("kind", "operation"),
    [("membership_added", "added"), ("membership_deleted", "deleted")],
)
def test_bot_membership_events_normalize_as_durable_ingress(kind, operation) -> None:
    from types import SimpleNamespace

    event = SimpleNamespace(
        header=SimpleNamespace(
            event_id="evt_membership",
            event_type=f"im.chat.member.bot.{operation}_v1",
            tenant_key="tenant_1",
            app_id="cli_alpha",
            create_time="1783987200000",
        ),
        event=SimpleNamespace(
            chat_id="oc_team",
            operator_id=SimpleNamespace(open_id="ou_operator"),
            operator_tenant_key="tenant_1",
            external=False,
            name="Alpha",
        ),
    )

    metadata, payload, correlation = _normalize_sdk_ingress(
        event,
        kind=kind,
        agent_id="agt_alpha",
        app_id="cli_alpha",
        generation=3,
        connection_id="conn_alpha",
        tenant_key="tenant_1",
        bot_principal_id="bot_alpha",
    )

    assert metadata.chat_id.startswith("oc_")
    assert metadata.chat_id != "oc_team"
    assert metadata.sender_principal_id == "ou_operator"
    assert payload.normalized_parts[0]["type"] == "membership_event"
    assert payload.normalized_parts[0]["operation"] == operation
    assert payload.normalized_parts[0]["remote_chat_id"] == "oc_team"
    assert correlation is None


def test_protocol_round_trips_a_strict_versioned_ndjson_frame() -> None:
    frame = ChannelFrame(
        frame_type=FrameType.EVENT,
        agent_id="agt_1",
        generation=7,
        sequence=3,
        payload={"event": "message", "data": {"text": "hello"}},
    )

    encoded = encode_frame(frame)

    assert encoded.endswith(b"\n")
    assert decode_frame(encoded) == frame


def test_update_card_ipc_is_exact_bounded_and_secret_free() -> None:
    frame = ChannelFrame(
        frame_type=FrameType.UPDATE_CARD,
        agent_id="agt_1",
        generation=7,
        sequence=4,
        payload={
            "request_id": "update_1",
            "message_id": "om_employee_card",
            "card": {"schema": "2.0", "body": {"elements": []}},
        },
    )

    assert decode_frame(encode_frame(frame)) == frame
    with pytest.raises(ProtocolError, match="update card"):
        encode_frame(
            ChannelFrame(
                FrameType.UPDATE_CARD,
                "agt_1",
                7,
                5,
                {**frame.payload, "extra": True},
            )
        )


@pytest.mark.parametrize("frame_type", [FrameType.SEND, FrameType.UPDATE_CARD])
def test_low_level_worker_executes_employee_owned_outbound(frame_type: FrameType) -> None:
    from types import SimpleNamespace

    calls: list[tuple[str, tuple, dict]] = []
    emitted: list[tuple[FrameType, dict]] = []

    class _Outbound:
        def send(self, *args, **kwargs):
            calls.append(("send", args, kwargs))
            return SimpleNamespace(success=True, message_id="om_employee")

        def update_card(self, *args, **kwargs):
            calls.append(("update_card", args, kwargs))
            return SimpleNamespace(success=True, message_id="om_employee")

    frame = ChannelFrame(
        frame_type,
        "agt_1",
        3,
        1,
        (
            {
                "request_id": "send_1",
                "target": "oc_team",
                "message": {"text": "hello"},
                "options": None,
            }
            if frame_type is FrameType.SEND
            else {
                "request_id": "update_1",
                "message_id": "om_employee",
                "card": {"schema": "2.0"},
            }
        ),
    )
    bootstrap = SimpleNamespace(app_id="cli_employee", generation=3)
    admission = SimpleNamespace(wait_snapshot=lambda **_kwargs: (1, "conn_employee"))
    emitter = SimpleNamespace(emit=lambda kind, payload: emitted.append((kind, payload)))

    _handle_low_level_outbound(frame, bootstrap, _Outbound(), admission, emitter)

    assert calls and calls[0][0] == (
        "send" if frame_type is FrameType.SEND else "update_card"
    )
    assert emitted == [
        (
            FrameType.HEALTH,
            {
                "operation": (
                    "send" if frame_type is FrameType.SEND else "update_card"
                ),
                "request_id": (
                    "send_1" if frame_type is FrameType.SEND else "update_1"
                ),
                "success": True,
                "app_id": "cli_employee",
                "generation": 3,
                "connection_id": "conn_employee",
                "message_id": "om_employee",
            },
        )
    ]
    with pytest.raises(ProtocolError, match="credential material"):
        encode_frame(
            ChannelFrame(
                FrameType.UPDATE_CARD,
                "agt_1",
                7,
                6,
                {**frame.payload, "card": {"token": "forbidden"}},
            )
        )


@pytest.mark.parametrize(
    "secret_key",
    ["AccessToken", "APIKey", "ClientSecret", "private-key", "PASSWORD"],
)
def test_ordinary_ipc_recursively_rejects_collapsed_secret_aliases(
    secret_key: str,
) -> None:
    frame = ChannelFrame(
        FrameType.EVENT,
        "agt_1",
        1,
        1,
        {"event": "fixture", "data": {"nested": {secret_key: "sentinel"}}},
    )

    with pytest.raises(ProtocolError, match="credential material"):
        encode_frame(frame)


@pytest.mark.parametrize(
    "ordinary_key",
    ["authorization_type", "access_token_expires_at", "password_policy"],
)
def test_ordinary_ipc_allows_non_secret_metadata_with_secret_words(
    ordinary_key: str,
) -> None:
    frame = ChannelFrame(
        FrameType.EVENT,
        "agt_1",
        1,
        1,
        {"event": "fixture", "data": {"nested": {ordinary_key: "safe"}}},
    )

    assert decode_frame(encode_frame(frame)) == frame


@pytest.mark.parametrize("secret_key", ["APIKey", "AccessToken"])
def test_ordinary_ipc_rejects_secret_inside_tuple_before_json_encode(
    secret_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encode_called = False

    def reject_unexpected_encode(_value: object) -> bytes:
        nonlocal encode_called
        encode_called = True
        raise AssertionError("secret-bearing tuple reached JSON encoder")

    monkeypatch.setattr(channel_protocol_module, "_encode", reject_unexpected_encode)
    frame = ChannelFrame(
        FrameType.EVENT,
        "agt_1",
        1,
        1,
        {
            "event": "fixture",
            "data": ({"safe": ({secret_key: "sentinel"},)},),
        },
    )

    with pytest.raises(ProtocolError, match="credential material"):
        encode_frame(frame)
    assert encode_called is False


def test_ordinary_ipc_legal_tuple_metadata_round_trips_as_json_list() -> None:
    frame = ChannelFrame(
        FrameType.EVENT,
        "agt_1",
        1,
        1,
        {
            "event": "fixture",
            "data": (
                {"authorization_type": "tenant"},
                ({"access_token_expires_at": 3600},),
            ),
        },
    )

    decoded = decode_frame(encode_frame(frame))

    assert decoded.payload == {
        "event": "fixture",
        "data": [
            {"authorization_type": "tenant"},
            [{"access_token_expires_at": 3600}],
        ],
    }


def _transport_contract() -> tuple[
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
    EmployeeIngressAck,
]:
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_channel_contract",
        normalized_parts=({"kind": "text", "text": "hello"},),
        attachment_descriptors=(),
    )
    digest = hashlib.sha256(payload.canonical_bytes).hexdigest()
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key="tenant_contract",
        agent_id="agt_channel_contract",
        bot_principal_id="bot_channel_contract",
        app_id="cli_channel_contract",
        channel_generation=7,
        connection_id="conn_channel_contract",
        event_id="evt_channel_contract",
        message_id="om_channel_contract",
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_channel_contract",
        thread_root_message_id="",
        sender_principal_id="ou_sender",
        received_at="2026-07-13T00:00:00Z",
        semantic_digest=digest,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    acceptance = IngressAcceptance(
        schema_version=1,
        acceptance_id="acc_channel_contract",
        envelope_id=payload.envelope_id,
        dedup_key=metadata.dedup_key,
        semantic_digest=metadata.semantic_digest,
        journal_sequence=9,
        journal_frame_hash="a" * 64,
        accepted_at="2026-07-13T00:00:00Z",
    )
    ack = EmployeeIngressAck(
        schema_version=1,
        request_id="req_channel_contract",
        acceptance=acceptance,
        agent_id=metadata.agent_id,
        app_id=metadata.app_id,
        channel_generation=metadata.channel_generation,
        connection_id=metadata.connection_id,
        semantic_digest=metadata.semantic_digest,
        duplicate=False,
        acknowledged_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return metadata, payload, ack


def test_protocol_round_trips_strict_ingress_and_canonical_ack_frames() -> None:
    metadata, payload, ack = _transport_contract()
    ingress = ChannelFrame(
        frame_type=FrameType.INGRESS,
        agent_id=metadata.agent_id,
        generation=metadata.channel_generation,
        sequence=4,
        payload={
            "request_id": ack.request_id,
            "app_id": metadata.app_id,
            "connection_id": metadata.connection_id,
            "metadata": metadata.to_dict(),
            "payload": payload.to_dict(),
            "action_correlation": None,
        },
    )
    ingress_ack = ChannelFrame(
        frame_type=FrameType.INGRESS_ACK,
        agent_id=metadata.agent_id,
        generation=metadata.channel_generation,
        sequence=5,
        payload={
            "request_id": ack.request_id,
            "app_id": ack.app_id,
            "connection_id": ack.connection_id,
            "ack": ack.to_dict(),
        },
    )

    assert decode_frame(encode_frame(ingress)) == ingress
    assert decode_frame(encode_frame(ingress_ack)) == ingress_ack


@pytest.mark.parametrize(
    ("frame_kind", "mutation"),
    [
        ("ingress", lambda value: value["payload"].update({"unknown": True})),
        ("ingress", lambda value: value["payload"]["metadata"].update({"app_secret": "x"})),
        ("ingress", lambda value: value.update({"generation": 8})),
        ("ingress", lambda value: value["payload"].update({"connection_id": "conn_other"})),
        ("ack", lambda value: value["payload"]["ack"].update({"request_id": "req_other"})),
        ("ack", lambda value: value.update({"agent_id": "agt_other"})),
        ("ack", lambda value: value.update({"generation": 8})),
        ("ack", lambda value: value["payload"]["ack"].update({"connection_id": "conn_other"})),
    ],
)
def test_protocol_rejects_malformed_stale_or_cross_owner_ingress_frames(
    frame_kind: str,
    mutation,
) -> None:
    metadata, payload, ack = _transport_contract()
    value = {
        "v": 1,
        "type": "INGRESS" if frame_kind == "ingress" else "INGRESS_ACK",
        "agent_id": metadata.agent_id,
        "generation": metadata.channel_generation,
        "sequence": 1,
        "payload": (
            {
                "request_id": ack.request_id,
                "app_id": metadata.app_id,
                "connection_id": metadata.connection_id,
                "metadata": metadata.to_dict(),
                "payload": payload.to_dict(),
                "action_correlation": None,
            }
            if frame_kind == "ingress"
            else {
                "request_id": ack.request_id,
                "app_id": ack.app_id,
                "connection_id": ack.connection_id,
                "ack": ack.to_dict(),
            }
        ),
    }
    mutation(value)

    with pytest.raises(ProtocolError):
        decode_frame((json.dumps(value, separators=(",", ":")) + "\n").encode())


@pytest.mark.parametrize(
    "raw",
    [
        {"v": 2, "type": "READY", "agent_id": "agt_1", "generation": 1, "sequence": 1, "payload": {}},
        {"v": 1, "type": "UNKNOWN", "agent_id": "agt_1", "generation": 1, "sequence": 1, "payload": {}},
        {"v": 1, "type": "READY", "agent_id": "agt_1", "generation": 1, "sequence": 1, "payload": {}, "extra": True},
    ],
)
def test_protocol_rejects_wrong_version_type_and_unknown_fields(raw: dict[str, object]) -> None:
    with pytest.raises(ProtocolError):
        decode_frame((json.dumps(raw) + "\n").encode())


def test_protocol_rejects_oversized_and_multiline_frames() -> None:
    with pytest.raises(ProtocolError):
        decode_frame(b"{}\n{}\n")
    with pytest.raises(ProtocolError):
        decode_frame(b"x" * (MAX_FRAME_BYTES + 1))


def _direct_bot_mention_event():
    from types import SimpleNamespace

    return SimpleNamespace(
        header=SimpleNamespace(
            event_id="event-mention",
            event_type="im.message.receive_v1",
            create_time="1783900800000",
            tenant_key="tenant_1",
            app_id="cli_employee",
        ),
        event=SimpleNamespace(
            sender=SimpleNamespace(
                sender_id=SimpleNamespace(
                    open_id="ou_employee_app_admin",
                    union_id="on_admin",
                ),
                sender_type="user",
                tenant_key="tenant_1",
            ),
            message=SimpleNamespace(
                message_id="om_mention",
                root_id="",
                parent_id="",
                thread_id="",
                chat_id="oc_team",
                chat_type="group",
                message_type="text",
                content='{"text":"@_user_1 /task ship it"}',
                mentions=(
                    SimpleNamespace(
                        key="@_user_1",
                        mentioned_type="bot",
                        id=SimpleNamespace(open_id="ou_employee_bot"),
                        tenant_key="tenant_1",
                    ),
                ),
            ),
        ),
    )


def _normalized_direct_bot_mention():
    return _normalize_sdk_ingress(
        _direct_bot_mention_event(),
        kind="message",
        agent_id="agt_employee",
        app_id="cli_employee",
        generation=7,
        connection_id="conn_employee",
        tenant_key="tenant_1",
        bot_principal_id="bot_employee",
    )


def test_worker_normalizes_direct_bot_mentions_inside_encrypted_payload() -> None:
    _metadata, payload, _correlation = _normalized_direct_bot_mention()

    part = payload.normalized_parts[0]
    assert part["sender_union_id"] == "on_admin"
    assert part["content"] == {"text": "@_user_1 /task ship it"}
    assert part["remote_chat_id"] == "oc_team"
    assert part["remote_message_id"] == "om_mention"
    assert part["remote_root_id"] == ""
    assert part["mentions"] == (
        {
            "key": "@_user_1",
            "mentioned_type": "bot",
            "open_id": "ou_employee_bot",
            "tenant_key": "tenant_1",
        },
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("content", "text"), "@_user_1 /task altered"),
        (("mentions", 0, "open_id"), "ou_other_bot"),
        (("mentions", 0, "tenant_key"), "tenant_other"),
        (("mentions", 0, "key"), "@_user_2"),
        (("mentions", 0, "mentioned_type"), "user"),
        (("sender_union_id",), "on_other"),
        (("remote_chat_id",), "oc_other"),
        (("remote_message_id",), "om_other"),
        (("remote_root_id",), "om_other_root"),
    ),
)
def test_message_identity_and_coordinates_are_bound_to_payload_digest(
    path: tuple[str | int, ...],
    replacement: str,
) -> None:
    _metadata, payload, _correlation = _normalized_direct_bot_mention()
    mutated = copy.deepcopy(payload.to_dict())
    target = mutated["normalized_parts"][0]
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = replacement

    mutated_payload = EmployeeIngressPayload.from_dict(mutated)

    assert mutated_payload.payload_sha256 != payload.payload_sha256


def test_encrypted_ingress_restart_replay_preserves_mention_and_union_identity(
    tmp_path,
) -> None:
    metadata, payload, _correlation = _normalized_direct_bot_mention()
    anchor = MemoryAnchor()
    key = b"channel-contract-ingress-key-32b"
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=b"channel-contract-hmac-key-32byte",
        writer_epoch=1,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "ingress-blobs",
            AesGcmEncryptionProvider(lambda _ref: key),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    ack = service.accept(metadata, payload, request_id="req_channel_identity")
    acceptance_id = ack.acceptance.acceptance_id
    service.close()
    writer.close()

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=b"channel-contract-hmac-key-32byte",
        writer_epoch=2,
    )
    service = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "ingress-blobs",
            AesGcmEncryptionProvider(lambda _ref: key),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    replayed = service.get_payload(acceptance_id)
    replayed_part = replayed.normalized_parts[0]

    assert replayed.payload_sha256 == payload.payload_sha256
    assert replayed_part["sender_union_id"] == "on_admin"
    assert replayed_part["mentions"] == payload.normalized_parts[0]["mentions"]

    service.close()
    writer.close()


def test_production_worker_main_reaches_only_the_low_level_durable_bridge() -> None:
    source = inspect.getsource(channel_worker_main)

    assert "run_low_level_employee_channel" in source
    assert "asyncio.run" not in source
    assert "create_employee_channel" not in source


def test_low_level_entry_hardens_before_credentials_or_sdk_import() -> None:
    source = inspect.getsource(run_low_level_employee_channel)

    assert source.index("apply_process_hardening()") < source.index(
        "decode_bootstrap"
    )
    assert source.index("emit_macos_sandbox_proof") < source.index(
        "decode_bootstrap"
    )
    assert source.index("collect_sdk_distribution_identity") < source.index(
        "emit_macos_sandbox_proof"
    )
    assert source.index("apply_process_hardening()") < source.index(
        "from lark_channel"
    )


def test_card_action_never_self_attests_user_value_as_trusted_correlation() -> None:
    from lark_channel.event.callback.model.p2_card_action_trigger import (
        P2CardActionTrigger,
    )

    event = P2CardActionTrigger(
        {
            "schema": "2.0",
            "header": {
                "event_id": "external-event-id",
                "event_type": "card.action.trigger",
                "create_time": "1783900800000",
                "app_id": "cli_contract",
                "tenant_key": "tenant-contract",
            },
            "event": {
                "operator": {"open_id": "ou_sender"},
                "action": {
                    "tag": "button",
                    "value": {"correlation_id": "user-controlled"},
                },
                "context": {
                    "open_message_id": "om_external",
                    "open_chat_id": "oc_external",
                },
            },
        }
    )

    metadata, _payload, correlation = _normalize_sdk_ingress(
        event,
        kind="card",
        agent_id="agt_contract",
        app_id="cli_contract",
        generation=2,
        connection_id="conn_contract",
        tenant_key="tenant-contract",
        bot_principal_id="bot_contract",
    )

    assert metadata.action_identity == ""
    assert correlation is None
    assert _payload.normalized_parts == (
        {
            "type": "card_action",
            "sender_id": "ou_sender",
            "sender_id_type": "open_id",
            "sender_type": "",
            "sender_tenant_key": "",
            "remote_chat_id": "oc_external",
            "remote_message_id": "om_external",
            "remote_root_id": "",
        },
    )

    event.header.event_id = ""
    with pytest.raises(ValueError, match="trusted event identity"):
        _normalize_sdk_ingress(
            event,
            kind="card",
            agent_id="agt_contract",
            app_id="cli_contract",
            generation=2,
            connection_id="conn_contract",
            tenant_key="tenant-contract",
            bot_principal_id="bot_contract",
        )


def test_card_normalization_never_reads_the_untrusted_action_object() -> None:
    class ExplosiveBody:
        operator = type(
            "Operator",
            (),
            {
                "open_id": "ou_sender",
                "sender_type": "user",
                "tenant_key": "tenant-contract",
            },
        )()
        context = type(
            "Context",
            (),
            {
                "open_message_id": "om_external",
                "open_chat_id": "oc_external",
            },
        )()

        @property
        def action(self):
            raise AssertionError("card action must remain opaque before issuance")

    event = type(
        "Event",
        (),
        {
            "header": type(
                "Header",
                (),
                {
                    "event_id": "external-event-id",
                    "event_type": "card.action.trigger",
                    "create_time": "1783900800000",
                    "app_id": "cli_contract",
                    "tenant_key": "tenant-contract",
                },
            )(),
            "event": ExplosiveBody(),
        },
    )()

    metadata, payload, correlation = _normalize_sdk_ingress(
        event,
        kind="card",
        agent_id="agt_contract",
        app_id="cli_contract",
        generation=2,
        connection_id="conn_contract",
        tenant_key="tenant-contract",
        bot_principal_id="bot_contract",
    )

    assert metadata.event_type == "card.action.trigger"
    assert payload.normalized_parts[0]["type"] == "card_action"
    assert correlation is None
