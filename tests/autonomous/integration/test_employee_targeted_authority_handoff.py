"""Fail-closed handoff coverage for frozen targeted-task authority."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from src.autonomous.authorization import EmployeeAuthorizationScope
from src.autonomous.ingress.models import EmployeeIngressMetadata, EmployeeIngressPayload
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.router import RouterAuthoritySnapshot, RouterLifecycleRecord
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.ingress.targeted_task import (
    TARGETED_TASK_INPUT_KIND,
    targeted_group_task_digest,
)
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning import composition as employee_composition
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

_TENANT_KEY = "tenant_1"
_AGENT_ID = "agt_alpha"
_BOT_PRINCIPAL_ID = "bot_alpha"
_APP_ID = "cli_alpha"
_GENERATION = 3
_CONNECTION_ID = "conn_alpha"
_RAW_CHAT_ID = "oc_targeted_group"
_RAW_MESSAGE_ID = "om_targeted_authority"
_RAW_ROOT_ID = "om_targeted_root"
_TARGET_OPEN_ID = "ou_bot_alpha"
_DESCRIPTION = "finish the targeted audit"


@contextmanager
def _accepted_targeted_message(
    tmp_path: Path,
) -> Iterator[tuple[EmployeeIngressService, EmployeeIngressMetadata, str]]:
    part = {
        "type": "message",
        "message_type": "text",
        "chat_type": "group",
        "content": {"text": f"@_user_1 /task {_DESCRIPTION}"},
        "sender_id": "ou_requester",
        "sender_union_id": "on_requester",
        "sender_id_type": "open_id",
        "sender_type": "user",
        "sender_tenant_key": _TENANT_KEY,
        "feishu_thread_id": "omt_targeted_authority",
        "mentions": (
            {
                "key": "@_user_1",
                "open_id": _TARGET_OPEN_ID,
                "tenant_key": _TENANT_KEY,
            },
        ),
        "remote_chat_id": _RAW_CHAT_ID,
        "remote_message_id": _RAW_MESSAGE_ID,
        "remote_root_id": _RAW_ROOT_ID,
    }
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + hashlib.sha256(b"targeted-authority").hexdigest(),
        normalized_parts=(part,),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key=_TENANT_KEY,
        agent_id=_AGENT_ID,
        bot_principal_id=_BOT_PRINCIPAL_ID,
        app_id=_APP_ID,
        channel_generation=_GENERATION,
        connection_id=_CONNECTION_ID,
        event_id="evt_targeted_authority",
        message_id="om_" + hashlib.sha256(_RAW_MESSAGE_ID.encode()).hexdigest(),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_" + hashlib.sha256(_RAW_CHAT_ID.encode()).hexdigest(),
        thread_root_message_id="om_"
        + hashlib.sha256(_RAW_ROOT_ID.encode()).hexdigest(),
        sender_principal_id="ou_requester",
        received_at="2026-08-12T00:00:00.000000Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=MemoryAnchor(),
        hmac_key=b"t" * 32,
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"i" * 32),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="key-active",
    )
    acceptance_id = ingress.accept(
        metadata,
        payload,
        request_id="req_targeted_authority",
    ).acceptance.acceptance_id
    assert ingress.record_snapshot(acceptance_id) is not None
    try:
        yield ingress, metadata, acceptance_id
    finally:
        ingress.close()
        writer.close()


def _authority() -> RouterAuthoritySnapshot:
    return RouterAuthoritySnapshot(
        tenant_key=_TENANT_KEY,
        agent_id=_AGENT_ID,
        bot_principal_id=_BOT_PRINCIPAL_ID,
        app_id=_APP_ID,
        channel_generation=_GENERATION,
        connection_id=_CONNECTION_ID,
        authorization_scope=EmployeeAuthorizationScope.MANAGED_GROUP,
        team_id=_RAW_CHAT_ID,
        requester_principal_id="ou_owner",
        projection_sequence=7,
        projection_hash="a" * 64,
        employee_version=1,
        tool="codex",
        model="",
        effort="high",
        effective_input_kind=TARGETED_TASK_INPUT_KIND,
        effective_input_digest=targeted_group_task_digest(_DESCRIPTION),
        target_bot_open_id_digest=hashlib.sha256(
            _TARGET_OPEN_ID.encode()
        ).hexdigest(),
    )


def _terminal_record(
    ingress: EmployeeIngressService,
    acceptance_id: str,
    authority: RouterAuthoritySnapshot,
) -> RouterLifecycleRecord:
    accepted = ingress.record_snapshot(acceptance_id)
    assert accepted is not None
    return RouterLifecycleRecord(
        aggregate_id=accepted.aggregate_id,
        acceptance_id=acceptance_id,
        envelope_id=accepted.metadata.envelope_id,
        tenant_key=_TENANT_KEY,
        agent_id=_AGENT_ID,
        bot_principal_id=_BOT_PRINCIPAL_ID,
        app_id=_APP_ID,
        channel_generation=_GENERATION,
        connection_id=_CONNECTION_ID,
        team_id=_RAW_CHAT_ID,
        message_id=accepted.metadata.message_id,
        event_type=accepted.metadata.event_type,
        requester_principal_id="ou_owner",
        state="terminal",
        accepted_sequence=accepted.acceptance.journal_sequence,
        authority=authority,
        queued_sequence=23,
        reason_code="authority_stale",
    )


class _FailureResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def task_failure_response(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace()


def _runtime(
    ingress: EmployeeIngressService,
    record: RouterLifecycleRecord,
) -> tuple[EmployeeDepartmentRuntime, _FailureResponses]:
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._closing = False
    runtime._ingress = ingress
    runtime._router = SimpleNamespace(
        record_snapshot=lambda acceptance_id: (
            record if acceptance_id == record.acceptance_id else None
        ),
        rebuild_projection=lambda: None,
        state=SimpleNamespace(by_acceptance_id={record.acceptance_id: record}),
    )
    responses = _FailureResponses()
    runtime._outbox_lifecycle = responses
    return runtime, responses


_AUTHORITY_MUTATIONS = (
    pytest.param({"tenant_key": "tenant_other"}, id="tenant"),
    pytest.param({"agent_id": "agt_other"}, id="agent"),
    pytest.param({"bot_principal_id": "bot_other"}, id="bot"),
    pytest.param({"app_id": "cli_other"}, id="app"),
    pytest.param({"channel_generation": _GENERATION + 1}, id="generation"),
    pytest.param({"connection_id": "conn_other"}, id="connection"),
    pytest.param({"effective_input_digest": "f" * 64}, id="effective-digest"),
    pytest.param({"target_bot_open_id_digest": "f" * 64}, id="target-digest"),
    pytest.param(
        {
            "authorization_scope": EmployeeAuthorizationScope.OWNER_P2P,
            "effective_input_kind": "",
            "effective_input_digest": "",
            "target_bot_open_id_digest": "",
        },
        id="non-targeted-authority",
    ),
)


@pytest.mark.parametrize("mutation", _AUTHORITY_MUTATIONS)
def test_mutated_targeted_authority_neither_claims_handoff_nor_responds(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    with _accepted_targeted_message(tmp_path) as (
        ingress,
        metadata,
        acceptance_id,
    ):
        record = _terminal_record(
            ingress,
            acceptance_id,
            replace(_authority(), **mutation),
        )
        runtime, responses = _runtime(ingress, record)

        assert runtime._employee_handoff_projection_result(  # noqa: SLF001
            acceptance_id=acceptance_id,
            ingress=ingress,
            router=runtime._router,
            channel_generation=metadata.channel_generation,
            connection_id=metadata.connection_id,
        ) is employee_composition._EmployeeHandoffProjection.INVALID_UNKNOWN
        assert runtime._reconcile_terminal_ingress() == 0  # noqa: SLF001

        persisted = ingress.record_snapshot(acceptance_id)
        assert persisted is not None and persisted.disposition is None
        assert responses.calls == []


def test_valid_targeted_authority_claims_handoff_before_failure_response(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_message(tmp_path) as (
        ingress,
        metadata,
        acceptance_id,
    ):
        record = _terminal_record(ingress, acceptance_id, _authority())
        runtime, responses = _runtime(ingress, record)

        assert runtime._employee_handoff_projection_result(  # noqa: SLF001
            acceptance_id=acceptance_id,
            ingress=ingress,
            router=runtime._router,
            channel_generation=metadata.channel_generation,
            connection_id=metadata.connection_id,
        ) is employee_composition._EmployeeHandoffProjection.OWNED
        assert runtime._reconcile_terminal_ingress() == 1  # noqa: SLF001

        persisted = ingress.record_snapshot(acceptance_id)
        assert persisted is not None and persisted.disposition is not None
        assert len(responses.calls) == 1


def test_valid_targeted_queue_uses_canonical_transport_after_ingress_alias_match(
    tmp_path: Path,
) -> None:
    with _accepted_targeted_message(tmp_path) as (
        ingress,
        _metadata,
        acceptance_id,
    ):
        record = replace(
            _terminal_record(ingress, acceptance_id, _authority()),
            state="queued",
            reason_code="",
        )
        runtime, _responses = _runtime(ingress, record)

        # The wait_for_anchored_message_acceptance call has already proven the
        # exact request alias.  Router keeps the canonical transport that was
        # frozen on acceptance, which need not equal that later alias.
        assert runtime._employee_handoff_projection_result(  # noqa: SLF001
            acceptance_id=acceptance_id,
            ingress=ingress,
            router=runtime._router,
            channel_generation=_GENERATION + 1,
            connection_id="conn_later_alias",
        ) is employee_composition._EmployeeHandoffProjection.OWNED


@pytest.mark.parametrize("state", ("queued", "dispatching"))
@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param(
            {"authorization_scope": EmployeeAuthorizationScope.OWNER_P2P},
            id="scope",
        ),
        pytest.param({"effective_input_kind": ""}, id="kind"),
        pytest.param({"effective_input_digest": "f" * 64}, id="input-digest"),
        pytest.param(
            {"target_bot_open_id_digest": "f" * 64},
            id="target-digest",
        ),
    ),
)
def test_targeted_queue_cannot_degrade_to_generic_ownership_after_mutation(
    tmp_path: Path,
    state: str,
    mutation: dict[str, object],
) -> None:
    with _accepted_targeted_message(tmp_path) as (
        ingress,
        metadata,
        acceptance_id,
    ):
        frozen = _authority()
        values = frozen.to_dict()
        values["authorization_scope"] = frozen.authorization_scope
        values.update(mutation)
        # Model an anchored projection mutation directly.  The production
        # parser normally rejects this shape; handoff must still fail closed if
        # a corrupted in-memory record reaches the read side.
        mutated = SimpleNamespace(**values)
        record = replace(
            _terminal_record(ingress, acceptance_id, mutated),
            state=state,
            reason_code="",
        )
        runtime, _responses = _runtime(ingress, record)

        assert runtime._employee_handoff_projection_result(  # noqa: SLF001
            acceptance_id=acceptance_id,
            ingress=ingress,
            router=runtime._router,
            channel_generation=metadata.channel_generation,
            connection_id=metadata.connection_id,
        ) is employee_composition._EmployeeHandoffProjection.INVALID_UNKNOWN
