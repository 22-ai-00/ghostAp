"""Durability, FIFO, isolation, and backpressure tests for employee Router queues."""

from __future__ import annotations

import hashlib
import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from src.autonomous.authorization import EmployeeAuthorizationScope
from src.autonomous.context.runtime import RuntimeRequesterChatAcl
from src.autonomous.domain import BotPrincipal, EmployeeDefinition, EmployeeState, WorkerType
from src.autonomous.ingress.models import EmployeeIngressMetadata, EmployeeIngressPayload
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.journal.frame import GENESIS_HASH, JournalEvent
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.supervisor.employee_channels import ChannelProcessState, ChannelProcessStatus
from src.autonomous.workforce.registry import ProjectedAgentRegistry

HMAC_KEY = b"employee-router-integration-hmac!!"
DATA_KEY = b"i" * 32


def _module():
    return importlib.import_module("src.autonomous.ingress.router")


def _payload(
    index: int,
    *,
    sender: str = "ou_requester",
    chat_type: str = "group",
    sender_union_id: str = "",
    attachment_descriptors: tuple[dict[str, object], ...] = (),
) -> EmployeeIngressPayload:
    part = {
        "type": "message",
        "message_type": "text",
        "chat_type": chat_type,
        "content": {"text": f"task {index}"},
        "sender_id": sender,
        "sender_union_id": sender_union_id,
        "sender_id_type": "open_id",
        "sender_type": "user",
        "sender_tenant_key": "tenant_1",
        "feishu_thread_id": f"omt_{index}",
    }
    digest = hashlib.sha256(str(index).encode()).hexdigest()
    return EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + digest,
        normalized_parts=(part,),
        attachment_descriptors=attachment_descriptors,
    )


def _metadata(
    payload: EmployeeIngressPayload,
    index: int,
    agent_id: str,
    *,
    chat_id: str = "oc_team",
) -> EmployeeIngressMetadata:
    suffix = hashlib.sha256(f"{agent_id}:{index}".encode()).hexdigest()[:24]
    return EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key="tenant_1",
        agent_id=agent_id,
        bot_principal_id=f"bot_{agent_id.removeprefix('agt_')}",
        app_id=f"cli_{agent_id.removeprefix('agt_')}",
        channel_generation=3,
        connection_id=f"conn_{agent_id.removeprefix('agt_')}",
        event_id=f"evt_{suffix}",
        message_id=f"om_{suffix}",
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id=chat_id,
        thread_root_message_id="om_root",
        sender_principal_id="ou_requester",
        received_at="2026-07-13T00:00:00Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=len(payload.attachment_descriptors),
        attachment_total_bytes=payload.attachment_total_bytes,
    )


class _Channels:
    def __init__(self, agent_ids: tuple[str, ...]) -> None:
        self.statuses = {
            agent_id: ChannelProcessStatus(
                agent_id=agent_id,
                app_id=f"cli_{agent_id.removeprefix('agt_')}",
                generation=3,
                pid=index + 100,
                state=ChannelProcessState.READY,
                tenant_key="tenant_1",
                bot_principal_id=f"bot_{agent_id.removeprefix('agt_')}",
                identity={
                    "app_id": f"cli_{agent_id.removeprefix('agt_')}",
                    "open_id": f"ou_bot_{agent_id.removeprefix('agt_')}",
                },
                ready_metadata={"connection_id": f"conn_{agent_id.removeprefix('agt_')}"},
            )
            for index, agent_id in enumerate(agent_ids)
        }

    def status(self, agent_id: str):
        return self.statuses.get(agent_id)


class _HealthyMembership:
    def is_degraded(self, _agent_id: str, _team_id: str) -> bool:
        return False


class _QueueCleanupStaging:
    def __init__(self) -> None:
        self.stage_calls = 0
        self.cleanup_calls: list[str] = []
        self.state = type(
            "State",
            (),
            {"by_acceptance_id": {}, "by_staging_id": {}},
        )()

    def stage(self, request) -> None:
        self.stage_calls += 1
        staging_id = f"stg_queue_{self.stage_calls}"
        self.state.by_acceptance_id[request.acceptance_id] = staging_id
        self.state.by_staging_id[staging_id] = type(
            "Record",
            (),
            {
                "staging_id": staging_id,
                "status": "completed",
                "cleanup_state": "none",
            },
        )()

    def completed_for_acceptance(self, acceptance_id: str):
        staging_id = self.state.by_acceptance_id.get(acceptance_id)
        if staging_id is None:
            return None
        record = self.state.by_staging_id[staging_id]
        return None if record.cleanup_state == "completed" else record

    def cleanup(self, staging_id: str) -> None:
        self.cleanup_calls.append(staging_id)
        self.state.by_staging_id[staging_id].cleanup_state = "completed"


class _SelectiveRejectAnchor(MemoryAnchor):
    def __init__(self) -> None:
        super().__init__()
        self.reject_sequence: int | None = None

    def compare_and_swap(
        self,
        expected_sequence: int,
        expected_hash: str,
        new_sequence: int,
        new_hash: str,
    ) -> bool:
        if new_sequence == self.reject_sequence:
            return False
        return super().compare_and_swap(
            expected_sequence,
            expected_hash,
            new_sequence,
            new_hash,
        )


def test_projection_rebuild_skips_replay_until_journal_head_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module_ref, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    replay_calls: list[int] = []
    original_replay = writer.replay

    def counted_replay(from_sequence: int = 1):
        replay_calls.append(from_sequence)
        return original_replay(from_sequence)

    monkeypatch.setattr(writer, "replay", counted_replay)
    try:
        ingress.rebuild_projection()
        router.rebuild_projection()
        assert replay_calls == []

        aggregate_id = "projection-refresh:test"
        writer.commit(
            (
                JournalEvent(
                    event_type="projection.refresh_requested",
                    aggregate_id=aggregate_id,
                    payload={},
                ),
            ),
            writer.get_aggregate_versions((aggregate_id,)),
        )

        ingress.rebuild_projection()
        router.rebuild_projection()
        assert replay_calls == [1, 1]

        ingress.rebuild_projection()
        router.rebuild_projection()
        assert replay_calls == [1, 1]
    finally:
        ingress.close()
        writer.close()


def _stack(
    tmp_path: Path,
    *,
    agent_ids: tuple[str, ...] = ("agt_alpha",),
    limits: tuple[int, int, int] = (4, 8, 16),
    inactive_agent_ids: tuple[str, ...] = (),
    anchor: MemoryAnchor | None = None,
    attachment_staging: object | None = None,
    requester_acl: object | None = None,
    membership_health: object | None = None,
    requester_principal_resolver=None,
    managed_group_registry_provider=None,
    managed_group_owner_id: str = "",
    employee_bot_ids_provider=None,
):
    module = _module()
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor or MemoryAnchor(),
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _key_ref: DATA_KEY),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    workforce = ProjectionState()
    workforce.cursor_sequence = 0
    workforce.cursor_hash = GENESIS_HASH
    for index, agent_id in enumerate(agent_ids):
        suffix = agent_id.removeprefix("agt_")
        workforce.employees[agent_id] = EmployeeDefinition(
            agent_id=agent_id,
            tenant_key="tenant_1",
            owner_principal_id="ou_owner",
            name=suffix,
            tool="codex",
            model="gpt-5.6-sol",
            effort="xhigh",
            worker_type=WorkerType.VISIBLE,
            state=(
                EmployeeState.DRAFT
                if agent_id in inactive_agent_ids
                else EmployeeState.ACTIVE
            ),
            bot_principal_id=f"bot_{suffix}",
            member_groups=("oc_team", "oc_other"),
            aggregate_version=index + 1,
        )
        workforce.bot_principals[f"bot_{suffix}"] = BotPrincipal(
            bot_principal_id=f"bot_{suffix}",
            tenant_key="tenant_1",
            agent_id=agent_id,
            app_id=f"cli_{suffix}",
            credential_ref=f"cred_{suffix}",
        )
    channels = _Channels(agent_ids)

    def new_router():
        return module.DurableEmployeeIngressRouter(
            writer=writer,
            ingress_service=ingress,
            registry_provider=lambda: ProjectedAgentRegistry(workforce),
            channel_status_provider=channels,
            requester_acl=requester_acl
            or RuntimeRequesterChatAcl(
                allowed_requesters=("ou_requester",),
                allowed_chats=("oc_team", "oc_other"),
            ),
            queue_limits=module.RouterQueueLimits(
                per_employee=limits[0], per_team=limits[1], global_limit=limits[2]
            ),
            membership_health=membership_health or _HealthyMembership(),
            attachment_staging=attachment_staging,
            requester_principal_resolver=requester_principal_resolver,
            managed_group_registry_provider=managed_group_registry_provider,
            managed_group_owner_id=managed_group_owner_id,
            employee_bot_ids_provider=employee_bot_ids_provider,
        )

    return module, writer, ingress, new_router


def _accept(
    ingress: EmployeeIngressService,
    index: int,
    agent_id: str = "agt_alpha",
    *,
    sender: str = "ou_requester",
    chat_id: str = "oc_team",
    chat_type: str = "group",
    sender_union_id: str = "",
    attachment_descriptors: tuple[dict[str, object], ...] = (),
) -> str:
    payload = _payload(
        index,
        sender=sender,
        chat_type=chat_type,
        sender_union_id=sender_union_id,
        attachment_descriptors=attachment_descriptors,
    )
    metadata = _metadata(payload, index, agent_id, chat_id=chat_id)
    metadata = replace(metadata, sender_principal_id=sender)
    ack = ingress.accept(
        metadata,
        payload,
        request_id=f"req_{agent_id.removeprefix('agt_')}_{index}",
    )
    return ack.acceptance.acceptance_id


def test_owner_p2p_routes_with_union_owner_without_group_membership(
    tmp_path: Path,
) -> None:
    class DenyIfConsulted:
        def __init__(self) -> None:
            self.calls = 0

        def is_degraded(self, _agent_id: str, _chat_id: str) -> bool:
            self.calls += 1
            raise AssertionError("OWNER_P2P must not consult membership health")

    membership = DenyIfConsulted()

    def resolve_owner(**values):
        if (
            values["owner_principal_id"] == "ou_owner"
            and values["sender_union_id"] == "on_owner"
        ):
            return "ou_owner"
        return None

    _, writer, ingress, new_router = _stack(
        tmp_path,
        requester_acl=RuntimeRequesterChatAcl(
            allowed_requesters=("ou_owner",),
            allowed_chats=("oc_team",),
        ),
        membership_health=membership,
        requester_principal_resolver=resolve_owner,
    )
    router = new_router()
    acceptance_id = _accept(
        ingress,
        91,
        sender="ou_employee_app_owner",
        chat_id="oc_owner_p2p",
        chat_type="p2p",
        sender_union_id="on_owner",
    )

    queued = router.route(acceptance_id)
    grant = router.peek_dispatch_candidate()

    assert queued.state == "queued"
    assert queued.authority is not None
    assert queued.authority.authorization_scope is EmployeeAuthorizationScope.OWNER_P2P
    assert queued.authority.requester_principal_id == "ou_owner"
    assert grant is not None
    assert grant.request.authorization_scope is EmployeeAuthorizationScope.OWNER_P2P
    assert grant.request.source_requester_principal_id == "ou_employee_app_owner"
    assert membership.calls == 0
    ingress.close()
    writer.close()


@pytest.mark.parametrize("sender_union_id", ["", "on_other"])
def test_owner_p2p_missing_or_non_owner_union_fails_closed(
    tmp_path: Path,
    sender_union_id: str,
) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        requester_acl=RuntimeRequesterChatAcl(
            allowed_requesters=("ou_owner",),
            allowed_chats=(),
        ),
        requester_principal_resolver=(
            lambda **values: (
                "ou_owner" if values["sender_union_id"] == "on_owner" else None
            )
        ),
    )
    router = new_router()
    acceptance_id = _accept(
        ingress,
        92,
        sender="ou_employee_app_owner",
        chat_id="oc_owner_p2p",
        chat_type="p2p",
        sender_union_id=sender_union_id,
    )

    rejected = router.route(acceptance_id)

    assert rejected.state == "terminal"
    assert rejected.reason_code in {"requester_denied", "sender_invalid"}
    ingress.close()
    writer.close()


def test_owner_p2p_without_explicit_union_resolver_fails_closed(
    tmp_path: Path,
) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        requester_acl=RuntimeRequesterChatAcl(
            allowed_requesters=("ou_owner",),
            allowed_chats=(),
        ),
    )
    router = new_router()
    acceptance_id = _accept(
        ingress,
        93,
        sender="ou_owner",
        chat_id="oc_owner_p2p",
        chat_type="p2p",
        sender_union_id="on_owner",
    )

    rejected = router.route(acceptance_id)

    assert rejected.state == "terminal"
    assert rejected.reason_code == "requester_denied"
    ingress.close()
    writer.close()


def test_targeted_group_task_routes_with_union_owner_and_freezes_input(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from src.trust.models import ManagedGroupOrigin
    from src.trust.registry import ManagedGroupRegistry

    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    registry.register(
        chat_id="oc_team",
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.OWNER_ADOPTED,
        receiving_bot_ref="main-bot",
        project_id="project-1",
        canonical_root_ref="/project",
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    def resolve_owner(**values):
        return (
            "ou_owner"
            if values["sender_union_id"] == "on_owner"
            and values["owner_principal_id"] == "ou_owner"
            else None
        )

    _, writer, ingress, new_router = _stack(
        tmp_path,
        requester_acl=RuntimeRequesterChatAcl(
            allowed_requesters=("ou_owner",),
            allowed_chats=("oc_team",),
        ),
        requester_principal_resolver=resolve_owner,
        managed_group_registry_provider=lambda: registry,
        managed_group_owner_id="ou_owner",
        employee_bot_ids_provider=lambda: frozenset({"ou_bot_alpha"}),
    )
    router = new_router()
    payload = _payload(
        94,
        sender="ou_employee_app_owner",
        sender_union_id="on_owner",
    )
    part = dict(payload.normalized_parts[0])
    part.update(
        content={"text": "@_user_1 /task finish audit"},
        mentions=(
            {
                "key": "@_user_1",
                "open_id": "ou_bot_alpha",
                "tenant_key": "tenant_1",
            },
        ),
        remote_chat_id="oc_team",
        remote_message_id="om_targeted",
        remote_root_id="om_root",
    )
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id=payload.envelope_id,
        normalized_parts=(part,),
        attachment_descriptors=(),
    )
    metadata = _metadata(payload, 94, "agt_alpha")
    metadata = replace(
        metadata,
        sender_principal_id="ou_employee_app_owner",
        message_id="om_" + hashlib.sha256(b"om_targeted").hexdigest(),
        chat_id="oc_" + hashlib.sha256(b"oc_team").hexdigest(),
        thread_root_message_id="om_" + hashlib.sha256(b"om_root").hexdigest(),
    )
    acceptance_id = ingress.accept(
        metadata,
        payload,
        request_id="req_targeted_task",
    ).acceptance.acceptance_id

    queued = router.route(acceptance_id)
    grant = router.peek_dispatch_candidate()

    assert queued.state == "queued"
    assert queued.authority is not None
    assert queued.authority.requester_principal_id == "ou_owner"
    assert queued.authority.effective_input_kind == "targeted_group_task_v1"
    assert queued.authority.target_bot_open_id_digest == hashlib.sha256(
        b"ou_bot_alpha"
    ).hexdigest()
    with pytest.raises(ValueError, match="effective input"):
        replace(
            queued.authority,
            effective_input_kind=None,
            effective_input_digest=None,
        )
    assert grant is not None and grant.targeted_task is not None
    assert grant.targeted_task.description == "finish audit"
    assert grant.request.source_requester_principal_id == "ou_employee_app_owner"
    assert grant.payload.payload_sha256 == payload.payload_sha256
    assert "finish audit" not in repr(grant)

    restarted = new_router()
    replayed = restarted.peek_dispatch_candidate()
    assert replayed is not None and replayed.targeted_task is not None
    assert replayed.targeted_task.input_digest == grant.targeted_task.input_digest
    ingress.close()
    writer.close()


@pytest.mark.parametrize(
    ("mention_open_id", "sender_union_id", "resolved_requester"),
    (
        ("ou_bot_beta", "on_owner", "ou_owner"),
        ("ou_bot_alpha", "on_other", None),
        ("ou_bot_alpha", "on_owner", "ou_bot_alpha"),
    ),
)
def test_targeted_group_task_rejects_foreign_target_union_or_resolver_confusion(
    tmp_path: Path,
    mention_open_id: str,
    sender_union_id: str,
    resolved_requester: str | None,
) -> None:
    from datetime import UTC, datetime

    from src.trust.models import ManagedGroupOrigin
    from src.trust.registry import ManagedGroupRegistry

    registry = ManagedGroupRegistry(tmp_path / "managed-groups.json")
    registry.register(
        chat_id="oc_team",
        owner_id="ou_owner",
        origin=ManagedGroupOrigin.OWNER_ADOPTED,
        receiving_bot_ref="main-bot",
        project_id="project-1",
        canonical_root_ref="/project",
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    _, writer, ingress, new_router = _stack(
        tmp_path,
        requester_acl=RuntimeRequesterChatAcl(
            allowed_requesters=("ou_owner", "ou_bot_alpha", "ou_bot_beta"),
            allowed_chats=("oc_team",),
        ),
        requester_principal_resolver=lambda **_values: resolved_requester,
        managed_group_registry_provider=lambda: registry,
        managed_group_owner_id="ou_owner",
        employee_bot_ids_provider=lambda: frozenset(
            {"ou_bot_alpha", "ou_bot_beta"}
        ),
    )
    router = new_router()
    base = _payload(
        95,
        sender="ou_employee_app_owner",
        sender_union_id=sender_union_id,
    )
    part = dict(base.normalized_parts[0])
    part.update(
        content={"text": "@_user_1 /task finish audit"},
        mentions=(
            {
                "key": "@_user_1",
                "open_id": mention_open_id,
                "tenant_key": "tenant_1",
            },
        ),
        remote_chat_id="oc_team",
        remote_message_id="om_targeted",
        remote_root_id="om_root",
    )
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id=base.envelope_id,
        normalized_parts=(part,),
        attachment_descriptors=(),
    )
    metadata = replace(
        _metadata(payload, 95, "agt_alpha"),
        sender_principal_id="ou_employee_app_owner",
        message_id="om_" + hashlib.sha256(b"om_targeted").hexdigest(),
        chat_id="oc_" + hashlib.sha256(b"oc_team").hexdigest(),
        thread_root_message_id="om_" + hashlib.sha256(b"om_root").hexdigest(),
    )
    acceptance_id = ingress.accept(
        metadata,
        payload,
        request_id=f"req_rejected_{mention_open_id}_{sender_union_id}",
    ).acceptance.acceptance_id

    rejected = router.route(acceptance_id)

    assert rejected.state == "terminal"
    assert rejected.reason_code in {"authority_denied", "requester_denied"}
    ingress.close()
    writer.close()


def _commit_dispatch(router, writer, acceptance_id: str):
    """Test-only simulation of the coordinator's Router event application."""

    candidate = router.peek_dispatch_candidate()
    assert candidate is not None and candidate.record.acceptance_id == acceptance_id
    with router._ingress.employee_dispatch_guard(  # noqa: SLF001
        router=router
    ), writer.transaction_guard():
        router.synchronize_projection_unlocked()
        event = router.preflight_dispatch_event_unlocked(
            acceptance_id=acceptance_id,
        )
        result = writer.commit(
            (event,),
            writer.get_aggregate_versions((event.aggregate_id,)),
            expected_head_sequence=router.state.cursor_sequence,
            expected_head_hash=router.state.cursor_hash or None,
        )
        router.apply_committed_frame_unlocked(result.frame)
    return replace(candidate, record=router.state.by_acceptance_id[acceptance_id])


def test_router_persists_complete_lifecycle_and_atomic_queue_position(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    acceptance_id = _accept(ingress, 1)

    queued = router.route(acceptance_id)
    grant = _commit_dispatch(router, writer, acceptance_id)
    completed = router.finish(acceptance_id, reason_code="completed")

    assert queued.state == "queued"
    assert queued.queue_position == 1
    assert queued.queued_sequence > queued.accepted_sequence
    assert grant is not None and grant.record.state == "dispatching"
    assert completed.state == "terminal"
    frames = tuple(writer.replay())
    event_types = [event.event_type for frame in frames for event in frame.events]
    assert event_types == [
        "employee.ingress.accepted",
        "employee.ingress.router_authorized",
        "employee.ingress.router_staging",
        "employee.ingress.router_queued",
        "employee.ingress.router_dispatching",
        "employee.ingress.router_terminal",
    ]
    queued_events = [
        (frame, event)
        for frame in frames
        for event in frame.events
        if event.event_type == "employee.ingress.router_queued"
    ]
    assert len(queued_events) == 1
    frame, event = queued_events[0]
    assert event.payload["queue_position"] == 1
    assert event.payload["authority"]["team_id"] == "oc_team"
    serialized_authority = repr(event.payload["authority"]).lower()
    assert "credential" not in serialized_authority
    assert "secret" not in serialized_authority
    assert "access_token" not in serialized_authority
    assert frame.sequence == queued.queued_sequence
    ingress.close()
    writer.close()


def test_invalid_router_transition_is_rejected_before_journal_commit(
    tmp_path: Path,
) -> None:
    module, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    acceptance_id = _accept(ingress, 1)
    before = writer.anchor.read()

    with writer.transaction_guard(), router._mutex:
        router.rebuild_projection()
        record = router.state.by_acceptance_id[acceptance_id]
        with pytest.raises(module.RouterProjectionError):
            router._transition_unlocked(record, "queued", {"queue_position": 1})

    assert writer.anchor.read() == before
    ingress.close()
    writer.close()


def test_router_replay_accepts_legacy_control_terminal_but_preflight_rejects(
    tmp_path: Path,
) -> None:
    module, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    acceptance_id = _accept(ingress, 1)
    router.rebuild_projection()
    record = router.state.by_acceptance_id[acceptance_id]
    event = JournalEvent(
        event_type="employee.ingress.router_terminal",
        aggregate_id=record.aggregate_id,
        payload={
            "acceptance_id": acceptance_id,
            "reason_code": "control_consumed",
        },
    )
    result = writer.commit(
        (event,),
        writer.get_aggregate_versions((event.aggregate_id,)),
    )

    with pytest.raises(module.RouterProjectionError, match="terminal reason"):
        router.preflight_frame_unlocked(result.frame)

    restarted = new_router()
    replayed = restarted.state.by_acceptance_id[acceptance_id]
    assert replayed.state == "terminal"
    assert replayed.reason_code == "control_consumed"
    ingress.close()
    writer.close()


def test_router_replay_backfills_legacy_authorized_requester_from_acceptance(
    tmp_path: Path,
) -> None:
    module, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    seed_acceptance = _accept(ingress, 1)
    router.route(seed_acceptance)
    authority = router.state.by_acceptance_id[seed_acceptance].authority
    assert authority is not None
    authority_payload = authority.to_dict()
    for field_name in (
        "authorization_scope",
        "effective_input_kind",
        "effective_input_digest",
        "target_bot_open_id_digest",
    ):
        authority_payload.pop(field_name)
    authority_payload["requester_principal_id"] = "ou_resolved_requester"

    legacy_acceptance = _accept(ingress, 2)
    router.rebuild_projection()
    record = router.state.by_acceptance_id[legacy_acceptance]
    event = JournalEvent(
        event_type="employee.ingress.router_authorized",
        aggregate_id=record.aggregate_id,
        payload={
            "acceptance_id": legacy_acceptance,
            "authority": authority_payload,
        },
    )
    result = writer.commit(
        (event,),
        writer.get_aggregate_versions((event.aggregate_id,)),
    )

    with pytest.raises(module.RouterProjectionError, match="authorized transition"):
        router.preflight_frame_unlocked(result.frame)

    restarted = new_router()
    replayed = restarted.state.by_acceptance_id[legacy_acceptance]
    assert replayed.state == "authorized"
    assert replayed.requester_principal_id == "ou_resolved_requester"
    assert replayed.authority is not None
    assert (
        replayed.authority.authorization_scope
        is EmployeeAuthorizationScope.MANAGED_GROUP
    )
    ingress.close()
    writer.close()


def test_durable_fifo_survives_router_restart(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    acceptance_ids = [_accept(ingress, index) for index in (1, 2, 3)]
    for acceptance_id in acceptance_ids:
        assert router.route(acceptance_id).state == "queued"

    restarted = new_router()
    observed: list[str] = []
    for expected in acceptance_ids:
        grant = _commit_dispatch(restarted, writer, expected)
        assert grant is not None
        observed.append(grant.record.acceptance_id)
        restarted.finish(expected, reason_code="completed")

    assert observed == acceptance_ids
    ingress.close()
    writer.close()


def test_restart_keeps_dispatching_work_fail_closed_without_redispatch(
    tmp_path: Path,
) -> None:
    _, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    first = _accept(ingress, 1)
    second = _accept(ingress, 2)
    assert router.route(first).state == "queued"
    assert router.route(second).state == "queued"
    grant = _commit_dispatch(router, writer, first)
    assert grant is not None and grant.record.acceptance_id == first

    restarted = new_router()

    assert restarted.state.by_acceptance_id[first].state == "dispatching"
    assert restarted.peek_dispatch_candidate() is None
    assert restarted.state.by_acceptance_id[second].state == "queued"
    ingress.close()
    writer.close()


def test_inbox_failure_helper_cannot_terminate_a_dispatching_grant(
    tmp_path: Path,
) -> None:
    _, writer, ingress, new_router = _stack(tmp_path)
    router = new_router()
    acceptance_id = _accept(ingress, 1)
    assert router.route(acceptance_id).state == "queued"
    grant = _commit_dispatch(router, writer, acceptance_id)
    assert grant is not None and grant.record.state == "dispatching"

    retained = router._terminal_inbox_failure(acceptance_id)

    assert retained.state == "dispatching"
    assert retained.reason_code == ""
    ingress.close()
    writer.close()


def test_queue_full_is_terminal_without_a_queue_event(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(tmp_path, limits=(1, 2, 2))
    router = new_router()
    first = _accept(ingress, 1)
    second = _accept(ingress, 2)
    assert router.route(first).state == "queued"

    rejected = router.route(second)

    assert rejected.state == "terminal"
    assert rejected.reason_code == "queue_full"
    queue_acceptances = {
        event.payload["acceptance_id"]
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "employee.ingress.router_queued"
    }
    assert queue_acceptances == {first}
    ingress.close()
    writer.close()


def test_per_employee_dispatch_is_one_but_another_employee_can_progress(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path, agent_ids=("agt_alpha", "agt_beta")
    )
    router = new_router()
    alpha_1 = _accept(ingress, 1, "agt_alpha")
    alpha_2 = _accept(ingress, 2, "agt_alpha")
    beta_1 = _accept(ingress, 3, "agt_beta")
    for acceptance_id in (alpha_1, alpha_2, beta_1):
        assert router.route(acceptance_id).state == "queued"

    first = _commit_dispatch(router, writer, alpha_1)
    second = router.peek_dispatch_candidate()

    assert first is not None and first.record.acceptance_id == alpha_1
    assert second is not None and second.record.acceptance_id == beta_1
    second = _commit_dispatch(router, writer, beta_1)
    assert router.peek_dispatch_candidate() is None
    router.finish(alpha_1, reason_code="completed")
    third = router.peek_dispatch_candidate()
    assert third is not None and third.record.acceptance_id == alpha_2
    ingress.close()
    writer.close()


def test_two_employees_are_isolated_under_team_and_global_queue_limits(
    tmp_path: Path,
) -> None:
    """EI-QUEUE-01: local integration evidence in local_process_harness."""

    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha = (
        _accept(ingress, 1, "agt_alpha"),
        _accept(ingress, 2, "agt_alpha"),
    )
    assert [router.route(item).state for item in alpha] == ["queued", "queued"]
    beta = (
        _accept(ingress, 3, "agt_beta"),
        _accept(ingress, 4, "agt_beta"),
    )
    barrier = Barrier(2)

    def admit(acceptance_id: str):
        barrier.wait()
        return router.route(acceptance_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(admit, beta))

    router.rebuild_projection()
    final_records = [router.state.by_acceptance_id[item] for item in (*alpha, *beta)]
    queued = [record for record in final_records if record.state == "queued"]
    rejected = [
        record for record in final_records if record.reason_code == "queue_rebalanced"
    ]
    assert {record.authority.agent_id for record in queued} == {"agt_alpha", "agt_beta"}
    assert len(rejected) == 1 and rejected[0].agent_id == "agt_alpha"
    queue_full = [record for record in final_records if record.reason_code == "queue_full"]
    assert len(queue_full) == 1 and queue_full[0].agent_id == "agt_beta"
    ingress.close()
    writer.close()


def test_no_pending_peer_does_not_reserve_shared_capacity(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha = (_accept(ingress, 1), _accept(ingress, 2))

    assert [router.route(item).state for item in alpha] == ["queued", "queued"]
    ingress.close()
    writer.close()


def test_inactive_pending_peer_does_not_reserve_shared_capacity(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        inactive_agent_ids=("agt_beta",),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha = (_accept(ingress, 1), _accept(ingress, 2))
    _accept(ingress, 3, "agt_beta")

    assert [router.route(item).state for item in alpha] == ["queued", "queued"]
    ingress.close()
    writer.close()


def test_unauthorized_pending_peer_does_not_reserve_shared_capacity(
    tmp_path: Path,
) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha = (_accept(ingress, 1), _accept(ingress, 2))
    beta = _accept(ingress, 3, "agt_beta", sender="ou_intruder")

    assert [router.route(item).state for item in alpha] == ["queued", "queued"]
    rejected = router.route(beta)
    assert rejected.state == "terminal"
    assert rejected.reason_code == "requester_denied"
    assert all(router.state.by_acceptance_id[item].state == "queued" for item in alpha)
    ingress.close()
    writer.close()


def test_two_router_instances_atomically_admit_one_acceptance_once(
    tmp_path: Path,
) -> None:
    _, writer, ingress, new_router = _stack(tmp_path)
    first = new_router()
    second = new_router()
    acceptance_id = _accept(ingress, 1)
    barrier = Barrier(2)

    def route(router):
        barrier.wait()
        return router.route(acceptance_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(route, (first, second)))

    assert all(record.state == "queued" for record in records)
    queued_events = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "employee.ingress.router_queued"
    ]
    assert len(queued_events) == 1
    assert queued_events[0].payload["acceptance_id"] == acceptance_id
    ingress.close()
    writer.close()


def test_new_employee_rebalances_latest_queued_peer_in_one_frame(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha_1 = _accept(ingress, 1, "agt_alpha")
    alpha_2 = _accept(ingress, 2, "agt_alpha")
    assert router.route(alpha_1).state == "queued"
    assert router.route(alpha_2).state == "queued"
    beta = _accept(ingress, 3, "agt_beta")

    admitted = router.route(beta)

    assert admitted.state == "queued"
    assert router.state.by_acceptance_id[alpha_1].state == "queued"
    victim = router.state.by_acceptance_id[alpha_2]
    assert victim.state == "terminal"
    assert victim.reason_code == "queue_rebalanced"
    rebalance_frames = [
        frame
        for frame in writer.replay()
        if [event.event_type for event in frame.events]
        == [
            "employee.ingress.router_terminal",
            "employee.ingress.router_queued",
        ]
    ]
    assert len(rebalance_frames) == 1
    assert [event.payload["acceptance_id"] for event in rebalance_frames[0].events] == [
        alpha_2,
        beta,
    ]
    assert set(rebalance_frames[0].expected_versions) == {
        router.state.by_acceptance_id[alpha_2].aggregate_id,
        router.state.by_acceptance_id[beta].aggregate_id,
    }
    ingress.close()
    writer.close()


def test_rebalanced_victim_cleans_its_completed_attachment_stage(tmp_path: Path) -> None:
    staging = _QueueCleanupStaging()
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
        attachment_staging=staging,
    )
    router = new_router()
    alpha_1 = _accept(ingress, 1, "agt_alpha")
    assert router.route(alpha_1).state == "queued"
    descriptor = (
        {
            "resource_type": "file",
            "resource_id": "file_queue",
            "mime_type": "text/plain",
            "size_bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
    )
    alpha_2 = _accept(
        ingress,
        2,
        "agt_alpha",
        attachment_descriptors=descriptor,
    )
    assert router.route(alpha_2).state == "queued"
    beta = _accept(ingress, 3, "agt_beta")

    admitted = router.route(beta)

    assert admitted.state == "queued"
    assert router.state.by_acceptance_id[alpha_2].reason_code == "queue_rebalanced"
    assert staging.cleanup_calls == ["stg_queue_1"]
    assert staging.state.by_staging_id["stg_queue_1"].cleanup_state == "completed"
    ingress.close()
    writer.close()


def test_route_never_rebalances_after_sampled_workforce_authority_changes(
    tmp_path: Path,
) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha = [_accept(ingress, index, "agt_alpha") for index in (1, 2)]
    for acceptance_id in alpha:
        assert router.route(acceptance_id).state == "queued"
    beta = _accept(ingress, 3, "agt_beta")
    original_resolve = router._resolve_authority
    samples = 0

    def resolve_then_revoke(metadata, payload):
        nonlocal samples
        result = original_resolve(metadata, payload)
        if result[0] is not None:
            samples += 1
            if samples == 2:
                aggregate_id = "workforce_race_route_final"
                writer.commit(
                    [
                        JournalEvent(
                            event_type="employee.state_changed",
                            aggregate_id=aggregate_id,
                            payload={"state": "draft"},
                        )
                    ],
                    writer.get_aggregate_versions([aggregate_id]),
                )
        return result

    router._resolve_authority = resolve_then_revoke

    terminal = router.route(beta)

    assert samples == 2
    assert terminal.state == "terminal"
    assert terminal.reason_code == "authority_stale"
    assert all(router.state.by_acceptance_id[item].state == "queued" for item in alpha)
    assert not any(
        event.event_type == "employee.ingress.router_terminal"
        and event.payload.get("reason_code") == "queue_rebalanced"
        for frame in writer.replay()
        for event in frame.events
    )
    ingress.close()
    writer.close()


def test_rebalance_does_not_churn_when_each_agent_owns_one_slot(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta", "agt_gamma"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha = _accept(ingress, 1, "agt_alpha")
    beta = _accept(ingress, 2, "agt_beta")
    assert router.route(alpha).state == "queued"
    assert router.route(beta).state == "queued"
    gamma = _accept(ingress, 3, "agt_gamma")

    rejected = router.route(gamma)

    assert rejected.state == "terminal"
    assert rejected.reason_code == "queue_full"
    assert router.state.by_acceptance_id[alpha].state == "queued"
    assert router.state.by_acceptance_id[beta].state == "queued"
    ingress.close()
    writer.close()


def test_rebalance_evicts_latest_item_from_most_overrepresented_agent(
    tmp_path: Path,
) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta", "agt_gamma"),
        limits=(2, 3, 3),
    )
    router = new_router()
    alpha = [_accept(ingress, index, "agt_alpha") for index in (1, 2)]
    beta = _accept(ingress, 3, "agt_beta")
    for acceptance_id in (*alpha, beta):
        assert router.route(acceptance_id).state == "queued"
    gamma = _accept(ingress, 4, "agt_gamma")

    admitted = router.route(gamma)

    assert admitted.state == "queued"
    assert router.state.by_acceptance_id[alpha[0]].state == "queued"
    assert router.state.by_acceptance_id[alpha[1]].reason_code == "queue_rebalanced"
    assert router.state.by_acceptance_id[beta].state == "queued"
    ingress.close()
    writer.close()


def test_team_full_rebalance_never_evicts_another_team(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta", "agt_gamma"),
        limits=(2, 2, 4),
    )
    router = new_router()
    alpha = [_accept(ingress, index, "agt_alpha") for index in (1, 2)]
    for acceptance_id in alpha:
        assert router.route(acceptance_id).state == "queued"
    other_team = _accept(ingress, 3, "agt_gamma", chat_id="oc_other")
    assert router.route(other_team).state == "queued"
    beta = _accept(ingress, 4, "agt_beta")

    admitted = router.route(beta)

    assert admitted.state == "queued"
    assert router.state.by_acceptance_id[alpha[1]].reason_code == "queue_rebalanced"
    assert router.state.by_acceptance_id[other_team].state == "queued"
    ingress.close()
    writer.close()


def test_global_full_rebalance_can_evict_another_team(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta", "agt_gamma"),
        limits=(2, 3, 3),
    )
    router = new_router()
    alpha = [
        _accept(ingress, index, "agt_alpha", chat_id="oc_other")
        for index in (1, 2)
    ]
    for acceptance_id in alpha:
        assert router.route(acceptance_id).state == "queued"
    beta = _accept(ingress, 3, "agt_beta")
    assert router.route(beta).state == "queued"
    gamma = _accept(ingress, 4, "agt_gamma")

    admitted = router.route(gamma)

    assert admitted.state == "queued"
    assert router.state.by_acceptance_id[alpha[1]].reason_code == "queue_rebalanced"
    assert router.state.by_acceptance_id[beta].state == "queued"
    ingress.close()
    writer.close()


def test_rebalance_never_evicts_dispatching_work(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha_dispatching = _accept(ingress, 1, "agt_alpha")
    assert router.route(alpha_dispatching).state == "queued"
    grant = _commit_dispatch(router, writer, alpha_dispatching)
    assert grant is not None and grant.record.acceptance_id == alpha_dispatching
    alpha_queued = [
        _accept(ingress, index, "agt_alpha") for index in (2, 3)
    ]
    for acceptance_id in alpha_queued:
        assert router.route(acceptance_id).state == "queued"
    beta = _accept(ingress, 4, "agt_beta")

    admitted = router.route(beta)

    assert admitted.state == "queued"
    assert router.state.by_acceptance_id[alpha_dispatching].state == "dispatching"
    assert router.state.by_acceptance_id[alpha_queued[0]].state == "queued"
    assert (
        router.state.by_acceptance_id[alpha_queued[1]].reason_code
        == "queue_rebalanced"
    )
    ingress.close()
    writer.close()


def test_rebalance_frame_replays_atomically_after_restart(tmp_path: Path) -> None:
    _, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
    )
    router = new_router()
    alpha = [_accept(ingress, index, "agt_alpha") for index in (1, 2)]
    for acceptance_id in alpha:
        assert router.route(acceptance_id).state == "queued"
    beta = _accept(ingress, 3, "agt_beta")
    assert router.route(beta).state == "queued"

    restarted = new_router()

    assert restarted.state.by_acceptance_id[alpha[0]].state == "queued"
    assert restarted.state.by_acceptance_id[alpha[1]].reason_code == "queue_rebalanced"
    assert restarted.state.by_acceptance_id[beta].state == "queued"
    ingress.close()
    writer.close()


def test_anchor_failure_cannot_publish_half_a_rebalance(tmp_path: Path) -> None:
    anchor = _SelectiveRejectAnchor()
    module, writer, ingress, new_router = _stack(
        tmp_path,
        agent_ids=("agt_alpha", "agt_beta"),
        limits=(2, 2, 2),
        anchor=anchor,
    )
    router = new_router()
    alpha = [_accept(ingress, index, "agt_alpha") for index in (1, 2)]
    for acceptance_id in alpha:
        assert router.route(acceptance_id).state == "queued"
    beta = _accept(ingress, 3, "agt_beta")
    anchor.reject_sequence = anchor.read().sequence + 3

    with pytest.raises(module.RouterWriteDisabledError):
        router.route(beta)

    assert all(router.state.by_acceptance_id[item].state == "queued" for item in alpha)
    assert router.state.by_acceptance_id[beta].state == "staging"
    router.rebuild_projection()
    assert all(router.state.by_acceptance_id[item].state == "queued" for item in alpha)
    assert router.state.by_acceptance_id[beta].state == "staging"
    unanchored = next(
        frame for frame in writer.replay() if frame.sequence == anchor.reject_sequence
    )
    assert [event.event_type for event in unanchored.events] == [
        "employee.ingress.router_terminal",
        "employee.ingress.router_queued",
    ]
    assert not any(
        frame.sequence <= anchor.read().sequence
        and {event.payload.get("acceptance_id") for event in frame.events}
        == {alpha[1], beta}
        for frame in writer.replay()
    )
    ingress.close()
    writer.close()
