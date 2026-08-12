"""Durable retry contract for authenticated employee ingress payload reads."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from src.autonomous.context.runtime import RuntimeRequesterChatAcl
from src.autonomous.domain import (
    BotPrincipal,
    EmployeeDefinition,
    EmployeeState,
    WorkerType,
)
from src.autonomous.ingress.models import (
    EmployeeIngressMetadata,
    EmployeeIngressPayload,
)
from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.router import (
    DurableEmployeeIngressRouter,
    RouterProjectionError,
    RouterQueueLimits,
    RouterWriteDisabledError,
)
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.blob_store import (
    AesGcmEncryptionProvider,
    BlobAuthenticationError,
    BlobReadError,
    BlobStore,
    KeyResolutionError,
)
from src.autonomous.journal.frame import GENESIS_HASH, JournalEvent
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning.composition import (
    EmployeeDepartmentRuntime,
    EmployeeMessageHandoffUnknownError,
)
from src.autonomous.supervisor.channel_models import ChannelProcessState
from src.autonomous.supervisor.employee_channels import ChannelProcessStatus
from src.autonomous.workforce.registry import ProjectedAgentRegistry


class _Membership:
    def is_degraded(self, _agent_id: str, _chat_id: str) -> bool:
        return False


def _stack(tmp_path):
    now = [datetime(2026, 8, 12, tzinfo=UTC)]
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=MemoryAnchor(),
        hmac_key=b"ingress-payload-retry-hmac-key!!",
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "blobs",
            AesGcmEncryptionProvider(lambda _key_ref: b"i" * 32),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    workforce = ProjectionState(cursor_hash=GENESIS_HASH)
    workforce.employees["agt_alpha"] = EmployeeDefinition(
        agent_id="agt_alpha",
        tenant_key="tenant_1",
        owner_principal_id="ou_owner",
        name="alpha",
        tool="codex",
        model="gpt-5.6-sol",
        worker_type=WorkerType.VISIBLE,
        state=EmployeeState.ACTIVE,
        bot_principal_id="bot_alpha",
        member_groups=("oc_team",),
        aggregate_version=1,
    )
    workforce.bot_principals["bot_alpha"] = BotPrincipal(
        bot_principal_id="bot_alpha",
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        app_id="cli_alpha",
        credential_ref="cred_alpha",
    )
    status = ChannelProcessStatus(
        agent_id="agt_alpha",
        app_id="cli_alpha",
        generation=3,
        pid=101,
        state=ChannelProcessState.READY,
        tenant_key="tenant_1",
        bot_principal_id="bot_alpha",
        identity={"app_id": "cli_alpha", "open_id": "ou_bot_alpha"},
        ready_metadata={"connection_id": "conn_alpha"},
    )
    router_kwargs = {
        "writer": writer,
        "ingress_service": ingress,
        "registry_provider": lambda: ProjectedAgentRegistry(workforce),
        "channel_status_provider": SimpleNamespace(status=lambda _agent_id: status),
        "requester_acl": RuntimeRequesterChatAcl(
            allowed_requesters=("ou_requester",),
            allowed_chats=("oc_team",),
        ),
        "queue_limits": RouterQueueLimits(4, 8, 16),
        "membership_health": _Membership(),
        "context_retry_base_seconds": 1.0,
        "context_retry_max_seconds": 4.0,
        "clock": lambda: now[0],
    }
    router = DurableEmployeeIngressRouter(**router_kwargs)
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "1" * 64,
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "group",
                "content": {"text": "durable task"},
                "sender_id": "ou_requester",
                "sender_union_id": "",
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": "tenant_1",
                "feishu_thread_id": "omt_1",
                "remote_chat_id": "oc_team",
                "remote_message_id": "om_1",
                "remote_root_id": "om_root",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        channel_generation=3,
        connection_id="conn_alpha",
        event_id="evt_1",
        message_id="om_" + hashlib.sha256(b"om_1").hexdigest(),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_" + hashlib.sha256(b"oc_team").hexdigest(),
        thread_root_message_id=(
            "om_" + hashlib.sha256(b"om_root").hexdigest()
        ),
        sender_principal_id="ou_requester",
        received_at="2026-08-12T00:00:00Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    acceptance_id = ingress.accept(
        metadata,
        payload,
        request_id="req_1",
    ).acceptance.acceptance_id
    return SimpleNamespace(
        now=now,
        writer=writer,
        ingress=ingress,
        router=router,
        router_kwargs=router_kwargs,
        payload=payload,
        acceptance_id=acceptance_id,
    )


@pytest.mark.parametrize(
    "transient",
    (
        BlobReadError("temporary storage read failed: /private/secret"),
        OSError("temporary filesystem read failed: /private/secret"),
    ),
)
def test_transient_payload_read_waits_then_routes_without_losing_command(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    transient: Exception,
) -> None:
    stack = _stack(tmp_path)
    original_read = stack.ingress.blob_store.read
    calls = 0

    def fail_once(ref):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise transient
        return original_read(ref)

    monkeypatch.setattr(stack.ingress.blob_store, "read", fail_once)
    try:
        first = stack.router.route(stack.acceptance_id)

        assert first.state == "accepted"
        assert first.inbox_failures == 1
        assert first.inbox_next_eligible_at == "2026-08-12T00:00:01Z"
        assert calls == 1

        before_due = stack.router.route(stack.acceptance_id)
        assert before_due == first
        assert calls == 1

        stack.now[0] += timedelta(seconds=1)
        assert stack.router.route(stack.acceptance_id).state == "queued"
        assert calls > 1
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_unconfirmed_handoff_durably_abandons_only_matching_accepted_transport(
    tmp_path,
) -> None:
    stack = _stack(tmp_path)
    try:
        started = time.monotonic()
        stale = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=4,
            connection_id="conn_reconnected",
        )
        assert time.monotonic() - started < 0.2
        assert stale.state == "accepted"

        abandoned = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=3,
            connection_id="conn_alpha",
        )
        assert abandoned.state == "terminal"
        assert abandoned.reason_code == "handoff_unconfirmed"
        assert stack.router.route(stack.acceptance_id) == abandoned

        restarted = DurableEmployeeIngressRouter(**stack.router_kwargs)
        replayed = restarted.record_snapshot(stack.acceptance_id)
        assert replayed == abandoned
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_unconfirmed_alias_miss_rechecks_concurrent_router_ownership(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    original_observe = stack.ingress.observe_anchored_message_acceptance

    def miss_after_queue(**kwargs):
        queued = stack.router.route(stack.acceptance_id)
        assert queued.state == "queued"
        return original_observe(**kwargs)

    monkeypatch.setattr(
        stack.ingress,
        "observe_anchored_message_acceptance",
        miss_after_queue,
    )
    try:
        current = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=4,
            connection_id="conn_missing_alias",
        )
        assert current.state == "queued"
        assert current.queued_sequence > 0
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_unconfirmed_handoff_accepts_an_exact_reconnect_transport_witness(
    tmp_path,
) -> None:
    stack = _stack(tmp_path)
    try:
        original = stack.ingress.record_snapshot(stack.acceptance_id)
        assert original is not None
        replay = stack.ingress.accept(
            replace(
                original.metadata,
                channel_generation=9,
                connection_id="conn_reconnected",
            ),
            stack.payload,
            request_id="req_reconnected",
        )
        assert replay.duplicate is True
        assert replay.acceptance.acceptance_id == stack.acceptance_id

        observed = stack.ingress.observe_anchored_message_acceptance(
            tenant_key=original.metadata.tenant_key,
            agent_id=original.metadata.agent_id,
            bot_principal_id=original.metadata.bot_principal_id,
            app_id=original.metadata.app_id,
            event_type=original.metadata.event_type,
            chat_id=original.metadata.chat_id,
            message_id=original.metadata.message_id,
            channel_generation=9,
            connection_id="conn_reconnected",
        )
        assert observed is not None
        assert observed.acceptance == replay.acceptance

        abandoned = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=9,
            connection_id="conn_reconnected",
        )

        assert abandoned.state == "terminal"
        assert abandoned.reason_code == "handoff_unconfirmed"
        # Router keeps the canonical first acceptance while the exact C2
        # witness remains a separate append-only Ingress fact.
        assert abandoned.channel_generation == 3
        assert abandoned.connection_id == "conn_alpha"
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_unconfirmed_handoff_without_deadline_waits_for_alias_witness_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    original_snapshot = stack.ingress.record_snapshot
    begin_contention = threading.Event()
    lock_acquired = threading.Event()
    release_lock = threading.Event()
    result: list[object] = []

    original = original_snapshot(stack.acceptance_id)
    assert original is not None
    replay = stack.ingress.accept(
        replace(
            original.metadata,
            channel_generation=9,
            connection_id="conn_contended_alias",
        ),
        stack.payload,
        request_id="req_contended_alias",
    )
    assert replay.duplicate is True

    def hold_ingress_after_canonical_snapshot() -> None:
        assert begin_contention.wait(1.0)
        with stack.ingress._mutex:  # noqa: SLF001 - deterministic contention
            lock_acquired.set()
            release_lock.wait(1.0)

    def snapshot_then_contend(*args, **kwargs):
        snapshot = original_snapshot(*args, **kwargs)
        begin_contention.set()
        assert lock_acquired.wait(1.0)
        return snapshot

    monkeypatch.setattr(stack.ingress, "record_snapshot", snapshot_then_contend)
    holder = threading.Thread(target=hold_ingress_after_canonical_snapshot)
    waiter = threading.Thread(
        target=lambda: result.append(
            stack.router.abandon_message_handoff(
                stack.acceptance_id,
                channel_generation=9,
                connection_id="conn_contended_alias",
            )
        )
    )
    holder.start()
    waiter.start()
    try:
        assert lock_acquired.wait(1.0)
        time.sleep(0.05)
    finally:
        release_lock.set()
        holder.join(timeout=1.0)
        waiter.join(timeout=1.0)

    assert not holder.is_alive()
    assert not waiter.is_alive()
    assert len(result) == 1
    assert result[0].state == "terminal"
    assert result[0].reason_code == "handoff_unconfirmed"
    stack.ingress.close()
    stack.writer.close()


def test_unconfirmed_handoff_clamps_long_alias_wait_to_ingress_limit(
    tmp_path,
) -> None:
    stack = _stack(tmp_path)
    try:
        original = stack.ingress.record_snapshot(stack.acceptance_id)
        assert original is not None
        replay = stack.ingress.accept(
            replace(
                original.metadata,
                channel_generation=9,
                connection_id="conn_long_deadline_alias",
            ),
            stack.payload,
            request_id="req_long_deadline_alias",
        )
        assert replay.duplicate is True

        abandoned = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=9,
            connection_id="conn_long_deadline_alias",
            deadline=time.monotonic() + 60.0,
        )

        assert abandoned.state == "terminal"
        assert abandoned.reason_code == "handoff_unconfirmed"
    finally:
        stack.ingress.close()
        stack.writer.close()


@pytest.mark.parametrize("prequeue_state", ("authorized", "staging"))
def test_unconfirmed_handoff_alias_uses_canonical_ingress_coordinates_after_authority(
    tmp_path,
    prequeue_state: str,
) -> None:
    stack = _stack(tmp_path)
    try:
        original = stack.ingress.record_snapshot(stack.acceptance_id)
        assert original is not None
        replay = stack.ingress.accept(
            replace(
                original.metadata,
                channel_generation=9,
                connection_id="conn_authorized_alias",
            ),
            stack.payload,
            request_id=f"req_{prequeue_state}_alias",
        )
        assert replay.duplicate is True

        with stack.router._mutex:  # noqa: SLF001 - freeze exact prequeue state
            stack.router.rebuild_projection()
            record, ingress_record, payload = stack.router._dispatch_snapshot(  # noqa: SLF001
                stack.acceptance_id
            )
            resolution, reason = stack.router._resolve_authority(  # noqa: SLF001
                ingress_record.metadata,
                payload,
            )
            assert reason == ""
            assert resolution is not None
            record = stack.router._transition_unlocked(  # noqa: SLF001
                record,
                "authorized",
                {
                    "authority": resolution.snapshot.to_dict(),
                    "source_requester_principal_id": record.requester_principal_id,
                },
            )
            if prequeue_state == "staging":
                stack.router._transition_unlocked(  # noqa: SLF001
                    record,
                    "staging",
                    {},
                )

        abandoned = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=9,
            connection_id="conn_authorized_alias",
            deadline=time.monotonic() + 0.5,
        )

        assert abandoned.state == "terminal"
        assert abandoned.reason_code == "handoff_unconfirmed"
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_unconfirmed_handoff_does_not_synchronously_sweep_attachments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)

    def fail_if_swept() -> None:
        pytest.fail("handoff must not block on attachment cleanup")

    monkeypatch.setattr(stack.router, "_sweep_terminal_attachments", fail_if_swept)
    try:
        abandoned = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=3,
            connection_id="conn_alpha",
        )

        assert abandoned.state == "terminal"
        assert abandoned.reason_code == "handoff_unconfirmed"
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_queued_commit_wins_over_unconfirmed_handoff_abandon(tmp_path) -> None:
    stack = _stack(tmp_path)
    try:
        queued = stack.router.route(stack.acceptance_id)
        assert queued.state == "queued"

        after_abandon = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=3,
            connection_id="conn_alpha",
        )

        assert after_abandon == queued
        restarted = DurableEmployeeIngressRouter(**stack.router_kwargs)
        assert restarted.record_snapshot(stack.acceptance_id) == queued
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_handoff_abandon_replays_before_taking_writer_transaction_guard(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    original_guard = stack.writer.transaction_guard
    original_rebuild = stack.router.rebuild_projection
    guard_depth = 0

    @contextmanager
    def tracked_guard():
        nonlocal guard_depth
        with original_guard():
            guard_depth += 1
            try:
                yield
            finally:
                guard_depth -= 1

    def checked_rebuild(**kwargs):
        assert guard_depth == 0, "Journal replay must not hold the writer transaction guard"
        return original_rebuild(**kwargs)

    monkeypatch.setattr(stack.writer, "transaction_guard", tracked_guard)
    monkeypatch.setattr(stack.router, "rebuild_projection", checked_rebuild)
    try:
        result = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=3,
            connection_id="conn_alpha",
        )

        assert result.state == "terminal"
        assert result.reason_code == "handoff_unconfirmed"
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_handoff_abandon_retries_head_drift_and_preserves_concurrent_queue_winner(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    competing_router = DurableEmployeeIngressRouter(**stack.router_kwargs)
    original_commit = stack.writer.commit
    injected = False

    def commit_after_queue_wins(events, expected_versions, **kwargs):
        nonlocal injected
        if not injected and any(
            event.event_type == "employee.ingress.router_terminal"
            for event in events
        ):
            injected = True
            assert competing_router.route(stack.acceptance_id).state == "queued"
        return original_commit(events, expected_versions, **kwargs)

    monkeypatch.setattr(stack.writer, "commit", commit_after_queue_wins)
    try:
        result = stack.router.abandon_message_handoff(
            stack.acceptance_id,
            channel_generation=3,
            connection_id="conn_alpha",
        )

        assert injected is True
        assert result.state == "queued"
        assert result.reason_code == ""
        assert stack.router.record_snapshot(stack.acceptance_id) == result
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_handoff_abandon_expired_deadline_does_not_mutate_router(tmp_path) -> None:
    stack = _stack(tmp_path)
    before = tuple(stack.writer.replay())
    try:
        started = time.monotonic()
        with pytest.raises(RouterWriteDisabledError, match="deadline"):
            stack.router.abandon_message_handoff(
                stack.acceptance_id,
                channel_generation=3,
                connection_id="conn_alpha",
                deadline=started - 1.0,
            )

        assert time.monotonic() - started < 0.25
        stack.router.rebuild_projection()
        assert stack.router.record_snapshot(stack.acceptance_id).state == "accepted"
        assert tuple(stack.writer.replay()) == before
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_handoff_projection_router_lock_contention_expires_without_late_mutation(
    tmp_path,
) -> None:
    stack = _stack(tmp_path)
    stack.router.rebuild_projection()
    assert stack.router.record_snapshot(stack.acceptance_id).state == "accepted"
    entered = threading.Event()
    release = threading.Event()

    def occupy_router() -> None:
        with stack.router._mutex:  # noqa: SLF001 - deterministic contention
            entered.set()
            release.wait(1)

    holder = threading.Thread(target=occupy_router)
    holder.start()
    assert entered.wait(1)
    runtime = object.__new__(EmployeeDepartmentRuntime)
    runtime._ingress = stack.ingress
    runtime._router = stack.router
    runtime._closing = False
    before = tuple(stack.writer.replay())

    try:
        started = time.monotonic()
        with pytest.raises(EmployeeMessageHandoffUnknownError):
            runtime.wait_for_employee_message_handoff(
                tenant_key="tenant_1",
                agent_id="agt_alpha",
                bot_principal_id="bot_alpha",
                app_id="cli_alpha",
                channel_generation=3,
                connection_id="conn_alpha",
                chat_id="oc_team",
                message_id="om_1",
                timeout=0.05,
            )
        assert time.monotonic() - started < 0.2
    finally:
        release.set()
        holder.join(1)

    assert tuple(stack.writer.replay()) == before
    assert stack.router.record_snapshot(stack.acceptance_id).state == "accepted"
    stack.ingress.close()
    stack.writer.close()


def test_handoff_abandon_writer_lock_contention_expires_without_late_mutation(
    tmp_path,
) -> None:
    stack = _stack(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def occupy_writer_transaction() -> None:
        with stack.writer._transaction_mutex:  # noqa: SLF001 - deterministic contention
            entered.set()
            release.wait(1)

    holder = threading.Thread(target=occupy_writer_transaction)
    holder.start()
    assert entered.wait(1)
    before = tuple(stack.writer.replay())

    try:
        started = time.monotonic()
        with pytest.raises(RouterWriteDisabledError, match="deadline"):
            stack.router.abandon_message_handoff(
                stack.acceptance_id,
                channel_generation=3,
                connection_id="conn_alpha",
                deadline=started + 0.05,
            )
        assert time.monotonic() - started < 0.2
    finally:
        release.set()
        holder.join(1)

    assert tuple(stack.writer.replay()) == before
    assert stack.router.record_snapshot(stack.acceptance_id).state == "accepted"
    stack.ingress.close()
    stack.writer.close()


def test_handoff_abandon_bounds_retries_under_continuous_head_churn(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    original_commit = stack.writer.commit
    churn_count = 0

    def commit_after_unrelated_head_advance(events, expected_versions, **kwargs):
        nonlocal churn_count
        if any(
            event.event_type == "employee.ingress.router_terminal"
            for event in events
        ):
            churn_count += 1
            aggregate_id = f"test-head-churn:{churn_count}"
            drift = JournalEvent(
                event_type="test.concurrent.head_advanced",
                aggregate_id=aggregate_id,
                payload={"attempt": churn_count},
            )
            original_commit(
                (drift,),
                stack.writer.get_aggregate_versions((aggregate_id,)),
            )
        return original_commit(events, expected_versions, **kwargs)

    monkeypatch.setattr(stack.writer, "commit", commit_after_unrelated_head_advance)
    try:
        with pytest.raises(RouterWriteDisabledError, match="stabilize"):
            stack.router.abandon_message_handoff(
                stack.acceptance_id,
                channel_generation=3,
                connection_id="conn_alpha",
            )

        assert churn_count == 4
        stack.router.rebuild_projection()
        assert stack.router.record_snapshot(stack.acceptance_id).state == "accepted"
        assert all(
            event.event_type != "employee.ingress.router_terminal"
            for frame in stack.writer.replay()
            for event in frame.events
        )
    finally:
        stack.ingress.close()
        stack.writer.close()


@pytest.mark.parametrize("state", ("queued", "dispatching"))
def test_handoff_terminal_reducer_rejects_postqueue_journal_history(
    tmp_path,
    state: str,
) -> None:
    stack = _stack(tmp_path)
    try:
        queued = stack.router.route(stack.acceptance_id)
        assert queued.state == "queued"
        if state == "dispatching":
            with stack.writer.transaction_guard(), stack.router._mutex:  # noqa: SLF001
                stack.router.rebuild_projection()
                queued = stack.router.state.by_acceptance_id[stack.acceptance_id]
                stack.router._transition_unlocked(  # noqa: SLF001
                    queued,
                    "dispatching",
                    {},
                )

        record = stack.router.state.by_acceptance_id[stack.acceptance_id]
        assert record.state == state
        invalid = JournalEvent(
            event_type="employee.ingress.router_terminal",
            aggregate_id=record.aggregate_id,
            payload={
                "acceptance_id": stack.acceptance_id,
                "reason_code": "handoff_unconfirmed",
            },
        )
        committed = stack.writer.commit(
            (invalid,),
            stack.writer.get_aggregate_versions((invalid.aggregate_id,)),
        )

        with pytest.raises(RouterProjectionError, match="terminal reason"):
            stack.router.preflight_frame_unlocked(committed.frame)
        with pytest.raises(RouterProjectionError, match="terminal reason"):
            DurableEmployeeIngressRouter(**stack.router_kwargs)
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_payload_retry_budget_and_eligibility_survive_router_restarts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    calls = 0

    def unavailable(_ref):
        nonlocal calls
        calls += 1
        raise KeyResolutionError("temporary key provider secret")

    monkeypatch.setattr(stack.ingress.blob_store, "read", unavailable)
    try:
        assert stack.router.route(stack.acceptance_id).inbox_failures == 1
        restarted = DurableEmployeeIngressRouter(**stack.router_kwargs)
        assert restarted.route(stack.acceptance_id).inbox_failures == 1
        assert calls == 1

        stack.now[0] += timedelta(seconds=1)
        assert restarted.route(stack.acceptance_id).inbox_failures == 2
        restarted = DurableEmployeeIngressRouter(**stack.router_kwargs)
        assert restarted.route(stack.acceptance_id).inbox_failures == 2
        assert calls == 2

        stack.now[0] += timedelta(seconds=2)
        terminal = restarted.route(stack.acceptance_id)
        assert terminal.state == "terminal"
        assert terminal.reason_code == "inbox_not_dispatchable"
        assert calls == 3

        journal = json.dumps(
            [
                [event.to_dict() for event in frame.events]
                for frame in stack.writer.replay()
            ],
            sort_keys=True,
        )
        assert "key provider secret" not in journal
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_payload_retry_eligibility_prevents_unbudgeted_read_on_service_restart(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = _stack(tmp_path)
    calls = 0

    def unavailable(_ref):
        nonlocal calls
        calls += 1
        raise KeyResolutionError("temporary key provider secret")

    monkeypatch.setattr(stack.ingress.blob_store, "read", unavailable)
    assert stack.router.route(stack.acceptance_id).inbox_failures == 1
    assert calls == 1
    stack.ingress.close()

    replacement_store = BlobStore(
        tmp_path / "blobs",
        AesGcmEncryptionProvider(lambda _key_ref: b"i" * 32),
    )
    replacement_read = replacement_store.read
    monkeypatch.setattr(replacement_store, "read", unavailable)
    replacement_ingress = EmployeeIngressService(
        writer=stack.writer,
        blob_store=replacement_store,
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    replacement_kwargs = {
        **stack.router_kwargs,
        "ingress_service": replacement_ingress,
    }
    try:
        restarted = DurableEmployeeIngressRouter(**replacement_kwargs)

        assert restarted.route(stack.acceptance_id).inbox_failures == 1
        assert calls == 1

        stack.now[0] += timedelta(seconds=1)
        monkeypatch.setattr(replacement_store, "read", replacement_read)
        assert restarted.route(stack.acceptance_id).state == "queued"
    finally:
        replacement_ingress.close()
        stack.writer.close()


@pytest.mark.parametrize("damage", ("missing", "corrupt", "auth", "schema", "digest"))
def test_permanent_payload_damage_terminalizes_without_retry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    stack = _stack(tmp_path)
    record = stack.ingress.state.by_acceptance_id[stack.acceptance_id]
    blob_path = stack.ingress.blob_store.root / f"{record.blob_ref.blob_id}.blob"
    if damage == "missing":
        blob_path.unlink()
    elif damage == "corrupt":
        blob_path.write_bytes(b"corrupt")
    elif damage == "auth":
        monkeypatch.setattr(
            stack.ingress.blob_store,
            "read",
            lambda _ref: (_ for _ in ()).throw(
                BlobAuthenticationError("authentication secret")
            ),
        )
    elif damage == "schema":
        monkeypatch.setattr(stack.ingress.blob_store, "read", lambda _ref: b"{}")
    else:
        other = replace(stack.payload, envelope_id="ing_" + "2" * 64)
        monkeypatch.setattr(
            stack.ingress.blob_store,
            "read",
            lambda _ref: other.canonical_bytes,
        )
    try:
        terminal = stack.router.route(stack.acceptance_id)

        assert terminal.state == "terminal"
        assert terminal.reason_code == "inbox_not_dispatchable"
        assert terminal.inbox_failures == 0
    finally:
        stack.ingress.close()
        stack.writer.close()


def test_runtime_does_not_read_or_classify_payload_before_retry_eligibility() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_retry_wait"
    pending = SimpleNamespace(disposition=None)
    routed = SimpleNamespace(state="accepted")
    calls: list[str] = []

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def rebuild_projection(self):
            return None

        def get_payload(self, _acceptance_id: str):
            pytest.fail("payload was read before durable retry eligibility")

        def gc_terminal_payloads(self):
            return 0

    class _Router:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: routed})

        def rebuild_projection(self):
            return None

        def is_inbox_candidate_eligible(self, observed_acceptance_id: str):
            assert observed_acceptance_id == acceptance_id
            calls.append("eligibility")
            return False

        def classify_targeted_group_task(self, *_args):
            pytest.fail("payload was classified before durable retry eligibility")

    class _Dispatch:
        employee_runtime = None

        def dispatch_next(self):
            calls.append("dispatch")
            return None

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Router()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = _Dispatch()  # type: ignore[assignment]  # noqa: SLF001

    assert runtime._drain_employee_dispatch_once() is False  # noqa: SLF001
    assert calls == ["eligibility", "dispatch"]
