"""Durable retry contract for authenticated employee ingress payload reads."""

from __future__ import annotations

import json
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
    RouterQueueLimits,
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
from src.autonomous.journal.frame import GENESIS_HASH
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.journal.writer import JournalWriter
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
        message_id="om_1",
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_team",
        thread_root_message_id="om_root",
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
