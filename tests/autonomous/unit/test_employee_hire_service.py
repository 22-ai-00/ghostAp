"""Focused RED/GREEN contract for durable visible-employee admission."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.autonomous.domain import EmployeeState, WorkerType
from src.autonomous.journal.anchor import FileAnchor, MemoryAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning import lark_app
from src.autonomous.provisioning.hire_port import EmployeeHireRequest
from src.autonomous.provisioning.hire_service import (
    HireAdmissionError,
    ProductionEmployeeHireService,
)
from src.autonomous.provisioning.hire_state import (
    DurableHireState,
    HireEffectState,
    HirePhase,
    HireProjection,
    HireProjectionError,
)
from src.autonomous.provisioning.verification import VerificationChallenge
from src.autonomous.workforce.projection import commit_workforce_events
from src.config.settings import Settings

HMAC_KEY = b"employee-hire-test-hmac-key-32-bytes"


def _request(
    *,
    message_id: str = "om_hire_1",
    employee_name: str = "Atlas",
    existing_app_id: str = "",
) -> EmployeeHireRequest:
    return EmployeeHireRequest(
        employee_name=employee_name,
        tool="codex",
        model="gpt-5.6-sol",
        effort="high",
        chat_id="oc_admin_dm",
        message_id=message_id,
        requester_principal_id="ou_admin",
        requester_union_id="on_admin",
        tenant_key="tenant-a",
        profile="standard",
        role="software engineer",
        persona="careful reviewer",
        personality_traits=("严谨", "主动沟通"),
        capabilities=("coding", "review", "file_read"),
        permissions=("file_read",),
        existing_app_id=existing_app_id,
    )


def test_hire_persists_complete_profile_before_external_provisioning(tmp_path: Path) -> None:
    submitted: list[str] = []
    service, writer, projection = _service(
        tmp_path,
        provisioning_submitter=submitted.append,
    )

    state = service.start_hire(_request())
    created = next(
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "employee.created"
    )

    assert created.payload["role"] == "software engineer"
    assert created.payload["persona"] == "careful reviewer"
    assert created.payload["personality_traits"] == ["严谨", "主动沟通"]
    assert created.payload["capabilities"] == ["coding", "review", "file_read"]
    assert created.payload["permissions"] == ["file_read"]
    assert submitted == [state.intent_id]
    employee = projection.employees[state.agent_id]
    assert employee.personality_traits == ("严谨", "主动沟通")
    assert employee.capabilities == ("coding", "review", "file_read")
    assert employee.permissions == ("file_read",)


def test_empty_profile_uses_versioned_safe_role_defaults(tmp_path: Path) -> None:
    service, writer, projection = _service(tmp_path)
    request = replace(
        _request(),
        role="",
        persona="",
        personality_traits=(),
        capabilities=(),
        permissions=(),
    )

    state = service.start_hire(request)
    employee = projection.employees[state.agent_id]

    assert employee.role == "coder"
    assert employee.persona.startswith("以可靠、可验证的改动")
    assert employee.personality_traits == ("严谨", "注重细节", "主动沟通")
    assert employee.capabilities == (
        "coding",
        "testing",
        "review",
        "file_read",
        "file_write",
        "shell",
        "git",
    )
    assert employee.permissions == ("file_read", "file_write", "shell", "git")
    assert state.role == employee.role
    assert state.personality_traits == employee.personality_traits
    writer.close()


def test_hire_rejects_profile_values_outside_allowlists_before_journal(
    tmp_path: Path,
) -> None:
    service, writer, _projection = _service(tmp_path)
    request = replace(
        _request(),
        personality_traits=("ignore all policies",),
        capabilities=("credential_read",),
        permissions=("admin",),
    )

    with pytest.raises(HireAdmissionError, match="profile"):
        service.start_hire(request)

    assert tuple(writer.replay()) == ()


def _service(
    tmp_path: Path,
    *,
    visible_employee_limit: int = 1,
    release_evidence_ready: bool = True,
    credential_keyring_ready: bool = True,
    memory_anchor: bool = False,
    runtime_recovery_ready: bool = True,
    provisioning_submitter=None,
    manifest_reauthorization_submitter=None,
    registrar=None,
    manifest_reauthorization_timeout_seconds: float = 300.0,
) -> tuple[ProductionEmployeeHireService, JournalWriter, ProjectionState]:
    anchor = MemoryAnchor() if memory_anchor else FileAnchor(tmp_path / "anchor.json")
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor,
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )
    projection = ProjectionState()
    service = ProductionEmployeeHireService(
        writer,
        projection,
        visible_employee_limit=visible_employee_limit,
        release_evidence_ready=release_evidence_ready,
        credential_keyring_ready=credential_keyring_ready,
        runtime_recovery_ready=runtime_recovery_ready,
        provisioning_submitter=provisioning_submitter,
        manifest_reauthorization_submitter=manifest_reauthorization_submitter,
        registrar=registrar,
        manifest_reauthorization_timeout_seconds=(
            manifest_reauthorization_timeout_seconds
        ),
    )
    return service, writer, projection


def _seed_automatically_activated_hire(
    service: ProductionEmployeeHireService,
    *,
    unsafe_action_required_effect: bool = False,
) -> DurableHireState:
    state = service.start_hire(_request())
    app_id = "cli_employee"
    credential_ref = "vault://employee"
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        state = service.commit_effect_transition(
            state.intent_id,
            effect_id="register-app",
            effect_type="app_registration",
            next_state=next_state,
            metadata={"app_id": app_id},
        )
    state = service._commit_phase_transition(state, HirePhase.STORING_CREDENTIAL)
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        state = service.commit_effect_transition(
            state.intent_id,
            effect_id="store-credential",
            effect_type="credential_storage",
            next_state=next_state,
            metadata={
                "app_id": app_id,
                "credential_ref": credential_ref,
            },
        )
    state = service._bind_principal(state, app_id, credential_ref)
    slash_metadata = {
        "app_id": app_id,
        "slash_spec_hash": "slash_hash",
        "slash_observed_hash": "slash_hash",
        "slash_verified_at": "100.0",
        "generation": "1",
    }
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        state = service.commit_effect_transition(
            state.intent_id,
            effect_id="slash-reconcile:1:1",
            effect_type="slash_reconciliation",
            next_state=next_state,
            metadata=slash_metadata,
        )
    channel_metadata = {
        "app_id": app_id,
        "identity_app_id": app_id,
        "generation": "1",
        "connection_id": "conn_generation_1",
        "channel_verified_at": "101.0",
    }
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        state = service.commit_effect_transition(
            state.intent_id,
            effect_id="channel-start:1",
            effect_type="channel_start",
            next_state=next_state,
            metadata=channel_metadata,
        )
    if unsafe_action_required_effect:
        state = service.commit_effect_transition(
            state.intent_id,
            effect_id="unsafe-recovery-evidence",
            effect_type="unsafe_recovery_probe",
            next_state=HireEffectState.PREPARED,
        )
        state = service.commit_effect_transition(
            state.intent_id,
            effect_id="unsafe-recovery-evidence",
            effect_type="unsafe_recovery_probe",
            next_state=HireEffectState.ACTION_REQUIRED,
        )
    state = service.begin_activation_verification(
        VerificationChallenge(
            hire_intent_id=state.intent_id,
            tenant_key=state.tenant_key,
            app_id=app_id,
            agent_id=state.agent_id,
            generation=1,
            requester_principal_id=state.requester_principal_id,
            requester_union_id=state.requester_union_id,
            expected_slash_spec_hash="slash_hash",
            nonce="nonce_generation_1",
            issued_at=102.0,
            expires_at=1_002.0,
        )
    )
    return service.commit_automatic_activation(
        state.intent_id,
        activated_at=103.0,
    )


def _reopen_service(
    tmp_path: Path,
    *,
    writer_epoch: int,
) -> tuple[ProductionEmployeeHireService, JournalWriter]:
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "anchor.json"),
        hmac_key=HMAC_KEY,
        writer_epoch=writer_epoch,
    )
    return (
        ProductionEmployeeHireService(
            writer,
            ProjectionState(),
            visible_employee_limit=1,
            release_evidence_ready=True,
            credential_keyring_ready=True,
        ),
        writer,
    )


def test_new_and_duplicate_hire_submit_only_after_all_commit_locks_release(
    tmp_path: Path,
) -> None:
    import src.autonomous.workforce.projection as workforce_projection

    observations = []
    service = None
    writer = None

    def submit(intent_id):
        assert service is not None and writer is not None
        observations.append(intent_id)
        assert not workforce_projection._WORKFORCE_COMMIT_LOCK._is_owned()
        assert not service._mutex._is_owned()
        assert writer._mutex.acquire(blocking=False)
        writer._mutex.release()

    service, writer, _projection = _service(
        tmp_path,
        provisioning_submitter=submit,
    )
    first = service.start_hire(_request())
    duplicate = service.start_hire(_request())

    assert duplicate == first
    assert observations == [first.intent_id, first.intent_id]


def test_admission_is_anchored_single_frame_and_updates_projection_cursor(
    tmp_path: Path,
) -> None:
    service, writer, projection = _service(tmp_path)

    state = service.start_hire(_request())

    frames = tuple(writer.replay())
    assert len(frames) == 1
    assert len(frames[0].events) == 1
    event = frames[0].events[0]
    assert event.event_type == "employee.created"
    assert event.aggregate_id == state.agent_id
    assert event.payload["hire_intent_id"] == state.intent_id
    assert event.payload["provisioning_attempt_id"] == state.attempt_id
    assert event.payload["planned_bot_principal_id"] == state.bot_principal_id
    assert event.payload["state"] == EmployeeState.PROVISIONING_APP.value
    assert event.payload["worker_type"] == WorkerType.VISIBLE.value
    assert event.payload["requester_union_id"] == "on_admin"
    assert projection.cursor_sequence == frames[0].sequence
    assert projection.cursor_hash == frames[0].frame_hash
    assert projection.employees[state.agent_id].state is EmployeeState.PROVISIONING_APP
    assert projection.employee_name_keys[("tenant-a", "atlas")] == state.agent_id
    assert writer.anchor.read().sequence == frames[0].sequence


def test_same_tenant_message_is_stable_and_idempotent(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path)
    request = _request()

    first = service.start_hire(request)
    second = service.start_hire(request)

    assert second == first
    assert first.intent_id.startswith("hire_")
    assert first.agent_id.startswith("agt_")
    assert first.bot_principal_id.startswith("bot_")
    assert first.attempt_id.startswith("attempt_")
    assert len(tuple(writer.replay())) == 1


def test_same_message_with_changed_request_fails_closed(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path)
    service.start_hire(_request())

    with pytest.raises(HireAdmissionError, match="idempotency"):
        service.start_hire(_request(employee_name="Changed"))

    assert len(tuple(writer.replay())) == 1


def test_same_message_with_changed_existing_app_fails_closed(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path)
    service.start_hire(_request(existing_app_id="cli_first_123"))

    with pytest.raises(HireAdmissionError, match="idempotency"):
        service.start_hire(_request(existing_app_id="cli_second_456"))

    assert len(tuple(writer.replay())) == 1


def test_existing_app_cannot_be_claimed_by_two_live_hires(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path, visible_employee_limit=2)
    service.start_hire(
        _request(
            message_id="om_first",
            employee_name="Atlas",
            existing_app_id="cli_shared_123",
        )
    )

    with pytest.raises(HireAdmissionError, match="existing app already assigned"):
        service.start_hire(
            _request(
                message_id="om_second",
                employee_name="Beacon",
                existing_app_id="cli_shared_123",
            )
        )

    assert len(tuple(writer.replay())) == 1


def test_new_hire_principal_records_desired_manifest_without_remote_evidence(
    tmp_path: Path,
) -> None:
    service, _writer, _projection = _service(tmp_path)
    state = service.start_hire(_request())

    service._bind_principal(state, "cli_employee_123", "cred_employee_123")

    principal = service.synchronize_projection().bot_principals[
        state.bot_principal_id
    ]
    manifest = lark_app.current_registration_manifest()
    assert principal.scopes == manifest.tenant_scopes
    assert principal.desired_manifest_hash == manifest.fingerprint()
    assert principal.observed_manifest_hash == ""


class _ManifestRegistrar:
    def __init__(
        self,
        *,
        result_app_id: str = "cli_employee_123",
        result_manifest_hash: str | None = None,
        trusted: bool = True,
        fail: bool = False,
    ):
        self.result_app_id = result_app_id
        self.result_manifest_hash = result_manifest_hash
        self.trusted = trusted
        self.fail = fail
        self.requests = []

    async def register(self, request, *, on_link, on_status=None):
        self.requests.append(request)
        on_link("https://open.feishu.cn/reauthorize", 60)
        if on_status is not None:
            on_status("polling")
        if self.fail:
            raise RuntimeError("remote registration failed")
        return lark_app.RegistrationResult(
            app_id=self.result_app_id,
            app_secret="remote-secret-123",
            manifest_hash=(
                self.result_manifest_hash
                or lark_app.current_registration_manifest().fingerprint()
            )
            if self.trusted
            else "",
            evidence_source=(
                lark_app.MANIFEST_EVIDENCE_SOURCE if self.trusted else ""
            ),
        )


class _BlockingManifestRegistrar(_ManifestRegistrar):
    async def register(self, request, *, on_link, on_status=None):
        import asyncio

        self.requests.append(request)
        await asyncio.sleep(60)
        raise AssertionError("timeout must cancel the registrar")


def _active_principal(service, writer):
    state = service.start_hire(_request())
    service._bind_principal(state, "cli_employee_123", "cred_employee_123")
    for phase in (
        EmployeeState.VALIDATING,
        EmployeeState.READY_PENDING_VERIFICATION,
        EmployeeState.ACTIVE,
    ):
        commit_workforce_events(
            writer,
            service.projection_state,
            (
                JournalEvent(
                    event_type="employee.state_changed",
                    aggregate_id=state.agent_id,
                    payload={"state": phase.value},
                ),
            ),
        )
    service.synchronize_projection()
    return state


@pytest.mark.asyncio
async def test_active_app_reauthorization_records_observed_only_after_trusted_result(
    tmp_path: Path,
) -> None:
    registrar = _ManifestRegistrar()
    submitted = []
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=submitted.append,
    )
    state = _active_principal(service, writer)

    operation = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id="om_roster_1",
    )
    before = service.synchronize_projection().bot_principals[state.bot_principal_id]
    assert before.observed_manifest_hash == ""
    assert submitted == [operation.operation_id]

    completed = await service.run_manifest_reauthorization(operation.operation_id)

    manifest_hash = lark_app.current_registration_manifest().fingerprint()
    principal = service.synchronize_projection().bot_principals[state.bot_principal_id]
    assert completed.phase.value == "committed"
    assert principal.observed_manifest_hash == manifest_hash
    assert registrar.requests[0].existing_app_id == "cli_employee_123"
    observed_events = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "bot_principal.manifest_observed"
    ]
    assert len(observed_events) == 1
    assert observed_events[0].payload == {
        "observed_manifest_hash": manifest_hash,
        "evidence_source": "lark_oapi.aregister_app/exact_app_id",
    }


@pytest.mark.asyncio
async def test_active_app_reauthorization_failure_is_durable_and_keeps_unknown(
    tmp_path: Path,
) -> None:
    registrar = _ManifestRegistrar(fail=True)
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )
    state = _active_principal(service, writer)
    operation = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id="om_roster_fail",
    )

    with pytest.raises(HireAdmissionError, match="manual action"):
        await service.run_manifest_reauthorization(operation.operation_id)

    principal = service.synchronize_projection().bot_principals[state.bot_principal_id]
    assert principal.observed_manifest_hash == ""
    assert service.get_manifest_reauthorization(operation.operation_id).phase.value == (
        "action_required"
    )
    assert not any(
        event.event_type == "bot_principal.manifest_observed"
        for frame in writer.replay()
        for event in frame.events
    )


@pytest.mark.asyncio
async def test_active_app_reauthorization_rejects_mismatched_remote_app(
    tmp_path: Path,
) -> None:
    registrar = _ManifestRegistrar(result_app_id="cli_other_456")
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )
    state = _active_principal(service, writer)
    operation = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id="om_roster_wrong_app",
    )

    with pytest.raises(HireAdmissionError, match="manual action"):
        await service.run_manifest_reauthorization(operation.operation_id)

    principal = service.synchronize_projection().bot_principals[state.bot_principal_id]
    assert principal.observed_manifest_hash == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registrar",
    [
        _ManifestRegistrar(trusted=False),
        _ManifestRegistrar(result_manifest_hash="sha256:stale-manifest"),
    ],
    ids=["missing-provenance", "manifest-mismatch"],
)
async def test_active_app_reauthorization_rejects_untrusted_receipt(
    tmp_path: Path,
    registrar: _ManifestRegistrar,
) -> None:
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )
    state = _active_principal(service, writer)
    operation = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id="om_roster_untrusted",
    )

    with pytest.raises(HireAdmissionError, match="manual action"):
        await service.run_manifest_reauthorization(operation.operation_id)

    current = service.get_manifest_reauthorization(operation.operation_id)
    assert current.phase.value == "action_required"
    assert service.synchronize_projection().bot_principals[
        state.bot_principal_id
    ].observed_manifest_hash == ""


def test_manifest_reauthorization_is_single_flight_per_principal(
    tmp_path: Path,
) -> None:
    submitted: list[str] = []
    service, writer, _projection = _service(
        tmp_path,
        registrar=_ManifestRegistrar(),
        manifest_reauthorization_submitter=submitted.append,
    )
    state = _active_principal(service, writer)

    def request(request_id: str):
        return service.request_manifest_reauthorization(
            tenant_key=state.tenant_key,
            agent_id=state.agent_id,
            request_id=request_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        operations = tuple(
            pool.map(request, ("om_roster_old_1", "om_roster_old_2"))
        )

    assert operations[0].operation_id == operations[1].operation_id
    assert submitted == [operations[0].operation_id]
    prepared = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "manifest.reauthorization.prepared"
    ]
    assert len(prepared) == 1


@pytest.mark.asyncio
async def test_active_app_reauthorization_timeout_keeps_remote_unknown(
    tmp_path: Path,
) -> None:
    registrar = _BlockingManifestRegistrar()
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
        manifest_reauthorization_timeout_seconds=0.01,
    )
    state = _active_principal(service, writer)
    operation = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id="om_roster_timeout",
    )

    with pytest.raises(HireAdmissionError, match="manual action"):
        await service.run_manifest_reauthorization(operation.operation_id)

    current = service.get_manifest_reauthorization(operation.operation_id)
    assert current.phase.value == "action_required"
    assert current.error_code == "registration_timeout"
    assert service.synchronize_projection().bot_principals[
        state.bot_principal_id
    ].observed_manifest_hash == ""
    with pytest.raises(HireAdmissionError, match="fresh request"):
        service.request_manifest_reauthorization(
            tenant_key=state.tenant_key,
            agent_id=state.agent_id,
            request_id="om_roster_timeout",
        )


def test_recovery_fail_closes_interrupted_remote_reauthorization(
    tmp_path: Path,
) -> None:
    registrar = _ManifestRegistrar()
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )
    state = _active_principal(service, writer)
    operation = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id="om_roster_interrupted",
    )
    service._commit_manifest_reauthorization_event(
        JournalEvent(
            event_type="manifest.reauthorization.executing",
            aggregate_id=operation.operation_id,
            payload={},
        )
    )
    restarted = ProductionEmployeeHireService(
        writer,
        ProjectionState(),
        visible_employee_limit=1,
        release_evidence_ready=True,
        credential_keyring_ready=True,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )

    assert restarted.recover_manifest_reauthorizations() == ()
    recovered = restarted.get_manifest_reauthorization(operation.operation_id)
    assert recovered.phase.value == "action_required"
    assert recovered.error_code == "interrupted_remote_outcome"
    assert restarted.synchronize_projection().bot_principals[
        state.bot_principal_id
    ].observed_manifest_hash == ""


def test_recovery_submits_only_one_legacy_duplicate_prepared_operation(
    tmp_path: Path,
) -> None:
    registrar = _ManifestRegistrar()
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )
    state = _active_principal(service, writer)
    first = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id="om_roster_first",
    )
    duplicate_id = "manifestreauth_legacy_duplicate"
    service._commit_manifest_reauthorization_event(
        JournalEvent(
            event_type="manifest.reauthorization.prepared",
            aggregate_id=duplicate_id,
            payload={
                "tenant_key": first.tenant_key,
                "agent_id": first.agent_id,
                "bot_principal_id": first.bot_principal_id,
                "app_id": first.app_id,
                "desired_manifest_hash": first.desired_manifest_hash,
                "message_id": "om_roster_duplicate",
                "employee_name": first.employee_name,
            },
        )
    )
    restarted = ProductionEmployeeHireService(
        writer,
        ProjectionState(),
        visible_employee_limit=1,
        release_evidence_ready=True,
        credential_keyring_ready=True,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )

    pending = restarted.recover_manifest_reauthorizations()

    assert pending == (first.operation_id,)
    duplicate = restarted.get_manifest_reauthorization(duplicate_id)
    assert duplicate is not None
    assert duplicate.phase.value == "action_required"
    assert duplicate.error_code == "duplicate_active_operation"


@pytest.mark.parametrize("blocking_phase", ["executing", "committed"])
def test_recovery_does_not_restart_prepared_behind_uncertain_or_committed_flow(
    tmp_path: Path,
    blocking_phase: str,
) -> None:
    registrar = _ManifestRegistrar()
    service, writer, _projection = _service(
        tmp_path,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )
    state = _active_principal(service, writer)
    first = service.request_manifest_reauthorization(
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        request_id=f"om_roster_{blocking_phase}",
    )
    service._commit_manifest_reauthorization_event(
        JournalEvent(
            event_type="manifest.reauthorization.executing",
            aggregate_id=first.operation_id,
            payload={},
        )
    )
    if blocking_phase == "committed":
        service._commit_manifest_reauthorization_event(
            JournalEvent(
                event_type="manifest.reauthorization.committed",
                aggregate_id=first.operation_id,
                payload={
                    "observed_manifest_hash": first.desired_manifest_hash,
                    "evidence_source": lark_app.MANIFEST_EVIDENCE_SOURCE,
                },
            )
        )
    duplicate_id = f"manifestreauth_legacy_{blocking_phase}_duplicate"
    service._commit_manifest_reauthorization_event(
        JournalEvent(
            event_type="manifest.reauthorization.prepared",
            aggregate_id=duplicate_id,
            payload={
                "tenant_key": first.tenant_key,
                "agent_id": first.agent_id,
                "bot_principal_id": first.bot_principal_id,
                "app_id": first.app_id,
                "desired_manifest_hash": first.desired_manifest_hash,
                "message_id": f"om_roster_{blocking_phase}_duplicate",
                "employee_name": first.employee_name,
            },
        )
    )
    restarted = ProductionEmployeeHireService(
        writer,
        ProjectionState(),
        visible_employee_limit=1,
        release_evidence_ready=True,
        credential_keyring_ready=True,
        registrar=registrar,
        manifest_reauthorization_submitter=lambda _operation_id: None,
    )

    assert restarted.recover_manifest_reauthorizations() == ()
    duplicate = restarted.get_manifest_reauthorization(duplicate_id)
    assert duplicate is not None
    assert duplicate.phase.value == "action_required"
    assert duplicate.error_code == "duplicate_active_operation"


def test_hire_replay_rejects_duplicate_live_existing_app_claim(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path, visible_employee_limit=2)
    service.start_hire(
        _request(existing_app_id="cli_shared_123")
    )
    first_frame = tuple(writer.replay())[0]
    first_event = first_frame.events[0]
    second_event = JournalEvent(
        event_type="employee.created",
        aggregate_id="agt_second",
        payload={
            **first_event.payload,
            "agent_id": "agt_second",
            "name": "Beacon",
            "hire_intent_id": "hire_second",
            "hire_message_id": "om_second",
            "planned_bot_principal_id": "bot_second",
            "provisioning_attempt_id": "attempt_second",
        },
    )

    with pytest.raises(HireProjectionError, match="duplicate existing_app_id admission"):
        HireProjection.rebuild(
            (
                first_frame,
                SimpleNamespace(sequence=2, events=(second_event,)),
            )
        )


def test_concurrent_existing_app_claim_submits_only_one_hire(tmp_path: Path) -> None:
    submitted: list[str] = []
    service, writer, _projection = _service(
        tmp_path,
        visible_employee_limit=2,
        provisioning_submitter=submitted.append,
    )
    requests = (
        _request(
            message_id="om_first",
            employee_name="Atlas",
            existing_app_id="cli_shared_123",
        ),
        _request(
            message_id="om_second",
            employee_name="Beacon",
            existing_app_id="cli_shared_123",
        ),
    )

    def admit(request):
        try:
            return service.start_hire(request)
        except HireAdmissionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(admit, requests))

    assert sum(isinstance(result, DurableHireState) for result in results) == 1
    assert sum(isinstance(result, HireAdmissionError) for result in results) == 1
    assert len(submitted) == 1
    assert len(tuple(writer.replay())) == 1


def test_archived_hire_releases_existing_app_for_rehire(tmp_path: Path) -> None:
    service, writer, projection = _service(tmp_path, visible_employee_limit=2)
    previous = service.start_hire(
        _request(existing_app_id="cli_shared_123")
    )
    commit_workforce_events(
        writer,
        projection,
        (
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id=previous.agent_id,
                payload={"state": "archived"},
            ),
        ),
    )

    current = service.start_hire(
        _request(
            message_id="om_second",
            employee_name="Beacon",
            existing_app_id="cli_shared_123",
        )
    )

    assert current.agent_id != previous.agent_id
    assert current.existing_app_id == "cli_shared_123"
    assert len(HireProjection.rebuild(writer.replay()).states) == 2


def test_existing_app_claim_rejects_authoritative_principal_binding(
    tmp_path: Path,
) -> None:
    service, writer, _projection = _service(tmp_path, visible_employee_limit=2)
    first = service.start_hire(_request())
    service._bind_principal(first, "cli_shared_123", "cred_first")

    with pytest.raises(HireAdmissionError, match="existing app already assigned"):
        service.start_hire(
            _request(
                message_id="om_second",
                employee_name="Beacon",
                existing_app_id="cli_shared_123",
            )
        )

    assert len(tuple(writer.replay())) == 2


def test_archived_principal_binding_releases_app_for_existing_app_hire(
    tmp_path: Path,
) -> None:
    service, writer, projection = _service(tmp_path, visible_employee_limit=2)
    first = service.start_hire(_request())
    service._bind_principal(first, "cli_shared_123", "cred_first")
    commit_workforce_events(
        writer,
        projection,
        (
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id=first.agent_id,
                payload={"state": "archived"},
            ),
        ),
    )

    second = service.start_hire(
        _request(
            message_id="om_second",
            employee_name="Beacon",
            existing_app_id="cli_shared_123",
        )
    )

    assert second.existing_app_id == "cli_shared_123"


def test_default_registration_cannot_select_app_owned_by_live_principal(
    tmp_path: Path,
) -> None:
    service, _writer, _projection = _service(tmp_path, visible_employee_limit=2)
    first = service.start_hire(_request())
    service._bind_principal(first, "cli_shared_123", "cred_first")
    second = service.start_hire(
        _request(message_id="om_second", employee_name="Beacon")
    )
    for next_state in (HireEffectState.PREPARED, HireEffectState.EXECUTING):
        second = service.commit_effect_transition(
            second.intent_id,
            effect_id="register-app",
            effect_type="app_registration",
            next_state=next_state,
        )

    with pytest.raises(HireAdmissionError, match="registered app already assigned"):
        service._commit_registered_app(second.intent_id, "cli_shared_123")

    current = service._hire_projection.get(second.intent_id)
    assert current is not None
    assert current.effect_state("register-app") is HireEffectState.EXECUTING


def test_default_registration_result_reserves_app_for_later_admission(
    tmp_path: Path,
) -> None:
    service, writer, _projection = _service(tmp_path, visible_employee_limit=2)
    first = service.start_hire(_request())
    for next_state in (HireEffectState.PREPARED, HireEffectState.EXECUTING):
        first = service.commit_effect_transition(
            first.intent_id,
            effect_id="register-app",
            effect_type="app_registration",
            next_state=next_state,
        )
    first = service._commit_registered_app(first.intent_id, "cli_shared_123")

    with pytest.raises(HireAdmissionError, match="existing app already assigned"):
        service.start_hire(
            _request(
                message_id="om_second",
                employee_name="Beacon",
                existing_app_id="cli_shared_123",
            )
        )

    assert dict(first.metadata_for("register-app"))["app_id"] == "cli_shared_123"
    assert len(tuple(writer.replay())) == 4


def test_concurrent_default_registration_results_reserve_app_once(
    tmp_path: Path,
) -> None:
    service, _writer, _projection = _service(tmp_path, visible_employee_limit=2)
    states = [
        service.start_hire(
            _request(message_id="om_first", employee_name="Atlas")
        ),
        service.start_hire(
            _request(message_id="om_second", employee_name="Beacon")
        ),
    ]
    for index, state in enumerate(states):
        for next_state in (HireEffectState.PREPARED, HireEffectState.EXECUTING):
            state = service.commit_effect_transition(
                state.intent_id,
                effect_id="register-app",
                effect_type="app_registration",
                next_state=next_state,
            )
        states[index] = state

    def reserve(state):
        try:
            return service._commit_registered_app(
                state.intent_id,
                "cli_shared_123",
            )
        except HireAdmissionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, states))

    assert sum(isinstance(result, DurableHireState) for result in results) == 1
    assert sum(isinstance(result, HireAdmissionError) for result in results) == 1


@pytest.mark.parametrize(
    "existing_app_id",
    ("cli_", "not_cli_existing", "cli_bad/app"),
)
def test_hire_service_rejects_invalid_existing_app_before_journal(
    tmp_path: Path,
    existing_app_id: str,
) -> None:
    service, writer, _projection = _service(tmp_path)

    with pytest.raises(HireAdmissionError, match="existing_app_id"):
        service.start_hire(_request(existing_app_id=existing_app_id))

    assert tuple(writer.replay()) == ()


def test_hire_replay_rejects_invalid_existing_app_identity(tmp_path: Path) -> None:
    from src.autonomous.provisioning.hire_state import HireProjectionError

    service, writer, _projection = _service(tmp_path)
    service.start_hire(_request(existing_app_id="cli_existing_123"))
    frame = tuple(writer.replay())[0]
    event = frame.events[0]
    invalid_event = JournalEvent(
        event_type=event.event_type,
        aggregate_id=event.aggregate_id,
        payload={**event.payload, "existing_app_id": "cli_bad/app"},
    )

    with pytest.raises(HireProjectionError, match="existing_app_id"):
        HireProjection.rebuild((replace(frame, events=(invalid_event,)),))


def test_duplicate_tenant_name_is_rejected_without_second_frame(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path, visible_employee_limit=2)
    service.start_hire(_request())

    with pytest.raises(HireAdmissionError, match="name"):
        service.start_hire(_request(message_id="om_hire_2", employee_name="atlas"))

    assert len(tuple(writer.replay())) == 1


def test_same_name_rehire_atomically_releases_archived_tombstone(tmp_path: Path) -> None:
    service, writer, projection = _service(tmp_path, visible_employee_limit=2)
    previous = service.start_hire(_request())
    commit_workforce_events(
        writer,
        projection,
        (
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id=previous.agent_id,
                payload={"state": "archived"},
            ),
        ),
    )

    current = service.start_hire(
        _request(message_id="om_hire_2", employee_name="ATLAS")
    )

    assert current.agent_id != previous.agent_id
    assert projection.employees[previous.agent_id].state is EmployeeState.ARCHIVED
    assert projection.employee_name_keys[("tenant-a", "atlas")] == current.agent_id
    assert [event.event_type for event in tuple(writer.replay())[-1].events] == [
        "employee.name_released",
        "employee.created",
    ]


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"visible_employee_limit": 0}, "visible_employee_limit"),
        ({"release_evidence_ready": False}, "release_evidence"),
        ({"credential_keyring_ready": False}, "credential_keyring"),
        ({"memory_anchor": True}, "production_anchor"),
        ({"runtime_recovery_ready": False}, "runtime_recovery"),
    ],
)
def test_readiness_gates_reject_before_journal_write(
    tmp_path: Path,
    overrides: dict[str, object],
    blocker: str,
) -> None:
    service, writer, projection = _service(tmp_path, **overrides)

    readiness = service.readiness()
    assert readiness.ready is False
    assert blocker in readiness.blockers
    with pytest.raises(HireAdmissionError, match=blocker):
        service.start_hire(_request())

    assert tuple(writer.replay()) == ()
    assert projection.cursor_sequence == 0


def test_admission_opens_only_after_runtime_recovery_completes(
    tmp_path: Path,
) -> None:
    service, writer, _projection = _service(
        tmp_path,
        runtime_recovery_ready=False,
    )
    with pytest.raises(HireAdmissionError, match="runtime_recovery"):
        service.start_hire(_request())

    service.mark_runtime_recovered()
    admitted = service.start_hire(_request())

    assert admitted.intent_id
    assert len(tuple(writer.replay())) == 1


def test_recover_rebuilds_hire_and_canonical_projection(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path)
    admitted = service.start_hire(_request())
    service.close()

    reopened_writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "anchor.json"),
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    recovered_service = ProductionEmployeeHireService(
        reopened_writer,
        ProjectionState(),
        visible_employee_limit=1,
        release_evidence_ready=True,
        credential_keyring_ready=True,
    )

    recovered = recovered_service.recover()

    assert recovered.get(admitted.intent_id) == admitted
    assert recovered_service.projection_state.cursor_sequence == 1
    assert recovered_service.start_hire(_request()) == admitted
    assert len(tuple(reopened_writer.replay())) == 1
    recovered_service.close()


def test_recover_reopens_exact_phase_only_shutdown_channel_history(
    tmp_path: Path,
) -> None:
    first, _writer, _projection = _service(tmp_path)
    active = _seed_automatically_activated_hire(first)
    assert active.phase is HirePhase.ACTIVE
    assert active.verification_consumed is True
    first.begin_channel_revalidation(
        active.intent_id,
        observed_generation=active.channel_generation,
    )
    first.close()

    restarted, restarted_writer = _reopen_service(tmp_path, writer_epoch=2)
    after_constructor = restarted.get_state(active.intent_id)
    assert after_constructor is not None
    assert after_constructor.phase is HirePhase.READY_PENDING_VERIFICATION

    recovered = restarted.recover().get(active.intent_id)

    assert recovered is not None
    assert recovered.phase is HirePhase.VALIDATING
    assert recovered.channel_generation == active.channel_generation
    assert recovered.verification_consumed is True
    assert all(
        effect_state is HireEffectState.COMMITTED
        for _effect_id, effect_state in recovered.effects
    )
    recovery_frames = [
        frame
        for frame in restarted_writer.replay()
        if any(
            event.event_type == "hire.channel.phase_only_recovery"
            for event in frame.events
        )
    ]
    assert len(recovery_frames) == 1
    assert [event.event_type for event in recovery_frames[0].events] == [
        "hire.channel.phase_only_recovery",
        "employee.state_changed",
    ]
    restarted.close()


def test_recover_refuses_shutdown_history_with_action_required_effect(
    tmp_path: Path,
) -> None:
    first, _writer, _projection = _service(tmp_path)
    active = _seed_automatically_activated_hire(
        first,
        unsafe_action_required_effect=True,
    )
    first.begin_channel_revalidation(
        active.intent_id,
        observed_generation=active.channel_generation,
    )
    first.close()

    restarted, restarted_writer = _reopen_service(tmp_path, writer_epoch=2)
    refused = restarted.recover().get(active.intent_id)

    assert refused is not None
    assert refused.phase is HirePhase.ACTION_REQUIRED
    assert (
        refused.effect_state("unsafe-recovery-evidence")
        is HireEffectState.ACTION_REQUIRED
    )
    assert not any(
        event.event_type == "hire.channel.phase_only_recovery"
        for frame in restarted_writer.replay()
        for event in frame.events
    )
    restarted.close()


def test_phase_only_recovery_survives_crash_between_pending_and_disposal(
    tmp_path: Path,
) -> None:
    first, _writer, _projection = _service(tmp_path)
    active = _seed_automatically_activated_hire(first)
    first.begin_channel_revalidation(
        active.intent_id,
        observed_generation=active.channel_generation,
    )
    first.close()

    pending_service, _pending_writer = _reopen_service(
        tmp_path,
        writer_epoch=2,
    )
    pending = pending_service.get_state(active.intent_id)
    assert pending is not None
    assert pending.phase is HirePhase.READY_PENDING_VERIFICATION
    pending_service.close()

    disposal_service, disposal_writer = _reopen_service(
        tmp_path,
        writer_epoch=3,
    )
    recovered = disposal_service.recover().get(active.intent_id)

    assert recovered is not None
    assert recovered.phase is HirePhase.VALIDATING
    assert recovered.verification_consumed is True
    recovery_frames = [
        frame
        for frame in disposal_writer.replay()
        if any(
            event.event_type == "hire.channel.phase_only_recovery"
            for event in frame.events
        )
    ]
    assert len(recovery_frames) == 1
    disposal_service.close()


def test_guarded_reopen_stays_pending_after_new_channel_before_fresh_challenge(
    tmp_path: Path,
) -> None:
    first, _writer, _projection = _service(tmp_path)
    active = _seed_automatically_activated_hire(first)
    first.begin_channel_revalidation(
        active.intent_id,
        observed_generation=active.channel_generation,
    )
    first.close()
    restarted, _restarted_writer = _reopen_service(tmp_path, writer_epoch=2)
    reopened = restarted.recover().get(active.intent_id)
    assert reopened is not None
    assert reopened.phase is HirePhase.VALIDATING

    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        reopened = restarted.commit_effect_transition(
            reopened.intent_id,
            effect_id="slash-reconcile:2:1",
            effect_type="slash_reconciliation",
            next_state=next_state,
            metadata=(
                {
                    "slash_spec_hash": "slash_hash",
                    "slash_observed_hash": "slash_hash",
                    "slash_verified_at": "104.0",
                }
                if next_state is HireEffectState.COMMITTED
                else None
            ),
        )
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        reopened = restarted.commit_effect_transition(
            reopened.intent_id,
            effect_id="channel-start:2",
            effect_type="employee_channel_start",
            next_state=next_state,
            metadata=(
                {
                    "app_id": reopened.app_id,
                    "identity_app_id": reopened.app_id,
                    "generation": "2",
                    "connection_id": "conn_generation_2",
                    "channel_verified_at": "105.0",
                }
                if next_state is HireEffectState.COMMITTED
                else None
            ),
        )

    interrupted = restarted.recover().get(active.intent_id)

    assert interrupted is not None
    assert interrupted.phase is HirePhase.VALIDATING
    assert interrupted.channel_generation == 2
    assert interrupted.verification_consumed is True
    restarted.close()


def test_hire_state_and_replay_projection_are_frozen(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path)
    state = service.start_hire(_request())
    rebuilt = HireProjection.rebuild(writer.replay())

    assert rebuilt.get(state.intent_id) == state
    assert state.phase is HirePhase.PROVISIONING_APP
    assert state.effects == ()
    assert HireEffectState.PREPARED.value == "prepared"
    with pytest.raises(FrozenInstanceError):
        state.phase = HirePhase.ACTIVE  # type: ignore[misc]
    with pytest.raises(TypeError):
        rebuilt.states[state.intent_id] = DurableHireState()  # type: ignore[index]

    service.close()


def test_effect_transition_commit_is_anchored_and_keeps_projection_cursor_current(
    tmp_path: Path,
) -> None:
    service, writer, projection = _service(tmp_path)
    admitted = service.start_hire(_request())

    prepared = service.commit_effect_transition(
        admitted.intent_id,
        effect_id="register-app",
        effect_type="app_registration",
        next_state=HireEffectState.PREPARED,
    )
    executing = service.commit_effect_transition(
        admitted.intent_id,
        effect_id="register-app",
        effect_type="app_registration",
        next_state=HireEffectState.EXECUTING,
    )

    frames = tuple(writer.replay())
    assert [frame.events[0].event_type for frame in frames] == [
        "employee.created",
        "hire.effect.prepared",
        "hire.effect.executing",
    ]
    assert prepared.effect_state("register-app") is HireEffectState.PREPARED
    assert executing.effect_state("register-app") is HireEffectState.EXECUTING
    assert projection.cursor_sequence == 3
    assert projection.cursor_hash == frames[-1].frame_hash
    assert writer.anchor.read().sequence == 3


def test_invalid_effect_transition_rejects_before_journal_write(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path)
    admitted = service.start_hire(_request())

    with pytest.raises(HireAdmissionError, match="effect transition"):
        service.commit_effect_transition(
            admitted.intent_id,
            effect_id="register-app",
            effect_type="app_registration",
            next_state=HireEffectState.EXECUTING,
        )

    assert len(tuple(writer.replay())) == 1


def test_closed_service_rejects_even_an_idempotent_request(tmp_path: Path) -> None:
    service, _writer, _projection = _service(tmp_path)
    request = _request()
    service.start_hire(request)
    service.close()

    with pytest.raises(HireAdmissionError, match="closed"):
        service.start_hire(request)


def test_hire_request_keeps_old_callers_compatible_but_service_requires_tenant(
    tmp_path: Path,
) -> None:
    legacy = EmployeeHireRequest(
        employee_name="Atlas",
        tool="codex",
        model="model",
        effort="high",
        chat_id="oc_dm",
        message_id="om_legacy",
        requester_principal_id="ou_admin",
    )
    assert legacy.tenant_key == ""
    assert legacy.profile == "standard"
    assert legacy.role == ""
    assert legacy.persona == ""
    assert legacy.personality_traits == ()
    assert legacy.capabilities == ()
    assert legacy.permissions == ()
    assert legacy.requester_union_id == ""
    service, writer, _projection = _service(tmp_path)

    with pytest.raises(HireAdmissionError, match="tenant_key"):
        service.start_hire(legacy)

    assert tuple(writer.replay()) == ()


def test_legacy_pending_hire_can_durably_bind_cross_app_identity(
    tmp_path: Path,
) -> None:
    service, writer, _projection = _service(tmp_path)
    pending = service.start_hire(replace(_request(), requester_union_id=""))

    bound = service.bind_requester_union_id(pending.intent_id, "on_admin")
    replayed = service.recover().get(pending.intent_id)

    assert bound.requester_union_id == "on_admin"
    assert replayed is not None and replayed.requester_union_id == "on_admin"
    assert [
        event.event_type for frame in writer.replay() for event in frame.events
    ][-1] == "hire.requester_identity_bound"


@pytest.mark.parametrize(
    "hire_request",
    [
        replace(_request(), effort="potato"),
        replace(_request(), model="gpt-5.6-sol/xhigh", effort="xhigh"),
        replace(_request(), profile="max"),
    ],
)
def test_hire_rejects_noncanonical_or_unsupported_model_components(
    tmp_path: Path,
    hire_request: EmployeeHireRequest,
) -> None:
    service, writer, _projection = _service(tmp_path)

    with pytest.raises(HireAdmissionError, match="invalid employee model selection"):
        service.start_hire(hire_request)

    assert tuple(writer.replay()) == ()


def test_journal_hmac_and_anchor_settings_are_strict_redacted_and_fail_closed() -> None:
    defaults = Settings(_env_file=None)
    assert defaults.autonomous_journal_hmac_key.get_secret_value() == ""
    assert defaults.autonomous_anchor_path == "~/.ghostap/autonomy/journal.anchor"
    assert defaults.autonomous_anchor_provider == ""
    assert defaults.autonomous_visible_employee_limit == 8
    assert defaults.autonomous_employee_auto_activation is True
    assert defaults.autonomous_employee_release_trust_socket == ""
    assert defaults.autonomous_employee_release_trust_timeout_seconds == 2.0
    assert defaults.autonomous_main_bot_audit_dir.endswith("main-bot-send-audit")
    assert defaults.autonomous_main_bot_audit_anchor_path.endswith(
        "main-bot-send-audit.anchor"
    )

    encoded = base64.urlsafe_b64encode(b"k" * 32).decode()
    configured = Settings(
        _env_file=None,
        autonomous_journal_hmac_key=encoded,
        autonomous_anchor_provider="file",
        autonomous_anchor_path="/var/lib/ghostap/journal.anchor",
    )
    assert encoded not in repr(configured)

    with pytest.raises(ValueError, match="autonomous_journal_hmac_key"):
        Settings(_env_file=None, autonomous_journal_hmac_key="too-short")
    with pytest.raises(ValueError, match="autonomous_anchor_provider"):
        Settings(_env_file=None, autonomous_anchor_provider="file anchor")
    with pytest.raises(ValueError, match="rejects booleans"):
        Settings(
            _env_file=None,
            autonomous_employee_release_trust_timeout_seconds=True,
        )
