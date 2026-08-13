"""Startup recovery for safely replayable ACTION_REQUIRED employees."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.autonomous.domain import EmployeeState, WorkerType
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionState
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning import composition, hire_service
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.provisioning.hire_service import ProductionEmployeeHireService
from src.autonomous.provisioning.hire_state import (
    DurableHireState,
    HireEffectState,
    HirePhase,
    HireProjection,
)
from src.autonomous.supervisor.employee_channels import (
    ChannelProcessState,
    ChannelProcessStatus,
)
from src.autonomous.workforce.projection import commit_workforce_events

HMAC_KEY = b"employee-runtime-recovery-key-32!"


@pytest.fixture
def run_asyncio_threads_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)


def _writer(tmp_path: Path, epoch: int) -> JournalWriter:
    return JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "anchor.json"),
        hmac_key=HMAC_KEY,
        writer_epoch=epoch,
    )


def _service(writer: JournalWriter) -> ProductionEmployeeHireService:
    return ProductionEmployeeHireService(
        writer,
        ProjectionState(),
        visible_employee_limit=1,
        release_evidence_ready=True,
        credential_keyring_ready=True,
    )


def _seed_active_employee(
    tmp_path: Path,
    *,
    begin_revalidation: bool = True,
) -> ProductionEmployeeHireService:
    writer = _writer(tmp_path, 1)
    projection = ProjectionState()
    commit_workforce_events(
        writer,
        projection,
        (
            JournalEvent(
                event_type="employee.created",
                aggregate_id="agt_recover",
                payload={
                    "agent_id": "agt_recover",
                    "tenant_key": "tenant-a",
                    "owner_principal_id": "ou_admin",
                    "requester_union_id": "on_admin",
                    "name": "Atlas",
                    "tool": "codex",
                    "model": "gpt-5.6-sol",
                    "effort": "high",
                    "profile": "standard",
                    "role": "software engineer",
                    "persona": "careful reviewer",
                    "personality_traits": [],
                    "capabilities": [],
                    "permissions": [],
                    "worker_type": "visible",
                    "state": "provisioning_app",
                    "hire_schema_version": 1,
                    "hire_intent_id": "hire_recover",
                    "hire_message_id": "om_recover",
                    "hire_chat_id": "oc_admin_dm",
                    "planned_bot_principal_id": "bot_recover",
                    "provisioning_attempt_id": "attempt_recover",
                },
            ),
        ),
    )
    service = ProductionEmployeeHireService(
        writer,
        projection,
        visible_employee_limit=1,
        release_evidence_ready=True,
        credential_keyring_ready=True,
    )
    intent_id = "hire_recover"
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        service.commit_effect_transition(
            intent_id,
            effect_id="register-app",
            effect_type="app_registration",
            next_state=next_state,
            metadata={"app_id": "cli_employee"}
            if next_state is HireEffectState.COMMITTED
            else None,
        )
    state = service.get_state(intent_id)
    assert state is not None
    service._commit_phase_transition(state, HirePhase.STORING_CREDENTIAL)  # noqa: SLF001
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        service.commit_effect_transition(
            intent_id,
            effect_id="store-credential",
            effect_type="credential_vault_put",
            next_state=next_state,
            metadata={"app_id": "cli_employee", "credential_ref": "vault://employee"},
        )
    state = service.get_state(intent_id)
    assert state is not None
    service._bind_principal(state, "cli_employee", "vault://employee")  # noqa: SLF001
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        service.commit_effect_transition(
            intent_id,
            effect_id="slash-reconcile:1:1",
            effect_type="slash_reconciliation",
            next_state=next_state,
            metadata={
                "slash_spec_hash": "slash_hash",
                "slash_observed_hash": "slash_hash",
                "slash_verified_at": "100.0",
            }
            if next_state is HireEffectState.COMMITTED
            else None,
        )
    for next_state in (
        HireEffectState.PREPARED,
        HireEffectState.EXECUTING,
        HireEffectState.COMMITTED,
    ):
        service.commit_effect_transition(
            intent_id,
            effect_id="channel-start:1",
            effect_type="employee_channel_start",
            next_state=next_state,
            metadata={
                "app_id": "cli_employee",
                "generation": "1",
                "identity_app_id": "cli_employee",
                "connection_id": "conn_generation_1",
                "channel_verified_at": "101.0",
            }
            if next_state is HireEffectState.COMMITTED
            else None,
        )
    state = service.get_state(intent_id)
    assert state is not None
    state = service._commit_phase_transition(state, HirePhase.VALIDATING)  # noqa: SLF001
    state = service._commit_phase_transition(  # noqa: SLF001
        state,
        HirePhase.READY_PENDING_VERIFICATION,
    )
    service.commit_automatic_activation(intent_id, activated_at=102.0)
    if begin_revalidation:
        service.begin_channel_revalidation(intent_id, observed_generation=1)
    service.close()
    return service


def test_normal_channel_revalidation_automatic_activation_replays_after_restart(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _service(_writer(tmp_path, 2))
    try:
        for next_state in (
            HireEffectState.PREPARED,
            HireEffectState.EXECUTING,
            HireEffectState.COMMITTED,
        ):
            service.commit_effect_transition(
                "hire_recover",
                effect_id="slash-reconcile:2:1",
                effect_type="slash_reconciliation",
                next_state=next_state,
                metadata={
                    "slash_spec_hash": "slash_hash",
                    "slash_observed_hash": "slash_hash",
                    "slash_verified_at": "200.0",
                }
                if next_state is HireEffectState.COMMITTED
                else None,
            )
        for next_state in (
            HireEffectState.PREPARED,
            HireEffectState.EXECUTING,
            HireEffectState.COMMITTED,
        ):
            service.commit_effect_transition(
                "hire_recover",
                effect_id="channel-start:2",
                effect_type="employee_channel_start",
                next_state=next_state,
                metadata={
                    "app_id": "cli_employee",
                    "generation": "2",
                    "identity_app_id": "cli_employee",
                    "connection_id": "conn_generation_2",
                    "channel_verified_at": "201.0",
                }
                if next_state is HireEffectState.COMMITTED
                else None,
            )
        service.prepare_automatic_activation("hire_recover")
        activated = service.commit_automatic_activation(
            "hire_recover",
            activated_at=202.0,
        )
        assert activated.phase is HirePhase.ACTIVE
        assert activated.automatic_reactivation_generation == 0
    finally:
        service.close()

    reopened = _service(_writer(tmp_path, 3))
    try:
        state = reopened.get_state("hire_recover")
        assert state is not None
        assert state.phase is HirePhase.ACTIVE
        assert state.channel_generation == 2
        assert state.verification_consumed is True
        assert state.automatic_reactivation_generation == 0
    finally:
        reopened.close()


@pytest.mark.parametrize(
    (
        "status_app_id",
        "status_generation",
        "status_tenant_key",
        "status_bot_principal_id",
        "identity_app_id",
        "connection_id",
    ),
    [
        (
            "cli_employee",
            1,
            "tenant-a",
            "bot_recover",
            "cli_employee",
            "conn_reconnected",
        ),
        (
            "cli_employee",
            1,
            "tenant-a",
            "bot_recover",
            "cli_reconnected",
            "conn_generation_1",
        ),
        (
            "cli_other",
            1,
            "tenant-a",
            "bot_recover",
            "cli_employee",
            "conn_generation_1",
        ),
        (
            "cli_employee",
            2,
            "tenant-a",
            "bot_recover",
            "cli_employee",
            "conn_generation_1",
        ),
        (
            "cli_employee",
            1,
            "tenant-other",
            "bot_recover",
            "cli_employee",
            "conn_generation_1",
        ),
        (
            "cli_employee",
            1,
            "tenant-a",
            "bot_other",
            "cli_employee",
            "conn_generation_1",
        ),
    ],
)
def test_channel_monitor_revalidates_ready_transport_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_app_id: str,
    status_generation: int,
    status_tenant_key: str,
    status_bot_principal_id: str,
    identity_app_id: str,
    connection_id: str,
) -> None:
    _seed_active_employee(tmp_path, begin_revalidation=False)
    service = _service(_writer(tmp_path, 2))
    status = ChannelProcessStatus(
        agent_id="agt_recover",
        app_id=status_app_id,
        generation=status_generation,
        pid=101,
        state=ChannelProcessState.READY,
        tenant_key=status_tenant_key,
        bot_principal_id=status_bot_principal_id,
        identity={"app_id": identity_app_id},
        ready_metadata={"connection_id": connection_id},
    )
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._channels = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        status=lambda agent_id: status if agent_id == "agt_recover" else None
    )
    submitted: list[str] = []
    monkeypatch.setattr(runtime, "_submit_intent", submitted.append)

    async def stop_after_cycle(_delay: float) -> None:
        runtime._closing = True  # noqa: SLF001

    monkeypatch.setattr(composition.asyncio, "sleep", stop_after_cycle)
    try:
        asyncio.run(runtime._monitor_channels())  # noqa: SLF001

        state = service.get_state("hire_recover")
        assert state is not None
        assert state.phase is HirePhase.VALIDATING
        assert state.channel_generation == 1
        assert state.automatic_reactivation_generation == 1
        assert runtime._target_channel_generation(state) == 2  # noqa: SLF001
        assert submitted == ["hire_recover"]
    finally:
        service.close()


def test_channel_monitor_leaves_matching_ready_transport_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_active_employee(tmp_path, begin_revalidation=False)
    service = _service(_writer(tmp_path, 2))
    status = ChannelProcessStatus(
        agent_id="agt_recover",
        app_id="cli_employee",
        generation=1,
        pid=101,
        state=ChannelProcessState.READY,
        tenant_key="tenant-a",
        bot_principal_id="bot_recover",
        identity={"app_id": "cli_employee"},
        ready_metadata={"connection_id": "conn_generation_1"},
    )
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._channels = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        status=lambda agent_id: status if agent_id == "agt_recover" else None
    )
    submitted: list[str] = []
    monkeypatch.setattr(runtime, "_submit_intent", submitted.append)

    async def stop_after_cycle(_delay: float) -> None:
        runtime._closing = True  # noqa: SLF001

    monkeypatch.setattr(composition.asyncio, "sleep", stop_after_cycle)
    try:
        asyncio.run(runtime._monitor_channels())  # noqa: SLF001

        state = service.get_state("hire_recover")
        assert state is not None
        assert state.phase is HirePhase.ACTIVE
        assert state.channel_generation == 1
        assert state.automatic_reactivation_generation == 0
        assert submitted == []
    finally:
        service.close()


def test_submit_intent_coalesces_inflight_revalidation_into_one_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._loop = object()  # type: ignore[assignment]  # noqa: SLF001
    submitted: list[concurrent.futures.Future[None]] = []

    def submit(coroutine, _loop):
        coroutine.close()
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        submitted.append(future)
        return future

    monkeypatch.setattr(
        composition.asyncio,
        "run_coroutine_threadsafe",
        submit,
    )

    runtime._submit_intent("hire_reconnect")  # noqa: SLF001
    runtime._submit_intent("hire_reconnect")  # noqa: SLF001
    runtime._submit_intent("hire_reconnect")  # noqa: SLF001
    assert len(submitted) == 1

    submitted[0].set_result(None)
    assert len(submitted) == 2

    submitted[1].set_result(None)
    assert len(submitted) == 2


def test_submit_intent_does_not_resurrect_pending_followup_during_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._loop = object()  # type: ignore[assignment]  # noqa: SLF001
    submitted: list[concurrent.futures.Future[None]] = []

    def submit(coroutine, _loop):
        coroutine.close()
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        submitted.append(future)
        return future

    monkeypatch.setattr(
        composition.asyncio,
        "run_coroutine_threadsafe",
        submit,
    )

    runtime._submit_intent("hire_reconnect")  # noqa: SLF001
    runtime._submit_intent("hire_reconnect")  # noqa: SLF001
    runtime._closing = True  # noqa: SLF001
    submitted[0].set_result(None)

    assert len(submitted) == 1
    assert runtime._pending_intent_resubmits == set()  # noqa: SLF001


def test_effect_replay_accepts_exact_duplicate_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    from src.autonomous.provisioning.hire_state import (
        HireProjection,
        HireProjectionError,
    )

    _seed_active_employee(tmp_path)
    service = _service(_writer(tmp_path, 2))
    try:
        service.commit_effect_transition(
            "hire_recover",
            effect_id="slash-reconcile:2:1",
            effect_type="slash_reconciliation",
            next_state=HireEffectState.PREPARED,
        )
    finally:
        service.close()

    writer = _writer(tmp_path, 3)
    exact_payload = {
        "effect_id": "slash-reconcile:2:1",
        "effect_type": "slash_reconciliation",
    }
    try:
        writer.commit(
            (JournalEvent("hire.effect.prepared", "hire_recover", exact_payload),),
            writer.get_aggregate_versions(("hire_recover",)),
        )
        HireProjection.rebuild(writer.replay())

        writer.commit(
            (
                JournalEvent(
                    "hire.effect.prepared",
                    "hire_recover",
                    {
                        **exact_payload,
                        "effect_type": "channel_start",
                    },
                ),
            ),
            writer.get_aggregate_versions(("hire_recover",)),
        )
        with pytest.raises(
            HireProjectionError,
            match="conflicting duplicate hire effect transition",
        ):
            HireProjection.rebuild(writer.replay())
    finally:
        writer.close()


def test_reconfigure_disposes_superseded_effect_before_automatic_activation(
    tmp_path: Path,
    run_asyncio_threads_inline: None,
) -> None:
    _seed_active_employee(tmp_path)
    service = _service(_writer(tmp_path, 2))
    service.commit_effect_transition(
        "hire_recover",
        effect_id="slash-reconcile:1:2",
        effect_type="slash_reconciliation",
        next_state=HireEffectState.PREPARED,
    )

    class Slash:
        async def reconcile(self):
            return SimpleNamespace(
                spec_hash="slash_hash",
                observed_hash="slash_hash",
                observed=(),
            )

    class Channels:
        def start(
            self,
            _agent_id,
            _app_id,
            _credential_ref,
            generation,
            _on_event,
        ):
            return SimpleNamespace(
                state=ChannelProcessState.READY,
                identity={"app_id": "cli_employee"},
                ready_metadata={"connection_id": f"conn_generation_{generation}"},
                error_code="",
            )

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._vault = SimpleNamespace(resolve=lambda *_args: "secret")  # noqa: SLF001
    runtime._slash_factory = lambda *_args: Slash()  # noqa: SLF001
    runtime._channels = Channels()  # type: ignore[assignment]  # noqa: SLF001
    try:
        before = service.get_state("hire_recover")
        assert before is not None
        assert before.automatic_reactivation_generation == 1

        asyncio.run(runtime._configure_intent("hire_recover"))  # noqa: SLF001

        state = service.get_state("hire_recover")
        assert state is not None
        assert state.phase is HirePhase.ACTIVE
        assert state.channel_generation == 2
        assert state.effect_state("slash-reconcile:1:2") is HireEffectState.ACTION_REQUIRED
        assert dict(state.metadata_for("slash-reconcile:1:2")) == {
            "error_code": "superseded_reconfiguration"
        }
        assert state.automatic_reactivation_generation == 0
    finally:
        service.close()


def _seed_action_required(
    tmp_path: Path,
    *,
    effect_type: str = "slash_reconciliation",
    effect_id: str = "slash-reconcile:2:1",
    terminal: bool = True,
    error_code: str = "recovery_exhausted",
) -> ProductionEmployeeHireService:
    _seed_active_employee(tmp_path)
    epoch_two = _service(_writer(tmp_path, 2))
    epoch_two.commit_effect_transition(
        "hire_recover",
        effect_id=effect_id,
        effect_type=effect_type,
        next_state=HireEffectState.PREPARED,
    )
    epoch_two.commit_effect_transition(
        "hire_recover",
        effect_id=effect_id,
        effect_type=effect_type,
        next_state=HireEffectState.EXECUTING,
    )
    if terminal:
        epoch_two.mark_recovery_action_required(
            "hire_recover",
            error_code=error_code,
        )
    else:
        commit_workforce_events(
            epoch_two._writer,  # noqa: SLF001
            epoch_two.projection_state,
            (
                JournalEvent(
                    event_type="employee.state_changed",
                    aggregate_id="agt_recover",
                    payload={"state": "action_required"},
                ),
            ),
        )
    epoch_two.close()
    return _service(_writer(tmp_path, 3))


def test_recovery_exhausted_slash_is_atomically_reopened_and_uses_fresh_attempt(
    tmp_path: Path,
) -> None:
    service = _seed_action_required(tmp_path)

    result = service.recover_replay_safe_action_required()

    assert result == hire_service.ActionRequiredRecoveryResult(
        eligible=1,
        repaired_intent_ids=("hire_recover",),
        skipped=0,
        failed=0,
    )
    state = service.get_state("hire_recover")
    assert state is not None
    assert state.phase is HirePhase.VALIDATING
    assert state.channel_generation == 1
    assert state.effect_state("slash-reconcile:2:1") is HireEffectState.ACTION_REQUIRED
    assert service.select_slash_reconcile_effect(
        state.intent_id,
        generation=2,
        force_refresh=True,
        allow_action_required_refresh=True,
    ) == "slash-reconcile:2:2"
    marker_frames = [
        frame
        for frame in service._writer.replay()  # noqa: SLF001
        if any(
            event.event_type == "hire.channel.phase_only_recovery"
            for event in frame.events
        )
    ]
    assert len(marker_frames) == 1
    assert [event.event_type for event in marker_frames[0].events] == [
        "hire.channel.phase_only_recovery",
        "employee.state_changed",
    ]
    assert marker_frames[0].events[0].payload == {"generation": 1}
    assert marker_frames[0].events[1].payload == {"state": "validating"}


def test_terminal_anchor_failed_slash_is_reopened_as_convergent_reconciliation(
    tmp_path: Path,
) -> None:
    service = _seed_action_required(
        tmp_path,
        error_code="terminal_anchor_failed",
    )

    result = service.recover_replay_safe_action_required()

    assert result == hire_service.ActionRequiredRecoveryResult(
        eligible=1,
        repaired_intent_ids=("hire_recover",),
        skipped=0,
        failed=0,
    )
    state = service.get_state("hire_recover")
    assert state is not None
    assert state.phase is HirePhase.VALIDATING
    assert service.select_slash_reconcile_effect(
        state.intent_id,
        generation=2,
        force_refresh=True,
        allow_action_required_refresh=True,
    ) == "slash-reconcile:2:2"


def test_same_epoch_crash_and_exhaustion_requires_restart_then_recovers(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path, begin_revalidation=False)
    writer_two = _writer(tmp_path, 2)
    assert writer_two.writer_epoch == 2
    same_epoch = _service(writer_two)
    same_epoch.begin_channel_revalidation("hire_recover", observed_generation=1)
    for next_state in (HireEffectState.PREPARED, HireEffectState.EXECUTING):
        same_epoch.commit_effect_transition(
            "hire_recover",
            effect_id="slash-reconcile:2:1",
            effect_type="slash_reconciliation",
            next_state=next_state,
        )
    same_epoch.mark_recovery_action_required(
        "hire_recover",
        error_code="recovery_exhausted",
    )

    assert same_epoch.recover_replay_safe_action_required() == (
        hire_service.ActionRequiredRecoveryResult(0, (), 1, 0)
    )
    same_epoch.close()

    writer_three = _writer(tmp_path, 3)
    assert writer_three.writer_epoch == 3
    restarted = _service(writer_three)

    assert restarted.recover_replay_safe_action_required() == (
        hire_service.ActionRequiredRecoveryResult(1, ("hire_recover",), 0, 0)
    )


def test_repaired_intent_reaches_active_with_fresh_effects_and_one_worker(
    tmp_path: Path,
    run_asyncio_threads_inline: None,
) -> None:
    service = _seed_action_required(tmp_path)
    repaired = service.recover_replay_safe_action_required()
    starts: list[int] = []

    class Slash:
        async def reconcile(self):
            return SimpleNamespace(
                spec_hash="slash_hash",
                observed_hash="slash_hash",
                observed=(),
            )

    class Channels:
        def start(
            self,
            _agent_id,
            _app_id,
            _credential_ref,
            generation,
            _on_event,
        ):
            starts.append(generation)
            return SimpleNamespace(
                state=ChannelProcessState.READY,
                identity={"app_id": "cli_employee"},
                ready_metadata={"connection_id": "conn_generation_2"},
                error_code="",
            )

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._vault = SimpleNamespace(resolve=lambda *_args: "secret")  # noqa: SLF001
    runtime._slash_factory = lambda *_args: Slash()  # noqa: SLF001
    runtime._channels = Channels()  # type: ignore[assignment]  # noqa: SLF001

    asyncio.run(
        runtime._configure_intent(  # noqa: SLF001
            repaired.repaired_intent_ids[0],
            force_slash_refresh=True,
            allow_action_required_refresh=True,
        )
    )

    state = service.get_state("hire_recover")
    assert state is not None
    assert state.phase is HirePhase.ACTIVE
    assert starts == [2]
    assert state.effect_state("slash-reconcile:2:1") is HireEffectState.ACTION_REQUIRED
    assert state.effect_state("slash-reconcile:2:2") is HireEffectState.COMMITTED
    assert state.effect_state("channel-start:2") is HireEffectState.COMMITTED
    assert not any(
        event.event_type == "hire.verification.challenge_issued"
        for frame in service._writer.replay()  # noqa: SLF001
        for event in frame.events
    )


def test_projection_consumes_phase_only_automatic_activation_proof_once(
    tmp_path: Path,
    run_asyncio_threads_inline: None,
) -> None:
    from src.autonomous.provisioning.hire_state import HireProjectionError

    service = _seed_action_required(tmp_path)
    try:
        repaired = service.recover_replay_safe_action_required()

        class Slash:
            async def reconcile(self):
                return SimpleNamespace(
                    spec_hash="slash_hash",
                    observed_hash="slash_hash",
                    observed=(),
                )

        class Channels:
            def start(
                self,
                _agent_id,
                _app_id,
                _credential_ref,
                generation,
                _on_event,
            ):
                return SimpleNamespace(
                    state=ChannelProcessState.READY,
                    identity={"app_id": "cli_employee"},
                    ready_metadata={"connection_id": f"conn_generation_{generation}"},
                    error_code="",
                )

        runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
        runtime._service = service  # noqa: SLF001
        runtime._vault = SimpleNamespace(resolve=lambda *_args: "secret")  # noqa: SLF001
        runtime._slash_factory = lambda *_args: Slash()  # noqa: SLF001
        runtime._channels = Channels()  # type: ignore[assignment]  # noqa: SLF001
        asyncio.run(
            runtime._configure_intent(  # noqa: SLF001
                repaired.repaired_intent_ids[0],
                force_slash_refresh=True,
                allow_action_required_refresh=True,
            )
        )

        state = service.get_state("hire_recover")
        assert state is not None
        assert state.phase is HirePhase.ACTIVE
        assert state.automatic_reactivation_generation == 0
        state = service._commit_phase_transition(  # noqa: SLF001
            state,
            HirePhase.VALIDATING,
        )
        state = service._commit_phase_transition(  # noqa: SLF001
            state,
            HirePhase.READY_PENDING_VERIFICATION,
        )

        class Frame:
            def __init__(self, sequence, events):
                self.sequence = sequence
                self.events = tuple(events)

        frames = list(service._writer.replay())  # noqa: SLF001
        duplicate = JournalEvent(
            "hire.activation.automatic",
            state.intent_id,
            {
                "tenant_key": state.tenant_key,
                "app_id": state.app_id,
                "agent_id": state.agent_id,
                "generation": state.channel_generation,
                "slash_spec_hash": state.slash_spec_hash,
                "channel_connection_id": state.channel_connection_id,
                "requester_principal_id": state.requester_principal_id,
                "requester_union_id": state.requester_union_id,
                "source": "channel_ready",
                "activated_at": state.channel_verified_at + 1.0,
            },
        )
        with pytest.raises(HireProjectionError, match="invalid automatic activation"):
            HireProjection.rebuild(  # type: ignore[arg-type]
                (*frames, Frame(frames[-1].sequence + 1, (duplicate,)))
            )
    finally:
        service.close()


def test_later_generation_exhaustion_ignores_disposed_prior_generation(
    tmp_path: Path,
    run_asyncio_threads_inline: None,
) -> None:
    service = _seed_action_required(tmp_path)
    repaired = service.recover_replay_safe_action_required()

    class Slash:
        async def reconcile(self):
            return SimpleNamespace(
                spec_hash="slash_hash",
                observed_hash="slash_hash",
                observed=(),
            )

    class Channels:
        def start(
            self,
            _agent_id,
            _app_id,
            _credential_ref,
            generation,
            _on_event,
        ):
            return SimpleNamespace(
                state=ChannelProcessState.READY,
                identity={"app_id": "cli_employee"},
                ready_metadata={"connection_id": f"conn_generation_{generation}"},
                error_code="",
            )

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._vault = SimpleNamespace(resolve=lambda *_args: "secret")  # noqa: SLF001
    runtime._slash_factory = lambda *_args: Slash()  # noqa: SLF001
    runtime._channels = Channels()  # type: ignore[assignment]  # noqa: SLF001
    asyncio.run(
        runtime._configure_intent(  # noqa: SLF001
            repaired.repaired_intent_ids[0],
            force_slash_refresh=True,
            allow_action_required_refresh=True,
        )
    )
    service.begin_channel_revalidation("hire_recover", observed_generation=2)
    service.close()

    epoch_four = _service(_writer(tmp_path, 4))
    for next_state in (HireEffectState.PREPARED, HireEffectState.EXECUTING):
        epoch_four.commit_effect_transition(
            "hire_recover",
            effect_id="slash-reconcile:3:1",
            effect_type="slash_reconciliation",
            next_state=next_state,
        )
    epoch_four.mark_recovery_action_required(
        "hire_recover",
        error_code="recovery_exhausted",
    )
    epoch_four.close()

    reopened = _service(_writer(tmp_path, 5))
    result = reopened.recover_replay_safe_action_required()

    assert result == hire_service.ActionRequiredRecoveryResult(1, ("hire_recover",), 0, 0)
    state = reopened.get_state("hire_recover")
    assert state is not None
    assert state.phase is HirePhase.VALIDATING
    assert state.effect_state("slash-reconcile:2:1") is HireEffectState.ACTION_REQUIRED
    assert state.effect_state("slash-reconcile:3:1") is HireEffectState.ACTION_REQUIRED
    assert reopened.select_slash_reconcile_effect(
        state.intent_id,
        generation=3,
        force_refresh=True,
        allow_action_required_refresh=True,
    ) == "slash-reconcile:3:2"


@pytest.mark.parametrize(
    ("effect_type", "effect_id", "terminal"),
    [
        ("app_registration", "register-app-retry", True),
        ("credential_vault_put", "store-credential-retry", True),
        ("employee_channel_start", "channel-start:2", True),
        ("slash_reconciliation", "slash-reconcile:2:1", False),
    ],
)
def test_unsafe_action_required_evidence_is_skipped(
    tmp_path: Path,
    effect_type: str,
    effect_id: str,
    terminal: bool,
) -> None:
    service = _seed_action_required(
        tmp_path,
        effect_type=effect_type,
        effect_id=effect_id,
        terminal=terminal,
    )

    result = service.recover_replay_safe_action_required()

    assert result == hire_service.ActionRequiredRecoveryResult(
        eligible=0,
        repaired_intent_ids=(),
        skipped=1,
        failed=0,
    )
    assert service.get_state("hire_recover").phase is HirePhase.ACTION_REQUIRED  # type: ignore[union-attr]


def test_recovery_anchor_failure_never_publishes_repaired_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _seed_action_required(tmp_path)
    monkeypatch.setattr(
        service._writer.anchor,  # noqa: SLF001
        "compare_and_swap",
        lambda *_args: False,
    )

    result = service.recover_replay_safe_action_required()

    assert result == hire_service.ActionRequiredRecoveryResult(
        eligible=1,
        repaired_intent_ids=(),
        skipped=0,
        failed=1,
    )
    assert service.get_state("hire_recover").phase is HirePhase.ACTION_REQUIRED  # type: ignore[union-attr]


def test_runtime_recover_is_single_flight_and_waits_for_actual_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(
        intent_id="hire_recover",
        agent_id="agt_recover",
        tenant_key="tenant-a",
        phase=HirePhase.VALIDATING,
        credential_ref="vault://employee",
        channel_generation=1,
    )
    calls = {"repair": 0, "configure": 0, "opened": 0}
    started = threading.Event()
    release = threading.Event()

    class Service:
        projection_state = SimpleNamespace(employees={})

        def recover(self):
            return SimpleNamespace(states={state.intent_id: state})

        def recover_replay_safe_action_required(self):
            calls["repair"] += 1
            return hire_service.ActionRequiredRecoveryResult(1, (state.intent_id,), 2, 0)

        def list_states(self):
            return (state,)

        def get_state(self, _intent_id):
            return state

        def mark_runtime_recovered(self):
            calls["opened"] += 1

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._execution_blockers = ("test-isolation",)  # noqa: SLF001
    monkeypatch.setattr(runtime, "_refresh_context_bindings", lambda _projection: True)
    monkeypatch.setattr(runtime, "_start_monitor_in_loop", lambda: None)
    runtime._start_loop()  # noqa: SLF001

    async def configure(intent_id, **kwargs):
        assert intent_id == state.intent_id
        assert kwargs["force_slash_refresh"] is True
        assert kwargs["allow_action_required_refresh"] is True
        calls["configure"] += 1
        started.set()
        await __import__("asyncio").to_thread(release.wait)
        state.phase = HirePhase.ACTIVE

    monkeypatch.setattr(runtime, "_configure_intent", configure)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(runtime.recover)
            assert started.wait(2.0)
            second = pool.submit(runtime.recover)
            assert not first.done()
            assert not second.done()
            release.set()
            expected = composition.EmployeeRecoverySummary(1, 1, 2, 0)
            assert first.result(timeout=2.0) == expected
            assert second.result(timeout=2.0) == expected

        assert calls == {"repair": 1, "configure": 1, "opened": 1}
        assert runtime.recover() == composition.EmployeeRecoverySummary(1, 1, 2, 0)
        assert calls == {"repair": 1, "configure": 1, "opened": 1}
    finally:
        runtime._closing = True  # noqa: SLF001
        assert runtime._loop is not None  # noqa: SLF001
        runtime._loop.call_soon_threadsafe(runtime._loop.stop)  # noqa: SLF001
        assert runtime._loop_thread is not None  # noqa: SLF001
        runtime._loop_thread.join(timeout=2.0)  # noqa: SLF001


def test_runtime_batch_repair_failure_prevents_all_external_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(
        intent_id="hire_repaired_before_batch_failure",
        agent_id="agt_repaired_before_batch_failure",
        tenant_key="tenant-a",
        phase=HirePhase.VALIDATING,
        credential_ref="vault://employee",
        channel_generation=1,
    )
    calls = {"configure": 0, "opened": 0}

    class Service:
        projection_state = SimpleNamespace(employees={})

        def recover(self):
            return SimpleNamespace(states={state.intent_id: state})

        def recover_replay_safe_action_required(self):
            return hire_service.ActionRequiredRecoveryResult(
                eligible=2,
                repaired_intent_ids=(state.intent_id,),
                skipped=0,
                failed=1,
            )

        def list_states(self):
            return (state,)

        def get_state(self, _intent_id):
            return state

        def mark_runtime_recovered(self):
            calls["opened"] += 1

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._execution_blockers = ("test-isolation",)  # noqa: SLF001
    monkeypatch.setattr(runtime, "_refresh_context_bindings", lambda _projection: True)
    monkeypatch.setattr(runtime, "_start_monitor_in_loop", lambda: None)
    runtime._start_loop()  # noqa: SLF001

    async def configure(*_args, **_kwargs):
        calls["configure"] += 1
        state.phase = HirePhase.ACTIVE

    monkeypatch.setattr(runtime, "_configure_intent", configure)
    try:
        with pytest.raises(RuntimeError, match="ACTION_REQUIRED repair failed"):
            runtime.recover()
        assert calls == {"configure": 0, "opened": 0}
    finally:
        runtime._closing = True  # noqa: SLF001
        assert runtime._loop is not None  # noqa: SLF001
        runtime._loop.call_soon_threadsafe(runtime._loop.stop)  # noqa: SLF001
        assert runtime._loop_thread is not None  # noqa: SLF001
        runtime._loop_thread.join(timeout=2.0)  # noqa: SLF001


def test_runtime_recover_allows_one_retry_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()
    attempts = 0
    expected = composition.EmployeeRecoverySummary(skipped=1)

    def recover_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient replay failure")
        return expected

    monkeypatch.setattr(runtime, "_recover_once", recover_once)

    with pytest.raises(RuntimeError, match="transient replay failure"):
        runtime.recover()
    assert runtime.recover() == expected
    assert runtime.recover() == expected
    assert attempts == 2

    exhausted = EmployeeDepartmentRuntime()
    exhausted_attempts = 0

    def always_fail():
        nonlocal exhausted_attempts
        exhausted_attempts += 1
        raise RuntimeError("persistent replay failure")

    monkeypatch.setattr(exhausted, "_recover_once", always_fail)
    for _call in range(3):
        with pytest.raises(RuntimeError, match="persistent replay failure"):
            exhausted.recover()
    assert exhausted_attempts == 2


def _runtime_with_startup_membership_audit(
    monkeypatch: pytest.MonkeyPatch,
    audit,
) -> tuple[EmployeeDepartmentRuntime, list[str]]:
    events: list[str] = []

    class Service:
        projection_state = SimpleNamespace(employees={})

        def recover(self):
            events.append("hire_replay")

        def recover_replay_safe_action_required(self):
            return hire_service.ActionRequiredRecoveryResult(0, (), 0, 0)

        def list_states(self):
            return ()

        def mark_runtime_recovered(self):
            events.append("admission_open")

    class Membership:
        def rebuild_projection(self):
            events.append("membership_replay")

        def recover_pending(self):
            events.append("membership_pending")

        def reconcile_projected_memberships(self):
            events.append("membership_audit_enter")
            result = audit()
            events.append("membership_audit_done")
            return result

    def noop() -> None:
        return None

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._membership = Membership()  # type: ignore[assignment]  # noqa: SLF001
    runtime._ingress = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        rebuild_projection=noop,
        gc_terminal_payloads=noop,
    )
    runtime._router = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        rebuild_projection=noop,
        recover_terminal_attachments=noop,
    )
    runtime._dispatch = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        employee_runtime=None,
    )
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        rebuild_projection=noop,
    )
    monkeypatch.setattr(runtime, "_repair_employee_dispatch_reporting", noop)
    monkeypatch.setattr(runtime, "_reconcile_retired_activation_ingress", noop)
    monkeypatch.setattr(runtime, "_reconcile_terminal_ingress", noop)
    monkeypatch.setattr(runtime, "_refresh_context_bindings", lambda _projection: True)
    monkeypatch.setattr(runtime, "_start_monitor_in_loop", noop)
    monkeypatch.setattr(runtime, "_start_reporting_worker", noop)
    monkeypatch.setattr(
        runtime,
        "_start_dispatch_worker",
        lambda: events.append("dispatch_start"),
    )
    return runtime, events


def test_runtime_membership_audit_precedes_admission_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_entered = threading.Event()
    release_audit = threading.Event()

    def audit():
        audit_entered.set()
        assert release_audit.wait(2.0)
        return SimpleNamespace(removed=0, degraded=0)

    runtime, events = _runtime_with_startup_membership_audit(monkeypatch, audit)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        recovery = pool.submit(runtime.recover)
        assert audit_entered.wait(1.0)
        assert "admission_open" not in events
        assert "dispatch_start" not in events
        assert recovery.done() is False
        release_audit.set()
        recovery.result(timeout=2.0)

    assert events.index("membership_audit_done") < events.index("admission_open")
    assert events.index("membership_audit_done") < events.index("dispatch_start")
    assert runtime._core_recovered is True  # noqa: SLF001


def test_runtime_membership_audit_failure_keeps_execution_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_audit():
        raise RuntimeError("remote membership audit unavailable")

    runtime, events = _runtime_with_startup_membership_audit(
        monkeypatch,
        fail_audit,
    )

    runtime.recover()

    assert "admission_open" not in events
    assert "dispatch_start" not in events
    assert runtime._execution_blockers == ("membership_audit",)  # noqa: SLF001
    assert runtime._core_recovered is False  # noqa: SLF001


def test_background_recovery_entry_retries_once_then_completes() -> None:
    from src.feishu.ws_client import FeishuWSClient

    attempts = 0
    failed_closed: list[str] = []
    published: list[bool] = []
    expected = composition.EmployeeRecoverySummary(1, 1, 0, 0)

    class Runtime:
        membership_service = None

        def recover(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient startup replay failure")
            return expected

        def fail_recovery(self, reason):
            failed_closed.append(reason)

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._employee_department_runtime = Runtime()
    client._handler_ctx = SimpleNamespace(handlers={"coco": object()})
    client._employee_runtime_recovery_error = None
    client._try_publish_restart_readiness = lambda: published.append(True)

    client._run_employee_runtime_recovery()

    assert attempts == 2
    assert failed_closed == []
    assert published == [True]
    assert client._employee_runtime_recovery_error is None


def test_background_recovery_retries_failed_repair_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.feishu.ws_client import FeishuWSClient

    calls = {"recover": 0, "repair": 0, "opened": 0}
    published: list[bool] = []

    class Service:
        projection_state = ProjectionState()

        def recover(self):
            calls["recover"] += 1

        def recover_replay_safe_action_required(self):
            calls["repair"] += 1
            if calls["repair"] == 1:
                return hire_service.ActionRequiredRecoveryResult(1, (), 0, 1)
            return hire_service.ActionRequiredRecoveryResult(0, (), 0, 0)

        def list_states(self):
            return ()

        def mark_runtime_recovered(self):
            calls["opened"] += 1

    def noop() -> None:
        return None

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._ingress = SimpleNamespace(  # noqa: SLF001
        rebuild_projection=noop,
        gc_terminal_payloads=noop,
    )
    runtime._router = SimpleNamespace(  # noqa: SLF001
        rebuild_projection=noop,
        recover_terminal_attachments=noop,
    )
    runtime._dispatch = SimpleNamespace(  # noqa: SLF001
        employee_runtime=None,
        recover_incomplete_attempts=noop,
        reconcile_terminal_snapshots=noop,
    )
    runtime._outbox = SimpleNamespace(rebuild_projection=noop)  # noqa: SLF001
    monkeypatch.setattr(runtime, "_reconcile_retired_activation_ingress", noop)
    monkeypatch.setattr(runtime, "_reconcile_terminal_ingress", noop)
    monkeypatch.setattr(runtime, "_refresh_context_bindings", lambda _projection: True)
    monkeypatch.setattr(runtime, "_start_dispatch_worker", noop)

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._employee_department_runtime = runtime
    client._handler_ctx = SimpleNamespace(handlers={"coco": object()})
    client._employee_runtime_recovery_error = None
    client._try_publish_restart_readiness = lambda: published.append(True)

    client._run_employee_runtime_recovery()

    assert calls == {"recover": 2, "repair": 2, "opened": 1}
    assert runtime._recovery_attempts == 2  # noqa: SLF001
    assert runtime._execution_blockers == ()  # noqa: SLF001
    assert runtime._core_recovered is True  # noqa: SLF001
    assert client._employee_runtime_recovery_error is None
    assert published == [True]


def test_background_recovery_entry_stops_after_second_failure() -> None:
    from src.feishu.ws_client import FeishuWSClient

    attempts = 0
    failed_closed: list[str] = []

    class Runtime:
        membership_service = None

        def recover(self):
            nonlocal attempts
            attempts += 1
            raise RuntimeError(f"persistent startup failure {attempts}")

        def fail_recovery(self, reason):
            failed_closed.append(reason)

    client = FeishuWSClient.__new__(FeishuWSClient)
    client._employee_department_runtime = Runtime()
    client._handler_ctx = SimpleNamespace(handlers={"coco": object()})
    client._employee_runtime_recovery_error = None
    client._try_publish_restart_readiness = lambda: pytest.fail(
        "failed recovery cannot publish readiness"
    )

    client._run_employee_runtime_recovery()

    assert attempts == 2
    assert failed_closed == ["background_recovery"]
    assert isinstance(client._employee_runtime_recovery_error, RuntimeError)


def test_reducer_accepts_one_atomic_frame_for_multiple_recovery_pairs() -> None:
    class Frame:
        def __init__(self, sequence, events, *, epoch=1):
            self.sequence = sequence
            self.events = tuple(events)
            self.writer_epoch = epoch

    frames = []
    sequence = 0
    for suffix in ("one", "two"):
        agent_id = f"agt_{suffix}"
        intent_id = f"hire_{suffix}"
        bot_id = f"bot_{suffix}"
        created = JournalEvent(
            event_type="employee.created",
            aggregate_id=agent_id,
            payload={
                "agent_id": agent_id,
                "tenant_key": "tenant-a",
                "owner_principal_id": "ou_admin",
                "name": suffix,
                "tool": "codex",
                "model": "gpt-5.6-sol",
                "effort": "high",
                "profile": "standard",
                "worker_type": "visible",
                "state": "provisioning_app",
                "hire_schema_version": 1,
                "hire_intent_id": intent_id,
                "hire_message_id": f"om_{suffix}",
                "hire_chat_id": "oc_admin_dm",
                "planned_bot_principal_id": bot_id,
                "provisioning_attempt_id": f"attempt_{suffix}",
            },
        )
        effect_payload = {
            "effect_id": "channel-start:1",
            "effect_type": "employee_channel_start",
        }
        committed_payload = {
            **effect_payload,
            "metadata": {
                "app_id": f"cli_{suffix}",
                "generation": "1",
                "identity_app_id": f"cli_{suffix}",
                "connection_id": f"conn_{suffix}",
                "channel_verified_at": "1.0",
            },
        }
        for event in (
            created,
            JournalEvent("hire.effect.prepared", intent_id, effect_payload),
            JournalEvent("hire.effect.executing", intent_id, effect_payload),
            JournalEvent("hire.effect.committed", intent_id, committed_payload),
            JournalEvent(
                "employee.state_changed",
                agent_id,
                {"state": "action_required"},
            ),
        ):
            sequence += 1
            frames.append(Frame(sequence, (event,)))
    sequence += 1
    frames.append(
        Frame(
            sequence,
            (
                JournalEvent(
                    "hire.channel.phase_only_recovery",
                    "hire_one",
                    {"generation": 1},
                ),
                JournalEvent(
                    "employee.state_changed",
                    "agt_one",
                    {"state": "validating"},
                ),
                JournalEvent(
                    "hire.channel.phase_only_recovery",
                    "hire_two",
                    {"generation": 1},
                ),
                JournalEvent(
                    "employee.state_changed",
                    "agt_two",
                    {"state": "validating"},
                ),
            ),
            epoch=2,
        )
    )

    projection = HireProjection.rebuild(frames)  # type: ignore[arg-type]

    assert projection.get("hire_one").phase is HirePhase.VALIDATING  # type: ignore[union-attr]
    assert projection.get("hire_two").phase is HirePhase.VALIDATING  # type: ignore[union-attr]


def test_batch_recovery_uses_one_journal_commit_on_anchor_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(_writer(tmp_path, 3))
    states = {
        suffix: DurableHireState(
            intent_id=f"hire_{suffix}",
            agent_id=f"agt_{suffix}",
            phase=HirePhase.ACTION_REQUIRED,
            channel_generation=1,
        )
        for suffix in ("one", "two")
    }
    service._hire_projection = HireProjection(  # noqa: SLF001
        states={state.intent_id: state for state in states.values()}
    )
    service.projection_state.employees.update(
        {
            state.agent_id: SimpleNamespace(
                worker_type=WorkerType.VISIBLE,
                state=EmployeeState.ACTION_REQUIRED,
            )
            for state in states.values()
        }
    )
    monkeypatch.setattr(
        service,
        "_has_exact_replay_safe_recovery_exhausted_history",
        lambda _state, _frames, *, recovery_writer_epoch: (
            recovery_writer_epoch == service._writer.writer_epoch  # noqa: SLF001
        ),
    )
    monkeypatch.setattr(hire_service, "validate_workforce_events", lambda *_args: None)
    commits: list[tuple[JournalEvent, ...]] = []

    def fail_anchor(events, *_args, **_kwargs):
        commits.append(tuple(events))
        return SimpleNamespace(state=hire_service.CommitState.DURABLE_NOT_ANCHORED)

    monkeypatch.setattr(service._writer, "commit", fail_anchor)  # noqa: SLF001

    result = service.recover_replay_safe_action_required()

    assert result == hire_service.ActionRequiredRecoveryResult(2, (), 0, 2)
    assert len(commits) == 1
    assert [event.event_type for event in commits[0]] == [
        "hire.channel.phase_only_recovery",
        "employee.state_changed",
        "hire.channel.phase_only_recovery",
        "employee.state_changed",
    ]


def test_runtime_summary_preserves_active_intent_when_peer_isolation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = {
        intent_id: SimpleNamespace(
            intent_id=intent_id,
            agent_id=f"agt_{intent_id}",
            tenant_key="tenant-a",
            phase=HirePhase.VALIDATING,
            credential_ref="vault://employee",
            channel_generation=1,
        )
        for intent_id in ("hire_active", "hire_failed")
    }

    class Service:
        projection_state = SimpleNamespace(employees={})

        def recover(self):
            return SimpleNamespace(states=states)

        def recover_replay_safe_action_required(self):
            return hire_service.ActionRequiredRecoveryResult(
                2,
                tuple(states),
                0,
                0,
            )

        def list_states(self):
            return tuple(states.values())

        def get_state(self, intent_id):
            return states[intent_id]

        def mark_runtime_recovered(self):
            pytest.fail("isolation failure must keep admission closed")

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._execution_blockers = ("test-isolation",)  # noqa: SLF001
    monkeypatch.setattr(runtime, "_refresh_context_bindings", lambda _projection: True)
    monitor_starts: list[bool] = []
    monkeypatch.setattr(
        runtime,
        "_start_monitor_in_loop",
        lambda: monitor_starts.append(True),
    )
    runtime._start_loop()  # noqa: SLF001

    async def configure(intent_id, **_kwargs):
        if intent_id == "hire_active":
            states[intent_id].phase = HirePhase.ACTIVE
            return
        raise RuntimeError("configuration failed")

    async def fail_isolation(_intent_id, **_kwargs):
        raise RuntimeError("durable isolation failed")

    monkeypatch.setattr(runtime, "_configure_intent", configure)
    monkeypatch.setattr(runtime, "_retry_recovery_intent", fail_isolation)
    try:
        assert runtime.recover() == composition.EmployeeRecoverySummary(2, 1, 0, 1)
        assert monitor_starts == []
    finally:
        runtime._closing = True  # noqa: SLF001
        assert runtime._loop is not None  # noqa: SLF001
        runtime._loop.call_soon_threadsafe(runtime._loop.stop)  # noqa: SLF001
        assert runtime._loop_thread is not None  # noqa: SLF001
        runtime._loop_thread.join(timeout=2.0)  # noqa: SLF001


def test_channel_acceptance_routes_second_task_while_first_execution_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admission lane must not share the long execution worker."""

    first_id = "acc_first"
    second_id = "acc_second"
    first = SimpleNamespace(disposition=None)
    second = SimpleNamespace(disposition=None)
    ingress_records = {first_id: first}
    router_records = {
        first_id: SimpleNamespace(state="queued"),
    }
    execution_entered = threading.Event()
    release_execution = threading.Event()

    class Ingress:
        state = SimpleNamespace(by_acceptance_id=ingress_records)

        def rebuild_projection(self):
            return self.state

        def record_snapshot(self, acceptance_id):
            return ingress_records.get(acceptance_id)

        def get_payload(self, acceptance_id):
            assert acceptance_id == second_id
            return SimpleNamespace(
                normalized_parts=({"type": "message"},),
            )

        def gc_terminal_payloads(self):
            return 0

    class Router:
        state = SimpleNamespace(by_acceptance_id=router_records)

        def rebuild_projection(self):
            return self.state

        def record_snapshot(self, acceptance_id):
            return router_records.get(acceptance_id)

        def is_inbox_candidate_eligible(self, acceptance_id):
            return acceptance_id == second_id

        def route(self, acceptance_id):
            assert acceptance_id == second_id
            routed = SimpleNamespace(state="queued")
            router_records[acceptance_id] = routed
            return routed

    class Dispatch:
        employee_runtime = None

        def prepare_next(self):
            return SimpleNamespace(acceptance_id=first_id)

        def execute_prepared(self, prepared):
            assert prepared.acceptance_id == first_id
            execution_entered.set()
            assert release_execution.wait(5.0)

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = Router()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = Dispatch()  # type: ignore[assignment]  # noqa: SLF001
    monkeypatch.setattr(runtime, "_owner_p2p_requester", lambda *_args: "ou_owner")
    monkeypatch.setattr(runtime, "_authorized_targeted_group_task", lambda *_args: None)
    monkeypatch.setattr(
        runtime,
        "_managed_employee_ingress_trust",
        lambda *_args: runtime._unknown_employee_ingress_trust(),  # noqa: SLF001
    )
    monkeypatch.setattr(runtime, "_handle_control_ingress", lambda *_args, **_kwargs: False)

    execution = threading.Thread(
        target=runtime._drain_employee_dispatch_once,  # noqa: SLF001
        daemon=True,
    )
    execution.start()
    assert execution_entered.wait(5.0)
    ingress_records[second_id] = second

    try:
        asyncio.run(
            runtime._handle_channel_event(  # noqa: SLF001
                "hire_alpha",
                3,
                {
                    "event": "durableIngressAccepted",
                    "data": {"acceptance_id": second_id},
                },
            )
        )
        assert router_records[second_id].state == "queued"
        assert execution.is_alive()
    finally:
        release_execution.set()
        execution.join(timeout=5.0)
    assert not execution.is_alive()


def test_channel_control_disposition_notifies_handoff_waiters_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance_id = "acc_status"
    record = SimpleNamespace(disposition=None)

    class Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: record})

        def record_snapshot(self, observed):
            assert observed == acceptance_id
            return record

        def get_payload(self, observed):
            assert observed == acceptance_id
            return SimpleNamespace(
                normalized_parts=({"type": "message"},),
            )

    class Router:
        state = SimpleNamespace(by_acceptance_id={})

        def record_snapshot(self, observed):
            assert observed == acceptance_id
            return None

        def is_inbox_candidate_eligible(self, observed):
            assert observed == acceptance_id
            return True

        def route(self, _acceptance_id):
            pytest.fail("a consumed control must not enter the Router queue")

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = Router()  # type: ignore[assignment]  # noqa: SLF001

    def consume_control(observed, **_kwargs):
        assert observed == acceptance_id
        record.disposition = SimpleNamespace(reason_code="status_completed")
        return True

    monkeypatch.setattr(runtime, "_handle_control_ingress", consume_control)
    revision = runtime._employee_handoff_revision  # noqa: SLF001

    asyncio.run(
        runtime._handle_channel_event(  # noqa: SLF001
            "hire_alpha",
            3,
            {
                "event": "durableIngressAccepted",
                "data": {"acceptance_id": acceptance_id},
            },
        )
    )

    assert record.disposition.reason_code == "status_completed"
    assert runtime._employee_handoff_revision == revision + 1  # noqa: SLF001


def test_channel_acceptance_notifies_even_when_fast_admission_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()

    def fail_admission(_acceptance_id: str) -> bool:
        raise RuntimeError("transient admission failure")

    monkeypatch.setattr(runtime, "_admit_employee_ingress_once", fail_admission)
    revision = runtime._employee_handoff_revision  # noqa: SLF001

    with pytest.raises(RuntimeError, match="transient admission failure"):
        asyncio.run(
            runtime._handle_channel_event(  # noqa: SLF001
                "hire_alpha",
                3,
                {
                    "event": "durableIngressAccepted",
                    "data": {"acceptance_id": "acc_retry"},
                },
            )
        )

    assert runtime._employee_handoff_revision == revision + 1  # noqa: SLF001


def test_team_projection_failure_after_acceptance_is_left_for_periodic_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance_id = "acc_team"
    agent_id = "agt_team"
    runtime = EmployeeDepartmentRuntime()
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        list_states=lambda: (
            SimpleNamespace(
                agent_id=agent_id,
                tenant_key="tenant-a",
                phase=HirePhase.ACTIVE,
                channel_generation=2,
                bot_principal_id="bot_team",
                app_id="cli_team",
            ),
        )
    )
    runtime._channels = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        status=lambda observed: SimpleNamespace(
            state=ChannelProcessState.READY,
            generation=2,
            ready_metadata={"connection_id": "conn_team"},
        )
        if observed == agent_id
        else None
    )
    events: list[str] = []
    accepted: list[object] = []

    def accept(metadata, payload, *, request_id):
        events.append("accept")
        accepted.append((metadata, payload, request_id))
        return SimpleNamespace(
            acceptance=SimpleNamespace(acceptance_id=acceptance_id)
        )

    runtime._ingress = SimpleNamespace(accept=accept)  # type: ignore[assignment]  # noqa: SLF001
    ledger_attempts: list[None] = []

    def fail_ledger(**_kwargs) -> None:
        events.append("ledger")
        ledger_attempts.append(None)
        raise RuntimeError("transient group ledger failure")

    runtime._group_ledger = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        publish=fail_ledger
    )
    admission_attempts: list[str] = []

    def project_then_fail(observed: str) -> None:
        events.append("admission")
        admission_attempts.append(observed)
        runtime._group_ledger.publish()

    monkeypatch.setattr(runtime, "_admit_employee_ingress_once", project_then_fail)
    backend = composition._RuntimeTeamBackend(runtime, lambda *_args: None)

    observed = backend.submit(
        run_id="run_team",
        step_id="step_build",
        target=composition.TeamTarget(agent_id, "Atlas"),
        tenant_key="tenant-a",
        chat_id="oc_team",
        message_id="om_team",
        requester_principal_id="ou_owner",
        instruction="implement the feature",
        deadline_at="2026-08-12T01:00:00Z",
    )

    assert observed == acceptance_id
    assert ledger_attempts == [None]
    assert len(accepted) == 1
    assert admission_attempts == [acceptance_id]
    assert events == ["accept", "admission", "ledger"]


def test_team_ingress_accept_failure_does_not_publish_group_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_id = "agt_team"
    runtime = EmployeeDepartmentRuntime()
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        list_states=lambda: (
            SimpleNamespace(
                agent_id=agent_id,
                tenant_key="tenant-a",
                phase=HirePhase.ACTIVE,
                channel_generation=2,
                bot_principal_id="bot_team",
                app_id="cli_team",
            ),
        )
    )
    runtime._channels = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        status=lambda _observed: SimpleNamespace(
            state=ChannelProcessState.READY,
            generation=2,
            ready_metadata={"connection_id": "conn_team"},
        )
    )

    def fail_accept(*_args, **_kwargs):
        raise RuntimeError("durable Inbox unavailable")

    runtime._ingress = SimpleNamespace(accept=fail_accept)  # type: ignore[assignment]  # noqa: SLF001
    ledger_attempts: list[None] = []
    runtime._group_ledger = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        publish=lambda **_kwargs: ledger_attempts.append(None)
    )
    admission_attempts: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_admit_employee_ingress_once",
        lambda observed: admission_attempts.append(observed),
    )
    backend = composition._RuntimeTeamBackend(runtime, lambda *_args: None)

    with pytest.raises(RuntimeError, match="durable Inbox unavailable"):
        backend.submit(
            run_id="run_team",
            step_id="step_build",
            target=composition.TeamTarget(agent_id, "Atlas"),
            tenant_key="tenant-a",
            chat_id="oc_team",
            message_id="om_team",
            requester_principal_id="ou_owner",
            instruction="implement the feature",
            deadline_at="2026-08-12T01:00:00Z",
        )

    assert ledger_attempts == []
    assert admission_attempts == []


def test_team_acceptance_attempts_fast_admission_and_notifies_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acceptance_id = "acc_team"
    agent_id = "agt_team"
    runtime = EmployeeDepartmentRuntime()
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        list_states=lambda: (
            SimpleNamespace(
                agent_id=agent_id,
                tenant_key="tenant-a",
                phase=HirePhase.ACTIVE,
                channel_generation=2,
                bot_principal_id="bot_team",
                app_id="cli_team",
            ),
        )
    )
    runtime._channels = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        status=lambda observed: SimpleNamespace(
            state=ChannelProcessState.READY,
            generation=2,
            ready_metadata={"connection_id": "conn_team"},
        )
        if observed == agent_id
        else None
    )
    runtime._ingress = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        accept=lambda *_args, **_kwargs: SimpleNamespace(
            acceptance=SimpleNamespace(acceptance_id=acceptance_id)
        )
    )
    runtime._group_ledger = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        publish=lambda **_kwargs: None
    )
    admission_attempts: list[str] = []
    notifications: list[None] = []

    def fail_admission(observed: str) -> None:
        admission_attempts.append(observed)
        raise RuntimeError("transient admission failure")

    monkeypatch.setattr(runtime, "_admit_employee_ingress_once", fail_admission)
    monkeypatch.setattr(
        runtime,
        "_notify_employee_handoff_progress",
        lambda: notifications.append(None),
    )
    backend = composition._RuntimeTeamBackend(runtime, lambda *_args: None)

    observed = backend.submit(
        run_id="run_team",
        step_id="step_build",
        target=composition.TeamTarget(agent_id, "Atlas"),
        tenant_key="tenant-a",
        chat_id="oc_team",
        message_id="om_team",
        requester_principal_id="ou_owner",
        instruction="implement the feature",
        deadline_at="2026-08-12T01:00:00Z",
    )

    assert observed == acceptance_id
    assert admission_attempts == [acceptance_id]
    assert notifications == [None]


def test_channel_acceptance_admits_during_dispatch_exception_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_wait_completed = threading.Event()
    backoff_entered = threading.Event()
    release_backoff = threading.Event()

    class ControlledStop:
        def __init__(self) -> None:
            self.wait_count = 0

        def clear(self) -> None:
            return None

        def set(self) -> None:
            release_backoff.set()

        def wait(self, _delay: float) -> bool:
            self.wait_count += 1
            if self.wait_count == 1:
                first_wait_completed.set()
                return False
            backoff_entered.set()
            assert release_backoff.wait(5.0)
            return True

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch_stop = ControlledStop()  # type: ignore[assignment]  # noqa: SLF001

    def fail_dispatch() -> bool:
        assert first_wait_completed.is_set()
        raise RuntimeError("transient dispatch failure")

    admitted: list[str] = []
    monkeypatch.setattr(runtime, "_drain_employee_dispatch_once", fail_dispatch)
    monkeypatch.setattr(
        runtime,
        "_admit_employee_ingress_once",
        lambda acceptance_id: admitted.append(acceptance_id) or True,
    )

    runtime._start_dispatch_worker()  # noqa: SLF001
    assert backoff_entered.wait(5.0)
    try:
        asyncio.run(
            runtime._handle_channel_event(  # noqa: SLF001
                "hire_alpha",
                3,
                {
                    "event": "durableIngressAccepted",
                    "data": {"acceptance_id": "acc_during_backoff"},
                },
            )
        )
        assert admitted == ["acc_during_backoff"]
        assert runtime._dispatch_thread is not None  # noqa: SLF001
        assert runtime._dispatch_thread.is_alive()  # noqa: SLF001
    finally:
        release_backoff.set()
        if runtime._dispatch_thread is not None:  # noqa: SLF001
            runtime._dispatch_thread.join(timeout=5.0)  # noqa: SLF001
    assert runtime._dispatch_thread is not None  # noqa: SLF001
    assert not runtime._dispatch_thread.is_alive()  # noqa: SLF001
