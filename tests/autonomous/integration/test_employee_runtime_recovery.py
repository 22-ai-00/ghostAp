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
from src.autonomous.supervisor.employee_channels import ChannelProcessState
from src.autonomous.workforce.projection import commit_workforce_events

HMAC_KEY = b"employee-runtime-recovery-key-32!"


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
            error_code="recovery_exhausted",
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
