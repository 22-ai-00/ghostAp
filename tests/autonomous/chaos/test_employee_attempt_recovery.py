"""Task 6 crash recovery contract for unknown employee ACP dispatch."""

from __future__ import annotations

import hashlib
import threading


class _TraceRLock:
    def __init__(self, name, level, held, trace):
        self._name = name
        self._level = level
        self._held = held
        self._trace = trace
        self._lock = threading.RLock()
        self._depth = {}

    def acquire(self, blocking=True, timeout=-1):
        acquired = self._lock.acquire(blocking, timeout)
        if not acquired:
            return False
        thread_id = threading.get_ident()
        depth = self._depth.get(thread_id, 0)
        if depth == 0:
            stack = self._held.setdefault(thread_id, [])
            if stack and self._level <= stack[-1][0]:
                self._lock.release()
                raise RuntimeError(
                    f"lock inversion: {stack[-1][1]} -> {self._name}"
                )
            stack.append((self._level, self._name))
            self._trace.append(
                (thread_id, "acquire", self._name, tuple(name for _, name in stack))
            )
        self._depth[thread_id] = depth + 1
        return True

    def release(self):
        thread_id = threading.get_ident()
        depth = self._depth[thread_id] - 1
        if depth == 0:
            del self._depth[thread_id]
            stack = self._held[thread_id]
            level, name = stack.pop()
            assert (level, name) == (self._level, self._name)
            self._trace.append(
                (thread_id, "release", self._name, tuple(item for _, item in stack))
            )
        else:
            self._depth[thread_id] = depth
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_args):
        self.release()


def test_unknown_dispatch_recovers_action_required_without_rerun(
    tmp_path,
    monkeypatch,
) -> None:
    """EI-RECOVERY-01 replays an unknown dispatch once and never reruns ACP."""

    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    acp_calls = []

    def forbidden_acp(*args, **kwargs):
        acp_calls.append((args, kwargs))
        raise AssertionError("recovery must not rerun ACP")

    monkeypatch.setattr(harness.engine, "_run_acp_session", forbidden_acp)
    harness.close()
    reopened = _reopen_recovery_harness(tmp_path, harness)
    recovered = reopened.coordinator.recover_incomplete_attempts()
    assert len(recovered) == 1
    assert recovered[0].attempt_id == prepared.binding.attempt_id
    assert recovered[0].status.value == "action_required"
    terminal_frame = tuple(reopened.writer.replay())[-1]
    assert [event.event_type for event in terminal_frame.events] == [
        "employee.history.recorded",
        "employee.execution_attempt.terminal",
        "employee.ingress.router_terminal",
    ]
    assert terminal_frame.events[1].payload["status"] == "action_required"
    sequence = terminal_frame.sequence

    assert reopened.restart().recover_incomplete_attempts() == ()
    assert reopened.writer.anchor.read().sequence == sequence
    assert acp_calls == []
    reopened.close()


def test_actor_terminal_is_preserved_when_gateway_recovery_follows_process_death(
    tmp_path,
    monkeypatch,
) -> None:
    """A durable Actor failure is not downgraded to an unknown dispatch outcome."""

    from src.autonomous.journal.frame import JournalEvent
    from src.autonomous.journal.writer import CommitState
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    acp_calls = []

    def forbidden_acp(*args, **kwargs):
        acp_calls.append((args, kwargs))
        raise AssertionError("recovery must not rerun ACP")

    monkeypatch.setattr(harness.engine, "_run_acp_session", forbidden_acp)
    aggregate = f"employee-assignment:{prepared.binding.attempt_id}"
    terminal = JournalEvent(
        event_type="employee.actor.assignment_terminal",
        aggregate_id=aggregate,
        payload={
            "agent_id": prepared.binding.agent_id,
            "status": "action_required",
            "error_code": "employee_session_failed",
            "output_digest": "",
        },
    )
    result = harness.writer.commit(
        (terminal,),
        harness.writer.get_aggregate_versions((aggregate,)),
    )
    assert result.state is CommitState.ANCHORED
    harness.close()

    reopened = _reopen_recovery_harness(
        tmp_path,
        harness,
        employee_runtime_mode="actor",
    )
    runtime = reopened.coordinator.employee_runtime
    assert runtime is not None
    try:
        runtime.recover()

        recovered = reopened.coordinator.recover_incomplete_attempts()
        assert len(recovered) == 1
        snapshot = reopened.coordinator.team_attempt_result(prepared.binding.acceptance_id)
        assert snapshot is not None
        assert snapshot.status == "action_required"
        assert snapshot.error_code == "employee_session_failed"
        gateway_terminals = [
            event
            for frame in reopened.writer.replay()
            for event in frame.events
            if event.aggregate_id == prepared.binding.attempt_id
            and event.event_type == "employee.execution_attempt.terminal"
        ]
        assert len(gateway_terminals) == 1
        assert reopened.coordinator.recover_incomplete_attempts() == ()
        assert acp_calls == []
    finally:
        reopened.close()


def test_actor_completed_digest_without_result_is_action_required_after_restart(
    tmp_path,
    monkeypatch,
) -> None:
    """A digest proves completion, but cannot replace the lost result body."""

    from src.autonomous.journal.frame import JournalEvent
    from src.autonomous.journal.writer import CommitState
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    acp_calls = []

    def forbidden_acp(*args, **kwargs):
        acp_calls.append((args, kwargs))
        raise AssertionError("recovery must not rerun ACP")

    monkeypatch.setattr(harness.engine, "_run_acp_session", forbidden_acp)
    aggregate = f"employee-assignment:{prepared.binding.attempt_id}"
    terminal = JournalEvent(
        event_type="employee.actor.assignment_terminal",
        aggregate_id=aggregate,
        payload={
            "agent_id": prepared.binding.agent_id,
            "status": "completed",
            "error_code": "",
            "output_digest": hashlib.sha256(b"lost result").hexdigest(),
        },
    )
    result = harness.writer.commit(
        (terminal,),
        harness.writer.get_aggregate_versions((aggregate,)),
    )
    assert result.state is CommitState.ANCHORED
    harness.close()

    reopened = _reopen_recovery_harness(
        tmp_path,
        harness,
        employee_runtime_mode="actor",
    )
    runtime = reopened.coordinator.employee_runtime
    assert runtime is not None
    try:
        runtime.recover()

        recovered = reopened.coordinator.recover_incomplete_attempts()
        assert len(recovered) == 1
        assert recovered[0].status.value == "action_required"
        snapshot = reopened.coordinator.team_attempt_result(prepared.binding.acceptance_id)
        assert snapshot is not None
        assert snapshot.status == "action_required"
        assert snapshot.error_code == "employee_result_unavailable_after_restart"
        assert reopened.coordinator.recover_incomplete_attempts() == ()
        assert acp_calls == []
    finally:
        reopened.close()


def _reopen_recovery_harness(
    tmp_path,
    prior,
    *,
    employee_runtime_mode="legacy_one_shot",
):
    """Open new Writer/Blob/Router/Data/Hire projection owners from disk."""

    from contextlib import contextmanager
    from types import SimpleNamespace

    from src.autonomous.data.projection import DataProjectionState
    from src.autonomous.data.service import EmployeeDataService
    from src.autonomous.gateway.coordinator import EmployeeDispatchCoordinator
    from src.autonomous.gateway.env_scope import EmployeeProcessEnvironmentMaterial
    from src.autonomous.ingress.projection import IngressProjectionState
    from src.autonomous.ingress.router import DurableEmployeeIngressRouter
    from src.autonomous.ingress.service import EmployeeIngressService
    from src.autonomous.journal.anchor import FileAnchor
    from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
    from src.autonomous.journal.projections import ProjectionState, apply_frame
    from src.autonomous.journal.writer import JournalWriter
    from src.autonomous.workforce.projection import workforce_projection_guard
    from src.autonomous.workforce.registry import ProjectedAgentRegistry

    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "journal-anchor.json"),
        hmac_key=b"real-coordinator-harness-key-32bytes",
    )
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "ingress-blobs",
            AesGcmEncryptionProvider(lambda _ref: b"i" * 32),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="ingress-key",
    )
    workforce = ProjectionState()
    workforce.employees.update(prior.workforce.employees)
    workforce.bot_principals.update(prior.workforce.bot_principals)
    for frame in writer.replay():
        apply_frame(workforce, frame)

    router = DurableEmployeeIngressRouter(
        writer=writer,
        ingress_service=ingress,
        registry_provider=lambda: ProjectedAgentRegistry(
            workforce,
            storage_base_path=str(tmp_path / "registry-employee"),
        ),
        channel_status_provider=prior.router._channels,  # noqa: SLF001
        requester_acl=prior.router._requester_acl,  # noqa: SLF001
        queue_limits=prior.router._limits,  # noqa: SLF001
        attachment_staging=prior.router._attachment_staging,  # noqa: SLF001
        membership_health=prior.router._membership_health,  # noqa: SLF001
        constraints_digest=prior.router._constraints_digest,  # noqa: SLF001
        system_prompt_token_reserve=prior.router._reserve,  # noqa: SLF001
    )
    data = EmployeeDataService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "data-blobs",
            AesGcmEncryptionProvider(lambda _ref: b"d" * 32),
        ),
        data_state=DataProjectionState(),
        active_key_id="data-key",
    )
    data.rebuild_projection()

    class _Hire:
        projection_state = workforce

        @contextmanager
        def employee_dispatch_guard(self):
            with workforce_projection_guard():
                yield

        def synchronize_projection_unlocked(self):
            for frame in writer.replay(
                from_sequence=self.projection_state.cursor_sequence + 1
            ):
                apply_frame(self.projection_state, frame)
            return self.projection_state

    coordinator_kwargs = dict(
        employee_runtime_mode=employee_runtime_mode,
        writer=writer,
        hire_service=_Hire(),
        ingress_service=ingress,
        router=router,
        data_service=data,
        channel_supervisor=prior.channels,
        team_runtime=prior.team_runtime,
        context_service=prior.context,
        environment_provider=lambda authority: EmployeeProcessEnvironmentMaterial(
            tenant_key=authority.tenant_key,
            agent_id=authority.agent_id,
            employee_version=authority.employee_version,
            credential_ref=authority.credential_ref,
            runtime_env={"PATH": "/usr/bin"},
            credential_env={},
        ),
        registry_factory=lambda state: ProjectedAgentRegistry(
            state,
            storage_base_path=str(tmp_path / "registry-employee"),
        ),
    )
    coordinator = EmployeeDispatchCoordinator(**coordinator_kwargs)

    def close():
        coordinator.close()
        data.close()
        ingress.close()
        writer.close()

    return SimpleNamespace(
        coordinator=coordinator,
        writer=writer,
        router=router,
        data=data,
        restart=lambda: EmployeeDispatchCoordinator(**coordinator_kwargs),
        close=close,
    )


def test_dispatch_lock_prefix_is_exposed_in_one_forward_order() -> None:
    """All Task 6 participants expose guards that avoid private-lock coupling."""

    from src.autonomous.data.service import EmployeeDataService
    from src.autonomous.ingress.router import DurableEmployeeIngressRouter
    from src.autonomous.ingress.service import EmployeeIngressService
    from src.autonomous.provisioning.hire_service import ProductionEmployeeHireService
    from src.autonomous.supervisor.employee_channels import EmployeeChannelSupervisor

    assert hasattr(ProductionEmployeeHireService, "employee_dispatch_guard")
    assert hasattr(EmployeeIngressService, "employee_dispatch_guard")
    assert not hasattr(DurableEmployeeIngressRouter, "employee_dispatch_guard")
    assert hasattr(EmployeeDataService, "employee_dispatch_guard")
    assert hasattr(EmployeeChannelSupervisor, "employee_dispatch_guard")


def test_committed_attempt_recovery_has_no_execution_entrypoint() -> None:
    """An unknown external outcome is terminalized without another ACP call."""

    from src.autonomous.ingress import dispatch

    assert hasattr(dispatch, "EmployeeDispatchCoordinator")
    assert hasattr(dispatch.EmployeeDispatchCoordinator, "recover_incomplete_attempts")


def test_recovery_commit_section_never_replays_full_journal(
    tmp_path,
    monkeypatch,
) -> None:
    from contextlib import contextmanager

    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    assert harness.coordinator.prepare_next() is not None
    original_guard = harness.writer.transaction_guard
    original_replay = harness.writer.replay
    in_transaction = False

    @contextmanager
    def guarded_transaction():
        nonlocal in_transaction
        with original_guard():
            in_transaction = True
            try:
                yield
            finally:
                in_transaction = False

    def checked_replay(*args, **kwargs):
        assert not in_transaction, "full Journal replay inside recovery commit"
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(harness.writer, "transaction_guard", guarded_transaction)
    monkeypatch.setattr(harness.writer, "replay", checked_replay)
    recovered = harness.coordinator.recover_incomplete_attempts()
    assert len(recovered) == 1
    harness.close()


def test_unknown_dispatch_recovery_has_no_public_terminal_event_builder() -> None:
    """Only the coordinator may build and commit recovery terminal events."""

    from src.autonomous.ingress import dispatch

    assert not hasattr(dispatch, "build_unknown_dispatch_terminal_events")


def test_hire_entrypoints_follow_workforce_hire_writer_lock_trace(
    tmp_path,
    monkeypatch,
) -> None:
    """Activation, revalidation, recovery, and close never invert the prefix."""

    import src.autonomous.workforce.projection as workforce_projection
    from tests.autonomous.integration.test_employee_hire_composition import (
        _activate_employee,
        _Channels,
        _Registrar,
        _runtime,
        _settings,
        _Slash,
    )

    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_args: None,
    )
    assert runtime.hire_service is not None
    service = runtime.hire_service
    held = {}
    trace = []
    monkeypatch.setattr(
        workforce_projection,
        "_WORKFORCE_COMMIT_LOCK",
        _TraceRLock("workforce", 0, held, trace),
    )
    service._mutex = _TraceRLock("hire", 1, held, trace)  # noqa: SLF001
    service._writer._mutex = _TraceRLock(  # noqa: SLF001
        "writer",
        2,
        held,
        trace,
    )

    active = _activate_employee(runtime, channels)
    service.begin_channel_revalidation(
        active.intent_id,
        observed_generation=active.channel_generation,
    )
    service.recover()
    runtime.close()

    acquired_stacks = [item[3] for item in trace if item[1] == "acquire"]
    assert ("workforce", "hire", "writer") in acquired_stacks
    assert all(
        not (stack[0] == "hire" and len(stack) > 1)
        for stack in acquired_stacks
    )


def test_coordinator_prepare_and_finalize_follow_complete_lock_order(
    tmp_path,
    monkeypatch,
) -> None:
    import src.autonomous.workforce.projection as workforce_projection
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    held = {}
    trace = []
    monkeypatch.setattr(
        workforce_projection,
        "_WORKFORCE_COMMIT_LOCK",
        _TraceRLock("workforce", 1, held, trace),
    )
    harness.hire._lock = _TraceRLock("hire", 2, held, trace)  # noqa: SLF001
    harness.ingress._mutex = _TraceRLock("ingress", 3, held, trace)  # noqa: SLF001
    harness.router._mutex = _TraceRLock("router", 4, held, trace)  # noqa: SLF001
    harness.data._mutex = _TraceRLock("data", 5, held, trace)  # noqa: SLF001
    harness.channels._lock = _TraceRLock("channel", 6, held, trace)  # noqa: SLF001
    harness.writer._transaction_mutex = _TraceRLock(  # noqa: SLF001
        "writer",
        7,
        held,
        trace,
    )
    original_assemble = harness.context.assemble
    original_stage = harness.data.stage_history_payload

    def assemble(*args, **kwargs):
        assert not held.get(threading.get_ident(), [])
        return original_assemble(*args, **kwargs)

    def stage(*args, **kwargs):
        assert not held.get(threading.get_ident(), [])
        return original_stage(*args, **kwargs)

    def acp(*_args, **_kwargs):
        assert not held.get(threading.get_ident(), [])
        return "done"

    monkeypatch.setattr(harness.context, "assemble", assemble)
    monkeypatch.setattr(harness.data, "stage_history_payload", stage)
    monkeypatch.setattr(harness.engine, "_run_acp_session", acp)

    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    harness.coordinator.execute_prepared(prepared)

    acquired_stacks = [item[3] for item in trace if item[1] == "acquire"]
    expected = (
        "workforce",
        "hire",
        "ingress",
        "router",
        "data",
        "channel",
        "writer",
    )
    assert expected in acquired_stacks
    assert all(
        tuple(expected.index(name) for name in stack)
        == tuple(sorted(expected.index(name) for name in stack))
        for stack in acquired_stacks
    )
    harness.close()
