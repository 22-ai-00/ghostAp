from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.autonomous.gateway.coordinator import (
    EmployeeCancellationOutcome,
    TeamAttemptSnapshot,
)
from src.autonomous.gateway.models import GatewayExecutionStatus
from src.autonomous.gateway.projection import (
    ATTEMPT_CANCEL_REQUESTED,
    ATTEMPT_TERMINAL,
)
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.team import (
    EmployeeTeamService,
    TeamRunState,
    TeamTarget,
)
from tests.autonomous.integration.test_employee_slock_gateway import (
    _binding,
    _real_coordinator_harness,
    _runtime_model,
)


def test_cancel_before_permit_execution_never_calls_slock(tmp_path, monkeypatch) -> None:
    harness = _real_coordinator_harness(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        harness.engine,
        "_run_acp_session",
        lambda *_args, **_kwargs: calls.append("called") or "done",
    )
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None

    outcome = harness.coordinator.request_cancel(
        agent_id=prepared.binding.agent_id,
        chat_id=prepared.binding.chat_id,
        requester_principal_id=prepared.binding.requester_principal_id,
        command_acceptance_id="acc_stop_1",
    )
    finalized = harness.coordinator.execute_prepared(prepared)

    assert outcome.status == "cancel_requested"
    assert finalized.status is GatewayExecutionStatus.CANCELED
    assert calls == []
    harness.close()


def test_team_owner_bound_cancellation_anchors_before_live_interrupt(
    tmp_path,
    monkeypatch,
) -> None:
    harness = _real_coordinator_harness(
        tmp_path,
        team_assignment=True,
        team_deadline_at="2026-07-14T00:02:00Z",
    )
    from tests.autonomous.integration.test_employee_slock_gateway import (
        _commit_team_effect,
    )

    aggregate = "teamrun_inactive:analysis"
    _commit_team_effect(harness.writer, aggregate, "prepared")
    _commit_team_effect(harness.writer, aggregate, "executing")
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    observed: list[bool] = []

    def cancel(binding):
        state = harness.coordinator.state.attempts[binding.attempt_id]
        observed.append(state.cancel_requested)
        return True

    monkeypatch.setattr(harness.coordinator._gateway, "cancel_attempt", cancel)  # noqa: SLF001

    outcome = harness.coordinator.request_team_cancel(
        acceptance_id=prepared.binding.acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )

    assert outcome.status == "cancel_requested"
    assert observed == [True]
    harness.close()


def test_team_queued_cancel_retries_when_dispatch_binds_after_head_capture(
    tmp_path,
    monkeypatch,
) -> None:
    """A bind racing queued cancel must still leave a durable cancel frame."""

    from tests.autonomous.integration.test_employee_slock_gateway import (
        _commit_team_effect,
    )

    harness = _real_coordinator_harness(
        tmp_path,
        team_assignment=True,
        team_deadline_at="2026-07-14T00:02:00Z",
    )
    aggregate = "teamrun_inactive:analysis"
    _commit_team_effect(harness.writer, aggregate, "prepared")
    _commit_team_effect(harness.writer, aggregate, "executing")
    acceptance_id = harness.acceptance_ids[0]
    cancel_checked_owner = threading.Event()
    allow_cancel_to_lock = threading.Event()
    original_active = harness.coordinator._team_assignment_effect_is_active  # noqa: SLF001

    def block_cancel_after_head_capture(part):
        if threading.current_thread().name == "queued-team-cancel":
            cancel_checked_owner.set()
            assert allow_cancel_to_lock.wait(2)
        return original_active(part)

    monkeypatch.setattr(
        harness.coordinator,
        "_team_assignment_effect_is_active",
        block_cancel_after_head_capture,
    )
    outcome: list[EmployeeCancellationOutcome] = []
    errors: list[BaseException] = []

    def cancel() -> None:
        try:
            outcome.append(
                harness.coordinator.request_team_cancel(
                    acceptance_id=acceptance_id,
                    team_run_id="teamrun_inactive",
                    team_step_id="analysis",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    thread = threading.Thread(target=cancel, name="queued-team-cancel")
    thread.start()
    assert cancel_checked_owner.wait(2)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    allow_cancel_to_lock.set()
    thread.join(3)

    assert not thread.is_alive()
    assert errors == []
    assert outcome == [
        EmployeeCancellationOutcome(
            "cancel_requested",
            prepared.binding.attempt_id,
            True,
        )
    ]
    frames = tuple(harness.writer.replay())
    bind_sequence = next(
        frame.sequence
        for frame in frames
        if any(
            event.event_type == "employee.execution_attempt.bound"
            for event in frame.events
        )
    )
    cancel_sequence = next(
        frame.sequence
        for frame in frames
        if any(event.event_type == ATTEMPT_CANCEL_REQUESTED for event in frame.events)
    )
    assert bind_sequence < cancel_sequence
    harness.close()


def test_runtime_team_backend_requires_observed_terminal_before_retry(
    monkeypatch,
) -> None:
    from src.autonomous.provisioning.composition import _RuntimeTeamBackend

    class _Dispatch:
        def __init__(self, outcome, snapshots):
            self.outcome = outcome
            self.snapshots = iter(snapshots)

        def request_team_cancel(self, **_kwargs):
            return self.outcome

        def team_attempt_result(self, _acceptance_id):
            value = next(self.snapshots)
            if isinstance(value, BaseException):
                raise value
            return value

    missing = _RuntimeTeamBackend(SimpleNamespace(_dispatch=None), lambda *_args: None)
    no_active = _RuntimeTeamBackend(
        SimpleNamespace(
            _dispatch=_Dispatch(EmployeeCancellationOutcome("no_active"), ())
        ),
        lambda *_args: None,
    )
    unavailable = _RuntimeTeamBackend(
        SimpleNamespace(
            _dispatch=_Dispatch(
                EmployeeCancellationOutcome("cancel_requested", "att_1", True),
                (RuntimeError("gateway unavailable"),),
            )
        ),
        lambda *_args: None,
    )

    assert missing.result("acc_missing").retry_allowed is False
    assert missing.cancel("acc_missing", run_id="run", step_id="step").retry_allowed is False
    assert no_active.cancel("acc_no_active", run_id="run", step_id="step").retry_allowed is False
    assert unavailable.cancel(
        "acc_unavailable",
        run_id="run",
        step_id="step",
    ).retry_allowed is False

    terminal = _RuntimeTeamBackend(
        SimpleNamespace(
            _dispatch=_Dispatch(
                EmployeeCancellationOutcome("already_terminal", "att_2", False),
                (TeamAttemptSnapshot("canceled", error_code="cancel_requested"),),
            )
        ),
        lambda *_args: None,
    )
    observed = terminal.cancel("acc_terminal", run_id="run", step_id="step")
    assert observed.status == "canceled"
    assert observed.retry_allowed is True


def test_runtime_team_backend_cancel_observation_timeout_is_not_retryable(
    monkeypatch,
) -> None:
    from src.autonomous.provisioning.composition import _RuntimeTeamBackend

    class _Dispatch:
        def request_team_cancel(self, **_kwargs):
            return EmployeeCancellationOutcome("cancel_requested", "att_pending", True)

        def team_attempt_result(self, _acceptance_id):
            return None

    ticks = iter((0.0, 6.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    backend = _RuntimeTeamBackend(
        SimpleNamespace(_dispatch=_Dispatch()),
        lambda *_args: None,
    )

    result = backend.cancel("acc_pending", run_id="run", step_id="step")

    assert result.status == "canceled"
    assert result.retry_allowed is False


def test_team_timeout_anchors_cancel_interrupts_live_runner_before_retry(
    tmp_path,
    monkeypatch,
) -> None:
    """One real runner proves timeout cancellation is the retry fence."""

    from src.autonomous.provisioning.composition import _RuntimeTeamBackend
    from src.autonomous.provisioning.hire_state import HirePhase

    harness = _real_coordinator_harness(tmp_path)
    harness.router.reject_dispatch_candidate(
        harness.acceptance_ids[0],
        reason_code="context_unavailable",
    )
    state = TeamRunState(
        run_id="teamrun_live_timeout",
        tenant_key="tenant_1",
        message_id="om_current",
        chat_id="oc_team",
        requester_principal_id="ou_requester",
        task_digest=hashlib.sha256(b"live timeout").hexdigest(),
    )
    entered = threading.Event()
    interrupted = threading.Event()

    def block_runner(*_args, **_kwargs):
        entered.set()
        assert interrupted.wait(3)
        raise CancelledError

    monkeypatch.setattr(harness.engine, "_run_acp_session", block_runner)

    hire_state = SimpleNamespace(
        agent_id="agt_alpha",
        tenant_key="tenant_1",
        phase=HirePhase.ACTIVE,
        channel_generation=3,
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
    )

    class _HireService:
        def synchronize_projection(self):
            return harness.workforce

        def list_states(self):
            return (hire_state,)

    runtime = SimpleNamespace(
        _dispatch=harness.coordinator,
        _ingress=harness.ingress,
        _channels=harness.channels,
        _require_service=lambda: _HireService(),
        _team_execution_ready_agent_ids=lambda _tenant, _chat: frozenset(
            {"agt_alpha"}
        ),
    )
    backend = _RuntimeTeamBackend(runtime, lambda *_args: None)
    service = EmployeeTeamService(
        writer=harness.writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=0.5,
        poll_seconds=0.001,
    )
    service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="team.run.created",
            aggregate_id=state.run_id,
            payload={
                "tenant_key": state.tenant_key,
                "message_id": state.message_id,
                "chat_id": state.chat_id,
                "requester_principal_id": state.requester_principal_id,
                "task_digest": state.task_digest,
                "max_handoffs": 8,
                "max_depth": 4,
                "max_fanout": 4,
            },
        )
    )
    known_acceptances = set(harness.ingress.state.by_acceptance_id)
    team_thread = threading.Thread(
        target=service._execute,  # noqa: SLF001
        args=(
            state,
            "live timeout",
            (TeamTarget("agt_alpha", "Alpha", "developer"),),
        ),
        daemon=True,
    )
    team_thread.start()
    deadline = time.monotonic() + 2
    first_acceptance = ""
    while time.monotonic() < deadline:
        new_ids = set(harness.ingress.state.by_acceptance_id) - known_acceptances
        if new_ids:
            first_acceptance = next(iter(new_ids))
            break
        time.sleep(0.001)
    assert first_acceptance
    assert harness.router.route(first_acceptance).state == "queued"
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    assert prepared.binding.acceptance_id == first_acceptance
    runner = threading.Thread(
        target=harness.coordinator.execute_prepared,
        args=(prepared,),
        daemon=True,
    )
    runner.start()
    assert entered.wait(2)

    original_cancel = harness.engine.cancel_employee_session

    def observe_cancel(agent_id):
        attempt = harness.coordinator.state.attempts[prepared.binding.attempt_id]
        assert attempt.cancel_requested
        interrupted.set()
        return original_cancel(agent_id)

    monkeypatch.setattr(harness.engine, "cancel_employee_session", observe_cancel)
    retry_aggregate = f"{state.run_id}:analysis-retry"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        retry_prepared = any(
            event.event_type == "team.step.prepared"
            and event.aggregate_id == retry_aggregate
            for frame in harness.writer.replay()
            for event in frame.events
        )
        if (
            retry_prepared
            and len(harness.ingress.state.by_acceptance_id)
            == len(known_acceptances) + 2
        ):
            break
        time.sleep(0.001)
    runner.join(3)

    assert not runner.is_alive()
    frames_and_events = [
        (frame.sequence, event)
        for frame in harness.writer.replay()
        for event in frame.events
    ]
    cancel_sequence = next(
        sequence
        for sequence, event in frames_and_events
        if event.event_type == ATTEMPT_CANCEL_REQUESTED
    )
    terminal_sequence = next(
        sequence
        for sequence, event in frames_and_events
        if event.event_type == ATTEMPT_TERMINAL
    )
    retry_sequence = next(
        sequence
        for sequence, event in frames_and_events
        if event.event_type == "team.step.prepared"
        and event.aggregate_id == retry_aggregate
    )
    assert cancel_sequence < terminal_sequence < retry_sequence
    assert len(harness.ingress.state.by_acceptance_id) == len(known_acceptances) + 2
    service.close()
    team_thread.join(3)
    assert not team_thread.is_alive()
    harness.close()


def test_team_queued_cancel_is_idempotent_after_effect_terminal_and_restart(
    tmp_path,
) -> None:
    from tests.autonomous.integration.test_employee_slock_gateway import (
        _commit_team_effect,
    )

    harness = _real_coordinator_harness(
        tmp_path,
        team_assignment=True,
        team_deadline_at="2026-07-14T00:02:00Z",
    )
    aggregate = "teamrun_inactive:analysis"
    _commit_team_effect(harness.writer, aggregate, "prepared")
    _commit_team_effect(harness.writer, aggregate, "executing")
    acceptance_id = harness.acceptance_ids[0]

    first = harness.coordinator.request_team_cancel(
        acceptance_id=acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )
    second = harness.coordinator.request_team_cancel(
        acceptance_id=acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )
    _commit_team_effect(harness.writer, aggregate, "action_required")
    after_effect = harness.coordinator.request_team_cancel(
        acceptance_id=acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )
    after_restart = harness.restart().request_team_cancel(
        acceptance_id=acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )

    assert [item.status for item in (first, second, after_effect, after_restart)] == [
        "cancel_requested",
        "cancel_requested",
        "cancel_requested",
        "cancel_requested",
    ]
    harness.close()


def test_team_live_cancel_is_idempotent_after_effect_terminal_and_restart(
    tmp_path,
) -> None:
    from tests.autonomous.integration.test_employee_slock_gateway import (
        _commit_team_effect,
    )

    harness = _real_coordinator_harness(
        tmp_path,
        team_assignment=True,
        team_deadline_at="2026-07-14T00:02:00Z",
    )
    aggregate = "teamrun_inactive:analysis"
    _commit_team_effect(harness.writer, aggregate, "prepared")
    _commit_team_effect(harness.writer, aggregate, "executing")
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None

    first = harness.coordinator.request_team_cancel(
        acceptance_id=prepared.binding.acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )
    _commit_team_effect(harness.writer, aggregate, "action_required")
    second = harness.coordinator.request_team_cancel(
        acceptance_id=prepared.binding.acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )
    restarted = harness.restart().request_team_cancel(
        acceptance_id=prepared.binding.acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )

    assert [item.status for item in (first, second, restarted)] == [
        "cancel_requested",
        "cancel_requested",
        "cancel_requested",
    ]
    harness.close()


def test_team_cancel_after_gateway_terminal_is_stably_already_terminal(tmp_path) -> None:
    from src.autonomous.ingress.dispatch import (
        GatewayExecutionResult,
        GatewayExecutionStatus,
    )
    from tests.autonomous.integration.test_employee_slock_gateway import (
        _commit_team_effect,
    )

    harness = _real_coordinator_harness(
        tmp_path,
        team_assignment=True,
        team_deadline_at="2026-07-14T00:02:00Z",
    )
    aggregate = "teamrun_inactive:analysis"
    _commit_team_effect(harness.writer, aggregate, "prepared")
    _commit_team_effect(harness.writer, aggregate, "executing")
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    harness.coordinator.finalize_attempt(
        prepared.binding.attempt_id,
        GatewayExecutionResult(GatewayExecutionStatus.COMPLETED, output="done"),
        request_text=prepared.prompt,
    )
    _commit_team_effect(harness.writer, aggregate, "committed")

    outcome = harness.restart().request_team_cancel(
        acceptance_id=prepared.binding.acceptance_id,
        team_run_id="teamrun_inactive",
        team_step_id="analysis",
    )

    assert outcome.status == "already_terminal"
    harness.close()


def test_terminal_first_stop_does_not_create_second_terminal(tmp_path, monkeypatch) -> None:
    harness = _real_coordinator_harness(tmp_path)
    monkeypatch.setattr(harness.engine, "_run_acp_session", lambda *_args, **_kwargs: "done")
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None
    completed = harness.coordinator.execute_prepared(prepared)
    before = harness.writer.get_last_frame().sequence

    outcome = harness.coordinator.request_cancel(
        agent_id=prepared.binding.agent_id,
        chat_id=prepared.binding.chat_id,
        requester_principal_id=prepared.binding.requester_principal_id,
        command_acceptance_id="acc_stop_2",
    )

    assert completed.status is GatewayExecutionStatus.COMPLETED
    assert outcome.status == "already_terminal"
    assert harness.writer.get_last_frame().sequence == before
    harness.close()


def test_stop_revalidates_original_requester_authority(tmp_path) -> None:
    harness = _real_coordinator_harness(tmp_path)
    prepared = harness.coordinator.prepare_next()
    assert prepared is not None

    outcome = harness.coordinator.request_cancel(
        agent_id=prepared.binding.agent_id,
        chat_id=prepared.binding.chat_id,
        requester_principal_id="ou_intruder",
        command_acceptance_id="acc_stop_3",
    )

    assert outcome.status == "forbidden"
    assert harness.coordinator.state.attempts[prepared.binding.attempt_id].cancel_requested is False
    harness.close()


def test_stop_allows_configured_admin_and_team_owner(tmp_path) -> None:
    for index, requester in enumerate(("ou_admin", "ou_owner")):
        harness = _real_coordinator_harness(tmp_path / str(index))
        prepared = harness.coordinator.prepare_next()
        assert prepared is not None
        harness.coordinator._admin_principal_ids = frozenset({"ou_admin"})
        harness.coordinator._team_owner_resolver = lambda _chat: "ou_owner"

        outcome = harness.coordinator.request_cancel(
            agent_id=prepared.binding.agent_id,
            chat_id=prepared.binding.chat_id,
            requester_principal_id=requester,
            command_acceptance_id=f"acc_stop_authority_{index}",
        )

        assert outcome.status == "cancel_requested"
        harness.close()


def test_gateway_running_cancel_invokes_engine_and_overrides_late_success() -> None:
    from src.autonomous.ingress import dispatch as module
    from src.slock_engine.models import AgentIdentity

    binding = _binding(module)
    entered = threading.Event()
    release = threading.Event()

    class _Engine:
        def __init__(self) -> None:
            self.canceled: list[str] = []

        def run_agent_session(self, *_args, **_kwargs):
            entered.set()
            assert release.wait(2)
            return "late success"

        def cancel_employee_session(self, agent_id: str) -> bool:
            self.canceled.append(agent_id)
            release.set()
            return True

    engine = _Engine()
    gateway = module.EmployeeSlockGateway(runtime_mode="legacy_one_shot")
    permit = gateway.issue_permit(
        binding=binding,
        prompt="budgeted",
        engine=engine,
        agent=AgentIdentity(
            agent_id=binding.agent_id,
            agent_type=binding.tool,
            model_name=_runtime_model(binding),
            model_profile=binding.profile,
            reasoning_effort=binding.effort,
            permissions=list(binding.permissions),
            security_profile="employee_v1",
        ),
        timeout_seconds=30,
        env={"HOME": "/tmp/employee"},
    )
    result = []
    thread = threading.Thread(target=lambda: result.append(gateway.execute_permit(permit)))
    thread.start()
    assert entered.wait(1)

    assert gateway.cancel_attempt(binding) is True
    thread.join(2)

    assert engine.canceled == [binding.agent_id]
    assert result[0].status is GatewayExecutionStatus.CANCELED


def test_runtime_consumes_exact_stop_before_router_admission() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_stop_control"
    record = SimpleNamespace(
        disposition=None,
        metadata=SimpleNamespace(
            agent_id="agt_alpha",
            chat_id="oc_team",
            message_id="om_current",
            sender_principal_id="ou_requester",
            tenant_key="tenant_1",
            thread_root_message_id="om_root",
        ),
    )
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: record})
    ingress.get_payload.return_value = SimpleNamespace(
        normalized_parts=({"content": {"text": " /stop "}},),
    )
    dispatch = MagicMock()
    dispatch.request_cancel.return_value = SimpleNamespace(status="cancel_requested")
    lifecycle = MagicMock()
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._dispatch = dispatch
    runtime._outbox_lifecycle = lifecycle
    runtime._drain_employee_outbox_once = MagicMock(return_value=True)

    assert runtime._handle_control_ingress(acceptance_id) is True

    dispatch.request_cancel.assert_called_once()
    lifecycle.command_response.assert_called_once()
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="stop_cancel_requested",
    )


def test_runtime_does_not_consume_non_control_text() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_normal"
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None)}
    )
    ingress.get_payload.return_value = SimpleNamespace(
        normalized_parts=({"content": {"text": "please stop later"}},),
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._dispatch = MagicMock()
    runtime._outbox_lifecycle = MagicMock()

    assert runtime._handle_control_ingress(acceptance_id) is False
    runtime._dispatch.request_cancel.assert_not_called()


def test_runtime_reconciles_membership_event_with_hash_bound_remote_chat() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_membership"
    remote_chat_id = "oc_team"
    metadata = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        app_id="cli_alpha",
        chat_id="oc_" + hashlib.sha256(remote_chat_id.encode()).hexdigest(),
    )
    record = SimpleNamespace(disposition=None, metadata=metadata)
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: record})
    ingress.get_payload.return_value = SimpleNamespace(
        normalized_parts=(
            {
                "type": "membership_event",
                "operation": "added",
                "remote_chat_id": remote_chat_id,
            },
        ),
    )
    membership = MagicMock()
    membership.reconcile_event.return_value = SimpleNamespace(state=SimpleNamespace(value="active"))
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._membership = membership
    runtime._membership_event_transport_is_current = MagicMock(return_value=True)

    assert runtime._handle_control_ingress(acceptance_id) is True

    runtime._membership_event_transport_is_current.assert_called_once_with(
        metadata,
        remote_chat_id,
    )
    membership.reconcile_event.assert_called_once_with(
        tenant_key="tenant_1",
        chat_id=remote_chat_id,
        agent_id="agt_alpha",
        app_id="cli_alpha",
        observed_is_member=True,
    )
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="membership_active",
    )


def test_runtime_rejects_membership_event_with_unbound_remote_chat() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_membership_tampered"
    metadata = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        app_id="cli_alpha",
        chat_id="oc_" + hashlib.sha256(b"oc_expected").hexdigest(),
    )
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={
            acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)
        }
    )
    ingress.get_payload.return_value = SimpleNamespace(
        normalized_parts=(
            {
                "type": "membership_event",
                "operation": "added",
                "remote_chat_id": "oc_tampered",
            },
        ),
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._membership = MagicMock()

    assert runtime._handle_control_ingress(acceptance_id) is True

    runtime._membership.reconcile_event.assert_not_called()
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="ignored",
        reason_code="membership_unmanaged",
    )


def test_runtime_consumes_history_through_authoritative_read_and_outbox() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_history_control"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="employee_app",
        chat_id="oc_team",
        message_id="om_current",
        sender_principal_id="ou_member",
        tenant_key="tenant_1",
        thread_root_message_id="om_root",
    )
    record = SimpleNamespace(disposition=None, metadata=metadata)
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: record})
    ingress.get_payload.return_value = SimpleNamespace(
        normalized_parts=(
            {"chat_type": "group", "content": {"text": " /history 14 "}},
        ),
    )
    history = MagicMock()
    history.query.return_value = SimpleNamespace(records=())
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._data = SimpleNamespace(
        query=history,
        memory_query=MagicMock(),
        service=SimpleNamespace(shard_timezone="UTC"),
    )
    runtime._outbox_lifecycle = MagicMock()
    runtime._drain_employee_outbox_once = MagicMock(return_value=True)

    assert runtime._handle_control_ingress(acceptance_id) is True

    request = history.query.call_args.args[0]
    assert request.principal_id == "ou_member"
    assert request.receiving_bot_app_id == "employee_app"
    assert request.chat_id == "oc_team"
    assert request.chat_type == "group"
    spec = history.query.call_args.args[1]
    from datetime import date

    assert (date.fromisoformat(spec.end_day) - date.fromisoformat(spec.start_day)).days == 13
    runtime._outbox_lifecycle.read_response.assert_called_once()
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="history_completed",
    )


def test_runtime_memory_ignores_payload_authority_and_uses_ingress_metadata() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_memory_control"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="employee_app",
        chat_id="oc_team",
        message_id="om_current",
        sender_principal_id="ou_member",
        tenant_key="tenant_1",
        thread_root_message_id="om_root",
    )
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={
            acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)
        }
    )
    ingress.get_payload.return_value = SimpleNamespace(
        normalized_parts=(
            {
                "chat_type": "group",
                "content": {
                    "text": "/memory",
                    "principal_id": "ou_admin",
                    "tenant_key": "tenant_forged",
                },
            },
        ),
    )
    memory = MagicMock()
    memory.query.return_value = SimpleNamespace(content="scoped summary")
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._data = SimpleNamespace(query=MagicMock(), memory_query=memory)
    runtime._outbox_lifecycle = MagicMock()
    runtime._drain_employee_outbox_once = MagicMock(return_value=True)

    assert runtime._handle_control_ingress(acceptance_id) is True

    request = memory.query.call_args.args[0]
    assert request.principal_id == "ou_member"
    assert request.tenant_key == "tenant_1"
    assert request.requested_agent_id == "agt_alpha"
    assert memory.query.call_args.args[1].full_l1 is False
