from __future__ import annotations

import hashlib
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.autonomous.gateway.coordinator import (
    EmployeeCancellationOutcome,
    TeamAttemptSnapshot,
)
from src.autonomous.gateway.models import GatewayExecutionStatus
from src.autonomous.gateway.projection import (
    ATTEMPT_CANCEL_REQUESTED,
)
from src.autonomous.runtime.employee_actor import EmployeeActorStatus
from tests.autonomous.integration.test_employee_team_gateway import (
    _real_coordinator_harness,
)


def test_cancel_before_permit_execution_never_calls_employee_session(tmp_path, monkeypatch) -> None:
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
    from tests.autonomous.integration.test_employee_team_gateway import (
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

    from tests.autonomous.integration.test_employee_team_gateway import (
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
        if any(event.event_type == "employee.execution_attempt.bound" for event in frame.events)
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
        SimpleNamespace(_dispatch=_Dispatch(EmployeeCancellationOutcome("no_active"), ())),
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
    assert (
        unavailable.cancel(
            "acc_unavailable",
            run_id="run",
            step_id="step",
        ).retry_allowed
        is False
    )

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


def test_team_queued_cancel_is_idempotent_after_effect_terminal_and_restart(
    tmp_path,
) -> None:
    from tests.autonomous.integration.test_employee_team_gateway import (
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
    from tests.autonomous.integration.test_employee_team_gateway import (
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
    from src.autonomous.gateway.models import (
        GatewayExecutionResult,
        GatewayExecutionStatus,
    )
    from tests.autonomous.integration.test_employee_team_gateway import (
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


def test_owner_p2p_status_anchors_scoped_runtime_snapshot_before_delivery() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_status_control"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="cli_alpha",
        bot_principal_id="bot_alpha",
        chat_id="oc_owner_p2p",
        message_id="om_status",
        sender_principal_id="ou_employee_app_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    record = SimpleNamespace(disposition=None, metadata=metadata)
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: record})
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "p2p",
                "content": {"text": " /status "},
            },
        ),
    )
    ingress.get_payload.return_value = payload
    runtime_snapshot = SimpleNamespace(
        status=EmployeeActorStatus.BUSY,
        mailbox_depth=2,
        active_assignment_id="att_private_marker",
    )
    employee_runtime = MagicMock()
    employee_runtime.inspect.return_value = runtime_snapshot
    dispatch = SimpleNamespace(
        employee_runtime=employee_runtime,
        scoped_attempt_status=MagicMock(
            return_value=SimpleNamespace(
                active_count=1,
                stopping_count=0,
                journal_sequence=12,
            )
        ),
    )
    lifecycle = MagicMock()
    events: list[str] = []
    lifecycle.status_response.side_effect = lambda **_kwargs: events.append("outbox")
    ingress.record_disposition.side_effect = lambda *_args, **_kwargs: events.append("disposition")
    department = EmployeeDepartmentRuntime()
    department._ingress = ingress
    department._dispatch = dispatch
    department._outbox_lifecycle = lifecycle
    department._owner_p2p_requester = MagicMock(return_value="ou_owner")
    department._drain_employee_outbox_once = MagicMock(side_effect=lambda: events.append("delivery") or True)

    assert department._handle_control_ingress(acceptance_id) is True

    employee_runtime.inspect.assert_called_once_with("agt_alpha")
    response = lifecycle.status_response.call_args.kwargs
    assert response["succeeded"] is True
    assert "执行中" in response["summary"]
    assert "1 个活动任务" in response["summary"]
    assert "队列：2" in response["summary"]
    assert "att_private_marker" not in response["summary"]
    assert "secret" not in response["summary"]
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="status_completed",
    )
    assert events == ["outbox", "disposition", "delivery"]


def test_owner_p2p_status_arguments_return_durable_usage_without_inspection() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_status_args"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="cli_alpha",
        bot_principal_id="bot_alpha",
        chat_id="oc_owner_p2p",
        message_id="om_status",
        sender_principal_id="ou_employee_app_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)}
    )
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "p2p",
                "content": {"text": "/status details"},
            },
        ),
    )
    ingress.get_payload.return_value = payload
    employee_runtime = MagicMock()
    lifecycle = MagicMock()
    department = EmployeeDepartmentRuntime()
    department._ingress = ingress
    department._dispatch = SimpleNamespace(
        employee_runtime=employee_runtime,
        scoped_attempt_status=MagicMock(),
    )
    department._outbox_lifecycle = lifecycle
    department._owner_p2p_requester = MagicMock(return_value="ou_owner")
    department._drain_employee_outbox_once = MagicMock(return_value=True)

    assert department._handle_control_ingress(acceptance_id) is True

    employee_runtime.inspect.assert_not_called()
    response = lifecycle.status_response.call_args.kwargs
    assert response["succeeded"] is False
    assert response["summary"] == "用法：/status"
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="status_invalid_arguments",
    )


def test_group_status_is_left_for_the_main_bot_group_command_gate() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
    from src.trust.models import ActorKind, EffectiveTrust, TrustZone

    acceptance_id = "acc_group_status"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="cli_alpha",
        bot_principal_id="bot_alpha",
        chat_id="oc_team",
        message_id="om_status",
        sender_principal_id="ou_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)}
    )
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "group",
                "content": {"text": "/status"},
            },
        ),
    )
    ingress.get_payload.return_value = payload
    lifecycle = MagicMock()
    department = EmployeeDepartmentRuntime()
    department._ingress = ingress
    department._dispatch = MagicMock()
    department._outbox_lifecycle = lifecycle
    department._owner_p2p_requester = MagicMock(return_value=None)
    department._managed_employee_ingress_trust = MagicMock(
        return_value=EffectiveTrust(
            zone=TrustZone.MANAGED_AGENT_GROUP,
            actor=ActorKind.OWNER,
            managed_group=None,
            group_revision=None,
            grant_revision=None,
        )
    )

    assert department._handle_control_ingress(acceptance_id) is False
    lifecycle.status_response.assert_not_called()
    ingress.record_disposition.assert_not_called()


def test_owner_p2p_status_reports_unavailable_without_allocating_actor() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_status_unavailable"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="cli_alpha",
        bot_principal_id="bot_alpha",
        chat_id="oc_owner_p2p",
        message_id="om_status",
        sender_principal_id="ou_employee_app_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)}
    )
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "p2p",
                "content": {"text": "/status"},
            },
        ),
    )
    ingress.get_payload.return_value = payload
    lifecycle = MagicMock()
    department = EmployeeDepartmentRuntime()
    department._ingress = ingress
    department._dispatch = SimpleNamespace(
        employee_runtime=None,
        scoped_attempt_status=MagicMock(),
    )
    department._outbox_lifecycle = lifecycle
    department._owner_p2p_requester = MagicMock(return_value="ou_owner")
    department._drain_employee_outbox_once = MagicMock(return_value=True)

    assert department._handle_control_ingress(acceptance_id) is True

    response = lifecycle.status_response.call_args.kwargs
    assert response["succeeded"] is False
    assert response["summary"] == "员工状态暂不可用，请稍后重试。"
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="status_unavailable",
    )


@pytest.mark.parametrize(
    ("status", "label"),
    [
        (EmployeeActorStatus.READY_COLD, "空闲（冷会话）"),
        (EmployeeActorStatus.BUSY, "执行中"),
        (EmployeeActorStatus.DEGRADED, "降级"),
        (EmployeeActorStatus.STOPPING, "停止中"),
    ],
)
def test_status_summary_uses_coarse_actor_states_without_identifiers(
    status,
    label,
) -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    employee_runtime = MagicMock()
    employee_runtime.inspect.return_value = SimpleNamespace(
        status=status,
        mailbox_depth=0,
        active_assignment_id="att_must_not_leak",
    )
    department = EmployeeDepartmentRuntime()
    department._dispatch = SimpleNamespace(
        employee_runtime=employee_runtime,
        scoped_attempt_status=MagicMock(
            return_value=SimpleNamespace(
                active_count=0,
                stopping_count=0,
                journal_sequence=2,
            )
        ),
    )

    summary = department._employee_status_summary(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        chat_id="oc_owner",
        thread_root_id="",
    )

    assert label in summary
    assert "无活动任务" in summary
    assert "att_must_not_leak" not in summary
    assert "agt_alpha" not in summary


def test_status_summary_requests_durable_counts_for_current_thread() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    employee_runtime = MagicMock()
    employee_runtime.inspect.return_value = SimpleNamespace(
        status=EmployeeActorStatus.BUSY,
        mailbox_depth=0,
        active_assignment_id="att_hidden",
    )
    scoped_attempt_status = MagicMock(
        return_value=SimpleNamespace(
            active_count=1,
            stopping_count=0,
            journal_sequence=8,
        )
    )
    department = EmployeeDepartmentRuntime()
    department._dispatch = SimpleNamespace(
        employee_runtime=employee_runtime,
        scoped_attempt_status=scoped_attempt_status,
    )

    summary = department._employee_status_summary(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        chat_id="oc_owner",
        thread_root_id="om_current_root",
    )

    assert "1 个活动任务" in summary
    assert "停止中" not in summary
    scoped_attempt_status.assert_called_once_with(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        chat_id="oc_owner",
        thread_root_id="om_current_root",
    )


def test_status_outbox_failure_does_not_terminalize_or_deliver_ingress() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_status_outbox_failure"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="cli_alpha",
        bot_principal_id="bot_alpha",
        chat_id="oc_owner_p2p",
        message_id="om_status",
        sender_principal_id="ou_employee_app_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)}
    )
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "p2p",
                "content": {"text": "/status"},
            },
        ),
    )
    ingress.get_payload.return_value = payload
    employee_runtime = MagicMock()
    employee_runtime.inspect.return_value = SimpleNamespace(
        status=EmployeeActorStatus.READY_COLD,
        mailbox_depth=0,
        active_assignment_id="",
    )
    lifecycle = MagicMock()
    lifecycle.status_response.side_effect = RuntimeError("journal unavailable")
    department = EmployeeDepartmentRuntime()
    department._ingress = ingress
    department._dispatch = SimpleNamespace(
        employee_runtime=employee_runtime,
        scoped_attempt_status=MagicMock(
            return_value=SimpleNamespace(
                active_count=0,
                stopping_count=0,
                journal_sequence=3,
            )
        ),
    )
    department._outbox_lifecycle = lifecycle
    department._owner_p2p_requester = MagicMock(return_value="ou_owner")
    department._drain_employee_outbox_once = MagicMock(return_value=True)

    with pytest.raises(RuntimeError, match="journal unavailable"):
        department._handle_control_ingress(acceptance_id)

    ingress.record_disposition.assert_not_called()
    department._drain_employee_outbox_once.assert_not_called()


def test_owner_p2p_stop_uses_union_canonical_owner() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_owner_p2p_stop"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        chat_id="oc_owner_p2p",
        message_id="om_current",
        sender_principal_id="ou_employee_app_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    record = SimpleNamespace(disposition=None, metadata=metadata)
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: record})
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "p2p",
                "content": {"text": "/stop"},
            },
        ),
    )
    ingress.get_payload.return_value = payload
    dispatch = MagicMock()
    dispatch.request_cancel.return_value = SimpleNamespace(status="cancel_requested")
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._dispatch = dispatch
    runtime._outbox_lifecycle = MagicMock()
    runtime._drain_employee_outbox_once = MagicMock(return_value=True)
    runtime._owner_p2p_requester = MagicMock(return_value="ou_owner")

    assert runtime._handle_control_ingress(acceptance_id) is True

    runtime._owner_p2p_requester.assert_called_once_with(record, payload)
    dispatch.request_cancel.assert_called_once_with(
        agent_id="agt_alpha",
        chat_id="oc_owner_p2p",
        requester_principal_id="ou_owner",
        command_acceptance_id=acceptance_id,
    )


def test_runtime_does_not_consume_non_control_text() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_normal"
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None)})
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
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)}
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


def test_group_history_is_left_for_the_main_bot_group_command_gate() -> None:
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
        normalized_parts=({"chat_type": "group", "content": {"text": " /history 14 "}},),
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
    runtime._owner_p2p_requester = MagicMock(return_value=None)

    assert runtime._handle_control_ingress(acceptance_id) is False

    history.query.assert_not_called()
    runtime._outbox_lifecycle.read_response.assert_not_called()
    runtime._drain_employee_outbox_once.assert_not_called()
    ingress.record_disposition.assert_not_called()


def test_owner_p2p_history_uses_canonical_owner_and_durable_outbox() -> None:
    from datetime import date

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_owner_p2p_history"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="employee_app",
        chat_id="oc_owner_p2p",
        message_id="om_current",
        sender_principal_id="ou_employee_app_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    record = SimpleNamespace(disposition=None, metadata=metadata)
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: record})
    payload = SimpleNamespace(
        normalized_parts=({"chat_type": "p2p", "content": {"text": " /history 14 "}},),
    )
    ingress.get_payload.return_value = payload
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
    runtime._owner_p2p_requester = MagicMock(return_value="ou_canonical_owner")

    assert runtime._handle_control_ingress(acceptance_id) is True

    runtime._owner_p2p_requester.assert_called_once_with(record, payload)
    request = history.query.call_args.args[0]
    assert request.principal_id == "ou_canonical_owner"
    assert request.receiving_bot_app_id == "employee_app"
    assert request.chat_id == "oc_owner_p2p"
    assert request.chat_type == "p2p"
    spec = history.query.call_args.args[1]
    assert (date.fromisoformat(spec.end_day) - date.fromisoformat(spec.start_day)).days == 13
    runtime._outbox_lifecycle.read_response.assert_called_once()
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="history_completed",
    )


def test_group_memory_is_left_for_the_main_bot_group_command_gate() -> None:
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
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)}
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
    runtime._owner_p2p_requester = MagicMock(return_value=None)

    assert runtime._handle_control_ingress(acceptance_id) is False

    memory.query.assert_not_called()
    runtime._outbox_lifecycle.read_response.assert_not_called()
    runtime._drain_employee_outbox_once.assert_not_called()
    ingress.record_disposition.assert_not_called()


def test_owner_p2p_memory_uses_canonical_owner_and_durable_outbox() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_owner_p2p_memory"
    metadata = SimpleNamespace(
        agent_id="agt_alpha",
        app_id="employee_app",
        chat_id="oc_owner_p2p",
        message_id="om_current",
        sender_principal_id="ou_employee_app_owner",
        tenant_key="tenant_1",
        thread_root_message_id="",
    )
    record = SimpleNamespace(disposition=None, metadata=metadata)
    ingress = MagicMock()
    ingress.state = SimpleNamespace(by_acceptance_id={acceptance_id: record})
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "chat_type": "p2p",
                "content": {
                    "text": "/memory",
                    "principal_id": "ou_forged",
                    "tenant_key": "tenant_forged",
                },
            },
        ),
    )
    ingress.get_payload.return_value = payload
    memory = MagicMock()
    memory.query.return_value = SimpleNamespace(content="scoped summary")
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._data = SimpleNamespace(query=MagicMock(), memory_query=memory)
    runtime._outbox_lifecycle = MagicMock()
    runtime._drain_employee_outbox_once = MagicMock(return_value=True)
    runtime._owner_p2p_requester = MagicMock(return_value="ou_canonical_owner")

    assert runtime._handle_control_ingress(acceptance_id) is True

    runtime._owner_p2p_requester.assert_called_once_with(record, payload)
    request = memory.query.call_args.args[0]
    assert request.principal_id == "ou_canonical_owner"
    assert request.tenant_key == "tenant_1"
    assert request.receiving_bot_app_id == "employee_app"
    assert request.chat_id == "oc_owner_p2p"
    assert request.chat_type == "p2p"
    assert request.requested_agent_id == "agt_alpha"
    assert memory.query.call_args.args[1].full_l1 is False
    runtime._outbox_lifecycle.read_response.assert_called_once()
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="terminal",
        reason_code="memory_completed",
    )


def test_group_stop_is_left_for_the_main_bot_group_command_gate() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_group_stop"
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
        by_acceptance_id={acceptance_id: SimpleNamespace(disposition=None, metadata=metadata)}
    )
    ingress.get_payload.return_value = SimpleNamespace(
        normalized_parts=({"chat_type": "group", "content": {"text": "/stop"}},),
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress
    runtime._dispatch = MagicMock()
    runtime._outbox_lifecycle = MagicMock()
    runtime._drain_employee_outbox_once = MagicMock(return_value=True)
    runtime._owner_p2p_requester = MagicMock(return_value=None)

    assert runtime._handle_control_ingress(acceptance_id) is False

    runtime._dispatch.request_cancel.assert_not_called()
    runtime._outbox_lifecycle.command_response.assert_not_called()
    runtime._drain_employee_outbox_once.assert_not_called()
    ingress.record_disposition.assert_not_called()
