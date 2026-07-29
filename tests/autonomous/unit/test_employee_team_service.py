from __future__ import annotations

import hashlib
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from src.autonomous.team import (
    EmployeeTeamService,
    TeamAdmissionError,
    TeamAttemptResult,
    TeamRunState,
    TeamTarget,
)
from src.autonomous.team.service import TeamServiceError
from tests.autonomous.workforce_helpers import make_writer


class _Backend:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, str, str]] = []
        self.notifications: list[tuple[str, str, str]] = []
        self.deadlines: list[str] = []

    def list_active(self, tenant_key: str, chat_id: str):
        assert (tenant_key, chat_id) == ("tenant_1", "oc_team")
        return (
            TeamTarget("agt_lead", "Lead", "coder"),
            TeamTarget("agt_review", "Review", "reviewer"),
        )

    def submit(self, *, run_id, step_id, target, instruction, **kwargs):
        self.submissions.append((step_id, target.agent_id, instruction))
        self.deadlines.append(kwargs["deadline_at"])
        return f"acc_{step_id}"

    def result(self, acceptance_id: str):
        step = acceptance_id.removeprefix("acc_")
        return TeamAttemptResult(
            "completed",
            output=f"output:{step}",
            history_record_id=f"hist_{step}",
        )

    def notify(self, message_id: str, chat_id: str, result: str) -> None:
        self.notifications.append((message_id, chat_id, result))


class _DeadlineBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.result_calls = 0

    def result(self, acceptance_id: str):
        self.result_calls += 1
        if self.result_calls == 1:
            return None
        return TeamAttemptResult(
            "timeout",
            history_record_id="hist_timeout",
            error_code="slock_session_timeout",
        )


class _TimeoutCancelBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.cancellations: list[tuple[str, str, str]] = []

    def result(self, acceptance_id: str):
        return None

    def cancel(self, acceptance_id: str, *, run_id: str, step_id: str):
        self.cancellations.append((acceptance_id, run_id, step_id))
        return TeamAttemptResult(
            "canceled",
            error_code="team_step_timeout",
            retry_allowed=False,
        )


class _CloseBarrierBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.result_entered = threading.Event()
        self.result_release = threading.Event()

    def result(self, acceptance_id: str):
        self.result_entered.set()
        assert self.result_release.wait(2)
        return None

    def cancel(self, acceptance_id: str, *, run_id: str, step_id: str):
        return TeamAttemptResult("canceled", error_code="team_service_stopping")


class _BlockingNotifyBackend(_Backend):
    def __init__(self, *, active: bool = True) -> None:
        super().__init__()
        self.active = active
        self.notify_entered = threading.Event()
        self.notify_release = threading.Event()

    def list_active(self, tenant_key: str, chat_id: str):
        return super().list_active(tenant_key, chat_id) if self.active else ()

    def notify(self, message_id: str, chat_id: str, result: str) -> None:
        self.notify_entered.set()
        assert self.notify_release.wait(2)
        super().notify(message_id, chat_id, result)


class _NoActiveEmployeeBackend(_Backend):
    def list_active(self, tenant_key: str, chat_id: str):
        assert (tenant_key, chat_id) == ("tenant_1", "oc_team")
        return ()


class _BlockingAdmissionBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.list_entered = threading.Event()
        self.list_release = threading.Event()

    def list_active(self, tenant_key: str, chat_id: str):
        self.list_entered.set()
        assert self.list_release.wait(2)
        return super().list_active(tenant_key, chat_id)


class _ConcurrentAdmissionBackend(_Backend):
    def __init__(self) -> None:
        super().__init__()
        self.list_barrier = threading.Barrier(2)
        self.result_release = threading.Event()

    def list_active(self, tenant_key: str, chat_id: str):
        self.list_barrier.wait(2)
        return super().list_active(tenant_key, chat_id)

    def result(self, acceptance_id: str):
        assert self.result_release.wait(2)
        return super().result(acceptance_id)


def test_team_admission_rejects_before_creating_run_without_active_employee(
    tmp_path,
) -> None:
    writer = make_writer(tmp_path)
    backend = _NoActiveEmployeeBackend()
    service = EmployeeTeamService(
        writer=writer, backend=backend, runtime_mode="legacy_pipeline"
    )

    with pytest.raises(TeamAdmissionError, match="no_active_team_employee") as error:
        service.start_task(
            tenant_key="tenant_1",
            message_id="om_no_employee",
            chat_id="oc_team",
            requester_principal_id="ou_user",
            task="大家能做个自我介绍嘛？",
        )

    assert error.value.error_code == "no_active_team_employee"
    assert not any(
        event.event_type == "team.run.created"
        for frame in writer.replay()
        for event in frame.events
    )
    assert backend.notifications == []
    service.close()
    writer.close()


def test_existing_terminal_team_run_is_not_readmitted(tmp_path) -> None:
    from src.autonomous.journal.frame import JournalEvent

    writer = make_writer(tmp_path)
    service = EmployeeTeamService(
        writer=writer, backend=_Backend(), runtime_mode="legacy_pipeline"
    )
    run_id = "teamrun_" + hashlib.sha256(b"tenant_1\0om_terminal_replay").hexdigest()
    service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="team.run.created",
            aggregate_id=run_id,
            payload={
                "tenant_key": "tenant_1",
                "message_id": "om_terminal_replay",
                "chat_id": "oc_team",
                "requester_principal_id": "ou_user",
                "task_digest": "0" * 64,
                "max_handoffs": 8,
                "max_depth": 4,
                "max_fanout": 4,
            },
        )
    )
    service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="team.run.action_required",
            aggregate_id=run_id,
            payload={"error_code": "analysis_failed"},
        )
    )

    with pytest.raises(TeamAdmissionError, match="team_run_action_required"):
        service.start_task(
            tenant_key="tenant_1",
            message_id="om_terminal_replay",
            chat_id="oc_team",
            requester_principal_id="ou_user",
            task="重复投递",
        )

    service.close()
    writer.close()


def test_close_during_admission_prevents_run_creation(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _BlockingAdmissionBackend()
    service = EmployeeTeamService(
        writer=writer, backend=backend, runtime_mode="legacy_pipeline"
    )
    errors: list[BaseException] = []

    def admit() -> None:
        try:
            service.start_task(
                tenant_key="tenant_1",
                message_id="om_close_race",
                chat_id="oc_team",
                requester_principal_id="ou_user",
                task="关闭竞态",
            )
        except BaseException as error:  # noqa: BLE001 - captured from worker thread
            errors.append(error)

    thread = threading.Thread(target=admit)
    thread.start()
    assert backend.list_entered.wait(2)
    service.close()
    backend.list_release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], TeamServiceError)
    assert str(errors[0]) == "team service is closed"
    assert not any(
        event.event_type == "team.run.created"
        for frame in writer.replay()
        for event in frame.events
    )
    assert backend.submissions == []
    assert backend.notifications == []
    writer.close()


def test_concurrent_same_message_creates_one_team_run(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _ConcurrentAdmissionBackend()
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=1,
        poll_seconds=0.001,
    )
    states: list[TeamRunState] = []
    errors: list[BaseException] = []

    def admit() -> None:
        try:
            states.append(
                service.start_task(
                    tenant_key="tenant_1",
                    message_id="om_concurrent",
                    chat_id="oc_team",
                    requester_principal_id="ou_user",
                    task="并发准入",
                )
            )
        except BaseException as error:  # noqa: BLE001 - captured from worker thread
            errors.append(error)

    threads = [threading.Thread(target=admit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert len(states) == 2
    assert states[0].run_id == states[1].run_id
    created = [
        event
        for frame in writer.replay()
        for event in frame.events
        if event.event_type == "team.run.created"
    ]
    assert len(created) == 1

    backend.result_release.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        completed = service.get_run(states[0].run_id)
        if completed is not None and completed.status == "completed":
            break
        time.sleep(0.01)
    assert completed is not None and completed.status == "completed"
    assert len(backend.submissions) == 3
    assert len(backend.notifications) == 1
    service.close()
    writer.close()


def test_v1_team_run_uses_fixed_analysis_review_synthesis_pipeline(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _Backend()
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=1,
        poll_seconds=0.001,
    )

    admitted = service.start_task(
        tenant_key="tenant_1",
        message_id="om_task",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="修复团队模式",
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = service.get_run(admitted.run_id)
        if state is not None and state.status == "completed":
            break
        time.sleep(0.01)

    assert state is not None and state.status == "completed"
    assert [item[:2] for item in backend.submissions] == [
        ("analysis", "agt_lead"),
        ("review", "agt_review"),
        ("synthesis", "agt_lead"),
    ]
    assert "output:analysis" in backend.submissions[1][2]
    assert "output:review" in backend.submissions[2][2]
    assert backend.notifications == [("om_task", "oc_team", "output:synthesis")]
    service.close()
    writer.close()


def test_v1_team_notification_preserves_durable_recipient_scope(tmp_path) -> None:
    class _ScopedBackend(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.recipient_scopes: list[tuple[str, str]] = []

        def notify(
            self,
            message_id: str,
            chat_id: str,
            result: str,
            *,
            tenant_key: str,
            requester_principal_id: str,
        ) -> None:
            self.recipient_scopes.append(
                (tenant_key, requester_principal_id)
            )
            super().notify(message_id, chat_id, result)

    writer = make_writer(tmp_path)
    backend = _ScopedBackend()
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=1,
        poll_seconds=0.001,
    )

    admitted = service.start_task(
        tenant_key="tenant_1",
        message_id="om_scoped_v1",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="修复后回报",
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = service.get_run(admitted.run_id)
        if state is not None and state.status != "running":
            break
        time.sleep(0.01)

    assert state is not None and state.status == "completed"
    assert backend.recipient_scopes == [("tenant_1", "ou_user")]
    service.close()
    writer.close()


def test_restart_marks_unfinished_run_action_required(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _Backend()
    service = EmployeeTeamService(
        writer=writer, backend=backend, runtime_mode="legacy_pipeline"
    )
    service._executor.shutdown(wait=True)
    service._closed = True

    # Anchor only the admission to model a crash before the first dispatch.
    from src.autonomous.journal.frame import JournalEvent

    service._commit(
        JournalEvent(
            event_type="team.run.created",
            aggregate_id="teamrun_crashed",
            payload={
                "tenant_key": "tenant_1",
                "message_id": "om_crashed",
                "chat_id": "oc_team",
                "requester_principal_id": "ou_user",
                "task_digest": "0" * 64,
                "max_handoffs": 8,
                "max_depth": 4,
                "max_fanout": 4,
            },
        )
    )

    assert service.recover() == 1
    assert service.get_run("teamrun_crashed").status == "action_required"
    assert any(
        event.event_type == "team.run.action_required"
        and event.payload == {"error_code": "restart_instruction_unavailable"}
        for frame in writer.replay()
        for event in frame.events
    )
    writer.close()


def test_deadline_final_poll_observes_gateway_terminal_result(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _DeadlineBackend()
    base = datetime(2026, 7, 14, tzinfo=UTC)
    clock_values = iter((base, base, base + timedelta(seconds=1)))
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=1,
        poll_seconds=0,
        clock=lambda: next(clock_values),
    )
    state = TeamRunState(
        run_id="teamrun_deadline",
        tenant_key="tenant_1",
        message_id="om_deadline",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task_digest="0" * 64,
    )
    result = service._run_step(
        state,
        step_id="analysis",
        depth=1,
        target=TeamTarget("agt_lead", "Lead", "coder"),
        instruction="deadline boundary",
    )

    assert backend.result_calls == 2
    assert result.status == "timeout"
    assert result.error_code == "slock_session_timeout"
    assert any(
        event.event_type == "team.step.failed"
        and event.payload["error_code"] == "slock_session_timeout"
        for frame in writer.replay()
        for event in frame.events
    )
    service.close()
    writer.close()


def test_team_step_deadline_preserves_microsecond_budget(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _Backend()
    base = datetime(2026, 7, 14, 0, 0, 0, 123456, tzinfo=UTC)
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=0.0005,
        poll_seconds=0,
        clock=lambda: base,
    )
    state = TeamRunState(
        run_id="teamrun_fractional",
        tenant_key="tenant_1",
        message_id="om_fractional",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task_digest="0" * 64,
    )

    result = service._run_step(
        state,
        step_id="analysis",
        depth=1,
        target=TeamTarget("agt_lead", "Lead"),
        instruction="fractional budget",
    )

    assert result.status == "completed"
    assert backend.deadlines == ["2026-07-14T00:00:00.123956Z"]
    service.close()
    writer.close()


def test_team_timeout_cancels_before_retry_admission(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _TimeoutCancelBackend()
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=0.001,
        poll_seconds=0.002,
    )
    state = TeamRunState(
        run_id="teamrun_timeout_cancel",
        tenant_key="tenant_1",
        message_id="om_timeout",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task_digest="0" * 64,
    )

    result = service._run_step(
        state,
        step_id="analysis",
        depth=1,
        target=TeamTarget("agt_lead", "Lead", "coder"),
        instruction="timeout and cancel",
    )

    assert backend.cancellations == [
        ("acc_analysis", "teamrun_timeout_cancel", "analysis")
    ]
    assert result.status == "canceled"
    service.close()
    writer.close()


def test_team_cancel_pending_does_not_admit_retry(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _TimeoutCancelBackend()
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=0.001,
        poll_seconds=0.002,
    )

    state = service.start_task(
        tenant_key="tenant_1",
        message_id="om_cancel_pending",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="do not overlap attempts",
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        current = service.get_run(state.run_id)
        if current is not None and current.status == "action_required":
            break
        time.sleep(0.01)

    assert len(backend.submissions) == 1
    service.close()
    writer.close()


def test_close_fence_prevents_retry_submission(tmp_path) -> None:
    writer = make_writer(tmp_path)
    backend = _CloseBarrierBackend()
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="legacy_pipeline",
        attempt_timeout_seconds=30,
        poll_seconds=0,
    )
    state = service.start_task(
        tenant_key="tenant_1",
        message_id="om_close",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="close during result",
    )
    assert backend.result_entered.wait(1)

    closer = threading.Thread(target=service.close)
    closer.start()
    assert service._stop.wait(1)  # noqa: SLF001
    backend.result_release.set()
    closer.join(2)

    assert not closer.is_alive()
    assert len(backend.submissions) == 1
    assert service.get_run(state.run_id).status in {"stopping", "action_required"}
    events = [
        event.event_type
        for frame in writer.replay()
        for event in frame.events
        if event.aggregate_id.startswith(state.run_id)
    ]
    stopping_index = events.index("team.run.stopping")
    assert "team.step.prepared" not in events[stopping_index + 1 :]
    writer.close()


def test_team_run_terminal_cannot_regress_to_stopping(tmp_path) -> None:
    from src.autonomous.journal.frame import JournalEvent
    from src.autonomous.team.service import TeamServiceError

    writer = make_writer(tmp_path)
    service = EmployeeTeamService(
        writer=writer, backend=_Backend(), runtime_mode="legacy_pipeline"
    )
    service._executor.shutdown(wait=True)  # noqa: SLF001
    service._closed = True  # noqa: SLF001
    service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="team.run.created",
            aggregate_id="teamrun_monotonic",
            payload={
                "tenant_key": "tenant_1",
                "message_id": "om_terminal",
                "chat_id": "oc_team",
                "requester_principal_id": "ou_user",
                "task_digest": "0" * 64,
                "max_handoffs": 8,
                "max_depth": 4,
                "max_fanout": 4,
            },
        )
    )
    service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="team.run.completed",
            aggregate_id="teamrun_monotonic",
            payload={"result_digest": "1" * 64, "history_record_id": "hist_done"},
        )
    )

    with pytest.raises(TeamServiceError):
        service._commit(  # noqa: SLF001
            JournalEvent(
                event_type="team.run.stopping",
                aggregate_id="teamrun_monotonic",
                payload={"reason_code": "team_service_stopping"},
            )
        )
    assert service.get_run("teamrun_monotonic").status == "completed"
    writer.close()


def test_team_run_reducer_rejects_non_exact_stopping_payload(tmp_path) -> None:
    from src.autonomous.journal.frame import JournalEvent
    from src.autonomous.team.service import TeamServiceError

    writer = make_writer(tmp_path)
    service = EmployeeTeamService(
        writer=writer, backend=_Backend(), runtime_mode="legacy_pipeline"
    )
    service._executor.shutdown(wait=True)  # noqa: SLF001
    service._closed = True  # noqa: SLF001
    service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="team.run.created",
            aggregate_id="teamrun_exact",
            payload={
                "tenant_key": "tenant_1",
                "message_id": "om_exact",
                "chat_id": "oc_team",
                "requester_principal_id": "ou_user",
                "task_digest": "0" * 64,
                "max_handoffs": 8,
                "max_depth": 4,
                "max_fanout": 4,
            },
        )
    )

    with pytest.raises(TeamServiceError):
        service._commit(  # noqa: SLF001
            JournalEvent(
                event_type="team.run.stopping",
                aggregate_id="teamrun_exact",
                payload={"reason_code": "team_service_stopping", "extra": True},
            )
        )
    writer.close()


def test_close_between_progress_check_and_initial_step_writes_nothing(tmp_path, monkeypatch) -> None:
    writer = make_writer(tmp_path)
    service = EmployeeTeamService(
        writer=writer, backend=_Backend(), runtime_mode="legacy_pipeline"
    )
    state = TeamRunState(
        run_id="teamrun_initial_fence",
        tenant_key="tenant_1",
        message_id="om_initial",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task_digest="0" * 64,
    )
    from src.autonomous.journal.frame import JournalEvent

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
    original = service._can_progress  # noqa: SLF001
    checked = False

    def close_after_check(current):
        nonlocal checked
        allowed = original(current)
        if not checked:
            checked = True
            service.close()
        return allowed

    monkeypatch.setattr(service, "_can_progress", close_after_check)
    service._run_step(  # noqa: SLF001
        state,
        step_id="analysis",
        depth=1,
        target=TeamTarget("agt_lead", "Lead"),
        instruction="fenced",
    )

    events = [
        event.event_type
        for frame in writer.replay()
        for event in frame.events
        if event.aggregate_id.startswith(state.run_id)
    ]
    stopping = events.index("team.run.stopping")
    assert not any(
        event.startswith(("team.step.", "team.effect."))
        for event in events[stopping + 1 :]
    )
    writer.close()


@pytest.mark.parametrize("active, expected", [(True, "completed"), (False, "action_required")])
def test_close_cannot_fence_after_notify_has_started(tmp_path, active, expected) -> None:
    writer = make_writer(tmp_path)
    backend = _BlockingNotifyBackend(active=active)
    service = EmployeeTeamService(
        writer=writer, backend=backend, runtime_mode="legacy_pipeline"
    )
    if active:
        state = service.start_task(
            tenant_key="tenant_1",
            message_id="om_notify_true",
            chat_id="oc_team",
            requester_principal_id="ou_user",
            task="notify fence",
        )
    else:
        from src.autonomous.journal.frame import JournalEvent

        state = TeamRunState(
            run_id="teamrun_failure_notify",
            tenant_key="tenant_1",
            message_id="om_notify_false",
            chat_id="oc_team",
            requester_principal_id="ou_user",
            task_digest="0" * 64,
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
        service._executor.submit(  # noqa: SLF001
            service._action_required,  # noqa: SLF001
            state,
            "analysis_failed",
        )
    assert backend.notify_entered.wait(1)
    closer = threading.Thread(target=service.close)
    closer.start()
    time.sleep(0.05)
    backend.notify_release.set()
    closer.join(2)

    assert service.get_run(state.run_id).status == expected
    assert not any(
        event.event_type == "team.run.stopping"
        and event.aggregate_id == state.run_id
        for frame in writer.replay()
        for event in frame.events
    )
    writer.close()
