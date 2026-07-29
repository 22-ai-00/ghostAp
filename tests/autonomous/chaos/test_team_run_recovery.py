from __future__ import annotations

from src.autonomous.journal.frame import JournalEvent
from src.autonomous.team import (
    CoordinatorAction,
    CoordinatorDecision,
    TeamAttemptResult,
    TeamCoordinatorActor,
    TeamRunPhase,
)
from tests.autonomous.team_helpers import ImmediateTeamBackend, make_team_storage


def _actor(writer, blobs, backend):
    return TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
    )


def _assert_no_lost_instruction(writer) -> None:
    assert all(
        event.payload.get("error_code") != "restart_instruction_unavailable"
        for frame in writer.replay()
        for event in frame.events
    )


def test_restart_from_planning_replays_encrypted_task(tmp_path) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = ImmediateTeamBackend()
    first = _actor(writer, blobs, backend)
    run_id = "teamrun2_planning"
    task_ref = first._publish_json(  # noqa: SLF001
        {"task": "恢复规划", "goal": "恢复规划", "done_criteria": ["review"]},
        tenant_key="tenant_1",
        run_id=run_id,
        kind="team_task",
    )
    first._commit(  # noqa: SLF001
        JournalEvent(
            "team.v2.run.created",
            run_id,
            {
                "tenant_key": "tenant_1",
                "chat_id": "oc_team",
                "project_id": "",
                "message_id": "om_planning",
                "requester_principal_id": "ou_user",
                "task_ref": task_ref.to_dict(),
                "goal": "encrypted-team-task:planning",
                "done_criteria": ["review"],
                "coordinator_session_key": "session",
                "coordinator_tool": "coco",
            },
        )
    )
    run = first.projection().runs[run_id]
    first._phase(run, TeamRunPhase.PLANNING, turn=1)  # noqa: SLF001
    first.close()

    second = _actor(writer, blobs, backend)
    assert second.recover() == 1
    second.drain()
    assert second.projection().runs[run_id].phase is TeamRunPhase.COMPLETED
    _assert_no_lost_instruction(writer)
    second.close()
    blobs.close()
    writer.close()


def test_restart_from_running_assignment_reuses_acceptance(tmp_path) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = ImmediateTeamBackend()
    backend.results["acc_recovery"] = TeamAttemptResult(
        "completed", "recovered contribution", "hist_recovery"
    )
    first = _actor(writer, blobs, backend)
    run_id = "teamrun2_running"
    task_ref = first._publish_json(  # noqa: SLF001
        {"task": "恢复运行", "goal": "恢复运行", "done_criteria": ["review"]},
        tenant_key="tenant_1",
        run_id=run_id,
        kind="team_task",
    )
    first._commit(  # noqa: SLF001
        JournalEvent(
            "team.v2.run.created",
            run_id,
            {
                "tenant_key": "tenant_1",
                "chat_id": "oc_team",
                "project_id": "",
                "message_id": "om_running",
                "requester_principal_id": "ou_user",
                "task_ref": task_ref.to_dict(),
                "goal": "encrypted-team-task:running",
                "done_criteria": ["review"],
                "coordinator_session_key": "session",
                "coordinator_tool": "coco",
            },
        )
    )
    run = first.projection().runs[run_id]
    first._phase(run, TeamRunPhase.PLANNING, turn=1)  # noqa: SLF001
    run = first.projection().runs[run_id]
    decision = CoordinatorDecision(
        CoordinatorAction.ASSIGN,
        ("agt_coder",),
        role="execute",
        instruction="恢复这条指令",
    )
    assignment_id = first._create_assignment(  # noqa: SLF001
        run, decision, ordinal=1, agent_id="agt_coder"
    )
    run = first.projection().runs[run_id]
    first._phase(run, TeamRunPhase.DISPATCHING, turn=2)  # noqa: SLF001
    assert first.claim(assignment_id, "agt_coder")
    first._effect(assignment_id, "employee_dispatch", "prepared")  # noqa: SLF001
    first._effect(assignment_id, "employee_dispatch", "executing")  # noqa: SLF001
    first._commit(  # noqa: SLF001
        JournalEvent(
            "team.v2.assignment.submitted",
            assignment_id,
            {"run_id": run_id, "acceptance_id": "acc_recovery"},
        )
    )
    first.close()

    second = _actor(writer, blobs, backend)
    second.recover()
    second.drain()
    assert second.projection().runs[run_id].phase is TeamRunPhase.COMPLETED
    assert all(item[0] != "1" for item in backend.submissions)
    _assert_no_lost_instruction(writer)
    second.close()
    blobs.close()
    writer.close()


def test_restart_after_contribution_commit_continues_to_review(tmp_path) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = ImmediateTeamBackend()
    first = _actor(writer, blobs, backend)
    original_phase = first._phase  # noqa: SLF001

    def crash_before_review(run, phase, **kwargs):
        if phase is TeamRunPhase.REVIEWING:
            raise SystemExit("simulated crash")
        return original_phase(run, phase, **kwargs)

    first._phase = crash_before_review  # type: ignore[method-assign] # noqa: SLF001
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_contribution",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="贡献提交后恢复",
    )
    first.drain()
    assert first.projection().runs[run.run_id].phase is TeamRunPhase.DISPATCHING
    first.close()

    second = _actor(writer, blobs, backend)
    second.recover()
    second.drain()
    assert second.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    _assert_no_lost_instruction(writer)
    second.close()
    blobs.close()
    writer.close()


class _CrashNotifyBackend(ImmediateTeamBackend):
    def notify(self, message_id, chat_id, result):
        raise SystemExit("simulated process death after notify executing")


class _RetryableNotifyBackend(ImmediateTeamBackend):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.idempotency_keys: list[str] = []

    def notify(
        self,
        message_id,
        chat_id,
        result,
        *,
        idempotency_key,
        **_scope,
    ):
        self.idempotency_keys.append(idempotency_key)
        if self.fail:
            raise RuntimeError("simulated final notification failure")
        self.notifications.append((message_id, chat_id, result))


class _FailOnceNotifyBackend(_RetryableNotifyBackend):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    def notify(self, *args, idempotency_key, **kwargs):
        self.idempotency_keys.append(idempotency_key)
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("simulated transient final delivery failure")
        message_id, chat_id, result = args
        self.notifications.append((message_id, chat_id, result))


class _FinalFailingBlockSuccessBackend(_RetryableNotifyBackend):
    def notify(self, *args, idempotency_key, **kwargs):
        message_id, chat_id, result = args
        if result.startswith("⚠️"):
            self.notifications.append((message_id, chat_id, result))
            return
        self.idempotency_keys.append(idempotency_key)
        raise RuntimeError("simulated persistent final delivery failure")


class _PersistentBlockNotificationFailureBackend(ImmediateTeamBackend):
    def __init__(self) -> None:
        super().__init__()
        self.targets = ()
        self.notification_attempts: list[str] = []

    def notify(
        self,
        _message_id,
        _chat_id,
        _result,
        *,
        idempotency_key,
        **_scope,
    ):
        self.notification_attempts.append(idempotency_key)
        raise RuntimeError("simulated persistent blocked notification failure")


def _seed_final_notification_action_required(
    writer,
    blobs,
    *,
    message_id: str,
):
    actor = _actor(writer, blobs, ImmediateTeamBackend())
    original_effect = actor._effect  # noqa: SLF001

    def crash_before_notify_execute(aggregate, effect_type, state):
        if (
            aggregate.endswith(":notify")
            and effect_type == "notify"
            and state == "executing"
        ):
            raise SystemExit("simulated process death after notify prepared")
        return original_effect(aggregate, effect_type, state)

    actor._effect = crash_before_notify_execute  # type: ignore[method-assign] # noqa: SLF001
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id=message_id,
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="构造历史 final notification 状态",
    )
    actor.drain()
    original_effect(f"{run.run_id}:notify", "notify", "action_required")
    return actor, run


def test_initial_block_notification_commit_precedes_only_terminal_transition(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    first_backend = ImmediateTeamBackend()
    first_backend.targets = ()
    first = _actor(writer, blobs, first_backend)
    original_phase = first._phase  # noqa: SLF001

    def crash_before_blocked(run, phase, **kwargs):
        if phase is TeamRunPhase.BLOCKED:
            raise SystemExit("simulated crash after blocked notification")
        return original_phase(run, phase, **kwargs)

    first._phase = crash_before_blocked  # type: ignore[method-assign] # noqa: SLF001
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_initial_block_crash",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="首次失败通知崩溃窗",
    )
    first.drain()

    projection = first.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKING
    assert (
        projection.effects[
            (f"{run.run_id}:blocked-notify:1", "notify")
        ]
        == "committed"
    )
    assert len(first_backend.notifications) == 1
    first.close()

    recovered_backend = ImmediateTeamBackend()
    second = _actor(writer, blobs, recovered_backend)
    assert second.recover() == 1
    second.drain()

    projection = second.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKED
    assert recovered_backend.notifications == []
    assert second.recover() == 0
    second.close()
    blobs.close()
    writer.close()


def test_block_notification_failures_are_bounded_and_abandoned(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = _PersistentBlockNotificationFailureBackend()
    actor = _actor(writer, blobs, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_block_notify_persistent",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="失败通知持续不可用",
    )
    actor.drain()
    assert actor.projection().runs[run.run_id].phase is TeamRunPhase.BLOCKING
    assert actor.recover() == 1
    actor.drain()
    assert actor.projection().runs[run.run_id].phase is TeamRunPhase.BLOCKING
    assert actor.recover() == 1
    actor.drain()

    projection = actor.projection()
    attempts = [
        (aggregate, state)
        for (aggregate, effect_type), state in projection.effects.items()
        if effect_type == "notify"
        and aggregate.startswith(f"{run.run_id}:blocked-notify:")
    ]
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKED
    assert len(attempts) == 3
    assert sum(state == "abandoned" for _aggregate, state in attempts) == 1
    assert len(backend.notification_attempts) == 3
    assert len(set(backend.notification_attempts)) == 1
    assert actor.recover() == 0
    actor.close()
    blobs.close()
    writer.close()


def test_restart_from_final_notify_executing_converges_without_duplicate(tmp_path) -> None:
    writer, blobs = make_team_storage(tmp_path)
    crashing = _CrashNotifyBackend()
    first = _actor(writer, blobs, crashing)
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_notify",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="通知边界恢复",
    )
    first.drain()
    projection = first.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.FINALIZING
    assert projection.effects[(f"{run.run_id}:notify", "notify")] == "executing"
    first.close()

    recovered_backend = ImmediateTeamBackend()
    second = _actor(writer, blobs, recovered_backend)
    second.recover()
    second.drain()
    assert second.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert len(recovered_backend.notifications) == 1
    _assert_no_lost_instruction(writer)
    second.close()
    blobs.close()
    writer.close()


def test_transient_final_delivery_failure_retries_without_blocking_run(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = _FailOnceNotifyBackend()
    actor = _actor(writer, blobs, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_transient",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="瞬时通知失败后继续交付",
    )
    actor.drain()

    projection = actor.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert len(backend.idempotency_keys) == 2
    assert len(set(backend.idempotency_keys)) == 1
    assert len(backend.notifications) == 1
    assert not backend.notifications[0][2].startswith("⚠️")
    actor.close()
    blobs.close()
    writer.close()


def test_persistent_final_delivery_failure_has_bounded_attempts(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    backend = _FinalFailingBlockSuccessBackend()
    actor = _actor(writer, blobs, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_persistent",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="持续通知失败后有界停车",
    )
    actor.drain()

    projection = actor.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKED
    final_attempts = [
        aggregate
        for (aggregate, effect_type), state in projection.effects.items()
        if effect_type == "notify"
        and (
            aggregate == f"{run.run_id}:notify"
            or aggregate.startswith(f"{run.run_id}:final-notify:")
        )
        and state == "action_required"
    ]
    assert len(final_attempts) == 3
    assert len(backend.idempotency_keys) == 3
    assert len(set(backend.idempotency_keys)) == 1
    assert actor.recover() == 0
    actor.close()
    blobs.close()
    writer.close()


def test_restart_from_final_notify_prepared_resumes_before_completion(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    first = _actor(writer, blobs, ImmediateTeamBackend())
    original_effect = first._effect  # noqa: SLF001

    def crash_before_notify_execute(aggregate, effect_type, state):
        if (
            aggregate.endswith(":notify")
            and effect_type == "notify"
            and state == "executing"
        ):
            raise SystemExit("simulated process death after notify prepared")
        return original_effect(aggregate, effect_type, state)

    first._effect = crash_before_notify_execute  # type: ignore[method-assign] # noqa: SLF001
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_prepared",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="通知 prepared 边界恢复",
    )
    first.drain()
    projection = first.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.FINALIZING
    assert projection.effects[(f"{run.run_id}:notify", "notify")] == "prepared"
    first.close()

    recovered_backend = _RetryableNotifyBackend()
    second = _actor(writer, blobs, recovered_backend)
    assert second.recover() == 1
    second.drain()

    projection = second.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert projection.effects[(f"{run.run_id}:notify", "notify")] == "committed"
    assert len(recovered_backend.notifications) == 1
    second.close()
    blobs.close()
    writer.close()


def test_completed_notification_repair_reopens_before_dispatch(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    first, run = _seed_final_notification_action_required(
        writer,
        blobs,
        message_id="om_notify_repair_crash",
    )
    seeded = first.projection().runs[run.run_id]
    first._commit(  # noqa: SLF001
        JournalEvent(
            "team.v2.run.completed",
            run.run_id,
            {
                "run_id": run.run_id,
                "result_ref": seeded.final_result_ref.to_dict(),
                "done_checks": dict(seeded.final_done_checks),
            },
        )
    )
    first.close()

    crashing = _actor(writer, blobs, _CrashNotifyBackend())
    assert crashing.recover() == 1
    crashing.drain()
    projection = crashing.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.FINALIZING
    assert any(
        state == "executing"
        for (aggregate, effect_type), state in projection.effects.items()
        if aggregate.startswith(f"{run.run_id}:final-notify:")
        and effect_type == "notify"
    )
    crashing.close()

    recovered_backend = _RetryableNotifyBackend()
    recovered = _actor(writer, blobs, recovered_backend)
    assert recovered.recover() == 1
    recovered.drain()
    assert (
        recovered.projection().runs[run.run_id].phase
        is TeamRunPhase.COMPLETED
    )
    assert len(recovered_backend.notifications) == 1
    recovered.close()
    blobs.close()
    writer.close()


def test_restart_from_final_notify_action_required_starts_new_attempt(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    first, run = _seed_final_notification_action_required(
        writer,
        blobs,
        message_id="om_notify_action_required",
    )
    projection = first.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.FINALIZING
    assert (
        projection.effects[(f"{run.run_id}:notify", "notify")]
        == "action_required"
    )
    first.close()

    recovered_backend = _RetryableNotifyBackend()
    second = _actor(writer, blobs, recovered_backend)
    assert second.recover() == 1
    second.drain()

    projection = second.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert (
        projection.effects[(f"{run.run_id}:final-notify:1", "notify")]
        == "committed"
    )
    assert len(recovered_backend.idempotency_keys) == 1
    assert len(recovered_backend.notifications) == 1
    second.close()
    blobs.close()
    writer.close()


def test_recover_repairs_completed_run_with_uncommitted_final_notification(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    first, run = _seed_final_notification_action_required(
        writer,
        blobs,
        message_id="om_notify_legacy_completed",
    )
    seeded = first.projection().runs[run.run_id]
    first._commit(  # noqa: SLF001
        JournalEvent(
            "team.v2.run.completed",
            run.run_id,
            {
                "run_id": run.run_id,
                "result_ref": seeded.final_result_ref.to_dict(),
                "done_checks": dict(seeded.final_done_checks),
            },
        )
    )
    assert first.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    first.close()

    recovered_backend = _RetryableNotifyBackend()
    second = _actor(writer, blobs, recovered_backend)
    assert second.recover() == 1
    second.drain()

    projection = second.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert (
        projection.effects[(f"{run.run_id}:final-notify:1", "notify")]
        == "committed"
    )
    assert len(recovered_backend.idempotency_keys) == 1
    assert len(recovered_backend.notifications) == 1
    assert second.recover() == 0
    second.close()
    blobs.close()
    writer.close()


def test_recover_does_not_repeat_committed_final_notification(tmp_path) -> None:
    writer, blobs = make_team_storage(tmp_path)
    first_backend = _RetryableNotifyBackend()
    first = _actor(writer, blobs, first_backend)
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_committed",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="已提交通知不得重发",
    )
    first.drain()
    assert first.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert len(first_backend.notifications) == 1
    first.close()

    recovered_backend = _RetryableNotifyBackend()
    second = _actor(writer, blobs, recovered_backend)
    assert second.recover() == 0
    second.drain()
    assert recovered_backend.notifications == []
    second.close()
    blobs.close()
    writer.close()


def test_restart_after_final_notify_commit_completes_without_resend(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    first_backend = _RetryableNotifyBackend()
    first = _actor(writer, blobs, first_backend)
    original_commit = first._commit  # noqa: SLF001

    def crash_before_run_completion(event):
        if event.event_type == "team.v2.run.completed":
            raise SystemExit("simulated crash after final notify commit")
        return original_commit(event)

    first._commit = crash_before_run_completion  # type: ignore[method-assign] # noqa: SLF001
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_committed_before_completion",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="通知提交后恢复完成",
    )
    first.drain()
    projection = first.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.FINALIZING
    assert projection.effects[(f"{run.run_id}:notify", "notify")] == "committed"
    assert len(first_backend.notifications) == 1
    first.close()

    recovered_backend = _RetryableNotifyBackend()
    second = _actor(writer, blobs, recovered_backend)
    assert second.recover() == 1
    second.drain()
    assert second.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert recovered_backend.notifications == []
    second.close()
    blobs.close()
    writer.close()
