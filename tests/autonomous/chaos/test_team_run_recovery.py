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
    assert projection.runs[run.run_id].phase is TeamRunPhase.REVISING
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
    assert projection.runs[run.run_id].phase is TeamRunPhase.REVISING
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


def test_restart_from_final_notify_action_required_starts_new_attempt(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    failing_backend = _RetryableNotifyBackend(fail=True)
    first = _actor(writer, blobs, failing_backend)
    original_phase = first._phase  # noqa: SLF001

    def crash_before_block_phase(run, phase, **kwargs):
        if phase is TeamRunPhase.BLOCKED:
            raise SystemExit("simulated process death before run blocking")
        return original_phase(run, phase, **kwargs)

    first._phase = crash_before_block_phase  # type: ignore[method-assign] # noqa: SLF001
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_action_required",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="通知 action-required 边界恢复",
    )
    first.drain()
    projection = first.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.REVISING
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
    assert recovered_backend.idempotency_keys == [
        failing_backend.idempotency_keys[0]
    ]
    assert len(recovered_backend.notifications) == 1
    second.close()
    blobs.close()
    writer.close()


def test_recover_repairs_completed_run_with_uncommitted_final_notification(
    tmp_path,
) -> None:
    writer, blobs = make_team_storage(tmp_path)
    failing_backend = _RetryableNotifyBackend(fail=True)
    first = _actor(writer, blobs, failing_backend)
    original_phase = first._phase  # noqa: SLF001

    def crash_before_block_phase(run, phase, **kwargs):
        if phase is TeamRunPhase.BLOCKED:
            raise SystemExit("simulated legacy completion window")
        return original_phase(run, phase, **kwargs)

    first._phase = crash_before_block_phase  # type: ignore[method-assign] # noqa: SLF001
    run = first.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_legacy_completed",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="修复历史 completed 通知",
    )
    first.drain()
    projection = first.projection()
    final_assignment = next(
        assignment
        for assignment in projection.assignments.values()
        if assignment.run_id == run.run_id and assignment.role == "finalize"
    )
    first._commit(  # noqa: SLF001
        JournalEvent(
            "team.v2.run.completed",
            run.run_id,
            {
                "run_id": run.run_id,
                "result_ref": final_assignment.contribution_ref.to_dict(),
                "done_checks": {
                    criterion: True for criterion in run.done_criteria
                },
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
    assert recovered_backend.idempotency_keys == [
        failing_backend.idempotency_keys[0]
    ]
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
