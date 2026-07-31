from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from src.autonomous.team import (
    CoordinatorAction,
    CoordinatorDecision,
    EmployeeTeamService,
    SessionCoordinatorDecisionProvider,
    TeamAdmissionError,
    TeamAttemptResult,
    TeamCoordinatorActor,
    TeamRunPhase,
)
from tests.autonomous.team_helpers import ImmediateTeamBackend, make_team_storage


def _actor(tmp_path, backend=None):
    writer, blobs = make_team_storage(tmp_path)
    actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend or ImmediateTeamBackend(),
        poll_seconds=0.001,
    )
    return writer, blobs, actor


def test_coordinator_persists_encrypted_task_and_completes_dynamic_run(tmp_path) -> None:
    backend = ImmediateTeamBackend()
    writer, blobs, actor = _actor(tmp_path, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_team",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="请实现 Python 功能并完成安全评审",
    )
    actor.drain()
    final = actor.projection().runs[run.run_id]

    assert final.phase is TeamRunPhase.COMPLETED
    assert [item[1] for item in backend.submissions] == [
        "agt_coder",
        "agt_reviewer",
        "agt_coder",
    ]
    assert len(backend.notifications) == 1
    assert "请实现 Python 功能" not in str(
        [event.payload for frame in writer.replay() for event in frame.events]
    )
    actor.close()
    blobs.close()
    writer.close()


def test_model_coordinator_decides_each_team_round(tmp_path) -> None:
    class _RoundDecisionProvider:
        def __init__(self) -> None:
            self.phases = []

        def __call__(self, run, _targets, task):
            self.phases.append((run.phase, task))
            return CoordinatorDecision(
                CoordinatorAction.ASSIGN,
                ("agt_coder",),
                role="execute",
                instruction="implement with evidence",
            )

        def decide_next(self, run, _targets, context, allowed_actions):
            self.phases.append((run.phase, context))
            if run.phase is TeamRunPhase.DISPATCHING:
                assert CoordinatorAction.REVIEW in allowed_actions
                assert tuple(item.agent_id for item in _targets) == ("agt_reviewer",)
                return CoordinatorDecision(
                    CoordinatorAction.REVIEW,
                    ("agt_reviewer",),
                    role="review",
                    instruction="independently review",
                    depends_on=(run.assignment_ids[0],),
                )
            if run.phase is TeamRunPhase.REVIEWING:
                assert CoordinatorAction.REVISE in allowed_actions
                return CoordinatorDecision(
                    CoordinatorAction.REVISE,
                    ("agt_coder",),
                    role="finalize",
                    instruction="revise from review",
                    depends_on=run.assignment_ids,
                )
            assert run.phase is TeamRunPhase.REVISING
            assert CoordinatorAction.COMPLETE in allowed_actions
            return CoordinatorDecision(
                CoordinatorAction.COMPLETE,
                done_checks={
                    "deliverable_non_empty": True,
                    "review_completed": True,
                },
            )

    provider = _RoundDecisionProvider()
    backend = ImmediateTeamBackend()
    writer, blobs = make_team_storage(tmp_path)
    actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
        decision_provider=provider,
    )
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_rounds",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="implement and review",
    )
    actor.drain()

    assert actor.projection().runs[run.run_id].phase is TeamRunPhase.COMPLETED
    assert [phase for phase, _context in provider.phases] == [
        TeamRunPhase.PLANNING,
        TeamRunPhase.DISPATCHING,
        TeamRunPhase.REVIEWING,
        TeamRunPhase.REVISING,
    ]
    actor.close()
    blobs.close()
    writer.close()


def test_explicit_mention_wins_and_assignment_claim_is_single_winner(tmp_path) -> None:
    backend = ImmediateTeamBackend()
    writer, blobs, actor = _actor(tmp_path, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_mention",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="@agt_reviewer 请先负责这个任务",
    )
    actor.drain()
    first = actor.projection().assignments[f"{run.run_id}:assignment:1"]
    assert first.agent_id == "agt_reviewer"

    winners = []
    # Completed assignments are fenced just as strictly as concurrent claims.
    threads = [
        threading.Thread(
            target=lambda: winners.append(actor.claim(first.assignment_id, first.agent_id))
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert winners == [False, False]
    actor.close()
    blobs.close()
    writer.close()


def test_coordinator_decision_rejects_bounds_and_forged_completion() -> None:
    with pytest.raises(ValueError, match="fanout"):
        CoordinatorDecision(
            CoordinatorAction.ASSIGN,
            tuple(f"agt_{index}" for index in range(5)),
            role="execute",
            instruction="work",
        )
    with pytest.raises(ValueError, match="forge"):
        CoordinatorDecision(
            CoordinatorAction.COMPLETE,
            done_checks={"review": False},
        )
    with pytest.raises(ValueError, match="agent ID"):
        CoordinatorDecision(
            CoordinatorAction.ASSIGN,
            ("not-an-agent",),
            role="execute",
            instruction="work",
        )


def test_configured_coordinator_session_is_reused_and_strictly_parsed(
    monkeypatch,
) -> None:
    calls = []

    class _Session:
        def set_tool_filter(self, _tool_filter):
            return None

        def send_prompt(self, prompt, timeout):
            calls.append((prompt, timeout))
            return SimpleNamespace(
                text=(
                    '{"action":"assign","agent_ids":["agt_coder"],'
                    '"role":"execute","instruction":"do it",'
                    '"depends_on":[],"done_checks":{},"reason_code":""}'
                )
            )

    created = []
    monkeypatch.setattr(
        "src.agent_session.create_auxiliary_session",
        lambda **kwargs: created.append(kwargs) or _Session(),
    )
    monkeypatch.setattr("src.agent_session.close_session_safely", lambda session: None)
    provider = SessionCoordinatorDecisionProvider(
        tool="codex",
        model="gpt-test",
        cwd_resolver=lambda _run: "/project",
    )
    run = SimpleNamespace(coordinator_session_key="session-key")
    targets = (
        SimpleNamespace(
            agent_id="agt_coder",
            role="coder",
            capabilities=("python",),
            runtime_status="ready_warm",
            mailbox_load=0,
        ),
    )
    first = provider(run, targets, "task")
    second = provider(run, targets, "task 2")
    assert first.agent_ids == second.agent_ids == ("agt_coder",)
    assert len(created) == 1
    assert created[0]["agent_type"] == "codex"
    assert created[0]["model_name"] == "gpt-test"
    assert len(calls) == 2
    provider.close()


def test_team_service_coordinator_mode_does_not_enter_legacy_pipeline(tmp_path) -> None:
    backend = ImmediateTeamBackend()
    writer, blobs = make_team_storage(tmp_path)
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="coordinator",
        blob_store=blobs,
        active_key_id="team-key",
        poll_seconds=0.001,
    )
    accepted = service.start_task(
        tenant_key="tenant_1",
        message_id="om_facade",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="实现并评审",
    )
    service._coordinator.drain()  # noqa: SLF001
    final = service.get_run(accepted.run_id)
    assert final is not None and final.status == "completed"
    assert all(step_id.isdigit() for step_id, *_rest in backend.submissions)
    service.close()
    blobs.close()
    writer.close()


def test_team_service_coordinator_rejects_empty_roster_before_durable_admission(
    tmp_path,
) -> None:
    backend = ImmediateTeamBackend()
    backend.targets = ()
    writer, blobs = make_team_storage(tmp_path)
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="coordinator",
        blob_store=blobs,
        active_key_id="team-key",
        poll_seconds=0.001,
    )

    with pytest.raises(TeamAdmissionError) as raised:
        service.start_task(
            tenant_key="tenant_1",
            message_id="om_no_coordinator_employee",
            chat_id="oc_team",
            requester_principal_id="ou_user",
            task="介绍一下团队能力",
        )

    assert raised.value.error_code == "no_active_team_employee"
    assert tuple(writer.replay()) == ()
    assert blobs.iter_blob_ids() == ()
    service.close()
    blobs.close()
    writer.close()


def test_team_service_coordinator_rejects_replayed_terminal_run(tmp_path) -> None:
    backend = ImmediateTeamBackend()
    writer, blobs = make_team_storage(tmp_path)
    service = EmployeeTeamService(
        writer=writer,
        backend=backend,
        runtime_mode="coordinator",
        blob_store=blobs,
        active_key_id="team-key",
        poll_seconds=0.001,
    )
    coordinates = {
        "tenant_key": "tenant_1",
        "message_id": "om_terminal_replay",
        "chat_id": "oc_team",
        "requester_principal_id": "ou_user",
        "task": "实现并评审",
    }

    first = service.start_task(**coordinates)
    service._coordinator.drain()  # noqa: SLF001
    assert service.get_run(first.run_id).status == "completed"
    event_count = sum(len(frame.events) for frame in writer.replay())
    notification_count = len(backend.notifications)

    with pytest.raises(TeamAdmissionError) as raised:
        service.start_task(**coordinates)

    assert raised.value.error_code == "team_run_completed"
    assert sum(len(frame.events) for frame in writer.replay()) == event_count
    assert len(backend.notifications) == notification_count
    service.close()
    blobs.close()
    writer.close()


def test_coordinator_notifies_requester_once_when_execution_blocks(tmp_path) -> None:
    class _FailingBackend(ImmediateTeamBackend):
        def submit(self, **kwargs):
            acceptance_id = super().submit(**kwargs)
            self.results[acceptance_id] = TeamAttemptResult(
                "action_required",
                error_code="employee_session_failed",
                retry_allowed=False,
            )
            return acceptance_id

    backend = _FailingBackend()
    writer, blobs, actor = _actor(tmp_path, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_blocked",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="执行后给出评审结果",
    )
    actor.drain()

    final = actor.projection().runs[run.run_id]
    assert final.phase is TeamRunPhase.BLOCKED
    assert final.error_code == "employee_session_failed"
    assert len(backend.notifications) == 1
    message_id, chat_id, content = backend.notifications[0]
    assert (message_id, chat_id) == ("om_blocked", "oc_team")
    assert "团队任务未完成" in content
    assert "employee_session_failed" in content
    assert actor.recover() == 0
    assert len(backend.notifications) == 1

    actor.close()
    blobs.close()
    writer.close()


def test_coordinator_notification_preserves_durable_recipient_scope(tmp_path) -> None:
    class _ScopedBackend(ImmediateTeamBackend):
        def __init__(self):
            super().__init__()
            self.recipient_scopes = []

        def submit(self, **kwargs):
            acceptance_id = super().submit(**kwargs)
            self.results[acceptance_id] = TeamAttemptResult(
                "action_required",
                error_code="employee_session_failed",
                retry_allowed=False,
            )
            return acceptance_id

        def notify(
            self,
            message_id,
            chat_id,
            content,
            *,
            idempotency_key,
            tenant_key,
            requester_principal_id,
        ):
            self.recipient_scopes.append(
                (
                    message_id,
                    chat_id,
                    content,
                    idempotency_key,
                    tenant_key,
                    requester_principal_id,
                )
            )

    backend = _ScopedBackend()
    writer, blobs, actor = _actor(tmp_path, backend)
    actor.start_task(
        tenant_key="tenant_1",
        message_id="om_scoped",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="执行并回报",
    )
    actor.drain()

    assert len(backend.recipient_scopes) == 1
    message_id, chat_id, content, key, tenant_key, requester = (
        backend.recipient_scopes[0]
    )
    assert (message_id, chat_id) == ("om_scoped", "oc_team")
    assert content.startswith("⚠️ 团队任务未完成")
    assert key
    assert (tenant_key, requester) == ("tenant_1", "ou_user")

    actor.close()
    blobs.close()
    writer.close()


def test_unknown_model_block_reason_is_persisted_and_shown_as_generic(tmp_path) -> None:
    unknown_reason = "dump_private_prompt"

    class _BlockingDecisionProvider:
        def __call__(self, _run, _targets, task):
            return CoordinatorDecision(
                CoordinatorAction.ASSIGN,
                ("agt_coder",),
                role="execute",
                instruction=task,
            )

        def decide_next(self, _run, _targets, _context, _allowed_actions):
            return CoordinatorDecision(
                CoordinatorAction.BLOCK,
                reason_code=unknown_reason,
            )

    backend = ImmediateTeamBackend()
    writer, blobs = make_team_storage(tmp_path)
    actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
        decision_provider=_BlockingDecisionProvider(),
    )
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_unknown_reason",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="执行但不要泄露内部错误",
    )
    actor.drain()

    final = actor.projection().runs[run.run_id]
    assert final.phase is TeamRunPhase.BLOCKED
    assert final.error_code == "team_task_failed"
    assert "team_task_failed" in backend.notifications[0][2]
    assert unknown_reason not in backend.notifications[0][2]
    assert unknown_reason not in str(
        [event.payload for frame in writer.replay() for event in frame.events]
    )

    actor.close()
    blobs.close()
    writer.close()


def test_unknown_backend_error_is_redacted_before_assignment_journal(
    tmp_path,
) -> None:
    unknown_error = "private_backend_detail"

    class _UnknownErrorBackend(ImmediateTeamBackend):
        def submit(self, **kwargs):
            acceptance_id = super().submit(**kwargs)
            self.results[acceptance_id] = TeamAttemptResult(
                "action_required",
                error_code=unknown_error,
                retry_allowed=False,
            )
            return acceptance_id

    backend = _UnknownErrorBackend()
    writer, blobs, actor = _actor(tmp_path, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_unknown_backend_error",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="执行并安全回报错误",
    )
    actor.drain()

    projection = actor.projection()
    final = projection.runs[run.run_id]
    assignment = projection.assignments[final.assignment_ids[0]]
    assert final.error_code == "team_task_failed"
    assert assignment.error_code == "team_task_failed"
    assert unknown_error not in str(
        [event.payload for frame in writer.replay() for event in frame.events]
    )
    assert unknown_error not in backend.notifications[0][2]

    actor.close()
    blobs.close()
    writer.close()


def test_failed_block_notification_retries_after_reopen_with_same_key(tmp_path) -> None:
    class _RetryingBackend(ImmediateTeamBackend):
        def __init__(self):
            super().__init__()
            self.fail_notifications = True
            self.notification_attempts = []

        def submit(self, **kwargs):
            acceptance_id = super().submit(**kwargs)
            self.results[acceptance_id] = TeamAttemptResult(
                "action_required",
                error_code="employee_session_failed",
                retry_allowed=False,
            )
            return acceptance_id

        def notify(self, message_id, chat_id, content, *, idempotency_key):
            self.notification_attempts.append(idempotency_key)
            if self.fail_notifications:
                raise RuntimeError("notification transport unavailable")
            self.notifications.append((message_id, chat_id, content))

    backend = _RetryingBackend()
    writer, blobs, first_actor = _actor(tmp_path, backend)
    run = first_actor.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_failure",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="执行并回报",
    )
    first_actor.drain()

    projection = first_actor.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKING
    assert (
        projection.effects[(f"{run.run_id}:blocked-notify:1", "notify")]
        == "action_required"
    )
    first_actor.close()

    backend.fail_notifications = False
    reopened_actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
    )
    assert reopened_actor.recover() == 1
    reopened_actor.drain()

    projection = reopened_actor.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKED
    assert (
        projection.effects[(f"{run.run_id}:blocked-notify:2", "notify")]
        == "committed"
    )
    assert len(backend.notifications) == 1
    assert len(backend.notification_attempts) == 2
    assert len(set(backend.notification_attempts)) == 1
    assert reopened_actor.recover() == 0

    reopened_actor.close()
    blobs.close()
    writer.close()


def test_notification_type_error_is_not_mistaken_for_legacy_signature(
    tmp_path,
) -> None:
    class _TypeErrorBackend(ImmediateTeamBackend):
        def __init__(self):
            super().__init__()
            self.notification_attempts = []

        def submit(self, **kwargs):
            acceptance_id = super().submit(**kwargs)
            self.results[acceptance_id] = TeamAttemptResult(
                "action_required",
                error_code="employee_session_failed",
                retry_allowed=False,
            )
            return acceptance_id

        def notify(self, _message_id, _chat_id, _content, *, idempotency_key):
            self.notification_attempts.append(idempotency_key)
            raise TypeError("internal transport serialization failed")

    backend = _TypeErrorBackend()
    writer, blobs, actor = _actor(tmp_path, backend)
    run = actor.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_type_error",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="执行并回报",
    )
    actor.drain()

    projection = actor.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKING
    assert len(backend.notification_attempts) == 1
    assert (
        projection.effects[(f"{run.run_id}:blocked-notify:1", "notify")]
        == "action_required"
    )

    actor.close()
    blobs.close()
    writer.close()


def test_executing_block_notification_is_replayed_after_process_crash(tmp_path) -> None:
    class _CrashOnceBackend(ImmediateTeamBackend):
        def __init__(self):
            super().__init__()
            self.crash_notification = True
            self.notification_attempts = []

        def submit(self, **kwargs):
            acceptance_id = super().submit(**kwargs)
            self.results[acceptance_id] = TeamAttemptResult(
                "action_required",
                error_code="employee_session_failed",
                retry_allowed=False,
            )
            return acceptance_id

        def notify(self, message_id, chat_id, content, *, idempotency_key):
            self.notification_attempts.append(idempotency_key)
            if self.crash_notification:
                raise SystemExit("simulated process crash")
            self.notifications.append((message_id, chat_id, content))

    backend = _CrashOnceBackend()
    writer, blobs, first_actor = _actor(tmp_path, backend)
    run = first_actor.start_task(
        tenant_key="tenant_1",
        message_id="om_notify_crash",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="执行并回报",
    )
    first_actor.drain()

    projection = first_actor.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKING
    assert (
        projection.effects[(f"{run.run_id}:blocked-notify:1", "notify")]
        == "executing"
    )
    assert len(backend.submissions) == 1
    first_actor.close()

    backend.crash_notification = False
    reopened_actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
    )
    assert reopened_actor.recover() == 1
    reopened_actor.drain()

    projection = reopened_actor.projection()
    assert projection.runs[run.run_id].phase is TeamRunPhase.BLOCKED
    assert (
        projection.effects[(f"{run.run_id}:blocked-notify:1", "notify")]
        == "committed"
    )
    assert len(backend.submissions) == 1
    assert len(backend.notifications) == 1
    assert len(backend.notification_attempts) == 2
    assert len(set(backend.notification_attempts)) == 1

    reopened_actor.close()
    blobs.close()
    writer.close()


def test_coordinator_accepts_bounded_parallel_fanout(tmp_path) -> None:
    backend = ImmediateTeamBackend()
    writer, blobs = make_team_storage(tmp_path)
    actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=backend,
        poll_seconds=0.001,
        decision_provider=lambda _run, _targets, task: CoordinatorDecision(
            CoordinatorAction.ASSIGN,
            ("agt_coder", "agt_reviewer"),
            role="execute",
            instruction=f"并行处理：{task}",
        ),
    )
    actor.start_task(
        tenant_key="tenant_1",
        message_id="om_fanout",
        chat_id="oc_team",
        requester_principal_id="ou_user",
        task="并行分析",
    )
    actor.drain()
    assert [item[1] for item in backend.submissions[:2]] == [
        "agt_coder",
        "agt_reviewer",
    ]
    actor.close()
    blobs.close()
    writer.close()


def test_selector_excludes_degraded_busy_and_roleless_targets() -> None:
    targets = (
        SimpleNamespace(
            agent_id="agt_none",
            role="",
            capabilities=(),
            runtime_status="ready",
            mailbox_load=0,
        ),
        SimpleNamespace(
            agent_id="agt_busy",
            role="coder",
            capabilities=("python",),
            runtime_status="busy",
            mailbox_load=0,
        ),
        SimpleNamespace(
            agent_id="agt_ready",
            role="coder",
            capabilities=("python",),
            runtime_status="ready_cold",
            mailbox_load=1,
        ),
    )
    selected = TeamCoordinatorActor._select_target(  # noqa: SLF001
        None, targets, "Python", role="execute"
    )
    assert selected.agent_id == "agt_ready"
