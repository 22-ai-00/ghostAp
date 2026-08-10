from __future__ import annotations

from src.autonomous.domain import EmployeeState, WorkerType
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionRepository, ProjectionState
from src.autonomous.membership.service import EmployeeMembershipService
from src.autonomous.provisioning.hire_service import ProductionEmployeeHireService
from src.autonomous.workforce.projection import commit_workforce_events
from tests.autonomous.workforce_helpers import employee_created, make_writer


def _employee(
    agent_id: str,
    name: str,
    *,
    tenant_key: str = "tenant_1",
    state: EmployeeState = EmployeeState.ACTIVE,
    worker_type: WorkerType = WorkerType.VISIBLE,
) -> JournalEvent:
    created = employee_created(agent_id, name)
    return JournalEvent(
        event_type=created.event_type,
        aggregate_id=created.aggregate_id,
        payload={
            **created.payload,
            "tenant_key": tenant_key,
            "state": state.value,
            "worker_type": worker_type.value,
        },
    )


def _hire_service(tmp_path, *employees: JournalEvent):
    writer = make_writer(tmp_path)
    state = ProjectionState()
    commit_workforce_events(writer, state, employees)
    return writer, ProductionEmployeeHireService(writer, state)


def _roster_identity(employees) -> list[tuple[str, str, EmployeeState]]:
    return [(employee.name, employee.agent_id, employee.state) for employee in employees]


def test_roster_lists_every_visible_lifecycle_for_exact_tenant(tmp_path) -> None:
    lifecycle = tuple(EmployeeState)
    visible = tuple(
        _employee(
            f"agt_visible_{index:02d}",
            f"Visible {len(lifecycle) - index:02d}",
            state=state,
        )
        for index, state in enumerate(lifecycle)
    )
    writer, service = _hire_service(
        tmp_path,
        *visible,
        _employee("agt_other_tenant", "Other tenant", tenant_key="tenant_2"),
        _employee("agt_logical", "Logical", worker_type=WorkerType.LOGICAL),
        _employee("agt_ephemeral", "Ephemeral", worker_type=WorkerType.EPHEMERAL),
    )

    roster = service.list_employee_roster("tenant_1")

    expected = sorted(
        (
            (event.payload["name"], event.aggregate_id, EmployeeState(event.payload["state"]))
            for event in visible
        ),
        key=lambda item: (item[0].casefold(), item[1]),
    )
    assert _roster_identity(roster) == expected
    writer.close()


def test_roster_can_hide_archived_without_hiding_other_non_active_states(tmp_path) -> None:
    writer, service = _hire_service(
        tmp_path,
        _employee("agt_draft", "Draft", state=EmployeeState.DRAFT),
        _employee("agt_active", "Active", state=EmployeeState.ACTIVE),
        _employee(
            "agt_action_required",
            "Action required",
            state=EmployeeState.ACTION_REQUIRED,
        ),
        _employee("agt_archived", "Archived", state=EmployeeState.ARCHIVED),
    )

    without_archived = service.list_employee_roster(
        "tenant_1",
        include_archived=False,
    )
    with_archived = service.list_employee_roster(
        "tenant_1",
        include_archived=True,
    )

    assert {employee.state for employee in without_archived} == {
        EmployeeState.DRAFT,
        EmployeeState.ACTIVE,
        EmployeeState.ACTION_REQUIRED,
    }
    assert {employee.state for employee in with_archived} == {
        EmployeeState.DRAFT,
        EmployeeState.ACTIVE,
        EmployeeState.ACTION_REQUIRED,
        EmployeeState.ARCHIVED,
    }
    writer.close()


def test_roster_sorting_is_stable_after_journal_replay(tmp_path) -> None:
    writer, service = _hire_service(
        tmp_path,
        _employee("agt_3", "zeta"),
        _employee("agt_2", "Alpha"),
        _employee("agt_1", "beta"),
    )

    before_restart = service.list_employee_roster("tenant_1")
    replayed = ProjectionRepository().rebuild(writer.replay())
    restarted = ProductionEmployeeHireService(writer, replayed)
    after_restart = restarted.list_employee_roster("tenant_1")

    expected_ids = ["agt_2", "agt_1", "agt_3"]
    assert [employee.agent_id for employee in before_restart] == expected_ids
    assert [employee.agent_id for employee in after_restart] == expected_ids
    assert _roster_identity(after_restart) == _roster_identity(before_restart)
    writer.close()


def test_membership_employee_list_remains_active_visible_only(tmp_path) -> None:
    writer, hire = _hire_service(
        tmp_path,
        _employee("agt_active", "Active"),
        _employee(
            "agt_action_required",
            "Action required",
            state=EmployeeState.ACTION_REQUIRED,
        ),
        _employee("agt_logical", "Logical", worker_type=WorkerType.LOGICAL),
        _employee("agt_ephemeral", "Ephemeral", worker_type=WorkerType.EPHEMERAL),
        _employee("agt_other_tenant", "Other tenant", tenant_key="tenant_2"),
    )
    membership = EmployeeMembershipService(
        writer=writer,
        hire_service=hire,
        remote=object(),
        admin_principal_ids=frozenset(),
        team_owner_resolver=lambda _chat_id: "",
        team_active_resolver=lambda _chat_id: True,
    )

    employees = membership.list_employees("tenant_1")

    assert [employee.agent_id for employee in employees] == ["agt_active"]
    writer.close()
