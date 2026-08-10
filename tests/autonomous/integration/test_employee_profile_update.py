from __future__ import annotations

from collections.abc import Callable

import pytest

from src.autonomous.domain import EmployeeState, WorkerType
from src.autonomous.journal import JournalWriter, MemoryAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionRepository, ProjectionState
from src.autonomous.journal.writer import AnchorMismatchError
from src.autonomous.provisioning.hire_port import EmployeeRoleUpdateRequest
from src.autonomous.provisioning.hire_service import (
    HireAdmissionError,
    ProductionEmployeeHireService,
)
from src.autonomous.workforce.projection import commit_workforce_events
from tests.autonomous.workforce_helpers import HMAC_KEY, employee_created


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


def _service(
    tmp_path,
    *employees: JournalEvent,
    admins: set[str] | None = None,
    admin_provider: Callable[[], frozenset[str]] | None = None,
    anchor: MemoryAnchor | None = None,
):
    writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=anchor or MemoryAnchor(),
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )
    state = ProjectionState()
    commit_workforce_events(writer, state, employees)
    mutable_admins = admins if admins is not None else {"ou_admin"}
    provider = admin_provider or (lambda: frozenset(mutable_admins))
    service = ProductionEmployeeHireService(
        writer,
        state,
        admin_principal_ids_provider=provider,
    )
    return writer, service, mutable_admins


def _request(
    employee: str = "agt_atlas",
    role: str = "后端工程师",
    *,
    tenant_key: str = "tenant_1",
    requester: str = "ou_admin",
    message_id: str = "om_role_1",
) -> EmployeeRoleUpdateRequest:
    return EmployeeRoleUpdateRequest(
        tenant_key=tenant_key,
        employee=employee,
        role=role,
        requester_principal_id=requester,
        message_id=message_id,
    )


@pytest.mark.parametrize("selector", ["agt_atlas", "Atlas"])
def test_role_update_commits_exact_fact_and_replays_by_id_or_unique_name(
    tmp_path,
    selector: str,
) -> None:
    writer, service, _admins = _service(
        tmp_path,
        _employee("agt_atlas", "Atlas"),
    )

    service.update_employee_role(
        _request(employee=selector, role="  后端\t工程师\n"),
    )

    frame = writer.get_last_frame()
    assert frame is not None
    assert frame.sequence == 2
    assert len(frame.events) == 1
    assert frame.events[0].event_type == "employee.profile_changed"
    assert frame.events[0].aggregate_id == "agt_atlas"
    assert frame.events[0].payload == {"role": "后端 工程师"}
    assert service.projection_state.employees["agt_atlas"].role == "后端 工程师"

    replayed = ProjectionRepository().rebuild(writer.replay())
    assert replayed.employees["agt_atlas"].role == "后端 工程师"
    writer.close()


def test_same_normalized_role_is_idempotent_without_a_new_frame(tmp_path) -> None:
    writer, service, _admins = _service(
        tmp_path,
        _employee("agt_atlas", "Atlas"),
    )
    service.update_employee_role(_request(role="后端 工程师"))
    committed = writer.get_last_frame()
    assert committed is not None

    service.update_employee_role(
        _request(role=" 后端\t工程师 ", message_id="om_role_retry"),
    )

    assert writer.get_last_frame() is committed
    assert writer.get_last_frame().sequence == 2
    assert service.projection_state.employees["agt_atlas"].role == "后端 工程师"
    writer.close()


def test_admin_provider_is_evaluated_for_each_role_update(tmp_path) -> None:
    admins: set[str] = set()
    writer, service, _admins = _service(
        tmp_path,
        _employee("agt_atlas", "Atlas"),
        admins=admins,
    )

    with pytest.raises(HireAdmissionError, match="authorized"):
        service.update_employee_role(_request())
    assert writer.get_last_frame().sequence == 1

    admins.add("ou_admin")
    service.update_employee_role(_request(message_id="om_role_after_admin_change"))

    assert writer.get_last_frame().sequence == 2
    assert service.projection_state.employees["agt_atlas"].role == "后端工程师"
    writer.close()


@pytest.mark.parametrize("role", ["", "  \n\t ", "职" * 81])
def test_role_update_rejects_invalid_role_without_writing(tmp_path, role: str) -> None:
    writer, service, _admins = _service(
        tmp_path,
        _employee("agt_atlas", "Atlas"),
    )

    with pytest.raises(HireAdmissionError, match="role"):
        service.update_employee_role(_request(role=role))

    assert writer.get_last_frame().sequence == 1
    assert service.projection_state.employees["agt_atlas"].role == ""
    writer.close()


def test_role_update_accepts_normalized_80_character_role(tmp_path) -> None:
    writer, service, _admins = _service(
        tmp_path,
        _employee("agt_atlas", "Atlas"),
    )
    role = "职" * 80

    service.update_employee_role(_request(role=f" {role} "))

    assert service.projection_state.employees["agt_atlas"].role == role
    assert writer.get_last_frame().sequence == 2
    writer.close()


@pytest.mark.parametrize(
    ("employees", "role_request"),
    [
        (
            (_employee("agt_atlas", "Atlas"),),
            _request(requester="ou_not_admin"),
        ),
        (
            (_employee("agt_other", "Atlas", tenant_key="tenant_2"),),
            _request(employee="Atlas", tenant_key="tenant_1"),
        ),
        (
            (
                _employee(
                    "agt_atlas",
                    "Atlas",
                    state=EmployeeState.ARCHIVED,
                ),
            ),
            _request(),
        ),
        (
            (
                _employee(
                    "agt_atlas",
                    "Atlas",
                    worker_type=WorkerType.LOGICAL,
                ),
            ),
            _request(),
        ),
        (
            (_employee("agt_atlas", "Atlas"),),
            _request(employee="Missing"),
        ),
        (
            (
                _employee("agt_target", "Atlas"),
                _employee("agt_other", "agt_target"),
            ),
            _request(employee="agt_target"),
        ),
    ],
    ids=[
        "non-admin",
        "cross-tenant",
        "archived",
        "non-visible",
        "unknown",
        "ambiguous-id-or-name",
    ],
)
def test_role_update_rejects_unauthorized_or_unresolvable_target_without_writing(
    tmp_path,
    employees: tuple[JournalEvent, ...],
    role_request: EmployeeRoleUpdateRequest,
) -> None:
    writer, service, _admins = _service(tmp_path, *employees)

    with pytest.raises(HireAdmissionError):
        service.update_employee_role(role_request)

    assert writer.get_last_frame().sequence == 1
    assert all(employee.role == "" for employee in service.projection_state.employees.values())
    writer.close()


def test_role_update_fails_closed_when_admin_provider_fails(tmp_path) -> None:
    def unavailable_admins() -> frozenset[str]:
        raise RuntimeError("admin backend secret")

    writer, service, _admins = _service(
        tmp_path,
        _employee("agt_atlas", "Atlas"),
        admin_provider=unavailable_admins,
    )

    with pytest.raises(HireAdmissionError, match="authorized"):
        service.update_employee_role(_request())

    assert writer.get_last_frame().sequence == 1
    assert service.projection_state.employees["agt_atlas"].role == ""
    writer.close()


class _SelectiveRejectAnchor(MemoryAnchor):
    def __init__(self) -> None:
        super().__init__()
        self.reject_sequence: int | None = None

    def compare_and_swap(
        self,
        expected_sequence: int,
        expected_hash: str,
        new_sequence: int,
        new_hash: str,
    ) -> bool:
        if new_sequence == self.reject_sequence:
            return False
        return super().compare_and_swap(
            expected_sequence,
            expected_hash,
            new_sequence,
            new_hash,
        )


def test_role_update_anchor_failure_does_not_publish_success_or_projection(tmp_path) -> None:
    anchor = _SelectiveRejectAnchor()
    writer, service, _admins = _service(
        tmp_path,
        _employee("agt_atlas", "Atlas"),
        anchor=anchor,
    )
    anchor.reject_sequence = 2

    with pytest.raises(AnchorMismatchError):
        service.update_employee_role(_request())

    assert anchor.read().sequence == 1
    assert service.projection_state.cursor_sequence == 1
    assert service.projection_state.employees["agt_atlas"].role == ""
    writer.close()
