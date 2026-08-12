"""Durable admission contract for production visible-employee hires."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from src.autonomous.domain import EmployeeState
from src.autonomous.journal.anchor import FileAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.journal.projections import ProjectionRepository, ProjectionState
from src.autonomous.journal.writer import JournalWriter
from src.autonomous.provisioning.hire_port import EmployeeHireRequest
from src.autonomous.provisioning.hire_service import (
    HireAdmissionError,
    ProductionEmployeeHireService,
)
from src.autonomous.provisioning.hire_state import HireProjection
from src.autonomous.workforce.projection import commit_workforce_events

HMAC_KEY = b"employee-hire-admission-test-key!"


def _request(
    *,
    message_id: str = "om_hire_1",
    employee_name: str = "Atlas",
    model: str = "gpt-5.6-sol",
    existing_app_id: str = "",
) -> EmployeeHireRequest:
    return EmployeeHireRequest(
        employee_name=employee_name,
        tool="traex",
        model=model,
        effort="high",
        chat_id="oc_admin_dm",
        message_id=message_id,
        requester_principal_id="ou_admin",
        requester_union_id="on_admin",
        tenant_key="tenant-a",
        profile="standard",
        role="software engineer",
        persona="careful reviewer",
        personality_traits=("严谨", "主动沟通"),
        capabilities=("coding", "review", "file_read", "file_write"),
        permissions=("file_read",),
        existing_app_id=existing_app_id,
    )


def _service(
    base: Path,
    *,
    visible_employee_limit: int = 3,
    release_evidence_ready: bool = True,
    credential_keyring_ready: bool = True,
    runtime_recovery_ready: bool = True,
    provisioning_submitter: Callable[[str], object] | None = None,
    admin_provider: Callable[[], object] | None = None,
) -> tuple[ProductionEmployeeHireService, JournalWriter, ProjectionState]:
    writer = JournalWriter.open(
        base / "journal",
        anchor=FileAnchor(base / "anchor.json"),
        hmac_key=HMAC_KEY,
        writer_epoch=1,
    )
    projection = ProjectionState()
    service = ProductionEmployeeHireService(
        writer,
        projection,
        visible_employee_limit=visible_employee_limit,
        release_evidence_ready=release_evidence_ready,
        credential_keyring_ready=credential_keyring_ready,
        runtime_recovery_ready=runtime_recovery_ready,
        provisioning_submitter=provisioning_submitter,
        admin_principal_ids_provider=(
            admin_provider or (lambda: frozenset({"ou_admin"}))
        ),
    )
    return service, writer, projection


def test_complete_profile_is_anchored_and_replayed_before_unlocked_submit(
    tmp_path: Path,
) -> None:
    import src.autonomous.workforce.projection as workforce_projection

    observations: dict[str, object] = {}
    service: ProductionEmployeeHireService
    writer: JournalWriter

    def submit(intent_id: str) -> None:
        observations["intent_id"] = intent_id
        observations["workforce_locked"] = (
            workforce_projection._WORKFORCE_COMMIT_LOCK._is_owned()
        )
        observations["hire_locked"] = service._mutex._is_owned()
        observations["journal_transaction_locked"] = (
            writer._transaction_mutex._is_owned()
        )
        observations["anchor"] = writer.anchor.read()
        observations["frames"] = tuple(writer.replay())
        observations["state"] = service.get_state(intent_id)

    service, writer, projection = _service(
        tmp_path,
        provisioning_submitter=submit,
    )

    admitted = service.start_hire(_request())

    frames = tuple(writer.replay())
    assert observations["intent_id"] == admitted.intent_id
    assert observations["workforce_locked"] is False
    assert observations["hire_locked"] is False
    assert observations["journal_transaction_locked"] is False
    assert observations["frames"] == frames
    assert observations["state"] == admitted
    assert observations["anchor"].sequence == frames[0].sequence
    assert observations["anchor"].frame_hash == frames[0].frame_hash
    assert [event.event_type for event in frames[0].events] == ["employee.created"]
    created = frames[0].events[0]
    assert created.payload["role"] == "software engineer"
    assert created.payload["persona"] == "careful reviewer"
    assert created.payload["personality_traits"] == ["严谨", "主动沟通"]
    assert created.payload["capabilities"] == [
        "coding",
        "review",
        "file_read",
        "file_write",
    ]
    assert created.payload["permissions"] == ["file_read"]
    assert projection.employees[admitted.agent_id].role == "software engineer"


def test_same_message_is_idempotent_and_rejects_every_persisted_field_drift(
    tmp_path: Path,
) -> None:
    admins = frozenset({"ou_admin", "ou_other"})
    service, writer, _projection = _service(
        tmp_path,
        admin_provider=lambda: admins,
    )
    request = _request()

    admitted = service.start_hire(request)
    assert service.start_hire(request) == admitted

    changes = {
        "employee_name": "Beacon",
        "tool": "codex",
        "model": "another-model",
        "effort": "low",
        "chat_id": "oc_other_dm",
        "requester_principal_id": "ou_other",
        "requester_union_id": "on_other",
        "profile": "max",
        "role": "reviewer",
        "persona": "different persona",
        "personality_traits": ("严谨", "好奇"),
        "capabilities": ("coding", "testing", "file_read", "file_write"),
        "permissions": ("file_write",),
        "existing_app_id": "cli_existing_123",
    }
    for field_name, value in changes.items():
        with pytest.raises(HireAdmissionError, match="idempotency"):
            service.start_hire(replace(request, **{field_name: value}))

    assert len(tuple(writer.replay())) == 1


def test_admin_provider_is_dynamic_and_hire_authorization_fails_closed(
    tmp_path: Path,
) -> None:
    admins: set[str] = set()
    service, writer, _projection = _service(
        tmp_path,
        admin_provider=lambda: frozenset(admins),
    )

    with pytest.raises(HireAdmissionError, match="authorized"):
        service.start_hire(_request())
    assert tuple(writer.replay()) == ()

    admins.add("ou_admin")
    admitted = service.start_hire(_request())
    admins.clear()

    with pytest.raises(HireAdmissionError, match="authorized"):
        service.start_hire(_request())
    assert HireProjection.rebuild(writer.replay()).get(admitted.intent_id) == admitted


def _raising_admin_provider() -> object:
    raise RuntimeError("admin backend unavailable")


@pytest.mark.parametrize(
    "provider",
    [
        _raising_admin_provider,
        lambda: "ou_admin",
        lambda: frozenset({" ou_admin"}),
    ],
)
def test_invalid_admin_provider_result_rejects_before_journal(
    tmp_path: Path,
    provider: Callable[[], object],
) -> None:
    service, writer, _projection = _service(
        tmp_path,
        admin_provider=provider,
    )

    with pytest.raises(HireAdmissionError, match="authorized"):
        service.start_hire(_request())

    assert tuple(writer.replay()) == ()


@pytest.mark.parametrize("requester_union_id", ["", " on_admin", "on_admin "])
def test_new_hire_requires_nonempty_trimmed_requester_union_id(
    tmp_path: Path,
    requester_union_id: str,
) -> None:
    service, writer, _projection = _service(tmp_path)

    with pytest.raises(HireAdmissionError, match="requester_union_id"):
        service.start_hire(
            replace(_request(), requester_union_id=requester_union_id)
        )

    assert tuple(writer.replay()) == ()


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"visible_employee_limit": 0}, "visible_employee_limit"),
        ({"release_evidence_ready": False}, "release_evidence"),
        ({"credential_keyring_ready": False}, "credential_keyring"),
        ({"runtime_recovery_ready": False}, "runtime_recovery"),
    ],
)
def test_readiness_gates_reject_before_journal(
    tmp_path: Path,
    overrides: dict[str, object],
    blocker: str,
) -> None:
    service, writer, _projection = _service(tmp_path, **overrides)

    with pytest.raises(HireAdmissionError, match=blocker):
        service.start_hire(_request())

    assert tuple(writer.replay()) == ()


def test_closed_admission_and_capacity_reject_without_extra_frames(
    tmp_path: Path,
) -> None:
    capacity_service, capacity_writer, _projection = _service(
        tmp_path / "capacity",
        visible_employee_limit=1,
    )
    capacity_service.start_hire(_request())

    with pytest.raises(HireAdmissionError, match="capacity"):
        capacity_service.start_hire(
            _request(message_id="om_hire_2", employee_name="Beacon")
        )
    assert len(tuple(capacity_writer.replay())) == 1

    closed_service, closed_writer, _projection = _service(tmp_path / "closed")
    closed_service.stop_admission()
    with pytest.raises(HireAdmissionError, match="admission_closed"):
        closed_service.start_hire(_request())
    assert tuple(closed_writer.replay()) == ()


def test_live_names_are_casefold_unique_and_archive_releases_name_in_create_frame(
    tmp_path: Path,
) -> None:
    service, writer, projection = _service(tmp_path)
    previous = service.start_hire(_request())

    with pytest.raises(HireAdmissionError, match="name"):
        service.start_hire(
            _request(message_id="om_hire_2", employee_name="atlas")
        )
    assert len(tuple(writer.replay())) == 1

    commit_workforce_events(
        writer,
        projection,
        (
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id=previous.agent_id,
                payload={"state": EmployeeState.ARCHIVED.value},
            ),
        ),
    )
    current = service.start_hire(
        _request(message_id="om_hire_2", employee_name="ATLAS")
    )

    final_frame = tuple(writer.replay())[-1]
    assert [event.event_type for event in final_frame.events] == [
        "employee.name_released",
        "employee.created",
    ]
    assert projection.employee_name_keys[("tenant-a", "atlas")] == current.agent_id


def test_existing_app_can_have_only_one_live_hire(tmp_path: Path) -> None:
    service, writer, _projection = _service(tmp_path)
    service.start_hire(_request(existing_app_id="cli_shared_123"))

    with pytest.raises(HireAdmissionError, match="existing app already assigned"):
        service.start_hire(
            _request(
                message_id="om_hire_2",
                employee_name="Beacon",
                existing_app_id="cli_shared_123",
            )
        )

    assert len(tuple(writer.replay())) == 1


def test_submit_failure_keeps_durable_admission_replayable(tmp_path: Path) -> None:
    def fail_submit(_intent_id: str) -> None:
        raise RuntimeError("queue unavailable")

    request = _request()
    service, writer, _projection = _service(
        tmp_path,
        provisioning_submitter=fail_submit,
    )

    with pytest.raises(HireAdmissionError, match="after durable admission"):
        service.start_hire(request)

    first_frame = tuple(writer.replay())[0]
    assert writer.anchor.read().sequence == first_frame.sequence
    intent_id = first_frame.events[0].payload["hire_intent_id"]
    service.close()

    reopened_writer = JournalWriter.open(
        tmp_path / "journal",
        anchor=FileAnchor(tmp_path / "anchor.json"),
        hmac_key=HMAC_KEY,
        writer_epoch=2,
    )
    replayed = ProductionEmployeeHireService(
        reopened_writer,
        ProjectionState(),
        visible_employee_limit=3,
        release_evidence_ready=True,
        credential_keyring_ready=True,
        admin_principal_ids_provider=lambda: frozenset({"ou_admin"}),
    )

    assert replayed.start_hire(request).intent_id == intent_id
    assert len(tuple(reopened_writer.replay())) == 1


def test_empty_model_round_trips_as_backend_default_through_both_projections(
    tmp_path: Path,
) -> None:
    service, writer, projection = _service(tmp_path)

    admitted = service.start_hire(_request(model=""))

    assert admitted.model == ""
    assert projection.employees[admitted.agent_id].model == ""
    assert HireProjection.rebuild(writer.replay()).get(admitted.intent_id).model == ""
    assert ProjectionRepository().rebuild(writer.replay()).employees[
        admitted.agent_id
    ].model == ""


@pytest.mark.parametrize(
    "changes",
    [
        {"model": None},
        {"model": " gpt-5.6-sol"},
        {"model": "gpt 5.6"},
        {"model": "gpt\t5.6"},
        {"model": "gpt\n5.6"},
        {"model": "gpt-5.6-sol/max/high"},
        {"tool": ""},
        {"profile": "turbo"},
        {"effort": "potato"},
    ],
)
def test_model_sentinel_still_rejects_invalid_model_tool_profile_and_effort(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    service, writer, _projection = _service(
        tmp_path,
        visible_employee_limit=1,
    )

    with pytest.raises(HireAdmissionError):
        service.start_hire(replace(_request(), **changes))

    assert tuple(writer.replay()) == ()
    admitted = service.start_hire(_request(model=""))
    assert admitted.model == ""
    assert len(tuple(writer.replay())) == 1


@pytest.mark.parametrize(
    "employee_name",
    [
        "",
        "   ",
        " Atlas",
        "Atlas ",
        "At\nlas",
        "At\tlas",
        "At\x00las",
        "At\x7flas",
        "A" * 81,
    ],
)
def test_registration_invalid_employee_name_is_rejected_without_capacity_use(
    tmp_path: Path,
    employee_name: str,
) -> None:
    service, writer, _projection = _service(
        tmp_path,
        visible_employee_limit=1,
    )

    with pytest.raises(HireAdmissionError):
        service.start_hire(_request(employee_name=employee_name))

    assert tuple(writer.replay()) == ()
    admitted = service.start_hire(_request(employee_name="A" * 80))
    assert admitted.employee_name == "A" * 80
    assert len(tuple(writer.replay())) == 1


@pytest.mark.parametrize(
    "tool",
    [
        "unknown",
        "Traex",
        "trae",
        " traex",
        "traex ",
    ],
)
def test_noncanonical_employee_tool_is_rejected_without_capacity_use(
    tmp_path: Path,
    tool: str,
) -> None:
    service, writer, _projection = _service(
        tmp_path,
        visible_employee_limit=1,
    )

    with pytest.raises(HireAdmissionError):
        service.start_hire(
            replace(_request(model=""), tool=tool, effort="default")
        )

    assert tuple(writer.replay()) == ()
    admitted = service.start_hire(_request(model=""))
    assert admitted.tool == "traex"
    assert admitted.model == ""
    assert len(tuple(writer.replay())) == 1


@pytest.mark.parametrize(
    "tool",
    ["coco", "claude", "aiden", "codex", "gemini", "traex", "grok"],
)
def test_canonical_employee_tools_accept_backend_default_model(
    tmp_path: Path,
    tool: str,
) -> None:
    service, writer, _projection = _service(tmp_path)

    admitted = service.start_hire(
        replace(_request(model=""), tool=tool, effort="default")
    )

    assert admitted.tool == tool
    assert admitted.model == ""
    assert len(tuple(writer.replay())) == 1
