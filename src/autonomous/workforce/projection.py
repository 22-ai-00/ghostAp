"""Journal-backed workforce projections and secret-free identity materialization."""

from __future__ import annotations

import copy
import re
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable

from ..domain import (
    BotPrincipal,
    EmployeeDefinition,
    EmployeeIdOrigin,
    EmployeeState,
)
from ..journal.frame import JournalEvent
from ..journal.writer import AnchorMismatchError, CommitState
from .authority import AuthorityMode, AuthoritySnapshot

if TYPE_CHECKING:
    from ..journal.projections import ProjectionState
    from ..journal.writer import CommitResult, JournalWriter


@dataclass
class WorkforceProjectionState:
    """Workforce portion embedded in the canonical ProjectionState."""

    employees: dict[str, EmployeeDefinition] = field(default_factory=dict)
    bot_principals: dict[str, BotPrincipal] = field(default_factory=dict)
    employee_name_keys: dict[tuple[str, str], str] = field(default_factory=dict)
    legacy_agent_aliases: dict[str, str] = field(default_factory=dict)
    legacy_source_hashes: dict[str, str] = field(default_factory=dict)
    authority_epoch: int = 0
    authority_mode: AuthorityMode = AuthorityMode.LEGACY_WRITE
    authority_cutover_sequence: int = 0

    def authority_snapshot(self) -> AuthoritySnapshot:
        """Return the replayable writer-authority view for mutation guards."""

        return AuthoritySnapshot(
            epoch=self.authority_epoch,
            mode=self.authority_mode,
            cutover_sequence=self.authority_cutover_sequence,
        )


_WORKFORCE_EVENTS = frozenset(
    {
        "employee.created",
        "employee.name_released",
        "employee.state_changed",
        "employee.profile_changed",
        "employee.membership_changed",
        "employee.membership.effect_prepared",
        "employee.membership.effect_executing",
        "employee.membership.effect_committed",
        "employee.membership.effect_action_required",
        "employee.legacy_alias_bound",
        "employee.bot_principal_bound",
        "bot_principal.bound",
        "bot_principal.manifest_desired",
        "bot_principal.manifest_observed",
        "credential.destroyed",
        "authority.cutover",
    }
)


def is_workforce_event(event_type: str) -> bool:
    """Return whether an event can change projected employee authority."""

    return event_type in _WORKFORCE_EVENTS
_PROFILE_FIELDS = frozenset(
    {
        "name",
        "emoji",
        "tool",
        "model",
        "profile",
        "effort",
        "role",
        "persona",
        "personality_traits",
        "capabilities",
        "permissions",
        "budget_template",
    }
)
_WORKFORCE_COMMIT_LOCK = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
_AGENT_ID_PATTERN = re.compile(r"agt_[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_BOT_PRINCIPAL_ID_PATTERN = re.compile(r"bot_[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@contextmanager
def workforce_projection_guard():
    """Serialize authority snapshots with workforce projection commits."""
    with _WORKFORCE_COMMIT_LOCK:
        yield


def _projection_error(message: str) -> Exception:
    from ..journal.projections import ProjectionError

    return ProjectionError(message)


def _employee(state: WorkforceProjectionState, agent_id: str) -> EmployeeDefinition:
    employee = state.employees.get(agent_id)
    if employee is None:
        raise _projection_error(f"unknown employee {agent_id}")
    return employee


def _principal(state: WorkforceProjectionState, principal_id: str) -> BotPrincipal:
    principal = state.bot_principals.get(principal_id)
    if principal is None:
        raise _projection_error(f"unknown bot principal {principal_id}")
    return principal


def _require_mutable_employee(employee: EmployeeDefinition) -> None:
    if employee.state is EmployeeState.ARCHIVED:
        raise _projection_error("archived employee is terminal")


def _name_key(tenant_key: str, name: str) -> tuple[str, str]:
    return (tenant_key, name.casefold())


def _reserve_name(
    state: WorkforceProjectionState,
    employee: EmployeeDefinition,
    *,
    previous_name: str = "",
) -> None:
    key = _name_key(employee.tenant_key, employee.name)
    owner = state.employee_name_keys.get(key)
    if owner is not None and owner != employee.agent_id:
        raise _projection_error(
            f"duplicate active employee name: {employee.name.casefold()}"
        )
    if previous_name and previous_name.casefold() != employee.name.casefold():
        previous_key = _name_key(employee.tenant_key, previous_name)
        if state.employee_name_keys.get(previous_key) == employee.agent_id:
            del state.employee_name_keys[previous_key]
    state.employee_name_keys[key] = employee.agent_id


def _create_employee(state: WorkforceProjectionState, event: JournalEvent) -> None:
    agent_id = str(event.payload.get("agent_id") or event.aggregate_id)
    if agent_id != event.aggregate_id:
        raise _projection_error("employee aggregate_id does not match agent_id")
    if _AGENT_ID_PATTERN.fullmatch(agent_id) is None:
        raise _projection_error("employee requires canonical agent_id")
    if agent_id in state.employees:
        raise _projection_error(f"employee already exists: {agent_id}")
    payload = dict(event.payload)
    payload["agent_id"] = agent_id
    payload.setdefault("created_at", event.timestamp)
    payload.setdefault("updated_at", event.timestamp)
    payload["aggregate_version"] = 1
    try:
        employee = EmployeeDefinition.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise _projection_error(f"invalid employee.created: {exc}") from exc
    _reserve_name(state, employee)
    state.employees[agent_id] = employee


def _release_employee_name(
    state: WorkforceProjectionState,
    event: JournalEvent,
) -> None:
    employee = _employee(state, event.aggregate_id)
    if employee.state is not EmployeeState.ARCHIVED:
        raise _projection_error("only archived employee names can be released")
    if set(event.payload) != {"name"}:
        raise _projection_error("employee name release payload is invalid")
    name = event.payload.get("name")
    if not isinstance(name, str) or name.casefold() != employee.name.casefold():
        raise _projection_error("employee name release does not match archive")
    key = _name_key(employee.tenant_key, employee.name)
    owner = state.employee_name_keys.get(key)
    if owner not in {None, employee.agent_id}:
        raise _projection_error("employee name release owner mismatch")
    if owner == employee.agent_id:
        del state.employee_name_keys[key]


def _change_employee_state(
    state: WorkforceProjectionState, event: JournalEvent
) -> None:
    employee = _employee(state, event.aggregate_id)
    _require_mutable_employee(employee)
    try:
        next_state = EmployeeState(event.payload["state"])
    except (KeyError, ValueError) as exc:
        raise _projection_error("invalid employee state") from exc
    state.employees[employee.agent_id] = replace(
        employee,
        state=next_state,
        updated_at=event.timestamp,
        aggregate_version=employee.aggregate_version + 1,
    )


def _change_employee_profile(
    state: WorkforceProjectionState, event: JournalEvent
) -> None:
    employee = _employee(state, event.aggregate_id)
    _require_mutable_employee(employee)
    unknown = set(event.payload) - _PROFILE_FIELDS
    if unknown:
        raise _projection_error(
            f"unsupported employee profile fields: {sorted(unknown)}"
        )
    values = {key: event.payload[key] for key in event.payload}
    try:
        updated = replace(
            employee,
            **values,
            updated_at=event.timestamp,
            aggregate_version=employee.aggregate_version + 1,
        )
    except (TypeError, ValueError) as exc:
        raise _projection_error(f"invalid employee profile: {exc}") from exc
    _reserve_name(state, updated, previous_name=employee.name)
    state.employees[employee.agent_id] = updated


def _change_membership(state: WorkforceProjectionState, event: JournalEvent) -> None:
    employee = _employee(state, event.aggregate_id)
    _require_mutable_employee(employee)
    groups = event.payload.get("member_groups")
    if not isinstance(groups, (list, tuple)) or not all(
        isinstance(group, str) and group for group in groups
    ):
        raise _projection_error("member_groups must be non-empty strings")
    state.employees[employee.agent_id] = replace(
        employee,
        member_groups=tuple(dict.fromkeys(groups)),
        updated_at=event.timestamp,
        aggregate_version=employee.aggregate_version + 1,
    )


def _record_membership_effect(
    _state: WorkforceProjectionState,
    _event: JournalEvent,
) -> None:
    """Membership effect facts are projected by the bounded membership view."""


def _bind_legacy_alias(state: WorkforceProjectionState, event: JournalEvent) -> None:
    employee = _employee(state, event.aggregate_id)
    _require_mutable_employee(employee)
    alias = event.payload.get("legacy_id_alias")
    source_hash = event.payload.get("source_hash")
    if not isinstance(alias, str) or not alias:
        raise _projection_error("legacy_id_alias is required")
    if not isinstance(source_hash, str) or not source_hash:
        raise _projection_error("source_hash is required")
    alias_owner = state.legacy_agent_aliases.get(alias)
    hash_owner = state.legacy_source_hashes.get(source_hash)
    if alias_owner not in (None, employee.agent_id):
        raise _projection_error("legacy alias already bound")
    if hash_owner not in (None, employee.agent_id):
        raise _projection_error("legacy source hash already bound")
    state.legacy_agent_aliases[alias] = employee.agent_id
    state.legacy_source_hashes[source_hash] = employee.agent_id
    state.employees[employee.agent_id] = replace(
        employee,
        id_origin=EmployeeIdOrigin.LEGACY_ALIAS,
        legacy_id_alias=alias,
        updated_at=event.timestamp,
        aggregate_version=employee.aggregate_version + 1,
    )


def _bind_employee_principal(
    state: WorkforceProjectionState, event: JournalEvent
) -> None:
    employee = _employee(state, event.aggregate_id)
    _require_mutable_employee(employee)
    agent_id = event.payload.get("agent_id")
    principal_id = event.payload.get("bot_principal_id")
    if agent_id != employee.agent_id or not isinstance(principal_id, str) or not principal_id:
        raise _projection_error("invalid employee bot principal binding")
    if _BOT_PRINCIPAL_ID_PATTERN.fullmatch(principal_id) is None:
        raise _projection_error("binding requires canonical bot_principal_id")
    if employee.bot_principal_id not in (None, principal_id):
        raise _projection_error("employee already bound to another bot principal")
    state.employees[employee.agent_id] = replace(
        employee,
        bot_principal_id=principal_id,
        updated_at=event.timestamp,
        aggregate_version=employee.aggregate_version + 1,
    )


def _bind_bot_principal(state: WorkforceProjectionState, event: JournalEvent) -> None:
    payload = dict(event.payload)
    principal_id = str(payload.get("bot_principal_id") or event.aggregate_id)
    if principal_id != event.aggregate_id:
        raise _projection_error("bot principal aggregate_id mismatch")
    if _BOT_PRINCIPAL_ID_PATTERN.fullmatch(principal_id) is None:
        raise _projection_error("binding requires canonical bot_principal_id")
    if principal_id in state.bot_principals:
        raise _projection_error("bot principal already exists")
    agent_id = payload.get("agent_id")
    employee = _employee(state, str(agent_id or ""))
    if employee.bot_principal_id != principal_id:
        raise _projection_error("bot binding events do not match")
    app_id = payload.get("app_id")
    credential_ref = payload.get("credential_ref")
    if payload.get("tenant_key") != employee.tenant_key:
        raise _projection_error("bot principal tenant does not match employee tenant")
    if not isinstance(app_id, str) or not app_id:
        raise _projection_error("bot principal app_id is required")
    if not isinstance(credential_ref, str) or not credential_ref:
        raise _projection_error("bot principal credential_ref is required")
    for existing in state.bot_principals.values():
        existing_employee = state.employees.get(existing.agent_id)
        if (
            app_id
            and existing.app_id == app_id
            and existing_employee is not None
            and existing_employee.state is not EmployeeState.ARCHIVED
        ):
            raise _projection_error(f"app_id already bound: {app_id}")
    payload["bot_principal_id"] = principal_id
    payload["aggregate_version"] = 1
    try:
        state.bot_principals[principal_id] = BotPrincipal.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise _projection_error(f"invalid bot principal: {exc}") from exc


def _observe_manifest(state: WorkforceProjectionState, event: JournalEvent) -> None:
    principal = _principal(state, event.aggregate_id)
    if set(event.payload) not in (
        {"observed_manifest_hash"},
        {"observed_manifest_hash", "evidence_source"},
    ):
        raise _projection_error("manifest observation payload is invalid")
    observed = event.payload.get("observed_manifest_hash")
    if not isinstance(observed, str) or not observed:
        raise _projection_error("observed_manifest_hash is required")
    evidence_source = event.payload.get("evidence_source")
    if evidence_source is not None and evidence_source != (
        "lark_oapi.aregister_app/exact_app_id"
    ):
        raise _projection_error("manifest evidence source is invalid")
    state.bot_principals[principal.bot_principal_id] = replace(
        principal,
        observed_manifest_hash=observed,
        manifest_evidence_source=str(evidence_source or ""),
        aggregate_version=principal.aggregate_version + 1,
    )


def _desire_manifest(state: WorkforceProjectionState, event: JournalEvent) -> None:
    principal = _principal(state, event.aggregate_id)
    if set(event.payload) != {"desired_manifest_hash", "scopes"}:
        raise _projection_error("manifest desired payload is invalid")
    desired = event.payload.get("desired_manifest_hash")
    scopes = event.payload.get("scopes")
    if not isinstance(desired, str) or not desired:
        raise _projection_error("desired_manifest_hash is required")
    if (
        isinstance(scopes, (str, bytes))
        or not isinstance(scopes, list)
        or any(not isinstance(scope, str) or not scope for scope in scopes)
    ):
        raise _projection_error("manifest scopes are invalid")
    state.bot_principals[principal.bot_principal_id] = replace(
        principal,
        desired_manifest_hash=desired,
        scopes=tuple(scopes),
        aggregate_version=principal.aggregate_version + 1,
    )


def _destroy_credential(state: WorkforceProjectionState, event: JournalEvent) -> None:
    principal = _principal(state, event.aggregate_id)
    credential_ref = event.payload.get("credential_ref")
    if credential_ref != principal.credential_ref:
        raise _projection_error("credential ref does not match bot principal")
    state.bot_principals[principal.bot_principal_id] = replace(
        principal,
        credential_ref="",
        aggregate_version=principal.aggregate_version + 1,
    )


def _cutover_authority(state: WorkforceProjectionState, event: JournalEvent) -> None:
    if event.aggregate_id != "workforce_authority":
        raise _projection_error("authority cutover aggregate is invalid")
    required = {"authority_epoch", "authority_mode", "cutover_sequence"}
    if set(event.payload) != required:
        raise _projection_error("authority cutover payload fields are invalid")
    epoch = event.payload.get("authority_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= state.authority_epoch:
        raise _projection_error("authority_epoch must increase")
    try:
        mode = AuthorityMode(event.payload["authority_mode"])
    except (KeyError, ValueError, TypeError) as exc:
        raise _projection_error("authority_mode is invalid") from exc
    cutover_sequence = event.payload.get("cutover_sequence")
    if (
        isinstance(cutover_sequence, bool)
        or not isinstance(cutover_sequence, int)
        or cutover_sequence < state.authority_cutover_sequence
    ):
        raise _projection_error("cutover_sequence must not decrease")
    state.authority_epoch = epoch
    state.authority_mode = mode
    state.authority_cutover_sequence = cutover_sequence


_APPLIERS = {
    "employee.created": _create_employee,
    "employee.name_released": _release_employee_name,
    "employee.state_changed": _change_employee_state,
    "employee.profile_changed": _change_employee_profile,
    "employee.membership_changed": _change_membership,
    "employee.membership.effect_prepared": _record_membership_effect,
    "employee.membership.effect_executing": _record_membership_effect,
    "employee.membership.effect_committed": _record_membership_effect,
    "employee.membership.effect_action_required": _record_membership_effect,
    "employee.legacy_alias_bound": _bind_legacy_alias,
    "employee.bot_principal_bound": _bind_employee_principal,
    "bot_principal.bound": _bind_bot_principal,
    "bot_principal.manifest_desired": _desire_manifest,
    "bot_principal.manifest_observed": _observe_manifest,
    "credential.destroyed": _destroy_credential,
    "authority.cutover": _cutover_authority,
}


def apply_workforce_event(state: WorkforceProjectionState, event: JournalEvent) -> bool:
    """Apply a workforce event and report whether the event was recognized."""

    applier = _APPLIERS.get(event.event_type)
    if applier is None:
        return False
    applier(state, event)
    return True


def _validate_bot_binding_pairs(events: tuple[JournalEvent, ...]) -> None:
    employee_events = tuple(
        event
        for event in events
        if event.event_type == "employee.bot_principal_bound"
    )
    principal_events = tuple(
        event for event in events if event.event_type == "bot_principal.bound"
    )
    if not employee_events and not principal_events:
        return
    if not employee_events or not principal_events:
        raise _projection_error("bot binding events must be in same transaction")
    if len(employee_events) != 1 or len(principal_events) != 1:
        raise _projection_error(
            "bot binding transaction requires exactly one event per aggregate"
        )
    employee_bindings = {
        (event.payload.get("agent_id"), event.payload.get("bot_principal_id"))
        for event in employee_events
    }
    principal_bindings = {
        (event.payload.get("agent_id"), event.payload.get("bot_principal_id") or event.aggregate_id)
        for event in principal_events
    }
    if employee_bindings != principal_bindings:
        raise _projection_error("bot binding events do not match")


def validate_workforce_frame_events(events: Iterable[JournalEvent]) -> None:
    """Validate transaction-level workforce invariants during replay."""

    workforce_events = tuple(
        event for event in events if event.event_type in _WORKFORCE_EVENTS
    )
    event_keys = [
        (event.aggregate_id, event.event_type) for event in workforce_events
    ]
    if len(event_keys) != len(set(event_keys)):
        raise _projection_error(
            "workforce transaction rejects duplicate event type per aggregate"
        )
    _validate_bot_binding_pairs(workforce_events)


def normalize_workforce_aggregate_versions(
    state: WorkforceProjectionState,
    aggregate_versions: Mapping[str, int],
    events: Iterable[JournalEvent],
) -> None:
    """Align workforce aggregate versions with the committed Journal frame."""

    workforce_ids = {
        event.aggregate_id
        for event in events
        if event.event_type in _WORKFORCE_EVENTS
    }
    for aggregate_id in workforce_ids:
        version = aggregate_versions[aggregate_id]
        employee = state.employees.get(aggregate_id)
        if employee is not None:
            state.employees[aggregate_id] = replace(
                employee,
                aggregate_version=version,
            )
        principal = state.bot_principals.get(aggregate_id)
        if principal is not None:
            state.bot_principals[aggregate_id] = replace(
                principal,
                aggregate_version=version,
            )


def _clone_projection_state(
    state: WorkforceProjectionState,
) -> WorkforceProjectionState:
    """Copy projection containers while sharing immutable domain values."""

    isolated = copy.copy(state)
    for attribute in (
        "employees",
        "bot_principals",
        "employee_name_keys",
        "legacy_agent_aliases",
        "legacy_source_hashes",
        "goals",
        "runs",
        "plans",
        "steps",
        "effects",
    ):
        values = getattr(state, attribute, None)
        if values is not None:
            setattr(isolated, attribute, dict(values))
    inbox = getattr(state, "inbox", None)
    if inbox is not None:
        isolated.inbox = {
            event_id: copy.copy(record) for event_id, record in inbox.items()
        }
    for attribute in ("dedup_keys", "occurrence_keys"):
        values = getattr(state, attribute, None)
        if values is not None:
            setattr(isolated, attribute, set(values))
    return isolated


def validate_workforce_events(
    state: WorkforceProjectionState,
    events: Iterable[JournalEvent],
) -> WorkforceProjectionState:
    """Validate a workforce transaction against an isolated projection copy."""

    event_values = tuple(events)
    if not event_values:
        raise ValueError("workforce transaction cannot be empty")
    if any(event.event_type not in _WORKFORCE_EVENTS for event in event_values):
        raise ValueError("workforce transaction contains a non-workforce event")
    validate_workforce_frame_events(event_values)
    isolated = _clone_projection_state(state)
    for event in event_values:
        apply_workforce_event(isolated, event)
    return isolated


def commit_workforce_events(
    writer: JournalWriter,
    state: ProjectionState,
    events: Iterable[JournalEvent],
) -> CommitResult:
    """Validate, durably commit, and apply one serialized workforce transaction."""

    event_values = tuple(events)
    with workforce_projection_guard(), writer.transaction_guard():
        return commit_workforce_events_unlocked(writer, state, event_values)


def commit_workforce_events_unlocked(
    writer: JournalWriter,
    state: ProjectionState,
    events: Iterable[JournalEvent],
) -> CommitResult:
    """Commit while the caller holds workforce then Journal transaction guards."""

    from ..journal.projections import apply_frame

    event_values = tuple(events)
    last_frame = writer.get_last_frame()
    writer_sequence = 0 if last_frame is None else last_frame.sequence
    writer_hash = "" if last_frame is None else last_frame.frame_hash
    if state.cursor_sequence != writer_sequence or state.cursor_hash != writer_hash:
        raise _projection_error("workforce projection is stale")
    validate_workforce_events(state, event_values)
    aggregate_ids = {event.aggregate_id for event in event_values}
    expected_versions = writer.get_aggregate_versions(aggregate_ids)
    result = writer.commit(
        event_values,
        expected_versions,
        expected_head_sequence=state.cursor_sequence,
        expected_head_hash=state.cursor_hash,
    )
    if result.state is not CommitState.ANCHORED:
        raise AnchorMismatchError("workforce commit was not anchored")
    apply_frame(state, result.frame)
    return result
