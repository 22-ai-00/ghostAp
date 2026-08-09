"""Durable single-source registry for managed Feishu groups.

The registry deliberately owns no Feishu, Project, retired team model, or Employee runtime
dependencies.  Callers run those operations as a saga and only hold this
registry's leaf lock while validating and replacing its JSON snapshot.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import secrets
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import (
    ManagedGroupOrigin,
    ManagedGroupRecord,
    ManagedGroupStatus,
    ProjectGrant,
)

_SCHEMA = "ghostap.managed_groups"
_VERSION = 1
_TOP_LEVEL_KEYS = {
    "groups",
    "migration_dispositions",
    "provision_intents",
    "revoke_intents",
    "revision",
    "schema",
    "version",
}
_INTENT_KEYS = {
    "canonical_root_ref",
    "create_dispatched_at",
    "create_state",
    "created_at",
    "operation_uuid",
    "origin",
    "owner_id",
    "project_id",
    "receiving_bot_ref",
    "remote_chat_id",
}
_LEGACY_INTENT_KEYS = _INTENT_KEYS - {
    "create_dispatched_at",
    "create_state",
    "operation_uuid",
    "remote_chat_id",
}
_V1_INTENT_KEYS = _INTENT_KEYS - {
    "create_dispatched_at",
    "create_state",
    "operation_uuid",
}
_ENTRY_KEYS = {"grant", "record"}
_RECORD_KEYS = {
    "canonical_root_ref",
    "chat_id",
    "created_at",
    "origin",
    "owner_id",
    "project_grant_id",
    "project_id",
    "receiving_bot_ref",
    "revision",
    "status",
}
_GRANT_KEYS = {
    "backend_binding_ids",
    "canonical_root_ref",
    "connected_target_refs",
    "grant_id",
    "managed_group_id",
    "owner_id",
    "project_id",
    "revision",
}
_REVOKE_KEYS = {"requested_at"}
_MIGRATION_KEYS = {"project_id", "reported", "status"}
_LEGACY_MIGRATION_KEYS = _MIGRATION_KEYS - {"reported"}


class ManagedGroupRegistryError(RuntimeError):
    """Base error for registry validation, conflicts, and persistence."""


class RegistryCorruptionError(ManagedGroupRegistryError):
    """The on-disk registry is malformed or uses an unsupported schema."""


class RegistryCommitUncertainError(ManagedGroupRegistryError):
    """A target replace may have committed but durability is not confirmed."""

    def __init__(self, message: str, *, committed: bool) -> None:
        super().__init__(message)
        self.committed = committed


class ManagedGroupConflictError(ManagedGroupRegistryError):
    """A retry conflicts with an already persisted lifecycle fact."""


class ManagedGroupValidationError(ManagedGroupRegistryError):
    """External migration/adoption validation did not establish provenance."""


@dataclass(frozen=True, slots=True)
class ManagedGroupProvisionBinding:
    provision_id: str
    chat_id: str
    project_id: str
    canonical_root_ref: str
    origin: ManagedGroupOrigin
    owner_id: str
    receiving_bot_ref: str


def single_owner_id(value: object) -> str:
    """Normalize the configured single-Owner identity, failing closed."""

    if isinstance(value, str):
        identities = tuple(
            item.strip()
            for item in value.replace(";", ",").split(",")
            if item.strip()
        )
    elif isinstance(value, (set, frozenset, list, tuple)):
        identities = tuple(
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        )
    else:
        return ""
    unique = frozenset(identities)
    return next(iter(unique)) if len(unique) == 1 else ""


def _require_dict(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RegistryCorruptionError(f"invalid {label} schema")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryCorruptionError(f"invalid {label}")
    return value


def _require_revision(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RegistryCorruptionError(f"invalid {label}")
    return value


def _parse_datetime(value: Any, label: str) -> datetime:
    raw = _require_string(value, label)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise RegistryCorruptionError(f"invalid {label}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RegistryCorruptionError(f"invalid {label}")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryCorruptionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _record_to_dict(record: ManagedGroupRecord) -> dict[str, Any]:
    return {
        "canonical_root_ref": record.canonical_root_ref,
        "chat_id": record.chat_id,
        "created_at": record.created_at.isoformat(),
        "origin": record.origin.value,
        "owner_id": record.owner_id,
        "project_grant_id": record.project_grant_id,
        "project_id": record.project_id,
        "receiving_bot_ref": record.receiving_bot_ref,
        "revision": record.revision,
        "status": record.status.value,
    }


def _grant_to_dict(grant: ProjectGrant) -> dict[str, Any]:
    return {
        "backend_binding_ids": list(grant.backend_binding_ids),
        "canonical_root_ref": grant.canonical_root_ref,
        "connected_target_refs": list(grant.connected_target_refs),
        "grant_id": grant.grant_id,
        "managed_group_id": grant.managed_group_id,
        "owner_id": grant.owner_id,
        "project_id": grant.project_id,
        "revision": grant.revision,
    }


def _record_from_dict(value: Any) -> ManagedGroupRecord:
    data = _require_dict(value, _RECORD_KEYS, "managed group record")
    try:
        status = ManagedGroupStatus(_require_string(data["status"], "record status"))
        origin = ManagedGroupOrigin(_require_string(data["origin"], "record origin"))
    except ValueError as exc:
        raise RegistryCorruptionError("invalid managed group enum") from exc
    return ManagedGroupRecord(
        chat_id=_require_string(data["chat_id"], "record chat_id"),
        revision=_require_revision(data["revision"], "record revision"),
        status=status,
        owner_id=_require_string(data["owner_id"], "record owner_id"),
        project_id=_require_string(data["project_id"], "record project_id"),
        canonical_root_ref=_require_string(
            data["canonical_root_ref"], "record canonical_root_ref"
        ),
        origin=origin,
        receiving_bot_ref=_require_string(
            data["receiving_bot_ref"], "record receiving_bot_ref"
        ),
        project_grant_id=_require_string(
            data["project_grant_id"], "record project_grant_id"
        ),
        created_at=_parse_datetime(data["created_at"], "record created_at"),
    )


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RegistryCorruptionError(f"invalid {label}")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise RegistryCorruptionError(f"duplicate {label}")
    return result


def _grant_from_dict(value: Any) -> ProjectGrant:
    data = _require_dict(value, _GRANT_KEYS, "project grant")
    return ProjectGrant(
        grant_id=_require_string(data["grant_id"], "grant grant_id"),
        revision=_require_revision(data["revision"], "grant revision"),
        owner_id=_require_string(data["owner_id"], "grant owner_id"),
        managed_group_id=_require_string(
            data["managed_group_id"], "grant managed_group_id"
        ),
        project_id=_require_string(data["project_id"], "grant project_id"),
        canonical_root_ref=_require_string(
            data["canonical_root_ref"], "grant canonical_root_ref"
        ),
        backend_binding_ids=_string_tuple(
            data["backend_binding_ids"], "grant backend_binding_ids"
        ),
        connected_target_refs=_string_tuple(
            data["connected_target_refs"], "grant connected_target_refs"
        ),
    )


class ManagedGroupRegistry:
    """Thread-safe durable provenance and grant registry.

    ``storage_path`` has no default by design.  Production and tests must
    inject the exact location rather than relying on cwd or process globals.
    """

    def __init__(self, storage_path: str | os.PathLike[str]) -> None:
        if not isinstance(storage_path, (str, os.PathLike)):
            raise TypeError("storage_path must be path-like")
        self._path = Path(storage_path)
        self._lock_path = self._path.with_name(f".{self._path.name}.lock")
        self._uncertain_path = self._path.with_name(
            f".{self._path.name}.commit-uncertain"
        )
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._revision = 0
        self._intents: dict[str, dict[str, Any]] = {}
        self._records: dict[str, ManagedGroupRecord] = {}
        self._grants: dict[str, ProjectGrant] = {}
        self._revokes: dict[str, dict[str, str]] = {}
        self._migration_dispositions: dict[str, dict[str, Any]] = {}
        self._commit_uncertain = False
        self._validate_storage_path()
        with self._disk_transaction():
            pass

    @property
    def storage_path(self) -> Path:
        return self._path

    @property
    def revision(self) -> int:
        with self._disk_transaction():
            return self._revision

    def managed_groups(self) -> tuple[ManagedGroupRecord, ...]:
        with self._disk_transaction():
            return tuple(self._records[key] for key in sorted(self._records))

    def project_grants(self) -> tuple[ProjectGrant, ...]:
        with self._disk_transaction():
            if self._commit_uncertain:
                return ()
            active_ids = {
                record.project_grant_id
                for record in self._records.values()
                if record.status is ManagedGroupStatus.ACTIVE
                and record.chat_id not in self._revokes
            }
            return tuple(
                self._grants[key]
                for key in sorted(self._grants)
                if key in active_ids
            )

    def record(self, chat_id: str) -> ManagedGroupRecord | None:
        with self._disk_transaction():
            return self._records.get(chat_id)

    def active_record(self, chat_id: str) -> ManagedGroupRecord | None:
        with self._disk_transaction():
            if self._commit_uncertain or chat_id in self._revokes:
                return None
            record = self._records.get(chat_id)
            if record is None or record.status is not ManagedGroupStatus.ACTIVE:
                return None
            return record

    def grant_for_chat(self, chat_id: str) -> ProjectGrant | None:
        with self._disk_transaction():
            record = self._records.get(chat_id)
            if (
                record is None
                or record.status is not ManagedGroupStatus.ACTIVE
                or chat_id in self._revokes
                or self._commit_uncertain
            ):
                return None
            return self._grants.get(record.project_grant_id)

    def trust_snapshot(
        self,
        chat_id: str,
    ) -> tuple[ManagedGroupRecord | None, ProjectGrant | None]:
        """Read one current group/grant pair in a single disk transaction."""

        with self._disk_transaction():
            if self._commit_uncertain or chat_id in self._revokes:
                return None, None
            record = self._records.get(chat_id)
            if record is None or record.status is not ManagedGroupStatus.ACTIVE:
                return None, None
            return record, self._grants.get(record.project_grant_id)

    def begin_provision(
        self,
        *,
        provision_id: str,
        owner_id: str,
        origin: ManagedGroupOrigin,
        receiving_bot_ref: str,
        project_id: str,
        canonical_root_ref: str,
        created_at: datetime,
    ) -> str:
        if not isinstance(origin, ManagedGroupOrigin):
            raise TypeError("origin must be ManagedGroupOrigin")
        key = self._runtime_string(provision_id, "provision_id")
        facts = {
            "canonical_root_ref": self._runtime_string(
                canonical_root_ref, "canonical_root_ref"
            ),
            "create_dispatched_at": None,
            "create_state": "prepared",
            "created_at": self._runtime_datetime(created_at).isoformat(),
            "origin": origin.value,
            "owner_id": self._runtime_string(owner_id, "owner_id"),
            "operation_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
            "project_id": self._runtime_string(project_id, "project_id"),
            "receiving_bot_ref": self._runtime_string(
                receiving_bot_ref, "receiving_bot_ref"
            ),
            "remote_chat_id": None,
        }
        with self._disk_transaction():
            existing = self._intents.get(key)
            if existing is not None:
                stable_existing = {
                    name: value
                    for name, value in existing.items()
                    if name not in {
                        "create_dispatched_at",
                        "create_state",
                        "created_at",
                        "remote_chat_id",
                    }
                }
                stable_retry = {
                    name: value
                    for name, value in facts.items()
                    if name not in {
                        "create_dispatched_at",
                        "create_state",
                        "created_at",
                        "remote_chat_id",
                    }
                }
                if stable_existing != stable_retry:
                    raise ManagedGroupConflictError("provision retry changed facts")
                return key
            self._intents[key] = facts
            try:
                self._persist_locked()
            except BaseException:
                self._intents.pop(key, None)
                raise
            return key

    def provision_chat_id(self, provision_id: str) -> str | None:
        key = self._runtime_string(provision_id, "provision_id")
        with self._disk_transaction():
            intent = self._intents.get(key)
            if intent is None:
                return None
            remote_chat_id = intent.get("remote_chat_id")
            return remote_chat_id if isinstance(remote_chat_id, str) else None

    def provision_binding(
        self,
        provision_id: str,
    ) -> ManagedGroupProvisionBinding | None:
        key = self._runtime_string(provision_id, "provision_id")
        with self._disk_transaction():
            intent = self._intents.get(key)
            if intent is None:
                return None
            chat_id = intent.get("remote_chat_id")
            if not isinstance(chat_id, str):
                return None
            return ManagedGroupProvisionBinding(
                provision_id=key,
                chat_id=chat_id,
                project_id=intent["project_id"],
                canonical_root_ref=intent["canonical_root_ref"],
                origin=ManagedGroupOrigin(intent["origin"]),
                owner_id=intent["owner_id"],
                receiving_bot_ref=intent["receiving_bot_ref"],
            )

    def provision_create_state(self, provision_id: str) -> str | None:
        key = self._runtime_string(provision_id, "provision_id")
        with self._disk_transaction():
            intent = self._intents.get(key)
            return None if intent is None else intent["create_state"]

    def prepare_create_dispatch(
        self,
        provision_id: str,
        *,
        dispatched_at: datetime,
    ) -> bool:
        key = self._runtime_string(provision_id, "provision_id")
        now = self._runtime_datetime(dispatched_at)
        with self._disk_transaction():
            intent = self._intents.get(key)
            if intent is None or intent.get("remote_chat_id") is not None:
                return False
            first_raw = intent.get("create_dispatched_at")
            if isinstance(first_raw, str):
                first = datetime.fromisoformat(first_raw)
                if now - first >= timedelta(hours=10):
                    return False
            else:
                intent["create_dispatched_at"] = now.isoformat()
            intent["create_state"] = "dispatched"
            self._persist_locked()
            return True

    def mark_create_outcome_unknown(self, provision_id: str) -> None:
        key = self._runtime_string(provision_id, "provision_id")
        with self._disk_transaction():
            intent = self._intents.get(key)
            if intent is None or intent.get("remote_chat_id") is not None:
                raise ManagedGroupConflictError("unknown or bound provision intent")
            intent["create_state"] = "outcome_unknown"
            self._persist_locked()

    def bind_provision_chat(self, provision_id: str, chat_id: str) -> str:
        key = self._runtime_string(provision_id, "provision_id")
        chat = self._runtime_string(chat_id, "chat_id")
        with self._disk_transaction():
            intent = self._intents.get(key)
            if intent is None:
                raise ManagedGroupConflictError("unknown provision intent")
            existing = intent.get("remote_chat_id")
            if existing == chat:
                return chat
            if existing is not None:
                raise ManagedGroupConflictError(
                    "provision intent is bound to a different chat"
                )
            intent["remote_chat_id"] = chat
            intent["create_state"] = "bound"
            self._persist_locked()
            return chat

    def abandon_provision(self, provision_id: str) -> bool:
        with self._disk_transaction():
            previous = self._intents.pop(provision_id, None)
            if previous is None:
                return False
            try:
                self._persist_locked()
            except BaseException:
                self._intents[provision_id] = previous
                raise
            return True

    def activate(
        self,
        *,
        provision_id: str,
        chat_id: str,
        project_id: str,
        canonical_root_ref: str,
        backend_binding_ids: tuple[str, ...] = (),
        connected_target_refs: tuple[str, ...] = (),
    ) -> tuple[ManagedGroupRecord, ProjectGrant]:
        chat = self._runtime_string(chat_id, "chat_id")
        project = self._runtime_string(project_id, "project_id")
        root = self._runtime_string(canonical_root_ref, "canonical_root_ref")
        backends = self._runtime_tuple(backend_binding_ids, "backend_binding_ids")
        targets = self._runtime_tuple(connected_target_refs, "connected_target_refs")
        with self._disk_transaction():
            intent = self._intents.get(provision_id)
            existing = self._records.get(chat)
            if intent is None:
                if existing is None:
                    raise ManagedGroupConflictError("unknown provision intent")
                grant = self._grants.get(existing.project_grant_id)
                if (
                    existing.status is ManagedGroupStatus.ACTIVE
                    and grant is not None
                    and existing.project_id == project
                    and existing.canonical_root_ref == root
                    and grant.backend_binding_ids == backends
                    and grant.connected_target_refs == targets
                ):
                    return existing, grant
                raise ManagedGroupConflictError("provision intent already consumed")
            if intent["project_id"] != project or intent["canonical_root_ref"] != root:
                raise ManagedGroupConflictError("bound project differs from provision intent")
            bound_chat = intent.get("remote_chat_id")
            if bound_chat is not None and bound_chat != chat:
                raise ManagedGroupConflictError(
                    "provision intent is bound to a different chat"
                )
            if existing is not None:
                if existing.status is ManagedGroupStatus.TOMBSTONED:
                    raise ManagedGroupConflictError("tombstoned chat cannot be reactivated")
                grant = self._grants.get(existing.project_grant_id)
                if grant is None:
                    raise ManagedGroupConflictError("active record has no grant")
                expected = (
                    intent["owner_id"],
                    intent["receiving_bot_ref"],
                    intent["origin"],
                    project,
                    root,
                    backends,
                    targets,
                )
                actual = (
                    existing.owner_id,
                    existing.receiving_bot_ref,
                    existing.origin.value,
                    existing.project_id,
                    existing.canonical_root_ref,
                    grant.backend_binding_ids,
                    grant.connected_target_refs,
                )
                if actual != expected:
                    raise ManagedGroupConflictError("chat already has a different grant")
                old_intent = self._intents.pop(provision_id)
                old_disposition = self._migration_dispositions.pop(chat, None)
                try:
                    self._persist_locked()
                except BaseException:
                    self._intents[provision_id] = old_intent
                    if old_disposition is not None:
                        self._migration_dispositions[chat] = old_disposition
                    raise
                return existing, grant

            revision = self._next_revision_locked()
            grant_id = f"managed-group:{chat}"
            record = ManagedGroupRecord(
                chat_id=chat,
                revision=revision,
                status=ManagedGroupStatus.ACTIVE,
                owner_id=intent["owner_id"],
                project_id=project,
                canonical_root_ref=root,
                origin=ManagedGroupOrigin(intent["origin"]),
                receiving_bot_ref=intent["receiving_bot_ref"],
                project_grant_id=grant_id,
                created_at=datetime.fromisoformat(intent["created_at"]),
            )
            grant = ProjectGrant(
                grant_id=grant_id,
                revision=revision,
                owner_id=record.owner_id,
                managed_group_id=chat,
                project_id=project,
                canonical_root_ref=root,
                backend_binding_ids=backends,
                connected_target_refs=targets,
            )
            old_revision = self._revision
            self._records[chat] = record
            self._grants[grant_id] = grant
            old_intent = self._intents.pop(provision_id)
            old_disposition = self._migration_dispositions.pop(chat, None)
            try:
                self._persist_locked()
            except BaseException:
                self._records.pop(chat, None)
                self._grants.pop(grant_id, None)
                self._intents[provision_id] = old_intent
                if old_disposition is not None:
                    self._migration_dispositions[chat] = old_disposition
                self._revision = old_revision
                raise
            return record, grant

    def register(
        self,
        *,
        chat_id: str,
        owner_id: str,
        origin: ManagedGroupOrigin,
        receiving_bot_ref: str,
        project_id: str,
        canonical_root_ref: str,
        created_at: datetime,
        backend_binding_ids: tuple[str, ...] = (),
        connected_target_refs: tuple[str, ...] = (),
    ) -> tuple[ManagedGroupRecord, ProjectGrant]:
        provision_id = f"register:{chat_id}"
        self.begin_provision(
            provision_id=provision_id,
            owner_id=owner_id,
            origin=origin,
            receiving_bot_ref=receiving_bot_ref,
            project_id=project_id,
            canonical_root_ref=canonical_root_ref,
            created_at=created_at,
        )
        self.bind_provision_chat(provision_id, chat_id)
        return self.activate(
            provision_id=provision_id,
            chat_id=chat_id,
            project_id=project_id,
            canonical_root_ref=canonical_root_ref,
            backend_binding_ids=backend_binding_ids,
            connected_target_refs=connected_target_refs,
        )

    def adopt_existing(
        self,
        *,
        chat_id: str,
        owner_id: str,
        receiving_bot_ref: str,
        project_id: str,
        canonical_root_ref: str,
        created_at: datetime,
        validator: Callable[[Mapping[str, Any]], bool],
        backend_binding_ids: tuple[str, ...] = (),
        connected_target_refs: tuple[str, ...] = (),
    ) -> tuple[ManagedGroupRecord, ProjectGrant]:
        facts = {
            "canonical_root_ref": canonical_root_ref,
            "chat_id": chat_id,
            "created_at": created_at,
            "owner_id": owner_id,
            "project_id": project_id,
            "receiving_bot_ref": receiving_bot_ref,
        }
        if not validator(facts):
            raise ManagedGroupValidationError("group membership/bot validation failed")
        return self.register(
            chat_id=chat_id,
            owner_id=owner_id,
            origin=ManagedGroupOrigin.OWNER_ADOPTED,
            receiving_bot_ref=receiving_bot_ref,
            project_id=project_id,
            canonical_root_ref=canonical_root_ref,
            created_at=created_at,
            backend_binding_ids=backend_binding_ids,
            connected_target_refs=connected_target_refs,
        )

    def import_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        validator: Callable[[Mapping[str, Any]], bool],
    ) -> tuple[ManagedGroupRecord, ProjectGrant] | None:
        required = {
            "bound_chat_created_at",
            "canonical_root_ref",
            "chat_id",
            "owner_id",
            "project_id",
            "receiving_bot_ref",
        }
        if not isinstance(candidate, Mapping) or not required.issubset(candidate):
            return None
        string_fields = (
            "canonical_root_ref",
            "chat_id",
            "owner_id",
            "project_id",
            "receiving_bot_ref",
        )
        if any(
            not isinstance(candidate[field], str) or not candidate[field]
            for field in string_fields
        ):
            return None
        created_at_value = candidate["bound_chat_created_at"]
        if (
            isinstance(created_at_value, bool)
            or not isinstance(created_at_value, (int, float))
            or not math.isfinite(created_at_value)
            or created_at_value <= 0
        ):
            return None
        if not validator(candidate):
            return None
        try:
            created_at = datetime.fromtimestamp(created_at_value).astimezone()
            origin_value = candidate.get(
                "origin", ManagedGroupOrigin.GHOSTAP_CREATED.value
            )
            origin = ManagedGroupOrigin(origin_value)
            return self.register(
                chat_id=candidate["chat_id"],
                owner_id=candidate["owner_id"],
                origin=origin,
                receiving_bot_ref=candidate["receiving_bot_ref"],
                project_id=candidate["project_id"],
                canonical_root_ref=candidate["canonical_root_ref"],
                created_at=created_at,
            )
        except (TypeError, ValueError, ManagedGroupRegistryError):
            return None

    def rotate_receiving_bot(
        self,
        *,
        chat_id: str,
        expected_bot_ref: str,
        new_bot_ref: str,
    ) -> ManagedGroupRecord:
        new_ref = self._runtime_string(new_bot_ref, "new_bot_ref")
        with self._disk_transaction():
            record = self._records.get(chat_id)
            if record is None or record.status is not ManagedGroupStatus.ACTIVE:
                raise ManagedGroupConflictError("chat is not active")
            if record.receiving_bot_ref == new_ref:
                return record
            if record.receiving_bot_ref != expected_bot_ref:
                raise ManagedGroupConflictError("receiving bot rotation is stale")
            old_revision = self._revision
            revision = self._next_revision_locked()
            updated = replace(
                record,
                revision=revision,
                receiving_bot_ref=new_ref,
            )
            self._records[chat_id] = updated
            try:
                self._persist_locked()
            except BaseException:
                self._records[chat_id] = record
                self._revision = old_revision
                raise
            return updated

    def tombstone(self, chat_id: str) -> ManagedGroupRecord:
        with self._disk_transaction():
            record = self._records.get(chat_id)
            if record is None:
                raise ManagedGroupConflictError("unknown managed group")
            if record.status is ManagedGroupStatus.TOMBSTONED:
                return record
            old_revision = self._revision
            revision = self._next_revision_locked()
            updated = replace(
                record,
                revision=revision,
                status=ManagedGroupStatus.TOMBSTONED,
            )
            grant = self._grants.pop(record.project_grant_id, None)
            revoke = self._revokes.pop(chat_id, None)
            self._records[chat_id] = updated
            try:
                self._persist_locked()
            except BaseException:
                self._records[chat_id] = record
                if grant is not None:
                    self._grants[grant.grant_id] = grant
                if revoke is not None:
                    self._revokes[chat_id] = revoke
                self._revision = old_revision
                raise
            return updated

    def begin_revoke(self, chat_id: str, *, requested_at: datetime) -> str:
        chat = self._runtime_string(chat_id, "chat_id")
        timestamp = self._runtime_datetime(requested_at).isoformat()
        with self._disk_transaction():
            record = self._records.get(chat)
            if record is None or record.status is not ManagedGroupStatus.ACTIVE:
                raise ManagedGroupConflictError("chat is not active")
            if chat in self._revokes:
                return chat
            self._revokes[chat] = {"requested_at": timestamp}
            self._persist_locked()
            return chat

    def cancel_revoke(self, chat_id: str) -> bool:
        chat = self._runtime_string(chat_id, "chat_id")
        with self._disk_transaction():
            previous = self._revokes.pop(chat, None)
            if previous is None:
                return False
            self._persist_locked()
            return True

    def pending_revokes(self) -> tuple[str, ...]:
        with self._disk_transaction():
            return tuple(sorted(self._revokes))

    def record_migration_disposition(
        self,
        chat_id: str,
        *,
        project_id: str,
        status: str,
    ) -> None:
        chat = self._runtime_string(chat_id, "chat_id")
        project = self._runtime_string(project_id, "project_id")
        if status not in {"ambiguous", "invalid", "unknown"}:
            raise ValueError("unsupported migration disposition")
        with self._disk_transaction():
            value = {"project_id": project, "reported": False, "status": status}
            existing = self._migration_dispositions.get(chat)
            if existing is not None and (
                existing["project_id"], existing["status"]
            ) == (project, status):
                return
            self._migration_dispositions[chat] = value
            self._persist_locked()

    def migration_dispositions(self) -> tuple[tuple[str, str, str], ...]:
        with self._disk_transaction():
            return tuple(
                (
                    chat_id,
                    value["project_id"],
                    value["status"],
                )
                for chat_id, value in sorted(self._migration_dispositions.items())
            )

    def unreported_migration_dispositions(self) -> tuple[tuple[str, str, str], ...]:
        with self._disk_transaction():
            return tuple(
                (chat_id, value["project_id"], value["status"])
                for chat_id, value in sorted(self._migration_dispositions.items())
                if value["reported"] is False
            )

    def mark_migration_reported(self, chat_id: str) -> bool:
        chat = self._runtime_string(chat_id, "chat_id")
        with self._disk_transaction():
            value = self._migration_dispositions.get(chat)
            if value is None or value["reported"] is True:
                return False
            value["reported"] = True
            try:
                self._persist_locked()
            except BaseException:
                value["reported"] = False
                raise
            return True

    def complete_revoke(self, chat_id: str) -> ManagedGroupRecord:
        chat = self._runtime_string(chat_id, "chat_id")
        with self._disk_transaction():
            if chat not in self._revokes:
                raise ManagedGroupConflictError("chat has no pending revoke")
            record = self._records.get(chat)
            if record is None:
                raise ManagedGroupConflictError("unknown managed group")
            if record.status is ManagedGroupStatus.TOMBSTONED:
                self._revokes.pop(chat, None)
                self._persist_locked()
                return record
            old_revision = self._revision
            revision = self._next_revision_locked()
            updated = replace(
                record,
                revision=revision,
                status=ManagedGroupStatus.TOMBSTONED,
            )
            grant = self._grants.pop(record.project_grant_id, None)
            revoke = self._revokes.pop(chat)
            self._records[chat] = updated
            try:
                self._persist_locked()
            except BaseException:
                self._records[chat] = record
                if grant is not None:
                    self._grants[grant.grant_id] = grant
                self._revokes[chat] = revoke
                self._revision = old_revision
                raise
            return updated

    def _next_revision_locked(self) -> int:
        self._revision += 1
        return self._revision

    @staticmethod
    def _runtime_string(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} is required")
        return value

    @staticmethod
    def _runtime_datetime(value: Any) -> datetime:
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError("created_at must be timezone-aware")
        return value

    @staticmethod
    def _runtime_tuple(value: Any, label: str) -> tuple[str, ...]:
        if not isinstance(value, tuple) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise TypeError(f"{label} must be tuple[str, ...]")
        if len(set(value)) != len(value):
            raise ValueError(f"{label} contains duplicates")
        return value

    def _load(self) -> None:
        self._validate_storage_path()
        if not self._path.exists():
            self._revision = 0
            self._intents = {}
            self._records = {}
            self._grants = {}
            self._revokes = {}
            self._migration_dispositions = {}
            self._commit_uncertain = self._uncertain_path.exists()
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
            legacy_optional = {"revoke_intents", "migration_dispositions"}
            if isinstance(payload, dict) and frozenset(payload) in {
                frozenset(_TOP_LEVEL_KEYS - legacy_optional),
                frozenset(_TOP_LEVEL_KEYS - {"migration_dispositions"}),
            }:
                data = dict(payload)
                data.setdefault("revoke_intents", {})
                data.setdefault("migration_dispositions", {})
            else:
                data = _require_dict(payload, _TOP_LEVEL_KEYS, "registry")
            if (
                data["schema"] != _SCHEMA
                or isinstance(data["version"], bool)
                or not isinstance(data["version"], int)
                or data["version"] != _VERSION
            ):
                raise RegistryCorruptionError("unsupported registry schema/version")
            revision = _require_revision(
                data["revision"], "registry revision", allow_zero=True
            )
            intents_raw = data["provision_intents"]
            revokes_raw = data["revoke_intents"]
            migrations_raw = data["migration_dispositions"]
            groups_raw = data["groups"]
            if (
                not isinstance(intents_raw, dict)
                or not isinstance(revokes_raw, dict)
                or not isinstance(migrations_raw, dict)
                or not isinstance(groups_raw, dict)
            ):
                raise RegistryCorruptionError("invalid registry collections")

            intents: dict[str, dict[str, Any]] = {}
            records: dict[str, ManagedGroupRecord] = {}
            grants: dict[str, ProjectGrant] = {}
            revokes: dict[str, dict[str, str]] = {}
            migrations: dict[str, dict[str, Any]] = {}
            for provision_id, raw_intent in intents_raw.items():
                _require_string(provision_id, "provision id")
                if not isinstance(raw_intent, dict) or frozenset(raw_intent) not in {
                    frozenset(_INTENT_KEYS),
                    frozenset(_LEGACY_INTENT_KEYS),
                    frozenset(_V1_INTENT_KEYS),
                }:
                    raise RegistryCorruptionError("invalid provision intent schema")
                intent = dict(raw_intent)
                intent.setdefault("create_dispatched_at", None)
                intent.setdefault(
                    "create_state",
                    "bound" if intent.get("remote_chat_id") else "prepared",
                )
                intent.setdefault(
                    "operation_uuid",
                    str(uuid.uuid5(uuid.NAMESPACE_URL, provision_id)),
                )
                intent.setdefault("remote_chat_id", None)
                _parse_datetime(intent["created_at"], "intent created_at")
                try:
                    ManagedGroupOrigin(intent["origin"])
                except (TypeError, ValueError) as exc:
                    raise RegistryCorruptionError("invalid intent origin") from exc
                for key in _INTENT_KEYS - {
                    "create_dispatched_at",
                    "created_at",
                    "origin",
                    "remote_chat_id",
                }:
                    _require_string(intent[key], f"intent {key}")
                if intent["create_state"] not in {
                    "bound",
                    "dispatched",
                    "outcome_unknown",
                    "prepared",
                }:
                    raise RegistryCorruptionError("invalid intent create_state")
                if intent["create_dispatched_at"] is not None:
                    _parse_datetime(
                        intent["create_dispatched_at"],
                        "intent create_dispatched_at",
                    )
                if intent["remote_chat_id"] is not None:
                    _require_string(
                        intent["remote_chat_id"], "intent remote_chat_id"
                    )
                intents[provision_id] = dict(intent)

            max_revision = 0
            for chat_id, raw_entry in groups_raw.items():
                _require_string(chat_id, "group key")
                entry = _require_dict(raw_entry, _ENTRY_KEYS, "group entry")
                record = _record_from_dict(entry["record"])
                if record.chat_id != chat_id or record.chat_id in records:
                    raise RegistryCorruptionError("group key/record mismatch")
                max_revision = max(max_revision, record.revision)
                raw_grant = entry["grant"]
                if record.status is ManagedGroupStatus.ACTIVE:
                    if raw_grant is None:
                        raise RegistryCorruptionError("active group is missing grant")
                    grant = _grant_from_dict(raw_grant)
                    if not self._grant_matches_record(grant, record):
                        raise RegistryCorruptionError("grant/group mismatch")
                    if grant.grant_id in grants:
                        raise RegistryCorruptionError("duplicate grant id")
                    grants[grant.grant_id] = grant
                    max_revision = max(max_revision, grant.revision)
                elif raw_grant is not None:
                    raise RegistryCorruptionError("tombstone must not retain active grant")
                records[chat_id] = record
            for chat_id, raw_revoke in revokes_raw.items():
                _require_string(chat_id, "revoke chat_id")
                revoke = _require_dict(raw_revoke, _REVOKE_KEYS, "revoke intent")
                _parse_datetime(revoke["requested_at"], "revoke requested_at")
                record = records.get(chat_id)
                if record is None or record.status is not ManagedGroupStatus.ACTIVE:
                    raise RegistryCorruptionError("revoke intent has no active group")
                revokes[chat_id] = dict(revoke)
            for chat_id, raw_disposition in migrations_raw.items():
                _require_string(chat_id, "migration chat_id")
                if not isinstance(raw_disposition, dict) or frozenset(
                    raw_disposition
                ) not in {
                    frozenset(_MIGRATION_KEYS),
                    frozenset(_LEGACY_MIGRATION_KEYS),
                }:
                    raise RegistryCorruptionError("invalid migration disposition schema")
                disposition = dict(raw_disposition)
                disposition.setdefault("reported", False)
                _require_string(disposition["project_id"], "migration project_id")
                if type(disposition["reported"]) is not bool:
                    raise RegistryCorruptionError("invalid migration reported flag")
                if disposition["status"] not in {"ambiguous", "invalid", "unknown"}:
                    raise RegistryCorruptionError("invalid migration status")
                migrations[chat_id] = dict(disposition)
            if revision < max_revision:
                raise RegistryCorruptionError("registry revision moved backwards")
        except RegistryCorruptionError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RegistryCorruptionError("cannot read managed group registry") from exc

        self._revision = revision
        self._intents = intents
        self._records = records
        self._grants = grants
        self._revokes = revokes
        self._migration_dispositions = migrations
        self._commit_uncertain = self._uncertain_path.exists()

    @staticmethod
    def _grant_matches_record(
        grant: ProjectGrant,
        record: ManagedGroupRecord,
    ) -> bool:
        return (
            grant.grant_id == record.project_grant_id
            and grant.owner_id == record.owner_id
            and grant.managed_group_id == record.chat_id
            and grant.project_id == record.project_id
            and grant.canonical_root_ref == record.canonical_root_ref
        )

    def _persist_locked(self) -> None:
        if self._commit_uncertain:
            raise RegistryCommitUncertainError(
                "registry has an unresolved commit",
                committed=True,
            )
        payload = {
            "groups": {
                chat_id: {
                    "grant": (
                        _grant_to_dict(self._grants[record.project_grant_id])
                        if record.status is ManagedGroupStatus.ACTIVE
                        else None
                    ),
                    "record": _record_to_dict(record),
                }
                for chat_id, record in sorted(self._records.items())
            },
            "migration_dispositions": {
                key: dict(value)
                for key, value in sorted(self._migration_dispositions.items())
            },
            "provision_intents": {
                key: dict(value) for key, value in sorted(self._intents.items())
            },
            "revoke_intents": {
                key: dict(value) for key, value in sorted(self._revokes.items())
            },
            "revision": self._revision,
            "schema": _SCHEMA,
            "version": _VERSION,
        }
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._validate_storage_path()
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        previous_raw = self._path.read_bytes() if self._path.exists() else None
        self._anchor_uncertain_locked(raw, previous_raw)
        temp_path = parent / f".{self._path.name}.{secrets.token_hex(12)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = -1
        replaced = False
        try:
            fd = os.open(temp_path, flags, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = -1
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
            replaced = True
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._clear_uncertain_locked()
        except OSError as exc:
            if replaced:
                self._commit_uncertain = True
                raise RegistryCommitUncertainError(
                    "registry replace committed but durability is uncertain",
                    committed=True,
                ) from exc
            self._clear_uncertain_best_effort_locked()
            raise
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    def _anchor_uncertain_locked(
        self,
        desired_raw: bytes,
        previous_raw: bytes | None,
    ) -> None:
        payload = {
            "desired_sha256": hashlib.sha256(desired_raw).hexdigest(),
            "previous_sha256": (
                hashlib.sha256(previous_raw).hexdigest()
                if previous_raw is not None
                else None
            ),
            "schema": "ghostap.managed_groups.commit_uncertain",
            "version": 1,
        }
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        temp_path = self._path.parent / (
            f".{self._uncertain_path.name}.{secrets.token_hex(12)}.tmp"
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temp_path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = -1
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._uncertain_path)
            self._fsync_parent_locked()
        except OSError:
            self._clear_uncertain_best_effort_locked()
            raise
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    def _clear_uncertain_locked(self) -> None:
        self._uncertain_path.unlink()
        self._fsync_parent_locked()
        self._commit_uncertain = False

    def _clear_uncertain_best_effort_locked(self) -> None:
        try:
            self._uncertain_path.unlink()
            self._fsync_parent_locked()
        except OSError:
            pass
        self._commit_uncertain = self._uncertain_path.exists()

    def _fsync_parent_locked(self) -> None:
        directory_fd = os.open(
            self._path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def reconcile_uncertain_commit(self) -> bool:
        """Resolve an anchored pre/post-replace crash state before reuse."""

        with self._disk_transaction():
            if not self._uncertain_path.exists():
                self._commit_uncertain = False
                return False
            if self._uncertain_path.is_symlink():
                raise RegistryCorruptionError("unsafe uncertain commit marker")
            try:
                marker = json.loads(self._uncertain_path.read_text("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RegistryCorruptionError(
                    "cannot read uncertain commit marker"
                ) from exc
            if (
                not isinstance(marker, dict)
                or set(marker)
                != {
                    "desired_sha256",
                    "previous_sha256",
                    "schema",
                    "version",
                }
                or marker["schema"]
                != "ghostap.managed_groups.commit_uncertain"
                or marker["version"] != 1
            ):
                raise RegistryCorruptionError("invalid uncertain commit marker")
            current_raw = self._path.read_bytes() if self._path.exists() else None
            current_hash = (
                hashlib.sha256(current_raw).hexdigest()
                if current_raw is not None
                else None
            )
            if current_hash not in {
                marker["desired_sha256"],
                marker["previous_sha256"],
            }:
                raise RegistryCorruptionError(
                    "uncertain commit target does not match either state"
                )
            self._fsync_parent_locked()
            self._clear_uncertain_locked()
            self._load()
            return current_hash == marker["desired_sha256"]

    def _validate_storage_path(self) -> None:
        if (
            self._path.is_symlink()
            or self._uncertain_path.is_symlink()
            or self._path.parent.is_symlink()
        ):
            raise RegistryCorruptionError("registry path must not use symlinks")

    @contextmanager
    def _disk_transaction(self):
        """Reload state under a stable lock before each disk transaction."""

        with self._lock:
            self._validate_storage_path()
            parent = self._path.parent
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                lock_fd = os.open(self._lock_path, flags, 0o600)
            except OSError as exc:
                raise RegistryCorruptionError("cannot open registry lock") from exc
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                self._load()
                yield
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)
    "operation_uuid",
