"""Durable single-source registry for managed Feishu groups.

The registry deliberately owns no Feishu, Project, Slock, or Employee runtime
dependencies.  Callers run those operations as a saga and only hold this
registry's leaf lock while validating and replacing its JSON snapshot.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
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
    "provision_intents",
    "revision",
    "schema",
    "version",
}
_INTENT_KEYS = {
    "canonical_root_ref",
    "created_at",
    "origin",
    "owner_id",
    "project_id",
    "receiving_bot_ref",
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


class ManagedGroupRegistryError(RuntimeError):
    """Base error for registry validation, conflicts, and persistence."""


class RegistryCorruptionError(ManagedGroupRegistryError):
    """The on-disk registry is malformed or uses an unsupported schema."""


class ManagedGroupConflictError(ManagedGroupRegistryError):
    """A retry conflicts with an already persisted lifecycle fact."""


class ManagedGroupValidationError(ManagedGroupRegistryError):
    """External migration/adoption validation did not establish provenance."""


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
        self._lock = threading.RLock()  # leaf lock: no external calls while held
        self._revision = 0
        self._intents: dict[str, dict[str, Any]] = {}
        self._records: dict[str, ManagedGroupRecord] = {}
        self._grants: dict[str, ProjectGrant] = {}
        self._load()

    @property
    def storage_path(self) -> Path:
        return self._path

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def managed_groups(self) -> tuple[ManagedGroupRecord, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def project_grants(self) -> tuple[ProjectGrant, ...]:
        with self._lock:
            active_ids = {
                record.project_grant_id
                for record in self._records.values()
                if record.status is ManagedGroupStatus.ACTIVE
            }
            return tuple(
                self._grants[key]
                for key in sorted(self._grants)
                if key in active_ids
            )

    def record(self, chat_id: str) -> ManagedGroupRecord | None:
        with self._lock:
            return self._records.get(chat_id)

    def active_record(self, chat_id: str) -> ManagedGroupRecord | None:
        record = self.record(chat_id)
        if record is None or record.status is not ManagedGroupStatus.ACTIVE:
            return None
        return record

    def grant_for_chat(self, chat_id: str) -> ProjectGrant | None:
        with self._lock:
            record = self._records.get(chat_id)
            if record is None or record.status is not ManagedGroupStatus.ACTIVE:
                return None
            return self._grants.get(record.project_grant_id)

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
        facts = {
            "canonical_root_ref": self._runtime_string(
                canonical_root_ref, "canonical_root_ref"
            ),
            "created_at": self._runtime_datetime(created_at).isoformat(),
            "origin": origin.value,
            "owner_id": self._runtime_string(owner_id, "owner_id"),
            "project_id": self._runtime_string(project_id, "project_id"),
            "receiving_bot_ref": self._runtime_string(
                receiving_bot_ref, "receiving_bot_ref"
            ),
        }
        key = self._runtime_string(provision_id, "provision_id")
        with self._lock:
            existing = self._intents.get(key)
            if existing is not None:
                stable_existing = {
                    name: value
                    for name, value in existing.items()
                    if name != "created_at"
                }
                stable_retry = {
                    name: value
                    for name, value in facts.items()
                    if name != "created_at"
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

    def abandon_provision(self, provision_id: str) -> bool:
        with self._lock:
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
        with self._lock:
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
                try:
                    self._persist_locked()
                except BaseException:
                    self._intents[provision_id] = old_intent
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
            try:
                self._persist_locked()
            except BaseException:
                self._records.pop(chat, None)
                self._grants.pop(grant_id, None)
                self._intents[provision_id] = old_intent
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
        with self._lock:
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
        with self._lock:
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
            self._records[chat_id] = updated
            try:
                self._persist_locked()
            except BaseException:
                self._records[chat_id] = record
                if grant is not None:
                    self._grants[grant.grant_id] = grant
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
        if not isinstance(value, datetime) or value.tzinfo is None:
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
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
            data = _require_dict(payload, _TOP_LEVEL_KEYS, "registry")
            if data["schema"] != _SCHEMA or data["version"] != _VERSION:
                raise RegistryCorruptionError("unsupported registry schema/version")
            revision = _require_revision(
                data["revision"], "registry revision", allow_zero=True
            )
            intents_raw = data["provision_intents"]
            groups_raw = data["groups"]
            if not isinstance(intents_raw, dict) or not isinstance(groups_raw, dict):
                raise RegistryCorruptionError("invalid registry collections")

            intents: dict[str, dict[str, Any]] = {}
            records: dict[str, ManagedGroupRecord] = {}
            grants: dict[str, ProjectGrant] = {}
            for provision_id, raw_intent in intents_raw.items():
                _require_string(provision_id, "provision id")
                intent = _require_dict(raw_intent, _INTENT_KEYS, "provision intent")
                _parse_datetime(intent["created_at"], "intent created_at")
                try:
                    ManagedGroupOrigin(intent["origin"])
                except (TypeError, ValueError) as exc:
                    raise RegistryCorruptionError("invalid intent origin") from exc
                for key in _INTENT_KEYS - {"created_at", "origin"}:
                    _require_string(intent[key], f"intent {key}")
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
            "provision_intents": {
                key: dict(value) for key, value in sorted(self._intents.items())
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
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_path = parent / f".{self._path.name}.{secrets.token_hex(12)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = -1
        try:
            fd = os.open(temp_path, flags, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = -1
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._path)
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
