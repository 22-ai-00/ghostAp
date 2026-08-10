"""Authorized employee-scoped attachment staging.

The WebSocket ACK path never calls this module.  It stores typed resource
descriptors in the encrypted ingress Blob.  A later authority gate constructs
``AuthorizedAttachmentStagingRequest`` and invokes this service explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import secrets
import stat
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from lark_oapi.api.im.v1 import GetMessageResourceRequest

from src.utils.path import canonicalize_user_home_path

from ..journal.frame import GENESIS_HASH, JournalEvent
from ..journal.writer import CommitState, JournalWriter
from .models import (
    MAX_INGRESS_ATTACHMENT_BYTES,
    MAX_INGRESS_ATTACHMENT_COUNT,
    MAX_INGRESS_TOTAL_ATTACHMENT_BYTES,
)

_CANONICAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_RESOURCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,511}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MIME = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z")
_STORAGE_NAME = re.compile(r"[a-z0-9]{1,128}\Z")
_RESOURCE_TYPES = frozenset({"file", "image"})
_DANGEROUS_SUFFIXES = frozenset(
    {
        ".app",
        ".bat",
        ".cmd",
        ".com",
        ".cpl",
        ".dll",
        ".dmg",
        ".exe",
        ".gadget",
        ".hta",
        ".jar",
        ".js",
        ".jse",
        ".lnk",
        ".msi",
        ".msp",
        ".pif",
        ".ps1",
        ".reg",
        ".scr",
        ".sh",
        ".vb",
        ".vbe",
        ".vbs",
        ".wsf",
    }
)
_ZIP_MIMES = frozenset(
    {
        "application/epub+zip",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    }
)
_EVENT_TYPES = frozenset(
    {
        "employee.ingress.attachment_staging_started",
        "employee.ingress.attachment_staging_parent_bound",
        "employee.ingress.attachment_staging_leaf_prepared",
        "employee.ingress.attachment_staging_completed",
        "employee.ingress.attachment_staging_failed",
        "employee.ingress.attachment_cleanup_started",
        "employee.ingress.attachment_cleanup_completed",
    }
)


class AttachmentError(RuntimeError):
    """Base class with deliberately redacted public messages."""


class AttachmentPolicyError(AttachmentError):
    """Descriptor count or declared-size policy failed."""


class AttachmentTimeoutError(AttachmentError):
    """The official resource download exceeded its bounded deadline."""


class AttachmentValidationError(AttachmentError):
    """Downloaded bytes do not match the accepted descriptor."""


class AttachmentStorageError(AttachmentError):
    """The rooted no-follow staging boundary could not be maintained."""


class AttachmentCredentialError(AttachmentError):
    """The target employee credential could not be leased."""


class AttachmentDownloadError(AttachmentError):
    """The official employee resource request failed."""


class AttachmentStateError(AttachmentError):
    """Journal staging state is inconsistent or not anchored."""


class CredentialResolver(Protocol):
    def resolve(self, credential_ref: str, agent_id: str, app_id: str) -> str: ...


class AttachmentDownloader(Protocol):
    def download(self, descriptor: EmployeeAttachmentDescriptor) -> DownloadedAttachment: ...


def _strict_string(value: object, name: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _canonical_id(value: object, name: str, prefix: str) -> str:
    result = _strict_string(value, name, maximum=256)
    if not result.startswith(prefix) or _CANONICAL_ID.fullmatch(result) is None:
        raise ValueError(f"{name} must use {prefix} identifier space")
    return result


def _official_resource_id(value: object) -> str:
    result = _strict_string(value, "resource_id", maximum=512)
    if _RESOURCE_ID.fullmatch(result) is None:
        raise ValueError("resource_id must be an official SDK key")
    return result


def _exact(value: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise TypeError(f"{name} must use the exact schema")
    return value


@dataclass(frozen=True, slots=True)
class EmployeeAttachmentDescriptor:
    """Typed SDK resource coordinate accepted before staging authorization."""

    schema_version: int
    message_id: str
    resource_type: str
    resource_id: str
    declared_mime_type: str
    declared_size_bytes: int
    declared_sha256: str
    user_filename: str

    _FIELDS = frozenset(
        {
            "schema_version",
            "message_id",
            "resource_type",
            "resource_id",
            "declared_mime_type",
            "declared_size_bytes",
            "declared_sha256",
            "user_filename",
        }
    )

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported attachment descriptor schema_version")
        object.__setattr__(self, "message_id", _canonical_id(self.message_id, "message_id", "om_"))
        if self.resource_type not in _RESOURCE_TYPES:
            raise ValueError("resource_type must be file or image")
        object.__setattr__(self, "resource_id", _official_resource_id(self.resource_id))
        expected_prefix = "img_" if self.resource_type == "image" else "file_"
        if not self.resource_id.startswith(expected_prefix):
            raise ValueError("resource_id does not match resource_type")
        mime = _strict_string(self.declared_mime_type, "declared_mime_type", maximum=255).casefold()
        if _MIME.fullmatch(mime) is None:
            raise ValueError("declared_mime_type is invalid")
        object.__setattr__(self, "declared_mime_type", mime)
        if (
            isinstance(self.declared_size_bytes, bool)
            or not isinstance(self.declared_size_bytes, int)
            or self.declared_size_bytes < 0
            or self.declared_size_bytes > MAX_INGRESS_ATTACHMENT_BYTES
        ):
            raise ValueError("declared_size_bytes is invalid")
        if not isinstance(self.declared_sha256, str) or _SHA256.fullmatch(self.declared_sha256) is None:
            raise ValueError("declared_sha256 must be lowercase sha256")
        _strict_string(self.user_filename, "user_filename", maximum=1024)

    def to_dict(self) -> dict[str, object]:
        return {field_name: getattr(self, field_name) for field_name in self._FIELDS}

    @classmethod
    def from_dict(cls, value: object) -> EmployeeAttachmentDescriptor:
        return cls(**_exact(value, cls._FIELDS, "attachment descriptor"))


@dataclass(frozen=True, slots=True)
class AuthorizedAttachmentStagingRequest:
    """Authority-gate output; raw event JSON must never construct this value."""

    schema_version: int
    acceptance_id: str
    envelope_id: str
    tenant_key: str
    agent_id: str
    app_id: str
    credential_ref: str
    descriptors: tuple[EmployeeAttachmentDescriptor, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported attachment request schema_version")
        object.__setattr__(self, "acceptance_id", _canonical_id(self.acceptance_id, "acceptance_id", "acc_"))
        object.__setattr__(self, "envelope_id", _canonical_id(self.envelope_id, "envelope_id", "ing_"))
        _strict_string(self.tenant_key, "tenant_key", maximum=512)
        object.__setattr__(self, "agent_id", _canonical_id(self.agent_id, "agent_id", "agt_"))
        object.__setattr__(self, "app_id", _canonical_id(self.app_id, "app_id", "cli_"))
        _strict_string(self.credential_ref, "credential_ref", maximum=512)
        values = tuple(self.descriptors)
        if not all(isinstance(item, EmployeeAttachmentDescriptor) for item in values):
            raise TypeError("descriptors must contain EmployeeAttachmentDescriptor")
        if not values:
            raise ValueError("descriptors must be non-empty")
        if len(values) > MAX_INGRESS_ATTACHMENT_COUNT:
            raise ValueError("descriptor count exceeds hard maximum")
        object.__setattr__(self, "descriptors", values)


@dataclass(frozen=True, slots=True)
class DownloadedAttachment:
    content: bytes
    file_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("downloaded content must be bytes")
        _strict_string(self.file_name, "downloaded file_name", maximum=1024)


@dataclass(frozen=True, slots=True)
class AttachmentPolicy:
    max_count: int = MAX_INGRESS_ATTACHMENT_COUNT
    max_file_bytes: int = MAX_INGRESS_ATTACHMENT_BYTES
    max_total_bytes: int = MAX_INGRESS_TOTAL_ATTACHMENT_BYTES

    def __post_init__(self) -> None:
        limits = (self.max_count, self.max_file_bytes, self.max_total_bytes)
        hard = (
            MAX_INGRESS_ATTACHMENT_COUNT,
            MAX_INGRESS_ATTACHMENT_BYTES,
            MAX_INGRESS_TOTAL_ATTACHMENT_BYTES,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in limits):
            raise ValueError("attachment policy limits must be integers")
        if any(value < 1 for value in limits) or any(value > ceiling for value, ceiling in zip(limits, hard)):
            raise ValueError("attachment policy limit exceeds hard boundary")

    def validate(self, descriptors: tuple[EmployeeAttachmentDescriptor, ...]) -> None:
        if len(descriptors) > self.max_count:
            raise AttachmentPolicyError("attachment count exceeds policy")
        if any(item.declared_size_bytes > self.max_file_bytes for item in descriptors):
            raise AttachmentPolicyError("attachment per-file size exceeds policy")
        if sum(item.declared_size_bytes for item in descriptors) > self.max_total_bytes:
            raise AttachmentPolicyError("attachment total size exceeds policy")


@dataclass(frozen=True, slots=True)
class AttachmentStagingRecord:
    staging_id: str
    aggregate_id: str
    acceptance_id: str
    envelope_id: str
    tenant_key: str
    agent_id: str
    app_id: str
    relative_paths: tuple[str, ...]
    temporary_paths: tuple[str, ...]
    content_hashes: tuple[str, ...]
    leaf_identities: tuple[tuple[int, int] | None, ...]
    parent_device: int | None = None
    parent_inode: int | None = None
    status: str = "started"
    failure_reason: str = ""
    cleanup_state: str = "none"


@dataclass(slots=True)
class AttachmentStagingState:
    by_staging_id: dict[str, AttachmentStagingRecord] = field(default_factory=dict)
    by_acceptance_id: dict[str, str] = field(default_factory=dict)
    cursor_sequence: int = 0
    cursor_hash: str = ""


def tenant_storage_component(tenant_key: str) -> str:
    """Return a fixed safe path component without exposing the tenant value."""

    _strict_string(tenant_key, "tenant_key", maximum=512)
    return "ten_" + hashlib.sha256(tenant_key.encode("utf-8")).hexdigest()


class LarkEmployeeAttachmentDownloader:
    """Thin official SDK adapter bound to one employee-scoped client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def download(self, descriptor: EmployeeAttachmentDescriptor) -> DownloadedAttachment:
        request = (
            GetMessageResourceRequest.builder()
            .message_id(descriptor.message_id)
            .file_key(descriptor.resource_id)
            .type(descriptor.resource_type)
            .build()
        )
        response = self._client.im.v1.message_resource.get(request)
        if not response.success() or response.file is None:
            raise AttachmentDownloadError("employee resource download failed")
        content = response.file.read(descriptor.declared_size_bytes + 1)
        if not isinstance(content, bytes):
            raise AttachmentDownloadError("employee resource download failed")
        filename = response.file_name
        if not isinstance(filename, str) or not filename:
            filename = descriptor.user_filename
        return DownloadedAttachment(content=content, file_name=filename)


def _default_downloader_builder(*, app_id: str, app_secret: str, timeout: float) -> AttachmentDownloader:
    import lark_oapi as lark

    client = (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(lark.LogLevel.WARNING)
        .timeout(timeout)
        .build()
    )
    return LarkEmployeeAttachmentDownloader(client)


class AttachmentStagingService:
    """Journal-owned staging with root-relative no-follow filesystem access."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        root: str | Path,
        credential_resolver: CredentialResolver,
        policy: AttachmentPolicy | None = None,
        download_timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be JournalWriter")
        if (
            isinstance(download_timeout_seconds, bool)
            or not isinstance(download_timeout_seconds, (int, float))
            or download_timeout_seconds <= 0
            or download_timeout_seconds > 300
        ):
            raise ValueError("download timeout is invalid")
        self._writer = writer
        self._root = canonicalize_user_home_path(root).absolute()
        self._credential_resolver = credential_resolver
        self._policy = policy or AttachmentPolicy()
        self._timeout = float(download_timeout_seconds)
        self._mutex = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closed = False
        self._root_fd = _open_secure_root(self._root)
        root_stat = os.fstat(self._root_fd)
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self._state = AttachmentStagingState()
        try:
            self._rebuild_unlocked()
        except BaseException:
            os.close(self._root_fd)
            self._closed = True
            raise

    @property
    def state(self) -> AttachmentStagingState:
        return self._state

    def __repr__(self) -> str:
        return f"AttachmentStagingService(root={self._root!s}, closed={self._closed})"

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            os.close(self._root_fd)

    def stage(self, request: AuthorizedAttachmentStagingRequest) -> AttachmentStagingRecord:
        """Download only after receiving an explicitly authorized request."""

        if not isinstance(request, AuthorizedAttachmentStagingRequest):
            raise TypeError("request must be AuthorizedAttachmentStagingRequest")
        self._policy.validate(request.descriptors)
        with self._mutex:
            self._ensure_open()
            self._sync_unlocked()
            if request.acceptance_id in self._state.by_acceptance_id:
                raise AttachmentStateError("attachment staging already exists")
            storage_names = tuple(self._allocate_storage_name() for _ in request.descriptors)
            if len(set(storage_names)) != len(storage_names):
                raise AttachmentStorageError("random storage names collided")
            base = (
                tenant_storage_component(request.tenant_key),
                request.agent_id,
                request.envelope_id,
            )
            relative_paths = tuple("/".join((*base, f"att_{name}.bin")) for name in storage_names)
            temporary_paths = tuple("/".join((*base, f".att_{name}.tmp")) for name in storage_names)
            staging_id = "stg_" + uuid.uuid4().hex
            aggregate_id = "astg_" + hashlib.sha256(staging_id.encode()).hexdigest()
            started = JournalEvent(
                event_type="employee.ingress.attachment_staging_started",
                aggregate_id=aggregate_id,
                payload={
                    "staging_id": staging_id,
                    "acceptance_id": request.acceptance_id,
                    "envelope_id": request.envelope_id,
                    "tenant_key": request.tenant_key,
                    "agent_id": request.agent_id,
                    "app_id": request.app_id,
                    "relative_paths": list(relative_paths),
                    "temporary_paths": list(temporary_paths),
                    "content_hashes": [item.declared_sha256 for item in request.descriptors],
                },
            )
            self._commit_unlocked(aggregate_id, started)
            record = self._state.by_staging_id[staging_id]
            try:
                parent_device, parent_inode = self._prepare_parent_unlocked(record)
                parent_bound = JournalEvent(
                    event_type="employee.ingress.attachment_staging_parent_bound",
                    aggregate_id=aggregate_id,
                    payload={
                        "staging_id": staging_id,
                        "parent_device": parent_device,
                        "parent_inode": parent_inode,
                    },
                )
                self._commit_unlocked(aggregate_id, parent_bound)
                record = self._state.by_staging_id[staging_id]
                downloader = self._open_employee_downloader(request)
                for index, descriptor in enumerate(request.descriptors):
                    downloaded = self._download_with_deadline(downloader, descriptor)
                    self._validate_download(descriptor, downloaded)
                    self._prepare_and_publish_leaf_unlocked(
                        record=record,
                        index=index,
                        content=downloaded.content,
                    )
                    record = self._state.by_staging_id[staging_id]
                self._verify_staged_record_unlocked(record)
                completed = JournalEvent(
                    event_type="employee.ingress.attachment_staging_completed",
                    aggregate_id=aggregate_id,
                    payload={"staging_id": staging_id},
                )
                self._commit_unlocked(aggregate_id, completed)
                result = self._state.by_staging_id[staging_id]
                self.trusted_paths(staging_id)
                return result
            except Exception as exc:
                normalized = self._normalize_failure(exc)
                self._fail_and_cleanup_unlocked(staging_id, normalized)
                raise normalized from None

    def trusted_paths(self, staging_id: str) -> tuple[Path, ...]:
        """Return only verified Gateway-produced paths from completed staging."""

        with self._mutex:
            self._ensure_open()
            self._sync_unlocked()
            record = self._state.by_staging_id.get(staging_id)
            if record is None:
                raise KeyError(staging_id)
            if record.status != "completed" or record.cleanup_state != "none":
                return ()
            paths: list[Path] = []
            for index, (relative, expected_hash) in enumerate(
                zip(record.relative_paths, record.content_hashes)
            ):
                content = self._read_trusted_unlocked(relative, record, index)
                if hashlib.sha256(content).hexdigest() != expected_hash:
                    raise AttachmentStorageError("trusted attachment hash is invalid")
                paths.append(self._root / relative)
            return tuple(paths)

    def completed_for_acceptance(
        self,
        acceptance_id: str,
    ) -> AttachmentStagingRecord | None:
        """Return one verified reusable stage for idempotent Router recovery."""

        with self._mutex:
            self._ensure_open()
            self._sync_unlocked()
            staging_id = self._state.by_acceptance_id.get(acceptance_id)
            if staging_id is None:
                return None
            record = self._state.by_staging_id[staging_id]
            if record.cleanup_state == "completed":
                return None
            if record.status != "completed" or record.cleanup_state != "none":
                raise AttachmentStateError("attachment staging is not reusable")
            self.trusted_paths(staging_id)
            return record

    def _verify_staged_record_unlocked(self, record: AttachmentStagingRecord) -> None:
        if record.status != "started" or record.cleanup_state != "none":
            raise AttachmentStateError("attachment staging is not live")
        for index, (relative, expected_hash) in enumerate(
            zip(record.relative_paths, record.content_hashes)
        ):
            content = self._read_trusted_unlocked(relative, record, index)
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise AttachmentStorageError("trusted attachment hash is invalid")

    def cleanup(self, staging_id: str) -> None:
        """Durably begin deletion before removing any owned server path."""

        with self._mutex:
            self._ensure_open()
            self._sync_unlocked()
            self._cleanup_unlocked(staging_id)

    def recover(self) -> int:
        """Converge interrupted staging and cleanup after process restart."""

        with self._mutex:
            self._ensure_open()
            self._sync_unlocked()
            recovered = 0
            for staging_id in tuple(sorted(self._state.by_staging_id)):
                record = self._state.by_staging_id[staging_id]
                if record.cleanup_state == "completed":
                    continue
                needs_recovery = (
                    record.status == "started"
                    or record.cleanup_state == "started"
                    or (record.status == "failed" and record.cleanup_state == "none")
                )
                if not needs_recovery:
                    continue
                if record.status == "started":
                    failed = JournalEvent(
                        event_type="employee.ingress.attachment_staging_failed",
                        aggregate_id=record.aggregate_id,
                        payload={"staging_id": staging_id, "reason": "restart_recovery"},
                    )
                    self._commit_unlocked(record.aggregate_id, failed)
                self._cleanup_unlocked(staging_id)
                recovered += 1
            return recovered

    def _allocate_storage_name(self) -> str:
        try:
            value = secrets.token_hex(16)
        except Exception:
            raise AttachmentStorageError("random storage allocation failed") from None
        if not isinstance(value, str) or _STORAGE_NAME.fullmatch(value) is None:
            raise AttachmentStorageError("random storage name is invalid")
        return value

    def _open_employee_downloader(
        self,
        request: AuthorizedAttachmentStagingRequest,
    ) -> AttachmentDownloader:
        try:
            secret = self._credential_resolver.resolve(
                request.credential_ref,
                request.agent_id,
                request.app_id,
            )
            if not isinstance(secret, str) or not secret:
                raise ValueError
            return _default_downloader_builder(
                app_id=request.app_id,
                app_secret=secret,
                timeout=self._timeout,
            )
        except Exception:
            raise AttachmentCredentialError("employee attachment credential unavailable") from None

    def _download_with_deadline(
        self,
        downloader: AttachmentDownloader,
        descriptor: EmployeeAttachmentDescriptor,
    ) -> DownloadedAttachment:
        result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                result_queue.put_nowait((True, downloader.download(descriptor)))
            except BaseException as exc:
                try:
                    result_queue.put_nowait((False, exc))
                except queue.Full:
                    pass

        thread = threading.Thread(target=run, name="employee-attachment-download", daemon=True)
        thread.start()
        try:
            succeeded, value = result_queue.get(timeout=self._timeout)
        except queue.Empty:
            raise AttachmentTimeoutError("employee attachment download timed out") from None
        if not succeeded:
            raise AttachmentDownloadError("employee attachment download failed") from None
        if not isinstance(value, DownloadedAttachment):
            raise AttachmentDownloadError("employee attachment download returned invalid data")
        return value

    @staticmethod
    def _validate_download(
        descriptor: EmployeeAttachmentDescriptor,
        downloaded: DownloadedAttachment,
    ) -> None:
        content = downloaded.content
        if len(content) != descriptor.declared_size_bytes:
            raise AttachmentValidationError("attachment size does not match descriptor")
        if hashlib.sha256(content).hexdigest() != descriptor.declared_sha256:
            raise AttachmentValidationError("attachment hash does not match descriptor")
        if _is_executable(content, descriptor.user_filename) or _is_executable(content, downloaded.file_name):
            raise AttachmentValidationError("attachment executable content is forbidden")
        detected = _detect_mime(content)
        declared = descriptor.declared_mime_type
        if not _mime_matches(declared, detected):
            raise AttachmentValidationError("attachment MIME does not match magic")

    def _prepare_parent_unlocked(
        self,
        record: AttachmentStagingRecord,
    ) -> tuple[int, int]:
        parts = _relative_parts(record.relative_paths[0])[:-1]
        fd = self._open_directory_unlocked(parts, create=True)
        try:
            parent_stat = os.fstat(fd)
            return parent_stat.st_dev, parent_stat.st_ino
        finally:
            os.close(fd)

    def _prepare_and_publish_leaf_unlocked(
        self,
        *,
        record: AttachmentStagingRecord,
        index: int,
        content: bytes,
    ) -> None:
        temporary = record.temporary_paths[index]
        final = record.relative_paths[index]
        temp_parts = _relative_parts(temporary)
        final_parts = _relative_parts(final)
        if temp_parts[:-1] != final_parts[:-1]:
            raise AttachmentStorageError("attachment path ownership mismatch")
        parent_fd = self._open_directory_unlocked(temp_parts[:-1], create=False)
        temp_name = temp_parts[-1]
        final_name = final_parts[-1]
        descriptor: int | None = None
        identity_anchored = False
        remove_unbound_temp = True
        try:
            self._verify_parent_identity(parent_fd, record)
            _require_absent_leaf(parent_fd, final_name)
            _require_absent_leaf(parent_fd, temp_name)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            descriptor = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            os.fsync(parent_fd)
            leaf_stat = os.fstat(descriptor)
            if not stat.S_ISREG(leaf_stat.st_mode) or leaf_stat.st_uid != os.getuid():
                raise AttachmentStorageError("attachment leaf identity is invalid")
            prepared = JournalEvent(
                event_type="employee.ingress.attachment_staging_leaf_prepared",
                aggregate_id=record.aggregate_id,
                payload={
                    "staging_id": record.staging_id,
                    "index": index,
                    "leaf_device": leaf_stat.st_dev,
                    "leaf_inode": leaf_stat.st_ino,
                },
            )
            self._commit_unlocked(record.aggregate_id, prepared)
            identity_anchored = True
            current = os.fstat(descriptor)
            identity = (leaf_stat.st_dev, leaf_stat.st_ino)
            if (
                (current.st_dev, current.st_ino) != identity
                or current.st_nlink != 1
            ):
                raise AttachmentStorageError("attachment leaf identity changed")
            _require_unique_exact_leaf(parent_fd, identity, temp_name)
            _write_all(descriptor, content)
            os.fsync(descriptor)
            _require_absent_leaf(parent_fd, final_name)
            os.replace(temp_name, final_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
            current = os.fstat(descriptor)
            if (
                (current.st_dev, current.st_ino) != identity
                or current.st_nlink != 1
            ):
                raise AttachmentStorageError("attachment leaf identity changed")
            _require_unique_exact_leaf(parent_fd, identity, final_name)
            os.close(descriptor)
            descriptor = None
        except AttachmentStorageError:
            raise
        except OSError:
            raise AttachmentStorageError("attachment leaf storage failed") from None
        except Exception:
            raise
        except BaseException:
            remove_unbound_temp = False
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if not identity_anchored and remove_unbound_temp:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            os.close(parent_fd)

    def _read_trusted_unlocked(
        self,
        relative: str,
        record: AttachmentStagingRecord,
        index: int,
    ) -> bytes:
        parts = _relative_parts(relative)
        parent_fd = self._open_directory_unlocked(parts[:-1], create=False)
        fd: int | None = None
        try:
            self._verify_parent_identity(parent_fd, record)
            fd = os.open(parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
            file_stat = os.fstat(fd)
            expected_identity = record.leaf_identities[index]
            if expected_identity is None or not _trusted_leaf_stat_matches(
                file_stat,
                expected_identity,
            ):
                raise AttachmentStorageError("trusted attachment leaf is invalid")
            pre_read_generation = _trusted_leaf_generation(file_stat)
            _require_unique_exact_leaf(parent_fd, expected_identity, parts[-1])
            content = _read_at_most(fd, self._policy.max_file_bytes + 1)
            file_stat = os.fstat(fd)
            post_read_generation = _trusted_leaf_generation(file_stat)
            if (
                not _trusted_leaf_stat_matches(file_stat, expected_identity)
                or post_read_generation != pre_read_generation
            ):
                raise AttachmentStorageError("trusted attachment leaf is invalid")
            _require_unique_exact_leaf(parent_fd, expected_identity, parts[-1])
            self._verify_current_trusted_leaf_unlocked(
                parts,
                record,
                index,
                post_read_generation,
            )
            return content
        except AttachmentStorageError:
            raise
        except OSError:
            raise AttachmentStorageError("trusted attachment cannot be opened") from None
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    def _verify_current_trusted_leaf_unlocked(
        self,
        parts: tuple[str, ...],
        record: AttachmentStagingRecord,
        index: int,
        expected_generation: tuple[int, ...],
    ) -> None:
        expected_identity = record.leaf_identities[index]
        if expected_identity is None:
            raise AttachmentStorageError("trusted attachment leaf is invalid")
        parent_fd = self._open_directory_unlocked(parts[:-1], create=False)
        fd: int | None = None
        try:
            self._verify_parent_identity(parent_fd, record)
            fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            current_stat = os.fstat(fd)
            if (
                not _trusted_leaf_stat_matches(current_stat, expected_identity)
                or _trusted_leaf_generation(current_stat) != expected_generation
            ):
                raise AttachmentStorageError("trusted attachment leaf is invalid")
            _require_unique_exact_leaf(parent_fd, expected_identity, parts[-1])
        except AttachmentStorageError:
            raise
        except OSError:
            raise AttachmentStorageError("trusted attachment cannot be reopened") from None
        finally:
            if fd is not None:
                os.close(fd)
            os.close(parent_fd)

    def _cleanup_unlocked(self, staging_id: str) -> None:
        record = self._state.by_staging_id.get(staging_id)
        if record is None:
            raise KeyError(staging_id)
        if record.cleanup_state == "completed":
            return
        if record.cleanup_state == "none":
            event = JournalEvent(
                event_type="employee.ingress.attachment_cleanup_started",
                aggregate_id=record.aggregate_id,
                payload={"staging_id": staging_id},
            )
            self._commit_unlocked(record.aggregate_id, event)
            record = self._state.by_staging_id[staging_id]
        self._dispose_owned_paths_unlocked(record)
        completed = JournalEvent(
            event_type="employee.ingress.attachment_cleanup_completed",
            aggregate_id=record.aggregate_id,
            payload={"staging_id": staging_id},
        )
        self._commit_unlocked(record.aggregate_id, completed)

    def _dispose_owned_paths_unlocked(self, record: AttachmentStagingRecord) -> None:
        if record.parent_device is None or record.parent_inode is None:
            # No leaf can be published before the parent identity is anchored.
            return
        parent_parts = _relative_parts(record.relative_paths[0])[:-1]
        parent_fd = self._open_directory_unlocked(parent_parts, create=False)
        try:
            self._verify_parent_identity(parent_fd, record)
            for index, identity in enumerate(record.leaf_identities):
                temporary = _relative_parts(record.temporary_paths[index])[-1]
                final = _relative_parts(record.relative_paths[index])[-1]
                if identity is None:
                    self._dispose_unbound_empty_temp_unlocked(
                        parent_fd,
                        temporary=temporary,
                        final=final,
                    )
                    continue
                self._erase_bound_leaf_unlocked(
                    parent_fd,
                    identity=identity,
                    temporary=temporary,
                    final=final,
                )
            os.fsync(parent_fd)
        except AttachmentStorageError:
            raise
        except OSError:
            raise AttachmentStorageError("attachment cleanup failed") from None
        finally:
            os.close(parent_fd)

    def _dispose_unbound_empty_temp_unlocked(
        self,
        parent_fd: int,
        *,
        temporary: str,
        final: str,
    ) -> None:
        if _leaf_lstat(parent_fd, final) is not None:
            raise AttachmentStorageError("unbound attachment leaf is not trusted")
        leaf = _leaf_lstat(parent_fd, temporary)
        if leaf is None:
            return
        identity = (leaf.st_dev, leaf.st_ino)
        if not _trusted_leaf_stat_matches(leaf, identity) or leaf.st_size != 0:
            raise AttachmentStorageError("unbound attachment leaf is not trusted")
        fd: int | None = None
        try:
            fd = os.open(
                temporary,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            current = os.fstat(fd)
            if (
                not _trusted_leaf_stat_matches(current, identity)
                or current.st_size != 0
            ):
                raise AttachmentStorageError("unbound attachment leaf is not trusted")
            _require_unique_exact_leaf(parent_fd, identity, temporary)
            os.unlink(temporary, dir_fd=parent_fd)
            os.fsync(parent_fd)
            if os.fstat(fd).st_nlink != 0 or _leaf_lstat(parent_fd, temporary) is not None:
                raise AttachmentStorageError("unbound attachment leaf was not erased")
        except AttachmentStorageError:
            raise
        except OSError:
            raise AttachmentStorageError("unbound attachment cleanup failed") from None
        finally:
            if fd is not None:
                os.close(fd)

    def _erase_bound_leaf_unlocked(
        self,
        parent_fd: int,
        *,
        identity: tuple[int, int],
        temporary: str,
        final: str,
    ) -> None:
        name = _locate_exact_leaf(parent_fd, identity, temporary, final)
        if name is None:
            return
        fd: int | None = None
        try:
            fd = os.open(name, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd)
            leaf_stat = os.fstat(fd)
            if not _trusted_leaf_stat_matches(leaf_stat, identity):
                raise AttachmentStorageError("attachment cleanup leaf identity changed")
            _require_unique_exact_leaf(parent_fd, identity, name)
            os.ftruncate(fd, 0)
            os.fsync(fd)
            current = os.fstat(fd)
            if not _trusted_leaf_stat_matches(current, identity) or current.st_size != 0:
                raise AttachmentStorageError("attachment cleanup leaf identity changed")
            _require_unique_exact_leaf(parent_fd, identity, name)
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            if os.fstat(fd).st_nlink != 0:
                raise AttachmentStorageError("attachment leaf was not erased")
        except AttachmentStorageError:
            raise
        except OSError:
            raise AttachmentStorageError("attachment cleanup failed") from None
        finally:
            if fd is not None:
                os.close(fd)

    @staticmethod
    def _verify_parent_identity(
        parent_fd: int,
        record: AttachmentStagingRecord,
    ) -> None:
        if record.parent_device is None or record.parent_inode is None:
            raise AttachmentStorageError("attachment parent identity is not bound")
        parent_stat = os.fstat(parent_fd)
        if (parent_stat.st_dev, parent_stat.st_ino) != (
            record.parent_device,
            record.parent_inode,
        ):
            raise AttachmentStorageError("attachment parent identity changed")

    def _fail_and_cleanup_unlocked(self, staging_id: str, error: AttachmentError) -> None:
        record = self._state.by_staging_id[staging_id]
        if record.status == "started":
            failed = JournalEvent(
                event_type="employee.ingress.attachment_staging_failed",
                aggregate_id=record.aggregate_id,
                payload={"staging_id": staging_id, "reason": _failure_reason(error)},
            )
            self._commit_unlocked(record.aggregate_id, failed)
        try:
            self._cleanup_unlocked(staging_id)
        except AttachmentError:
            pass

    @staticmethod
    def _normalize_failure(error: Exception) -> AttachmentError:
        if isinstance(error, AttachmentError):
            return error
        return AttachmentDownloadError("employee attachment staging failed")

    def _ensure_open(self) -> None:
        if self._closed:
            raise AttachmentStateError("attachment staging service is closed")
        try:
            current = os.stat(self._root, follow_symlinks=False)
        except OSError:
            raise AttachmentStorageError("attachment root identity changed") from None
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != self._root_identity:
            raise AttachmentStorageError("attachment root identity changed")

    def _open_directory_unlocked(self, parts: tuple[str, ...], *, create: bool) -> int:
        self._ensure_open()
        descriptor = os.dup(self._root_fd)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            for part in parts:
                if not _CANONICAL_ID.fullmatch(part):
                    raise AttachmentStorageError("attachment parent path is invalid")
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise AttachmentStorageError("attachment parent path is missing") from None
                    try:
                        os.mkdir(part, 0o700, dir_fd=descriptor)
                        os.fsync(descriptor)
                    except OSError:
                        raise AttachmentStorageError("attachment parent cannot be created") from None
                    try:
                        child = os.open(part, flags, dir_fd=descriptor)
                    except OSError:
                        raise AttachmentStorageError("attachment parent is not trusted") from None
                except OSError:
                    raise AttachmentStorageError("attachment parent is not trusted") from None
                child_stat = os.fstat(child)
                if not stat.S_ISDIR(child_stat.st_mode) or child_stat.st_uid != os.getuid():
                    os.close(child)
                    raise AttachmentStorageError("attachment parent identity is invalid")
                os.fchmod(child, 0o700)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _sync_unlocked(self) -> None:
        anchor = self._writer.anchor.read()
        expected_hash = "" if anchor.sequence == 0 else anchor.frame_hash
        if (self._state.cursor_sequence, self._state.cursor_hash) != (anchor.sequence, expected_hash):
            self._rebuild_unlocked()

    def _rebuild_unlocked(self) -> None:
        fresh = AttachmentStagingState()
        anchor = self._writer.anchor.read()
        anchored_hash = GENESIS_HASH
        for frame in self._writer.replay():
            if frame.sequence > anchor.sequence:
                break
            for event in frame.events:
                if event.event_type in _EVENT_TYPES:
                    _apply_event(fresh, event)
            fresh.cursor_sequence = frame.sequence
            fresh.cursor_hash = frame.frame_hash
            anchored_hash = frame.frame_hash
        if anchored_hash != anchor.frame_hash:
            raise AttachmentStateError("attachment projection cannot verify anchor")
        self._state = fresh

    def _commit_unlocked(self, aggregate_id: str, event: JournalEvent) -> None:
        with self._writer.transaction_guard():
            self._sync_unlocked()
            candidate = AttachmentStagingState(
                by_staging_id=dict(self._state.by_staging_id),
                by_acceptance_id=dict(self._state.by_acceptance_id),
                cursor_sequence=self._state.cursor_sequence,
                cursor_hash=self._state.cursor_hash,
            )
            if event.event_type in _EVENT_TYPES:
                _apply_event(candidate, event)
            result = self._writer.commit(
                [event],
                self._writer.get_aggregate_versions([aggregate_id]),
                expected_head_sequence=self._state.cursor_sequence,
                expected_head_hash=self._state.cursor_hash or None,
            )
            if result.state is not CommitState.ANCHORED:
                raise AttachmentStateError("attachment lifecycle was not anchored")
            candidate.cursor_sequence = result.frame.sequence
            candidate.cursor_hash = result.frame.frame_hash
            self._state = candidate


def _failure_reason(error: AttachmentError) -> str:
    if isinstance(error, AttachmentTimeoutError):
        return "timeout"
    if isinstance(error, AttachmentValidationError):
        return "validation"
    if isinstance(error, AttachmentStorageError):
        return "storage"
    if isinstance(error, AttachmentCredentialError):
        return "credential"
    return "download"


def _apply_event(state: AttachmentStagingState, event: JournalEvent) -> None:
    payload = event.payload
    if event.event_type == "employee.ingress.attachment_staging_started":
        fields = {
            "staging_id",
            "acceptance_id",
            "envelope_id",
            "tenant_key",
            "agent_id",
            "app_id",
            "relative_paths",
            "temporary_paths",
            "content_hashes",
        }
        if set(payload) != fields:
            raise AttachmentStateError("invalid attachment staging start")
        staging_id = _canonical_id(payload["staging_id"], "staging_id", "stg_")
        acceptance_id = _canonical_id(payload["acceptance_id"], "acceptance_id", "acc_")
        expected_aggregate = "astg_" + hashlib.sha256(staging_id.encode()).hexdigest()
        if event.aggregate_id != expected_aggregate:
            raise AttachmentStateError("attachment staging aggregate mismatch")
        if staging_id in state.by_staging_id or acceptance_id in state.by_acceptance_id:
            raise AttachmentStateError("duplicate attachment staging identity")
        relative_paths = _path_collection(payload["relative_paths"])
        temporary_paths = _path_collection(payload["temporary_paths"])
        hashes = _hash_collection(payload["content_hashes"])
        if not relative_paths or len({len(relative_paths), len(temporary_paths), len(hashes)}) != 1:
            raise AttachmentStateError("attachment staging path cardinality mismatch")
        if len(set((*relative_paths, *temporary_paths))) != len(relative_paths) * 2:
            raise AttachmentStateError("attachment staging paths must be unique")
        record = AttachmentStagingRecord(
            staging_id=staging_id,
            aggregate_id=event.aggregate_id,
            acceptance_id=acceptance_id,
            envelope_id=_canonical_id(payload["envelope_id"], "envelope_id", "ing_"),
            tenant_key=_strict_string(payload["tenant_key"], "tenant_key", maximum=512),
            agent_id=_canonical_id(payload["agent_id"], "agent_id", "agt_"),
            app_id=_canonical_id(payload["app_id"], "app_id", "cli_"),
            relative_paths=relative_paths,
            temporary_paths=temporary_paths,
            content_hashes=hashes,
            leaf_identities=tuple(None for _ in relative_paths),
        )
        expected_base = (
            tenant_storage_component(record.tenant_key),
            record.agent_id,
            record.envelope_id,
        )
        if any(_relative_parts(value)[:-1] != expected_base for value in (*relative_paths, *temporary_paths)):
            raise AttachmentStateError("attachment staging path owner mismatch")
        state.by_staging_id[staging_id] = record
        state.by_acceptance_id[acceptance_id] = staging_id
        return
    if event.event_type == "employee.ingress.attachment_staging_parent_bound":
        if set(payload) != {"staging_id", "parent_device", "parent_inode"}:
            raise AttachmentStateError("invalid attachment parent binding")
        staging_id = payload.get("staging_id")
        if not isinstance(staging_id, str) or staging_id not in state.by_staging_id:
            raise AttachmentStateError(
                "attachment parent references unknown staging"
            )
        record = state.by_staging_id[staging_id]
        if event.aggregate_id != record.aggregate_id:
            raise AttachmentStateError("attachment parent aggregate mismatch")
        parent_device = payload.get("parent_device")
        parent_inode = payload.get("parent_inode")
        if (
            isinstance(parent_device, bool)
            or not isinstance(parent_device, int)
            or parent_device < 0
            or isinstance(parent_inode, bool)
            or not isinstance(parent_inode, int)
            or parent_inode <= 0
        ):
            raise AttachmentStateError("attachment parent identity is invalid")
        if (
            record.status != "started"
            or record.cleanup_state != "none"
            or record.parent_device is not None
            or record.parent_inode is not None
        ):
            raise AttachmentStateError("attachment parent is already bound")
        state.by_staging_id[staging_id] = replace(
            record,
            parent_device=parent_device,
            parent_inode=parent_inode,
        )
        return
    if event.event_type == "employee.ingress.attachment_staging_leaf_prepared":
        if set(payload) != {
            "staging_id",
            "index",
            "leaf_device",
            "leaf_inode",
        }:
            raise AttachmentStateError("invalid attachment leaf binding")
        staging_id = payload.get("staging_id")
        if not isinstance(staging_id, str) or staging_id not in state.by_staging_id:
            raise AttachmentStateError("attachment leaf references unknown staging")
        record = state.by_staging_id[staging_id]
        if event.aggregate_id != record.aggregate_id:
            raise AttachmentStateError("attachment leaf aggregate mismatch")
        index = payload.get("index")
        leaf_device = payload.get("leaf_device")
        leaf_inode = payload.get("leaf_inode")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(record.leaf_identities)
            or isinstance(leaf_device, bool)
            or not isinstance(leaf_device, int)
            or leaf_device < 0
            or isinstance(leaf_inode, bool)
            or not isinstance(leaf_inode, int)
            or leaf_inode <= 0
        ):
            raise AttachmentStateError("attachment leaf identity is invalid")
        if (
            record.status != "started"
            or record.cleanup_state != "none"
            or record.parent_device is None
            or record.parent_inode is None
            or record.leaf_identities[index] is not None
        ):
            raise AttachmentStateError("attachment leaf cannot be prepared")
        identities = list(record.leaf_identities)
        identity = (leaf_device, leaf_inode)
        if identity in identities:
            raise AttachmentStateError("attachment leaf identity is duplicated")
        identities[index] = identity
        state.by_staging_id[staging_id] = replace(
            record,
            leaf_identities=tuple(identities),
        )
        return
    if set(payload) not in ({"staging_id"}, {"staging_id", "reason"}):
        raise AttachmentStateError("invalid attachment lifecycle event")
    staging_id = payload.get("staging_id")
    if not isinstance(staging_id, str) or staging_id not in state.by_staging_id:
        raise AttachmentStateError("attachment lifecycle references unknown staging")
    record = state.by_staging_id[staging_id]
    if event.aggregate_id != record.aggregate_id:
        raise AttachmentStateError("attachment lifecycle aggregate mismatch")
    if event.event_type == "employee.ingress.attachment_staging_completed":
        if (
            record.status != "started"
            or record.cleanup_state != "none"
            or record.parent_device is None
            or record.parent_inode is None
            or any(identity is None for identity in record.leaf_identities)
        ):
            raise AttachmentStateError("attachment staging cannot complete")
        record = replace(record, status="completed")
    elif event.event_type == "employee.ingress.attachment_staging_failed":
        if (
            record.status != "started"
            or record.cleanup_state != "none"
            or set(payload) != {"staging_id", "reason"}
        ):
            raise AttachmentStateError("attachment staging cannot fail")
        reason = _strict_string(payload["reason"], "failure reason", maximum=64)
        record = replace(record, status="failed", failure_reason=reason)
    elif event.event_type == "employee.ingress.attachment_cleanup_started":
        if record.status not in {"completed", "failed"} or record.cleanup_state != "none":
            raise AttachmentStateError("attachment cleanup cannot start")
        record = replace(record, cleanup_state="started")
    elif event.event_type == "employee.ingress.attachment_cleanup_completed":
        if (
            record.status not in {"completed", "failed"}
            or record.cleanup_state != "started"
        ):
            raise AttachmentStateError("attachment cleanup was not started")
        record = replace(record, cleanup_state="completed")
    else:
        raise AttachmentStateError("unknown attachment lifecycle event")
    state.by_staging_id[staging_id] = record


def _path_collection(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AttachmentStateError("attachment paths must be a list")
    result = tuple(value)
    for item in result:
        _relative_parts(item)
    return result


def _hash_collection(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and _SHA256.fullmatch(item) for item in value):
        raise AttachmentStateError("attachment hashes must be lowercase sha256")
    return tuple(value)


def _relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AttachmentStorageError("attachment relative path is invalid")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} or "/" in part or "\\" in part for part in parts):
        raise AttachmentStorageError("attachment relative path is invalid")
    return parts


def _open_secure_root(root: Path) -> int:
    parts = root.parts[1:] if root.is_absolute() else root.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise AttachmentStorageError("attachment root is invalid")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open("/" if root.is_absolute() else ".", flags)
        for index, part in enumerate(parts):
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except OSError:
                    raise AttachmentStorageError("attachment root cannot be created") from None
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except OSError:
                    raise AttachmentStorageError("attachment root is not trusted") from None
            except OSError:
                raise AttachmentStorageError("attachment root is not trusted") from None
            os.close(descriptor)
            descriptor = child
            if index == len(parts) - 1:
                os.fchmod(descriptor, 0o700)
        root_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid():
            raise AttachmentStorageError("attachment root identity is invalid")
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _require_absent_leaf(parent_fd: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise AttachmentStorageError("attachment leaf cannot be inspected") from None
    raise AttachmentStorageError("attachment leaf already exists")


def _leaf_lstat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise AttachmentStorageError("attachment leaf cannot be inspected") from None


def _trusted_leaf_stat_matches(
    file_stat: os.stat_result,
    identity: tuple[int, int],
) -> bool:
    return (
        (file_stat.st_dev, file_stat.st_ino) == identity
        and stat.S_ISREG(file_stat.st_mode)
        and stat.S_IMODE(file_stat.st_mode) == 0o600
        and file_stat.st_uid == os.getuid()
        and file_stat.st_nlink == 1
    )


def _trusted_leaf_generation(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_uid,
        file_stat.st_nlink,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _require_unique_exact_leaf(
    parent_fd: int,
    identity: tuple[int, int],
    expected_name: str,
) -> None:
    expected = _leaf_lstat(parent_fd, expected_name)
    if (
        expected is None
        or (expected.st_dev, expected.st_ino) != identity
        or expected.st_nlink != 1
    ):
        raise AttachmentStorageError("attachment leaf identity changed")
    exact_names = []
    for name in os.listdir(parent_fd):
        leaf_stat = _leaf_lstat(parent_fd, name)
        if leaf_stat is not None and (leaf_stat.st_dev, leaf_stat.st_ino) == identity:
            exact_names.append(name)
    if exact_names != [expected_name]:
        raise AttachmentStorageError("attachment leaf identity has multiple names")


def _locate_exact_leaf(
    parent_fd: int,
    identity: tuple[int, int],
    temporary: str,
    final: str,
) -> str | None:
    expected_names: list[str] = []
    for name in (temporary, final):
        leaf_stat = _leaf_lstat(parent_fd, name)
        if leaf_stat is None:
            continue
        if (leaf_stat.st_dev, leaf_stat.st_ino) != identity:
            raise AttachmentStorageError("attachment cleanup leaf identity changed")
        expected_names.append(name)
    if len(expected_names) > 1:
        raise AttachmentStorageError("attachment leaf identity has multiple names")
    for name in os.listdir(parent_fd):
        if name in {temporary, final}:
            continue
        leaf_stat = _leaf_lstat(parent_fd, name)
        if leaf_stat is not None and (leaf_stat.st_dev, leaf_stat.st_ino) == identity:
            raise AttachmentStorageError("attachment leaf identity was displaced")
    return expected_names[0] if expected_names else None


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise AttachmentStorageError("attachment write failed")
        view = view[written:]


def _read_at_most(fd: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _is_executable(content: bytes, filename: str) -> bool:
    lower_name = filename.casefold()
    if any(lower_name.endswith(suffix) for suffix in _DANGEROUS_SUFFIXES):
        return True
    executable_magic = (
        b"\x7fELF",
        b"MZ",
        b"#!",
        b"\xfe\xed\xfa\xce",
        b"\xfe\xed\xfa\xcf",
        b"\xce\xfa\xed\xfe",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
    )
    return content.startswith(executable_magic)


def _detect_mime(content: bytes) -> str:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"PK\x03\x04"):
        return "application/zip"
    if content.startswith(b"ID3") or (len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0):
        return "audio/mpeg"
    if content.startswith(b"RIFF") and content[8:12] == b"WAVE":
        return "audio/wav"
    if len(content) >= 12 and content[4:8] == b"ftyp":
        return "video/mp4"
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    if "\x00" in text:
        return "application/octet-stream"
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return "text/plain"
    return "application/json"


def _mime_matches(declared: str, detected: str) -> bool:
    if declared == "application/octet-stream":
        return True
    if detected == "application/zip" and declared in _ZIP_MIMES:
        return True
    if detected == "text/plain" and declared.startswith("text/"):
        return True
    return declared == detected


__all__ = [
    "AttachmentCredentialError",
    "AttachmentDownloadError",
    "AttachmentError",
    "AttachmentPolicy",
    "AttachmentPolicyError",
    "AttachmentStagingRecord",
    "AttachmentStagingService",
    "AttachmentStateError",
    "AttachmentStorageError",
    "AttachmentTimeoutError",
    "AttachmentValidationError",
    "AuthorizedAttachmentStagingRequest",
    "DownloadedAttachment",
    "EmployeeAttachmentDescriptor",
    "LarkEmployeeAttachmentDownloader",
    "tenant_storage_component",
]
