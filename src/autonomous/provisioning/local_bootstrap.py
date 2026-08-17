"""Private local key bootstrap for the built-in employee runtime."""

from __future__ import annotations

import base64
import binascii
import errno
import fcntl
import json
import os
import secrets
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.keyring import EmployeeDataKeyring
from ..workforce.credential_vault import CredentialKeyring

_SECRET_FILENAME = "employee-runtime-secrets.json"
_LOCK_FILENAME = ".employee-runtime-secrets.lock"
_LOCK_OPEN_ATTEMPTS = 8
_MAX_ENVELOPE_BYTES = 16 * 1024
_ENVELOPE_FIELDS = frozenset(
    {
        "version",
        "journal_hmac_key",
        "credential_key_id",
        "credential_key",
        "data_key_id",
        "data_key",
    }
)


class LocalEmployeeBootstrapError(RuntimeError):
    """Local employee security material is absent or unsafe."""

    def __init__(self) -> None:
        super().__init__(type(self).__name__)


@dataclass(frozen=True)
class LocalEmployeeRuntimeMaterial:
    """Decoded employee keys kept only in process memory."""

    journal_hmac_key: bytes = field(repr=False)
    credential_keyring: CredentialKeyring = field(repr=False)
    data_keyring: EmployeeDataKeyring = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.journal_hmac_key, bytes) or len(self.journal_hmac_key) < 32:
            raise LocalEmployeeBootstrapError()


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LocalEmployeeBootstrapError()
        result[key] = value
    return result


def _decode_key(value: object) -> bytes:
    if not isinstance(value, str):
        raise LocalEmployeeBootstrapError()
    try:
        decoded = base64.b64decode(value, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LocalEmployeeBootstrapError() from exc
    if len(decoded) != 32:
        raise LocalEmployeeBootstrapError()
    return decoded


def _encode_key(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode_envelope(raw: bytes) -> LocalEmployeeRuntimeMaterial:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_object)
    except LocalEmployeeBootstrapError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LocalEmployeeBootstrapError() from exc
    if not isinstance(value, dict) or set(value) != _ENVELOPE_FIELDS:
        raise LocalEmployeeBootstrapError()
    if type(value["version"]) is not int or value["version"] != 1:
        raise LocalEmployeeBootstrapError()
    credential_key_id = value["credential_key_id"]
    data_key_id = value["data_key_id"]
    if (
        not isinstance(credential_key_id, str)
        or credential_key_id != "local-credential-v1"
        or not isinstance(data_key_id, str)
        or data_key_id != "local-data-v1"
    ):
        raise LocalEmployeeBootstrapError()
    try:
        return LocalEmployeeRuntimeMaterial(
            journal_hmac_key=_decode_key(value["journal_hmac_key"]),
            credential_keyring=CredentialKeyring(
                keys={credential_key_id: _decode_key(value["credential_key"])},
                active_key_id=credential_key_id,
            ),
            data_keyring=EmployeeDataKeyring(
                keys={data_key_id: _decode_key(value["data_key"])},
                active_key_id=data_key_id,
            ),
        )
    except LocalEmployeeBootstrapError:
        raise
    except Exception as exc:
        raise LocalEmployeeBootstrapError() from exc


def _new_envelope() -> bytes:
    value = {
        "version": 1,
        "journal_hmac_key": _encode_key(secrets.token_bytes(32)),
        "credential_key_id": "local-credential-v1",
        "credential_key": _encode_key(secrets.token_bytes(32)),
        "data_key_id": "local-data-v1",
        "data_key": _encode_key(secrets.token_bytes(32)),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _open_state_directory(state_dir: str | Path) -> int:
    path = Path(state_dir).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise LocalEmployeeBootstrapError() from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise LocalEmployeeBootstrapError()
        os.fchmod(descriptor, 0o700)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_existing(directory_fd: int) -> bytes | None:
    try:
        descriptor = os.open(
            _SECRET_FILENAME,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        raise LocalEmployeeBootstrapError() from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size <= 0
            or metadata.st_size > _MAX_ENVELOPE_BYTES
        ):
            raise LocalEmployeeBootstrapError()
        chunks: list[bytes] = []
        remaining = _MAX_ENVELOPE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw or len(raw) > _MAX_ENVELOPE_BYTES:
            raise LocalEmployeeBootstrapError()
        return raw
    finally:
        os.close(descriptor)


def _open_lock_file(directory_fd: int) -> int:
    """Open the bootstrap lock despite Darwin's concurrent openat create race."""

    for attempt in range(_LOCK_OPEN_ATTEMPTS):
        try:
            return os.open(
                _LOCK_FILENAME,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            if exc.errno != errno.ENOENT or attempt + 1 == _LOCK_OPEN_ATTEMPTS:
                raise
    raise AssertionError("unreachable")


def _write_new(directory_fd: int, raw: bytes) -> None:
    temporary = f".{_SECRET_FILENAME}.{uuid.uuid4().hex}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise LocalEmployeeBootstrapError()
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(
            temporary,
            _SECRET_FILENAME,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except LocalEmployeeBootstrapError:
        raise
    except OSError as exc:
        raise LocalEmployeeBootstrapError() from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def load_or_create_local_employee_material(
    state_dir: str | Path,
) -> LocalEmployeeRuntimeMaterial:
    """Load or atomically create mode-restricted local employee keys."""

    directory_fd = _open_state_directory(state_dir)
    lock_fd = -1
    try:
        try:
            lock_fd = _open_lock_file(directory_fd)
            lock_metadata = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_uid != os.geteuid()
            ):
                raise LocalEmployeeBootstrapError()
            os.fchmod(lock_fd, 0o600)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            raw = _read_existing(directory_fd)
            if raw is None:
                raw = _new_envelope()
                _write_new(directory_fd, raw)
            return _decode_envelope(raw)
        except LocalEmployeeBootstrapError:
            raise
        except OSError as exc:
            raise LocalEmployeeBootstrapError() from exc
    finally:
        if lock_fd >= 0:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


def _secret_text(value: object) -> str:
    getter = getattr(value, "get_secret_value", None)
    if not callable(getter):
        return ""
    raw = getter()
    return raw if isinstance(raw, str) else ""


def resolve_employee_runtime_material(settings: Any) -> LocalEmployeeRuntimeMaterial:
    """Use one complete explicit key set or the private local bootstrap."""

    journal_text = _secret_text(getattr(settings, "autonomous_journal_hmac_key", None))
    credential_text = _secret_text(getattr(settings, "autonomous_credential_keys", None))
    credential_id = getattr(settings, "autonomous_credential_active_key_id", "")
    data_text = _secret_text(getattr(settings, "autonomous_data_keys", None))
    data_id = getattr(settings, "autonomous_data_active_key_id", "")
    configured = (
        bool(journal_text),
        bool(credential_text),
        bool(credential_id),
        bool(data_text),
        bool(data_id),
    )
    if any(configured):
        if not all(configured):
            raise LocalEmployeeBootstrapError()
        try:
            journal_key = base64.b64decode(
                journal_text,
                altchars=b"-_",
                validate=True,
            )
            return LocalEmployeeRuntimeMaterial(
                journal_hmac_key=journal_key,
                credential_keyring=CredentialKeyring.from_settings(settings),
                data_keyring=EmployeeDataKeyring.from_settings(settings),
            )
        except Exception as exc:
            raise LocalEmployeeBootstrapError() from exc
    return load_or_create_local_employee_material(settings.autonomous_state_dir)


__all__ = [
    "LocalEmployeeBootstrapError",
    "LocalEmployeeRuntimeMaterial",
    "load_or_create_local_employee_material",
    "resolve_employee_runtime_material",
]
