"""Encrypted content-addressed blob storage for sensitive runtime payloads."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Protocol

from Crypto.Cipher import AES

BLOB_MAGIC = "GHOSTAP-BLOB"
BLOB_SCHEMA_VERSION = 1
_HEX_CHARS = frozenset("0123456789abcdef")


class BlobError(RuntimeError):
    """Base class for blob-store failures."""


class BlobFormatError(BlobError):
    """The on-disk envelope is malformed or unsupported."""


class BlobIntegrityError(BlobError):
    """The content address or plaintext hash does not match."""


class BlobMissingError(BlobIntegrityError):
    """The referenced blob does not exist."""


class BlobReadError(BlobIntegrityError):
    """The referenced blob exists but could not be read."""


class BlobAuthenticationError(BlobError):
    """Authenticated decryption failed."""


class BlobPublishError(BlobError):
    """A durable publication boundary failed."""


class InvalidEncryptionKeyError(BlobError):
    """The resolved encryption key is invalid."""


class KeyResolutionError(BlobError):
    """A key reference could not be resolved."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX_CHARS for char in value)
    )


def _random_nonce() -> bytes:
    return secrets.token_bytes(12)


def _write_bytes(
    path: str | Path,
    value: bytes,
    *,
    dir_fd: int | None = None,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600, dir_fd=dir_fd)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as file:
            file.write(value)
            file.flush()
    finally:
        os.close(descriptor)


def _fsync_file(path: str | Path, *, dir_fd: int | None = None) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, dir_fd=dir_fd)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise BlobIntegrityError("blob temporary is not a regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(
    source: str | Path,
    target: str | Path,
    *,
    dir_fd: int | None = None,
) -> None:
    os.replace(source, target, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class BlobRef:
    """Immutable reference to one encrypted content-addressed blob."""

    blob_hash: str = ""
    payload_hash: str = ""
    labels_hash: str = ""
    key_ref: str = ""
    size: int = 0
    labels: Mapping[str, str] | None = None
    blob_id: str = ""
    content_hash: str = ""
    ciphertext_hash: str = ""

    def __post_init__(self) -> None:
        blob_aliases = [
            value
            for value in (self.blob_hash, self.blob_id, self.ciphertext_hash)
            if value
        ]
        payload_aliases = [
            value
            for value in (self.payload_hash, self.content_hash)
            if value
        ]
        if len(set(blob_aliases)) > 1:
            raise ValueError("conflicting blob hash aliases")
        if len(set(payload_aliases)) > 1:
            raise ValueError("conflicting payload hash aliases")
        blob_hash = self.blob_hash or self.blob_id or self.ciphertext_hash
        payload_hash = self.payload_hash or self.content_hash
        if not blob_hash:
            raise ValueError("blob hash is required")
        if not payload_hash:
            raise ValueError("payload hash is required")
        labels = dict(self.labels or {})
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in labels.items()
        ):
            raise ValueError("labels must contain only string keys and values")
        object.__setattr__(self, "blob_hash", blob_hash)
        object.__setattr__(self, "blob_id", blob_hash)
        object.__setattr__(self, "ciphertext_hash", blob_hash)
        object.__setattr__(self, "payload_hash", payload_hash)
        object.__setattr__(self, "content_hash", payload_hash)
        object.__setattr__(
            self,
            "labels",
            MappingProxyType(labels),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blob_id": self.blob_id,
            "content_hash": self.content_hash,
            "ciphertext_hash": self.ciphertext_hash,
            "payload_hash": self.payload_hash,
            "labels_hash": self.labels_hash,
            "size": self.size,
            "labels": dict(self.labels or {}),
            "key_ref": self.key_ref,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BlobRef:
        if not isinstance(value, Mapping):
            raise ValueError("blob reference must be a mapping")
        blob_aliases = [
            item
            for item in (
                value.get("blob_hash"),
                value.get("blob_id"),
                value.get("ciphertext_hash"),
            )
            if item
        ]
        payload_aliases = [
            item
            for item in (value.get("payload_hash"), value.get("content_hash"))
            if item
        ]
        if any(not isinstance(item, str) for item in (*blob_aliases, *payload_aliases)):
            raise ValueError("blob hash aliases must be strings")
        if len(set(blob_aliases)) > 1:
            raise ValueError("conflicting blob hash aliases")
        if len(set(payload_aliases)) > 1:
            raise ValueError("conflicting payload hash aliases")
        required = {
            "blob_id",
            "content_hash",
            "ciphertext_hash",
            "payload_hash",
            "labels_hash",
            "size",
            "labels",
            "key_ref",
        }
        if set(value) != required:
            raise ValueError("invalid blob reference fields")
        for field_name in (
            "blob_id",
            "content_hash",
            "ciphertext_hash",
            "payload_hash",
            "labels_hash",
        ):
            if not _is_sha256(value[field_name]):
                raise ValueError(f"invalid {field_name}")
        if value["blob_id"] != value["ciphertext_hash"]:
            raise ValueError("conflicting blob hash aliases")
        if value["content_hash"] != value["payload_hash"]:
            raise ValueError("conflicting payload hash aliases")
        if not isinstance(value["key_ref"], str) or not value["key_ref"]:
            raise ValueError("invalid key_ref")
        size = value["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("invalid blob size")
        labels = value["labels"]
        if not isinstance(labels, dict) or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in labels.items()
        ):
            raise ValueError("invalid blob labels")
        if _sha256(_canonical_json(labels)) != value["labels_hash"]:
            raise ValueError("blob labels hash mismatch")
        return cls(
            blob_id=value["blob_id"],
            ciphertext_hash=value["ciphertext_hash"],
            payload_hash=value["payload_hash"],
            content_hash=value["content_hash"],
            labels_hash=value["labels_hash"],
            key_ref=value["key_ref"],
            size=size,
            labels=labels,
        )


class EncryptionProvider(Protocol):
    """Authenticated encryption boundary used by BlobStore."""

    def encrypt(
        self,
        plaintext: bytes,
        *,
        key_ref: str,
        aad: bytes,
        nonce: bytes,
    ) -> tuple[bytes, bytes]: ...

    def decrypt(
        self,
        ciphertext: bytes,
        *,
        key_ref: str,
        aad: bytes,
        nonce: bytes,
        tag: bytes,
    ) -> bytes: ...


class AesGcmEncryptionProvider:
    """AES-256-GCM provider backed by an injected key resolver."""

    def __init__(self, key_resolver: Callable[[str], bytes]) -> None:
        self._key_resolver = key_resolver

    def _resolve(self, key_ref: str) -> bytes:
        try:
            key = self._key_resolver(key_ref)
        except Exception as exc:
            raise KeyResolutionError(f"failed to resolve key_ref {key_ref!r}") from exc
        if not isinstance(key, bytes) or len(key) != 32:
            raise InvalidEncryptionKeyError("AES-GCM key must be exactly 32 bytes")
        return key

    def encrypt(
        self,
        plaintext: bytes,
        *,
        key_ref: str,
        aad: bytes,
        nonce: bytes,
    ) -> tuple[bytes, bytes]:
        cipher = AES.new(self._resolve(key_ref), AES.MODE_GCM, nonce=nonce)
        cipher.update(aad)
        return cipher.encrypt_and_digest(plaintext)

    def decrypt(
        self,
        ciphertext: bytes,
        *,
        key_ref: str,
        aad: bytes,
        nonce: bytes,
        tag: bytes,
    ) -> bytes:
        cipher = AES.new(self._resolve(key_ref), AES.MODE_GCM, nonce=nonce)
        cipher.update(aad)
        try:
            return cipher.decrypt_and_verify(ciphertext, tag)
        except ValueError as exc:
            raise BlobAuthenticationError("blob authentication failed") from exc


def _aad(*, key_ref: str, labels_hash: str) -> bytes:
    return _canonical_json(
        {
            "magic": BLOB_MAGIC,
            "schema_version": BLOB_SCHEMA_VERSION,
            "key_ref": key_ref,
            "labels_hash": labels_hash,
        }
    )


class BlobStore:
    """Durably publish and read encrypted content-addressed blobs."""

    def __init__(self, root: str | Path, encryption: EncryptionProvider) -> None:
        self.root = Path(root).expanduser()
        self.encryption = encryption
        self._root_fd = self._open_root(self.root)
        self._root_identity = os.fstat(self._root_fd)
        self._root_finalizer = weakref.finalize(self, os.close, self._root_fd)

    def __enter__(self) -> BlobStore:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return not self._root_finalizer.alive

    def close(self) -> None:
        if self._root_finalizer.alive:
            self._root_finalizer()
            self._root_fd = -1

    def _require_open(self) -> None:
        if self.closed:
            raise BlobIntegrityError("blob store is closed")

    def stage_and_publish(
        self,
        payload: bytes,
        labels: Mapping[str, str],
        key_ref: str,
    ) -> BlobRef:
        self._require_open()
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if not isinstance(labels, Mapping):
            raise TypeError("labels must be a mapping")
        if not isinstance(key_ref, str) or not key_ref:
            raise ValueError("key_ref must be a non-empty string")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in labels.items()
        ):
            raise ValueError("labels must contain only string keys and values")
        labels_value = dict(labels)
        labels_hash = _sha256(_canonical_json(labels_value))
        payload_hash = _sha256(payload)
        nonce = _random_nonce()
        aad = _aad(key_ref=key_ref, labels_hash=labels_hash)
        ciphertext, tag = self.encryption.encrypt(
            payload,
            key_ref=key_ref,
            aad=aad,
            nonce=nonce,
        )
        envelope = _canonical_json(
            {
                "magic": BLOB_MAGIC,
                "schema_version": BLOB_SCHEMA_VERSION,
                "key_ref": key_ref,
                "labels_hash": labels_hash,
                "payload_hash": payload_hash,
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "tag": base64.b64encode(tag).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            }
        )
        blob_hash = _sha256(envelope)
        ref = BlobRef(
            blob_hash=blob_hash,
            payload_hash=payload_hash,
            labels_hash=labels_hash,
            key_ref=key_ref,
            size=len(payload),
            labels=labels_value,
        )
        target = f"{blob_hash}.blob"
        temporary = f".blob-{secrets.token_hex(16)}.tmp"
        try:
            _write_bytes(temporary, envelope, dir_fd=self._root_fd)
            _fsync_file(temporary, dir_fd=self._root_fd)
            try:
                target_stat = os.stat(
                    target,
                    dir_fd=self._root_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                target_stat = None
            if target_stat is not None:
                if not stat.S_ISREG(target_stat.st_mode):
                    raise BlobIntegrityError(
                        "existing content address is not a regular file"
                    )
                if self._read_leaf_bytes(target) != envelope:
                    raise BlobIntegrityError(
                        "existing content address contains different bytes"
                    )
                os.unlink(temporary, dir_fd=self._root_fd)
                self._fsync_root()
                return ref
            _atomic_replace(temporary, target, dir_fd=self._root_fd)
            descriptor = self._open_leaf(target)
            os.close(descriptor)
            self._fsync_root()
            return ref
        except BlobError:
            self._unlink_if_present(temporary)
            raise
        except Exception as exc:
            self._unlink_if_present(temporary)
            boundary = next(
                (
                    name
                    for name in (
                        "_write_bytes",
                        "_fsync_file",
                        "_atomic_replace",
                        "_fsync_directory",
                    )
                    if name in str(exc)
                ),
                "blob publish",
            )
            raise BlobPublishError(f"{boundary} failed") from None

    def read(self, ref: BlobRef) -> bytes:
        self._require_open()
        if not isinstance(ref, BlobRef):
            raise TypeError("ref must be BlobRef")
        if not _is_sha256(ref.blob_hash):
            raise BlobIntegrityError("blob hash is not sha256 hex")
        if not _is_sha256(ref.payload_hash):
            raise BlobIntegrityError("payload hash is not sha256 hex")
        if not _is_sha256(ref.labels_hash):
            raise BlobIntegrityError("labels hash is not sha256 hex")
        filename = f"{ref.blob_hash}.blob"
        try:
            raw = self._read_leaf_bytes(filename)
        except FileNotFoundError as exc:
            raise BlobMissingError("blob is missing") from exc
        except OSError as exc:
            raise BlobReadError(f"blob read failed: {exc}") from exc
        if _sha256(raw) != ref.blob_hash:
            raise BlobIntegrityError("blob hash mismatch")
        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BlobFormatError("blob envelope is malformed") from exc
        required = {
            "magic",
            "schema_version",
            "key_ref",
            "labels_hash",
            "payload_hash",
            "nonce",
            "tag",
            "ciphertext",
        }
        if not isinstance(envelope, dict) or set(envelope) != required:
            raise BlobFormatError("invalid blob envelope fields")
        if envelope["magic"] != BLOB_MAGIC:
            raise BlobFormatError("invalid blob magic")
        if (
            type(envelope["schema_version"]) is not int
            or envelope["schema_version"] != BLOB_SCHEMA_VERSION
        ):
            raise BlobFormatError("unsupported blob schema")
        if envelope["key_ref"] != ref.key_ref:
            raise BlobIntegrityError("key_ref mismatch")
        if envelope["labels_hash"] != ref.labels_hash:
            raise BlobAuthenticationError("labels hash does not match reference")
        if envelope["payload_hash"] != ref.payload_hash:
            raise BlobIntegrityError("payload hash does not match reference")
        try:
            nonce = base64.b64decode(envelope["nonce"], validate=True)
            tag = base64.b64decode(envelope["tag"], validate=True)
            ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
        except (TypeError, ValueError) as exc:
            raise BlobFormatError("invalid base64 blob field") from exc
        if len(nonce) != 12 or len(tag) != 16:
            raise BlobFormatError("invalid nonce or authentication tag")
        plaintext = self.encryption.decrypt(
            ciphertext,
            key_ref=ref.key_ref,
            aad=_aad(key_ref=ref.key_ref, labels_hash=ref.labels_hash),
            nonce=nonce,
            tag=tag,
        )
        expected_labels_hash = _sha256(_canonical_json(dict(ref.labels or {})))
        if expected_labels_hash != ref.labels_hash:
            raise BlobIntegrityError("labels hash does not match labels")
        if len(plaintext) != ref.size:
            raise BlobIntegrityError("plaintext size mismatch")
        if _sha256(plaintext) != ref.payload_hash:
            raise BlobIntegrityError("payload hash mismatch")
        return plaintext

    def iter_blob_ids(self) -> tuple[str, ...]:
        """Return owned content-address IDs without parsing encrypted labels."""
        self._require_open()
        blob_ids: list[str] = []
        for filename in sorted(os.listdir(self._root_fd)):
            if not filename.endswith(".blob"):
                continue
            blob_id = filename.removesuffix(".blob")
            if not _is_sha256(blob_id):
                continue
            descriptor = self._open_leaf(filename)
            os.close(descriptor)
            blob_ids.append(blob_id)
        return tuple(blob_ids)

    def quarantine_blob(self, blob_id: str) -> Path:
        """Move one owned blob into the private quarantine directory."""
        self._require_open()
        if not _is_sha256(blob_id):
            raise BlobIntegrityError("blob id is not sha256 hex")
        filename = f"{blob_id}.blob"
        descriptor = self._open_leaf(filename)
        os.close(descriptor)
        quarantine_fd = self._open_child_directory("quarantine")
        try:
            try:
                os.stat(filename, dir_fd=quarantine_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise BlobIntegrityError("quarantine target already exists")
            os.replace(
                filename,
                filename,
                src_dir_fd=self._root_fd,
                dst_dir_fd=quarantine_fd,
            )
            os.fsync(quarantine_fd)
            self._fsync_root()
        finally:
            os.close(quarantine_fd)
        return self.root / "quarantine" / filename

    def restore_quarantined_blob(self, blob_id: str) -> Path:
        """Restore one authenticated owned blob from private quarantine."""

        self._require_open()
        if not _is_sha256(blob_id):
            raise BlobIntegrityError("blob id is not sha256 hex")
        filename = f"{blob_id}.blob"
        try:
            descriptor = self._open_leaf(filename)
        except FileNotFoundError:
            descriptor = -1
        if descriptor >= 0:
            os.close(descriptor)
            return self.root / filename

        quarantine_fd = self._open_child_directory("quarantine")
        source_fd = -1
        try:
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                source_fd = os.open(filename, flags, dir_fd=quarantine_fd)
            except FileNotFoundError as exc:
                raise BlobMissingError("quarantined blob is missing") from exc
            source_stat = os.fstat(source_fd)
            if (
                not stat.S_ISREG(source_stat.st_mode)
                or stat.S_IMODE(source_stat.st_mode) != 0o600
            ):
                raise BlobIntegrityError(
                    "quarantined blob leaf must be a regular 0600 file"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            if _sha256(b"".join(chunks)) != blob_id:
                raise BlobIntegrityError("quarantined blob content address mismatch")
            os.close(source_fd)
            source_fd = -1
            os.replace(
                filename,
                filename,
                src_dir_fd=quarantine_fd,
                dst_dir_fd=self._root_fd,
            )
            descriptor = self._open_leaf(filename)
            os.close(descriptor)
            os.fsync(quarantine_fd)
            self._fsync_root()
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            os.close(quarantine_fd)
        return self.root / filename

    @staticmethod
    def _open_root(root: Path) -> int:
        parts = root.parts[1:] if root.is_absolute() else root.parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise BlobPublishError("invalid blob root")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = -1
        try:
            current_path = Path("/") if root.is_absolute() else Path(".")
            descriptor = os.open(str(current_path), flags)
            for part in parts:
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    os.fsync(descriptor)
                    _fsync_directory(current_path)
                    child = os.open(part, flags, dir_fd=descriptor)
                child_stat = os.fstat(child)
                if not stat.S_ISDIR(child_stat.st_mode):
                    os.close(child)
                    raise BlobPublishError("blob root is not a directory")
                os.close(descriptor)
                descriptor = child
                current_path /= part
            os.fchmod(descriptor, 0o700)
            return descriptor
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            raise BlobPublishError("failed to open blob root") from None

    def _open_leaf(self, filename: str) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(filename, flags, dir_fd=self._root_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise BlobIntegrityError("blob leaf is not a safe regular file") from exc
        leaf_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(leaf_stat.st_mode)
            or stat.S_IMODE(leaf_stat.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise BlobIntegrityError("blob leaf must be a regular 0600 file")
        return descriptor

    def _read_leaf_bytes(self, filename: str) -> bytes:
        descriptor = self._open_leaf(filename)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(descriptor)

    def _open_child_directory(self, name: str) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            os.mkdir(name, mode=0o700, dir_fd=self._root_fd)
            os.fsync(self._root_fd)
        except FileExistsError:
            pass
        try:
            descriptor = os.open(name, flags, dir_fd=self._root_fd)
            os.fchmod(descriptor, 0o700)
            return descriptor
        except Exception:
            raise BlobIntegrityError("invalid blob quarantine directory") from None

    def _unlink_if_present(self, filename: str) -> None:
        try:
            os.unlink(filename, dir_fd=self._root_fd)
        except (FileNotFoundError, OSError):
            pass

    def _fsync_root(self) -> None:
        os.fsync(self._root_fd)
        try:
            current = os.stat(self.root, follow_symlinks=False)
        except OSError:
            return
        if (
            current.st_dev == self._root_identity.st_dev
            and current.st_ino == self._root_identity.st_ino
        ):
            _fsync_directory(self.root)
