"""Document materializers and canonical employee memory reads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

from .models import DataKind
from .projection import DataProjectionState

_AGENT_ID_PATTERN = re.compile(r"agt_[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass(frozen=True)
class DocumentPath:
    """Resolved canonical path for a materialized document."""

    agent_id: str
    kind: DataKind
    relative: str
    absolute: Path


class EmployeeDocumentMaterializer:
    """Writes canonical employee documents from projection state."""

    def __init__(self, agents_root: str | Path) -> None:
        self._root = Path(agents_root).expanduser().absolute()
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_fd = _open_root_fd(self._root)
        os.close(root_fd)

    @property
    def root(self) -> Path:
        return self._root

    def resolve_path(self, agent_id: str, kind: DataKind, source_id: str = "") -> DocumentPath:
        _require_agent_id(agent_id)
        if kind is DataKind.L1_MEMORY:
            relative = "memory/MEMORY.md"
        elif kind is DataKind.MEMORY_SUMMARY:
            safe_id = hashlib.sha256(source_id.encode()).hexdigest()[:16]
            relative = f"memory/summary_{safe_id}.md"
        elif kind is DataKind.SKILL_PROFILE:
            relative = "skill_profile.json"
        elif kind is DataKind.REASONING:
            safe_id = hashlib.sha256(source_id.encode()).hexdigest()
            relative = f"reasoning/{safe_id}.json"
        elif kind is DataKind.KNOWLEDGE_PAGE:
            if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", source_id) is None:
                raise ValueError("invalid knowledge page id")
            relative = f"workspace/wiki/{source_id}.md"
        elif kind is DataKind.KNOWLEDGE_INDEX:
            relative = "workspace/wiki/index.md"
        elif kind is DataKind.KNOWLEDGE_REVIEW:
            safe_id = hashlib.sha256(source_id.encode()).hexdigest()
            relative = f"knowledge/review/{safe_id}.json"
        else:
            raise ValueError(f"unsupported kind: {kind}")
        return DocumentPath(
            agent_id=agent_id,
            kind=kind,
            relative=relative,
            absolute=self._root / agent_id / relative,
        )

    def materialize(
        self,
        agent_id: str,
        kind: DataKind,
        source_id: str,
        content: bytes,
        content_hash: str,
    ) -> DocumentPath:
        """Write one document file atomically with fsync."""
        if hashlib.sha256(content).hexdigest() != content_hash:
            raise ValueError("content hash verification failed")
        doc_path = self.resolve_path(agent_id, kind, source_id)
        with _open_rooted_parent(
            self._root,
            _document_storage_relative(doc_path),
            create=True,
        ) as (parent_fd, filename):
            stem = Path(filename).stem
            self._cleanup_stale_temps(parent_fd, stem)
            temp_name = f".{stem}-{secrets.token_hex(8)}.tmp"
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
            try:
                os.fchmod(fd, 0o600)
                _write_all(fd, content)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rename(
                temp_name,
                filename,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        return doc_path

    def materialize_from_state(
        self,
        state: DataProjectionState,
        agent_id: str,
        blob_reader: Any,
    ) -> list[DocumentPath]:
        """Rebuild all documents for one agent from projected metadata + blob reads."""
        paths: list[DocumentPath] = []
        for doc_id, metadata in state.employee_documents.items():
            if metadata.agent_id != agent_id or metadata.tombstoned:
                continue
            latest_key = (
                metadata.tenant_key,
                metadata.agent_id,
                metadata.kind.value,
                metadata.source_id,
            )
            if state.latest_employee_document.get(latest_key) != doc_id:
                continue
            from ..journal.blob_store import BlobRef
            ref = BlobRef.from_dict(metadata.blob_ref)
            content = blob_reader(ref)
            path = self.materialize(
                agent_id=agent_id,
                kind=metadata.kind,
                source_id=metadata.source_id,
                content=content,
                content_hash=metadata.content_hash,
            )
            paths.append(path)
        return paths

    def verify(self, agent_id: str, kind: DataKind, source_id: str, expected_hash: str) -> bool:
        """Check if file on disk matches expected content hash."""
        doc_path = self.resolve_path(agent_id, kind, source_id)
        try:
            content = _read_rooted_regular_bytes(
                self._root,
                _document_storage_relative(doc_path),
                missing_ok=True,
            )
        except MemoryIntegrityError:
            return False
        if content is None:
            return False
        return hashlib.sha256(content).hexdigest() == expected_hash

    @staticmethod
    def _cleanup_stale_temps(directory_fd: int, stem: str) -> None:
        prefix = f".{stem}-"
        for name in os.listdir(directory_fd):
            if name.startswith(prefix) and name.endswith(".tmp"):
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except OSError:
                    pass


class EmployeeMemoryFacade:
    """Read only projection-owned canonical employee documents."""

    def __init__(
        self,
        *,
        materializer: EmployeeDocumentMaterializer,
        state: DataProjectionState,
        projection_guard: Callable[[], ContextManager[Any]] | None = None,
    ) -> None:
        self._materializer = materializer
        self._state = state
        self._projection_guard = projection_guard or nullcontext

    def read_l1(
        self,
        agent_id: str,
        tenant_key: str,
    ) -> str | None:
        """Read tenant-owned L1, distinguishing absence from corruption."""
        if not isinstance(tenant_key, str) or not tenant_key.strip():
            raise MemoryAccessError("tenant_key is required")
        with self._projection_guard():
            return self._read_projected_canonical(
                tenant_key,
                agent_id,
                DataKind.L1_MEMORY,
                "l1_memory",
            )

    def read_memory_summary(
        self, agent_id: str, tenant_key: str, chat_id: str, thread_root_id: str = ""
    ) -> str | None:
        """Read chat-scoped memory summary."""
        from .models import EmployeeDataDocumentV1
        source_id = EmployeeDataDocumentV1.memory_summary_source_id(
            chat_id=chat_id, thread_root_id=thread_root_id
        )
        with self._projection_guard():
            return self._read_projected_canonical(
                tenant_key,
                agent_id,
                DataKind.MEMORY_SUMMARY,
                source_id,
            )

    def read_skill_profile(self, agent_id: str) -> dict | None:
        """Read skill profile JSON."""
        content = self._read_canonical(agent_id, DataKind.SKILL_PROFILE, "skill_profile")
        if content is None:
            return None
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None

    def is_canonical(self, agent_id: str) -> bool:
        """Check if agent has any canonical documents."""
        for key in self._state.latest_employee_document:
            if key[1] == agent_id:
                return True
        return False

    def _read_canonical(self, agent_id: str, kind: DataKind, source_id: str) -> str | None:
        doc_path = self._materializer.resolve_path(agent_id, kind, source_id)
        try:
            return _read_rooted_regular_text(
                self._materializer.root,
                _document_storage_relative(doc_path),
                missing_ok=True,
            )
        except MemoryIntegrityError:
            return None

    def _read_projected_canonical(
        self,
        tenant_key: str,
        agent_id: str,
        kind: DataKind,
        source_id: str,
    ) -> str | None:
        key = (tenant_key, agent_id, kind.value, source_id)
        foreign = [
            other_key
            for other_key in self._state.latest_employee_document
            if other_key[1:] == key[1:] and other_key[0] != tenant_key
        ]
        if foreign:
            raise MemoryAccessError("document belongs to another tenant")
        document_id = self._state.latest_employee_document.get(key)
        doc_path = self._materializer.resolve_path(agent_id, kind, source_id)
        if document_id is None:
            if _rooted_path_exists(
                self._materializer.root,
                _document_storage_relative(doc_path),
            ):
                raise MemoryIntegrityError(
                    "materialized document has no projected owner"
                )
            return None
        metadata = self._state.employee_documents.get(document_id)
        if metadata is None:
            raise MemoryIntegrityError("latest document metadata is missing")
        if (
            metadata.tombstoned
            or metadata.document_id != document_id
            or metadata.tenant_key != tenant_key
            or metadata.agent_id != agent_id
            or metadata.kind is not kind
            or metadata.source_id != source_id
        ):
            raise MemoryIntegrityError("projected document binding is invalid")
        content = _read_rooted_regular_text(
            self._materializer.root,
            _document_storage_relative(doc_path),
            missing_ok=False,
        )
        assert content is not None
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != metadata.content_hash:
            raise MemoryIntegrityError("materialized document hash mismatch")
        return content

class MemoryAccessError(RuntimeError):
    """Memory ownership cannot authorize this read."""


class MemoryIntegrityError(RuntimeError):
    """Projected and materialized memory do not form one valid document."""


class _RootedPathMissing(FileNotFoundError):
    pass


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise MemoryIntegrityError("memory file write failed")
        view = view[written:]


def _document_storage_relative(doc_path: DocumentPath) -> str:
    return f"{doc_path.agent_id}/{doc_path.relative}"


def _require_agent_id(agent_id: str) -> None:
    if (
        not isinstance(agent_id, str)
        or _AGENT_ID_PATTERN.fullmatch(agent_id) is None
    ):
        raise MemoryAccessError("agent_id must be canonical")


def _open_root_fd(root: Path) -> int:
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(root, flags)
    except FileNotFoundError:
        raise _RootedPathMissing from None
    except OSError:
        raise MemoryIntegrityError("memory root is not trusted") from None
    root_stat = os.fstat(fd)
    if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid():
        os.close(fd)
        raise MemoryIntegrityError("memory root identity is invalid")
    return fd


def _relative_parts(relative: str) -> tuple[str, ...]:
    parts = tuple(Path(relative).parts)
    if (
        not parts
        or any(part in ("", ".", "..") or "/" in part for part in parts)
        or Path(relative).is_absolute()
    ):
        raise MemoryIntegrityError("memory relative path is invalid")
    return parts


@contextmanager
def _open_rooted_parent(
    root: Path,
    relative: str,
    *,
    create: bool,
) -> Iterator[tuple[int, str]]:
    parts = _relative_parts(relative)
    opened: list[int] = []
    try:
        opened.append(_open_root_fd(root))
        directory_flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for component in parts[:-1]:
            try:
                child_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=opened[-1],
                )
            except FileNotFoundError:
                if not create:
                    raise _RootedPathMissing from None
                try:
                    os.mkdir(component, 0o700, dir_fd=opened[-1])
                except FileExistsError:
                    pass
                except OSError:
                    raise MemoryIntegrityError(
                        "memory parent directory cannot be created"
                    ) from None
                try:
                    child_fd = os.open(
                        component,
                        directory_flags,
                        dir_fd=opened[-1],
                    )
                except OSError:
                    raise MemoryIntegrityError(
                        "memory parent directory is not trusted"
                    ) from None
            except OSError:
                raise MemoryIntegrityError(
                    "memory parent directory is not trusted"
                ) from None
            child_stat = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(child_stat.st_mode)
                or child_stat.st_uid != os.getuid()
            ):
                os.close(child_fd)
                raise MemoryIntegrityError(
                    "memory parent directory identity is invalid"
                )
            opened.append(child_fd)
        yield opened[-1], parts[-1]
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _rooted_path_exists(root: Path, relative: str) -> bool:
    try:
        with _open_rooted_parent(root, relative, create=False) as (
            parent_fd,
            filename,
        ):
            try:
                os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
                return True
            except FileNotFoundError:
                return False
            except OSError:
                raise MemoryIntegrityError(
                    "memory path inspection failed"
                ) from None
    except _RootedPathMissing:
        return False


def _read_rooted_regular_bytes(
    root: Path,
    relative: str,
    *,
    missing_ok: bool,
) -> bytes | None:
    try:
        with _open_rooted_parent(root, relative, create=False) as (
            parent_fd,
            filename,
        ):
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                fd = os.open(filename, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise MemoryIntegrityError(
                    "projected memory file is missing"
                ) from None
            except OSError:
                raise MemoryIntegrityError("memory file open failed") from None
            try:
                file_stat = os.fstat(fd)
                if (
                    not stat.S_ISREG(file_stat.st_mode)
                    or file_stat.st_uid != os.getuid()
                ):
                    raise MemoryIntegrityError(
                        "memory file identity is invalid"
                    )
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(fd, 64 * 1024)
                    if not chunk:
                        return b"".join(chunks)
                    chunks.append(chunk)
            finally:
                os.close(fd)
    except _RootedPathMissing:
        if missing_ok:
            return None
        raise MemoryIntegrityError("projected memory file is missing") from None


def _read_rooted_regular_text(
    root: Path,
    relative: str,
    *,
    missing_ok: bool,
) -> str | None:
    content = _read_rooted_regular_bytes(
        root,
        relative,
        missing_ok=missing_ok,
    )
    if content is None:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        raise MemoryIntegrityError("memory file is not valid UTF-8") from None
