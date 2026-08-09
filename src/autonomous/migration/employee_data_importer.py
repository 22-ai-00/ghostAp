"""Idempotent one-time migration of legacy Employee execution history and memory."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.models import (
    DataKind,
    EmployeeDataDocumentV1,
    ExecutionAttemptContext,
    ExecutionHistoryPayloadV1,
    ExecutionHistoryRecordV1,
    SafeExecutionSummary,
)
from ..data.service import DataConflictError, EmployeeDataService
from ..journal.writer import CommitState


@dataclass(frozen=True)
class LegacySourceLocator:
    """Stable locator for one legacy file."""

    relative_path: str
    kind: str

    @property
    def locator_hash(self) -> str:
        return hashlib.sha256(
            f"{self.relative_path}|{self.kind}".encode()
        ).hexdigest()


@dataclass
class ImportManifest:
    """Per-file import tracking."""

    locator_hash: str
    content_hash: str
    state: str = "pending"
    imported_ids: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class EmployeeDataImportResult:
    """Result of one employee data migration run."""

    agent_id: str
    history_imported: int = 0
    history_skipped: int = 0
    documents_imported: int = 0
    documents_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)


class EmployeeDataImporter:
    """Migrates legacy execution_history.jsonl and memory files for one employee."""

    def __init__(
        self,
        *,
        service: EmployeeDataService,
        legacy_base: str | Path,
        tenant_key: str,
        owner_principal_id: str,
    ) -> None:
        self._service = service
        self._legacy_base = Path(legacy_base).expanduser()
        self._tenant_key = tenant_key
        self._owner_principal_id = owner_principal_id

    def import_employee(self, agent_id: str) -> EmployeeDataImportResult:
        """Run idempotent migration for one canonical employee."""
        result = EmployeeDataImportResult(agent_id=agent_id)
        agent_dir = self._legacy_base / "agents" / agent_id
        try:
            mode = agent_dir.lstat().st_mode
        except FileNotFoundError:
            return result
        if not stat.S_ISDIR(mode):
            result.errors.append("legacy agent directory is not a regular directory")
            return result
        self._import_history(agent_id, agent_dir, result)
        self._import_l1_memory(agent_id, agent_dir, result)
        self._import_skill_profile(agent_id, agent_dir, result)
        self._import_reasoning(agent_id, agent_dir, result)
        return result

    def _import_history(
        self, agent_id: str, agent_dir: Path, result: EmployeeDataImportResult
    ) -> None:
        history_file = agent_dir / "execution_history.jsonl"
        try:
            raw = self._read_regular_file(history_file)
        except FileNotFoundError:
            return
        except OSError as exc:
            result.errors.append(f"history read failed: {exc}")
            return
        locator = LegacySourceLocator(
            relative_path=f"agents/{agent_id}/execution_history.jsonl",
            kind="execution_history",
        )
        state = self._service.state
        import_key = (self._tenant_key, agent_id, locator.locator_hash)
        if import_key in state.legacy_data_sources:
            result.history_skipped += 1
            return
        content_hash = hashlib.sha256(raw).hexdigest()
        lines = raw.strip().split(b"\n") if raw.strip() else []
        imported_ids: list[str] = []
        errors_before = len(result.errors)
        for ordinal, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                result.errors.append(f"history line {ordinal}: malformed JSON")
                continue
            record_id = self._import_history_row(agent_id, row, ordinal, locator, content_hash, result)
            if record_id:
                imported_ids.append(record_id)
        if len(result.errors) != errors_before:
            return
        self._commit_import_event(agent_id, locator, content_hash, imported_ids)

    def _import_history_row(
        self,
        agent_id: str,
        row: dict[str, Any],
        ordinal: int,
        locator: LegacySourceLocator,
        file_hash: str,
        result: EmployeeDataImportResult,
    ) -> str | None:
        try:
            row_id = hashlib.sha256(
                f"{locator.locator_hash}|{file_hash}|{ordinal}".encode()
            ).hexdigest()[:16]
            ts = row.get("timestamp") or row.get("ended_at") or row.get("ts", "")
            if isinstance(ts, (int, float)):
                from datetime import UTC, datetime
                ended_at = datetime.fromtimestamp(ts, tz=UTC).isoformat()
            else:
                ended_at = str(ts) if ts else "2026-01-01T00:00:00+00:00"
            if not ended_at.endswith("+00:00") and not ended_at.endswith("Z"):
                ended_at = ended_at + "+00:00"
            ended_at = ended_at.replace("Z", "+00:00")
            duration = row.get("duration_ms") or row.get("duration", 0)
            if isinstance(duration, float):
                duration = int(duration)
            success = row.get("success", row.get("status") == "completed")
            status = "completed" if success else "failed"
            started_at_str = row.get("started_at", "")
            if not started_at_str:
                from datetime import UTC, datetime, timedelta
                ended_dt = datetime.fromisoformat(ended_at)
                started_dt = ended_dt - timedelta(milliseconds=max(duration, 0))
                started_at_str = started_dt.isoformat()
            if not started_at_str.endswith("+00:00"):
                started_at_str = started_at_str.replace("Z", "+00:00")
                if not started_at_str.endswith("+00:00"):
                    started_at_str += "+00:00"
            run_id = f"legacy_run_{row_id}"
            attempt_id = f"legacy_attempt_{row_id}"
            ctx = ExecutionAttemptContext(
                tenant_key=self._tenant_key,
                agent_id=agent_id,
                owner_principal_id=self._owner_principal_id,
                requester_principal_id="legacy_attribution_unknown",
                task_id=row.get("task_id", f"legacy_task_{row_id}"),
                run_id=run_id,
                attempt_id=attempt_id,
                message_id=row.get("message_id", f"legacy_msg_{row_id}"),
                thread_root_id=row.get("thread_root_id", ""),
                chat_id=row.get("chat_id", "legacy_unknown"),
                tool=row.get("tool", "unknown"),
                model=row.get("model", "unknown"),
                effort=row.get("effort", "unknown"),
                started_at=started_at_str,
                terminal_epoch=1,
            )
            record = ExecutionHistoryRecordV1.from_attempt(
                ctx,
                ended_at=ended_at,
                status=status,
                safe_summary=SafeExecutionSummary.build(
                    status=status, tool_count=0, attachment_count=0
                ),
                prompt_tokens=row.get("prompt_tokens", 0),
                completion_tokens=row.get("completion_tokens", 0),
                tool_usage=(),
                predecessor_sequence=0,
                predecessor_hash="",
                shard_timezone="UTC",
            )
            payload = ExecutionHistoryPayloadV1(
                record_id=record.record_id,
                occurrence_key=record.occurrence_key,
                request_text=str(row.get("prompt", ""))[:100_000],
                result_text=str(row.get("result", ""))[:100_000],
                error_detail=str(row.get("error", ""))[:10_000],
            )
            self._service.record_history(record, payload)
            result.history_imported += 1
            return record.record_id
        except DataConflictError:
            result.history_skipped += 1
            return None
        except (ValueError, TypeError, KeyError) as exc:
            result.errors.append(f"history row {ordinal}: {type(exc).__name__}")
            return None

    def _import_l1_memory(
        self, agent_id: str, agent_dir: Path, result: EmployeeDataImportResult
    ) -> None:
        memory_path = agent_dir / "memory" / "MEMORY.md"
        root_path = agent_dir / "MEMORY.md"
        if memory_path.exists() and root_path.exists():
            result.errors.append("multiple legacy L1 files exist")
            return
        root_legacy = False
        if not memory_path.exists():
            memory_path = root_path
            root_legacy = memory_path.exists()
        if not memory_path.exists():
            return
        imported = self._import_document(
            agent_id, memory_path, DataKind.L1_MEMORY, "l1_memory",
            "text/markdown", result,
        )
        if imported and root_legacy:
            try:
                self._retire_root_memory(agent_dir, memory_path)
            except OSError as exc:
                result.errors.append(f"root memory retirement failed: {type(exc).__name__}")

    def _import_skill_profile(
        self, agent_id: str, agent_dir: Path, result: EmployeeDataImportResult
    ) -> None:
        skill_path = agent_dir / "skill_profile.json"
        if not skill_path.exists():
            return
        self._import_document(
            agent_id, skill_path, DataKind.SKILL_PROFILE, "skill_profile",
            "application/json", result,
        )

    def _import_reasoning(
        self, agent_id: str, agent_dir: Path, result: EmployeeDataImportResult
    ) -> None:
        reasoning_dir = agent_dir / "reasoning"
        try:
            mode = reasoning_dir.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(mode):
            result.errors.append("reasoning source is not a regular directory")
            return
        for path in sorted(reasoning_dir.iterdir()):
            if not path.is_file() or not path.name.endswith(".json"):
                continue
            source_id = path.stem
            self._import_document(
                agent_id, path, DataKind.REASONING, source_id,
                "application/json", result,
            )

    def _import_document(
        self,
        agent_id: str,
        file_path: Path,
        kind: DataKind,
        source_id: str,
        content_type: str,
        result: EmployeeDataImportResult,
    ) -> bool:
        try:
            content = self._read_regular_file(file_path)
        except OSError as exc:
            result.errors.append(f"document read failed: {exc}")
            return False
        content_hash = hashlib.sha256(content).hexdigest()
        doc_id = f"data_{hashlib.sha256(f'{agent_id}|{kind.value}|{source_id}'.encode()).hexdigest()[:16]}"
        from datetime import UTC, datetime
        doc = EmployeeDataDocumentV1(
            document_id=doc_id,
            tenant_key=self._tenant_key,
            agent_id=agent_id,
            owner_principal_id=self._owner_principal_id,
            kind=kind,
            version=1,
            source_id=source_id,
            created_at=datetime.now(UTC).isoformat(),
            predecessor_sequence=0,
            predecessor_hash="",
            content_type=content_type,
            content_hash=content_hash,
        )
        try:
            self._service.publish_document(doc, content)
            result.documents_imported += 1
            return True
        except DataConflictError:
            existing = self._service.state.employee_documents.get(doc_id)
            if (
                existing is not None
                and existing.tenant_key == self._tenant_key
                and existing.agent_id == agent_id
                and existing.kind is kind
                and existing.source_id == source_id
                and existing.content_hash == content_hash
            ):
                result.documents_skipped += 1
                return True
            result.errors.append(f"document {kind.value}/{source_id}: conflict")
            return False
        except (ValueError, TypeError) as exc:
            result.errors.append(f"document {kind.value}/{source_id}: {type(exc).__name__}")
            return False

    @staticmethod
    def _read_regular_file(path: Path) -> bytes:
        """Open a legacy leaf without following symlinks and verify its inode type."""

        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("legacy source is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return stream.read()
        finally:
            os.close(descriptor)

    @staticmethod
    def _retire_root_memory(agent_dir: Path, source: Path) -> None:
        """Move a verified imported root MEMORY out of all active read paths."""

        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        retired_dir = agent_dir / ".legacy-imported"
        retired_dir.mkdir(mode=0o700, exist_ok=True)
        target = retired_dir / f"MEMORY.{digest}.md"
        if target.exists():
            if not stat.S_ISREG(target.lstat().st_mode) or target.read_bytes() != content:
                raise OSError("legacy retirement target conflict")
            source.unlink()
        else:
            os.replace(source, target)
        directory_fd = os.open(agent_dir, os.O_RDONLY | os.O_DIRECTORY)
        retired_fd = os.open(retired_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
            os.fsync(retired_fd)
        finally:
            os.close(retired_fd)
            os.close(directory_fd)

    def _commit_import_event(
        self,
        agent_id: str,
        locator: LegacySourceLocator,
        content_hash: str,
        imported_ids: list[str],
    ) -> None:
        from ..journal.frame import JournalEvent
        event = JournalEvent(
            event_type="employee.legacy_data_imported",
            aggregate_id=f"legacy-data:{hashlib.sha256(self._tenant_key.encode()).hexdigest()[:16]}:{agent_id}:{locator.locator_hash[:16]}",
            payload={
                "tenant_key": self._tenant_key,
                "agent_id": agent_id,
                "source_locator_hash": locator.locator_hash,
                "content_hash": content_hash,
                "imported_ids": imported_ids[:100],
                "state": "imported",
            },
        )
        writer = self._service._writer
        with self._service.employee_dispatch_guard(), writer.transaction_guard():
            self._service.synchronize_projection_unlocked()
            self._service._require_write_authority_unlocked()
            versions = writer.get_aggregate_versions([event.aggregate_id])
            result = writer.commit(
                [event],
                versions,
                expected_head_sequence=self._service.state.cursor_sequence,
                expected_head_hash=self._service.state.cursor_hash or None,
            )
            if result.state is not CommitState.ANCHORED:
                raise RuntimeError("legacy import manifest was not anchored")
            self._service.apply_committed_frame_unlocked(result.frame)
