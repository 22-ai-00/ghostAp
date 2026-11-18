"""Daily JSONL history materializer with descriptor-relative no-follow writes."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .projection import DataProjectionState, HistoryMetadataRecord


@dataclass(frozen=True)
class ShardManifest:
    """Integrity manifest for one daily JSONL shard."""

    tenant_key: str
    agent_id: str
    day: str
    source_sequence: int
    source_hash: str
    content_hash: str
    row_count: int


class DailyHistoryMaterializer:
    """Writes and repairs disposable daily JSONL shards from projection state."""

    def __init__(self, agents_root: str | Path) -> None:
        self._root = Path(agents_root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def materialize_day(
        self,
        state: DataProjectionState,
        tenant_key: str,
        agent_id: str,
        day: str,
    ) -> ShardManifest | None:
        """Write one day's JSONL shard. Returns manifest or None if empty."""
        day_key = (tenant_key, agent_id, day)
        record_ids = state.history_by_employee_day.get(day_key, [])
        if not record_ids:
            return None
        rows: list[bytes] = []
        max_sequence = 0
        max_hash = ""
        for record_id in record_ids:
            metadata = state.history_records.get(record_id)
            if metadata is None or metadata.tombstoned:
                continue
            row = self._metadata_to_row(metadata)
            rows.append(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
            if metadata.publish_sequence > max_sequence:
                max_sequence = metadata.publish_sequence
                max_hash = metadata.publish_frame_hash
        if not rows:
            return None
        content = b"\n".join(rows) + b"\n"
        content_hash = hashlib.sha256(content).hexdigest()
        history_dir = self._root / agent_id / "history"
        history_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._cleanup_stale_temps(history_dir, day)
        target = history_dir / f"{day}.jsonl"
        temp_name = f".{day}-{secrets.token_hex(8)}.tmp"
        temp_path = history_dir / temp_name
        flags = (
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(str(temp_path), flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, content)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(temp_path), str(target))
        self._fsync_dir(history_dir)
        return ShardManifest(
            tenant_key=tenant_key,
            agent_id=agent_id,
            day=day,
            source_sequence=max_sequence,
            source_hash=max_hash,
            content_hash=content_hash,
            row_count=len(rows),
        )

    def materialize_all(
        self,
        state: DataProjectionState,
    ) -> list[ShardManifest]:
        """Rebuild all shards deterministically."""
        manifests: list[ShardManifest] = []
        for (tenant_key, agent_id, day) in sorted(state.history_by_employee_day):
            manifest = self.materialize_day(state, tenant_key, agent_id, day)
            if manifest is not None:
                manifests.append(manifest)
        return manifests

    @staticmethod
    def _metadata_to_row(metadata: HistoryMetadataRecord) -> dict[str, Any]:
        return {
            "record_id": metadata.record_id,
            "occurrence_key": metadata.occurrence_key,
            "tenant_key": metadata.tenant_key,
            "agent_id": metadata.agent_id,
            "requester_principal_id": metadata.requester_principal_id,
            "task_id": metadata.task_id,
            "shard_day": metadata.shard_day,
            "status": metadata.status,
            "started_at": metadata.started_at,
            "ended_at": metadata.ended_at,
            "duration_ms": metadata.duration_ms,
            "tool": metadata.tool,
            "model": metadata.model,
            "effort": metadata.effort,
            "safe_summary": metadata.safe_summary_text,
            "prompt_tokens": metadata.prompt_tokens,
            "completion_tokens": metadata.completion_tokens,
            "total_tokens": metadata.total_tokens,
            "publish_sequence": metadata.publish_sequence,
            "blob_payload_hash": metadata.blob_ref.get("content_hash", ""),
        }

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _cleanup_stale_temps(directory: Path, day: str) -> None:
        """Remove leftover .tmp files from a prior crashed materialization."""
        prefix = f".{day}-"
        for entry in directory.iterdir():
            if entry.name.startswith(prefix) and entry.name.endswith(".tmp"):
                try:
                    entry.unlink()
                except OSError:
                    pass
