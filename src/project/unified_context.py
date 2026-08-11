"""Chat-scoped, in-memory context shared by programming modes."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ContextEntryType(Enum):
    CONVERSATION = "conversation"
    SESSION_SNAPSHOT = "session_snapshot"
    MODE_TRANSITION = "mode_transition"
    DEEP_ENGINE_RESULT = "deep_result"
    AI_SUMMARY = "ai_summary"
    FILE_CHANGE = "file_change"


class ContextSourceMode(Enum):
    SMART = "smart"
    COCO = "coco"
    CLAUDE = "claude"
    AIDEN = "aiden"
    CODEX = "codex"
    GEMINI = "gemini"
    TRAEX = "traex"
    GROK = "grok"
    SHELL = "shell"
    DEEP_ENGINE = "deep_engine"


@dataclass
class ContextEntry:
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    seq: int = 0
    entry_type: ContextEntryType = ContextEntryType.CONVERSATION
    source_mode: ContextSourceMode = ContextSourceMode.SMART
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ContextVersion:
    version_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    version_number: int = 0
    reason: str = ""
    source_mode: ContextSourceMode = ContextSourceMode.SMART
    summary: str = ""
    entry_count: int = 0
    last_seq: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class ContextBridgeSummary:
    from_mode: ContextSourceMode = ContextSourceMode.SMART
    to_mode: ContextSourceMode = ContextSourceMode.SMART
    summary_text: str = ""
    key_decisions: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_injection_prompt(self) -> str:
        parts = [f"[Context from previous {self.from_mode.value} session]"]
        if self.summary_text:
            parts.append(self.summary_text)
        if self.key_decisions:
            parts.append("Key decisions:")
            parts.extend(f"  - {item}" for item in self.key_decisions)
        if self.files_modified:
            parts.append(f"Files modified: {', '.join(self.files_modified)}")
        if self.pending_tasks:
            parts.append("Pending tasks:")
            parts.extend(f"  - {item}" for item in self.pending_tasks)
        parts.append("[End of context]")
        return "\n".join(parts)


class UnifiedContext:
    """One bounded context stream for a single chat/project pair."""

    def __init__(
        self,
        project_id: str,
        max_entries: int = 200,
        max_versions: int = 50,
    ) -> None:
        self.project_id = project_id
        self.max_entries = max_entries
        self.max_versions = max_versions
        self._mu = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._entries: list[ContextEntry] = []
        self._versions: list[ContextVersion] = []
        self._version_number = 0
        self._next_seq = 1
        self._last_bridge: Optional[ContextBridgeSummary] = None

    @property
    def entries(self) -> list[ContextEntry]:
        with self._mu:
            return list(self._entries)

    @property
    def versions(self) -> list[ContextVersion]:
        with self._mu:
            return list(self._versions)

    @property
    def last_bridge_summary(self) -> Optional[ContextBridgeSummary]:
        with self._mu:
            return self._last_bridge

    @property
    def entry_count(self) -> int:
        with self._mu:
            return len(self._entries)

    def add_entry(self, entry: ContextEntry) -> ContextEntry:
        with self._mu:
            if entry.seq <= 0:
                entry.seq = self._next_seq
                self._next_seq += 1
            self._entries.append(entry)
            if self.max_entries > 0 and len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries :]
            return entry

    def add_conversation(
        self,
        role: str,
        content: str,
        source_mode: ContextSourceMode,
        message_id: Optional[str] = None,
    ) -> ContextEntry:
        return self.add_entry(
            ContextEntry(
                entry_type=ContextEntryType.CONVERSATION,
                source_mode=source_mode,
                content=content,
                metadata={"role": role, "message_id": message_id},
            )
        )

    def add_session_snapshot(
        self,
        session_data: dict,
        source_mode: ContextSourceMode,
    ) -> ContextEntry:
        return self.add_entry(
            ContextEntry(
                entry_type=ContextEntryType.SESSION_SNAPSHOT,
                source_mode=source_mode,
                content=f"Session ended: {session_data.get('session_id', 'unknown')}",
                metadata=session_data,
            )
        )

    def add_mode_transition(
        self,
        from_mode: ContextSourceMode,
        to_mode: ContextSourceMode,
        reason: str = "",
    ) -> ContextEntry:
        return self.add_entry(
            ContextEntry(
                entry_type=ContextEntryType.MODE_TRANSITION,
                source_mode=from_mode,
                content=f"{from_mode.value} -> {to_mode.value}",
                metadata={
                    "from_mode": from_mode.value,
                    "to_mode": to_mode.value,
                    "reason": reason,
                },
            )
        )

    def add_deep_engine_result(self, data: dict) -> ContextEntry:
        return self.add_entry(
            ContextEntry(
                entry_type=ContextEntryType.DEEP_ENGINE_RESULT,
                source_mode=ContextSourceMode.DEEP_ENGINE,
                content=f"Deep Engine completed: {data.get('name', 'unknown')}",
                metadata=data,
            )
        )

    def get_entries_by_type(
        self,
        entry_type: ContextEntryType,
    ) -> list[ContextEntry]:
        with self._mu:
            return [entry for entry in self._entries if entry.entry_type == entry_type]

    def create_version(
        self,
        reason: str,
        source_mode: ContextSourceMode,
        summary: str = "",
    ) -> ContextVersion:
        with self._mu:
            self._version_number += 1
            version = ContextVersion(
                version_number=self._version_number,
                reason=reason,
                source_mode=source_mode,
                summary=summary,
                entry_count=len(self._entries),
                last_seq=self._entries[-1].seq if self._entries else 0,
            )
            self._versions.append(version)
            if self.max_versions > 0 and len(self._versions) > self.max_versions:
                self._versions = self._versions[-self.max_versions :]
            return version

    def get_version(self, version_number: int) -> Optional[ContextVersion]:
        with self._mu:
            return next(
                (v for v in self._versions if v.version_number == version_number),
                None,
            )

    def build_bridge_summary(
        self,
        from_mode: ContextSourceMode,
        to_mode: ContextSourceMode,
        max_items: int = 10,
    ) -> ContextBridgeSummary:
        with self._mu:
            bridgeable = {
                ContextEntryType.CONVERSATION,
                ContextEntryType.AI_SUMMARY,
                ContextEntryType.DEEP_ENGINE_RESULT,
                ContextEntryType.FILE_CHANGE,
            }
            recent = [entry for entry in self._entries if entry.entry_type in bridgeable][
                -max_items:
            ]
            lines: list[str] = []
            files: list[str] = []
            for entry in recent:
                if entry.entry_type == ContextEntryType.CONVERSATION:
                    lines.append(
                        f"{entry.metadata.get('role', 'unknown')}: {entry.content[:300]}"
                    )
                elif entry.entry_type == ContextEntryType.DEEP_ENGINE_RESULT:
                    for task in entry.metadata.get("tasks", []):
                        if task.get("status") == "completed" and task.get("result"):
                            lines.append(
                                f"[completed task] {task.get('title', '')}: "
                                f"{task['result'][:150]}"
                            )
                elif entry.entry_type == ContextEntryType.FILE_CHANGE:
                    files.append(entry.content)
            bridge = ContextBridgeSummary(
                from_mode=from_mode,
                to_mode=to_mode,
                summary_text="\n".join(lines[-8:]),
                files_modified=files,
            )
            self._last_bridge = bridge
            return bridge

    def consume_bridge_summary(self) -> Optional[ContextBridgeSummary]:
        with self._mu:
            bridge = self._last_bridge
            self._last_bridge = None
            return bridge


class ProjectContextManager:
    """The sole in-memory owner of chat-scoped project context."""

    def __init__(self, max_entries: int = 200, max_versions: int = 50) -> None:
        self._contexts: dict[tuple[str, str], UnifiedContext] = {}
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._max_entries = max_entries
        self._max_versions = max_versions

    @property
    def store(self) -> ProjectContextManager:
        """Compatibility facade for callers that previously used a nested store."""
        return self

    @staticmethod
    def _key(project_id: str, chat_id: str) -> tuple[str, str]:
        return chat_id, project_id

    def get_or_create(self, project_id: str, *, chat_id: str = "") -> UnifiedContext:
        key = self._key(project_id, chat_id)
        with self._lock:
            context = self._contexts.get(key)
            if context is None:
                context = UnifiedContext(
                    project_id,
                    max_entries=self._max_entries,
                    max_versions=self._max_versions,
                )
                self._contexts[key] = context
                logger.info("[ContextStore] created context for chat/project %r", key)
            return context

    def get(self, project_id: str, *, chat_id: str = "") -> Optional[UnifiedContext]:
        with self._lock:
            return self._contexts.get(self._key(project_id, chat_id))

    @staticmethod
    def _source(value: object, default: ContextSourceMode) -> ContextSourceMode:
        return value if isinstance(value, ContextSourceMode) else ContextSourceMode(str(value or default.value))

    def update_context(
        self,
        project_id: str,
        *,
        conversation: Optional[dict] = None,
        session_snapshot: Optional[dict] = None,
        mode_transition: Optional[dict] = None,
        deep_result: Optional[dict] = None,
        chat_id: str = "",
    ) -> int:
        """Append the production context payloads and return the added count."""
        if not project_id or not project_id.strip():
            return 0
        context = self.get_or_create(project_id, chat_id=chat_id)
        added = 0
        if conversation:
            context.add_conversation(
                role=conversation["role"],
                content=conversation["content"],
                source_mode=self._source(
                    conversation.get("source_mode"), ContextSourceMode.SMART
                ),
                message_id=conversation.get("message_id"),
            )
            added += 1
        if session_snapshot:
            context.add_session_snapshot(
                session_snapshot["data"],
                self._source(
                    session_snapshot.get("source_mode"), ContextSourceMode.SMART
                ),
            )
            added += 1
        if mode_transition:
            context.add_mode_transition(
                self._source(mode_transition.get("from_mode"), ContextSourceMode.SMART),
                self._source(mode_transition.get("to_mode"), ContextSourceMode.SMART),
                mode_transition.get("reason", ""),
            )
            added += 1
        if deep_result:
            context.add_deep_engine_result(deep_result["data"])
            added += 1
        return added
