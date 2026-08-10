"""Narrow injected ports for employee data production composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import DataKind


@dataclass(frozen=True)
class AuthenticatedExecutionTerminal:
    """Trusted terminal outcome from orchestration, not ACP output."""

    attempt_id: str
    status: str
    request_text: str
    result_text: str
    error_detail: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_usage: tuple = ()
    attachments: tuple = ()


@dataclass(frozen=True)
class PublishEmployeeDocumentCommand:
    """Trusted document publish from orchestration."""

    agent_id: str
    tenant_key: str
    owner_principal_id: str
    kind: DataKind
    source_id: str
    content: bytes
    content_type: str
    chat_id: str = ""
    thread_root_id: str = ""
    idempotency_key: str = ""


@dataclass(frozen=True)
class HistoryQuerySpec:
    """Query parameters for history range reads."""

    start_day: str
    end_day: str
    page_size: int = 50
    cursor: tuple[str, int, str] | None = None


@dataclass(frozen=True)
class MemoryQuerySpec:
    """Query parameters for memory reads."""

    agent_id: str
    chat_id: str = ""
    thread_root_id: str = ""
    full_l1: bool = False


class EmployeeDataSink(Protocol):
    """Write port for terminal outcomes and document publishes."""

    def record_terminal(self, terminal: AuthenticatedExecutionTerminal) -> None: ...

    def publish_document(self, command: PublishEmployeeDocumentCommand) -> None: ...
