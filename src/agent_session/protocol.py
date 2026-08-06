"""SyncSession Protocol definition."""

from __future__ import annotations

from typing import Callable, Optional, Protocol

from ..acp.models import ACPEvent, PromptResult
from ..utils.retry import RetryPolicy


class SyncSession(Protocol):
    """A minimal sync session interface used by handlers."""

    session_id: str
    created_at: float
    last_active: float
    message_count: int
    last_query: str
    is_resumed: bool

    def describe_agent(self) -> str: ...
    def start(self, startup_timeout: float = 60) -> str: ...
    def load_session(self, session_id: str, timeout: float) -> None: ...
    def load_local_history(self, session_id: Optional[str] = None, limit: int = 200) -> list[dict]: ...
    def send_prompt(
        self, text: str, on_event: Optional[Callable[[ACPEvent], None]] = None, timeout: Optional[int] = None
    ) -> PromptResult: ...
    def send_prompt_with_retry(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        before_retry: Optional[Callable[[int, Exception], None]] = None,
        total_timeout: Optional[float] = None,
    ) -> PromptResult: ...
    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool | None: ...
    def close(self) -> None: ...
    def to_snapshot(self) -> dict: ...
    def get_session_info(self) -> str: ...

    def is_server_running(self) -> bool: ...
    def is_server_healthy(self, healthcheck_timeout: float = 2.0) -> bool: ...
