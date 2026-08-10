"""SyncSession Protocol definition."""

from __future__ import annotations

import threading
from typing import Callable, Optional, Protocol

from ..acp.models import ACPEvent, PromptResult
from ..utils.retry import RetryPolicy, prompt_with_retry


class _PromptRetryMixin:
    """Share bounded retry plumbing across concrete sessions and wrappers."""

    _cancel_event: threading.Event

    def send_prompt_with_retry(
        self,
        text: str,
        on_event: Optional[Callable[[ACPEvent], None]] = None,
        timeout: Optional[int] = None,
        retry_policy: Optional[RetryPolicy] = None,
        before_retry: Optional[Callable[[int, Exception], None]] = None,
        total_timeout: Optional[float] = None,
    ) -> PromptResult:
        return prompt_with_retry(
            lambda: self.send_prompt(text, on_event=on_event, timeout=timeout),
            self._cancel_event,
            retry_policy=retry_policy,
            before_retry=before_retry,
            total_timeout=total_timeout,
        )


class _SessionWrapper(_PromptRetryMixin):
    """Transparent wrapper with one cancellation and attribute contract."""

    def __init__(self, inner: "SyncSession", cancel_event: threading.Event | None = None):
        self._inner = inner
        self._cancel_event = cancel_event or threading.Event()

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @property
    def session_id(self) -> str:
        return self._inner.session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._inner.session_id = value

    @property
    def last_active(self) -> float:
        return self._inner.last_active

    @last_active.setter
    def last_active(self, value: float) -> None:
        self._inner.last_active = value

    def cancel(self, wait: bool = False, timeout: float = 2.0) -> bool | None:
        self._cancel_event.set()
        try:
            return self._inner.cancel(wait=wait, timeout=timeout)
        except TypeError:
            return self._inner.cancel()


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
