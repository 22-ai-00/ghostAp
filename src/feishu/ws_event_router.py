"""Feishu WebSocket event routing helpers."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class WSErrorAction(str, Enum):
    USE_COMPAT_FALLBACK = "use_compat_fallback"
    REPLY_INTERNAL_ERROR = "reply_internal_error"
    LOG_AND_CONTINUE = "log_and_continue"
    PROPAGATE = "propagate"


@dataclass(frozen=True)
class WSErrorClassification:
    action: WSErrorAction
    phase: str
    user_reachable: bool


def classify_ws_error(error: Exception, *, phase: str) -> WSErrorClassification:
    if phase == "import_guard":
        return WSErrorClassification(WSErrorAction.USE_COMPAT_FALLBACK, phase, False)
    if phase == "dispatch":
        return WSErrorClassification(WSErrorAction.REPLY_INTERNAL_ERROR, phase, True)
    if phase in {"cleanup", "dedup", "best_effort_notify"}:
        return WSErrorClassification(WSErrorAction.LOG_AND_CONTINUE, phase, False)
    return WSErrorClassification(WSErrorAction.PROPAGATE, phase, False)


class DuplicateMessageCache(Protocol):
    def is_duplicate(self, message_id: str) -> bool: ...

    def discard(self, message_id: str) -> None: ...

    def mark_seen(self, message_id: str) -> None: ...


class MessageIngressGuard:
    """Own message ingress expiry and duplicate checks for WS events."""

    def __init__(
        self,
        *,
        message_cache: DuplicateMessageCache,
        message_expire_seconds: int,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._message_cache = message_cache
        self._message_expire_seconds = message_expire_seconds
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._owners: dict[str, str] = {}

    def is_message_expired(self, create_time: int) -> bool:
        if not create_time:
            return False
        message_age_ms = self._clock_ms() - create_time
        return message_age_ms > self._message_expire_seconds * 1000

    def is_duplicate_message(self, message_id: str) -> bool:
        with self._lock:
            if message_id in self._owners:
                return True
            return self._message_cache.is_duplicate(message_id)

    def reserve(self, message_id: str) -> str | None:
        """Atomically reserve one message until its admitted task terminates."""

        with self._lock:
            if message_id in self._owners:
                return None
            if self._message_cache.is_duplicate(message_id):
                return None
            owner = secrets.token_urlsafe(24)
            self._owners[message_id] = owner
            return owner

    def owns(self, message_id: str, owner: str) -> bool:
        """Return whether ``owner`` still controls the in-flight reservation."""

        with self._lock:
            return self._owners.get(message_id) == owner

    def commit(self, message_id: str, owner: str) -> bool:
        """CAS an owned reservation into the bounded completed-message cache."""

        with self._lock:
            if self._owners.get(message_id) != owner:
                return False
            self._message_cache.mark_seen(message_id)
            del self._owners[message_id]
            return True

    def release(self, message_id: str, owner: str) -> bool:
        """CAS-release an owned reservation so a later delivery may retry."""

        with self._lock:
            if self._owners.get(message_id) != owner:
                return False
            self._message_cache.discard(message_id)
            del self._owners[message_id]
            return True
