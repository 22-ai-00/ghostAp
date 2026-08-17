"""Tamper-evident audit of main-Bot outbound message attempts."""

from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..journal.anchor import AnchorProvider, FileAnchor
from ..journal.frame import JournalEvent
from ..journal.writer import CommitState, JournalWriter

_AGGREGATE_ID = "main-bot-send-audit"
_EVENT_TYPE = "main_bot.send_attempted"
_OPERATIONS = frozenset({"create", "reply", "patch"})


class MainBotSendAuditLog:
    """Record every logical outbound mutation before network dispatch."""

    def __init__(self, writer: JournalWriter, *, external_audit: Any = None) -> None:
        if not isinstance(writer, JournalWriter):
            raise TypeError("writer must be JournalWriter")
        self.writer = writer
        self.external_audit = external_audit
        self._lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._complete = True

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        anchor_path: str | Path,
        hmac_key: bytes,
        writer_epoch: int | None = None,
        anchor: AnchorProvider | None = None,
        external_audit: Any = None,
    ) -> MainBotSendAuditLog:
        return cls(
            JournalWriter.open_at_verified_tail(
                Path(directory).expanduser(),
                anchor=anchor if anchor is not None else FileAnchor(anchor_path),
                hmac_key=hmac_key,
                writer_epoch=writer_epoch if writer_epoch is not None else time.time_ns(),
                aggregate_id=_AGGREGATE_ID,
            ),
            external_audit=external_audit,
        )

    def record_attempt(
        self,
        tenant_key: str,
        operation: str,
        target: str,
        *,
        attempted_at: float | None = None,
    ) -> None:
        if not isinstance(tenant_key, str) or len(tenant_key) > 512:
            raise ValueError("invalid tenant key")
        if operation not in _OPERATIONS:
            raise ValueError("invalid main Bot operation")
        if not isinstance(target, str) or not target or len(target) > 2048:
            raise ValueError("invalid main Bot target")
        if attempted_at is not None and (
            isinstance(attempted_at, bool)
            or not isinstance(attempted_at, (int, float))
            or not math.isfinite(float(attempted_at))
            or attempted_at <= 0
        ):
            raise ValueError("invalid main Bot audit timestamp")
        attempt_id = uuid.uuid4().hex
        tenant_hash = (
            hashlib.sha256(tenant_key.encode("utf-8")).hexdigest()
            if tenant_key
            else ""
        )
        target_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()
        with self._lock:
            try:
                timestamp = time.time() if attempted_at is None else attempted_at
                event = JournalEvent(
                    event_type=_EVENT_TYPE,
                    aggregate_id=_AGGREGATE_ID,
                    payload={
                        "attempt_id": attempt_id,
                        "tenant_hash": tenant_hash,
                        "operation": operation,
                        "target_hash": target_hash,
                        "attempted_at": float(timestamp),
                    },
                    timestamp=float(timestamp),
                )
                if self.external_audit is not None:
                    self.external_audit.record_main_bot_send_attempt(
                        attempt_id=attempt_id,
                        tenant_hash=tenant_hash,
                        operation=operation,
                        target_hash=target_hash,
                        attempted_at=float(timestamp),
                    )
                expected = self.writer.get_aggregate_versions((_AGGREGATE_ID,))
                result = self.writer.commit((event,), expected)
                if result.state is not CommitState.ANCHORED:
                    raise RuntimeError("main Bot send audit was not anchored")
            except BaseException:
                self._complete = False
                raise

    def mark_incomplete(self, _error: Exception | None = None) -> None:
        with self._lock:
            self._complete = False

    @property
    def activation_fence_ready(self) -> bool:
        """Whether activation can be fenced locally and by any external audit."""

        with self._lock:
            if not self._complete:
                return False
            external = self.external_audit
            if external is None:
                return True
            advertised = getattr(
                external,
                "main_bot_activation_fence_ready",
                None,
            )
            return (
                advertised is True
                and callable(
                    getattr(external, "acquire_main_bot_activation_fence", None)
                )
                and callable(
                    getattr(external, "release_main_bot_activation_fence", None)
                )
            )

    @contextmanager
    def activation_fence(
        self,
        tenant_key: str,
        target_hashes: tuple[str, ...],
    ) -> Iterator[None]:
        """Exclude outbound admission across the final audit/activation commit."""

        targets = self._validate_activation_fence_coordinates(
            tenant_key,
            target_hashes,
        )
        external_fence_id = ""
        self._lock.acquire()
        try:
            if not self._complete:
                raise RuntimeError("main Bot send audit is incomplete")
            external = self.external_audit
            if external is not None:
                if not self.activation_fence_ready:
                    self._complete = False
                    raise RuntimeError(
                        "external main Bot activation fence is unavailable"
                    )
                try:
                    external_fence_id = external.acquire_main_bot_activation_fence(
                        tenant_key,
                        targets,
                    )
                except BaseException:
                    self._complete = False
                    raise
                if not isinstance(external_fence_id, str) or not external_fence_id:
                    self._complete = False
                    raise RuntimeError("external main Bot activation fence is invalid")
            try:
                yield
            finally:
                if external is not None and external_fence_id:
                    try:
                        external.release_main_bot_activation_fence(
                            tenant_key,
                            targets,
                            fence_id=external_fence_id,
                        )
                    except BaseException:
                        self._complete = False
                        raise
        finally:
            self._lock.release()

    @staticmethod
    def _validate_activation_fence_coordinates(
        tenant_key: str,
        target_hashes: tuple[str, ...],
    ) -> tuple[str, ...]:
        if not isinstance(tenant_key, str) or not tenant_key or len(tenant_key) > 512:
            raise ValueError("invalid activation fence tenant")
        if (
            not isinstance(target_hashes, tuple)
            or not target_hashes
            or len(target_hashes) > 64
            or target_hashes != tuple(sorted(set(target_hashes)))
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in target_hashes
            )
        ):
            raise ValueError("invalid activation fence target set")
        return target_hashes

    def count_attempts(self, tenant_key: str, start: float, end: float) -> int:
        return self._count_attempts(tenant_key, start, end, target_hash=None)

    def count_target_attempts(
        self,
        tenant_key: str,
        target_hash: str,
        start: float,
        end: float,
    ) -> int:
        """Count main-Bot mutations aimed at one canonical ingress target."""

        if (
            not isinstance(target_hash, str)
            or len(target_hash) != 64
            or any(character not in "0123456789abcdef" for character in target_hash)
        ):
            raise ValueError("invalid target hash")
        return self._count_attempts(
            tenant_key,
            start,
            end,
            target_hash=target_hash,
        )

    def _count_attempts(
        self,
        tenant_key: str,
        start: float,
        end: float,
        *,
        target_hash: str | None,
    ) -> int:
        if not isinstance(tenant_key, str) or not tenant_key:
            raise ValueError("tenant key is required")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (start, end)
        ) or start > end:
            raise ValueError("invalid main Bot audit window")
        with self._lock:
            if not self._complete:
                raise RuntimeError("main Bot send audit is incomplete")
            local_count = 0
            tenant_hash = hashlib.sha256(tenant_key.encode("utf-8")).hexdigest()
            try:
                for frame in self.writer.replay():
                    for event in frame.events:
                        if event.event_type != _EVENT_TYPE:
                            continue
                        payload = event.payload
                        attempted_at = payload.get("attempted_at")
                        event_tenant = payload.get("tenant_hash")
                        event_target = payload.get("target_hash")
                        if (
                            isinstance(attempted_at, (int, float))
                            and not isinstance(attempted_at, bool)
                            and float(start) <= float(attempted_at) <= float(end)
                            and event_tenant in {"", tenant_hash}
                            and (target_hash is None or event_target == target_hash)
                        ):
                            local_count += 1
            except BaseException:
                self._complete = False
                raise
            if self.external_audit is None:
                return local_count
            if target_hash is None:
                external_count = self.external_audit.count_main_bot_send_attempts(
                    tenant_key,
                    float(start),
                    float(end),
                )
            else:
                external_count = self.external_audit.count_main_bot_target_send_attempts(
                    tenant_key,
                    target_hash,
                    float(start),
                    float(end),
                )
            if (
                isinstance(external_count, bool)
                or not isinstance(external_count, int)
                or external_count < local_count
            ):
                self._complete = False
                raise RuntimeError("external main Bot audit is behind local audit")
            return external_count

    def close(self) -> None:
        self.writer.close()


__all__ = ["MainBotSendAuditLog"]
