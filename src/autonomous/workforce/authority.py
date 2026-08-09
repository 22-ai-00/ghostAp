"""Writer-authority fencing for the legacy Team employee registry."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class AuthorityMode(str, Enum):
    LEGACY_WRITE = "legacy_write"
    SHADOW_READ = "shadow_read"
    V5_WRITE = "v5_write"
    V5_ONLY = "v5_only"


@dataclass(frozen=True)
class AuthoritySnapshot:
    epoch: int
    mode: AuthorityMode
    cutover_sequence: int = 0


class StaleAuthorityEpoch(RuntimeError):
    """A legacy mutation no longer owns the employee writer authority."""


class LegacyMutationGuard:
    """Validate legacy writes against a live authority snapshot."""

    _WRITABLE_MODES = frozenset(
        {AuthorityMode.LEGACY_WRITE, AuthorityMode.SHADOW_READ}
    )

    def __init__(
        self,
        snapshot_provider: Callable[[], AuthoritySnapshot],
        *,
        expected_epoch: int,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._expected_epoch = expected_epoch
        self._serialization_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

    def assert_writable(
        self,
        operation: str,
        *,
        validated_epoch: int | None = None,
    ) -> int:
        with self._serialization_lock:
            return self._assert_writable(operation, validated_epoch)

    @contextmanager
    def write_lease(
        self,
        operation: str,
        *,
        validated_epoch: int | None = None,
    ) -> Iterator[int]:
        """Linearize validation and the complete legacy mutation."""

        with self._serialization_lock:
            yield self._assert_writable(operation, validated_epoch)

    def cutover(
        self,
        advance: Callable[[], AuthoritySnapshot],
        *,
        on_success: Callable[[], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> AuthoritySnapshot:
        """Wait for legacy writes, then advance authority under the same lock."""

        with self._serialization_lock:
            try:
                before = self._snapshot_provider()
                after = advance()
                observed = self._snapshot_provider()
                if after != observed:
                    raise RuntimeError(
                        "authority cutover callback did not publish snapshot"
                    )
                if after.epoch <= before.epoch:
                    raise ValueError("authority cutover must increase epoch")
                if on_success is not None:
                    on_success()
                return after
            finally:
                if on_finish is not None:
                    on_finish()

    def _assert_writable(
        self,
        operation: str,
        validated_epoch: int | None,
    ) -> int:
        snapshot = self._snapshot_provider()
        expected = (
            self._expected_epoch
            if validated_epoch is None
            else validated_epoch
        )
        if snapshot.epoch != expected or snapshot.mode not in self._WRITABLE_MODES:
            raise StaleAuthorityEpoch(
                f"legacy {operation} rejected: expected authority epoch "
                f"{expected}, got {snapshot.epoch} in {snapshot.mode.value}"
            )
        return snapshot.epoch
