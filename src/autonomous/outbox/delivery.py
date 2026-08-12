"""Employee-owned delivery coordinator for Durable Outbox snapshots."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..supervisor.channel_models import EmployeeChannelOutboundError
from .models import (
    DeliveryEffectState,
    EmployeeOutboxBinding,
    employee_outbox_uuid,
)
from .projection import OutboxRecord
from .service import EmployeeOutboxService, OutboxDeadlineExceededError

logger = logging.getLogger(__name__)

_DEFAULT_PENDING_DELIVERY_BATCH = 16
_MAX_PENDING_DELIVERY_BATCH = 64


@dataclass(frozen=True, slots=True)
class EmployeeDeliveryAuthority:
    app_id: str
    generation: int
    connection_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.app_id, str) or not self.app_id:
            raise ValueError("delivery authority app_id is required")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("delivery authority generation is invalid")
        if not isinstance(self.connection_id, str) or not self.connection_id:
            raise ValueError("delivery authority connection_id is required")


@dataclass(frozen=True, slots=True)
class EmployeeOutboxDrainResult:
    """Bounded delivery scan outcome without exposing transport error details."""

    pending_count: int
    attempted_outbox_ids: tuple[str, ...]
    delivered_outbox_ids: tuple[str, ...]
    failed_outbox_ids: tuple[str, ...]

    @property
    def made_progress(self) -> bool:
        return bool(self.delivered_outbox_ids)

    @property
    def has_pending(self) -> bool:
        return self.pending_count > 0


class EmployeeOutboxItemDeliveryError(RuntimeError):
    """One record hit an external delivery failure safe to isolate and retry."""


class EmployeeOutboxDrainDeadlineExceeded(TimeoutError):
    """The caller's absolute monotonic delivery deadline was exhausted."""


class EmployeeOutboxReceiptIntegrityError(RuntimeError):
    """A successful Channel receipt contradicted the frozen delivery authority."""


class EmployeeCardChannels(Protocol):
    def send(
        self,
        agent_id: str,
        *,
        generation: int,
        target: str,
        message: Any,
        options: Any = None,
        deadline: float | None = None,
    ) -> Any: ...

    def update_card(
        self,
        agent_id: str,
        *,
        generation: int,
        message_id: str,
        card: dict[str, Any],
        deadline: float | None = None,
    ) -> Any: ...


class EmployeeOutboxDeliveryCoordinator:
    """Anchor each delivery effect before invoking the employee child."""

    def __init__(
        self,
        *,
        outbox: EmployeeOutboxService,
        channels: EmployeeCardChannels,
        authority_resolver: Callable[[OutboxRecord], EmployeeDeliveryAuthority],
    ) -> None:
        if not isinstance(outbox, EmployeeOutboxService):
            raise TypeError("outbox must be EmployeeOutboxService")
        if not callable(authority_resolver):
            raise TypeError("authority_resolver must be callable")
        self._outbox = outbox
        self._channels = channels
        self._authority_resolver = authority_resolver
        self._pending_outbox_ids: deque[str] = deque()
        self._delivery_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock

    def deliver_pending(
        self,
        *,
        max_items: int = _DEFAULT_PENDING_DELIVERY_BATCH,
        deadline: float | None = None,
    ) -> EmployeeOutboxDrainResult:
        """Attempt a bounded rotating batch, isolating failures by Outbox ID.

        The cursor advances when a record is selected, including when its
        authority or transport is unavailable.  A permanently broken oldest
        record therefore cannot starve later employees, while the ring still
        revisits that record on a later call.
        """

        if type(max_items) is not int or max_items < 1 or max_items > _MAX_PENDING_DELIVERY_BATCH:
            raise ValueError(f"max_items must be between 1 and {_MAX_PENDING_DELIVERY_BATCH}")
        self._validate_deadline(deadline)
        try:
            with self._deadline_lock(deadline):
                return self._deliver_pending_locked(
                    max_items=max_items,
                    deadline=deadline,
                )
        except OutboxDeadlineExceededError as exc:
            raise EmployeeOutboxDrainDeadlineExceeded(str(exc)) from exc

    def _deliver_pending_locked(
        self,
        *,
        max_items: int,
        deadline: float | None,
    ) -> EmployeeOutboxDrainResult:
        self._raise_if_deadline_expired(deadline)
        pending = self._call_outbox(
            self._outbox.list_pending_delivery_records,
            deadline=deadline,
        )
        self._raise_if_deadline_expired(deadline)
        if not pending:
            self._pending_outbox_ids.clear()
            return EmployeeOutboxDrainResult(
                pending_count=0,
                attempted_outbox_ids=(),
                delivered_outbox_ids=(),
                failed_outbox_ids=(),
            )

        live_ids = {record.outbox_id for record in pending}
        queued_ids: set[str] = set()
        retained: deque[str] = deque()
        for outbox_id in self._pending_outbox_ids:
            if outbox_id in live_ids and outbox_id not in queued_ids:
                retained.append(outbox_id)
                queued_ids.add(outbox_id)
        for record in pending:
            if record.outbox_id not in queued_ids:
                retained.append(record.outbox_id)
                queued_ids.add(record.outbox_id)
        self._pending_outbox_ids = retained

        attempt_limit = min(max_items, len(self._pending_outbox_ids))
        attempted: list[str] = []
        delivered: list[str] = []
        failed: list[str] = []
        for _attempt in range(attempt_limit):
            self._raise_if_deadline_expired(deadline)
            outbox_id = self._pending_outbox_ids.popleft()
            attempted.append(outbox_id)
            try:
                self.deliver(outbox_id, deadline=deadline)
            except EmployeeOutboxItemDeliveryError as exc:
                failed.append(outbox_id)
                self._pending_outbox_ids.append(outbox_id)
                logger.warning(
                    "employee Outbox delivery deferred: outbox_id=%s error=%s",
                    outbox_id,
                    type(exc).__name__,
                )
                continue
            except BaseException:
                self._pending_outbox_ids.appendleft(outbox_id)
                raise
            delivered.append(outbox_id)
        return EmployeeOutboxDrainResult(
            pending_count=len(pending),
            attempted_outbox_ids=tuple(attempted),
            delivered_outbox_ids=tuple(delivered),
            failed_outbox_ids=tuple(failed),
        )

    def deliver(
        self,
        outbox_id: str,
        snapshot_version: int | None = None,
        *,
        deadline: float | None = None,
    ) -> EmployeeOutboxBinding | None:
        self._validate_deadline(deadline)
        try:
            with self._deadline_lock(deadline):
                return self._deliver_locked(
                    outbox_id,
                    snapshot_version,
                    deadline,
                )
        except OutboxDeadlineExceededError as exc:
            raise EmployeeOutboxDrainDeadlineExceeded(str(exc)) from exc

    def _deliver_locked(
        self,
        outbox_id: str,
        snapshot_version: int | None,
        deadline: float | None,
    ) -> EmployeeOutboxBinding | None:
        self._raise_if_deadline_expired(deadline)
        record = self._call_outbox(
            self._outbox.get_record,
            outbox_id,
            deadline=deadline,
        )
        version = record.latest_version if snapshot_version is None else snapshot_version
        if record.binding is not None and record.binding.bound_snapshot_version >= version:
            return record.binding
        effect = self._call_outbox(
            self._outbox.prepare_delivery,
            outbox_id,
            version,
            deadline=deadline,
        )
        if effect.state is DeliveryEffectState.COMMITTED:
            return self._call_outbox(
                self._outbox.get_record,
                outbox_id,
                deadline=deadline,
            ).binding
        if effect.state is DeliveryEffectState.PREPARED:
            effect = self._call_outbox(
                self._outbox.mark_effect_executing,
                effect.effect_id,
                deadline=deadline,
            )
        if effect.state is not DeliveryEffectState.EXECUTING:
            raise RuntimeError("Outbox delivery effect is not executable")
        version = effect.snapshot_version
        snapshot = self._call_outbox(
            self._outbox.get_snapshot,
            outbox_id,
            version,
            deadline=deadline,
        )

        record = self._call_outbox(
            self._outbox.get_record,
            outbox_id,
            deadline=deadline,
        )
        authority = self._authority_resolver(record)
        if not isinstance(authority, EmployeeDeliveryAuthority):
            raise RuntimeError("employee delivery authority is unavailable")
        self._raise_if_deadline_expired(deadline)
        if record.binding is None:
            options: dict[str, Any] = {"uuid": employee_outbox_uuid(outbox_id)}
            if snapshot.thread_root_message_id:
                options.update(
                    {
                        "reply_to": snapshot.thread_root_message_id,
                        "reply_in_thread": True,
                    }
                )
            try:
                send_kwargs: dict[str, Any] = {
                    "generation": authority.generation,
                    "target": snapshot.chat_id,
                    "message": {"card": snapshot.to_dict()["card_json"]},
                    "options": options,
                }
                if deadline is not None:
                    send_kwargs["deadline"] = deadline
                receipt = self._channels.send(record.agent_id, **send_kwargs)
            except (EmployeeChannelOutboundError, ConnectionError, TimeoutError) as exc:
                raise EmployeeOutboxItemDeliveryError("employee delivery transport is unavailable") from exc
        else:
            try:
                update_kwargs: dict[str, Any] = {
                    "generation": authority.generation,
                    "message_id": record.binding.message_id,
                    "card": snapshot.to_dict()["card_json"],
                }
                if deadline is not None:
                    update_kwargs["deadline"] = deadline
                receipt = self._channels.update_card(
                    record.agent_id,
                    **update_kwargs,
                )
            except (EmployeeChannelOutboundError, ConnectionError, TimeoutError) as exc:
                raise EmployeeOutboxItemDeliveryError("employee delivery transport is unavailable") from exc
        self._validate_receipt(receipt, authority, record.binding)
        return self._call_outbox(
            self._outbox.commit_delivery,
            effect.effect_id,
            app_id=receipt.app_id,
            generation=receipt.generation,
            connection_id=receipt.connection_id,
            message_id=receipt.message_id,
            deadline=deadline,
        )

    @staticmethod
    def _call_outbox(
        operation: Callable[..., Any],
        *args: object,
        deadline: float | None,
        **kwargs: object,
    ) -> Any:
        """Keep the pre-deadline call shape for narrow injected test doubles."""

        if deadline is not None:
            kwargs["deadline"] = deadline
        return operation(*args, **kwargs)

    def _deadline_lock(self, deadline: float | None):
        """Acquire the coordinator lock within the caller's shared deadline."""

        if deadline is None:
            return self._delivery_lock
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            raise EmployeeOutboxDrainDeadlineExceeded("employee Outbox delivery lock deadline exceeded")
        return _AcquiredDeliveryLock(
            self._delivery_lock,
            min(remaining, threading.TIMEOUT_MAX),
        )

    @staticmethod
    def _validate_deadline(deadline: float | None) -> None:
        if deadline is None:
            return
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline)):
            raise ValueError("employee Outbox delivery deadline is invalid")

    @staticmethod
    def _raise_if_deadline_expired(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= float(deadline):
            raise EmployeeOutboxDrainDeadlineExceeded("employee Outbox delivery deadline exceeded")

    @staticmethod
    def _validate_receipt(
        receipt: Any,
        authority: EmployeeDeliveryAuthority,
        current_binding: EmployeeOutboxBinding | None,
    ) -> None:
        valid = (
            getattr(receipt, "success", None) is True
            and getattr(receipt, "app_id", None) == authority.app_id
            and getattr(receipt, "generation", None) == authority.generation
            and getattr(receipt, "connection_id", None) == authority.connection_id
            and isinstance(getattr(receipt, "message_id", None), str)
            and bool(receipt.message_id)
            and (current_binding is None or receipt.message_id == current_binding.message_id)
        )
        if not valid:
            raise EmployeeOutboxReceiptIntegrityError("employee delivery receipt does not match authority")


class _AcquiredDeliveryLock:
    """Small context adapter for deadline-bounded RLock acquisition."""

    def __init__(self, lock: threading.RLock, timeout: float) -> None:
        self._lock = lock
        self._timeout = timeout

    def __enter__(self) -> None:
        if not self._lock.acquire(timeout=self._timeout):
            raise EmployeeOutboxDrainDeadlineExceeded("employee Outbox delivery lock deadline exceeded")

    def __exit__(self, *_args: object) -> None:
        self._lock.release()


__all__ = [
    "EmployeeDeliveryAuthority",
    "EmployeeOutboxDrainDeadlineExceeded",
    "EmployeeOutboxDrainResult",
    "EmployeeOutboxDeliveryCoordinator",
    "EmployeeOutboxItemDeliveryError",
    "EmployeeOutboxReceiptIntegrityError",
]
