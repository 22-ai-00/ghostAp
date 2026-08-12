"""Employee-owned delivery coordinator for Durable Outbox snapshots."""

from __future__ import annotations

import logging
import threading
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
from .service import EmployeeOutboxService

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


class EmployeeCardChannels(Protocol):
    def send(
        self,
        agent_id: str,
        *,
        generation: int,
        target: str,
        message: Any,
        options: Any = None,
    ) -> Any: ...

    def update_card(
        self,
        agent_id: str,
        *,
        generation: int,
        message_id: str,
        card: dict[str, Any],
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
    ) -> EmployeeOutboxDrainResult:
        """Attempt a bounded rotating batch, isolating failures by Outbox ID.

        The cursor advances when a record is selected, including when its
        authority or transport is unavailable.  A permanently broken oldest
        record therefore cannot starve later employees, while the ring still
        revisits that record on a later call.
        """

        if (
            type(max_items) is not int
            or max_items < 1
            or max_items > _MAX_PENDING_DELIVERY_BATCH
        ):
            raise ValueError(
                f"max_items must be between 1 and {_MAX_PENDING_DELIVERY_BATCH}"
            )
        with self._delivery_lock:
            return self._deliver_pending_locked(max_items=max_items)

    def _deliver_pending_locked(
        self,
        *,
        max_items: int,
    ) -> EmployeeOutboxDrainResult:
        pending = self._outbox.list_pending_delivery_records()
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
            outbox_id = self._pending_outbox_ids.popleft()
            attempted.append(outbox_id)
            try:
                self.deliver(outbox_id)
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
    ) -> EmployeeOutboxBinding | None:
        with self._delivery_lock:
            return self._deliver_locked(outbox_id, snapshot_version)

    def _deliver_locked(
        self,
        outbox_id: str,
        snapshot_version: int | None,
    ) -> EmployeeOutboxBinding | None:
        record = self._outbox.get_record(outbox_id)
        version = record.latest_version if snapshot_version is None else snapshot_version
        if record.binding is not None and record.binding.bound_snapshot_version >= version:
            return record.binding
        effect = self._outbox.prepare_delivery(outbox_id, version)
        if effect.state is DeliveryEffectState.COMMITTED:
            return self._outbox.get_record(outbox_id).binding
        if effect.state is DeliveryEffectState.PREPARED:
            effect = self._outbox.mark_effect_executing(effect.effect_id)
        if effect.state is not DeliveryEffectState.EXECUTING:
            raise RuntimeError("Outbox delivery effect is not executable")
        version = effect.snapshot_version
        snapshot = self._outbox.get_snapshot(outbox_id, version)

        record = self._outbox.get_record(outbox_id)
        authority = self._authority_resolver(record)
        if not isinstance(authority, EmployeeDeliveryAuthority):
            raise RuntimeError("employee delivery authority is unavailable")
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
                receipt = self._channels.send(
                    record.agent_id,
                    generation=authority.generation,
                    target=snapshot.chat_id,
                    message={"card": snapshot.to_dict()["card_json"]},
                    options=options,
                )
            except (EmployeeChannelOutboundError, ConnectionError, TimeoutError) as exc:
                raise EmployeeOutboxItemDeliveryError(
                    "employee delivery transport is unavailable"
                ) from exc
        else:
            try:
                receipt = self._channels.update_card(
                    record.agent_id,
                    generation=authority.generation,
                    message_id=record.binding.message_id,
                    card=snapshot.to_dict()["card_json"],
                )
            except (EmployeeChannelOutboundError, ConnectionError, TimeoutError) as exc:
                raise EmployeeOutboxItemDeliveryError(
                    "employee delivery transport is unavailable"
                ) from exc
        self._validate_receipt(receipt, authority, record.binding)
        return self._outbox.commit_delivery(
            effect.effect_id,
            app_id=receipt.app_id,
            generation=receipt.generation,
            connection_id=receipt.connection_id,
            message_id=receipt.message_id,
        )

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
            raise EmployeeOutboxItemDeliveryError(
                "employee delivery receipt does not match authority"
            )


__all__ = [
    "EmployeeDeliveryAuthority",
    "EmployeeOutboxDrainResult",
    "EmployeeOutboxDeliveryCoordinator",
    "EmployeeOutboxItemDeliveryError",
]
