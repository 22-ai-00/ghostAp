"""Outbox draining, retirement delivery recovery, and channel supervisor loops."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from src.utils.async_helpers import safe_wait_for

from ..acceptance.main_bot_warning_outbox import (
    MainBotWarningRetryableDeliveryError,
)
from ..context.group_memory import EmployeeGroupMemoryStore
from ..context.lark_source import LarkEmployeeMessageSourceFactory
from ..context.runtime import (
    RuntimeEmployeeGenerationAuthority,
    parse_requester_acl,
)
from ..context.service import AuthorizedGroupMemoryReader
from ..domain import EmployeeState
from ..ingress.service import (
    IngressBlobRetryableError,
    IngressConflictError,
)
from ..journal.frame import JournalEvent
from ..outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDrainDeadlineExceeded,
    EmployeeOutboxDrainResult,
    EmployeeOutboxItemDeliveryError,
)
from ..supervisor.employee_channels import (
    ChannelLaunchUnavailable,
)
from .external_mutation_gate import (
    ExternalMutationFenced,
    ExternalMutationKind,
    ExternalTaskTerminalError,
    external_task_name,
    is_external_task,
)
from .fire_state import FireEffectState, FirePhase
from .hire_state import (
    DurableHireState,
    HireEffectState,
)
from .notification_state import (
    HireNotificationPhase,
    hire_notification_message_uuid,
    rebuild_hire_notification_projection,
)
from .slash_commands import SlashCommandReconciler, VerifiedSlashState
from .slash_lark import LarkSlashCommandAPI

if TYPE_CHECKING:
    from .composition import (
        DurableHireState,
        EmployeeDeliveryAuthority,
        EmployeeMessageSourceFactory,
        EmployeeOutboxDrainResult,
        EmployeeRecoverySummary,
        ProjectedAgentRegistry,
        ProjectionState,
        VerifiedSlashState,
        _SlashReconciler,
        _ValidatedTargetedOwnership,
    )

logger = logging.getLogger(__name__)


def _composition_module():
    """Return the composition module for runtime-patchable symbols."""
    from . import composition as _comp_mod

    return _comp_mod


class EmployeeRuntimeRecoveryMixin:
    """Mixin providing outbox close drain, loop recovery, and channel monitoring."""

    def _request_employee_outbox_delivery(self) -> None:
        """Wake production reporting, with a narrow synchronous embed fallback."""

        reporting_thread = self._reporting_thread
        if reporting_thread is not None and reporting_thread.is_alive():
            self._notify_employee_reporting()
            return
        self._drain_employee_outbox_once()

    def _submit_employee_outbox_close_drain(
        self,
        *,
        deadline: float,
    ) -> concurrent.futures.Future[None]:
        """Start the one runtime-owned close drain without caller-thread I/O."""

        self._require_employee_outbox_close_time(deadline)
        with self._employee_outbox_close_drain_lock:
            existing = self._employee_outbox_close_drain_future
            if existing is not None:
                if not existing.done() or existing.cancelled():
                    return existing
                failure = existing.exception()
                if not isinstance(
                    failure,
                    EmployeeOutboxDrainDeadlineExceeded,
                ):
                    return existing
                # A terminal deadline fence is safe to resume on a later close:
                # delivery effects are anchored first, CREATE retries reuse the
                # stable UUID, and PATCH targets the existing binding.  Keep
                # every in-flight, cancelled, or differently failed future so
                # an unknown outcome can never be invoked a second time.
                self._employee_outbox_close_drain_future = None
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="employee-outbox-close-drain",
            )
            try:
                future = executor.submit(
                    self._drain_employee_outbox_until_idle,
                    deadline=deadline,
                )
            except Exception:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            self._employee_outbox_close_drain_executor = executor
            self._employee_outbox_close_drain_future = future

        def release_executor(_done: concurrent.futures.Future[None]) -> None:
            with self._employee_outbox_close_drain_lock:
                if self._employee_outbox_close_drain_executor is executor:
                    self._employee_outbox_close_drain_executor = None
            executor.shutdown(wait=False, cancel_futures=False)

        future.add_done_callback(release_executor)
        return future

    def _await_employee_outbox_close_drain(self, *, deadline: float) -> None:
        """Observe the unique drain only within this close caller's budget."""

        future = self._submit_employee_outbox_close_drain(deadline=deadline)
        remaining = max(0.0, deadline - _composition_module().time.monotonic())
        if remaining <= 0.0:
            raise EmployeeOutboxDrainDeadlineExceeded(
                "employee Outbox close deadline exceeded"
            )
        try:
            future.result(timeout=remaining)
        except EmployeeOutboxDrainDeadlineExceeded:
            raise
        except concurrent.futures.TimeoutError as exc:
            if future.done():
                # ``concurrent.futures.TimeoutError`` aliases built-in
                # ``TimeoutError``.  A terminal plain TimeoutError belongs to
                # the drain operation and may represent an unknown outcome;
                # preserve that failed future instead of classifying it as a
                # safe caller-wait deadline.
                raise
            raise EmployeeOutboxDrainDeadlineExceeded(
                "employee Outbox close drain is still pending"
            ) from exc

    def _drain_employee_outbox_until_idle(
        self,
        *,
        max_batches: int | None = None,
        deadline: float | None = None,
    ) -> None:
        """Drain complete fair rounds within one absolute close-time deadline."""

        if max_batches is None:
            max_batches = _composition_module()._EMPLOYEE_OUTBOX_CLOSE_MAX_BATCHES
        if type(max_batches) is not int or max_batches < 1:
            raise ValueError("employee Outbox close batch limit is invalid")
        if deadline is None:
            deadline = _composition_module().time.monotonic() + _composition_module()._EMPLOYEE_OUTBOX_CLOSE_DRAIN_SECONDS
        elif isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline)):
            raise ValueError("employee Outbox close deadline is invalid")
        deadline = float(deadline)

        batches = 0
        while batches < max_batches:
            self._require_employee_outbox_close_time(deadline)
            result = self._deliver_employee_outbox_batch(
                max_items=_composition_module()._EMPLOYEE_OUTBOX_DELIVERY_BATCH,
                deadline=deadline,
            )
            batches += 1
            if result is None or not result.attempted_outbox_ids:
                return
            round_pending = result.pending_count
            round_attempted = len(result.attempted_outbox_ids)
            round_delivered = len(result.delivered_outbox_ids)
            if round_pending < round_attempted:
                raise RuntimeError("employee Outbox drain result is inconsistent")

            while round_attempted < round_pending:
                if batches >= max_batches:
                    self._require_employee_outbox_close_time(deadline)
                    if not self._employee_outbox_has_pending(
                        deadline=deadline,
                    ):
                        return
                    raise RuntimeError("employee Outbox did not drain")
                self._require_employee_outbox_close_time(deadline)
                result = self._deliver_employee_outbox_batch(
                    max_items=min(
                        _composition_module()._EMPLOYEE_OUTBOX_DELIVERY_BATCH,
                        round_pending - round_attempted,
                    ),
                    deadline=deadline,
                )
                batches += 1
                if result is None or not result.attempted_outbox_ids:
                    raise RuntimeError("employee Outbox drain stopped mid-round")
                round_attempted += len(result.attempted_outbox_ids)
                round_delivered += len(result.delivered_outbox_ids)

            if round_delivered == 0:
                raise EmployeeOutboxItemDeliveryError("employee Outbox close round is unresolved")

        self._require_employee_outbox_close_time(deadline)
        if not self._employee_outbox_has_pending(deadline=deadline):
            return
        raise RuntimeError("employee Outbox did not drain")

    def _employee_outbox_has_pending(
        self,
        *,
        deadline: float | None = None,
    ) -> bool:
        """Read-only close probe after admission and reporting are stopped."""

        outbox = self._outbox
        pending_records = getattr(outbox, "list_pending_delivery_records", None)
        if not callable(pending_records):
            raise RuntimeError("employee Outbox pending probe is unavailable")
        if deadline is None:
            return bool(pending_records())
        return bool(pending_records(deadline=deadline))

    @staticmethod
    def _require_employee_outbox_close_time(deadline: float) -> None:
        if _composition_module().time.monotonic() >= deadline:
            raise EmployeeOutboxDrainDeadlineExceeded("employee Outbox close deadline exceeded")

    def _deliver_employee_outbox_batch(
        self,
        *,
        max_items: int,
        deadline: float | None = None,
    ) -> EmployeeOutboxDrainResult | None:
        """Serialize one coordinator batch while preserving its full outcome."""

        delivery_lock = self._employee_outbox_delivery_lock
        if deadline is None:
            with delivery_lock:
                delivery = self._outbox_delivery
                if self._outbox is None or delivery is None:
                    return None
                return delivery.deliver_pending(max_items=max_items)
        remaining = max(0.0, deadline - _composition_module().time.monotonic())
        if remaining <= 0.0 or not delivery_lock.acquire(timeout=remaining):
            raise EmployeeOutboxDrainDeadlineExceeded("employee Outbox delivery lock deadline exceeded")
        try:
            self._require_employee_outbox_close_time(deadline)
            delivery = self._outbox_delivery
            if self._outbox is None or delivery is None:
                return None
            return delivery.deliver_pending(
                max_items=max_items,
                deadline=deadline,
            )
        finally:
            delivery_lock.release()

    def _drain_employee_outbox_once(self) -> bool:
        # Control responses can request an eager drain while the independent
        # reporting worker is polling.  The coordinator owns the rotating fair
        # cursor; this runtime lock also supports narrow injected test doubles.
        result = self._deliver_employee_outbox_batch(
            max_items=_composition_module()._EMPLOYEE_OUTBOX_DELIVERY_BATCH,
        )
        if result is None:
            return False
        if result.failed_outbox_ids and not result.delivered_outbox_ids:
            # Normal reporting backs off after one all-failed batch.  Close uses
            # the full result to finish the initial fair round first.
            raise EmployeeOutboxItemDeliveryError("employee Outbox batch delivery deferred")
        return bool(result.attempted_outbox_ids)

    def _drain_main_bot_warning_outbox_once(self) -> bool:
        """Deliver one bounded durable main-Bot warning batch."""

        outbox = self._main_bot_warning_outbox
        transport = self._main_bot_warning_transport
        if outbox is None or transport is None:
            return False
        if not outbox.has_local_pending_records():
            return False
        result = outbox.recover_pending(
            transport,
            max_items=_composition_module()._EMPLOYEE_OUTBOX_DELIVERY_BATCH,
            # One absolute budget covers the whole batch.  A slow synchronous
            # SDK call remains owned by MainBotWarningOutbox's single-flight
            # future after expiry; the reporting worker can therefore service
            # Employee repair/outbox work without canceling or duplicating it.
            deadline=_composition_module().time.monotonic() + _composition_module()._MAIN_BOT_WARNING_REPORTING_DRAIN_SECONDS,
        )
        if result.failed_warning_ids and not (result.committed_warning_ids or result.action_required_warning_ids):
            raise MainBotWarningRetryableDeliveryError("main Bot warning delivery batch is deferred")
        return bool(result.attempted_warning_ids)

    def _drain_main_bot_warning_outbox_until_idle(
        self,
        *,
        max_batches: int | None = None,
        deadline: float | None = None,
    ) -> None:
        """Finish warning delivery or retain all resources for a later retry."""

        if max_batches is None:
            max_batches = _composition_module()._EMPLOYEE_OUTBOX_CLOSE_MAX_BATCHES
        if type(max_batches) is not int or max_batches < 1:
            raise ValueError("main Bot warning close batch limit is invalid")
        if deadline is None:
            deadline = _composition_module().time.monotonic() + _composition_module()._EMPLOYEE_OUTBOX_CLOSE_DRAIN_SECONDS
        elif isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(float(deadline)):
            raise ValueError("main Bot warning close deadline is invalid")
        deadline = float(deadline)
        outbox = self._main_bot_warning_outbox
        if outbox is None:
            return
        transport = self._main_bot_warning_transport
        initial_pending = outbox.pending_records(deadline=deadline)
        remaining_initial_ids = {record.warning_id for record in initial_pending}
        if not remaining_initial_ids:
            return
        pending = initial_pending

        for _batch in range(max_batches):
            self._require_employee_outbox_close_time(deadline)
            if not pending:
                return
            pending_ids = {record.warning_id for record in pending}
            remaining_initial_ids.intersection_update(pending_ids)
            if not remaining_initial_ids:
                break
            if transport is None:
                raise _composition_module().MainBotWarningPreparationError("main Bot warning transport is unavailable during close")
            result = outbox.recover_pending(
                transport,
                max_items=min(
                    _composition_module()._EMPLOYEE_OUTBOX_DELIVERY_BATCH,
                    len(pending),
                    len(remaining_initial_ids),
                ),
                deadline=deadline,
            )
            if not result.attempted_warning_ids:
                raise _composition_module().MainBotWarningPreparationError("main Bot warning close drain made no progress")
            remaining_initial_ids.difference_update(result.attempted_warning_ids)
            if not remaining_initial_ids:
                break
            pending = outbox.pending_records(deadline=deadline)

        self._require_employee_outbox_close_time(deadline)
        pending = outbox.pending_records(deadline=deadline)
        if not pending:
            return
        if remaining_initial_ids:
            raise RuntimeError("main Bot warning Outbox did not complete its fair close round")
        raise MainBotWarningRetryableDeliveryError("main Bot warning close delivery remains unresolved")

    def _resolve_outbox_delivery_authority(
        self,
        record: Any,
    ) -> EmployeeDeliveryAuthority:
        if self._service is None or self._channels is None:
            raise RuntimeError("employee delivery authority is unavailable")
        snapshot = getattr(
            self._service,
            "current_employee_transport_snapshot",
            None,
        )
        if not callable(snapshot):
            raise RuntimeError("employee delivery authority snapshot is unavailable")
        employees, principals, hire_states = snapshot()
        matching_employees = tuple(
            employee
            for employee in employees
            if employee.agent_id == record.agent_id and employee.tenant_key == record.tenant_key
        )
        if not matching_employees:
            raise EmployeeOutboxItemDeliveryError("employee delivery identity is unavailable")
        if len(matching_employees) != 1:
            raise RuntimeError("employee delivery identity is ambiguous")
        employee = matching_employees[0]
        matching_principals = tuple(
            principal
            for principal in principals
            if principal.bot_principal_id == employee.bot_principal_id
            and principal.tenant_key == record.tenant_key
            and principal.agent_id == record.agent_id
        )
        matching_hires = tuple(
            state
            for state in hire_states
            if state.tenant_key == record.tenant_key and state.agent_id == record.agent_id
        )
        if not matching_principals or not matching_hires:
            raise EmployeeOutboxItemDeliveryError("employee delivery identity is unavailable")
        if len(matching_principals) != 1 or len(matching_hires) != 1:
            raise RuntimeError("employee delivery identity is ambiguous")
        principal = matching_principals[0]
        hire = matching_hires[0]
        if hire.phase in {_composition_module().HirePhase.RETIRING, _composition_module().HirePhase.ACTION_REQUIRED}:
            if employee.state not in {
                EmployeeState.RETIRING,
                EmployeeState.ACTION_REQUIRED,
            }:
                raise EmployeeOutboxItemDeliveryError("employee retirement delivery identity is stale")
            self._require_retirement_delivery_lease(record, hire)
        elif hire.phase is not _composition_module().HirePhase.ACTIVE or employee.state is not EmployeeState.ACTIVE:
            raise EmployeeOutboxItemDeliveryError("employee delivery identity is not active")
        status = self._channels.status(record.agent_id)
        connection_id = getattr(status, "ready_metadata", {}).get(
            "connection_id",
            "",
        )
        if (
            status is None
            or not employee.bot_principal_id
            or hire.bot_principal_id != employee.bot_principal_id
            or not principal.credential_ref
            or principal.credential_ref != hire.credential_ref
            or getattr(status, "state", None) is not _composition_module().ChannelProcessState.READY
            or principal.app_id != hire.app_id
            or getattr(status, "tenant_key", None) != record.tenant_key
            or getattr(status, "agent_id", None) != record.agent_id
            or getattr(status, "bot_principal_id", None) != employee.bot_principal_id
            or getattr(status, "app_id", None) != hire.app_id
            or getattr(status, "identity", {}).get("app_id") != hire.app_id
            or getattr(status, "generation", None) != hire.channel_generation
            or hire.channel_identity_app_id != hire.app_id
            or hire.channel_connection_id != connection_id
            or not isinstance(connection_id, str)
            or not connection_id
        ):
            raise EmployeeOutboxItemDeliveryError("employee delivery Channel is not current")
        return EmployeeDeliveryAuthority(
            app_id=hire.app_id,
            generation=hire.channel_generation,
            connection_id=connection_id,
        )

    def _require_retirement_delivery_lease(
        self,
        record: Any,
        hire: DurableHireState,
    ) -> None:
        """Authorize only cutoff-owned response delivery before quiesce commits."""

        fire = self._fire
        if fire is None:
            raise EmployeeOutboxItemDeliveryError("employee retirement delivery lease is unavailable")
        matching_fires = tuple(
            state
            for state in fire.list_states()
            if state.tenant_key == record.tenant_key
            and state.agent_id == record.agent_id
            and state.phase is not FirePhase.SUPERSEDED
        )
        if not matching_fires:
            raise EmployeeOutboxItemDeliveryError("employee retirement delivery lease is unavailable")
        if len(matching_fires) != 1:
            raise RuntimeError("employee retirement delivery lease is ambiguous")
        fire_state = matching_fires[0]
        if (
            fire_state.phase not in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
            or fire_state.effect_state("execution_quiesce")
            not in {
                FireEffectState.EXECUTING,
                FireEffectState.ACTION_REQUIRED,
            }
            or fire_state.bot_principal_id != hire.bot_principal_id
            or fire_state.app_id != hire.app_id
            or fire_state.credential_ref != hire.credential_ref
            or not self._retirement_outbox_is_cutoff_owned(
                record,
                requested_sequence=fire_state.requested_sequence,
            )
        ):
            raise EmployeeOutboxItemDeliveryError("employee retirement delivery lease is unavailable")

    def _retirement_outbox_is_cutoff_owned(
        self,
        record: Any,
        *,
        requested_sequence: int,
    ) -> bool:
        """Bind a retiring Outbox item to work accepted before `/fire`."""

        if type(requested_sequence) is not int or requested_sequence <= 0:
            return False
        attempt_id = getattr(record, "attempt_id", "")
        latest = getattr(record, "latest", None)
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or latest is None
            or getattr(getattr(latest, "state", None), "terminal", False) is not True
        ):
            return False
        if attempt_id.startswith("control_"):
            router = self._router
            ingress = self._ingress
            acceptance_id = attempt_id.removeprefix("control_")
            if router is None or ingress is None or not acceptance_id:
                return False
            router.rebuild_projection()
            ingress.rebuild_projection()
            routed = router.record_snapshot(acceptance_id)
            ingress_record = ingress.record_snapshot(acceptance_id)
            metadata = getattr(ingress_record, "metadata", None)
            acceptance = getattr(ingress_record, "acceptance", None)
            disposition = getattr(ingress_record, "disposition", None)
            expected_thread_index = getattr(metadata, "thread_root_message_id", "") or getattr(
                metadata, "message_id", ""
            )
            if (
                routed is None
                or not isinstance(metadata, _composition_module().EmployeeIngressMetadata)
                or acceptance is None
                or disposition is None
                or disposition.state != "terminal"
                or routed.tenant_key != record.tenant_key
                or routed.agent_id != record.agent_id
                or routed.event_type != "im.message.receive_v1"
                or routed.accepted_sequence <= 0
                or routed.accepted_sequence != getattr(acceptance, "journal_sequence", None)
                or routed.accepted_sequence >= requested_sequence
                or not _composition_module()._outbox_coordinate_matches_ingress_index(
                    metadata.chat_id,
                    record.chat_id,
                    "oc_",
                )
                or not _composition_module()._outbox_coordinate_matches_ingress_index(
                    expected_thread_index,
                    record.thread_root_message_id,
                    "om_",
                )
                or (
                    metadata.tenant_key,
                    metadata.agent_id,
                    metadata.bot_principal_id,
                    metadata.app_id,
                    metadata.channel_generation,
                    metadata.connection_id,
                )
                != (
                    routed.tenant_key,
                    routed.agent_id,
                    routed.bot_principal_id,
                    routed.app_id,
                    routed.channel_generation,
                    routed.connection_id,
                )
            ):
                return False
            if routed.queued_sequence > 0:
                if (
                    routed.state != "terminal"
                    or disposition.reason_code != routed.reason_code
                    or getattr(routed, "authority", None) is None
                    or record.chat_id != getattr(routed.authority, "team_id", None)
                ):
                    return False
                # Reconciliation validates the encrypted payload and appends
                # this deterministic response before committing the matching
                # terminal disposition.  That disposition is the durable
                # proof after terminal payload GC; retirement delivery must
                # not require the plaintext blob to remain readable.
                return True
            else:
                if not disposition.reason_code.startswith(
                    (
                        "stop_",
                        "history_",
                        "memory_",
                        "status_",
                        "task_",
                    )
                ):
                    return False
                # Control handlers likewise append their deterministic Outbox
                # response before recording the allowlisted terminal reason.
                return True
        dispatch = self._dispatch
        if dispatch is None:
            return False
        attempt = dispatch.state.attempts.get(attempt_id)
        return bool(
            attempt is not None
            and attempt.binding.tenant_key == record.tenant_key
            and attempt.binding.agent_id == record.agent_id
            and attempt.binding.chat_id == record.chat_id
            and attempt.binding.thread_root_id == record.thread_root_message_id
            and attempt.dispatch_committed
            and attempt.dispatch_sequence > 0
            and attempt.dispatch_sequence < requested_sequence
            and bool(attempt.terminal_status)
        )

    def _recover_retirement_delivery_channels(self) -> tuple[str, ...]:
        """Restore only frozen outbound authority for retirement obligations."""

        service = self._service
        channels = self._channels
        fire = self._fire
        outbox = self._outbox
        start_delivery_only = getattr(channels, "start_delivery_only", None)
        snapshot = getattr(service, "current_employee_transport_snapshot", None)
        if (
            service is None
            or channels is None
            or fire is None
            or outbox is None
            or not callable(start_delivery_only)
            or not callable(snapshot)
        ):
            return ()
        fire_states = fire.list_states()
        retirement_identities = tuple(
            sorted(
                {
                    (state.tenant_key, state.agent_id)
                    for state in fire_states
                    if state.phase in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
                    and state.effect_state("execution_quiesce")
                    in {
                        FireEffectState.EXECUTING,
                        FireEffectState.ACTION_REQUIRED,
                    }
                }
            )
        )
        if not retirement_identities:
            return ()
        employees, principals, hire_states = snapshot()
        pending = outbox.list_pending_delivery_records()
        recovered: list[str] = []
        for tenant_key, agent_id in retirement_identities:
            matching_employees = tuple(
                employee
                for employee in employees
                if employee.tenant_key == tenant_key and employee.agent_id == agent_id
            )
            matching_hires = tuple(
                candidate
                for candidate in hire_states
                if candidate.tenant_key == tenant_key and candidate.agent_id == agent_id
            )
            matching_fires = tuple(
                state
                for state in fire_states
                if state.tenant_key == tenant_key
                and state.agent_id == agent_id
                and state.phase is not FirePhase.SUPERSEDED
            )
            if not (len(matching_employees) == len(matching_hires) == len(matching_fires) == 1):
                raise RuntimeError("employee retirement delivery authority is ambiguous")
            employee = matching_employees[0]
            hire = matching_hires[0]
            fire_state = matching_fires[0]
            matching_principals = tuple(
                principal
                for principal in principals
                if principal.tenant_key == tenant_key
                and principal.agent_id == agent_id
                and principal.bot_principal_id == hire.bot_principal_id
            )
            if len(matching_principals) != 1:
                raise RuntimeError("employee retirement delivery authority is ambiguous")
            principal = matching_principals[0]
            if (
                hire.phase not in {_composition_module().HirePhase.RETIRING, _composition_module().HirePhase.ACTION_REQUIRED}
                or employee.state not in {EmployeeState.RETIRING, EmployeeState.ACTION_REQUIRED}
                or employee.bot_principal_id != hire.bot_principal_id
                or principal.app_id != hire.app_id
                or principal.credential_ref != hire.credential_ref
                or not hire.app_id
                or not hire.credential_ref
                or type(hire.channel_generation) is not int
                or hire.channel_generation <= 0
                or hire.channel_identity_app_id != hire.app_id
                or not isinstance(hire.channel_connection_id, str)
                or not hire.channel_connection_id.startswith("conn_")
                or fire_state.phase not in {FirePhase.RETIRING, FirePhase.ACTION_REQUIRED}
                or fire_state.effect_state("execution_quiesce")
                not in {
                    FireEffectState.EXECUTING,
                    FireEffectState.ACTION_REQUIRED,
                }
                or fire_state.bot_principal_id != hire.bot_principal_id
                or fire_state.app_id != hire.app_id
                or fire_state.credential_ref != hire.credential_ref
            ):
                raise RuntimeError("employee retirement delivery authority is unavailable")
            obligations = tuple(
                record
                for record in pending
                if record.tenant_key == hire.tenant_key
                and record.agent_id == hire.agent_id
                and self._retirement_outbox_is_cutoff_owned(
                    record,
                    requested_sequence=fire_state.requested_sequence,
                )
            )
            if not obligations:
                continue
            status = channels.status(hire.agent_id)
            if status is not None and getattr(status, "state", None) is (_composition_module().ChannelProcessState.READY):
                if not self._retirement_delivery_status_matches(
                    status,
                    hire,
                    require_delivery_only=False,
                ):
                    raise RuntimeError("employee retirement delivery Channel is stale")
                continue
            try:
                status = start_delivery_only(
                    hire.agent_id,
                    hire.app_id,
                    hire.credential_ref,
                    hire.channel_generation,
                    frozen_connection_id=hire.channel_connection_id,
                )
            except ChannelLaunchUnavailable as exc:
                logger.warning(
                    "employee retirement delivery Channel launch deferred: agent_id=%s error=%s",
                    hire.agent_id,
                    type(exc).__name__,
                )
                continue
            if not self._retirement_delivery_status_matches(
                status,
                hire,
                require_delivery_only=True,
            ):
                channels.stop(hire.agent_id)
                logger.warning(
                    "employee retirement delivery Channel readiness deferred: agent_id=%s",
                    hire.agent_id,
                )
                continue
            recovered.append(hire.agent_id)
        return tuple(recovered)

    @staticmethod
    def _retirement_delivery_status_matches(
        status: object,
        hire: DurableHireState,
        *,
        require_delivery_only: bool,
    ) -> bool:
        identity = getattr(status, "identity", {})
        ready_metadata = getattr(status, "ready_metadata", {})
        return bool(
            getattr(status, "state", None) is _composition_module().ChannelProcessState.READY
            and getattr(status, "tenant_key", None) == hire.tenant_key
            and getattr(status, "agent_id", None) == hire.agent_id
            and getattr(status, "bot_principal_id", None) == hire.bot_principal_id
            and getattr(status, "app_id", None) == hire.app_id
            and getattr(status, "generation", None) == hire.channel_generation
            and isinstance(identity, Mapping)
            and identity.get("app_id") == hire.channel_identity_app_id
            and isinstance(ready_metadata, Mapping)
            and ready_metadata.get("connection_id") == hire.channel_connection_id
            and (not require_delivery_only or getattr(status, "delivery_only", None) is True)
        )

    def _reconcile_terminal_ingress(self) -> int:
        ingress = self._ingress
        router = self._router
        if ingress is None or router is None:
            return 0
        router.rebuild_projection()
        ingress.rebuild_projection()
        reconciled = 0
        for acceptance_id, record in tuple(router.state.by_acceptance_id.items()):
            ingress_record = ingress.state.by_acceptance_id.get(acceptance_id)
            if (
                record.state == "terminal"
                and record.reason_code != "control_consumed"
                and ingress_record is not None
                and ingress_record.disposition is None
            ):
                try:
                    queue_owned_message = (
                        getattr(record, "queued_sequence", 0) > 0
                        and getattr(record, "event_type", None) == "im.message.receive_v1"
                    )
                    ownership = None
                    if queue_owned_message:
                        ownership = self._validated_employee_terminal_ownership(
                            acceptance_id,
                            ingress_record,
                            record,
                        )
                        if ownership is None:
                            # Even an execution-shaped terminal cannot dispose
                            # a queued message whose frozen authority no longer
                            # authenticates against its encrypted payload.
                            continue
                    needs_task_failure = (
                        queue_owned_message
                        and getattr(record, "reason_code", "") not in _composition_module()._EMPLOYEE_EXECUTION_TERMINAL_REASONS
                    )
                    if needs_task_failure and not self._anchor_employee_task_failure_response(
                        acceptance_id,
                        ingress_record,
                        record,
                        ownership=ownership,
                    ):
                        # Keep the encrypted payload live and retry on the next
                        # drain/restart until the Employee failure response is
                        # durably anchored.  Never silently disposition a task
                        # for which Main already yielded queue ownership.
                        continue
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code=record.reason_code,
                    )
                    reconciled += 1
                except IngressBlobRetryableError:
                    # One temporarily unreadable encrypted payload must not
                    # starve later independently valid terminal records.
                    continue
                except IngressConflictError:
                    pass
        return reconciled

    def _anchor_employee_task_failure_response(
        self,
        acceptance_id: str,
        ingress_record: object,
        router_record: object,
        *,
        ownership: _ValidatedTargetedOwnership | None = None,
    ) -> bool:
        """Anchor one queue-owned user-task failure before payload GC."""

        lifecycle = self._outbox_lifecycle
        if lifecycle is None:
            return False
        ownership = ownership or self._validated_employee_terminal_ownership(
            acceptance_id, ingress_record, router_record
        )
        if ownership is None:
            return False
        lifecycle.task_failure_response(
            tenant_key=ownership.metadata.tenant_key,
            agent_id=ownership.metadata.agent_id,
            chat_id=ownership.remote_chat_id,
            thread_root_message_id=(ownership.remote_root_id or ownership.remote_message_id),
            command_acceptance_id=acceptance_id,
        )
        return True

    def _refresh_context_bindings(self, projection: ProjectionState) -> bool:
        if self._context_source_factory is None or self._service is None:
            return True
        try:
            with self._context_binding_lock:
                current: dict[str, tuple[str, str, int]] = {}
                states = self._service.list_states()
                for state in states:
                    if state.phase is not _composition_module().HirePhase.ACTIVE:
                        continue
                    principal = projection.bot_principals.get(state.bot_principal_id)
                    if principal is None or not principal.credential_ref:
                        continue
                    current[state.agent_id] = (
                        principal.app_id,
                        principal.credential_ref,
                        state.channel_generation,
                    )
                previous = dict(self._context_bindings)
                non_active = {state.agent_id for state in states if state.phase is not _composition_module().HirePhase.ACTIVE}
                changed_or_removed = {
                    agent_id for agent_id, old_binding in previous.items() if current.get(agent_id) != old_binding
                }
                for agent_id in non_active | changed_or_removed:
                    if (
                        agent_id not in self._context_explicit_invalidations
                        and agent_id not in self._context_projection_invalidations
                    ):
                        self._context_source_factory.invalidate_employee(agent_id)
                        self._context_projection_invalidations.add(agent_id)
                for agent_id in current:
                    if (
                        agent_id in self._context_projection_invalidations
                        and agent_id not in self._context_explicit_invalidations
                    ):
                        self._context_source_factory.reactivate_employee(agent_id)
                        self._context_projection_invalidations.discard(agent_id)
                self._context_bindings = current
            return True
        except Exception as exc:
            logger.error(
                "employee Context binding refresh failed: %s",
                type(exc).__name__,
            )
            return False

    def _compose_context(
        self,
        settings: Any,
        *,
        context_source_factory: EmployeeMessageSourceFactory | None,
        group_memory_backend: Any,
    ) -> None:
        """Compose execution-only Context; failures never block first hire."""
        if not self._runtime_enabled or getattr(settings, "autonomous_visible_employee_limit", 0) == 0:
            self._context_blockers = ("employee_context",)
            return
        if self._service is None or self._writer is None or self._vault is None:
            self._context_blockers = ("employee_context",)
            return
        try:
            legacy_base = str(
                _composition_module().canonicalize_user_home_path(
                    getattr(settings, "autonomous_employee_storage_base", None) or _composition_module().default_employee_storage_base()
                )
            )
            if self._data is None:
                raise RuntimeError("canonical employee data is unavailable")

            self._context_acl = parse_requester_acl(settings)

            def registry_provider() -> ProjectedAgentRegistry:
                assert self._service is not None
                return _composition_module().ProjectedAgentRegistry(
                    self._service.synchronize_projection(),
                    storage_base_path=legacy_base,
                )

            generation = RuntimeEmployeeGenerationAuthority(
                hire_service_provider=lambda: self._service,
                channel_supervisor=self._channels,
                data_composition=self._data,
            )
            source_factory = context_source_factory or LarkEmployeeMessageSourceFactory(
                credential_resolver=self._vault,
                request_timeout_seconds=getattr(
                    settings,
                    "autonomous_context_fetch_timeout_seconds",
                    30.0,
                ),
            )
            backend = group_memory_backend or EmployeeGroupMemoryStore(legacy_base)
            self._group_memory_backend = backend
            self._owns_group_memory_backend = False
            group_reader = AuthorizedGroupMemoryReader(
                registry_provider=registry_provider,
                requester_acl=self._context_acl,
                backend=backend,
            )
            self._context_source_factory = source_factory
            self._context_service = _composition_module().EmployeeContextService(
                registry_provider=registry_provider,
                generation_authority=generation,
                requester_acl=self._context_acl,
                data_composition=self._data,
                group_memory_reader=group_reader,
                source_factory=source_factory,
                config=_composition_module().ThreadContextConfig.from_settings(settings),
                group_ledger=self._group_ledger,
            )
            self._context_blockers = ()
        except Exception as exc:
            logger.error(
                "employee Context composition unavailable: %s",
                type(exc).__name__,
                exc_info=True,
            )
            if self._context_source_factory is not None:
                try:
                    self._context_source_factory.close()
                except Exception:
                    pass
            if self._owns_group_memory_backend and self._group_memory_backend is not None:
                try:
                    self._group_memory_backend.shutdown()
                except Exception:
                    pass
            self._group_memory_backend = None
            self._owns_group_memory_backend = False
            self._context_source_factory = None
            self._context_service = None
            self._context_blockers = ("employee_context",)
            if not self._execution_blockers:
                self._execution_blockers = ("employee_gateway",)

    @staticmethod
    async def _quiesce_loop() -> None:
        current = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
        for task in tasks:
            if is_external_task(task):
                continue
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.get_running_loop().shutdown_default_executor()

    @staticmethod
    def _default_slash_factory(app_id: str, app_secret: str) -> _SlashReconciler:
        client = _composition_module().lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
        return SlashCommandReconciler(
            LarkSlashCommandAPI(client, expected_app_id=app_id)
        )

    def _start_loop(self) -> None:
        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            def keep_selector_wakeable() -> None:
                # Some constrained hosts reject writes to asyncio's internal
                # socketpair.  Thread-safe submissions still reach ``_ready``
                # but cannot wake an indefinitely blocked selector.  Keeping a
                # short timer pending provides a bounded, network-free wakeup
                # path for both activity submission and shutdown.
                if not self._closing:
                    loop.call_later(0.1, keep_selector_wakeable)

            loop.call_soon(self._loop_ready.set)
            loop.call_soon(keep_selector_wakeable)
            loop.run_forever()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._loop_thread = _composition_module().threading.Thread(
            target=run,
            name="employee-hire-activities",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(5.0) or self._loop is None:
            raise RuntimeError("employee activity loop failed to start")

    def _submit_intent(self, intent_id: str) -> None:
        future: concurrent.futures.Future[Any]
        with self._future_lock:
            if self._closing or self._loop is None:
                raise RuntimeError("employee runtime is closing")
            existing = self._intent_futures.get(intent_id)
            if existing is not None and not existing.done():
                self._pending_intent_resubmits.add(intent_id)
                return
            self._pending_intent_resubmits.discard(intent_id)
            future = self._launch_intent_activity_locked(intent_id)

        self._attach_intent_completion(intent_id, future)

    def _launch_intent_activity_locked(
        self,
        intent_id: str,
    ) -> concurrent.futures.Future[Any]:
        """Install one intent activity while ``_future_lock`` is held."""

        loop = self._loop
        if self._closing or loop is None:
            raise RuntimeError("employee runtime is closing")
        coroutine = self._configure_and_notify(intent_id)
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        except BaseException:
            coroutine.close()
            raise
        self._intent_futures[intent_id] = future
        self._futures.add(future)
        return future

    def _attach_intent_completion(
        self,
        intent_id: str,
        future: concurrent.futures.Future[Any],
    ) -> None:
        """Coalesce any in-flight revalidation into exactly one successor."""

        def complete(done: concurrent.futures.Future[Any]) -> None:
            replacement: concurrent.futures.Future[Any] | None = None
            replacement_error: BaseException | None = None
            with self._future_lock:
                self._futures.discard(done)
                if self._intent_futures.get(intent_id) is done:
                    self._intent_futures.pop(intent_id, None)
                    if intent_id in self._pending_intent_resubmits:
                        self._pending_intent_resubmits.discard(intent_id)
                        if not self._closing and self._loop is not None:
                            try:
                                replacement = self._launch_intent_activity_locked(intent_id)
                            except BaseException as exc:
                                replacement_error = exc
            try:
                done.result()
            except Exception as exc:
                logger.error(
                    "employee provisioning activity failed: %s",
                    type(exc).__name__,
                )
            if replacement_error is not None:
                logger.error(
                    "employee provisioning resubmit failed: %s",
                    type(replacement_error).__name__,
                )
            if replacement is not None:
                self._attach_intent_completion(intent_id, replacement)

        future.add_done_callback(complete)

    async def _configure_and_notify(self, intent_id: str) -> None:
        succeeded = False
        try:
            await self._configure_intent(intent_id)
            succeeded = True
        except Exception as exc:
            logger.error(
                "employee provisioning attempt failed; entering automatic recovery: %s",
                type(exc).__name__,
                exc_info=True,
            )
            current = self._require_service().get_state(intent_id)
            has_disposed_failure = current is not None and any(
                effect_state is HireEffectState.ACTION_REQUIRED for _effect_id, effect_state in current.effects
            )
            if not has_disposed_failure:
                succeeded = await self._retry_recovery_intent(intent_id)
        state = self._require_service().get_state(intent_id)
        if state is None:
            return
        status = (
            self._terminal_notification_status(state)
            if succeeded
            else "action_required"
            if state.phase is _composition_module().HirePhase.ACTION_REQUIRED
            or any(effect_state is HireEffectState.ACTION_REQUIRED for _effect_id, effect_state in state.effects)
            else ""
        )
        if status:
            try:
                self._submit_coroutine(
                    self._retry_terminal_notification(state.intent_id, status),
                    label=f"{status} notification",
                )
            except RuntimeError:
                if not self._closing:
                    raise

    async def _notify_hire_terminal(
        self,
        state: DurableHireState,
        status: str,
    ) -> bool:
        if self._notification_async_lock is None:
            self._notification_async_lock = asyncio.Lock()
        async with self._notification_async_lock:
            current = self._require_service().get_state(state.intent_id)
            if current is None or self._terminal_notification_status(current) != status:
                return False
            return await self._notify_hire_terminal_locked(current, status)

    async def _notify_hire_terminal_locked(
        self,
        state: DurableHireState,
        status: str,
    ) -> bool:
        callback = self._notification_status
        writer = self._writer
        if callback is None or writer is None:
            return False
        aggregate_id = f"hire-notification:{state.intent_id}:{status}"
        notification = rebuild_hire_notification_projection(tuple(writer.replay())).get(aggregate_id)
        if notification is not None and notification.phase is HireNotificationPhase.COMMITTED:
            return True
        message_uuid = hire_notification_message_uuid(state.intent_id, status)
        base_payload = {
            "intent_id": state.intent_id,
            "status": status,
            "message_uuid": message_uuid,
        }

        def commit(event_type: str, **extra: str) -> None:
            with writer.transaction_guard():
                last = writer.get_last_frame()
                result = writer.commit(
                    (
                        JournalEvent(
                            event_type=event_type,
                            aggregate_id=aggregate_id,
                            payload={**base_payload, **extra},
                        ),
                    ),
                    writer.get_aggregate_versions((aggregate_id,)),
                    expected_head_sequence=0 if last is None else last.sequence,
                    expected_head_hash="" if last is None else last.frame_hash,
                )
            if result.state.value != "anchored":
                raise RuntimeError("hire notification event was not anchored")

        if notification is None:
            commit("hire.notification.prepared")
        elif notification.phase in {
            HireNotificationPhase.EXECUTING,
            HireNotificationPhase.ACTION_REQUIRED,
        }:
            commit("hire.notification.retry_requested")
        current = self._require_service().get_state(state.intent_id)
        if current is None or self._terminal_notification_status(current) != status:
            return False
        state = current
        commit("hire.notification.executing")
        try:
            receipt = await safe_wait_for(
                asyncio.to_thread(callback, state, status),
                timeout=15.0,
                action="employee hire notification",
            )
        except Exception:
            commit("hire.notification.action_required")
            return False
        if receipt is None or receipt is False:
            commit("hire.notification.action_required")
            return False
        receipt_ref = (
            receipt
            if isinstance(receipt, str) and receipt
            else getattr(receipt, "message_id", "") or f"ack:{message_uuid}"
        )
        commit("hire.notification.committed", receipt_ref=receipt_ref)
        return True

    async def _retry_terminal_notification(
        self,
        intent_id: str,
        status: str,
    ) -> bool:
        if self._notification_status is None:
            return False
        for delay in (0.0, *_composition_module()._RECOVERY_RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            if self._closing:
                return False
            state = self._require_service().get_state(intent_id)
            if state is None or self._terminal_notification_status(state) != status:
                return False
            if await self._notify_hire_terminal(state, status):
                return True
        return False

    def _submit_coroutine(
        self,
        coroutine: Any,
        *,
        label: str,
    ) -> concurrent.futures.Future[Any]:
        if self._closing or self._loop is None:
            if hasattr(coroutine, "close"):
                coroutine.close()
            raise RuntimeError("employee runtime is closing")
        future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self._loop,
        )
        with self._future_lock:
            self._futures.add(future)

        def complete(done: concurrent.futures.Future[Any]) -> None:
            with self._future_lock:
                self._futures.discard(done)
            try:
                done.result()
            except Exception as exc:
                logger.error(
                    "employee %s activity failed: %s",
                    label,
                    type(exc).__name__,
                )

        future.add_done_callback(complete)
        return future

    async def _recover_runtime(
        self,
        pending_intents: list[str],
    ) -> EmployeeRecoverySummary:
        failed_intents: list[str] = []
        if pending_intents:
            results = await asyncio.gather(
                *(
                    self._configure_intent(
                        intent_id,
                        force_slash_refresh=True,
                    )
                    for intent_id in pending_intents
                ),
                return_exceptions=True,
            )
            failed_intents = [
                intent_id
                for intent_id, result in zip(
                    pending_intents,
                    results,
                    strict=True,
                )
                if isinstance(result, BaseException)
            ]
        if failed_intents:
            retry_results = await asyncio.gather(
                *(
                    self._retry_recovery_intent(intent_id)
                    for intent_id in failed_intents
                ),
                return_exceptions=True,
            )
            failures = sum(isinstance(result, BaseException) or result is False for result in retry_results)
            if any(isinstance(result, BaseException) for result in retry_results):
                logger.error(
                    "employee recovery could not be durably isolated for %d intent(s)",
                    failures,
                )
                self._execution_blockers = ("hire_recovery",)
                return _composition_module().EmployeeRecoverySummary()
            if failures:
                logger.error(
                    "employee recovery isolated %d exhausted intent(s) as action_required",
                    failures,
                )
        self._start_monitor_in_loop()
        await self._retry_terminal_notifications()
        if self._membership_audit_ready:
            self._require_service().mark_runtime_recovered()
        self._start_reporting_worker()
        if not self._execution_blockers:
            self._start_dispatch_worker()
        return _composition_module().EmployeeRecoverySummary()

    async def _retry_recovery_intent(
        self,
        intent_id: str,
    ) -> bool:
        for delay in _composition_module()._RECOVERY_RETRY_DELAYS:
            if self._closing:
                raise RuntimeError("employee runtime is closing")
            await asyncio.sleep(delay)
            try:
                await self._configure_intent(
                    intent_id,
                    force_slash_refresh=True,
                )
                return True
            except Exception as exc:
                logger.warning(
                    "employee provisioning recovery attempt failed: %s",
                    type(exc).__name__,
                    exc_info=True,
                )
                continue
        self._require_service().mark_recovery_action_required(
            intent_id,
            error_code="recovery_exhausted",
        )
        return False

    def _start_monitor_in_loop(self) -> None:
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_channels())

    async def _monitor_channels(self) -> None:
        while not self._closing:
            service = self._require_service()
            channels = self._channels
            if channels is not None:
                for state in service.list_states():
                    if state.phase not in {
                        _composition_module().HirePhase.ACTIVE,
                        _composition_module().HirePhase.READY_PENDING_VERIFICATION,
                    }:
                        continue
                    try:
                        channel_status = channels.status(state.agent_id)
                        if self._closing:
                            return
                        same_generation = (
                            channel_status is not None and channel_status.generation == state.channel_generation
                        )
                        unhealthy_current_channel = same_generation and (
                            channel_status.state
                            in {
                                _composition_module().ChannelProcessState.CRASHED,
                                _composition_module().ChannelProcessState.FAILED,
                                _composition_module().ChannelProcessState.STOPPED,
                            }
                            or (
                                channel_status.state is _composition_module().ChannelProcessState.STARTING
                                and channel_status.error_code == "channel-reconnecting"
                                and isinstance(
                                    channel_status.ready_metadata.get("reconnecting_at"),
                                    (int, float),
                                )
                                and _composition_module().time.time() - channel_status.ready_metadata["reconnecting_at"]
                                >= _composition_module()._CHANNEL_RECONNECT_GRACE_SECONDS
                            )
                        )
                        ready_binding_drift = (
                            channel_status is not None
                            and state.phase is _composition_module().HirePhase.ACTIVE
                            and channel_status.state is _composition_module().ChannelProcessState.READY
                            and (
                                channel_status.generation != state.channel_generation
                                or channel_status.app_id != state.app_id
                                or channel_status.tenant_key != state.tenant_key
                                or channel_status.bot_principal_id != state.bot_principal_id
                                or not isinstance(
                                    channel_status.identity,
                                    Mapping,
                                )
                                or channel_status.identity.get("app_id") != state.channel_identity_app_id
                                or not isinstance(
                                    channel_status.ready_metadata,
                                    Mapping,
                                )
                                or channel_status.ready_metadata.get("connection_id") != state.channel_connection_id
                            )
                        )
                        if channel_status is not None and (unhealthy_current_channel or ready_binding_drift):
                            if self._closing:
                                return
                            service.begin_channel_revalidation(
                                state.intent_id,
                                observed_generation=state.channel_generation,
                            )
                            if self._closing:
                                return
                            self._submit_intent(state.intent_id)
                        elif state.phase is _composition_module().HirePhase.READY_PENDING_VERIFICATION:
                            if self._closing:
                                return
                            self._resume_pending_activation(state)
                    except Exception as exc:
                        logger.error(
                            "employee Channel monitor failed closed: %s",
                            type(exc).__name__,
                        )
            if self._closing:
                return
            await asyncio.sleep(2.0)

    async def _retry_terminal_notifications(self) -> None:
        if self._notification_status is None:
            return
        pending = [
            self._retry_terminal_notification(state.intent_id, status)
            for state in self._require_service().list_states()
            if (status := self._terminal_notification_status(state))
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @staticmethod
    def _terminal_notification_status(state: DurableHireState) -> str:
        has_action_required_effect = any(
            effect_state is HireEffectState.ACTION_REQUIRED for _effect_id, effect_state in state.effects
        )
        return (
            "ready"
            if state.phase is _composition_module().HirePhase.READY_PENDING_VERIFICATION
            else "active"
            if state.phase is _composition_module().HirePhase.ACTIVE
            else "action_required"
            if state.phase is _composition_module().HirePhase.ACTION_REQUIRED or has_action_required_effect
            else ""
        )

    def _resume_pending_activation(self, state: DurableHireState) -> None:
        """Replace a legacy pending activation with a fresh automatic run."""
        service = self._require_service()
        current = service.get_state(state.intent_id)
        if (
            current is None
            or current.phase is not _composition_module().HirePhase.READY_PENDING_VERIFICATION
            or current.channel_generation <= 0
        ):
            return
        service.begin_channel_revalidation(
            current.intent_id,
            observed_generation=current.channel_generation,
        )
        self._submit_intent(current.intent_id)

    async def _configure_intent(
        self,
        intent_id: str,
        *,
        force_slash_refresh: bool = False,
    ) -> None:
        service = self._require_service()
        state = service.get_state(intent_id)
        if state is None:
            return
        if state.phase in {
            _composition_module().HirePhase.PROVISIONING_APP,
            _composition_module().HirePhase.STORING_CREDENTIAL,
        }:
            state = await service.run_provisioning(intent_id)
        if state.phase not in {_composition_module().HirePhase.CONFIGURING, _composition_module().HirePhase.VALIDATING}:
            return
        for launch_attempt in range(2):
            state = service.get_state(intent_id) or state
            generation = self._target_channel_generation(state)
            selected_slash_effect_id = service.select_slash_reconcile_effect(
                state.intent_id,
                generation=generation,
                force_refresh=force_slash_refresh,
            )
            current_effect_ids = {
                selected_slash_effect_id,
                f"channel-start:{generation}",
            }
            effect_types = dict(state.effect_types)
            for effect_id, effect_state in state.effects:
                if (
                    effect_state not in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
                    or effect_id in current_effect_ids
                ):
                    continue
                effect_type = effect_types.get(effect_id, "")
                parts = effect_id.split(":")
                effect_generation = (
                    int(parts[1])
                    if (
                        effect_type == "slash_reconciliation"
                        and len(parts) == 3
                        and parts[0] == "slash-reconcile"
                        and parts[1].isdigit()
                    )
                    or (
                        effect_type == "employee_channel_start"
                        and len(parts) == 2
                        and parts[0] == "channel-start"
                        and parts[1].isdigit()
                    )
                    else None
                )
                if effect_generation is None or effect_generation > generation:
                    continue
                state = service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type=effect_type,
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": "superseded_reconfiguration"},
                )
            await self._reconcile_slash(
                state,
                generation=generation,
                force_refresh=force_slash_refresh,
            )
            state = service.get_state(intent_id) or state
            try:
                await self._start_channel(state)
            except RuntimeError:
                if launch_attempt == 0:
                    continue
                raise
            service.prepare_automatic_activation(state.intent_id)
            service.commit_automatic_activation(
                state.intent_id,
                activated_at=_composition_module().time.time(),
            )
            if not self._refresh_context_bindings(service.projection_state):
                self._context_blockers = ("context_binding_sync",)
            return

    @staticmethod
    async def _await_external_task_terminal(
        task: asyncio.Task[Any],
    ) -> tuple[Any, bool]:
        """Defer caller cancellation until an external task truly stops."""

        cancelled = False
        while True:
            try:
                return await asyncio.shield(task), cancelled
            except asyncio.CancelledError as exc:
                current = asyncio.current_task()
                caller_cancelled = cancelled or (current is not None and current.cancelling() > 0)
                if task.cancelled():
                    raise ExternalTaskTerminalError(
                        caller_cancelled=caller_cancelled,
                        child_cancelled=True,
                    ) from exc
                if task.done():
                    try:
                        return task.result(), True
                    except asyncio.CancelledError as child_cancel:
                        raise ExternalTaskTerminalError(
                            caller_cancelled=True,
                            child_cancelled=True,
                        ) from child_cancel
                    except Exception as child_error:
                        raise ExternalTaskTerminalError(
                            caller_cancelled=True,
                            child_cancelled=False,
                        ) from child_error
                cancelled = caller_cancelled
            except Exception as exc:
                if cancelled:
                    raise ExternalTaskTerminalError(
                        caller_cancelled=True,
                        child_cancelled=False,
                    ) from exc
                raise

    async def _reconcile_slash(
        self,
        state: DurableHireState,
        *,
        generation: int,
        force_refresh: bool,
    ) -> VerifiedSlashState:
        service = self._require_service()
        effect_id = service.select_slash_reconcile_effect(
            state.intent_id,
            generation=generation,
            force_refresh=force_refresh,
        )
        current = service.get_state(state.intent_id) or state
        try:
            lease = self._external_mutation_gate.acquire(
                current.tenant_key,
                current.agent_id,
                ExternalMutationKind.SLASH_RECONCILIATION,
            )
        except ExternalMutationFenced:
            raise RuntimeError("employee is retiring") from None
        external_started = False
        disposition_anchored = False
        deferred_cancel = False
        try:
            effect_state = current.effect_state(effect_id)
            if effect_state is None:
                current = service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="slash_reconciliation",
                    next_state=HireEffectState.PREPARED,
                )
                effect_state = current.effect_state(effect_id)
            if effect_state is HireEffectState.PREPARED:
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="slash_reconciliation",
                    next_state=HireEffectState.EXECUTING,
                )
            current = service.get_state(state.intent_id) or state
            if current.effect_state(effect_id) is HireEffectState.COMMITTED:
                return _composition_module().VerifiedSlashState(
                    spec_hash=current.slash_spec_hash,
                    observed_hash=current.slash_observed_hash,
                    observed=(),
                )
            if current.effect_state(effect_id) is not HireEffectState.EXECUTING:
                raise RuntimeError("Slash reconciliation requires manual action")
            if self._vault is None or self._slash_factory is None:
                raise RuntimeError("Slash composition unavailable")
            if self._closing:
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="slash_reconciliation",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": "recovery_exhausted"},
                )
                disposition_anchored = True
                raise asyncio.CancelledError
            resolve_task = asyncio.create_task(
                asyncio.to_thread(
                    self._vault.resolve,
                    current.credential_ref,
                    current.agent_id,
                    current.app_id,
                ),
                name=external_task_name("slash-credential-resolve"),
            )
            try:
                secret, resolve_cancelled = await self._await_external_task_terminal(resolve_task)
            except ExternalTaskTerminalError as exc:
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="slash_reconciliation",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": "recovery_exhausted"},
                )
                disposition_anchored = True
                if exc.caller_cancelled:
                    raise asyncio.CancelledError from None
                raise RuntimeError("Slash credential resolution failed") from None
            except Exception:
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="slash_reconciliation",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": "recovery_exhausted"},
                )
                disposition_anchored = True
                raise RuntimeError("Slash credential resolution failed") from None
            deferred_cancel = deferred_cancel or resolve_cancelled
            if deferred_cancel or self._closing:
                secret = ""
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="slash_reconciliation",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": "recovery_exhausted"},
                )
                disposition_anchored = True
                raise asyncio.CancelledError
            reconciler = self._slash_factory(current.app_id, secret)
            external_started = True
            reconcile_task = asyncio.create_task(
                reconciler.reconcile(),
                name=external_task_name("slash-reconciliation"),
            )
            try:
                verified, reconcile_cancelled = await self._await_external_task_terminal(reconcile_task)
                deferred_cancel = deferred_cancel or reconcile_cancelled
            except Exception as exc:
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="slash_reconciliation",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": "recovery_exhausted"},
                )
                disposition_anchored = True
                if isinstance(exc, ExternalTaskTerminalError) and exc.caller_cancelled:
                    raise asyncio.CancelledError from None
                raise RuntimeError("Slash reconciliation failed") from None
            finally:
                secret = ""
            service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="slash_reconciliation",
                next_state=HireEffectState.COMMITTED,
                metadata={
                    "slash_spec_hash": verified.spec_hash,
                    "slash_observed_hash": verified.observed_hash,
                    "slash_verified_at": str(_composition_module().time.time()),
                },
            )
            disposition_anchored = True
        finally:
            if external_started and not disposition_anchored:
                current = service.get_state(state.intent_id) or state
                if current.effect_state(effect_id) in {
                    HireEffectState.PREPARED,
                    HireEffectState.EXECUTING,
                }:
                    try:
                        service.commit_effect_transition(
                            state.intent_id,
                            effect_id=effect_id,
                            effect_type="slash_reconciliation",
                            next_state=HireEffectState.ACTION_REQUIRED,
                            metadata={"error_code": "terminal_anchor_failed"},
                        )
                    except Exception:
                        pass
                    else:
                        disposition_anchored = True
            if not external_started or disposition_anchored:
                lease.release()
        if deferred_cancel:
            raise asyncio.CancelledError
        return verified

    async def _start_channel(
        self,
        state: DurableHireState,
        *,
        force_next_generation: bool = False,
    ) -> Any:
        service = self._require_service()
        generation = state.channel_generation + 1 if force_next_generation else self._target_channel_generation(state)
        effect_id = f"channel-start:{generation}"
        current = service.get_state(state.intent_id) or state
        try:
            lease = self._external_mutation_gate.acquire(
                current.tenant_key,
                current.agent_id,
                ExternalMutationKind.CHANNEL_START,
            )
        except ExternalMutationFenced:
            raise RuntimeError("employee is retiring") from None
        external_started = False
        disposition_anchored = False
        deferred_cancel = False
        try:
            effect_state = current.effect_state(effect_id)
            if effect_state is None:
                current = service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_channel_start",
                    next_state=HireEffectState.PREPARED,
                )
                effect_state = current.effect_state(effect_id)
            if effect_state is HireEffectState.PREPARED:
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_channel_start",
                    next_state=HireEffectState.EXECUTING,
                )
            if self._channels is None:
                raise RuntimeError("employee Channel composition unavailable")
            channel_task = asyncio.create_task(
                asyncio.to_thread(
                    self._channels.start,
                    current.agent_id,
                    current.app_id,
                    current.credential_ref,
                    generation,
                    self._event_callback(current.intent_id, generation),
                ),
                name=external_task_name("employee-channel-start"),
            )
            external_started = True
            try:
                status, deferred_cancel = await self._await_external_task_terminal(channel_task)
            except Exception as exc:
                error_code = (
                    "start-cancelled"
                    if isinstance(exc, ExternalTaskTerminalError) and exc.child_cancelled
                    else f"start-{type(exc).__name__}"
                )
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_channel_start",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": error_code},
                )
                disposition_anchored = True
                if isinstance(exc, ExternalTaskTerminalError) and exc.caller_cancelled:
                    raise asyncio.CancelledError from None
                raise RuntimeError("employee Channel start failed") from None
            identity_app_id = status.identity.get("app_id")
            connection_id = status.ready_metadata.get("connection_id")
            if (
                status.state is not _composition_module().ChannelProcessState.READY
                or identity_app_id != current.app_id
                or not isinstance(connection_id, str)
                or not connection_id
            ):
                error_code = getattr(status, "error_code", "") or "invalid-ready"
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_channel_start",
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata={"error_code": error_code},
                )
                disposition_anchored = True
                if deferred_cancel:
                    raise asyncio.CancelledError
                raise RuntimeError("employee Channel did not become ready")
            current = service.get_state(state.intent_id) or state
            if current.effect_state(effect_id) is HireEffectState.EXECUTING:
                service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_channel_start",
                    next_state=HireEffectState.COMMITTED,
                    metadata={
                        "app_id": current.app_id,
                        "generation": str(generation),
                        "identity_app_id": identity_app_id,
                        "connection_id": connection_id,
                        "channel_verified_at": str(_composition_module().time.time()),
                    },
                )
            disposition_anchored = True
        finally:
            if external_started and not disposition_anchored:
                current = service.get_state(state.intent_id) or state
                if current.effect_state(effect_id) in {
                    HireEffectState.PREPARED,
                    HireEffectState.EXECUTING,
                }:
                    try:
                        service.commit_effect_transition(
                            state.intent_id,
                            effect_id=effect_id,
                            effect_type="employee_channel_start",
                            next_state=HireEffectState.ACTION_REQUIRED,
                            metadata={"error_code": "terminal_anchor_failed"},
                        )
                    except Exception:
                        pass
                    else:
                        disposition_anchored = True
            if not external_started or disposition_anchored:
                lease.release()
        if deferred_cancel:
            raise asyncio.CancelledError
        return status

    @staticmethod
    def _target_channel_generation(state: DurableHireState) -> int:
        attempts: list[tuple[int, HireEffectState]] = []
        for effect_id, effect_state in state.effects:
            if not effect_id.startswith("channel-start:"):
                continue
            generation_text = effect_id.removeprefix("channel-start:")
            if generation_text.isdigit() and int(generation_text) > 0:
                attempts.append((int(generation_text), effect_state))
        if attempts:
            attempted_generation, effect_state = max(attempts)
            if effect_state is HireEffectState.ACTION_REQUIRED:
                return attempted_generation + 1
            if attempted_generation > state.channel_generation:
                return attempted_generation
        generation = state.channel_generation or 1
        if state.phase is _composition_module().HirePhase.VALIDATING and state.channel_generation > 0:
            return generation + 1
        return generation
