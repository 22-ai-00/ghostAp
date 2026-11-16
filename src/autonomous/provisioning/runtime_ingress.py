"""Ingress admission, handoff finalization, control, and channel-event helpers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import math
import re
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ...trust.models import ActorKind, EffectiveTrust, TrustZone
from ...trust.resolver import TrustZoneResolver
from ..acceptance.main_bot_warning_outbox import (
    MainBotWarningRetryableDeliveryError,
)
from ..authorization import EmployeeAuthorizationScope
from ..context.group_ledger import GroupEventPayload
from ..context.models import (
    AuthorizedContextRequest,
)
from ..data.ports import HistoryQuerySpec, MemoryQuerySpec
from ..data.query import AuthenticatedDataRequest, QueryDeniedError
from ..domain import EmployeeState, WorkerType
from ..gateway.coordinator import (
    EmployeeDispatchReportingRecoveryResult,
)
from ..ingress.router import (
    RouterResponseObligation,
)
from ..ingress.service import (
    IngressBlobError,
    IngressBlobRetryableError,
    IngressConflictError,
)
from ..ingress.targeted_task import (
    TARGETED_TASK_INPUT_KIND,
    TargetedTaskParseResult,
    TargetedTaskState,
    is_group_slash_observation,
    parse_targeted_group_task,
)
from ..membership import (
    MembershipBindingError,
)
from .hire_state import (
    HireEffectState,
)

if TYPE_CHECKING:
    from .composition import (
        EffectiveTrust,
        EmployeeIngressMetadata,
        EmployeeIngressPayload,
        ReadyEmployeeIngressTarget,
        TargetedTaskParseResult,
        _EmployeeHandoffProjection,
        _ValidatedTargetedOwnership,
        threading,
    )

logger = logging.getLogger(__name__)


def _composition_module():
    """Return the composition module for runtime-patchable symbols."""
    from . import composition as _comp_mod

    return _comp_mod


class EmployeeRuntimeIngressMixin:
    """Mixin providing ingress admission, handoff, control, and event flows."""

    def _resolve_employee_requester_principal(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        owner_principal_id: str,
        sender_principal_id: str,
        sender_union_id: str,
        app_id: str = "",
        bot_principal_id: str = "",
        channel_generation: int = 0,
        connection_id: str = "",
        channel_identity_app_id: str = "",
    ) -> str | None:
        """Map an employee-app Open ID to the main-bot owner via union ID."""

        values = (
            tenant_key,
            agent_id,
            owner_principal_id,
            sender_principal_id,
            sender_union_id,
        )
        if any(not isinstance(value, str) for value in values):
            return None
        if not sender_union_id:
            return None
        service = self._service
        if service is None:
            return None
        service.synchronize_projection()
        candidates = tuple(
            state for state in service.list_states() if state.tenant_key == tenant_key and state.agent_id == agent_id
        )
        if len(candidates) != 1:
            return None
        state = candidates[0]
        if (
            state.phase is not _composition_module().HirePhase.ACTIVE
            or state.requester_principal_id != owner_principal_id
            or state.requester_union_id != sender_union_id
        ):
            return None
        strict_transport = any(
            (
                app_id,
                bot_principal_id,
                channel_generation,
                connection_id,
                channel_identity_app_id,
            )
        )
        if strict_transport and (
            state.app_id != app_id
            or state.bot_principal_id != bot_principal_id
            or state.channel_generation != channel_generation
            or state.channel_connection_id != connection_id
            or state.channel_identity_app_id != channel_identity_app_id
        ):
            return None
        return owner_principal_id

    def _authorized_targeted_group_task(
        self,
        record: Any,
        payload: EmployeeIngressPayload | None,
    ) -> TargetedTaskParseResult | None:
        """Use the Router's independent authority fence for one group command."""

        if not isinstance(payload, _composition_module().EmployeeIngressPayload):
            return None
        router = self._router
        metadata = getattr(record, "metadata", None)
        part = payload.normalized_parts[0] if len(payload.normalized_parts) == 1 else None
        if (
            not isinstance(metadata, _composition_module().EmployeeIngressMetadata)
            or not isinstance(part, Mapping)
            or not isinstance(part.get("mentions"), tuple)
            or not is_group_slash_observation(part)
            or not self._employee_ingress_transport_is_current(metadata)
        ):
            return None
        if router is None:
            return TargetedTaskParseResult(TargetedTaskState.INDETERMINATE)
        try:
            return router.classify_targeted_group_task(metadata, payload)
        except Exception:
            return TargetedTaskParseResult(TargetedTaskState.INDETERMINATE)

    def _owner_p2p_requester(
        self,
        record: Any,
        payload: EmployeeIngressPayload,
    ) -> str | None:
        """Resolve one SDK P2P sender to the durable main-Bot owner."""

        if len(payload.normalized_parts) != 1:
            return None
        part = payload.normalized_parts[0]
        metadata = getattr(record, "metadata", None)
        if not isinstance(metadata, _composition_module().EmployeeIngressMetadata):
            return None
        if (
            not isinstance(part, Mapping)
            or part.get("type") != "message"
            or part.get("chat_type") != "p2p"
            or part.get("sender_type") != "user"
            or part.get("sender_id_type") != "open_id"
            or part.get("sender_id") != metadata.sender_principal_id
            or part.get("sender_tenant_key") != metadata.tenant_key
            or not self._employee_ingress_transport_is_current(metadata)
        ):
            return None
        sender_union_id = part.get("sender_union_id")
        if not isinstance(sender_union_id, str) or not sender_union_id:
            return None
        service = self._service
        if service is None:
            return None
        projection = service.synchronize_projection()
        states = tuple(
            state
            for state in service.list_states()
            if state.tenant_key == metadata.tenant_key
            and state.agent_id == metadata.agent_id
            and state.phase is _composition_module().HirePhase.ACTIVE
        )
        if len(states) != 1:
            return None
        state = states[0]
        employee = projection.employees.get(metadata.agent_id)
        if (
            employee is None
            or employee.agent_id != metadata.agent_id
            or employee.tenant_key != metadata.tenant_key
            or employee.owner_principal_id != state.requester_principal_id
            or employee.state is not EmployeeState.ACTIVE
            or employee.worker_type is not WorkerType.VISIBLE
            or employee.bot_principal_id != metadata.bot_principal_id
        ):
            return None
        owner_principal_id = employee.owner_principal_id
        canonical = self._resolve_employee_requester_principal(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            owner_principal_id=owner_principal_id,
            sender_principal_id=metadata.sender_principal_id,
            sender_union_id=sender_union_id,
        )
        coordinates = _composition_module()._bound_remote_coordinates(metadata, part)
        thread_id = part.get("feishu_thread_id", "")
        if (
            canonical != owner_principal_id
            or coordinates is None
            or not isinstance(thread_id, str)
            or self._context_acl is None
        ):
            return None
        chat_id, message_id, root_id = coordinates
        try:
            request = AuthorizedContextRequest(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                bot_principal_id=metadata.bot_principal_id,
                app_id=metadata.app_id,
                channel_generation=metadata.channel_generation,
                chat_id=chat_id,
                thread_root_message_id=root_id or message_id,
                feishu_thread_id=thread_id,
                current_message_id=message_id,
                requester_principal_id=canonical,
                source_requester_principal_id=metadata.sender_principal_id,
                authorization_scope=EmployeeAuthorizationScope.OWNER_P2P,
            )
            authorized = self._context_acl.is_authorized(request)
        except Exception:
            return None
        return canonical if authorized is True else None

    @contextmanager
    def _employee_admission_singleflight(self, acceptance_id: str):
        """Serialize one acceptance without blocking unrelated acceptance IDs."""

        with self._employee_admission_locks_guard:
            entry = self._employee_admission_locks.get(acceptance_id)
            if entry is None:
                lock = _composition_module().threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
                references = 1
            else:
                lock, references = entry
                references += 1
            self._employee_admission_locks[acceptance_id] = (lock, references)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._employee_admission_locks_guard:
                current = self._employee_admission_locks.get(acceptance_id)
                if current is None or current[0] is not lock:
                    raise RuntimeError("employee admission singleflight state drifted")
                remaining = current[1] - 1
                if remaining == 0:
                    del self._employee_admission_locks[acceptance_id]
                else:
                    self._employee_admission_locks[acceptance_id] = (
                        lock,
                        remaining,
                    )

    def _admit_employee_ingress_once(self, acceptance_id: str) -> bool:
        """Apply policy and route one accepted Inbox item without executing it.

        This path is deliberately idempotent.  Channel acceptance callbacks use
        it to cross the Router queue boundary even while the serialized dispatch
        worker is blocked waiting for a long employee execution.  The periodic
        worker reuses the same operation as the durable recovery scan.  Every
        entrance shares the same per-acceptance transaction fence so policy,
        payload reads, routing, and disposition cannot observe split snapshots.
        """

        if not isinstance(acceptance_id, str) or not acceptance_id or getattr(self, "_closing", False):
            return False
        try:
            with self._employee_admission_singleflight(acceptance_id):
                return self._admit_employee_ingress_once_serialized(acceptance_id)
        finally:
            self._notify_employee_reporting()

    def _admit_employee_ingress_once_serialized(self, acceptance_id: str) -> bool:
        """Perform one admission while its acceptance-specific fence is held."""

        if getattr(self, "_closing", False):
            return False
        ingress = self._ingress
        router = self._router
        if ingress is None or router is None:
            # Preserve the startup-time control lane: Channel callbacks can
            # arrive before the full dispatch composition is attached.
            return bool(self._handle_control_ingress(acceptance_id))
        record = self._employee_handoff_record_snapshot(ingress, acceptance_id)
        if record is None or getattr(record, "disposition", None) is not None:
            return False
        metadata = getattr(record, "metadata", None)
        if (
            getattr(metadata, "event_type", None) == "im.message.receive_v1"
            and getattr(record, "transport_message_proof", True) is not True
        ):
            # Legacy public message acceptance did not prove that the callback
            # transport matched the encrypted payload.  Router deliberately
            # excludes it; converge it here before candidate eligibility so a
            # missing Router record cannot raise/HoL later accepted messages.
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code="invalid_transport_proof",
                )
            except IngressConflictError:
                pass
            return True
        routed = self._employee_handoff_record_snapshot(router, acceptance_id)
        if routed is not None and getattr(routed, "state", None) in {
            "queued",
            "dispatching",
            "terminal",
        }:
            return False
        eligibility = getattr(router, "is_inbox_candidate_eligible", None)
        if callable(eligibility) and not eligibility(acceptance_id):
            return False
        try:
            payload = ingress.get_payload(acceptance_id)
        except IngressBlobRetryableError:
            router.defer_inbox_candidate(acceptance_id)
            return True
        except IngressBlobError:
            # Let the Router durably converge every authenticated permanent
            # payload failure to inbox_not_dispatchable.
            router.route(acceptance_id)
            return True
        first = payload.normalized_parts[0] if len(payload.normalized_parts) == 1 else None
        if (
            isinstance(first, Mapping)
            and first.get("type") == "membership_event"
            and self._handle_control_ingress(acceptance_id)
        ):
            return True
        try:
            owner_p2p_requester = self._owner_p2p_requester(record, payload)
        except Exception:
            owner_p2p_requester = None
        targeted_group_task = self._authorized_targeted_group_task(record, payload)
        try:
            trust = self._managed_employee_ingress_trust(record, payload)
        except Exception:
            trust = self._unknown_employee_ingress_trust()
        if self._handle_control_ingress(
            acceptance_id,
            targeted_group_task=targeted_group_task,
            targeted_group_task_classified=True,
        ):
            return True
        if targeted_group_task is not None and targeted_group_task.state is TargetedTaskState.INDETERMINATE:
            # One independent Router pass is the bounded recovery for a
            # classifier dependency failure.  It either queues the authenticated
            # command or records an explicit terminal denial.
            self._record_employee_ingress_group_event(acceptance_id)
            routed = self._employee_handoff_record_snapshot(router, acceptance_id)
            if routed is None or getattr(routed, "state", None) not in {
                "queued",
                "dispatching",
                "terminal",
            }:
                router.route(acceptance_id)
            return True
        if (
            owner_p2p_requester is None
            and targeted_group_task is None
            and trust is not None
            and trust.zone is not TrustZone.MANAGED_AGENT_GROUP
        ):
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="ignored",
                    reason_code="authority_denied",
                )
            except IngressConflictError:
                pass
            return True
        if (
            owner_p2p_requester is None
            and targeted_group_task is None
            and trust is not None
            and trust.actor is ActorKind.EMPLOYEE
        ):
            # Employee Channel SDK ingress has no authenticated causal envelope.
            # Only the main Bot reply path can correlate a server parent/root
            # message to an anchored Outbox record.
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="ignored",
                    reason_code="authority_denied",
                )
            except IngressConflictError:
                pass
            return True
        # Employee Bot subscriptions also observe group slash commands owned by
        # the main Bot.  Consume those after employee-specific controls.
        if owner_p2p_requester is None and self._handle_main_bot_group_command_ingress(
            acceptance_id,
            targeted_group_task=targeted_group_task,
        ):
            return True
        # Project shared group context before routing so terminal payload GC can
        # never race the context snapshot used by dispatch.
        if owner_p2p_requester is None:
            self._record_employee_ingress_group_event(acceptance_id)
        routed = self._employee_handoff_record_snapshot(router, acceptance_id)
        if routed is None or getattr(routed, "state", None) not in {
            "queued",
            "dispatching",
            "terminal",
        }:
            router.route(acceptance_id)
            return True
        return False

    def _drain_employee_dispatch_once(self) -> bool:
        ingress = self._ingress
        router = self._router
        dispatch = self._dispatch
        if ingress is None or router is None or dispatch is None:
            return False
        worked = False
        employee_runtime = getattr(dispatch, "employee_runtime", None)
        if employee_runtime is not None:
            employee_runtime.sweep_idle()
        ingress.rebuild_projection()
        router.rebuild_projection()
        for acceptance_id in tuple(ingress.state.by_acceptance_id):
            worked = self._admit_employee_ingress_once(acceptance_id) or worked
        prepare_next = getattr(dispatch, "prepare_next", None)
        execute_prepared = getattr(dispatch, "execute_prepared", None)
        try:
            if callable(prepare_next) and callable(execute_prepared):
                prepared = prepare_next()
                if prepared is not None:
                    # The Router dispatching state and attempt binding are now
                    # anchored and visible.  Wake Main-Bot handoff waiters before
                    # the potentially long external Agent execution begins.
                    self._notify_employee_handoff_progress()
                    self._notify_employee_reporting()
                    execute_prepared(prepared)
                    self._notify_employee_reporting()
                    worked = True
            else:
                worked = dispatch.dispatch_next() is not None or worked
        except Exception:
            # prepare/execute can fail after Router dispatching or terminal
            # attempt facts have committed but before the corresponding Outbox
            # snapshot is appended.  Repair that exact gap on the next tick;
            # never require a process restart to make the Employee reply visible.
            self._dispatch_recovery_pending = True
            self._notify_employee_reporting()
            raise
        if self._fire is not None:
            try:
                worked = bool(self._fire.reconcile_draining()) or worked
            except Exception as exc:
                logger.error(
                    "employee drain reconciliation failed closed: %s",
                    type(exc).__name__,
                )
        reporting_thread = self._reporting_thread
        if reporting_thread is None or not reporting_thread.is_alive():
            # Preserve deterministic one-shot drains used by recovery tools and
            # narrow embedders.  Production workers always use the independent
            # reporting thread, so external delivery never serializes the Agent
            # execution lane.
            worked = self._drain_employee_reporting_once() or worked
        else:
            self._notify_employee_reporting()
        return worked

    def _drain_employee_reporting_once(self) -> bool:
        """Repair and deliver durable views without sharing the execution lane."""

        worked = False
        now = _composition_module().time.monotonic()
        warning_deferred = False
        if now >= getattr(self, "_warning_reporting_not_before", 0.0):
            try:
                worked = self._drain_main_bot_warning_outbox_once()
            except MainBotWarningRetryableDeliveryError as exc:
                warning_deferred = True
                failures = getattr(self, "_warning_reporting_failures", 0) + 1
                self._warning_reporting_failures = failures
                self._warning_reporting_not_before = now + min(
                    0.5 * (2 ** min(failures - 1, 4)),
                    5.0,
                )
                if failures == 1 or now >= getattr(
                    self, "_warning_reporting_next_log_at", 0.0
                ):
                    logger.error(
                        "main Bot warning reporting deferred; retrying with backoff: %s",
                        type(exc).__name__,
                    )
                    self._warning_reporting_next_log_at = now + 30.0
            else:
                if getattr(self, "_warning_reporting_failures", 0):
                    logger.info("main Bot warning reporting recovered")
                self._warning_reporting_failures = 0
                self._warning_reporting_not_before = 0.0
                self._warning_reporting_next_log_at = 0.0
        fatal_repair_error: Exception | None = None
        repair_deferred = bool(getattr(self, "_dispatch_recovery_pending", False))
        if getattr(self, "_dispatch_recovery_pending", False) and now >= getattr(
            self, "_reporting_repair_not_before", 0.0
        ):
            try:
                worked = self._repair_employee_dispatch_reporting() or worked
            except _composition_module().EmployeeDispatchReportingDeferredError as exc:
                failures = getattr(self, "_reporting_repair_failures", 0) + 1
                self._reporting_repair_failures = failures
                self._reporting_repair_not_before = now + min(
                    0.1 * (2 ** min(failures - 1, 6)),
                    5.0,
                )
                logger.error(
                    "employee reporting repair failed closed: %s",
                    type(exc).__name__,
                )
                repair_deferred = True
            except Exception as exc:
                # Preserve the independent terminal-ingress convergence pass
                # even when dispatch repair exposes a programming or integrity
                # fault.  The fault is still fatal for this tick and is
                # re-raised immediately after that pass; it must never be
                # converted into retry/backoff or permit GC.
                fatal_repair_error = exc
                repair_deferred = True
            else:
                self._reporting_repair_failures = 0
                self._reporting_repair_not_before = 0.0
                repair_deferred = bool(getattr(self, "_dispatch_recovery_pending", False))

        # A bad dispatch-attempt repair is independent from already-terminal
        # Router records.  Always reconcile those records so one poison attempt
        # cannot head-of-line block a healthy failure response.  Projection and
        # anchor truth errors escape this tick; reconcile isolates only explicit
        # record-local payload conflicts.
        worked = self._reconcile_terminal_ingress() > 0 or worked
        if fatal_repair_error is not None:
            raise fatal_repair_error
        # A retryable encrypted payload can postpone creation of a retirement
        # response until after startup recovery.  Re-evaluate the strict frozen
        # lease after every reconciliation and before delivery, so that late
        # response can restore outbound-only transport without reopening ingress.
        recovered_delivery_channels = self._recover_retirement_delivery_channels()
        worked = bool(recovered_delivery_channels) or worked
        # Already-anchored snapshots remain safe to deliver even while one
        # attempt repair is deferred.  Delivery isolates typed record-local
        # transport faults; integrity failures escape and fence this tick.
        worked = self._drain_employee_outbox_once() or worked
        if warning_deferred or repair_deferred:
            return worked

        gc_now = _composition_module().time.monotonic()
        if gc_now < getattr(self, "_employee_dispatch_next_gc_at", 0.0):
            return worked
        if self._outbox is not None:
            worked = self._outbox.gc_superseded_snapshots() > 0 or worked
        if self._ingress is not None:
            worked = self._ingress.gc_terminal_payloads() > 0 or worked
        # Advance only after both GC components complete successfully.  Either
        # failure remains immediately retryable on the next reporting tick.
        self._employee_dispatch_next_gc_at = gc_now + 60.0
        return worked

    def _repair_employee_dispatch_reporting(self) -> bool:
        """Boundedly repair committed attempts and their missing Outbox views."""

        with self._reporting_repair_lock:
            return self._repair_employee_dispatch_reporting_unlocked()

    def _repair_employee_dispatch_reporting_unlocked(self) -> bool:
        """Repair reporting after the caller acquires the single-flight lock."""

        if not getattr(self, "_dispatch_recovery_pending", False):
            return False
        dispatch = self._dispatch
        if dispatch is None:
            raise RuntimeError("employee dispatch recovery is unavailable")

        repair_reporting = getattr(dispatch, "repair_reporting", None)
        if callable(repair_reporting):
            result = repair_reporting()
            if not isinstance(result, EmployeeDispatchReportingRecoveryResult):
                raise TypeError("employee dispatch reporting result is invalid")
            if result.deferred_attempt_ids:
                self._dispatch_recovery_pending = True
                raise _composition_module().EmployeeDispatchReportingDeferredError(
                    "employee dispatch reporting repair remains deferred",
                    failed_attempt_ids=result.deferred_attempt_ids,
                    repaired_count=result.recovered_count,
                )
            self._dispatch_recovery_pending = False
            return result.made_progress

        # Compatibility path for narrow test/integration dispatch ports.  The
        # production coordinator exposes the atomic batch result above.
        recovered_count = 0
        recovery_error: Exception | None = None
        try:
            recovered = dispatch.recover_incomplete_attempts()
            recovered_count += len(recovered) if isinstance(recovered, tuple) else 0
        except Exception as exc:
            # A finalize call may have committed the terminal fact before its
            # lifecycle append raised.  Snapshot reconciliation must still run
            # in this same repair attempt.
            recovery_error = exc
            if isinstance(exc, _composition_module().EmployeeDispatchReportingDeferredError):
                recovered_count += exc.repaired_count

        snapshot_error: Exception | None = None
        try:
            reconciled = dispatch.reconcile_terminal_snapshots()
            if type(reconciled) is int and reconciled > 0:
                recovered_count += reconciled
        except Exception as exc:
            snapshot_error = exc
            if isinstance(exc, _composition_module().EmployeeDispatchReportingDeferredError):
                recovered_count += exc.repaired_count

        if isinstance(recovery_error, _composition_module().EmployeeDispatchReportingDeferredError) and snapshot_error is None:
            # One bounded second pass distinguishes a pre-commit transient from
            # the expected "terminal committed, Outbox append failed" window.
            try:
                recovered = dispatch.recover_incomplete_attempts()
                recovered_count += len(recovered) if isinstance(recovered, tuple) else 0
            except Exception as exc:
                recovery_error = exc
                if isinstance(exc, _composition_module().EmployeeDispatchReportingDeferredError):
                    recovered_count += exc.repaired_count
            else:
                recovery_error = None

        if recovery_error is not None or snapshot_error is not None:
            self._dispatch_recovery_pending = True
            errors = tuple(error for error in (recovery_error, snapshot_error) if error is not None)
            fatal = next(
                (
                    error
                    for error in errors
                    if not isinstance(
                        error,
                        _composition_module().EmployeeDispatchReportingDeferredError,
                    )
                ),
                None,
            )
            if fatal is not None:
                raise RuntimeError("employee dispatch reporting recovery failed") from fatal
            raise errors[0]
        self._dispatch_recovery_pending = False
        return recovered_count > 0

    @staticmethod
    def _unknown_employee_ingress_trust() -> EffectiveTrust:
        return EffectiveTrust(
            zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
            actor=ActorKind.UNKNOWN,
            managed_group=None,
            group_revision=None,
            grant_revision=None,
        )

    def trusted_employee_bot_open_ids(
        self,
        *,
        tenant_key: str | None = None,
        chat_id: str | None = None,
    ) -> frozenset[str]:
        """Return only current READY employee Bot Open IDs.

        Workforce ``bot_principal_id`` values are internal identifiers and are
        deliberately never compared with Feishu ``open_id`` values.
        """

        return frozenset(
            self._ready_employee_ingress_targets(
                tenant_key=tenant_key,
                chat_id=chat_id,
            )
        )

    def resolve_ready_employee_bot_target(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        bot_open_id: str,
    ) -> ReadyEmployeeIngressTarget | None:
        """Freeze one unambiguous READY employee transport binding."""

        if not isinstance(bot_open_id, str) or not bot_open_id.startswith("ou_") or len(bot_open_id) > 256:
            return None
        targets, ambiguous = self._ready_employee_ingress_target_snapshot(
            tenant_key=tenant_key,
            chat_id=chat_id,
        )
        if bot_open_id in ambiguous:
            raise _composition_module().EmployeeTargetResolutionUnknownError("employee target binding is ambiguous")
        return targets.get(bot_open_id)

    def _ready_employee_ingress_targets(
        self,
        *,
        tenant_key: str | None = None,
        chat_id: str | None = None,
    ) -> dict[str, ReadyEmployeeIngressTarget]:
        targets, _ambiguous = self._ready_employee_ingress_target_snapshot(
            tenant_key=tenant_key,
            chat_id=chat_id,
        )
        return targets

    def _ready_employee_ingress_target_snapshot(
        self,
        *,
        tenant_key: str | None = None,
        chat_id: str | None = None,
    ) -> tuple[dict[str, ReadyEmployeeIngressTarget], frozenset[str]]:
        scoped = tenant_key is not None or chat_id is not None
        if scoped and (
            not isinstance(tenant_key, str) or not tenant_key or not isinstance(chat_id, str) or not chat_id
        ):
            return {}, frozenset()
        service = self._service
        channels = self._channels
        if service is None or channels is None:
            raise _composition_module().EmployeeTargetResolutionUnknownError("employee target dependencies are unavailable")
        snapshot = getattr(service, "current_employee_transport_snapshot", None)
        if not callable(snapshot):
            raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot API is unavailable")
        try:
            employees, principals, durable_states = snapshot()
        except (OSError, TimeoutError) as exc:
            raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot is unavailable") from exc
        except (TypeError, ValueError) as exc:
            raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot structure is invalid") from exc
        if not all(isinstance(values, tuple) for values in (employees, principals, durable_states)):
            raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot structure is invalid")
        principal_by_id: dict[str, Any] = {}
        for principal in principals:
            bot_principal_id = getattr(principal, "bot_principal_id", None)
            if (
                not isinstance(bot_principal_id, str)
                or not isinstance(getattr(principal, "tenant_key", None), str)
                or not isinstance(getattr(principal, "agent_id", None), str)
                or not isinstance(getattr(principal, "app_id", None), str)
                or not isinstance(getattr(principal, "credential_ref", None), str)
            ):
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot structure is invalid")
            if bot_principal_id in principal_by_id:
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee target principal binding is ambiguous")
            principal_by_id[bot_principal_id] = principal
        active_states: dict[str, Any] = {}
        ambiguous_states: set[str] = set()
        for state in durable_states:
            phase = getattr(getattr(state, "phase", None), "value", None)
            if not isinstance(phase, str):
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot structure is invalid")
            if phase != "active":
                continue
            agent_id = getattr(state, "agent_id", None)
            if not isinstance(agent_id, str):
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot structure is invalid")
            if agent_id in active_states:
                ambiguous_states.add(agent_id)
            else:
                active_states[agent_id] = state
        result: dict[str, ReadyEmployeeIngressTarget] = {}
        ambiguous: set[str] = set()
        seen_employee_ids: set[str] = set()
        for employee in employees:
            employee_state = getattr(employee, "state", None)
            worker_type = getattr(employee, "worker_type", None)
            employee_agent_id = getattr(employee, "agent_id", None)
            employee_tenant_key = getattr(employee, "tenant_key", None)
            employee_bot_principal_id = getattr(
                employee,
                "bot_principal_id",
                None,
            )
            member_groups = getattr(employee, "member_groups", None)
            if (
                not isinstance(employee_state, EmployeeState)
                or not isinstance(worker_type, WorkerType)
                or not isinstance(employee_agent_id, str)
                or not isinstance(employee_tenant_key, str)
                or (employee_bot_principal_id is not None and not isinstance(employee_bot_principal_id, str))
                or not isinstance(member_groups, tuple)
            ):
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee target snapshot structure is invalid")
            if employee_agent_id in seen_employee_ids:
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee target employee binding is ambiguous")
            seen_employee_ids.add(employee_agent_id)
            if (
                employee_state is not EmployeeState.ACTIVE
                or worker_type is not WorkerType.VISIBLE
                or (scoped and (employee_tenant_key != tenant_key or chat_id not in member_groups))
            ):
                continue
            principal = principal_by_id.get(employee_bot_principal_id)
            try:
                status = channels.status(employee_agent_id)
            except (OSError, TimeoutError) as exc:
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee Channel status is unavailable") from exc
            durable = active_states.get(employee_agent_id)
            if status is not None and not isinstance(
                getattr(status, "state", None),
                _composition_module().ChannelProcessState,
            ):
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee Channel status structure is invalid")
            identity = getattr(status, "identity", None)
            ready_metadata = getattr(status, "ready_metadata", None)
            open_id = identity.get("open_id") if isinstance(identity, Mapping) else None
            generation = getattr(status, "generation", None)
            connection_id = ready_metadata.get("connection_id") if isinstance(ready_metadata, Mapping) else None
            if getattr(status, "state", None) is not _composition_module().ChannelProcessState.READY:
                continue
            if (
                not isinstance(identity, Mapping)
                or not isinstance(ready_metadata, Mapping)
                or not isinstance(open_id, str)
                or not open_id.startswith("ou_")
                or type(generation) is not int
                or generation <= 0
                or not isinstance(connection_id, str)
                or not connection_id.startswith("conn_")
                or not isinstance(getattr(status, "agent_id", None), str)
                or not isinstance(getattr(status, "tenant_key", None), str)
                or not isinstance(getattr(status, "bot_principal_id", None), str)
                or not isinstance(getattr(status, "app_id", None), str)
                or not isinstance(identity.get("app_id"), str)
            ):
                raise _composition_module().EmployeeTargetResolutionUnknownError("employee READY Channel structure is invalid")
            if employee_agent_id in ambiguous_states:
                ambiguous.add(open_id)
                continue
            binding_matches = (
                principal is not None
                and durable is not None
                and principal.tenant_key == employee_tenant_key
                and principal.agent_id == employee_agent_id
                and isinstance(principal.credential_ref, str)
                and bool(principal.credential_ref)
                and getattr(status, "agent_id", None) == employee_agent_id
                and getattr(status, "tenant_key", None) == employee_tenant_key
                and getattr(status, "bot_principal_id", None) == employee_bot_principal_id
                and getattr(status, "app_id", None) == principal.app_id
                and type(generation) is int
                and generation > 0
                and isinstance(connection_id, str)
                and connection_id.startswith("conn_")
                and isinstance(identity, Mapping)
                and identity.get("app_id") == principal.app_id
                and isinstance(open_id, str)
                and open_id.startswith("ou_")
                and getattr(durable, "tenant_key", None) == employee_tenant_key
                and getattr(durable, "agent_id", None) == employee_agent_id
                and getattr(durable, "bot_principal_id", None) == employee_bot_principal_id
                and getattr(durable, "app_id", None) == principal.app_id
                and getattr(durable, "channel_generation", None) == generation
                and getattr(durable, "channel_connection_id", None) == connection_id
                and getattr(durable, "channel_identity_app_id", None) == principal.app_id
            )
            if not binding_matches:
                # A live READY process can already have consumed this origin.
                # Snapshot drift therefore cannot be treated as a proven
                # no-target result even when its frozen binding is unusable.
                ambiguous.add(open_id)
                continue
            target = _composition_module().ReadyEmployeeIngressTarget(
                tenant_key=employee_tenant_key,
                chat_id=chat_id or "",
                agent_id=employee_agent_id,
                bot_principal_id=employee_bot_principal_id,
                app_id=principal.app_id,
                bot_open_id=open_id,
                channel_generation=generation,
                connection_id=connection_id,
            )
            existing = result.get(open_id)
            if existing is not None and existing != target:
                ambiguous.add(open_id)
            else:
                result[open_id] = target
        for open_id in ambiguous:
            result.pop(open_id, None)
        return result, frozenset(ambiguous)

    def wait_for_employee_message_acceptance(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        channel_generation: int,
        connection_id: str,
        chat_id: str,
        message_id: str,
        timeout: float,
    ) -> bool:
        """Prove that one employee transport anchored this exact message."""

        ingress = self._ingress
        if ingress is None:
            return False
        values = (tenant_key, agent_id, bot_principal_id, app_id, chat_id, message_id)
        safe_identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
        if any(
            not isinstance(value, str) or not value or len(value) > 256 or safe_identifier.fullmatch(value) is None
            for value in values
        ) or (
            not agent_id.startswith("agt_")
            or not bot_principal_id.startswith("bot_")
            or not app_id.startswith("cli_")
            or not chat_id.startswith("oc_")
            or not message_id.startswith("om_")
        ):
            return False
        indexed_chat_id = "oc_" + hashlib.sha256(chat_id.encode("utf-8")).hexdigest()
        indexed_message_id = "om_" + hashlib.sha256(message_id.encode("utf-8")).hexdigest()
        acceptance = ingress.wait_for_anchored_message_acceptance(
            tenant_key=tenant_key,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            event_type="im.message.receive_v1",
            chat_id=indexed_chat_id,
            message_id=indexed_message_id,
            channel_generation=channel_generation,
            connection_id=connection_id,
            timeout=timeout,
        )
        return (
            getattr(acceptance, "status", None) == "accepted"
            and getattr(acceptance, "acceptance", None) is not None
            and getattr(acceptance, "channel_generation", None) == channel_generation
            and getattr(acceptance, "connection_id", None) == connection_id
        )

    def _employee_handoff_finalization_state(
        self,
    ) -> tuple[
        threading.Lock,
        dict[tuple[str, ...], concurrent.futures.Future[Any]],
    ]:
        """Return the runtime-owned singleflight state; support narrow doubles."""

        lock = getattr(self, "_employee_handoff_finalization_lock", None)
        if lock is None:
            lock = _composition_module().threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._employee_handoff_finalization_lock = lock
        futures = getattr(self, "_employee_handoff_finalization_futures", None)
        if futures is None:
            futures = {}
            self._employee_handoff_finalization_futures = futures
        return lock, futures

    def _submit_employee_handoff_finalization(
        self,
        key: tuple[str, ...],
        action: Callable[[], Any],
    ) -> concurrent.futures.Future[Any] | None:
        """Submit one durable finalizer without tying it to the public deadline."""

        if not key or not all(isinstance(value, str) and value for value in key):
            raise ValueError("employee handoff finalization key is invalid")
        if not callable(action):
            raise TypeError("employee handoff finalization action must be callable")
        lock, futures = self._employee_handoff_finalization_state()
        with lock:
            existing = futures.get(key)
            if existing is not None and not existing.done():
                return existing
            if existing is not None:
                futures.pop(key, None)
            if getattr(self, "_closing", False) or getattr(
                self,
                "_employee_handoff_finalization_closed",
                False,
            ):
                return None
            for completed_key, completed in tuple(futures.items()):
                if completed.done():
                    futures.pop(completed_key, None)
            if len(futures) >= _composition_module()._EMPLOYEE_HANDOFF_FINALIZATION_MAX_PENDING:
                return None
            executor = getattr(
                self,
                "_employee_handoff_finalization_executor",
                None,
            )
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_composition_module()._EMPLOYEE_HANDOFF_FINALIZATION_WORKERS,
                    thread_name_prefix="employee-handoff-finalization",
                )
                self._employee_handoff_finalization_executor = executor
            future = executor.submit(action)
            futures[key] = future

        def completed(done: concurrent.futures.Future[Any]) -> None:
            try:
                done.result()
            except concurrent.futures.CancelledError:
                logger.error("employee handoff durable finalization was canceled")
            except Exception:
                logger.error(
                    "employee handoff durable finalization failed",
                    exc_info=True,
                )
            finally:
                with lock:
                    if futures.get(key) is done:
                        futures.pop(key, None)

        future.add_done_callback(completed)
        return future

    @staticmethod
    def _await_employee_handoff_finalization(
        future: concurrent.futures.Future[Any] | None,
        *,
        deadline: float,
    ) -> Any:
        """Observe a finalizer only within the caller's wall-clock budget."""

        if future is None:
            raise _composition_module().EmployeeMessageHandoffUnknownError(
                "employee handoff durable finalization is unavailable"
            )
        remaining = max(0.0, deadline - _composition_module().time.monotonic())
        try:
            result = future.result(timeout=remaining)
            if _composition_module().time.monotonic() > deadline:
                raise _composition_module().EmployeeMessageHandoffUnknownError(
                    "employee handoff durable finalization completed after the public deadline"
                )
            return result
        except concurrent.futures.TimeoutError as exc:
            if not future.done():
                raise _composition_module().EmployeeMessageHandoffUnknownError(
                    "employee handoff durable finalization is still pending"
                ) from exc
            raise _composition_module().EmployeeMessageHandoffUnknownError(
                "employee handoff durable finalization timed out"
            ) from exc
        except AssertionError:
            raise
        except _composition_module().EmployeeMessageHandoffUnknownError:
            raise
        except Exception as exc:
            raise _composition_module().EmployeeMessageHandoffUnknownError(
                "employee handoff durable finalization failed"
            ) from exc

    def wait_for_employee_message_handoff(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        channel_generation: int,
        connection_id: str,
        chat_id: str,
        message_id: str,
        timeout: float,
    ) -> bool:
        """Wait until exact ingress has entered an employee-owned durable lane."""

        if (
            type(channel_generation) is not int
            or channel_generation <= 0
            or not isinstance(connection_id, str)
            or not connection_id.startswith("conn_")
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
            or timeout > 30
        ):
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff coordinates or timeout are invalid")
        ingress = self._ingress
        router = self._router
        if ingress is None or router is None:
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff runtime is unavailable")
        values = (tenant_key, agent_id, bot_principal_id, app_id, chat_id, message_id)
        safe_identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
        if any(
            not isinstance(value, str) or not value or len(value) > 256 or safe_identifier.fullmatch(value) is None
            for value in values
        ):
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff coordinates are invalid")
        indexed_chat_id = "oc_" + hashlib.sha256(chat_id.encode("utf-8")).hexdigest()
        indexed_message_id = "om_" + hashlib.sha256(message_id.encode("utf-8")).hexdigest()
        deadline = _composition_module().time.monotonic() + float(timeout)
        remaining = max(0.0, deadline - _composition_module().time.monotonic())
        finalization_reserve = min(
            _composition_module()._EMPLOYEE_HANDOFF_FINALIZATION_RESERVE_SECONDS,
            remaining / 2.0,
        )
        try:
            outcome = ingress.wait_for_anchored_message_acceptance(
                tenant_key=tenant_key,
                agent_id=agent_id,
                bot_principal_id=bot_principal_id,
                app_id=app_id,
                event_type="im.message.receive_v1",
                chat_id=indexed_chat_id,
                message_id=indexed_message_id,
                channel_generation=channel_generation,
                connection_id=connection_id,
                timeout=max(0.0, remaining - finalization_reserve),
            )
        except AssertionError:
            raise
        except Exception as exc:
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee acceptance outcome is unavailable") from exc
        status = getattr(outcome, "status", None)
        if status == "denied":
            return False
        acceptance = getattr(outcome, "acceptance", None)
        if (
            status != "accepted"
            or acceptance is None
            or getattr(outcome, "channel_generation", None) != channel_generation
            or getattr(outcome, "connection_id", None) != connection_id
        ):
            denial = self._submit_employee_handoff_finalization(
                (
                    "deny",
                    tenant_key,
                    agent_id,
                    bot_principal_id,
                    app_id,
                    indexed_chat_id,
                    indexed_message_id,
                    str(channel_generation),
                    connection_id,
                ),
                lambda: self._deny_unwitnessed_employee_handoff(
                    ingress=ingress,
                    tenant_key=tenant_key,
                    agent_id=agent_id,
                    bot_principal_id=bot_principal_id,
                    app_id=app_id,
                    indexed_chat_id=indexed_chat_id,
                    indexed_message_id=indexed_message_id,
                    channel_generation=channel_generation,
                    connection_id=connection_id,
                    deadline=None,
                ),
            )
            outcome = self._await_employee_handoff_finalization(
                denial,
                deadline=deadline,
            )
            if getattr(outcome, "status", None) == "denied":
                return False
            acceptance = getattr(outcome, "acceptance", None)
            if (
                getattr(outcome, "status", None) != "accepted"
                or acceptance is None
                or getattr(outcome, "channel_generation", None) != channel_generation
                or getattr(outcome, "connection_id", None) != connection_id
            ):
                raise _composition_module().EmployeeMessageHandoffUnknownError("employee acceptance winner lacks an exact transport witness")
        progress, revision_lock = self._employee_handoff_wait_state()
        first_observation = True
        while True:
            if getattr(self, "_closing", False):
                raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff runtime is closing")
            observation_remaining = deadline - finalization_reserve - _composition_module().time.monotonic()
            immediate_first_observation = first_observation and observation_remaining <= 0
            if immediate_first_observation:
                acquired_revision = False
                observed_revision = self._employee_handoff_revision
                observed_progress = progress
            elif observation_remaining <= 0:
                break
            else:
                acquired_revision = revision_lock.acquire(timeout=min(observation_remaining, _composition_module().threading.TIMEOUT_MAX))
            if not immediate_first_observation and not acquired_revision:
                break
            if acquired_revision:
                try:
                    observed_revision = self._employee_handoff_revision
                    observed_progress = self._employee_handoff_event
                    if getattr(self, "_closing", False):
                        raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff runtime is closing")
                finally:
                    revision_lock.release()
            first_observation = False
            try:
                result = self._employee_handoff_projection_result(
                    acceptance_id=acceptance.acceptance_id,
                    ingress=ingress,
                    router=router,
                    channel_generation=channel_generation,
                    connection_id=connection_id,
                    deadline=deadline,
                    allow_immediate=observation_remaining <= 0,
                )
            except AssertionError:
                raise
            except Exception as exc:
                raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff projection is unavailable") from exc
            if result is _composition_module()._EmployeeHandoffProjection.OWNED:
                return True
            if result is _composition_module()._EmployeeHandoffProjection.DURABLY_DENIED:
                return False
            if result is _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN:
                raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff ownership is unproven")
            if immediate_first_observation:
                break
            remaining = deadline - finalization_reserve - _composition_module().time.monotonic()
            if remaining <= 0 or not revision_lock.acquire(timeout=min(remaining, _composition_module().threading.TIMEOUT_MAX)):
                break
            try:
                if getattr(self, "_closing", False):
                    raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff runtime is closing")
                remaining = deadline - finalization_reserve - _composition_module().time.monotonic()
                if remaining <= 0:
                    break
                if self._employee_handoff_revision != observed_revision:
                    continue
            finally:
                revision_lock.release()
            # Event.wait() does not reacquire the revision lock when its
            # timeout expires, so a contended notifier cannot extend the
            # caller's hard handoff deadline.  The revision recheck above and
            # the clear-under-lock protocol prevent a transition between the
            # projection read and registration from being lost: each
            # notification seals the observed epoch and installs a fresh one.
            observed_progress.wait(remaining)
        abandonment = self._submit_employee_handoff_finalization(
            (
                "abandon",
                acceptance.acceptance_id,
                str(channel_generation),
                connection_id,
            ),
            lambda: self._abandon_unconfirmed_employee_handoff(
                acceptance_id=acceptance.acceptance_id,
                ingress=ingress,
                router=router,
                channel_generation=channel_generation,
                connection_id=connection_id,
                deadline=None,
            ),
        )
        return self._await_employee_handoff_finalization(
            abandonment,
            deadline=deadline,
        )

    @staticmethod
    def _deny_unwitnessed_employee_handoff(
        *,
        ingress: object,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        indexed_chat_id: str,
        indexed_message_id: str,
        channel_generation: int,
        connection_id: str,
        deadline: float | None,
    ) -> object:
        """Fence a timed-out logical message or recover its exact winning alias."""

        deny = getattr(ingress, "deny_message_acceptance", None)
        wait = getattr(ingress, "wait_for_anchored_message_acceptance", None)
        if not callable(deny) or not callable(wait):
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee acceptance denial is unavailable")
        coordinates = {
            "tenant_key": tenant_key,
            "agent_id": agent_id,
            "bot_principal_id": bot_principal_id,
            "app_id": app_id,
            "event_type": "im.message.receive_v1",
            "chat_id": indexed_chat_id,
            "message_id": indexed_message_id,
            "channel_generation": channel_generation,
            "connection_id": connection_id,
        }
        try:
            outcome = deny(**coordinates, deadline=deadline)
        except AssertionError:
            raise
        except Exception as exc:
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee acceptance denial was not durably anchored") from exc
        status = getattr(outcome, "status", None)
        if status == "denied":
            return outcome
        if status != "accepted" or getattr(outcome, "acceptance", None) is None:
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee acceptance denial returned an invalid outcome")
        try:
            remaining = (
                30.0
                if deadline is None
                else max(0.0, deadline - _composition_module().time.monotonic())
            )
            exact = wait(**coordinates, timeout=remaining)
        except AssertionError:
            raise
        except Exception as exc:
            raise _composition_module().EmployeeMessageHandoffUnknownError(
                "employee acceptance race lacks an exact transport witness"
            ) from exc
        canonical = getattr(outcome, "acceptance", None)
        if getattr(exact, "status", None) != "accepted" or getattr(exact, "acceptance", None) != canonical:
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee acceptance race lacks an exact transport witness")
        return exact

    def _employee_handoff_wait_state(
        self,
    ) -> tuple[threading.Event, threading.Lock]:
        """Return lock-free wait state; lazy creation supports narrow doubles."""

        progress = getattr(self, "_employee_handoff_event", None)
        revision_lock = getattr(self, "_employee_handoff_revision_lock", None)
        if progress is None or revision_lock is None:
            progress = _composition_module().threading.Event()
            revision_lock = _composition_module().threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
            self._employee_handoff_event = progress
            self._employee_handoff_revision_lock = revision_lock
            self._employee_handoff_revision = 0
        return progress, revision_lock

    def _notify_employee_handoff_progress(self) -> None:
        """Wake handoff waiters after one serialized Employee drain attempt."""

        progress, revision_lock = self._employee_handoff_wait_state()
        with revision_lock:
            self._employee_handoff_revision += 1
            self._employee_handoff_event = _composition_module().threading.Event()
            progress.set()

    @staticmethod
    def _employee_handoff_record_snapshot(
        owner: object,
        acceptance_id: str,
        *,
        deadline: float | None = None,
        allow_immediate: bool = False,
    ) -> object | None:
        snapshot = getattr(owner, "record_snapshot", None)
        if callable(snapshot):
            if isinstance(
                owner,
                (_composition_module().EmployeeIngressService, _composition_module().DurableEmployeeIngressRouter),
            ):
                return snapshot(
                    acceptance_id,
                    deadline=deadline,
                    allow_immediate=allow_immediate,
                )
            return snapshot(acceptance_id)
        state = getattr(owner, "state", None)
        records = getattr(state, "by_acceptance_id", None)
        return records.get(acceptance_id) if isinstance(records, Mapping) else None

    def _employee_handoff_projection_result(
        self,
        *,
        acceptance_id: str,
        ingress: object,
        router: object,
        channel_generation: int,
        connection_id: str,
        deadline: float | None = None,
        allow_immediate: bool = False,
    ) -> _EmployeeHandoffProjection:
        """Classify durable handoff state without collapsing invalid evidence."""

        # The Inbox lookup already proved the exact request-side transport
        # alias.  Inbox and Router retain canonical coordinates, which may be an
        # older alias; their mutual consistency is authoritative at this layer.
        del channel_generation, connection_id

        record = self._employee_handoff_record_snapshot(
            ingress,
            acceptance_id,
            deadline=deadline,
            allow_immediate=allow_immediate,
        )
        if record is None:
            return _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
        disposition = getattr(record, "disposition", None)
        metadata = getattr(record, "metadata", None)
        if not self._employee_handoff_records_match(
            acceptance_id,
            record,
            None,
        ):
            return _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
        if disposition is not None:
            if (
                not isinstance(metadata, _composition_module().EmployeeIngressMetadata)
                or getattr(disposition, "acceptance_id", None) != acceptance_id
            ):
                return _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
            if self._employee_handoff_has_response(disposition.reason_code):
                return _composition_module()._EmployeeHandoffProjection.OWNED

        routed = self._employee_handoff_record_snapshot(
            router,
            acceptance_id,
            deadline=deadline,
            allow_immediate=allow_immediate,
        )
        if routed is None:
            return (
                _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
                if disposition is not None
                else _composition_module()._EmployeeHandoffProjection.PENDING
            )
        if not self._employee_handoff_records_match(
            acceptance_id,
            record,
            routed,
        ):
            return _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
        state = getattr(routed, "state", None)
        queued_sequence = getattr(routed, "queued_sequence", 0)
        if type(queued_sequence) is not int or queued_sequence < 0:
            return _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
        event_type = getattr(routed, "event_type", None)
        if event_type != "im.message.receive_v1":
            return _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
        if state in {"queued", "dispatching"}:
            # Validate every message rather than trusting the targeted marker.
            # A corrupted scope/kind/digest must not erase that marker and fall
            # through to generic ownership.
            return (
                _composition_module()._EmployeeHandoffProjection.OWNED
                if queued_sequence > 0
                and self._validated_employee_terminal_ownership(
                    acceptance_id,
                    record,
                    routed,
                    deadline=deadline,
                )
                is not None
                else _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
            )
        if state == "terminal":
            reason_code = getattr(routed, "reason_code", None)
            if queued_sequence > 0:
                return (
                    _composition_module()._EmployeeHandoffProjection.OWNED
                    if self._validated_employee_terminal_ownership(
                        acceptance_id,
                        record,
                        routed,
                        deadline=deadline,
                    )
                    is not None
                    else _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
                )
            if disposition is not None:
                return (
                    _composition_module()._EmployeeHandoffProjection.DURABLY_DENIED
                    if getattr(disposition, "state", None) == "terminal"
                    and getattr(disposition, "reason_code", None) == reason_code
                    and reason_code
                    in {
                        "authority_denied",
                        "membership_degraded",
                        "handoff_unconfirmed",
                    }
                    else _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
                )
            if reason_code not in {
                "authority_denied",
                "membership_degraded",
                "handoff_unconfirmed",
            }:
                return _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
            authority = getattr(routed, "authority", None)
            if authority is None:
                return (
                    _composition_module()._EmployeeHandoffProjection.DURABLY_DENIED
                    if getattr(routed, "team_id", None) == metadata.chat_id
                    and getattr(routed, "requester_principal_id", None) == metadata.sender_principal_id
                    else _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
                )
            return (
                _composition_module()._EmployeeHandoffProjection.DURABLY_DENIED
                if self._validated_employee_terminal_ownership(
                    acceptance_id,
                    record,
                    routed,
                    deadline=deadline,
                )
                is not None
                else _composition_module()._EmployeeHandoffProjection.INVALID_UNKNOWN
            )
        # accepted/authorized/staging and their durable Inbox retries remain
        # cancelable before Router queue ownership is granted.
        return _composition_module()._EmployeeHandoffProjection.PENDING

    @staticmethod
    def _employee_handoff_records_match(
        acceptance_id: str,
        ingress_record: object,
        router_record: object | None,
    ) -> bool:
        """Validate immutable acceptance summaries before classifying ownership."""

        metadata = getattr(ingress_record, "metadata", None)
        acceptance = getattr(ingress_record, "acceptance", None)
        if not isinstance(metadata, _composition_module().EmployeeIngressMetadata) or acceptance is None:
            return False
        if (
            getattr(acceptance, "acceptance_id", None) != acceptance_id
            or getattr(acceptance, "envelope_id", None) != metadata.envelope_id
            or getattr(acceptance, "dedup_key", None) != metadata.dedup_key
            or getattr(acceptance, "semantic_digest", None) != metadata.semantic_digest
            or metadata.semantic_digest != metadata.payload_sha256
            or getattr(ingress_record, "aggregate_id", None) != metadata.dedup_key
        ):
            return False
        if router_record is None:
            return True
        return (
            getattr(router_record, "acceptance_id", None) == acceptance_id
            and getattr(router_record, "aggregate_id", None) == metadata.dedup_key
            and getattr(router_record, "envelope_id", None) == metadata.envelope_id
            and getattr(router_record, "tenant_key", None) == metadata.tenant_key
            and getattr(router_record, "agent_id", None) == metadata.agent_id
            and getattr(router_record, "bot_principal_id", None) == metadata.bot_principal_id
            and getattr(router_record, "app_id", None) == metadata.app_id
            and getattr(router_record, "channel_generation", None) == metadata.channel_generation
            and getattr(router_record, "connection_id", None) == metadata.connection_id
            and getattr(router_record, "message_id", None) == metadata.message_id
            and getattr(router_record, "event_type", None) == metadata.event_type
            and getattr(router_record, "accepted_sequence", None) == getattr(acceptance, "journal_sequence", None)
        )

    @staticmethod
    def _employee_targeted_authority_bound(record: object) -> bool:
        """Return whether frozen authority describes one targeted group task."""

        authority = getattr(record, "authority", None)
        return (
            getattr(authority, "authorization_scope", None) is EmployeeAuthorizationScope.MANAGED_GROUP
            and getattr(authority, "effective_input_kind", "") == TARGETED_TASK_INPUT_KIND
            and isinstance(
                getattr(authority, "effective_input_digest", None),
                str,
            )
            and len(getattr(authority, "effective_input_digest", "")) == 64
            and isinstance(
                getattr(authority, "target_bot_open_id_digest", None),
                str,
            )
            and len(getattr(authority, "target_bot_open_id_digest", "")) == 64
        )

    @classmethod
    def _employee_targeted_queue_owned(cls, record: object) -> bool:
        """Return whether the Employee durably accepted response ownership."""

        queued_sequence = getattr(record, "queued_sequence", 0)
        return type(queued_sequence) is int and queued_sequence > 0 and cls._employee_targeted_authority_bound(record)

    def _validated_employee_terminal_ownership(
        self,
        acceptance_id: str,
        ingress_record: object,
        router_record: object,
        *,
        deadline: float | None = None,
    ) -> _ValidatedTargetedOwnership | None:
        """Bind queued response ownership back to encrypted Inbox authority."""

        ingress = self._ingress
        metadata = getattr(ingress_record, "metadata", None)
        authority = getattr(router_record, "authority", None)
        if (
            ingress is None
            or not isinstance(metadata, _composition_module().EmployeeIngressMetadata)
            or authority is None
            or getattr(router_record, "event_type", None) != "im.message.receive_v1"
        ):
            return None
        frozen_coordinates = (
            metadata.tenant_key,
            metadata.agent_id,
            metadata.bot_principal_id,
            metadata.app_id,
            metadata.channel_generation,
            metadata.connection_id,
        )
        if frozen_coordinates != (
            getattr(router_record, "tenant_key", None),
            getattr(router_record, "agent_id", None),
            getattr(router_record, "bot_principal_id", None),
            getattr(router_record, "app_id", None),
            getattr(router_record, "channel_generation", None),
            getattr(router_record, "connection_id", None),
        ) or frozen_coordinates != (
            getattr(authority, "tenant_key", None),
            getattr(authority, "agent_id", None),
            getattr(authority, "bot_principal_id", None),
            getattr(authority, "app_id", None),
            getattr(authority, "channel_generation", None),
            getattr(authority, "connection_id", None),
        ):
            return None
        frozen_targeted = self._employee_targeted_authority_bound(router_record)
        response_obligation = getattr(router_record, "response_obligation", None)
        if (
            getattr(router_record, "queued_sequence", 0) > 0
            and frozen_targeted
            and isinstance(response_obligation, RouterResponseObligation)
        ):
            if (
                getattr(router_record, "indexed_chat_id", None) != metadata.chat_id
                or getattr(router_record, "indexed_thread_root_message_id", None) != metadata.thread_root_message_id
                or not response_obligation.is_bound_to(router_record, authority)
            ):
                return None
            reply_to = response_obligation.reply_to_message_id
            return _composition_module()._ValidatedTargetedOwnership(
                metadata=metadata,
                remote_chat_id=response_obligation.chat_id,
                remote_message_id=(reply_to if response_obligation.reply_coordinate_kind == "message" else ""),
                remote_root_id=(reply_to if response_obligation.reply_coordinate_kind == "root" else ""),
            )
        try:
            payload = ingress.get_payload(
                acceptance_id,
                **({} if deadline is None else {"deadline": deadline}),
            )
        except IngressBlobRetryableError:
            raise
        except IngressBlobError:
            return None
        part = (
            payload.normalized_parts[0]
            if isinstance(payload, _composition_module().EmployeeIngressPayload) and len(payload.normalized_parts) == 1
            else None
        )
        coordinates = _composition_module()._bound_remote_coordinates(metadata, part) if isinstance(part, Mapping) else None
        if coordinates is None:
            return None
        remote_chat_id, remote_message_id, remote_root_id = coordinates
        if remote_chat_id != getattr(authority, "team_id", None):
            return None

        mentions = part.get("mentions")
        mention = mentions[0] if isinstance(mentions, tuple) and len(mentions) == 1 else None
        target_bot_open_id = mention.get("open_id") if isinstance(mention, Mapping) else None
        payload_targeted = None
        if isinstance(target_bot_open_id, str) and target_bot_open_id.startswith("ou_"):
            targeted = parse_targeted_group_task(
                metadata=metadata,
                part=part,
                expected_bot_open_id=target_bot_open_id,
                expected_tenant_key=metadata.tenant_key,
            )
            if targeted.state is TargetedTaskState.TARGETED_VALID:
                payload_targeted = targeted
        frozen_targeted = self._employee_targeted_authority_bound(router_record)
        if payload_targeted is not None or frozen_targeted:
            if (
                payload_targeted is None
                or not frozen_targeted
                or hashlib.sha256(target_bot_open_id.encode("utf-8")).hexdigest()
                != getattr(authority, "target_bot_open_id_digest", "")
                or payload_targeted.input_digest != getattr(authority, "effective_input_digest", "")
            ):
                return None

        return _composition_module()._ValidatedTargetedOwnership(
            metadata=metadata,
            remote_chat_id=remote_chat_id,
            remote_message_id=remote_message_id,
            remote_root_id=remote_root_id,
        )

    def _abandon_unconfirmed_employee_handoff(
        self,
        *,
        acceptance_id: str,
        ingress: object,
        router: object,
        channel_generation: int,
        connection_id: str,
        deadline: float | None,
    ) -> bool:
        """Fence pending work before Main Bot reports an unconfirmed handoff."""

        abandon = getattr(router, "abandon_message_handoff", None)
        if not callable(abandon):
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee Router handoff denial is unavailable")
        try:
            abandoned = abandon(
                acceptance_id,
                channel_generation=channel_generation,
                connection_id=connection_id,
                deadline=deadline,
            )
        except AssertionError:
            raise
        except Exception as exc:
            logger.warning(
                "employee handoff cancellation failed closed",
                exc_info=True,
            )
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee Router handoff denial was not durably anchored") from exc
        if (
            getattr(abandoned, "state", None) == "terminal"
            and getattr(abandoned, "reason_code", None) == "handoff_unconfirmed"
        ):
            return False
        try:
            result = self._employee_handoff_projection_result(
                acceptance_id=acceptance_id,
                ingress=ingress,
                router=router,
                channel_generation=channel_generation,
                connection_id=connection_id,
                deadline=deadline,
            )
        except AssertionError:
            raise
        except Exception as exc:
            raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff ownership could not be rechecked") from exc
        if result is _composition_module()._EmployeeHandoffProjection.OWNED:
            return True
        if result is _composition_module()._EmployeeHandoffProjection.DURABLY_DENIED:
            return False
        raise _composition_module().EmployeeMessageHandoffUnknownError("employee handoff was neither owned nor durably denied")

    @staticmethod
    def _employee_handoff_has_response(reason_code: object) -> bool:
        """Recognize only dispositions preceded by an anchored employee reply."""

        if reason_code == "task_invalid_arguments":
            return True
        if reason_code in {
            "stop_no_active",
            "stop_forbidden",
            "stop_already_terminal",
            "stop_ambiguous",
            "stop_cancel_requested",
        }:
            return True
        if reason_code in {
            "status_completed",
            "status_unavailable",
            "status_invalid_arguments",
        }:
            return True
        return reason_code in {
            "history_completed",
            "history_denied",
            "history_failed",
            "memory_completed",
            "memory_denied",
            "memory_failed",
        }

    def _employee_ingress_transport_is_current(
        self,
        metadata: EmployeeIngressMetadata,
    ) -> bool:
        service = self._service
        channels = self._channels
        if service is None or channels is None:
            return False
        try:
            projection = service.synchronize_projection()
            employee = projection.employees.get(metadata.agent_id)
            if employee is None:
                return False
            principal = projection.bot_principals.get(employee.bot_principal_id)
            status = channels.status(metadata.agent_id)
            identity = getattr(status, "identity", None)
            ready_metadata = getattr(status, "ready_metadata", None)
            return (
                employee.state is EmployeeState.ACTIVE
                and employee.worker_type is WorkerType.VISIBLE
                and employee.tenant_key == metadata.tenant_key
                and employee.bot_principal_id == metadata.bot_principal_id
                and principal is not None
                and principal.tenant_key == metadata.tenant_key
                and principal.agent_id == metadata.agent_id
                and principal.app_id == metadata.app_id
                and bool(principal.credential_ref)
                and isinstance(identity, Mapping)
                and isinstance(ready_metadata, Mapping)
                and getattr(status, "state", None) is _composition_module().ChannelProcessState.READY
                and getattr(status, "tenant_key", None) == metadata.tenant_key
                and getattr(status, "agent_id", None) == metadata.agent_id
                and getattr(status, "bot_principal_id", None) == metadata.bot_principal_id
                and getattr(status, "app_id", None) == metadata.app_id
                and getattr(status, "generation", None) == metadata.channel_generation
                and identity.get("app_id") == metadata.app_id
                and ready_metadata.get("connection_id") == metadata.connection_id
            )
        except Exception:
            return False

    def _membership_event_transport_is_current(
        self,
        metadata: EmployeeIngressMetadata,
        remote_chat_id: str,
    ) -> bool:
        """Gate membership events without inferring identity rotation."""

        registry = self._managed_group_registry
        service = self._service
        channels = self._channels
        if (
            type(registry) is not _composition_module().ManagedGroupRegistry
            or service is None
            or channels is None
            or not isinstance(remote_chat_id, str)
        ):
            return False
        try:
            group, grant = registry.trust_snapshot(remote_chat_id)
            if group is None or grant is None:
                return False
            projection = service.synchronize_projection()
            employee = projection.employees.get(metadata.agent_id)
            if employee is None:
                return False
            principal = projection.bot_principals.get(employee.bot_principal_id)
            status = channels.status(metadata.agent_id)
            identity = getattr(status, "identity", None)
            ready_metadata = getattr(status, "ready_metadata", None)
            return (
                employee.state is EmployeeState.ACTIVE
                and employee.worker_type is WorkerType.VISIBLE
                and employee.tenant_key == metadata.tenant_key
                and employee.agent_id == metadata.agent_id
                and employee.bot_principal_id == metadata.bot_principal_id
                and principal is not None
                and principal.tenant_key == metadata.tenant_key
                and principal.agent_id == metadata.agent_id
                and principal.app_id == metadata.app_id
                and isinstance(identity, Mapping)
                and isinstance(ready_metadata, Mapping)
                and getattr(status, "state", None) is _composition_module().ChannelProcessState.READY
                and getattr(status, "tenant_key", None) == metadata.tenant_key
                and getattr(status, "agent_id", None) == metadata.agent_id
                and getattr(status, "bot_principal_id", None) == metadata.bot_principal_id
                and getattr(status, "app_id", None) == metadata.app_id
                and getattr(status, "generation", None) == metadata.channel_generation
                and identity.get("app_id") == metadata.app_id
                and ready_metadata.get("connection_id") == metadata.connection_id
            )
        except Exception:
            return False

    def _managed_employee_ingress_trust(
        self,
        record: Any,
        payload: EmployeeIngressPayload,
    ) -> EffectiveTrust | None:
        registry = self._managed_group_registry
        if type(registry) is not _composition_module().ManagedGroupRegistry:
            return None
        if len(payload.normalized_parts) != 1:
            return self._unknown_employee_ingress_trust()
        part = payload.normalized_parts[0]
        if not isinstance(part, Mapping):
            return self._unknown_employee_ingress_trust()
        coordinates = _composition_module()._bound_remote_coordinates(record.metadata, part)
        if coordinates is None:
            return self._unknown_employee_ingress_trust()
        chat_id, _message_id, _root_id = coordinates
        sender_id = part.get("sender_id")
        if not isinstance(sender_id, str) or not sender_id:
            return self._unknown_employee_ingress_trust()
        group, grant = registry.trust_snapshot(chat_id)
        return TrustZoneResolver(
            owner_id=self._managed_group_owner_id,
            managed_groups=(() if group is None else (group,)),
            project_grants=(() if grant is None else (grant,)),
            employee_bot_ids=self.trusted_employee_bot_open_ids(),
        ).resolve(
            sender_id=sender_id,
            chat_id=chat_id,
            chat_type=str(part.get("chat_type") or "group"),
        )

    def _reconcile_retired_activation_ingress(self) -> None:
        """Fail closed old prompt effects and prevent their ingress replay."""
        service = self._require_service()
        ingress = self._ingress
        if ingress is None:
            return
        retired_types = {
            "employee_activation_required_reply",
            "employee_status_reply",
        }
        retired_event_ids: set[str] = set()
        for snapshot in service.list_states():
            state = snapshot
            effect_types = dict(state.effect_types)
            for effect_id, effect_state in snapshot.effects:
                effect_type = effect_types.get(effect_id, "")
                if effect_type not in retired_types:
                    continue
                metadata = dict(state.metadata_for(effect_id))
                event_id = metadata.get("ingress_event_id", "")
                if event_id:
                    retired_event_ids.add(event_id)
                if effect_state not in {
                    HireEffectState.PREPARED,
                    HireEffectState.EXECUTING,
                }:
                    continue
                metadata.setdefault(
                    "error_code",
                    "manual_activation_retired_unknown_outcome",
                )
                state = service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type=effect_type,
                    next_state=HireEffectState.ACTION_REQUIRED,
                    metadata=metadata,
                )
        for acceptance_id, record in tuple(ingress.state.by_acceptance_id.items()):
            if record.disposition is not None or record.metadata.event_id not in retired_event_ids:
                continue
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="ignored",
                    reason_code="activation_retired",
                )
            except IngressConflictError:
                pass

    def _handle_control_ingress(
        self,
        acceptance_id: str,
        *,
        targeted_group_task: TargetedTaskParseResult | None = None,
        targeted_group_task_classified: bool = False,
    ) -> bool:
        """Consume durable membership, data, status, and stop controls."""
        ingress = self._ingress
        if ingress is None:
            return False
        ingress.rebuild_projection()
        record = ingress.state.by_acceptance_id.get(acceptance_id)
        if record is None:
            return False
        if record.disposition is not None:
            return record.disposition.reason_code.startswith(("stop_", "membership_", "history_", "memory_", "status_"))

        def finish(state: str, reason_code: str) -> bool:
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state=state,
                    reason_code=reason_code,
                )
            except IngressConflictError:
                pass
            return True

        try:
            payload = ingress.get_payload(acceptance_id)
        except Exception:
            return False
        first = payload.normalized_parts[0] if len(payload.normalized_parts) == 1 else None
        is_membership_event = isinstance(first, Mapping) and first.get("type") == "membership_event"
        if not isinstance(targeted_group_task_classified, bool):
            raise TypeError("targeted task classification flag must be bool")
        if (
            not is_membership_event
            and isinstance(first, Mapping)
            and isinstance(first.get("mentions"), tuple)
            and is_group_slash_observation(first)
            and (
                targeted_group_task
                if targeted_group_task_classified
                else self._authorized_targeted_group_task(record, payload)
            )
            is not None
        ):
            # Uniquely addressed group `/task` commands are routed or receive
            # usage by the group command gate, never by generic controls.
            return False
        owner_p2p_requester = None if is_membership_event else self._owner_p2p_requester(record, payload)
        texts: list[str] = []
        for part in payload.normalized_parts:
            content = part.get("content") if isinstance(part, Mapping) else None
            value = content.get("text") if isinstance(content, Mapping) else None
            if isinstance(value, str):
                texts.append(value.strip())
        status_match = re.fullmatch(r"/status(?:\s+(.+))?", texts[0], re.IGNORECASE) if len(texts) == 1 else None
        data_control = self._parse_data_control(texts)
        if (
            not is_membership_event
            and owner_p2p_requester is None
            and isinstance(first, Mapping)
            and first.get("chat_type") == "group"
            and (status_match is not None or data_control is not None or texts == ["/stop"])
        ):
            return False
        if is_membership_event:
            remote_chat_id = first.get("remote_chat_id")
            if not isinstance(remote_chat_id, str) or not self._membership_event_transport_is_current(
                record.metadata,
                remote_chat_id,
            ):
                return finish("ignored", "membership_unmanaged")
        else:
            try:
                trust = self._managed_employee_ingress_trust(record, payload)
            except Exception:
                trust = self._unknown_employee_ingress_trust()
            if owner_p2p_requester is None and trust is not None and trust.zone is not TrustZone.MANAGED_AGENT_GROUP:
                return finish("ignored", "authority_denied")
            if owner_p2p_requester is None and trust is not None and trust.actor is ActorKind.EMPLOYEE:
                return False
        if is_membership_event:
            if self._membership is None:
                return finish("terminal", "membership_unavailable")
            metadata = record.metadata
            remote_chat_id = first.get("remote_chat_id")
            operation = first.get("operation")
            expected_chat_index = (
                "oc_" + hashlib.sha256(remote_chat_id.encode()).hexdigest()
                if isinstance(remote_chat_id, str) and remote_chat_id
                else ""
            )
            if expected_chat_index != metadata.chat_id or operation not in {"added", "deleted"}:
                return finish("ignored", "membership_unmanaged")
            try:
                outcome = self._membership.reconcile_event(
                    tenant_key=metadata.tenant_key,
                    chat_id=remote_chat_id,
                    agent_id=metadata.agent_id,
                    app_id=metadata.app_id,
                    observed_is_member=operation == "added",
                )
            except MembershipBindingError:
                return finish("ignored", "membership_unmanaged")
            return finish(
                "terminal",
                f"membership_{outcome.state.value}",
            )
        if status_match is not None:
            # Every employee Bot observes ambient group slash commands.  A
            # group response here would therefore broadcast one card per
            # employee.  Only the union-resolved owner P2P lane is an
            # unambiguous employee control channel.
            if owner_p2p_requester is None:
                if isinstance(first, Mapping) and first.get("chat_type") == "group":
                    return False
                return finish("ignored", "authority_denied")
            return self._handle_status_control(
                acceptance_id=acceptance_id,
                record=record,
                payload=payload,
                has_arguments=status_match.group(1) is not None,
            )
        if data_control is not None:
            return self._handle_data_control(
                acceptance_id=acceptance_id,
                command=data_control[0],
                record=record,
                payload=payload,
                requester_principal_id=(owner_p2p_requester or record.metadata.sender_principal_id),
                history_days=data_control[1],
            )
        if texts != ["/stop"]:
            return False
        dispatch = self._dispatch
        lifecycle = self._outbox_lifecycle
        if dispatch is None or lifecycle is None:
            return False
        metadata = record.metadata
        coordinates = _composition_module()._bound_remote_coordinates(metadata, first)
        if coordinates is None:
            return finish("terminal", "stop_coordinates_invalid")
        remote_chat_id, remote_message_id, remote_root_id = coordinates
        outcome = dispatch.request_cancel(
            agent_id=metadata.agent_id,
            chat_id=remote_chat_id,
            requester_principal_id=(owner_p2p_requester or metadata.sender_principal_id),
            command_acceptance_id=acceptance_id,
        )
        lifecycle.command_response(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            chat_id=remote_chat_id,
            thread_root_message_id=remote_root_id or remote_message_id,
            command_acceptance_id=acceptance_id,
            status=outcome.status,
        )
        finish("terminal", f"stop_{outcome.status}")
        self._request_employee_outbox_delivery()
        return True

    def _handle_status_control(
        self,
        *,
        acceptance_id: str,
        record: Any,
        payload: EmployeeIngressPayload,
        has_arguments: bool,
    ) -> bool:
        """Publish an owner-scoped, Journal-backed employee status view."""

        ingress = self._ingress
        lifecycle = self._outbox_lifecycle
        if ingress is None or lifecycle is None:
            return False
        metadata = record.metadata
        first = payload.normalized_parts[0] if len(payload.normalized_parts) == 1 else None
        coordinates = _composition_module()._bound_remote_coordinates(metadata, first) if isinstance(first, Mapping) else None
        if coordinates is None:
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code="status_coordinates_invalid",
                )
            except IngressConflictError:
                pass
            return True
        remote_chat_id, remote_message_id, remote_root_id = coordinates
        if has_arguments:
            summary = "用法：/status"
            succeeded = False
            reason = "invalid_arguments"
        else:
            try:
                summary = self._employee_status_summary(
                    tenant_key=metadata.tenant_key,
                    agent_id=metadata.agent_id,
                    chat_id=remote_chat_id,
                    thread_root_id=remote_root_id,
                )
                succeeded = True
                reason = "completed"
            except Exception:
                logger.exception("employee status inspection failed closed")
                summary = "员工状态暂不可用，请稍后重试。"
                succeeded = False
                reason = "unavailable"
        lifecycle.status_response(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            chat_id=remote_chat_id,
            thread_root_message_id=remote_root_id or remote_message_id,
            command_acceptance_id=acceptance_id,
            summary=summary,
            succeeded=succeeded,
        )
        try:
            ingress.record_disposition(
                acceptance_id,
                state="terminal",
                reason_code=f"status_{reason}",
            )
        except IngressConflictError:
            pass
        self._request_employee_outbox_delivery()
        return True

    def _employee_status_summary(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        chat_id: str,
        thread_root_id: str,
    ) -> str:
        """Combine durable attempt facts with a coarse process-local view."""

        dispatch = self._dispatch
        employee_runtime = getattr(dispatch, "employee_runtime", None) if dispatch is not None else None
        if employee_runtime is None:
            raise RuntimeError("employee runtime is unavailable")
        actor = employee_runtime.inspect(agent_id)
        status = getattr(actor.status, "value", actor.status)
        labels = {
            "recovering": "恢复中",
            "ready_cold": "空闲（冷会话）",
            "starting_session": "会话启动中",
            "ready_warm": "空闲（热会话）",
            "busy": "执行中",
            "degraded": "降级",
            "stopping": "停止中",
            "stopped": "已停止",
        }
        if status not in labels:
            raise RuntimeError("employee runtime status is invalid")
        durable = dispatch.scoped_attempt_status(
            tenant_key=tenant_key,
            agent_id=agent_id,
            chat_id=chat_id,
            thread_root_id=thread_root_id,
        )
        active_count = durable.active_count
        stopping_count = durable.stopping_count
        if not active_count:
            task_view = "当前会话无活动任务"
        elif active_count == 1 and stopping_count:
            task_view = "当前会话有 1 个活动任务（停止中）"
        elif stopping_count:
            task_view = f"当前会话有 {active_count} 个活动任务（其中 {stopping_count} 个停止中）"
        else:
            task_view = f"当前会话有 {active_count} 个活动任务"
        mailbox_depth = getattr(actor, "mailbox_depth", 0)
        if type(mailbox_depth) is not int or mailbox_depth < 0:
            raise RuntimeError("employee mailbox depth is invalid")
        return f"员工状态：{labels[status]}\n任务：{task_view}\n队列：{mailbox_depth}"

    def _handle_main_bot_group_command_ingress(
        self,
        acceptance_id: str,
        *,
        targeted_group_task: TargetedTaskParseResult | None = None,
    ) -> bool:
        """Route one addressed task; suppress every other group slash observation."""

        ingress = self._ingress
        if ingress is None:
            return False
        record = ingress.state.by_acceptance_id.get(acceptance_id)
        if record is None or record.disposition is not None:
            return False
        metadata = record.metadata
        if metadata.event_type != "im.message.receive_v1" or metadata.action_identity:
            return False
        try:
            payload = ingress.get_payload(acceptance_id)
        except Exception:
            return False
        first = payload.normalized_parts[0] if len(payload.normalized_parts) == 1 else None
        if not isinstance(first, Mapping) or first.get("type") != "message" or first.get("chat_type") != "group":
            return False
        content = first.get("content")
        text = content.get("text") if isinstance(content, Mapping) else None
        if not isinstance(text, str):
            return False
        if targeted_group_task is not None and targeted_group_task.state is TargetedTaskState.TARGETED_VALID:
            return False
        if targeted_group_task is not None and targeted_group_task.state is TargetedTaskState.INDETERMINATE:
            return False
        if targeted_group_task is not None and targeted_group_task.state is TargetedTaskState.TARGETED_INVALID:
            lifecycle = self._outbox_lifecycle
            coordinates = _composition_module()._bound_remote_coordinates(metadata, first)
            if lifecycle is None or coordinates is None:
                return False
            remote_chat_id, remote_message_id, remote_root_id = coordinates
            # Anchor the idempotent employee-owned response before consuming
            # ingress so a crash can safely replay this command.
            lifecycle.task_usage_response(
                tenant_key=metadata.tenant_key,
                agent_id=metadata.agent_id,
                chat_id=remote_chat_id,
                thread_root_message_id=remote_root_id or remote_message_id,
                command_acceptance_id=acceptance_id,
            )
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code="task_invalid_arguments",
                )
            except IngressConflictError:
                pass
            self._request_employee_outbox_delivery()
            return True
        if not is_group_slash_observation(first):
            return False
        try:
            ingress.record_disposition(
                acceptance_id,
                state="ignored",
                reason_code="main_bot_group_command",
            )
        except IngressConflictError:
            pass
        return True

    @staticmethod
    def _parse_data_control(texts: list[str]) -> tuple[str, int] | None:
        if texts == ["/memory"]:
            return ("/memory", 0)
        if len(texts) != 1:
            return None
        matched = re.fullmatch(r"/history(?:\s+([1-9]|[12][0-9]|3[01]))?", texts[0])
        if matched is None:
            return None
        return ("/history", int(matched.group(1) or "7"))

    def _handle_data_control(
        self,
        *,
        acceptance_id: str,
        command: str,
        record: Any,
        payload: Any,
        requester_principal_id: str,
        history_days: int = 7,
    ) -> bool:
        data = self._data
        lifecycle = self._outbox_lifecycle
        ingress = self._ingress
        if data is None or lifecycle is None or ingress is None:
            return False
        metadata = record.metadata
        first = payload.normalized_parts[0]
        coordinates = _composition_module()._bound_remote_coordinates(metadata, first) if isinstance(first, Mapping) else None
        if coordinates is None:
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code=f"{command.removeprefix('/')}_coordinates_invalid",
                )
            except IngressConflictError:
                pass
            return True
        remote_chat_id, remote_message_id, remote_root_id = coordinates
        chat_type = first.get("chat_type", "") if isinstance(first, Mapping) else ""
        request = AuthenticatedDataRequest(
            principal_id=requester_principal_id,
            tenant_key=metadata.tenant_key,
            receiving_bot_app_id=metadata.app_id,
            chat_id=remote_chat_id,
            chat_type=chat_type,
            thread_root_id=remote_root_id or remote_message_id,
            requested_agent_id=metadata.agent_id,
        )
        succeeded = False
        reason = "failed"
        try:
            if command == "/history":
                from datetime import datetime, timedelta
                from zoneinfo import ZoneInfo

                end = datetime.now(ZoneInfo(data.service.shard_timezone)).date()
                start = end - timedelta(days=history_days - 1)
                result = data.query.query(
                    request,
                    HistoryQuerySpec(
                        start_day=start.isoformat(),
                        end_day=end.isoformat(),
                        page_size=20,
                    ),
                )
                rows = [f"{item.ended_at[:19]} · {item.status} · {item.safe_summary_text}" for item in result.records]
                summary = f"最近 {history_days} 天暂无可见执行记录。" if not rows else "\n".join(rows)
            else:
                result = data.memory_query.query(
                    request,
                    MemoryQuerySpec(agent_id=metadata.agent_id),
                )
                summary = result.content or "当前会话暂无员工记忆摘要。"
            succeeded = True
            reason = "completed"
        except QueryDeniedError:
            summary = "权限不足，无法读取该员工数据。"
            reason = "denied"
        except Exception:
            logger.exception("employee data control failed closed")
            summary = "员工数据暂不可用，请稍后重试或联系管理员。"
        lifecycle.read_response(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            chat_id=remote_chat_id,
            thread_root_message_id=remote_root_id or remote_message_id,
            command_acceptance_id=acceptance_id,
            command=command,
            summary=summary,
            succeeded=succeeded,
        )
        try:
            ingress.record_disposition(
                acceptance_id,
                state="terminal",
                reason_code=f"{command.removeprefix('/')}_{reason}",
            )
        except IngressConflictError:
            pass
        self._request_employee_outbox_delivery()
        return True

    def _event_callback(
        self,
        intent_id: str,
        generation: int,
    ) -> Callable[[dict[str, Any]], None]:
        def callback(payload: dict[str, Any]) -> None:
            if self._closing or self._loop is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                self._handle_channel_event(intent_id, generation, payload),
                self._loop,
            )

            def completed(done: concurrent.futures.Future[Any]) -> None:
                try:
                    done.result()
                except Exception as exc:
                    logger.error(
                        "employee Channel event failed closed: %s",
                        type(exc).__name__,
                    )

            future.add_done_callback(completed)

        return callback

    def _submit_employee_admission(
        self,
        acceptance_id: str,
    ) -> concurrent.futures.Future[bool] | None:
        """Submit ingress admission outside the event loop's shared executor."""

        with self._employee_admission_executor_lock:
            if self._employee_admission_closed:
                raise RuntimeError("employee admission executor is closed")
            if len(self._employee_admission_futures) >= self._employee_admission_max_pending:
                # The acceptance is already anchored.  Bound volatile callback
                # work and let the durable dispatch scan pick it up.
                self._dispatch_wakeup.set()
                return None
            executor = self._employee_admission_executor
            if executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4,
                    thread_name_prefix="employee-ingress-admission",
                )
                self._employee_admission_executor = executor
            future = executor.submit(
                self._admit_employee_ingress_once,
                acceptance_id,
            )
            self._employee_admission_futures.add(future)

        def completed(done: concurrent.futures.Future[bool]) -> None:
            with self._employee_admission_executor_lock:
                self._employee_admission_futures.discard(done)

        future.add_done_callback(completed)
        return future

    def _shutdown_employee_admission_executor(self, *, timeout: float) -> bool:
        """Boundedly quiesce the runtime-owned Channel admission executor."""

        with self._employee_admission_executor_lock:
            self._employee_admission_closed = True
            futures = tuple(self._employee_admission_futures)
            executor = self._employee_admission_executor
        pending: set[concurrent.futures.Future[bool]] = set()
        if futures:
            _done, pending = concurrent.futures.wait(futures, timeout=timeout)
            for future in pending:
                future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            with self._employee_admission_executor_lock:
                if self._employee_admission_executor is executor:
                    self._employee_admission_executor = None
        return not pending

    def _shutdown_employee_handoff_finalization_executor(
        self,
        *,
        timeout: float,
    ) -> bool:
        """Fence finalizer admission and boundedly observe irreversible appends."""

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout < 0
        ):
            raise ValueError("employee handoff finalization timeout is invalid")
        lock, futures_by_key = self._employee_handoff_finalization_state()
        with lock:
            self._employee_handoff_finalization_closed = True
            futures = tuple(futures_by_key.values())
            executor = getattr(
                self,
                "_employee_handoff_finalization_executor",
                None,
            )
        pending: set[concurrent.futures.Future[Any]] = set()
        if futures:
            _done, pending = concurrent.futures.wait(
                futures,
                timeout=float(timeout),
            )
        if executor is not None:
            # A queued finalizer is still a recovery obligation: cancelling it
            # could leave an accepted handoff without a durable terminal
            # decision.  Fence new submissions, but let admitted work finish.
            executor.shutdown(wait=False, cancel_futures=False)
            with lock:
                if self._employee_handoff_finalization_executor is executor:
                    self._employee_handoff_finalization_executor = None
        return not pending

    async def _handle_channel_event(
        self,
        _intent_id: str,
        _generation: int,
        payload: dict[str, Any],
    ) -> None:
        event_name = payload.get("event")
        if event_name == "durableIngressAccepted":
            data = payload.get("data")
            acceptance_id = data.get("acceptance_id") if isinstance(data, dict) else None
            if isinstance(acceptance_id, str) and acceptance_id:
                try:
                    future = self._submit_employee_admission(acceptance_id)
                    if future is not None:
                        await asyncio.wrap_future(
                            future,
                        )
                finally:
                    self._notify_employee_handoff_progress()
            return
        return

    def _record_employee_ingress_group_event(self, acceptance_id: str) -> bool:
        ingress = self._ingress
        ledger = self._group_ledger
        if ingress is None or ledger is None:
            return False
        record = ingress.state.by_acceptance_id.get(acceptance_id)
        if record is None:
            return False
        payload = ingress.get_payload(acceptance_id)
        if len(payload.normalized_parts) != 1:
            return False
        part = payload.normalized_parts[0]
        if not isinstance(part, Mapping) or part.get("chat_type") != "group":
            return False
        coordinates = _composition_module()._bound_remote_coordinates(record.metadata, part)
        if coordinates is None:
            return False
        chat_id, message_id, _root_id = coordinates
        content = part.get("content")
        text = content.get("text") if isinstance(content, Mapping) else content
        if not isinstance(text, str) or not text:
            return False
        sender_id = part.get("sender_id")
        sender_union_id = part.get("sender_union_id", "")
        service = self._service
        is_team_assignment = getattr(record.metadata, "event_type", "") == "ghostap.team.assignment.v1"
        if (
            not isinstance(sender_id, str)
            or not sender_id
            or sender_id != record.metadata.sender_principal_id
            or not isinstance(sender_union_id, str)
            or service is None
        ):
            return False
        service.synchronize_projection()
        states = tuple(
            state
            for state in service.list_states()
            if state.tenant_key == record.metadata.tenant_key and state.agent_id == record.metadata.agent_id
        )
        if len(states) != 1:
            return False
        owner_principal_id = states[0].requester_principal_id
        team_run_id = part.get("team_run_id") if is_team_assignment else None
        team_step_id = part.get("team_step_id") if is_team_assignment else None
        if is_team_assignment:
            canonical_sender_id = (
                sender_id
                if part.get("type") == "team_assignment"
                and sender_id == owner_principal_id
                and all(isinstance(value, str) and value for value in (team_run_id, team_step_id))
                and record.metadata.action_identity == f"team:{team_run_id}:{team_step_id}"
                else None
            )
        else:
            canonical_sender_id = self._resolve_employee_requester_principal(
                tenant_key=record.metadata.tenant_key,
                agent_id=record.metadata.agent_id,
                owner_principal_id=owner_principal_id,
                sender_principal_id=sender_id,
                sender_union_id=sender_union_id,
            )
        if canonical_sender_id is None:
            return False
        ledger.publish(
            tenant_key=record.metadata.tenant_key,
            chat_id=chat_id,
            thread_id=str(part.get("feishu_thread_id", "")),
            message_id=message_id,
            transport_principal_id=("main_bot" if is_team_assignment else record.metadata.bot_principal_id),
            transport_event_id=record.metadata.event_id,
            payload=GroupEventPayload(
                sender_id=record.metadata.sender_principal_id,
                sender_id_type=str(part.get("sender_id_type", "")),
                sender_type=str(part.get("sender_type", "")),
                sender_tenant_key=str(part.get("sender_tenant_key", "")),
                text=text,
                timestamp=_composition_module().time.time(),
            ),
            causal_event_id=(f"{team_run_id}:{team_step_id}" if is_team_assignment else ""),
        )
        return True
