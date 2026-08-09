"""Unique durable dispatch/terminal coordinator for visible employees."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from ..context.models import ContextUnavailableError, ContextUnavailableReason
from ..data.models import (
    DataKind,
    ExecutionAttemptContext,
    ExecutionHistoryPayloadV1,
    ExecutionHistoryRecordV1,
    SafeExecutionSummary,
)
from ..data.ports import AuthenticatedExecutionTerminal, EmployeeDataSink, PublishEmployeeDocumentCommand
from ..data.service import EmployeeDataService
from ..domain import EmployeeState, WorkerType
from ..ingress.models import parse_canonical_utc
from ..journal.frame import JournalEvent
from ..journal.projections import apply_frame
from ..journal.writer import CommitState, JournalWriter
from ..runtime.employee_supervisor import EmployeeRuntimeSupervisor
from ..supervisor.channel_models import ChannelProcessState
from ..team.runtime import TeamRuntimeResolutionError, team_runtime_guard
from ..workforce.registry import ProjectedAgentRegistry
from .context_prompt import RenderedEmployeePrompt, render_employee_context
from .env_scope import (
    EmployeeEnvironmentAuthority,
    EmployeeProcessEnvironmentMaterial,
    build_employee_backend_env,
)
from .models import (
    DispatchBinding,
    DispatchPermit,
    GatewayExecutionResult,
    GatewayExecutionStatus,
)
from .projection import (
    ATTEMPT_BOUND,
    ATTEMPT_CANCEL_REQUESTED,
    ATTEMPT_DISPATCH_COMMITTED,
    ATTEMPT_TERMINAL,
    GatewayProjectionState,
    reduce_gateway_frame,
)
from .team import EmployeeTeamGateway

logger = logging.getLogger(__name__)

_TRANSIENT_CONTEXT_REASONS = frozenset(
    {
        ContextUnavailableReason.PAGINATION,
        ContextUnavailableReason.ORDERING,
        ContextUnavailableReason.REVISION,
        ContextUnavailableReason.DEADLINE,
        ContextUnavailableReason.SOURCE,
    }
)
_TEAM_ASSIGNMENT_FIELDS = frozenset(
    {
        "type",
        "message_type",
        "chat_type",
        "content",
        "team_instruction",
        "sender_id",
        "sender_id_type",
        "sender_type",
        "sender_tenant_key",
        "feishu_thread_id",
        "team_run_id",
        "team_step_id",
        "team_deadline_at",
    }
)


class EmployeeDispatchError(RuntimeError):
    """Dispatch authority or atomic lifecycle validation failed closed."""


class _ProjectionHeadChanged(RuntimeError):
    """A domain moved after lock-free projection synchronization."""


@dataclass(frozen=True, slots=True)
class PreparedEmployeeDispatch:
    binding: DispatchBinding
    permit: DispatchPermit
    prompt: str


@dataclass(frozen=True, slots=True)
class FinalizedEmployeeAttempt:
    attempt_id: str
    status: GatewayExecutionStatus
    history_record_id: str
    frame_sequence: int


@dataclass(frozen=True, slots=True)
class EmployeeCancellationOutcome:
    status: str
    attempt_id: str = ""
    changed: bool = False


@dataclass(frozen=True, slots=True)
class TeamAttemptSnapshot:
    status: str
    output: str = ""
    history_record_id: str = ""
    error_code: str = ""


EnvironmentProvider = Callable[
    [EmployeeEnvironmentAuthority],
    EmployeeProcessEnvironmentMaterial,
]
RegistryFactory = Callable[[object], ProjectedAgentRegistry]


class EmployeeAttemptLifecycle(Protocol):
    def queued(self, binding: DispatchBinding) -> object: ...

    def running(self, binding: DispatchBinding) -> object: ...

    def terminal(
        self,
        binding: DispatchBinding,
        result: GatewayExecutionResult,
    ) -> object: ...


class EmployeeDispatchCoordinator:
    """Own the only Router queued -> ACP -> terminal attempt path."""

    def __init__(
        self,
        *,
        writer: JournalWriter,
        hire_service: object,
        ingress_service: object,
        router: object,
        data_service: EmployeeDataService,
        data_sink: EmployeeDataSink | None = None,
        channel_supervisor: object,
        team_runtime: object,
        context_service: object,
        environment_provider: EnvironmentProvider,
        registry_factory: RegistryFactory | None = None,
        gateway: EmployeeTeamGateway | None = None,
        timeout_seconds: float = 600.0,
        clock: Callable[[], datetime] | None = None,
        attempt_lifecycle: EmployeeAttemptLifecycle | None = None,
        admin_principal_ids: frozenset[str] = frozenset(),
        team_owner_resolver: Callable[[str], str] | None = None,
        employee_runtime_mode: str,
        employee_session_idle_ttl_seconds: float = 900.0,
        shadow_observer: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if not callable(environment_provider):
            raise TypeError("environment_provider is required")
        self._writer = writer
        self._hire = hire_service
        self._ingress = ingress_service
        self._router = router
        self._data = data_service
        self._data_sink = data_sink
        self._channels = channel_supervisor
        self._team_runtime = team_runtime
        self._context = context_service
        self._environment_provider = environment_provider
        self._registry_factory = registry_factory or ProjectedAgentRegistry
        self._employee_runtime = (
            EmployeeRuntimeSupervisor(
                writer=writer,
                idle_ttl_seconds=employee_session_idle_ttl_seconds,
            )
            if gateway is None and employee_runtime_mode == "actor"
            else None
        )
        self._gateway = gateway or EmployeeTeamGateway(
            runtime_mode=employee_runtime_mode,
            runtime_supervisor=self._employee_runtime,
            shadow_observer=shadow_observer,
        )
        self._timeout_seconds = float(timeout_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._attempt_lifecycle = attempt_lifecycle
        self._gateway_state = GatewayProjectionState()
        self._projection_sync_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._terminal_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._admin_principal_ids = frozenset(admin_principal_ids)
        self._team_owner_resolver = team_owner_resolver or (lambda _chat_id: "")

    @property
    def employee_runtime(self) -> EmployeeRuntimeSupervisor | None:
        return self._employee_runtime

    def close(self) -> None:
        close = getattr(self._gateway, "close", None)
        if callable(close):
            close()

    @property
    def state(self) -> GatewayProjectionState:
        with self._projection_sync_lock:
            return self._gateway_state.clone()

    def prepare_next(self) -> PreparedEmployeeDispatch | None:
        """Prepare with bounded retries when the Journal head races."""

        for _attempt in range(3):
            try:
                return self._prepare_next_once()
            except _ProjectionHeadChanged:
                continue
        raise EmployeeDispatchError("employee dispatch head remained unstable")

    def _prepare_next_once(self) -> PreparedEmployeeDispatch | None:
        """Anchor Router dispatch and immutable attempt authority in one frame."""

        grant = self._router.peek_dispatch_candidate()
        if grant is None:
            return None
        part = (
            grant.payload.normalized_parts[0]
            if len(grant.payload.normalized_parts) == 1
            else None
        )
        is_team_assignment = grant.record.event_type == "ghostap.team.assignment.v1"
        team_instruction = ""
        team_deadline: datetime | None = None
        if is_team_assignment:
            valid_team_part = isinstance(part, Mapping) and frozenset(part) == _TEAM_ASSIGNMENT_FIELDS
            if valid_team_part:
                team_instruction = part.get("team_instruction")  # type: ignore[assignment]
                valid_team_part = (
                    isinstance(team_instruction, str)
                    and bool(team_instruction)
                    and team_instruction == team_instruction.strip()
                    and len(team_instruction) <= 14_000
                    and part.get("type") == "team_assignment"
                    and part.get("content") == team_instruction
                    and all(
                        isinstance(part.get(key), str)
                        and bool(part.get(key))
                        and part.get(key) == str(part.get(key)).strip()
                        for key in (
                            "sender_id",
                            "sender_tenant_key",
                            "team_run_id",
                            "team_step_id",
                        )
                    )
                )
            if not valid_team_part:
                self._router.reject_dispatch_candidate(
                    grant.record.acceptance_id,
                    reason_code="team_assignment_invalid",
                )
                return None
            try:
                team_deadline = parse_canonical_utc(
                    part["team_deadline_at"], "team_deadline_at"
                )
            except (TypeError, ValueError):
                self._router.reject_dispatch_candidate(
                    grant.record.acceptance_id,
                    reason_code="team_assignment_invalid",
                )
                return None
            if not self._team_assignment_effect_is_active(part):
                self._router.reject_dispatch_candidate(
                    grant.record.acceptance_id,
                    reason_code="team_step_inactive",
                )
                return None
        try:
            snapshot = self._context.assemble(grant.request)
        except ContextUnavailableError as exc:
            if exc.reason in _TRANSIENT_CONTEXT_REASONS:
                partial = getattr(self._context, "assemble_canonical_partial", None)
                if not callable(partial):
                    self._router.defer_dispatch_candidate(
                        grant.record.acceptance_id,
                        terminal_reason="canonical_context_unavailable",
                    )
                    return None
                try:
                    causal_event_id = (
                        f"{part['team_run_id']}:{part['team_step_id']}"
                        if is_team_assignment
                        else ""
                    )
                    snapshot = partial(
                        grant.request,
                        warning_reason=exc.reason,
                        causal_event_id=causal_event_id,
                    )
                except ContextUnavailableError:
                    record = self._router.defer_dispatch_candidate(
                        grant.record.acceptance_id,
                        terminal_reason="canonical_context_unavailable",
                    )
                    logger.warning(
                        "employee canonical context unavailable; candidate %s",
                        "terminal" if record.state == "terminal" else "deferred",
                    )
                    return None
            else:
                record = self._router.reject_dispatch_candidate(
                    grant.record.acceptance_id,
                    reason_code="context_unavailable",
                )
                logger.warning(
                    "employee dispatch context unavailable; candidate %s: reason=%s",
                    "terminal" if record.state == "terminal" else "deferred",
                    exc.reason.value,
                )
                return None
        self._validate_context_watermark(grant.request, snapshot)
        if team_deadline is not None and team_deadline <= self._clock():
            self._router.reject_dispatch_candidate(
                grant.record.acceptance_id,
                reason_code="team_step_expired",
            )
            return None

        registry = self._registry_factory(self._hire.projection_state)
        employee = registry.get(grant.request.tenant_key, grant.request.agent_id)
        agent = registry.as_execution_identity(
            grant.request.tenant_key,
            grant.request.agent_id,
        )
        projected_binding = registry.context_binding(
            tenant_key=grant.request.tenant_key,
            agent_id=grant.request.agent_id,
            bot_principal_id=grant.record.bot_principal_id,
            app_id=grant.record.app_id,
            chat_id=grant.request.chat_id,
        )
        if employee is None or agent is None or projected_binding is None:
            raise EmployeeDispatchError("employee authority is unavailable")
        system_instruction = agent.system_prompt
        if team_instruction:
            system_instruction = (
                f"{system_instruction}\n\n## TEAM_COORDINATOR_INSTRUCTION\n"
                f"{team_instruction}"
            )
        rendered = render_employee_context(
            snapshot,
            system_instruction=system_instruction,
            constraints_digest=grant.request.constraints_digest,
        )
        environment_authority = EmployeeEnvironmentAuthority(
            tenant_key=employee.tenant_key,
            agent_id=employee.agent_id,
            employee_version=employee.aggregate_version,
            credential_ref=projected_binding.principal.credential_ref,
        )
        try:
            material = self._environment_provider(environment_authority)
        except Exception:
            raise EmployeeDispatchError("employee environment provider failed") from None
        if not isinstance(material, EmployeeProcessEnvironmentMaterial) or material.authority != environment_authority:
            raise EmployeeDispatchError("employee environment authority mismatch")
        employee_home = str(Path(agent.workspace_path).parent)
        try:
            env = build_employee_backend_env(
                material,
                agent_type=agent.agent_type,
                employee_home=employee_home,
                workspace_path=agent.workspace_path,
            )
        except Exception:
            raise EmployeeDispatchError("employee environment projection failed") from None
        captured_head = self._presynchronize_domains()
        if team_instruction and not self._team_assignment_effect_is_active(part):
            self._router.reject_dispatch_candidate(
                grant.record.acceptance_id,
                reason_code="team_step_inactive",
            )
            return None
        # Detect a team stop committed during the lock-free effect scan before
        # entering any Team activation boundary; the guarded check below is
        # repeated to close later races as well.
        self._require_presynchronized_head(captured_head)

        try:
            activation_context = self._team_runtime.employee_activation_guard(
                chat_id=grant.request.chat_id,
            )
            with activation_context as team_binding, ExitStack() as stack:
                stack.enter_context(self._projection_sync_lock)
                stack.enter_context(self._hire.employee_dispatch_guard())
                stack.enter_context(self._ingress.employee_dispatch_guard(router=self._router))
                stack.enter_context(self._data.employee_dispatch_guard())
                stack.enter_context(self._channels.employee_dispatch_guard())
                stack.enter_context(self._writer.transaction_guard())
                self._require_presynchronized_head(captured_head)
                current_registry = self._registry_factory(self._hire.projection_state)
                current = current_registry.get(
                    grant.request.tenant_key,
                    grant.request.agent_id,
                )
                current_agent = current_registry.as_execution_identity(
                    grant.request.tenant_key,
                    grant.request.agent_id,
                )
                current_projected_binding = current_registry.context_binding(
                    tenant_key=grant.request.tenant_key,
                    agent_id=grant.request.agent_id,
                    bot_principal_id=grant.record.bot_principal_id,
                    app_id=grant.record.app_id,
                    chat_id=grant.request.chat_id,
                )
                if (
                    current != employee
                    or current_agent != agent
                    or current_projected_binding is None
                    or current_projected_binding.employee != projected_binding.employee
                    or current_projected_binding.principal != projected_binding.principal
                ):
                    raise EmployeeDispatchError("employee authority changed")
                ingress_identity = self._ingress.dispatch_identity_unlocked(grant.record.acceptance_id)
                ingress_metadata = ingress_identity[2]
                if (
                    ingress_identity[0] != grant.record.aggregate_id
                    or ingress_identity[1] != grant.record.acceptance_id
                    or ingress_identity[3] != grant.payload.payload_sha256
                    or ingress_metadata.envelope_id != grant.record.envelope_id
                    or ingress_metadata.tenant_key != grant.record.tenant_key
                    or ingress_metadata.agent_id != grant.record.agent_id
                    or ingress_metadata.message_id != grant.record.message_id
                ):
                    raise EmployeeDispatchError("employee ingress identity changed")
                current_router = self._router.state.by_acceptance_id.get(grant.record.acceptance_id)
                if current_router != grant.record:
                    raise EmployeeDispatchError("employee Router grant changed")
                self._validate_employee(current, grant)
                status = self._channels.status(grant.request.agent_id)
                authority_connection = self._validate_channel(status, grant)
                router_event = self._router.preflight_dispatch_event_unlocked(
                    acceptance_id=grant.record.acceptance_id,
                )
                binding = self._build_binding(
                    grant=grant,
                    employee=current,
                    team_binding=team_binding,
                    rendered=rendered,
                    authority_connection=authority_connection,
                )
                warning_events = tuple(
                    JournalEvent(
                        event_type="context.warning.recorded",
                        aggregate_id=f"context-warning:{binding.attempt_id}:{item.code}",
                        payload={
                            "attempt_id": binding.attempt_id,
                            "quality": snapshot.quality.value,
                            "code": item.code,
                            "source": item.source,
                        },
                    )
                    for item in snapshot.warnings
                )
                events = (
                    router_event,
                    JournalEvent(
                        event_type=ATTEMPT_BOUND,
                        aggregate_id=binding.attempt_id,
                        payload={"binding": binding.to_dict()},
                    ),
                    JournalEvent(
                        event_type=ATTEMPT_DISPATCH_COMMITTED,
                        aggregate_id=binding.attempt_id,
                        payload={
                            "attempt_id": binding.attempt_id,
                            "permit_id": binding.permit_id,
                        },
                    ),
                    *warning_events,
                )
                result = self._commit_events_unlocked(events)
                self._apply_committed_frame_unlocked(result.frame)
        except TeamRuntimeResolutionError:
            self._router.reject_dispatch_candidate(
                grant.record.acceptance_id,
                reason_code="team_unavailable",
            )
            return None
        timeout_seconds = self._timeout_seconds
        if team_deadline is not None:
            timeout_seconds = min(
                timeout_seconds,
                (team_deadline - self._clock()).total_seconds(),
            )
            if timeout_seconds <= 0:
                self.finalize_attempt(
                    binding.attempt_id,
                    GatewayExecutionResult(
                        GatewayExecutionStatus.TIMEOUT,
                        safe_error_code="team_step_deadline_expired",
                    ),
                    request_text=rendered.prompt,
                )
                return None
        permit = self._gateway.issue_permit(
            binding=binding,
            prompt=rendered.prompt,
            engine=team_binding.engine,
            agent=agent,
            timeout_seconds=timeout_seconds,
            env=env,
        )
        if self._attempt_lifecycle is not None:
            self._attempt_lifecycle.queued(binding)
        return PreparedEmployeeDispatch(binding, permit, rendered.prompt)

    def _team_assignment_effect_is_active(
        self,
        part: Mapping[str, object] | None,
    ) -> bool:
        """Read the owning team effect before the Journal-head CAS section."""

        if part is None:
            return False
        run_id = part.get("team_run_id")
        step_id = part.get("team_step_id")
        if not isinstance(run_id, str) or not run_id:
            return False
        if not isinstance(step_id, str) or not step_id:
            return False
        # Coordinator mode anchors the full assignment identity.  The retained
        # Team ``legacy_pipeline`` writes the same effect under run/step.  Both
        # are Team-owned Journal facts; newest matching evidence wins.
        effect_coordinates = {
            (f"{run_id}:assignment:{step_id}", "team.v2.effect."),
            (f"{run_id}:{step_id}", "team.effect."),
        }
        state = ""
        for frame in self._writer.replay():
            for event in frame.events:
                if event.payload.get("effect_type") != "employee_dispatch":
                    continue
                for aggregate_id, event_prefix in effect_coordinates:
                    if event.aggregate_id == aggregate_id and event.event_type.startswith(
                        event_prefix
                    ):
                        state = event.event_type.rsplit(".", 1)[-1]
                        break
        return state in {"prepared", "executing"}

    def _presynchronize_domains(self) -> tuple[int, str]:
        """Perform potentially full projection repair before the commit section."""

        with self._projection_sync_lock:
            with self._hire.employee_dispatch_guard():
                self._hire.synchronize_projection_unlocked()
            with self._ingress.employee_dispatch_guard(router=self._router):
                self._ingress.synchronize_projection_unlocked()
                self._router.synchronize_projection_unlocked()
            with self._data.employee_dispatch_guard():
                self._data.synchronize_projection_unlocked()
            self._synchronize_gateway_unlocked()
            last = self._writer.get_last_frame()
            return (0, "") if last is None else (last.sequence, last.frame_hash)

    def _require_presynchronized_head(self, expected: tuple[int, str]) -> None:
        """Verify every projection and the Journal still share one captured head."""

        last = self._writer.get_last_frame()
        current = (0, "") if last is None else (last.sequence, last.frame_hash)
        projection_heads = (
            (
                self._hire.projection_state.cursor_sequence,
                self._hire.projection_state.cursor_hash,
            ),
            (self._ingress.state.cursor_sequence, self._ingress.state.cursor_hash),
            (self._router.state.cursor_sequence, self._router.state.cursor_hash),
            (self._data.state.cursor_sequence, self._data.state.cursor_hash),
            (
                self._gateway_state.cursor_sequence,
                self._gateway_state.cursor_hash,
            ),
        )
        if current != expected or any(head != expected for head in projection_heads):
            raise _ProjectionHeadChanged

    def execute_prepared(
        self,
        prepared: PreparedEmployeeDispatch,
    ) -> FinalizedEmployeeAttempt:
        if self._attempt_lifecycle is not None:
            self._attempt_lifecycle.running(prepared.binding)
        result = self._gateway.execute_permit(prepared.permit)
        return self.finalize_attempt(
            prepared.binding.attempt_id,
            result,
            request_text=prepared.prompt,
        )

    def team_attempt_result(self, acceptance_id: str) -> TeamAttemptSnapshot | None:
        """Read Router, Gateway, and History at one verified Journal head."""

        if not isinstance(acceptance_id, str) or not acceptance_id.startswith("acc_"):
            raise ValueError("team acceptance_id is required")
        for _attempt in range(3):
            captured_head = self._presynchronize_domains()
            with self._projection_sync_lock:
                gateway = self._gateway_state.clone()
            router = self._router.state.clone()
            attempt_id = gateway.attempt_by_acceptance_id.get(acceptance_id)
            snapshot: TeamAttemptSnapshot | None = None
            if attempt_id:
                lifecycle = gateway.attempts.get(attempt_id)
                if lifecycle is not None and lifecycle.terminal_status:
                    try:
                        payload = self._data.get_history_payload(
                            lifecycle.history_record_id
                        )
                    except Exception:
                        snapshot = TeamAttemptSnapshot(
                            "action_required",
                            history_record_id=lifecycle.history_record_id,
                            error_code="team_history_unavailable",
                        )
                    else:
                        completed = lifecycle.terminal_status == "completed"
                        snapshot = TeamAttemptSnapshot(
                            lifecycle.terminal_status,
                            output=payload.result_text if completed else "",
                            history_record_id=lifecycle.history_record_id,
                            error_code="" if completed else payload.error_detail,
                        )
            if snapshot is None:
                routed = router.by_acceptance_id.get(acceptance_id)
                if routed is not None and routed.state == "terminal":
                    status = {
                        "team_step_expired": "timeout",
                        "team_step_canceled": "canceled",
                    }.get(routed.reason_code, "action_required")
                    snapshot = TeamAttemptSnapshot(
                        status,
                        error_code=routed.reason_code,
                    )
            try:
                self._require_presynchronized_head(captured_head)
            except _ProjectionHeadChanged:
                continue
            return snapshot
        raise EmployeeDispatchError("team attempt result head remained unstable")

    def dispatch_next(self) -> FinalizedEmployeeAttempt | None:
        prepared = self.prepare_next()
        return None if prepared is None else self.execute_prepared(prepared)

    def finalize_attempt(
        self,
        attempt_id: str,
        execution_result: GatewayExecutionResult,
        *,
        request_text: str = "",
    ) -> FinalizedEmployeeAttempt:
        """Commit terminal facts, then append the employee response snapshot."""

        with self._terminal_lock:
            self._synchronize_gateway_from_journal()
            lifecycle = self._gateway_state.attempts.get(attempt_id)
            if lifecycle is not None and lifecycle.cancel_requested and not lifecycle.terminal_status:
                execution_result = GatewayExecutionResult(
                    status=GatewayExecutionStatus.CANCELED,
                    safe_error_code="cancel_requested",
                )
            finalized = self._finalize_attempt_without_reporting(
                attempt_id,
                execution_result,
                request_text=request_text,
            )
            if execution_result.status is GatewayExecutionStatus.COMPLETED:
                assert lifecycle is not None
                self._publish_completed_artifacts(
                    lifecycle.binding,
                    execution_result,
                    request_text=request_text,
                )
        if self._attempt_lifecycle is not None:
            self._synchronize_gateway_from_journal()
            lifecycle = self._gateway_state.attempts.get(attempt_id)
            if lifecycle is None:
                raise EmployeeDispatchError("terminal attempt binding is unavailable")
            self._attempt_lifecycle.terminal(lifecycle.binding, execution_result)
        return finalized

    def request_cancel(
        self,
        *,
        agent_id: str,
        chat_id: str,
        requester_principal_id: str,
        command_acceptance_id: str,
    ) -> EmployeeCancellationOutcome:
        """Anchor an authorized cancellation before touching the live gateway."""

        if not all(
            isinstance(value, str) and value and value == value.strip()
            for value in (
                agent_id,
                chat_id,
                requester_principal_id,
                command_acceptance_id,
            )
        ):
            raise ValueError("cancellation coordinates are required")
        with self._terminal_lock:
            self._synchronize_gateway_from_journal()
            matching = tuple(
                record
                for record in self._gateway_state.attempts.values()
                if record.binding.agent_id == agent_id
                and record.binding.chat_id == chat_id
                and record.dispatch_committed
            )
            if not matching:
                return EmployeeCancellationOutcome("no_active")
            live = tuple(record for record in matching if not record.terminal_status)
            target = live[0] if len(live) == 1 else max(
                matching,
                key=lambda item: item.dispatch_sequence,
            )
            try:
                team_owner = self._team_owner_resolver(chat_id)
            except Exception:
                team_owner = ""
            if requester_principal_id not in {
                target.binding.requester_principal_id,
                *self._admin_principal_ids,
                team_owner,
            }:
                return EmployeeCancellationOutcome(
                    "forbidden",
                    target.binding.attempt_id,
                )
            if not live:
                return EmployeeCancellationOutcome(
                    "already_terminal",
                    target.binding.attempt_id,
                )
            if len(live) != 1:
                return EmployeeCancellationOutcome("ambiguous")
            target = live[0]
            if target.cancel_requested:
                return EmployeeCancellationOutcome(
                    "cancel_requested",
                    target.binding.attempt_id,
                    False,
                )
            requested_at = self._timestamp()
            for _attempt in range(3):
                captured_head = self._presynchronize_domains()
                try:
                    with team_runtime_guard(), ExitStack() as stack:
                        stack.enter_context(self._projection_sync_lock)
                        stack.enter_context(self._hire.employee_dispatch_guard())
                        stack.enter_context(self._ingress.employee_dispatch_guard(router=self._router))
                        stack.enter_context(self._data.employee_dispatch_guard())
                        stack.enter_context(self._channels.employee_dispatch_guard())
                        stack.enter_context(self._writer.transaction_guard())
                        self._require_presynchronized_head(captured_head)
                        current = self._gateway_state.attempts.get(
                            target.binding.attempt_id
                        )
                        if current is None:
                            return EmployeeCancellationOutcome("no_active")
                        if current.terminal_status:
                            return EmployeeCancellationOutcome(
                                "already_terminal",
                                current.binding.attempt_id,
                            )
                        if current.cancel_requested:
                            return EmployeeCancellationOutcome(
                                "cancel_requested",
                                current.binding.attempt_id,
                                False,
                            )
                        event = JournalEvent(
                            event_type=ATTEMPT_CANCEL_REQUESTED,
                            aggregate_id=current.binding.attempt_id,
                            payload={
                                "attempt_id": current.binding.attempt_id,
                                "cancel_epoch": 1,
                                "requester_principal_id": requester_principal_id,
                                "command_acceptance_id": command_acceptance_id,
                                "requested_at": requested_at,
                            },
                        )
                        commit = self._commit_events_unlocked((event,))
                        self._apply_committed_frame_unlocked(commit.frame)
                        binding = current.binding
                    self._gateway.cancel_attempt(binding)
                    return EmployeeCancellationOutcome(
                        "cancel_requested",
                        binding.attempt_id,
                        True,
                    )
                except _ProjectionHeadChanged:
                    continue
            raise EmployeeDispatchError("employee cancellation head remained unstable")

    def request_team_cancel(
        self,
        *,
        acceptance_id: str,
        team_run_id: str,
        team_step_id: str,
    ) -> EmployeeCancellationOutcome:
        """Cancel only the attempt bound to the encrypted Team assignment owner."""

        if not all(
            isinstance(value, str) and value and value == value.strip()
            for value in (acceptance_id, team_run_id, team_step_id)
        ) or not acceptance_id.startswith("acc_"):
            raise ValueError("team cancellation coordinates are required")
        try:
            payload = self._ingress.get_payload(acceptance_id)
        except Exception:
            raise EmployeeDispatchError("team cancellation authority unavailable") from None
        part = (
            payload.normalized_parts[0]
            if len(payload.normalized_parts) == 1
            else None
        )
        if (
            not isinstance(part, Mapping)
            or part.get("type") != "team_assignment"
            or part.get("team_run_id") != team_run_id
            or part.get("team_step_id") != team_step_id
        ):
            raise EmployeeDispatchError("team cancellation owner mismatch")
        requester = part.get("sender_id")
        if not isinstance(requester, str) or not requester:
            raise EmployeeDispatchError("team cancellation requester unavailable")
        with self._terminal_lock:
            requested_at = self._timestamp()
            for _attempt in range(3):
                captured_head = self._presynchronize_domains()
                owner_active = self._team_assignment_effect_is_active(part)
                try:
                    with team_runtime_guard(), ExitStack() as stack:
                        stack.enter_context(self._projection_sync_lock)
                        stack.enter_context(self._hire.employee_dispatch_guard())
                        stack.enter_context(
                            self._ingress.employee_dispatch_guard(router=self._router)
                        )
                        stack.enter_context(self._data.employee_dispatch_guard())
                        stack.enter_context(self._channels.employee_dispatch_guard())
                        stack.enter_context(self._writer.transaction_guard())
                        self._require_presynchronized_head(captured_head)
                        attempt_id = self._gateway_state.attempt_by_acceptance_id.get(
                            acceptance_id
                        )
                        routed = self._router.state.by_acceptance_id.get(acceptance_id)
                        if attempt_id is None:
                            if routed is None:
                                return EmployeeCancellationOutcome("no_active")
                            if routed.state == "terminal":
                                status = (
                                    "cancel_requested"
                                    if routed.reason_code == "team_step_canceled"
                                    else "already_terminal"
                                )
                                return EmployeeCancellationOutcome(status)
                            if routed.state != "queued":
                                return EmployeeCancellationOutcome("no_active")
                            if not owner_active:
                                raise EmployeeDispatchError(
                                    "team cancellation owner mismatch"
                                )
                            event = self._router.preflight_rejection_event_unlocked(
                                acceptance_id=acceptance_id,
                                reason_code="team_step_canceled",
                            )
                            commit = self._commit_events_unlocked((event,))
                            self._apply_committed_frame_unlocked(commit.frame)
                            return EmployeeCancellationOutcome(
                                "cancel_requested",
                                changed=True,
                            )
                        current = self._gateway_state.attempts.get(attempt_id)
                        if current is None:
                            raise EmployeeDispatchError(
                                "team cancellation attempt mismatch"
                            )
                        if current.binding.acceptance_id != acceptance_id:
                            raise EmployeeDispatchError(
                                "team cancellation attempt mismatch"
                            )
                        if current.terminal_status:
                            return EmployeeCancellationOutcome(
                                "already_terminal", attempt_id
                            )
                        if current.cancel_requested:
                            return EmployeeCancellationOutcome(
                                "cancel_requested", attempt_id, False
                            )
                        if not owner_active:
                            raise EmployeeDispatchError(
                                "team cancellation owner mismatch"
                            )
                        event = JournalEvent(
                            event_type=ATTEMPT_CANCEL_REQUESTED,
                            aggregate_id=attempt_id,
                            payload={
                                "attempt_id": attempt_id,
                                "cancel_epoch": 1,
                                "requester_principal_id": requester,
                                "command_acceptance_id": acceptance_id,
                                "requested_at": requested_at,
                            },
                        )
                        commit = self._commit_events_unlocked((event,))
                        self._apply_committed_frame_unlocked(commit.frame)
                        binding = current.binding
                    self._gateway.cancel_attempt(binding)
                    return EmployeeCancellationOutcome(
                        "cancel_requested", attempt_id, True
                    )
                except _ProjectionHeadChanged:
                    continue
            raise EmployeeDispatchError("team cancellation head remained unstable")

    def _finalize_attempt_without_reporting(
        self,
        attempt_id: str,
        execution_result: GatewayExecutionResult,
        *,
        request_text: str = "",
    ) -> FinalizedEmployeeAttempt:
        """Stage history lock-free, then commit all terminal metadata together."""

        self._synchronize_gateway_from_journal()
        lifecycle = self._gateway_state.attempts.get(attempt_id)
        if lifecycle is None or not lifecycle.dispatch_committed:
            raise EmployeeDispatchError("attempt is not dispatch committed")
        if lifecycle.terminal_status:
            if lifecycle.terminal_status != execution_result.status.value or lifecycle.result_digest != _result_digest(
                execution_result
            ):
                raise EmployeeDispatchError("attempt terminal result conflicts")
            return FinalizedEmployeeAttempt(
                attempt_id,
                execution_result.status,
                lifecycle.history_record_id,
                lifecycle.terminal_sequence,
            )
        ended_at = self._timestamp()
        record, payload = self._history_models(
            lifecycle.binding,
            execution_result,
            request_text=request_text,
            ended_at=ended_at,
        )
        staged = self._data.stage_history_payload(record, payload)
        anchored = False
        try:
            for _attempt in range(3):
                captured_head = self._presynchronize_domains()
                try:
                    with team_runtime_guard(), ExitStack() as stack:
                        stack.enter_context(self._projection_sync_lock)
                        stack.enter_context(self._hire.employee_dispatch_guard())
                        stack.enter_context(self._ingress.employee_dispatch_guard(router=self._router))
                        stack.enter_context(self._data.employee_dispatch_guard())
                        stack.enter_context(self._channels.employee_dispatch_guard())
                        stack.enter_context(self._writer.transaction_guard())
                        self._require_presynchronized_head(captured_head)
                        lifecycle = self._gateway_state.attempts.get(attempt_id)
                        if lifecycle is None:
                            raise EmployeeDispatchError("attempt disappeared during terminal commit")
                        if lifecycle.terminal_status:
                            if (
                                lifecycle.terminal_status != execution_result.status.value
                                or lifecycle.result_digest != _result_digest(execution_result)
                            ):
                                raise EmployeeDispatchError("attempt terminal result conflicts")
                            raced = FinalizedEmployeeAttempt(
                                attempt_id,
                                execution_result.status,
                                lifecycle.history_record_id,
                                lifecycle.terminal_sequence,
                            )
                        else:
                            raced = None
                        if raced is None:
                            history_event = self._data.preflight_history_event_unlocked(staged)
                            router_event = self._router.preflight_terminal_event_unlocked(
                                acceptance_id=lifecycle.binding.acceptance_id,
                                reason_code=execution_result.status.value,
                            )
                            terminal_event = JournalEvent(
                                event_type=ATTEMPT_TERMINAL,
                                aggregate_id=attempt_id,
                                payload={
                                    "attempt_id": attempt_id,
                                    "terminal_epoch": 1,
                                    "status": execution_result.status.value,
                                    "result_digest": _result_digest(execution_result),
                                    "history_record_id": record.record_id,
                                    "ended_at": ended_at,
                                },
                            )
                            commit = self._commit_events_unlocked((history_event, terminal_event, router_event))
                            anchored = True
                            self._apply_committed_frame_unlocked(commit.frame)
                            return FinalizedEmployeeAttempt(
                                attempt_id,
                                execution_result.status,
                                record.record_id,
                                commit.frame.sequence,
                            )
                    self._data.quarantine_staged_history(staged)
                    return raced
                except _ProjectionHeadChanged:
                    continue
            raise EmployeeDispatchError("employee terminal head remained unstable")
        except Exception:
            if not anchored:
                self._data.quarantine_staged_history(staged)
            raise

    def recover_incomplete_attempts(self) -> tuple[FinalizedEmployeeAttempt, ...]:
        """Reconcile Actor facts, then terminalize unknowns without re-execution."""

        self._recover_legacy_router_dispatches()
        self._synchronize_gateway_from_journal()
        pending = tuple(
            attempt_id
            for attempt_id, record in self._gateway_state.attempts.items()
            if record.dispatch_committed and not record.terminal_status
        )
        return tuple(
            self.finalize_attempt(
                attempt_id,
                self._recovered_execution_result(attempt_id),
            )
            for attempt_id in pending
        )

    def _recovered_execution_result(
        self,
        attempt_id: str,
    ) -> GatewayExecutionResult:
        runtime = self._employee_runtime
        terminal = runtime.terminal(attempt_id) if runtime is not None else None
        if terminal is None:
            return GatewayExecutionResult(
                GatewayExecutionStatus.ACTION_REQUIRED,
                safe_error_code="unknown_dispatch_outcome",
            )
        if terminal.status == "completed":
            if terminal.output:
                return GatewayExecutionResult(
                    GatewayExecutionStatus.COMPLETED,
                    output=terminal.output,
                )
            return GatewayExecutionResult(
                GatewayExecutionStatus.ACTION_REQUIRED,
                safe_error_code="employee_result_unavailable_after_restart",
            )
        status = {
            "timeout": GatewayExecutionStatus.TIMEOUT,
            "canceled": GatewayExecutionStatus.CANCELED,
            "action_required": GatewayExecutionStatus.ACTION_REQUIRED,
        }.get(terminal.status)
        if status is None:
            return GatewayExecutionResult(
                GatewayExecutionStatus.ACTION_REQUIRED,
                safe_error_code="unknown_dispatch_outcome",
            )
        return GatewayExecutionResult(
            status,
            safe_error_code=terminal.error_code or f"employee_actor_{status.value}",
        )

    def reconcile_terminal_snapshots(self) -> int:
        """Re-emit terminal lifecycle facts without executing ACP."""

        if self._attempt_lifecycle is None:
            return 0
        self._data.rebuild_projection()
        self._synchronize_gateway_from_journal()
        reconciled = 0
        for lifecycle in tuple(self._gateway_state.attempts.values()):
            if not lifecycle.terminal_status:
                continue
            payload = self._data.get_history_payload(lifecycle.history_record_id)
            status = GatewayExecutionStatus(lifecycle.terminal_status)
            result = GatewayExecutionResult(
                status=status,
                output=(
                    payload.result_text
                    if status is GatewayExecutionStatus.COMPLETED
                    else ""
                ),
                safe_error_code=(
                    ""
                    if status is GatewayExecutionStatus.COMPLETED
                    else payload.error_detail
                ),
            )
            if status is GatewayExecutionStatus.COMPLETED:
                self._publish_completed_artifacts(
                    lifecycle.binding,
                    result,
                    request_text=payload.request_text,
                )
            self._attempt_lifecycle.terminal(lifecycle.binding, result)
            reconciled += 1
        return reconciled

    def _publish_completed_artifacts(
        self,
        binding: DispatchBinding,
        result: GatewayExecutionResult,
        *,
        request_text: str,
    ) -> None:
        """Publish deterministic canonical documents before reporting success."""

        sink = self._data_sink
        if sink is None:
            raise EmployeeDispatchError("canonical employee data sink is unavailable")
        output = result.output[:100_000]
        common = {
            "agent_id": binding.agent_id,
            "tenant_key": binding.tenant_key,
            "owner_principal_id": binding.owner_principal_id,
            "idempotency_key": binding.attempt_id,
        }
        l1 = (
            "# Employee Memory\n\n"
            f"- Last completed task: {binding.task_id}\n"
            f"- Attempt: {binding.attempt_id}\n"
            f"- Tool: {binding.tool}\n"
            f"- Model: {binding.model}\n\n"
            "## Last Result\n\n"
            f"{output}"
        ).encode("utf-8")
        skill_profile = json.dumps(
            {
                "agent_id": binding.agent_id,
                "effort": binding.effort,
                "last_attempt_id": binding.attempt_id,
                "model": binding.model,
                "profile": binding.profile,
                "tool": binding.tool,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        reasoning = json.dumps(
            {
                "attempt_id": binding.attempt_id,
                "request_digest": hashlib.sha256(request_text.encode("utf-8")).hexdigest(),
                "result_digest": hashlib.sha256(result.output.encode("utf-8")).hexdigest(),
                "status": result.status.value,
                "task_id": binding.task_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        commands = (
            PublishEmployeeDocumentCommand(
                **common,
                kind=DataKind.L1_MEMORY,
                source_id=DataKind.L1_MEMORY.value,
                content=l1,
                content_type="text/markdown",
            ),
            PublishEmployeeDocumentCommand(
                **common,
                kind=DataKind.MEMORY_SUMMARY,
                source_id="",
                content=output.encode("utf-8"),
                content_type="text/markdown",
                chat_id=binding.chat_id,
                thread_root_id=binding.thread_root_id,
            ),
            PublishEmployeeDocumentCommand(
                **common,
                kind=DataKind.SKILL_PROFILE,
                source_id=DataKind.SKILL_PROFILE.value,
                content=skill_profile,
                content_type="application/json",
            ),
            PublishEmployeeDocumentCommand(
                **common,
                kind=DataKind.REASONING,
                source_id=binding.task_id,
                content=reasoning,
                content_type="application/json",
            ),
        )
        for command in commands:
            sink.publish_document(command)
        enqueue_knowledge = getattr(sink, "enqueue_knowledge_terminal", None)
        if callable(enqueue_knowledge):
            enqueue_knowledge(
                AuthenticatedExecutionTerminal(
                    attempt_id=binding.attempt_id,
                    status="completed",
                    request_text=request_text,
                    result_text=result.output,
                    error_detail="",
                )
            )

    def _recover_legacy_router_dispatches(self) -> int:
        """Dispose Router-only dispatches without ever re-running ACP.

        Older code could anchor ``router_dispatching`` without an attempt
        binding in that frame.  Its external outcome is unknowable, so the
        only safe recovery is a durable ``action_required`` terminal.
        """

        for _attempt in range(3):
            captured_head = self._presynchronize_domains()
            try:
                return self._recover_legacy_router_dispatches_once(captured_head)
            except _ProjectionHeadChanged:
                continue
        raise EmployeeDispatchError("employee recovery head remained unstable")

    def _recover_legacy_router_dispatches_once(
        self,
        captured_head: tuple[int, str],
    ) -> int:
        with team_runtime_guard(), ExitStack() as stack:
            stack.enter_context(self._projection_sync_lock)
            stack.enter_context(self._hire.employee_dispatch_guard())
            stack.enter_context(self._ingress.employee_dispatch_guard(router=self._router))
            stack.enter_context(self._data.employee_dispatch_guard())
            stack.enter_context(self._channels.employee_dispatch_guard())
            stack.enter_context(self._writer.transaction_guard())
            self._require_presynchronized_head(captured_head)
            legacy = tuple(
                record
                for record in self._router.state.by_acceptance_id.values()
                if record.state == "dispatching"
                and record.acceptance_id not in self._gateway_state.attempt_by_acceptance_id
            )
            if not legacy:
                return 0
            events = tuple(
                self._router.preflight_terminal_event_unlocked(
                    acceptance_id=record.acceptance_id,
                    reason_code=GatewayExecutionStatus.ACTION_REQUIRED.value,
                )
                for record in legacy
            )
            commit = self._commit_events_unlocked(events)
            self._apply_committed_frame_unlocked(commit.frame)
            return len(events)

    def _commit_events_unlocked(self, events: tuple[JournalEvent, ...]):
        versions = self._writer.get_aggregate_versions({event.aggregate_id for event in events})
        result = self._writer.commit(
            events,
            versions,
            expected_head_sequence=self._gateway_state.cursor_sequence,
            expected_head_hash=self._gateway_state.cursor_hash or None,
        )
        if result.state is not CommitState.ANCHORED:
            raise EmployeeDispatchError("employee dispatch frame was not anchored")
        return result

    def _apply_committed_frame_unlocked(self, frame) -> None:
        reduce_gateway_frame(self._gateway_state, frame)
        self._router.apply_committed_frame_unlocked(frame)
        self._data.apply_committed_frame_unlocked(frame)
        self._ingress.apply_committed_frame_unlocked(frame)
        apply_frame(self._hire.projection_state, frame)

    def _synchronize_gateway_from_journal(self) -> None:
        with self._projection_sync_lock:
            self._synchronize_gateway_unlocked()

    def _synchronize_gateway_unlocked(self) -> None:
        last = self._writer.get_last_frame()
        expected_sequence = 0 if last is None else last.sequence
        expected_hash = "" if last is None else last.frame_hash
        if (
            self._gateway_state.cursor_sequence == expected_sequence
            and self._gateway_state.cursor_hash == expected_hash
        ):
            return
        fresh = GatewayProjectionState()
        for frame in self._writer.replay():
            reduce_gateway_frame(fresh, frame)
        self._gateway_state = fresh

    def _build_binding(
        self,
        *,
        grant,
        employee,
        team_binding,
        rendered: RenderedEmployeePrompt,
        authority_connection: str,
    ) -> DispatchBinding:
        acceptance = grant.record.acceptance_id
        digest = hashlib.sha256(acceptance.encode()).hexdigest()
        authority = grant.record.authority
        assert authority is not None
        return DispatchBinding(
            schema_version=1,
            permit_id="prm_" + _stable_hash("permit", acceptance),
            attempt_id="att_" + _stable_hash("attempt", acceptance),
            acceptance_id=acceptance,
            ingress_aggregate_id=grant.record.aggregate_id,
            envelope_id=grant.record.envelope_id,
            payload_digest=grant.payload.payload_sha256,
            tenant_key=employee.tenant_key,
            agent_id=employee.agent_id,
            employee_version=employee.aggregate_version,
            owner_principal_id=employee.owner_principal_id,
            bot_principal_id=authority.bot_principal_id,
            app_id=authority.app_id,
            channel_generation=authority.channel_generation,
            ingress_connection_id=authority.connection_id,
            authority_connection_id=authority_connection,
            requester_principal_id=grant.request.requester_principal_id,
            task_id="task_" + digest,
            run_id="run_" + digest,
            message_id=grant.request.current_message_id,
            thread_root_id=grant.request.thread_root_message_id,
            thread_id=grant.request.feishu_thread_id,
            chat_id=grant.request.chat_id,
            team_identity=team_binding.engine_identity,
            team_chat_id=team_binding.chat_id,
            team_root_identity=team_binding.root_identity,
            tool=employee.tool,
            model=employee.model,
            profile=employee.profile,
            effort=employee.effort,
            security_profile="employee_v1",
            capabilities=employee.capabilities,
            permissions=employee.permissions,
            constraints_digest=grant.request.constraints_digest,
            system_prompt_token_reserve=grant.request.system_prompt_token_reserve,
            render_contract_digest=rendered.render_contract_digest,
            context_snapshot_hash=rendered.context_snapshot_hash,
            context_watermark_digest=rendered.context_watermark_digest,
            dispatch_committed_at=self._timestamp(),
        )

    def _history_models(
        self,
        binding: DispatchBinding,
        result: GatewayExecutionResult,
        *,
        request_text: str,
        ended_at: str,
    ) -> tuple[ExecutionHistoryRecordV1, ExecutionHistoryPayloadV1]:
        context = ExecutionAttemptContext(
            tenant_key=binding.tenant_key,
            agent_id=binding.agent_id,
            owner_principal_id=binding.owner_principal_id,
            requester_principal_id=binding.requester_principal_id,
            task_id=binding.task_id,
            run_id=binding.run_id,
            attempt_id=binding.attempt_id,
            message_id=binding.message_id,
            thread_root_id=binding.thread_root_id,
            chat_id=binding.chat_id,
            tool=binding.tool,
            model=binding.model,
            effort=binding.effort,
            started_at=binding.dispatch_committed_at,
        )
        head = self._data.get_head()
        category = (
            "timeout"
            if result.status is GatewayExecutionStatus.TIMEOUT
            else ("none" if result.status is GatewayExecutionStatus.COMPLETED else "unknown")
        )
        record = ExecutionHistoryRecordV1.from_attempt(
            context,
            ended_at=ended_at,
            status=result.status.value,
            safe_summary=SafeExecutionSummary.build(
                status=result.status.value,
                error_category=category,
            ),
            prompt_tokens=0,
            completion_tokens=0,
            predecessor_sequence=head.sequence,
            predecessor_hash=head.logical_hash,
            shard_timezone=self._data.shard_timezone,
        )
        payload = ExecutionHistoryPayloadV1(
            record_id=record.record_id,
            occurrence_key=record.occurrence_key,
            request_text=request_text,
            result_text=result.output,
            error_detail=result.safe_error_code,
        )
        return record, payload

    @staticmethod
    def _validate_employee(employee, grant) -> None:
        authority = grant.record.authority
        if (
            employee is None
            or employee.state is not EmployeeState.ACTIVE
            or employee.worker_type is not WorkerType.VISIBLE
            or grant.request.chat_id not in employee.member_groups
            or authority is None
            or employee.aggregate_version != authority.employee_version
            or (employee.tool, employee.model, employee.effort) != (authority.tool, authority.model, authority.effort)
        ):
            raise EmployeeDispatchError("employee dispatch authority is stale")

    @staticmethod
    def _validate_channel(status, grant) -> str:
        authority = grant.record.authority
        ready = getattr(status, "ready_metadata", None)
        connection = ready.get("connection_id") if isinstance(ready, Mapping) else None
        if (
            authority is None
            or status is None
            or status.state is not ChannelProcessState.READY
            or status.agent_id != authority.agent_id
            or status.app_id != authority.app_id
            or status.tenant_key != authority.tenant_key
            or status.bot_principal_id != authority.bot_principal_id
            or status.generation != authority.channel_generation
            or not isinstance(connection, str)
            or not connection
        ):
            raise EmployeeDispatchError("employee channel authority is stale")
        return connection

    @staticmethod
    def _validate_context_watermark(request, snapshot) -> None:
        watermark = snapshot.watermark
        if (
            watermark is None
            or watermark.tenant_key != request.tenant_key
            or watermark.chat_id != request.chat_id
            or watermark.thread_root_id != request.thread_root_message_id
            or watermark.feishu_thread_id != request.feishu_thread_id
            or snapshot.constraints_digest != request.constraints_digest
            or snapshot.system_prompt_tokens_reserved != request.system_prompt_token_reserve
        ):
            raise EmployeeDispatchError("employee Context watermark is stale")

    def _timestamp(self) -> str:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise EmployeeDispatchError("coordinator clock must return aware datetime")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _stable_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def _result_digest(result: GatewayExecutionResult) -> str:
    encoded = json.dumps(
        {
            "status": result.status.value,
            "output": result.output,
            "safe_error_code": result.safe_error_code,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "EmployeeCancellationOutcome",
    "EmployeeDispatchCoordinator",
    "EmployeeDispatchError",
    "FinalizedEmployeeAttempt",
    "PreparedEmployeeDispatch",
]
