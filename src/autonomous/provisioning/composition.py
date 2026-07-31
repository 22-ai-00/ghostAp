"""Single-owner production composition for durable visible employee hiring."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import inspect
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import lark_oapi as lark

from src.slock_engine.memory_manager import (
    MemoryManager,
    default_slock_storage_base,
)
from src.utils.async_helpers import safe_wait_for
from src.utils.path import canonicalize_user_home_path

from ...trust.models import ActorKind, EffectiveTrust, TrustZone
from ...trust.registry import ManagedGroupRegistry
from ...trust.resolver import TrustZoneResolver
from ..acceptance.main_bot_audit import MainBotSendAuditLog
from ..acceptance.release_trust import ReleaseTrustProvider
from ..context.group_ledger import GroupContextLedger, GroupEventPayload
from ..context.lark_source import LarkEmployeeMessageSourceFactory
from ..context.models import ContextUnavailableError, ThreadContextConfig
from ..context.runtime import (
    RuntimeEmployeeGenerationAuthority,
    parse_requester_acl,
)
from ..context.service import AuthorizedGroupMemoryReader, EmployeeContextService
from ..context.source import EmployeeMessageSourceFactory
from ..data.composition import (
    EmployeeDataComposition,
    build_employee_data_composition,
)
from ..data.keyring import EmployeeDataKeyring
from ..data.models import DataKind
from ..data.ports import HistoryQuerySpec, MemoryQuerySpec
from ..data.query import AuthenticatedDataRequest, EmployeeDataSubject, QueryDeniedError
from ..domain import EmployeeState, WorkerType
from ..gateway.coordinator import EmployeeDispatchCoordinator
from ..gateway.env_scope import (
    EmployeeEnvironmentAuthority,
    EmployeeProcessEnvironmentMaterial,
)
from ..ingress.attachments import AttachmentStagingService
from ..ingress.models import EmployeeIngressMetadata, EmployeeIngressPayload
from ..ingress.projection import IngressProjectionState
from ..ingress.router import DurableEmployeeIngressRouter, RouterQueueLimits
from ..ingress.service import EmployeeIngressService, IngressConflictError
from ..journal.anchor import FileAnchor
from ..journal.frame import JournalEvent
from ..journal.projections import ProjectionState
from ..journal.writer import CommitState, JournalWriter
from ..manager.cards import EmployeeRuntimeCardView
from ..membership import (
    EmployeeMembershipService,
    LarkMembershipAPI,
    MembershipBindingError,
)
from ..outbox.delivery import (
    EmployeeDeliveryAuthority,
    EmployeeOutboxDeliveryCoordinator,
)
from ..outbox.lifecycle import EmployeeOutboxLifecycle
from ..outbox.models import employee_outbox_id
from ..outbox.projection import OutboxProjectionState
from ..outbox.service import EmployeeOutboxService
from ..supervisor.employee_channels import (
    ChannelProcessState,
    EmployeeChannelSupervisor,
)
from ..team import (
    EmployeeTeamService,
    SessionCoordinatorDecisionProvider,
    TeamAttemptResult,
    TeamTarget,
)
from ..workforce.credential_vault import CredentialReceipt, CredentialVault
from ..workforce.registry import ProjectedAgentRegistry
from .fire_authority import JournalFireAuthority
from .fire_effects import (
    AtomicEmployeeArchive,
    ChannelStopEffect,
    CredentialDestroyEffect,
    ExecutionQuiesceEffect,
    MembershipCleanupEffect,
    SlashCleanupEffect,
)
from .fire_service import EmployeeFireService
from .hire_service import HireReadiness, ProductionEmployeeHireService
from .hire_state import (
    DurableHireState,
    HireEffectState,
    HirePhase,
)
from .lark_app import LarkAppRegistrar
from .local_bootstrap import resolve_employee_runtime_material
from .notification_state import (
    HireNotificationPhase,
    hire_notification_message_uuid,
    rebuild_hire_notification_projection,
)
from .slash_commands import SlashCommandReconciler, VerifiedSlashState
from .slash_lark import LarkSlashCommandAPI
from .verification import (
    ChannelVerificationEvidence,
    SlashVerificationEvidence,
    TenantIngressEvidence,
    VerificationBinding,
    VerificationChallenge,
    VerificationCoordinates,
    VerificationOutcome,
    VerificationRouter,
)

logger = logging.getLogger(__name__)

_RECOVERY_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)
_CHANNEL_RECONNECT_GRACE_SECONDS = 180.0


def _stable_feishu_uuid(value: str) -> str:
    """Return a deterministic key within Feishu's 50-character UUID limit."""

    return hashlib.sha256(value.encode()).hexdigest()[:50]


def _bound_remote_coordinates(
    metadata: EmployeeIngressMetadata,
    part: Mapping[str, object],
) -> tuple[str, str, str] | None:
    """Recover raw Feishu coordinates from the encrypted, hash-bound payload."""

    values: list[str] = []
    for indexed, field, prefix in (
        (metadata.chat_id, "remote_chat_id", "oc_"),
        (metadata.message_id, "remote_message_id", "om_"),
        (metadata.thread_root_message_id, "remote_root_id", "om_"),
    ):
        raw = part.get(field)
        if raw is None:
            # Internal team assignments retain raw coordinates in metadata.
            # SDK ingress always writes all three encrypted raw fields.
            if indexed.startswith(prefix) and len(indexed) == len(prefix) + 64:
                return None
            values.append(indexed)
            continue
        if not isinstance(raw, str):
            return None
        if not raw:
            if indexed:
                return None
            values.append("")
            continue
        if prefix + hashlib.sha256(raw.encode()).hexdigest() != indexed:
            return None
        values.append(raw)
    return (values[0], values[1], values[2])


class _ChannelSupervisor(Protocol):
    def start(
        self,
        agent_id: str,
        app_id: str,
        credential_ref: str,
        generation: int,
        on_event: Callable[[dict[str, Any]], None],
    ) -> Any: ...

    def status(self, agent_id: str) -> Any: ...

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

    def close(self) -> None: ...


class _RuntimeTeamBackend:
    """Adapt TeamRun work to the canonical employee ingress/gateway pipeline."""

    def __init__(
        self,
        runtime: "EmployeeDepartmentRuntime",
        notify: Callable[[str, str, str], object],
    ) -> None:
        self._runtime = runtime
        self._notify = notify

    def list_active(self, tenant_key: str, chat_id: str) -> tuple[TeamTarget, ...]:
        runtime = self._runtime
        service = runtime._require_service()
        ready_agent_ids = runtime._team_execution_ready_agent_ids(
            tenant_key,
            chat_id,
        )
        projection = service.synchronize_projection()
        targets: list[TeamTarget] = []
        for state in service.list_states():
            employee = projection.employees.get(state.agent_id)
            status = runtime._channels.status(state.agent_id) if runtime._channels else None
            if (
                state.phase is HirePhase.ACTIVE
                and employee is not None
                and employee.tenant_key == tenant_key
                and chat_id in employee.member_groups
                and getattr(status, "state", None) is ChannelProcessState.READY
                and getattr(status, "generation", None) == state.channel_generation
                and employee.agent_id in ready_agent_ids
            ):
                targets.append(
                    TeamTarget(
                        employee.agent_id,
                        employee.name,
                        employee.role,
                        tuple(employee.capabilities),
                        "ready",
                        0,
                    )
                )
        return tuple(sorted(targets, key=lambda item: item.agent_id))

    def submit(
        self,
        *,
        run_id: str,
        step_id: str,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
        deadline_at: str,
    ) -> str:
        runtime = self._runtime
        service = runtime._require_service()
        ingress = runtime._ingress
        channels = runtime._channels
        if ingress is None or channels is None:
            raise RuntimeError("team employee ingress is unavailable")
        state = next(
            (
                candidate
                for candidate in service.list_states()
                if candidate.agent_id == target.agent_id
                and candidate.tenant_key == tenant_key
                and candidate.phase is HirePhase.ACTIVE
            ),
            None,
        )
        status = channels.status(target.agent_id)
        ready_metadata = getattr(status, "ready_metadata", None)
        connection_id = (
            ready_metadata.get("connection_id")
            if isinstance(ready_metadata, Mapping)
            else ""
        )
        if (
            state is None
            or getattr(status, "state", None) is not ChannelProcessState.READY
            or getattr(status, "generation", None) != state.channel_generation
            or not isinstance(connection_id, str)
            or not connection_id
        ):
            raise RuntimeError("team employee channel authority is unavailable")
        stable = hashlib.sha256(f"{run_id}\0{step_id}\0{target.agent_id}".encode()).hexdigest()
        payload = EmployeeIngressPayload(
            schema_version=1,
            envelope_id=f"ing_{stable}",
            normalized_parts=(
                {
                    "type": "team_assignment",
                    "message_type": "text",
                    "chat_type": "group",
                    "content": instruction,
                    "team_instruction": instruction,
                    "sender_id": requester_principal_id,
                    "sender_id_type": "open_id",
                    "sender_type": "user",
                    "sender_tenant_key": tenant_key,
                    "feishu_thread_id": "",
                    "team_run_id": run_id,
                    "team_step_id": step_id,
                    "team_deadline_at": deadline_at,
                },
            ),
            attachment_descriptors=(),
        )
        metadata = EmployeeIngressMetadata(
            schema_version=1,
            envelope_id=payload.envelope_id,
            tenant_key=tenant_key,
            agent_id=state.agent_id,
            bot_principal_id=state.bot_principal_id,
            app_id=state.app_id,
            channel_generation=state.channel_generation,
            connection_id=connection_id,
            event_id=f"evt_{stable}",
            message_id=message_id,
            event_type="ghostap.team.assignment.v1",
            action_identity=f"team:{run_id}:{step_id}",
            chat_id=chat_id,
            thread_root_message_id=message_id,
            sender_principal_id=requester_principal_id,
            received_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            semantic_digest=payload.payload_sha256,
            payload_sha256=payload.payload_sha256,
            payload_size_bytes=payload.canonical_size_bytes,
            attachment_count=0,
            attachment_total_bytes=0,
        )
        ack = ingress.accept(metadata, payload, request_id=f"req_{stable}")
        ledger = getattr(runtime, "_group_ledger", None)
        if ledger is not None:
            ledger.publish(
                tenant_key=tenant_key,
                chat_id=chat_id,
                thread_id="",
                message_id=message_id,
                transport_principal_id="main_bot",
                transport_event_id=f"evt_{stable}",
                payload=GroupEventPayload(
                    sender_id=requester_principal_id,
                    sender_id_type="open_id",
                    sender_type="user",
                    sender_tenant_key=tenant_key,
                    text=instruction,
                    timestamp=time.time(),
                ),
                causal_event_id=f"{run_id}:{step_id}",
            )
        return ack.acceptance.acceptance_id

    def result(self, acceptance_id: str) -> TeamAttemptResult | None:
        runtime = self._runtime
        dispatch = runtime._dispatch
        if dispatch is None:
            return TeamAttemptResult(
                "action_required",
                error_code="team_gateway_unavailable",
                retry_allowed=False,
            )
        try:
            snapshot = dispatch.team_attempt_result(acceptance_id)
        except Exception:
            return TeamAttemptResult(
                "action_required",
                error_code="team_gateway_unavailable",
                retry_allowed=False,
            )
        return None if snapshot is None else TeamAttemptResult(
            snapshot.status,
            output=snapshot.output,
            history_record_id=snapshot.history_record_id,
            error_code=snapshot.error_code,
            retry_allowed=True,
        )

    def publish_collaboration(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        agent_id: str,
        team_run_id: str,
        assignment_id: str,
        acceptance_id: str,
    ) -> str:
        """Bind one completed assignment to its employee-Bot Outbox publication."""

        runtime = self._runtime
        dispatch = runtime._dispatch
        outbox = runtime._outbox
        delivery = runtime._outbox_delivery
        ingress = runtime._ingress
        if dispatch is None or outbox is None or delivery is None or ingress is None:
            raise RuntimeError("team collaboration transport is unavailable")
        dispatch_state = dispatch.state
        attempt_id = dispatch_state.attempt_by_acceptance_id.get(acceptance_id)
        lifecycle = dispatch_state.attempts.get(attempt_id or "")
        binding = getattr(lifecycle, "binding", None)
        if (
            binding is None
            or binding.attempt_id != attempt_id
            or binding.acceptance_id != acceptance_id
            or binding.tenant_key != tenant_key
            or binding.agent_id != agent_id
            or binding.chat_id != chat_id
        ):
            raise RuntimeError("team collaboration attempt authority is invalid")
        payload = ingress.get_payload(acceptance_id)
        if len(payload.normalized_parts) != 1:
            raise RuntimeError("team collaboration payload is invalid")
        part = payload.normalized_parts[0]
        expected_step = assignment_id.rsplit(":", 1)[-1]
        if (
            not isinstance(part, Mapping)
            or part.get("team_run_id") != team_run_id
            or part.get("team_step_id") != expected_step
            or assignment_id != f"{team_run_id}:assignment:{expected_step}"
        ):
            raise RuntimeError("team collaboration coordinates are invalid")
        outbox_id = employee_outbox_id(tenant_key, agent_id, attempt_id)
        snapshot = outbox.get_snapshot(outbox_id)
        if not snapshot.state.terminal:
            raise RuntimeError("team collaboration Outbox is not terminal")
        publication = delivery.deliver(outbox_id)
        message_id = getattr(publication, "message_id", "")
        if not isinstance(message_id, str) or not message_id:
            raise RuntimeError("team collaboration delivery is unavailable")
        causal_event_id = "collab_" + hashlib.sha256(
            "\0".join(
                (outbox_id, message_id, team_run_id, assignment_id)
            ).encode()
        ).hexdigest()
        if not outbox.record_collaboration_publication(
            outbox_id=outbox_id,
            team_run_id=team_run_id,
            assignment_id=assignment_id,
            causal_event_id=causal_event_id,
        ):
            raise RuntimeError("team collaboration publication conflicts")
        return causal_event_id

    def submit_direct(
        self,
        *,
        target: TeamTarget,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        requester_principal_id: str,
        instruction: str,
    ) -> str:
        runtime = self._runtime
        service = runtime._require_service()
        ingress = runtime._ingress
        channels = runtime._channels
        if ingress is None or channels is None:
            raise RuntimeError("direct employee ingress is unavailable")
        state = next(
            (
                candidate
                for candidate in service.list_states()
                if candidate.agent_id == target.agent_id
                and candidate.tenant_key == tenant_key
                and candidate.phase is HirePhase.ACTIVE
            ),
            None,
        )
        status = channels.status(target.agent_id)
        ready_metadata = getattr(status, "ready_metadata", None)
        connection_id = (
            ready_metadata.get("connection_id")
            if isinstance(ready_metadata, Mapping)
            else ""
        )
        if (
            state is None
            or getattr(status, "state", None) is not ChannelProcessState.READY
            or getattr(status, "generation", None) != state.channel_generation
            or not isinstance(connection_id, str)
            or not connection_id
        ):
            raise RuntimeError("direct employee channel authority is unavailable")
        stable = hashlib.sha256(
            f"direct\0{tenant_key}\0{chat_id}\0{message_id}\0{target.agent_id}".encode()
        ).hexdigest()
        payload = EmployeeIngressPayload(
            schema_version=1,
            envelope_id=f"ing_{stable}",
            normalized_parts=(
                {
                    "type": "message",
                    "message_type": "text",
                    "chat_type": "group",
                    "content": {"text": instruction},
                    "sender_id": requester_principal_id,
                    "sender_id_type": "open_id",
                    "sender_type": "user",
                    "sender_tenant_key": tenant_key,
                    "feishu_thread_id": "",
                },
            ),
            attachment_descriptors=(),
        )
        metadata = EmployeeIngressMetadata(
            schema_version=1,
            envelope_id=payload.envelope_id,
            tenant_key=tenant_key,
            agent_id=state.agent_id,
            bot_principal_id=state.bot_principal_id,
            app_id=state.app_id,
            channel_generation=state.channel_generation,
            connection_id=connection_id,
            event_id=f"evt_{stable}",
            message_id=message_id,
            event_type="im.message.receive_v1",
            action_identity=f"direct:{stable}",
            chat_id=chat_id,
            thread_root_message_id=message_id,
            sender_principal_id=requester_principal_id,
            received_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            semantic_digest=payload.payload_sha256,
            payload_sha256=payload.payload_sha256,
            payload_size_bytes=payload.canonical_size_bytes,
            attachment_count=0,
            attachment_total_bytes=0,
        )
        ack = ingress.accept(metadata, payload, request_id=f"req_{stable}")
        return ack.acceptance.acceptance_id

    def cancel(
        self,
        acceptance_id: str,
        *,
        run_id: str,
        step_id: str,
    ) -> TeamAttemptResult:
        dispatch = self._runtime._dispatch
        if dispatch is None:
            return TeamAttemptResult(
                "action_required",
                error_code="team_gateway_unavailable",
                retry_allowed=False,
            )
        try:
            outcome = dispatch.request_team_cancel(
                acceptance_id=acceptance_id,
                team_run_id=run_id,
                team_step_id=step_id,
            )
        except Exception:
            return TeamAttemptResult(
                "action_required",
                error_code="team_gateway_unavailable",
                retry_allowed=False,
            )
        if outcome.status in {"cancel_requested", "already_terminal"}:
            # A retry is safe only after the canceled attempt has a durable
            # Gateway terminal frame. Give the live interruption a bounded
            # window to finish; absence of a terminal remains non-retryable.
            terminal_deadline = time.monotonic() + 5.0
            while True:
                try:
                    terminal = dispatch.team_attempt_result(acceptance_id)
                except Exception:
                    return TeamAttemptResult(
                        "action_required",
                        error_code="team_gateway_unavailable",
                        retry_allowed=False,
                    )
                if terminal is not None:
                    return TeamAttemptResult(
                        terminal.status,
                        output=terminal.output,
                        history_record_id=terminal.history_record_id,
                        error_code=terminal.error_code,
                        retry_allowed=True,
                    )
                if time.monotonic() >= terminal_deadline:
                    break
                time.sleep(0.01)
            return TeamAttemptResult(
                "canceled",
                error_code="team_step_timeout",
                retry_allowed=False,
            )
        return TeamAttemptResult(
            "action_required",
            error_code="team_cancel_failed",
            retry_allowed=False,
        )

    def notify(
        self,
        message_id: str,
        chat_id: str,
        result: str,
        *,
        idempotency_key: str = "",
        tenant_key: str = "",
        requester_principal_id: str = "",
    ) -> None:
        try:
            signature = inspect.signature(self._notify)
        except (TypeError, ValueError):
            if idempotency_key:
                self._notify(message_id, chat_id, result, idempotency_key)
                return
            self._notify(message_id, chat_id, result)
            return

        base_args = (message_id, chat_id, result)
        candidates: list[tuple[tuple[object, ...], dict[str, str]]] = []
        fixed_positional = tuple(
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        )
        explicit_idempotency_parameter = signature.parameters.get(
            "idempotency_key"
        )
        prefer_legacy_positional_key = (
            idempotency_key
            and len(fixed_positional) >= 4
            and (
                explicit_idempotency_parameter is None
                or explicit_idempotency_parameter.kind
                is inspect.Parameter.POSITIONAL_ONLY
            )
        )
        if tenant_key and requester_principal_id:
            scoped_kwargs = {
                "tenant_key": tenant_key,
                "requester_principal_id": requester_principal_id,
            }
            if idempotency_key:
                if len(fixed_positional) >= 6:
                    candidates.append(
                        (
                            (
                                *base_args,
                                idempotency_key,
                                tenant_key,
                                requester_principal_id,
                            ),
                            {},
                        )
                    )
                if prefer_legacy_positional_key:
                    candidates.append(
                        ((*base_args, idempotency_key), scoped_kwargs)
                    )
                candidates.extend(
                    (
                        (
                            base_args,
                            {
                                **scoped_kwargs,
                                "idempotency_key": idempotency_key,
                            },
                        ),
                    )
                )
                if not prefer_legacy_positional_key:
                    candidates.append(
                        ((*base_args, idempotency_key), scoped_kwargs)
                    )
            candidates.append((base_args, scoped_kwargs))
        if idempotency_key:
            candidates.extend(
                (
                    (base_args, {"idempotency_key": idempotency_key}),
                    ((*base_args, idempotency_key), {}),
                )
            )
        candidates.append((base_args, {}))
        for args, kwargs in candidates:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            self._notify(*args, **kwargs)
            return
        raise TypeError("unsupported employee team notification callback signature")


class _SlashReconciler(Protocol):
    async def reconcile(self) -> VerifiedSlashState: ...

    async def cleanup(self) -> VerifiedSlashState: ...

    async def observe_empty(self) -> bool: ...


SlashReconcilerFactory = Callable[[str, str], _SlashReconciler]
MainBotSendAudit = Callable[[str, str, float, float], int]
EmployeeEnvironmentProvider = Callable[
    [EmployeeEnvironmentAuthority],
    EmployeeProcessEnvironmentMaterial,
]


class _SlockMembershipHealth:
    """Treat an unavailable or ambiguous activated Slock as degraded."""

    def __init__(self, manager: object) -> None:
        self._manager = manager

    def is_degraded(self, _agent_id: str, team_id: str) -> bool:
        try:
            self._manager.resolve_employee_engine(chat_id=team_id)
        except Exception:
            return True
        return False


@dataclass(frozen=True)
class RuntimeReadiness:
    ready: bool
    blockers: tuple[str, ...]


class EmployeeDepartmentRuntime:
    """Own Journal, Vault, Saga, Channel children and the activity loop."""

    def __init__(
        self,
        *,
        blockers: tuple[str, ...] = (),
        runtime_enabled: bool = False,
        managed_group_registry: ManagedGroupRegistry | None = None,
        managed_group_owner_id: str = "",
    ) -> None:
        self._blockers = blockers
        self._runtime_enabled = runtime_enabled is True
        self._service: ProductionEmployeeHireService | None = None
        self._writer: JournalWriter | None = None
        self._vault: CredentialVault | None = None
        self._data_keyring: EmployeeDataKeyring | None = None
        self._channels: _ChannelSupervisor | None = None
        self._data: EmployeeDataComposition | None = None
        self._ingress: EmployeeIngressService | None = None
        self._router: DurableEmployeeIngressRouter | None = None
        self._attachments: AttachmentStagingService | None = None
        self._dispatch: EmployeeDispatchCoordinator | None = None
        self._outbox: EmployeeOutboxService | None = None
        self._outbox_delivery: EmployeeOutboxDeliveryCoordinator | None = None
        self._outbox_lifecycle: EmployeeOutboxLifecycle | None = None
        self._membership: EmployeeMembershipService | None = None
        self._fire: EmployeeFireService | None = None
        self._team: EmployeeTeamService | None = None
        self._dispatch_thread: threading.Thread | None = None
        self._dispatch_stop = threading.Event()
        self._execution_blockers: tuple[str, ...] = ()
        self._slock_manager: object | None = None
        self._environment_provider: EmployeeEnvironmentProvider | None = None
        self._context_source_factory: EmployeeMessageSourceFactory | None = None
        self._context_service: EmployeeContextService | None = None
        self._group_ledger: GroupContextLedger | None = None
        self._context_acl: Any = None
        self._group_memory_backend: Any = None
        self._owns_group_memory_backend = False
        self._context_blockers: tuple[str, ...] = ()
        self._context_bindings: dict[str, tuple[str, str, int]] = {}
        self._context_projection_invalidations: set[str] = set()
        self._context_explicit_invalidations: set[str] = set()
        self._context_binding_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._slash_factory: SlashReconcilerFactory | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        self._futures: set[concurrent.futures.Future[Any]] = set()
        self._intent_futures: dict[str, concurrent.futures.Future[Any]] = {}
        self._future_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._closing = False
        self._challenges: dict[str, VerificationChallenge] = {}
        self._verification_router: VerificationRouter | None = None
        self._main_bot_send_audit: MainBotSendAudit | None = None
        self._main_bot_activation_audit: MainBotSendAuditLog | None = None
        self._notification_status: Callable[[DurableHireState, str], object] | None = None
        self._owned_main_bot_send_audit: MainBotSendAuditLog | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._activation_lock = threading.RLock()  # leaf lock: never held while acquiring a LockLevel lock
        self._notification_async_lock: asyncio.Lock | None = None
        self._automatic_activation = False
        self._core_recovered = False
        self._recovery_trace: list[str] = []
        self._managed_group_registry = managed_group_registry
        self._managed_group_owner_id = managed_group_owner_id

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        registrar: Any = None,
        channel_supervisor: _ChannelSupervisor | None = None,
        slash_reconciler_factory: SlashReconcilerFactory | None = None,
        main_bot_send_audit: MainBotSendAudit | None = None,
        release_trust_provider: ReleaseTrustProvider | None = None,
        notification_link: Callable[[DurableHireState, str, int], object] | None = None,
        notification_status: Callable[[DurableHireState, str], object] | None = None,
        team_notification: Callable[..., object] | None = None,
        context_source_factory: EmployeeMessageSourceFactory | None = None,
        group_memory_backend: Any = None,
        slock_engine_manager: object | None = None,
        employee_environment_provider: EmployeeEnvironmentProvider | None = None,
        membership_health: Any = None,
        manager_client_factory: Callable[[], Any] | None = None,
        recover_immediately: bool = True,
        managed_group_registry: ManagedGroupRegistry | None = None,
        managed_group_owner_id: str = "",
    ) -> EmployeeDepartmentRuntime:
        limit = getattr(settings, "autonomous_visible_employee_limit", 0)
        if limit == 0:
            if release_trust_provider is not None:
                release_trust_provider.close()
            return cls(
                blockers=("visible_employee_limit",),
                managed_group_registry=managed_group_registry,
                managed_group_owner_id=managed_group_owner_id,
            )
        if notification_link is None:
            if release_trust_provider is not None:
                release_trust_provider.close()
            return cls(
                blockers=("registration_notifier",),
                managed_group_registry=managed_group_registry,
                managed_group_owner_id=managed_group_owner_id,
            )
        if release_trust_provider is not None:
            try:
                release_trust_provider.close()
            except Exception:
                logger.warning("unused employee release provider close failed")

        runtime = cls(
            runtime_enabled=True,
            managed_group_registry=managed_group_registry,
            managed_group_owner_id=managed_group_owner_id,
        )
        try:
            material = resolve_employee_runtime_material(settings)
            credential_root = canonicalize_user_home_path(
                settings.autonomous_credential_dir
            )
            vault = CredentialVault(credential_root, material.credential_keyring)
            writer = JournalWriter.open(
                Path(settings.autonomous_journal_dir).expanduser(),
                anchor=FileAnchor(settings.autonomous_anchor_path),
                hmac_key=material.journal_hmac_key,
                writer_epoch=time.time_ns(),
            )
        except Exception as exc:
            logger.error(
                "employee department durable composition unavailable: %s",
                type(exc).__name__,
            )
            try:
                vault.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            return cls(blockers=("durable_configuration",))

        runtime._writer = writer
        runtime._vault = vault
        runtime._data_keyring = material.data_keyring
        runtime._slock_manager = slock_engine_manager
        runtime._environment_provider = employee_environment_provider
        owned_main_bot_send_audit: MainBotSendAuditLog | None = None
        if main_bot_send_audit is None:
            try:
                owned_main_bot_send_audit = MainBotSendAuditLog.open(
                    settings.autonomous_main_bot_audit_dir,
                    anchor_path=settings.autonomous_main_bot_audit_anchor_path,
                    hmac_key=material.journal_hmac_key,
                )
                main_bot_send_audit = owned_main_bot_send_audit.count_target_attempts
            except Exception as exc:
                logger.error(
                    "local main Bot send audit unavailable: %s",
                    type(exc).__name__,
                )
                try:
                    writer.close()
                finally:
                    vault.close()
                return cls(blockers=("main_bot_send_audit",))
        runtime._owned_main_bot_send_audit = owned_main_bot_send_audit
        audit_owner = getattr(main_bot_send_audit, "__self__", None)
        runtime._main_bot_activation_audit = (
            owned_main_bot_send_audit
            if owned_main_bot_send_audit is not None
            else audit_owner
            if isinstance(audit_owner, MainBotSendAuditLog)
            else None
        )
        runtime._notification_status = notification_status
        runtime._automatic_activation = (
            getattr(settings, "autonomous_employee_auto_activation", False) is True
        )
        try:
            runtime._start_loop()
            runtime._compose_execution_storage(settings)
            runtime._channels = channel_supervisor or EmployeeChannelSupervisor(
                secret_resolver=vault.resolve,
                ingress_service=runtime._ingress,
                ingress_binding_resolver=(runtime._resolve_ingress_binding if runtime._ingress is not None else None),
                ingress_ack_timeout=getattr(
                    settings,
                    "autonomous_employee_ingress_ack_timeout_seconds",
                    1.5,
                ),
            )
            runtime._slash_factory = slash_reconciler_factory or cls._default_slash_factory
            runtime._main_bot_send_audit = main_bot_send_audit
            service = ProductionEmployeeHireService(
                writer,
                ProjectionState(),
                visible_employee_limit=limit,
                release_evidence_ready=True,
                credential_keyring_ready=True,
                registrar=registrar or LarkAppRegistrar(),
                credential_vault=vault,
                on_registration_link=notification_link,
                on_registration_status=notification_status,
                provisioning_submitter=runtime._submit_intent,
                manifest_reauthorization_submitter=(
                    runtime._submit_manifest_reauthorization
                ),
                runtime_recovery_ready=False,
                workspace_projector=(
                    runtime._data.workspace_projector
                    if runtime._data is not None
                    else None
                ),
            )
            runtime._service = service
            runtime._verification_router = VerificationRouter(nonce_consumer=service)
            runtime._compose_membership(
                settings,
                manager_client_factory=manager_client_factory,
            )
            runtime._compose_context(
                settings,
                context_source_factory=context_source_factory,
                group_memory_backend=group_memory_backend,
            )
            runtime._compose_dispatch(
                settings,
                membership_health=membership_health or runtime._membership,
            )
            runtime._compose_fire(settings)
            if team_notification is not None and runtime._writer is not None:
                team_runtime_mode = getattr(
                    settings,
                    "autonomous_team_runtime_mode",
                    "coordinator",
                )
                decision_provider = None
                if team_runtime_mode == "coordinator":
                    if slock_engine_manager is None:
                        raise RuntimeError("team coordinator project resolver is unavailable")

                    def resolve_coordinator_cwd(run: object) -> str:
                        binding = slock_engine_manager.resolve_employee_engine(
                            chat_id=run.chat_id
                        )
                        return binding.canonical_root

                    decision_provider = SessionCoordinatorDecisionProvider(
                        tool=getattr(
                            settings, "autonomous_team_coordinator_tool", "coco"
                        ),
                        model=getattr(
                            settings, "autonomous_team_coordinator_model", ""
                        ),
                        profile=getattr(
                            settings, "autonomous_team_coordinator_profile", ""
                        ),
                        effort=getattr(
                            settings, "autonomous_team_coordinator_effort", ""
                        ),
                        cwd_resolver=resolve_coordinator_cwd,
                    )
                runtime._team = EmployeeTeamService(
                    writer=runtime._writer,
                    backend=_RuntimeTeamBackend(runtime, team_notification),
                    attempt_timeout_seconds=float(
                        settings.autonomous_team_step_timeout_seconds
                    ),
                    runtime_mode=team_runtime_mode,
                    blob_store=(
                        runtime._ingress.blob_store
                        if runtime._ingress is not None
                        else None
                    ),
                    active_key_id=(
                        runtime._data_keyring.active_key_id
                        if runtime._data_keyring is not None
                        else ""
                    ),
                    coordinator_tool=getattr(
                        settings, "autonomous_team_coordinator_tool", "coco"
                    ),
                    coordinator_model=getattr(
                        settings, "autonomous_team_coordinator_model", ""
                    ),
                    coordinator_profile=getattr(
                        settings, "autonomous_team_coordinator_profile", ""
                    ),
                    coordinator_effort=getattr(
                        settings, "autonomous_team_coordinator_effort", ""
                    ),
                    coordinator_decision_provider=decision_provider,
                )
            if recover_immediately:
                runtime.recover()
            return runtime
        except Exception as exc:
            logger.error(
                "employee department runtime composition unavailable: %s",
                type(exc).__name__,
            )
            runtime.close()
            return cls(blockers=("runtime_composition",))

    @property
    def hire_service(self) -> ProductionEmployeeHireService | None:
        return self._service

    @property
    def context_service(self) -> EmployeeContextService | None:
        return self._context_service

    @property
    def membership_service(self) -> EmployeeMembershipService | None:
        return self._membership

    @property
    def fire_service(self) -> EmployeeFireService | None:
        return self._fire

    @property
    def team_service(self) -> EmployeeTeamService | None:
        return self._team

    @property
    def data_composition(self) -> EmployeeDataComposition | None:
        return self._data

    @property
    def ingress_service(self) -> EmployeeIngressService | None:
        return self._ingress

    @property
    def ingress_router(self) -> DurableEmployeeIngressRouter | None:
        return self._router

    @property
    def dispatch_coordinator(self) -> EmployeeDispatchCoordinator | None:
        return self._dispatch

    @property
    def outbox_service(self) -> EmployeeOutboxService | None:
        return self._outbox

    @property
    def outbox_delivery(self) -> EmployeeOutboxDeliveryCoordinator | None:
        return self._outbox_delivery

    @property
    def main_bot_outbound_audit(self) -> MainBotSendAuditLog | None:
        return self._main_bot_activation_audit

    def readiness(self) -> RuntimeReadiness:
        return self.hire_readiness()

    def list_employee_runtime_statuses(
        self,
        tenant_key: str,
    ) -> tuple[EmployeeRuntimeCardView, ...]:
        """Return secret-free runtime views without exposing projections to handlers."""

        service = self._require_service()
        projection = service.synchronize_projection()
        employees = sorted(
            (
                employee
                for employee in projection.employees.values()
                if employee.tenant_key == tenant_key
                and employee.worker_type is WorkerType.VISIBLE
                and employee.state is not EmployeeState.ARCHIVED
            ),
            key=lambda employee: (employee.name.casefold(), employee.agent_id),
        )
        return tuple(self._employee_runtime_view(employee) for employee in employees)

    def get_employee_runtime_status(
        self,
        tenant_key: str,
        agent_id: str,
    ) -> EmployeeRuntimeCardView:
        for view in self.list_employee_runtime_statuses(tenant_key):
            if view.agent_id == agent_id:
                return view
        raise KeyError(agent_id)

    def recycle_employee_session(
        self,
        tenant_key: str,
        agent_id: str,
        *,
        reason: str = "admin_requested",
    ) -> EmployeeRuntimeCardView:
        self._assert_employee_tenant(tenant_key, agent_id)
        runtime = self._dispatch.employee_runtime if self._dispatch is not None else None
        if runtime is None:
            raise RuntimeError("employee actor runtime is unavailable")
        runtime.recycle(agent_id, reason)
        return self.get_employee_runtime_status(tenant_key, agent_id)

    def rebuild_employee_workspace(
        self,
        tenant_key: str,
        agent_id: str,
    ) -> EmployeeRuntimeCardView:
        self._assert_employee_tenant(tenant_key, agent_id)
        projector = self._data.workspace_projector if self._data is not None else None
        if projector is None:
            raise RuntimeError("employee workspace projector is unavailable")
        projector.rebuild(tenant_key, agent_id)
        return self.get_employee_runtime_status(tenant_key, agent_id)

    def lint_employee_knowledge(self, tenant_key: str, agent_id: str) -> object:
        self._assert_employee_tenant(tenant_key, agent_id)
        knowledge = self._data.knowledge_service if self._data is not None else None
        if knowledge is None:
            raise RuntimeError("employee knowledge service is unavailable")
        return knowledge.lint(tenant_key, agent_id)

    def retry_employee_knowledge_review(
        self,
        tenant_key: str,
        agent_id: str,
        review_id: str,
    ) -> str:
        self._assert_employee_tenant(tenant_key, agent_id)
        knowledge = self._data.knowledge_service if self._data is not None else None
        if knowledge is None:
            raise RuntimeError("employee knowledge service is unavailable")
        return knowledge.retry_review_item(
            review_id,
            tenant_key=tenant_key,
            agent_id=agent_id,
        )

    def _assert_employee_tenant(self, tenant_key: str, agent_id: str) -> object:
        projection = self._require_service().synchronize_projection()
        employee = projection.employees.get(agent_id)
        if employee is None or employee.tenant_key != tenant_key:
            raise KeyError(agent_id)
        return employee

    def _employee_runtime_view(self, employee: object) -> EmployeeRuntimeCardView:
        agent_id = str(getattr(employee, "agent_id"))
        channel = self._channels.status(agent_id) if self._channels is not None else None
        bot_state = (
            str(getattr(getattr(channel, "state", None), "value", "stopped"))
            if channel is not None
            else "stopped"
        )
        bot_generation = int(getattr(channel, "generation", 0) or 0)
        runtime = self._dispatch.employee_runtime if self._dispatch is not None else None
        actor = runtime.inspect(agent_id) if runtime is not None else None
        actor_state = (
            str(getattr(getattr(actor, "status", None), "value", "ready_cold"))
            if actor is not None
            else "legacy_one_shot"
        )
        mailbox_depth = int(getattr(actor, "mailbox_depth", 0) or 0)
        active_assignment_id = str(
            getattr(actor, "active_assignment_id", "") or ""
        )
        active_run_id = ""
        checkpoint = ""
        context_quality = "complete"
        warnings: list[str] = []
        if self._dispatch is not None:
            attempts = [
                attempt
                for attempt in self._dispatch.state.attempts.values()
                if attempt.binding.agent_id == agent_id and not attempt.terminal_status
            ]
            if attempts:
                attempt = max(attempts, key=lambda item: item.dispatch_sequence)
                active_assignment_id = active_assignment_id or attempt.binding.task_id
                active_run_id = attempt.binding.run_id
                checkpoint = f"journal:{attempt.dispatch_sequence}"
                for frame in self.journal_frames():
                    for event in frame.events:
                        if (
                            event.event_type == "context.warning.recorded"
                            and event.payload.get("attempt_id")
                            == attempt.binding.attempt_id
                        ):
                            context_quality = str(
                                event.payload.get("quality") or "canonical_partial"
                            )
                            warnings.append(str(event.payload.get("code") or "source_unavailable"))
        knowledge_generation = 0
        review_item_ids: tuple[str, ...] = ()
        if self._data is not None:
            data_state = self._data.service.rebuild_projection()
            knowledge_generation = max(
                (
                    document.version
                    for document in data_state.employee_documents.values()
                    if document.agent_id == agent_id
                    and document.kind
                    in {DataKind.KNOWLEDGE_PAGE, DataKind.KNOWLEDGE_INDEX}
                    and not document.tombstoned
                ),
                default=0,
            )
            if self._data.knowledge_service is not None:
                review_item_ids = self._data.knowledge_service.list_review_items(
                    str(getattr(employee, "tenant_key", "")),
                    agent_id,
                )
        employee_state = str(getattr(getattr(employee, "state", None), "value", ""))
        can_accept = (
            employee_state == EmployeeState.ACTIVE.value
            and bot_state == ChannelProcessState.READY.value
            and actor_state not in {"degraded", "stopping", "stopped"}
            and not self._execution_blockers
        )
        return EmployeeRuntimeCardView(
            agent_id=agent_id,
            name=str(getattr(employee, "name", agent_id)),
            emoji=str(getattr(employee, "emoji", "🤖")),
            role=str(getattr(employee, "role", "")),
            tool=str(getattr(employee, "tool", "")),
            model=str(getattr(employee, "model", "")),
            employee_state=employee_state,
            bot_state=bot_state,
            bot_generation=bot_generation,
            actor_state=actor_state,
            mailbox_depth=mailbox_depth,
            can_accept=can_accept,
            identity_version=int(getattr(employee, "aggregate_version", 0)),
            knowledge_generation=knowledge_generation,
            active_assignment_id=active_assignment_id,
            active_run_id=active_run_id,
            last_checkpoint=checkpoint,
            context_quality=context_quality,
            context_warnings=tuple(dict.fromkeys(warnings)),
            review_item_ids=review_item_ids,
        )

    def hire_readiness(self) -> RuntimeReadiness:
        if self._service is None:
            return RuntimeReadiness(False, self._blockers or ("not_composed",))
        activation_audit = self._main_bot_activation_audit
        if (
            activation_audit is None
            or activation_audit.activation_fence_ready is not True
        ):
            return RuntimeReadiness(False, ("main_bot_activation_fence",))
        service_readiness: HireReadiness = self._service.readiness()
        return RuntimeReadiness(service_readiness.ready, service_readiness.blockers)

    def execution_readiness(
        self,
        agent_id: str | None = None,
        *,
        chat_id: str | None = None,
    ) -> RuntimeReadiness:
        """Probe ACTIVE employee Context without weakening hire readiness."""
        if chat_id is not None and (
            not isinstance(chat_id, str) or not chat_id.strip()
        ):
            return RuntimeReadiness(False, ("context_group_history",))
        readiness, projection, active = self._prepare_execution_probe(agent_id)
        if not readiness.ready or not active:
            return readiness
        for state in active:
            employee_readiness = self._probe_employee_execution(
                projection,
                state,
                chat_id=chat_id,
            )
            if not employee_readiness.ready:
                return employee_readiness
        return RuntimeReadiness(True, ())

    def _team_execution_ready_agent_ids(
        self,
        tenant_key: str,
        chat_id: str,
    ) -> frozenset[str]:
        """Probe one team snapshot once, then isolate per-employee failures."""

        if not isinstance(chat_id, str) or not chat_id.strip():
            return frozenset()
        readiness, projection, active = self._prepare_execution_probe()
        if not readiness.ready:
            return frozenset()
        ready: set[str] = set()
        for state in active:
            if state.tenant_key != tenant_key:
                continue
            employee = projection.employees.get(state.agent_id)
            if employee is None or chat_id not in employee.member_groups:
                continue
            employee_readiness = self._probe_employee_execution(
                projection,
                state,
                chat_id=chat_id,
                probe_remote=False,
            )
            if employee_readiness.ready:
                ready.add(state.agent_id)
        return frozenset(ready)

    def _prepare_execution_probe(
        self,
        agent_id: str | None = None,
    ) -> tuple[RuntimeReadiness, Any, tuple[DurableHireState, ...]]:
        """Synchronize the common readiness boundary once per probe batch."""

        projection = None
        active: tuple[DurableHireState, ...] = ()
        hire = self.hire_readiness()
        if not hire.ready:
            return hire, projection, active
        if self._execution_blockers:
            return RuntimeReadiness(False, self._execution_blockers), projection, active
        if (
            self._ingress is None
            or self._router is None
            or self._dispatch is None
            or self._outbox is None
            or self._outbox_delivery is None
            or self._data is None
            or self._context_service is None
        ):
            return RuntimeReadiness(False, ("employee_gateway",)), projection, active
        if self._channels is None or not callable(getattr(self._channels, "update_card", None)):
            return RuntimeReadiness(False, ("employee_outbox",)), projection, active
        if self._service is None:
            return RuntimeReadiness(False, ("not_composed",)), projection, active
        try:
            projection = self._service.synchronize_projection()
            active = tuple(
                state
                for state in self._service.list_states()
                if state.phase is HirePhase.ACTIVE
                and (agent_id is None or state.agent_id == agent_id)
            )
            if not active:
                blocker = ("employee_not_active",) if agent_id is not None else ()
                return RuntimeReadiness(not blocker, blocker), projection, active
            if self._context_source_factory is None:
                return (
                    RuntimeReadiness(
                        False,
                        self._context_blockers or ("employee_context",),
                    ),
                    projection,
                    active,
                )
            if not getattr(self._context_acl, "configured", False):
                return (
                    RuntimeReadiness(False, ("context_request_authority",)),
                    projection,
                    active,
                )
            self._data.service.rebuild_projection()
            projection = self._service.synchronize_projection()
            if not self._refresh_context_bindings(projection):
                return (
                    RuntimeReadiness(False, ("context_binding_sync",)),
                    projection,
                    active,
                )
            head = self._data.service.get_head()
            if (
                head.sequence != projection.cursor_sequence
                or head.logical_hash != projection.cursor_hash
            ):
                return (
                    RuntimeReadiness(False, ("context_projection_stale",)),
                    projection,
                    active,
                )
            return RuntimeReadiness(True, ()), projection, active
        except Exception:
            return RuntimeReadiness(False, ("employee_context",)), projection, active

    def _probe_employee_execution(
        self,
        projection: Any,
        state: DurableHireState,
        *,
        chat_id: str | None,
        probe_remote: bool = True,
    ) -> RuntimeReadiness:
        if chat_id is not None and (
            not isinstance(chat_id, str) or not chat_id.strip()
        ):
            return RuntimeReadiness(False, ("context_group_history",))
        try:
            employee = projection.employees.get(state.agent_id)
            principal = projection.bot_principals.get(state.bot_principal_id)
            status = self._channels.status(state.agent_id) if self._channels else None
            status_state = getattr(status, "state", None)
            if (
                employee is None
                or principal is None
                or employee.bot_principal_id != state.bot_principal_id
                or principal.agent_id != state.agent_id
                or principal.tenant_key != state.tenant_key
                or principal.app_id != state.app_id
                or not principal.credential_ref
            ):
                return RuntimeReadiness(False, ("context_binding",))
            if (
                status_state is not ChannelProcessState.READY
                or getattr(status, "generation", None) != state.channel_generation
                or getattr(status, "identity", {}).get("app_id") != state.app_id
                or getattr(status, "ready_metadata", {}).get("connection_id")
                != state.channel_connection_id
            ):
                return RuntimeReadiness(False, ("context_generation",))
            if chat_id is not None and chat_id not in employee.member_groups:
                return RuntimeReadiness(False, ("context_group_membership",))
            # Team admission is based on anchored workforce membership and the
            # current READY Channel binding.  Live Context probes run when the
            # task is dispatched, where failures remain fail-closed, but they
            # must not block chat routing or overwrite durable group status.
            if not probe_remote:
                return RuntimeReadiness(True, ())
            if self._context_source_factory.probe(principal) is not True:
                return RuntimeReadiness(False, ("context_credentials",))
            probe_group_history = getattr(
                self._context_source_factory,
                "probe_group_history",
                None,
            )
            groups_to_probe = (
                (chat_id,)
                if chat_id is not None
                else tuple(sorted(employee.member_groups))
            )
            if groups_to_probe and not callable(probe_group_history):
                return RuntimeReadiness(False, ("context_group_history",))
            for group_id in groups_to_probe:
                if probe_group_history(principal, group_id) is not True:
                    return RuntimeReadiness(False, ("context_group_history",))
            return RuntimeReadiness(True, ())
        except Exception:
            return RuntimeReadiness(False, ("employee_context",))

    def recover(self) -> None:
        """Replay first, then resume only recoverable durable phases."""
        if self._service is None:
            return
        self._core_recovered = False
        self._recovery_trace.clear()
        self._recovery_trace.append("journal_projection")
        projection = self._service.recover()
        pending_manifest_reauthorizations = (
            self._service.recover_manifest_reauthorizations()
        )
        if self._membership is not None:
            self._membership.rebuild_projection()
            if self._runtime_enabled:
                try:
                    self._membership.recover_pending()
                except Exception as exc:
                    logger.error(
                        "employee membership recovery failed closed: %s",
                        type(exc).__name__,
                    )
                    self._execution_blockers = ("membership_recovery",)
        if self._data is not None:
            try:
                self._recovery_trace.append("data_projection")
                self._recover_employee_data(self._service.projection_state)
                self._recovery_trace.append("workspace_projection")
            except Exception as exc:
                logger.error(
                    "employee data recovery failed closed: %s",
                    type(exc).__name__,
                )
                self._execution_blockers = ("employee_data_recovery",)
        if not self._execution_blockers and self._group_ledger is not None:
            try:
                self._recovery_trace.append("group_ledger")
                self._group_ledger.rebuild_projection()
            except Exception as exc:
                logger.error(
                    "employee group ledger recovery failed closed: %s",
                    type(exc).__name__,
                )
                self._execution_blockers = ("group_ledger_recovery",)
        employee_runtime = (
            self._dispatch.employee_runtime if self._dispatch is not None else None
        )
        if not self._execution_blockers and employee_runtime is not None:
            try:
                self._recovery_trace.append("actor_mailboxes")
                employee_runtime.recover()
            except Exception as exc:
                logger.error(
                    "employee actor recovery failed closed: %s",
                    type(exc).__name__,
                )
                self._execution_blockers = ("employee_actor_recovery",)
        execution_recovered = not self._execution_blockers
        if execution_recovered:
            try:
                assert self._ingress is not None
                assert self._router is not None
                assert self._dispatch is not None
                assert self._outbox is not None
                self._ingress.rebuild_projection()
                self._reconcile_recovered_activation_required_ingress()
                self._router.rebuild_projection()
                self._outbox.rebuild_projection()
                self._dispatch.recover_incomplete_attempts()
                self._dispatch.reconcile_terminal_snapshots()
                self._router.recover_terminal_attachments()
                self._reconcile_terminal_ingress()
                self._ingress.gc_terminal_payloads()
            except Exception as exc:
                logger.error(
                    "employee execution recovery failed closed: %s",
                    type(exc).__name__,
                )
                self._execution_blockers = ("employee_recovery",)
                execution_recovered = False
        if execution_recovered and self._team is not None:
            try:
                self._recovery_trace.append("team_coordinator")
                recovered_team_runs = self._team.recover()
                if recovered_team_runs:
                    logger.warning(
                        "employee team recovered %d interrupted run(s)",
                        recovered_team_runs,
                    )
            except Exception as exc:
                logger.error(
                    "employee team recovery failed closed: %s",
                    type(exc).__name__,
                )
                self._execution_blockers = ("team_recovery",)
                execution_recovered = False
        if self._fire is not None and self._runtime_enabled:
            try:
                self._fire.recover()
            except Exception as exc:
                logger.error(
                    "employee retirement recovery failed closed: %s",
                    type(exc).__name__,
                )
                self._execution_blockers = ("fire_recovery",)
                execution_recovered = False
        if not self._refresh_context_bindings(self._service.projection_state):
            self._context_blockers = ("context_binding_sync",)
        self._core_recovered = not self._execution_blockers and not self._context_blockers
        if not self._runtime_enabled:
            self._service.mark_runtime_recovered()
            return
        pending_intents: list[str] = []
        status_reply_intents: list[str] = []
        for state in projection.states.values():
            effect_types = dict(state.effect_types)
            unresolved_phase_reply = any(
                effect_state
                in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
                and effect_types.get(effect_id) == "employee_status_reply"
                and self._phase_status_decision(
                    dict(state.metadata_for(effect_id)).get("error_code", "")
                )
                is not None
                for effect_id, effect_state in state.effects
            )
            if (
                unresolved_phase_reply
                and state.phase
                in {
                    HirePhase.VALIDATING,
                    HirePhase.READY_PENDING_VERIFICATION,
                    HirePhase.ACTION_REQUIRED,
                }
                and state.credential_ref
                and state.channel_generation > 0
            ):
                if state.phase is HirePhase.READY_PENDING_VERIFICATION:
                    self._service.begin_channel_revalidation(
                        state.intent_id,
                        observed_generation=state.channel_generation,
                    )
                status_reply_intents.append(state.intent_id)
                continue
            if (
                state.phase
                in {
                    HirePhase.CONFIGURING,
                    HirePhase.VALIDATING,
                    HirePhase.READY_PENDING_VERIFICATION,
                    HirePhase.ACTIVE,
                }
                and state.credential_ref
                and state.channel_generation > 0
            ):
                self._service.begin_channel_revalidation(
                    state.intent_id,
                    observed_generation=state.channel_generation,
                )
                pending_intents.append(state.intent_id)
            if state.phase in {
                HirePhase.PROVISIONING_APP,
                HirePhase.STORING_CREDENTIAL,
                HirePhase.CONFIGURING,
                HirePhase.VALIDATING,
            }:
                pending_intents.append(state.intent_id)
        recover_notifications = (
            self._notification_status is not None
            and any(
                self._terminal_notification_status(state)
                for state in projection.states.values()
            )
        )
        for operation_id in pending_manifest_reauthorizations:
            self._submit_manifest_reauthorization(operation_id)
        if pending_intents or status_reply_intents or recover_notifications:
            self._recovery_trace.append("employee_channels")
            self._submit_coroutine(
                self._recover_runtime(
                    list(dict.fromkeys(pending_intents)),
                    list(dict.fromkeys(status_reply_intents)),
                ),
                label="recovery",
            )
        else:
            self._recovery_trace.extend(("employee_channels", "admission_open"))
            self._service.mark_runtime_recovered()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._start_monitor_in_loop)
            if execution_recovered:
                self._start_dispatch_worker()

    def journal_frames(self) -> tuple[Any, ...]:
        return tuple(self._writer.replay()) if self._writer is not None else ()

    @property
    def recovery_trace(self) -> tuple[str, ...]:
        return tuple(self._recovery_trace)

    def record_group_event(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        thread_id: str,
        message_id: str,
        sender_id: str,
        text: str,
        transport_principal_id: str = "main_bot",
    ) -> bool:
        """Publish one main-Bot observation into the canonical group ledger."""

        ledger = self._group_ledger
        if ledger is None or not all((tenant_key, chat_id, message_id, sender_id)):
            if self._runtime_enabled:
                raise RuntimeError("employee group ledger is unavailable")
            return False
        ledger.publish(
            tenant_key=tenant_key,
            chat_id=chat_id,
            thread_id=thread_id,
            message_id=message_id,
            transport_principal_id=transport_principal_id,
            transport_event_id=f"main:{message_id}",
            payload=GroupEventPayload(
                sender_id=sender_id,
                sender_id_type="open_id",
                sender_type="user",
                sender_tenant_key=tenant_key,
                text=text,
                timestamp=time.time(),
            ),
        )
        return True

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        errors: list[str] = []

        def cleanup(label: str, action: Callable[[], Any]) -> bool:
            try:
                action()
                return True
            except Exception as exc:
                errors.append(f"{label}:{type(exc).__name__}")
                return False

        if self._ingress is not None:
            cleanup("ingress_admission", self._ingress.stop_admission)
        if self._outbox is not None:
            cleanup("outbox_admission", self._outbox.stop_admission)
        if self._service is not None:
            cleanup("hire_admission", self._service.stop_admission)
        if self._context_service is not None:
            cleanup("context_admission", self._context_service.stop_admission)
        team_safe = True
        if self._team is not None:
            team_safe = cleanup("team_service", self._team.close)
        self._dispatch_stop.set()
        dispatch_safe = True
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)
            dispatch_safe = not self._dispatch_thread.is_alive()
        dispatch_close = getattr(self._dispatch, "close", None)
        if dispatch_safe and callable(dispatch_close):
            dispatch_safe = cleanup("employee_actors", dispatch_close)
        with self._future_lock:
            futures = tuple(self._futures)
        activities_safe = True
        if futures:
            _done, pending = concurrent.futures.wait(futures, timeout=5.0)
            for future in pending:
                future.cancel()
            activities_safe = not pending
        if self._loop is not None and self._loop.is_running():
            quiesce = asyncio.run_coroutine_threadsafe(
                self._quiesce_loop(),
                self._loop,
            )
            activities_safe = cleanup("activity_loop", lambda: quiesce.result(timeout=5.0)) and activities_safe
        context_safe = True
        if self._context_service is not None:
            context_safe = cleanup("context_drain", self._context_service.drain)
        if self._context_source_factory is not None:
            context_safe = cleanup("context_sources", self._context_source_factory.close) and context_safe
        outbox_safe = True
        if self._outbox is not None and self._outbox_delivery is not None:
            def drain_outbox() -> None:
                for _attempt in range(10_000):
                    if not self._drain_employee_outbox_once():
                        return
                raise RuntimeError("employee Outbox did not drain")

            outbox_safe = cleanup("outbox_drain", drain_outbox)
        resources_safe = (
            team_safe
            and dispatch_safe
            and activities_safe
            and context_safe
            and outbox_safe
        )
        if resources_safe:
            if self._channels is not None:
                resources_safe = cleanup("channels", self._channels.close)
        if resources_safe and self._owns_group_memory_backend and self._group_memory_backend is not None:
            resources_safe = cleanup(
                "group_memory",
                self._group_memory_backend.shutdown,
            )
        if resources_safe and self._attachments is not None:
            resources_safe = cleanup("attachments", self._attachments.close)
        if resources_safe and self._ingress is not None:
            resources_safe = cleanup("ingress", self._ingress.close)
        if resources_safe and self._outbox is not None:
            resources_safe = cleanup("outbox", self._outbox.close)
        if resources_safe and self._data is not None:
            resources_safe = cleanup("data", self._data.close)
        if resources_safe:
            if self._service is not None:
                resources_safe = cleanup(
                    "hire_service",
                    self._service.close,
                )
            elif self._writer is not None:
                resources_safe = cleanup("writer", self._writer.close)
        if resources_safe and self._vault is not None:
            resources_safe = cleanup("vault", self._vault.close)
        if resources_safe and self._owned_main_bot_send_audit is not None:
            resources_safe = cleanup(
                "main_bot_send_audit",
                self._owned_main_bot_send_audit.close,
            )
        if not resources_safe:
            errors.append("dependent_resources_held")
        if resources_safe and self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if resources_safe and self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        if errors:
            logger.error("employee runtime close errors: %s", ",".join(errors))
        if not resources_safe:
            self._closing = False

    def invalidate_employee_context(self, agent_id: str) -> None:
        with self._context_binding_lock:
            if self._context_source_factory is not None:
                self._context_source_factory.invalidate_employee(agent_id)
            self._context_explicit_invalidations.add(agent_id)
            self._context_bindings.pop(agent_id, None)

    def reactivate_employee_context(self, agent_id: str) -> None:
        """Re-open admission after a durable replacement binding is installed."""
        with self._context_binding_lock:
            if self._context_source_factory is not None:
                self._context_source_factory.reactivate_employee(agent_id)
            self._context_explicit_invalidations.discard(agent_id)
            self._context_projection_invalidations.discard(agent_id)

    def _compose_execution_storage(self, settings: Any) -> None:
        """Compose the data, durable Inbox, and attachment owners."""

        if (
            not self._runtime_enabled
            or getattr(
                settings,
                "autonomous_visible_employee_limit",
                0,
            )
            == 0
        ):
            self._execution_blockers = ("employee_ingress",)
            return
        if self._writer is None or self._vault is None or self._data_keyring is None:
            self._execution_blockers = ("employee_ingress",)
            return
        try:
            legacy_base = str(
                canonicalize_user_home_path(
                    getattr(settings, "autonomous_slock_storage_base", None)
                    or default_slock_storage_base()
                )
            )
            self._data = build_employee_data_composition(
                settings=settings,
                writer=self._writer,
                keyring=self._data_keyring,
                admin_principal_ids=frozenset(getattr(settings, "admin_user_ids", ()) or ()),
                main_bot_app_id=getattr(settings, "app_id", ""),
                agents_root=Path(legacy_base).expanduser() / "agents",
                legacy_base=legacy_base,
                subject_resolver=self._resolve_data_subject,
                auto_cutover=False,
            )
            self._ingress = EmployeeIngressService.from_keyring(
                writer=self._writer,
                ingress_state=IngressProjectionState(),
                keyring=self._data_keyring,
                blob_root=getattr(
                    settings,
                    "autonomous_employee_ingress_blob_dir",
                ),
            )
            self._group_ledger = GroupContextLedger(
                writer=self._writer,
                blob_store=self._ingress.blob_store,
                active_key_id=self._data_keyring.active_key_id,
                config=ThreadContextConfig(),
                blob_retainer=self._ingress.retain_shared_blob,
                blob_releaser=self._ingress.release_shared_blob,
            )
            self._outbox = EmployeeOutboxService.from_keyring(
                writer=self._writer,
                outbox_state=OutboxProjectionState(),
                keyring=self._data_keyring,
                blob_root=getattr(
                    settings,
                    "autonomous_employee_outbox_blob_dir",
                ),
            )
            self._attachments = AttachmentStagingService(
                writer=self._writer,
                root=getattr(
                    settings,
                    "autonomous_employee_attachment_staging_dir",
                ),
                credential_resolver=self._vault,
                download_timeout_seconds=getattr(
                    settings,
                    "autonomous_context_fetch_timeout_seconds",
                    30.0,
                ),
            )
            self._execution_blockers = ()
        except Exception as exc:
            logger.error(
                "employee execution storage composition unavailable: %s",
                type(exc).__name__,
                exc_info=True,
            )
            if self._attachments is not None:
                try:
                    self._attachments.close()
                except Exception:
                    pass
            if self._ingress is not None:
                try:
                    self._ingress.close()
                except Exception:
                    pass
            if self._outbox is not None:
                try:
                    self._outbox.close()
                except Exception:
                    pass
            if self._data is not None:
                try:
                    self._data.close()
                except Exception:
                    pass
            self._attachments = None
            self._ingress = None
            self._group_ledger = None
            self._outbox = None
            self._data = None
            self._execution_blockers = ("employee_ingress",)

    def _resolve_data_subject(
        self,
        tenant_key: str,
        agent_id: str,
    ) -> EmployeeDataSubject | None:
        service = self._service
        if service is None:
            return None
        projection = service.synchronize_projection()
        employee = projection.employees.get(agent_id)
        if employee is None or employee.tenant_key != tenant_key:
            return None
        return EmployeeDataSubject(
            tenant_key=employee.tenant_key,
            agent_id=employee.agent_id,
            owner_principal_id=employee.owner_principal_id,
            member_groups=tuple(employee.member_groups),
        )

    def _recover_employee_data(self, projection: ProjectionState) -> None:
        data = self._data
        if data is None:
            return
        data.service.rebuild_projection()
        if data.state.data_authority.mode == "legacy":
            from ..migration.slock_data_importer import SlockDataImporter

            legacy_base = data.memory_facade.legacy_base_path
            if legacy_base is None:
                raise RuntimeError("legacy data base is unavailable")
            with data.service.legacy_import_scope():
                for employee in sorted(
                    projection.employees.values(),
                    key=lambda item: item.agent_id,
                ):
                    if employee.state is EmployeeState.ARCHIVED:
                        continue
                    result = SlockDataImporter(
                        service=data.service,
                        legacy_base=legacy_base,
                        tenant_key=employee.tenant_key,
                        owner_principal_id=employee.owner_principal_id,
                    ).import_employee(employee.agent_id)
                    if result.errors:
                        raise RuntimeError("legacy employee data import failed")
            data.service.cutover_to_canonical()
        data.rebuild_all()
        if data.knowledge_service is not None:
            data.knowledge_service.recover()

    def _compose_membership(
        self,
        settings: Any,
        *,
        manager_client_factory: Callable[[], Any] | None,
    ) -> None:
        """Compose real Bot membership using manager mutation and employee observation."""

        if (
            not self._runtime_enabled
            or self._writer is None
            or self._service is None
            or self._vault is None
            or not callable(manager_client_factory)
            or self._slock_manager is None
        ):
            self._membership = None
            return

        def employee_client_provider(
            agent_id: str,
            app_id: str,
            credential_ref: str,
        ) -> Any:
            app_secret = self._vault.resolve(credential_ref, agent_id, app_id)
            return lark.Client.builder().app_id(app_id).app_secret(app_secret).build()

        def team_owner(chat_id: str) -> str:
            try:
                getter = getattr(self._slock_manager, "get_activated_engine", None)
                engine = getter(chat_id) if callable(getter) else None
                channel = getattr(engine, "channel", None)
                return str(getattr(channel, "owner_id", "") or "")
            except Exception:
                return ""

        try:
            remote = LarkMembershipAPI(
                manager_client_factory(),
                employee_client_provider=employee_client_provider,
            )
            self._membership = EmployeeMembershipService(
                writer=self._writer,
                hire_service=self._service,
                remote=remote,
                admin_principal_ids=frozenset(
                    getattr(settings, "admin_user_ids", ()) or ()
                ),
                team_owner_resolver=team_owner,
                team_active_resolver=lambda chat_id: bool(
                    self._slock_manager.get_activated_engine(chat_id)
                ),
            )
        except Exception as exc:
            logger.error(
                "employee membership composition unavailable: %s",
                type(exc).__name__,
            )
            self._membership = None

    def _compose_fire(self, settings: Any) -> None:
        """Compose the one-way Journal-backed employee retirement workflow."""

        if (
            self._writer is None
            or self._service is None
            or self._ingress is None
            or self._vault is None
            or self._channels is None
            or self._slash_factory is None
            or self._loop is None
        ):
            self._fire = None
            return

        def run_async(coroutine: Any) -> Any:
            if self._loop is None or self._closing:
                if hasattr(coroutine, "close"):
                    coroutine.close()
                raise RuntimeError("employee runtime is closing")
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            return future.result(
                timeout=float(
                    getattr(settings, "autonomous_context_fetch_timeout_seconds", 30.0)
                )
            )

        legacy_base = str(
            canonicalize_user_home_path(
                getattr(settings, "autonomous_slock_storage_base", None)
                or default_slock_storage_base()
            )
        )
        authority = JournalFireAuthority(
            writer=self._writer,
            hire_service=self._service,
            ingress_service=self._ingress,
            admin_principal_ids=frozenset(
                getattr(settings, "admin_user_ids", ()) or ()
            ),
        )
        self._fire = EmployeeFireService(
            writer=self._writer,
            authority=authority,
            effects={
                "execution_quiesce": ExecutionQuiesceEffect(
                    self._dispatch,
                    grace_seconds=float(
                        getattr(settings, "autonomous_fire_grace_seconds", 5.0)
                    ),
                ),
                "slash_cleanup": SlashCleanupEffect(
                    reconciler_factory=self._slash_factory,
                    credential_resolver=self._vault.resolve,
                    async_runner=run_async,
                ),
                "channel_stop": ChannelStopEffect(self._channels),
                "membership_cleanup": MembershipCleanupEffect(
                    self._membership,
                    self._service,
                ),
                "credential_destroy": CredentialDestroyEffect(self._vault),
                "archive_move": AtomicEmployeeArchive(
                    Path(legacy_base).expanduser() / "agents"
                ),
            },
        )

    def _resolve_ingress_binding(self, agent_id: str, app_id: str) -> tuple[str, str]:
        service = self._service
        if service is None:
            raise RuntimeError("employee workforce projection is unavailable")
        projection = service.synchronize_projection()
        employee = projection.employees.get(agent_id)
        if employee is None or employee.bot_principal_id == "":
            raise RuntimeError("employee ingress binding is unavailable")
        principal = projection.bot_principals.get(employee.bot_principal_id)
        if principal is None or principal.agent_id != agent_id or principal.app_id != app_id:
            raise RuntimeError("employee ingress principal is unavailable")
        return employee.tenant_key, principal.bot_principal_id

    def _compose_dispatch(self, settings: Any, *, membership_health: Any) -> None:
        if self._execution_blockers:
            return
        if (
            self._service is None
            or self._writer is None
            or self._ingress is None
            or self._data is None
            or self._channels is None
            or self._context_service is None
            or self._outbox is None
        ):
            self._execution_blockers = ("employee_gateway",)
            return
        if self._slock_manager is None or not callable(getattr(self._slock_manager, "employee_activation_guard", None)):
            self._execution_blockers = ("slock_gateway",)
            return
        if not callable(self._environment_provider):
            self._execution_blockers = ("employee_environment",)
            return
        try:
            legacy_base = str(
                canonicalize_user_home_path(
                    getattr(settings, "autonomous_slock_storage_base", None)
                    or default_slock_storage_base()
                )
            )

            def registry_provider() -> ProjectedAgentRegistry:
                assert self._service is not None
                return ProjectedAgentRegistry(
                    self._service.synchronize_projection(),
                    storage_base_path=legacy_base,
                )

            health = membership_health or _SlockMembershipHealth(self._slock_manager)
            constraints_digest = hashlib.sha256(b"ghostap.employee-execution-constraints.v1").hexdigest()
            self._router = DurableEmployeeIngressRouter(
                writer=self._writer,
                ingress_service=self._ingress,
                registry_provider=registry_provider,
                channel_status_provider=self._channels,
                requester_acl=self._context_acl,
                queue_limits=RouterQueueLimits(
                    per_employee=getattr(
                        settings,
                        "autonomous_employee_queue_per_employee_limit",
                    ),
                    per_team=getattr(
                        settings,
                        "autonomous_employee_queue_per_team_limit",
                    ),
                    global_limit=getattr(
                        settings,
                        "autonomous_employee_queue_global_limit",
                    ),
                ),
                membership_health=health,
                attachment_staging=self._attachments,
                constraints_digest=constraints_digest,
                system_prompt_token_reserve=getattr(
                    settings,
                    "autonomous_employee_system_prompt_token_reserve",
                ),
                context_retry_base_seconds=getattr(
                    settings,
                    "autonomous_context_retry_base_seconds",
                ),
                context_retry_max_seconds=getattr(
                    settings,
                    "autonomous_context_retry_max_seconds",
                ),
                managed_group_registry_provider=(
                    (lambda: self._managed_group_registry)
                    if type(self._managed_group_registry) is ManagedGroupRegistry
                    else None
                ),
                managed_group_owner_id=self._managed_group_owner_id,
                employee_bot_ids_provider=self.trusted_employee_bot_open_ids,
                requester_principal_resolver=(
                    self._resolve_employee_requester_principal
                ),
            )
            outbox_lifecycle = EmployeeOutboxLifecycle(self._outbox)
            self._dispatch = EmployeeDispatchCoordinator(
                writer=self._writer,
                hire_service=self._service,
                ingress_service=self._ingress,
                router=self._router,
                data_service=self._data.service,
                data_sink=self._data,
                channel_supervisor=self._channels,
                slock_manager=self._slock_manager,
                context_service=self._context_service,
                environment_provider=self._environment_provider,
                registry_factory=lambda state: ProjectedAgentRegistry(
                    state,
                    storage_base_path=legacy_base,
                ),
                attempt_lifecycle=outbox_lifecycle,
                admin_principal_ids=frozenset(
                    getattr(settings, "admin_user_ids", ()) or ()
                ),
                team_owner_resolver=lambda chat_id: str(
                    getattr(
                        getattr(
                            self._slock_manager.get_activated_engine(chat_id),
                            "channel",
                            None,
                        ),
                        "owner_id",
                        "",
                    )
                    or ""
                ),
                timeout_seconds=getattr(
                    settings,
                    "autonomous_team_step_timeout_seconds",
                ),
                employee_runtime_mode=getattr(
                    settings,
                    "autonomous_employee_runtime_mode",
                    "actor",
                ),
                employee_session_idle_ttl_seconds=getattr(
                    settings,
                    "autonomous_employee_session_idle_ttl_seconds",
                    900.0,
                ),
                shadow_observer=(
                    self._record_employee_shadow_observation
                    if getattr(
                        settings,
                        "autonomous_employee_runtime_mode",
                        "actor",
                    )
                    == "shadow"
                    else None
                ),
            )
            self._outbox_lifecycle = outbox_lifecycle
            self._outbox_delivery = EmployeeOutboxDeliveryCoordinator(
                outbox=self._outbox,
                channels=self._channels,
                authority_resolver=self._resolve_outbox_delivery_authority,
            )
            self._execution_blockers = ()
        except Exception as exc:
            logger.error(
                "employee dispatch composition unavailable: %s",
                type(exc).__name__,
            )
            self._router = None
            self._dispatch = None
            self._outbox_delivery = None
            self._outbox_lifecycle = None
            self._execution_blockers = ("employee_gateway",)

    def _start_dispatch_worker(self) -> None:
        if self._dispatch is None or self._router is None or self._ingress is None:
            return
        if self._dispatch_thread is not None and self._dispatch_thread.is_alive():
            return
        self._dispatch_stop.clear()

        def run() -> None:
            delay = 0.05
            while not self._dispatch_stop.wait(delay):
                try:
                    worked = self._drain_employee_dispatch_once()
                    delay = 0.05 if worked else min(delay * 2.0, 1.0)
                except Exception as exc:
                    if isinstance(exc, ContextUnavailableError):
                        logger.warning(
                            "employee dispatch rejected unavailable context: reason=%s",
                            exc.reason.value,
                        )
                    else:
                        logger.error(
                            "employee dispatch worker failed closed: %s",
                            type(exc).__name__,
                        )
                    delay = min(max(delay, 0.05) * 2.0, 5.0)

        self._dispatch_thread = threading.Thread(
            target=run,
            name="employee-durable-dispatch",
            daemon=True,
        )
        self._dispatch_thread.start()

    def _record_employee_shadow_observation(
        self,
        observation: Mapping[str, object],
    ) -> None:
        """Anchor digest-only shadow comparisons without changing legacy output."""

        writer = self._writer
        if writer is None:
            return
        allowed = {
            "stage",
            "attempt_id",
            "agent_id",
            "context_snapshot_hash",
            "instruction_digest",
            "legacy_input_digest",
            "actor_input_digest",
            "input_match",
            "mismatch_code",
            "status",
            "output_digest",
            "safe_error_code",
        }
        payload = {
            key: value
            for key, value in dict(observation).items()
            if key in allowed and isinstance(value, (str, bool))
        }
        attempt_id = str(payload.get("attempt_id") or "")
        stage = str(payload.get("stage") or "")
        if not attempt_id.startswith("att_") or stage not in {"input", "terminal"}:
            return
        aggregate = f"employee-shadow:{attempt_id}"
        event = JournalEvent(
            event_type=f"employee.shadow.{stage}_observed",
            aggregate_id=aggregate,
            payload=payload,
        )
        with writer.transaction_guard():
            last = writer.get_last_frame()
            result = writer.commit(
                (event,),
                writer.get_aggregate_versions((aggregate,)),
                expected_head_sequence=0 if last is None else last.sequence,
                expected_head_hash="" if last is None else last.frame_hash,
            )
        if result.state is not CommitState.ANCHORED:
            raise RuntimeError("employee shadow audit was not anchored")

    def _resolve_employee_requester_principal(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        owner_principal_id: str,
        sender_principal_id: str,
        sender_union_id: str,
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
        if sender_principal_id == owner_principal_id and sender_principal_id:
            return owner_principal_id
        if not sender_union_id:
            return None
        service = self._service
        if service is None:
            return None
        service.synchronize_projection()
        candidates = tuple(
            state
            for state in service.list_states()
            if state.tenant_key == tenant_key and state.agent_id == agent_id
        )
        if len(candidates) != 1:
            return None
        state = candidates[0]
        if (
            state.phase is not HirePhase.ACTIVE
            or state.requester_principal_id != owner_principal_id
            or state.requester_union_id != sender_union_id
        ):
            return None
        return owner_principal_id

    def _drain_employee_dispatch_once(self) -> bool:
        ingress = self._ingress
        router = self._router
        dispatch = self._dispatch
        if ingress is None or router is None or dispatch is None:
            return False
        employee_runtime = getattr(dispatch, "employee_runtime", None)
        if employee_runtime is not None:
            employee_runtime.sweep_idle()
        ingress.rebuild_projection()
        router.rebuild_projection()
        worked = False
        for acceptance_id, record in tuple(ingress.state.by_acceptance_id.items()):
            if record.disposition is None:
                try:
                    payload = ingress.get_payload(acceptance_id)
                    first = (
                        payload.normalized_parts[0]
                        if len(payload.normalized_parts) == 1
                        else None
                    )
                    if (
                        isinstance(first, Mapping)
                        and first.get("type") == "membership_event"
                        and self._handle_control_ingress(acceptance_id)
                    ):
                        worked = True
                        continue
                    trust = self._managed_employee_ingress_trust(
                        record,
                        payload,
                    )
                except Exception:
                    trust = self._unknown_employee_ingress_trust()
                if trust is not None and trust.zone is not TrustZone.MANAGED_AGENT_GROUP:
                    try:
                        ingress.record_disposition(
                            acceptance_id,
                            state="ignored",
                            reason_code="authority_denied",
                        )
                    except IngressConflictError:
                        pass
                    worked = True
                    continue
                if trust is not None and trust.actor is ActorKind.EMPLOYEE:
                    # Employee Channel SDK ingress has no authenticated causal
                    # envelope.  Only the main Bot reply path can correlate a
                    # server parent/root message to an anchored Outbox record.
                    try:
                        ingress.record_disposition(
                            acceptance_id,
                            state="ignored",
                            reason_code="authority_denied",
                        )
                    except IngressConflictError:
                        pass
                    worked = True
                    continue
                if self._handle_control_ingress(acceptance_id):
                    worked = True
                    continue
                # Employee Bot subscriptions also observe group slash commands
                # owned by the main Bot.  Consume those after employee-specific
                # controls so commands such as ``/role list`` cannot be routed
                # into every employee mailbox as coding work.
                if self._handle_main_bot_group_command_ingress(acceptance_id):
                    worked = True
                    continue
                # The ingress payload may be reclaimed as soon as dispatch reaches a
                # terminal disposition.  Project shared group context on this same
                # serialized path before routing so context assembly cannot race GC.
                self._record_employee_ingress_group_event(acceptance_id)
                routed = router.state.by_acceptance_id.get(acceptance_id)
                if routed is None or routed.state not in {
                    "queued",
                    "dispatching",
                    "terminal",
                }:
                    router.route(acceptance_id)
                    worked = True
        worked = dispatch.dispatch_next() is not None or worked
        worked = self._reconcile_terminal_ingress() > 0 or worked
        worked = self._drain_employee_outbox_once() or worked
        if self._outbox is not None:
            worked = self._outbox.gc_superseded_snapshots() > 0 or worked
        return ingress.gc_terminal_payloads() > 0 or worked

    @staticmethod
    def _unknown_employee_ingress_trust() -> EffectiveTrust:
        return EffectiveTrust(
            zone=TrustZone.EXTERNAL_OR_UNKNOWN_GROUP,
            actor=ActorKind.UNKNOWN,
            managed_group=None,
            group_revision=None,
            grant_revision=None,
        )

    def trusted_employee_bot_open_ids(self) -> frozenset[str]:
        """Return only current READY employee Bot Open IDs.

        Workforce ``bot_principal_id`` values are internal identifiers and are
        deliberately never compared with Feishu ``open_id`` values.
        """

        service = self._service
        channels = self._channels
        if service is None or channels is None:
            return frozenset()
        projection = service.synchronize_projection()
        result: set[str] = set()
        for employee in projection.employees.values():
            if (
                employee.state is not EmployeeState.ACTIVE
                or employee.worker_type is not WorkerType.VISIBLE
                or not employee.bot_principal_id
            ):
                continue
            principal = projection.bot_principals.get(employee.bot_principal_id)
            status = channels.status(employee.agent_id)
            identity = getattr(status, "identity", None)
            open_id = identity.get("open_id") if isinstance(identity, Mapping) else None
            if (
                principal is not None
                and getattr(status, "state", None) is ChannelProcessState.READY
                and getattr(status, "agent_id", None) == employee.agent_id
                and getattr(status, "tenant_key", None) == employee.tenant_key
                and getattr(status, "bot_principal_id", None)
                == employee.bot_principal_id
                and getattr(status, "app_id", None) == principal.app_id
                and isinstance(open_id, str)
                and open_id.startswith("ou_")
            ):
                result.add(open_id)
        return frozenset(result)

    def is_valid_employee_continuation(
        self,
        *,
        sender_open_id: str,
        chat_id: str,
        message_id: str,
    ) -> bool:
        """Validate one Employee reply against durable outbound causality.

        ``message_id`` is the server-reported parent/root message, never an
        Employee-supplied payload field.  A READY identity alone is
        insufficient: its exact Channel generation and connection must match
        a delivered terminal Outbox record with an anchored collaboration
        publication.
        """

        service = self._service
        channels = self._channels
        outbox = self._outbox
        if (
            service is None
            or channels is None
            or outbox is None
            or not isinstance(sender_open_id, str)
            or not sender_open_id.startswith("ou_")
            or not isinstance(chat_id, str)
            or not chat_id.startswith("oc_")
            or not isinstance(message_id, str)
            or not message_id.startswith("om_")
        ):
            return False
        try:
            projection = service.synchronize_projection()
            outbox.rebuild_projection()
            candidates: list[tuple[Any, Any, Any]] = []
            for employee in projection.employees.values():
                if (
                    employee.state is not EmployeeState.ACTIVE
                    or employee.worker_type is not WorkerType.VISIBLE
                    or not employee.bot_principal_id
                ):
                    continue
                principal = projection.bot_principals.get(
                    employee.bot_principal_id
                )
                status = channels.status(employee.agent_id)
                identity = getattr(status, "identity", None)
                ready_metadata = getattr(status, "ready_metadata", None)
                if (
                    principal is None
                    or not isinstance(identity, Mapping)
                    or not isinstance(ready_metadata, Mapping)
                    or identity.get("open_id") != sender_open_id
                    or identity.get("app_id") != principal.app_id
                    or getattr(status, "state", None)
                    is not ChannelProcessState.READY
                    or getattr(status, "agent_id", None) != employee.agent_id
                    or getattr(status, "tenant_key", None)
                    != employee.tenant_key
                    or getattr(status, "bot_principal_id", None)
                    != employee.bot_principal_id
                    or getattr(status, "app_id", None) != principal.app_id
                    or type(getattr(status, "generation", None)) is not int
                    or not ready_metadata.get("connection_id")
                ):
                    continue
                candidates.append((employee, principal, status))
            if len(candidates) != 1:
                return False
            employee, principal, status = candidates[0]
            connection_id = status.ready_metadata["connection_id"]
            records = tuple(
                record
                for record in outbox.state.by_outbox_id.values()
                if record.tenant_key == employee.tenant_key
                and record.agent_id == employee.agent_id
                and record.chat_id == chat_id
                and record.binding is not None
                and record.binding.message_id == message_id
                and record.binding.app_id == principal.app_id
                and record.binding.generation == status.generation
                and record.binding.connection_id == connection_id
                and record.latest.state.terminal
            )
            if len(records) != 1:
                return False
            record = records[0]
            publications = tuple(
                event
                for frame in outbox._writer.replay()
                for event in frame.events
                if event.event_type
                == "employee.outbox.collaboration_published"
                and event.aggregate_id == record.outbox_id
                and event.payload.get("tenant_key") == employee.tenant_key
                and event.payload.get("chat_id") == chat_id
                and event.payload.get("agent_id") == employee.agent_id
                and event.payload.get("app_id") == principal.app_id
                and event.payload.get("generation") == status.generation
                and all(
                    isinstance(event.payload.get(field), str)
                    and bool(event.payload[field])
                    for field in (
                        "team_run_id",
                        "assignment_id",
                        "causal_event_id",
                    )
                )
            )
            return len(publications) == 1
        except Exception:
            logger.warning(
                "employee continuation validation failed closed",
                exc_info=True,
            )
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
            type(registry) is not ManagedGroupRegistry
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
                and getattr(status, "state", None)
                is ChannelProcessState.READY
                and getattr(status, "tenant_key", None) == metadata.tenant_key
                and getattr(status, "agent_id", None) == metadata.agent_id
                and getattr(status, "bot_principal_id", None)
                == metadata.bot_principal_id
                and getattr(status, "app_id", None) == metadata.app_id
                and getattr(status, "generation", None)
                == metadata.channel_generation
                and identity.get("app_id") == metadata.app_id
                and ready_metadata.get("connection_id")
                == metadata.connection_id
            )
        except Exception:
            return False

    def _managed_employee_ingress_trust(
        self,
        record: Any,
        payload: EmployeeIngressPayload,
    ) -> EffectiveTrust | None:
        registry = self._managed_group_registry
        if type(registry) is not ManagedGroupRegistry:
            return None
        if len(payload.normalized_parts) != 1:
            return self._unknown_employee_ingress_trust()
        part = payload.normalized_parts[0]
        if not isinstance(part, Mapping):
            return self._unknown_employee_ingress_trust()
        coordinates = _bound_remote_coordinates(record.metadata, part)
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

    def _reconcile_recovered_activation_required_ingress(self) -> int:
        """Terminalize old Inbox records whose reply is unknown or rejected."""

        service = self._service
        ingress = self._ingress
        if service is None or ingress is None:
            return 0
        unresolved_events: dict[tuple[str, str, str, str, str], str] = {}
        terminal_reasons = {
            "activation_required_reply_outcome_unknown": (
                "activation_reply_outcome_unknown"
            ),
            "activation_required_reply_recovery_rejected": (
                "activation_reply_recovery_rejected"
            ),
        }
        for state in service.list_states():
            effect_types = dict(state.effect_types)
            for effect_id, effect_state in state.effects:
                metadata = dict(state.metadata_for(effect_id))
                if (
                    effect_state is HireEffectState.ACTION_REQUIRED
                    and effect_types.get(effect_id)
                    == "employee_activation_required_reply"
                    and metadata.get("error_code") in terminal_reasons
                    and isinstance(metadata.get("ingress_event_id"), str)
                    and metadata["ingress_event_id"]
                ):
                    error_code = metadata["error_code"]
                    unresolved_events[
                        (
                            state.tenant_key,
                            state.agent_id,
                            state.bot_principal_id,
                            state.app_id,
                            metadata["ingress_event_id"],
                        )
                    ] = terminal_reasons[error_code]
        reconciled = 0
        for acceptance_id, record in tuple(ingress.state.by_acceptance_id.items()):
            metadata = record.metadata
            key = (
                metadata.tenant_key,
                metadata.agent_id,
                metadata.bot_principal_id,
                metadata.app_id,
                metadata.event_id,
            )
            reason_code = unresolved_events.get(key)
            if record.disposition is not None or reason_code is None:
                continue
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code=reason_code,
                )
                reconciled += 1
            except IngressConflictError:
                pass
        return reconciled

    def _resume_recoverable_activation_required_replies(self) -> int:
        """Finish PREPARED/COMMITTED old replies before Router admission starts."""

        service = self._service
        ingress = self._ingress
        if service is None or ingress is None:
            return 0
        service.synchronize_projection()
        ingress.rebuild_projection()
        recoverable: dict[
            tuple[str, str, str, str, str],
            tuple[str, str],
        ] = {}
        for state in service.list_states():
            effect_types = dict(state.effect_types)
            for effect_id, effect_state in state.effects:
                metadata = dict(state.metadata_for(effect_id))
                event_id = metadata.get("ingress_event_id")
                if (
                    effect_state
                    in {HireEffectState.PREPARED, HireEffectState.COMMITTED}
                    and effect_types.get(effect_id)
                    == "employee_activation_required_reply"
                    and isinstance(event_id, str)
                    and event_id
                ):
                    recoverable[
                        (
                            state.tenant_key,
                            state.agent_id,
                            state.bot_principal_id,
                            state.app_id,
                            event_id,
                        )
                    ] = (state.intent_id, effect_id)
        resumed = 0
        for acceptance_id, record in tuple(ingress.state.by_acceptance_id.items()):
            metadata = record.metadata
            key = (
                metadata.tenant_key,
                metadata.agent_id,
                metadata.bot_principal_id,
                metadata.app_id,
                metadata.event_id,
            )
            effect_binding = recoverable.get(key)
            if record.disposition is not None or effect_binding is None:
                continue
            intent_id, effect_id = effect_binding
            try:
                if self._handle_control_ingress(acceptance_id):
                    resumed += 1
                    continue
                current = service.get_state(intent_id)
                if current is None:
                    raise RuntimeError("employee activation reply state is unavailable")
                effect_state = current.effect_state(effect_id)
                if effect_state is HireEffectState.COMMITTED:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_required",
                    )
                    resumed += 1
                elif effect_state is HireEffectState.PREPARED:
                    service.commit_effect_transition(
                        current.intent_id,
                        effect_id=effect_id,
                        effect_type="employee_activation_required_reply",
                        next_state=HireEffectState.ACTION_REQUIRED,
                        metadata={
                            **dict(current.metadata_for(effect_id)),
                            "error_code": (
                                "activation_required_reply_recovery_rejected"
                            ),
                        },
                    )
                    self._reconcile_recovered_activation_required_ingress()
            except Exception as exc:
                logger.error(
                    "employee activation-required reply recovery failed closed: %s",
                    type(exc).__name__,
                )
                current = service.get_state(intent_id)
                if (
                    current is not None
                    and current.effect_state(effect_id)
                    is HireEffectState.EXECUTING
                ):
                    service.commit_effect_transition(
                        current.intent_id,
                        effect_id=effect_id,
                        effect_type="employee_activation_required_reply",
                        next_state=HireEffectState.ACTION_REQUIRED,
                        metadata={
                            **dict(current.metadata_for(effect_id)),
                            "error_code": (
                                "activation_required_reply_outcome_unknown"
                            ),
                        },
                    )
                    self._reconcile_recovered_activation_required_ingress()
                elif (
                    current is not None
                    and current.effect_state(effect_id)
                    is HireEffectState.PREPARED
                ):
                    service.commit_effect_transition(
                        current.intent_id,
                        effect_id=effect_id,
                        effect_type="employee_activation_required_reply",
                        next_state=HireEffectState.ACTION_REQUIRED,
                        metadata={
                            **dict(current.metadata_for(effect_id)),
                            "error_code": (
                                "activation_required_reply_recovery_rejected"
                            ),
                        },
                    )
                    self._reconcile_recovered_activation_required_ingress()
                elif (
                    current is not None
                    and current.effect_state(effect_id)
                    is HireEffectState.COMMITTED
                ):
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_required",
                    )
        return resumed

    def _handle_control_ingress(self, acceptance_id: str) -> bool:
        """Consume exact durable employee controls before Router admission."""

        ingress = self._ingress
        if ingress is None:
            return False
        ingress.rebuild_projection()
        record = ingress.state.by_acceptance_id.get(acceptance_id)
        if record is None:
            return False
        if record.disposition is not None:
            return record.disposition.reason_code.startswith(
                ("stop_", "membership_", "history_", "memory_", "activation_")
            )
        try:
            payload = ingress.get_payload(acceptance_id)
        except Exception:
            return False
        first = (
            payload.normalized_parts[0]
            if len(payload.normalized_parts) == 1
            else None
        )
        is_membership_event = (
            isinstance(first, Mapping)
            and first.get("type") == "membership_event"
        )
        if is_membership_event:
            remote_chat_id = first.get("remote_chat_id")
            if not isinstance(remote_chat_id, str) or not self._membership_event_transport_is_current(
                record.metadata,
                remote_chat_id,
            ):
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="ignored",
                        reason_code="membership_unmanaged",
                    )
                except IngressConflictError:
                    pass
                return True
        if not is_membership_event:
            try:
                trust = self._managed_employee_ingress_trust(record, payload)
            except Exception:
                trust = self._unknown_employee_ingress_trust()
            if trust is not None and trust.zone is not TrustZone.MANAGED_AGENT_GROUP:
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="ignored",
                        reason_code="authority_denied",
                    )
                except IngressConflictError:
                    pass
                return True
            if trust is not None and trust.actor is ActorKind.EMPLOYEE:
                return False
        if is_membership_event:
            if self._membership is None:
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="membership_unavailable",
                    )
                except IngressConflictError:
                    pass
                return True
            metadata = record.metadata
            remote_chat_id = first.get("remote_chat_id")
            operation = first.get("operation")
            expected_chat_index = (
                "oc_" + hashlib.sha256(remote_chat_id.encode()).hexdigest()
                if isinstance(remote_chat_id, str) and remote_chat_id
                else ""
            )
            if (
                expected_chat_index != metadata.chat_id
                or operation not in {"added", "deleted"}
            ):
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="ignored",
                        reason_code="membership_unmanaged",
                    )
                except IngressConflictError:
                    pass
                return True
            try:
                outcome = self._membership.reconcile_event(
                    tenant_key=metadata.tenant_key,
                    chat_id=remote_chat_id,
                    agent_id=metadata.agent_id,
                    app_id=metadata.app_id,
                    observed_is_member=operation == "added",
                )
            except MembershipBindingError:
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="ignored",
                        reason_code="membership_unmanaged",
                    )
                except IngressConflictError:
                    pass
                return True
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code=f"membership_{outcome.state.value}",
                )
            except IngressConflictError:
                pass
            return True
        texts: list[str] = []
        for part in payload.normalized_parts:
            content = part.get("content") if isinstance(part, Mapping) else None
            value = content.get("text") if isinstance(content, Mapping) else None
            if isinstance(value, str):
                texts.append(value.strip())
        if texts == ["/status"]:
            router = self._router
            if router is not None and not router.claim_control(
                acceptance_id,
                command="/status",
            ):
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_denied",
                    )
                except IngressConflictError:
                    pass
                return True
            handled = self._handle_durable_activation_status(acceptance_id)
            if not handled:
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_denied",
                    )
                except IngressConflictError:
                    pass
            return True
        if (
            isinstance(first, Mapping)
            and self._handle_pending_verification_group_mention(
                acceptance_id=acceptance_id,
                record=record,
                first=first,
            )
        ):
            return True
        data_control = self._parse_data_control(texts)
        if data_control is not None:
            return self._handle_data_control(
                acceptance_id=acceptance_id,
                command=data_control[0],
                record=record,
                payload=payload,
                history_days=data_control[1],
            )
        if texts != ["/stop"]:
            return False
        dispatch = self._dispatch
        lifecycle = self._outbox_lifecycle
        if dispatch is None or lifecycle is None:
            return False
        metadata = record.metadata
        coordinates = _bound_remote_coordinates(metadata, first)
        if coordinates is None:
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code="stop_coordinates_invalid",
                )
            except IngressConflictError:
                pass
            return True
        remote_chat_id, _remote_message_id, remote_root_id = coordinates
        outcome = dispatch.request_cancel(
            agent_id=metadata.agent_id,
            chat_id=remote_chat_id,
            requester_principal_id=metadata.sender_principal_id,
            command_acceptance_id=acceptance_id,
        )
        lifecycle.command_response(
            tenant_key=metadata.tenant_key,
            agent_id=metadata.agent_id,
            chat_id=remote_chat_id,
            thread_root_message_id=remote_root_id or _remote_message_id,
            command_acceptance_id=acceptance_id,
            status=outcome.status,
        )
        try:
            ingress.record_disposition(
                acceptance_id,
                state="terminal",
                reason_code=f"stop_{outcome.status}",
            )
        except IngressConflictError:
            pass
        self._drain_employee_outbox_once()
        return True

    def _handle_main_bot_group_command_ingress(self, acceptance_id: str) -> bool:
        """Ignore main-Bot slash commands observed by an employee group Bot."""

        ingress = self._ingress
        if ingress is None:
            return False
        record = ingress.state.by_acceptance_id.get(acceptance_id)
        if record is None or record.disposition is not None:
            return False
        metadata = record.metadata
        if (
            metadata.event_type != "im.message.receive_v1"
            or metadata.action_identity
        ):
            return False
        try:
            payload = ingress.get_payload(acceptance_id)
        except Exception:
            return False
        first = payload.normalized_parts[0] if len(payload.normalized_parts) == 1 else None
        if (
            not isinstance(first, Mapping)
            or first.get("type") != "message"
            or first.get("chat_type") != "group"
        ):
            return False
        content = first.get("content")
        text = content.get("text") if isinstance(content, Mapping) else None
        if not isinstance(text, str) or not text.lstrip().startswith("/"):
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

    def _handle_pending_verification_group_mention(
        self,
        *,
        acceptance_id: str,
        record: Any,
        first: Mapping[str, object],
    ) -> bool:
        """Reply only to a hash-bound group message solely @mentioning this bot."""

        service = self._service
        channels = self._channels
        ingress = self._ingress
        if service is None or channels is None or ingress is None:
            return False
        metadata = getattr(record, "metadata", None)
        if not isinstance(metadata, EmployeeIngressMetadata):
            return False
        coordinates = _bound_remote_coordinates(metadata, first)
        if coordinates is None:
            return False
        remote_chat_id, remote_message_id, _remote_root_id = coordinates
        mentions = first.get("mentions")
        if not isinstance(mentions, tuple) or len(mentions) != 1:
            return False
        mention = mentions[0]
        if not isinstance(mention, Mapping) or set(mention) != {
            "key",
            "mentioned_type",
            "open_id",
            "tenant_key",
        }:
            return False
        status = channels.status(metadata.agent_id)
        identity = getattr(status, "identity", None)
        ready_metadata = getattr(status, "ready_metadata", None)
        employee_open_id = (
            identity.get("open_id") if isinstance(identity, Mapping) else None
        )
        connection_id = (
            ready_metadata.get("connection_id")
            if isinstance(ready_metadata, Mapping)
            else None
        )
        if (
            first.get("type") != "message"
            or first.get("message_type") != "text"
            or first.get("chat_type") != "group"
            or first.get("sender_type") != "user"
            or first.get("sender_id_type") != "open_id"
            or first.get("sender_id") != metadata.sender_principal_id
            or first.get("sender_tenant_key") != metadata.tenant_key
            or metadata.event_type != "im.message.receive_v1"
            or mention.get("mentioned_type") != "bot"
            or not isinstance(mention.get("key"), str)
            or not mention.get("key")
            or mention.get("open_id") != employee_open_id
            or mention.get("tenant_key") != metadata.tenant_key
            or getattr(status, "state", None) is not ChannelProcessState.READY
            or getattr(status, "agent_id", None) != metadata.agent_id
            or getattr(status, "app_id", None) != metadata.app_id
            or getattr(status, "tenant_key", None) != metadata.tenant_key
            or getattr(status, "bot_principal_id", None)
            != metadata.bot_principal_id
            or not isinstance(identity, Mapping)
            or identity.get("app_id") != metadata.app_id
            or not isinstance(employee_open_id, str)
            or not employee_open_id
        ):
            return False
        projection = service.synchronize_projection()
        candidates = tuple(
            state
            for state in service.list_states()
            if state.agent_id == metadata.agent_id
        )
        if len(candidates) != 1:
            return False
        state = candidates[0]
        employee = projection.employees.get(state.agent_id)
        effect_id = f"activation-required-reply:{metadata.event_id}"
        effect_state = state.effect_state(effect_id)
        effect_type = dict(state.effect_types).get(effect_id)
        effect_metadata = dict(state.metadata_for(effect_id))
        current_ingress_binding = (
            state.channel_generation == metadata.channel_generation
            and state.channel_connection_id == metadata.connection_id
        )
        recoverable_old_ingress = (
            state.channel_generation > metadata.channel_generation
            and effect_state
            in {HireEffectState.PREPARED, HireEffectState.COMMITTED}
            and effect_type == "employee_activation_required_reply"
            and effect_metadata.get("ingress_event_id") == metadata.event_id
        )
        if (
            state.phase is not HirePhase.READY_PENDING_VERIFICATION
            or state.tenant_key != metadata.tenant_key
            or state.app_id != metadata.app_id
            or state.bot_principal_id != metadata.bot_principal_id
            or getattr(status, "generation", None) != state.channel_generation
            or connection_id != state.channel_connection_id
            or not (current_ingress_binding or recoverable_old_ingress)
            or employee is None
            or employee.owner_principal_id != state.requester_principal_id
            or remote_chat_id not in employee.member_groups
        ):
            return False
        if not self._send_activation_required_notice(
            state=state,
            acceptance_id=acceptance_id,
            event_id=metadata.event_id,
            target_chat_id=remote_chat_id,
            reply_message_id=remote_message_id,
        ):
            return False
        try:
            ingress.record_disposition(
                acceptance_id,
                state="terminal",
                reason_code="activation_required",
            )
        except IngressConflictError:
            pass
        return True

    def _send_activation_required_notice(
        self,
        *,
        state: DurableHireState,
        acceptance_id: str,
        event_id: str,
        target_chat_id: str,
        reply_message_id: str,
    ) -> bool:
        """Durably send one stable group reply explaining pending verification."""

        service = self._require_service()
        channels = self._channels
        if channels is None:
            return False
        effect_id = f"activation-required-reply:{event_id}"
        effect_state = state.effect_state(effect_id)
        if effect_state is None:
            state = service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_activation_required_reply",
                next_state=HireEffectState.PREPARED,
                metadata={"ingress_event_id": event_id},
            )
            effect_state = state.effect_state(effect_id)
        if effect_state is HireEffectState.PREPARED:
            state = service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_activation_required_reply",
                next_state=HireEffectState.EXECUTING,
            )
            effect_state = state.effect_state(effect_id)
        if effect_state is HireEffectState.COMMITTED:
            return True
        if effect_state is not HireEffectState.EXECUTING:
            return False
        stable_uuid = _stable_feishu_uuid(
            f"employee-activation-required:{state.agent_id}:{acceptance_id}"
        )
        receipt = channels.send(
            state.agent_id,
            generation=state.channel_generation,
            target=target_chat_id,
            message={
                "text": (
                    f"{state.employee_name} 当前尚未完成身份验证。"
                    "请先私聊该员工发送 /status，激活成功后再回到群里 @ 它。"
                )
            },
            options={"uuid": stable_uuid, "reply_to": reply_message_id},
        )
        if (
            getattr(receipt, "success", False) is not True
            or getattr(receipt, "app_id", "") != state.app_id
            or getattr(receipt, "generation", 0) != state.channel_generation
            or getattr(receipt, "connection_id", "")
            != state.channel_connection_id
            or not getattr(receipt, "request_id", "")
            or not getattr(receipt, "message_id", "")
        ):
            raise RuntimeError("employee activation-required receipt is invalid")
        service.commit_effect_transition(
            state.intent_id,
            effect_id=effect_id,
            effect_type="employee_activation_required_reply",
            next_state=HireEffectState.COMMITTED,
            metadata={
                "send_request_id": receipt.request_id,
                "ingress_event_id": event_id,
                "reply_app_id": receipt.app_id,
                "reply_message_id": receipt.message_id,
                "generation": str(receipt.generation),
                "connection_id": receipt.connection_id,
                "main_bot_send_count": "0",
            },
        )
        return True

    def _handle_durable_activation_status(self, acceptance_id: str) -> bool:
        """Verify a pending employee from its durable official-SDK ingress."""

        ingress = self._ingress
        service = self._service
        if ingress is None or service is None:
            return False
        with self._activation_lock:
            ingress.rebuild_projection()
            record = ingress.state.by_acceptance_id.get(acceptance_id)
            if record is None:
                return False
            if record.disposition is not None:
                return record.disposition.reason_code.startswith("activation_")
            candidates = tuple(
                candidate
                for candidate in service.list_states()
                if candidate.agent_id == record.metadata.agent_id
            )
            if len(candidates) != 1:
                return False
            state = candidates[0]
            active_state = state if state.phase is HirePhase.ACTIVE else None
            pending_state = (
                state
                if state.phase is HirePhase.READY_PENDING_VERIFICATION
                else None
            )
            try:
                payload = ingress.get_payload(acceptance_id)
            except Exception:
                return False
            first = (
                payload.normalized_parts[0]
                if len(payload.normalized_parts) == 1
                else None
            )
            content = first.get("content") if isinstance(first, Mapping) else None
            command = content.get("text") if isinstance(content, Mapping) else None
            sender_id = first.get("sender_id") if isinstance(first, Mapping) else None
            sender_union_id = (
                first.get("sender_union_id") if isinstance(first, Mapping) else None
            )
            metadata = record.metadata
            coordinates = _bound_remote_coordinates(metadata, first)
            if coordinates is None:
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_denied",
                    )
                except IngressConflictError:
                    pass
                return True
            _remote_chat_id, remote_message_id, _remote_root_id = coordinates
            received_at = datetime.fromisoformat(
                metadata.received_at.replace("Z", "+00:00")
            ).timestamp()
            stable_identity_valid = (
                first is not None
                and first.get("type") == "message"
                and first.get("message_type") == "text"
                and first.get("chat_type") == "p2p"
                and first.get("sender_type") == "user"
                and isinstance(command, str)
                and command.strip() == "/status"
                and isinstance(sender_id, str)
                and sender_id
                and sender_id == metadata.sender_principal_id
                and isinstance(sender_union_id, str)
                and sender_union_id == state.requester_union_id
                and first.get("sender_tenant_key") == metadata.tenant_key
                and metadata.tenant_key == state.tenant_key
                and metadata.app_id == state.app_id
                and metadata.agent_id == state.agent_id
                and metadata.bot_principal_id == state.bot_principal_id
                and metadata.event_type == "im.message.receive_v1"
            )
            identity_valid = (
                stable_identity_valid
                and metadata.channel_generation == state.channel_generation
                and metadata.connection_id == state.channel_connection_id
            )
            phase_reply_effect_id = (
                f"verification-retry-reply:{metadata.event_id}"
            )
            phase_reply_code = dict(
                state.metadata_for(phase_reply_effect_id)
            ).get("error_code", "")
            original_phase = self._phase_status_decision(phase_reply_code)
            if original_phase is not None:
                if not stable_identity_valid:
                    try:
                        ingress.record_disposition(
                            acceptance_id,
                            state="terminal",
                            reason_code="activation_denied",
                        )
                    except IngressConflictError:
                        pass
                    return True
                if (
                    state.effect_state(phase_reply_effect_id)
                    is not HireEffectState.COMMITTED
                    and not self._send_activation_retry_notice(
                        state=state,
                        generation=state.channel_generation,
                        event_id=metadata.event_id,
                        sender_id=sender_id,
                        message=self._status_unavailable_message_for_phase(
                            original_phase
                        ),
                        decision_code=phase_reply_code,
                    )
                ):
                    raise RuntimeError("employee phase status reply unavailable")
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code=f"activation_status_{original_phase.value}",
                    )
                except IngressConflictError:
                    pass
                if state.phase is HirePhase.VALIDATING:
                    self._promote_after_phase_status_reply(state.intent_id)
                return True
            if active_state is None and pending_state is None:
                if not identity_valid:
                    try:
                        ingress.record_disposition(
                            acceptance_id,
                            state="terminal",
                            reason_code="activation_denied",
                        )
                    except IngressConflictError:
                        pass
                    return True
                message = self._status_unavailable_message(state)
                if not self._send_activation_retry_notice(
                    state=state,
                    generation=metadata.channel_generation,
                    event_id=metadata.event_id,
                    sender_id=sender_id,
                    message=message,
                    decision_code=f"phase_status:{state.phase.value}",
                ):
                    raise RuntimeError("employee phase status reply unavailable")
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code=f"activation_status_{state.phase.value}",
                    )
                except IngressConflictError:
                    pass
                if state.phase is HirePhase.VALIDATING:
                    self._promote_after_phase_status_reply(state.intent_id)
                return True
            if active_state is not None:
                if not identity_valid:
                    try:
                        ingress.record_disposition(
                            acceptance_id,
                            state="terminal",
                            reason_code="activation_denied",
                        )
                    except IngressConflictError:
                        pass
                    return True
                if active_state.activation_ingress_event_id == metadata.event_id:
                    self._send_activation_success_notice(
                        state=active_state,
                        event_id=metadata.event_id,
                        sender_id=sender_id,
                    )
                    self._schedule_activation_notification(active_state)
                    reason_code = "activation_verified"
                else:
                    if not self._send_already_active_notice(
                        state=active_state,
                        acceptance_id=acceptance_id,
                        sender_id=sender_id,
                    ):
                        raise RuntimeError("employee already-active reply unavailable")
                    reason_code = "activation_already_active"
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code=reason_code,
                    )
                except IngressConflictError:
                    pass
                with service.employee_dispatch_guard():
                    service.synchronize_projection_unlocked()
                return True
            assert pending_state is not None
            state = pending_state
            if (
                stable_identity_valid
                and metadata.channel_generation < state.channel_generation
            ):
                self._send_activation_retry_notice(
                    state=state,
                    generation=state.channel_generation,
                    event_id=metadata.event_id,
                    sender_id=sender_id,
                    message=(
                        "Employee session refreshed. Send /status again."
                    ),
                )
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_generation_refreshed",
                    )
                except IngressConflictError:
                    pass
                return True
            if not identity_valid:
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_denied",
                    )
                except IngressConflictError:
                    pass
                return True
            challenge = self._challenge_for_state(state)
            now = time.time()
            challenge_valid = (
                challenge is not None
                and challenge.issued_at <= received_at <= challenge.expires_at
                and now <= challenge.expires_at
            )
            if not challenge_valid:
                if challenge is None or now > challenge.expires_at:
                    challenge = self._renew_activation_challenge(state)
                self._send_activation_retry_notice(
                    state=service.get_state(state.intent_id) or state,
                    generation=metadata.channel_generation,
                    event_id=metadata.event_id,
                    sender_id=sender_id,
                    message=(
                        "Activation window refreshed. Send /status again."
                    ),
                )
                try:
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code="activation_challenge_renewed",
                    )
                except IngressConflictError:
                    pass
                return True
            activated = self._complete_status_activation(
                state=state,
                challenge=challenge,
                generation=metadata.channel_generation,
                event_id=metadata.event_id,
                message_id=remote_message_id,
                sender_id=sender_id,
                sender_union_id=sender_union_id,
                command=command.strip(),
                is_p2p=True,
                reply_options={},
                received_at=received_at,
            )
            try:
                ingress.record_disposition(
                    acceptance_id,
                    state="terminal",
                    reason_code=(
                        "activation_verified" if activated else "activation_denied"
                    ),
                )
            except IngressConflictError:
                pass
            with service.employee_dispatch_guard():
                service.synchronize_projection_unlocked()
            return True

    @staticmethod
    def _status_unavailable_message(state: DurableHireState) -> str:
        return EmployeeDepartmentRuntime._status_unavailable_message_for_phase(
            state.phase
        )

    @staticmethod
    def _status_unavailable_message_for_phase(phase: HirePhase) -> str:
        if phase is HirePhase.ACTION_REQUIRED:
            return (
                "Employee setup requires administrator action. Ask the "
                "administrator to run /roster; /status cannot activate it yet."
            )
        if phase in {HirePhase.RETIRING, HirePhase.ARCHIVED}:
            return "This employee is retired and cannot be activated."
        return (
            f"Employee setup is currently {phase.value}; activation is not "
            "ready. Send /status again after setup completes."
        )

    @staticmethod
    def _phase_status_decision(value: str) -> HirePhase | None:
        prefix = "phase_status:"
        if not value.startswith(prefix):
            return None
        try:
            phase = HirePhase(value.removeprefix(prefix))
        except ValueError:
            return None
        if phase in {HirePhase.ACTIVE, HirePhase.READY_PENDING_VERIFICATION}:
            return None
        return phase

    def _promote_after_phase_status_reply(self, intent_id: str) -> None:
        service = self._require_service()
        current = service.get_state(intent_id)
        if (
            current is None
            or current.phase is not HirePhase.VALIDATING
            or not current.slash_spec_hash
            or current.slash_spec_hash != current.slash_observed_hash
            or current.channel_generation <= 0
            or current.channel_identity_app_id != current.app_id
            or not current.channel_connection_id
        ):
            return
        router = self._verification_router
        if router is None:
            raise RuntimeError("employee Verification Router unavailable")
        challenge = router.issue_challenge(
            VerificationBinding(
                hire_intent_id=current.intent_id,
                tenant_key=current.tenant_key,
                app_id=current.app_id,
                agent_id=current.agent_id,
                generation=current.channel_generation,
                requester_principal_id=current.requester_principal_id,
                requester_union_id=current.requester_union_id,
                expected_slash_spec_hash=current.slash_spec_hash,
            )
        )
        service.begin_activation_verification(challenge)
        self._challenges[current.intent_id] = challenge

    def _send_already_active_notice(
        self,
        *,
        state: DurableHireState,
        acceptance_id: str,
        sender_id: str,
    ) -> bool:
        """Durably and idempotently acknowledge a new status event after ACTIVE."""

        writer = self._writer
        channels = self._channels
        if writer is None or channels is None:
            return False
        aggregate_id = f"hire-already-active:{acceptance_id}"
        existing = [
            event.event_type
            for frame in writer.replay()
            for event in frame.events
            if event.aggregate_id == aggregate_id
        ]
        committed_event = "hire.activation.already_active_reply.committed"
        if committed_event in existing:
            return True

        def commit(event_type: str, payload: Mapping[str, object]) -> None:
            with writer.transaction_guard():
                last = writer.get_last_frame()
                result = writer.commit(
                    (
                        JournalEvent(
                            event_type=event_type,
                            aggregate_id=aggregate_id,
                            payload=dict(payload),
                        ),
                    ),
                    writer.get_aggregate_versions((aggregate_id,)),
                    expected_head_sequence=0 if last is None else last.sequence,
                    expected_head_hash="" if last is None else last.frame_hash,
                )
            if result.state.value != "anchored":
                raise RuntimeError("already-active reply event was not anchored")

        base_payload = {
            "intent_id": state.intent_id,
            "acceptance_id": acceptance_id,
            "app_id": state.app_id,
            "generation": state.channel_generation,
        }
        prepared_event = "hire.activation.already_active_reply.prepared"
        executing_event = "hire.activation.already_active_reply.executing"
        if prepared_event not in existing:
            commit(prepared_event, base_payload)
        if executing_event not in existing:
            commit(executing_event, base_payload)
        stable_uuid = _stable_feishu_uuid(
            f"employee-already-active:{acceptance_id}"
        )
        receipt = channels.send(
            state.agent_id,
            generation=state.channel_generation,
            target=sender_id,
            message={"text": f"{state.employee_name} is already active."},
            options={"uuid": stable_uuid},
        )
        if (
            getattr(receipt, "success", False) is not True
            or getattr(receipt, "app_id", "") != state.app_id
            or getattr(receipt, "generation", 0) != state.channel_generation
            or getattr(receipt, "connection_id", "")
            != state.channel_connection_id
            or not getattr(receipt, "request_id", "")
            or not getattr(receipt, "message_id", "")
        ):
            raise RuntimeError("employee already-active reply receipt is invalid")
        commit(
            committed_event,
            {
                **base_payload,
                "send_request_id": receipt.request_id,
                "reply_message_id": receipt.message_id,
                "connection_id": receipt.connection_id,
                "uuid": stable_uuid,
            },
        )
        return True

    def _send_activation_success_notice(
        self,
        *,
        state: DurableHireState,
        event_id: str,
        sender_id: str,
    ) -> bool:
        """Durably acknowledge successful activation in the employee DM."""

        writer = self._writer
        channels = self._channels
        if writer is None or channels is None:
            return False
        aggregate_id = f"hire-activation-success:{event_id}"
        existing = [
            event.event_type
            for frame in writer.replay()
            for event in frame.events
            if event.aggregate_id == aggregate_id
        ]
        committed_event = "hire.activation.success_reply.committed"
        if committed_event in existing:
            return True

        def commit(event_type: str, payload: Mapping[str, object]) -> None:
            with writer.transaction_guard():
                last = writer.get_last_frame()
                result = writer.commit(
                    (
                        JournalEvent(
                            event_type=event_type,
                            aggregate_id=aggregate_id,
                            payload=dict(payload),
                        ),
                    ),
                    writer.get_aggregate_versions((aggregate_id,)),
                    expected_head_sequence=0 if last is None else last.sequence,
                    expected_head_hash="" if last is None else last.frame_hash,
                )
            if result.state.value != "anchored":
                raise RuntimeError("activation success reply event was not anchored")

        base_payload = {
            "intent_id": state.intent_id,
            "event_id": event_id,
            "app_id": state.app_id,
            "generation": state.channel_generation,
        }
        prepared_event = "hire.activation.success_reply.prepared"
        executing_event = "hire.activation.success_reply.executing"
        if prepared_event not in existing:
            commit(prepared_event, base_payload)
        if executing_event not in existing:
            commit(executing_event, base_payload)
        stable_uuid = _stable_feishu_uuid(f"employee-activation-success:{event_id}")
        receipt = channels.send(
            state.agent_id,
            generation=state.channel_generation,
            target=sender_id,
            message={"text": f"{state.employee_name} is active."},
            options={"uuid": stable_uuid},
        )
        if (
            getattr(receipt, "success", False) is not True
            or getattr(receipt, "app_id", "") != state.app_id
            or getattr(receipt, "generation", 0) != state.channel_generation
            or getattr(receipt, "connection_id", "")
            != state.channel_connection_id
            or not getattr(receipt, "request_id", "")
            or not getattr(receipt, "message_id", "")
        ):
            raise RuntimeError("employee activation success reply receipt is invalid")
        commit(
            committed_event,
            {
                **base_payload,
                "send_request_id": receipt.request_id,
                "reply_message_id": receipt.message_id,
                "connection_id": receipt.connection_id,
                "uuid": stable_uuid,
            },
        )
        return True

    def _challenge_for_state(
        self,
        state: DurableHireState,
    ) -> VerificationChallenge | None:
        challenge = self._challenges.get(state.intent_id)
        if challenge is not None and challenge.nonce == state.verification_nonce:
            return challenge
        if (
            not state.verification_nonce
            or state.verification_issued_at <= 0
            or state.verification_expires_at <= state.verification_issued_at
        ):
            return None
        challenge = VerificationChallenge(
            hire_intent_id=state.intent_id,
            tenant_key=state.tenant_key,
            app_id=state.app_id,
            agent_id=state.agent_id,
            generation=state.channel_generation,
            requester_principal_id=state.requester_principal_id,
            requester_union_id=state.requester_union_id,
            expected_slash_spec_hash=state.slash_spec_hash,
            nonce=state.verification_nonce,
            issued_at=state.verification_issued_at,
            expires_at=state.verification_expires_at,
        )
        self._challenges[state.intent_id] = challenge
        return challenge

    def _send_activation_retry_notice(
        self,
        *,
        state: DurableHireState,
        generation: int,
        event_id: str,
        sender_id: str,
        message: str,
        decision_code: str = "",
    ) -> bool:
        service = self._require_service()
        channels = self._channels
        if channels is None:
            return False
        effect_id = f"verification-retry-reply:{event_id}"
        effect_state = state.effect_state(effect_id)
        if effect_state is None:
            state = service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_status_reply",
                next_state=HireEffectState.PREPARED,
                metadata=(
                    {"error_code": decision_code} if decision_code else None
                ),
            )
            effect_state = state.effect_state(effect_id)
        if effect_state is HireEffectState.PREPARED:
            state = service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_status_reply",
                next_state=HireEffectState.EXECUTING,
            )
            effect_state = state.effect_state(effect_id)
        if effect_state is HireEffectState.COMMITTED:
            return True
        if effect_state is not HireEffectState.EXECUTING:
            return False
        stable_uuid = _stable_feishu_uuid(f"employee-activation-retry:{event_id}")
        receipt = channels.send(
            state.agent_id,
            generation=generation,
            target=sender_id,
            message={"text": message},
            options={"uuid": stable_uuid},
        )
        if (
            getattr(receipt, "success", False) is not True
            or getattr(receipt, "app_id", "") != state.app_id
            or getattr(receipt, "generation", 0) != generation
            or getattr(receipt, "connection_id", "")
            != state.channel_connection_id
            or not getattr(receipt, "request_id", "")
            or not getattr(receipt, "message_id", "")
        ):
            raise RuntimeError("employee activation retry receipt is invalid")
        service.commit_effect_transition(
            state.intent_id,
            effect_id=effect_id,
            effect_type="employee_status_reply",
            next_state=HireEffectState.COMMITTED,
            metadata={
                "send_request_id": receipt.request_id,
                "ingress_event_id": event_id,
                "reply_app_id": receipt.app_id,
                "reply_message_id": receipt.message_id,
                "generation": str(receipt.generation),
                "connection_id": receipt.connection_id,
                "main_bot_send_count": "0",
            },
        )
        return True

    def _send_activation_incomplete_notice(
        self,
        *,
        state: DurableHireState,
        generation: int,
        event_id: str,
        sender_id: str,
    ) -> bool:
        """Durably correct a preflight reply when final activation is rejected."""

        service = self._require_service()
        channels = self._channels
        if channels is None:
            return False
        effect_id = f"verification-incomplete-reply:{event_id}"
        effect_state = state.effect_state(effect_id)
        if effect_state is None:
            state = service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_status_reply",
                next_state=HireEffectState.PREPARED,
            )
            effect_state = state.effect_state(effect_id)
        if effect_state is HireEffectState.PREPARED:
            state = service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_status_reply",
                next_state=HireEffectState.EXECUTING,
            )
            effect_state = state.effect_state(effect_id)
        if effect_state is HireEffectState.COMMITTED:
            return True
        if effect_state is not HireEffectState.EXECUTING:
            return False
        stable_uuid = _stable_feishu_uuid(
            f"employee-activation-incomplete:{event_id}"
        )
        receipt = channels.send(
            state.agent_id,
            generation=generation,
            target=sender_id,
            message={"text": "Activation did not complete. Send /status again."},
            options={"uuid": stable_uuid},
        )
        if (
            getattr(receipt, "success", False) is not True
            or getattr(receipt, "app_id", "") != state.app_id
            or getattr(receipt, "generation", 0) != generation
            or getattr(receipt, "connection_id", "")
            != state.channel_connection_id
            or not getattr(receipt, "request_id", "")
            or not getattr(receipt, "message_id", "")
        ):
            raise RuntimeError("employee activation incomplete receipt is invalid")
        service.commit_effect_transition(
            state.intent_id,
            effect_id=effect_id,
            effect_type="employee_status_reply",
            next_state=HireEffectState.COMMITTED,
            metadata={
                "send_request_id": receipt.request_id,
                "ingress_event_id": event_id,
                "reply_app_id": receipt.app_id,
                "reply_message_id": receipt.message_id,
                "generation": str(receipt.generation),
                "connection_id": receipt.connection_id,
                "main_bot_send_count": "0",
            },
        )
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
        history_days: int = 7,
    ) -> bool:
        data = self._data
        lifecycle = self._outbox_lifecycle
        ingress = self._ingress
        if data is None or lifecycle is None or ingress is None:
            return False
        metadata = record.metadata
        first = payload.normalized_parts[0]
        coordinates = (
            _bound_remote_coordinates(metadata, first)
            if isinstance(first, Mapping)
            else None
        )
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
            principal_id=metadata.sender_principal_id,
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
                rows = [
                    f"{item.ended_at[:19]} · {item.status} · {item.safe_summary_text}"
                    for item in result.records
                ]
                summary = (
                    f"最近 {history_days} 天暂无可见执行记录。"
                    if not rows
                    else "\n".join(rows)
                )
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
        self._drain_employee_outbox_once()
        return True

    def _drain_employee_outbox_once(self) -> bool:
        outbox = self._outbox
        delivery = self._outbox_delivery
        if outbox is None or delivery is None:
            return False
        outbox.rebuild_projection()
        pending = sorted(
            (
                record
                for record in outbox.state.by_outbox_id.values()
                if record.binding is None or record.binding.bound_snapshot_version < record.latest_version
            ),
            key=lambda record: (record.latest.created_at, record.outbox_id),
        )
        if not pending:
            return False
        delivery.deliver(pending[0].outbox_id)
        return True

    def _resolve_outbox_delivery_authority(
        self,
        record: Any,
    ) -> EmployeeDeliveryAuthority:
        if self._service is None or self._channels is None:
            raise RuntimeError("employee delivery authority is unavailable")
        projection = self._service.synchronize_projection()
        employee = projection.employees.get(record.agent_id)
        if employee is None or employee.tenant_key != record.tenant_key or not employee.bot_principal_id:
            raise RuntimeError("employee delivery identity is unavailable")
        principal = projection.bot_principals.get(employee.bot_principal_id)
        active = next(
            (
                state
                for state in self._service.list_states()
                if state.agent_id == record.agent_id and state.phase is HirePhase.ACTIVE
            ),
            None,
        )
        status = self._channels.status(record.agent_id)
        connection_id = getattr(status, "ready_metadata", {}).get(
            "connection_id",
            "",
        )
        if (
            principal is None
            or active is None
            or status is None
            or active.tenant_key != record.tenant_key
            or active.bot_principal_id != employee.bot_principal_id
            or principal.tenant_key != record.tenant_key
            or principal.agent_id != record.agent_id
            or getattr(status, "state", None) is not ChannelProcessState.READY
            or principal.app_id != active.app_id
            or getattr(status, "app_id", None) != active.app_id
            or getattr(status, "identity", {}).get("app_id") != active.app_id
            or getattr(status, "generation", None) != active.channel_generation
            or not isinstance(connection_id, str)
            or not connection_id
        ):
            raise RuntimeError("employee delivery Channel is not current")
        return EmployeeDeliveryAuthority(
            app_id=active.app_id,
            generation=active.channel_generation,
            connection_id=connection_id,
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
                    ingress.record_disposition(
                        acceptance_id,
                        state="terminal",
                        reason_code=record.reason_code,
                    )
                    reconciled += 1
                except IngressConflictError:
                    pass
        return reconciled

    def rewrap_employee_credential(
        self,
        *,
        agent_id: str,
        app_id: str,
        credential_ref: str,
    ) -> CredentialReceipt:
        """Drain employee clients around an atomic Vault key rewrap."""
        if self._vault is None:
            raise RuntimeError("employee credential Vault is unavailable")
        self.invalidate_employee_context(agent_id)
        receipt = self._vault.rewrap(
            credential_ref,
            agent_id,
            app_id,
        )
        self.reactivate_employee_context(agent_id)
        return receipt

    def _refresh_context_bindings(self, projection: ProjectionState) -> bool:
        if self._context_source_factory is None or self._service is None:
            return True
        try:
            with self._context_binding_lock:
                current: dict[str, tuple[str, str, int]] = {}
                states = self._service.list_states()
                for state in states:
                    if state.phase is not HirePhase.ACTIVE:
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
                non_active = {state.agent_id for state in states if state.phase is not HirePhase.ACTIVE}
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
                canonicalize_user_home_path(
                    getattr(settings, "autonomous_slock_storage_base", None)
                    or default_slock_storage_base()
                )
            )
            if self._data is None:
                self._data = build_employee_data_composition(
                    settings=settings,
                    writer=self._writer,
                    admin_principal_ids=frozenset(getattr(settings, "admin_user_ids", ()) or ()),
                    main_bot_app_id=getattr(settings, "app_id", ""),
                    agents_root=Path(legacy_base).expanduser() / "agents",
                    legacy_base=legacy_base,
                )
            backend = group_memory_backend
            if backend is None:
                backend = MemoryManager(str(Path(legacy_base).expanduser()))
                self._owns_group_memory_backend = True
            self._group_memory_backend = backend
            self._context_acl = parse_requester_acl(settings)

            def registry_provider() -> ProjectedAgentRegistry:
                assert self._service is not None
                return ProjectedAgentRegistry(
                    self._service.projection_state,
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
            group_reader = AuthorizedGroupMemoryReader(
                registry_provider=registry_provider,
                requester_acl=self._context_acl,
                backend=backend,
            )
            self._context_source_factory = source_factory
            self._context_service = EmployeeContextService(
                registry_provider=registry_provider,
                generation_authority=generation,
                requester_acl=self._context_acl,
                data_composition=self._data,
                group_memory_reader=group_reader,
                source_factory=source_factory,
                config=ThreadContextConfig.from_settings(settings),
                group_ledger=self._group_ledger,
            )
            self._context_blockers = ()
        except Exception as exc:
            logger.error(
                "employee Context composition unavailable: %s",
                type(exc).__name__,
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
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.get_running_loop().shutdown_default_executor()

    @staticmethod
    def _default_slash_factory(app_id: str, app_secret: str) -> _SlashReconciler:
        client = lark.Client.builder().app_id(app_id).app_secret(app_secret).build()
        return SlashCommandReconciler(LarkSlashCommandAPI(client))

    def _start_loop(self) -> None:
        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.call_soon(self._loop_ready.set)
            loop.run_forever()
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        self._loop_thread = threading.Thread(
            target=run,
            name="employee-hire-activities",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(5.0) or self._loop is None:
            raise RuntimeError("employee activity loop failed to start")

    def _submit_intent(self, intent_id: str) -> None:
        if self._closing or self._loop is None:
            raise RuntimeError("employee runtime is closing")
        with self._future_lock:
            existing = self._intent_futures.get(intent_id)
            if existing is not None and not existing.done():
                return
            future = asyncio.run_coroutine_threadsafe(
                self._configure_and_notify(intent_id),
                self._loop,
            )
            self._intent_futures[intent_id] = future
            self._futures.add(future)

        def complete(done: concurrent.futures.Future[Any]) -> None:
            with self._future_lock:
                self._futures.discard(done)
                if self._intent_futures.get(intent_id) is done:
                    self._intent_futures.pop(intent_id, None)
            try:
                done.result()
            except Exception as exc:
                logger.error(
                    "employee provisioning activity failed: %s",
                    type(exc).__name__,
                )

        future.add_done_callback(complete)

    def _submit_manifest_reauthorization(self, operation_id: str) -> None:
        service = self._require_service()
        self._submit_coroutine(
            service.run_manifest_reauthorization(operation_id),
            label="manifest reauthorization",
        )

    async def _configure_and_notify(self, intent_id: str) -> None:
        succeeded = False
        try:
            await self._configure_intent(intent_id)
            succeeded = True
        except Exception:
            current = self._require_service().get_state(intent_id)
            has_disposed_failure = current is not None and any(
                effect_state is HireEffectState.ACTION_REQUIRED
                for _effect_id, effect_state in current.effects
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
            if state.phase is HirePhase.ACTION_REQUIRED
            or any(
                effect_state is HireEffectState.ACTION_REQUIRED
                for _effect_id, effect_state in state.effects
            )
            else ""
        )
        if status:
            await self._retry_terminal_notification(state.intent_id, status)

    async def _notify_hire_terminal(
        self,
        state: DurableHireState,
        status: str,
    ) -> bool:
        if self._notification_async_lock is None:
            self._notification_async_lock = asyncio.Lock()
        async with self._notification_async_lock:
            current = self._require_service().get_state(state.intent_id)
            if (
                current is None
                or self._terminal_notification_status(current) != status
            ):
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
        notification = rebuild_hire_notification_projection(
            tuple(writer.replay())
        ).get(aggregate_id)
        if (
            notification is not None
            and notification.phase is HireNotificationPhase.COMMITTED
        ):
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
        if (
            current is None
            or self._terminal_notification_status(current) != status
        ):
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
            else getattr(receipt, "message_id", "")
            or f"ack:{message_uuid}"
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
        for delay in (0.0, *_RECOVERY_RETRY_DELAYS):
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
    ) -> None:
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

    async def _recover_runtime(
        self,
        pending_intents: list[str],
        status_reply_intents: list[str],
    ) -> None:
        failed_intents: list[str] = []
        if status_reply_intents:
            status_results = await asyncio.gather(
                *(
                    self._recover_status_reply_channel(intent_id)
                    for intent_id in status_reply_intents
                ),
                return_exceptions=True,
            )
            status_failures = sum(
                isinstance(result, BaseException) for result in status_results
            )
            if status_failures:
                logger.error(
                    "employee phase-status Channel recovery failed for %d intent(s)",
                    status_failures,
                )
                self._execution_blockers = ("status_reply_recovery",)
        if pending_intents:
            results = await asyncio.gather(
                *(self._configure_intent(intent_id, force_slash_refresh=True) for intent_id in pending_intents),
                return_exceptions=True,
            )
            failed_intents = [
                intent_id
                for intent_id, result in zip(pending_intents, results, strict=True)
                if isinstance(result, BaseException)
            ]
        self._start_monitor_in_loop()
        if failed_intents:
            retry_results = await asyncio.gather(
                *(self._retry_recovery_intent(intent_id) for intent_id in failed_intents),
                return_exceptions=True,
            )
            failures = sum(
                isinstance(result, BaseException) or result is False
                for result in retry_results
            )
            if any(isinstance(result, BaseException) for result in retry_results):
                logger.error(
                    "employee recovery could not be durably isolated for %d intent(s)",
                    failures,
                )
                self._execution_blockers = ("hire_recovery",)
                return
            if failures:
                logger.error(
                    "employee recovery isolated %d exhausted intent(s) as action_required",
                    failures,
                )
        await self._retry_terminal_notifications()
        self._resume_recoverable_activation_required_replies()
        self._recovery_trace.append("admission_open")
        self._require_service().mark_runtime_recovered()
        if not self._execution_blockers:
            self._start_dispatch_worker()

    async def _recover_status_reply_channel(self, intent_id: str) -> None:
        state = self._require_service().get_state(intent_id)
        if state is None:
            return
        await self._reconcile_slash(
            state,
            generation=state.channel_generation + 1,
            force_refresh=True,
        )
        state = self._require_service().get_state(intent_id) or state
        await self._start_channel(state, force_next_generation=True)

    async def _retry_recovery_intent(self, intent_id: str) -> bool:
        for delay in _RECOVERY_RETRY_DELAYS:
            if self._closing:
                raise RuntimeError("employee runtime is closing")
            await asyncio.sleep(delay)
            try:
                await self._configure_intent(
                    intent_id,
                    force_slash_refresh=True,
                )
                return True
            except Exception:
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
                        HirePhase.ACTIVE,
                        HirePhase.READY_PENDING_VERIFICATION,
                    }:
                        continue
                    try:
                        channel_status = channels.status(state.agent_id)
                        if self._closing:
                            return
                        if (
                            channel_status is not None
                            and channel_status.generation
                            == state.channel_generation
                            and (
                                channel_status.state
                                in {
                                    ChannelProcessState.CRASHED,
                                    ChannelProcessState.FAILED,
                                    ChannelProcessState.STOPPED,
                                }
                                or (
                                    channel_status.state
                                    is ChannelProcessState.STARTING
                                    and channel_status.error_code
                                    == "channel-reconnecting"
                                    and isinstance(
                                        channel_status.ready_metadata.get(
                                            "reconnecting_at"
                                        ),
                                        (int, float),
                                    )
                                    and time.time()
                                    - channel_status.ready_metadata[
                                        "reconnecting_at"
                                    ]
                                    >= _CHANNEL_RECONNECT_GRACE_SECONDS
                                )
                            )
                        ):
                            if self._closing:
                                return
                            service.begin_channel_revalidation(
                                state.intent_id,
                                observed_generation=state.channel_generation,
                            )
                            if self._closing:
                                return
                            self._submit_intent(state.intent_id)
                        elif (
                            state.phase is HirePhase.READY_PENDING_VERIFICATION
                            and state.verification_expires_at <= time.time()
                        ):
                            if self._closing:
                                return
                            self._renew_activation_challenge(state)
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
            effect_state is HireEffectState.ACTION_REQUIRED
            for _effect_id, effect_state in state.effects
        )
        return (
            "ready"
            if state.phase is HirePhase.READY_PENDING_VERIFICATION
            else "active"
            if state.phase is HirePhase.ACTIVE
            else "action_required"
            if state.phase is HirePhase.ACTION_REQUIRED
            or has_action_required_effect
            else ""
        )

    def _renew_activation_challenge(
        self,
        state: DurableHireState,
    ) -> VerificationChallenge:
        with self._activation_lock:
            service = self._require_service()
            current = service.get_state(state.intent_id)
            if (
                current is None
                or current.phase is not HirePhase.READY_PENDING_VERIFICATION
                or current.verification_consumed
            ):
                raise RuntimeError("employee activation challenge is not renewable")
            router = self._verification_router
            if router is None:
                raise RuntimeError("employee Verification Router unavailable")
            challenge = router.issue_challenge(
                VerificationBinding(
                    hire_intent_id=current.intent_id,
                    tenant_key=current.tenant_key,
                    app_id=current.app_id,
                    agent_id=current.agent_id,
                    generation=current.channel_generation,
                    requester_principal_id=current.requester_principal_id,
                    requester_union_id=current.requester_union_id,
                    expected_slash_spec_hash=current.slash_spec_hash,
                )
            )
            service.renew_activation_verification(challenge)
            self._challenges[current.intent_id] = challenge
            return challenge

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
        if state.phase in {HirePhase.PROVISIONING_APP, HirePhase.STORING_CREDENTIAL}:
            state = await service.run_provisioning(intent_id)
        if state.phase not in {HirePhase.CONFIGURING, HirePhase.VALIDATING}:
            return
        for launch_attempt in range(2):
            state = service.get_state(intent_id) or state
            generation = self._target_channel_generation(state)
            slash = await self._reconcile_slash(
                state,
                generation=generation,
                force_refresh=force_slash_refresh,
            )
            state = service.get_state(intent_id) or state
            try:
                channel = await self._start_channel(state)
            except RuntimeError:
                if launch_attempt == 0:
                    continue
                raise
            binding = VerificationBinding(
                hire_intent_id=state.intent_id,
                tenant_key=state.tenant_key,
                app_id=state.app_id,
                agent_id=state.agent_id,
                generation=channel.generation,
                requester_principal_id=state.requester_principal_id,
                requester_union_id=state.requester_union_id,
                expected_slash_spec_hash=slash.spec_hash,
            )
            router = self._verification_router
            if router is None:
                raise RuntimeError("employee Verification Router unavailable")
            challenge = router.issue_challenge(binding)
            service.begin_activation_verification(challenge)
            self._challenges[state.intent_id] = challenge
            if self._automatic_activation:
                service.commit_automatic_activation(
                    state.intent_id,
                    activated_at=time.time(),
                )
                self._challenges.pop(state.intent_id, None)
            return

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
            return VerifiedSlashState(
                spec_hash=current.slash_spec_hash,
                observed_hash=current.slash_observed_hash,
                observed=(),
            )
        if current.effect_state(effect_id) is not HireEffectState.EXECUTING:
            raise RuntimeError("Slash reconciliation requires manual action")
        if self._vault is None or self._slash_factory is None:
            raise RuntimeError("Slash composition unavailable")
        secret = await asyncio.to_thread(
            self._vault.resolve,
            current.credential_ref,
            current.agent_id,
            current.app_id,
        )
        reconciler = self._slash_factory(current.app_id, secret)
        try:
            verified = await reconciler.reconcile()
        finally:
            del secret
        service.commit_effect_transition(
            state.intent_id,
            effect_id=effect_id,
            effect_type="slash_reconciliation",
            next_state=HireEffectState.COMMITTED,
            metadata={
                "slash_spec_hash": verified.spec_hash,
                "slash_observed_hash": verified.observed_hash,
                "slash_verified_at": str(time.time()),
            },
        )
        return verified

    async def _start_channel(
        self,
        state: DurableHireState,
        *,
        force_next_generation: bool = False,
    ) -> Any:
        service = self._require_service()
        generation = (
            state.channel_generation + 1
            if force_next_generation
            else self._target_channel_generation(state)
        )
        effect_id = f"channel-start:{generation}"
        current = service.get_state(state.intent_id) or state
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
        try:
            status = await asyncio.to_thread(
                self._channels.start,
                state.agent_id,
                state.app_id,
                state.credential_ref,
                generation,
                self._event_callback(state.intent_id, generation),
            )
        except Exception as exc:
            service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_channel_start",
                next_state=HireEffectState.ACTION_REQUIRED,
                metadata={"error_code": f"start-{type(exc).__name__}"},
            )
            raise RuntimeError("employee Channel start failed") from None
        identity_app_id = status.identity.get("app_id")
        connection_id = status.ready_metadata.get("connection_id")
        if (
            status.state is not ChannelProcessState.READY
            or identity_app_id != state.app_id
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
            raise RuntimeError("employee Channel did not become ready")
        current = service.get_state(state.intent_id) or state
        if current.effect_state(effect_id) is HireEffectState.EXECUTING:
            service.commit_effect_transition(
                state.intent_id,
                effect_id=effect_id,
                effect_type="employee_channel_start",
                next_state=HireEffectState.COMMITTED,
                metadata={
                    "app_id": state.app_id,
                    "generation": str(generation),
                    "identity_app_id": identity_app_id,
                    "connection_id": connection_id,
                    "channel_verified_at": str(time.time()),
                },
            )
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
        if state.phase is HirePhase.VALIDATING and state.channel_generation > 0:
            return generation + 1
        return generation

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
                await asyncio.to_thread(self._handle_control_ingress, acceptance_id)
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
        coordinates = _bound_remote_coordinates(record.metadata, part)
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
        if (
            not isinstance(sender_id, str)
            or not sender_id
            or not isinstance(sender_union_id, str)
            or service is None
        ):
            return False
        service.synchronize_projection()
        states = tuple(
            state
            for state in service.list_states()
            if state.tenant_key == record.metadata.tenant_key
            and state.agent_id == record.metadata.agent_id
        )
        if len(states) != 1:
            return False
        owner_principal_id = states[0].requester_principal_id
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
            transport_principal_id=record.metadata.bot_principal_id,
            transport_event_id=record.metadata.event_id,
            payload=GroupEventPayload(
                sender_id=canonical_sender_id,
                sender_id_type=str(part.get("sender_id_type", "")),
                sender_type=str(part.get("sender_type", "")),
                sender_tenant_key=str(part.get("sender_tenant_key", "")),
                text=text,
                timestamp=time.time(),
            ),
        )
        return True

    def _complete_status_activation(
        self,
        *,
        state: DurableHireState,
        challenge: VerificationChallenge,
        generation: int,
        event_id: str,
        message_id: str,
        sender_id: str,
        sender_union_id: str,
        command: str,
        is_p2p: bool,
        reply_options: dict[str, Any],
        received_at: float | None,
    ) -> bool:
        """Anchor the employee reply and commit a verified activation."""

        with self._activation_lock:
            service = self._require_service()
            current = service.get_state(state.intent_id)
            if current is None or current.phase is not HirePhase.READY_PENDING_VERIFICATION:
                return current is not None and current.phase is HirePhase.ACTIVE
            audit = self._main_bot_send_audit
            if audit is None:
                raise RuntimeError("main Bot send audit is unavailable")
            activation_audit = self._main_bot_activation_audit
            if (
                activation_audit is None
                or activation_audit.activation_fence_ready is not True
            ):
                raise RuntimeError("main Bot activation fence is unavailable")
            preflight_audited_at = time.time()
            audit_target_hashes = self._activation_audit_target_hashes(
                current,
                ingress_message_id=message_id,
            )
            preflight_main_bot_send_count = self._count_main_bot_target_attempts(
                audit,
                tenant_key=current.tenant_key,
                target_hashes=audit_target_hashes,
                start=challenge.issued_at,
                end=preflight_audited_at,
            )
            if preflight_main_bot_send_count != 0:
                router = self._verification_router
                if router is None:
                    raise RuntimeError("employee Verification Router unavailable")
                replacement = router.issue_challenge(
                    VerificationBinding(
                        hire_intent_id=current.intent_id,
                        tenant_key=current.tenant_key,
                        app_id=current.app_id,
                        agent_id=current.agent_id,
                        generation=current.channel_generation,
                        requester_principal_id=current.requester_principal_id,
                        requester_union_id=current.requester_union_id,
                        expected_slash_spec_hash=current.slash_spec_hash,
                    )
                )
                current = service.reissue_activation_verification_after_audit_collision(
                    replacement
                )
                self._challenges[current.intent_id] = replacement
                self._send_activation_retry_notice(
                    state=current,
                    generation=generation,
                    event_id=event_id,
                    sender_id=sender_id,
                    message=(
                        "Activation window reset after a conflicting main Bot send. "
                        "Send /status again."
                    ),
                )
                return False
            effect_id = f"verification-status-reply:{event_id}"
            effect_state = current.effect_state(effect_id)
            if effect_state is None:
                current = service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_status_reply",
                    next_state=HireEffectState.PREPARED,
                )
                effect_state = current.effect_state(effect_id)
            if effect_state is HireEffectState.PREPARED:
                current = service.commit_effect_transition(
                    state.intent_id,
                    effect_id=effect_id,
                    effect_type="employee_status_reply",
                    next_state=HireEffectState.EXECUTING,
                )
                effect_state = current.effect_state(effect_id)
            if effect_state is HireEffectState.COMMITTED:
                self._send_activation_incomplete_notice(
                    state=current,
                    generation=generation,
                    event_id=event_id,
                    sender_id=sender_id,
                )
                return False
            if effect_state is not HireEffectState.EXECUTING or self._channels is None:
                return False
            stable_reply_uuid = _stable_feishu_uuid(
                f"employee-activation-preflight:{event_id}"
            )
            receipt = self._channels.send(
                current.agent_id,
                generation=generation,
                target=sender_id,
                message={
                    "text": (
                        f"{current.employee_name} activation verification started; "
                        "not active yet."
                    )
                },
                options={"uuid": stable_reply_uuid, **reply_options},
            )
            if getattr(receipt, "success", False) is not True:
                raise RuntimeError("employee status reply was not acknowledged")
            send_request_id = getattr(receipt, "request_id", "")
            reply_app_id = getattr(receipt, "app_id", "")
            reply_generation = getattr(receipt, "generation", 0)
            reply_connection_id = getattr(receipt, "connection_id", "")
            reply_message_id = getattr(receipt, "message_id", "")
            if (
                not isinstance(send_request_id, str)
                or not send_request_id
                or reply_app_id != current.app_id
                or reply_generation != generation
                or reply_connection_id != current.channel_connection_id
                or not isinstance(reply_message_id, str)
                or not reply_message_id
            ):
                raise RuntimeError("employee status reply receipt is invalid")
            ingress_at = (
                max(time.time(), challenge.issued_at)
                if received_at is None
                else received_at
            )
            fence = activation_audit.activation_fence(
                current.tenant_key,
                audit_target_hashes,
            )
            with fence:
                evaluated_at = time.time()
                main_bot_send_count = self._count_main_bot_target_attempts(
                    audit,
                    tenant_key=current.tenant_key,
                    target_hashes=audit_target_hashes,
                    start=challenge.issued_at,
                    end=evaluated_at,
                )
                reply_effect_metadata = {
                    "send_request_id": send_request_id,
                    "ingress_event_id": event_id,
                    "reply_app_id": reply_app_id,
                    "reply_message_id": reply_message_id,
                    "generation": str(reply_generation),
                    "connection_id": reply_connection_id,
                    "main_bot_send_count": str(main_bot_send_count),
                }
                if main_bot_send_count != 0:
                    current = service.commit_effect_transition(
                        state.intent_id,
                        effect_id=effect_id,
                        effect_type="employee_status_reply",
                        next_state=HireEffectState.COMMITTED,
                        metadata=reply_effect_metadata,
                    )
                    self._send_activation_incomplete_notice(
                        state=current,
                        generation=generation,
                        event_id=event_id,
                        sender_id=sender_id,
                    )
                    return False
                coordinates = VerificationCoordinates(
                    hire_intent_id=current.intent_id,
                    tenant_key=current.tenant_key,
                    app_id=current.app_id,
                    agent_id=current.agent_id,
                    generation=generation,
                    nonce=challenge.nonce,
                )
                router = self._verification_router
                if router is None:
                    raise RuntimeError("employee Verification Router unavailable")
                decision = router.evaluate_for_atomic_commit(
                    challenge,
                    slash=SlashVerificationEvidence(
                        coordinates=coordinates,
                        desired_spec_hash=current.slash_spec_hash,
                        observed_spec_hash=current.slash_observed_hash,
                        reconciled=current.slash_spec_hash
                        == current.slash_observed_hash,
                        verified_at=current.slash_verified_at,
                    ),
                    channel=ChannelVerificationEvidence(
                        coordinates=coordinates,
                        identity_app_id=current.channel_identity_app_id,
                        connection_id=current.channel_connection_id,
                        ready=True,
                        verified_at=current.channel_verified_at,
                    ),
                    ingress=TenantIngressEvidence(
                        coordinates=coordinates,
                        event_id=event_id,
                        message_id=message_id,
                        sender_principal_id=sender_id,
                        sender_union_id=sender_union_id,
                        command=command,
                        is_p2p=is_p2p,
                        reply_succeeded=True,
                        reply_app_id=reply_app_id,
                        employee_send_request_id=send_request_id,
                        main_bot_send_count=main_bot_send_count,
                        received_at=ingress_at,
                    ),
                    current_generation=generation,
                    now=evaluated_at,
                )
                if decision.outcome is not VerificationOutcome.READY:
                    current = service.commit_effect_transition(
                        state.intent_id,
                        effect_id=effect_id,
                        effect_type="employee_status_reply",
                        next_state=HireEffectState.COMMITTED,
                        metadata=reply_effect_metadata,
                    )
                    self._send_activation_incomplete_notice(
                        state=current,
                        generation=generation,
                        event_id=event_id,
                        sender_id=sender_id,
                    )
                    return False
                active = service.commit_activation(
                    decision,
                    reply_effect_id=effect_id,
                    reply_effect_metadata=reply_effect_metadata,
                )
            self._send_activation_success_notice(
                state=active,
                event_id=event_id,
                sender_id=sender_id,
            )
            if not self._refresh_context_bindings(service.projection_state):
                self._context_blockers = ("context_binding_sync",)
            self._schedule_activation_notification(active)
            return True

    @staticmethod
    def _activation_audit_target_hashes(
        state: DurableHireState,
        *,
        ingress_message_id: str,
    ) -> tuple[str, ...]:
        """Cover every known main-Bot reply/create coordinate for the requester."""

        canonical_ingress_hash = ingress_message_id.removeprefix("om_")
        if re.fullmatch(r"[0-9a-f]{64}", canonical_ingress_hash) is None:
            canonical_ingress_hash = hashlib.sha256(
                ingress_message_id.encode()
            ).hexdigest()
        target_hashes = {canonical_ingress_hash}
        for target in (
            state.requester_union_id,
            state.requester_principal_id,
            state.chat_id,
            state.message_id,
        ):
            if target:
                target_hashes.add(hashlib.sha256(target.encode()).hexdigest())
        return tuple(sorted(target_hashes))

    @staticmethod
    def _count_main_bot_target_attempts(
        audit: MainBotSendAudit,
        *,
        tenant_key: str,
        target_hashes: tuple[str, ...],
        start: float,
        end: float,
    ) -> int:
        total = 0
        for target_hash in target_hashes:
            count = audit(tenant_key, target_hash, start, end)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise RuntimeError("main Bot send audit is invalid")
            total += count
        return total

    def _schedule_activation_notification(self, state: DurableHireState) -> None:
        loop = self._loop
        if loop is None or self._notification_status is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._retry_terminal_notification(state.intent_id, "active"),
            loop,
        )
        try:
            future.result(timeout=15.0)
            service = self._require_service()
            with service.employee_dispatch_guard():
                service.synchronize_projection_unlocked()
        except Exception as exc:
            logger.error(
                "employee activation notification failed closed: %s",
                type(exc).__name__,
            )

    def _require_service(self) -> ProductionEmployeeHireService:
        if self._service is None:
            raise RuntimeError("employee hire service unavailable")
        return self._service


__all__ = ["EmployeeDepartmentRuntime", "RuntimeReadiness"]
