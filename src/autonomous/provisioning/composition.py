"""Single-owner production composition for durable visible employee hiring."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
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

from src.utils.async_helpers import safe_wait_for
from src.utils.path import canonicalize_user_home_path

from ...trust.models import ActorKind, EffectiveTrust, TrustZone
from ...trust.registry import ManagedGroupRegistry
from ...trust.resolver import TrustZoneResolver
from ..acceptance.main_bot_audit import MainBotSendAuditLog
from ..authorization import EmployeeAuthorizationScope
from ..context.group_ledger import GroupContextLedger, GroupEventPayload
from ..context.group_memory import EmployeeGroupMemoryStore
from ..context.lark_source import LarkEmployeeMessageSourceFactory
from ..context.models import (
    AuthorizedContextRequest,
    ContextUnavailableError,
    ThreadContextConfig,
)
from ..context.runtime import (
    RuntimeEmployeeGenerationAuthority,
    parse_requester_acl,
)
from ..context.service import AuthorizedGroupMemoryReader, EmployeeContextService
from ..context.source import EmployeeMessageSourceFactory
from ..data.composition import (
    EmployeeDataComposition,
    LegacyEmployeeDataUnsupportedError,
    build_employee_data_composition,
)
from ..data.keyring import EmployeeDataKeyring
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
from ..ingress.service import (
    EmployeeIngressService,
    IngressBlobError,
    IngressBlobRetryableError,
    IngressConflictError,
)
from ..ingress.targeted_task import (
    TargetedTaskParseResult,
    TargetedTaskState,
    is_group_slash_observation,
)
from ..journal.anchor import FileAnchor
from ..journal.frame import JournalEvent
from ..journal.projections import ProjectionState
from ..journal.writer import JournalWriter
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
from ..outbox.projection import OutboxProjectionState
from ..outbox.service import EmployeeOutboxService
from ..supervisor.employee_channels import (
    ChannelProcessState,
    EmployeeChannelSupervisor,
)
from ..team.coordinator import SessionCoordinatorDecisionProvider
from ..team.service import (
    EmployeeTeamService,
    TeamAttemptResult,
    TeamTarget,
)
from ..workforce.credential_vault import CredentialVault
from ..workforce.identity import default_employee_storage_base
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

logger = logging.getLogger(__name__)

_RECOVERY_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)
_CHANNEL_RECONNECT_GRACE_SECONDS = 180.0




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
        self._notify(
            message_id,
            chat_id,
            result,
            idempotency_key=idempotency_key,
            tenant_key=tenant_key,
            requester_principal_id=requester_principal_id,
        )


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


class _TeamMembershipHealth:
    """Treat an unavailable or ambiguous activated Team as degraded."""

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


@dataclass(frozen=True)
class EmployeeRecoverySummary:
    eligible: int = 0
    recovered: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass(frozen=True, slots=True)
class ReadyEmployeeIngressTarget:
    """Frozen READY authority used to correlate one cross-App handoff."""

    tenant_key: str
    chat_id: str
    agent_id: str
    bot_principal_id: str
    app_id: str
    bot_open_id: str


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
        self._team_runtime: object | None = None
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
        self._recovery_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._recovery_future: concurrent.futures.Future[EmployeeRecoverySummary] | None = None
        self._recovery_attempts = 0
        self._closing = False
        self._notification_status: Callable[[DurableHireState, str], object] | None = None
        self._owned_main_bot_send_audit: MainBotSendAuditLog | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._notification_async_lock: asyncio.Lock | None = None
        self._core_recovered = False
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
        notification_link: Callable[[DurableHireState, str, int], object] | None = None,
        notification_status: Callable[[DurableHireState, str], object] | None = None,
        team_notification: Callable[..., object] | None = None,
        context_source_factory: EmployeeMessageSourceFactory | None = None,
        group_memory_backend: Any = None,
        team_runtime: object | None = None,
        employee_environment_provider: EmployeeEnvironmentProvider | None = None,
        membership_health: Any = None,
        manager_client_factory: Callable[[], Any] | None = None,
        recover_immediately: bool = True,
        managed_group_registry: ManagedGroupRegistry | None = None,
        managed_group_owner_id: str = "",
    ) -> EmployeeDepartmentRuntime:
        limit = getattr(settings, "autonomous_visible_employee_limit", 0)
        if limit == 0:
            return cls(
                blockers=("visible_employee_limit",),
                managed_group_registry=managed_group_registry,
                managed_group_owner_id=managed_group_owner_id,
            )
        if notification_link is None:
            return cls(
                blockers=("registration_notifier",),
                managed_group_registry=managed_group_registry,
                managed_group_owner_id=managed_group_owner_id,
            )
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
        runtime._team_runtime = team_runtime
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
        runtime._notification_status = notification_status
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
                runtime_recovery_ready=False,
                workspace_projector=(
                    runtime._data.workspace_projector
                    if runtime._data is not None
                    else None
                ),
                admin_principal_ids_provider=lambda: frozenset(
                    getattr(settings, "admin_user_ids", ()) or ()
                ),
            )
            runtime._service = service
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
                if team_runtime is None:
                    raise RuntimeError("team coordinator project resolver is unavailable")

                def resolve_coordinator_cwd(run: object) -> str:
                    binding = team_runtime.resolve_employee_engine(chat_id=run.chat_id)
                    return binding.canonical_root

                decision_provider = SessionCoordinatorDecisionProvider(
                    tool=getattr(settings, "autonomous_team_coordinator_tool", "coco"),
                    model=getattr(settings, "autonomous_team_coordinator_model", ""),
                    profile=getattr(settings, "autonomous_team_coordinator_profile", ""),
                    effort=getattr(settings, "autonomous_team_coordinator_effort", ""),
                    cwd_resolver=resolve_coordinator_cwd,
                )
                runtime._team = EmployeeTeamService(
                    writer=runtime._writer,
                    backend=_RuntimeTeamBackend(runtime, team_notification),
                    attempt_timeout_seconds=float(
                        settings.autonomous_team_step_timeout_seconds
                    ),
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
    def dispatch_coordinator(self) -> EmployeeDispatchCoordinator | None:
        return self._dispatch



    @property
    def main_bot_outbound_audit(self) -> MainBotSendAuditLog | None:
        return self._owned_main_bot_send_audit

    def readiness(self) -> RuntimeReadiness:
        return self.hire_readiness()









    def hire_readiness(self) -> RuntimeReadiness:
        if self._service is None:
            return RuntimeReadiness(False, self._blockers or ("not_composed",))
        activation_audit = self._owned_main_bot_send_audit
        if (
            activation_audit is None
            or activation_audit.activation_fence_ready is not True
        ):
            return RuntimeReadiness(False, ("main_bot_activation_fence",))
        service_readiness: HireReadiness = self._service.readiness()
        return RuntimeReadiness(service_readiness.ready, service_readiness.blockers)


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

    def recover(self) -> EmployeeRecoverySummary:
        """Run startup recovery once and wait for its actual terminal result."""

        with self._recovery_lock:
            future = self._recovery_future
            leader = future is None
            if future is None:
                future = concurrent.futures.Future()
                self._recovery_future = future
                self._recovery_attempts += 1
                attempt_number = self._recovery_attempts
        if not leader:
            return future.result()
        try:
            summary = self._recover_once()
        except BaseException as exc:
            future.set_exception(exc)
            with self._recovery_lock:
                if (
                    self._recovery_future is future
                    and isinstance(exc, Exception)
                    and attempt_number < 2
                    and not self._closing
                ):
                    self._recovery_future = None
            raise
        future.set_result(summary)
        return summary

    def _recover_once(self) -> EmployeeRecoverySummary:
        """Replay durable state, dispose retired prompts, and resume automatically."""
        if self._service is None:
            return EmployeeRecoverySummary()
        self._core_recovered = False
        self._service.recover()
        repair = (
            self._service.recover_replay_safe_action_required()
            if self._runtime_enabled
            else None
        )
        base_summary = EmployeeRecoverySummary(
            eligible=0 if repair is None else repair.eligible,
            recovered=0,
            skipped=0 if repair is None else repair.skipped,
            failed=0 if repair is None else repair.failed,
        )
        repaired_intents = set(
            () if repair is None else repair.repaired_intent_ids
        )
        if repair is not None and repair.failed:
            raise RuntimeError(
                "durable ACTION_REQUIRED repair failed: "
                f"{repair.failed}/{repair.eligible} intents"
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
                self._recover_employee_data()
            except Exception as exc:
                logger.error(
                    "employee data recovery failed closed: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self._execution_blockers = ("employee_data_recovery",)
        if not self._execution_blockers and self._group_ledger is not None:
            try:
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
                self._reconcile_retired_activation_ingress()
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
        self._core_recovered = (
            not self._execution_blockers and not self._context_blockers
        )
        if not self._runtime_enabled:
            self._service.mark_runtime_recovered()
            return base_summary
        pending_intents: list[str] = []
        recovery_states = tuple(self._service.list_states())
        for state in recovery_states:
            if state.intent_id in repaired_intents:
                pending_intents.append(state.intent_id)
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
                for state in recovery_states
            )
        )
        if pending_intents or recover_notifications:
            recovery_future = self._submit_coroutine(
                self._recover_runtime(
                    list(dict.fromkeys(pending_intents)),
                    repaired_intents=repaired_intents,
                    base_summary=base_summary,
                ),
                label="recovery",
            )
            return recovery_future.result()
        else:
            if not base_summary.failed:
                self._service.mark_runtime_recovered()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._start_monitor_in_loop)
            if execution_recovered:
                self._start_dispatch_worker()
            return base_summary





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
                    getattr(settings, "autonomous_employee_storage_base", None)
                    or default_employee_storage_base()
                )
            )
            self._data = build_employee_data_composition(
                settings=settings,
                writer=self._writer,
                keyring=self._data_keyring,
                admin_principal_ids=frozenset(getattr(settings, "admin_user_ids", ()) or ()),
                main_bot_app_id=getattr(settings, "app_id", ""),
                agents_root=Path(legacy_base).expanduser() / "agents",
                subject_resolver=self._resolve_data_subject,
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
            self._execution_blockers = (
                "legacy_employee_data_unsupported"
                if isinstance(exc, LegacyEmployeeDataUnsupportedError)
                else "employee_ingress"
            ,)

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

    def _recover_employee_data(self) -> None:
        data = self._data
        if data is None:
            return
        data.service.rebuild_projection()
        if data.state.data_authority.mode != "canonical":
            raise LegacyEmployeeDataUnsupportedError(
                "legacy employee data requires an offline migration"
            )
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
            or self._team_runtime is None
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
                getter = getattr(self._team_runtime, "get_activated_engine", None)
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
                    self._team_runtime.get_activated_engine(chat_id)
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
                getattr(settings, "autonomous_employee_storage_base", None)
                or default_employee_storage_base()
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
        if self._team_runtime is None or not callable(getattr(self._team_runtime, "employee_activation_guard", None)):
            self._execution_blockers = ("team_runtime",)
            return
        if not callable(self._environment_provider):
            self._execution_blockers = ("employee_environment",)
            return
        try:
            legacy_base = str(
                canonicalize_user_home_path(
                    getattr(settings, "autonomous_employee_storage_base", None)
                    or default_employee_storage_base()
                )
            )

            def registry_provider() -> ProjectedAgentRegistry:
                assert self._service is not None
                return ProjectedAgentRegistry(
                    self._service.synchronize_projection(),
                    storage_base_path=legacy_base,
                )

            health = membership_health or _TeamMembershipHealth(self._team_runtime)
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
                team_runtime=self._team_runtime,
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
                            self._team_runtime.get_activated_engine(chat_id),
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
                employee_session_idle_ttl_seconds=getattr(
                    settings,
                    "autonomous_employee_session_idle_ttl_seconds",
                    900.0,
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

        if not isinstance(payload, EmployeeIngressPayload):
            return None
        router = self._router
        metadata = getattr(record, "metadata", None)
        part = (
            payload.normalized_parts[0]
            if len(payload.normalized_parts) == 1
            else None
        )
        if (
            not isinstance(metadata, EmployeeIngressMetadata)
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
        if not isinstance(metadata, EmployeeIngressMetadata):
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
            and state.phase is HirePhase.ACTIVE
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
        coordinates = _bound_remote_coordinates(metadata, part)
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
                routed = router.state.by_acceptance_id.get(acceptance_id)
                if routed is not None and routed.state in {
                    "queued",
                    "dispatching",
                    "terminal",
                }:
                    continue
                eligibility = getattr(router, "is_inbox_candidate_eligible", None)
                if callable(eligibility) and not eligibility(acceptance_id):
                    continue
                try:
                    payload = ingress.get_payload(acceptance_id)
                except IngressBlobRetryableError:
                    router.defer_inbox_candidate(acceptance_id)
                    worked = True
                    continue
                except IngressBlobError:
                    # Let the Router durably converge every authenticated
                    # permanent payload failure to inbox_not_dispatchable.
                    router.route(acceptance_id)
                    worked = True
                    continue
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
                try:
                    owner_p2p_requester = self._owner_p2p_requester(
                        record,
                        payload,
                    )
                except Exception:
                    owner_p2p_requester = None
                targeted_group_task = self._authorized_targeted_group_task(
                    record,
                    payload,
                )
                try:
                    trust = self._managed_employee_ingress_trust(
                        record,
                        payload,
                    )
                except Exception:
                    trust = self._unknown_employee_ingress_trust()
                if self._handle_control_ingress(
                    acceptance_id,
                    targeted_group_task=targeted_group_task,
                    targeted_group_task_classified=True,
                ):
                    worked = True
                    continue
                if (
                    targeted_group_task is not None
                    and targeted_group_task.state
                    is TargetedTaskState.INDETERMINATE
                ):
                    # One independent Router pass is the bounded recovery for
                    # a classifier dependency failure.  The Router either
                    # queues the authenticated command or records an explicit
                    # terminal denial; this record never remains a hot retry.
                    self._record_employee_ingress_group_event(acceptance_id)
                    routed = router.state.by_acceptance_id.get(acceptance_id)
                    if routed is None or routed.state not in {
                        "queued",
                        "dispatching",
                        "terminal",
                    }:
                        router.route(acceptance_id)
                    worked = True
                    continue
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
                    worked = True
                    continue
                if (
                    owner_p2p_requester is None
                    and targeted_group_task is None
                    and trust is not None
                    and trust.actor is ActorKind.EMPLOYEE
                ):
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
                # Employee Bot subscriptions also observe group slash commands
                # owned by the main Bot.  Consume those after employee-specific
                # controls so main-Bot slash commands cannot be routed
                # into every employee mailbox as coding work.
                if (
                    owner_p2p_requester is None
                    and self._handle_main_bot_group_command_ingress(
                        acceptance_id,
                        targeted_group_task=targeted_group_task,
                    )
                ):
                    worked = True
                    continue
                # The ingress payload may be reclaimed as soon as dispatch reaches a
                # terminal disposition.  Project shared group context on this same
                # serialized path before routing so context assembly cannot race GC.
                if owner_p2p_requester is None:
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
        if self._fire is not None:
            try:
                worked = bool(self._fire.reconcile_draining()) or worked
            except Exception as exc:
                logger.error(
                    "employee drain reconciliation failed closed: %s",
                    type(exc).__name__,
                )
        worked = self._reconcile_terminal_ingress() > 0 or worked
        worked = self._drain_employee_outbox_once() or worked
        gc_now = time.monotonic()
        if gc_now < getattr(self, "_employee_dispatch_next_gc_at", 0.0):
            return worked
        if self._outbox is not None:
            worked = self._outbox.gc_superseded_snapshots() > 0 or worked
        worked = ingress.gc_terminal_payloads() > 0 or worked
        self._employee_dispatch_next_gc_at = gc_now + 60.0
        return worked

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

        if (
            not isinstance(bot_open_id, str)
            or not bot_open_id.startswith("ou_")
            or len(bot_open_id) > 256
        ):
            return None
        return self._ready_employee_ingress_targets(
            tenant_key=tenant_key,
            chat_id=chat_id,
        ).get(bot_open_id)

    def _ready_employee_ingress_targets(
        self,
        *,
        tenant_key: str | None = None,
        chat_id: str | None = None,
    ) -> dict[str, ReadyEmployeeIngressTarget]:
        scoped = tenant_key is not None or chat_id is not None
        if scoped and (
            not isinstance(tenant_key, str)
            or not tenant_key
            or not isinstance(chat_id, str)
            or not chat_id
        ):
            return {}
        service = self._service
        channels = self._channels
        if service is None or channels is None:
            return {}
        projection = service.synchronize_projection()
        result: dict[str, ReadyEmployeeIngressTarget] = {}
        ambiguous: set[str] = set()
        for employee in projection.employees.values():
            if (
                employee.state is not EmployeeState.ACTIVE
                or employee.worker_type is not WorkerType.VISIBLE
                or not employee.bot_principal_id
                or (
                    scoped
                    and (
                        employee.tenant_key != tenant_key
                        or chat_id not in employee.member_groups
                    )
                )
            ):
                continue
            principal = projection.bot_principals.get(employee.bot_principal_id)
            status = channels.status(employee.agent_id)
            identity = getattr(status, "identity", None)
            open_id = identity.get("open_id") if isinstance(identity, Mapping) else None
            if (
                principal is not None
                and principal.tenant_key == employee.tenant_key
                and principal.agent_id == employee.agent_id
                and getattr(status, "state", None) is ChannelProcessState.READY
                and getattr(status, "agent_id", None) == employee.agent_id
                and getattr(status, "tenant_key", None) == employee.tenant_key
                and getattr(status, "bot_principal_id", None)
                == employee.bot_principal_id
                and getattr(status, "app_id", None) == principal.app_id
                and isinstance(open_id, str)
                and open_id.startswith("ou_")
            ):
                target = ReadyEmployeeIngressTarget(
                    tenant_key=employee.tenant_key,
                    chat_id=chat_id or "",
                    agent_id=employee.agent_id,
                    bot_principal_id=employee.bot_principal_id,
                    app_id=principal.app_id,
                    bot_open_id=open_id,
                )
                existing = result.get(open_id)
                if existing is not None and existing != target:
                    ambiguous.add(open_id)
                else:
                    result[open_id] = target
        for open_id in ambiguous:
            result.pop(open_id, None)
        return result

    def wait_for_employee_message_acceptance(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
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
            not isinstance(value, str)
            or not value
            or len(value) > 256
            or safe_identifier.fullmatch(value) is None
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
        indexed_message_id = "om_" + hashlib.sha256(
            message_id.encode("utf-8")
        ).hexdigest()
        acceptance = ingress.wait_for_anchored_message_acceptance(
            tenant_key=tenant_key,
            agent_id=agent_id,
            bot_principal_id=bot_principal_id,
            app_id=app_id,
            event_type="im.message.receive_v1",
            chat_id=indexed_chat_id,
            message_id=indexed_message_id,
            timeout=timeout,
        )
        return acceptance is not None

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
                and getattr(status, "state", None)
                is ChannelProcessState.READY
                and getattr(status, "tenant_key", None)
                == metadata.tenant_key
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
        for acceptance_id, record in tuple(
            ingress.state.by_acceptance_id.items()
        ):
            if (
                record.disposition is not None
                or record.metadata.event_id not in retired_event_ids
            ):
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
            return record.disposition.reason_code.startswith(
                ("stop_", "membership_", "history_", "memory_", "status_")
            )

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
        first = (
            payload.normalized_parts[0]
            if len(payload.normalized_parts) == 1
            else None
        )
        is_membership_event = (
            isinstance(first, Mapping)
            and first.get("type") == "membership_event"
        )
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
        owner_p2p_requester = (
            None
            if is_membership_event
            else self._owner_p2p_requester(record, payload)
        )
        if is_membership_event:
            remote_chat_id = first.get("remote_chat_id")
            if (
                not isinstance(remote_chat_id, str)
                or not self._membership_event_transport_is_current(
                    record.metadata,
                    remote_chat_id,
                )
            ):
                return finish("ignored", "membership_unmanaged")
        else:
            try:
                trust = self._managed_employee_ingress_trust(record, payload)
            except Exception:
                trust = self._unknown_employee_ingress_trust()
            if (
                owner_p2p_requester is None
                and trust is not None
                and trust.zone is not TrustZone.MANAGED_AGENT_GROUP
            ):
                return finish("ignored", "authority_denied")
            if (
                owner_p2p_requester is None
                and trust is not None
                and trust.actor is ActorKind.EMPLOYEE
            ):
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
            if (
                expected_chat_index != metadata.chat_id
                or operation not in {"added", "deleted"}
            ):
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
        texts: list[str] = []
        for part in payload.normalized_parts:
            content = part.get("content") if isinstance(part, Mapping) else None
            value = content.get("text") if isinstance(content, Mapping) else None
            if isinstance(value, str):
                texts.append(value.strip())
        status_match = (
            re.fullmatch(r"/status(?:\s+(.+))?", texts[0], re.IGNORECASE)
            if len(texts) == 1
            else None
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
        data_control = self._parse_data_control(texts)
        if data_control is not None:
            return self._handle_data_control(
                acceptance_id=acceptance_id,
                command=data_control[0],
                record=record,
                payload=payload,
                requester_principal_id=(
                    owner_p2p_requester
                    or record.metadata.sender_principal_id
                ),
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
            return finish("terminal", "stop_coordinates_invalid")
        remote_chat_id, remote_message_id, remote_root_id = coordinates
        outcome = dispatch.request_cancel(
            agent_id=metadata.agent_id,
            chat_id=remote_chat_id,
            requester_principal_id=(
                owner_p2p_requester or metadata.sender_principal_id
            ),
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
        self._drain_employee_outbox_once()
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
        first = (
            payload.normalized_parts[0]
            if len(payload.normalized_parts) == 1
            else None
        )
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
        self._drain_employee_outbox_once()
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
        employee_runtime = (
            getattr(dispatch, "employee_runtime", None)
            if dispatch is not None
            else None
        )
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
            task_view = (
                f"当前会话有 {active_count} 个活动任务"
                f"（其中 {stopping_count} 个停止中）"
            )
        else:
            task_view = f"当前会话有 {active_count} 个活动任务"
        mailbox_depth = getattr(actor, "mailbox_depth", 0)
        if type(mailbox_depth) is not int or mailbox_depth < 0:
            raise RuntimeError("employee mailbox depth is invalid")
        return (
            f"员工状态：{labels[status]}\n"
            f"任务：{task_view}\n"
            f"队列：{mailbox_depth}"
        )

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
        if not isinstance(text, str):
            return False
        if (
            targeted_group_task is not None
            and targeted_group_task.state is TargetedTaskState.TARGETED_VALID
        ):
            return False
        if (
            targeted_group_task is not None
            and targeted_group_task.state is TargetedTaskState.INDETERMINATE
        ):
            return False
        if (
            targeted_group_task is not None
            and targeted_group_task.state is TargetedTaskState.TARGETED_INVALID
        ):
            lifecycle = self._outbox_lifecycle
            coordinates = _bound_remote_coordinates(metadata, first)
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
            self._drain_employee_outbox_once()
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
                    getattr(settings, "autonomous_employee_storage_base", None)
                    or default_employee_storage_base()
                )
            )
            if self._data is None:
                raise RuntimeError("canonical employee data is unavailable")

            self._context_acl = parse_requester_acl(settings)

            def registry_provider() -> ProjectedAgentRegistry:
                assert self._service is not None
                return ProjectedAgentRegistry(
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
        *,
        repaired_intents: set[str],
        base_summary: EmployeeRecoverySummary,
    ) -> EmployeeRecoverySummary:
        def summarize_durable_outcomes() -> EmployeeRecoverySummary:
            recovered = 0
            for intent_id in repaired_intents:
                try:
                    state = self._require_service().get_state(intent_id)
                except Exception:
                    continue
                if state is not None and state.phase is HirePhase.ACTIVE:
                    recovered += 1
            return EmployeeRecoverySummary(
                eligible=base_summary.eligible,
                recovered=recovered,
                skipped=base_summary.skipped,
                failed=base_summary.eligible - recovered,
            )

        failed_intents: list[str] = []
        if pending_intents:
            results = await asyncio.gather(
                *(
                    self._configure_intent(
                        intent_id,
                        force_slash_refresh=True,
                        allow_action_required_refresh=intent_id in repaired_intents,
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
                    self._retry_recovery_intent(
                        intent_id,
                        allow_action_required_refresh=intent_id in repaired_intents,
                    )
                    for intent_id in failed_intents
                ),
                return_exceptions=True,
            )
            failures = sum(
                isinstance(result, BaseException) or result is False
                for result in retry_results
            )
            if any(
                isinstance(result, BaseException)
                for result in retry_results
            ):
                logger.error(
                    "employee recovery could not be durably isolated for %d intent(s)",
                    failures,
                )
                self._execution_blockers = ("hire_recovery",)
                return summarize_durable_outcomes()
            if failures:
                logger.error(
                    "employee recovery isolated %d exhausted intent(s) as action_required",
                    failures,
                )
        self._start_monitor_in_loop()
        await self._retry_terminal_notifications()
        if not base_summary.failed:
            self._require_service().mark_runtime_recovered()
        if not self._execution_blockers:
            self._start_dispatch_worker()
        return summarize_durable_outcomes()


    async def _retry_recovery_intent(
        self,
        intent_id: str,
        *,
        allow_action_required_refresh: bool = False,
    ) -> bool:
        for delay in _RECOVERY_RETRY_DELAYS:
            if self._closing:
                raise RuntimeError("employee runtime is closing")
            await asyncio.sleep(delay)
            try:
                await self._configure_intent(
                    intent_id,
                    force_slash_refresh=True,
                    allow_action_required_refresh=allow_action_required_refresh,
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
                        elif state.phase is HirePhase.READY_PENDING_VERIFICATION:
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

    def _resume_pending_activation(self, state: DurableHireState) -> None:
        """Replace a legacy pending activation with a fresh automatic run."""
        service = self._require_service()
        current = service.get_state(state.intent_id)
        if (
            current is None
            or current.phase is not HirePhase.READY_PENDING_VERIFICATION
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
        allow_action_required_refresh: bool = False,
    ) -> None:
        service = self._require_service()
        state = service.get_state(intent_id)
        if state is None:
            return
        if state.phase in {
            HirePhase.PROVISIONING_APP,
            HirePhase.STORING_CREDENTIAL,
        }:
            state = await service.run_provisioning(intent_id)
        if state.phase not in {HirePhase.CONFIGURING, HirePhase.VALIDATING}:
            return
        for launch_attempt in range(2):
            state = service.get_state(intent_id) or state
            generation = self._target_channel_generation(state)
            selected_slash_effect_id = service.select_slash_reconcile_effect(
                state.intent_id,
                generation=generation,
                force_refresh=force_slash_refresh,
                allow_action_required_refresh=allow_action_required_refresh,
            )
            current_effect_ids = {
                selected_slash_effect_id,
                f"channel-start:{generation}",
            }
            effect_types = dict(state.effect_types)
            for effect_id, effect_state in state.effects:
                if (
                    effect_state
                    not in {HireEffectState.PREPARED, HireEffectState.EXECUTING}
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
                allow_action_required_refresh=allow_action_required_refresh,
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
                activated_at=time.time(),
            )
            if not self._refresh_context_bindings(service.projection_state):
                self._context_blockers = ("context_binding_sync",)
            return

    async def _reconcile_slash(
        self,
        state: DurableHireState,
        *,
        generation: int,
        force_refresh: bool,
        allow_action_required_refresh: bool,
    ) -> VerifiedSlashState:
        service = self._require_service()
        effect_id = service.select_slash_reconcile_effect(
            state.intent_id,
            generation=generation,
            force_refresh=force_refresh,
            allow_action_required_refresh=allow_action_required_refresh,
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
            or sender_id != record.metadata.sender_principal_id
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
                sender_id=record.metadata.sender_principal_id,
                sender_id_type=str(part.get("sender_id_type", "")),
                sender_type=str(part.get("sender_type", "")),
                sender_tenant_key=str(part.get("sender_tenant_key", "")),
                text=text,
                timestamp=time.time(),
            ),
        )
        return True





    def _require_service(self) -> ProductionEmployeeHireService:
        if self._service is None:
            raise RuntimeError("employee hire service unavailable")
        return self._service


__all__ = [
    "EmployeeDepartmentRuntime",
    "ReadyEmployeeIngressTarget",
    "RuntimeReadiness",
]
