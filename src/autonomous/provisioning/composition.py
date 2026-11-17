"""Single-owner production composition for durable visible employee hiring."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import lark_oapi as lark

from src.utils.path import canonicalize_user_home_path

from ...trust.registry import ManagedGroupRegistry
from ..acceptance.main_bot_warning_outbox import (
    MainBotWarningConflictError,
    MainBotWarningOutbox,
    MainBotWarningOutboxError,
    MainBotWarningTransport,
    main_bot_warning_idempotency_key,
)
from ..context.group_ledger import GroupContextLedger
from ..context.models import (
    ContextUnavailableError,
    ThreadContextConfig,
)
from ..context.service import EmployeeContextService
from ..context.source import EmployeeMessageSourceFactory
from ..data.composition import (
    EmployeeDataComposition,
    LegacyEmployeeDataUnsupportedError,
    build_employee_data_composition,
)
from ..data.keyring import EmployeeDataKeyring
from ..data.query import EmployeeDataSubject
from ..gateway.coordinator import (
    EmployeeDispatchCoordinator,
)
from ..gateway.env_scope import (
    EmployeeEnvironmentAuthority,
    EmployeeProcessEnvironmentMaterial,
)
from ..gateway.models import EmployeeDispatchReportingDeferredError
from ..ingress.attachments import AttachmentStagingService
from ..ingress.models import (
    EmployeeIngressMetadata,
    EmployeeIngressPayload,  # noqa: F401
)
from ..ingress.projection import IngressProjectionState
from ..ingress.router import (
    DurableEmployeeIngressRouter,
    RouterQueueLimits,
)
from ..ingress.service import (
    EmployeeIngressService,
)
from ..journal.anchor import FileAnchor
from ..journal.projections import ProjectionState
from ..journal.writer import JournalWriter
from ..membership import (
    EmployeeMembershipService,
    LarkMembershipAPI,
)
from ..outbox.delivery import (
    EmployeeOutboxDeliveryCoordinator,
)
from ..outbox.lifecycle import EmployeeOutboxLifecycle
from ..outbox.projection import OutboxProjectionState
from ..outbox.service import EmployeeOutboxService
from ..supervisor.employee_channels import (
    ChannelProcessState,
    EmployeeChannelSupervisor,
)
from ..workforce.credential_vault import CredentialVault
from ..workforce.identity import default_employee_storage_base
from ..workforce.registry import ProjectedAgentRegistry
from .external_mutation_gate import (
    EmployeeExternalMutationGate,
)
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
    HirePhase,
)
from .lark_app import LarkAppRegistrar
from .local_bootstrap import resolve_employee_runtime_material
from .slash_commands import VerifiedSlashState

logger = logging.getLogger(__name__)

_RECOVERY_RETRY_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6)
_CHANNEL_RECONNECT_GRACE_SECONDS = 180.0
_EMPLOYEE_HANDOFF_FINALIZATION_RESERVE_SECONDS = 0.25
_EMPLOYEE_HANDOFF_FINALIZATION_CLOSE_SECONDS = 5.0
_EMPLOYEE_HANDOFF_FINALIZATION_MAX_PENDING = 64
_EMPLOYEE_HANDOFF_FINALIZATION_WORKERS = 4
_EMPLOYEE_OUTBOX_CLOSE_DRAIN_SECONDS = 5.0
_EMPLOYEE_OUTBOX_CLOSE_MAX_BATCHES = 10_000
_EMPLOYEE_OUTBOX_DELIVERY_BATCH = 16
_MAIN_BOT_WARNING_REPORTING_DRAIN_SECONDS = 0.25
_EMPLOYEE_EXECUTION_TERMINAL_REASONS = frozenset({"completed", "failed", "canceled", "timeout", "action_required"})


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


def _outbox_coordinate_matches_ingress_index(
    indexed_value: object,
    raw_value: object,
    prefix: str,
) -> bool:
    """Bind an outbound raw coordinate to its durable one-way Inbox index."""

    return bool(
        isinstance(indexed_value, str)
        and isinstance(raw_value, str)
        and raw_value
        and indexed_value == prefix + hashlib.sha256(raw_value.encode()).hexdigest()
    )


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

    def close(self) -> None: ...


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
    """Treat an unavailable or unverified team chat as degraded."""

    def __init__(
        self,
        manager: object | None,
        registry: object | None = None,
    ) -> None:
        self._manager = manager
        self._registry = registry

    def is_degraded(self, _agent_id: str, team_id: str) -> bool:
        if self._manager is not None:
            try:
                self._manager.resolve_employee_engine(chat_id=team_id)
            except Exception:
                return True
            return False
        active_record = getattr(self._registry, "active_record", None)
        if callable(active_record):
            try:
                return active_record(team_id) is None
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


class EmployeeMessageHandoffUnknownError(RuntimeError):
    """The Main Bot could not durably prove either handoff or nonexecution."""


class EmployeeTargetResolutionUnknownError(RuntimeError):
    """The Main Bot could not prove whether a READY employee target exists."""


class MainBotWarningPreparationError(RuntimeError):
    """A terminal main-Bot warning could not cross its durable anchor."""


class EmployeeRuntimeCloseWaitTimeout(TimeoutError):
    """A caller stopped waiting for the runtime's active close attempt."""


class _EmployeeHandoffProjection(Enum):
    OWNED = "owned"
    DURABLY_DENIED = "durably_denied"
    INVALID_UNKNOWN = "invalid_unknown"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class ReadyEmployeeIngressTarget:
    """Frozen READY authority used to correlate one cross-App handoff."""

    tenant_key: str
    chat_id: str
    agent_id: str
    bot_principal_id: str
    app_id: str
    bot_open_id: str
    channel_generation: int
    connection_id: str


@dataclass(frozen=True, slots=True)
class _ValidatedTargetedOwnership:
    """Authenticated Inbox payload bound to one frozen targeted authority."""

    metadata: EmployeeIngressMetadata
    remote_chat_id: str
    remote_message_id: str
    remote_root_id: str


from .runtime_ingress import EmployeeRuntimeIngressMixin  # noqa: E402
from .runtime_recovery import EmployeeRuntimeRecoveryMixin  # noqa: E402


class EmployeeDepartmentRuntime(
    EmployeeRuntimeIngressMixin,
    EmployeeRuntimeRecoveryMixin,
):
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
        self._external_mutation_gate = EmployeeExternalMutationGate()
        self._dispatch_thread: threading.Thread | None = None
        self._dispatch_stop = threading.Event()
        self._dispatch_wakeup = threading.Event()
        self._dispatch_recovery_pending = False
        self._reporting_thread: threading.Thread | None = None
        self._reporting_stop = threading.Event()
        self._reporting_wakeup = threading.Event()
        self._reporting_lifecycle_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._reporting_repair_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._reporting_repair_failures = 0
        self._reporting_repair_not_before = 0.0
        self._warning_reporting_failures = 0
        self._warning_reporting_not_before = 0.0
        self._warning_reporting_next_log_at = 0.0
        self._employee_admission_locks_guard = (
            threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        self._employee_admission_locks: dict[
            str,
            tuple[threading.RLock, int],
        ] = {}
        self._employee_admission_executor_lock = (
            threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        self._employee_admission_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._employee_admission_futures: set[concurrent.futures.Future[bool]] = set()
        self._employee_admission_closed = False
        self._employee_admission_max_pending = 64
        self._employee_handoff_finalization_lock = (
            threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        self._employee_handoff_finalization_executor: (
            concurrent.futures.ThreadPoolExecutor | None
        ) = None
        self._employee_handoff_finalization_futures: dict[
            tuple[str, ...],
            concurrent.futures.Future[Any],
        ] = {}
        self._employee_handoff_finalization_closed = False
        self._employee_outbox_delivery_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._employee_outbox_close_drain_lock = (
            threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        self._employee_outbox_close_drain_executor: (
            concurrent.futures.ThreadPoolExecutor | None
        ) = None
        self._employee_outbox_close_drain_future: (
            concurrent.futures.Future[None] | None
        ) = None
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
        self._pending_intent_resubmits: set[str] = set()
        self._future_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._recovery_lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._recovery_future: concurrent.futures.Future[EmployeeRecoverySummary] | None = None
        self._recovery_attempts = 0
        self._closing = False
        self._close_incomplete = False
        self._close_attempt_lock = (
            threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        self._close_attempt: concurrent.futures.Future[None] | None = None
        self._close_complete = False
        self._employee_handoff_event = threading.Event()
        self._employee_handoff_revision_lock = (
            threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        )
        self._employee_handoff_revision = 0
        self._notification_status: Callable[[DurableHireState, str], object] | None = None
        self._owned_main_bot_send_audit: Any = None
        self._main_bot_warning_outbox: MainBotWarningOutbox | None = None
        self._main_bot_warning_transport: MainBotWarningTransport | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._notification_async_lock: asyncio.Lock | None = None
        self._core_recovered = False
        self._membership_audit_ready = not self._runtime_enabled
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
            credential_root = canonicalize_user_home_path(settings.autonomous_credential_dir)
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
        EmployeeFireService.restore_external_mutation_fences(
            writer,
            runtime._external_mutation_gate,
        )
        runtime._team_runtime = team_runtime
        runtime._environment_provider = employee_environment_provider
        runtime._owned_main_bot_send_audit = None
        del main_bot_send_audit, team_notification
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
                workspace_projector=(runtime._data.workspace_projector if runtime._data is not None else None),
                admin_principal_ids_provider=lambda: frozenset(getattr(settings, "admin_user_ids", ()) or ()),
                external_mutation_gate=runtime._external_mutation_gate,
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
            if recover_immediately:
                runtime.recover()
            return runtime
        except Exception as exc:
            logger.error(
                "employee department runtime composition unavailable: %s",
                type(exc).__name__,
            )
            try:
                runtime.close()
            except Exception:
                logger.error(
                    "employee department runtime cleanup incomplete",
                    exc_info=True,
                )
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
    def data_composition(self) -> EmployeeDataComposition | None:
        return self._data

    @property
    def dispatch_coordinator(self) -> EmployeeDispatchCoordinator | None:
        return self._dispatch

    @property
    def main_bot_outbound_audit(self) -> Any:
        return self._owned_main_bot_send_audit

    def bind_main_bot_warning_transport(
        self,
        transport: MainBotWarningTransport,
    ) -> None:
        """Late-bind the main-Bot transport after handlers are constructed."""

        if not callable(getattr(transport, "send_warning", None)):
            raise TypeError("main Bot warning transport must provide send_warning")
        with self._reporting_lifecycle_lock:
            if self._closing:
                raise RuntimeError("employee runtime is closing")
            self._main_bot_warning_transport = transport
        # Idempotently creates the worker after recover() ran without a
        # transport, or wakes the existing worker so a transport swap takes
        # effect without waiting for its polling backoff.
        self._start_reporting_worker()

    def queue_main_bot_warning(
        self,
        *,
        tenant_key: str,
        chat_id: str,
        message_id: str,
        text: str,
    ) -> bool:
        """Anchor one origin-bound warning and wake its recovery worker.

        ``True`` means this origin is durably fenced by either the newly
        prepared warning or an earlier first-writer-wins warning.  Transport is
        deliberately never called on the WS task.
        """

        if self._closing:
            raise MainBotWarningPreparationError("main Bot warning admission is closing")
        outbox = self._main_bot_warning_outbox
        if outbox is None:
            raise MainBotWarningPreparationError("main Bot warning Outbox is unavailable")
        try:
            outbox.enqueue(
                tenant_key=tenant_key,
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                idempotency_key=main_bot_warning_idempotency_key(
                    tenant_key,
                    chat_id,
                    message_id,
                ),
            )
        except MainBotWarningConflictError:
            # The aggregate is origin-bound.  Preserve its first durable copy
            # rather than creating a contradictory second user-visible reply.
            logger.error("main Bot warning origin already has different durable content")
        except (MainBotWarningOutboxError, OSError, TimeoutError) as exc:
            raise MainBotWarningPreparationError("main Bot warning could not be durably prepared") from exc
        self._notify_employee_reporting()
        return True

    def readiness(self) -> RuntimeReadiness:
        return self.hire_readiness()

    def hire_readiness(self) -> RuntimeReadiness:
        if self._service is None:
            return RuntimeReadiness(False, self._blockers or ("not_composed",))
        service_readiness: HireReadiness = self._service.readiness()
        return RuntimeReadiness(service_readiness.ready, service_readiness.blockers)

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
                if state.phase is HirePhase.ACTIVE and (agent_id is None or state.agent_id == agent_id)
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
            if head.sequence != projection.cursor_sequence or head.logical_hash != projection.cursor_hash:
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
        if chat_id is not None and (not isinstance(chat_id, str) or not chat_id.strip()):
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
                or getattr(status, "ready_metadata", {}).get("connection_id") != state.channel_connection_id
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
            groups_to_probe = (chat_id,) if chat_id is not None else tuple(sorted(employee.member_groups))
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
        if self._main_bot_warning_outbox is not None:
            self._main_bot_warning_outbox.rebuild_projection()
            if self._main_bot_warning_transport is not None:
                self._start_reporting_worker()
        if self._service is None:
            return EmployeeRecoverySummary()
        self._core_recovered = False
        self._membership_audit_ready = not self._runtime_enabled or self._membership is None
        self._service.recover()
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
            if self._runtime_enabled and not self._execution_blockers:
                try:
                    membership_audit = self._membership.reconcile_projected_memberships()
                except Exception as exc:
                    logger.error(
                        "employee membership startup audit failed closed: %s",
                        type(exc).__name__,
                    )
                    self._execution_blockers = ("membership_audit",)
                else:
                    self._membership_audit_ready = True
                    if membership_audit.removed or membership_audit.degraded:
                        logger.warning(
                            "employee membership startup audit reconciled removed=%d degraded=%d",
                            membership_audit.removed,
                            membership_audit.degraded,
                        )
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
        outbox_projection_recovered = False
        if self._runtime_enabled and self._outbox is not None:
            try:
                self._outbox.rebuild_projection()
            except Exception as exc:
                logger.error(
                    "employee reporting recovery failed closed: %s",
                    type(exc).__name__,
                )
            else:
                outbox_projection_recovered = True
                # Anchored replies are independent from actor/team execution
                # recovery.  Start their delivery lane as soon as its own
                # projection and Channel authority are available.
                self._start_reporting_worker()
        employee_runtime = self._dispatch.employee_runtime if self._dispatch is not None else None
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
                self._router.rebuild_projection()
                if not outbox_projection_recovered:
                    self._outbox.rebuild_projection()
                self._dispatch_recovery_pending = True
                self._repair_employee_dispatch_reporting()
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
            else:
                # A delivery-only Channel launch is ordinary recoverable I/O.
                # Let its exception reach ``recover()`` so the existing bounded
                # single-flight retry can make a second attempt; never cache a
                # successful summary while retirement obligations are stranded.
                self._recover_retirement_delivery_channels()
        if not self._refresh_context_bindings(self._service.projection_state):
            self._context_blockers = ("context_binding_sync",)
        self._core_recovered = not self._execution_blockers and not self._context_blockers
        if not self._runtime_enabled:
            self._service.mark_runtime_recovered()
            return EmployeeRecoverySummary()
        pending_intents: list[str] = []
        recovery_states = tuple(self._service.list_states())
        for state in recovery_states:
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
        recover_notifications = self._notification_status is not None and any(
            self._terminal_notification_status(state) for state in recovery_states
        )
        if pending_intents or recover_notifications:
            recovery_future = self._submit_coroutine(
                self._recover_runtime(
                    list(dict.fromkeys(pending_intents)),
                ),
                label="recovery",
            )
            return recovery_future.result()
        else:
            if self._membership_audit_ready:
                self._service.mark_runtime_recovered()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._start_monitor_in_loop)
            self._start_reporting_worker()
            if execution_recovered:
                self._start_dispatch_worker()
            return EmployeeRecoverySummary()

    def close(self, *, timeout_seconds: float | None = None) -> None:
        """Join one close attempt without mistaking in-progress for success.

        ``timeout_seconds`` bounds only a caller waiting for an attempt already
        owned by another thread.  It never cancels that owner or shortens the
        close stages' own safety deadlines.
        """

        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds < 0
        ):
            raise ValueError("employee runtime close wait timeout is invalid")
        with self._close_attempt_lock:
            if self._close_complete:
                return
            attempt = self._close_attempt
            owner = attempt is None
            if attempt is None:
                attempt = concurrent.futures.Future()
                self._close_attempt = attempt
        if not owner:
            try:
                attempt.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError as exc:
                if attempt.done():
                    # ``Future.result`` uses the same TimeoutError type for a
                    # caller wait expiry and for an owner's terminal timeout.
                    # Re-observe a now-terminal attempt to preserve its exact
                    # result instead of misreporting it as a waiter deadline.
                    attempt.result()
                raise EmployeeRuntimeCloseWaitTimeout(
                    "employee runtime close attempt is still in progress"
                ) from exc
            return
        try:
            self._close_once()
        except BaseException as exc:
            with self._close_attempt_lock:
                if self._close_attempt is attempt:
                    self._close_attempt = None
            attempt.set_exception(exc)
            raise
        else:
            with self._close_attempt_lock:
                self._close_complete = True
                if self._close_attempt is attempt:
                    self._close_attempt = None
            attempt.set_result(None)

    def _close_once(self) -> None:
        """Drain owned work or raise with the precise incomplete stage."""

        with self._reporting_lifecycle_lock:
            with self._future_lock:
                self._closing = True
                self._close_incomplete = False
                self._pending_intent_resubmits.clear()
        self._notify_employee_handoff_progress()
        errors: list[str] = []

        def cleanup(label: str, action: Callable[[], Any]) -> bool:
            try:
                action()
                return True
            except Exception as exc:
                errors.append(f"{label}:{type(exc).__name__}")
                return False

        try:
            handoff_finalization_safe = (
                self._shutdown_employee_handoff_finalization_executor(
                    timeout=_EMPLOYEE_HANDOFF_FINALIZATION_CLOSE_SECONDS,
                )
            )
            if not handoff_finalization_safe:
                errors.append("handoff_finalization:TimeoutError")
        except Exception as exc:
            errors.append(f"handoff_finalization:{type(exc).__name__}")
            handoff_finalization_safe = False
        if not handoff_finalization_safe:
            # A running finalizer may own the Ingress/Router/Journal locks while
            # completing an irreversible append.  Do not wait on those locks or
            # close their dependencies underneath it; a later close call can
            # resume teardown after the tracked future reaches terminal.
            errors.append("dependent_resources_held")
            self._close_incomplete = True
            logger.error("employee runtime close errors: %s", ",".join(errors))
            raise RuntimeError(
                "employee runtime close incomplete: " + ",".join(errors)
            )

        if self._ingress is not None:
            cleanup("ingress_admission", self._ingress.stop_admission)
        if self._service is not None:
            cleanup("hire_admission", self._service.stop_admission)
        if self._context_service is not None:
            cleanup("context_admission", self._context_service.stop_admission)
        self._dispatch_stop.set()
        self._dispatch_wakeup.set()
        dispatch_safe = True
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)
            dispatch_safe = not self._dispatch_thread.is_alive()
            if not dispatch_safe:
                errors.append("dispatch_worker:TimeoutError")
        try:
            admission_safe = self._shutdown_employee_admission_executor(
                timeout=5.0,
            )
            if not admission_safe:
                errors.append("employee_admission_executor:TimeoutError")
        except Exception as exc:
            errors.append(f"employee_admission_executor:{type(exc).__name__}")
            admission_safe = False
        dispatch_close = getattr(self._dispatch, "close", None)
        if dispatch_safe and callable(dispatch_close):
            dispatch_safe = cleanup("employee_actors", dispatch_close)
        reporting_safe = dispatch_safe and admission_safe
        if reporting_safe:
            with self._reporting_lifecycle_lock:
                self._closing = True
                self._reporting_stop.set()
                self._reporting_wakeup.set()
                reporting_thread = self._reporting_thread
            if reporting_thread is not None:
                reporting_thread.join(timeout=5.0)
                reporting_safe = not reporting_thread.is_alive()
                if not reporting_safe:
                    errors.append("reporting_worker:TimeoutError")
        with self._future_lock:
            futures = tuple(self._futures)
        activities_safe = True
        if futures:
            _done, pending = concurrent.futures.wait(futures, timeout=5.0)
            for future in pending:
                future.cancel()
            activities_safe = not pending
            if pending:
                errors.append("employee_activities:TimeoutError")
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
        outbox_safe = reporting_safe
        if reporting_safe and self._outbox is not None:
            outbox_safe = cleanup(
                "outbox_admission",
                self._outbox.stop_admission,
            )
        outbox_close_deadline = time.monotonic() + _EMPLOYEE_OUTBOX_CLOSE_DRAIN_SECONDS
        if outbox_safe and self._outbox is not None and self._outbox_delivery is not None:
            outbox_safe = cleanup(
                "outbox_drain",
                lambda: self._await_employee_outbox_close_drain(
                    deadline=outbox_close_deadline,
                ),
            )
        warning_outbox_safe = reporting_safe
        if warning_outbox_safe and self._main_bot_warning_outbox is not None:
            warning_outbox_safe = cleanup(
                "main_bot_warning_drain",
                lambda: self._drain_main_bot_warning_outbox_until_idle(
                    deadline=outbox_close_deadline,
                ),
            )
        resources_safe = (
            dispatch_safe
            and admission_safe
            and reporting_safe
            and activities_safe
            and context_safe
            and outbox_safe
            and warning_outbox_safe
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
        if resources_safe and self._main_bot_warning_outbox is not None:
            resources_safe = cleanup(
                "main_bot_warning_outbox",
                self._main_bot_warning_outbox.close,
            )
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
        if not resources_safe:
            errors.append("dependent_resources_held")
        if resources_safe and self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if resources_safe and self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
            if self._loop_thread.is_alive():
                resources_safe = False
                errors.append("activity_loop_thread:TimeoutError")
        if errors:
            logger.error("employee runtime close errors: %s", ",".join(errors))
        if not resources_safe:
            self._close_incomplete = True
        if errors:
            raise RuntimeError("employee runtime close incomplete: " + ",".join(errors))

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
                    getattr(settings, "autonomous_employee_storage_base", None) or default_employee_storage_base()
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
            self._main_bot_warning_outbox = MainBotWarningOutbox.from_keyring(
                writer=self._writer,
                keyring=self._data_keyring,
                blob_root=getattr(
                    settings,
                    "autonomous_main_bot_warning_blob_dir",
                    "~/.ghostap/autonomy/main-bot-warning-blobs",
                ),
                main_app_id=getattr(settings, "app_id", ""),
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
            if self._main_bot_warning_outbox is not None:
                try:
                    self._main_bot_warning_outbox.close()
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
            self._main_bot_warning_outbox = None
            self._data = None
            self._execution_blockers = (
                "legacy_employee_data_unsupported"
                if isinstance(exc, LegacyEmployeeDataUnsupportedError)
                else "employee_ingress",
            )

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
            raise LegacyEmployeeDataUnsupportedError("legacy employee data requires an offline migration")
        data.rebuild_all()

    def _resolve_team_owner(self, chat_id: str) -> str:
        if self._team_runtime is not None:
            try:
                getter = getattr(self._team_runtime, "get_activated_engine", None)
                engine = getter(chat_id) if callable(getter) else None
                channel = getattr(engine, "channel", None)
                return str(getattr(channel, "owner_id", "") or "")
            except Exception:
                return ""
        active_record = getattr(self._managed_group_registry, "active_record", None)
        if callable(active_record):
            try:
                record = active_record(chat_id)
                return str(getattr(record, "owner_id", "") or "")
            except Exception:
                return ""
        return ""

    def _is_team_active(self, chat_id: str) -> bool:
        if self._team_runtime is not None:
            try:
                return bool(self._team_runtime.get_activated_engine(chat_id))
            except Exception:
                return False
        active_record = getattr(self._managed_group_registry, "active_record", None)
        if callable(active_record):
            try:
                return active_record(chat_id) is not None
            except Exception:
                return False
        return False

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

        try:
            remote = LarkMembershipAPI(
                manager_client_factory(),
                employee_client_provider=employee_client_provider,
            )
            self._membership = EmployeeMembershipService(
                writer=self._writer,
                hire_service=self._service,
                remote=remote,
                admin_principal_ids=frozenset(getattr(settings, "admin_user_ids", ()) or ()),
                team_owner_resolver=self._resolve_team_owner,
                team_active_resolver=self._is_team_active,
                external_mutation_gate=self._external_mutation_gate,
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
            return future.result(timeout=float(getattr(settings, "autonomous_context_fetch_timeout_seconds", 30.0)))

        legacy_base = str(
            canonicalize_user_home_path(
                getattr(settings, "autonomous_employee_storage_base", None) or default_employee_storage_base()
            )
        )
        authority = JournalFireAuthority(
            writer=self._writer,
            hire_service=self._service,
            ingress_service=self._ingress,
            admin_principal_ids=frozenset(getattr(settings, "admin_user_ids", ()) or ()),
        )
        self._fire = EmployeeFireService(
            writer=self._writer,
            authority=authority,
            effects={
                "execution_quiesce": ExecutionQuiesceEffect(
                    self._dispatch,
                    grace_seconds=float(getattr(settings, "autonomous_fire_grace_seconds", 5.0)),
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
                "archive_move": AtomicEmployeeArchive(Path(legacy_base).expanduser() / "agents"),
            },
            external_mutation_gate=self._external_mutation_gate,
            external_mutation_wait_seconds=float(getattr(settings, "autonomous_fire_grace_seconds", 5.0)),
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
        if not callable(self._environment_provider):
            self._execution_blockers = ("employee_environment",)
            return
        try:
            legacy_base = str(
                canonicalize_user_home_path(
                    getattr(settings, "autonomous_employee_storage_base", None) or default_employee_storage_base()
                )
            )

            def registry_provider() -> ProjectedAgentRegistry:
                assert self._service is not None
                return ProjectedAgentRegistry(
                    self._service.synchronize_projection(),
                    storage_base_path=legacy_base,
                )

            health = membership_health or _TeamMembershipHealth(
                self._team_runtime,
                self._managed_group_registry,
            )
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
                requester_principal_resolver=(self._resolve_employee_requester_principal),
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
                admin_principal_ids=frozenset(getattr(settings, "admin_user_ids", ()) or ()),
                team_owner_resolver=self._resolve_team_owner,
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
        self._start_reporting_worker()
        if self._dispatch is None or self._router is None or self._ingress is None:
            return
        if self._dispatch_thread is not None and self._dispatch_thread.is_alive():
            return
        self._dispatch_stop.clear()
        self._dispatch_wakeup.set()

        def run() -> None:
            delay = 0.05
            # Keep the established stop-port contract (`wait`) while the
            # separate wakeup event short-circuits idle/backoff latency.
            while not self._dispatch_stop.wait(0.0):
                self._dispatch_wakeup.wait(delay)
                self._dispatch_wakeup.clear()
                stop_is_set = getattr(self._dispatch_stop, "is_set", None)
                if callable(stop_is_set) and stop_is_set():
                    break
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
                finally:
                    self._notify_employee_handoff_progress()

        self._dispatch_thread = threading.Thread(
            target=run,
            name="employee-durable-dispatch",
            daemon=True,
        )
        self._dispatch_thread.start()

    def _start_reporting_worker(self) -> None:
        """Run durable Employee reporting independently from Agent execution."""

        employee_reporting_ready = self._outbox is not None and self._outbox_delivery is not None
        warning_reporting_ready = (
            self._main_bot_warning_outbox is not None and self._main_bot_warning_transport is not None
        )
        if not employee_reporting_ready and not warning_reporting_ready:
            return

        def run() -> None:
            delay = 0.05
            while not self._reporting_stop.is_set():
                self._reporting_wakeup.wait(delay)
                self._reporting_wakeup.clear()
                if self._reporting_stop.is_set():
                    break
                try:
                    worked = self._drain_employee_reporting_once()
                    delay = 0.05 if worked else min(delay * 2.0, 1.0)
                except Exception as exc:
                    # Component-level faults are isolated by the drain method.
                    # Keep this outer fence for programming/projection failures
                    # so one bad tick cannot permanently kill reporting.
                    logger.error(
                        "employee reporting worker failed closed: %s",
                        type(exc).__name__,
                    )
                    delay = min(max(delay, 0.05) * 2.0, 5.0)
                finally:
                    self._notify_employee_handoff_progress()

        with self._reporting_lifecycle_lock:
            if self._closing:
                return
            if self._reporting_thread is not None and self._reporting_thread.is_alive():
                self._reporting_wakeup.set()
                return
            self._reporting_stop.clear()
            self._reporting_wakeup.set()
            reporting_thread = threading.Thread(
                target=run,
                name="employee-durable-reporting",
                daemon=True,
            )
            self._reporting_thread = reporting_thread
            reporting_thread.start()

    def _notify_employee_reporting(self) -> None:
        """Wake the reporting lane after a durable state transition."""

        self._reporting_wakeup.set()


    def _require_service(self) -> ProductionEmployeeHireService:
        if self._service is None:
            raise RuntimeError("employee hire service unavailable")
        return self._service


__all__ = [
    "EmployeeDepartmentRuntime",
    "EmployeeDispatchReportingDeferredError",
    "EmployeeMessageHandoffUnknownError",
    "EmployeeRuntimeCloseWaitTimeout",
    "EmployeeTargetResolutionUnknownError",
    "MainBotWarningPreparationError",
    "ReadyEmployeeIngressTarget",
    "RuntimeReadiness",
]
