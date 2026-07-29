from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from src.autonomous.acceptance.main_bot_audit import MainBotSendAuditLog
from src.autonomous.context import (
    AuthorizedContextRequest,
    ContextLayer,
    ContextMessage,
    MessagePage,
    ResolvedThread,
)
from src.autonomous.data.models import DataKind, ExecutionAttemptContext
from src.autonomous.data.ports import PublishEmployeeDocumentCommand
from src.autonomous.gateway.coordinator import EmployeeDispatchCoordinator
from src.autonomous.gateway.env_scope import EmployeeProcessEnvironmentMaterial
from src.autonomous.ingress.models import EmployeeIngressMetadata, EmployeeIngressPayload
from src.autonomous.journal.anchor import MemoryAnchor
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.provisioning.composition import (
    EmployeeDepartmentRuntime,
    RuntimeReadiness,
    _RuntimeTeamBackend,
)
from src.autonomous.provisioning.hire_port import EmployeeHireRequest
from src.autonomous.provisioning.hire_state import (
    DurableHireState,
    HireEffectState,
    HirePhase,
)
from src.autonomous.provisioning.lark_app import RegistrationResult
from src.autonomous.provisioning.slash_commands import VerifiedSlashState
from src.autonomous.supervisor.employee_channels import ChannelProcessState, ChannelSendReceipt
from src.autonomous.team.service import EmployeeTeamService
from src.autonomous.workforce.projection import commit_workforce_events


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _settings(
    tmp_path: Path,
    *,
    limit: int,
    context_configured: bool = False,
) -> SimpleNamespace:
    keyring = {
        "version": 1,
        "keys": {"k1": _b64(b"v" * 32)},
    }
    settings = SimpleNamespace(
        autonomous_visible_employee_limit=limit,
        autonomous_journal_dir=str(tmp_path / "journal"),
        autonomous_journal_hmac_key=SecretStr(_b64(b"j" * 32)),
        autonomous_anchor_provider="file",
        autonomous_anchor_path=str(tmp_path / "anchor.json"),
        autonomous_credential_dir=str(tmp_path / "vault"),
        autonomous_credential_keys=SecretStr(json.dumps(keyring)),
        autonomous_credential_active_key_id="k1",
        autonomous_data_keys=SecretStr(json.dumps(keyring)),
        autonomous_data_active_key_id="k1",
        autonomous_data_blob_dir=str(tmp_path / "data-blobs"),
        autonomous_employee_ingress_blob_dir=str(tmp_path / "ingress-blobs"),
        autonomous_employee_outbox_blob_dir=str(tmp_path / "outbox-blobs"),
        autonomous_employee_attachment_staging_dir=str(tmp_path / "attachments"),
        autonomous_state_dir=str(tmp_path / "state"),
        autonomous_slock_storage_base=str(tmp_path / "slock"),
        autonomous_worker_sandbox_verified=True,
        autonomous_employee_service_instance_id="ghostap-prod-a",
        autonomous_main_bot_audit_dir=str(tmp_path / "main-bot-audit"),
        autonomous_main_bot_audit_anchor_path=str(tmp_path / "main-bot-audit.anchor"),
    )
    if context_configured:
        settings.autonomous_employee_system_prompt_token_reserve = 4096
        settings.autonomous_employee_queue_per_employee_limit = 8
        settings.autonomous_employee_queue_per_team_limit = 32
        settings.autonomous_employee_queue_global_limit = 128
        settings.autonomous_history_timezone = "UTC"
        settings.autonomous_history_max_range_days = 31
        settings.autonomous_history_page_size = 50
        settings.autonomous_thread_context_max_messages = 200
        settings.autonomous_thread_context_max_chars = 400_000
        settings.autonomous_group_context_max_messages = 50
        settings.autonomous_context_max_tokens = 128_000
        settings.autonomous_thread_context_page_size = 50
        settings.autonomous_group_context_page_size = 20
        settings.autonomous_context_fetch_timeout_seconds = 30.0
        settings.autonomous_context_retry_base_seconds = 1.0
        settings.autonomous_context_retry_max_seconds = 30.0
        settings.autonomous_team_step_timeout_seconds = 600.0
        settings.autonomous_context_max_pages = 200
        settings.autonomous_manager_acl = "ou_admin"
        settings.admin_user_ids = frozenset({"ou_admin"})
        settings.allowed_chat_ids = frozenset()
        settings.app_id = "cli_manager"
    return settings


class _Registrar:
    async def register(self, request, *, on_link, on_status=None):
        del request, on_status
        on_link("https://open.feishu.cn/register/one-shot", 60)
        return RegistrationResult("cli_employee", "runtime-vault-only-secret")


class _DistinctRegistrar:
    async def register(self, request, *, on_link, on_status=None):
        del on_status
        on_link("https://open.feishu.cn/register/one-shot", 60)
        suffix = request.name.casefold()
        return RegistrationResult(
            f"cli_employee_{suffix}",
            f"runtime-vault-only-secret-{suffix}",
        )


class _RecordingExistingRegistrar:
    def __init__(self) -> None:
        self.requests = []

    async def register(self, request, *, on_link, on_status=None):
        del on_status
        self.requests.append(request)
        on_link("https://open.feishu.cn/register/existing", 60)
        return RegistrationResult(
            request.existing_app_id,
            "runtime-vault-only-secret-existing",
        )


class _Slash:
    async def reconcile(self):
        return VerifiedSlashState(
            spec_hash="slash_hash",
            observed_hash="slash_hash",
            observed=(),
        )


class _FailingSlash:
    async def reconcile(self):
        raise RuntimeError("tenant Slash state did not converge")


class _BlockingSlash:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    async def reconcile(self):
        self.calls += 1
        self.entered.set()
        await asyncio.to_thread(self.release.wait, 5)
        return await _Slash().reconcile()


class _FailOnceSlash:
    def __init__(self, *, failures: int = 1) -> None:
        self.calls = 0
        self.failures = failures

    async def reconcile(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("injected transient recovery failure")
        return await _Slash().reconcile()


class _Channels:
    def __init__(self) -> None:
        self.started: list[tuple[str, str, str, int]] = []
        self.callbacks: dict[str, object] = {}
        self.sent: list[tuple[str, int, str, object, object]] = []
        self.statuses: dict[str, object] = {}
        self.closed = False

    def start(self, agent_id, app_id, credential_ref, generation, on_event):
        self.started.append((agent_id, app_id, credential_ref, generation))
        self.callbacks[agent_id] = on_event
        status = SimpleNamespace(
            agent_id=agent_id,
            app_id=app_id,
            tenant_key="tenant-a",
            bot_principal_id="bot_" + agent_id.removeprefix("agt_"),
            state=ChannelProcessState.READY,
            generation=generation,
            identity={"app_id": app_id, "open_id": "ou_employee"},
            ready_metadata={"connection_id": "conn_runtime"},
        )
        self.statuses[agent_id] = status
        return status

    def status(self, agent_id):
        return self.statuses.get(agent_id)

    @contextmanager
    def employee_dispatch_guard(self):
        yield

    def crash(self, agent_id: str) -> None:
        status = self.statuses[agent_id]
        self.statuses[agent_id] = SimpleNamespace(**{**vars(status), "state": ChannelProcessState.CRASHED})

    def send(self, agent_id, *, generation, target, message, options=None):
        self.sent.append((agent_id, generation, target, message, options))
        status = self.statuses[agent_id]
        return ChannelSendReceipt(
            request_id="send_runtime",
            success=True,
            app_id=status.identity["app_id"],
            generation=generation,
            connection_id=status.ready_metadata["connection_id"],
            message_id="om_employee_reply",
        )

    def update_card(self, agent_id, *, generation, message_id, card):
        status = self.statuses[agent_id]
        self.sent.append((agent_id, generation, message_id, card, "update"))
        return ChannelSendReceipt(
            request_id="update_runtime",
            success=True,
            app_id=status.identity["app_id"],
            generation=generation,
            connection_id=status.ready_metadata["connection_id"],
            message_id=message_id,
        )

    def recover(self, desired):
        return {
            item.agent_id: self.start(
                item.agent_id,
                item.app_id,
                item.credential_ref,
                item.generation,
                item.on_event,
            )
            for item in desired
        }

    def close(self):
        self.closed = True


class _SlockManager:
    @contextmanager
    def employee_activation_guard(self, **_kwargs):
        raise RuntimeError("no activated Slock in composition-only test")
        yield

    def resolve_employee_engine(self, **_kwargs):
        raise RuntimeError("no activated Slock in composition-only test")


class _HealthyMembership:
    def is_degraded(self, _agent_id: str, _team_id: str) -> bool:
        return False


class _ContextSourceFactory:
    def __init__(
        self,
        *,
        probe_ready: bool = True,
        group_probe_ready: bool | dict[str, bool] = True,
    ) -> None:
        self.probe_ready = probe_ready
        self.group_probe_ready = group_probe_ready
        self.probed: list[tuple[str, str, str]] = []
        self.group_probed: list[tuple[str, str]] = []
        self.invalidated: list[str] = []
        self.reactivated: list[str] = []
        self.closed = False

    def probe(self, principal):
        self.probed.append((principal.agent_id, principal.app_id, principal.credential_ref))
        return self.probe_ready

    def probe_group_history(self, principal, chat_id):
        self.group_probed.append((principal.agent_id, chat_id))
        if isinstance(self.group_probe_ready, dict):
            return self.group_probe_ready.get(chat_id, False)
        return self.group_probe_ready

    def open(self, **_kwargs):
        raise AssertionError("readiness must not perform message API calls")

    def invalidate_employee(self, agent_id):
        self.invalidated.append(agent_id)

    def reactivate_employee(self, agent_id):
        self.reactivated.append(agent_id)

    def close(self):
        self.closed = True


class _GroupMemory:
    def read_group_memory(self, _chat_id):
        return ""


class _StableGroupMemory(_GroupMemory):
    def __init__(self, content: str) -> None:
        self.content = content

    def read_group_memory(self, _chat_id):
        return self.content


class _RuntimeMessageSource:
    def __init__(self, scope) -> None:
        self.scope = scope
        self._thread_pass = 0

    def resolve_thread(self):
        return ResolvedThread(
            self.scope.thread_root_message_id,
            self.scope.feishu_thread_id,
            self.scope.current_message_id,
        )

    def list_thread_messages(self, *, page_token="", page_size=50):
        del page_token, page_size
        self._thread_pass += 1
        messages = (
            self._message(
                self.scope.thread_root_message_id,
                text="root",
                create_time=1_000,
                position=0,
                root_id="",
                current=False,
            ),
            self._message(
                self.scope.current_message_id,
                text="current",
                create_time=2_000,
                position=1,
                root_id=self.scope.thread_root_message_id,
                current=True,
            ),
        )
        return MessagePage(messages, False)

    def list_chat_messages(self, *, page_token="", page_size=20):
        del page_token, page_size
        return MessagePage((), False)

    def reset_chat_traversal(self):
        return None

    def close(self):
        return None

    def _message(
        self,
        message_id,
        *,
        text,
        create_time,
        position,
        root_id,
        current,
    ):
        return ContextMessage(
            message_id=message_id,
            sender_id="ou_admin",
            sender_type="user",
            text=text,
            timestamp=create_time / 1000,
            is_system=False,
            edited=False,
            deleted=False,
            chat_id=self.scope.chat_id,
            thread_id=self.scope.feishu_thread_id,
            root_id=root_id,
            parent_id=root_id,
            sender_id_type="open_id",
            sender_tenant_key=self.scope.tenant_key,
            msg_type="text",
            create_time_ms=create_time,
            update_time_ms=create_time,
            message_position=position,
            thread_message_position=position,
            is_current=current,
        )


class _AssemblingContextSourceFactory(_ContextSourceFactory):
    @contextmanager
    def open(self, *, scope, principal):
        del principal
        yield _RuntimeMessageSource(scope)


class _ReplayContextSource(_RuntimeMessageSource):
    def __init__(self, scope, *, split_pages: bool) -> None:
        super().__init__(scope)
        self._split_pages = split_pages

    def _thread_messages(self):
        root = replace(
            self._message(
                self.scope.thread_root_message_id,
                text="root-edited",
                create_time=1_000,
                position=0,
                root_id="",
                current=False,
            ),
            update_time_ms=1_500,
            edited=True,
        )
        deleted = replace(
            self._message(
                "om_runtime_deleted",
                text="",
                create_time=1_800,
                position=1,
                root_id=self.scope.thread_root_message_id,
                current=False,
            ),
            update_time_ms=2_200,
            deleted=True,
        )
        current = self._message(
            self.scope.current_message_id,
            text="current",
            create_time=3_000,
            position=2,
            root_id=self.scope.thread_root_message_id,
            current=True,
        )
        return root, deleted, current

    def list_thread_messages(self, *, page_token="", page_size=50):
        del page_size
        messages = self._thread_messages()
        if self._split_pages:
            if not page_token:
                return MessagePage(messages[:2], True, "next")
            assert page_token == "next"
            return MessagePage(messages[2:], False)
        assert not page_token
        return MessagePage(messages, False)

    def list_chat_messages(self, *, page_token="", page_size=20):
        del page_token, page_size
        root = self._thread_messages()[0]
        group = self._message(
            "om_runtime_group",
            text="group",
            create_time=2_500,
            position=10,
            root_id="",
            current=False,
        )
        return MessagePage((group, root), False)


class _ReplayContextSourceFactory(_ContextSourceFactory):
    def __init__(self, *, split_pages: bool) -> None:
        super().__init__()
        self._split_pages = split_pages

    @contextmanager
    def open(self, *, scope, principal):
        del principal
        yield _ReplayContextSource(
            scope,
            split_pages=self._split_pages,
        )


class _FailingStartChannels(_Channels):
    def start(self, agent_id, app_id, credential_ref, generation, on_event):
        del agent_id, app_id, credential_ref, generation, on_event
        raise RuntimeError("injected crash before Channel READY")


class _FailOnceStartChannels(_Channels):
    def __init__(self) -> None:
        super().__init__()
        self.attempted_generations: list[int] = []

    def start(self, agent_id, app_id, credential_ref, generation, on_event):
        self.attempted_generations.append(generation)
        if len(self.attempted_generations) == 1:
            raise RuntimeError("injected first Channel launch failure")
        return super().start(
            agent_id,
            app_id,
            credential_ref,
            generation,
            on_event,
        )


class _FailingVerificationRouter:
    def issue_challenge(self, _binding):
        raise RuntimeError("injected crash after Channel READY")


class _ForgedReceiptChannels(_Channels):
    def send(self, agent_id, *, generation, target, message, options=None):
        del agent_id, target, message, options
        return ChannelSendReceipt(
            request_id="send_forged",
            success=True,
            app_id="cli_main_bot",
            generation=generation,
            connection_id="conn_runtime",
            message_id="om_forged",
        )


def _request() -> EmployeeHireRequest:
    return EmployeeHireRequest(
        employee_name="Atlas",
        tool="codex",
        model="gpt-5.6-sol",
        effort="high",
        chat_id="oc_admin_dm",
        message_id="om_composition",
        requester_principal_id="ou_admin",
        requester_union_id="on_admin",
        tenant_key="tenant-a",
    )


def _runtime(
    settings: SimpleNamespace,
    *,
    release_evidence_ready: bool,
    **kwargs: object,
) -> EmployeeDepartmentRuntime:
    """Compose the built-in local runtime; release verdicts are obsolete."""
    del release_evidence_ready
    kwargs.setdefault("slock_engine_manager", _SlockManager())
    kwargs.setdefault(
        "employee_environment_provider",
        lambda authority: EmployeeProcessEnvironmentMaterial(
            authority.tenant_key,
            authority.agent_id,
            authority.employee_version,
            authority.credential_ref,
            {"PATH": "/usr/bin"},
            {},
        ),
    )
    return EmployeeDepartmentRuntime.from_settings(settings, **kwargs)


def test_task7_runtime_owns_durable_ingress_router_and_gateway(tmp_path: Path) -> None:
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )

    assert runtime.ingress_service is not None
    assert runtime.ingress_router is not None
    assert runtime.dispatch_coordinator is not None
    assert runtime.dispatch_coordinator.employee_runtime is not None
    assert runtime.outbox_service is not None
    assert runtime.outbox_delivery is not None
    assert runtime.fire_service is not None
    assert runtime.execution_readiness().ready is True
    invalid_chat = runtime.execution_readiness(chat_id=" ")
    assert invalid_chat.ready is False
    assert invalid_chat.blockers == ("context_group_history",)
    runtime.close()


def test_team_batch_skips_nonmembers_before_employee_probe() -> None:
    runtime = EmployeeDepartmentRuntime()
    member = SimpleNamespace(agent_id="agent-member", tenant_key="tenant-a")
    nonmember = SimpleNamespace(agent_id="agent-other", tenant_key="tenant-a")
    projection = SimpleNamespace(
        employees={
            member.agent_id: SimpleNamespace(member_groups=frozenset({"oc_team"})),
            nonmember.agent_id: SimpleNamespace(member_groups=frozenset({"oc_other"})),
        }
    )
    runtime._prepare_execution_probe = lambda _agent_id=None: (  # type: ignore[method-assign]  # noqa: SLF001
        RuntimeReadiness(True, ()),
        projection,
        (member, nonmember),
    )
    probed: list[str] = []

    def probe(_projection, state, *, chat_id, probe_remote=True):
        assert chat_id == "oc_team"
        assert probe_remote is False
        probed.append(state.agent_id)
        return RuntimeReadiness(True, ())

    runtime._probe_employee_execution = probe  # type: ignore[method-assign]  # noqa: SLF001

    assert runtime._team_execution_ready_agent_ids(  # noqa: SLF001
        "tenant-a",
        "oc_team",
    ) == frozenset({"agent-member"})
    assert probed == ["agent-member"]


def test_existing_app_id_reaches_runtime_registrar(tmp_path: Path) -> None:
    registrar = _RecordingExistingRegistrar()
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=registrar,
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None

    admitted = runtime.hire_service.start_hire(
        replace(_request(), existing_app_id="cli_existing_123")
    )
    deadline = time.monotonic() + 5
    current = admitted
    while time.monotonic() < deadline:
        current = runtime.hire_service.get_state(admitted.intent_id) or current
        if current.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)

    assert current.phase is HirePhase.READY_PENDING_VERIFICATION
    assert len(registrar.requests) == 1
    assert registrar.requests[0].existing_app_id == "cli_existing_123"
    assert current.existing_app_id == "cli_existing_123"
    assert current.app_id == "cli_existing_123"
    runtime.close()


def test_runtime_composes_canonical_membership_service(tmp_path: Path) -> None:
    manager_client = object()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        manager_client_factory=lambda: manager_client,
    )

    assert runtime.membership_service is not None
    assert runtime.ingress_router is not None
    assert runtime.ingress_router._membership_health is runtime.membership_service
    runtime.close()


def test_task7_execution_readiness_requires_slock_gateway(tmp_path: Path) -> None:
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        slock_engine_manager=None,
    )

    assert runtime.hire_readiness().ready is True
    assert runtime.execution_readiness().blockers == ("slock_gateway",)
    runtime.close()


def test_task7_execution_readiness_requires_environment_provider(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        employee_environment_provider=None,
    )

    assert runtime.hire_readiness().ready is True
    assert runtime.execution_readiness().blockers == ("employee_environment",)
    runtime.close()


def test_phase4_execution_readiness_requires_employee_card_update_capability(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    channels.update_card = None  # type: ignore[method-assign]
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )

    assert runtime.hire_readiness().ready is True
    assert runtime.execution_readiness().blockers == ("employee_outbox",)
    runtime.close()


def test_task7_recovers_committed_attempts_before_starting_dispatch(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    with (
        patch.object(
            EmployeeDispatchCoordinator,
            "recover_incomplete_attempts",
            autospec=True,
            side_effect=lambda _self: events.append("recover") or (),
        ),
        patch.object(
            EmployeeDepartmentRuntime,
            "_start_dispatch_worker",
            autospec=True,
            side_effect=lambda _self: events.append("start"),
        ),
    ):
        runtime = _runtime(
            _settings(tmp_path, limit=1, context_configured=True),
            release_evidence_ready=True,
            registrar=_Registrar(),
            channel_supervisor=_Channels(),
            slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
            notification_link=lambda *_: None,
            context_source_factory=_ContextSourceFactory(),
            group_memory_backend=_GroupMemory(),
        )

    assert events == ["recover", "start"]
    runtime.close()


def test_task7_execution_recovery_failure_is_a_stable_blocker(
    tmp_path: Path,
) -> None:
    with patch.object(
        EmployeeDispatchCoordinator,
        "recover_incomplete_attempts",
        autospec=True,
        side_effect=RuntimeError("injected recovery failure"),
    ):
        runtime = _runtime(
            _settings(tmp_path, limit=1, context_configured=True),
            release_evidence_ready=True,
            registrar=_Registrar(),
            channel_supervisor=_Channels(),
            slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
            notification_link=lambda *_: None,
            context_source_factory=_ContextSourceFactory(),
            group_memory_backend=_GroupMemory(),
        )

    assert runtime.hire_readiness().ready is True
    assert runtime.execution_readiness().blockers == ("employee_recovery",)
    assert runtime._dispatch_thread is None
    runtime.close()


def test_task7_stuck_dispatch_holds_journal_and_vault_open(tmp_path: Path) -> None:
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )
    assert runtime._dispatch_thread is not None
    runtime._dispatch_stop.set()
    runtime._dispatch_thread.join(timeout=1)

    class _StuckThread:
        alive = True

        def join(self, timeout: float) -> None:
            assert timeout == 5.0

        def is_alive(self) -> bool:
            return self.alive

    stuck = _StuckThread()
    runtime._dispatch_thread = stuck  # type: ignore[assignment]
    assert runtime._writer is not None
    assert runtime._vault is not None

    runtime.close()

    assert runtime._writer._closed is False
    assert runtime._vault._root_finalizer.alive is True
    stuck.alive = False
    runtime.close()
    assert runtime._writer._closed is True
    assert runtime._vault._root_finalizer.alive is False


def test_task7_runtime_routes_anchored_inbox_into_owned_queue(tmp_path: Path) -> None:
    channels = _Channels()
    settings = _settings(tmp_path, limit=1, context_configured=True)
    settings.allowed_chat_ids = frozenset({"oc_employee_team"})
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    active = _activate_employee(runtime, channels)
    channels.statuses[active.agent_id].tenant_key = active.tenant_key
    channels.statuses[active.agent_id].bot_principal_id = active.bot_principal_id
    assert runtime._writer is not None
    assert runtime.hire_service is not None
    assert runtime.ingress_service is not None
    assert runtime.ingress_router is not None
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    runtime._dispatch_stop.set()
    assert runtime._dispatch_thread is not None
    runtime._dispatch_thread.join(timeout=1)

    class _NoDispatch:
        def dispatch_next(self):
            return None

    runtime._dispatch = _NoDispatch()  # type: ignore[assignment]
    suffix = hashlib.sha256(b"composition-route").hexdigest()
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id=f"ing_{suffix}",
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "group",
                "content": {"text": "route me"},
                "sender_id": "ou_employee_app_admin",
                "sender_union_id": "on_admin",
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": active.tenant_key,
                "feishu_thread_id": "omt_composition",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key=active.tenant_key,
        agent_id=active.agent_id,
        bot_principal_id=active.bot_principal_id,
        app_id=active.app_id,
        channel_generation=active.channel_generation,
        connection_id=active.channel_connection_id,
        event_id=f"evt_{suffix[:24]}",
        message_id=f"om_{suffix[:24]}",
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_employee_team",
        thread_root_message_id="om_runtime_root",
        sender_principal_id="ou_employee_app_admin",
        received_at="2026-07-14T00:00:00Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    acceptance = runtime.ingress_service.accept(
        metadata,
        payload,
        request_id="req_composition_route",
    )

    assert runtime._drain_employee_dispatch_once() is True
    acceptance_id = acceptance.acceptance.acceptance_id
    routed = runtime.ingress_router.state.by_acceptance_id[acceptance_id]
    assert routed.state == "queued", routed.reason_code
    assert routed.requester_principal_id == active.requester_principal_id
    assert runtime.ingress_service.state.by_acceptance_id[acceptance_id].disposition is None
    runtime._dispatch_thread = None
    runtime.close()


def test_pending_group_direct_mention_is_durable_and_not_routed(
    tmp_path: Path,
) -> None:
    class _UnknownFirstReceiptChannels(_Channels):
        def send(self, agent_id, *, generation, target, message, options=None):
            receipt = super().send(
                agent_id,
                generation=generation,
                target=target,
                message=message,
                options=options,
            )
            if len(self.sent) == 1:
                return replace(receipt, request_id="", message_id="")
            return receipt

    channels = _UnknownFirstReceiptChannels()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    assert runtime.hire_service is not None
    assert runtime.ingress_service is not None
    assert runtime.ingress_router is not None
    assert runtime._writer is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = admitted
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id) or pending
        if pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending.phase is HirePhase.READY_PENDING_VERIFICATION
    channels.statuses[pending.agent_id].tenant_key = pending.tenant_key
    channels.statuses[pending.agent_id].bot_principal_id = pending.bot_principal_id
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=pending.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    runtime._dispatch_stop.set()
    assert runtime._dispatch_thread is not None
    runtime._dispatch_thread.join(timeout=1)

    class _NoDispatch:
        def dispatch_next(self):
            return None

    runtime._dispatch = _NoDispatch()  # type: ignore[assignment]
    raw_chat_id = "oc_employee_team"
    raw_message_id = "om_pending_direct_mention"
    suffix = hashlib.sha256(b"pending-direct-mention").hexdigest()
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id=f"ing_{suffix}",
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "group",
                "content": {"text": "@_user_1 你好"},
                "mentions": (
                    {
                        "key": "@_user_1",
                        "mentioned_type": "bot",
                        "open_id": channels.statuses[pending.agent_id].identity[
                            "open_id"
                        ],
                        "tenant_key": pending.tenant_key,
                    },
                ),
                "sender_id": "ou_employee_app_admin",
                "sender_union_id": pending.requester_union_id,
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": pending.tenant_key,
                "feishu_thread_id": "",
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": "",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key=pending.tenant_key,
        agent_id=pending.agent_id,
        bot_principal_id=pending.bot_principal_id,
        app_id=pending.app_id,
        channel_generation=pending.channel_generation,
        connection_id=pending.channel_connection_id,
        event_id=f"evt_{suffix}",
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        thread_root_message_id="",
        sender_principal_id="ou_employee_app_admin",
        received_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    accepted = runtime.ingress_service.accept(
        metadata,
        payload,
        request_id="req_pending_direct_mention",
    )

    acceptance_id = accepted.acceptance.acceptance_id
    effect_id = f"activation-required-reply:{metadata.event_id}"
    with pytest.raises(
        RuntimeError,
        match="activation-required receipt is invalid",
    ):
        runtime._drain_employee_dispatch_once()
    executing = runtime.hire_service.get_state(pending.intent_id)
    assert executing is not None
    assert executing.effect_state(effect_id) is HireEffectState.EXECUTING
    runtime.ingress_service.rebuild_projection()
    assert (
        runtime.ingress_service.state.by_acceptance_id[acceptance_id].disposition
        is None
    )
    assert runtime._drain_employee_dispatch_once() is True
    runtime.ingress_service.rebuild_projection()
    disposition = runtime.ingress_service.state.by_acceptance_id[
        acceptance_id
    ].disposition
    assert disposition is not None
    assert disposition.state == "terminal"
    assert disposition.reason_code == "activation_required"
    unrouted = runtime.ingress_router.state.by_acceptance_id[acceptance_id]
    assert unrouted.state == "accepted"
    assert unrouted.authority is None
    assert len(channels.sent) == 2
    assert channels.sent[0][2] == raw_chat_id
    expected_options = {
        "uuid": hashlib.sha256(
            f"employee-activation-required:{pending.agent_id}:{acceptance_id}".encode()
        ).hexdigest()[:50],
        "reply_to": raw_message_id,
    }
    assert channels.sent[0][4] == channels.sent[1][4] == expected_options
    effect_events = [
        event.event_type
        for frame in runtime.journal_frames()
        for event in frame.events
        if event.payload.get("effect_id") == effect_id
    ]
    assert effect_events == [
        "hire.effect.prepared",
        "hire.effect.executing",
        "hire.effect.committed",
    ]
    runtime.hire_service.recover()
    recovered = runtime.hire_service.get_state(pending.intent_id)
    assert recovered is not None
    assert recovered.effect_state(effect_id) is HireEffectState.COMMITTED
    runtime._drain_employee_dispatch_once()
    assert len(channels.sent) == 2
    runtime._dispatch_thread = None
    runtime.close()


def test_pending_group_reply_unknown_outcome_is_disposed_across_runtime_restart(
    tmp_path: Path,
) -> None:
    class _UnknownReceiptChannels(_Channels):
        def send(self, *args, **kwargs):
            receipt = super().send(*args, **kwargs)
            return replace(receipt, request_id="", message_id="")

    class _RestartChannels(_Channels):
        def start(self, *args, **kwargs):
            status = super().start(*args, **kwargs)
            status.ready_metadata["connection_id"] = "conn_runtime_restarted"
            return status

    settings = _settings(tmp_path, limit=1, context_configured=True)
    first_channels = _UnknownReceiptChannels()
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=first_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = admitted
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id) or pending
        if pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending.phase is HirePhase.READY_PENDING_VERIFICATION
    first_channels.statuses[pending.agent_id].tenant_key = pending.tenant_key
    first_channels.statuses[pending.agent_id].bot_principal_id = (
        pending.bot_principal_id
    )
    commit_workforce_events(
        first._writer,
        first.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=pending.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    first._dispatch_stop.set()
    assert first._dispatch_thread is not None
    first._dispatch_thread.join(timeout=1)

    class _NoDispatch:
        def dispatch_next(self):
            return None

    first._dispatch = _NoDispatch()  # type: ignore[assignment]
    accepted = _accept_pending_group_mention(
        first,
        pending,
        bot_open_id=first_channels.statuses[pending.agent_id].identity[
            "open_id"
        ],
        suffix="pending_restart_unknown",
    )
    acceptance_id = accepted.acceptance.acceptance_id
    event_id = "evt_" + hashlib.sha256(
        b"pending_restart_unknown"
    ).hexdigest()
    effect_id = f"activation-required-reply:{event_id}"
    with pytest.raises(
        RuntimeError,
        match="activation-required receipt is invalid",
    ):
        first._drain_employee_dispatch_once()
    unresolved = first.hire_service.get_state(pending.intent_id)
    assert unresolved is not None
    assert unresolved.effect_state(effect_id) is HireEffectState.EXECUTING
    first._dispatch_thread = None
    first.close()

    restarted_channels = _RestartChannels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    deadline = time.monotonic() + 7
    recovered = None
    while time.monotonic() < deadline:
        recovered = restarted.hire_service.get_state(pending.intent_id)
        if (
            recovered is not None
            and recovered.phase is HirePhase.READY_PENDING_VERIFICATION
            and recovered.channel_generation == pending.channel_generation + 1
            and recovered.channel_connection_id == "conn_runtime_restarted"
        ):
            break
        time.sleep(0.05)
    assert recovered is not None
    assert recovered.phase is HirePhase.READY_PENDING_VERIFICATION
    assert recovered.channel_generation == pending.channel_generation + 1
    assert recovered.effect_state(effect_id) is HireEffectState.ACTION_REQUIRED
    assert recovered.metadata_for(effect_id)["error_code"] == (
        "activation_required_reply_outcome_unknown"
    )
    assert restarted_channels.sent == []

    status = _accept_durable_status(
        restarted,
        recovered,
        suffix="pending_restart_status",
        generation=recovered.channel_generation,
        connection_id=recovered.channel_connection_id,
    )
    restarted._handle_control_ingress(status.acceptance.acceptance_id)
    while time.monotonic() < deadline:
        active = restarted.hire_service.get_state(pending.intent_id)
        if active is not None and active.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)
    assert active is not None
    assert active.phase is HirePhase.ACTIVE
    assert len(restarted_channels.sent) == 2
    restarted.ingress_service.rebuild_projection()
    old_disposition = restarted.ingress_service.state.by_acceptance_id[
        acceptance_id
    ].disposition
    assert old_disposition is not None
    assert old_disposition.reason_code == "activation_reply_outcome_unknown"
    restarted.close()


@pytest.mark.parametrize(
    "retain_membership",
    [True, False],
    ids=("binding-valid", "membership-revoked"),
)
def test_prepared_group_reply_resumes_on_new_channel_generation(
    tmp_path: Path,
    retain_membership: bool,
) -> None:
    class _RestartChannels(_Channels):
        def start(self, *args, **kwargs):
            status = super().start(*args, **kwargs)
            status.bot_principal_id = pending.bot_principal_id
            status.ready_metadata["connection_id"] = "conn_prepared_restarted"
            return status

    settings = _settings(tmp_path, limit=1, context_configured=True)
    first_channels = _Channels()
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=first_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = admitted
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id) or pending
        if pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending.phase is HirePhase.READY_PENDING_VERIFICATION
    first_channels.statuses[pending.agent_id].tenant_key = pending.tenant_key
    first_channels.statuses[pending.agent_id].bot_principal_id = (
        pending.bot_principal_id
    )
    commit_workforce_events(
        first._writer,
        first.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=pending.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    first._dispatch_stop.set()
    assert first._dispatch_thread is not None
    first._dispatch_thread.join(timeout=1)

    class _NoDispatch:
        def dispatch_next(self):
            return None

    first._dispatch = _NoDispatch()  # type: ignore[assignment]
    accepted = _accept_pending_group_mention(
        first,
        pending,
        bot_open_id=first_channels.statuses[pending.agent_id].identity[
            "open_id"
        ],
        suffix="pending_restart_prepared",
    )
    acceptance_id = accepted.acceptance.acceptance_id
    event_id = "evt_" + hashlib.sha256(
        b"pending_restart_prepared"
    ).hexdigest()
    effect_id = f"activation-required-reply:{event_id}"
    original_transition = first.hire_service.commit_effect_transition

    def fail_before_execute(intent_id, **kwargs):
        if (
            kwargs.get("effect_type")
            == "employee_activation_required_reply"
            and kwargs.get("next_state") is HireEffectState.EXECUTING
        ):
            raise RuntimeError("injected crash before external dispatch")
        return original_transition(intent_id, **kwargs)

    first.hire_service.commit_effect_transition = fail_before_execute  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="before external dispatch"):
        first._drain_employee_dispatch_once()
    prepared = first.hire_service.get_state(pending.intent_id)
    assert prepared is not None
    assert prepared.effect_state(effect_id) is HireEffectState.PREPARED
    assert first_channels.sent == []
    first.hire_service.commit_effect_transition = original_transition  # type: ignore[method-assign]
    if not retain_membership:
        commit_workforce_events(
            first._writer,
            first.hire_service.projection_state,
            (
                JournalEvent(
                    event_type="employee.membership_changed",
                    aggregate_id=pending.agent_id,
                    payload={"member_groups": []},
                ),
            ),
        )
    first._dispatch_thread = None
    first.close()

    restarted_channels = _RestartChannels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    deadline = time.monotonic() + 7
    recovered = None
    disposition = None
    expected_effect_state = (
        HireEffectState.COMMITTED
        if retain_membership
        else HireEffectState.ACTION_REQUIRED
    )
    while time.monotonic() < deadline:
        recovered = restarted.hire_service.get_state(pending.intent_id)
        restarted.ingress_service.rebuild_projection()
        disposition = restarted.ingress_service.state.by_acceptance_id[
            acceptance_id
        ].disposition
        if (
            recovered is not None
            and recovered.phase is HirePhase.READY_PENDING_VERIFICATION
            and recovered.channel_generation == pending.channel_generation + 1
            and recovered.channel_connection_id == "conn_prepared_restarted"
            and recovered.effect_state(effect_id) is expected_effect_state
            and disposition is not None
        ):
            break
        time.sleep(0.05)
    assert recovered is not None
    assert recovered.channel_generation == pending.channel_generation + 1
    assert recovered.effect_state(effect_id) is expected_effect_state
    assert disposition is not None
    if retain_membership:
        assert disposition.reason_code == "activation_required"
        assert len(restarted_channels.sent) == 1
        assert restarted_channels.sent[0][4] == {
            "uuid": hashlib.sha256(
                f"employee-activation-required:{pending.agent_id}:{acceptance_id}".encode()
            ).hexdigest()[:50],
            "reply_to": "om_pending_restart_prepared",
        }
        assert recovered.metadata_for(effect_id)["generation"] == str(
            recovered.channel_generation
        )
        assert recovered.metadata_for(effect_id)["connection_id"] == (
            "conn_prepared_restarted"
        )
    else:
        assert disposition.reason_code == "activation_reply_recovery_rejected"
        assert recovered.metadata_for(effect_id)["error_code"] == (
            "activation_required_reply_recovery_rejected"
        )
        assert restarted_channels.sent == []

    status = _accept_durable_status(
        restarted,
        recovered,
        suffix="prepared_restart_status",
        generation=recovered.channel_generation,
        connection_id=recovered.channel_connection_id,
    )
    restarted._handle_control_ingress(status.acceptance.acceptance_id)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        active = restarted.hire_service.get_state(pending.intent_id)
        if active is not None and active.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)
    assert active is not None
    assert active.phase is HirePhase.ACTIVE
    restarted.close()


def test_team_assignment_uses_canonical_employee_ingress_queue(tmp_path: Path) -> None:
    channels = _Channels()
    settings = _settings(tmp_path, limit=1, context_configured=True)
    settings.allowed_chat_ids = frozenset({"oc_employee_team"})
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=_AssemblingContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    active = _activate_employee(runtime, channels)
    channels.statuses[active.agent_id].tenant_key = active.tenant_key
    channels.statuses[active.agent_id].bot_principal_id = active.bot_principal_id
    assert runtime._writer is not None
    assert runtime.hire_service is not None
    assert runtime.team_service is not None
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    runtime._dispatch_stop.set()
    assert runtime._dispatch_thread is not None
    runtime._dispatch_thread.join(timeout=1)

    class _NoDispatch:
        def dispatch_next(self):
            return None

    real_dispatch = runtime._dispatch
    runtime._dispatch = _NoDispatch()  # type: ignore[assignment]
    backend = runtime.team_service._backend
    target = backend.list_active(active.tenant_key, "oc_employee_team")[0]
    runtime.team_service._commit_effect(  # noqa: SLF001
        "teamrun_integration:analysis",
        "employee_dispatch",
        "prepared",
    )
    runtime.team_service._commit_effect(  # noqa: SLF001
        "teamrun_integration:analysis",
        "employee_dispatch",
        "executing",
    )
    acceptance_id = backend.submit(
        run_id="teamrun_integration",
        step_id="analysis",
        target=target,
        tenant_key=active.tenant_key,
        chat_id="oc_employee_team",
        message_id="om_team_task",
        requester_principal_id="ou_admin",
        instruction="分析并修复团队模式",
        deadline_at="2099-01-01T00:00:00Z",
    )

    assert runtime._drain_employee_dispatch_once() is True
    routed = runtime.ingress_router.state.by_acceptance_id[acceptance_id]
    assert routed.state == "queued", routed.reason_code
    payload = runtime.ingress_service.get_payload(acceptance_id)
    assert payload.normalized_parts[0]["type"] == "team_assignment"
    assert payload.normalized_parts[0]["team_instruction"] == "分析并修复团队模式"
    original_check = real_dispatch._team_assignment_effect_is_active  # noqa: SLF001
    check_calls = 0

    def race_after_active_check(part):
        nonlocal check_calls
        check_calls += 1
        active = original_check(part)
        if check_calls == 2:
            assert active is True
            runtime.team_service._commit_effect(  # noqa: SLF001
                "teamrun_integration:analysis",
                "employee_dispatch",
                "action_required",
            )
        return active

    real_dispatch._team_assignment_effect_is_active = race_after_active_check  # type: ignore[method-assign]  # noqa: SLF001
    # The effect changes after the check but before dispatch commit. The head
    # CAS must force a retry, which then terminalizes the stale assignment.
    assert real_dispatch.prepare_next() is None
    assert check_calls == 3
    terminal = runtime.ingress_router.state.by_acceptance_id[acceptance_id]
    assert terminal.state == "terminal"
    assert terminal.reason_code == "team_step_inactive"
    runtime._dispatch = real_dispatch
    runtime._dispatch_thread = None
    runtime.close()


def test_runtime_recovers_team_before_it_can_start_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: list[str] = []
    original_team_recover = EmployeeTeamService.recover
    original_runtime_recover = EmployeeDepartmentRuntime.recover

    def recover_team(service):
        assert observed == ["runtime"]
        observed.append("team")
        return original_team_recover(service)

    def recover_runtime(runtime):
        observed.append("runtime")
        return original_runtime_recover(runtime)

    monkeypatch.setattr(EmployeeTeamService, "recover", recover_team)
    monkeypatch.setattr(EmployeeDepartmentRuntime, "recover", recover_runtime)
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )

    assert observed == ["runtime", "team"]
    assert runtime.team_service is not None
    assert runtime.team_service.runtime_mode == "coordinator"
    runtime.close()


def test_runtime_composition_can_defer_all_recovery_until_notifier_is_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recoveries: list[str] = []
    original_recover = EmployeeDepartmentRuntime.recover

    def recover_runtime(runtime):
        recoveries.append("recover")
        return original_recover(runtime)

    monkeypatch.setattr(EmployeeDepartmentRuntime, "recover", recover_runtime)
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        recover_immediately=False,
    )

    assert recoveries == []
    runtime.recover()
    assert recoveries == ["recover"]
    runtime.close()


def _persist_queued_team_assignment_then_crash(root: str, pipe) -> None:
    """Subprocess half of the restart E2E; deliberately skips runtime.close()."""

    import os

    settings = _settings(Path(root), limit=1, context_configured=True)
    settings.allowed_chat_ids = frozenset({"oc_employee_team"})
    channels = _Channels()
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=_AssemblingContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    active = _activate_employee(runtime, channels)
    channels.statuses[active.agent_id].tenant_key = active.tenant_key
    channels.statuses[active.agent_id].bot_principal_id = active.bot_principal_id
    assert runtime._writer is not None
    assert runtime.hire_service is not None
    assert runtime.team_service is not None
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    runtime._dispatch_stop.set()
    assert runtime._dispatch_thread is not None
    runtime._dispatch_thread.join(timeout=2)
    assert not runtime._dispatch_thread.is_alive()
    runtime.team_service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="team.run.created",
            aggregate_id="teamrun_restart_e2e",
            payload={
                "tenant_key": active.tenant_key,
                "message_id": "om_team_restart",
                "chat_id": "oc_employee_team",
                "requester_principal_id": "ou_admin",
                "task_digest": "0" * 64,
                "max_handoffs": 8,
                "max_depth": 4,
                "max_fanout": 4,
            },
        )
    )
    aggregate = "teamrun_restart_e2e:analysis"
    runtime.team_service._commit_effect(  # noqa: SLF001
        aggregate, "employee_dispatch", "prepared"
    )
    runtime.team_service._commit_effect(  # noqa: SLF001
        aggregate, "employee_dispatch", "executing"
    )
    backend = runtime.team_service._backend  # noqa: SLF001
    target = backend.list_active(active.tenant_key, "oc_employee_team")[0]
    acceptance_id = backend.submit(
        run_id="teamrun_restart_e2e",
        step_id="analysis",
        target=target,
        tenant_key=active.tenant_key,
        chat_id="oc_employee_team",
        message_id="om_team_restart",
        requester_principal_id="ou_admin",
        instruction="queued before abrupt process exit",
        deadline_at="2099-01-01T00:00:00Z",
    )
    assert runtime.ingress_router is not None
    routed = runtime.ingress_router.route(acceptance_id)
    assert routed.state == "queued", routed.reason_code
    # Persist the exact pre-v2 retry frame so runtime reconstruction, rather
    # than only the Router unit, proves upgrade compatibility after the crash.
    runtime.team_service._commit(  # noqa: SLF001
        JournalEvent(
            event_type="employee.ingress.router_context_retry",
            aggregate_id=routed.aggregate_id,
            payload={"acceptance_id": acceptance_id, "failure_count": 1},
        )
    )
    pipe.send(acceptance_id)
    pipe.close()
    os._exit(0)


def test_queued_team_assignment_restart_never_dispatches_acp(tmp_path: Path) -> None:
    """A real process crash is recovered before Context/Slock/ACP dispatch."""

    import multiprocessing

    settings = _settings(tmp_path, limit=1, context_configured=True)
    settings.allowed_chat_ids = frozenset({"oc_employee_team"})
    process_context = multiprocessing.get_context("spawn")
    parent_pipe, child_pipe = process_context.Pipe(
        duplex=False
    )
    process = process_context.Process(
        target=_persist_queued_team_assignment_then_crash,
        args=(str(tmp_path), child_pipe),
    )
    process.start()
    child_pipe.close()
    assert parent_pipe.poll(10), "crashed runtime did not persist queued assignment"
    acceptance_id = parent_pipe.recv()
    process.join(10)
    assert process.exitcode == 0

    from src.slock_engine.manager import ActivatedSlockBinding

    class _CountingEngine:
        def __init__(self):
            self.run_agent_session_calls = 0

        def run_agent_session(self, *_args, **_kwargs):
            self.run_agent_session_calls += 1
            return "must not execute"

    class _CountingSlock:
        def __init__(self, engine):
            self.engine = engine
            self.activation_calls = 0

        @contextmanager
        def employee_activation_guard(self, *, chat_id, **_kwargs):
            self.activation_calls += 1
            yield ActivatedSlockBinding(
                engine_identity="e" * 64,
                chat_id=chat_id,
                root_identity="r" * 64,
                canonical_root=str(tmp_path),
                channel_id=chat_id,
                engine=self.engine,
            )

        def resolve_employee_engine(self, **_kwargs):
            return self.engine

        def close(self):
            return None

    class _CountingContextSource(_RuntimeMessageSource):
        def __init__(self, scope, factory):
            super().__init__(scope)
            self._factory = factory

        def list_thread_messages(self, **kwargs):
            self._factory.api_calls += 1
            return super().list_thread_messages(**kwargs)

        def list_chat_messages(self, **kwargs):
            self._factory.api_calls += 1
            return super().list_chat_messages(**kwargs)

    class _CountingContextFactory(_ContextSourceFactory):
        def __init__(self):
            super().__init__()
            self.open_calls = 0
            self.api_calls = 0

        @contextmanager
        def open(self, *, scope, principal):
            del principal
            self.open_calls += 1
            yield _CountingContextSource(scope, self)

    engine = _CountingEngine()
    slock = _CountingSlock(engine)
    context_source = _CountingContextFactory()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=context_source,
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
        slock_engine_manager=slock,
    )
    try:
        deadline = time.monotonic() + 5
        record = None
        while time.monotonic() < deadline:
            assert restarted.ingress_router is not None
            restarted.ingress_router.rebuild_projection()
            record = restarted.ingress_router.state.by_acceptance_id.get(
                acceptance_id
            )
            if record is not None and record.state == "terminal":
                break
            restarted._drain_employee_dispatch_once()
            time.sleep(0.01)

        assert record is not None and record.state == "terminal"
        assert restarted.dispatch_coordinator is not None
        assert not restarted.dispatch_coordinator.state.attempts
        assert context_source.open_calls == 0
        assert context_source.api_calls == 0
        assert slock.activation_calls == 0
        assert engine.run_agent_session_calls == 0
    finally:
        restarted.close()


def _activate_employee(
    runtime: EmployeeDepartmentRuntime,
    channels: _Channels,
) -> DurableHireState:
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    ack = _accept_durable_status(
        runtime,
        pending,
        suffix="context_ready",
        generation=pending.channel_generation,
        connection_id=pending.channel_connection_id,
    )
    runtime._handle_control_ingress(ack.acceptance.acceptance_id)
    active = None
    while time.monotonic() < deadline:
        active = runtime.hire_service.get_state(admitted.intent_id)
        if active is not None and active.phase is HirePhase.ACTIVE:
            return active
        time.sleep(0.02)
    raise AssertionError("employee did not become ACTIVE")


def test_new_employee_auto_activates_after_local_channel_ready(tmp_path: Path) -> None:
    settings = _settings(tmp_path, limit=1)
    settings.autonomous_employee_auto_activation = True
    channels = _Channels()
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None

    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    current = admitted
    while time.monotonic() < deadline:
        current = runtime.hire_service.get_state(admitted.intent_id) or current
        if current.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)

    assert current.phase is HirePhase.ACTIVE
    assert current.channel_generation == 1
    assert current.activation_ingress_event_id == ""
    assert current.activation_source == "channel_ready"
    assert current.verification_consumed is True
    event_types = [
        event.event_type
        for frame in runtime.journal_frames()
        for event in frame.events
        if event.aggregate_id == current.intent_id
    ]
    assert "hire.activation.automatic" in event_types
    runtime.close()


def test_restart_auto_activates_all_local_employees_on_new_generation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=2)
    settings.autonomous_employee_auto_activation = True
    first_channels = _Channels()
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_DistinctRegistrar(),
        channel_supervisor=first_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    second = first.hire_service.start_hire(
        replace(
            _request(),
            employee_name="Vega",
            message_id="om_composition_second_auto",
        )
    )
    deadline = time.monotonic() + 5
    active_by_intent: dict[str, DurableHireState] = {}
    while time.monotonic() < deadline:
        active_by_intent = {
            intent_id: state
            for intent_id in (admitted.intent_id, second.intent_id)
            if (state := first.hire_service.get_state(intent_id)) is not None
            and state.phase is HirePhase.ACTIVE
        }
        if len(active_by_intent) == 2:
            break
        time.sleep(0.02)
    assert len(active_by_intent) == 2
    first.close()

    restarted_channels = _Channels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 7
    recovered_by_intent: dict[str, DurableHireState] = {}
    while time.monotonic() < deadline:
        recovered_by_intent = {
            intent_id: state
            for intent_id in (admitted.intent_id, second.intent_id)
            if (state := restarted.hire_service.get_state(intent_id)) is not None
            and state.phase is HirePhase.ACTIVE
            and state.channel_generation == 2
        }
        if len(recovered_by_intent) == 2:
            break
        time.sleep(0.05)

    assert len(recovered_by_intent) == 2
    assert sorted(item[3] for item in restarted_channels.started) == [2, 2]
    automatic_events = [
        event
        for frame in restarted.journal_frames()
        for event in frame.events
        if event.event_type == "hire.activation.automatic"
        and event.aggregate_id in {admitted.intent_id, second.intent_id}
    ]
    assert len(automatic_events) == 4
    restarted.close()


def test_production_factory_has_no_boolean_release_bypass() -> None:
    assert "release_evidence_ready" not in inspect.signature(EmployeeDepartmentRuntime.from_settings).parameters
    assert "release_trust_provider" in inspect.signature(EmployeeDepartmentRuntime.from_settings).parameters
    assert "resume_external" not in inspect.signature(EmployeeDepartmentRuntime.recover).parameters


def test_local_composition_does_not_require_release_provider(tmp_path: Path) -> None:
    settings = _settings(tmp_path, limit=1)
    settings.autonomous_data_keys = SecretStr(
        json.dumps({"version": 1, "keys": {"k1": _b64(b"d" * 32)}})
    )
    settings.autonomous_data_active_key_id = "k1"
    settings.autonomous_state_dir = str(tmp_path / "state")

    runtime = EmployeeDepartmentRuntime.from_settings(
        settings,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    try:
        assert runtime.hire_service is not None
        assert runtime.hire_readiness().ready is True
    finally:
        runtime.close()


def test_local_composition_canonicalizes_symlinked_home_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(linked_home))
    settings = _settings(tmp_path, limit=1)
    settings.autonomous_credential_dir = "~/.ghostap/credentials"
    settings.autonomous_data_blob_dir = "~/.ghostap/data-blobs"
    settings.autonomous_employee_ingress_blob_dir = "~/.ghostap/ingress-blobs"
    settings.autonomous_employee_outbox_blob_dir = "~/.ghostap/outbox-blobs"
    settings.autonomous_employee_attachment_staging_dir = "~/.ghostap/attachments"
    settings.autonomous_slock_storage_base = "~/.ghostap/slock"

    runtime = EmployeeDepartmentRuntime.from_settings(
        settings,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    try:
        assert runtime.hire_service is not None
        assert runtime.hire_readiness().ready is True
        assert runtime._data is not None
        assert runtime._ingress is not None
        assert runtime._outbox is not None
        assert runtime._attachments is not None
        assert (real_home / ".ghostap/credentials").is_dir()
        assert (real_home / ".ghostap/data-blobs").is_dir()
        assert (real_home / ".ghostap/ingress-blobs").is_dir()
        assert (real_home / ".ghostap/outbox-blobs").is_dir()
        assert (real_home / ".ghostap/attachments").is_dir()
    finally:
        runtime.close()


class _ExternalTrustProvider:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ExternalTrustSession:
    def __init__(self) -> None:
        self.closed = False
        self.expired = False
        self.anchors: dict[str, MemoryAnchor] = {}
        self.audit_sequence = 0

    def valid(self, _now: float) -> bool:
        return not self.expired

    def renew_if_needed(self, _now: float, *, renewal_window_seconds: float) -> bool:
        assert renewal_window_seconds == 120.0
        return not self.expired

    def close(self) -> None:
        self.closed = True

    def journal_anchor(self, scope: str):
        assert scope in {
            "employee-journal:ghostap-prod-a",
            "main-bot-audit:ghostap-prod-a",
        }
        anchor = self.anchors.setdefault(scope, MemoryAnchor())
        anchor.production_safe = True
        return anchor

    def record_main_bot_send_attempt(self, **_kwargs) -> None:
        self.audit_sequence += 1

    def count_main_bot_send_attempts(self, _tenant_key, _start, _end) -> int:
        return self.audit_sequence

    def count_main_bot_target_send_attempts(
        self,
        _tenant_key,
        _target_hash,
        _start,
        _end,
    ) -> int:
        return 0


def test_external_main_bot_audit_without_activation_fence_blocks_hire_readiness(
    tmp_path: Path,
) -> None:
    audit = MainBotSendAuditLog.open(
        tmp_path / "external-main-bot-audit",
        anchor_path=tmp_path / "external-main-bot-audit.anchor",
        hmac_key=b"a" * 32,
        external_audit=object(),
    )
    runtime = EmployeeDepartmentRuntime.from_settings(
        _settings(tmp_path, limit=1),
        main_bot_send_audit=audit.count_target_attempts,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    try:
        assert runtime.hire_service is not None
        assert runtime.hire_readiness().blockers == (
            "main_bot_activation_fence",
        )
    finally:
        runtime.close()
        audit.close()


def test_fenced_custom_main_bot_audit_is_shared_with_outbound_and_not_owned(
    tmp_path: Path,
) -> None:
    audit = MainBotSendAuditLog.open(
        tmp_path / "custom-main-bot-audit",
        anchor_path=tmp_path / "custom-main-bot-audit.anchor",
        hmac_key=b"a" * 32,
    )
    runtime = EmployeeDepartmentRuntime.from_settings(
        _settings(tmp_path, limit=1),
        main_bot_send_audit=audit.count_target_attempts,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    assert runtime.hire_readiness().ready is True
    assert runtime.main_bot_outbound_audit is audit
    runtime.main_bot_outbound_audit.record_attempt(
        "tenant-a",
        "reply",
        "om-requester",
        attempted_at=time.time(),
    )
    assert audit.count_target_attempts(
        "tenant-a",
        hashlib.sha256(b"om-requester").hexdigest(),
        0.0,
        time.time(),
    ) == 1

    runtime.close()
    assert audit.count_attempts("tenant-a", 0.0, time.time()) == 1
    audit.close()


def test_plain_main_bot_audit_callable_without_activation_fence_blocks_hire_readiness(
    tmp_path: Path,
) -> None:
    runtime = EmployeeDepartmentRuntime.from_settings(
        _settings(tmp_path, limit=1),
        main_bot_send_audit=lambda *_: 0,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    try:
        assert runtime.hire_service is not None
        assert runtime.hire_readiness().blockers == (
            "main_bot_activation_fence",
        )
    finally:
        runtime.close()


def test_legacy_release_provider_is_closed_and_not_required(tmp_path: Path) -> None:
    provider = _ExternalTrustProvider()
    session = _ExternalTrustSession()
    runtime = EmployeeDepartmentRuntime.from_settings(
        _settings(tmp_path, limit=1),
        release_trust_provider=provider,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    assert runtime.hire_service is not None
    assert runtime.hire_readiness().ready is True
    assert runtime.main_bot_outbound_audit is not None
    runtime.main_bot_outbound_audit.record_attempt(
        "tenant-a",
        "reply",
        "om-main-bot",
        attempted_at=time.time(),
    )
    assert session.audit_sequence == 0
    assert provider.closed is True
    runtime.close()
    assert session.closed is False


def test_legacy_release_session_expiry_cannot_close_local_admission(tmp_path: Path) -> None:
    provider = _ExternalTrustProvider()
    session = _ExternalTrustSession()
    runtime = EmployeeDepartmentRuntime.from_settings(
        _settings(tmp_path, limit=1, context_configured=True),
        release_trust_provider=provider,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        slock_engine_manager=_SlockManager(),
        employee_environment_provider=lambda authority: EmployeeProcessEnvironmentMaterial(
            authority.tenant_key,
            authority.agent_id,
            authority.employee_version,
            authority.credential_ref,
            {"PATH": "/usr/bin"},
            {},
        ),
    )
    session.expired = True

    assert runtime.hire_readiness().ready is True
    assert runtime.execution_readiness().ready is True
    assert runtime.hire_service is not None
    assert "admission_closed" not in runtime.hire_service.readiness().blockers
    assert provider.closed is True
    runtime.close()


def test_limit_zero_is_dormant_without_touching_durable_paths(tmp_path: Path) -> None:
    runtime = _runtime(
        _settings(tmp_path, limit=0),
        release_evidence_ready=True,
    )

    assert runtime.hire_service is None
    assert runtime.readiness().ready is False
    assert "visible_employee_limit" in runtime.readiness().blockers
    assert not (tmp_path / "journal").exists()
    assert not (tmp_path / "vault").exists()
    runtime.close()


def test_context_configuration_failure_does_not_block_first_hire(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    assert runtime.context_service is None
    assert runtime.hire_readiness().ready is True
    assert runtime.execution_readiness().blockers == ("employee_gateway",)
    assert runtime.hire_service is not None
    assert runtime.hire_service.start_hire(_request()).intent_id
    runtime.close()


def test_active_employee_requires_employee_scoped_context_probe_and_same_head(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _ContextSourceFactory()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    assert runtime.context_service is not None
    assert runtime.data_composition is not None
    assert runtime.data_composition.document_materializer.root == (tmp_path / "slock" / "agents")
    active = _activate_employee(runtime, channels)

    readiness = runtime.execution_readiness(active.agent_id)

    assert readiness.ready is True
    assert source_factory.probed == [(active.agent_id, active.app_id, active.credential_ref)]
    data_head = runtime.data_composition.service.get_head()
    workforce = runtime.hire_service.projection_state  # type: ignore[union-attr]
    assert (data_head.sequence, data_head.logical_hash) == (
        workforce.cursor_sequence,
        workforce.cursor_hash,
    )
    runtime.invalidate_employee_context(active.agent_id)
    assert source_factory.invalidated == [active.agent_id]
    runtime.reactivate_employee_context(active.agent_id)
    receipt = runtime.rewrap_employee_credential(
        agent_id=active.agent_id,
        app_id=active.app_id,
        credential_ref=active.credential_ref,
    )
    assert receipt.credential_ref == active.credential_ref
    assert source_factory.invalidated == [active.agent_id, active.agent_id]
    assert source_factory.reactivated == [active.agent_id, active.agent_id]
    runtime.close()
    assert source_factory.closed is True


def test_active_employee_probe_failure_blocks_execution_not_hiring(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _ContextSourceFactory(probe_ready=False)
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)

    assert runtime.hire_readiness().ready is True
    readiness = runtime.execution_readiness(active.agent_id)
    assert readiness.ready is False
    assert readiness.blockers == ("context_credentials",)
    runtime.close()


def test_group_history_permission_blocks_execution_but_not_team_admission(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _ContextSourceFactory(group_probe_ready=False)
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)
    assert runtime._writer is not None
    assert runtime.hire_service is not None
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )

    readiness = runtime.execution_readiness(active.agent_id)

    assert readiness.ready is False
    assert readiness.blockers == ("context_group_history",)
    assert source_factory.group_probed == [
        (active.agent_id, "oc_employee_team")
    ]
    assert runtime.team_service is not None
    assert len(runtime.team_service._backend.list_active(  # noqa: SLF001
        active.tenant_key,
        "oc_employee_team",
    )) == 1
    assert source_factory.group_probed == [
        (active.agent_id, "oc_employee_team")
    ]
    runtime.close()


def test_team_admission_uses_durable_membership_without_remote_group_probe(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _ContextSourceFactory(
        group_probe_ready={"oc_bad_team": False, "oc_good_team": True}
    )
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)
    assert runtime._writer is not None
    assert runtime.hire_service is not None
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_bad_team", "oc_good_team"]},
            ),
        ),
    )
    assert runtime.team_service is not None
    backend = runtime.team_service._backend  # noqa: SLF001

    assert len(backend.list_active(active.tenant_key, "oc_bad_team")) == 1
    assert len(backend.list_active(active.tenant_key, "oc_good_team")) == 1
    assert source_factory.group_probed == []
    runtime.close()


def test_execution_readiness_fails_closed_when_projection_sync_raises(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)
    assert runtime.hire_service is not None

    def fail_sync():
        raise RuntimeError("injected projection replay failure")

    runtime.hire_service.synchronize_projection = fail_sync  # type: ignore[method-assign]

    readiness = runtime.execution_readiness(active.agent_id)
    assert readiness.ready is False
    assert readiness.blockers == ("employee_context",)
    runtime.close()


def test_runtime_owned_context_service_assembles_authorized_snapshot(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _AssemblingContextSourceFactory()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)
    assert runtime._writer is not None
    assert runtime.hire_service is not None
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    assert runtime.context_service is not None
    snapshot = runtime.context_service.assemble(
        AuthorizedContextRequest(
            tenant_key=active.tenant_key,
            agent_id=active.agent_id,
            bot_principal_id=active.bot_principal_id,
            app_id=active.app_id,
            channel_generation=active.channel_generation,
            chat_id="oc_employee_team",
            thread_root_message_id="om_runtime_root",
            feishu_thread_id="omt_runtime",
            current_message_id="om_runtime_current",
            requester_principal_id="ou_admin",
        )
    )

    assert [item.text for item in snapshot.thread_messages] == [
        "root",
        "current",
    ]
    assert snapshot.thread_messages[-1].is_current is True
    runtime.close()


def test_recovery_invalidates_employee_context_after_durable_retirement(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _ContextSourceFactory()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)
    assert runtime._writer is not None
    assert runtime.hire_service is not None

    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.state_changed",
                aggregate_id=active.agent_id,
                payload={"state": HirePhase.ARCHIVED.value},
            ),
        ),
    )
    runtime.recover()

    assert source_factory.invalidated == [active.agent_id]
    assert runtime.execution_readiness(active.agent_id).blockers == ("employee_not_active",)
    runtime.close()


def test_explicit_context_invalidation_cannot_race_with_projection_reactivation(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _ContextSourceFactory()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)
    assert runtime.hire_service is not None
    source_factory.invalidate_employee(active.agent_id)
    runtime._context_projection_invalidations.add(active.agent_id)
    reactivate_entered = threading.Event()
    release_reactivate = threading.Event()
    explicit_done = threading.Event()
    refresh_done = threading.Event()
    original_reactivate = source_factory.reactivate_employee

    def blocking_reactivate(agent_id: str) -> None:
        reactivate_entered.set()
        assert release_reactivate.wait(2)
        original_reactivate(agent_id)

    source_factory.reactivate_employee = blocking_reactivate  # type: ignore[method-assign]
    refresher = threading.Thread(
        target=lambda: (
            runtime._refresh_context_bindings(
                runtime.hire_service.projection_state,
            ),
            refresh_done.set(),
        )
    )
    invalidator = threading.Thread(
        target=lambda: (
            runtime.invalidate_employee_context(active.agent_id),
            explicit_done.set(),
        )
    )

    refresher.start()
    assert reactivate_entered.wait(2)
    invalidator.start()
    assert not explicit_done.wait(0.05)
    release_reactivate.set()
    refresher.join(2)
    invalidator.join(2)

    assert not refresher.is_alive()
    assert not invalidator.is_alive()
    assert refresh_done.is_set()
    assert explicit_done.is_set()
    assert source_factory.reactivated == [active.agent_id]
    assert source_factory.invalidated == [active.agent_id, active.agent_id]
    assert active.agent_id in runtime._context_explicit_invalidations
    runtime.close()


def test_missing_requester_authority_blocks_before_employee_probe(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1, context_configured=True)
    settings.autonomous_manager_acl = ""
    settings.admin_user_ids = frozenset()
    channels = _Channels()
    source_factory = _ContextSourceFactory()
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)

    readiness = runtime.execution_readiness(active.agent_id)

    assert readiness.ready is False
    assert readiness.blockers == ("context_request_authority",)
    assert source_factory.probed == []
    runtime.close()


def test_runtime_closes_context_before_channels_data_writer_and_vault(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    source_factory = _ContextSourceFactory()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    assert runtime.context_service is not None
    assert runtime.data_composition is not None
    assert runtime.hire_service is not None
    events: list[str] = []

    runtime.context_service.stop_admission = lambda: events.append("context_admission")  # type: ignore[method-assign]
    runtime.context_service.drain = lambda: events.append("context_drain")  # type: ignore[method-assign]
    source_factory.close = lambda: events.append("context_sources")  # type: ignore[method-assign]

    channel_attempts = 0

    def failing_channels_close():
        nonlocal channel_attempts
        channel_attempts += 1
        events.append("channels")
        if channel_attempts == 1:
            raise RuntimeError("injected close failure")

    channels.close = failing_channels_close  # type: ignore[method-assign]
    original_data_close = runtime.data_composition.close
    runtime.data_composition.close = lambda: (events.append("data"), original_data_close())[-1]  # type: ignore[method-assign]
    original_service_close = runtime.hire_service.close
    runtime.hire_service.close = lambda: (events.append("writer"), original_service_close())[-1]  # type: ignore[method-assign]
    original_vault_close = runtime._vault.close
    runtime._vault.close = lambda: (events.append("vault"), original_vault_close())[-1]  # type: ignore[method-assign,union-attr]

    runtime.close()

    assert events == [
        "context_admission",
        "context_drain",
        "context_sources",
        "channels",
    ]
    runtime.close()
    assert events == [
        "context_admission",
        "context_drain",
        "context_sources",
        "channels",
        "context_admission",
        "context_drain",
        "context_sources",
        "channels",
        "data",
        "writer",
        "vault",
    ]


def test_hire_and_data_writes_resynchronize_shared_journal_heads(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=2, context_configured=True),
        release_evidence_ready=True,
        registrar=_DistinctRegistrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )
    active = _activate_employee(runtime, channels)
    assert runtime.data_composition is not None
    assert runtime.hire_service is not None
    employee = runtime.hire_service.projection_state.employees[active.agent_id]

    runtime.data_composition.publish_document(
        PublishEmployeeDocumentCommand(
            agent_id=active.agent_id,
            tenant_key=active.tenant_key,
            owner_principal_id=employee.owner_principal_id,
            kind=DataKind.L1_MEMORY,
            source_id="l1_memory",
            content=b"safe memory",
            content_type="text/markdown",
        )
    )
    second_request = EmployeeHireRequest(
        employee_name="Vega",
        tool="codex",
        model="gpt-5.6-sol",
        effort="high",
        chat_id="oc_admin_dm",
        message_id="om_composition_second",
        requester_principal_id="ou_admin",
        requester_union_id="on_admin",
        tenant_key="tenant-a",
    )
    attempt = ExecutionAttemptContext(
        tenant_key=active.tenant_key,
        agent_id=active.agent_id,
        owner_principal_id=employee.owner_principal_id,
        requester_principal_id="ou_admin",
        task_id="task_projection_sync",
        run_id="run_projection_sync",
        attempt_id="attempt_projection_sync",
        message_id="om_projection_sync",
        thread_root_id="om_projection_sync",
        chat_id=active.chat_id,
        tool=active.tool,
        model=active.model,
        effort=active.effort,
        started_at="2026-07-13T00:00:00+00:00",
        terminal_epoch=1,
    )
    barrier = threading.Barrier(3)
    errors: list[Exception] = []
    hired: list[DurableHireState] = []

    def start_second_hire() -> None:
        barrier.wait()
        try:
            hired.append(runtime.hire_service.start_hire(second_request))
        except Exception as exc:
            errors.append(exc)

    def start_data_attempt() -> None:
        barrier.wait()
        try:
            runtime.data_composition.service.start_attempt(attempt)
        except Exception as exc:
            errors.append(exc)

    workers = [
        threading.Thread(target=start_second_hire),
        threading.Thread(target=start_data_attempt),
    ]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(5)
    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert len(hired) == 1
    second = hired[0]
    assert second.agent_id != active.agent_id
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        second = runtime.hire_service.get_state(second.intent_id)
        if second is not None and second.phase in {
            HirePhase.READY_PENDING_VERIFICATION,
            HirePhase.ACTION_REQUIRED,
        }:
            break
        time.sleep(0.02)
    assert second is not None
    assert second.phase is HirePhase.READY_PENDING_VERIFICATION
    runtime.hire_service.commit_effect_transition(
        active.intent_id,
        effect_id="projection-sync-check",
        effect_type="projection_sync_check",
        next_state=HireEffectState.PREPARED,
    )
    runtime.hire_service.recover()
    runtime.data_composition.service.rebuild_projection()
    data_head = runtime.data_composition.service.get_head()
    workforce = runtime.hire_service.projection_state
    assert (data_head.sequence, data_head.logical_hash) == (
        workforce.cursor_sequence,
        workforce.cursor_hash,
    )
    runtime.close()


def test_source_close_failure_holds_writer_and_vault_until_retry(
    tmp_path: Path,
) -> None:
    source_factory = _ContextSourceFactory()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=source_factory,
        group_memory_backend=_GroupMemory(),
    )
    assert runtime.hire_service is not None
    events: list[str] = []
    original_service_close = runtime.hire_service.close
    runtime.hire_service.close = lambda: (events.append("writer"), original_service_close())[-1]  # type: ignore[method-assign]
    original_vault_close = runtime._vault.close
    runtime._vault.close = lambda: (events.append("vault"), original_vault_close())[-1]  # type: ignore[method-assign,union-attr]
    source_factory.close = lambda: (_ for _ in ()).throw(RuntimeError("blocked"))  # type: ignore[method-assign]

    runtime.close()

    assert events == []
    source_factory.close = lambda: events.append("sources")  # type: ignore[method-assign]
    runtime.close()
    assert events == ["sources", "writer", "vault"]


def test_context_binding_and_probe_recover_after_restart_reverification(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1, context_configured=True)
    settings.autonomous_thread_context_max_chars = 22
    first_channels = _Channels()
    first_source = _ReplayContextSourceFactory(split_pages=True)
    group_memory = _StableGroupMemory("l2-memory")
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=first_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=first_source,
        group_memory_backend=group_memory,
    )
    active = _activate_employee(first, first_channels)
    assert first._writer is not None
    assert first.hire_service is not None
    commit_workforce_events(
        first._writer,
        first.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    assert first.data_composition is not None
    employee = first.hire_service.synchronize_projection().employees[active.agent_id]
    first.data_composition.publish_document(
        PublishEmployeeDocumentCommand(
            agent_id=active.agent_id,
            tenant_key=active.tenant_key,
            owner_principal_id=employee.owner_principal_id,
            kind=DataKind.L1_MEMORY,
            source_id="l1_memory",
            content=b"l1-memory",
            content_type="text/markdown",
        )
    )
    assert (
        first.data_composition.memory_facade.read_l1(
            active.agent_id,
            active.tenant_key,
            allow_unscoped_legacy=False,
        )
        == "l1-memory"
    )
    context_request = AuthorizedContextRequest(
        tenant_key=active.tenant_key,
        agent_id=active.agent_id,
        bot_principal_id=active.bot_principal_id,
        app_id=active.app_id,
        channel_generation=active.channel_generation,
        chat_id="oc_employee_team",
        thread_root_message_id="om_runtime_root",
        feishu_thread_id="omt_runtime",
        current_message_id="om_runtime_current",
        requester_principal_id="ou_admin",
    )
    assert first.context_service is not None
    before_restart = first.context_service.assemble(context_request)
    first.close()

    restarted_channels = _Channels()
    restarted_source = _ReplayContextSourceFactory(split_pages=False)
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=restarted_source,
        group_memory_backend=_StableGroupMemory("l2-memory"),
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 7
    pending = None
    while time.monotonic() < deadline:
        pending = restarted.hire_service.get_state(active.intent_id)
        if (
            pending is not None
            and pending.phase is HirePhase.READY_PENDING_VERIFICATION
            and pending.channel_generation == 2
        ):
            break
        time.sleep(0.05)
    assert pending is not None
    ack = _accept_durable_status(
        restarted,
        pending,
        suffix="context_restart",
        generation=pending.channel_generation,
        connection_id=pending.channel_connection_id,
    )
    restarted._handle_control_ingress(ack.acceptance.acceptance_id)
    recovered = None
    while time.monotonic() < deadline:
        recovered = restarted.hire_service.get_state(active.intent_id)
        if recovered is not None and recovered.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)
    assert recovered is not None
    assert restarted.execution_readiness(recovered.agent_id).ready is True
    assert restarted_source.probed == [(recovered.agent_id, recovered.app_id, recovered.credential_ref)]
    assert restarted.data_composition is not None
    data_head = restarted.data_composition.service.get_head()
    projection = restarted.hire_service.projection_state
    assert (data_head.sequence, data_head.logical_hash) == (
        projection.cursor_sequence,
        projection.cursor_hash,
    )
    assert restarted.context_service is not None
    after_restart = restarted.context_service.assemble(
        replace(
            context_request,
            channel_generation=recovered.channel_generation,
        )
    )
    assert after_restart.snapshot_hash == before_restart.snapshot_hash
    assert after_restart.watermark == before_restart.watermark
    assert after_restart.thread_messages == before_restart.thread_messages
    assert after_restart.group_messages == before_restart.group_messages
    assert after_restart.layer_metrics == before_restart.layer_metrics
    assert after_restart.trimming_trace == before_restart.trimming_trace
    assert [record.layer for record in after_restart.trimming_trace] == [
        ContextLayer.L2_GROUP,
        ContextLayer.L1_MEMORY,
        ContextLayer.GROUP_RECENT,
    ]
    assert after_restart.l1_summary == ""
    assert after_restart.l2_summary == ""
    assert after_restart.group_messages == ()
    deleted = next(message for message in after_restart.thread_messages if message.message_id == "om_runtime_deleted")
    assert deleted.deleted is True
    assert deleted.text == ""
    assert after_restart.thread_messages[0].edited is True
    assert after_restart.thread_messages[0].text == "root-edited"
    restarted.close()


def test_missing_release_evidence_does_not_block_local_runtime(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=False,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )

    assert runtime.hire_service is not None
    assert runtime.readiness().ready is True
    assert (tmp_path / "journal").exists()
    runtime.close()


def test_existing_journal_recovers_without_release_state(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1, context_configured=True)
    initial = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )
    initial.close()

    restarted = _runtime(
        settings,
        release_evidence_ready=False,
        notification_link=lambda *_: None,
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
    )

    assert restarted.hire_service is not None
    assert restarted.hire_readiness().ready is True
    assert restarted.execution_readiness().ready is True
    restarted.close()


def test_release_claim_settings_are_not_required_for_local_runtime(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    settings.autonomous_employee_release_id = ""
    settings.autonomous_employee_commit_sha = ""
    settings.autonomous_employee_service_instance_id = ""
    settings.autonomous_employee_staging_tenant_hash = ""
    settings.autonomous_employee_production_tenant_hash = ""
    settings.autonomous_employee_release_evidence_bundle = str(tmp_path / "missing-evidence.jsonl")
    settings.autonomous_employee_release_checkpoint = str(tmp_path / "missing-checkpoint.json")

    runtime = EmployeeDepartmentRuntime.from_settings(
        settings,
        notification_link=lambda *_: None,
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
    )

    assert runtime.hire_service is not None
    assert runtime.readiness().ready is True
    assert (tmp_path / "journal").exists()
    runtime.close()


def test_local_main_bot_audit_is_composed_automatically(
    tmp_path: Path,
) -> None:
    runtime = EmployeeDepartmentRuntime.from_settings(
        _settings(tmp_path, limit=1),
        notification_link=lambda *_: None,
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
    )

    assert runtime.hire_service is not None
    assert runtime.main_bot_outbound_audit is not None
    assert runtime.readiness().ready is True
    runtime.close()


def test_ready_runtime_wires_saga_slash_channel_and_durable_challenge(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None

    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        if state is not None and state.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)

    assert state is not None
    assert state.phase is HirePhase.READY_PENDING_VERIFICATION
    assert state.slash_spec_hash == state.slash_observed_hash == "slash_hash"
    assert state.channel_generation == 1
    assert state.channel_connection_id == "conn_runtime"
    assert state.verification_nonce
    assert channels.started == [(state.agent_id, state.app_id, state.credential_ref, 1)]
    event_types = [event.event_type for frame in runtime.journal_frames() for event in frame.events]
    for effect in ("slash-reconcile:1:1", "channel-start:1"):
        positions = [
            index
            for index, frame in enumerate(runtime.journal_frames())
            if frame.events[0].payload.get("effect_id") == effect
        ]
        assert len(positions) == 3
        assert positions == sorted(positions)
    assert event_types[-1] == "employee.state_changed"

    runtime.close()
    assert channels.closed is True


def test_failed_channel_start_is_disposed_and_retried_with_next_generation(
    tmp_path: Path,
) -> None:
    channels = _FailOnceStartChannels()
    statuses: list[str] = []
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        notification_status=lambda _state, status: (
            statuses.append(status) or "om_notification"
        ),
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        if (
            state is not None
            and state.phase is HirePhase.READY_PENDING_VERIFICATION
            and statuses == ["ready"]
        ):
            break
        time.sleep(0.02)

    assert state is not None
    assert state.phase is HirePhase.READY_PENDING_VERIFICATION
    assert state.channel_generation == 2
    assert channels.attempted_generations == [1, 2]
    assert state.effect_state("channel-start:1") is HireEffectState.ACTION_REQUIRED
    assert state.effect_state("channel-start:2") is HireEffectState.COMMITTED
    assert statuses == ["ready"]
    runtime.close()


def test_channel_action_required_effect_emits_terminal_notification(
    tmp_path: Path,
) -> None:
    statuses: list[str] = []
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_FailingStartChannels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        notification_status=lambda _state, status: (
            statuses.append(status) or "om_action_required"
        ),
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        events = [
            event.event_type
            for frame in runtime.journal_frames()
            for event in frame.events
            if event.aggregate_id
            == f"hire-notification:{admitted.intent_id}:action_required"
        ]
        if statuses == ["action_required"] and "hire.notification.committed" in events:
            break
        time.sleep(0.05)

    assert state is not None
    assert state.phase is HirePhase.CONFIGURING
    assert any(
        effect_state is HireEffectState.ACTION_REQUIRED
        for _effect_id, effect_state in state.effects
    )
    assert statuses == ["action_required"]
    assert events.count("hire.notification.committed") == 1
    runtime.close()


def test_duplicate_hire_resubmission_reuses_one_configuration_activity(
    tmp_path: Path,
) -> None:
    slash = _BlockingSlash()
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: slash,
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    request = _request()
    admitted = runtime.hire_service.start_hire(request)
    assert slash.entered.wait(5)

    for _ in range(5):
        duplicate = runtime.hire_service.start_hire(request)
        assert duplicate.intent_id == admitted.intent_id
    slash.release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        if state is not None and state.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)

    assert state is not None
    assert state.phase is HirePhase.READY_PENDING_VERIFICATION
    assert slash.calls == 1
    assert len(channels.started) == 1
    runtime.close()


def test_duplicate_hire_during_recovery_does_not_start_second_configuration(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    request = _request()
    admitted = first.hire_service.start_hire(request)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    first.close()

    slash = _BlockingSlash()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: slash,
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    assert slash.entered.wait(5)

    duplicate = restarted.hire_service.start_hire(request)
    assert duplicate.intent_id == admitted.intent_id
    slash.release.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        recovered = restarted.hire_service.get_state(admitted.intent_id)
        if (
            recovered is not None
            and recovered.phase is HirePhase.READY_PENDING_VERIFICATION
            and recovered.channel_generation == 2
        ):
            break
        time.sleep(0.02)

    assert recovered is not None
    assert recovered.phase is HirePhase.READY_PENDING_VERIFICATION
    assert slash.calls == 1
    restarted.close()


def test_transient_recovery_failure_is_supervised_and_converges_in_process(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    first.close()

    slash = _FailOnceSlash(failures=4)
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: slash,
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        recovered = restarted.hire_service.get_state(admitted.intent_id)
        if (
            recovered is not None
            and recovered.phase is HirePhase.READY_PENDING_VERIFICATION
            and recovered.channel_generation == 2
            and restarted.readiness().ready is True
        ):
            break
        time.sleep(0.05)

    assert recovered is not None
    assert recovered.phase is HirePhase.READY_PENDING_VERIFICATION
    assert slash.calls == 5
    assert restarted.readiness().ready is True
    restarted.close()


def test_exhausted_hire_recovery_is_isolated_as_action_required(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _FailingSlash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id)
        if (
            pending is not None
            and pending.phase is HirePhase.CONFIGURING
            and pending.effect_state("slash-reconcile:1:1") is HireEffectState.EXECUTING
        ):
            break
        time.sleep(0.02)
    assert pending is not None
    first.close()

    slash = _FailOnceSlash(failures=100)
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: slash,
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        recovered = restarted.hire_service.get_state(admitted.intent_id)
        if (
            recovered is not None
            and recovered.phase is HirePhase.ACTION_REQUIRED
            and restarted.readiness().ready is True
        ):
            break
        time.sleep(0.05)

    assert recovered is not None
    assert recovered.phase is HirePhase.ACTION_REQUIRED
    assert recovered.effect_state("slash-reconcile:1:1") is HireEffectState.ACTION_REQUIRED
    assert recovered.metadata_for("slash-reconcile:1:1")["error_code"] == "recovery_exhausted"
    assert slash.calls == 6
    assert restarted.readiness().ready is True
    restarted.close()


def test_new_hire_reports_ready_terminal_status(tmp_path: Path) -> None:
    statuses: list[str] = []
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        notification_status=lambda _state, status: (
            statuses.append(status) or "om_notification"
        ),
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        if (
            state is not None
            and state.phase is HirePhase.READY_PENDING_VERIFICATION
            and statuses == ["ready"]
        ):
            break
        time.sleep(0.02)

    assert state is not None and state.phase is HirePhase.READY_PENDING_VERIFICATION
    assert statuses == ["ready"]
    notification_events = [
        event.event_type
        for frame in runtime.journal_frames()
        for event in frame.events
        if event.aggregate_id.endswith(":ready")
    ]
    assert notification_events == [
        "hire.notification.prepared",
        "hire.notification.executing",
        "hire.notification.committed",
    ]
    runtime.close()


def test_ready_notification_retries_automatically_after_unacknowledged_reply(
    tmp_path: Path,
) -> None:
    attempts: list[str] = []

    def notify(_state, status: str):
        if status != "ready":
            return "om_other_notification"
        attempts.append(status)
        return None if len(attempts) == 1 else "om_ready_notification"

    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        notification_status=notify,
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        events = [
            event.event_type
            for frame in runtime.journal_frames()
            for event in frame.events
            if event.aggregate_id
            == f"hire-notification:{admitted.intent_id}:ready"
        ]
        if attempts == ["ready", "ready"] and "hire.notification.committed" in events:
            break
        time.sleep(0.05)

    assert state is not None
    assert state.phase is HirePhase.READY_PENDING_VERIFICATION
    assert attempts == ["ready", "ready"]
    assert events.count("hire.notification.action_required") == 1
    assert events.count("hire.notification.committed") == 1
    runtime.close()


def test_notification_lock_rechecks_phase_before_sending() -> None:
    runtime = EmployeeDepartmentRuntime()
    ready = DurableHireState(intent_id="hire_1", phase=HirePhase.READY_PENDING_VERIFICATION)
    holder = SimpleNamespace(current=ready)
    runtime._service = SimpleNamespace(
        get_state=lambda _intent_id: holder.current,
    )
    calls: list[str] = []
    runtime._notification_status = lambda _state, status: calls.append(status) or "om_1"

    async def race() -> bool:
        runtime._notification_async_lock = asyncio.Lock()
        await runtime._notification_async_lock.acquire()
        task = asyncio.create_task(runtime._notify_hire_terminal(ready, "ready"))
        await asyncio.sleep(0)
        holder.current = replace(ready, phase=HirePhase.ACTIVE)
        runtime._notification_async_lock.release()
        return await task

    assert asyncio.run(race()) is False
    assert calls == []


def test_channel_monitor_revalidates_stalled_reconnect(monkeypatch) -> None:
    runtime = EmployeeDepartmentRuntime()
    state = DurableHireState(
        intent_id="hire_reconnecting",
        agent_id="agt_reconnecting",
        phase=HirePhase.ACTIVE,
        channel_generation=7,
    )
    revalidations: list[tuple[str, int]] = []
    submissions: list[str] = []

    class Service:
        @staticmethod
        def list_states():
            return (state,)

        @staticmethod
        def begin_channel_revalidation(intent_id: str, *, observed_generation: int):
            revalidations.append((intent_id, observed_generation))

    runtime._service = Service()
    runtime._channels = SimpleNamespace(
        status=lambda _agent_id: SimpleNamespace(
            agent_id=state.agent_id,
            generation=state.channel_generation,
            state=ChannelProcessState.STARTING,
            error_code="channel-reconnecting",
            ready_metadata={"reconnecting_at": time.time() - 181.0},
        )
    )
    runtime._submit_intent = submissions.append  # type: ignore[method-assign]

    async def stop_monitor(_seconds: float) -> None:
        runtime._closing = True

    monkeypatch.setattr(asyncio, "sleep", stop_monitor)
    asyncio.run(runtime._monitor_channels())

    assert revalidations == [(state.intent_id, state.channel_generation)]
    assert submissions == [state.intent_id]


def test_channel_monitor_stops_before_revalidation_when_close_starts_mid_iteration(
    monkeypatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()
    state = DurableHireState(
        intent_id="hire_shutdown_race",
        agent_id="agt_shutdown_race",
        phase=HirePhase.ACTIVE,
        channel_generation=7,
    )
    revalidations: list[tuple[str, int]] = []
    submissions: list[str] = []

    class Service:
        @staticmethod
        def list_states():
            return (state,)

        @staticmethod
        def begin_channel_revalidation(intent_id: str, *, observed_generation: int):
            revalidations.append((intent_id, observed_generation))

    def close_during_status(_agent_id: str):
        runtime._closing = True
        return SimpleNamespace(
            agent_id=state.agent_id,
            generation=state.channel_generation,
            state=ChannelProcessState.CRASHED,
            error_code="channel-crashed",
            ready_metadata={},
        )

    runtime._service = Service()
    runtime._channels = SimpleNamespace(status=close_during_status)
    runtime._submit_intent = submissions.append  # type: ignore[method-assign]

    async def stop_monitor(_seconds: float) -> None:
        runtime._closing = True

    monkeypatch.setattr(asyncio, "sleep", stop_monitor)
    asyncio.run(runtime._monitor_channels())

    assert revalidations == []
    assert submissions == []


def test_team_notification_internal_type_error_is_not_retried_without_key() -> None:
    attempts: list[str] = []

    def notify(
        _message_id: str,
        _chat_id: str,
        _result: str,
        idempotency_key: str,
    ) -> None:
        attempts.append(idempotency_key)
        raise TypeError("internal serialization failed")

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)

    with pytest.raises(TypeError, match="internal serialization failed"):
        backend.notify(
            "om_1",
            "oc_team",
            "result",
            idempotency_key="stable-key",
        )

    assert attempts == ["stable-key"]


def test_team_notification_forwards_durable_recipient_scope() -> None:
    notifications: list[tuple[str, str, str, str, str, str]] = []

    def notify(
        message_id: str,
        chat_id: str,
        result: str,
        *,
        idempotency_key: str,
        tenant_key: str,
        requester_principal_id: str,
    ) -> None:
        notifications.append(
            (
                message_id,
                chat_id,
                result,
                idempotency_key,
                tenant_key,
                requester_principal_id,
            )
        )

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert notifications == [
        (
            "om_1",
            "oc_team",
            "result",
            "stable-key",
            "tenant-a",
            "ou_requester",
        )
    ]


def test_team_notification_retries_scope_without_optional_idempotency_key() -> None:
    notifications: list[tuple[str, str]] = []

    def notify(
        _message_id: str,
        _chat_id: str,
        _result: str,
        *,
        tenant_key: str,
        requester_principal_id: str,
    ) -> None:
        notifications.append((tenant_key, requester_principal_id))

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert notifications == [("tenant-a", "ou_requester")]


def test_team_notification_supports_legacy_fourth_key_with_scoped_kwargs() -> None:
    notifications: list[tuple[str, str, str]] = []

    def notify(
        _message_id: str,
        _chat_id: str,
        _result: str,
        stable_uuid: str,
        *,
        tenant_key: str,
        requester_principal_id: str,
    ) -> None:
        notifications.append(
            (stable_uuid, tenant_key, requester_principal_id)
        )

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert notifications == [
        ("stable-key", "tenant-a", "ou_requester")
    ]


def test_team_notification_prefers_legacy_fourth_key_over_catch_all_kwargs() -> None:
    notifications: list[tuple[str, str, str]] = []

    def notify(
        _message_id: str,
        _chat_id: str,
        _result: str,
        stable_uuid: str = "",
        **kwargs: str,
    ) -> None:
        notifications.append(
            (
                stable_uuid,
                kwargs["tenant_key"],
                kwargs["requester_principal_id"],
            )
        )

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert notifications == [
        ("stable-key", "tenant-a", "ou_requester")
    ]


def test_team_notification_prefers_positional_only_key_over_catch_all_kwargs() -> None:
    notifications: list[tuple[str, dict[str, str]]] = []

    def notify(
        _message_id: str,
        _chat_id: str,
        _result: str,
        idempotency_key: str = "",
        /,
        **kwargs: str,
    ) -> None:
        notifications.append((idempotency_key, kwargs))

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert notifications == [
        (
            "stable-key",
            {
                "tenant_key": "tenant-a",
                "requester_principal_id": "ou_requester",
            },
        )
    ]


def test_team_notification_prefers_all_positional_only_scope_over_kwargs() -> None:
    notifications: list[tuple[str, str, str, dict[str, str]]] = []

    def notify(
        _message_id: str,
        _chat_id: str,
        _result: str,
        idempotency_key: str = "",
        tenant_key: str = "",
        requester_principal_id: str = "",
        /,
        **kwargs: str,
    ) -> None:
        notifications.append(
            (
                idempotency_key,
                tenant_key,
                requester_principal_id,
                kwargs,
            )
        )

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert notifications == [
        ("stable-key", "tenant-a", "ou_requester", {})
    ]


def test_team_notification_preserves_legacy_fourth_positional_parameter() -> None:
    attempts: list[str] = []

    def notify(_message_id, _chat_id, _result, stable_uuid) -> None:
        attempts.append(stable_uuid)

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert attempts == ["stable-key"]


def test_team_notification_supports_positional_only_recipient_scope() -> None:
    attempts: list[tuple[str, str, str]] = []

    def notify(
        _message_id,
        _chat_id,
        _result,
        stable_uuid,
        tenant,
        requester,
        /,
    ) -> None:
        attempts.append((stable_uuid, tenant, requester))

    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert attempts == [("stable-key", "tenant-a", "ou_requester")]


def test_team_notification_preserves_three_argument_callback() -> None:
    attempts: list[tuple[str, str, str]] = []
    backend = _RuntimeTeamBackend(
        SimpleNamespace(),
        lambda message_id, chat_id, result: attempts.append(
            (message_id, chat_id, result)
        ),
    )

    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert attempts == [("om_1", "oc_team", "result")]


def test_team_notification_preserves_opaque_callback_four_arg_shape() -> None:
    class _OpaqueNotify:
        @property
        def __signature__(self):
            raise ValueError("opaque")

        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def __call__(self, *args: object) -> None:
            self.calls.append(args)

    notify = _OpaqueNotify()
    backend = _RuntimeTeamBackend(SimpleNamespace(), notify)
    backend.notify(
        "om_1",
        "oc_team",
        "result",
        idempotency_key="stable-key",
        tenant_key="tenant-a",
        requester_principal_id="ou_requester",
    )

    assert notify.calls == [("om_1", "oc_team", "result", "stable-key")]


@pytest.mark.parametrize(
    ("transport_success", "expected_effect_state"),
    (
        (True, "committed"),
        (False, "action_required"),
    ),
)
def test_recovered_team_notification_crosses_main_bot_audit_boundary_once(
    tmp_path: Path,
    transport_success: bool,
    expected_effect_state: str,
) -> None:
    from src.autonomous.team import (
        TeamAttemptResult,
        TeamCoordinatorActor,
        TeamRunPhase,
    )
    from src.feishu.handlers.base import BaseHandler
    from src.feishu.ws_client import FeishuWSClient
    from src.project.mapper import MessageLinker
    from src.thread.manager import get_current_tenant_key
    from tests.autonomous.team_helpers import (
        ImmediateTeamBackend,
        make_team_storage,
    )

    class _InitialFailureBackend(ImmediateTeamBackend):
        def submit(self, **kwargs):
            acceptance_id = super().submit(**kwargs)
            self.results[acceptance_id] = TeamAttemptResult(
                "action_required",
                error_code="employee_session_failed",
                retry_allowed=False,
            )
            return acceptance_id

        def notify(
            self,
            _message_id,
            _chat_id,
            _content,
            *,
            idempotency_key,
        ) -> None:
            assert idempotency_key
            raise RuntimeError("initial notification transport unavailable")

    writer, blobs = make_team_storage(tmp_path / "team")
    first_actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=_InitialFailureBackend(),
        poll_seconds=0.001,
    )
    run = first_actor.start_task(
        tenant_key="tenant_1",
        message_id="om_recovered_team",
        chat_id="oc_team",
        requester_principal_id="ou_requester",
        task="恢复后向请求者汇报",
    )
    first_actor.drain()
    first_projection = first_actor.projection()
    assert first_projection.runs[run.run_id].phase is TeamRunPhase.BLOCKING
    assert (
        first_projection.effects[
            (f"{run.run_id}:blocked-notify:1", "notify")
        ]
        == "action_required"
    )
    first_actor.close()

    audit = MainBotSendAuditLog.open(
        tmp_path / "main-bot-audit",
        anchor_path=tmp_path / "main-bot-audit.anchor",
        hmac_key=b"a" * 32,
    )
    linker = MessageLinker()
    transport_calls: list[object] = []
    transport_tenants: list[str | None] = []

    def reply(request):
        transport_calls.append(request)
        transport_tenants.append(get_current_tenant_key())
        return SimpleNamespace(
            success=lambda: transport_success,
            code=0 if transport_success else 230001,
            msg="" if transport_success else "reply rejected",
            data=(
                SimpleNamespace(message_id="om_team_result")
                if transport_success
                else None
            ),
        )

    api_client = SimpleNamespace(
        im=SimpleNamespace(
            v1=SimpleNamespace(
                message=SimpleNamespace(reply=reply),
            )
        )
    )
    handler = BaseHandler(
        SimpleNamespace(
            api_client_factory=lambda: api_client,
            settings=SimpleNamespace(
                default_reply_mode="message",
                im_api_max_retries=1,
                ref_note_enabled=False,
            ),
            main_bot_outbound_audit=audit.record_attempt,
            main_bot_outbound_audit_failure=audit.mark_incomplete,
            tenant_key_resolver=get_current_tenant_key,
            message_linker=linker,
        )
    )
    client = FeishuWSClient.__new__(FeishuWSClient)
    client._message_linker = linker
    client._get_chat_mode = lambda _chat_id: "group"
    client._reply_text = handler.reply_text
    recovered_backend = _RuntimeTeamBackend(
        SimpleNamespace(),
        client._reply_employee_team_message,
    )
    reopened_actor = TeamCoordinatorActor(
        writer=writer,
        blob_store=blobs,
        active_key_id="team-key",
        backend=recovered_backend,
        poll_seconds=0.001,
    )

    try:
        assert reopened_actor.recover() == 1
        reopened_actor.drain()

        projection = reopened_actor.projection()
        assert projection.runs[run.run_id].phase is (
            TeamRunPhase.BLOCKED
            if transport_success
            else TeamRunPhase.BLOCKING
        )
        assert (
            projection.effects[
                (f"{run.run_id}:blocked-notify:2", "notify")
            ]
            == expected_effect_state
        )
        assert len(transport_calls) == 1
        assert transport_tenants == ["tenant_1"]
        assert get_current_tenant_key() is None

        origin = linker.query("om_recovered_team")
        assert origin is not None
        assert origin["chat_id"] == "oc_team"
        assert origin["sender_id"] == "ou_requester"
        assert origin["tenant_key"] == "tenant_1"

        upper_bound = time.time() + 1
        assert audit.count_target_attempts(
            "tenant_1",
            hashlib.sha256(b"om_recovered_team").hexdigest(),
            0,
            upper_bound,
        ) == 1
        assert audit.count_target_attempts(
            "tenant_1",
            hashlib.sha256(b"oc_team").hexdigest(),
            0,
            upper_bound,
        ) == 1
        if not transport_success:
            assert all(
                state != "committed"
                for (aggregate, effect_type), state in projection.effects.items()
                if aggregate.startswith(f"{run.run_id}:blocked-notify:")
                and effect_type == "notify"
            )
    finally:
        reopened_actor.close()
        audit.close()
        blobs.close()
        writer.close()


def test_channel_monitor_stops_before_submit_when_close_starts_during_revalidation(
    monkeypatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()
    state = DurableHireState(
        intent_id="hire_shutdown_submit_race",
        agent_id="agt_shutdown_submit_race",
        phase=HirePhase.ACTIVE,
        channel_generation=8,
    )
    revalidations: list[tuple[str, int]] = []
    submissions: list[str] = []

    class Service:
        @staticmethod
        def list_states():
            return (state,)

        @staticmethod
        def begin_channel_revalidation(intent_id: str, *, observed_generation: int):
            revalidations.append((intent_id, observed_generation))
            runtime._closing = True

    runtime._service = Service()
    runtime._channels = SimpleNamespace(
        status=lambda _agent_id: SimpleNamespace(
            agent_id=state.agent_id,
            generation=state.channel_generation,
            state=ChannelProcessState.CRASHED,
            error_code="channel-crashed",
            ready_metadata={},
        )
    )
    runtime._submit_intent = submissions.append  # type: ignore[method-assign]

    async def stop_monitor(_seconds: float) -> None:
        runtime._closing = True

    monkeypatch.setattr(asyncio, "sleep", stop_monitor)
    asyncio.run(runtime._monitor_channels())

    assert revalidations == [(state.intent_id, state.channel_generation)]
    assert submissions == []


def test_channel_monitor_stops_before_challenge_renewal_when_close_starts(
    monkeypatch,
) -> None:
    runtime = EmployeeDepartmentRuntime()
    state = DurableHireState(
        intent_id="hire_shutdown_renewal_race",
        agent_id="agt_shutdown_renewal_race",
        phase=HirePhase.READY_PENDING_VERIFICATION,
        channel_generation=9,
        verification_expires_at=1.0,
    )
    renewals: list[str] = []
    runtime._service = SimpleNamespace(list_states=lambda: (state,))
    runtime._channels = SimpleNamespace(
        status=lambda _agent_id: SimpleNamespace(
            agent_id=state.agent_id,
            generation=state.channel_generation,
            state=ChannelProcessState.READY,
            error_code="",
            ready_metadata={},
        )
    )
    runtime._renew_activation_challenge = (  # type: ignore[method-assign]
        lambda current: renewals.append(current.intent_id)
    )

    def close_during_expiry_check() -> float:
        runtime._closing = True
        return 2.0

    async def stop_monitor(_seconds: float) -> None:
        runtime._closing = True

    monkeypatch.setattr(time, "time", close_during_expiry_check)
    monkeypatch.setattr(asyncio, "sleep", stop_monitor)
    asyncio.run(runtime._monitor_channels())

    assert renewals == []


def test_recovery_wires_guarded_reopen_to_fresh_generation_configuration() -> None:
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    state = DurableHireState(
        intent_id="hire_guarded_reopen",
        agent_id="agt_guarded_reopen",
        phase=HirePhase.VALIDATING,
        credential_ref="vault://guarded-reopen",
        channel_generation=11,
    )
    revalidations: list[tuple[str, int]] = []
    configurations: list[tuple[str, bool, int]] = []
    submissions: list[str] = []

    class Service:
        projection_state = SimpleNamespace(bot_principals={})
        runtime_recovered = False

        @staticmethod
        def recover():
            return SimpleNamespace(states={state.intent_id: state})

        @staticmethod
        def recover_manifest_reauthorizations():
            return ()

        @staticmethod
        def begin_channel_revalidation(
            intent_id: str,
            *,
            observed_generation: int,
        ):
            revalidations.append((intent_id, observed_generation))
            return state

        @classmethod
        def mark_runtime_recovered(cls) -> None:
            cls.runtime_recovered = True

    async def configure(
        intent_id: str,
        *,
        force_slash_refresh: bool = False,
    ) -> None:
        configurations.append(
            (
                intent_id,
                force_slash_refresh,
                runtime._target_channel_generation(state),
            )
        )

    async def no_terminal_notifications() -> None:
        return None

    def submit(coroutine, *, label: str) -> None:
        submissions.append(label)
        asyncio.run(coroutine)

    runtime._service = Service()  # type: ignore[assignment]
    runtime._execution_blockers = ("test-isolation",)
    runtime._configure_intent = configure  # type: ignore[method-assign]
    runtime._submit_coroutine = submit  # type: ignore[method-assign]
    runtime._start_monitor_in_loop = lambda: None  # type: ignore[method-assign]
    runtime._retry_terminal_notifications = (  # type: ignore[method-assign]
        no_terminal_notifications
    )
    runtime._resume_recoverable_activation_required_replies = (  # type: ignore[method-assign]
        lambda: None
    )

    runtime.recover()

    assert revalidations == [(state.intent_id, state.channel_generation)]
    assert configurations == [
        (state.intent_id, True, state.channel_generation + 1)
    ]
    assert submissions == ["recovery"]
    assert Service.runtime_recovered is True


def test_ready_notification_retries_after_runtime_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path, limit=1)
    first_attempts: list[str] = []
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        notification_status=lambda _state, status: (
            first_attempts.append(status) or None
        ),
    )
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = first.hire_service.get_state(admitted.intent_id)
        events = [
            event.event_type
            for frame in first.journal_frames()
            for event in frame.events
            if event.aggregate_id
            == f"hire-notification:{admitted.intent_id}:ready"
        ]
        if (
            state is not None
            and state.phase is HirePhase.READY_PENDING_VERIFICATION
            and "hire.notification.action_required" in events
        ):
            break
        time.sleep(0.02)
    assert first_attempts == ["ready"]
    first.close()

    restarted_attempts: list[str] = []
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        notification_status=lambda _state, status: (
            restarted_attempts.append(status) or "om_recovered_notification"
        ),
    )
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        events = [
            event.event_type
            for frame in restarted.journal_frames()
            for event in frame.events
            if event.aggregate_id
            == f"hire-notification:{admitted.intent_id}:ready"
        ]
        if (
            restarted_attempts == ["ready"]
            and "hire.notification.committed" in events
            and restarted.readiness().ready is True
        ):
            break
        time.sleep(0.05)

    assert restarted_attempts == ["ready"]
    assert events.count("hire.notification.committed") == 1
    assert restarted.readiness().ready is True
    restarted.close()


def test_active_notification_retries_after_unacknowledged_reply(tmp_path: Path) -> None:
    active_attempts: list[str] = []

    def notify(_state, status: str):
        if status != "active":
            return "om_ready_notification"
        active_attempts.append(status)
        return None if len(active_attempts) == 1 else "om_active_notification"

    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        notification_status=notify,
    )
    active = _activate_employee(runtime, channels)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if active_attempts == ["active", "active"]:
            break
        time.sleep(0.02)

    events = [
        event.event_type
        for frame in runtime.journal_frames()
        for event in frame.events
        if event.aggregate_id == f"hire-notification:{active.intent_id}:active"
    ]
    assert active_attempts == ["active", "active"]
    assert events.count("hire.notification.action_required") == 1
    assert events.count("hire.notification.committed") == 1
    runtime.close()


def test_new_hire_reports_action_required_after_bounded_retries(tmp_path: Path) -> None:
    statuses: list[str] = []
    settings = _settings(tmp_path, limit=1)
    settings.autonomous_employee_auto_activation = True
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _FailingSlash(),
        notification_link=lambda *_: None,
        notification_status=lambda _state, status: (
            statuses.append(status) or "om_notification"
        ),
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        if (
            state is not None
            and state.phase is HirePhase.ACTION_REQUIRED
            and statuses == ["action_required"]
        ):
            break
        time.sleep(0.05)

    assert state is not None and state.phase is HirePhase.ACTION_REQUIRED
    assert statuses == ["action_required"]
    runtime.close()


def test_action_required_notification_retries_automatically(tmp_path: Path) -> None:
    attempts: list[str] = []

    def notify(_state, status: str):
        if status != "action_required":
            return "om_other_notification"
        attempts.append(status)
        return None if len(attempts) == 1 else "om_action_required"

    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _FailingSlash(),
        notification_link=lambda *_: None,
        notification_status=notify,
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        events = [
            event.event_type
            for frame in runtime.journal_frames()
            for event in frame.events
            if event.aggregate_id
            == f"hire-notification:{admitted.intent_id}:action_required"
        ]
        if (
            state is not None
            and state.phase is HirePhase.ACTION_REQUIRED
            and attempts == ["action_required", "action_required"]
            and "hire.notification.committed" in events
        ):
            break
        time.sleep(0.05)

    assert state is not None and state.phase is HirePhase.ACTION_REQUIRED
    assert attempts == ["action_required", "action_required"]
    assert events.count("hire.notification.action_required") == 1
    assert events.count("hire.notification.retry_requested") == 1
    assert events.count("hire.notification.committed") == 1
    runtime.close()


def test_main_bot_audit_open_failure_blocks_runtime(tmp_path: Path) -> None:
    settings = _settings(tmp_path, limit=1)

    with patch(
        "src.autonomous.provisioning.composition.MainBotSendAuditLog.open",
        side_effect=OSError("injected audit failure"),
    ):
        runtime = EmployeeDepartmentRuntime.from_settings(
            settings,
            notification_link=lambda *_: None,
        )

    assert runtime.hire_service is None
    assert runtime.readiness().blockers == ("main_bot_send_audit",)
    runtime.close()


def test_real_employee_status_ingress_and_employee_send_are_required_for_active(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    ack = _accept_durable_status(
        runtime,
        pending,
        suffix="status",
        generation=pending.channel_generation,
        connection_id=pending.channel_connection_id,
    )
    runtime._handle_control_ingress(ack.acceptance.acceptance_id)
    while time.monotonic() < deadline:
        active = runtime.hire_service.get_state(admitted.intent_id)
        if active is not None and active.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)

    assert active is not None and active.phase is HirePhase.ACTIVE
    assert active.activation_ingress_event_id == (
        "evt_" + hashlib.sha256(b"status").hexdigest()
    )
    assert active.activation_send_request_id == "send_runtime"
    assert channels.sent == [
        (
            active.agent_id,
            1,
            "ou_employee_app_admin",
            {"text": "Atlas activation verification started; not active yet."},
            {
                "uuid": hashlib.sha256(
                    (
                        "employee-activation-preflight:evt_"
                        + hashlib.sha256(b"status").hexdigest()
                    ).encode()
                ).hexdigest()[:50]
            },
        ),
        (
            active.agent_id,
            1,
            "ou_employee_app_admin",
            {"text": "Atlas is active."},
            {
                "uuid": hashlib.sha256(
                    (
                        "employee-activation-success:evt_"
                        + hashlib.sha256(b"status").hexdigest()
                    ).encode()
                ).hexdigest()[:50]
            },
        ),
    ]
    runtime.close()


def test_activation_fence_blocks_main_bot_outbound_until_atomic_commit(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    assert runtime.main_bot_outbound_audit is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None

    commit_entered = threading.Event()
    allow_commit = threading.Event()
    outbound_done = threading.Event()
    original_commit = runtime.hire_service.commit_activation

    def blocked_commit(*args, **kwargs):
        commit_entered.set()
        assert allow_commit.wait(2)
        return original_commit(*args, **kwargs)

    runtime.hire_service.commit_activation = blocked_commit  # type: ignore[method-assign]
    activation_result: list[bool] = []
    activation = threading.Thread(
        target=lambda: activation_result.append(
            runtime._complete_status_activation(
                state=pending,
                challenge=runtime._challenges[pending.intent_id],
                generation=pending.channel_generation,
                event_id="evt_fenced_activation",
                message_id="om_fenced_activation",
                sender_id="ou_employee_app_admin",
                sender_union_id=pending.requester_union_id,
                command="/status",
                is_p2p=True,
                reply_options={},
                received_at=time.time(),
            )
        )
    )
    activation.start()
    assert commit_entered.wait(2)

    outbound = threading.Thread(
        target=lambda: (
            runtime.main_bot_outbound_audit.record_attempt(
                pending.tenant_key,
                "reply",
                pending.message_id,
                attempted_at=time.time(),
            ),
            outbound_done.set(),
        )
    )
    outbound.start()
    assert not outbound_done.wait(0.1)

    allow_commit.set()
    activation.join(timeout=2)
    outbound.join(timeout=2)
    assert activation_result == [True]
    assert outbound_done.is_set()
    runtime.close()


def test_sibling_main_bot_mutation_with_requester_alias_blocks_activation(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    assert runtime.main_bot_outbound_audit is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None

    runtime.main_bot_outbound_audit.record_attempt(
        pending.tenant_key,
        "reply",
        "om_sibling_message",
        attempted_at=time.time(),
    )
    runtime.main_bot_outbound_audit.record_attempt(
        pending.tenant_key,
        "reply",
        pending.requester_principal_id,
        attempted_at=time.time(),
    )

    assert runtime._complete_status_activation(
        state=pending,
        challenge=runtime._challenges[pending.intent_id],
        generation=pending.channel_generation,
        event_id="evt_sibling_mutation",
        message_id="om_employee_status",
        sender_id="ou_employee_app_admin",
        sender_union_id=pending.requester_union_id,
        command="/status",
        is_p2p=True,
        reply_options={},
        received_at=time.time(),
    ) is False
    current = runtime.hire_service.get_state(admitted.intent_id)
    assert current is not None
    assert current.phase is HirePhase.READY_PENDING_VERIFICATION
    assert channels.sent[-1][3] == {
        "text": (
            "Activation window reset after a conflicting main Bot send. "
            "Send /status again."
        )
    }
    runtime.close()


def test_main_bot_target_send_blocks_employee_reply_before_activation(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    audited_targets: list[str] = []
    requester_target_hash = hashlib.sha256(b"ou_admin").hexdigest()
    collision_at = 0.0

    def audit(_tenant: str, target_hash: str, start: float, end: float) -> int:
        audited_targets.append(target_hash)
        return int(
            target_hash == requester_target_hash
            and start <= collision_at <= end
        )

    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    runtime._main_bot_send_audit = audit
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    challenge = runtime._challenges[pending.intent_id]
    collision_at = time.time()

    activated = runtime._complete_status_activation(
        state=pending,
        challenge=challenge,
        generation=pending.channel_generation,
        event_id="evt_main_bot_collision",
        message_id="om_main_bot_collision",
        sender_id="ou_employee_app_admin",
        sender_union_id=pending.requester_union_id,
        command="/status",
        is_p2p=True,
        reply_options={},
        received_at=time.time(),
    )

    assert activated is False
    assert channels.sent == [
        (
            pending.agent_id,
            pending.channel_generation,
            "ou_employee_app_admin",
            {
                "text": (
                    "Activation window reset after a conflicting main Bot send. "
                    "Send /status again."
                )
            },
            {
                "uuid": hashlib.sha256(
                    b"employee-activation-retry:evt_main_bot_collision"
                ).hexdigest()[:50]
            },
        )
    ]
    assert set(audited_targets) == {
        hashlib.sha256(value).hexdigest()
        for value in (
            b"om_main_bot_collision",
            b"on_admin",
            b"ou_admin",
            b"oc_admin_dm",
            b"om_composition",
        )
    }
    current = runtime.hire_service.get_state(admitted.intent_id)
    assert current is not None
    assert current.phase is HirePhase.READY_PENDING_VERIFICATION
    assert current.verification_nonce != challenge.nonce
    assert current.verification_issued_at > collision_at
    assert current.effect_state("verification-status-reply:evt_main_bot_collision") is None
    assert runtime._complete_status_activation(
        state=current,
        challenge=runtime._challenges[current.intent_id],
        generation=current.channel_generation,
        event_id="evt_main_bot_collision_retry",
        message_id="om_main_bot_collision_retry",
        sender_id="ou_employee_app_admin",
        sender_union_id=current.requester_union_id,
        command="/status",
        is_p2p=True,
        reply_options={},
        received_at=time.time(),
    )
    active = runtime.hire_service.get_state(admitted.intent_id)
    assert active is not None and active.phase is HirePhase.ACTIVE
    assert [item[3] for item in channels.sent[-2:]] == [
        {"text": "Atlas activation verification started; not active yet."},
        {"text": "Atlas is active."},
    ]
    runtime.close()


def test_post_reply_main_bot_collision_sends_durable_incomplete_notice(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    audit_calls = 0

    def audit(_tenant: str, target_hash: str, _start: float, _end: float) -> int:
        nonlocal audit_calls
        audit_calls += 1
        return int(
            audit_calls > 5
            and target_hash == hashlib.sha256(b"ou_admin").hexdigest()
        )

    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    runtime._main_bot_send_audit = audit
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None

    activated = runtime._complete_status_activation(
        state=pending,
        challenge=runtime._challenges[pending.intent_id],
        generation=pending.channel_generation,
        event_id="evt_post_reply_collision",
        message_id="om_post_reply_collision",
        sender_id="ou_employee_app_admin",
        sender_union_id=pending.requester_union_id,
        command="/status",
        is_p2p=True,
        reply_options={},
        received_at=time.time(),
    )

    assert activated is False
    assert [item[3] for item in channels.sent] == [
        {"text": "Atlas activation verification started; not active yet."},
        {"text": "Activation did not complete. Send /status again."},
    ]
    assert channels.sent[1][4] == {
        "uuid": hashlib.sha256(
            b"employee-activation-incomplete:evt_post_reply_collision"
        ).hexdigest()[:50]
    }
    current = runtime.hire_service.get_state(admitted.intent_id)
    assert current is not None and current.phase is HirePhase.READY_PENDING_VERIFICATION
    assert (
        current.effect_state("verification-status-reply:evt_post_reply_collision")
        is HireEffectState.COMMITTED
    )
    assert (
        current.effect_state("verification-incomplete-reply:evt_post_reply_collision")
        is HireEffectState.COMMITTED
    )
    runtime.close()


def test_committed_preflight_reply_replay_sends_idempotent_incomplete_notice(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    effect_id = "verification-status-reply:evt_committed_preflight_replay"
    for next_state in (HireEffectState.PREPARED, HireEffectState.EXECUTING):
        pending = runtime.hire_service.commit_effect_transition(
            pending.intent_id,
            effect_id=effect_id,
            effect_type="employee_status_reply",
            next_state=next_state,
        )
    pending = runtime.hire_service.commit_effect_transition(
        pending.intent_id,
        effect_id=effect_id,
        effect_type="employee_status_reply",
        next_state=HireEffectState.COMMITTED,
        metadata={
            "send_request_id": "req_preflight",
            "ingress_event_id": "evt_committed_preflight_replay",
            "reply_app_id": pending.app_id,
            "reply_message_id": "om_preflight",
            "generation": str(pending.channel_generation),
            "connection_id": pending.channel_connection_id,
            "main_bot_send_count": "1",
        },
    )

    def replay() -> bool:
        return runtime._complete_status_activation(
            state=pending,
            challenge=runtime._challenges[pending.intent_id],
            generation=pending.channel_generation,
            event_id="evt_committed_preflight_replay",
            message_id="om_committed_preflight_replay",
            sender_id="ou_employee_app_admin",
            sender_union_id=pending.requester_union_id,
            command="/status",
            is_p2p=True,
            reply_options={},
            received_at=time.time(),
        )

    assert replay() is False
    assert replay() is False
    assert [item[3] for item in channels.sent] == [
        {"text": "Activation did not complete. Send /status again."}
    ]
    current = runtime.hire_service.get_state(pending.intent_id)
    assert current is not None
    assert (
        current.effect_state(
            "verification-incomplete-reply:evt_committed_preflight_replay"
        )
        is HireEffectState.COMMITTED
    )
    runtime.close()


def test_challenge_expiring_during_employee_reply_sends_incomplete_notice(
    tmp_path: Path,
) -> None:
    class _SlowChannels(_Channels):
        def send(self, *args, **kwargs):
            if not self.sent:
                time.sleep(0.05)
            return super().send(*args, **kwargs)

    channels = _SlowChannels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    challenge = replace(
        runtime._challenges[pending.intent_id],
        expires_at=time.time() + 0.01,
    )

    activated = runtime._complete_status_activation(
        state=pending,
        challenge=challenge,
        generation=pending.channel_generation,
        event_id="evt_expired_during_reply",
        message_id="om_expired_during_reply",
        sender_id="ou_employee_app_admin",
        sender_union_id=pending.requester_union_id,
        command="/status",
        is_p2p=True,
        reply_options={},
        received_at=time.time(),
    )

    assert activated is False
    assert [item[3] for item in channels.sent] == [
        {"text": "Atlas activation verification started; not active yet."},
        {"text": "Activation did not complete. Send /status again."},
    ]
    assert channels.sent[1][4] == {
        "uuid": hashlib.sha256(
            b"employee-activation-incomplete:evt_expired_during_reply"
        ).hexdigest()[:50]
    }
    current = runtime.hire_service.get_state(admitted.intent_id)
    assert current is not None and current.phase is HirePhase.READY_PENDING_VERIFICATION
    runtime.close()


def test_expired_activation_notice_is_explicit_and_idempotent(tmp_path: Path) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None

    for _ in range(2):
        assert runtime._send_activation_retry_notice(
            state=runtime.hire_service.get_state(pending.intent_id) or pending,
            generation=pending.channel_generation,
            event_id="evt_expired_status",
            sender_id="ou_employee_app_admin",
            message="Activation window refreshed. Send /status again.",
        )

    assert channels.sent == [
        (
            pending.agent_id,
            pending.channel_generation,
            "ou_employee_app_admin",
            {"text": "Activation window refreshed. Send /status again."},
            {
                "uuid": hashlib.sha256(
                    b"employee-activation-retry:evt_expired_status"
                ).hexdigest()[:50]
            },
        )
    ]
    current = runtime.hire_service.get_state(pending.intent_id)
    assert current is not None
    assert current.phase is HirePhase.READY_PENDING_VERIFICATION
    assert (
        current.effect_state("verification-retry-reply:evt_expired_status")
        is HireEffectState.COMMITTED
    )
    runtime.close()


def test_activation_retry_replay_reuses_uuid_after_unknown_receipt(
    tmp_path: Path,
) -> None:
    class _UnknownFirstReceiptChannels(_Channels):
        def send(self, *args, **kwargs):
            receipt = super().send(*args, **kwargs)
            if len(self.sent) == 1:
                return replace(receipt, request_id="")
            return receipt

    channels = _UnknownFirstReceiptChannels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None

    event_id = "evt_retry_unknown_receipt"
    expected_uuid = hashlib.sha256(
        f"employee-activation-retry:{event_id}".encode()
    ).hexdigest()[:50]
    with pytest.raises(RuntimeError, match="retry receipt is invalid"):
        runtime._send_activation_retry_notice(
            state=pending,
            generation=pending.channel_generation,
            event_id=event_id,
            sender_id="ou_employee_app_admin",
            message="Activation window refreshed. Send /status again.",
        )

    assert runtime._send_activation_retry_notice(
        state=runtime.hire_service.get_state(pending.intent_id) or pending,
        generation=pending.channel_generation,
        event_id=event_id,
        sender_id="ou_employee_app_admin",
        message="Activation window refreshed. Send /status again.",
    )

    assert [item[4] for item in channels.sent] == [
        {"uuid": expected_uuid},
        {"uuid": expected_uuid},
    ]
    current = runtime.hire_service.get_state(pending.intent_id)
    assert current is not None
    assert (
        current.effect_state(f"verification-retry-reply:{event_id}")
        is HireEffectState.COMMITTED
    )
    runtime.close()


def _accept_pending_group_mention(
    runtime,
    state,
    *,
    bot_open_id: str,
    suffix: str,
):
    raw_chat_id = "oc_employee_team"
    raw_message_id = f"om_{suffix}"
    digest = hashlib.sha256(suffix.encode()).hexdigest()
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id=f"ing_{digest}",
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "group",
                "content": {"text": "@_user_1 你好"},
                "mentions": (
                    {
                        "key": "@_user_1",
                        "mentioned_type": "bot",
                        "open_id": bot_open_id,
                        "tenant_key": state.tenant_key,
                    },
                ),
                "sender_id": "ou_employee_app_admin",
                "sender_union_id": state.requester_union_id,
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": state.tenant_key,
                "feishu_thread_id": "",
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": "",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        bot_principal_id=state.bot_principal_id,
        app_id=state.app_id,
        channel_generation=state.channel_generation,
        connection_id=state.channel_connection_id,
        event_id=f"evt_{digest}",
        message_id="om_" + hashlib.sha256(raw_message_id.encode()).hexdigest(),
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        thread_root_message_id="",
        sender_principal_id="ou_employee_app_admin",
        received_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    return runtime.ingress_service.accept(
        metadata,
        payload,
        request_id=f"req_{suffix}",
    )


def _accept_durable_status(
    runtime,
    state,
    *,
    suffix: str,
    generation: int,
    connection_id: str,
):
    raw_message_id = f"om_{suffix}"
    raw_chat_id = f"chat-{suffix}"
    message_id = "om_" + hashlib.sha256(raw_message_id.encode()).hexdigest()
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + hashlib.sha256(suffix.encode()).hexdigest(),
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "p2p",
                "content": {"text": "/status"},
                "sender_id": "ou_employee_app_admin",
                "sender_union_id": state.requester_union_id,
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": state.tenant_key,
                "feishu_thread_id": "",
                "remote_chat_id": raw_chat_id,
                "remote_message_id": raw_message_id,
                "remote_root_id": "",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key=state.tenant_key,
        agent_id=state.agent_id,
        bot_principal_id=state.bot_principal_id,
        app_id=state.app_id,
        channel_generation=generation,
        connection_id=connection_id,
        event_id="evt_" + hashlib.sha256(suffix.encode()).hexdigest(),
        message_id=message_id,
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_" + hashlib.sha256(raw_chat_id.encode()).hexdigest(),
        thread_root_message_id="",
        sender_principal_id="ou_employee_app_admin",
        received_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    return runtime.ingress_service.accept(
        metadata,
        payload,
        request_id=f"req_{suffix}",
    )


@pytest.mark.parametrize(
    ("phase", "message_fragment"),
    (
        (HirePhase.VALIDATING, "currently validating"),
        (HirePhase.ACTION_REQUIRED, "requires administrator action"),
    ),
)
def test_status_before_activation_durably_explains_current_phase(
    tmp_path: Path,
    phase: HirePhase,
    message_fragment: str,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    current = runtime.hire_service._commit_phase_transition(pending, phase)
    ack = _accept_durable_status(
        runtime,
        current,
        suffix=f"early_status_{phase.value}",
        generation=current.channel_generation,
        connection_id=current.channel_connection_id,
    )

    assert runtime._handle_control_ingress(ack.acceptance.acceptance_id)
    assert runtime._handle_control_ingress(ack.acceptance.acceptance_id)

    runtime.ingress_service.rebuild_projection()
    disposition = runtime.ingress_service.state.by_acceptance_id[
        ack.acceptance.acceptance_id
    ].disposition
    assert disposition is not None
    assert disposition.reason_code == f"activation_status_{phase.value}"
    assert len(channels.sent) == 1
    assert message_fragment in channels.sent[0][3]["text"]
    runtime.close()


def test_early_status_reply_unknown_outcome_replays_with_same_uuid(
    tmp_path: Path,
) -> None:
    class _UnknownFirstReceiptChannels(_Channels):
        def send(self, *args, **kwargs):
            receipt = super().send(*args, **kwargs)
            if len(self.sent) == 1:
                return replace(receipt, request_id="")
            return receipt

    channels = _UnknownFirstReceiptChannels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    current = runtime.hire_service._commit_phase_transition(
        pending,
        HirePhase.ACTION_REQUIRED,
    )
    ack = _accept_durable_status(
        runtime,
        current,
        suffix="early_status_unknown_receipt",
        generation=current.channel_generation,
        connection_id=current.channel_connection_id,
    )

    with pytest.raises(RuntimeError, match="retry receipt is invalid"):
        runtime._handle_control_ingress(ack.acceptance.acceptance_id)
    assert runtime._handle_control_ingress(ack.acceptance.acceptance_id)

    expected_uuid = hashlib.sha256(
        (
            "employee-activation-retry:evt_"
            + hashlib.sha256(b"early_status_unknown_receipt").hexdigest()
        ).encode()
    ).hexdigest()[:50]
    assert [item[4]["uuid"] for item in channels.sent] == [
        expected_uuid,
        expected_uuid,
    ]
    runtime.close()


@pytest.mark.parametrize(
    "phase",
    (HirePhase.VALIDATING, HirePhase.ACTION_REQUIRED),
)
def test_early_status_unknown_reply_recovers_original_decision_after_restart(
    tmp_path: Path,
    phase: HirePhase,
) -> None:
    class _UnknownReceiptChannels(_Channels):
        def send(self, *args, **kwargs):
            receipt = super().send(*args, **kwargs)
            return replace(receipt, request_id="")

    settings = _settings(tmp_path, limit=1)
    first_channels = _UnknownReceiptChannels()
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=first_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    current = first.hire_service._commit_phase_transition(pending, phase)
    ack = _accept_durable_status(
        first,
        current,
        suffix=f"restart_early_status_{phase.value}",
        generation=current.channel_generation,
        connection_id=current.channel_connection_id,
    )
    acceptance_id = ack.acceptance.acceptance_id
    with pytest.raises(RuntimeError, match="retry receipt is invalid"):
        first._handle_control_ingress(acceptance_id)
    first_uuid = first_channels.sent[-1][4]["uuid"]
    if phase is HirePhase.VALIDATING:
        unresolved = first.hire_service.get_state(admitted.intent_id)
        assert unresolved is not None
        first.hire_service._commit_phase_transition(
            unresolved,
            HirePhase.READY_PENDING_VERIFICATION,
        )
    first.close()

    restarted_channels = _Channels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        recovered_before_reply = restarted.hire_service.get_state(
            admitted.intent_id
        )
        if (
            recovered_before_reply is not None
            and recovered_before_reply.channel_generation
            == current.channel_generation + 1
            and recovered_before_reply.agent_id in restarted_channels.statuses
        ):
            break
        time.sleep(0.05)
    assert recovered_before_reply is not None
    assert (
        recovered_before_reply.channel_generation
        == current.channel_generation + 1
    )

    assert restarted._handle_control_ingress(acceptance_id)

    restarted.ingress_service.rebuild_projection()
    disposition = restarted.ingress_service.state.by_acceptance_id[
        acceptance_id
    ].disposition
    recovered = restarted.hire_service.get_state(admitted.intent_id)
    assert disposition is not None
    assert disposition.reason_code == f"activation_status_{phase.value}"
    assert restarted_channels.sent[-1][4]["uuid"] == first_uuid
    assert recovered is not None
    assert recovered.phase is (
        HirePhase.READY_PENDING_VERIFICATION
        if phase is HirePhase.VALIDATING
        else HirePhase.ACTION_REQUIRED
    )
    assert recovered.activation_ingress_event_id == ""
    restarted.close()


def test_stale_generation_status_prompts_once_then_current_status_activates(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1, context_configured=True),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 7
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    old_generation = pending.channel_generation
    old_connection_id = pending.channel_connection_id
    channels.crash(pending.agent_id)
    while time.monotonic() < deadline:
        refreshed = runtime.hire_service.get_state(admitted.intent_id)
        if (
            refreshed is not None
            and refreshed.phase is HirePhase.READY_PENDING_VERIFICATION
            and refreshed.channel_generation == old_generation + 1
        ):
            break
        time.sleep(0.05)
    assert refreshed is not None
    old_ack = _accept_durable_status(
        runtime,
        refreshed,
        suffix="stale_generation_status",
        generation=old_generation,
        connection_id=old_connection_id,
    )

    assert runtime._handle_control_ingress(old_ack.acceptance.acceptance_id)
    assert runtime._handle_control_ingress(old_ack.acceptance.acceptance_id)
    runtime.ingress_service.rebuild_projection()
    old_disposition = runtime.ingress_service.state.by_acceptance_id[
        old_ack.acceptance.acceptance_id
    ].disposition
    assert old_disposition is not None
    assert old_disposition.reason_code == "activation_generation_refreshed"
    assert channels.sent == [
        (
            refreshed.agent_id,
            refreshed.channel_generation,
            "ou_employee_app_admin",
            {"text": "Employee session refreshed. Send /status again."},
            {
                "uuid": hashlib.sha256(
                    b"employee-activation-retry:evt_"
                    + hashlib.sha256(b"stale_generation_status").hexdigest().encode()
                ).hexdigest()[:50]
            },
        )
    ]

    current_ack = _accept_durable_status(
        runtime,
        refreshed,
        suffix="current_generation_status",
        generation=refreshed.channel_generation,
        connection_id=refreshed.channel_connection_id,
    )
    assert runtime._handle_control_ingress(current_ack.acceptance.acceptance_id)
    active = runtime.hire_service.get_state(admitted.intent_id)
    assert active is not None and active.phase is HirePhase.ACTIVE
    assert len(channels.sent) == 3
    runtime.close()


def test_distinct_status_event_after_activation_replies_already_active_once(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    active = _activate_employee(runtime, channels)
    original_activation_event_id = active.activation_ingress_event_id
    repeated = _accept_durable_status(
        runtime,
        active,
        suffix="already_active_distinct_event",
        generation=active.channel_generation,
        connection_id=active.channel_connection_id,
    )

    assert runtime._handle_control_ingress(repeated.acceptance.acceptance_id)
    assert runtime._handle_control_ingress(repeated.acceptance.acceptance_id)

    runtime.ingress_service.rebuild_projection()
    disposition = runtime.ingress_service.state.by_acceptance_id[
        repeated.acceptance.acceptance_id
    ].disposition
    assert disposition is not None
    assert disposition.reason_code == "activation_already_active"
    assert channels.sent[-1] == (
        active.agent_id,
        active.channel_generation,
        "ou_employee_app_admin",
        {"text": "Atlas is already active."},
        {
            "uuid": hashlib.sha256(
                (
                    "employee-already-active:"
                    + repeated.acceptance.acceptance_id
                ).encode()
            ).hexdigest()[:50]
        },
    )
    assert len(channels.sent) == 3
    current = runtime.hire_service.get_state(active.intent_id)
    assert current is not None
    assert current.phase is HirePhase.ACTIVE
    assert current.activation_ingress_event_id == original_activation_event_id
    assert sum(
        event.event_type == "hire.activation.verified"
        for frame in runtime.journal_frames()
        for event in frame.events
    ) == 1
    runtime.close()


def test_concurrent_duplicate_active_status_callbacks_send_one_reply(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    active = _activate_employee(runtime, channels)
    repeated = _accept_durable_status(
        runtime,
        active,
        suffix="already_active_concurrent",
        generation=active.channel_generation,
        connection_id=active.channel_connection_id,
    )
    barrier = threading.Barrier(3)

    def handle() -> bool:
        barrier.wait()
        return runtime._handle_control_ingress(repeated.acceptance.acceptance_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(handle) for _ in range(2)]
        barrier.wait()
        assert [future.result(timeout=5) for future in futures] == [True, True]

    runtime.ingress_service.rebuild_projection()
    disposition = runtime.ingress_service.state.by_acceptance_id[
        repeated.acceptance.acceptance_id
    ].disposition
    assert disposition is not None
    assert disposition.reason_code == "activation_already_active"
    assert len(channels.sent) == 3
    assert channels.sent[-1][3] == {"text": "Atlas is already active."}
    assert channels.sent[-1][4] == {
        "uuid": hashlib.sha256(
            (
                "employee-already-active:"
                + repeated.acceptance.acceptance_id
            ).encode()
        ).hexdigest()[:50]
    }
    assert sum(
        event.event_type == "hire.activation.verified"
        for frame in runtime.journal_frames()
        for event in frame.events
    ) == 1
    runtime.close()


def test_durable_employee_status_ingress_activates_before_general_router(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    audited_targets: list[str] = []
    notification_statuses: list[str] = []

    def audit(_tenant: str, target_hash: str, _start: float, _end: float) -> int:
        audited_targets.append(target_hash)
        return 0

    settings = _settings(tmp_path, limit=1, context_configured=True)
    settings.allowed_chat_ids = frozenset({"oc_employee_team"})
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        context_source_factory=_ContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
        notification_status=lambda _state, status: (
            notification_statuses.append(status) or "om_notification"
        ),
    )
    runtime._main_bot_send_audit = audit
    assert runtime.hire_service is not None
    assert runtime.ingress_service is not None
    assert runtime.ingress_router is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    pending = None
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    runtime._challenges.pop(pending.intent_id)
    reconstructed_challenge = runtime._challenge_for_state(pending)
    assert reconstructed_challenge is not None
    runtime._challenges[pending.intent_id] = replace(
        reconstructed_challenge,
        expires_at=time.time() + 5,
    )

    raw_message_id = "om_durable_status"
    event_id = "evt_" + hashlib.sha256(b"durable-status-event").hexdigest()
    message_id = "om_" + hashlib.sha256(raw_message_id.encode()).hexdigest()
    chat_id = "oc_" + hashlib.sha256(b"durable-status-chat").hexdigest()
    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + hashlib.sha256(b"durable-status-envelope").hexdigest(),
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "p2p",
                "content": {"text": "/status"},
                "sender_id": "ou_employee_app_admin",
                "sender_union_id": "on_admin",
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": pending.tenant_key,
                "feishu_thread_id": "",
                "remote_chat_id": "durable-status-chat",
                "remote_message_id": raw_message_id,
                "remote_root_id": "",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key=pending.tenant_key,
        agent_id=pending.agent_id,
        bot_principal_id=pending.bot_principal_id,
        app_id=pending.app_id,
        channel_generation=pending.channel_generation,
        connection_id=pending.channel_connection_id,
        event_id=event_id,
        message_id=message_id,
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id=chat_id,
        thread_root_message_id="",
        sender_principal_id="ou_employee_app_admin",
        received_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    ack = runtime.ingress_service.accept(
        metadata,
        payload,
        request_id="req_durable_status",
    )
    original_record_disposition = runtime.ingress_service.record_disposition
    dropped_verified_disposition = False

    def drop_first_verified_disposition(
        acceptance_id: str,
        *,
        state: str,
        reason_code: str,
    ) -> object:
        nonlocal dropped_verified_disposition
        if reason_code == "activation_verified" and not dropped_verified_disposition:
            dropped_verified_disposition = True
            return None
        return original_record_disposition(
            acceptance_id,
            state=state,
            reason_code=reason_code,
        )

    runtime.ingress_service.record_disposition = drop_first_verified_disposition  # type: ignore[method-assign]
    callback = channels.callbacks[pending.agent_id]
    callback(
        {
            "event": "durableIngressAccepted",
            "data": {
                "acceptance_id": ack.acceptance.acceptance_id,
                "agent_id": pending.agent_id,
                "generation": pending.channel_generation,
            },
        }
    )

    active = None
    while time.monotonic() < deadline:
        active = runtime.hire_service.get_state(admitted.intent_id)
        if active is not None and active.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)

    assert active is not None and active.phase is HirePhase.ACTIVE
    runtime.ingress_service.rebuild_projection()
    assert (
        runtime.ingress_service.state.by_acceptance_id[
            ack.acceptance.acceptance_id
        ].disposition
        is None
    )
    callback(
        {
            "event": "durableIngressAccepted",
            "data": {
                "acceptance_id": ack.acceptance.acceptance_id,
                "agent_id": pending.agent_id,
                "generation": pending.channel_generation,
            },
        }
    )
    while time.monotonic() < deadline:
        runtime.ingress_service.rebuild_projection()
        if (
            runtime.ingress_service.state.by_acceptance_id[
                ack.acceptance.acceptance_id
            ].disposition
            is not None
        ):
            break
        time.sleep(0.02)
    disposition = runtime.ingress_service.state.by_acceptance_id[
        ack.acceptance.acceptance_id
    ].disposition
    assert disposition is not None
    assert disposition.reason_code == "activation_verified"
    router_record = runtime.ingress_router.state.by_acceptance_id[
        ack.acceptance.acceptance_id
    ]
    assert router_record.state == "terminal"
    assert router_record.reason_code == "control_consumed"
    expected_audit_targets = sorted(
        {
            hashlib.sha256(value).hexdigest()
            for value in (
                raw_message_id.encode(),
                b"on_admin",
                b"ou_admin",
                b"oc_admin_dm",
                b"om_composition",
            )
        }
    )
    assert audited_targets == expected_audit_targets * 2
    activation_frame = next(
        frame
        for frame in runtime.journal_frames()
        if any(event.event_type == "hire.activation.verified" for event in frame.events)
    )
    assert [event.event_type for event in activation_frame.events] == [
        "hire.effect.committed",
        "hire.verification.nonce_consumed",
        "hire.activation.verified",
        "employee.state_changed",
    ]
    assert channels.sent[-1] == (
        active.agent_id,
        active.channel_generation,
        "ou_employee_app_admin",
        {"text": "Atlas is active."},
        {
            "uuid": hashlib.sha256(
                (
                    "employee-activation-success:"
                    + active.activation_ingress_event_id
                ).encode()
            ).hexdigest()[:50]
        },
    )
    assert len(channels.sent) == 2
    notification_deadline = time.monotonic() + 2
    while (
        time.monotonic() < notification_deadline
        and "active" not in notification_statuses
    ):
        time.sleep(0.02)
    assert notification_statuses.count("active") == 1
    runtime.close()


def test_forged_employee_send_identity_cannot_activate(tmp_path: Path) -> None:
    channels = _ForgedReceiptChannels()
    runtime = _runtime(
        _settings(tmp_path, limit=1),
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = runtime.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    ack = _accept_durable_status(
        runtime,
        pending,
        suffix="forged_status",
        generation=pending.channel_generation,
        connection_id=pending.channel_connection_id,
    )
    with pytest.raises(RuntimeError, match="receipt is invalid"):
        runtime._handle_control_ingress(ack.acceptance.acceptance_id)

    rejected = runtime.hire_service.get_state(admitted.intent_id)
    assert rejected is not None
    assert rejected.phase is HirePhase.READY_PENDING_VERIFICATION
    assert rejected.activation_send_request_id == ""
    runtime.close()


def test_limit_zero_hard_disables_runtime_even_with_existing_employee_state(
    tmp_path: Path,
) -> None:
    first_channels = _Channels()
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=first_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = first.hire_service.get_state(admitted.intent_id)
        if state is not None and state.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert state is not None
    first.close()

    settings.autonomous_visible_employee_limit = 0
    recovered = _runtime(
        settings,
        release_evidence_ready=False,
    )

    assert recovered.hire_service is None
    assert recovered.readiness().blockers == ("visible_employee_limit",)
    recovered.close()


def test_hard_closed_nonterminal_replay_cannot_resubmit_external_activity(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _FailingSlash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        failed = first.hire_service.get_state(admitted.intent_id)
        if (
            failed is not None
            and failed.phase is HirePhase.CONFIGURING
            and failed.effect_state("slash-reconcile:1:1") is HireEffectState.EXECUTING
        ):
            break
        time.sleep(0.02)
    assert failed is not None
    first.close()

    settings.autonomous_visible_employee_limit = 0
    channels = _Channels()
    recovered = _runtime(
        settings,
        release_evidence_ready=False,
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
    )
    assert recovered.hire_service is None
    assert recovered.readiness().blockers == ("visible_employee_limit",)
    time.sleep(0.2)
    assert channels.started == []
    recovered.close()


def test_crashed_active_channel_advances_generation_and_requires_reverification(
    tmp_path: Path,
) -> None:
    channels = _Channels()
    settings = _settings(tmp_path, limit=1, context_configured=True)
    settings.allowed_chat_ids = frozenset({"oc_employee_team"})
    runtime = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
        team_notification=lambda *_: None,
        context_source_factory=_AssemblingContextSourceFactory(),
        group_memory_backend=_GroupMemory(),
        membership_health=_HealthyMembership(),
    )
    assert runtime.hire_service is not None
    admitted = runtime.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = runtime.hire_service.get_state(admitted.intent_id)
        if state is not None and state.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert state is not None
    ack = _accept_durable_status(
        runtime,
        state,
        suffix="crash_revalidation",
        generation=state.channel_generation,
        connection_id=state.channel_connection_id,
    )
    runtime._handle_control_ingress(ack.acceptance.acceptance_id)
    while time.monotonic() < deadline:
        active = runtime.hire_service.get_state(admitted.intent_id)
        if active is not None and active.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)
    assert active is not None and active.phase is HirePhase.ACTIVE
    channels.statuses[active.agent_id].tenant_key = active.tenant_key
    channels.statuses[active.agent_id].bot_principal_id = active.bot_principal_id
    assert runtime._writer is not None
    commit_workforce_events(
        runtime._writer,
        runtime.hire_service.projection_state,
        (
            JournalEvent(
                event_type="employee.membership_changed",
                aggregate_id=active.agent_id,
                payload={"member_groups": ["oc_employee_team"]},
            ),
        ),
    )
    assert runtime.team_service is not None
    backend = runtime.team_service._backend  # noqa: SLF001
    old_target = backend.list_active(active.tenant_key, "oc_employee_team")[0]

    channels.crash(active.agent_id)
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        revalidating = runtime.hire_service.get_state(admitted.intent_id)
        if (
            revalidating is not None
            and revalidating.phase is HirePhase.READY_PENDING_VERIFICATION
            and revalidating.channel_generation == 2
        ):
            break
        time.sleep(0.05)

    assert revalidating is not None
    assert revalidating.phase is HirePhase.READY_PENDING_VERIFICATION
    assert revalidating.channel_generation == 2
    assert [item[3] for item in channels.started] == [1, 2]

    acceptance_ids = set(runtime.ingress_service.state.by_acceptance_id)
    with pytest.raises(RuntimeError, match="channel authority is unavailable"):
        backend.submit(
            run_id="teamrun_stale_generation",
            step_id="analysis",
            target=old_target,
            tenant_key=active.tenant_key,
            chat_id="oc_employee_team",
            message_id="om_stale_generation",
            requester_principal_id="ou_admin",
            instruction="旧 Channel 快照不得执行",
            deadline_at="2099-01-01T00:00:00Z",
        )
    assert set(runtime.ingress_service.state.by_acceptance_id) == acceptance_ids

    ack = _accept_durable_status(
        runtime,
        revalidating,
        suffix="crash_revalidation_generation_2",
        generation=revalidating.channel_generation,
        connection_id=revalidating.channel_connection_id,
    )
    runtime._handle_control_ingress(ack.acceptance.acceptance_id)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        reverified = runtime.hire_service.get_state(admitted.intent_id)
        if reverified is not None and reverified.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)
    assert reverified is not None and reverified.phase is HirePhase.ACTIVE
    acceptance_id = backend.submit(
        run_id="teamrun_reverified_generation",
        step_id="analysis",
        target=old_target,
        tenant_key=active.tenant_key,
        chat_id="oc_employee_team",
        message_id="om_reverified_generation",
        requester_principal_id="ou_admin",
        instruction="完成验证后允许执行",
        deadline_at="2099-01-01T00:00:00Z",
    )
    assert acceptance_id in runtime.ingress_service.state.by_acceptance_id
    runtime.close()


def test_restart_replaces_active_channel_generation_and_challenge(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    first_channels = _Channels()
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=first_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    old_nonce = pending.verification_nonce
    ack = _accept_durable_status(
        first,
        pending,
        suffix="restart",
        generation=pending.channel_generation,
        connection_id=pending.channel_connection_id,
    )
    first._handle_control_ingress(ack.acceptance.acceptance_id)
    while time.monotonic() < deadline:
        active = first.hire_service.get_state(admitted.intent_id)
        if active is not None and active.phase is HirePhase.ACTIVE:
            break
        time.sleep(0.02)
    assert active is not None and active.phase is HirePhase.ACTIVE
    first.close()

    restarted_channels = _Channels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        reverified = restarted.hire_service.get_state(admitted.intent_id)
        if (
            reverified is not None
            and reverified.phase is HirePhase.READY_PENDING_VERIFICATION
            and reverified.channel_generation == 2
        ):
            break
        time.sleep(0.05)

    assert reverified is not None
    assert reverified.phase is HirePhase.READY_PENDING_VERIFICATION
    assert reverified.channel_generation == 2
    assert reverified.verification_nonce != old_nonce
    assert [item[3] for item in restarted_channels.started] == [2]
    restarted.close()


def test_restart_slash_drift_cannot_reuse_old_verified_hash(tmp_path: Path) -> None:
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        pending = first.hire_service.get_state(admitted.intent_id)
        if pending is not None and pending.phase is HirePhase.READY_PENDING_VERIFICATION:
            break
        time.sleep(0.02)
    assert pending is not None
    old_nonce = pending.verification_nonce
    first.close()

    restarted_channels = _Channels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _FailingSlash(),
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = restarted.hire_service.get_state(admitted.intent_id)
        if state is not None and state.phase is HirePhase.VALIDATING:
            time.sleep(0.1)
            break
        time.sleep(0.02)

    state = restarted.hire_service.get_state(admitted.intent_id)
    assert state is not None
    assert state.phase is HirePhase.VALIDATING
    assert state.verification_nonce == old_nonce
    assert restarted_channels.started == []
    restarted.close()


def test_restart_after_failed_channel_attempts_forces_new_generation_slash(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_FailingStartChannels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        crashed = first.hire_service.get_state(admitted.intent_id)
        if (
            crashed is not None
            and crashed.effect_state("slash-reconcile:2:1") is HireEffectState.COMMITTED
            and crashed.effect_state("channel-start:2") is HireEffectState.ACTION_REQUIRED
        ):
            break
        time.sleep(0.02)
    assert crashed is not None
    first.close()

    restarted_channels = _Channels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _FailingSlash(),
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = restarted.hire_service.get_state(admitted.intent_id)
        if state is not None and state.effect_state("slash-reconcile:3:1") is not None:
            break
        time.sleep(0.02)

    state = restarted.hire_service.get_state(admitted.intent_id)
    assert state is not None
    assert state.phase is HirePhase.CONFIGURING
    assert state.effect_state("slash-reconcile:3:1") is HireEffectState.EXECUTING
    assert restarted_channels.started == []
    restarted.close()


def test_restart_after_channel_commit_fences_generation_before_slash_refresh(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, limit=1)
    first = _runtime(
        settings,
        release_evidence_ready=True,
        registrar=_Registrar(),
        channel_supervisor=_Channels(),
        slash_reconciler_factory=lambda _app_id, _secret: _Slash(),
        notification_link=lambda *_: None,
    )
    assert first.hire_service is not None
    first._verification_router = _FailingVerificationRouter()  # type: ignore[assignment]
    admitted = first.hire_service.start_hire(_request())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        crashed = first.hire_service.get_state(admitted.intent_id)
        if (
            crashed is not None
            and crashed.phase is HirePhase.CONFIGURING
            and crashed.channel_generation == 1
            and crashed.effect_state("channel-start:1") is HireEffectState.COMMITTED
        ):
            break
        time.sleep(0.02)
    assert crashed is not None
    first.close()

    restarted_channels = _Channels()
    restarted = _runtime(
        settings,
        release_evidence_ready=True,
        channel_supervisor=restarted_channels,
        slash_reconciler_factory=lambda _app_id, _secret: _FailingSlash(),
        notification_link=lambda *_: None,
    )
    assert restarted.hire_service is not None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = restarted.hire_service.get_state(admitted.intent_id)
        if state is not None and state.effect_state("slash-reconcile:2:1") is not None:
            break
        time.sleep(0.02)

    state = restarted.hire_service.get_state(admitted.intent_id)
    assert state is not None
    assert state.phase is HirePhase.VALIDATING
    assert state.channel_generation == 1
    assert state.effect_state("slash-reconcile:2:1") is HireEffectState.EXECUTING
    assert restarted_channels.started == []
    restarted.close()
