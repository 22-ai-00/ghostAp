from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.autonomous.ingress.projection import IngressProjectionState
from src.autonomous.ingress.service import EmployeeIngressService
from src.autonomous.journal.blob_store import AesGcmEncryptionProvider, BlobStore
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.provisioning.external_mutation_gate import (
    EmployeeExternalMutationGate,
)
from src.autonomous.provisioning.fire_authority import JournalFireAuthority
from src.autonomous.provisioning.fire_service import (
    EmployeeFireRequest,
    EmployeeFireService,
    FireServiceError,
)
from src.autonomous.provisioning.fire_state import FIRE_EFFECT_ORDER, FirePhase
from src.autonomous.provisioning.hire_state import HireEffectState
from src.autonomous.supervisor.employee_channels import ChannelProcessState
from tests.autonomous.integration.test_employee_runtime_recovery import (
    _seed_active_employee,
    _writer,
)
from tests.autonomous.integration.test_employee_runtime_recovery import (
    _service as _recovery_service,
)
from tests.autonomous.unit.test_employee_hire_admission import (
    _request as _hire_request,
)
from tests.autonomous.unit.test_employee_hire_admission import (
    _service as _hire_service,
)


class _VaultSpy:
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, *_args):
        self.calls += 1
        raise AssertionError("credential lookup must not run after retirement fence")


class _ChannelSpy:
    def __init__(self) -> None:
        self.calls = 0

    def start(self, *_args):
        self.calls += 1
        raise AssertionError("Channel start must not run after retirement fence")


class _BlockingSlash:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def reconcile(self):
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(
            spec_hash="slash_hash_2",
            observed_hash="slash_hash_2",
            observed=(),
        )


class _BlockingChannel:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def start(self, _agent_id, app_id, _credential_ref, generation, _callback):
        self.started.set()
        assert self.release.wait(2.0)
        return SimpleNamespace(
            state=ChannelProcessState.READY,
            identity={"app_id": app_id},
            ready_metadata={"connection_id": f"conn_generation_{generation}"},
            error_code="",
        )


class _CancelledSlash:
    async def reconcile(self):
        raise asyncio.CancelledError


class _BlockingFailSlash:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def reconcile(self):
        self.started.set()
        await self.release.wait()
        raise RuntimeError("slash failed after caller cancellation")


class _CancelledChannel:
    def start(self, *_args):
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_retirement_fence_blocks_slash_and_channel_before_external_calls(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    vault = _VaultSpy()
    channels = _ChannelSpy()
    slash_factory_calls = []
    runtime._service = service  # noqa: SLF001
    runtime._vault = vault  # type: ignore[assignment]  # noqa: SLF001
    runtime._channels = channels  # type: ignore[assignment]  # noqa: SLF001
    runtime._slash_factory = lambda *_args: slash_factory_calls.append(_args)  # noqa: SLF001
    runtime._external_mutation_gate.restore_retirement_fence(  # noqa: SLF001
        "tenant-a",
        "agt_recover",
    )
    state = service.get_state("hire_recover")
    assert state is not None

    try:
        with pytest.raises(RuntimeError, match="retiring"):
            await runtime._reconcile_slash(  # noqa: SLF001
                state,
                generation=2,
                force_refresh=True,
                allow_action_required_refresh=False,
            )
        with pytest.raises(RuntimeError, match="retiring"):
            await runtime._start_channel(state)  # noqa: SLF001

        assert vault.calls == 0
        assert slash_factory_calls == []
        assert channels.calls == 0
    finally:
        service.close()


@pytest.mark.asyncio
async def test_retirement_waits_for_inflight_slash_terminal_anchor(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    slash = _BlockingSlash()
    runtime._service = service  # noqa: SLF001
    runtime._vault = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        resolve=lambda *_args: "secret"
    )
    runtime._slash_factory = lambda *_args: slash  # noqa: SLF001
    state = service.get_state("hire_recover")
    assert state is not None
    reconcile = asyncio.create_task(
        runtime._reconcile_slash(  # noqa: SLF001
            state,
            generation=2,
            force_refresh=True,
            allow_action_required_refresh=False,
        )
    )
    await asyncio.wait_for(slash.started.wait(), timeout=1.0)
    retirement = asyncio.create_task(
        asyncio.to_thread(
            runtime._external_mutation_gate.begin_retirement,  # noqa: SLF001
            "tenant-a",
            "agt_recover",
            timeout_seconds=1.0,
        )
    )
    await asyncio.sleep(0.02)
    assert retirement.done() is False

    slash.release.set()
    await reconcile
    assert await retirement is True
    current = service.get_state("hire_recover")
    assert current is not None
    assert (
        current.effect_state("slash-reconcile:2:1")
        is HireEffectState.COMMITTED
    )
    service.close()


@pytest.mark.asyncio
async def test_slash_credential_failure_is_recorded_as_replay_safe(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._vault = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        resolve=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("cannot schedule new futures after shutdown")
        )
    )
    runtime._slash_factory = lambda *_args: pytest.fail(  # noqa: SLF001
        "Slash API must not run when credential resolution failed"
    )
    state = service.get_state("hire_recover")
    assert state is not None

    with pytest.raises(RuntimeError, match="credential resolution failed"):
        await runtime._reconcile_slash(  # noqa: SLF001
            state,
            generation=2,
            force_refresh=True,
            allow_action_required_refresh=False,
        )

    current = service.get_state("hire_recover")
    assert current is not None
    assert current.effect_state("slash-reconcile:2:1") is HireEffectState.ACTION_REQUIRED
    assert dict(current.metadata_for("slash-reconcile:2:1")) == {
        "error_code": "recovery_exhausted"
    }
    service.close()


@pytest.mark.asyncio
async def test_closing_runtime_does_not_start_slash_credential_resolution(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._closing = True  # noqa: SLF001
    runtime._vault = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        resolve=lambda *_args: pytest.fail("closing runtime must not use its executor")
    )
    runtime._slash_factory = lambda *_args: pytest.fail(  # noqa: SLF001
        "closing runtime must not call the Slash API"
    )
    state = service.get_state("hire_recover")
    assert state is not None

    with pytest.raises(asyncio.CancelledError):
        await runtime._reconcile_slash(  # noqa: SLF001
            state,
            generation=2,
            force_refresh=True,
            allow_action_required_refresh=False,
        )

    current = service.get_state("hire_recover")
    assert current is not None
    assert dict(current.metadata_for("slash-reconcile:2:1")) == {
        "error_code": "recovery_exhausted"
    }
    service.close()


@pytest.mark.asyncio
async def test_retirement_waits_for_inflight_channel_ready_anchor(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    channels = _BlockingChannel()
    runtime._service = service  # noqa: SLF001
    runtime._channels = channels  # type: ignore[assignment]  # noqa: SLF001
    state = service.get_state("hire_recover")
    assert state is not None
    channel = asyncio.create_task(
        runtime._start_channel(state)  # noqa: SLF001
    )
    assert await asyncio.to_thread(channels.started.wait, 1.0)
    retirement = asyncio.create_task(
        asyncio.to_thread(
            runtime._external_mutation_gate.begin_retirement,  # noqa: SLF001
            "tenant-a",
            "agt_recover",
            timeout_seconds=1.0,
        )
    )
    await asyncio.sleep(0.02)
    assert retirement.done() is False

    channels.release.set()
    await channel
    assert await retirement is True
    current = service.get_state("hire_recover")
    assert current is not None
    assert current.effect_state("channel-start:2") is HireEffectState.COMMITTED
    service.close()


@pytest.mark.asyncio
async def test_child_cancelled_slash_is_disposed_before_releasing_lease(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._vault = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        resolve=lambda *_args: "secret"
    )
    runtime._slash_factory = lambda *_args: _CancelledSlash()  # noqa: SLF001
    state = service.get_state("hire_recover")
    assert state is not None

    with pytest.raises(RuntimeError, match="Slash reconciliation failed"):
        await runtime._reconcile_slash(  # noqa: SLF001
            state,
            generation=2,
            force_refresh=True,
            allow_action_required_refresh=False,
        )

    current = service.get_state("hire_recover")
    assert current is not None
    assert (
        current.effect_state("slash-reconcile:2:1")
        is HireEffectState.ACTION_REQUIRED
    )
    assert runtime._external_mutation_gate.begin_retirement(  # noqa: SLF001
        "tenant-a",
        "agt_recover",
        timeout_seconds=0,
    ) is True
    service.close()


@pytest.mark.asyncio
async def test_child_cancelled_channel_is_disposed_before_releasing_lease(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = service  # noqa: SLF001
    runtime._channels = _CancelledChannel()  # type: ignore[assignment]  # noqa: SLF001
    state = service.get_state("hire_recover")
    assert state is not None

    with pytest.raises(RuntimeError, match="Channel start failed"):
        await runtime._start_channel(state)  # noqa: SLF001

    current = service.get_state("hire_recover")
    assert current is not None
    assert current.effect_state("channel-start:2") is HireEffectState.ACTION_REQUIRED
    assert dict(current.metadata_for("channel-start:2"))["error_code"] == (
        "start-cancelled"
    )
    assert runtime._external_mutation_gate.begin_retirement(  # noqa: SLF001
        "tenant-a",
        "agt_recover",
        timeout_seconds=0,
    ) is True
    service.close()


@pytest.mark.asyncio
async def test_cancelled_slash_with_child_failure_preserves_cancellation(
    tmp_path: Path,
) -> None:
    _seed_active_employee(tmp_path)
    service = _recovery_service(_writer(tmp_path, 2))
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    slash = _BlockingFailSlash()
    runtime._service = service  # noqa: SLF001
    runtime._vault = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        resolve=lambda *_args: "secret"
    )
    runtime._slash_factory = lambda *_args: slash  # noqa: SLF001
    state = service.get_state("hire_recover")
    assert state is not None
    activity = asyncio.create_task(
        runtime._reconcile_slash(  # noqa: SLF001
            state,
            generation=2,
            force_refresh=True,
            allow_action_required_refresh=False,
        )
    )
    await asyncio.wait_for(slash.started.wait(), timeout=1.0)
    activity.cancel()
    slash.release.set()

    with pytest.raises(asyncio.CancelledError):
        await activity

    current = service.get_state("hire_recover")
    assert current is not None
    assert (
        current.effect_state("slash-reconcile:2:1")
        is HireEffectState.ACTION_REQUIRED
    )
    assert runtime._external_mutation_gate.begin_retirement(  # noqa: SLF001
        "tenant-a",
        "agt_recover",
        timeout_seconds=0,
    ) is True
    service.close()


class _FireEffect:
    def execute(self, _state) -> None:
        return None

    def observe(self, _state) -> bool:
        return True


def test_durable_fence_marker_replays_through_real_hire_authority(tmp_path: Path) -> None:
    hire, writer, _projection = _hire_service(tmp_path / "hire")
    admitted = hire.start_hire(_hire_request())
    ingress = EmployeeIngressService(
        writer=writer,
        blob_store=BlobStore(
            tmp_path / "ingress-blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fence-marker-ingress-key-32bytes"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    gate = EmployeeExternalMutationGate()
    authority = JournalFireAuthority(
        writer=writer,
        hire_service=hire,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    fire = EmployeeFireService(
        writer=writer,
        authority=authority,
        effects={name: _FireEffect() for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=gate,
    )

    result = fire.start_fire(
        EmployeeFireRequest(
            employee=admitted.agent_id,
            tenant_key="tenant-a",
            message_id="om_fire_marker_projection",
            chat_id="oc_admin_dm",
            requester_principal_id="ou_admin",
        )
    )

    assert result.phase is FirePhase.ARCHIVED
    assert gate.is_fenced("tenant-a", admitted.agent_id) is True
    assert hire.synchronize_projection().employees[admitted.agent_id].state.value == (
        "archived"
    )
    ingress.close()
    hire.close()


@pytest.mark.parametrize(
    ("effect_id", "effect_type"),
    (
        ("slash-reconcile:2:1", "slash_reconciliation"),
        ("channel-start:2", "employee_channel_start"),
    ),
)
def test_restart_blocks_fire_for_real_executing_hire_mutator(
    tmp_path: Path,
    effect_id: str,
    effect_type: str,
) -> None:
    _seed_active_employee(tmp_path)
    crashed = _recovery_service(_writer(tmp_path, 2))
    crashed.commit_effect_transition(
        "hire_recover",
        effect_id=effect_id,
        effect_type=effect_type,
        next_state=HireEffectState.PREPARED,
    )
    crashed.commit_effect_transition(
        "hire_recover",
        effect_id=effect_id,
        effect_type=effect_type,
        next_state=HireEffectState.EXECUTING,
    )
    crashed.close()

    restarted = _recovery_service(_writer(tmp_path, 3))
    ingress = EmployeeIngressService(
        writer=restarted._writer,  # noqa: SLF001
        blob_store=BlobStore(
            tmp_path / "restart-ingress-blobs",
            AesGcmEncryptionProvider(
                lambda _key: b"fence-marker-ingress-key-32bytes"
            ),
        ),
        ingress_state=IngressProjectionState(),
        active_key_id="k1",
    )
    authority = JournalFireAuthority(
        writer=restarted._writer,  # noqa: SLF001
        hire_service=restarted,
        ingress_service=ingress,
        admin_principal_ids=frozenset({"ou_admin"}),
    )
    calls: list[str] = []

    class _RecordingEffect:
        def execute(self, _state):
            calls.append("execute")

        def observe(self, _state):
            calls.append("observe")
            return True

    fire = EmployeeFireService(
        writer=restarted._writer,  # noqa: SLF001
        authority=authority,
        effects={name: _RecordingEffect() for name in FIRE_EFFECT_ORDER},
        external_mutation_gate=EmployeeExternalMutationGate(),
    )

    with pytest.raises(FireServiceError, match="external mutation outcome"):
        fire.start_fire(
            EmployeeFireRequest(
                employee="agt_recover",
                tenant_key="tenant-a",
                message_id=f"om_restart_pending_{effect_type}",
                chat_id="oc_admin_dm",
                requester_principal_id="ou_admin",
            )
        )

    frames = tuple(restarted._writer.replay())  # noqa: SLF001
    assert calls == []
    assert all(
        event.event_type != "fire.requested"
        for frame in frames
        for event in frame.events
    )
    # The full pre-admission marker is the durable recovery cursor and the
    # permanent no-new-mutations fence.  It must survive even though the old
    # EXECUTING call prevents fire.requested admission.
    assert any(
        event.event_type == "employee.external_mutation_fenced"
        for frame in frames
        for event in frame.events
    )
    ingress.close()
    restarted.close()
