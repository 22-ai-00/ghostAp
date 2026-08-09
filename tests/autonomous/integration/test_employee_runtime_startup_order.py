from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.autonomous.migration.employee_workspace_v1 import (
    EmployeeWorkspaceMigrationError,
    EmployeeWorkspaceV1Migrator,
)
from src.autonomous.supervisor import (
    EMPLOYEE_RECOVERY_ORDER,
    EmployeeLifecycleSupervisor,
    EmployeeRecoverySnapshot,
)
from src.autonomous.workspace import EmployeeWorkspaceProjector
from src.trust.models import ActorKind, TrustZone
from tests.autonomous.workforce_helpers import (
    commit_events,
    employee_created,
    make_writer,
    replay_state,
)


def test_employee_recovery_and_shutdown_use_fixed_inverse_order() -> None:
    calls = []
    closes = []
    recoverers = {
        stage: (lambda _snapshot, stage=stage: calls.append(stage))
        for stage in EMPLOYEE_RECOVERY_ORDER
    }
    closers = {
        stage: (lambda stage=stage: closes.append(stage))
        for stage in EMPLOYEE_RECOVERY_ORDER
    }
    supervisor = EmployeeLifecycleSupervisor(recoverers, closers)
    report = supervisor.recover(EmployeeRecoverySnapshot((), 0, ""))
    assert report.ready is True
    assert tuple(calls) == EMPLOYEE_RECOVERY_ORDER
    assert supervisor.shutdown() == tuple(reversed(EMPLOYEE_RECOVERY_ORDER))
    assert tuple(closes) == tuple(reversed(EMPLOYEE_RECOVERY_ORDER))


@pytest.mark.parametrize(
    "failed_stage",
    ("workspace_projection", "data_projection", "actor_mailboxes"),
)
def test_employee_ready_fails_closed_at_first_broken_dependency(failed_stage) -> None:
    calls = []

    def recover(stage):
        calls.append(stage)
        if stage == failed_stage:
            raise RuntimeError("broken")

    supervisor = EmployeeLifecycleSupervisor(
        {
            stage: (lambda _snapshot, stage=stage: recover(stage))
            for stage in EMPLOYEE_RECOVERY_ORDER
        }
    )
    report = supervisor.recover(EmployeeRecoverySnapshot((), 0, ""))
    assert report.ready is False
    assert report.blocker == failed_stage
    assert "admission_open" not in calls


def test_legacy_employee_workspace_migration_is_canonical_and_idempotent(tmp_path) -> None:
    writer = make_writer(tmp_path)
    state = SimpleNamespace()
    projection = replay_state(writer)
    commit_events(writer, projection, employee_created("agt_legacy", "Legacy"))
    projector = EmployeeWorkspaceProjector(
        tmp_path / "agents", state_provider=lambda: replay_state(writer)
    )
    migrator = EmployeeWorkspaceV1Migrator(projector)
    first = migrator.migrate((("tenant_1", "agt_legacy"),))
    agents = tmp_path / "agents" / "agt_legacy" / "workspace" / "AGENTS.md"
    agents.write_text("legacy mutable escalation", encoding="utf-8")
    second = migrator.migrate((("tenant_1", "agt_legacy"),))
    assert first.snapshots == second.snapshots
    assert "legacy mutable escalation" not in agents.read_text(encoding="utf-8")
    del state
    writer.close()


def test_legacy_employee_workspace_migration_rejects_symlink(tmp_path) -> None:
    writer = make_writer(tmp_path)
    projection = replay_state(writer)
    commit_events(writer, projection, employee_created("agt_legacy", "Legacy"))
    root = tmp_path / "agents"
    root.mkdir()
    (root / "agt_legacy").symlink_to(tmp_path / "outside")
    migrator = EmployeeWorkspaceV1Migrator(
        EmployeeWorkspaceProjector(root, state_provider=lambda: replay_state(writer))
    )
    with pytest.raises(EmployeeWorkspaceMigrationError, match="unsafe"):
        migrator.migrate((("tenant_1", "agt_legacy"),))
    writer.close()


def test_production_data_recovery_requeues_pending_knowledge() -> None:
    calls: list[str] = []

    class _DataService:
        def rebuild_projection(self) -> None:
            calls.append("data_projection")

    class _Knowledge:
        def recover(self) -> int:
            calls.append("knowledge_queue")
            return 1

    class _Data:
        service = _DataService()
        state = SimpleNamespace(data_authority=SimpleNamespace(mode="canonical"))
        knowledge_service = _Knowledge()

        def rebuild_all(self) -> None:
            calls.append("workspace_projection")

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    runtime._data = _Data()  # type: ignore[assignment]  # noqa: SLF001
    runtime._recover_employee_data(SimpleNamespace(employees={}))  # noqa: SLF001

    assert calls == ["data_projection", "workspace_projection", "knowledge_queue"]


def test_team_recovery_waits_for_shared_dispatch_projections() -> None:
    calls: list[str] = []
    empty_state = SimpleNamespace(by_acceptance_id={})
    workforce = SimpleNamespace(employees={}, bot_principals={})

    class _Service:
        projection_state = workforce

        def recover(self):
            return SimpleNamespace(states={})

        def recover_manifest_reauthorizations(self):
            return ()

        def list_states(self):
            return ()

        def mark_runtime_recovered(self):
            calls.append("admission_open")

    class _Ingress:
        state = empty_state

        def rebuild_projection(self):
            calls.append("ingress_projection")

        def gc_terminal_payloads(self):
            return 0

    class _Router:
        state = empty_state

        def rebuild_projection(self):
            calls.append("router_projection")

        def recover_terminal_attachments(self):
            return 0

    class _Outbox:
        def rebuild_projection(self):
            calls.append("outbox_projection")

    class _Dispatch:
        employee_runtime = None

        def recover_incomplete_attempts(self):
            calls.append("dispatch_projection")

        def reconcile_terminal_snapshots(self):
            return 0

    class _Team:
        def recover(self):
            calls.append("team_coordinator")
            return 0

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    runtime._service = _Service()  # type: ignore[assignment]  # noqa: SLF001
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Router()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = _Outbox()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = _Dispatch()  # type: ignore[assignment]  # noqa: SLF001
    runtime._team = _Team()  # type: ignore[assignment]  # noqa: SLF001
    runtime.recover()

    assert calls.index("dispatch_projection") < calls.index("team_coordinator")
    assert calls[-1] == "admission_open"


def test_background_recovery_failure_closes_employee_admission_only() -> None:
    from unittest.mock import MagicMock

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    service = SimpleNamespace(stop_admission=MagicMock())
    runtime._service = service  # type: ignore[assignment]  # noqa: SLF001
    runtime._core_recovered = True  # noqa: SLF001

    runtime.fail_recovery("membership_recovery")

    assert runtime._core_recovered is False  # noqa: SLF001
    assert runtime._execution_blockers == ("membership_recovery",)  # noqa: SLF001
    service.stop_admission.assert_called_once_with()


def test_production_dispatch_tick_reaps_idle_employee_sessions() -> None:
    calls: list[str] = []
    empty_state = SimpleNamespace(by_acceptance_id={})

    class _Projection:
        state = empty_state

        def rebuild_projection(self):
            return None

    class _Ingress(_Projection):
        def gc_terminal_payloads(self):
            return 0

    class _Runtime:
        def sweep_idle(self):
            calls.append("sweep_idle")
            return 1

    class _Dispatch:
        employee_runtime = _Runtime()

        def dispatch_next(self):
            return None

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Projection()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = _Dispatch()  # type: ignore[assignment]  # noqa: SLF001

    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == ["sweep_idle"]


def test_idle_dispatch_ticks_refresh_on_head_change_and_throttle_blob_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    head = {"sequence": 0, "hash": ""}
    calls = {"ingress_rebuild": 0, "router_rebuild": 0, "blob_gc": 0}

    class _Guard:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    class _Projection:
        def __init__(self, name: str) -> None:
            self._name = name
            self.state = SimpleNamespace(
                by_acceptance_id={},
                cursor_sequence=0,
                cursor_hash="",
            )

        def rebuild_projection(self):
            if (
                self.state.cursor_sequence,
                self.state.cursor_hash,
            ) == (head["sequence"], head["hash"]):
                return
            calls[f"{self._name}_rebuild"] += 1
            self.state.cursor_sequence = head["sequence"]
            self.state.cursor_hash = head["hash"]

        def synchronize_projection_unlocked(self):
            if (
                self.state.cursor_sequence,
                self.state.cursor_hash,
            ) != (head["sequence"], head["hash"]):
                self.rebuild_projection()

    class _Ingress(_Projection):
        def employee_dispatch_guard(self, *, router):
            assert router is runtime._router  # noqa: SLF001
            return _Guard()

        def gc_terminal_payloads(self):
            calls["blob_gc"] += 1
            return 0

    class _Dispatch:
        employee_runtime = None

        def dispatch_next(self):
            return None

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    monkeypatch.setattr(
        "src.autonomous.provisioning.composition.time.monotonic",
        lambda: now[0],
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress("ingress")  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Projection("router")  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = _Dispatch()  # type: ignore[assignment]  # noqa: SLF001

    runtime._drain_employee_dispatch_once()  # noqa: SLF001
    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == {"ingress_rebuild": 0, "router_rebuild": 0, "blob_gc": 1}

    head.update(sequence=1, hash="frame-1")
    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == {"ingress_rebuild": 1, "router_rebuild": 1, "blob_gc": 1}

    now[0] += 60.0
    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == {"ingress_rebuild": 1, "router_rebuild": 1, "blob_gc": 2}


def test_production_dispatch_projects_group_context_before_routing_and_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    acceptance_id = "acc_group_message"
    pending = SimpleNamespace(disposition=None)

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def rebuild_projection(self):
            return None

        def get_payload(self, observed_acceptance_id: str):
            assert observed_acceptance_id == acceptance_id
            return SimpleNamespace(
                normalized_parts=({"type": "message"},),
            )

        def gc_terminal_payloads(self):
            calls.append("gc_payload")
            return 0

    class _Router:
        state = SimpleNamespace(by_acceptance_id={})

        def rebuild_projection(self):
            return None

        def route(self, routed_acceptance_id: str):
            assert routed_acceptance_id == acceptance_id
            calls.append("route")

    class _Dispatch:
        employee_runtime = None

        def dispatch_next(self):
            calls.append("dispatch")
            return None

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Router()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = _Dispatch()  # type: ignore[assignment]  # noqa: SLF001
    monkeypatch.setattr(
        runtime,
        "_record_employee_ingress_group_event",
        lambda observed_acceptance_id: calls.append("project_group_context"),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_control_ingress",
        lambda observed_acceptance_id: calls.append("handle_control") or False,
    )
    monkeypatch.setattr(
        runtime,
        "_managed_employee_ingress_trust",
        lambda _record, _payload: SimpleNamespace(
            zone=TrustZone.MANAGED_AGENT_GROUP,
            actor=ActorKind.OWNER,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_main_bot_group_command_ingress",
        lambda observed_acceptance_id: calls.append("command_gate") or False,
    )

    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == [
        "handle_control",
        "command_gate",
        "project_group_context",
        "route",
        "dispatch",
        "gc_payload",
    ]


def test_production_dispatch_ignores_main_bot_group_command_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    acceptance_id = "acc_ambient_message"
    pending = SimpleNamespace(
        disposition=None,
        metadata=SimpleNamespace(
            event_type="im.message.receive_v1",
            action_identity="",
        ),
    )

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def rebuild_projection(self):
            return None

        def get_payload(self, observed_acceptance_id: str):
            assert observed_acceptance_id == acceptance_id
            return SimpleNamespace(
                normalized_parts=(
                    {
                        "type": "message",
                        "chat_type": "group",
                        "content": {"text": "/help"},
                    },
                )
            )

        def record_disposition(
            self,
            observed_acceptance_id: str,
            *,
            state: str,
            reason_code: str,
        ):
            assert observed_acceptance_id == acceptance_id
            calls.append("command_gate")
            pending.disposition = SimpleNamespace(state=state, reason_code=reason_code)

        def gc_terminal_payloads(self):
            calls.append("gc_payload")
            return 0

    class _Router:
        state = SimpleNamespace(by_acceptance_id={})

        def rebuild_projection(self):
            return None

        def route(self, _acceptance_id: str):
            calls.append("route")

    class _Dispatch:
        employee_runtime = None

        def dispatch_next(self):
            calls.append("dispatch")
            return None

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Router()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = _Dispatch()  # type: ignore[assignment]  # noqa: SLF001
    monkeypatch.setattr(
        runtime,
        "_record_employee_ingress_group_event",
        lambda _acceptance_id: calls.append("project_group_context"),
    )
    monkeypatch.setattr(runtime, "_handle_control_ingress", lambda _acceptance_id: False)
    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == ["command_gate", "dispatch", "gc_payload"]
    assert pending.disposition.reason_code == "main_bot_group_command"


def test_employee_group_projection_maps_app_open_id_to_owner_principal() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
    from src.autonomous.provisioning.hire_state import HirePhase

    acceptance_id = "acc_app_scoped_sender"
    metadata = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        event_id="evt_employee",
        chat_id="oc_team",
        message_id="om_message",
        thread_root_message_id="",
        sender_principal_id="ou_employee_app",
    )
    part = {
        "type": "message",
        "message_type": "text",
        "chat_type": "group",
        "content": {"text": "/help"},
        "sender_id": "ou_employee_app",
        "sender_union_id": "on_owner",
        "sender_id_type": "open_id",
        "sender_type": "user",
        "sender_tenant_key": "tenant_1",
        "feishu_thread_id": "",
    }
    ingress = SimpleNamespace(
        state=SimpleNamespace(
            by_acceptance_id={acceptance_id: SimpleNamespace(metadata=metadata)}
        ),
        get_payload=lambda _acceptance_id: SimpleNamespace(normalized_parts=(part,)),
    )
    state = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        phase=HirePhase.ACTIVE,
        requester_principal_id="ou_main_app_owner",
        requester_union_id="on_owner",
    )
    service = SimpleNamespace(
        synchronize_projection=lambda: None,
        list_states=lambda: (state,),
    )
    published: list[object] = []
    ledger = SimpleNamespace(publish=lambda **kwargs: published.append(kwargs))

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = ingress  # type: ignore[assignment]  # noqa: SLF001
    runtime._service = service  # type: ignore[assignment]  # noqa: SLF001
    runtime._group_ledger = ledger  # type: ignore[assignment]  # noqa: SLF001

    assert runtime._record_employee_ingress_group_event(acceptance_id) is True  # noqa: SLF001
    payload = published[0]["payload"]  # type: ignore[index]
    assert payload.sender_id == "ou_main_app_owner"
    assert payload.sender_id_type == "open_id"


@pytest.mark.asyncio
async def test_channel_acceptance_callback_does_not_repeat_group_projection_after_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    calls: list[str] = []
    acceptance_id = "acc_group_message"

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    monkeypatch.setattr(
        runtime,
        "_record_employee_ingress_group_event",
        lambda observed_acceptance_id: calls.append("project_group_context"),
    )
    monkeypatch.setattr(
        runtime,
        "_handle_control_ingress",
        lambda observed_acceptance_id: calls.append("handle_control") or False,
    )

    await runtime._handle_channel_event(  # noqa: SLF001
        "intent_1",
        1,
        {
            "event": "durableIngressAccepted",
            "data": {"acceptance_id": acceptance_id},
        },
    )

    assert calls == ["handle_control"]


def test_enabled_runtime_fails_closed_when_group_ledger_cannot_anchor() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)

    with pytest.raises(RuntimeError, match="group ledger"):
        runtime.record_group_event(
            tenant_key="tenant_1",
            chat_id="oc_team",
            thread_id="",
            message_id="om_event",
            sender_id="ou_user",
            text="team task",
        )
