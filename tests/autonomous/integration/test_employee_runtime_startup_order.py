from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.trust.models import ActorKind, TrustZone


def test_hire_readiness_uses_owned_main_bot_send_audit(tmp_path) -> None:
    from src.autonomous.acceptance.main_bot_audit import MainBotSendAuditLog
    from src.autonomous.provisioning.composition import (
        EmployeeDepartmentRuntime,
        RuntimeReadiness,
    )
    from src.autonomous.provisioning.hire_service import HireReadiness

    audit = MainBotSendAuditLog.open(
        tmp_path / "main-bot-audit",
        anchor_path=tmp_path / "main-bot-audit.anchor",
        hmac_key=b"a" * 32,
    )
    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._owned_main_bot_send_audit = audit  # noqa: SLF001
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        readiness=lambda: HireReadiness(True, ())
    )

    try:
        assert runtime.hire_readiness() == RuntimeReadiness(True, ())
    finally:
        audit.close()


def test_dispatch_soft_failure_preserves_owned_main_bot_audit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.autonomous.provisioning.composition as composition
    from src.autonomous.acceptance.main_bot_audit import MainBotSendAuditLog

    audit = MainBotSendAuditLog.open(
        tmp_path / "main-bot-audit",
        anchor_path=tmp_path / "main-bot-audit.anchor",
        hmac_key=b"a" * 32,
    )
    runtime = composition.EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._owned_main_bot_send_audit = audit  # noqa: SLF001
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        synchronize_projection=lambda: SimpleNamespace(),
    )
    runtime._writer = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._ingress = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._data = SimpleNamespace(service=object())  # type: ignore[assignment]  # noqa: SLF001
    runtime._channels = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._context_service = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox = object()  # type: ignore[assignment]  # noqa: SLF001
    runtime._team_runtime = SimpleNamespace(employee_activation_guard=lambda: None)  # noqa: SLF001
    runtime._environment_provider = lambda _authority: None  # noqa: SLF001
    settings = SimpleNamespace(
        autonomous_employee_storage_base=str(tmp_path),
        autonomous_employee_queue_per_employee_limit=1,
        autonomous_employee_queue_per_team_limit=1,
        autonomous_employee_queue_global_limit=1,
        autonomous_employee_system_prompt_token_reserve=1,
        autonomous_context_retry_base_seconds=0.1,
        autonomous_context_retry_max_seconds=1.0,
    )

    def fail_router(**_kwargs):
        raise RuntimeError("projection broken")

    monkeypatch.setattr(composition, "DurableEmployeeIngressRouter", fail_router)

    try:
        runtime._compose_dispatch(settings, membership_health=object())  # noqa: SLF001

        assert runtime._execution_blockers == ("employee_gateway",)  # noqa: SLF001
        assert runtime.main_bot_outbound_audit is audit
    finally:
        audit.close()


def test_team_recovery_waits_for_shared_dispatch_projections() -> None:
    calls: list[str] = []
    empty_state = SimpleNamespace(by_acceptance_id={})
    workforce = SimpleNamespace(employees={}, bot_principals={})

    class _Service:
        projection_state = workforce

        def recover(self):
            return SimpleNamespace(states={})

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
