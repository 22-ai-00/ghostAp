from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

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
        lambda observed_acceptance_id, **_kwargs: calls.append("handle_control")
        or False,
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
        lambda observed_acceptance_id, **_kwargs: calls.append("command_gate")
        or False,
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


def test_production_dispatch_routes_owner_p2p_without_group_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    acceptance_id = "acc_owner_p2p"
    pending = SimpleNamespace(disposition=None)

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def rebuild_projection(self):
            return None

        def get_payload(self, _acceptance_id: str):
            return SimpleNamespace(normalized_parts=({"type": "message"},))

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
        "_owner_p2p_requester",
        lambda _record, _payload: "ou_owner",
    )
    monkeypatch.setattr(
        runtime,
        "_managed_employee_ingress_trust",
        lambda _record, _payload: runtime._unknown_employee_ingress_trust(),  # noqa: SLF001
    )
    monkeypatch.setattr(
        runtime,
        "_handle_control_ingress",
        lambda _acceptance_id, **_kwargs: calls.append("handle_control") or False,
    )
    monkeypatch.setattr(
        runtime,
        "_handle_main_bot_group_command_ingress",
        lambda _acceptance_id, **_kwargs: calls.append("command_gate") or False,
    )
    monkeypatch.setattr(
        runtime,
        "_record_employee_ingress_group_event",
        lambda _acceptance_id: calls.append("project_group_context"),
    )

    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == ["handle_control", "route", "dispatch", "gc_payload"]


@pytest.mark.parametrize(
    "command",
    [
        "/help",
        "/status",
        "/status details",
        "/task unaddressed",
        "/tasks",
        "@_user_1 /task addressed elsewhere",
        "@_user_1 /help",
    ],
)
def test_production_dispatch_ignores_main_bot_group_command_observation(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
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
                        "content": {"text": command},
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
    monkeypatch.setattr(
        runtime,
        "_handle_control_ingress",
        lambda _acceptance_id, **_kwargs: False,
    )
    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == ["command_gate", "dispatch", "gc_payload"]
    assert pending.disposition.reason_code == "main_bot_group_command"


def test_production_dispatch_retries_indeterminate_targeted_group_task_once_per_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.ingress.targeted_task import (
        TargetedTaskParseResult,
        TargetedTaskState,
        targeted_group_task_digest,
    )
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    calls: list[str] = []
    acceptance_id = "acc_targeted_task"
    pending = SimpleNamespace(
        disposition=None,
        metadata=SimpleNamespace(
            event_type="im.message.receive_v1",
            action_identity="",
        ),
    )
    task = TargetedTaskParseResult(
        TargetedTaskState.TARGETED_VALID,
        description="finish audit",
        input_digest=targeted_group_task_digest("finish audit"),
    )

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def rebuild_projection(self):
            return None

        def get_payload(self, _acceptance_id: str):
            return SimpleNamespace(
                normalized_parts=(
                    {
                        "type": "message",
                        "chat_type": "group",
                        "content": {"text": "@_user_1 /task finish audit"},
                        "mentions": (
                            {
                                "key": "@_user_1",
                                "open_id": "ou_bot_alpha",
                                "tenant_key": "tenant_1",
                            },
                        ),
                    },
                )
            )

        def record_disposition(self, *_args, **_kwargs):
            pytest.fail("authorized targeted task was consumed as a control")

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

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._router = _Router()  # type: ignore[assignment]  # noqa: SLF001
    runtime._dispatch = _Dispatch()  # type: ignore[assignment]  # noqa: SLF001
    classifications = 0

    classifications_by_drain = (
        TargetedTaskParseResult(TargetedTaskState.INDETERMINATE),
        task,
    )

    def classify_once(_record, _payload):
        nonlocal classifications
        classifications += 1
        if classifications > len(classifications_by_drain):
            pytest.fail("an ingress drain classified the targeted task twice")
        return classifications_by_drain[classifications - 1]

    monkeypatch.setattr(
        runtime,
        "_authorized_targeted_group_task",
        classify_once,
    )
    monkeypatch.setattr(
        runtime,
        "_managed_employee_ingress_trust",
        lambda _record, _payload: runtime._unknown_employee_ingress_trust(),  # noqa: SLF001
    )
    monkeypatch.setattr(
        runtime,
        "_record_employee_ingress_group_event",
        lambda _value: calls.append("project_group_context"),
    )

    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert pending.disposition is None
    assert calls == ["dispatch", "gc_payload"]
    assert classifications == 1

    calls.clear()
    runtime._drain_employee_dispatch_once()  # noqa: SLF001

    assert calls == ["project_group_context", "route", "dispatch"]
    assert classifications == 2


def test_targeted_group_task_authority_ignores_unavailable_ingress_payload() -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    runtime = EmployeeDepartmentRuntime()
    runtime._router = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        classify_targeted_group_task=lambda *_args: pytest.fail(
            "unavailable payload reached Router classification"
        )
    )

    assert runtime._authorized_targeted_group_task(  # noqa: SLF001
        SimpleNamespace(metadata=SimpleNamespace()),
        None,
    ) is None


def test_targeted_group_task_usage_is_anchored_before_ingress_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.ingress.targeted_task import (
        TargetedTaskParseResult,
        TargetedTaskState,
    )
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    calls: list[str] = []
    acceptance_id = "acc_task_usage"
    metadata = SimpleNamespace(
        event_type="im.message.receive_v1",
        action_identity="",
        tenant_key="tenant_1",
        agent_id="agt_alpha",
    )
    pending = SimpleNamespace(disposition=None, metadata=metadata)

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def get_payload(self, _acceptance_id: str):
            return SimpleNamespace(
                normalized_parts=(
                    {
                        "type": "message",
                        "chat_type": "group",
                        "content": {"text": "@_user_1 /task"},
                    },
                )
            )

        def record_disposition(self, *_args, **kwargs):
            calls.append(f"disposition:{kwargs['reason_code']}")

    lifecycle = SimpleNamespace(
        task_usage_response=lambda **_kwargs: calls.append("outbox")
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_lifecycle = lifecycle  # type: ignore[assignment]  # noqa: SLF001
    monkeypatch.setattr(
        "src.autonomous.provisioning.composition._bound_remote_coordinates",
        lambda _metadata, _part: ("oc_team", "om_message", "om_root"),
    )
    monkeypatch.setattr(
        runtime,
        "_drain_employee_outbox_once",
        lambda: calls.append("drain") or True,
    )

    assert runtime._handle_main_bot_group_command_ingress(  # noqa: SLF001
        acceptance_id,
        targeted_group_task=TargetedTaskParseResult(
            TargetedTaskState.TARGETED_INVALID
        ),
    )
    assert calls == [
        "outbox",
        "disposition:task_invalid_arguments",
        "drain",
    ]


def test_targeted_group_task_usage_does_not_consume_ingress_when_outbox_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.ingress.targeted_task import (
        TargetedTaskParseResult,
        TargetedTaskState,
    )
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    calls: list[str] = []
    acceptance_id = "acc_task_usage_outbox_failure"
    metadata = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        event_type="im.message.receive_v1",
        action_identity="",
    )
    pending = SimpleNamespace(disposition=None, metadata=metadata)

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def get_payload(self, _acceptance_id: str):
            return SimpleNamespace(
                normalized_parts=(
                    {
                        "type": "message",
                        "chat_type": "group",
                        "content": {"text": "@_user_1 /task"},
                    },
                )
            )

        def record_disposition(self, *_args, **_kwargs):
            calls.append("disposition")

    def fail_outbox(**_kwargs):
        calls.append("outbox")
        raise RuntimeError("injected outbox failure")

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_lifecycle = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        task_usage_response=fail_outbox
    )
    monkeypatch.setattr(
        "src.autonomous.provisioning.composition._bound_remote_coordinates",
        lambda _metadata, _part: ("oc_team", "om_message", "om_root"),
    )

    with pytest.raises(RuntimeError, match="injected outbox failure"):
        runtime._handle_main_bot_group_command_ingress(  # noqa: SLF001
            acceptance_id,
            targeted_group_task=TargetedTaskParseResult(
                TargetedTaskState.TARGETED_INVALID
            ),
        )

    assert calls == ["outbox"]


def test_targeted_group_task_usage_drains_anchored_response_after_disposition_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.ingress.service import IngressConflictError
    from src.autonomous.ingress.targeted_task import (
        TargetedTaskParseResult,
        TargetedTaskState,
    )
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    calls: list[str] = []
    acceptance_id = "acc_task_usage_disposition_race"
    metadata = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        event_type="im.message.receive_v1",
        action_identity="",
    )
    pending = SimpleNamespace(disposition=None, metadata=metadata)

    class _Ingress:
        state = SimpleNamespace(by_acceptance_id={acceptance_id: pending})

        def get_payload(self, _acceptance_id: str):
            return SimpleNamespace(
                normalized_parts=(
                    {
                        "type": "message",
                        "chat_type": "group",
                        "content": {"text": "@_user_1 /task"},
                    },
                )
            )

        def record_disposition(self, *_args, **_kwargs):
            calls.append("disposition")
            raise IngressConflictError("injected disposition race")

    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = _Ingress()  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_lifecycle = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        task_usage_response=lambda **_kwargs: calls.append("outbox")
    )
    monkeypatch.setattr(
        "src.autonomous.provisioning.composition._bound_remote_coordinates",
        lambda _metadata, _part: ("oc_team", "om_message", "om_root"),
    )
    monkeypatch.setattr(
        runtime,
        "_drain_employee_outbox_once",
        lambda: calls.append("drain") or True,
    )

    assert runtime._handle_main_bot_group_command_ingress(  # noqa: SLF001
        acceptance_id,
        targeted_group_task=TargetedTaskParseResult(
            TargetedTaskState.TARGETED_INVALID
        ),
    )
    assert calls == ["outbox", "disposition", "drain"]


def test_generic_control_lane_leaves_targeted_group_task_for_command_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.autonomous.ingress.targeted_task import (
        TargetedTaskParseResult,
        TargetedTaskState,
        targeted_group_task_digest,
    )
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime

    acceptance_id = "acc_targeted_control_bypass"
    payload = SimpleNamespace(
        normalized_parts=(
            {
                "type": "message",
                "chat_type": "group",
                "content": {"text": "@_user_1 /task finish audit"},
                "mentions": (
                    {
                        "key": "@_user_1",
                        "open_id": "ou_bot_alpha",
                        "tenant_key": "tenant_1",
                    },
                ),
            },
        )
    )
    record = SimpleNamespace(disposition=None, metadata=SimpleNamespace())
    runtime = EmployeeDepartmentRuntime()
    runtime._ingress = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        rebuild_projection=lambda: None,
        state=SimpleNamespace(by_acceptance_id={acceptance_id: record}),
        get_payload=lambda _value: payload,
    )
    task = TargetedTaskParseResult(
        TargetedTaskState.TARGETED_VALID,
        description="finish audit",
        input_digest=targeted_group_task_digest("finish audit"),
    )
    monkeypatch.setattr(
        runtime,
        "_authorized_targeted_group_task",
        lambda _record, _payload: task,
    )

    assert runtime._handle_control_ingress(acceptance_id) is False  # noqa: SLF001


def test_employee_group_projection_preserves_source_open_id_for_partial_context() -> None:
    from src.autonomous.context import (
        ContextUnavailableReason,
        ThreadContextConfig,
    )
    from src.autonomous.context.group_ledger import GroupContextLedger
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
    from src.autonomous.provisioning.hire_state import HirePhase
    from tests.autonomous.integration.test_employee_context_service import _request

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
    assert payload.sender_id == "ou_employee_app"
    assert payload.sender_id_type == "open_id"
    record = SimpleNamespace(
        message_id="om_message",
        chat_id="oc_team",
        thread_id="",
        payload_ref=object(),
    )
    partial_ledger = object.__new__(GroupContextLedger)
    partial_ledger._config = ThreadContextConfig()  # noqa: SLF001
    partial_ledger._blobs = SimpleNamespace(  # noqa: SLF001
        read=lambda _ref: payload.to_bytes()
    )
    partial_ledger.window = lambda **_kwargs: SimpleNamespace(  # type: ignore[method-assign]
        records=(record,)
    )

    snapshot = partial_ledger.assemble_partial(
        _request(
            agent_id="agt_alpha",
            chat_id="oc_team",
            thread_root_message_id="om_message",
            feishu_thread_id="",
            current_message_id="om_message",
            requester_principal_id="ou_main_app_owner",
            source_requester_principal_id="ou_employee_app",
        ),
        warning_reason=ContextUnavailableReason.ORDERING,
    )

    assert snapshot.thread_messages[0].sender_id == "ou_employee_app"


@pytest.mark.parametrize(
    ("projected_owner", "hire_union", "transport_current"),
    [
        ("ou_rotated_owner", "on_owner", True),
        ("ou_original_owner", "on_different", True),
        ("ou_original_owner", "on_owner", False),
    ],
)
def test_owner_p2p_status_fails_closed_on_identity_or_transport_drift(
    projected_owner: str,
    hire_union: str,
    transport_current: bool,
) -> None:
    from src.autonomous.context.runtime import RuntimeRequesterChatAcl
    from src.autonomous.domain import (
        BotPrincipal,
        EmployeeDefinition,
        EmployeeState,
        WorkerType,
    )
    from src.autonomous.ingress.models import (
        EmployeeIngressMetadata,
        EmployeeIngressPayload,
    )
    from src.autonomous.journal.projections import ProjectionState
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
    from src.autonomous.provisioning.hire_state import HirePhase

    payload = EmployeeIngressPayload(
        schema_version=1,
        envelope_id="ing_" + "1" * 64,
        normalized_parts=(
            {
                "type": "message",
                "message_type": "text",
                "chat_type": "p2p",
                "content": {"text": "/status"},
                "sender_id": "ou_employee_app_owner",
                "sender_union_id": "on_owner",
                "sender_id_type": "open_id",
                "sender_type": "user",
                "sender_tenant_key": "tenant_1",
                "feishu_thread_id": "",
            },
        ),
        attachment_descriptors=(),
    )
    metadata = EmployeeIngressMetadata(
        schema_version=1,
        envelope_id=payload.envelope_id,
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
        channel_generation=3,
        connection_id="conn_alpha",
        event_id="evt_owner_drift",
        message_id="om_owner_drift",
        event_type="im.message.receive_v1",
        action_identity="",
        chat_id="oc_owner_p2p",
        thread_root_message_id="",
        sender_principal_id="ou_employee_app_owner",
        received_at="2026-08-10T00:00:00Z",
        semantic_digest=payload.payload_sha256,
        payload_sha256=payload.payload_sha256,
        payload_size_bytes=payload.canonical_size_bytes,
        attachment_count=0,
        attachment_total_bytes=0,
    )
    projection = ProjectionState()
    projection.employees["agt_alpha"] = EmployeeDefinition(
        agent_id="agt_alpha",
        tenant_key="tenant_1",
        owner_principal_id=projected_owner,
        worker_type=WorkerType.VISIBLE,
        state=EmployeeState.ACTIVE,
        bot_principal_id="bot_alpha",
    )
    projection.bot_principals["bot_alpha"] = BotPrincipal(
        bot_principal_id="bot_alpha",
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        app_id="cli_alpha",
        credential_ref="cred_alpha",
    )
    hire_state = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        phase=HirePhase.ACTIVE,
        requester_principal_id="ou_original_owner",
        requester_union_id=hire_union,
    )
    service = SimpleNamespace(
        synchronize_projection=lambda: projection,
        list_states=lambda: (hire_state,),
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._service = service  # type: ignore[assignment]  # noqa: SLF001
    runtime._context_acl = RuntimeRequesterChatAcl(  # noqa: SLF001
        allowed_requesters=("ou_original_owner",),
    )
    runtime._employee_ingress_transport_is_current = MagicMock(  # noqa: SLF001
        return_value=transport_current
    )
    acceptance_id = "acc_owner_status_denied"
    record = SimpleNamespace(disposition=None, metadata=metadata)
    ingress = MagicMock()
    ingress.state = SimpleNamespace(
        by_acceptance_id={acceptance_id: record}
    )
    ingress.get_payload.return_value = payload
    lifecycle = MagicMock()
    dispatch = MagicMock()
    runtime._ingress = ingress  # type: ignore[assignment]  # noqa: SLF001
    runtime._outbox_lifecycle = lifecycle  # noqa: SLF001
    runtime._dispatch = dispatch  # noqa: SLF001

    assert runtime._owner_p2p_requester(  # noqa: SLF001
        record,
        payload,
    ) is None
    assert runtime._handle_control_ingress(acceptance_id) is True  # noqa: SLF001
    ingress.record_disposition.assert_called_once_with(
        acceptance_id,
        state="ignored",
        reason_code="authority_denied",
    )
    lifecycle.status_response.assert_not_called()
    dispatch.employee_runtime.inspect.assert_not_called()


@pytest.mark.parametrize(
    ("field_name", "drifted"),
    [
        ("app_id", "cli_rotated"),
        ("bot_principal_id", "bot_rotated"),
        ("channel_generation", 4),
        ("connection_id", "conn_rotated"),
        ("channel_identity_app_id", "cli_rotated"),
    ],
)
def test_union_owner_resolution_binds_durable_hire_transport(
    field_name: str,
    drifted: object,
) -> None:
    from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
    from src.autonomous.provisioning.hire_state import HirePhase

    state = SimpleNamespace(
        tenant_key="tenant_1",
        agent_id="agt_alpha",
        phase=HirePhase.ACTIVE,
        requester_principal_id="ou_owner",
        requester_union_id="on_owner",
        app_id="cli_alpha",
        bot_principal_id="bot_alpha",
        channel_generation=3,
        channel_connection_id="conn_alpha",
        channel_identity_app_id="cli_alpha",
    )
    runtime = EmployeeDepartmentRuntime()
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        synchronize_projection=lambda: None,
        list_states=lambda: (state,),
    )
    values = {
        "tenant_key": "tenant_1",
        "agent_id": "agt_alpha",
        "owner_principal_id": "ou_owner",
        "sender_principal_id": "ou_employee_app_owner",
        "sender_union_id": "on_owner",
        "app_id": "cli_alpha",
        "bot_principal_id": "bot_alpha",
        "channel_generation": 3,
        "connection_id": "conn_alpha",
        "channel_identity_app_id": "cli_alpha",
    }

    assert runtime._resolve_employee_requester_principal(**values) == "ou_owner"  # noqa: SLF001
    values[field_name] = drifted
    assert runtime._resolve_employee_requester_principal(**values) is None  # noqa: SLF001


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
        lambda observed_acceptance_id, **_kwargs: calls.append("handle_control")
        or False,
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
