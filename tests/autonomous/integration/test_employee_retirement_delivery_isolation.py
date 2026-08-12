from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.autonomous.domain import EmployeeState
from src.autonomous.provisioning.composition import EmployeeDepartmentRuntime
from src.autonomous.provisioning.fire_state import FireEffectState, FirePhase
from src.autonomous.provisioning.hire_state import HirePhase
from src.autonomous.supervisor.channel_models import ChannelProcessState
from src.autonomous.supervisor.employee_channels import ChannelLaunchUnavailable


def _hire(agent_id: str) -> SimpleNamespace:
    suffix = agent_id.removeprefix("agt_")
    return SimpleNamespace(
        tenant_key="tenant_1",
        agent_id=agent_id,
        phase=HirePhase.RETIRING,
        bot_principal_id=f"bot_{suffix}",
        app_id=f"cli_{suffix}",
        credential_ref=f"cred_{suffix}",
        channel_generation=3,
        channel_identity_app_id=f"cli_{suffix}",
        channel_connection_id=f"conn_{suffix}",
    )


def _status(hire: SimpleNamespace, state: ChannelProcessState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        tenant_key=hire.tenant_key,
        agent_id=hire.agent_id,
        bot_principal_id=hire.bot_principal_id,
        app_id=hire.app_id,
        generation=hire.channel_generation,
        identity={"app_id": hire.app_id},
        ready_metadata={"connection_id": hire.channel_connection_id},
        delivery_only=True,
    )


@pytest.mark.parametrize("first_failure", ["launch", "readiness"])
def test_failed_retirement_channel_does_not_block_healthy_outbox_drain(
    first_failure: str,
) -> None:
    hires = (_hire("agt_alpha"), _hire("agt_beta"))
    employees = tuple(
        SimpleNamespace(
            tenant_key=hire.tenant_key,
            agent_id=hire.agent_id,
            bot_principal_id=hire.bot_principal_id,
            state=EmployeeState.RETIRING,
        )
        for hire in hires
    )
    principals = tuple(
        SimpleNamespace(
            tenant_key=hire.tenant_key,
            agent_id=hire.agent_id,
            bot_principal_id=hire.bot_principal_id,
            app_id=hire.app_id,
            credential_ref=hire.credential_ref,
        )
        for hire in hires
    )
    fire_states = tuple(
        SimpleNamespace(
            tenant_key=hire.tenant_key,
            agent_id=hire.agent_id,
            phase=FirePhase.RETIRING,
            bot_principal_id=hire.bot_principal_id,
            app_id=hire.app_id,
            credential_ref=hire.credential_ref,
            requested_sequence=50,
            effect_state=lambda _effect: FireEffectState.EXECUTING,
        )
        for hire in hires
    )
    starts: list[str] = []
    stops: list[str] = []
    events: list[str] = []

    def start_delivery_only(agent_id: str, *_args: object, **_kwargs: object) -> object:
        starts.append(agent_id)
        hire = next(item for item in hires if item.agent_id == agent_id)
        if agent_id == "agt_alpha":
            if first_failure == "launch":
                raise ChannelLaunchUnavailable("injected launch failure")
            return _status(hire, ChannelProcessState.FAILED)
        return _status(hire, ChannelProcessState.READY)

    runtime = EmployeeDepartmentRuntime(runtime_enabled=True)
    runtime._service = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        current_employee_transport_snapshot=lambda: (
            employees,
            principals,
            hires,
        )
    )
    runtime._channels = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        status=lambda _agent_id: None,
        start_delivery_only=start_delivery_only,
        stop=lambda agent_id: stops.append(agent_id),
    )
    runtime._fire = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        list_states=lambda: fire_states,
    )
    runtime._outbox = SimpleNamespace(  # type: ignore[assignment]  # noqa: SLF001
        list_pending_delivery_records=lambda: tuple(
            SimpleNamespace(tenant_key=hire.tenant_key, agent_id=hire.agent_id)
            for hire in hires
        ),
    )
    runtime._retirement_outbox_is_cutoff_owned = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda *_args, **_kwargs: True
    )
    runtime._reconcile_terminal_ingress = lambda: 0  # type: ignore[method-assign]  # noqa: SLF001
    runtime._drain_employee_outbox_once = (  # type: ignore[method-assign]  # noqa: SLF001
        lambda: events.append("healthy_outbox") or True
    )
    runtime._employee_dispatch_next_gc_at = float("inf")  # noqa: SLF001

    assert runtime._drain_employee_reporting_once() is True  # noqa: SLF001

    assert starts == ["agt_alpha", "agt_beta"]
    assert stops == (["agt_alpha"] if first_failure == "readiness" else [])
    assert events == ["healthy_outbox"]
