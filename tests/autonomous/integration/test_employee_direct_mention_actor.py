from __future__ import annotations

from types import SimpleNamespace

from src.autonomous.ingress import (
    GroupRouteKind,
    GroupRouteRequest,
    decide_group_route,
)
from src.autonomous.team import EmployeeTeamService, TeamTarget
from tests.autonomous.workforce_helpers import make_writer


class _DirectBackend:
    def __init__(self) -> None:
        self.direct = []

    def submit_direct(self, **kwargs):
        self.direct.append(kwargs)
        return "acc_direct"


def test_direct_mention_wakes_only_named_employee_mailbox(tmp_path) -> None:
    decision = decide_group_route(
        GroupRouteRequest(
            tenant_key="tenant_1",
            chat_id="oc_team",
            sender_principal_id="ou_user",
            sender_type="user",
            sender_tenant_key="tenant_1",
            text="@agt_alpha 修复测试",
            mentioned_agent_ids=("agt_alpha",),
        )
    )
    assert decision.kind is GroupRouteKind.DIRECT_EMPLOYEE
    assert decision.target_agent_id == "agt_alpha"
    assert decision.wake_model is True

    writer = make_writer(tmp_path)
    backend = _DirectBackend()
    service = EmployeeTeamService(
        writer=writer, backend=backend, runtime_mode="legacy_pipeline"
    )
    acceptance_id = service.dispatch_direct(
        target=TeamTarget("agt_alpha", "Alpha", "coder", ("python",)),
        tenant_key="tenant_1",
        chat_id="oc_team",
        message_id="om_direct",
        requester_principal_id="ou_user",
        instruction="@Alpha 修复测试",
    )
    assert acceptance_id == "acc_direct"
    assert [item["target"].agent_id for item in backend.direct] == ["agt_alpha"]
    assert not any(
        event.event_type.startswith("team.run")
        for frame in writer.replay()
        for event in frame.events
    )
    service.close()
    writer.close()


def test_multiple_mentions_become_team_task_not_direct_fanout() -> None:
    decision = decide_group_route(
        GroupRouteRequest(
            tenant_key="tenant_1",
            chat_id="oc_team",
            sender_principal_id="ou_user",
            sender_type="user",
            sender_tenant_key="tenant_1",
            text="@A @B 一起评审",
            mentioned_agent_ids=("agt_a", "agt_b"),
        )
    )
    assert decision.kind is GroupRouteKind.TEAM_TASK


def test_runtime_direct_dispatch_creates_one_normal_employee_ingress(tmp_path) -> None:
    from src.autonomous.provisioning.composition import _RuntimeTeamBackend
    from src.autonomous.provisioning.hire_state import HirePhase
    from tests.autonomous.integration.test_employee_team_gateway import (
        _real_coordinator_harness,
    )

    harness = _real_coordinator_harness(tmp_path)
    employee = harness.workforce.employees["agt_alpha"]
    hire_state = SimpleNamespace(
        agent_id=employee.agent_id,
        tenant_key=employee.tenant_key,
        phase=HirePhase.ACTIVE,
        channel_generation=3,
        bot_principal_id="bot_alpha",
        app_id="cli_alpha",
    )

    class _HireService:
        def list_states(self):
            return (hire_state,)

    runtime = SimpleNamespace(
        _ingress=harness.ingress,
        _channels=harness.channels,
        _require_service=lambda: _HireService(),
    )
    backend = _RuntimeTeamBackend(runtime, lambda *_args: None)
    acceptance_id = backend.submit_direct(
        target=TeamTarget("agt_alpha", "Alpha", "coder"),
        tenant_key="tenant_1",
        chat_id="oc_team",
        message_id="om_direct_runtime",
        requester_principal_id="ou_requester",
        instruction="direct task",
    )
    record = harness.ingress.state.by_acceptance_id[acceptance_id]
    assert record.metadata.agent_id == "agt_alpha"
    assert record.metadata.event_type == "im.message.receive_v1"
    assert record.metadata.action_identity.startswith("direct:")
    harness.close()
