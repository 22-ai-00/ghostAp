from __future__ import annotations

import pytest

from src.autonomous.authorization import EmployeeAuthorizationScope
from src.autonomous.journal.frame import JournalEvent
from src.autonomous.workforce.registry import (
    ProjectedAgentRegistry,
    ProjectedCredentialError,
)
from tests.autonomous.workforce_helpers import commit_events, seed_workforce_state


def test_projected_registry_is_global_and_returns_fresh_employee_execution_identity(tmp_path) -> None:
    _, state = seed_workforce_state(tmp_path)
    storage_base = tmp_path / "employee-store"
    registry = ProjectedAgentRegistry(state, storage_base_path=str(storage_base))

    employee = registry.get("tenant_1", "agt_1")
    first = registry.as_execution_identity("tenant_1", "agt_1")
    second = registry.as_execution_identity("tenant_1", "agt_1")

    assert employee is not None
    assert first is not None
    assert second is not None
    assert first is not second
    assert first.agent_id == employee.agent_id
    assert first.name == employee.name
    assert first.agent_type == employee.tool
    assert first.model_name == employee.model
    assert first.system_prompt == employee.persona
    assert first.memory_path == str(
        storage_base / "agents" / employee.agent_id / "memory" / "MEMORY.md"
    )
    assert first.workspace_path == str(
        storage_base / "agents" / employee.agent_id / "workspace"
    )
    assert not hasattr(first, "app_id")
    assert not hasattr(first, "credential_ref")


def test_membership_filter_does_not_change_global_identity(tmp_path) -> None:
    writer, state = seed_workforce_state(tmp_path)
    registry = ProjectedAgentRegistry(state)

    assert [
        employee.agent_id
        for employee in registry.list_agents("tenant_1", "oc_team")
    ] == ["agt_1"]
    assert registry.list_agents("tenant_1", "oc_other") == []
    assert registry.get("tenant_1", "agt_1") is not None

    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.membership_changed",
            aggregate_id="agt_1",
            payload={"member_groups": ["oc_other"]},
        ),
    )
    assert registry.get("tenant_1", "agt_1") is not None
    assert [
        employee.agent_id
        for employee in registry.list_agents("tenant_1", "oc_other")
    ] == ["agt_1"]


def test_projected_registry_is_tenant_scoped_and_read_only(tmp_path) -> None:
    _, state = seed_workforce_state(tmp_path)
    registry = ProjectedAgentRegistry(state)

    assert registry.get("tenant_2", "agt_1") is None
    assert registry.find_by_name("tenant_2", "Atlas") is None
    assert not hasattr(registry, "register")
    assert not hasattr(registry, "remove")


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("get", ("agt_1",)),
        ("find_by_name", ("Atlas",)),
        ("list_agents", ()),
        ("as_execution_identity", ("agt_1",)),
    ],
)
def test_projected_registry_rejects_empty_tenant(
    tmp_path,
    method: str,
    args: tuple[str, ...],
) -> None:
    _, state = seed_workforce_state(tmp_path)
    registry = ProjectedAgentRegistry(state)

    with pytest.raises(ValueError, match="tenant_key"):
        getattr(registry, method)("", *args)


def test_archived_employee_is_hidden_but_tombstone_is_retained(tmp_path) -> None:
    writer, state = seed_workforce_state(tmp_path)
    registry = ProjectedAgentRegistry(state)
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "archived"},
        ),
    )

    assert "agt_1" in state.employees
    assert registry.get("tenant_1", "agt_1") is None
    assert registry.find_by_name("tenant_1", "Atlas") is None
    assert registry.list_agents("tenant_1") == []
    assert registry.as_execution_identity("tenant_1", "agt_1") is None


def test_context_binding_requires_active_visible_membership_and_exact_principal(
    tmp_path,
) -> None:
    writer, state = seed_workforce_state(tmp_path)
    registry = ProjectedAgentRegistry(state)
    arguments = {
        "tenant_key": "tenant_1",
        "agent_id": "agt_1",
        "bot_principal_id": "bot_1",
        "app_id": "cli_1",
        "chat_id": "oc_team",
        "requester_principal_id": "ou_requester",
        "authorization_scope": EmployeeAuthorizationScope.MANAGED_GROUP,
    }

    assert registry.context_binding(**arguments) is None
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "active"},
        ),
    )

    binding = registry.context_binding(**arguments)
    assert binding is not None
    assert binding.employee.agent_id == "agt_1"
    assert binding.principal.bot_principal_id == "bot_1"
    assert binding.projection_sequence == state.cursor_sequence
    assert binding.projection_hash == state.cursor_hash
    for changes in (
        {"tenant_key": "tenant_2"},
        {"bot_principal_id": "bot_other"},
        {"app_id": "cli_other"},
        {"chat_id": "oc_other"},
    ):
        assert registry.context_binding(**{**arguments, **changes}) is None


def test_context_binding_rejects_destroyed_credential(tmp_path) -> None:
    writer, state = seed_workforce_state(tmp_path)
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "active"},
        ),
    )
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="credential.destroyed",
            aggregate_id="bot_1",
            payload={"credential_ref": "cred_1"},
        ),
    )

    with pytest.raises(ProjectedCredentialError):
        ProjectedAgentRegistry(state).context_binding(
            tenant_key="tenant_1",
            agent_id="agt_1",
            bot_principal_id="bot_1",
            app_id="cli_1",
            chat_id="oc_team",
            requester_principal_id="ou_requester",
            authorization_scope=EmployeeAuthorizationScope.MANAGED_GROUP,
        )


def test_owner_p2p_binding_requires_projected_owner_not_group_membership(
    tmp_path,
) -> None:
    writer, state = seed_workforce_state(tmp_path)
    commit_events(
        writer,
        state,
        JournalEvent(
            event_type="employee.state_changed",
            aggregate_id="agt_1",
            payload={"state": "active"},
        ),
    )
    registry = ProjectedAgentRegistry(state)

    binding = registry.context_binding(
        tenant_key="tenant_1",
        agent_id="agt_1",
        bot_principal_id="bot_1",
        app_id="cli_1",
        chat_id="oc_owner_p2p",
        requester_principal_id="ou_admin",
        authorization_scope=EmployeeAuthorizationScope.OWNER_P2P,
    )

    assert binding is not None
    assert binding.authorization_scope is EmployeeAuthorizationScope.OWNER_P2P
    assert registry.context_binding(
        tenant_key="tenant_1",
        agent_id="agt_1",
        bot_principal_id="bot_1",
        app_id="cli_1",
        chat_id="oc_owner_p2p",
        requester_principal_id="ou_not_owner",
        authorization_scope=EmployeeAuthorizationScope.OWNER_P2P,
    ) is None
