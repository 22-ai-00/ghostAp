"""Read-only, tenant-aware employee registry projected from the Journal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.acp.employee_selection import compose_employee_model_selection
from src.autonomous.workforce.identity import AgentIdentity, default_employee_storage_base

from ..domain import BotPrincipal, EmployeeDefinition, EmployeeState, WorkerType
from .projection import workforce_projection_guard

if TYPE_CHECKING:
    from ..journal.projections import ProjectionState


class AmbiguousEmployeeName(LookupError):
    """More than one projected employee matched a scoped name lookup."""


class ProjectedBindingError(RuntimeError):
    """Projected employee/principal authority is internally inconsistent."""


class ProjectedCredentialError(ProjectedBindingError):
    """The projected employee credential is not usable."""


@dataclass(frozen=True, slots=True)
class ProjectedContextBinding:
    employee: EmployeeDefinition
    principal: BotPrincipal
    projection_sequence: int
    projection_hash: str


class ProjectedAgentRegistry:
    """Expose immutable projected identities without legacy write methods."""

    def __init__(
        self,
        state: ProjectionState,
        *,
        storage_base_path: str = "",
    ) -> None:
        self._state = state
        self._storage_base = Path(
            storage_base_path or default_employee_storage_base()
        )

    def get(self, tenant_key: str, agent_id: str) -> EmployeeDefinition | None:
        self._require_tenant(tenant_key)
        employee = self._state.employees.get(agent_id)
        if (
            employee is None
            or employee.tenant_key != tenant_key
            or employee.state is EmployeeState.ARCHIVED
        ):
            return None
        return employee

    def find_by_name(
        self,
        tenant_key: str,
        name: str,
        channel_id: str | None = None,
    ) -> EmployeeDefinition | None:
        self._require_tenant(tenant_key)
        normalized = name.casefold()
        matches = [
            employee
            for employee in self._state.employees.values()
            if employee.tenant_key == tenant_key
            and employee.state is not EmployeeState.ARCHIVED
            and employee.name.casefold() == normalized
            and (
                channel_id is None
                or channel_id in employee.member_groups
            )
        ]
        if len(matches) > 1:
            raise AmbiguousEmployeeName(
                f"ambiguous employee name in tenant {tenant_key}: {name}"
            )
        return matches[0] if matches else None

    def list_agents(
        self,
        tenant_key: str,
        channel_id: str | None = None,
    ) -> list[EmployeeDefinition]:
        self._require_tenant(tenant_key)
        return [
            employee
            for employee in self._state.employees.values()
            if employee.tenant_key == tenant_key
            and employee.state is not EmployeeState.ARCHIVED
            and (
                channel_id is None
                or channel_id in employee.member_groups
            )
        ]

    def as_execution_identity(
        self,
        tenant_key: str,
        agent_id: str,
    ) -> AgentIdentity | None:
        self._require_tenant(tenant_key)
        employee = self.get(tenant_key, agent_id)
        if employee is None:
            return None
        agent_dir = self._storage_base / "agents" / employee.agent_id
        groups = list(employee.member_groups)
        return AgentIdentity(
            agent_id=employee.agent_id,
            name=employee.name,
            emoji=employee.emoji,
            agent_type=employee.tool,
            model_name=compose_employee_model_selection(
                employee.tool,
                employee.model,
                employee.profile,
                employee.effort,
            ),
            model_profile=employee.profile,
            reasoning_effort=employee.effort,
            system_prompt=employee.persona,
            role=employee.role or "custom",
            permissions=list(employee.permissions),
            memory_path=str(agent_dir / "memory" / "MEMORY.md"),
            notes_path=str(agent_dir / "NOTES.md"),
            workspace_path=str(agent_dir / "workspace"),
            owner_group=groups[0] if groups else "",
            member_groups=groups,
            created_at=employee.created_at,
            personality_traits=list(employee.personality_traits),
            capabilities=list(employee.capabilities),
            security_profile="employee_v1",
        )

    def context_binding(
        self,
        *,
        tenant_key: str,
        agent_id: str,
        bot_principal_id: str,
        app_id: str,
        chat_id: str,
    ) -> ProjectedContextBinding | None:
        """Atomically resolve one ACTIVE visible employee execution binding."""
        self._require_tenant(tenant_key)
        with workforce_projection_guard():
            employee = self._state.employees.get(agent_id)
            if (
                employee is None
                or employee.tenant_key != tenant_key
                or employee.state is not EmployeeState.ACTIVE
                or employee.worker_type is not WorkerType.VISIBLE
                or chat_id not in employee.member_groups
            ):
                return None
            if employee.bot_principal_id != bot_principal_id:
                return None
            principal = self._state.bot_principals.get(bot_principal_id)
            if principal is None:
                raise ProjectedBindingError("missing projected bot principal")
            if (
                principal.tenant_key != tenant_key
                or principal.agent_id != agent_id
            ):
                raise ProjectedBindingError(
                    "projected bot principal binding mismatch"
                )
            if principal.app_id != app_id:
                return None
            if not principal.credential_ref:
                raise ProjectedCredentialError(
                    "projected credential is unavailable"
                )
            return ProjectedContextBinding(
                employee=employee,
                principal=principal,
                projection_sequence=getattr(self._state, "cursor_sequence", 0),
                projection_hash=getattr(self._state, "cursor_hash", ""),
            )

    @staticmethod
    def _require_tenant(tenant_key: str) -> None:
        if not tenant_key:
            raise ValueError("tenant_key is required")
