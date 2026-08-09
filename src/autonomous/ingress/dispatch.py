"""Compatibility facade for the employee execution gateway.

New code belongs under :mod:`src.autonomous.gateway`; this import path remains
stable for the Phase 3 integration boundary.
"""

from ..gateway import (
    RENDER_CONTRACT_DIGEST,
    AgentExecutionSpec,
    DispatchBinding,
    DispatchPermit,
    DispatchPermitAuthorityError,
    DispatchPermitConsumedError,
    EmployeeActionRequiredError,
    EmployeeCancellationOutcome,
    EmployeeDispatchCoordinator,
    EmployeeDispatchError,
    EmployeeTeamGateway,
    FinalizedEmployeeAttempt,
    GatewayExecutionResult,
    GatewayExecutionStatus,
    PreparedEmployeeDispatch,
    RenderedEmployeePrompt,
    render_employee_context,
)
from ..gateway.env_scope import (
    EmployeeEnvironmentAuthority,
    EmployeeProcessEnvironmentMaterial,
)

__all__ = [
    "AgentExecutionSpec",
    "EmployeeDispatchCoordinator",
    "EmployeeCancellationOutcome",
    "EmployeeDispatchError",
    "FinalizedEmployeeAttempt",
    "PreparedEmployeeDispatch",
    "RENDER_CONTRACT_DIGEST",
    "RenderedEmployeePrompt",
    "DispatchBinding",
    "DispatchPermit",
    "DispatchPermitConsumedError",
    "GatewayExecutionResult",
    "GatewayExecutionStatus",
    "DispatchPermitAuthorityError",
    "EmployeeActionRequiredError",
    "EmployeeTeamGateway",
    "EmployeeEnvironmentAuthority",
    "EmployeeProcessEnvironmentMaterial",
    "render_employee_context",
]
