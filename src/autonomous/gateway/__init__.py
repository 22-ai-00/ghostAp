"""Durable visible-employee execution gateway."""

from .context_prompt import (
    RENDER_CONTRACT_DIGEST,
    RenderedEmployeePrompt,
    render_employee_context,
)
from .coordinator import (
    EmployeeCancellationOutcome,
    EmployeeDispatchCoordinator,
    EmployeeDispatchError,
    FinalizedEmployeeAttempt,
    PreparedEmployeeDispatch,
)
from .models import (
    AgentExecutionSpec,
    DispatchBinding,
    DispatchPermit,
    DispatchPermitConsumedError,
    GatewayExecutionResult,
    GatewayExecutionStatus,
)
from .projection import (
    GatewayProjectionError,
    GatewayProjectionState,
    reduce_gateway_frame,
)
from .team import (
    DispatchPermitAuthorityError,
    EmployeeActionRequiredError,
    EmployeeTeamGateway,
)

__all__ = [
    "AgentExecutionSpec",
    "RENDER_CONTRACT_DIGEST",
    "RenderedEmployeePrompt",
    "DispatchBinding",
    "DispatchPermit",
    "DispatchPermitConsumedError",
    "EmployeeCancellationOutcome",
    "EmployeeDispatchCoordinator",
    "EmployeeDispatchError",
    "GatewayExecutionResult",
    "GatewayExecutionStatus",
    "GatewayProjectionError",
    "GatewayProjectionState",
    "DispatchPermitAuthorityError",
    "EmployeeActionRequiredError",
    "EmployeeTeamGateway",
    "FinalizedEmployeeAttempt",
    "PreparedEmployeeDispatch",
    "reduce_gateway_frame",
    "render_employee_context",
]
