"""Actor-only execution gateway for persistent employees."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass

from src.acp.employee_selection import compose_employee_model_selection

from ..runtime.employee_actor import EmployeeAssignment, EmployeeAssignmentTerminal
from ..runtime.employee_session import EmployeeSessionBootstrap
from ..runtime.employee_supervisor import EmployeeRuntimeSupervisor
from ..workforce.identity import AgentIdentity
from .models import (
    AgentExecutionSpec,
    DispatchBinding,
    DispatchPermit,
    GatewayExecutionResult,
    GatewayExecutionStatus,
)


class DispatchPermitAuthorityError(RuntimeError):
    """A permit was not issued by this live gateway or no longer has authority."""


@dataclass(frozen=True, slots=True)
class _IssuedPermit:
    permit: DispatchPermit
    agent: AgentExecutionSpec


class EmployeeTeamGateway:
    """Mint capabilities and execute them only through the persistent Actor."""

    def __init__(
        self,
        *,
        runtime_supervisor: EmployeeRuntimeSupervisor | None,
    ) -> None:
        if runtime_supervisor is None:
            raise ValueError("employee Actor runtime supervisor is required")
        self._runtime = runtime_supervisor
        self._lock = threading.Lock()  # leaf lock: never held while acquiring a LockLevel lock
        self._issued: dict[str, _IssuedPermit] = {}
        self._running: dict[str, _IssuedPermit] = {}
        self._pre_canceled: set[str] = set()
        self._canceled: dict[str, DispatchPermit] = {}
        self._cancel_requested: set[str] = set()

    def issue_permit(
        self,
        *,
        binding: DispatchBinding,
        prompt: str,
        engine: object,
        agent: AgentIdentity,
        timeout_seconds: float,
        env: dict[str, str],
    ) -> DispatchPermit:
        self._validate_agent_binding(binding, agent)
        if (
            not binding.prompt_digest
            or hashlib.sha256(prompt.encode()).hexdigest()
            != binding.prompt_digest
        ):
            raise DispatchPermitAuthorityError("permit prompt binding mismatch")
        frozen_agent = AgentExecutionSpec.from_agent(agent)
        permit = DispatchPermit(
            binding=binding,
            prompt=prompt,
            engine=engine,
            agent=frozen_agent,
            timeout_seconds=timeout_seconds,
            env=env,
        )
        with self._lock:
            if (
                binding.permit_id in self._issued
                or binding.permit_id in self._running
                or binding.permit_id in self._canceled
            ):
                raise DispatchPermitAuthorityError("permit identity already issued")
            if binding.permit_id in self._pre_canceled:
                self._pre_canceled.remove(binding.permit_id)
                self._canceled[binding.permit_id] = permit
                self._cancel_requested.add(binding.permit_id)
            else:
                self._issued[binding.permit_id] = _IssuedPermit(permit, frozen_agent)
        return permit

    def cancel_attempt(self, binding: DispatchBinding) -> bool:
        """Cancel an issued/running permit or latch cancellation before issue."""

        if not isinstance(binding, DispatchBinding):
            raise TypeError("binding must be DispatchBinding")
        with self._lock:
            permit_id = binding.permit_id
            issued = self._issued.pop(permit_id, None)
            running = self._running.get(permit_id)
            self._cancel_requested.add(permit_id)
            if issued is not None:
                self._canceled[permit_id] = issued.permit
            elif running is None:
                self._pre_canceled.add(permit_id)
        if running is not None:
            self._runtime.cancel(binding.attempt_id)
        return running is not None or issued is not None

    def execute_permit(self, permit: DispatchPermit) -> GatewayExecutionResult:
        if not isinstance(permit, DispatchPermit):
            raise TypeError("permit must be DispatchPermit")
        with self._lock:
            canceled = self._canceled.get(permit.binding.permit_id)
            if canceled is permit:
                del self._canceled[permit.binding.permit_id]
                self._cancel_requested.discard(permit.binding.permit_id)
                permit.claim()
                return GatewayExecutionResult(
                    status=GatewayExecutionStatus.CANCELED,
                    safe_error_code="employee_session_canceled",
                )
            issued = self._issued.get(permit.binding.permit_id)
            if issued is None or issued.permit is not permit:
                raise DispatchPermitAuthorityError("permit was not issued by gateway")
            del self._issued[permit.binding.permit_id]
            self._running[permit.binding.permit_id] = issued
        try:
            permit.claim()
            agent = issued.agent.materialize()
            self._validate_agent_binding(permit.binding, agent)
        except BaseException:
            with self._lock:
                self._running.pop(permit.binding.permit_id, None)
                self._cancel_requested.discard(permit.binding.permit_id)
            raise

        try:
            opener = getattr(permit.engine, "open_employee_session", None)
            if not callable(opener):
                raise DispatchPermitAuthorityError(
                    "permit engine lacks reusable employee sessions"
                )
            bootstrap = EmployeeSessionBootstrap.from_agent(
                tenant_key=permit.binding.tenant_key,
                agent=agent,
                project_root=agent.workspace_path,
                identity_version=permit.binding.employee_version,
            )
            assignment = EmployeeAssignment(
                assignment_id=permit.binding.attempt_id,
                bootstrap=bootstrap,
                prompt=permit.prompt,
                timeout_seconds=permit.timeout_seconds,
                payload_ref=permit.binding.envelope_id,
                session_factory=lambda _bootstrap: opener(
                    agent,
                    env=dict(permit.env),
                ),
            )
            self._runtime.submit(assignment)
            terminal = self._runtime.wait_terminal(
                assignment.assignment_id,
                timeout=permit.timeout_seconds + 5.0,
            )
            return self._complete_actor_execution(permit, terminal)
        except TimeoutError:
            result = GatewayExecutionResult(
                status=GatewayExecutionStatus.TIMEOUT,
                safe_error_code="employee_session_timeout",
            )
        except CancelledError:
            result = GatewayExecutionResult(
                status=GatewayExecutionStatus.CANCELED,
                safe_error_code="employee_session_canceled",
            )
        except Exception:
            result = GatewayExecutionResult(
                status=GatewayExecutionStatus.FAILED,
                safe_error_code="employee_session_failed",
            )
        with self._lock:
            self._running.pop(permit.binding.permit_id, None)
            canceled_after_start = permit.binding.permit_id in self._cancel_requested
            self._cancel_requested.discard(permit.binding.permit_id)
        if canceled_after_start:
            return GatewayExecutionResult(
                status=GatewayExecutionStatus.CANCELED,
                safe_error_code="employee_session_canceled",
            )
        return result

    def close(self) -> None:
        self._runtime.close()

    def _complete_actor_execution(
        self,
        permit: DispatchPermit,
        terminal: EmployeeAssignmentTerminal,
    ) -> GatewayExecutionResult:
        status = {
            "completed": GatewayExecutionStatus.COMPLETED,
            "timeout": GatewayExecutionStatus.TIMEOUT,
            "canceled": GatewayExecutionStatus.CANCELED,
            "action_required": GatewayExecutionStatus.ACTION_REQUIRED,
        }.get(terminal.status, GatewayExecutionStatus.FAILED)
        with self._lock:
            self._running.pop(permit.binding.permit_id, None)
            canceled = permit.binding.permit_id in self._cancel_requested
            self._cancel_requested.discard(permit.binding.permit_id)
        if canceled:
            status = GatewayExecutionStatus.CANCELED
        return GatewayExecutionResult(
            status=status,
            output=terminal.output if status is GatewayExecutionStatus.COMPLETED else "",
            safe_error_code=(
                ""
                if status is GatewayExecutionStatus.COMPLETED
                else terminal.error_code or f"employee_actor_{status.value}"
            ),
        )

    @staticmethod
    def _validate_agent_binding(
        binding: DispatchBinding,
        agent: AgentIdentity,
    ) -> None:
        coordinates = (
            agent.agent_id == binding.agent_id,
            agent.agent_type == binding.tool,
            agent.model_name
            == compose_employee_model_selection(
                binding.tool,
                binding.model,
                binding.profile,
                binding.effort,
            ),
            agent.model_profile == binding.profile,
            agent.reasoning_effort == binding.effort,
            tuple(sorted(agent.permissions)) == binding.permissions,
            tuple(sorted(agent.capabilities)) == binding.capabilities,
            agent.security_profile == binding.security_profile == "employee_v1",
        )
        if not all(coordinates):
            raise DispatchPermitAuthorityError("permit agent binding mismatch")


__all__ = ["DispatchPermitAuthorityError", "EmployeeTeamGateway"]
