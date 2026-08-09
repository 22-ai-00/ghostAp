"""One-shot execution gateway for persistent employees."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Mapping
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


class EmployeeActionRequiredError(RuntimeError):
    """Execution stopped safely and requires an explicit human decision."""


@dataclass(frozen=True, slots=True)
class _IssuedPermit:
    permit: DispatchPermit
    agent: AgentExecutionSpec


class EmployeeTeamGateway:
    """Mint and consume process-local capabilities for already-anchored attempts."""

    def __init__(
        self,
        *,
        runtime_mode: str,
        runtime_supervisor: EmployeeRuntimeSupervisor | None = None,
        shadow_observer: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        if runtime_mode not in {"legacy_one_shot", "shadow", "actor"}:
            raise ValueError("invalid employee runtime mode")
        if runtime_mode == "actor" and runtime_supervisor is None:
            raise ValueError("actor mode requires employee runtime supervisor")
        self._runtime_mode = runtime_mode
        self._runtime = runtime_supervisor
        self._shadow_observer = shadow_observer
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
        target = running or issued
        if target is None:
            return False
        if running is not None:
            if self._runtime_mode == "actor" and self._runtime is not None:
                self._runtime.cancel(binding.attempt_id)
            else:
                cancel = getattr(running.permit.engine, "cancel_employee_session", None)
                if not callable(cancel):
                    cancel = getattr(running.permit.engine, "stop_agent", None)
                if callable(cancel):
                    try:
                        cancel(binding.agent_id)
                    except Exception:
                        pass
        return True

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
            runner = getattr(permit.engine, "run_agent_session", None)
            if self._runtime_mode != "actor" and not callable(runner):
                raise DispatchPermitAuthorityError("permit engine is invalid")
            if self._runtime_mode == "shadow":
                self._observe_shadow_input(permit, agent)
        except BaseException:
            with self._lock:
                self._running.pop(permit.binding.permit_id, None)
                self._cancel_requested.discard(permit.binding.permit_id)
            raise
        result: GatewayExecutionResult
        try:
            if self._runtime_mode == "actor":
                assert self._runtime is not None
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
            output = runner(  # type: ignore[misc]
                agent,
                permit.prompt,
                timeout=permit.timeout_seconds,
                env=dict(permit.env),
            )
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
        except EmployeeActionRequiredError:
            result = GatewayExecutionResult(
                status=GatewayExecutionStatus.ACTION_REQUIRED,
                safe_error_code="employee_session_action_required",
            )
        except Exception:
            result = GatewayExecutionResult(
                status=GatewayExecutionStatus.FAILED,
                safe_error_code="employee_session_failed",
            )
        else:
            if not isinstance(output, str) or not output:
                result = GatewayExecutionResult(
                    status=GatewayExecutionStatus.FAILED,
                    safe_error_code="employee_session_failed",
                )
            else:
                result = GatewayExecutionResult(
                    status=GatewayExecutionStatus.COMPLETED,
                    output=output,
                )
        with self._lock:
            self._running.pop(permit.binding.permit_id, None)
            canceled_after_start = permit.binding.permit_id in self._cancel_requested
            self._cancel_requested.discard(permit.binding.permit_id)
        if canceled_after_start:
            result = GatewayExecutionResult(
                status=GatewayExecutionStatus.CANCELED,
                safe_error_code="employee_session_canceled",
            )
        if self._runtime_mode == "shadow":
            self._observe_shadow_terminal(permit, result)
        return result

    def _observe_shadow_input(
        self,
        permit: DispatchPermit,
        agent: AgentIdentity,
    ) -> None:
        observer = self._shadow_observer
        if observer is None:
            return
        try:
            bootstrap = EmployeeSessionBootstrap.from_agent(
                tenant_key=permit.binding.tenant_key,
                agent=agent,
                project_root=agent.workspace_path,
                identity_version=permit.binding.employee_version,
            )
            actor_input = bootstrap.wrap_prompt(permit.prompt)
            preview_legacy = getattr(
                permit.engine,
                "preview_employee_session_prompt",
                None,
            )
            if not callable(preview_legacy):
                raise RuntimeError("legacy employee prompt preview is unavailable")
            legacy_input = preview_legacy(agent, permit.prompt)
            if not isinstance(legacy_input, str) or not legacy_input:
                raise RuntimeError("legacy employee prompt preview is invalid")
            observer(
                {
                    "stage": "input",
                    "attempt_id": permit.binding.attempt_id,
                    "agent_id": permit.binding.agent_id,
                    "context_snapshot_hash": permit.binding.context_snapshot_hash,
                    "instruction_digest": bootstrap.instruction_digest,
                    "legacy_input_digest": hashlib.sha256(legacy_input.encode()).hexdigest(),
                    "actor_input_digest": hashlib.sha256(actor_input.encode()).hexdigest(),
                    "input_match": legacy_input == actor_input,
                }
            )
        except Exception:
            try:
                observer(
                    {
                        "stage": "input",
                        "attempt_id": permit.binding.attempt_id,
                        "agent_id": permit.binding.agent_id,
                        "context_snapshot_hash": permit.binding.context_snapshot_hash,
                        "instruction_digest": "",
                        "legacy_input_digest": "",
                        "actor_input_digest": "",
                        "input_match": False,
                        "mismatch_code": "shadow_bootstrap_unavailable",
                    }
                )
            except Exception:
                pass

    def _observe_shadow_terminal(
        self,
        permit: DispatchPermit,
        result: GatewayExecutionResult,
    ) -> None:
        observer = self._shadow_observer
        if observer is None:
            return
        try:
            observer(
                {
                    "stage": "terminal",
                    "attempt_id": permit.binding.attempt_id,
                    "agent_id": permit.binding.agent_id,
                    "status": result.status.value,
                    "output_digest": (
                        hashlib.sha256(result.output.encode()).hexdigest()
                        if result.output
                        else ""
                    ),
                    "safe_error_code": result.safe_error_code,
                }
            )
        except Exception:
            pass

    def close(self) -> None:
        if self._runtime is not None:
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


__all__ = [
    "DispatchPermitAuthorityError",
    "EmployeeActionRequiredError",
    "EmployeeTeamGateway",
]
